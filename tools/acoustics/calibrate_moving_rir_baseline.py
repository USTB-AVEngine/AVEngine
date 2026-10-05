#!/usr/bin/env python3
"""Measure independent native RLR repeatability before choosing motion budgets."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import combinations
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.continuous_scene import position_at
from avengine.acoustics.moving_rir_acceptance import compare_rirs
from avengine.acoustics.moving_energy_policies import summarize_checks
from tools.timeline.pilot_continuous_scenes import native_renderer


def read(path):return json.loads(Path(path).read_text())


def scalar_errors(check):
    result={name:abs(check[name]) for name in ('direct_energy_difference_db','total_energy_difference_db',
        'drr_difference_db','direct_delay_difference_ms','foa_error_deg','geometry_error_deg')}
    result.update({f'octave_{b["center_hz"]}_energy_db':abs(b['energy_difference_db']) for b in check['octave_energy']})
    return result


def distribution(checks):
    measured=[c for c in checks if c['status']=='measured']
    if not measured:return {'measured':0,'invalid':len(checks),'metrics':{}}
    errors=[scalar_errors(c) for c in measured]
    return {'measured':len(measured),'invalid':len(checks)-len(measured),'metrics':{k:{
        'median':float(np.median([c[k] for c in errors])),
        'p95':float(np.percentile([c[k] for c in errors],95)),
        'maximum':float(max(c[k] for c in errors))} for k in errors[0]}}


def render_repeat(task):
    inputs_path,profile,threads,repeat,house,poses,output,permuted_order=task
    inputs=read(inputs_path);requested=deepcopy(inputs)
    requested['simulation'].update(diffraction=profile=='on',max_diffraction_order=10 if profile=='on' else 0)
    root=Path(output)/f'{profile}_threads{threads}'/house['house']/f'repeat{repeat}'
    root.mkdir(parents=True,exist_ok=False)
    start,cpu=time.perf_counter(),time.process_time()
    renderer,_=native_renderer(requested,house,threads=threads)
    actual=renderer.setup_report['configuration_readback']
    assert actual['thread_count']==threads and actual['diffraction']==(profile=='on')
    write_json(root/'native_setup.json',renderer.setup_report)
    costs=[]
    order=np.random.default_rng(20261004060+repeat).permutation(len(poses)) if permuted_order else np.arange(len(poses))
    for index in order:
        pose=poses[int(index)]
        batch=renderer.render([pose['position_m']]);path=root/f'pose_{pose["pose_index"]:03d}.npy'
        with path.open('xb') as f:np.save(f,batch.samples[0],allow_pickle=False)
        costs.append({'pose_index':pose['pose_index'],'CPU_seconds':batch.process_cpu_seconds,'wall_seconds':batch.wall_seconds})
    row={'house':house['house'],'profile':profile,'threads':threads,'repeat':repeat,'native_calls':len(costs),
        'native_CPU_seconds':sum(c['CPU_seconds'] for c in costs),'native_wall_seconds':sum(c['wall_seconds'] for c in costs),
        'stage_wall_seconds':time.perf_counter()-start,'stage_CPU_seconds':time.process_time()-cpu,
        'costs':costs,'independent_native_context':True,'native_random_setting':'existing SDK default; no exposed seed field',
        'temporal_coherence':False,'render_pose_order':order.tolist(),'permuted_order':permuted_order,'root':str(root)}
    write_json(root/'receipt.json',row)
    print(json.dumps({k:row[k] for k in ('house','profile','threads','repeat','native_calls','native_CPU_seconds')}),flush=True)
    return row


def make_plans(inputs,round3):
    plans=[]
    for house in inputs['houses']:
        scene=read(round3/'native_windows_correct_clock'/(house['house']+'__observed_scene.json'))
        track=scene['sources']['voice2']['trajectory']
        fixed=[c['time_s'] for c in read(round3/'native_mouth_energy'/house['house']/'result.json')['checks']]
        rng=np.random.default_rng(2026100406+len(plans));poses=[]
        for _ in range(10000):
            t=float(rng.uniform(.1,29.9));p=position_at(track,t)
            if min(abs(t-x) for x in fixed)<.04:continue
            if any(np.linalg.norm(p-np.asarray(q['position_m']))<.015 for q in poses):continue
            i=len(poses);poses.append({'pose_index':i,'time_s':t,'position_m':p.tolist(),
                'split':'calibration' if i<16 else 'heldout_good'})
            if len(poses)==20:break
        assert len(poses)==20,'need 20 distinct independently selected poses per house'
        plans.append({'house':house,'poses':poses,'listener':scene['listener_m'],'forward':scene['camera_forward']})
    return plans


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--round3',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--permuted-order',action='store_true',help='Vary call order across independent contexts to test pose reproducibility beyond identical replay')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not 1<=args.workers<=8:raise ValueError('CPU only, max8 workers x8 native threads')
    args.output.mkdir(parents=True,exist_ok=False)
    inputs=read(args.inputs);plans=make_plans(inputs,args.round3)
    write_json(args.output/'predeclared_pose_plan.json',{'pose_count':40,'repeats':3,'profiles':['off','on'],
        'native_thread_counts':[8,1],'native_seed':'no exposed field; SDK default unchanged','plans':plans,
        'fixed80_times_excluded':True,'permuted_order':args.permuted_order,'split_calibration_poses':32,'heldout_good_poses':8})
    tasks=[(str(args.inputs),profile,threads,repeat,p['house'],p['poses'],str(args.output),args.permuted_order)
           for profile in ('off','on') for threads in (8,1) for repeat in range(3) for p in plans]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        costs=list(pool.map(render_repeat,tasks))
    reports=[]
    for profile in ('off','on'):
        for threads in (8,1):
            checks=[]
            for p in plans:
                for pose in p['poses']:
                    waves=[np.load(args.output/f'{profile}_threads{threads}'/p['house']['house']/f'repeat{r}'/f'pose_{pose["pose_index"]:03d}.npy',allow_pickle=False) for r in range(3)]
                    for left,right in combinations(range(3),2):
                        c=compare_rirs(waves[left],waves[right],source=pose['position_m'],listener=np.asarray(p['listener']),forward=np.asarray(p['forward']))
                        checks.append({'house':p['house']['house'],'pose_index':pose['pose_index'],'split':pose['split'],
                            'repeats':[left,right],'byte_equal_waveform':bool(np.array_equal(waves[left],waves[right])),**c})
            selected=[c for c in costs if c['profile']==profile and c['threads']==threads]
            row={'profile':profile,'threads':threads,'checks':checks,'distribution':distribution(checks),
                'calibration_distribution':distribution([c for c in checks if c['split']=='calibration']),
                'heldout_good_distribution':distribution([c for c in checks if c['split']=='heldout_good']),
                'old_engineering_qualification':summarize_checks(checks),'byte_equal_comparisons':sum(c['byte_equal_waveform'] for c in checks),
                'native_calls':sum(c['native_calls'] for c in selected),'native_CPU_seconds':sum(c['native_CPU_seconds'] for c in selected),
                'native_wall_sum_seconds':sum(c['native_wall_seconds'] for c in selected)}
            write_json(args.output/f'{profile}_threads{threads}/summary.json',row);reports.append(row)
    write_json(args.output/'summary.json',{'results':reports,'wall_seconds':time.perf_counter()-start,
        'pose_count':40,'independent_repeats':3,'permuted_order':args.permuted_order,'native_calls':sum(c['native_calls'] for c in costs),
        'native_jobs':costs,'criteria_selected':False,'formal_questions':0})


if __name__=='__main__':main()
