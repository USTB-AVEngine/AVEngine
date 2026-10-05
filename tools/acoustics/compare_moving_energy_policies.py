#!/usr/bin/env python3
"""Compare four diffraction/refinement arms against fixed native mouth midpoints."""
from __future__ import annotations
import argparse,json,sqlite3,time,os
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from avengine.acoustics.moving_energy_policies import summarize_checks,midpoint_checks,adaptive_keyframes
from avengine.acoustics.moving_rir_acceptance import compare_rirs
from avengine.timeline.continuous_scene import plan_rir_keyframes,position_at
from avengine.timeline.continuous_pilot import PositionRIRCache,write_json
from tools.timeline.pilot_continuous_scenes import native_renderer

def read(path):return json.loads(Path(path).read_text())

class ReadOnlyRIRs:
    def __init__(self,root):
        self.root=Path(root);db=sqlite3.connect(f'file:{self.root}/positions.sqlite3?mode=ro',uri=True)
        self.points=[(i,json.loads(p)) for i,p in db.execute('SELECT id,pose FROM rirs WHERE completed=1')];db.close()
        self.tree=cKDTree([p for _,p in self.points]);self.memo={}
    def get(self,position):
        distance,index=self.tree.query(position)
        if distance>1e-10:raise ValueError('requested fixed native pose missing from readonly cache')
        index=self.points[index][0]
        if index not in self.memo:self.memo[index]=np.load(self.root/f'rir_{index:06d}.npy',allow_pickle=False)
        return self.memo[index]

def work(task):
    inputs_path,round3,output,house_id=task;inputs=read(inputs_path);house=next(h for h in inputs['houses'] if h['house']==house_id)
    previous=Path(round3);root=Path(output)/house_id;root.mkdir(parents=True,exist_ok=True)
    write_json(root/f'worker_{os.getpid()}.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'output_root':str(root),
        'command':'compare_moving_energy_policies.py','compute_device':'CPU'})
    scene=read(previous/'native_windows_correct_clock'/(house_id+'__observed_scene.json'))
    track=scene['sources']['voice2']['trajectory'];listener=np.asarray(scene['listener_m']);forward=np.asarray(scene['camera_forward'])
    reference=read(previous/'native_mouth_energy'/house_id/'result.json')
    times=[c['time_s'] for c in reference['checks']];assert len(times)==40
    write_json(root/'fixed_midpoints.json',{'times_s':times,'positions_m':[position_at(track,t).tolist() for t in times],
        'authority':str(previous/'native_mouth_energy'/house_id/'result.json'),'same_80_points_across_all_arms':True})
    keys=plan_rir_keyframes(track,30,listener,max_seconds=.25,max_distance_m=.5)
    old=ReadOnlyRIRs(previous/'native_delivery_correct_clock'/house_id/'rir_cache')
    oldref=ReadOnlyRIRs(previous/'native_mouth_energy'/house_id/'independent_native_cache')
    filters=[old.get(k['position_m']) for k in keys]
    achecks=midpoint_checks(track,keys,filters,times,oldref.get,listener,forward)
    historical=read(max((previous/'native_delivery_correct_clock').glob('summary_*.json')))
    audio=next(r['audio'] for r in historical['scenes'] if r['scene']['house']==house_id)
    rows=[]
    for label,adaptive in (('a',False),('c',True)):
        path=root/(label+'_result.json')
        if path.exists():rows.append(read(path));continue
        start,cpu=time.perf_counter(),time.process_time()
        selected,refinement=adaptive_keyframes(track,keys,old.get,listener,forward) if adaptive else (keys,None)
        checks=achecks if selected==keys else midpoint_checks(track,selected,[old.get(k['position_m']) for k in selected],times,oldref.get,listener,forward)
        result={'arm':label,'diffraction':False,'max_diffraction_order':0,'adaptive':adaptive,
            'qualification':summarize_checks(checks),'checks':checks,'refinement':refinement,'production_keys':len(selected),
            'production_RIRs_per_scene':audio['RIR_cache']['new_rirs'],'production_native_CPU_seconds':audio['RIR_cache']['native_cpu_seconds'],
            'cost_source':'retained measured round3 audio cache statistics; no new native calls',
            'incremental_native_RIRs':0,'evaluation_wall_seconds':time.perf_counter()-start,'evaluation_CPU_seconds':time.process_time()-cpu}
        write_json(path,result);rows.append(result)
        print('ARM_DONE',house_id,label,result['qualification']['pass_count'],flush=True)
    requested=deepcopy(inputs);requested['simulation'].update(diffraction=True,max_diffraction_order=10)
    write_json(root/'diffraction_on_inputs.json',requested)
    start,cpu=time.perf_counter(),time.process_time();renderer,identity=native_renderer(requested,house)
    write_json(root/'diffraction_on_native_setup.json',renderer.setup_report)
    cache=PositionRIRCache(root/'diffraction_on_production_cache',renderer,identity)
    before=cache.stats.copy()
    for endpoint,spec in scene['sources'].items():
        sourcekeys=plan_rir_keyframes(spec['trajectory'],30,listener,max_seconds=.25,max_distance_m=.5)
        for i,k in enumerate(sourcekeys):
            cache.get(k['position_m'])
            if i%40==0:print('DIFFRACTION_NATIVE_KEY',house_id,endpoint,i,flush=True)
    filters=[cache.get(k['position_m']) for k in keys]
    production_stats=cache.stats.copy();production_cpu=time.process_time()-cpu;production_wall=time.perf_counter()-start
    # A separate native context/cache supplies midpoint references, never a
    # reused diffraction-off waveform or an estimate treated as truth.
    ref_renderer,ref_identity=native_renderer(requested,house)
    ref_cache=PositionRIRCache(root/'diffraction_on_reference_cache',ref_renderer,ref_identity)
    bchecks=midpoint_checks(track,keys,filters,times,ref_cache.get,listener,forward)
    for label,adaptive in (('b',False),('d',True)):
        selected,refinement=adaptive_keyframes(track,keys,cache.get,listener,forward) if adaptive else (keys,None)
        checks=bchecks if selected==keys else midpoint_checks(track,selected,[cache.get(k['position_m']) for k in selected],times,ref_cache.get,listener,forward)
        result={'arm':label,'diffraction':True,'max_diffraction_order':10,'adaptive':adaptive,
            'qualification':summarize_checks(checks),'checks':checks,'refinement':refinement,'production_keys':len(selected),
            'production_RIRs_per_scene':production_stats['new_rirs']+production_stats['cache_hits']*0,
            'production_native_CPU_seconds':production_stats['native_cpu_seconds'],
            'production_stage_CPU_seconds':production_cpu,'production_stage_wall_seconds':production_wall,
            'midpoint_reference_stats':ref_cache.stats.copy(),'cost_source':'fresh native measurement unless completed exact-pose cache resumed',
            'reused_b_for_identical_d':label=='d' and selected==keys,
            'incremental_native_RIRs':0 if label=='d' and selected==keys else production_stats['new_rirs']}
        write_json(root/(label+'_result.json'),result);rows.append(result)
        print('ARM_DONE',house_id,label,result['qualification']['pass_count'],flush=True)
    cache.db.close();ref_cache.db.close()
    write_json(root/'summary.json',{'house':house_id,'arms':rows,'criteria_unchanged':True})
    return {'house':house_id,'arms':rows,'criteria_unchanged':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--round3',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not 1<=a.workers<=8:raise ValueError('CPU-only comparison with 1..8 workers')
    inputs=read(a.inputs);a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(work,[(str(a.inputs),str(a.round3),str(a.output),h['house']) for h in inputs['houses']]))
    write_json(a.output/'summary.json',{'results':results,'fixed_reference_samples':80,'criteria_unchanged':True})

if __name__=='__main__':main()
