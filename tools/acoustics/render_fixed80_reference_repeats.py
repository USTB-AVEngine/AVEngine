#!/usr/bin/env python3
"""Three independent native reference measurements at retained fixed80 poses."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
from avengine.timeline.continuous_scene import position_at
from avengine.timeline.continuous_pilot import write_json
from tools.acoustics.calibrate_moving_rir_baseline import render_repeat


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--round3',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not 1<=args.workers<=8:raise ValueError('CPU only, 1..8 workers')
    args.output.mkdir(parents=True,exist_ok=False);inputs=read(args.inputs);plans=[]
    for house in inputs['houses']:
        hid=house['house'];scene=read(args.round3/'native_windows_correct_clock'/(hid+'__observed_scene.json'))
        times=[c['time_s'] for c in read(args.round3/'native_mouth_energy'/hid/'result.json')['checks']]
        poses=[];by_position={};time_indices=[]
        for t in times:
            p=position_at(scene['sources']['voice2']['trajectory'],t).tolist();key=tuple(round(x,12) for x in p)
            if key not in by_position:
                by_position[key]=len(poses);poses.append({'pose_index':len(poses),'position_m':p})
            time_indices.append(by_position[key])
        plans.append({'house':house,'times_s':times,'poses':poses,'time_pose_indices':time_indices,
            'listener':scene['listener_m'],'forward':scene['camera_forward']})
    write_json(args.output/'plan.json',{'plans':plans,'profile':'on','threads':1,'repeats':3,
        'native_metrics_aggregation':'median absolute error over three native references; no waveform averaging',
        'calibration_thresholds_changed':False})
    tasks=[(str(args.inputs),'on',1,rep,p['house'],p['poses'],str(args.output),True) for rep in range(3) for p in plans]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(render_repeat,tasks))
    write_json(args.output/'summary.json',{'native_jobs':results,'native_calls':sum(r['native_calls'] for r in results),
        'native_CPU_seconds':sum(r['native_CPU_seconds'] for r in results),'all_references_independent_of_endpoint_context':True})


if __name__=='__main__':main()
