#!/usr/bin/env python3
"""Offline nearest/arrival-aligned controls; never replace production qualification."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.fft import rfft,irfft,next_fast_len,rfftfreq
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.continuous_scene import plan_rir_keyframes,position_at,interpolated_filter
from avengine.acoustics.moving_rir_acceptance import compare_rirs
from avengine.acoustics.moving_energy_policies import summarize_checks
from tools.acoustics.compare_moving_energy_policies import ReadOnlyRIRs

def read(path):return json.loads(Path(path).read_text())
def shift(wave,samples):
    # Zero-padded fractional delay, no wrap into the impulse onset or gain fit.
    margin=int(np.ceil(abs(samples)))+64;size=next_fast_len(wave.shape[1]+2*margin)
    a=np.pad(wave,((0,0),(margin,size-wave.shape[1]-margin)))
    moved=irfft(rfft(a,axis=1)*np.exp(-2j*np.pi*rfftfreq(size)*samples),n=size,axis=1)
    return moved[:,margin:margin+wave.shape[1]]

def work(task):
    previous,arms,output,house=task;previous=Path(previous);root=Path(arms)/house
    scene=read(previous/'native_windows_correct_clock'/(house+'__observed_scene.json'));track=scene['sources']['voice2']['trajectory'];listener=np.array(scene['listener_m']);forward=np.array(scene['camera_forward'])
    keys=plan_rir_keyframes(track,30,listener,max_seconds=.25,max_distance_m=.5);samples=[k['sample'] for k in keys]
    times=read(root/'fixed_midpoints.json')['times_s'];results=[]
    for label,prod,ref in [('a',previous/'native_delivery_correct_clock'/house/'rir_cache',previous/'native_mouth_energy'/house/'independent_native_cache'),
        ('b',root/'diffraction_on_production_cache',root/'diffraction_on_reference_cache')]:
        production=ReadOnlyRIRs(prod);reference=ReadOnlyRIRs(ref)
        filters=[production.get(k['position_m']) for k in keys];nearest_checks=[];aligned_checks=[];start=time.process_time()
        for t in times:
            sample=round(t*16000);i=min(len(keys)-2,max(0,int(np.searchsorted(samples,sample,side='right'))-1));u=(sample-samples[i])/(samples[i+1]-samples[i]);source=position_at(track,t);true=reference.get(source)
            k=i if u<=.5 else i+1
            nearest_checks.append({'time_s':t,**compare_rirs(filters[k],true,source=source,listener=listener,forward=forward)})
            distance=np.linalg.norm(source-listener);aligned=[]
            for j in (i,i+1):
                offset=(distance-np.linalg.norm(np.array(keys[j]['position_m'])-listener))/343*16000
                aligned.append(shift(filters[j],offset))
            estimate=interpolated_filter(aligned,[samples[i],samples[i+1]],sample)
            aligned_checks.append({'time_s':t,**compare_rirs(estimate,true,source=source,listener=listener,forward=forward)})
        results.append({'diffraction_arm':label,'nearest':summarize_checks(nearest_checks),'arrival_aligned_linear':summarize_checks(aligned_checks),
            'nearest_checks':nearest_checks,'arrival_aligned_linear_checks':aligned_checks,'offline_CPU_seconds':time.process_time()-start,
            'new_native_calls':0,'production_policy_changed':False,'experimental_controls_only':True})
    Path(output).mkdir(parents=True,exist_ok=True);write_json(Path(output)/(house+'.json'),{'house':house,'controls':results});return {'house':house,'controls':results}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--round3',type=Path,required=True);p.add_argument('--arms',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    houses=[x['house'] for x in read(a.arms/'summary.json')['results']]
    with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(work,[(str(a.round3),str(a.arms),str(a.output),h) for h in houses]))
    write_json(a.output/'summary.json',{'results':results,'thresholds_changed':False,'production_policy_changed':False})
if __name__=='__main__':main()
