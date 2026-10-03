#!/usr/bin/env python3
"""Native LOS, repeatability and phase-cancellation controls at fixed poses."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from avengine.acoustics.moving_energy_policies import summarize_checks
from avengine.acoustics.moving_rir_acceptance import compare_rirs
from avengine.timeline.continuous_scene import position_at,plan_rir_keyframes,interpolated_filter
from avengine.timeline.continuous_pilot import write_json
from tools.timeline.pilot_continuous_scenes import native_renderer
from tools.acoustics.compare_moving_energy_policies import ReadOnlyRIRs

def read(p):return json.loads(Path(p).read_text())
def work(task):
    ip,previous,out,house_id=task;inputs=read(ip);house=next(h for h in inputs['houses'] if h['house']==house_id)
    previous=Path(previous);root=Path(out)/house_id;root.mkdir(parents=True,exist_ok=True)
    start,cpu=time.perf_counter(),time.process_time();renderer,identity=native_renderer(inputs,house)
    write_json(root/'native_setup.json',renderer.setup_report)
    scene=read(previous/'native_windows_correct_clock'/(house_id+'__observed_scene.json'))
    track=scene['sources']['voice2']['trajectory'];listener=np.asarray(scene['listener_m']);forward=np.asarray(scene['camera_forward'])
    reference=read(previous/'native_mouth_energy'/house_id/'result.json')
    cache=ReadOnlyRIRs(previous/'native_delivery_correct_clock'/house_id/'rir_cache')
    keys=plan_rir_keyframes(track,30,listener,max_seconds=.25,max_distance_m=.5)
    filters=[cache.get(k['position_m']) for k in keys];samples=[k['sample'] for k in keys]
    # RLR ray queries are valid only after at least one completed simulation.
    warm=renderer.render([position_at(track,4.1).tolist()])
    rays=[];phase=[]
    for check in reference['checks']:
        t=check['time_s'];position=position_at(track,t);delta=listener-position;distance=float(np.linalg.norm(delta));direction=delta/distance
        hit=renderer.context.trace_ray_first_hit(position.tolist(),direction.tolist(),.001,max(.002,distance-.001))
        rays.append({'time_s':t,'blocked_between_source_and_listener':bool(hit.hit),
            'first_hit_distance_m':float(hit.distance) if hit.hit else None,'source_listener_distance_m':distance})
        i=min(len(keys)-2,max(0,int(np.searchsorted(samples,round(t*16000),side='right'))-1))
        a,b=filters[i][0],filters[i+1][0];n=max(len(a),len(b));aa=np.pad(a,(0,n-len(a)));bb=np.pad(b,(0,n-len(b)))
        u=(round(t*16000)-samples[i])/(samples[i+1]-samples[i]);mixture=(1-u)*aa+u*bb
        # Exact algebraic covariance; no fitted gain or admission adjustment.
        expected=(1-u)**2*np.square(aa).sum()+u*u*np.square(bb).sum()+2*u*(1-u)*(aa@bb)
        phase.append({'time_s':t,'endpoint_arrival_distance_change_samples':
            float((np.linalg.norm(np.asarray(keys[i+1]['position_m'])-listener)-np.linalg.norm(np.asarray(keys[i]['position_m'])-listener))/343*16000),
            'endpoint_W_cosine':float((aa@bb)/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-30)),
            'mix_energy':float(np.square(mixture).sum()),'algebraic_mix_energy':float(expected),
            'max_octave_error_db':max(abs(b['energy_difference_db']) for b in check['octave_energy'])})
    repeat=[]
    for t in (4.1,7.7):
        position=position_at(track,t);calls=[]
        for j in range(3):
            batch=renderer.render([position.tolist()]);wave=batch.samples[0]
            np.save(root/f'repeat_{t:.1f}_{j}.npy',wave,allow_pickle=False);calls.append(wave)
        checks=[compare_rirs(calls[j],calls[0],source=position,listener=listener,forward=forward) for j in (1,2)]
        controls={'same_array':compare_rirs(calls[0],calls[0],source=position,listener=listener,forward=forward),
            'known_bad_6dB_gain':compare_rirs(calls[0]*2,calls[0],source=position,listener=listener,forward=forward)}
        repeat.append({'time_s':t,'native_repeat_qualification':summarize_checks(checks),'checks':checks,
            'controls':{k:summarize_checks([v]) for k,v in controls.items()}})
    report={'house':house_id,'LOS':rays,'blocked_reference_samples':sum(r['blocked_between_source_and_listener'] for r in rays),
        'phase_controls':phase,'repeatability':repeat,'native_repeat_calls':6,'native_warmup_calls':1,
        'native_warmup_CPU_seconds':warm.process_cpu_seconds,
        'wall_seconds':time.perf_counter()-start,'process_CPU_seconds':time.process_time()-cpu,
        'no_admission_threshold_changed':True,'no_STARSS_waveform_used':True}
    write_json(root/'summary.json',report);return report
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--round3',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    inputs=read(a.inputs);a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(work,[(str(a.inputs),str(a.round3),str(a.output),h['house']) for h in inputs['houses']]))
    write_json(a.output/'summary.json',{'results':rows})
if __name__=='__main__':main()
