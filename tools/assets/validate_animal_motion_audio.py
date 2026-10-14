"""All-raw-independent native RLR qualification of measured moving mouths."""
import argparse,copy,json,math,time,traceback
from pathlib import Path
import numpy as np
from avengine.acoustics.moving_rir_acceptance import compare_rirs,qualify_energy
from avengine.acoustics.moving_rir_calibration import CorrelatedRIRTimeline,effective_emission_filter
from avengine.acoustics.house_acceptance_metrics import foa_direction
from avengine.timeline.continuous_pilot import PositionRIRCache
from avengine.timeline.v7_registered_motion_audio import position_at
from tools.timeline.pilot_continuous_scenes import native_renderer

def main():
 p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ray-multiplier',type=int,default=9);p.add_argument('--step',type=float,default=.2);p.add_argument('--fresh-process',action='store_true');p.add_argument('--frequency-coherence',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);out={'status':'pending','motion_receipt':str(a.motion),'thresholds_changed':False,'independent_reference_count_per_midpoint':3};start=time.process_time()
 motion=json.loads(a.motion.read_text());house=json.loads(a.house.read_text());runtime=json.loads(a.runtime.read_text());frames=motion['frames'];track=[{'time_s':f['time_s'],'position_m':f['mouth_world_m']} for f in frames]
 if motion.get('checks',{}).get('status')!='pass':raise ValueError('native_geometry_not_qualified')
 listener=np.asarray(motion['listener_m']);forward=np.asarray(motion['camera_forward']);forward[1]=0;forward/=np.linalg.norm(forward)
 house['room']['placement']['camera_m']=listener.tolist();house['room']['placement']['source_1_m']=track[0]['position_m']
 for k in ('direct_ray_count','indirect_ray_count','source_ray_count'):runtime['simulation'][k]*=a.ray_multiplier
 runtime['simulation']['thread_count']=1
 renderer,identity=native_renderer(runtime,house)
 if a.fresh_process:
  from avengine.assets.animal_motion.acoustics import FreshProcessRenderer
  setup=renderer.setup_report;del renderer
  import gc
  gc.collect();renderer=FreshProcessRenderer(runtime,house,a.output/'fresh_native',setup);identity['context_policy']='fresh_process_per_RIR_deterministic_native_replay'
 (a.output/'setup.json').write_text(json.dumps(renderer.setup_report,indent=2));cache=PositionRIRCache(a.output/'cache',renderer,identity)
 timeline_type=CorrelatedRIRTimeline;revision=None
 if a.frequency_coherence:
  from avengine.assets.animal_motion.rir_crossfade import FrequencyCoherentTimeline,REVISION
  timeline_type=FrequencyCoherentTimeline;revision=REVISION
 sr=16000;n=round(frames[-1]['time_s']*sr)+1;keys=sorted(set([*range(0,n,round(a.step*sr)),n-1]));directions={};refs={};checks=[]
 def native(sample):
  pos=position_at(track,sample/sr);wave=cache.get(pos);d=foa_direction(wave,listener,pos,forward,sample_rate=sr);angle=d['check']['angle_deg']
  directions[sample]={'time_s':sample/sr,'mouth_m':pos.tolist(),'angle_deg':angle,'native':d}
  if angle is None or angle>5:raise ValueError('native_RIR_direction_exceeds_5deg')
  return wave
 def reference(sample):
  if sample in refs:return refs[sample]
  waves=[];pos=position_at(track,sample/sr)
  for rep in range(3):
   path=a.output/f'reference_{sample:06d}_{rep}.npy'
   if path.exists():wave=np.load(path,allow_pickle=False)
   else:wave=renderer.render([pos.tolist()]).samples[0];np.save(path,wave,allow_pickle=False)
   waves.append(wave)
  refs[sample]=waves;return waves
 try:
  for k in keys:
   native(k);print('RIR_KEY',k/sr,flush=True)
  refinements=[]
  for iteration in range(8):
   timeline=timeline_type([native(k) for k in keys],keys);checks=[];new=[];segments=[]
   for lo,hi in zip(keys[:-1],keys[1:]):
    mid=(lo+hi)//2;estimate=effective_emission_filter(timeline,mid,n);pos=position_at(track,mid/sr)
    raw=[compare_rirs(estimate,w,source=pos,listener=listener,forward=forward,sample_rate=sr) for w in reference(mid)];gate=qualify_energy(raw)
    # Also bound source-direction variation throughout every native 30 Hz
    # knot in this interval. The RIR gate checks the independently read sound.
    pts=np.asarray([position_at(track,t) for t in [lo/sr,hi/sr,*[f['time_s'] for f in frames if lo/sr<f['time_s']<hi/sr]]])-listener
    pts/=np.linalg.norm(pts,axis=1,keepdims=True);sweep=float(np.degrees(np.arccos(np.clip((pts@pts.T).min(),-1,1))))
    segment={'start_s':lo/sr,'end_s':hi/sr,'midpoint_s':mid/sr,'source_direction_sweep_deg':sweep,'raw_independent_checks':raw,'qualification':gate}
    segments.append(segment);print('RIR_INTERVAL',lo/sr,hi/sr,gate['status'],gate['failures'],flush=True)
    if gate['status']=='pass' and sweep<=5:checks.extend(raw)
    else:
     refinements.append(segment)
     if (hi-lo)/sr<=1/60:raise ValueError('RIR_unchanged_absolute_gates_fail_at_1over60s')
     new.append(mid)
    (a.output/'progress.json').write_text(json.dumps({'iteration':iteration,'segments':segments,'keys':keys,'cache_stats':cache.stats},indent=2))
    if len(new)>=2:raise ValueError('two_failed_RIR_intervals_stop_immediately_for_diagnosis')
   if not new:break
   if len(new)>=2:
    # Stop and inspect before scheduling a whole failing refinement batch.
    raise ValueError('two_failed_RIR_intervals_stop_for_diagnosis')
   keys=sorted(set(keys+new))
  else:raise ValueError('RIR_refinement_budget')
  gate=qualify_energy(checks)
  first=native(keys[0]);bad=compare_rirs(first*2,first,source=position_at(track,0),listener=listener,forward=forward,sample_rate=sr)
  negative=qualify_energy([bad])
  out.update(status='pass' if gate['status']=='pass' and negative['status']=='fail' else 'fail',asset_id=motion['asset_id'],segments=segments,qualification=gate,directions=list(directions.values()),refinements=refinements,
      maximum_native_direction_error_deg=max(d['angle_deg'] for d in directions.values()),maximum_segment_direction_sweep_deg=max(s['source_direction_sweep_deg'] for s in segments),
      maximum_abs_energy_error_db=max(abs(x) for c in checks for x in [c['direct_energy_difference_db'],c['total_energy_difference_db'],*[b['energy_difference_db'] for b in c['octave_energy']]]),
      amplitude_6dB_negative_control={'check':bad,'qualification':negative},cache_stats=cache.stats,simulation=identity['simulation'],native_pose_keys=keys,
      policy='all three independent raw native references individually pass unchanged 1dB/5deg/delay/decay gates; no median, gain fit or noise subtraction',convolution='existing correlated emission-aligned kernel, independently tested midpoints')
 except Exception as error:out.update(status='blocked',error=repr(error),traceback=traceback.format_exc(),directions=list(directions.values()),cache_stats=cache.stats)
 finally:
  cache.db.close();out['cpu_s']=time.process_time()-start;out['fresh_process_per_RIR']=a.fresh_process;out['frequency_coherence']=a.frequency_coherence;out['crossfade_revision']=revision;(a.output/'audio.json').write_text(json.dumps(out,indent=2))
 print({k:v for k,v in out.items() if k not in ('segments','directions','refinements','traceback','amplitude_6dB_negative_control')},flush=True)
 if out['status']!='pass':raise SystemExit(2)
if __name__=='__main__':main()
