"""Two-interval batches of fresh native references with unchanged strict gates."""
import argparse,json,pathlib,sqlite3,subprocess,sys,os,time,copy,shutil
import numpy as np
from avengine.assets.animal_motion.acoustics import FreshProcessRenderer
from avengine.assets.animal_motion.rir_crossfade import FrequencyCoherentTimeline,REVISION
from avengine.acoustics.moving_rir_calibration import effective_emission_filter
from avengine.acoustics.moving_rir_acceptance import compare_rirs,qualify_energy
from avengine.acoustics.house_acceptance_metrics import foa_direction
from avengine.timeline.v7_registered_motion_audio import position_at

def main():
 p=argparse.ArgumentParser();p.add_argument('--previous',type=pathlib.Path,required=True);p.add_argument('--motion',type=pathlib.Path,required=True);p.add_argument('--house',type=pathlib.Path,required=True);p.add_argument('--runtime',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--cores',default='20,21,22,23,24,25');p.add_argument('--refine-failed',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);shutil.copytree(a.previous/'cache',a.output/'cache');cores=[int(x) for x in a.cores.split(',')];assert 1<=len(cores)<=6
 for path in a.previous.glob('reference_*.npy'):shutil.copy2(path,a.output/path.name)
 inputs=json.loads(a.runtime.read_text());house=json.loads(a.house.read_text());m=json.loads(a.motion.read_text());frames=m['frames'];track=[{'time_s':f['time_s'],'position_m':f['mouth_world_m']} for f in frames];house['room']['placement']['camera_m']=m['listener_m'];listener=np.array(m['listener_m']);forward=np.array(m['camera_forward']);forward[1]=0;forward/=np.linalg.norm(forward)
 oldid=json.loads((a.output/'cache/configuration.json').read_text());inputs['simulation']=oldid['simulation']
 sr=16000;n=round(frames[-1]['time_s']*sr)+1;keys=sorted(set([*range(0,n,3200),n-1]));db=sqlite3.connect(a.output/'cache/positions.sqlite3');filters=[];directions=[]
 if a.refine_failed:
  previous=json.loads((a.previous/'progress.json').read_text())
  keys=sorted(set([0,n-1,*[round(v*sr) for x in previous['segments'] for v in (x['start_s'],x['end_s'])],*keys]))
  failed=[x for x in previous['segments'] if x['qualification']['status']!='pass']
  for interval in failed:
   lo,hi=round(interval['start_s']*sr),round(interval['end_s']*sr)
   if (hi-lo)/sr<=1/60:raise ValueError('minimum native interval reached; no threshold relaxation')
   mid=(lo+hi)//2;files=[a.previous/f'reference_{mid:06d}_{rep}.npy' for rep in range(3)]
   waves=[np.load(f,allow_pickle=False) for f in files]
   if not all(np.array_equal(waves[0],x) for x in waves[1:]):raise ValueError('only identical deterministic native references can be promoted to an endpoint')
   point=position_at(track,mid/sr);key=json.dumps(point.tolist(),separators=(',',':'));db.execute('INSERT INTO rirs(pose) VALUES (?) ON CONFLICT(pose) DO NOTHING',(key,));idx=db.execute('SELECT id FROM rirs WHERE pose=?',(key,)).fetchone()[0];target=a.output/'cache'/f'rir_{idx:06d}.npy'
   if target.exists() and not np.array_equal(np.load(target,allow_pickle=False),waves[0]):raise ValueError('promoted native endpoint differs')
   if not target.exists():shutil.copy2(files[0],target)
   db.execute('UPDATE rirs SET completed=1 WHERE id=?',(idx,));db.commit();keys.append(mid)
  keys=sorted(set(keys))
  (a.output/'refinement.json').write_text(json.dumps({'failed_intervals':failed,'new_native_keys':keys,'reference_waveforms_modified':False,'new_midpoints_require_three_fresh_native_executions':True},indent=2))
 for k in keys:
  point=position_at(track,k/sr);r=db.execute('SELECT id,completed FROM rirs WHERE pose=?',(json.dumps(point.tolist(),separators=(',',':')),)).fetchone()
  if r is None or not r[1]:raise ValueError('existing endpoint cache incomplete')
  wave=np.load(a.output/'cache'/f'rir_{r[0]:06d}.npy',allow_pickle=False);filters.append(wave);d=foa_direction(wave,listener,point,forward,sample_rate=sr);directions.append({'time_s':k/sr,'mouth_m':point.tolist(),'angle_deg':d['check']['angle_deg'],'native':d})
 db.close();timeline=FrequencyCoherentTimeline(filters,keys);intervals=list(zip(keys[:-1],keys[1:]));segments=[];checks=[];stopped=[];start=time.time();requests=a.output/'fresh_native';requests.mkdir()
 result={'asset_id':m['asset_id'],'motion_receipt':str(a.motion),'previous_native_cache':str(a.previous),'crossfade_revision':REVISION,'thresholds_changed':False,'independent_reference_count_per_midpoint':3,'reference_rule':'every raw independently executed native reference; no mean or median','worker_cpu_affinities':cores,'status':'pending'}
 try:
  for batch_start in range(0,len(intervals),2):
   batch=intervals[batch_start:batch_start+2];jobs=[]
   for lo,hi in batch:
    mid=(lo+hi)//2;point=position_at(track,mid/sr).tolist()
    for rep in range(3):
     wave=a.output/f'reference_{mid:06d}_{rep}.npy'
     if wave.exists():continue
     folder=requests/f'{mid:06d}_{rep}';folder.mkdir();request=folder/'request.json';request.write_text(json.dumps({'inputs':inputs,'house':house,'position_m':point,'wave':str(wave),'receipt':str(folder/'receipt.json')},indent=2));jobs.append((request,folder))
   for offset in range(0,len(jobs),len(cores)):
    running=[]
    for core,(request,folder) in zip(cores,jobs[offset:offset+len(cores)]):
     log=(folder/'native.log').open('wb');proc=subprocess.Popen([sys.executable,'-m','tools.assets.render_animal_rir_request','--request',str(request)],stdout=log,stderr=subprocess.STDOUT,preexec_fn=lambda core=core:os.sched_setaffinity(0,{core}))
     running.append((proc,log,core));print('NATIVE_REFERENCE_PID',proc.pid,'CPU',core,request,flush=True)
    for proc,log,core in running:
     code=proc.wait();log.close()
     if code:raise RuntimeError('fresh native reference failed:'+str(code))
   for lo,hi in batch:
    mid=(lo+hi)//2;point=position_at(track,mid/sr);estimate=effective_emission_filter(timeline,mid,n);raw=[compare_rirs(estimate,np.load(a.output/f'reference_{mid:06d}_{rep}.npy',allow_pickle=False),source=point,listener=listener,forward=forward,sample_rate=sr) for rep in range(3)];verdict=qualify_energy(raw)
    pts=np.array([position_at(track,t) for t in [lo/sr,hi/sr,*[f['time_s'] for f in frames if lo/sr<f['time_s']<hi/sr]]])-listener;pts/=np.linalg.norm(pts,axis=1,keepdims=True);sweep=float(np.degrees(np.arccos(np.clip((pts@pts.T).min(),-1,1))))
    segments.append({'start_s':lo/sr,'end_s':hi/sr,'midpoint_s':mid/sr,'raw_independent_checks':raw,'source_direction_sweep_deg':sweep,'qualification':verdict});checks.extend(raw);print('QUALIFIED_SEGMENT',lo/sr,verdict['status'],verdict['failures'],flush=True)
    if verdict['status']!='pass' or sweep>5:stopped.append(segments[-1])
   (a.output/'progress.json').write_text(json.dumps({'segments':segments,'failed_intervals':stopped},indent=2))
   if len(stopped)>=2:raise ValueError('two_failed_intervals_stop_for_diagnosis')
  qualification=qualify_energy(checks);negative=compare_rirs(filters[0]*2,filters[0],source=position_at(track,0),listener=listener,forward=forward,sample_rate=sr);bad=qualify_energy([negative])
  okay=qualification['status']=='pass' and bad['status']=='fail' and all(d['angle_deg'] is not None and d['angle_deg']<=5 for d in directions) and all(s['source_direction_sweep_deg']<=5 for s in segments)
  result.update(status='pass' if okay else 'fail',qualification=qualification,segments=segments,directions=directions,amplitude_6dB_negative_control={'check':negative,'qualification':bad},maximum_native_direction_error_deg=max(d['angle_deg'] for d in directions),maximum_segment_direction_sweep_deg=max(s['source_direction_sweep_deg'] for s in segments),maximum_abs_energy_error_db=max(abs(x) for c in checks for x in [c['direct_energy_difference_db'],c['total_energy_difference_db'],*[b['energy_difference_db'] for b in c['octave_energy']]]))
 except Exception as error:
  import traceback
  result.update(status='blocked',error=repr(error),traceback=traceback.format_exc(),segments=segments,directions=directions)
 result['wall_s']=time.time()-start;(a.output/'audio.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ('segments','directions','traceback','amplitude_6dB_negative_control')},flush=True)
 if result['status']!='pass':raise SystemExit(2)
if __name__=='__main__':main()
