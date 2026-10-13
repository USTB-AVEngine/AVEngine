"""Fresh exact-pose RIR endpoints, at most six single-core native children."""
import argparse,json,os,sqlite3,subprocess,sys
from pathlib import Path
import numpy as np
from tools.timeline.pilot_continuous_scenes import native_renderer
from avengine.timeline.v7_registered_motion_audio import position_at
from avengine.acoustics.house_acceptance_metrics import foa_direction

def main():
 p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cores',default='20,21,22,23,24,25');p.add_argument('--immutable-wave-root',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cache=a.output/'cache';cache.mkdir();m=json.loads(a.motion.read_text());inputs=json.loads(a.runtime.read_text());h=json.loads(a.house.read_text());h['room']['placement']['camera_m']=m['listener_m'];track=[{'time_s':f['time_s'],'position_m':f['mouth_world_m']} for f in m['frames']];h['room']['placement']['source_1_m']=track[0]['position_m']
 if m.get('checks',{}).get('status')!='pass':raise ValueError('native geometry is not qualified')
 for k in ('direct_ray_count','indirect_ray_count','source_ray_count'):inputs['simulation'][k]*=9
 renderer,identity=native_renderer(inputs,h);setup=renderer.setup_report;del renderer
 identity['context_policy']='fresh_process_per_RIR_deterministic_native_replay';(cache/'configuration.json').write_text(json.dumps(identity,indent=2));(a.output/'setup.json').write_text(json.dumps(setup,indent=2));db=sqlite3.connect(cache/'positions.sqlite3');db.execute('CREATE TABLE rirs(id INTEGER PRIMARY KEY, pose TEXT UNIQUE NOT NULL, completed INTEGER NOT NULL DEFAULT 0)');db.commit()
 n=round(m['frames'][-1]['time_s']*16000)+1;keys=sorted(set([*range(0,n,3200),n-1]));cores=[int(x) for x in a.cores.split(',')];direction=[];forward=np.array(m['camera_forward']);forward[1]=0;forward/=np.linalg.norm(forward)
 for start in range(0,len(keys),len(cores)):
  running=[]
  for core,k in zip(cores,keys[start:start+len(cores)]):
   point=position_at(track,k/16000);key=json.dumps(point.tolist(),separators=(',',':'));db.execute('INSERT INTO rirs(pose) VALUES (?)',(key,));idx=db.execute('SELECT id FROM rirs WHERE pose=?',(key,)).fetchone()[0];db.commit();folder=a.output/f'native_{idx:06d}';folder.mkdir();wave=cache/f'rir_{idx:06d}.npy';request=folder/'request.json';request.write_text(json.dumps({'inputs':inputs,'house':h,'position_m':point.tolist(),'wave':str(wave),'receipt':str(folder/'receipt.json'),'immutable_wave_root':str(a.immutable_wave_root) if a.immutable_wave_root is not None else None},indent=2));log=(folder/'native.log').open('wb');proc=subprocess.Popen([sys.executable,'-m','tools.assets.render_animal_rir_request','--request',str(request)],stdout=log,stderr=subprocess.STDOUT,preexec_fn=lambda core=core:os.sched_setaffinity(0,{core}));running.append((proc,log,idx,k,point,wave));print('RIR_ENDPOINT_PID',proc.pid,core,k/16000,flush=True)
  failures=0
  for proc,log,idx,k,point,wave in running:
   code=proc.wait();log.close()
   if code:raise RuntimeError('native endpoint failed')
   db.execute('UPDATE rirs SET completed=1 WHERE id=?',(idx,));db.commit();d=foa_direction(np.load(wave,allow_pickle=False),np.array(m['listener_m']),point,forward,sample_rate=16000);err=d['check']['angle_deg'];direction.append({'time_s':k/16000,'angle_deg':err})
   if err is None or err>5:failures+=1
  (a.output/'endpoint_directions.json').write_text(json.dumps(direction,indent=2))
  if failures:raise ValueError('native endpoint direction exceeds unchanged 5deg; stop before references')
 db.close();(a.output/'endpoints_complete.json').write_text(json.dumps({'status':'pass','asset_id':m['asset_id'],'motion':str(a.motion),'keys':keys,'directions':direction},indent=2))
if __name__=='__main__':main()
