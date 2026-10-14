"""Separate native same-pose variance from interpolation error without fitting."""
import argparse,json,itertools,time
from pathlib import Path
import numpy as np
from avengine.acoustics.moving_rir_acceptance import compare_rirs,qualify_energy
from avengine.timeline.v7_registered_motion_audio import position_at
from tools.timeline.pilot_continuous_scenes import native_renderer

def main():
 p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--existing',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--quality',type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 d=json.loads(a.motion.read_text());track=[{'time_s':f['time_s'],'position_m':f['mouth_world_m']} for f in d['frames']];listener=np.array(d['listener_m']);forward=np.array(d['camera_forward']);forward[1]=0;forward/=np.linalg.norm(forward);rows=[]
 def check(waves,sample,label):
  pos=position_at(track,sample/16000);cs=[compare_rirs(x,y,source=pos,listener=listener,forward=forward,sample_rate=16000) for x,y in itertools.combinations(waves,2)];row={'setting':label,'sample':sample,'checks':cs,'qualification':qualify_energy(cs)};rows.append(row);(a.output/'variance.json').write_text(json.dumps(rows,indent=2));print('VARIANCE',label,sample,row['qualification'],flush=True)
 for sample in (1600,8000):
  files=[a.existing/f'reference_{sample:06d}_{i}.npy' for i in range(3)]
  if all(f.is_file() for f in files):check([np.load(f,allow_pickle=False) for f in files],sample,'existing_9x_same_pose_references')
 inputs=json.loads(a.runtime.read_text());house=json.loads(a.house.read_text());house['room']['placement']['camera_m']=listener.tolist();house['room']['placement']['source_1_m']=track[0]['position_m']
 for k in ('direct_ray_count','indirect_ray_count','source_ray_count'):inputs['simulation'][k]*=a.quality
 inputs['simulation']['thread_count']=1;renderer,identity=native_renderer(inputs,house);(a.output/'setup.json').write_text(json.dumps(identity,indent=2))
 for sample in (1600,8000):
  waves=[]
  for i in range(3):
   path=a.output/f'reference_{sample:06d}_{i}.npy';wave=renderer.render([position_at(track,sample/16000).tolist()]).samples[0];np.save(path,wave,allow_pickle=False);waves.append(wave);print('PROBE_RIR',sample,i,flush=True)
  check(waves,sample,f'native_{a.quality}x_same_pose_references')
if __name__=='__main__':main()
