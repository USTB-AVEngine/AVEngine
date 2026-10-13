"""Compare endpoint-only crossfades against all already recorded raw references."""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
from avengine.assets.animal_motion.rir_crossfade import FrequencyCoherentTimeline,REVISION
from avengine.acoustics.moving_rir_calibration import effective_emission_filter
from avengine.acoustics.moving_rir_acceptance import compare_rirs,qualify_energy
from avengine.timeline.v7_registered_motion_audio import position_at

def main():
 p=argparse.ArgumentParser();p.add_argument('--audio',type=Path,required=True);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 m=json.loads(a.motion.read_text());track=[{'time_s':f['time_s'],'position_m':f['mouth_world_m']} for f in m['frames']];db=sqlite3.connect(a.audio/'cache/positions.sqlite3');sr=16000;n=round(m['frames'][-1]['time_s']*sr)+1;keys=sorted(set([*range(0,n,3200),n-1]));filters=[]
 for k in keys:
  point=position_at(track,k/sr);row=db.execute('SELECT id,completed FROM rirs WHERE pose=?',(json.dumps(point.tolist(),separators=(',',':')),)).fetchone()
  if row is None or not row[1]:raise ValueError('missing original native endpoint')
  filters.append(np.load(a.audio/'cache'/f'rir_{row[0]:06d}.npy',allow_pickle=False))
 timeline=FrequencyCoherentTimeline(filters,keys);forward=np.array(m['camera_forward']);forward[1]=0;forward/=np.linalg.norm(forward);listener=np.array(m['listener_m']);results=[]
 for lo,hi in zip(keys[:-1],keys[1:]):
  mid=(lo+hi)//2;files=[a.audio/f'reference_{mid:06d}_{rep}.npy' for rep in range(3)]
  if not all(f.exists() for f in files):continue
  est=effective_emission_filter(timeline,mid,n);checks=[compare_rirs(est,np.load(f,allow_pickle=False),source=position_at(track,mid/sr),listener=listener,forward=forward,sample_rate=sr) for f in files]
  row={'start_s':lo/sr,'end_s':hi/sr,'checks':checks,'qualification':qualify_energy(checks)};results.append(row);print('OCTAVE_COHERENCE',lo/sr,hi/sr,row['qualification']['status'],row['qualification']['failures'],flush=True)
 all_checks=[c for r in results for c in r['checks']];result={'revision':REVISION,'source_native_audio':str(a.audio),'motion_receipt':str(a.motion),'status':qualify_energy(all_checks),'segments':results,'measured_intervals':len(results),'requested_intervals':len(keys)-1,'full_coverage':len(results)==len(keys)-1,'native_RIRs_modified':False,'reference_fit':False};(a.output/'audit.json').write_text(json.dumps(result,indent=2));db.close()
if __name__=='__main__':main()
