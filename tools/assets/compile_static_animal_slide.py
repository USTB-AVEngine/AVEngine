"""Flat-ground compiler for explicitly unanimated registered quadrupeds."""
import argparse,json,pathlib
import numpy as np
from avengine.assets.animal_motion.slide import StandingSlide
from avengine.assets.animal_motion.checks import frame_checks,direction_negative_controls

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=pathlib.Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--speed',type=float,default=.03);p.add_argument('--seconds',type=float,default=4.8);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 record=next(x for x in json.loads(a.inventory.read_text())['assets'] if x['asset_id']==a.asset);slide=StandingSlide(record,speed=a.speed);frames=[]
 for i in range(round(a.seconds*30)+1):
  parts,row=slide.frame(i/30);root=np.asarray(row['world_root_m']);row.update(frame_index=i,root_world_m=root.tolist(),head_world_m=(root+row['head_m']).tolist(),tail_world_m=(root+row['tail_m']).tolist(),mouth_world_m=(root+row['mouth_m']).tolist());frames.append(row)
 checks=frame_checks(frames,native=False,mode='slide_no_animation');negative=direction_negative_controls(frames)
 result={'asset_id':a.asset,'status':'flat_pass_native_pending' if checks['status']==negative['status']=='pass' else 'flat_fail','motion_mode':'slide_no_animation','joint_animation_present':False,'frame_count':len(frames),'frames':frames,'checks':checks,'direction_negative_controls':negative,'source_master':str(slide.rig.path),'support_plane':slide.support_plane,'native_status':'not_run'}
 (a.output/'motion.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('frames','checks','direction_negative_controls')},flush=True)
 if result['status']!='flat_pass_native_pending':raise SystemExit(2)
if __name__=='__main__':main()
