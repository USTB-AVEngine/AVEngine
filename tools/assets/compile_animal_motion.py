"""Compile a bounded target-preserving walking probe for native validation."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from avengine.assets.animal_motion.profiles import Rig
from avengine.assets.animal_motion.walk import ConstrainedWalk
from avengine.assets.animal_motion.checks import frame_checks,direction_negative_controls
from avengine.assets.roster_motion.pose import rigid_glb
from avengine.assets.roster_motion.episode import externalize_unchanged_images

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=4.8);p.add_argument('--speed',type=float,default=.08);p.add_argument('--period',type=float,default=2.4);p.add_argument('--write-poses',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);frames=[];start=time.process_time()
    record=next(r for r in json.loads(a.inventory.read_text())['assets'] if r['asset_id']==a.asset)
    result={'asset_id':a.asset,'status':'pending','source_inventory':str(a.inventory),'thresholds_changed':False}
    try:
        rig=Rig.load(record);walk=ConstrainedWalk(rig,speed=a.speed,period=a.period);result.update(provenance=rig.provenance,support_plane=walk.support_plane,source_master=str(rig.path),source_source_time_forward=True,source_mirrored=False,root_speed_m_s=a.speed,gait_period_s=a.period)
        (a.output/'frames').mkdir(exist_ok=True);ed=externalize_unchanged_images(rig.document,rig.binary,a.output/'textures') if a.write_poses else None
        for i,t in enumerate(np.arange(round(a.seconds*30)+1)/30.):
            parts,row=walk.frame(float(t));row['frame_index']=i
            if a.write_poses:
                path=a.output/'frames'/f'{i:04d}.glb';rigid_glb(ed,rig.binary,parts,path);row['pose_glb']=str(path)
            frames.append(row)
        check_frames=[{**f,'root_world_m':f['world_root_m'],
            'head_world_m':(np.asarray(f['head_m'])+f['world_root_m']).tolist(),
            'tail_world_m':(np.asarray(f['tail_m'])+f['world_root_m']).tolist()} for f in frames]
        checks=frame_checks(check_frames)
        controls=direction_negative_controls(check_frames)
        result.update(status='flat_pass_native_pending' if checks['status']==controls['status']=='pass' else 'flat_fail',
            checks=checks,direction_negative_controls=controls,frames=frames,frame_count=len(frames),
            max_contact_delta_m=max(abs(v['floor_delta_m']) for f in frames for v in f['feet'].values() if v['in_contact']),
            max_all_paw_gap_m=max(abs(v['floor_delta_m']) for f in frames for v in f['feet'].values()),
            min_full_skin_m=min(f['body_floor_min_m'] for f in frames),
            max_lock_error_m=max(v['support_lock_error_m'] for f in frames for v in f['feet'].values() if v['in_contact']),
            max_reach_error_m=max(f['ik_reach_error_m'] for f in frames))
    except Exception as e:result.update(status='failed',error=repr(e),traceback=traceback.format_exc(),frames=frames)
    result['cpu_s']=time.process_time()-start;(a.output/'motion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print({k:v for k,v in result.items() if k not in ('frames','provenance','support_plane','traceback','checks','direction_negative_controls')},flush=True)
    if result['status'] in ('failed','flat_fail'):raise SystemExit(2)
if __name__=='__main__':main()
