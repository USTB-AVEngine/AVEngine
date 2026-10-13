"""Re-render an identical qualified pose sequence with a true 3-D camera."""
import argparse,json,copy,time,subprocess
from pathlib import Path
import numpy as np
from avengine.capture.continuous_native_video import NativeContinuousVideo
from avengine.assets.animal_motion.native import camera,instantiate,overlay
from avengine.assets.animal_motion.checks import frame_checks,direction_negative_controls
from avengine.assets.roster_motion.contact import yaw_matrix
from avengine.assets.animal_motion.profiles import Rig
from avengine.assets.roster_motion.pose import rigid_glb
from avengine.assets.roster_motion.episode import externalize_unchanged_images

def main():
 p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 old=json.loads(a.motion.read_text());house=json.loads(a.house.read_text());inputs=json.loads(a.runtime.read_text());eye=np.asarray(old['listener_m']);fs=old['frames'];target=(np.asarray(fs[0]['root_world_m'])+fs[-1]['root_world_m'])/2+[0,.3,0];forward=target-eye;forward/=np.linalg.norm(forward)
 v=NativeContinuousVideo(house,{'listener_m':eye.tolist(),'camera_forward':forward.tolist(),'sources':{}},inputs,[],width=640,height=480,source_material_cache_directory=a.output/'materials');start=time.process_time();result=copy.deepcopy(old);result.update(frames=[],images=[],status='pending',geometry_source_receipt=str(a.motion),camera_repair='full 3-D look-at; generic production camera_basis intentionally removes pitch and is inappropriate for low animal inspection');encoder=None
 try:
  eye,cf,cr,cu=camera(v,eye,target);velocity=(np.asarray(fs[-1]['root_world_m'])-fs[0]['root_world_m'])/(fs[-1]['time_s']-fs[0]['time_s']);selected={0,len(fs)//4,len(fs)//2,3*len(fs)//4,len(fs)-1};rm=v.sim.get_rigid_object_manager()
  # Recover the unchanged per-frame body yaw from the source head/root data.
  rotation=yaw_matrix(np.arctan2(-velocity[0],-velocity[2]));cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(a.output/'motion.mp4')];encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  for i,original in enumerate(fs):
   f=copy.deepcopy(original);matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=f['root_world_m'];obj=instantiate(v,f['pose_glb'],matrix);native=np.asarray(obj.root_scene_node.absolute_transformation(),float)
   bn=obj.root_scene_node.create_child();bn.transformation=v.mn.Matrix4(np.asarray(f['mouth_bone_matrix']));mouth=bn.create_child();mouth.translation=v.mn.Vector3(f['mouth_bone_local_m']);actual=np.asarray(mouth.absolute_translation)
   expected=np.asarray(original['mouth_world_m']);error=float(np.linalg.norm(actual-expected))
   if error>.0001:raise ValueError('replay_pose_or_yaw_changed_actual_mouth')
   f['native_emitter_readback_error_m']=error;f['mouth_world_m']=actual.tolist();f['root_world_m']=native[:3,3].tolist()
   obs=v.sim.get_sensor_observations();rgb=np.asarray(obs['rgb'])[...,:3];mask=np.asarray(obs['semantic'])==700001;yy,xx=np.where(mask);f['visible_pixels']=int(mask.sum());f['complete_silhouette_in_frame']=bool(len(xx)>100 and min(xx.min(),yy.min(),639-xx.max(),479-yy.max())>=3);encoder.stdin.write(rgb.tobytes())
   if i in selected:
    image=a.output/f'frame_{i:04d}.png';overlay(rgb,eye,(cf,cr,cu),f['head_world_m'],f['tail_world_m'],np.asarray(f['root_world_m'])+[0.,.02,0.],velocity).save(image);result['images'].append(str(image))
   result['frames'].append(f);rm.remove_object_by_id(obj.object_id)
  encoder.stdin.close();code=encoder.wait();encoder=None
  if code:raise ValueError('encoder_failed')
  checks=frame_checks(result['frames'],native=True)
  if not all(f['complete_silhouette_in_frame'] for f in result['frames']):checks['status']='fail';checks['failures'].append('clipped_or_under_100_visible_pixels')
  result.update(status='geometry_visual_pass_audio_pending' if checks['status']=='pass' else 'geometry_visual_fail',checks=checks,direction_negative_controls=direction_negative_controls(result['frames']),min_visible_pixels=min(f['visible_pixels'] for f in result['frames']),camera_forward=cf.tolist(),renderer=v.receipt)
  record=next(x for x in json.loads(a.inventory.read_text())['assets'] if x['asset_id']==old['asset_id']);rig=Rig.load(record);(a.output/'before/frames').mkdir(parents=True,exist_ok=True);ed=externalize_unchanged_images(rig.document,rig.binary,a.output/'before/textures');result['before_images']=[]
  for i in sorted(selected):
   f=fs[i];g=rig.globals(f['source_phase_s']);parts=rig.skin(g);marks=rig.landmarks(g);path=a.output/'before/frames'/f'{i:04d}.glb';rigid_glb(ed,rig.binary,parts,path);matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=f['root_world_m'];obj=instantiate(v,path,matrix);obs=v.sim.get_sensor_observations();rgb=np.asarray(obs['rgb'])[...,:3];image=a.output/'before'/f'frame_{i:04d}.png';head=rotation@marks['head']+matrix[:3,3];tail=rotation@marks['tail']+matrix[:3,3];overlay(rgb,eye,(cf,cr,cu),head,tail,matrix[:3,3]+[0,.02,0],velocity).save(image);result['before_images'].append(str(image));rm.remove_object_by_id(obj.object_id)
 except Exception as e:
  import traceback
  result.update(status='failed',error=repr(e),traceback=traceback.format_exc())
 finally:
  if encoder is not None:
   encoder.stdin.close();encoder.wait()
  v.close();result['replay_cpu_s']=time.process_time()-start;(a.output/'native_motion.json').write_text(json.dumps(result,indent=2))
 print({k:v for k,v in result.items() if k in ('asset_id','status','error','min_visible_pixels','replay_cpu_s')},flush=True)
 if result['status']!='geometry_visual_pass_audio_pending':raise SystemExit(2)
if __name__=='__main__':main()
