"""Moving native UE skeletal replay, actual bones, mouth and rendered paw soles."""
import argparse,json,math,time,traceback
from pathlib import Path
import numpy as np
from avengine.assets.animal_motion.ue_runtime import apply_native_rgb_profile,launch_native,camera_helpers,apply_camera,to_ue,from_ue,close_native
from avengine.assets.animal_motion.ue_floor import NativeFloorReference
from avengine.assets.animal_motion.stage import OriginalVerticalStageRay
from avengine.assets.animal_motion.native import StageClearance,overlay
from avengine.assets.animal_motion.checks import frame_checks,direction_negative_controls
from avengine.assets.roster_motion.io import read_glb
from avengine.assets.roster_motion.pose import pose,globals_at,quat_matrix,accessor
from avengine.assets.roster_motion.episode import triangle_mesh
from avengine.backends.spear_ue.research_runtime import _read_unreal_pose_vector,attach_skeletal_emitter_component,read_scene_component_pose,close_scene_capture

def unbox(v):
 for _ in range(4):
  if not isinstance(v,dict):return v
  lowered={str(k).casefold():x for k,x in v.items()}
  if 'returnvalue' in lowered:v=lowered['returnvalue'];continue
  if 'return_value' in lowered:v=lowered['return_value'];continue
  if len(v)==1 and isinstance(next(iter(v.values())),dict):v=next(iter(v.values()));continue
  return lowered
 raise ValueError('native transform wrapper nesting exceeds four')
def xyz(v):
 return np.asarray(_read_unreal_pose_vector(v,keys=('X','Y','Z'),owner='animal_native_transform'),float)
def matrix(v):
 v=unbox(v);m=np.eye(4);q=unbox(v['rotation']);m[:3,:3]=quat_matrix([q[k] for k in ('x','y','z','w')])@np.diag(xyz(v['scale3d']));m[:3,3]=xyz(v['translation']);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--motion',type=Path,required=True);p.add_argument('--imported',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 h=json.loads(a.house.read_text());m=json.loads(a.motion.read_text());imp=json.loads(a.imported.read_text());ue=h['input']['ue_config'].copy();ue.update(uproject=str(a.root/'ue/SpearSim/SpearSim.uproject'),output_root=str(a.root),graphics_adapter=0,physical_gpu_index=0,network_guard_so=str(a.root/'ue/network_guard.so'),unpause_animation_sampling=True)
 if imp['status']!='native_clip_imported_pose_readback_pending_geometry':raise ValueError('native clip readback is missing')
 (a.output/'ue_config.json').write_text(json.dumps(ue,indent=2))
 instance,game,launch=launch_native(ue,h['input']['ue_map_path']);spawn,depth_reader,rgb_reader,*_=camera_helpers();cam=components=footcam=footcomponents=actor=component=emitter=None;result={'asset_id':m['asset_id'],'house':h['house'],'backend':'UE/SPEAR','status':'pending','frames':[],'images':[],'motion_mode':'walk_animation','launch':launch}
 def frame(fn=None):
  with instance.begin_frame():
   if fn:fn()
  with instance.end_frame():pass
 try:
  nativefloor=NativeFloorReference(ue['native_floor_surface']);mesh=nativefloor.mesh;rays=OriginalVerticalStageRay(Path(ue['native_floor_surface']['native_editor_audit']),mesh=mesh);clearance=StageClearance(mesh,m['root_m'][1])
  from avengine.acoustics.runtime import load_habitat_runtime
  runtime=json.loads((a.root/'inputs/runtime.json').read_text());hs,_=load_habitat_runtime(runtime_mode='current-installed',**runtime['runtime']);pf=hs.PathFinder()
  if not pf.load_nav_mesh(h['input']['navmesh_source']):raise ValueError('admitted UE-derived NavMesh missing')
  d,b=read_glb(m['motion_glb']);names={n['name']:i for i,n in enumerate(d['nodes'])};joint_names=m['bone_names'];C=np.array([[1,0,0],[0,0,1],[0,1,0]]);A=np.eye(4);A[:3,:3]=C*100;Ai=np.linalg.inv(A)
  p0=np.array(m['root_m']);reference=np.asarray(m['frames'][0]['mouth_m']);end=np.asarray(m['frames'][-1]['world_root_m']);center=p0+end/2+[0,.3,0];eye=center+[1.25,.65,-.45];delta=center-eye
  view={'position_m':eye.tolist(),'yaw_deg':math.degrees(math.atan2(delta[2],delta[0])),'pitch_deg':math.degrees(math.atan2(delta[1],np.linalg.norm(delta[[0,2]]))),'hfov_deg':85.}
  def setup():
   nonlocal actor,component,emitter,cam,components,footcam,footcomponents
   actor=game.unreal_service.spawn_actor(uclass='ASkeletalMeshActor',spawn_parameters={'SpawnCollisionHandlingOverride':'AlwaysSpawn'});actor.SetActorEnableCollision(bNewActorEnableCollision=False);component=game.unreal_service.get_component_by_class(actor=actor,uclass='USkeletalMeshComponent');component.SetMobility(NewMobility='Movable');asset=game.unreal_service.load_object(uclass='USkeletalMesh',name=imp['skeletal_mesh']);component.SetSkeletalMeshAsset(NewMesh=asset);component.SetCollisionEnabled(NewType='NoCollision');component.set_property_value(property_name='VisibilityBasedAnimTickOption',property_value='AlwaysTickPoseAndRefreshBones');component.SetComponentTickEnabled(bEnabled=True)
   clip=game.unreal_service.load_object(uclass='UAnimSequence',name=imp['animation']);component.PlayAnimation(NewAnimToPlay=clip,bLooping=False);component.SetPlayRate(Rate=0.);component.Stop()
   offset=(C@np.asarray(m['mouth_bone_local_m'])*100).tolist();emitter=attach_skeletal_emitter_component(game,actor_id=m['asset_id'],skeletal_component=component,attachment={'attachment_type':'bone','name':imp['bone_mapping'][m['mouth_bone']],'local_offset_cm':offset})
   cam,components=spawn(game,horizontal_fov_deg=85,width=640,height=480);footcam,footcomponents=spawn(game,horizontal_fov_deg=85,width=512,height=512)
   for key in ['rgb','object_ids']:footcomponents[key].set_property_value(property_name='bCaptureEveryFrame',property_value=False)
   footcomponents['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList');footcomponents['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[actor]);apply_camera(cam,view);apply_native_rgb_profile(components['rgb'],ue)
  frame(setup)
  for _ in range(45):frame()
  rows=m['frames'][:a.limit] if a.limit else m['frames'];selected={0,len(rows)//2,len(rows)-1}
  for i,source in enumerate(rows):
   root=p0+np.asarray(source['world_root_m']);t=source['time_s']
   def update():
    actor.K2_SetActorLocationAndRotation(NewLocation=dict(zip(('X','Y','Z'),to_ue(root))),NewRotation={'Roll':0.,'Pitch':0.,'Yaw':0.},bSweep=False,bTeleport=True);component.SetPosition(InPos=t,bFireNotifies=False)
   frame(update);frame()
   with instance.begin_frame():
    raw={name:component.GetSocketTransform(InSocketName=imp['bone_mapping'][name],TransformSpace='RTS_World',as_dict=True) for name in joint_names};mouth=np.asarray(from_ue(read_scene_component_pose(emitter)['location_cm']));actual_time=float(component.GetPosition());actual_root=np.asarray(from_ue(xyz(actor.K2_GetActorLocation(as_dict=True))))
   with instance.end_frame():pass
   if i==0:(a.output/'native_transform_sample.json').write_text(json.dumps(raw,indent=2))
   g={j:np.eye(4) for j in range(len(d['nodes']))}
   for name,value in raw.items():g[names[name]]=Ai@matrix(value)@A
   parts,_=pose(d,b,None,globals_override=g,ground=False,canonical_flip=False);vertices,faces=triangle_mesh(d,b,parts);distance=clearance.check(vertices,faces);near=vertices[vertices[:,1]<p0[1]+.08];skin_floor=float(np.min(near[:,1]-rays.many(near)))
   expected=globals_at(d,b,d['animations'][0],t);bone_error=max(float(np.linalg.norm(g[names[name]][:3,3]-(expected[names[name]][:3,3]+root))) for name in joint_names)
   head=g[names[m['head_bone']]][:3,3];tail=g[names[m['tail_bone']]][:3,3];expected_mouth=(g[names[m['mouth_bone']]]@np.r_[m['mouth_bone_local_m'],1.])[:3];mouth_error=float(np.linalg.norm(mouth-expected_mouth));feet={}
   for side,foot in source['feet'].items():
    pi,vi=foot['support_vertex'];support=parts[pi]['positions'][vi];center_sole=np.asarray(foot['sole_world_m'])+p0;fy=float(rays.many([center_sole])[0]);feye=center_sole.copy();feye[1]=fy-.22;fview={'position_m':feye.tolist(),'yaw_deg':0.,'pitch_deg':90.}
    frame(lambda:apply_camera(footcam,fview));frame()
    with instance.begin_frame():pass
    with instance.end_frame():depth=depth_reader(footcomponents['depth']).copy()
    hh,ww=depth.shape;yy,xx=np.indices(depth.shape);focal=ww/(2*math.tan(math.radians(85)/2));dd=depth/np.sqrt(1+((xx+.5-ww/2)/focal)**2+((yy+.5-hh/2)/focal)**2);px=feye[0]+dd*(yy+.5-hh/2)/focal;pz=feye[2]+dd*(xx+.5-ww/2)/focal;py=feye[1]+dd
    mask=np.isfinite(depth)&(depth>.05)&(depth<.9)&((px-center_sole[0])**2+(pz-center_sole[2])**2<.065**2)
    if not mask.any():raise ValueError('native rendered paw sole missing:'+side)
    iy,ix=np.unravel_index(np.flatnonzero(mask)[np.argmin(py[mask])],depth.shape);sole=np.array([px[iy,ix],py[iy,ix],pz[iy,ix]]);floor=float(rays.many([sole])[0]);feet[side]={**foot,'support_world_m':support.tolist(),'sole_world_m':sole.tolist(),'floor_delta_m':float(sole[1]-floor),'native_sole_pixels':int(mask.sum()),'native_depth_min_pixel':[int(ix),int(iy)]}
   row={**source,'frame_index':i,'root_world_m':actual_root.tolist(),'head_world_m':head.tolist(),'tail_world_m':tail.tolist(),'mouth_world_m':mouth.tolist(),'feet':feet,'body_floor_min_m':skin_floor,'minimum_nonfloor_clearance_m':distance,'navmesh_horizontal_error_m':float(np.linalg.norm((np.asarray(pf.snap_point(actual_root))-actual_root)[[0,2]])),'native_emitter_readback_error_m':mouth_error,'native_bone_position_max_error_m':bone_error,'native_animation_time_s':actual_time}
   if bone_error>.0002 or abs(actual_time-t)>.002:raise ValueError('native pose differs from metric export')
   with instance.begin_frame():pass
   with instance.end_frame():rgb=rgb_reader(components['rgb']).copy();scene_depth=depth_reader(components['depth']).copy()
   def isolate():
    components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_UseShowOnlyList');components['depth'].set_property_value(property_name='ShowOnlyActors',property_value=[actor])
   frame(isolate);frame()
   with instance.begin_frame():pass
   with instance.end_frame():animal_depth=depth_reader(components['depth']).copy()
   frame(lambda:components['depth'].set_property_value(property_name='PrimitiveRenderMode',property_value='PRM_RenderScenePrimitives'))
   mask=np.isfinite(animal_depth)&(animal_depth>.03)&(animal_depth<30)&(np.abs(scene_depth-animal_depth)<.001);yy,xx=np.where(mask);row['visible_pixels']=int(mask.sum());row['complete_silhouette_in_frame']=bool(len(xx)>100 and min(xx.min(),yy.min(),639-xx.max(),479-yy.max())>=3)
   if i in selected:
    fwd=delta/np.linalg.norm(delta);right=np.cross(fwd,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,fwd);image=a.output/f'frame_{i:04d}.png';overlay(rgb[...,::-1],eye,(fwd,right,up),head,tail,actual_root+[0,.02,0],np.array([0,0,-.08])).save(image);result['images'].append(str(image))
   result['frames'].append(row)
   if i%15==0:(a.output/'progress.json').write_text(json.dumps(result,indent=2));print('UE_FRAME',i,bone_error,mouth_error,min(f['floor_delta_m'] for f in feet.values()),flush=True)
  checks=frame_checks(result['frames'],native=True)
  if not all(f['complete_silhouette_in_frame'] for f in result['frames']):checks['status']='fail';checks['failures'].append('native_whole_animal_clipped_or_invisible')
  result.update(status='geometry_pass_audio_pending' if checks['status']=='pass' else 'geometry_fail',checks=checks,direction_negative_controls=direction_negative_controls(result['frames']),listener_m=eye.tolist(),camera_forward=(delta/np.linalg.norm(delta)).tolist(),frame_count=len(rows),full_original_skin_reconstructed_from_actual_native_bones=True,native_soles_measured_by_UE_depth=True)
  if a.limit:result['status']='native_smoke_measured_not_admitted'
 except Exception as e:result.update(status='failed',error=repr(e),traceback=traceback.format_exc())
 finally:
  if footcam is not None:
   for name in ('object_ids','depth','rgb'):
    try:close_scene_capture(instance=instance,game=game,camera=None,capture=footcomponents[name])
    except Exception:pass
   try:frame(lambda:footcam.K2_DestroyActor())
   except Exception:pass
  if actor is not None:
   try:frame(lambda:actor.K2_DestroyActor())
   except Exception:pass
  close_native(instance,cam,components);(a.output/'native_motion.json').write_text(json.dumps(result,indent=2))
 print(result['status'],result.get('error'),flush=True)
 if result['status'] not in ('geometry_pass_audio_pending','native_smoke_measured_not_admitted'):raise SystemExit(2)
if __name__=='__main__':main()
