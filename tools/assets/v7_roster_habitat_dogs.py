"""Replay full rigid-pose clips with native Habitat stage physics and FOA nodes."""
import argparse,copy,io,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,rigid_glb,viewbytes
from v7_human_roster_habitat import make_object,camera,project
from v7_roster_contact_ik import yaw_matrix
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
def externalize(d,b,folder):
 d=copy.deepcopy(d);folder.mkdir(parents=True,exist_ok=True)
 for i,im in enumerate(d.get('images',[])):
  if 'bufferView'not in im:continue
  payload=viewbytes(d,b,im['bufferView']);dest=folder/('texture_'+str(i)+'.png')
  if not dest.exists():Image.open(io.BytesIO(payload)).resize((1024,1024),Image.Resampling.LANCZOS).save(dest)
  im.clear();im['uri']='../textures/'+dest.name
 return d
def stage_floor(sim,r,world,ignore=None):
 hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(world+[0,.15,0]),r.magnum.Vector3(0,-1,0)),.6).hits;hits=[h for h in hits if h.object_id!=ignore]
 return float(hits[0].point[1])if hits else None
def run(a):
 T=a.task_root;R=T/'round2';rows=[{'asset_id':breed+'_'+family,'breed':breed,'family':family}for breed in ['beagle','german_shepherd']for family in ['Idle','Walking']]
 if a.only:rows=[x for x in rows if x['asset_id']==a.only]
 views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());out=[]
 for house in ['mp3d','hm3d']:
  if a.house and a.house!=house:continue
  v=next(x for x in views if x['house']==house);path=json.loads((R/'registrations'/('path_'+house+'.json')).read_text())
  for row in rows:
   aid=row['asset_id'];D=R/'assets/dogs'/row['breed'];d,b=read_glb(D/'motion_runtime.glb');action=next(x for x in d['animations']if x['name']==row['family']+'_Contact_'+house);planned=json.loads((D/(house+'_'+row['family']+'.json')).read_text());path={'points_m':planned['path_points_m'],'yaw_rad':planned['yaw_rad'],'time_s':[i/30 for i in range(planned['frame_count'])]};dest=T/'round3/assets/native_replay'/house/aid;folder=dest/'frames';folder.mkdir(parents=True,exist_ok=True);ed=externalize(d,b,dest/'textures');receipt=T/'round3/evidence/native_habitat_dogs'/house/(aid+'.json')
   if receipt.exists():
    cached=json.loads(receipt.read_text())
    if cached.get('frame_count')==len(path['time_s']):out.append({k:v for k,v in cached.items()if k!='frames'});continue
   from v7_roster_disk import check
   check(T,32*1024**2)
   r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));config,uuids,_,scene=_make_configuration(inputs,None,T/'round3/images/habitat_dogs'/house/aid,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);config.sim_cfg.enable_physics=True;config.sim_cfg.gpu_device_id=0
   for ag in config.agents:
    for sensor in ag.sensor_specifications:sensor.resolution=[640,640]
   med=r.habitat_sim.metadata.MetadataMediator(config.sim_cfg);m=med.pbr_shader_template_manager;pb=m.create_template('v7_human_direct',register_template=False);pb.enable_ibl=False;pb.enable_direct_lights=True;pb.map_mat_txtr_to_linear=True;pb.map_output_to_srgb=True;m.register_template(pb,'v7_human_direct');med.set_curr_default_pbr_attributes_handle('v7_human_direct');config.metadata_mediator=med
   rec={'asset_id':aid,'house':house,'source_clip':action['name'],'source_geometric_status':planned['geometric_status'],'frames':[],'images':[],'scene':str(scene),'status':'fail'}
   with r.habitat_sim.Simulator(config)as sim:
    sim.pathfinder.load_nav_mesh(str(R/'navmesh'/(house+'.navmesh')))
    for fi,t in enumerate(path['time_s']):
     ps,ev=pose(d,b,action,float(t),ground=False,canonical_flip=False,emitter_bone='Xtra Mouth');fpath=folder/(str(fi).zfill(3)+'.glb');rigid_glb(ed,b,ps,fpath);root=np.array(path['points_m'][fi]);yaw=path['yaw_rad'][fi];rot=yaw_matrix(yaw);obj=make_object(sim,r,fpath,root,yaw);emitter=obj.root_scene_node.create_child();emitter.translation=r.magnum.Vector3(ev['mouth_m']);native=np.asarray(emitter.absolute_translation);expected=root+rot@np.asarray(ev['mouth_m']);eerror=float(np.linalg.norm(native-expected));foots=[]
     for side,ft in planned['frames'][fi]['feet'].items():
      point=np.array(ft.get('support_world_m',ft['ankle_world_m']));fy=stage_floor(sim,r,point,obj.object_id);foots.append({'side':side,'in_contact':ft['in_contact'],'floor_y_m':fy,'support_delta_m':None if fy is None or not ft['in_contact']else float(point[1]-fy)})
     obj.translation=r.magnum.Vector3(root+[0,.04,0]);collision=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root)
     snap=np.asarray(sim.pathfinder.snap_point(root));naverror=float(np.linalg.norm((snap-root)[[0,2]]));frame={'frame':fi,'time_s':float(t),'native_emitter_world_m':native.tolist(),'emitter_readback_error_m':eerror,'stage_contacts':foots,'raised_body_collision_4cm':collision,'navigation_horizontal_error_m':naverror};rec['frames'].append(frame)
     if fi in [0,len(path['time_s'])//3,2*len(path['time_s'])//3,len(path['time_s'])-1]:
      eye=root+rot@np.array([0,.85,-2.1]);sim.initialize_agent(0,camera(r,eye,root+[0,.42,0]));obs=sim.get_sensor_observations();rgb=np.asarray(obs[uuids['rgb']])[...,:3];mask=np.asarray(obs[uuids['semantic']])==211;im=Image.fromarray(rgb);dr=ImageDraw.Draw(im);x,y=project(r,sim,uuids['rgb'],native);dr.ellipse((x-3,y-3,x+3,y+3),outline=(0,255,0),width=2);ip=T/'round3/images/habitat_dogs'/house/aid/(str(fi).zfill(3)+'.jpg');ip.parent.mkdir(parents=True,exist_ok=True);im.save(ip,quality=85);rec['images'].append({'frame':fi,'path':str(ip),'visible_pixels':int(mask.sum()),'highlight_ratio_rgb_ge_245':float(np.mean(np.all(rgb[mask]>=245,axis=1)))if mask.any()else None,'mouth_pixel':[x,y]})
     sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
     if fi not in [0,len(path['time_s'])//3,2*len(path['time_s'])//3,len(path['time_s'])-1]:fpath.unlink()
    # Old vertex-only sitting prototypes are not round3 seat admissions.
   vals=[abs(f['support_delta_m'])for e in rec['frames']for f in e['stage_contacts']if f['in_contact']and f['support_delta_m']is not None];missing=sum(f['in_contact']and f['floor_y_m']is None for e in rec['frames']for f in e['stage_contacts']);rec.update(maximum_native_support_delta_m=max(vals)if vals else None,missing_stage_contacts=int(missing),body_collision_frames=sum(x['raised_body_collision_4cm']for x in rec['frames']),maximum_emitter_readback_error_m=max(x['emitter_readback_error_m']for x in rec['frames']),maximum_navmesh_horizontal_error_m=max(x['navigation_horizontal_error_m']for x in rec['frames']),frame_count=len(rec['frames']),status='pass'if planned['geometric_status']=='pass'and vals and max(vals)<.012 and not missing and not any(x['raised_body_collision_4cm']for x in rec['frames'])and all(x['emitter_readback_error_m']<.0001 and x['navigation_horizontal_error_m']<.002 for x in rec['frames'])else'fail',collision_method='native rigid-body contact_test at +4cm to exclude intended ground contact; NavMesh clearance separately checked',mouth_status='full replay FOA SceneGraph child readback; native skeletal bones not claimed')
   json_write(receipt,rec);out.append({k:v for k,v in rec.items()if k not in ['frames']});print('HABITAT_MOTION',house,aid,rec['status'],rec['maximum_native_support_delta_m'],rec['body_collision_frames'],flush=True)
 json_write(T/'round3/evidence'/('habitat_dog_motion_'+(a.only or 'all')+'.json'),out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--only');p.add_argument('--house');run(p.parse_args())
