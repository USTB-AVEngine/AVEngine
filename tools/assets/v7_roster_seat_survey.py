"""Survey native semantic seats and review silent fallback chairs in real homes."""
import argparse,json,math,copy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,rigid_glb
from v7_human_roster_habitat import make_object,camera,project
from v7_roster_contact_ik import FootPlant,yaw_matrix
from v7_roster_disk import check
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
SEATS={'chair','sofa','couch','bench','stool','armchair','seat','ottoman'}
def floor(sim,r,p,base,ignore=()):
 hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3([p[0],base+.20,p[2]]),r.magnum.Vector3(0,-1,0)),.65).hits
 hits=[h for h in hits if h.object_id not in ignore]
 if not hits:return float('nan')
 return float(hits[0].point[1])
def run(T):
 R=T/'round3';chairs=json.loads((R/'registrations/chair_roster.json').read_text())['chairs'];views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());out=[];placements=[]
 for v in views:
  house=v['house'];check(T,64*1024**2);r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inp=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,uuids,_,scene=_make_configuration(inp,None,R/'images/habitat_chairs'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
  for a in cfg.agents:
   for sensor in a.sensor_specifications:sensor.resolution=[640,640]
  med=r.habitat_sim.metadata.MetadataMediator(cfg.sim_cfg);m=med.pbr_shader_template_manager;pb=m.create_template('v7_chair_direct',register_template=False);pb.enable_ibl=False;pb.enable_direct_lights=True;pb.map_mat_txtr_to_linear=True;pb.map_output_to_srgb=True;m.register_template(pb,'v7_chair_direct');med.set_curr_default_pbr_attributes_handle('v7_chair_direct');cfg.metadata_mediator=med
  with r.habitat_sim.Simulator(cfg)as sim:
   path=json.loads((T/'round2/registrations'/('path_'+house+'.json')).read_text());route=np.asarray(path['points_m']);sim.pathfinder.load_nav_mesh(str(T/'round2/navmesh'/(house+'.navmesh')));sim.pathfinder.seed(20261004);objects=sim.semantic_scene.objects or [];seatrows=[];targets=[]
   for o in objects:
    if o is None or o.category is None:continue
    category=o.category.name();bb=o.obb.to_aabb();center=np.asarray(bb.center());size=np.asarray(bb.size())
    if category in ['table','tv','television']:targets.append({'category':category,'center_m':center.tolist()})
    if category not in SEATS:continue
    low=center-size/2;samples=[]
    for dx in [-.10,0,.10]:
     for dz in [-.10,0,.10]:
      p=np.array([center[0]+dx,low[1]+.66,center[2]+dz]);hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(p),r.magnum.Vector3(0,-1,0)),.50).hits
      if hits and low[1]+.25<float(hits[0].point[1])<low[1]+.62:samples.append(list(hits[0].point))
    sr={'house':house,'semantic_id':str(o.id),'category':category,'bbox_center_m':center.tolist(),'bbox_size_m':size.tolist(),'region_id':str(o.region.id)if o.region else None,'seat_surface_samples_m':samples,'source':'native semantic object OBB converted to AABB and native stage downward rays; no scene mesh modified','status':'surface_candidate'if len(samples)>=5 and np.ptp(np.asarray(samples)[:,1])<.04 else'needs_surface_review'}
    if samples:sr['seat_surface_center_m']=np.median(samples,axis=0).tolist()
    seatrows.append(sr)
   out.append({'house':house,'scene':str(scene),'native_semantic_object_count':len(objects),'seat_count':len(seatrows),'seats':seatrows,'facing_targets':targets,'preference':'existing labelled seat before fallback chair; candidates require body/foot/scene fit receipt'})
   json_write(R/'evidence'/('existing_seats_'+house+'.json'),out[-1])
   # Each fallback is reviewed independently; main reference walking corridor is protected.
   for ch in chairs:
    cd,cb=read_glb(Path(ch['habitat_glb']));p,e=pose(cd,cb,canonical_flip=False,ground=False);rad=max(np.linalg.norm(x['positions'][:,[0,2]],axis=1).max()for x in p);placed=None;actor=None
    for attempt in range(500):
     snap=np.asarray(sim.pathfinder.get_random_navigable_point());root=snap.copy();root[1]=floor(sim,r,root,root[1])
     if not np.isfinite(root).all():continue
     corridor=float(np.linalg.norm(route[:,[0,2]]-root[[0,2]],axis=1).min())
     if corridor<rad+.65+.08:continue
     for yaw in [0,math.pi/2,math.pi,-math.pi/2]:
      actor=make_object(sim,r,Path(ch['habitat_glb']),root,yaw,semantic=6100002);actor.translation=r.magnum.Vector3(root+[0,.05,0]);bad=bool(actor.contact_test());actor.translation=r.magnum.Vector3(root);rot=yaw_matrix(yaw);ds=[]
      for pt in ch['support_points_m']:
       wp=root+rot@np.asarray(pt);fy=floor(sim,r,wp,root[1],(actor.object_id,));ds.append(float(wp[1]-fy))
      if not bad and all(np.isfinite(x)and abs(x)<.012 for x in ds):placed={'asset_id':ch['asset_id'],'house':house,'root_m':root.tolist(),'yaw_rad':yaw,'native_raised_5cm_collision':False,'support_floor_deltas_m':ds,'minimum_reference_route_distance_m':corridor,'protected_route_clearance_m':corridor-rad-.65,'source_navmesh':path['navmesh'],'status':'pass_candidate','produces_sound':False};break
      sim.get_rigid_object_manager().remove_object_by_id(actor.object_id);actor=None
     if placed:break
    if not placed:placements.append({'asset_id':ch['asset_id'],'house':house,'status':'no_safe_candidate'});continue
    eye=root+yaw_matrix(yaw)@np.array([0,1.05,-2.6]);sim.initialize_agent(0,camera(r,eye,root+[0,.55,0]));obs=sim.get_sensor_observations();rgb=np.asarray(obs[uuids['rgb']])[...,:3];mask=np.asarray(obs[uuids['semantic']])==6100002;ip=R/'images/habitat_chairs'/house/(ch['asset_id']+'.jpg');ip.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(ip,quality=85);placed.update(image=str(ip),visible_pixels=int(mask.sum()),highlight_ratio_rgb_ge_245=float(np.mean(np.all(rgb[mask]>=245,axis=1)))if mask.any()else None)
    sim.get_rigid_object_manager().remove_object_by_id(actor.object_id);placements.append(placed);print('CHAIR_NATIVE',house,ch['asset_id'],placed['status'],placed['visible_pixels'],flush=True)
 json_write(R/'evidence/existing_seats_native.json',out);json_write(R/'evidence/chair_native_placements.json',placements)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
