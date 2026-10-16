"""Recover existing seats from actual native RGB-D-semantic surfaces, not faulty bounds."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from v7_human_roster_materials import json_write
from v7_human_roster_habitat import camera
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
SEATS={'chair','sofa','couch','bench','stool','armchair','seat','ottoman'}
def point(r,rc,x,y,depth,w=640):
 nd=np.array([2*(x+.5)/w-1,1-2*(y+.5)/w,-1,1]);p=np.linalg.inv(np.asarray(rc.projection_matrix))@nd;p/=p[3];p[:3]*=-depth/p[2];p[3]=1;return (np.linalg.inv(np.asarray(rc.camera_matrix))@p)[:3]

def run(T):
 R=T/'round3';views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text());out=[]
 for v in views:
  house=v['house'];r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg');inp=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,uuids,_,scene=_make_configuration(inp,None,R/'images/seat_scan'/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
  for sn in cfg.agents[0].sensor_specifications:sn.resolution=[640,640];sn.hfov=100;sn.position=r.magnum.Vector3(0,0,0)
  dep=r.habitat_sim.CameraSensorSpec();dep.uuid='seat_scan_depth';dep.sensor_type=r.habitat_sim.SensorType.DEPTH;dep.resolution=[640,640];dep.hfov=100;dep.position=r.magnum.Vector3(0,0,0);cfg.agents[0].sensor_specifications.append(dep)
  seats=[];targets=[];seen=set()
  with r.habitat_sim.Simulator(cfg)as sim:
   path=json.loads((T/'round2/registrations'/('path_'+house+'.json')).read_text());sim.pathfinder.load_nav_mesh(path['navmesh']);sim.pathfinder.seed(20261004);objects=sim.semantic_scene.objects or [];labels={i:o.category.name()for i,o in enumerate(objects)if o is not None and o.category is not None}
   for frame in range(100):
    p=np.asarray(sim.pathfinder.get_random_navigable_point());hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(p+[0,.2,0]),r.magnum.Vector3(0,-1,0)),.7).hits
    if not hits:continue
    p[1]=hits[0].point[1];eye=p+[0,1.15,0];sim.initialize_agent(0,camera(r,eye,p+[.001,0,0]));obs=sim.get_sensor_observations();sem=np.asarray(obs[uuids['semantic']]);depth=np.asarray(obs['seat_scan_depth']);rc=sim._sensors['seat_scan_depth']._sensor_object.render_camera
    for sid in np.unique(sem):
     cat=labels.get(int(sid))
     if cat not in SEATS and cat not in {'table','tv','television'}:continue
     yy,xx=np.where(sem==sid)
     if len(xx)<25:continue
     samples=[]
     for j in np.linspace(0,len(xx)-1,min(len(xx),80),dtype=int):
      x,y=int(xx[j]),int(yy[j]);z=float(depth[y,x])
      if not 0<z<3:continue
      wp=point(r,rc,x,y,z)
      if cat in SEATS:
       if not .25<wp[1]-p[1]<.65 or x<2 or y<2 or x>637 or y>637:continue
       a=point(r,rc,x+2,y,float(depth[y,x+2]));b=point(r,rc,x,y+2,float(depth[y+2,x]));normal=np.cross(a-wp,b-wp);length=np.linalg.norm(normal)
       if length<1e-8 or abs(normal[1])/length<.85:continue
      samples.append(wp)
     if len(samples)<5:continue
     pts=np.asarray(samples);center=np.median(pts,axis=0);key=(int(sid),round(center[0]*2),round(center[2]*2))
     if key in seen:continue
     seen.add(key)
     if cat in SEATS:
      if np.ptp(pts[:,1])>.06:continue
      seats.append({'house':house,'semantic_id':str(sid),'category':cat,'seat_surface_center_m':center.tolist(),'seat_surface_samples_m':pts.tolist(),'measured_floor_y_m':float(p[1]),'seat_height_above_floor_m':float(center[1]-p[1]),'status':'surface_candidate','source':'native RGB-D-semantic mask, depth unprojection and horizontal surface normals','bounds_used':False,'scan_frame':frame})
     else:targets.append({'category':cat,'semantic_id':str(sid),'center_m':center.tolist(),'source':'native depth surface samples'})
   rec={'house':house,'scene':str(scene),'native_semantic_object_count':len(objects),'labelled_seat_object_count':sum(c in SEATS for c in labels.values()),'seat_count':len(seats),'seats':seats,'facing_targets':targets,'scan_frames':100,'preference':'existing measured semantic seat first; source object AABB unreliable and not used'}
   out.append(rec);json_write(R/'evidence'/('existing_seats_rgbd_'+house+'.json'),rec);print('RGBD_SEATS',house,len(seats),flush=True)
 json_write(R/'evidence/existing_seats_native.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
