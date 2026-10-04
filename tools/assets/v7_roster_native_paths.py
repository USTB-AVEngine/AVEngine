"""Build native Recast paths in the real MP3D/HM3D rooms."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from v7_human_roster_materials import json_write
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
def sample_curve(sim,r,start,end,count=121,amp=.12):
 a=np.array(start);b=np.array(end);d=b-a;d[1]=0;length=np.linalg.norm(d);f=d/length;right=np.array([f[2],0,-f[0]]);u=np.linspace(0,1,count)
 raw=a[None,:]+u[:,None]*d+np.sin(u*math.pi)[:,None]*right*amp
 snaps=np.array([sim.pathfinder.snap_point(p)for p in raw])
 if not np.isfinite(snaps).all()or np.max(np.linalg.norm((raw-snaps)[:,[0,2]],axis=1))>.002:return None
 clearance=np.array([sim.pathfinder.distance_to_closest_obstacle(p,.8)for p in snaps])
 if clearance.min()<.025:return None
 floors=[]
 for p in snaps:
  hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(p+[0,.15,0]),r.magnum.Vector3(0,-1,0)),.6).hits
  if not hits:return None
  floors.append(float(hits[0].point[1]))
 if np.ptp(floors)>.08:return None
 raw[:,1]=floors
 lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(raw,axis=0),axis=1))];duration=lengths[-1]/1.2;n=int(math.ceil(duration*30));dist=np.linspace(0,lengths[-1],n+1);points=np.column_stack([np.interp(dist,lengths,raw[:,i])for i in range(3)]);navpoints=np.array([sim.pathfinder.snap_point(p)for p in points]);ts=np.linspace(0,duration,n+1);v=np.gradient(points,ts,axis=0);yaw=np.unwrap(np.arctan2(-v[:,0],-v[:,2]));yawrate=np.gradient(yaw,ts)
 return {'points_m':points.tolist(),'navmesh_points_m':navpoints.tolist(),'time_s':ts.tolist(),'yaw_rad':yaw.tolist(),'speed_m_s':np.linalg.norm(v[:,[0,2]],axis=1).tolist(),'max_turn_speed_deg_s':float(np.max(np.abs(yawrate))*180/math.pi),'duration_s':duration,'length_m':float(lengths[-1]),'minimum_navmesh_boundary_clearance_m':float(clearance.min()),'physical_floor_range_m':float(np.ptp(floors)),'status':'pass'}
def main(a):
 T=a.task_root;R=T/'round2';out=[]
 views=json.loads(Path('/data/jzy/tmp/claude_hardbank_20260923/asset_materials_20261003/evidence/table_views_v3.json').read_text())
 for v in views:
  r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=v['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg')
  inputs=load_and_validate_inputs(Path(v['room_manifest']),Path(v['capture_request']));cfg,_,_,scene=_make_configuration(inputs,None,R/'navmesh'/v['house'],mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path)
  cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.create_renderer=False;cfg.sim_cfg.gpu_device_id=0
  with r.habitat_sim.Simulator(cfg)as sim:
   ns=r.habitat_sim.NavMeshSettings();ns.set_defaults();ns.agent_height=2.05;ns.agent_radius=.46;ns.agent_max_climb=.12;ns.agent_max_slope=10
   assert sim.recompute_navmesh(sim.pathfinder,ns),'Recast failed';sim.pathfinder.seed(20261004)
   result=None
   for i in range(4000):
    p=sim.pathfinder.get_random_navigable_point();q=sim.pathfinder.get_random_navigable_point()
    if not 2.7<=np.linalg.norm((np.asarray(q)-np.asarray(p))[[0,2]])<=5:continue
    if abs(p[1]-q[1])>.08:continue
    result=sample_curve(sim,r,p,q)
    if result:break
   if result is None:raise RuntimeError('No cleared curved 3.6m path in '+v['house'])
   nav=R/'navmesh'/(v['house']+'.navmesh');nav.parent.mkdir(parents=True,exist_ok=True);sim.pathfinder.save_nav_mesh(str(nav))
   rec={**result,'house':v['house'],'scene':str(scene),'view_input':v,'agent_height_m':ns.agent_height,'agent_radius_m':ns.agent_radius,'navmesh':str(nav),'candidate_attempts':i+1,'coordinate_system':'+Y up, metre; navmesh Y separately retained from physical-floor ray Y'}
   json_write(R/'registrations'/('path_'+v['house']+'.json'),rec);out.append(rec);print('NATIVE_PATH',v['house'],result['duration_s'],result['max_turn_speed_deg_s'],flush=True)
 json_write(R/'evidence/native_paths.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);main(p.parse_args())
