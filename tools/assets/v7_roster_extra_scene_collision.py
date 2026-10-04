"""Exact source-room triangle checks for stationary actions, chairs and dog motions."""
import argparse,json,sys,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,accessor
from v7_roster_contact_ik import yaw_matrix
from v7_roster_scene_collision import model
from v7_roster_disk import check
ENVS={}
def initialize(T):
 global ENVS
 T=Path(T);sys.path.insert(0,str(T/'round3/pydeps'));import fcl
 tri=np.load(T/'evidence/ue_source_triangles.npz')['triangles'].astype(float)/100;path=json.loads((T/'round2/registrations/path_kujiale_v3.json').read_text());roots=np.asarray(path['points_m']);upr=np.stack([-roots[:,2],roots[:,0],roots[:,1]],axis=1);p=json.loads((T/'evidence/ue_source_floor_plan.json').read_text())['selected'];center=np.asarray(p['location_cm'])/100
 for label,lo,hi,fy in [('route',upr.min(0)-[1.05,1.05,.1],upr.max(0)+[1.05,1.05,2.2],float(np.median(upr[:,2]))),('stationary',center-[1.6,1.6,.1],center+[1.6,1.6,2.2],center[2])]:
  keep=np.all(tri.max(1)>=lo,1)&np.all(tri.min(1)<=hi,1);near=tri[keep];nn=np.cross(near[:,1]-near[:,0],near[:,2]-near[:,0]);nz=abs(nn[:,2])/np.maximum(np.linalg.norm(nn,axis=1),1e-9);floor=(nz>.98)&(near[:,:,2].max(1)<fy+.02)&(near[:,:,2].min(1)>fy-.02);near=near[~floor];ENVS[label]=(model(near.reshape(-1,3),np.arange(near.size//3).reshape(-1,3)),len(near),int(floor.sum()))
 ENVS['center']=center;ENVS['path']=path
def body(d,b,ps,root,yaw):
 rot=yaw_matrix(yaw);vs=[];ix=[];offset=0
 for p in ps:
  v=p['positions']@rot.T+root;vs.append(np.stack([-v[:,2],v[:,0],v[:,1]],axis=1));pr=d['meshes'][d['nodes'][p['node']]['mesh']]['primitives'][p['primitive']];ix.append(accessor(d,b,pr['indices']).astype(int).reshape(-1,3)+offset);offset+=len(v)
 return model(np.concatenate(vs),np.concatenate(ix))
def collide(obj,env):
 import fcl
 result=fcl.CollisionResult();n=fcl.collide(obj,env,fcl.CollisionRequest(num_max_contacts=1),result);dist=fcl.DistanceResult();v=float(fcl.distance(obj,env,fcl.DistanceRequest(),dist));return bool(n),v
def one(args):
 T,kind,aid,family=args;T=Path(T);frames=[];center=ENVS['center'];cr=np.asarray([center[1],center[2],-center[0]]);env,count,removed=ENVS['route'if kind=='dog'else'stationary']
 if kind=='dog':
  D=T/'round2/assets/dogs'/aid;d,b=read_glb(D/'motion_runtime.glb');plan=json.loads((D/('kujiale_'+family+'.json')).read_text());clip=next(a for a in d['animations']if a['name']==plan['clip']);roots=plan['path_points_m'];yaws=plan['yaw_rad'];times=np.arange(plan['frame_count'])/30
 elif kind=='human':
  D=T/'round2/assets/humans'/aid;d,b=read_glb(D/'final_ue_root.glb');plan=json.loads((T/'round3/evidence/extra_actions'/aid/(family+'.json')).read_text());clip=next(a for a in d['animations']if a['name']==plan['clip']);offset=np.asarray(plan.get('chair_fit',{}).get('human_local_offset_m',[0,0,0]));roots=[cr+offset]*plan['frame_count'];yaws=np.zeros(plan['frame_count']);times=np.arange(plan['frame_count'])/30
 else:
  d,b=read_glb(T/'round3/assets/chairs'/(aid+'.glb'));plan={'geometric_status':'pass'};clip=None;roots=[cr];yaws=[0];times=[0]
 for i,t in enumerate(times):
  ps,e=pose(d,b,clip,float(t),ground=False,canonical_flip=kind=='human',emitter_bone='Xtra Mouth'if kind=='dog'else None);c,dist=collide(body(d,b,ps,np.asarray(roots[i]),float(yaws[i])),env);frames.append({'frame':i,'time_s':float(t),'collision':c,'minimum_non_floor_distance_m':dist})
 rec={'kind':kind,'asset_id':aid,'family':family,'status':'pass'if not any(f['collision']for f in frames)and min(f['minimum_non_floor_distance_m']for f in frames)>.02 else'fail','frames':frames,'collision_frames':sum(f['collision']for f in frames),'minimum_non_floor_distance_m':min(f['minimum_non_floor_distance_m']for f in frames),'scene_triangles':count,'ground_triangles_removed':removed,'source_scene_npz':str(T/'evidence/ue_source_triangles.npz'),'native_ue_physics':'not_claimed','method':'exact source-USD and full posed mesh BVHs at true coordinates every 30Hz key, near-horizontal true ground +/-2cm excluded'}
 dest=T/'round3/evidence/extra_scene_collision'/kind/aid/(family+'.json');json_write(dest,rec);print('EXTRA_SCENE',kind,aid,family,rec['status'],rec['minimum_non_floor_distance_m'],flush=True);return {k:v for k,v in rec.items()if k!='frames'}
def run(T):
 check(T,32*1024**2);rows=json.loads((T/'registrations/selection.json').read_text())['avatars'];jobs=[(str(T),'human',r['asset_id'],f)for r in rows for f in ['Standing_Talk','Turning_Left90','Sitting_Talk']]+[(str(T),'dog',b,f)for b in ['beagle','german_shepherd']for f in ['Idle','Walking']]+[(str(T),'chair',p.stem,'Static')for p in(T/'round3/assets/chairs').glob('*.glb')]
 with ProcessPoolExecutor(max_workers=3,initializer=initialize,initargs=(str(T),))as pool:out=list(pool.map(one,jobs))
 json_write(T/'round3/evidence/extra_scene_collision_results.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
