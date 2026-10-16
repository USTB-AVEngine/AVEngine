"""Full-frame scene triangle collision checks against the exact Kujiale USD mesh."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,accessor
from v7_roster_contact_ik import yaw_matrix
from v7_roster_disk import check
ENV=None
def model(vertices,indices):
 import fcl

 if not len(vertices)or not len(indices):raise ValueError('Empty collision BVH; scene/path coordinate check failed')
 m=fcl.BVHModel();m.beginModel(len(vertices),len(indices));m.addSubModel(np.asarray(vertices,float),np.asarray(indices,np.int32));m.endModel();return fcl.CollisionObject(m)
def initialize(T):
 global ENV
 sys.path.insert(0,str(Path(T)/'round3/pydeps'));import fcl
 T=Path(T);path=json.loads((T/'round2/registrations/path_kujiale_v3.json').read_text());roots=np.asarray(path['points_m']);up=np.stack([-roots[:,2],roots[:,0],roots[:,1]],axis=1);tri=np.load(T/'evidence/ue_source_triangles.npz')['triangles'].astype(float)/100
 lo=up.min(0)-[1.05,1.05,.1];hi=up.max(0)+[1.05,1.05,2.2];keep=np.all(tri.max(1)>=lo,1)&np.all(tri.min(1)<=hi,1);tri=tri[keep]
 normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);nz=abs(normal[:,2])/np.maximum(np.linalg.norm(normal,axis=1),1e-9);floor=(nz>.98)&(tri[:,:,2].max(1)<np.median(up[:,2])+.02)&(tri[:,:,2].min(1)>np.median(up[:,2])-.02);removed=int(floor.sum());tri=tri[~floor]
 ENV=(model(tri.reshape(-1,3),np.arange(tri.size//3).reshape(-1,3)),len(tri),removed,path)
def one(arg):
 import fcl
 T,aid=arg;T=Path(T);env,count,removed,path=ENV;D=T/'round2/assets/humans'/aid;d,b=read_glb(D/'final_ue_root.glb');clip=next(a for a in d['animations']if a['name']=='Walking_Contact_kujiale');plan=json.loads((D/'walking_kujiale.json').read_text());out=[];path=json.loads(Path(plan.get('path_manifest')or T/'round2/registrations/path_kujiale_v3.json').read_text())
 for i,t in enumerate(path['time_s']):
  parts,e=pose(d,b,clip,float(t),ground=False);root=np.asarray(path['points_m'][i]);rot=yaw_matrix(path['yaw_rad'][i]);verts=[];idx=[];offset=0
  for p in parts:
   v=p['positions']@rot.T+root;verts.append(np.stack([-v[:,2],v[:,0],v[:,1]],axis=1));pr=d['meshes'][d['nodes'][p['node']]['mesh']]['primitives'][p['primitive']];idx.append(accessor(d,b,pr['indices']).astype(int).reshape(-1,3)+offset);offset+=len(v)
  body=model(np.concatenate(verts),np.concatenate(idx));cr=fcl.CollisionResult();n=fcl.collide(body,env,fcl.CollisionRequest(num_max_contacts=1),cr);dr=fcl.DistanceResult();distance=float(fcl.distance(body,env,fcl.DistanceRequest(),dr))
  out.append({'frame':i,'time_s':t,'scene_collision':bool(n),'minimum_non_floor_distance_m':distance})
 rec={'asset_id':aid,'status':'pass'if not any(r['scene_collision']for r in out)and min(r['minimum_non_floor_distance_m']for r in out)>.02 else'fail','collision_frames':sum(r['scene_collision']for r in out),'minimum_non_floor_distance_m':min(r['minimum_non_floor_distance_m']for r in out),'frames':out,'frame_count':len(out),'scene_triangles_in_path_volume':count,'expected_floor_triangles_removed':removed,'method':'FCL exact triangle BVHs, full posed mesh at every 30Hz key; remove near-horizontal true ground within 2cm of measured route floor, no body lift','source_usd':path['source_usd'],'source_scene_npz':str(T/'evidence/ue_source_triangles.npz'),'native_ue_physics':'not_claimed; imported scene source bounds were separately verified'}
 json_write(T/'round3/evidence/scene_collision'/(aid+'.json'),rec);print('SCENE_MOTION',aid,rec['status'],rec['collision_frames'],rec['minimum_non_floor_distance_m'],flush=True);return {k:v for k,v in rec.items()if k!='frames'}
def run(T):
 check(T,32*1024**2);rows=json.loads((T/'registrations/selection.json').read_text())['avatars']
 with ProcessPoolExecutor(max_workers=3,initializer=initialize,initargs=(str(T),))as pool:out=list(pool.map(one,[(str(T),r['asset_id'])for r in rows]))
 json_write(T/'round3/evidence/scene_motion_collision_results.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
