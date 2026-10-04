"""Check licensed parrot geometry, true source-floor support and UE camera rays."""
import argparse,json,sys,math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,accessor
from v7_roster_ue_camera_plan import visible
from v7_roster_scene_collision import model
from v7_roster_disk import check
def downward_hit(p,tri):
 lo,hi=tri.min(1),tri.max(1);keep=(lo[:,0]<=p[0])&(hi[:,0]>=p[0])&(lo[:,2]<=p[2])&(hi[:,2]>=p[2]);tri=tri[keep]
 if not len(tri):return None
 o=p+[0,.15,0];v=np.array([0.,-1,0]);e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];h=np.cross(np.broadcast_to(v,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h);ok=abs(det)>1e-9;inv=np.where(ok,1/np.where(ok,det,1),0);s=o-tri[:,0];u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1);w=q@v*inv;dist=np.einsum('ij,ij->i',e2,q)*inv;valid=ok&(u>=-1e-7)&(w>=-1e-7)&(u+w<=1+1e-7)&(dist>=0)&(dist<.5)
 return None if not valid.any()else float(p[1]-(o[1]-dist[valid].min()))
def run(T):
 check(T,4*1024**2);R=T/'round4';sys.path.insert(0,str(T/'round3/pydeps'));import fcl
 meta=json.loads((R/'assets/parrot/source.json').read_text());d,b=read_glb(R/'assets/parrot/runtime.glb');ps,_=pose(d,b,canonical_flip=False,ground=False);verts=np.concatenate([p['positions']for p in ps]);idx=[];offset=0
 for p in ps:
  pr=d['meshes'][d['nodes'][p['node']]['mesh']]['primitives'][p['primitive']];ix=accessor(d,b,pr['indices']).astype(int).reshape(-1,3);idx.append(ix+offset);offset+=len(p['positions'])
 faces=np.concatenate(idx);surface_distance=float(cKDTree(verts).query(meta['beak_local_glb_m'])[0]);minimum_y=float(verts[:,1].min());height=float(np.ptp(verts[:,1]))
 raw=np.load(T/'evidence/ue_source_triangles.npz')['triangles'].astype(float)/100;scene=np.stack([raw[:,:,1],raw[:,:,2],-raw[:,:,0]],axis=2);cm=json.loads((T/'evidence/ue_source_floor_plan.json').read_text())['selected']['location_cm'];root=np.array([cm[1],cm[2],-cm[0]])/100
 lo=scene.min(1);hi=scene.max(1);near=scene[np.all(hi>=root-[1.7,.1,1.7],1)&np.all(lo<=root+[1.7,1.2,1.7],1)];nn=np.cross(near[:,1]-near[:,0],near[:,2]-near[:,0]);ny=abs(nn[:,1])/np.maximum(np.linalg.norm(nn,axis=1),1e-9);floor=(ny>.98)&(near[:,:,1].max(1)<root[1]+.02)&(near[:,:,1].min(1)>root[1]-.02);obstacles=near[~floor]
 body=model(verts+root,faces);env=model(obstacles.reshape(-1,3),np.arange(obstacles.size//3).reshape(-1,3));cr=fcl.CollisionResult();collision=bool(fcl.collide(body,env,fcl.CollisionRequest(num_max_contacts=1),cr));dr=fcl.DistanceResult();distance=float(fcl.distance(body,env,fcl.DistanceRequest(),dr))
 pc=meta['perch_patch']['center_m'];supports=[downward_hit(root+[pc[0]+x,0,pc[2]+z],near)for x,z in [(0,0),(-.07,0),(.07,0),(0,-.07),(0,.07)]]
 points=[root+[0,.425,0],root+[0,.03,0],root+[0,.84,0],root+np.asarray(meta['beak_local_glb_m']),root+[.10,.60,0],root+[-.10,.60,0]]
 cameras=[]
 for angle in [0,math.pi/6,-math.pi/6,math.pi/3,-math.pi/3]:
  if len(cameras)>=2:break
  for distance_camera in [1.35,1.10,.95]:
   eye=root+np.array([math.sin(angle)*distance_camera,.72,-math.cos(angle)*distance_camera])
   if all(visible(eye,p,scene,lo,hi)for p in points):
    cameras.append({'eye_glb_m':eye.tolist(),'target_glb_m':(root+[0,.425,0]).tolist(),'fov_degrees':50,'source_clear_rays':True,'source_ray_count':len(points)});break
 passed=surface_distance<1e-5 and abs(minimum_y)<1e-5 and abs(height-.85)<.002 and not collision and distance>.02 and all(s is not None and abs(s)<.012 for s in supports)and len(cameras)==2
 rec={'asset_id':meta['asset_id'],'status':'pass'if passed else'fail','root_m':root.tolist(),'source_floor_support_deltas_m':supports,'minimum_geometry_y_m':minimum_y,'total_geometry_height_m':height,'beak_nearest_exported_vertex_m':surface_distance,'source_collision':collision,'minimum_non_floor_distance_m':distance,'scene_obstacle_triangles':len(obstacles),'expected_ground_triangles_excluded':int(floor.sum()),'cameras':cameras,'source_triangles':str(T/'evidence/ue_source_triangles.npz'),'method':'whole exported mesh versus actual composed source-USD triangles FCL; five floor rays and six camera rays','native_ue_physics':'not_claimed'}
 json_write(R/'evidence/parrot_source_geometry.json',rec);print('PARROT_SOURCE_GEOMETRY',rec['status'],distance,supports,len(cameras));return rec
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
