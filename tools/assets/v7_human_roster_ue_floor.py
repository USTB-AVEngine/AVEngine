"""Locate an empty review volume in the actual USD geometry; no stage mutation."""
import argparse,json,sys
from pathlib import Path
import numpy as np

def triangles(stage):
 from pxr import Usd,UsdGeom
 cache=UsdGeom.XformCache(Usd.TimeCode.Default());out=[];floor=[];bounds=[]
 for prim in stage.Traverse():
  if not prim.IsA(UsdGeom.Mesh):continue
  mesh=UsdGeom.Mesh(prim)
  if UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible:continue
  pts=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64);matrix=np.asarray(cache.GetLocalToWorldTransform(prim))
  world=np.c_[pts,np.ones(len(pts))]@matrix;world=world[:,:3]
  # USD is RH Z-up; this native USD import reflects Y into UE's LH basis.
  world*=np.array([100,-100,100]);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=int);ids=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=int)
  if np.all(counts==3):ix=ids.reshape(-1,3)
  else:
   ix=[];o=0
   for n in counts:
    ix.extend([ids[o],ids[o+j],ids[o+j+1]]for j in range(1,n-1));o+=n
   ix=np.asarray(ix)
  tri=world[ix].astype(np.float32);out.append(tri);bounds.append({'path':str(prim.GetPath()),'min_cm':world.min(0).tolist(),'max_cm':world.max(0).tolist()})
  if '/floor/'in str(prim.GetPath()):floor.append(tri)
 return np.concatenate(out),np.concatenate(floor),bounds

_RAY_BOUNDS={}
def ray(tri,start,end):
 start=np.asarray(start,float);direction=np.asarray(end,float)-start;lo=np.minimum(start,end)-1e-4;hi=np.maximum(start,end)+1e-4
 key=id(tri)
 if key not in _RAY_BOUNDS:_RAY_BOUNDS[key]=(tri.min(1),tri.max(1))
 bbox_lo,bbox_hi=_RAY_BOUNDS[key]
 use=np.all(bbox_hi>=lo,axis=1)&np.all(bbox_lo<=hi,axis=1);f=tri[use].astype(float)
 if not len(f):return []
 e1=f[:,1]-f[:,0];e2=f[:,2]-f[:,0];h=np.cross(direction,e2);a=np.einsum('ij,ij->i',e1,h);valid=np.abs(a)>1e-9;inv=np.zeros_like(a);inv[valid]=1/a[valid];s=start-f[:,0];u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,e1);v=inv*(q@direction);t=inv*np.einsum('ij,ij->i',e2,q);valid&=(u>=-1e-6)&(v>=-1e-6)&(u+v<=1+1e-6)&(t>=0)&(t<=1)
 ts=np.unique(np.round(t[valid],8))
 return [(start+direction*x).tolist()for x in ts]
def run(a):
 sys.path.insert(0,str(a.task_root/'pydeps'))
 from pxr import Usd
 source=Path('/data/avengine_external/ue-assets/kujiale_0020_full_home_current_d702c47_20260821T0130Z/derived/kujiale_0020_full_home_preview_surface.usda')
 cache=a.task_root/'evidence/ue_source_triangles.npz'
 if cache.exists():
  c=np.load(cache);tri=c['triangles'];floor=c['floor'];bounds=json.loads((a.task_root/'evidence/ue_source_usd_point_bounds.json').read_text())['meshes']
 else:tri,floor,bounds=triangles(Usd.Stage.Open(str(source)))
 lo=tri.min((0,1));hi=tri.max((0,1))
 if not cache.exists():np.savez_compressed(cache,triangles=tri,floor=floor)
 print('ACTUAL_POINTS_BOUNDS_CM',lo.tolist(),hi.tolist(),'TRIANGLES',len(tri),flush=True)
 (a.task_root/'evidence/ue_source_usd_point_bounds.json').write_text(json.dumps({'bounds_min_cm':lo.tolist(),'bounds_max_cm':hi.tolist(),'meshes':bounds},indent=2))
 candidates=[]
 for x in np.arange(lo[0]+80,hi[0]-330,40):
  for y in np.arange(lo[1]+80,hi[1]-80,40):
   hs=ray(floor,[x,y,200],[x,y,-70]);gs=[p for p in hs if -5<=p[2]<=30]
   if not gs:continue
   z=max(p[2]for p in gs);clear=True
   # Include body height, both feet, and broad compound props in the review volume.
   for dx,dy in [(0,0),(-35,0),(35,0),(0,-65),(0,65),(-35,-65),(35,65)]:
    if ray(tri,[x+dx,y+dy,z+2],[x+dx,y+dy,z+205]):clear=False;break
   if not clear:continue
   eye=[x+330,y,z+110];target=[x,y,z+95]
   if ray(tri,eye,target):continue
   if ray(tri,[x+330,y,z+2],[x+330,y,z+200]):continue
   candidates.append({'location_cm':[float(x),float(y),float(z)],'camera_cm':eye,'floor_hits_cm':gs})
 if not candidates:raise RuntimeError('No empty review volume found in source geometry')
 asset_bounds=[]
 for pose_file in (a.task_root/'assets/runtime').glob('*/*/poses.json'):
  info=json.loads(pose_file.read_text());aa=np.array(info['poses']['stand']['bounds_min_m']);bb=np.array(info['poses']['stand']['bounds_max_m'])
  asset_bounds.append((np.array([-bb[2],aa[0],aa[1]])*100,np.array([-aa[2],bb[0],bb[1]])*100))
 union_lo=np.min([x[0]for x in asset_bounds],0);union_hi=np.max([x[1]for x in asset_bounds],0);mesh_lo,mesh_hi=_RAY_BOUNDS[id(tri)]
 for candidate in candidates:
  pos=np.array(candidate['location_cm']);aa=union_lo+pos;bb=union_hi+pos;aa[2]=pos[2]+2
  candidate['all_assets_union_aabb_room_triangle_overlaps_above_floor']=int((np.all(mesh_hi>aa,axis=1)&np.all(mesh_lo<bb,axis=1)).sum())
 safe=[c for c in candidates if c['all_assets_union_aabb_room_triangle_overlaps_above_floor']==0]
 if not safe:raise RuntimeError('No candidate fits union of all standing assets without room geometry overlap')
 selected=safe[len(safe)//2]
 (a.task_root/'evidence/ue_source_floor_plan.json').write_text(json.dumps({'source':str(source),'method':'triangle intersections on composed visible source USD; UE native mesh bounds must agree','source_to_ue':'100*(x,-y,z)','bounds_min_cm':lo.tolist(),'bounds_max_cm':hi.tolist(),'triangles':len(tri),'candidates':candidates,'selected':selected,'selection_volume_local_cm':{'min':union_lo.tolist(),'max':union_hi.tolist()},'empty_volume_check':'source triangles vs union of all 120 standing-asset bounds; excludes floor contact zone <=2cm'},indent=2))
 print('REVIEW_CANDIDATES',len(candidates),selected,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args())
