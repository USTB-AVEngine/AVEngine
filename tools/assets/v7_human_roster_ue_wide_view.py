"""Find a broad unobstructed native-room review view from composed USD triangles."""
from pathlib import Path
import argparse
import json,sys,math
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,required=True);T=parser.parse_args().task_root;sys.path.insert(0,str(T/'worktree/tools/assets'));from v7_human_roster_ue_floor import ray,_RAY_BOUNDS
c=np.load(T/'evidence/ue_source_triangles.npz');tri=c['triangles'];floor=c['floor'];lo=tri.min(1);hi=tri.max(1);_RAY_BOUNDS[id(tri)]=(lo,hi);bounds_lo=tri.min((0,1));bounds_hi=tri.max((0,1));old=json.loads((T/'evidence/ue_source_floor_plan.json').read_text());un_lo=np.array(old['selection_volume_local_cm']['min']);un_hi=np.array(old['selection_volume_local_cm']['max']);candidates=[]
for x in np.arange(bounds_lo[0]+85,bounds_hi[0]-85,40):
 for y in np.arange(bounds_lo[1]+85,bounds_hi[1]-85,40):
  gs=[p for p in ray(floor,[x,y,200],[x,y,-70])if -5<=p[2]<=30]
  if not gs:continue
  z=max(p[2]for p in gs);pos=np.array([x,y,z])
  for phi in [0,90,180,270]:
   th=math.radians(phi);R=np.array([[math.cos(th),-math.sin(th),0],[math.sin(th),math.cos(th),0],[0,0,1]]);corners=np.array([[a,b,d]for a in [un_lo[0],un_hi[0]]for b in [un_lo[1],un_hi[1]]for d in [2,un_hi[2]]])@R.T+pos
   aa=corners.min(0);bb=corners.max(0)
   if np.any(np.all(hi>aa,axis=1)&np.all(lo<bb,axis=1)):continue
   direction=np.array([math.cos(th),math.sin(th),0]);eye=pos+direction*330+[0,0,110]
   if np.any(eye[:2]<bounds_lo[:2]+10)or np.any(eye[:2]>bounds_hi[:2]-10):continue
   targets=[pos+[0,0,95],*corners]
   if any(ray(tri,eye,p)for p in targets):continue
   candidate={'location_cm':pos.tolist(),'camera_cm':eye.tolist(),'floor_hits_cm':gs,'heading_deg':phi,'actor_yaw_deg':(90+phi)%360,'all_assets_union_aabb_room_triangle_overlaps_above_floor':0,'box_corner_sightlines_clear':True};candidates.append(candidate)
   print('CLEAR_WIDE_VIEW',len(candidates),candidate,flush=True)
if not candidates:raise RuntimeError('No full standing-union view with eight unobstructed corner sightlines')
# Prefer a straight +X convention where possible, then a central wide view.
preferred=[x for x in candidates if x['heading_deg']==0]or candidates
old.update(candidates=candidates,selected=preferred[len(preferred)//2],selection_method='actual floor rays, zero room-triangle AABB overlap for every asset, eight union-corner plus centre source-geometry sightlines',clear_view_count=len(candidates))
(T/'evidence/ue_source_floor_plan.json').write_text(json.dumps(old,indent=2));print('SELECTED',old['selected'],flush=True)
