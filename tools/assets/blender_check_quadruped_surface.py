"""Strict nonadjacent crossings compared with the target's binding pose."""
import argparse,json,pathlib,sys,time,math
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
REGIONS=("Body","Head","Tail","FL","FR","HL","HR")
def segments_hit(a,b,c,p,q):
 e1=b-a;e2=c-a;d=q-p;h=np.cross(d,e2);det=np.einsum("ij,ij->i",e1,h)
 usable=np.abs(det)>1e-18
 inv=np.divide(1.,det,out=np.zeros_like(det),where=usable)
 s=p-a;u=np.einsum("ij,ij->i",s,h)*inv;v=np.einsum("ij,ij->i",d,np.cross(s,e1))*inv;t=np.einsum("ij,ij->i",e2,np.cross(s,e1))*inv
 return usable&(u>1e-7)&(v>1e-7)&(u+v<1-1e-7)&(t>1e-7)&(t<1-1e-7)
def intersection_pairs(vertices,faces,weld,tree):
 pairs=np.asarray([(a,b) for a,b in tree.overlap(tree) if a<b],dtype=int)
 if not len(pairs):return set(),0
 fa=weld[faces[pairs[:,0]]];fb=weld[faces[pairs[:,1]]]
 pairs=pairs[~(fa[:,:,None]==fb[:,None,:]).any(axis=(1,2))]
 if not len(pairs):return set(),0
 ta=vertices[faces[pairs[:,0]]];tb=vertices[faces[pairs[:,1]]];hit=np.zeros(len(pairs),dtype=bool)
 for i in range(3):
  hit|=segments_hit(tb[:,0],tb[:,1],tb[:,2],ta[:,i],ta[:,(i+1)%3])
  hit|=segments_hit(ta[:,0],ta[:,1],ta[:,2],tb[:,i],tb[:,(i+1)%3])
 na=np.cross(ta[:,1]-ta[:,0],ta[:,2]-ta[:,0]);nb=np.cross(tb[:,1]-tb[:,0],tb[:,2]-tb[:,0])
 an=np.linalg.norm(na,axis=1);bn=np.linalg.norm(nb,axis=1)
 normcross=np.linalg.norm(np.cross(na,nb),axis=1);plane=np.abs(np.einsum("ij,ij->i",na,tb[:,0]-ta[:,0]))
 coplanar=(an>1e-12)&(bn>1e-12)&(normcross<1e-7*an*bn)&(plane<1e-8*an)
 return {tuple(map(int,x)) for x in pairs[hit]},int(coplanar.sum())
def regional_faces(pairs,labels):
 groups={}
 for a,b in pairs:
  ra,rb=int(labels[a]),int(labels[b]);key=tuple(sorted((ra,rb)))
  group=groups.setdefault(key,{})
  group.setdefault(ra,set()).add(a);group.setdefault(rb,set()).add(b)
 return groups
def macro_penetrations(pairs,labels,baseline_groups,areas,static_region_areas):
 failures=[]
 for key,group in regional_faces(pairs,labels).items():
  for region,ids in group.items():
   new=ids-baseline_groups.get(key,{}).get(region,set())
   new_area=float(areas[list(new)].sum()) if new else 0.
   budget=float(static_region_areas[region]*.005)
   if new_area>budget+1e-12:
    failures.append({"region_pair":[REGIONS[i] for i in key],"affected_region":REGIONS[region],
     "newly_intersecting_face_area_m2":new_area,"region_static_area_m2":float(static_region_areas[region]),
     "new_area_fraction":new_area/max(float(static_region_areas[region]),1e-20),"maximum_fraction":.005,"new_face_count":len(new)})
 return failures
def main():
 args=sys.argv[sys.argv.index("--")+1:];ap=argparse.ArgumentParser()
 ap.add_argument("--glb",type=pathlib.Path,required=True);ap.add_argument("--output",type=pathlib.Path,required=True)
 ap.add_argument("--action",default="Walking_Repaired");ap.add_argument("--frames",type=int,default=145);a=ap.parse_args(args)
 start=time.process_time();bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
 scene.render.fps=30;scene.render.threads_mode="FIXED";scene.render.threads=1
 bpy.ops.import_scene.gltf(filepath=str(a.glb),merge_vertices=False)
 rigs=[o for o in scene.objects if o.type=="ARMATURE"];meshes=[o for o in scene.objects if o.type=="MESH"]
 if len(rigs)!=1:raise ValueError("one_target_armature_required")
 rig=rigs[0];rig.animation_data.action=None;rig.animation_data.use_nla=False
 for bone in rig.pose.bones:bone.matrix_basis.identity()
 bpy.context.view_layer.update()
 def geometry():
  dep=bpy.context.evaluated_depsgraph_get();allv=[];allf=[];off=0
  for o in meshes:
   ev=o.evaluated_get(dep);m=ev.to_mesh();m.calc_loop_triangles()
   v=np.empty(len(m.vertices)*3,dtype=float);m.vertices.foreach_get("co",v);v=v.reshape(-1,3)
   mat=np.array(o.matrix_world);v=v@mat[:3,:3].T+mat[:3,3]
   f=np.empty(len(m.loop_triangles)*3,dtype=np.int64);m.loop_triangles.foreach_get("vertices",f);f=f.reshape(-1,3)
   allv.append(v);allf.append(f+off);off+=len(v);ev.to_mesh_clear()
  return np.concatenate(allv),np.concatenate(allf)
 v0,f=geometry();regional_weights=[]
 for o in meshes:
  weights=np.zeros((len(o.data.vertices),len(REGIONS)))
  for vert in o.data.vertices:
   for group in vert.groups:
    name=o.vertex_groups[group.group].name
    region=next((i for i,side in enumerate(REGIONS[3:],3) if name.endswith("_"+side)),
     1 if name.startswith("Head") else 2 if name.startswith("Tail") else 0)
    weights[vert.index,region]+=group.weight
  regional_weights.append(weights)
 vw=np.concatenate(regional_weights)
 if len(vw)!=len(v0):raise ValueError("evaluated mesh changed vertex count")
 labels=vw[f].mean(1).argmax(1)
 _,weld=np.unique(np.round(v0,7),axis=0,return_inverse=True)
 tree=BVHTree.FromPolygons(v0.tolist(),f.tolist(),all_triangles=True,epsilon=0.)
 baseline,coplanar0=intersection_pairs(v0,f,weld,tree);baseline_groups=regional_faces(baseline,labels)
 t0=v0[f];e0=np.max(np.linalg.norm(t0-np.roll(t0,1,axis=1),axis=2),axis=1)
 static_area=np.linalg.norm(np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0]),axis=1)/2
 region_areas=np.bincount(labels,weights=static_area,minlength=len(REGIONS))
 tolerance=max(12,math.ceil(len(baseline)*.005))
 act=next(x for x in bpy.data.actions if a.action in x.name);rig.animation_data.action=act
 begin,end=map(float,act.frame_range);samples=np.linspace(begin,end,a.frames)
 rows=[];newpairs=set();uncertain=False;relative_failures=[]
 for index,frame in enumerate(samples):
  scene.frame_set(math.floor(frame),subframe=frame-math.floor(frame));bpy.context.view_layer.update();v,f1=geometry()
  if not np.array_equal(f,f1):raise ValueError("pose_changed_triangle_topology")
  tri=v[f];edge=np.max(np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2),axis=1)
  area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;rat=edge/np.maximum(e0,1e-12)
  shard=rat>3.;tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True,epsilon=0.)
  hits,coplanar=intersection_pairs(v,f,weld,tree);new=hits-baseline;newpairs|=new;uncertain|=(coplanar>coplanar0)
  excess=max(0,len(hits)-len(baseline));macro=macro_penetrations(hits,labels,baseline_groups,area,region_areas)
  if excess>tolerance:relative_failures.append({"frame":index,"reason":"intersection_count_increase","increase":excess,"tolerance":tolerance})
  if macro:relative_failures.append({"frame":index,"reason":"new_macroscopic_penetration","regions":macro})
  row={"frame":index,"blender_frame":frame,"self_intersections":len(hits),"excess_over_static_count":excess,
   "new_pair_id_count_diagnostic":len(new),"macroscopic_penetration_failures":macro,"coplanar_candidates":coplanar,
   "shard_share_area":float(area[shard].sum()/max(area.sum(),1e-20)),"max_edge_growth":float(rat.max()),
   "nonfinite_vertices":int((~np.isfinite(v)).any(axis=1).sum()),"new_pair_examples":[list(p) for p in sorted(new)[:10]]}
  rows.append(row)
  if index%30==0:print("SURFACE_FRAME",index,"pairs",len(hits),"baseline",len(baseline),"excess",excess,"macro_failures",len(macro),flush=True)
 result={"schema":"blender_quadruped_relative_surface_check_v2","input":str(a.glb),"action":a.action,
  "frames":rows,"frame_count":len(rows),"source_baseline_true_intersections":len(baseline),
  "source_baseline_coplanar_candidates":coplanar0,"new_true_triangle_pair_count_diagnostic":len(newpairs),
  "relative_failures":relative_failures,"coplanar_status":"unresolved_increase" if uncertain else "no_increase",
  "worst_shard_share_area":max(x["shard_share_area"] for x in rows),"shard_edge_growth":3.,
  "status":"pass" if not relative_failures and not uncertain else "fail",
  "absolute_self_intersection_status":"clear" if not baseline and not newpairs and not uncertain and coplanar0==0 else "present_or_unresolved",
  "thresholds":{"count_increase_maximum_pairs":tolerance,"count_increase_fraction":.005,"minimum_count_tolerance":12,
   "new_intersecting_region_face_area_maximum_fraction":.005,
   "reason":"Owner permits a small count increment over static self-intersections. 0.5% or 12 pairs covers fine surface numerical changes; independent 0.5% region-area gate prevents disappearing inherited pairs from hiding substantial new limb/body or tail/body intrusion."},
  "static_region_areas_m2":dict(zip(REGIONS,region_areas.tolist())),"cpu_s":time.process_time()-start,
  "method":"Complete BVH candidate set; exclude common rest geometric vertex; strict interior segment/triangle crossings. Reset all pose transforms and disable NLA for binding baseline. Regional new face area counted relative to same baseline region pair."}
 a.output.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ("frames","relative_failures")},flush=True)
 if result["status"]!="pass":raise SystemExit(2)
if __name__=="__main__":main()
