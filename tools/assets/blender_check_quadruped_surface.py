"""Owner-approved regional penetration gate against the unmodified binding pose."""
import argparse,json,pathlib,sys,time,math
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"src"))
from avengine.assets.quadruped_penetration import pair_footprints,visible_penetration_gate
REGIONS=("Body","Head","Tail","FL","FR","HL","HR")
def segments_hit(a,b,c,p,q):
 e1=b-a;e2=c-a;d=q-p;h=np.cross(d,e2);det=np.einsum("ij,ij->i",e1,h)
 usable=np.abs(det)>1e-18
 inv=np.divide(1.,det,out=np.zeros_like(det),where=usable)
 s=p-a;u=np.einsum("ij,ij->i",s,h)*inv;v=np.einsum("ij,ij->i",d,np.cross(s,e1))*inv;t=np.einsum("ij,ij->i",e2,np.cross(s,e1))*inv
 return usable&(u>1e-7)&(v>1e-7)&(u+v<1-1e-7)&(t>1e-7)&(t<1-1e-7)
def intersection_candidates(vertices,faces,weld,tree):
 pairs=np.asarray([(a,b) for a,b in tree.overlap(tree) if a<b],dtype=int).reshape(-1,2)
 if not len(pairs):return set(),set()
 fa=weld[faces[pairs[:,0]]];fb=weld[faces[pairs[:,1]]]
 pairs=pairs[~(fa[:,:,None]==fb[:,None,:]).any(axis=(1,2))]
 if not len(pairs):return set(),set()
 ta=vertices[faces[pairs[:,0]]];tb=vertices[faces[pairs[:,1]]];hit=np.zeros(len(pairs),dtype=bool)
 for i in range(3):
  hit|=segments_hit(tb[:,0],tb[:,1],tb[:,2],ta[:,i],ta[:,(i+1)%3])
  hit|=segments_hit(ta[:,0],ta[:,1],ta[:,2],tb[:,i],tb[:,(i+1)%3])
 na=np.cross(ta[:,1]-ta[:,0],ta[:,2]-ta[:,0]);nb=np.cross(tb[:,1]-tb[:,0],tb[:,2]-tb[:,0])
 an=np.linalg.norm(na,axis=1);bn=np.linalg.norm(nb,axis=1)
 normcross=np.linalg.norm(np.cross(na,nb),axis=1);plane=np.abs(np.einsum("ij,ij->i",na,tb[:,0]-ta[:,0]))
 coplanar=(an>1e-12)&(bn>1e-12)&(normcross<1e-7*an*bn)&(plane<1e-8*an)
 return {tuple(map(int,x)) for x in pairs[hit]},{tuple(map(int,x)) for x in pairs[coplanar]}
def intersection_pairs(vertices,faces,weld,tree):
 hits,coplanar=intersection_candidates(vertices,faces,weld,tree)
 return hits,len(coplanar)
def face_areas(v,f):
 t=v[f];return np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
def footprints(v,f,pairs):
 p=np.asarray(sorted(pairs),dtype=np.int64).reshape(-1,2)
 area,depth=pair_footprints(v,f,p)
 return p,area,depth
def regional_areas(p,patches,labels,areas):
 out={}
 if not len(p):return out
 codes=np.sort(labels[p],axis=1)
 for key in np.unique(codes,axis=0):
  key=tuple(map(int,key));selected=(codes==key).all(1)
  values=np.bincount(p[selected].ravel(),weights=patches[selected].ravel(),minlength=len(labels))
  out[key]=np.minimum(values,areas)
 return out
def regional_measurements(p,area,depth,labels,areas,baseline_areas,basecodes,basedepths):
 if not len(p):return []
 code=p[:,0]*len(labels)+p[:,1]
 location=np.searchsorted(basecodes,code);matched=location<len(basecodes)
 safe=np.minimum(location,max(0,len(basecodes)-1))
 if len(basecodes):matched&=basecodes[safe]==code
 else:matched[:]=False
 prior=np.zeros_like(depth)
 if matched.any():prior[matched]=basedepths[safe[matched]]
 newdepth=np.maximum(depth-prior,0)
 codes=np.sort(labels[p],axis=1);current=regional_areas(p,area,labels,areas);rows=[]
 for key,values in current.items():
  priorarea=baseline_areas.get(key,np.zeros(len(labels)))
  selected=(codes==key).all(1)
  rows.append({"region_pair":[REGIONS[i] for i in key],
   "new_area_m2":float(np.maximum(values-priorarea,0).sum()),
   "current_local_crossed_surface_area_m2":float(values.sum()),
   "binding_local_crossed_surface_area_m2":float(priorarea.sum()),
   "new_maximum_depth_m":float(newdepth[selected].max()),
   "absolute_maximum_depth_m_diagnostic":float(depth[selected].max()),
   "crossing_pair_count_diagnostic":int(selected.sum())})
 return rows
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
 weldsort=np.argsort(weld,kind="stable");weldstart=np.r_[0,np.flatnonzero(np.diff(weld[weldsort]))+1]
 rest_seams=np.linalg.norm(np.maximum.reduceat(v0[weldsort],weldstart)-np.minimum.reduceat(v0[weldsort],weldstart),axis=1)
 tree=BVHTree.FromPolygons(v0.tolist(),f.tolist(),all_triangles=True,epsilon=0.)
 baseline,coplanar0=intersection_candidates(v0,f,weld,tree)
 bp,ba,bd=footprints(v0,f,baseline|coplanar0)
 basecodes=bp[:,0]*len(f)+bp[:,1]
 static_area=face_areas(v0,f);baseline_areas=regional_areas(bp,ba,labels,static_area)
 t0=v0[f];e0=np.max(np.linalg.norm(t0-np.roll(t0,1,axis=1),axis=2),axis=1)
 region_areas=np.bincount(labels,weights=static_area,minlength=len(REGIONS))
 act=next(x for x in bpy.data.actions if a.action in x.name);rig.animation_data.action=act
 begin,end=map(float,act.frame_range);samples=np.linspace(begin,end,a.frames)
 rows=[];newpairs=set();failures=[]
 for index,frame in enumerate(samples):
  scene.frame_set(math.floor(frame),subframe=frame-math.floor(frame));bpy.context.view_layer.update();v,f1=geometry()
  if not np.array_equal(f,f1):raise ValueError("pose_changed_triangle_topology")
  tri=v[f];edge=np.max(np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2),axis=1)
  area=face_areas(v,f);rat=edge/np.maximum(e0,1e-12);shard=rat>3.
  tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True,epsilon=0.)
  hits,coplanar=intersection_candidates(v,f,weld,tree);new=hits-baseline;newpairs|=new
  p,pa,pd=footprints(v,f,hits|coplanar)
  regions=regional_measurements(p,pa,pd,labels,area,baseline_areas,basecodes,bd)
  macro=visible_penetration_gate(regions)
  seam=np.linalg.norm(np.maximum.reduceat(v[weldsort],weldstart)-np.minimum.reduceat(v[weldsort],weldstart),axis=1)
  new_seam=float(np.maximum(seam-rest_seams,0).max())
  if macro:failures.append({"frame":index,"reason":"new_visible_regional_penetration","regions":macro})
  if float(rat.max())>1.5+1e-6:failures.append({"frame":index,"reason":"edge_growth_over_1_5","maximum":float(rat.max())})
  if new_seam>1e-5:failures.append({"frame":index,"reason":"new_uv_seam_tear","maximum_gap_increase_m":new_seam})
  if not np.isfinite(v).all():failures.append({"frame":index,"reason":"nonfinite_vertices"})
  row={"frame":index,"blender_frame":float(frame),"self_intersections":len(hits),
   "excess_over_static_count_diagnostic":max(0,len(hits)-len(baseline)),
   "new_pair_id_count_diagnostic":len(new),"regional_penetration_measurements":regions,
   "macroscopic_penetration_failures":macro,"coplanar_candidates_diagnostic":len(coplanar),
   "shard_share_area":float(area[shard].sum()/max(area.sum(),1e-20)),"max_edge_growth":float(rat.max()),
   "new_maximum_uv_seam_gap_m":new_seam,"nonfinite_vertices":int((~np.isfinite(v)).any(axis=1).sum()),
   "new_pair_examples_diagnostic":[list(p) for p in sorted(new)[:10]]}
  rows.append(row)
  if index%30==0:print("SURFACE_FRAME",index,"pairs",len(hits),"baseline",len(baseline),"new_visible_failures",len(macro),"stretch",row["max_edge_growth"],flush=True)
 result={"schema":"blender_quadruped_visible_regional_surface_check_v3","gate_revision":"owner_20261014_023x",
  "input":str(a.glb),"action":a.action,"frames":rows,"frame_count":len(rows),
  "source_baseline_true_intersections":len(baseline),"source_baseline_coplanar_candidates":len(coplanar0),
  "new_true_triangle_pair_count_diagnostic":len(newpairs),"intersection_counts_are_admission_criteria":False,
  "relative_failures":failures,"worst_shard_share_area":max(x["shard_share_area"] for x in rows),
  "maximum_edge_growth":max(x["max_edge_growth"] for x in rows),"shard_edge_growth":3.,
  "status":"pass" if not failures else "fail",
  "thresholds":{"cross_region_new_area_maximum_m2":25e-6,"new_penetration_depth_maximum_m":.002,
   "same_region_new_area_is_admission_criterion":False,"maximum_edge_growth":1.5,
   "maximum_uv_seam_gap_increase_m":1e-5,
   "reason":"Owner 02:3x: count increases are mesh-layer noise; 2mm is below a production pixel at approximately 2m/720p. New crossed surface area between regions is limited to 5x5mm. UV seam tolerance 10 micrometres accommodates float32 evaluation and is below visible scale."},
  "static_region_areas_m2":dict(zip(REGIONS,region_areas.tolist())),"cpu_s":time.process_time()-start,
  "method":"Complete BVH candidates; exclude common rest geometric vertex. Exact Sutherland-Hodgman clipping of each crossing face to the other face's orthogonal triangular prism and inward plane gives local penetration footprint area and normal depth. Both surfaces are measured; overlapping footprints are summed and capped by each face area, then positive per-face area increments over binding pose are summed by region pair. Depth is the positive increase over the same binding triangle pair; absent binding pairs have zero baseline. This local surface metric is not a volumetric intersection calculation. Coplanar candidates are measured by footprint, never rejected by count. Pose reset and NLA disabled for binding baseline. Longest triangle-edge growth and geometric UV seam separation measured on every frame."}
 a.output.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ("frames","relative_failures")},flush=True)
 if result["status"]!="pass":raise SystemExit(2)
if __name__=="__main__":main()
