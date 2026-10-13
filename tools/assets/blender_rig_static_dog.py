"""Landmark fit, Blender heat/envelope skinning, and CC0 motion transfer."""
import argparse, json, pathlib, sys, time, math
import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion
from mathutils.kdtree import KDTree
HAB_TO_BLENDER=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
# Source Dog.blend: +X forward, Z up, +Y left. Target: -Z forward, Y up.
DONOR_TO_HAB=np.array([[0,-1,0],[0,0,1],[-1,0,0]],float)
DONOR_TO_BLENDER=HAB_TO_BLENDER@DONOR_TO_HAB
SIDES=("FL","FR","HL","HR")
DONOR_CHAINS={"FL":("Bone.017","Bone.018","Bone.019"),"FR":("Bone.014","Bone.015","Bone.016"),"HL":("Bone.008","Bone.009","Bone.010"),"HR":("Bone.011","Bone.012","Bone.013")}
def asmatrix(m):return np.array(m,dtype=float)
def unitrot(m):
 r=asmatrix(m)[:3,:3];return r/np.linalg.norm(r,axis=0)
def sample_donor(template):
 bpy.ops.wm.open_mainfile(filepath=str(template))
 scene=bpy.context.scene;scene.render.threads_mode="FIXED";scene.render.threads=1
 donor=next(o for o in bpy.data.objects if o.type=="ARMATURE")
 names=[b.name for b in donor.data.bones if b.use_deform]
 idle=bpy.data.actions.get("Idle");walk=bpy.data.actions.get("Walking")
 if not idle or not walk:raise ValueError("CC0_source_missing_Idle_or_Walking")
 donor.animation_data_create();donor.animation_data.action=idle;scene.frame_set(0);bpy.context.view_layer.update()
 reference={n:asmatrix(donor.matrix_world@donor.pose.bones[n].matrix) for n in names}
 mats={};frames=np.linspace(float(walk.frame_range[0]),float(walk.frame_range[1]),41)
 for action in (idle,walk):
  donor.animation_data.action=action;sample=[]
  for f in frames:
   scene.frame_set(int(f),subframe=float(f-int(f)));bpy.context.view_layer.update()
   sample.append({n:asmatrix(donor.matrix_world@donor.pose.bones[n].matrix) for n in names})
  mats[action.name]=sample
 footmap={}
 for side,chain in DONOR_CHAINS.items():
  footmap[side]=(DONOR_TO_HAB@reference[chain[2]][:3,3]).tolist()
 # Geometry establishes side mapping, rather than inherited names.
 assert footmap["FL"][0]<footmap["FR"][0] and footmap["HL"][0]<footmap["HR"][0]
 assert footmap["FL"][2]<footmap["HL"][2] and footmap["FR"][2]<footmap["HR"][2]
 source_span=float(np.ptp([m[2,3] for m in reference.values()]))
 return reference,mats,frames,source_span,footmap

def fit_bones(vertices,registration):
 feet=np.array([registration["landmarks_local_m"][f"foot_{i}"] for i in range(4)],float)
 order=np.argsort(feet[:,2]);order=np.r_[order[:2][np.argsort(feet[order[:2],0])],order[2:][np.argsort(feet[order[2:],0])]]
 feet=feet[order];head=np.asarray(registration["head_center_local_m"]);mouth=np.asarray(registration["emit_point_local_m"])
 floor=float(vertices[:,1].min());height=float(np.ptp(vertices[:,1]));width=float(np.ptp(vertices[:,0]))
 rear=feet[2:].mean(0);front=feet[:2].mean(0)
 torso_y=floor+height*.53
 back=np.array([np.mean(feet[:,0]),torso_y,rear[2]*.85]);chest=np.array([np.mean(feet[:2,0]),torso_y,front[2]*.9]);middle=(back+chest)/2
 # Own central rear surface, above limb level: resolve a tail attachment.
 mask=(vertices[:,2]>rear[2]*.7)&(vertices[:,1]>floor+height*.48)&(np.abs(vertices[:,0]-back[0])<width*.20)
 rv=vertices[mask]
 if len(rv)<10:raise ValueError("own_target_tail_root_geometry_unresolved")
 zlo=np.quantile(rv[:,2],.65);tail=np.median(rv[rv[:,2]>=zlo],axis=0);tail[0]=back[0]
 tailtip=np.median(rv[rv[:,2]>=np.quantile(rv[:,2],.96)],axis=0)
 if np.linalg.norm(tailtip-tail)<height*.02:tailtip=tail+[0,height*.04,height*.08]
 midtail=tail+(tailtip-tail)*.5;aftertail=tailtip+(tailtip-tail)*.25
 names=["Body","Spine","Chest","Neck","Head","Tail","Tail1","Tail2"]
 heads=[back,middle,chest,(chest+head)/2,head,tail,midtail,tailtip]
 tails=[middle,chest,(chest+head)/2,head,mouth,midtail,tailtip,aftertail]
 parents=[-1,0,1,2,3,0,5,6]
 mappings=["Bone","Bone.001","Bone.001","Bone.002","Bone.003","Bone.004","Bone.005","Bone.006"]
 for side,foot in zip(SIDES,feet):
  # Follow own limb columns. No donor joint location or weight is reused.
  top_y=floor+height*.48
  near=np.linalg.norm((vertices-foot)[:,[0,2]],axis=1)<min(width*.35,height*.18)
  cross=vertices[near&(vertices[:,1]>top_y-height*.08)&(vertices[:,1]<top_y+height*.06)]
  hip=np.array([foot[0]*.8,top_y,foot[2]*.95])
  if len(cross)>8:hip[[0,2]]=np.median(cross[:,[0,2]],axis=0)
  ankle=foot+[0,height*.065,0]
  knee=(hip+ankle)/2;knee[2]+=height*(.06 if side.startswith("F") else -.08)
  toe=foot+[0,height*.012,-height*.01]
  toe_end=toe+[0,0,-height*.04]
  start=len(names)
  for bn,point,end,pi,dn in zip(("Hip","Knee","Ankle","Toe"),(hip,knee,ankle,toe),(knee,ankle,toe,toe_end),((2 if side.startswith("F") else 0),start,start+1,start+2),(*DONOR_CHAINS[side],DONOR_CHAINS[side][2])):
   names.append(bn+"_"+side);heads.append(point);tails.append(end);parents.append(pi);mappings.append(dn)
 return names,np.array(heads),np.array(tails),parents,mappings,feet,height,order

def envelope_weights(v,heads,tails):
 scores=[]
 scale=max(np.ptp(v[:,2]),np.ptp(v[:,1]))
 for h,t in zip(heads,tails):
  d=t-h;l=max(float(d@d),1e-12);u=np.clip((v-h)@d/l,0,1);nearest=h+u[:,None]*d
  distance=np.linalg.norm(v-nearest,axis=1);scores.append(np.exp(-3*(distance/max(scale*.14,1e-6))**2))
 w=np.array(scores).T+1e-20
 return w/w.sum(1,keepdims=True)

def main():
 args=sys.argv[sys.argv.index("--")+1:];ap=argparse.ArgumentParser();ap.add_argument("--request",type=pathlib.Path,required=True);a=ap.parse_args(args)
 req=json.loads(a.request.read_text());out=pathlib.Path(req["output"]);start=time.process_time()
 reference,donor_mats,source_frames,source_span,footmap=sample_donor(req["template"])
 bpy.ops.wm.read_factory_settings(use_empty=True)
 bpy.context.scene.render.threads_mode="FIXED";bpy.context.scene.render.threads=1
 bpy.ops.import_scene.gltf(filepath=req["rest_glb"],merge_vertices=False)
 meshes=[o for o in bpy.context.scene.objects if o.type=="MESH"]
 for o in meshes:
  bpy.ops.object.select_all(action="DESELECT");o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 binding_weld_counts=[]
 for o in meshes:
  if req["method"]!="heat":continue
  initial=len(o.data.vertices)
  bpy.ops.object.select_all(action="DESELECT");o.select_set(True);bpy.context.view_layer.objects.active=o
  bpy.ops.object.mode_set(mode="EDIT");bpy.ops.mesh.select_all(action="SELECT")
  bpy.ops.mesh.remove_doubles(threshold=2e-7)
  bpy.ops.object.mode_set(mode="OBJECT")
  binding_weld_counts.append({"original_vertices":initial,"binding_vertices":len(o.data.vertices),"purpose":"coincident seam weld in temporary Blender weight solve only; final GLB uses original vertices/faces"})
 vbl=np.concatenate([np.array([o.matrix_world@vv.co for vv in o.data.vertices]) for o in meshes])
 vertices=vbl@HAB_TO_BLENDER
 names,heads,tails,parents,mappings,feet,height,order=fit_bones(vertices,req["registration"])
 hbl=heads@HAB_TO_BLENDER.T;tbl=tails@HAB_TO_BLENDER.T
 data=bpy.data.armatures.new("OwnTargetSkeleton");rig=bpy.data.objects.new("OwnTargetRig",data);bpy.context.collection.objects.link(rig)
 bpy.ops.object.select_all(action="DESELECT");rig.select_set(True);bpy.context.view_layer.objects.active=rig
 bpy.ops.object.mode_set(mode="EDIT")
 for i,name in enumerate(names):
  bone=data.edit_bones.new(name);bone.head=Vector(hbl[i]);bone.tail=Vector(tbl[i])
  if (bone.tail-bone.head).length<1e-5:bone.tail=bone.head+Vector((0,0,.01))
  if parents[i]>=0:bone.parent=data.edit_bones[names[parents[i]]]
  bone.use_connect=False
 bpy.ops.object.mode_set(mode="OBJECT");bpy.context.view_layer.update()
 bind_bl=np.array([rig.matrix_world@data.bones[n].matrix_local for n in names])
 conv=np.eye(4);conv[:3,:3]=HAB_TO_BLENDER
 # C^-1 is C.T; glTF joint matrices are in Y-up Habitat coordinates.
 rest=np.array([conv.T@m for m in bind_bl])
 weights=[];heat_errors=[];weight_solve_counts=[]
 for o in meshes:
  bpy.ops.object.select_all(action="DESELECT");o.select_set(True);rig.select_set(True);bpy.context.view_layer.objects.active=rig
  if req["method"]=="heat":
   try:bpy.ops.object.parent_set(type="ARMATURE_AUTO")
   except RuntimeError as e:heat_errors.append(str(e))
  else:
   # Blender bone envelopes, fitted to this animal's measured dimensions.
   for i,bone in enumerate(rig.data.bones):
    radius=height*(.16 if bone.name in ("Body","Spine","Chest") else .11 if bone.name in ("Neck","Head") else .07)
    bone.head_radius=radius;bone.tail_radius=radius*.8;bone.envelope_distance=radius*1.6
   bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
  local=np.array([o.matrix_world@vv.co for vv in o.data.vertices])@HAB_TO_BLENDER
  w=np.zeros((len(local),len(names)))
  groups={g.index:names.index(g.name) for g in o.vertex_groups if g.name in names}
  for vi,vert in enumerate(o.data.vertices):
   for gg in vert.groups:
    if gg.group in groups:w[vi,groups[gg.group]]=gg.weight
  unweighted=w.sum(1)<1e-8
  weight_solve_counts.append({"mesh":o.name,"vertices":len(local),"unweighted_after_blender":int(unweighted.sum()),"envelope_fallback_vertices":int(unweighted.sum())})
  if np.any(unweighted):w[unweighted]=envelope_weights(local[unweighted],heads,tails)
  w/=w.sum(1,keepdims=True)
  weights.append(w)
 w=np.concatenate(weights)
 if req["method"] in ("envelope","blender_envelope"):
  repo=pathlib.Path(__file__).resolve().parents[2]
  import importlib.util
  spec=importlib.util.spec_from_file_location("quadruped_skin",repo/"src/avengine/assets/quadruped_skin.py")
  skinmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(skinmod)
  w=skinmod.capsule_skin_weights(vertices,names,heads,feet,height,req["registration"]["landmarks_local_m"]["head_bbox_min"],paw_only=req["method"]=="blender_envelope")
 before=w.copy();sole_count={};head_count=0
 separ=np.linalg.norm(feet[:,None,[0,2]]-feet[None,:,[0,2]],axis=2);np.fill_diagonal(separ,np.inf)
 distance=np.linalg.norm(vertices[:,None,[0,2]]-feet[None,:,[0,2]],axis=2);nearest=distance.argmin(1)
 for pi,side in enumerate(SIDES):
  radius=min(height*.14,separ[pi].min()*.40)
  sole=(nearest==pi)&(vertices[:,1]<feet[pi,1]+height*.095)&(distance[:,pi]<radius)
  if req["method"]=="heat":w[sole]=0;w[sole,names.index("Ankle_"+side)]=1;sole_count[side]=int(sole.sum())
  if not sole.any():raise ValueError("semantic_sole_empty:"+side)
 head=np.asarray(req["registration"]["head_center_local_m"]);mouth=np.asarray(req["registration"]["emit_point_local_m"])
 hb=req["registration"]["landmarks_local_m"];headmin=np.asarray(hb["head_bbox_min"]);headmax=np.asarray(hb["head_bbox_max"])
 headmask=np.all(vertices>=headmin,axis=1)&np.all(vertices<=headmax,axis=1)&(vertices[:,1]>headmin[1]+height*.02)
 mouthmask=np.linalg.norm(vertices-mouth,axis=1)<height*.09
 headmask|=mouthmask
 if req["method"]=="heat":w[headmask]=0;w[headmask,names.index("Head")]=1
 head_count=int(headmask.sum())
 selection=np.argsort(w,axis=1)[:,-4:];values=np.take_along_axis(w,selection,axis=1);values/=values.sum(1,keepdims=True)
 actions={"Idle":np.tile(rest,(len(source_frames),1,1,1))}
 amplitude=float(req.get("motion_amplitude",.10))
 if not 0<amplitude<=1:raise ValueError("motion_amplitude_out_of_range")
 torso_amplitude=0.0
 for action in ("Walking",):
  frames=[]
  for sf in range(len(source_frames)):
   g=[]
   for i,source_name in enumerate(mappings):
    mat=rest[i].copy()
    if parents[i]>=0:mat[:3,3]=(g[parents[i]]@np.linalg.inv(rest[parents[i]])@rest[i])[:3,3]
    d=unitrot(donor_mats[action][sf][source_name])@unitrot(reference[source_name]).T
    d=DONOR_TO_HAB@d@DONOR_TO_HAB.T
    q=Matrix(d.tolist()).to_quaternion()
    if q.w<0:q.negate()
    q=Quaternion((1,0,0,0)).slerp(q,amplitude if i>=8 else torso_amplitude)
    mat[:3,:3]=np.array(q.to_matrix())@rest[i][:3,:3]
    if i==0:
     # Remove donor horizontal root motion; motion compiler owns -Z travel.
     dy=donor_mats[action][sf][source_name][2,3]-reference[source_name][2,3]
     mat[1,3]+=float(dy)*height/max(source_span,1e-5)*torso_amplitude
    g.append(mat)
   # Distal Toe follows its ankle with exactly its own fitted rest offset.
   for side in SIDES:
    ti=names.index("Toe_"+side);ai=names.index("Ankle_"+side);g[ti]=g[ai]@np.linalg.inv(rest[ai])@rest[ti]
   frames.append(g)
  actions[action]=np.array(frames)
 np.savez_compressed(out/"binding.npz",positions=vertices,joints=selection.astype(np.uint16),weights=values.astype(np.float32),rest_matrices=rest,times=(source_frames-source_frames[0])/30,**actions)
 report={"schema":"blender_target_landmark_rig_v1","blender_version":bpy.app.version_string,"bone_names":names,"parent_indices":parents,"fitted_head_points_habitat_m":heads.tolist(),"fitted_tail_points_habitat_m":tails.tolist(),"semantic_paws":{s:feet[i].tolist() for i,s in enumerate(SIDES)},"registration_foot_order":order.tolist(),"target_front":"-Z","target_up":"+Y","donor_basis_to_habitat":DONOR_TO_HAB.tolist(),"donor_side_mapping_from_geometry":footmap,"source_donor_bone_map":dict(zip(names,mappings)),"motion_transfer":"world left rotation delta relative original CC0 Idle, forward source samples; fitted target bone lengths and own mesh","motion_amplitude":amplitude,"torso_head_tail_motion_amplitude":torso_amplitude,"torso_amplitude_reason":"retain fitted upper-body/Head/Tail Idle local pose after measured cross-region area rejection; CC0 Walking limb transfer and unchanged 48mm stride/8mm foot lift remain active","weights_method":req["method"],"paw_envelope_radius_height_fraction":.25 if req["method"]=="blender_envelope" else None,"envelope_postprocessing":"continuous_ankle_body_paw_envelope" if req["method"]=="blender_envelope" else "continuous_multijoint_capsule" if req["method"]=="envelope" else "heat_semantic_repairs","fallback_method":"normalized own fitted bone-segment Gaussian envelopes","heat_errors":heat_errors,"blender_weight_solve_counts":weight_solve_counts,"temporary_binding_mesh_weld":binding_weld_counts,"semantic_sole_vertices":sole_count,"head_vertices":head_count,"weight_fixups":"heat: semantic sole/head repair; envelope: smooth Voronoi-boundary multijoint capsules; blender_envelope: continuous distal paw/body envelopes without abrupt sole/head overwrites; both envelope recipes retain own Body/Head/Tail and <=4 influences without truncation","target_vertices":len(vertices),"deform_bones":len(names),"cpu_s":time.process_time()-start,"status":"candidate_not_admitted"}
 (out/"blender_binding.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print("BLENDER_BINDING",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
