"""Build a Blender-fitted derivative without replacing the target mesh."""
from __future__ import annotations
import argparse, copy, json, pathlib, subprocess, sys, time
import numpy as np
from scipy.spatial import cKDTree
from avengine.assets.roster_motion.io import read_glb, write_glb
from avengine.assets.roster_motion.pose import pose, rigid_glb, accessor
from avengine.assets.roster_motion.contact import serialize_action

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--targets",type=pathlib.Path,required=True)
 ap.add_argument("--asset",required=True)
 ap.add_argument("--template",type=pathlib.Path,required=True)
 ap.add_argument("--blender",type=pathlib.Path,required=True)
 ap.add_argument("--output",type=pathlib.Path,required=True)
 ap.add_argument("--weights",choices=("heat","envelope","blender_envelope"),default="heat")
 ap.add_argument("--geometry-derivative",type=pathlib.Path)
 ap.add_argument("--motion-amplitude",type=float,default=.10)
 a=ap.parse_args()
 target=next(x for x in json.loads(a.targets.read_text())["assets"] if x["asset_id"]==a.asset)
 a.output.mkdir(parents=True,exist_ok=False)
 row=target["registration"];source=pathlib.Path(target["inventory_record"]["files"]["runtime_glb"]["path"])
 geometry_source=a.geometry_derivative or source
 d,b=read_glb(geometry_source)
 if d.get("skins"):raise ValueError("target_already_has_skin")
 parts,_=pose(d,b,None,ground=False,canonical_flip=False)
 rigid_glb(d,b,parts,a.output/"rest.glb")
 vertices=np.concatenate([p["positions"] for p in parts])
 np.savez_compressed(a.output/"source_vertices.npz",positions=vertices)
 req={"asset_id":a.asset,"source":str(source),"geometry_source":str(geometry_source),"rest_glb":str(a.output/"rest.glb"),"template":str(a.template),"output":str(a.output),"registration":row,"method":a.weights,"motion_amplitude":a.motion_amplitude}
 request=a.output/"request.json";request.write_text(json.dumps(req,ensure_ascii=False,indent=2))
 cmd=[str(a.blender),"-b","--factory-startup","--disable-autoexec","-t","1","--python-exit-code","2","--python",str(pathlib.Path(__file__).with_name("blender_rig_static_quadruped.py")),"--","--request",str(request)]
 print("BLENDER",cmd,flush=True)
 run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
 print(run.stdout,flush=True)
 if run.returncode:raise RuntimeError("Blender failed: "+str(run.returncode))
 blender_warnings=[line for line in run.stdout.splitlines() if "Bone Heat Weighting" in line]
 (a.output/"blender_stdout.txt").write_text(run.stdout)
 report=json.loads((a.output/"blender_binding.json").read_text())
 report["blender_heat_warnings"]=blender_warnings
 (a.output/"blender_binding.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 z=np.load(a.output/"binding.npz");fitted=z["positions"]
 distance,order=cKDTree(fitted).query(vertices,k=1)
 if distance.max()>2e-6:raise ValueError("blender_target_geometry_correspondence_failed:"+str(distance.max()))
 weights=z["weights"][order];joints=z["joints"][order]
 nd,nb=read_glb(a.output/"rest.glb");binary=bytearray(nb)
 def add(x,kind,component=5126):
  dtype={5126:np.float32,5123:np.uint16}[component];v=np.asarray(x,dtype=dtype)
  binary.extend(bytes((-len(binary))%4));view=len(nd["bufferViews"]);nd["bufferViews"].append({"buffer":0,"byteOffset":len(binary),"byteLength":v.nbytes});binary.extend(v.tobytes());idx=len(nd["accessors"]);ac={"bufferView":view,"componentType":component,"count":len(v),"type":kind}
  if kind=="SCALAR":ac.update(min=[float(v.min())],max=[float(v.max())])
  nd["accessors"].append(ac);return idx
 names=report["bone_names"];parents=report["parent_indices"];rest=z["rest_matrices"]
 from scipy.spatial.transform import Rotation
 nodes=[]
 for i,name in enumerate(names):
  matrix=rest[i] if parents[i]<0 else np.linalg.inv(rest[parents[i]])@rest[i]
  scale=np.linalg.norm(matrix[:3,:3],axis=0)
  nodes.append({"name":name,"translation":matrix[:3,3].tolist(),"rotation":Rotation.from_matrix(matrix[:3,:3]/scale).as_quat().tolist(),"scale":scale.tolist()})
 roots=[]
 for i,pi in enumerate(parents):
  if pi<0:roots.append(i)
  else:nodes[pi].setdefault("children",[]).append(i)
 offset=0
 for oldnode in nd["nodes"]:
  if "mesh" not in oldnode:continue
  idx=len(nodes);nodes.append({"name":"OwnTargetMesh_"+str(oldnode["mesh"]),"mesh":oldnode["mesh"],"skin":0});roots.append(idx)
  for prim in nd["meshes"][oldnode["mesh"]]["primitives"]:
   count=nd["accessors"][prim["attributes"]["POSITION"]]["count"]
   prim["attributes"]["JOINTS_0"]=add(joints[offset:offset+count],"VEC4",5123)
   prim["attributes"]["WEIGHTS_0"]=add(weights[offset:offset+count],"VEC4")
   offset+=count
 assert offset==len(vertices)
 nd["nodes"]=nodes;nd["scenes"]=[{"nodes":roots}];nd["scene"]=0
 ibm=np.linalg.inv(rest)
 nd["skins"]=[{"name":"TargetLandmarkFittedQuaterniusStructure","joints":list(range(len(names))),"inverseBindMatrices":add(ibm.transpose(0,2,1).reshape(-1,16),"MAT4"),"skeleton":parents.index(-1)}]
 nd["buffers"][0]["byteLength"]=len(binary);nb=bytes(binary)
 times=z["times"]
 for action in ("Idle","Walking"):
  mat=z[action]
  globals_frames=[{**{i:mat[fi,i] for i in range(len(names))},**{i:np.eye(4) for i in roots if i>=len(names)}} for fi in range(len(times))]
  nd,nb=serialize_action(nd,nb,action,times,globals_frames)
 nd.setdefault("asset",{}).setdefault("extras",{}).update({"target_asset_id":a.asset,"donor_license":"CC0-1.0","donor_geometry_copied":False,"source_geometry":str(source)})
 dest=a.output/"motion.glb";write_glb(dest,nd,nb)
 bound=pose(nd,nb,None,ground=False,canonical_flip=False)[0]
 err=float(np.max(np.abs(vertices-np.concatenate([p["positions"] for p in bound]))))
 if err>2e-6:raise ValueError("rest_skin_changed_geometry:"+str(err))
 manifest={"schema":"owned_target_blender_rig_candidate","anchors":[{"anchor_id":"body","joint_id":"Body"},{"anchor_id":"head","joint_id":"Head"},*[{"anchor_id":n,"joint_id":"Toe_"+s} for s,n in zip(("FL","FR","HL","HR"),("paw_front_left","paw_front_right","paw_hind_left","paw_hind_right"))]],"template_license":"CC0-1.0"}
 mp=a.output/"manifest.json";mp.write_text(json.dumps(manifest,indent=2))
 head_index=names.index("Head")
 mouth=np.asarray(row["emit_point_local_m"])
 emitter=(np.linalg.inv(rest[head_index])@np.r_[mouth,1])[:3]
 candidate={"rotation":np.eye(3).tolist(),"translation_m":[0,0,0],"scale_xyz":[1,1,1],"emitter_head_local_m":emitter.tolist(),"source_package_manifest":str(mp),"motion_runtime_glb":str(dest)}
 cp=a.output/"candidate.json";cp.write_text(json.dumps(candidate,indent=2))
 record=copy.deepcopy(target["inventory_record"]);record.update(has_skeleton=True,has_walking=True,walking_candidate=str(cp));record["files"]["motion_runtime_glb"]={"path":str(dest),"skins":1,"joint_count":len(names),"animations":["Idle","Walking"]}
 (a.output/"inventory.json").write_text(json.dumps({"assets":[record]},indent=2))
 receipt={"asset_id":a.asset,"status":"rigged_candidate_not_admitted","source_glb":str(source),"motion_glb":str(dest),"donor_template":str(a.template),"template_license":"CC0-1.0","rest_vertex_max_error_m":err,"blender_vertex_correspondence_max_error_m":float(distance.max()),"vertex_count":len(vertices),"joints":len(names),"triangle_topology_preserved_relative_to_geometry_source":True,"original_triangle_topology_preserved":a.geometry_derivative is None,"target_materials_preserved_relative_to_geometry_source":True,"geometry_source":str(geometry_source),"geometry_repair_used":a.geometry_derivative is not None,"blender":report,"inventory":str(a.output/"inventory.json"),"mouth_head_local_m":emitter.tolist()}
 (a.output/"rigging.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
 print({k:v for k,v in receipt.items() if k!="blender"},flush=True)
if __name__=="__main__":main()
