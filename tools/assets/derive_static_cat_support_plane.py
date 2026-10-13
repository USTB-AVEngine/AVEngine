"""Derive a static cat using one rigid four-paw leveling transform.

The original binary mesh, UV, texture and material payloads are preserved.
All registration landmarks follow the same transform; no surface repair.
"""
import argparse,copy,hashlib,json,pathlib
import numpy as np
from avengine.assets.animal_motion.slide import StandingSlide,registration_record
from avengine.assets.roster_motion.io import read_glb,write_glb
from avengine.assets.roster_motion.contact import yaw_matrix

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--inventory",type=pathlib.Path,required=True);p.add_argument("--asset",required=True);p.add_argument("--output",type=pathlib.Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 record=copy.deepcopy(next(x for x in json.loads(a.inventory.read_text())["assets"] if x["asset_id"]==a.asset))
 row=copy.deepcopy(registration_record(record));row.pop("minor_variants",None)
 slide=StandingSlide(record);vertices=slide.vertices
 paws=np.array([vertices[i] for i in slide.foot_indices.values()])
 fit=np.linalg.lstsq(np.c_[paws[:,0],paws[:,2],np.ones(4)],paws[:,1],rcond=None)[0]
 normal=np.array([-fit[0],1.,-fit[1]]);normal/=np.linalg.norm(normal)
 up=np.array([0.,1.,0.]);cross=np.cross(normal,up);cos=float(normal@up)
 tilt=float(np.degrees(np.arccos(np.clip(cos,-1,1))))
 if tilt>10:raise ValueError("four_paw_support_plane_tilt_exceeds_10deg")
 k=np.array([[0.,-cross[2],cross[1]],[cross[2],0.,-cross[0]],[-cross[1],cross[0],0.]])
 level=np.eye(3)+k+k@k/(1+cos)
 leveled=vertices@level.T;shift=np.array([0.,.0004-leveled[:,1].min(),0.])
 residual=float(np.ptp((paws@level.T)[:,1]))
 if residual>.012:raise ValueError("four_paw_plane_residual_exceeds_12mm")
 rotation=level@yaw_matrix(slide.support_plane["heading_correction_rad"])
 matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=shift
 original=slide.rig.path;doc,binary=read_glb(original);assert not doc.get("skins") and not doc.get("animations")
 materials=copy.deepcopy(doc.get("materials"));meshes=copy.deepcopy(doc.get("meshes"))
 scene=doc["scenes"][doc.get("scene",0)];roots=list(scene["nodes"]);index=len(doc["nodes"])
 doc["nodes"].append({"name":"OwnedRigidFourPawLevel","matrix":matrix.T.ravel().tolist(),"children":roots});scene["nodes"]=[index]
 out=a.output/"static.glb";write_glb(out,doc,binary)
 d,b=read_glb(out);assert b==binary and d.get("materials")==materials and d.get("meshes")==meshes
 def point(v):return (rotation@np.asarray(v,float)+shift).tolist()
 for key in ("emit_point_local_m","head_center_local_m"):
  if key in row:row[key]=point(row[key])
 row["landmarks_local_m"]={name:point(value) for name,value in row.get("landmarks_local_m",{}).items()}
 if "source_static_metadata" in row and "mouth_m" in row["source_static_metadata"]:row["source_static_metadata"]["mouth_m"]=point(row["source_static_metadata"]["mouth_m"])
 low=(leveled+shift).min(0);high=(leveled+shift).max(0)
 row.update(runtime_glb=str(out),body_only_glb=str(out),bounds_min_m=low.tolist(),bounds_max_m=high.tolist(),dimensions_m=(high-low).tolist(),runtime_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),derived_from=str(original),walk_allowed=False)
 registry=a.output/"registration.json";registry.write_text(json.dumps({"animals":[row]},ensure_ascii=False,indent=2)+"\n")
 record["registry"]=str(registry);record["files"]={key:{**info,"path":str(out),"skins":0,"animations":[]} for key,info in record["files"].items() if key in ("runtime_glb","body_only_glb","habitat_glb")}
 record.update(has_skeleton=False,has_walking=False,registered_walk_allowed=False)
 inventory=a.output/"inventory.json";inventory.write_text(json.dumps({"assets":[record]},ensure_ascii=False,indent=2)+"\n")
 confirm=StandingSlide(record);foot_heights=[float(confirm.vertices[i,1]) for i in confirm.foot_indices.values()]
 geometry={"asset_id":a.asset,"method":"single rigid four-paw plane fit then global clearance translation","source":str(original),"source_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),"derived":str(out),"matrix":matrix.tolist(),"tilt_deg":tilt,"paw_residual_m":residual,"minimum_mesh_y_m":float(confirm.vertices[:,1].min()),"paw_y_m":foot_heights,"binary_payload_unchanged":True,"materials_unchanged":True,"intrinsic_size_change":False,"surface_repair":False,"native_status":"not_run","source_registry_modified":False}
 (a.output/"derivation.json").write_text(json.dumps(geometry,indent=2)+"\n")
 print(json.dumps(geometry,ensure_ascii=False),flush=True)
 if max(foot_heights)>.012:raise SystemExit(2)
if __name__=="__main__":main()
