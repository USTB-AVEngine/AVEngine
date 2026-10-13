"""Verify original geometry, UVs, PBR values and decoded texture pixels."""
import argparse,hashlib,io,json,pathlib
import numpy as np
from PIL import Image
from avengine.assets.roster_motion.io import read_glb,viewbytes
from avengine.assets.roster_motion.pose import pose,accessor
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def primitives(d):
 return [p for n in d["nodes"] if "mesh" in n for p in d["meshes"][n["mesh"]]["primitives"]]
def pixels(d,b):
 result=[]
 for image in d.get("images",[]):
  if "uri" in image:raise ValueError("external source images require explicit base path")
  result.append(np.array(Image.open(io.BytesIO(viewbytes(d,b,image["bufferView"]))).convert("RGBA")))
 return result
def check(source,derived):
 sd,sb=read_glb(source);dd,db=read_glb(derived)
 sp,_=pose(sd,sb,None,ground=False,canonical_flip=False);dp,_=pose(dd,db,None,ground=False,canonical_flip=False)
 sps,dps=primitives(sd),primitives(dd);errors=[];max_position=max_normal=0.
 if len(sps)!=len(dps):raise ValueError("primitive count changed")
 for i,(s,d,sv,dv) in enumerate(zip(sps,dps,sp,dp)):
  max_position=max(max_position,float(np.max(np.abs(sv["positions"]-dv["positions"]))))
  if sv["normals"] is not None and dv["normals"] is not None:max_normal=max(max_normal,float(np.max(np.abs(sv["normals"]-dv["normals"]))))
  for name in ("indices",):
   if (name in s)!=(name in d) or name in s and not np.array_equal(accessor(sd,sb,s[name]),accessor(dd,db,d[name])):errors.append(f"primitive_{i}_{name}_changed")
  if s.get("material")!=d.get("material") or s.get("mode",4)!=d.get("mode",4):errors.append(f"primitive_{i}_material_or_mode_changed")
  for name,aid in s["attributes"].items():
   if name.startswith("TEXCOORD"):
    if name not in d["attributes"] or not np.array_equal(accessor(sd,sb,aid),accessor(dd,db,d["attributes"][name])):errors.append(f"primitive_{i}_{name}_changed")
   if name.startswith("COLOR"):
    values=accessor(sd,sb,aid);dest=accessor(dd,db,d["attributes"][name]) if name in d["attributes"] else np.ones_like(values)
    if not np.allclose(values,dest,rtol=0,atol=1e-7):errors.append(f"primitive_{i}_{name}_changed")
 if max_position>2e-6:errors.append("rest_position_changed")
 if max_normal>2e-6:errors.append("rest_normals_changed")
 if sd.get("materials",[])!=dd.get("materials",[]):errors.append("PBR_material_values_changed")
 src,dst=pixels(sd,sb),pixels(dd,db)
 if len(src)!=len(dst) or any(not np.array_equal(a,b) for a,b in zip(src,dst)):errors.append("decoded_texture_pixels_changed")
 sbounds=np.array([np.concatenate([p["positions"] for p in sp]).min(0),np.concatenate([p["positions"] for p in sp]).max(0)])
 dbounds=np.array([np.concatenate([p["positions"] for p in dp]).min(0),np.concatenate([p["positions"] for p in dp]).max(0)])
 return {"status":"pass" if not errors else "fail","source_glb":str(source),"source_sha256":sha(source),"derived_glb":str(derived),"derived_sha256":sha(derived),
  "rest_position_max_error_m":max_position,"rest_normal_max_component_error":max_normal,"binding_bounds_max_error_m":float(np.max(np.abs(sbounds-dbounds))),
  "original_bounds_m":sbounds.tolist(),"derived_bounds_m":dbounds.tolist(),"primitive_count":len(sps),"decoded_images":len(src),
  "decoded_pixels_equal":not any("texture" in e for e in errors),"PBR_material_values_equal":sd.get("materials",[])==dd.get("materials",[]),"triangle_indices_and_UVs_unchanged":not any("indices" in e or "TEXCOORD" in e for e in errors),"failures":errors}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--rigging",type=pathlib.Path,required=True);p.add_argument("--output",type=pathlib.Path,required=True);a=p.parse_args()
 receipt=json.loads(a.rigging.read_text());result=check(receipt["source_glb"],receipt["motion_glb"])
 a.output.write_text(json.dumps(result,indent=2));print(result,flush=True)
 if result["status"]!="pass":raise SystemExit(2)
if __name__=="__main__":main()
