from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Any

import unreal

REQUEST_ENV = "AVENGINE_EXTRA_ANIMAL_UE_VERIFY_REQUEST"
SCHEMA = "avengine_p11_extra_animal_ue_verify_request_v1"
RESULT_SCHEMA = "avengine_p11_extra_animal_ue_verify_result_v1"

def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def path_of(value, label):
    fn=getattr(value, "get_path_name", None)
    need(callable(fn), label + " has no path name")
    return str(fn())

def default_component(bp):
    cls=bp.generated_class()
    need(cls is not None, "Blueprint generated class missing")
    obj=unreal.get_default_object(cls)
    need(obj is not None, "Blueprint CDO missing")
    for name in ("skeletal_mesh_component","SkeletalMeshComponent"):
        try:
            value=obj.get_editor_property(name)
            if value is not None:
                return value
        except Exception:
            pass
    scs=bp.get_editor_property("simple_construction_script")
    for node in scs.get_editor_property("root_nodes"):
        value=node.get_editor_property("component_template")
        if isinstance(value, unreal.SkeletalMeshComponent):
            return value
    raise RuntimeError("Blueprint has no SkeletalMeshComponent")

def main():
    req_path=Path(os.environ["AVENGINE_EXTRA_ANIMAL_UE_VERIFY_REQUEST"]).resolve()
    req=json.loads(req_path.read_text(encoding="utf-8"))
    need(req.get("schema")==SCHEMA, "verify request schema mismatch")
    output=Path(req["output"]).resolve()
    need(not output.exists() and not output.is_symlink(), "refusing output " + str(output))
    imp=json.loads(Path(req["import_manifest"]).read_text(encoding="utf-8"))
    bps=json.loads(Path(req["blueprint_manifest"]).read_text(encoding="utf-8"))
    need(imp.get("status")=="pass" and imp.get("asset_count")==4, "import manifest is not pass/4")
    need(bps.get("status")=="pass" and bps.get("asset_count")==4, "Blueprint manifest is not pass/4")
    bp_by_id={x["asset_id"]:x for x in bps["assets"]}
    rows=[]
    for entry in imp["assets"]:
        aid=entry["asset_id"]
        content=entry["content"]
        mesh_path=content["skeletal_mesh"]
        skeleton_path=content["skeleton"]
        mesh=unreal.load_asset(mesh_path)
        skeleton=unreal.load_asset(skeleton_path)
        need(mesh is not None and isinstance(mesh, unreal.SkeletalMesh), aid + " SkeletalMesh reload failed")
        need(skeleton is not None and isinstance(skeleton, unreal.Skeleton), aid + " Skeleton reload failed")
        mesh_skeleton=mesh.get_editor_property("skeleton")
        need(mesh_skeleton is not None and path_of(mesh_skeleton, aid+" mesh skeleton")==path_of(skeleton, aid+" skeleton"), aid+" mesh skeleton mismatch")
        ref_pose=skeleton.get_reference_pose()
        names=[str(x) for x in ref_pose.get_bone_names()]
        need(names and "bone_0" in names, aid+" fresh skeleton bone readback missing")
        animations={}
        for semantic in ("Idle","Walking"):
            path=content["animations"][semantic]["object_path"]
            sequence=unreal.load_asset(path)
            need(sequence is not None and isinstance(sequence, unreal.AnimSequence), aid+" "+semantic+" reload failed")
            seq_skeleton=sequence.get_editor_property("skeleton")
            need(seq_skeleton is not None and path_of(seq_skeleton, aid+" "+semantic+" skeleton")==path_of(skeleton, aid+" "+semantic+" skeleton"), aid+" "+semantic+" skeleton mismatch")
            length=float(sequence.get_editor_property("sequence_length"))
            need(math.isfinite(length) and length>0, aid+" "+semantic+" length invalid")
            animations[semantic]={"object_path":path,"sequence_length_seconds":length,"skeleton":path_of(seq_skeleton, aid+" "+semantic+" skeleton")}
        bp_path=bp_by_id[aid]["blueprint_asset"]
        bp=unreal.load_asset(bp_path)
        need(bp is not None, aid+" Blueprint reload failed")
        component=default_component(bp)
        bp_mesh=component.get_editor_property("skeletal_mesh")
        need(bp_mesh is not None and path_of(bp_mesh, aid+" Blueprint mesh")==path_of(mesh, aid+" mesh"), aid+" Blueprint mesh mismatch")
        bounds=mesh.get_imported_bounds()
        extent=[float(bounds.box_extent.x),float(bounds.box_extent.y),float(bounds.box_extent.z)]
        need(all(math.isfinite(x) and x>0 for x in extent), aid+" fresh bounds invalid")
        rows.append({
            "asset_id":aid,
            "skeletal_mesh":mesh_path,
            "skeleton":skeleton_path,
            "fresh_bone_count":len(names),
            "fresh_bone_names":names,
            "animations":animations,
            "blueprint_asset":bp_path,
            "blueprint_mesh_readback":path_of(bp_mesh, aid+" Blueprint mesh"),
            "bounds_origin_cm":[float(bounds.origin.x),float(bounds.origin.y),float(bounds.origin.z)],
            "bounds_extent_cm":extent,
            "status":"pass"})
    result={"schema":RESULT_SCHEMA,"status":"pass","research_only":True,"qualification_claim":False,
            "asset_count":len(rows),"import_manifest":str(Path(req["import_manifest"]).resolve()),
            "blueprint_manifest":str(Path(req["blueprint_manifest"]).resolve()),"source_request":str(req_path),
            "assets":rows,
            "claim_boundary":"Fresh separate UnrealEditor reload of four imported SkeletalMesh, Skeleton, Idle, Walking and Blueprint bindings; no map save, registry mutation, or admission."}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    unreal.log("AVENGINE_P11_EXTRA_ANIMAL_UE_FRESH_RELOAD_OK output="+str(output))
if __name__=="__main__":
    main()
