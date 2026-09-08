from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

import unreal

REQUEST_ENV = "AVENGINE_EXTRA_ANIMAL_UE_BLUEPRINT_REQUEST"
SCHEMA = "avengine_p11_extra_animal_ue_blueprint_request_v1"
RESULT_SCHEMA = "avengine_p11_extra_animal_ue_blueprint_result_v1"

def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def write_new(path: Path, value: dict[str, Any]):
    need(not path.exists() and not path.is_symlink(), "refusing output " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

def set_prop(obj, name, value):
    try:
        obj.set_editor_property(name, value)
    except Exception as error:
        raise RuntimeError("could not set property " + name + ": " + str(error)) from error

def component_from_blueprint(bp):
    cls = bp.generated_class()
    need(cls is not None, "Blueprint generated class missing")
    default = unreal.get_default_object(cls)
    need(default is not None, "Blueprint default object missing")
    for name in ("skeletal_mesh_component", "SkeletalMeshComponent"):
        try:
            comp = default.get_editor_property(name)
            if comp is not None:
                return comp
        except Exception:
            pass
    try:
        scs = bp.get_editor_property("simple_construction_script")
        for node in scs.get_editor_property("root_nodes"):
            comp = node.get_editor_property("component_template")
            if isinstance(comp, unreal.SkeletalMeshComponent):
                return comp
    except Exception:
        pass
    raise RuntimeError("Blueprint has no SkeletalMeshComponent")

def main():
    request_path = Path(os.environ["AVENGINE_EXTRA_ANIMAL_UE_BLUEPRINT_REQUEST"]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    need(request.get("schema") == SCHEMA, "request schema mismatch")
    output = Path(request["output"]).resolve()
    blueprint_dir = str(request["blueprint_directory"])
    need(blueprint_dir.startswith("/Game/"), "Blueprint directory must be /Game")
    need(not unreal.EditorAssetLibrary.does_directory_exist(blueprint_dir), "Blueprint directory already exists: " + blueprint_dir)
    need(unreal.EditorAssetLibrary.make_directory(blueprint_dir), "could not make Blueprint directory")
    imported = json.loads(Path(request["import_manifest"]).read_text(encoding="utf-8"))
    need(imported.get("status") == "pass" and imported.get("asset_count") == 4, "import manifest is not pass/4")
    assets = []
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    for entry in imported["assets"]:
        aid = entry["asset_id"]
        mesh_path = entry["content"]["skeletal_mesh"]
        mesh = unreal.load_asset(mesh_path)
        need(mesh is not None and isinstance(mesh, unreal.SkeletalMesh), aid + " mesh missing")
        name = "BP_" + aid
        factory = unreal.BlueprintFactory()
        set_prop(factory, "parent_class", unreal.SkeletalMeshActor)
        bp = asset_tools.create_asset(
            asset_name=name, package_path=blueprint_dir,
            asset_class=unreal.Blueprint,
            factory=factory)
        need(bp is not None, aid + " Blueprint create failed")
        factory = None
        try:
            factory = bp.get_editor_property("factory")
        except Exception:
            pass
        comp = component_from_blueprint(bp)
        # The generated class is SkeletalMeshActor; set only the imported mesh.
        # Runtime probes own the action selection and frame clock.
        try:
            comp.set_skeletal_mesh_asset(new_mesh=mesh)
            bind_method = "SkeletalMeshComponent.set_skeletal_mesh_asset"
        except Exception:
            set_prop(comp, "skeletal_mesh", mesh)
            bind_method = "SkeletalMeshComponent.skeletal_mesh_property"
        try:
            comp.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_SINGLE_NODE)
        except Exception:
            pass
        unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_loaded_asset(
            asset_to_save=bp
        )
        bp_path = blueprint_dir + "/" + name + "." + name
        cls_path = bp_path + "_C"
        # Reload the saved asset and verify its default component still carries
        # the exact imported mesh.
        saved = unreal.load_asset(bp_path)
        need(saved is not None, aid + " saved Blueprint reload failed")
        saved_comp = component_from_blueprint(saved)
        saved_mesh = saved_comp.get_editor_property("skeletal_mesh")
        need(saved_mesh is not None and saved_mesh.get_path_name() == mesh.get_path_name(),
             aid + " Blueprint mesh readback mismatch")
        assets.append({
            "asset_id": aid, "original_asset_id": entry.get("original_asset_id", aid),
            "skeletal_mesh": mesh_path, "blueprint_asset": bp_path,
            "blueprint_class_path": cls_path, "bind_method": bind_method,
            "status": "pass"})
    result = {
        "schema": RESULT_SCHEMA, "status": "pass",
        "research_only": True, "qualification_claim": False,
        "blueprint_directory": blueprint_dir,
        "import_manifest": str(Path(request["import_manifest"]).resolve()),
        "source_request": str(request_path), "asset_count": len(assets),
        "assets": assets,
        "claim_boundary": "Transient runtime probes may spawn these imported SkeletalMeshActor Blueprints; no map save or formal registry mutation."
    }
    write_new(output, result)
    unreal.log("AVENGINE_P11_EXTRA_ANIMAL_UE_BLUEPRINT_OK output=" + str(output))

if __name__ == "__main__":
    main()
