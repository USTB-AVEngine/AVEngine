"""Import four seated skeletal GLBs and create independent UE Blueprints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import posixpath
import re
from typing import Any

import unreal

from import_spear_3d_front_sample_editor import _import_glb


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_asset(path: str) -> Any:
    value = unreal.load_asset(name=path)
    _require(value is not None, f"could not load UE asset: {path}")
    return value


def _component_from_blueprint(blueprint: Any) -> Any:
    generated = blueprint.generated_class()
    _require(generated is not None, "Blueprint has no generated class")
    default = unreal.get_default_object(generated)
    _require(default is not None, "Blueprint generated class has no default object")
    for name in ("skeletal_mesh_component", "SkeletalMeshComponent"):
        try:
            component = default.get_editor_property(name)
        except Exception:
            continue
        if component is not None:
            return component
    raise RuntimeError("Blueprint has no SkeletalMeshComponent")


def _configure_component(component: Any, mesh: Any, animation: Any) -> None:
    component.set_animation_mode(animation_mode=unreal.AnimationMode.ANIMATION_SINGLE_NODE)
    component.set_skeletal_mesh_asset(new_mesh=mesh)
    play_data = unreal.SingleAnimationPlayData(
        anim_to_play=animation, saved_position=0.0, saved_play_rate=1.0
    )
    component.set_editor_property("animation_data", play_data)


def _create_blueprint(*, mesh_path: str, animation_path: str, blueprint_dir: str, blueprint_name: str) -> str:
    blueprint_path = posixpath.join(blueprint_dir, f"{blueprint_name}.{blueprint_name}")
    _require(
        not unreal.EditorAssetLibrary.does_asset_exist(blueprint_path),
        f"refusing to replace Blueprint: {blueprint_path}",
    )
    factory = unreal.BlueprintFactory()
    factory.set_editor_property("parent_class", unreal.SkeletalMeshActor)
    blueprint = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        asset_name=blueprint_name,
        package_path=blueprint_dir,
        asset_class=unreal.Blueprint,
        factory=factory,
    )
    _require(blueprint is not None, "could not create seated human Blueprint")
    _configure_component(_component_from_blueprint(blueprint), _load_asset(mesh_path), _load_asset(animation_path))
    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_loaded_asset(asset_to_save=blueprint)
    return blueprint_path


def _find_animation(assets: dict[str, list[str]], name: str) -> str:
    candidates = []
    for class_name, paths in assets.items():
        if "AnimSequence" in class_name or class_name == "AnimationSequence":
            candidates.extend(paths)
    matches = [path for path in candidates if path.rsplit("/", 1)[-1].split(".", 1)[0] == name]
    _require(len(matches) == 1, f"expected one {name} animation, found {matches}")
    return matches[0]


def main() -> None:
    request_value = os.environ.get("AVENGINE_SEATED_HUMAN_UE_IMPORT_REQUEST")
    _require(request_value is not None, "AVENGINE_SEATED_HUMAN_UE_IMPORT_REQUEST is required")
    request_path = Path(request_value).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    _require(request.get("kind") == "avengine_seated_human_ue_import_request_v1", "unsupported seated import request")
    content_root = str(request["content_root"])
    _require(re.fullmatch(r"/Game/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+", content_root) is not None, "unsafe content root")
    assets = request.get("assets")
    _require(isinstance(assets, list) and len(assets) == 4, "request must contain four seated assets")
    output = request_path.with_name("seated_human_ue_import_manifest.json")
    _require(not output.exists() and not output.is_symlink(), f"refusing to replace manifest: {output}")
    records = []
    seen = set()
    for item in assets:
        _require(isinstance(item, dict), "asset request must be an object")
        asset_id = str(item["asset_id"])
        _require(asset_id not in seen, f"duplicate asset ID: {asset_id}")
        seen.add(asset_id)
        source = Path(str(item["source_glb"])).expanduser().resolve()
        destination = str(item["destination"])
        _require(source.is_file() and source.suffix.casefold() == ".glb", f"missing GLB: {source}")
        _require(destination.startswith(content_root + "/"), f"destination outside content root: {destination}")
        imported = _import_glb(source, destination)
        mesh_paths = imported.get("SkeletalMesh", [])
        _require(len(mesh_paths) == 1, f"{asset_id} must import one SkeletalMesh")
        animation_path = _find_animation(imported, str(item.get("animation_name", "Seated_Idle")))
        blueprint_path = _create_blueprint(
            mesh_path=mesh_paths[0],
            animation_path=animation_path,
            blueprint_dir=f"{destination}/Blueprints",
            blueprint_name=f"BP_{asset_id}",
        )
        mesh = _load_asset(mesh_paths[0])
        bounds = mesh.get_bounds()
        records.append(
            {
                "asset_id": asset_id,
                "source_glb": str(source),
                "destination": destination,
                "skeletal_mesh": mesh_paths[0],
                "animation": animation_path,
                "blueprint": blueprint_path,
                "bounds_cm": {
                    "origin": [float(value) for value in bounds.origin],
                    "box_extent": [float(value) for value in bounds.box_extent],
                },
                "emitter_offset_avengine_m": item["emitter_offset_avengine_m"],
                "seat_reference": item["seat_reference"],
            }
        )
    output.write_text(
        json.dumps(
            {
                "kind": "avengine_seated_human_ue_import_manifest_v1",
                "status": "pass",
                "research_only": True,
                "assets": records,
                "claim_boundary": "UE skeletal seated-idle bindings; no formal admission or transition action",
            },
            indent=2,
            sort_keys=True,
        ) + chr(10),
        encoding="utf-8",
    )
    unreal.log_warning(f"AVENGINE_SEATED_HUMAN_UE_IMPORT_OK output={output}")


if __name__ == "__main__":
    main()
