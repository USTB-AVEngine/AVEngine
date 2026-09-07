"""Import research-only animated animal GLBs into isolated UE SkeletalMesh roots.

The request is explicit and data driven.  This helper only imports and
read-backs source-supplied SkeletalMesh/Skeleton/AnimSequence assets; it does
not mutate the shared registry, create a map, spawn an actor, or claim dataset
admission.  Run inside UnrealEditor with:
  -run=pythonscript -script=<this file>
and AVENGINE_EXTRA_ANIMAL_UE_REQUEST pointing to a JSON request.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import unreal

REQUEST_ENV = "AVENGINE_EXTRA_ANIMAL_UE_REQUEST"
SCHEMA = "avengine_p11_extra_animal_ue_import_request_v1"
RESULT_SCHEMA = "avengine_p11_extra_animal_ue_import_result_v1"
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_]+$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label}: {path}") from error
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _source_path(value: str, request_path: Path) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = request_path.parent / raw
    raw = raw.resolve()
    require(raw.is_file() and not raw.is_symlink() and raw.suffix.casefold() == ".glb",
            f"source GLB is missing or unsafe: {raw}")
    return raw


def _class_name(path: str) -> str:
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=path)
    require(data is not None, f"could not read AssetData: {path}")
    class_path = data.get_editor_property("asset_class_path")
    return str(class_path.get_editor_property("asset_name"))


def _assets_by_class(directory: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    paths = unreal.EditorAssetLibrary.list_assets(
        directory_path=directory, recursive=True, include_folder=False
    )
    for path in sorted(str(value) for value in paths):
        result.setdefault(_class_name(path), []).append(path)
    return result


def _import_glb(source: Path, destination: str) -> dict[str, list[str]]:
    require(not unreal.EditorAssetLibrary.does_directory_exist(destination),
            f"refusing existing UE directory: {destination}")
    require(unreal.EditorAssetLibrary.make_directory(destination),
            f"could not create UE directory: {destination}")
    task = unreal.AssetImportTask()
    for name, value in (
        ("async_", False), ("automated", True), ("destination_path", destination),
        ("filename", str(source)), ("replace_existing", False),
        ("replace_existing_settings", False), ("save", False),
    ):
        try:
            task.set_editor_property(name, value)
        except Exception:
            if name in {"replace_existing_settings", "async_"}:
                continue
            raise
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    objects = task.get_objects()
    require(objects, f"UE Interchange imported no objects: {source}")
    unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_directory(
        directory_path=destination, only_if_is_dirty=False, recursive=True
    )
    return _assets_by_class(destination)


def _asset(path: str) -> Any:
    value = unreal.load_asset(path)
    require(value is not None, f"could not reload UE asset: {path}")
    return value


def _path(value: Any) -> str:
    getter = getattr(value, "get_path_name", None)
    require(callable(getter), f"asset has no path name: {value!r}")
    return str(getter())


def _vector(value: Any, names: Sequence[str], label: str) -> list[float]:
    values = []
    for name in names:
        try:
            item = value.get_editor_property(name)
        except Exception:
            item = getattr(value, name)
        values.append(float(item))
    require(all(math.isfinite(item) for item in values), f"{label} is non-finite")
    return values


def _animation_candidates(animations: Mapping[str, str], semantic: str) -> list[str]:
    needle = semantic.casefold()
    return sorted(
        path for name, path in animations.items()
        if name.casefold() == needle or name.casefold().endswith("_" + needle)
    )


def _bone_count(mesh: Any) -> int | None:
    for method_name in ("get_ref_skeleton",):
        method = getattr(mesh, method_name, None)
        if not callable(method):
            continue
        try:
            reference = method()
        except Exception:
            continue
        for name in ("get_raw_bone_num", "get_num", "get_num_bones"):
            getter = getattr(reference, name, None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:
                    continue
    return None


def _mesh_readback(mesh_path: str, skeleton_path: str, animation_paths: Mapping[str, str]) -> dict[str, Any]:
    mesh = _asset(mesh_path)
    skeleton = _asset(skeleton_path)
    require(isinstance(mesh, unreal.SkeletalMesh), f"not a SkeletalMesh: {mesh_path}")
    mesh_skeleton = mesh.get_editor_property("skeleton")
    require(mesh_skeleton is not None and _path(mesh_skeleton) == _path(skeleton),
            f"SkeletalMesh references wrong Skeleton: {mesh_path}")
    animations = {}
    for semantic, path in animation_paths.items():
        sequence = _asset(path)
        sequence_skeleton = sequence.get_editor_property("skeleton")
        require(sequence_skeleton is not None and _path(sequence_skeleton) == _path(skeleton),
                f"AnimSequence {semantic} references wrong Skeleton: {path}")
        try:
            length = float(sequence.get_editor_property("sequence_length"))
        except Exception:
            length = float(sequence.get_editor_property("sequence_length")) if hasattr(sequence, "sequence_length") else 0.0
        require(math.isfinite(length) and length > 0.0, f"AnimSequence {semantic} has invalid length: {length}")
        animations[semantic] = {
            "object_path": path,
            "skeleton": _path(sequence_skeleton),
            "sequence_length_seconds": length,
        }
    bounds = mesh.get_imported_bounds()
    origin = _vector(bounds.origin, ("x", "y", "z"), f"{mesh_path} bounds origin")
    extent = _vector(bounds.box_extent, ("x", "y", "z"), f"{mesh_path} bounds extent")
    require(all(value > 0.0 for value in extent), f"{mesh_path} has degenerate bounds")
    materials = []
    for slot in mesh.get_editor_property("materials"):
        interface = slot.get_editor_property("material_interface")
        require(interface is not None, f"{mesh_path} has null material slot")
        materials.append({
            "slot_name": str(slot.get_editor_property("material_slot_name")),
            "material_path": _path(interface),
        })
    return {
        "skeletal_mesh": mesh_path,
        "skeleton": skeleton_path,
        "bone_count": _bone_count(mesh),
        "bounds": {
            "origin_cm": origin,
            "box_extent_cm": extent,
            "minimum_cm": [origin[i] - extent[i] for i in range(3)],
            "maximum_cm": [origin[i] + extent[i] for i in range(3)],
        },
        "materials": materials,
        "animations": animations,
    }


def _inspect_asset(entry: Mapping[str, Any], destination: str) -> dict[str, Any]:
    classes = _assets_by_class(destination)
    skeletal = classes.get("SkeletalMesh", [])
    skeletons = classes.get("Skeleton", [])
    anim_records = {}
    for path in classes.get("AnimSequence", []):
        name = Path(path).name.rsplit(".", 1)[0]
        require(name not in anim_records, f"duplicate AnimSequence name {name}: {destination}")
        anim_records[name] = path
    require(len(skeletal) == 1, f"{entry['asset_id']}: expected one SkeletalMesh, got {skeletal}")
    require(len(skeletons) == 1, f"{entry['asset_id']}: expected one Skeleton, got {skeletons}")
    expected = entry.get("expected_animation_names", ["Idle", "Walking"])
    require(isinstance(expected, list) and set(expected) == {"Idle", "Walking"},
            f"{entry['asset_id']}: expected animations must be Idle and Walking")
    selected = {}
    for semantic in ("Idle", "Walking"):
        candidates = _animation_candidates(anim_records, semantic)
        require(len(candidates) == 1,
                f"{entry['asset_id']}: expected one {semantic} AnimSequence, candidates={candidates}, all={sorted(anim_records)}")
        selected[semantic] = candidates[0]
    content = _mesh_readback(skeletal[0], skeletons[0], selected)
    content["assets_by_class"] = classes
    content["animation_name_by_semantic"] = {
        semantic: Path(path).name.rsplit(".", 1)[0]
        for semantic, path in selected.items()
    }
    return content


def _request_items(request: Mapping[str, Any], request_path: Path) -> list[dict[str, Any]]:
    require(request.get("schema") == SCHEMA, f"request schema must be {SCHEMA}")
    namespace = request.get("namespace")
    require(isinstance(namespace, str) and namespace.startswith("/Game/"),
            "request namespace must be a /Game/ path")
    segments = namespace.removeprefix("/Game/").split("/")
    require(segments and all(_SAFE_SEGMENT.fullmatch(segment) for segment in segments),
            "request namespace contains unsafe segment")
    assets = request.get("assets")
    require(isinstance(assets, list) and assets, "request assets must be a non-empty list")
    seen_ids: set[str] = set()
    seen_destinations: set[str] = set()
    result = []
    for item in assets:
        require(isinstance(item, Mapping), "request asset entry must be an object")
        aid = item.get("asset_id")
        require(isinstance(aid, str) and _SAFE_SEGMENT.fullmatch(aid) and aid not in seen_ids,
                f"asset_id is unsafe or duplicated: {aid!r}")
        source = item.get("source_glb")
        require(isinstance(source, str), f"{aid}: source_glb is required")
        destination = item.get("destination")
        expected_destination = f"{namespace}/{aid}"
        require(destination == expected_destination,
                f"{aid}: destination must be {expected_destination}")
        require(destination not in seen_destinations, f"destination duplicated: {destination}")
        require(not unreal.EditorAssetLibrary.does_directory_exist(destination),
                f"{aid}: destination already exists in UE: {destination}")
        seen_ids.add(aid)
        seen_destinations.add(destination)
        result.append({
            "asset_id": aid, "source_glb": str(_source_path(source, request_path)),
            "destination": destination,
            "expected_animation_names": item.get("expected_animation_names", ["Idle", "Walking"]),
            "external_asset_json": item.get("external_asset_json"),
            "original_asset_id": item.get("original_asset_id", aid),
        })
    return result


def main() -> None:
    request_value = os.environ.get(REQUEST_ENV)
    require(request_value, f"{REQUEST_ENV} is required")
    request_path = Path(request_value).expanduser().resolve()
    request = _read_json(request_path, "UE import request")
    require(isinstance(request, Mapping), "UE import request must be an object")
    output_value = request.get("output")
    require(isinstance(output_value, str), "request.output is required")
    output = Path(output_value).expanduser()
    if not output.is_absolute():
        output = (request_path.parent / output).resolve()
    else:
        output = output.resolve()
    items = _request_items(request, request_path)
    imported = []
    stages = []
    try:
        for entry in items:
            _import_glb(Path(entry["source_glb"]), entry["destination"])
            content = _inspect_asset(entry, entry["destination"])
            imported.append({
                "asset_id": entry["asset_id"],
                "original_asset_id": entry["original_asset_id"],
                "source_glb": entry["source_glb"],
                "external_asset_json": entry["external_asset_json"],
                "destination": entry["destination"],
                "content": content,
            })
            stages.append({"asset_id": entry["asset_id"], "status": "pass"})
    except BaseException as error:
        failure = {
            "schema": RESULT_SCHEMA, "status": "fail",
            "research_only": True, "qualification_claim": False,
            "namespace": request["namespace"],
            "failed_asset": entry.get("asset_id") if "entry" in locals() else None,
            "error": f"{type(error).__name__}: {error}",
            "completed_assets": imported, "stages": stages,
            "source_request": str(request_path),
        }
        _write_new(output.with_suffix(".failure.json"), failure)
        unreal.log_error(f"AVENGINE_P11_EXTRA_ANIMAL_UE_IMPORT_FAILED output={output} error={error}")
        raise
    result = {
        "schema": RESULT_SCHEMA, "status": "pass",
        "research_only": True, "qualification_claim": False,
        "namespace": request["namespace"], "asset_count": len(imported),
        "producer": str(Path(__file__).resolve()),
        "source_request": str(request_path), "stages": stages,
        "assets": imported,
        "claim_boundary": (
            "UE skeletal/animation binding is a research candidate. "
            "No formal registry mutation, actor spawn, map save, or dataset admission."
        ),
    }
    _write_new(output, result)
    unreal.log(f"AVENGINE_P11_EXTRA_ANIMAL_UE_IMPORT_OK output={output} assets={len(imported)}")


if __name__ == "__main__":
    main()
