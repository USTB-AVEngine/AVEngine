"""Import and assemble the prepared ReplicaCAD apt_0 scene inside UE 5.5.

Run this file through UnrealEditor/``-run=pythonscript``.  It reads only the
prepared execution request and writes a result outside the UE project.  The
managed content root is the exact ``scene.content_root`` declared by that
request; replacement is opt-in and never targets the old SPEAR content tree.

Environment variables:

``AVENGINE_REPLICACAD_EXECUTION_REQUEST``
    Request produced with ``prepare_spear_replicacad_scene.py
    --prepared-glb-dir ...``.
``AVENGINE_REPLICACAD_EDITOR_RESULT``
    External JSON result path.
``AVENGINE_REPLICACAD_REPLACE_EXISTING``
    Set to ``1`` only for an intentional rebuild of the managed content root.
``AVENGINE_REPLICACAD_REUSE_IMPORTED``
    Set to ``1`` to validate and reuse a complete previously imported Meshes
    subtree while rebuilding only the comparison map.  This is useful after an
    editor-script failure that happened after the 101 GLBs were saved.
``AVENGINE_REPLICACAD_VERIFY_ONLY``
    Set to ``1`` in a second editor process to reload assets/map without writes.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import unreal


REQUEST_SCHEMA = "avengine_optional_spear_replicacad_execution_v1"
RESULT_SCHEMA = "avengine_optional_spear_replicacad_editor_result_v1"
EXPECTED_COUNTS = {
    "logical_import_count": 87,
    "source_glb_count": 101,
    "expected_imported_static_mesh_asset_count": 127,
    "logical_instance_count": 120,
    "logical_instances_by_kind": {"stage": 1, "rigid": 113, "articulated": 6},
    "expected_runtime_mesh_actor_count": 171,
    "articulated_visual_occurrence_count": 31,
}
DATASET_LIGHT_LUMENS_PER_SCALED_UNIT = 250.0
DATASET_LIGHT_ATTENUATION_RADIUS_CM = 650.0


def _load_json(path: Path, owner: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{owner} is missing or not a direct file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {owner}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{owner} root must be an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, ensure_ascii=False, sort_keys=True
    ) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_request(request: Mapping[str, Any]) -> tuple[str, str]:
    if request.get("schema") != REQUEST_SCHEMA or request.get("counts") != EXPECTED_COUNTS:
        raise RuntimeError("ReplicaCAD apt_0 execution request closure differs")
    if request.get("backend_role") != "comparison_visual":
        raise RuntimeError("ReplicaCAD backend role differs")
    authority = request.get("authority", {})
    if authority.get("backend_may_replan") is not False:
        raise RuntimeError("ReplicaCAD UE backend must not replan")
    preparation = request.get("glb_preparation", {})
    if (
        preparation.get("status") != "pass"
        or preparation.get("source_glb_count") != 101
        or preparation.get("prepared_mesh_asset_count") != 127
    ):
        raise RuntimeError("ReplicaCAD prepared GLB closure differs")
    scene = request.get("scene", {})
    content_root = scene.get("content_root")
    if (
        not isinstance(content_root, str)
        or not re.fullmatch(r"/Game/AVEngine/Optional/ReplicaCAD/[A-Za-z0-9_]+", content_root)
    ):
        raise RuntimeError(f"unsafe ReplicaCAD managed content root: {content_root!r}")
    map_path = f"{content_root}/Maps/apt_0_comparison"
    return content_root, map_path


def _class_name(asset_path: str) -> str:
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
    class_path = data.get_editor_property("asset_class_path")
    return str(class_path.get_editor_property("asset_name"))


def _assets_by_class(directory: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in sorted(
        str(value)
        for value in unreal.EditorAssetLibrary.list_assets(
            directory_path=directory, recursive=True, include_folder=False
        )
    ):
        result.setdefault(_class_name(path), []).append(path)
    return result


def _import_sources(request: Mapping[str, Any]) -> dict[str, list[str]]:
    imported: dict[str, list[str]] = {}
    for source in request["pbr_import"]["source_meshes"]:
        source_id = source["mesh_source_id"]
        prepared = Path(source["editor_import_source_glb_path"]).resolve()
        if not prepared.is_file() or prepared.is_symlink():
            raise RuntimeError(f"prepared ReplicaCAD GLB is missing: {prepared}")
        destination = source["destination_content_path"]
        if unreal.EditorAssetLibrary.does_directory_exist(destination):
            raise RuntimeError(f"ReplicaCAD import destination already exists: {destination}")
        if not unreal.EditorAssetLibrary.make_directory(destination):
            raise RuntimeError(f"cannot create ReplicaCAD import destination: {destination}")
        task = unreal.AssetImportTask()
        task.set_editor_property("async_", True)
        task.set_editor_property("automated", True)
        task.set_editor_property("destination_path", destination)
        task.set_editor_property("filename", str(prepared))
        task.set_editor_property("replace_existing", False)
        task.set_editor_property("replace_existing_settings", False)
        task.set_editor_property("save", False)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        if not task.get_objects():
            raise RuntimeError(f"UE Interchange imported no assets from {prepared}")
        unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
        unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_directory(
            directory_path=destination,
            only_if_is_dirty=False,
            recursive=True,
        )
        assets = _assets_by_class(destination)
        static_meshes = sorted(assets.get("StaticMesh", []))
        expected = int(source["source_inventory"]["mesh_count"])
        if len(static_meshes) != expected:
            raise RuntimeError(
                f"{source_id} imported {len(static_meshes)} StaticMeshes, expected {expected}"
            )
        imported[source_id] = static_meshes
    return imported


def _load_imported_sources(request: Mapping[str, Any]) -> dict[str, list[str]]:
    imported: dict[str, list[str]] = {}
    for source in request["pbr_import"]["source_meshes"]:
        assets = _assets_by_class(source["destination_content_path"])
        static_meshes = sorted(assets.get("StaticMesh", []))
        expected = int(source["source_inventory"]["mesh_count"])
        if len(static_meshes) != expected:
            raise RuntimeError(
                f"reloaded {source['mesh_source_id']} has {len(static_meshes)} "
                f"StaticMeshes, expected {expected}"
            )
        imported[source["mesh_source_id"]] = static_meshes
    return imported


def _unreal_transform(value: Mapping[str, Any]) -> unreal.Transform:
    translation = value["translation_cm"]
    rotation = value["rotation_xyzw"]
    scale = value["scale_xyz"]
    numbers = [float(item) for item in (*translation, *rotation, *scale)]
    if not all(math.isfinite(item) for item in numbers) or min(scale) <= 0.0:
        raise RuntimeError("ReplicaCAD UE transform is invalid")
    # In UE 5.5 the generated ``Transform(...)`` Python constructor is routed
    # through KismetMathLibrary.MakeTransform and therefore expects a Rotator,
    # even though the actual FTransform ``rotation`` editor property is a Quat.
    # Populate the native properties explicitly so the authoritative quaternion
    # is preserved without a lossy or version-specific constructor conversion.
    transform = unreal.Transform()
    transform.set_editor_property(
        "translation",
        unreal.Vector(x=translation[0], y=translation[1], z=translation[2]),
    )
    transform.set_editor_property(
        "rotation",
        unreal.Quat(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3]),
    )
    transform.set_editor_property(
        "scale3d",
        unreal.Vector(x=scale[0], y=scale[1], z=scale[2]),
    )
    return transform


def _actor_tag(value: str) -> unreal.Name:
    return unreal.Name(value.replace(" ", "_"))


def _spawn_mesh_actors(
    request: Mapping[str, Any], imported: Mapping[str, list[str]]
) -> tuple[list[str], int]:
    logical_ids: list[str] = []
    actor_count = 0
    for spawn in request["spawns"]:
        spawn_id = spawn["spawn_id"]
        logical_ids.append(spawn_id)
        observed_for_spawn = 0
        for visual in spawn["visual_instances"]:
            mesh_paths = imported[visual["mesh_source_id"]]
            transform = _unreal_transform(visual["world_transform_ue"])
            for mesh_index, mesh_path in enumerate(mesh_paths):
                mesh = unreal.load_asset(mesh_path)
                if mesh is None or not isinstance(mesh, unreal.StaticMesh):
                    raise RuntimeError(f"cannot load imported StaticMesh: {mesh_path}")
                actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                    actor_class=unreal.StaticMeshActor,
                    location=transform.get_editor_property("translation"),
                    # The complete quaternion is applied immediately below.
                    # Supplying an identity Rotator here avoids another UE 5.5
                    # Quat/Rotator Python-binding ambiguity during the spawn.
                    rotation=unreal.Rotator(),
                    transient=False,
                )
                if actor is None:
                    raise RuntimeError(f"cannot spawn StaticMeshActor for {mesh_path}")
                if not actor.set_actor_transform(transform, sweep=False, teleport=True):
                    # set_actor_transform returns False when no movement occurred on
                    # some UE versions; verify through the transform readback below.
                    pass
                component = actor.static_mesh_component
                component.set_static_mesh(mesh)
                component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
                component.set_editor_property("cast_shadow", True)
                component.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
                label = (
                    f"AVEngine__{spawn_id}__{visual['visual_id']}__mesh_{mesh_index:03d}"
                ).replace(":", "_")
                actor.set_actor_label(label, mark_dirty=False)
                actor.tags = [
                    _actor_tag("avengine_comparison_visual"),
                    _actor_tag(f"spawn_id={spawn_id}"),
                    _actor_tag(f"asset_kind={spawn['asset_kind']}"),
                    _actor_tag(f"visual_id={visual['visual_id']}"),
                ]
                observed_for_spawn += 1
                actor_count += 1
        if observed_for_spawn != int(spawn["expected_mesh_actor_count"]):
            raise RuntimeError(
                f"{spawn_id} spawned {observed_for_spawn} mesh actors, expected "
                f"{spawn['expected_mesh_actor_count']}"
            )
    return logical_ids, actor_count


def _spawn_dataset_lights(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for value in request["lighting"]["lights"]:
        scaled = float(value["dataset_scaled_intensity"])
        if scaled < 0.0:
            continue
        position = value["ue_position_cm"]
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            actor_class=unreal.PointLight,
            location=unreal.Vector(*position),
            rotation=unreal.Rotator(),
            transient=False,
        )
        if actor is None:
            raise RuntimeError(f"cannot spawn dataset point light {value['light_id']}")
        component = actor.point_light_component
        component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        component.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
        lumens = scaled * DATASET_LIGHT_LUMENS_PER_SCALED_UNIT
        component.set_intensity(lumens)
        component.set_attenuation_radius(DATASET_LIGHT_ATTENUATION_RADIUS_CM)
        component.set_light_color(unreal.LinearColor(*value["color_rgb"], 1.0))
        component.set_cast_shadows(True)
        actor.set_actor_label(f"AVEngine__dataset_light_{value['light_id']}", mark_dirty=False)
        actor.tags = [
            _actor_tag("avengine_dataset_light"),
            _actor_tag("blocks_source_center=false"),
        ]
        records.append(
            {
                "light_id": value["light_id"],
                "source_scaled_intensity": scaled,
                "ue_intensity_lumens": lumens,
                "cast_shadows": True,
                "blocks_source_center": False,
            }
        )
    return records


def _pbr_readback(content_root: str) -> dict[str, Any]:
    assets = _assets_by_class(f"{content_root}/Meshes")
    material_count = sum(
        len(assets.get(name, [])) for name in ("Material", "MaterialInstanceConstant")
    )
    texture_count = sum(
        len(assets.get(name, []))
        for name in ("Texture2D", "TextureCube", "VirtualTexture2D")
    )
    if material_count <= 0 or texture_count <= 0:
        raise RuntimeError("ReplicaCAD imported PBR material/texture readback is empty")
    return {
        "material_asset_count": material_count,
        "texture_asset_count": texture_count,
        "material_overrides_applied": False,
    }


def _assemble(request: Mapping[str, Any], content_root: str, map_path: str) -> dict[str, Any]:
    replace = os.environ.get("AVENGINE_REPLICACAD_REPLACE_EXISTING") == "1"
    reuse_imported = os.environ.get("AVENGINE_REPLICACAD_REUSE_IMPORTED") == "1"
    if replace and reuse_imported:
        raise RuntimeError(
            "ReplicaCAD replace and reuse-imported modes are mutually exclusive"
        )
    if unreal.EditorAssetLibrary.does_directory_exist(content_root):
        if reuse_imported:
            imported = _load_imported_sources(request)
        elif not replace:
            raise RuntimeError(
                f"managed ReplicaCAD content already exists; set explicit replace flag: {content_root}"
            )
        else:
            if not unreal.EditorAssetLibrary.delete_directory(content_root):
                raise RuntimeError(f"cannot delete managed ReplicaCAD content: {content_root}")
            imported = _import_sources(request)
    else:
        if reuse_imported:
            raise RuntimeError(
                f"cannot reuse missing ReplicaCAD managed content: {content_root}"
            )
        imported = _import_sources(request)
    if unreal.EditorAssetLibrary.does_asset_exist(map_path):
        if not reuse_imported:
            raise RuntimeError(f"unexpected existing ReplicaCAD map: {map_path}")
        if not unreal.EditorAssetLibrary.delete_asset(map_path):
            raise RuntimeError(f"cannot replace incomplete ReplicaCAD map: {map_path}")
    if not unreal.EditorLevelLibrary.new_level(map_path):
        raise RuntimeError(f"cannot create ReplicaCAD comparison map: {map_path}")
    logical_ids, actor_count = _spawn_mesh_actors(request, imported)
    light_records = _spawn_dataset_lights(request)
    if not unreal.EditorLevelLibrary.save_current_level():
        raise RuntimeError(f"cannot save ReplicaCAD comparison map: {map_path}")
    return {
        "schema": RESULT_SCHEMA,
        "status": "pass",
        "counts": {
            "imported_source_glb_count": len(imported),
            "imported_static_mesh_asset_count": sum(len(value) for value in imported.values()),
            "logical_instance_count": len(logical_ids),
            "logical_instances_by_kind": dict(request["counts"]["logical_instances_by_kind"]),
            "spawned_static_mesh_actor_count": actor_count,
            "articulated_visual_occurrence_count": request["counts"]["articulated_visual_occurrence_count"],
        },
        "logical_spawn_ids": logical_ids,
        "pbr_readback": _pbr_readback(content_root),
        "map": {
            "object_path": map_path,
            "mesh_collision": "NoCollision",
            "navigation_and_source_center_authority": "Habitat-native AVEngine",
        },
        "lighting": {
            "positive_dataset_light_count": len(light_records),
            "negative_dataset_lights": "recorded in request, not representable as UE point lights",
            "calibration_lumens_per_dataset_scaled_unit": DATASET_LIGHT_LUMENS_PER_SCALED_UNIT,
            "attenuation_radius_cm": DATASET_LIGHT_ATTENUATION_RADIUS_CM,
            "records": light_records,
        },
        "reload_verification": "not_run",
    }


def _verify_only(
    request: Mapping[str, Any], content_root: str, map_path: str
) -> dict[str, Any]:
    imported = _load_imported_sources(request)
    if not unreal.EditorLevelLibrary.load_level(map_path):
        raise RuntimeError(f"cannot reload ReplicaCAD comparison map: {map_path}")
    actors = unreal.EditorLevelLibrary.get_all_level_actors()
    mesh_actors = [
        actor
        for actor in actors
        if isinstance(actor, unreal.StaticMeshActor)
        and unreal.Name("avengine_comparison_visual") in actor.tags
    ]
    dataset_lights = [
        actor
        for actor in actors
        if isinstance(actor, unreal.PointLight)
        and unreal.Name("avengine_dataset_light") in actor.tags
    ]
    if len(mesh_actors) != 171 or len(dataset_lights) != 5:
        raise RuntimeError(
            f"ReplicaCAD map reload counts differ: meshes={len(mesh_actors)} "
            f"lights={len(dataset_lights)}"
        )
    logical_ids = [spawn["spawn_id"] for spawn in request["spawns"]]
    result = {
        "schema": RESULT_SCHEMA,
        "status": "pass",
        "counts": {
            "imported_source_glb_count": len(imported),
            "imported_static_mesh_asset_count": sum(len(value) for value in imported.values()),
            "logical_instance_count": len(logical_ids),
            "logical_instances_by_kind": dict(request["counts"]["logical_instances_by_kind"]),
            "spawned_static_mesh_actor_count": len(mesh_actors),
            "articulated_visual_occurrence_count": request["counts"]["articulated_visual_occurrence_count"],
        },
        "logical_spawn_ids": logical_ids,
        "pbr_readback": _pbr_readback(content_root),
        "map": {"object_path": map_path, "reloaded": True},
        "lighting": {"positive_dataset_light_count": len(dataset_lights)},
        "reload_verification": "pass",
    }
    return result


def main() -> None:
    request_path = Path(os.environ["AVENGINE_REPLICACAD_EXECUTION_REQUEST"]).resolve()
    result_path = Path(os.environ["AVENGINE_REPLICACAD_EDITOR_RESULT"]).resolve()
    request = _load_json(request_path, "ReplicaCAD execution request")
    content_root, map_path = _validate_request(request)
    if os.environ.get("AVENGINE_REPLICACAD_VERIFY_ONLY") == "1":
        result = _verify_only(request, content_root, map_path)
    else:
        result = _assemble(request, content_root, map_path)
    _write_json(result_path, result)
    unreal.log(
        "SPEAR_REPLICACAD_EDITOR_OK "
        f"meshes={result['counts']['spawned_static_mesh_actor_count']} "
        f"map={map_path} result={result_path}"
    )


if __name__ == "__main__":
    main()
