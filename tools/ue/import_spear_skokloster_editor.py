"""Import the prepared Skokloster GLB into one isolated SPEAR/UE content root.

Run with UnrealEditor ``-run=pythonscript``.  A second fresh editor process
must set ``AVENGINE_SKOKLOSTER_VERIFY_ONLY=1`` to prove saved-asset and map
reload.  This script never captures frames and never touches GPU selection.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import unreal

REQUEST_SCHEMA = "avengine_skokloster_ue_import_request_v1"
RESULT_SCHEMA = "avengine_skokloster_ue_import_result_v1"
CONTENT_ROOT = "/Game/MyAssets/Audioset/Scenes/skokloster_castle"
MAP_PATH = f"{CONTENT_ROOT}/Maps/skokloster_castle_strict"
ACTOR_TAG = unreal.Name("avengine_skokloster_castle_surface")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"request is missing or not a direct file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("request root must be an object")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate(request: Mapping[str, Any]) -> Path:
    if request.get("schema") != REQUEST_SCHEMA:
        raise RuntimeError("Skokloster import request schema drift")
    if (
        request.get("content_root") != CONTENT_ROOT
        or request.get("map_path") != MAP_PATH
    ):
        raise RuntimeError("Skokloster import target is not the isolated reviewed root")
    if request.get("expected_static_mesh_asset_count") != 1:
        raise RuntimeError("Skokloster expected mesh count drift")
    coordinate = request.get("coordinate_contract", {})
    if (
        coordinate.get("source_to_habitat") != "H=(S.x,S.z,-S.y)"
        or coordinate.get("prepared_glb") != "canonical glTF metres, +Y up, -Z forward"
        or coordinate.get("runtime_actor_transform") != "identity"
    ):
        raise RuntimeError("Skokloster coordinate contract drift")
    source = Path(str(request.get("prepared_glb_path", ""))).resolve()
    if (
        not source.is_file()
        or source.is_symlink()
        or source.suffix.casefold() != ".glb"
    ):
        raise RuntimeError(f"prepared Skokloster GLB is missing: {source}")
    return source


def _class_name(asset_path: str) -> str:
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
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


def _asset_readback() -> tuple[dict[str, list[str]], dict[str, int]]:
    assets = _assets_by_class(f"{CONTENT_ROOT}/Imported")
    mesh_count = len(assets.get("StaticMesh", []))
    material_count = sum(
        len(assets.get(name, [])) for name in ("Material", "MaterialInstanceConstant")
    )
    texture_count = sum(
        len(assets.get(name, []))
        for name in ("Texture2D", "TextureCube", "VirtualTexture2D")
    )
    if mesh_count != 1 or material_count < 1 or texture_count < 1:
        raise RuntimeError(
            "Skokloster Interchange readback differs: "
            f"meshes={mesh_count} materials={material_count} textures={texture_count}"
        )
    return assets, {
        "static_mesh_assets": mesh_count,
        "material_assets": material_count,
        "texture_assets": texture_count,
    }


def _property(value: Any, name: str) -> Any:
    try:
        return value.get_editor_property(name)
    except (AttributeError, RuntimeError):
        return getattr(value, name)


def _vector3(value: Any) -> list[float]:
    result = [float(_property(value, axis)) for axis in ("x", "y", "z")]
    if not all(math.isfinite(component) for component in result):
        raise RuntimeError(f"non-finite UE vector readback: {result}")
    return result


def _rotator(value: Any) -> list[float]:
    result = [float(_property(value, axis)) for axis in ("pitch", "yaw", "roll")]
    if not all(math.isfinite(component) for component in result):
        raise RuntimeError(f"non-finite UE rotator readback: {result}")
    return result


def _mesh_geometry_readback(mesh_path: str) -> dict[str, Any]:
    mesh = unreal.load_asset(mesh_path)
    if mesh is None or not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError(f"cannot load StaticMesh for geometry readback: {mesh_path}")
    try:
        bounding_box = mesh.get_bounding_box()
        minimum = _vector3(_property(bounding_box, "min"))
        maximum = _vector3(_property(bounding_box, "max"))
        bounds_api = "UStaticMesh.get_bounding_box"
    except (AttributeError, RuntimeError):
        bounds = _property(mesh, "extended_bounds")
        origin = _vector3(_property(bounds, "origin"))
        extent = _vector3(_property(bounds, "box_extent"))
        minimum = [origin[index] - extent[index] for index in range(3)]
        maximum = [origin[index] + extent[index] for index in range(3)]
        bounds_api = "UStaticMesh.extended_bounds"
    size = [maximum[index] - minimum[index] for index in range(3)]
    if not all(math.isfinite(value) and value > 0.0 for value in size):
        raise RuntimeError(
            f"Skokloster StaticMesh has invalid local bounds: min={minimum} max={maximum}"
        )
    material_slots: list[dict[str, Any]] = []
    for index, slot in enumerate(list(_property(mesh, "static_materials"))):
        interface = _property(slot, "material_interface")
        material_slots.append(
            {
                "index": index,
                "slot_name": str(_property(slot, "material_slot_name")),
                "imported_slot_name": str(
                    _property(slot, "imported_material_slot_name")
                ),
                "material_object_path": (
                    str(interface.get_path_name()) if interface is not None else None
                ),
            }
        )
    if not material_slots or not any(
        slot["material_object_path"] for slot in material_slots
    ):
        raise RuntimeError("Skokloster StaticMesh material-slot readback is empty")
    return {
        "static_mesh_object_path": mesh_path,
        "local_bounds_cm": {
            "minimum": minimum,
            "maximum": maximum,
            "size": size,
            "readback_api": bounds_api,
        },
        "material_slot_count": len(material_slots),
        "material_slots": material_slots,
    }


def _actor_transform_readback(actor: Any) -> dict[str, list[float]]:
    return {
        "location_cm": _vector3(actor.get_actor_location()),
        "rotation_pitch_yaw_roll_degrees": _rotator(actor.get_actor_rotation()),
        "scale": _vector3(actor.get_actor_scale3d()),
    }


def _import(source: Path) -> dict[str, list[str]]:
    destination = f"{CONTENT_ROOT}/Imported"
    if not unreal.EditorAssetLibrary.make_directory(destination):
        raise RuntimeError(f"cannot create import destination: {destination}")
    task = unreal.AssetImportTask()
    task.set_editor_property("async_", False)
    task.set_editor_property("automated", True)
    task.set_editor_property("destination_path", destination)
    task.set_editor_property("filename", str(source))
    task.set_editor_property("replace_existing", False)
    task.set_editor_property("replace_existing_settings", False)
    task.set_editor_property("save", False)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    if not task.get_objects():
        raise RuntimeError("UE Interchange imported no Skokloster assets")
    unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
    subsystem = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
    subsystem.save_directory(
        directory_path=destination, only_if_is_dirty=False, recursive=True
    )
    return _asset_readback()[0]


def _spawn_surface(mesh_path: str) -> Any:
    mesh = unreal.load_asset(mesh_path)
    if mesh is None or not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError(f"cannot reload imported StaticMesh: {mesh_path}")
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        actor_class=unreal.StaticMeshActor,
        location=unreal.Vector(0.0, 0.0, 0.0),
        rotation=unreal.Rotator(),
        transient=False,
    )
    if actor is None:
        raise RuntimeError("cannot spawn Skokloster StaticMeshActor")
    component = actor.static_mesh_component
    component.set_static_mesh(mesh)
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    component.set_editor_property("cast_shadow", True)
    component.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
    actor.set_actor_label("AVEngine__skokloster_castle_surface", mark_dirty=False)
    actor.tags = [ACTOR_TAG, unreal.Name("habitat_navigation_authority")]
    return actor


def _content_filesystem_path() -> Path:
    project_content = Path(
        str(
            unreal.Paths.convert_relative_path_to_full(
                unreal.Paths.project_content_dir()
            )
        )
    ).resolve()
    return project_content.joinpath(*CONTENT_ROOT.removeprefix("/Game/").split("/"))


def _assert_content_root_absent(
    asset_registry_root_exists: bool, filesystem_root_exists: bool
) -> None:
    if asset_registry_root_exists or filesystem_root_exists:
        raise RuntimeError(
            "isolated Skokloster content root already exists; no-clobber import refused: "
            f"{CONTENT_ROOT}"
        )


def _assemble(request: Mapping[str, Any], source: Path) -> dict[str, Any]:
    root_exists = unreal.EditorAssetLibrary.does_directory_exist(CONTENT_ROOT)
    filesystem_root = _content_filesystem_path()
    _assert_content_root_absent(root_exists, filesystem_root.exists())
    assets = _import(source)
    meshes = sorted(assets.get("StaticMesh", []))
    geometry_readback = [_mesh_geometry_readback(path) for path in meshes]
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).new_level(MAP_PATH):
        raise RuntimeError(f"cannot create Skokloster map: {MAP_PATH}")
    actor = _spawn_surface(meshes[0])
    if not unreal.get_editor_subsystem(
        unreal.LevelEditorSubsystem
    ).save_current_level():
        raise RuntimeError(f"cannot save Skokloster map: {MAP_PATH}")
    readback_assets, counts = _asset_readback()
    counts["spawned_static_mesh_actors"] = 1
    return {
        "schema": RESULT_SCHEMA,
        "status": "pass",
        "mode": "import_and_assemble",
        "content_root": CONTENT_ROOT,
        "managed_content_filesystem_path": str(filesystem_root),
        "map_path": MAP_PATH,
        "counts": counts,
        "scene_content": {
            "static_mesh_count": 1,
            "static_meshes": sorted(readback_assets.get("StaticMesh", [])),
            "static_mesh_geometry_readback": geometry_readback,
            "class_counts": counts,
        },
        "actor_labels": [actor.get_actor_label()],
        "actor_transform_readback": _actor_transform_readback(actor),
        "coordinate_contract": dict(request["coordinate_contract"]),
        "collision": "NoCollision; Habitat navmesh remains navigation authority",
        "reload_verification": "pending_fresh_editor",
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _verify_only(request: Mapping[str, Any]) -> dict[str, Any]:
    readback_assets, counts = _asset_readback()
    filesystem_root = _content_filesystem_path()
    if not filesystem_root.is_dir():
        raise RuntimeError(
            f"fresh-reload filesystem content root is missing: {filesystem_root}"
        )
    meshes = sorted(readback_assets.get("StaticMesh", []))
    geometry_readback = [_mesh_geometry_readback(path) for path in meshes]
    if not unreal.EditorLevelLibrary.load_level(MAP_PATH):
        raise RuntimeError(f"cannot fresh-reload Skokloster map: {MAP_PATH}")
    mesh_actors = [
        actor
        for actor in unreal.EditorLevelLibrary.get_all_level_actors()
        if isinstance(actor, unreal.StaticMeshActor) and ACTOR_TAG in actor.tags
    ]
    if len(mesh_actors) != 1:
        raise RuntimeError(
            f"fresh map reload found {len(mesh_actors)} Skokloster actors"
        )
    counts["spawned_static_mesh_actors"] = len(mesh_actors)
    return {
        "schema": RESULT_SCHEMA,
        "status": "pass",
        "mode": "fresh_editor_verify_only",
        "content_root": CONTENT_ROOT,
        "managed_content_filesystem_path": str(filesystem_root),
        "map_path": MAP_PATH,
        "counts": counts,
        "scene_content": {
            "static_mesh_count": 1,
            "static_meshes": meshes,
            "static_mesh_geometry_readback": geometry_readback,
            "class_counts": counts,
        },
        "actor_labels": [actor.get_actor_label() for actor in mesh_actors],
        "actor_transform_readback": _actor_transform_readback(mesh_actors[0]),
        "coordinate_contract": dict(request["coordinate_contract"]),
        "reload_verification": "pass",
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def main() -> None:
    request_path = Path(os.environ["AVENGINE_SKOKLOSTER_IMPORT_REQUEST"]).resolve()
    result_path = Path(os.environ["AVENGINE_SKOKLOSTER_EDITOR_RESULT"]).resolve()
    request = _load(request_path)
    source = _validate(request)
    result = (
        _verify_only(request)
        if os.environ.get("AVENGINE_SKOKLOSTER_VERIFY_ONLY") == "1"
        else _assemble(request, source)
    )
    _write(result_path, result)
    unreal.log(
        "SKOKLOSTER_EDITOR_OK "
        f"mode={result['mode']} map={MAP_PATH} result={result_path}"
    )


if __name__ == "__main__":
    main()
