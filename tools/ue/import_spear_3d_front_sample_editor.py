"""Import the local 3D-FRONT Toolbox sample proxy into a persistent UE map.

Run through ``UnrealEditor -run=pythonscript`` with
``AVENGINE_3D_FRONT_MAP_REQUEST`` pointing at an absolute JSON request.  The
input GLB is a local derived review proxy; neither it nor the original dataset
is copied into the AVEngine repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

import unreal


SCHEMA = "avengine_optional_3d_front_sample_ue_map_v1"


def _asset_class_name(asset_path: str) -> str:
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
    class_path = data.get_editor_property("asset_class_path")
    return str(class_path.get_editor_property("asset_name"))


def _assets_by_class(directory: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in sorted(
        str(value)
        for value in unreal.EditorAssetLibrary.list_assets(
            directory_path=directory,
            recursive=True,
            include_folder=False,
        )
    ):
        result.setdefault(_asset_class_name(path), []).append(path)
    return result


def _import_glb(glb_path: Path, destination: str) -> dict[str, list[str]]:
    if unreal.EditorAssetLibrary.does_directory_exist(destination):
        raise RuntimeError(f"import destination already exists: {destination}")
    if not unreal.EditorAssetLibrary.make_directory(destination):
        raise RuntimeError(f"could not create import destination: {destination}")
    task = unreal.AssetImportTask()
    task.set_editor_property("async_", False)
    task.set_editor_property("automated", True)
    task.set_editor_property("destination_path", destination)
    task.set_editor_property("filename", str(glb_path))
    task.set_editor_property("replace_existing", False)
    task.set_editor_property("replace_existing_settings", False)
    task.set_editor_property("save", False)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    if not task.get_objects():
        raise RuntimeError(f"UE Interchange imported no assets from {glb_path}")
    unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_directory(
        directory_path=destination,
        only_if_is_dirty=False,
        recursive=True,
    )
    return _assets_by_class(destination)


def _spawn_static_meshes(mesh_paths: list[str]) -> list[str]:
    labels = []
    for index, mesh_path in enumerate(mesh_paths):
        mesh = unreal.load_asset(mesh_path)
        if mesh is None or not isinstance(mesh, unreal.StaticMesh):
            raise RuntimeError(f"cannot load imported StaticMesh: {mesh_path}")
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            actor_class=unreal.StaticMeshActor,
            location=unreal.Vector(0.0, 0.0, 0.0),
            rotation=unreal.Rotator(),
            transient=False,
        )
        if actor is None:
            raise RuntimeError(f"cannot spawn StaticMeshActor for {mesh_path}")
        actor.static_mesh_component.set_static_mesh(mesh)
        actor.static_mesh_component.set_collision_enabled(
            unreal.CollisionEnabled.NO_COLLISION
        )
        actor.static_mesh_component.set_editor_property("cast_shadow", True)
        actor.static_mesh_component.set_editor_property(
            "mobility", unreal.ComponentMobility.STATIC
        )
        label = f"AVEngine_3D_FRONT_sample_mesh_{index:03d}"
        actor.set_actor_label(label, mark_dirty=False)
        actor.tags = [
            unreal.Name("avengine_comparison_visual"),
            unreal.Name("dataset=3D_FRONT_FUTURE_official_toolbox_sample"),
        ]
        labels.append(label)
    return labels


def main() -> None:
    request_value = os.environ.get("AVENGINE_3D_FRONT_MAP_REQUEST")
    if not request_value:
        raise RuntimeError("AVENGINE_3D_FRONT_MAP_REQUEST is not set")
    request_path = Path(request_value).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    glb_path = Path(request["glb_path"]).expanduser().resolve()
    result_path = Path(request["result_path"]).expanduser().resolve()
    content_root = str(request["content_root"])
    map_path = str(request["map_path"])
    if not glb_path.is_file() or glb_path.suffix.casefold() != ".glb":
        raise RuntimeError(f"derived GLB does not exist: {glb_path}")
    if not re.fullmatch(
        r"/Game/AVEngine/Optional/ThreeDFront/[A-Za-z0-9_]+", content_root
    ):
        raise RuntimeError(f"unsafe managed content root: {content_root!r}")
    if not map_path.startswith(f"{content_root}/Maps/"):
        raise RuntimeError("map_path must be inside content_root/Maps")
    if unreal.EditorAssetLibrary.does_asset_exist(map_path):
        raise RuntimeError(f"map already exists: {map_path}")

    assets = _import_glb(glb_path, f"{content_root}/Scene")
    meshes = sorted(assets.get("StaticMesh", []))
    textures = sorted(assets.get("Texture2D", []))
    materials = sorted(
        assets.get("Material", []) + assets.get("MaterialInstanceConstant", [])
    )
    if len(meshes) < 9 or not textures or not materials:
        raise RuntimeError(
            "3D-FRONT sample import closure failed: "
            f"meshes={len(meshes)} materials={len(materials)} textures={len(textures)}"
        )

    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_subsystem.new_level(map_path):
        raise RuntimeError(f"could not create map: {map_path}")
    labels = _spawn_static_meshes(meshes)
    if not level_subsystem.save_current_level():
        raise RuntimeError(f"could not save map: {map_path}")

    result = {
        "schema": SCHEMA,
        "status": "pass",
        "backend_role": "comparison_visual",
        "scene_id": str(request["scene_id"]),
        "map_path": map_path,
        "content_root": content_root,
        "source_glb": str(glb_path),
        "counts": {
            "static_mesh_assets": len(meshes),
            "material_assets": len(materials),
            "texture_assets": len(textures),
            "spawned_static_mesh_actors": len(labels),
        },
        "actor_labels": labels,
        "external_asset_embedded_in_repository": False,
        "claim_boundary": (
            "Persistent UE assets imported from a local derived official Toolbox "
            "five-object sample proxy; not a complete 3D-FRONT house."
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    unreal.log_warning(f"AVENGINE_3D_FRONT_MAP_OK result={result_path}")


if __name__ == "__main__":
    main()
