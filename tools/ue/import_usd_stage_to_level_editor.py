"""Import an external USD stage into ordinary Unreal assets and a saved level.

Run through Unreal Editor with ``AVENGINE_USD_LEVEL_IMPORT_REQUEST`` pointing
to an absolute JSON request.  Unlike ``create_spear_kujiale_map_editor.py``,
this tool does not leave an ``AUsdStageActor`` that needs the USD SDK at game
runtime: the editor importer materializes assets under a requested /Game path
and saves the resulting actors into a new level.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import unreal


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


request_value = os.environ.get("AVENGINE_USD_LEVEL_IMPORT_REQUEST")
require(bool(request_value), "AVENGINE_USD_LEVEL_IMPORT_REQUEST is not set")
request_path = Path(str(request_value)).expanduser().resolve()
require(request_path.is_file(), f"request does not exist: {request_path}")
request = json.loads(request_path.read_text(encoding="utf-8"))

source_usd = Path(str(request.get("source_usd", ""))).expanduser().resolve()
map_path = str(request.get("map_path", ""))
destination_path = str(request.get("destination_path", ""))
result_path = Path(str(request.get("result_path", ""))).expanduser().resolve()
require(
    source_usd.is_file() and source_usd.suffix.lower() in {".usd", ".usda", ".usdc"},
    f"source_usd is not a USD file: {source_usd}",
)
for owner, value in (("map_path", map_path), ("destination_path", destination_path)):
    require(value.startswith("/Game/"), f"{owner} must start with /Game/")
require(
    not unreal.EditorAssetLibrary.does_asset_exist(map_path),
    f"map already exists: {map_path}",
)
require(
    not unreal.EditorAssetLibrary.does_directory_exist(destination_path),
    f"destination already exists: {destination_path}",
)

level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
require(level_subsystem.new_level(map_path), f"could not create level: {map_path}")

options = unreal.UsdStageImportOptions()
option_values = {
    "import_actors": True,
    "import_geometry": True,
    "import_materials": True,
    "import_level_sequences": False,
    "import_sounds": False,
    "import_groom_assets": False,
    "import_skeletal_animations": False,
    "import_sparse_volume_textures": False,
    "import_only_used_materials": True,
    "merge_identical_material_slots": True,
    "prim_path_folder_structure": True,
    "replace_existing": False,
}
applied_options = {}
for name, value in option_values.items():
    try:
        options.set_editor_property(name, value)
    except Exception as error:
        if name == "replace_existing":
            # Replacement policy is controlled by AssetImportTask on UE 5.5;
            # not every UsdStageImportOptions build exposes this bool.
            continue
        raise RuntimeError(f"could not set USD import option {name}: {error}") from error
    applied_options[name] = value

factory = unreal.UsdStageImportFactory()
task = unreal.AssetImportTask()
task.set_editor_property("filename", str(source_usd))
task.set_editor_property("destination_path", destination_path)
task.set_editor_property("automated", True)
task.set_editor_property("save", True)
task.set_editor_property("replace_existing", False)
task.set_editor_property("replace_existing_settings", False)
task.set_editor_property("factory", factory)
task.set_editor_property("options", options)

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
asset_tools.import_asset_tasks([task])
imported_paths = [str(value) for value in task.get_editor_property("imported_object_paths")]
require(imported_paths, "USD stage import created no object paths")
require(level_subsystem.save_current_level(), f"could not save level: {map_path}")
unreal.EditorAssetLibrary.save_directory(destination_path, only_if_is_dirty=False)

actors = unreal.EditorLevelLibrary.get_all_level_actors()
actor_classes = {}
for actor in actors:
    name = actor.get_class().get_name()
    actor_classes[name] = actor_classes.get(name, 0) + 1
usd_stage_actor_count = int(actor_classes.get("UsdStageActor", 0))
require(
    usd_stage_actor_count == 0,
    f"imported level still contains {usd_stage_actor_count} UsdStageActor(s)",
)

asset_paths = unreal.EditorAssetLibrary.list_assets(
    destination_path, recursive=True, include_folder=False
)
asset_classes = {}
for object_path in asset_paths:
    data = unreal.EditorAssetLibrary.find_asset_data(object_path)
    class_name = str(data.asset_class_path.asset_name)
    asset_classes[class_name] = asset_classes.get(class_name, 0) + 1
static_mesh_count = int(asset_classes.get("StaticMesh", 0))
require(static_mesh_count > 0, "USD stage import created no StaticMesh assets")
require(len(actors) > 0, "USD stage import created no level actors")

result = {
    "schema": "avengine_editor_usd_level_import_v1",
    "status": "pass",
    "research_only": True,
    "qualification_claim": False,
    "source_usd": str(source_usd),
    "map_path": map_path,
    "destination_path": destination_path,
    "imported_object_path_count": len(imported_paths),
    "imported_object_paths_sample": imported_paths[:100],
    "level_actor_count": len(actors),
    "level_actor_classes": dict(sorted(actor_classes.items())),
    "destination_asset_count": len(asset_paths),
    "destination_asset_classes": dict(sorted(asset_classes.items())),
    "static_mesh_count": static_mesh_count,
    "usd_stage_actor_count": usd_stage_actor_count,
    "applied_options": applied_options,
    "claim_boundary": (
        "editor-materialized Unreal assets for internal research packaging; "
        "does not admit a scene, question, or dataset sample"
    ),
}
result_path.parent.mkdir(parents=True, exist_ok=True)
require(
    not result_path.exists() and not result_path.is_symlink(),
    f"refusing to replace result: {result_path}",
)
result_path.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
unreal.log_warning(f"AVENGINE_USD_LEVEL_IMPORT_OK result={result_path}")
