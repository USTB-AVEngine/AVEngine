"""Import an external USD stage into ordinary Unreal assets and a saved level.

Run through Unreal Editor with ``AVENGINE_USD_LEVEL_IMPORT_REQUEST`` pointing
to an absolute JSON request.  Unlike ``create_spear_kujiale_map_editor.py``,
this tool does not leave an ``AUsdStageActor`` that needs the USD SDK at game
runtime: the editor importer materializes assets under a requested /Game path
and saves the resulting actors into a new level.
"""

from __future__ import annotations

import json
import math
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
raw_lights = request.get("visual_lights", [])
require(isinstance(raw_lights, list), "visual_lights must be a list")
visual_lights = []
light_ids = set()
for index, raw in enumerate(raw_lights):
    require(isinstance(raw, dict), f"visual_lights[{index}] must be an object")
    light_id = raw.get("light_id")
    require(
        isinstance(light_id, str) and light_id and light_id not in light_ids,
        f"visual_lights[{index}].light_id must be unique and nonempty",
    )
    light_ids.add(light_id)
    position = raw.get("position_ue_cm")
    require(
        isinstance(position, list)
        and len(position) == 3
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in position
        ),
        f"visual_lights[{index}].position_ue_cm is invalid",
    )
    record = {
        "light_id": light_id,
        "position_ue_cm": [float(value) for value in position],
    }
    for name in (
        "intensity_lumens",
        "attenuation_radius_cm",
        "temperature_kelvin",
        "source_radius_cm",
        "soft_source_radius_cm",
    ):
        value = raw.get(name)
        require(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
            and (name in {"source_radius_cm", "soft_source_radius_cm"} or float(value) > 0.0),
            f"visual_lights[{index}].{name} is invalid",
        )
        record[name] = float(value)
    visual_lights.append(record)

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
light_readbacks = []
for light in visual_lights:
    position = light["position_ue_cm"]
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.PointLight,
        unreal.Vector(position[0], position[1], position[2]),
        unreal.Rotator(0.0, 0.0, 0.0),
    )
    require(actor is not None, f"could not spawn visual light {light['light_id']}")
    actor.set_actor_label(f"AVEngine_{light['light_id']}")
    actor.tags = ["avengine_visual_only_light"]
    component = actor.get_component_by_class(unreal.PointLightComponent)
    require(component is not None, f"visual light {light['light_id']} lacks component")
    component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    component.set_editor_property("intensity", light["intensity_lumens"])
    component.set_editor_property("attenuation_radius", light["attenuation_radius_cm"])
    component.set_editor_property("cast_shadows", True)
    component.set_editor_property("source_radius", light["source_radius_cm"])
    component.set_editor_property("soft_source_radius", light["soft_source_radius_cm"])
    component.set_editor_property("use_temperature", True)
    component.set_editor_property("temperature", light["temperature_kelvin"])
    light_readbacks.append({
        **light,
        "intensity_readback": float(component.get_editor_property("intensity")),
        "temperature_readback": float(component.get_editor_property("temperature")),
    })
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
    "visual_lights": light_readbacks,
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
