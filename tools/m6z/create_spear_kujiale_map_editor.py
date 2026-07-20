"""Create a UE map containing one external USD stage.

Run this script through Unreal Editor with AVENGINE_KUJIALE_MAP_REQUEST set to
an absolute JSON request path.  The saved map stores a reference to the local
derived USD stage; it does not embed or copy the InteriorAgent dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import unreal


request_path_value = os.environ.get("AVENGINE_KUJIALE_MAP_REQUEST")
if not request_path_value:
    raise RuntimeError("AVENGINE_KUJIALE_MAP_REQUEST is not set")
request_path = Path(request_path_value).expanduser().resolve()
request = json.loads(request_path.read_text(encoding="utf-8"))

map_path = str(request["map_path"])
usd_path = Path(request["usd_path"]).expanduser().resolve()
result_path = Path(request["result_path"]).expanduser().resolve()
if not map_path.startswith("/Game/"):
    raise RuntimeError("map_path must start with /Game/")
if not usd_path.is_file() or usd_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
    raise RuntimeError(f"derived USD stage does not exist: {usd_path}")
if unreal.EditorAssetLibrary.does_asset_exist(map_path):
    raise RuntimeError(
        f"map already exists: {map_path}; use a new map path or remove it explicitly"
    )

level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if not level_subsystem.new_level(map_path):
    raise RuntimeError(f"could not create level: {map_path}")

usd_stage_actor_class = getattr(unreal, "UsdStageActor", None)
if usd_stage_actor_class is None:
    # UE 5.5's MinimalAPI USD class is not exposed as a generated Python
    # attribute in every headless editor launch, but the native class remains
    # loadable by its stable script path.
    usd_stage_actor_class = unreal.load_class(
        None, "/Script/USDStage.UsdStageActor"
    )
if usd_stage_actor_class is None:
    raise RuntimeError("could not load /Script/USDStage.UsdStageActor")

actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
    usd_stage_actor_class,
    unreal.Vector(0.0, 0.0, 0.0),
    unreal.Rotator(0.0, 0.0, 0.0),
)
if actor is None:
    raise RuntimeError("could not spawn UsdStageActor")
actor.set_actor_label(f"AVEngine_{request['scene_id']}_external_USD")
actor.tags = ["avengine_comparison_visual", "avengine_external_usd"]
actor.set_initial_load_set(unreal.UsdInitialLoadSet.LOAD_ALL)
actor.set_root_layer(str(usd_path))
actor.set_stage_state(unreal.UsdStageState.OPENED_AND_LOADED)

if not level_subsystem.save_current_level():
    raise RuntimeError(f"could not save level: {map_path}")

actor_classes: dict[str, int] = {}
for item in unreal.EditorLevelLibrary.get_all_level_actors():
    name = item.get_class().get_name()
    actor_classes[name] = actor_classes.get(name, 0) + 1

result = {
    "status": "pass",
    "backend_role": "comparison_visual",
    "scene_id": str(request["scene_id"]),
    "map_path": map_path,
    "usd_path": str(usd_path),
    "root_layer_readback": str(actor.root_layer.file_path),
    "stage_state": str(actor.stage_state),
    "initial_load_set": str(actor.initial_load_set),
    "level_actor_classes": actor_classes,
    "external_asset_embedded": False,
    "claim_boundary": (
        "UE map references a local derived USD stage; downloaded dataset bytes "
        "remain external and are not part of the AVEngine repository."
    ),
}
result_path.parent.mkdir(parents=True, exist_ok=True)
result_path.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
unreal.log_warning(f"AVENGINE_KUJIALE_MAP_OK result={result_path}")
