#!/usr/bin/env python3
"""Prepare one native, dynamic-rig full-occlusion/reappearance episode.

The source 0323 capture already contains a real two-frame fully-occluded state
for ``source2``.  This tool replays the retained prefix forwards and backwards
so the same native actor passes behind the same apartment occluder and then
reappears.  It changes no mesh, room or animation asset, and emits only compact
planning contracts; native RGB/depth/target-only truth is still produced by
the normal SPEAR capture tool.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import canonical_json_sha256, write_json  # noqa: E402
from avengine.sensor_rig_trajectory import (  # noqa: E402
    materialize_sensor_rig_trajectory,
)


SCHEMA = "avengine_native_full_occlusion_reappearance_recipe_v1"
EPISODE_ID = "border_collie_human__dynamic_full_occlusion_reappearance_0323_v1"
TARGET_SLOT = "source2"
FRAME_INDEX_MAP = tuple(range(30)) + tuple(range(28, -1, -1)) + tuple(range(16))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }


def _remap(values: Sequence[Any]) -> list[Any]:
    _require(len(values) == 75, "source track must contain exactly 75 frames")
    return [deepcopy(values[index]) for index in FRAME_INDEX_MAP]


def _camera_state(frame: Mapping[str, Any], frame_index: int) -> dict[str, Any]:
    world_from_rig = deepcopy(frame["world_from_rig"])
    position = world_from_rig["translation_m"]
    # Habitat yaw is carried by the normalized program and linearly changes
    # from 55.0 to 55.2 degrees.  The UE conversion is the frozen AVEngine
    # Habitat->SPEAR camera convention used by the source suite.
    yaw = 55.0 + (0.2 * frame_index / 74.0)
    return {
        "frame_index": frame_index,
        "pts_ticks": frame_index * 3200,
        "habitat_position_m": deepcopy(position),
        "habitat_yaw_deg": yaw,
        "ue_position_cm": [
            position[0] * 100.0,
            position[2] * 100.0,
            position[1] * 100.0,
        ],
        "ue_yaw_deg": -yaw - 90.0,
        "world_from_rig": world_from_rig,
        "pose_hash": frame["pose_hash"],
    }


def build_recipe(
    *,
    source_fact: Mapping[str, Any],
    source_suite: Mapping[str, Any],
    source_fact_path: Path,
    source_suite_path: Path,
    episode_id: str = EPISODE_ID,
) -> dict[str, Any]:
    _require(len(FRAME_INDEX_MAP) == 75, "internal frame map must contain 75 frames")
    _require(FRAME_INDEX_MAP[28:31] == (28, 29, 28), "occlusion turn-point drift")
    _require(source_fact.get("time", {}).get("frame_count") == 75, "Fact frame drift")
    _require(source_fact.get("episode_id") == source_suite["scenarios"][0]["scenario_id"], "source Episode identity drift")
    scenario = deepcopy(source_suite["scenarios"][0])
    source_plan = scenario["plan"]
    source_frames = source_plan["frames"]
    _require(len(source_frames) == 75, "source suite must contain 75 frames")

    first_pose = source_fact["listener"]
    sensor_rig = materialize_sensor_rig_trajectory(
        trajectory_id=f"{episode_id}__sensor_rig",
        program={
            "kind": "ROTATE_IN_PLACE",
            "position_m": first_pose["position_m"],
            "start_yaw_deg": float(first_pose["yaw_deg"]),
            "end_yaw_deg": float(first_pose["yaw_deg"]) + 0.2,
            "yaw_interpolation": "SHORTEST_ARC",
        },
    )

    tracks = source_fact["tracks"]["instances"]
    bank_episode = {
        "episode_id": episode_id,
        "motion_case": "both_moving_dynamic_listener_full_occlusion_reappearance",
        "source_center_paths_m": {
            slot: _remap(track["emitter_position_m"])
            for slot, track in sorted(tracks.items())
        },
        "source_root_paths_m": {
            slot: _remap(track["root_position_m"])
            for slot, track in sorted(tracks.items())
        },
        "statistics": {
            "source_episode_id": source_fact["episode_id"],
            "frame_index_map": list(FRAME_INDEX_MAP),
            "target_source_slot_id": TARGET_SLOT,
            "native_visibility_expectation": {
                "source_frames_28_29": "fully_occluded in retained native truth",
                "derived_turn_frames": [28, 29, 30],
                "derived_reappearance_begins_after_frame": 30,
                "authority": "must_be_revalidated_by_same_renderer_metric_depth_and_target_only",
            },
        },
    }
    trajectory_bank = {
        "schema": "avengine_room_trajectory_bank_v2",
        "seed": 20260809,
        "frame_count": 75,
        "frame_rate_hz": 15,
        "seconds_per_episode": 5,
        "source_slots": sorted(tracks),
        "episode_count": 1,
        "motion_case_counts": {bank_episode["motion_case"]: 1},
        "claim_boundary": "research trajectory recipe; native visibility must be recaptured",
        "path_semantics": {
            "source_center_paths_m": "asset-bound world emitter points",
            "source_root_paths_m": "asset-bound actor roots",
        },
        "semantics": "retained 0323 native paths remapped for a true reappearance pass",
        "episodes": [bank_episode],
    }

    new_frames: list[dict[str, Any]] = []
    for frame_index, source_index in enumerate(FRAME_INDEX_MAP):
        frame = deepcopy(source_frames[source_index])
        frame["frame_index"] = frame_index
        frame["pts_ticks"] = frame_index * 3200
        frame["camera_state"] = _camera_state(sensor_rig["frames"][frame_index], frame_index)
        # Actor roots in the source suite and trajectory bank are exact copies
        # of the same retained frame.  Keep animation phase/time from that
        # source frame so reverse playback does not invent a new pose.
        new_frames.append(frame)

    scenario["scenario_id"] = episode_id
    scenario["variant_id"] = "dynamic_full_occlusion_reappearance_v1"
    scenario["scenario_directory"] = episode_id
    scenario["plan"]["frames"] = new_frames
    scenario["plan"]["camera"] = {
        **deepcopy(source_plan["camera"]),
        "habitat_position_m": deepcopy(first_pose["position_m"]),
        "habitat_yaw_deg": float(first_pose["yaw_deg"]),
        "ue_position_cm": _camera_state(sensor_rig["frames"][0], 0)["ue_position_cm"],
        "ue_yaw_deg": _camera_state(sensor_rig["frames"][0], 0)["ue_yaw_deg"],
        "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
        "dynamic": True,
    }
    scenario.pop("static_camera_upgrade", None)
    scenario["dynamic_camera_upgrade"] = {
        "schema": sensor_rig["schema"],
        "trajectory_id": sensor_rig["trajectory_id"],
        "first_pose_hash": sensor_rig["frames"][0]["pose_hash"],
        "last_pose_hash": sensor_rig["frames"][-1]["pose_hash"],
        "unique_pose_count": len({frame["pose_hash"] for frame in sensor_rig["frames"]}),
    }
    scenario["authoritative_capture_request"] = {
        "request_id": f"{episode_id}__native_capture",
        "episode_id": episode_id,
        "scenario_type": "full_occlusion_to_reappearance",
        "target_source_slot_id": TARGET_SLOT,
        "fact_path": "PENDING_FACT_COMPILATION",
        "fact_sha256": "PENDING_FACT_COMPILATION",
    }
    suite = {
        **deepcopy(source_suite),
        "scenarios": [scenario],
        "camera_upgrade": {
            "schema": "avengine_dynamic_spear_suite_camera_upgrade_v1",
            "source_suite": str(source_suite_path.resolve()),
            "source_suite_sha256": _sha256(source_suite_path),
            "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
            "qualification_claim": False,
        },
    }

    bindings = []
    for actor in source_plan["actors"]:
        source_slot = actor["actor_id"].removesuffix("_actor")
        instance = next(item for item in source_fact["instances"] if item["instance_id"] == source_slot)
        bindings.append(
            {
                "source_slot_id": source_slot,
                "asset_id": actor["asset_id"],
                "asset_revision": instance["registry"]["asset_revision"],
                "semantic_anchor_id": instance["emitter"]["anchor_id"],
                "emitter_offset_m": deepcopy(instance["emitter"]["offset_m"]),
                "offset_space": instance["emitter"]["offset_space"],
                "local_anatomical_forward_axis": actor[
                    "habitat_local_anatomical_forward_axis"
                ],
            }
        )
    binding_report = {
        "schema": "avengine_asset_emitter_scenario_report_v1",
        "status": "pass",
        "method": "retained_0323_exact_asset_emitter_binding",
        "scenario_count": 1,
        "scenarios": [
            {
                "trajectory_episode_id": episode_id,
                "output_episode_id": episode_id,
                "binding_report": {
                    "schema": "avengine_asset_emitter_binding_report_v1",
                    "status": "pass",
                    "method": "constant_asset_root_offset",
                    "episode_count": 1,
                    "listener_position_m": deepcopy(first_pose["position_m"]),
                    "bindings": bindings,
                    "mouth_animation_required": False,
                    "skeleton_lookup_required": False,
                },
            }
        ],
    }

    return {
        "schema": SCHEMA,
        "status": "prepared_pending_native_recapture",
        "qualification_claim": False,
        "episode_id": episode_id,
        "target_source_slot_id": TARGET_SLOT,
        "source_frame_index_map": list(FRAME_INDEX_MAP),
        "inputs": [
            _file_record(source_fact_path, role="retained_source_fact"),
            _file_record(source_suite_path, role="retained_source_native_suite"),
        ],
        "trajectory_bank": trajectory_bank,
        "sensor_rig_trajectory": sensor_rig,
        "asset_emitter_binding_report": binding_report,
        "suite_execution_plan": suite,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fact", type=Path, required=True)
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--episode-id", default=EPISODE_ID)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_fact_path = args.source_fact.resolve()
    source_suite_path = args.source_suite.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    recipe = build_recipe(
        source_fact=_load(source_fact_path),
        source_suite=_load(source_suite_path),
        source_fact_path=source_fact_path,
        source_suite_path=source_suite_path,
        episode_id=args.episode_id,
    )
    write_json(output / "trajectory_bank.json", recipe.pop("trajectory_bank"))
    write_json(output / "sensor_rig_trajectory.json", recipe.pop("sensor_rig_trajectory"))
    write_json(output / "asset_emitter_binding_report.json", recipe.pop("asset_emitter_binding_report"))
    write_json(output / "suite_execution_plan.pending_fact.json", recipe.pop("suite_execution_plan"))
    recipe["outputs"] = {
        name: {
            "path": str((output / name).resolve()),
            "sha256": _sha256(output / name),
        }
        for name in [
            "trajectory_bank.json",
            "sensor_rig_trajectory.json",
            "asset_emitter_binding_report.json",
            "suite_execution_plan.pending_fact.json",
        ]
    }
    recipe["recipe_identity_sha256"] = canonical_json_sha256(recipe)
    write_json(output / "recipe.json", recipe)
    print(f"NATIVE_FULL_OCCLUSION_REAPPEARANCE_RECIPE_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
