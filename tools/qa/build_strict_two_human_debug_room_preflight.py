#!/usr/bin/env python3
"""Prepare fail-closed visual probes for two additional cooked SPEAR maps."""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from copy import deepcopy
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.sensor_rig_trajectory import (
    materialize_sensor_rig_trajectory,
)

DEFAULT_PLAN = (
    REPOSITORY / "examples/qa/native_strict_two_human_debug_room_canary_plan_v1.json"
)
FRAME_COUNT = 75
TICKS_PER_FRAME = 3200


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_silence(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(80000 * 2 * 2))


def _actor_pose(
    state: dict[str, Any], *, root: list[float], camera: list[float], frame_index: int
) -> dict[str, Any]:
    result = deepcopy(state)
    yaw = -math.degrees(math.atan2(camera[0] - root[0], camera[2] - root[2]))
    yaw_rad = math.radians(yaw)
    result.update(
        {
            "frame_index": frame_index,
            "action_id": "idle",
            "action_phase": 0.0,
            "action_time_ticks": frame_index * TICKS_PER_FRAME,
            "actor_yaw_ue_deg": yaw,
            "anatomical_forward_habitat_world": [
                -math.sin(yaw_rad),
                0.0,
                math.cos(yaw_rad),
            ],
            "anatomical_forward_ue_world": [
                -math.sin(yaw_rad),
                math.cos(yaw_rad),
                0.0,
            ],
            "rotation_xyzw": [
                0.0,
                math.sin(-yaw_rad / 2.0),
                0.0,
                math.cos(-yaw_rad / 2.0),
            ],
            "translation_m": root,
            "translation_ue_cm": [
                root[0] * 100.0,
                root[2] * 100.0,
                root[1] * 100.0,
            ],
        }
    )
    return result


def _candidate_suite(
    *, base_suite: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    suite = deepcopy(base_suite)
    source_scenario = suite["scenarios"][0]
    scenario = deepcopy(source_scenario)
    episode_id = f"{candidate['candidate_id']}__strict_two_human_visual_probe_v1"
    camera = [float(value) for value in candidate["camera_translation_m"]]
    yaw = float(candidate["camera_habitat_yaw_deg"])
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id=f"{episode_id}__sensor_rig",
        program={"kind": "HOLD", "position_m": camera, "yaw_deg": yaw},
    )
    _require(len(trajectory["frames"]) == FRAME_COUNT, "sensor trajectory drift")
    actor_declarations = {
        item["actor_id"]: item for item in source_scenario["plan"]["actors"]
    }
    first_states = {
        item["actor_id"]: item
        for item in source_scenario["plan"]["frames"][0]["actor_states"]
    }
    roots = {
        "source1_actor": [
            float(value)
            for value in candidate["actors"]["source1"]["root_translation_m"]
        ],
        "source2_actor": [
            float(value)
            for value in candidate["actors"]["source2"]["root_translation_m"]
        ],
    }
    frames: list[dict[str, Any]] = []
    for frame_index in range(FRAME_COUNT):
        pose = trajectory["frames"][frame_index]
        world = pose["world_from_rig"]
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "camera_state": {
                    "frame_index": frame_index,
                    "pts_ticks": frame_index * TICKS_PER_FRAME,
                    "habitat_position_m": camera,
                    "habitat_yaw_deg": yaw,
                    "ue_position_cm": [
                        camera[0] * 100.0,
                        camera[2] * 100.0,
                        camera[1] * 100.0,
                    ],
                    "ue_yaw_deg": -yaw - 90.0,
                    "world_from_rig": world,
                    "pose_hash": pose["pose_hash"],
                },
                "actor_states": [
                    _actor_pose(
                        first_states[actor_id],
                        root=roots[actor_id],
                        camera=camera,
                        frame_index=frame_index,
                    )
                    for actor_id in ("source1_actor", "source2_actor")
                ],
            }
        )
    _require(
        set(actor_declarations) == {"source1_actor", "source2_actor"},
        "strict actor declarations drift",
    )
    scenario["scenario_id"] = episode_id
    scenario["variant_id"] = "debug_room_visual_probe_v1"
    scenario["scenario_directory"] = episode_id
    scenario["native_scene"] = {
        "map": candidate["map_path"],
        "layout": "native_map_unchanged",
        "lighting": "native_map_unchanged",
        "claim_boundary": "pending native visual review",
    }
    scenario["plan"]["room"] = {
        "room_id": candidate["proposed_room_id"],
        "room_capsule_id": "PENDING_NATIVE_DEBUG_ROOM_CAPSULE",
        "room_capsule_revision": "pending",
        "source_scene_provenance": {
            "provider": "SPEAR_Unreal",
            "scene_id": candidate["scene_id"],
            "upstream_role": "visual_map_probe_pending_room_closure",
        },
    }
    scenario["plan"]["frames"] = frames
    scenario["plan"]["camera"] = {
        "listener_id": "listener0",
        "habitat_position_m": camera,
        "habitat_yaw_deg": yaw,
        "ue_position_cm": [
            camera[0] * 100.0,
            camera[2] * 100.0,
            camera[1] * 100.0,
        ],
        "ue_yaw_deg": -yaw - 90.0,
        "horizontal_fov_deg": 105.0,
        "sensor_rig_trajectory_id": trajectory["trajectory_id"],
        "dynamic": False,
    }
    scenario["static_camera_upgrade"] = {
        "schema": trajectory["schema"],
        "trajectory_id": trajectory["trajectory_id"],
        "pose_hash": trajectory["frames"][0]["pose_hash"],
    }
    scenario.pop("dynamic_camera_upgrade", None)
    scenario["authoritative_capture_request"] = {
        "request_id": f"{episode_id}__visual_probe",
        "episode_id": episode_id,
        "scenario_type": "debug_room_visual_only_probe",
        "target_source_slot_id": "source1",
        "fact_path": "NOT_APPLICABLE_VISUAL_PROBE",
        "fact_sha256": "NOT_APPLICABLE_VISUAL_PROBE",
    }
    suite["native_map"] = candidate["map_path"]
    suite["scenarios"] = [scenario]
    suite["debug_room_probe_boundary"] = {
        "room_id": candidate["proposed_room_id"],
        "provisional_floor_y_m": candidate["provisional_floor_y_m"],
        "provisional_placement_is_floor_evidence": False,
        "audio_is_acoustic_evidence": False,
        "qualification_claim": False,
    }
    return suite


def build(
    plan_path: Path, output: Path, *, spear_executable: Path
) -> Path:
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    plan = _load(plan_path)
    _require(
        plan.get("schema")
        == "avengine_native_strict_two_human_debug_room_canary_plan_v1",
        "plan schema drift",
    )
    _require(plan.get("formal_episode_count") == 0, "formal count must remain zero")
    _require(plan.get("qualification_claim") is False, "qualification forbidden")
    _require(
        plan["gpu_policy"]
        == {
            "physical_gpu_index": 1,
            "graphics_adapter_argument": 1,
            "required_idle_compute_process_count": 0,
            "forbidden_physical_gpu_indices": [0, 3],
        },
        "GPU1 policy drift",
    )
    package_text = (
        _resolve(plan["package_manifest"]).read_text(encoding="utf-8").lower()
    )
    base_suite = _load(_resolve(plan["base_strict_suite"]))
    _require(
        len(base_suite["scenarios"]) == 1, "base strict suite must contain one scenario"
    )
    output.mkdir(parents=True)
    silence = output / "transport_silence_5s_16k_stereo.wav"
    _write_silence(silence)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(plan["candidates"]):
        fragment = candidate["map_package_fragment"].lower()
        _require(
            fragment in package_text, f"cooked map missing: {candidate['candidate_id']}"
        )
        candidate_root = output / candidate["candidate_id"]
        candidate_root.mkdir()
        suite = _candidate_suite(base_suite=base_suite, candidate=candidate)
        suite_path = candidate_root / "suite_execution_plan.pending_native.json"
        _write(suite_path, suite)
        episode_id = suite["scenarios"][0]["scenario_id"]
        visual_output = (
            output.parent / "debug_room_visual_probes" / candidate["candidate_id"]
        )
        visual_request = {
            "schema": "avengine_native_strict_two_human_debug_room_visual_probe_request_v1",
            "status": "ready_pending_gpu1_idle_gate",
            "candidate_id": candidate["candidate_id"],
            "proposed_room_id": candidate["proposed_room_id"],
            "episode_id": episode_id,
            "frame_indices": plan["visual_probe_contract"]["frame_indices"],
            "gpu_policy": plan["gpu_policy"],
            "audio_policy": plan["visual_probe_contract"]["audio_policy"],
            "audio_is_acoustic_evidence": False,
            "suite_plan": str(suite_path.resolve()),
            "transport_audio_wav": str(silence.resolve()),
            "output_root": str(visual_output.resolve()),
            "capture_argv": [
                "/data/jzy/miniconda3/envs/spear-env/bin/python",
                "tools/qa/capture_spear_native_pixel_episode.py",
                "--suite-plan",
                str(suite_path.resolve()),
                "--scenario-id",
                episode_id,
                "--audio-wav",
                str(silence.resolve()),
                "--spear-executable",
                str(spear_executable),
                "--output",
                str(visual_output.resolve()),
                "--rpc-port",
                str(39720 + index),
                "--graphics-adapter",
                "1",
                "--frame-index",
                "0",
                "--frame-index",
                "15",
                "--frame-index",
                "74",
            ],
        }
        visual_path = candidate_root / "visual_probe_request.json"
        _write(visual_path, visual_request)
        room_profile = {
            "schema": "avengine_pending_room_runtime_profile_v1",
            "status": "pending_native_floor_collision_and_visual_probe",
            "profile_id": candidate["candidate_id"],
            "proposed_room_id": candidate["proposed_room_id"],
            "scene_id": candidate["scene_id"],
            "map_path": candidate["map_path"],
            "provisional_floor_y_m": candidate["provisional_floor_y_m"],
            "provisional_floor_is_authority": False,
            "proposed_camera_translation_m": candidate["camera_translation_m"],
            "proposed_actor_roots_m": {
                key: value["root_translation_m"]
                for key, value in candidate["actors"].items()
            },
            "ready": False,
        }
        room_path = candidate_root / "room_runtime_profile.pending.json"
        _write(room_path, room_profile)
        acoustic = {
            "schema": "avengine_pending_exact_room_acoustic_plan_v1",
            "status": "blocked_pending_exact_native_surface_export_and_material_review",
            "proposed_room_id": candidate["proposed_room_id"],
            "map_path": candidate["map_path"],
            "source_slots": ["source1", "source2"],
            "planned_exact_rir_jobs_per_episode": 2,
            "listener_translation_m": candidate["camera_translation_m"],
            "source_root_translations_m": {
                key: value["root_translation_m"]
                for key, value in candidate["actors"].items()
            },
            "missing_authorities": [
                "exact_native_surface_geometry",
                "reviewed_acoustic_material_mapping",
                "native_floor_and_collision_safe_placement",
            ],
            "executable": False,
            "counts_as_exact_rir_evidence": False,
        }
        acoustic_path = candidate_root / "exact_acoustic_plan.pending.json"
        _write(acoustic_path, acoustic)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "proposed_room_id": candidate["proposed_room_id"],
                "map_cooked": True,
                "visual_probe_status": "ready_pending_gpu1_idle_gate",
                "floor_placement_status": "provisional_pending_native_readback",
                "acoustic_status": acoustic["status"],
                "ready_room": False,
                "suite": str(suite_path.resolve()),
                "visual_probe_request": str(visual_path.resolve()),
                "room_runtime_profile": str(room_path.resolve()),
                "acoustic_plan": str(acoustic_path.resolve()),
            }
        )
    result = {
        "schema": "avengine_native_strict_two_human_debug_room_preflight_v1",
        "status": "pass_cpu_package_and_probe_planning_pending_native_and_acoustic_gates",
        "candidate_count": 2,
        "cooked_map_count": 2,
        "native_visual_probe_pass_count": 0,
        "exact_acoustic_closure_count": 0,
        "additional_ready_room_count": 0,
        "final_required_ready_room_count": 3,
        "final_multi_room_100_authorized": False,
        "target_allocation_if_both_pass": plan[
            "final_multi_room_target_allocation_if_both_pass"
        ],
        "gpu_policy": plan["gpu_policy"],
        "rows": rows,
        "next_gate": "run_each_three-frame_visual_probe_on_idle_physical_GPU1_then_review_floor_collision_lighting_and_export_exact_acoustic_surfaces",
    }
    result_path = output / "preflight.json"
    _write(result_path, result)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spear-executable", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.plan.resolve(),
        args.output.resolve(),
        spear_executable=args.spear_executable,
    )
    print(f"STRICT_TWO_HUMAN_DEBUG_ROOM_PREFLIGHT_OK preflight={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
