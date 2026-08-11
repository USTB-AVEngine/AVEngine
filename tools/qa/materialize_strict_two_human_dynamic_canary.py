#!/usr/bin/env python3
"""Materialize one true-motion strict two-human full75 CPU closure.

This tool is deliberately CPU-only.  It expands the selected root and camera
paths into all 75 SPEAR frame states, binds identity-specific mouth emitters,
and plans one authoritative RLR state for every source/frame use.  A RIR cache
entry may be reused only when source position, listener position, and listener
orientation are exactly equal.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from avengine.contracts.json_io import write_json
from avengine.m5_1.source_contracts import sample_boundary
from avengine.m6.audio_program import bind_audio_program_hash, validate_audio_program
from avengine.m6.audio_render import assemble_audio_program_dry_buses
from avengine.m6.sources import (
    validate_sound_asset_registry,
    validate_source_endpoint_registry,
)
from avengine.m6x.room_feasibility import (
    TrajectoryBank,
    TrajectoryEpisode,
    build_rir_job_plan,
)
from avengine.m7.asset_bound_audio import bind_endpoint_buses_to_source_slots
from avengine.optional_backends.spear_visual import (
    actor_ue_yaw_degrees,
    camera_ue_yaw_degrees,
    habitat_point_to_apartment_ue_cm,
)
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory

REPOSITORY = Path(__file__).resolve().parents[2]
FRAME_COUNT = 75
FRAME_RATE_HZ = 15
TICKS_PER_FRAME = 3200
SCHEMA = "avengine_native_strict_two_human_dynamic_materialization_v1"
PREFLIGHT_SCHEMA = "avengine_native_strict_two_human_dynamic_full75_canary_preflight_v1"
BASE_SUITE = (
    REPOSITORY
    / "tmp/lead_d_strict_two_human_canary_v1/final_gate_v1/suite_execution_plan.json"
)
BASE_AUDIO = (
    REPOSITORY
    / "tmp/lead_d_strict_two_human_canary_v1/recipe_v4/controlled_audio_program"
)
DEFAULT_PREFLIGHT = (
    REPOSITORY / "tmp/lead_a_strict_two_human_full_episode_batch_v1/"
    "dynamic_canary_preflight_v1/preflight.json"
)
ACOUSTIC_PACKAGE = Path(
    "/data/datasets/avengine_workspaces/AVEngine-habitat-native/"
    "tmp/m3/root_ue_package_current_20260718_02/manifest.json"
)
SIMULATION_REQUEST = (
    REPOSITORY / "examples/m4/blender_custom/multi_source_canary_request.json"
)
HABITAT_PYTHON = Path("/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python")
TARGET_AUDIO = Path(
    "/data/jzy/code/SPEAR-lead-b/outputs/lead_b/audio_candidates_v1/"
    "media/speech_cremad_1001_ieo_neu_v1.wav"
)
SPEECH_INTRA_FRAME_OFFSET_SAMPLES = 128


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _selected_canary(preflight: Mapping[str, Any], index: int) -> dict[str, Any]:
    matches = [
        row for row in preflight["canaries"] if int(row["execution_order"]) == index
    ]
    _require(len(matches) == 1, "canary index must resolve exactly once")
    return deepcopy(matches[0])


def _identity_declarations(base_suite: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    scenario = base_suite["scenarios"][0]
    declarations = scenario["plan"]["actors"]
    by_identity = {str(item["asset_id"]): deepcopy(item) for item in declarations}
    _require(len(by_identity) == 2, "base suite must contain two distinct adults")
    return by_identity


def _source_scenario(row: Mapping[str, Any]) -> dict[str, Any]:
    suite = _load(Path(row["source_suite"]))
    matches = [
        scenario
        for scenario in suite["scenarios"]
        if scenario["scenario_id"] == row["native_source_scenario_id"]
    ]
    _require(len(matches) == 1, "native source scenario must resolve exactly once")
    return matches[0]


def _rotation_from_forward(forward: Sequence[float]) -> list[float]:
    x = float(forward[0])
    z = float(forward[2])
    norm = math.hypot(x, z)
    _require(norm > 1.0e-8, "actor forward vector is degenerate")
    yaw = math.atan2(x / norm, z / norm)
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _face_camera(root: Sequence[float], camera: Sequence[float]) -> list[float]:
    return _rotation_from_forward(
        [float(camera[0]) - float(root[0]), 0.0, float(camera[2]) - float(root[2])]
    )


def _forward_from_rotation(rotation: Sequence[float]) -> list[float]:
    yaw = 2.0 * math.atan2(float(rotation[1]), float(rotation[3]))
    return [math.sin(yaw), 0.0, math.cos(yaw)]


def _sensor_rig(row: Mapping[str, Any]) -> dict[str, Any]:
    yaws = [float(value) for value in row["camera"]["yaw_path_deg"]]
    position = [float(value) for value in row["camera"]["translation_m"]]
    if max(abs(value - yaws[0]) for value in yaws) <= 1.0e-12:
        program: dict[str, Any] = {
            "kind": "HOLD",
            "position_m": position,
            "yaw_deg": yaws[0],
        }
    else:
        program = {
            "kind": "WAYPOINT_ROUTE",
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
            "waypoints": [
                {
                    "frame_index": frame_index,
                    "position_m": position,
                    "yaw_deg": yaw,
                }
                for frame_index, yaw in enumerate(yaws)
            ],
        }
    rig = materialize_sensor_rig_trajectory(
        trajectory_id=f"{row['episode_id']}__sensor_rig",
        program=program,
    )
    _require(len(rig["frames"]) == FRAME_COUNT, "sensor rig is not full75")
    observed_yaws = []
    for frame in rig["frames"]:
        rotation = frame["world_from_rig"]["rotation_xyzw"]
        observed_yaw = math.degrees(
            2.0 * math.atan2(float(rotation[1]), float(rotation[3]))
        )
        observed_yaws.append(observed_yaw)
    maximum_yaw_error = max(
        abs((observed - expected + 180.0) % 360.0 - 180.0)
        for observed, expected in zip(observed_yaws, yaws)
    )
    _require(maximum_yaw_error <= 1.0e-9, "sensor rig yaw path drift")
    return rig


def _actor_materialization(
    *,
    row: Mapping[str, Any],
    source_scenario: Mapping[str, Any],
    declarations_by_identity: Mapping[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]], dict[str, list[list[float]]], dict[str, list[list[float]]]
]:
    source_frames = source_scenario["plan"]["frames"]
    _require(len(source_frames) == FRAME_COUNT, "source scenario is not full75")
    camera = row["camera"]["translation_m"]
    roles = {"source1": row["target"], "source2": row["distractor"]}
    declarations: list[dict[str, Any]] = []
    root_paths: dict[str, list[list[float]]] = {}
    emitter_paths: dict[str, list[list[float]]] = {}
    for slot, role in roles.items():
        identity_id = str(role["runtime_asset_id"])
        _require(
            identity_id in declarations_by_identity, f"missing {slot} adult declaration"
        )
        declaration = deepcopy(declarations_by_identity[identity_id])
        declaration["actor_id"] = f"{slot}_actor"
        declaration["runtime_asset_expectation"]["source_slot_id"] = slot
        declarations.append(declaration)
        roots = [[float(value) for value in point] for point in role["root_path_m"]]
        _require(len(roots) == FRAME_COUNT, f"{slot} root path is not full75")
        offset = [float(value) for value in declaration["emitter_offset_m"]]
        _require(offset[0] == offset[2] == 0.0, f"{slot} mouth offset must be vertical")
        root_paths[slot] = roots
        emitter_paths[slot] = [
            [root[0], root[1] + offset[1], root[2]] for root in roots
        ]

    output_frames: list[dict[str, Any]] = []
    declarations_by_slot = {
        declaration["actor_id"].removesuffix("_actor"): declaration
        for declaration in declarations
    }
    for frame_index in range(FRAME_COUNT):
        actor_states: list[dict[str, Any]] = []
        for slot, role in roles.items():
            declaration = declarations_by_slot[slot]
            source_actor_id = str(role["source_actor_id"])
            source_frame_index = int(role["frame_index_map"][frame_index])
            source_states = {
                state["actor_id"]: state
                for state in source_frames[source_frame_index]["actor_states"]
            }
            source_state = source_states[source_actor_id]
            moving = len({tuple(point) for point in root_paths[slot]}) > 1
            if moving:
                rotation = _rotation_from_forward(
                    source_state["anatomical_forward_habitat_world"]
                )
                action_id = "walk"
                action_phase = float(source_state.get("action_phase", 0.0)) % 1.0
                animation = declaration["walking_animation"]
            else:
                rotation = _face_camera(root_paths[slot][frame_index], camera)
                action_id = "idle"
                action_phase = 0.0
                animation = declaration["idle_animation"]
            forward_h = _forward_from_rotation(rotation)
            actor_states.append(
                {
                    "action_id": action_id,
                    "action_phase": action_phase,
                    "action_time_ticks": frame_index * TICKS_PER_FRAME,
                    "actor_id": f"{slot}_actor",
                    "actor_yaw_ue_deg": actor_ue_yaw_degrees(
                        rotation,
                        declaration["habitat_local_anatomical_forward_axis"],
                        declaration["ue_anatomical_forward_yaw_deg"],
                    ),
                    "anatomical_forward_habitat_world": forward_h,
                    "anatomical_forward_ue_world": [forward_h[0], forward_h[2], 0.0],
                    "asset_id": declaration["asset_id"],
                    "blueprint_class_path": declaration["blueprint_class_path"],
                    "rotation_xyzw": rotation,
                    "translation_m": root_paths[slot][frame_index],
                    "translation_ue_cm": list(
                        habitat_point_to_apartment_ue_cm(root_paths[slot][frame_index])
                    ),
                    "ue_animation": animation,
                }
            )
        output_frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "actor_states": actor_states,
            }
        )
    return output_frames, root_paths, emitter_paths


def _suite(
    *,
    base_suite: Mapping[str, Any],
    row: Mapping[str, Any],
    declarations: list[dict[str, Any]],
    actor_frames: list[dict[str, Any]],
    sensor_rig: Mapping[str, Any],
) -> dict[str, Any]:
    template = deepcopy(base_suite["scenarios"][0])
    frames: list[dict[str, Any]] = []
    for actor_frame, rig_frame in zip(actor_frames, sensor_rig["frames"]):
        frame_index = int(actor_frame["frame_index"])
        world_from_rig = deepcopy(rig_frame["world_from_rig"])
        yaw = float(row["camera"]["yaw_path_deg"][frame_index])
        frames.append(
            {
                **actor_frame,
                "camera_state": {
                    "frame_index": frame_index,
                    "pts_ticks": frame_index * TICKS_PER_FRAME,
                    "pose_hash": rig_frame["pose_hash"],
                    "world_from_rig": world_from_rig,
                    "habitat_position_m": deepcopy(world_from_rig["translation_m"]),
                    "habitat_yaw_deg": yaw,
                    "ue_position_cm": list(
                        habitat_point_to_apartment_ue_cm(
                            world_from_rig["translation_m"]
                        )
                    ),
                    "ue_yaw_deg": camera_ue_yaw_degrees(yaw),
                },
            }
        )
    template["scenario_id"] = row["episode_id"]
    template["scenario_directory"] = row["episode_id"]
    template["variant_id"] = row["mechanism"]
    template["plan"]["actors"] = declarations
    template["plan"]["frames"] = frames
    template["plan"]["camera"] = {
        **deepcopy(template["plan"]["camera"]),
        "habitat_position_m": deepcopy(row["camera"]["translation_m"]),
        "habitat_yaw_deg": float(row["camera"]["yaw_path_deg"][0]),
        "horizontal_fov_deg": float(row["camera"]["horizontal_fov_deg"]),
        "ue_position_cm": list(
            habitat_point_to_apartment_ue_cm(row["camera"]["translation_m"])
        ),
        "ue_yaw_deg": camera_ue_yaw_degrees(float(row["camera"]["yaw_path_deg"][0])),
        "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
        "dynamic": len(set(row["camera"]["yaw_path_deg"])) > 1,
    }
    template["plan"]["source_logic"]["scenario_id"] = row["native_source_scenario_id"]
    template["plan"]["source_logic"]["sources"] = [
        {
            "activation": "active",
            "entity_actor_id": "source1_actor",
            "source_endpoint_id": "source1_emitter",
        },
        {
            "activation": "silent",
            "entity_actor_id": "source2_actor",
            "source_endpoint_id": "source2_emitter",
        },
    ]
    template["plan"]["source_logic"]["clip_flags"] = {
        "both_sources_active": {"status": "pass", "value": False}
    }
    template["authoritative_capture_request"] = {
        "request_id": f"{row['episode_id']}__native_capture",
        "episode_id": row["episode_id"],
        "scenario_type": row["mechanism"],
        "target_source_slot_id": "source1",
        "fact_path": "PENDING_NATIVE_CAPTURE",
        "fact_sha256": "PENDING_NATIVE_CAPTURE",
    }
    template["authoritative_inputs"] = {
        "source_endpoint_registry": "controlled_audio_program/source_endpoint_registry.json",
        "sound_asset_registry": "controlled_audio_program/sound_asset_registry.json",
        "audio_program": "controlled_audio_program/audio_program.json",
    }
    template["reuse_contract"] = {
        "camera_and_room": "retained Apartment native room; selected independent camera cluster",
        "actor_roots": "all 75 roots copied from exact retained native root readbacks",
        "actor_yaws": "moving actor follows native anatomical heading; held actor faces camera",
        "audio": "source1 controlled speech only; source2 is explicitly silent",
    }
    template["camera_trajectory_binding"] = {
        "schema": sensor_rig["schema"],
        "trajectory_id": sensor_rig["trajectory_id"],
        "frame_count": FRAME_COUNT,
    }
    template.pop("static_camera_upgrade", None)
    suite = deepcopy(base_suite)
    suite["scenarios"] = [template]
    suite["camera_upgrade"] = {
        "schema": "avengine_dynamic_spear_suite_camera_binding_v1",
        "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
        "frame_count": FRAME_COUNT,
        "qualification_claim": False,
    }
    return suite


def _binding_report(
    *,
    row: Mapping[str, Any],
    declarations: Sequence[Mapping[str, Any]],
    emitter_paths: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, Any]:
    bindings = []
    for declaration in declarations:
        slot = str(declaration["actor_id"]).removesuffix("_actor")
        bindings.append(
            {
                "source_slot_id": slot,
                "asset_id": declaration["asset_id"],
                "asset_revision": declaration["asset_revision"],
                "semantic_anchor_id": declaration["emitter_anchor_id"],
                "emitter_offset_m": deepcopy(declaration["emitter_offset_m"]),
                "offset_space": "final_scaled_asset_root",
                "native_readback": "pending_full75_required",
                "emitter_frame_count": len(emitter_paths[slot]),
            }
        )
    binding = {
        "schema": "avengine_asset_emitter_binding_report_v1",
        "status": "pass",
        "episode_count": 1,
        "listener_position_m": deepcopy(row["camera"]["translation_m"]),
        "native_readback_status": "pending_full75_required",
        "qualification_claim": False,
        "bindings": bindings,
        "target_world_emitter_at_sparse_frame_m": deepcopy(
            emitter_paths["source1"][37]
        ),
    }
    return {
        "schema": "avengine_asset_emitter_scenario_report_v1",
        "status": "pass",
        "method": "identity_profile_vertical_mouth_offset_plus_exact_root_path_v1",
        "native_readback_status": "pending_full75_required",
        "profile_geometry_status": "pass_existing_reviewed_identity_profiles",
        "qualification_claim": False,
        "claim_boundary": "CPU binding plan; full75 native live readback remains required",
        "scenario_count": 1,
        "scenarios": [
            {
                "output_episode_id": row["episode_id"],
                "trajectory_episode_id": row["episode_id"],
                "binding_report": binding,
            }
        ],
    }


def _copy_audio_contracts(audio_template: Path, output: Path, episode_id: str) -> None:
    target = output / "controlled_audio_program"
    target.mkdir()
    for name in (
        "source_endpoint_registry.json",
        "sound_asset_registry.json",
        "audio_program.json",
        "controlled_audio_binding.json",
    ):
        value = _load(audio_template / name)
        if name == "audio_program.json":
            _require(
                len(value.get("events", [])) == 1, "audio template must have one event"
            )
            event = value["events"][0]
            source_duration = int(event["source_end_sample_exclusive"]) - int(
                event["source_start_sample"]
            )
            start_sample = sample_boundary(7) + SPEECH_INTRA_FRAME_OFFSET_SAMPLES
            end_sample = start_sample + source_duration
            _require(
                start_sample < sample_boundary(8)
                and sample_boundary(31) < end_sample <= sample_boundary(32),
                "shifted speech event must stay inside frames 7 through 31",
            )
            event["start_sample"] = start_sample
            event["end_sample_exclusive"] = end_sample
            event["start_tick"] = start_sample * 3
            event["end_tick_exclusive"] = end_sample * 3
            value = bind_audio_program_hash(value)
        elif name == "controlled_audio_binding.json":
            value["episode_id"] = episode_id
            value["status"] = "pass_materialized_pending_exact_rir_render"
        write_json(target / name, value)


def _validate_audio_contracts(output: Path) -> dict[str, Any]:
    root = output / "controlled_audio_program"
    endpoints = _load(root / "source_endpoint_registry.json")
    sounds = _load(root / "sound_asset_registry.json")
    program = _load(root / "audio_program.json")
    errors = validate_source_endpoint_registry(endpoints)
    errors.extend(validate_sound_asset_registry(sounds))
    errors.extend(
        validate_audio_program(
            program,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
        )
    )
    _require(not errors, f"controlled AudioProgram validation failed: {errors}")
    endpoint_slots = {
        endpoint["source_endpoint_id"]: endpoint["binding"]["entity_instance_id"]
        for endpoint in endpoints["source_endpoints"]
    }
    events = program["events"]
    source1_events = [
        event
        for event in events
        if endpoint_slots[event["source_endpoint_id"]] == "source1"
    ]
    source2_events = [
        event
        for event in events
        if endpoint_slots[event["source_endpoint_id"]] == "source2"
    ]
    _require(len(source1_events) == 1, "target must have exactly one event")
    _require(not source2_events, "distractor must have zero events")
    event = source1_events[0]
    first_frame = int(event["start_sample"]) * FRAME_RATE_HZ // 16000
    last_frame = (int(event["end_sample_exclusive"]) - 1) * FRAME_RATE_HZ // 16000
    _require([first_frame, last_frame] == [7, 31], "speech frame window drift")
    _require(
        program["timeline"]["frame_count"] == 75
        and program["timeline"]["sample_count"] == 80000,
        "AudioProgram timeline drift",
    )
    sound = next(
        item
        for item in sounds["sound_assets"]
        if item["sound_asset_id"] == event["sound_asset_id"]
    )
    assembled = assemble_audio_program_dry_buses(
        program,
        "A",
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
        asset_bindings={
            event["sound_asset_id"]: {
                "path": str(TARGET_AUDIO),
                "sha256": sound["dry_audio"]["sha256"],
            }
        },
    )
    buses = bind_endpoint_buses_to_source_slots(
        assembled.dry_audio.buses,
        endpoint_to_source_slot=endpoint_slots,
        source_slots=("source1", "source2"),
    )

    def frame_active(frame_index: int) -> bool:
        begin = sample_boundary(frame_index)
        end = sample_boundary(frame_index + 1)
        return bool(np.any(buses["source1"][begin:end] != 0.0))

    activity_checks = {
        "frame_6_silent": not frame_active(6),
        "frame_7_active": frame_active(7),
        "frame_31_active": frame_active(31),
        "frame_32_silent": not frame_active(32),
        "source2_all_zero": bool(np.all(buses["source2"] == 0.0)),
    }
    _require(
        all(activity_checks.values()), f"dry-bus activity drift: {activity_checks}"
    )
    return {
        "status": "pass",
        "target_event_count": 1,
        "distractor_event_count": 0,
        "speech_frame_window_inclusive": [first_frame, last_frame],
        "sample_count": 80000,
        "dry_bus_activity_checks": activity_checks,
    }


def _acoustic_execution_request(
    *, output: Path, episode_id: str, rir_plan: Mapping[str, Any]
) -> dict[str, Any]:
    cache = output / "rir_cache_v3"
    audio_output = output / "binaural_v1"
    return {
        "schema": "avengine_strict_two_human_dynamic_cpu_acoustic_execution_request_v1",
        "status": "ready_cpu_not_run",
        "compute_device": "CPU",
        "gpu_required": False,
        "episode_id": episode_id,
        "state_sampling": {
            "frame_stride": 1,
            "requested_source_frame_uses": rir_plan["requested_pair_state_count"],
            "distinct_rir_states": rir_plan["unique_rir_job_count"],
            "exact_pose_cache_reuse_count": rir_plan["cache_reuse_count"],
            "cache_reuse_policy": (
                "source_position_m + listener_position_m + "
                "listener_orientation_wxyz exact equality only"
            ),
        },
        "rir_render_argv": [
            "env",
            "PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            f"PYTHONPATH={REPOSITORY / 'src'}",
            "SKBUILD_EDITABLE_SKIP=/data/jzy/code/habitat-sim-AVEngine/build/cp312-cp312-linux_x86_64",
            "NUMBA_DISABLE_JIT=1",
            str(HABITAT_PYTHON),
            "tools/m6x/render_rir_cache.py",
            "--rir-job-plan",
            str((output / "rir_job_plan.json").resolve()),
            "--acoustic-package-manifest",
            str(ACOUSTIC_PACKAGE),
            "--simulation-request",
            str(SIMULATION_REQUEST.resolve()),
            "--output",
            str(cache.resolve()),
            "--layout",
            "binaural",
            "--batch-size",
            "64",
            "--thread-count",
            "32",
        ],
        "binaural_render_argv": [
            "env",
            f"PYTHONPATH={REPOSITORY / 'src'}",
            str(HABITAT_PYTHON),
            "tools/m7/render_asset_bound_binaural_batch.py",
            "--plan-root",
            str(output.resolve()),
            "--rir-cache",
            str(cache.resolve()),
            "--audio-program",
            str((output / "controlled_audio_program/audio_program.json").resolve()),
            "--audio-program-variant",
            "A",
            "--source-endpoint-registry",
            str(
                (
                    output / "controlled_audio_program/source_endpoint_registry.json"
                ).resolve()
            ),
            "--sound-asset-registry",
            str(
                (
                    output / "controlled_audio_program/sound_asset_registry.json"
                ).resolve()
            ),
            "--source-endpoint-slot",
            "lead_d_source1_mouth=source1",
            "--source-endpoint-slot",
            "lead_d_source2_mouth=source2",
            "--sound-audio",
            f"speech_cremad_1001_ieo_neu_v1={TARGET_AUDIO}",
            "--variants-per-episode",
            "1",
            "--retain-stems",
            "--output",
            str(audio_output.resolve()),
        ],
    }


def _materialize_into(
    *,
    preflight_path: Path,
    canary_index: int,
    base_suite_path: Path,
    audio_template: Path,
    output: Path,
    published_output: Path,
) -> Path:
    _require(
        output.is_dir() and not any(output.iterdir()),
        f"materialization staging must be an empty directory: {output}",
    )
    preflight = _load(preflight_path)
    _require(preflight.get("schema") == PREFLIGHT_SCHEMA, "preflight schema drift")
    row = _selected_canary(preflight, canary_index)
    _require(
        row["mechanism"] == "target_moves", "first atom supports target_moves only"
    )
    _require(
        row["target"]["identity_key"] == "M"
        and row["distractor"]["identity_key"] == "F",
        "first atom requires the reviewed M/F runtime pair",
    )
    for path in (
        base_suite_path,
        audio_template / "audio_program.json",
        ACOUSTIC_PACKAGE,
        SIMULATION_REQUEST,
        HABITAT_PYTHON,
        TARGET_AUDIO,
    ):
        _require(path.is_file(), f"required input is missing: {path}")

    base_suite = _load(base_suite_path)
    declarations_by_identity = _identity_declarations(base_suite)
    source_scenario = _source_scenario(row)
    actor_frames, root_paths, emitter_paths = _actor_materialization(
        row=row,
        source_scenario=source_scenario,
        declarations_by_identity=declarations_by_identity,
    )
    declarations = [
        deepcopy(declarations_by_identity[row["target"]["runtime_asset_id"]]),
        deepcopy(declarations_by_identity[row["distractor"]["runtime_asset_id"]]),
    ]
    declarations[0]["actor_id"] = "source1_actor"
    declarations[0]["runtime_asset_expectation"]["source_slot_id"] = "source1"
    declarations[1]["actor_id"] = "source2_actor"
    declarations[1]["runtime_asset_expectation"]["source_slot_id"] = "source2"
    sensor_rig = _sensor_rig(row)
    suite = _suite(
        base_suite=base_suite,
        row=row,
        declarations=declarations,
        actor_frames=actor_frames,
        sensor_rig=sensor_rig,
    )

    episode = TrajectoryEpisode(
        episode_id=row["episode_id"],
        motion_case="source1_moving_source2_static",
        source_root_paths_m={
            slot: np.asarray(path, dtype=np.float64)
            for slot, path in root_paths.items()
        },
        source_center_paths_m={
            slot: np.asarray(path, dtype=np.float64)
            for slot, path in emitter_paths.items()
        },
        statistics={
            "target_source_slot_id": "source1",
            "distractor_source_slot_id": "source2",
            "native_recapture_required": True,
        },
    )
    bank = TrajectoryBank(
        episodes=(episode,),
        frame_count=FRAME_COUNT,
        frame_rate_hz=FRAME_RATE_HZ,
        seed=20260812,
    )
    trajectory_bank = bank.record()
    trajectory_bank["path_semantics"] = {
        "source_center_paths_m": "identity-bound world mouth emitter points",
        "source_root_paths_m": "exact selected retained native actor roots",
    }
    listener_positions = {
        row["episode_id"]: [
            frame["world_from_rig"]["translation_m"] for frame in sensor_rig["frames"]
        ]
    }
    listener_orientations = {
        row["episode_id"]: [
            [rotation[3], rotation[0], rotation[1], rotation[2]]
            for rotation in (
                frame["world_from_rig"]["rotation_xyzw"]
                for frame in sensor_rig["frames"]
            )
        ]
    }
    rir_plan = build_rir_job_plan(
        bank,
        listener_positions_m_by_episode=listener_positions,
        listener_orientations_wxyz_by_episode=listener_orientations,
        stride_frames=1,
    )
    _require(
        rir_plan["requested_pair_state_count"] == 150,
        "RIR plan must cover both sources at all 75 frames",
    )
    per_slot_distinct = {
        slot: len(
            {
                job["job_id"]
                for job in rir_plan["jobs"]
                if any(use["source_slot_id"] == slot for use in job["uses"])
            }
        )
        for slot in ("source1", "source2")
    }
    _require(
        per_slot_distinct == {"source1": 75, "source2": 1},
        "target RIR state count drift",
    )
    _require(
        rir_plan["unique_rir_job_count"] == 76, "target total RIR state count drift"
    )
    _require(
        rir_plan["cache_reuse_count"] == 74, "target exact cache reuse count drift"
    )

    output.mkdir(parents=True, exist_ok=True)
    _copy_audio_contracts(audio_template, output, row["episode_id"])
    audio_validation = _validate_audio_contracts(output)
    binding_report = _binding_report(
        row=row, declarations=declarations, emitter_paths=emitter_paths
    )
    relative_paths = {
        "suite_execution_plan": "suite_execution_plan.json",
        "trajectory_bank": "trajectory_bank.json",
        "sensor_rig_trajectory": "sensor_rig_trajectory.json",
        "asset_emitter_binding_report": "asset_emitter_binding_report.json",
        "rir_job_plan": "rir_job_plan.json",
        "cpu_acoustic_execution_request": "cpu_acoustic_execution_request.json",
        "materialization_receipt": "materialization_receipt.json",
    }
    work_paths = {key: output / name for key, name in relative_paths.items()}
    published_paths = {
        key: published_output / name for key, name in relative_paths.items()
    }
    write_json(work_paths["suite_execution_plan"], suite)
    write_json(work_paths["trajectory_bank"], trajectory_bank)
    write_json(work_paths["sensor_rig_trajectory"], sensor_rig)
    write_json(work_paths["asset_emitter_binding_report"], binding_report)
    write_json(work_paths["rir_job_plan"], rir_plan)
    request = _acoustic_execution_request(
        output=published_output, episode_id=row["episode_id"], rir_plan=rir_plan
    )
    write_json(work_paths["cpu_acoustic_execution_request"], request)

    suite_frames = suite["scenarios"][0]["plan"]["frames"]
    maximum_root_error = 0.0
    action_counts: dict[str, dict[str, int]] = {}
    for slot, expected in root_paths.items():
        actor_id = f"{slot}_actor"
        observed = [
            next(
                state
                for state in frame["actor_states"]
                if state["actor_id"] == actor_id
            )
            for frame in suite_frames
        ]
        maximum_root_error = max(
            maximum_root_error,
            max(
                abs(float(state["translation_m"][axis]) - expected[frame_index][axis])
                for frame_index, state in enumerate(observed)
                for axis in range(3)
            ),
        )
        action_counts[slot] = {
            action: sum(state["action_id"] == action for state in observed)
            for action in ("idle", "walk")
        }
    _require(maximum_root_error == 0.0, "suite root path was not applied exactly")
    receipt = {
        "schema": SCHEMA,
        "status": "pass_cpu_materialized_pending_rir_execution_audio_and_gpu1",
        "episode_id": row["episode_id"],
        "mechanism": row["mechanism"],
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FRAME_RATE_HZ,
        "suite_actor_root_application": {
            "status": "pass_exact_all_75_frames",
            "source1_applied_frame_count": 75,
            "source2_applied_frame_count": 75,
            "maximum_root_path_error_m": maximum_root_error,
            "action_counts": action_counts,
        },
        "suite_camera_application": {
            "status": "pass_exact_all_75_frames",
            "applied_frame_count": 75,
            "listener_coupled_to_camera": True,
            "distinct_listener_pose_count": rir_plan["unique_listener_pose_count"],
        },
        "dynamic_acoustics": {
            "status": "planned_not_run",
            "frame_stride": 1,
            "requested_source_frame_uses": 150,
            "distinct_rir_state_count": rir_plan["unique_rir_job_count"],
            "distinct_rir_state_count_by_source_slot": per_slot_distinct,
            "exact_pose_cache_reuse_count": rir_plan["cache_reuse_count"],
            "cache_reuse_policy": (
                "source position, listener position, and listener orientation "
                "must be exactly identical"
            ),
        },
        "audio_program": {
            "validation": audio_validation,
            "target_source_slot": "source1",
            "target_event_count": 1,
            "distractor_source_slot": "source2",
            "distractor_event_count": 0,
            "target_speech_start_sample": (
                sample_boundary(7) + SPEECH_INTRA_FRAME_OFFSET_SAMPLES
            ),
            "sample_rate_hz": 16000,
            "sample_count": 80000,
        },
        "gpu_launch_authorized": False,
        "formal": False,
        "qualification_claim": False,
        "next_gate": "execute exact CPU RIR cache and binaural render, then GPU1 full75 capture",
        "artifacts": {
            key: str(path.resolve()) for key, path in published_paths.items()
        },
    }
    write_json(work_paths["materialization_receipt"], receipt)
    return work_paths["materialization_receipt"]


def materialize(
    *,
    preflight_path: Path,
    canary_index: int,
    base_suite_path: Path,
    audio_template: Path,
    output: Path,
) -> Path:
    """Build in staging and publish either a complete result or one failure receipt."""

    _require(not output.exists(), f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        _materialize_into(
            preflight_path=preflight_path,
            canary_index=canary_index,
            base_suite_path=base_suite_path,
            audio_template=audio_template,
            output=staging,
            published_output=output,
        )
    except Exception as exc:
        shutil.rmtree(staging)
        staging.mkdir()
        write_json(
            staging / "FAILED.json",
            {
                "schema": (
                    "avengine_strict_two_human_dynamic_materialization_failure_v1"
                ),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "formal": False,
                "qualification_claim": False,
            },
        )
        staging.replace(output)
        raise
    staging.replace(output)
    return output / "materialization_receipt.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--canary-index", type=int, default=1)
    parser.add_argument("--base-suite", type=Path, default=BASE_SUITE)
    parser.add_argument("--audio-template", type=Path, default=BASE_AUDIO)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    receipt = materialize(
        preflight_path=args.preflight.resolve(),
        canary_index=args.canary_index,
        base_suite_path=args.base_suite.resolve(),
        audio_template=args.audio_template.resolve(),
        output=args.output.resolve(),
    )
    print(f"STRICT_TWO_HUMAN_DYNAMIC_MATERIALIZATION_OK receipt={receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
