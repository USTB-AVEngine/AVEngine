#!/usr/bin/env python3
# HISTORICAL TOOL (single-repo closure, 2026-08-21): this script built or
# validates retained strict-two-human evidence recorded against the
# pre-closure transition environment (sibling Habitat fork, sound-spaces,
# SPEAR-lead-b, and multi-repo SPEAR checkouts). The hard-coded absolute
# paths below are a frozen historical record, not current inputs. The current
# production chain runs on the installed runtime prefix and external data
# roots under /data/avengine_external; do not use this tool for new work.
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
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from avengine.camera_pose import yaw_rotation_xyzw
from avengine.contracts.json_io import canonical_json_sha256, write_json
from avengine.m5_1.source_contracts import sample_boundary
from avengine.m6.audio_program import bind_audio_program_hash, validate_audio_program
from avengine.m6.audio_render import (
    assemble_audio_program_dry_buses,
    assemble_semantic_audio_program_dry_buses,
)
from avengine.m6.registry import bind_content_hash
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
from avengine.qa.actor_motion_profile import (
    bind_planning_episode,
    build_actor_motion_profile,
    build_actor_motion_profile_from_planning,
    is_planning_actor_motion_profile,
    materialize_profile_frames,
    source_center_paths,
    validate_actor_motion_profile,
)
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory
from avengine.security import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)

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
CONTROLLED_SOUND_CONTENT_REGISTRY = Path(
    "/data/jzy/code/SPEAR-lead-b/outputs/lead_b/audio_candidates_v1/"
    "controlled_sound_content_registry_v1.json"
)
CONTROLLED_AUDIO_ROOT = Path(
    "/data/jzy/code/SPEAR-lead-b/outputs/lead_b/audio_candidates_v1/media"
)
RUNTIME_REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
# Retained exclusively for the legacy template adapter.  Planning rows carry
# exact half-open sample bounds and never derive them from a frame-local offset.
SPEECH_INTRA_FRAME_OFFSET_SAMPLES = 128
ANIMATION_TICKS_PER_PHASE_CYCLE = 51_200
INTERPOLATED_PATH_METHODS = {
    "arc_length_interpolation_of_native_polyline_v1",
    "equal_arc_interpolation_of_exact_native_human_polyline_v1",
}
LEGACY_CAMERA_PAN_ACOUSTICS = {
    "motion_case": "source1_static_source2_static_camera_pan",
    "per_slot_distinct": {"source1": 75, "source2": 75},
    "unique": 150,
    "reuse": 0,
}
MOTION_PROFILE_REQUIRED_MECHANISMS = {
    "target_moves",
    "distractor_moves",
    "both_move",
}


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


def _validate_planning_materialization_authority(
    *,
    binding: Mapping[str, Any],
    preflight_path: Path,
    canary_index: int,
    motion_candidate_path: Path | None,
) -> None:
    """Bind planning identity, runtime, content, camera, roots, and provenance."""

    row = binding["value"]
    _require(isinstance(row, Mapping), "planning episode row is not an object")
    mechanism = row.get("mechanism")
    _require(
        mechanism in MOTION_PROFILE_REQUIRED_MECHANISMS,
        "planning entry currently requires an actor-motion-profile mechanism",
    )
    _require(
        motion_candidate_path is not None,
        "planning dynamic row lacks a generic native motion authority; refusing natural-cadence inference",
    )
    preflight_row = _selected_canary(_load(preflight_path), canary_index)
    for key in ("episode_id", "mechanism", "target_side", "camera"):
        _require(
            row.get(key) == preflight_row.get(key),
            f"planning/preflight {key} binding drift",
        )
    required_role_fields = {
        "identity_key",
        "runtime_asset_id",
        "runtime_revision",
        "source_slot_id",
        "root_path_m",
        "frame_index_map",
    }
    for role_name in ("target", "distractor"):
        role = row.get(role_name)
        old_role = preflight_row.get(role_name)
        _require(
            isinstance(role, Mapping) and isinstance(old_role, Mapping),
            f"planning {role_name} authority is missing",
        )
        for key in required_role_fields:
            _require(
                key in role and role.get(key) == old_role.get(key),
                f"planning {role_name} {key} binding drift",
            )
        provenance = role.get("path_provenance", row.get("native_source_scenario_id"))
        old_provenance = old_role.get(
            "path_provenance", preflight_row.get("native_source_scenario_id")
        )
        _require(
            provenance is not None and provenance == old_provenance,
            f"planning {role_name} source provenance binding drift",
        )
    for key in (
        "content_id",
        "sound_asset_id",
        "voice_id",
        "voice_policy",
        "speech_frame_window_inclusive",
        "speech_sample_count",
        "speech_sample_rate_hz",
        "speech_channel_count",
        "speech_audio_uri",
    ):
        _require(
            row["target"].get(key) == preflight_row["target"].get(key),
            f"planning target audio {key} binding drift",
        )
    _require(
        row["distractor"].get("voice_policy") == "silent",
        "planning distractor must remain silent",
    )
    candidate = _load(motion_candidate_path)
    _require(
        candidate.get("legacy_episode_id") == row.get("episode_id")
        and candidate.get("mechanism") == mechanism,
        "planning/motion candidate episode or mechanism binding drift",
    )
    declarations = candidate.get("actor_declarations")
    actors = candidate.get("actors")
    _require(
        isinstance(declarations, Mapping) and isinstance(actors, Mapping),
        "motion candidate actor authorities are missing",
    )
    for role_name in ("target", "distractor"):
        slot = candidate.get(f"{role_name}_slot")
        actor = actors.get(slot) if isinstance(slot, str) else None
        _require(isinstance(actor, Mapping), f"candidate {role_name} actor is missing")
        declaration = declarations.get(actor.get("actor_id"))
        _require(
            isinstance(declaration, Mapping)
            and actor.get("asset_id") == row[role_name].get("runtime_asset_id")
            and declaration.get("asset_revision")
            == row[role_name].get("runtime_revision"),
            f"planning {role_name} runtime binding drift",
        )
        candidate_roots = actor.get("root_path_m")
        planning_roots = row[role_name].get("root_path_m")
        _require(
            isinstance(candidate_roots, list)
            and len(candidate_roots) == FRAME_COUNT
            and isinstance(planning_roots, list)
            and len(planning_roots) == FRAME_COUNT
            and candidate_roots[0] == planning_roots[0]
            and candidate_roots[-1] == planning_roots[-1],
            f"planning {role_name} root endpoint binding drift",
        )
        if actor.get("moving") is True:
            authority = actor.get("native_motion_authority")
            interval = actor.get("native_rate_active_interval")
            source_provenance = row[role_name].get("path_provenance", {})
            _require(
                isinstance(authority, Mapping)
                and isinstance(interval, Mapping)
                and authority.get("native_source_scenario_id")
                == source_provenance.get("native_source_scenario_id")
                and interval.get("native_source_frame_range_inclusive")
                == authority.get("native_source_frame_range_inclusive")
                and interval.get("time_scale") == 1.0
                and interval.get("global_time_stretch_applied") is False,
                f"planning {role_name} lacks native interval/phase authority",
            )
        else:
            _require(
                candidate_roots == planning_roots,
                f"planning {role_name} static root binding drift",
            )


def _identity_declarations(base_suite: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    scenario = base_suite["scenarios"][0]
    declarations = scenario["plan"]["actors"]
    by_identity = {str(item["asset_id"]): deepcopy(item) for item in declarations}
    _require(len(by_identity) == 2, "base suite must contain two distinct adults")
    return by_identity


def _planning_base_suite(
    row: Mapping[str, Any],
    *,
    suite_cache: Mapping[Path, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a clean human-only Apartment suite shell.

    The native animal suite is read only to bind the room/map shell.  No actor,
    source, qualification, authoritative-input, or reuse claim is copied.
    """

    target = row.get("target")
    _require(isinstance(target, Mapping), "planning target authority is missing")
    authority = target.get("motion_profile_authority")
    _require(
        isinstance(authority, Mapping),
        "planning target motion profile authority is missing",
    )
    source = authority.get("source_path")
    _require(isinstance(source, Mapping), "planning target source authority is missing")
    suite_path = Path(str(source.get("source_suite", ""))).resolve()
    _require(suite_path.is_file(), f"planning source suite is missing: {suite_path}")
    suite = (
        deepcopy(dict(suite_cache[suite_path]))
        if suite_cache is not None and suite_path in suite_cache
        else _load(suite_path)
    )
    scenarios = suite.get("scenarios")
    scenario_id = source.get("native_source_scenario_id")
    matches = (
        [
            scenario
            for scenario in scenarios
            if isinstance(scenarios, list)
            and isinstance(scenario, Mapping)
            and scenario.get("scenario_id") == scenario_id
        ]
        if isinstance(scenarios, list)
        else []
    )
    _require(
        len(matches) == 1,
        "planning source suite scenario must resolve exactly once",
    )
    source_scenario = matches[0]
    source_plan = source_scenario.get("plan")
    _require(isinstance(source_plan, Mapping), "planning source plan is missing")
    source_room = source_plan.get("room")
    source_coordinate = source_plan.get("coordinate_contract")
    source_render = source_scenario.get("render")
    source_native_scene = source_scenario.get("native_scene")
    _require(
        isinstance(source_room, Mapping)
        and isinstance(source_coordinate, Mapping)
        and isinstance(source_render, Mapping)
        and isinstance(source_native_scene, Mapping),
        "planning source Apartment shell is incomplete",
    )
    return {
        "schema": "avengine_optional_spear_apartment_suite_v1",
        "backend_role": "planning_human_materialization",
        "native_map": suite.get("native_map"),
        "lighting_profile": deepcopy(source_native_scene.get("lighting_profile")),
        "authority": {
            "room_layout": "native Apartment map selected by source authority",
            "actor_state": "planning row plus runtime human registry",
            "audio": "planning semantic AudioProgram",
            "qualification_claim": False,
        },
        "scenarios": [
            {
                "schema": "avengine_optional_spear_apartment_scenario_v1",
                "scenario_id": "PENDING_PLANNING_EPISODE",
                "scenario_directory": "PENDING_PLANNING_EPISODE",
                "variant_id": "A",
                "backend_role": "planning_human_materialization",
                "native_scene": {
                    "map": source_native_scene.get("map"),
                    "layout": source_native_scene.get("layout"),
                    "lighting": source_native_scene.get("lighting"),
                    "lighting_profile": deepcopy(
                        source_native_scene.get("lighting_profile")
                    ),
                    "outdoor_view": source_native_scene.get("outdoor_view"),
                },
                "render": deepcopy(dict(source_render)),
                "authoritative_inputs": {},
                "reuse_contract": {
                    "room_and_map_only": "selected native Apartment shell",
                    "actor_or_source_authority_reused": False,
                    "qualification_claim": False,
                },
                "plan": {
                    "schema": "avengine_optional_spear_visual_plan_v1",
                    "backend_role": "planning_human_materialization",
                    "authority": {
                        "actor_state": "planning actor motion profile v2",
                        "backend_may_replan": False,
                        "room_identity_and_layout": "selected native Apartment shell",
                        "source_center_placement": "runtime human emitter declarations",
                        "source_logic": "planning semantic AudioProgram",
                        "qualification_claim": False,
                    },
                    "room": deepcopy(dict(source_room)),
                    "render": {
                        "fps_den": 1,
                        "fps_num": FRAME_RATE_HZ,
                        "frame_count": FRAME_COUNT,
                        "ticks_per_frame": TICKS_PER_FRAME,
                    },
                    "camera": {},
                    "coordinate_contract": deepcopy(dict(source_coordinate)),
                    "actors": [],
                    "frames": [],
                    "source_logic": {
                        "schema": "avengine_planning_human_source_logic_v1",
                        "scenario_id": "PENDING_PLANNING_EPISODE",
                        "variant_id": "A",
                        "sources": [],
                        "clip_flags": {},
                    },
                    "qualification": {
                        "status": "not_claimed",
                        "qualification_claim": False,
                        "claim_boundary": "pending native human capture and review",
                    },
                },
            }
        ],
    }


def _planning_materialization_authorities(
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the planning profile once and reuse its loaded suite as template."""

    suite_cache: dict[Path, Mapping[str, Any]] = {}
    profile = build_actor_motion_profile_from_planning(
        planning_manifest_path=str(binding["path"]),
        episode_id=str(binding["value"]["episode_id"]),
        source_suite_cache=suite_cache,
    )
    validate_actor_motion_profile(profile)
    base_suite = _planning_base_suite(binding["value"], suite_cache=suite_cache)
    return profile, base_suite


def _source_scenarios(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    suite = _load(Path(row["source_suite"]))
    by_id = {str(scenario["scenario_id"]): scenario for scenario in suite["scenarios"]}
    _require(
        len(by_id) == len(suite["scenarios"]),
        "source suite scenario IDs must be unique",
    )
    result: dict[str, dict[str, Any]] = {}
    for slot, role in (("source1", row["target"]), ("source2", row["distractor"])):
        provenance = role.get("path_provenance", {})
        scenario_id = provenance.get("native_source_scenario_id") or row.get(
            "native_source_scenario_id"
        )
        _require(
            isinstance(scenario_id, str) and scenario_id in by_id,
            f"{slot} native source scenario must resolve exactly once",
        )
        result[slot] = by_id[scenario_id]
    declared_ids = row.get("native_source_scenario_ids")
    if declared_ids is not None:
        _require(
            declared_ids
            == [
                row["target"]["path_provenance"]["native_source_scenario_id"],
                row["distractor"]["path_provenance"]["native_source_scenario_id"],
            ]
            and len(set(declared_ids)) == 2,
            "counterfactual source scenario binding drift",
        )
    return result


def _rotation_from_forward(forward: Sequence[float]) -> list[float]:
    x = float(forward[0])
    z = float(forward[2])
    norm = math.hypot(x, z)
    _require(norm > 1.0e-8, "actor forward vector is degenerate")
    yaw_deg = math.degrees(math.atan2(x / norm, z / norm))
    return yaw_rotation_xyzw(yaw_deg)


def _face_camera(root: Sequence[float], camera: Sequence[float]) -> list[float]:
    return _rotation_from_forward(
        [float(camera[0]) - float(root[0]), 0.0, float(camera[2]) - float(root[2])]
    )


def _unwrap_phase_path(phases: Sequence[float]) -> list[float]:
    _require(len(phases) == FRAME_COUNT, "animation phase path is not full75")
    wrapped = [float(value) for value in phases]
    _require(
        all(0.0 <= value < 1.0 for value in wrapped),
        "animation phases must be wrapped to [0, 1)",
    )
    unwrapped = [wrapped[0]]
    for previous, current in pairwise(wrapped):
        advance = (current - previous) % 1.0
        _require(
            0.0 < advance < 0.5,
            "moving animation phase must advance continuously without a large jump",
        )
        unwrapped.append(unwrapped[-1] + advance)
    return unwrapped


def _arc_length_animation_timing(
    *, role: Mapping[str, Any], roots: Sequence[Sequence[float]]
) -> dict[str, Any] | None:
    provenance = role.get("path_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("method") not in INTERPOLATED_PATH_METHODS
    ):
        return None
    _require(
        provenance.get("interior_output_roots_exact_native_frame_readbacks") is False
        and provenance.get("endpoints_exact_native_readbacks") is True,
        "interpolated path provenance must distinguish anchors from interior roots",
    )
    _require(
        int(provenance.get("output_root_count", -1)) == FRAME_COUNT
        and int(provenance.get("output_unique_root_count_at_1mm", -1)) == FRAME_COUNT,
        "interpolated slow-walk path must have 75 unique roots",
    )
    phases = role.get("per_frame_action_phase")
    forwards = role.get("per_frame_anatomical_forward_habitat_world")
    yaws = role.get("per_frame_tangent_yaw_habitat_deg")
    _require(
        isinstance(phases, list)
        and isinstance(forwards, list)
        and isinstance(yaws, list)
        and len(phases) == len(forwards) == len(yaws) == FRAME_COUNT,
        "interpolated slow walk needs 75 phase/forward/yaw samples",
    )
    unwrapped = _unwrap_phase_path(phases)
    segment_lengths = [
        math.hypot(
            float(current[0]) - float(previous[0]),
            float(current[2]) - float(previous[2]),
        )
        for previous, current in pairwise(roots)
    ]
    _require(
        len(segment_lengths) == FRAME_COUNT - 1 and min(segment_lengths) > 1.0e-6,
        "interpolated slow-walk roots must move every frame",
    )
    phase_advances = [current - previous for previous, current in pairwise(unwrapped)]
    phase_per_meter = [
        phase / distance for phase, distance in zip(phase_advances, segment_lengths)
    ]
    maximum_phase_per_meter_error = max(phase_per_meter) - min(phase_per_meter)
    _require(
        maximum_phase_per_meter_error <= 1.0e-8,
        "animation phase is not proportional to root arc length",
    )
    maximum_forward_angular_error_deg = 0.0
    maximum_yaw_error_deg = 0.0
    for frame_index, (previous, current) in enumerate(pairwise(roots)):
        dx = float(current[0]) - float(previous[0])
        dz = float(current[2]) - float(previous[2])
        norm = math.hypot(dx, dz)
        expected = [dx / norm, 0.0, dz / norm]
        observed = [float(value) for value in forwards[frame_index]]
        observed_norm = math.hypot(observed[0], observed[2])
        _require(observed_norm > 1.0e-8, "slow-walk forward vector is degenerate")
        dot = max(
            -1.0,
            min(
                1.0,
                expected[0] * observed[0] / observed_norm
                + expected[2] * observed[2] / observed_norm,
            ),
        )
        maximum_forward_angular_error_deg = max(
            maximum_forward_angular_error_deg, math.degrees(math.acos(dot))
        )
        expected_yaw = math.degrees(math.atan2(expected[0], expected[2])) % 360.0
        yaw_error = abs(
            (float(yaws[frame_index]) - expected_yaw + 180.0) % 360.0 - 180.0
        )
        maximum_yaw_error_deg = max(maximum_yaw_error_deg, yaw_error)
    _require(
        maximum_forward_angular_error_deg <= 1.0e-5 and maximum_yaw_error_deg <= 1.0e-5,
        "slow-walk forward/yaw path is not tangent to the applied roots",
    )
    path_length_m = sum(segment_lengths)
    episode_span_seconds = (FRAME_COUNT - 1) / FRAME_RATE_HZ
    phase_cycles = unwrapped[-1] - unwrapped[0]
    return {
        "schema": "avengine_arc_length_bound_animation_timing_v1",
        "status": "pass",
        "mode": "arc_length_preserving_native_stride_v1",
        "path_provenance": deepcopy(dict(provenance)),
        "action_phase_path": [float(value) for value in phases],
        "unwrapped_action_phase_path": unwrapped,
        "action_time_ticks_path": [
            round(value * ANIMATION_TICKS_PER_PHASE_CYCLE) for value in unwrapped
        ],
        "path_length_m": path_length_m,
        "episode_span_seconds": episode_span_seconds,
        "average_root_speed_m_per_second": path_length_m / episode_span_seconds,
        "phase_cycle_count": phase_cycles,
        "stride_distance_m_per_phase_cycle": path_length_m / phase_cycles,
        "maximum_segment_length_delta_m": max(segment_lengths) - min(segment_lengths),
        "maximum_phase_per_meter_error": maximum_phase_per_meter_error,
        "maximum_forward_angular_error_deg": maximum_forward_angular_error_deg,
        "maximum_tangent_yaw_error_deg": maximum_yaw_error_deg,
        "claim_boundary": (
            "interior roots are deterministic arc-length interpolation of the "
            "retained native polyline, not exact native per-frame readbacks"
        ),
    }


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
    source_scenarios: Mapping[str, Mapping[str, Any]],
    declarations_by_identity: Mapping[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[list[float]]],
    dict[str, list[list[float]]],
    dict[str, dict[str, Any]],
]:
    camera = row["camera"]["translation_m"]
    roles = {"source1": row["target"], "source2": row["distractor"]}
    declarations: list[dict[str, Any]] = []
    root_paths: dict[str, list[list[float]]] = {}
    emitter_paths: dict[str, list[list[float]]] = {}
    animation_timing: dict[str, dict[str, Any]] = {}
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
        timing = _arc_length_animation_timing(role=role, roots=roots)
        if timing is not None:
            animation_timing[slot] = timing
        source_frames = source_scenarios[slot]["plan"]["frames"]
        frame_index_map = role["frame_index_map"]
        _require(
            len(source_frames) == FRAME_COUNT and len(frame_index_map) == FRAME_COUNT,
            f"{slot} source scenario/frame map is not full75",
        )
        for source_frame_index in frame_index_map:
            _require(
                0 <= int(source_frame_index) < FRAME_COUNT
                and any(
                    state["actor_id"] == role["source_actor_id"]
                    for state in source_frames[int(source_frame_index)]["actor_states"]
                ),
                f"{slot} native source actor/frame provenance drift",
            )

    output_frames: list[dict[str, Any]] = []
    declarations_by_slot = {
        declaration["actor_id"].removesuffix("_actor"): declaration
        for declaration in declarations
    }
    for frame_index in range(FRAME_COUNT):
        actor_states: list[dict[str, Any]] = []
        for slot, role in roles.items():
            declaration = declarations_by_slot[slot]
            moving = len({tuple(point) for point in root_paths[slot]}) > 1
            if moving:
                timing = animation_timing.get(slot)
                if timing is None:
                    source_frames = source_scenarios[slot]["plan"]["frames"]
                    source_actor_id = str(role["source_actor_id"])
                    source_frame_index = int(role["frame_index_map"][frame_index])
                    source_states = {
                        state["actor_id"]: state
                        for state in source_frames[source_frame_index]["actor_states"]
                    }
                    source_state = source_states[source_actor_id]
                    forward = source_state["anatomical_forward_habitat_world"]
                    action_phase = float(source_state.get("action_phase", 0.0)) % 1.0
                    action_time_ticks = frame_index * TICKS_PER_FRAME
                    timing_mode = "nearest_native_state_phase_v1"
                else:
                    forward = role["per_frame_anatomical_forward_habitat_world"][
                        frame_index
                    ]
                    action_phase = float(timing["action_phase_path"][frame_index])
                    action_time_ticks = int(
                        timing["action_time_ticks_path"][frame_index]
                    )
                    timing_mode = timing["mode"]
                rotation = _rotation_from_forward(forward)
                action_id = "walk"
                animation = declaration["walking_animation"]
            else:
                rotation = _face_camera(root_paths[slot][frame_index], camera)
                action_id = "idle"
                action_phase = 0.0
                action_time_ticks = 0
                timing_mode = "held_idle_v1"
                animation = declaration["idle_animation"]
            forward_h = _forward_from_rotation(rotation)
            actor_states.append(
                {
                    "action_id": action_id,
                    "action_phase": action_phase,
                    "action_time_ticks": action_time_ticks,
                    "animation_timing_mode": timing_mode,
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
    return output_frames, root_paths, emitter_paths, animation_timing


def _actor_materialization_from_profile(
    profile: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[list[float]]],
    dict[str, list[list[float]]],
    dict[str, dict[str, Any]],
]:
    """Adapt hash-bound profile states to the SPEAR suite without inferring motion."""

    validate_actor_motion_profile(profile)
    candidate = profile["authorities"]["candidate"]["value"]
    candidate_actors = candidate["actors"]
    declarations_by_actor = candidate["actor_declarations"]
    profile_frames = materialize_profile_frames(profile)
    _require(
        len(profile_frames) == FRAME_COUNT
        and int(candidate["frame_count"]) == FRAME_COUNT,
        "motion profile is not full75",
    )
    declarations = [
        deepcopy(declarations_by_actor[candidate_actors[slot]["actor_id"]])
        for slot in candidate_actors
    ]
    root_paths: dict[str, list[list[float]]] = {
        str(slot): [] for slot in candidate_actors
    }
    output_frames: list[dict[str, Any]] = []
    action_counts: dict[str, dict[str, int]] = {
        str(slot): {
            str(action_id): 0
            for action_id in declarations_by_actor[actor["actor_id"]][
                "animation_paths_by_action_id"
            ]
        }
        for slot, actor in candidate_actors.items()
    }
    for frame_index, profile_frame in enumerate(profile_frames):
        output_states: list[dict[str, Any]] = []
        profile_states = profile_frame["actor_states"]
        _require(
            [state["slot_id"] for state in profile_states] == list(candidate_actors),
            f"profile actor order drift at f{frame_index}",
        )
        for profile_state in profile_states:
            slot = str(profile_state["slot_id"])
            actor = candidate_actors[slot]
            declaration = declarations_by_actor[profile_state["actor_id"]]
            forward_path = actor.get("anatomical_forward_habitat_world_path")
            _require(
                isinstance(forward_path, list) and len(forward_path) == FRAME_COUNT,
                f"profile forward path is not full75 for {slot}",
            )
            forward = [float(value) for value in forward_path[frame_index]]
            rotation = _rotation_from_forward(forward)
            expected_actor_yaw = actor_ue_yaw_degrees(
                rotation,
                declaration["habitat_local_anatomical_forward_axis"],
                declaration["ue_anatomical_forward_yaw_deg"],
            )
            _require(
                abs(
                    (
                        expected_actor_yaw
                        - float(profile_state["actor_yaw_ue_deg"])
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                )
                <= 1.0e-8,
                f"profile forward/yaw projection drift at f{frame_index} {slot}",
            )
            root = [float(value) for value in profile_state["translation_m"]]
            root_paths[slot].append(root)
            action_id = str(profile_state["action_id"])
            _require(
                action_id in action_counts[slot],
                f"profile action {action_id!r} is not declared at f{frame_index} {slot}",
            )
            action_counts[slot][action_id] += 1
            output_states.append(
                {
                    **deepcopy(profile_state),
                    "asset_id": declaration["asset_id"],
                    "blueprint_class_path": declaration["blueprint_class_path"],
                    "rotation_xyzw": rotation,
                    "anatomical_forward_habitat_world": forward,
                    "anatomical_forward_ue_world": [forward[0], forward[2], 0.0],
                }
            )
        output_frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": int(profile_frame["pts_ticks"]),
                "canonical_motion_profile_frame_sha256": profile_frame[
                    "canonical_frame_sha256"
                ],
                "actor_states": output_states,
            }
        )
    expected_roots = {
        str(slot): [[float(value) for value in root] for root in actor["root_path_m"]]
        for slot, actor in candidate_actors.items()
    }
    _require(root_paths == expected_roots, "profile frame/root-path authority drift")
    emitter_paths = source_center_paths(profile)
    motion_semantics = {
        str(slot): {
            "moving": actor["moving"],
            "native_rate_active_interval": deepcopy(
                actor["native_rate_active_interval"]
            ),
            "native_rate_action_segments": deepcopy(
                actor.get("native_rate_action_segments", [])
            ),
            "trajectory_preflight": deepcopy(actor["trajectory_preflight"]),
            "action_counts": action_counts[str(slot)],
        }
        for slot, actor in candidate_actors.items()
    }
    return (
        declarations,
        output_frames,
        root_paths,
        emitter_paths,
        motion_semantics,
    )


def _validate_rir_plan_against_profile(
    rir_plan: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    expectation = profile["rir_expectation"]
    candidate = profile["authorities"]["candidate"]["value"]
    candidate_episode_id = str(candidate["candidate_episode_id"])
    if is_planning_actor_motion_profile(profile):
        output_episode_id = candidate_episode_id
    else:
        old_row = profile["authorities"]["selected_old_row"]["value"]
        output_episode_id = str(old_row["episode_id"])

    def normalize_episode_id(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(output_episode_id, candidate_episode_id)
        if isinstance(value, list):
            return [normalize_episode_id(item) for item in value]
        if isinstance(value, Mapping):
            return {
                normalize_episode_id(key): normalize_episode_id(item)
                for key, item in value.items()
            }
        return value

    compared = {
        "stride_frames": int(rir_plan["stride_frames"]),
        "requested_pair_state_count": int(rir_plan["requested_pair_state_count"]),
        "unique_rir_job_count": int(rir_plan["unique_rir_job_count"]),
    }
    _require(
        all(compared[key] == int(expectation[key]) for key in compared),
        "actual RIR plan does not match actor motion profile expectation",
    )
    actual_plan_sha256 = canonical_json_sha256(rir_plan)
    normalized_actual_plan_sha256 = canonical_json_sha256(
        normalize_episode_id(rir_plan)
    )
    _require(
        normalized_actual_plan_sha256 == expectation["canonical_plan_sha256"],
        "actual RIR plan content does not match actor motion profile expectation",
    )
    return {
        "status": "pass_actual_plan_matches_profile_expectation",
        "profile_content_sha256": profile["profile_content_sha256"],
        "profile_expected_plan_sha256": expectation["canonical_plan_sha256"],
        "actual_plan_sha256": actual_plan_sha256,
        "normalized_actual_plan_sha256": normalized_actual_plan_sha256,
        "normalization": {
            "field": "episode_id",
            "from": output_episode_id,
            "to": candidate_episode_id,
        },
        "compared_counts": compared,
        "derived_cache_reuse_count": (
            compared["requested_pair_state_count"] - compared["unique_rir_job_count"]
        ),
    }


def _suite(
    *,
    base_suite: Mapping[str, Any],
    row: Mapping[str, Any],
    declarations: list[dict[str, Any]],
    actor_frames: list[dict[str, Any]],
    sensor_rig: Mapping[str, Any],
    motion_profile: Mapping[str, Any] | None = None,
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
    planning_profile = motion_profile is not None and is_planning_actor_motion_profile(
        motion_profile
    )
    template["variant_id"] = "A" if planning_profile else row["mechanism"]
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
    template["plan"]["source_logic"]["scenario_id"] = row["episode_id"]
    template["plan"]["source_logic"]["variant_id"] = "A"
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
    }
    template["authoritative_inputs"] = (
        {
            "semantic_sound_content_registry": (
                "controlled_audio_program/semantic_sound_content_registry.json"
            ),
            "semantic_source_endpoint_registry": (
                "controlled_audio_program/semantic_source_endpoint_registry.json"
            ),
            "audio_program": "controlled_audio_program/audio_program.json",
        }
        if planning_profile
        else {
            "source_endpoint_registry": (
                "controlled_audio_program/source_endpoint_registry.json"
            ),
            "sound_asset_registry": "controlled_audio_program/sound_asset_registry.json",
            "audio_program": "controlled_audio_program/audio_program.json",
        }
    )
    camera_pan = row["mechanism"] == "camera_pan_both_static"
    profile_bound = motion_profile is not None
    template["reuse_contract"] = {
        "camera_and_room": (
            "retained Apartment native room; mechanism-only common camera center "
            "with 75 applied listener orientations"
            if camera_pan
            else "retained Apartment native room; selected independent camera cluster"
        ),
        "actor_roots": (
            "all roots are consumed frame-by-frame from the semantic actor "
            "motion profile"
            if profile_bound
            else "both actor roots are held at exact retained native human readbacks"
        ),
        "actor_yaws": (
            "all actions, phases, native indices, and headings are consumed "
            "frame-by-frame from the semantic actor motion profile"
            if profile_bound
            else "both held actors face the camera center and remain static"
        ),
        "audio": "source1 controlled speech only; source2 is explicitly silent",
    }
    template["camera_trajectory_binding"] = {
        "schema": sensor_rig["schema"],
        "trajectory_id": sensor_rig["trajectory_id"],
        "frame_count": FRAME_COUNT,
    }
    if motion_profile is not None:
        profile_binding = {
            "schema": motion_profile["schema"],
            "profile_content_sha256": motion_profile["profile_content_sha256"],
            "frame_count": len(motion_profile["frames"]),
            "qualification_claim": False,
        }
        template["actor_motion_profile_binding"] = profile_binding
    template.pop("static_camera_upgrade", None)
    suite = deepcopy(base_suite)
    suite["scenarios"] = [template]
    suite["camera_upgrade"] = {
        "schema": "avengine_dynamic_spear_suite_camera_binding_v1",
        "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
        "frame_count": FRAME_COUNT,
        "qualification_claim": False,
    }
    if planning_profile:
        suite["authority"] = {
            "room_layout": "native Apartment map selected by source authority",
            "actor_state": "planning row plus runtime human registry",
            "audio": "planning semantic AudioProgram",
            "qualification_claim": False,
        }
    return suite


def _emitter_offset_space_from_registry(
    declaration: Mapping[str, Any],
) -> str:
    """Bind emitter offset space to the exact runtime-registry actor/anchor."""

    registry = _load(RUNTIME_REGISTRY)
    assets = registry.get("assets")
    _require(
        registry.get("schema") == "avengine_source_asset_runtime_registry_v1"
        and isinstance(assets, list),
        "runtime registry authority is invalid",
    )
    matches = [
        record
        for record in assets
        if isinstance(record, Mapping)
        and record.get("asset_id") == declaration.get("asset_id")
        and record.get("revision") == declaration.get("asset_revision")
    ]
    _require(
        len(matches) == 1,
        "actor declaration must resolve exactly once in runtime registry",
    )
    record = matches[0]
    anchors = record.get("emitter_anchors")
    default_anchor_id = record.get("default_emitter_anchor_id")
    anchor_matches = [
        anchor
        for anchor in anchors
        if isinstance(anchors, list)
        and isinstance(anchor, Mapping)
        and anchor.get("anchor_id") == declaration.get("emitter_anchor_id")
        and anchor.get("anchor_id") == default_anchor_id
    ]
    _require(
        len(anchor_matches) == 1
        and anchor_matches[0].get("offset_m") == declaration.get("emitter_offset_m")
        and anchor_matches[0].get("offset_space") == "final_scaled_asset_root"
        and (
            "emitter_offset_space" not in declaration
            or declaration["emitter_offset_space"] == anchor_matches[0]["offset_space"]
        ),
        "actor emitter declaration/runtime registry drift",
    )
    return str(anchor_matches[0]["offset_space"])


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
                "offset_space": _emitter_offset_space_from_registry(declaration),
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


def _controlled_target_sound(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    catalog = _load(CONTROLLED_SOUND_CONTENT_REGISTRY)
    sound_asset_id = str(row["target"]["sound_asset_id"])
    matches = [
        item
        for item in catalog.get("assets", [])
        if item.get("sound_asset_id") == sound_asset_id
    ]
    _require(len(matches) == 1, "target controlled sound must resolve exactly once")
    sound = matches[0]
    target = row["target"]
    sample_rate_hz = int(sound["audio"]["sample_rate_hz"])
    channel_count = int(sound["audio"]["channel_count"])
    audio_uri = sound["audio"]["uri"]
    declared_sample_rate_hz = target.get("speech_sample_rate_hz")
    declared_channel_count = target.get("speech_channel_count")
    declared_audio_uri = target.get("speech_audio_uri")
    _require(
        sound["content"]["speaker_id"] == target["voice_id"]
        and sound["content"]["statement_id"] == target["content_id"]
        and sound["content"]["transcript"] == target["transcript"]
        and int(sound["audio"]["sample_count"]) == int(target["speech_sample_count"])
        and sample_rate_hz == 16_000
        and (
            "speech_sample_rate_hz" not in target
            or declared_sample_rate_hz == sample_rate_hz
        )
        and channel_count == 1
        and (
            "speech_channel_count" not in target
            or declared_channel_count == channel_count
        )
        and isinstance(audio_uri, str)
        and audio_uri
        and ("speech_audio_uri" not in target or declared_audio_uri == audio_uri),
        "target controlled sound metadata drift",
    )
    audio_path = CONTROLLED_AUDIO_ROOT / f"{sound_asset_id}.wav"
    _require(audio_path.is_file(), f"target controlled audio is missing: {audio_path}")
    return sound, audio_path


def _planning_target_audio(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Resolve one planning event by semantic IDs and exact row sample bounds."""

    target = row.get("target")
    audio = row.get("audio_program")
    _require(isinstance(target, Mapping), "planning target audio role is missing")
    _require(isinstance(audio, Mapping), "planning AudioProgram is missing")
    event = audio.get("target_event")
    _require(isinstance(event, Mapping), "planning target audio event is missing")
    start = event.get("start_sample")
    end = event.get("end_sample_exclusive")
    source_count = event.get("source_sample_count")
    _require(
        audio.get("mode") == "one_active_of_n"
        and audio.get("active_source_slots") == ["source1"]
        and audio.get("silent_source_slots") == ["source2"]
        and event.get("sound_asset_id") == target.get("sound_asset_id")
        and event.get("voice_id") == target.get("voice_id")
        and event.get("content_id") == target.get("content_id")
        and type(start) is int
        and type(end) is int
        and type(source_count) is int
        and 0 <= start < end <= 80_000
        and end - start == source_count == target.get("speech_sample_count")
        and event.get("source_sample_rate_hz")
        == target.get("speech_sample_rate_hz")
        == 16_000
        and event.get("source_channel_count") == target.get("speech_channel_count") == 1
        and event.get("source_audio_uri") == target.get("speech_audio_uri"),
        "planning target audio semantic/sample authority drift",
    )
    _, path = _controlled_target_sound(row)
    return deepcopy(dict(event)), path


def _copy_audio_contracts(
    audio_template: Path | None,
    output: Path,
    row: Mapping[str, Any],
    *,
    planning_mode: bool = False,
) -> None:
    if planning_mode:
        _write_planning_audio_contracts(output, row)
        return
    _require(audio_template is not None, "legacy audio template is required")
    target = output / "controlled_audio_program"
    target.mkdir()
    controlled_sound, target_audio = _controlled_target_sound(row)

    endpoints = _load(audio_template / "source_endpoint_registry.json")
    roles = {"source1": row["target"], "source2": row["distractor"]}
    for endpoint in endpoints["source_endpoints"]:
        slot = str(endpoint["binding"]["entity_instance_id"])
        role = roles[slot]
        endpoint["binding"]["entity_asset_id"] = role["runtime_asset_id"]
        endpoint["binding"]["entity_asset_revision"] = role["runtime_revision"]
    endpoints = bind_content_hash(endpoints)

    sounds = _load(audio_template / "sound_asset_registry.json")
    _require(len(sounds.get("sound_assets", [])) == 1, "sound template closure drift")
    sound = sounds["sound_assets"][0]
    sound_asset_id = str(controlled_sound["sound_asset_id"])
    sound["sound_asset_id"] = sound_asset_id
    sound["instance_lineage_id"] = controlled_sound["content"]["speaker_id"]
    sound["dry_audio"] = {
        "channel_count": int(controlled_sound["audio"]["channel_count"]),
        "sample_count": int(controlled_sound["audio"]["sample_count"]),
        "sample_rate_hz": int(controlled_sound["audio"]["sample_rate_hz"]),
        "sha256": controlled_sound["audio"]["sha256"],
        "uri": controlled_sound["audio"]["uri"],
    }
    sound["provenance"] = {
        "license": controlled_sound["license"]["license_id"],
        "origin": "lead_b_controlled_sound_content_registry_v1",
        "rights_evidence_sha256": controlled_sound["license"]["evidence"]["sha256"],
        "rights_status": "licensed",
    }
    sounds = bind_content_hash(sounds)

    program = _load(audio_template / "audio_program.json")
    _require(len(program.get("events", [])) == 1, "audio template must have one event")
    event = program["events"][0]
    speech_window = [
        int(value) for value in row["target"]["speech_frame_window_inclusive"]
    ]
    source_duration = int(controlled_sound["audio"]["sample_count"])
    start_sample = sample_boundary(speech_window[0]) + SPEECH_INTRA_FRAME_OFFSET_SAMPLES
    end_sample = start_sample + source_duration
    first_frame = start_sample * FRAME_RATE_HZ // 16000
    last_frame = (end_sample - 1) * FRAME_RATE_HZ // 16000
    _require(
        [first_frame, last_frame] == speech_window,
        "controlled speech duration does not match the declared frame window",
    )
    event["sound_asset_id"] = sound_asset_id
    event["source_start_sample"] = 0
    event["source_end_sample_exclusive"] = source_duration
    event["start_sample"] = start_sample
    event["end_sample_exclusive"] = end_sample
    event["start_tick"] = start_sample * 3
    event["end_tick_exclusive"] = end_sample * 3
    program = bind_audio_program_hash(program)

    binding = _load(audio_template / "controlled_audio_binding.json")
    binding["episode_id"] = row["episode_id"]
    binding["controlled_content"]["source1"] = {
        "language": controlled_sound["content"]["language"],
        "sound_asset_id": sound_asset_id,
        "statement_id": controlled_sound["content"]["statement_id"],
        "transcript": controlled_sound["content"]["transcript"],
    }
    binding["controlled_content"]["source2"] = None
    binding["sound_audio_paths"] = {sound_asset_id: str(target_audio)}
    binding["status"] = "pass_materialized_pending_exact_rir_render"

    for name, value in (
        ("source_endpoint_registry.json", endpoints),
        ("sound_asset_registry.json", sounds),
        ("audio_program.json", program),
        ("controlled_audio_binding.json", binding),
    ):
        write_json(target / name, value)


def _write_planning_audio_contracts(output: Path, row: Mapping[str, Any]) -> None:
    target = output / "controlled_audio_program"
    target.mkdir()
    event_authority, target_audio = _planning_target_audio(row)
    endpoint_ids = {
        "source1_emitter": "source1",
        "source2_emitter": "source2",
    }
    endpoint_registry = {
        "schema": "avengine_semantic_source_endpoint_registry_v1",
        "registry_id": f"{row['episode_id']}__semantic_endpoints",
        "revision": "planning_v1",
        "source_endpoint_ids": endpoint_ids,
    }
    content_id = str(event_authority["content_id"])
    content_registry = {
        "schema": "avengine_semantic_sound_content_registry_v1",
        "registry_id": f"{row['episode_id']}__semantic_sound_content",
        "revision": "planning_v1",
        "contents": [
            {
                "content_id": content_id,
                "sound_asset_id": event_authority["sound_asset_id"],
                "voice_id": event_authority["voice_id"],
                "source_audio_uri": event_authority["source_audio_uri"],
                "sample_rate_hz": event_authority["source_sample_rate_hz"],
                "channel_count": event_authority["source_channel_count"],
                "sample_count": event_authority["source_sample_count"],
            }
        ],
    }
    start = int(event_authority["start_sample"])
    end = int(event_authority["end_sample_exclusive"])
    program = {
        "schema": "avengine_semantic_audio_program_v1",
        "program_id": f"{row['episode_id']}__semantic_audio",
        "revision": "planning_v1",
        "mode": "one_active_of_n",
        "timeline": {
            "time_base_hz": 48_000,
            "ticks_per_frame": TICKS_PER_FRAME,
            "video_fps": FRAME_RATE_HZ,
            "frame_count": FRAME_COUNT,
            "sample_rate_hz": 16_000,
            "ticks_per_sample": 3,
            "sample_count": 80_000,
        },
        "candidate_source_endpoint_ids": sorted(endpoint_ids),
        "events": [
            {
                "event_id": f"{row['episode_id']}__target_speech",
                "source_endpoint_id": "source1_emitter",
                "content_id": content_id,
                "start_tick": start * 3,
                "end_tick_exclusive": end * 3,
                "start_sample": start,
                "end_sample_exclusive": end,
                "source_start_sample": 0,
                "source_end_sample_exclusive": int(
                    event_authority["source_sample_count"]
                ),
                "source_sample_rate_hz": int(event_authority["source_sample_rate_hz"]),
                "source_channel_count": int(event_authority["source_channel_count"]),
                "source_sample_count": int(event_authority["source_sample_count"]),
                "linear_gain": 1.0,
                "fade_samples": 0,
                "render_source_stem": True,
            }
        ],
        "source_specific_stems": True,
        "admission_state": "research",
        "program_content_sha256": "PENDING",
    }
    program = bind_audio_program_hash(program)
    binding = {
        "schema": "avengine_semantic_audio_binding_v1",
        "episode_id": row["episode_id"],
        "variant_id": "A",
        "content_bindings": {
            content_id: {
                "content_id": content_id,
                "path": str(target_audio.resolve()),
                "sample_rate_hz": event_authority["source_sample_rate_hz"],
                "channel_count": event_authority["source_channel_count"],
                "sample_count": event_authority["source_sample_count"],
            }
        },
    }
    for name, value in (
        ("semantic_source_endpoint_registry.json", endpoint_registry),
        ("semantic_sound_content_registry.json", content_registry),
        ("audio_program.json", program),
        ("semantic_audio_binding.json", binding),
    ):
        write_json(target / name, value)


def _validate_audio_contracts(
    output: Path,
    *,
    target_audio: Path,
    expected_speech_window: Sequence[int],
) -> dict[str, Any]:
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
    expected_window = [int(value) for value in expected_speech_window]
    _require([first_frame, last_frame] == expected_window, "speech frame window drift")
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
                "path": str(target_audio),
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
        f"frame_{first_frame - 1}_silent": not frame_active(first_frame - 1),
        f"frame_{first_frame}_active": frame_active(first_frame),
        f"frame_{last_frame}_active": frame_active(last_frame),
        f"frame_{last_frame + 1}_silent": not frame_active(last_frame + 1),
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
        "target_sound_asset_id": event["sound_asset_id"],
        "target_active_sample_count": int(
            event["end_sample_exclusive"] - event["start_sample"]
        ),
        "sample_count": 80000,
        "dry_bus_activity_checks": activity_checks,
    }


def _validate_planning_audio_contracts(
    output: Path,
    *,
    expected_event: Mapping[str, Any],
) -> dict[str, Any]:
    root = output / "controlled_audio_program"
    endpoints = _load(root / "semantic_source_endpoint_registry.json")
    contents = _load(root / "semantic_sound_content_registry.json")
    program = _load(root / "audio_program.json")
    binding = _load(root / "semantic_audio_binding.json")
    assembled = assemble_semantic_audio_program_dry_buses(
        program,
        "A",
        source_endpoint_ids=endpoints["source_endpoint_ids"],
        semantic_content_registry=contents,
        content_bindings=binding["content_bindings"],
    )
    compiled = assembled.compiled_program
    _require(len(compiled.events) == 1, "planning target must have one audio event")
    event = compiled.events[0]
    _require(
        event.start_sample == expected_event["start_sample"]
        and event.end_sample_exclusive == expected_event["end_sample_exclusive"]
        and event.sound_asset_id == expected_event["content_id"],
        "planning AudioProgram exact event authority drift",
    )
    buses = bind_endpoint_buses_to_source_slots(
        assembled.dry_audio.buses,
        endpoint_to_source_slot=endpoints["source_endpoint_ids"],
        source_slots=("source1", "source2"),
    )
    _require(
        not np.any(buses["source1"][: event.start_sample])
        and np.any(buses["source1"][event.start_sample : event.end_sample_exclusive])
        and not np.any(buses["source1"][event.end_sample_exclusive :])
        and not np.any(buses["source2"]),
        "planning semantic dry-bus activity drift",
    )
    return {
        "status": "pass_semantic_content_exact_event_no_file_digest",
        "target_event_count": 1,
        "distractor_event_count": 0,
        "target_content_id": expected_event["content_id"],
        "start_sample": event.start_sample,
        "end_sample_exclusive": event.end_sample_exclusive,
        "target_active_sample_count": event.end_sample_exclusive - event.start_sample,
        "sample_count": 80_000,
        "binding_mode": "semantic_content_id_and_declared_audio_metadata_v1",
    }


def _acoustic_execution_request(
    *,
    output: Path,
    episode_id: str,
    rir_plan: Mapping[str, Any],
    target_sound_asset_id: str,
    target_audio: Path,
    planning_mode: bool = False,
) -> dict[str, Any]:
    cache = output / "rir_cache_v3"
    audio_output = output / "binaural_v1"
    audio_root = output / "controlled_audio_program"
    binaural_argv = [
        "env",
        f"PYTHONPATH={REPOSITORY / 'src'}",
        str(HABITAT_PYTHON),
        "tools/m7/render_asset_bound_binaural_batch.py",
        "--plan-root",
        str(output.resolve()),
        "--rir-cache",
        str(cache.resolve()),
        "--audio-program",
        str((audio_root / "audio_program.json").resolve()),
        "--audio-program-variant",
        "A",
    ]
    if planning_mode:
        binaural_argv.extend(
            [
                "--semantic-source-endpoint-registry",
                str((audio_root / "semantic_source_endpoint_registry.json").resolve()),
                "--semantic-sound-content-registry",
                str((audio_root / "semantic_sound_content_registry.json").resolve()),
                "--semantic-audio-binding",
                str((audio_root / "semantic_audio_binding.json").resolve()),
            ]
        )
    else:
        binaural_argv.extend(
            [
                "--source-endpoint-registry",
                str((audio_root / "source_endpoint_registry.json").resolve()),
                "--sound-asset-registry",
                str((audio_root / "sound_asset_registry.json").resolve()),
                "--source-endpoint-slot",
                "lead_d_source1_mouth=source1",
                "--source-endpoint-slot",
                "lead_d_source2_mouth=source2",
                "--sound-audio",
                f"{target_sound_asset_id}={target_audio}",
            ]
        )
    binaural_argv.extend(
        [
            "--variants-per-episode",
            "1",
            "--retain-stems",
            "--output",
            str(audio_output.resolve()),
        ]
    )
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
            "tools/acoustics/render_rir_cache.py",
            *(["--semantic-no-file-evidence"] if planning_mode else []),
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
        "binaural_render_argv": binaural_argv,
    }


def _materialize_into(
    *,
    preflight_path: Path | None,
    canary_index: int,
    base_suite_path: Path | None,
    audio_template: Path,
    output: Path,
    published_output: Path,
    motion_candidate_path: Path | None = None,
    planning_binding: Mapping[str, Any] | None = None,
    planning_profile: Mapping[str, Any] | None = None,
    planning_suite: Mapping[str, Any] | None = None,
) -> Path:
    _require(
        output.is_dir() and not any(output.iterdir()),
        f"materialization staging must be an empty directory: {output}",
    )
    planning_mode = planning_binding is not None and motion_candidate_path is None
    if planning_mode:
        assert planning_binding is not None
        _require(
            planning_profile is not None and planning_suite is not None,
            "planning materialization authorities were not prepared",
        )
        row = deepcopy(dict(planning_binding["value"]))
    else:
        _require(
            preflight_path is not None, "legacy materialization requires preflight"
        )
        preflight = _load(preflight_path)
        _require(preflight.get("schema") == PREFLIGHT_SCHEMA, "preflight schema drift")
        row = _selected_canary(preflight, canary_index)
    mechanism = str(row["mechanism"])
    _require(
        mechanism in MOTION_PROFILE_REQUIRED_MECHANISMS
        or mechanism in {"both_static", "camera_pan_both_static"},
        "unsupported dynamic mechanism",
    )
    _require(
        mechanism not in MOTION_PROFILE_REQUIRED_MECHANISMS
        or motion_candidate_path is not None
        or planning_mode,
        f"{mechanism} requires --motion-candidate; legacy root inference is disabled",
    )
    if not planning_mode:
        _require(
            {row["target"]["identity_key"], row["distractor"]["identity_key"]}
            == {"M", "F"},
            "dynamic canary requires the reviewed M/F runtime pair",
        )
    target_event: Mapping[str, Any] | None = None
    if planning_mode:
        target_event, target_audio = _planning_target_audio(row)
    else:
        _, target_audio = _controlled_target_sound(row)
    required_paths = [
        CONTROLLED_SOUND_CONTENT_REGISTRY,
        ACOUSTIC_PACKAGE,
        SIMULATION_REQUEST,
        HABITAT_PYTHON,
        target_audio,
    ]
    if not planning_mode:
        required_paths.append(audio_template / "audio_program.json")
    if not planning_mode:
        assert base_suite_path is not None
        required_paths.append(base_suite_path)
    if motion_candidate_path is not None:
        required_paths.append(motion_candidate_path)
    for path in required_paths:
        _require(path.is_file(), f"required input is missing: {path}")

    base_suite = (
        deepcopy(dict(planning_suite)) if planning_mode else _load(base_suite_path)  # type: ignore[arg-type]
    )
    motion_profile: dict[str, Any] | None = None
    if planning_mode or mechanism in MOTION_PROFILE_REQUIRED_MECHANISMS:
        if planning_mode:
            motion_profile = deepcopy(dict(planning_profile))
            motion_candidate = motion_profile["authorities"]["candidate"]["value"]
        else:
            assert motion_candidate_path is not None
            assert preflight_path is not None
            assert base_suite_path is not None
            motion_candidate = _load(motion_candidate_path)
            motion_profile = build_actor_motion_profile(
                candidate_path=motion_candidate_path,
                candidate=motion_candidate,
                old_preflight_path=preflight_path,
                selected_old_row=row,
                base_suite_path=base_suite_path,
                base_suite=base_suite,
            )
        validate_actor_motion_profile(motion_profile)
        (
            declarations,
            actor_frames,
            root_paths,
            emitter_paths,
            animation_timing,
        ) = _actor_materialization_from_profile(motion_profile)
        motion_case = str(motion_candidate["mechanism"])
    else:
        declarations_by_identity = _identity_declarations(base_suite)
        source_scenarios = _source_scenarios(row)
        (
            actor_frames,
            root_paths,
            emitter_paths,
            animation_timing,
        ) = _actor_materialization(
            row=row,
            source_scenarios=source_scenarios,
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
        motion_case = str(LEGACY_CAMERA_PAN_ACOUSTICS["motion_case"])
    sensor_rig = _sensor_rig(row)
    listener_orientations_xyzw = [
        tuple(float(value) for value in frame["world_from_rig"]["rotation_xyzw"])
        for frame in sensor_rig["frames"]
    ]
    distinct_listener_orientation_count = len(set(listener_orientations_xyzw))
    if mechanism == "camera_pan_both_static":
        _require(
            distinct_listener_orientation_count == FRAME_COUNT,
            "camera pan must apply 75 distinct listener orientations",
        )
    else:
        _require(
            distinct_listener_orientation_count == 1,
            f"{mechanism} must retain one listener orientation",
        )
    suite = _suite(
        base_suite=base_suite,
        row=row,
        declarations=declarations,
        actor_frames=actor_frames,
        sensor_rig=sensor_rig,
        motion_profile=motion_profile,
    )

    episode = TrajectoryEpisode(
        episode_id=row["episode_id"],
        motion_case=motion_case,
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
    if motion_profile is not None:
        motion_candidate = motion_profile["authorities"]["candidate"]["value"]
        if is_planning_actor_motion_profile(motion_profile):
            root_path_provenance = {
                str(slot): {
                    "method": "planning_row_runtime_profile_v1",
                    "planning_source_path_authority": deepcopy(
                        actor.get("planning_source_path_authority")
                    ),
                    "runtime_motion_authority": deepcopy(
                        actor.get("runtime_motion_authority")
                    ),
                    "native_rate_active_interval": deepcopy(
                        actor.get("native_rate_active_interval")
                    ),
                    "native_rate_action_segments": deepcopy(
                        actor.get("native_rate_action_segments", [])
                    ),
                }
                for slot, actor in motion_candidate["actors"].items()
            }
        else:
            root_path_provenance = {
                str(slot): {
                    "method": "hash_bound_actor_motion_profile_v1",
                    "profile_content_sha256": motion_profile["profile_content_sha256"],
                    "native_motion_authority": deepcopy(
                        actor.get("native_motion_authority")
                    ),
                    "native_rate_active_interval": deepcopy(
                        actor.get("native_rate_active_interval")
                    ),
                    "native_rate_action_segments": deepcopy(
                        actor.get("native_rate_action_segments", [])
                    ),
                }
                for slot, actor in motion_candidate["actors"].items()
            }
    else:
        root_path_provenance = {
            "source1": deepcopy(
                row["target"].get(
                    "path_provenance",
                    {"method": "exact_selected_retained_native_actor_roots_v1"},
                )
            ),
            "source2": deepcopy(
                row["distractor"].get(
                    "path_provenance",
                    {"method": "exact_selected_retained_native_actor_roots_v1"},
                )
            ),
        }
    trajectory_bank["path_semantics"] = {
        "source_center_paths_m": "identity-bound world mouth emitter points",
        "source_root_paths_m": (
            "exactly applied requested roots; native-vs-derived authority is "
            "declared separately for each source slot"
        ),
        "source_root_path_provenance": root_path_provenance,
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
        rir_plan["requested_pair_state_count"] == FRAME_COUNT * len(root_paths),
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
    if motion_profile is not None:
        rir_profile_comparison = _validate_rir_plan_against_profile(
            rir_plan, motion_profile
        )
    else:
        _require(
            per_slot_distinct == LEGACY_CAMERA_PAN_ACOUSTICS["per_slot_distinct"],
            "legacy camera pan RIR slot state count drift",
        )
        _require(
            rir_plan["unique_rir_job_count"] == LEGACY_CAMERA_PAN_ACOUSTICS["unique"],
            "legacy camera pan total RIR state count drift",
        )
        _require(
            rir_plan["cache_reuse_count"] == LEGACY_CAMERA_PAN_ACOUSTICS["reuse"],
            "legacy camera pan exact cache reuse count drift",
        )
        rir_profile_comparison = None

    output.mkdir(parents=True, exist_ok=True)
    _copy_audio_contracts(
        None if planning_mode else audio_template,
        output,
        row,
        planning_mode=planning_mode,
    )
    audio_validation = (
        _validate_planning_audio_contracts(
            output,
            expected_event=target_event,
        )
        if planning_mode and target_event is not None
        else _validate_audio_contracts(
            output,
            target_audio=target_audio,
            expected_speech_window=row["target"]["speech_frame_window_inclusive"],
        )
    )
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
    if motion_profile is not None:
        relative_paths["actor_motion_profile"] = "actor_motion_profile.json"
    work_paths = {key: output / name for key, name in relative_paths.items()}
    published_paths = {
        key: published_output / name for key, name in relative_paths.items()
    }
    write_json(work_paths["suite_execution_plan"], suite)
    write_json(work_paths["trajectory_bank"], trajectory_bank)
    write_json(work_paths["sensor_rig_trajectory"], sensor_rig)
    write_json(work_paths["asset_emitter_binding_report"], binding_report)
    write_json(work_paths["rir_job_plan"], rir_plan)
    if motion_profile is not None:
        write_json(work_paths["actor_motion_profile"], motion_profile)
    request = _acoustic_execution_request(
        output=published_output,
        episode_id=row["episode_id"],
        rir_plan=rir_plan,
        target_sound_asset_id=row["target"]["sound_asset_id"],
        target_audio=target_audio,
        planning_mode=planning_mode,
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
            "root_path_provenance": trajectory_bank["path_semantics"][
                "source_root_path_provenance"
            ],
            "animation_timing": animation_timing,
        },
        "suite_camera_application": {
            "status": "pass_exact_all_75_frames",
            "applied_frame_count": 75,
            "listener_coupled_to_camera": True,
            "distinct_listener_pose_count": rir_plan["unique_listener_pose_count"],
            "distinct_listener_orientation_count": (
                distinct_listener_orientation_count
            ),
            "habitat_yaw_path_deg": [
                float(value) for value in row["camera"]["yaw_path_deg"]
            ],
            "habitat_yaw_span_deg": (
                max(float(value) for value in row["camera"]["yaw_path_deg"])
                - min(float(value) for value in row["camera"]["yaw_path_deg"])
            ),
        },
        "dynamic_acoustics": {
            "status": "planned_not_run",
            "frame_stride": 1,
            "requested_source_frame_uses": rir_plan["requested_pair_state_count"],
            "distinct_rir_state_count": rir_plan["unique_rir_job_count"],
            "distinct_rir_state_count_by_source_slot": per_slot_distinct,
            "exact_pose_cache_reuse_count": rir_plan["cache_reuse_count"],
            "cache_reuse_policy": (
                "source position, listener position, and listener orientation "
                "must be exactly identical"
            ),
            "actor_motion_profile_comparison": rir_profile_comparison,
        },
        "audio_program": {
            "validation": audio_validation,
            "target_source_slot": "source1",
            "target_event_count": 1,
            "distractor_source_slot": "source2",
            "distractor_event_count": 0,
            "target_speech_start_sample": (
                target_event["start_sample"]
                if target_event is not None
                else sample_boundary(
                    int(row["target"]["speech_frame_window_inclusive"][0])
                )
                + 128
            ),
            "target_sound_asset_id": row["target"]["sound_asset_id"],
            "target_audio_path": str(target_audio),
            "sample_rate_hz": 16000,
            "sample_count": 80000,
        },
        "actor_motion_profile": (
            (
                {
                    "status": "pass_bound_and_consumed_frame_by_frame",
                    "schema": motion_profile["schema"],
                    "profile_content_sha256": motion_profile["profile_content_sha256"],
                    "canonical_frame_sha256": [
                        frame["canonical_frame_sha256"]
                        for frame in motion_profile["frames"]
                    ],
                    "derived_action_counts": action_counts,
                    "derived_rir_counts": rir_profile_comparison["compared_counts"],
                    "legacy_root_motion_inference_used": False,
                    "qualification_claim": False,
                }
                if is_planning_actor_motion_profile(motion_profile)
                else {
                    "status": "pass_bound_and_consumed_frame_by_frame",
                    "schema": motion_profile["schema"],
                    "profile_content_sha256": motion_profile["profile_content_sha256"],
                    "candidate_value_sha256": motion_profile["authorities"][
                        "candidate"
                    ]["canonical_value_sha256"],
                    "canonical_frame_sha256": [
                        frame["canonical_frame_sha256"]
                        for frame in motion_profile["frames"]
                    ],
                    "derived_action_counts": action_counts,
                    "derived_rir_counts": rir_profile_comparison["compared_counts"],
                    "legacy_root_motion_inference_used": False,
                    "qualification_claim": False,
                }
            )
            if motion_profile is not None and rir_profile_comparison is not None
            else {
                "status": "explicit_legacy_camera_pan_adapter",
                "legacy_root_motion_inference_used": True,
                "qualification_claim": False,
            }
        ),
        "planning_episode_authority": (
            {
                "status": "pass_absolute_manifest_unique_episode_and_semantic_binding",
                "path": planning_binding["path"],
                "json_pointer": planning_binding["json_pointer"],
                "episode_id": planning_binding["value"]["episode_id"],
                "mechanism": planning_binding["value"]["mechanism"],
            }
            if planning_binding is not None
            else None
        ),
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


def _fresh_output_path(raw_output: Path) -> Path:
    """Return an absolute output path after rejecting lexical symlink traversal."""

    output = Path(raw_output)
    if not output.is_absolute():
        output = Path.cwd() / output
    _require(".." not in output.parts, "output path may not contain parent traversal")
    current = Path(output.anchor)
    for part in output.parts[1:]:
        current /= part
        _require(
            not current.is_symlink(),
            f"output path may not contain symlinks: {current}",
        )
    _require(
        not output.exists(),
        f"refusing to overwrite output: {output}",
    )
    return output


def _publish_staging(staging: Path, output: Path) -> Path:
    """Publish without replacement and remove unpublished staging on failure."""

    publish_policy = WorkspacePathPolicy.from_roots([output.parent])
    try:
        return atomic_publish_directory(publish_policy, staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize(
    *,
    preflight_path: Path | None,
    canary_index: int,
    base_suite_path: Path | None,
    audio_template: Path,
    output: Path,
    motion_candidate_path: Path | None = None,
    planning_manifest_path: Path | None = None,
    planning_episode_id: str | None = None,
) -> Path:
    """Build in staging and publish either a complete result or one failure receipt."""

    output = _fresh_output_path(output)
    planning_binding: Mapping[str, Any] | None = None
    planning_profile: dict[str, Any] | None = None
    planning_suite: dict[str, Any] | None = None
    planning_values = (
        planning_manifest_path,
        planning_episode_id,
    )
    if any(value is not None for value in planning_values):
        _require(
            all(value is not None for value in planning_values),
            "planning manifest reference requires path and episode selector",
        )
        assert planning_manifest_path is not None
        assert planning_episode_id is not None
        planning_binding = bind_planning_episode(
            planning_manifest_path=planning_manifest_path,
            episode_id=planning_episode_id,
        )
        if motion_candidate_path is not None:
            _require(
                preflight_path is not None,
                "legacy planning binding with motion candidate requires preflight",
            )
            _validate_planning_materialization_authority(
                binding=planning_binding,
                preflight_path=preflight_path,
                canary_index=canary_index,
                motion_candidate_path=motion_candidate_path,
            )
        else:
            row = planning_binding["value"]
            mechanism = row.get("mechanism")
            if mechanism in MOTION_PROFILE_REQUIRED_MECHANISMS:
                roles = (row.get("target"), row.get("distractor"))
                _require(
                    all(
                        isinstance(role, Mapping)
                        and isinstance(role.get("motion_profile_authority"), Mapping)
                        for role in roles
                    ),
                    (
                        "planning dynamic row lacks a generic native motion "
                        "authority; refusing natural-cadence inference"
                    ),
                )
            _require(
                row.get("mechanism")
                in MOTION_PROFILE_REQUIRED_MECHANISMS
                | {"both_static", "camera_pan_both_static"},
                "planning-only materialization requires a supported mechanism",
            )
            planning_profile, planning_suite = _planning_materialization_authorities(
                planning_binding
            )
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
            motion_candidate_path=motion_candidate_path,
            planning_binding=planning_binding,
            planning_profile=planning_profile,
            planning_suite=planning_suite,
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
        _publish_staging(staging, output)
        raise
    _publish_staging(staging, output)
    return output / "materialization_receipt.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--canary-index", type=int, default=1)
    parser.add_argument("--base-suite", type=Path, default=BASE_SUITE)
    parser.add_argument("--audio-template", type=Path, default=BASE_AUDIO)
    parser.add_argument("--motion-candidate", type=Path)
    parser.add_argument("--planning-manifest", type=Path)
    parser.add_argument("--planning-episode-id")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    receipt = materialize(
        preflight_path=(
            args.preflight.resolve()
            if args.preflight
            and not (
                args.planning_manifest is not None and args.motion_candidate is None
            )
            else None
        ),
        canary_index=args.canary_index,
        base_suite_path=(
            args.base_suite.resolve()
            if args.base_suite
            and not (
                args.planning_manifest is not None and args.motion_candidate is None
            )
            else None
        ),
        audio_template=args.audio_template.resolve(),
        output=args.output,
        motion_candidate_path=(
            args.motion_candidate.resolve() if args.motion_candidate else None
        ),
        planning_manifest_path=(
            args.planning_manifest.resolve() if args.planning_manifest else None
        ),
        planning_episode_id=args.planning_episode_id,
    )
    print(f"STRICT_TWO_HUMAN_DYNAMIC_MATERIALIZATION_OK receipt={receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
