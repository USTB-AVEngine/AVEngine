#!/usr/bin/env python3
"""Build CPU-only native-rate full75 dynamic candidate preflights.

The retained target/distractor/both moving windows are placed inside a 75-frame
episode at their native 15 Hz rate.  Frames outside each active interval bind
Idle and hold the nearest boundary root.  This builder never launches Unreal,
does not reuse the legacy stretched RIRs, and deliberately emits
``RELEASE_BLOCKED`` receipts until fresh pixels, live ground contact, and live
foot-plant evidence exist.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

FRAME_COUNT = 75
FRAME_RATE_HZ = 15
EPISODE_DURATION_SECONDS = 5.0
TIMELINE_TICKS_PER_SECOND = 48_000
ANIMATION_TICKS_PER_PHASE_CYCLE = 51_200
FRAME_TICKS = TIMELINE_TICKS_PER_SECOND // FRAME_RATE_HZ

EXPECTED_PHASE_CADENCE_HZ = 0.9375
EXPECTED_PHASE_ADVANCE_PER_FRAME = EXPECTED_PHASE_CADENCE_HZ / FRAME_RATE_HZ
EXPECTED_NATIVE_SPEED_MIN_M_S = 0.73
# 0.851 retains the measured 0.850134 m/s source window while its rounded
# release-facing envelope remains 0.73--0.85 m/s.
EXPECTED_NATIVE_SPEED_MAX_M_S = 0.851
MAX_SPEED_RELATIVE_ERROR = 1.0e-6
NUMERIC_TOLERANCE = 1.0e-9
MINIMUM_DEAD_ZONE_MARGIN_FRACTION = 0.05
MINIMUM_ACTOR_SEPARATION_M = 1.0
MINIMUM_CAMERA_DEPTH_M = 1.3
MAXIMUM_CAMERA_DEPTH_M = 6.5
SAFE_FRAME_MIN_FRACTION = 0.04
SAFE_FRAME_MAX_FRACTION = 0.96

PREFLIGHT_SCHEMA = (
    "avengine_strict_two_human_native_rate_dynamic_candidate_preflight_v1"
)
RECEIPT_SCHEMA = "avengine_strict_two_human_native_rate_dynamic_candidate_receipt_v1"
BLOCKED_STATUS = "RELEASE_BLOCKED"

CASE_ORDER = ("target_moves", "distractor_moves", "both_move")
CASE_OUTPUT_STEMS = {
    "target_moves": "native_strict_two_human_target_moves_native_rate_candidate_v1",
    "distractor_moves": (
        "native_strict_two_human_distractor_moves_native_rate_candidate_v1"
    ),
    "both_move": "native_strict_two_human_both_move_native_rate_candidate_v1",
}
MOVING_SLOTS = {
    "target_moves": ("source1",),
    "distractor_moves": ("source2",),
    "both_move": ("source1", "source2"),
}

RELEASE_BLOCKERS = [
    {
        "code": "fresh_native_pixels_not_verified",
        "message": (
            "the native-rate timing revision has no fresh normal/target-only "
            "full75 pixel capture"
        ),
    },
    {
        "code": "live_ground_contact_not_verified",
        "message": (
            "live floor identity, foot/toe traces, and residual ground gaps are pending"
        ),
    },
    {
        "code": "live_foot_plant_sync_not_verified",
        "message": (
            "canonical contact phases and planted-foot slip are not yet measured"
        ),
    },
    {
        "code": "live_walking_asset_readback_not_verified",
        "message": (
            "CPU phase cadence is retained-native authority; live Walking and "
            "Idle/Walking skeletal transition continuity are not read back"
        ),
    },
    {
        "code": "retimed_exact_rir_not_built",
        "message": (
            "the unchanged sound event must be propagated along the new candidate "
            "emitter timing with a fresh exact RIR plan"
        ),
    },
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, value: object) -> None:
    _require(not path.exists(), f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _require(not temporary.exists(), f"stale temporary output exists: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _pairwise(values: Sequence[Any]):
    return pairwise(values)


def _horizontal_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(
        float(right[0]) - float(left[0]), float(right[2]) - float(left[2])
    )


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(_horizontal_distance(left, right) for left, right in _pairwise(points))


def _arc_length_resample(
    points: Sequence[Sequence[float]], output_count: int
) -> list[list[float]]:
    _require(
        len(points) >= 2 and output_count >= 2, "polyline resample is underdetermined"
    )
    cumulative = [0.0]
    for previous, current in _pairwise(points):
        cumulative.append(cumulative[-1] + _horizontal_distance(previous, current))
    _require(cumulative[-1] > 0.0, "moving native polyline has zero length")
    result: list[list[float]] = []
    segment = 0
    for output_index in range(output_count):
        distance = cumulative[-1] * output_index / (output_count - 1)
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < distance:
            segment += 1
        segment_length = cumulative[segment + 1] - cumulative[segment]
        _require(segment_length > 0.0, "native polyline contains a zero-length segment")
        alpha = (distance - cumulative[segment]) / segment_length
        result.append(
            [
                float(points[segment][axis])
                + alpha
                * (float(points[segment + 1][axis]) - float(points[segment][axis]))
                for axis in range(3)
            ]
        )
    return result


def _scenario(suite: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    scenarios = suite.get("scenarios")
    _require(
        isinstance(scenarios, list) and len(scenarios) == 1, "expected one scenario"
    )
    scenario = scenarios[0]
    _require(isinstance(scenario, Mapping), "scenario is not an object")
    plan = scenario.get("plan")
    _require(isinstance(plan, Mapping), "scenario plan is missing")
    frames = plan.get("frames")
    _require(
        isinstance(frames, list)
        and len(frames) == FRAME_COUNT
        and [frame.get("frame_index") for frame in frames] == list(range(FRAME_COUNT)),
        "legacy suite is not exact full75",
    )
    return scenario, plan


def _declarations(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    actors = plan.get("actors")
    _require(isinstance(actors, list) and len(actors) == 2, "expected two actors")
    result = {
        str(actor.get("actor_id")): actor
        for actor in actors
        if isinstance(actor, Mapping) and isinstance(actor.get("actor_id"), str)
    }
    _require(
        set(result) == {"source1_actor", "source2_actor"},
        "actor declaration identity drift",
    )
    return result


def _legacy_states(plan: Mapping[str, Any], actor_id: str) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for frame in plan["frames"]:
        matches = [
            item
            for item in frame.get("actor_states", [])
            if isinstance(item, Mapping) and item.get("actor_id") == actor_id
        ]
        _require(len(matches) == 1, f"{actor_id} does not resolve once per frame")
        result.append(matches[0])
    return result


def _native_window(
    *, timing: Mapping[str, Any], states: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    provenance = timing.get("path_provenance")
    _require(isinstance(provenance, Mapping), "moving timing lacks path provenance")
    native_range = provenance.get("native_source_frame_indices_inclusive")
    _require(
        isinstance(native_range, list)
        and len(native_range) == 2
        and all(isinstance(value, int) for value in native_range),
        "native source frame range is invalid",
    )
    native_start, native_end = int(native_range[0]), int(native_range[1])
    native_interval_count = native_end - native_start
    native_sample_count = native_interval_count + 1
    _require(native_interval_count > 0, "native window has no interval")
    _require(
        int(provenance.get("native_anchor_count", -1)) == native_sample_count,
        "native anchor/range closure failed",
    )

    stored_anchors = provenance.get("native_anchors_m")
    if isinstance(stored_anchors, list) and len(stored_anchors) == native_sample_count:
        anchors = [[float(value) for value in point] for point in stored_anchors]
        root_authority = "retained_exact_native_anchor_list_v1"
        exact_native_anchor_readback_claim = True
    else:
        legacy_roots = [state.get("translation_m") for state in states]
        _require(
            all(isinstance(point, list) and len(point) == 3 for point in legacy_roots),
            "legacy root path is incomplete",
        )
        anchors = _arc_length_resample(legacy_roots, native_sample_count)
        root_authority = (
            "retained_straight_polyline_equal_arc_native_count_reconstruction_v1"
        )
        exact_native_anchor_readback_claim = False

    measured_path_length = _path_length(anchors)
    declared_path_length = float(timing.get("path_length_m", math.nan))
    _require(
        math.isfinite(declared_path_length)
        and abs(measured_path_length - declared_path_length) <= 1.0e-6,
        "native path-length closure failed",
    )
    native_duration = native_interval_count / FRAME_RATE_HZ
    native_speed = measured_path_length / native_duration
    _require(
        EXPECTED_NATIVE_SPEED_MIN_M_S <= native_speed <= EXPECTED_NATIVE_SPEED_MAX_M_S,
        f"native speed is outside the retained 0.73--0.85 m/s envelope: {native_speed}",
    )

    stored_phases = provenance.get("native_phases_unwrapped")
    if isinstance(stored_phases, list) and len(stored_phases) == native_sample_count:
        phases = [float(value) for value in stored_phases]
        phase_authority = "retained_exact_native_unwrapped_phase_list_v1"
    else:
        phase_path = timing.get("action_phase_path")
        _require(
            isinstance(phase_path, list) and len(phase_path) == FRAME_COUNT,
            "legacy action phase path is missing",
        )
        start_phase = float(phase_path[0])
        total_advance = float(timing.get("phase_cycle_count", math.nan))
        _require(math.isfinite(total_advance), "native phase advance is missing")
        step = total_advance / native_interval_count
        phases = [start_phase + step * index for index in range(native_sample_count)]
        phase_authority = "retained_phase_endpoints_native_count_reconstruction_v1"

    phase_steps = [current - previous for previous, current in _pairwise(phases)]
    _require(
        phase_steps
        and max(abs(value - EXPECTED_PHASE_ADVANCE_PER_FRAME) for value in phase_steps)
        <= NUMERIC_TOLERANCE,
        "native phase path is not exact 0.0625 cycle/frame",
    )
    phase_cadence = (phases[-1] - phases[0]) / native_duration
    _require(
        abs(phase_cadence - EXPECTED_PHASE_CADENCE_HZ) <= NUMERIC_TOLERANCE,
        "native phase cadence is not 0.9375 Hz",
    )
    old_output_span = float(timing.get("episode_span_seconds", math.nan))
    old_speed = float(timing.get("average_root_speed_m_per_second", math.nan))
    _require(
        old_output_span > native_duration and old_speed < native_speed,
        "legacy stretch evidence drift",
    )
    return {
        "native_source_frame_range_inclusive": [native_start, native_end],
        "native_interval_count": native_interval_count,
        "native_sample_count": native_sample_count,
        "anchors_m": anchors,
        "unwrapped_phases": phases,
        "path_length_m": measured_path_length,
        "native_duration_seconds": native_duration,
        "native_average_speed_m_s": native_speed,
        "native_phase_cadence_hz": phase_cadence,
        "root_authority": root_authority,
        "phase_authority": phase_authority,
        "exact_native_anchor_readback_claim": exact_native_anchor_readback_claim,
        "native_source_scenario_id": provenance.get("native_source_scenario_id"),
        "native_source_actor_id": provenance.get("native_source_actor_id"),
        "legacy_full75_output_span_seconds": old_output_span,
        "legacy_full75_average_speed_m_s": old_speed,
        "legacy_global_stretch_factor": old_output_span / native_duration,
    }


def _active_start(*, native_interval_count: int, speech_window: Sequence[int]) -> int:
    speech_midpoint = (int(speech_window[0]) + int(speech_window[1])) / 2.0
    start = round(speech_midpoint - native_interval_count / 2.0)
    # Leave at least one Idle frame on both sides so action/root boundary
    # continuity is observable before GPU execution.
    return max(1, min(start, FRAME_COUNT - native_interval_count - 2))


def _translation_ue_cm(point: Sequence[float]) -> list[float]:
    return [float(point[0]) * 100.0, float(point[2]) * 100.0, float(point[1]) * 100.0]


def _moving_actor_plan(
    *,
    slot: str,
    declaration: Mapping[str, Any],
    legacy_states: Sequence[Mapping[str, Any]],
    native: Mapping[str, Any],
    speech_window: Sequence[int],
) -> dict[str, Any]:
    start = _active_start(
        native_interval_count=int(native["native_interval_count"]),
        speech_window=speech_window,
    )
    end = start + int(native["native_interval_count"])
    anchors = copy.deepcopy(native["anchors_m"])
    phases = list(native["unwrapped_phases"])
    roots: list[list[float]] = []
    actions: list[str] = []
    animations: list[str] = []
    wrapped_phases: list[float] = []
    action_ticks: list[int] = []
    native_frame_indices: list[int] = []
    modes: list[str] = []
    native_source_start = int(native["native_source_frame_range_inclusive"][0])
    for frame_index in range(FRAME_COUNT):
        if frame_index < start:
            root = anchors[0]
            action = "idle"
            animation = str(declaration["idle_animation"])
            phase = 0.0
            ticks = 0
            native_index = native_source_start
            mode = "held_idle_before_native_rate_active_interval_v1"
        elif frame_index <= end:
            local_index = frame_index - start
            root = anchors[local_index]
            action = "walk"
            animation = str(declaration["walking_animation"])
            unwrapped_phase = float(phases[local_index])
            ticks = round(unwrapped_phase * ANIMATION_TICKS_PER_PHASE_CYCLE)
            phase = (ticks / ANIMATION_TICKS_PER_PHASE_CYCLE) % 1.0
            native_index = native_source_start + local_index
            mode = "native_15hz_active_interval_v1"
        else:
            root = anchors[-1]
            action = "idle"
            animation = str(declaration["idle_animation"])
            phase = 0.0
            ticks = 0
            native_index = native_source_start + int(native["native_interval_count"])
            mode = "held_idle_after_native_rate_active_interval_v1"
        roots.append([float(value) for value in root])
        actions.append(action)
        animations.append(animation)
        wrapped_phases.append(phase)
        action_ticks.append(ticks)
        native_frame_indices.append(native_index)
        modes.append(mode)

    active_roots = roots[start : end + 1]
    active_path_length = _path_length(active_roots)
    active_duration = (end - start) / FRAME_RATE_HZ
    active_speed = active_path_length / active_duration
    native_speed = float(native["native_average_speed_m_s"])
    speed_relative_error = abs(active_speed / native_speed - 1.0)
    active_ticks = action_ticks[start : end + 1]
    tick_deltas = [current - previous for previous, current in _pairwise(active_ticks)]
    active_phase_advance = (
        active_ticks[-1] - active_ticks[0]
    ) / ANIMATION_TICKS_PER_PHASE_CYCLE
    active_cadence = active_phase_advance / active_duration
    pre_boundary_step = _horizontal_distance(roots[start - 1], roots[start])
    post_boundary_step = _horizontal_distance(roots[end], roots[end + 1])
    _require(speed_relative_error <= MAX_SPEED_RELATIVE_ERROR, "active speed drift")
    _require(
        all(value == FRAME_TICKS for value in tick_deltas), "active tick cadence drift"
    )
    _require(
        abs(active_cadence - EXPECTED_PHASE_CADENCE_HZ) <= NUMERIC_TOLERANCE,
        "active phase cadence drift",
    )
    _require(
        pre_boundary_step <= NUMERIC_TOLERANCE
        and post_boundary_step <= NUMERIC_TOLERANCE,
        "action boundary also teleports the actor root",
    )

    yaw = float(legacy_states[0].get("actor_yaw_ue_deg", 0.0))
    forward_habitat = copy.deepcopy(
        legacy_states[0].get("anatomical_forward_habitat_world", [0.0, 0.0, 1.0])
    )
    return {
        "slot_id": slot,
        "actor_id": f"{slot}_actor",
        "asset_id": declaration.get("asset_id"),
        "moving": True,
        "native_rate_active_interval": {
            "output_frame_range_inclusive": [start, end],
            "output_interval_count": end - start,
            "output_sample_count": end - start + 1,
            "native_source_frame_range_inclusive": copy.deepcopy(
                native["native_source_frame_range_inclusive"]
            ),
            "native_interval_count": native["native_interval_count"],
            "native_sample_count": native["native_sample_count"],
            "native_frame_rate_hz": FRAME_RATE_HZ,
            "output_frame_rate_hz": FRAME_RATE_HZ,
            "time_scale": 1.0,
            "global_time_stretch_applied": False,
            "outside_action_id": "idle",
            "outside_root_policy": "hold_nearest_boundary_root",
            "placement_policy": "center_native_window_on_unchanged_speech_window_v1",
            "speech_window_inclusive": list(speech_window),
            "speech_overlap_frame_count": max(
                0,
                min(end, int(speech_window[1])) - max(start, int(speech_window[0])) + 1,
            ),
        },
        "native_motion_authority": {
            key: copy.deepcopy(native[key])
            for key in (
                "native_source_frame_range_inclusive",
                "native_interval_count",
                "native_sample_count",
                "path_length_m",
                "native_duration_seconds",
                "native_average_speed_m_s",
                "native_phase_cadence_hz",
                "root_authority",
                "phase_authority",
                "exact_native_anchor_readback_claim",
                "native_source_scenario_id",
                "native_source_actor_id",
            )
        },
        "root_path_m": roots,
        "translation_ue_cm_path": [_translation_ue_cm(point) for point in roots],
        "action_id_path": actions,
        "ue_animation_path": animations,
        "action_phase_path": wrapped_phases,
        "action_time_ticks_path": action_ticks,
        "native_source_frame_index_path": native_frame_indices,
        "animation_timing_mode_path": modes,
        "actor_yaw_ue_deg_path": [yaw] * FRAME_COUNT,
        "anatomical_forward_habitat_world_path": [
            copy.deepcopy(forward_habitat) for _ in range(FRAME_COUNT)
        ],
        "trajectory_preflight": {
            "status": "PASS_CPU_NATIVE_RATE_TRAJECTORY",
            "active_path_length_m": active_path_length,
            "active_duration_seconds": active_duration,
            "active_average_speed_m_s": active_speed,
            "native_average_speed_m_s": native_speed,
            "speed_relative_error": speed_relative_error,
            "expected_rounded_speed_envelope_m_s": [0.73, 0.85],
            "maximum_unrounded_speed_m_s": EXPECTED_NATIVE_SPEED_MAX_M_S,
            "active_phase_advance_cycles": active_phase_advance,
            "active_phase_cadence_hz": active_cadence,
            "phase_advance_per_frame_cycles": EXPECTED_PHASE_ADVANCE_PER_FRAME,
            "timeline_ticks_per_second": TIMELINE_TICKS_PER_SECOND,
            "animation_ticks_per_phase_cycle": ANIMATION_TICKS_PER_PHASE_CYCLE,
            "active_tick_delta_path": tick_deltas,
            "pre_active_action_transition_root_step_m": pre_boundary_step,
            "post_active_action_transition_root_step_m": post_boundary_step,
            "action_transition_root_continuity": "pass",
            "skeletal_pose_transition_continuity_status": (
                "pending_live_runtime_blend_readback"
            ),
            "foot_plant_sync_status": "pending_live_runtime_evidence",
            "ground_contact_status": "pending_live_runtime_evidence",
        },
        "legacy_slow_motion_evidence": {
            "decision": "reject_nonrelease_pipeline_evidence_only",
            "legacy_full75_average_speed_m_s": native[
                "legacy_full75_average_speed_m_s"
            ],
            "native_rate_average_speed_m_s": native["native_average_speed_m_s"],
            "legacy_global_stretch_factor": native["legacy_global_stretch_factor"],
            "upgraded_to_pass": False,
        },
    }


def _static_actor_plan(
    *,
    slot: str,
    declaration: Mapping[str, Any],
    legacy_states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = [float(value) for value in legacy_states[0]["translation_m"]]
    yaw = float(legacy_states[0].get("actor_yaw_ue_deg", 0.0))
    forward_habitat = copy.deepcopy(
        legacy_states[0].get("anatomical_forward_habitat_world", [0.0, 0.0, 1.0])
    )
    return {
        "slot_id": slot,
        "actor_id": f"{slot}_actor",
        "asset_id": declaration.get("asset_id"),
        "moving": False,
        "native_rate_active_interval": None,
        "root_path_m": [copy.deepcopy(root) for _ in range(FRAME_COUNT)],
        "translation_ue_cm_path": [
            _translation_ue_cm(root) for _ in range(FRAME_COUNT)
        ],
        "action_id_path": ["idle"] * FRAME_COUNT,
        "ue_animation_path": [str(declaration["idle_animation"])] * FRAME_COUNT,
        "action_phase_path": [0.0] * FRAME_COUNT,
        "action_time_ticks_path": [0] * FRAME_COUNT,
        "native_source_frame_index_path": [None] * FRAME_COUNT,
        "animation_timing_mode_path": ["held_idle_all75_v1"] * FRAME_COUNT,
        "actor_yaw_ue_deg_path": [yaw] * FRAME_COUNT,
        "anatomical_forward_habitat_world_path": [
            copy.deepcopy(forward_habitat) for _ in range(FRAME_COUNT)
        ],
        "trajectory_preflight": {
            "status": "PASS_CPU_HELD_IDLE_TRAJECTORY",
            "horizontal_path_length_m": 0.0,
            "unique_root_count": 1,
            "foot_plant_sync_status": "not_applicable_idle",
            "ground_contact_status": "pending_live_runtime_evidence",
        },
    }


def _project(
    *,
    point: Sequence[float],
    height_m: float,
    camera_position: Sequence[float],
    camera_yaw_deg: float,
    horizontal_fov_deg: float,
) -> tuple[float, float, float]:
    yaw = math.radians(camera_yaw_deg)
    forward = (-math.sin(yaw), -math.cos(yaw))
    right = (-forward[1], forward[0])
    dx = float(point[0]) - float(camera_position[0])
    dz = float(point[2]) - float(camera_position[2])
    depth = dx * forward[0] + dz * forward[1]
    lateral = dx * right[0] + dz * right[1]
    tangent_horizontal = math.tan(math.radians(horizontal_fov_deg) / 2.0)
    tangent_vertical = tangent_horizontal * 720.0 / 1280.0
    x_fraction = 0.5 + lateral / (2.0 * depth * tangent_horizontal)
    y_fraction = 0.5 - (float(point[1]) + height_m - float(camera_position[1])) / (
        2.0 * depth * tangent_vertical
    )
    return depth, x_fraction, y_fraction


def _side(values: Sequence[float]) -> str:
    if max(values) < 0.5:
        return "left"
    if min(values) > 0.5:
        return "right"
    return "crosses_dead_zone"


def _projection_preflight(
    *,
    plan: Mapping[str, Any],
    actors: Mapping[str, Mapping[str, Any]],
    declarations: Mapping[str, Mapping[str, Any]],
    legacy_states_by_slot: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    camera = plan.get("camera")
    _require(
        isinstance(camera, Mapping) and camera.get("dynamic") is False,
        "camera is not static",
    )
    position = camera.get("habitat_position_m")
    _require(
        isinstance(position, list) and len(position) == 3, "camera position is invalid"
    )
    yaw = float(camera.get("habitat_yaw_deg", math.nan))
    hfov = float(camera.get("horizontal_fov_deg", math.nan))
    _require(
        math.isfinite(yaw) and 0.0 < hfov < 180.0,
        "camera projection contract is invalid",
    )
    actor_metrics: dict[str, Any] = {}
    for slot in ("source1", "source2"):
        actor = actors[slot]
        root_path = actor["root_path_m"]
        mouth_height = float(declarations[f"{slot}_actor"]["emitter_offset_m"][1])
        projections = [
            _project(
                point=point,
                height_m=0.0,
                camera_position=position,
                camera_yaw_deg=yaw,
                horizontal_fov_deg=hfov,
            )
            for point in root_path
        ]
        mouth_y = [
            _project(
                point=point,
                height_m=mouth_height,
                camera_position=position,
                camera_yaw_deg=yaw,
                horizontal_fov_deg=hfov,
            )[2]
            for point in root_path
        ]
        head_y = [
            _project(
                point=point,
                height_m=2.0,
                camera_position=position,
                camera_yaw_deg=yaw,
                horizontal_fov_deg=hfov,
            )[2]
            for point in root_path
        ]
        depths = [value[0] for value in projections]
        x_values = [value[1] for value in projections]
        root_y = [value[2] for value in projections]
        legacy_x_values = [
            _project(
                point=state["translation_m"],
                height_m=0.0,
                camera_position=position,
                camera_yaw_deg=yaw,
                horizontal_fov_deg=hfov,
            )[1]
            for state in legacy_states_by_slot[slot]
        ]
        expected_side = _side(legacy_x_values)
        observed_side = _side(x_values)
        dead_zone_margin = min(abs(value - 0.5) for value in x_values)
        _require(
            expected_side in {"left", "right"} and observed_side == expected_side,
            f"{slot} projected side drift",
        )
        _require(
            dead_zone_margin >= MINIMUM_DEAD_ZONE_MARGIN_FRACTION,
            f"{slot} enters the projected side dead zone",
        )
        _require(
            min(depths) >= MINIMUM_CAMERA_DEPTH_M
            and max(depths) <= MAXIMUM_CAMERA_DEPTH_M,
            f"{slot} projected depth is outside the strict corridor",
        )
        _require(
            min(x_values) > SAFE_FRAME_MIN_FRACTION
            and max(x_values) < SAFE_FRAME_MAX_FRACTION
            and min(root_y + mouth_y + head_y) > SAFE_FRAME_MIN_FRACTION
            and max(root_y + mouth_y + head_y) < SAFE_FRAME_MAX_FRACTION,
            f"{slot} static pinhole envelope exits the safe frame",
        )
        actor_metrics[slot] = {
            "expected_side_from_legacy_static_camera_projection": expected_side,
            "observed_side": observed_side,
            "center_x_fraction_range": [min(x_values), max(x_values)],
            "dead_zone_margin_fraction": dead_zone_margin,
            "camera_depth_m_range": [min(depths), max(depths)],
            "root_y_fraction_range": [min(root_y), max(root_y)],
            "mouth_y_fraction_range": [min(mouth_y), max(mouth_y)],
            "head_y_fraction_range": [min(head_y), max(head_y)],
        }

    separations = [
        _horizontal_distance(left, right)
        for left, right in zip(
            actors["source1"]["root_path_m"], actors["source2"]["root_path_m"]
        )
    ]
    _require(
        actor_metrics["source1"]["observed_side"]
        != actor_metrics["source2"]["observed_side"],
        "the two actors do not remain on opposite projected sides",
    )
    _require(
        min(separations) >= MINIMUM_ACTOR_SEPARATION_M,
        "actor separation falls below the CPU safety floor",
    )
    return {
        "status": "PASS_CPU_STATIC_PINHOLE_ONLY",
        "authority": "analytic_static_camera_pinhole_projection_v1",
        "claim_boundary": (
            "center/root/mouth/head analytic projection only; no fresh pixels, "
            "segmentation, occlusion, collision, or metric-depth claim"
        ),
        "camera": {
            "habitat_position_m": copy.deepcopy(position),
            "habitat_yaw_deg": yaw,
            "horizontal_fov_deg": hfov,
            "dynamic": False,
        },
        "actors": actor_metrics,
        "target_slot": "source1",
        "target_side": actor_metrics["source1"]["observed_side"],
        "minimum_actor_horizontal_separation_m": min(separations),
        "minimum_required_actor_horizontal_separation_m": MINIMUM_ACTOR_SEPARATION_M,
    }


def _frame_plan(actors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for frame_index in range(FRAME_COUNT):
        actor_states: list[dict[str, Any]] = []
        for slot in ("source1", "source2"):
            actor = actors[slot]
            actor_states.append(
                {
                    "actor_id": actor["actor_id"],
                    "slot_id": slot,
                    "translation_m": actor["root_path_m"][frame_index],
                    "translation_ue_cm": actor["translation_ue_cm_path"][frame_index],
                    "action_id": actor["action_id_path"][frame_index],
                    "ue_animation": actor["ue_animation_path"][frame_index],
                    "action_phase": actor["action_phase_path"][frame_index],
                    "action_time_ticks": actor["action_time_ticks_path"][frame_index],
                    "animation_timing_mode": actor["animation_timing_mode_path"][
                        frame_index
                    ],
                    "native_source_frame_index": actor[
                        "native_source_frame_index_path"
                    ][frame_index],
                    "actor_yaw_ue_deg": actor["actor_yaw_ue_deg_path"][frame_index],
                }
            )
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * FRAME_TICKS,
                "frame_coverage_end_ticks": (frame_index + 1) * FRAME_TICKS,
                "actor_states": actor_states,
            }
        )
    return frames


def _mechanism_preflight(
    mechanism: str,
    actors: Mapping[str, Mapping[str, Any]],
    speech_window: Sequence[int],
) -> dict[str, Any]:
    moving = tuple(slot for slot in ("source1", "source2") if actors[slot]["moving"])
    _require(moving == MOVING_SLOTS[mechanism], "moving-slot mechanism closure failed")
    active_sets: dict[str, set[int]] = {}
    for slot in moving:
        start, end = actors[slot]["native_rate_active_interval"][
            "output_frame_range_inclusive"
        ]
        active_sets[slot] = set(range(start, end + 1))
    speech_set = set(range(int(speech_window[0]), int(speech_window[1]) + 1))
    if mechanism == "target_moves":
        _require(
            speech_set <= active_sets["source1"],
            "target active interval no longer covers the unchanged speech window",
        )
        overlap = len(speech_set)
    elif mechanism == "distractor_moves":
        overlap = len(active_sets["source2"] & speech_set)
        _require(
            overlap == len(active_sets["source2"]),
            "distractor motion left speech window",
        )
    else:
        both_active = active_sets["source1"] & active_sets["source2"]
        overlap = len(both_active & speech_set)
        _require(overlap > 0, "both_move has no simultaneous speech-window motion")
    return {
        "status": "PASS_CPU_MECHANISM_TIMING",
        "expected_moving_slots": list(MOVING_SLOTS[mechanism]),
        "observed_moving_slots": list(moving),
        "unchanged_speech_window_inclusive": list(speech_window),
        "mechanism_speech_overlap_frame_count": overlap,
        "both_moving_overlap_frame_count": (
            len(active_sets["source1"] & active_sets["source2"])
            if mechanism == "both_move"
            else 0
        ),
    }


def build_candidate(source_directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    materialization_path = source_directory / "materialization_receipt.json"
    suite_path = source_directory / "suite_execution_plan.json"
    materialization = _load(materialization_path)
    suite = _load(suite_path)
    scenario, plan = _scenario(suite)
    mechanism = str(materialization.get("mechanism"))
    _require(mechanism in CASE_ORDER, f"unsupported mechanism: {mechanism}")
    _require(
        materialization.get("episode_id") == scenario.get("scenario_id"),
        "legacy episode identity drift",
    )
    _require(
        materialization.get("frame_count") == FRAME_COUNT
        and materialization.get("frame_rate_hz") == FRAME_RATE_HZ,
        "legacy materialization is not 75 frames at 15 Hz",
    )
    audio_program = materialization.get("audio_program")
    _require(isinstance(audio_program, Mapping), "audio program is missing")
    validation = audio_program.get("validation")
    _require(
        isinstance(validation, Mapping)
        and validation.get("status") == "pass"
        and audio_program.get("target_source_slot") == "source1"
        and audio_program.get("distractor_source_slot") == "source2"
        and audio_program.get("target_event_count") == 1
        and audio_program.get("distractor_event_count") == 0,
        "controlled audio event contract drift",
    )
    speech_window = validation.get("speech_frame_window_inclusive")
    _require(
        isinstance(speech_window, list)
        and len(speech_window) == 2
        and all(isinstance(value, int) for value in speech_window),
        "speech frame window is invalid",
    )
    declarations = _declarations(plan)
    legacy_states_by_slot = {
        slot: _legacy_states(plan, f"{slot}_actor") for slot in ("source1", "source2")
    }
    root_application = materialization.get("suite_actor_root_application")
    _require(
        isinstance(root_application, Mapping), "legacy root application is missing"
    )
    timings = root_application.get("animation_timing")
    _require(isinstance(timings, Mapping), "legacy moving timing is missing")
    actors: dict[str, dict[str, Any]] = {}
    for slot in ("source1", "source2"):
        timing = timings.get(slot)
        if slot in MOVING_SLOTS[mechanism]:
            _require(isinstance(timing, Mapping), f"{slot} moving timing is missing")
            native = _native_window(timing=timing, states=legacy_states_by_slot[slot])
            actors[slot] = _moving_actor_plan(
                slot=slot,
                declaration=declarations[f"{slot}_actor"],
                legacy_states=legacy_states_by_slot[slot],
                native=native,
                speech_window=speech_window,
            )
        else:
            _require(timing is None, f"{slot} unexpectedly has moving timing")
            actors[slot] = _static_actor_plan(
                slot=slot,
                declaration=declarations[f"{slot}_actor"],
                legacy_states=legacy_states_by_slot[slot],
            )

    projection = _projection_preflight(
        plan=plan,
        actors=actors,
        declarations=declarations,
        legacy_states_by_slot=legacy_states_by_slot,
    )
    mechanism_preflight = _mechanism_preflight(mechanism, actors, speech_window)
    legacy_episode_id = str(materialization["episode_id"])
    candidate_episode_id = f"{legacy_episode_id}__native_rate_candidate_v1"
    frame_plan = _frame_plan(actors)
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": BLOCKED_STATUS,
        "cpu_candidate_preflight_status": "PASS_CPU_NATIVE_RATE_CANDIDATE",
        "release_qualified": False,
        "qualification_claim": False,
        "formal": False,
        "formal_episode_count": 0,
        "gpu_used": False,
        "gpu_launch_authorized": False,
        "candidate_episode_id": candidate_episode_id,
        "legacy_episode_id": legacy_episode_id,
        "mechanism": mechanism,
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FRAME_RATE_HZ,
        "episode_duration_seconds": EPISODE_DURATION_SECONDS,
        "last_frame_pts_seconds": (FRAME_COUNT - 1) / FRAME_RATE_HZ,
        "frame_coverage_end_seconds": FRAME_COUNT / FRAME_RATE_HZ,
        "timeline_ticks_per_second": TIMELINE_TICKS_PER_SECOND,
        "frame_ticks": FRAME_TICKS,
        "target_slot": "source1",
        "distractor_slot": "source2",
        "target_side": projection["target_side"],
        "camera": copy.deepcopy(plan["camera"]),
        "actor_declarations": {
            actor_id: copy.deepcopy(declaration)
            for actor_id, declaration in declarations.items()
        },
        "actors": actors,
        "frames": frame_plan,
        "mechanism_preflight": mechanism_preflight,
        "projection_preflight": projection,
        "audio_event_contract": {
            "status": "PASS_UNCHANGED_SOUND_EVENT_PROGRAM",
            "sound_event_content_and_timing_modified": False,
            "source_activation_modified": False,
            "audio_program": copy.deepcopy(audio_program),
            "speech_frame_window_inclusive": list(speech_window),
            "target_speech_start_sample": audio_program.get(
                "target_speech_start_sample"
            ),
            "existing_exact_rir_reuse_authorized": False,
            "fresh_exact_rir_required": True,
            "future_emitter_trajectory_policy": (
                "mouth_anchor_tracks_candidate_root_after_live_ground_snap_v1"
            ),
            "claim_boundary": (
                "sound content, sample timing, and source activation are unchanged; "
                "spatial propagation must be rebuilt for the native-rate root timing"
            ),
        },
        "source_activation_contract": {
            "status": "PASS_UNCHANGED_SOURCE_ACTIVATION",
            "modified": False,
            "source_logic": copy.deepcopy(plan["source_logic"]),
        },
        "release_gate_statuses": {
            "cpu_native_rate_trajectory": "pass",
            "cpu_static_pinhole_projection": "pass",
            "sound_event_invariance": "pass",
            "fresh_full75_pixels": "pending",
            "live_ground_contact": "pending",
            "live_foot_plant_sync": "pending",
            "live_walking_asset_readback": "pending",
            "live_idle_walk_pose_transition": "pending",
            "fresh_exact_rir": "pending",
            "gpu_capture": "not_run",
        },
        "claim_boundary": (
            "CPU candidate only. It fixes the legacy time stretch but does not "
            "establish fresh pixel visibility, live ground contact, foot-plant sync, "
            "live animation readback, collision safety, or exact acoustic propagation."
        ),
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": BLOCKED_STATUS,
        "candidate_episode_id": candidate_episode_id,
        "legacy_episode_id": legacy_episode_id,
        "mechanism": mechanism,
        "candidate_preflight_file": f"{CASE_OUTPUT_STEMS[mechanism]}.json",
        "cpu_candidate_preflight_status": "PASS_CPU_NATIVE_RATE_CANDIDATE",
        "release_qualified": False,
        "qualification_claim": False,
        "formal_episode_count": 0,
        "gpu_used": False,
        "gpu_launch_authorized": False,
        "first_blocker": copy.deepcopy(RELEASE_BLOCKERS[0]),
        "release_blockers": copy.deepcopy(RELEASE_BLOCKERS),
        "release_gate_statuses": copy.deepcopy(preflight["release_gate_statuses"]),
        "trajectory_summary": {
            slot: copy.deepcopy(actors[slot]["trajectory_preflight"])
            for slot in ("source1", "source2")
        },
        "active_intervals": {
            slot: copy.deepcopy(actors[slot]["native_rate_active_interval"])
            for slot in MOVING_SLOTS[mechanism]
        },
        "mechanism_preflight": copy.deepcopy(mechanism_preflight),
        "projection_preflight": copy.deepcopy(projection),
        "audio_event_invariance_status": "pass_unchanged_content_and_sample_timing",
        "legacy_slow_motion_decision": (
            "reject_nonrelease_pipeline_evidence_only_not_upgraded"
        ),
        "claim_boundary": copy.deepcopy(preflight["claim_boundary"]),
        "input_artifacts": [
            "materialization_receipt.json",
            "suite_execution_plan.json",
        ],
    }
    return preflight, receipt


def build_all(source_root: Path) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for mechanism in CASE_ORDER:
        preflight, receipt = build_candidate(source_root / mechanism)
        _require(preflight["mechanism"] == mechanism, "source-root case ordering drift")
        result[mechanism] = (preflight, receipt)
    return result


def write_all(
    output_dir: Path,
    candidates: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[Path]:
    written: list[Path] = []
    for mechanism in CASE_ORDER:
        preflight, receipt = candidates[mechanism]
        stem = CASE_OUTPUT_STEMS[mechanism]
        preflight_path = output_dir / f"{stem}.json"
        receipt_path = output_dir / f"{stem}_receipt.json"
        _write_new(preflight_path, preflight)
        _write_new(receipt_path, receipt)
        written.extend((preflight_path, receipt_path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = build_all(args.source_root.resolve())
    written = write_all(args.output_dir.resolve(), candidates)
    print(
        "STRICT_TWO_HUMAN_NATIVE_RATE_DYNAMIC_CANDIDATES_OK "
        f"count={len(candidates)} files={len(written)} status={BLOCKED_STATUS} "
        f"output_dir={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
