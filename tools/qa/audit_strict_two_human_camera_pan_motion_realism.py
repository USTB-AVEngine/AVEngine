#!/usr/bin/env python3
"""Audit camera-pan motion realism against authoritative 15 Hz camera poses.

This is a CPU-only, fail-closed release audit.  Static camera poses from
different Episodes are inventory samples, not a trajectory and not a source of
angular speed.  A release candidate needs either a native per-frame trajectory
or a versioned, explicitly approved camera-motion profile, plus a 1:1 active
interval with HOLD outside it.  Geometry, pixel visibility, and exact runtime
readback remain useful pipeline evidence but cannot supply motion authority.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "avengine_native_strict_two_human_camera_pan_motion_realism_audit_v1"
EXPECTED_SOURCE_SCHEMA = "avengine_optional_spear_apartment_suite_v1"
EXPECTED_CANDIDATE_EPISODE = "strict2h_dynamic_canary_04_camera_pan_both_static_v2"
EXPECTED_FRAME_COUNT = 75
EXPECTED_FRAME_RATE_HZ = 15.0
NUMERIC_TOLERANCE = 1.0e-9

MOTION_AUTHORITY_KEYS = {
    "approved_camera_motion_profile_id",
    "camera_motion_profile_id",
    "global_time_stretch_applied",
    "native_camera_trajectory_id",
    "native_frame_rate_hz",
    "native_source_frame_indices_inclusive",
    "native_source_frame_range_inclusive",
    "native_rate_active_interval",
    "outside_active_interval_policy",
    "time_scale",
}
CAMERA_PATH_KEYS = (
    "habitat_yaw_path_deg",
    "yaw_path_deg",
    "ue_yaw_path_deg",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path | None, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(payload, end="")
        return
    if path.exists():
        raise RuntimeError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _wrapped_delta_deg(current: float, previous: float) -> float:
    return (current - previous + 180.0) % 360.0 - 180.0


def _linear_quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot compute a quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    alpha = index - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _distribution(values: Sequence[float], *, empty_reason: str) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "status": "undefined_empty",
            "sample_count": 0,
            "reason": empty_reason,
        }
    return {
        "status": "computed",
        "sample_count": len(finite),
        "minimum": min(finite),
        "p25": _linear_quantile(finite, 0.25),
        "median": _linear_quantile(finite, 0.5),
        "p75": _linear_quantile(finite, 0.75),
        "maximum": max(finite),
    }


def _frame_rate_hz(scenario: Mapping[str, Any]) -> float:
    render = scenario.get("render")
    if isinstance(render, Mapping) and "frame_rate_hz" in render:
        return float(render["frame_rate_hz"])
    plan = scenario.get("plan")
    if isinstance(plan, Mapping):
        plan_render = plan.get("render")
        if isinstance(plan_render, Mapping):
            numerator = float(plan_render.get("fps_num", 0.0))
            denominator = float(plan_render.get("fps_den", 0.0))
            if numerator > 0.0 and denominator > 0.0:
                return numerator / denominator
    raise RuntimeError(f"scenario frame rate is missing: {scenario.get('scenario_id')}")


def _mapping_yaw(value: object) -> float | None:
    if not isinstance(value, Mapping):
        return None
    for key in (
        "habitat_yaw_deg",
        "yaw_deg",
        "camera_yaw_deg",
        "listener_yaw_deg",
    ):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    rotation = value.get("rotation_deg")
    if (
        isinstance(rotation, Sequence)
        and not isinstance(rotation, (str, bytes))
        and len(rotation) == 3
        and isinstance(rotation[2], (int, float))
    ):
        return float(rotation[2])
    return None


def _extract_camera_yaw_path(plan: Mapping[str, Any]) -> tuple[list[float], str | None]:
    camera = plan.get("camera")
    if isinstance(camera, Mapping):
        for key in CAMERA_PATH_KEYS:
            path = camera.get(key)
            if (
                isinstance(path, Sequence)
                and not isinstance(path, (str, bytes))
                and len(path) >= 2
                and all(isinstance(value, (int, float)) for value in path)
            ):
                yaws = [float(value) for value in path]
                if key == "ue_yaw_path_deg":
                    # The sign does not matter to absolute-speed statistics, but
                    # retain a consistent Habitat-style orientation convention.
                    yaws = [-90.0 - value for value in yaws]
                return yaws, f"plan.camera.{key}"

    frames = plan.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        return [], None
    yaws: list[float] = []
    selected_field: str | None = None
    for frame in frames:
        if not isinstance(frame, Mapping):
            return [], None
        found: tuple[float, str] | None = None
        for field in (
            "camera",
            "listener",
            "view_pose",
            "camera_pose",
            "listener_pose",
        ):
            yaw = _mapping_yaw(frame.get(field))
            if yaw is not None:
                found = (yaw, f"plan.frames[].{field}")
                break
        if found is None:
            for field in (
                "habitat_camera_yaw_deg",
                "camera_yaw_deg",
                "listener_yaw_deg",
            ):
                yaw = frame.get(field)
                if isinstance(yaw, (int, float)):
                    found = (float(yaw), f"plan.frames[].{field}")
                    break
        if found is None:
            return [], None
        if selected_field is None:
            selected_field = found[1]
        elif selected_field != found[1]:
            raise RuntimeError(
                "camera yaw authority field changes within one trajectory"
            )
        yaws.append(found[0])
    return yaws, selected_field


def _positive_segments(
    yaws: Sequence[float], frame_rate_hz: float
) -> tuple[list[float], list[float]]:
    speeds: list[float] = []
    durations: list[float] = []
    active_interval_count = 0
    for previous, current in pairwise(yaws):
        speed = abs(_wrapped_delta_deg(float(current), float(previous))) * frame_rate_hz
        if speed > NUMERIC_TOLERANCE:
            speeds.append(speed)
            active_interval_count += 1
        elif active_interval_count:
            durations.append(active_interval_count / frame_rate_hz)
            active_interval_count = 0
    if active_interval_count:
        durations.append(active_interval_count / frame_rate_hz)
    return speeds, durations


def audit_inventory(suite: Mapping[str, Any]) -> dict[str, Any]:
    _require(suite.get("schema") == EXPECTED_SOURCE_SCHEMA, "source suite schema drift")
    scenarios = suite.get("scenarios")
    _require(isinstance(scenarios, list) and scenarios, "source suite has no scenarios")

    camera_keysets: Counter[tuple[str, ...]] = Counter()
    frame_counts: Counter[int] = Counter()
    positions: Counter[tuple[float, float, float]] = Counter()
    static_yaws: Counter[float] = Counter()
    frame_rates: Counter[float] = Counter()
    dynamic_fields: Counter[str] = Counter()
    dynamic_scenario_count = 0
    dynamic_frame_count = 0
    positive_speeds: list[float] = []
    positive_segment_durations: list[float] = []

    for scenario in scenarios:
        _require(isinstance(scenario, Mapping), "source scenario is not an object")
        plan = scenario.get("plan")
        _require(isinstance(plan, Mapping), "source scenario plan is missing")
        camera = plan.get("camera")
        _require(isinstance(camera, Mapping), "source scenario camera is missing")
        camera_keysets[tuple(sorted(str(key) for key in camera))] += 1
        position = camera.get("habitat_position_m")
        _require(
            isinstance(position, list)
            and len(position) == 3
            and all(isinstance(value, (int, float)) for value in position),
            "source static camera position is invalid",
        )
        positions[tuple(round(float(value), 9) for value in position)] += 1
        yaw = camera.get("habitat_yaw_deg")
        _require(isinstance(yaw, (int, float)), "source static camera yaw is invalid")
        static_yaws[round(float(yaw), 9)] += 1
        frames = plan.get("frames")
        _require(isinstance(frames, list), "source plan frames are missing")
        frame_counts[len(frames)] += 1
        frame_rate = _frame_rate_hz(scenario)
        frame_rates[frame_rate] += 1
        yaw_path, authority_field = _extract_camera_yaw_path(plan)
        if yaw_path:
            _require(
                len(yaw_path) == len(frames),
                "source camera yaw path does not align with plan frames",
            )
            dynamic_scenario_count += 1
            dynamic_frame_count += len(yaw_path)
            assert authority_field is not None
            dynamic_fields[authority_field] += 1
            speeds, durations = _positive_segments(yaw_path, frame_rate)
            positive_speeds.extend(speeds)
            positive_segment_durations.extend(durations)

    unique_position_items = sorted([[*key], count] for key, count in positions.items())
    unique_yaw_items = sorted([key, count] for key, count in static_yaws.items())
    return {
        "status": "pass_inventory_readback",
        "scenario_count": len(scenarios),
        "total_plan_frame_count": sum(
            count * occurrences for count, occurrences in frame_counts.items()
        ),
        "plan_frame_count_distribution": [
            {"frame_count": key, "scenario_count": value}
            for key, value in sorted(frame_counts.items())
        ],
        "frame_rate_hz_distribution": [
            {"frame_rate_hz": key, "scenario_count": value}
            for key, value in sorted(frame_rates.items())
        ],
        "camera_keyset_distribution": [
            {"fields": list(key), "scenario_count": value}
            for key, value in sorted(camera_keysets.items())
        ],
        "static_camera_inventory": {
            "episode_count": len(scenarios),
            "unique_position_count": len(positions),
            "unique_habitat_yaw_count": len(static_yaws),
            "positions": unique_position_items
            if len(unique_position_items) <= 8
            else [],
            "habitat_yaws_deg": unique_yaw_items if len(unique_yaw_items) <= 16 else [],
            "cross_episode_pose_differences_are_motion_samples": False,
        },
        "dynamic_camera_inventory": {
            "scenario_count": dynamic_scenario_count,
            "frame_count": dynamic_frame_count,
            "authority_field_distribution": [
                {"field": key, "scenario_count": value}
                for key, value in sorted(dynamic_fields.items())
            ],
            "positive_angular_speed_sample_count": len(positive_speeds),
            "continuous_pan_segment_count": len(positive_segment_durations),
            "absolute_angular_speed_deg_s_distribution": _distribution(
                positive_speeds,
                empty_reason=(
                    "no authoritative per-frame camera trajectory contains a positive yaw step"
                ),
            ),
            "continuous_pan_duration_s_distribution": _distribution(
                positive_segment_durations,
                empty_reason=("no authoritative positive-yaw camera segment exists"),
            ),
        },
    }


def _candidate(preflight: Mapping[str, Any], frame_rate_hz: float) -> dict[str, Any]:
    canaries = preflight.get("canaries")
    _require(isinstance(canaries, list) and len(canaries) == 1, "expected one canary")
    row = canaries[0]
    _require(isinstance(row, Mapping), "candidate row is invalid")
    _require(row.get("episode_id") == EXPECTED_CANDIDATE_EPISODE, "episode drift")
    camera = row.get("camera")
    _require(isinstance(camera, Mapping), "candidate camera is missing")
    yaws = camera.get("yaw_path_deg")
    _require(
        isinstance(yaws, list)
        and len(yaws) == EXPECTED_FRAME_COUNT
        and all(isinstance(value, (int, float)) for value in yaws),
        "candidate yaw path is not exact full75",
    )
    yaw_values = [float(value) for value in yaws]
    deltas = [
        _wrapped_delta_deg(current, previous)
        for previous, current in pairwise(yaw_values)
    ]
    yaw_span = max(yaw_values) - min(yaw_values)
    nominal_clip_duration = len(yaw_values) / frame_rate_hz
    sampled_interval_duration = (len(yaw_values) - 1) / frame_rate_hz
    provenance = camera.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    recognized_authority_fields = sorted(MOTION_AUTHORITY_KEYS.intersection(provenance))
    return {
        "episode_id": row["episode_id"],
        "candidate_revision": row.get("candidate_revision"),
        "mechanism": row.get("mechanism"),
        "target_side": row.get("target_side"),
        "frame_count": len(yaw_values),
        "frame_rate_hz": frame_rate_hz,
        "yaw_start_deg": yaw_values[0],
        "yaw_end_deg": yaw_values[-1],
        "yaw_span_deg": yaw_span,
        "unique_yaw_count": len({round(value, 12) for value in yaw_values}),
        "nonzero_interframe_step_count": sum(
            abs(value) > NUMERIC_TOLERANCE for value in deltas
        ),
        "minimum_signed_interframe_step_deg": min(deltas),
        "maximum_signed_interframe_step_deg": max(deltas),
        "uniform_interframe_step": max(deltas) - min(deltas) <= NUMERIC_TOLERANCE,
        "nominal_clip_duration_s": nominal_clip_duration,
        "sampled_interval_duration_s": sampled_interval_duration,
        "nominal_clip_angular_velocity_deg_s": yaw_span / nominal_clip_duration,
        "interframe_slope_angular_velocity_deg_s": yaw_span / sampled_interval_duration,
        "full_clip_linear_interpolation": True,
        "declared_active_interval": None,
        "outside_active_interval_hold_frame_count": 0,
        "camera_provenance": dict(provenance),
        "recognized_motion_authority_fields": recognized_authority_fields,
        "native_or_approved_motion_authority_present": bool(
            recognized_authority_fields
        ),
    }


def _machine_evidence(
    finalization: Mapping[str, Any], visual_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    capture = finalization.get("capture")
    _require(isinstance(capture, Mapping), "strict finalization capture is missing")
    runtime = capture.get("runtime")
    visibility = capture.get("visibility_gate")
    _require(isinstance(runtime, Mapping), "runtime evidence is missing")
    _require(isinstance(visibility, Mapping), "visibility evidence is missing")
    transforms = runtime.get("transform_readbacks")
    _require(isinstance(transforms, Mapping), "camera transform evidence is missing")
    target = visibility.get("target_speech")
    distractor = visibility.get("distractor_all_frames")
    _require(isinstance(target, Mapping), "target visibility evidence is missing")
    _require(
        isinstance(distractor, Mapping), "distractor visibility evidence is missing"
    )
    findings = visual_receipt.get("review", {}).get("findings", {})
    _require(isinstance(findings, Mapping), "visual findings are missing")
    return {
        "preserved_not_recomputed": True,
        "strict_finalization_status": finalization.get("status"),
        "dynamic_full75_canary_pass": finalization.get("dynamic_full75_canary_pass"),
        "captured_frame_count": capture.get("captured_frame_count"),
        "runtime_status": runtime.get("status"),
        "camera_transform_readback_status": transforms.get("status"),
        "camera_readback_count": transforms.get("camera_readback_count"),
        "normal_camera_yaw_span_deg": transforms.get("normal_camera_yaw_span_deg"),
        "normal_distinct_camera_yaw_count": transforms.get(
            "normal_distinct_camera_yaw_count"
        ),
        "maximum_camera_location_error_cm": transforms.get(
            "maximum_camera_location_error_cm"
        ),
        "maximum_camera_rotation_error_deg": transforms.get(
            "maximum_camera_rotation_error_deg"
        ),
        "visibility_status": visibility.get("status"),
        "target_speech_minimum_visible_pixels": target.get("minimum_visible_pixels"),
        "target_speech_minimum_visible_fraction": target.get(
            "minimum_visible_fraction"
        ),
        "distractor_minimum_visible_pixels": distractor.get("minimum_visible_pixels"),
        "distractor_minimum_visible_fraction": distractor.get(
            "minimum_visible_fraction"
        ),
        "visual_camera_pan_finding": findings.get("camera_pan"),
        "identity_continuity_finding": findings.get("identity_continuity"),
        "visible_floor_clearance_gap_beneath_both_characters": findings.get(
            "visible_floor_clearance_gap_beneath_both_characters"
        ),
        "evidence_classification": "nonformal_pipeline_canary_evidence_only",
    }


def build_receipt(
    *,
    suite: Mapping[str, Any],
    source_suite_path: str,
    preflight: Mapping[str, Any],
    finalization: Mapping[str, Any],
    visual_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = audit_inventory(suite)
    rates = inventory["frame_rate_hz_distribution"]
    _require(len(rates) == 1, "source suite frame rate is not unique")
    frame_rate_hz = float(rates[0]["frame_rate_hz"])
    candidate = _candidate(preflight, frame_rate_hz)
    machine = _machine_evidence(finalization, visual_receipt)
    dynamic = inventory["dynamic_camera_inventory"]
    native_motion_missing = (
        dynamic["scenario_count"] == 0
        and dynamic["positive_angular_speed_sample_count"] == 0
    )
    motion_profile_missing = not candidate[
        "native_or_approved_motion_authority_present"
    ]
    _require(
        native_motion_missing and motion_profile_missing,
        "this receipt is the fail-closed missing-authority path only",
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "reject_release_missing_camera_motion_authority",
        "candidate_decision": "REJECT_RELEASE_KEEP_PIPELINE_CANARY",
        "release_qualified": False,
        "release_classification": "nonrelease_pipeline_evidence_only",
        "first_blocker": {
            "code": "missing_native_or_approved_camera_motion_profile",
            "message": (
                "the audited inventory has no per-frame camera motion and the candidate names no approved motion profile"
            ),
        },
        "source_authority": {
            "suite_path": source_suite_path,
            "suite_schema": suite.get("schema"),
            "scope": "all scenarios and all plan frames",
            "inventory": inventory,
        },
        "current_candidate": candidate,
        "time_stretch_assessment": {
            "status": "not_computable_no_native_camera_motion_source",
            "is_proven_resampling_of_a_native_pan": False,
            "reason": (
                "the candidate is explicit full75 linear interpolation, but no native pan path, source frame range, native duration, or approved speed profile exists to define a time-scale ratio"
            ),
            "release_implication": (
                "absence of a computable stretch factor does not pass realism; missing motion authority is itself release-blocking"
            ),
        },
        "native_rate_candidate_search": {
            "status": "no_candidate_under_audited_authority",
            "candidate_count": 0,
            "candidates": [],
            "reason": (
                "all 1000 inventory Episodes use one static pose; static poses may not be ordered into a trajectory or used to invent angular speed"
            ),
        },
        "required_replacement_contract": {
            "motion_authority": (
                "native 15 Hz per-frame camera trajectory or versioned owner-approved camera motion profile"
            ),
            "active_interval_mapping": "1:1 source samples at 15 Hz; time_scale=1.0",
            "outside_active_interval_policy": "HOLD first/last authorized yaw",
            "global_time_stretch_allowed": False,
            "cross_episode_static_pose_interpolation_allowed": False,
            "geometry_constraints": [
                "speaking target remains right of the midline dead zone",
                "silent distractor remains left of the midline dead zone",
                "both 2 m actor envelopes stay inside frame with depth clearance",
            ],
            "runtime_reverification": [
                "fresh all75 normal and target-only pixels",
                "all75 camera transform readback",
                "all75 target/distractor visibility and metric-depth gates",
                "independent ground-contact release receipt",
            ],
        },
        "preserved_machine_evidence": machine,
        "independent_release_gates": {
            "motion_realism": "reject_missing_authority",
            "ground_contact": "blocked_unqualified_visible_floor_clearance_gap",
            "fresh_pixels": "existing_candidate_machine_pass_preserved_not_reused_for_replacement",
        },
        "audit_execution": {
            "cpu_only": True,
            "gpu_used": False,
            "files_mutated_in_source_repo": False,
            "other_gate_results_recomputed": False,
        },
        "formal": False,
        "formal_episode_count": 0,
        "qualification_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--candidate-preflight", type=Path, required=True)
    parser.add_argument("--strict-finalization", type=Path, required=True)
    parser.add_argument("--visual-receipt", type=Path, required=True)
    parser.add_argument("--output", type=str, default="-")
    args = parser.parse_args()
    source_suite = args.source_suite.resolve()
    receipt = build_receipt(
        suite=_load(source_suite),
        source_suite_path=str(source_suite),
        preflight=_load(args.candidate_preflight.resolve()),
        finalization=_load(args.strict_finalization.resolve()),
        visual_receipt=_load(args.visual_receipt.resolve()),
    )
    output = None if args.output == "-" else Path(args.output).resolve()
    _write(output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
