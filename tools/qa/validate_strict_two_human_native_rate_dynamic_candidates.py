#!/usr/bin/env python3
"""Validate a fail-closed native-rate full75 dynamic candidate pair."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

BUILDER_PATH = Path(__file__).with_name(
    "build_strict_two_human_native_rate_dynamic_candidates.py"
)
SPEC = importlib.util.spec_from_file_location(
    "native_rate_dynamic_builder", BUILDER_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import builder: {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)

REPLAY_FLOAT_ABSOLUTE_TOLERANCE = 1.0e-12
REPLAY_FLOAT_RELATIVE_TOLERANCE = 1.0e-12


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _close(left: float, right: float, tolerance: float = 1.0e-9) -> bool:
    return (
        math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance
    )


def _replay_equivalent(observed: object, expected: object) -> bool:
    """Compare JSON-shaped replay data without requiring identical IEEE tails."""

    if isinstance(observed, bool) or isinstance(expected, bool):
        return type(observed) is type(expected) and observed == expected
    if isinstance(observed, int) or isinstance(expected, int):
        return type(observed) is type(expected) and observed == expected
    if isinstance(observed, float) or isinstance(expected, float):
        return (
            isinstance(observed, float)
            and isinstance(expected, float)
            and math.isfinite(observed)
            and math.isfinite(expected)
            and math.isclose(
                observed,
                expected,
                rel_tol=REPLAY_FLOAT_RELATIVE_TOLERANCE,
                abs_tol=REPLAY_FLOAT_ABSOLUTE_TOLERANCE,
            )
        )
    if isinstance(observed, Mapping) or isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and isinstance(expected, Mapping)
            and set(observed) == set(expected)
            and all(
                _replay_equivalent(observed[key], expected[key]) for key in observed
            )
        )
    if isinstance(observed, list) or isinstance(expected, list):
        return (
            isinstance(observed, list)
            and isinstance(expected, list)
            and len(observed) == len(expected)
            and all(
                _replay_equivalent(observed_item, expected_item)
                for observed_item, expected_item in zip(observed, expected, strict=True)
            )
        )
    return type(observed) is type(expected) and observed == expected


def _horizontal_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(
        float(right[0]) - float(left[0]), float(right[2]) - float(left[2])
    )


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(
        _horizontal_distance(previous, current)
        for previous, current in pairwise(points)
    )


def _validate_release_boundary(
    preflight: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    _require(
        preflight.get("schema") == BUILDER.PREFLIGHT_SCHEMA, "preflight schema drift"
    )
    _require(receipt.get("schema") == BUILDER.RECEIPT_SCHEMA, "receipt schema drift")
    for document in (preflight, receipt):
        _require(
            document.get("status") == BUILDER.BLOCKED_STATUS, "release status drift"
        )
        _require(
            document.get("release_qualified") is False,
            "candidate became release-qualified",
        )
        _require(
            document.get("qualification_claim") is False,
            "candidate gained qualification claim",
        )
        _require(
            document.get("formal_episode_count") == 0, "formal denominator changed"
        )
        _require(document.get("gpu_used") is False, "CPU-only receipt claims GPU use")
        _require(
            document.get("gpu_launch_authorized") is False,
            "CPU-only receipt authorizes GPU",
        )
    _require(preflight.get("formal") is False, "candidate became formal")
    _require(
        receipt.get("first_blocker") == BUILDER.RELEASE_BLOCKERS[0]
        and receipt.get("release_blockers") == BUILDER.RELEASE_BLOCKERS,
        "fail-closed blocker closure drift",
    )
    expected_gates = {
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
    }
    _require(
        preflight.get("release_gate_statuses") == expected_gates
        and receipt.get("release_gate_statuses") == expected_gates,
        "release-gate status closure drift",
    )
    _require(
        receipt.get("legacy_slow_motion_decision")
        == "reject_nonrelease_pipeline_evidence_only_not_upgraded",
        "legacy slow-motion evidence was upgraded",
    )


def _validate_full75(preflight: Mapping[str, Any]) -> None:
    _require(
        preflight.get("frame_count") == BUILDER.FRAME_COUNT
        and preflight.get("frame_rate_hz") == BUILDER.FRAME_RATE_HZ
        and preflight.get("episode_duration_seconds")
        == BUILDER.EPISODE_DURATION_SECONDS
        and _close(
            float(preflight.get("last_frame_pts_seconds", math.nan)),
            (BUILDER.FRAME_COUNT - 1) / BUILDER.FRAME_RATE_HZ,
        )
        and _close(
            float(preflight.get("frame_coverage_end_seconds", math.nan)),
            BUILDER.FRAME_COUNT / BUILDER.FRAME_RATE_HZ,
        ),
        "full75/5-second closure failed",
    )
    frames = preflight.get("frames")
    actors = preflight.get("actors")
    _require(
        isinstance(frames, list) and len(frames) == BUILDER.FRAME_COUNT,
        "frame plan missing",
    )
    _require(isinstance(actors, Mapping), "actor array plan is missing")
    for expected_index, frame in enumerate(frames):
        _require(
            isinstance(frame, Mapping)
            and frame.get("frame_index") == expected_index
            and frame.get("pts_ticks") == expected_index * BUILDER.FRAME_TICKS
            and frame.get("frame_coverage_end_ticks")
            == (expected_index + 1) * BUILDER.FRAME_TICKS,
            f"frame timeline drift at f{expected_index}",
        )
        states = frame.get("actor_states")
        _require(
            isinstance(states, list)
            and [state.get("slot_id") for state in states] == ["source1", "source2"],
            f"actor-state closure drift at f{expected_index}",
        )
        for state in states:
            slot = str(state["slot_id"])
            actor = actors[slot]
            expected = {
                "actor_id": actor["actor_id"],
                "slot_id": slot,
                "translation_m": actor["root_path_m"][expected_index],
                "translation_ue_cm": actor["translation_ue_cm_path"][expected_index],
                "action_id": actor["action_id_path"][expected_index],
                "ue_animation": actor["ue_animation_path"][expected_index],
                "action_phase": actor["action_phase_path"][expected_index],
                "action_time_ticks": actor["action_time_ticks_path"][expected_index],
                "animation_timing_mode": actor["animation_timing_mode_path"][
                    expected_index
                ],
                "native_source_frame_index": actor["native_source_frame_index_path"][
                    expected_index
                ],
                "actor_yaw_ue_deg": actor["actor_yaw_ue_deg_path"][expected_index],
            }
            _require(
                dict(state) == expected,
                f"frame/actor path mismatch at f{expected_index} {slot}",
            )


def _validate_actor(
    actor: Mapping[str, Any], *, moving: bool, expected_slot: str
) -> None:
    _require(
        actor.get("slot_id") == expected_slot
        and actor.get("actor_id") == f"{expected_slot}_actor"
        and actor.get("moving") is moving,
        f"actor identity/motion drift for {expected_slot}",
    )
    path = actor.get("root_path_m")
    ue_path = actor.get("translation_ue_cm_path")
    actions = actor.get("action_id_path")
    animations = actor.get("ue_animation_path")
    phases = actor.get("action_phase_path")
    ticks = actor.get("action_time_ticks_path")
    modes = actor.get("animation_timing_mode_path")
    native_indices = actor.get("native_source_frame_index_path")
    _require(
        all(
            isinstance(values, list) and len(values) == BUILDER.FRAME_COUNT
            for values in (
                path,
                ue_path,
                actions,
                animations,
                phases,
                ticks,
                modes,
                native_indices,
            )
        ),
        f"full75 actor paths are incomplete for {expected_slot}",
    )
    _require(
        all(
            all(
                _close(float(observed), expected)
                for observed, expected in zip(
                    ue_point,
                    BUILDER._translation_ue_cm(root_point),
                )
            )
            for root_point, ue_point in zip(path, ue_path)
        ),
        f"Habitat/UE translation conversion drift for {expected_slot}",
    )
    if not moving:
        _require(
            actor.get("native_rate_active_interval") is None,
            "static actor gained interval",
        )
        _require(
            len({tuple(float(value) for value in point) for point in path}) == 1
            and set(actions) == {"idle"}
            and {float(value) for value in phases} == {0.0}
            and {int(value) for value in ticks} == {0},
            "static actor is not held Idle all75",
        )
        return

    interval = actor.get("native_rate_active_interval")
    authority = actor.get("native_motion_authority")
    trajectory = actor.get("trajectory_preflight")
    legacy = actor.get("legacy_slow_motion_evidence")
    _require(
        isinstance(interval, Mapping)
        and isinstance(authority, Mapping)
        and isinstance(trajectory, Mapping)
        and isinstance(legacy, Mapping),
        f"moving authority is incomplete for {expected_slot}",
    )
    output_range = interval.get("output_frame_range_inclusive")
    native_range = interval.get("native_source_frame_range_inclusive")
    _require(
        isinstance(output_range, list)
        and len(output_range) == 2
        and isinstance(native_range, list)
        and len(native_range) == 2,
        "active range shape drift",
    )
    start, end = (int(output_range[0]), int(output_range[1]))
    native_start, native_end = (int(native_range[0]), int(native_range[1]))
    output_intervals = end - start
    native_intervals = native_end - native_start
    _require(
        1 <= start < end <= BUILDER.FRAME_COUNT - 2
        and output_intervals == native_intervals
        and interval.get("output_interval_count") == output_intervals
        and interval.get("output_sample_count") == output_intervals + 1
        and interval.get("native_interval_count") == native_intervals
        and interval.get("native_sample_count") == native_intervals + 1
        and interval.get("native_frame_rate_hz") == BUILDER.FRAME_RATE_HZ
        and interval.get("output_frame_rate_hz") == BUILDER.FRAME_RATE_HZ
        and interval.get("time_scale") == 1.0
        and interval.get("global_time_stretch_applied") is False
        and interval.get("outside_action_id") == "idle"
        and interval.get("outside_root_policy") == "hold_nearest_boundary_root",
        "native-rate active interval contract failed",
    )
    _require(
        actions[:start] == ["idle"] * start
        and actions[start : end + 1] == ["walk"] * (end - start + 1)
        and actions[end + 1 :] == ["idle"] * (BUILDER.FRAME_COUNT - end - 1),
        "Walking is not isolated to the active interval",
    )
    first_root = tuple(float(value) for value in path[start])
    last_root = tuple(float(value) for value in path[end])
    _require(
        all(
            tuple(float(value) for value in point) == first_root
            for point in path[:start]
        )
        and all(
            tuple(float(value) for value in point) == last_root
            for point in path[end + 1 :]
        ),
        "Idle frames do not hold nearest active boundary roots",
    )
    _require(
        _horizontal_distance(path[start - 1], path[start]) <= BUILDER.NUMERIC_TOLERANCE
        and _horizontal_distance(path[end], path[end + 1]) <= BUILDER.NUMERIC_TOLERANCE,
        "action transition also contains a position jump",
    )
    active_path = path[start : end + 1]
    path_length = _path_length(active_path)
    duration = output_intervals / BUILDER.FRAME_RATE_HZ
    speed = path_length / duration
    native_speed = float(authority.get("native_average_speed_m_s", math.nan))
    _require(
        _close(path_length, float(authority.get("path_length_m", math.nan)), 1.0e-6)
        and _close(speed, native_speed, 1.0e-6)
        and BUILDER.EXPECTED_NATIVE_SPEED_MIN_M_S
        <= speed
        <= BUILDER.EXPECTED_NATIVE_SPEED_MAX_M_S
        and _close(
            speed,
            float(trajectory.get("active_average_speed_m_s", math.nan)),
            1.0e-6,
        ),
        "active root speed/path-length closure failed",
    )
    active_ticks = [int(value) for value in ticks[start : end + 1]]
    _require(
        all(
            current - previous == BUILDER.FRAME_TICKS
            for previous, current in pairwise(active_ticks)
        ),
        "active animation ticks are not native 15 Hz",
    )
    phase_advance = (
        active_ticks[-1] - active_ticks[0]
    ) / BUILDER.ANIMATION_TICKS_PER_PHASE_CYCLE
    cadence = phase_advance / duration
    _require(
        _close(cadence, BUILDER.EXPECTED_PHASE_CADENCE_HZ)
        and _close(
            cadence,
            float(trajectory.get("active_phase_cadence_hz", math.nan)),
        )
        and all(
            _close(
                float(phases[frame_index]),
                (int(ticks[frame_index]) / BUILDER.ANIMATION_TICKS_PER_PHASE_CYCLE)
                % 1.0,
            )
            for frame_index in range(start, end + 1)
        ),
        "active phase/tick cadence closure failed",
    )
    _require(
        {float(value) for value in phases[:start] + phases[end + 1 :]} == {0.0}
        and {int(value) for value in ticks[:start] + ticks[end + 1 :]} == {0},
        "Idle phase/ticks are not held",
    )
    _require(
        trajectory.get("foot_plant_sync_status") == "pending_live_runtime_evidence"
        and trajectory.get("ground_contact_status") == "pending_live_runtime_evidence"
        and trajectory.get("skeletal_pose_transition_continuity_status")
        == "pending_live_runtime_blend_readback"
        and legacy.get("decision") == "reject_nonrelease_pipeline_evidence_only"
        and legacy.get("upgraded_to_pass") is False
        and float(legacy.get("legacy_full75_average_speed_m_s", math.inf)) < 0.31,
        "pending live evidence or legacy rejection drift",
    )


def _validate_mechanism(preflight: Mapping[str, Any]) -> None:
    mechanism = preflight.get("mechanism")
    _require(mechanism in BUILDER.CASE_ORDER, "mechanism is invalid")
    actors = preflight.get("actors")
    _require(
        isinstance(actors, Mapping) and set(actors) == {"source1", "source2"},
        "actor plan is incomplete",
    )
    expected_moving = BUILDER.MOVING_SLOTS[mechanism]
    for slot in ("source1", "source2"):
        _validate_actor(
            actors[slot], moving=slot in expected_moving, expected_slot=slot
        )
    mechanism_preflight = preflight.get("mechanism_preflight")
    _require(
        isinstance(mechanism_preflight, Mapping)
        and mechanism_preflight.get("status") == "PASS_CPU_MECHANISM_TIMING"
        and tuple(mechanism_preflight.get("expected_moving_slots", []))
        == expected_moving
        and tuple(mechanism_preflight.get("observed_moving_slots", []))
        == expected_moving
        and int(mechanism_preflight.get("mechanism_speech_overlap_frame_count", 0)) > 0,
        "mechanism timing preflight drift",
    )
    if mechanism == "both_move":
        _require(
            int(mechanism_preflight.get("both_moving_overlap_frame_count", 0)) > 0,
            "both_move lost simultaneous motion",
        )


def _validate_projection(preflight: Mapping[str, Any]) -> None:
    projection = preflight.get("projection_preflight")
    _require(
        isinstance(projection, Mapping)
        and projection.get("status") == "PASS_CPU_STATIC_PINHOLE_ONLY"
        and projection.get("authority")
        == "analytic_static_camera_pinhole_projection_v1",
        "static projection preflight drift",
    )
    metrics = projection.get("actors")
    actors = preflight.get("actors")
    declarations = preflight.get("actor_declarations")
    camera = projection.get("camera")
    _require(isinstance(metrics, Mapping), "projected actor metrics missing")
    _require(
        isinstance(actors, Mapping)
        and isinstance(declarations, Mapping)
        and isinstance(camera, Mapping)
        and camera.get("dynamic") is False,
        "projection replay inputs are incomplete",
    )
    sides: list[str] = []
    for slot in ("source1", "source2"):
        item = metrics.get(slot)
        _require(isinstance(item, Mapping), f"projected metrics missing for {slot}")
        expected_side = item.get("expected_side_from_legacy_static_camera_projection")
        observed_side = item.get("observed_side")
        x_range = item.get("center_x_fraction_range")
        depth_range = item.get("camera_depth_m_range")
        _require(
            expected_side == observed_side
            and observed_side in {"left", "right"}
            and isinstance(x_range, list)
            and len(x_range) == 2
            and isinstance(depth_range, list)
            and len(depth_range) == 2
            and float(item.get("dead_zone_margin_fraction", -1.0))
            >= BUILDER.MINIMUM_DEAD_ZONE_MARGIN_FRACTION
            and BUILDER.MINIMUM_CAMERA_DEPTH_M
            <= float(depth_range[0])
            <= float(depth_range[1])
            <= BUILDER.MAXIMUM_CAMERA_DEPTH_M,
            f"side/depth projection failed for {slot}",
        )
        root_path = actors[slot].get("root_path_m")
        declaration = declarations.get(f"{slot}_actor")
        _require(
            isinstance(root_path, list)
            and len(root_path) == BUILDER.FRAME_COUNT
            and isinstance(declaration, Mapping),
            f"projection root/declaration missing for {slot}",
        )
        mouth_height = float(declaration["emitter_offset_m"][1])
        projected = [
            BUILDER._project(
                point=point,
                height_m=0.0,
                camera_position=camera["habitat_position_m"],
                camera_yaw_deg=float(camera["habitat_yaw_deg"]),
                horizontal_fov_deg=float(camera["horizontal_fov_deg"]),
            )
            for point in root_path
        ]
        projected_mouth_y = [
            BUILDER._project(
                point=point,
                height_m=mouth_height,
                camera_position=camera["habitat_position_m"],
                camera_yaw_deg=float(camera["habitat_yaw_deg"]),
                horizontal_fov_deg=float(camera["horizontal_fov_deg"]),
            )[2]
            for point in root_path
        ]
        projected_head_y = [
            BUILDER._project(
                point=point,
                height_m=2.0,
                camera_position=camera["habitat_position_m"],
                camera_yaw_deg=float(camera["habitat_yaw_deg"]),
                horizontal_fov_deg=float(camera["horizontal_fov_deg"]),
            )[2]
            for point in root_path
        ]
        computed_depth = [value[0] for value in projected]
        computed_x = [value[1] for value in projected]
        computed_root_y = [value[2] for value in projected]
        computed_ranges = {
            "center_x_fraction_range": [min(computed_x), max(computed_x)],
            "camera_depth_m_range": [min(computed_depth), max(computed_depth)],
            "root_y_fraction_range": [min(computed_root_y), max(computed_root_y)],
            "mouth_y_fraction_range": [
                min(projected_mouth_y),
                max(projected_mouth_y),
            ],
            "head_y_fraction_range": [min(projected_head_y), max(projected_head_y)],
        }
        for key, computed_range in computed_ranges.items():
            claimed_range = item.get(key)
            _require(
                isinstance(claimed_range, list)
                and len(claimed_range) == 2
                and all(
                    _close(float(claimed), computed)
                    for claimed, computed in zip(claimed_range, computed_range)
                ),
                f"computed projection range mismatch for {slot} {key}",
            )
        computed_side = BUILDER._side(computed_x)
        computed_margin = min(abs(value - 0.5) for value in computed_x)
        _require(
            computed_side == observed_side
            and _close(
                computed_margin,
                float(item.get("dead_zone_margin_fraction", math.nan)),
            ),
            f"computed side/dead-zone mismatch for {slot}",
        )
        sides.append(str(observed_side))
    _require(sides[0] != sides[1], "actors do not stay on opposite sides")
    computed_separation = min(
        _horizontal_distance(left, right)
        for left, right in zip(
            actors["source1"]["root_path_m"],
            actors["source2"]["root_path_m"],
        )
    )
    _require(
        preflight.get("target_side") == sides[0]
        and projection.get("target_side") == sides[0]
        and float(projection.get("minimum_actor_horizontal_separation_m", 0.0))
        >= BUILDER.MINIMUM_ACTOR_SEPARATION_M,
        "target side or actor separation drift",
    )
    _require(
        _close(
            computed_separation,
            float(projection["minimum_actor_horizontal_separation_m"]),
        ),
        "computed actor separation mismatch",
    )


def _validate_audio(
    preflight: Mapping[str, Any], *, source_directory: Path | None
) -> None:
    contract = preflight.get("audio_event_contract")
    _require(
        isinstance(contract, Mapping)
        and contract.get("status") == "PASS_UNCHANGED_SOUND_EVENT_PROGRAM"
        and contract.get("sound_event_content_and_timing_modified") is False
        and contract.get("source_activation_modified") is False
        and contract.get("existing_exact_rir_reuse_authorized") is False
        and contract.get("fresh_exact_rir_required") is True,
        "sound event invariance boundary drift",
    )
    audio = contract.get("audio_program")
    _require(
        isinstance(audio, Mapping)
        and audio.get("sample_rate_hz") == 16_000
        and audio.get("sample_count") == 80_000
        and audio.get("target_source_slot") == "source1"
        and audio.get("distractor_source_slot") == "source2"
        and audio.get("target_event_count") == 1
        and audio.get("distractor_event_count") == 0,
        "controlled sound event contract drift",
    )
    activation = preflight.get("source_activation_contract")
    _require(
        isinstance(activation, Mapping)
        and activation.get("status") == "PASS_UNCHANGED_SOURCE_ACTIVATION"
        and activation.get("modified") is False,
        "source activation contract drift",
    )
    source_logic = activation.get("source_logic")
    _require(isinstance(source_logic, Mapping), "source logic copy is missing")
    source_rows = source_logic.get("sources")
    _require(
        isinstance(source_rows, list)
        and [row.get("entity_actor_id") for row in source_rows]
        == ["source1_actor", "source2_actor"]
        and [row.get("activation") for row in source_rows] == ["active", "silent"],
        "target/distractor source activation drift",
    )
    if source_directory is not None:
        source = _load(source_directory / "materialization_receipt.json")
        source_suite = _load(source_directory / "suite_execution_plan.json")
        scenarios = source_suite.get("scenarios")
        _require(
            audio == source.get("audio_program"),
            "candidate altered the authoritative audio program",
        )
        _require(
            isinstance(scenarios, list)
            and len(scenarios) == 1
            and source_logic == scenarios[0]["plan"]["source_logic"],
            "candidate altered the authoritative source activation",
        )


def validate_pair(
    preflight: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    source_directory: Path | None = None,
    replay: bool = False,
) -> None:
    _validate_release_boundary(preflight, receipt)
    _require(
        preflight.get("candidate_episode_id") == receipt.get("candidate_episode_id")
        and preflight.get("legacy_episode_id") == receipt.get("legacy_episode_id")
        and preflight.get("mechanism") == receipt.get("mechanism"),
        "preflight/receipt identity drift",
    )
    _validate_full75(preflight)
    _validate_mechanism(preflight)
    _validate_projection(preflight)
    _validate_audio(preflight, source_directory=source_directory)
    _require(
        receipt.get("mechanism_preflight") == preflight.get("mechanism_preflight")
        and receipt.get("projection_preflight")
        == preflight.get("projection_preflight"),
        "receipt summary does not match candidate preflight",
    )
    actors = preflight["actors"]
    expected_trajectory_summary = {
        slot: actors[slot]["trajectory_preflight"] for slot in ("source1", "source2")
    }
    expected_intervals = {
        slot: actors[slot]["native_rate_active_interval"]
        for slot in BUILDER.MOVING_SLOTS[preflight["mechanism"]]
    }
    _require(
        receipt.get("trajectory_summary") == expected_trajectory_summary
        and receipt.get("active_intervals") == expected_intervals,
        "receipt trajectory summary does not match candidate preflight",
    )
    if replay:
        _require(source_directory is not None, "replay requires source directory")
        expected_preflight, expected_receipt = BUILDER.build_candidate(
            source_directory.resolve()
        )
        _require(
            _replay_equivalent(dict(preflight), expected_preflight),
            "preflight cross-runtime replay mismatch",
        )
        _require(
            _replay_equivalent(dict(receipt), expected_receipt),
            "receipt cross-runtime replay mismatch",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    source_directory = (
        args.source_directory.resolve() if args.source_directory is not None else None
    )
    validate_pair(
        _load(args.preflight.resolve()),
        _load(args.receipt.resolve()),
        source_directory=source_directory,
        replay=args.replay,
    )
    print(
        "STRICT_TWO_HUMAN_NATIVE_RATE_DYNAMIC_CANDIDATE_VALID "
        f"status={BUILDER.BLOCKED_STATUS} preflight={args.preflight.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
