#!/usr/bin/env python3
"""Build a CPU-only, fail-closed motion-realism release receipt.

The gate intentionally treats the existing continuous full75 dynamic canaries as
pipeline evidence only.  A short retained native motion window may not be spread
over the complete 75-frame episode.  Release requires an explicit native-rate
active interval, Idle outside that interval, canonical Walking cadence, and
per-active-frame foot-plant evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

FRAME_COUNT = 75
FRAME_RATE_HZ = 15.0
TIMELINE_TICKS_PER_SECOND = 48_000
ANIMATION_TICKS_PER_PHASE_CYCLE = 51_200

RECEIPT_SCHEMA = "avengine_strict_two_human_motion_realism_receipt_v1"
PROFILE_SCHEMA = "avengine_strict_two_human_motion_realism_profile_v1"
FOOT_PLANT_SCHEMA = "avengine_strict_two_human_foot_plant_sync_v1"

MAX_NATIVE_SPEED_RELATIVE_ERROR = 0.05
MAX_CANONICAL_CADENCE_RELATIVE_ERROR = 0.02
MAX_TIME_SCALE_ABS_ERROR = 1.0e-6
MAX_PLANTED_FOOT_SLIP_M_PER_FRAME = 0.02
NUMERIC_TOLERANCE = 1.0e-6

REQUIRED_FOOT_BONES = [
    "Bip01 L Foot",
    "Bip01 L Toe0",
    "Bip01 R Foot",
    "Bip01 R Toe0",
]


def _pairwise(values: Sequence[Any]):
    return pairwise(values)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _close(left: float, right: float, tolerance: float = NUMERIC_TOLERANCE) -> bool:
    return (
        math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance
    )


def _horizontal_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(
        float(right[0]) - float(left[0]), float(right[2]) - float(left[2])
    )


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(_horizontal_distance(left, right) for left, right in _pairwise(points))


def _unwrap_phase_path(phases: Sequence[float]) -> list[float]:
    _require(bool(phases), "phase path is empty")
    wrapped = [float(value) for value in phases]
    _require(all(0.0 <= value < 1.0 for value in wrapped), "phase is outside [0,1)")
    unwrapped = [wrapped[0]]
    for previous, current in _pairwise(wrapped):
        advance = (current - previous) % 1.0
        _require(0.0 < advance < 0.5, "active Walking phase is not continuous")
        unwrapped.append(unwrapped[-1] + advance)
    return unwrapped


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
        "suite frame closure is not exact full75",
    )
    return scenario, plan


def _actor_declarations(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    actors = plan.get("actors")
    _require(isinstance(actors, list), "actor declarations are missing")
    result = {
        str(actor.get("actor_id")): actor
        for actor in actors
        if isinstance(actor, Mapping) and isinstance(actor.get("actor_id"), str)
    }
    _require(len(result) == len(actors), "actor declarations are not uniquely keyed")
    return result


def _actor_states(plan: Mapping[str, Any], actor_id: str) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for frame in plan["frames"]:
        matches = [
            state
            for state in frame.get("actor_states", [])
            if isinstance(state, Mapping) and state.get("actor_id") == actor_id
        ]
        _require(len(matches) == 1, f"{actor_id} must resolve once in every frame")
        result.append(matches[0])
    return result


def _block(code: str, message: str, **evidence: object) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if evidence:
        result["evidence"] = evidence
    return result


def _legacy_facts(
    *, timing: Mapping[str, Any], states: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    provenance = timing.get("path_provenance")
    _require(isinstance(provenance, Mapping), "moving timing lacks path provenance")
    native_range = provenance.get("native_source_frame_indices_inclusive")
    _require(
        isinstance(native_range, list)
        and len(native_range) == 2
        and all(isinstance(value, int) for value in native_range),
        "native source frame range is missing",
    )
    native_start, native_end = (int(native_range[0]), int(native_range[1]))
    _require(native_end > native_start, "native source frame range has no interval")
    native_interval_count = native_end - native_start
    native_sample_count = native_interval_count + 1
    native_anchor_count = int(provenance.get("native_anchor_count", -1))
    path_length_m = float(timing.get("path_length_m", math.nan))
    output_span_seconds = float(timing.get("episode_span_seconds", math.nan))
    phase_cycles = float(timing.get("phase_cycle_count", math.nan))
    _require(
        path_length_m > 0.0 and output_span_seconds > 0.0 and phase_cycles > 0.0,
        "moving timing metrics are missing",
    )
    native_duration_seconds = native_interval_count / FRAME_RATE_HZ
    native_average_speed = path_length_m / native_duration_seconds
    observed_average_speed = float(
        timing.get(
            "average_root_speed_m_per_second", path_length_m / output_span_seconds
        )
    )
    native_phase_cadence = phase_cycles / native_duration_seconds
    observed_phase_cadence = phase_cycles / output_span_seconds
    roots = [state.get("translation_m") for state in states]
    _require(
        all(isinstance(root, list) and len(root) == 3 for root in roots),
        "actor root path is missing from suite states",
    )
    walk_frame_count = sum(state.get("action_id") == "walk" for state in states)
    unique_roots = len({tuple(float(value) for value in root) for root in roots})
    return {
        "native_source_frame_range_inclusive": [native_start, native_end],
        "native_interval_count": native_interval_count,
        "native_sample_count": native_sample_count,
        "native_anchor_count": native_anchor_count,
        "native_anchor_count_matches_range": native_anchor_count == native_sample_count,
        "path_length_m": path_length_m,
        "native_duration_seconds": native_duration_seconds,
        "output_span_seconds": output_span_seconds,
        "global_time_stretch_factor": output_span_seconds / native_duration_seconds,
        "native_rate_average_root_speed_m_s": native_average_speed,
        "observed_full75_average_root_speed_m_s": observed_average_speed,
        "speed_ratio_observed_to_native": observed_average_speed / native_average_speed,
        "native_phase_advance_cycles": phase_cycles,
        "native_rate_phase_cycles_per_second": native_phase_cadence,
        "observed_full75_phase_cycles_per_second": observed_phase_cadence,
        "cadence_ratio_observed_to_native": observed_phase_cadence
        / native_phase_cadence,
        "suite_walk_frame_count": walk_frame_count,
        "suite_unique_root_count": unique_roots,
        "suite_full_path_length_m": _path_length(roots),
    }


def _legacy_blockers(facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if facts.get("native_anchor_count_matches_range") is not True:
        blockers.append(
            _block(
                "native_anchor_range_mismatch",
                "native anchor count does not close the declared source interval",
            )
        )
    if (
        float(facts["global_time_stretch_factor"])
        > 1.0 + MAX_NATIVE_SPEED_RELATIVE_ERROR
        and int(facts["suite_walk_frame_count"]) == FRAME_COUNT
    ):
        blockers.append(
            _block(
                "global_time_stretch_of_short_native_anchor_window",
                "short native anchor window was spread across all 75 Walking frames",
                stretch_factor=facts["global_time_stretch_factor"],
            )
        )
    if (
        abs(float(facts["speed_ratio_observed_to_native"]) - 1.0)
        > MAX_NATIVE_SPEED_RELATIVE_ERROR
    ):
        blockers.append(
            _block(
                "output_speed_outside_native_rate_envelope",
                "full75 root speed is outside the retained native-window envelope",
                observed_m_s=facts["observed_full75_average_root_speed_m_s"],
                native_rate_m_s=facts["native_rate_average_root_speed_m_s"],
                maximum_relative_error=MAX_NATIVE_SPEED_RELATIVE_ERROR,
            )
        )
    if (
        abs(float(facts["cadence_ratio_observed_to_native"]) - 1.0)
        > MAX_CANONICAL_CADENCE_RELATIVE_ERROR
    ):
        blockers.append(
            _block(
                "output_cadence_outside_native_rate_envelope",
                "Walking phase cadence was slowed with the root trajectory",
                observed_cycles_per_second=facts[
                    "observed_full75_phase_cycles_per_second"
                ],
                native_rate_cycles_per_second=facts[
                    "native_rate_phase_cycles_per_second"
                ],
                maximum_relative_error=MAX_CANONICAL_CADENCE_RELATIVE_ERROR,
            )
        )
    return blockers


def _profile_blockers(
    *,
    profile: object,
    slot: str,
    actor_id: str,
    declaration: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(profile, Mapping):
        blockers = [
            _block(
                "missing_motion_realism_profile",
                f"{slot} has no release-qualified motion-realism profile",
            )
        ]
        blockers.extend(_legacy_blockers(facts))
        blockers.append(
            _block(
                "missing_live_foot_plant_sync_evidence",
                "per-active-frame live foot/toe floor traces are absent",
            )
        )
        return blockers, {"profile_present": False}

    blockers: list[dict[str, Any]] = []
    if not (
        profile.get("schema") == PROFILE_SCHEMA
        and profile.get("status") == "pass"
        and profile.get("release_qualified") is True
        and profile.get("slot_id") == slot
        and profile.get("actor_id") == actor_id
    ):
        blockers.append(
            _block(
                "invalid_motion_realism_profile",
                f"{slot} motion-realism profile identity/status/schema is invalid",
            )
        )

    interval = profile.get("native_rate_active_interval")
    if not isinstance(interval, Mapping):
        blockers.append(
            _block("missing_native_rate_active_interval", "active interval is missing")
        )
        return blockers, {"profile_present": True}
    output_range = interval.get("output_frame_range_inclusive")
    native_range = interval.get("native_source_frame_range_inclusive")
    if not (
        isinstance(output_range, list)
        and len(output_range) == 2
        and all(isinstance(value, int) for value in output_range)
    ):
        blockers.append(
            _block("invalid_output_active_interval", "output active range is invalid")
        )
        return blockers, {"profile_present": True}
    start, end = int(output_range[0]), int(output_range[1])
    valid_bounds = 0 <= start < end < FRAME_COUNT
    if not valid_bounds:
        blockers.append(
            _block(
                "invalid_output_active_interval",
                "output active range is outside full75",
            )
        )
        return blockers, {"profile_present": True}
    expected_native_range = list(facts["native_source_frame_range_inclusive"])
    output_interval_count = end - start
    native_interval_count = int(facts["native_interval_count"])
    interval_contract_pass = (
        native_range == expected_native_range
        and int(interval.get("native_frame_rate_hz", -1)) == int(FRAME_RATE_HZ)
        and int(interval.get("output_frame_rate_hz", -1)) == int(FRAME_RATE_HZ)
        and int(interval.get("output_interval_count", -1)) == output_interval_count
        and output_interval_count == native_interval_count
        and interval.get("global_time_stretch_applied") is False
        and _close(
            float(interval.get("time_scale", math.nan)), 1.0, MAX_TIME_SCALE_ABS_ERROR
        )
        and interval.get("outside_action_id") == "idle"
        and interval.get("outside_root_policy") == "hold_boundary_root"
    )
    if not interval_contract_pass:
        blockers.append(
            _block(
                "native_rate_active_interval_contract_failed",
                "active interval is not a 1:1 native-rate interval or permits global stretch",
            )
        )

    walking_path = declaration.get("walking_animation")
    active_states = list(states[start : end + 1])
    outside_before = list(states[:start])
    outside_after = list(states[end + 1 :])
    if not all(
        state.get("action_id") == "walk" and state.get("ue_animation") == walking_path
        for state in active_states
    ):
        blockers.append(
            _block(
                "active_interval_not_canonical_walking",
                "every active frame must bind the declared Walking asset",
            )
        )
    idle_path = declaration.get("idle_animation")
    if not all(
        state.get("action_id") == "idle" and state.get("ue_animation") == idle_path
        for state in outside_before + outside_after
    ):
        blockers.append(
            _block(
                "non_idle_action_outside_active_interval",
                "frames outside the active interval must bind Idle",
            )
        )
    active_roots = [state["translation_m"] for state in active_states]
    first_root = tuple(float(value) for value in active_roots[0])
    last_root = tuple(float(value) for value in active_roots[-1])
    if not all(
        tuple(float(value) for value in state["translation_m"]) == first_root
        for state in outside_before
    ) or not all(
        tuple(float(value) for value in state["translation_m"]) == last_root
        for state in outside_after
    ):
        blockers.append(
            _block(
                "root_not_held_outside_active_interval",
                "inactive frames must hold the nearest active-interval boundary root",
            )
        )

    active_path_length = _path_length(active_roots)
    active_duration = output_interval_count / FRAME_RATE_HZ
    active_average_speed = active_path_length / active_duration
    native_speed = float(facts["native_rate_average_root_speed_m_s"])
    speed_relative_error = abs(active_average_speed / native_speed - 1.0)
    root_speed = profile.get("root_speed")
    root_speed_claim_pass = isinstance(root_speed, Mapping) and (
        root_speed.get("authority") == "retained_native_anchor_window_v1"
        and _close(
            float(root_speed.get("native_average_speed_m_s", math.nan)), native_speed
        )
        and _close(
            float(root_speed.get("output_active_average_speed_m_s", math.nan)),
            active_average_speed,
        )
        and float(root_speed.get("maximum_relative_error", math.inf))
        <= MAX_NATIVE_SPEED_RELATIVE_ERROR
        and speed_relative_error <= MAX_NATIVE_SPEED_RELATIVE_ERROR
    )
    if not root_speed_claim_pass:
        blockers.append(
            _block(
                "active_root_speed_outside_native_rate_envelope",
                "active root speed is not closed to the retained native anchor window",
                output_active_m_s=active_average_speed,
                native_rate_m_s=native_speed,
                relative_error=speed_relative_error,
            )
        )

    walking_clip = profile.get("walking_clip")
    phase_facts: dict[str, Any] = {}
    if not isinstance(walking_clip, Mapping):
        blockers.append(
            _block(
                "missing_canonical_walking_clip_contract",
                "Walking clip contract is missing",
            )
        )
    else:
        play_length = float(walking_clip.get("live_play_length_seconds", math.nan))
        play_rate = float(walking_clip.get("live_play_rate", math.nan))
        ticks_per_cycle = int(walking_clip.get("animation_ticks_per_phase_cycle", -1))
        ticks_per_second = int(walking_clip.get("timeline_ticks_per_second", -1))
        canonical_cadence = 1.0 / play_length if play_length > 0.0 else math.nan
        expected_tick_play_length = (
            ticks_per_cycle / ticks_per_second if ticks_per_second > 0 else math.nan
        )
        native_cadence = float(facts["native_rate_phase_cycles_per_second"])
        phases = [float(state.get("action_phase", math.nan)) for state in active_states]
        try:
            unwrapped = _unwrap_phase_path(phases)
            active_phase_cycles = unwrapped[-1] - unwrapped[0]
        except RuntimeError:
            active_phase_cycles = math.nan
        observed_cadence = active_phase_cycles / active_duration
        tick_values = [state.get("action_time_ticks") for state in active_states]
        expected_tick_delta = round(TIMELINE_TICKS_PER_SECOND / FRAME_RATE_HZ)
        tick_path_pass = all(isinstance(value, int) for value in tick_values) and all(
            int(current) - int(previous) == expected_tick_delta
            for previous, current in _pairwise(tick_values)
        )
        phase_tick_pass = all(isinstance(value, int) for value in tick_values) and all(
            abs(
                float(state["action_phase"])
                - (
                    (int(state["action_time_ticks"]) / ANIMATION_TICKS_PER_PHASE_CYCLE)
                    % 1.0
                )
            )
            <= NUMERIC_TOLERANCE
            for state in active_states
        )
        cadence_pass = (
            math.isfinite(canonical_cadence)
            and abs(canonical_cadence / native_cadence - 1.0)
            <= MAX_CANONICAL_CADENCE_RELATIVE_ERROR
            and abs(observed_cadence / canonical_cadence - 1.0)
            <= MAX_CANONICAL_CADENCE_RELATIVE_ERROR
        )
        clip_contract_pass = (
            walking_clip.get("asset_path") == walking_path
            and walking_clip.get("play_length_readback_source")
            == "live_uanimationasset_play_length_v1"
            and _close(play_rate, 1.0, MAX_TIME_SCALE_ABS_ERROR)
            and ticks_per_cycle == ANIMATION_TICKS_PER_PHASE_CYCLE
            and ticks_per_second == TIMELINE_TICKS_PER_SECOND
            and _close(play_length, expected_tick_play_length)
            and _close(
                float(walking_clip.get("canonical_cycles_per_second", math.nan)),
                canonical_cadence,
            )
            and cadence_pass
            and tick_path_pass
            and phase_tick_pass
        )
        phase_facts = {
            "live_play_length_seconds": play_length,
            "canonical_cycles_per_second": canonical_cadence,
            "observed_active_cycles_per_second": observed_cadence,
            "active_phase_advance_cycles": active_phase_cycles,
            "tick_path_pass": tick_path_pass,
            "phase_tick_pass": phase_tick_pass,
        }
        if not clip_contract_pass:
            blockers.append(
                _block(
                    "canonical_walking_clip_phase_contract_failed",
                    "active phase/ticks/cadence are not bound to live Walking at play rate 1",
                )
            )

    foot = profile.get("foot_plant_sync")
    foot_facts: dict[str, Any] = {}
    if not isinstance(foot, Mapping):
        blockers.append(
            _block(
                "missing_live_foot_plant_sync_evidence",
                "per-active-frame live foot/toe floor traces are absent",
            )
        )
    else:
        play_length = float(
            walking_clip.get("live_play_length_seconds", math.nan)
            if isinstance(walking_clip, Mapping)
            else math.nan
        )
        half_frame_phase = (
            0.5 / (FRAME_RATE_HZ * play_length) if play_length > 0.0 else math.nan
        )
        phase_error = float(foot.get("maximum_phase_error_cycles", math.inf))
        slip = float(foot.get("maximum_planted_foot_slip_m_per_frame", math.inf))
        foot_pass = (
            foot.get("schema") == FOOT_PLANT_SCHEMA
            and foot.get("status") == "pass"
            and foot.get("contact_phase_authority_status") == "pass"
            and isinstance(foot.get("contact_phase_authority"), str)
            and bool(foot.get("contact_phase_authority"))
            and "PENDING" not in str(foot.get("contact_phase_authority"))
            and foot.get("walking_asset_path") == walking_path
            and foot.get("runtime_evidence_kind")
            == "live_per_active_frame_foot_toe_floor_trace_v1"
            and foot.get("runtime_frame_indices") == list(range(start, end + 1))
            and foot.get("bones") == REQUIRED_FOOT_BONES
            and foot.get("ground_contact_release_gate_status") == "pass"
            and foot.get("all_samples_pass") is True
            and math.isfinite(half_frame_phase)
            and phase_error <= half_frame_phase + NUMERIC_TOLERANCE
            and slip <= MAX_PLANTED_FOOT_SLIP_M_PER_FRAME
        )
        foot_facts = {
            "maximum_phase_error_cycles": phase_error,
            "half_frame_phase_tolerance_cycles": half_frame_phase,
            "maximum_planted_foot_slip_m_per_frame": slip,
            "maximum_allowed_planted_foot_slip_m_per_frame": MAX_PLANTED_FOOT_SLIP_M_PER_FRAME,
        }
        if not foot_pass:
            blockers.append(
                _block(
                    "foot_plant_sync_contract_failed",
                    "canonical contact phases and live floor traces do not close",
                )
            )

    evaluated = {
        "profile_present": True,
        "output_frame_range_inclusive": [start, end],
        "output_interval_count": output_interval_count,
        "active_duration_seconds": active_duration,
        "active_path_length_m": active_path_length,
        "active_average_root_speed_m_s": active_average_speed,
        "native_speed_relative_error": speed_relative_error,
        "phase": phase_facts,
        "foot_plant": foot_facts,
    }
    return blockers, evaluated


def build_receipt(materialization_root: Path) -> dict[str, Any]:
    receipt_path = materialization_root / "materialization_receipt.json"
    suite_path = materialization_root / "suite_execution_plan.json"
    receipt = _load(receipt_path)
    suite = _load(suite_path)
    scenario, plan = _scenario(suite)
    _require(receipt.get("frame_count") == FRAME_COUNT, "materialization is not full75")
    _require(
        receipt.get("frame_rate_hz") == int(FRAME_RATE_HZ), "frame rate is not 15 Hz"
    )
    _require(
        receipt.get("episode_id") == scenario.get("scenario_id"),
        "episode identity drift",
    )
    root_application = receipt.get("suite_actor_root_application")
    _require(
        isinstance(root_application, Mapping), "root application receipt is missing"
    )
    timing_by_slot = root_application.get("animation_timing")
    _require(
        isinstance(timing_by_slot, Mapping) and timing_by_slot, "no moving timing slots"
    )
    profiles = root_application.get("motion_realism_profiles", {})
    _require(
        isinstance(profiles, Mapping), "motion-realism profile collection is invalid"
    )
    declarations = _actor_declarations(plan)

    slot_receipts: list[dict[str, Any]] = []
    for slot in ("source1", "source2"):
        timing = timing_by_slot.get(slot)
        if not isinstance(timing, Mapping):
            continue
        actor_id = f"{slot}_actor"
        _require(actor_id in declarations, f"missing declaration for {actor_id}")
        states = _actor_states(plan, actor_id)
        facts = _legacy_facts(timing=timing, states=states)
        blockers, evaluated = _profile_blockers(
            profile=profiles.get(slot),
            slot=slot,
            actor_id=actor_id,
            declaration=declarations[actor_id],
            states=states,
            facts=facts,
        )
        slot_receipts.append(
            {
                "slot_id": slot,
                "actor_id": actor_id,
                "status": "pass" if not blockers else "reject",
                "first_blocker": blockers[0] if blockers else None,
                "blockers": blockers,
                "native_rate_facts": facts,
                "profile_evaluation": evaluated,
            }
        )
    _require(slot_receipts, "no moving actor slots were evaluated")
    rejected = [item for item in slot_receipts if item["status"] == "reject"]
    first = None
    if rejected:
        first = {
            "slot_id": rejected[0]["slot_id"],
            **rejected[0]["first_blocker"],
        }
    return {
        "schema": RECEIPT_SCHEMA,
        "status": (
            "pass_motion_realism_release_gate"
            if not rejected
            else "reject_nonrelease_motion_realism_gate"
        ),
        "episode_id": receipt["episode_id"],
        "mechanism": receipt["mechanism"],
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": int(FRAME_RATE_HZ),
        "formal_episode_count": 0,
        "qualification_claim": False,
        "gpu_used": False,
        "scope": "cpu_only_motion_realism_audit",
        "release_classification": (
            "motion_realism_gate_pass_only_other_release_gates_still_required"
            if not rejected
            else "nonrelease_pipeline_evidence_only"
        ),
        "first_blocker": first,
        "threshold_contract": {
            "native_speed_authority": "retained native anchor path and exact 15 Hz source interval",
            "maximum_native_speed_relative_error": MAX_NATIVE_SPEED_RELATIVE_ERROR,
            "canonical_cadence_authority": "live Walking play length at play rate 1 plus exact animation ticks",
            "maximum_canonical_cadence_relative_error": MAX_CANONICAL_CADENCE_RELATIVE_ERROR,
            "phase_error_authority": "half of one 15 Hz output-frame phase advance",
            "maximum_planted_foot_slip_m_per_frame": MAX_PLANTED_FOOT_SLIP_M_PER_FRAME,
            "ground_clearance_authority": "separate accepted strict ground-contact release profile",
        },
        "moving_slots": slot_receipts,
        "other_gate_results_recomputed": False,
        "input_artifacts": [
            "materialization_receipt.json",
            "suite_execution_plan.json",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_receipt(args.materialization_root.resolve())
    _write(args.output.resolve(), result)
    print(
        "STRICT_TWO_HUMAN_MOTION_REALISM_RECEIPT_OK "
        f"status={result['status']} output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
