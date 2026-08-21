#!/usr/bin/env python3
"""Fail-closed finalizer for strict two-human dynamic full75 canaries."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.qa.actor_motion_profile import (
    materialize_profile_frames,
    validate_actor_motion_profile,
)

FRAME_COUNT = 75
SAMPLE_COUNT = 80_000
SAMPLE_RATE_HZ = 16_000
TARGET_SPEECH_VISIBLE_PIXELS_MINIMUM = 10_000
TARGET_SPEECH_VISIBLE_FRACTION_MINIMUM = 0.8
DISTRACTOR_VISIBLE_PIXELS_MINIMUM = 5_000
DISTRACTOR_VISIBLE_FRACTION_MINIMUM = 0.5
LEGACY_CAMERA_PAN_ACOUSTICS = {
    "unique": 150,
    "source1": 75,
    "source2": 75,
    "reuse": 0,
}
LEGACY_CAMERA_PAN_MOTION = {
    "action_counts": {
        "source1": {"idle": 75, "walk": 0},
        "source2": {"idle": 75, "walk": 0},
    },
    "listener_orientation_count": 75,
}
LEGACY_CAMERA_PAN_MECHANISM = "camera_pan_both_static"
PROFILE_ACTOR_STATE_CORE_KEYS = (
    "actor_id",
    "slot_id",
    "translation_m",
    "translation_ue_cm",
    "action_id",
    "ue_animation",
    "action_phase",
    "action_time_ticks",
    "animation_timing_mode",
    "native_source_frame_index",
    "actor_yaw_ue_deg",
)
GROUND_CONTACT_READBACK_SCHEMA = "avengine_native_live_ground_contact_readback_v1"
GROUND_CONTACT_PROFILE_SCHEMA = (
    "avengine_strict_two_human_ground_contact_release_profile_v1"
)
GROUND_CONTACT_BONES = {
    "left": {"foot": "Bip01 L Foot", "toe": "Bip01 L Toe0"},
    "right": {"foot": "Bip01 R Foot", "toe": "Bip01 R Toe0"},
}


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _wav_float32(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    _require(path.is_file(), f"WAV missing: {path}")
    with path.open("rb") as stream:
        _require(stream.read(4) == b"RIFF", f"invalid RIFF header: {path}")
        stream.read(4)
        _require(stream.read(4) == b"WAVE", f"invalid WAVE header: {path}")
        fmt: tuple[int, int, int, int] | None = None
        payload: bytes | None = None
        while True:
            chunk_id = stream.read(4)
            if not chunk_id:
                break
            size_raw = stream.read(4)
            _require(len(size_raw) == 4, f"truncated WAV chunk: {path}")
            size = struct.unpack("<I", size_raw)[0]
            chunk = stream.read(size)
            _require(len(chunk) == size, f"truncated WAV payload: {path}")
            if size % 2:
                _require(len(stream.read(1)) == 1, f"truncated WAV padding: {path}")
            if chunk_id == b"fmt ":
                _require(size >= 16, f"short fmt chunk: {path}")
                tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", chunk[:16])
                fmt = (tag, channels, rate, bits)
            elif chunk_id == b"data":
                payload = chunk
    _require(fmt is not None and payload is not None, f"WAV chunks missing: {path}")
    tag, channels, rate, bits = fmt
    _require(
        tag == 3 and channels == 2 and rate == SAMPLE_RATE_HZ and bits == 32,
        f"dynamic binaural WAV must be IEEE float32 stereo 16 kHz: {fmt}",
    )
    samples = np.frombuffer(payload, dtype="<f4").reshape(-1, channels).copy()
    _require(
        samples.shape == (SAMPLE_COUNT, 2), f"WAV sample shape drift: {samples.shape}"
    )
    _require(np.isfinite(samples).all(), f"nonfinite WAV samples: {path}")
    return samples, {
        "format_tag": tag,
        "channel_count": channels,
        "sample_rate_hz": rate,
        "sample_count": int(samples.shape[0]),
        "sample_width_bytes": bits // 8,
        "peak_absolute": float(np.max(np.abs(samples))),
    }


def _scenario(materialization_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    suite = _load(materialization_root / "suite_execution_plan.json")
    scenarios = suite.get("scenarios")
    _require(
        isinstance(scenarios, list) and len(scenarios) == 1,
        "suite scenario closure failed",
    )
    scenario = scenarios[0]
    _require(isinstance(scenario, dict), "suite scenario is malformed")
    plan = scenario.get("plan")
    _require(isinstance(plan, dict), "suite plan is missing")
    return scenario, plan


def _normalize_profile_episode_id(
    value: Any, *, output_episode_id: str, profile_episode_id: str
) -> Any:
    if isinstance(value, str):
        return value.replace(output_episode_id, profile_episode_id)
    if isinstance(value, list):
        return [
            _normalize_profile_episode_id(
                item,
                output_episode_id=output_episode_id,
                profile_episode_id=profile_episode_id,
            )
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            _normalize_profile_episode_id(
                key,
                output_episode_id=output_episode_id,
                profile_episode_id=profile_episode_id,
            ): _normalize_profile_episode_id(
                item,
                output_episode_id=output_episode_id,
                profile_episode_id=profile_episode_id,
            )
            for key, item in value.items()
        }
    return value


def _profile_motion_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    candidate = profile["authorities"]["candidate"]["value"]
    actors = candidate["actors"]
    declarations = candidate["actor_declarations"]
    profile_frames = materialize_profile_frames(profile)
    action_counts: dict[str, dict[str, int]] = {}
    mode_counts: dict[str, dict[str, int]] = {}
    transitions: dict[str, list[dict[str, Any]]] = {}
    motion_semantics: dict[str, dict[str, Any]] = {}
    for slot, actor in actors.items():
        actor_id = actor["actor_id"]
        states = [
            next(
                state
                for state in frame["actor_states"]
                if state["actor_id"] == actor_id and state["slot_id"] == slot
            )
            for frame in profile_frames
        ]
        declared_actions = declarations[actor_id]["animation_paths_by_action_id"]
        action_counts[str(slot)] = {
            str(action): sum(state["action_id"] == action for state in states)
            for action in declared_actions
        }
        observed_modes = sorted(
            {str(state["animation_timing_mode"]) for state in states}
        )
        mode_counts[str(slot)] = {
            mode: sum(state["animation_timing_mode"] == mode for state in states)
            for mode in observed_modes
        }
        transitions[str(slot)] = [
            {
                "frame_index": frame_index,
                "from_action_id": states[frame_index - 1]["action_id"],
                "to_action_id": states[frame_index]["action_id"],
            }
            for frame_index in range(1, len(states))
            if states[frame_index]["action_id"] != states[frame_index - 1]["action_id"]
        ]
        motion_semantics[str(slot)] = {
            "moving": actor["moving"],
            "native_rate_active_interval": actor["native_rate_active_interval"],
            "native_rate_action_segments": actor.get("native_rate_action_segments", []),
            "trajectory_preflight": actor["trajectory_preflight"],
            "action_counts": action_counts[str(slot)],
        }
    return {
        "profile_frames": profile_frames,
        "action_counts": action_counts,
        "animation_timing_mode_counts": mode_counts,
        "action_transitions": transitions,
        "motion_semantics": motion_semantics,
    }


def _validate_profile_rir_plan(
    materialization_root: Path, profile: Mapping[str, Any]
) -> dict[str, Any]:
    plan = _load(materialization_root / "rir_job_plan.json")
    jobs = plan.get("jobs")
    _require(isinstance(jobs, list), "RIR plan jobs missing")
    expectation = profile["rir_expectation"]
    candidate = profile["authorities"]["candidate"]["value"]
    old_row = profile["authorities"]["selected_old_row"]["value"]
    output_episode_id = str(old_row["episode_id"])
    profile_episode_id = str(candidate["candidate_episode_id"])
    compared = {
        "stride_frames": plan.get("stride_frames"),
        "requested_pair_state_count": plan.get("requested_pair_state_count"),
        "unique_rir_job_count": plan.get("unique_rir_job_count"),
    }
    _require(
        compared
        == {
            key: expectation[key]
            for key in (
                "stride_frames",
                "requested_pair_state_count",
                "unique_rir_job_count",
            )
        },
        "actual RIR stride/requested/unique counts drift from motion profile",
    )
    normalized_plan = _normalize_profile_episode_id(
        plan,
        output_episode_id=output_episode_id,
        profile_episode_id=profile_episode_id,
    )
    normalized_hash = canonical_json_sha256(normalized_plan)
    _require(
        normalized_hash == expectation["canonical_plan_sha256"],
        "actual normalized RIR plan hash drift from motion profile",
    )
    source_slots = list(candidate["actors"])
    per_slot_distinct = {
        str(slot): len(
            {
                job["job_id"]
                for job in jobs
                if any(use.get("source_slot_id") == slot for use in job.get("uses", []))
            }
        )
        for slot in source_slots
    }
    uses = [use for job in jobs for use in job.get("uses", [])]
    _require(
        len(jobs) == compared["unique_rir_job_count"]
        and len(uses) == compared["requested_pair_state_count"],
        "actual RIR jobs/uses do not close profile counts",
    )
    return {
        "status": "pass_actual_normalized_plan_matches_motion_profile",
        "stride_frames": compared["stride_frames"],
        "requested_pair_state_count": compared["requested_pair_state_count"],
        "unique_rir_job_count": compared["unique_rir_job_count"],
        "distinct_rir_state_count_by_source_slot": per_slot_distinct,
        "exact_pose_cache_reuse_count": (
            int(compared["requested_pair_state_count"])
            - int(compared["unique_rir_job_count"])
        ),
        "actual_plan_sha256": canonical_json_sha256(plan),
        "normalized_actual_plan_sha256": normalized_hash,
        "profile_expected_plan_sha256": expectation["canonical_plan_sha256"],
        "episode_id_normalization": {
            "from": output_episode_id,
            "to": profile_episode_id,
        },
    }


def _validate_bound_motion_profile(
    materialization_root: Path,
    receipt: Mapping[str, Any],
    scenario: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    profile_path = materialization_root / "actor_motion_profile.json"
    profile = _load(profile_path)
    validate_actor_motion_profile(profile)
    candidate = profile["authorities"]["candidate"]["value"]
    _require(
        candidate.get("mechanism") == receipt.get("mechanism")
        and candidate.get("legacy_episode_id") == receipt.get("episode_id"),
        "motion profile/materialization identity drift",
    )
    receipt_binding = receipt.get("actor_motion_profile", {})
    candidate_binding = profile["authorities"]["candidate"]
    motion = _profile_motion_contract(profile)
    _require(
        receipt_binding.get("status") == "pass_bound_and_consumed_frame_by_frame"
        and receipt_binding.get("schema") == profile.get("schema")
        and receipt_binding.get("profile_content_sha256")
        == profile.get("profile_content_sha256")
        and receipt_binding.get("candidate_document_sha256")
        == candidate_binding.get("document_sha256")
        and receipt_binding.get("candidate_value_sha256")
        == candidate_binding.get("canonical_value_sha256")
        and receipt_binding.get("canonical_frame_sha256")
        == [frame["canonical_frame_sha256"] for frame in motion["profile_frames"]]
        and receipt_binding.get("derived_action_counts") == motion["action_counts"]
        and receipt_binding.get("legacy_root_motion_inference_used") is False
        and receipt_binding.get("qualification_claim") is False,
        "materialization receipt motion-profile binding drift",
    )
    suite_binding = scenario.get("actor_motion_profile_binding", {})
    _require(
        suite_binding
        == {
            "schema": profile["schema"],
            "profile_content_sha256": profile["profile_content_sha256"],
            "frame_count": len(motion["profile_frames"]),
            "qualification_claim": False,
        },
        "suite motion-profile binding drift",
    )
    _require(
        len(frames) == len(motion["profile_frames"]),
        "suite/profile frame count drift",
    )
    for suite_frame, profile_frame in zip(
        frames, motion["profile_frames"], strict=True
    ):
        observed_states = []
        for state in suite_frame.get("actor_states", []):
            _require(
                all(key in state for key in PROFILE_ACTOR_STATE_CORE_KEYS),
                f"suite profile actor-state core is incomplete at f{profile_frame['frame_index']}",
            )
            observed_states.append(
                {key: state[key] for key in PROFILE_ACTOR_STATE_CORE_KEYS}
            )
        _require(
            suite_frame.get("frame_index") == profile_frame["frame_index"]
            and suite_frame.get("pts_ticks") == profile_frame["pts_ticks"]
            and suite_frame.get("canonical_motion_profile_frame_sha256")
            == profile_frame["canonical_frame_sha256"]
            and observed_states == profile_frame["actor_states"],
            f"suite actor-state core drift from profile at f{profile_frame['frame_index']}",
        )
    root_application = receipt.get("suite_actor_root_application", {})
    expected_provenance = {
        str(slot): {
            "method": "hash_bound_actor_motion_profile_v1",
            "profile_content_sha256": profile["profile_content_sha256"],
            "native_motion_authority": actor.get("native_motion_authority"),
            "native_rate_active_interval": actor.get("native_rate_active_interval"),
            "native_rate_action_segments": actor.get("native_rate_action_segments", []),
        }
        for slot, actor in candidate["actors"].items()
    }
    _require(
        root_application.get("status") == "pass_exact_all_75_frames"
        and root_application.get("maximum_root_path_error_m") == 0.0
        and root_application.get("action_counts") == motion["action_counts"]
        and root_application.get("root_path_provenance") == expected_provenance
        and root_application.get("animation_timing") == motion["motion_semantics"],
        "suite root/action/profile application drift",
    )
    rir = _validate_profile_rir_plan(materialization_root, profile)
    dynamic_comparison = receipt.get("dynamic_acoustics", {}).get(
        "actor_motion_profile_comparison", {}
    )
    _require(
        receipt_binding.get("derived_rir_counts")
        == {
            "stride_frames": rir["stride_frames"],
            "requested_pair_state_count": rir["requested_pair_state_count"],
            "unique_rir_job_count": rir["unique_rir_job_count"],
        },
        "receipt RIR counts drift from profile-derived actual plan",
    )
    _require(
        dynamic_comparison.get("status")
        == "pass_actual_plan_matches_profile_expectation"
        and dynamic_comparison.get("profile_content_sha256")
        == profile["profile_content_sha256"]
        and dynamic_comparison.get("profile_expected_plan_sha256")
        == rir["profile_expected_plan_sha256"]
        and dynamic_comparison.get("actual_plan_sha256") == rir["actual_plan_sha256"]
        and dynamic_comparison.get("normalized_actual_plan_sha256")
        == rir["normalized_actual_plan_sha256"]
        and dynamic_comparison.get("compared_counts")
        == {
            "stride_frames": rir["stride_frames"],
            "requested_pair_state_count": rir["requested_pair_state_count"],
            "unique_rir_job_count": rir["unique_rir_job_count"],
        }
        and dynamic_comparison.get("derived_cache_reuse_count")
        == rir["exact_pose_cache_reuse_count"],
        "materializer RIR/profile comparison receipt drift",
    )
    base_frames = profile["authorities"]["base_suite"]["value"]["scenarios"][0]["plan"][
        "frames"
    ]
    base_rotations = {
        tuple(frame["camera_state"]["world_from_rig"]["rotation_xyzw"])
        for frame in base_frames
    }
    return {
        "status": "pass_hash_bound_profile_consumed_exactly",
        "profile_file_sha256": sha256_file(profile_path),
        "profile_document_canonical_sha256": canonical_json_sha256(profile),
        "profile_content_sha256": profile["profile_content_sha256"],
        "candidate_document_sha256": candidate_binding.get("document_sha256"),
        "candidate_value_sha256": candidate_binding["canonical_value_sha256"],
        "action_counts": motion["action_counts"],
        "animation_timing_mode_counts": motion["animation_timing_mode_counts"],
        "action_transitions": motion["action_transitions"],
        "live_skeletal_transition_evidence": {
            "status": "pending_gpu_runtime_readback",
            "cpu_profile_sequence_validated": True,
            "qualification_claim": False,
            "claim_boundary": (
                "declared Idle/Walk transitions are exact in the CPU profile and "
                "suite, but live skeletal transition readback remains pending"
            ),
        },
        "expected_listener_orientation_count": len(base_rotations),
        "rir_plan": rir,
    }


def _validate_materialization(materialization_root: Path) -> dict[str, Any]:
    receipt = _load(materialization_root / "materialization_receipt.json")
    mechanism = receipt.get("mechanism")
    _require(isinstance(mechanism, str) and bool(mechanism), "mechanism is missing")
    _require(
        receipt.get("status")
        == "pass_cpu_materialized_pending_rir_execution_audio_and_gpu1"
        and receipt.get("frame_count") == FRAME_COUNT
        and receipt.get("gpu_launch_authorized") is False
        and receipt.get("formal") is False,
        "materialization receipt boundary drift",
    )
    audio_receipt = receipt.get("audio_program", {})
    audio = audio_receipt.get("validation", {})
    checks = audio.get("dry_bus_activity_checks", {})
    speech_window = audio.get("speech_frame_window_inclusive")
    _require(
        isinstance(speech_window, list)
        and len(speech_window) == 2
        and all(isinstance(value, int) for value in speech_window),
        "AudioProgram speech window missing",
    )
    first_frame, last_frame = speech_window
    expected_checks = {
        f"frame_{first_frame - 1}_silent": True,
        f"frame_{first_frame}_active": True,
        f"frame_{last_frame}_active": True,
        f"frame_{last_frame + 1}_silent": True,
        "source2_all_zero": True,
    }
    program = _load(
        materialization_root / "controlled_audio_program/audio_program.json"
    )
    events = program.get("events", [])
    _require(len(events) == 1, "AudioProgram event closure failed")
    event = events[0]
    active_sample_count = int(event["end_sample_exclusive"]) - int(
        event["start_sample"]
    )
    _require(
        audio.get("status") == "pass"
        and audio.get("target_event_count") == 1
        and audio.get("distractor_event_count") == 0
        and first_frame == 7
        and last_frame < FRAME_COUNT - 1
        and audio.get("target_active_sample_count") == active_sample_count
        and audio.get("target_sound_asset_id") == event.get("sound_asset_id")
        and audio_receipt.get("target_sound_asset_id") == event.get("sound_asset_id")
        and checks == expected_checks,
        "AudioProgram activity closure failed",
    )
    scenario, plan = _scenario(materialization_root)
    _require(
        "static_camera_upgrade" not in scenario,
        "static camera authority leaked into dynamic suite",
    )
    frames = plan.get("frames")
    _require(
        isinstance(frames, list)
        and len(frames) == FRAME_COUNT
        and [frame.get("frame_index") for frame in frames] == list(range(FRAME_COUNT)),
        "suite full75 frame closure failed",
    )
    rig = scenario.get("camera_trajectory_binding", {})
    _require(
        rig.get("schema") == "avengine_sensor_rig_trajectory_v1"
        and rig.get("frame_count") == FRAME_COUNT,
        "dynamic sensor-rig authority missing",
    )
    source_logic = plan.get("source_logic", {})
    sources = {
        item.get("entity_actor_id"): item.get("activation")
        for item in source_logic.get("sources", [])
        if isinstance(item, Mapping)
    }
    _require(
        sources == {"source1_actor": "active", "source2_actor": "silent"}
        and source_logic.get("clip_flags", {})
        .get("both_sources_active", {})
        .get("value")
        is False,
        "suite source activation contradicts AudioProgram",
    )
    profile_path = materialization_root / "actor_motion_profile.json"
    if profile_path.is_file():
        motion_profile = _validate_bound_motion_profile(
            materialization_root, receipt, scenario, frames
        )
        rir_contract = motion_profile["rir_plan"]
        expected = {
            "stride": rir_contract["stride_frames"],
            "requested": rir_contract["requested_pair_state_count"],
            "unique": rir_contract["unique_rir_job_count"],
            "per_slot": rir_contract["distinct_rir_state_count_by_source_slot"],
            "reuse": rir_contract["exact_pose_cache_reuse_count"],
        }
        expected_listener_orientation_count = motion_profile[
            "expected_listener_orientation_count"
        ]
        action_counts = motion_profile["action_counts"]
        root_provenance = receipt["suite_actor_root_application"][
            "root_path_provenance"
        ]
        animation_timing = receipt["suite_actor_root_application"]["animation_timing"]
    else:
        _require(
            mechanism == LEGACY_CAMERA_PAN_MECHANISM,
            f"{mechanism}: actor_motion_profile.json is required",
        )
        _require(
            receipt.get("actor_motion_profile")
            == {
                "status": "explicit_legacy_camera_pan_adapter",
                "legacy_root_motion_inference_used": True,
                "qualification_claim": False,
            },
            "legacy camera-pan adapter receipt drift",
        )
        motion_profile = {
            "status": "explicit_legacy_camera_pan_adapter",
            "live_skeletal_transition_evidence": {
                "status": "not_applicable_static_actors"
            },
        }
        expected = {
            "stride": 1,
            "requested": FRAME_COUNT * 2,
            "unique": LEGACY_CAMERA_PAN_ACOUSTICS["unique"],
            "per_slot": {
                "source1": LEGACY_CAMERA_PAN_ACOUSTICS["source1"],
                "source2": LEGACY_CAMERA_PAN_ACOUSTICS["source2"],
            },
            "reuse": LEGACY_CAMERA_PAN_ACOUSTICS["reuse"],
        }
        expected_listener_orientation_count = LEGACY_CAMERA_PAN_MOTION[
            "listener_orientation_count"
        ]
        action_counts = LEGACY_CAMERA_PAN_MOTION["action_counts"]
        root_application = receipt.get("suite_actor_root_application", {})
        _require(
            root_application.get("status") == "pass_exact_all_75_frames"
            and root_application.get("maximum_root_path_error_m") == 0.0
            and root_application.get("action_counts") == action_counts,
            "legacy camera-pan root/action application drift",
        )
        root_provenance = root_application.get("root_path_provenance", {})
        animation_timing = root_application.get("animation_timing", {})

    dynamic = receipt.get("dynamic_acoustics", {})
    _require(
        dynamic.get("frame_stride") == expected["stride"]
        and dynamic.get("requested_source_frame_uses") == expected["requested"]
        and dynamic.get("distinct_rir_state_count") == expected["unique"]
        and dynamic.get("distinct_rir_state_count_by_source_slot")
        == expected["per_slot"]
        and dynamic.get("exact_pose_cache_reuse_count") == expected["reuse"],
        "materialized dynamic acoustic counts drift",
    )

    camera_application = receipt.get("suite_camera_application", {})
    sensor_rig = _load(materialization_root / "sensor_rig_trajectory.json")
    rig_frames = sensor_rig.get("frames", [])
    _require(
        camera_application.get("status") == "pass_exact_all_75_frames"
        and camera_application.get("applied_frame_count") == FRAME_COUNT
        and camera_application.get("listener_coupled_to_camera") is True
        and camera_application.get("distinct_listener_orientation_count")
        == expected_listener_orientation_count
        and len(rig_frames) == FRAME_COUNT,
        "suite listener orientation application drift",
    )
    rig_rotations = [
        tuple(float(value) for value in frame["world_from_rig"]["rotation_xyzw"])
        for frame in rig_frames
    ]
    suite_rotations = [
        tuple(
            float(value)
            for value in frame["camera_state"]["world_from_rig"]["rotation_xyzw"]
        )
        for frame in frames
    ]
    _require(
        suite_rotations == rig_rotations
        and len(set(rig_rotations)) == expected_listener_orientation_count,
        "suite camera rotations do not exactly match sensor-rig authority",
    )
    yaw_path = [float(frame["camera_state"]["habitat_yaw_deg"]) for frame in frames]
    yaw_span_deg = max(yaw_path) - min(yaw_path)
    if mechanism == LEGACY_CAMERA_PAN_MECHANISM and not profile_path.is_file():
        _require(
            camera_application.get("distinct_listener_pose_count") == FRAME_COUNT
            and camera_application.get("habitat_yaw_path_deg") == yaw_path
            and abs(float(camera_application.get("habitat_yaw_span_deg")) - 6.0)
            <= 1.0e-9
            and yaw_span_deg >= 5.9
            and all(current > previous for previous, current in pairwise(yaw_path)),
            "camera pan must apply 75 monotonic orientations spanning at least 5.9 degrees",
        )
        for slot in ("source1", "source2"):
            actor_id = f"{slot}_actor"
            states = [
                next(
                    state
                    for state in frame["actor_states"]
                    if state["actor_id"] == actor_id
                )
                for frame in frames
            ]
            translations = {
                tuple(float(value) for value in state["translation_m"])
                for state in states
            }
            _require(
                len(translations) == 1
                and all(
                    state.get("action_id") == "idle"
                    and state.get("animation_timing_mode") == "held_idle_v1"
                    for state in states
                ),
                f"{slot}: camera-pan actor must remain static and idle",
            )
    return {
        "status": "pass",
        "episode_id": receipt["episode_id"],
        "mechanism": mechanism,
        "frame_count": FRAME_COUNT,
        "requested_source_frame_uses": expected["requested"],
        "rir_stride_frames": expected["stride"],
        "expected_unique_rir_job_count": expected["unique"],
        "expected_rir_count_by_source_slot": expected["per_slot"],
        "exact_pose_cache_reuse_count": expected["reuse"],
        "speech_frame_window_inclusive": speech_window,
        "target_active_sample_count": active_sample_count,
        "target_sound_asset_id": event["sound_asset_id"],
        "root_path_provenance": root_provenance,
        "animation_timing": animation_timing,
        "action_counts": action_counts,
        "motion_profile": motion_profile,
        "distinct_listener_orientation_count": len(set(rig_rotations)),
        "camera_yaw_span_deg": yaw_span_deg,
    }


def _validate_acoustics(
    materialization_root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    plan = _load(materialization_root / "rir_job_plan.json")
    jobs = plan.get("jobs")
    _require(isinstance(jobs, list), "RIR plan jobs missing")
    uses = [use for job in jobs for use in job.get("uses", [])]
    distinct_by_slot = {
        slot: len(
            {
                job["job_id"]
                for job in jobs
                if any(use.get("source_slot_id") == slot for use in job.get("uses", []))
            }
        )
        for slot in expected["expected_rir_count_by_source_slot"]
    }
    _require(
        len(jobs) == expected["expected_unique_rir_job_count"]
        and plan.get("unique_rir_job_count") == len(jobs)
        and plan.get("stride_frames") == expected["rir_stride_frames"]
        and plan.get("requested_pair_state_count")
        == expected["requested_source_frame_uses"]
        and len(uses) == expected["requested_source_frame_uses"]
        and distinct_by_slot == expected["expected_rir_count_by_source_slot"]
        and plan.get("cache_reuse_count") == expected["exact_pose_cache_reuse_count"],
        "plan-side dynamic RIR closure failed",
    )
    cache = _load(materialization_root / "rir_cache_v3/receipt.json")
    _require(
        cache.get("status") == "pass"
        and cache.get("full_plan_complete") is True
        and cache.get("selected_job_count") == len(jobs),
        "dynamic RIR cache is incomplete",
    )
    delivery = _load(materialization_root / "binaural_v1/delivery.json")
    _require(
        delivery.get("status") == "pass"
        and delivery.get("episode_count") == 1
        and delivery.get("sample_count") == 1
        and delivery.get("both_sources_active") is False
        and delivery.get("qualification_claim") is False
        and delivery.get("source_activity_contract")
        == "m6_audio_program_event_windows_v1"
        and delivery.get("sensor_rig_rir_alignment", {}).get("checked_use_count")
        == expected["requested_source_frame_uses"],
        "dynamic binaural delivery contract failed",
    )
    samples = _load(materialization_root / "binaural_v1/samples.json")
    rows = samples.get("samples")
    _require(
        samples.get("status") == "pass" and isinstance(rows, list) and len(rows) == 1,
        "binaural sample closure failed",
    )
    row = rows[0]
    summary = row.get("source_activity_summary", {})
    _require(
        row.get("episode_id") == expected["episode_id"]
        and row.get("both_sources_active") is False
        and summary.get("active_source_slots") == ["source1"]
        and summary.get("silent_source_slots") == ["source2"]
        and summary.get("active_sample_count_by_source_slot")
        == {"source1": expected["target_active_sample_count"], "source2": 0},
        "rendered source activity closure failed",
    )
    audio_root = materialization_root / "binaural_v1/audio/binaural"
    stem_root = audio_root / "stems"
    sample_id = row["sample_id"]
    mixture, mixture_contract = _wav_float32(audio_root / f"{sample_id}.wav")
    source1, source1_contract = _wav_float32(stem_root / "source1" / f"{sample_id}.wav")
    source2, source2_contract = _wav_float32(stem_root / "source2" / f"{sample_id}.wav")
    _require(np.count_nonzero(source1) > 0, "target binaural stem is silent")
    _require(
        np.count_nonzero(source2) == 0, "distractor binaural stem is not exactly silent"
    )
    _require(
        np.array_equal(mixture, source1 + source2), "mixture is not the exact stem sum"
    )
    return {
        "status": "pass",
        "requested_source_frame_uses": expected["requested_source_frame_uses"],
        "rir_stride_frames": expected["rir_stride_frames"],
        "unique_rir_job_count": len(jobs),
        "selected_cache_job_count": cache["selected_job_count"],
        "listener_aligned_use_count": expected["requested_source_frame_uses"],
        "mixture": mixture_contract,
        "source1": source1_contract,
        "source2": source2_contract,
        "source2_exact_zero": True,
        "mixture_exact_stem_sum": True,
    }


def _validate_runtime_transform_readbacks(
    readbacks: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    frames = plan.get("frames", [])
    _require(len(frames) == FRAME_COUNT, "runtime transform plan is not full75")
    passes = {"normal": readbacks.get("normal", [])}
    passes.update(
        {
            f"target_only_{slot}": items
            for slot, items in readbacks.get("target_only", {}).items()
        }
    )
    _require(
        set(passes) == {"normal", "target_only_source1", "target_only_source2"}
        and all(len(items) == FRAME_COUNT for items in passes.values()),
        "runtime transform readback pass closure failed",
    )

    maximum_camera_location_error_cm = 0.0
    maximum_camera_rotation_error_deg = 0.0
    maximum_actor_location_error_cm = 0.0
    maximum_actor_rotation_error_deg = 0.0
    normal_camera_yaws: list[float] = []

    def angular_error(observed: float, expected: float) -> float:
        return abs((observed - expected + 180.0) % 360.0 - 180.0)

    for pass_name, items in passes.items():
        for frame_index, observed in enumerate(items):
            expected_frame = frames[frame_index]
            _require(
                observed.get("camera", {}).get("frame_index") == frame_index,
                f"{pass_name}: camera frame index drift at {frame_index}",
            )
            observed_camera = observed["camera"]
            expected_camera = expected_frame["camera_state"]
            _require(
                observed_camera.get("expected_pose_hash")
                == expected_camera.get("pose_hash"),
                f"{pass_name}: camera pose authority drift at {frame_index}",
            )
            camera_location_errors = [
                abs(float(actual) - float(expected))
                for actual, expected in zip(
                    observed_camera["location_cm"],
                    expected_camera["ue_position_cm"],
                )
            ]
            camera_rotation_errors = [
                angular_error(float(actual), float(expected))
                for actual, expected in zip(
                    observed_camera["rotation_deg"],
                    [0.0, 0.0, expected_camera["ue_yaw_deg"]],
                )
            ]
            maximum_camera_location_error_cm = max(
                maximum_camera_location_error_cm, *camera_location_errors
            )
            maximum_camera_rotation_error_deg = max(
                maximum_camera_rotation_error_deg, *camera_rotation_errors
            )
            if pass_name == "normal":
                normal_camera_yaws.append(float(observed_camera["rotation_deg"][2]))

            expected_actors = {
                state["actor_id"]: state for state in expected_frame["actor_states"]
            }
            observed_actors = observed.get("actors", {})
            _require(
                set(observed_actors) == set(expected_actors),
                f"{pass_name}: actor transform set drift at {frame_index}",
            )
            for actor_id, expected_actor in expected_actors.items():
                observed_actor = observed_actors[actor_id]
                _require(
                    observed_actor.get("frame_index") == frame_index,
                    f"{pass_name}: {actor_id} frame index drift at {frame_index}",
                )
                actor_location_errors = [
                    abs(float(actual) - float(expected))
                    for actual, expected in zip(
                        observed_actor["location_cm"],
                        expected_actor["translation_ue_cm"],
                    )
                ]
                actor_rotation_errors = [
                    angular_error(float(actual), float(expected))
                    for actual, expected in zip(
                        observed_actor["rotation_deg"],
                        [0.0, 0.0, expected_actor["actor_yaw_ue_deg"]],
                    )
                ]
                maximum_actor_location_error_cm = max(
                    maximum_actor_location_error_cm, *actor_location_errors
                )
                maximum_actor_rotation_error_deg = max(
                    maximum_actor_rotation_error_deg, *actor_rotation_errors
                )

    _require(
        maximum_camera_location_error_cm <= 1.0e-6
        and maximum_camera_rotation_error_deg <= 1.0e-6
        and maximum_actor_location_error_cm <= 1.0e-6
        and maximum_actor_rotation_error_deg <= 1.0e-6,
        "runtime camera/actor transform readback drift",
    )
    return {
        "status": "pass_exact_all_normal_and_target_only_frames",
        "readback_pass_count": 3,
        "camera_readback_count": 225,
        "actor_readback_count": 450,
        "maximum_camera_location_error_cm": maximum_camera_location_error_cm,
        "maximum_camera_rotation_error_deg": maximum_camera_rotation_error_deg,
        "maximum_actor_location_error_cm": maximum_actor_location_error_cm,
        "maximum_actor_rotation_error_deg": maximum_actor_rotation_error_deg,
        "normal_distinct_camera_yaw_count": len(set(normal_camera_yaws)),
        "normal_camera_yaw_span_deg": max(normal_camera_yaws) - min(normal_camera_yaws),
    }


def _ground_contact_failure(reason: str) -> dict[str, Any]:
    return {
        "status": "fail",
        "release_authorized": False,
        "first_blocker": reason,
        "claim_boundary": (
            "missing_or_out_of_range_live_foot_floor_evidence_blocks_release"
        ),
    }


def _strict_ground_contact_release(
    assets: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    samples = assets.get("sampled_frames")
    _require(isinstance(samples, list), "live ground samples are missing")
    _require(
        [sample.get("frame_index") for sample in samples] == [0, 37, 74],
        "live ground samples must close f0/f37/f74",
    )
    # Evidence completeness is the first release question.  Legacy captures have
    # neither profiles nor live readbacks; diagnose the missing measurements before
    # considering any profile/tolerance semantics so the failure cannot be mistaken
    # for a threshold-configuration problem.
    for sample in samples:
        frame_index = int(sample["frame_index"])
        records = sample.get("per_instance")
        _require(
            isinstance(records, Mapping) and set(records) == {"source1", "source2"},
            f"frame {frame_index}: ground-contact instance closure failed",
        )
        for slot, record in records.items():
            actor_id = f"{slot}_actor"
            _require(
                isinstance(record, Mapping),
                f"{actor_id} ground sample is missing at frame {frame_index}",
            )
            ground = record.get("live_ground_contact_readback")
            _require(
                isinstance(ground, Mapping)
                and ground.get("schema") == GROUND_CONTACT_READBACK_SCHEMA
                and ground.get("status") == "pass_instrumented_measurement_only"
                and ground.get("ue_length_unit") == "centimeter",
                f"{actor_id} live ground readback is missing at frame {frame_index}",
            )
    declarations = {actor["actor_id"]: actor for actor in plan["actors"]}
    _require(
        set(declarations) == {"source1_actor", "source2_actor"},
        "ground-contact actor declaration closure failed",
    )
    profiles: dict[str, Mapping[str, Any]] = {}
    for actor_id, declaration in declarations.items():
        profile = declaration.get("ground_contact_release_profile")
        _require(
            isinstance(profile, Mapping),
            f"{actor_id} ground-contact release profile is missing",
        )
        _require(
            profile.get("schema") == GROUND_CONTACT_PROFILE_SCHEMA,
            f"{actor_id} ground-contact release profile schema drift",
        )
        _require(
            profile.get("ue_length_unit") == "centimeter"
            and profile.get("bone_names") == GROUND_CONTACT_BONES,
            f"{actor_id} ground-contact unit/bone declaration drift",
        )
        authority = profile.get("clearance_interval_authority")
        _require(
            isinstance(authority, Mapping)
            and authority.get("derived_from_live_diagnostic") is True
            and isinstance(authority.get("artifact"), str)
            and bool(authority["artifact"]),
            f"{actor_id} clearance interval lacks live diagnostic authority",
        )
        expected_actor = profile.get("expected_floor_hit_actor")
        expected_components = profile.get("expected_floor_hit_components")
        _require(
            isinstance(expected_actor, (str, int))
            and isinstance(expected_components, list)
            and expected_components
            and all(isinstance(value, (str, int)) for value in expected_components),
            f"{actor_id} expected floor identity is incomplete",
        )
        intervals = profile.get("support_anchor_clearance_interval_cm_by_action")
        _require(
            isinstance(intervals, Mapping) and set(intervals) == {"idle", "walk"},
            f"{actor_id} action clearance intervals are incomplete",
        )
        for action_id, interval in intervals.items():
            _require(
                isinstance(interval, list)
                and len(interval) == 2
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and np.isfinite(float(value))
                    for value in interval
                )
                and float(interval[0]) <= float(interval[1]),
                f"{actor_id} {action_id} clearance interval is invalid",
            )
        for key in (
            "minimum_individual_anchor_clearance_cm",
            "minimum_floor_normal_z",
        ):
            value = profile.get(key)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value)),
                f"{actor_id} {key} is not a declared finite threshold",
            )
        snap = profile.get("runtime_visual_ground_snap")
        _require(
            isinstance(snap, Mapping)
            and snap.get("schema") == "ue_dynamic_ground_snap_v1"
            and snap.get("target") == "attached_visual_actor_root_component"
            and snap.get("timeline_anchor_mutation_allowed") is False
            and snap.get("emitter_or_rir_mutation_allowed") is False,
            f"{actor_id} runtime visual ground snap contract is incomplete",
        )
        _require(
            float(snap.get("maximum_abs_correction_cm", 16.0)) <= 15.0
            and float(snap.get("maximum_abs_correction_cm", 0.0)) > 0.0
            and 0.0 <= float(snap.get("residual_tolerance_cm", 1.0)) <= 0.1,
            f"{actor_id} runtime visual ground snap exceeds normalization limits",
        )
        profiles[actor_id] = profile

    observed_floor_actors: set[str | int] = set()
    observed_floor_components: set[str | int] = set()
    support_clearances: list[float] = []
    individual_clearances: list[float] = []
    floor_normal_z_values: list[float] = []
    snap_corrections: list[float] = []
    snap_residuals: list[float] = []
    trace_count = 0
    per_actor_sample_count = {actor_id: 0 for actor_id in declarations}
    for sample in samples:
        frame_index = int(sample["frame_index"])
        expected_states = {
            state["actor_id"]: state
            for state in plan["frames"][frame_index]["actor_states"]
        }
        records = sample.get("per_instance")
        _require(
            isinstance(records, Mapping) and set(records) == {"source1", "source2"},
            f"frame {frame_index}: ground-contact instance closure failed",
        )
        for slot, record in records.items():
            actor_id = f"{slot}_actor"
            profile = profiles[actor_id]
            action_id = expected_states[actor_id]["action_id"]
            _require(
                record.get("current_action", {}).get("action_id") == action_id,
                f"{actor_id} ground sample action drift at frame {frame_index}",
            )
            ground = record.get("live_ground_contact_readback")
            _require(
                isinstance(ground, Mapping)
                and ground.get("schema") == GROUND_CONTACT_READBACK_SCHEMA
                and ground.get("status") == "pass_instrumented_measurement_only"
                and ground.get("ue_length_unit") == "centimeter",
                f"{actor_id} live ground readback is missing at frame {frame_index}",
            )
            sides = ground.get("sides")
            _require(
                isinstance(sides, Mapping) and set(sides) == set(GROUND_CONTACT_BONES),
                f"{actor_id} foot-side closure failed at frame {frame_index}",
            )
            snap = ground.get("runtime_visual_ground_snap")
            declared_snap = profile["runtime_visual_ground_snap"]
            _require(
                isinstance(snap, Mapping)
                and snap.get("schema") == "ue_dynamic_ground_snap_v1"
                and snap.get("status") == "passed"
                and snap.get("target") == "attached_visual_actor_root_component"
                and snap.get("timeline_anchor_mutated") is False
                and snap.get("emitter_or_rir_mutated") is False
                and snap.get("bounds_role") == "action_only_not_release_evidence",
                f"{actor_id} runtime visual ground snap did not pass",
            )
            correction_cm = float(snap["applied_z_correction_cm"])
            residual_cm = float(snap["residual_clearance_cm"])
            anchor_error_cm = float(snap["maximum_timeline_anchor_error_cm"])
            _require(
                np.isfinite(correction_cm)
                and abs(correction_cm)
                <= float(declared_snap["maximum_abs_correction_cm"])
                and np.isfinite(residual_cm)
                and abs(residual_cm) <= float(declared_snap["residual_tolerance_cm"])
                and anchor_error_cm <= 1.0e-6,
                f"{actor_id} runtime visual ground snap metric failed",
            )
            snap_trace = snap.get("floor_trace", {})
            _require(
                snap_trace.get("hit_actor") == profile["expected_floor_hit_actor"]
                and snap_trace.get("hit_component")
                in profile["expected_floor_hit_components"],
                f"{actor_id} ground snap hit an undeclared floor object",
            )
            snap_corrections.append(correction_cm)
            snap_residuals.append(residual_cm)
            sample_clearances: list[float] = []
            for side, expected_bones in GROUND_CONTACT_BONES.items():
                anchors = sides[side].get("anchors")
                _require(
                    isinstance(anchors, Mapping)
                    and set(anchors) == set(expected_bones),
                    f"{actor_id} {side} anchor closure failed at frame {frame_index}",
                )
                for anchor_kind, expected_bone in expected_bones.items():
                    anchor = anchors[anchor_kind]
                    trace = anchor.get("floor_trace")
                    _require(
                        anchor.get("bone_name") == expected_bone
                        and isinstance(trace, Mapping)
                        and trace.get("status") == "hit"
                        and trace.get("profile_name") == "BlockAll"
                        and trace.get("trace_complex") is True,
                        f"{actor_id} {expected_bone} trace authority drift",
                    )
                    clearance = float(anchor["bone_to_floor_clearance_cm"])
                    normal_z = float(trace["hit_normal_ue"][2])
                    _require(
                        np.isfinite(clearance) and np.isfinite(normal_z),
                        f"{actor_id} {expected_bone} trace is nonfinite",
                    )
                    _require(
                        clearance
                        >= float(profile["minimum_individual_anchor_clearance_cm"]),
                        f"{actor_id} {expected_bone} penetrates below its threshold",
                    )
                    _require(
                        normal_z >= float(profile["minimum_floor_normal_z"]),
                        f"{actor_id} {expected_bone} did not hit an allowed floor normal",
                    )
                    hit_actor = trace.get("hit_actor")
                    hit_component = trace.get("hit_component")
                    _require(
                        hit_actor == profile["expected_floor_hit_actor"]
                        and hit_component in profile["expected_floor_hit_components"],
                        f"{actor_id} {expected_bone} hit an undeclared floor object",
                    )
                    observed_floor_actors.add(hit_actor)
                    observed_floor_components.add(hit_component)
                    individual_clearances.append(clearance)
                    floor_normal_z_values.append(normal_z)
                    sample_clearances.append(clearance)
                    trace_count += 1
            support_clearance = min(sample_clearances)
            interval = profile["support_anchor_clearance_interval_cm_by_action"][
                action_id
            ]
            _require(
                float(interval[0]) <= support_clearance <= float(interval[1]),
                f"{actor_id} {action_id} support-anchor clearance is outside "
                f"the declared interval at frame {frame_index}",
            )
            support_clearances.append(support_clearance)
            per_actor_sample_count[actor_id] += 1
    _require(trace_count == 24, "ground-contact trace count is not 24")
    _require(
        per_actor_sample_count == {"source1_actor": 3, "source2_actor": 3},
        "ground-contact actor sample count drift",
    )
    return {
        "status": "pass",
        "release_authorized": True,
        "authority": (
            "live_four_bone_world_readback_plus_complex_runtime_floor_trace_v1"
        ),
        "sample_frame_indices": [0, 37, 74],
        "actor_sample_count": 6,
        "trace_count": trace_count,
        "runtime_visual_ground_snap_count": len(snap_corrections),
        "minimum_applied_visual_z_correction_cm": min(snap_corrections),
        "maximum_applied_visual_z_correction_cm": max(snap_corrections),
        "maximum_abs_visual_ground_snap_residual_cm": max(
            abs(value) for value in snap_residuals
        ),
        "timeline_anchor_and_emitter_mutation_count": 0,
        "observed_floor_actors": sorted(str(value) for value in observed_floor_actors),
        "observed_floor_components": sorted(
            str(value) for value in observed_floor_components
        ),
        "minimum_support_anchor_clearance_cm": min(support_clearances),
        "maximum_support_anchor_clearance_cm": max(support_clearances),
        "minimum_individual_anchor_clearance_cm": min(individual_clearances),
        "maximum_individual_anchor_clearance_cm": max(individual_clearances),
        "minimum_floor_normal_z": min(floor_normal_z_values),
    }


def _evaluate_ground_contact_release(
    assets: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a durable rejection instead of treating absent evidence as pass."""

    try:
        return _strict_ground_contact_release(assets, plan)
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        return _ground_contact_failure(str(error))


def _validate_runtime(capture_root: Path, materialization_root: Path) -> dict[str, Any]:
    readbacks = _load(capture_root / "runtime_readbacks.json")
    _require(
        len(readbacks.get("normal", [])) == FRAME_COUNT,
        "normal runtime readback count drift",
    )
    target_only = readbacks.get("target_only", {})
    _require(
        set(target_only) == {"source1", "source2"}
        and all(len(items) == FRAME_COUNT for items in target_only.values()),
        "target-only runtime readback count drift",
    )
    assets = _load(capture_root / "runtime_asset_readbacks.json")
    _require(
        assets.get("status") == "pass"
        and assets.get("frame_index") == 74
        and set(assets.get("per_instance", {})) == {"source1", "source2"},
        "dynamic live asset closure drift",
    )
    _, plan = _scenario(materialization_root)
    transforms = _validate_runtime_transform_readbacks(readbacks, plan)
    declared = {actor["actor_id"]: actor for actor in plan["actors"]}
    sampling = assets.get("sampling_contract", {})
    samples = assets.get("sampled_frames", [])
    _require(
        sampling.get("schema") == "avengine_native_spear_runtime_asset_sampling_v1"
        and sampling.get("status") == "pass"
        and sampling.get("frame_indices") == [0, 37, 74]
        and isinstance(samples, list)
        and [sample.get("frame_index") for sample in samples] == [0, 37, 74],
        "dynamic live asset sampling closure failed",
    )
    maximum_action_position_error_seconds = 0.0
    maximum_root_emitter_error_m = 0.0
    sampled_actions: dict[str, list[str]] = {"source1": [], "source2": []}
    for sample in samples:
        frame_index = int(sample["frame_index"])
        states = {
            state["actor_id"]: state
            for state in plan["frames"][frame_index]["actor_states"]
        }
        _require(
            sample.get("status") == "pass"
            and set(sample.get("per_instance", {})) == {"source1", "source2"},
            f"frame {frame_index}: sampled live asset closure drift",
        )
        for slot, record in sample["per_instance"].items():
            actor_id = f"{slot}_actor"
            state = states[actor_id]
            action = record.get("current_action", {})
            idle = record.get("standing_idle", {})
            emitter = record.get("emitter_native_readback", {})
            action_error = float(action.get("absolute_position_error_seconds", 1.0))
            emitter_error = float(emitter.get("maximum_absolute_error_m", 1.0))
            _require(
                record.get("status") == "pass"
                and idle.get("status") == "pass"
                and idle.get("runtime_loaded_handle") == idle.get("expected_handle")
                and action.get("status") == "pass"
                and action.get("action_id") == state["action_id"]
                and action.get("expected_path") == state["ue_animation"]
                and action.get("current_animation_path") == state["ue_animation"]
                and action.get("observed_animation_asset_handle")
                == action.get("expected_handle")
                == action.get("runtime_loaded_handle")
                and action_error <= 1.0e-4
                and emitter.get("status") == "pass"
                and emitter_error <= 1.0e-6
                and declared[actor_id]["animation_paths_by_action_id"][
                    state["action_id"]
                ]
                == state["ue_animation"],
                f"{slot}: frame {frame_index} live action/emitter closure failed",
            )
            maximum_action_position_error_seconds = max(
                maximum_action_position_error_seconds, action_error
            )
            maximum_root_emitter_error_m = max(
                maximum_root_emitter_error_m, emitter_error
            )
            sampled_actions[slot].append(str(action["action_id"]))
    ground_contact = _evaluate_ground_contact_release(assets, plan)
    return {
        "status": "pass",
        "normal_frame_readback_count": FRAME_COUNT,
        "target_only_frame_readback_count": 150,
        "live_asset_readback_frames": [0, 37, 74],
        "sampled_current_actions": sampled_actions,
        "maximum_action_position_error_seconds": maximum_action_position_error_seconds,
        "maximum_root_emitter_error_m": maximum_root_emitter_error_m,
        "ground_contact_release_gate": ground_contact,
        "transform_readbacks": transforms,
    }


def _evaluate_visibility_gate(
    truth: Mapping[str, Any], speech_frame_window_inclusive: Sequence[int]
) -> dict[str, Any]:
    _require(
        truth.get("schema") == "avengine_qa_pixel_visibility_truth_v1"
        and truth.get("status") == "computed_modal_target_only_v1",
        "pixel truth contract drift",
    )
    _require(
        truth.get("frame_indices") == list(range(FRAME_COUNT)),
        "pixel frame index drift",
    )
    _require(truth.get("resolution_hw") == [720, 1280], "pixel resolution drift")
    per_instance = truth.get("per_instance", {})
    target_frames = per_instance.get("source1", {}).get("frames", [])
    distractor_frames = per_instance.get("source2", {}).get("frames", [])
    _require(
        len(target_frames) == len(distractor_frames) == FRAME_COUNT,
        "pixel per-instance frame count drift",
    )
    _require(
        len(speech_frame_window_inclusive) == 2,
        "target speech-frame window is malformed",
    )
    start, end = [int(value) for value in speech_frame_window_inclusive]
    target_speech = [
        item for item in target_frames if start <= int(item["frame_index"]) <= end
    ]
    _require(
        len(target_speech) == end - start + 1, "target speech-frame truth incomplete"
    )

    def edge_touch(item: Mapping[str, Any]) -> bool:
        x0, y0, x1, y1 = [int(value) for value in item["target_bbox_xyxy_px"]]
        return x0 < 1 or y0 < 1 or x1 > 1278 or y1 > 718

    target_failures = [
        {
            "frame_index": int(item["frame_index"]),
            "visible_pixels": int(item["visible_pixels"]),
            "visible_fraction": float(item["visible_fraction"]),
            "bbox_touches_edge": edge_touch(item),
        }
        for item in target_speech
        if int(item["visible_pixels"]) < TARGET_SPEECH_VISIBLE_PIXELS_MINIMUM
        or float(item["visible_fraction"]) < TARGET_SPEECH_VISIBLE_FRACTION_MINIMUM
        or edge_touch(item)
    ]
    distractor_failures = [
        {
            "frame_index": int(item["frame_index"]),
            "visible_pixels": int(item["visible_pixels"]),
            "visible_fraction": float(item["visible_fraction"]),
            "bbox_touches_edge": edge_touch(item),
        }
        for item in distractor_frames
        if int(item["visible_pixels"]) < DISTRACTOR_VISIBLE_PIXELS_MINIMUM
        or float(item["visible_fraction"]) < DISTRACTOR_VISIBLE_FRACTION_MINIMUM
        or edge_touch(item)
    ]
    status = "pass" if not target_failures and not distractor_failures else "fail"
    return {
        "status": status,
        "authority": "native_normal_vs_target_only_metric_depth_v1",
        "thresholds": {
            "target_speech_visible_pixels_minimum": TARGET_SPEECH_VISIBLE_PIXELS_MINIMUM,
            "target_speech_visible_fraction_minimum": TARGET_SPEECH_VISIBLE_FRACTION_MINIMUM,
            "distractor_visible_pixels_minimum": DISTRACTOR_VISIBLE_PIXELS_MINIMUM,
            "distractor_visible_fraction_minimum": DISTRACTOR_VISIBLE_FRACTION_MINIMUM,
            "bbox_interior_margin_pixels": 1,
        },
        "target_speech": {
            "frame_window_inclusive": [start, end],
            "frame_count": len(target_speech),
            "minimum_visible_pixels": min(
                int(item["visible_pixels"]) for item in target_speech
            ),
            "minimum_visible_fraction": min(
                float(item["visible_fraction"]) for item in target_speech
            ),
            "failing_frame_count": len(target_failures),
            "failures": target_failures,
        },
        "distractor_all_frames": {
            "frame_count": len(distractor_frames),
            "minimum_visible_pixels": min(
                int(item["visible_pixels"]) for item in distractor_frames
            ),
            "minimum_visible_fraction": min(
                float(item["visible_fraction"]) for item in distractor_frames
            ),
            "failing_frame_count": len(distractor_failures),
            "failures": distractor_failures,
        },
    }


def _validate_capture(
    capture_root: Path,
    launch_receipt_path: Path,
    materialization_root: Path,
    episode_id: str,
    speech_frame_window_inclusive: Sequence[int],
) -> dict[str, Any]:
    launch = _load(launch_receipt_path)
    _require(
        launch.get("status") == "pass"
        and launch.get("capture_process_exit_code") == 0
        and launch.get("physical_gpu_index") == 1
        and launch.get("graphics_adapter_argument") == 1
        and launch.get("forbidden_physical_gpu_indices_used") == [],
        "GPU1 launch receipt failed",
    )
    manifest = _load(capture_root / "manifest.json")
    frame_contract = manifest.get("frame_contract", {})
    _require(
        manifest.get("status") == "pass"
        and manifest.get("scenario_id") == episode_id
        and frame_contract.get("frame_count") == FRAME_COUNT
        and frame_contract.get("captured_frame_indices") == list(range(FRAME_COUNT)),
        "dynamic capture manifest/frame closure failed",
    )
    rgb = sorted((capture_root / "rgb_frames").glob("frame_*.png"))
    _require(
        [path.name for path in rgb]
        == [f"frame_{index:06d}.png" for index in range(FRAME_COUNT)],
        "dynamic RGB sequence incomplete",
    )
    depth_path = capture_root / "metric_depth_native.npz"
    masks_path = capture_root / "native_pixel_masks_depth_authority_v1.npz"
    _require(
        depth_path.is_file() and masks_path.is_file(), "dynamic depth/masks missing"
    )
    with np.load(depth_path) as arrays:
        _require(
            set(arrays.files)
            == {
                "normal_depth_m",
                "target_only_source1_depth_m",
                "target_only_source2_depth_m",
            }
            and all(
                arrays[key].shape == (FRAME_COUNT, 720, 1280) for key in arrays.files
            ),
            "dynamic metric-depth arrays drift",
        )
    with np.load(masks_path) as arrays:
        _require(
            all(arrays[key].shape == (FRAME_COUNT, 720, 1280) for key in arrays.files)
            and np.all(np.count_nonzero(arrays["target_only_source1"], axis=(1, 2)) > 0)
            and np.all(
                np.count_nonzero(arrays["target_only_source2"], axis=(1, 2)) > 0
            ),
            "dynamic target-only arrays drift",
        )
    for video_name in ("native_rgb_visual_only.mp4", "native_rgb_binaural.mp4"):
        video = capture_root / video_name
        _require(video.is_file(), f"dynamic video missing: {video}")
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,r_frame_rate,width,height",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(completed.stdout)["streams"][0]
        _require(
            stream
            == {
                "width": 1280,
                "height": 720,
                "r_frame_rate": "15/1",
                "nb_frames": "75",
            },
            f"dynamic video stream drift: {stream}",
        )
    runtime = _validate_runtime(capture_root, materialization_root)
    visibility = _evaluate_visibility_gate(
        _load(capture_root / "pixel_visibility_truth.json"),
        speech_frame_window_inclusive,
    )
    return {
        "status": "pass",
        "captured_frame_count": FRAME_COUNT,
        "physical_gpu_index": 1,
        "forbidden_physical_gpu_indices_used": [],
        "runtime": runtime,
        "visibility_gate": visibility,
    }


def finalize(
    *,
    materialization_root: Path,
    output: Path,
    capture_root: Path | None = None,
    launch_receipt_path: Path | None = None,
) -> Path:
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    materialization = _validate_materialization(materialization_root)
    acoustics = _validate_acoustics(materialization_root, materialization)
    full_mode = capture_root is not None or launch_receipt_path is not None
    _require(
        not full_mode or (capture_root is not None and launch_receipt_path is not None),
        "full finalization requires both capture root and launch receipt",
    )
    result: dict[str, Any] = {
        "schema": "avengine_native_strict_two_human_dynamic_full75_finalization_v1",
        "status": "pass_cpu_ready_for_gpu1" if not full_mode else "pass",
        "episode_id": materialization["episode_id"],
        "mechanism": materialization["mechanism"],
        "dynamic_full75_canary_pass": False,
        "cpu_pre_capture_gate_pass": True,
        "gpu_launch_authorized": not full_mode,
        "formal_episode_count": 0,
        "formal": False,
        "qualification_claim": False,
        "materialization": materialization,
        "acoustics": acoustics,
        "capture": None,
        "artifacts": {
            "materialization_root": str(materialization_root.resolve()),
            "rir_cache_receipt": str(
                (materialization_root / "rir_cache_v3/receipt.json").resolve()
            ),
            "binaural_delivery": str(
                (materialization_root / "binaural_v1/delivery.json").resolve()
            ),
        },
    }
    if full_mode:
        assert capture_root is not None and launch_receipt_path is not None
        result["capture"] = _validate_capture(
            capture_root,
            launch_receipt_path,
            materialization_root,
            materialization["episode_id"],
            materialization["speech_frame_window_inclusive"],
        )
        result["gpu_launch_authorized"] = False
        result["artifacts"]["capture_root"] = str(capture_root.resolve())
        result["artifacts"]["launch_receipt"] = str(launch_receipt_path.resolve())
        visibility_pass = result["capture"]["visibility_gate"]["status"] == "pass"
        ground_contact = result["capture"]["runtime"]["ground_contact_release_gate"]
        ground_contact_pass = (
            ground_contact.get("status") == "pass"
            and ground_contact.get("release_authorized") is True
        )
        result["dynamic_full75_canary_pass"] = visibility_pass and ground_contact_pass
        if not ground_contact_pass:
            result["status"] = "rejected_ground_contact_release_gate"
            result["rejection_reason"] = ground_contact["first_blocker"]
            result["supersedes_machine_only_finalization"] = str(
                (
                    materialization_root
                    / "post_capture_finalization_v1"
                    / "finalization.json"
                ).resolve()
            )
        elif not visibility_pass:
            result["status"] = "rejected_visibility_gate"
            result["rejection_reason"] = (
                "native per-frame target/distractor visibility did not meet the "
                "frozen dynamic strict-two-human thresholds"
            )
            result["supersedes_machine_only_finalization"] = str(
                (
                    materialization_root
                    / "post_capture_finalization_v1"
                    / "finalization.json"
                ).resolve()
            )
    output.mkdir(parents=True)
    path = output / "finalization.json"
    _write(path, result)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = finalize(
        materialization_root=args.materialization_root.resolve(),
        capture_root=args.capture_root.resolve() if args.capture_root else None,
        launch_receipt_path=(
            args.launch_receipt.resolve() if args.launch_receipt else None
        ),
        output=args.output.resolve(),
    )
    finalized = _load(result)
    if finalized["status"] == "rejected_visibility_gate":
        print(f"STRICT_TWO_HUMAN_DYNAMIC_FULL75_FINALIZER_REJECTED result={result}")
        return 2
    print(f"STRICT_TWO_HUMAN_DYNAMIC_FULL75_FINALIZER_OK result={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
