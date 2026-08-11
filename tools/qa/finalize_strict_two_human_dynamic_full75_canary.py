#!/usr/bin/env python3
"""Fail-closed finalizer for strict two-human dynamic full75 canaries."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

FRAME_COUNT = 75
SAMPLE_COUNT = 80_000
SAMPLE_RATE_HZ = 16_000
TARGET_SPEECH_VISIBLE_PIXELS_MINIMUM = 10_000
TARGET_SPEECH_VISIBLE_FRACTION_MINIMUM = 0.8
DISTRACTOR_VISIBLE_PIXELS_MINIMUM = 5_000
DISTRACTOR_VISIBLE_FRACTION_MINIMUM = 0.5
EXPECTED_ACOUSTICS = {
    "target_moves": {"unique": 76, "source1": 75, "source2": 1, "reuse": 74},
    "distractor_moves": {"unique": 76, "source1": 1, "source2": 75, "reuse": 74},
    "both_move": {"unique": 150, "source1": 75, "source2": 75, "reuse": 0},
    "camera_pan_both_static": {
        "unique": 150,
        "source1": 75,
        "source2": 75,
        "reuse": 0,
    },
}
INTERPOLATED_PATH_METHODS = {
    "arc_length_interpolation_of_native_polyline_v1",
    "equal_arc_interpolation_of_exact_native_human_polyline_v1",
}
EXPECTED_MOTION = {
    "target_moves": {
        "action_counts": {
            "source1": {"idle": 0, "walk": 75},
            "source2": {"idle": 75, "walk": 0},
        },
        "interpolated_slots": ["source1"],
        "listener_orientation_count": 1,
    },
    "distractor_moves": {
        "action_counts": {
            "source1": {"idle": 75, "walk": 0},
            "source2": {"idle": 0, "walk": 75},
        },
        "interpolated_slots": ["source2"],
        "listener_orientation_count": 1,
    },
    "both_move": {
        "action_counts": {
            "source1": {"idle": 0, "walk": 75},
            "source2": {"idle": 0, "walk": 75},
        },
        "interpolated_slots": ["source1", "source2"],
        "listener_orientation_count": 1,
    },
    "camera_pan_both_static": {
        "action_counts": {
            "source1": {"idle": 75, "walk": 0},
            "source2": {"idle": 75, "walk": 0},
        },
        "interpolated_slots": [],
        "listener_orientation_count": 75,
    },
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


def _validate_materialization(materialization_root: Path) -> dict[str, Any]:
    receipt = _load(materialization_root / "materialization_receipt.json")
    mechanism = receipt.get("mechanism")
    _require(mechanism in EXPECTED_ACOUSTICS, f"unsupported mechanism: {mechanism}")
    expected = EXPECTED_ACOUSTICS[str(mechanism)]
    _require(
        receipt.get("status")
        == "pass_cpu_materialized_pending_rir_execution_audio_and_gpu1"
        and receipt.get("frame_count") == FRAME_COUNT
        and receipt.get("gpu_launch_authorized") is False
        and receipt.get("formal") is False,
        "materialization receipt boundary drift",
    )
    dynamic = receipt.get("dynamic_acoustics", {})
    _require(
        dynamic.get("requested_source_frame_uses") == 150
        and dynamic.get("distinct_rir_state_count") == expected["unique"]
        and dynamic.get("distinct_rir_state_count_by_source_slot")
        == {"source1": expected["source1"], "source2": expected["source2"]}
        and dynamic.get("exact_pose_cache_reuse_count") == expected["reuse"],
        "materialized dynamic acoustic counts drift",
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
    root_application = receipt.get("suite_actor_root_application", {})
    expected_motion = EXPECTED_MOTION[str(mechanism)]
    _require(
        root_application.get("status") == "pass_exact_all_75_frames"
        and root_application.get("maximum_root_path_error_m") == 0.0
        and root_application.get("action_counts") == expected_motion["action_counts"],
        "suite action/root application drift",
    )
    root_provenance = root_application.get("root_path_provenance", {})
    animation_timing = root_application.get("animation_timing", {})
    interpolated_slots = []
    for slot in ("source1", "source2"):
        provenance = root_provenance.get(slot, {})
        if provenance.get("method") not in INTERPOLATED_PATH_METHODS:
            continue
        interpolated_slots.append(slot)
        timing = animation_timing.get(slot, {})
        _require(
            provenance.get("interior_output_roots_exact_native_frame_readbacks")
            is False
            and provenance.get("endpoints_exact_native_readbacks") is True
            and timing.get("schema") == "avengine_arc_length_bound_animation_timing_v1"
            and timing.get("status") == "pass"
            and timing.get("mode") == "arc_length_preserving_native_stride_v1"
            and len(timing.get("action_phase_path", [])) == FRAME_COUNT
            and len(timing.get("action_time_ticks_path", [])) == FRAME_COUNT
            and timing.get("phase_cycle_count", 0.0) > 0.0
            and timing.get("average_root_speed_m_per_second", 0.0) > 0.0
            and timing.get("maximum_phase_per_meter_error", 1.0) <= 1.0e-8
            and timing.get("maximum_forward_angular_error_deg", 1.0) <= 1.0e-5
            and timing.get("maximum_tangent_yaw_error_deg", 1.0) <= 1.0e-5,
            "arc-length-bound slow-walk timing closure failed",
        )
        actor_states = [
            next(
                state
                for state in frame["actor_states"]
                if state["actor_id"] == f"{slot}_actor"
            )
            for frame in frames
        ]
        _require(
            all(
                state.get("animation_timing_mode")
                == "arc_length_preserving_native_stride_v1"
                and abs(
                    float(state["action_phase"])
                    - float(timing["action_phase_path"][frame_index])
                )
                <= 1.0e-12
                and int(state["action_time_ticks"])
                == int(timing["action_time_ticks_path"][frame_index])
                for frame_index, state in enumerate(actor_states)
            ),
            "suite slow-walk phase path was not applied exactly",
        )
    _require(
        interpolated_slots == expected_motion["interpolated_slots"],
        "moving source provenance/timing slot drift",
    )

    camera_application = receipt.get("suite_camera_application", {})
    sensor_rig = _load(materialization_root / "sensor_rig_trajectory.json")
    rig_frames = sensor_rig.get("frames", [])
    _require(
        camera_application.get("status") == "pass_exact_all_75_frames"
        and camera_application.get("applied_frame_count") == FRAME_COUNT
        and camera_application.get("listener_coupled_to_camera") is True
        and camera_application.get("distinct_listener_orientation_count")
        == expected_motion["listener_orientation_count"]
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
        and len(set(rig_rotations)) == expected_motion["listener_orientation_count"],
        "suite camera rotations do not exactly match sensor-rig authority",
    )
    yaw_path = [float(frame["camera_state"]["habitat_yaw_deg"]) for frame in frames]
    yaw_span_deg = max(yaw_path) - min(yaw_path)
    if mechanism == "camera_pan_both_static":
        _require(
            camera_application.get("distinct_listener_pose_count") == FRAME_COUNT
            and camera_application.get("habitat_yaw_path_deg") == yaw_path
            and abs(float(camera_application.get("habitat_yaw_span_deg")) - 6.0)
            <= 1.0e-9
            and yaw_span_deg >= 5.9
            and all(
                current > previous for previous, current in zip(yaw_path, yaw_path[1:])
            ),
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
        "requested_source_frame_uses": 150,
        "expected_unique_rir_job_count": expected["unique"],
        "expected_rir_count_by_source_slot": {
            "source1": expected["source1"],
            "source2": expected["source2"],
        },
        "exact_pose_cache_reuse_count": expected["reuse"],
        "speech_frame_window_inclusive": speech_window,
        "target_active_sample_count": active_sample_count,
        "target_sound_asset_id": event["sound_asset_id"],
        "root_path_provenance": root_provenance,
        "animation_timing": animation_timing,
        "action_counts": root_application["action_counts"],
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
    _require(
        len(jobs) == expected["expected_unique_rir_job_count"]
        and plan.get("unique_rir_job_count") == len(jobs)
        and plan.get("requested_pair_state_count") == 150
        and len(uses) == 150
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
        == 150,
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
        "requested_source_frame_uses": 150,
        "unique_rir_job_count": len(jobs),
        "selected_cache_job_count": cache["selected_job_count"],
        "listener_aligned_use_count": 150,
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
    return {
        "status": "pass",
        "normal_frame_readback_count": FRAME_COUNT,
        "target_only_frame_readback_count": 150,
        "live_asset_readback_frames": [0, 37, 74],
        "sampled_current_actions": sampled_actions,
        "maximum_action_position_error_seconds": maximum_action_position_error_seconds,
        "maximum_root_emitter_error_m": maximum_root_emitter_error_m,
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
        result["dynamic_full75_canary_pass"] = visibility_pass
        if not visibility_pass:
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
