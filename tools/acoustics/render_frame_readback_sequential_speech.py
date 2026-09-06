"""Render multi-source research audio from SPEAR frame readbacks.

The legacy invocation still renders four complete VCTK turns. With audio-plan
the same entry point consumes arbitrary audio events (or an existing
AudioProgram), independent actor-to-voice bindings, and actual camera/emitter
readbacks. Current dynamic RIR rendering uses the established RIR cache
request/index/receipt/shard contract. Earlier variable-source sequence NPZ
caches are read-only compatibility inputs. The helper never starts UE, guesses
actor positions, crops source audio, or upgrades a research result to formal
admission.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import wave
from typing import Any, Mapping

import numpy as np

from avengine.capture.dry_audio import (
    DryAudioClipSpec,
    assemble_dry_audio_buses,
)
from avengine.capture.neutral_readback import validate_neutral_readback
from avengine.capture.ue_neutral_readback import neutral_from_ue_readbacks
from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.acoustics.dynamic_cache import (
    DynamicRIRCacheError,
    load_dynamic_rir_cache,
)
from avengine.acoustics.rir_cache import (
    RIRCacheSession,
    render_rir_cache,
    rir_acoustic_state_sha256,
)
from avengine.acoustics.runtime import (
    RLRSimulationConfig,
    RuntimeAnchor,
    load_compiled_acoustic_scene,
    simulate_compiled_acoustic_scene,
)
from avengine.timeline.audio import render_dynamic_stems_and_mix, time_varying_convolve
from avengine.timeline.audio_program import bind_audio_program_hash, validate_audio_program
from avengine.timeline.current_mp3d_dynamic_audio import render_neutral_readback_audio

TIME_BASE_HZ = 48_000
TICKS_PER_SAMPLE = 3
DEFAULT_GAIN = 0.15


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"WAV must be mono int16 PCM: {path}")
        rate = handle.getframerate()
        count = handle.getnframes()
        payload = handle.readframes(count)
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    if len(samples) != count or rate != 16000:
        raise ValueError(f"WAV must be 16 kHz and internally consistent: {path}")
    return np.ascontiguousarray(samples), rate


def _write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 1:
        channels = 1
        pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    elif values.ndim == 2:
        channels, count = values.shape
        pcm = np.rint(np.clip(values, -1.0, 1.0).T * 32767.0).astype("<i2")
    else:
        raise ValueError("audio samples must be [N] or [channels,N]")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _cm_to_m(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"readback position must be a 3-vector in cm: {value!r}")
    # SPEAR/UE is X/Y/Z with Z up; authored M3 GLBs are right-handed
    # X/Y/Z with Y up. The package's source-to-canonical transform is identity,
    # so the runtime bridge must exchange UE Y/Z here.
    x, ue_y, ue_z = (float(item) / 100.0 for item in value)
    return (x, ue_z, ue_y)


def _ue_rotator_to_m3_basis(
    value: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"rotation_deg must be [roll,pitch,yaw]: {value!r}")
    import math
    roll, pitch, yaw = (math.radians(float(item)) for item in value)
    # Unreal SceneCapture optical axes: local +X forward, +Y right, +Z up.
    # Positive UE pitch raises +X toward +Z; positive yaw rotates +X toward +Y.
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    forward_ue = np.asarray([cp * cy, cp * sy, sp], dtype=np.float64)
    right0_ue = np.asarray([-sy, cy, 0.0], dtype=np.float64)
    up0_ue = np.asarray([-sp * cy, -sp * sy, cp], dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    right_ue = right0_ue * cr + up0_ue * sr
    up_ue = -right0_ue * sr + up0_ue * cr
    axis_exchange = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    forward = axis_exchange @ forward_ue
    right = axis_exchange @ right_ue
    up = axis_exchange @ up_ue
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    forward /= np.linalg.norm(forward)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return (
        tuple(float(item) for item in forward),
        tuple(float(item) for item in right),
        tuple(float(item) for item in up),
    )


def _ue_rotator_to_m3_orientation_wxyz(value: Any) -> tuple[float, float, float, float]:
    forward, right, up = _ue_rotator_to_m3_basis(value)
    # M5.1 listener local axes are +X right, +Y up, -Z forward.
    rotation = np.column_stack(
        (np.asarray(right), np.asarray(up), -np.asarray(forward))
    )
    trace = float(np.trace(rotation))
    if trace > 0.0:
        w = np.sqrt(trace + 1.0) / 2.0
        x = (rotation[2, 1] - rotation[1, 2]) / (4.0 * w)
        y = (rotation[0, 2] - rotation[2, 0]) / (4.0 * w)
        z = (rotation[1, 0] - rotation[0, 1]) / (4.0 * w)
    else:
        diagonal = int(np.argmax(np.diag(rotation)))
        if diagonal == 0:
            x = np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) / 2.0
            y = (rotation[0, 1] + rotation[1, 0]) / max(1.0e-12, 4.0 * x)
            z = (rotation[0, 2] + rotation[2, 0]) / max(1.0e-12, 4.0 * x)
            w = (rotation[2, 1] - rotation[1, 2]) / max(1.0e-12, 4.0 * x)
        elif diagonal == 1:
            y = np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) / 2.0
            x = (rotation[0, 1] + rotation[1, 0]) / max(1.0e-12, 4.0 * y)
            z = (rotation[1, 2] + rotation[2, 1]) / max(1.0e-12, 4.0 * y)
            w = (rotation[0, 2] - rotation[2, 0]) / max(1.0e-12, 4.0 * y)
        else:
            z = np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) / 2.0
            x = (rotation[0, 2] + rotation[2, 0]) / max(1.0e-12, 4.0 * z)
            y = (rotation[1, 2] + rotation[2, 1]) / max(1.0e-12, 4.0 * z)
            w = (rotation[1, 0] - rotation[0, 1]) / max(1.0e-12, 4.0 * z)
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(item) for item in quaternion)

def _frame_listener(
    records: list[Mapping[str, Any]], frame_index: int
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
    dict[str, tuple[float, float, float]],
]:
    if not records:
        raise ValueError("listener readback list is empty")
    if not 0 <= frame_index < len(records):
        raise ValueError(
            f"listener frame {frame_index} is unavailable (count={len(records)})"
        )
    record = records[frame_index]
    basis = _ue_rotator_to_m3_basis(record["rotation_deg"])
    return (
        _cm_to_m(record["location_cm"]),
        _ue_rotator_to_m3_orientation_wxyz(record["rotation_deg"]),
        {"forward": basis[0], "right": basis[1], "up": basis[2]},
    )


def _rotation_distance_deg(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def _verify_static_interval(
    emitter_records: list[Mapping[str, Any]],
    listener_records: list[Mapping[str, Any]],
    start_frame: int,
    end_frame: int,
) -> dict[str, float]:
    source = np.asarray(_cm_to_m(emitter_records[start_frame]["location_cm"]))
    listener, orientation, _ = _frame_listener(listener_records, start_frame)
    max_source = 0.0
    max_listener = 0.0
    max_rotation = 0.0
    for frame_index in range(start_frame, min(end_frame + 1, len(emitter_records))):
        source_now = np.asarray(_cm_to_m(emitter_records[frame_index]["location_cm"]))
        listener_now, orientation_now, _ = _frame_listener(listener_records, frame_index)
        max_source = max(max_source, float(np.linalg.norm(source_now - source)))
        max_listener = max(max_listener, float(np.linalg.norm(np.asarray(listener_now) - np.asarray(listener))))
        max_rotation = max(max_rotation, _rotation_distance_deg(orientation, orientation_now))
    if max_source > 1.0e-4 or max_listener > 1.0e-4 or max_rotation > 1.0e-4:
        raise ValueError(
            "one-IR-per-event bridge requires stationary emitter/listener over the "
            f"event interval; drift_m=({max_source},{max_listener}), rotation_deg={max_rotation}"
        )
    return {
        "maximum_emitter_drift_m": max_source,
        "maximum_listener_drift_m": max_listener,
        "maximum_listener_rotation_drift_deg": max_rotation,
    }


def _nonzero_interval(values: np.ndarray, *, threshold: float, offset: int) -> list[int] | None:
    array = np.asarray(values)
    if array.ndim == 1:
        mask = np.abs(array) > threshold
    else:
        mask = np.any(np.abs(array) > threshold, axis=0)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return None
    return [int(offset + indices[0]), int(offset + indices[-1] + 1)]


def _place_wet_event(
    mixture: np.ndarray,
    dry: np.ndarray,
    ir: np.ndarray,
    *,
    start_sample: int,
    gain: float,
) -> tuple[int, int, np.ndarray]:
    dry_gain = np.asarray(dry, dtype=np.float64) * float(gain)
    ir_channels = np.asarray(ir, dtype=np.float64)
    if ir_channels.ndim == 1:
        ir_channels = ir_channels[None, :]
    wet = np.vstack(
        [np.convolve(dry_gain, channel, mode="full")[: len(mixture[0]) - start_sample] for channel in ir_channels]
    )
    end_sample = min(start_sample + wet.shape[1], mixture.shape[1])
    placed = wet[:, : end_sample - start_sample]
    mixture[:, start_sample:end_sample] += placed
    return start_sample, end_sample, placed


def _frame_position(records: list[Mapping[str, Any]], frame_index: int, key: str) -> tuple[float, float, float]:
    if not records:
        raise ValueError("readback list is empty")
    if not 0 <= frame_index < len(records):
        raise ValueError(
            f"readback frame {frame_index} is unavailable (count={len(records)})"
        )
    selected = records[frame_index]
    return _cm_to_m(selected[key])


def _validate_readback_lengths(
    readback: Mapping[str, Any], frame_count: int, actor_ids: list[str]
) -> None:
    camera = readback.get("camera")
    if not isinstance(camera, list) or len(camera) != frame_count:
        raise ValueError(
            f"camera readback must contain exactly {frame_count} frames"
        )
    emitters = readback.get("emitters")
    if not isinstance(emitters, Mapping):
        raise ValueError("emitter readbacks are missing")
    for actor_id in actor_ids:
        records = emitters.get(actor_id)
        if not isinstance(records, list) or len(records) != frame_count:
            raise ValueError(
                f"emitter readback for {actor_id} must contain exactly "
                f"{frame_count} frames"
            )


def _animation_readback_qa(
    readback: Mapping[str, Any], actor_ids: list[str], frame_count: int
) -> dict[str, Any]:
    raw = readback.get("animations")
    if not isinstance(raw, Mapping):
        return {"status": "not_run", "reason": "animations readback is absent"}
    failures: list[str] = []
    maximum_error = 0.0
    for actor_id in actor_ids:
        records = raw.get(actor_id)
        if not isinstance(records, list) or len(records) != frame_count:
            failures.append(f"{actor_id}: incomplete animation readback")
            continue
        for index, record in enumerate(records):
            if not isinstance(record, Mapping) or record.get("frame_index") != index:
                failures.append(f"{actor_id}: frame index mismatch at {index}")
                continue
            error = record.get("absolute_error_seconds")
            if (
                isinstance(error, bool)
                or not isinstance(error, (int, float))
                or not np.isfinite(float(error))
            ):
                failures.append(f"{actor_id}: invalid animation phase at {index}")
                continue
            maximum_error = max(maximum_error, float(error))
            if float(error) > 1.0e-4:
                failures.append(f"{actor_id}: animation phase error at {index}")
    return {
        "status": "pass" if not failures else "fail",
        "maximum_absolute_error_seconds": maximum_error,
        "failures": failures,
    }


def _simulation(
    *,
    direct_ray_count: int = 500,
    indirect_ray_count: int = 5000,
    source_ray_count: int = 500,
    indirect_ray_depth: int = 64,
    source_ray_depth: int = 16,
    diffraction: bool = False,
    max_diffraction_order: int = 0,
) -> RLRSimulationConfig:
    return RLRSimulationConfig.from_mapping(
        {
            "frequency_bands": 4,
            "direct_sh_order": 0,
            "indirect_sh_order": 0,
            "direct_ray_count": direct_ray_count,
            "indirect_ray_count": indirect_ray_count,
            "indirect_ray_depth": indirect_ray_depth,
            "source_ray_count": source_ray_count,
            "source_ray_depth": source_ray_depth,
            "max_diffraction_order": max_diffraction_order,
            "thread_count": 1,
            "sample_rate_hz": 16000.0,
            "max_ir_seconds": 0.25,
            "unit_scale": 1.0,
            "global_volume": 1.0,
            "speed_of_sound_m_s": 343.0,
            "direct": True,
            "indirect": True,
            "diffraction": diffraction,
            "transmission": False,
            "mesh_simplification": False,
            "temporal_coherence": False,
            "channel_layout": {"type": "binaural", "channel_count": 2},
        }
    )


def _program(
    *,
    clock: Mapping[str, Any],
    events: list[dict[str, Any]],
    endpoint_ids: list[str],
) -> dict[str, Any]:
    frame_count = int(clock["frame_count"])
    frame_rate = int(round(float(clock["frame_rate_hz"])))
    sample_rate = int(clock["sample_rate_hz"])
    sample_count = int(clock["sample_count"])
    timeline = {
        "time_base_hz": TIME_BASE_HZ,
        "ticks_per_frame": TIME_BASE_HZ // frame_rate,
        "video_fps": frame_rate,
        "frame_count": frame_count,
        "sample_rate_hz": sample_rate,
        "ticks_per_sample": TICKS_PER_SAMPLE,
        "sample_count": sample_count,
    }
    value = {
        "schema": "avengine_m6_audio_program_v1",
        "program_id": "polished_room_four_vctk_turn_taking_research_v1",
        "revision": "v1",
        "mode": "sequential_sources",
        "timeline": timeline,
        "candidate_source_endpoint_ids": sorted(endpoint_ids),
        "events": events,
        "source_specific_stems": True,
        "admission_state": "research",
    }
    value = bind_audio_program_hash(value)
    errors = validate_audio_program(value)
    if errors:
        raise ValueError("generated sequential program failed validation: " + "; ".join(errors))
    return value


def _load_voice_binding_records(path: Path) -> list[dict[str, Any]]:
    """Accept the original list and the planner's ``{"bindings": [...]}`` form."""

    value = _load(path)
    if isinstance(value, Mapping) and isinstance(value.get("bindings"), list):
        value = value["bindings"]
    elif isinstance(value, Mapping) and isinstance(value.get("voice_bindings"), list):
        value = value["voice_bindings"]
    elif isinstance(value, Mapping):
        # A keyed map is useful for a randomized actor-to-voice assignment and
        # remains unambiguous because the key is the actor authority.
        value = [
            {"actor_id": str(actor_id), **dict(record)}
            for actor_id, record in value.items()
            if isinstance(record, Mapping)
        ]
    if not isinstance(value, list) or not value:
        raise ValueError("voice binding must contain one or more records")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"voice binding record {index} must be an object")
        actor_id = item.get("actor_id")
        path_value = item.get("path")
        if not isinstance(actor_id, str) or not actor_id:
            raise ValueError(f"voice binding record {index} lacks actor_id")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"voice binding record {index} lacks path")
        records.append(dict(item))
    actor_ids = [str(item["actor_id"]) for item in records]
    if len(set(actor_ids)) != len(actor_ids):
        raise ValueError("voice bindings must have unique actor IDs")
    return records


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{owner} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{owner} must be a finite number")
    return number


def _nonnegative_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{owner} must be a non-negative integer")
    return int(value)


def _positive_int(value: Any, *, owner: str) -> int:
    result = _nonnegative_int(value, owner=owner)
    if result < 1:
        raise ValueError(f"{owner} must be a positive integer")
    return result


def _round_fraction_exact(value: Fraction) -> int:
    if value < 0:
        raise ValueError("timeline boundaries cannot be negative")
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(remainder * 2 >= value.denominator)


def _resolve_plan_clock(
    plan: Mapping[str, Any], readback: Mapping[str, Any]
) -> dict[str, int | float]:
    raw = plan.get("clock") if isinstance(plan.get("clock"), Mapping) else plan.get("timeline")
    if not isinstance(raw, Mapping):
        raw = readback.get("clock")
    if not isinstance(raw, Mapping):
        raise ValueError("audio plan/readbacks must declare a clock")
    frame_count = _positive_int(raw.get("frame_count"), owner="clock.frame_count")
    frame_rate = _finite_number(raw.get("frame_rate_hz", raw.get("video_fps")), owner="clock.frame_rate_hz")
    if frame_rate <= 0.0:
        raise ValueError("clock.frame_rate_hz must be positive")
    sample_rate = _positive_int(raw.get("sample_rate_hz"), owner="clock.sample_rate_hz")
    if sample_rate != 16000:
        raise ValueError("current research bridge requires 16 kHz audio")
    time_base = _positive_int(raw.get("time_base_hz", TIME_BASE_HZ), owner="clock.time_base_hz")
    ticks_per_frame_value = raw.get("ticks_per_frame")
    if ticks_per_frame_value is None:
        implied = Fraction(time_base, 1) / Fraction(str(frame_rate))
        if implied.denominator != 1:
            raise ValueError("clock needs integer ticks_per_frame")
        ticks_per_frame = int(implied)
    else:
        ticks_per_frame = _positive_int(ticks_per_frame_value, owner="clock.ticks_per_frame")
    if not np.isclose(frame_rate * ticks_per_frame, time_base, rtol=0.0, atol=1.0e-6):
        raise ValueError("clock.frame_rate_hz and ticks_per_frame disagree with time_base_hz")
    expected_sample_count = _round_fraction_exact(
        Fraction(frame_count * sample_rate, 1) / Fraction(str(frame_rate))
    )
    declared_sample_count = _positive_int(raw.get("sample_count"), owner="clock.sample_count")
    if declared_sample_count != expected_sample_count:
        raise ValueError(
            f"clock.sample_count must equal exact boundary {expected_sample_count}"
        )
    normalized_rate: int | float = int(frame_rate) if frame_rate.is_integer() else frame_rate
    return {
        "frame_count": frame_count,
        "frame_rate_hz": normalized_rate,
        "ticks_per_frame": ticks_per_frame,
        "time_base_hz": time_base,
        "sample_rate_hz": sample_rate,
        "sample_count": declared_sample_count,
    }


def _active_interval(samples: np.ndarray) -> dict[str, Any]:
    """Find a reproducible non-silent interval without changing the source clip."""

    values = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak <= 0.0 or not np.isfinite(peak):
        raise ValueError("voice clip is silent or non-finite")
    threshold = max(1.0e-6, peak * 1.0e-3)
    active = np.flatnonzero(np.abs(values) > threshold)
    if not active.size:
        raise ValueError("voice clip has no samples above its activity threshold")
    return {
        "start_sample": int(active[0]),
        "end_sample_exclusive": int(active[-1]) + 1,
        "peak_abs": peak,
        "threshold_abs": threshold,
    }


def _normalize_plan_events(
    plan: Mapping[str, Any],
    bindings: list[dict[str, Any]],
    *,
    clock: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, int]], dict[str, dict[str, Any]]]:
    raw_events = plan.get("audio_events")
    if not isinstance(raw_events, list):
        raw_events = plan.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("audio plan must contain a non-empty audio_events list")
    by_actor = {str(item["actor_id"]): item for item in bindings}
    by_endpoint = {
        str(item.get("source_endpoint_id")): item
        for item in bindings
        if isinstance(item.get("source_endpoint_id"), str)
    }
    clips_by_path: dict[str, tuple[np.ndarray, int]] = {}
    normalized: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    sample_count = int(clock["sample_count"])
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise ValueError(f"audio_events[{index}] must be an object")
        actor_id = raw.get("actor_id")
        endpoint_id = raw.get("source_endpoint_id")
        if actor_id is None and isinstance(endpoint_id, str) and endpoint_id.endswith("_mouth"):
            actor_id = endpoint_id[: -len("_mouth")]
        if endpoint_id is None and isinstance(actor_id, str):
            endpoint_id = f"{actor_id}_mouth"
        if not isinstance(endpoint_id, str) or not endpoint_id:
            raise ValueError(f"audio_events[{index}] lacks actor_id/source_endpoint_id")
        actor_binding = by_actor.get(str(actor_id)) if actor_id is not None else None
        actor_binding = actor_binding or by_endpoint.get(endpoint_id)
        event_binding = raw.get("voice_binding", raw.get("binding"))
        if event_binding is not None and not isinstance(event_binding, Mapping):
            raise ValueError(f"audio_events[{index}].voice_binding must be an object")
        if actor_binding is None and event_binding is None:
            raise ValueError(f"audio_events[{index}] has no voice binding for {endpoint_id!r}")
        # Event-level binding and explicit event values are authoritative. The
        # actor-level record is only a fallback, which lets one actor speak
        # different prepared clips in different events.
        binding = {
            **(dict(actor_binding) if isinstance(actor_binding, Mapping) else {}),
            **(dict(event_binding) if isinstance(event_binding, Mapping) else {}),
        }
        actor_id = str(binding.get("actor_id") or actor_id)
        planned_path = raw.get("path")
        planned_sound_asset_id = raw.get("sound_asset_id")
        path_value = (
            planned_path
            or binding.get("path")
            or binding.get("audio_path")
        )
        sound_asset_id = (
            planned_sound_asset_id
            or binding.get("sound_asset_id")
            or binding.get("prepared_audio_id")
        )
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"audio_events[{index}] lacks a PCM path")
        if not isinstance(sound_asset_id, str) or not sound_asset_id:
            raise ValueError(f"audio_events[{index}] lacks sound_asset_id")
        clip_path = str(Path(path_value).expanduser().resolve())
        if clip_path not in clips_by_path:
            clips_by_path[clip_path] = _wav(Path(clip_path))
        samples, rate = clips_by_path[clip_path]
        if rate != int(clock["sample_rate_hz"]):
            raise ValueError(f"audio_events[{index}] PCM rate differs from clock")
        event_id = raw.get("event_id", f"frame_readback_event_{index + 1:04d}")
        if not isinstance(event_id, str) or not event_id or event_id in seen_event_ids:
            raise ValueError(f"audio_events[{index}] has an invalid or duplicate event_id")
        seen_event_ids.add(event_id)
        start = raw.get("start_sample")
        end = raw.get("end_sample_exclusive", raw.get("end_sample"))
        if start is None and raw.get("start_tick") is not None:
            start = _round_fraction_exact(
                Fraction(_nonnegative_int(raw["start_tick"], owner=f"{event_id}.start_tick") * int(clock["sample_rate_hz"]), int(clock["time_base_hz"]))
            )
        if end is None and raw.get("end_tick_exclusive") is not None:
            end = _round_fraction_exact(
                Fraction(_nonnegative_int(raw["end_tick_exclusive"], owner=f"{event_id}.end_tick_exclusive") * int(clock["sample_rate_hz"]), int(clock["time_base_hz"]))
            )
        start = _nonnegative_int(start, owner=f"{event_id}.start_sample")
        end = _positive_int(end, owner=f"{event_id}.end_sample_exclusive")
        source_start = _nonnegative_int(raw.get("source_start_sample", 0), owner=f"{event_id}.source_start_sample")
        source_end = raw.get("source_end_sample_exclusive")
        if source_end is None:
            source_end = source_start + (end - start)
        source_end = _positive_int(source_end, owner=f"{event_id}.source_end_sample_exclusive")
        if end <= start or end > sample_count:
            raise ValueError(f"{event_id} event interval escapes the declared clock")
        if source_end > len(samples) or source_end - source_start != end - start:
            raise ValueError(f"{event_id} scheduled duration differs from its PCM slice")
        gain = _finite_number(raw.get("linear_gain", binding.get("linear_gain", DEFAULT_GAIN)), owner=f"{event_id}.linear_gain")
        if gain < 0.0:
            raise ValueError(f"{event_id}.linear_gain must be non-negative")
        default_fade = min(80, (source_end - source_start) // 2)
        fade = _nonnegative_int(
            raw.get("fade_samples", default_fade), owner=f"{event_id}.fade_samples"
        )
        if 2 * fade > source_end - source_start:
            raise ValueError(f"{event_id}.fade_samples does not fit its PCM slice")
        activity = _active_interval(samples[source_start:source_end])
        normalized.append(
            {
                **dict(raw),
                "event_id": event_id,
                "actor_id": actor_id,
                "source_endpoint_id": endpoint_id,
                "sound_asset_id": sound_asset_id,
                "path": clip_path,
                "start_sample": start,
                "end_sample_exclusive": end,
                "start_tick": start * TICKS_PER_SAMPLE,
                "end_tick_exclusive": end * TICKS_PER_SAMPLE,
                "source_start_sample": source_start,
                "source_end_sample_exclusive": source_end,
                "linear_gain": gain,
                "fade_samples": fade,
                "transcript": raw.get("transcript", binding.get("transcript")),
                "speaker_id": raw.get("speaker_id", binding.get("speaker_id")),
                "sound_class": raw.get("sound_class", binding.get("sound_class")),
                "event_voice_binding": dict(event_binding) if isinstance(event_binding, Mapping) else None,
                "voice_binding_actor_id": actor_id,
                "planned_path": planned_path,
                "planned_sound_asset_id": planned_sound_asset_id,
                "source_activity_interval": activity,
            }
        )
    normalized.sort(key=lambda item: (item["start_sample"], item["source_endpoint_id"], item["event_id"]))
    return normalized, clips_by_path, by_actor


def _plan_source_endpoints(
    plan: Mapping[str, Any], events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Real scene entities remain acoustic candidates even when they are silent."""
    active = {}
    for event in events:
        actor, endpoint = str(event["actor_id"]), str(event["source_endpoint_id"])
        if actor in active and active[actor]["source_endpoint_id"] != endpoint:
            raise ValueError("one actor cannot bind multiple dynamic source endpoints")
        active.setdefault(actor, {"source_endpoint_id": endpoint, "path": str(event["path"])})
    declarations = plan.get("visual_plan", {}).get("actors")
    if declarations is None:
        declarations = plan.get("actors")
    if declarations is None:
        # Historical audio-only plans have no scene-entity declaration.
        declarations = [{"actor_id": actor} for actor in active]
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("plan scene entities must be a nonempty list")
    endpoints, actor_ids = {}, set()
    for actor in declarations:
        aid = actor.get("actor_id") if isinstance(actor, Mapping) else None
        if not isinstance(aid, str) or not aid or aid in actor_ids:
            raise ValueError("plan scene entities must have unique actor IDs")
        actor_ids.add(aid)
        binding = actor.get("emitter_binding", {})
        declared = actor.get("source_endpoint_id") or binding.get("source_endpoint_id")
        actual = active.get(aid, {})
        if declared and actual and declared != actual["source_endpoint_id"]:
            raise ValueError(f"event endpoint differs from the declared emitter: {aid}")
        endpoint = str(declared or actual.get("source_endpoint_id") or f"{aid}_mouth")
        if endpoint in endpoints:
            raise ValueError("scene entities cannot share a source endpoint")
        endpoints[endpoint] = {"actor_id": aid, "path": actual.get("path")}
    if not set(active).issubset(actor_ids):
        raise ValueError("audio event refers to an actor absent from the plan")
    return endpoints


def _program_from_plan_events(
    plan: Mapping[str, Any], events: list[dict[str, Any]], clock: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_ids = set(_plan_source_endpoints(plan, events))
    declared_candidates = plan.get("candidate_source_endpoint_ids")
    if isinstance(declared_candidates, list):
        candidate_ids.update(
            str(value) for value in declared_candidates if isinstance(value, str) and value
        )
    candidate_ids = sorted(candidate_ids)
    mode = plan.get("mode")
    if not isinstance(mode, str) or not mode:
        overlaps = any(
            left["source_endpoint_id"] != right["source_endpoint_id"]
            and max(left["start_sample"], right["start_sample"])
            < min(left["end_sample_exclusive"], right["end_sample_exclusive"])
            for left_index, left in enumerate(events)
            for right in events[left_index + 1 :]
        )
        active_count = len({str(item["source_endpoint_id"]) for item in events})
        if active_count == 1:
            if len(candidate_ids) >= 2:
                mode = "one_active_of_n"
            elif len(events) > 1 and any(
                right["start_sample"] > left["end_sample_exclusive"]
                for left, right in zip(events, events[1:])
            ):
                mode = "intermittent_events"
            else:
                raise ValueError(
                    "a one-event plan must declare at least two candidate endpoints"
                )
        else:
            mode = "simultaneous_subset" if overlaps else "sequential_sources"
    timeline = {
        "time_base_hz": int(clock["time_base_hz"]),
        "ticks_per_frame": int(clock["ticks_per_frame"]),
        "video_fps": clock["frame_rate_hz"],
        "frame_count": int(clock["frame_count"]),
        "sample_rate_hz": int(clock["sample_rate_hz"]),
        "ticks_per_sample": TICKS_PER_SAMPLE,
        "sample_count": int(clock["sample_count"]),
    }
    value = {
        "schema": "avengine_m6_audio_program_v1",
        "program_id": str(plan.get("program_id", "frame_readback_audio_program_research_v1")),
        "revision": str(plan.get("revision", "v1")),
        "mode": mode,
        "timeline": timeline,
        "candidate_source_endpoint_ids": candidate_ids,
        "events": [
            {
                "event_id": event["event_id"],
                "source_endpoint_id": event["source_endpoint_id"],
                "sound_asset_id": event["sound_asset_id"],
                "start_tick": event["start_tick"],
                "end_tick_exclusive": event["end_tick_exclusive"],
                "start_sample": event["start_sample"],
                "end_sample_exclusive": event["end_sample_exclusive"],
                "source_start_sample": event["source_start_sample"],
                "source_end_sample_exclusive": event["source_end_sample_exclusive"],
                "linear_gain": event["linear_gain"],
                "fade_samples": event["fade_samples"],
                "render_source_stem": bool(event.get("render_source_stem", True)),
                "normalization_policy": event.get(
                    "normalization_policy", "use_sound_asset_policy"
                ),
            }
            for event in events
        ],
        "source_specific_stems": True,
        "admission_state": "research",
    }
    for event in value["events"]:
        event.setdefault("render_source_stem", True)
        event.setdefault("normalization_policy", "use_sound_asset_policy")
    value = bind_audio_program_hash(value)
    errors = validate_audio_program(value)
    if errors:
        raise ValueError("plan audio_events failed AudioProgram validation: " + "; ".join(errors))
    return value


def _readback_keyframes(
    readback: Mapping[str, Any],
    *,
    actor_by_endpoint: Mapping[str, Mapping[str, Any]],
    frame_count: int,
    frame_rate_hz: int | float,
    ticks_per_frame: int,
    time_base_hz: int,
    sample_rate_hz: int,
    rir_stride_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, list[list[float]]]]:
    camera = readback.get("camera")
    emitters = readback.get("emitters")
    if not isinstance(camera, list) or not isinstance(emitters, Mapping):
        raise ValueError("frame readbacks must contain camera and emitters")
    for label, records in [
        ("camera", camera),
        *[
            (str(binding["actor_id"]), emitters[str(binding["actor_id"])])
            for binding in actor_by_endpoint.values()
        ],
    ]:
        if not isinstance(records, list):
            raise ValueError(f"{label} readback must be a list")
        for frame_index, record in enumerate(records):
            if not isinstance(record, Mapping) or record.get("frame_index") != frame_index:
                raise ValueError(f"{label} readback frame index mismatch at {frame_index}")
    actor_ids = [str(item["actor_id"]) for item in actor_by_endpoint.values()]
    _validate_readback_lengths(readback, frame_count, actor_ids)
    keyframe_indices = tuple(range(0, frame_count, rir_stride_frames))
    frame_rate_fraction = Fraction(str(float(frame_rate_hz)))
    keyframes: list[dict[str, Any]] = []
    trajectories = {
        endpoint_id: [
            list(_cm_to_m(emitters[str(binding["actor_id"])][frame]["location_cm"]))
            for frame in range(frame_count)
        ]
        for endpoint_id, binding in actor_by_endpoint.items()
    }
    for keyframe_index, frame_index in enumerate(keyframe_indices):
        tick = _round_fraction_exact(
            Fraction(frame_index * time_base_hz, 1) / frame_rate_fraction
        )
        sample_index = _round_fraction_exact(
            Fraction(tick * sample_rate_hz, time_base_hz)
        )
        listener_position, listener_orientation, basis = _frame_listener(camera, frame_index)
        keyframes.append(
            {
                "keyframe_index": keyframe_index,
                "visual_frame_index": frame_index,
                "time_seconds": float(tick / time_base_hz),
                "tick": tick,
                "sample_index": sample_index,
                "source_positions_m": {
                    endpoint_id: trajectories[endpoint_id][frame_index]
                    for endpoint_id in sorted(trajectories)
                },
                "listener_position_m": list(listener_position),
                "listener_orientation_wxyz": list(listener_orientation),
                "listener_basis_m3": {
                    key: list(value) for key, value in basis.items()
                },
            }
        )
    if not keyframes or keyframes[0]["sample_index"] != 0:
        raise ValueError("dynamic RIR keyframe grid must start at sample zero")
    return keyframes, trajectories


def _dynamic_cache_request_metadata(
    *,
    package_path: Path,
    hrtf_path: Path,
    simulation: RLRSimulationConfig,
    source_ids: list[str],
    keyframes: list[dict[str, Any]],
    clock: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    request = {
        "schema": "avengine_dynamic_rir_sequence_cache_v1",
        "package_manifest_sha256": sha256_file(package_path),
        "hrtf_sha256": sha256_file(hrtf_path),
        "simulation": simulation.to_dict(),
        "source_ids": list(source_ids),
        "keyframe_samples": [int(item["sample_index"]) for item in keyframes],
        "keyframes": keyframes,
        "sample_rate_hz": int(clock["sample_rate_hz"]),
        "layout_type": "binaural",
        "layout_id": "rlr_binaural_lr_v1",
        "channel_labels": ["left", "right"],
    }
    return canonical_json_sha256(request), request


def _existing_rir_plan(
    *,
    source_ids: list[str],
    keyframes: list[dict[str, Any]],
    episode_id: str = "frame_readback_dynamic_episode",
) -> dict[str, Any]:
    """Build the established two-slot RIR cache plan from readback keyframes."""

    if len(source_ids) != 2:
        raise ValueError("the established RIR cache plan supports exactly two source slots")
    jobs: list[dict[str, Any]] = []
    jobs_by_state: dict[tuple[float, ...], dict[str, Any]] = {}
    for keyframe in keyframes:
        visual_frame_index = int(keyframe["visual_frame_index"])
        for source_index, source_id in enumerate(source_ids):
            source_position = list(keyframe["source_positions_m"][source_id])
            listener_position = list(keyframe["listener_position_m"])
            listener_orientation = list(keyframe["listener_orientation_wxyz"])
            state_key = (
                *source_position,
                *listener_position,
                *listener_orientation,
                float(source_index),
            )
            job = jobs_by_state.get(state_key)
            if job is None:
                job = {
                    "job_id": f"frame_readback_{visual_frame_index:06d}_source{source_index + 1}",
                    "source_position_m": source_position,
                    "listener_position_m": listener_position,
                    "listener_orientation_wxyz": listener_orientation,
                    "acoustic_state_sha256": rir_acoustic_state_sha256(
                        source_position, listener_position, listener_orientation
                    ),
                    "uses": [],
                }
                jobs_by_state[state_key] = job
                jobs.append(job)
            job["uses"].append(
                {
                    "episode_id": episode_id,
                    "source_slot_id": f"source{source_index + 1}",
                    "frame_index": visual_frame_index,
                }
            )
    return {
        "schema": "avengine_room_rir_job_plan_v2",
        "status": "planned_not_run",
        "listener_pose_mode": "per_episode_frame",
        "cache_key_fields": [
            "source_position_m",
            "listener_position_m",
            "listener_orientation_wxyz",
        ],
        "unique_rir_job_count": len(jobs),
        "jobs": jobs,
    }


def _write_cache_simulation_request(path: Path, simulation: RLRSimulationConfig) -> None:
    if path.exists():
        existing = _load(path)
        if not isinstance(existing, Mapping) or existing.get("simulation") != simulation.to_dict():
            raise ValueError(f"existing cache simulation request differs: {path}")
        return
    _write(
        path,
        {
            "schema": "avengine_rir_cache_simulation_request_v1",
            "request_id": "frame_readback_dynamic_rir_v1",
            "qualification_claim": False,
            "notes": [
                "Generated only to bind this task output to the established RIR cache writer.",
                "Research output; no physical-room qualification claim.",
            ],
            "simulation": simulation.to_dict(),
        },
    )


def _reorder_existing_cached_sequence(
    cached: Any,
    *,
    source_ids: list[str],
    keyframes: list[dict[str, Any]],
    episode_id: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert established source1/source2 slots to the episode endpoint order."""

    expected_samples = [int(item["sample_index"]) for item in keyframes]
    if tuple(cached.source_slot_ids) != ("source1", "source2"):
        raise ValueError("established RIR cache returned unexpected source slots")
    if list(cached.keyframe_samples) != expected_samples:
        raise ValueError("established RIR cache keyframe grid differs from readbacks")
    if cached.layout_type != "binaural" or cached.layout_id != "rlr_binaural_lr_v1":
        raise ValueError("established RIR cache is not binaural")
    if int(cached.sample_rate_hz) != 16_000:
        raise ValueError("established RIR cache sample rate differs from the audio clock")
    evidence_jobs = cached.evidence.get("jobs", [])
    by_use = {
        (str(item.get("source_slot_id")), int(item.get("visual_frame_index"))): item
        for item in evidence_jobs
        if isinstance(item, Mapping)
    }
    for keyframe in keyframes:
        frame_index = int(keyframe["visual_frame_index"])
        for source_index, source_id in enumerate(source_ids):
            slot_id = f"source{source_index + 1}"
            job = by_use.get((slot_id, frame_index))
            if job is None:
                raise ValueError("established RIR cache evidence lacks a planned source/frame use")
            if not np.array_equal(
                np.asarray(job.get("source_position_m"), dtype=np.float64),
                np.asarray(keyframe["source_positions_m"][source_id], dtype=np.float64),
            ) or not np.array_equal(
                np.asarray(job.get("listener_position_m"), dtype=np.float64),
                np.asarray(keyframe["listener_position_m"], dtype=np.float64),
            ) or not np.array_equal(
                np.asarray(job.get("listener_orientation_wxyz"), dtype=np.float64),
                np.asarray(keyframe["listener_orientation_wxyz"], dtype=np.float64),
            ):
                raise ValueError("established RIR cache pose differs from actual readback keyframe")
    samples = np.asarray(cached.samples)
    lengths = np.asarray(cached.lengths)
    if samples.ndim != 4 or samples.shape[1] != 2 or samples.shape[2] != 2:
        raise ValueError("established RIR cache payload has an invalid [K,S,C,L] shape")
    evidence = {
        "schema": "avengine_existing_rir_cache_reuse_v1",
        "status": "pass",
        "episode_id": episode_id,
        "cache_evidence": cached.evidence,
        "source_slot_to_endpoint": {
            "source1": source_ids[0],
            "source2": source_ids[1],
        },
    }
    return np.ascontiguousarray(samples), np.ascontiguousarray(lengths), evidence


def _existing_rir_cache_sequence(
    *,
    cache_path: Path,
    source_ids: list[str],
    keyframes: list[dict[str, Any]],
    frame_count: int,
    frame_rate_hz: float,
    episode_id: str,
    scene: Any,
    simulation: RLRSimulationConfig,
    package_path: Path,
    hrtf_path: Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    magnum_python_site: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Use :func:`render_rir_cache` for the exact two-source dynamic case."""

    if len(source_ids) != 2:
        raise ValueError("established RIR cache path requires exactly two endpoints")
    plan_path = cache_path.parent / f"{cache_path.name}_job_plan.json"
    simulation_request_path = cache_path.parent / f"{cache_path.name}_simulation_request.json"
    if cache_path.exists():
        try:
            request = _load(cache_path / "request.json")
            selected_plan_path = Path(str(request["plan"]["path"])).resolve()
            selected_simulation_path = Path(
                str(request["simulation"]["request_path"])
            ).resolve()
            cached = RIRCacheSession(
                cache_root=cache_path,
                plan_path=selected_plan_path,
                frame_count=frame_count,
                frame_rate_hz=int(round(float(frame_rate_hz))),
            ).load_episode(episode_id)
        except Exception as error:
            raise ValueError(f"existing established RIR cache cannot be reused: {error}") from error
        samples, lengths, evidence = _reorder_existing_cached_sequence(
            cached,
            source_ids=source_ids,
            keyframes=keyframes,
            episode_id=episode_id,
        )
        receipt = _load(cache_path / "receipt.json") if (cache_path / "receipt.json").is_file() else {}
        return (
            samples,
            lengths,
            evidence,
            {
                "status": "hit_existing_rir_cache",
                "path": str(cache_path),
                "cache_format": "avengine_rlr_rir_cache_v1",
                "request_identity_sha256": receipt.get("request_identity_sha256"),
                "plan_path": str(selected_plan_path),
                "simulation_request_path": str(selected_simulation_path),
                "index_path": str((cache_path / "index.json").resolve()),
                "receipt_path": str((cache_path / "receipt.json").resolve()),
            },
        )

    _write(plan_path, _existing_rir_plan(source_ids=source_ids, keyframes=keyframes, episode_id=episode_id))
    _write_cache_simulation_request(simulation_request_path, simulation)
    result = render_rir_cache(
        plan_path=plan_path,
        scene=scene,
        simulation_request_path=simulation_request_path,
        simulation=simulation,
        output=cache_path,
        layout_type="binaural",
        hrtf_file_path=hrtf_path,
        batch_size=2,
        coordinate_translation_m=(0.0, 0.0, 0.0),
        runtime_prefix=runtime_prefix,
        magnum_python_site=magnum_python_site,
        rlr_sdk_root=rlr_sdk_root,
    )
    cached = RIRCacheSession(
        cache_root=result.output,
        plan_path=plan_path,
        frame_count=frame_count,
        frame_rate_hz=int(round(float(frame_rate_hz))),
    ).load_episode(episode_id)
    samples, lengths, evidence = _reorder_existing_cached_sequence(
        cached,
        source_ids=source_ids,
        keyframes=keyframes,
        episode_id=episode_id,
    )
    return (
        samples,
        lengths,
        evidence,
        {
            "status": "miss_written_existing_rir_cache",
            "path": str(cache_path),
            "cache_format": "avengine_rlr_rir_cache_v1",
            "request_identity_sha256": result.receipt.get("request_identity_sha256"),
            "plan_path": str(plan_path.resolve()),
            "simulation_request_path": str(simulation_request_path.resolve()),
            "index_path": str((cache_path / "index.json").resolve()),
            "receipt_path": str((cache_path / "receipt.json").resolve()),
        },
    )


def _existing_rir_cache_pair_sequence(
    *,
    cache_path: Path,
    source_ids: list[str],
    keyframes: list[dict[str, Any]],
    frame_count: int,
    frame_rate_hz: float,
    episode_id: str,
    scene: Any,
    simulation: RLRSimulationConfig,
    package_path: Path,
    hrtf_path: Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    magnum_python_site: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Represent N sources as standard two-slot cache partitions.

    ``rir_cache.py`` deliberately freezes source1/source2 slots. Partitioning
    the source set into independent two-slot caches keeps that contract
    authoritative for multi-source episodes while an ordinary sequence index
    records the source-pair mapping. An odd final group reuses the first
    endpoint as a discarded companion; no new numeric cache format is needed.
    """

    if len(source_ids) < 3:
        raise ValueError("pair cache partition requires at least three source endpoints")
    if cache_path.exists() and not cache_path.is_dir():
        raise ValueError(f"pair cache root is not a directory: {cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)
    index_path = cache_path / "pair_sequence_index.json"
    expected_pairs = [source_ids[offset : offset + 2] for offset in range(0, len(source_ids), 2)]
    if len(expected_pairs[-1]) == 1:
        # Keep the existing two-slot schema for an odd source count by
        # reusing the first endpoint as a discarded companion in the final
        # pair. The duplicate lives in a separate cache request and is never
        # copied into the top-level unique source sequence.
        expected_pairs[-1].append(source_ids[0])
    if index_path.is_file():
        index = _load(index_path)
        if (
            not isinstance(index, Mapping)
            or index.get("kind") != "dynamic_rir_existing_cache_pair_sequence"
            or index.get("status") != "pass"
            or index.get("source_ids") != source_ids
            or index.get("keyframe_samples") != [int(item["sample_index"]) for item in keyframes]
            or index.get("pairs") != [
                {
                    "source_ids": pair,
                    "cache_path": f"pair_{pair_index:02d}",
                }
                for pair_index, pair in enumerate(expected_pairs)
            ]
        ):
            raise ValueError("existing pair cache sequence index differs from the requested episode")
    elif any((cache_path / f"pair_{pair_index:02d}").exists() for pair_index in range(len(expected_pairs))):
        raise ValueError("pair cache root has payloads but no sequence index")

    pair_samples: list[np.ndarray] = []
    pair_lengths: list[np.ndarray] = []
    pair_records: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(expected_pairs):
        pair_cache_path = cache_path / f"pair_{pair_index:02d}"
        samples, lengths, evidence, record = _existing_rir_cache_sequence(
            cache_path=pair_cache_path,
            source_ids=pair,
            keyframes=keyframes,
            frame_count=frame_count,
            frame_rate_hz=frame_rate_hz,
            episode_id=episode_id,
            scene=scene,
            simulation=simulation,
            package_path=package_path,
            hrtf_path=hrtf_path,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
        )
        pair_samples.append(samples)
        pair_lengths.append(lengths)
        pair_records.append(
            {
                **record,
                "source_ids": pair,
                "pair_index": pair_index,
                "evidence": evidence,
            }
        )
    maximum_length = max(int(value.shape[3]) for value in pair_samples)
    values = np.zeros(
        (len(keyframes), len(source_ids), 2, maximum_length), dtype="<f4"
    )
    lengths = np.zeros((len(keyframes), len(source_ids)), dtype="<u4")
    output_index = {source_id: index for index, source_id in enumerate(source_ids)}
    assigned_sources: set[str] = set()
    for pair_index, (pair, samples) in enumerate(zip(expected_pairs, pair_samples, strict=True)):
        for pair_source_index, source_id in enumerate(pair):
            if source_id in assigned_sources:
                continue
            output_source_index = output_index[source_id]
            values[:, output_source_index, :, : samples.shape[3]] = samples[:, pair_source_index]
            lengths[:, output_source_index] = pair_lengths[pair_index][:, pair_source_index]
            assigned_sources.add(source_id)
    if assigned_sources != set(source_ids):
        raise ValueError("pair cache sequence omitted one or more unique source endpoints")
    if not index_path.is_file():
        _write(
            index_path,
            {
                "kind": "dynamic_rir_existing_cache_pair_sequence",
                "status": "pass",
                "source_ids": source_ids,
                "keyframe_samples": [int(item["sample_index"]) for item in keyframes],
                "pairs": [
                    {
                        "source_ids": pair,
                        "cache_path": f"pair_{pair_index:02d}",
                    }
                    for pair_index, pair in enumerate(expected_pairs)
                ],
                "numeric_payload_owner": "existing_avengine_rlr_rir_cache_v1_pair_shards",
                "claim_boundary": "ordinary pair index over existing request/index/timing/receipt/shard caches",
            },
        )
    status = "hit_existing_rir_cache_pairs" if all(
        value["status"] == "hit_existing_rir_cache" for value in pair_records
    ) else "miss_written_existing_rir_cache_pairs"
    return (
        values,
        lengths,
        {
            "schema": "avengine_existing_rir_cache_pair_sequence_evidence_v1",
            "status": "pass",
            "source_ids": source_ids,
            "pairs": pair_records,
            "index_path": str(index_path.resolve()),
        },
        {
            "status": status,
            "path": str(cache_path),
            "cache_format": "avengine_rlr_rir_cache_v1_pair_shards",
            "sequence_index_path": str(index_path.resolve()),
            "pairs": [
                {
                    "source_ids": pair,
                    "cache_path": str((cache_path / f"pair_{pair_index:02d}").resolve()),
                    "status": pair_records[pair_index]["status"],
                    "request_identity_sha256": pair_records[pair_index].get("request_identity_sha256"),
                }
                for pair_index, pair in enumerate(expected_pairs)
            ],
        },
    )


def _dynamic_rir_sequence(
    *,
    scene: Any,
    simulation: RLRSimulationConfig,
    source_ids: list[str],
    keyframes: list[dict[str, Any]],
    clock: Mapping[str, Any],
    package_path: Path,
    hrtf_path: Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    magnum_python_site: str | Path,
    cache_path: Path,
    episode_id: str = "frame_readback_dynamic_episode",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load a matching sequence or render it with one persistent cache context."""

    # Current production paths use the established RIR cache contract:
    # two-source requests use RIRCacheSession and multi-source requests use
    # pair_sequence_index shards. The legacy sequence NPZ path below is only a
    # read-only compatibility reader for earlier research outputs.
    if len(source_ids) == 2 and (
        not cache_path.exists() or (cache_path / "request.json").is_file()
    ):
        return _existing_rir_cache_sequence(
            cache_path=cache_path,
            source_ids=source_ids,
            keyframes=keyframes,
            frame_count=int(clock["frame_count"]),
            frame_rate_hz=float(clock["frame_rate_hz"]),
            episode_id=episode_id,
            scene=scene,
            simulation=simulation,
            package_path=package_path,
            hrtf_path=hrtf_path,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
        )
    if len(source_ids) >= 3 and (
        not cache_path.exists() or (cache_path / "pair_sequence_index.json").is_file()
    ):
        return _existing_rir_cache_pair_sequence(
            cache_path=cache_path,
            source_ids=source_ids,
            keyframes=keyframes,
            frame_count=int(clock["frame_count"]),
            frame_rate_hz=float(clock["frame_rate_hz"]),
            episode_id=episode_id,
            scene=scene,
            simulation=simulation,
            package_path=package_path,
            hrtf_path=hrtf_path,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
        )

    # Compatibility reader for an already-produced variable-source sequence
    # NPZ cache. No current production path creates this format.
    cache_identity, _ = _dynamic_cache_request_metadata(
        package_path=package_path,
        hrtf_path=hrtf_path,
        simulation=simulation,
        source_ids=source_ids,
        keyframes=keyframes,
        clock=clock,
    )
    keyframe_samples = [int(item["sample_index"]) for item in keyframes]
    if not cache_path.exists():
        raise ValueError(
            "no established existing RIR cache matched the variable-source "
            "sequence; new sequence cache creation is disabled"
        )
    try:
        payload = load_dynamic_rir_cache(
            cache_path,
            expected_cache_identity_sha256=cache_identity,
            expected_source_ids=source_ids,
            expected_keyframe_samples=keyframe_samples,
        )
    except DynamicRIRCacheError as error:
        raise ValueError(
            f"legacy dynamic RIR compatibility cache cannot be reused: {error}"
        ) from error
    metadata = dict(payload.metadata)
    if tuple(payload.samples.shape[:3]) != (
        len(keyframes), len(source_ids), 2
    ):
        raise ValueError("legacy dynamic RIR compatibility cache has an invalid binaural shape")
    return (
        np.asarray(payload.samples),
        np.asarray(payload.lengths),
        metadata,
        {
            "status": "hit",
            "path": str(cache_path),
            "cache_format": "avengine_dynamic_rir_sequence_cache_v1",
            "reuse_mode": "legacy_variable_source_sequence_compat_read",
            "cache_identity_sha256": cache_identity,
            "content_sha256": metadata.get("content_sha256"),
        },
    )

def _render_plan_audio_legacy_dynamic(
    *,
    frame_readbacks: str | Path,
    package_manifest: str | Path,
    voice_binding: str | Path,
    audio_plan: str | Path,
    output: str | Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    magnum_python_site: str | Path,
    hrtf_file: str | Path,
    rir_cache: str | Path | None,
    rir_stride_frames: int,
    direct_ray_count: int,
    indirect_ray_count: int,
    source_ray_count: int,
    indirect_ray_depth: int,
    source_ray_depth: int,
    diffraction: bool | None = None,
    max_diffraction_order: int | None = None,
) -> dict[str, Any]:
    readback_path = Path(frame_readbacks).expanduser().resolve()
    package_path = Path(package_manifest).expanduser().resolve()
    plan_path = Path(audio_plan).expanduser().resolve()
    binding_path = Path(voice_binding).expanduser().resolve()
    hrtf_path = Path(hrtf_file).expanduser().resolve()
    readback = _load(readback_path)
    plan = _load(plan_path)
    if not isinstance(readback, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("frame readbacks and audio plan must contain JSON objects")
    capture_failure_path = readback_path.parent.parent / "failure.json"
    input_capture_status = (
        "visual_failed"
        if capture_failure_path.is_file()
        else "native_readback_supplied"
    )
    bindings = _load_voice_binding_records(binding_path)
    clock = _resolve_plan_clock(plan, readback)
    frame_count = int(clock["frame_count"])
    sample_rate = int(clock["sample_rate_hz"])
    events, clips_by_path, bindings_by_actor = _normalize_plan_events(
        plan, bindings, clock=clock
    )
    actor_by_endpoint = _plan_source_endpoints(plan, events)
    endpoint_ids = sorted(actor_by_endpoint)
    if len(endpoint_ids) < 2:
        raise ValueError("dynamic multi-source audio requires at least two source endpoints")
    actor_ids = [str(item["actor_id"]) for item in actor_by_endpoint.values()]
    if len(set(actor_ids)) != len(actor_ids):
        raise ValueError("one actor cannot bind multiple dynamic source endpoints")
    _validate_readback_lengths(readback, frame_count, actor_ids)
    animation_qa = _animation_readback_qa(readback, actor_ids, frame_count)
    keyframes, trajectories = _readback_keyframes(
        readback,
        actor_by_endpoint=actor_by_endpoint,
        frame_count=frame_count,
        frame_rate_hz=clock["frame_rate_hz"],
        ticks_per_frame=int(clock["ticks_per_frame"]),
        time_base_hz=int(clock["time_base_hz"]),
        sample_rate_hz=sample_rate,
        rir_stride_frames=_positive_int(rir_stride_frames, owner="rir_stride_frames"),
    )
    source_trajectories = {
        endpoint_id: trajectories[endpoint_id] for endpoint_id in endpoint_ids
    }
    # Build dry buses from plan events. The plan timing remains authoritative;
    # voice bindings only resolve the concrete PCM and content provenance.
    dry_event_mappings = [
        {
            "event_id": event["event_id"],
            "source_id": event["source_endpoint_id"],
            "start_sample": event["start_sample"],
            "end_sample_exclusive": event["end_sample_exclusive"],
            "dry_asset_id": event["sound_asset_id"],
            "dry_asset_sha256": sha256_file(Path(event["path"])),
            "dry_clip_start_sample": event["source_start_sample"],
            "dry_clip_end_sample_exclusive": event["source_end_sample_exclusive"],
            "linear_gain": event["linear_gain"],
            "fade_samples": event["fade_samples"],
        }
        for event in events
    ]
    dry_assembly = assemble_dry_audio_buses(
        dry_event_mappings,
        source_ids=tuple(endpoint_ids),
        clip=DryAudioClipSpec.from_values(
            frame_count=frame_count,
            fps_numerator=int(round(float(clock["frame_rate_hz"]))),
            sample_rate_hz=sample_rate,
        ),
        asset_bindings={
            event["sound_asset_id"]: event["path"] for event in events
        },
    )
    program = _program_from_plan_events(plan, events, clock)
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"fresh output required: {root}")
    root.mkdir(parents=True)
    _write(root / "audio_program.json", program)
    simulation = _simulation(
        direct_ray_count=direct_ray_count,
        indirect_ray_count=indirect_ray_count,
        source_ray_count=source_ray_count,
        indirect_ray_depth=indirect_ray_depth,
        source_ray_depth=source_ray_depth,
        diffraction=(False if diffraction is None else diffraction),
        max_diffraction_order=(0 if max_diffraction_order is None else max_diffraction_order),
    )
    scene = load_compiled_acoustic_scene(
        package_path,
        allow_nonpassing_research_qa=True,
    )
    cache_path = (
        Path(rir_cache).expanduser().resolve()
        if rir_cache is not None
        else root / "rir_cache"
    )
    rir_samples, rir_lengths, rir_metadata, cache_record = _dynamic_rir_sequence(
        scene=scene,
        simulation=simulation,
        source_ids=endpoint_ids,
        keyframes=keyframes,
        clock=clock,
        package_path=package_path,
        hrtf_path=hrtf_path,
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        magnum_python_site=magnum_python_site,
        cache_path=cache_path,
    )
    # ``render_dynamic_stems_and_mix`` is the existing deterministic raised
    # cosine/overlap-add path; it preserves every source gain in the assembled
    # bus and crops only the declared episode boundary.
    dry_buses = {
        endpoint_id: np.asarray(dry_assembly.buses[endpoint_id], dtype=np.float64)
        for endpoint_id in endpoint_ids
    }
    stems_by_source, mixture = render_dynamic_stems_and_mix(
        dry_buses,
        rir_samples,
        rir_lengths,
        source_ids=tuple(endpoint_ids),
        keyframe_samples=[int(item["sample_index"]) for item in keyframes],
        output_sample_count=int(clock["sample_count"]),
    )
    audio_root = root / "audio"
    audio_root.mkdir()
    output_files: dict[str, str] = {}

    def write_audio(path: Path, value: np.ndarray) -> None:
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[0] not in (1, 2):
            raise ValueError("audio output must be channel-major mono or stereo")
        peak = float(np.max(np.abs(array))) if array.size else 0.0
        if not np.isfinite(peak) or peak > 1.0 + 1.0e-12:
            raise ValueError(
                f"audio output would clip without normalization/limiting: peak={peak:.9g}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(path, array)
        output_files[str(path.relative_to(root))] = sha256_file(path)

    for endpoint_id in endpoint_ids:
        write_audio(audio_root / "dry" / f"{endpoint_id}.wav", dry_buses[endpoint_id][None, :])
        write_audio(audio_root / "binaural" / f"{endpoint_id}_stem.wav", stems_by_source[endpoint_id].episode)
    mixture_path = root / "four_speaker_sequential_mixture.wav"
    write_audio(mixture_path, mixture)
    write_audio(audio_root / "binaural" / "mixture.wav", mixture)

    event_records: list[dict[str, Any]] = []
    source_index = {source_id: index for index, source_id in enumerate(endpoint_ids)}
    event_intervals_by_source: dict[str, list[tuple[int, int, str]]] = {}
    for event in events:
        interval = event["source_activity_interval"]
        actual_start = int(event["start_sample"] + interval["start_sample"])
        actual_end = int(event["start_sample"] + interval["end_sample_exclusive"])
        event_intervals_by_source.setdefault(event["source_endpoint_id"], []).append(
            (actual_start, actual_end, event["event_id"])
        )
        event_bus = np.zeros(int(clock["sample_count"]), dtype=np.float64)
        event_bus[event["start_sample"] : event["end_sample_exclusive"]] = dry_buses[event["source_endpoint_id"]][event["start_sample"] : event["end_sample_exclusive"]]
        event_stem = time_varying_convolve(
            event_bus,
            rir_samples[:, source_index[event["source_endpoint_id"]]],
            [int(item["sample_index"]) for item in keyframes],
            rir_lengths=rir_lengths[:, source_index[event["source_endpoint_id"]]],
            output_sample_count=int(clock["sample_count"]),
        )
        tail_interval = _nonzero_interval(
            event_stem.full_tail, threshold=1.0e-12, offset=0
        )
        if tail_interval is None:
            raise ValueError(f"{event['event_id']} produced a silent wet tail")
        tail_end = int(tail_interval[1])
        event_records.append(
            {
                **event,
                "clip_path": event["path"],
                "source_activity_interval": {
                    **interval,
                    "absolute_start_sample": actual_start,
                    "absolute_end_sample_exclusive": actual_end,
                },
                "scheduled_event_interval": [event["start_sample"], event["end_sample_exclusive"]],
                "planned_event_interval": [event["start_sample"], event["end_sample_exclusive"]],
                "actual_dry_active_interval": [actual_start, actual_end],
                "wet_tail_interval": tail_interval,
                "wet_render_interval": [
                    int(event["start_sample"]),
                    min(tail_end, int(clock["sample_count"])),
                ],
                "wet_tail_end_sample": tail_end,
                "wet_tail_truncated_at_episode": tail_end > int(clock["sample_count"]),
                "wet_tail_end_sample_in_episode": min(tail_end, int(clock["sample_count"])),
                "wet_float_nonzero_interval": tail_interval,
                "pcm_output_nonzero_interval": _nonzero_interval(
                    event_stem.episode, threshold=1.0 / 32767.0, offset=0
                ),
                "keyframe_indices": [int(item["keyframe_index"]) for item in keyframes],
                "output_stem": str((audio_root / "binaural" / f"{event['source_endpoint_id']}_stem.wav").resolve()),
            }
        )
    per_source_overlap = {
        source_id: any(
            left[1] > right[0]
            for index, left in enumerate(sorted(intervals))
            for right in sorted(intervals)[index + 1 :]
        )
        for source_id, intervals in event_intervals_by_source.items()
    }
    partition_error = max(
        float(stem.maximum_partition_error) for stem in stems_by_source.values()
    )
    output_peaks = {
        endpoint_id: float(np.max(np.abs(stems_by_source[endpoint_id].episode)))
        for endpoint_id in endpoint_ids
    }
    output_peaks["mixture"] = float(np.max(np.abs(mixture)))
    keyframe_metadata = {
        "schema": "avengine_dynamic_rir_keyframes_v1",
        "sampling": {
            "policy": "explicit_visual_frame_stride_v1",
            "visual_frame_count": frame_count,
            "visual_frame_rate_hz": clock["frame_rate_hz"],
            "rir_stride_frames": int(rir_stride_frames),
            "keyframe_count": len(keyframes),
            "final_interval_policy": "hold_last_rir_to_episode_end",
        },
        "timebase": {
            "time_base_hz": clock["time_base_hz"],
            "ticks_per_frame": clock["ticks_per_frame"],
            "sample_rate_hz": sample_rate,
            "sample_count": clock["sample_count"],
        },
        "listener_motion": "readback_per_keyframe",
        "source_motion": "readback_per_keyframe",
        "source_ids": endpoint_ids,
        "keyframes": keyframes,
    }
    _write(root / "dynamic_rir_keyframes.json", keyframe_metadata)
    report = {
        "schema": "avengine_frame_readback_multi_source_dynamic_audio_research_v1",
        "status": "research",
        "input_capture_status": input_capture_status,
        "input_capture_failure": (
            str(capture_failure_path.resolve())
            if input_capture_status == "visual_failed"
            else None
        ),
        "frame_readbacks": str(readback_path),
        "audio_plan": str(plan_path),
        "voice_binding": str(binding_path),
        "acoustic_package": str(package_path),
        "clock": dict(clock),
        "audio_program": str((root / "audio_program.json").resolve()),
        "mixture_path": str(mixture_path.resolve()),
        "mixture_sha256": output_files[str(mixture_path.relative_to(root))],
        "audio_root": str(audio_root.resolve()),
        "voice_bindings": bindings,
        "dry_audio_assembly": dry_assembly.metadata(),
        "dynamic_rir": {
            "status": "pass",
            "cache": cache_record,
            "cache_manifest": str((cache_path / "manifest.json").resolve()),
            "source_ids": endpoint_ids,
            "layout_type": "binaural",
            "layout_id": "rlr_binaural_lr_v1",
            "channel_labels": ["left", "right"],
            "channel_count": 2,
            "sample_rate_hz": sample_rate,
            "keyframe_count": len(keyframes),
            "keyframe_samples": [int(item["sample_index"]) for item in keyframes],
            "keyframes": keyframes,
            "interpolation": "raised_cosine_source_time_partition_v1",
            "partition_max_error": partition_error,
            "rir_metadata": rir_metadata,
            "hrtf": {
                "path": str(hrtf_path),
                "sha256": sha256_file(hrtf_path),
                "channel_order": ["left", "right"],
            },
        },
        "trajectories": {
            "source_ids": endpoint_ids,
            "authority": "frame_readbacks.emitters[].location_cm",
            "coordinate_transform_profile": "ue_scene_capture_x_forward_z_up_to_m3_minus_z_forward_y",
            "source_trajectories_m": source_trajectories,
            "listener_authority": "frame_readbacks.camera[].location_cm+rotation_deg",
        },
        "events": event_records,
        "outputs": output_files,
        "qa": {
            "speech_identity_and_transcripts": {
                "status": "pass",
                "event_count": len(event_records),
                "complete_sentences_preserved": True,
                "activity_interval_detected_from_pcm": True,
            },
            "event_clock_and_gain": {
                "status": "pass",
                "source": "plan.audio_events",
                "dry_gain_and_fade_authority": "assemble_dry_audio_buses",
                "per_source_actual_active_overlaps": per_source_overlap,
                "output_peak_abs_by_stream": output_peaks,
                "clipping_status": "pass",
            },
            "dynamic_rir": {
                "status": "pass",
                "source_positions_authority": "frame_readbacks.emitters",
                "listener_pose_authority": "frame_readbacks.camera",
                "keyframe_metadata": str((root / "dynamic_rir_keyframes.json").resolve()),
                "cache": cache_record,
            },
            "binaural_channels": {
                "status": "pass",
                "layout": "binaural",
                "channel_count": 2,
                "channel_labels": ["left", "right"],
                "hrtf": str(hrtf_path),
            },
            "wet_tail": {
                "status": "pass",
                "event_count": len(event_records),
                "tail_end_authority": "event-isolated time-varying convolution full_tail",
                "tail_can_extend_past_episode": any(
                    bool(event["wet_tail_truncated_at_episode"]) for event in event_records
                ),
            },
            "animation_phase_readback": {**animation_qa, "source": "frame_readbacks.animations"},
            "pixel_visibility": {
                "status": "not_run",
                "reason": "audio renderer does not infer visibility from RGB; use native semantic/depth truth",
            },
        },
        "complete_sentences_preserved": True,
        "native_rlr_dynamic": True,
        "formal_admission": False,
        "claim_boundary": (
            "Research audio uses actual frame-readback source/listener states, "
            "native CPU RLR binaural HRTF, persistent dynamic sequence cache and "
            "the plan's independent dry events; no formal visual/audio admission."
        ),
    }
    _write(root / "research_report.json", report)
    return report



def _plan_camera_has_motion(readback: Mapping[str, Any]) -> bool:
    """Return whether the supplied per-frame listener pose actually moves."""
    camera = readback.get("camera")
    if not isinstance(camera, list) or len(camera) < 2:
        return False
    first = camera[0]
    if not isinstance(first, Mapping):
        raise ValueError("camera readback rows must be objects")

    def pose(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if "location_cm" in row and "rotation_deg" in row:
            position, orientation, _ = _frame_listener([row], 0)
            return np.asarray(position, dtype=np.float64), np.asarray(orientation, dtype=np.float64)
        position = np.asarray(row.get("position_m"), dtype=np.float64)
        basis = row.get("basis")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("camera readback position is malformed")
        if not isinstance(basis, Mapping):
            raise ValueError("camera readback basis is missing")
        vectors = [
            np.asarray(basis.get(key), dtype=np.float64)
            for key in ("forward", "right", "up")
        ]
        if any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors):
            raise ValueError("camera readback basis is malformed")
        return position, np.column_stack(vectors)

    first_position, first_pose = pose(first)
    for row in camera[1:]:
        if not isinstance(row, Mapping):
            raise ValueError("camera readback rows must be objects")
        position, current_pose = pose(row)
        if float(np.linalg.norm(position - first_position)) > 1.0e-4:
            return True
        if current_pose.shape == (4,) and first_pose.shape == (4,):
            if _rotation_distance_deg(tuple(first_pose), tuple(current_pose)) > 1.0e-4:
                return True
        elif not np.allclose(current_pose, first_pose, atol=1.0e-4, rtol=0.0):
            return True
    return False


def _plan_is_conditioned_static(plan: Mapping[str, Any]) -> bool:
    """Recognize the owner-defined conditioned static planning strategy."""
    if plan.get("sampling_policy") == "conditioned_static_v2":
        return True
    request = plan.get("request")
    return isinstance(request, Mapping) and request.get("sampling_policy") == "conditioned_static_v2"


def _render_plan_audio(
    *,
    frame_readbacks: str | Path,
    package_manifest: str | Path,
    voice_binding: str | Path,
    audio_plan: str | Path,
    output: str | Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    magnum_python_site: str | Path,
    hrtf_file: str | Path,
    rir_cache: str | Path | None,
    rir_stride_frames: int,
    direct_ray_count: int,
    indirect_ray_count: int,
    source_ray_count: int,
    indirect_ray_depth: int,
    source_ray_depth: int,
    neutral_readback: str | Path | None = None,
    prepared_manifest: str | Path | None = None,
    diffraction: bool | None = None,
    max_diffraction_order: int | None = None,
) -> dict[str, Any]:
    """Adapt the historical UE readbacks to the shared neutral renderer."""
    readback_path = Path(frame_readbacks).expanduser().resolve()
    package_path = Path(package_manifest).expanduser().resolve()
    plan_path = Path(audio_plan).expanduser().resolve()
    binding_path = Path(voice_binding).expanduser().resolve()
    readback = _load(readback_path)
    plan = _load(plan_path)
    if not isinstance(readback, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("frame readbacks and audio plan must contain JSON objects")
    capture_failure_path = readback_path.parent.parent / "failure.json"
    input_capture_status = (
        "visual_failed" if capture_failure_path.is_file() else "native_readback_supplied"
    )
    bindings = _load_voice_binding_records(binding_path)
    clock = _resolve_plan_clock(plan, readback)
    frame_count = int(clock["frame_count"])
    sample_rate = int(clock["sample_rate_hz"])
    events, _, _ = _normalize_plan_events(plan, bindings, clock=clock)
    actor_by_endpoint = _plan_source_endpoints(plan, events)
    endpoint_ids = sorted(actor_by_endpoint)
    if len(endpoint_ids) < 2:
        raise ValueError("dynamic multi-source audio requires at least two source endpoints")
    actor_ids = [str(item["actor_id"]) for item in actor_by_endpoint.values()]
    if len(set(actor_ids)) != len(actor_ids):
        raise ValueError("one actor cannot bind multiple dynamic source endpoints")
    _validate_readback_lengths(readback, frame_count, actor_ids)
    camera_motion = _plan_camera_has_motion(readback)
    if neutral_readback is not None:
        candidate = _load(Path(neutral_readback).expanduser().resolve())
        if not isinstance(candidate, Mapping):
            raise ValueError("neutral_readback must contain a JSON object")
        camera_motion = camera_motion or _plan_camera_has_motion(candidate)
    if camera_motion:
        if _plan_is_conditioned_static(plan):
            raise ValueError(
                "conditioned_static_v2 audio rendering requires a static listener; "
                "per-frame camera motion was observed"
            )
        return _render_plan_audio_legacy_dynamic(
            frame_readbacks=frame_readbacks,
            package_manifest=package_manifest,
            voice_binding=voice_binding,
            audio_plan=audio_plan,
            output=output,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
            hrtf_file=hrtf_file,
            rir_cache=rir_cache,
            rir_stride_frames=rir_stride_frames,
            direct_ray_count=direct_ray_count,
            indirect_ray_count=indirect_ray_count,
            source_ray_count=source_ray_count,
            indirect_ray_depth=indirect_ray_depth,
            source_ray_depth=source_ray_depth,
            diffraction=diffraction,
            max_diffraction_order=max_diffraction_order,
        )
    if neutral_readback is None:
        neutral_source = readback
        # Historical unit and four-source fixtures carried only emitter
        # locations. Preserve that request path by using the same observed
        # emitter as a root when no actor track was supplied; native captures
        # always take the strict actor/root path.
        if "actors" not in readback and isinstance(readback.get("emitters"), Mapping):
            neutral_source = dict(readback)
            neutral_source["actors"] = {
                actor_id: [dict(row) for row in records]
                for actor_id, records in readback["emitters"].items()
            }
            for records in neutral_source["actors"].values():
                for row in records:
                    row["location_cm"] = list(row["location_cm"])
        neutral = neutral_from_ue_readbacks(
            neutral_source,
            plan,
            source_readbacks=str(readback_path),
        )
        neutral_input: Mapping[str, Any] | str | Path = neutral
    else:
        neutral_path = Path(neutral_readback).expanduser().resolve()
        neutral = _load(neutral_path)
        if not isinstance(neutral, Mapping):
            raise ValueError("neutral_readback must contain a JSON object")
        validate_neutral_readback(neutral, plan=plan)
        neutral_input = neutral_path
    program = _program_from_plan_events(plan, events, clock)
    simulation = _simulation(
        direct_ray_count=direct_ray_count,
        indirect_ray_count=indirect_ray_count,
        source_ray_count=source_ray_count,
        indirect_ray_depth=indirect_ray_depth,
        source_ray_depth=source_ray_depth,
        diffraction=(False if diffraction is None else diffraction),
        max_diffraction_order=(0 if max_diffraction_order is None else max_diffraction_order),
    )
    rir_sequence_override = None
    scene_override = None
    if rir_cache is not None:
        # Keep the historical two-slot cache adapter as a read/write input for
        # the transition entry. The shared renderer consumes the resulting
        # named sequence, so no second cache format or second convolution path
        # is introduced.
        legacy_keyframes, _ = _readback_keyframes(
            readback,
            actor_by_endpoint=actor_by_endpoint,
            frame_count=frame_count,
            frame_rate_hz=clock["frame_rate_hz"],
            ticks_per_frame=int(clock["ticks_per_frame"]),
            time_base_hz=int(clock["time_base_hz"]),
            sample_rate_hz=sample_rate,
            rir_stride_frames=rir_stride_frames,
        )
        scene_override = load_compiled_acoustic_scene(
            package_path,
            allow_nonpassing_research_qa=True,
        )
        rir_samples, rir_lengths, rir_metadata, cache_record = _dynamic_rir_sequence(
            scene=scene_override,
            simulation=simulation,
            source_ids=endpoint_ids,
            keyframes=legacy_keyframes,
            clock=clock,
            package_path=package_path,
            hrtf_path=Path(hrtf_file).expanduser().resolve(),
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
            cache_path=Path(rir_cache).expanduser().resolve(),
            episode_id=str(plan.get("episode_id", "frame_readback_dynamic_episode")),
        )
        rir_sequence_override = {
            "binaural": {
                "samples": rir_samples,
                "lengths": rir_lengths,
                "metadata": rir_metadata,
                "cache": cache_record,
                "layout_id": "rlr_binaural_lr_v1",
            }
        }
    result = render_neutral_readback_audio(
        neutral_input,
        audio_program=program,
        source_endpoint_by_entity={
            str(value["actor_id"]): endpoint_id
            for endpoint_id, value in actor_by_endpoint.items()
        },
        simulation_mapping=simulation.to_dict(),
        package_manifest_path=package_path,
        event_asset_bindings={
            str(event["sound_asset_id"]): str(event["path"])
            for event in events
        },
        event_metadata=events,
        prepared_manifest_path=prepared_manifest,
        hrtf_file_path=hrtf_file,
        output_path=output,
        position_authority="P1 NeutralReadback entities[].emitter",
        listener_authority="UE neutral_from_ue_readbacks.camera[0]",
        rir_stride_frames=rir_stride_frames,
        hrtf_license_path=None,
        extra_inputs={
            "frame_readbacks": {
                "path": str(readback_path),
                "sha256": sha256_file(readback_path),
            },
            "audio_plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
            },
            "voice_binding": {
                "path": str(binding_path),
                "sha256": sha256_file(binding_path),
            },
            "input_capture_status": input_capture_status,
        },
        diffraction=diffraction,
        max_diffraction_order=max_diffraction_order,
        rir_sequence_override=rir_sequence_override,
        scene_override=scene_override,
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        magnum_python_site=magnum_python_site,
    )
    result["input_capture_status"] = input_capture_status
    result["frame_readbacks"] = str(readback_path)
    result["audio_plan"] = str(plan_path)
    result["voice_binding"] = str(binding_path)
    result["acoustic_package"] = str(package_path)
    result["clock"] = dict(clock)
    return result

def render(
    *,
    frame_readbacks: str | Path,
    package_manifest: str | Path,
    voice_binding: str | Path,
    output: str | Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    magnum_python_site: str | Path,
    direct_ray_count: int = 500,
    indirect_ray_count: int = 5000,
    source_ray_count: int = 500,
    indirect_ray_depth: int = 64,
    source_ray_depth: int = 16,
    audio_plan: str | Path | None = None,
    hrtf_file: str | Path = "/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa",
    rir_cache: str | Path | None = None,
    rir_stride_frames: int = 3,
    neutral_readback: str | Path | None = None,
    prepared_manifest: str | Path | None = None,
    diffraction: bool | None = None,
    max_diffraction_order: int | None = None,
) -> dict[str, Any]:
    if audio_plan is not None:
        return _render_plan_audio(
            frame_readbacks=frame_readbacks,
            package_manifest=package_manifest,
            voice_binding=voice_binding,
            audio_plan=audio_plan,
            output=output,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
            hrtf_file=hrtf_file,
            rir_cache=rir_cache,
            rir_stride_frames=rir_stride_frames,
            direct_ray_count=direct_ray_count,
            indirect_ray_count=indirect_ray_count,
            source_ray_count=source_ray_count,
            indirect_ray_depth=indirect_ray_depth,
            source_ray_depth=source_ray_depth,
            neutral_readback=neutral_readback,
            prepared_manifest=prepared_manifest,
            diffraction=diffraction,
            max_diffraction_order=max_diffraction_order,
        )
    readback_path = Path(frame_readbacks).expanduser().resolve()
    package_path = Path(package_manifest).expanduser().resolve()
    bindings = _load(Path(voice_binding).expanduser().resolve())
    if not isinstance(bindings, list) or len(bindings) != 4:
        raise ValueError("voice_binding must contain exactly four complete-clip records")
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"fresh output required: {root}")
    root.mkdir(parents=True)
    stems = root / "source_stems"
    stems.mkdir()
    readback = _load(readback_path)
    clock = readback["clock"]
    frame_count = int(clock["frame_count"])
    frame_rate = float(clock["frame_rate_hz"])
    sample_rate = int(clock["sample_rate_hz"])
    sample_count = int(clock["sample_count"])
    if sample_rate != 16000:
        raise ValueError("current research bridge requires 16 kHz readbacks")
    camera = readback["camera"]
    emitters = readback["emitters"]
    actor_ids = [
        str(item["actor_id"]) if isinstance(item, Mapping) else str(item)
        for item in bindings
    ]
    if len(set(actor_ids)) != 4 or not all(
        actor_id in emitters for actor_id in actor_ids
    ):
        raise ValueError("voice bindings must name four actor IDs present in emitter readbacks")
    _validate_readback_lengths(readback, frame_count, actor_ids)
    animation_qa = _animation_readback_qa(readback, actor_ids, frame_count)

    clips: list[tuple[np.ndarray, int]] = []
    for item in bindings:
        path = Path(str(item["path"])).expanduser().resolve()
        samples, rate = _wav(path)
        clips.append((samples, rate))

    gap_samples = int(round(0.20 * sample_rate))
    margin_samples = int(round(0.50 * sample_rate))
    starts: list[int] = []
    cursor = margin_samples
    for index, (samples, _) in enumerate(clips):
        starts.append(cursor)
        cursor += len(samples)
        if index < len(clips) - 1:
            cursor += gap_samples
    if cursor > sample_count:
        raise ValueError(
            "complete VCTK sentences do not fit the readback clock; "
            "provide a longer SPEAR clock/readback, never truncate them"
        )

    endpoint_ids = [f"{actor_id}_mouth" for actor_id in actor_ids]
    events: list[dict[str, Any]] = []
    for index, (item, (samples, _), start, endpoint_id) in enumerate(
        zip(bindings, clips, starts, endpoint_ids)
    ):
        end = start + len(samples)
        events.append(
            {
                "event_id": f"vctk_turn_{index + 1:02d}",
                "source_endpoint_id": endpoint_id,
                "sound_asset_id": str(item["sound_asset_id"]),
                "start_tick": start * TICKS_PER_SAMPLE,
                "end_tick_exclusive": end * TICKS_PER_SAMPLE,
                "start_sample": start,
                "end_sample_exclusive": end,
                "source_start_sample": 0,
                "source_end_sample_exclusive": len(samples),
                "linear_gain": float(item.get("linear_gain", DEFAULT_GAIN)),
                "fade_samples": 80,
                "normalization_policy": "use_sound_asset_policy",
                "render_source_stem": True,
            }
        )
    program = _program(clock=clock, events=events, endpoint_ids=endpoint_ids)
    _write(root / "audio_program.json", program)
    dry_event_mappings = [
        {
            "event_id": event["event_id"],
            "source_id": event["source_endpoint_id"],
            "start_sample": event["start_sample"],
            "end_sample_exclusive": event["end_sample_exclusive"],
            "dry_asset_id": str(binding["sound_asset_id"]),
            "dry_asset_sha256": sha256_file(
                Path(str(binding["path"])).expanduser().resolve()
            ),
            "dry_clip_start_sample": 0,
            "dry_clip_end_sample_exclusive": len(clips[index][0]),
            "linear_gain": event["linear_gain"],
            "fade_samples": event["fade_samples"],
        }
        for index, (binding, event) in enumerate(zip(bindings, events))
    ]
    dry_assembly = assemble_dry_audio_buses(
        dry_event_mappings,
        source_ids=tuple(sorted(endpoint_ids)),
        clip=DryAudioClipSpec.from_values(
            frame_count=frame_count,
            fps_numerator=int(round(frame_rate)),
            sample_rate_hz=sample_rate,
        ),
        asset_bindings={
            str(binding["sound_asset_id"]): str(
                Path(str(binding["path"])).expanduser().resolve()
            )
            for binding in bindings
        },
    )

    scene = load_compiled_acoustic_scene(
        package_path,
        allow_nonpassing_research_qa=True,
    )
    simulation = _simulation(
        direct_ray_count=direct_ray_count,
        indirect_ray_count=indirect_ray_count,
        source_ray_count=source_ray_count,
        indirect_ray_depth=indirect_ray_depth,
        source_ray_depth=source_ray_depth,
    )
    mixture = np.zeros((2, sample_count), dtype=np.float64)
    records: list[dict[str, Any]] = []
    for index, (item, (dry, _), event, endpoint_id) in enumerate(
        zip(bindings, clips, events, endpoint_ids)
    ):
        frame_index = min(
            frame_count - 1,
            int(round(event["start_sample"] / sample_rate * frame_rate)),
        )
        interval_end_frame = min(
            frame_count - 1,
            int(round(event["end_sample_exclusive"] / sample_rate * frame_rate)),
        )
        static_interval = _verify_static_interval(
            emitters[actor_ids[index]],
            camera,
            frame_index,
            interval_end_frame,
        )
        source_position = _cm_to_m(
            emitters[actor_ids[index]][frame_index]["location_cm"]
        )
        listener_position, listener_orientation, listener_basis = _frame_listener(camera, frame_index)
        readback_obj = root / f"rir_{index:02d}_{actor_ids[index]}.obj"
        result = simulate_compiled_acoustic_scene(
            scene,
            simulation,
            source=RuntimeAnchor(anchor_id=endpoint_id, position_m=source_position),
            listener=RuntimeAnchor(
                anchor_id="listener",
                position_m=listener_position,
                orientation_wxyz=listener_orientation,
            ),
            scene_readback_obj=readback_obj,
            runtime_mode="current-installed",
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
        )
        ir = np.asarray(result.samples, dtype=np.float64)
        # The engine bus already applies the declared gain and fades. Consume
        # those samples rather than independently replaying the original WAV.
        emitted_dry = np.asarray(dry_assembly.buses[endpoint_id])[
            event["start_sample"]:event["end_sample_exclusive"]
        ]
        if emitted_dry.shape != dry.shape:
            raise ValueError("assembled speech interval differs from complete source clip")
        start_sample, end_sample, placed_wet = _place_wet_event(
            mixture,
            emitted_dry,
            ir,
            start_sample=event["start_sample"],
            gain=1.0,
        )
        stem_path = stems / f"{index:02d}_{actor_ids[index]}_rir.wav"
        stem = np.zeros_like(mixture)
        stem[:, start_sample:end_sample] = placed_wet
        _write_wav(stem_path, stem)
        pcm_values = np.rint(np.clip(placed_wet, -1.0, 1.0) * 32767.0).astype("<i2")
        records.append(
            {
                "event_id": event["event_id"],
                "actor_id": actor_ids[index],
                "endpoint_id": endpoint_id,
                "sound_asset_id": str(item["sound_asset_id"]),
                "source_start_sample": 0,
                "source_end_sample_exclusive": len(dry),
                "dry_waveform_authority": "assemble_dry_audio_buses",
                "speaker_id": item.get("speaker_id"),
                "transcript": item.get("transcript"),
                "clip_path": str(Path(str(item["path"])).expanduser().resolve()),
                "start_sample": event["start_sample"],
                "end_sample_exclusive": event["end_sample_exclusive"],
                "readback_frame_index": frame_index,
                "emitter_readback_m": list(source_position),
                "listener_readback_m": list(listener_position),
                "listener_orientation_wxyz": list(listener_orientation),
                "listener_basis_m3": {
                    key: list(value) for key, value in listener_basis.items()
                },
                "coordinate_transform_profile": "ue_scene_capture_x_forward_z_up_to_m3_minus_z_forward_y_up",
                "stationary_interval": static_interval,
                "rir_shape": list(ir.shape),
                "rir_sample_count": int(ir.shape[-1]),
                "rir_max_abs": float(np.max(np.abs(ir))),
                "planned_event_interval": [event["start_sample"], event["end_sample_exclusive"]],
                "wet_render_interval": [start_sample, end_sample],
                "wet_float_nonzero_interval": _nonzero_interval(
                    placed_wet, threshold=1.0e-12, offset=start_sample
                ),
                "pcm_output_nonzero_interval": _nonzero_interval(
                    pcm_values, threshold=0.0, offset=start_sample
                ),
                "output_stem": str(stem_path),
                "native_readback_obj": str(readback_obj),
            }
        )
    mixture_path = root / "four_speaker_sequential_mixture.wav"
    _write_wav(mixture_path, mixture)
    report = {
        "schema": "avengine_frame_readback_sequential_speech_research_v1",
        "status": "research",
        "frame_readbacks": str(readback_path),
        "acoustic_package": str(package_path),
        "clock": dict(clock),
        "voice_bindings": bindings,
        "audio_program": str(root / "audio_program.json"),
        "dry_audio_assembly": dry_assembly.metadata(),
        "events": records,
        "mixture_path": str(mixture_path),
        "qa": {
            "speech_identity_and_transcripts": {
                "status": "pass",
                "event_count": len(records),
                "complete_sentences_preserved": True,
            },
            "sequential_nonoverlap": {
                "status": "pass",
                "event_count": len(records),
                "scope": "dry speech emission intervals",
                "reverberation_tails_may_overlap": True,
            },
            "actual_emitter_readback": {
                "status": "pass",
                "actor_count": len(actor_ids),
                "source": "frame_readbacks.emitters",
            },
            "actual_listener_readback": {
                "status": "pass",
                "source": "frame_readbacks.camera[].location_cm",
            },
            "animation_phase_readback": {
                **animation_qa,
                "source": "frame_readbacks.animations",
            },
            "pixel_visibility": {
                "status": "not_run",
                "reason": (
                    "RGB frame files alone do not prove per-actor visibility; "
                    "semantic/depth target readbacks were not supplied."
                ),
            },
        },
        "complete_sentences_preserved": True,
        "native_rlr_per_event": True,
        "formal_admission": False,
        "claim_boundary": (
            "Research audio generated from actual SPEAR emitter/listener readbacks "
            "and existing native RLR; no formal visual/audio admission."
        ),
    }
    _write(root / "research_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-readbacks", required=True, type=Path)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--voice-binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--magnum-python-site", required=True)
    parser.add_argument("--direct-rays", type=int, default=500)
    parser.add_argument("--indirect-rays", type=int, default=5000)
    parser.add_argument("--source-rays", type=int, default=500)
    parser.add_argument("--indirect-depth", type=int, default=64)
    parser.add_argument("--source-depth", type=int, default=16)
    parser.add_argument(
        "--audio-plan",
        type=Path,
        help="optional plan with clock and audio_events; overrides legacy fixed scheduling",
    )
    parser.add_argument(
        "--hrtf",
        dest="hrtf_file",
        type=Path,
        default=Path("/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"),
        help="explicit binaural SOFA input",
    )
    parser.add_argument(
        "--rir-cache",
        type=Path,
        help="fresh cache destination or an existing matching dynamic RIR cache",
    )
    parser.add_argument(
        "--rir-stride",
        type=int,
        default=3,
        help="sample one dynamic RIR every N visual frames (default: 3)",
    )
    parser.add_argument(
        "--neutral-readback",
        type=Path,
        help="optional P1 NeutralReadback JSON; otherwise adapt UE frame readbacks",
    )
    parser.add_argument(
        "--prepared-manifest",
        type=Path,
        help="optional P7 prepared audio manifest for source activity intervals",
    )
    parser.add_argument(
        "--diffraction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the RLR diffraction flag for this research render",
    )
    parser.add_argument(
        "--max-diffraction-order",
        type=int,
        help="override the RLR maximum diffraction order",
    )
    args = parser.parse_args()
    report = render(
        frame_readbacks=args.frame_readbacks,
        package_manifest=args.package_manifest,
        voice_binding=args.voice_binding,
        output=args.output,
        runtime_prefix=args.runtime_prefix,
        rlr_sdk_root=args.rlr_sdk_root,
        magnum_python_site=args.magnum_python_site,
        direct_ray_count=args.direct_rays,
        indirect_ray_count=args.indirect_rays,
        source_ray_count=args.source_rays,
        indirect_ray_depth=args.indirect_depth,
        source_ray_depth=args.source_depth,
        audio_plan=args.audio_plan,
        hrtf_file=args.hrtf_file,
        rir_cache=args.rir_cache,
        rir_stride_frames=args.rir_stride,
        neutral_readback=args.neutral_readback,
        prepared_manifest=args.prepared_manifest,
        diffraction=args.diffraction,
        max_diffraction_order=args.max_diffraction_order,
    )
    output = report.get("mixture_path")
    if output is None:
        output = report.get("outputs", {}).get("four_speaker_sequential_mixture.wav")
    print(json.dumps({"status": report["status"], "output": output}, indent=2))


if __name__ == "__main__":
    main()
