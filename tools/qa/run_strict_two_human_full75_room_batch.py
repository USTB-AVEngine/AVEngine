#!/usr/bin/env python3
"""Fail-closed controller for one same-room strict full75 batch.

This module contains the CPU-visible contract and orchestration state machine.
The native SPEAR adapter is deliberately injected: staging and unit tests can
exercise every no-clobber/resume/failure rule without launching Unreal.  A
production adapter must keep one room process open, create fresh actors for
each Episode, publish an fsynced raw-ready receipt, and prove that the prior
Episode's actors and segmentation proxies no longer exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

REQUEST_SCHEMA = "avengine_native_strict_two_human_room_batch_request_v1"
RESOLVED_SCHEMA = "avengine_native_strict_two_human_room_batch_resolved_v1"
CHECKPOINT_SCHEMA = "avengine_native_strict_two_human_room_batch_checkpoint_v1"
RAW_RECEIPT_SCHEMA = "avengine_native_strict_two_human_raw_capture_receipt_v1"
FINAL_RECEIPT_SCHEMA = "avengine_native_strict_two_human_episode_receipt_v1"
BATCH_RECEIPT_SCHEMA = "avengine_native_strict_two_human_room_batch_receipt_v1"
NATIVE_ADAPTER_SCHEMA = "avengine_native_strict_two_human_room_session_v1"

FRAME_COUNT = 75
HEIGHT = 720
WIDTH = 1280
FPS = 15
SAMPLE_RATE_HZ = 16_000
SAMPLE_COUNT = 80_000
SOURCE_SLOTS = ("source1", "source2")
PRODUCTION_BATCH_SIZE = 10
RESET_CANARY_BATCH_SIZE = 2
QUEUE_DEPTH = 2

# Runtime environments are an execution contract, not a repository-local
# convenience.  GitHub's checked-in runbook assigns UE/SPEAR RPC work to
# ``spear-env`` and Habitat/RLR/AVEngine CPU work to the native Python 3.12
# environment.  A checkout-local ``.venv`` is intentionally not accepted.
SPEAR_CAPTURE_PYTHON = Path("/data/jzy/miniconda3/envs/spear-env/bin/python")
AVENGINE_CPU_PYTHON = Path(
    "/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python"
)
RUNTIME_ENVIRONMENTS = {
    "spear_capture_python": str(SPEAR_CAPTURE_PYTHON),
    "avengine_cpu_python": str(AVENGINE_CPU_PYTHON),
}

# Raw binary memmaps intentionally have no variable-size NPY header.  Shape,
# byte order and dtype are frozen here and repeated in every raw-ready receipt.
RAW_MEMMAP_CONTRACT: dict[str, dict[str, Any]] = {
    "normal_depth_m.f16le": {
        "shape": [FRAME_COUNT, HEIGHT, WIDTH],
        "dtype": "<f2",
        "size_bytes": FRAME_COUNT * HEIGHT * WIDTH * 2,
        "semantics": "normal_scene_metric_depth_m",
    },
    "target_only_source1_depth_m.f16le": {
        "shape": [FRAME_COUNT, HEIGHT, WIDTH],
        "dtype": "<f2",
        "size_bytes": FRAME_COUNT * HEIGHT * WIDTH * 2,
        "semantics": "source1_show_only_metric_depth_m",
    },
    "target_only_source2_depth_m.f16le": {
        "shape": [FRAME_COUNT, HEIGHT, WIDTH],
        "dtype": "<f2",
        "size_bytes": FRAME_COUNT * HEIGHT * WIDTH * 2,
        "semantics": "source2_show_only_metric_depth_m",
    },
    "normal_object_ids.u32le": {
        "shape": [FRAME_COUNT, HEIGHT, WIDTH],
        "dtype": "<u4",
        "size_bytes": FRAME_COUNT * HEIGHT * WIDTH * 4,
        "semantics": "normal_scene_raw_object_ids_uint32",
    },
}
RAW_MEMMAP_TOTAL_BYTES = sum(
    int(value["size_bytes"]) for value in RAW_MEMMAP_CONTRACT.values()
)

CANONICAL_MECHANISMS = {
    "both_static",
    "target_moves",
    "distractor_moves",
    "both_move",
    "camera_pan_both_static",
}
LEGACY_MECHANISM_ALIASES = {
    "static": "both_static",
    "camera_pan": "camera_pan_both_static",
}

CPU_WORKER_POLICY = {
    "worker_count": 1,
    "queue_depth": QUEUE_DEPTH,
    "nice_increment": 10,
    "thread_cap_per_library": 2,
    "environment": {
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_mechanism(value: object) -> str:
    mechanism = LEGACY_MECHANISM_ALIASES.get(str(value), str(value))
    _require(mechanism in CANONICAL_MECHANISMS, f"unknown mechanism: {value}")
    return mechanism


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish JSON with file fsync, atomic rename, then parent-directory fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish a new JSON file without ever replacing a path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link gives us atomic create-if-absent semantics.
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    _require(bool(token), "Episode id does not yield a safe path token")
    return token


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _wav_contract(path: Path) -> dict[str, int]:
    _require(path.is_file(), f"authoritative binaural WAV is missing: {path}")
    with path.open("rb") as stream:
        _require(stream.read(4) == b"RIFF", f"invalid WAV RIFF header: {path}")
        stream.read(4)
        _require(stream.read(4) == b"WAVE", f"invalid WAV WAVE header: {path}")
        fmt: tuple[int, int, int] | None = None
        data_size: int | None = None
        while True:
            chunk_id = stream.read(4)
            if not chunk_id:
                break
            raw_size = stream.read(4)
            _require(len(raw_size) == 4, f"truncated WAV chunk: {path}")
            size = struct.unpack("<I", raw_size)[0]
            payload = stream.read(size)
            _require(len(payload) == size, f"truncated WAV payload: {path}")
            if size % 2:
                _require(len(stream.read(1)) == 1, f"truncated WAV padding: {path}")
            if chunk_id == b"fmt ":
                _require(size >= 16, f"short WAV fmt chunk: {path}")
                tag, channels, sample_rate, _, _, bits = struct.unpack(
                    "<HHIIHH", payload[:16]
                )
                _require(tag in {1, 3}, f"unsupported WAV encoding: {tag}")
                _require(bits > 0 and bits % 8 == 0, f"invalid WAV width: {bits}")
                fmt = channels, sample_rate, channels * bits // 8
            elif chunk_id == b"data":
                data_size = size
    _require(fmt is not None and data_size is not None, f"WAV chunks missing: {path}")
    channels, sample_rate, bytes_per_frame = fmt
    _require(
        bytes_per_frame > 0 and data_size % bytes_per_frame == 0, "WAV frame drift"
    )
    sample_count = data_size // bytes_per_frame
    _require(
        (channels, sample_rate, sample_count) == (2, SAMPLE_RATE_HZ, SAMPLE_COUNT),
        f"binaural WAV contract drift: {(channels, sample_rate, sample_count)}",
    )
    return {
        "channel_count": channels,
        "sample_rate_hz": sample_rate,
        "sample_count": sample_count,
    }


def _validate_suite(
    path: Path, *, episode_id: str, native_map: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    suite = _load(path)
    _require(
        suite.get("schema") == "avengine_optional_spear_apartment_suite_v1",
        "suite schema drift",
    )
    _require(suite.get("native_map") == native_map, "batch native-map drift")
    scenarios = suite.get("scenarios")
    _require(isinstance(scenarios, list), "suite scenarios are missing")
    matches = [item for item in scenarios if item.get("scenario_id") == episode_id]
    _require(len(matches) == 1, "Episode must resolve to exactly one suite scenario")
    scenario = matches[0]
    frames = scenario.get("plan", {}).get("frames")
    _require(
        isinstance(frames, list)
        and len(frames) == FRAME_COUNT
        and [item.get("frame_index") for item in frames] == list(range(FRAME_COUNT)),
        "suite does not contain one ordered full75 Episode",
    )
    for frame in frames:
        states = frame.get("actor_states")
        _require(
            isinstance(states, list)
            and [item.get("actor_id") for item in states]
            == ["source1_actor", "source2_actor"],
            "per-frame two-actor closure drift",
        )
        camera = frame.get("camera_state")
        _require(
            isinstance(camera, Mapping)
            and isinstance(camera.get("pose_hash"), str)
            and camera["pose_hash"],
            "per-frame camera pose identity is missing",
        )
    return suite, scenario


def _validate_acoustics(
    episode: Mapping[str, Any], *, mechanism: str
) -> dict[str, Any]:
    canonical_mechanism = _canonical_mechanism(mechanism)
    evidence = episode.get("acoustic_evidence")
    _require(isinstance(evidence, Mapping), "acoustic evidence is missing")
    required = {"exact_rir_plan", "rir_cache", "binaural_delivery"}
    _require(set(evidence) == required, "acoustic evidence keys drift")
    paths = {key: Path(str(evidence[key])).resolve() for key in required}
    values = {key: _load(path) for key, path in paths.items()}
    jobs = values["exact_rir_plan"].get("jobs")
    _require(isinstance(jobs, list) and bool(jobs), "RIR plan jobs are missing")
    by_slot: dict[str, int] = {slot: 0 for slot in SOURCE_SLOTS}
    for job in jobs:
        slot = job.get("source_slot_id", job.get("source_slot"))
        if slot is None:
            uses = job.get("uses")
            _require(isinstance(uses, list) and uses, "RIR job uses are missing")
            use_slots = {item.get("source_slot_id") for item in uses}
            _require(len(use_slots) == 1, "one RIR cache job spans source slots")
            slot = next(iter(use_slots))
        _require(slot in by_slot, "RIR job source-slot drift")
        by_slot[str(slot)] += 1
    _require(
        all(count > 0 for count in by_slot.values()),
        f"RIR plan does not cover both source slots: {by_slot}",
    )
    observed_count = len(jobs)
    plan = values["exact_rir_plan"]
    declared_unique = plan.get("unique_rir_job_count")
    if declared_unique is not None:
        _require(
            int(declared_unique) == observed_count,
            "RIR plan declared unique-job count drift",
        )
    declared_by_slot = plan.get("distinct_rir_state_count_by_source_slot")
    if declared_by_slot is not None:
        _require(
            declared_by_slot == by_slot,
            "RIR plan declared per-source count drift",
        )
    cache = values["rir_cache"]
    _require(
        cache.get("status") == "pass"
        and cache.get("full_plan_complete") is True
        and int(cache.get("selected_job_count", -1)) == observed_count,
        "exact RIR cache is incomplete",
    )
    delivery = values["binaural_delivery"]
    _require(
        delivery.get("status") == "pass"
        and int(delivery.get("episode_count", -1)) == 1
        and delivery.get("qualification_claim") is False,
        "binaural delivery boundary drift",
    )
    return {
        "status": "pass_precomputed_before_gpu",
        "canonical_mechanism": canonical_mechanism,
        "expected_unique_rir_job_count": observed_count,
        "expected_rir_count_by_source_slot": by_slot,
        "paths": {key: str(path) for key, path in paths.items()},
        "sha256": {key: _sha256(path) for key, path in paths.items()},
    }


def _validate_motion_realism(
    episode: Mapping[str, Any],
    *,
    episode_id: str,
    mechanism: str,
    purpose: str,
) -> dict[str, Any]:
    """Bind release-grade motion evidence; never infer it from a long timeline."""

    canonical_mechanism = _canonical_mechanism(mechanism)
    if canonical_mechanism == "both_static":
        return {"status": "not_applicable_static", "release_qualified": True}
    evidence_value = episode.get("motion_realism_evidence")
    if purpose != "production_room_shard" and evidence_value is None:
        return {
            "status": "release_blocked_pipeline_canary_only",
            "release_qualified": False,
            "reason": "independent motion-realism receipt is absent",
        }
    _require(
        isinstance(evidence_value, str) and evidence_value,
        f"{episode_id}: dynamic production row lacks motion-realism evidence",
    )
    path = Path(evidence_value).resolve()
    _require(path.is_file(), f"motion-realism evidence is missing: {path}")
    value = _load(path)
    active = value.get("active_interval_gate")
    speed = value.get("speed_gate")
    phase = value.get("clip_phase_foot_plant_sync_gate")
    _require(
        value.get("schema") == "avengine_strict_two_human_motion_realism_receipt_v1"
        and value.get("status") == "pass"
        and value.get("episode_id") == episode_id
        and _canonical_mechanism(value.get("mechanism")) == canonical_mechanism
        and value.get("release_qualified") is True
        and value.get("no_global_time_stretch") is True
        and isinstance(active, Mapping)
        and active.get("status") == "pass"
        and isinstance(speed, Mapping)
        and speed.get("status") == "pass"
        and isinstance(phase, Mapping)
        and phase.get("status") == "pass",
        f"{episode_id}: motion-realism release gate failed",
    )
    per_actor = speed.get("per_moving_actor")
    moving_actor_count = speed.get("moving_actor_count")
    _require(
        isinstance(per_actor, Mapping)
        and len(per_actor) >= 1
        and isinstance(moving_actor_count, int)
        and not isinstance(moving_actor_count, bool)
        and moving_actor_count == len(per_actor),
        f"{episode_id}: moving-actor speed inventory failed",
    )
    per_actor_intervals = active.get("per_moving_actor")
    if per_actor_intervals is not None:
        _require(
            isinstance(per_actor_intervals, Mapping)
            and set(per_actor_intervals) == set(per_actor),
            f"{episode_id}: per-actor active interval inventory failed",
        )
    else:
        per_actor_intervals = {actor_id: active for actor_id in per_actor}
    for actor_id, record in per_actor.items():
        _require(isinstance(record, Mapping), f"{actor_id}: speed record missing")
        interval_record = per_actor_intervals[actor_id]
        _require(
            isinstance(interval_record, Mapping),
            f"{actor_id}: active interval record missing",
        )
        interval = interval_record.get("active_frame_interval_inclusive")
        _require(
            isinstance(interval, list)
            and len(interval) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in interval
            )
            and 0 <= interval[0] < interval[1] < FRAME_COUNT
            and interval_record.get("active_frame_count")
            == interval[1] - interval[0] + 1
            and interval_record.get("mapping_kind") == "native_rate_active_interval"
            and interval_record.get("active_speed_evaluated_only") is True,
            f"{actor_id}: active motion interval contract failed",
        )
        measured = record.get("measured_active_speed_mps")
        minimum = record.get("minimum_release_speed_mps")
        maximum = record.get("maximum_release_speed_mps")
        source_interval = record.get("source_native_frame_interval_inclusive")
        _require(
            all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in (measured, minimum, maximum)
            )
            and 0.0 < float(minimum) <= float(measured) <= float(maximum) <= 4.0
            and record.get("active_frame_interval_inclusive") == interval,
            f"{actor_id}: active speed gate failed",
        )
        _require(
            isinstance(source_interval, list)
            and len(source_interval) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in source_interval
            )
            and source_interval[0] < source_interval[1]
            and source_interval[1] - source_interval[0] == interval[1] - interval[0],
            f"{actor_id}: native-rate clip interval mapping failed",
        )
    phase_by_actor = phase.get("per_moving_actor")
    _require(
        isinstance(phase_by_actor, Mapping)
        and set(phase_by_actor) == set(per_actor)
        and all(
            isinstance(record, Mapping)
            and record.get("phase_progression_monotonic") is True
            and record.get("foot_plant_sync") is True
            and record.get("phase_freeze_detected") is False
            for record in phase_by_actor.values()
        ),
        f"{episode_id}: clip phase/foot-plant synchronization failed",
    )
    return {
        "status": "pass_release_qualified",
        "release_qualified": True,
        "path": str(path),
        "sha256": _sha256(path),
        "no_global_time_stretch": True,
        "active_interval_gate": dict(active),
        "speed_gate": dict(speed),
        "clip_phase_foot_plant_sync_gate": dict(phase),
    }


@dataclass(frozen=True)
class EpisodeSpec:
    ordinal: int
    episode_id: str
    mechanism: str
    target_source_slot: str
    target_side: str
    speech_frame_window_inclusive: tuple[int, int]
    suite_plan: Path
    audio_wav: Path
    output_root: Path
    scenario: dict[str, Any]
    bindings: dict[str, Any]

    @property
    def token(self) -> str:
        return f"{self.ordinal:02d}_{_safe_token(self.episode_id)}"


@dataclass(frozen=True)
class ResolvedBatch:
    request_path: Path
    request: dict[str, Any]
    request_sha256: str
    purpose: str
    batch_id: str
    native_map: str
    output_root: Path
    episodes: tuple[EpisodeSpec, ...]
    execution_authorized: bool


def resolve_request(path: Path) -> ResolvedBatch:
    request = _load(path)
    _require(request.get("schema") == REQUEST_SCHEMA, "batch request schema drift")
    _require(
        request.get("stop_on_first_fail") is True, "stop-on-first-fail is required"
    )
    _require(request.get("formal_episode_count") == 0, "formal count must remain zero")
    _require(
        request.get("qualification_claim") is False, "qualification claim forbidden"
    )
    _require(
        request.get("ground_contact_release_qualified") is False,
        "ground-contact release blocker must remain explicit",
    )
    release_blockers = request.get("release_blockers")
    _require(
        isinstance(release_blockers, list)
        and any("ground-contact" in str(item) for item in release_blockers),
        "ground-contact release blocker text is missing",
    )
    _require(
        isinstance(request.get("execution_authorized"), bool),
        "execution authorization must be an explicit boolean",
    )
    _require(
        isinstance(request.get("motion_realism_release_qualified"), bool),
        "motion-realism release state must be an explicit boolean",
    )
    _require(request.get("graphics_adapter_argument") == 1, "adapter must remain 1")
    _require(request.get("physical_gpu_index") == 1, "physical GPU must remain 1")
    _require(
        request.get("forbidden_physical_gpu_indices") == [0, 3],
        "forbidden GPU policy drift",
    )
    purpose = str(request.get("purpose"))
    expected_size = {
        "segmentation_reset_canary": RESET_CANARY_BATCH_SIZE,
        "production_room_shard": PRODUCTION_BATCH_SIZE,
    }.get(purpose)
    _require(expected_size is not None, f"unsupported batch purpose: {purpose}")
    if purpose == "segmentation_reset_canary":
        _require(
            request["motion_realism_release_qualified"] is False,
            "reset canary cannot claim motion-realism release qualification",
        )
        _require(
            any("motion-realism" in str(item) for item in release_blockers),
            "reset canary motion-realism blocker text is missing",
        )
    else:
        _require(
            request["motion_realism_release_qualified"] is True,
            "production shard requires motion-realism release qualification",
        )
    _require(request.get("episode_count") == expected_size, "Episode count drift")
    _require(
        request.get("room_loaded_once") is True, "room-loaded-once contract missing"
    )
    _require(request.get("fresh_actors_per_episode") is True, "fresh actors required")
    _require(
        request.get("segmentation_reset_and_negative_check_per_episode") is True,
        "segmentation reset/negative check required",
    )
    _require(
        request.get("cpu_finalize_queue_depth") == QUEUE_DEPTH, "queue depth drift"
    )
    _require(
        request.get("raw_memmap_contract") == RAW_MEMMAP_CONTRACT,
        "raw memmap contract drift",
    )
    _require(
        request.get("raw_memmap_total_bytes") == RAW_MEMMAP_TOTAL_BYTES,
        "raw memmap total size drift",
    )
    _require(
        request.get("cpu_worker_policy") == CPU_WORKER_POLICY, "CPU worker policy drift"
    )
    _require(
        request.get("runtime_environments") == RUNTIME_ENVIRONMENTS,
        "official Conda runtime environment contract drift",
    )
    if purpose == "production_room_shard":
        canary_path = Path(str(request.get("segmentation_reset_canary_receipt", "")))
        _require(canary_path.is_file(), "two-Episode reset canary receipt is required")
        expected_canary_sha256 = request.get("segmentation_reset_canary_receipt_sha256")
        _require(
            isinstance(expected_canary_sha256, str)
            and len(expected_canary_sha256) == 64
            and _sha256(canary_path) == expected_canary_sha256,
            "two-Episode reset canary receipt digest drift",
        )
        canary = _load(canary_path)
        _require(
            canary.get("schema") == BATCH_RECEIPT_SCHEMA
            and canary.get("purpose") == "segmentation_reset_canary"
            and canary.get("status") == "pass"
            and canary.get("episode_pass_count") == RESET_CANARY_BATCH_SIZE
            and canary.get("native_adapter_schema") == NATIVE_ADAPTER_SCHEMA
            and canary.get("segmentation_reset_gate", {}).get("status") == "pass"
            and canary.get("segmentation_reset_gate", {}).get(
                "all_episode_teardowns_passed"
            )
            is True,
            "two-Episode reset canary did not pass",
        )

    batch_id = str(request.get("batch_id"))
    _require(_safe_token(batch_id) == batch_id, "batch id must already be path-safe")
    native_map = str(request.get("native_map"))
    _require(native_map.startswith("/Game/"), "native map path is invalid")
    output_root = Path(str(request.get("output_root"))).resolve()
    _require(not output_root.is_symlink(), "batch output root cannot be a symlink")
    rows = request.get("episodes")
    _require(
        isinstance(rows, list) and len(rows) == expected_size, "Episode rows drift"
    )

    specs: list[EpisodeSpec] = []
    episode_ids: set[str] = set()
    outputs: set[Path] = set()
    binding_hashes: set[str] = set()
    for ordinal, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"Episode row {ordinal} is not an object")
        _require(row.get("ordinal") == ordinal, "Episode ordinals must be contiguous")
        episode_id = str(row.get("episode_id"))
        _require(episode_id not in episode_ids, "duplicate Episode id")
        episode_ids.add(episode_id)
        mechanism = _canonical_mechanism(row.get("mechanism"))
        target_source_slot = str(row.get("target_source_slot"))
        _require(target_source_slot in SOURCE_SLOTS, "target source-slot drift")
        target_side = str(row.get("target_side"))
        _require(target_side in {"left", "right"}, "target side drift")
        speech_window = row.get("speech_frame_window_inclusive")
        _require(
            isinstance(speech_window, list)
            and len(speech_window) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in speech_window
            )
            and 0 <= speech_window[0] <= speech_window[1] < FRAME_COUNT,
            "target speech frame window drift",
        )
        suite_path = Path(str(row.get("suite_plan"))).resolve()
        audio_path = Path(str(row.get("audio_wav"))).resolve()
        _require(suite_path.is_file(), f"suite plan is missing: {suite_path}")
        suite, scenario = _validate_suite(
            suite_path, episode_id=episode_id, native_map=native_map
        )
        wav = _wav_contract(audio_path)
        acoustics = _validate_acoustics(row, mechanism=mechanism)
        motion_realism = _validate_motion_realism(
            row,
            episode_id=episode_id,
            mechanism=mechanism,
            purpose=purpose,
        )
        output = Path(str(row.get("output_root"))).resolve()
        expected_output = (
            output_root / "episodes" / f"{ordinal:02d}_{_safe_token(episode_id)}"
        )
        _require(output == expected_output, "Episode output path is not canonical")
        _require(_within(output, output_root), "Episode output escapes batch root")
        _require(output not in outputs, "duplicate Episode output root")
        outputs.add(output)
        bindings = {
            "suite_plan": str(suite_path),
            "suite_sha256": _sha256(suite_path),
            "scenario_sha256": _canonical_sha256(scenario),
            "audio_wav": str(audio_path),
            "audio_sha256": _sha256(audio_path),
            "audio_contract": wav,
            "acoustics": acoustics,
            "motion_realism": motion_realism,
            "native_map": suite["native_map"],
            "mechanism": mechanism,
        }
        binding_sha = _canonical_sha256(bindings)
        _require(
            binding_sha not in binding_hashes, "duplicate complete Episode binding"
        )
        binding_hashes.add(binding_sha)
        bindings["binding_sha256"] = binding_sha
        specs.append(
            EpisodeSpec(
                ordinal=ordinal,
                episode_id=episode_id,
                mechanism=mechanism,
                target_source_slot=target_source_slot,
                target_side=target_side,
                speech_frame_window_inclusive=(speech_window[0], speech_window[1]),
                suite_plan=suite_path,
                audio_wav=audio_path,
                output_root=output,
                scenario=scenario,
                bindings=bindings,
            )
        )

    return ResolvedBatch(
        request_path=path.resolve(),
        request=request,
        request_sha256=_canonical_sha256(request),
        purpose=purpose,
        batch_id=batch_id,
        native_map=native_map,
        output_root=output_root,
        episodes=tuple(specs),
        execution_authorized=bool(request["execution_authorized"]),
    )


def require_execution_authorized(batch: ResolvedBatch) -> None:
    _require(
        batch.execution_authorized,
        "batch execution is not authorized; CPU resolution remains available",
    )


def validate_raw_ready_receipt(
    path: Path, *, batch: ResolvedBatch, episode: EpisodeSpec
) -> dict[str, Any]:
    receipt = _load(path)
    _require(receipt.get("schema") == RAW_RECEIPT_SCHEMA, "raw receipt schema drift")
    _require(receipt.get("status") == "pass_raw_ready", "raw receipt did not pass")
    _require(
        receipt.get("batch_request_sha256") == batch.request_sha256, "raw request drift"
    )
    _require(receipt.get("episode_id") == episode.episode_id, "raw Episode drift")
    _require(
        receipt.get("input_binding_sha256") == episode.bindings["binding_sha256"],
        "raw input binding drift",
    )
    _require(receipt.get("frame_count") == FRAME_COUNT, "raw frame count drift")
    _require(receipt.get("rgb_frame_count") == FRAME_COUNT, "raw RGB count drift")
    _require(
        receipt.get("formal_episode_count") == 0
        and receipt.get("qualification_claim") is False
        and receipt.get("ground_contact_release_qualified") is False
        and receipt.get("motion_realism_release_qualified")
        is batch.request["motion_realism_release_qualified"],
        "raw claim boundary drift",
    )
    _require(
        receipt.get("runtime_readback_counts")
        == {"normal": 75, "target_only_source1": 75, "target_only_source2": 75},
        "raw runtime readback count drift",
    )
    teardown = receipt.get("episode_teardown")
    teardown_flags = {
        "actors_destroyed",
        "segmentation_terminated",
        "prior_stable_names_absent",
        "prior_actor_handles_absent",
        "prior_stable_actor_names_absent",
        "prior_proxy_descriptors_absent",
        "proxy_filters_cleared",
        "show_only_list_cleared",
    }
    _require(
        isinstance(teardown, Mapping)
        and all(teardown.get(key) is True for key in teardown_flags)
        and teardown.get("remaining_controlled_actor_handle_count") == 0
        and teardown.get("remaining_controlled_stable_name_count") == 0
        and teardown.get("remaining_controlled_proxy_descriptor_count") == 0,
        "actor/proxy teardown negative existence gate failed",
    )
    files = receipt.get("raw_memmaps")
    _require(isinstance(files, Mapping), "raw memmap inventory is missing")
    _require(set(files) == set(RAW_MEMMAP_CONTRACT), "raw memmap file set drift")
    spool_root = path.parent.resolve()
    for name, contract in RAW_MEMMAP_CONTRACT.items():
        record = files[name]
        _require(record.get("shape") == contract["shape"], f"{name}: shape drift")
        _require(record.get("dtype") == contract["dtype"], f"{name}: dtype drift")
        _require(
            record.get("size_bytes") == contract["size_bytes"], f"{name}: size drift"
        )
        raw_path = Path(str(record.get("path"))).resolve()
        _require(raw_path == spool_root / name, f"{name}: noncanonical raw path")
        _require(raw_path.is_file(), f"{name}: raw file missing")
        _require(
            raw_path.stat().st_size == contract["size_bytes"],
            f"{name}: file size drift",
        )
        digest = record.get("sha256")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and digest == _sha256(raw_path),
            f"{name}: content digest drift",
        )
    rgb = receipt.get("rgb_frames")
    rgb_root = spool_root / "rgb_frames"
    rgb_paths = [rgb_root / f"frame_{index:06d}.png" for index in range(75)]
    _require(
        isinstance(rgb, Mapping)
        and rgb.get("file_count") == FRAME_COUNT
        and rgb.get("names") == [f"frame_{index:06d}.png" for index in range(75)],
        "raw RGB inventory drift",
    )
    _require(
        all(path.is_file() and path.stat().st_size > 0 for path in rgb_paths),
        "raw RGB file missing",
    )
    rgb_inventory = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in rgb_paths
    ]
    _require(
        rgb.get("inventory_sha256") == _canonical_sha256(rgb_inventory),
        "raw RGB inventory digest drift",
    )
    metadata = receipt.get("metadata")
    required_metadata = {
        "runtime_readbacks.json",
        "runtime_asset_readbacks.json",
        "normal_object_id_descriptors.json",
        "capture_context.json",
    }
    _require(
        isinstance(metadata, Mapping) and set(metadata) == required_metadata,
        "raw metadata inventory drift",
    )
    for name in required_metadata:
        path_value = spool_root / name
        record = metadata[name]
        _require(path_value.is_file(), f"raw metadata missing: {name}")
        _require(record.get("path") == str(path_value), f"{name}: metadata path drift")
        _require(
            record.get("size_bytes") == path_value.stat().st_size,
            f"{name}: metadata size drift",
        )
        _require(
            record.get("sha256") == _sha256(path_value),
            f"{name}: metadata digest drift",
        )
    context = _load(spool_root / "capture_context.json")
    reset_begin = context.get("segmentation_begin")
    _require(
        context.get("schema")
        == "avengine_native_strict_two_human_raw_capture_context_v1"
        and context.get("episode_id") == episode.episode_id
        and context.get("native_map") == batch.native_map
        and context.get("frame_indices") == list(range(FRAME_COUNT))
        and isinstance(context.get("camera_pose_ids"), list)
        and len(context["camera_pose_ids"]) == FRAME_COUNT
        and isinstance(reset_begin, Mapping)
        and reset_begin.get("status") == "pass"
        and reset_begin.get("prior_stable_names_absent") is True
        and reset_begin.get("proxy_filters_cleared") is True
        and reset_begin.get("show_only_list_cleared") is True,
        "raw capture/reset context drift",
    )
    return receipt


def _checkpoint_path(batch: ResolvedBatch) -> Path:
    return batch.output_root / "checkpoint.json"


def _receipt_path(episode: EpisodeSpec) -> Path:
    return episode.output_root / "FINAL_READY.json"


def _raw_ready_candidates(episode: EpisodeSpec) -> list[Path]:
    if not episode.output_root.is_dir():
        return []
    return sorted(episode.output_root.glob("attempt_*/raw_spool/RAW_READY.json"))


def _next_attempt_root(episode: EpisodeSpec) -> Path:
    episode.output_root.mkdir(parents=True, exist_ok=True)
    attempts = []
    for path in episode.output_root.glob("attempt_*"):
        match = re.fullmatch(r"attempt_(\d{3})", path.name)
        if match:
            attempts.append(int(match.group(1)))
    value = max(attempts, default=0) + 1
    result = episode.output_root / f"attempt_{value:03d}"
    result.mkdir(exist_ok=False)
    _fsync_directory(episode.output_root)
    return result


def _validate_final_receipt(
    path: Path, *, batch: ResolvedBatch, episode: EpisodeSpec
) -> dict[str, Any]:
    receipt = _load(path)
    _require(
        receipt.get("schema") == FINAL_RECEIPT_SCHEMA, "final receipt schema drift"
    )
    _require(receipt.get("status") == "pass", "Episode finalization did not pass")
    _require(receipt.get("episode_id") == episode.episode_id, "final Episode drift")
    _require(
        receipt.get("batch_request_sha256") == batch.request_sha256,
        "final request drift",
    )
    _require(
        receipt.get("input_binding_sha256") == episode.bindings["binding_sha256"],
        "final input binding drift",
    )
    contract = receipt.get("capture_contract")
    _require(
        contract
        == {
            "normal_rgb_frames": 75,
            "normal_metric_depth_frames": 75,
            "source1_target_only_depth_frames": 75,
            "source2_target_only_depth_frames": 75,
            "normal_runtime_readbacks": 75,
            "target_only_runtime_readbacks": 150,
            "live_asset_readback": True,
        },
        "final strict capture contract drift",
    )
    _require(receipt.get("formal_episode_count") == 0, "final formal count drift")
    _require(receipt.get("qualification_claim") is False, "final claim boundary drift")
    _require(
        receipt.get("ground_contact_release_qualified") is False,
        "final ground-contact boundary drift",
    )
    _require(
        receipt.get("motion_realism_release_qualified")
        is batch.request["motion_realism_release_qualified"],
        "final motion-realism boundary drift",
    )
    raw_ready = Path(str(receipt.get("raw_ready", ""))).resolve()
    _require(raw_ready.is_file(), "final raw-ready binding is missing")
    _require(
        raw_ready in _raw_ready_candidates(episode),
        "final raw-ready binding is not a canonical Episode attempt",
    )
    _require(
        receipt.get("raw_ready_sha256") == _sha256(raw_ready),
        "final raw-ready digest drift",
    )
    validate_raw_ready_receipt(raw_ready, batch=batch, episode=episode)
    manifest_path = Path(str(receipt.get("manifest", ""))).resolve()
    finalized_output = Path(str(receipt.get("finalized_output", ""))).resolve()
    _require(
        manifest_path.is_file()
        and manifest_path.parent == finalized_output
        and finalized_output == raw_ready.parents[1] / "finalized_output",
        "final manifest path is not canonical",
    )
    _require(
        receipt.get("manifest_sha256") == _sha256(manifest_path),
        "final manifest digest drift",
    )
    manifest = _load(manifest_path)
    _require(
        manifest.get("schema")
        == "avengine_native_strict_two_human_raw_finalization_manifest_v1"
        and manifest.get("status") == "pass"
        and manifest.get("episode_id") == episode.episode_id
        and manifest.get("input_binding_sha256") == episode.bindings["binding_sha256"]
        and manifest.get("capture_contract") == contract
        and manifest.get("formal_episode_count") == 0
        and manifest.get("qualification_claim") is False
        and manifest.get("ground_contact_release_qualified") is False
        and manifest.get("motion_realism_release_qualified")
        is batch.request["motion_realism_release_qualified"],
        "final manifest binding drift",
    )
    return receipt


def _validate_batch_receipt(path: Path, *, batch: ResolvedBatch) -> dict[str, Any]:
    receipt = _load(path)
    expected_ids = [episode.episode_id for episode in batch.episodes]
    _require(
        receipt.get("schema") == BATCH_RECEIPT_SCHEMA
        and receipt.get("status") == "pass"
        and receipt.get("purpose") == batch.purpose
        and receipt.get("batch_id") == batch.batch_id
        and receipt.get("native_map") == batch.native_map
        and receipt.get("batch_request_sha256") == batch.request_sha256
        and receipt.get("episode_count") == len(expected_ids)
        and receipt.get("episode_pass_count") == len(expected_ids)
        and receipt.get("episode_ids_in_order") == expected_ids
        and receipt.get("native_adapter_schema") == NATIVE_ADAPTER_SCHEMA
        and receipt.get("formal_episode_count") == 0
        and receipt.get("qualification_claim") is False
        and receipt.get("ground_contact_release_qualified") is False
        and receipt.get("motion_realism_release_qualified")
        is batch.request["motion_realism_release_qualified"],
        "batch receipt binding drift",
    )
    gate = receipt.get("segmentation_reset_gate")
    _require(
        isinstance(gate, Mapping)
        and gate.get("status") == "pass"
        and gate.get("all_episode_teardowns_passed") is True
        and gate.get("episode_ids") == expected_ids,
        "batch segmentation-reset gate drift",
    )
    per_episode = receipt.get("per_episode_receipts")
    _require(isinstance(per_episode, Mapping), "batch per-Episode receipts missing")
    for episode in batch.episodes:
        final_path = _receipt_path(episode)
        _require(
            per_episode.get(episode.episode_id) == str(final_path),
            "batch per-Episode receipt path drift",
        )
        _validate_final_receipt(final_path, batch=batch, episode=episode)
    return receipt


class NativeRoomSession(Protocol):
    """Required same-process adapter implemented beside the SPEAR backend."""

    def capture_episode_raw(
        self, *, episode: EpisodeSpec, attempt_root: Path, batch: ResolvedBatch
    ) -> Path:
        """Return an atomically published RAW_READY.json path."""

    def close(self) -> None:
        """Close the one shared packaged process."""


class FutureLike(Protocol):
    def result(self) -> Path: ...
    def done(self) -> bool: ...


class FinalizeQueue(Protocol):
    def submit(
        self,
        *,
        batch: ResolvedBatch,
        episode: EpisodeSpec,
        raw_ready: Path,
        attempt_root: Path,
    ) -> FutureLike: ...

    def close(self) -> None: ...


@dataclass
class _PendingFinalize:
    episode: EpisodeSpec
    future: FutureLike


def _write_checkpoint(
    batch: ResolvedBatch,
    *,
    status: str,
    completed: Sequence[str],
    raw_ready: Sequence[str],
    pending: Sequence[str],
    failed_episode_id: str | None = None,
    error: str | None = None,
) -> None:
    _atomic_write_json(
        _checkpoint_path(batch),
        {
            "schema": CHECKPOINT_SCHEMA,
            "status": status,
            "batch_id": batch.batch_id,
            "purpose": batch.purpose,
            "batch_request_sha256": batch.request_sha256,
            "completed_episode_ids": list(completed),
            "raw_ready_episode_ids": list(raw_ready),
            "pending_episode_ids": list(pending),
            "failed_episode_id": failed_episode_id,
            "error": error,
            "stop_on_first_observed_fail": True,
        },
    )


def execute_batch(
    batch: ResolvedBatch,
    *,
    session_factory: Callable[[ResolvedBatch], NativeRoomSession],
    finalize_queue_factory: Callable[[Mapping[str, Any]], FinalizeQueue],
    resume: bool,
) -> Path:
    """Execute a shared-room batch with raw/finalize overlap and hard backpressure.

    Native capture failures are synchronous and stop before the next Episode.
    CPU failures stop capture as soon as observed; at most the one Episode
    already inside the native capture call can finish after that CPU failure
    occurs.  No raw artifact is a passing Episode until FINAL_READY validates.
    """

    require_execution_authorized(batch)
    root = batch.output_root
    if root.exists() and not resume:
        raise FileExistsError(f"refusing to replace batch output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "request_snapshot.json"
    if snapshot.is_file():
        retained = _load(snapshot)
        _require(retained == batch.request, "resume request differs from snapshot")
    else:
        _atomic_write_json(snapshot, batch.request)

    completed: list[str] = []
    final_receipts: dict[str, dict[str, Any]] = {}
    raw_by_episode: dict[str, Path] = {}
    todo: list[EpisodeSpec] = []
    for episode in batch.episodes:
        final_path = _receipt_path(episode)
        if final_path.is_file():
            final_receipts[episode.episode_id] = _validate_final_receipt(
                final_path, batch=batch, episode=episode
            )
            completed.append(episode.episode_id)
            continue
        candidates = _raw_ready_candidates(episode)
        if candidates:
            _require(len(candidates) == 1, "multiple raw-ready attempts require audit")
            validate_raw_ready_receipt(candidates[0], batch=batch, episode=episode)
            raw_by_episode[episode.episode_id] = candidates[0]
        todo.append(episode)

    if not todo:
        receipt_path = root / "BATCH_READY.json"
        if receipt_path.is_file():
            _validate_batch_receipt(receipt_path, batch=batch)
            return receipt_path

    queue = finalize_queue_factory(CPU_WORKER_POLICY)
    pending: deque[_PendingFinalize] = deque()
    session: NativeRoomSession | None = None

    def collect_one(*, block: bool) -> None:
        if not pending or (not block and not pending[0].future.done()):
            return
        item = pending[0]
        final_path = item.future.result()
        final_receipts[item.episode.episode_id] = _validate_final_receipt(
            final_path, batch=batch, episode=item.episode
        )
        pending.popleft()
        completed.append(item.episode.episode_id)

    active_episode_id: str | None = None
    try:
        # Resume can replay CPU finalization from a complete raw receipt without
        # opening SPEAR.  Partial attempts without RAW_READY are preserved and
        # recaptured into a new no-clobber attempt directory.
        for episode in todo:
            raw_ready = raw_by_episode.get(episode.episode_id)
            if raw_ready is not None:
                attempt_root = raw_ready.parents[1]
                pending.append(
                    _PendingFinalize(
                        episode,
                        queue.submit(
                            batch=batch,
                            episode=episode,
                            raw_ready=raw_ready,
                            attempt_root=attempt_root,
                        ),
                    )
                )
                while len(pending) >= QUEUE_DEPTH:
                    collect_one(block=True)

        capture_todo = [
            episode for episode in todo if episode.episode_id not in raw_by_episode
        ]
        if capture_todo:
            session = session_factory(batch)
        for episode in capture_todo:
            collect_one(block=False)
            while len(pending) >= QUEUE_DEPTH:
                collect_one(block=True)
            active_episode_id = episode.episode_id
            attempt_root = _next_attempt_root(episode)
            _write_checkpoint(
                batch,
                status="running",
                completed=completed,
                raw_ready=list(raw_by_episode),
                pending=[item.episode.episode_id for item in pending],
            )
            assert session is not None
            raw_ready = session.capture_episode_raw(
                episode=episode,
                attempt_root=attempt_root,
                batch=batch,
            )
            validate_raw_ready_receipt(raw_ready, batch=batch, episode=episode)
            raw_by_episode[episode.episode_id] = raw_ready
            pending.append(
                _PendingFinalize(
                    episode,
                    queue.submit(
                        batch=batch,
                        episode=episode,
                        raw_ready=raw_ready,
                        attempt_root=attempt_root,
                    ),
                )
            )
            active_episode_id = None
        if session is not None:
            session.close()
            session = None
        while pending:
            collect_one(block=True)
    except BaseException as exc:
        _write_checkpoint(
            batch,
            status="fail_closed",
            completed=completed,
            raw_ready=list(raw_by_episode),
            pending=[item.episode.episode_id for item in pending],
            failed_episode_id=(
                active_episode_id
                or (pending[0].episode.episode_id if pending else None)
            ),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if session is not None:
            session.close()
        queue.close()

    expected = [episode.episode_id for episode in batch.episodes]
    _require(set(completed) == set(expected), "batch Episode closure is incomplete")
    teardown_by_episode: dict[str, dict[str, Any]] = {}
    for episode in batch.episodes:
        final = final_receipts[episode.episode_id]
        raw = _load(Path(str(final["raw_ready"])))
        teardown_by_episode[episode.episode_id] = dict(raw["episode_teardown"])
    receipt = {
        "schema": BATCH_RECEIPT_SCHEMA,
        "status": "pass",
        "purpose": batch.purpose,
        "batch_id": batch.batch_id,
        "native_map": batch.native_map,
        "room_process_launch_count": 1,
        "native_adapter_schema": NATIVE_ADAPTER_SCHEMA,
        "episode_count": len(batch.episodes),
        "episode_pass_count": len(completed),
        "episode_ids_in_order": expected,
        "batch_request_sha256": batch.request_sha256,
        "capture_contract": {
            "normal_rgb_frames": FRAME_COUNT * len(batch.episodes),
            "normal_metric_depth_frames": FRAME_COUNT * len(batch.episodes),
            "target_only_depth_frames": 2 * FRAME_COUNT * len(batch.episodes),
            "native_render_pass_frames": 3 * FRAME_COUNT * len(batch.episodes),
        },
        "raw_memmap_contract": RAW_MEMMAP_CONTRACT,
        "cpu_worker_policy": CPU_WORKER_POLICY,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "ground_contact_release_qualified": False,
        "motion_realism_release_qualified": batch.request[
            "motion_realism_release_qualified"
        ],
        "release_blockers": list(batch.request["release_blockers"]),
        "segmentation_reset_gate": {
            "status": "pass",
            "episode_ids": expected,
            "all_episode_teardowns_passed": True,
            "per_episode": teardown_by_episode,
        },
        "per_episode_receipts": {
            episode_id: str(
                _receipt_path(
                    next(
                        episode
                        for episode in batch.episodes
                        if episode.episode_id == episode_id
                    )
                )
            )
            for episode_id in expected
        },
    }
    receipt_path = root / "BATCH_READY.json"
    _atomic_write_json_new(receipt_path, receipt)
    _write_checkpoint(
        batch,
        status="pass",
        completed=expected,
        raw_ready=expected,
        pending=[],
    )
    return receipt_path


def resolved_plan(batch: ResolvedBatch) -> dict[str, Any]:
    return {
        "schema": RESOLVED_SCHEMA,
        "status": "pass_cpu_only_no_gpu_launched",
        "batch_id": batch.batch_id,
        "purpose": batch.purpose,
        "native_map": batch.native_map,
        "episode_count": len(batch.episodes),
        "room_process_launch_count": 1,
        "batch_request_sha256": batch.request_sha256,
        "raw_memmap_contract": RAW_MEMMAP_CONTRACT,
        "raw_memmap_total_bytes_per_episode": RAW_MEMMAP_TOTAL_BYTES,
        "raw_memmap_total_bytes_for_batch": RAW_MEMMAP_TOTAL_BYTES
        * len(batch.episodes),
        "cpu_worker_policy": CPU_WORKER_POLICY,
        "episodes": [
            {
                "ordinal": episode.ordinal,
                "episode_id": episode.episode_id,
                "mechanism": episode.mechanism,
                "target_source_slot": episode.target_source_slot,
                "target_side": episode.target_side,
                "speech_frame_window_inclusive": list(
                    episode.speech_frame_window_inclusive
                ),
                "output_root": str(episode.output_root),
                "input_binding_sha256": episode.bindings["binding_sha256"],
                "expected_unique_rir_job_count": episode.bindings["acoustics"][
                    "expected_unique_rir_job_count"
                ],
            }
            for episode in batch.episodes
        ],
        "gates": {
            "full75_only": True,
            "normal_rgb_depth": True,
            "two_target_only_depth_passes": True,
            "runtime_readback_every_native_pass": True,
            "live_asset_readback": True,
            "fresh_actors_per_episode": True,
            "segmentation_reset_and_negative_existence_check": True,
            "fsync_and_atomic_raw_ready": True,
            "stop_on_first_observed_fail": True,
            "production_requires_two_episode_reset_canary": True,
        },
        "formal_episode_count": 0,
        "qualification_claim": False,
        "execution_authorized": batch.execution_authorized,
        "ground_contact_release_qualified": False,
        "motion_realism_release_qualified": batch.request[
            "motion_realism_release_qualified"
        ],
        "release_blockers": list(batch.request["release_blockers"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--resolved-output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and resolve the CPU contract; this staging runner never launches GPU.",
    )
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error(
            "local staging is CPU-only; integrate an audited NativeRoomSession and "
            "run the two-Episode reset canary before enabling execution"
        )
    batch = resolve_request(args.request.resolve())
    _atomic_write_json_new(args.resolved_output.resolve(), resolved_plan(batch))
    print(f"STRICT_TWO_HUMAN_ROOM_BATCH_DRY_RUN_OK output={args.resolved_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
