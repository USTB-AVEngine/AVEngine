#!/usr/bin/env python3
"""Verify the complete M7 asset-bound binaural throughput batch.

This is an artifact-level verifier, rather than a claim based on a successful
launcher.  It checks that every delivered WAV/sidecar is a valid exact M7
sample, that all RIR uses point at the final asset-bound emitter centers, and
optionally compares every matching sample to a prior delivery byte-for-byte.
It does not turn this research batch or its review videos into dataset
admission.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    sha256_file,
    write_json,
)
from avengine.spatial_audio.audio import read_float32_wav
from avengine.timeline.audio_program import (
    AudioProgramError,
    materialize_audio_program_variant,
    validate_audio_program,
)
from avengine.routes.room_feasibility import rir_acoustic_state_sha256
from avengine.dataset.sensor_rig import (
    M7SensorRigError,
    m7_sensor_rig_binding,
    m7_sensor_rig_pose_series,
    validate_m7_rir_listener_alignment,
)


FRAME_COUNT = 75
SOURCE_SLOTS = ("source1", "source2")
SAMPLE_RATE_HZ = 16_000
SAMPLE_COUNT = 80_000
SCHEMA = "avengine_m7_asset_bound_batch_correctness_report_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACOUSTIC_SELECTION_FIELDS = {
    "schema",
    "selection_mode",
    "registry_selection_applied",
    "room_ref",
    "profile_ref",
    "binding_id",
    "registry_selection_content_sha256",
    "effective_selection_content_sha256",
    "acoustic_package_manifest_sha256",
    "simulation_request_sha256",
    "input_receipt_sha256",
    "binding_content_sha256",
}
_AUDIO_PROGRAM_SAMPLE_FIELDS = frozenset(
    {
        "audio_program_binding",
        "audio_program_instance_path",
        "audio_program_instance_sha256",
    }
)


class BatchVerificationError(RuntimeError):
    """A batch artifact does not satisfy its declared M7 contract."""


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchVerificationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BatchVerificationError(f"JSON object required: {path}")
    return value


def _validated_acoustic_selection_binding(
    value: Any,
) -> tuple[dict[str, Any], str | None]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema")
        != "avengine_rir_cache_acoustic_selection_binding_v1"
        or set(value) != _ACOUSTIC_SELECTION_FIELDS
    ):
        raise BatchVerificationError(
            "batch acoustic_selection_binding is invalid"
        )
    binding = dict(value)
    mode = binding.get("selection_mode")
    binding_sha256 = binding.get("binding_content_sha256")
    if mode == "explicit_legacy_unbound":
        if (
            binding_sha256 is not None
            or binding.get("registry_selection_applied") is not False
            or binding.get("room_ref") is not None
            or binding.get("profile_ref") is not None
            or binding.get("binding_id") is not None
            or any(
                binding.get(field) is not None
                for field in (
                    "registry_selection_content_sha256",
                    "effective_selection_content_sha256",
                    "acoustic_package_manifest_sha256",
                    "simulation_request_sha256",
                    "input_receipt_sha256",
                )
            )
        ):
            raise BatchVerificationError(
                "legacy unbound batch fabricated an acoustic identity"
            )
        return binding, None
    if (
        mode
        not in {
            "explicit_legacy",
            "registry",
            "registry_with_verified_equivalent_overrides",
        }
        or not isinstance(binding_sha256, str)
        or _SHA256_RE.fullmatch(binding_sha256) is None
        or canonical_json_sha256(
            {
                key: item
                for key, item in binding.items()
                if key != "binding_content_sha256"
            }
        )
        != binding_sha256
    ):
        raise BatchVerificationError(
            "batch acoustic_selection_binding hash is invalid"
        )
    if mode == "explicit_legacy":
        effective_sha256 = binding.get(
            "effective_selection_content_sha256"
        )
        input_receipt_sha256 = binding.get("input_receipt_sha256")
        if (
            binding.get("registry_selection_applied") is not False
            or binding.get("room_ref") is not None
            or binding.get("profile_ref") is not None
            or binding.get("binding_id") is not None
            or binding.get("registry_selection_content_sha256") is not None
            or (
                (effective_sha256 is None) != (input_receipt_sha256 is None)
            )
            or (
                effective_sha256 is not None
                and (
                    not isinstance(effective_sha256, str)
                    or _SHA256_RE.fullmatch(effective_sha256) is None
                    or not isinstance(input_receipt_sha256, str)
                    or _SHA256_RE.fullmatch(input_receipt_sha256) is None
                )
            )
        ):
            raise BatchVerificationError(
                "explicit legacy batch contains a registry identity"
            )
    else:
        room_ref = binding.get("room_ref")
        profile_ref = binding.get("profile_ref")
        if (
            binding.get("registry_selection_applied") is not True
            or not isinstance(room_ref, Mapping)
            or set(room_ref) != {"registry_id", "room_id", "revision"}
            or not isinstance(profile_ref, Mapping)
            or set(profile_ref) != {"profile_id", "revision"}
            or not isinstance(binding.get("binding_id"), str)
            or not binding["binding_id"]
            or any(
                not isinstance(binding.get(field), str)
                or _SHA256_RE.fullmatch(binding[field]) is None
                for field in (
                    "registry_selection_content_sha256",
                    "effective_selection_content_sha256",
                    "input_receipt_sha256",
                )
            )
        ):
            raise BatchVerificationError(
                "registry batch lacks its exact room/profile binding"
            )
    if any(
        not isinstance(binding.get(field), str)
        or _SHA256_RE.fullmatch(binding[field]) is None
        for field in (
            "acoustic_package_manifest_sha256",
            "simulation_request_sha256",
        )
    ):
        raise BatchVerificationError(
            "batch acoustic package/simulation identity is invalid"
        )
    return binding, binding_sha256


def _program_sample_fields(value: Mapping[str, Any]) -> set[str]:
    return _AUDIO_PROGRAM_SAMPLE_FIELDS.intersection(value)


def _batch_relative_file(
    batch_root: Path,
    raw_path: Any,
    *,
    owner: str,
) -> Path:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or Path(raw_path).is_absolute()
    ):
        raise BatchVerificationError(f"{owner} must be a relative path")
    root = batch_root.resolve()
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BatchVerificationError(f"{owner} escapes the batch root") from exc
    if not resolved.is_file():
        raise BatchVerificationError(f"{owner} is not a regular file")
    return resolved


def _audio_program_instance(
    *,
    batch_root: Path,
    sample: Mapping[str, Any],
    sample_id: str,
    asset_ids_by_source_slot: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate one optional M6 AudioProgram instance bound to an M7 sample."""

    present = _program_sample_fields(sample)
    if not present:
        return None
    if present != _AUDIO_PROGRAM_SAMPLE_FIELDS:
        missing = sorted(_AUDIO_PROGRAM_SAMPLE_FIELDS - present)
        raise BatchVerificationError(
            f"sample {sample_id} has an incomplete AudioProgram binding; "
            f"missing={missing}"
        )

    binding = sample.get("audio_program_binding")
    if not isinstance(binding, Mapping):
        raise BatchVerificationError(
            f"sample {sample_id} audio_program_binding must be an object"
        )
    program_ref = binding.get("audio_program_ref")
    variant_id = binding.get("variant_id")
    materialized_hash = binding.get("materialized_program_content_sha256")
    assembly_hash = binding.get("dry_audio_assembly_content_sha256")
    endpoint_to_slot = binding.get("source_endpoint_to_source_slot")
    if not isinstance(program_ref, Mapping):
        raise BatchVerificationError(
            f"sample {sample_id} AudioProgram reference must be an object"
        )
    program_id = program_ref.get("program_id")
    revision = program_ref.get("revision")
    canonical_program_hash = program_ref.get("program_content_sha256")
    if (
        not isinstance(program_id, str)
        or not program_id
        or not isinstance(revision, str)
        or not revision
        or not isinstance(canonical_program_hash, str)
        or _SHA256_RE.fullmatch(canonical_program_hash) is None
        or variant_id not in {"A", "B"}
        or not isinstance(materialized_hash, str)
        or _SHA256_RE.fullmatch(materialized_hash) is None
        or not isinstance(assembly_hash, str)
        or _SHA256_RE.fullmatch(assembly_hash) is None
        or not isinstance(endpoint_to_slot, Mapping)
        or not endpoint_to_slot
        or any(
            not isinstance(endpoint_id, str)
            or not endpoint_id
            or source_slot not in SOURCE_SLOTS
            for endpoint_id, source_slot in endpoint_to_slot.items()
        )
        or len(endpoint_to_slot) != len(SOURCE_SLOTS)
        or set(endpoint_to_slot.values()) != set(SOURCE_SLOTS)
        or set(asset_ids_by_source_slot) != set(SOURCE_SLOTS)
    ):
        raise BatchVerificationError(
            f"sample {sample_id} AudioProgram binding fields are invalid"
        )

    instance_path = _batch_relative_file(
        batch_root,
        sample.get("audio_program_instance_path"),
        owner=f"sample {sample_id} audio_program_instance_path",
    )
    instance_sha256 = sample.get("audio_program_instance_sha256")
    if (
        not isinstance(instance_sha256, str)
        or _SHA256_RE.fullmatch(instance_sha256) is None
        or sha256_file(instance_path) != instance_sha256
    ):
        raise BatchVerificationError(
            f"sample {sample_id} AudioProgram instance hash differs"
        )
    instance = _json(instance_path)
    if (
        instance.get("schema") != "avengine_m7_m6_audio_program_instance_v1"
        or instance.get("status") != "pass"
        or instance.get("audio_program_binding") != binding
    ):
        raise BatchVerificationError(
            f"sample {sample_id} AudioProgram instance binding differs"
        )

    program = instance.get("materialized_audio_program")
    base_program = instance.get("base_audio_program")
    if not isinstance(program, Mapping) or not isinstance(base_program, Mapping):
        raise BatchVerificationError(
            f"sample {sample_id} base or materialized AudioProgram is missing"
        )
    base_errors = validate_audio_program(base_program)
    program_errors = validate_audio_program(program)
    if base_errors or program_errors:
        first_error = (base_errors or program_errors)[0]
        raise BatchVerificationError(
            f"sample {sample_id} AudioProgram is invalid: {first_error}"
        )
    if (
        base_program.get("program_id") != program_id
        or base_program.get("revision") != revision
        or base_program.get("program_content_sha256") != canonical_program_hash
        or program.get("program_id") != program_id
        or program.get("revision") != revision
        or program.get("program_content_sha256") != materialized_hash
        or set(program.get("candidate_source_endpoint_ids", ()))
        != set(endpoint_to_slot)
    ):
        raise BatchVerificationError(
            f"sample {sample_id} materialized AudioProgram differs from its binding"
        )
    try:
        expected_materialized = materialize_audio_program_variant(
            base_program,
            variant_id,
        )
    except AudioProgramError as error:
        raise BatchVerificationError(
            f"sample {sample_id} AudioProgram variant is invalid: {error}"
        ) from error
    if expected_materialized != program:
        raise BatchVerificationError(
            f"sample {sample_id} materialized AudioProgram differs from its base variant"
        )
    mode = program.get("mode")

    assembly = instance.get("dry_audio_assembly")
    if not isinstance(assembly, Mapping):
        raise BatchVerificationError(
            f"sample {sample_id} dry audio assembly is missing"
        )
    declared_assembly_hash = assembly.get("assembly_content_sha256")
    expected_assembly_hash = canonical_json_sha256(
        {
            key: value
            for key, value in assembly.items()
            if key != "assembly_content_sha256"
        }
    )
    source_ids = assembly.get("source_ids")
    if (
        declared_assembly_hash != assembly_hash
        or expected_assembly_hash != assembly_hash
        or not isinstance(source_ids, list)
        or len(source_ids) != len(endpoint_to_slot)
        or set(source_ids) != set(endpoint_to_slot)
    ):
        raise BatchVerificationError(
            f"sample {sample_id} dry audio assembly differs from its binding"
        )

    intervals = {source_slot: [] for source_slot in SOURCE_SLOTS}
    for event in program["events"]:
        intervals[endpoint_to_slot[event["source_endpoint_id"]]].append(
            (event["start_sample"], event["end_sample_exclusive"])
        )
    simultaneous_active_sample_count = sum(
        max(0, min(left_end, right_end) - max(left_start, right_start))
        for left_start, left_end in intervals["source1"]
        for right_start, right_end in intervals["source2"]
    )
    expected_both_sources_active = simultaneous_active_sample_count > 0
    activity_summary = instance.get("source_activity_summary")
    if activity_summary is not None:
        if (
            not isinstance(activity_summary, Mapping)
            or activity_summary.get("both_sources_active")
            is not expected_both_sources_active
            or activity_summary.get("simultaneous_active_sample_count")
            != simultaneous_active_sample_count
        ):
            raise BatchVerificationError(
                f"sample {sample_id} source activity summary differs from AudioProgram"
            )
    if (
        "source_activity_summary" in sample
        and sample.get("source_activity_summary") != activity_summary
    ):
        raise BatchVerificationError(
            f"sample {sample_id} source activity summary differs from its instance"
        )
    sound_semantics = instance.get("sound_asset_semantics")
    used_sound_ids = {
        event["sound_asset_id"] for event in program["events"]
    }
    if (
        not isinstance(sound_semantics, Mapping)
        or set(sound_semantics) != used_sound_ids
        or any(
            not isinstance(sound_class, str) or not sound_class
            for sound_class in sound_semantics.values()
        )
    ):
        raise BatchVerificationError(
            f"sample {sample_id} AudioProgram sound semantics are invalid"
        )
    mapped_events = instance.get("mapped_events")
    if mapped_events != [
        {
            **dict(event),
            "source_slot_id": endpoint_to_slot[event["source_endpoint_id"]],
            "semantic_sound_class": sound_semantics[event["sound_asset_id"]],
        }
        for event in program["events"]
    ]:
        raise BatchVerificationError(
            f"sample {sample_id} mapped AudioProgram events differ"
        )
    if sample.get("both_sources_active") is not expected_both_sources_active:
        raise BatchVerificationError(
            f"sample {sample_id} both_sources_active differs from AudioProgram events"
        )
    active_slots = {
        endpoint_to_slot[event["source_endpoint_id"]] for event in program["events"]
    }
    return {
        "program_id": program_id,
        "revision": revision,
        "variant_id": variant_id,
        "mode": mode,
        "active_source_slots": sorted(active_slots),
        "both_sources_active": expected_both_sources_active,
    }


def _finite_paths(value: Any, *, owner: str) -> np.ndarray:
    paths = np.asarray(value, dtype=np.float64)
    if paths.shape != (2, FRAME_COUNT, 3) or not np.all(np.isfinite(paths)):
        raise BatchVerificationError(f"{owner} must be finite [2,75,3]")
    return np.ascontiguousarray(paths)


def _same_orientation(first: np.ndarray, second: np.ndarray) -> bool:
    return (
        first.shape == (4,)
        and second.shape == (4,)
        and np.all(np.isfinite(first))
        and np.all(np.isfinite(second))
        and np.isclose(
            abs(float(np.dot(first, second))),
            1.0,
            rtol=0.0,
            atol=1.0e-9,
        )
    )


def _verify_sensor_rig_closure(
    *, plan_root: Path, batch_root: Path
) -> dict[str, Any]:
    """Verify the optional plan -> cache evidence -> batch rig closure."""

    rir_plan = _json(plan_root / "rir_job_plan.json")
    pose_mode = rir_plan.get("listener_pose_mode", "fixed")
    if pose_mode not in {"fixed", "per_episode_frame"}:
        raise BatchVerificationError(
            f"unsupported RIR listener_pose_mode: {pose_mode!r}"
        )
    plan_sidecar = plan_root / "sensor_rig_trajectory.json"
    batch_sidecar = batch_root / "labels" / "sensor_rig_trajectory.json"
    plan_has_sidecar = plan_sidecar.exists() or plan_sidecar.is_symlink()
    batch_has_sidecar = batch_sidecar.exists() or batch_sidecar.is_symlink()
    delivery = _json(batch_root / "delivery.json")
    samples_value = _json(batch_root / "samples.json").get("samples")
    episodes_value = _json(batch_root / "episodes.json").get("episodes")
    samples = samples_value if isinstance(samples_value, list) else []
    episodes = episodes_value if isinstance(episodes_value, list) else []

    if not plan_has_sidecar:
        if pose_mode == "per_episode_frame":
            raise BatchVerificationError(
                "per_episode_frame RIR plan lacks sensor_rig_trajectory.json"
            )
        if (
            batch_has_sidecar
            or delivery.get("sensor_rig_trajectory") is not None
            or any(
                isinstance(row, Mapping)
                and row.get("sensor_rig_trajectory") is not None
                for row in (*samples, *episodes)
            )
        ):
            raise BatchVerificationError(
                "legacy fixed plan has an unbound batch sensor-rig declaration"
            )
        return {
            "listener_pose_mode": "fixed",
            "binding": None,
            "checked_cached_use_count": 0,
            "compatibility": "legacy_fixed_plan_without_sidecar",
        }

    if not plan_sidecar.is_file():
        raise BatchVerificationError(
            "plan sensor_rig_trajectory.json is not a regular file"
        )
    trajectory = _json(plan_sidecar)
    try:
        binding = m7_sensor_rig_binding(trajectory)
        poses = m7_sensor_rig_pose_series(trajectory)
        alignment = validate_m7_rir_listener_alignment(
            rir_job_plan=rir_plan,
            sensor_rig_trajectory=trajectory,
        )
    except M7SensorRigError as exc:
        raise BatchVerificationError(
            f"plan sensor-rig/RIR alignment is invalid: {exc}"
        ) from exc
    if not batch_has_sidecar or not batch_sidecar.is_file():
        raise BatchVerificationError("batch sensor-rig sidecar is missing")
    batch_trajectory = _json(batch_sidecar)
    if (
        batch_sidecar.read_bytes() != plan_sidecar.read_bytes()
        or batch_trajectory != trajectory
        or m7_sensor_rig_binding(batch_trajectory) != binding
    ):
        raise BatchVerificationError(
            "batch sensor-rig sidecar differs from the plan"
        )
    outputs = delivery.get("outputs")
    if (
        delivery.get("sensor_rig_trajectory") != binding
        or delivery.get("sensor_rig_rir_alignment") != alignment
        or not isinstance(outputs, Mapping)
        or outputs.get("sensor_rig_trajectory")
        != "labels/sensor_rig_trajectory.json"
    ):
        raise BatchVerificationError(
            "batch delivery sensor-rig binding differs from the plan"
        )
    for owner, rows in (("sample", samples), ("episode", episodes)):
        for row in rows:
            if (
                not isinstance(row, Mapping)
                or row.get("sensor_rig_trajectory") != binding
            ):
                raise BatchVerificationError(
                    f"batch {owner} sensor-rig binding differs from the plan"
                )

    raw_jobs = rir_plan.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise BatchVerificationError("RIR plan has no jobs for sensor-rig verification")
    jobs_by_id: dict[str, Mapping[str, Any]] = {}
    jobs_by_use: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for job in raw_jobs:
        job_id = job.get("job_id") if isinstance(job, Mapping) else None
        uses = job.get("uses") if isinstance(job, Mapping) else None
        if (
            not isinstance(job_id, str)
            or job_id in jobs_by_id
            or not isinstance(uses, list)
            or not uses
        ):
            raise BatchVerificationError(
                "RIR plan jobs are invalid for sensor-rig verification"
            )
        jobs_by_id[job_id] = job
        for use in uses:
            if not isinstance(use, Mapping):
                raise BatchVerificationError("RIR plan use is malformed")
            episode_id = use.get("episode_id")
            source_slot = use.get("source_slot_id")
            frame_index = use.get("frame_index")
            if (
                not isinstance(episode_id, str)
                or not episode_id
                or source_slot not in SOURCE_SLOTS
                or isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or not 0 <= frame_index < len(poses.pose_hashes)
            ):
                raise BatchVerificationError(
                    "RIR plan use mapping is invalid"
                )
            key = (episode_id, source_slot, frame_index)
            if key in jobs_by_use:
                raise BatchVerificationError("RIR plan use mapping is duplicated")
            jobs_by_use[key] = job

    checked_cached_uses = 0
    for episode in episodes:
        assert isinstance(episode, Mapping)
        episode_id = episode.get("episode_id")
        cache = episode.get("rir_cache")
        cached_jobs = cache.get("jobs") if isinstance(cache, Mapping) else None
        if not isinstance(episode_id, str) or not isinstance(cached_jobs, list):
            raise BatchVerificationError(
                "episode cache evidence is invalid for sensor-rig verification"
            )
        for cached in cached_jobs:
            if not isinstance(cached, Mapping):
                raise BatchVerificationError("cached RIR use is malformed")
            slot = cached.get("source_slot_id")
            frame = cached.get("visual_frame_index")
            job_id = cached.get("job_id")
            if (
                slot not in SOURCE_SLOTS
                or isinstance(frame, bool)
                or not isinstance(frame, int)
                or not 0 <= frame < len(poses.pose_hashes)
                or not isinstance(job_id, str)
            ):
                raise BatchVerificationError("cached RIR use identity is invalid")
            planned = jobs_by_use.get((episode_id, slot, frame))
            if planned is None or jobs_by_id.get(job_id) is not planned:
                raise BatchVerificationError(
                    "cached RIR use differs from its episode/source/frame plan mapping"
                )
            planned_source = np.asarray(
                planned.get("source_position_m"), dtype=np.float64
            )
            planned_listener = np.asarray(
                planned.get(
                    "listener_position_m",
                    rir_plan.get("listener_position_m"),
                ),
                dtype=np.float64,
            )
            planned_orientation = np.asarray(
                planned.get(
                    "listener_orientation_wxyz",
                    rir_plan.get("listener_orientation_wxyz"),
                ),
                dtype=np.float64,
            )
            cached_listener = np.asarray(
                cached.get("listener_position_m"), dtype=np.float64
            )
            cached_orientation = np.asarray(
                cached.get("listener_orientation_wxyz"), dtype=np.float64
            )
            expected_listener = poses.positions_m[frame]
            expected_orientation = poses.orientations_wxyz[frame]
            if (
                planned_source.shape != (3,)
                or not np.all(np.isfinite(planned_source))
                or planned_listener.shape != (3,)
                or cached_listener.shape != (3,)
                or not np.allclose(
                    planned_listener,
                    expected_listener,
                    rtol=0.0,
                    atol=1.0e-9,
                )
                or not np.allclose(
                    cached_listener,
                    expected_listener,
                    rtol=0.0,
                    atol=1.0e-9,
                )
                or not _same_orientation(
                    planned_orientation, expected_orientation
                )
                or not _same_orientation(
                    cached_orientation, expected_orientation
                )
            ):
                raise BatchVerificationError(
                    "cached RIR Listener pose differs from plan/trajectory"
                )
            state_sha256 = rir_acoustic_state_sha256(
                planned_source,
                planned_listener,
                planned_orientation,
            )
            if (
                planned.get("acoustic_state_sha256", state_sha256)
                != state_sha256
                or cached.get("acoustic_state_sha256") != state_sha256
            ):
                raise BatchVerificationError(
                    "cached RIR acoustic-state identity differs from plan/trajectory"
                )
            checked_cached_uses += 1
    if checked_cached_uses != len(jobs_by_use):
        raise BatchVerificationError(
            "batch cache evidence does not cover every sensor-rig-bound RIR use"
        )
    return {
        "listener_pose_mode": pose_mode,
        "binding": binding,
        "rir_alignment": alignment,
        "checked_cached_use_count": checked_cached_uses,
    }


def _bank(plan_root: Path) -> tuple[dict[str, int], np.ndarray, Mapping[str, Any]]:
    record = _json(plan_root / "trajectory_bank.json")
    episode_count = record.get("episode_count")
    if (
        isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count < 1
        or record.get("frame_count") != FRAME_COUNT
    ):
        raise BatchVerificationError("trajectory bank count or frame count differs")
    arrays = np.load(plan_root / "trajectory_bank.npz", allow_pickle=False)
    source_slots = tuple(str(value) for value in arrays["source_slot_ids"])
    episode_ids = tuple(str(value) for value in arrays["episode_ids"])
    centers = np.asarray(arrays["source_center_paths_m"], dtype=np.float64)
    expected_shape = (len(episode_ids), 2, FRAME_COUNT, 3)
    if source_slots != SOURCE_SLOTS or centers.shape != expected_shape:
        raise BatchVerificationError("trajectory bank arrays differ from M7 layout")
    if len(episode_ids) != len(set(episode_ids)) or not np.all(np.isfinite(centers)):
        raise BatchVerificationError("trajectory bank episode IDs or centers are invalid")
    if len(episode_ids) != episode_count:
        raise BatchVerificationError("trajectory bank record and arrays differ")
    return {item: index for index, item in enumerate(episode_ids)}, centers, record


def _assert_center_gate(plan_root: Path, episode_ids: Sequence[str]) -> dict[str, float]:
    gate = _json(plan_root / "navmesh_center_gate.json")
    rows = gate.get("sources")
    if gate.get("status") != "pass" or not isinstance(rows, Mapping):
        raise BatchVerificationError("asset-bound navmesh center gate is invalid")
    minima: dict[str, float] = {}
    for episode_id in episode_ids:
        for slot in SOURCE_SLOTS:
            value = rows.get(f"{episode_id}::{slot}")
            if not isinstance(value, Mapping) or value.get("status") != "pass":
                raise BatchVerificationError(f"center gate fails for {episode_id} {slot}")
            clearance = float(value.get("minimum_navmesh_clearance_m"))
            if not np.isfinite(clearance) or clearance < 0.0:
                raise BatchVerificationError("center gate clearance is invalid")
            minima[f"{episode_id}::{slot}"] = clearance
    return minima


def _assert_rir_uses(plan_root: Path, indices: Mapping[str, int], centers: np.ndarray) -> dict[str, int]:
    plan = _json(plan_root / "rir_job_plan.json")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or plan.get("unique_rir_job_count") != len(jobs):
        raise BatchVerificationError("RIR job plan is malformed")
    uses_by_episode: Counter[str] = Counter()
    use_count = 0
    for row in jobs:
        if not isinstance(row, Mapping):
            raise BatchVerificationError("RIR job is not an object")
        position = np.asarray(row.get("source_position_m"), dtype=np.float64)
        uses = row.get("uses")
        if position.shape != (3,) or not np.all(np.isfinite(position)) or not isinstance(uses, list) or not uses:
            raise BatchVerificationError("RIR job omits a finite source position or use")
        for use in uses:
            if not isinstance(use, Mapping):
                raise BatchVerificationError("RIR job use is malformed")
            episode_id = use.get("episode_id")
            slot = use.get("source_slot_id")
            frame = use.get("frame_index")
            if (
                not isinstance(episode_id, str)
                or episode_id not in indices
                or slot not in SOURCE_SLOTS
                or isinstance(frame, bool)
                or not isinstance(frame, int)
                or not 0 <= frame < FRAME_COUNT
            ):
                raise BatchVerificationError("RIR job use has an invalid episode/source/frame")
            slot_index = SOURCE_SLOTS.index(slot)
            error = float(np.max(np.abs(position - centers[indices[episode_id], slot_index, frame])))
            if error > 1.0e-9:
                raise BatchVerificationError(
                    f"RIR job source position differs from final emitter center by {error:.3g} m"
                )
            uses_by_episode[episode_id] += 1
            use_count += 1
    if set(uses_by_episode) != set(indices) or any(count != 50 for count in uses_by_episode.values()):
        raise BatchVerificationError("each selected route must retain 25 RIR keyframes per source")
    return {"unique_jobs": len(jobs), "uses": use_count, "uses_per_episode": 50}


def _binding_assets(plan_root: Path) -> dict[str, dict[str, str]]:
    record = _json(plan_root / "asset_emitter_binding_report.json")
    rows = record.get("scenarios")
    if record.get("status") != "pass" or not isinstance(rows, list):
        raise BatchVerificationError("asset-emitter binding report is invalid")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("output_episode_id"), str):
            raise BatchVerificationError("asset-emitter binding scenario is invalid")
        report = row.get("binding_report")
        bindings = report.get("bindings") if isinstance(report, Mapping) else None
        if not isinstance(bindings, list) or len(bindings) != 2:
            raise BatchVerificationError("asset-emitter binding omits a source")
        assets: dict[str, str] = {}
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise BatchVerificationError("asset-emitter binding is malformed")
            slot, asset = binding.get("source_slot_id"), binding.get("asset_id")
            if slot not in SOURCE_SLOTS or slot in assets or not isinstance(asset, str) or not asset:
                raise BatchVerificationError("asset-emitter binding has an invalid asset")
            assets[slot] = asset
        result[row["output_episode_id"]] = assets
    if not result:
        raise BatchVerificationError("asset-emitter binding report is empty")
    return result


def _assert_episode_cache_index(
    *, plan_root: Path, batch_root: Path, expected_assets: Mapping[str, Mapping[str, str]]
) -> dict[str, int]:
    plan = _json(plan_root / "rir_job_plan.json")
    plan_rows = plan.get("jobs")
    index = _json(batch_root / "episodes.json")
    episodes = index.get("episodes")
    if not isinstance(plan_rows, list) or not isinstance(episodes, list):
        raise BatchVerificationError("RIR plan or episode cache index is malformed")
    jobs_by_id = {
        row.get("job_id"): row
        for row in plan_rows
        if isinstance(row, Mapping) and isinstance(row.get("job_id"), str)
    }
    if len(jobs_by_id) != len(plan_rows):
        raise BatchVerificationError("RIR plan repeats or omits a job ID")
    rows_by_episode = {
        row.get("episode_id"): row
        for row in episodes
        if isinstance(row, Mapping) and isinstance(row.get("episode_id"), str)
    }
    if set(rows_by_episode) != set(expected_assets):
        raise BatchVerificationError("episode cache index differs from selected asset-bound episodes")
    jobs_checked = 0
    for episode_id, row in rows_by_episode.items():
        if row.get("asset_ids_by_source_slot") != expected_assets[episode_id]:
            raise BatchVerificationError("episode cache index has a wrong asset binding")
        cache = row.get("rir_cache")
        jobs = cache.get("jobs") if isinstance(cache, Mapping) else None
        if not isinstance(jobs, list) or len(jobs) != 50:
            raise BatchVerificationError("each episode cache index must contain 50 RIR uses")
        seen: set[tuple[str, int]] = set()
        for cached in jobs:
            if not isinstance(cached, Mapping):
                raise BatchVerificationError("episode cache job is malformed")
            job_id = cached.get("job_id")
            slot = cached.get("source_slot_id")
            frame = cached.get("visual_frame_index")
            planned = jobs_by_id.get(job_id)
            if (
                not isinstance(job_id, str)
                or slot not in SOURCE_SLOTS
                or isinstance(frame, bool)
                or not isinstance(frame, int)
                or not isinstance(planned, Mapping)
            ):
                raise BatchVerificationError("episode cache job has an invalid ID/source/frame")
            position = np.asarray(cached.get("source_position_m"), dtype=np.float64)
            planned_position = np.asarray(planned.get("source_position_m"), dtype=np.float64)
            if (
                position.shape != (3,)
                or planned_position.shape != (3,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(planned_position))
                or np.max(np.abs(position - planned_position)) > 1.0e-9
                or not any(
                    use.get("episode_id") == episode_id
                    and use.get("source_slot_id") == slot
                    and use.get("frame_index") == frame
                    for use in planned.get("uses", ())
                    if isinstance(use, Mapping)
                )
            ):
                raise BatchVerificationError("episode cache job differs from its RIR plan use")
            key = (slot, frame)
            if key in seen:
                raise BatchVerificationError("episode cache repeats one source/frame RIR use")
            seen.add(key)
            jobs_checked += 1
        if {slot for slot, _ in seen} != set(SOURCE_SLOTS):
            raise BatchVerificationError("episode cache omits one source slot")
    return {"episode_count": len(rows_by_episode), "checked_cache_uses": jobs_checked}


def _verify_batch(
    batch_root: Path,
    *,
    expected_assets: Mapping[str, Mapping[str, str]],
    expected_sample_count: int = 1_000,
) -> dict[str, Any]:
    if (
        isinstance(expected_sample_count, bool)
        or not isinstance(expected_sample_count, int)
        or expected_sample_count < 1
    ):
        raise BatchVerificationError(
            "expected_sample_count must be a positive integer"
        )
    delivery = _json(batch_root / "delivery.json")
    samples_record = _json(batch_root / "samples.json")
    episodes_record = _json(batch_root / "episodes.json")
    samples = samples_record.get("samples")
    episodes = episodes_record.get("episodes")
    (
        acoustic_selection_binding,
        acoustic_selection_binding_sha256,
    ) = _validated_acoustic_selection_binding(
        delivery.get("acoustic_selection_binding")
    )
    episode_count = delivery.get("episode_count")
    variants_per_episode = delivery.get("variants_per_episode")
    has_audio_program_samples = isinstance(samples, list) and any(
        isinstance(row, Mapping) and bool(_program_sample_fields(row))
        for row in samples
    )
    if (
        delivery.get("status") != "pass"
        or samples_record.get("acoustic_selection_binding")
        != acoustic_selection_binding
        or episodes_record.get("acoustic_selection_binding")
        != acoustic_selection_binding
        or delivery.get("sample_count") != expected_sample_count
        or samples_record.get("status") != "pass"
        or episodes_record.get("status") != "pass"
        or samples_record.get("sample_count") != expected_sample_count
        or isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count != len(expected_assets)
        or isinstance(variants_per_episode, bool)
        or not isinstance(variants_per_episode, int)
        or variants_per_episode < 1
        or episode_count * variants_per_episode != expected_sample_count
        or (
            not has_audio_program_samples
            and delivery.get("both_sources_active") is not True
        )
        or not isinstance(samples, list)
        or len(samples) != expected_sample_count
        or not isinstance(episodes, list)
        or len(episodes) != episode_count
    ):
        raise BatchVerificationError(
            "batch delivery summary differs from the expected M7 sample count"
        )
    for episode in episodes:
        cache = episode.get("rir_cache") if isinstance(episode, Mapping) else None
        if (
            not isinstance(episode, Mapping)
            or "acoustic_selection_binding_sha256" not in episode
            or episode.get("acoustic_selection_binding_sha256")
            != acoustic_selection_binding_sha256
            or not isinstance(cache, Mapping)
            or cache.get("acoustic_selection_binding")
            != acoustic_selection_binding
        ):
            raise BatchVerificationError(
                "episode cache acoustic selection binding differs from the batch"
            )
    audio_root = batch_root / "audio" / "binaural"
    sample_ids: set[str] = set()
    variants: defaultdict[str, set[int]] = defaultdict(set)
    assets_by_episode: dict[str, Mapping[str, str]] = {}
    maximum_peak = 0.0
    audio_program_sample_count = 0
    audio_program_mode_counts: Counter[str] = Counter()
    sample_activity_flags: list[bool] = []
    for row in samples:
        if not isinstance(row, Mapping):
            raise BatchVerificationError("sample record is not an object")
        sample_id = row.get("sample_id")
        episode_id = row.get("episode_id")
        variant = row.get("variant_index")
        assets = row.get("asset_ids_by_source_slot")
        audio = row.get("audio")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not isinstance(episode_id, str)
            or episode_id not in expected_assets
            or isinstance(variant, bool)
            or not isinstance(variant, int)
            or not 0 <= variant < variants_per_episode
            or not isinstance(assets, Mapping)
            or dict(assets) != dict(expected_assets[episode_id])
            or not isinstance(audio, Mapping)
            or "acoustic_selection_binding_sha256" not in row
            or row.get("acoustic_selection_binding_sha256")
            != acoustic_selection_binding_sha256
        ):
            raise BatchVerificationError("sample identity/binding differs from the plan")
        program_instance = _audio_program_instance(
            batch_root=batch_root,
            sample=row,
            sample_id=sample_id,
            asset_ids_by_source_slot=assets,
        )
        if program_instance is None:
            if row.get("both_sources_active") is not True:
                raise BatchVerificationError(
                    "legacy sample must keep both_sources_active=true"
                )
        else:
            audio_program_sample_count += 1
            audio_program_mode_counts[str(program_instance["mode"])] += 1
        sample_activity_flags.append(bool(row["both_sources_active"]))
        if assets_by_episode.setdefault(episode_id, assets) != assets:
            raise BatchVerificationError("one episode has inconsistent asset bindings")
        mixture = audio.get("mixture")
        if (
            audio.get("channel_count") != 2
            or audio.get("sample_rate_hz") != SAMPLE_RATE_HZ
            or audio.get("sample_count") != SAMPLE_COUNT
            or audio.get("layout") != "native_RLR_HRTF_binaural_left_right"
            or audio.get("mixture_is_exact_stem_sum_before_delivery") is not True
            or not isinstance(mixture, Mapping)
            or not isinstance(mixture.get("path"), str)
            or not isinstance(mixture.get("audio_sha256"), str)
            or not isinstance(mixture.get("sidecar_path"), str)
        ):
            raise BatchVerificationError("sample audio contract is invalid")
        wav = audio_root / str(mixture["path"])
        sidecar = audio_root / str(mixture["sidecar_path"])
        if sha256_file(wav) != mixture["audio_sha256"]:
            raise BatchVerificationError(f"sample WAV hash differs: {sample_id}")
        if (
            not isinstance(mixture.get("sidecar_sha256"), str)
            or sha256_file(sidecar) != mixture["sidecar_sha256"]
        ):
            raise BatchVerificationError(f"sample sidecar hash differs: {sample_id}")
        rendered = read_float32_wav(wav, sidecar_path=sidecar, verify_sidecar=True)
        sidecar_metadata = (
            rendered.sidecar.get("metadata")
            if isinstance(rendered.sidecar, Mapping)
            else None
        )
        if (
            not isinstance(sidecar_metadata, Mapping)
            or "acoustic_selection_binding_sha256" not in sidecar_metadata
            or sidecar_metadata.get("acoustic_selection_binding_sha256")
            != acoustic_selection_binding_sha256
        ):
            raise BatchVerificationError(
                f"sample sidecar acoustic selection differs: {sample_id}"
            )
        if rendered.sample_rate_hz != SAMPLE_RATE_HZ or rendered.samples.shape != (2, SAMPLE_COUNT):
            raise BatchVerificationError(f"sample WAV header differs: {sample_id}")
        sample_ids.add(sample_id)
        variants[episode_id].add(variant)
        maximum_peak = max(maximum_peak, float(np.max(np.abs(rendered.samples))))
    if set(variants) != set(expected_assets) or any(
        value != set(range(variants_per_episode)) for value in variants.values()
    ):
        raise BatchVerificationError(
            "each selected route must have every declared variant exactly once"
        )
    if has_audio_program_samples and delivery.get("both_sources_active") is not all(
        sample_activity_flags
    ):
        raise BatchVerificationError(
            "batch both_sources_active summary differs from its samples"
        )
    if has_audio_program_samples and audio_program_sample_count != len(samples):
        raise BatchVerificationError(
            "legacy and AudioProgram-bound samples may not be mixed in one batch"
        )
    if has_audio_program_samples and variants_per_episode != 1:
        raise BatchVerificationError(
            "AudioProgram batches require one program instance per visual episode"
        )
    result = {
        "sample_count": len(sample_ids),
        "episode_count": len(variants),
        "variants_per_episode": variants_per_episode,
        "maximum_readback_peak_absolute": maximum_peak,
        "acoustic_selection_binding": acoustic_selection_binding,
        "acoustic_selection_binding_sha256": (
            acoustic_selection_binding_sha256
        ),
    }
    if audio_program_sample_count:
        result["audio_program_instance_sample_count"] = audio_program_sample_count
        result["audio_program_mode_counts"] = dict(
            sorted(audio_program_mode_counts.items())
        )
    return result


def _compare_reference(batch_root: Path, reference_root: Path) -> Mapping[str, Any]:
    current_record = _json(batch_root / "samples.json")
    reference_record = _json(reference_root / "samples.json")
    if current_record.get(
        "acoustic_selection_binding"
    ) != reference_record.get("acoustic_selection_binding"):
        raise BatchVerificationError(
            "reference batch acoustic selection binding differs"
        )
    current = current_record.get("samples")
    reference = reference_record.get("samples")
    if not isinstance(current, list) or not isinstance(reference, list):
        raise BatchVerificationError("reference batch samples are malformed")
    current_rows = {str(row.get("sample_id")): row for row in current if isinstance(row, Mapping)}
    reference_rows = {str(row.get("sample_id")): row for row in reference if isinstance(row, Mapping)}
    shared = sorted(set(current_rows) & set(reference_rows))
    if not shared:
        raise BatchVerificationError("reference batch shares no sample IDs")
    equal = 0
    for sample_id in shared:
        first = current_rows[sample_id]
        second = reference_rows[sample_id]
        first_audio = first.get("audio")
        second_audio = second.get("audio")
        first_mix = first_audio.get("mixture") if isinstance(first_audio, Mapping) else None
        second_mix = second_audio.get("mixture") if isinstance(second_audio, Mapping) else None
        if not isinstance(first_mix, Mapping) or not isinstance(second_mix, Mapping):
            raise BatchVerificationError("reference sample lacks mixture metadata")
        first_path = batch_root / "audio" / "binaural" / str(first_mix["path"])
        second_path = reference_root / "audio" / "binaural" / str(second_mix["path"])
        if first_path.read_bytes() != second_path.read_bytes():
            raise BatchVerificationError(f"reference byte mismatch: {sample_id}")
        equal += 1
    return {"shared_sample_count": len(shared), "byte_identical_sample_count": equal}


def verify(
    *,
    plan_root: Path,
    batch_root: Path,
    output: Path,
    reference_batch_root: Path | None = None,
    expected_sample_count: int = 1_000,
) -> Path:
    """Write one pass report after inspecting every batch audio artifact."""

    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise BatchVerificationError(f"refusing to replace report: {output}")
    plan_root = plan_root.resolve()
    batch_root = batch_root.resolve()
    sensor_rig = _verify_sensor_rig_closure(
        plan_root=plan_root,
        batch_root=batch_root,
    )
    indices, centers, bank_record = _bank(plan_root)
    clearances = _assert_center_gate(plan_root, tuple(indices))
    rir = _assert_rir_uses(plan_root, indices, centers)
    assets = _binding_assets(plan_root)
    if set(assets) != set(indices):
        raise BatchVerificationError("bound asset episode IDs differ from the selected bank")
    batch = _verify_batch(
        batch_root,
        expected_assets=assets,
        expected_sample_count=expected_sample_count,
    )
    episode_cache = _assert_episode_cache_index(
        plan_root=plan_root,
        batch_root=batch_root,
        expected_assets=assets,
    )
    reference = (
        None
        if reference_batch_root is None
        else _compare_reference(batch_root, reference_batch_root.resolve())
    )
    motion_counts = Counter(str(row["motion_case"]) for row in bank_record["episodes"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": "artifact-level throughput correctness only; no new native RLR execution, visual render, or dataset admission",
        "plan_root": str(plan_root),
        "batch_root": str(batch_root),
        "trajectory_bank": {
            "selected_episode_count": len(indices),
            "frame_count": FRAME_COUNT,
            "motion_case_counts": dict(sorted(motion_counts.items())),
            "minimum_source_center_clearance_m": min(clearances.values()),
        },
        "rir_plan": rir,
        "sensor_rig_trajectory": sensor_rig,
        "episode_cache_index": episode_cache,
        "acoustic_selection_binding": batch[
            "acoustic_selection_binding"
        ],
        "batch": batch,
        "reference_byte_comparison": reference,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-batch-root", type=Path)
    parser.add_argument("--expected-sample-count", type=int, default=1_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify(
        plan_root=args.plan_root,
        batch_root=args.batch_root,
        output=args.output,
        reference_batch_root=args.reference_batch_root,
        expected_sample_count=args.expected_sample_count,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
