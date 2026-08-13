"""Persistent native-RLR rendering into a resumable room-impulse cache."""

from __future__ import annotations

import hashlib
import importlib
from importlib.machinery import ExtensionFileLoader
import json
import math
import os
import resource
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m3.runtime import (
    CompiledAcousticScene,
    RuntimeContractError,
    _native_configuration,
    _upload_report,
    _verify_upload_report,
    load_habitat_runtime,
)
from avengine.m3.rlr_material_import import _validate_rlr_document
from avengine.m4.runtime import (
    BINAURAL_LAYOUT_ID,
    FOA_LAYOUT_ID,
    M4SimulationConfig,
    _native_layout,
    simulation_with_layout,
)
from avengine.m6x.room_feasibility import (
    RIR_JOB_PLAN_SCHEMA,
    SOURCE_SLOTS,
    rir_acoustic_state_sha256,
)
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


RIR_CACHE_REQUEST_SCHEMA = "avengine_rlr_rir_cache_request_v1"
RIR_CACHE_INDEX_SCHEMA = "avengine_rlr_rir_cache_index_v1"
RIR_CACHE_RECEIPT_SCHEMA = "avengine_rlr_rir_cache_receipt_v1"
RIR_CACHE_TIMING_SCHEMA = "avengine_rlr_rir_cache_timing_v1"
RIR_CACHE_ACOUSTIC_SELECTION_INPUT_SCHEMA = (
    "avengine_rir_cache_acoustic_selection_v1"
)
RIR_CACHE_ACOUSTIC_SELECTION_BINDING_SCHEMA = (
    "avengine_rir_cache_acoustic_selection_binding_v1"
)
RIR_CACHE_ACOUSTIC_SELECTION_SIDECAR_SCHEMA = (
    "avengine_rir_cache_acoustic_selection_sidecar_v1"
)
RIR_CACHE_ACOUSTIC_SELECTION_NAME = "acoustic_selection.json"
SEMANTIC_RIR_CACHE_REQUEST_SCHEMA = "avengine_semantic_rir_cache_request_v1"
SEMANTIC_RIR_CACHE_INDEX_SCHEMA = "avengine_semantic_rir_cache_index_v1"
SEMANTIC_RIR_CACHE_RECEIPT_SCHEMA = "avengine_semantic_rir_cache_receipt_v1"


class RIRCacheError(RuntimeContractError):
    """A cache request, native response, or retained shard is invalid."""


@dataclass(frozen=True)
class RIRBatchResult:
    """One native simulation call copied out of the persistent context."""

    samples: tuple[np.ndarray, ...]
    sample_rate_hz: int
    layout_id: str
    channel_labels: tuple[str, ...]
    indirect_ray_efficiency: float
    wall_seconds: float
    process_cpu_seconds: float


@dataclass(frozen=True)
class RIRCacheResult:
    output: Path
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class SemanticAcousticScene:
    """Decoded native scene selected by paths and semantic structure only."""

    manifest_path: Path
    package_id: str
    material_database_bytes: bytes
    material_categories: tuple[str, ...]
    material_name_by_category: Mapping[str, str]
    material_index_by_category: Mapping[str, int]
    objects: tuple[Mapping[str, Any], ...]
    triangle_count_by_material: Mapping[str, int]


@dataclass(frozen=True)
class CachedRIREpisode:
    """One episode's source-slot RIR grid reopened from a retained cache."""

    samples: np.ndarray
    lengths: np.ndarray
    source_slot_ids: tuple[str, ...]
    visual_frame_indices: tuple[int, ...]
    keyframe_samples: tuple[int, ...]
    sample_rate_hz: int
    layout_type: str
    layout_id: str
    channel_labels: tuple[str, ...]
    evidence: Mapping[str, Any]


def _finite_vector(value: Any, length: int, *, owner: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise RIRCacheError(f"{owner} must contain {length} finite numbers")
    return tuple(float(item) for item in value)


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RIRCacheError(f"{owner} must be a positive integer")
    return value


def _unit_orientation(value: Any, *, owner: str) -> tuple[float, ...]:
    orientation = _finite_vector(value, 4, owner=owner)
    if not math.isclose(
        math.sqrt(sum(component * component for component in orientation)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RIRCacheError(f"{owner} must be unit normalized")
    return orientation


def validate_rir_job_plan(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Validate and normalize the source/Listener states consumed by RLR.

    Historical fixed-listener v2 plans stored one pose at the plan root.  They
    remain readable and are normalized into an explicit repeated pose on every
    returned job.  New plans already carry those fields per job and may vary
    the Listener pose by episode/frame.
    """

    if not isinstance(value, Mapping) or value.get("schema") != RIR_JOB_PLAN_SCHEMA:
        raise RIRCacheError(f"RIR plan schema must be {RIR_JOB_PLAN_SCHEMA}")
    if value.get("status") != "planned_not_run":
        raise RIRCacheError("RIR plan must have status planned_not_run")
    pose_mode = value.get("listener_pose_mode")
    legacy_fixed = pose_mode is None
    if pose_mode not in {None, "fixed", "per_episode_frame"}:
        raise RIRCacheError("RIR plan listener_pose_mode is invalid")
    fixed_listener: tuple[float, ...] | None = None
    fixed_orientation: tuple[float, ...] | None = None
    if legacy_fixed or pose_mode == "fixed":
        fixed_listener = _finite_vector(
            value.get("listener_position_m"), 3, owner="listener position"
        )
        fixed_orientation = _unit_orientation(
            value.get("listener_orientation_wxyz"),
            owner="listener orientation",
        )
    elif (
        value.get("listener_position_m") is not None
        or value.get("listener_orientation_wxyz") is not None
    ):
        raise RIRCacheError(
            "per-episode Listener plans cannot declare one fixed top-level pose"
        )
    if not legacy_fixed and value.get("cache_key_fields") != [
        "source_position_m",
        "listener_position_m",
        "listener_orientation_wxyz",
    ]:
        raise RIRCacheError("RIR plan cache key fields do not bind Listener pose")

    raw_jobs = value.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise RIRCacheError("RIR plan jobs must be a nonempty list")
    jobs: list[dict[str, Any]] = []
    ids: set[str] = set()
    acoustic_states: set[tuple[float, ...]] = set()
    all_uses: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(raw_jobs):
        if not isinstance(raw, Mapping):
            raise RIRCacheError(f"RIR job {index} must be an object")
        job_id = raw.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RIRCacheError(f"RIR job {index} lacks a stable job_id")
        if job_id in ids:
            raise RIRCacheError(f"duplicate RIR job_id: {job_id}")
        position = _finite_vector(
            raw.get("source_position_m"),
            3,
            owner=f"RIR job {job_id} source position",
        )
        raw_listener = raw.get("listener_position_m")
        raw_orientation = raw.get("listener_orientation_wxyz")
        if raw_listener is None and raw_orientation is None and fixed_listener is not None:
            listener = fixed_listener
            orientation = fixed_orientation
        elif raw_listener is None or raw_orientation is None:
            raise RIRCacheError(
                f"RIR job {job_id} must bind both Listener position and orientation"
            )
        else:
            listener = _finite_vector(
                raw_listener,
                3,
                owner=f"RIR job {job_id} listener position",
            )
            orientation = _unit_orientation(
                raw_orientation,
                owner=f"RIR job {job_id} listener orientation",
            )
        assert orientation is not None
        if (legacy_fixed or pose_mode == "fixed") and (
            listener != fixed_listener or orientation != fixed_orientation
        ):
            raise RIRCacheError(
                f"RIR job {job_id} Listener pose differs from fixed plan pose"
            )
        state_key = (*position, *listener, *orientation)
        if state_key in acoustic_states:
            raise RIRCacheError(
                "RIR plan contains duplicate acoustic positions and Listener poses"
            )
        state_sha256 = rir_acoustic_state_sha256(
            position,
            listener,
            orientation,
        )
        declared_state_sha256 = raw.get("acoustic_state_sha256")
        if declared_state_sha256 is not None and declared_state_sha256 != state_sha256:
            raise RIRCacheError(
                f"RIR job {job_id} acoustic-state SHA-256 differs from its pose"
            )
        if not legacy_fixed and declared_state_sha256 is None:
            raise RIRCacheError(
                f"RIR job {job_id} lacks an acoustic-state SHA-256"
            )
        uses = raw.get("uses")
        if not isinstance(uses, list) or not uses:
            raise RIRCacheError(f"RIR job {job_id} has no uses")
        normalized_uses: list[dict[str, Any]] = []
        for use in uses:
            if (
                not isinstance(use, Mapping)
                or use.get("source_slot_id") not in {"source1", "source2"}
                or not isinstance(use.get("episode_id"), str)
                or isinstance(use.get("frame_index"), bool)
                or not isinstance(use.get("frame_index"), int)
                or int(use["frame_index"]) < 0
            ):
                raise RIRCacheError(f"RIR job {job_id} contains an invalid use")
            use_key = (
                str(use["episode_id"]),
                str(use["source_slot_id"]),
                int(use["frame_index"]),
            )
            if use_key in all_uses:
                raise RIRCacheError(
                    "RIR plan maps one episode/source/frame use more than once"
                )
            all_uses.add(use_key)
            normalized_uses.append(dict(use))
        jobs.append(
            {
                "job_id": job_id,
                "acoustic_state_sha256": state_sha256,
                "source_position_m": list(position),
                "listener_position_m": list(listener),
                "listener_orientation_wxyz": list(orientation),
                "uses": normalized_uses,
            }
        )
        ids.add(job_id)
        acoustic_states.add(state_key)
    if value.get("unique_rir_job_count") != len(jobs):
        raise RIRCacheError("RIR plan unique job count differs from jobs")
    return tuple(jobs)


def validate_semantic_rir_job_plan(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate and normalize one complete full75 semantic RIR plan."""

    if not isinstance(value, Mapping) or value.get("schema") != RIR_JOB_PLAN_SCHEMA:
        raise RIRCacheError(f"semantic RIR plan schema must be {RIR_JOB_PLAN_SCHEMA}")
    if value.get("status") != "planned_not_run":
        raise RIRCacheError("semantic RIR plan must have status planned_not_run")
    pose_mode = value.get("listener_pose_mode")
    if pose_mode not in {"fixed", "per_episode_frame"}:
        raise RIRCacheError("semantic RIR listener pose mode is invalid")
    base_fields = {
        "schema",
        "status",
        "listener_pose_mode",
        "cache_key_fields",
        "jobs",
        "unique_rir_job_count",
        "stride_frames",
        "requested_pair_state_count",
    }
    full_fields = {
        "claim_boundary",
        "producer_backend",
        "cache_artifact",
        "source_acoustic_profile",
        "slot_identity_affects_cache_key",
        "dry_audio_independent",
        "unique_listener_pose_count",
        "cache_reuse_count",
    }
    pose_fields = (
        {"listener_position_m", "listener_orientation_wxyz"}
        if pose_mode == "fixed"
        else set()
    )
    observed_fields = frozenset(value)
    minimal_shape = base_fields | pose_fields
    full_shape = minimal_shape | full_fields
    if observed_fields not in {frozenset(minimal_shape), frozenset(full_shape)}:
        raise RIRCacheError(
            "semantic RIR plan fields must be the minimal or full planning shape"
        )
    is_full_shape = observed_fields == full_shape
    fixed_listener: tuple[float, ...] | None = None
    fixed_orientation: tuple[float, ...] | None = None
    if pose_mode == "fixed":
        fixed_listener = _finite_vector(
            value["listener_position_m"], 3, owner="listener position"
        )
        fixed_orientation = _unit_orientation(
            value["listener_orientation_wxyz"], owner="listener orientation"
        )
    if value.get("cache_key_fields") != [
        "source_position_m",
        "listener_position_m",
        "listener_orientation_wxyz",
    ]:
        raise RIRCacheError("semantic RIR plan cache key fields are invalid")
    raw_jobs = value.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise RIRCacheError("semantic RIR plan jobs must be nonempty")
    jobs: list[dict[str, Any]] = []
    ids: set[str] = set()
    states: set[tuple[float, ...]] = set()
    listener_states: set[tuple[float, ...]] = set()
    uses_seen: set[tuple[str, str, int]] = set()
    for ordinal, raw in enumerate(raw_jobs):
        if not isinstance(raw, Mapping):
            raise RIRCacheError(f"semantic RIR job {ordinal} is not an object")
        base_job_fields = {"job_id", "source_position_m", "uses"}
        pose_job_fields = {"listener_position_m", "listener_orientation_wxyz"}
        allowed_job_shapes = {
            frozenset(base_job_fields | pose_job_fields),
            frozenset(base_job_fields | pose_job_fields | {"acoustic_state_sha256"}),
        }
        if pose_mode == "fixed":
            allowed_job_shapes |= {
                frozenset(base_job_fields),
                frozenset(base_job_fields | {"acoustic_state_sha256"}),
            }
        if frozenset(raw) not in allowed_job_shapes:
            raise RIRCacheError(f"semantic RIR job {ordinal} fields are invalid")
        state_id = raw.get("acoustic_state_sha256")
        if state_id is not None and (
            not isinstance(state_id, str)
            or len(state_id) != 64
            or any(character not in "0123456789abcdef" for character in state_id)
        ):
            raise RIRCacheError("semantic RIR acoustic state identity is invalid")
        job_id = raw.get("job_id")
        if not isinstance(job_id, str) or not job_id or job_id in ids:
            raise RIRCacheError("semantic RIR job identity is invalid")
        source = _finite_vector(
            raw.get("source_position_m"), 3, owner=f"semantic job {job_id} source"
        )
        has_listener = "listener_position_m" in raw
        has_orientation = "listener_orientation_wxyz" in raw
        if has_listener != has_orientation:
            raise RIRCacheError(f"semantic job {job_id} listener pose is incomplete")
        if not has_listener:
            if fixed_listener is None or fixed_orientation is None:
                raise RIRCacheError(f"semantic job {job_id} listener pose is missing")
            listener = fixed_listener
            orientation = fixed_orientation
        else:
            listener = _finite_vector(
                raw["listener_position_m"],
                3,
                owner=f"semantic job {job_id} listener",
            )
            orientation = _unit_orientation(
                raw["listener_orientation_wxyz"],
                owner=f"semantic job {job_id} orientation",
            )
        if pose_mode == "fixed" and (
            listener != fixed_listener or orientation != fixed_orientation
        ):
            raise RIRCacheError("semantic fixed-listener plan contains pose drift")
        state = (*source, *listener, *orientation)
        if state in states:
            raise RIRCacheError("semantic RIR plan contains duplicate pose states")
        raw_uses = raw.get("uses")
        if not isinstance(raw_uses, list) or not raw_uses:
            raise RIRCacheError(f"semantic job {job_id} uses are missing")
        uses: list[dict[str, Any]] = []
        for use in raw_uses:
            if (
                not isinstance(use, Mapping)
                or set(use) != {"episode_id", "source_slot_id", "frame_index"}
                or use.get("source_slot_id") not in SOURCE_SLOTS
                or not isinstance(use.get("episode_id"), str)
                or not use["episode_id"]
                or isinstance(use.get("frame_index"), bool)
                or not isinstance(use.get("frame_index"), int)
                or use["frame_index"] < 0
            ):
                raise RIRCacheError(f"semantic job {job_id} use is invalid")
            key = (
                str(use["episode_id"]),
                str(use["source_slot_id"]),
                int(use["frame_index"]),
            )
            if key in uses_seen:
                raise RIRCacheError("semantic RIR plan maps one use more than once")
            uses_seen.add(key)
            uses.append(dict(use))
        jobs.append(
            {
                "job_id": job_id,
                "source_position_m": list(source),
                "listener_position_m": list(listener),
                "listener_orientation_wxyz": list(orientation),
                "uses": uses,
            }
        )
        ids.add(job_id)
        states.add(state)
        listener_states.add((*listener, *orientation))
    unique_job_count = value.get("unique_rir_job_count")
    stride_frames = value.get("stride_frames")
    requested_count = value.get("requested_pair_state_count")
    if (
        isinstance(unique_job_count, bool)
        or not isinstance(unique_job_count, int)
        or unique_job_count != len(jobs)
    ):
        raise RIRCacheError("semantic RIR plan unique job count is invalid")
    if (
        isinstance(stride_frames, bool)
        or not isinstance(stride_frames, int)
        or stride_frames != 1
    ):
        raise RIRCacheError("semantic RIR plan stride must be one frame")
    episode_ids = {episode_id for episode_id, _, _ in uses_seen}
    if len(episode_ids) != 1:
        raise RIRCacheError("semantic RIR plan must contain exactly one episode")
    episode_id = next(iter(episode_ids))
    expected_uses = {
        (episode_id, source_slot_id, frame_index)
        for source_slot_id in SOURCE_SLOTS
        for frame_index in range(75)
    }
    if uses_seen != expected_uses:
        raise RIRCacheError(
            "semantic RIR plan must cover source1/source2 frames 0 through 74 exactly once"
        )
    if (
        isinstance(requested_count, bool)
        or not isinstance(requested_count, int)
        or requested_count != len(expected_uses)
    ):
        raise RIRCacheError("semantic RIR plan requested use count is invalid")
    if is_full_shape:
        unique_listener_count = value.get("unique_listener_pose_count")
        reuse_count = value.get("cache_reuse_count")
        if (
            not isinstance(value.get("claim_boundary"), str)
            or not value["claim_boundary"]
            or value.get("producer_backend") != "RLR Audio Propagation"
            or value.get("cache_artifact") != "room impulse response (RIR)"
            or value.get("source_acoustic_profile") != "omnidirectional_point_source_v1"
            or value.get("slot_identity_affects_cache_key") is not False
            or value.get("dry_audio_independent") is not True
            or isinstance(unique_listener_count, bool)
            or not isinstance(unique_listener_count, int)
            or unique_listener_count != len(listener_states)
            or isinstance(reuse_count, bool)
            or not isinstance(reuse_count, int)
            or reuse_count != len(expected_uses) - len(jobs)
        ):
            raise RIRCacheError("semantic RIR full planning metadata is invalid")
    return tuple(jobs)


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class _NativeRIRBatchRenderer:
    """One scene upload and fixed native endpoint slots reused across batches."""

    def __init__(
        self,
        scene: CompiledAcousticScene,
        simulation: M4SimulationConfig,
        *,
        batch_size: int,
        initial_positions_m: Sequence[Sequence[float]],
        listener_position_m: Sequence[float],
        listener_orientation_wxyz: Sequence[float],
        layout_type: str,
        channel_count: int,
        hrtf_file_path: str,
        source_radius_m: float,
        listener_radius_m: float,
    ) -> None:
        if not isinstance(scene, CompiledAcousticScene):
            raise RIRCacheError("scene must be a CompiledAcousticScene")
        self.batch_size = _positive_int(batch_size, owner="native batch size")
        if len(initial_positions_m) != self.batch_size:
            raise RIRCacheError("initial position count differs from native batch size")
        self.source_ids = tuple(
            f"cache_slot_{index:04d}" for index in range(self.batch_size)
        )
        self.layout_type = layout_type
        self.channel_count = channel_count
        self.selected = simulation_with_layout(
            simulation,
            layout_type=layout_type,
            channel_count=channel_count,
        )
        if self.selected.temporal_coherence:
            raise RIRCacheError("RIR cache requires temporal_coherence=false")
        self.layout_id = (
            BINAURAL_LAYOUT_ID if layout_type == "binaural" else FOA_LAYOUT_ID
        )
        self.channel_labels = (
            ("left", "right") if layout_type == "binaural" else ("W", "Y", "Z", "X")
        )
        if layout_type == "binaural":
            hrtf = Path(hrtf_file_path).resolve()
            if not hrtf.is_file():
                raise RIRCacheError("binaural RIR cache requires a readable HRTF")
            hrtf_file_path = str(hrtf)
        elif hrtf_file_path:
            raise RIRCacheError("HRTF is only valid for a binaural cache")

        setup_wall_start = time.perf_counter()
        setup_cpu_start = time.process_time()
        habitat_module, runtime_report = load_habitat_runtime()
        native_configuration, config_readback = _native_configuration(
            habitat_module, self.selected
        )
        self.context = habitat_module.RLRAcousticContext(native_configuration)
        with tempfile.TemporaryDirectory(prefix="avengine-rir-cache-db-") as temp_dir:
            private_database = Path(temp_dir) / "material_database.json"
            private_database.write_bytes(scene.material_database_bytes)
            if sha256_file(private_database) != scene.material_database_sha256:
                raise RIRCacheError("private material database snapshot changed")
            raw_upload = self.context.load_acoustic_scene(
                str(private_database),
                list(scene.material_categories),
                list(scene.objects),
            )
        upload_report = _upload_report(raw_upload)
        _verify_upload_report(scene, upload_report)
        for source_id, position in zip(
            self.source_ids, initial_positions_m, strict=True
        ):
            self.context.add_source(source_id, position, source_radius_m)
        self.listener_id = "cache_listener0"
        self.context.add_listener(
            self.listener_id,
            listener_position_m,
            listener_orientation_wxyz,
            _native_layout(habitat_module, layout_type),
            channel_count,
            listener_radius_m,
            hrtf_file_path,
        )
        self.current_listener_position_m = _finite_vector(
            listener_position_m, 3, owner="initial listener position"
        )
        self.current_listener_orientation_wxyz = _unit_orientation(
            listener_orientation_wxyz,
            owner="initial listener orientation",
        )
        self.listener_pose_update_count = 0
        self.setup_report = {
            "runtime": runtime_report,
            "configuration_readback": config_readback,
            "upload_report": upload_report,
            "wall_seconds": time.perf_counter() - setup_wall_start,
            "process_cpu_seconds": time.process_time() - setup_cpu_start,
            "peak_rss_bytes": _rss_bytes(),
            "context_policy": {
                "lifetime": "one_persistent_context_per_cache_run",
                "source_slot_count": self.batch_size,
                "scene_upload_count": 1,
                "temporal_coherence": False,
                "compute_device": "CPU",
                "configured_thread_count": self.selected.thread_count,
                "listener_pose_policy": "update_before_each_changed_pose_batch",
            },
        }

    def render(
        self,
        positions_m: Sequence[Sequence[float]],
        *,
        listener_position_m: Sequence[float] | None = None,
        listener_orientation_wxyz: Sequence[float] | None = None,
    ) -> RIRBatchResult:
        if not positions_m or len(positions_m) > self.batch_size:
            raise RIRCacheError("batch position count is outside native slot capacity")
        requested = [
            _finite_vector(value, 3, owner="batch source position")
            for value in positions_m
        ]
        if (listener_position_m is None) != (listener_orientation_wxyz is None):
            raise RIRCacheError(
                "batch Listener position and orientation must be supplied together"
            )
        requested_listener = (
            self.current_listener_position_m
            if listener_position_m is None
            else _finite_vector(
                listener_position_m, 3, owner="batch listener position"
            )
        )
        requested_orientation = (
            self.current_listener_orientation_wxyz
            if listener_orientation_wxyz is None
            else _unit_orientation(
                listener_orientation_wxyz,
                owner="batch listener orientation",
            )
        )
        if (
            requested_listener != self.current_listener_position_m
            or requested_orientation != self.current_listener_orientation_wxyz
        ):
            self.context.set_listener_pose(
                self.listener_id,
                requested_listener,
                requested_orientation,
            )
            self.current_listener_position_m = requested_listener
            self.current_listener_orientation_wxyz = requested_orientation
            self.listener_pose_update_count += 1
        padded = requested + [requested[-1]] * (self.batch_size - len(requested))
        for source_id, position in zip(self.source_ids, padded, strict=True):
            self.context.set_source_position(source_id, position)

        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        raw_irs = self.context.simulate_owned()
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        by_id: dict[str, np.ndarray] = {}
        for raw_ir in raw_irs:
            listener_id = str(getattr(raw_ir, "listener_id", ""))
            source_id = str(getattr(raw_ir, "source_id", ""))
            if listener_id != self.listener_id or source_id not in self.source_ids:
                raise RIRCacheError("native RLR returned an undeclared pair")
            if source_id in by_id:
                raise RIRCacheError("native RLR returned a duplicate pair")
            try:
                sample_rate = float(raw_ir.sample_rate)
                samples = np.array(raw_ir.samples, dtype="<f4", order="C", copy=True)
                sample_count = int(raw_ir.sample_count)
                observed_channels = int(raw_ir.channel_count)
            except (AttributeError, TypeError, ValueError) as exc:
                raise RIRCacheError(
                    f"native RLR returned a malformed IR: {exc}"
                ) from exc
            if not math.isclose(
                sample_rate,
                self.selected.sample_rate_hz,
                rel_tol=1.0e-6,
                abs_tol=1.0e-3,
            ):
                raise RIRCacheError("native RIR sample rate differs from request")
            if (
                observed_channels != self.channel_count
                or samples.shape != (self.channel_count, sample_count)
                or sample_count < 2
                or not samples.flags.c_contiguous
                or not np.all(np.isfinite(samples))
                or not np.any(samples != 0.0)
            ):
                raise RIRCacheError("native RIR payload is invalid")
            by_id[source_id] = samples
        if set(by_id) != set(self.source_ids):
            raise RIRCacheError("native RLR omitted a fixed cache source slot")
        receipts = list(self.context.source_registration_receipts())
        if len(receipts) != self.batch_size:
            raise RIRCacheError("native source receipt count differs from batch size")
        for source_id, position, receipt in zip(
            self.source_ids, padded, receipts, strict=True
        ):
            expected = np.asarray(position, dtype=np.float32).astype(np.float64)
            observed = np.asarray(receipt.position, dtype=np.float64)
            if (
                str(receipt.source_id) != source_id
                or observed.shape != (3,)
                or not np.array_equal(observed, expected)
                or receipt.native_realized is not True
            ):
                raise RIRCacheError("native source receipt differs from cache request")
        listener_receipts = list(self.context.listener_registration_receipts())
        if len(listener_receipts) != 1:
            raise RIRCacheError("native listener receipt count differs from request")
        listener_receipt = listener_receipts[0]
        expected_listener = np.asarray(
            requested_listener, dtype=np.float32
        ).astype(np.float64)
        expected_orientation = np.asarray(
            requested_orientation, dtype=np.float32
        ).astype(np.float64)
        observed_listener = np.asarray(
            listener_receipt.position, dtype=np.float64
        )
        observed_orientation = np.asarray(
            listener_receipt.orientation_wxyz, dtype=np.float64
        )
        if (
            str(listener_receipt.listener_id) != self.listener_id
            or observed_listener.shape != (3,)
            or observed_orientation.shape != (4,)
            or not np.array_equal(observed_listener, expected_listener)
            or not np.array_equal(observed_orientation, expected_orientation)
            or listener_receipt.native_realized is not True
        ):
            raise RIRCacheError("native Listener receipt differs from cache request")
        efficiency = float(self.context.indirect_ray_efficiency())
        if not math.isfinite(efficiency) or not 0.0 <= efficiency <= 1.0:
            raise RIRCacheError("native indirect ray efficiency is invalid")
        return RIRBatchResult(
            samples=tuple(
                by_id[source_id] for source_id in self.source_ids[: len(requested)]
            ),
            sample_rate_hz=int(round(self.selected.sample_rate_hz)),
            layout_id=self.layout_id,
            channel_labels=self.channel_labels,
            indirect_ray_efficiency=efficiency,
            wall_seconds=wall_seconds,
            process_cpu_seconds=cpu_seconds,
        )


def _atomic_savez(path: Path, *, compressed: bool, arrays: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    if temporary.exists():
        temporary.unlink()
    writer = np.savez_compressed if compressed else np.savez
    writer(temporary, **arrays)
    os.rename(temporary, path)


def _read_semantic_shard(path: Path) -> dict[str, Any]:
    """Decode one structural/sample shard without file-evidence fields."""

    required = {
        "job_indices",
        "job_ids",
        "source_positions_m",
        "listener_positions_m",
        "listener_orientations_wxyz",
        "lengths",
        "samples",
        "sample_rate_hz",
        "layout_id",
        "channel_labels",
        "simulate_wall_seconds",
        "simulate_process_cpu_seconds",
        "indirect_ray_efficiency",
    }
    with np.load(path, allow_pickle=False) as value:
        if set(value.files) != required:
            raise RIRCacheError(
                f"semantic RIR shard fields differ from contract: {path}"
            )
        result = {name: np.asarray(value[name]).copy() for name in required}
    job_indices = result["job_indices"]
    count = job_indices.shape[0] if job_indices.ndim == 1 else 0
    samples = result["samples"]
    lengths = result["lengths"]
    job_ids = result["job_ids"]
    labels = result["channel_labels"]
    scalar_unicode = result["layout_id"]
    exact_dtypes = (
        result["job_indices"].dtype == np.dtype("<u4")
        and lengths.dtype == np.dtype("<u4")
        and result["sample_rate_hz"].dtype == np.dtype("<u4")
        and samples.dtype == np.dtype("<f4")
        and result["source_positions_m"].dtype == np.dtype("<f8")
        and result["listener_positions_m"].dtype == np.dtype("<f8")
        and result["listener_orientations_wxyz"].dtype == np.dtype("<f8")
        and result["simulate_wall_seconds"].dtype == np.dtype("<f8")
        and result["simulate_process_cpu_seconds"].dtype == np.dtype("<f8")
        and result["indirect_ray_efficiency"].dtype == np.dtype("<f8")
        and job_ids.dtype.kind == "U"
        and labels.dtype.kind == "U"
        and scalar_unicode.dtype.kind == "U"
    )
    if (
        count < 1
        or not exact_dtypes
        or job_indices.shape != (count,)
        or any(not value for value in job_ids.tolist())
        or len(set(job_ids.tolist())) != count
        or samples.ndim != 3
        or samples.shape[0] != count
        or samples.shape[1] != 2
        or job_ids.shape != (count,)
        or result["source_positions_m"].shape != (count, 3)
        or result["listener_positions_m"].shape != (count, 3)
        or result["listener_orientations_wxyz"].shape != (count, 4)
        or lengths.shape != (count,)
        or result["sample_rate_hz"].shape != ()
        or int(result["sample_rate_hz"].item()) < 1
        or scalar_unicode.shape != ()
        or not str(scalar_unicode.item())
        or labels.shape != (2,)
        or tuple(labels.tolist()) != ("left", "right")
        or any(
            result[name].shape != ()
            for name in (
                "simulate_wall_seconds",
                "simulate_process_cpu_seconds",
                "indirect_ray_efficiency",
            )
        )
        or not all(array.flags.c_contiguous for array in result.values())
        or not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(result["source_positions_m"]))
        or not np.all(np.isfinite(result["listener_positions_m"]))
        or not np.all(np.isfinite(result["listener_orientations_wxyz"]))
        or not math.isfinite(float(result["simulate_wall_seconds"]))
        or float(result["simulate_wall_seconds"]) < 0
        or not math.isfinite(float(result["simulate_process_cpu_seconds"]))
        or float(result["simulate_process_cpu_seconds"]) < 0
        or not math.isfinite(float(result["indirect_ray_efficiency"]))
        or not 0 <= float(result["indirect_ray_efficiency"]) <= 1
    ):
        raise RIRCacheError(f"semantic RIR shard arrays are inconsistent: {path}")
    orientation_norms = np.linalg.norm(result["listener_orientations_wxyz"], axis=1)
    if not np.allclose(orientation_norms, 1.0, rtol=0.0, atol=1.0e-6):
        raise RIRCacheError(
            f"semantic RIR shard orientations are not unit normalized: {path}"
        )
    for row, raw_length in enumerate(lengths):
        length = int(raw_length)
        if (
            length < 2
            or length > samples.shape[2]
            or not np.any(samples[row, :, :length])
            or np.any(samples[row, :, length:])
        ):
            raise RIRCacheError(f"semantic RIR shard samples are invalid: {path}")
    return result


def _verify_semantic_shard_request(
    retained: Mapping[str, np.ndarray],
    *,
    path: Path,
    expected_job_indices: np.ndarray,
    expected_jobs: Sequence[Mapping[str, Any]],
    expected_positions: Sequence[Sequence[float]],
    expected_listener_positions: Sequence[Sequence[float]],
    expected_listener_orientations: Sequence[Sequence[float]],
    sample_rate_hz: int,
    layout_id: str,
    channel_labels: Sequence[str],
) -> None:
    if (
        not np.array_equal(retained["job_indices"], expected_job_indices)
        or not np.array_equal(
            retained["job_ids"],
            np.asarray([str(job["job_id"]) for job in expected_jobs]),
        )
        or not np.array_equal(
            retained["source_positions_m"],
            np.asarray(expected_positions, dtype="<f8"),
        )
        or not np.array_equal(
            retained["listener_positions_m"],
            np.asarray(expected_listener_positions, dtype="<f8"),
        )
        or not np.array_equal(
            retained["listener_orientations_wxyz"],
            np.asarray(expected_listener_orientations, dtype="<f8"),
        )
        or retained["sample_rate_hz"].shape != ()
        or int(retained["sample_rate_hz"].item()) != sample_rate_hz
        or retained["layout_id"].shape != ()
        or str(retained["layout_id"].item()) != layout_id
        or not np.array_equal(
            retained["channel_labels"], np.asarray(tuple(channel_labels))
        )
    ):
        raise RIRCacheError(f"semantic retained shard differs from request: {path}")
    for field in ("simulate_wall_seconds", "simulate_process_cpu_seconds"):
        value = retained[field]
        if value.shape != () or not math.isfinite(float(value)) or float(value) < 0:
            raise RIRCacheError(f"semantic retained shard timing is invalid: {path}")
    efficiency = retained["indirect_ray_efficiency"]
    if (
        efficiency.shape != ()
        or not math.isfinite(float(efficiency))
        or not 0 <= float(efficiency) <= 1
    ):
        raise RIRCacheError(f"semantic retained shard efficiency is invalid: {path}")


def _semantic_selection_binding(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {
            "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
            "selection_mode": "explicit_legacy_unbound",
            "registry_selection_applied": False,
            "room_ref": None,
            "profile_ref": None,
            "binding_id": None,
        }
    if not isinstance(value, Mapping):
        raise RIRCacheError("semantic acoustic selection must be an object")
    mode = value.get("selection_mode")
    room_ref = value.get("room_ref")
    profile_ref = value.get("profile_ref")
    binding_id = value.get("binding_id")
    applied = value.get("registry_selection_applied")
    if mode in {"registry", "registry_with_verified_equivalent_overrides"}:
        if (
            applied is not True
            or not isinstance(room_ref, Mapping)
            or set(room_ref) != {"registry_id", "room_id", "revision"}
            or not all(isinstance(item, str) and item for item in room_ref.values())
            or not isinstance(profile_ref, Mapping)
            or set(profile_ref) != {"profile_id", "revision"}
            or not all(isinstance(item, str) and item for item in profile_ref.values())
            or not isinstance(binding_id, str)
            or not binding_id
        ):
            raise RIRCacheError("semantic registry acoustic selection is incomplete")
    elif mode in {"explicit_legacy", "explicit_legacy_unbound"}:
        if (
            applied is not False
            or room_ref is not None
            or profile_ref is not None
            or binding_id is not None
        ):
            raise RIRCacheError("semantic explicit acoustic selection is invalid")
    else:
        raise RIRCacheError("semantic acoustic selection mode is invalid")
    return {
        "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
        "selection_mode": mode,
        "registry_selection_applied": applied,
        "room_ref": deepcopy(room_ref),
        "profile_ref": deepcopy(profile_ref),
        "binding_id": binding_id,
    }


def _semantic_selected_path(root: Path, raw: Any, *, owner: str) -> Path:
    """Select a declared regular input confined under its package root."""

    if not isinstance(raw, str) or not raw:
        raise RIRCacheError(f"{owner} path is missing")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise RIRCacheError(f"{owner} path must be package-relative and confined")
    absolute_root = Path(os.path.abspath(root))
    selected = absolute_root / relative
    if _semantic_path_has_symlink_component(selected) or not selected.is_file():
        raise RIRCacheError(f"{owner} must be a non-symlink regular file")
    resolved_root = absolute_root.resolve(strict=True)
    resolved = selected.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise RIRCacheError(f"{owner} escapes its selected package root")
    return resolved


def _semantic_path_has_symlink_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _semantic_regular_file(
    path: Path, *, owner: str, absolute_required: bool = False
) -> Path:
    raw = Path(path)
    if absolute_required and not raw.is_absolute():
        raise RIRCacheError(f"{owner} must be an absolute regular file")
    absolute = Path(os.path.abspath(raw))
    if _semantic_path_has_symlink_component(absolute) or not absolute.is_file():
        raise RIRCacheError(f"{owner} must be a non-symlink regular file")
    return absolute.resolve(strict=True)


def load_semantic_acoustic_scene(manifest_path: Path) -> SemanticAcousticScene:
    """Load decoded scene structure while never computing file evidence."""

    path = _semantic_regular_file(manifest_path, owner="semantic acoustic manifest")
    root = path.parent
    manifest = load_json(path)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != "avengine_acoustic_scene_package_v1"
        or manifest.get("package_mode") != "research_candidate"
        or not isinstance(manifest.get("package_id"), str)
        or not manifest["package_id"]
    ):
        raise RIRCacheError("semantic acoustic manifest contract is invalid")
    arrays = manifest.get("arrays")
    materials = manifest.get("materials")
    geometry = manifest.get("geometry")
    if not all(isinstance(item, Mapping) for item in (arrays, materials, geometry)):
        raise RIRCacheError("semantic acoustic manifest structure is incomplete")

    def array(name: str, dtype: str, shape_tail: tuple[int, ...]) -> np.ndarray:
        record = arrays.get(name)
        if (
            not isinstance(record, Mapping)
            or record.get("format") != "npy"
            or record.get("dtype") != dtype
            or record.get("memory_order") != "C"
        ):
            raise RIRCacheError(f"semantic acoustic {name} declaration is invalid")
        selected = _semantic_selected_path(
            root, record.get("path"), owner=f"semantic acoustic {name}"
        )
        try:
            value = np.load(selected, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RIRCacheError(f"semantic acoustic {name} is unreadable") from exc
        declared_shape = record.get("shape")
        if (
            value.dtype != np.dtype(dtype)
            or not value.flags.c_contiguous
            or value.shape[1:] != shape_tail
            or not isinstance(declared_shape, list)
            or list(value.shape) != declared_shape
        ):
            raise RIRCacheError(f"semantic acoustic {name} array contract drift")
        return np.array(value, dtype=dtype, order="C", copy=True)

    vertices = array("vertices", "<f4", (3,))
    triangles = array("triangles", "<u4", (3,))
    material_ids = array("triangle_material_ids", "<u4", ())
    if (
        not np.all(np.isfinite(vertices))
        or len(triangles) != len(material_ids)
        or int(triangles.max(initial=0)) >= len(vertices)
        or isinstance(geometry.get("vertex_count"), bool)
        or not isinstance(geometry.get("vertex_count"), int)
        or geometry.get("vertex_count") != len(vertices)
        or isinstance(geometry.get("triangle_count"), bool)
        or not isinstance(geometry.get("triangle_count"), int)
        or geometry.get("triangle_count") != len(triangles)
        or geometry.get("index_space") != "global_vertex_array"
        or geometry.get("transform_policy") != "baked_to_canonical_world"
    ):
        raise RIRCacheError("semantic acoustic geometry contract drift")
    category_record = materials.get("categories")
    database_record = materials.get("rlr_database")
    if not isinstance(category_record, Mapping) or not isinstance(
        database_record, Mapping
    ):
        raise RIRCacheError("semantic acoustic material declarations are missing")
    category_path = _semantic_selected_path(
        root, category_record.get("path"), owner="semantic material categories"
    )
    database_path = _semantic_selected_path(
        root, database_record.get("path"), owner="semantic material database"
    )
    categories_document = load_json(category_path)
    try:
        database_payload = database_path.read_bytes()
        database = json.loads(database_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RIRCacheError("semantic material database is unreadable") from exc
    database_errors = _validate_rlr_document(database)
    if database_errors:
        raise RIRCacheError(
            "semantic material database is invalid: " + "; ".join(database_errors)
        )
    if (
        not isinstance(categories_document, Mapping)
        or set(categories_document)
        != {
            "schema",
            "mapping_id",
            "room_id",
            "mapping_source_kind",
            "fallback_category",
            "categories",
        }
        or categories_document.get("schema")
        != "avengine_acoustic_material_categories_v1"
        or categories_document.get("mapping_id") != materials.get("mapping_id")
        or categories_document.get("mapping_source_kind")
        != materials.get("mapping_source_kind")
        or not isinstance(categories_document.get("room_id"), str)
        or not categories_document["room_id"]
        or categories_document.get("fallback_category") is not None
    ):
        raise RIRCacheError("semantic material category document is invalid")
    raw_categories = categories_document["categories"]
    database_materials = database.get("materials")
    if (
        not isinstance(raw_categories, list)
        or not raw_categories
        or materials.get("category_count") != len(raw_categories)
        or not isinstance(database_materials, list)
        or not database_materials
    ):
        raise RIRCacheError("semantic acoustic material structure is invalid")
    categories: list[str] = []
    material_names: dict[str, str] = {}
    material_indices: dict[str, int] = {}
    for index, category in enumerate(raw_categories):
        if (
            not isinstance(category, Mapping)
            or set(category)
            != {
                "category_name",
                "fallback",
                "human_override",
                "mapping_confidence",
                "mapping_source",
                "material_id",
                "material_key",
                "randomized",
                "rlr_match",
                "rlr_material_name",
                "source_material_name",
            }
            or isinstance(category.get("material_id"), bool)
            or not isinstance(category.get("material_id"), int)
            or category.get("material_id") != index
            or not isinstance(category.get("category_name"), str)
            or not category["category_name"]
            or category.get("fallback") is not False
            or not isinstance(category.get("rlr_material_name"), str)
            or not category["rlr_material_name"]
        ):
            raise RIRCacheError("semantic acoustic material category is invalid")
        name = str(category["category_name"])
        matches = [
            material_index
            for material_index, item in enumerate(database_materials)
            if isinstance(item, Mapping)
            and name.casefold()
            in {str(label).casefold() for label in item.get("labels", [])}
        ]
        if len(matches) != 1:
            raise RIRCacheError("semantic material category mapping is ambiguous")
        material = database_materials[matches[0]]
        material_name = material.get("name")
        if (
            not isinstance(material_name, str)
            or not material_name
            or category["rlr_material_name"] != material_name
        ):
            raise RIRCacheError("semantic material name declaration is invalid")
        categories.append(name)
        material_names[name] = material_name
        material_indices[name] = matches[0]
    if len(set(categories)) != len(categories):
        raise RIRCacheError("semantic material categories are duplicated")
    if {int(item) for item in np.unique(material_ids)} != set(range(len(categories))):
        raise RIRCacheError("semantic material identifiers are not closed")
    raw_objects = manifest.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise RIRCacheError("semantic acoustic objects are missing")
    native_objects: list[dict[str, Any]] = []
    object_ids: set[str] = set()
    next_vertex = next_triangle = 0
    for item in raw_objects:
        if not isinstance(item, Mapping):
            raise RIRCacheError("semantic acoustic object is invalid")
        object_id = item.get("object_id")
        if not isinstance(object_id, str) or not object_id or object_id in object_ids:
            raise RIRCacheError("semantic acoustic object identity is invalid")
        vertex_count = item.get("vertex_count")
        triangle_count = item.get("triangle_count")
        if (
            isinstance(vertex_count, bool)
            or not isinstance(vertex_count, int)
            or isinstance(triangle_count, bool)
            or not isinstance(triangle_count, int)
            or isinstance(item.get("vertex_offset"), bool)
            or not isinstance(item.get("vertex_offset"), int)
            or item.get("vertex_offset") != next_vertex
            or isinstance(item.get("triangle_offset"), bool)
            or not isinstance(item.get("triangle_offset"), int)
            or item.get("triangle_offset") != next_triangle
            or vertex_count < 3
            or triangle_count < 1
        ):
            raise RIRCacheError("semantic acoustic object ranges are invalid")
        vertex_end = next_vertex + vertex_count
        triangle_end = next_triangle + triangle_count
        object_triangles = triangles[next_triangle:triangle_end].astype(
            np.uint64, copy=False
        )
        if (
            vertex_end > len(vertices)
            or triangle_end > len(triangles)
            or int(object_triangles.min(initial=next_vertex)) < next_vertex
            or int(object_triangles.max(initial=next_vertex)) >= vertex_end
        ):
            raise RIRCacheError("semantic acoustic object escapes its array range")
        native_objects.append(
            {
                "object_id": object_id,
                "vertices": np.ascontiguousarray(vertices[next_vertex:vertex_end]),
                "triangles": np.ascontiguousarray(
                    object_triangles - next_vertex, dtype="<u4"
                ),
                "triangle_material_ids": np.ascontiguousarray(
                    material_ids[next_triangle:triangle_end]
                ),
                "position": (0.0, 0.0, 0.0),
                "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            }
        )
        object_ids.add(object_id)
        next_vertex = vertex_end
        next_triangle = triangle_end
    if next_vertex != len(vertices) or next_triangle != len(triangles):
        raise RIRCacheError("semantic acoustic object ranges do not cover arrays")
    return SemanticAcousticScene(
        manifest_path=path,
        package_id=str(manifest["package_id"]),
        material_database_bytes=database_payload,
        material_categories=tuple(categories),
        material_name_by_category=material_names,
        material_index_by_category=material_indices,
        objects=tuple(native_objects),
        triangle_count_by_material={
            name: int(np.count_nonzero(material_ids == index))
            for index, name in enumerate(categories)
        },
    )


def _load_semantic_habitat_runtime() -> tuple[Any, Any, dict[str, Any]]:
    """Load the required native API and validate paths without file evidence."""

    try:
        quaternion_module = importlib.import_module("quaternion")
        habitat_module = importlib.import_module("habitat_sim")
        binding_module = importlib.import_module(
            "habitat_sim._ext.habitat_sim_bindings"
        )
    except (ImportError, OSError) as exc:
        raise RIRCacheError("semantic Habitat/RLR runtime is unavailable") from exc
    required = (
        "RLRContextConfiguration",
        "RLRAcousticContext",
        "RLRChannelLayoutType",
    )
    if (
        getattr(habitat_module, "audio_enabled", None) is not True
        or any(getattr(habitat_module, name, None) is None for name in required)
        or any(getattr(binding_module, name, None) is None for name in required)
        or any(
            getattr(habitat_module, name) is not getattr(binding_module, name)
            for name in required
        )
        or any(
            getattr(getattr(habitat_module, name), "__module__", None)
            != binding_module.__name__
            for name in required
        )
    ):
        raise RIRCacheError("semantic Habitat runtime lacks the required RLR API")

    def regular_module_path(module: Any, *, owner: str) -> Path:
        raw = Path(str(getattr(module, "__file__", "")))
        return _semantic_regular_file(
            raw, owner=f"{owner} module path", absolute_required=True
        )

    quaternion_path = regular_module_path(quaternion_module, owner="quaternion runtime")
    habitat_path = regular_module_path(habitat_module, owner="Habitat runtime")
    binding_path = regular_module_path(binding_module, owner="Habitat binding")
    if binding_path.suffix != ".so":
        raise RIRCacheError("semantic Habitat binding is not a compiled extension")
    library = _semantic_regular_file(
        binding_path.parent / "libRLRAudioPropagation.so",
        owner="RLR library path",
        absolute_required=True,
    )
    if library.parent != binding_path.parent:
        raise RIRCacheError("RLR library escapes the selected binding directory")
    habitat_spec = getattr(habitat_module, "__spec__", None)
    binding_spec = getattr(binding_module, "__spec__", None)
    habitat_search = getattr(habitat_spec, "submodule_search_locations", None)
    habitat_roots = (
        tuple(Path(os.path.abspath(str(location))) for location in habitat_search)
        if habitat_search
        else ()
    )
    if (
        getattr(habitat_spec, "name", None) != "habitat_sim"
        or Path(str(getattr(habitat_spec, "origin", ""))) != habitat_path
        or habitat_path.parent not in habitat_roots
        or not any(binding_path.is_relative_to(root) for root in habitat_roots)
        or any(
            _semantic_path_has_symlink_component(root)
            or not root.is_absolute()
            or not root.is_dir()
            for root in habitat_roots
        )
        or getattr(binding_spec, "name", None)
        != "habitat_sim._ext.habitat_sim_bindings"
        or Path(str(getattr(binding_spec, "origin", ""))) != binding_path
        or getattr(binding_spec, "parent", None) != "habitat_sim._ext"
        or not isinstance(getattr(binding_spec, "loader", None), ExtensionFileLoader)
    ):
        raise RIRCacheError("semantic Habitat import specifications are invalid")
    return (
        habitat_module,
        binding_module,
        {
            "schema": "avengine_semantic_habitat_rlr_runtime_v1",
            "binding_api": "habitat_sim.RLRAcousticContext_v1",
            "quaternion_module_path": str(quaternion_path),
            "habitat_module_path": str(habitat_path),
            "binding_module_path": str(binding_path),
            "rlr_library_path": str(library),
        },
    )


def _semantic_upload_summary(scene: SemanticAcousticScene, raw: Any) -> dict[str, Any]:
    """Validate structural native upload readback and omit evidence payloads."""

    expected_objects = [str(item["object_id"]) for item in scene.objects]
    observed_counts = {
        str(key): int(value)
        for key, value in dict(raw.triangle_count_by_material).items()
    }
    observed_calls = {
        str(key): int(value)
        for key, value in dict(raw.material_upload_call_count).items()
    }
    expected_calls = {name: 0 for name in scene.material_categories}
    expected_receipts: dict[tuple[str, str], int] = {}
    for item in scene.objects:
        material_ids = np.asarray(item["triangle_material_ids"], dtype=np.uint32)
        for index, name in enumerate(scene.material_categories):
            count = int(np.count_nonzero(material_ids == index))
            if count:
                expected_calls[name] += 1
                expected_receipts[(str(item["object_id"]), name)] = count
    observed_receipts: dict[tuple[str, str], int] = {}
    for item in raw.material_upload_receipts:
        key = (str(item.object_id), str(item.material_category))
        triangle_count = int(item.triangle_count)
        if key in observed_receipts or int(item.index_count) != triangle_count * 3:
            raise RIRCacheError("semantic native material upload receipt is invalid")
        observed_receipts[key] = triangle_count
    observed_names = {
        str(key): str(value)
        for key, value in dict(raw.resolved_material_name_by_category).items()
    }
    observed_indices = {
        str(key): int(value)
        for key, value in dict(raw.resolved_material_index_by_category).items()
    }
    if (
        int(raw.object_count) != len(scene.objects)
        or int(raw.vertex_count) != sum(len(item["vertices"]) for item in scene.objects)
        or int(raw.triangle_count)
        != sum(len(item["triangles"]) for item in scene.objects)
        or int(raw.material_category_count) != len(scene.material_categories)
        or [str(item) for item in raw.object_ids] != expected_objects
        or observed_counts != dict(scene.triangle_count_by_material)
        or observed_calls != expected_calls
        or observed_receipts != expected_receipts
        or observed_names != dict(scene.material_name_by_category)
        or observed_indices != dict(scene.material_index_by_category)
    ):
        raise RIRCacheError("semantic native acoustic upload structure drift")
    return {
        "status": "pass_structural_native_upload",
        "object_count": len(scene.objects),
        "vertex_count": sum(len(item["vertices"]) for item in scene.objects),
        "triangle_count": sum(len(item["triangles"]) for item in scene.objects),
        "material_category_count": len(scene.material_categories),
        "object_ids": expected_objects,
        "triangle_count_by_material": dict(scene.triangle_count_by_material),
        "material_upload_call_count": expected_calls,
        "resolved_material_name_by_category": dict(scene.material_name_by_category),
        "resolved_material_index_by_category": dict(scene.material_index_by_category),
    }


class _SemanticNativeRIRBatchRenderer(_NativeRIRBatchRenderer):
    """Native renderer whose setup path never produces file evidence."""

    def __init__(
        self,
        scene: SemanticAcousticScene,
        simulation: M4SimulationConfig,
        *,
        batch_size: int,
        initial_positions_m: Sequence[Sequence[float]],
        listener_position_m: Sequence[float],
        listener_orientation_wxyz: Sequence[float],
        layout_type: str,
        channel_count: int,
        hrtf_file_path: str,
        source_radius_m: float,
        listener_radius_m: float,
    ) -> None:
        if not isinstance(scene, SemanticAcousticScene):
            raise RIRCacheError("semantic scene type is invalid")
        self.batch_size = _positive_int(batch_size, owner="native batch size")
        if len(initial_positions_m) != self.batch_size:
            raise RIRCacheError("initial position count differs from native batch size")
        self.source_ids = tuple(
            f"cache_slot_{index:04d}" for index in range(self.batch_size)
        )
        self.layout_type = layout_type
        self.channel_count = channel_count
        self.selected = simulation_with_layout(
            simulation, layout_type=layout_type, channel_count=channel_count
        )
        if self.selected.temporal_coherence:
            raise RIRCacheError("semantic RIR cache requires temporal_coherence=false")
        self.layout_id = (
            BINAURAL_LAYOUT_ID if layout_type == "binaural" else FOA_LAYOUT_ID
        )
        self.channel_labels = (
            ("left", "right") if layout_type == "binaural" else ("W", "Y", "Z", "X")
        )
        if layout_type == "binaural":
            hrtf_file_path = str(
                _semantic_regular_file(
                    Path(hrtf_file_path),
                    owner="semantic binaural HRTF",
                    absolute_required=True,
                )
            )
        elif hrtf_file_path:
            raise RIRCacheError("HRTF is only valid for a binaural cache")

        setup_wall_start = time.perf_counter()
        setup_cpu_start = time.process_time()
        habitat_module, binding_module, runtime_report = (
            _load_semantic_habitat_runtime()
        )
        native_configuration, config_readback = _native_configuration(
            habitat_module, self.selected
        )
        self.context = habitat_module.RLRAcousticContext(native_configuration)
        context_type = type(self.context)
        if (
            context_type is not habitat_module.RLRAcousticContext
            or context_type is not binding_module.RLRAcousticContext
            or getattr(context_type, "__module__", None) != binding_module.__name__
            or getattr(context_type.simulate_owned, "__module__", None)
            != binding_module.__name__
        ):
            raise RIRCacheError("semantic RLR context is not the compiled binding type")
        with tempfile.TemporaryDirectory(prefix="avengine-semantic-rir-db-") as temp:
            private_database = Path(temp) / "material_database.json"
            private_database.write_bytes(scene.material_database_bytes)
            raw_upload = self.context.load_acoustic_scene(
                str(private_database),
                list(scene.material_categories),
                list(scene.objects),
            )
        upload_summary = _semantic_upload_summary(scene, raw_upload)
        for source_id, position in zip(
            self.source_ids, initial_positions_m, strict=True
        ):
            self.context.add_source(source_id, position, source_radius_m)
        self.listener_id = "cache_listener0"
        self.context.add_listener(
            self.listener_id,
            listener_position_m,
            listener_orientation_wxyz,
            _native_layout(habitat_module, layout_type),
            channel_count,
            listener_radius_m,
            hrtf_file_path,
        )
        self.current_listener_position_m = _finite_vector(
            listener_position_m, 3, owner="initial listener position"
        )
        self.current_listener_orientation_wxyz = _unit_orientation(
            listener_orientation_wxyz, owner="initial listener orientation"
        )
        self.listener_pose_update_count = 0
        self.setup_report = {
            "schema": "avengine_semantic_native_rir_setup_v1",
            "runtime": runtime_report,
            "configuration_readback": config_readback,
            "upload": upload_summary,
            "wall_seconds": time.perf_counter() - setup_wall_start,
            "process_cpu_seconds": time.process_time() - setup_cpu_start,
            "compute_device": "CPU",
            "qualification_claim": False,
        }
        self.native_simulate_owned_call_count = 0
        self.native_realized_job_count = 0

    def render(
        self,
        positions_m: Sequence[Sequence[float]],
        *,
        listener_position_m: Sequence[float] | None = None,
        listener_orientation_wxyz: Sequence[float] | None = None,
    ) -> RIRBatchResult:
        result = super().render(
            positions_m,
            listener_position_m=listener_position_m,
            listener_orientation_wxyz=listener_orientation_wxyz,
        )
        self.native_simulate_owned_call_count += 1
        self.native_realized_job_count += len(positions_m)
        return result


def _read_shard(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as value:
        required = {
            "job_indices",
            "job_ids",
            "source_positions_m",
            "lengths",
            "samples",
            "ir_sha256",
            "sample_rate_hz",
            "layout_id",
            "channel_labels",
            "simulate_wall_seconds",
            "simulate_process_cpu_seconds",
            "indirect_ray_efficiency",
        }
        listener_pose_fields = {
            "listener_positions_m",
            "listener_orientations_wxyz",
            "acoustic_state_sha256",
        }
        observed_fields = set(value.files)
        if frozenset(observed_fields) not in {
            frozenset(required),
            frozenset(required | listener_pose_fields),
        }:
            raise RIRCacheError(f"RIR shard fields differ from contract: {path}")
        result = {
            key: np.asarray(value[key]).copy()
            for key in observed_fields
        }
    count = len(result["job_indices"])
    samples = result["samples"]
    if (
        count < 1
        or result["job_ids"].shape != (count,)
        or result["source_positions_m"].shape != (count, 3)
        or result["lengths"].shape != (count,)
        or samples.ndim != 3
        or samples.shape[0] != count
        or result["ir_sha256"].shape != (count,)
        or np.any(result["lengths"] < 2)
        or np.any(result["lengths"] > samples.shape[2])
        or not np.all(np.isfinite(samples))
    ):
        raise RIRCacheError(f"RIR shard arrays are inconsistent: {path}")
    if "listener_positions_m" in result and (
        result["listener_positions_m"].shape != (count, 3)
        or result["listener_orientations_wxyz"].shape != (count, 4)
        or result["acoustic_state_sha256"].shape != (count,)
        or not np.all(np.isfinite(result["listener_positions_m"]))
        or not np.all(np.isfinite(result["listener_orientations_wxyz"]))
        or not np.allclose(
            np.linalg.norm(result["listener_orientations_wxyz"], axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise RIRCacheError(f"RIR shard Listener-pose arrays are inconsistent: {path}")
    for row in range(count):
        length = int(result["lengths"][row])
        digest = hashlib.sha256(
            np.ascontiguousarray(samples[row, :, :length]).tobytes(order="C")
        ).hexdigest()
        if digest != str(result["ir_sha256"][row]):
            raise RIRCacheError(f"RIR shard payload hash differs: {path} row {row}")
    return result


def _verify_shard_request(
    retained: Mapping[str, np.ndarray],
    *,
    path: Path,
    expected_job_indices: np.ndarray,
    expected_job_ids: Sequence[str],
    expected_source_positions_m: Sequence[Sequence[float]],
    expected_listener_positions_m: Sequence[Sequence[float]] | None,
    expected_listener_orientations_wxyz: Sequence[Sequence[float]] | None,
    expected_acoustic_state_sha256: Sequence[str] | None,
    sample_rate_hz: int,
    layout_id: str,
    channel_labels: Sequence[str],
) -> None:
    samples = retained["samples"]
    lengths = retained["lengths"]
    if not np.array_equal(retained["job_indices"], expected_job_indices):
        raise RIRCacheError(f"retained shard job range differs from request: {path}")
    if not np.array_equal(retained["job_ids"], np.asarray(tuple(expected_job_ids))):
        raise RIRCacheError(f"retained shard job IDs differ from request: {path}")
    if not np.array_equal(
        retained["source_positions_m"],
        np.asarray(expected_source_positions_m, dtype="<f8"),
    ):
        raise RIRCacheError(
            f"retained shard source positions differ from request: {path}"
        )
    expected_pose_values = (
        expected_listener_positions_m,
        expected_listener_orientations_wxyz,
        expected_acoustic_state_sha256,
    )
    if any(value is None for value in expected_pose_values) and not all(
        value is None for value in expected_pose_values
    ):
        raise RIRCacheError("expected Listener-pose shard fields are incomplete")
    if expected_listener_positions_m is not None:
        if "listener_positions_m" not in retained:
            raise RIRCacheError(
                f"retained shard lacks Listener-pose binding: {path}"
            )
        if (
            not np.array_equal(
                retained["listener_positions_m"],
                np.asarray(expected_listener_positions_m, dtype="<f8"),
            )
            or not np.array_equal(
                retained["listener_orientations_wxyz"],
                np.asarray(expected_listener_orientations_wxyz, dtype="<f8"),
            )
            or not np.array_equal(
                retained["acoustic_state_sha256"],
                np.asarray(tuple(expected_acoustic_state_sha256)),
            )
        ):
            raise RIRCacheError(
                f"retained shard Listener pose differs from request: {path}"
            )
    if (
        retained["sample_rate_hz"].shape != ()
        or int(retained["sample_rate_hz"]) != sample_rate_hz
        or retained["layout_id"].shape != ()
        or str(retained["layout_id"]) != layout_id
        or not np.array_equal(
            retained["channel_labels"], np.asarray(tuple(channel_labels))
        )
        or samples.shape[1] != len(channel_labels)
    ):
        raise RIRCacheError(f"retained shard audio layout differs from request: {path}")
    for field in ("simulate_wall_seconds", "simulate_process_cpu_seconds"):
        value = retained[field]
        if value.shape != () or not math.isfinite(float(value)) or float(value) < 0.0:
            raise RIRCacheError(f"retained shard timing is invalid: {path}")
    efficiency = retained["indirect_ray_efficiency"]
    if (
        efficiency.shape != ()
        or not math.isfinite(float(efficiency))
        or not 0.0 <= float(efficiency) <= 1.0
    ):
        raise RIRCacheError(f"retained shard ray efficiency is invalid: {path}")
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if not np.any(samples[row, :, :length] != 0.0) or np.any(
            samples[row, :, length:] != 0.0
        ):
            raise RIRCacheError(f"retained shard signal padding is invalid: {path}")


def _declared_sha256(value: Any, *, owner: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RIRCacheError(f"{owner} SHA-256 is malformed")
    return value


def _canonical_record_sha256(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    owner: str,
) -> str:
    declared = _declared_sha256(value.get(hash_field), owner=owner)
    try:
        observed = canonical_json_sha256(
            {
                key: item
                for key, item in value.items()
                if key != hash_field
            }
        )
    except (TypeError, ValueError) as exc:
        raise RIRCacheError(f"{owner} is not canonical JSON") from exc
    if observed != declared:
        raise RIRCacheError(f"{owner} differs from its canonical content")
    return declared


def _verified_selection_file(
    value: Any,
    *,
    path_field: str,
    size_field: str,
    expected_sha256: str,
    owner: str,
    expected_path: Path | None = None,
    registry_verified: bool = False,
) -> str:
    if not isinstance(value, Mapping):
        raise RIRCacheError(f"{owner} selection record is invalid")
    raw_path = value.get(path_field)
    declared_sha256 = _declared_sha256(value.get("sha256"), owner=owner)
    byte_size = value.get(size_field)
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or isinstance(byte_size, bool)
        or not isinstance(byte_size, int)
        or byte_size < 0
    ):
        raise RIRCacheError(f"{owner} selection record is invalid")
    if registry_verified and (
        value.get("verification_status") != "verified"
        or value.get("exists") is not True
    ):
        raise RIRCacheError(f"{owner} registry path was not physically verified")
    try:
        path = Path(raw_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RIRCacheError(f"{owner} selection path cannot be resolved") from exc
    if registry_verified and declared_sha256 != expected_sha256:
        raise RIRCacheError(
            f"{owner} override SHA-256 differs from the registry-selected "
            "physical file"
        )
    if (
        not path.is_file()
        or byte_size != path.stat().st_size
        or declared_sha256 != expected_sha256
        or sha256_file(path) != declared_sha256
    ):
        raise RIRCacheError(
            f"{owner} selection identity differs from the effective RIR input"
        )
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise RIRCacheError(
            f"{owner} selection path differs from the effective RIR input"
        )
    return declared_sha256


def _validate_acoustic_selection_receipt(
    value: Mapping[str, Any],
    *,
    scene_manifest_path: Path,
    scene_manifest_sha256: str,
    simulation_request_path: Path,
    simulation_request_sha256: str,
) -> dict[str, Any]:
    """Authenticate a tool selection receipt against the actual core inputs."""

    if value.get("schema") != RIR_CACHE_ACOUSTIC_SELECTION_INPUT_SCHEMA:
        raise RIRCacheError(
            "RIR acoustic selection receipt has an unsupported schema"
        )
    _canonical_record_sha256(
        value,
        hash_field="effective_selection_content_sha256",
        owner="RIR acoustic selection receipt",
    )
    effective_inputs = value.get("effective_inputs")
    if not isinstance(effective_inputs, Mapping):
        raise RIRCacheError("RIR acoustic selection effective inputs are invalid")
    _verified_selection_file(
        effective_inputs.get("acoustic_package_manifest"),
        path_field="path",
        size_field="byte_size",
        expected_path=scene_manifest_path,
        expected_sha256=scene_manifest_sha256,
        owner="acoustic package manifest",
    )
    _verified_selection_file(
        effective_inputs.get("simulation_request"),
        path_field="path",
        size_field="byte_size",
        expected_path=simulation_request_path,
        expected_sha256=simulation_request_sha256,
        owner="RIR simulation request",
    )

    mode = value.get("selection_mode")
    registry_resolution = value.get("registry_resolution")
    overrides = value.get("explicit_overrides")
    applied = value.get("registry_selection_applied_to_effective_inputs")
    if (
        not isinstance(overrides, Mapping)
        or set(overrides) != {
            "acoustic_package_manifest",
            "simulation_request",
        }
        or any(not isinstance(item, bool) for item in overrides.values())
        or not isinstance(applied, Mapping)
        or set(applied) != {
            "acoustic_package_manifest",
            "simulation_request",
        }
        or any(not isinstance(item, bool) for item in applied.values())
    ):
        raise RIRCacheError("RIR acoustic selection override declaration is invalid")

    if mode == "explicit":
        if (
            registry_resolution is not None
            or any(applied.values())
            or overrides["acoustic_package_manifest"] is not True
        ):
            raise RIRCacheError(
                "explicit legacy RIR input must declare no registry selection"
            )
        return {
            "selection_mode": "explicit_legacy",
            "registry_selection_applied": False,
            "room_ref": None,
            "profile_ref": None,
            "binding_id": None,
            "registry_selection_content_sha256": None,
        }

    if mode not in {
        "registry",
        "registry_with_explicit_overrides",
        "registry_with_verified_equivalent_overrides",
    }:
        raise RIRCacheError("RIR acoustic selection mode is invalid")
    if not isinstance(registry_resolution, Mapping):
        raise RIRCacheError("registry RIR input lacks its profile selection receipt")
    if (
        registry_resolution.get("schema")
        != "avengine_acoustic_profile_selection_v1"
        or registry_resolution.get("verification_status") != "verified"
    ):
        raise RIRCacheError(
            "registry RIR input does not contain a verified profile selection"
        )
    registry_selection_sha256 = _canonical_record_sha256(
        registry_resolution,
        hash_field="selection_content_sha256",
        owner="registry acoustic profile selection",
    )
    if value.get("simulation_profile") != registry_resolution.get(
        "simulation_profile"
    ):
        raise RIRCacheError(
            "RIR selection simulation profile differs from registry resolution"
        )
    room_ref = registry_resolution.get("room_ref")
    profile_ref = registry_resolution.get("profile_ref")
    binding_id = registry_resolution.get("binding_id")
    if (
        not isinstance(room_ref, Mapping)
        or set(room_ref) != {"registry_id", "room_id", "revision"}
        or any(not isinstance(item, str) or not item for item in room_ref.values())
        or not isinstance(profile_ref, Mapping)
        or set(profile_ref) != {"profile_id", "revision"}
        or any(
            not isinstance(item, str) or not item
            for item in profile_ref.values()
        )
        or not isinstance(binding_id, str)
        or not binding_id
    ):
        raise RIRCacheError(
            "registry RIR input lacks an exact room_ref/profile_ref identity"
        )
    paths = registry_resolution.get("paths")
    if not isinstance(paths, Mapping):
        raise RIRCacheError("registry RIR input path evidence is invalid")
    _verified_selection_file(
        paths.get("acoustic_package_manifest"),
        path_field="resolved_path",
        size_field="size_bytes",
        expected_sha256=scene_manifest_sha256,
        owner="acoustic package manifest",
        registry_verified=True,
    )
    _verified_selection_file(
        paths.get("selected_simulation_request"),
        path_field="resolved_path",
        size_field="size_bytes",
        expected_sha256=simulation_request_sha256,
        owner="RIR simulation request",
        registry_verified=True,
    )
    has_override = any(overrides.values())
    if mode == "registry" and has_override:
        raise RIRCacheError("registry RIR input declares an undeclared override mode")
    if mode != "registry" and not has_override:
        raise RIRCacheError("registry override mode declares no explicit override")
    expected_applied = {
        "acoustic_package_manifest": not overrides[
            "acoustic_package_manifest"
        ],
        "simulation_request": not overrides["simulation_request"],
    }
    if dict(applied) != expected_applied:
        raise RIRCacheError(
            "registry selection application flags differ from explicit overrides"
        )
    return {
        "selection_mode": (
            "registry"
            if mode == "registry"
            else "registry_with_verified_equivalent_overrides"
        ),
        "registry_selection_applied": True,
        "room_ref": deepcopy(dict(room_ref)),
        "profile_ref": deepcopy(dict(profile_ref)),
        "binding_id": binding_id,
        "registry_selection_content_sha256": registry_selection_sha256,
    }


def _acoustic_selection_binding(
    receipt: Mapping[str, Any] | None,
    *,
    scene_manifest_path: Path,
    scene_manifest_sha256: str,
    simulation_request_path: Path,
    simulation_request_sha256: str,
) -> dict[str, Any]:
    if receipt is None:
        value: dict[str, Any] = {
            "schema": RIR_CACHE_ACOUSTIC_SELECTION_BINDING_SCHEMA,
            "selection_mode": "explicit_legacy",
            "registry_selection_applied": False,
            "room_ref": None,
            "profile_ref": None,
            "binding_id": None,
            "registry_selection_content_sha256": None,
            "effective_selection_content_sha256": None,
            "acoustic_package_manifest_sha256": scene_manifest_sha256,
            "simulation_request_sha256": simulation_request_sha256,
            "input_receipt_sha256": None,
        }
    else:
        normalized_receipt = deepcopy(dict(receipt))
        selection = _validate_acoustic_selection_receipt(
            normalized_receipt,
            scene_manifest_path=scene_manifest_path,
            scene_manifest_sha256=scene_manifest_sha256,
            simulation_request_path=simulation_request_path,
            simulation_request_sha256=simulation_request_sha256,
        )
        value = {
            "schema": RIR_CACHE_ACOUSTIC_SELECTION_BINDING_SCHEMA,
            **selection,
            "effective_selection_content_sha256": normalized_receipt[
                "effective_selection_content_sha256"
            ],
            "acoustic_package_manifest_sha256": scene_manifest_sha256,
            "simulation_request_sha256": simulation_request_sha256,
            "input_receipt_sha256": canonical_json_sha256(
                normalized_receipt
            ),
        }
    value["binding_content_sha256"] = canonical_json_sha256(value)
    return value


def _verify_acoustic_selection_binding(value: Any) -> str:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != RIR_CACHE_ACOUSTIC_SELECTION_BINDING_SCHEMA
    ):
        raise RIRCacheError("RIR cache acoustic selection binding is invalid")
    binding_sha256 = _canonical_record_sha256(
        value,
        hash_field="binding_content_sha256",
        owner="RIR cache acoustic selection binding",
    )
    _declared_sha256(
        value.get("acoustic_package_manifest_sha256"),
        owner="RIR acoustic package manifest binding",
    )
    _declared_sha256(
        value.get("simulation_request_sha256"),
        owner="RIR simulation request binding",
    )
    mode = value.get("selection_mode")
    applied = value.get("registry_selection_applied")
    if not isinstance(applied, bool):
        raise RIRCacheError(
            "RIR cache registry selection application flag is invalid"
        )
    receipt_sha256 = value.get("input_receipt_sha256")
    effective_sha256 = value.get("effective_selection_content_sha256")
    if mode == "explicit_legacy":
        if (
            applied
            or value.get("room_ref") is not None
            or value.get("profile_ref") is not None
            or value.get("binding_id") is not None
            or value.get("registry_selection_content_sha256") is not None
        ):
            raise RIRCacheError(
                "explicit legacy RIR identity must contain no registry selection"
            )
        if (receipt_sha256 is None) != (effective_sha256 is None):
            raise RIRCacheError(
                "explicit legacy RIR selection receipt hashes are incomplete"
            )
        if receipt_sha256 is not None:
            _declared_sha256(
                receipt_sha256,
                owner="RIR acoustic selection input receipt",
            )
            _declared_sha256(
                effective_sha256,
                owner="RIR effective acoustic selection",
            )
        return binding_sha256
    if mode not in {
        "registry",
        "registry_with_verified_equivalent_overrides",
    } or not applied:
        raise RIRCacheError("RIR cache registry selection identity is invalid")
    _declared_sha256(
        receipt_sha256,
        owner="RIR acoustic selection input receipt",
    )
    _declared_sha256(
        effective_sha256,
        owner="RIR effective acoustic selection",
    )
    room_ref = value.get("room_ref")
    profile_ref = value.get("profile_ref")
    if (
        not isinstance(room_ref, Mapping)
        or set(room_ref) != {"registry_id", "room_id", "revision"}
        or any(not isinstance(item, str) or not item for item in room_ref.values())
        or not isinstance(profile_ref, Mapping)
        or set(profile_ref) != {"profile_id", "revision"}
        or any(
            not isinstance(item, str) or not item
            for item in profile_ref.values()
        )
        or not isinstance(value.get("binding_id"), str)
        or not value["binding_id"]
    ):
        raise RIRCacheError("RIR cache registry room/profile binding is invalid")
    _declared_sha256(
        value.get("registry_selection_content_sha256"),
        owner="registry acoustic profile selection",
    )
    if value.get("registry_selection_applied") is not True:
        raise RIRCacheError(
            "RIR cache registry selection was not applied to the binding"
        )
    return binding_sha256


def _request_section(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = value.get(name)
    if not isinstance(section, Mapping):
        raise RIRCacheError(f"RIR cache request {name} section is invalid")
    return section


def _request_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact canonical subset used by the v1 producer."""

    plan = _request_section(value, "plan")
    scene = _request_section(value, "acoustic_scene")
    simulation = _request_section(value, "simulation")
    output = _request_section(value, "output")
    runtime = _request_section(value, "runtime_policy")
    plan_sha256 = _declared_sha256(plan.get("sha256"), owner="RIR plan")
    scene_sha256 = _declared_sha256(
        scene.get("package_content_sha256"),
        owner="acoustic package content",
    )
    effective_simulation = simulation.get("effective")
    if not isinstance(effective_simulation, Mapping) or not effective_simulation:
        raise RIRCacheError("RIR cache effective simulation is invalid")
    hrtf_sha256 = output.get("hrtf_sha256")
    if hrtf_sha256 is not None:
        hrtf_sha256 = _declared_sha256(hrtf_sha256, owner="HRTF")
    layout_type = output.get("layout_type")
    if layout_type not in {"binaural", "ambisonics"}:
        raise RIRCacheError("RIR cache request layout type is invalid")
    batch_size = _positive_int(
        runtime.get("native_batch_size"),
        owner="RIR cache native batch size",
    )
    job_offset = plan.get("selected_job_offset")
    if isinstance(job_offset, bool) or not isinstance(job_offset, int) or job_offset < 0:
        raise RIRCacheError("RIR cache selected job offset is invalid")
    job_count = _positive_int(
        plan.get("selected_job_count"),
        owner="RIR cache selected job count",
    )
    translation = _finite_vector(
        runtime.get("coordinate_translation_m"),
        3,
        owner="RIR cache coordinate translation",
    )
    radii: dict[str, float | int] = {}
    for name in ("source_radius_m", "listener_radius_m"):
        radius = runtime.get(name)
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(float(radius))
            or float(radius) < 0.0
        ):
            raise RIRCacheError(f"RIR cache {name} is invalid")
        radii[name] = radius
    compressed = output.get("compressed_npz_shards")
    if not isinstance(compressed, bool):
        raise RIRCacheError("RIR cache compression policy is invalid")
    payload = {
        "plan_sha256": plan_sha256,
        "scene_sha256": scene_sha256,
        "simulation": dict(effective_simulation),
        "hrtf_sha256": hrtf_sha256,
        "layout_type": layout_type,
        "batch_size": batch_size,
        "job_offset": job_offset,
        "job_count": job_count,
        "translation_m": list(translation),
        "source_radius_m": radii["source_radius_m"],
        "listener_radius_m": radii["listener_radius_m"],
        "compressed": compressed,
    }
    acoustic_selection_binding = value.get("acoustic_selection_binding")
    if acoustic_selection_binding is not None:
        _verify_acoustic_selection_binding(acoustic_selection_binding)
        payload["acoustic_selection_binding"] = deepcopy(
            dict(acoustic_selection_binding)
        )
    selected_states_sha256 = plan.get("selected_acoustic_states_sha256")
    state_binding = plan.get("acoustic_state_binding")
    if selected_states_sha256 is None and state_binding is None:
        return payload
    if state_binding != "source_listener_pose_per_job_v1":
        raise RIRCacheError("RIR cache request acoustic-state binding is invalid")
    listener_update_policy = runtime.get("listener_pose_update_policy")
    if listener_update_policy != "set_listener_pose_on_change_v1":
        raise RIRCacheError("RIR cache request Listener update policy is invalid")
    payload["acoustic_state_binding"] = state_binding
    payload["selected_acoustic_states_sha256"] = _declared_sha256(
        selected_states_sha256,
        owner="selected RIR acoustic states",
    )
    payload["listener_pose_update_policy"] = listener_update_policy
    return payload


def _verify_request_identity(value: Mapping[str, Any]) -> str:
    declared = _declared_sha256(
        value.get("request_identity_sha256"),
        owner="RIR cache request identity",
    )
    recomputed = canonical_json_sha256(_request_identity_payload(value))
    if recomputed != declared:
        raise RIRCacheError(
            "RIR cache request identity differs from its canonical request fields"
        )
    return recomputed


def _acoustic_selection_sidecar(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _request_section(request, "acoustic_selection_binding")
    binding_sha256 = _verify_acoustic_selection_binding(binding)
    request_identity_sha256 = _verify_request_identity(request)
    value: dict[str, Any] = {
        "schema": RIR_CACHE_ACOUSTIC_SELECTION_SIDECAR_SCHEMA,
        "request_identity_sha256": request_identity_sha256,
        "acoustic_selection_binding_sha256": binding_sha256,
        "acoustic_selection_binding": deepcopy(dict(binding)),
    }
    value["sidecar_content_sha256"] = canonical_json_sha256(value)
    return value


def _verify_acoustic_selection_sidecar(
    cache_root: Path,
    request: Mapping[str, Any],
    *,
    request_identity_sha256: str,
) -> dict[str, Any]:
    """Close the retained selection sidecar over one exact cache request."""

    binding = request.get("acoustic_selection_binding")
    sidecar_path = cache_root / RIR_CACHE_ACOUSTIC_SELECTION_NAME
    if binding is None:
        # Read compatibility for pre-binding explicit caches.  Such caches
        # remain explicitly unbound and can never claim registry selection.
        if sidecar_path.exists():
            legacy_sidecar = load_json(sidecar_path)
            if (
                not isinstance(legacy_sidecar, Mapping)
                or legacy_sidecar.get("schema")
                != RIR_CACHE_ACOUSTIC_SELECTION_INPUT_SCHEMA
                or legacy_sidecar.get("selection_mode") != "explicit"
                or legacy_sidecar.get("registry_resolution") is not None
            ):
                raise RIRCacheError(
                    "pre-binding RIR cache cannot claim registry selection"
                )
            _canonical_record_sha256(
                legacy_sidecar,
                hash_field="effective_selection_content_sha256",
                owner="legacy explicit RIR acoustic selection receipt",
            )
            legacy_receipt: Mapping[str, Any] | None = legacy_sidecar
        else:
            legacy_receipt = None
        return {
            "schema": RIR_CACHE_ACOUSTIC_SELECTION_BINDING_SCHEMA,
            "selection_mode": "explicit_legacy_unbound",
            "registry_selection_applied": False,
            "room_ref": None,
            "profile_ref": None,
            "binding_id": None,
            "registry_selection_content_sha256": None,
            "effective_selection_content_sha256": None,
            "acoustic_package_manifest_sha256": None,
            "simulation_request_sha256": None,
            "input_receipt_sha256": (
                canonical_json_sha256(legacy_receipt)
                if legacy_receipt is not None
                else None
            ),
            "binding_content_sha256": None,
        }
    binding_sha256 = _verify_acoustic_selection_binding(binding)
    if not sidecar_path.is_file():
        raise RIRCacheError(
            "RIR cache lacks its request-bound acoustic selection sidecar"
        )
    sidecar = load_json(sidecar_path)
    if (
        not isinstance(sidecar, Mapping)
        or sidecar.get("schema")
        != RIR_CACHE_ACOUSTIC_SELECTION_SIDECAR_SCHEMA
    ):
        raise RIRCacheError("RIR cache acoustic selection sidecar is invalid")
    _canonical_record_sha256(
        sidecar,
        hash_field="sidecar_content_sha256",
        owner="RIR cache acoustic selection sidecar",
    )
    if (
        sidecar.get("request_identity_sha256")
        != request_identity_sha256
        or sidecar.get("acoustic_selection_binding_sha256")
        != binding_sha256
        or sidecar.get("acoustic_selection_binding") != binding
    ):
        raise RIRCacheError(
            "RIR cache acoustic selection sidecar differs from request identity"
        )
    return deepcopy(dict(binding))


def _retain_acoustic_selection_sidecar(
    output: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Write or validate the core selection sidecar before native rendering."""

    expected = _acoustic_selection_sidecar(request)
    path = output / RIR_CACHE_ACOUSTIC_SELECTION_NAME
    if path.is_file():
        if load_json(path) != expected:
            raise RIRCacheError(
                "existing RIR cache has a different acoustic selection binding"
            )
    else:
        write_json(path, expected)
    return _verify_acoustic_selection_sidecar(
        output,
        request,
        request_identity_sha256=request["request_identity_sha256"],
    )


def _verified_external_file(
    raw_path: Any,
    expected_sha256: Any,
    *,
    owner: str,
) -> dict[str, Any]:
    if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
        raise RIRCacheError(f"{owner} path must be absolute")
    expected = _declared_sha256(expected_sha256, owner=owner)
    try:
        path = Path(raw_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RIRCacheError(f"{owner} path cannot be resolved: {raw_path}") from exc
    if not path.is_file():
        raise RIRCacheError(f"{owner} is not a regular file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RIRCacheError(f"{owner} file SHA-256 differs from the cache request")
    return {
        "declared_path": raw_path,
        "resolved_path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": actual,
    }


def _verify_request_external_inputs(
    value: Mapping[str, Any],
    *,
    expected_plan_path: Path,
) -> dict[str, Any]:
    """Authenticate every request-declared external input still available."""

    plan = _request_section(value, "plan")
    plan_record = _verified_external_file(
        plan.get("path"),
        plan.get("sha256"),
        owner="RIR plan",
    )
    if Path(plan_record["resolved_path"]) != expected_plan_path.resolve(strict=True):
        raise RIRCacheError("RIR cache request plan path differs from the selected plan")

    scene = _request_section(value, "acoustic_scene")
    scene_record = _verified_external_file(
        scene.get("manifest_path"),
        scene.get("manifest_sha256"),
        owner="acoustic package manifest",
    )
    package_id = scene.get("package_id")
    package_content_sha256 = _declared_sha256(
        scene.get("package_content_sha256"),
        owner="acoustic package content",
    )
    if not isinstance(package_id, str) or not package_id:
        raise RIRCacheError("RIR cache acoustic package_id is invalid")
    try:
        manifest = load_json(Path(scene_record["resolved_path"]))
    except (OSError, ValueError, TypeError) as exc:
        raise RIRCacheError("RIR cache acoustic package manifest is unreadable") from exc
    manifest_content = manifest.get("package_content_sha256")
    recomputed_content = canonical_json_sha256(
        {
            key: item
            for key, item in manifest.items()
            if key != "package_content_sha256"
        }
    )
    if (
        manifest.get("package_id") != package_id
        or manifest_content != package_content_sha256
        or recomputed_content != package_content_sha256
    ):
        raise RIRCacheError(
            "RIR cache acoustic scene identity differs from its actual manifest"
        )
    scene_record.update(
        {
            "package_id": package_id,
            "package_content_sha256": package_content_sha256,
            "manifest_content_identity_verified": True,
        }
    )

    simulation = _request_section(value, "simulation")
    simulation_record = _verified_external_file(
        simulation.get("request_path"),
        simulation.get("request_sha256"),
        owner="RIR simulation request",
    )
    acoustic_selection_binding = value.get("acoustic_selection_binding")
    if acoustic_selection_binding is not None:
        binding_sha256 = _verify_acoustic_selection_binding(
            acoustic_selection_binding
        )
        if (
            acoustic_selection_binding.get(
                "acoustic_package_manifest_sha256"
            )
            != scene_record["sha256"]
            or acoustic_selection_binding.get("simulation_request_sha256")
            != simulation_record["sha256"]
        ):
            raise RIRCacheError(
                "RIR acoustic selection binding differs from the request's "
                "effective package or simulation input"
            )
    else:
        binding_sha256 = None

    output = _request_section(value, "output")
    layout_type = output.get("layout_type")
    hrtf_path = output.get("hrtf_path")
    hrtf_sha256 = output.get("hrtf_sha256")
    if layout_type == "binaural":
        hrtf_record: Mapping[str, Any] | None = _verified_external_file(
            hrtf_path,
            hrtf_sha256,
            owner="binaural HRTF",
        )
    elif layout_type == "ambisonics" and hrtf_path is None and hrtf_sha256 is None:
        hrtf_record = None
    else:
        raise RIRCacheError("RIR cache HRTF declaration differs from its layout")
    return {
        "status": "pass",
        "plan": plan_record,
        "acoustic_scene": scene_record,
        "simulation_request": simulation_record,
        "hrtf": hrtf_record,
        "acoustic_selection_binding_sha256": binding_sha256,
    }


def _request_record(
    *,
    plan_path: Path,
    scene: CompiledAcousticScene,
    simulation_request_path: Path,
    simulation: M4SimulationConfig,
    acoustic_selection_receipt: Mapping[str, Any] | None,
    hrtf_file_path: Path | None,
    layout_type: str,
    channel_count: int,
    batch_size: int,
    job_offset: int,
    job_count: int,
    full_plan_job_count: int,
    selected_jobs: Sequence[Mapping[str, Any]],
    translation_m: Sequence[float],
    source_radius_m: float,
    listener_radius_m: float,
    compressed: bool,
) -> dict[str, Any]:
    effective_simulation = simulation_with_layout(
        simulation,
        layout_type=layout_type,
        channel_count=channel_count,
    )
    simulation_request_sha256 = sha256_file(simulation_request_path)
    acoustic_selection_binding = _acoustic_selection_binding(
        acoustic_selection_receipt,
        scene_manifest_path=scene.manifest_path,
        scene_manifest_sha256=scene.manifest_sha256,
        simulation_request_path=simulation_request_path,
        simulation_request_sha256=simulation_request_sha256,
    )
    request = {
        "schema": RIR_CACHE_REQUEST_SCHEMA,
        "plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
            "full_job_count": full_plan_job_count,
            "selected_job_offset": job_offset,
            "selected_job_count": job_count,
            "acoustic_state_binding": "source_listener_pose_per_job_v1",
            "selected_acoustic_states_sha256": canonical_json_sha256(
                [job["acoustic_state_sha256"] for job in selected_jobs]
            ),
        },
        "acoustic_scene": {
            "manifest_path": str(scene.manifest_path),
            "manifest_sha256": scene.manifest_sha256,
            "package_id": scene.package_id,
            "package_content_sha256": scene.package_content_sha256,
        },
        "simulation": {
            "request_path": str(simulation_request_path),
            "request_sha256": simulation_request_sha256,
            "effective": effective_simulation.to_dict(),
        },
        "acoustic_selection_binding": acoustic_selection_binding,
        "output": {
            "layout_type": layout_type,
            "channel_count": channel_count,
            "layout_id": (
                BINAURAL_LAYOUT_ID if layout_type == "binaural" else FOA_LAYOUT_ID
            ),
            "hrtf_path": str(hrtf_file_path) if hrtf_file_path else None,
            "hrtf_sha256": sha256_file(hrtf_file_path) if hrtf_file_path else None,
            "compressed_npz_shards": compressed,
        },
        "runtime_policy": {
            "native_batch_size": batch_size,
            "coordinate_translation_m": list(translation_m),
            "source_radius_m": source_radius_m,
            "listener_radius_m": listener_radius_m,
            "persistent_context": True,
            "listener_pose_update_policy": "set_listener_pose_on_change_v1",
            "scene_upload_count": 1,
            "compute_device": "CPU",
            "gpu_acceleration": False,
        },
    }
    request["request_identity_sha256"] = canonical_json_sha256(
        _request_identity_payload(request)
    )
    _verify_request_identity(request)
    _verify_request_external_inputs(request, expected_plan_path=plan_path)
    return request


def render_rir_cache(
    *,
    plan_path: Path,
    scene: CompiledAcousticScene,
    simulation_request_path: Path,
    simulation: M4SimulationConfig,
    output: Path,
    acoustic_selection_receipt: Mapping[str, Any] | None = None,
    layout_type: str = "binaural",
    hrtf_file_path: Path | None = None,
    batch_size: int = 8,
    job_offset: int = 0,
    job_limit: int | None = None,
    coordinate_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
    source_radius_m: float = 0.0,
    listener_radius_m: float = 0.0,
    compressed: bool = True,
    renderer_factory: Callable[..., Any] = _NativeRIRBatchRenderer,
) -> RIRCacheResult:
    """Render selected jobs, retaining one resumable NPZ shard per native call."""

    started = time.perf_counter()
    plan_path = Path(plan_path).resolve()
    simulation_request_path = Path(simulation_request_path).resolve()
    output = Path(output).resolve()
    if layout_type not in {"binaural", "ambisonics"}:
        raise RIRCacheError("layout_type must be binaural or ambisonics")
    channel_count = 2 if layout_type == "binaural" else 4
    native_batch_size = _positive_int(batch_size, owner="batch size")
    if (
        isinstance(job_offset, bool)
        or not isinstance(job_offset, int)
        or job_offset < 0
    ):
        raise RIRCacheError("job offset must be a nonnegative integer")
    if job_limit is not None:
        job_limit = _positive_int(job_limit, owner="job limit")
    translation = _finite_vector(
        list(coordinate_translation_m), 3, owner="coordinate translation"
    )
    for value, owner in (
        (source_radius_m, "source radius"),
        (listener_radius_m, "listener radius"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RIRCacheError(f"{owner} must be finite and nonnegative")

    plan = load_json(plan_path)
    jobs = validate_rir_job_plan(plan)
    if job_offset >= len(jobs):
        raise RIRCacheError("job offset is outside the RIR plan")
    stop = len(jobs) if job_limit is None else min(len(jobs), job_offset + job_limit)
    selected_jobs = jobs[job_offset:stop]
    if not selected_jobs:
        raise RIRCacheError("selected RIR job range is empty")
    hrtf = Path(hrtf_file_path).resolve() if hrtf_file_path else None
    request = _request_record(
        plan_path=plan_path,
        scene=scene,
        simulation_request_path=simulation_request_path,
        simulation=simulation,
        acoustic_selection_receipt=acoustic_selection_receipt,
        hrtf_file_path=hrtf,
        layout_type=layout_type,
        channel_count=channel_count,
        batch_size=native_batch_size,
        job_offset=job_offset,
        job_count=len(selected_jobs),
        full_plan_job_count=len(jobs),
        selected_jobs=selected_jobs,
        translation_m=translation,
        source_radius_m=float(source_radius_m),
        listener_radius_m=float(listener_radius_m),
        compressed=compressed,
    )
    request_path = output / "request.json"
    retained_timing: dict[str, Any] | None = None
    if output.exists():
        if not request_path.is_file() or load_json(request_path) != request:
            raise RIRCacheError("existing cache output has a different request")
        _verify_acoustic_selection_sidecar(
            output,
            request,
            request_identity_sha256=request["request_identity_sha256"],
        )
        timing_path = output / "timing.json"
        if timing_path.is_file():
            candidate = load_json(timing_path)
            if (
                candidate.get("schema") != RIR_CACHE_TIMING_SCHEMA
                or candidate.get("status") != "pass"
                or candidate.get("selected_job_count") != len(selected_jobs)
            ):
                raise RIRCacheError("existing cache timing record is invalid")
            retained_timing = dict(candidate)
    else:
        output.mkdir(parents=True)
        write_json(request_path, request)
        _retain_acoustic_selection_sidecar(output, request)
    shards_root = output / "shards"
    shards_root.mkdir(exist_ok=True)

    translation_array = np.asarray(translation, dtype=np.float64)
    expected_sample_rate_hz = int(round(simulation.sample_rate_hz))
    expected_layout_id = request["output"]["layout_id"]
    expected_channel_labels = (
        ("left", "right") if layout_type == "binaural" else ("W", "Y", "Z", "X")
    )
    jobs_by_listener_pose: dict[
        tuple[float, ...], list[tuple[int, Mapping[str, Any]]]
    ] = {}
    for selected_index, job in enumerate(selected_jobs):
        listener_key = (
            *tuple(float(value) for value in job["listener_position_m"]),
            *tuple(float(value) for value in job["listener_orientation_wxyz"]),
        )
        jobs_by_listener_pose.setdefault(listener_key, []).append(
            (job_offset + selected_index, job)
        )
    batch_specs: list[dict[str, Any]] = []
    for listener_key, indexed_jobs in jobs_by_listener_pose.items():
        for start in range(0, len(indexed_jobs), native_batch_size):
            chunk = indexed_jobs[start : start + native_batch_size]
            batch_specs.append(
                {
                    "absolute_job_indices": [item[0] for item in chunk],
                    "jobs": [item[1] for item in chunk],
                    "listener_position_m": (
                        np.asarray(listener_key[:3], dtype=np.float64)
                        + translation_array
                    ).tolist(),
                    "listener_orientation_wxyz": list(listener_key[3:]),
                }
            )
    renderer: Any | None = None
    setup_report: dict[str, Any] | None = None
    batch_records: list[dict[str, Any]] = []
    new_job_count = 0
    completed_job_count = 0
    try:
        for batch_index, batch_spec in enumerate(batch_specs):
            shard_path = shards_root / f"shard_{batch_index:06d}.npz"
            absolute_indices = np.asarray(
                batch_spec["absolute_job_indices"], dtype="<u4"
            )
            batch_jobs = batch_spec["jobs"]
            listener_position = batch_spec["listener_position_m"]
            listener_orientation = batch_spec["listener_orientation_wxyz"]
            positions = [
                (
                    np.asarray(job["source_position_m"], dtype=np.float64)
                    + translation_array
                ).tolist()
                for job in batch_jobs
            ]
            listener_positions = [listener_position] * len(batch_jobs)
            listener_orientations = [listener_orientation] * len(batch_jobs)
            acoustic_state_sha256 = [
                str(job["acoustic_state_sha256"]) for job in batch_jobs
            ]
            if shard_path.is_file():
                retained = _read_shard(shard_path)
                _verify_shard_request(
                    retained,
                    path=shard_path,
                    expected_job_indices=absolute_indices,
                    expected_job_ids=[job["job_id"] for job in batch_jobs],
                    expected_source_positions_m=positions,
                    expected_listener_positions_m=listener_positions,
                    expected_listener_orientations_wxyz=listener_orientations,
                    expected_acoustic_state_sha256=acoustic_state_sha256,
                    sample_rate_hz=expected_sample_rate_hz,
                    layout_id=expected_layout_id,
                    channel_labels=expected_channel_labels,
                )
                batch_records.append(
                    {
                        "batch_index": batch_index,
                        "job_count": len(batch_jobs),
                        "simulate_wall_seconds": float(
                            retained["simulate_wall_seconds"]
                        ),
                        "simulate_process_cpu_seconds": float(
                            retained["simulate_process_cpu_seconds"]
                        ),
                        "serialization_wall_seconds": 0.0,
                        "resumed": True,
                    }
                )
                completed_job_count += len(batch_jobs)
                continue
            if renderer is None:
                initial = positions + [positions[-1]] * (
                    native_batch_size - len(positions)
                )
                renderer = renderer_factory(
                    scene,
                    simulation,
                    batch_size=native_batch_size,
                    initial_positions_m=initial,
                    listener_position_m=listener_position,
                    listener_orientation_wxyz=listener_orientation,
                    layout_type=layout_type,
                    channel_count=channel_count,
                    hrtf_file_path=str(hrtf) if hrtf else "",
                    source_radius_m=float(source_radius_m),
                    listener_radius_m=float(listener_radius_m),
                )
                setup_report = dict(renderer.setup_report)
            if plan.get("listener_pose_mode") in {None, "fixed"}:
                # Preserve the historical fixed-listener renderer protocol:
                # the constructor owns the one pose and render() only receives
                # source positions.  Per-frame plans require the pose-aware
                # protocol below and fail rather than silently staying fixed.
                result: RIRBatchResult = renderer.render(positions)
            else:
                result = renderer.render(
                    positions,
                    listener_position_m=listener_position,
                    listener_orientation_wxyz=listener_orientation,
                )
            maximum_length = max(samples.shape[1] for samples in result.samples)
            padded = np.zeros(
                (len(result.samples), channel_count, maximum_length), dtype="<f4"
            )
            lengths = np.empty(len(result.samples), dtype="<u4")
            hashes: list[str] = []
            for row, samples in enumerate(result.samples):
                length = samples.shape[1]
                padded[row, :, :length] = samples
                lengths[row] = length
                hashes.append(hashlib.sha256(samples.tobytes(order="C")).hexdigest())
            serialize_start = time.perf_counter()
            _atomic_savez(
                shard_path,
                compressed=compressed,
                arrays={
                    "job_indices": absolute_indices,
                    "job_ids": np.asarray([job["job_id"] for job in batch_jobs]),
                    "source_positions_m": np.asarray(positions, dtype="<f8"),
                    "listener_positions_m": np.asarray(
                        listener_positions, dtype="<f8"
                    ),
                    "listener_orientations_wxyz": np.asarray(
                        listener_orientations, dtype="<f8"
                    ),
                    "acoustic_state_sha256": np.asarray(acoustic_state_sha256),
                    "lengths": lengths,
                    "samples": padded,
                    "ir_sha256": np.asarray(hashes),
                    "sample_rate_hz": np.asarray(result.sample_rate_hz, dtype="<u4"),
                    "layout_id": np.asarray(result.layout_id),
                    "channel_labels": np.asarray(result.channel_labels),
                    "simulate_wall_seconds": np.asarray(
                        result.wall_seconds, dtype="<f8"
                    ),
                    "simulate_process_cpu_seconds": np.asarray(
                        result.process_cpu_seconds, dtype="<f8"
                    ),
                    "indirect_ray_efficiency": np.asarray(
                        result.indirect_ray_efficiency, dtype="<f8"
                    ),
                },
            )
            serialization_seconds = time.perf_counter() - serialize_start
            new_job_count += len(batch_jobs)
            completed_job_count += len(batch_jobs)
            batch_records.append(
                {
                    "batch_index": batch_index,
                    "job_count": len(batch_jobs),
                    "simulate_wall_seconds": result.wall_seconds,
                    "simulate_process_cpu_seconds": result.process_cpu_seconds,
                    "serialization_wall_seconds": serialization_seconds,
                    "resumed": False,
                }
            )
            write_json(
                output / "progress.json",
                {
                    "schema": RIR_CACHE_RECEIPT_SCHEMA,
                    "status": "research_only",
                    "completed_batch_count": batch_index + 1,
                    "batch_count": len(batch_specs),
                    "completed_job_count": completed_job_count,
                    "selected_job_count": len(selected_jobs),
                },
            )
            print(
                "RIR_CACHE_BATCH_OK "
                f"batch={batch_index + 1}/{len(batch_specs)} "
                f"jobs={len(batch_jobs)} wall={result.wall_seconds:.3f}s",
                flush=True,
            )
    except Exception:
        write_json(
            output / "FAILED.json",
            {
                "schema": RIR_CACHE_RECEIPT_SCHEMA,
                "status": "fail",
                "completed_batch_count": sum(
                    (shards_root / f"shard_{index:06d}.npz").is_file()
                    for index in range(len(batch_specs))
                ),
                "batch_count": len(batch_specs),
            },
        )
        raise

    index_entries: list[dict[str, Any]] = []
    all_batch_records: list[dict[str, Any]] = []
    for batch_index, batch_spec in enumerate(batch_specs):
        shard_path = shards_root / f"shard_{batch_index:06d}.npz"
        retained = _read_shard(shard_path)
        batch_jobs = batch_spec["jobs"]
        expected_indices = np.asarray(
            batch_spec["absolute_job_indices"], dtype="<u4"
        )
        expected_positions = [
            (
                np.asarray(job["source_position_m"], dtype=np.float64)
                + translation_array
            ).tolist()
            for job in batch_jobs
        ]
        expected_listener_positions = [
            batch_spec["listener_position_m"]
        ] * len(batch_jobs)
        expected_listener_orientations = [
            batch_spec["listener_orientation_wxyz"]
        ] * len(batch_jobs)
        expected_acoustic_states = [
            str(job["acoustic_state_sha256"]) for job in batch_jobs
        ]
        _verify_shard_request(
            retained,
            path=shard_path,
            expected_job_indices=expected_indices,
            expected_job_ids=[job["job_id"] for job in batch_jobs],
            expected_source_positions_m=expected_positions,
            expected_listener_positions_m=expected_listener_positions,
            expected_listener_orientations_wxyz=expected_listener_orientations,
            expected_acoustic_state_sha256=expected_acoustic_states,
            sample_rate_hz=expected_sample_rate_hz,
            layout_id=expected_layout_id,
            channel_labels=expected_channel_labels,
        )
        for row, job in enumerate(batch_jobs):
            index_entries.append(
                {
                    "job_id": job["job_id"],
                    "job_index": int(expected_indices[row]),
                    "shard": shard_path.relative_to(output).as_posix(),
                    "row": row,
                    "sample_count": int(retained["lengths"][row]),
                    "ir_sha256": str(retained["ir_sha256"][row]),
                    "source_position_m": retained["source_positions_m"][row].tolist(),
                    "listener_position_m": retained[
                        "listener_positions_m"
                    ][row].tolist(),
                    "listener_orientation_wxyz": retained[
                        "listener_orientations_wxyz"
                    ][row].tolist(),
                    "acoustic_state_sha256": str(
                        retained["acoustic_state_sha256"][row]
                    ),
                }
            )
        matching = next(
            record for record in batch_records if record["batch_index"] == batch_index
        )
        all_batch_records.append(matching)
    index_entries.sort(key=lambda entry: int(entry["job_index"]))
    write_json(
        output / "index.json",
        {
            "schema": RIR_CACHE_INDEX_SCHEMA,
            "status": "pass",
            "request_identity_sha256": request["request_identity_sha256"],
            "acoustic_selection_binding_sha256": request[
                "acoustic_selection_binding"
            ]["binding_content_sha256"],
            "acoustic_selection_mode": request[
                "acoustic_selection_binding"
            ]["selection_mode"],
            "acoustic_state_binding": "source_listener_pose_per_job_v1",
            "full_plan_complete": len(selected_jobs) == len(jobs) and job_offset == 0,
            "selected_job_count": len(selected_jobs),
            "entries": index_entries,
        },
    )
    simulate_seconds = sum(
        float(record["simulate_wall_seconds"]) for record in all_batch_records
    )
    serialization_seconds = sum(
        float(record["serialization_wall_seconds"]) for record in all_batch_records
    )
    jobs_per_second = len(selected_jobs) / simulate_seconds
    if new_job_count == 0 and retained_timing is not None:
        timing = retained_timing
    else:
        timing = {
            "schema": RIR_CACHE_TIMING_SCHEMA,
            "status": "pass",
            "setup": setup_report,
            "batches": all_batch_records,
            "selected_job_count": len(selected_jobs),
            "new_job_count": new_job_count,
            "simulate_wall_seconds": simulate_seconds,
            "serialization_wall_seconds": serialization_seconds,
            "run_wall_seconds": time.perf_counter() - started,
            "jobs_per_simulate_second": jobs_per_second,
            "projected_full_plan_simulate_seconds": len(jobs) / jobs_per_second,
        }
        write_json(output / "timing.json", timing)
    total_bytes = sum(path.stat().st_size for path in shards_root.glob("*.npz"))
    receipt = {
        "schema": RIR_CACHE_RECEIPT_SCHEMA,
        "status": "pass",
        "qualification_claim": False,
        "claim_boundary": (
            "native RLR output cache for the exact request; room/material truth "
            "remains governed by the supplied acoustic package"
        ),
        "request_identity_sha256": request["request_identity_sha256"],
        "acoustic_selection_binding_sha256": request[
            "acoustic_selection_binding"
        ]["binding_content_sha256"],
        "acoustic_selection_mode": request[
            "acoustic_selection_binding"
        ]["selection_mode"],
        "full_plan_complete": len(selected_jobs) == len(jobs) and job_offset == 0,
        "full_plan_job_count": len(jobs),
        "selected_job_count": len(selected_jobs),
        "retained_shard_count": len(batch_specs),
        "retained_shard_bytes": total_bytes,
        "sample_rate_hz": int(round(simulation.sample_rate_hz)),
        "layout_type": layout_type,
        "layout_id": request["output"]["layout_id"],
        "channel_count": channel_count,
        "producer_backend": "RLR Audio Propagation",
        "cache_artifact": "room impulse response (RIR)",
        "dry_audio_independent": True,
        "acoustic_state_binding": "source_listener_pose_per_job_v1",
        "listener_pose_update_policy": "set_listener_pose_on_change_v1",
        "retained_payload_hash_verified": True,
        "rerun_byte_identity_claim": False,
        "native_random_seed_control": "unavailable_in_current_RLR_API",
        "compute_device": "CPU",
        "configured_thread_count": simulation.thread_count,
        "outputs": {
            "request": "request.json",
            "acoustic_selection": RIR_CACHE_ACOUSTIC_SELECTION_NAME,
            "index": "index.json",
            "timing": "timing.json",
            "shards": "shards/",
        },
    }
    write_json(output / "receipt.json", receipt)
    failed_path = output / "FAILED.json"
    if failed_path.exists():
        failed_path.unlink()
    write_json(output / "progress.json", receipt)
    return RIRCacheResult(output=output, receipt=receipt)


def _render_semantic_rir_cache_staging(
    *,
    plan_path: Path,
    scene: SemanticAcousticScene,
    simulation_request_path: Path,
    simulation: M4SimulationConfig,
    output: Path,
    acoustic_selection: Mapping[str, Any] | None = None,
    layout_type: str = "binaural",
    hrtf_file_path: Path | None = None,
    batch_size: int = 8,
    coordinate_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
    source_radius_m: float = 0.0,
    listener_radius_m: float = 0.0,
    compressed: bool = True,
    renderer_factory: Callable[..., Any] = _SemanticNativeRIRBatchRenderer,
) -> RIRCacheResult:
    """Render a fresh structural/sample cache without file evidence."""

    started = time.perf_counter()
    raw_plan = Path(plan_path)
    raw_simulation = Path(simulation_request_path)
    raw_output = Path(output)
    plan_path = _semantic_regular_file(raw_plan, owner="semantic RIR plan")
    simulation_request_path = _semantic_regular_file(
        raw_simulation, owner="semantic simulation request"
    )
    output = Path(os.path.abspath(raw_output))
    if _semantic_path_has_symlink_component(output):
        raise RIRCacheError(
            "semantic RIR output path must not contain symlink components"
        )
    if output.exists():
        raise RIRCacheError("semantic RIR output must be a fresh path")
    if not isinstance(scene, SemanticAcousticScene):
        raise RIRCacheError("semantic RIR scene is invalid")
    if layout_type != "binaural":
        raise RIRCacheError("semantic RIR mode supports binaural output only")
    channel_count = 2
    native_batch_size = _positive_int(batch_size, owner="semantic batch size")
    if not isinstance(compressed, bool):
        raise RIRCacheError("semantic shard compression policy must be boolean")
    translation = _finite_vector(
        list(coordinate_translation_m), 3, owner="semantic coordinate translation"
    )
    for value, owner in (
        (source_radius_m, "semantic source radius"),
        (listener_radius_m, "semantic listener radius"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise RIRCacheError(f"{owner} must be finite and nonnegative")
    plan = load_json(plan_path)
    jobs = validate_semantic_rir_job_plan(plan)
    binding = _semantic_selection_binding(acoustic_selection)
    native_execution = renderer_factory is _SemanticNativeRIRBatchRenderer
    if hrtf_file_path is None:
        raise RIRCacheError("semantic binaural RIR mode requires an HRTF")
    effective_simulation = simulation_with_layout(
        simulation, layout_type=layout_type, channel_count=channel_count
    )
    hrtf = _semantic_regular_file(
        Path(hrtf_file_path),
        owner="semantic HRTF",
        absolute_required=True,
    )
    layout_id = BINAURAL_LAYOUT_ID if layout_type == "binaural" else FOA_LAYOUT_ID
    channel_labels = (
        ("left", "right") if layout_type == "binaural" else ("W", "Y", "Z", "X")
    )
    request = {
        "schema": SEMANTIC_RIR_CACHE_REQUEST_SCHEMA,
        "status": "ready_structural_and_sample_validation",
        "qualification_claim": False,
        "plan": {
            "path": str(plan_path),
            "full_job_count": len(jobs),
            "selected_job_offset": 0,
            "selected_job_count": len(jobs),
            "acoustic_state_binding": "source_listener_pose_per_job_v1",
        },
        "acoustic_scene": {
            "manifest_path": str(scene.manifest_path),
            "package_id": scene.package_id,
        },
        "simulation": {
            "request_path": str(simulation_request_path),
            "effective": effective_simulation.to_dict(),
        },
        "acoustic_selection_binding": binding,
        "output": {
            "layout_type": layout_type,
            "channel_count": channel_count,
            "layout_id": layout_id,
            "hrtf_path": str(hrtf) if hrtf else None,
            "compressed_npz_shards": compressed,
        },
        "runtime_policy": {
            "native_batch_size": native_batch_size,
            "coordinate_translation_m": list(translation),
            "source_radius_m": float(source_radius_m),
            "listener_radius_m": float(listener_radius_m),
            "persistent_context": True,
            "listener_pose_update_policy": "set_listener_pose_on_change_v1",
            "scene_upload_count": 1,
            "compute_device": "CPU",
            "gpu_acceleration": False,
            "execution_mode": (
                "native_default" if native_execution else "injected_test_double"
            ),
        },
    }
    output.mkdir(parents=True)
    write_json(output / "request.json", request)
    shards_root = output / "shards"
    shards_root.mkdir()
    translation_array = np.asarray(translation, dtype=np.float64)
    grouped: dict[tuple[float, ...], list[tuple[int, Mapping[str, Any]]]] = {}
    for index, job in enumerate(jobs):
        listener_key = (
            *tuple(float(value) for value in job["listener_position_m"]),
            *tuple(float(value) for value in job["listener_orientation_wxyz"]),
        )
        grouped.setdefault(listener_key, []).append((index, job))
    batch_specs: list[dict[str, Any]] = []
    for listener_key, indexed_jobs in grouped.items():
        for start in range(0, len(indexed_jobs), native_batch_size):
            selected = indexed_jobs[start : start + native_batch_size]
            batch_specs.append(
                {
                    "job_indices": [item[0] for item in selected],
                    "jobs": [item[1] for item in selected],
                    "listener_position_m": (
                        np.asarray(listener_key[:3], dtype=np.float64)
                        + translation_array
                    ).tolist(),
                    "listener_orientation_wxyz": list(listener_key[3:]),
                }
            )
    renderer: Any | None = None
    setup_report: Mapping[str, Any] | None = None
    batch_records: list[dict[str, Any]] = []
    try:
        for batch_index, spec in enumerate(batch_specs):
            jobs_in_batch = spec["jobs"]
            positions = [
                (
                    np.asarray(job["source_position_m"], dtype=np.float64)
                    + translation_array
                ).tolist()
                for job in jobs_in_batch
            ]
            if renderer is None:
                initial = positions + [positions[-1]] * (
                    native_batch_size - len(positions)
                )
                renderer = renderer_factory(
                    scene,
                    simulation,
                    batch_size=native_batch_size,
                    initial_positions_m=initial,
                    listener_position_m=spec["listener_position_m"],
                    listener_orientation_wxyz=spec["listener_orientation_wxyz"],
                    layout_type=layout_type,
                    channel_count=channel_count,
                    hrtf_file_path=str(hrtf) if hrtf else "",
                    source_radius_m=float(source_radius_m),
                    listener_radius_m=float(listener_radius_m),
                )
                setup_report = deepcopy(dict(renderer.setup_report))
            if plan.get("listener_pose_mode") == "fixed":
                result: RIRBatchResult = renderer.render(positions)
            else:
                result = renderer.render(
                    positions,
                    listener_position_m=spec["listener_position_m"],
                    listener_orientation_wxyz=spec["listener_orientation_wxyz"],
                )
            if (
                not isinstance(result, RIRBatchResult)
                or len(result.samples) != len(jobs_in_batch)
                or result.sample_rate_hz
                != int(round(effective_simulation.sample_rate_hz))
                or result.layout_id != BINAURAL_LAYOUT_ID
                or tuple(result.channel_labels) != ("left", "right")
                or not math.isfinite(float(result.wall_seconds))
                or float(result.wall_seconds) < 0
                or not math.isfinite(float(result.process_cpu_seconds))
                or float(result.process_cpu_seconds) < 0
                or not math.isfinite(float(result.indirect_ray_efficiency))
                or not 0 <= float(result.indirect_ray_efficiency) <= 1
            ):
                raise RIRCacheError("semantic renderer result contract is invalid")
            for samples in result.samples:
                if (
                    not isinstance(samples, np.ndarray)
                    or samples.dtype != np.dtype("<f4")
                    or samples.ndim != 2
                    or samples.shape[0] != 2
                    or samples.shape[1] < 2
                    or not samples.flags.c_contiguous
                    or not np.all(np.isfinite(samples))
                    or not np.any(samples)
                ):
                    raise RIRCacheError(
                        "semantic renderer returned an invalid binaural sample payload"
                    )
            maximum_length = max(item.shape[1] for item in result.samples)
            padded = np.zeros(
                (len(result.samples), channel_count, maximum_length), dtype="<f4"
            )
            lengths = np.empty(len(result.samples), dtype="<u4")
            for row, samples in enumerate(result.samples):
                length = samples.shape[1]
                padded[row, :, :length] = samples
                lengths[row] = length
            serialization_started = time.perf_counter()
            shard_path = shards_root / f"shard_{batch_index:06d}.npz"
            _atomic_savez(
                shard_path,
                compressed=compressed,
                arrays={
                    "job_indices": np.asarray(spec["job_indices"], dtype="<u4"),
                    "job_ids": np.asarray([job["job_id"] for job in jobs_in_batch]),
                    "source_positions_m": np.asarray(positions, dtype="<f8"),
                    "listener_positions_m": np.asarray(
                        [spec["listener_position_m"]] * len(jobs_in_batch),
                        dtype="<f8",
                    ),
                    "listener_orientations_wxyz": np.asarray(
                        [spec["listener_orientation_wxyz"]] * len(jobs_in_batch),
                        dtype="<f8",
                    ),
                    "lengths": lengths,
                    "samples": padded,
                    "sample_rate_hz": np.asarray(result.sample_rate_hz, dtype="<u4"),
                    "layout_id": np.asarray(result.layout_id),
                    "channel_labels": np.asarray(result.channel_labels),
                    "simulate_wall_seconds": np.asarray(
                        result.wall_seconds, dtype="<f8"
                    ),
                    "simulate_process_cpu_seconds": np.asarray(
                        result.process_cpu_seconds, dtype="<f8"
                    ),
                    "indirect_ray_efficiency": np.asarray(
                        result.indirect_ray_efficiency, dtype="<f8"
                    ),
                },
            )
            batch_records.append(
                {
                    "batch_index": batch_index,
                    "job_count": len(jobs_in_batch),
                    "simulate_wall_seconds": result.wall_seconds,
                    "simulate_process_cpu_seconds": result.process_cpu_seconds,
                    "serialization_wall_seconds": (
                        time.perf_counter() - serialization_started
                    ),
                }
            )
            write_json(
                output / "progress.json",
                {
                    "schema": SEMANTIC_RIR_CACHE_RECEIPT_SCHEMA,
                    "status": "research_only",
                    "completed_batch_count": batch_index + 1,
                    "batch_count": len(batch_specs),
                    "completed_job_count": sum(
                        item["job_count"] for item in batch_records
                    ),
                    "selected_job_count": len(jobs),
                    "qualification_claim": False,
                },
            )
    except Exception:
        write_json(
            output / "FAILED.json",
            {
                "schema": SEMANTIC_RIR_CACHE_RECEIPT_SCHEMA,
                "status": "fail",
                "completed_batch_count": len(batch_records),
                "batch_count": len(batch_specs),
                "qualification_claim": False,
            },
        )
        raise
    native_call_count = (
        int(getattr(renderer, "native_simulate_owned_call_count", 0))
        if native_execution
        else 0
    )
    native_job_count = (
        int(getattr(renderer, "native_realized_job_count", 0))
        if native_execution
        else 0
    )
    if native_execution and (
        native_call_count != len(batch_records) or native_job_count != len(jobs)
    ):
        raise RIRCacheError("semantic native execution counters are incomplete")
    index_entries: list[dict[str, Any]] = []
    for batch_index, spec in enumerate(batch_specs):
        path = shards_root / f"shard_{batch_index:06d}.npz"
        retained = _read_semantic_shard(path)
        jobs_in_batch = spec["jobs"]
        positions = [
            (
                np.asarray(job["source_position_m"], dtype=np.float64)
                + translation_array
            ).tolist()
            for job in jobs_in_batch
        ]
        _verify_semantic_shard_request(
            retained,
            path=path,
            expected_job_indices=np.asarray(spec["job_indices"], dtype="<u4"),
            expected_jobs=jobs_in_batch,
            expected_positions=positions,
            expected_listener_positions=[spec["listener_position_m"]]
            * len(jobs_in_batch),
            expected_listener_orientations=[spec["listener_orientation_wxyz"]]
            * len(jobs_in_batch),
            sample_rate_hz=int(round(simulation.sample_rate_hz)),
            layout_id=layout_id,
            channel_labels=channel_labels,
        )
        for row, job in enumerate(jobs_in_batch):
            index_entries.append(
                {
                    "job_id": job["job_id"],
                    "job_index": int(spec["job_indices"][row]),
                    "shard": path.relative_to(output).as_posix(),
                    "row": row,
                    "sample_count": int(retained["lengths"][row]),
                    "source_position_m": retained["source_positions_m"][row].tolist(),
                    "listener_position_m": retained["listener_positions_m"][
                        row
                    ].tolist(),
                    "listener_orientation_wxyz": retained["listener_orientations_wxyz"][
                        row
                    ].tolist(),
                }
            )
    index_entries.sort(key=lambda item: int(item["job_index"]))
    index = {
        "schema": SEMANTIC_RIR_CACHE_INDEX_SCHEMA,
        "status": "pass",
        "qualification_claim": False,
        "full_plan_complete": True,
        "selected_job_count": len(jobs),
        "acoustic_state_binding": "source_listener_pose_per_job_v1",
        "acoustic_selection_mode": binding["selection_mode"],
        "entries": index_entries,
    }
    write_json(output / "index.json", index)
    simulate_seconds = sum(item["simulate_wall_seconds"] for item in batch_records)
    timing = {
        "schema": "avengine_semantic_rir_cache_timing_v1",
        "status": "pass",
        "setup": setup_report,
        "batches": batch_records,
        "selected_job_count": len(jobs),
        "simulate_wall_seconds": simulate_seconds,
        "serialization_wall_seconds": sum(
            item["serialization_wall_seconds"] for item in batch_records
        ),
        "run_wall_seconds": time.perf_counter() - started,
        "jobs_per_simulate_second": (
            len(jobs) / simulate_seconds if simulate_seconds > 0 else None
        ),
    }
    write_json(output / "timing.json", timing)
    receipt = {
        "schema": SEMANTIC_RIR_CACHE_RECEIPT_SCHEMA,
        "status": "pass",
        "qualification_claim": False,
        "claim_boundary": (
            (
                "native CPU RIR samples with structural pose/use, native source/listener "
                "receipts, and decoded-sample validation"
            )
            if native_execution
            else "test-double samples with structural and decoded-sample validation"
        ),
        "native_execution": native_execution,
        "native_scene_upload_structurally_validated": native_execution,
        "native_source_listener_receipts_validated": native_execution,
        "native_realized_job_count": native_job_count,
        "native_simulate_owned_call_count": native_call_count,
        "full_plan_complete": True,
        "full_plan_job_count": len(jobs),
        "selected_job_count": len(jobs),
        "retained_shard_count": len(batch_specs),
        "sample_rate_hz": int(round(simulation.sample_rate_hz)),
        "layout_type": layout_type,
        "layout_id": layout_id,
        "channel_count": channel_count,
        "producer_backend": (
            "RLR Audio Propagation"
            if native_execution
            else "test_only_injected_renderer"
        ),
        "cache_artifact": "room impulse response (RIR)",
        "dry_audio_independent": True,
        "acoustic_state_binding": "source_listener_pose_per_job_v1",
        "listener_pose_update_policy": "set_listener_pose_on_change_v1",
        "acoustic_selection_mode": binding["selection_mode"],
        "compute_device": "CPU",
        "configured_thread_count": simulation.thread_count,
        "outputs": {
            "request": "request.json",
            "index": "index.json",
            "timing": "timing.json",
            "shards": "shards/",
        },
    }
    write_json(output / "receipt.json", receipt)
    write_json(output / "progress.json", receipt)
    return RIRCacheResult(output=output, receipt=receipt)


def render_semantic_rir_cache(
    *,
    plan_path: Path,
    scene: SemanticAcousticScene,
    simulation_request_path: Path,
    simulation: M4SimulationConfig,
    output: Path,
    acoustic_selection: Mapping[str, Any] | None = None,
    layout_type: str = "binaural",
    hrtf_file_path: Path | None = None,
    batch_size: int = 8,
    coordinate_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
    source_radius_m: float = 0.0,
    listener_radius_m: float = 0.0,
    compressed: bool = True,
    renderer_factory: Callable[..., Any] = _SemanticNativeRIRBatchRenderer,
) -> RIRCacheResult:
    """Render privately and atomically publish a fresh semantic cache."""

    destination = Path(os.path.abspath(output))
    if (
        _semantic_path_has_symlink_component(destination)
        or destination.exists()
        or destination.is_symlink()
    ):
        raise RIRCacheError("semantic RIR output must be a fresh non-symlink path")
    parent = destination.parent
    if _semantic_path_has_symlink_component(parent) or not parent.is_dir():
        raise RIRCacheError("semantic RIR output parent must be an existing directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    staging.rmdir()
    try:
        result = _render_semantic_rir_cache_staging(
            plan_path=plan_path,
            scene=scene,
            simulation_request_path=simulation_request_path,
            simulation=simulation,
            output=staging,
            acoustic_selection=acoustic_selection,
            layout_type=layout_type,
            hrtf_file_path=hrtf_file_path,
            batch_size=batch_size,
            coordinate_translation_m=coordinate_translation_m,
            source_radius_m=source_radius_m,
            listener_radius_m=listener_radius_m,
            compressed=compressed,
            renderer_factory=renderer_factory,
        )
        policy = WorkspacePathPolicy.from_roots([parent])
        published = atomic_publish_directory(policy, result.output, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RIRCacheResult(output=published, receipt=result.receipt)


class RIRCacheSession:
    """One validated cache/plan closure with process-local shard residency.

    This is the batch counterpart to :func:`load_cached_rir_episode`: request,
    plan and index closure are checked once, and a shard is fully read and
    hash-verified only on first use.  It is intentionally a read-only cache;
    callers never mutate RIR arrays or suppress a normal validation check.
    """

    def __init__(
        self,
        *,
        cache_root: str | Path,
        plan_path: str | Path,
        frame_count: int,
        frame_rate_hz: int,
        shared_shard_cache: dict[Path, dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(cache_root).resolve()
        plan_source = Path(plan_path).resolve()
        if not self.root.is_dir() or not plan_source.is_file():
            raise RIRCacheError("cached episode requires a cache root and RIR plan")
        self.frames = _positive_int(frame_count, owner="cached episode frame count")
        self.fps = _positive_int(frame_rate_hz, owner="cached episode frame rate")
        request = load_json(self.root / "request.json")
        receipt = load_json(self.root / "receipt.json")
        index = load_json(self.root / "index.json")
        jobs = validate_rir_job_plan(load_json(plan_source))
        self.plan_sha256 = sha256_file(plan_source)
        request_identity = _verify_request_identity(request)
        acoustic_selection_binding = _verify_acoustic_selection_sidecar(
            self.root,
            request,
            request_identity_sha256=request_identity,
        )
        binding_sha256 = acoustic_selection_binding.get(
            "binding_content_sha256"
        )
        self.external_input_identity = _verify_request_external_inputs(
            request,
            expected_plan_path=plan_source,
        )
        request_plan = request.get("plan", {})
        self.listener_pose_bound = (
            request_plan.get("acoustic_state_binding")
            == "source_listener_pose_per_job_v1"
        )
        if self.listener_pose_bound:
            expected_states_sha256 = canonical_json_sha256(
                [job["acoustic_state_sha256"] for job in jobs]
            )
            if (
                request_plan.get("selected_acoustic_states_sha256")
                != expected_states_sha256
                or receipt.get("acoustic_state_binding")
                != "source_listener_pose_per_job_v1"
                or receipt.get("listener_pose_update_policy")
                != "set_listener_pose_on_change_v1"
                or index.get("acoustic_state_binding")
                != "source_listener_pose_per_job_v1"
            ):
                raise RIRCacheError(
                    "RIR cache Listener-pose binding differs across closure"
                )
        if (
            request.get("schema") != RIR_CACHE_REQUEST_SCHEMA
            or request_plan.get("sha256") != self.plan_sha256
            or request_plan.get("full_job_count") != len(jobs)
            or request_plan.get("selected_job_offset") != 0
            or request_plan.get("selected_job_count") != len(jobs)
            or receipt.get("schema") != RIR_CACHE_RECEIPT_SCHEMA
            or receipt.get("status") != "pass"
            or receipt.get("full_plan_complete") is not True
            or receipt.get("full_plan_job_count") != len(jobs)
            or receipt.get("selected_job_count") != len(jobs)
            or receipt.get("request_identity_sha256") != request_identity
            or index.get("schema") != RIR_CACHE_INDEX_SCHEMA
            or index.get("status") != "pass"
            or index.get("full_plan_complete") is not True
            or index.get("selected_job_count") != len(jobs)
            or index.get("request_identity_sha256") != request_identity
        ):
            raise RIRCacheError("RIR cache request/receipt/index closure is invalid")
        if request.get("acoustic_selection_binding") is not None and (
            receipt.get("acoustic_selection_binding_sha256")
            != binding_sha256
            or index.get("acoustic_selection_binding_sha256")
            != binding_sha256
            or receipt.get("acoustic_selection_mode")
            != acoustic_selection_binding["selection_mode"]
            or index.get("acoustic_selection_mode")
            != acoustic_selection_binding["selection_mode"]
        ):
            raise RIRCacheError(
                "RIR cache acoustic selection binding differs across closure"
            )
        output = request.get("output")
        if not isinstance(output, Mapping):
            raise RIRCacheError("RIR cache output contract is missing")
        self.layout_type = output.get("layout_type")
        self.layout_id = output.get("layout_id")
        self.channel_count = output.get("channel_count")
        self.expected_labels = (
            ("left", "right") if self.layout_type == "binaural" else
            ("W", "Y", "Z", "X") if self.layout_type == "ambisonics" else ()
        )
        self.sample_rate_hz = receipt.get("sample_rate_hz")
        if (
            not self.expected_labels or not isinstance(self.layout_id, str)
            or self.channel_count != len(self.expected_labels)
            or isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int) or self.sample_rate_hz < 1
        ):
            raise RIRCacheError("RIR cache audio contract is invalid")
        runtime_policy = request.get("runtime_policy")
        if not isinstance(runtime_policy, Mapping):
            raise RIRCacheError("RIR cache runtime policy is invalid")
        self.translation_m = np.asarray(
            _finite_vector(
                runtime_policy.get("coordinate_translation_m"),
                3,
                owner="RIR cache coordinate translation",
            ),
            dtype=np.float64,
        )
        raw_entries = index.get("entries")
        if not isinstance(raw_entries, list) or len(raw_entries) != len(jobs):
            raise RIRCacheError("RIR cache index does not close over the full plan")
        jobs_by_id = {str(job["job_id"]): job for job in jobs}
        self.entries_by_id: dict[str, Mapping[str, Any]] = {}
        for entry in raw_entries:
            if not isinstance(entry, Mapping):
                raise RIRCacheError("RIR cache index contains a malformed entry")
            job_id = entry.get("job_id")
            if not isinstance(job_id, str) or job_id in self.entries_by_id:
                raise RIRCacheError("RIR cache index job IDs are invalid")
            if self.listener_pose_bound:
                job = jobs_by_id.get(job_id)
                if job is None:
                    raise RIRCacheError("RIR cache index job is absent from plan")
                expected_source = (
                    np.asarray(job["source_position_m"], dtype=np.float64)
                    + self.translation_m
                )
                expected_listener = (
                    np.asarray(job["listener_position_m"], dtype=np.float64)
                    + self.translation_m
                )
                if (
                    entry.get("acoustic_state_sha256")
                    != job["acoustic_state_sha256"]
                    or not np.array_equal(
                        np.asarray(
                            _finite_vector(
                                entry.get("source_position_m"),
                                3,
                                owner="RIR cache index source position",
                            ),
                            dtype=np.float64,
                        ),
                        expected_source,
                    )
                    or not np.array_equal(
                        np.asarray(
                            _finite_vector(
                                entry.get("listener_position_m"),
                                3,
                                owner="RIR cache index Listener position",
                            ),
                            dtype=np.float64,
                        ),
                        expected_listener,
                    )
                    or not np.array_equal(
                        np.asarray(
                            _unit_orientation(
                                entry.get("listener_orientation_wxyz"),
                                owner="RIR cache index Listener orientation",
                            ),
                            dtype=np.float64,
                        ),
                        np.asarray(
                            job["listener_orientation_wxyz"], dtype=np.float64
                        ),
                    )
                ):
                    raise RIRCacheError(
                        "RIR cache index Listener pose differs from plan"
                    )
            self.entries_by_id[job_id] = entry
        if set(self.entries_by_id) != {job["job_id"] for job in jobs}:
            raise RIRCacheError("RIR cache index job IDs differ from the plan")
        self.jobs_by_episode: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
        for job in jobs:
            for use in job["uses"]:
                episode_id = str(use["episode_id"])
                key = (str(use["source_slot_id"]), int(use["frame_index"]))
                by_use = self.jobs_by_episode.setdefault(episode_id, {})
                if key in by_use:
                    raise RIRCacheError("episode use resolves to multiple RIR jobs")
                by_use[key] = job
        self.request_identity_sha256 = request_identity
        self.acoustic_selection_binding = deepcopy(
            acoustic_selection_binding
        )
        self.acoustic_scene_identity = self.external_input_identity["acoustic_scene"]
        self.retained_shards = (
            shared_shard_cache if shared_shard_cache is not None else {}
        )

    def load_episode(self, episode_id: str) -> CachedRIREpisode:
        if not isinstance(episode_id, str) or not episode_id:
            raise RIRCacheError("cached episode_id must be nonempty")
        jobs_by_use = self.jobs_by_episode.get(episode_id)
        if not jobs_by_use:
            raise RIRCacheError(f"episode {episode_id!r} has no planned RIR jobs")
        source_slots = tuple(SOURCE_SLOTS)
        frame_sets = {
            source_slot: {
                frame_index for slot, frame_index in jobs_by_use if slot == source_slot
            }
            for source_slot in source_slots
        }
        if (
            any(not values for values in frame_sets.values())
            or len({tuple(sorted(values)) for values in frame_sets.values()}) != 1
        ):
            raise RIRCacheError("episode source slots do not share one RIR keyframe grid")
        visual_frames = tuple(sorted(frame_sets[source_slots[0]]))
        if visual_frames[0] != 0 or any(
            value < 0 or value >= self.frames for value in visual_frames
        ):
            raise RIRCacheError("episode RIR visual-frame grid is outside the clip")
        keyframe_samples = tuple(
            int(round(value * self.sample_rate_hz / self.fps))
            for value in visual_frames
        )
        if keyframe_samples[0] != 0 or any(
            right <= left
            for left, right in zip(keyframe_samples, keyframe_samples[1:])
        ):
            raise RIRCacheError("episode RIR sample grid is invalid")

        rows: list[list[np.ndarray]] = []
        lengths = np.empty((len(visual_frames), len(source_slots)), dtype="<u4")
        evidence_jobs: list[dict[str, Any]] = []
        for frame_ordinal, visual_frame in enumerate(visual_frames):
            frame_values: list[np.ndarray] = []
            for source_ordinal, source_slot in enumerate(source_slots):
                job = jobs_by_use[(source_slot, visual_frame)]
                entry = self.entries_by_id[job["job_id"]]
                shard_relative = entry.get("shard")
                row = entry.get("row")
                if (
                    not isinstance(shard_relative, str)
                    or Path(shard_relative).is_absolute()
                    or isinstance(row, bool)
                    or not isinstance(row, int)
                    or row < 0
                ):
                    raise RIRCacheError("RIR cache episode index reference is invalid")
                shard_path = (self.root / shard_relative).resolve()
                try:
                    shard_path.relative_to(self.root)
                except ValueError as exc:
                    raise RIRCacheError("RIR cache shard escapes the cache root") from exc
                retained = self.retained_shards.get(shard_path)
                if retained is None:
                    retained = _read_shard(shard_path)
                    self.retained_shards[shard_path] = retained
                if row >= len(retained["job_ids"]):
                    raise RIRCacheError("RIR cache episode row is outside its shard")
                length = int(retained["lengths"][row])
                samples = np.ascontiguousarray(retained["samples"][row, :, :length])
                expected_source_position = (
                    np.asarray(job["source_position_m"], dtype=np.float64)
                    + self.translation_m
                )
                expected_listener_position = (
                    np.asarray(job["listener_position_m"], dtype=np.float64)
                    + self.translation_m
                )
                listener_pose_mismatch = False
                if self.listener_pose_bound:
                    listener_pose_mismatch = (
                        "listener_positions_m" not in retained
                        or not np.array_equal(
                            retained["listener_positions_m"][row],
                            expected_listener_position,
                        )
                        or not np.array_equal(
                            retained["listener_orientations_wxyz"][row],
                            np.asarray(
                                job["listener_orientation_wxyz"], dtype="<f8"
                            ),
                        )
                        or str(retained["acoustic_state_sha256"][row])
                        != job["acoustic_state_sha256"]
                        or entry.get("listener_position_m")
                        != expected_listener_position.tolist()
                        or entry.get("listener_orientation_wxyz")
                        != job["listener_orientation_wxyz"]
                        or entry.get("acoustic_state_sha256")
                        != job["acoustic_state_sha256"]
                    )
                if (
                    str(retained["job_ids"][row]) != job["job_id"]
                    or int(retained["job_indices"][row]) != int(entry.get("job_index"))
                    or length != int(entry.get("sample_count"))
                    or str(retained["ir_sha256"][row]) != entry.get("ir_sha256")
                    or not np.array_equal(
                        retained["source_positions_m"][row],
                        expected_source_position,
                    )
                    or listener_pose_mismatch
                    or samples.shape[0] != self.channel_count
                    or str(retained["layout_id"]) != self.layout_id
                    or int(retained["sample_rate_hz"]) != self.sample_rate_hz
                    or not np.array_equal(
                        retained["channel_labels"], np.asarray(self.expected_labels)
                    )
                ):
                    raise RIRCacheError("RIR cache episode entry differs from plan/shard")
                frame_values.append(samples)
                lengths[frame_ordinal, source_ordinal] = length
                evidence_jobs.append(
                    {
                        "job_id": job["job_id"],
                        "source_slot_id": source_slot,
                        "visual_frame_index": visual_frame,
                        "ir_sha256": entry["ir_sha256"],
                        "source_position_m": list(job["source_position_m"]),
                        "listener_position_m": list(job["listener_position_m"]),
                        "listener_orientation_wxyz": list(
                            job["listener_orientation_wxyz"]
                        ),
                        "acoustic_state_sha256": job[
                            "acoustic_state_sha256"
                        ],
                        "realized_source_position_m": (
                            retained["source_positions_m"][row].tolist()
                        ),
                        "realized_listener_position_m": (
                            retained["listener_positions_m"][row].tolist()
                            if "listener_positions_m" in retained
                            else expected_listener_position.tolist()
                        ),
                    }
                )
            rows.append(frame_values)

        maximum_length = max(value.shape[1] for row in rows for value in row)
        samples = np.zeros(
            (len(visual_frames), len(source_slots), self.channel_count, maximum_length),
            dtype="<f4",
        )
        for frame_ordinal, frame_values in enumerate(rows):
            for source_ordinal, value in enumerate(frame_values):
                samples[frame_ordinal, source_ordinal, :, : value.shape[1]] = value
        evidence = {
            "schema": "avengine_cached_rir_episode_v1",
            "status": "pass",
            "episode_id": episode_id,
            "cache_request_identity_sha256": self.request_identity_sha256,
            "acoustic_selection_binding": deepcopy(
                self.acoustic_selection_binding
            ),
            "plan_sha256": self.plan_sha256,
            "acoustic_state_binding": (
                "source_listener_pose_per_job_v1"
                if self.listener_pose_bound
                else "legacy_fixed_listener_via_plan_sha256"
            ),
            "source_slot_ids": list(source_slots),
            "visual_frame_indices": list(visual_frames),
            "keyframe_samples": list(keyframe_samples),
            "layout_type": self.layout_type,
            "layout_id": self.layout_id,
            "channel_labels": list(self.expected_labels),
            "sample_rate_hz": self.sample_rate_hz,
            "jobs": evidence_jobs,
        }
        return CachedRIREpisode(
            samples=np.ascontiguousarray(samples),
            lengths=np.ascontiguousarray(lengths),
            source_slot_ids=source_slots,
            visual_frame_indices=visual_frames,
            keyframe_samples=keyframe_samples,
            sample_rate_hz=self.sample_rate_hz,
            layout_type=str(self.layout_type),
            layout_id=self.layout_id,
            channel_labels=self.expected_labels,
            evidence=evidence,
        )


def load_cached_rir_episode(
    *, cache_root: str | Path, plan_path: str | Path, episode_id: str,
    frame_count: int, frame_rate_hz: int,
    shared_shard_cache: dict[Path, dict[str, Any]] | None = None,
) -> CachedRIREpisode:
    """One-shot compatibility wrapper around :class:`RIRCacheSession`."""

    return RIRCacheSession(
        cache_root=cache_root, plan_path=plan_path, frame_count=frame_count,
        frame_rate_hz=frame_rate_hz, shared_shard_cache=shared_shard_cache,
    ).load_episode(episode_id)


__all__ = [
    "CachedRIREpisode",
    "RIRCacheSession",
    "RIRBatchResult",
    "RIRCacheError",
    "RIRCacheResult",
    "RIR_CACHE_ACOUSTIC_SELECTION_BINDING_SCHEMA",
    "RIR_CACHE_ACOUSTIC_SELECTION_NAME",
    "RIR_CACHE_ACOUSTIC_SELECTION_SIDECAR_SCHEMA",
    "RIR_CACHE_INDEX_SCHEMA",
    "RIR_CACHE_RECEIPT_SCHEMA",
    "RIR_CACHE_REQUEST_SCHEMA",
    "RIR_CACHE_TIMING_SCHEMA",
    "CachedRIREpisode",
    "RIRBatchResult",
    "RIRCacheError",
    "RIRCacheResult",
    "RIRCacheSession",
    "load_cached_rir_episode",
    "render_rir_cache",
    "validate_rir_job_plan",
]
