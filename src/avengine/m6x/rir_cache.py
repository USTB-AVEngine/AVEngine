"""Persistent native-RLR rendering into a resumable room-impulse cache."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import resource
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

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
    "render_rir_cache",
    "load_cached_rir_episode",
    "validate_rir_job_plan",
]
