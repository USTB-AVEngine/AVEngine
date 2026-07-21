"""Persistent native-RLR rendering into a resumable room-impulse cache."""

from __future__ import annotations

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
from avengine.m6x.room_feasibility import RIR_JOB_PLAN_SCHEMA


RIR_CACHE_REQUEST_SCHEMA = "avengine_rlr_rir_cache_request_v1"
RIR_CACHE_INDEX_SCHEMA = "avengine_rlr_rir_cache_index_v1"
RIR_CACHE_RECEIPT_SCHEMA = "avengine_rlr_rir_cache_receipt_v1"
RIR_CACHE_TIMING_SCHEMA = "avengine_rlr_rir_cache_timing_v1"


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


def validate_rir_job_plan(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Validate the source-agnostic plan consumed by native RLR."""

    if not isinstance(value, Mapping) or value.get("schema") != RIR_JOB_PLAN_SCHEMA:
        raise RIRCacheError(f"RIR plan schema must be {RIR_JOB_PLAN_SCHEMA}")
    if value.get("status") != "planned_not_run":
        raise RIRCacheError("RIR plan must have status planned_not_run")
    _finite_vector(value.get("listener_position_m"), 3, owner="listener position")
    orientation = _finite_vector(
        value.get("listener_orientation_wxyz"),
        4,
        owner="listener orientation",
    )
    if not math.isclose(
        math.sqrt(sum(component * component for component in orientation)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RIRCacheError("listener orientation must be unit normalized")
    raw_jobs = value.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise RIRCacheError("RIR plan jobs must be a nonempty list")
    jobs: list[dict[str, Any]] = []
    ids: set[str] = set()
    positions: set[tuple[float, float, float]] = set()
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
        if position in positions:
            raise RIRCacheError("RIR plan contains duplicate acoustic positions")
        uses = raw.get("uses")
        if not isinstance(uses, list) or not uses:
            raise RIRCacheError(f"RIR job {job_id} has no uses")
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
        jobs.append(
            {
                "job_id": job_id,
                "source_position_m": list(position),
                "uses": [dict(use) for use in uses],
            }
        )
        ids.add(job_id)
        positions.add(position)
    if value.get("unique_rir_job_count") != len(jobs):
        raise RIRCacheError("RIR plan unique job count differs from jobs")
    return tuple(jobs)


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class _NativeRIRBatchRenderer:
    """One scene upload and fixed native source slots reused across all batches."""

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
            },
        }

    def render(self, positions_m: Sequence[Sequence[float]]) -> RIRBatchResult:
        if not positions_m or len(positions_m) > self.batch_size:
            raise RIRCacheError("batch position count is outside native slot capacity")
        requested = [
            _finite_vector(value, 3, owner="batch source position")
            for value in positions_m
        ]
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
        if set(value.files) != required:
            raise RIRCacheError(f"RIR shard fields differ from contract: {path}")
        result = {key: np.asarray(value[key]).copy() for key in required}
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


def _request_record(
    *,
    plan_path: Path,
    scene: CompiledAcousticScene,
    simulation_request_path: Path,
    simulation: M4SimulationConfig,
    hrtf_file_path: Path | None,
    layout_type: str,
    channel_count: int,
    batch_size: int,
    job_offset: int,
    job_count: int,
    full_plan_job_count: int,
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
    return {
        "schema": RIR_CACHE_REQUEST_SCHEMA,
        "plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
            "full_job_count": full_plan_job_count,
            "selected_job_offset": job_offset,
            "selected_job_count": job_count,
        },
        "acoustic_scene": {
            "manifest_path": str(scene.manifest_path),
            "manifest_sha256": scene.manifest_sha256,
            "package_id": scene.package_id,
            "package_content_sha256": scene.package_content_sha256,
        },
        "simulation": {
            "request_path": str(simulation_request_path),
            "request_sha256": sha256_file(simulation_request_path),
            "effective": effective_simulation.to_dict(),
        },
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
            "scene_upload_count": 1,
            "compute_device": "CPU",
            "gpu_acceleration": False,
        },
        "request_identity_sha256": canonical_json_sha256(
            {
                "plan_sha256": sha256_file(plan_path),
                "scene_sha256": scene.package_content_sha256,
                "simulation": effective_simulation.to_dict(),
                "hrtf_sha256": (
                    sha256_file(hrtf_file_path) if hrtf_file_path else None
                ),
                "layout_type": layout_type,
                "batch_size": batch_size,
                "job_offset": job_offset,
                "job_count": job_count,
                "translation_m": list(translation_m),
                "source_radius_m": source_radius_m,
                "listener_radius_m": listener_radius_m,
                "compressed": compressed,
            }
        ),
    }


def render_rir_cache(
    *,
    plan_path: Path,
    scene: CompiledAcousticScene,
    simulation_request_path: Path,
    simulation: M4SimulationConfig,
    output: Path,
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
        hrtf_file_path=hrtf,
        layout_type=layout_type,
        channel_count=channel_count,
        batch_size=native_batch_size,
        job_offset=job_offset,
        job_count=len(selected_jobs),
        full_plan_job_count=len(jobs),
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
    shards_root = output / "shards"
    shards_root.mkdir(exist_ok=True)

    translation_array = np.asarray(translation, dtype=np.float64)
    translated_listener = (
        np.asarray(plan["listener_position_m"], dtype=np.float64) + translation_array
    ).tolist()
    listener_orientation = plan["listener_orientation_wxyz"]
    expected_sample_rate_hz = int(round(simulation.sample_rate_hz))
    expected_layout_id = request["output"]["layout_id"]
    expected_channel_labels = (
        ("left", "right") if layout_type == "binaural" else ("W", "Y", "Z", "X")
    )
    batch_ranges = [
        (start, min(start + native_batch_size, len(selected_jobs)))
        for start in range(0, len(selected_jobs), native_batch_size)
    ]
    renderer: Any | None = None
    setup_report: dict[str, Any] | None = None
    batch_records: list[dict[str, Any]] = []
    new_job_count = 0
    try:
        for batch_index, (start, end) in enumerate(batch_ranges):
            shard_path = shards_root / f"shard_{batch_index:06d}.npz"
            absolute_indices = np.arange(
                job_offset + start, job_offset + end, dtype="<u4"
            )
            batch_jobs = selected_jobs[start:end]
            positions = [
                (
                    np.asarray(job["source_position_m"], dtype=np.float64)
                    + translation_array
                ).tolist()
                for job in batch_jobs
            ]
            if shard_path.is_file():
                retained = _read_shard(shard_path)
                _verify_shard_request(
                    retained,
                    path=shard_path,
                    expected_job_indices=absolute_indices,
                    expected_job_ids=[job["job_id"] for job in batch_jobs],
                    expected_source_positions_m=positions,
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
                    listener_position_m=translated_listener,
                    listener_orientation_wxyz=listener_orientation,
                    layout_type=layout_type,
                    channel_count=channel_count,
                    hrtf_file_path=str(hrtf) if hrtf else "",
                    source_radius_m=float(source_radius_m),
                    listener_radius_m=float(listener_radius_m),
                )
                setup_report = dict(renderer.setup_report)
            result: RIRBatchResult = renderer.render(positions)
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
                    "batch_count": len(batch_ranges),
                    "completed_job_count": end,
                    "selected_job_count": len(selected_jobs),
                },
            )
            print(
                "RIR_CACHE_BATCH_OK "
                f"batch={batch_index + 1}/{len(batch_ranges)} "
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
                    for index in range(len(batch_ranges))
                ),
                "batch_count": len(batch_ranges),
            },
        )
        raise

    index_entries: list[dict[str, Any]] = []
    all_batch_records: list[dict[str, Any]] = []
    for batch_index, (start, end) in enumerate(batch_ranges):
        shard_path = shards_root / f"shard_{batch_index:06d}.npz"
        retained = _read_shard(shard_path)
        batch_jobs = selected_jobs[start:end]
        expected_indices = np.arange(job_offset + start, job_offset + end, dtype="<u4")
        expected_positions = [
            (
                np.asarray(job["source_position_m"], dtype=np.float64)
                + translation_array
            ).tolist()
            for job in batch_jobs
        ]
        _verify_shard_request(
            retained,
            path=shard_path,
            expected_job_indices=expected_indices,
            expected_job_ids=[job["job_id"] for job in batch_jobs],
            expected_source_positions_m=expected_positions,
            sample_rate_hz=expected_sample_rate_hz,
            layout_id=expected_layout_id,
            channel_labels=expected_channel_labels,
        )
        for row, job in enumerate(batch_jobs):
            index_entries.append(
                {
                    "job_id": job["job_id"],
                    "job_index": job_offset + start + row,
                    "shard": shard_path.relative_to(output).as_posix(),
                    "row": row,
                    "sample_count": int(retained["lengths"][row]),
                    "ir_sha256": str(retained["ir_sha256"][row]),
                    "source_position_m": retained["source_positions_m"][row].tolist(),
                }
            )
        matching = next(
            record for record in batch_records if record["batch_index"] == batch_index
        )
        all_batch_records.append(matching)
    write_json(
        output / "index.json",
        {
            "schema": RIR_CACHE_INDEX_SCHEMA,
            "status": "pass",
            "request_identity_sha256": request["request_identity_sha256"],
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
        "full_plan_complete": len(selected_jobs) == len(jobs) and job_offset == 0,
        "full_plan_job_count": len(jobs),
        "selected_job_count": len(selected_jobs),
        "retained_shard_count": len(batch_ranges),
        "retained_shard_bytes": total_bytes,
        "sample_rate_hz": int(round(simulation.sample_rate_hz)),
        "layout_type": layout_type,
        "layout_id": request["output"]["layout_id"],
        "channel_count": channel_count,
        "producer_backend": "RLR Audio Propagation",
        "cache_artifact": "room impulse response (RIR)",
        "dry_audio_independent": True,
        "retained_payload_hash_verified": True,
        "rerun_byte_identity_claim": False,
        "native_random_seed_control": "unavailable_in_current_RLR_API",
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
    failed_path = output / "FAILED.json"
    if failed_path.exists():
        failed_path.unlink()
    write_json(output / "progress.json", receipt)
    return RIRCacheResult(output=output, receipt=receipt)


__all__ = [
    "RIRBatchResult",
    "RIRCacheError",
    "RIRCacheResult",
    "RIR_CACHE_INDEX_SCHEMA",
    "RIR_CACHE_RECEIPT_SCHEMA",
    "RIR_CACHE_REQUEST_SCHEMA",
    "RIR_CACHE_TIMING_SCHEMA",
    "render_rir_cache",
    "validate_rir_job_plan",
]
