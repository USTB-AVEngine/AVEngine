"""Owned reader for completed semantic RIR caches.

This module validates schema, selected paths, job/use closure, native execution
metadata, and decoded sample arrays.  It intentionally ignores legacy digest
and byte-size evidence retained by the producer; no native RLR work runs here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import load_json
from avengine.m4.runtime import M4SimulationConfig
from avengine.m5.audio import M5_AUDIO_SAMPLE_RATE_HZ
from avengine.m6x.rir_cache import RIRCacheError, validate_semantic_rir_job_plan
from avengine.m6x.room_feasibility import SOURCE_SLOTS

_SEMANTIC_NATIVE_CLAIM = (
    "native CPU RIR samples with structural pose/use, native source/listener "
    "receipts, and decoded-sample validation"
)


@dataclass(frozen=True)
class SemanticRIREpisode:
    """One digest-free RIR episode closed by schema and sample metadata."""

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


def _semantic_finite_vector(value: Any, length: int, *, owner: str) -> list[float]:
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
    return [float(item) for item in value]


def _semantic_unit_orientation(value: Any, *, owner: str) -> list[float]:
    result = _semantic_finite_vector(value, 4, owner=owner)
    if not math.isclose(
        math.sqrt(sum(component * component for component in result)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RIRCacheError(f"{owner} must be unit normalized")
    return result


def _resolve_semantic_regular_path(
    path: Path, *, owner: str, directory: bool = False
) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise RIRCacheError(f"{owner} may not contain parent traversal")
    absolute = Path(os.path.abspath(raw))
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise RIRCacheError(f"{owner} may not contain symlink components")
    if (directory and not absolute.is_dir()) or (
        not directory and not absolute.is_file()
    ):
        raise RIRCacheError(f"{owner} is missing or has the wrong type")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise RIRCacheError(f"{owner} physical path differs from its selection")
    return resolved


def _semantic_acoustic_selection_binding(value: Any) -> dict[str, Any]:
    """Project selection identity without consuming legacy digest evidence."""

    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema",
            "selection_mode",
            "registry_selection_applied",
            "room_ref",
            "profile_ref",
            "binding_id",
        }
        or value.get("schema") != "avengine_rir_cache_acoustic_selection_binding_v1"
    ):
        raise RIRCacheError("semantic RIR acoustic selection is invalid")
    mode = value.get("selection_mode")
    applied = value.get("registry_selection_applied")
    room_ref = value.get("room_ref")
    profile_ref = value.get("profile_ref")
    binding_id = value.get("binding_id")
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
            raise RIRCacheError(
                "semantic registry RIR acoustic selection is incomplete"
            )
    elif mode in {"explicit_legacy", "explicit_legacy_unbound"}:
        if (
            applied is not False
            or room_ref is not None
            or profile_ref is not None
            or binding_id is not None
        ):
            raise RIRCacheError(
                "semantic explicit RIR acoustic selection fabricated an identity"
            )
    else:
        raise RIRCacheError("semantic RIR acoustic selection mode is invalid")
    return {
        "schema": value["schema"],
        "selection_mode": mode,
        "registry_selection_applied": applied,
        "room_ref": deepcopy(room_ref),
        "profile_ref": deepcopy(profile_ref),
        "binding_id": binding_id,
    }


def _read_semantic_rir_shard(path: Path) -> dict[str, np.ndarray]:
    """Read structural/sample fields while ignoring legacy digest/size arrays."""

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
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise RIRCacheError(
                    f"semantic RIR shard fields differ from contract: {path}"
                )
            loaded = {name: np.asarray(archive[name]) for name in required}
            if any(not array.flags.c_contiguous for array in loaded.values()):
                raise RIRCacheError(
                    f"semantic RIR shard arrays must be C-contiguous: {path}"
                )
            result = {name: array.copy(order="C") for name, array in loaded.items()}
    except RIRCacheError:
        raise
    except Exception as exc:
        raise RIRCacheError(f"semantic RIR shard is unreadable: {path}") from exc
    samples = result["samples"]
    lengths = result["lengths"]
    count = samples.shape[0] if samples.ndim == 3 else 0
    if (
        count < 1
        or samples.shape[1] != 2
        or result["job_indices"].shape != (count,)
        or result["job_ids"].shape != (count,)
        or result["source_positions_m"].shape != (count, 3)
        or result["listener_positions_m"].shape != (count, 3)
        or result["listener_orientations_wxyz"].shape != (count, 4)
        or lengths.shape != (count,)
        or samples.dtype != np.dtype("<f4")
        or result["job_indices"].dtype != np.dtype("<u4")
        or lengths.dtype != np.dtype("<u4")
        or result["source_positions_m"].dtype != np.dtype("<f8")
        or result["listener_positions_m"].dtype != np.dtype("<f8")
        or result["listener_orientations_wxyz"].dtype != np.dtype("<f8")
        or result["job_ids"].dtype.kind != "U"
        or any(not value for value in result["job_ids"].tolist())
        or len(set(result["job_ids"].tolist())) != count
        or len(set(result["job_indices"].tolist())) != count
        or any(not array.flags.c_contiguous for array in result.values())
        or np.any(lengths < 2)
        or np.any(lengths > samples.shape[2])
        or not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(result["source_positions_m"]))
        or not np.all(np.isfinite(result["listener_positions_m"]))
        or not np.all(np.isfinite(result["listener_orientations_wxyz"]))
        or not np.allclose(
            np.linalg.norm(result["listener_orientations_wxyz"], axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-6,
        )
        or result["sample_rate_hz"].shape != ()
        or result["sample_rate_hz"].dtype != np.dtype("<u4")
        or int(result["sample_rate_hz"].item()) != M5_AUDIO_SAMPLE_RATE_HZ
        or result["layout_id"].shape != ()
        or result["layout_id"].dtype.kind != "U"
        or str(result["layout_id"].item()) != "rlr_binaural_lr_v1"
        or result["channel_labels"].shape != (2,)
        or result["channel_labels"].dtype.kind != "U"
        or tuple(str(item) for item in result["channel_labels"]) != ("left", "right")
        or any(
            result[name].shape != () or result[name].dtype != np.dtype("<f8")
            for name in (
                "simulate_wall_seconds",
                "simulate_process_cpu_seconds",
                "indirect_ray_efficiency",
            )
        )
    ):
        raise RIRCacheError(f"semantic RIR shard metadata is invalid: {path}")
    wall = float(result["simulate_wall_seconds"])
    cpu = float(result["simulate_process_cpu_seconds"])
    efficiency = float(result["indirect_ray_efficiency"])
    if (
        not all(math.isfinite(value) for value in (wall, cpu, efficiency))
        or wall < 0
        or cpu < 0
        or not 0 <= efficiency <= 1
    ):
        raise RIRCacheError(f"semantic RIR shard timing is invalid: {path}")
    for row, raw_length in enumerate(lengths):
        length = int(raw_length)
        if not np.any(samples[row, :, :length]) or np.any(samples[row, :, length:]):
            raise RIRCacheError(
                f"semantic RIR shard active/padding samples are invalid: {path} row {row}"
            )
    return result


def _semantic_exact_mapping(
    value: Any, fields: set[str], *, owner: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RIRCacheError(f"{owner} fields differ from semantic contract")
    return value


def _semantic_nonnegative_number(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise RIRCacheError(f"{owner} must be finite and nonnegative")
    return float(value)


def _semantic_canonical_absolute_path(value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RIRCacheError(f"{owner} must be a nonempty absolute path")
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or os.path.abspath(value) != value
    ):
        raise RIRCacheError(f"{owner} must be a canonical absolute path")
    return path


def _semantic_positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RIRCacheError(f"{owner} must be a positive integer")
    return value


def _semantic_upload_structure(value: Mapping[str, Any]) -> None:
    object_count = _semantic_positive_int(
        value.get("object_count"), owner="semantic upload object count"
    )
    vertex_count = _semantic_positive_int(
        value.get("vertex_count"), owner="semantic upload vertex count"
    )
    triangle_count = _semantic_positive_int(
        value.get("triangle_count"), owner="semantic upload triangle count"
    )
    category_count = _semantic_positive_int(
        value.get("material_category_count"),
        owner="semantic upload material category count",
    )
    object_ids = value.get("object_ids")
    maps = [
        value.get("triangle_count_by_material"),
        value.get("material_upload_call_count"),
        value.get("resolved_material_name_by_category"),
        value.get("resolved_material_index_by_category"),
    ]
    if (
        vertex_count < 3
        or triangle_count < 1
        or not isinstance(object_ids, list)
        or len(object_ids) != object_count
        or any(not isinstance(item, str) or not item for item in object_ids)
        or len(set(object_ids)) != object_count
        or any(not isinstance(item, Mapping) for item in maps)
    ):
        raise RIRCacheError("semantic native upload structure is invalid")
    material_keys = set(maps[0])
    if (
        len(material_keys) != category_count
        or any(not isinstance(key, str) or not key for key in material_keys)
        or any(set(item) != material_keys for item in maps[1:])
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in maps[0].values()
        )
        or sum(maps[0].values()) != triangle_count
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 1 <= item <= object_count
            for item in maps[1].values()
        )
        or any(not isinstance(item, str) or not item for item in maps[2].values())
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in maps[3].values()
        )
    ):
        raise RIRCacheError("semantic native material upload is invalid")


class SemanticRIRCacheSession:
    """Digest-free reader for planning RIR caches.

    This path validates schema, selected paths, job/use bijections, poses, and
    decoded sample metadata.  It intentionally neither reads nor verifies any
    file digest or byte-size field retained by the legacy cache producer.
    """

    def __init__(
        self,
        *,
        cache_root: Path,
        plan_path: Path,
        expected_episode_id: str,
        frame_count: int,
        frame_rate_hz: int,
        shared_shard_cache: dict[Path, dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.root = _resolve_semantic_regular_path(
            cache_root, owner="semantic RIR cache root", directory=True
        )
        self.plan_path = _resolve_semantic_regular_path(
            plan_path, owner="semantic selected RIR plan"
        )
        if frame_count != 75 or frame_rate_hz != 15:
            raise RIRCacheError("semantic RIR cache requires full75/15Hz")
        self.frames = frame_count
        self.fps = frame_rate_hz

        def fixed_json(path: Path, root: Path, *, owner: str) -> Mapping[str, Any]:
            resolved = _resolve_semantic_regular_path(path, owner=owner)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise RIRCacheError(f"{owner} escapes its selected root") from exc
            value = load_json(resolved)
            if not isinstance(value, Mapping):
                raise RIRCacheError(f"{owner} must contain an object")
            return value

        plan = fixed_json(
            self.plan_path, self.plan_path.parent, owner="semantic RIR plan"
        )
        request = fixed_json(
            self.root / "request.json", self.root, owner="semantic RIR request"
        )
        receipt = fixed_json(
            self.root / "receipt.json", self.root, owner="semantic RIR receipt"
        )
        index = fixed_json(
            self.root / "index.json", self.root, owner="semantic RIR index"
        )
        timing = fixed_json(
            self.root / "timing.json", self.root, owner="semantic RIR timing"
        )
        try:
            raw_jobs = list(validate_semantic_rir_job_plan(plan))
        except RIRCacheError as exc:
            raise RIRCacheError("semantic RIR job plan is invalid") from exc
        self.jobs_by_id: dict[str, dict[str, Any]] = {}
        self.jobs_by_episode: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
        use_count = 0
        for ordinal, normalized_job in enumerate(raw_jobs):
            job_id = str(normalized_job["job_id"])
            job = {
                "job_id": job_id,
                "job_index": ordinal,
                "source_position_m": tuple(normalized_job["source_position_m"]),
                "listener_position_m": tuple(normalized_job["listener_position_m"]),
                "listener_orientation_wxyz": list(
                    normalized_job["listener_orientation_wxyz"]
                ),
            }
            self.jobs_by_id[job_id] = job
            for raw_use in normalized_job["uses"]:
                episode_id = str(raw_use["episode_id"])
                key = (str(raw_use["source_slot_id"]), int(raw_use["frame_index"]))
                self.jobs_by_episode.setdefault(episode_id, {})[key] = job
                use_count += 1
        if use_count != self.frames * len(SOURCE_SLOTS) or set(
            self.jobs_by_episode
        ) != {expected_episode_id}:
            raise RIRCacheError("semantic RIR plan episode selection is invalid")

        request = _semantic_exact_mapping(
            request,
            {
                "schema",
                "status",
                "qualification_claim",
                "plan",
                "acoustic_scene",
                "simulation",
                "acoustic_selection_binding",
                "output",
                "runtime_policy",
            },
            owner="semantic RIR cache request",
        )
        request_plan = _semantic_exact_mapping(
            request["plan"],
            {
                "path",
                "full_job_count",
                "selected_job_offset",
                "selected_job_count",
                "acoustic_state_binding",
            },
            owner="semantic RIR cache request plan",
        )
        request_scene = _semantic_exact_mapping(
            request["acoustic_scene"],
            {"manifest_path", "package_id"},
            owner="semantic RIR cache acoustic scene",
        )
        request_simulation = _semantic_exact_mapping(
            request["simulation"],
            {"request_path", "effective"},
            owner="semantic RIR cache simulation",
        )
        request_output = _semantic_exact_mapping(
            request["output"],
            {
                "layout_type",
                "channel_count",
                "layout_id",
                "hrtf_path",
                "compressed_npz_shards",
            },
            owner="semantic RIR cache output",
        )
        runtime = _semantic_exact_mapping(
            request["runtime_policy"],
            {
                "native_batch_size",
                "coordinate_translation_m",
                "source_radius_m",
                "listener_radius_m",
                "persistent_context",
                "listener_pose_update_policy",
                "scene_upload_count",
                "compute_device",
                "gpu_acceleration",
                "execution_mode",
            },
            owner="semantic RIR cache runtime policy",
        )
        try:
            effective_simulation = M4SimulationConfig.from_mapping(
                request_simulation["effective"]
            )
        except Exception as exc:
            raise RIRCacheError(
                "semantic RIR cache effective simulation is invalid"
            ) from exc
        layout = effective_simulation.channel_layout
        native_batch_size = runtime.get("native_batch_size")
        request_plan_path = _semantic_canonical_absolute_path(
            request_plan.get("path"), owner="semantic RIR request plan path"
        )
        _semantic_canonical_absolute_path(
            request_scene.get("manifest_path"),
            owner="semantic RIR acoustic scene manifest path",
        )
        _semantic_canonical_absolute_path(
            request_simulation.get("request_path"),
            owner="semantic RIR simulation request path",
        )
        _semantic_canonical_absolute_path(
            request_output.get("hrtf_path"), owner="semantic RIR HRTF path"
        )
        if (
            request.get("schema") != "avengine_semantic_rir_cache_request_v1"
            or request.get("status") != "ready_structural_and_sample_validation"
            or request.get("qualification_claim") is not False
            or request_plan_path != self.plan_path
            or isinstance(request_plan.get("full_job_count"), bool)
            or request_plan.get("full_job_count") != len(raw_jobs)
            or isinstance(request_plan.get("selected_job_offset"), bool)
            or request_plan.get("selected_job_offset") != 0
            or isinstance(request_plan.get("selected_job_count"), bool)
            or request_plan.get("selected_job_count") != len(raw_jobs)
            or request_plan.get("acoustic_state_binding")
            != "source_listener_pose_per_job_v1"
            or not isinstance(request_scene.get("package_id"), str)
            or not request_scene["package_id"]
            or request_output.get("layout_type") != "binaural"
            or request_output.get("channel_count") != 2
            or request_output.get("layout_id") != "rlr_binaural_lr_v1"
            or not isinstance(request_output.get("compressed_npz_shards"), bool)
            or layout.layout_type != "binaural"
            or layout.channel_count != 2
            or effective_simulation.sample_rate_hz != M5_AUDIO_SAMPLE_RATE_HZ
            or effective_simulation.temporal_coherence is not False
            or isinstance(native_batch_size, bool)
            or not isinstance(native_batch_size, int)
            or native_batch_size < 1
            or runtime.get("persistent_context") is not True
            or isinstance(runtime.get("scene_upload_count"), bool)
            or runtime.get("scene_upload_count") != 1
            or runtime.get("compute_device") != "CPU"
            or runtime.get("gpu_acceleration") is not False
            or runtime.get("execution_mode") != "native_default"
            or runtime.get("listener_pose_update_policy")
            != "set_listener_pose_on_change_v1"
        ):
            raise RIRCacheError("semantic RIR cache request is invalid")
        _semantic_nonnegative_number(
            runtime.get("source_radius_m"), owner="semantic RIR source radius"
        )
        _semantic_nonnegative_number(
            runtime.get("listener_radius_m"), owner="semantic RIR listener radius"
        )
        translation = _semantic_finite_vector(
            runtime.get("coordinate_translation_m"),
            3,
            owner="semantic RIR cache coordinate translation",
        )
        self.native_batch_size = native_batch_size
        self.effective_simulation = effective_simulation
        grouped_job_indices: dict[tuple[float, ...], list[int]] = {}
        for job_index, job in enumerate(raw_jobs):
            listener_key = (
                *tuple(float(item) for item in job["listener_position_m"]),
                *tuple(float(item) for item in job["listener_orientation_wxyz"]),
            )
            grouped_job_indices.setdefault(listener_key, []).append(job_index)
        self.expected_batches: list[dict[str, Any]] = []
        self.expected_batch_row_by_job_index: dict[int, tuple[int, int]] = {}
        for listener_key, grouped_indices in grouped_job_indices.items():
            for start in range(0, len(grouped_indices), native_batch_size):
                indices = grouped_indices[start : start + native_batch_size]
                batch_index = len(self.expected_batches)
                spec = {
                    "batch_index": batch_index,
                    "job_indices": indices,
                    "shard": f"shards/shard_{batch_index:06d}.npz",
                    "listener_position_m": (
                        np.asarray(listener_key[:3], dtype=np.float64)
                        + np.asarray(translation, dtype=np.float64)
                    ),
                    "listener_orientation_wxyz": np.asarray(
                        listener_key[3:], dtype=np.float64
                    ),
                }
                self.expected_batches.append(spec)
                for row, job_index in enumerate(indices):
                    self.expected_batch_row_by_job_index[job_index] = (batch_index, row)
        self.acoustic_selection_binding = _semantic_acoustic_selection_binding(
            request.get("acoustic_selection_binding")
        )
        selection_mode = self.acoustic_selection_binding["selection_mode"]
        self.translation_m = np.asarray(translation, dtype=np.float64)
        receipt = _semantic_exact_mapping(
            receipt,
            {
                "schema",
                "status",
                "qualification_claim",
                "claim_boundary",
                "native_execution",
                "native_scene_upload_structurally_validated",
                "native_source_listener_receipts_validated",
                "native_realized_job_count",
                "native_simulate_owned_call_count",
                "full_plan_complete",
                "full_plan_job_count",
                "selected_job_count",
                "retained_shard_count",
                "sample_rate_hz",
                "layout_type",
                "layout_id",
                "channel_count",
                "producer_backend",
                "cache_artifact",
                "dry_audio_independent",
                "acoustic_state_binding",
                "listener_pose_update_policy",
                "acoustic_selection_mode",
                "compute_device",
                "configured_thread_count",
                "outputs",
            },
            owner="semantic RIR cache receipt",
        )
        receipt_outputs = _semantic_exact_mapping(
            receipt["outputs"],
            {"request", "index", "timing", "shards"},
            owner="semantic RIR cache receipt outputs",
        )
        simulate_call_count = receipt.get("native_simulate_owned_call_count")
        retained_shard_count = receipt.get("retained_shard_count")
        if (
            receipt.get("schema") != "avengine_semantic_rir_cache_receipt_v1"
            or receipt.get("status") != "pass"
            or receipt.get("qualification_claim") is not False
            or receipt.get("claim_boundary") != _SEMANTIC_NATIVE_CLAIM
            or receipt.get("native_execution") is not True
            or receipt.get("native_scene_upload_structurally_validated") is not True
            or receipt.get("native_source_listener_receipts_validated") is not True
            or isinstance(receipt.get("native_realized_job_count"), bool)
            or receipt.get("native_realized_job_count") != len(raw_jobs)
            or isinstance(simulate_call_count, bool)
            or not isinstance(simulate_call_count, int)
            or simulate_call_count < 1
            or isinstance(retained_shard_count, bool)
            or not isinstance(retained_shard_count, int)
            or retained_shard_count != simulate_call_count
            or receipt.get("full_plan_complete") is not True
            or isinstance(receipt.get("full_plan_job_count"), bool)
            or receipt.get("full_plan_job_count") != len(raw_jobs)
            or isinstance(receipt.get("selected_job_count"), bool)
            or receipt.get("selected_job_count") != len(raw_jobs)
            or isinstance(receipt.get("sample_rate_hz"), bool)
            or receipt.get("sample_rate_hz") != M5_AUDIO_SAMPLE_RATE_HZ
            or receipt.get("layout_type") != "binaural"
            or receipt.get("layout_id") != "rlr_binaural_lr_v1"
            or isinstance(receipt.get("channel_count"), bool)
            or receipt.get("channel_count") != 2
            or receipt.get("producer_backend") != "RLR Audio Propagation"
            or receipt.get("cache_artifact") != "room impulse response (RIR)"
            or receipt.get("dry_audio_independent") is not True
            or receipt.get("acoustic_state_binding")
            != "source_listener_pose_per_job_v1"
            or receipt.get("listener_pose_update_policy")
            != "set_listener_pose_on_change_v1"
            or receipt.get("acoustic_selection_mode") != selection_mode
            or receipt.get("compute_device") != "CPU"
            or isinstance(receipt.get("configured_thread_count"), bool)
            or receipt.get("configured_thread_count")
            != effective_simulation.thread_count
            or receipt_outputs
            != {
                "request": "request.json",
                "index": "index.json",
                "timing": "timing.json",
                "shards": "shards/",
            }
        ):
            raise RIRCacheError("semantic RIR cache receipt is invalid")

        timing = _semantic_exact_mapping(
            timing,
            {
                "schema",
                "status",
                "setup",
                "batches",
                "selected_job_count",
                "simulate_wall_seconds",
                "serialization_wall_seconds",
                "run_wall_seconds",
                "jobs_per_simulate_second",
            },
            owner="semantic RIR cache timing",
        )
        timing_setup = _semantic_exact_mapping(
            timing["setup"],
            {
                "schema",
                "runtime",
                "configuration_readback",
                "upload",
                "wall_seconds",
                "process_cpu_seconds",
                "compute_device",
                "qualification_claim",
            },
            owner="semantic RIR native setup",
        )
        timing_runtime = _semantic_exact_mapping(
            timing_setup["runtime"],
            {
                "schema",
                "binding_api",
                "quaternion_module_path",
                "habitat_module_path",
                "binding_module_path",
                "rlr_library_path",
            },
            owner="semantic RIR native runtime",
        )
        config_fields = {
            "frequency_bands",
            "direct_sh_order",
            "indirect_sh_order",
            "direct_ray_count",
            "indirect_ray_count",
            "indirect_ray_depth",
            "source_ray_count",
            "source_ray_depth",
            "max_diffraction_order",
            "thread_count",
            "sample_rate_hz",
            "max_ir_seconds",
            "unit_scale",
            "global_volume",
            "direct",
            "indirect",
            "diffraction",
            "transmission",
            "mesh_simplification",
            "temporal_coherence",
        }
        config_readback = _semantic_exact_mapping(
            timing_setup["configuration_readback"],
            config_fields,
            owner="semantic RIR configuration readback",
        )
        runtime_paths = {
            name: _semantic_canonical_absolute_path(
                timing_runtime.get(name), owner=f"semantic RIR runtime {name}"
            )
            for name in (
                "quaternion_module_path",
                "habitat_module_path",
                "binding_module_path",
                "rlr_library_path",
            )
        }
        binding_path = runtime_paths["binding_module_path"]
        library_path = runtime_paths["rlr_library_path"]
        habitat_path = runtime_paths["habitat_module_path"]
        binding_layout_valid = (
            habitat_path.name == "__init__.py"
            and habitat_path.parent.name == "habitat_sim"
            and binding_path.parent.name == "_ext"
            and binding_path.parent.parent.name == "habitat_sim"
            and re.fullmatch(
                r"habitat_sim_bindings\.[A-Za-z0-9_-]+\.so", binding_path.name
            )
            is not None
        )
        timing_upload = _semantic_exact_mapping(
            timing_setup["upload"],
            {
                "status",
                "object_count",
                "vertex_count",
                "triangle_count",
                "material_category_count",
                "object_ids",
                "triangle_count_by_material",
                "material_upload_call_count",
                "resolved_material_name_by_category",
                "resolved_material_index_by_category",
            },
            owner="semantic RIR native upload",
        )
        integer_config_fields = {
            "frequency_bands",
            "direct_sh_order",
            "indirect_sh_order",
            "direct_ray_count",
            "indirect_ray_count",
            "indirect_ray_depth",
            "source_ray_count",
            "source_ray_depth",
            "max_diffraction_order",
            "thread_count",
        }
        float_config_fields = {
            "sample_rate_hz",
            "max_ir_seconds",
            "unit_scale",
            "global_volume",
        }
        boolean_config_fields = {
            "direct",
            "indirect",
            "diffraction",
            "transmission",
            "mesh_simplification",
            "temporal_coherence",
        }
        config_valid = all(
            config_readback[name] == getattr(effective_simulation, name)
            and type(config_readback[name]) is type(getattr(effective_simulation, name))
            for name in integer_config_fields | boolean_config_fields
        ) and all(
            not isinstance(config_readback[name], bool)
            and isinstance(config_readback[name], (int, float))
            and math.isfinite(float(config_readback[name]))
            and math.isclose(
                float(config_readback[name]),
                float(getattr(effective_simulation, name)),
                rel_tol=1.0e-6,
                abs_tol=1.0e-6,
            )
            for name in float_config_fields
        )
        _semantic_upload_structure(timing_upload)
        setup_wall_seconds = _semantic_nonnegative_number(
            timing_setup.get("wall_seconds"), owner="semantic native setup wall time"
        )
        _semantic_nonnegative_number(
            timing_setup.get("process_cpu_seconds"),
            owner="semantic native setup CPU time",
        )
        raw_timing_batches = timing.get("batches")
        timing_batches_valid = isinstance(raw_timing_batches, list)
        if not timing_batches_valid:
            raw_timing_batches = []
        timing_batch_job_count = 0
        for batch_index, batch in enumerate(raw_timing_batches):
            expected_batch = (
                self.expected_batches[batch_index]
                if batch_index < len(self.expected_batches)
                else None
            )
            if (
                not isinstance(batch, Mapping)
                or set(batch)
                != {
                    "batch_index",
                    "job_count",
                    "simulate_wall_seconds",
                    "simulate_process_cpu_seconds",
                    "serialization_wall_seconds",
                }
                or isinstance(batch.get("batch_index"), bool)
                or batch.get("batch_index") != batch_index
                or isinstance(batch.get("job_count"), bool)
                or not isinstance(batch.get("job_count"), int)
                or not 1 <= batch["job_count"] <= native_batch_size
                or expected_batch is None
                or batch["job_count"] != len(expected_batch["job_indices"])
                or any(
                    isinstance(batch.get(name), bool)
                    or not isinstance(batch.get(name), (int, float))
                    or not math.isfinite(float(batch[name]))
                    or float(batch[name]) < 0
                    for name in (
                        "simulate_wall_seconds",
                        "simulate_process_cpu_seconds",
                        "serialization_wall_seconds",
                    )
                )
            ):
                timing_batches_valid = False
                continue
            timing_batch_job_count += batch["job_count"]
        simulate_seconds = sum(
            float(batch["simulate_wall_seconds"])
            for batch in raw_timing_batches
            if isinstance(batch, Mapping)
            and isinstance(batch.get("simulate_wall_seconds"), (int, float))
            and not isinstance(batch.get("simulate_wall_seconds"), bool)
        )
        serialization_seconds = sum(
            float(batch["serialization_wall_seconds"])
            for batch in raw_timing_batches
            if isinstance(batch, Mapping)
            and isinstance(batch.get("serialization_wall_seconds"), (int, float))
            and not isinstance(batch.get("serialization_wall_seconds"), bool)
        )
        expected_rate = (
            len(raw_jobs) / simulate_seconds if simulate_seconds > 0 else None
        )
        timing_numbers_valid = all(
            isinstance(timing.get(name), (int, float))
            and not isinstance(timing.get(name), bool)
            and math.isfinite(float(timing[name]))
            and float(timing[name]) >= 0
            for name in (
                "simulate_wall_seconds",
                "serialization_wall_seconds",
                "run_wall_seconds",
            )
        )
        if (
            timing.get("schema") != "avengine_semantic_rir_cache_timing_v1"
            or timing.get("status") != "pass"
            or isinstance(timing.get("selected_job_count"), bool)
            or timing.get("selected_job_count") != len(raw_jobs)
            or not timing_batches_valid
            or len(raw_timing_batches) != simulate_call_count
            or len(raw_timing_batches) != len(self.expected_batches)
            or timing_batch_job_count != len(raw_jobs)
            or not timing_numbers_valid
            or float(timing["run_wall_seconds"]) + 1.0e-9
            < (setup_wall_seconds + simulate_seconds + serialization_seconds)
            or not math.isclose(
                float(timing["simulate_wall_seconds"]),
                simulate_seconds,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(timing["serialization_wall_seconds"]),
                serialization_seconds,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            or (
                expected_rate is None
                and timing.get("jobs_per_simulate_second") is not None
            )
            or (
                expected_rate is not None
                and (
                    isinstance(timing.get("jobs_per_simulate_second"), bool)
                    or not isinstance(
                        timing.get("jobs_per_simulate_second"), (int, float)
                    )
                    or not math.isclose(
                        float(timing["jobs_per_simulate_second"]),
                        expected_rate,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                )
            )
            or timing_setup.get("schema") != "avengine_semantic_native_rir_setup_v1"
            or timing_setup.get("compute_device") != "CPU"
            or timing_setup.get("qualification_claim") is not False
            or timing_runtime.get("schema")
            != "avengine_semantic_habitat_rlr_runtime_v1"
            or timing_runtime.get("binding_api") != "habitat_sim.RLRAcousticContext_v1"
            or binding_path.suffix != ".so"
            or library_path.name != "libRLRAudioPropagation.so"
            or library_path.parent != binding_path.parent
            or not binding_layout_valid
            or not config_valid
            or timing_upload.get("status") != "pass_structural_native_upload"
        ):
            raise RIRCacheError("semantic RIR cache timing is invalid")

        index = _semantic_exact_mapping(
            index,
            {
                "schema",
                "status",
                "qualification_claim",
                "full_plan_complete",
                "selected_job_count",
                "acoustic_state_binding",
                "acoustic_selection_mode",
                "entries",
            },
            owner="semantic RIR cache index",
        )
        raw_entries = index.get("entries")
        if (
            index.get("schema") != "avengine_semantic_rir_cache_index_v1"
            or index.get("status") != "pass"
            or index.get("qualification_claim") is not False
            or index.get("full_plan_complete") is not True
            or isinstance(index.get("selected_job_count"), bool)
            or index.get("selected_job_count") != len(raw_jobs)
            or index.get("acoustic_state_binding") != "source_listener_pose_per_job_v1"
            or index.get("acoustic_selection_mode") != selection_mode
            or not isinstance(raw_entries, list)
            or len(raw_entries) != len(raw_jobs)
        ):
            raise RIRCacheError("semantic RIR cache index is invalid")
        self.entries_by_id: dict[str, dict[str, Any]] = {}
        referenced_rows: set[tuple[Path, int]] = set()
        for ordinal, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, Mapping) or set(raw_entry) != {
                "job_id",
                "job_index",
                "shard",
                "row",
                "sample_count",
                "source_position_m",
                "listener_position_m",
                "listener_orientation_wxyz",
            }:
                raise RIRCacheError("semantic RIR cache entry is invalid")
            job_id = raw_entry.get("job_id")
            job = self.jobs_by_id.get(str(job_id))
            shard_relative = raw_entry.get("shard")
            row = raw_entry.get("row")
            sample_count = raw_entry.get("sample_count")
            expected_batch_index, expected_row = (
                self.expected_batch_row_by_job_index.get(ordinal, (-1, -1))
            )
            expected_shard = (
                self.expected_batches[expected_batch_index]["shard"]
                if expected_batch_index >= 0
                else None
            )
            if (
                job is None
                or job["job_index"] != ordinal
                or isinstance(raw_entry.get("job_index"), bool)
                or raw_entry.get("job_index") != ordinal
                or not isinstance(shard_relative, str)
                or shard_relative != expected_shard
                or Path(shard_relative).is_absolute()
                or ".." in Path(shard_relative).parts
                or isinstance(row, bool)
                or not isinstance(row, int)
                or row != expected_row
                or isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count < 2
            ):
                raise RIRCacheError("semantic RIR cache entry reference is invalid")
            shard_candidate = self.root / shard_relative
            shard_path = _resolve_semantic_regular_path(
                shard_candidate, owner="semantic RIR cache shard"
            )
            try:
                shard_path.relative_to(self.root)
            except ValueError as exc:
                raise RIRCacheError(
                    "semantic RIR cache shard escapes its root"
                ) from exc
            if not shard_path.is_file() or (shard_path, row) in referenced_rows:
                raise RIRCacheError("semantic RIR cache shard row is invalid")
            referenced_rows.add((shard_path, row))
            expected_source = (
                np.asarray(job["source_position_m"], dtype=np.float64)
                + self.translation_m
            ).tolist()
            expected_listener = (
                np.asarray(job["listener_position_m"], dtype=np.float64)
                + self.translation_m
            ).tolist()
            if (
                raw_entry.get("source_position_m") != expected_source
                or raw_entry.get("listener_position_m") != expected_listener
                or raw_entry.get("listener_orientation_wxyz")
                != job["listener_orientation_wxyz"]
            ):
                raise RIRCacheError(
                    "semantic RIR cache entry pose differs from its plan"
                )
            self.entries_by_id[str(job_id)] = {
                "job_index": ordinal,
                "shard_path": shard_path,
                "row": row,
                "sample_count": sample_count,
                "source_position_m": expected_source,
                "listener_position_m": expected_listener,
                "listener_orientation_wxyz": job["listener_orientation_wxyz"],
            }
        if set(self.entries_by_id) != set(self.jobs_by_id):
            raise RIRCacheError("semantic RIR cache job/index closure is invalid")
        self.retained_shards = (
            shared_shard_cache if shared_shard_cache is not None else {}
        )
        shard_paths = {path for path, _ in referenced_rows}
        if len(shard_paths) != receipt["retained_shard_count"]:
            raise RIRCacheError("semantic RIR cache shard count is invalid")
        for shard_path in shard_paths:
            retained = self.retained_shards.get(shard_path)
            if retained is None:
                retained = _read_semantic_rir_shard(shard_path)
                self.retained_shards[shard_path] = retained
            referenced = {row for path, row in referenced_rows if path == shard_path}
            if referenced != set(range(len(retained["job_ids"]))):
                raise RIRCacheError(
                    "semantic RIR cache shard rows are not exactly indexed"
                )
        for batch_index, spec in enumerate(self.expected_batches):
            shard_path = self.root / spec["shard"]
            retained = self.retained_shards[shard_path]
            timing_batch = raw_timing_batches[batch_index]
            expected_indices = spec["job_indices"]
            expected_job_ids = [
                str(raw_jobs[job_index]["job_id"]) for job_index in expected_indices
            ]
            expected_listener_positions = np.repeat(
                spec["listener_position_m"][None, :],
                len(expected_indices),
                axis=0,
            )
            expected_listener_orientations = np.repeat(
                spec["listener_orientation_wxyz"][None, :],
                len(expected_indices),
                axis=0,
            )
            if (
                retained["job_indices"].tolist() != expected_indices
                or retained["job_ids"].tolist() != expected_job_ids
                or not np.array_equal(
                    retained["listener_positions_m"], expected_listener_positions
                )
                or not np.array_equal(
                    retained["listener_orientations_wxyz"],
                    expected_listener_orientations,
                )
                or not math.isclose(
                    float(retained["simulate_wall_seconds"]),
                    float(timing_batch["simulate_wall_seconds"]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    float(retained["simulate_process_cpu_seconds"]),
                    float(timing_batch["simulate_process_cpu_seconds"]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ):
                raise RIRCacheError(
                    "semantic RIR batch, shard, and timing closure is invalid"
                )

    def load_episode(self, episode_id: str) -> SemanticRIREpisode:
        jobs_by_use = self.jobs_by_episode.get(episode_id)
        if not jobs_by_use:
            raise RIRCacheError(f"semantic RIR cache lacks episode {episode_id!r}")
        frame_sets = {
            slot: {frame for source, frame in jobs_by_use if source == slot}
            for slot in SOURCE_SLOTS
        }
        if (
            any(not values for values in frame_sets.values())
            or len({tuple(sorted(values)) for values in frame_sets.values()}) != 1
            or any(values != set(range(self.frames)) for values in frame_sets.values())
        ):
            raise RIRCacheError(
                "semantic RIR source slots do not share one keyframe grid"
            )
        visual_frames = tuple(sorted(frame_sets[SOURCE_SLOTS[0]]))
        keyframe_samples = tuple(
            round(frame * M5_AUDIO_SAMPLE_RATE_HZ / self.fps) for frame in visual_frames
        )
        if keyframe_samples[0] != 0 or any(
            right <= left for left, right in pairwise(keyframe_samples)
        ):
            raise RIRCacheError("semantic RIR keyframe sample grid is invalid")
        rows: list[list[np.ndarray]] = []
        lengths = np.empty((len(visual_frames), len(SOURCE_SLOTS)), dtype="<u4")
        for frame_ordinal, frame in enumerate(visual_frames):
            frame_values: list[np.ndarray] = []
            for source_ordinal, slot in enumerate(SOURCE_SLOTS):
                job = jobs_by_use[(slot, frame)]
                entry = self.entries_by_id[job["job_id"]]
                shard_path = entry["shard_path"]
                retained = self.retained_shards.get(shard_path)
                if retained is None:
                    retained = _read_semantic_rir_shard(shard_path)
                    self.retained_shards[shard_path] = retained
                row = entry["row"]
                if row >= len(retained["job_ids"]):
                    raise RIRCacheError("semantic RIR row is outside its shard")
                length = int(retained["lengths"][row])
                if (
                    str(retained["job_ids"][row]) != job["job_id"]
                    or int(retained["job_indices"][row]) != entry["job_index"]
                    or length != entry["sample_count"]
                    or not np.array_equal(
                        retained["source_positions_m"][row],
                        np.asarray(entry["source_position_m"], dtype=np.float64),
                    )
                    or not np.array_equal(
                        retained["listener_positions_m"][row],
                        np.asarray(entry["listener_position_m"], dtype=np.float64),
                    )
                    or not np.array_equal(
                        retained["listener_orientations_wxyz"][row],
                        np.asarray(
                            entry["listener_orientation_wxyz"], dtype=np.float64
                        ),
                    )
                ):
                    raise RIRCacheError(
                        "semantic RIR shard row differs from plan/index metadata"
                    )
                frame_values.append(
                    np.ascontiguousarray(retained["samples"][row, :, :length])
                )
                lengths[frame_ordinal, source_ordinal] = length
            rows.append(frame_values)
        maximum_length = max(value.shape[1] for row in rows for value in row)
        samples = np.zeros(
            (len(visual_frames), len(SOURCE_SLOTS), 2, maximum_length),
            dtype="<f4",
        )
        for frame_ordinal, frame_values in enumerate(rows):
            for source_ordinal, value in enumerate(frame_values):
                samples[frame_ordinal, source_ordinal, :, : value.shape[1]] = value
        evidence = {
            "schema": "avengine_m7_semantic_cached_rir_episode_v1",
            "status": "pass",
            "episode_id": episode_id,
            "verification_scope": "schema_path_job_and_sample_metadata_v1",
            "qualification_claim": False,
            "source_slot_ids": list(SOURCE_SLOTS),
            "visual_frame_indices": list(visual_frames),
            "keyframe_samples": list(keyframe_samples),
            "layout_type": "binaural",
            "layout_id": "rlr_binaural_lr_v1",
            "channel_labels": ["left", "right"],
            "sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
        }
        return SemanticRIREpisode(
            samples=np.ascontiguousarray(samples),
            lengths=np.ascontiguousarray(lengths),
            source_slot_ids=SOURCE_SLOTS,
            visual_frame_indices=visual_frames,
            keyframe_samples=keyframe_samples,
            sample_rate_hz=M5_AUDIO_SAMPLE_RATE_HZ,
            layout_type="binaural",
            layout_id="rlr_binaural_lr_v1",
            channel_labels=("left", "right"),
            evidence=evidence,
        )


__all__ = ["SemanticRIRCacheSession", "SemanticRIREpisode"]
