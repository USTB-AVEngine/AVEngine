#!/usr/bin/env python3
"""Assemble many binaural training items from one completed asset-bound cache.

This is deliberately an audio-only batch stage.  It groups all dry-audio
variants of a route together, opens that route's already-completed RIR grid
once, and never starts Habitat or RLR.  RGB/Topdown videos remain a small
separately selected review subset rather than a 1,000-way duplicated render.

The legacy path maps each concrete visual asset to one declared recording.
The M6 path instead consumes validated AudioProgram events and keeps visual
asset identity, sound asset identity, and RIR source-slot identity explicit.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import struct
import time
import wave
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m4.runtime import M4SimulationConfig
from avengine.m4.audio import read_float32_wav, write_float32_wav
from avengine.m5.audio import (
    M5_AUDIO_SAMPLE_COUNT,
    M5_AUDIO_SAMPLE_RATE_HZ,
    raised_cosine_partition,
)
from avengine.m6.audio_render import (
    assemble_audio_program_dry_buses,
    assemble_semantic_audio_program_dry_buses,
)
from avengine.m6.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
    sound_index,
)
from avengine.m6x.rir_cache import (
    RIRCacheError,
    RIRCacheSession,
    validate_semantic_rir_job_plan,
)
from avengine.m7.asset_bound_audio import (
    AssetBoundAudioError,
    PreparedDryAudio,
    bind_endpoint_buses_to_source_slots,
    float32_stems_and_exact_mix,
    prepare_dry_audio,
    render_asset_bound_binaural,
)
from avengine.m7.sensor_rig import (
    M7SensorRigError,
    m7_sensor_rig_binding,
    m7_sensor_rig_pose_series,
    validate_m7_rir_listener_alignment,
)
from avengine.security import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)

SCHEMA = "avengine_m7_asset_bound_binaural_batch_delivery_v1"
SOURCE_SLOTS = ("source1", "source2")
_SEMANTIC_NATIVE_CLAIM = (
    "native CPU RIR samples with structural pose/use, native source/listener "
    "receipts, and decoded-sample validation"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STABLE_PATH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
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


def _validated_acoustic_selection_binding(
    value: Any,
) -> tuple[dict[str, Any], str | None]:
    """Return one authenticated cache binding and its row-reference hash."""

    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "avengine_rir_cache_acoustic_selection_binding_v1"
        or set(value) != _ACOUSTIC_SELECTION_FIELDS
    ):
        raise AssetBoundAudioError("RIR cache acoustic_selection_binding is invalid")
    binding = deepcopy(dict(value))
    mode = binding.get("selection_mode")
    binding_sha256 = binding.get("binding_content_sha256")
    if mode == "explicit_legacy_unbound":
        if (
            binding_sha256 is not None
            or binding.get("registry_selection_applied") is not False
            or binding.get("room_ref") is not None
            or binding.get("profile_ref") is not None
            or binding.get("binding_id") is not None
        ):
            raise AssetBoundAudioError(
                "legacy unbound RIR cache fabricated an acoustic identity"
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
        raise AssetBoundAudioError(
            "RIR cache acoustic_selection_binding hash is invalid"
        )
    return binding, binding_sha256


def _semantic_acoustic_selection_summary(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Project an already validated RIR selection without file evidence."""

    return {
        "schema": "avengine_m7_semantic_rir_selection_summary_v1",
        "selection_mode": binding["selection_mode"],
        "registry_selection_applied": binding["registry_selection_applied"],
        "room_ref": deepcopy(binding["room_ref"]),
        "profile_ref": deepcopy(binding["profile_ref"]),
        "binding_id": binding["binding_id"],
        "qualification_claim": False,
    }


@dataclass(frozen=True)
class AudioProgramSpec:
    """One M6 program document and the exact route variant to materialize."""

    path: Path
    variant_id: str = "A"


@dataclass(frozen=True)
class _SemanticRIREpisode:
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
        raise AssetBoundAudioError(f"{owner} must contain {length} finite numbers")
    return [float(item) for item in value]


def _semantic_unit_orientation(value: Any, *, owner: str) -> list[float]:
    result = _semantic_finite_vector(value, 4, owner=owner)
    if not math.isclose(
        math.sqrt(sum(component * component for component in result)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise AssetBoundAudioError(f"{owner} must be unit normalized")
    return result


def _resolve_semantic_regular_path(
    path: Path, *, owner: str, directory: bool = False
) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise AssetBoundAudioError(f"{owner} may not contain parent traversal")
    absolute = Path(os.path.abspath(raw))
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise AssetBoundAudioError(f"{owner} may not contain symlink components")
    if (directory and not absolute.is_dir()) or (
        not directory and not absolute.is_file()
    ):
        raise AssetBoundAudioError(f"{owner} is missing or has the wrong type")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise AssetBoundAudioError(f"{owner} physical path differs from its selection")
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
        raise AssetBoundAudioError("semantic RIR acoustic selection is invalid")
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
            raise AssetBoundAudioError(
                "semantic registry RIR acoustic selection is incomplete"
            )
    elif mode in {"explicit_legacy", "explicit_legacy_unbound"}:
        if (
            applied is not False
            or room_ref is not None
            or profile_ref is not None
            or binding_id is not None
        ):
            raise AssetBoundAudioError(
                "semantic explicit RIR acoustic selection fabricated an identity"
            )
    else:
        raise AssetBoundAudioError("semantic RIR acoustic selection mode is invalid")
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
                raise AssetBoundAudioError(
                    f"semantic RIR shard fields differ from contract: {path}"
                )
            loaded = {name: np.asarray(archive[name]) for name in required}
            if any(not array.flags.c_contiguous for array in loaded.values()):
                raise AssetBoundAudioError(
                    f"semantic RIR shard arrays must be C-contiguous: {path}"
                )
            result = {name: array.copy(order="C") for name, array in loaded.items()}
    except AssetBoundAudioError:
        raise
    except Exception as exc:
        raise AssetBoundAudioError(f"semantic RIR shard is unreadable: {path}") from exc
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
        raise AssetBoundAudioError(f"semantic RIR shard metadata is invalid: {path}")
    wall = float(result["simulate_wall_seconds"])
    cpu = float(result["simulate_process_cpu_seconds"])
    efficiency = float(result["indirect_ray_efficiency"])
    if (
        not all(math.isfinite(value) for value in (wall, cpu, efficiency))
        or wall < 0
        or cpu < 0
        or not 0 <= efficiency <= 1
    ):
        raise AssetBoundAudioError(f"semantic RIR shard timing is invalid: {path}")
    for row, raw_length in enumerate(lengths):
        length = int(raw_length)
        if not np.any(samples[row, :, :length]) or np.any(samples[row, :, length:]):
            raise AssetBoundAudioError(
                f"semantic RIR shard active/padding samples are invalid: {path} row {row}"
            )
    return result


def _semantic_exact_mapping(
    value: Any, fields: set[str], *, owner: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AssetBoundAudioError(f"{owner} fields differ from semantic contract")
    return value


def _semantic_nonnegative_number(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise AssetBoundAudioError(f"{owner} must be finite and nonnegative")
    return float(value)


def _semantic_canonical_absolute_path(value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AssetBoundAudioError(f"{owner} must be a nonempty absolute path")
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or os.path.abspath(value) != value
    ):
        raise AssetBoundAudioError(f"{owner} must be a canonical absolute path")
    return path


def _semantic_positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AssetBoundAudioError(f"{owner} must be a positive integer")
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
        raise AssetBoundAudioError("semantic native upload structure is invalid")
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
        raise AssetBoundAudioError("semantic native material upload is invalid")


class _SemanticRIRCacheSession:
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
            raise AssetBoundAudioError("semantic RIR cache requires full75/15Hz")
        self.frames = frame_count
        self.fps = frame_rate_hz

        def fixed_json(path: Path, root: Path, *, owner: str) -> Mapping[str, Any]:
            resolved = _resolve_semantic_regular_path(path, owner=owner)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise AssetBoundAudioError(
                    f"{owner} escapes its selected root"
                ) from exc
            value = load_json(resolved)
            if not isinstance(value, Mapping):
                raise AssetBoundAudioError(f"{owner} must contain an object")
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
            raise AssetBoundAudioError("semantic RIR job plan is invalid") from exc
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
            raise AssetBoundAudioError("semantic RIR plan episode selection is invalid")

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
            raise AssetBoundAudioError(
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
            raise AssetBoundAudioError("semantic RIR cache request is invalid")
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
            raise AssetBoundAudioError("semantic RIR cache receipt is invalid")

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
        try:
            binding_path.relative_to(habitat_path.parent)
            binding_within_habitat = True
        except ValueError:
            binding_within_habitat = False
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
            or not binding_within_habitat
            or not config_valid
            or timing_upload.get("status") != "pass_structural_native_upload"
        ):
            raise AssetBoundAudioError("semantic RIR cache timing is invalid")

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
            raise AssetBoundAudioError("semantic RIR cache index is invalid")
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
                raise AssetBoundAudioError("semantic RIR cache entry is invalid")
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
                raise AssetBoundAudioError(
                    "semantic RIR cache entry reference is invalid"
                )
            shard_candidate = self.root / shard_relative
            shard_path = _resolve_semantic_regular_path(
                shard_candidate, owner="semantic RIR cache shard"
            )
            try:
                shard_path.relative_to(self.root)
            except ValueError as exc:
                raise AssetBoundAudioError(
                    "semantic RIR cache shard escapes its root"
                ) from exc
            if not shard_path.is_file() or (shard_path, row) in referenced_rows:
                raise AssetBoundAudioError("semantic RIR cache shard row is invalid")
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
                raise AssetBoundAudioError(
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
            raise AssetBoundAudioError(
                "semantic RIR cache job/index closure is invalid"
            )
        self.retained_shards = (
            shared_shard_cache if shared_shard_cache is not None else {}
        )
        shard_paths = {path for path, _ in referenced_rows}
        if len(shard_paths) != receipt["retained_shard_count"]:
            raise AssetBoundAudioError("semantic RIR cache shard count is invalid")
        for shard_path in shard_paths:
            retained = self.retained_shards.get(shard_path)
            if retained is None:
                retained = _read_semantic_rir_shard(shard_path)
                self.retained_shards[shard_path] = retained
            referenced = {row for path, row in referenced_rows if path == shard_path}
            if referenced != set(range(len(retained["job_ids"]))):
                raise AssetBoundAudioError(
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
                raise AssetBoundAudioError(
                    "semantic RIR batch, shard, and timing closure is invalid"
                )

    def load_episode(self, episode_id: str) -> _SemanticRIREpisode:
        jobs_by_use = self.jobs_by_episode.get(episode_id)
        if not jobs_by_use:
            raise AssetBoundAudioError(
                f"semantic RIR cache lacks episode {episode_id!r}"
            )
        frame_sets = {
            slot: {frame for source, frame in jobs_by_use if source == slot}
            for slot in SOURCE_SLOTS
        }
        if (
            any(not values for values in frame_sets.values())
            or len({tuple(sorted(values)) for values in frame_sets.values()}) != 1
            or any(values != set(range(self.frames)) for values in frame_sets.values())
        ):
            raise AssetBoundAudioError(
                "semantic RIR source slots do not share one keyframe grid"
            )
        visual_frames = tuple(sorted(frame_sets[SOURCE_SLOTS[0]]))
        keyframe_samples = tuple(
            round(frame * M5_AUDIO_SAMPLE_RATE_HZ / self.fps) for frame in visual_frames
        )
        if keyframe_samples[0] != 0 or any(
            right <= left for left, right in pairwise(keyframe_samples)
        ):
            raise AssetBoundAudioError("semantic RIR keyframe sample grid is invalid")
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
                    raise AssetBoundAudioError("semantic RIR row is outside its shard")
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
                    raise AssetBoundAudioError(
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
        return _SemanticRIREpisode(
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


def _load_sensor_rig_contract(plan_root: Path) -> dict[str, Any] | None:
    """Load the optional plan-side rig and close it against every RIR use."""

    rir_plan = load_json(plan_root / "rir_job_plan.json")
    pose_mode = rir_plan.get("listener_pose_mode", "fixed")
    if pose_mode not in {"fixed", "per_episode_frame"}:
        raise AssetBoundAudioError(f"unsupported RIR listener_pose_mode: {pose_mode!r}")
    trajectory_path = plan_root / "sensor_rig_trajectory.json"
    trajectory_declared = trajectory_path.exists() or trajectory_path.is_symlink()
    if not trajectory_declared:
        if pose_mode == "per_episode_frame":
            raise AssetBoundAudioError(
                "per_episode_frame RIR plan requires sensor_rig_trajectory.json"
            )
        return None
    if not trajectory_path.is_file():
        raise AssetBoundAudioError("sensor_rig_trajectory.json must be a regular file")
    trajectory = load_json(trajectory_path)
    try:
        binding = m7_sensor_rig_binding(trajectory)
        poses = m7_sensor_rig_pose_series(trajectory)
        alignment = validate_m7_rir_listener_alignment(
            rir_job_plan=rir_plan,
            sensor_rig_trajectory=trajectory,
        )
    except M7SensorRigError as exc:
        raise AssetBoundAudioError(
            f"sensor-rig/RIR alignment is invalid: {exc}"
        ) from exc
    if len(poses.pose_hashes) != 75:
        raise AssetBoundAudioError("sensor rig must contain the 75 formal M7 frames")
    return {
        "trajectory": dict(trajectory),
        "binding": dict(binding),
        "rir_alignment": dict(alignment),
        "source_path": trajectory_path,
    }


def _slot_mapping(values: list[str], *, name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if separator != "=" or not key or not value:
            raise AssetBoundAudioError(f"{name} must use ASSET_ID=value")
        if key in result:
            raise AssetBoundAudioError(f"{name} specifies {key!r} more than once")
        result[key] = value
    if not result:
        raise AssetBoundAudioError(f"{name} must contain at least one asset")
    return result


def _optional_slot_mapping(values: list[str] | None, *, name: str) -> dict[str, str]:
    return _slot_mapping(values, name=name) if values else {}


def audio_program_specs(
    paths: Sequence[Path], variants: Sequence[str]
) -> tuple[AudioProgramSpec, ...]:
    """Pair repeated CLI program paths with variants, defaulting each to A."""

    if not paths:
        if variants:
            raise AssetBoundAudioError(
                "audio-program-variant requires at least one audio-program"
            )
        return ()
    if variants and len(variants) != len(paths):
        raise AssetBoundAudioError(
            "audio-program-variant count must equal audio-program count"
        )
    selected = tuple(variants) if variants else ("A",) * len(paths)
    return tuple(
        AudioProgramSpec(path=Path(path), variant_id=variant_id)
        for path, variant_id in zip(paths, selected)
    )


def _slot_numbers(values: list[str], *, name: str) -> dict[str, float]:
    raw = _slot_mapping(values, name=name)
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            number = float(value)
        except ValueError as exc:
            raise AssetBoundAudioError(f"{name} {key!r} must be numeric") from exc
        if not np.isfinite(number) or number < 0.0:
            raise AssetBoundAudioError(
                f"{name} {key!r} must be finite and non-negative"
            )
        result[key] = number
    return result


def _pcm16_wave_header(path: str | Path) -> tuple[int, int]:
    source = Path(path).resolve()
    if not source.is_file():
        raise AssetBoundAudioError(f"dry audio is not a regular file: {source}")
    try:
        with wave.open(str(source), "rb") as handle:
            if (
                handle.getsampwidth() != 2
                or handle.getcomptype() != "NONE"
                or handle.getnchannels() not in {1, 2}
            ):
                raise AssetBoundAudioError(
                    "dry audio must be uncompressed PCM16 WAVE with one or two channels"
                )
            return handle.getframerate(), handle.getnframes()
    except (OSError, wave.Error) as exc:
        raise AssetBoundAudioError(f"cannot read PCM WAVE {source}: {exc}") from exc


def variant_start_samples(
    *, source_sample_rate_hz: int, source_sample_count: int, variant_count: int
) -> tuple[int, ...]:
    """Choose distinct full-five-second source windows without looping audio."""

    if (
        isinstance(source_sample_rate_hz, bool)
        or not isinstance(source_sample_rate_hz, int)
        or source_sample_rate_hz < 1
        or isinstance(source_sample_count, bool)
        or not isinstance(source_sample_count, int)
        or source_sample_count < 1
        or isinstance(variant_count, bool)
        or not isinstance(variant_count, int)
        or variant_count < 1
    ):
        raise AssetBoundAudioError("source header or variant count is invalid")
    needed = int(
        np.ceil(M5_AUDIO_SAMPLE_COUNT * source_sample_rate_hz / M5_AUDIO_SAMPLE_RATE_HZ)
    )
    maximum_start = source_sample_count - needed
    if maximum_start < 0:
        raise AssetBoundAudioError(
            "dry audio is shorter than one five-second episode; looping is disabled"
        )
    if variant_count == 1:
        return (0,)
    values = tuple(
        round(value)
        for value in np.linspace(0, maximum_start, variant_count, endpoint=True)
    )
    if len(set(values)) != len(values):
        raise AssetBoundAudioError(
            "dry audio is too short to provide distinct non-looped starts for all variants"
        )
    return values


def _binding_assets(plan_root: Path) -> dict[str, dict[str, str]]:
    report = load_json(plan_root / "asset_emitter_binding_report.json")
    if report.get("status") != "pass" or not isinstance(report.get("scenarios"), list):
        raise AssetBoundAudioError("asset-bound plan binding report is invalid")
    result: dict[str, dict[str, str]] = {}
    for raw in report["scenarios"]:
        episode_id = raw.get("output_episode_id") if isinstance(raw, Mapping) else None
        if (
            not isinstance(episode_id, str)
            or _STABLE_PATH_ID_RE.fullmatch(episode_id) is None
        ):
            raise AssetBoundAudioError(
                "asset-bound scenario requires a path-safe stable output episode ID"
            )
        binding = raw.get("binding_report")
        raw_bindings = binding.get("bindings") if isinstance(binding, Mapping) else None
        if not isinstance(raw_bindings, list) or len(raw_bindings) != len(SOURCE_SLOTS):
            raise AssetBoundAudioError("asset-bound scenario omits source bindings")
        assets: dict[str, str] = {}
        for source in raw_bindings:
            slot = source.get("source_slot_id") if isinstance(source, Mapping) else None
            asset_id = source.get("asset_id") if isinstance(source, Mapping) else None
            if (
                slot not in SOURCE_SLOTS
                or slot in assets
                or not isinstance(asset_id, str)
                or not asset_id
            ):
                raise AssetBoundAudioError(
                    "asset-bound scenario contains an invalid asset"
                )
            assets[slot] = asset_id
        if set(assets) != set(SOURCE_SLOTS):
            raise AssetBoundAudioError("asset-bound scenario omits a source binding")
        if episode_id in result:
            raise AssetBoundAudioError("asset-bound report repeats an episode ID")
        result[episode_id] = assets
    if not result:
        raise AssetBoundAudioError("asset-bound report contains no episodes")
    return result


def _write_and_verify(
    path: Path,
    samples: np.ndarray,
    *,
    role: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = write_float32_wav(
        path,
        samples,
        M5_AUDIO_SAMPLE_RATE_HZ,
        metadata=dict(metadata) | {"role": role},
    )
    readback = read_float32_wav(
        artifact.audio_path, sidecar_path=artifact.sidecar_path, verify_sidecar=True
    )
    expected = np.asarray(samples, dtype=np.float32)
    if readback.samples.shape != expected.shape or not np.array_equal(
        readback.samples, expected
    ):
        raise AssetBoundAudioError(f"float32 WAVE readback differs: {path}")
    return {
        "path": artifact.audio_path.name,
        "sidecar_path": artifact.sidecar_path.name,
        "audio_sha256": artifact.audio_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
        "peak_absolute": float(np.max(np.abs(expected))),
    }


def _write_semantic_and_verify(
    path: Path,
    samples: np.ndarray,
    *,
    role: str,
) -> dict[str, Any]:
    """Write an exact float32 WAVE without a file-digest sidecar.

    The semantic planning branch validates the decoded samples directly.  It
    deliberately emits no file SHA or byte-size field; legacy delivery keeps
    using ``_write_and_verify`` and its authenticated sidecar unchanged.
    """

    expected = np.asarray(samples, dtype=np.float32)
    if expected.ndim != 2 or expected.shape[0] < 1 or expected.shape[1] < 1:
        raise AssetBoundAudioError("semantic WAVE samples must be channel-major")
    if not np.all(np.isfinite(expected)):
        raise AssetBoundAudioError("semantic WAVE samples must be finite")
    channel_count, frame_count = expected.shape
    block_align = channel_count * 4
    byte_rate = M5_AUDIO_SAMPLE_RATE_HZ * block_align
    if channel_count > 65_535 or block_align > 65_535:
        raise AssetBoundAudioError("semantic WAVE channel count is too large")
    converted = np.ascontiguousarray(expected.T, dtype="<f4")

    def chunk(chunk_id: bytes, payload: bytes) -> bytes:
        padding = b"\x00" if len(payload) % 2 else b""
        return chunk_id + struct.pack("<I", len(payload)) + payload + padding

    fmt = struct.pack(
        "<HHIIHH",
        3,
        channel_count,
        M5_AUDIO_SAMPLE_RATE_HZ,
        byte_rate,
        block_align,
        32,
    )
    body = (
        b"WAVE"
        + chunk(b"fmt ", fmt)
        + chunk(b"fact", struct.pack("<I", frame_count))
        + chunk(b"data", converted.tobytes(order="C"))
    )
    payload = b"RIFF" + struct.pack("<I", len(body)) + body
    if len(body) > (1 << 32) - 1:
        raise AssetBoundAudioError("semantic WAVE exceeds RIFF limits")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except OSError as exc:
        raise AssetBoundAudioError(f"cannot write semantic WAVE: {exc}") from exc
    readback = read_float32_wav(path, verify_sidecar=False)
    if readback.samples.shape != expected.shape or not np.array_equal(
        readback.samples, expected
    ):
        raise AssetBoundAudioError(f"semantic float32 WAVE readback differs: {path}")
    return {
        "path": path.name,
        "role": role,
        "sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
        "sample_count": frame_count,
        "channel_count": channel_count,
        "sample_encoding": "IEEE_FLOAT32_LE",
        "peak_absolute": float(np.max(np.abs(expected))),
        "verification": "exact_decoded_sample_equality_no_file_digest_v1",
    }


def _prepare_asset_variants(
    *,
    asset_audio: Mapping[str, str],
    asset_channel_policies: Mapping[str, str],
    asset_gains: Mapping[str, float],
    required_asset_ids: set[str],
    variants_per_episode: int,
    fade_samples: int,
) -> tuple[dict[tuple[str, int], PreparedDryAudio], dict[str, Any]]:
    if set(asset_audio) != required_asset_ids:
        missing = sorted(required_asset_ids - set(asset_audio))
        extra = sorted(set(asset_audio) - required_asset_ids)
        raise AssetBoundAudioError(
            f"asset-audio IDs must exactly match bound assets; missing={missing}, extra={extra}"
        )
    if set(asset_channel_policies) != required_asset_ids:
        raise AssetBoundAudioError(
            "asset-channel-policy IDs must exactly match bound assets"
        )
    if set(asset_gains) != required_asset_ids:
        raise AssetBoundAudioError(
            "asset-linear-gain IDs must exactly match bound assets"
        )
    prepared: dict[tuple[str, int], PreparedDryAudio] = {}
    records: dict[str, Any] = {}
    for asset_id in sorted(required_asset_ids):
        sample_rate_hz, sample_count = _pcm16_wave_header(asset_audio[asset_id])
        starts = variant_start_samples(
            source_sample_rate_hz=sample_rate_hz,
            source_sample_count=sample_count,
            variant_count=variants_per_episode,
        )
        variants = []
        for variant_index, start in enumerate(starts):
            item = prepare_dry_audio(
                asset_audio[asset_id],
                channel_policy=asset_channel_policies[asset_id],
                source_start_sample=start,
                linear_gain=asset_gains[asset_id],
                fade_samples=fade_samples,
            )
            prepared[(asset_id, variant_index)] = item
            variants.append(
                {
                    "variant_index": variant_index,
                    "source_start_sample": start,
                    "record": item.record,
                }
            )
        records[asset_id] = {
            "source_sample_rate_hz": sample_rate_hz,
            "source_sample_count": sample_count,
            "variant_count": variants_per_episode,
            "variants": variants,
        }
    return prepared, records


@dataclass(frozen=True)
class PreparedAudioProgramVariant:
    dry_by_source_slot: Mapping[str, np.ndarray]
    audio_program_binding: Mapping[str, Any]
    instance_record: Mapping[str, Any]
    source_activity_summary: Mapping[str, Any]


def _registry_ref(registry: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        "registry_id": registry["registry_id"],
        "revision": registry["revision"],
        "registry_content_sha256": registry["registry_content_sha256"],
        "input_path": str(path.resolve()),
        "input_sha256": sha256_file(path),
    }


def _source_activity_summary(
    program: Mapping[str, Any],
    endpoint_to_source_slot: Mapping[str, str],
) -> dict[str, Any]:
    intervals: dict[str, list[tuple[int, int]]] = {slot: [] for slot in SOURCE_SLOTS}
    for event in program["events"]:
        intervals[endpoint_to_source_slot[event["source_endpoint_id"]]].append(
            (event["start_sample"], event["end_sample_exclusive"])
        )
    active = tuple(slot for slot in SOURCE_SLOTS if intervals[slot])
    silent = tuple(slot for slot in SOURCE_SLOTS if not intervals[slot])
    overlap = sum(
        max(0, min(left_end, right_end) - max(left_start, right_start))
        for left_start, left_end in intervals["source1"]
        for right_start, right_end in intervals["source2"]
    )
    active_counts = {
        slot: sum(end - start for start, end in intervals[slot])
        for slot in SOURCE_SLOTS
    }
    return {
        "active_source_slots": list(active),
        "silent_source_slots": list(silent),
        "active_sample_count_by_source_slot": active_counts,
        "simultaneous_active_sample_count": overlap,
        "both_sources_have_events": len(active) == len(SOURCE_SLOTS),
        "both_sources_active": overlap > 0,
    }


def _prepare_audio_program_variants(
    *,
    specs: Sequence[AudioProgramSpec],
    source_endpoint_registry_path: Path,
    sound_asset_registry_path: Path,
    endpoint_to_source_slot: Mapping[str, str],
    sound_audio: Mapping[str, str],
) -> tuple[dict[int, PreparedAudioProgramVariant], dict[str, Any]]:
    endpoint_registry_path = source_endpoint_registry_path.resolve()
    sound_registry_path = sound_asset_registry_path.resolve()
    endpoints = load_source_endpoint_registry(endpoint_registry_path)
    sounds = load_sound_asset_registry(sound_registry_path)
    sound_records = sound_index(sounds)
    loaded: list[tuple[AudioProgramSpec, Mapping[str, Any]]] = []
    required_endpoints: set[str] = set()
    required_sounds: set[str] = set()
    for spec in specs:
        path = spec.path.resolve()
        if not path.is_file():
            raise AssetBoundAudioError(f"AudioProgram is missing: {path}")
        program = load_json(path)
        candidates = program.get("candidate_source_endpoint_ids")
        events = program.get("events")
        if not isinstance(candidates, list) or not isinstance(events, list):
            raise AssetBoundAudioError(f"AudioProgram is malformed: {path}")
        required_endpoints.update(str(value) for value in candidates)
        required_sounds.update(
            str(event.get("sound_asset_id"))
            for event in events
            if isinstance(event, Mapping)
        )
        loaded.append(
            (AudioProgramSpec(path=path, variant_id=spec.variant_id), program)
        )
    if set(endpoint_to_source_slot) != required_endpoints:
        raise AssetBoundAudioError(
            "source-endpoint-slot keys must exactly match all AudioProgram candidates"
        )
    missing = sorted(required_sounds - set(sound_audio))
    if missing:
        raise AssetBoundAudioError(
            f"sound-audio bindings are missing used M6 sounds: {missing}"
        )
    unknown_sounds = sorted(required_sounds - set(sound_records))
    if unknown_sounds:
        raise AssetBoundAudioError(
            f"AudioProgram uses unregistered sound assets: {unknown_sounds}"
        )
    asset_bindings = {
        sound_id: {
            "path": str(Path(sound_audio[sound_id]).resolve()),
            "sha256": sound_records[sound_id]["dry_audio"]["sha256"],
        }
        for sound_id in required_sounds
    }
    endpoint_registry_ref = _registry_ref(endpoints, endpoint_registry_path)
    sound_registry_ref = _registry_ref(sounds, sound_registry_path)
    prepared: dict[int, PreparedAudioProgramVariant] = {}
    library: list[dict[str, Any]] = []
    for variant_index, (spec, base_program) in enumerate(loaded):
        assembled = assemble_audio_program_dry_buses(
            base_program,
            spec.variant_id,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
            asset_bindings=asset_bindings,
        )
        materialized = assembled.materialized_program
        compiled = assembled.compiled_program
        if (
            compiled.frame_count != 75
            or materialized["timeline"]["sample_count"] != M5_AUDIO_SAMPLE_COUNT
            or materialized["timeline"]["sample_rate_hz"] != M5_AUDIO_SAMPLE_RATE_HZ
        ):
            raise AssetBoundAudioError(
                "M7 requires a 75-frame, 16 kHz, 80,000-sample AudioProgram"
            )
        mapping = {
            endpoint_id: endpoint_to_source_slot[endpoint_id]
            for endpoint_id in compiled.candidate_source_endpoint_ids
        }
        dry_by_slot = bind_endpoint_buses_to_source_slots(
            assembled.dry_audio.buses,
            endpoint_to_source_slot=mapping,
            source_slots=SOURCE_SLOTS,
        )
        activity = _source_activity_summary(materialized, mapping)
        binding = {
            "audio_program_ref": {
                "program_id": base_program["program_id"],
                "revision": base_program["revision"],
                "program_content_sha256": base_program["program_content_sha256"],
            },
            "variant_id": spec.variant_id,
            "materialized_program_content_sha256": materialized[
                "program_content_sha256"
            ],
            "source_endpoint_to_source_slot": mapping,
            "dry_audio_assembly_content_sha256": (
                assembled.dry_audio.assembly_content_sha256
            ),
        }
        mapped_events = [
            {
                **dict(event),
                "source_slot_id": mapping[event["source_endpoint_id"]],
                "semantic_sound_class": sound_records[event["sound_asset_id"]][
                    "semantic_sound_class"
                ],
            }
            for event in materialized["events"]
        ]
        instance = {
            "schema": "avengine_m7_m6_audio_program_instance_v1",
            "status": "pass",
            "audio_program_binding": binding,
            "source_endpoint_registry_ref": endpoint_registry_ref,
            "sound_asset_registry_ref": sound_registry_ref,
            "program_input": {
                "path": str(spec.path),
                "sha256": sha256_file(spec.path),
            },
            "materialized_audio_program": materialized,
            "base_audio_program": base_program,
            "sound_asset_semantics": {
                sound_id: sound_records[sound_id]["semantic_sound_class"]
                for sound_id in sorted(required_sounds)
            },
            "mapped_events": mapped_events,
            "source_activity_summary": activity,
            "dry_audio_assembly": assembled.dry_audio.metadata(),
        }
        prepared[variant_index] = PreparedAudioProgramVariant(
            dry_by_source_slot=dry_by_slot,
            audio_program_binding=binding,
            instance_record=instance,
            source_activity_summary=activity,
        )
        library.append(
            {
                "variant_index": variant_index,
                "audio_program_binding": binding,
                "source_activity_summary": activity,
                "dry_audio_assembly": assembled.dry_audio.metadata(),
            }
        )
    return prepared, {
        "schema": "avengine_m7_m6_audio_program_dry_bus_library_v1",
        "status": "pass",
        "source_endpoint_registry_ref": endpoint_registry_ref,
        "sound_asset_registry_ref": sound_registry_ref,
        "programs": library,
        "normalization": False,
        "limiting": False,
        "looping": False,
    }


def _prepare_semantic_audio_program_variants(
    *,
    specs: Sequence[AudioProgramSpec],
    expected_episode_id: str,
    semantic_source_endpoint_registry_path: Path,
    semantic_sound_content_registry_path: Path,
    semantic_audio_binding_path: Path,
) -> tuple[dict[int, PreparedAudioProgramVariant], dict[str, Any]]:
    """Prepare planning buses through semantic IDs, never file digests."""

    endpoint_path = semantic_source_endpoint_registry_path.resolve()
    content_path = semantic_sound_content_registry_path.resolve()
    binding_path = semantic_audio_binding_path.resolve()
    endpoints = load_json(endpoint_path)
    contents = load_json(content_path)
    binding = load_json(binding_path)
    if (
        endpoints.get("schema") != "avengine_semantic_source_endpoint_registry_v1"
        or set(endpoints)
        != {"schema", "registry_id", "revision", "source_endpoint_ids"}
        or not isinstance(endpoints.get("source_endpoint_ids"), Mapping)
    ):
        raise AssetBoundAudioError("semantic source endpoint registry is invalid")
    endpoint_to_source_slot = dict(endpoints["source_endpoint_ids"])
    if set(endpoint_to_source_slot.values()) != set(SOURCE_SLOTS) or len(
        set(endpoint_to_source_slot.values())
    ) != len(endpoint_to_source_slot):
        raise AssetBoundAudioError(
            "semantic endpoints must form an exact source-slot bijection"
        )
    if contents.get(
        "schema"
    ) != "avengine_semantic_sound_content_registry_v1" or not isinstance(
        contents.get("contents"), list
    ):
        raise AssetBoundAudioError("semantic sound content registry is invalid")
    if (
        binding.get("schema") != "avengine_semantic_audio_binding_v1"
        or set(binding) != {"schema", "episode_id", "variant_id", "content_bindings"}
        or binding.get("episode_id") != expected_episode_id
        or binding.get("variant_id") != "A"
        or not isinstance(binding.get("content_bindings"), Mapping)
    ):
        raise AssetBoundAudioError("semantic audio binding is invalid")
    registry_content_ids = {
        record.get("content_id")
        for record in contents["contents"]
        if isinstance(record, Mapping)
    }
    if (
        len(registry_content_ids) != len(contents["contents"])
        or None in registry_content_ids
        or set(binding["content_bindings"]) != registry_content_ids
    ):
        raise AssetBoundAudioError(
            "semantic registry and local content binding IDs differ"
        )
    prepared: dict[int, PreparedAudioProgramVariant] = {}
    library: list[dict[str, Any]] = []
    for variant_index, spec in enumerate(specs):
        path = spec.path.resolve()
        if not path.is_file():
            raise AssetBoundAudioError(f"semantic AudioProgram is missing: {path}")
        if spec.variant_id != binding["variant_id"]:
            raise AssetBoundAudioError(
                "semantic AudioProgram variant differs from its binding"
            )
        program = load_json(path)
        assembled = assemble_semantic_audio_program_dry_buses(
            program,
            spec.variant_id,
            source_endpoint_ids=endpoint_to_source_slot,
            semantic_content_registry=contents,
            content_bindings=binding["content_bindings"],
        )
        compiled = assembled.compiled_program
        mapping = {
            endpoint_id: endpoint_to_source_slot[endpoint_id]
            for endpoint_id in compiled.candidate_source_endpoint_ids
        }
        dry_by_slot = bind_endpoint_buses_to_source_slots(
            assembled.dry_audio.buses,
            endpoint_to_source_slot=mapping,
            source_slots=SOURCE_SLOTS,
        )
        activity = _source_activity_summary(assembled.materialized_program, mapping)
        program_binding = {
            "binding_mode": ("semantic_content_id_and_declared_audio_metadata_v1"),
            "audio_program_ref": {
                "program_id": program["program_id"],
                "revision": program["revision"],
                "program_content_sha256": program["program_content_sha256"],
            },
            "variant_id": spec.variant_id,
            "source_endpoint_to_source_slot": mapping,
            "dry_audio_assembly_content_sha256": (
                assembled.dry_audio.assembly_content_sha256
            ),
        }
        content_by_id = {
            record["content_id"]: record for record in contents["contents"]
        }
        instance = {
            "schema": "avengine_m7_semantic_audio_program_instance_v1",
            "status": "pass",
            "qualification_claim": False,
            "audio_program_binding": program_binding,
            "semantic_source_endpoint_registry_ref": {
                "schema": endpoints["schema"],
                "registry_id": endpoints["registry_id"],
                "revision": endpoints["revision"],
            },
            "semantic_sound_content_registry_ref": {
                "schema": contents["schema"],
                "registry_id": contents["registry_id"],
                "revision": contents["revision"],
            },
            "semantic_audio_binding_ref": {
                "episode_id": binding["episode_id"],
                "variant_id": binding["variant_id"],
            },
            "materialized_audio_program": dict(assembled.materialized_program),
            "mapped_events": [
                {
                    **dict(event),
                    "source_slot_id": mapping[event["source_endpoint_id"]],
                    "semantic_content": dict(content_by_id[event["content_id"]]),
                }
                for event in assembled.materialized_program["events"]
            ],
            "source_activity_summary": activity,
            "dry_audio_assembly": assembled.dry_audio.metadata(),
        }
        prepared[variant_index] = PreparedAudioProgramVariant(
            dry_by_source_slot=dry_by_slot,
            audio_program_binding=program_binding,
            instance_record=instance,
            source_activity_summary=activity,
        )
        library.append(
            {
                "variant_index": variant_index,
                "audio_program_binding": program_binding,
                "source_activity_summary": activity,
                "dry_audio_assembly": assembled.dry_audio.metadata(),
            }
        )
    return prepared, {
        "schema": "avengine_m7_semantic_audio_program_dry_bus_library_v1",
        "status": "pass",
        "qualification_claim": False,
        "binding_mode": "semantic_content_id_and_declared_audio_metadata_v1",
        "semantic_source_endpoint_registry_ref": {
            "schema": endpoints["schema"],
            "registry_id": endpoints["registry_id"],
            "revision": endpoints["revision"],
        },
        "semantic_sound_content_registry_ref": {
            "schema": contents["schema"],
            "registry_id": contents["registry_id"],
            "revision": contents["revision"],
        },
        "programs": library,
        "normalization": False,
        "limiting": False,
        "looping": False,
    }


def _fresh_output_path(raw_output: Path) -> Path:
    """Return an absolute output path after rejecting lexical symlink traversal."""

    output = Path(raw_output)
    if not output.is_absolute():
        output = Path.cwd() / output
    if ".." in output.parts:
        raise AssetBoundAudioError("output path may not contain parent traversal")
    current = Path(output.anchor)
    for part in output.parts[1:]:
        current /= part
        if current.is_symlink():
            raise AssetBoundAudioError(
                f"output path may not contain symlinks: {current}"
            )
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    return output


def _derived_output_path(root: Path, filename: str) -> Path:
    """Return one direct child and reject any derived lexical escape."""

    candidate = root / filename
    if candidate.parent != root:
        raise AssetBoundAudioError("derived audio output escapes its selected root")
    return candidate


def render_batch(
    *,
    plan_root: Path,
    rir_cache_root: Path,
    asset_audio: Mapping[str, str] | None,
    asset_channel_policies: Mapping[str, str] | None,
    asset_gains: Mapping[str, float] | None,
    variants_per_episode: int | None,
    fade_samples: int,
    maximum_mixture_peak: float,
    retain_stems: bool,
    output: Path,
    audio_program_specs: Sequence[AudioProgramSpec] = (),
    source_endpoint_registry_path: Path | None = None,
    sound_asset_registry_path: Path | None = None,
    endpoint_to_source_slot: Mapping[str, str] | None = None,
    sound_audio: Mapping[str, str] | None = None,
    semantic_source_endpoint_registry_path: Path | None = None,
    semantic_sound_content_registry_path: Path | None = None,
    semantic_audio_binding_path: Path | None = None,
) -> Path:
    started = time.perf_counter()
    raw_plan_root = Path(plan_root)
    raw_rir_cache_root = Path(rir_cache_root)
    output = _fresh_output_path(output)
    program_mode = bool(audio_program_specs)
    semantic_paths = (
        semantic_source_endpoint_registry_path,
        semantic_sound_content_registry_path,
        semantic_audio_binding_path,
    )
    semantic_program_mode = program_mode and all(
        value is not None for value in semantic_paths
    )
    partial_semantic_mode = any(
        value is not None for value in semantic_paths
    ) and not all(value is not None for value in semantic_paths)
    legacy_program_values = (
        source_endpoint_registry_path,
        sound_asset_registry_path,
        endpoint_to_source_slot,
        sound_audio,
    )
    if partial_semantic_mode:
        raise AssetBoundAudioError(
            "semantic AudioPrograms require all three semantic inputs"
        )
    if semantic_program_mode and any(
        value is not None and value != {} for value in legacy_program_values
    ):
        raise AssetBoundAudioError(
            "semantic and legacy AudioProgram inputs are mutually exclusive"
        )
    if semantic_program_mode:
        raw_plan_path = raw_plan_root / "rir_job_plan.json"
        if (
            raw_plan_root.is_symlink()
            or not raw_plan_root.is_dir()
            or raw_rir_cache_root.is_symlink()
            or not raw_rir_cache_root.is_dir()
            or raw_plan_path.is_symlink()
            or not raw_plan_path.is_file()
        ):
            raise AssetBoundAudioError(
                "semantic RIR plan root, cache root, and selected plan must be "
                "regular non-symlink inputs"
            )
        resolved_plan_root = raw_plan_root.resolve()
        resolved_plan_path = raw_plan_path.resolve()
        try:
            resolved_plan_path.relative_to(resolved_plan_root)
        except ValueError as exc:
            raise AssetBoundAudioError(
                "semantic selected RIR plan escapes its plan root"
            ) from exc
    plan_root = raw_plan_root.resolve()
    rir_cache_root = raw_rir_cache_root.resolve()
    if not np.isfinite(maximum_mixture_peak) or not 0.0 < maximum_mixture_peak <= 1.0:
        raise AssetBoundAudioError("maximum_mixture_peak must be in (0, 1]")
    bindings = _binding_assets(plan_root)
    sensor_rig_contract = _load_sensor_rig_contract(plan_root)
    sensor_rig_binding = (
        None if sensor_rig_contract is None else dict(sensor_rig_contract["binding"])
    )
    required_assets = {asset for slots in bindings.values() for asset in slots.values()}
    legacy_prepared: dict[tuple[str, int], PreparedDryAudio] = {}
    program_prepared: dict[int, PreparedAudioProgramVariant] = {}
    if program_mode:
        if any((asset_audio, asset_channel_policies, asset_gains)):
            raise AssetBoundAudioError(
                "asset-audio declarations cannot be mixed with M6 AudioPrograms"
            )
        if not semantic_program_mode and (
            source_endpoint_registry_path is None
            or sound_asset_registry_path is None
            or endpoint_to_source_slot is None
            or sound_audio is None
        ):
            raise AssetBoundAudioError(
                "M6 AudioPrograms require either the semantic input triple or both "
                "legacy registries, endpoint-slot mappings, and sound-audio paths"
            )
        program_variant_count = len(audio_program_specs)
        if program_variant_count != 1:
            raise AssetBoundAudioError(
                "M7 AudioProgram mode currently requires exactly one program "
                "instance per visual episode"
            )
        if (
            variants_per_episode is not None
            and variants_per_episode != program_variant_count
        ):
            raise AssetBoundAudioError(
                "variants_per_episode must equal the number of AudioProgram specs"
            )
        variants_per_episode = program_variant_count
        if semantic_program_mode:
            if len(bindings) != 1:
                raise AssetBoundAudioError(
                    "semantic AudioProgram mode requires exactly one bound episode"
                )
            assert semantic_source_endpoint_registry_path is not None
            assert semantic_sound_content_registry_path is not None
            assert semantic_audio_binding_path is not None
            program_prepared, dry_library_record = (
                _prepare_semantic_audio_program_variants(
                    specs=audio_program_specs,
                    expected_episode_id=next(iter(bindings)),
                    semantic_source_endpoint_registry_path=(
                        semantic_source_endpoint_registry_path
                    ),
                    semantic_sound_content_registry_path=(
                        semantic_sound_content_registry_path
                    ),
                    semantic_audio_binding_path=semantic_audio_binding_path,
                )
            )
        else:
            assert source_endpoint_registry_path is not None
            assert sound_asset_registry_path is not None
            assert endpoint_to_source_slot is not None
            assert sound_audio is not None
            program_prepared, dry_library_record = _prepare_audio_program_variants(
                specs=audio_program_specs,
                source_endpoint_registry_path=source_endpoint_registry_path,
                sound_asset_registry_path=sound_asset_registry_path,
                endpoint_to_source_slot=endpoint_to_source_slot,
                sound_audio=sound_audio,
            )
    else:
        if (
            not asset_audio
            or not asset_channel_policies
            or not asset_gains
            or source_endpoint_registry_path is not None
            or sound_asset_registry_path is not None
            or endpoint_to_source_slot
            or sound_audio
            or any(value is not None for value in semantic_paths)
        ):
            raise AssetBoundAudioError(
                "legacy mode requires asset audio/channel/gain declarations and "
                "does not accept AudioProgram inputs"
            )
        variants_per_episode = (
            10 if variants_per_episode is None else variants_per_episode
        )
        if (
            isinstance(variants_per_episode, bool)
            or not isinstance(variants_per_episode, int)
            or variants_per_episode < 1
        ):
            raise AssetBoundAudioError("variants_per_episode must be positive")
        legacy_prepared, dry_variant_records = _prepare_asset_variants(
            asset_audio=asset_audio,
            asset_channel_policies=asset_channel_policies,
            asset_gains=asset_gains,
            required_asset_ids=required_assets,
            variants_per_episode=variants_per_episode,
            fade_samples=fade_samples,
        )
        dry_library_record = {
            "schema": "avengine_m7_asset_bound_dry_variant_library_v1",
            "status": "pass",
            "assets": dry_variant_records,
            "normalization": False,
            "limiting": False,
            "looping": False,
        }
    assert isinstance(variants_per_episode, int)
    if output.parent.exists() and output.parent.is_symlink():
        raise AssetBoundAudioError("output parent may not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to replace staging output: {staging}")
    phase_seconds: defaultdict[str, float] = defaultdict(float)
    sample_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    partitions_by_keyframe_grid: dict[tuple[int, ...], np.ndarray] = {}
    shared_rir_shards: dict[Path, dict[str, Any]] = {}
    program_instance_paths: dict[int, str] = {}
    program_instance_sha256s: dict[int, str] = {}
    try:
        staging.mkdir()
        if sensor_rig_contract is not None:
            sensor_rig_relative = Path("labels") / "sensor_rig_trajectory.json"
            sensor_rig_output = staging / sensor_rig_relative
            sensor_rig_output.parent.mkdir(parents=True, exist_ok=True)
            source = Path(sensor_rig_contract["source_path"])
            shutil.copyfile(source, sensor_rig_output)
            if (
                sensor_rig_output.read_bytes() != source.read_bytes()
                or m7_sensor_rig_binding(load_json(sensor_rig_output))
                != sensor_rig_binding
            ):
                raise AssetBoundAudioError(
                    "copied sensor-rig sidecar differs from the plan"
                )
        if program_mode:
            for variant_index, prepared_program in program_prepared.items():
                relative = (
                    Path("labels")
                    / "audio_program_instances"
                    / f"v{variant_index:02d}.json"
                )
                path = staging / relative
                write_json(path, prepared_program.instance_record)
                program_instance_paths[variant_index] = relative.as_posix()
                if not semantic_program_mode:
                    program_instance_sha256s[variant_index] = sha256_file(path)
        mixture_root = staging / "audio" / "binaural"
        stem_root = mixture_root / "stems"
        if semantic_program_mode:
            rir_session = _SemanticRIRCacheSession(
                cache_root=rir_cache_root,
                plan_path=plan_root / "rir_job_plan.json",
                expected_episode_id=next(iter(bindings)),
                frame_count=75,
                frame_rate_hz=15,
                shared_shard_cache=shared_rir_shards,
            )
            acoustic_selection_binding = rir_session.acoustic_selection_binding
            acoustic_selection_binding_sha256 = None
        else:
            rir_session = RIRCacheSession(
                cache_root=rir_cache_root,
                plan_path=plan_root / "rir_job_plan.json",
                frame_count=75,
                frame_rate_hz=15,
                shared_shard_cache=shared_rir_shards,
            )
            (
                acoustic_selection_binding,
                acoustic_selection_binding_sha256,
            ) = _validated_acoustic_selection_binding(
                rir_session.acoustic_selection_binding
            )
        cache_load_count = 0
        for episode_ordinal, episode_id in enumerate(sorted(bindings)):
            cache_started = time.perf_counter()
            cached = rir_session.load_episode(episode_id)
            phase_seconds["rir_cache_load"] += time.perf_counter() - cache_started
            cache_load_count += 1
            if (
                cached.layout_type != "binaural"
                or cached.sample_rate_hz != M5_AUDIO_SAMPLE_RATE_HZ
                or cached.channel_labels != ("left", "right")
                or cached.source_slot_ids != SOURCE_SLOTS
            ):
                raise AssetBoundAudioError("cache is not 16 kHz two-channel binaural")
            if not semantic_program_mode and (
                cached.evidence.get("acoustic_selection_binding")
                != acoustic_selection_binding
            ):
                raise AssetBoundAudioError(
                    "episode RIR cache acoustic selection differs from its session"
                )
            partition_key = tuple(cached.keyframe_samples)
            partition_weights = partitions_by_keyframe_grid.get(partition_key)
            if partition_weights is None:
                partition_weights = raised_cosine_partition(
                    partition_key, M5_AUDIO_SAMPLE_COUNT
                )
                partitions_by_keyframe_grid[partition_key] = partition_weights
            episode_record: dict[str, Any] = {
                "episode_id": episode_id,
                "episode_ordinal": episode_ordinal,
                "asset_ids_by_source_slot": bindings[episode_id],
                "cache_load_policy": "loaded_once_then_reused_for_all_episode_variants",
            }
            if semantic_program_mode:
                episode_record["rir_cache"] = {
                    "binding_mode": "schema_path_job_and_sample_metadata_v1",
                    "layout_type": cached.layout_type,
                    "sample_rate_hz": cached.sample_rate_hz,
                    "channel_labels": list(cached.channel_labels),
                    "source_slot_ids": list(cached.source_slot_ids),
                    "keyframe_samples": list(cached.keyframe_samples),
                    "verification_scope": ("schema_path_job_and_sample_metadata_v1"),
                    "qualification_claim": False,
                }
            else:
                episode_record["rir_cache"] = cached.evidence
                episode_record["acoustic_selection_binding_sha256"] = (
                    acoustic_selection_binding_sha256
                )
            if sensor_rig_binding is not None:
                episode_record["sensor_rig_trajectory"] = dict(sensor_rig_binding)
            episode_records.append(episode_record)
            for variant_index in range(variants_per_episode):
                sample_id = f"{episode_id}__v{variant_index:02d}"
                mixture_path = _derived_output_path(mixture_root, f"{sample_id}.wav")
                stem_paths = {
                    slot: _derived_output_path(stem_root / slot, f"{sample_id}.wav")
                    for slot in SOURCE_SLOTS
                }
                prepared_program = (
                    program_prepared[variant_index] if program_mode else None
                )
                dry_by_source = (
                    dict(prepared_program.dry_by_source_slot)
                    if prepared_program is not None
                    else {
                        slot: legacy_prepared[
                            (bindings[episode_id][slot], variant_index)
                        ].samples
                        for slot in SOURCE_SLOTS
                    }
                )
                render_started = time.perf_counter()
                stems, _mixture64 = render_asset_bound_binaural(
                    dry_by_source,
                    rir_samples=cached.samples,
                    rir_lengths=cached.lengths,
                    source_ids=cached.source_slot_ids,
                    keyframe_samples=cached.keyframe_samples,
                    partition_weights=partition_weights,
                )
                stored_stems, stored_mixture = float32_stems_and_exact_mix(
                    stems, source_ids=cached.source_slot_ids
                )
                phase_seconds["dynamic_convolution"] += (
                    time.perf_counter() - render_started
                )
                peak = float(np.max(np.abs(stored_mixture)))
                if peak > maximum_mixture_peak:
                    raise AssetBoundAudioError(
                        f"{sample_id} mixture peak {peak:.6f} exceeds maximum_mixture_peak "
                        f"{maximum_mixture_peak:.6f}; reduce the declared input gain"
                    )
                write_started = time.perf_counter()
                mixture_metadata = {
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "variant_index": variant_index,
                    "mixture": "exact_source1_plus_source2_stem_sum",
                    "normalization": False,
                    "limiting": False,
                }
                if not semantic_program_mode:
                    mixture_metadata["acoustic_selection_binding_sha256"] = (
                        acoustic_selection_binding_sha256
                    )
                if prepared_program is not None:
                    mixture_metadata.update(
                        {
                            "audio_program_mode": True,
                            "audio_program_binding": dict(
                                prepared_program.audio_program_binding
                            ),
                            "audio_program_instance_path": (
                                program_instance_paths[variant_index]
                            ),
                        }
                    )
                    if not semantic_program_mode:
                        mixture_metadata["audio_program_instance_sha256"] = (
                            program_instance_sha256s[variant_index]
                        )
                mixture_record = (
                    _write_semantic_and_verify(
                        mixture_path,
                        stored_mixture,
                        role="m7_asset_bound_binaural_training_mixture",
                    )
                    if semantic_program_mode
                    else _write_and_verify(
                        mixture_path,
                        stored_mixture,
                        role="m7_asset_bound_binaural_training_mixture",
                        metadata=mixture_metadata,
                    )
                )
                stem_records: dict[str, Any] = {}
                if retain_stems:
                    for slot in SOURCE_SLOTS:
                        stem_metadata = {
                            "sample_id": sample_id,
                            "episode_id": episode_id,
                            "variant_index": variant_index,
                            "source_slot_id": slot,
                        }
                        if not semantic_program_mode:
                            stem_metadata["acoustic_selection_binding_sha256"] = (
                                acoustic_selection_binding_sha256
                            )
                        if prepared_program is not None:
                            stem_metadata.update(
                                {
                                    "audio_program_mode": True,
                                    "audio_program_binding": dict(
                                        prepared_program.audio_program_binding
                                    ),
                                    "audio_program_instance_path": (
                                        program_instance_paths[variant_index]
                                    ),
                                }
                            )
                            if not semantic_program_mode:
                                stem_metadata["audio_program_instance_sha256"] = (
                                    program_instance_sha256s[variant_index]
                                )
                        stem_records[slot] = (
                            _write_semantic_and_verify(
                                stem_paths[slot],
                                stored_stems[slot],
                                role="m7_asset_bound_binaural_training_stem",
                            )
                            if semantic_program_mode
                            else _write_and_verify(
                                stem_paths[slot],
                                stored_stems[slot],
                                role="m7_asset_bound_binaural_training_stem",
                                metadata=stem_metadata,
                            )
                        )
                phase_seconds["wave_write_and_readback"] += (
                    time.perf_counter() - write_started
                )
                sample_record: dict[str, Any] = {
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "variant_index": variant_index,
                    "asset_ids_by_source_slot": bindings[episode_id],
                    "audio": {
                        "sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
                        "sample_count": M5_AUDIO_SAMPLE_COUNT,
                        "channel_count": 2,
                        "layout": "native_RLR_HRTF_binaural_left_right",
                        "mixture": mixture_record,
                        "stems_retained": retain_stems,
                        "stems": stem_records,
                        "mixture_is_exact_stem_sum_before_delivery": True,
                        "peak_absolute": peak,
                    },
                }
                if not semantic_program_mode:
                    sample_record["acoustic_selection_binding_sha256"] = (
                        acoustic_selection_binding_sha256
                    )
                if sensor_rig_binding is not None:
                    sample_record["sensor_rig_trajectory"] = dict(sensor_rig_binding)
                if prepared_program is None:
                    sample_record.update(
                        {
                            "dry_variant_ids_by_source_slot": {
                                slot: {
                                    "asset_id": bindings[episode_id][slot],
                                    "variant_index": variant_index,
                                }
                                for slot in SOURCE_SLOTS
                            },
                            "both_sources_active": True,
                        }
                    )
                else:
                    activity = dict(prepared_program.source_activity_summary)
                    program_fields = {
                        "audio_program_binding": dict(
                            prepared_program.audio_program_binding
                        ),
                        "audio_program_instance_path": (
                            program_instance_paths[variant_index]
                        ),
                        "source_activity_summary": activity,
                        "both_sources_active": activity["both_sources_active"],
                        "source_activity_contract": (
                            "m6_audio_program_event_windows_v1"
                        ),
                    }
                    if not semantic_program_mode:
                        program_fields["audio_program_instance_sha256"] = (
                            program_instance_sha256s[variant_index]
                        )
                    sample_record.update(program_fields)
                sample_records.append(sample_record)
        write_json(staging / "dry_audio_variants.json", dry_library_record)
        acoustic_output = (
            _semantic_acoustic_selection_summary(acoustic_selection_binding)
            if semantic_program_mode
            else acoustic_selection_binding
        )
        write_json(
            staging / "episodes.json",
            {
                "schema": "avengine_m7_asset_bound_episode_cache_index_v1",
                "status": "pass",
                "acoustic_selection_binding": acoustic_output,
                "episodes": episode_records,
            },
        )
        write_json(
            staging / "samples.json",
            {
                "schema": "avengine_m7_asset_bound_binaural_training_samples_v1",
                "status": "pass",
                "acoustic_selection_binding": acoustic_output,
                "sample_count": len(sample_records),
                "samples": sample_records,
            },
        )
        total_wall = time.perf_counter() - started
        timing = {
            "schema": "avengine_m7_asset_bound_binaural_batch_timing_v1",
            "status": "pass",
            "native_rlr_calls": 0,
            "visual_render_calls": 0,
            "rir_cache_load_count": cache_load_count,
            "rir_cache_load_policy": "one_load_per_route_then_all_its_variants",
            "rir_shard_residency": {
                "policy": (
                    "validate_schema_job_and_sample_metadata_once_then_reuse_v1"
                    if semantic_program_mode
                    else "verify_each_native_RIR_shard_once_then_reuse_in_process"
                ),
                "resident_shard_count": len(shared_rir_shards),
            },
            "episode_count": len(bindings),
            "variants_per_episode": variants_per_episode,
            "sample_count": len(sample_records),
            "phase_seconds": dict(phase_seconds),
            "wall_seconds": total_wall,
            "samples_per_wall_second": len(sample_records) / total_wall,
            "projected_1000_sample_seconds": 1000.0
            / (len(sample_records) / total_wall),
        }
        if not semantic_program_mode:
            timing["rir_shard_residency"]["resident_sample_payload_bytes"] = int(
                sum(value["samples"].nbytes for value in shared_rir_shards.values())
            )
        write_json(staging / "timing.json", timing)
        delivery = {
            "schema": SCHEMA,
            "status": "pass",
            "qualification_claim": False,
            "claim_boundary": (
                "research throughput delivery: reuses a completed asset-bound RIR "
                "cache, writes binaural mixtures and labels, and does not render video "
                "or claim dataset admission"
            ),
            "episode_count": len(bindings),
            "variants_per_episode": variants_per_episode,
            "sample_count": len(sample_records),
            "acoustic_selection_binding": acoustic_output,
            "both_sources_active": (
                all(record["both_sources_active"] for record in sample_records)
                if program_mode
                else True
            ),
            "binaural_layout": "native_RLR_HRTF_binaural_left_right",
            "outputs": {
                "dry_audio_variants": "dry_audio_variants.json",
                "episode_cache_index": "episodes.json",
                "samples": "samples.json",
                "timing": "timing.json",
                "mixtures": "audio/binaural/",
                "stems": "audio/binaural/stems/" if retain_stems else None,
            },
        }
        if sensor_rig_binding is not None:
            delivery["sensor_rig_trajectory"] = dict(sensor_rig_binding)
            delivery["sensor_rig_rir_alignment"] = dict(
                sensor_rig_contract["rir_alignment"]
            )
            delivery["outputs"]["sensor_rig_trajectory"] = (
                "labels/sensor_rig_trajectory.json"
            )
        if program_mode:
            delivery["source_activity_contract"] = "m6_audio_program_event_windows_v1"
            delivery["audio_program_binding_mode"] = (
                "semantic_content_id_and_declared_audio_metadata_v1"
                if semantic_program_mode
                else "authenticated_m6_registry_file_binding_v1"
            )
            delivery["outputs"]["audio_program_instances"] = (
                "labels/audio_program_instances/"
            )
        write_json(staging / "delivery.json", delivery)
        publish_policy = WorkspacePathPolicy.from_roots([output.parent])
        output = atomic_publish_directory(publish_policy, staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--rir-cache", type=Path, required=True)
    parser.add_argument("--asset-audio", action="append")
    parser.add_argument("--asset-channel-policy", action="append")
    parser.add_argument("--asset-linear-gain", action="append")
    parser.add_argument("--audio-program", type=Path, action="append", default=[])
    parser.add_argument("--audio-program-variant", action="append", default=[])
    parser.add_argument("--source-endpoint-registry", type=Path)
    parser.add_argument("--sound-asset-registry", type=Path)
    parser.add_argument("--source-endpoint-slot", action="append")
    parser.add_argument("--sound-audio", action="append")
    parser.add_argument("--semantic-source-endpoint-registry", type=Path)
    parser.add_argument("--semantic-sound-content-registry", type=Path)
    parser.add_argument("--semantic-audio-binding", type=Path)
    parser.add_argument("--variants-per-episode", type=int)
    parser.add_argument("--fade-samples", type=int, default=80)
    parser.add_argument("--maximum-mixture-peak", type=float, default=0.95)
    parser.add_argument("--retain-stems", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    programs = audio_program_specs(args.audio_program, args.audio_program_variant)
    result = render_batch(
        plan_root=args.plan_root,
        rir_cache_root=args.rir_cache,
        asset_audio=_optional_slot_mapping(args.asset_audio, name="asset-audio"),
        asset_channel_policies=_optional_slot_mapping(
            args.asset_channel_policy, name="asset-channel-policy"
        ),
        asset_gains=(
            _slot_numbers(args.asset_linear_gain, name="asset-linear-gain")
            if args.asset_linear_gain
            else {}
        ),
        variants_per_episode=args.variants_per_episode,
        fade_samples=args.fade_samples,
        maximum_mixture_peak=args.maximum_mixture_peak,
        retain_stems=args.retain_stems,
        output=args.output,
        audio_program_specs=programs,
        source_endpoint_registry_path=args.source_endpoint_registry,
        sound_asset_registry_path=args.sound_asset_registry,
        endpoint_to_source_slot=_optional_slot_mapping(
            args.source_endpoint_slot, name="source-endpoint-slot"
        ),
        sound_audio=_optional_slot_mapping(args.sound_audio, name="sound-audio"),
        semantic_source_endpoint_registry_path=(args.semantic_source_endpoint_registry),
        semantic_sound_content_registry_path=(args.semantic_sound_content_registry),
        semantic_audio_binding_path=args.semantic_audio_binding,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
