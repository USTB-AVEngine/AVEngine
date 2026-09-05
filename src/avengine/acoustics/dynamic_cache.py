"""Read-only compatibility loader for legacy dynamic binaural RIR sequences.

Current production rendering uses the established
avengine.acoustics.rir_cache request/index/receipt/shard contract. This module
only validates already-produced variable-source sequence caches from earlier
research tasks so their audio can be replayed or inspected. It deliberately
has no production cache writer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256


DYNAMIC_RIR_CACHE_SCHEMA = "avengine_dynamic_rir_sequence_cache_v1"


class DynamicRIRCacheError(ValueError):
    """A dynamic sequence cache is malformed or does not match the request."""


@dataclass(frozen=True)
class DynamicRIRCachePayload:
    """Validated arrays and metadata loaded from one dynamic cache."""

    samples: np.ndarray
    lengths: np.ndarray
    metadata: Mapping[str, Any]


def _canonical_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicRIRCacheError("dynamic RIR cache metadata must be an object")
    result = json.loads(json.dumps(dict(value), sort_keys=True))
    if not isinstance(result, dict):  # pragma: no cover - guarded by Mapping
        raise DynamicRIRCacheError("dynamic RIR cache metadata is not JSON")
    return result


def _array_digest(samples: np.ndarray, lengths: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(samples).tobytes(order="C"))
    digest.update(np.ascontiguousarray(lengths).tobytes(order="C"))
    return digest.hexdigest()


def _request_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Select result-changing request fields from an on-disk manifest."""

    generated = {
        "array_shape",
        "array_dtype",
        "lengths_shape",
        "lengths_dtype",
        "content_sha256",
        "cache_identity_sha256",
        # Runtime evidence is useful to retain but must not make a matching
        # request miss merely because native timing or receipt values differ.
        "native_renderer_setup",
        "ir_sha256_by_keyframe_source",
        "indirect_ray_efficiency",
        "timings",
        "interpolation",
        "final_interval_policy",
        "native_context_policy",
        "keyframe_count",
    }
    return {key: item for key, item in value.items() if key not in generated}


def _validate_arrays(
    samples: Any,
    lengths: Any,
    *,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples)
    counts = np.asarray(lengths)
    if values.ndim != 4 or values.dtype.kind not in "fiu":
        raise DynamicRIRCacheError(
            "dynamic RIR cache samples must have [keyframe,source,channel,sample] shape"
        )
    if counts.ndim != 2 or counts.dtype.kind not in "iu":
        raise DynamicRIRCacheError(
            "dynamic RIR cache lengths must have [keyframe,source] integer shape"
        )
    if tuple(counts.shape) != tuple(values.shape[:2]):
        raise DynamicRIRCacheError(
            "dynamic RIR cache lengths do not match keyframe/source dimensions"
        )
    values = np.ascontiguousarray(values, dtype="<f4")
    counts = np.ascontiguousarray(counts, dtype="<u4")
    if not np.all(np.isfinite(values)):
        raise DynamicRIRCacheError("dynamic RIR cache samples contain non-finite values")
    if np.any(counts < 1) or np.any(counts > values.shape[3]):
        raise DynamicRIRCacheError("dynamic RIR cache lengths escape the padded extent")
    source_ids = metadata.get("source_ids")
    keyframe_samples = metadata.get("keyframe_samples")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != values.shape[1]
        or any(not isinstance(value, str) or not value for value in source_ids)
        or source_ids != sorted(set(source_ids))
    ):
        raise DynamicRIRCacheError("dynamic RIR cache source IDs are invalid")
    if (
        not isinstance(keyframe_samples, list)
        or len(keyframe_samples) != values.shape[0]
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in keyframe_samples
        )
        or not keyframe_samples
        or keyframe_samples[0] != 0
        or any(right <= left for left, right in zip(keyframe_samples, keyframe_samples[1:]))
    ):
        raise DynamicRIRCacheError("dynamic RIR cache keyframe sample grid is invalid")
    return values, counts


def load_dynamic_rir_cache(
    root: str | Path,
    *,
    expected_cache_identity_sha256: str | None = None,
    expected_source_ids: Sequence[str] | None = None,
    expected_keyframe_samples: Sequence[int] | None = None,
) -> DynamicRIRCachePayload:
    """Read and validate one persisted dynamic sequence cache."""

    requested_root = Path(root).expanduser()
    if requested_root.is_symlink():
        raise DynamicRIRCacheError(f"dynamic RIR cache directory is a symlink: {requested_root}")
    cache_root = requested_root.resolve()
    if not cache_root.is_dir() or cache_root.is_symlink():
        raise DynamicRIRCacheError(
            f"dynamic RIR cache directory is unavailable: {cache_root}"
        )
    manifest_path = cache_root / "manifest.json"
    sequence_path = cache_root / "sequence.npz"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(sequence_path, allow_pickle=False) as payload:
            samples = np.array(payload["samples"], copy=True)
            lengths = np.array(payload["lengths"], copy=True)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise DynamicRIRCacheError(f"cannot read dynamic RIR cache: {exc}") from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != DYNAMIC_RIR_CACHE_SCHEMA
    ):
        raise DynamicRIRCacheError("dynamic RIR cache manifest schema is invalid")
    normalized = _canonical_metadata(manifest)
    values, counts = _validate_arrays(samples, lengths, metadata=normalized)
    if (
        normalized.get("array_shape") != list(values.shape)
        or normalized.get("lengths_shape") != list(counts.shape)
    ):
        raise DynamicRIRCacheError(
            "dynamic RIR cache array shape metadata differs from payload"
        )
    if (
        normalized.get("array_dtype") != values.dtype.str
        or normalized.get("lengths_dtype") != counts.dtype.str
    ):
        raise DynamicRIRCacheError(
            "dynamic RIR cache array dtype metadata differs from payload"
        )
    if normalized.get("content_sha256") != _array_digest(values, counts):
        raise DynamicRIRCacheError(
            "dynamic RIR cache payload integrity differs from manifest"
        )
    if normalized.get("cache_identity_sha256") != canonical_json_sha256(
        _request_metadata(normalized)
    ):
        raise DynamicRIRCacheError("dynamic RIR cache identity differs from manifest")
    if (
        expected_cache_identity_sha256 is not None
        and normalized.get("cache_identity_sha256") != expected_cache_identity_sha256
    ):
        raise DynamicRIRCacheError(
            "dynamic RIR cache identity differs from requested episode"
        )
    if (
        expected_source_ids is not None
        and normalized.get("source_ids") != list(expected_source_ids)
    ):
        raise DynamicRIRCacheError(
            "dynamic RIR cache source IDs differ from requested episode"
        )
    if (
        expected_keyframe_samples is not None
        and normalized.get("keyframe_samples") != list(expected_keyframe_samples)
    ):
        raise DynamicRIRCacheError(
            "dynamic RIR cache keyframe grid differs from requested episode"
        )
    values.setflags(write=False)
    counts.setflags(write=False)
    return DynamicRIRCachePayload(values, counts, normalized)


__all__ = [
    "DYNAMIC_RIR_CACHE_SCHEMA",
    "DynamicRIRCacheError",
    "DynamicRIRCachePayload",
    "load_dynamic_rir_cache",
]
