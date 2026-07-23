"""Deterministic native-360 and legacy label helpers for the v4_3 experiment.

AVEngine uses a full-circle listener-local azimuth: front is 0 degrees,
right is +90, left is -90 and rear is +/-180.  The historical v4 model has
only 180 output bins and was trained with broadside at 90 degrees.  Its label
space therefore cannot represent AVEngine's front/back distinction.

The compatibility helpers keep that loss explicit by folding onto [0, 180].
The retraining path instead uses the native 360-degree labels and circular
targets defined in this module.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


class LegacyV4AudioError(ValueError):
    """A v4 compatibility input violates the explicit adapter contract."""


def _listener_local_azimuth_deg(
    source_position_m: Any,
    listener_position_m: Any,
    listener_orientation_wxyz: Any,
) -> float:
    """Mirror AVEngine's public listener convention without importing a backend."""

    try:
        source = np.asarray(source_position_m, dtype=np.float64)
        listener = np.asarray(listener_position_m, dtype=np.float64)
        quaternion = np.asarray(listener_orientation_wxyz, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LegacyV4AudioError("listener pose must be finite numeric data") from exc
    if (
        source.shape != (3,)
        or listener.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(listener))
        or not np.all(np.isfinite(quaternion))
    ):
        raise LegacyV4AudioError("listener pose must contain finite xyz/wxyz data")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise LegacyV4AudioError("listener orientation must be unit normalized")
    world_direction = source - listener
    if float(np.linalg.norm(world_direction)) <= 0.0:
        raise LegacyV4AudioError("source and listener positions must differ")
    w = float(quaternion[0])
    inverse_vector = -quaternion[1:]
    uv = np.cross(inverse_vector, world_direction)
    uuv = np.cross(inverse_vector, uv)
    local = world_direction + 2.0 * (w * uv + uuv)
    azimuth = math.degrees(math.atan2(float(local[0]), -float(local[2])))
    if math.isclose(azimuth, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        return 0.0
    if azimuth > 180.0:
        azimuth -= 360.0
    return float(azimuth)


def native_to_legacy_azimuth_deg(native_azimuth_deg: Any) -> np.ndarray:
    """Rotate and fold AVEngine azimuth into the legacy 180-degree label space.

    The mapping is:

    * AVEngine front (0) -> legacy broadside (90)
    * AVEngine right (+90) -> legacy right endfire (0)
    * AVEngine left (-90) -> legacy left endfire (180)
    * AVEngine rear (+/-180) -> legacy broadside (90)
    """

    try:
        native = np.asarray(native_azimuth_deg, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LegacyV4AudioError("native azimuth must be finite numeric data") from exc
    if native.size == 0 or not np.all(np.isfinite(native)):
        raise LegacyV4AudioError("native azimuth must be finite and non-empty")
    wrapped = np.mod(90.0 - native, 360.0)
    folded = np.where(wrapped <= 180.0, wrapped, 360.0 - wrapped)
    return folded.astype(np.float64, copy=False)


def native_azimuth_to_bin360(native_azimuth_deg: Any) -> np.ndarray:
    """Return AVEngine azimuth in the 360-bin [0, 360) representation."""

    try:
        native = np.asarray(native_azimuth_deg, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LegacyV4AudioError("native azimuth must be finite numeric data") from exc
    if native.size == 0 or not np.all(np.isfinite(native)):
        raise LegacyV4AudioError("native azimuth must be finite and non-empty")
    return np.mod(native, 360.0).astype(np.float64, copy=False)


def legacy_rows_for_native_360_bins() -> np.ndarray:
    """Map each native 360-degree bin to its pretrained legacy-head row.

    The old head has no distinct row for 180 degrees, so the single native
    bin that folds exactly to legacy 180 is conservatively clamped to row 179.
    Copying rows with this map produces an explicitly front/back-symmetric
    initialization.  Subsequent 360-degree training can break that symmetry.
    """

    native_bins = np.arange(360, dtype=np.float64)
    folded = native_to_legacy_azimuth_deg(native_bins)
    return np.clip(np.rint(folded), 0, 179).astype(np.int64)


def circular_gaussian_targets(
    native_azimuth_deg: Any,
    *,
    sigma_deg: float = 2.0,
) -> np.ndarray:
    """Build normalized circular Gaussian targets with shape [..., 360]."""

    centers = native_azimuth_to_bin360(native_azimuth_deg)
    if not math.isfinite(sigma_deg) or sigma_deg <= 0.0:
        raise LegacyV4AudioError("sigma_deg must be finite and positive")
    bins = np.arange(360, dtype=np.float64)
    delta = np.abs(centers[..., None] - bins)
    distance = np.minimum(delta, 360.0 - delta)
    targets = np.exp(-0.5 * np.square(distance / float(sigma_deg)))
    maximum = np.max(targets, axis=-1, keepdims=True)
    return (targets / maximum).astype(np.float32)


def resample_position_track(
    positions_m: Any,
    *,
    source_frame_rate_hz: float,
    target_duration_seconds: float,
    target_frame_count: int,
) -> np.ndarray:
    """Linearly sample a source-position track over a deterministic time crop."""

    try:
        positions = np.asarray(positions_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LegacyV4AudioError("positions_m must be finite [frames, 3] data") from exc
    if (
        positions.ndim != 2
        or positions.shape[0] < 2
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise LegacyV4AudioError("positions_m must be finite [frames, 3] data")
    if (
        not np.isfinite(source_frame_rate_hz)
        or source_frame_rate_hz <= 0
        or not np.isfinite(target_duration_seconds)
        or target_duration_seconds <= 0
        or isinstance(target_frame_count, bool)
        or not isinstance(target_frame_count, int)
        or target_frame_count <= 0
    ):
        raise LegacyV4AudioError("frame rate, duration and frame count must be positive")

    source_times = np.arange(positions.shape[0], dtype=np.float64)
    source_times /= float(source_frame_rate_hz)
    target_times = (
        np.arange(target_frame_count, dtype=np.float64)
        * float(target_duration_seconds)
        / float(target_frame_count)
    )
    if target_times[-1] > source_times[-1] + 1.0e-12:
        raise LegacyV4AudioError("the requested crop extends beyond the position track")
    return np.column_stack(
        [
            np.interp(target_times, source_times, positions[:, axis])
            for axis in range(3)
        ]
    )


def label_tracks_for_source(
    positions_m: Any,
    *,
    source_frame_rate_hz: float,
    target_duration_seconds: float,
    target_frame_count: int,
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
) -> dict[str, list[float]]:
    """Return native full-circle and legacy folded label tracks."""

    sampled_positions = resample_position_track(
        positions_m,
        source_frame_rate_hz=source_frame_rate_hz,
        target_duration_seconds=target_duration_seconds,
        target_frame_count=target_frame_count,
    )
    native = np.asarray(
        [
            _listener_local_azimuth_deg(
                position,
                listener_position_m,
                listener_orientation_wxyz,
            )
            for position in sampled_positions
        ],
        dtype=np.float64,
    )
    legacy = native_to_legacy_azimuth_deg(native)
    return {
        "native_360_azimuth_deg": native.tolist(),
        "legacy_folded_180_azimuth_deg": legacy.tolist(),
    }


def deterministic_split_samples(
    dataset_index: Mapping[str, Any],
    *,
    split: str,
    offset: int,
    limit: int,
) -> list[Mapping[str, Any]]:
    """Select a stable slice without invoking the legacy random-choice loader."""

    if dataset_index.get("status") != "pass":
        raise LegacyV4AudioError("dataset index status is not pass")
    values = dataset_index.get("samples")
    if not isinstance(values, list):
        raise LegacyV4AudioError("dataset index lacks a samples list")
    if (
        not isinstance(split, str)
        or not split
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
    ):
        raise LegacyV4AudioError("split, offset and limit are invalid")
    selected = [
        value
        for value in values
        if isinstance(value, Mapping) and value.get("split") == split
    ]
    result = selected[offset : offset + limit]
    if not result:
        raise LegacyV4AudioError("the requested deterministic sample slice is empty")
    return result


def caption_for_asset(asset_id: str) -> str:
    """Map the current generated-asset families to the v4 text cue classes."""

    normalized = asset_id.casefold()
    if "human" in normalized:
        return "human speech"
    if "cat" in normalized or "abyssinian" in normalized:
        return "cat meowing"
    dog_tokens = ("dog", "collie", "beagle", "labrador", "retriever")
    if any(token in normalized for token in dog_tokens):
        return "dog barking"
    raise LegacyV4AudioError(f"no legacy v4 caption mapping for asset: {asset_id}")
