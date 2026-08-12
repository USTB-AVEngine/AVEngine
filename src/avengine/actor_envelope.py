"""Planning-only sampled actor envelopes derived from strict CPU skinning.

An envelope carries explicit source/action authority, but remains
formal-ineligible until it is bound to live runtime readbacks and the
downstream visual review gate.  World bounds are always computed by
transforming all eight local AABB corners; rotating only min/max is forbidden.

The action envelope is the union at a deterministic rate *and* every authored
channel timestamp.  It deliberately makes no continuous-containment claim for
the intervals between those samples: skinned compositions containing rotation
cannot in general be bounded by endpoint extrema alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from avengine.m2.skinning import (
    CompiledSkinning,
    action_time_bounds,
    sample_action_vertices,
)


ACTOR_ENVELOPE_SCHEMA = "avengine_actor_action_envelope_v1"
WORLD_AABB_SCHEMA = "avengine_actor_world_aabb_v1"
_ALGORITHM_REVISION = 2
_FORMAL_INELIGIBILITY_REASONS = (
    "sampled planning envelope is not a live renderer bounds readback",
    "discrete samples do not prove continuous containment between samples",
    "human visual review is not bound to this envelope",
)


class ActorEnvelopeError(ValueError):
    """An envelope request or transform violates the strict contract."""


def _canonical_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _finite_vec3(value: Sequence[float], *, owner: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ActorEnvelopeError(f"{owner} must contain three finite numbers")
    return tuple(_canonical_float(component) for component in array)  # type: ignore[return-value]


@dataclass(frozen=True)
class AxisAlignedBounds:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = _finite_vec3(self.minimum, owner="minimum")
        maximum = _finite_vec3(self.maximum, owner="maximum")
        if any(low > high for low, high in zip(minimum, maximum, strict=True)):
            raise ActorEnvelopeError("minimum must be component-wise <= maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def to_record(self) -> dict[str, list[float]]:
        return {"minimum": list(self.minimum), "maximum": list(self.maximum)}


@dataclass(frozen=True)
class SourceAssetAuthority:
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ActorEnvelopeError("source authority path must be non-empty")

    def to_record(self) -> dict[str, Any]:
        return {"path": self.path}


@dataclass(frozen=True)
class ActorActionEnvelope:
    source_asset: SourceAssetAuthority
    skin_index: int
    action_name: str
    sample_rate_hz: float
    sample_times_seconds: tuple[float, ...]
    padding_m: float
    sampled_bounds: AxisAlignedBounds
    padded_bounds: AxisAlignedBounds
    schema: str = field(default=ACTOR_ENVELOPE_SCHEMA, init=False)
    algorithm_revision: int = field(default=_ALGORITHM_REVISION, init=False)
    qualification_state: str = field(default="planning_only", init=False)
    qualification_claim: bool = field(default=False, init=False)
    formal_eligible: bool = field(default=False, init=False)
    formal_ineligibility_reasons: tuple[str, ...] = field(
        default=_FORMAL_INELIGIBILITY_REASONS, init=False
    )

    def authority_record(self) -> dict[str, Any]:
        return {
            "schema": "avengine_actor_envelope_authority_v1",
            "source_asset": self.source_asset.to_record(),
            "skinning": {
                "schema": "avengine_compiled_actor_skinning_v1",
                "skin_index": self.skin_index,
            },
            "action": {
                "name": self.action_name,
            },
            "sampling": {
                "algorithm_revision": self.algorithm_revision,
                "coverage_kind": "authored_keys_plus_fixed_rate_samples",
                "continuous_conservative_claim": False,
                "sample_rate_hz": self.sample_rate_hz,
                "sample_count": len(self.sample_times_seconds),
                "sample_times_seconds": list(self.sample_times_seconds),
                "padding_m": self.padding_m,
            },
        }

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority": self.authority_record(),
            "bounds": {
                "sampled": self.sampled_bounds.to_record(),
                "padded": self.padded_bounds.to_record(),
            },
            "qualification": {
                "state": self.qualification_state,
                "claim": self.qualification_claim,
                "formal_eligible": self.formal_eligible,
                "formal_ineligibility_reasons": list(self.formal_ineligibility_reasons),
            },
        }


@dataclass(frozen=True)
class WorldAabb:
    local_bounds: AxisAlignedBounds
    actor_from_asset: tuple[tuple[float, float, float, float], ...]
    transformed_corners: tuple[tuple[float, float, float], ...]
    bounds: AxisAlignedBounds
    schema: str = field(default=WORLD_AABB_SCHEMA, init=False)
    qualification_state: str = field(default="planning_only", init=False)
    qualification_claim: bool = field(default=False, init=False)
    formal_eligible: bool = field(default=False, init=False)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "local_bounds": self.local_bounds.to_record(),
            "actor_from_asset": [list(row) for row in self.actor_from_asset],
            "transformed_corners": [
                list(corner) for corner in self.transformed_corners
            ],
            "world_bounds": self.bounds.to_record(),
            "qualification": {
                "state": self.qualification_state,
                "claim": self.qualification_claim,
                "formal_eligible": self.formal_eligible,
            },
        }


def _bounds_from_points(points: np.ndarray, *, owner: str) -> AxisAlignedBounds:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ActorEnvelopeError(f"{owner} must be a non-empty Nx3 array")
    if not np.all(np.isfinite(array)):
        raise ActorEnvelopeError(f"{owner} contains non-finite coordinates")
    return AxisAlignedBounds(
        minimum=tuple(np.min(array, axis=0)),  # type: ignore[arg-type]
        maximum=tuple(np.max(array, axis=0)),  # type: ignore[arg-type]
    )


def sample_action_aabb(
    compiled: CompiledSkinning, action_name: str, time_seconds: float
) -> AxisAlignedBounds:
    return _bounds_from_points(
        sample_action_vertices(compiled, action_name, time_seconds),
        owner="sampled actor vertices",
    )


def union_bounds(bounds: Iterable[AxisAlignedBounds]) -> AxisAlignedBounds:
    materialized = tuple(bounds)
    if not materialized:
        raise ActorEnvelopeError("cannot union an empty bounds collection")
    return AxisAlignedBounds(
        minimum=tuple(
            min(item.minimum[axis] for item in materialized) for axis in range(3)
        ),
        maximum=tuple(
            max(item.maximum[axis] for item in materialized) for axis in range(3)
        ),
    )


def union_action_aabb(
    compiled: CompiledSkinning,
    action_name: str,
    sample_times_seconds: Sequence[float],
) -> AxisAlignedBounds:
    if not sample_times_seconds:
        raise ActorEnvelopeError("sample_times_seconds cannot be empty")
    times = tuple(float(value) for value in sample_times_seconds)
    if not all(math.isfinite(value) for value in times):
        raise ActorEnvelopeError("sample_times_seconds must be finite")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ActorEnvelopeError("sample_times_seconds must be strictly increasing")
    return union_bounds(
        sample_action_aabb(compiled, action_name, time) for time in times
    )


def action_sample_times(
    compiled: CompiledSkinning, action_name: str, *, sample_rate_hz: float = 120.0
) -> tuple[float, ...]:
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, (int, float))
        or not math.isfinite(float(sample_rate_hz))
        or float(sample_rate_hz) <= 0.0
        or float(sample_rate_hz) > 10_000.0
    ):
        raise ActorEnvelopeError("sample_rate_hz must be finite and in (0, 10000]")
    rate = float(sample_rate_hz)
    start, end = action_time_bounds(compiled, action_name)
    regular_times: list[float]
    if end == start:
        regular_times = [start]
    else:
        interval_count = max(1, int(math.ceil((end - start) * rate - 1.0e-12)))
        regular_times = [
            start + ordinal / rate for ordinal in range(interval_count + 1)
        ]
        regular_times = [value for value in regular_times if value < end]
        regular_times.append(end)

    # A fixed-rate grid alone can miss short STEP states or extrema authored
    # between grid points.  Every channel key is therefore a mandatory sample.
    authored_times = [
        float(timestamp)
        for channel in compiled.action(action_name).channels
        for timestamp in channel.timestamps_seconds
    ]
    return tuple(
        sorted({_canonical_float(value) for value in (*regular_times, *authored_times)})
    )


def _padded(bounds: AxisAlignedBounds, padding_m: float) -> AxisAlignedBounds:
    return AxisAlignedBounds(
        minimum=tuple(value - padding_m for value in bounds.minimum),
        maximum=tuple(value + padding_m for value in bounds.maximum),
    )


def build_action_envelope(
    compiled: CompiledSkinning,
    action_name: str,
    *,
    sample_rate_hz: float = 120.0,
    padding_m: float = 0.02,
    source_asset_path: str | Path | None = None,
) -> ActorActionEnvelope:
    """Build a sampled planning envelope for one source action."""

    if (
        isinstance(padding_m, bool)
        or not isinstance(padding_m, (int, float))
        or not math.isfinite(float(padding_m))
        or float(padding_m) < 0.0
    ):
        raise ActorEnvelopeError("padding_m must be a finite non-negative number")
    padding = _canonical_float(float(padding_m))
    path_value = (
        source_asset_path if source_asset_path is not None else compiled.source_path
    )
    if path_value is None:
        raise ActorEnvelopeError(
            "source_asset_path is required when the compiled document has no source path"
        )
    try:
        source_path = Path(path_value).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ActorEnvelopeError(
            f"source asset does not exist: {path_value}"
        ) from error
    if not source_path.is_file():
        raise ActorEnvelopeError(f"source asset is not a regular file: {source_path}")
    path = str(source_path)
    source = SourceAssetAuthority(path=path)
    times = action_sample_times(compiled, action_name, sample_rate_hz=sample_rate_hz)
    sampled = union_action_aabb(compiled, action_name, times)
    padded = _padded(sampled, padding)
    return ActorActionEnvelope(
        source_asset=source,
        skin_index=compiled.skin_index,
        action_name=action_name,
        sample_rate_hz=_canonical_float(float(sample_rate_hz)),
        sample_times_seconds=times,
        padding_m=padding,
        sampled_bounds=sampled,
        padded_bounds=padded,
    )


def aabb_corners(bounds: AxisAlignedBounds) -> tuple[tuple[float, float, float], ...]:
    """Return all eight corners in deterministic x-major product order."""

    return tuple(
        (float(x), float(y), float(z))
        for x, y, z in itertools.product(
            (bounds.minimum[0], bounds.maximum[0]),
            (bounds.minimum[1], bounds.maximum[1]),
            (bounds.minimum[2], bounds.maximum[2]),
        )
    )


def materialize_world_aabb(
    local_bounds: AxisAlignedBounds,
    actor_from_asset: Sequence[Sequence[float]],
) -> WorldAabb:
    """Transform all eight local corners and re-bound them in world/actor space."""

    matrix = np.asarray(actor_from_asset, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ActorEnvelopeError("actor_from_asset must be a finite 4x4 matrix")
    if not np.allclose(
        matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1.0e-12
    ):
        raise ActorEnvelopeError(
            "actor_from_asset must be affine with [0,0,0,1] last row"
        )
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1.0e-12:
        raise ActorEnvelopeError(
            "actor_from_asset linear transform must be nonsingular"
        )
    local_corners = np.asarray(aabb_corners(local_bounds), dtype=np.float64)
    homogeneous = np.concatenate(
        [local_corners, np.ones((8, 1), dtype=np.float64)], axis=1
    )
    transformed = (matrix @ homogeneous.T).T[:, :3]
    corners = tuple(
        tuple(_canonical_float(component) for component in row)  # type: ignore[misc]
        for row in transformed
    )
    bounds = _bounds_from_points(transformed, owner="transformed AABB corners")
    matrix_tuple = tuple(
        tuple(_canonical_float(component) for component in row) for row in matrix
    )
    return WorldAabb(
        local_bounds=local_bounds,
        actor_from_asset=matrix_tuple,
        transformed_corners=corners,
        bounds=bounds,
    )
