"""Room-agnostic deterministic camera-position and yaw enumeration.

This module owns only camera-center placement.  It deliberately does not know
about semantic targets, actors, source paths, pair feasibility, visibility,
or diversity ranking; those are downstream joins over the returned candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Iterator, Sequence

import numpy as np

from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.room_feasibility import (
    RoomFeasibilityCompiler,
    RoomFeasibilityError,
)


DEFAULT_YAW_DEGREES = tuple(float(value) for value in range(0, 360, 30))


class CameraPlacementError(ValueError):
    """Camera placement inputs cannot be enumerated safely."""


@dataclass(frozen=True)
class CameraPlacement:
    """One camera center and yaw that passed only placement feasibility."""

    height_id: str
    position_m: tuple[float, float, float]
    yaw_deg: float
    raster_pixel_rc: tuple[int, int]

    @property
    def stable_key(self) -> tuple[object, ...]:
        return (
            self.position_m,
            self.yaw_deg,
            self.height_id.encode("utf-8"),
            self.raster_pixel_rc,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "height_id": self.height_id,
            "position_m": list(self.position_m),
            "yaw_deg": self.yaw_deg,
            "raster_pixel_rc": list(self.raster_pixel_rc),
            "placement_status": "camera_center_feasible_only",
            "claim_boundary": (
                "camera center, height, floor mask, navmesh and rigid clearance "
                "only; no target, source/path, pairing, visibility or ranking claim"
            ),
        }


def _finite_number(value: object, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CameraPlacementError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CameraPlacementError(f"{owner} must be a finite number")
    return result


def _nonnegative_number(value: object, *, owner: str) -> float:
    result = _finite_number(value, owner=owner)
    if result < 0.0:
        raise CameraPlacementError(f"{owner} must be nonnegative")
    return result


def _positive_number(value: object, *, owner: str) -> float:
    result = _finite_number(value, owner=owner)
    if result <= 0.0:
        raise CameraPlacementError(f"{owner} must be positive")
    return result


def _height_records(
    value: Sequence[tuple[str, Real]],
) -> tuple[tuple[str, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise CameraPlacementError(
            "camera_center_heights_m must be a nonempty sequence"
        )
    result = []
    seen = set()
    for index, record in enumerate(value):
        if (
            not isinstance(record, Sequence)
            or isinstance(record, (str, bytes))
            or len(record) != 2
        ):
            raise CameraPlacementError(
                f"camera_center_heights_m[{index}] must contain id and height"
            )
        height_id, raw_height = record
        if not isinstance(height_id, str) or not height_id.strip():
            raise CameraPlacementError(
                f"camera_center_heights_m[{index}] id must be nonempty"
            )
        if height_id in seen:
            raise CameraPlacementError("camera height ids must be unique")
        seen.add(height_id)
        result.append(
            (
                height_id,
                _finite_number(
                    raw_height,
                    owner=f"camera_center_heights_m[{index}] height",
                ),
            )
        )
    return tuple(sorted(result, key=lambda item: (item[1], item[0].encode("utf-8"))))


def _yaw_values(value: Iterable[Real]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise CameraPlacementError("yaw_degrees must be an iterable of numbers")
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise CameraPlacementError(
            "yaw_degrees must be an iterable of numbers"
        ) from exc
    if not raw:
        raise CameraPlacementError("yaw_degrees must be nonempty")
    normalized = tuple(
        _finite_number(item, owner=f"yaw_degrees[{index}]") % 360.0
        for index, item in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise CameraPlacementError(
            "yaw_degrees contains duplicates after wrapping to [0, 360)"
        )
    return tuple(sorted(normalized))


def _floor_mask(
    value: np.ndarray | None, obstacle_map: RuntimeObstacleMap
) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value, dtype=np.bool_)
    navmesh = np.asarray(obstacle_map.binary_navmesh, dtype=np.bool_)
    if mask.shape != navmesh.shape:
        raise CameraPlacementError(
            "floor_feasible_mask shape differs from the room navmesh"
        )
    if np.any(mask & ~navmesh):
        raise CameraPlacementError(
            "floor_feasible_mask must be a subset of the room navmesh"
        )
    return mask


def iter_feasible_camera_placements(
    obstacle_map: RuntimeObstacleMap,
    *,
    camera_center_heights_m: Sequence[tuple[str, Real]],
    minimum_navmesh_clearance_m: Real = 0.02,
    minimum_rigid_clearance_m: Real = 0.0,
    sample_spacing_m: Real = 0.75,
    floor_feasible_mask: np.ndarray | None = None,
    yaw_degrees: Iterable[Real] = DEFAULT_YAW_DEGREES,
) -> Iterator[CameraPlacement]:
    """Yield stable legal centers crossed with explicit yaw values.

    The caller supplies any live intended-floor/island mask.  This pure layer
    intersects it with height-specific camera-center feasibility and never
    imports a Simulator or PathFinder.
    """

    if not isinstance(obstacle_map, RuntimeObstacleMap):
        raise CameraPlacementError("obstacle_map must be a RuntimeObstacleMap")
    heights = _height_records(camera_center_heights_m)
    yaws = _yaw_values(yaw_degrees)
    nav_clearance = _nonnegative_number(
        minimum_navmesh_clearance_m, owner="minimum navmesh clearance"
    )
    rigid_clearance = _nonnegative_number(
        minimum_rigid_clearance_m, owner="minimum rigid clearance"
    )
    spacing = _positive_number(sample_spacing_m, owner="sample spacing")
    intended_floor = _floor_mask(floor_feasible_mask, obstacle_map)
    compiler = RoomFeasibilityCompiler(obstacle_map)
    for height_id, height in heights:
        try:
            region = compiler.compile(
                source_center_height_m=height,
                minimum_navmesh_clearance_m=nav_clearance,
                minimum_rigid_clearance_m=rigid_clearance,
                sample_spacing_m=spacing,
            )
        except RoomFeasibilityError as exc:
            raise CameraPlacementError(
                f"camera height {height_id!r} has no feasible region: {exc}"
            ) from exc
        pixels = tuple(
            (int(pixel[0]), int(pixel[1]))
            for pixel in np.asarray(region.sample_pixels_rc, dtype=np.int64)
            if intended_floor is None
            or bool(intended_floor[int(pixel[0]), int(pixel[1])])
        )
        for pixel in sorted(pixels):
            position = tuple(float(item) for item in region.pixel_to_world(pixel))
            for yaw in yaws:
                yield CameraPlacement(
                    height_id=height_id,
                    position_m=position,
                    yaw_deg=yaw,
                    raster_pixel_rc=pixel,
                )


__all__ = [
    "DEFAULT_YAW_DEGREES",
    "CameraPlacement",
    "CameraPlacementError",
    "iter_feasible_camera_placements",
]
