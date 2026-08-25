"""Derived M2 RGB + top-down review media.

This module is intentionally outside the formal M2 capture path.  It consumes
already captured RGB frames and recorded actor states, draws a synchronized
QA panel from a Habitat navmesh plus optional scene-descriptor object OBBs,
and writes a separate review-only MP4.  The panel is not a sensor, does not
create a second view or run an object detector, and cannot make an asset
qualification claim.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from avengine.contracts.json_io import (
    canonical_json_bytes,
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
)


EVIDENCE_SCHEMA = "avengine_m2_topdown_review_evidence_v1"
RGB_STACK_HASH_ALGORITHM = "avengine_review_rgb_stack_sha256_v1"
COMPOSITE_RGB_STACK_HASH_ALGORITHM = "avengine_review_composite_rgb_sha256_v1"
NAVMESH_HASH_ALGORITHM = "avengine_navmesh_binary_sha256_v1"
SEMANTIC_OBJECT_FOOTPRINT_POLICY = (
    "habitat_semantic_scene_obb_local_to_world_cube_corners_xz_v1"
)
SEMANTIC_OBJECT_SOURCE = "habitat_sim.semantic_scene.objects[].obb.local_to_world"
SEMANTIC_OBJECT_MAX_FOOTPRINT_AREA_M2 = 16.0
SEMANTIC_OBJECT_MAX_FOOTPRINT_SPAN_M = 6.0
CAMERA_HEADING_POLICY = "world_from_rig_local_negative_z_v1"
ACTOR_HEADING_TRAJECTORY_POLICY = "trajectory_tangent_nearest_nonzero_v1"
ACTOR_HEADING_TRUSTED_AXIS_POLICY = "trusted_actor_local_forward_axis_v1"
ACTOR_IDLE_HEADING_POLICY = "nearest_nonzero_trajectory_tangent"
ACTOR_MOVEMENT_EPSILON_M = 1.0e-6
VIDEO_READBACK_MAXIMUM_FRAME_MAE = 8.0
VIDEO_READBACK_MAXIMUM_RMSE = 15.0

_STRUCTURAL_SEMANTIC_CATEGORIES = frozenset(
    {
        "",
        "beam",
        "ceiling",
        "column",
        "door",
        "floor",
        "lighting",
        "misc",
        "objects",
        "railing",
        "stairs",
        "unknown",
        "void",
        "wall",
        "window",
    }
)


class TopdownReviewError(RuntimeError):
    """A derived top-down review artifact could not be produced safely."""


ReviewVideoEncoder = Callable[..., int]


@dataclass(frozen=True)
class _PanelProjection:
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    navmesh_shape_hw: tuple[int, int]
    viewport_xyxy: tuple[int, int, int, int]
    navmesh_roi_rc: tuple[int, int, int, int]

    def world_to_panel(self, world_position: Sequence[float]) -> tuple[float, float]:
        nav_x, nav_y = habitat_xz_to_navmesh_pixel(
            world_position,
            navmesh_shape_hw=self.navmesh_shape_hw,
            bounds=(self.bounds_min_xyz, self.bounds_max_xyz),
        )
        nav_height, nav_width = self.navmesh_shape_hw
        left, top, right, bottom = self.viewport_xyxy
        width = max(right - left, 1)
        height = max(bottom - top, 1)
        x_ratio = nav_x / max(nav_width - 1, 1)
        y_ratio = nav_y / max(nav_height - 1, 1)
        return (
            float(left) + x_ratio * float(width - 1),
            float(top) + y_ratio * float(height - 1),
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TopdownReviewError("review metadata keys must be strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TopdownReviewError(
        f"review metadata contains unsupported value {type(value).__name__}"
    )


def _positive_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool):
        raise TopdownReviewError(f"{owner} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TopdownReviewError(f"{owner} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise TopdownReviewError(f"{owner} must be a positive integer")
    return result


def _finite_float(value: Any, *, owner: str, minimum: float, inclusive: bool) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TopdownReviewError(f"{owner} must be finite") from exc
    valid_minimum = result >= minimum if inclusive else result > minimum
    if not math.isfinite(result) or not valid_minimum:
        comparison = "at least" if inclusive else "greater than"
        raise TopdownReviewError(f"{owner} must be {comparison} {minimum}")
    return result


def _canonical_hash(value: Any, *, owner: str) -> str:
    try:
        return canonical_json_sha256(_jsonable(value))
    except (TypeError, ValueError) as exc:
        raise TopdownReviewError(f"{owner} is not finite canonical JSON") from exc


def _coerce_bounds(
    bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if isinstance(bounds, Mapping):
        minimum = bounds.get("minimum", bounds.get("lower"))
        maximum = bounds.get("maximum", bounds.get("upper"))
        if minimum is None or maximum is None:
            raise TopdownReviewError(
                "navmesh bounds mapping requires minimum/maximum or lower/upper"
            )
    else:
        if len(bounds) != 2:
            raise TopdownReviewError("navmesh bounds must contain minimum and maximum")
        minimum, maximum = bounds
    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,):
        raise TopdownReviewError("navmesh bounds must be two XYZ vectors")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise TopdownReviewError("navmesh bounds must be finite")
    if high[0] <= low[0] or high[2] <= low[2]:
        raise TopdownReviewError("navmesh X/Z bounds must have positive spans")
    return (
        tuple(float(component) for component in low),
        tuple(float(component) for component in high),
    )


def _world_xz(world_position: Sequence[float]) -> tuple[float, float]:
    value = np.asarray(world_position, dtype=np.float64)
    if value.shape == (2,):
        x, z = value
    elif value.shape == (3,):
        x, z = value[0], value[2]
    else:
        raise TopdownReviewError("world position must be XZ or XYZ")
    if not np.isfinite(x) or not np.isfinite(z):
        raise TopdownReviewError("world position must be finite")
    return float(x), float(z)


def habitat_xz_to_navmesh_pixel(
    world_position: Sequence[float],
    *,
    navmesh_shape_hw: Sequence[int],
    bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
) -> tuple[float, float]:
    """Map Habitat world X/Z to the raw ``get_topdown_view`` image.

    Habitat's binary top-down grid stores increasing Z along rows and
    increasing X along columns.  This function returns ``(column, row)`` and
    maps the closed X/Z bounds to the visible pixel extrema.  Values outside
    the bounds remain outside the image instead of being silently clipped.
    """

    shape = tuple(int(component) for component in navmesh_shape_hw)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise TopdownReviewError("navmesh shape must be positive HxW")
    low, high = _coerce_bounds(bounds)
    x, z = _world_xz(world_position)
    x_ratio = (x - low[0]) / (high[0] - low[0])
    z_ratio = (z - low[2]) / (high[2] - low[2])
    return x_ratio * float(shape[1] - 1), z_ratio * float(shape[0] - 1)


def _semantic_category_key(value: str) -> str:
    return "_".join(value.strip().casefold().replace("-", " ").split())


def _convex_hull_xz(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        raise TopdownReviewError("semantic object footprint is degenerate")

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if hull.shape[0] < 3:
        raise TopdownReviewError("semantic object footprint is degenerate")
    return hull


def _polygon_area_xz(polygon: np.ndarray) -> float:
    x = polygon[:, 0]
    z = polygon[:, 1]
    return abs(float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))) * 0.5


def semantic_object_footprint_from_obb(
    *,
    object_id: str,
    category: str,
    local_to_world: Any,
    source: str = SEMANTIC_OBJECT_SOURCE,
    maximum_area_m2: float = SEMANTIC_OBJECT_MAX_FOOTPRINT_AREA_M2,
    maximum_span_m: float = SEMANTIC_OBJECT_MAX_FOOTPRINT_SPAN_M,
) -> dict[str, Any] | None:
    """Build one descriptor-derived world-XZ footprint from a Habitat OBB.

    Habitat's MP3D binding stores the OBB half extents in ``local_to_world``.
    Transforming the eight corners of a ``[-1, 1]^3`` cube is therefore the
    authoritative route.  ``obb.aabb``/``obb.to_aabb()`` are deliberately not
    used because this pinned binding can incorrectly include the world origin.
    Structural/unlabelled and implausibly large footprints are excluded.
    """

    if not isinstance(object_id, str) or not object_id:
        raise TopdownReviewError("semantic object id must be a non-empty string")
    if not isinstance(category, str):
        raise TopdownReviewError("semantic object category must be a string")
    if not isinstance(source, str) or not source:
        raise TopdownReviewError("semantic object source must be a non-empty string")
    category_key = _semantic_category_key(category)
    if category_key in _STRUCTURAL_SEMANTIC_CATEGORIES:
        return None
    maximum_area = _finite_float(
        maximum_area_m2,
        owner="semantic object maximum footprint area",
        minimum=0.0,
        inclusive=False,
    )
    maximum_span = _finite_float(
        maximum_span_m,
        owner="semantic object maximum footprint span",
        minimum=0.0,
        inclusive=False,
    )
    matrix = _matrix_from_transform(local_to_world, owner="semantic object OBB")
    local_corners = np.asarray(
        [[x, y, z, 1.0] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float64,
    )
    world = (matrix @ local_corners.T).T
    if not np.all(np.isfinite(world)) or not np.allclose(world[:, 3], 1.0):
        raise TopdownReviewError("semantic object OBB produced invalid world corners")
    try:
        polygon = _convex_hull_xz(world[:, [0, 2]])
    except TopdownReviewError:
        return None
    area = _polygon_area_xz(polygon)
    minimum = np.min(polygon, axis=0)
    maximum = np.max(polygon, axis=0)
    spans = maximum - minimum
    if area <= 1.0e-6 or area > maximum_area or float(np.max(spans)) > maximum_span:
        return None
    return {
        "object_id": object_id,
        "category": category_key,
        "source": source,
        "polygon_xz_m": polygon.tolist(),
        "bounds_min_xz_m": minimum.tolist(),
        "bounds_max_xz_m": maximum.tolist(),
        "footprint_area_m2": area,
    }


def _coerce_semantic_object_footprints(
    values: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    footprints: list[dict[str, Any]] = []
    expected_fields = {
        "object_id",
        "category",
        "source",
        "polygon_xz_m",
        "bounds_min_xz_m",
        "bounds_max_xz_m",
        "footprint_area_m2",
    }
    for index, raw in enumerate(values or ()):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise TopdownReviewError(
                f"semantic object footprint {index} fields are invalid"
            )
        object_id = raw.get("object_id")
        category = raw.get("category")
        source = raw.get("source")
        if not isinstance(object_id, str) or not object_id:
            raise TopdownReviewError(f"semantic object footprint {index} id is invalid")
        if not isinstance(category, str) or (
            _semantic_category_key(category) in _STRUCTURAL_SEMANTIC_CATEGORIES
        ):
            raise TopdownReviewError(
                f"semantic object footprint {index} category is invalid"
            )
        if source != SEMANTIC_OBJECT_SOURCE:
            raise TopdownReviewError(
                f"semantic object footprint {index} source is invalid"
            )
        polygon = np.asarray(raw.get("polygon_xz_m"), dtype=np.float64)
        if (
            polygon.ndim != 2
            or polygon.shape[0] < 3
            or polygon.shape[1] != 2
            or not np.all(np.isfinite(polygon))
        ):
            raise TopdownReviewError(
                f"semantic object footprint {index} polygon is invalid"
            )
        hull = _convex_hull_xz(polygon)
        if hull.shape != polygon.shape or not np.allclose(hull, polygon, atol=1.0e-9):
            raise TopdownReviewError(
                f"semantic object footprint {index} polygon is not canonical"
            )
        minimum = np.min(polygon, axis=0)
        maximum = np.max(polygon, axis=0)
        area = _polygon_area_xz(polygon)
        declared_minimum = np.asarray(raw.get("bounds_min_xz_m"), dtype=np.float64)
        declared_maximum = np.asarray(raw.get("bounds_max_xz_m"), dtype=np.float64)
        declared_area = _finite_float(
            raw.get("footprint_area_m2"),
            owner=f"semantic object footprint {index} area",
            minimum=0.0,
            inclusive=False,
        )
        if (
            declared_minimum.shape != (2,)
            or declared_maximum.shape != (2,)
            or not np.all(np.isfinite(declared_minimum))
            or not np.all(np.isfinite(declared_maximum))
            or not np.allclose(declared_minimum, minimum, atol=1.0e-9)
            or not np.allclose(declared_maximum, maximum, atol=1.0e-9)
            or not math.isclose(declared_area, area, abs_tol=1.0e-9)
            or area > SEMANTIC_OBJECT_MAX_FOOTPRINT_AREA_M2
            or float(np.max(maximum - minimum)) > SEMANTIC_OBJECT_MAX_FOOTPRINT_SPAN_M
        ):
            raise TopdownReviewError(
                f"semantic object footprint {index} derived geometry differs"
            )
        footprints.append(_jsonable(raw))
    return tuple(footprints)


def _matrix_from_transform(value: Any, *, owner: str) -> np.ndarray:
    if isinstance(value, Mapping):
        if "matrix" in value:
            return _matrix_from_transform(value["matrix"], owner=owner)
        if "translation_m" not in value or "rotation_xyzw" not in value:
            raise TopdownReviewError(
                f"{owner} requires translation_m and rotation_xyzw"
            )
        translation = np.asarray(value["translation_m"], dtype=np.float64)
        quaternion = np.asarray(value["rotation_xyzw"], dtype=np.float64)
        if translation.shape != (3,) or quaternion.shape != (4,):
            raise TopdownReviewError(f"{owner} transform has invalid vector shapes")
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
            raise TopdownReviewError(f"{owner} transform must be finite")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-12:
            raise TopdownReviewError(f"{owner} quaternion has zero norm")
        x, y, z, w = quaternion / norm
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )
        matrix[:3, 3] = translation
        return matrix
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise TopdownReviewError(f"{owner} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise TopdownReviewError(f"{owner} matrix has an invalid homogeneous row")
    return matrix.copy()


def _actor_matrix(frame_record: Mapping[str, Any], *, index: int) -> np.ndarray:
    if "world_from_actor" in frame_record:
        value = frame_record["world_from_actor"]
    elif "root_transform" in frame_record:
        value = frame_record["root_transform"]
    else:
        raise TopdownReviewError(
            f"frame record {index} lacks world_from_actor/root_transform"
        )
    return _matrix_from_transform(value, owner=f"frame record {index}")


def _camera_contract(
    room_camera_request: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    rig: Mapping[str, Any]
    if isinstance(room_camera_request.get("primary_camera_rig"), Mapping):
        rig = room_camera_request["primary_camera_rig"]
    elif isinstance(room_camera_request.get("sensor_contract"), Mapping):
        rig = room_camera_request["sensor_contract"]
    else:
        rig = room_camera_request
    if "world_from_rig" not in rig:
        raise TopdownReviewError("room camera request lacks world_from_rig")
    calibration = rig.get("shared_calibration")
    if not isinstance(calibration, Mapping):
        raise TopdownReviewError("room camera request lacks shared_calibration")
    try:
        hfov = float(calibration["hfov_degrees"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TopdownReviewError("room camera request lacks a numeric HFOV") from exc
    if not math.isfinite(hfov) or not 0.0 < hfov < 180.0:
        raise TopdownReviewError("camera HFOV must be between 0 and 180 degrees")
    return _matrix_from_transform(rig["world_from_rig"], owner="room camera"), hfov


def _source_position(source: Mapping[str, Any], *, frame_index: int) -> np.ndarray:
    per_frame = source.get("world_positions_m", source.get("positions_m"))
    if per_frame is not None:
        if not isinstance(per_frame, Sequence) or frame_index >= len(per_frame):
            raise TopdownReviewError("source per-frame positions are incomplete")
        position = np.asarray(per_frame[frame_index], dtype=np.float64)
    else:
        transform = source.get(
            "world_from_source",
            source.get("world_from_anchor", source.get("transform")),
        )
        if transform is not None:
            position = _matrix_from_transform(transform, owner="source anchor")[:3, 3]
        else:
            raw_position = source.get(
                "world_position_m",
                source.get("position_m", source.get("translation_m")),
            )
            if raw_position is None:
                raise TopdownReviewError("source anchor lacks a world position")
            position = np.asarray(raw_position, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise TopdownReviewError("source anchor position must be finite XYZ")
    return position


def _source_label(source: Mapping[str, Any], index: int) -> str:
    raw = source.get(
        "label", source.get("source_id", source.get("anchor_id", f"source{index}"))
    )
    return str(raw).encode("ascii", errors="replace").decode("ascii")[:18]


def _focus_interval(
    values: Sequence[float],
    *,
    global_low: float,
    global_high: float,
    margin_m: float,
    minimum_span_m: float,
) -> tuple[float, float]:
    global_span = global_high - global_low
    requested_low = min(values) - margin_m
    requested_high = max(values) + margin_m
    span = min(max(requested_high - requested_low, minimum_span_m), global_span)
    center = 0.5 * (requested_low + requested_high)
    low = center - 0.5 * span
    high = center + 0.5 * span
    if low < global_low:
        high += global_low - low
        low = global_low
    if high > global_high:
        low -= high - global_high
        high = global_high
    low = max(low, global_low)
    high = min(high, global_high)
    return float(low), float(high)


def _automatic_focus_bounds(
    *,
    global_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    actor_matrices: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    source_anchors: Sequence[Mapping[str, Any]],
    frame_count: int,
    margin_m: float,
    minimum_span_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    global_low, global_high = _coerce_bounds(global_bounds)
    positions = [matrix[:3, 3] for matrix in actor_matrices]
    positions.append(camera_matrix[:3, 3])
    for source in source_anchors:
        positions.extend(
            _source_position(source, frame_index=index) for index in range(frame_count)
        )
    x_values = [float(position[0]) for position in positions]
    z_values = [float(position[2]) for position in positions]
    x_low, x_high = _focus_interval(
        x_values,
        global_low=global_low[0],
        global_high=global_high[0],
        margin_m=margin_m,
        minimum_span_m=minimum_span_m,
    )
    z_low, z_high = _focus_interval(
        z_values,
        global_low=global_low[2],
        global_high=global_high[2],
        margin_m=margin_m,
        minimum_span_m=minimum_span_m,
    )
    return (
        (x_low, global_low[1], z_low),
        (x_high, global_high[1], z_high),
    )


def _local_axis_heading_xz(
    matrix: np.ndarray, local_axis: Sequence[float], *, owner: str
) -> np.ndarray:
    axis = np.asarray(local_axis, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise TopdownReviewError(f"{owner} local forward axis must be finite XYZ")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-12:
        raise TopdownReviewError(f"{owner} local forward axis must be non-zero")
    heading = matrix[:3, :3] @ (axis / axis_norm)
    xz = heading[[0, 2]]
    norm = float(np.linalg.norm(xz))
    if norm <= 1.0e-12:
        raise TopdownReviewError(f"{owner} forward axis has no world-XZ projection")
    return xz / norm


def _trajectory_actor_headings_xz(
    actor_matrices: Sequence[np.ndarray], *, movement_epsilon_m: float
) -> tuple[np.ndarray, ...]:
    if not actor_matrices:
        raise TopdownReviewError("actor trajectory must contain at least one frame")
    positions = np.asarray(
        [matrix[[0, 2], 3] for matrix in actor_matrices], dtype=np.float64
    )
    segments: list[np.ndarray | None] = []
    for index in range(max(len(positions) - 1, 0)):
        delta = positions[index + 1] - positions[index]
        norm = float(np.linalg.norm(delta))
        segments.append(delta / norm if norm > movement_epsilon_m else None)

    headings: list[np.ndarray | None] = [None] * len(positions)
    for index in range(len(positions)):
        previous = segments[index - 1] if index > 0 else None
        following = segments[index] if index < len(segments) else None
        if previous is not None and following is not None:
            combined = previous + following
            norm = float(np.linalg.norm(combined))
            headings[index] = combined / norm if norm > 1.0e-12 else following
        elif following is not None:
            headings[index] = following
        elif previous is not None:
            headings[index] = previous

    moving_indices = [
        index for index, heading in enumerate(headings) if heading is not None
    ]
    if not moving_indices:
        raise TopdownReviewError(
            "actor trajectory has no non-zero tangent and no trusted forward axis"
        )
    for index, heading in enumerate(headings):
        if heading is None:
            nearest = min(
                moving_indices,
                key=lambda candidate: (abs(candidate - index), candidate),
            )
            headings[index] = headings[nearest]
    return tuple(np.asarray(heading, dtype=np.float64) for heading in headings)


def _actor_headings_xz(
    actor_matrices: Sequence[np.ndarray],
    *,
    trusted_local_forward_axis: Sequence[float] | None,
    trusted_forward_axis_source: str | None,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    if trusted_local_forward_axis is None:
        if trusted_forward_axis_source is not None:
            raise TopdownReviewError(
                "trusted actor forward-axis source requires an explicit local axis"
            )
        headings = _trajectory_actor_headings_xz(
            actor_matrices, movement_epsilon_m=ACTOR_MOVEMENT_EPSILON_M
        )
        policy = ACTOR_HEADING_TRAJECTORY_POLICY
        axis_value: list[float] | None = None
        source_value: str | None = None
        idle_policy: str | None = ACTOR_IDLE_HEADING_POLICY
        movement_epsilon: float | None = ACTOR_MOVEMENT_EPSILON_M
    else:
        if not isinstance(trusted_forward_axis_source, str) or not (
            trusted_forward_axis_source.strip()
        ):
            raise TopdownReviewError(
                "trusted actor local forward axis requires a non-empty source"
            )
        axis = np.asarray(trusted_local_forward_axis, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise TopdownReviewError(
                "trusted actor local forward axis must be finite XYZ"
            )
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            raise TopdownReviewError(
                "trusted actor local forward axis must be non-zero"
            )
        axis /= norm
        headings = tuple(
            _local_axis_heading_xz(matrix, axis, owner="actor")
            for matrix in actor_matrices
        )
        policy = ACTOR_HEADING_TRUSTED_AXIS_POLICY
        axis_value = [float(component) for component in axis]
        source_value = trusted_forward_axis_source
        idle_policy = None
        movement_epsilon = None
    heading_values = [
        [float(component) for component in heading] for heading in headings
    ]
    binding = {
        "policy": policy,
        "trusted_local_forward_axis": axis_value,
        "trusted_forward_axis_source": source_value,
        "idle_policy": idle_policy,
        "movement_epsilon_m": movement_epsilon,
        "frame_count": len(headings),
        "canonical_content_sha256": _canonical_hash(
            heading_values, owner="actor headings"
        ),
    }
    return headings, binding


def _point_int(value: tuple[float, float]) -> tuple[int, int]:
    return int(round(value[0])), int(round(value[1]))


def _draw_callout(
    draw: ImageDraw.ImageDraw,
    *,
    image_size_wh: tuple[int, int],
    point: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    slot: int,
) -> None:
    offsets = ((8, -17), (8, 7), (-27, -17), (-27, 7), (8, -27), (-27, -27))
    offset_x, offset_y = offsets[slot % len(offsets)]
    raw_x = point[0] + offset_x
    raw_y = point[1] + offset_y
    text_box = draw.textbbox((0, 0), text)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    image_width, image_height = image_size_wh
    x = min(max(raw_x, 1), max(image_width - text_width - 3, 1))
    y = min(max(raw_y, 1), max(image_height - text_height - 3, 1))
    leader_end = (
        x if x > point[0] else x + text_width + 2,
        y + max(text_height // 2, 1),
    )
    draw.line((*point, *leader_end), fill=color, width=1)
    draw.rectangle(
        (x - 1, y - 1, x + text_width + 2, y + text_height + 1),
        fill=(18, 21, 25),
        outline=color,
        width=1,
    )
    draw.text((x + 1, y - text_box[1]), text, fill=color)


def _source_legend(
    draw: ImageDraw.ImageDraw,
    sources: Sequence[Mapping[str, Any]],
    *,
    maximum_width: int,
) -> str:
    result = ""
    for index, source in enumerate(sources):
        token = f"S{index}={_source_label(source, index)}"
        candidate = token if not result else f"{result}  {token}"
        if draw.textlength(candidate) > maximum_width:
            return f"{result}  ..." if result else "sources=..."
        result = candidate
    return result


def _panel_direction(
    projection: _PanelProjection,
    origin: np.ndarray,
    direction_xz: np.ndarray,
    *,
    pixel_length: float,
) -> tuple[float, float]:
    start = np.asarray(projection.world_to_panel(origin), dtype=np.float64)
    world_endpoint = origin.copy()
    world_endpoint[0] += float(direction_xz[0])
    world_endpoint[2] += float(direction_xz[1])
    end = np.asarray(projection.world_to_panel(world_endpoint), dtype=np.float64)
    delta = end - start
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-12:
        return 0.0, -pixel_length
    delta *= pixel_length / norm
    return float(delta[0]), float(delta[1])


def _prepare_panel(
    navmesh_binary_map: np.ndarray,
    *,
    bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    focus_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    panel_size_wh: tuple[int, int],
) -> tuple[Image.Image, _PanelProjection]:
    navmesh = np.asarray(navmesh_binary_map)
    if navmesh.ndim != 2 or navmesh.size == 0:
        raise TopdownReviewError("navmesh binary map must be a non-empty HxW array")
    if navmesh.dtype.kind not in "biu":
        raise TopdownReviewError("navmesh binary map must be boolean/integer")
    navmesh = navmesh != 0
    if len(panel_size_wh) != 2:
        raise TopdownReviewError("top-down panel size must be WxH")
    panel_width = _positive_integer(panel_size_wh[0], owner="top-down panel width")
    panel_height = _positive_integer(panel_size_wh[1], owner="top-down panel height")
    if panel_width < 16 or panel_height < 16:
        raise TopdownReviewError("top-down panel must be at least 16x16 pixels")
    low, high = _coerce_bounds(bounds)
    focus_low, focus_high = _coerce_bounds(focus_bounds)
    if (
        focus_low[0] < low[0]
        or focus_low[2] < low[2]
        or focus_high[0] > high[0]
        or focus_high[2] > high[2]
    ):
        raise TopdownReviewError("top-down focus bounds escape navmesh bounds")

    raw_height, raw_width = navmesh.shape
    focus_min_pixel = habitat_xz_to_navmesh_pixel(
        focus_low, navmesh_shape_hw=navmesh.shape, bounds=(low, high)
    )
    focus_max_pixel = habitat_xz_to_navmesh_pixel(
        focus_high, navmesh_shape_hw=navmesh.shape, bounds=(low, high)
    )
    column_start = max(0, int(math.floor(min(focus_min_pixel[0], focus_max_pixel[0]))))
    column_end = min(
        raw_width,
        int(math.ceil(max(focus_min_pixel[0], focus_max_pixel[0]))) + 1,
    )
    row_start = max(0, int(math.floor(min(focus_min_pixel[1], focus_max_pixel[1]))))
    row_end = min(
        raw_height,
        int(math.ceil(max(focus_min_pixel[1], focus_max_pixel[1]))) + 1,
    )
    if column_end <= column_start or row_end <= row_start:
        raise TopdownReviewError("top-down focus produced an empty navmesh ROI")
    navmesh = navmesh[row_start:row_end, column_start:column_end]

    title_height = 16 if panel_height >= 48 else 10
    footer_height = 34 if panel_height >= 120 else (13 if panel_height >= 96 else 2)
    padding = 6 if min(panel_width, panel_height) >= 64 else 2
    available_width = max(panel_width - 2 * padding, 1)
    available_height = max(panel_height - title_height - footer_height - 2 * padding, 1)
    nav_height, nav_width = navmesh.shape
    scale = min(available_width / nav_width, available_height / nav_height)
    fitted_width = max(1, int(round(nav_width * scale)))
    fitted_height = max(1, int(round(nav_height * scale)))
    left = (panel_width - fitted_width) // 2
    top = title_height + padding + (available_height - fitted_height) // 2
    right = left + fitted_width
    bottom = top + fitted_height

    panel = Image.new("RGB", (panel_width, panel_height), (18, 21, 25))
    navigation_color = np.asarray([195, 201, 207], dtype=np.uint8)
    blocked_color = np.asarray([48, 53, 59], dtype=np.uint8)
    nav_rgb = np.where(navmesh[..., None], navigation_color, blocked_color).astype(
        np.uint8
    )
    nav_image = Image.fromarray(nav_rgb, mode="RGB").resize(
        (fitted_width, fitted_height), resample=Image.Resampling.NEAREST
    )
    panel.paste(nav_image, (left, top))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((left, top, right - 1, bottom - 1), outline=(8, 10, 12), width=1)
    if footer_height >= 13:
        draw.text(
            (padding, panel_height - footer_height + 2),
            "light=nav  brown=descriptor object",
            fill=(175, 181, 187),
        )
    projection = _PanelProjection(
        bounds_min_xyz=focus_low,
        bounds_max_xyz=focus_high,
        navmesh_shape_hw=(nav_height, nav_width),
        viewport_xyxy=(left, top, right, bottom),
        navmesh_roi_rc=(row_start, column_start, row_end, column_end),
    )
    return panel, projection


def _draw_semantic_object_footprints(
    image: Image.Image,
    projection: _PanelProjection,
    footprints: Sequence[Mapping[str, Any]],
    *,
    reserved_points: Sequence[tuple[float, float]] = (),
) -> Image.Image:
    if not footprints:
        return image
    focus_low = np.asarray(
        [projection.bounds_min_xyz[0], projection.bounds_min_xyz[2]],
        dtype=np.float64,
    )
    focus_high = np.asarray(
        [projection.bounds_max_xyz[0], projection.bounds_max_xyz[2]],
        dtype=np.float64,
    )
    visible: list[tuple[Mapping[str, Any], list[tuple[float, float]], float]] = []
    for footprint in footprints:
        minimum = np.asarray(footprint["bounds_min_xz_m"], dtype=np.float64)
        maximum = np.asarray(footprint["bounds_max_xz_m"], dtype=np.float64)
        if np.any(maximum < focus_low) or np.any(minimum > focus_high):
            continue
        points = [
            projection.world_to_panel(point) for point in footprint["polygon_xz_m"]
        ]
        visible.append((footprint, points, float(footprint["footprint_area_m2"])))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    outline = (151, 105, 62, 255)
    for _footprint, points, _area in visible:
        overlay_draw.polygon(points, fill=(181, 133, 84, 78))
        overlay_draw.line(points + [points[0]], fill=outline, width=1)
    left, top, right, bottom = projection.viewport_xyxy
    clipped = Image.new("RGBA", image.size, (0, 0, 0, 0))
    clipped.paste(overlay.crop((left, top, right, bottom)), (left, top))
    result = Image.alpha_composite(image.convert("RGBA"), clipped).convert("RGB")
    draw = ImageDraw.Draw(result)

    # Label only the largest visible objects, and reject colliding text boxes.
    # This keeps the 240 px QA panel readable even in dense MP3D rooms.
    occupied: list[tuple[int, int, int, int]] = []
    maximum_labels = min(6, max(2, image.width // 60)) if image.width >= 96 else 0
    labelled = 0
    for footprint, points, _area in sorted(
        visible,
        key=lambda item: (-item[2], str(item[0]["object_id"])),
    ):
        if labelled >= maximum_labels:
            break
        center_x = int(round(sum(point[0] for point in points) / len(points)))
        center_y = int(round(sum(point[1] for point in points) / len(points)))
        if not (left <= center_x < right and top <= center_y < bottom):
            continue
        label = str(footprint["category"])[:12]
        text_box = draw.textbbox((center_x + 2, center_y + 2), label)
        if (
            text_box[0] < left
            or text_box[1] < top
            or text_box[2] >= right
            or text_box[3] >= bottom
            or any(
                text_box[0] - 8 <= point[0] <= text_box[2] + 8
                and text_box[1] - 8 <= point[1] <= text_box[3] + 8
                for point in reserved_points
            )
            or any(
                not (
                    text_box[2] < other[0]
                    or text_box[0] > other[2]
                    or text_box[3] < other[1]
                    or text_box[1] > other[3]
                )
                for other in occupied
            )
        ):
            continue
        draw.rectangle(text_box, fill=(32, 27, 22))
        draw.text((center_x + 2, center_y + 2), label, fill=(222, 190, 154))
        occupied.append(text_box)
        labelled += 1
    return result


def _draw_panel_overlays(
    base_panel: Image.Image,
    projection: _PanelProjection,
    *,
    actor_matrices: Sequence[np.ndarray],
    actor_headings_xz: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    camera_hfov_degrees: float,
    source_anchors: Sequence[Mapping[str, Any]],
    semantic_object_footprints: Sequence[Mapping[str, Any]],
    frame_index: int,
    frame_number: int,
) -> np.ndarray:
    actor_positions = [matrix[:3, 3] for matrix in actor_matrices]
    camera_position = camera_matrix[:3, 3]
    source_positions = [
        _source_position(source, frame_index=frame_index) for source in source_anchors
    ]
    reserved_points = [
        projection.world_to_panel(camera_position),
        projection.world_to_panel(actor_positions[frame_index]),
        *(projection.world_to_panel(position) for position in source_positions),
    ]
    image = _draw_semantic_object_footprints(
        base_panel.copy(),
        projection,
        semantic_object_footprints,
        reserved_points=reserved_points,
    )
    draw = ImageDraw.Draw(image)
    trajectory = [
        _point_int(projection.world_to_panel(position)) for position in actor_positions
    ]
    if len(trajectory) > 1:
        draw.line(trajectory, fill=(93, 112, 130), width=2)
        draw.line(trajectory[: frame_index + 1], fill=(244, 153, 47), width=3)

    camera_point = projection.world_to_panel(camera_position)
    camera_heading = _local_axis_heading_xz(
        camera_matrix, (0.0, 0.0, -1.0), owner="camera"
    )
    half_fov = math.radians(camera_hfov_degrees * 0.5)

    def rotate(direction: np.ndarray, angle: float) -> np.ndarray:
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return np.asarray(
            [
                cosine * direction[0] - sine * direction[1],
                sine * direction[0] + cosine * direction[1],
            ],
            dtype=np.float64,
        )

    fov_length = max(18.0, min(image.size) * 0.18)
    left_delta = _panel_direction(
        projection,
        camera_position,
        rotate(camera_heading, -half_fov),
        pixel_length=fov_length,
    )
    right_delta = _panel_direction(
        projection,
        camera_position,
        rotate(camera_heading, half_fov),
        pixel_length=fov_length,
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    wedge = [
        camera_point,
        (camera_point[0] + left_delta[0], camera_point[1] + left_delta[1]),
        (camera_point[0] + right_delta[0], camera_point[1] + right_delta[1]),
    ]
    overlay_draw.polygon(wedge, fill=(46, 154, 255, 55))
    overlay_draw.line(wedge + [wedge[0]], fill=(46, 154, 255, 190), width=2)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    cx, cy = _point_int(camera_point)
    draw.ellipse(
        (cx - 4, cy - 4, cx + 4, cy + 4), fill=(46, 154, 255), outline=(0, 0, 0)
    )
    if image.width >= 80:
        _draw_callout(
            draw,
            image_size_wh=image.size,
            point=(cx, cy),
            text="CAM",
            color=(20, 91, 158),
            slot=0,
        )

    source_palette = (
        (69, 196, 118),
        (180, 91, 214),
        (240, 205, 74),
        (230, 97, 96),
        (59, 201, 207),
    )
    for source_index, source in enumerate(source_anchors):
        position = source_positions[source_index]
        sx, sy = _point_int(projection.world_to_panel(position))
        color = source_palette[source_index % len(source_palette)]
        draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=color, outline=(0, 0, 0))
        if image.width >= 80:
            _draw_callout(
                draw,
                image_size_wh=image.size,
                point=(sx, sy),
                text=f"S{source_index}",
                color=color,
                slot=source_index + 2,
            )

    actor_position = actor_positions[frame_index]
    actor_point = projection.world_to_panel(actor_position)
    if len(actor_headings_xz) != len(actor_matrices):
        raise TopdownReviewError("actor heading count differs from actor frame count")
    actor_heading = np.asarray(actor_headings_xz[frame_index], dtype=np.float64)
    if actor_heading.shape != (2,) or not np.all(np.isfinite(actor_heading)):
        raise TopdownReviewError("actor heading must be finite world XZ")
    actor_delta = _panel_direction(
        projection,
        actor_position,
        actor_heading,
        pixel_length=max(10.0, min(image.size) * 0.055),
    )
    ax, ay = _point_int(actor_point)
    actor_end = (
        int(round(actor_point[0] + actor_delta[0])),
        int(round(actor_point[1] + actor_delta[1])),
    )
    draw.line((ax, ay, *actor_end), fill=(255, 118, 34), width=3)
    draw.ellipse(
        (ax - 5, ay - 5, ax + 5, ay + 5),
        fill=(255, 142, 45),
        outline=(0, 0, 0),
        width=1,
    )
    if image.width >= 80:
        _draw_callout(
            draw,
            image_size_wh=image.size,
            point=(ax, ay),
            text="A",
            color=(151, 67, 11),
            slot=1,
        )

    draw.text(
        (4, 3),
        f"TOP-DOWN QA  frame {frame_number:04d}/{len(actor_matrices) - 1:04d}",
        fill=(235, 238, 241),
    )
    if image.width >= 96 and image.height >= 64:
        left, top, _right, _bottom = projection.viewport_xyxy
        draw.line((left + 3, top + 3, left + 17, top + 3), fill=(222, 70, 70), width=2)
        draw.text((left + 18, top - 2), "+X", fill=(150, 35, 35))
        draw.line((left + 3, top + 3, left + 3, top + 17), fill=(67, 115, 225), width=2)
        draw.text((left + 6, top + 10), "+Z", fill=(35, 73, 157))
    if image.height >= 120:
        draw.text(
            (4, image.height - 21),
            "A=actor  CAM=camera",
            fill=(175, 181, 187),
        )
        source_legend = _source_legend(
            draw, source_anchors, maximum_width=max(image.width - 8, 1)
        )
        if source_legend:
            draw.text((4, image.height - 11), source_legend, fill=(175, 181, 187))
    return np.asarray(image, dtype=np.uint8)


def render_topdown_panel(
    navmesh_binary_map: np.ndarray,
    *,
    bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    frame_records: Sequence[Mapping[str, Any]],
    room_camera_request: Mapping[str, Any],
    frame_index: int,
    source_anchors: Sequence[Mapping[str, Any]] | None = None,
    semantic_object_footprints: Sequence[Mapping[str, Any]] | None = None,
    trusted_actor_local_forward_axis: Sequence[float] | None = None,
    trusted_actor_forward_axis_source: str | None = None,
    panel_size_wh: tuple[int, int] = (240, 240),
    focus_margin_m: float = 1.0,
    minimum_focus_span_m: float = 4.0,
) -> np.ndarray:
    """Render one deterministic QA panel without creating a sensor view."""

    records = tuple(frame_records)
    if not records:
        raise TopdownReviewError("top-down review requires frame records")
    if not 0 <= int(frame_index) < len(records):
        raise TopdownReviewError("top-down frame index is out of range")
    actor_matrices = tuple(
        _actor_matrix(record, index=index) for index, record in enumerate(records)
    )
    actor_headings, _actor_heading_binding = _actor_headings_xz(
        actor_matrices,
        trusted_local_forward_axis=trusted_actor_local_forward_axis,
        trusted_forward_axis_source=trusted_actor_forward_axis_source,
    )
    camera_matrix, camera_hfov = _camera_contract(room_camera_request)
    anchors = tuple(
        room_camera_request.get("sources", ())
        if source_anchors is None
        else source_anchors
    )
    if not all(isinstance(anchor, Mapping) for anchor in anchors):
        raise TopdownReviewError("source anchors must be JSON objects")
    footprints = _coerce_semantic_object_footprints(semantic_object_footprints)
    margin = _finite_float(
        focus_margin_m, owner="top-down focus margin", minimum=0.0, inclusive=True
    )
    minimum_span = _finite_float(
        minimum_focus_span_m,
        owner="top-down minimum focus span",
        minimum=0.0,
        inclusive=False,
    )
    focus_bounds = _automatic_focus_bounds(
        global_bounds=bounds,
        actor_matrices=actor_matrices,
        camera_matrix=camera_matrix,
        source_anchors=anchors,
        frame_count=len(records),
        margin_m=margin,
        minimum_span_m=minimum_span,
    )
    base, projection = _prepare_panel(
        navmesh_binary_map,
        bounds=bounds,
        focus_bounds=focus_bounds,
        panel_size_wh=panel_size_wh,
    )
    record = records[int(frame_index)]
    try:
        frame_number = int(record.get("frame_index", frame_index))
    except (TypeError, ValueError) as exc:
        raise TopdownReviewError("frame_index metadata must be an integer") from exc
    return _draw_panel_overlays(
        base,
        projection,
        actor_matrices=actor_matrices,
        actor_headings_xz=actor_headings,
        camera_matrix=camera_matrix,
        camera_hfov_degrees=camera_hfov,
        source_anchors=anchors,
        semantic_object_footprints=footprints,
        frame_index=int(frame_index),
        frame_number=frame_number,
    )


def _rgb_frame(value: np.ndarray, *, owner: str) -> np.ndarray:
    frame = np.asarray(value)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise TopdownReviewError(f"{owner} must be uint8 RGB/RGBA HxWx3/4")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise TopdownReviewError(f"{owner} has an empty image extent")
    return frame


def _coerce_rgb_frames(
    rgb_frames: Sequence[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, ...]:
    if isinstance(rgb_frames, np.ndarray):
        if rgb_frames.ndim != 4:
            raise TopdownReviewError("RGB frame stack must be NxHxWxC")
        values = tuple(rgb_frames[index] for index in range(len(rgb_frames)))
    else:
        values = tuple(rgb_frames)
    if not values:
        raise TopdownReviewError("top-down review requires at least one RGB frame")
    frames = tuple(
        _rgb_frame(frame, owner=f"RGB frame {index}")
        for index, frame in enumerate(values)
    )
    expected_shape = frames[0].shape
    if any(frame.shape != expected_shape for frame in frames[1:]):
        raise TopdownReviewError("RGB frame shape changed within the sequence")
    return frames


def _array_sequence_digest(algorithm: str) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(algorithm.encode("ascii") + b"\0")
    return digest


def _update_array_sequence_digest(
    digest: "hashlib._Hash", value: np.ndarray, *, index: int
) -> None:
    array = np.asarray(value)
    metadata = {
        "index": index,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))


def _array_sequence_hash(arrays: Sequence[np.ndarray], *, algorithm: str) -> str:
    digest = _array_sequence_digest(algorithm)
    for index, value in enumerate(arrays):
        _update_array_sequence_digest(digest, value, index=index)
    return digest.hexdigest()


def _review_encoder_frame(
    value: np.ndarray, expected_shape: tuple[int, ...] | None
) -> np.ndarray:
    frame = np.asarray(value)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise TopdownReviewError("review encoder frame must be uint8 RGB HxWx3")
    if expected_shape is not None and frame.shape != expected_shape:
        raise TopdownReviewError(
            f"review encoder frame shape changed: expected {expected_shape}, got {frame.shape}"
        )
    if frame.shape[0] % 2 or frame.shape[1] % 2:
        raise TopdownReviewError("review encoder requires even frame dimensions")
    return np.ascontiguousarray(frame)


def encode_review_rgb_frames(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    *,
    fps: int,
    preset: str = "veryfast",
) -> int:
    """Atomically stream RGB frames to FFmpeg without staging PNG files."""

    frame_rate = _positive_integer(fps, owner="review video FPS")
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".mp4":
        raise TopdownReviewError("review video output must use the .mp4 suffix")
    if destination.exists():
        raise TopdownReviewError(f"review video output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    try:
        first = _review_encoder_frame(next(iterator), None)
    except StopIteration as exc:
        raise TopdownReviewError("review video requires at least one frame") from exc
    height, width, _channels = first.shape

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(frame_rate),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        str(temporary),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise TopdownReviewError(
            f"could not start FFmpeg review encoder: {exc}"
        ) from exc
    count = 0
    try:
        assert process.stdin is not None
        process.stdin.write(first.tobytes())
        count = 1
        for frame in iterator:
            encoded = _review_encoder_frame(frame, first.shape)
            process.stdin.write(encoded.tobytes())
            count += 1
        process.stdin.close()
        return_code = process.wait()
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if return_code != 0:
            raise TopdownReviewError(
                f"FFmpeg raw RGB encoder returned {return_code}: {stderr.strip()}"
            )
        try:
            # The temporary lives in the destination directory, so a hard-link
            # publish is same-filesystem and atomically fails if another writer
            # won the destination name after our preflight.
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise TopdownReviewError(
                f"review video output appeared during encoding: {destination}"
            ) from exc
        temporary.unlink()
        return count
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        temporary.unlink(missing_ok=True)
        raise


def _file_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def _unlink_owned_file(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        if _file_identity(path) == identity:
            path.unlink()
    except FileNotFoundError:
        return


def _write_numpy_array_exclusive(
    value: np.ndarray, destination: Path
) -> tuple[int, int]:
    """Atomically publish one NPY without replacing a concurrent writer."""

    if destination.suffix.lower() != ".npy":
        raise TopdownReviewError("derived array output must use the .npy suffix")
    if destination.exists() or destination.is_symlink():
        raise TopdownReviewError(f"derived array output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.save(handle, np.asarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise TopdownReviewError(
            f"derived array output appeared during publication: {destination}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _file_identity(destination)


def _write_json_exclusive(value: Any, destination: Path) -> tuple[int, int]:
    """Atomically publish pretty JSON without replacing a concurrent writer."""

    if destination.exists() or destination.is_symlink():
        raise TopdownReviewError(f"JSON output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise TopdownReviewError(
            f"JSON output appeared during publication: {destination}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _file_identity(destination)


def _compose_frame(main_rgb: np.ndarray, panel_rgb: np.ndarray) -> np.ndarray:
    main = np.ascontiguousarray(main_rgb[..., :3])
    panel = np.ascontiguousarray(panel_rgb[..., :3])
    content_height = max(main.shape[0], panel.shape[0])
    content_width = main.shape[1] + panel.shape[1]
    encoded_height = content_height + content_height % 2
    encoded_width = content_width + content_width % 2
    result = np.zeros((encoded_height, encoded_width, 3), dtype=np.uint8)
    main_y = (content_height - main.shape[0]) // 2
    panel_y = (content_height - panel.shape[0]) // 2
    result[main_y : main_y + main.shape[0], : main.shape[1]] = main
    result[
        panel_y : panel_y + panel.shape[0],
        main.shape[1] : main.shape[1] + panel.shape[1],
    ] = panel
    return result


def _input_artifact_records(
    input_artifacts: Mapping[str, str | Path] | None,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, raw_path in (input_artifacts or {}).items():
        if not isinstance(name, str) or not name:
            raise TopdownReviewError("input artifact names must be non-empty strings")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise TopdownReviewError(f"input artifact is missing: {path}")
        records[name] = {
            "path": str(path),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def compose_topdown_review(
    *,
    rgb_frames: Sequence[np.ndarray] | np.ndarray,
    frame_records: Sequence[Mapping[str, Any]],
    room_camera_request: Mapping[str, Any],
    navmesh_binary_map: np.ndarray,
    navmesh_bounds: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    output_dir: str | Path,
    source_anchors: Sequence[Mapping[str, Any]] | None = None,
    semantic_object_footprints: Sequence[Mapping[str, Any]] | None = None,
    trusted_actor_local_forward_axis: Sequence[float] | None = None,
    trusted_actor_forward_axis_source: str | None = None,
    panel_size_wh: tuple[int, int] | None = None,
    focus_margin_m: float = 1.0,
    minimum_focus_span_m: float = 4.0,
    fps: int = 15,
    output_name: str = "rgb_topdown_review.mp4",
    navmesh_metadata: Mapping[str, Any] | None = None,
    input_artifacts: Mapping[str, str | Path] | None = None,
    review_video_encode: ReviewVideoEncoder = encode_review_rgb_frames,
) -> dict[str, Any]:
    """Compose and hash-bind a synchronized, QA-only top-down review video."""

    frame_rate = _positive_integer(fps, owner="review video FPS")
    frames = _coerce_rgb_frames(rgb_frames)
    records = tuple(frame_records)
    if len(records) != len(frames):
        raise TopdownReviewError(
            "RGB frame count differs from actor frame record count"
        )
    if not all(isinstance(record, Mapping) for record in records):
        raise TopdownReviewError("frame records must be JSON objects")
    actor_matrices = tuple(
        _actor_matrix(record, index=index) for index, record in enumerate(records)
    )
    actor_headings, actor_heading_binding = _actor_headings_xz(
        actor_matrices,
        trusted_local_forward_axis=trusted_actor_local_forward_axis,
        trusted_forward_axis_source=trusted_actor_forward_axis_source,
    )
    camera_matrix, camera_hfov = _camera_contract(room_camera_request)
    anchors = tuple(
        room_camera_request.get("sources", ())
        if source_anchors is None
        else source_anchors
    )
    if not all(isinstance(anchor, Mapping) for anchor in anchors):
        raise TopdownReviewError("source anchors must be JSON objects")
    footprints = _coerce_semantic_object_footprints(semantic_object_footprints)
    margin = _finite_float(
        focus_margin_m, owner="top-down focus margin", minimum=0.0, inclusive=True
    )
    minimum_span = _finite_float(
        minimum_focus_span_m,
        owner="top-down minimum focus span",
        minimum=0.0,
        inclusive=False,
    )
    focus_bounds = _automatic_focus_bounds(
        global_bounds=navmesh_bounds,
        actor_matrices=actor_matrices,
        camera_matrix=camera_matrix,
        source_anchors=anchors,
        frame_count=len(records),
        margin_m=margin,
        minimum_span_m=minimum_span,
    )
    main_height, main_width = frames[0].shape[:2]
    panel_size = panel_size_wh or (main_height, main_height)
    base_panel, projection = _prepare_panel(
        navmesh_binary_map,
        bounds=navmesh_bounds,
        focus_bounds=focus_bounds,
        panel_size_wh=panel_size,
    )
    navmesh = np.asarray(navmesh_binary_map)
    bounds_low, bounds_high = _coerce_bounds(navmesh_bounds)
    metadata = {} if navmesh_metadata is None else _jsonable(navmesh_metadata)
    artifacts = _input_artifact_records(input_artifacts)
    if footprints and "semantic_descriptor" not in artifacts:
        raise TopdownReviewError(
            "non-empty semantic object footprints require a semantic_descriptor "
            "input artifact"
        )

    rgb_sha256 = _array_sequence_hash(frames, algorithm=RGB_STACK_HASH_ALGORITHM)
    navmesh_sha256 = _array_sequence_hash((navmesh,), algorithm=NAVMESH_HASH_ALGORITHM)
    frame_records_sha256 = _canonical_hash(records, owner="frame records")
    room_request_sha256 = _canonical_hash(
        room_camera_request, owner="room camera request"
    )
    anchors_sha256 = _canonical_hash(anchors, owner="source anchors")
    footprints_sha256 = _canonical_hash(footprints, owner="semantic object footprints")
    metadata_sha256 = _canonical_hash(metadata, owner="navmesh metadata")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination_name = Path(output_name)
    if (
        destination_name.name != output_name
        or destination_name.suffix.lower() != ".mp4"
    ):
        raise TopdownReviewError("output_name must be one local .mp4 filename")
    destination = output / destination_name
    navmesh_destination = output / "navmesh_binary.npy"
    evidence_path = output / "topdown_review_evidence.json"
    if (
        destination.exists()
        or navmesh_destination.exists()
        or navmesh_destination.is_symlink()
        or evidence_path.exists()
    ):
        raise TopdownReviewError("top-down review output already exists")

    composite_digest = _array_sequence_digest(COMPOSITE_RGB_STACK_HASH_ALGORITHM)
    composite_count = 0
    composite_shape: list[int] | None = None

    def composite_frames() -> Iterable[np.ndarray]:
        nonlocal composite_count, composite_shape
        for index, main_frame in enumerate(frames):
            try:
                frame_number = int(records[index].get("frame_index", index))
            except (TypeError, ValueError) as exc:
                raise TopdownReviewError(
                    "frame_index metadata must be an integer"
                ) from exc
            panel = _draw_panel_overlays(
                base_panel,
                projection,
                actor_matrices=actor_matrices,
                actor_headings_xz=actor_headings,
                camera_matrix=camera_matrix,
                camera_hfov_degrees=camera_hfov,
                source_anchors=anchors,
                semantic_object_footprints=footprints,
                frame_index=index,
                frame_number=frame_number,
            )
            composite = _compose_frame(main_frame, panel)
            _update_array_sequence_digest(composite_digest, composite, index=index)
            composite_count += 1
            if composite_shape is None:
                composite_shape = list(composite.shape)
            yield composite

    navmesh_identity: tuple[int, int] | None = None
    video_identity: tuple[int, int] | None = None
    try:
        navmesh_identity = _write_numpy_array_exclusive(navmesh, navmesh_destination)
        encoded_count = review_video_encode(
            composite_frames(), destination, fps=frame_rate
        )
        if destination.is_file():
            video_identity = _file_identity(destination)
        if encoded_count != len(frames):
            raise TopdownReviewError(
                "review encoder frame count differs from requested frame count"
            )
        if composite_count != len(frames) or composite_shape is None:
            raise TopdownReviewError(
                "review encoder did not consume every composite RGB frame"
            )
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise TopdownReviewError("review encoder did not create a non-empty MP4")
        if (
            _array_sequence_hash(frames, algorithm=RGB_STACK_HASH_ALGORITHM)
            != rgb_sha256
        ):
            raise TopdownReviewError("RGB input frames changed during composition")
        if (
            _array_sequence_hash((navmesh,), algorithm=NAVMESH_HASH_ALGORITHM)
            != navmesh_sha256
        ):
            raise TopdownReviewError("navmesh input changed during composition")
        if _canonical_hash(records, owner="frame records") != frame_records_sha256:
            raise TopdownReviewError("frame records changed during composition")
        if (
            _canonical_hash(room_camera_request, owner="room camera request")
            != room_request_sha256
        ):
            raise TopdownReviewError("room camera request changed during composition")
        if _canonical_hash(anchors, owner="source anchors") != anchors_sha256:
            raise TopdownReviewError("source anchors changed during composition")
        if (
            _canonical_hash(footprints, owner="semantic object footprints")
            != footprints_sha256
        ):
            raise TopdownReviewError(
                "semantic object footprints changed during composition"
            )
    except BaseException:
        _unlink_owned_file(destination, video_identity)
        _unlink_owned_file(navmesh_destination, navmesh_identity)
        raise

    panel_width, panel_height = base_panel.size
    content_height = max(main_height, panel_height)
    content_width = main_width + panel_width
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "status": "pass",
        "evidence_kind": "derived_m2_rgb_topdown_qa_review",
        "review_only": True,
        "qa_only": True,
        "formal_view": False,
        "qualification_claim": False,
        "view_id": None,
        "formal_view_ids": [],
        "formal_capture_modified": False,
        "sensor_view_created": False,
        "qa_policy": {
            "panel_kind": "navmesh_semantic_descriptor_topdown",
            "derived_postprocess": True,
            "formal_view": False,
            "qualification_claim": False,
            "view_id": None,
            "navmesh_semantics": "binary_navigability_not_object_identity",
            "descriptor_semantics_not_object_detection": True,
            "semantic_scene_access_created_sensor": False,
            "camera_heading_policy": CAMERA_HEADING_POLICY,
            "actor_heading_policy": actor_heading_binding["policy"],
        },
        "timeline": {
            "frame_count": len(frames),
            "frame_rate_hz": frame_rate,
            "synchronized_one_panel_per_rgb_frame": True,
        },
        "inputs": {
            "rgb_frames": {
                "hash_algorithm": RGB_STACK_HASH_ALGORITHM,
                "content_sha256": rgb_sha256,
                "frame_count": len(frames),
                "frame_shape": list(frames[0].shape),
                "dtype": frames[0].dtype.str,
            },
            "composite_frames": {
                "hash_algorithm": COMPOSITE_RGB_STACK_HASH_ALGORITHM,
                "content_sha256": composite_digest.hexdigest(),
                "frame_count": composite_count,
                "frame_shape": composite_shape,
                "dtype": np.dtype(np.uint8).str,
            },
            "frame_records": {
                "canonical_content_sha256": frame_records_sha256,
                "frame_count": len(records),
            },
            "actor_heading": actor_heading_binding,
            "room_camera_request": {
                "canonical_content_sha256": room_request_sha256,
                "hfov_degrees": camera_hfov,
            },
            "source_anchors": {
                "canonical_content_sha256": anchors_sha256,
                "count": len(anchors),
            },
            "semantic_object_footprints": {
                "policy": SEMANTIC_OBJECT_FOOTPRINT_POLICY,
                "source": SEMANTIC_OBJECT_SOURCE,
                "descriptor_semantics_not_object_detection": True,
                "semantic_descriptor_artifact": (
                    "semantic_descriptor"
                    if "semantic_descriptor" in artifacts
                    else None
                ),
                "excluded_category_keys": sorted(_STRUCTURAL_SEMANTIC_CATEGORIES),
                "maximum_footprint_area_m2": (SEMANTIC_OBJECT_MAX_FOOTPRINT_AREA_M2),
                "maximum_footprint_span_m": (SEMANTIC_OBJECT_MAX_FOOTPRINT_SPAN_M),
                "canonical_content_sha256": footprints_sha256,
                "count": len(footprints),
                "objects": list(footprints),
            },
            "navmesh_binary_map": {
                "hash_algorithm": NAVMESH_HASH_ALGORITHM,
                "content_sha256": navmesh_sha256,
                "shape": list(navmesh.shape),
                "dtype": navmesh.dtype.str,
                "bounds_min_xyz": list(bounds_low),
                "bounds_max_xyz": list(bounds_high),
                "metadata": metadata,
                "metadata_content_sha256": metadata_sha256,
            },
            "artifacts": artifacts,
        },
        "layout": {
            "main_rgb_size_wh": [main_width, main_height],
            "topdown_panel_size_wh": [panel_width, panel_height],
            "content_size_wh": [content_width, content_height],
            "encoded_size_wh": [
                content_width + content_width % 2,
                content_height + content_height % 2,
            ],
            "topdown_panel_side": "right",
            "focus": {
                "policy": "camera_actor_trajectory_sources_aabb_v1",
                "margin_m": margin,
                "minimum_span_m": minimum_span,
                "effective_bounds_min_xyz": list(projection.bounds_min_xyz),
                "effective_bounds_max_xyz": list(projection.bounds_max_xyz),
                "navmesh_roi_rc_exclusive": list(projection.navmesh_roi_rc),
            },
        },
        "encoder": {
            "contract": "streamed_uint8_rgb_frames",
            "png_staging": False,
            "callable": (
                f"{getattr(review_video_encode, '__module__', '<unknown>')}."
                f"{getattr(review_video_encode, '__qualname__', '<unknown>')}"
            ),
            "readback_validation": {
                "decoder_contract": "ffmpeg_full_rgb24_v1",
                "maximum_frame_mean_absolute_error": (VIDEO_READBACK_MAXIMUM_FRAME_MAE),
                "maximum_root_mean_squared_error": VIDEO_READBACK_MAXIMUM_RMSE,
            },
        },
        "output": {
            "video": file_record(destination, relative_to=output),
            "navmesh_binary_map": file_record(navmesh_destination, relative_to=output),
        },
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    try:
        _write_json_exclusive(evidence, evidence_path)
    except BaseException:
        _unlink_owned_file(destination, video_identity)
        _unlink_owned_file(navmesh_destination, navmesh_identity)
        raise
    return evidence


def _path_without_symlinks(path: Path, *, owner: str) -> Path:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise TopdownReviewError(f"{owner} path contains a symbolic link")
    return absolute


def _verified_evidence_artifact(
    record: Any,
    *,
    owner: str,
    evidence_directory: Path,
    confined: bool,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "byte_size",
        "sha256",
    }:
        raise TopdownReviewError(f"{owner} file record fields are invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TopdownReviewError(f"{owner} artifact path is invalid")
    declared = Path(raw_path)
    if confined and declared.is_absolute():
        raise TopdownReviewError(f"{owner} must use a relative output path")
    if not confined and not declared.is_absolute():
        raise TopdownReviewError(f"{owner} input path must be absolute")
    candidate = declared if declared.is_absolute() else evidence_directory / declared
    artifact = _path_without_symlinks(candidate, owner=owner)
    if confined:
        try:
            artifact.relative_to(evidence_directory)
        except ValueError as exc:
            raise TopdownReviewError(
                f"{owner} escapes the top-down evidence directory"
            ) from exc
    if not artifact.is_file():
        raise TopdownReviewError(f"{owner} artifact is missing")
    declared_size = record.get("byte_size")
    declared_sha256 = record.get("sha256")
    if (
        isinstance(declared_size, bool)
        or not isinstance(declared_size, int)
        or declared_size < 0
        or not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha256)
    ):
        raise TopdownReviewError(f"{owner} byte size/SHA-256 declaration is invalid")
    try:
        actual_size = artifact.stat().st_size
        actual_sha256 = sha256_file(artifact)
    except OSError as exc:
        raise TopdownReviewError(f"{owner} artifact is unreadable: {exc}") from exc
    if declared_size != actual_size or declared_sha256 != actual_sha256:
        raise TopdownReviewError(f"{owner} artifact bytes changed")
    return artifact


def _declared_input_binding(
    inputs: Mapping[str, Any], name: str, errors: list[str]
) -> Mapping[str, Any] | None:
    value = inputs.get(name)
    if not isinstance(value, Mapping):
        errors.append(f"inputs.{name} binding is missing")
        return None
    return value


def _recomputed_json_hash(value: Any, *, owner: str, errors: list[str]) -> str | None:
    try:
        return _canonical_hash(value, owner=owner)
    except TopdownReviewError as exc:
        errors.append(str(exc))
        return None


def _probe_review_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,"
            "r_frame_rate,nb_read_frames:format=format_name"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=30.0
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TopdownReviewError(f"could not probe output.video: {exc}") from exc
    if completed.returncode != 0:
        raise TopdownReviewError(
            "output.video is not a probeable MP4: " + completed.stderr.strip()
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise TopdownReviewError(
            "ffprobe returned invalid JSON for output.video"
        ) from exc
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    format_value = payload.get("format") if isinstance(payload, Mapping) else None
    if (
        not isinstance(streams, list)
        or len(streams) != 1
        or not isinstance(streams[0], Mapping)
    ):
        raise TopdownReviewError("output.video must contain exactly one stream")
    stream = dict(streams[0])
    if stream.get("codec_type") != "video" or stream.get("codec_name") != "h264":
        raise TopdownReviewError("output.video must contain one H.264 video stream")
    if not isinstance(format_value, Mapping) or "mp4" not in str(
        format_value.get("format_name", "")
    ).split(","):
        raise TopdownReviewError("output.video container is not MP4")
    try:
        decode = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TopdownReviewError(f"could not decode output.video: {exc}") from exc
    if decode.returncode != 0 or decode.stderr.strip():
        raise TopdownReviewError(
            "output.video failed full decode: " + decode.stderr.strip()
        )
    return stream


def _decode_review_video_rgb(
    path: Path, *, width: int, height: int, frame_count: int
) -> np.ndarray:
    try:
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            check=False,
            capture_output=True,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TopdownReviewError(f"could not decode output.video RGB: {exc}") from exc
    stderr = decoded.stderr.decode("utf-8", errors="replace").strip()
    if decoded.returncode != 0 or stderr:
        raise TopdownReviewError("output.video RGB readback failed: " + stderr)
    expected_bytes = width * height * 3 * frame_count
    if len(decoded.stdout) != expected_bytes:
        raise TopdownReviewError(
            "output.video RGB readback byte count differs from its timeline/layout"
        )
    return np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(
        frame_count, height, width, 3
    )


def _size_pair(value: Any, *, owner: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise TopdownReviewError(f"{owner} must be a positive [width, height]")
    return (
        _positive_integer(value[0], owner=f"{owner} width"),
        _positive_integer(value[1], owner=f"{owner} height"),
    )


def _strict_core_capture_errors(
    core_path: Path, verified_inputs: Mapping[str, Path]
) -> list[str]:
    """Dispatch the bound capture through its schema-specific strict verifier."""

    from avengine.assets.habitat_capture import (
        EVIDENCE_SCHEMA as HABITAT_CAPTURE_EVIDENCE_SCHEMA,
        verify_saved_capture_arrays,
    )
    from avengine.assets.local_tr_review import (
        EVIDENCE_SCHEMA as LOCAL_TR_REVIEW_EVIDENCE_SCHEMA,
        verify_local_tr_review_evidence,
    )
    from avengine.assets.variant_review import verify_variant_review_evidence

    try:
        core = load_json(core_path)
    except (OSError, ValueError) as exc:
        return [f"bound core capture evidence is invalid: {exc}"]
    schema = core.get("schema")
    wrapper_path = verified_inputs.get("variant_review_evidence")
    if schema == LOCAL_TR_REVIEW_EVIDENCE_SCHEMA:
        errors: list[str] = []
        if wrapper_path is not None:
            errors.append("local-TR core capture must not bind a variant wrapper")
        try:
            errors.extend(verify_local_tr_review_evidence(core_path))
        except (OSError, TypeError, ValueError, KeyError) as exc:
            errors.append(f"local-TR verifier rejected malformed evidence: {exc}")
        return errors
    if schema == HABITAT_CAPTURE_EVIDENCE_SCHEMA:
        if wrapper_path is None:
            return ["rotation-only core capture lacks a verified variant wrapper"]
        expected_core = wrapper_path.parent / "evidence.json"
        if expected_core.resolve() != core_path.resolve():
            return ["variant wrapper does not share the bound core capture"]
        try:
            return [
                *verify_saved_capture_arrays(core, core_path.parent),
                *verify_variant_review_evidence(wrapper_path),
            ]
        except (OSError, TypeError, ValueError, KeyError) as exc:
            return [f"variant verifier rejected malformed evidence: {exc}"]
    return ["bound core capture schema has no supported strict verifier"]


def verify_topdown_review_evidence(path: str | Path) -> list[str]:
    """Rehash a QA-only review and independently rebuild its source bindings."""

    try:
        evidence_path = _path_without_symlinks(
            Path(path), owner="top-down review evidence"
        )
    except TopdownReviewError as exc:
        return [str(exc)]
    if not evidence_path.is_file():
        return ["top-down review evidence is not a regular file"]
    try:
        value = load_json(evidence_path)
    except (OSError, ValueError) as exc:
        return [f"top-down review evidence is unreadable: {exc}"]

    errors: list[str] = []
    declared_hash = value.get("evidence_content_sha256")
    hash_payload = dict(value)
    hash_payload.pop("evidence_content_sha256", None)
    try:
        actual_hash = canonical_json_sha256(hash_payload)
    except (TypeError, ValueError):
        actual_hash = None
    if declared_hash != actual_hash:
        errors.append("top-down review evidence content hash differs")

    qa_policy = value.get("qa_policy")
    expected_qa_policy_fields = {
        "panel_kind",
        "derived_postprocess",
        "formal_view",
        "qualification_claim",
        "view_id",
        "navmesh_semantics",
        "descriptor_semantics_not_object_detection",
        "semantic_scene_access_created_sensor",
        "camera_heading_policy",
        "actor_heading_policy",
    }
    descriptor_panel_policy = bool(
        isinstance(qa_policy, Mapping)
        and set(qa_policy) == expected_qa_policy_fields
        and qa_policy.get("panel_kind") == "navmesh_semantic_descriptor_topdown"
        and qa_policy.get("navmesh_semantics")
        == "binary_navigability_not_object_identity"
        and qa_policy.get("descriptor_semantics_not_object_detection") is True
        and qa_policy.get("semantic_scene_access_created_sensor") is False
    )
    heading_policy_valid = bool(
        isinstance(qa_policy, Mapping)
        and qa_policy.get("camera_heading_policy") == CAMERA_HEADING_POLICY
        and qa_policy.get("actor_heading_policy")
        in {ACTOR_HEADING_TRAJECTORY_POLICY, ACTOR_HEADING_TRUSTED_AXIS_POLICY}
    )
    qa_claim_valid = (
        value.get("schema") == EVIDENCE_SCHEMA
        and value.get("status") == "pass"
        and value.get("evidence_kind") == "derived_m2_rgb_topdown_qa_review"
        and value.get("review_only") is True
        and value.get("qa_only") is True
        and value.get("formal_view") is False
        and value.get("qualification_claim") is False
        and value.get("view_id") is None
        and value.get("formal_view_ids") == []
        and value.get("formal_capture_modified") is False
        and value.get("sensor_view_created") is False
        and isinstance(qa_policy, Mapping)
        and descriptor_panel_policy
        and heading_policy_valid
        and qa_policy.get("derived_postprocess") is True
        and qa_policy.get("formal_view") is False
        and qa_policy.get("qualification_claim") is False
        and qa_policy.get("view_id") is None
    )
    if not qa_claim_valid:
        errors.append("top-down review QA-only claim is invalid")

    timeline = value.get("timeline")
    timeline_frame_count: int | None = None
    timeline_fps: int | None = None
    if not isinstance(timeline, Mapping) or set(timeline) != {
        "frame_count",
        "frame_rate_hz",
        "synchronized_one_panel_per_rgb_frame",
    }:
        errors.append("top-down review timeline is invalid")
    else:
        try:
            timeline_frame_count = _positive_integer(
                timeline.get("frame_count"), owner="timeline frame_count"
            )
            timeline_fps = _positive_integer(
                timeline.get("frame_rate_hz"), owner="timeline frame_rate_hz"
            )
        except TopdownReviewError as exc:
            errors.append(str(exc))
        if timeline.get("synchronized_one_panel_per_rgb_frame") is not True:
            errors.append("top-down review timeline is not synchronized")

    layout = value.get("layout")
    encoded_size: tuple[int, int] | None = None
    main_size: tuple[int, int] | None = None
    panel_size: tuple[int, int] | None = None
    focus_value: Mapping[str, Any] | None = None
    focus_margin: float | None = None
    minimum_focus_span: float | None = None
    focus_bounds_value: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None
    focus_roi: tuple[int, int, int, int] | None = None
    if not isinstance(layout, Mapping) or set(layout) != {
        "main_rgb_size_wh",
        "topdown_panel_size_wh",
        "content_size_wh",
        "encoded_size_wh",
        "topdown_panel_side",
        "focus",
    }:
        errors.append("top-down review layout is invalid")
    else:
        try:
            main_size = _size_pair(
                layout.get("main_rgb_size_wh"), owner="main RGB size"
            )
            panel_size = _size_pair(
                layout.get("topdown_panel_size_wh"), owner="top-down panel size"
            )
            content_size = _size_pair(
                layout.get("content_size_wh"), owner="content size"
            )
            encoded_size = _size_pair(
                layout.get("encoded_size_wh"), owner="encoded size"
            )
        except TopdownReviewError as exc:
            errors.append(str(exc))
        else:
            expected_content = (
                main_size[0] + panel_size[0],
                max(main_size[1], panel_size[1]),
            )
            expected_encoded = (
                expected_content[0] + expected_content[0] % 2,
                expected_content[1] + expected_content[1] % 2,
            )
            if content_size != expected_content or encoded_size != expected_encoded:
                errors.append("top-down review layout size formula differs")
        if layout.get("topdown_panel_side") != "right":
            errors.append("top-down panel is not on the right")
        raw_focus = layout.get("focus")
        if not isinstance(raw_focus, Mapping) or set(raw_focus) != {
            "policy",
            "margin_m",
            "minimum_span_m",
            "effective_bounds_min_xyz",
            "effective_bounds_max_xyz",
            "navmesh_roi_rc_exclusive",
        }:
            errors.append("top-down focus binding is invalid")
        else:
            focus_value = raw_focus
            if raw_focus.get("policy") != "camera_actor_trajectory_sources_aabb_v1":
                errors.append("top-down focus policy is invalid")
            try:
                focus_margin = _finite_float(
                    raw_focus.get("margin_m"),
                    owner="focus margin",
                    minimum=0.0,
                    inclusive=True,
                )
                minimum_focus_span = _finite_float(
                    raw_focus.get("minimum_span_m"),
                    owner="minimum focus span",
                    minimum=0.0,
                    inclusive=False,
                )
                focus_bounds_value = _coerce_bounds(
                    (
                        raw_focus.get("effective_bounds_min_xyz"),
                        raw_focus.get("effective_bounds_max_xyz"),
                    )
                )
                raw_roi = raw_focus.get("navmesh_roi_rc_exclusive")
                if (
                    not isinstance(raw_roi, list)
                    or len(raw_roi) != 4
                    or any(
                        isinstance(item, bool) or not isinstance(item, int)
                        for item in raw_roi
                    )
                ):
                    raise TopdownReviewError(
                        "focus navmesh ROI must be four integer bounds"
                    )
                focus_roi = tuple(raw_roi)
            except (TypeError, TopdownReviewError) as exc:
                errors.append(f"top-down focus values are invalid: {exc}")

    encoder = value.get("encoder")
    expected_encoder_callable = (
        f"{encode_review_rgb_frames.__module__}.{encode_review_rgb_frames.__qualname__}"
    )
    if (
        not isinstance(encoder, Mapping)
        or set(encoder)
        != {"contract", "png_staging", "callable", "readback_validation"}
        or encoder.get("contract") != "streamed_uint8_rgb_frames"
        or encoder.get("png_staging") is not False
        or encoder.get("callable") != expected_encoder_callable
        or encoder.get("readback_validation")
        != {
            "decoder_contract": "ffmpeg_full_rgb24_v1",
            "maximum_frame_mean_absolute_error": (VIDEO_READBACK_MAXIMUM_FRAME_MAE),
            "maximum_root_mean_squared_error": VIDEO_READBACK_MAXIMUM_RMSE,
        }
    ):
        errors.append("top-down review encoder contract is invalid")

    evidence_directory = evidence_path.parent
    output = value.get("output")
    video_path: Path | None = None
    video_stream: Mapping[str, Any] | None = None
    saved_navmesh: np.ndarray | None = None
    if not isinstance(output, Mapping) or set(output) != {
        "video",
        "navmesh_binary_map",
    }:
        errors.append("top-down review output binding is missing")
    else:
        video_record = output.get("video")
        try:
            video_path = _verified_evidence_artifact(
                video_record,
                owner="output.video",
                evidence_directory=evidence_directory,
                confined=True,
            )
            if video_path.suffix.lower() != ".mp4" or video_path.stat().st_size <= 0:
                errors.append("output.video is not a non-empty MP4")
            else:
                video_stream = _probe_review_video(video_path)
        except (OSError, TopdownReviewError) as exc:
            errors.append(str(exc))
        try:
            navmesh_output_path = _verified_evidence_artifact(
                output.get("navmesh_binary_map"),
                owner="output.navmesh_binary_map",
                evidence_directory=evidence_directory,
                confined=True,
            )
            if navmesh_output_path.suffix.lower() != ".npy":
                raise TopdownReviewError(
                    "output.navmesh_binary_map must be an NPY artifact"
                )
            loaded_navmesh = np.load(
                navmesh_output_path, mmap_mode="r", allow_pickle=False
            )
            if not isinstance(loaded_navmesh, np.ndarray):
                raise TopdownReviewError(
                    "output.navmesh_binary_map NPY did not contain an array"
                )
            saved_navmesh = loaded_navmesh
        except (OSError, ValueError, TopdownReviewError) as exc:
            errors.append(str(exc))

    if video_stream is not None:
        try:
            probed_size = (
                _positive_integer(int(video_stream.get("width")), owner="video width"),
                _positive_integer(
                    int(video_stream.get("height")), owner="video height"
                ),
            )
            probed_frames = _positive_integer(
                int(video_stream.get("nb_read_frames")), owner="video frame count"
            )
            average_rate = Fraction(str(video_stream.get("avg_frame_rate")))
            nominal_rate = Fraction(str(video_stream.get("r_frame_rate")))
        except (TopdownReviewError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"output.video stream metadata is invalid: {exc}")
        else:
            if encoded_size is not None and probed_size != encoded_size:
                errors.append("output.video size differs from encoded layout")
            if (
                timeline_frame_count is not None
                and probed_frames != timeline_frame_count
            ):
                errors.append("output.video frame count differs from timeline")
            if timeline_fps is not None and (
                average_rate != timeline_fps or nominal_rate != timeline_fps
            ):
                errors.append("output.video frame rate differs from timeline")
        if video_stream.get("pix_fmt") != "yuv420p":
            errors.append("output.video pixel format is not yuv420p")

    inputs_value = value.get("inputs")
    if not isinstance(inputs_value, Mapping):
        errors.append("top-down review input bindings are missing")
        return errors
    expected_input_fields = {
        "rgb_frames",
        "composite_frames",
        "frame_records",
        "actor_heading",
        "room_camera_request",
        "source_anchors",
        "semantic_object_footprints",
        "navmesh_binary_map",
        "artifacts",
    }
    if set(inputs_value) != expected_input_fields:
        errors.append("top-down review input binding fields are invalid")
    artifacts_value = inputs_value.get("artifacts")
    verified_inputs: dict[str, Path] = {}
    required_artifacts = {
        "core_capture_evidence",
        "navmesh",
        "rgb_array",
        "room_manifest",
        "room_request",
    }
    if not isinstance(artifacts_value, Mapping):
        errors.append("inputs.artifacts bindings are missing")
    else:
        missing = required_artifacts - set(artifacts_value)
        if missing:
            errors.append(
                "inputs.artifacts lacks required bindings: "
                + ", ".join(sorted(missing))
            )
        for name, record in artifacts_value.items():
            try:
                verified_inputs[name] = _verified_evidence_artifact(
                    record,
                    owner=f"inputs.artifacts.{name}",
                    evidence_directory=evidence_directory,
                    confined=False,
                )
            except TopdownReviewError as exc:
                errors.append(str(exc))

    semantic_binding = inputs_value.get("semantic_object_footprints")
    verified_footprints: tuple[Mapping[str, Any], ...] | None = None
    expected_semantic_fields = {
        "policy",
        "source",
        "descriptor_semantics_not_object_detection",
        "semantic_descriptor_artifact",
        "excluded_category_keys",
        "maximum_footprint_area_m2",
        "maximum_footprint_span_m",
        "canonical_content_sha256",
        "count",
        "objects",
    }
    if not isinstance(semantic_binding, Mapping) or set(semantic_binding) != (
        expected_semantic_fields
    ):
        errors.append("inputs.semantic_object_footprints binding is invalid")
    else:
        objects = semantic_binding.get("objects")
        if not isinstance(objects, list) or not all(
            isinstance(item, Mapping) for item in objects
        ):
            errors.append("semantic object footprints list is invalid")
            objects = None
        if (
            semantic_binding.get("policy") != SEMANTIC_OBJECT_FOOTPRINT_POLICY
            or semantic_binding.get("source") != SEMANTIC_OBJECT_SOURCE
            or semantic_binding.get("descriptor_semantics_not_object_detection")
            is not True
            or semantic_binding.get("excluded_category_keys")
            != sorted(_STRUCTURAL_SEMANTIC_CATEGORIES)
            or semantic_binding.get("maximum_footprint_area_m2")
            != SEMANTIC_OBJECT_MAX_FOOTPRINT_AREA_M2
            or semantic_binding.get("maximum_footprint_span_m")
            != SEMANTIC_OBJECT_MAX_FOOTPRINT_SPAN_M
        ):
            errors.append("semantic object footprint policy is invalid")
        if objects is not None:
            try:
                normalized_objects = _coerce_semantic_object_footprints(objects)
            except TopdownReviewError as exc:
                errors.append(str(exc))
            else:
                verified_footprints = normalized_objects
                actual_footprints_hash = _recomputed_json_hash(
                    normalized_objects,
                    owner="semantic object footprints",
                    errors=errors,
                )
                if (
                    actual_footprints_hash is not None
                    and semantic_binding.get("canonical_content_sha256")
                    != actual_footprints_hash
                ):
                    errors.append("semantic object footprints canonical hash differs")
                if semantic_binding.get("count") != len(objects):
                    errors.append("semantic object footprint count differs")
        descriptor_name = semantic_binding.get("semantic_descriptor_artifact")
        if descriptor_name is None:
            if objects:
                errors.append(
                    "non-empty semantic object footprints lack a descriptor "
                    "artifact binding"
                )
        elif descriptor_name != "semantic_descriptor":
            errors.append("semantic descriptor artifact name is invalid")
        elif descriptor_name not in verified_inputs:
            errors.append("semantic descriptor artifact binding is not verified")

    navmesh_binding = _declared_input_binding(
        inputs_value, "navmesh_binary_map", errors
    )
    navmesh_bounds_value: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None
    navmesh_metadata_value: Mapping[str, Any] | None = None
    if navmesh_binding is not None:
        expected_navmesh_fields = {
            "hash_algorithm",
            "content_sha256",
            "shape",
            "dtype",
            "bounds_min_xyz",
            "bounds_max_xyz",
            "metadata",
            "metadata_content_sha256",
        }
        if set(navmesh_binding) != expected_navmesh_fields:
            errors.append("navmesh binary map binding fields are invalid")
        if navmesh_binding.get("hash_algorithm") != NAVMESH_HASH_ALGORITHM:
            errors.append("navmesh binary map hash algorithm differs")
        try:
            navmesh_bounds_value = _coerce_bounds(
                (
                    navmesh_binding.get("bounds_min_xyz"),
                    navmesh_binding.get("bounds_max_xyz"),
                )
            )
        except (TypeError, TopdownReviewError) as exc:
            errors.append(f"navmesh binary map bounds are invalid: {exc}")

        metadata_value = navmesh_binding.get("metadata")
        expected_metadata_fields = {
            "qa_id",
            "meters_per_pixel",
            "height_m",
            "navigable_pixel_count",
            "room_id",
        }
        if not isinstance(metadata_value, Mapping) or set(metadata_value) != (
            expected_metadata_fields
        ):
            errors.append("navmesh binary map metadata is invalid")
        else:
            navmesh_metadata_value = metadata_value
            metadata_hash = _recomputed_json_hash(
                metadata_value, owner="navmesh metadata", errors=errors
            )
            if (
                metadata_hash is not None
                and navmesh_binding.get("metadata_content_sha256") != metadata_hash
            ):
                errors.append("navmesh metadata canonical hash differs")
            if not isinstance(metadata_value.get("qa_id"), str) or not str(
                metadata_value.get("qa_id")
            ):
                errors.append("navmesh metadata qa_id is invalid")
            if not isinstance(metadata_value.get("room_id"), str) or not str(
                metadata_value.get("room_id")
            ):
                errors.append("navmesh metadata room_id is invalid")
            try:
                _finite_float(
                    metadata_value.get("meters_per_pixel"),
                    owner="navmesh meters_per_pixel",
                    minimum=0.0,
                    inclusive=False,
                )
                height_value = float(metadata_value.get("height_m"))
                if not math.isfinite(height_value):
                    raise TopdownReviewError("navmesh height_m must be finite")
                _positive_integer(
                    metadata_value.get("navigable_pixel_count"),
                    owner="navmesh navigable_pixel_count",
                )
            except (TypeError, ValueError, TopdownReviewError) as exc:
                errors.append(f"navmesh metadata values are invalid: {exc}")

        if saved_navmesh is not None:
            if (
                saved_navmesh.ndim != 2
                or saved_navmesh.size == 0
                or saved_navmesh.dtype.kind not in "biu"
            ):
                errors.append("saved navmesh binary map is not a non-empty integer HxW")
            else:
                unique_values = np.unique(saved_navmesh)
                if not np.all(np.isin(unique_values, (0, 1))):
                    errors.append("saved navmesh binary map contains non-binary values")
                if not np.any(saved_navmesh):
                    errors.append("saved navmesh binary map has no navigable pixels")
                navmesh_hash = _array_sequence_hash(
                    (saved_navmesh,), algorithm=NAVMESH_HASH_ALGORITHM
                )
                if navmesh_binding.get("content_sha256") != navmesh_hash:
                    errors.append("navmesh binary map hash differs from saved NPY")
                if navmesh_binding.get("shape") != list(saved_navmesh.shape):
                    errors.append("navmesh binary map shape differs from saved NPY")
                if navmesh_binding.get("dtype") != saved_navmesh.dtype.str:
                    errors.append("navmesh binary map dtype differs from saved NPY")
                if isinstance(metadata_value, Mapping) and metadata_value.get(
                    "navigable_pixel_count"
                ) != int(np.count_nonzero(saved_navmesh)):
                    errors.append(
                        "navmesh navigable pixel count differs from saved NPY"
                    )

        if (
            saved_navmesh is not None
            and navmesh_bounds_value is not None
            and focus_bounds_value is not None
            and focus_roi is not None
            and panel_size is not None
        ):
            try:
                _panel, rebuilt_projection = _prepare_panel(
                    saved_navmesh,
                    bounds=navmesh_bounds_value,
                    focus_bounds=focus_bounds_value,
                    panel_size_wh=panel_size,
                )
            except TopdownReviewError as exc:
                errors.append(f"top-down focus cannot be rebuilt: {exc}")
            else:
                if rebuilt_projection.navmesh_roi_rc != focus_roi:
                    errors.append("top-down focus ROI differs from saved navmesh map")

    actor_heading_binding = _declared_input_binding(
        inputs_value, "actor_heading", errors
    )
    rebuilt_actor_headings: tuple[np.ndarray, ...] | None = None
    if actor_heading_binding is not None:
        expected_actor_heading_fields = {
            "policy",
            "trusted_local_forward_axis",
            "trusted_forward_axis_source",
            "idle_policy",
            "movement_epsilon_m",
            "frame_count",
            "canonical_content_sha256",
        }
        if set(actor_heading_binding) != expected_actor_heading_fields:
            errors.append("actor heading binding fields are invalid")
        if actor_heading_binding.get("policy") != (
            qa_policy.get("actor_heading_policy")
            if isinstance(qa_policy, Mapping)
            else None
        ):
            errors.append("actor heading policy differs from QA policy")

    frame_binding = _declared_input_binding(inputs_value, "frame_records", errors)
    if frame_binding is not None and set(frame_binding) != {
        "canonical_content_sha256",
        "frame_count",
    }:
        errors.append("frame records binding fields are invalid")
    core_path = verified_inputs.get("core_capture_evidence")
    core_frames: list[Mapping[str, Any]] | None = None
    if core_path is not None:
        strict_core_errors = _strict_core_capture_errors(core_path, verified_inputs)
        errors.extend(
            f"core capture strict verification: {error}" for error in strict_core_errors
        )
        try:
            core = load_json(core_path)
        except (OSError, ValueError) as exc:
            errors.append(f"core_capture_evidence JSON is invalid: {exc}")
        else:
            frames = core.get("frames")
            if not isinstance(frames, list) or not all(
                isinstance(frame, Mapping) for frame in frames
            ):
                errors.append("core_capture_evidence.frames is invalid")
            else:
                core_frames = frames
                if timeline_frame_count is not None and len(frames) != (
                    timeline_frame_count
                ):
                    errors.append("core capture frame count differs from timeline")
                if frame_binding is not None:
                    actual = _recomputed_json_hash(
                        frames, owner="core capture frames", errors=errors
                    )
                    if actual is not None and (
                        frame_binding.get("canonical_content_sha256") != actual
                    ):
                        errors.append(
                            "frame records canonical hash differs from core capture"
                        )
                    if frame_binding.get("frame_count") != len(frames):
                        errors.append("frame record count differs from core capture")

    if actor_heading_binding is not None and core_frames is not None:
        policy = actor_heading_binding.get("policy")
        if policy == ACTOR_HEADING_TRAJECTORY_POLICY:
            trusted_axis = None
            trusted_source = None
        elif policy == ACTOR_HEADING_TRUSTED_AXIS_POLICY:
            trusted_axis = actor_heading_binding.get("trusted_local_forward_axis")
            trusted_source = actor_heading_binding.get("trusted_forward_axis_source")
        else:
            trusted_axis = None
            trusted_source = None
            errors.append("actor heading policy is unsupported")
        if policy in {
            ACTOR_HEADING_TRAJECTORY_POLICY,
            ACTOR_HEADING_TRUSTED_AXIS_POLICY,
        }:
            try:
                actor_matrices = tuple(
                    _actor_matrix(frame, index=index)
                    for index, frame in enumerate(core_frames)
                )
                rebuilt_actor_headings, expected_actor_binding = _actor_headings_xz(
                    actor_matrices,
                    trusted_local_forward_axis=trusted_axis,
                    trusted_forward_axis_source=trusted_source,
                )
            except TopdownReviewError as exc:
                errors.append(f"actor heading binding cannot be rebuilt: {exc}")
            else:
                if dict(actor_heading_binding) != expected_actor_binding:
                    errors.append("actor heading binding differs from core trajectory")

    room_binding = _declared_input_binding(inputs_value, "room_camera_request", errors)
    source_binding = _declared_input_binding(inputs_value, "source_anchors", errors)
    if room_binding is not None and set(room_binding) != {
        "canonical_content_sha256",
        "hfov_degrees",
    }:
        errors.append("room camera request binding fields are invalid")
    if source_binding is not None and set(source_binding) != {
        "canonical_content_sha256",
        "count",
    }:
        errors.append("source anchor binding fields are invalid")
    room_path = verified_inputs.get("room_request")
    loaded_room_request: Mapping[str, Any] | None = None
    loaded_sources: list[Mapping[str, Any]] | None = None
    if room_path is not None:
        try:
            room_request = load_json(room_path)
        except (OSError, ValueError) as exc:
            errors.append(f"room_request JSON is invalid: {exc}")
        else:
            loaded_room_request = room_request
            room_hash = _recomputed_json_hash(
                room_request, owner="room request", errors=errors
            )
            if (
                room_binding is not None
                and room_hash is not None
                and room_binding.get("canonical_content_sha256") != room_hash
            ):
                errors.append("room camera request canonical hash differs")
            sources = room_request.get("sources", [])
            if not isinstance(sources, list) or not all(
                isinstance(source, Mapping) for source in sources
            ):
                errors.append("room_request.sources is invalid")
            elif source_binding is not None:
                loaded_sources = sources
                source_hash = _recomputed_json_hash(
                    sources, owner="room request sources", errors=errors
                )
                if source_hash is not None and (
                    source_binding.get("canonical_content_sha256") != source_hash
                ):
                    errors.append(
                        "source anchors canonical hash differs from room request"
                    )
                if source_binding.get("count") != len(sources):
                    errors.append("source anchor count differs from room request")
            else:
                loaded_sources = sources
            try:
                _camera_matrix, rebuilt_hfov = _camera_contract(room_request)
            except TopdownReviewError as exc:
                errors.append(f"room camera contract is invalid: {exc}")
            else:
                if (
                    room_binding is not None
                    and room_binding.get("hfov_degrees") != rebuilt_hfov
                ):
                    errors.append("room camera HFOV binding differs")
            qa_views = room_request.get("qa_views")
            topdown_views = (
                [
                    item
                    for item in qa_views
                    if isinstance(item, Mapping) and item.get("kind") == "topdown"
                ]
                if isinstance(qa_views, list)
                else []
            )
            if len(topdown_views) != 1:
                errors.append("room request top-down QA declaration is invalid")
            elif navmesh_metadata_value is not None:
                declaration = topdown_views[0]
                for field in ("qa_id", "meters_per_pixel", "height_m"):
                    if navmesh_metadata_value.get(field) != declaration.get(field):
                        errors.append(
                            f"navmesh metadata {field} differs from room request"
                        )
                if navmesh_metadata_value.get("room_id") != room_request.get("room_id"):
                    errors.append("navmesh metadata room_id differs from room request")

    rgb_binding = _declared_input_binding(inputs_value, "rgb_frames", errors)
    if rgb_binding is not None and set(rgb_binding) != {
        "hash_algorithm",
        "content_sha256",
        "frame_count",
        "frame_shape",
        "dtype",
    }:
        errors.append("RGB frame binding fields are invalid")
    composite_binding = _declared_input_binding(
        inputs_value, "composite_frames", errors
    )
    if composite_binding is not None and set(composite_binding) != {
        "hash_algorithm",
        "content_sha256",
        "frame_count",
        "frame_shape",
        "dtype",
    }:
        errors.append("composite RGB frame binding fields are invalid")
    rgb_path = verified_inputs.get("rgb_array")
    loaded_rgb_frames: tuple[np.ndarray, ...] | None = None
    if rgb_path is not None:
        try:
            array = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
            frames = _coerce_rgb_frames(array)
        except (OSError, ValueError, TopdownReviewError) as exc:
            errors.append(f"rgb_array NPY is invalid: {exc}")
        else:
            loaded_rgb_frames = frames
            rgb_hash = _array_sequence_hash(frames, algorithm=RGB_STACK_HASH_ALGORITHM)
            if timeline_frame_count is not None and len(frames) != timeline_frame_count:
                errors.append("RGB frame count differs from timeline")
            if (
                main_size is not None
                and (
                    frames[0].shape[1],
                    frames[0].shape[0],
                )
                != main_size
            ):
                errors.append("main RGB layout size differs from rgb_array NPY")
            if rgb_binding is not None:
                if rgb_binding.get("hash_algorithm") != RGB_STACK_HASH_ALGORITHM:
                    errors.append("RGB stack hash algorithm differs")
                if rgb_binding.get("content_sha256") != rgb_hash:
                    errors.append("RGB stack hash differs from rgb_array NPY")
                if rgb_binding.get("frame_count") != len(frames):
                    errors.append("RGB frame count differs from rgb_array NPY")
                if rgb_binding.get("frame_shape") != list(frames[0].shape):
                    errors.append("RGB frame shape differs from rgb_array NPY")
                if rgb_binding.get("dtype") != frames[0].dtype.str:
                    errors.append("RGB dtype differs from rgb_array NPY")

    if (
        core_frames is not None
        and loaded_room_request is not None
        and loaded_sources is not None
        and navmesh_bounds_value is not None
        and saved_navmesh is not None
        and panel_size is not None
        and focus_value is not None
        and focus_margin is not None
        and minimum_focus_span is not None
        and loaded_rgb_frames is not None
        and verified_footprints is not None
        and rebuilt_actor_headings is not None
    ):
        try:
            rebuilt_actor_matrices = tuple(
                _actor_matrix(frame, index=index)
                for index, frame in enumerate(core_frames)
            )
            rebuilt_camera_matrix, _rebuilt_hfov = _camera_contract(loaded_room_request)
            rebuilt_focus_bounds = _automatic_focus_bounds(
                global_bounds=navmesh_bounds_value,
                actor_matrices=rebuilt_actor_matrices,
                camera_matrix=rebuilt_camera_matrix,
                source_anchors=loaded_sources,
                frame_count=len(core_frames),
                margin_m=focus_margin,
                minimum_span_m=minimum_focus_span,
            )
            _panel, rebuilt_projection = _prepare_panel(
                saved_navmesh,
                bounds=navmesh_bounds_value,
                focus_bounds=rebuilt_focus_bounds,
                panel_size_wh=panel_size,
            )
        except TopdownReviewError as exc:
            errors.append(f"top-down focus derivation cannot be rebuilt: {exc}")
        else:
            if focus_value.get("effective_bounds_min_xyz") != list(
                rebuilt_projection.bounds_min_xyz
            ) or focus_value.get("effective_bounds_max_xyz") != list(
                rebuilt_projection.bounds_max_xyz
            ):
                errors.append("top-down focus bounds differ from bound inputs")
            if focus_value.get("navmesh_roi_rc_exclusive") != list(
                rebuilt_projection.navmesh_roi_rc
            ):
                errors.append("top-down focus ROI differs from bound inputs")
            expected_composites: list[np.ndarray] = []
            try:
                for index, main_frame in enumerate(loaded_rgb_frames):
                    frame_number = int(core_frames[index].get("frame_index", index))
                    rebuilt_panel = _draw_panel_overlays(
                        _panel,
                        rebuilt_projection,
                        actor_matrices=rebuilt_actor_matrices,
                        actor_headings_xz=rebuilt_actor_headings,
                        camera_matrix=rebuilt_camera_matrix,
                        camera_hfov_degrees=_rebuilt_hfov,
                        source_anchors=loaded_sources,
                        semantic_object_footprints=verified_footprints,
                        frame_index=index,
                        frame_number=frame_number,
                    )
                    expected_composites.append(
                        _compose_frame(main_frame, rebuilt_panel)
                    )
            except (IndexError, TypeError, ValueError, TopdownReviewError) as exc:
                errors.append(f"composite RGB frames cannot be rebuilt: {exc}")
            else:
                composite_hash = _array_sequence_hash(
                    expected_composites,
                    algorithm=COMPOSITE_RGB_STACK_HASH_ALGORITHM,
                )
                if composite_binding is not None:
                    if composite_binding.get("hash_algorithm") != (
                        COMPOSITE_RGB_STACK_HASH_ALGORITHM
                    ):
                        errors.append("composite RGB hash algorithm differs")
                    if composite_binding.get("content_sha256") != composite_hash:
                        errors.append("composite RGB hash differs from bound inputs")
                    if composite_binding.get("frame_count") != len(expected_composites):
                        errors.append("composite RGB frame count differs")
                    if composite_binding.get("frame_shape") != list(
                        expected_composites[0].shape
                    ):
                        errors.append("composite RGB frame shape differs")
                    if composite_binding.get("dtype") != (
                        expected_composites[0].dtype.str
                    ):
                        errors.append("composite RGB dtype differs")
                if video_path is not None:
                    try:
                        decoded_rgb = _decode_review_video_rgb(
                            video_path,
                            width=expected_composites[0].shape[1],
                            height=expected_composites[0].shape[0],
                            frame_count=len(expected_composites),
                        )
                    except TopdownReviewError as exc:
                        errors.append(str(exc))
                    else:
                        expected_rgb = np.stack(expected_composites).astype(np.float32)
                        difference = np.abs(
                            decoded_rgb.astype(np.float32) - expected_rgb
                        )
                        maximum_frame_mae = float(
                            np.max(np.mean(difference, axis=(1, 2, 3)))
                        )
                        root_mean_squared_error = float(
                            np.sqrt(np.mean(np.square(difference)))
                        )
                        if (
                            maximum_frame_mae > VIDEO_READBACK_MAXIMUM_FRAME_MAE
                            or root_mean_squared_error > VIDEO_READBACK_MAXIMUM_RMSE
                        ):
                            errors.append(
                                "output.video pixels differ from rebuilt composite "
                                f"frames (max_frame_mae={maximum_frame_mae:.6f}, "
                                f"rmse={root_mean_squared_error:.6f})"
                            )
    return errors


__all__ = [
    "EVIDENCE_SCHEMA",
    "NAVMESH_HASH_ALGORITHM",
    "RGB_STACK_HASH_ALGORITHM",
    "SEMANTIC_OBJECT_FOOTPRINT_POLICY",
    "SEMANTIC_OBJECT_MAX_FOOTPRINT_AREA_M2",
    "SEMANTIC_OBJECT_MAX_FOOTPRINT_SPAN_M",
    "SEMANTIC_OBJECT_SOURCE",
    "TopdownReviewError",
    "compose_topdown_review",
    "encode_review_rgb_frames",
    "habitat_xz_to_navmesh_pixel",
    "render_topdown_panel",
    "semantic_object_footprint_from_obb",
    "verify_topdown_review_evidence",
]
