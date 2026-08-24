"""Shared exact-frame materialization and raw two-source route descriptors.

This module deliberately separates route execution facts from downstream
difficulty labels.  It never rejects a pair because the sources approach,
cross, exchange endpoints, change listener-relative order, or leave a camera
field of view.  Backends still own their independent geometry/readback gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Sequence

import numpy as np

from avengine.m6x.trajectory import (
    M6XTrajectoryError,
    resample_polyline_by_arc_length,
)


class RouteDescriptorError(ValueError):
    """A planned route cannot be materialized or described faithfully."""


def _finite(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RouteDescriptorError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RouteDescriptorError(f"{owner} must be a finite number")
    return 0.0 if result == 0.0 else result


def _points(value: Any, *, owner: str, minimum_count: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise RouteDescriptorError(f"{owner} must be finite [point,3]") from error
    if (
        result.ndim != 2
        or result.shape[0] < minimum_count
        or result.shape[1] != 3
        or not np.all(np.isfinite(result))
    ):
        raise RouteDescriptorError(f"{owner} must be finite [point,3]")
    return np.ascontiguousarray(result)


def _normalized_yaw(value: float) -> float:
    result = (float(value) + 180.0) % 360.0 - 180.0
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class MaterializedRoute:
    """One explicit hold/straight/polyline expanded to exact frame positions."""

    geometry: str
    waypoints_m: np.ndarray
    positions_m: np.ndarray
    path_length_m: float
    segment_lengths_m: tuple[float, ...]
    segment_index_by_frame: tuple[int | None, ...]
    tangent_xy_by_frame: tuple[tuple[float, float] | None, ...]
    tangent_yaw_degrees_by_frame: tuple[float | None, ...]
    segment_boundaries: tuple[dict[str, Any], ...]
    moving: bool

    def summary(self) -> dict[str, Any]:
        return {
            "route_geometry": self.geometry,
            "moving": self.moving,
            "waypoints_m": self.waypoints_m.tolist(),
            "path_length_m": self.path_length_m,
            "segment_lengths_m": list(self.segment_lengths_m),
            "segment_index_by_frame": list(self.segment_index_by_frame),
            "tangent_yaw_degrees_by_frame": list(
                self.tangent_yaw_degrees_by_frame
            ),
            "segment_boundaries": [dict(item) for item in self.segment_boundaries],
            "materialization": "shared_exact_arc_length_samples",
        }


def materialize_route(
    waypoints_m: Any,
    sample_count: int,
    *,
    geometry: str | None = None,
    heading_yaw_degrees: Real | None = None,
    owner: str = "route",
) -> MaterializedRoute:
    """Materialize a hold, straight line, or multi-segment polyline.

    Moving routes use :func:`resample_polyline_by_arc_length`.  Segment
    membership and tangent direction are derived from the same cumulative arc
    lengths, so the replay layer does not infer headings from adjacent frames.
    """

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise RouteDescriptorError("sample_count must be a positive integer")
    requested = geometry
    if requested is not None and requested not in {"hold", "straight", "polyline"}:
        raise RouteDescriptorError(f"unsupported route geometry: {requested!r}")
    minimum = 1 if requested == "hold" else 2
    points = _points(waypoints_m, owner=owner, minimum_count=minimum)
    if requested is None:
        requested = "hold" if len(points) == 1 else ("straight" if len(points) == 2 else "polyline")
    if requested == "hold":
        if len(points) != 1:
            raise RouteDescriptorError("hold route must contain exactly one waypoint")
        yaw = None if heading_yaw_degrees is None else _normalized_yaw(
            _finite(heading_yaw_degrees, owner=f"{owner} heading_yaw_degrees")
        )
        positions = np.repeat(points, sample_count, axis=0)
        tangent = None if yaw is None else (
            math.cos(math.radians(yaw)),
            math.sin(math.radians(yaw)),
        )
        return MaterializedRoute(
            geometry="hold",
            waypoints_m=points,
            positions_m=np.ascontiguousarray(positions),
            path_length_m=0.0,
            segment_lengths_m=(),
            segment_index_by_frame=tuple(None for _ in range(sample_count)),
            tangent_xy_by_frame=tuple(tangent for _ in range(sample_count)),
            tangent_yaw_degrees_by_frame=tuple(yaw for _ in range(sample_count)),
            segment_boundaries=(),
            moving=False,
        )
    if requested == "straight" and len(points) != 2:
        raise RouteDescriptorError("straight route must contain exactly two waypoints")
    if requested == "polyline" and len(points) < 3:
        raise RouteDescriptorError("polyline route must contain at least three waypoints")

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    planar_lengths = np.linalg.norm(deltas[:, :2], axis=1)
    if np.any(lengths <= 1.0e-9) or np.any(planar_lengths <= 1.0e-9):
        raise RouteDescriptorError(
            f"{owner} consecutive waypoints must differ in the route plane"
        )
    total = float(np.sum(lengths))
    try:
        positions = resample_polyline_by_arc_length(
            points, sample_count, owner=owner
        )
    except M6XTrajectoryError as error:
        raise RouteDescriptorError(str(error)) from error
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, total, sample_count)
    segment_indices = np.searchsorted(cumulative, targets, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(lengths) - 1)
    tangents = deltas[:, :2] / planar_lengths[:, None]
    tangent_by_frame = tuple(
        (float(tangents[index, 0]), float(tangents[index, 1]))
        for index in segment_indices
    )
    yaw_by_frame = tuple(
        _normalized_yaw(math.degrees(math.atan2(value[1], value[0])))
        for value in tangent_by_frame
    )
    boundaries: list[dict[str, Any]] = []
    for waypoint_index, distance in enumerate(cumulative):
        sample_position = (
            0.0 if total == 0.0 else float(distance / total * (sample_count - 1))
        )
        nearest = int(round(sample_position))
        exact_frame = (
            nearest if abs(sample_position - nearest) <= 1.0e-9 else None
        )
        boundaries.append(
            {
                "waypoint_index": waypoint_index,
                "arc_distance_m": float(distance),
                "sample_position": sample_position,
                "lower_frame_index": int(math.floor(sample_position)),
                "upper_frame_index": int(math.ceil(sample_position)),
                "exact_frame_index": exact_frame,
            }
        )
    return MaterializedRoute(
        geometry=requested,
        waypoints_m=points,
        positions_m=positions,
        path_length_m=total,
        segment_lengths_m=tuple(float(value) for value in lengths),
        segment_index_by_frame=tuple(int(value) for value in segment_indices),
        tangent_xy_by_frame=tangent_by_frame,
        tangent_yaw_degrees_by_frame=yaw_by_frame,
        segment_boundaries=tuple(boundaries),
        moving=True,
    )


def _continuous_closest(
    left: np.ndarray, right: np.ndarray, *, fps: float
) -> dict[str, Any]:
    best: tuple[float, int, float] | None = None
    for frame_index in range(len(left) - 1):
        relative_start = left[frame_index] - right[frame_index]
        relative_delta = (left[frame_index + 1] - left[frame_index]) - (
            right[frame_index + 1] - right[frame_index]
        )
        denominator = float(np.dot(relative_delta, relative_delta))
        fraction = 0.0 if denominator <= 1.0e-15 else min(
            1.0,
            max(0.0, -float(np.dot(relative_start, relative_delta)) / denominator),
        )
        distance = float(np.linalg.norm(relative_start + fraction * relative_delta))
        candidate = (distance, frame_index, fraction)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        distance = float(np.linalg.norm(left[0] - right[0]))
        best = (distance, 0, 0.0)
    distance, frame_index, fraction = best
    frame_position = float(frame_index + fraction)
    return {
        "distance_m": distance,
        "frame_position": frame_position,
        "time_seconds": frame_position / fps,
        "lower_frame_index": frame_index,
        "upper_frame_index": min(frame_index + 1, len(left) - 1),
        "interval_fraction": fraction,
    }


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _path_intersections(left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for left_index, (a0, a1) in enumerate(zip(left, left[1:])):
        p = np.asarray(a0[:2], dtype=np.float64)
        r = np.asarray(a1[:2], dtype=np.float64) - p
        for right_index, (b0, b1) in enumerate(zip(right, right[1:])):
            q = np.asarray(b0[:2], dtype=np.float64)
            s = np.asarray(b1[:2], dtype=np.float64) - q
            denominator = _cross_2d(r, s)
            offset = q - p
            if abs(denominator) <= 1.0e-12:
                if abs(_cross_2d(offset, r)) > 1.0e-12:
                    continue
                r_norm = float(np.dot(r, r))
                if r_norm <= 1.0e-15:
                    continue
                lower, upper = sorted(
                    (float(np.dot(offset, r) / r_norm), float(np.dot(q + s - p, r) / r_norm))
                )
                overlap_start = max(0.0, lower)
                overlap_end = min(1.0, upper)
                if overlap_start <= overlap_end + 1.0e-12:
                    events.append(
                        {
                            "kind": "collinear_overlap",
                            "source1_segment_index": left_index,
                            "source2_segment_index": right_index,
                            "source1_fraction_interval": [overlap_start, overlap_end],
                        }
                    )
                continue
            left_fraction = _cross_2d(offset, s) / denominator
            right_fraction = _cross_2d(offset, r) / denominator
            if (
                -1.0e-12 <= left_fraction <= 1.0 + 1.0e-12
                and -1.0e-12 <= right_fraction <= 1.0 + 1.0e-12
            ):
                point = p + left_fraction * r
                events.append(
                    {
                        "kind": "point",
                        "source1_segment_index": left_index,
                        "source2_segment_index": right_index,
                        "source1_fraction": float(left_fraction),
                        "source2_fraction": float(right_fraction),
                        "position_xy_m": point.tolist(),
                    }
                )
    return events


def describe_route_pair(
    source1: MaterializedRoute,
    source2: MaterializedRoute,
    *,
    fps: Real,
    listener_position_xy_m: Sequence[Real] | None = None,
    listener_yaw_degrees: Real | None = None,
    proximity_reference_m: Real = 0.60,
) -> dict[str, Any]:
    """Return raw pair geometry and listener-relative clues without filtering."""

    rate = _finite(fps, owner="fps")
    if rate <= 0.0:
        raise RouteDescriptorError("fps must be positive")
    reference = _finite(proximity_reference_m, owner="proximity_reference_m")
    if reference < 0.0:
        raise RouteDescriptorError("proximity_reference_m cannot be negative")
    left = np.asarray(source1.positions_m, dtype=np.float64)
    right = np.asarray(source2.positions_m, dtype=np.float64)
    if left.shape != right.shape:
        raise RouteDescriptorError("source routes must have the same frame shape")
    planar = np.linalg.norm(left[:, :2] - right[:, :2], axis=1)
    root = np.linalg.norm(left - right, axis=1)
    closest_index = int(np.argmin(planar))
    continuous = _continuous_closest(left[:, :2], right[:, :2], fps=rate)
    source1_to_source2_start = float(
        np.linalg.norm(left[-1, :2] - right[0, :2])
    )
    source2_to_source1_start = float(
        np.linalg.norm(right[-1, :2] - left[0, :2])
    )
    swap_error = source1_to_source2_start + source2_to_source1_start
    intersections = _path_intersections(source1.waypoints_m, source2.waypoints_m)
    result: dict[str, Any] = {
        "frame_count": len(left),
        "pair_planar_distance_m_by_frame": planar.tolist(),
        "pair_root_distance_m_by_frame": root.tolist(),
        "closest_discrete": {
            "frame_index": closest_index,
            "time_seconds": closest_index / rate,
            "planar_distance_m": float(planar[closest_index]),
            "root_distance_m": float(root[closest_index]),
        },
        "closest_continuous_plan": continuous,
        "proximity_reference": {
            "distance_m": reference,
            "role": "raw_difficulty_descriptor_only_not_legality_gate",
            "below_reference_frame_indices": np.flatnonzero(
                planar < reference
            ).astype(int).tolist(),
        },
        "endpoint_exchange": {
            "source1_end_to_source2_start_m": source1_to_source2_start,
            "source2_end_to_source1_start_m": source2_to_source1_start,
            "total_swap_error_m": swap_error,
            "exact_endpoint_exchange": swap_error <= 1.0e-9,
        },
        "path_crossing": {
            "spatial_path_intersection": bool(intersections),
            "intersection_events": intersections,
            "continuous_center_overlap": continuous["distance_m"] <= 1.0e-9,
        },
        "identity_continuity": {
            "source1_assignment_changes": False,
            "source2_assignment_changes": False,
            "endpoint_exchange_does_not_swap_source_or_actor_ids": True,
        },
    }
    if (listener_position_xy_m is None) != (listener_yaw_degrees is None):
        raise RouteDescriptorError(
            "listener position and yaw must either both be present or both be absent"
        )
    if listener_position_xy_m is None:
        result["listener_relative"] = None
        return result
    listener = np.asarray(
        [_finite(value, owner="listener_position_xy_m") for value in listener_position_xy_m],
        dtype=np.float64,
    )
    if listener.shape != (2,):
        raise RouteDescriptorError("listener_position_xy_m must contain two numbers")
    yaw = _finite(listener_yaw_degrees, owner="listener_yaw_degrees")
    radians = math.radians(yaw)
    right_axis = np.asarray([-math.sin(radians), math.cos(radians)])
    left_right = (left[:, :2] - listener) @ right_axis
    right_right = (right[:, :2] - listener) @ right_axis
    delta = left_right - right_right
    order = [
        "same_right_coordinate"
        if abs(value) <= 1.0e-9
        else ("source1_right_of_source2" if value > 0.0 else "source1_left_of_source2")
        for value in delta
    ]
    nonzero_signs = np.sign(delta[np.abs(delta) > 1.0e-9])
    changed = bool(
        len(nonzero_signs) > 1 and np.any(nonzero_signs[1:] != nonzero_signs[:-1])
    )

    def bearings(points: np.ndarray) -> list[float]:
        values = []
        for point in points[:, :2]:
            offset = point - listener
            absolute = math.degrees(math.atan2(float(offset[1]), float(offset[0])))
            values.append(_normalized_yaw(absolute - yaw))
        return values

    result["listener_relative"] = {
        "listener_position_xy_m": listener.tolist(),
        "listener_yaw_degrees": yaw,
        "source1_relative_bearing_degrees_by_frame": bearings(left),
        "source2_relative_bearing_degrees_by_frame": bearings(right),
        "source1_right_coordinate_m_by_frame": left_right.tolist(),
        "source2_right_coordinate_m_by_frame": right_right.tolist(),
        "right_order_by_frame": order,
        "right_order_changed": changed,
        "right_order_transition_frame_indices": [
            index
            for index in range(1, len(order))
            if order[index] != order[index - 1]
        ],
        "role": "raw_descriptor_only_not_legality_gate",
    }
    return result


__all__ = [
    "MaterializedRoute",
    "RouteDescriptorError",
    "describe_route_pair",
    "materialize_route",
]
