"""Deterministic MP3D floor samples from one live Habitat PathFinder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

import numpy as np


class MP3DFloorSampleError(ValueError):
    """Live PathFinder or retained actor routes cannot support an export."""


def _number(value: Any, *, owner: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MP3DFloorSampleError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise MP3DFloorSampleError(f"{owner} must be finite and >= {minimum}")
    return result


def _point(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DFloorSampleError(f"{owner} must be one finite XYZ point") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise MP3DFloorSampleError(f"{owner} must be one finite XYZ point")
    return result


def _routes(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise MP3DFloorSampleError("actor routes must be a non-empty mapping")
    result = {}
    for actor_id in sorted(value, key=lambda item: str(item).encode("utf-8")):
        if not isinstance(actor_id, str) or not actor_id:
            raise MP3DFloorSampleError("actor route IDs must be non-empty strings")
        try:
            points = np.asarray(value[actor_id], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MP3DFloorSampleError(
                f"actor route {actor_id} must contain finite XYZ points"
            ) from exc
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) < 2
            or not np.all(np.isfinite(points))
        ):
            raise MP3DFloorSampleError(
                f"actor route {actor_id} must contain at least two finite XYZ points"
            )
        result[actor_id] = np.ascontiguousarray(points)
    return result


def _snap_record(
    pathfinder: Any,
    requested: np.ndarray,
    *,
    maximum_horizontal_error_m: float,
    maximum_vertical_error_m: float,
    minimum_clearance_m: float,
    required_island_id: int | None,
    authority: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "coordinate_space": "habitat_world_xyz_m",
        "authority": authority,
        "requested_xyz_m": requested.tolist(),
        "navigable": False,
        "snapped_xyz_m": None,
        "horizontal_snap_error_m": None,
        "vertical_snap_error_m": None,
        "island_id": None,
        "clearance_m": None,
        "status": "fail",
        "failure_reason": None,
    }
    try:
        navigable = bool(pathfinder.is_navigable(requested, maximum_vertical_error_m))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise MP3DFloorSampleError("PathFinder is_navigable failed") from exc
    record["navigable"] = navigable
    if not navigable:
        record["failure_reason"] = "not_navigable"
        return record
    try:
        snapped = np.asarray(pathfinder.snap_point(requested), dtype=np.float64)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise MP3DFloorSampleError("PathFinder snap_point failed") from exc
    if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
        raise MP3DFloorSampleError("PathFinder returned a non-finite snap point")
    horizontal = float(np.linalg.norm(snapped[[0, 2]] - requested[[0, 2]]))
    vertical = abs(float(snapped[1] - requested[1]))
    try:
        island_id = int(pathfinder.get_island(snapped))
        clearance = float(pathfinder.distance_to_closest_obstacle(snapped, 10.0))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise MP3DFloorSampleError("PathFinder island/clearance readback failed") from exc
    if not math.isfinite(clearance) or clearance < 0.0:
        raise MP3DFloorSampleError("PathFinder returned invalid obstacle clearance")
    record.update(
        {
            "snapped_xyz_m": snapped.tolist(),
            "horizontal_snap_error_m": horizontal,
            "vertical_snap_error_m": vertical,
            "island_id": island_id,
            "clearance_m": clearance,
        }
    )
    if horizontal > maximum_horizontal_error_m:
        record["failure_reason"] = "horizontal_snap_error"
    elif vertical > maximum_vertical_error_m:
        record["failure_reason"] = "vertical_snap_error"
    elif required_island_id is not None and island_id != required_island_id:
        record["failure_reason"] = "different_island"
    elif clearance < minimum_clearance_m:
        record["failure_reason"] = "insufficient_clearance"
    else:
        record["status"] = "pass"
    return record


def _uniform_subset_indices(count: int, maximum_count: int) -> tuple[int, ...]:
    if count <= maximum_count:
        return tuple(range(count))
    return tuple(
        sorted(
            {
                int(round(value))
                for value in np.linspace(0, count - 1, maximum_count)
            }
        )
    )


def _canonical_selected_records(
    records: Sequence[Mapping[str, Any]], maximum_count: int
) -> tuple[Mapping[str, Any], ...]:
    by_point: dict[tuple[float, float, float], Mapping[str, Any]] = {}
    for record in records:
        if record.get("status") != "pass":
            continue
        point = tuple(float(value) for value in record["snapped_xyz_m"])
        by_point.setdefault(point, record)
    ordered = [by_point[key] for key in sorted(by_point)]
    indices = _uniform_subset_indices(len(ordered), maximum_count)
    return tuple(ordered[index] for index in indices)


def export_live_mp3d_floor_samples(
    pathfinder: Any,
    *,
    room_id: str,
    actor_routes_m: Mapping[str, Sequence[Sequence[Real]]],
    meters_per_pixel: Real = 0.25,
    maximum_sample_count: int = 128,
    minimum_sample_count: int = 4,
    candidate_scan_multiplier: int = 16,
    maximum_horizontal_snap_error_m: Real = 0.10,
    maximum_vertical_snap_error_m: Real = 0.25,
    minimum_clearance_m: Real = 0.05,
    maximum_route_floor_height_span_m: Real = 0.50,
) -> dict[str, Any]:
    """Export a bounded floor sample set from live navmesh pixels and readback."""

    if not isinstance(room_id, str) or not room_id:
        raise MP3DFloorSampleError("room_id must be a non-empty string")
    for value, owner in (
        (maximum_sample_count, "maximum_sample_count"),
        (minimum_sample_count, "minimum_sample_count"),
        (candidate_scan_multiplier, "candidate_scan_multiplier"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise MP3DFloorSampleError(f"{owner} must be a positive integer")
    if minimum_sample_count > maximum_sample_count:
        raise MP3DFloorSampleError("minimum_sample_count cannot exceed maximum_sample_count")
    resolution = _number(meters_per_pixel, owner="meters_per_pixel", minimum=1.0e-6)
    max_horizontal = _number(
        maximum_horizontal_snap_error_m,
        owner="maximum_horizontal_snap_error_m",
        minimum=0.0,
    )
    max_vertical = _number(
        maximum_vertical_snap_error_m,
        owner="maximum_vertical_snap_error_m",
        minimum=0.0,
    )
    min_clearance = _number(
        minimum_clearance_m, owner="minimum_clearance_m", minimum=0.0
    )
    max_floor_span = _number(
        maximum_route_floor_height_span_m,
        owner="maximum_route_floor_height_span_m",
        minimum=0.0,
    )
    if not bool(getattr(pathfinder, "is_loaded", False)):
        raise MP3DFloorSampleError("Habitat PathFinder has no loaded navmesh")
    routes = _routes(actor_routes_m)
    route_records = []
    route_islands = set()
    route_snapped_points = []
    for actor_id, points in routes.items():
        for frame_index, point in enumerate(points):
            record = _snap_record(
                pathfinder,
                point,
                maximum_horizontal_error_m=max_horizontal,
                maximum_vertical_error_m=max_vertical,
                minimum_clearance_m=0.0,
                required_island_id=None,
                authority="retained_actor_route_live_pathfinder_readback",
            )
            record.update({"actor_id": actor_id, "frame_index": frame_index})
            if record["status"] != "pass":
                raise MP3DFloorSampleError(
                    f"actor route {actor_id} frame {frame_index} failed live PathFinder: "
                    f"{record['failure_reason']}"
                )
            route_records.append(record)
            route_islands.add(int(record["island_id"]))
            route_snapped_points.append(record["snapped_xyz_m"])
    if len(route_islands) != 1:
        raise MP3DFloorSampleError(
            f"retained actor routes cross PathFinder islands: {sorted(route_islands)}"
        )
    required_island = next(iter(route_islands))
    if required_island < 0:
        raise MP3DFloorSampleError("actor routes resolved to an invalid PathFinder island")
    snapped_routes = np.asarray(route_snapped_points, dtype=np.float64)
    floor_span = float(np.ptp(snapped_routes[:, 1]))
    if floor_span > max_floor_span:
        raise MP3DFloorSampleError(
            f"actor route floor-height span {floor_span} exceeds {max_floor_span}"
        )
    floor_height = float(np.median(snapped_routes[:, 1]))

    try:
        bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float64)
        topdown = np.asarray(
            pathfinder.get_topdown_view(resolution, floor_height), dtype=np.uint8
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise MP3DFloorSampleError("PathFinder topdown/bounds export failed") from exc
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[1] <= bounds[0])
    ):
        raise MP3DFloorSampleError("PathFinder bounds must be finite increasing XYZ")
    if (
        topdown.ndim != 2
        or topdown.size == 0
        or not np.any(topdown)
        or np.any(~np.isin(topdown, (0, 1)))
    ):
        raise MP3DFloorSampleError("PathFinder topdown must be a non-empty 0/1 map")
    rows, columns = np.nonzero(topdown)
    pixel_x = float((bounds[1, 0] - bounds[0, 0]) / topdown.shape[1])
    pixel_z = float((bounds[1, 2] - bounds[0, 2]) / topdown.shape[0])
    requested = np.stack(
        (
            bounds[0, 0] + (columns.astype(np.float64) + 0.5) * pixel_x,
            np.full(len(rows), floor_height, dtype=np.float64),
            bounds[0, 2] + (rows.astype(np.float64) + 0.5) * pixel_z,
        ),
        axis=1,
    )
    scan_limit = max(maximum_sample_count, maximum_sample_count * candidate_scan_multiplier)
    scan_indices = _uniform_subset_indices(len(requested), min(len(requested), scan_limit))
    sample_records = tuple(
        _snap_record(
            pathfinder,
            requested[index],
            maximum_horizontal_error_m=max_horizontal,
            maximum_vertical_error_m=max_vertical,
            minimum_clearance_m=min_clearance,
            required_island_id=required_island,
            authority="live_habitat_declared_navmesh_topdown_pixel_and_pathfinder_readback",
        )
        for index in scan_indices
    )
    selected = _canonical_selected_records(sample_records, maximum_sample_count)
    if len(selected) < minimum_sample_count:
        failures: dict[str, int] = {}
        for record in sample_records:
            reason = str(record.get("failure_reason") or "pass")
            failures[reason] = failures.get(reason, 0) + 1
        raise MP3DFloorSampleError(
            f"live PathFinder produced only {len(selected)} accepted floor samples; "
            f"need {minimum_sample_count}; outcomes={dict(sorted(failures.items()))}"
        )
    selected_points = [list(record["snapped_xyz_m"]) for record in selected]
    selected_set = {tuple(point) for point in selected_points}
    recorded_samples = []
    for record in sample_records:
        value = dict(record)
        snapped = value.get("snapped_xyz_m")
        value["selected_for_camera_floor_samples"] = (
            isinstance(snapped, list) and tuple(snapped) in selected_set
        )
        recorded_samples.append(value)
    return {
        "room_id": room_id,
        "coordinate_space": "habitat_world_xyz_m",
        "authority": "live_habitat_pathfinder_declared_navmesh_export",
        "research_only": True,
        "episode_counted": False,
        "native_habitat_floor_export": True,
        "native_habitat_pixel_status": "pending",
        "four_point_mini_pilot_status": "pending",
        "camera_floor_samples_m": selected_points,
        "machine_evidence": {
            "pathfinder_navmesh_loaded": True,
            "required_actor_route_island_id": required_island,
            "actor_route_readbacks": route_records,
            "actor_route_floor_height_median_m": floor_height,
            "actor_route_floor_height_span_m": floor_span,
            "navmesh_bounds_m": bounds.tolist(),
            "topdown_shape_hw": list(topdown.shape),
            "topdown_meters_per_pixel_requested": resolution,
            "topdown_pixel_size_x_m": pixel_x,
            "topdown_pixel_size_z_m": pixel_z,
            "topdown_navigable_pixel_count": int(len(requested)),
            "candidate_scan_count": len(sample_records),
            "selected_sample_count": len(selected_points),
            "thresholds": {
                "maximum_horizontal_snap_error_m": max_horizontal,
                "maximum_vertical_snap_error_m": max_vertical,
                "minimum_clearance_m": min_clearance,
                "maximum_route_floor_height_span_m": max_floor_span,
            },
            "sample_readbacks": recorded_samples,
        },
    }


__all__ = ["MP3DFloorSampleError", "export_live_mp3d_floor_samples"]
