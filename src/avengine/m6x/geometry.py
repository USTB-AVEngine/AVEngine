"""Runtime-authoritative room-obstacle and source-center checks for M6.x.

The old Apartment review drew migrated, hand-selected AABBs while its route
gate used the same incomplete list.  This module deliberately takes a
different boundary: the loaded Habitat ``PathFinder`` is the authority for
the baked stage and every loaded rigid object's collision OBB is retained as
an additional obstacle.  The same snapshot is consumed by the point gate and
the Topdown renderer.

Only source-center points are checked.  No body capsule, articulated-link
volume, or full-body collision claim is made here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OBSTACLE_MAP_SCHEMA = "avengine_m6x_runtime_obstacle_map_v1"
SOURCE_CENTER_GATE_SCHEMA = "avengine_m6x_source_center_obstacle_gate_v1"


class M6XGeometryError(ValueError):
    """Runtime geometry could not be represented without ambiguity."""


def _finite_point(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise M6XGeometryError(f"{owner} must contain three finite numbers") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise M6XGeometryError(f"{owner} must contain three finite numbers")
    return result


def _positive_number(value: Any, *, owner: str, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise M6XGeometryError(f"{owner} must be a finite {qualifier} number")
    return float(value)


def _finite_number(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise M6XGeometryError(f"{owner} must be a finite number")
    return float(value)


def _convex_hull_xz(points: np.ndarray) -> list[list[float]]:
    """Return a deterministic counter-clockwise 2-D monotone-chain hull."""

    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.all(np.isfinite(value)):
        raise M6XGeometryError("footprint points must be finite [point,2]")
    unique = sorted({(float(point[0]), float(point[1])) for point in value})
    if len(unique) < 3:
        raise M6XGeometryError("collision footprint has fewer than three points")

    def cross(
        origin: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

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
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise M6XGeometryError("collision footprint is degenerate")
    return [[x, z] for x, z in hull]


def _object_world_obb(value: Any, mn: Any) -> dict[str, Any]:
    bounds = value.collision_shape_aabb
    lower = _finite_point(bounds.min, owner=f"{value.handle} collision AABB minimum")
    upper = _finite_point(bounds.max, owner=f"{value.handle} collision AABB maximum")
    if np.any(upper <= lower):
        raise M6XGeometryError(f"{value.handle} collision AABB is degenerate")

    transform = value.transformation
    local_corners = np.asarray(
        [
            (x, y, z)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    world_corners = np.asarray(
        [transform.transform_point(mn.Vector3(point)) for point in local_corners],
        dtype=np.float64,
    )
    if world_corners.shape != (8, 3) or not np.all(np.isfinite(world_corners)):
        raise M6XGeometryError(f"{value.handle} world collision corners are invalid")

    local_center = (lower + upper) * 0.5
    local_half = (upper - lower) * 0.5
    world_center = _finite_point(
        transform.transform_point(mn.Vector3(local_center)),
        owner=f"{value.handle} world collision center",
    )
    axes: list[list[float]] = []
    half_extents: list[float] = []
    for axis_index in range(3):
        unit = np.zeros(3, dtype=np.float64)
        unit[axis_index] = 1.0
        endpoint = _finite_point(
            transform.transform_point(mn.Vector3(local_center + unit)),
            owner=f"{value.handle} collision axis",
        )
        axis = endpoint - world_center
        scale = float(np.linalg.norm(axis))
        if not math.isfinite(scale) or scale <= 1.0e-12:
            raise M6XGeometryError(f"{value.handle} collision transform is singular")
        axes.append((axis / scale).tolist())
        half_extents.append(float(local_half[axis_index] * scale))
    axes_array = np.asarray(axes, dtype=np.float64)
    if not np.allclose(axes_array @ axes_array.T, np.eye(3), atol=1.0e-5):
        raise M6XGeometryError(
            f"{value.handle} collision transform contains unsupported shear"
        )

    return {
        "local_aabb_m": {"minimum": lower.tolist(), "maximum": upper.tolist()},
        "world_obb": {
            "center_m": world_center.tolist(),
            "axes_xyz": axes,
            "half_extents_m": half_extents,
        },
        "world_aabb_m": {
            "minimum": np.min(world_corners, axis=0).tolist(),
            "maximum": np.max(world_corners, axis=0).tolist(),
        },
        "world_corners_m": world_corners.tolist(),
        "footprint_xz_m": _convex_hull_xz(world_corners[:, (0, 2)]),
    }


def extract_loaded_rigid_obstacles(
    object_manager: Any,
    mn: Any,
    *,
    excluded_object_ids: Iterable[int] = (),
    excluded_handle_prefixes: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Read every loaded rigid collision OBB from the live Habitat scene.

    Scenario actors and legacy debug source markers should be excluded by ID
    or handle.  The function must be called after the fixed room is loaded and
    before scenario entities are injected whenever practical.
    """

    excluded_ids = {int(value) for value in excluded_object_ids}
    excluded_prefixes = tuple(str(value) for value in excluded_handle_prefixes)
    try:
        objects = object_manager.get_objects_by_handle_substring().values()
    except (AttributeError, TypeError) as exc:
        raise M6XGeometryError("Habitat rigid-object manager is unavailable") from exc
    records: list[dict[str, Any]] = []
    for value in sorted(objects, key=lambda item: str(item.handle).encode("utf-8")):
        object_id = int(value.object_id)
        handle = str(value.handle)
        if object_id in excluded_ids or any(
            handle.startswith(prefix) for prefix in excluded_prefixes
        ):
            continue
        record = {
            "object_id": object_id,
            "handle": handle,
            "source": "live_habitat_rigid_collision_shape",
        }
        record.update(_object_world_obb(value, mn))
        records.append(record)
    return records


@dataclass(frozen=True)
class RuntimeObstacleMap:
    """In-memory map shared by placement checks and diagnostic rendering."""

    binary_navmesh: np.ndarray
    bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    floor_height_m: float
    meters_per_pixel: float
    rigid_obstacles: tuple[Mapping[str, Any], ...]
    # Runtime-only authority reference.  It is intentionally absent from the
    # JSON summary, but lets the gate reject a different or subsequently
    # reloaded PathFinder instead of silently mixing two room geometries.
    _pathfinder: Any | None = field(default=None, repr=False, compare=False)

    def summary(self) -> dict[str, Any]:
        value = np.asarray(self.binary_navmesh, dtype=np.uint8)
        return {
            "schema": OBSTACLE_MAP_SCHEMA,
            "authority": (
                "live_habitat_declared_navmesh_plus_loaded_rigid_collision_obbs"
            ),
            "claim_boundary": (
                "source-center placement and Topdown only; no body-volume claim"
            ),
            "floor_height_m": self.floor_height_m,
            "meters_per_pixel": self.meters_per_pixel,
            "bounds_m": [list(item) for item in self.bounds_m],
            "navmesh_shape_hw": list(value.shape),
            "navigable_pixel_count": int(np.count_nonzero(value)),
            "rigid_obstacle_count": len(self.rigid_obstacles),
            "rigid_obstacles": [dict(item) for item in self.rigid_obstacles],
        }


def build_runtime_obstacle_map(
    pathfinder: Any,
    object_manager: Any,
    mn: Any,
    *,
    floor_height_m: float,
    meters_per_pixel: float = 0.02,
    excluded_object_ids: Iterable[int] = (),
    excluded_handle_prefixes: Iterable[str] = (),
) -> RuntimeObstacleMap:
    """Snapshot baked-stage navigation and separately loaded room objects."""

    # Habitat scenes may legitimately place the operating floor below world
    # Y=0, so this is finite rather than nonnegative.
    floor = _finite_number(floor_height_m, owner="floor_height_m")
    resolution = _positive_number(meters_per_pixel, owner="meters_per_pixel")
    if not bool(getattr(pathfinder, "is_loaded", False)):
        raise M6XGeometryError("Habitat PathFinder has no loaded navmesh")
    binary = np.asarray(pathfinder.get_topdown_view(resolution, floor), dtype=np.uint8)
    if (
        binary.ndim != 2
        or binary.size == 0
        or not np.any(binary)
        or np.any(~np.isin(binary, (0, 1)))
    ):
        raise M6XGeometryError("Habitat returned an invalid topdown navmesh")
    bounds_raw = np.asarray(pathfinder.get_bounds(), dtype=np.float64)
    if (
        bounds_raw.shape != (2, 3)
        or not np.all(np.isfinite(bounds_raw))
        or np.any(bounds_raw[1] <= bounds_raw[0])
    ):
        raise M6XGeometryError("Habitat returned invalid navmesh bounds")
    rigid = extract_loaded_rigid_obstacles(
        object_manager,
        mn,
        excluded_object_ids=excluded_object_ids,
        excluded_handle_prefixes=excluded_handle_prefixes,
    )
    return RuntimeObstacleMap(
        binary_navmesh=np.ascontiguousarray(binary),
        bounds_m=(tuple(bounds_raw[0]), tuple(bounds_raw[1])),
        floor_height_m=floor,
        meters_per_pixel=resolution,
        rigid_obstacles=tuple(rigid),
        _pathfinder=pathfinder,
    )


def _assert_pathfinder_matches_snapshot(
    pathfinder: Any, obstacle_map: RuntimeObstacleMap
) -> None:
    """Reject accidental gate/Topdown authority splits.

    Identity alone is insufficient because a caller can reuse one
    ``PathFinder`` object and load another navmesh into it.  Re-reading the
    inexpensive binary slice and bounds once per gate also detects that case.
    """

    if not isinstance(obstacle_map, RuntimeObstacleMap):
        raise M6XGeometryError("obstacle_map must be a RuntimeObstacleMap")
    if obstacle_map._pathfinder is None:
        raise M6XGeometryError(
            "source-center gate requires a live RuntimeObstacleMap built by "
            "build_runtime_obstacle_map"
        )
    if pathfinder is not obstacle_map._pathfinder:
        raise M6XGeometryError(
            "gate PathFinder differs from the RuntimeObstacleMap authority"
        )
    if not bool(getattr(pathfinder, "is_loaded", False)):
        raise M6XGeometryError("RuntimeObstacleMap PathFinder is no longer loaded")

    current_bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float64)
    retained_bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)
    if (
        current_bounds.shape != (2, 3)
        or not np.all(np.isfinite(current_bounds))
        or not np.allclose(current_bounds, retained_bounds, rtol=0.0, atol=1.0e-6)
    ):
        raise M6XGeometryError(
            "PathFinder bounds changed after RuntimeObstacleMap creation"
        )
    current_binary = np.asarray(
        pathfinder.get_topdown_view(
            obstacle_map.meters_per_pixel, obstacle_map.floor_height_m
        ),
        dtype=np.uint8,
    )
    retained_binary = np.asarray(obstacle_map.binary_navmesh, dtype=np.uint8)
    if current_binary.shape != retained_binary.shape or not np.array_equal(
        current_binary, retained_binary
    ):
        raise M6XGeometryError(
            "PathFinder navmesh changed after RuntimeObstacleMap creation"
        )


def point_to_world_obb_clearance(
    point_m: Sequence[float], obstacle: Mapping[str, Any]
) -> tuple[float, bool]:
    """Return exact point-to-OBB distance and whether the point is inside."""

    point = _finite_point(point_m, owner="source center")
    try:
        obb = obstacle["world_obb"]
        center = _finite_point(obb["center_m"], owner="OBB center")
        axes = np.asarray(obb["axes_xyz"], dtype=np.float64)
        half = np.asarray(obb["half_extents_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise M6XGeometryError("rigid obstacle OBB is invalid") from exc
    if (
        axes.shape != (3, 3)
        or half.shape != (3,)
        or not np.all(np.isfinite(axes))
        or not np.all(np.isfinite(half))
        or np.any(half <= 0.0)
        or not np.allclose(axes @ axes.T, np.eye(3), atol=1.0e-5)
    ):
        raise M6XGeometryError("rigid obstacle OBB is invalid")
    projected = axes @ (point - center)
    excess = np.maximum(np.abs(projected) - half, 0.0)
    inside = bool(np.all(np.abs(projected) <= half + 1.0e-9))
    return float(np.linalg.norm(excess)), inside


def evaluate_source_center_gate(
    pathfinder: Any,
    obstacle_map: RuntimeObstacleMap,
    trajectories_m: Mapping[str, Any],
    *,
    maximum_floor_snap_xz_m: float = 0.02,
    maximum_floor_y_delta_m: float = 0.25,
    minimum_navmesh_clearance_m: float = 0.0,
    minimum_rigid_clearance_m: float = 0.0,
) -> dict[str, Any]:
    """Check only source centers against the same live obstacle authority.

    The point's X/Z is projected onto the fixed operating floor for the
    PathFinder query.  Loaded rigid collision OBBs are checked in full 3-D.
    A point inside any OBB fails even when the configured clearance is zero.
    """

    _assert_pathfinder_matches_snapshot(pathfinder, obstacle_map)

    max_snap = _positive_number(
        maximum_floor_snap_xz_m, owner="maximum_floor_snap_xz_m", allow_zero=True
    )
    max_y = _positive_number(
        maximum_floor_y_delta_m, owner="maximum_floor_y_delta_m", allow_zero=True
    )
    min_nav = _positive_number(
        minimum_navmesh_clearance_m,
        owner="minimum_navmesh_clearance_m",
        allow_zero=True,
    )
    min_rigid = _positive_number(
        minimum_rigid_clearance_m,
        owner="minimum_rigid_clearance_m",
        allow_zero=True,
    )
    if not isinstance(trajectories_m, Mapping) or not trajectories_m:
        raise M6XGeometryError("at least one source-center trajectory is required")

    source_records: dict[str, Any] = {}
    for source_id in sorted(trajectories_m, key=lambda item: str(item).encode("utf-8")):
        if not isinstance(source_id, str) or not source_id:
            raise M6XGeometryError("source IDs must be nonempty strings")
        try:
            points = np.asarray(trajectories_m[source_id], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise M6XGeometryError(
                f"{source_id} trajectory must be finite [frame,3]"
            ) from exc
        if (
            points.ndim != 2
            or points.shape[0] < 1
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise M6XGeometryError(f"{source_id} trajectory must be finite [frame,3]")

        frames: list[dict[str, Any]] = []
        failed: list[int] = []
        nav_clearances: list[float] = []
        rigid_clearances: list[float] = []
        for frame_index, point in enumerate(points):
            floor_query = np.asarray(
                [point[0], obstacle_map.floor_height_m, point[2]],
                dtype=np.float64,
            )
            navigable = bool(pathfinder.is_navigable(floor_query, max_y))
            snapped = np.asarray(pathfinder.snap_point(floor_query), dtype=np.float64)
            if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
                raise M6XGeometryError("Habitat returned an invalid navmesh snap")
            snap_xz = float(np.linalg.norm(snapped[(0, 2),] - floor_query[(0, 2),]))
            nav_clearance = float(
                pathfinder.distance_to_closest_obstacle(snapped, 10.0)
            )
            if not math.isfinite(nav_clearance) or nav_clearance < 0.0:
                raise M6XGeometryError("Habitat returned invalid obstacle clearance")

            nearest_rigid: dict[str, Any] | None = None
            rigid_clearance = math.inf
            inside_rigid = False
            for obstacle in obstacle_map.rigid_obstacles:
                clearance, inside = point_to_world_obb_clearance(point, obstacle)
                if clearance < rigid_clearance or (
                    math.isclose(clearance, rigid_clearance) and inside
                ):
                    rigid_clearance = clearance
                    inside_rigid = inside
                    nearest_rigid = {
                        "object_id": obstacle.get("object_id"),
                        "handle": obstacle.get("handle"),
                    }
            rigid_pass = not inside_rigid and (
                math.isinf(rigid_clearance) or rigid_clearance >= min_rigid
            )
            passed = (
                navigable
                and snap_xz <= max_snap
                and nav_clearance >= min_nav
                and rigid_pass
            )
            if not passed:
                failed.append(frame_index)
            nav_clearances.append(nav_clearance)
            rigid_clearances.append(rigid_clearance)
            frames.append(
                {
                    "frame_index": frame_index,
                    "status": "pass" if passed else "fail",
                    "source_center_m": point.tolist(),
                    "floor_query_m": floor_query.tolist(),
                    "snapped_floor_m": snapped.tolist(),
                    "floor_navigable": navigable,
                    "floor_snap_xz_m": snap_xz,
                    "navmesh_clearance_m": nav_clearance,
                    "inside_loaded_rigid_obstacle": inside_rigid,
                    "nearest_rigid_obstacle": nearest_rigid,
                    "rigid_obstacle_clearance_m": (
                        None if math.isinf(rigid_clearance) else rigid_clearance
                    ),
                }
            )
        finite_rigid = [value for value in rigid_clearances if math.isfinite(value)]
        source_records[source_id] = {
            "status": "fail" if failed else "pass",
            "frame_count": len(frames),
            "failed_frame_indices": failed,
            "minimum_navmesh_clearance_m": min(nav_clearances),
            "minimum_loaded_rigid_clearance_m": (
                min(finite_rigid) if finite_rigid else None
            ),
            "frames": frames,
        }

    failed_sources = {
        source_id: record["failed_frame_indices"]
        for source_id, record in source_records.items()
        if record["status"] != "pass"
    }
    return {
        "schema": SOURCE_CENTER_GATE_SCHEMA,
        "status": "fail" if failed_sources else "pass",
        "authority": obstacle_map.summary(),
        "semantics": (
            "source_center_xz_vs_loaded_navmesh_and_source_center_xyz_vs_"
            "loaded_rigid_collision_obb"
        ),
        "full_body_collision_claim": False,
        "pathfinder_snapshot_match": True,
        "thresholds": {
            "maximum_floor_snap_xz_m": max_snap,
            "maximum_floor_y_delta_m": max_y,
            "minimum_navmesh_clearance_m": min_nav,
            "minimum_rigid_clearance_m": min_rigid,
        },
        "failed_source_frame_indices": failed_sources,
        "sources": source_records,
    }


__all__ = [
    "M6XGeometryError",
    "OBSTACLE_MAP_SCHEMA",
    "RuntimeObstacleMap",
    "SOURCE_CENTER_GATE_SCHEMA",
    "build_runtime_obstacle_map",
    "evaluate_source_center_gate",
    "extract_loaded_rigid_obstacles",
    "point_to_world_obb_clearance",
]
