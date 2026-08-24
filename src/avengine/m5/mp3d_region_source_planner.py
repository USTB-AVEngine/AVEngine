"""CPU MP3D region-specific source/path route planning.

The camera bank is an input binding only.  Route generation is driven by the
declared .house polygons and a real Habitat PathFinder loaded from the declared
MP3D navmesh.  No camera yaw, pixels, semantic target, visibility or difficulty
value is consulted while choosing source points or routes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from avengine.m5.mp3d_regions import MP3DHouseFloorPlan, MP3DRegion, parse_mp3d_house
from avengine.m6x.route_descriptors import (
    MaterializedRoute,
    RouteDescriptorError,
    describe_route_pair,
    materialize_route,
)


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
MOTION_CASES = (
    "static_static",
    "s1_moving_s2_static",
    "s1_static_s2_moving",
    "both_moving",
)
PROXIMITY_REFERENCE_M = 0.60
POINT_EPSILON_M = 1.0e-7
BOUNDARY_EPSILON_M = 1.0e-8


class MP3DRegionSourcePlannerError(ValueError):
    """The declared MP3D region route inputs cannot be planned truthfully."""


@dataclass(frozen=True)
class CameraBinding:
    region_instance_id: str
    region_index: int
    selection_order: int
    placement_id: str
    floor_position_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    yaw_deg: float
    height_id: str
    request_path: str


@dataclass(frozen=True)
class NativePoint:
    position_m: np.ndarray
    island_id: int
    clearance_m: float


@dataclass(frozen=True)
class MovingRoute:
    waypoints_m: np.ndarray
    planar_route: MaterializedRoute
    positions_m: np.ndarray
    pathfinder_record: Mapping[str, Any]


@dataclass(frozen=True)
class RouteFamily:
    family_id: str
    source1: MovingRoute
    source2: MovingRoute
    exchange_capable: bool
    geometry_signature: tuple[Any, ...]


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool):
        raise MP3DRegionSourcePlannerError(f"{owner} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionSourcePlannerError(f"{owner} must be finite") from exc
    if not math.isfinite(result):
        raise MP3DRegionSourcePlannerError(f"{owner} must be finite")
    return 0.0 if result == 0.0 else result


def _finite_xyz(value: Any, *, owner: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MP3DRegionSourcePlannerError(f"{owner} must contain three numbers")
    if len(value) != 3:
        raise MP3DRegionSourcePlannerError(f"{owner} must contain three numbers")
    return tuple(
        _finite_number(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _load_json(path: str | Path, *, owner: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise MP3DRegionSourcePlannerError(f"{owner} is not a file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MP3DRegionSourcePlannerError(f"{owner} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise MP3DRegionSourcePlannerError(f"{owner} must be a JSON object")
    return value


def _point_on_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    cross = (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (
        point[0] - first[0]
    )
    if abs(cross) > BOUNDARY_EPSILON_M:
        return False
    return (
        min(first[0], second[0]) - BOUNDARY_EPSILON_M
        <= point[0]
        <= max(first[0], second[0]) + BOUNDARY_EPSILON_M
        and min(first[1], second[1]) - BOUNDARY_EPSILON_M
        <= point[1]
        <= max(first[1], second[1]) + BOUNDARY_EPSILON_M
    )


def _point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        if (current[1] > point[1]) != (previous[1] > point[1]):
            denominator = previous[1] - current[1]
            crossing_x = (previous[0] - current[0]) * (
                point[1] - current[1]
            ) / denominator + current[0]
            if point[0] < crossing_x:
                inside = not inside
        previous = current
    return inside


def _region_contains(region: MP3DRegion, point_m: Sequence[float]) -> bool:
    point = np.asarray(point_m, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return False
    y = float(point[1])
    if (
        y < region.habitat_bbox_min_xyz_m[1] - 0.30
        or y > region.habitat_bbox_max_xyz_m[1] + 0.30
    ):
        return False
    horizontal = (float(point[0]), float(point[2]))
    for floor_polygon in region.floor_polygons:
        if (
            y < floor_polygon.floor_elevation_m - 0.30
            or y > floor_polygon.floor_elevation_m + 0.30
        ):
            continue
        if _point_in_polygon(horizontal, floor_polygon.horizontal_xz_m):
            return True
    return False


def _path_signature(points_m: np.ndarray) -> tuple[tuple[float, float], ...]:
    points = np.round(np.asarray(points_m, dtype=np.float64)[:, (0, 2)], 6)
    forward = tuple((float(row[0]), float(row[1])) for row in points)
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


def _pair_signature(
    source1: MovingRoute, source2: MovingRoute
) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(
        sorted(
            (_path_signature(source1.waypoints_m), _path_signature(source2.waypoints_m))
        )
    )


def _region_floor_height(region: MP3DRegion) -> float:
    values = [polygon.floor_elevation_m for polygon in region.floor_polygons]
    if not values:
        raise MP3DRegionSourcePlannerError(
            f"region {region.region_instance_id} has no floor polygon"
        )
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _pathfinder_bounds(pathfinder: Any) -> np.ndarray:
    try:
        bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float64)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise MP3DRegionSourcePlannerError(
            "Habitat PathFinder did not expose finite get_bounds()"
        ) from exc
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise MP3DRegionSourcePlannerError("PathFinder bounds are not [2,3]")
    if np.any(bounds[1] <= bounds[0]):
        raise MP3DRegionSourcePlannerError("PathFinder bounds are not increasing")
    return bounds


def _snap_navigable(
    pathfinder: Any,
    point_m: np.ndarray,
    *,
    maximum_y_delta_m: float,
) -> np.ndarray | None:
    try:
        if not bool(pathfinder.is_navigable(point_m, maximum_y_delta_m)):
            return None
        snapped = np.asarray(pathfinder.snap_point(point_m), dtype=np.float64)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise MP3DRegionSourcePlannerError(
            "Habitat PathFinder navigation query failed"
        ) from exc
    if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
        raise MP3DRegionSourcePlannerError("PathFinder returned an invalid snap point")
    return np.ascontiguousarray(snapped)


def _native_points_for_region(
    *,
    pathfinder: Any,
    region: MP3DRegion,
    sample_spacing_m: float,
    maximum_y_delta_m: float,
) -> tuple[NativePoint, ...]:
    if sample_spacing_m <= 0.0 or not math.isfinite(sample_spacing_m):
        raise MP3DRegionSourcePlannerError("sample_spacing_m must be positive")
    floor_y = _region_floor_height(region)
    ranges: list[tuple[float, float, float, float]] = []
    for polygon in region.floor_polygons:
        points = np.asarray(polygon.horizontal_xz_m, dtype=np.float64)
        ranges.append(
            (
                float(np.min(points[:, 0])),
                float(np.max(points[:, 0])),
                float(np.min(points[:, 1])),
                float(np.max(points[:, 1])),
            )
        )
    candidates: dict[tuple[float, float, int], NativePoint] = {}
    for min_x, max_x, min_z, max_z in ranges:
        x_values = np.arange(min_x, max_x + sample_spacing_m * 0.5, sample_spacing_m)
        z_values = np.arange(min_z, max_z + sample_spacing_m * 0.5, sample_spacing_m)
        for x in x_values:
            for z in z_values:
                requested = np.asarray([x, floor_y, z], dtype=np.float64)
                if not _region_contains(region, requested):
                    continue
                snapped = _snap_navigable(
                    pathfinder,
                    requested,
                    maximum_y_delta_m=maximum_y_delta_m,
                )
                if snapped is None or not _region_contains(region, snapped):
                    continue
                try:
                    island_id = int(pathfinder.get_island(snapped))
                    clearance = float(
                        pathfinder.distance_to_closest_obstacle(snapped, 10.0)
                    )
                except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                    raise MP3DRegionSourcePlannerError(
                        "PathFinder did not provide island/clearance readback"
                    ) from exc
                if not math.isfinite(clearance):
                    raise MP3DRegionSourcePlannerError(
                        "PathFinder returned non-finite obstacle clearance"
                    )
                key = (
                    round(float(snapped[0]), 6),
                    round(float(snapped[2]), 6),
                    island_id,
                )
                candidates[key] = NativePoint(
                    position_m=snapped,
                    island_id=island_id,
                    clearance_m=clearance,
                )
    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            item.island_id,
            round(float(item.position_m[0]), 6),
            round(float(item.position_m[2]), 6),
        ),
    )
    if len(ordered) < 2:
        raise MP3DRegionSourcePlannerError(
            f"region {region.region_instance_id} has fewer than two native "
            "PathFinder points inside its .house polygon"
        )
    return tuple(ordered)


def _deduplicate_waypoints(points_m: Any) -> np.ndarray:
    try:
        points = np.asarray(points_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionSourcePlannerError(
            "PathFinder returned invalid waypoints"
        ) from exc
    if (
        points.ndim != 2
        or points.shape[0] < 2
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise MP3DRegionSourcePlannerError("PathFinder returned invalid waypoints")
    kept = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - kept[-1])) > POINT_EPSILON_M:
            kept.append(point)
    result = np.asarray(kept, dtype=np.float64)
    if result.shape[0] < 2:
        raise MP3DRegionSourcePlannerError("PathFinder returned a zero-length path")
    return np.ascontiguousarray(result)


def _materialize_path(
    *,
    pathfinder: Any,
    shortest_path_factory: Callable[[], Any],
    region: MP3DRegion,
    start: NativePoint,
    end: NativePoint,
    expected_island_id: int,
) -> MovingRoute | None:
    query = shortest_path_factory()
    query.requested_start = np.asarray(start.position_m, dtype=np.float64)
    query.requested_end = np.asarray(end.position_m, dtype=np.float64)
    try:
        found = bool(pathfinder.find_path(query))
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise MP3DRegionSourcePlannerError(
            "Habitat PathFinder find_path failed"
        ) from exc
    if not found:
        return None
    waypoints = _deduplicate_waypoints(getattr(query, "points", None))
    geodesic = _finite_number(
        getattr(query, "geodesic_distance", math.inf),
        owner="PathFinder geodesic_distance",
    )
    if geodesic <= POINT_EPSILON_M:
        return None
    if any(
        not _region_contains(region, point)
        or _snap_navigable(pathfinder, np.asarray(point), maximum_y_delta_m=0.30)
        is None
        or int(pathfinder.get_island(np.asarray(point))) != expected_island_id
        for point in waypoints
    ):
        return None
    planar_waypoints = np.asarray(
        [[float(point[0]), float(point[2]), 0.0] for point in waypoints],
        dtype=np.float64,
    )
    try:
        planar_route = materialize_route(
            planar_waypoints,
            FRAME_COUNT,
            geometry="polyline" if len(planar_waypoints) >= 3 else "straight",
            owner="MP3D region PathFinder polyline",
        )
    except RouteDescriptorError as exc:
        raise MP3DRegionSourcePlannerError(str(exc)) from exc
    planar_cumulative = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float64),
            np.cumsum(
                np.linalg.norm(np.diff(planar_waypoints[:, (0, 2)], axis=0), axis=1)
            ),
        )
    )
    target = np.linspace(0.0, float(planar_cumulative[-1]), FRAME_COUNT)
    world_y = np.interp(target, planar_cumulative, waypoints[:, 1])
    positions = np.column_stack(
        (planar_route.positions_m[:, 0], world_y, planar_route.positions_m[:, 1])
    )
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    for point in positions:
        if (
            not _region_contains(region, point)
            or _snap_navigable(pathfinder, point, maximum_y_delta_m=0.30) is None
            or int(pathfinder.get_island(point)) != expected_island_id
        ):
            return None
    pathfinder_record = {
        "status": "pass",
        "path_exists": True,
        "all_waypoints_navigable": True,
        "all_materialized_frames_navigable": True,
        "region_membership": "all_frames_inside_house_polygon",
        "island_id": expected_island_id,
        "geodesic_distance_m": geodesic,
        "waypoint_count": int(len(waypoints)),
    }
    return MovingRoute(
        waypoints_m=waypoints,
        planar_route=planar_route,
        positions_m=positions,
        pathfinder_record=pathfinder_record,
    )


def _reverse_route(
    *,
    pathfinder: Any,
    shortest_path_factory: Callable[[], Any],
    region: MP3DRegion,
    route: MovingRoute,
    expected_island_id: int,
) -> MovingRoute | None:
    del pathfinder, shortest_path_factory, region, expected_island_id
    reversed_waypoints = np.ascontiguousarray(route.waypoints_m[::-1], dtype=np.float64)
    planar_waypoints = np.asarray(
        [[float(point[0]), float(point[2]), 0.0] for point in reversed_waypoints],
        dtype=np.float64,
    )
    try:
        planar_route = materialize_route(
            planar_waypoints,
            FRAME_COUNT,
            geometry="polyline" if len(planar_waypoints) >= 3 else "straight",
            owner="MP3D region reversed PathFinder polyline",
        )
    except RouteDescriptorError as exc:
        raise MP3DRegionSourcePlannerError(str(exc)) from exc
    cumulative = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float64),
            np.cumsum(
                np.linalg.norm(np.diff(planar_waypoints[:, (0, 2)], axis=0), axis=1)
            ),
        )
    )
    target = np.linspace(0.0, float(cumulative[-1]), FRAME_COUNT)
    world_y = np.interp(target, cumulative, reversed_waypoints[:, 1])
    positions = np.column_stack(
        (planar_route.positions_m[:, 0], world_y, planar_route.positions_m[:, 1])
    )
    return MovingRoute(
        waypoints_m=reversed_waypoints,
        planar_route=planar_route,
        positions_m=np.ascontiguousarray(positions, dtype=np.float64),
        pathfinder_record={
            "status": "pass",
            "path_exists": True,
            "all_waypoints_navigable": True,
            "all_materialized_frames_navigable": True,
            "region_membership": "inherited_from_forward_path_reversed",
            "island_id": route.pathfinder_record["island_id"],
            "geodesic_distance_m": route.pathfinder_record["geodesic_distance_m"],
            "waypoint_count": route.pathfinder_record["waypoint_count"],
        },
    )


def _hold_route(point: NativePoint) -> tuple[MaterializedRoute, np.ndarray]:
    planar = np.asarray([[float(point.position_m[0]), float(point.position_m[2]), 0.0]])
    try:
        route = materialize_route(
            planar, FRAME_COUNT, geometry="hold", owner="MP3D region hold"
        )
    except RouteDescriptorError as exc:
        raise MP3DRegionSourcePlannerError(str(exc)) from exc
    return route, np.repeat(point.position_m[None, :], FRAME_COUNT, axis=0)


def _route_pair_record(
    *,
    route1: MaterializedRoute,
    route2: MaterializedRoute,
    positions1: np.ndarray,
    positions2: np.ndarray,
) -> dict[str, Any]:
    try:
        descriptor = describe_route_pair(
            route1,
            route2,
            fps=FRAME_RATE_HZ,
            proximity_reference_m=PROXIMITY_REFERENCE_M,
        )
    except RouteDescriptorError as exc:
        raise MP3DRegionSourcePlannerError(str(exc)) from exc
    endpoint = descriptor["endpoint_exchange"]
    initial_distance = float(np.linalg.norm(positions1[0] - positions2[0]))
    final_distance = float(np.linalg.norm(positions1[-1] - positions2[-1]))
    endpoint["initial_pair_distance_m"] = initial_distance
    endpoint["final_pair_distance_m"] = final_distance
    endpoint["exact_endpoint_exchange"] = bool(
        endpoint["total_swap_error_m"] <= POINT_EPSILON_M
        and initial_distance > POINT_EPSILON_M
        and final_distance > POINT_EPSILON_M
    )
    endpoint["interpretation"] = (
        "descriptor_only_requires_distinct_initial_and_final_pair_endpoints"
    )
    return {
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FRAME_RATE_HZ,
        "source_ids": ["source1", "source2"],
        "source1_positions_m": positions1.tolist(),
        "source2_positions_m": positions2.tolist(),
        "route_descriptors": descriptor,
        "pathfinder_authority": {
            "source1": None,
            "source2": None,
        },
        "native_projection_visibility": {
            "status": "pending_same_renderer_native_execution",
            "projection_by_frame": None,
            "in_frame_by_frame": None,
            "occlusion_by_frame": None,
            "visible_pixels_by_frame": None,
        },
        "collision_status": "pending_native_articulated_body_execution",
        "audio_rir_status": "pending",
        "candidate_filtering": "none",
        "legality_effect": "none",
    }


def _case_record(
    *,
    motion_case: str,
    family: RouteFamily,
) -> dict[str, Any]:
    if motion_case == "static_static":
        route1, positions1 = _hold_route(
            NativePoint(family.source1.positions_m[0], 0, 0.0)
        )
        route2, positions2 = _hold_route(
            NativePoint(family.source2.positions_m[0], 0, 0.0)
        )
        path_record1 = {"status": "pass", "path_exists": True, "hold": True}
        path_record2 = {"status": "pass", "path_exists": True, "hold": True}
    elif motion_case == "s1_moving_s2_static":
        route1, positions1 = family.source1.planar_route, family.source1.positions_m
        route2, positions2 = _hold_route(
            NativePoint(family.source2.positions_m[0], 0, 0.0)
        )
        path_record1 = family.source1.pathfinder_record
        path_record2 = {"status": "pass", "path_exists": True, "hold": True}
    elif motion_case == "s1_static_s2_moving":
        route1, positions1 = _hold_route(
            NativePoint(family.source1.positions_m[0], 0, 0.0)
        )
        route2, positions2 = family.source2.planar_route, family.source2.positions_m
        path_record1 = {"status": "pass", "path_exists": True, "hold": True}
        path_record2 = family.source2.pathfinder_record
    elif motion_case == "both_moving":
        route1, positions1 = family.source1.planar_route, family.source1.positions_m
        route2, positions2 = family.source2.planar_route, family.source2.positions_m
        path_record1 = family.source1.pathfinder_record
        path_record2 = family.source2.pathfinder_record
    else:
        raise MP3DRegionSourcePlannerError(f"unknown motion case: {motion_case}")
    record = _route_pair_record(
        route1=route1,
        route2=route2,
        positions1=np.asarray(positions1, dtype=np.float64),
        positions2=np.asarray(positions2, dtype=np.float64),
    )
    record["motion_case"] = motion_case
    record["pathfinder_authority"] = {
        "source1": dict(path_record1),
        "source2": dict(path_record2),
    }
    return record


def _candidate_pair_routes(
    *,
    pathfinder: Any,
    shortest_path_factory: Callable[[], Any],
    region: MP3DRegion,
    points: Sequence[NativePoint],
    expected_island_id: int,
    seed: int,
    required_family_count: int,
) -> tuple[RouteFamily, ...]:
    ordered = list(points)
    random.Random(seed).shuffle(ordered)
    route_pool: list[MovingRoute] = []
    seen_paths: set[tuple[tuple[float, float], ...]] = set()
    for first, second in itertools.combinations(ordered, 2):
        route = _materialize_path(
            pathfinder=pathfinder,
            shortest_path_factory=shortest_path_factory,
            region=region,
            start=first,
            end=second,
            expected_island_id=expected_island_id,
        )
        if route is None:
            continue
        signature = _path_signature(route.waypoints_m)
        if signature in seen_paths:
            continue
        seen_paths.add(signature)
        route_pool.append(route)
        if len(route_pool) >= max(required_family_count + 3, 8):
            break
    if len(route_pool) < required_family_count + 1:
        raise MP3DRegionSourcePlannerError(
            f"region {region.region_instance_id} has too few distinct route geometries"
        )
    if len(route_pool) < 2:
        raise MP3DRegionSourcePlannerError(
            f"region {region.region_instance_id} has fewer than two distinct "
            "PathFinder route geometries"
        )
    families: list[RouteFamily] = []
    family_signatures: set[tuple[Any, ...]] = set()
    for family_index in range(required_family_count):
        if family_index == 0:
            left = route_pool[0]
            right = _reverse_route(
                pathfinder=pathfinder,
                shortest_path_factory=shortest_path_factory,
                region=region,
                route=left,
                expected_island_id=expected_island_id,
            )
            if right is None:
                continue
            exchange_capable = True
        else:
            left = route_pool[family_index]
            right = route_pool[family_index + 1]
            exchange_capable = False
        signature = _pair_signature(left, right)
        if signature in family_signatures:
            continue
        family_signatures.add(signature)
        families.append(
            RouteFamily(
                family_id=f"route_family_{family_index + 1:02d}",
                source1=left,
                source2=right,
                exchange_capable=exchange_capable,
                geometry_signature=signature,
            )
        )
    if len(families) != required_family_count:
        raise MP3DRegionSourcePlannerError(
            f"region {region.region_instance_id} produced {len(families)}/"
            f"{required_family_count} distinct route families"
        )
    return tuple(families)


def _validate_camera_bindings(
    bank: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> dict[str, tuple[CameraBinding, ...]]:
    region_banks = bank.get("region_banks")
    if not isinstance(region_banks, list):
        raise MP3DRegionSourcePlannerError("region_viewpoint_banks lacks region_banks")
    memberships = sidecar.get("placement_memberships")
    if not isinstance(memberships, list):
        raise MP3DRegionSourcePlannerError("sidecar lacks placement_memberships")
    membership_by_placement = {
        str(record.get("placement_id")): record for record in memberships
    }
    result: dict[str, tuple[CameraBinding, ...]] = {}
    for bank_record in sorted(
        region_banks, key=lambda item: int(item.get("region_index", -1))
    ):
        region_id = str(bank_record.get("region_instance_id"))
        selected = bank_record.get("selected_candidates")
        if not isinstance(selected, list) or len(selected) != 4:
            raise MP3DRegionSourcePlannerError(
                f"region {region_id} must retain exactly four camera bindings"
            )
        bindings: list[CameraBinding] = []
        seen_floors: set[int] = set()
        for selected_record in sorted(
            selected, key=lambda item: int(item.get("selection_order", 0))
        ):
            placement_id = str(selected_record.get("placement_id"))
            membership = membership_by_placement.get(placement_id)
            if membership is None:
                raise MP3DRegionSourcePlannerError(
                    f"camera placement {placement_id} is absent from sidecar"
                )
            if (
                membership.get("membership_status") != "unique"
                or membership.get("primary_region_instance_id") != region_id
            ):
                raise MP3DRegionSourcePlannerError(
                    f"camera placement {placement_id} is not uniquely bound to {region_id}"
                )
            camera = selected_record.get("camera")
            if not isinstance(camera, Mapping):
                raise MP3DRegionSourcePlannerError(
                    f"camera placement {placement_id} lacks camera record"
                )
            floor_index = int(camera["floor_sample_index"])
            if floor_index in seen_floors:
                raise MP3DRegionSourcePlannerError(
                    f"region {region_id} repeats camera floor position"
                )
            seen_floors.add(floor_index)
            bindings.append(
                CameraBinding(
                    region_instance_id=region_id,
                    region_index=int(bank_record["region_index"]),
                    selection_order=int(selected_record["selection_order"]),
                    placement_id=placement_id,
                    floor_position_m=_finite_xyz(
                        camera["floor_position_m"], owner="camera floor position"
                    ),
                    position_m=_finite_xyz(
                        camera["position_m"], owner="camera position"
                    ),
                    yaw_deg=_finite_number(camera["yaw_deg"], owner="camera yaw"),
                    height_id=str(camera["height_id"]),
                    request_path=str(selected_record["m1_capture_request_path"]),
                )
            )
        result[region_id] = tuple(bindings)
    if len(result) != 10:
        raise MP3DRegionSourcePlannerError(
            f"expected 10 region banks, got {len(result)}"
        )
    return result


def _family_record(
    *,
    region: MP3DRegion,
    family: RouteFamily,
    binding: CameraBinding,
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "route_family_id": family.family_id,
        "region_instance_id": region.region_instance_id,
        "region_index": region.region_index,
        "camera_binding": {
            "selection_order": binding.selection_order,
            "placement_id": binding.placement_id,
            "floor_position_m": list(binding.floor_position_m),
            "position_m": list(binding.position_m),
            "yaw_deg": binding.yaw_deg,
            "height_id": binding.height_id,
            "m1_capture_request_path": binding.request_path,
            "role": "metadata_binding_only_not_used_for_route_generation_or_filtering",
        },
        "route_generation": {
            "authority": "habitat_pathfinder_loaded_declared_mp3d_navmesh",
            "region_membership": "source_points_and_path_frames_inside_house_polygon",
            "camera_inputs_used": [],
            "camera_inputs_not_used": [
                "camera_floor_position",
                "camera_position",
                "camera_yaw",
                "rgb_or_semantic_pixels",
                "visibility",
                "composition",
            ],
            "seed_scope": "region_and_route_family_only",
            "collision_status": "pending_native_articulated_body_execution",
        },
        "cases": dict(cases),
    }


def build_region_source_route_bank(
    *,
    bank: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    house_plan: MP3DHouseFloorPlan,
    pathfinder: Any,
    shortest_path_factory: Callable[[], Any],
    seed: int = 20_260_824,
    sample_spacing_m: float = 0.10,
    maximum_y_delta_m: float = 0.30,
    required_family_count: int = 4,
) -> dict[str, Any]:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or required_family_count != 4
    ):
        raise MP3DRegionSourcePlannerError(
            "seed must be integer and required_family_count must remain four"
        )
    _pathfinder_bounds(pathfinder)
    bindings_by_region = _validate_camera_bindings(bank, sidecar)
    family_records: list[dict[str, Any]] = []
    region_summaries: list[dict[str, Any]] = []
    total_cases = 0
    exchange_cases = 0
    for region_id, bindings in sorted(
        bindings_by_region.items(),
        key=lambda item: int(next(binding.region_index for binding in item[1])),
    ):
        region_index = bindings[0].region_index
        region = next(
            (
                candidate
                for candidate in house_plan.regions
                if candidate.region_instance_id == region_id
            ),
            None,
        )
        if region is None or region.region_index != region_index:
            raise MP3DRegionSourcePlannerError(
                f"camera bank region {region_id} is absent from .house plan"
            )
        native_points = _native_points_for_region(
            pathfinder=pathfinder,
            region=region,
            sample_spacing_m=sample_spacing_m,
            maximum_y_delta_m=maximum_y_delta_m,
        )
        islands = defaultdict(list)
        for point in native_points:
            islands[point.island_id].append(point)
        island_options = sorted(
            islands.items(), key=lambda item: (-len(item[1]), int(item[0]))
        )
        route_result: tuple[RouteFamily, ...] | None = None
        selected_island = None
        failure_reasons: list[str] = []
        for island_id, island_points in island_options:
            try:
                route_result = _candidate_pair_routes(
                    pathfinder=pathfinder,
                    shortest_path_factory=shortest_path_factory,
                    region=region,
                    points=island_points,
                    expected_island_id=int(island_id),
                    seed=seed + region_index * 1009,
                    required_family_count=required_family_count,
                )
            except MP3DRegionSourcePlannerError as exc:
                failure_reasons.append(f"island {island_id}: {exc}")
                continue
            selected_island = int(island_id)
            break
        if route_result is None or selected_island is None:
            raise MP3DRegionSourcePlannerError(
                f"region {region_id} cannot produce four PathFinder families: "
                + " | ".join(failure_reasons)
            )
        cases_by_family: list[dict[str, Any]] = []
        for family, binding in zip(route_result, bindings, strict=True):
            cases = {
                motion_case: _case_record(motion_case=motion_case, family=family)
                for motion_case in MOTION_CASES
            }
            family_record = _family_record(
                region=region,
                family=family,
                binding=binding,
                cases=cases,
            )
            family_records.append(family_record)
            cases_by_family.append(family_record)
            total_cases += len(cases)
            exchange_cases += sum(
                int(
                    case["route_descriptors"]["endpoint_exchange"][
                        "exact_endpoint_exchange"
                    ]
                )
                for case in cases.values()
            )
        region_summaries.append(
            {
                "region_instance_id": region_id,
                "region_index": region_index,
                "native_pathfinder_point_count": len(native_points),
                "selected_island_id": selected_island,
                "camera_binding_count": len(bindings),
                "route_family_count": len(cases_by_family),
                "case_count": sum(len(record["cases"]) for record in cases_by_family),
                "native_capture_status": "pending",
                "audio_rir_status": "pending",
                "visibility_status": "pending",
                "collision_status": "pending",
            }
        )
    if total_cases != 160 or len(family_records) != 40:
        raise MP3DRegionSourcePlannerError(
            f"published {len(family_records)} families/{total_cases} cases; expected 40/160"
        )
    return {
        "schema": "avengine_mp3d_region_source_route_bank_v1",
        "artifact_status": "complete_cpu_region_source_route_bank",
        "research_only": True,
        "episode_counted": False,
        "formal_release_status": "not_run",
        "camera_binding_count": 40,
        "route_family_count": 40,
        "case_count": 160,
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FRAME_RATE_HZ,
        "motion_cases": list(MOTION_CASES),
        "exchange_case_count": exchange_cases,
        "candidate_filtering": "none",
        "legality_effect": "none",
        "native_capture_status": "pending",
        "audio_rir_status": "pending",
        "visibility_status": "pending",
        "collision_status": "pending",
        "camera_route_generation_decoupled": True,
        "regions": region_summaries,
        "families": family_records,
        "claim_boundary": (
            "CPU PathFinder/.house source-center route plan only; native articulated "
            "collision, UE/Habitat actor execution, visibility, occlusion, audio/RIR "
            "and formal Episode remain pending"
        ),
    }


def load_region_source_inputs(
    *,
    bank_path: str | Path,
    sidecar_path: str | Path,
    house_path: str | Path,
    pathfinder: Any,
    shortest_path_factory: Callable[[], Any],
    seed: int = 20_260_824,
    sample_spacing_m: float = 0.10,
    maximum_y_delta_m: float = 0.30,
) -> dict[str, Any]:
    bank = _load_json(bank_path, owner="region viewpoint bank")
    sidecar = _load_json(sidecar_path, owner="camera region sidecar")
    house_plan = parse_mp3d_house(house_path)
    return build_region_source_route_bank(
        bank=bank,
        sidecar=sidecar,
        house_plan=house_plan,
        pathfinder=pathfinder,
        shortest_path_factory=shortest_path_factory,
        seed=seed,
        sample_spacing_m=sample_spacing_m,
        maximum_y_delta_m=maximum_y_delta_m,
    )


__all__ = [
    "FRAME_COUNT",
    "FRAME_RATE_HZ",
    "MOTION_CASES",
    "MP3DRegionSourcePlannerError",
    "build_region_source_route_bank",
    "load_region_source_inputs",
]
