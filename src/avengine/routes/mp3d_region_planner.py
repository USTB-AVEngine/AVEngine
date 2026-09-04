"""Bounded MP3D region source-route planning.

The planner consumes parsed .house regions and an explicit camera selection only
as a descriptive binding. Source routes are generated from a real PathFinder
inside the selected region. This CPU module does not run actors, RLR, cameras,
or formal admission. It adapts the retained PathFinder route-family behavior
from 47adb44 and 46afac7 to the current routes owner and bounded request API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from avengine.rooms.mp3d_regions import MP3DHouseFloorPlan, MP3DRegion
from avengine.routes.trajectory import (
    M6XTrajectoryError,
    resample_polyline_by_arc_length,
)


class MP3DRegionRouteError(ValueError):
    """A bounded MP3D region route request cannot be planned."""


MOTION_CASES = (
    "static_static",
    "source1_moving_source2_static",
    "source1_static_source2_moving",
    "both_moving",
)
_MOVING_SOURCES: Mapping[str, frozenset[str]] = {
    "static_static": frozenset(),
    "source1_moving_source2_static": frozenset({"source1"}),
    "source1_static_source2_moving": frozenset({"source2"}),
    "both_moving": frozenset({"source1", "source2"}),
    # Readable aliases used by the historical region pilot.
    "s1_moving_s2_static": frozenset({"source1"}),
    "s1_static_s2_moving": frozenset({"source2"}),
}
_POINT_EPSILON_M = 1.0e-7


@dataclass(frozen=True)
class RegionRoute:
    waypoints_m: np.ndarray
    positions_m: np.ndarray
    island_id: int
    geodesic_distance_m: float

    def descriptor(self) -> dict[str, Any]:
        return {
            "geometry": "pathfinder_polyline",
            "waypoints_m": self.waypoints_m.tolist(),
            "frame_count": int(len(self.positions_m)),
            "positions_m": self.positions_m.tolist(),
            "island_id": self.island_id,
            "geodesic_distance_m": self.geodesic_distance_m,
            "pathfinder_status": "pass",
            "all_materialized_frames_in_region": True,
        }


def _number(value: Any, *, owner: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise MP3DRegionRouteError(f"{owner} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionRouteError(f"{owner} must be a finite number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise MP3DRegionRouteError(f"{owner} must be finite and >= {minimum}")
    return result


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MP3DRegionRouteError(f"{owner} must be a positive integer")
    return value


def _point(value: Any, *, owner: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise MP3DRegionRouteError(f"{owner} must contain three finite numbers")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionRouteError(f"{owner} must contain three finite numbers") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise MP3DRegionRouteError(f"{owner} must contain three finite numbers")
    return np.ascontiguousarray(result)


def _is_navigable(pathfinder: Any, point: np.ndarray, maximum_y_delta_m: float) -> bool:
    try:
        return bool(pathfinder.is_navigable(point, maximum_y_delta_m))
    except TypeError:
        return bool(pathfinder.is_navigable(point))


def _snap(pathfinder: Any, point: np.ndarray) -> np.ndarray | None:
    try:
        value = np.asarray(pathfinder.snap_point(point), dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    return np.ascontiguousarray(value)


def _region_floor_height(region: MP3DRegion) -> float:
    elevations = [
        polygon.floor_elevation_m for polygon in region.floor_polygons
    ]
    if not elevations:
        raise MP3DRegionRouteError(
            f"region {region.region_instance_id} has no floor polygon"
        )
    return float(np.median(np.asarray(elevations, dtype=np.float64)))


def _candidate_points(
    region: MP3DRegion,
    pathfinder: Any,
    *,
    sample_spacing_m: float,
    maximum_y_delta_m: float,
    maximum_candidate_points: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    spacing = _number(
        sample_spacing_m,
        owner="sample_spacing_m",
        minimum=1.0e-6,
    )
    maximum = _positive_int(
        maximum_candidate_points,
        owner="maximum_candidate_points",
    )
    floor_y = _region_floor_height(region)
    raw: list[np.ndarray] = []
    for polygon in region.floor_polygons:
        horizontal = np.asarray(polygon.horizontal_xz_m, dtype=np.float64)
        x_low, z_low = np.min(horizontal, axis=0)
        x_high, z_high = np.max(horizontal, axis=0)
        x_count = max(1, int(math.ceil((x_high - x_low) / spacing)) + 1)
        z_count = max(1, int(math.ceil((z_high - z_low) / spacing)) + 1)
        total = x_count * z_count
        stride = max(1, int(math.ceil(math.sqrt(total / maximum))))
        xs = np.linspace(x_low, x_high, max(1, (x_count - 1) // stride + 1))
        zs = np.linspace(z_low, z_high, max(1, (z_count - 1) // stride + 1))
        for z in zs:
            for x in xs:
                point = np.asarray([x, floor_y, z], dtype=np.float64)
                if not region.contains(point):
                    continue
                snapped = _snap(pathfinder, point)
                if snapped is None or not _is_navigable(
                    pathfinder, snapped, maximum_y_delta_m
                ):
                    continue
                if not region.contains(snapped, y_tolerance_m=maximum_y_delta_m):
                    continue
                raw.append(snapped)
    # The centroid is useful for narrow polygons that have no grid centre.
    for polygon in region.floor_polygons:
        horizontal = np.asarray(polygon.horizontal_xz_m, dtype=np.float64)
        point = np.asarray(
            [float(np.mean(horizontal[:, 0])), floor_y, float(np.mean(horizontal[:, 1]))],
            dtype=np.float64,
        )
        snapped = _snap(pathfinder, point)
        if (
            snapped is not None
            and _is_navigable(pathfinder, snapped, maximum_y_delta_m)
            and region.contains(snapped, y_tolerance_m=maximum_y_delta_m)
        ):
            raw.append(snapped)

    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for point in raw:
        key = tuple(float(value) for value in np.round(point, decimals=6))
        unique.setdefault(key, point)
    points = list(unique.values())
    if len(points) > maximum:
        rng = random.Random(int(seed))
        rng.shuffle(points)
        points = points[:maximum]
    points.sort(key=lambda value: tuple(float(item) for item in value))
    if len(points) < 2:
        raise MP3DRegionRouteError(
            f"region {region.region_instance_id} has fewer than two "
            "PathFinder candidate points"
        )
    return tuple(np.ascontiguousarray(point) for point in points)


def _path_between(
    region: MP3DRegion,
    pathfinder: Any,
    shortest_path_factory: Callable[[], Any],
    start: np.ndarray,
    end: np.ndarray,
    *,
    frame_count: int,
    maximum_y_delta_m: float,
) -> RegionRoute | None:
    query = shortest_path_factory()
    try:
        query.requested_start = np.asarray(start, dtype=np.float32)
        query.requested_end = np.asarray(end, dtype=np.float32)
        found = bool(pathfinder.find_path(query))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if not found:
        return None
    try:
        points = np.asarray(query.points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        return None
    if not np.all(np.isfinite(points)):
        return None
    if any(
        not region.contains(point, y_tolerance_m=maximum_y_delta_m)
        for point in points
    ):
        return None
    try:
        positions = resample_polyline_by_arc_length(
            points,
            frame_count,
            owner="MP3D region PathFinder route",
        )
    except M6XTrajectoryError:
        return None
    if any(
        not region.contains(point, y_tolerance_m=maximum_y_delta_m)
        for point in positions
    ):
        return None
    try:
        island = int(pathfinder.get_island(start))
    except (AttributeError, TypeError, ValueError, OverflowError):
        island = -1
    if island < 0:
        return None
    try:
        distance = float(query.geodesic_distance)
    except (AttributeError, TypeError, ValueError, OverflowError):
        distance = float(
            np.linalg.norm(np.diff(points[:, (0, 2)], axis=0), axis=1).sum()
        )
    if not math.isfinite(distance) or distance <= _POINT_EPSILON_M:
        return None
    return RegionRoute(
        waypoints_m=np.ascontiguousarray(points),
        positions_m=np.ascontiguousarray(positions),
        island_id=island,
        geodesic_distance_m=distance,
    )


def _route_signature(route: RegionRoute) -> tuple[tuple[float, float, float], ...]:
    points = np.round(route.waypoints_m, decimals=6)
    forward = tuple(tuple(float(item) for item in row) for row in points)
    reverse = tuple(tuple(float(item) for item in row) for row in points[::-1])
    return min(forward, reverse)


def _pair_minimum_separation(left: RegionRoute, right: RegionRoute) -> float:
    return float(
        np.min(
            np.linalg.norm(
                left.positions_m[:, (0, 2)] - right.positions_m[:, (0, 2)],
                axis=1,
            )
        )
    )


def _route_endpoints_distinct(
    candidate: RegionRoute,
    retained: Sequence[RegionRoute],
    minimum_distance_m: float,
) -> bool:
    if minimum_distance_m <= 0.0:
        return True
    candidate_endpoints = (candidate.positions_m[0], candidate.positions_m[-1])
    for route in retained:
        retained_endpoints = (route.positions_m[0], route.positions_m[-1])
        if any(
            float(np.linalg.norm(left[[0, 2]] - right[[0, 2]]))
            < minimum_distance_m
            for left in candidate_endpoints
            for right in retained_endpoints
        ):
            return False
    return True


def _moving_sources(motion_case: str) -> frozenset[str]:
    try:
        return _MOVING_SOURCES[motion_case]
    except KeyError as exc:
        raise MP3DRegionRouteError(
            f"unsupported motion_case {motion_case!r}; choose one of "
            f"{sorted(_MOVING_SOURCES)}"
        ) from exc


def _case_record(
    family_id: str,
    motion_case: str,
    left: RegionRoute,
    right: RegionRoute,
    *,
    frame_rate_hz: int,
) -> dict[str, Any]:
    moving = _moving_sources(motion_case)
    source1 = (
        left.positions_m
        if "source1" in moving
        else np.repeat(left.positions_m[0][None, :], len(left.positions_m), axis=0)
    )
    source2 = (
        right.positions_m
        if "source2" in moving
        else np.repeat(right.positions_m[0][None, :], len(right.positions_m), axis=0)
    )
    return {
        "motion_case": motion_case,
        "frame_count": int(len(source1)),
        "frame_rate_hz": frame_rate_hz,
        "source1_positions_m": np.asarray(source1, dtype=np.float64).tolist(),
        "source2_positions_m": np.asarray(source2, dtype=np.float64).tolist(),
        "route_descriptors": {
            "source1": {
                **left.descriptor(),
                "moving": "source1" in moving,
            },
            "source2": {
                **right.descriptor(),
                "moving": "source2" in moving,
            },
            "endpoint_exchange": {
                "exact_endpoint_exchange": False,
                "meaning": "not evaluated by this bounded route planner",
            },
        },
        "route_family_id": family_id,
    }


def _region_ids(
    plan: MP3DHouseFloorPlan,
    camera_plan: Mapping[str, Any],
    requested: Sequence[int] | None,
) -> tuple[int, ...]:
    by_index = plan.by_region_index
    if requested is None:
        raw = camera_plan.get("requested_region_indices")
        if not isinstance(raw, list):
            raw = sorted(by_index)
    else:
        raw = list(requested)
    values: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise MP3DRegionRouteError("region_indices must contain integers")
        values.append(int(item))
    if not values or len(values) != len(set(values)):
        raise MP3DRegionRouteError("region_indices must be nonempty and unique")
    if any(item not in by_index for item in values):
        raise MP3DRegionRouteError("region_indices contains an unknown region")
    return tuple(values)


def _cameras_by_region(camera_plan: Mapping[str, Any]) -> dict[int, list[Mapping[str, Any]]]:
    records = camera_plan.get("regions")
    if not isinstance(records, list):
        raise MP3DRegionRouteError("camera plan lacks regions")
    result: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise MP3DRegionRouteError("camera plan regions must be objects")
        index = record.get("region_index")
        cameras = record.get("cameras")
        if isinstance(index, bool) or not isinstance(index, int) or not isinstance(cameras, list):
            raise MP3DRegionRouteError("camera plan region record is malformed")
        result[int(index)] = [
            camera for camera in cameras if isinstance(camera, Mapping)
        ]
    return result


def build_region_route_plan(
    plan: MP3DHouseFloorPlan,
    camera_plan: Mapping[str, Any],
    pathfinder: Any,
    shortest_path_factory: Callable[[], Any],
    *,
    region_indices: Sequence[int] | None = None,
    route_families_per_region: int = 1,
    motion_cases: Sequence[str] = MOTION_CASES,
    frame_count: int = 75,
    frame_rate_hz: int = 15,
    sample_spacing_m: float = 0.50,
    maximum_y_delta_m: float = 0.30,
    maximum_candidate_points: int = 64,
    maximum_route_attempts: int = 256,
    minimum_pair_separation_m: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a finite source-route plan for selected parsed regions.

    Counts and clocks are request parameters. Camera records are retained as a
    binding description only and are never consulted while generating routes.
    """

    if not isinstance(plan, MP3DHouseFloorPlan):
        raise MP3DRegionRouteError("plan must be an MP3DHouseFloorPlan")
    if not isinstance(camera_plan, Mapping):
        raise MP3DRegionRouteError("camera_plan must be an object")
    families_count = _positive_int(
        route_families_per_region,
        owner="route_families_per_region",
    )
    frames = _positive_int(frame_count, owner="frame_count")
    fps = _positive_int(frame_rate_hz, owner="frame_rate_hz")
    candidate_limit = _positive_int(
        maximum_candidate_points,
        owner="maximum_candidate_points",
    )
    attempt_limit = _positive_int(
        maximum_route_attempts,
        owner="maximum_route_attempts",
    )
    separation = _number(
        minimum_pair_separation_m,
        owner="minimum_pair_separation_m",
        minimum=0.0,
    )
    y_delta = _number(
        maximum_y_delta_m,
        owner="maximum_y_delta_m",
        minimum=0.0,
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MP3DRegionRouteError("seed must be an integer")
    if isinstance(motion_cases, (str, bytes)) or not isinstance(motion_cases, Sequence):
        raise MP3DRegionRouteError("motion_cases must be a nonempty sequence")
    if any(not isinstance(value, str) or not value for value in motion_cases):
        raise MP3DRegionRouteError("motion_cases must contain nonempty strings")
    motion_values = tuple(motion_cases)
    if not motion_values or len(set(motion_values)) != len(motion_values):
        raise MP3DRegionRouteError("motion_cases must be nonempty and unique")
    for motion in motion_values:
        _moving_sources(motion)

    selected_indices = _region_ids(plan, camera_plan, region_indices)
    cameras = _cameras_by_region(camera_plan)
    outputs: list[dict[str, Any]] = []
    total_families = 0
    total_cases = 0
    for region_index in selected_indices:
        region = plan.by_region_index[region_index]
        if not cameras.get(region_index):
            raise MP3DRegionRouteError(
                f"camera plan has no cameras for region {region.region_instance_id}"
            )
        points = _candidate_points(
            region,
            pathfinder,
            sample_spacing_m=sample_spacing_m,
            maximum_y_delta_m=y_delta,
            maximum_candidate_points=candidate_limit,
            seed=seed + region_index * 1009,
        )
        ordered = list(points)
        random.Random(seed + region_index * 7919).shuffle(ordered)
        routes: list[RegionRoute] = []
        signatures: set[tuple[tuple[float, float, float], ...]] = set()
        attempts = 0
        route_target = max(families_count * 4, 2)
        for first_index, first in enumerate(ordered):
            if len(routes) >= route_target or attempts >= attempt_limit:
                break
            for second in ordered[first_index + 1 :]:
                if len(routes) >= route_target or attempts >= attempt_limit:
                    break
                attempts += 1
                route = _path_between(
                    region,
                    pathfinder,
                    shortest_path_factory,
                    first,
                    second,
                    frame_count=frames,
                    maximum_y_delta_m=y_delta,
                )
                if route is None:
                    continue
                signature = _route_signature(route)
                if signature in signatures:
                    continue
                if not _route_endpoints_distinct(route, routes, separation):
                    continue
                signatures.add(signature)
                routes.append(route)
        families: list[tuple[RegionRoute, RegionRoute]] = []
        used_route_indices: set[int] = set()
        for left_index in range(len(routes)):
            if len(families) >= families_count:
                break
            if left_index in used_route_indices:
                continue
            for right_index in range(left_index + 1, len(routes)):
                if right_index in used_route_indices:
                    continue
                if _pair_minimum_separation(
                    routes[left_index], routes[right_index]
                ) < separation:
                    continue
                families.append((routes[left_index], routes[right_index]))
                used_route_indices.update((left_index, right_index))
                break
        if len(families) != families_count:
            raise MP3DRegionRouteError(
                f"region {region.region_instance_id} produced {len(families)}/"
                f"{families_count} route families after {attempts} bounded attempts"
            )
        family_records: list[dict[str, Any]] = []
        region_case_count = 0
        for family_index, (left, right) in enumerate(families, start=1):
            family_id = (
                f"{region.region_instance_id.replace(':', '_')}"
                f"_route_family_{family_index:02d}"
            )
            cases = {
                motion: _case_record(
                    family_id,
                    motion,
                    left,
                    right,
                    frame_rate_hz=fps,
                )
                for motion in motion_values
            }
            family_records.append(
                {
                    "route_family_id": family_id,
                    "region_instance_id": region.region_instance_id,
                    "region_index": region_index,
                    "camera_binding": dict(cameras[region_index][(family_index - 1) % len(cameras[region_index])]),
                    "camera_inputs_used_for_route_generation": False,
                    "exchange_capable": False,
                    "source_route_pair_minimum_separation_m": _pair_minimum_separation(left, right),
                    "cases": cases,
                }
            )
            region_case_count += len(cases)
        outputs.append(
            {
                **region.label_record(),
                "camera_count": len(cameras[region_index]),
                "candidate_point_count": len(points),
                "route_family_count": len(family_records),
                "case_count": region_case_count,
                "route_attempt_count": attempts,
                "route_families": family_records,
            }
        )
        total_families += len(family_records)
        total_cases += region_case_count

    return {
        "artifact_kind": "mp3d_region_source_route_plan",
        "research_only": True,
        "episode_counted": False,
        "house_id": plan.house_id,
        "region_count": len(outputs),
        "route_family_count": total_families,
        "case_count": total_cases,
        "parameters": {
            "requested_region_indices": list(selected_indices),
            "route_families_per_region": families_count,
            "motion_cases": list(motion_values),
            "frame_count": frames,
            "frame_rate_hz": fps,
            "sample_spacing_m": float(sample_spacing_m),
            "maximum_y_delta_m": y_delta,
            "maximum_candidate_points": candidate_limit,
            "maximum_route_attempts": attempt_limit,
            "minimum_pair_separation_m": separation,
            "seed": int(seed),
        },
        "route_generation": {
            "authority": "declared .house polygons plus supplied Habitat PathFinder",
            "camera_inputs_used": False,
            "visibility_used": False,
            "audio_used": False,
            "bounded_search": True,
        },
        "regions": outputs,
        "downstream_dependencies": {
            "actor_track_materializer": {
                "status": "not_run",
                "requires": [
                    "current source asset/runtime-profile bundle per source slot",
                    "room manifest and M1 request for the same MP3D scene",
                    "a frame clock compatible with this plan",
                ],
            },
            "native_capture": {
                "status": "not_run",
                "requires": [
                    "installed Habitat runtime and the selected room assets",
                    "capture-scoped multi-actor runner",
                    "selected camera request from the camera binding",
                ],
            },
            "rlr_audio": {
                "status": "not_run",
                "requires": [
                    "current compiled MP3D acoustic package",
                    "simulation request and named endpoint registry",
                    "dynamic audio adapter after native emitter readback",
                ],
            },
        },
    }


__all__ = [
    "MOTION_CASES",
    "MP3DRegionRouteError",
    "RegionRoute",
    "build_region_route_plan",
]