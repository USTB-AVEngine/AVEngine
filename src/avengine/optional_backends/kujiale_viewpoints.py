"""Deterministic CPU prefilter for InteriorAgent/Kujiale viewpoints.

The planner consumes the retained room-polygon/descendant-footprint raster. It
does not import USD, Unreal, SPEAR, Habitat, or a renderer. Its visibility and
line-of-sight results are two-dimensional planning evidence only: every chosen
view still needs fresh UE pixels, camera/FOV readback, and a four-point
mini-pilot before production use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
import re
from typing import Any

import numpy as np

from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.room_feasibility import FeasibleRegionIndex
from avengine.m6x.route_descriptors import (
    MaterializedRoute,
    describe_route_pair,
    materialize_route,
)


CPU_PREFILTER_AUTHORITY = (
    "kujiale_room_polygon_plus_usd_descendant_footprint_raster_cpu_prefilter"
)
CPU_PREFILTER_BOUNDARY = (
    "2D camera/corridor planning only; not a Habitat navmesh, native pixel, "
    "full-body collision, UE/SPEAR certification, or dataset-admission claim"
)


class KujialeViewpointError(ValueError):
    """Kujiale CPU viewpoint planning inputs cannot produce a bounded plan."""


def _finite(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise KujialeViewpointError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise KujialeViewpointError(f"{owner} must be a finite number")
    return 0.0 if result == 0.0 else result


def _positive(value: Any, *, owner: str) -> float:
    result = _finite(value, owner=owner)
    if result <= 0.0:
        raise KujialeViewpointError(f"{owner} must be positive")
    return result


def _count(value: Any, *, owner: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise KujialeViewpointError(f"{owner} must be an integer >= {minimum}")
    return value


def _normalized_yaw(value: Real) -> float:
    yaw = (_finite(value, owner="camera yaw") + 180.0) % 360.0 - 180.0
    return 0.0 if yaw == 0.0 else yaw


def _yaw_delta(left: Real, right: Real) -> float:
    return abs(
        (_normalized_yaw(left) - _normalized_yaw(right) + 180.0) % 360.0
        - 180.0
    )


def _identifier(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KujialeViewpointError(f"{owner} must be a non-empty string")
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    if not result:
        raise KujialeViewpointError(f"{owner} has no usable identifier characters")
    return result


def _common_region(
    region_by_source: Mapping[str, FeasibleRegionIndex],
) -> tuple[RuntimeObstacleMap, np.ndarray]:
    if set(region_by_source) != {"source1", "source2"}:
        raise KujialeViewpointError(
            "region_by_source must contain exactly source1 and source2"
        )
    first = region_by_source["source1"]
    second = region_by_source["source2"]
    if not isinstance(first, FeasibleRegionIndex) or not isinstance(
        second, FeasibleRegionIndex
    ):
        raise KujialeViewpointError("regions must be FeasibleRegionIndex values")
    if first.obstacle_map is not second.obstacle_map:
        raise KujialeViewpointError("source regions must share one obstacle map")
    if first.feasible_mask.shape != second.feasible_mask.shape:
        raise KujialeViewpointError("source region mask shapes differ")
    common = np.asarray(first.feasible_mask & second.feasible_mask, dtype=np.bool_)
    if not np.any(common):
        raise KujialeViewpointError("source-region intersection is empty")
    return first.obstacle_map, np.ascontiguousarray(common)


def _pixel_sizes(
    obstacle_map: RuntimeObstacleMap, shape: tuple[int, int]
) -> tuple[float, float]:
    bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)
    return (
        float((bounds[1, 0] - bounds[0, 0]) / shape[1]),
        float((bounds[1, 2] - bounds[0, 2]) / shape[0]),
    )


def _pixel_to_xy(
    obstacle_map: RuntimeObstacleMap,
    shape: tuple[int, int],
    pixel_rc: Sequence[int],
) -> tuple[float, float]:
    row, col = int(pixel_rc[0]), int(pixel_rc[1])
    pixel_x, pixel_z = _pixel_sizes(obstacle_map, shape)
    bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)
    return (
        float(bounds[0, 0] + (col + 0.5) * pixel_x),
        float(bounds[0, 2] + (row + 0.5) * pixel_z),
    )


def _xy_to_pixel(
    obstacle_map: RuntimeObstacleMap,
    shape: tuple[int, int],
    point_xy: Sequence[float],
) -> tuple[int, int] | None:
    pixel_x, pixel_z = _pixel_sizes(obstacle_map, shape)
    bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)
    col = int(math.floor((float(point_xy[0]) - bounds[0, 0]) / pixel_x))
    row = int(math.floor((float(point_xy[1]) - bounds[0, 2]) / pixel_z))
    if 0 <= row < shape[0] and 0 <= col < shape[1]:
        return row, col
    return None


def _segment_pixels(
    obstacle_map: RuntimeObstacleMap,
    shape: tuple[int, int],
    start_xy: Sequence[float],
    end_xy: Sequence[float],
) -> tuple[tuple[int, int], ...]:
    pixel_x, pixel_z = _pixel_sizes(obstacle_map, shape)
    bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)
    start_pixel = _xy_to_pixel(obstacle_map, shape, start_xy)
    end_pixel = _xy_to_pixel(obstacle_map, shape, end_xy)
    if start_pixel is None or end_pixel is None:
        return ()
    x0 = (float(start_xy[0]) - bounds[0, 0]) / pixel_x
    z0 = (float(start_xy[1]) - bounds[0, 2]) / pixel_z
    x1 = (float(end_xy[0]) - bounds[0, 0]) / pixel_x
    z1 = (float(end_xy[1]) - bounds[0, 2]) / pixel_z
    delta_x = x1 - x0
    delta_z = z1 - z0
    step_x = 1 if delta_x > 0.0 else (-1 if delta_x < 0.0 else 0)
    step_z = 1 if delta_z > 0.0 else (-1 if delta_z < 0.0 else 0)

    def initial_boundary_time(value: float, delta: float, step: int) -> float:
        if step > 0:
            return (math.floor(value) + 1.0 - value) / delta
        if step < 0:
            return (value - math.floor(value)) / -delta
        return math.inf

    time_x = initial_boundary_time(x0, delta_x, step_x)
    time_z = initial_boundary_time(z0, delta_z, step_z)
    delta_time_x = math.inf if step_x == 0 else 1.0 / abs(delta_x)
    delta_time_z = math.inf if step_z == 0 else 1.0 / abs(delta_z)
    row, col = start_pixel
    pixels: list[tuple[int, int]] = []

    def append(candidate_row: int, candidate_col: int) -> bool:
        if not 0 <= candidate_row < shape[0] or not 0 <= candidate_col < shape[1]:
            return False
        pixel = (candidate_row, candidate_col)
        if not pixels or pixels[-1] != pixel:
            pixels.append(pixel)
        return True

    if not append(row, col):
        return ()
    iterations = 0
    while (row, col) != end_pixel:
        iterations += 1
        if iterations > shape[0] + shape[1] + 4:
            return ()
        if time_x + 1.0e-12 < time_z:
            col += step_x
            time_x += delta_time_x
            if not append(row, col):
                return ()
        elif time_z + 1.0e-12 < time_x:
            row += step_z
            time_z += delta_time_z
            if not append(row, col):
                return ()
        else:
            # A corner crossing touches both side-adjacent cells before the
            # diagonal cell. Retaining all three is conservative and prevents
            # a segment from threading exactly through two blocked corners.
            if not append(row, col + step_x) or not append(row + step_z, col):
                return ()
            row += step_z
            col += step_x
            time_x += delta_time_x
            time_z += delta_time_z
            if not append(row, col):
                return ()
    return tuple(pixels)


def _segment_clear(
    obstacle_map: RuntimeObstacleMap,
    mask: np.ndarray,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
) -> bool:
    pixels = _segment_pixels(obstacle_map, mask.shape, start_xy, end_xy)
    return bool(pixels) and all(bool(mask[row, col]) for row, col in pixels)


def _sample_pixels(
    mask: np.ndarray, spacing_m: float, pixel_size_m: float
) -> np.ndarray:
    step = max(1, int(round(spacing_m / pixel_size_m)))
    sample_mask = np.zeros_like(mask, dtype=np.bool_)
    sample_mask[::step, ::step] = True
    pixels = np.argwhere(mask & sample_mask)
    if not len(pixels):
        pixels = np.argwhere(mask)
    return np.ascontiguousarray(pixels, dtype=np.int32)


def _limit_pixels(pixels: np.ndarray, maximum_count: int) -> np.ndarray:
    if len(pixels) <= maximum_count:
        return pixels
    indices = np.linspace(0, len(pixels) - 1, maximum_count, dtype=np.int64)
    return np.ascontiguousarray(pixels[indices], dtype=np.int32)


def natural_distance_band(
    motion_profiles: Mapping[str, Any], species: Sequence[str]
) -> tuple[float, float]:
    profiles = motion_profiles.get("profiles")
    clip_seconds = _positive(
        motion_profiles.get("clip_seconds"), owner="motion_profiles.clip_seconds"
    )
    if not isinstance(profiles, Mapping):
        raise KujialeViewpointError("motion_profiles.profiles must be an object")
    requested = tuple(str(item) for item in species)
    if not requested:
        raise KujialeViewpointError("at least one species motion profile is required")
    minimums: list[float] = []
    maximums: list[float] = []
    for species_id in requested:
        profile = profiles.get(species_id)
        if not isinstance(profile, Mapping):
            raise KujialeViewpointError(
                f"motion profile is missing species {species_id!r}"
            )
        minimums.append(
            _positive(
                profile.get("walk_speed_mps_min"),
                owner=f"{species_id}.walk_speed_mps_min",
            )
            * clip_seconds
        )
        maximums.append(
            _positive(
                profile.get("walk_speed_mps_max"),
                owner=f"{species_id}.walk_speed_mps_max",
            )
            * clip_seconds
        )
    minimum = max(minimums)
    maximum = min(maximums)
    if minimum > maximum:
        raise KujialeViewpointError(
            "requested species have no shared natural-distance interval"
        )
    return minimum, maximum


def generate_straight_corridors(
    *,
    region_by_source: Mapping[str, FeasibleRegionIndex],
    motion_profiles: Mapping[str, Any],
    species: Sequence[str] = ("human", "dog"),
    maximum_corridors: int = 256,
    origin_spacing_m: Real = 0.50,
    direction_step_degrees: Real = 15.0,
) -> list[dict[str, Any]]:
    """Enumerate finite straight chords whose entire raster trace is feasible."""

    maximum = _count(maximum_corridors, owner="maximum_corridors")
    obstacle_map, common = _common_region(region_by_source)
    minimum_distance, maximum_distance = natural_distance_band(
        motion_profiles, species
    )
    origin_spacing = _positive(origin_spacing_m, owner="origin_spacing_m")
    direction_step = _positive(
        direction_step_degrees, owner="direction_step_degrees"
    )
    if direction_step > 90.0:
        raise KujialeViewpointError("direction_step_degrees must not exceed 90")
    pixel_x, pixel_z = _pixel_sizes(obstacle_map, common.shape)
    pixels = [
        (int(row), int(col))
        for row, col in _sample_pixels(
            common, origin_spacing, min(pixel_x, pixel_z)
        )
    ]
    points = [_pixel_to_xy(obstacle_map, common.shape, pixel) for pixel in pixels]
    by_direction: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(12)
    }
    length_samples = sorted(
        {
            minimum_distance,
            0.5 * (minimum_distance + maximum_distance),
            maximum_distance,
        },
        reverse=True,
    )
    seen_segments: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    direction_count = int(math.ceil(360.0 / direction_step))
    for start in points:
        for direction_index in range(direction_count):
            angle = _normalized_yaw(-180.0 + direction_index * direction_step)
            radians = math.radians(angle)
            for distance in length_samples:
                end = (
                    float(start[0]) + math.cos(radians) * distance,
                    float(start[1]) + math.sin(radians) * distance,
                )
                if not _segment_clear(obstacle_map, common, start, end):
                    continue
                endpoints = sorted(
                    (
                        (round(float(start[0]), 6), round(float(start[1]), 6)),
                        (round(float(end[0]), 6), round(float(end[1]), 6)),
                    )
                )
                segment_key = (endpoints[0], endpoints[1])
                if segment_key in seen_segments:
                    continue
                seen_segments.add(segment_key)
                undirected = angle % 180.0
                direction_bin = min(11, int(undirected // 15.0))
                by_direction[direction_bin].append(
                    {
                        "start_xy_m": [float(start[0]), float(start[1])],
                        "end_xy_m": [float(end[0]), float(end[1])],
                        "length_m": float(distance),
                        "undirected_angle_deg": float(undirected),
                    }
                )
    for records in by_direction.values():
        records.sort(
            key=lambda item: (
                -round(float(item["length_m"]), 9),
                item["start_xy_m"],
                item["end_xy_m"],
            )
        )
    selected: list[dict[str, Any]] = []
    ordinal = 0
    while len(selected) < maximum:
        added = False
        for direction_bin in range(12):
            records = by_direction[direction_bin]
            if ordinal < len(records):
                selected.append(records[ordinal])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        ordinal += 1
    if not selected:
        raise KujialeViewpointError(
            "room has no fully feasible straight chord in the natural-distance "
            f"band [{minimum_distance:.3f}, {maximum_distance:.3f}] m"
        )
    floor = float(obstacle_map.floor_height_m)
    profiles = motion_profiles["profiles"]
    clip_seconds = float(motion_profiles["clip_seconds"])
    for index, corridor in enumerate(selected, start=1):
        corridor["corridor_id"] = f"corridor_{index:04d}"
        corridor["route_geometry"] = "straight"
        corridor["waypoints_xy_m"] = [
            list(corridor["start_xy_m"]),
            list(corridor["end_xy_m"]),
        ]
        corridor["waypoints_dataset_xyz_m"] = [
            [corridor["start_xy_m"][0], corridor["start_xy_m"][1], floor],
            [corridor["end_xy_m"][0], corridor["end_xy_m"][1], floor],
        ]
        corridor["start_world_m"] = [
            corridor["start_xy_m"][0],
            floor,
            corridor["start_xy_m"][1],
        ]
        corridor["end_world_m"] = [
            corridor["end_xy_m"][0],
            floor,
            corridor["end_xy_m"][1],
        ]
        corridor["natural_distance_band_m"] = [minimum_distance, maximum_distance]
        corridor["nominal_clip_speed_mps"] = (
            float(corridor["length_m"]) / clip_seconds
        )
        corridor["species_profiles"] = list(species)
        corridor["motion_profile_by_species"] = {
            species_id: {
                "walk_speed_mps_min": float(
                    profiles[species_id]["walk_speed_mps_min"]
                ),
                "walk_speed_mps_max": float(
                    profiles[species_id]["walk_speed_mps_max"]
                ),
                "clip_seconds": clip_seconds,
            }
            for species_id in species
        }
        corridor["authority"] = CPU_PREFILTER_AUTHORITY
        corridor["claim_boundary"] = CPU_PREFILTER_BOUNDARY
    return selected


def _turn_is_simple(
    previous: Sequence[float], current: Sequence[float], following: Sequence[float]
) -> bool:
    incoming = np.asarray(current, dtype=np.float64) - np.asarray(
        previous, dtype=np.float64
    )
    outgoing = np.asarray(following, dtype=np.float64) - np.asarray(
        current, dtype=np.float64
    )
    denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if denominator <= 1.0e-12:
        return False
    angle = math.degrees(
        math.acos(
            min(1.0, max(-1.0, float(np.dot(incoming, outgoing)) / denominator))
        )
    )
    return 20.0 <= angle <= 160.0


def _proper_segment_intersection(
    a0: Sequence[float],
    a1: Sequence[float],
    b0: Sequence[float],
    b1: Sequence[float],
) -> bool:
    def orientation(
        origin: Sequence[float], left: Sequence[float], right: Sequence[float]
    ) -> float:
        return (float(left[0]) - float(origin[0])) * (
            float(right[1]) - float(origin[1])
        ) - (float(left[1]) - float(origin[1])) * (
            float(right[0]) - float(origin[0])
        )

    first = orientation(a0, a1, b0)
    second = orientation(a0, a1, b1)
    third = orientation(b0, b1, a0)
    fourth = orientation(b0, b1, a1)
    return (
        first * second < -1.0e-12 and third * fourth < -1.0e-12
    )


def _route_key(points: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    forward = tuple(
        (round(float(point[0]), 6), round(float(point[1]), 6))
        for point in points
    )
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


def generate_polyline_corridors(
    *,
    region_by_source: Mapping[str, FeasibleRegionIndex],
    motion_profiles: Mapping[str, Any],
    species: Sequence[str] = ("human", "dog"),
    maximum_corridors: int = 256,
    waypoint_spacing_m: Real = 0.25,
    maximum_segments: int = 4,
    maximum_segment_length_m: Real = 1.80,
) -> list[dict[str, Any]]:
    """Search simple multi-segment paths on the actors' common feasible raster.

    Room names, room types, camera positions and image semantics never enter
    the search.  Each retained edge is checked by conservative raster
    supercover, and every route is valid in both actor-specific feasible masks.
    """

    maximum = _count(maximum_corridors, owner="maximum_corridors")
    segment_limit = _count(
        maximum_segments, owner="maximum_segments", minimum=2
    )
    if segment_limit > 6:
        raise KujialeViewpointError("maximum_segments must not exceed 6")
    waypoint_spacing = _positive(waypoint_spacing_m, owner="waypoint_spacing_m")
    maximum_segment_length = _positive(
        maximum_segment_length_m, owner="maximum_segment_length_m"
    )
    obstacle_map, common = _common_region(region_by_source)
    minimum_distance, maximum_distance = natural_distance_band(
        motion_profiles, species
    )
    pixel_x, pixel_z = _pixel_sizes(obstacle_map, common.shape)
    pixels = [
        (int(row), int(col))
        for row, col in _sample_pixels(
            common, waypoint_spacing, min(pixel_x, pixel_z)
        )
    ]
    points = [_pixel_to_xy(obstacle_map, common.shape, pixel) for pixel in pixels]
    if len(points) < 3:
        raise KujialeViewpointError(
            "room has fewer than three sampled feasible polyline waypoints"
        )
    neighbors: dict[int, list[tuple[int, float]]] = {index: [] for index in range(len(points))}
    for left_index, left in enumerate(points):
        for right_index in range(left_index + 1, len(points)):
            right = points[right_index]
            distance = math.dist(left, right)
            if (
                distance + 1.0e-12 < waypoint_spacing * 0.90
                or distance > maximum_segment_length + 1.0e-12
                or not _segment_clear(obstacle_map, common, left, right)
            ):
                continue
            neighbors[left_index].append((right_index, distance))
            neighbors[right_index].append((left_index, distance))
    for source_index, values in neighbors.items():
        source = points[source_index]
        values.sort(
            key=lambda value: (
                round(
                    math.degrees(
                        math.atan2(
                            points[value[0]][1] - source[1],
                            points[value[0]][0] - source[0],
                        )
                    )
                    % 360.0,
                    6,
                ),
                -round(value[1], 9),
                value[0],
            )
        )
        # A dense free raster can connect one node to hundreds of others.
        # Keep the two longest clear edges per 30-degree direction bucket: the
        # resulting graph remains target-agnostic while bounding DFS branching.
        direction_counts = {index: 0 for index in range(12)}
        bounded = []
        for value in values:
            target = points[value[0]]
            angle = math.degrees(
                math.atan2(target[1] - source[1], target[0] - source[0])
            ) % 360.0
            direction_index = min(11, int(angle // 30.0))
            if direction_counts[direction_index] >= 2:
                continue
            direction_counts[direction_index] += 1
            bounded.append(value)
        neighbors[source_index] = bounded

    candidate_limit = max(maximum * 24, 256)
    candidate_records: dict[tuple[tuple[float, float], ...], dict[str, Any]] = {}
    per_origin_limit = max(8, int(math.ceil(candidate_limit / len(points))))
    for origin_index in range(len(points)):
        retained_for_origin = 0

        def visit(path: list[int], length_m: float) -> None:
            nonlocal retained_for_origin
            if (
                retained_for_origin >= per_origin_limit
                or len(candidate_records) >= candidate_limit
            ):
                return
            segment_count = len(path) - 1
            if segment_count >= 2 and minimum_distance - 1.0e-9 <= length_m <= maximum_distance + 1.0e-9:
                route_points = [points[index] for index in path]
                key = _route_key(route_points)
                if key not in candidate_records:
                    lengths = [
                        math.dist(left, right)
                        for left, right in zip(route_points, route_points[1:])
                    ]
                    candidate_records[key] = {
                        "waypoints_xy_m": [list(point) for point in route_points],
                        "length_m": float(sum(lengths)),
                        "segment_lengths_m": lengths,
                        "segment_yaw_degrees": [
                            _normalized_yaw(
                                math.degrees(
                                    math.atan2(
                                        right[1] - left[1], right[0] - left[0]
                                    )
                                )
                            )
                            for left, right in zip(route_points, route_points[1:])
                        ],
                    }
                    retained_for_origin += 1
                return
            if segment_count >= segment_limit:
                return
            if (
                length_m
                + (segment_limit - segment_count) * maximum_segment_length
                < minimum_distance - 1.0e-9
            ):
                return
            current = path[-1]
            for next_index, edge_length in neighbors[current]:
                if next_index in path or length_m + edge_length > maximum_distance + 1.0e-9:
                    continue
                if len(path) >= 2 and not _turn_is_simple(
                    points[path[-2]], points[current], points[next_index]
                ):
                    continue
                if any(
                    _proper_segment_intersection(
                        points[path[index]],
                        points[path[index + 1]],
                        points[current],
                        points[next_index],
                    )
                    for index in range(max(0, len(path) - 2))
                ):
                    continue
                visit([*path, next_index], length_m + edge_length)
                if (
                    retained_for_origin >= per_origin_limit
                    or len(candidate_records) >= candidate_limit
                ):
                    return

        visit([origin_index], 0.0)
        if len(candidate_records) >= candidate_limit:
            break
    if not candidate_records:
        raise KujialeViewpointError(
            "room has no simple conservative-supercover polyline in the natural-distance "
            f"band [{minimum_distance:.3f}, {maximum_distance:.3f}] m"
        )
    candidates = list(candidate_records.values())
    candidates.sort(
        key=lambda item: (
            -len(item["segment_lengths_m"]),
            abs(float(item["length_m"]) - 0.5 * (minimum_distance + maximum_distance)),
            item["waypoints_xy_m"],
        )
    )
    # Round-robin by initial 30-degree direction provides geometric variety;
    # it is a sampling preference, never a legality condition.
    by_direction: dict[int, list[dict[str, Any]]] = {index: [] for index in range(12)}
    for candidate in candidates:
        initial_yaw = float(candidate["segment_yaw_degrees"][0]) % 360.0
        by_direction[min(11, int(initial_yaw // 30.0))].append(candidate)
    selected: list[dict[str, Any]] = []
    ordinal = 0
    while len(selected) < maximum:
        added = False
        for direction_index in range(12):
            values = by_direction[direction_index]
            if ordinal >= len(values):
                continue
            selected.append(values[ordinal])
            added = True
            if len(selected) == maximum:
                break
        if not added:
            break
        ordinal += 1
    floor = float(obstacle_map.floor_height_m)
    profiles = motion_profiles["profiles"]
    clip_seconds = float(motion_profiles["clip_seconds"])
    for index, corridor in enumerate(selected, start=1):
        waypoints_xy = corridor["waypoints_xy_m"]
        corridor.update(
            {
                "corridor_id": f"corridor_{index:04d}",
                "route_geometry": "polyline",
                "start_xy_m": list(waypoints_xy[0]),
                "end_xy_m": list(waypoints_xy[-1]),
                "waypoints_dataset_xyz_m": [
                    [float(point[0]), float(point[1]), floor]
                    for point in waypoints_xy
                ],
                "waypoints_habitat_m": [
                    [float(point[0]), floor, float(point[1])]
                    for point in waypoints_xy
                ],
                "start_world_m": [waypoints_xy[0][0], floor, waypoints_xy[0][1]],
                "end_world_m": [waypoints_xy[-1][0], floor, waypoints_xy[-1][1]],
                "natural_distance_band_m": [minimum_distance, maximum_distance],
                "nominal_clip_speed_mps": float(corridor["length_m"]) / clip_seconds,
                "species_profiles": list(species),
                "motion_profile_by_species": {
                    species_id: {
                        "walk_speed_mps_min": float(profiles[species_id]["walk_speed_mps_min"]),
                        "walk_speed_mps_max": float(profiles[species_id]["walk_speed_mps_max"]),
                        "clip_seconds": clip_seconds,
                    }
                    for species_id in species
                },
                "simple_non_repeated_vertices": True,
                "simple_non_repeated_edges": True,
                "segment_legality": "conservative_supercover_pass_for_both_source_masks",
                "authority": CPU_PREFILTER_AUTHORITY,
                "claim_boundary": CPU_PREFILTER_BOUNDARY,
            }
        )
    return selected


def _lerp_xy(
    start: Sequence[float], end: Sequence[float], fraction: float
) -> tuple[float, float]:
    return (
        float(start[0]) + (float(end[0]) - float(start[0])) * fraction,
        float(start[1]) + (float(end[1]) - float(start[1])) * fraction,
    )


def _corridor_waypoints_xy(corridor: Mapping[str, Any]) -> list[list[float]]:
    raw = corridor.get("waypoints_xy_m")
    if raw is None:
        raw = [corridor.get("start_xy_m"), corridor.get("end_xy_m")]
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or len(raw) < 2
    ):
        raise KujialeViewpointError("corridor waypoints are invalid")
    result = []
    for index, point in enumerate(raw):
        if (
            isinstance(point, (str, bytes))
            or not isinstance(point, Sequence)
            or len(point) != 2
        ):
            raise KujialeViewpointError(
                f"corridor waypoint {index} must contain two numbers"
            )
        result.append(
            [
                _finite(point[0], owner=f"corridor waypoint {index} x"),
                _finite(point[1], owner=f"corridor waypoint {index} y"),
            ]
        )
    return result


def _materialized_corridor(
    corridor: Mapping[str, Any], *, reverse: bool = False
) -> MaterializedRoute:
    cached = corridor.get("_reverse_materialized" if reverse else "_forward_materialized")
    if isinstance(cached, MaterializedRoute):
        return cached
    points_xy = _corridor_waypoints_xy(corridor)
    if reverse:
        points_xy.reverse()
    floor = float(corridor.get("waypoints_dataset_xyz_m", [[0.0, 0.0, 0.0]])[0][2])
    geometry = str(corridor.get("route_geometry", "straight"))
    return materialize_route(
        [[point[0], point[1], floor] for point in points_xy],
        75,
        geometry=geometry,
        owner=f"corridor {corridor.get('corridor_id', '<unknown>')}",
    )


def _bearing(camera_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    delta_x = float(target_xy[0]) - float(camera_xy[0])
    delta_y = float(target_xy[1]) - float(camera_xy[1])
    if math.hypot(delta_x, delta_y) <= 1.0e-9:
        return 0.0
    return _normalized_yaw(math.degrees(math.atan2(delta_y, delta_x)))


def _in_fov(
    camera_xy: Sequence[float],
    yaw_deg: float,
    target_xy: Sequence[float],
    hfov_deg: float,
) -> bool:
    if math.dist(camera_xy, target_xy) <= 1.0e-9:
        return True
    return (
        _yaw_delta(_bearing(camera_xy, target_xy), yaw_deg)
        <= hfov_deg * 0.5 + 1.0e-9
    )


def _side_value(
    camera_xy: Sequence[float], yaw_deg: float, target_xy: Sequence[float]
) -> float:
    theta = math.radians(yaw_deg)
    forward_x, forward_y = math.cos(theta), math.sin(theta)
    delta_x = float(target_xy[0]) - float(camera_xy[0])
    delta_y = float(target_xy[1]) - float(camera_xy[1])
    distance = math.hypot(delta_x, delta_y)
    if distance <= 1.0e-9:
        return 0.0
    return (forward_x * delta_y - forward_y * delta_x) / distance


def _coverage_jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _near_duplicate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    horizontal = math.dist(left["position_xy_m"], right["position_xy_m"])
    height = abs(float(left["height_m"]) - float(right["height_m"]))
    yaw = _yaw_delta(float(left["yaw_ue_deg"]), float(right["yaw_ue_deg"]))
    return (
        horizontal < 0.35
        and height < 0.20
        and yaw < 15.0
        and _coverage_jaccard(left["_visible"], right["_visible"]) > 0.95
    )


def _pose_diversity(
    candidate: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]
) -> float:
    if not selected:
        return 1.0
    return min(
        math.dist(candidate["position_xy_m"], item["position_xy_m"])
        + abs(float(candidate["height_m"]) - float(item["height_m"]))
        + _yaw_delta(
            float(candidate["yaw_ue_deg"]), float(item["yaw_ue_deg"])
        )
        / 90.0
        + (1.0 - _coverage_jaccard(candidate["_visible"], item["_visible"]))
        for item in selected
    )


def _candidate_rank(
    candidate: Mapping[str, Any],
    *,
    selected: Sequence[Mapping[str, Any]],
    covered_visible: set[int],
    covered_corridors: set[str],
) -> tuple[Any, ...]:
    new_corridors = len(candidate["_corridors"] - covered_corridors)
    new_visible = len(candidate["_visible"] - covered_visible)
    return (
        -int(candidate["_has_pair"]),
        -new_corridors,
        -new_visible,
        -round(_pose_diversity(candidate, selected), 9),
        -len(candidate["_corridors"]),
        -len(candidate["_visible"]),
        candidate["candidate_id"],
    )


def _direction_record(
    *,
    corridor: Mapping[str, Any],
    camera_xy: Sequence[float],
    yaw_deg: float,
    hfov_deg: float,
    side_margin: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for direction_id, reverse in (("forward", False), ("reverse", True)):
        materialized = _materialized_corridor(corridor, reverse=reverse)
        points = materialized.positions_m[:, :2]
        anchors = (points[4], points[22])
        in_fov = [
            _in_fov(camera_xy, yaw_deg, point, hfov_deg) for point in points
        ]
        side_values = [_side_value(camera_xy, yaw_deg, point) for point in anchors]
        result.append(
            {
                "direction": direction_id,
                "route_geometry": materialized.geometry,
                "start_xy_m": points[0].tolist(),
                "end_xy_m": points[-1].tolist(),
                "waypoints_xy_m": materialized.waypoints_m[:, :2].tolist(),
                "positions_xy_m": points.tolist(),
                "segment_index_by_frame": list(
                    materialized.segment_index_by_frame
                ),
                "tangent_yaw_degrees_by_frame": list(
                    materialized.tangent_yaw_degrees_by_frame
                ),
                "segment_boundaries": [
                    dict(item) for item in materialized.segment_boundaries
                ],
                "in_fov_by_frame": in_fov,
                "in_fov_frame_count": sum(in_fov),
                "all_frames_in_fov": all(in_fov),
                "anchor_side_values": side_values,
                "minimum_anchor_side_magnitude": min(
                    abs(value) for value in side_values
                ),
                "side_margin_reference": side_margin,
                "side_margin_satisfied": all(
                    abs(value) + 1.0e-12 >= side_margin for value in side_values
                ),
                "fov_and_side_role": "raw_difficulty_descriptors_not_legality_gates",
                "motion_profile_by_species": dict(
                    corridor.get("motion_profile_by_species", {})
                ),
                "_materialized": materialized,
            }
        )
    return result


def _continuous_pair_separation(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    left_start = np.asarray(left["start_xy_m"], dtype=np.float64)
    left_end = np.asarray(left["end_xy_m"], dtype=np.float64)
    right_start = np.asarray(right["start_xy_m"], dtype=np.float64)
    right_end = np.asarray(right["end_xy_m"], dtype=np.float64)
    relative_start = left_start - right_start
    relative_velocity = (left_end - left_start) - (right_end - right_start)
    velocity_squared = float(np.dot(relative_velocity, relative_velocity))
    if velocity_squared <= 1.0e-15:
        fraction = 0.0
    else:
        fraction = min(
            1.0,
            max(
                0.0,
                -float(np.dot(relative_start, relative_velocity))
                / velocity_squared,
            ),
        )
    separation = relative_start + fraction * relative_velocity
    return float(np.linalg.norm(separation))


def _compatible_direction_pairs(
    direction_by_corridor: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_separation_m: float,
    maximum_records: int,
    listener_position_xy_m: Sequence[float] | None = None,
    listener_yaw_degrees: float | None = None,
    stop_after_count: int | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    directed = [
        {
            "direction_id": f"{corridor_id}:{direction['direction']}",
            "corridor_id": corridor_id,
            **dict(direction),
        }
        for corridor_id in sorted(direction_by_corridor)
        for direction in direction_by_corridor[corridor_id]
    ]
    total_count = len(directed) * (len(directed) - 1) // 2
    candidate_pairs = [
        (left, right)
        for left_index, left in enumerate(directed)
        for right in directed[left_index + 1 :]
    ]
    candidate_pairs.sort(
        key=lambda value: (
            value[0]["corridor_id"] != value[1]["corridor_id"],
            value[0]["direction_id"],
            value[1]["direction_id"],
        )
    )
    pairs: list[dict[str, Any]] = []
    record_limit = min(maximum_records, stop_after_count or maximum_records)
    for left, right in candidate_pairs[:record_limit]:
        descriptor = describe_route_pair(
            left["_materialized"],
            right["_materialized"],
            fps=15,
            listener_position_xy_m=listener_position_xy_m,
            listener_yaw_degrees=listener_yaw_degrees,
            proximity_reference_m=minimum_separation_m,
        )
        pairs.append(
            {
                "source1_direction_id": left["direction_id"],
                "source2_direction_id": right["direction_id"],
                "minimum_continuous_linear_separation_m": descriptor[
                    "closest_continuous_plan"
                ]["distance_m"],
                "opposite_sides_at_first_anchor": (
                    float(left["anchor_side_values"][0])
                    * float(right["anchor_side_values"][0])
                    < 0.0
                ),
                "raw_descriptors": descriptor,
            }
        )
    pairs.sort(
        key=lambda item: (
            -round(float(item["minimum_continuous_linear_separation_m"]), 9),
            item["source1_direction_id"],
            item["source2_direction_id"],
        )
    )
    return total_count, pairs[:maximum_records]


def _serialize_direction(direction: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in direction.items() if not key.startswith("_")}


def _actor_route_from_materialized(
    materialized: MaterializedRoute,
    *,
    actor_id: str,
    species_id: str,
    root_height_m: float,
    motion_profile: Mapping[str, Any],
) -> tuple[MaterializedRoute, dict[str, Any]]:
    points = np.asarray(materialized.waypoints_m, dtype=np.float64).copy()
    points[:, 2] = root_height_m
    if materialized.moving:
        adjusted = materialize_route(
            points,
            75,
            geometry=materialized.geometry,
            owner=f"{actor_id} generated route",
        )
        profile = {
            "route_geometry": adjusted.geometry,
            "waypoints_xyz_m": adjusted.waypoints_m.tolist(),
            "motion_profile": {
                "species_id": species_id,
                **dict(motion_profile),
            },
        }
    else:
        heading = materialized.tangent_yaw_degrees_by_frame[0]
        adjusted = materialize_route(
            points,
            75,
            geometry="hold",
            heading_yaw_degrees=heading,
            owner=f"{actor_id} generated hold",
        )
        profile = {
            "route_geometry": "hold",
            "hold_xyz_m": adjusted.waypoints_m[0].tolist(),
            "heading_yaw_ue_deg": heading,
        }
    return adjusted, profile


def _hold_at_route_start(materialized: MaterializedRoute) -> MaterializedRoute:
    return materialize_route(
        [materialized.waypoints_m[0]],
        75,
        geometry="hold",
        heading_yaw_degrees=materialized.tangent_yaw_degrees_by_frame[0],
        owner="generated static route",
    )


def _motion_case_records(
    direction_by_corridor: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    listener_position_xy_m: Sequence[float],
    listener_yaw_degrees: float,
    proximity_reference_m: float,
) -> list[dict[str, Any]]:
    directed = [
        direction
        for corridor_id in sorted(direction_by_corridor)
        for direction in direction_by_corridor[corridor_id]
    ]
    if len(directed) < 2:
        raise KujialeViewpointError("at least two directed routes are required")
    source1_moving = directed[0]["_materialized"]
    # Same-corridor reverse is deliberately preferred: endpoint exchange and
    # close approach are retained raw cases rather than filtered away.
    source2_moving = directed[1]["_materialized"]
    source1_hold = _hold_at_route_start(source1_moving)
    source2_hold = _hold_at_route_start(source2_moving)
    cases = (
        ("static_static", source1_hold, source2_hold),
        ("source1_moving_source2_static", source1_moving, source2_hold),
        ("source1_static_source2_moving", source1_hold, source2_moving),
        ("both_moving", source1_moving, source2_moving),
    )
    result = []
    for motion_case, human_route, dog_route in cases:
        first_direction = directed[0]
        human_profile = first_direction.get("motion_profile_by_species", {}).get(
            "human",
            {
                "walk_speed_mps_min": 1.0e-9,
                "walk_speed_mps_max": 1.0e6,
                "clip_seconds": 5.0,
            },
        )
        dog_profile = first_direction.get("motion_profile_by_species", {}).get(
            "dog",
            {
                "walk_speed_mps_min": 1.0e-9,
                "walk_speed_mps_max": 1.0e6,
                "clip_seconds": 5.0,
            },
        )
        human_adjusted, human_request = _actor_route_from_materialized(
            human_route,
            actor_id="human0",
            species_id="human",
            root_height_m=0.064,
            motion_profile=human_profile,
        )
        dog_adjusted, dog_request = _actor_route_from_materialized(
            dog_route,
            actor_id="dog0",
            species_id="dog",
            root_height_m=0.0,
            motion_profile=dog_profile,
        )
        descriptors = describe_route_pair(
            human_adjusted,
            dog_adjusted,
            fps=15,
            listener_position_xy_m=listener_position_xy_m,
            listener_yaw_degrees=listener_yaw_degrees,
            proximity_reference_m=proximity_reference_m,
        )
        result.append(
            {
                "motion_case": motion_case,
                "source_actor_identity": {
                    "source1": {"source_id": "m6z_human0_mouth", "actor_id": "human0"},
                    "source2": {"source_id": "m6z_dog0_muzzle", "actor_id": "dog0"},
                    "identity_changes_during_route": False,
                },
                "routes": {"human0": human_request, "dog0": dog_request},
                "raw_descriptors": descriptors,
                "legality_excludes": [
                    "continuous_0.60m",
                    "side_margin",
                    "crossing_or_exchange",
                    "camera_fov_or_occlusion",
                ],
            }
        )
    return result


def build_viewpoint_plan(
    *,
    room_id: str,
    room_type: str,
    region_by_source: Mapping[str, FeasibleRegionIndex],
    corridors: Sequence[Mapping[str, Any]],
    view_count: int = 4,
    hfov_degrees: Real = 105.0,
    camera_heights_m: Sequence[Real] = (1.05, 1.50, 1.85),
    camera_spacing_m: Real = 0.50,
    coverage_spacing_m: Real = 0.25,
    yaw_step_degrees: Real = 30.0,
    side_margin: Real = 0.06,
    maximum_camera_positions: int = 64,
    maximum_coverage_samples: int = 256,
    screening_corridor_count: int = 64,
    minimum_pair_separation_m: Real = 0.60,
    maximum_pair_records: int = 256,
) -> dict[str, Any]:
    """Select 4--8 CPU requests and attach post-hoc route descriptors.

    Camera legality comes only from the retained feasible raster.  Visibility,
    side, crossing/exchange and pair distance can influence descriptive ranking
    but never remove a legal camera or source route.
    """

    requested = _count(view_count, owner="view_count", minimum=4)
    if requested > 8:
        raise KujialeViewpointError("view_count must lie in [4, 8]")
    hfov = _positive(hfov_degrees, owner="hfov_degrees")
    if hfov >= 180.0:
        raise KujialeViewpointError("hfov_degrees must be smaller than 180")
    camera_spacing = _positive(camera_spacing_m, owner="camera_spacing_m")
    coverage_spacing = _positive(coverage_spacing_m, owner="coverage_spacing_m")
    yaw_step = _positive(yaw_step_degrees, owner="yaw_step_degrees")
    if yaw_step > 90.0:
        raise KujialeViewpointError("yaw_step_degrees must not exceed 90")
    margin = _finite(side_margin, owner="side_margin")
    if not 0.0 <= margin < 1.0:
        raise KujialeViewpointError("side_margin must lie in [0, 1)")
    camera_limit = _count(
        maximum_camera_positions, owner="maximum_camera_positions", minimum=8
    )
    coverage_limit = _count(
        maximum_coverage_samples, owner="maximum_coverage_samples", minimum=16
    )
    corridor_limit = _count(
        screening_corridor_count, owner="screening_corridor_count", minimum=4
    )
    pair_separation = _positive(
        minimum_pair_separation_m, owner="minimum_pair_separation_m"
    )
    pair_record_limit = _count(
        maximum_pair_records, owner="maximum_pair_records", minimum=4
    )
    heights = sorted(
        {_positive(value, owner="camera height") for value in camera_heights_m}
    )
    if not heights:
        raise KujialeViewpointError("camera_heights_m must be non-empty")
    room_key = _identifier(room_id, owner="room_id")
    obstacle_map, common = _common_region(region_by_source)
    pixel_x, pixel_z = _pixel_sizes(obstacle_map, common.shape)
    coverage_pixels = _limit_pixels(
        _sample_pixels(common, coverage_spacing, min(pixel_x, pixel_z)),
        coverage_limit,
    )
    camera_pixels = _limit_pixels(
        _sample_pixels(common, camera_spacing, min(pixel_x, pixel_z)),
        camera_limit,
    )
    coverage_points = [
        _pixel_to_xy(obstacle_map, common.shape, pixel)
        for pixel in coverage_pixels
    ]
    camera_points = [
        _pixel_to_xy(obstacle_map, common.shape, pixel) for pixel in camera_pixels
    ]
    if not camera_points:
        raise KujialeViewpointError("room has no sampled camera positions")
    corridor_values = [dict(item) for item in corridors]
    if not corridor_values:
        raise KujialeViewpointError("corridors must be non-empty")
    for corridor in corridor_values:
        corridor["_forward_materialized"] = _materialized_corridor(corridor)
        corridor["_reverse_materialized"] = _materialized_corridor(
            corridor, reverse=True
        )
    screening_corridors = corridor_values[:corridor_limit]

    yaw_grid = {
        _normalized_yaw(-180.0 + index * yaw_step)
        for index in range(int(math.ceil(360.0 / yaw_step)))
    }
    candidates: list[dict[str, Any]] = []
    for camera_index, camera_xy in enumerate(camera_points):
        los_targets = {
            index
            for index, target in enumerate(coverage_points)
            if _segment_clear(obstacle_map, common, camera_xy, target)
        }
        for yaw in sorted(yaw_grid):
            visible = {
                index
                for index in los_targets
                if _in_fov(camera_xy, yaw, coverage_points[index], hfov)
            }
            preferred_corridors: set[str] = set()
            for corridor in screening_corridors:
                materialized = corridor["_forward_materialized"]
                probe_indices = (0, len(materialized.positions_m) // 2, 74)
                if any(
                    _in_fov(
                        camera_xy,
                        yaw,
                        materialized.positions_m[probe_index, :2],
                        hfov,
                    )
                    for probe_index in probe_indices
                ):
                    preferred_corridors.add(str(corridor["corridor_id"]))
            for height_index, height in enumerate(heights):
                candidates.append(
                    {
                        "candidate_id": (
                            f"c{camera_index:04d}_h{height_index:02d}_"
                            f"y{int(round((yaw + 180.0) * 10.0)):04d}"
                        ),
                        "position_xy_m": [
                            float(camera_xy[0]),
                            float(camera_xy[1]),
                        ],
                        "height_m": height,
                        "yaw_ue_deg": yaw,
                        "_visible": visible,
                        "_corridors": preferred_corridors,
                        "_has_pair": bool(screening_corridors),
                    }
                )
    if not candidates:
        raise KujialeViewpointError(
            "no camera pose sees a clear natural-speed corridor with side margin"
        )
    candidates.sort(key=lambda item: item["candidate_id"])

    selected: list[dict[str, Any]] = []
    covered_visible: set[int] = set()
    covered_corridors: set[str] = set()

    def choose(pool: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        available = [
            item
            for item in pool
            if item not in selected
            and not any(_near_duplicate(item, retained) for retained in selected)
        ]
        if not available:
            return None
        return min(
            available,
            key=lambda item: _candidate_rank(
                item,
                selected=selected,
                covered_visible=covered_visible,
                covered_corridors=covered_corridors,
            ),
        )

    tier_order: list[float] = []
    median_height = heights[len(heights) // 2]
    for value in (median_height, heights[0], heights[-1]):
        if value not in tier_order:
            tier_order.append(value)
    for height in tier_order:
        if len(selected) >= requested:
            break
        candidate = choose(
            [item for item in candidates if float(item["height_m"]) == height]
        )
        if candidate is None:
            continue
        selected.append(candidate)
        covered_visible |= candidate["_visible"]
        covered_corridors |= candidate["_corridors"]
    while len(selected) < requested:
        candidate = choose(candidates)
        if candidate is None:
            break
        selected.append(candidate)
        covered_visible |= candidate["_visible"]
        covered_corridors |= candidate["_corridors"]
    if len(selected) != requested:
        raise KujialeViewpointError(
            f"only {len(selected)} non-duplicate viewpoints qualify; requested {requested}"
        )

    feasible_area = float(np.count_nonzero(common) * pixel_x * pixel_z)
    coverage_sample_area = feasible_area / len(coverage_points)
    viewpoint_records: list[dict[str, Any]] = []
    subsets: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(selected, start=1):
        viewpoint_id = f"{room_key}_view_{index:02d}"
        position_xy = candidate["position_xy_m"]
        height = float(candidate["height_m"])
        yaw_ue = float(candidate["yaw_ue_deg"])
        complete_directions: dict[str, list[dict[str, Any]]] = {}
        for corridor in corridor_values:
            directions = _direction_record(
                corridor=corridor,
                camera_xy=position_xy,
                yaw_deg=yaw_ue,
                hfov_deg=hfov,
                side_margin=margin,
            )
            complete_directions[str(corridor["corridor_id"])] = directions
        corridor_ids = sorted(complete_directions)
        if not corridor_ids:
            raise KujialeViewpointError(
                f"selected candidate {candidate['candidate_id']} lost all corridors "
                "during complete re-evaluation"
            )
        compatible_pair_count, compatible_pairs = _compatible_direction_pairs(
            complete_directions,
            minimum_separation_m=pair_separation,
            maximum_records=pair_record_limit,
            listener_position_xy_m=position_xy,
            listener_yaw_degrees=yaw_ue,
        )
        motion_cases = _motion_case_records(
            complete_directions,
            listener_position_xy_m=position_xy,
            listener_yaw_degrees=yaw_ue,
            proximity_reference_m=pair_separation,
        )
        viewpoint_records.append(
            {
                "viewpoint_id": viewpoint_id,
                "position_xyz_m": [position_xy[0], position_xy[1], height],
                "position_ue_cm": [
                    position_xy[0] * 100.0,
                    position_xy[1] * 100.0,
                    height * 100.0,
                ],
                "position_habitat_m": [position_xy[0], height, position_xy[1]],
                "yaw_ue_deg": yaw_ue,
                "yaw_habitat_deg": _normalized_yaw(-90.0 - yaw_ue),
                "horizontal_fov_deg": hfov,
                "visible_feasible_sample_count": len(candidate["_visible"]),
                "estimated_visible_feasible_area_m2": (
                    len(candidate["_visible"]) * coverage_sample_area
                ),
                "eligible_corridor_count": len(corridor_ids),
                "geometry_legal_corridor_count": len(corridor_ids),
                "compatible_both_moving_pair_count": compatible_pair_count,
                "motion_case_count": len(motion_cases),
                "corridor_subset": f"corridors/{viewpoint_id}.json",
                "native_pixel_validation_status": "pending",
                "ue_spear_certification_status": "pending",
                "mini_pilot_status": "pending",
            }
        )
        subsets[viewpoint_id] = {
            "status": "cpu_prefilter_complete",
            "viewpoint_id": viewpoint_id,
            "authority": CPU_PREFILTER_AUTHORITY,
            "claim_boundary": CPU_PREFILTER_BOUNDARY,
            "side_margin": margin,
            "minimum_pair_separation_m": pair_separation,
            "difficulty_descriptor_policy": (
                "side, 0.60m, crossing, exchange, FOV and occlusion are raw "
                "descriptors only and never remove a geometry-valid route"
            ),
            "corridors": [
                {
                    "corridor_id": corridor_id,
                    "directions": [
                        _serialize_direction(direction)
                        for direction in complete_directions[corridor_id]
                    ],
                }
                for corridor_id in corridor_ids
            ],
            "compatible_both_moving_pair_count": compatible_pair_count,
            "compatible_both_moving_pairs": compatible_pairs,
            "four_point_motion_cases": motion_cases,
        }

    return {
        "bank": {
            "status": "cpu_prefilter_complete",
            "research_only": True,
            "episode_counted": False,
            "formal_dataset_count": 0,
            "qualification_claim": False,
            "room_id": room_id,
            "room_type": room_type,
            "authority": CPU_PREFILTER_AUTHORITY,
            "claim_boundary": CPU_PREFILTER_BOUNDARY,
            "navigation_authority": obstacle_map.authority,
            "navigation_is_habitat_navmesh": False,
            "native_pixel_validation_status": "pending",
            "ue_spear_certification_status": "pending",
            "viewpoint_count": len(viewpoint_records),
            "selection_policy": (
                "geometry-valid camera pool with pose/coverage preferences; "
                "difficulty clues are not legality gates"
            ),
            "parameters": {
                "hfov_degrees": hfov,
                "camera_heights_m": heights,
                "camera_spacing_m": camera_spacing,
                "coverage_spacing_m": coverage_spacing,
                "yaw_step_degrees": yaw_step,
                "side_margin": margin,
                "maximum_camera_positions": camera_limit,
                "maximum_coverage_samples": coverage_limit,
                "screening_corridor_count": min(
                    corridor_limit, len(corridor_values)
                ),
                "minimum_pair_separation_m": pair_separation,
                "maximum_pair_records": pair_record_limit,
            },
            "feasible_area_m2": feasible_area,
            "corridor_count": len(corridor_values),
            "viewpoints": viewpoint_records,
        },
        "corridor_subsets": subsets,
    }


__all__ = [
    "CPU_PREFILTER_AUTHORITY",
    "CPU_PREFILTER_BOUNDARY",
    "KujialeViewpointError",
    "build_viewpoint_plan",
    "generate_polyline_corridors",
    "generate_straight_corridors",
    "natural_distance_band",
]
