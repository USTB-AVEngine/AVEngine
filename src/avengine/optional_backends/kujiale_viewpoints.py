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
ROUTE_FRAME_COUNT = 75
ROUTE_FPS = 15
ROUTE_MOVEMENT_SPAN_SECONDS = (ROUTE_FRAME_COUNT - 1) / ROUTE_FPS


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


def _shared_obstacle_map(
    region_by_source: Mapping[str, FeasibleRegionIndex],
) -> RuntimeObstacleMap:
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
    return first.obstacle_map


def _common_region(
    region_by_source: Mapping[str, FeasibleRegionIndex],
) -> tuple[RuntimeObstacleMap, np.ndarray]:
    obstacle_map = _shared_obstacle_map(region_by_source)
    first = region_by_source["source1"]
    second = region_by_source["source2"]
    common = np.asarray(first.feasible_mask & second.feasible_mask, dtype=np.bool_)
    if not np.any(common):
        raise KujialeViewpointError("source-region intersection is empty")
    return obstacle_map, np.ascontiguousarray(common)


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
    _declared_clip_seconds = _positive(
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
            * ROUTE_MOVEMENT_SPAN_SECONDS
        )
        maximums.append(
            _positive(
                profile.get("walk_speed_mps_max"),
                owner=f"{species_id}.walk_speed_mps_max",
            )
            * ROUTE_MOVEMENT_SPAN_SECONDS
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
    declared_clip_seconds = float(motion_profiles["clip_seconds"])
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
        corridor["replay_speed_mps"] = (
            float(corridor["length_m"]) / ROUTE_MOVEMENT_SPAN_SECONDS
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
                "clip_seconds": declared_clip_seconds,
                "movement_span_seconds": ROUTE_MOVEMENT_SPAN_SECONDS,
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


def _segments_intersect_or_touch(
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

    def on_segment(
        start: Sequence[float], point: Sequence[float], end: Sequence[float]
    ) -> bool:
        return (
            min(float(start[0]), float(end[0])) - 1.0e-12
            <= float(point[0])
            <= max(float(start[0]), float(end[0])) + 1.0e-12
            and min(float(start[1]), float(end[1])) - 1.0e-12
            <= float(point[1])
            <= max(float(start[1]), float(end[1])) + 1.0e-12
        )

    first = orientation(a0, a1, b0)
    second = orientation(a0, a1, b1)
    third = orientation(b0, b1, a0)
    fourth = orientation(b0, b1, a1)
    if first * second < -1.0e-12 and third * fourth < -1.0e-12:
        return True
    return (
        (abs(first) <= 1.0e-12 and on_segment(a0, b0, a1))
        or (abs(second) <= 1.0e-12 and on_segment(a0, b1, a1))
        or (abs(third) <= 1.0e-12 and on_segment(b0, a0, b1))
        or (abs(fourth) <= 1.0e-12 and on_segment(b0, a1, b1))
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
                    _segments_intersect_or_touch(
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
    declared_clip_seconds = float(motion_profiles["clip_seconds"])
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
                "replay_speed_mps": (
                    float(corridor["length_m"]) / ROUTE_MOVEMENT_SPAN_SECONDS
                ),
                "species_profiles": list(species),
                "motion_profile_by_species": {
                    species_id: {
                        "walk_speed_mps_min": float(profiles[species_id]["walk_speed_mps_min"]),
                        "walk_speed_mps_max": float(profiles[species_id]["walk_speed_mps_max"]),
                        "clip_seconds": declared_clip_seconds,
                        "movement_span_seconds": ROUTE_MOVEMENT_SPAN_SECONDS,
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


def _describe_directed_pairs(
    direction_by_corridor: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_separation_m: float,
    maximum_records: int,
    listener_position_xy_m: Sequence[float] | None = None,
    listener_yaw_degrees: float | None = None,
) -> dict[str, Any]:
    directed = [
        {
            "direction_id": f"{corridor_id}:{direction['direction']}",
            "corridor_id": corridor_id,
            **dict(direction),
        }
        for corridor_id in sorted(direction_by_corridor)
        for direction in direction_by_corridor[corridor_id]
    ]
    combination_count = len(directed) * len(directed)
    candidate_pairs = [
        (left, right)
        for left in directed
        for right in directed
    ]
    candidate_pairs.sort(
        key=lambda value: (
            value[0]["corridor_id"] != value[1]["corridor_id"],
            value[0]["direction"] == value[1]["direction"],
            value[0]["direction_id"],
            value[1]["direction_id"],
        )
    )
    pairs: list[dict[str, Any]] = []
    for left, right in candidate_pairs[:maximum_records]:
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
                "continuous_closest_planar_distance_m": descriptor[
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
            -round(float(item["continuous_closest_planar_distance_m"]), 9),
            item["source1_direction_id"],
            item["source2_direction_id"],
        )
    )
    return {
        "directed_pair_combination_count": combination_count,
        "described_pair_count": len(pairs),
        "described_pair_records": pairs,
        "description_scope": (
            "only described_pair_records carry computed pair clues; the larger "
            "directed combination count is arithmetic and not a compatibility claim"
        ),
    }


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
    route_pair_ordinal: int,
) -> list[dict[str, Any]]:
    corridor_ids = sorted(direction_by_corridor)
    if route_pair_ordinal >= len(corridor_ids):
        raise KujialeViewpointError(
            "distinct per-view route request exceeds available corridors"
        )
    selected_corridor_id = corridor_ids[route_pair_ordinal]
    directed = list(direction_by_corridor[selected_corridor_id])
    if len(directed) != 2:
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
                "movement_span_seconds": ROUTE_MOVEMENT_SPAN_SECONDS,
            },
        )
        dog_profile = first_direction.get("motion_profile_by_species", {}).get(
            "dog",
            {
                "walk_speed_mps_min": 1.0e-9,
                "walk_speed_mps_max": 1.0e6,
                "clip_seconds": 5.0,
                "movement_span_seconds": ROUTE_MOVEMENT_SPAN_SECONDS,
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
                "route_selection": {
                    "corridor_id": selected_corridor_id,
                    "selection_ordinal": route_pair_ordinal,
                    "selection_role": "deterministic_geometric_variety_not_legality_gate",
                },
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


def validate_camera_catalog_for_viewpoint_selection(
    *,
    camera_catalog: Mapping[str, Any],
    scene_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Cross-check and return the complete target-agnostic camera pool."""

    catalog_binding = camera_catalog.get("scene_binding")
    if not isinstance(catalog_binding, Mapping):
        raise KujialeViewpointError("camera catalog lacks scene_binding")
    for key in (
        "dataset_id",
        "scene_id",
        "room_id",
        "room_scope",
        "map_path",
        "source_stage_name",
    ):
        if catalog_binding.get(key) != scene_binding.get(key):
            raise KujialeViewpointError(
                f"camera catalog and current scene binding {key} differ"
            )
    if camera_catalog.get("room_id") != scene_binding.get("room_id"):
        raise KujialeViewpointError("camera catalog room_id differs")
    if camera_catalog.get("room_scope") != scene_binding.get("room_scope"):
        raise KujialeViewpointError("camera catalog room_scope differs")
    if camera_catalog.get("candidate_filtering") != "none":
        raise KujialeViewpointError("camera catalog must retain the unfiltered pool")
    if camera_catalog.get("studio_used") is not False:
        raise KujialeViewpointError("camera catalog must declare studio_used=false")
    raw_heights = camera_catalog.get("camera_height_variants_above_floor_m")
    raw_yaws = camera_catalog.get("yaw_degrees")
    raw_placements = camera_catalog.get("placements")
    if not isinstance(raw_heights, Mapping) or not raw_heights:
        raise KujialeViewpointError("camera catalog lacks height variants")
    if (
        isinstance(raw_yaws, (str, bytes))
        or not isinstance(raw_yaws, Sequence)
        or not raw_yaws
    ):
        raise KujialeViewpointError("camera catalog lacks yaw variants")
    if not isinstance(raw_placements, list) or not raw_placements:
        raise KujialeViewpointError("camera catalog has no placements")
    heights = {
        str(height_id): _positive(value, owner=f"camera height {height_id}")
        for height_id, value in raw_heights.items()
    }
    yaws = {_normalized_yaw(value) for value in raw_yaws}
    obstacle_map = camera_catalog.get("obstacle_map")
    if not isinstance(obstacle_map, Mapping):
        raise KujialeViewpointError("camera catalog lacks obstacle_map")
    floor = _finite(obstacle_map.get("floor_height_m"), owner="camera catalog floor")
    placement_ids: set[str] = set()
    poses: set[tuple[float, float, float, float]] = set()
    by_floor_position: dict[tuple[int, int, float, float], set[tuple[str, float]]] = {}
    records = []
    for index, raw in enumerate(raw_placements):
        if not isinstance(raw, Mapping):
            raise KujialeViewpointError(f"camera placement {index} must be an object")
        placement_id = _identifier(
            raw.get("placement_id"), owner=f"camera placement {index} id"
        )
        if placement_id in placement_ids:
            raise KujialeViewpointError("camera placement ids must be unique")
        placement_ids.add(placement_id)
        position = raw.get("position_m")
        if (
            isinstance(position, (str, bytes))
            or not isinstance(position, Sequence)
            or len(position) != 3
        ):
            raise KujialeViewpointError(
                f"camera placement {placement_id} position_m is invalid"
            )
        xyz = [
            _finite(value, owner=f"camera placement {placement_id} position")
            for value in position
        ]
        yaw = _normalized_yaw(raw.get("yaw_deg"))
        height_id = str(raw.get("height_id"))
        if height_id not in heights or yaw not in yaws:
            raise KujialeViewpointError(
                f"camera placement {placement_id} uses an undeclared height or yaw"
            )
        if not math.isclose(xyz[1] - floor, heights[height_id], abs_tol=1.0e-9):
            raise KujialeViewpointError(
                f"camera placement {placement_id} height differs from catalog"
            )
        pixel = raw.get("raster_pixel_rc")
        if (
            isinstance(pixel, (str, bytes))
            or not isinstance(pixel, Sequence)
            or len(pixel) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pixel)
        ):
            raise KujialeViewpointError(
                f"camera placement {placement_id} raster pixel is invalid"
            )
        if (
            raw.get("room_instance_id") != scene_binding.get("room_id")
            or raw.get("room_scope") != scene_binding.get("room_scope")
            or raw.get("placement_status") != "camera_center_feasible_only"
        ):
            raise KujialeViewpointError(
                f"camera placement {placement_id} identity or status differs"
            )
        if (
            raw.get("semantic_target") is not None
            or raw.get("source_path_join_status") != "pending"
            or raw.get("difficulty_status") != "not_evaluated"
        ):
            raise KujialeViewpointError(
                f"camera placement {placement_id} is not target-agnostic"
            )
        pose = (xyz[0], xyz[2], xyz[1], yaw)
        if pose in poses:
            raise KujialeViewpointError("camera catalog contains duplicate poses")
        poses.add(pose)
        floor_key = (int(pixel[0]), int(pixel[1]), xyz[0], xyz[2])
        by_floor_position.setdefault(floor_key, set()).add((height_id, yaw))
        records.append(
            {
                "placement_id": placement_id,
                "position_xy_m": [xyz[0], xyz[2]],
                "height_m": xyz[1],
                "height_id": height_id,
                "yaw_ue_deg": yaw,
                "raster_pixel_rc": [int(pixel[0]), int(pixel[1])],
                "room_type_annotation": raw.get("room_type_annotation"),
            }
        )
    declared_count = camera_catalog.get("placement_count")
    if declared_count != len(records):
        raise KujialeViewpointError("camera catalog placement_count differs")
    expected_variants = {(height_id, yaw) for height_id in heights for yaw in yaws}
    if any(variants != expected_variants for variants in by_floor_position.values()):
        raise KujialeViewpointError(
            "camera catalog does not completely cross every floor position, height and yaw"
        )
    if camera_catalog.get("sampled_floor_position_count") != len(by_floor_position):
        raise KujialeViewpointError(
            "camera catalog sampled_floor_position_count differs"
        )
    return records


def build_viewpoint_plan(
    *,
    room_id: str,
    room_type: str,
    scene_binding: Mapping[str, Any],
    camera_catalog: Mapping[str, Any],
    region_by_source: Mapping[str, FeasibleRegionIndex],
    corridors: Sequence[Mapping[str, Any]],
    view_count: int = 4,
    hfov_degrees: Real = 105.0,
    side_margin: Real = 0.06,
    screening_corridor_count: int = 64,
    minimum_pair_separation_m: Real = 0.60,
    maximum_pair_records: int = 256,
) -> dict[str, Any]:
    """Select 4--8 CPU requests and attach post-hoc route descriptors.

    Camera legality and the complete pose pool come only from the supplied
    target-agnostic camera catalog. Source masks generate corridors only.
    Visibility, LOS, side, crossing/exchange and pair distance can influence
    descriptive ranking but never remove a legal camera or source route.
    """

    requested = _count(view_count, owner="view_count", minimum=4)
    if requested > 8:
        raise KujialeViewpointError("view_count must lie in [4, 8]")
    hfov = _positive(hfov_degrees, owner="hfov_degrees")
    if hfov >= 180.0:
        raise KujialeViewpointError("hfov_degrees must be smaller than 180")
    margin = _finite(side_margin, owner="side_margin")
    if not 0.0 <= margin < 1.0:
        raise KujialeViewpointError("side_margin must lie in [0, 1)")
    corridor_limit = _count(
        screening_corridor_count, owner="screening_corridor_count", minimum=4
    )
    pair_separation = _positive(
        minimum_pair_separation_m, owner="minimum_pair_separation_m"
    )
    pair_record_limit = _count(
        maximum_pair_records, owner="maximum_pair_records", minimum=4
    )
    room_key = _identifier(room_id, owner="room_id")
    if scene_binding.get("room_id") != room_id:
        raise KujialeViewpointError("room_id differs from current scene binding")
    camera_placements = validate_camera_catalog_for_viewpoint_selection(
        camera_catalog=camera_catalog,
        scene_binding=scene_binding,
    )
    heights = sorted({float(item["height_m"]) for item in camera_placements})
    obstacle_map = _shared_obstacle_map(region_by_source)
    route_los_mask = np.asarray(obstacle_map.binary_navmesh, dtype=np.bool_)
    corridor_values = [dict(item) for item in corridors]
    if not corridor_values:
        raise KujialeViewpointError("corridors must be non-empty")
    if len(corridor_values) < requested:
        raise KujialeViewpointError(
            f"only {len(corridor_values)} distinct corridors exist; "
            f"requested {requested} per-view routes"
        )
    for corridor in corridor_values:
        corridor["_forward_materialized"] = _materialized_corridor(corridor)
        corridor["_reverse_materialized"] = _materialized_corridor(
            corridor, reverse=True
        )
    screening_corridors = corridor_values[:corridor_limit]

    candidates: list[dict[str, Any]] = []
    for placement in camera_placements:
        camera_xy = placement["position_xy_m"]
        yaw = float(placement["yaw_ue_deg"])
        visible: set[int] = set()
        preferred_corridors: set[str] = set()
        for corridor_index, corridor in enumerate(screening_corridors):
            materialized = corridor["_forward_materialized"]
            probe_indices = (0, len(materialized.positions_m) // 2, 74)
            for local_index, probe_index in enumerate(probe_indices):
                target = materialized.positions_m[probe_index, :2]
                if _in_fov(camera_xy, yaw, target, hfov) and _segment_clear(
                    obstacle_map, route_los_mask, camera_xy, target
                ):
                    visible.add(corridor_index * len(probe_indices) + local_index)
                    preferred_corridors.add(str(corridor["corridor_id"]))
        candidates.append(
            {
                "candidate_id": placement["placement_id"],
                "camera_catalog_placement_id": placement["placement_id"],
                "position_xy_m": list(camera_xy),
                "height_m": float(placement["height_m"]),
                "height_id": placement["height_id"],
                "yaw_ue_deg": yaw,
                "raster_pixel_rc": list(placement["raster_pixel_rc"]),
                "_visible": visible,
                "_corridors": preferred_corridors,
                "_has_pair": bool(screening_corridors),
            }
        )
    if not candidates:
        raise KujialeViewpointError(
            "camera catalog contains no finite target-agnostic placements"
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

    camera_pixel_size = _positive(
        camera_catalog.get("meters_per_pixel"),
        owner="camera catalog meters_per_pixel",
    )
    feasible_area = (
        _count(
            camera_catalog.get("free_space_raster_pixel_count"),
            owner="camera catalog free_space_raster_pixel_count",
        )
        * camera_pixel_size
        * camera_pixel_size
    )
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
        pair_descriptions = _describe_directed_pairs(
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
            route_pair_ordinal=index - 1,
        )
        viewpoint_records.append(
            {
                "viewpoint_id": viewpoint_id,
                "camera_catalog_placement_id": candidate[
                    "camera_catalog_placement_id"
                ],
                "camera_catalog_raster_pixel_rc": candidate["raster_pixel_rc"],
                "camera_height_id": candidate["height_id"],
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
                "visible_route_probe_count": len(candidate["_visible"]),
                "eligible_corridor_count": len(corridor_ids),
                "geometry_legal_corridor_count": len(corridor_ids),
                "directed_pair_combination_count": pair_descriptions[
                    "directed_pair_combination_count"
                ],
                "described_pair_count": pair_descriptions[
                    "described_pair_count"
                ],
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
            **pair_descriptions,
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
            "camera_pool_authority": camera_catalog.get("authority"),
            "camera_pool_candidate_filtering": camera_catalog.get(
                "candidate_filtering"
            ),
            "camera_pool_placement_count": len(camera_placements),
            "camera_pool_is_complete_declared_catalog": True,
            "source_route_navigation_authority": obstacle_map.authority,
            "navigation_is_habitat_navmesh": False,
            "native_pixel_validation_status": "pending",
            "ue_spear_certification_status": "pending",
            "viewpoint_count": len(viewpoint_records),
            "selection_policy": (
                "complete target-agnostic camera catalog followed by FOV/LOS/"
                "pose-diversity preferences; difficulty clues are not legality gates"
            ),
            "parameters": {
                "hfov_degrees": hfov,
                "camera_heights_m_from_catalog": heights,
                "yaw_degrees_from_catalog": list(camera_catalog.get("yaw_degrees")),
                "side_margin": margin,
                "screening_corridor_count": min(
                    corridor_limit, len(corridor_values)
                ),
                "minimum_pair_separation_m": pair_separation,
                "maximum_pair_records": pair_record_limit,
            },
            "camera_free_space_area_m2": feasible_area,
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
    "validate_camera_catalog_for_viewpoint_selection",
]
