"""Pure-raster source-center navigation for rooms without Habitat navmeshes.

The adapter intentionally exposes only the small PathFinder surface consumed
by :mod:`avengine.m6x.room_feasibility`.  Its authority is an explicit room
polygon and explicit blocking footprints; it must never be described as a
Habitat-native navigation result.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.m6x.geometry import RuntimeObstacleMap


class RasterPathfinderError(ValueError):
    """A raster room or path query is malformed or has no valid route."""


@dataclass
class RasterShortestPath:
    """Minimal query object compatible with Habitat ``ShortestPath`` usage."""

    requested_start: Any = None
    requested_end: Any = None
    points: list[np.ndarray] | None = None
    geodesic_distance: float = math.inf


def _point(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RasterPathfinderError(
            f"{owner} must contain three finite numbers"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise RasterPathfinderError(f"{owner} must contain three finite numbers")
    return result


def _positive(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RasterPathfinderError(f"{owner} must be a finite positive number")
    return float(value)


def _nonnegative(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise RasterPathfinderError(f"{owner} must be a finite nonnegative number")
    return float(value)


def _polygon(value: Any, *, owner: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RasterPathfinderError(f"{owner} must be finite [point,2]") from exc
    if (
        result.ndim != 2
        or result.shape[0] < 3
        or result.shape[1] != 2
        or not np.all(np.isfinite(result))
    ):
        raise RasterPathfinderError(f"{owner} must be finite [point,2]")
    return result


def _points_in_polygon(
    x: np.ndarray, z: np.ndarray, polygon_xz_m: np.ndarray
) -> np.ndarray:
    """Vectorized even/odd fill for cell centers in a possibly concave polygon."""

    inside = np.zeros(np.broadcast_shapes(x.shape, z.shape), dtype=np.bool_)
    x_values, z_values = np.broadcast_arrays(x, z)
    previous = polygon_xz_m[-1]
    for current in polygon_xz_m:
        crosses = (current[1] > z_values) != (previous[1] > z_values)
        denominator = previous[1] - current[1]
        if abs(float(denominator)) > 1.0e-15:
            crossing_x = (previous[0] - current[0]) * (
                z_values - current[1]
            ) / denominator + current[0]
            inside ^= crosses & (x_values < crossing_x)
        previous = current
    return np.ascontiguousarray(inside)


def _erode_binary(
    mask: np.ndarray, clearance_m: float, resolution_m: float
) -> np.ndarray:
    result = np.asarray(mask, dtype=np.bool_).copy()
    if clearance_m <= 0.0:
        return np.ascontiguousarray(result)
    radius = int(math.ceil(clearance_m / resolution_m))
    source = result.copy()
    height, width = source.shape
    for row_delta in range(-radius, radius + 1):
        for col_delta in range(-radius, radius + 1):
            distance = math.hypot(
                max(abs(row_delta) - 0.5, 0.0) * resolution_m,
                max(abs(col_delta) - 0.5, 0.0) * resolution_m,
            )
            if distance > clearance_m + 1.0e-12:
                continue
            shifted = np.zeros_like(source)
            source_row_start = max(0, -row_delta)
            source_row_end = min(height, height - row_delta)
            source_col_start = max(0, -col_delta)
            source_col_end = min(width, width - col_delta)
            shifted[
                source_row_start + row_delta : source_row_end + row_delta,
                source_col_start + col_delta : source_col_end + col_delta,
            ] = source[
                source_row_start:source_row_end,
                source_col_start:source_col_end,
            ]
            result &= shifted
    return np.ascontiguousarray(result)


class RasterPathfinder:
    """A deterministic eight-neighbour A* navigator over one retained raster."""

    is_loaded = True

    def __init__(
        self,
        binary_navmesh: Any,
        *,
        bounds_m: Sequence[Sequence[float]],
        floor_height_m: float,
    ) -> None:
        binary = np.asarray(binary_navmesh, dtype=np.uint8)
        bounds = np.asarray(bounds_m, dtype=np.float64)
        if (
            binary.ndim != 2
            or binary.size == 0
            or not np.any(binary)
            or np.any(~np.isin(binary, (0, 1)))
        ):
            raise RasterPathfinderError("binary_navmesh must be a nonempty 0/1 map")
        if (
            bounds.shape != (2, 3)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[1] <= bounds[0])
        ):
            raise RasterPathfinderError("bounds_m must be increasing finite [2,3]")
        if isinstance(floor_height_m, bool) or not isinstance(floor_height_m, Real):
            raise RasterPathfinderError("floor_height_m must be finite")
        floor = float(floor_height_m)
        if not math.isfinite(floor):
            raise RasterPathfinderError("floor_height_m must be finite")
        self._binary = np.ascontiguousarray(binary)
        self._bounds = np.ascontiguousarray(bounds)
        self._floor = floor
        self._pixel_x = float((bounds[1, 0] - bounds[0, 0]) / binary.shape[1])
        self._pixel_z = float((bounds[1, 2] - bounds[0, 2]) / binary.shape[0])
        self._nav_pixels = np.argwhere(binary != 0)
        self._clearance_m = self._build_clearance_map()

    @property
    def meters_per_pixel(self) -> float:
        if not math.isclose(self._pixel_x, self._pixel_z, rel_tol=0.0, abs_tol=1.0e-9):
            raise RasterPathfinderError("raster pixels are not square")
        return self._pixel_x

    def get_topdown_view(
        self, meters_per_pixel: float, floor_height: float
    ) -> np.ndarray:
        if not math.isclose(
            float(meters_per_pixel), self.meters_per_pixel, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise RasterPathfinderError(
                "requested resolution differs from retained raster"
            )
        if not math.isclose(
            float(floor_height), self._floor, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise RasterPathfinderError("requested floor differs from retained raster")
        return self._binary.copy()

    def get_bounds(self) -> np.ndarray:
        return self._bounds.copy()

    def _pixel_for_point(self, point: np.ndarray) -> tuple[int, int] | None:
        col = int(math.floor((point[0] - self._bounds[0, 0]) / self._pixel_x))
        row = int(math.floor((point[2] - self._bounds[0, 2]) / self._pixel_z))
        if 0 <= row < self._binary.shape[0] and 0 <= col < self._binary.shape[1]:
            return row, col
        return None

    def _point_for_pixel(self, row: int, col: int) -> np.ndarray:
        return np.asarray(
            [
                self._bounds[0, 0] + (col + 0.5) * self._pixel_x,
                self._floor,
                self._bounds[0, 2] + (row + 0.5) * self._pixel_z,
            ],
            dtype=np.float64,
        )

    def is_navigable(self, point: Any, maximum_y_delta: float = 0.25) -> bool:
        value = _point(point, owner="navigation point")
        if abs(float(value[1] - self._floor)) > float(maximum_y_delta) + 1.0e-12:
            return False
        pixel = self._pixel_for_point(value)
        return pixel is not None and bool(self._binary[pixel])

    def snap_point(self, point: Any) -> np.ndarray:
        value = _point(point, owner="snap point")
        pixel = self._pixel_for_point(value)
        if pixel is not None and self._binary[pixel]:
            # A retained raster cell represents a continuous navigable square,
            # not only its center.  Preserve an in-cell x/z position so a
            # resampled diagonal path is not spuriously rejected by a
            # sub-pixel snap-distance gate.
            return np.asarray([value[0], self._floor, value[2]], dtype=np.float64)
        target_col = (value[0] - self._bounds[0, 0]) / self._pixel_x - 0.5
        target_row = (value[2] - self._bounds[0, 2]) / self._pixel_z - 0.5
        deltas = np.stack(
            (
                (self._nav_pixels[:, 0] - target_row) * self._pixel_z,
                (self._nav_pixels[:, 1] - target_col) * self._pixel_x,
            ),
            axis=1,
        )
        nearest = self._nav_pixels[int(np.argmin(np.sum(deltas * deltas, axis=1)))]
        return self._point_for_pixel(int(nearest[0]), int(nearest[1]))

    def _build_clearance_map(self) -> np.ndarray:
        nav = self._binary.astype(np.bool_)
        height, width = nav.shape
        boundary = np.zeros_like(nav)
        for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.zeros_like(nav)
            source_row_start = max(0, -row_delta)
            source_row_end = min(height, height - row_delta)
            source_col_start = max(0, -col_delta)
            source_col_end = min(width, width - col_delta)
            shifted[
                source_row_start + row_delta : source_row_end + row_delta,
                source_col_start + col_delta : source_col_end + col_delta,
            ] = nav[
                source_row_start:source_row_end,
                source_col_start:source_col_end,
            ]
            boundary |= nav & ~shifted
        distances = np.full(nav.shape, np.inf, dtype=np.float64)
        queue: list[tuple[float, int, int]] = []
        initial = 0.5 * min(self._pixel_x, self._pixel_z)
        for row, col in np.argwhere(boundary):
            row_i, col_i = int(row), int(col)
            distances[row_i, col_i] = initial
            heapq.heappush(queue, (initial, row_i, col_i))
        neighbours = tuple(
            (
                row_delta,
                col_delta,
                math.hypot(row_delta * self._pixel_z, col_delta * self._pixel_x),
            )
            for row_delta in (-1, 0, 1)
            for col_delta in (-1, 0, 1)
            if row_delta or col_delta
        )
        while queue:
            distance, row, col = heapq.heappop(queue)
            if distance > distances[row, col] + 1.0e-12:
                continue
            for row_delta, col_delta, cost in neighbours:
                next_row = row + row_delta
                next_col = col + col_delta
                if not (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and nav[next_row, next_col]
                ):
                    continue
                candidate = distance + cost
                if candidate + 1.0e-12 < distances[next_row, next_col]:
                    distances[next_row, next_col] = candidate
                    heapq.heappush(queue, (candidate, next_row, next_col))
        distances[~nav] = 0.0
        return np.ascontiguousarray(distances)

    def distance_to_closest_obstacle(
        self, point: Any, maximum_search_radius: float
    ) -> float:
        value = _point(point, owner="clearance point")
        pixel = self._pixel_for_point(value)
        if pixel is None or not self._binary[pixel]:
            return 0.0
        return float(min(self._clearance_m[pixel], float(maximum_search_radius)))

    @staticmethod
    def _simplify_pixels(pixels: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(pixels) <= 2:
            return pixels
        retained = [pixels[0]]
        previous_delta = (
            pixels[1][0] - pixels[0][0],
            pixels[1][1] - pixels[0][1],
        )
        for index in range(1, len(pixels) - 1):
            next_delta = (
                pixels[index + 1][0] - pixels[index][0],
                pixels[index + 1][1] - pixels[index][1],
            )
            if next_delta != previous_delta:
                retained.append(pixels[index])
            previous_delta = next_delta
        retained.append(pixels[-1])
        return retained

    def find_path(self, query: Any) -> bool:
        start = self.snap_point(query.requested_start)
        end = self.snap_point(query.requested_end)
        start_pixel = self._pixel_for_point(start)
        end_pixel = self._pixel_for_point(end)
        if start_pixel is None or end_pixel is None:
            return False
        if start_pixel == end_pixel:
            query.points = [start.copy(), end.copy()]
            query.geodesic_distance = 0.0
            return True

        height, width = self._binary.shape
        costs = np.full((height, width), np.inf, dtype=np.float64)
        parent_row = np.full((height, width), -1, dtype=np.int32)
        parent_col = np.full((height, width), -1, dtype=np.int32)
        costs[start_pixel] = 0.0

        def heuristic(row: int, col: int) -> float:
            return math.hypot(
                (row - end_pixel[0]) * self._pixel_z,
                (col - end_pixel[1]) * self._pixel_x,
            )

        queue: list[tuple[float, float, int, int]] = [
            (heuristic(*start_pixel), 0.0, *start_pixel)
        ]
        neighbours = tuple(
            (
                row_delta,
                col_delta,
                math.hypot(row_delta * self._pixel_z, col_delta * self._pixel_x),
            )
            for row_delta in (-1, 0, 1)
            for col_delta in (-1, 0, 1)
            if row_delta or col_delta
        )
        found = False
        while queue:
            _priority, cost, row, col = heapq.heappop(queue)
            if cost > costs[row, col] + 1.0e-12:
                continue
            if (row, col) == end_pixel:
                found = True
                break
            for row_delta, col_delta, step_cost in neighbours:
                next_row = row + row_delta
                next_col = col + col_delta
                if not (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and self._binary[next_row, next_col]
                ):
                    continue
                if (
                    row_delta
                    and col_delta
                    and not (
                        self._binary[row, next_col] and self._binary[next_row, col]
                    )
                ):
                    continue
                candidate = cost + step_cost
                if candidate + 1.0e-12 >= costs[next_row, next_col]:
                    continue
                costs[next_row, next_col] = candidate
                parent_row[next_row, next_col] = row
                parent_col[next_row, next_col] = col
                heapq.heappush(
                    queue,
                    (
                        candidate + heuristic(next_row, next_col),
                        candidate,
                        next_row,
                        next_col,
                    ),
                )
        if not found:
            return False
        pixels = [end_pixel]
        row, col = end_pixel
        while (row, col) != start_pixel:
            row, col = int(parent_row[row, col]), int(parent_col[row, col])
            if row < 0 or col < 0:
                return False
            pixels.append((row, col))
        pixels.reverse()
        pixels = self._simplify_pixels(pixels)
        query.points = [self._point_for_pixel(row, col) for row, col in pixels]
        query.geodesic_distance = float(costs[end_pixel])
        return True


def build_polygon_raster_obstacle_map(
    *,
    polygon_xz_m: Sequence[Sequence[float]],
    rigid_obstacles: Sequence[Mapping[str, Any]],
    floor_height_m: float,
    meters_per_pixel: float = 0.03,
    padding_m: float = 0.06,
    minimum_clearance_m: float = 0.0,
    authority: str = "declared_room_polygon_plus_declared_blocking_footprints",
    claim_boundary: str = (
        "source-center-only polygon/footprint navigation; not a Habitat navmesh "
        "or body-volume collision claim"
    ),
) -> tuple[RasterPathfinder, RuntimeObstacleMap]:
    """Rasterize one polygon and its blocking footprints into a shared authority."""

    polygon = _polygon(polygon_xz_m, owner="room polygon")
    resolution = _positive(meters_per_pixel, owner="meters_per_pixel")
    padding = _positive(padding_m, owner="padding_m")
    clearance = _nonnegative(minimum_clearance_m, owner="minimum_clearance_m")
    minimum_x = (
        math.floor((float(np.min(polygon[:, 0])) - padding) / resolution) * resolution
    )
    maximum_x = (
        math.ceil((float(np.max(polygon[:, 0])) + padding) / resolution) * resolution
    )
    minimum_z = (
        math.floor((float(np.min(polygon[:, 1])) - padding) / resolution) * resolution
    )
    maximum_z = (
        math.ceil((float(np.max(polygon[:, 1])) + padding) / resolution) * resolution
    )
    width = int(round((maximum_x - minimum_x) / resolution))
    height = int(round((maximum_z - minimum_z) / resolution))
    if width < 2 or height < 2:
        raise RasterPathfinderError("room polygon raster is degenerate")
    x = minimum_x + (np.arange(width, dtype=np.float64) + 0.5) * resolution
    z = minimum_z + (np.arange(height, dtype=np.float64) + 0.5) * resolution
    x_grid, z_grid = np.meshgrid(x, z)
    navigable = _points_in_polygon(x_grid, z_grid, polygon)
    retained_obstacles = tuple(dict(item) for item in rigid_obstacles)
    for index, obstacle in enumerate(retained_obstacles):
        if obstacle.get("blocks_source_center", True) is False:
            continue
        footprint = _polygon(
            obstacle.get("footprint_xz_m"), owner=f"rigid_obstacles[{index}] footprint"
        )
        row_min = max(
            0,
            int(math.floor((float(np.min(footprint[:, 1])) - minimum_z) / resolution)),
        )
        row_max = min(
            height,
            int(math.ceil((float(np.max(footprint[:, 1])) - minimum_z) / resolution)),
        )
        col_min = max(
            0,
            int(math.floor((float(np.min(footprint[:, 0])) - minimum_x) / resolution)),
        )
        col_max = min(
            width,
            int(math.ceil((float(np.max(footprint[:, 0])) - minimum_x) / resolution)),
        )
        if row_min >= row_max or col_min >= col_max:
            continue
        blocked = _points_in_polygon(
            x_grid[row_min:row_max, col_min:col_max],
            z_grid[row_min:row_max, col_min:col_max],
            footprint,
        )
        navigable[row_min:row_max, col_min:col_max] &= ~blocked
    navigable = _erode_binary(navigable, clearance, resolution)
    floor = float(floor_height_m)
    bounds = (
        (minimum_x, floor - 0.5, minimum_z),
        (maximum_x, floor + 3.5, maximum_z),
    )
    pathfinder = RasterPathfinder(
        np.ascontiguousarray(navigable, dtype=np.uint8),
        bounds_m=bounds,
        floor_height_m=floor,
    )
    obstacle_map = RuntimeObstacleMap(
        binary_navmesh=np.ascontiguousarray(navigable, dtype=np.uint8),
        bounds_m=bounds,
        floor_height_m=floor,
        meters_per_pixel=resolution,
        rigid_obstacles=retained_obstacles,
        authority=authority,
        claim_boundary=claim_boundary,
        rigid_obstacles_baked_into_navmesh=True,
        _pathfinder=pathfinder,
    )
    return pathfinder, obstacle_map


__all__ = [
    "RasterPathfinder",
    "RasterPathfinderError",
    "RasterShortestPath",
    "build_polygon_raster_obstacle_map",
]
