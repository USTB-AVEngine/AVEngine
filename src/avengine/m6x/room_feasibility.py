"""Compile reusable room feasibility and finite source-center trajectory banks.

The complete output of this module is a rasterized feasible region.  Possible
continuous trajectories inside that region are infinite, so the trajectory
bank is deliberately a finite deterministic sample.  Every sampled path is
closed again through the retained room's source-center authority; no
body/capsule volume is inferred.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from avengine.m6x.geometry import (
    RuntimeObstacleMap,
    evaluate_source_center_gate,
    point_to_world_obb_clearance,
)


FEASIBLE_REGION_SCHEMA = "avengine_room_feasible_region_v1"
TRAJECTORY_BANK_SCHEMA = "avengine_room_trajectory_bank_v2"
RIR_JOB_PLAN_SCHEMA = "avengine_room_rir_job_plan_v2"
TRAJECTORY_COVERAGE_SCHEMA = "avengine_room_trajectory_coverage_v1"
TRAJECTORY_DIVERSITY_SCHEMA = "avengine_room_trajectory_diversity_v1"
SOURCE_SLOTS = ("source1", "source2")
MOTION_CASES = (
    "static_static",
    "source1_moving_source2_static",
    "source1_static_source2_moving",
    "both_moving",
)


class RoomFeasibilityError(ValueError):
    """A room region or sampled trajectory cannot be compiled safely."""


def _finite_number(value: Any, *, owner: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RoomFeasibilityError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise RoomFeasibilityError(f"{owner} must be finite and >= {minimum}")
    return result


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RoomFeasibilityError(f"{owner} must be a positive integer")
    return value


def _label_components(
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Label four-connected raster components without a SciPy dependency."""

    value = np.asarray(mask, dtype=np.bool_)
    labels = np.zeros(value.shape, dtype=np.int32)
    components: list[dict[str, Any]] = []
    height, width = value.shape
    component_id = 0
    for row, col in np.argwhere(value):
        row_i, col_i = int(row), int(col)
        if labels[row_i, col_i] != 0:
            continue
        component_id += 1
        queue: deque[tuple[int, int]] = deque(((row_i, col_i),))
        labels[row_i, col_i] = component_id
        count = 0
        row_min = row_max = row_i
        col_min = col_max = col_i
        row_total = 0
        col_total = 0
        while queue:
            current_row, current_col = queue.popleft()
            count += 1
            row_total += current_row
            col_total += current_col
            row_min = min(row_min, current_row)
            row_max = max(row_max, current_row)
            col_min = min(col_min, current_col)
            col_max = max(col_max, current_col)
            for next_row, next_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and value[next_row, next_col]
                    and labels[next_row, next_col] == 0
                ):
                    labels[next_row, next_col] = component_id
                    queue.append((next_row, next_col))
        components.append(
            {
                "component_id": component_id,
                "pixel_count": count,
                "pixel_bounds_rc_inclusive": [
                    [row_min, col_min],
                    [row_max, col_max],
                ],
                "pixel_centroid_rc": [row_total / count, col_total / count],
            }
        )
    return np.ascontiguousarray(labels), tuple(components)


def _erode_mask(
    mask: np.ndarray,
    *,
    minimum_clearance_m: float,
    pixel_size_x_m: float,
    pixel_size_z_m: float,
) -> np.ndarray:
    """Conservatively erode a binary map by a metric disk."""

    result = np.asarray(mask, dtype=np.bool_).copy()
    if minimum_clearance_m <= 0.0:
        return np.ascontiguousarray(result)
    row_radius = int(math.ceil(minimum_clearance_m / pixel_size_z_m))
    col_radius = int(math.ceil(minimum_clearance_m / pixel_size_x_m))
    source = result.copy()
    height, width = source.shape
    for row_delta in range(-row_radius, row_radius + 1):
        for col_delta in range(-col_radius, col_radius + 1):
            # Distance from the source pixel center to the nearest edge of a
            # shifted raster cell.  This includes the immediately adjacent
            # cells for a one-pixel metric margin even when Habitat's bounds
            # divided by shape are a few micrometers larger than the requested
            # meters-per-pixel value.
            metric_distance = math.hypot(
                max(abs(row_delta) - 0.5, 0.0) * pixel_size_z_m,
                max(abs(col_delta) - 0.5, 0.0) * pixel_size_x_m,
            )
            if metric_distance > minimum_clearance_m + 1.0e-12:
                continue
            shifted = np.zeros_like(source)
            source_row_start = max(0, -row_delta)
            source_row_end = min(height, height - row_delta)
            source_col_start = max(0, -col_delta)
            source_col_end = min(width, width - col_delta)
            target_row_start = source_row_start + row_delta
            target_row_end = source_row_end + row_delta
            target_col_start = source_col_start + col_delta
            target_col_end = source_col_end + col_delta
            shifted[
                target_row_start:target_row_end,
                target_col_start:target_col_end,
            ] = source[
                source_row_start:source_row_end,
                source_col_start:source_col_end,
            ]
            result &= shifted
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class FeasibleRegionIndex:
    """One raster source-center feasibility index at an explicit height."""

    obstacle_map: RuntimeObstacleMap
    source_center_height_m: float
    minimum_navmesh_clearance_m: float
    minimum_rigid_clearance_m: float
    feasible_mask: np.ndarray
    component_labels: np.ndarray
    components: tuple[Mapping[str, Any], ...]
    sample_pixels_rc: np.ndarray

    @property
    def pixel_size_x_m(self) -> float:
        bounds = np.asarray(self.obstacle_map.bounds_m, dtype=np.float64)
        return float((bounds[1, 0] - bounds[0, 0]) / self.feasible_mask.shape[1])

    @property
    def pixel_size_z_m(self) -> float:
        bounds = np.asarray(self.obstacle_map.bounds_m, dtype=np.float64)
        return float((bounds[1, 2] - bounds[0, 2]) / self.feasible_mask.shape[0])

    def pixel_to_world(
        self, pixel_rc: Sequence[int], *, height_m: float | None = None
    ) -> np.ndarray:
        pixel = np.asarray(pixel_rc, dtype=np.int64)
        if pixel.shape != (2,):
            raise RoomFeasibilityError("pixel_rc must contain row and column")
        row, col = (int(pixel[0]), int(pixel[1]))
        map_height, map_width = self.feasible_mask.shape
        if not 0 <= row < map_height or not 0 <= col < map_width:
            raise RoomFeasibilityError("pixel_rc lies outside the feasibility map")
        bounds = np.asarray(self.obstacle_map.bounds_m, dtype=np.float64)
        return np.asarray(
            [
                bounds[0, 0] + (col + 0.5) * self.pixel_size_x_m,
                self.source_center_height_m if height_m is None else float(height_m),
                bounds[0, 2] + (row + 0.5) * self.pixel_size_z_m,
            ],
            dtype=np.float64,
        )

    def sample_points_m(self) -> np.ndarray:
        return np.ascontiguousarray(
            np.stack([self.pixel_to_world(pixel) for pixel in self.sample_pixels_rc])
        )

    def summary(self) -> dict[str, Any]:
        pixel_area = self.pixel_size_x_m * self.pixel_size_z_m
        components = []
        for component in self.components:
            record = dict(component)
            record["approximate_area_m2"] = record["pixel_count"] * pixel_area
            components.append(record)
        return {
            "schema": FEASIBLE_REGION_SCHEMA,
            "semantics": (
                "complete raster source-center feasible region; not the infinite "
                "set of continuous trajectories"
            ),
            "claim_boundary": "source center only; no body-volume collision claim",
            "source_center_height_m": self.source_center_height_m,
            "minimum_navmesh_clearance_m": self.minimum_navmesh_clearance_m,
            "minimum_rigid_clearance_m": self.minimum_rigid_clearance_m,
            "mask_shape_hw": list(self.feasible_mask.shape),
            "bounds_m": [list(item) for item in self.obstacle_map.bounds_m],
            "pixel_size_x_m": self.pixel_size_x_m,
            "pixel_size_z_m": self.pixel_size_z_m,
            "navmesh_pixel_count": int(
                np.count_nonzero(self.obstacle_map.binary_navmesh)
            ),
            "feasible_pixel_count": int(np.count_nonzero(self.feasible_mask)),
            "approximate_feasible_area_m2": float(
                np.count_nonzero(self.feasible_mask) * pixel_area
            ),
            "component_count": len(self.components),
            "sample_point_count": int(len(self.sample_pixels_rc)),
            "components": components,
        }


class RoomFeasibilityCompiler:
    """Build height-specific source-center masks from one runtime snapshot."""

    def __init__(self, obstacle_map: RuntimeObstacleMap):
        if not isinstance(obstacle_map, RuntimeObstacleMap):
            raise RoomFeasibilityError("obstacle_map must be a RuntimeObstacleMap")
        self.obstacle_map = obstacle_map

    def compile(
        self,
        *,
        source_center_height_m: float,
        minimum_navmesh_clearance_m: float = 0.02,
        minimum_rigid_clearance_m: float = 0.0,
        sample_spacing_m: float = 0.25,
    ) -> FeasibleRegionIndex:
        height_m = _finite_number(source_center_height_m, owner="source center height")
        nav_clearance = _finite_number(
            minimum_navmesh_clearance_m,
            owner="minimum navmesh clearance",
            minimum=0.0,
        )
        rigid_clearance = _finite_number(
            minimum_rigid_clearance_m,
            owner="minimum rigid clearance",
            minimum=0.0,
        )
        spacing = _finite_number(
            sample_spacing_m, owner="sample spacing", minimum=1.0e-9
        )
        navmesh = np.asarray(self.obstacle_map.binary_navmesh, dtype=np.bool_)
        bounds = np.asarray(self.obstacle_map.bounds_m, dtype=np.float64)
        pixel_x = float((bounds[1, 0] - bounds[0, 0]) / navmesh.shape[1])
        pixel_z = float((bounds[1, 2] - bounds[0, 2]) / navmesh.shape[0])
        feasible = _erode_mask(
            navmesh,
            minimum_clearance_m=nav_clearance,
            pixel_size_x_m=pixel_x,
            pixel_size_z_m=pixel_z,
        )

        rows, cols = np.nonzero(feasible)
        if len(rows) and not self.obstacle_map.rigid_obstacles_baked_into_navmesh:
            x = bounds[0, 0] + (cols.astype(np.float64) + 0.5) * pixel_x
            z = bounds[0, 2] + (rows.astype(np.float64) + 0.5) * pixel_z
            points = np.stack((x, np.full_like(x, height_m), z), axis=1)
            blocked = np.zeros(len(points), dtype=np.bool_)
            for obstacle in self.obstacle_map.rigid_obstacles:
                if obstacle.get("blocks_source_center", True) is False:
                    continue
                try:
                    obb = obstacle["world_obb"]
                    center = np.asarray(obb["center_m"], dtype=np.float64)
                    axes = np.asarray(obb["axes_xyz"], dtype=np.float64)
                    half = np.asarray(obb["half_extents_m"], dtype=np.float64)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise RoomFeasibilityError(
                        "runtime obstacle contains an invalid world OBB"
                    ) from exc
                projected = (points - center) @ axes.T
                excess = np.maximum(np.abs(projected) - half, 0.0)
                distances = np.linalg.norm(excess, axis=1)
                inside = np.all(np.abs(projected) <= half + 1.0e-9, axis=1)
                blocked |= inside | (distances < rigid_clearance)
            feasible[rows[blocked], cols[blocked]] = False

        labels, components = _label_components(feasible)
        if not components:
            raise RoomFeasibilityError("room has no feasible source-center region")
        row_step = max(1, int(round(spacing / pixel_z)))
        col_step = max(1, int(round(spacing / pixel_x)))
        sample_mask = np.zeros_like(feasible)
        sample_mask[::row_step, ::col_step] = True
        sample_pixels = [
            tuple(int(value) for value in item)
            for item in np.argwhere(feasible & sample_mask)
        ]
        sampled_components = {int(labels[item]) for item in sample_pixels}
        for component in components:
            component_id = int(component["component_id"])
            if component_id in sampled_components:
                continue
            members = np.argwhere(labels == component_id)
            centroid = np.asarray(component["pixel_centroid_rc"], dtype=np.float64)
            nearest = members[
                int(np.argmin(np.linalg.norm(members - centroid, axis=1)))
            ]
            sample_pixels.append((int(nearest[0]), int(nearest[1])))
        sample_pixels.sort()
        return FeasibleRegionIndex(
            obstacle_map=self.obstacle_map,
            source_center_height_m=height_m,
            minimum_navmesh_clearance_m=nav_clearance,
            minimum_rigid_clearance_m=rigid_clearance,
            feasible_mask=np.ascontiguousarray(feasible),
            component_labels=labels,
            components=components,
            sample_pixels_rc=np.ascontiguousarray(
                np.asarray(sample_pixels, dtype=np.int32)
            ),
        )


@dataclass(frozen=True)
class TrajectoryEpisode:
    episode_id: str
    motion_case: str
    source_root_paths_m: Mapping[str, np.ndarray]
    source_center_paths_m: Mapping[str, np.ndarray]
    statistics: Mapping[str, Any]


@dataclass(frozen=True)
class TrajectoryBank:
    episodes: tuple[TrajectoryEpisode, ...]
    frame_count: int
    frame_rate_hz: int
    seed: int

    def record(self, *, include_paths: bool = True) -> dict[str, Any]:
        records = []
        for episode in self.episodes:
            value: dict[str, Any] = {
                "episode_id": episode.episode_id,
                "motion_case": episode.motion_case,
                "statistics": dict(episode.statistics),
            }
            if include_paths:
                value["source_root_paths_m"] = {
                    key: np.asarray(path, dtype=np.float64).tolist()
                    for key, path in sorted(episode.source_root_paths_m.items())
                }
                value["source_center_paths_m"] = {
                    key: np.asarray(path, dtype=np.float64).tolist()
                    for key, path in sorted(episode.source_center_paths_m.items())
                }
            records.append(value)
        counts = {
            motion_case: sum(
                episode.motion_case == motion_case for episode in self.episodes
            )
            for motion_case in MOTION_CASES
        }
        return {
            "schema": TRAJECTORY_BANK_SCHEMA,
            "semantics": (
                "source-slot trajectories are independent of the dry audio and "
                "optional visual entity bound to each slot"
            ),
            "claim_boundary": "source center only; no body-volume collision claim",
            "source_slots": list(SOURCE_SLOTS),
            "frame_count": self.frame_count,
            "frame_rate_hz": self.frame_rate_hz,
            "seconds_per_episode": self.frame_count / self.frame_rate_hz,
            "seed": self.seed,
            "episode_count": len(self.episodes),
            "motion_case_counts": counts,
            "episodes": records,
        }


@dataclass(frozen=True)
class TrajectoryCoverage:
    """Geodesic raster distance from all feasible cells to the route bank."""

    distance_to_trajectory_m: np.ndarray
    trajectory_seed_mask: np.ndarray
    record: Mapping[str, Any]


def _path_seed_pixels(
    region: FeasibleRegionIndex,
    feasible_mask: np.ndarray,
    points_m: np.ndarray,
) -> tuple[set[tuple[int, int]], int]:
    """Densely rasterize one source path and project tiny raster mismatches."""

    bounds = np.asarray(region.obstacle_map.bounds_m, dtype=np.float64)
    pixel_x = region.pixel_size_x_m
    pixel_z = region.pixel_size_z_m
    map_height, map_width = feasible_mask.shape
    seeds: set[tuple[int, int]] = set()
    projected_count = 0
    points = np.asarray(points_m, dtype=np.float64)
    for start, end in zip(points[:-1], points[1:], strict=True):
        horizontal_distance = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        sample_count = max(
            2,
            int(math.ceil(horizontal_distance / (0.5 * min(pixel_x, pixel_z)))) + 1,
        )
        segment = np.linspace(start, end, sample_count)
        for point in segment:
            col = int(round((point[0] - bounds[0, 0]) / pixel_x - 0.5))
            row = int(round((point[2] - bounds[0, 2]) / pixel_z - 0.5))
            if not (0 <= row < map_height and 0 <= col < map_width):
                continue
            if feasible_mask[row, col]:
                seeds.add((row, col))
                continue
            nearest: tuple[float, int, int] | None = None
            for row_delta in range(-2, 3):
                for col_delta in range(-2, 3):
                    candidate_row = row + row_delta
                    candidate_col = col + col_delta
                    if (
                        0 <= candidate_row < map_height
                        and 0 <= candidate_col < map_width
                        and feasible_mask[candidate_row, candidate_col]
                    ):
                        distance = math.hypot(row_delta * pixel_z, col_delta * pixel_x)
                        candidate = (distance, candidate_row, candidate_col)
                        if nearest is None or candidate < nearest:
                            nearest = candidate
            if nearest is not None:
                seeds.add((nearest[1], nearest[2]))
                projected_count += 1
    if len(points) == 1:
        return _path_seed_pixels(
            region,
            feasible_mask,
            np.repeat(points, 2, axis=0),
        )
    return seeds, projected_count


def evaluate_trajectory_coverage(
    region_by_source: Mapping[str, FeasibleRegionIndex],
    bank: TrajectoryBank,
    *,
    thresholds_m: Sequence[float] = (0.25, 0.50, 1.00),
    minimum_half_meter_fraction: float = 0.90,
    maximum_gap_m: float = 1.50,
) -> TrajectoryCoverage:
    """Measure route coverage over the complete feasible-region intersection.

    Distances are eight-neighbour geodesic distances within the feasible
    raster.  A wall therefore cannot make a route on its other side appear to
    cover an otherwise nearby cell.
    """

    if set(region_by_source) != set(SOURCE_SLOTS):
        raise RoomFeasibilityError(
            "region_by_source must contain exactly source1 and source2"
        )
    source1_region = region_by_source["source1"]
    source2_region = region_by_source["source2"]
    if source1_region.obstacle_map is not source2_region.obstacle_map:
        raise RoomFeasibilityError("coverage regions must share one obstacle map")
    feasible = np.asarray(
        source1_region.feasible_mask & source2_region.feasible_mask, dtype=np.bool_
    )
    if not np.any(feasible):
        raise RoomFeasibilityError("coverage region intersection is empty")
    parsed_thresholds = tuple(
        sorted(
            {
                _finite_number(value, owner="coverage threshold", minimum=0.0)
                for value in thresholds_m
            }
        )
    )
    if not parsed_thresholds:
        raise RoomFeasibilityError("at least one coverage threshold is required")
    half_fraction_gate = _finite_number(
        minimum_half_meter_fraction,
        owner="minimum half-meter coverage fraction",
        minimum=0.0,
    )
    if half_fraction_gate > 1.0:
        raise RoomFeasibilityError(
            "minimum half-meter coverage fraction cannot exceed one"
        )
    maximum_gap_gate = _finite_number(
        maximum_gap_m, owner="maximum coverage gap", minimum=0.0
    )

    seed_pixels: set[tuple[int, int]] = set()
    projected_seed_samples = 0
    source_path_count = 0
    for episode in bank.episodes:
        for path in episode.source_center_paths_m.values():
            path_seeds, projected = _path_seed_pixels(
                source1_region,
                feasible,
                np.asarray(path, dtype=np.float64),
            )
            seed_pixels.update(path_seeds)
            projected_seed_samples += projected
            source_path_count += 1
    if not seed_pixels:
        raise RoomFeasibilityError("trajectory bank produced no feasible raster seeds")

    distances = np.full(feasible.shape, np.inf, dtype=np.float64)
    seed_mask = np.zeros(feasible.shape, dtype=np.bool_)
    queue: list[tuple[float, int, int]] = []
    for row, col in sorted(seed_pixels):
        distances[row, col] = 0.0
        seed_mask[row, col] = True
        heapq.heappush(queue, (0.0, row, col))
    pixel_x = source1_region.pixel_size_x_m
    pixel_z = source1_region.pixel_size_z_m
    neighbours = tuple(
        (
            row_delta,
            col_delta,
            math.hypot(row_delta * pixel_z, col_delta * pixel_x),
        )
        for row_delta in (-1, 0, 1)
        for col_delta in (-1, 0, 1)
        if row_delta or col_delta
    )
    map_height, map_width = feasible.shape
    while queue:
        distance, row, col = heapq.heappop(queue)
        if distance > distances[row, col] + 1.0e-12:
            continue
        for row_delta, col_delta, cost in neighbours:
            next_row = row + row_delta
            next_col = col + col_delta
            if not (
                0 <= next_row < map_height
                and 0 <= next_col < map_width
                and feasible[next_row, next_col]
            ):
                continue
            candidate = distance + cost
            if candidate + 1.0e-12 < distances[next_row, next_col]:
                distances[next_row, next_col] = candidate
                heapq.heappush(queue, (candidate, next_row, next_col))

    finite = distances[feasible]
    if not np.all(np.isfinite(finite)):
        raise RoomFeasibilityError(
            "at least one feasible component has no sampled trajectory"
        )
    threshold_fractions = {
        f"within_{threshold:.2f}m": float(np.mean(finite <= threshold + 1.0e-12))
        for threshold in parsed_thresholds
    }
    half_meter_fraction = float(np.mean(finite <= 0.50 + 1.0e-12))
    maximum_gap = float(np.max(finite))
    status = (
        "pass"
        if half_meter_fraction >= half_fraction_gate and maximum_gap <= maximum_gap_gate
        else "fail"
    )
    component_records = []
    combined_labels, combined_components = _label_components(feasible)
    for component in combined_components:
        component_id = int(component["component_id"])
        values = distances[combined_labels == component_id]
        component_records.append(
            {
                "component_id": component_id,
                "pixel_count": int(len(values)),
                "trajectory_seed_pixel_count": int(
                    np.count_nonzero(seed_mask & (combined_labels == component_id))
                ),
                "mean_gap_m": float(np.mean(values)),
                "p95_gap_m": float(np.percentile(values, 95)),
                "maximum_gap_m": float(np.max(values)),
            }
        )
    record = {
        "schema": TRAJECTORY_COVERAGE_SCHEMA,
        "status": status,
        "distance_semantics": (
            "eight-neighbour feasible-raster geodesic distance to nearest "
            "densely rasterized source-center trajectory"
        ),
        "claim_boundary": "source centers only; no body-volume coverage claim",
        "episode_count": len(bank.episodes),
        "source_path_count": source_path_count,
        "feasible_pixel_count": int(len(finite)),
        "trajectory_seed_pixel_count": int(np.count_nonzero(seed_mask)),
        "projected_seed_sample_count": projected_seed_samples,
        "coverage_fraction_by_threshold": threshold_fractions,
        "mean_gap_m": float(np.mean(finite)),
        "p50_gap_m": float(np.percentile(finite, 50)),
        "p90_gap_m": float(np.percentile(finite, 90)),
        "p95_gap_m": float(np.percentile(finite, 95)),
        "maximum_gap_m": maximum_gap,
        "gate": {
            "minimum_fraction_within_0.50m": half_fraction_gate,
            "maximum_gap_m": maximum_gap_gate,
            "observed_fraction_within_0.50m": half_meter_fraction,
            "observed_maximum_gap_m": maximum_gap,
        },
        "component_count": len(component_records),
        "components": component_records,
    }
    return TrajectoryCoverage(
        distance_to_trajectory_m=np.ascontiguousarray(distances),
        trajectory_seed_mask=np.ascontiguousarray(seed_mask),
        record=record,
    )


class TrajectoryBankBuilder:
    """Sample two source slots without binding paths to an entity or dry sound."""

    def __init__(
        self,
        *,
        pathfinder: Any,
        obstacle_map: RuntimeObstacleMap,
        region_by_source: Mapping[str, FeasibleRegionIndex],
        shortest_path_factory: Callable[[], Any],
        source_path_materializer: Callable[
            [Mapping[str, np.ndarray]], Mapping[str, np.ndarray]
        ]
        | None = None,
    ) -> None:
        if set(region_by_source) != set(SOURCE_SLOTS):
            raise RoomFeasibilityError(
                "region_by_source must contain exactly source1 and source2"
            )
        if any(
            index.obstacle_map is not obstacle_map
            for index in region_by_source.values()
        ):
            raise RoomFeasibilityError("all region indexes must share one obstacle map")
        self.pathfinder = pathfinder
        self.obstacle_map = obstacle_map
        self.region_by_source = dict(region_by_source)
        self.shortest_path_factory = shortest_path_factory
        self.source_path_materializer = source_path_materializer
        # Offset the two slots so a static/static candidate does not place
        # both source centers in the same tiny component by construction.
        self._static_component_cursor = {"source1": 0, "source2": 1}

    @staticmethod
    def _resample_polyline(points_m: np.ndarray, frame_count: int) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            raise RoomFeasibilityError("shortest path returned an invalid polyline")
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        if cumulative[-1] <= 1.0e-9:
            raise RoomFeasibilityError("shortest path has zero length")
        targets = np.linspace(0.0, cumulative[-1], frame_count)
        result = np.empty((frame_count, 3), dtype=np.float64)
        for axis in range(3):
            result[:, axis] = np.interp(targets, cumulative, points[:, axis])
        return np.ascontiguousarray(result)

    def _static_path(
        self, source_slot: str, rng: np.random.Generator, frame_count: int
    ) -> np.ndarray:
        index = self.region_by_source[source_slot]
        pixels = index.sample_pixels_rc
        labels = index.component_labels[pixels[:, 0], pixels[:, 1]]
        component_ids = tuple(sorted(set(int(value) for value in labels)))
        cursor = self._static_component_cursor[source_slot]
        self._static_component_cursor[source_slot] = cursor + 1
        mandatory_component_attempts = 2 * len(component_ids)
        if cursor < mandatory_component_attempts:
            # Seed every disconnected feasible component once.  Remaining
            # static samples are drawn from the complete sample array, which
            # makes their frequency proportional to feasible area instead of
            # over-weighting tiny pockets.
            component_id = component_ids[cursor % len(component_ids)]
            members = np.flatnonzero(labels == component_id)
            pixel = pixels[int(members[int(rng.integers(len(members)))])]
        else:
            pixel = pixels[int(rng.integers(len(pixels)))]
        point = index.pixel_to_world(pixel, height_m=self.obstacle_map.floor_height_m)
        snapped = np.asarray(self.pathfinder.snap_point(point), dtype=np.float64)
        return np.ascontiguousarray(np.repeat(snapped[None, :], frame_count, axis=0))

    def _moving_path(
        self,
        source_slot: str,
        rng: np.random.Generator,
        *,
        frame_count: int,
        minimum_distance_m: float,
        maximum_distance_m: float,
        path_attempts: int,
    ) -> tuple[np.ndarray, float]:
        index = self.region_by_source[source_slot]
        pixels = index.sample_pixels_rc
        labels = index.component_labels[pixels[:, 0], pixels[:, 1]]
        eligible_components = tuple(
            component_id
            for component_id in sorted(set(int(value) for value in labels))
            if int(np.count_nonzero(labels == component_id)) >= 2
        )
        if not eligible_components:
            raise RoomFeasibilityError("no component contains two trajectory samples")
        for _attempt in range(path_attempts):
            component_id = eligible_components[
                int(rng.integers(len(eligible_components)))
            ]
            member_indices = np.flatnonzero(labels == component_id)
            selected = rng.choice(member_indices, size=2, replace=False)
            start = index.pixel_to_world(
                pixels[int(selected[0])], height_m=self.obstacle_map.floor_height_m
            )
            end = index.pixel_to_world(
                pixels[int(selected[1])], height_m=self.obstacle_map.floor_height_m
            )
            horizontal_distance = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
            # Geodesic distance cannot be shorter than the horizontal chord.
            # Reject an impossible upper-bound candidate before asking a native
            # or raster pathfinder to perform the comparatively expensive search.
            # A conservative lower prefilter still permits detours around room
            # obstacles to lift a short chord into the requested route band.
            if (
                horizontal_distance > maximum_distance_m
                or horizontal_distance < minimum_distance_m * 0.5
            ):
                continue
            query = self.shortest_path_factory()
            query.requested_start = np.asarray(
                self.pathfinder.snap_point(start), dtype=np.float64
            )
            query.requested_end = np.asarray(
                self.pathfinder.snap_point(end), dtype=np.float64
            )
            if not bool(self.pathfinder.find_path(query)):
                continue
            distance = float(query.geodesic_distance)
            if (
                not math.isfinite(distance)
                or distance < minimum_distance_m
                or distance > maximum_distance_m
            ):
                continue
            points = np.asarray(query.points, dtype=np.float64)
            return self._resample_polyline(points, frame_count), distance
        raise RoomFeasibilityError(
            f"could not sample a {minimum_distance_m:g}-{maximum_distance_m:g} m "
            f"path for {source_slot} after {path_attempts} attempts"
        )

    def _fast_candidate_gate(
        self,
        sources: Mapping[str, np.ndarray],
        *,
        maximum_floor_snap_xz_m: float,
        minimum_navmesh_clearance_m: float,
        minimum_rigid_clearance_m: float,
    ) -> tuple[bool, dict[str, float]]:
        """Reject candidate points without rebuilding the room snapshot.

        The complete accepted bank is still sent through
        :func:`evaluate_source_center_gate` once.  This candidate filter uses
        the same point predicates but deliberately omits that function's
        expensive whole-navmesh snapshot comparison on every trial.
        """

        minimum_clearances: dict[str, float] = {}
        for source_id, path in sources.items():
            observed_clearances: list[float] = []
            for point in path:
                floor_query = np.asarray(
                    [point[0], self.obstacle_map.floor_height_m, point[2]],
                    dtype=np.float64,
                )
                if not bool(self.pathfinder.is_navigable(floor_query, 0.25)):
                    return False, {}
                snapped = np.asarray(
                    self.pathfinder.snap_point(floor_query), dtype=np.float64
                )
                if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
                    return False, {}
                snap_xz = float(np.linalg.norm(snapped[[0, 2]] - floor_query[[0, 2]]))
                if snap_xz > maximum_floor_snap_xz_m:
                    return False, {}
                nav_clearance = float(
                    self.pathfinder.distance_to_closest_obstacle(snapped, 10.0)
                )
                if (
                    not math.isfinite(nav_clearance)
                    or nav_clearance < minimum_navmesh_clearance_m
                ):
                    return False, {}
                if not self.obstacle_map.rigid_obstacles_baked_into_navmesh:
                    for obstacle in self.obstacle_map.rigid_obstacles:
                        if obstacle.get("blocks_source_center", True) is False:
                            continue
                        rigid_clearance, inside = point_to_world_obb_clearance(
                            point, obstacle
                        )
                        if inside or rigid_clearance < minimum_rigid_clearance_m:
                            return False, {}
                observed_clearances.append(nav_clearance)
            minimum_clearances[source_id] = min(observed_clearances)
        return True, minimum_clearances

    def build(
        self,
        *,
        episodes_per_motion_case: int = 50,
        frame_count: int = 75,
        frame_rate_hz: int = 15,
        seed: int = 20_260_721,
        minimum_route_distance_m: float = 3.5,
        maximum_route_distance_m: float = 5.5,
        minimum_pair_separation_m: float = 0.30,
        maximum_floor_snap_xz_m: float = 0.03,
        episode_attempts: int = 250,
        path_attempts: int = 250,
    ) -> TrajectoryBank:
        count = _positive_int(
            episodes_per_motion_case, owner="episodes per motion case"
        )
        frames = _positive_int(frame_count, owner="frame count")
        fps = _positive_int(frame_rate_hz, owner="frame rate")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RoomFeasibilityError("seed must be an integer")
        minimum_distance = _finite_number(
            minimum_route_distance_m, owner="minimum route distance", minimum=0.01
        )
        maximum_distance = _finite_number(
            maximum_route_distance_m,
            owner="maximum route distance",
            minimum=minimum_distance,
        )
        separation = _finite_number(
            minimum_pair_separation_m, owner="minimum pair separation", minimum=0.0
        )
        max_snap = _finite_number(
            maximum_floor_snap_xz_m, owner="maximum floor snap", minimum=0.0
        )
        attempts = _positive_int(episode_attempts, owner="episode attempts")
        per_path_attempts = _positive_int(path_attempts, owner="path attempts")
        rng = np.random.default_rng(seed)
        episodes: list[TrajectoryEpisode] = []
        signatures: set[tuple[Any, ...]] = set()
        nav_clearance_threshold = min(
            index.minimum_navmesh_clearance_m
            for index in self.region_by_source.values()
        )
        rigid_clearance_threshold = min(
            index.minimum_rigid_clearance_m for index in self.region_by_source.values()
        )
        accepted_moving_geometries: dict[str, set[bytes]] = {
            source_slot: set() for source_slot in SOURCE_SLOTS
        }

        def moving_geometry_signature(path: np.ndarray) -> bytes:
            points = np.round(np.asarray(path)[:, (0, 2)], decimals=6)
            forward = points.tobytes()
            reverse = np.ascontiguousarray(points[::-1]).tobytes()
            return min(forward, reverse)

        for motion_case in MOTION_CASES:
            source1_moving = motion_case in {
                "source1_moving_source2_static",
                "both_moving",
            }
            source2_moving = motion_case in {
                "source1_static_source2_moving",
                "both_moving",
            }
            accepted = 0
            for _episode_attempt in range(attempts * count):
                if accepted >= count:
                    break
                try:
                    source1_path, source1_distance = (
                        self._moving_path(
                            "source1",
                            rng,
                            frame_count=frames,
                            minimum_distance_m=minimum_distance,
                            maximum_distance_m=maximum_distance,
                            path_attempts=per_path_attempts,
                        )
                        if source1_moving
                        else (self._static_path("source1", rng, frames), 0.0)
                    )
                    source2_path, source2_distance = (
                        self._moving_path(
                            "source2",
                            rng,
                            frame_count=frames,
                            minimum_distance_m=minimum_distance,
                            maximum_distance_m=maximum_distance,
                            path_attempts=per_path_attempts,
                        )
                        if source2_moving
                        else (self._static_path("source2", rng, frames), 0.0)
                    )
                except RoomFeasibilityError:
                    continue
                moving_candidates = {
                    "source1": (source1_moving, source1_path),
                    "source2": (source2_moving, source2_path),
                }
                candidate_moving_signatures = {
                    source_slot: moving_geometry_signature(path)
                    for source_slot, (is_moving, path) in moving_candidates.items()
                    if is_moving
                }
                if any(
                    signature in accepted_moving_geometries[source_slot]
                    for source_slot, signature in candidate_moving_signatures.items()
                ):
                    continue
                roots = {
                    "source1": source1_path,
                    "source2": source2_path,
                }
                materialized = (
                    self.source_path_materializer(roots)
                    if self.source_path_materializer is not None
                    else roots
                )
                sources = {
                    key: np.ascontiguousarray(np.asarray(path, dtype=np.float64))
                    for key, path in sorted(materialized.items())
                }
                if (
                    set(sources) != set(SOURCE_SLOTS)
                    or any(path.shape != (frames, 3) for path in sources.values())
                    or any(not np.all(np.isfinite(path)) for path in sources.values())
                ):
                    raise RoomFeasibilityError(
                        "source_path_materializer must return finite source1/source2 "
                        "[frame,3] paths"
                    )
                pair = tuple(sources[source_slot] for source_slot in SOURCE_SLOTS)
                pair_distances = np.linalg.norm(
                    pair[0][:, (0, 2)] - pair[1][:, (0, 2)], axis=1
                )
                if float(np.min(pair_distances)) < separation:
                    continue
                signature = (
                    motion_case,
                    *np.round(source1_path[[0, -1]][:, (0, 2)].reshape(-1), 4),
                    *np.round(source2_path[[0, -1]][:, (0, 2)].reshape(-1), 4),
                )
                if signature in signatures:
                    continue
                candidate_passed, source_minimum_clearances = self._fast_candidate_gate(
                    sources,
                    maximum_floor_snap_xz_m=max_snap,
                    minimum_navmesh_clearance_m=nav_clearance_threshold,
                    minimum_rigid_clearance_m=rigid_clearance_threshold,
                )
                if not candidate_passed:
                    continue
                episode_id = f"{motion_case}_{accepted:03d}"
                duration = frames / fps
                statistics = {
                    "source1": {
                        "motion": "moving" if source1_moving else "static",
                        "geodesic_distance_m": source1_distance,
                        "mean_speed_m_s": source1_distance / duration,
                    },
                    "source2": {
                        "motion": "moving" if source2_moving else "static",
                        "geodesic_distance_m": source2_distance,
                        "mean_speed_m_s": source2_distance / duration,
                    },
                    "minimum_source_pair_xz_separation_m": float(
                        np.min(pair_distances)
                    ),
                    "source_center_gate_status": "pass",
                    "minimum_source_navmesh_clearance_m": source_minimum_clearances,
                }
                episodes.append(
                    TrajectoryEpisode(
                        episode_id=episode_id,
                        motion_case=motion_case,
                        source_root_paths_m={
                            key: np.ascontiguousarray(path)
                            for key, path in roots.items()
                        },
                        source_center_paths_m=sources,
                        statistics=statistics,
                    )
                )
                for (
                    source_slot,
                    moving_signature,
                ) in candidate_moving_signatures.items():
                    accepted_moving_geometries[source_slot].add(moving_signature)
                signatures.add(signature)
                accepted += 1
            if accepted != count:
                raise RoomFeasibilityError(
                    f"generated only {accepted}/{count} {motion_case} episodes"
                )

        # One aggregate call retains the strict snapshot and point authority
        # without rescanning the complete room for every route.
        aggregate_paths = {
            f"{episode.episode_id}::{source_id}": path
            for episode in episodes
            for source_id, path in episode.source_center_paths_m.items()
        }
        aggregate_gate = evaluate_source_center_gate(
            self.pathfinder,
            self.obstacle_map,
            aggregate_paths,
            maximum_floor_snap_xz_m=max_snap,
            minimum_navmesh_clearance_m=nav_clearance_threshold,
            minimum_rigid_clearance_m=rigid_clearance_threshold,
        )
        if aggregate_gate["status"] != "pass":
            raise RoomFeasibilityError(
                "aggregate authoritative source-center gate rejected the sampled bank"
            )
        return TrajectoryBank(
            episodes=tuple(episodes),
            frame_count=frames,
            frame_rate_hz=fps,
            seed=seed,
        )


def evaluate_trajectory_diversity(
    bank: TrajectoryBank,
    *,
    minimum_unique_undirected_fraction: float = 0.95,
    minimum_unique_start_fraction: float = 0.70,
    minimum_unique_end_fraction: float = 0.70,
) -> dict[str, Any]:
    """Gate independent source-slot paths instead of episode combinations."""

    gates = {
        "minimum_unique_undirected_fraction": _finite_number(
            minimum_unique_undirected_fraction,
            owner="minimum unique undirected fraction",
            minimum=0.0,
        ),
        "minimum_unique_start_fraction": _finite_number(
            minimum_unique_start_fraction,
            owner="minimum unique start fraction",
            minimum=0.0,
        ),
        "minimum_unique_end_fraction": _finite_number(
            minimum_unique_end_fraction,
            owner="minimum unique end fraction",
            minimum=0.0,
        ),
    }
    if any(value > 1.0 for value in gates.values()):
        raise RoomFeasibilityError("trajectory diversity fractions cannot exceed one")

    source_records: dict[str, Any] = {}
    for source_slot in SOURCE_SLOTS:
        moving_paths = []
        for episode in bank.episodes:
            path = np.asarray(
                episode.source_root_paths_m[source_slot], dtype=np.float64
            )
            path_length = float(
                np.linalg.norm(np.diff(path[:, (0, 2)], axis=0), axis=1).sum()
            )
            if path_length > 1.0e-8:
                moving_paths.append(path)
        directional_signatures: set[bytes] = set()
        undirected_signatures: set[bytes] = set()
        starts: set[tuple[float, float]] = set()
        ends: set[tuple[float, float]] = set()
        straightness: list[float] = []
        for path in moving_paths:
            points = np.round(path[:, (0, 2)], decimals=6)
            forward = points.tobytes()
            reverse = np.ascontiguousarray(points[::-1]).tobytes()
            directional_signatures.add(forward)
            undirected_signatures.add(min(forward, reverse))
            starts.add(tuple(float(value) for value in points[0]))
            ends.add(tuple(float(value) for value in points[-1]))
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            chord = float(np.linalg.norm(points[-1] - points[0]))
            straightness.append(chord / length if length > 1.0e-8 else 1.0)
        use_count = len(moving_paths)
        denominator = max(use_count, 1)
        record = {
            "moving_path_use_count": use_count,
            "unique_directional_path_count": len(directional_signatures),
            "unique_undirected_path_count": len(undirected_signatures),
            "unique_start_count": len(starts),
            "unique_end_count": len(ends),
            "unique_undirected_fraction": len(undirected_signatures) / denominator,
            "unique_start_fraction": len(starts) / denominator,
            "unique_end_fraction": len(ends) / denominator,
            "median_straightness": (
                float(np.median(straightness)) if straightness else None
            ),
        }
        record["status"] = (
            "pass"
            if use_count > 0
            and record["unique_undirected_fraction"]
            >= gates["minimum_unique_undirected_fraction"]
            and record["unique_start_fraction"]
            >= gates["minimum_unique_start_fraction"]
            and record["unique_end_fraction"] >= gates["minimum_unique_end_fraction"]
            else "fail"
        )
        source_records[source_slot] = record
    return {
        "schema": TRAJECTORY_DIVERSITY_SCHEMA,
        "status": (
            "pass"
            if all(record["status"] == "pass" for record in source_records.values())
            else "fail"
        ),
        "semantics": (
            "path uniqueness is measured per source slot; visual/audio bindings "
            "and the other slot cannot make a repeated path unique"
        ),
        "gate": gates,
        "sources": source_records,
    }


def build_rir_job_plan(
    bank: TrajectoryBank,
    *,
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
    stride_frames: int = 3,
) -> dict[str, Any]:
    """Deduplicate exact source/listener states into a reusable RIR work plan.

    This function only plans jobs.  Native RLR execution remains a separate
    operation and must bind the room acoustic package and simulation profile.
    """

    stride = _positive_int(stride_frames, owner="RIR stride frames")
    listener = np.asarray(listener_position_m, dtype=np.float64)
    orientation = np.asarray(listener_orientation_wxyz, dtype=np.float64)
    if (
        listener.shape != (3,)
        or orientation.shape != (4,)
        or not (np.all(np.isfinite(listener)) and np.all(np.isfinite(orientation)))
    ):
        raise RoomFeasibilityError("listener pose is invalid")
    jobs_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    use_count = 0
    for episode in bank.episodes:
        for source_slot, path in sorted(episode.source_center_paths_m.items()):
            points = np.asarray(path, dtype=np.float64)
            for frame_index in range(0, bank.frame_count, stride):
                point = points[frame_index]
                # RLR's current point-source propagation depends on the source
                # state, not on which logical slot or dry waveform uses it.
                # Keep slot identity on the use record so one RIR can serve
                # either source when their acoustic state is identical.
                key = tuple(float(value) for value in point)
                job = jobs_by_key.get(key)
                if job is None:
                    job = {
                        "job_id": f"rir_{len(jobs_by_key):06d}",
                        "source_position_m": point.tolist(),
                        "uses": [],
                    }
                    jobs_by_key[key] = job
                job["uses"].append(
                    {
                        "episode_id": episode.episode_id,
                        "source_slot_id": source_slot,
                        "frame_index": frame_index,
                    }
                )
                use_count += 1
    jobs = list(jobs_by_key.values())
    return {
        "schema": RIR_JOB_PLAN_SCHEMA,
        "status": "planned_not_run",
        "claim_boundary": (
            "RLR execution plan for dry-audio-independent RIR outputs; native RLR "
            "has not run"
        ),
        "producer_backend": "RLR Audio Propagation",
        "cache_artifact": "room impulse response (RIR)",
        "source_acoustic_profile": "omnidirectional_point_source_v1",
        "slot_identity_affects_cache_key": False,
        "dry_audio_independent": True,
        "listener_position_m": listener.tolist(),
        "listener_orientation_wxyz": orientation.tolist(),
        "stride_frames": stride,
        "requested_pair_state_count": use_count,
        "unique_rir_job_count": len(jobs),
        "cache_reuse_count": use_count - len(jobs),
        "jobs": jobs,
    }


__all__ = [
    "FEASIBLE_REGION_SCHEMA",
    "MOTION_CASES",
    "RIR_JOB_PLAN_SCHEMA",
    "RoomFeasibilityCompiler",
    "RoomFeasibilityError",
    "TRAJECTORY_BANK_SCHEMA",
    "TRAJECTORY_COVERAGE_SCHEMA",
    "TRAJECTORY_DIVERSITY_SCHEMA",
    "TrajectoryBank",
    "TrajectoryBankBuilder",
    "TrajectoryCoverage",
    "TrajectoryEpisode",
    "build_rir_job_plan",
    "evaluate_trajectory_coverage",
    "evaluate_trajectory_diversity",
    "SOURCE_SLOTS",
]
