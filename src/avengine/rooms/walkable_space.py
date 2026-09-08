"""Small adapters around existing navigation; all coordinates are meters/+Y."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

from avengine.routes.raster_pathfinder import RasterShortestPath


@dataclass
class RasterWalkableSpace:
    pathfinder: Any
    metadata: dict[str, Any]

    def is_navigable(self, point):
        return bool(self.pathfinder.is_navigable(np.asarray(point, dtype=float)))

    def shortest_path(self, start, end):
        from avengine.rooms.qa_episode import _path
        if np.linalg.norm(np.asarray(start)-end) < 1e-8:
            return np.asarray([start, end], dtype=float)
        return _path(self.pathfinder, np.asarray(start), np.asarray(end))

    def sample_navigable(self, rng, region=None):
        points = self.points(region)
        if not len(points):
            raise ValueError('requested region has no navigable cells')
        return points[int(rng.integers(len(points)))].copy()

    def points(self, region=None):
        from avengine.rooms.qa_episode import navigation_points
        points = navigation_points(self.pathfinder, self.metadata)
        if region is not None:
            bounds = np.asarray(region, dtype=float)
            points = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
        return points

    def floor_height(self, point):
        return float(self.metadata['floor_height_m'])

    def bounds(self):
        return np.asarray(self.pathfinder.get_bounds(), dtype=float)

    def route_bank(self):
        return None


@dataclass
class NativeRouteWalkableSpace(RasterWalkableSpace):
    routes: Sequence[Mapping[str, Any]]
    frame_rate_hz: float

    def shortest_path(self, start, end):
        raise ValueError('native route-bank space only permits retained native routes')

    def route_bank(self):
        return self.routes


@dataclass
class HabitatWalkableSpace:
    pathfinder: Any
    metadata: dict[str, Any]

    def is_navigable(self, point):
        return bool(self.pathfinder.is_navigable(np.asarray(point, dtype=float)))

    def shortest_path(self, start, end):
        import habitat_sim
        query = habitat_sim.ShortestPath()
        query.requested_start = np.asarray(start, dtype=np.float32)
        query.requested_end = np.asarray(end, dtype=np.float32)
        return np.asarray(query.points, dtype=float) if self.pathfinder.find_path(query) else None

    def sample_navigable(self, rng, region=None):
        # Habitat's RNG belongs to this PathFinder, and receives the request RNG seed.
        self.pathfinder.seed(int(rng.integers(0, 2**31-1)))
        for _ in range(512):
            p = np.asarray(self.pathfinder.get_random_navigable_point(), dtype=float)
            if np.all(np.isfinite(p)) and (region is None or np.all((p >= region[0]) & (p <= region[1]))):
                return p
        raise ValueError('native navmesh has no sampled point in the requested region')

    def floor_height(self, point):
        point = np.asarray(self.pathfinder.snap_point(np.asarray(point, dtype=np.float32)), dtype=float)
        if not np.all(np.isfinite(point)):
            raise ValueError('native navmesh cannot measure floor at this point')
        return float(point[1])

    def bounds(self):
        return np.asarray(self.pathfinder.get_bounds(), dtype=float)

    def route_bank(self):
        return None


def camera_grid(space, *, step_m=.55, height_above_floor_m=1.55, region=None):
    """Full existing-resolution grid; no ranked/truncated position subset."""
    bounds = space.bounds() if region is None else np.asarray(region, dtype=float)
    result = []
    for x in np.arange(bounds[0, 0]+step_m/2, bounds[1, 0], step_m):
        for z in np.arange(bounds[0, 2]+step_m/2, bounds[1, 2], step_m):
            probe = [x, space.metadata['floor_height_m'], z]
            try:
                floor = space.floor_height(probe)
            except ValueError:
                continue
            point = [x, floor, z]
            if space.is_navigable(point):
                result.append([float(x), floor+height_above_floor_m, float(z)])
    return result
