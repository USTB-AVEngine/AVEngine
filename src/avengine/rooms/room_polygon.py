"""Optional room placement constraints in world X/Z metres.

Interior scan holes are filled; the existing navigation still owns standability.
No geometry/runtime dependency is loaded for records without a polygon.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import numpy as np


class RoomPolygonError(ValueError):
    """A malformed polygon or an exhausted bounded placement search."""


def room_polygon_record(room):
    if not isinstance(room, Mapping):
        return None
    if "floor_polygon_xz_m" in room:
        return room
    package = room.get("room_package")
    if not isinstance(package, Mapping):
        return None
    for candidate in (package, package.get("planning_inputs"),
                      package.get("legacy_catalog_entry")):
        if isinstance(candidate, Mapping) and "floor_polygon_xz_m" in candidate:
            return candidate
    return None


class RoomPolygon:
    def __init__(self, geojson, *, boundary_margin_m=0.25,
                 path_sample_step_m=0.25, path_tolerance_m=0.1,
                 max_point_attempts=512, max_route_attempts=32):
        from shapely.geometry import Polygon, shape
        from shapely.ops import unary_union

        for name, value in (("boundary_margin_m", boundary_margin_m),
                            ("path_sample_step_m", path_sample_step_m),
                            ("path_tolerance_m", path_tolerance_m)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise RoomPolygonError(f"{name} must be finite and nonnegative")
        if path_sample_step_m <= 0:
            raise RoomPolygonError("path_sample_step_m must be positive")
        for name, value in (("max_point_attempts", max_point_attempts),
                            ("max_route_attempts", max_route_attempts)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RoomPolygonError(f"{name} must be a positive integer")
        try:
            raw = shape(geojson)
            if raw.geom_type not in ("Polygon", "MultiPolygon") or raw.is_empty:
                raise ValueError("expected nonempty GeoJSON Polygon or MultiPolygon")
            parts = [raw] if raw.geom_type == "Polygon" else list(raw.geoms)
            # Owner policy: ignore every interior scan/furniture hole.
            exteriors = [Polygon(part.exterior) for part in parts]
            if any(not part.is_valid or not np.all(np.isfinite(part.bounds))
                   for part in exteriors):
                raise ValueError("invalid or nonfinite polygon exterior")
            self.geometry = unary_union(exteriors)
            if self.geometry.is_empty or not self.geometry.is_valid:
                raise ValueError("invalid exterior union")
        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            raise RoomPolygonError(f"invalid floor_polygon_xz_m: {exc}") from exc
        self.boundary_margin_m = float(boundary_margin_m)
        self.path_sample_step_m = float(path_sample_step_m)
        self.path_tolerance_m = float(path_tolerance_m)
        self.max_point_attempts = max_point_attempts
        self.max_route_attempts = max_route_attempts
        # Paths are not eroded. Tolerance only admits <=0.1 m navmesh rounding.
        self.path_geometry = self.geometry.buffer(self.path_tolerance_m)
        self.boundary = self.geometry.boundary
        self.bounds_xz = np.asarray(self.geometry.bounds, dtype=float)

    @classmethod
    def from_room(cls, room, request=None):
        record = room_polygon_record(room)
        if record is None:
            return None
        config = dict(record.get("room_polygon_placement") or {})
        config.update((request or {}).get("room_polygon_placement") or {})
        return cls(record["floor_polygon_xz_m"], **config)

    def placement_mask(self, points):
        from shapely import points as make_points, contains, distance
        values = np.asarray(points, dtype=float).reshape(-1, 3)
        x0, z0, x1, z1 = self.bounds_xz
        mask = np.all(np.isfinite(values), axis=1)
        mask &= ((values[:, 0] >= x0) & (values[:, 0] <= x1)
                 & (values[:, 2] >= z0) & (values[:, 2] <= z1))
        ids = np.flatnonzero(mask)
        if len(ids):
            probes = make_points(values[ids, 0], values[ids, 2])
            mask[ids] = (contains(self.geometry, probes)
                         & (distance(self.boundary, probes) >= self.boundary_margin_m))
        return mask

    def placement_contains(self, point):
        return bool(self.placement_mask([point])[0])

    def path_contains(self, point):
        from shapely.geometry import Point
        value = np.asarray(point, dtype=float)
        return bool(value.shape == (3,) and np.all(np.isfinite(value))
                    and self.path_geometry.covers(Point(value[0], value[2])))

    def route_samples(self, route):
        values = np.asarray(route, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.all(np.isfinite(values)):
            raise RoomPolygonError("room_polygon_route_invalid_vertices")
        result = [values[0]]
        for start, end in zip(values[:-1], values[1:]):
            length = float(np.linalg.norm(end - start))
            # Include each vertex and insert points no farther than 0.25 m apart.
            count = max(1, int(math.ceil(length / self.path_sample_step_m)))
            result.extend(start + (end - start) * (i / count) for i in range(1, count + 1))
        return np.asarray(result)

    def route_reason(self, route, *, endpoints=True):
        from shapely.geometry import LineString
        try:
            samples = self.route_samples(route)
        except RoomPolygonError as exc:
            return str(exc)
        if endpoints and not self.placement_mask(samples[[0, -1]]).all():
            return "room_polygon_route_endpoint_below_boundary_margin"
        if not all(self.path_contains(point) for point in samples):
            return "room_polygon_route_vertex_or_0.25m_sample_outside"
        # Also catch a short concave excursion between the inserted samples.
        xz = np.asarray(route, dtype=float)[:, [0, 2]]
        if len(xz) > 1 and not self.path_geometry.covers(LineString(xz)):
            return "room_polygon_route_segment_outside"
        return None


class PolygonPathfinder:
    """Planning-only proxy; the underlying whole-house navmesh stays intact."""
    def __init__(self, pathfinder, polygon):
        self.base = pathfinder
        self.room_polygon = polygon

    def __getattr__(self, name):
        return getattr(self.base, name)

    def is_navigable(self, point, *args, **kwargs):
        return self.room_polygon.path_contains(point) and bool(self.base.is_navigable(point, *args, **kwargs))

    def get_topdown_view(self, resolution, floor, *args, **kwargs):
        grid = np.asarray(self.base.get_topdown_view(resolution, floor, *args, **kwargs)).copy()
        rows, cols = np.nonzero(grid)
        bounds = np.asarray(self.base.get_bounds())
        points = np.column_stack((bounds[0, 0] + (cols + .5) * resolution,
                                  np.full(len(rows), floor),
                                  bounds[0, 2] + (rows + .5) * resolution))
        keep = self.room_polygon.placement_mask(points)
        grid[rows[~keep], cols[~keep]] = 0
        return grid

    def find_path(self, query):
        if not self.room_polygon.placement_mask([query.requested_start, query.requested_end]).all():
            return False
        if not self.base.find_path(query):
            return False
        return self.room_polygon.route_reason(query.points) is None


class PolygonWalkableSpace:
    def __init__(self, space, polygon, floor_y_m=None):
        self.source_spec = None
        self.floor_y_m = floor_y_m
        self.base = space
        self.room_polygon = polygon
        self.pathfinder = PolygonPathfinder(space.pathfinder, polygon)
        self.metadata = dict(space.metadata)
        if floor_y_m is not None:
            if not math.isfinite(float(floor_y_m)):
                raise RoomPolygonError("floor_y_m must be finite")
            self.metadata["floor_height_m"] = float(floor_y_m)
        self.last_polygon_failure = None
        self._qa_native_group_cache = {}

    def __getattr__(self, name):
        value = getattr(self.base, name)
        if name == "points":
            def filtered(region=None):
                bounds = self.bounds() if region is None else np.asarray(region)
                points = np.asarray(value(region), dtype=float).reshape(-1, 3)
                keep = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
                return points[keep & self.room_polygon.placement_mask(points)]
            return filtered
        return value

    def bounds(self):
        bounds = np.asarray(self.base.bounds(), dtype=float).copy()
        x0, z0, x1, z1 = self.room_polygon.bounds_xz
        bounds[0, [0, 2]] = np.maximum(bounds[0, [0, 2]], [x0, z0])
        bounds[1, [0, 2]] = np.minimum(bounds[1, [0, 2]], [x1, z1])
        if self.floor_y_m is not None:
            bounds[0, 1] = max(bounds[0, 1], float(self.floor_y_m) - .3)
            bounds[1, 1] = min(bounds[1, 1], float(self.floor_y_m) + .3)
        return bounds

    def is_navigable(self, point):
        return self.pathfinder.is_navigable(point)

    def sample_navigable(self, rng, region=None):
        region = self.bounds() if region is None else region
        reader = getattr(self, "points", None)
        if callable(reader):
            points = reader(region)
            if not len(points):
                raise RoomPolygonError("room_polygon_no_navigable_points_with_boundary_margin")
            return points[int(rng.integers(len(points)))].copy()
        pf = self.base.pathfinder
        if hasattr(pf, "get_random_navigable_point"):
            pf.seed(int(rng.integers(0, 2**31 - 1)))
        for _ in range(self.room_polygon.max_point_attempts):
            if hasattr(pf, "get_random_navigable_point") and hasattr(pf, "snap_point"):
                # Draw in the small room bbox, then let the unchanged native
                # navmesh supply the actual support. Whole-house rejection is
                # impractical for a tiny split room in a large multi-storey house.
                bounds = np.asarray(region, dtype=float)
                x0, z0, x1, z1 = self.room_polygon.bounds_xz
                low = np.maximum(bounds[0, [0, 2]], [x0, z0])
                high = np.minimum(bounds[1, [0, 2]], [x1, z1])
                if np.any(high <= low):
                    break
                x, z = rng.uniform(low, high)
                probe = [x, self.metadata["floor_height_m"], z]
                if not self.room_polygon.placement_contains(probe):
                    continue
                point = np.asarray(pf.snap_point(np.asarray(probe, dtype=np.float32)), dtype=float)
            else:
                point = (np.asarray(pf.get_random_navigable_point(), dtype=float)
                         if hasattr(pf, "get_random_navigable_point")
                         else np.asarray(self.base.sample_navigable(rng, region), dtype=float))
            if (region is not None and not np.all(
                    (point >= np.asarray(region)[0]) & (point <= np.asarray(region)[1]))):
                continue
            if self.room_polygon.placement_contains(point) and self.base.is_navigable(point):
                return point
        raise RoomPolygonError(
            f"room_polygon_point_attempts_exhausted:{self.room_polygon.max_point_attempts}; "
            f"no native navigable point at boundary_margin_m={self.room_polygon.boundary_margin_m}")

    def shortest_path(self, start, end):
        if not self.room_polygon.placement_mask([start, end]).all():
            self.last_polygon_failure = "room_polygon_route_endpoint_below_boundary_margin"
            return None
        route = self.base.shortest_path(start, end)
        if route is not None:
            reason = self.room_polygon.route_reason(route)
            if reason:
                self.last_polygon_failure = reason
                return None
        return route

    def route_bank(self):
        bank = self.base.route_bank()
        if bank is None:
            return None
        return [row for row in bank
                if self.room_polygon.route_reason(row["points_m"]) is None]


def constrain_space_to_room(space, room, request=None):
    record = room_polygon_record(room)
    if isinstance(space, PolygonWalkableSpace):
        if record is None:
            return space
        spec = {key: record.get(key) for key in
                ("floor_polygon_xz_m", "floor_y_m", "room_polygon_placement")}
        if request is None and space.source_spec == spec:
            return space
        base = space.base
    else:
        base = space
    polygon = RoomPolygon.from_room(room, request)
    if polygon is None:
        return base
    result = PolygonWalkableSpace(base, polygon, record.get("floor_y_m"))
    result.source_spec = deepcopy({key: record.get(key) for key in
                                   ("floor_polygon_xz_m", "floor_y_m", "room_polygon_placement")})
    return result


def constrain_pathfinder_to_room(pathfinder, room, request=None):
    polygon = RoomPolygon.from_room(room, request)
    return pathfinder if polygon is None else PolygonPathfinder(pathfinder, polygon)
