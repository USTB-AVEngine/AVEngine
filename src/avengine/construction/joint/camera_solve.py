"""Demand-driven fixed horizontal cameras; continuous yaw, finite floor grid.

All angles are degrees, image-right positive. BodySpec is the caller's enclosing
all-pose/all-requested-heading envelope (the same box used by core framing).
Occlusion is sampled planning geometry, NEVER native mask certification.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import math
import time
import numpy as np

from .types import BodySpec, CameraPose
from .geometry import body_box, camera_coordinates, framing

_TAU = 2 * math.pi
_EPS = 1e-10  # numerical tolerance, not a scene-admission margin
_KINDS = {"inside_full_body", "outside_full_body", "outside_left", "outside_right",
          "distance_range", "azimuth_sector", "occluded_by", "line_of_sight_clear", "rest_full_body"}


@dataclass(frozen=True)
class PointRole:
    kind: str
    point_m: tuple[float, float, float]
    body: BodySpec = BodySpec()
    name: str = ""
    pixel_margin_px: float = 0.
    distance_m: tuple[float, float] = (0., math.inf)
    azimuth_deg: tuple[float, float] | None = None
    occluder_id: str | None = None
    heading_range_deg: tuple[float, float] | None = None

    def __post_init__(self):
        p = np.asarray(self.point_m, float)
        if self.kind not in _KINDS or p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("role requires a supported kind and finite world point")
        object.__setattr__(self, "point_m", tuple(map(float, p)))
        if not math.isfinite(self.pixel_margin_px) or self.pixel_margin_px < 0:
            raise ValueError("pixel margin must be nonnegative")
        lo, hi = self.distance_m
        if not math.isfinite(lo) or math.isnan(hi) or not 0 <= lo <= hi:
            raise ValueError("invalid distance range")
        if self.kind == "azimuth_sector" and self.azimuth_deg is None:
            raise ValueError("azimuth_sector needs an explicit interval")
        for interval in (self.azimuth_deg, self.heading_range_deg):
            if interval is not None and (len(interval) != 2 or not np.isfinite(interval).all()):
                raise ValueError("angular interval must have two finite bounds")
        if self.kind == "occluded_by" and not self.occluder_id:
            raise ValueError("occluded_by needs an explicit registered identity")


@dataclass(frozen=True)
class RouteRole:
    points_m: tuple[tuple[float, float, float], ...]
    start_role: str | PointRole = "inside_full_body"
    end_role: str | PointRole = "inside_full_body"
    body: BodySpec = BodySpec()
    name: str = "route"
    pixel_margin_px: float = 0.
    distance_change: str | None = None  # nearer / farther
    min_distance_change_m: float = 0.
    waypoint_roles: tuple[PointRole, ...] = ()

    def __post_init__(self):
        p = np.asarray(self.points_m, float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError("route needs at least two finite world points")
        object.__setattr__(self, "points_m", tuple(map(tuple, p.tolist())))
        for endpoint in (self.start_role, self.end_role):
            if not isinstance(endpoint, PointRole) and endpoint not in _KINDS:
                raise ValueError("invalid endpoint role")
        if not math.isfinite(self.pixel_margin_px) or self.pixel_margin_px < 0:
            raise ValueError("pixel margin must be nonnegative")
        if self.distance_change not in {None, "nearer", "farther"}:
            raise ValueError("distance_change must be nearer or farther")
        if not math.isfinite(self.min_distance_change_m) or self.min_distance_change_m < 0:
            raise ValueError("distance change must be nonnegative")


@dataclass
class CameraSolution:
    camera: CameraPose
    yaw_margin_deg: float
    wall_clearance_m: float
    role_margins: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)


@dataclass
class CameraSolveResult:
    solutions: tuple[CameraSolution, ...]
    diagnostics: dict

    @property
    def cameras(self):
        return tuple(s.camera for s in self.solutions)


@dataclass
class RolePointResult:
    points_m: np.ndarray
    indices: np.ndarray
    mask: np.ndarray
    diagnostics: dict


def roles_for_endpoints(kind, start_m, end_m, *, body=BodySpec(), pixel_margin_px=0.,
                        min_distance_change_m=None, waypoints_m=None,
                        hidden_point_m=None, occluder_id=None, name="route"):
    """One-line endpoint -> roles helper; route construction stays route_between's job.

    kind: approach, recede, enter_left, enter_right, through_occlusion.
    For a routed path pass waypoints_m (including both endpoints). A hidden point
    is required for through_occlusion and must lie on the supplied polyline.
    Distance thresholds and extra pixel margins come from the caller/QA rule.
    """
    if kind not in {"approach", "recede", "enter_left", "enter_right", "through_occlusion"}:
        raise ValueError("unsupported endpoint relation")
    if kind in {"approach", "recede"} and min_distance_change_m is None:
        raise ValueError("supply the answer's required distance change")
    path = np.asarray(waypoints_m if waypoints_m is not None else [start_m, end_m], float)
    if not np.allclose(path[0], start_m) or not np.allclose(path[-1], end_m):
        raise ValueError("polyline endpoints must match the declared endpoints")
    extra = ()
    if kind == "through_occlusion":
        if hidden_point_m is None or not occluder_id:
            raise ValueError("supply a hidden waypoint and registered occluder_id")
        if waypoints_m is None:
            path = np.asarray([start_m, hidden_point_m, end_m], float)
        if _path_distance(np.asarray(hidden_point_m), path, horizontal=False) > _EPS:
            raise ValueError("hidden waypoint must lie on the route")
        extra = (PointRole("occluded_by", hidden_point_m, body, name+".hidden", occluder_id=occluder_id),
                 PointRole("line_of_sight_clear", start_m, body, name+".start_clear"),
                 PointRole("line_of_sight_clear", end_m, body, name+".end_clear"))
    return (RouteRole(tuple(map(tuple, path)),
                      {"enter_left": "outside_left", "enter_right": "outside_right"}.get(kind, "inside_full_body"),
                      "inside_full_body", body, name, pixel_margin_px,
                      {"approach": "nearer", "recede": "farther"}.get(kind),
                      float(min_distance_change_m or 0.), extra),)


def _merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if hi < lo - _EPS:
            continue
        if out and lo <= out[-1][1] + _EPS:
            out[-1] = (out[-1][0], max(hi, out[-1][1]))
        else:
            out.append((lo, hi))
    return out


def _arc(lo, hi):
    width = hi - lo
    if width >= _TAU - _EPS:
        return [(0., _TAU)]
    if width < -_EPS:
        return []
    lo %= _TAU
    hi = lo + max(0., width)
    return [(lo, hi)] if hi <= _TAU else [(0., hi-_TAU), (lo, _TAU)]


def _intersect(a, b):
    return _merge([(max(x, u), min(y, v)) for x, y in a for u, v in b
                   if max(x, u) <= min(y, v) + _EPS])


def _sin_constraint(a, b, c):
    """Exact circular intervals for a*sin(yaw)+b*cos(yaw) >= c."""
    radius = math.hypot(a, b)
    if radius <= _EPS:
        return [(0., _TAU)] if c <= _EPS else []
    ratio = c / radius
    if ratio > 1 + _EPS:
        return []
    if ratio <= -1:
        return [(0., _TAU)]
    center = math.atan2(a, b)
    half = math.acos(max(-1., min(1., ratio)))
    return _arc(center-half, center+half)


def _planes_intersection(planes):
    out = [(0., _TAU)]
    for a, b, c in planes:
        out = _intersect(out, _sin_constraint(float(a), float(b), float(c)))
        if not out:
            break
    return out


def _point_intervals(role, position, template):
    kind = role.kind
    if kind == "rest_full_body":
        return _merge(_point_intervals(replace(role, kind="inside_full_body"), position, template) +
                      _point_intervals(replace(role, kind="outside_full_body"), position, template))
    if kind in {"distance_range", "occluded_by", "line_of_sight_clear"}:
        return [(0., _TAU)]
    if kind == "azimuth_sector":
        d = np.asarray(role.point_m) - position
        if np.linalg.norm(d[[0, 2]]) <= _EPS:
            return []
        bearing = math.atan2(d[0], -d[2])
        lo, hi = np.radians(role.azimuth_deg)
        width = hi-lo if hi >= lo else (hi-lo) % _TAU
        return _arc(bearing-lo-width, bearing-lo)
    xyz = body_box(role.point_m, role.body) - position
    x, y, z = xyz.T
    inside = kind == "inside_full_body"
    m = -role.pixel_margin_px if inside else role.pixel_margin_px
    th, tv = (template.width/2+m)/template.focal_px, (template.height/2+m)/template.focal_px
    if inside and (th <= 0 or tv <= 0):
        return []
    # front, right, left, top, bottom, each a*sin(yaw)+b*cos(yaw)>=c.
    planes = [np.column_stack((x, -z, np.full(len(x), .01))),
              np.column_stack((th*x-z, -th*z-x, np.zeros(len(x)))),
              np.column_stack((th*x+z, -th*z+x, np.zeros(len(x)))),
              np.column_stack((tv*x, -tv*z, y)),
              np.column_stack((tv*x, -tv*z, -y))]
    if inside:
        return _planes_intersection(np.concatenate(planes))
    selected = {"outside_left": [2], "outside_right": [1]}.get(kind, range(5))
    out = []
    for i in selected:
        constraints = -planes[i]
        if kind in {"outside_left", "outside_right"}:
            # Behind the camera both side clip planes can be violated. Require
            # the named lateral half-space too, so left/right is unambiguous.
            sign = -1 if kind == "outside_left" else 1
            lateral = np.column_stack((sign*z, sign*x, np.zeros(len(x))))
            constraints = np.concatenate((constraints, lateral))
        out.extend(_planes_intersection(constraints))
    return _merge(out)



def _outside_slack(camera, role):
    """Strict separating-plane slack in metres, also valid behind the camera."""
    xyz = camera_coordinates(camera, body_box(role.point_m, role.body))
    x, y, z = xyz.T
    th = (camera.width/2+role.pixel_margin_px)/camera.focal_px
    tv = (camera.height/2+role.pixel_margin_px)/camera.focal_px
    planes = np.column_stack((z-.01, th*z-x, th*z+x, tv*z-y, tv*z+y))
    if role.kind in {"outside_left", "outside_right"}:
        side = 2 if role.kind == "outside_left" else 1
        sign = -1 if side == 2 else 1
        return float(min((-planes[:, side]).min(), (sign*x).min()))
    return float((-planes).min(axis=0).max())


def _strict_framing_ok(camera, role):
    if role.kind.startswith("outside_"):
        return _outside_slack(camera, role) > 0
    if role.kind == "rest_full_body":
        state, edge = framing(camera, role.point_m, role.body)
        return ((state == "inside" and edge >= role.pixel_margin_px) or
                _outside_slack(camera, replace(role, kind="outside_full_body")) > 0)
    return True

def _circular_components(intervals):
    if len(intervals) > 1 and intervals[0][0] <= _EPS and intervals[-1][1] >= _TAU-_EPS:
        return [(intervals[-1][0], intervals[0][1]+_TAU)] + intervals[1:-1]
    return intervals


def _inside_yaws(polygon, point):
    """Original camera_legal tests the point .75m ahead; solve that exactly.

    Circle/polygon boundary intersections partition all possible headings, so
    concave rooms and holes work without a sampled heading table.
    """
    from shapely.geometry import Point
    reach = .75  # existing camera_legal contract, not a new clearance
    cuts = [0., _TAU]
    parts = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    for part in parts:
        for ring in (part.exterior, *part.interiors):
            coords = np.asarray(ring.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                q, d = a-point, b-a
                aa, bb, cc = float(d@d), float(2*q@d), float(q@q-reach**2)
                if aa <= _EPS:
                    continue
                discriminant = bb*bb-4*aa*cc
                if discriminant < -_EPS:
                    continue
                root = math.sqrt(max(0., discriminant))
                for t in ((-bb-root)/(2*aa), (-bb+root)/(2*aa)):
                    if -_EPS <= t <= 1+_EPS:
                        v = q + t*d
                        cuts.append(math.atan2(v[0], -v[1]) % _TAU)
    cuts = sorted(set(cuts))
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid = (lo+hi)/2
        if polygon.covers(Point(point + reach*np.array([math.sin(mid), -math.cos(mid)]))):
            out.append((lo, hi))
    # Include isolated legal tangencies as zero-width domains too.
    for a in cuts:
        if polygon.covers(Point(point + reach*np.array([math.sin(a), -math.cos(a)]))):
            out.append((a, a))
    return _merge(out)


def _cache(context):
    if not hasattr(context, "_camera_solve_cache"):
        context._camera_solve_cache = {}
    return context._camera_solve_cache


def _positions(context):
    from shapely.geometry import Point, shape
    cache = _cache(context)
    if "positions" in cache:
        return cache["positions"]
    polygon = shape(context.room["floor_polygon"])
    rows = []
    rejected = Counter()
    for p in context.points_m:
        point = Point(p[[0, 2]])
        clearance = float(polygon.boundary.distance(point))
        if not polygon.covers(point) or clearance+1e-7 < .5:
            rejected["wall_or_room"] += 1
            continue
        if not context.walkable_map.contains(p[[0, 2]], "human"):
            rejected["walkable_camera"] += 1
            continue
        if context.pathfinder is not None:
            snap = np.asarray(context.pathfinder.snap_point(p), float)
            if snap.shape != (3,) or not np.isfinite(snap).all() or np.linalg.norm((snap-p)[[0, 2]]) > .02:
                rejected["camera_navmesh"] += 1
                continue
        inward = _inside_yaws(polygon, p[[0, 2]])
        if inward:
            rows.append((p.copy(), clearance, inward))
        else:
            rejected["inward"] += 1
    cache["positions"] = (tuple(rows), dict(rejected))
    return cache["positions"]


def _expand(roles):
    points, routes = [], []
    for i, role in enumerate(roles):
        if isinstance(role, RouteRole):
            name = role.name or str(i)
            routes.append((name, role))
            for suffix, spec, p in ((".start", role.start_role, role.points_m[0]),
                                    (".end", role.end_role, role.points_m[-1])):
                point = (replace(spec, point_m=p) if isinstance(spec, PointRole) else
                         PointRole(spec, p, role.body, pixel_margin_px=role.pixel_margin_px))
                points.append((name+suffix, point))
            points.extend((p.name or name+".via"+str(j), p) for j, p in enumerate(role.waypoint_roles))
        elif isinstance(role, PointRole):
            points.append((role.name or f"{i}:{role.kind}", role))
        else:
            raise TypeError("expected PointRole or RouteRole")
    names = [n for n, _ in points+routes]
    if len(names) != len(set(names)):
        raise ValueError("role names must be unique")
    return points, routes


def _path_distance(point, path, *, horizontal=True):
    p, q = np.asarray(point), np.asarray(path)
    if horizontal:
        p, q = p[[0, 2]], q[:, [0, 2]]
    a, v = q[:-1], np.diff(q, axis=0)
    length2 = np.sum(v*v, axis=1)
    t = np.clip(np.sum((p-a)*v, axis=1)/np.where(length2 > 0, length2, 1.), 0., 1.)
    return float(np.linalg.norm(a+t[:, None]*v-p, axis=1).min())


def _distance_checks(position, points, routes):
    margins, failures = {}, []
    for name, role in points:
        d = float(np.linalg.norm((np.asarray(role.point_m)-position)[[0, 2]]))
        low, high = role.distance_m
        low = max(low, 1.5 if role.body.species == "human" else 0.)
        margin = min(d-low, high-d)
        margins[name] = {"distance_m": d, "distance_slack_m": margin}
        if margin < -_EPS:
            failures.append((name, "distance"))
    for name, role in routes:
        path = np.asarray(role.points_m)
        dmin = _path_distance(position, path)
        margins[name] = {"route_camera_distance_m": dmin}
        if role.body.species == "human" and dmin < 1.5-_EPS:
            failures.append((name, "route_camera_distance"))
        if role.distance_change:
            d = np.linalg.norm((path[[0, -1]]-position)[:, [0, 2]], axis=1)
            delta = float(d[0]-d[1] if role.distance_change == "nearer" else d[1]-d[0])
            margins[name].update(distance_change_m=delta,
                                 distance_change_slack_m=delta-role.min_distance_change_m)
            if delta < role.min_distance_change_m-_EPS:
                failures.append((name, "distance_change"))
    return margins, failures


def _ray_check(context, position, role):
    ray = context.raycaster
    if ray is None or getattr(ray, "source", "unknown") != "visual_mesh":
        return False, {"reason": "missing_visual_geometry", "authority": "unknown"}
    # Same envelope ray pattern as RoomContext.grid, cached independently of yaw.
    key = ("rays", id(ray), tuple(position), role.point_m, role.body)
    cache = _cache(context)
    if key not in cache:
        r = role.body.radius_m + role.body.margin_m
        angles = np.arange(12) * (_TAU/12)
        offsets = np.array([[r*math.cos(a), h, r*math.sin(a)]
                            for h in (.05, role.body.height_m/2, role.body.height_m+role.body.margin_m)
                            for a in angles])
        delta = np.asarray(role.point_m) + offsets - position
        ranges = np.linalg.norm(delta, axis=1)
        hit, ids = ray.first_hit(position, delta)
        cache[key] = (np.asarray(hit), np.asarray(ids), ranges)
    hit, ids, ranges = cache[key]
    known = not np.isnan(hit).any()
    blocked = hit < ranges  # no new depth cutoff; envelope already has caller margin
    if role.kind == "occluded_by":
        passed = known and bool(np.all(blocked & (ids == role.occluder_id)))
    else:
        passed = known and bool(np.all(~blocked))
    return passed, {"authority": "sampled_outer_envelope_broad_phase_only",
                    "native_pixel_truth": False, "sample_count": len(hit),
                    "blocked_fraction": float(blocked.mean()),
                    "blocker_ids": sorted(set(map(str, ids[blocked]))),
                    "reason": "pass" if passed else ("unknown_rays" if not known else "ray_role")}


def _angular_slack(intervals, yaw):
    if intervals == [(0., _TAU)]:
        return 180.
    for lo, hi in _circular_components(intervals):
        for y in (yaw % _TAU, yaw % _TAU + _TAU):
            if lo-_EPS <= y <= hi+_EPS:
                return math.degrees(min(y-lo, hi-y))
    return -math.inf


def solve_cameras(context, roles, *, limit=None, camera_heights_m=None,
                  hfov_deg=85., width=None, height=None):
    """Solve all legal floor-grid sites/heights; no position/yaw candidate cap.

    limit only truncates the sorted return value. yaw_margin_deg is the maximum
    symmetric yaw perturbation within a feasible connected component. Height
    uses the context's existing height set; there is no pitch. Empty results
    prove only the selected floor grid/envelope/height domain was exhausted.
    """
    start = time.monotonic()
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    points, routes = _expand(tuple(roles))
    heights = tuple(camera_heights_m) if camera_heights_m is not None else context.camera_heights_m
    if not heights or any(not math.isfinite(h) or h <= .05 for h in heights):
        raise ValueError("use the existing positive camera height set")
    template = CameraPose((0., 0., 0.), 0., hfov_deg=hfov_deg,
                          width=context.camera_width if width is None else width,
                          height=context.camera_height if height is None else height)
    positions, position_rejections = _positions(context)
    counts = {name: Counter() for name, _ in points+routes}
    solutions = []
    considered = 0
    for ground, clearance, inward in positions:
        for h in heights:
            considered += 1
            position = ground + [0., h, 0.]
            margins, failures = _distance_checks(position, points, routes)
            if failures:
                for name, stage in failures:
                    counts[name][stage] += 1
                continue
            feasible, per_role = inward, {}
            for name, role in points:
                allowed = _point_intervals(role, position, template)
                per_role[name] = allowed
                joint = _intersect(feasible, allowed)
                if not joint:
                    counts[name]["angular_empty" if not allowed else "angular_intersection"] += 1
                    feasible = []
                    break
                feasible = joint
            if not feasible:
                continue
            for name, role in points:
                if role.kind in {"occluded_by", "line_of_sight_clear"}:
                    passed, evidence = _ray_check(context, position, role)
                    margins[name].update(evidence)
                    if not passed:
                        counts[name][evidence["reason"]] += 1
                        feasible = []
            for lo, hi in _circular_components(feasible):
                yaw = (lo+hi)/2
                camera = replace(template, position_m=tuple(position),
                                 yaw_deg=(math.degrees(yaw)+180)%360-180,
                                 camera_id=f"solved_{len(solutions):06d}")
                strict_failures = [name for name, role in points
                                   if not _strict_framing_ok(camera, role)]
                if strict_failures:
                    for name in strict_failures:
                        counts[name]["outside_boundary_touch"] += 1
                    continue
                role_margins = {name: dict(value) for name, value in margins.items()}
                for name, role in points:
                    role_margins[name]["yaw_slack_deg"] = _angular_slack(per_role[name], yaw)
                    if role.kind == "rest_full_body":
                        role_margins[name]["framing_state"] = framing(camera, role.point_m, role.body)[0]
                    if role.kind.startswith("outside_"):
                        role_margins[name]["separating_plane_slack_m"] = _outside_slack(camera, role)
                    if role.kind == "inside_full_body":
                        xyz = camera_coordinates(camera, body_box(role.point_m, role.body))
                        uv = camera.focal_px * xyz[:, :2] / xyz[:, 2, None]
                        role_margins[name]["edge_margin_px"] = float(min(
                            (camera.width/2-np.abs(uv[:, 0])).min(),
                            (camera.height/2-np.abs(uv[:, 1])).min()))
                solutions.append(CameraSolution(camera, math.degrees((hi-lo)/2), clearance, role_margins,
                    {"authority": "planning_geometry_only", "pitch_deg": 0.,
                     "heading_envelope": "caller_body_covers_requested_heading_range"}))
    solutions.sort(key=lambda s: (-s.yaw_margin_deg, -s.wall_clearance_m,
                                  s.camera.position_m, s.camera.yaw_deg))
    total = len(solutions)
    if limit is not None:
        solutions = solutions[:limit]
    return CameraSolveResult(tuple(solutions), {
        "positions_in_grid": len(context.points_m), "legal_positions": len(positions),
        "heights_m": list(heights), "position_height_pairs": considered,
        "position_rejections": position_rejections,
        "role_rejections": {name: dict(value) for name, value in counts.items()},
        "rejection_counting": "distance independent; angular first eliminating role; rays independent",
        "feasible_components": total, "returned": len(solutions), "elapsed_s": time.monotonic()-start,
        "continuous_room_infeasibility_proved": False, "native_pixel_truth": False})


def role_points(context, camera, role):
    """Reverse query: substitute every context grid point into a PointRole.

    Uses the same exact yaw-domain/framing logic as solve_cameras; a single role
    mask can be intersected with other role masks and existing context.grid.
    """
    if not isinstance(role, PointRole):
        raise TypeError("reverse query requires a PointRole template")
    mask = np.zeros(len(context.points_m), bool)
    rejected = Counter()
    position = np.asarray(camera.position_m)
    for i, p in enumerate(context.points_m):
        current = replace(role, point_m=tuple(p))
        _, failures = _distance_checks(position, [("point", current)], [])
        if failures:
            rejected[failures[0][1]] += 1
            continue
        allowed = _point_intervals(current, position, camera)
        if _angular_slack(allowed, math.radians(camera.yaw_deg)) < -_EPS:
            rejected["angular"] += 1
            continue
        if not _strict_framing_ok(camera, current):
            rejected["outside_boundary_touch"] += 1
            continue
        if current.kind in {"occluded_by", "line_of_sight_clear"}:
            passed, evidence = _ray_check(context, position, current)
            if not passed:
                rejected[evidence["reason"]] += 1
                continue
        mask[i] = True
    indices = np.flatnonzero(mask)
    return RolePointResult(context.points_m[indices], indices, mask,
                           {"tested": len(mask), "matched": len(indices), "rejections": dict(rejected),
                            "native_pixel_truth": False})


def check_camera_roles(context, camera, roles):
    """Check a supplied camera against the same roles, useful for legacy replay.

    Returns a planning-only record. Full scene legality/route execution still
    belongs to certify_plan and the family/native predicates.
    """
    from .context import camera_legal
    points, routes = _expand(tuple(roles))
    position = np.asarray(camera.position_m)
    margins, failures = _distance_checks(position, points, routes)
    legal, clearance = camera_legal(context.room, camera)
    if not legal:
        failures.append(("camera", "camera_legal"))
    for name, role in points:
        slack = _angular_slack(_point_intervals(role, position, camera), math.radians(camera.yaw_deg))
        margins[name]["yaw_slack_deg"] = slack
        if slack < -_EPS:
            failures.append((name, "angular"))
        if not _strict_framing_ok(camera, role):
            failures.append((name, "outside_boundary_touch"))
        if role.kind in {"occluded_by", "line_of_sight_clear"}:
            passed, evidence = _ray_check(context, position, role)
            margins[name].update(evidence)
            if not passed:
                failures.append((name, evidence["reason"]))
    return {"passed": not failures, "failures": failures, "role_margins": margins,
            "wall_clearance_m": clearance, "authority": "planning_geometry_only",
            "native_pixel_truth": False}
