"""Species-aware straight/rounded stand-walk-stand routes on the v2 walkable graph."""
from __future__ import annotations
import heapq
import math
import numpy as np
from .types import ActorPlan, BodySpec

SPEED_BANDS = {"human": (.8, 1.4), "dog": (.5, 1.2), "cat": (.3, .8), "slide": (.2, .5)}


class ConstructionError(ValueError):
    def __init__(self, reason, details=None):
        self.reason, self.details = reason, dict(details or {})
        super().__init__(reason)


def minimum_track_distance(a, b):
    """Exact minimum of piecewise-linear relative XZ motion, including between frames."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = a[..., [0, 2]] - b[..., [0, 2]]
    if delta.ndim == 1:
        return float(np.linalg.norm(delta))
    if len(delta) < 2:
        return float(np.linalg.norm(delta, axis=1).min())
    v = np.diff(delta, axis=0)
    den = np.sum(v*v, axis=1)
    t = np.clip(-np.sum(delta[:-1]*v, axis=1) / np.maximum(den, 1e-30), 0, 1)
    return float(np.linalg.norm(delta[:-1] + t[:, None]*v, axis=1).min())


def validate_track(context, positions_m, species="human", *, camera=None, snap=True):
    p = np.asarray(positions_m, float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ConstructionError("invalid_route_points")
    m = context.walkable_map
    if not m.contains_many(p[:, [0, 2]], species).all():
        raise ConstructionError("walkable_map_cells", {"rejected_frames": int((~m.contains_many(p[:, [0, 2]], species)).sum())})
    segments = np.unique(np.concatenate((p[:-1, [0, 2]], p[1:, [0, 2]]), axis=1), axis=0)
    for segment in segments:
        if not m.segment_free(segment[:2], segment[2:], species):
            raise ConstructionError("walkable_map_segment", {"segment_xz_m": segment.tolist()})
    q, max_snap = p.copy(), 0.
    nav_y_offset = None
    if context.pathfinder is not None:
        unique, inverse = np.unique(p, axis=0, return_inverse=True)
        snapped = np.asarray([context.pathfinder.snap_point(v) for v in unique], float)
        if snapped.shape != unique.shape or not np.isfinite(snapped).all():
            raise ConstructionError("navmesh_snap_nonfinite")
        horizontal = np.linalg.norm((snapped-unique)[:, [0, 2]], axis=1)
        # Recast NavMesh Y can be elevated above the physical scan floor (e.g.
        # 0.166m in HM3D 00591). It is NOT a root/sole height measurement.
        # Keep actual floor Y; native execution likewise gates horizontal drift.
        nav_y_offset = [float((snapped[:, 1]-unique[:, 1]).min()),
                        float((snapped[:, 1]-unique[:, 1]).max())]
        max_snap = float(horizontal.max())
        if max_snap > .002:
            raise ConstructionError("navmesh_snap_tolerance", {"horizontal_m": max_snap, "navmesh_y_offset_range_m": nav_y_offset})
        if snap:
            q[:, [0, 2]] = snapped[inverse][:, [0, 2]]
            if not m.contains_many(q[:, [0, 2]], species).all():
                raise ConstructionError("snapped_route_outside_map")
            for seg in np.unique(np.concatenate((q[:-1, [0, 2]], q[1:, [0, 2]]), axis=1), axis=0):
                if not m.segment_free(seg[:2], seg[2:], species):
                    raise ConstructionError("snapped_segment_outside_map")
    if np.max(np.abs(q[:, 1]-m.floor_reference)) > .08:
        raise ConstructionError("route_changes_floor")
    distance = None
    if camera is not None:
        distance = minimum_track_distance(q, np.asarray(camera.position_m))
        required = 1.5 if species == "human" else .5
        if distance < required - 1e-8:
            raise ConstructionError("camera_standoff", {"minimum_m": distance, "required_m": required})
    return q, {"map_full_frames": len(q), "map_supercover_segments": len(segments),
               "navmesh_checked": context.pathfinder is not None, "max_snap_horizontal_m": max_snap,
               "minimum_camera_distance_m": distance, "navmesh_y_offset_range_m": nav_y_offset,
               "vertical_authority": "physical_scan_floor_not_navmesh_height"}


def route_between(context, start_m, end_m, *, species="human", max_expansions=100000):
    m = context.walkable_map
    start, end = np.asarray(start_m, float), np.asarray(end_m, float)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all():
        raise ConstructionError("invalid_route_endpoints")
    if not m.contains_many(np.array([start[[0, 2]], end[[0, 2]]]), species).all():
        raise ConstructionError("route_endpoints_not_walkable")
    if m.segment_free(start[[0, 2]], end[[0, 2]], species):
        return np.array([start, end])
    def cell(p):
        x, z = np.floor((p[[0, 2]]-m.origin)/m.resolution).astype(int)
        return int(z), int(x)
    first, goal = cell(start), cell(end)
    mask = m.layers[species]
    queue, cost, previous = [(0., first)], {first: 0.}, {}
    closed = set()
    neighbours = tuple((z, x) for z in (-1, 0, 1) for x in (-1, 0, 1) if z or x)
    while queue:
        _, u = heapq.heappop(queue)
        if u in closed:
            continue
        if u == goal:
            break
        closed.add(u)
        if len(closed) >= max_expansions:
            raise ConstructionError("route_search_budget", {"expanded": len(closed)})
        for dz, dx in neighbours:
            v = u[0]+dz, u[1]+dx
            if not (0 <= v[0] < m.shape[0] and 0 <= v[1] < m.shape[1]) or not mask[v]:
                continue
            if dz and dx and (not mask[u[0]+dz, u[1]] or not mask[u[0], u[1]+dx]):
                continue
            c = cost[u] + math.hypot(dz, dx)
            if c < cost.get(v, math.inf):
                cost[v], previous[v] = c, u
                h = math.hypot(v[0]-goal[0], v[1]-goal[1])
                heapq.heappush(queue, (c+h, v))
    if goal not in cost:
        raise ConstructionError("no_grid_route", {"expanded": len(closed), "domain": "selected_species_v2_grid"})
    cells, u = [goal], goal
    while u != first:
        u = previous[u]
        cells.append(u)
    cells.reverse()
    raw = [start]
    raw.extend(np.array([m.origin[0]+(x+.5)*m.resolution, m.floor_reference,
                         m.origin[1]+(z+.5)*m.resolution]) for z, x in cells[1:-1])
    raw.append(end)
    pulled, i = [raw[0]], 0
    while i < len(raw)-1:
        j = len(raw)-1
        while j > i+1 and not m.segment_free(raw[i][[0, 2]], raw[j][[0, 2]], species):
            j -= 1
        pulled.append(raw[j])
        i = j
    return np.array(pulled)


def _rounded_pieces(points, radius):
    """Tangent fillets in XZ; recomputed collision checks follow sampling."""
    points = np.asarray(points, float)
    pieces, at = [], points[0]
    def line(a, b):
        length = float(np.linalg.norm((b-a)[[0, 2]]))
        if length > 1e-9:
            pieces.append(("line", a.copy(), b.copy(), length))
    for i in range(1, len(points)-1):
        a, b, c = points[i-1], points[i], points[i+1]
        d0, d1 = (b-a)[[0, 2]], (c-b)[[0, 2]]
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if min(l0, l1) < 1e-8:
            raise ConstructionError("duplicate_waypoints")
        d0, d1 = d0/l0, d1/l1
        turn = math.atan2(d0[0]*d1[1]-d0[1]*d1[0], float(d0@d1))
        if abs(turn) < 1e-6:
            continue
        if abs(turn) > math.radians(170) or radius <= 0:
            raise ConstructionError("corner_requires_explicit_stop", {"waypoint": i})
        trim = min(radius * math.tan(abs(turn)/2), .45*l0, .45*l1)
        r = trim / math.tan(abs(turn)/2)
        before, after = b.copy(), b.copy()
        before[[0, 2]] -= trim*d0
        after[[0, 2]] += trim*d1
        line(at, before)
        sign = 1 if turn > 0 else -1
        center = before[[0, 2]] + sign*r*np.array([-d0[1], d0[0]])
        pieces.append(("arc", before.copy(), after.copy(), abs(turn)*r, center, turn, r))
        at = after
    line(at, points[-1])
    return pieces


def _sample_pieces(pieces, distances):
    ends = np.cumsum([p[3] for p in pieces])
    result = np.empty((len(distances), 3))
    heads = np.empty(len(distances))
    for i, piece in enumerate(pieces):
        begin = 0. if i == 0 else ends[i-1]
        select = (distances >= begin) & ((distances < ends[i]) if i < len(pieces)-1 else (distances <= ends[i]+1e-7))
        d = distances[select]-begin
        if piece[0] == "line":
            result[select] = piece[1] + d[:, None]/piece[3]*(piece[2]-piece[1])
            diff = piece[2]-piece[1]
            heads[select] = math.atan2(diff[2], diff[0])
        else:
            _, a, b, length, center, turn, radius = piece
            angle = np.sign(turn)*d/radius
            v = a[[0, 2]]-center
            xz = center + np.column_stack((np.cos(angle)*v[0]-np.sin(angle)*v[1],
                                            np.sin(angle)*v[0]+np.cos(angle)*v[1]))
            result[select] = np.column_stack((xz[:, 0], a[1]+(b[1]-a[1])*d/length, xz[:, 1]))
            tangent = np.sign(turn)*np.array([-v[1], v[0]])
            heads[select] = math.atan2(tangent[1], tangent[0])+angle
    return result, heads


def _distance_clock(times, length, speed, ramp_s=.3):
    if length < speed*ramp_s:
        raise ConstructionError("route_too_short_for_speed_and_ramps", {"length_m": length})
    duration = length/speed+ramp_s
    t = np.clip(times, 0, duration)
    u = np.minimum(t/ramp_s, 1.)
    begin = speed*ramp_s*(u**3-.5*u**4)
    middle = speed*(t-ramp_s/2)
    v = np.minimum((duration-t)/ramp_s, 1.)
    tail = length-speed*ramp_s*(v**3-.5*v**4)
    distances = np.where(t < ramp_s, begin, np.where(t > duration-ramp_s, tail, middle))
    return distances, duration


def static_actor(entity_id, root_m, body=BodySpec(), *, duration_s=30., fps=30., facing_rad=None):
    root = np.asarray(root_m, float)
    if root.shape != (3,) or not np.isfinite(root).all() or duration_s <= 0 or fps <= 0:
        raise ConstructionError("invalid_static_actor")
    n = round(duration_s*fps)+1
    if facing_rad is None:
        # Stable, diverse facing without perturbing any certified geometry later.
        import hashlib
        facing_rad = int.from_bytes(hashlib.sha256(entity_id.encode()).digest()[:4], "little") / 2**32 * 2*math.pi
    return ActorPlan(entity_id, np.repeat(root[None], n, axis=0), body,
                     np.full(n, facing_rad), {"route_length_m": 0., "walk_intervals_s": []})


def construct_route(context, waypoints_m, *, species="human", speed_m_s=1., start_s=1., dwell_s=None,
                    duration_s=30., fps=30., corner_radius_m=.8, camera=None, entity_id="target", body=None):
    lo, hi = SPEED_BANDS[species]
    if not lo <= speed_m_s <= hi:
        raise ConstructionError("species_speed_band", {"species": species, "allowed_m_s": [lo, hi]})
    if start_s < 0 or duration_s <= 0 or fps <= 0 or start_s > duration_s:
        raise ConstructionError("invalid_route_clock")
    points = np.asarray(waypoints_m, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ConstructionError("invalid_waypoints")
    dwell = np.zeros(len(points)) if dwell_s is None else np.asarray(dwell_s, float)
    if dwell.shape != (len(points),) or not np.isfinite(dwell).all() or np.any(dwell < 0):
        raise ConstructionError("invalid_dwell_times")
    times = np.arange(round(duration_s*fps)+1)/fps
    out = np.repeat(points[0:1], len(times), axis=0)
    headings = np.zeros(len(times))
    stops = [0] + [i for i in range(1, len(points)-1) if dwell[i] > 0] + [len(points)-1]
    current, total, intervals = start_s + dwell[0], 0., []
    for left, right in zip(stops[:-1], stops[1:]):
        pieces = _rounded_pieces(points[left:right+1], corner_radius_m)
        if not pieces:
            raise ConstructionError("zero_length_route")
        length = sum(p[3] for p in pieces)
        distances, walk_s = _distance_clock(times-current, length, speed_m_s)
        if current+walk_s+dwell[right] > duration_s+1e-9:
            raise ConstructionError("route_exceeds_parent_duration", {"end_s": current+walk_s+dwell[right]})
        position, heading = _sample_pieces(pieces, distances)
        use = times >= current
        out[use], headings[use] = position[use], heading[use]
        if left == 0:
            headings[~use] = heading[0]
        intervals.append([current, current+walk_s])
        current += walk_s+dwell[right]
        total += length
    out, validation = validate_track(context, out, species, camera=camera)
    return ActorPlan(entity_id, out, body or BodySpec.for_species(species), headings,
                     {"route_length_m": total, "walk_intervals_s": intervals, "speed_m_s": speed_m_s,
                      "ramp_s": .3, "waypoints_m": points.tolist(), "dwell_s": dwell.tolist(),
                      "validation": validation, "clock": {"fps": fps, "duration_s": duration_s}})
