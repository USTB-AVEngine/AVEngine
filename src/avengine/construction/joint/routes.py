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


def validate_track(context, positions_m, species="human", *, camera=None, snap=True, allow_map_fallback=None):
    p = np.asarray(positions_m, float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ConstructionError("invalid_route_points")
    m = context.walkable_map
    misses=[]
    if not m.contains_many(p[:,[0,2]],species).all():
        misses.append("walkable_map_cells")
    segments=np.unique(np.concatenate((p[:-1,[0,2]],p[1:,[0,2]]),axis=1),axis=0)
    for segment in segments:
        if not m.segment_free(segment[:2],segment[2:],species):
            misses.append("walkable_map_segment")
            break
    fallback=None
    allow = getattr(context,"allow_native_map_fallback",False) if allow_map_fallback is None else bool(allow_map_fallback)
    if misses:
        if not allow:
            raise ConstructionError(misses[0],{"map_failures":misses})
        fallback=_native_map_fallback(context,p,species)
        fallback["map_failures"]=misses
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
            if fallback is None and not m.contains_many(q[:, [0, 2]], species).all():
                raise ConstructionError("snapped_route_outside_map")
            for seg in np.unique(np.concatenate((q[:-1, [0, 2]], q[1:, [0, 2]]), axis=1), axis=0):
                if fallback is None and not m.segment_free(seg[:2], seg[2:], species):
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
               "vertical_authority": "physical_scan_floor_not_navmesh_height", "native_map_fallback": fallback}


def _native_map_fallback(context, positions, species):
    """A proposal screen only; final native full-pose physics is still pending."""
    from avengine.rooms.walkable_map import SPECIES_RADII_M
    pf=context.pathfinder
    ray=getattr(context,"floor_raycaster",getattr(context,"raycaster",None))
    if pf is None or not hasattr(pf,"distance_to_closest_obstacle"):
        raise ConstructionError("map_fallback_requires_original_navmesh_clearance")
    if ray is None or getattr(ray,"source",None)!="visual_mesh":
        raise ConstructionError("map_fallback_requires_original_visual_floor")
    # Check continuous subsegments at <=2.5cm, not only the 30Hz endpoints.
    counts=np.ceil(np.linalg.norm(np.diff(positions[:,[0,2]],axis=0),axis=1)/.025)
    if counts.sum()+len(positions)>100000:
        raise ConstructionError("map_fallback_continuous_sampling_budget")
    samples=[positions]
    for a,b in zip(positions[:-1],positions[1:]):
        n=math.ceil(float(np.linalg.norm((b-a)[[0,2]]))/.025)
        if n>1:
            samples.append(a+np.arange(1,n)[:,None]/n*(b-a))
    points=np.unique(np.concatenate(samples),axis=0)
    snapped=np.asarray([pf.snap_point(v)for v in points],float)
    if snapped.shape!=points.shape or not np.isfinite(snapped).all():
        raise ConstructionError("map_fallback_navmesh_nonfinite")
    error=float(np.linalg.norm((snapped-points)[:,[0,2]],axis=1).max())
    if error>.002:
        raise ConstructionError("map_fallback_navmesh_horizontal",{"maximum_error_m":error})
    clearance=np.asarray([pf.distance_to_closest_obstacle(q,1.)for q in snapped],float)
    clearance=np.minimum(clearance,1.)
    required=SPECIES_RADII_M[species]
    if not np.isfinite(clearance).all() or clearance.min()<required-1e-8:
        raise ConstructionError("map_fallback_body_clearance",{"minimum_m":float(clearance.min()),"required_radius_m":required})
    distance,_=ray.first_hit(points+[0,.2,0],np.tile([0.,-1.,0.],(len(points),1)))
    floor_y=points[:,1]+.2-distance
    error_y=np.abs(points[:,1]-floor_y)
    if not np.isfinite(distance).all() or np.any(distance>.85) or np.any(error_y>.03+1e-7):
        raise ConstructionError("map_fallback_physical_floor",{"maximum_root_floor_gap_m":float(error_y.max())})
    return {"status":"pending_native_sweep","continuous_samples":len(points),
            "maximum_navmesh_horizontal_error_m":error,"minimum_native_clearance_m":float(clearance.min()),
            "body_radius_m":required,"maximum_original_scan_floor_gap_m":float(error_y.max()),
            "original_walkable_map_modified":False,"full_native_pose_and_final_scene_checks_required":True}


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



def _framed_route(actor, camera, required_framing):
    if required_framing is None:
        return actor
    if camera is None:
        raise ConstructionError("route_framing_requires_camera")
    from dataclasses import asdict
    from .certification import track_framing_check
    check=track_framing_check(camera,actor,required=required_framing)
    actor.metadata["framing_constraint"]=asdict(check)
    if check.status!="pass":
        raise ConstructionError("route_framing_constraint",asdict(check))
    return actor


def construct_route(context, waypoints_m, *, species="human", speed_m_s=1., start_s=1., dwell_s=None,
                    duration_s=30., fps=30., corner_radius_m=.8, camera=None, entity_id="target", body=None,
                    required_framing=None):
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
    actor = ActorPlan(entity_id, out, body or BodySpec.for_species(species), headings,
                     {"route_length_m": total, "walk_intervals_s": intervals, "speed_m_s": speed_m_s,
                      "ramp_s": .3, "waypoints_m": points.tolist(), "dwell_s": dwell.tolist(),
                      "validation": validation, "clock": {"fps": fps, "duration_s": duration_s}})
    return _framed_route(actor,camera,required_framing)


def bind_distance_clock(context, distance_m, *, origin_m, heading_rad, species="human",
                        corner=None, camera=None, entity_id="target", body=None, fps=30., ramp_s=.5,
                        required_framing=None):
    """Bind an already admitted cumulative clock to one straight/rounded path.

    corner=(start_distance_m, radius_m, signed_turn_rad), as in the existing
    native straight_route parameterization. No time/distance rescaling or native
    replanning occurs. Call certify_plan for the full multi-entity obligations.
    """
    from avengine.assets.roster_motion.straight_route import _shape_xy
    distance=np.asarray(distance_m,float)
    origin=np.asarray(origin_m,float)
    if (distance.ndim!=1 or len(distance)<2 or not np.isfinite(distance).all()
            or distance[0]<-1e-9 or np.any(np.diff(distance)<-1e-9)
            or origin.shape!=(3,) or not np.isfinite(origin).all() or not math.isfinite(heading_rad)
            or fps<=0 or not 0<=ramp_s<=.5):
        raise ConstructionError("invalid_fixed_distance_clock")
    if corner is not None:
        sc,radius,turn=corner
        if not np.isfinite(corner).all() or sc<0 or radius<=0:
            raise ConstructionError("invalid_fixed_clock_corner")
    xz,headings=_shape_xy(distance,origin[[0,2]],heading_rad,corner)
    positions=np.column_stack((xz[:,0],np.full(len(xz),origin[1]),xz[:,1]))
    positions,validation=validate_track(context,positions,species,camera=camera,snap=False)
    active=np.diff(distance)>1e-8
    edges=np.diff(np.r_[False,active,False].astype(int))
    intervals=[[float(a/fps),float(b/fps)]for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    actor = ActorPlan(entity_id,positions,body or BodySpec.for_species(species),headings,
        {"route_length_m":float(distance[-1]-distance[0]),"walk_intervals_s":intervals,
         "ramp_s":ramp_s,"distance_clock_bound":True,"distance_clock_m":distance.tolist(),
         "fixed_clock_geometry":{"origin_m":origin.tolist(),"heading_rad":heading_rad,
                                 "corner":None if corner is None else list(corner)},
         "validation":validation,"clock":{"fps":fps,"duration_s":(len(distance)-1)/fps}})
    return _framed_route(actor,camera,required_framing)


def straight_then_arc(context, shadow_root_m, tangent_xz, distance_m, shadow_distance_m, *,
                      turn_radius_m=.8, turn_sign=1, species="human", camera=None,
                      entity_id="target", body=None, fps=30., ramp_s=.5):
    """Same tangent before/after a specified stop anchor; compact return arc after it."""
    direction=np.asarray(tangent_xz,float)
    shadow=np.asarray(shadow_root_m,float)
    if (direction.shape!=(2,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-9
            or shadow.shape!=(3,) or not np.isfinite(shadow).all() or turn_radius_m<=0
            or shadow_distance_m<0 or turn_sign not in {-1,1}):
        raise ConstructionError("invalid_straight_then_arc")
    direction=direction/np.linalg.norm(direction)
    origin=shadow.copy();origin[[0,2]]-=shadow_distance_m*direction
    distance=np.asarray(distance_m,float)
    if distance.ndim!=1 or len(distance)<2 or not np.isfinite(distance).all():
        raise ConstructionError("invalid_fixed_distance_clock")
    arc_length=max(0.,float(distance[-1])-shadow_distance_m)
    turn=turn_sign*arc_length/turn_radius_m
    result=bind_distance_clock(context,distance,origin_m=origin,
        heading_rad=math.atan2(direction[1],direction[0]),species=species,
        corner=(shadow_distance_m,turn_radius_m,turn),camera=camera,entity_id=entity_id,
        body=body,fps=fps,ramp_s=ramp_s)
    result.metadata["shadow_distance_m"]=float(shadow_distance_m)
    return result


def repeat_bouts(actor, offsets_s=(0.,10.), *, fps=30., duration_s=30., source_window_s=(0.,10.),
                 context=None, camera=None, turn_duration_s=1.):
    """Repeat one out-and-back route on the SAME geometric path and source clock.

    Return geometry reverses arc progress, not time, so sounding/moving windows
    keep their temporal placement. Paused turns are explicit; after the last
    requested bout the actor remains at that endpoint (no automatic third walk).
    """
    from copy import deepcopy
    p=np.asarray(actor.positions_m,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or fps<=0 or duration_s<=0:
        raise ConstructionError("invalid_repeat_prototype")
    a,b=map(float,source_window_s);offsets=tuple(map(float,offsets_s))
    if (not offsets or not 0<=a<b or b>(len(p)-1)/fps+1e-9 or turn_duration_s<=0
            or any(x<0 or x+b-a>duration_s+1e-9 for x in offsets)
            or any(y<x+b-a-1e-9 for x,y in zip(offsets[:-1],offsets[1:]))):
        raise ConstructionError("invalid_repeat_windows")
    i,j=round(a*fps),round(b*fps)
    segment=p[i:j+1]
    length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(segment[:,[0,2]],axis=0),axis=1))]
    if length[-1]<=1e-8:
        raise ConstructionError("static_prototype_has_no_bout")
    moving=np.diff(length)>1e-8
    begin=int(np.flatnonzero(moving)[0]);end=int(np.flatnonzero(moving)[-1])+1
    unique,index=np.unique(length,return_index=True)
    path=segment[index]
    tangent=np.diff(path[:,[0,2]],axis=0)
    path_heading=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
    path_heading=np.r_[path_heading,path_heading[-1]]
    if actor.headings_rad is not None:
        declared=np.asarray(actor.headings_rad,float)[i:j+1]
        if declared.shape!=(len(segment),) or not np.isfinite(declared).all():
            raise ConstructionError("invalid_repeat_headings")
        path_heading=np.unwrap(declared[index])
    n=round(duration_s*fps)+1;times=np.arange(n)/fps
    positions=np.repeat(segment[:1],n,axis=0);headings=np.full(n,path_heading[0])
    intervals=[];turns=[];bouts=[];previous_end=None;previous_heading=path_heading[0]
    for k,offset in enumerate(offsets):
        reverse=bool(k%2)
        progress=length[-1]-length if reverse else length
        trajectory=np.column_stack([np.interp(progress,unique,path[:,axis])for axis in range(3)])
        heading=np.interp(progress,unique,path_heading)+(math.pi if reverse else 0.)
        start=round(offset*fps);finish=start+len(segment)
        if not np.allclose(positions[start],trajectory[0],atol=1e-8):
            raise ConstructionError("repeat_endpoint_discontinuity")
        positions[start:finish]=trajectory;positions[finish:]=trajectory[-1]
        headings[start:finish]=heading;headings[finish:]=heading[-1]
        move_begin=offset+begin/fps;move_end=offset+end/fps
        if previous_end is not None:
            if move_begin-previous_end < turn_duration_s+1/fps:
                raise ConstructionError("insufficient_stationary_turn_time")
            t0,t1=move_begin-turn_duration_s-1/fps,move_begin-1/fps
            difference=(heading[0]-previous_heading+math.pi)%(2*math.pi)-math.pi
            # Use the nearest equivalent yaw to avoid a numeric 2*pi jump.
            headings[start:]-=heading[0]-(previous_heading+difference)
            before=(times>=previous_end)&(times<t0);headings[before]=previous_heading
            turn=(times>=t0)&(times<=t1)
            u=(times[turn]-t0)/(t1-t0);headings[turn]=previous_heading+difference*(u*u*(3-2*u))
            hold=(times>t1)&(times<move_begin);headings[hold]=previous_heading+difference
            turns.append([t0,t1])
        active=np.diff(length)>1e-8;edges=np.diff(np.r_[False,active,False].astype(int))
        intervals.extend([[offset+x/fps,offset+y/fps]for x,y in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))])
        bouts.append({"offset_s":offset,"reverse":reverse,"movement_s":[move_begin,move_end]})
        previous_end=move_end;previous_heading=float(headings[min(n-1,round(move_end*fps))])
    metadata=deepcopy(actor.metadata)
    metadata.update(walk_intervals_s=intervals,stationary_turn_intervals_s=turns,bouts=bouts,
                    route_length_m=float(length[-1]*len(offsets)),
                    clock={"fps":fps,"duration_s":duration_s},repeat_geometry="same_path_reverse_arc_progress")
    if "fixed_clock_geometry" in metadata:
        metadata["prototype_fixed_clock_geometry"]=metadata.pop("fixed_clock_geometry")
    metadata["distance_clock_m"]=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions[:,[0,2]],axis=0),axis=1))].tolist()
    if context is not None:
        positions,metadata["validation"]=validate_track(context,positions,actor.body.species,camera=camera,snap=False)
    return ActorPlan(actor.entity_id,positions,actor.body,headings,metadata)


def out_and_back_shape(origin_m, tangent_xz, distance_m, turn_distance_m, *,
                       turn_interval_s=(4.8,5.8), turn_sign=1, fps=30.):
    """Return positions and tangent headings for an unchanged out-and-back clock.

    This pure shape has no room, NavMesh, floor, gait or native certification.
    It can key canonical pose caches; actual routes must still use
    out_and_back/validate_track and the original full native pose checks.
    """
    distance=np.asarray(distance_m,float);origin=np.asarray(origin_m,float);direction=np.asarray(tangent_xz,float)
    if (distance.ndim!=1 or len(distance)<2 or not np.isfinite(distance).all()or np.any(np.diff(distance)<-1e-9)
            or not math.isfinite(fps) or fps<=0 or not math.isfinite(turn_distance_m)
            or distance[0]<-1e-9 or turn_distance_m<=0 or distance[-1]>2*turn_distance_m+1e-7
            or origin.shape!=(3,)or not np.isfinite(origin).all()or direction.shape!=(2,)
            or not np.isfinite(direction).all()or np.linalg.norm(direction)<1e-9 or turn_sign not in {-1,1}):
        raise ConstructionError("invalid_out_and_back_clock")
    direction=direction/np.linalg.norm(direction);times=np.arange(len(distance))/fps
    a,b=map(float,turn_interval_s)
    if not 0<=a<b<=times[-1]:raise ConstructionError("invalid_stationary_turn_interval")
    turn=(times>=a-1/fps)&(times<=b+1/fps)
    if not turn.any()or not np.all(np.abs(distance[turn]-turn_distance_m)<=1e-7):
        raise ConstructionError("turn_requires_stationary_shadow_anchor")
    if np.any((distance>turn_distance_m+1e-7)&(times<b)):
        raise ConstructionError("return_walk_precedes_completed_turn")
    progress=np.where(distance<=turn_distance_m,distance,2*turn_distance_m-distance)
    positions=np.repeat(origin[None],len(distance),axis=0);positions[:,[0,2]]+=progress[:,None]*direction
    theta=math.atan2(direction[1],direction[0]);u=np.clip((times-a)/(b-a),0,1)
    headings=theta+turn_sign*math.pi*(u*u*(3-2*u))
    return positions,headings


def out_and_back(context, origin_m, tangent_xz, distance_m, turn_distance_m, *,
                 turn_interval_s=(4.8,5.8), turn_sign=1, species="human", camera=None,
                 entity_id="target", body=None, fps=30., ramp_s=.5):
    """Bind a supplied two-bout clock to the same line, with an explicit paused turn.

    Global clock 0..L..2L maps to spatial progress 0..L..0. The clock is never
    scaled. A native full-pose check must still verify the stationary 180deg turn.
    """
    if not math.isfinite(ramp_s) or not 0<=ramp_s<=.5:
        raise ConstructionError("invalid_out_and_back_ramp")
    positions,headings=out_and_back_shape(origin_m,tangent_xz,distance_m,turn_distance_m,
        turn_interval_s=turn_interval_s,turn_sign=turn_sign,fps=fps)
    distance=np.asarray(distance_m,float)
    times=np.arange(len(distance))/fps
    a,b=map(float,turn_interval_s)
    positions,validation=validate_track(context,positions,species,camera=camera,snap=False)
    moving=np.diff(distance)>1e-8;edges=np.diff(np.r_[False,moving,False].astype(int))
    intervals=[[x/fps,y/fps]for x,y in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    return ActorPlan(entity_id,positions,body or BodySpec.for_species(species),headings,
        {"distance_clock_bound":True,"distance_clock_m":distance.tolist(),"route_length_m":float(distance[-1]-distance[0]),
         "walk_intervals_s":intervals,"ramp_s":ramp_s,"stationary_turn_intervals_s":[[a,b]],
         "turnaround_distance_m":turn_distance_m,"validation":validation,
         "native_stationary_turn_physics_required":True,"clock":{"fps":fps,"duration_s":float(times[-1])}})
