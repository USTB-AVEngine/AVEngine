"""Joint static placement and continuous camera yaw over untruncated room domains.

This is a proposal search. Native masks, original physical/acoustic checks and
the family's answer predicate remain the certificate authorities.
"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
import math
import time
import numpy as np
from .types import CameraPose, ScenePlan
from .routes import static_actor
from .layouts import _separated


def enumerate_joint_static_layouts(context, bodies, roles, *, point_masks=None,
        distance_bands_m=None, minimum_distance_gaps=(), distance_offsets_m=None,
        framing_bodies=None, camera_heights_m=None, max_candidates=256,
        budget_s=120., max_expansions=None, diagnostics=None):
    """Assign all actors and a fixed camera together, intersecting yaw domains.

    roles has inside/outside/any for every actor. point_masks are over
    context.points_m. No nearest-24 point beam or fixed camera/yaw table is used.
    Explicit distance bands and (actor_a, actor_b, minimum_absolute_gap_m)
    constraints use horizontal distances from root + distance_offsets_m[actor].
    Supply rotated mouth offsets when the answer uses actual emitter distance.
    Body-to-camera owner clearance always uses the body root separately.

    framing_bodies can enclose multiple visual variants at an endpoint. Geometry
    remains proposal evidence; each native visual must be independently checked.
    max_candidates/budget_s/max_expansions bound computation, never establish
    continuous-space infeasibility. diagnostics is populated in place.
    """
    from avengine.rooms.walkable_map import SPECIES_RADII_M
    from .camera_solve import (PointRole, _positions, _point_intervals, _intersect,
                              _circular_components, check_camera_roles)
    if set(bodies) != set(roles) or any(r not in {"inside", "outside", "any"} for r in roles.values()):
        raise ValueError("one inside/outside/any role per body required")
    if max_candidates < 0 or not math.isfinite(budget_s) or budget_s < 0:
        raise ValueError("invalid search budget")
    if max_expansions is not None and max_expansions < 0:
        raise ValueError("invalid expansion budget")
    band_input = dict(distance_bands_m or {})
    offset_input = dict(distance_offsets_m or {})
    envelope_input = dict(framing_bodies or {})
    mask_input = dict(point_masks or {})
    if any(set(d)-set(bodies) for d in (band_input, offset_input, envelope_input, mask_input)):
        raise ValueError("constraint refers to an unknown actor")
    bands, offsets, envelopes, masks = {}, {}, {}, {}
    points = np.asarray(context.points_m, float)
    for key, body in bodies.items():
        lo, hi = band_input.get(key, (0., math.inf))
        if not math.isfinite(lo) or math.isnan(hi) or not 0 <= lo <= hi:
            raise ValueError("invalid requested distance band")
        bands[key] = float(lo), float(hi)
        offset = np.asarray(offset_input.get(key, [0., 0., 0.]), float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("invalid reference-point offset")
        offsets[key] = offset
        envelopes[key] = envelope_input.get(key, body)
        mask = np.asarray(mask_input.get(key, np.ones(len(points), bool)))
        if mask.shape != (len(points),) or mask.dtype != np.dtype(bool):
            raise ValueError("point_masks must be boolean arrays over context.points_m")
        masks[key] = mask.copy()
    gaps = []
    for a, b, minimum in minimum_distance_gaps:
        if a == b or not {a, b} <= set(bodies) or not math.isfinite(minimum) or minimum < 0:
            raise ValueError("invalid requested distance gap")
        gaps.append((a, b, float(minimum)))
    heights = tuple(context.camera_heights_m if camera_heights_m is None else camera_heights_m)
    if not heights or any(not math.isfinite(h) or h <= .05 for h in heights):
        raise ValueError("positive camera heights required")
    diag = diagnostics if diagnostics is not None else {}
    diag.update(status="searching", domain_points=len(points), camera_sites=0,
                camera_height_pairs=0, actor_domain_sizes={}, candidate_partial_assignments=0,
                yaw_intersection_rejections=0, spacing_rejections=0, distance_gap_rejections=0,
                returned_candidates=0, continuous_room_infeasibility_proved=False,
                native_pixel_truth=False)
    started = time.monotonic()

    def stopped():
        if time.monotonic()-started >= budget_s:
            diag["status"] = "budget_exhausted"
            return True
        if max_expansions is not None and diag["candidate_partial_assignments"] >= max_expansions:
            diag["status"] = "expansion_budget_exhausted"
            return True
        return False

    if not bodies or max_candidates == 0:
        diag["status"] = "empty_request" if not bodies else "candidate_limit"
        return
    if stopped():
        return
    # Map and original NavMesh feasibility are camera-independent, computed once.
    nav = np.ones(len(points), bool)
    if context.pathfinder is not None and len(points):
        snap = np.asarray([context.pathfinder.snap_point(p) for p in points], float)
        nav = np.isfinite(snap).all(axis=1)
        nav &= np.linalg.norm((snap-points)[:, [0, 2]], axis=1) <= .002
    domains = {}
    for key, body in bodies.items():
        allowed = masks[key] & nav & context.walkable_map.contains_many(points[:, [0, 2]], body.species)
        extra = max(0., body.radius_m-SPECIES_RADII_M[body.species])
        if extra:
            for angle in np.arange(16)*2*math.pi/16:
                allowed &= context.walkable_map.contains_many(points[:, [0, 2]] +
                    extra*np.array([math.cos(angle), math.sin(angle)]), body.species)
        domains[key] = np.flatnonzero(allowed)
        diag["actor_domain_sizes"][key] = len(domains[key])
    if any(not len(d) for d in domains.values()):
        diag["status"] = "selected_point_domain_exhausted"
        return
    sites, position_rejections = _positions(context)
    diag["camera_sites"] = len(sites)
    diag["camera_position_rejections"] = position_rejections
    # Ranking retains every site. Boundary sites usually leave more room in front.
    centre = points[:, [0, 2]].mean(axis=0)
    sites = sorted(sites, key=lambda row: (-float(np.linalg.norm(row[0][[0, 2]]-centre)),
                                         tuple(row[0])))
    template = CameraPose((0., 0., 0.), 0., width=context.camera_width, height=context.camera_height)
    role_kind = {"inside": "inside_full_body", "outside": "outside_full_body", "any": "rest_full_body"}

    def width(intervals):
        return max((b-a for a, b in _circular_components(intervals)), default=-math.inf)

    def site_plans(ground, clearance, inward, height):
        if stopped():
            return
        position = ground + [0., height, 0.]
        diag["camera_height_pairs"] += 1
        local, references, role_records = {}, {}, {}
        for key, body in bodies.items():
            rows = []
            references[key] = {}
            role_records[key] = {}
            ids = domains[key]
            root_distance = np.linalg.norm((points[ids]-position)[:, [0, 2]], axis=1)
            reference_distance = np.linalg.norm((points[ids]+offsets[key]-position)[:, [0, 2]], axis=1)
            lo, hi = bands[key]
            allowed = ((root_distance >= (1.5 if body.species == "human" else .5)) &
                       (reference_distance >= lo) & (reference_distance <= hi))
            for index, distance in zip(ids[allowed], reference_distance[allowed]):
                if stopped():
                    return
                role = PointRole(role_kind[roles[key]], tuple(points[index]), envelopes[key], name=key)
                intervals = _intersect(inward, _point_intervals(role, position, template))
                if intervals:
                    rows.append((int(index), intervals))
                    references[key][int(index)] = float(distance)
                    role_records[key][int(index)] = role
            local[key] = rows
            if not rows:
                return
        order = sorted(bodies, key=lambda key: (len(local[key]), key))

        def assign(depth, placed, indices, feasible):
            if stopped():
                return
            if depth == len(order):
                for lo, hi in sorted(_circular_components(feasible), key=lambda v: -(v[1]-v[0])):
                    yaw = (lo+hi)/2
                    camera = replace(template, position_m=tuple(map(float, position)),
                        yaw_deg=float((math.degrees(yaw)+180)%360-180),
                        camera_id="joint_static_%06d" % diag["returned_candidates"])
                    check = check_camera_roles(context, camera,
                        [role_records[k][indices[k]] for k in bodies])
                    if not check["passed"]:
                        continue
                    actors = {k: static_actor(k, placed[k], bodies[k]) for k in bodies}
                    yield ScenePlan(camera, actors, metadata={
                        "construction": "joint_static_points_and_continuous_camera_yaw",
                        "roles": dict(roles), "selected_point_indices": dict(indices),
                        "camera_yaw_margin_deg": math.degrees((hi-lo)/2),
                        "camera_wall_clearance_m": float(clearance),
                        "reference_distances_m": {k: references[k][indices[k]] for k in bodies},
                        "requested_distance_gaps": [list(v) for v in gaps],
                        "camera_role_check": check,
                        "native_pixel_preflight_required": True})
                return
            key = order[depth]
            choices = []
            for index, intervals in local[key]:
                if stopped():
                    return
                diag["candidate_partial_assignments"] += 1
                point = points[index]
                if not _separated(point, bodies[key], placed, bodies):
                    diag["spacing_rejections"] += 1
                    continue
                rejected = False
                for a, b, minimum in gaps:
                    other = b if key == a else a if key == b else None
                    if other in indices and abs(references[key][index]-references[other][indices[other]]) < minimum:
                        rejected = True
                        break
                if rejected:
                    diag["distance_gap_rejections"] += 1
                    continue
                joint = _intersect(feasible, intervals)
                if not joint:
                    diag["yaw_intersection_rejections"] += 1
                    continue
                choices.append((-width(joint), index, joint))
            for _, index, joint in sorted(choices, key=lambda row: (row[0], row[1])):
                yield from assign(depth+1, {**placed, key: points[index]},
                                  {**indices, key: index}, joint)

        yield from assign(0, {}, {}, inward)

    active = deque(site_plans(ground, clearance, inward, h)
                   for ground, clearance, inward in sites for h in heights)
    try:
        while active and not stopped():
            iterator = active.popleft()
            candidate = next(iterator, None)
            if candidate is None:
                continue
            active.append(iterator)
            diag["returned_candidates"] += 1
            yield candidate
            if diag["returned_candidates"] >= max_candidates:
                diag["status"] = "candidate_limit"
                return
        if diag["status"] == "searching":
            diag["status"] = "selected_joint_domain_exhausted"
    finally:
        diag["elapsed_s"] = time.monotonic()-started
