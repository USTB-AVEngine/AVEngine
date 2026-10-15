"""Deterministic geometric layouts shared by visible/static/occlusion plugins."""
from __future__ import annotations
import math
import heapq
from itertools import islice
import numpy as np
from .types import ScenePlan
from .routes import ConstructionError, static_actor, validate_track
from .geometry import framing


def _separated(point, body, placed, bodies):
    for key, other in placed.items():
        ob = bodies[key]
        required = 1. if "human" in {body.species,ob.species} and {body.species,ob.species}<={"human","dog","cat"} else body.radius_m+ob.radius_m
        if np.linalg.norm((point-other)[[0, 2]]) < required+.02:
            return False
    if body.species == "human":
        people = [p for key,p in placed.items() if bodies[key].species == "human"]
        if len(people) >= 2:
            a,b = people[:2]
            axis = (b-a)[[0,2]]
            delta = (point-a)[[0,2]]
            if abs(axis[0]*delta[1]-axis[1]*delta[0])/max(np.linalg.norm(axis),1e-9) < .35:
                return False
    return True


def _rank_points(points, camera, preferred=3.8, limit=48):
    if not len(points):
        return points
    distances = np.linalg.norm((points-np.asarray(camera.position_m))[:, [0, 2]], axis=1)
    order = np.argsort(np.abs(distances-preferred), kind="stable")
    # A small deterministic spatial beam, not thousands of near-identical points.
    chosen = []
    for i in order:
        p = points[i]
        if all(np.linalg.norm((p-q)[[0, 2]]) >= .3 for q in chosen):
            chosen.append(p)
        if len(chosen) >= limit:
            break
    return np.array(chosen)


def enumerate_static_layouts(context, bodies, roles, *, occlusion_pair=None, max_candidates=256, floor_clearance_m=.002):
    """Solve requested role geometry, then let the family apply original predicates.

    A pair's overlap is an explicit target, solved by depth ratio and lateral
    offset. All remaining roles are placed in disjoint feasible domains.
    Proxies propose occlusion; exact native identity/pixel preflight is separate.
    """
    if set(bodies) != set(roles) or any(v not in {"inside", "outside", "any"} for v in roles.values()):
        raise ValueError("one valid framing role per body required")
    if not math.isfinite(floor_clearance_m) or not 0<=floor_clearance_m<=.014:
        raise ValueError("planned floor clearance must stay within original 15mm support tolerance")
    if not bodies or max_candidates < 1:
        return
    candidate_points=context.points_m.copy()
    candidate_points[:,1]+=floor_clearance_m
    if occlusion_pair is not None:
        target, foreground, lower, upper = occlusion_pair
        if target == foreground or not {target, foreground} <= set(bodies) or not 0 <= lower <= upper <= 1:
            raise ValueError("invalid requested occlusion pair")
    # Filter actual NavMesh XZ once per room point, before the product of roles.
    # An invalid fixed speaker point must not consume 256 otherwise valid layouts.
    nav = np.ones(len(context.points_m),bool)
    if context.pathfinder is not None and len(context.points_m):
        snapped = np.asarray([context.pathfinder.snap_point(p)for p in context.points_m],float)
        nav = np.isfinite(snapped).all(axis=1)
        nav &= np.linalg.norm((snapped-context.points_m)[:,[0,2]],axis=1) <= .002
    yielded = 0
    for camera in context.cameras:
        domains = {}
        for key, body in bodies.items():
            grid = context.grid(camera, body)
            allowed = context.walkable_map.contains_many(context.points_m[:, [0, 2]], body.species) & nav
            from avengine.rooms.walkable_map import SPECIES_RADII_M
            extra = max(0.,body.radius_m-SPECIES_RADII_M[body.species])
            if extra:
                for angle in np.arange(16)*2*math.pi/16:
                    allowed &= context.walkable_map.contains_many(context.points_m[:,[0,2]]+
                                extra*np.array([math.cos(angle),math.sin(angle)]),body.species)
            allowed &= grid.distances_m >= (1.5 if body.species == "human" else .5) + .02
            frame_states = (np.asarray([framing(camera,p,body)[0]for p in candidate_points])
                            if floor_clearance_m else grid.frustum_state)
            if roles[key] != "any":
                allowed &= frame_states == roles[key]
            else:
                allowed &= frame_states != "edge"
            if roles[key] == "inside":
                # Ranking filter only: missing rays are retained for explicit unknown.
                allowed &= ~np.isfinite(grid.visible_fraction) | (grid.visible_fraction > .25)
            domains[key] = _rank_points(candidate_points[allowed], camera)
        if any(not len(points) for points in domains.values()):
            continue

        def fill(placed):
            remaining = sorted((key for key in bodies if key not in placed),key=lambda k:(len(domains[k]),k))
            if not remaining:
                yield placed
                return
            initial = (0,)*len(remaining)
            heap,seen = [(0,initial)],{initial}
            expanded = 0
            while heap and expanded < max(1024,max_candidates*16):
                _,indices = heapq.heappop(heap);expanded += 1
                result = dict(placed);valid = True
                for key,index in zip(remaining,indices):
                    point = domains[key][index]
                    if not _separated(point,bodies[key],result,bodies):
                        valid = False;break
                    result[key] = point
                if valid:
                    yield result
                for dimension,key in enumerate(remaining):
                    if indices[dimension]+1 >= min(24,len(domains[key])):
                        continue
                    adjacent = list(indices);adjacent[dimension] += 1;adjacent = tuple(adjacent)
                    if adjacent not in seen:
                        seen.add(adjacent);heapq.heappush(heap,(sum(adjacent),adjacent))

        if occlusion_pair is None:
            seeds = ({},)
        else:
            def pair_seeds():
                tb, fb = bodies[target], bodies[foreground]
                for tp in domains[target][:32]:
                    delta = tp-np.asarray(camera.position_m)
                    depth = float(delta@camera.forward)
                    for ratio in (.55, .65, .75):
                        # Lateral overlap candidates derived from angular widths.
                        front_radius_at_target = fb.radius_m/ratio
                        wanted = (lower+upper)/2
                        shift = max(0., front_radius_at_target+tb.radius_m-2*tb.radius_m*wanted)*ratio
                        for lateral in (0., -shift, shift):
                            fp = np.asarray(camera.position_m)+ratio*delta+lateral*camera.right
                            fp[1] = tp[1]
                            if context.walkable_map.floor_y is not None:
                                cell = np.floor((fp[[0,2]]-context.walkable_map.origin)/context.walkable_map.resolution).astype(int)
                                if 0<=cell[0]<context.walkable_map.shape[1] and 0<=cell[1]<context.walkable_map.shape[0]:
                                    y=context.walkable_map.floor_y[cell[1],cell[0]]
                                    if math.isfinite(y):fp[1]=y+floor_clearance_m
                            if not context.walkable_map.contains(fp[[0, 2]], fb.species):
                                continue
                            if np.linalg.norm((fp-np.asarray(camera.position_m))[[0, 2]]) < (1.5 if fb.species == "human" else .5)+.02:
                                continue
                            state, _ = framing(camera, fp, fb)
                            if state == "edge" or (roles[foreground] != "any" and state != roles[foreground]):
                                continue
                            if not _separated(fp, fb, {target: tp}, bodies):
                                continue
                            try:
                                validate_track(context,[fp],fb.species,camera=camera,snap=False)
                            except ConstructionError:
                                continue
                            prediction = context.predict(camera, tp, tb, occluders=((foreground, fp, fb),))
                            if prediction.state == "unknown" or not lower <= prediction.occluded_fraction <= upper:
                                continue
                            yield {target: tp, foreground: fp}
            seeds = pair_seeds()
        # Round-robin distinct geometric seeds, then use a fair product beam
        # within each seed. A failing first target cannot monopolize the budget.
        active = [iter(fill(seed))for seed in islice(seeds,32)]
        while active:
            remaining = []
            for iterator in active:
                try:
                    positions = next(iterator)
                except StopIteration:
                    continue
                remaining.append(iterator)
                actors = {key: static_actor(key, point, bodies[key]) for key, point in positions.items()}
                for actor in actors.values():
                    actor.metadata["planned_floor_clearance_m"]=floor_clearance_m
                yield ScenePlan(camera, actors, metadata={"construction": "shared_static_role_domains",
                    "roles": dict(roles), "occlusion_pair": occlusion_pair, "native_pixel_preflight_required": True})
                yielded += 1
                if yielded >= max_candidates:
                    return
            active = remaining
