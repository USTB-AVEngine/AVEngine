"""Deterministic geometric layouts shared by visible/static/occlusion plugins."""
from __future__ import annotations
import math
import numpy as np
from .types import ScenePlan
from .routes import static_actor
from .geometry import framing


def _separated(point, body, placed, bodies):
    for key, other in placed.items():
        ob = bodies[key]
        required = 1. if body.species == ob.species == "human" else body.radius_m+ob.radius_m
        if np.linalg.norm((point-other)[[0, 2]]) < required+.02:
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


def enumerate_static_layouts(context, bodies, roles, *, occlusion_pair=None, max_candidates=256):
    """Solve requested role geometry, then let the family apply original predicates.

    A pair's overlap is an explicit target, solved by depth ratio and lateral
    offset. All remaining roles are placed in disjoint feasible domains.
    Proxies propose occlusion; exact native identity/pixel preflight is separate.
    """
    if set(bodies) != set(roles) or any(v not in {"inside", "outside", "any"} for v in roles.values()):
        raise ValueError("one valid framing role per body required")
    if not bodies or max_candidates < 1:
        return
    if occlusion_pair is not None:
        target, foreground, lower, upper = occlusion_pair
        if target == foreground or not {target, foreground} <= set(bodies) or not 0 <= lower <= upper <= 1:
            raise ValueError("invalid requested occlusion pair")
    yielded = 0
    for camera in context.cameras:
        domains = {}
        for key, body in bodies.items():
            grid = context.grid(camera, body)
            allowed = context.walkable_map.contains_many(context.points_m[:, [0, 2]], body.species)
            allowed &= grid.distances_m >= (1.5 if body.species == "human" else .5) + .02
            if roles[key] != "any":
                allowed &= grid.frustum_state == roles[key]
            else:
                allowed &= grid.frustum_state != "edge"
            if roles[key] == "inside":
                # Ranking filter only: missing rays are retained for explicit unknown.
                allowed &= ~np.isfinite(grid.visible_fraction) | (grid.visible_fraction > .25)
            domains[key] = _rank_points(context.points_m[allowed], camera)
        if any(not len(points) for points in domains.values()):
            continue

        def fill(placed):
            remaining = [key for key in bodies if key not in placed]
            if not remaining:
                yield placed
                return
            key = min(remaining, key=lambda k: len(domains[k]))
            for point in domains[key][:24]:
                if _separated(point, bodies[key], placed, bodies):
                    yield from fill({**placed, key: point})

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
                            if not context.walkable_map.contains(fp[[0, 2]], fb.species):
                                continue
                            if np.linalg.norm((fp-np.asarray(camera.position_m))[[0, 2]]) < (1.5 if fb.species == "human" else .5)+.02:
                                continue
                            state, _ = framing(camera, fp, fb)
                            if state == "edge" or (roles[foreground] != "any" and state != roles[foreground]):
                                continue
                            if not _separated(fp, fb, {target: tp}, bodies):
                                continue
                            prediction = context.predict(camera, tp, tb, occluders=((foreground, fp, fb),))
                            if prediction.state == "unknown" or not lower <= prediction.occluded_fraction <= upper:
                                continue
                            yield {target: tp, foreground: fp}
            seeds = pair_seeds()
        for seed in seeds:
            for positions in fill(seed):
                actors = {key: static_actor(key, point, bodies[key]) for key, point in positions.items()}
                yield ScenePlan(camera, actors, metadata={"construction": "shared_static_role_domains",
                    "roles": dict(roles), "occlusion_pair": occlusion_pair, "native_pixel_preflight_required": True})
                yielded += 1
                if yielded >= max_candidates:
                    return
