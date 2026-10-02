"""Read existing navmeshes with PathFinder; CPU mesh rays, no Simulator."""

from __future__ import annotations

import itertools
import math
from collections import deque

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, box


def cells_polygon(points, indices, step, scope):
    if not indices:
        return GeometryCollection()
    xy = (np.floor(points[indices][:, [0, 2]] / step) + 0.5) * step
    half = step / 2
    cells = shapely.box(
        xy[:, 0] - half, xy[:, 1] - half, xy[:, 0] + half, xy[:, 1] + half
    )
    return shapely.union_all(cells).intersection(scope)


def sample_navigation(pf, hs, geometry, floor_y, p):
    step = p["grid_step_m"]
    minx, minz, maxx, maxz = geometry.bounds
    xs = np.arange(math.floor(minx / step) * step + step / 2, maxx, step)
    zs = np.arange(math.floor(minz / step) * step + step / 2, maxz, step)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    inside = shapely.contains_xy(geometry, xx, zz)
    indices = np.argwhere(inside)
    points = []
    cells = []
    clearance = []
    islands = []
    for ix, iz in indices:
        probe = np.array([xs[ix], floor_y, zs[iz]], np.float32)
        q = np.asarray(pf.snap_point(probe), float)
        if (
            not np.isfinite(q).all()
            or abs(q[1] - floor_y) > p["floor_height_separation_m"]
        ):
            continue
        if np.linalg.norm(q[[0, 2]] - probe[[0, 2]]) > p["nav_snap_horizontal_max_m"]:
            continue
        if not shapely.contains_xy(geometry, q[0], q[2]):
            continue
        points.append(q.tolist())
        cells.append((int(ix), int(iz)))
        clearance.append(
            float(pf.distance_to_closest_obstacle(q, p["clearance_query_radius_m"]))
        )
        islands.append(int(pf.get_island(q)))
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    lookup = {cell: i for i, cell in enumerate(cells)}
    adj = [[] for _ in points]
    checked = 0
    rejected = 0
    for i, (ix, iz) in enumerate(cells):
        for dx, dz in [(1, 0), (0, 1)]:
            j = lookup.get((ix + dx, iz + dz))
            if j is None:
                continue
            checked += 1
            if (
                islands[i] != islands[j]
                or abs(points[i, 1] - points[j, 1]) > p["floor_height_separation_m"]
            ):
                rejected += 1
                continue
            sp = hs.ShortestPath()
            sp.requested_start = points[i]
            sp.requested_end = points[j]
            if (
                not pf.find_path(sp)
                or sp.geodesic_distance > step * p["nav_edge_max_geodesic_factor"]
            ):
                rejected += 1
                continue
            adj[i].append(j)
            adj[j].append(i)
    comps = components(adj)
    main = max(comps, key=len, default=[])
    main_geometry = cells_polygon(points, main, step, geometry)
    ratios = np.asarray(clearance) >= p["clearance_min_m"]
    result = dict(
        sample_grid_count=int(inside.sum()),
        navigable_sample_count=len(points),
        component_count=len(comps),
        component_sample_counts=sorted([len(c) for c in comps], reverse=True),
        nav_main_area_m2=float(main_geometry.area),
        nav_sample_area_m2=float(
            cells_polygon(points, list(range(len(points))), step, geometry).area
        ),
        clearance_ge_0_5_ratio=float(ratios.mean()) if len(points) else None,
        clearance_ge_0_5_floor_grid_ratio=(
            float(ratios.sum() / inside.sum()) if inside.sum() else None
        ),
        checked_grid_edges=checked,
        rejected_grid_edges=rejected,
        method="world grid centres; clipped cell area; four neighbours checked by native geodesic",
    )
    return result, points, np.asarray(clearance), adj, comps


def components(adj, allowed=None):
    allowed = set(range(len(adj))) if allowed is None else set(allowed)
    seen = set()
    out = []
    for start in sorted(allowed):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v in allowed and v not in seen:
                    seen.add(v)
                    stack.append(v)
        out.append(comp)
    return out


def farthest_sample(points, indices, k):
    if not indices:
        return []
    indices = np.asarray(sorted(indices))
    xy = points[indices][:, [0, 2]]
    start = int(np.argmin(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)))
    chosen = [start]
    distance = np.sum((xy - xy[start]) ** 2, axis=1)
    while len(chosen) < min(k, len(indices)):
        nxt = int(np.argmax(distance))
        if nxt in chosen:
            break
        chosen.append(nxt)
        distance = np.minimum(distance, np.sum((xy - xy[nxt]) ** 2, axis=1))
    return indices[chosen].tolist()


def ray_clear_batch(mesh, start, ends, tolerance):
    if not len(ends):
        return np.array([], bool)
    ends = np.asarray(ends, dtype=float)
    directions = ends - start
    length = np.linalg.norm(directions, axis=1)
    directions /= length[:, None]
    origins = np.repeat(np.asarray(start)[None, :], len(ends), axis=0)
    locations, rays, _ = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    clear = np.ones(len(ends), bool)
    if len(rays):
        distances = np.linalg.norm(locations - origins[rays], axis=1)
        clear[rays] = distances >= length[rays] - tolerance
    return clear


def placement(mesh, points, clearance, adj, p, allowed=None):
    """Find a real three-ray witness on the existing navmesh, on CPU.

    0.5 m clearance remains a reported quality metric. It is NOT the radius of
    both a camera and every source. Separate pose clearances prevent discarding
    nearly all otherwise supported navmesh samples in ordinary furnished rooms.
    """
    comps = components(adj, allowed)
    main = max(comps, key=len, default=[])
    cams = farthest_sample(
        points,
        [
            i
            for i in main
            if clearance[i]
            >= p.get("placement_camera_clearance_m", p["clearance_min_m"])
        ],
        p["placement_camera_samples"],
    )
    sources = farthest_sample(
        points,
        [
            i
            for i in main
            if clearance[i]
            >= p.get("placement_source_clearance_m", p["clearance_min_m"])
        ],
        p["placement_source_samples"],
    )
    tested_rays = 0
    tested_pairs = 0
    visible_rays = 0
    for ci in cams:
        camera = points[ci] + [0, p["camera_height_m"], 0]
        target_ids = [i for i in sources if i != ci]
        ends = points[target_ids] + [0, p["source_height_m"], 0]
        lengths = np.linalg.norm(ends - camera, axis=1)
        keep = (lengths >= p["distance_min_m"]) & (lengths <= p["distance_max_m"])
        ends = ends[keep]
        clear = ray_clear_batch(mesh, camera, ends, p["ray_endpoint_tolerance_m"])
        tested_rays += len(ends)
        ends = ends[clear]
        visible_rays += len(ends)
        # Vectorized filters change speed, not distance, FOV or raw-mesh rays.
        ii, jj = np.triu_indices(len(ends), 1)
        if not len(ii):
            continue
        dist = np.linalg.norm(ends[ii] - ends[jj], axis=1)
        a = ends[ii][:, [0, 2]] - camera[[0, 2]]
        b = ends[jj][:, [0, 2]] - camera[[0, 2]]
        cosine = np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        keep = (
            (dist >= p["distance_min_m"])
            & (dist <= p["distance_max_m"])
            & (angle <= p["camera_hfov_deg"])
        )
        for i, j, d, ang in zip(ii[keep], jj[keep], dist[keep], angle[keep]):
            tested_pairs += 1
            tested_rays += 1
            if not ray_clear_batch(
                mesh, ends[i], [ends[j]], p["ray_endpoint_tolerance_m"]
            )[0]:
                continue
            return dict(
                found=True,
                camera_m=camera.tolist(),
                source_1_m=ends[i].tolist(),
                source_2_m=ends[j].tolist(),
                horizontal_angle_deg=float(ang),
                pairwise_distances_m=[
                    float(np.linalg.norm(ends[i] - camera)),
                    float(np.linalg.norm(ends[j] - camera)),
                    float(d),
                ],
                tested_rays=tested_rays,
                tested_pairs=tested_pairs,
                camera_source_clear_rays=visible_rays,
                camera_candidates=len(cams),
                source_candidates=len(sources),
                camera_clearance_min_m=p.get(
                    "placement_camera_clearance_m", p["clearance_min_m"]
                ),
                source_clearance_min_m=p.get(
                    "placement_source_clearance_m", p["clearance_min_m"]
                ),
                search="deterministic bounded CPU search; native navmesh support; three clear raw scan-mesh segments",
                acoustics="not_run",
            )
    return dict(
        found=False,
        tested_rays=tested_rays,
        tested_pairs=tested_pairs,
        camera_source_clear_rays=visible_rays,
        camera_candidates=len(cams),
        source_candidates=len(sources),
        search="no witness found within declared sample budget; not an impossibility proof",
        acoustics="not_run",
    )
