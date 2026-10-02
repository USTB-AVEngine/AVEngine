"""Furniture-seeded geodesic partition with clearance-weighted minimum cuts."""

from __future__ import annotations

import math
import networkx as nx
import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .navigation import cells_polygon, components


def furniture_clusters(furniture, floor_y, p):
    anchors = [
        f
        for f in furniture
        if f["category"] in p["anchor_categories"]
        and f["height_range_m"][0] <= floor_y + p["camera_height_m"]
        and f["height_range_m"][1] >= floor_y - p["floor_height_separation_m"]
    ]
    graph = [[] for _ in anchors]
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            if (
                np.linalg.norm(
                    np.array(anchors[i]["centre_xz_m"]) - anchors[j]["centre_xz_m"]
                )
                <= p["furniture_cluster_link_m"]
            ):
                graph[i].append(j)
                graph[j].append(i)
    out = []
    for ids in components(graph):
        xy = np.array([anchors[i]["centre_xz_m"] for i in ids])
        out.append(
            dict(
                centre_xz_m=xy.mean(axis=0).tolist(),
                instance_ids=[anchors[i]["instance_id"] for i in ids],
                categories=[anchors[i]["category"] for i in ids],
            )
        )
    return out


def split_triggers(scope, clusters, p):
    reasons = []
    if scope.area > p["split_area_trigger_m2"]:
        reasons.append("SPLIT_LARGE_AREA")
    if any(
        np.linalg.norm(np.array(a["centre_xz_m"]) - b["centre_xz_m"])
        > p["split_cluster_separation_m"]
        for i, a in enumerate(clusters)
        for b in clusters[i + 1 :]
    ):
        reasons.append("SPLIT_SEPARATED_FURNITURE")
    convexity = scope.area / scope.convex_hull.area if scope.convex_hull.area else 0
    if convexity < p["split_convexity_trigger"]:
        reasons.append("SPLIT_LOW_CONVEXITY")
    return reasons, float(convexity)


def graph_distances(points, adj, seeds):
    rows = []
    cols = []
    values = []
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            values.append(float(np.linalg.norm(points[i] - points[j])))
    graph = csr_matrix((values, (rows, cols)), shape=(len(points), len(points)))
    return np.atleast_2d(dijkstra(graph, directed=False, indices=seeds))


def propose(scope, points, clearance, adj, clusters, p):
    if not len(points):
        return dict(status="review", reason_codes=["SPLIT_NO_NAVIGATION"]), []
    comps = components(adj)
    main = max(comps, key=len)
    seeds = []
    for cluster in clusters:
        target = np.array(cluster["centre_xz_m"])
        seed = min(main, key=lambda i: np.linalg.norm(points[i, [0, 2]] - target))
        if seed not in seeds:
            seeds.append(seed)
    seeds = seeds[: p["max_split_seeds"]]
    # Do not fabricate furniture centres for an empty or single-anchor room.
    if len(seeds) < 2:
        return (
            dict(
                status="review",
                reason_codes=["SPLIT_INSUFFICIENT_FURNITURE_SEEDS"],
                seed_count=len(seeds),
            ),
            [],
        )
    distances = graph_distances(points, adj, seeds)
    groups = []
    cut_records = []

    def cut(allowed, seed_labels):
        if len(seed_labels) == 1:
            groups.append(sorted(allowed))
            return
        left = seed_labels[0]
        right = seed_labels[1:]
        dleft = distances[left]
        dright = np.min(distances[right], axis=0)
        source, sink = "source", "sink"
        g = nx.DiGraph()
        aset = set(allowed)
        for i in allowed:
            # Cost of assigning the node to the opposite seed's partition.
            dl = float(dleft[i])
            dr = float(dright[i])
            weight = p["split_geodesic_cost_weight"]
            g.add_edge(source, i, capacity=weight * dr)
            g.add_edge(i, sink, capacity=weight * dl)
            if dl <= p["split_terminal_radius_m"]:
                g[source][i]["capacity"] = 1e9
            if dr <= p["split_terminal_radius_m"]:
                g[i][sink]["capacity"] = 1e9
            for j in adj[i]:
                if j in aset:
                    cap = (
                        min(clearance[i], clearance[j]) + p["grid_step_m"] / 2
                    ) ** 2 * p["grid_step_m"]
                    g.add_edge(i, j, capacity=float(cap))
        value, (a, b) = nx.minimum_cut(g, source, sink)
        a = set(a) & aset
        b = set(b) & aset
        if not a or not b:
            # An explicit fallback retains the auditable furniture Voronoi result.
            a = {i for i in allowed if dleft[i] <= dright[i]}
            b = aset - a
            method = "geodesic_voronoi_fallback"
        else:
            method = "clearance_weighted_min_cut"
        cut_records.append(
            dict(
                method=method,
                capacity=float(value),
                left_samples=len(a),
                right_samples=len(b),
            )
        )
        if a:
            groups.append(sorted(a))
        if b:
            cut(sorted(b), right)

    cut(sorted(main), list(range(len(seeds))))
    # Disconnected pieces are registered independently rather than silently lost.
    for comp in comps:
        if comp is not main:
            groups.append(sorted(comp))
    connected = []
    for group in groups:
        connected.extend(components(adj, group))
    geometries = [cells_polygon(points, g, p["grid_step_m"], scope) for g in connected]
    covered = shapely.union_all(geometries)
    return (
        dict(
            status="proposed",
            reason_codes=[],
            seed_indices=seeds,
            seed_points_m=points[seeds].tolist(),
            method="furniture seeds / geodesic data costs / clearance-weighted min-cut / native-connected components",
            cuts=cut_records,
            part_sample_indices=connected,
            unassigned_floor_area_m2=float(scope.difference(covered).area),
            unassigned_reason="non-navigable or unsampled ground is not assigned to a proposed activity mask",
        ),
        geometries,
    )
