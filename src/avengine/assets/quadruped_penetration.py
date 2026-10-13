"""Local surface penetration footprints, measured in metres.

A crossing face is clipped to the other face's orthogonal triangular prism and
inward half-space. Its remaining surface area and greatest normal depth are a
conservative local footprint. Summed overlapping footprints are capped by each
face's area; they can overestimate area, never deliberately undercount it.
"""
import numpy as np

DEPTH_LIMIT_M = .002
CROSS_REGION_AREA_LIMIT_M2 = 25e-6


def _clip(polygons, counts, origin, inward):
    """Vectorized Sutherland-Hodgman clipping of convex planar polygons."""
    size = polygons.shape[1]
    positions = np.arange(size)[None, :]
    valid = positions < counts[:, None]
    prev_index = np.where(positions == 0, counts[:, None] - 1, positions - 1)
    previous = np.take_along_axis(polygons, prev_index[..., None], axis=1)
    dcur = np.einsum("nij,nj->ni", polygons - origin[:, None, :], inward)
    dprev = np.einsum("nij,nj->ni", previous - origin[:, None, :], inward)
    cur_in = dcur >= -1e-12
    prev_in = dprev >= -1e-12
    crossing = valid & (cur_in != prev_in)
    denominator = dprev - dcur
    fraction = np.divide(dprev, denominator, out=np.zeros_like(dcur),
                         where=np.abs(denominator) > 1e-30)
    intersection = previous + fraction[..., None] * (polygons - previous)
    emit = np.stack((crossing, valid & cur_in), axis=2).reshape(len(counts), -1)
    points = np.stack((intersection, polygons), axis=2).reshape(len(counts), -1, 3)
    destination = np.cumsum(emit, axis=1) - 1
    rows, cols = np.nonzero(emit)
    out = np.zeros_like(polygons)
    if len(rows):
        if int(destination[rows, cols].max()) >= size:
            raise ValueError("clipped triangle exceeded polygon capacity")
        out[rows, destination[rows, cols]] = points[rows, cols]
    return out, emit.sum(axis=1)


def inward_footprint(source, target):
    """Exact polygon area/depth of source inside target's local inward prism."""
    source = np.asarray(source, dtype=float).reshape(-1, 3, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3, 3)
    count = len(source)
    if not count:
        return np.empty(0), np.empty(0)
    normal = np.cross(target[:, 1] - target[:, 0], target[:, 2] - target[:, 0])
    length = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, length[:, None], out=np.zeros_like(normal),
                       where=length[:, None] > 1e-20)
    polygons = np.zeros((count, 8, 3))
    polygons[:, :3] = source
    counts = np.full(count, 3, dtype=int)
    for edge in range(3):
        direction = target[:, (edge + 1) % 3] - target[:, edge]
        inward = np.cross(normal, direction)
        scale = np.linalg.norm(inward, axis=1)
        inward = np.divide(inward, scale[:, None], out=np.zeros_like(inward),
                           where=scale[:, None] > 1e-20)
        polygons, counts = _clip(polygons, counts, target[:, edge], inward)
    polygons, counts = _clip(polygons, counts, target[:, 0], -normal)
    positions = np.arange(8)[None, :]
    following_index = np.where(positions + 1 < counts[:, None], positions + 1, 0)
    following = np.take_along_axis(polygons, following_index[..., None], axis=1)
    cross = np.cross(polygons, following)
    cross[positions >= counts[:, None]] = 0
    area = np.linalg.norm(cross.sum(axis=1), axis=1) / 2
    depth = -np.einsum("nij,nj->ni", polygons - target[:, None, 0], normal)
    depth[positions >= counts[:, None]] = 0
    depth = np.maximum(depth.max(axis=1), 0)
    degenerate = (counts < 3) | (length <= 1e-20)
    area[degenerate] = 0
    depth[degenerate] = 0
    return area, depth


def pair_footprints(vertices, faces, pairs):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return np.empty((0, 2)), np.empty((0, 2))
    first = vertices[faces[pairs[:, 0]]]
    second = vertices[faces[pairs[:, 1]]]
    area_a, depth_a = inward_footprint(first, second)
    area_b, depth_b = inward_footprint(second, first)
    return np.column_stack((area_a, area_b)), np.column_stack((depth_a, depth_b))


def visible_penetration_gate(groups):
    """Counts intentionally have no input into this gate."""
    failures = []
    for row in groups:
        different = row["region_pair"][0] != row["region_pair"][1]
        area_fail = different and row["new_area_m2"] > CROSS_REGION_AREA_LIMIT_M2 + 1e-12
        depth_fail = row["new_maximum_depth_m"] > DEPTH_LIMIT_M + 1e-9
        if area_fail or depth_fail:
            failures.append(dict(row, area_rejected=area_fail, depth_rejected=depth_fail))
    return failures
