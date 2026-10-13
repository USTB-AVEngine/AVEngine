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


def inward_footprint(source, target, *, source_binding=None, target_binding=None, region_rotation=None):
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
    if source_binding is not None:
        def barycentric(points, triangle):
            e1=triangle[:,1]-triangle[:,0];e2=triangle[:,2]-triangle[:,0]
            n=np.cross(e1,e2);den=np.einsum("ij,ij->i",n,n)
            q=points-triangle[:,None,0]
            u=np.divide(np.einsum("nij,nj->ni",np.cross(q,e2[:,None]),n),den[:,None],out=np.zeros(q.shape[:2]),where=den[:,None]>1e-40)
            vv=np.divide(np.einsum("nij,nj->ni",np.cross(e1[:,None],q),n),den[:,None],out=np.zeros(q.shape[:2]),where=den[:,None]>1e-40)
            return u,vv
        u,vv=barycentric(polygons,source)
        a0=source_binding[:,None,0]+u[...,None]*(source_binding[:,None,1]-source_binding[:,None,0])+vv[...,None]*(source_binding[:,None,2]-source_binding[:,None,0])
        ub,vb=barycentric(polygons,target)
        b0=target_binding[:,None,0]+ub[...,None]*(target_binding[:,None,1]-target_binding[:,None,0])+vb[...,None]*(target_binding[:,None,2]-target_binding[:,None,0])
        target_current=target[:,None,0]+ub[...,None]*(target[:,None,1]-target[:,None,0])+vb[...,None]*(target[:,None,2]-target[:,None,0])
        inherited_separation=a0-b0
        if region_rotation is not None:
            inherited_separation=np.einsum("nij,nkj->nki",region_rotation,inherited_separation)
        # Actual inward increment of matched material separation. Tangential
        # motion does not increase penetration. Bound by current inward depth
        # so travel across an original outward gap is not counted as penetration.
        relative=(polygons-target_current)-inherited_separation
        inward=-np.einsum("nij,nj->ni",relative,normal)
        depth=np.maximum(0.,np.minimum(depth,inward))
        depth[positions>=counts[:,None]]=0
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
    cross_total = sum(row["new_area_m2"] for row in groups
                      if row["region_pair"][0] != row["region_pair"][1])
    if cross_total > CROSS_REGION_AREA_LIMIT_M2 + 1e-12 and not any(
            row["area_rejected"] for row in failures):
        failures.append({
            "region_pair": ["all_different_regions", "combined"],
            "new_area_m2": float(cross_total), "new_maximum_depth_m": 0.,
            "area_rejected": True, "depth_rejected": False,
            "area_scope": "all_different_region_pairs_in_this_frame",
        })
    return failures

def semantic_vertex_regions(vertices, feet, head_min, head_max, tail_root, floor, height):
    """Stable anatomical regions from binding geometry, independent of skin weights."""
    v=np.asarray(vertices,float);feet=np.asarray(feet,float)
    labels=np.zeros(len(v),dtype=np.int64)
    nearest=np.linalg.norm(v[:,None,[0,2]]-feet[None,:,[0,2]],axis=2).argmin(1)
    horizontal=np.linalg.norm(v[:,[0,2]]-feet[nearest][:,[0,2]],axis=1)
    lower=(v[:,1]-floor)<=height*.30
    limb=((v[:,1]-floor)<=height*.50)&(horizontal<=height*.18)
    labels[lower|limb]=nearest[lower|limb]+3
    tail=(v[:,2]>=tail_root[2]-height*.025)&((v[:,1]-floor)>=height*.35)
    labels[tail]=2
    head=np.all(v>=np.asarray(head_min),axis=1)&np.all(v<=np.asarray(head_max),axis=1)
    labels[head]=1
    return labels

def relative_pair_footprints(area,depth,binding_area,binding_depth):
    """Binding inward overlap exists even for pairs without a surface crossing."""
    return np.maximum(np.asarray(area)-binding_area,0),np.maximum(np.asarray(depth)-binding_depth,0)


def pair_new_depth(vertices,binding_vertices,faces,pairs,labels,region_rotations):
    p=np.asarray(pairs,dtype=np.int64).reshape(-1,2)
    if not len(p):return np.empty((0,2))
    first=vertices[faces[p[:,0]]];second=vertices[faces[p[:,1]]]
    first0=binding_vertices[faces[p[:,0]]];second0=binding_vertices[faces[p[:,1]]]
    region=np.where(labels[p[:,0]]==labels[p[:,1]],labels[p[:,0]],0)
    common=region_rotations[region]
    _,a=inward_footprint(first,second,source_binding=first0,target_binding=second0,region_rotation=common)
    _,b=inward_footprint(second,first,source_binding=second0,target_binding=first0,region_rotation=common)
    return np.column_stack((a,b))
