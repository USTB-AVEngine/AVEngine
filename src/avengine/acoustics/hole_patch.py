"""Conservative, acoustic-only planar repairs; uncertain openings stay unmodified.

The input package and all visual, semantic and navigation assets are read only.
Decision parameters are serialized before an evaluation batch is admitted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from shapely.geometry import Polygon, Point

RULES = {
    "version": "planar_boundary_v1",
    "weld_m": 1e-5,
    "exit_cluster_radius_m": .75,
    "plane_max_residual_m": .03,
    "min_hole_area_m2": .02,
    "max_hole_area_m2": 20.,
    "max_boundary_vertices": 4000,
    "patch_margin_m": .05,
    "outside_offset_m": .03,
    "structural_boundary_fraction": .95,
    "opening_context_m": .30,
    "visibility_grid_step_m": .10,
    "navigation_agent_height_m": 1.5,
    "navigation_radius_m": .10,
    "shell_outside_offset_m": .03,
    "unknown_policy": "review_without_patch",
    "shell_policy": "require_every_escape_to_have_an_accepted_structural_hole; protect_opening_rays",
}
STRUCTURAL = {"wall", "ceiling", "floor", "ceiling_lower", "ceiling_panel", "shower_wall", "shower_floor"}


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def escaped_rays(leakage):
    result = []
    for oi, row in enumerate(leakage):
        for ray in row["escaped_rays"]:
            result.append({"origin_index": oi, "room_label": row["room_label"], "role": row["role"],
                           "origin_m": row["origin_m"], **ray})
    return result


def exit_clusters(rays, bounds, radius=.75):
    """Connected components on each AABB face, retaining original ray identities."""
    by_face = defaultdict(list)
    bounds = np.asarray(bounds)
    for i, ray in enumerate(rays):
        p = np.asarray(ray["aabb_exit_m"])
        face = int(np.argmin(np.abs(bounds - p)))
        by_face[face].append(i)
    out = []
    for face, indices in sorted(by_face.items()):
        points = np.asarray([rays[i]["aabb_exit_m"] for i in indices])
        neighbours = cKDTree(points).query_ball_tree(cKDTree(points), radius)
        seen = set()
        for start in range(len(indices)):
            if start in seen:
                continue
            stack, group = [start], []
            seen.add(start)
            while stack:
                u = stack.pop(); group.append(indices[u])
                for v in neighbours[u]:
                    if v not in seen:
                        seen.add(v); stack.append(v)
            p = np.asarray([rays[i]["aabb_exit_m"] for i in group])
            out.append({"cluster_id": len(out), "aabb_face": face, "ray_ids": sorted(group),
                        "centroid_m": p.mean(0).tolist(), "bounds_m": [p.min(0).tolist(), p.max(0).tolist()]})
    return out


def boundary_loops(vertices, triangles, weld_m):
    """Weld category seams for detection only; never change package vertices."""
    _, first, inverse = np.unique(np.round(np.asarray(vertices)/weld_m).astype(np.int64),
                                  axis=0, return_index=True, return_inverse=True)
    v = np.asarray(vertices)[first].astype(float)
    t = inverse[np.asarray(triangles)]
    e = np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))
    e.sort(axis=1)
    unique, ix, count = np.unique(e, axis=0, return_index=True, return_counts=True)
    mask = count == 1
    edges, faces = unique[mask], ix[mask] % len(t)
    adj = defaultdict(list)
    face_by_edge = {}
    for edge, fi in zip(edges, faces):
        a, b = map(int, edge)
        if a == b:
            continue
        adj[a].append(b); adj[b].append(a); face_by_edge[(min(a,b),max(a,b))] = int(fi)
    seen, loops, irregular = set(), [], 0
    for root in sorted(adj):
        if root in seen:
            continue
        stack, comp = [root], []
        seen.add(root)
        while stack:
            u = stack.pop(); comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w); stack.append(w)
        if len(comp)<3 or any(len(adj[u]) != 2 for u in comp):
            irregular += 1
            continue
        order, prev, cur = [], -1, min(comp)
        while cur not in order:
            order.append(cur)
            nxt = next(w for w in adj[cur] if w != prev)
            prev, cur = cur, nxt
        if cur != order[0] or len(order) != len(comp):
            irregular += 1
            continue
        fis = [face_by_edge[tuple(sorted((a,b)))] for a,b in zip(order,order[1:]+order[:1])]
        loops.append((v[order], np.asarray(fis,dtype=int)))
    return loops, {"welded_vertex_count": len(v), "boundary_edge_count": len(edges),
                   "closed_loop_count": len(loops), "irregular_component_count": irregular}


def quad_frame(quad):
    q = np.asarray(quad, float)
    u = q[1]-q[0]; w = q[3]-q[0]
    lu, lw = np.linalg.norm(u), np.linalg.norm(w)
    u /= lu; w /= lw
    n = np.cross(u,w); n /= np.linalg.norm(n)
    return q[0], u, w, n, lu, lw


def ray_quad_intersections(origins, directions, quad, maximum=None):
    """Return positive first-plane distances, with infinity outside the rectangle."""
    origins, directions = np.asarray(origins,float),np.asarray(directions,float)
    c,u,w,n,lu,lw=quad_frame(quad)
    denom=directions@n
    with np.errstate(divide="ignore",invalid="ignore"):
        distance=((c-origins)@n)/denom
        delta=origins+distance[:,None]*directions-c
    keep=(np.abs(denom)>1e-10)&(distance>.005)&(delta@u>=-1e-6)&(delta@u<=lu+1e-6)&(delta@w>=-1e-6)&(delta@w<=lw+1e-6)
    if maximum is not None:
        keep &= distance < np.asarray(maximum)-.005
    return np.where(keep,distance,np.inf)


def quad_grid(quad, step=.1):
    c,u,w,n,lu,lw=quad_frame(quad)
    a=np.linspace(0,lu,max(3,int(np.ceil(lu/step))+1))
    b=np.linspace(0,lw,max(3,int(np.ceil(lw/step))+1))
    aa,bb=np.meshgrid(a,b,indexing="ij")
    return c+aa.ravel()[:,None]*u+bb.ravel()[:,None]*w


def nav_prism_intersections(quad, nav_vertices, nav_triangles, height=1.5, radius=.1):
    """Clip the entire quad against navigable triangles extruded to agent height.

    Unlike grid snapping, this detects a narrow walkable strip between samples.
    The navmesh already represents supported floors. Horizontal radius is added
    to protect the swept agent volume; vertical tolerance is only 5 mm.
    """
    q=np.asarray(quad,float);nv=np.asarray(nav_vertices,float);nt=np.asarray(nav_triangles,int)
    tri=nv[nt];lo=tri.min(1);hi=tri.max(1);hi[:,1]+=height
    pad=np.array([radius,.005,radius])
    ids=np.flatnonzero(np.all(hi+pad>=q.min(0),axis=1)&np.all(lo-pad<=q.max(0),axis=1))
    hits=[]
    for i in ids:
        prism=np.concatenate((tri[i],tri[i]+[0,height,0]))
        try:
            equations=ConvexHull(prism).equations
        except Exception:
            # A degenerate nav triangle cannot prove a clear candidate.
            hits.append(int(i));continue
        poly=q.copy()
        for eq in equations:
            normal,offset=eq[:3],eq[3]
            offset-=radius*np.linalg.norm(normal[[0,2]])+.005*abs(normal[1])
            clipped=[]
            for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                da,db=a@normal+offset,b@normal+offset
                if da<=1e-9:clipped.append(a)
                if (da<=0)!=(db<=0):clipped.append(a+(b-a)*da/(da-db))
            poly=np.asarray(clipped)
            if not len(poly):break
        if len(poly):hits.append(int(i))
    return hits


def loop_candidates(vertices, triangles, labels, leakage, rules=RULES):
    """Use leakage clusters to associate closed, planar, semantic boundary holes."""
    v=np.asarray(vertices,float);t=np.asarray(triangles)
    rays=escaped_rays(leakage)
    bounds=np.stack((v.min(0),v.max(0)))
    clusters=exit_clusters(rays,bounds,rules["exit_cluster_radius_m"])
    loops,stats=boundary_loops(v,t,rules["weld_m"])
    origins=np.asarray([r['origin_m'] for r in rays]).reshape(-1,3)
    dirs=np.asarray([r['direction'] for r in rays]).reshape(-1,3)
    centers=v[t].mean(1)
    tree=cKDTree(centers)
    opening_ids=np.flatnonzero(["door" in name or "window" in name for name in labels])
    opening_vertices=np.unique(t[opening_ids]) if len(opening_ids) else []
    opening_tree=cKDTree(v[opening_vertices]) if len(opening_vertices) else None
    candidates=[];excluded=Counter()
    for li,(points,adjfaces) in enumerate(loops):
        if len(points)>rules['max_boundary_vertices']:
            excluded['boundary_too_large']+=1;continue
        c=points.mean(0);_,_,axes=np.linalg.svd(points-c,full_matrices=False)
        u,w,n=axes
        residual=float(np.max(np.abs((points-c)@n)))
        if residual>rules['plane_max_residual_m']:
            excluded['nonplanar']+=1;continue
        xy=np.stack(((points-c)@u,(points-c)@w),axis=1)
        polygon=Polygon(xy)
        if not polygon.is_valid or not rules['min_hole_area_m2']<=polygon.area<=rules['max_hole_area_m2']:
            excluded['invalid_or_area']+=1;continue
        lower=xy.min(0)-rules['patch_margin_m'];upper=xy.max(0)+rules['patch_margin_m']
        q=c+np.array([[lower[0],lower[1]],[upper[0],lower[1]],[upper[0],upper[1]],[lower[0],upper[1]]])@axes[:2]
        distances=ray_quad_intersections(origins,dirs,q)
        ray_ids=np.flatnonzero(np.isfinite(distances)).tolist()
        if not ray_ids:
            excluded['no_escaped_ray']+=1;continue
        # A hole ring has no surface inside. An outer wall perimeter does.
        nearby=tree.query_ball_point(c,float(np.linalg.norm(upper-lower)))
        inside=[]
        for fi in nearby:
            p=centers[fi]
            if abs((p-c)@n)<=rules['plane_max_residual_m'] and polygon.contains(Point([(p-c)@u,(p-c)@w])):
                inside.append(fi)
        if inside:
            excluded['surface_inside_loop']+=1;continue
        counts=Counter(labels[int(i)] for i in adjfaces)
        name,count=counts.most_common(1)[0]
        structural=sum(k for label,k in counts.items() if label in STRUCTURAL)/len(adjfaces)
        reasons=[]
        if structural<rules['structural_boundary_fraction'] or name not in STRUCTURAL:
            reasons.append('UNCERTAIN_SURFACE_SEMANTICS')
        side=(origins[ray_ids]-c)@n
        if np.all(side>0):n=-n
        elif not np.all(side<0):reasons.append('ROOM_SIDE_AMBIGUOUS')
        q=q+n*rules['outside_offset_m']
        opening_distance=None
        if opening_tree is not None:
            opening_distance=float(opening_tree.query(quad_grid(q,rules['visibility_grid_step_m']))[0].min())
            if opening_distance<rules['opening_context_m']:
                reasons.append('SEMANTIC_DOOR_OR_WINDOW_CONTEXT')
        # Unlabelled wall apertures resembling doors/windows remain review.
        if abs(n[1])<.15 and polygon.area>.2:
            reasons.append('WALL_APERTURE_REQUIRES_OPENING_REVIEW')
        candidates.append({'candidate_id':f'loop_{li}', 'boundary_loop_vertices_m':points.tolist(),
            'quad_m':q.tolist(),'plane_normal_outward':n.tolist(),'plane_residual_max_m':residual,
            'hole_area_m2':float(polygon.area),'patch_area_m2':float(np.prod(upper-lower)),
            'semantic_label':name,'adjacent_semantic_counts':dict(counts),'structural_boundary_fraction':structural,
            'opening_distance_m':opening_distance,'outside_offset_m':rules['outside_offset_m'],
            'ray_ids':ray_ids,'exit_cluster_ids':[x['cluster_id'] for x in clusters if set(x['ray_ids'])&set(ray_ids)],
            'reasons':reasons,'status':'review' if reasons else 'candidate'})
    stats['excluded_loop_counts']=dict(excluded)
    return candidates,rays,clusters,stats


def check_direct_paths(quad, pairs):
    if not pairs:return []
    a=np.asarray([p['listener_m'] for p in pairs],float);b=np.asarray([p['source_m'] for p in pairs],float)
    lengths=np.linalg.norm(b-a,axis=1)
    distances=ray_quad_intersections(a,(b-a)/lengths[:,None],quad,lengths)
    return [pairs[int(i)] for i in np.flatnonzero(np.isfinite(distances))]


def shell_quads(bounds, offset=.03):
    lo,hi=np.asarray(bounds,float).copy();lo-=offset;hi+=offset
    result=[]
    for axis in range(3):
        others=[a for a in range(3) if a!=axis]
        for side in (0,1):
            q=np.zeros((4,3));q[:,axis]=(lo if side==0 else hi)[axis]
            q[:,others]=[[lo[others[0]],lo[others[1]]],[hi[others[0]],lo[others[1]]],
                          [hi[others[0]],hi[others[1]]],[lo[others[0]],hi[others[1]]]]
            result.append({'candidate_id':f'shell_{axis}_{side}','quad_m':q.tolist(),
                           'patch_area_m2':float(np.prod(hi[others]-lo[others])),
                           'outside_offset_m':offset,'axis':axis,'side':side})
    return result


def export_derived_glb(package_manifest, patches, output):
    """Export source package objects plus patch quads, preserving every base face.

    Vertices are returned to the original coordinate frame by its inverse affine
    transform. Recompilation then uses the unchanged material proposal/database.
    The baseline frames are signed axis permutations, so this is exactly lossless.
    """
    p=Path(package_manifest);m=json.loads(p.read_text())
    v=np.load(p.parent/m['arrays']['vertices']['path'],allow_pickle=False)
    t=np.load(p.parent/m['arrays']['triangles']['path'],allow_pickle=False)
    matrix=np.asarray(m['geometry']['source_to_canonical']['matrix_row_major']).reshape(4,4)
    inv=np.linalg.inv(matrix)
    source_v=(v.astype(float)@inv[:3,:3].T+inv[:3,3]).astype('<f4')
    materials=sorted({o['source_material_name'] for o in m['objects']})
    document={'asset':{'version':'2.0','generator':'AVEngine acoustic-only hole patch'},'scene':0,
              'scenes':[{'nodes':[]}],'nodes':[],'meshes':[],'materials':[{'name':s} for s in materials],
              'accessors':[],'bufferViews':[],'buffers':[]}
    binary=bytearray()
    def accessor(array,typ,component,target):
        a=np.ascontiguousarray(array)
        while len(binary)%4:binary.append(0)
        offset=len(binary);binary.extend(a.tobytes())
        vi=len(document['bufferViews']);document['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':a.nbytes,'target':target})
        ai=len(document['accessors']);r={'bufferView':vi,'componentType':component,'count':len(a),'type':typ}
        if typ=='VEC3':r.update(min=a.min(0).tolist(),max=a.max(0).tolist())
        document['accessors'].append(r);return ai
    def obj(name,vertices,faces,material):
        va=accessor(vertices,'VEC3',5126,34962);ta=accessor(faces.reshape(-1).astype('<u4'),'SCALAR',5125,34963)
        mi=len(document['meshes']);document['meshes'].append({'primitives':[{'attributes':{'POSITION':va},'indices':ta,'material':materials.index(material),'mode':4}]})
        ni=len(document['nodes']);document['nodes'].append({'name':name,'mesh':mi});document['scenes'][0]['nodes'].append(ni)
    for o in m['objects']:
        vo,vc,to,tc=[int(o[k]) for k in ('vertex_offset','vertex_count','triangle_offset','triangle_count')]
        obj(o['object_id'],source_v[vo:vo+vc],t[to:to+tc]-vo,o['source_material_name'])
    for patch in patches:
        q=np.asarray(patch['quad_m'],float)
        sv=(q@inv[:3,:3].T+inv[:3,3]).astype('<f4')
        obj(patch['candidate_id'],sv,np.array([[0,1,2],[0,2,3]],dtype='<u4'),patch['semantic_label'])
    document['buffers']=[{'byteLength':len(binary)}]
    payload=json.dumps(document,separators=(',',':')).encode();payload+=b' '*((-len(payload))%4)
    binary.extend(b'\x00'*((-len(binary))%4))
    total=12+8+len(payload)+8+len(binary)
    with Path(output).open('xb') as f:
        f.write(struct.pack('<4sII',b'glTF',2,total));f.write(struct.pack('<I4s',len(payload),b'JSON'));f.write(payload)
        f.write(struct.pack('<I4s',len(binary),b'BIN\x00'));f.write(binary)
    return Path(output)


def planar_cluster_candidates(vertices, triangles, labels, leakage, rooms, rules=RULES):
    """Recover small horizontal holes whose scan boundary is fragmented.

    An existing structural plane must support at least six of eight samples
    around the proposed rectangle. A room AABB alone never supplies a surface.
    Rooms and their floor polygons are the original, preselected witnesses.
    """
    v=np.asarray(vertices);t=np.asarray(triangles)
    rays=escaped_rays(leakage);out=[];excluded=Counter()
    if not rays:return out,dict(excluded)
    origin=np.asarray([r['origin_m'] for r in rays]);direction=np.asarray([r['direction'] for r in rays])
    for kind,sign in (('ceiling',1),('floor',-1)):
        ids=np.flatnonzero([x==kind or x in (('ceiling_lower','ceiling_panel') if kind=='ceiling' else ('shower_floor',)) for x in labels])
        if not len(ids):continue
        points=v[t[ids]].astype(float);cross=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
        length=np.linalg.norm(cross,axis=1)
        flat=(length>0)&(np.abs(cross[:,1])>=.99*length)
        centers=points[flat].mean(1);faceids=ids[flat]
        for room in rooms:
            polygon=shape_polygon(room.get('floor_polygon'))
            if polygon is None:continue
            camera=np.asarray(room['placement']['camera_m']);floor=camera[1]-1.5
            scope=polygon.buffer(.20)
            inroom=np.asarray([scope.contains(Point(p[[0,2]])) for p in centers])
            if kind=='ceiling':inroom &= (centers[:,1]>camera[1]+.5)&(centers[:,1]<camera[1]+3.)
            else:inroom &= np.abs(centers[:,1]-floor)<.15
            near=centers[inroom]
            nearids=faceids[inroom]
            if len(near)<8:excluded['insufficient_plane_support']+=1;continue
            bins=np.round(near[:,1]/.05).astype(int)
            bucket=Counter(bins).most_common(1)[0][0]
            plateau=near[bins==bucket];plateauids=nearids[bins==bucket]
            y=float(np.median(plateau[:,1]));coplanar=np.abs(plateau[:,1]-y)<=rules['plane_max_residual_m']
            plateau=plateau[coplanar];plateauids=plateauids[coplanar]
            if len(plateau)<8:excluded['insufficient_coplanar_support']+=1;continue
            mask=(direction[:,1]*sign>.1)&(np.asarray([r['room_label']==room['room_label'] for r in rays]))
            with np.errstate(divide='ignore',invalid='ignore'):
                distance=(y-origin[:,1])/direction[:,1]
                hits=origin+distance[:,None]*direction
            ix=[int(i) for i in np.flatnonzero(mask&(distance>.01)) if scope.contains(Point(hits[i,[0,2]]))]
            if not ix:continue
            xy=hits[ix][:,[0,2]];tree=cKDTree(xy);adj=tree.query_ball_tree(tree,rules['exit_cluster_radius_m']);seen=set()
            for start in range(len(ix)):
                if start in seen:continue
                stack=[start];seen.add(start);comp=[]
                while stack:
                    u=stack.pop();comp.append(u)
                    for w in adj[u]:
                        if w not in seen:seen.add(w);stack.append(w)
                rayids=[ix[i] for i in comp];loc=hits[rayids][:,[0,2]]
                low=loc.min(0)-rules['patch_margin_m'];high=loc.max(0)+rules['patch_margin_m']
                area=float(np.prod(high-low))
                if len(rayids)<2 or not rules['min_hole_area_m2']<=area<=rules['max_hole_area_m2']:
                    excluded['sparse_or_area']+=1;continue
                mid=(low+high)*.5;pad=.15
                ring=np.array([[low[0]-pad,low[1]-pad],[mid[0],low[1]-pad],[high[0]+pad,low[1]-pad],
                               [high[0]+pad,mid[1]],[high[0]+pad,high[1]+pad],[mid[0],high[1]+pad],
                               [low[0]-pad,high[1]+pad],[low[0]-pad,mid[1]]])
                ringdist,ringidx=cKDTree(plateau[:,[0,2]]).query(ring)
                support=ringdist<=.25
                reasons=[]
                if int(support.sum())<6:reasons.append('INCOMPLETE_STRUCTURAL_RING_SUPPORT')
                if not all(scope.contains(Point(x)) for x in ring):reasons.append('ROOM_FOOTPRINT_EDGE_AMBIGUOUS')
                q=np.array([[low[0],y+sign*rules['outside_offset_m'],low[1]],
                            [high[0],y+sign*rules['outside_offset_m'],low[1]],
                            [high[0],y+sign*rules['outside_offset_m'],high[1]],
                            [low[0],y+sign*rules['outside_offset_m'],high[1]]])
                counts=Counter(labels[int(plateauids[i])] for i in ringidx[support])
                label=counts.most_common(1)[0][0] if counts else kind
                out.append({'candidate_id':f'plane_{kind}_{room["room_label"]}_{len(out)}','room_label':room['room_label'],
                    'quad_m':q.tolist(),'patch_area_m2':area,'hole_area_m2':None,'plane_normal_outward':[0,sign,0],
                    'outside_offset_m':rules['outside_offset_m'],'plane_y_m':y,'plane_support_triangles':len(plateau),
                    'ring_sample_count':8,'ring_supported_count':int(support.sum()),'ring_distances_m':ringdist.tolist(),
                    'semantic_label':label,'adjacent_semantic_counts':dict(counts),'ray_ids':rayids,'exit_cluster_ids':[],
                    'reasons':reasons,'status':'review' if reasons else 'candidate',
                    'geometry_method':'escaped-ray clusters on an existing horizontal semantic plane with an enclosing structural ring'})
    return out,dict(excluded)


def shape_polygon(value):
    if not value:return None
    from shapely.geometry import shape
    polygon=shape(value)
    return polygon if not polygon.is_empty and polygon.is_valid else None
