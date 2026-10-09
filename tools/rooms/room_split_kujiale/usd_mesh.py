"""Read original USD meshes, materials and instance semantics without mutation."""
from __future__ import annotations
import re, math
from collections import defaultdict, Counter
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from PIL import Image
from tools.rooms.room_split_kujiale.adapter import transform_points
from tools.rooms.room_screening.geometry import classify_category


def material_at(prim):
    """Read binding relationships without applying MaterialBindingAPI to inputs."""
    p = prim
    while p and p.GetPath().pathString != '/':
        rel = p.GetRelationship('material:binding')
        if rel and rel.GetTargets(): return prim.GetStage().GetPrimAtPath(rel.GetTargets()[0])
        p = p.GetParent()
    return None


def material_colour(prim, cache):
    from pxr import Usd, UsdShade
    mat = material_at(prim)
    if not mat: return np.array([.65, .65, .65])
    key = str(mat.GetPath())
    if key in cache: return cache[key]
    colour = np.array([.65, .65, .65])
    for p in Usd.PrimRange(mat):
        if p.GetTypeName() != 'Shader': continue
        shader = UsdShade.Shader(p)
        col = shader.GetInput('BaseColor_Color')
        if col and col.Get() is not None:
            colour = np.asarray(col.Get(), float)[:3]
            tex = shader.GetInput('BaseColor_Tex')
            flag = shader.GetInput('IsBaseColorTex')
            if tex and flag and float(flag.Get() or 0) > .5:
                asset = tex.Get()
                path = Path(asset.resolvedPath) if asset and asset.resolvedPath else None
                if path and path.is_file():
                    tkey = 'texture:' + str(path)
                    if tkey not in cache:
                        with Image.open(path) as im:
                            im.thumbnail((64, 64)); a = np.asarray(im.convert('RGB'), float) / 255
                        # Authored colour is linear; texture RGB is decoded as sRGB.
                        lin = np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)
                        cache[tkey] = lin.mean((0, 1))
                    colour = colour * cache[tkey]
            break
        col = shader.GetInput('diffuseColor')
        if col and col.Get() is not None: colour = np.asarray(col.Get(), float)[:3]; break
    cache[key] = np.clip(colour, 0, 1)
    return cache[key]


def mesh_arrays(prim, xform_cache):
    from pxr import UsdGeom
    m = UsdGeom.Mesh(prim)
    points = np.asarray(m.GetPointsAttr().Get(), float)
    counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), int)
    flat = np.asarray(m.GetFaceVertexIndicesAttr().Get(), int)
    if len(points) == 0 or len(flat) == 0: return np.empty((0, 3)), np.empty((0, 3), int), np.empty(0, int)
    if counts.sum() != len(flat) or flat.min() < 0 or flat.max() >= len(points):
        raise ValueError('Invalid USD face array: ' + str(prim.GetPath()))
    world = (np.c_[points, np.ones(len(points))] @ np.asarray(xform_cache.GetLocalToWorldTransform(prim), float))[:, :3]
    holes = set(m.GetHoleIndicesAttr().Get() or [])
    if np.all(counts == 3) and not holes:
        return world, flat.reshape(-1, 3), np.arange(len(counts))
    faces, face_ids, off = [], [], 0
    for i, n in enumerate(counts):
        face = flat[off:off+n]; off += n
        if i in holes: continue
        if n < 3: raise ValueError('USD face has fewer than three vertices')
        for k in range(1, n-1): faces.append([face[0], face[k], face[k+1]]); face_ids.append(i)
    return world, np.asarray(faces, int).reshape(-1, 3), np.asarray(face_ids, int)


def read_original(house):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(house['source_stage'])
    if not stage or stage.GetCompositionErrors(): raise ValueError('USD composition failed')
    unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if str(UsdGeom.GetStageUpAxis(stage)) != 'Z': raise ValueError('Unexpected original USD up axis')
    matrix = np.asarray(house['matrix'], float)
    xcache, ccache = UsdGeom.XformCache(), {}
    vertices, faces, colours, render_wall, render_ceiling = [], [], [], [], []
    instance_faces, categories, paths = defaultdict(list), {}, {}
    offset = 0; counts = Counter(); skipped = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != 'Mesh': continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            skipped += 1; continue
        world, ff, face_ids = mesh_arrays(prim, xcache)
        if not len(ff): continue
        canonical = transform_points(world * unit, matrix)
        path = str(prim.GetPath()); bits = path.split('/')
        token = bits[4] if len(bits) > 4 else bits[-1]
        category = re.sub(r'_\d+.*$', '', token).replace('_', ' ').lower()
        inst = '/'.join(bits[:5])
        categories[inst] = category; paths[inst] = path
        counts[category] += len(ff)
        col = material_colour(prim, ccache)
        cc = np.repeat(col[None, :], len(ff), axis=0)
        # Honour face material subsets when the original stage provides them.
        for sub in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Mesh(prim)):
            ids = np.asarray(sub.GetIndicesAttr().Get() or [], int)
            if len(ids): cc[np.isin(face_ids, ids)] = material_colour(sub.GetPrim(), ccache)
        faces.append(ff + offset); vertices.append(canonical.astype(np.float32)); colours.append(cc.astype(np.float32))
        # Rendering flags do not alter collision/ray geometry.
        render_wall.extend([category == 'wall'] * len(ff))
        render_ceiling.extend(['ceiling' in category] * len(ff))
        instance_faces[inst].append(ff + offset); offset += len(canonical)
    verts = np.concatenate(vertices); fs = np.concatenate(faces)
    instances, markers = [], []
    for i, (key, ff) in enumerate(instance_faces.items()):
        cat = categories[key]; ids = np.concatenate(ff); tris = verts[ids]
        rec = dict(instance_id=i, category=cat, role=classify_category(cat), region_id=-1,
                   triangles=tris, semantic_source=house['source_stage'], usd_instance=key)
        if rec['role'] == 'blocker': instances.append(rec)
        if 'door' in cat or 'stair' in cat or cat in ('steps', 'step'):
            xz = tris[:, :, [0, 2]].reshape(-1, 2)
            rec.update(centre_xz_m=xz.mean(axis=0).tolist(), extent_xz_m=np.ptp(xz, axis=0).tolist(),
                       height_range_m=[float(tris[:, :, 1].min()), float(tris[:, :, 1].max())])
            markers.append(rec)
    return SimpleNamespace(vertices=verts, faces=fs, colours=np.concatenate(colours),
                           wall=np.asarray(render_wall, bool), ceiling=np.asarray(render_ceiling, bool),
                           instances=instances, markers=markers,
                           receipt=dict(source_stage=house['source_stage'], matrix_source=house['matrix_source'],
                                        source_to_canonical=house['source_to_canonical'], stage_up_axis='Z', stage_meters_per_unit=unit,
                                        vertices=len(verts), triangles=len(fs), invisible_meshes_skipped=skipped,
                                        triangle_categories=dict(counts), mesh_transform='composed original USD row-vector world matrix, stage metre scale, recorded package column-vector canonical matrix',
                                        material_colour='authored linear base colour multiplied by mean decoded sRGB texture; face material subsets preserved',
                                        material_cache_entries=len(ccache), source_modified=False))
