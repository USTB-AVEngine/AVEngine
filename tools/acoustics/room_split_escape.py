"""Original 2026-10-03 CPU ray-escape measurement and whole-house setup.

spherical_directions, ray_checks, alternate_listener and room_manifest are
verbatim from accept_houses.py (3414629). strict_zero_scene is verbatim from
train_room_acceptance.py (4664ee6). No ray, threshold, geometry or material
rule is changed. All external paths arrive through arguments.
"""
from __future__ import annotations

from collections import Counter
import collections
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def spherical_directions(n):
    k = np.arange(n)
    y = 1 - 2 * (k + .5) / n
    phi = k * np.pi * (3 - np.sqrt(5))
    return np.stack((np.sqrt(1 - y*y) * np.cos(phi), y, np.sqrt(1 - y*y) * np.sin(phi)), axis=1)

def ray_checks(context, origins, bounds, n=512):
    directions = spherical_directions(n)
    maximum = max(30., float(np.linalg.norm(bounds[1] - bounds[0])) * 2)
    result = []
    for room, role, origin in origins:
        escaped, near = [], 0
        for i, direction in enumerate(directions):
            hit = context.trace_ray_first_hit(list(origin), direction.tolist(), .01, maximum)
            # A raw geometric escape is never explained away as a semantic window.
            if not hit.hit:
                endpoint = np.asarray(origin) + maximum * direction
                positive = [(bounds[side, ax] - origin[ax]) / direction[ax]
                            for ax in range(3) if abs(direction[ax]) > 1e-10
                            for side in (0, 1)
                            if (bounds[side, ax] - origin[ax]) / direction[ax] > 0]
                shell = np.asarray(origin) + (min(positive) if positive else maximum) * direction
                escaped.append({"direction_index": i, "direction": direction.tolist(),
                                "aabb_exit_m": shell.tolist(), "end_m": endpoint.tolist()})
            elif hit.distance < .05:
                near += 1
        result.append({"room_label": room, "role": role, "origin_m": list(origin),
                       "ray_count": n, "escape_count": len(escaped), "escape_fraction": len(escaped) / n,
                       "near_surface_hits": near, "escaped_rays": escaped,
                       "hole_localization": "intersection with scan AABB is a locator, not a recovered hole boundary"})
    return result

def alternate_listener(placement):
    listener = np.asarray(placement["camera_m"],float)
    delta = np.asarray(placement["source_1_m"],float)-listener
    delta[1]=0.
    delta /= np.linalg.norm(delta)
    return (listener+.4*delta).tolist()

def room_manifest(h, dataset_config):
    # Only change the concrete external assets in v6's room manifest schema.
    sid = h["scan_id"]
    scene = str(Path(h["scene_directory"]) / (sid + ".glb"))
    return {"schema": "avengine_room_package_v1", "room_id": h["house"], "room_kind": "habitat_native",
            "geometry_representation": "real_surface_mesh",
            "coordinate_system": {"handedness": "right", "linear_unit": "meter", "up_axis": "+Y",
                                  "forward_axis": "-Z", "quaternion_order": "xyzw"},
            "assets": [{"role": role, "path": path, "license": h["family"].upper() + " dataset terms",
                        "redistribution": "external_test_asset_not_committed"}
                       for role, path in (("render_surface_mesh", scene), ("semantic_surface_mesh", h["semantic_source"]),
                                          ("semantic_descriptor", h["annotation_source"]), ("navmesh", h["navmesh_source"]), ("scene_dataset_config", str(dataset_config)))],
            "scene": {"scene_id": scene, "scene_id_kind": "path", "enable_physics": False,
                      "dataset_config_path": str(dataset_config),
                      "load_semantic_mesh": True, "navmesh_path": h["navmesh_source"], "navmesh_policy": "load_declared"},
            "navigation": {"agent_height_m": 1.5, "agent_radius_m": .1, "include_static_objects": False},
            "connectivity_pairs": [{"pair_id": "declared_source_witness_pair_not_recomputed",
                                    "start_m": [h["acoustic_rooms"][0]["placement"]["source_1_m"][0],
                                                h["acoustic_rooms"][0]["placement"]["source_1_m"][1]-1.2,
                                                h["acoustic_rooms"][0]["placement"]["source_1_m"][2]],
                                    "end_m": [h["acoustic_rooms"][0]["placement"]["source_2_m"][0],
                                              h["acoustic_rooms"][0]["placement"]["source_2_m"][1]-1.2,
                                              h["acoustic_rooms"][0]["placement"]["source_2_m"][2]]}],
            "ray_checks": [], "openings": [],
            "acoustics": {"status": "deferred_to_m3", "reason": "RLR research package is compiled below"},
            "provenance": {"source": h["registry_source"], "source_revision": "existing dataset source; CPU audit"},
            "semantics": {"interpretation": "same v6 semantic mesh parser and canonicalization"}}

def strict_zero_scene(scene,destination):
    """Derive runtime-owned arrays; retain every positive double-area face."""
    objects=[];records=[];omitted={};counts=collections.Counter()
    for obj in scene.objects:
        faces=np.asarray(obj['triangles']);v=np.asarray(obj['vertices'],dtype=np.float64)
        p=v[faces];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);mask=np.all(cross==0,axis=1)
        # Preserve every positive-area face and let the unchanged native
        # uploader decide. A float32 cross product can cancel even when the
        # native float64 product of float32 edge vectors remains nonzero.
        if mask.any():
            omitted[obj['object_id']]=np.flatnonzero(mask)
            records.append(dict(object_id=obj['object_id'],original_faces=len(faces),removed_faces=int(mask.sum()),maximum_removed_area_m2=0.0))
        if mask.all():continue
        derived=dict(obj);derived['triangles']=np.ascontiguousarray(faces[~mask],dtype='<u4')
        derived['triangle_material_ids']=np.ascontiguousarray(obj['triangle_material_ids'][~mask],dtype='<u4')
        objects.append(derived)
        for mat,n in zip(*np.unique(derived['triangle_material_ids'],return_counts=True),strict=True):counts[scene.material_categories[int(mat)]]+=int(n)
    if not objects:raise ValueError('No positive-area acoustic faces')
    receipt=dict(source_package_manifest=str(scene.manifest_path),removed_faces=sum(x['removed_faces'] for x in records),by_object=records,maximum_removed_area_m2=0.0,tolerance_m2=0.0,original_package_modified=False,objects_vertices_and_materials_unchanged=True,operation='派生几何：清退化三角形',qualification='authorized CPU training selection; not a new formally compiled production package')
    if receipt['removed_faces']:
        destination.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(destination/'removed_faces.npz',**omitted)
        # Persist exact derived native inputs rather than only an in-memory view.
        arrays={}
        for i,obj in enumerate(objects):
            for field in ('vertices','triangles','triangle_material_ids'):arrays[f'object_{i}_{field}']=np.asarray(obj[field])
        np.savez_compressed(destination/'native_arrays.npz',**arrays)
        receipt['native_arrays_path']=str(destination/'native_arrays.npz')
        receipt['omitted_indices_path']=str(destination/'removed_faces.npz')
        save(destination/'receipt.json',receipt)
    return replace(scene,objects=tuple(objects),triangle_count_by_material=dict(counts)),receipt
