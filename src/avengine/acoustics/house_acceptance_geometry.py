"""Zero-surface diagnostics for meshes rejected by the native CPU ray engine.

This is an opt-in diagnostic view of already validated input arrays. It neither
changes raw datasets nor declares a repaired or production-qualified package.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import numpy as np


def valid_native_triangles(vertices, triangles):
    """Keep nonzero-area, distinct-index faces in the native float32 coordinates."""
    vertices = np.asarray(vertices, dtype="<f4")
    triangles = np.asarray(triangles)
    keep = np.empty(len(triangles), dtype=bool)
    for start in range(0, len(triangles), 200_000):
        end = min(start + 200_000, len(triangles))
        faces = triangles[start:end]
        points = vertices[faces]
        # Check both precisions: a native float32 cross product can vanish
        # although arithmetic promoted to double retains a tiny residue.
        cross32 = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
        points64 = points.astype(np.float64)
        cross64 = np.cross(points64[:, 1]-points64[:, 0], points64[:, 2]-points64[:, 0])
        repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
        keep[start:end] = ~repeated & np.any(cross32 != 0, axis=1) & np.any(cross64 != 0, axis=1)
    return keep


def diagnostic_scene_without_degenerate_faces(scene, output):
    """Keep input identities as provenance; verify the derived native upload.

    The existing renderer computes native geometry expectations from the derived
    objects. The source package stays unchanged. Callers must force review when
    this opt-in path is used.
    """
    objects, removed, counts, removals = [], {}, Counter(), []
    for item in scene.objects:
        keep = valid_native_triangles(item["vertices"], item["triangles"])
        invalid = np.flatnonzero(~keep)
        removed[item["object_id"]] = invalid
        if len(invalid):
            removals.append({"object_id": item["object_id"], "removed_faces": len(invalid),
                             "original_faces": len(keep)})
        if not np.any(keep):
            continue
        derived = dict(item)
        derived["triangles"] = np.ascontiguousarray(item["triangles"][keep], dtype="<u4")
        derived["triangle_material_ids"] = np.ascontiguousarray(item["triangle_material_ids"][keep], dtype="<u4")
        for material, count in zip(*np.unique(derived["triangle_material_ids"], return_counts=True), strict=True):
            counts[scene.material_categories[int(material)]] += int(count)
        objects.append(derived)
    if not objects:
        raise ValueError("No nondegenerate acoustic triangles remain")
    path = output / "diagnostic_omitted_faces.npz"
    np.savez_compressed(path, **removed)
    evidence = {"mode": "diagnostic_without_zero_surface_faces",
                "source_package_manifest": str(scene.manifest_path),
                "omitted_faces": sum(x["removed_faces"] for x in removals),
                "by_object": removals, "omitted_indices_path": str(path),
                "original_dataset_modified": False, "source_package_modified": False,
                "v6_geometry_identical": False, "production_qualified": False,
                "policy": "float32 repeated-index or exactly zero cross product only; no positive-area threshold, hole filling or material changes"}
    return replace(scene, objects=tuple(objects), triangle_count_by_material=dict(counts)), evidence
