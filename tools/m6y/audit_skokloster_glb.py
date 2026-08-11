#!/usr/bin/env python3
"""Audit the exact Habitat Skokloster GLB for visual and acoustic staging."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from avengine.m3.gltf import extract_triangle_scene, load_glb
from avengine.m3.qa import geometry_report, triangle_areas


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _node_local_matrix(node: Mapping[str, Any]) -> np.ndarray:
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=np.float64)
        _require(values.shape == (16,), "node matrix must have 16 values")
        return values.reshape((4, 4), order="F")
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=float)
    rotation = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=float)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=float)
    _require(
        translation.shape == (3,) and rotation.shape == (4,) and scale.shape == (3,),
        "node TRS shape drift",
    )
    x, y, z, w = rotation / np.linalg.norm(rotation)
    rotation_matrix = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _node_world_matrices(document: Mapping[str, Any]) -> list[np.ndarray]:
    nodes = document.get("nodes")
    _require(isinstance(nodes, list), "GLB nodes must be an array")
    parents: list[int | None] = [None] * len(nodes)
    for parent_index, value in enumerate(nodes):
        _require(isinstance(value, Mapping), "GLB node must be an object")
        for child in value.get("children", []):
            child_index = int(child)
            _require(
                0 <= child_index < len(nodes) and parents[child_index] is None,
                "GLB node graph is invalid",
            )
            parents[child_index] = parent_index
    results: list[np.ndarray | None] = [None] * len(nodes)

    def world(index: int) -> np.ndarray:
        existing = results[index]
        if existing is not None:
            return existing
        local = _node_local_matrix(nodes[index])
        parent = parents[index]
        result = local if parent is None else world(parent) @ local
        results[index] = result
        return result

    return [world(index) for index in range(len(nodes))]


def audit(glb_path: Path, navmesh_path: Path) -> dict[str, Any]:
    glb_path = glb_path.resolve()
    navmesh_path = navmesh_path.resolve()
    _require(glb_path.is_file(), f"GLB is missing: {glb_path}")
    _require(navmesh_path.is_file(), f"navmesh is missing: {navmesh_path}")
    loaded = load_glb(glb_path)
    document = loaded.document
    scene = extract_triangle_scene(glb_path)
    # The container is glTF, but this particular legacy Habitat asset stores
    # Z-up/+Y-front coordinates in POSITION.  The navmesh and Habitat state
    # files prove the exact conversion below; treating it as ordinary glTF
    # Y-up moves the castle onto the wrong axes.
    source_to_habitat_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _require(
        math.isclose(
            float(np.linalg.det(source_to_habitat_matrix[:3, :3])),
            1.0,
            abs_tol=1.0e-12,
        ),
        "source-to-Habitat conversion must preserve winding",
    )
    habitat_vertices = np.ascontiguousarray(
        scene.vertices.astype(np.float64, copy=False)
        @ source_to_habitat_matrix[:3, :3].T,
        dtype="<f4",
    )
    source_to_habitat = {
        "matrix_row_major": source_to_habitat_matrix.reshape(-1).tolist(),
        "source": (
            "Legacy Habitat test-scene POSITION data in metres, Z up and +Y "
            "front, stored in a glTF 2.0 container"
        ),
        "reviewed": True,
    }
    report = geometry_report(
        habitat_vertices,
        scene.triangles,
        source_sha256=scene.source_sha256,
        representation="habitat_test_scene_real_surface_glb",
        source_to_canonical=source_to_habitat,
        objects=scene.objects,
    )
    areas = triangle_areas(habitat_vertices, scene.triangles)
    diagonal = report["bounds_m"]["diagonal"]
    area_threshold = max(1.0e-14, diagonal * diagonal * 1.0e-14)
    zero_indices = np.flatnonzero(areas <= area_threshold)
    material_names = np.asarray(scene.triangle_source_material_names, dtype=object)
    zero_records = []
    for triangle_index in zero_indices:
        owning = next(
            item
            for item in scene.objects
            if item["triangle_offset"]
            <= triangle_index
            < item["triangle_offset"] + item["triangle_count"]
        )
        zero_records.append(
            {
                "triangle_index": int(triangle_index),
                "area_m2": float(areas[triangle_index]),
                "indices": scene.triangles[triangle_index].astype(int).tolist(),
                "source_material_name": str(material_names[triangle_index]),
                "object_id": owning["object_id"],
            }
        )
    worlds = _node_world_matrices(document)
    mesh_nodes = [
        (index, int(node["mesh"]), worlds[index])
        for index, node in enumerate(document.get("nodes", []))
        if isinstance(node, Mapping) and "mesh" in node
    ]
    identity = np.eye(4)
    nonidentity = [
        {
            "node_index": node_index,
            "mesh_index": mesh_index,
            "world_matrix_row_major": matrix.reshape(-1).tolist(),
        }
        for node_index, mesh_index, matrix in mesh_nodes
        if not np.allclose(matrix, identity, atol=1.0e-12, rtol=0.0)
    ]
    _require(
        len(zero_records) == 2, "source does not contain exactly two zero-area faces"
    )
    return {
        "schema": "avengine_skokloster_glb_navmesh_audit_v1",
        "status": "pass",
        "scene_id": "skokloster-castle",
        "source_files": {
            "glb": {"path": str(glb_path), "byte_size": glb_path.stat().st_size},
            "navmesh": {
                "path": str(navmesh_path),
                "byte_size": navmesh_path.stat().st_size,
            },
        },
        "glb": {
            "version": 2,
            "asset": document.get("asset"),
            "default_scene": document.get("scene", 0),
            "counts": {
                field: len(document.get(field, []))
                for field in (
                    "scenes",
                    "nodes",
                    "meshes",
                    "materials",
                    "textures",
                    "images",
                    "accessors",
                    "bufferViews",
                )
            },
            "extensions_used": document.get("extensionsUsed", []),
            "extensions_required": document.get("extensionsRequired", []),
            "mesh_node_count": len(mesh_nodes),
            "nonidentity_mesh_node_world_transform_count": len(nonidentity),
            "nonidentity_mesh_node_world_transforms": nonidentity,
        },
        "coordinate_contract": {
            "source": "legacy Habitat test-scene POSITION, metres, Z up, +Y front",
            "habitat": "right-handed metres, +Y up, -Z forward",
            "source_to_habitat": "H=(S.x,S.z,-S.y)",
            "source_to_habitat_matrix_row_major": source_to_habitat["matrix_row_major"],
            "habitat_to_ue_cm": "U_cm=(100*H.x,100*H.z,100*H.y)",
            "ue_interchange_note": (
                "Bake H=(S.x,S.z,-S.y) into a prepared canonical glTF before "
                "UE Interchange. Interchange then owns canonical glTF Y-up to UE "
                "conversion; runtime actors remain at identity."
            ),
        },
        "expanded_scene": {
            "vertex_count": len(scene.vertices),
            "triangle_count": len(scene.triangles),
            "object_count": len(scene.objects),
            "source_primitive_count": scene.source_primitive_count,
            "source_node_instance_count": scene.source_node_instance_count,
            "material_names": sorted(set(scene.triangle_source_material_names)),
            "source_bounds_m": {
                "min": scene.vertices.min(axis=0).astype(float).tolist(),
                "max": scene.vertices.max(axis=0).astype(float).tolist(),
            },
            "habitat_bounds_m": report["bounds_m"],
        },
        "raw_topology": {
            "status": report["status"],
            "area_threshold_m2_inclusive": area_threshold,
            "zero_area_triangle_count": len(zero_records),
            "zero_area_triangles": zero_records,
            **report["topology"],
        },
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True, type=Path)
    parser.add_argument("--navmesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(args.glb, args.navmesh)
    _require(not args.output.exists(), f"refusing to replace output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "SKOKLOSTER_GLB_AUDIT_OK "
        f"triangles={result['expanded_scene']['triangle_count']} zero_area=2",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
