"""Locate authored-GLB topology roots and optionally derive a fresh acoustic proxy.

The source GLB and its original QA result are never modified. Duplicate
triangles are removed only in a separately named derived GLB, with an
explanation of every removed source object/material group.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping

import numpy as np

from avengine.acoustics.gltf import extract_triangle_scene


def _coord_key(value: np.ndarray) -> bytes:
    return np.asarray(value, dtype="<f4").tobytes()


def _triangle_key(vertices: np.ndarray, triangle: np.ndarray) -> tuple[bytes, ...]:
    return tuple(sorted(_coord_key(vertices[int(index)]) for index in triangle))


def _object_spans(objects: Iterable[Mapping[str, Any]]) -> list[tuple[int, int, str]]:
    return [
        (
            int(item["triangle_offset"]),
            int(item["triangle_offset"]) + int(item["triangle_count"]),
            str(item["object_id"]),
        )
        for item in objects
    ]


def _object_for_triangle(spans: list[tuple[int, int, str]], index: int) -> str:
    for start, stop, object_id in spans:
        if start <= index < stop:
            return object_id
    raise ValueError(f"triangle index is outside source object partitions: {index}")


def _root_cause_report(
    scene: Any, source_path: Path
) -> tuple[dict[str, Any], list[int], list[int]]:
    spans = _object_spans(scene.objects)
    duplicate_groups: dict[tuple[bytes, ...], list[int]] = defaultdict(list)
    edge_groups: dict[tuple[bytes, bytes], list[int]] = defaultdict(list)
    triangle_keys: list[tuple[bytes, ...]] = []
    for index, triangle in enumerate(scene.triangles):
        key = _triangle_key(scene.vertices, triangle)
        triangle_keys.append(key)
        duplicate_groups[key].append(index)
        coords = [_coord_key(scene.vertices[int(vertex)]) for vertex in triangle]
        for first, second in (
            (coords[0], coords[1]),
            (coords[1], coords[2]),
            (coords[2], coords[0]),
        ):
            edge_groups[tuple(sorted((first, second)))].append(index)
    material_names = list(scene.triangle_source_material_names)
    quantized = np.round(scene.vertices.astype(np.float64), 6)
    zero_area_indices = []
    for index, triangle in enumerate(scene.triangles):
        points = quantized[triangle]
        cross = np.cross(points[1] - points[0], points[2] - points[0])
        if float(np.dot(cross, cross)) <= 1.0e-20:
            zero_area_indices.append(index)
    zero_area_records = [
        {
            "triangle_index": int(index),
            "object_id": _object_for_triangle(spans, index),
            "source_material_name": str(material_names[index]),
            "root_cause": "zero_area_after_native_six_decimal_quantization",
        }
        for index in zero_area_indices
    ]
    duplicate_records: list[dict[str, Any]] = []
    duplicate_indices: list[int] = []
    for occurrences in duplicate_groups.values():
        if len(occurrences) < 2:
            continue
        duplicate_indices.extend(occurrences[1:])
        records = [
            {
                "triangle_index": int(index),
                "object_id": _object_for_triangle(spans, index),
                "source_material_name": str(material_names[index]),
            }
            for index in occurrences
        ]
        duplicate_records.append(
            {
                "occurrence_count": len(records),
                "objects": sorted({item["object_id"] for item in records}),
                "materials": sorted({item["source_material_name"] for item in records}),
                "triangles": records,
                "root_cause": (
                    "same_surface_triangle_repeated within one object"
                    if len({item["object_id"] for item in records}) == 1
                    else "coincident_surface_triangles across authored objects"
                ),
            }
        )
    nonmanifold_records: list[dict[str, Any]] = []
    for occurrences in edge_groups.values():
        if len(occurrences) <= 2:
            continue
        records = [
            {
                "triangle_index": int(index),
                "object_id": _object_for_triangle(spans, index),
                "source_material_name": str(material_names[index]),
            }
            for index in occurrences
        ]
        nonmanifold_records.append(
            {
                "occurrence_count": len(records),
                "objects": sorted({item["object_id"] for item in records}),
                "materials": sorted({item["source_material_name"] for item in records}),
                "triangles": records,
                "root_cause": (
                    "inter_object_junction_or_overlap"
                    if len({item["object_id"] for item in records}) > 1
                    else "same_object_nonmanifold_edge"
                ),
            }
        )
    duplicate_records.sort(key=lambda item: (-item["occurrence_count"], item["triangles"][0]["triangle_index"]))
    nonmanifold_records.sort(key=lambda item: (-item["occurrence_count"], item["triangles"][0]["triangle_index"]))
    report = {
        "schema": "avengine_authored_geometry_qa_root_cause_v1",
        "status": "pass",
        "source_glb": str(source_path),
        "source_glb_sha256": scene.source_sha256,
        "source_vertex_count": int(len(scene.vertices)),
        "source_triangle_count": int(len(scene.triangles)),
        "source_object_count": len(scene.objects),
        "duplicate_triangle_count": sum(len(item["triangles"]) - 1 for item in duplicate_records),
        "duplicate_group_count": len(duplicate_records),
        "nonmanifold_edge_group_count": len(nonmanifold_records),
        "zero_area_after_six_decimal_count": len(zero_area_indices),
        "duplicate_groups": duplicate_records[:200],
        "nonmanifold_edge_groups": nonmanifold_records[:200],
        "zero_area_triangles": zero_area_records[:200],
        "claim_boundary": (
            "Object-level topology diagnosis only; does not alter the source GLB "
            "or formal M3 QA result."
        ),
    }
    return report, sorted(set(duplicate_indices)), sorted(set(zero_area_indices))


def _write_glb(scene: Any, kept_indices: list[int], output: Path, source_path: Path) -> dict[str, Any]:
    spans = _object_spans(scene.objects)
    material_names = sorted({str(scene.triangle_source_material_names[index]) for index in kept_indices})
    material_index = {name: index for index, name in enumerate(material_names)}
    groups: OrderedDict[tuple[str, str], list[np.ndarray]] = OrderedDict()
    for index in kept_indices:
        object_id = _object_for_triangle(spans, index)
        material_name = str(scene.triangle_source_material_names[index])
        groups.setdefault((object_id, material_name), []).append(scene.triangles[index])
    binary = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []

    def append_bytes(payload: bytes, target: int) -> int:
        while len(binary) % 4:
            binary.extend(b"\x00")
        offset = len(binary)
        binary.extend(payload)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(payload),
                "target": target,
            }
        )
        return len(buffer_views) - 1

    for group_index, ((object_id, material_name), triangles) in enumerate(groups.items()):
        source_triangles = np.asarray(triangles, dtype="<u4")
        unique_vertices = np.unique(source_triangles.reshape(-1))
        local_map = {int(value): index for index, value in enumerate(unique_vertices)}
        local_indices = np.asarray(
            [[local_map[int(value)] for value in triangle] for triangle in source_triangles],
            dtype="<u4",
        ).reshape(-1)
        local_vertices = np.asarray(scene.vertices[unique_vertices], dtype="<f4")
        position_view = append_bytes(local_vertices.tobytes(), 34962)
        index_view = append_bytes(local_indices.tobytes(), 34963)
        position_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": position_view,
                "componentType": 5126,
                "count": len(local_vertices),
                "type": "VEC3",
                "min": local_vertices.min(axis=0).astype(float).tolist(),
                "max": local_vertices.max(axis=0).astype(float).tolist(),
            }
        )
        index_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": index_view,
                "componentType": 5125,
                "count": len(local_indices),
                "type": "SCALAR",
            }
        )
        meshes.append(
            {
                "name": f"proxy_{group_index:04d}_{object_id}",
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor},
                        "indices": index_accessor,
                        "material": material_index[material_name],
                        "mode": 4,
                        "extras": {
                            "source_object_id": object_id,
                            "source_material_name": material_name,
                        },
                    }
                ],
                "extras": {
                    "source_object_id": object_id,
                    "source_material_name": material_name,
                },
            }
        )
        nodes.append({"mesh": group_index, "name": f"proxy_node_{group_index:04d}"})
        group_records.append(
            {
                "source_object_id": object_id,
                "source_material_name": material_name,
                "triangle_count": int(len(source_triangles)),
                "vertex_count": int(len(local_vertices)),
            }
        )
    document = {
        "asset": {"version": "2.0", "generator": "AVEngine authored acoustic proxy helper"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.6, 0.6, 0.6, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
            }
            for name in material_names
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_payload = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    while len(json_payload) % 4:
        json_payload += b" "
    while len(binary) % 4:
        binary.extend(b"\x00")
    payload = bytearray(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_payload) + 8 + len(binary)))
    payload.extend(struct.pack("<II", len(json_payload), 0x4E4F534A))
    payload.extend(json_payload)
    payload.extend(struct.pack("<II", len(binary), 0x004E4942))
    payload.extend(binary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "schema": "avengine_authored_geometry_deduplicated_proxy_v1",
        "status": "pass",
        "source_glb": str(source_path),
        "source_glb_sha256": scene.source_sha256,
        "derived_glb": str(output),
        "derived_glb_sha256": hashlib.sha256(payload).hexdigest(),
        "source_triangle_count": int(len(scene.triangles)),
        "derived_triangle_count": int(len(kept_indices)),
        "removed_triangle_count": int(len(scene.triangles) - len(kept_indices)),
        "material_names": material_names,
        "group_count": len(group_records),
        "groups": group_records,
        "claim_boundary": (
            "Fresh derived acoustic research proxy with duplicate coordinate-identical "
            "triangles removed; original GLB and original QA remain untouched."
        ),
    }


def derive(source_glb: str | Path, output_glb: str | Path, report_path: str | Path) -> dict[str, Any]:
    output = Path(output_glb).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace proxy GLB: {output}")
    if report_file.exists() or report_file.is_symlink():
        raise FileExistsError(f"refusing to replace root-cause report: {report_file}")
    source = Path(source_glb).expanduser().resolve()
    scene = extract_triangle_scene(source)
    root_report, duplicate_indices, zero_area_indices = _root_cause_report(
        scene, source
    )
    removed_set = set(duplicate_indices) | set(zero_area_indices)
    kept = [index for index in range(len(scene.triangles)) if index not in removed_set]
    proxy_report = _write_glb(scene, kept, output, source)
    proxy_report["removed_duplicate_triangle_count"] = len(
        set(duplicate_indices) - set(zero_area_indices)
    )
    proxy_report["removed_zero_area_triangle_count"] = len(
        set(zero_area_indices)
    )
    report = {
        **root_report,
        "derived_proxy": proxy_report,
        "status": "pass",
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glb", required=True, type=Path)
    parser.add_argument("--output-glb", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = derive(args.source_glb, args.output_glb, args.report)
    print(json.dumps({
        "status": report["status"],
        "source_glb": report["source_glb"],
        "derived_glb": report["derived_proxy"]["derived_glb"],
        "source_triangle_count": report["source_triangle_count"],
        "derived_triangle_count": report["derived_proxy"]["derived_triangle_count"],
        "duplicate_group_count": report["duplicate_group_count"],
        "nonmanifold_edge_group_count": report["nonmanifold_edge_group_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
