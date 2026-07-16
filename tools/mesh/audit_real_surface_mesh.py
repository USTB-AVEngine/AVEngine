"""Audit a glTF/GLB as evidence for the M1 real-surface geometry gate.

Run this script with Blender rather than the system Python so that glTF scene
nodes, transforms, and mesh instances are evaluated consistently::

    blender --background --factory-startup --python-exit-code 2 \
      --python tools/mesh/audit_real_surface_mesh.py -- \
      --input path/to/scene.glb --output path/to/mesh_audit.json

The input asset is never modified.  The gate is deliberately conservative:
the known 252-triangle legacy bounds export, an asset below the configured
complexity floor, or an asset made exclusively from simple box-topology mesh
nodes is not accepted as proof of real render-surface geometry.  A failed gate
still writes its JSON report before raising unless ``--allow-gate-failure`` is
provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import bpy


KNOWN_LEGACY_AABB_TRIANGLE_COUNT = 252
DEFAULT_MINIMUM_TRIANGLES = 1_000
POSITION_DECIMALS = 7


def parse_args() -> argparse.Namespace:
    script_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Audit GLB/glTF geometry without modifying the source asset."
    )
    parser.add_argument("--input", required=True, help="Input .glb or .gltf scene")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument(
        "--minimum-triangles",
        type=int,
        default=DEFAULT_MINIMUM_TRIANGLES,
        help=(
            "Minimum expanded scene triangle count for the real-surface gate "
            f"(default: {DEFAULT_MINIMUM_TRIANGLES})"
        ),
    )
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help="Write a failing audit without making the Blender script fail",
    )
    args = parser.parse_args(script_args)
    if args.minimum_triangles <= KNOWN_LEGACY_AABB_TRIANGLE_COUNT:
        parser.error(
            "--minimum-triangles must be greater than the known 252-triangle "
            "legacy AABB proxy signature"
        )
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rounded_position(values: Iterable[float]) -> tuple[float, float, float]:
    return tuple(round(float(value), POSITION_DECIMALS) for value in values)


def gltf_position(
    blender_world_position: Iterable[float],
) -> tuple[float, float, float]:
    """Convert Blender world (+Z up) back to glTF world (+Y up)."""

    x, y, z = blender_world_position
    return (float(x), float(z), float(-y))


def bounds_record(
    positions: list[tuple[float, float, float]], coordinate_space: str
) -> dict:
    if not positions:
        return {
            "coordinate_space": coordinate_space,
            "min": None,
            "max": None,
            "extent": None,
            "diagonal": None,
        }
    minimum = [min(position[axis] for position in positions) for axis in range(3)]
    maximum = [max(position[axis] for position in positions) for axis in range(3)]
    extent = [maximum[axis] - minimum[axis] for axis in range(3)]
    return {
        "coordinate_space": coordinate_space,
        "min": minimum,
        "max": maximum,
        "extent": extent,
        "diagonal": math.sqrt(sum(value * value for value in extent)),
    }


def is_simple_box_topology(mesh: bpy.types.Mesh) -> bool:
    """Return whether one evaluated mesh is topologically a rectangular box.

    glTF commonly duplicates cube vertices across face normals, so this test
    compares unique positions rather than Blender vertex indices.
    """

    if len(mesh.loop_triangles) != 12:
        return False
    positions = {rounded_position(vertex.co) for vertex in mesh.vertices}
    if len(positions) != 8:
        return False
    return all(
        len({position[axis] for position in positions}) == 2 for axis in range(3)
    )


def reset_scene() -> None:
    # This only clears Blender's transient audit scene; it does not touch the
    # source glTF/GLB or any project file on disk.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def import_gltf(path: Path) -> None:
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender glTF importer did not finish: {sorted(result)}")


def audit_scene(input_path: Path, minimum_triangles: int) -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    nodes = sorted(bpy.context.scene.objects, key=lambda obj: obj.name_full)
    mesh_nodes = [node for node in nodes if node.type == "MESH"]
    node_types = Counter(node.type for node in nodes)
    material_names: set[str] = set()
    blender_positions: list[tuple[float, float, float]] = []
    mesh_records = []
    vertex_count = 0
    triangle_count = 0
    box_topology_count = 0

    for node in mesh_nodes:
        evaluated = node.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        if mesh is None:
            raise RuntimeError(f"Unable to evaluate mesh node {node.name_full!r}")
        try:
            mesh.calc_loop_triangles()
            vertices = len(mesh.vertices)
            triangles = len(mesh.loop_triangles)
            vertex_count += vertices
            triangle_count += triangles
            box_topology = is_simple_box_topology(mesh)
            box_topology_count += int(box_topology)
            for material in mesh.materials:
                if material is not None:
                    material_names.add(material.name_full)
            world_matrix = evaluated.matrix_world
            positions = [tuple(world_matrix @ vertex.co) for vertex in mesh.vertices]
            blender_positions.extend(positions)
            mesh_records.append(
                {
                    "node": node.name_full,
                    "mesh_datablock": node.data.name_full,
                    "vertices": vertices,
                    "triangles": triangles,
                    "simple_box_topology": box_topology,
                    "bounds": bounds_record(
                        [gltf_position(position) for position in positions],
                        "glTF world, right-handed, +Y up, metres as encoded",
                    ),
                }
            )
        finally:
            evaluated.to_mesh_clear()

    gltf_positions = [gltf_position(position) for position in blender_positions]
    all_mesh_nodes_are_boxes = bool(mesh_nodes) and box_topology_count == len(
        mesh_nodes
    )
    gate_reasons = []
    if not mesh_nodes or triangle_count == 0:
        gate_reasons.append("scene_has_no_triangle_mesh_surface")
    if triangle_count == KNOWN_LEGACY_AABB_TRIANGLE_COUNT:
        gate_reasons.append("known_legacy_252_triangle_aabb_proxy_signature")
    if triangle_count < minimum_triangles:
        gate_reasons.append(
            f"triangle_count_{triangle_count}_below_minimum_{minimum_triangles}"
        )
    if all_mesh_nodes_are_boxes:
        gate_reasons.append(
            "all_mesh_nodes_have_simple_box_topology_and_cannot_prove_render_surfaces"
        )

    return {
        "schema": "avengine_real_surface_mesh_audit_v1",
        "input": str(input_path),
        "vertices": vertex_count,
        "triangles": triangle_count,
        "meshes": len(mesh_nodes),
        "materials": len(material_names),
        "nodes": len(nodes),
        "bounds": bounds_record(
            gltf_positions, "glTF world, right-handed, +Y up, metres as encoded"
        ),
        "sha256": file_sha256(input_path),
        "bytes": input_path.stat().st_size,
        "details": {
            "counting_semantics": (
                "evaluated mesh-node instances; shared meshes are counted once per node"
            ),
            "node_types": dict(sorted(node_types.items())),
            "unique_mesh_datablocks": len({node.data.name_full for node in mesh_nodes}),
            "material_names": sorted(material_names),
            "mesh_breakdown": mesh_records,
            "blender_world_bounds": bounds_record(
                blender_positions, "Blender transient audit world, right-handed, +Z up"
            ),
        },
        "aabb_proxy_indicators": {
            "known_legacy_triangle_signature": (
                triangle_count == KNOWN_LEGACY_AABB_TRIANGLE_COUNT
            ),
            "known_legacy_triangle_count": KNOWN_LEGACY_AABB_TRIANGLE_COUNT,
            "simple_box_topology_meshes": box_topology_count,
            "all_mesh_nodes_are_simple_boxes": all_mesh_nodes_are_boxes,
            "note": (
                "Bounds are reported only as measurements; they are never emitted or "
                "accepted as replacement surface geometry."
            ),
        },
        "real_surface_gate": {
            "status": "fail" if gate_reasons else "pass",
            "minimum_triangles": minimum_triangles,
            "rejects_known_252_triangle_aabb_proxy": True,
            "rejects_all_simple_box_topology_scenes": True,
            "reasons": gate_reasons,
        },
    }


def write_report(report: dict, output_path: Path | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded, encoding="utf-8", newline="\n")
    # Blender writes its own status messages to stdout, so a report file is the
    # machine-readable interface.  This compact line remains useful interactively.
    print("AVENGINE_MESH_AUDIT_JSON=" + json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() not in {".glb", ".gltf"}:
        raise ValueError(f"Expected a .glb or .gltf input, got {input_path.name!r}")

    output_path = Path(args.output).resolve() if args.output else None
    reset_scene()
    import_gltf(input_path)
    report = audit_scene(input_path, args.minimum_triangles)
    write_report(report, output_path)
    if report["real_surface_gate"]["status"] != "pass" and not args.allow_gate_failure:
        reasons = ", ".join(report["real_surface_gate"]["reasons"])
        raise RuntimeError(f"Real-surface mesh gate failed: {reasons}")


if __name__ == "__main__":
    main()
