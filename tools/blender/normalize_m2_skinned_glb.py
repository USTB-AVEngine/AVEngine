#!/usr/bin/env python3
"""Bake an armature object transform into a Habitat-oriented GLB candidate.

The source Rocketbox candidate carries a 0.01 armature ancestor scale that
stock Habitat applies to the render hierarchy but not to the generated URDF
link transforms.  This tool creates a derived, metre-scale candidate and
refuses the result unless sampled deformed world-space vertices are preserved
before the edit and after a clean GLB round trip.

It is a compiler normalisation step only.  A successful report does not grant
``canary_qualified`` status or replace deformation/contact/human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import bpy
import numpy as np


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples-per-action", type=int, default=9)
    return parser.parse_args(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_glb_json(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise RuntimeError(f"not a GLB: {path}")
    version, length = struct.unpack_from("<II", payload, 4)
    if version != 2 or length != len(payload):
        raise RuntimeError("GLB header/declared length mismatch")
    offset = 12
    documents: list[dict[str, Any]] = []
    while offset < len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(payload):
            raise RuntimeError("truncated GLB chunk")
        if chunk_type == 0x4E4F534A:
            documents.append(json.loads(payload[offset:end].decode("utf-8")))
        offset = end
    if offset != len(payload) or len(documents) != 1:
        raise RuntimeError("expected exactly one GLB JSON chunk")
    return documents[0]


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.actions):
        bpy.data.actions.remove(block)


def _import(path: Path) -> tuple[Any, Any]:
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB import failed: {result}")
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(
            f"expected one armature, found {[obj.name for obj in armatures]}"
        )
    armature = armatures[0]
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and any(
            modifier.type == "ARMATURE" and modifier.object == armature
            for modifier in obj.modifiers
        )
    ]
    if len(meshes) != 1:
        raise RuntimeError(
            f"expected one mesh skinned by {armature.name!r}, found {[obj.name for obj in meshes]}"
        )
    return armature, meshes[0]


def _required_actions(armature: Any) -> dict[str, Any]:
    actions: dict[str, Any] = {}
    for action in bpy.data.actions:
        lower = action.name.lower()
        semantic = "walk" if "walk" in lower else "idle" if "idle" in lower else None
        if semantic is None:
            continue
        if semantic in actions:
            raise RuntimeError(f"ambiguous {semantic} actions")
        actions[semantic] = action
    if set(actions) != {"walk", "idle"}:
        raise RuntimeError(f"required Walk/Idle actions not unique: {list(actions)}")
    armature.animation_data_create()
    return actions


def _sample_frames(action: Any, count: int) -> list[float]:
    if count < 3:
        raise RuntimeError("samples-per-action must be at least 3")
    start, end = (float(value) for value in action.frame_range)
    if not math.isfinite(start + end) or end <= start:
        raise RuntimeError(f"invalid action range: {action.name} {action.frame_range}")
    return np.linspace(start, end, count, dtype=np.float64).tolist()


def _evaluated_vertices(mesh: Any) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    temporary = evaluated.to_mesh()
    try:
        matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
        local = np.empty((len(temporary.vertices), 4), dtype=np.float64)
        for index, vertex in enumerate(temporary.vertices):
            local[index, :3] = vertex.co
            local[index, 3] = 1.0
        return (matrix @ local.T).T[:, :3]
    finally:
        evaluated.to_mesh_clear()


def _snapshots(
    armature: Any,
    mesh: Any,
    actions: dict[str, Any],
    sample_frames: dict[str, list[float]],
) -> dict[str, list[np.ndarray]]:
    snapshots: dict[str, list[np.ndarray]] = {}
    if armature.animation_data is None:
        armature.animation_data_create()
    for semantic, action in actions.items():
        armature.animation_data.action = action
        per_action: list[np.ndarray] = []
        for frame in sample_frames[semantic]:
            bpy.context.scene.frame_set(int(math.floor(frame)), subframe=frame % 1.0)
            bpy.context.view_layer.update()
            vertices = _evaluated_vertices(mesh)
            if not np.all(np.isfinite(vertices)):
                raise RuntimeError(f"non-finite deformed vertex in {semantic}@{frame}")
            per_action.append(vertices)
        snapshots[semantic] = per_action
    armature.animation_data.action = None
    return snapshots


def _maximum_error(
    expected: dict[str, list[np.ndarray]], actual: dict[str, list[np.ndarray]]
) -> float:
    maximum = 0.0
    if set(expected) != set(actual):
        raise RuntimeError("action snapshot keys changed")
    for semantic in sorted(expected):
        if len(expected[semantic]) != len(actual[semantic]):
            raise RuntimeError("action snapshot count changed")
        for left, right in zip(expected[semantic], actual[semantic], strict=True):
            if left.shape != right.shape:
                raise RuntimeError(
                    f"deformed vertex shape changed: {left.shape} != {right.shape}"
                )
            maximum = max(maximum, float(np.max(np.abs(left - right))))
    return maximum


def _action_manifest(
    actions: dict[str, Any], sample_frames: dict[str, list[float]]
) -> list[dict[str, Any]]:
    return [
        {
            "semantic": semantic,
            "blender_action": action.name,
            "frame_range": [float(value) for value in action.frame_range],
            "sampled_frames": sample_frames[semantic],
            "fcurve_count": len(action.fcurves),
        }
        for semantic, action in sorted(actions.items())
    ]


def _apply_armature_transform(armature: Any) -> dict[str, Any]:
    before = {
        "translation": list(armature.location),
        "rotation_mode": armature.rotation_mode,
        "rotation_quaternion": list(armature.rotation_quaternion),
        "scale": list(armature.scale),
        "matrix_world": np.asarray(armature.matrix_world, dtype=np.float64).tolist(),
    }
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    # Only the inherited 0.01 scale is incompatible with Habitat's separate
    # URDF/render loading path.  Preserve the authored actor translation and
    # rotation as explicit node TR instead of baking more than necessary.
    result = bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if "FINISHED" not in result:
        raise RuntimeError(f"armature transform apply failed: {result}")
    bpy.context.view_layer.update()
    after_matrix = np.asarray(armature.matrix_world, dtype=np.float64)
    if not np.allclose(np.asarray(armature.scale), np.ones(3), atol=1.0e-7):
        raise RuntimeError(f"armature scale did not become identity: {armature.scale}")
    return {
        "before": before,
        "after": {
            "translation": list(armature.location),
            "rotation_quaternion": list(armature.rotation_quaternion),
            "scale": list(armature.scale),
            "matrix_world": after_matrix.tolist(),
        },
    }


def _export(path: Path, armature: Any, mesh: Any) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_image_format="AUTO",
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"GLB export failed: {result}")


def _joint_ancestor_scale(document: dict[str, Any]) -> dict[str, Any]:
    skins = document.get("skins", [])
    nodes = document.get("nodes", [])
    if len(skins) != 1:
        raise RuntimeError("normalised GLB must have one skin")
    joints = set(skins[0].get("joints", []))
    parents: dict[int, int] = {}
    for parent, node in enumerate(nodes):
        for child in node.get("children", []):
            if child in parents:
                raise RuntimeError("normalised GLB node has multiple parents")
            parents[child] = parent
    roots = [joint for joint in joints if parents.get(joint) not in joints]
    if len(roots) != 1:
        raise RuntimeError(f"normalised skin has ambiguous roots: {roots}")
    chain: list[dict[str, Any]] = []
    cursor = parents.get(roots[0])
    maximum_scale_error = 0.0
    while cursor is not None:
        node = nodes[cursor]
        scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        error = float(np.max(np.abs(scale - 1.0)))
        maximum_scale_error = max(maximum_scale_error, error)
        chain.append(
            {"node": cursor, "name": node.get("name"), "scale": scale.tolist()}
        )
        cursor = parents.get(cursor)
    root_node = nodes[roots[0]]
    root_translation = np.asarray(
        root_node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    root_rotation = np.asarray(
        root_node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64
    )
    root_scale = np.asarray(root_node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    root_identity_error = max(
        float(np.max(np.abs(root_translation))),
        float(np.max(np.abs(root_scale - 1.0))),
        min(
            float(np.max(np.abs(root_rotation - [0.0, 0.0, 0.0, 1.0]))),
            float(np.max(np.abs(root_rotation + [0.0, 0.0, 0.0, 1.0]))),
        ),
    )
    return {
        "skin_root_node": roots[0],
        "skin_root_name": root_node.get("name"),
        "skin_root_translation": root_translation.tolist(),
        "skin_root_rotation_xyzw": root_rotation.tolist(),
        "skin_root_scale": root_scale.tolist(),
        "skin_root_identity_max_abs_error": root_identity_error,
        "ancestors": chain,
        "maximum_ancestor_scale_error": maximum_scale_error,
    }


def main() -> int:
    args = _arguments()
    source = args.input.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output == source:
        raise RuntimeError("output must differ from the immutable source")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    _clear_scene()
    armature, mesh = _import(source)
    actions = _required_actions(armature)
    frames = {
        semantic: _sample_frames(action, args.samples_per_action)
        for semantic, action in actions.items()
    }
    before = _snapshots(armature, mesh, actions, frames)
    transform = _apply_armature_transform(armature)
    after_apply = _snapshots(armature, mesh, actions, frames)
    apply_error = _maximum_error(before, after_apply)
    if apply_error > 5.0e-5:
        raise RuntimeError(
            f"armature transform changed sampled deformation: {apply_error:.9g} m"
        )
    _export(output, armature, mesh)

    _clear_scene()
    roundtrip_armature, roundtrip_mesh = _import(output)
    roundtrip_actions = _required_actions(roundtrip_armature)
    roundtrip = _snapshots(
        roundtrip_armature, roundtrip_mesh, roundtrip_actions, frames
    )
    roundtrip_error = _maximum_error(before, roundtrip)
    if roundtrip_error > 1.0e-4:
        raise RuntimeError(
            f"normalised GLB round trip changed sampled deformation: {roundtrip_error:.9g} m"
        )

    output_document = _read_glb_json(output)
    ancestor_scale = _joint_ancestor_scale(output_document)
    if ancestor_scale["maximum_ancestor_scale_error"] > 1.0e-7:
        raise RuntimeError(f"skin ancestor scale was not baked: {ancestor_scale}")
    mouth_actions = []
    for action in output_document.get("animations", []):
        for channel in action.get("channels", []):
            target = channel.get("target", {})
            node_index = target.get("node")
            name = (
                output_document["nodes"][node_index].get("name")
                if isinstance(node_index, int)
                else None
            )
            if isinstance(name, str) and "mouth" in name.lower():
                mouth_actions.append(
                    {
                        "action": action.get("name"),
                        "node": name,
                        "path": target.get("path"),
                    }
                )

    report = {
        "schema": "avengine_m2_skinned_glb_normalisation_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "input": {
            "path": str(source),
            "sha256": _sha256(source),
            "byte_size": source.stat().st_size,
        },
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "byte_size": output.stat().st_size,
        },
        "actions": _action_manifest(roundtrip_actions, frames),
        "transform": transform,
        "maximum_world_vertex_error_after_apply_m": apply_error,
        "maximum_world_vertex_error_after_glb_roundtrip_m": roundtrip_error,
        "skin_ancestor_scale": ancestor_scale,
        "mouth_animation_channels": mouth_actions,
        "notes": [
            "This report proves only transform baking and sampled GLB round-trip equivalence.",
            "Skin-root rebasing is a separate fail-closed GLB compiler step.",
            "Deformation/contact/provenance/Habitat playback and human mesh-animation alignment review remain required.",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "pass", "output": str(output), "report": str(report_path)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
