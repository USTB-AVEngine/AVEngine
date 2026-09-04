"""Rigidly align a generated animal rig to a reviewed cardinal heading.

Image-to-3D systems may preserve anatomy while choosing an arbitrary yaw for
the exported scene.  TokenRig then produces a correct skeleton in that same
oblique frame, but downstream front/hind and left/right inference expects a
cardinal frame.  This stage rotates every scene root by one shared world-Z
matrix.  It never changes mesh topology, material, skeleton hierarchy, skin
weights, or animation data.

The source front yaw is deliberately review evidence, not guessed from a
breed name or bone name.  A future automatic head/tail classifier can produce
the same scalar without changing this contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import bpy
from mathutils import Matrix


SCHEMA = "avengine_generated_animal_heading_normalization_v1"
CARDINAL_YAWS = {
    "positive-x": 0.0,
    "positive-y": 90.0,
    "negative-x": 180.0,
    "negative-y": -90.0,
}


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviewed-source-front-yaw-deg", type=float, required=True)
    parser.add_argument(
        "--target-front-axis",
        choices=tuple(CARDINAL_YAWS),
        default="positive-x",
    )
    parser.add_argument("--review-evidence", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe {label}: {path}")
    return path


def require_new_output(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def scene_summary():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    skinned_meshes = [
        obj
        for obj in meshes
        if any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
    ]
    return {
        "mesh_count": len(meshes),
        "skinned_mesh_count": len(skinned_meshes),
        "skinned_meshes": sorted(obj.name for obj in skinned_meshes),
        "armature_count": len(armatures),
        "bone_count": sum(len(obj.data.bones) for obj in armatures),
        "material_count": len(bpy.data.materials),
        "image_count": len(bpy.data.images),
        "action_count": len(bpy.data.actions),
        "root_objects": sorted(
            obj.name for obj in bpy.context.scene.objects if obj.parent is None
        ),
    }


def main():
    args = parse_argv()
    source = require_input(args.input, "rigged animal GLB")
    evidence = require_input(args.review_evidence, "heading review evidence")
    output = require_new_output(args.output, "normalized GLB")
    manifest = require_new_output(args.manifest, "normalization manifest")
    if not math.isfinite(args.reviewed_source_front_yaw_deg):
        raise SystemExit("--reviewed-source-front-yaw-deg must be finite")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    before = scene_summary()
    if before["skinned_mesh_count"] != 1 or before["armature_count"] != 1:
        raise RuntimeError(
            "expected one skinned animal mesh and one armature; unskinned "
            f"helper meshes are allowed, got {before}"
        )
    if before["action_count"] != 0:
        raise RuntimeError("heading normalization must run before animation")

    target_yaw = CARDINAL_YAWS[args.target_front_axis]
    delta_yaw = target_yaw - args.reviewed_source_front_yaw_deg
    rotation = Matrix.Rotation(math.radians(delta_yaw), 4, "Z")
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    if not roots:
        raise RuntimeError("imported scene has no root objects")
    for root in roots:
        root.matrix_world = rotation @ root.matrix_world

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_all_vertex_colors=True,
        export_vertex_color="ACTIVE",
    )
    after = scene_summary()
    for key in (
        "mesh_count",
        "skinned_mesh_count",
        "armature_count",
        "bone_count",
        "material_count",
        "image_count",
        "action_count",
    ):
        if after[key] != before[key]:
            raise RuntimeError(f"rigid heading rotation changed {key}: {before} -> {after}")

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "technical_spike_only_pending_reaudit_and_animation_qa",
        "formal_dataset_registration_authorized": False,
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "review_evidence": {
            "path": str(evidence),
            "sha256": sha256_file(evidence),
        },
        "heading": {
            "reviewed_source_front_yaw_deg": args.reviewed_source_front_yaw_deg,
            "target_front_axis": args.target_front_axis,
            "target_front_yaw_deg": target_yaw,
            "applied_world_z_yaw_deg": delta_yaw,
            "policy": "single_rigid_world_rotation_for_every_scene_root",
        },
        "preservation_contract": {
            "mesh_topology_changed": False,
            "material_changed": False,
            "skeleton_hierarchy_changed": False,
            "skin_weights_changed": False,
            "animation_present_or_changed": False,
        },
        "scene_before": before,
        "scene_after": after,
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_ANIMAL_HEADING_NORMALIZATION_OK "
        f"delta_yaw_deg={delta_yaw:.6f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
