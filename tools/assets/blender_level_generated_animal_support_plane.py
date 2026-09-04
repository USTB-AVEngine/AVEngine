"""Level a generated quadruped from two independent visible-foot authorities.

Image-to-3D output can be anatomically usable while the complete animal is
exported with a non-zero pitch or roll.  After heading normalization and
target-native rigging, this stage assigns visible mesh vertices to the nearest
complete semantic leaf-bone segment, fits the four mutually exclusive foot
bottoms, and independently repeats the measurement from distal-two-bone skin
weight ownership.  Both authorities must pass and agree before one rigid
rotation and vertical translation are applied.  Mesh topology, materials,
hierarchy, and skin weights are untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


ASSET_TOOLS_ROOT = Path(__file__).resolve().parent
if str(ASSET_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSET_TOOLS_ROOT))

from generated_animal_support_plane import (  # noqa: E402
    evaluate_dual_authority_support_plane,
)
from generated_animal_support_plane_contract import POLICY  # noqa: E402
from generated_quadruped_semantics import infer_quadruped_semantics  # noqa: E402


SCHEMA = "avengine_generated_animal_support_plane_leveling_v2"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--front-axis",
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
        required=True,
    )
    parser.add_argument("--review-evidence", type=Path, required=True)
    parser.add_argument(
        "--plane-source",
        choices=("mesh-foot-bottoms",),
        default="mesh-foot-bottoms",
        help=(
            "Visible mesh-foot authority.  V2 uses mutually exclusive nearest "
            "complete-leaf-segment corridors and requires an independent "
            "distal-two-bone weight-owned crosscheck; no endpoint, expanded "
            "radius, or weight fallback exists."
        ),
    )
    parser.add_argument("--maximum-tilt-deg", type=float, default=30.0)
    parser.add_argument(
        "--maximum-foot-plane-residual-ratio",
        type=float,
        default=0.02,
        help=(
            "Reject when any semantic foot differs from the fitted support "
            "plane by more than this fraction of the target mesh diagonal."
        ),
    )
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


def mesh_bbox(mesh):
    points = [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
    minimum = np.asarray(
        [min(point[axis] for point in points) for axis in range(3)],
        dtype=np.float64,
    )
    maximum = np.asarray(
        [max(point[axis] for point in points) for axis in range(3)],
        dtype=np.float64,
    )
    return minimum, maximum - minimum


def semantic_records(armature):
    records = []
    for bone in armature.data.bones:
        head = armature.matrix_world @ bone.head_local
        tail = armature.matrix_world @ bone.tail_local
        records.append(
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent is not None else None,
                "children": [child.name for child in bone.children],
                "head_world": [float(value) for value in head],
                "tail_world": [float(value) for value in tail],
            }
        )
    return records


def distal_two_weight_scores(mesh, semantics):
    """Return one exclusive-authority score column per semantic limb."""

    chain_by_leaf = {}
    for _label, chain in semantics.chains().items():
        if chain and chain[-1] in semantics.foot_leaves:
            chain_by_leaf[chain[-1]] = list(chain)
    if set(chain_by_leaf) != set(semantics.foot_leaves):
        raise RuntimeError("semantic foot chains are incomplete")

    bone_owner = {}
    for index, leaf in enumerate(semantics.foot_leaves):
        chain = chain_by_leaf[leaf]
        if len(chain) < 2:
            raise RuntimeError(
                f"semantic foot chain has fewer than two bones: {leaf}"
            )
        for bone_name in chain[-2:]:
            previous = bone_owner.setdefault(bone_name, index)
            if previous != index:
                raise RuntimeError(
                    "distal semantic bone belongs to more than one foot"
                )

    group_names = {group.index: group.name for group in mesh.vertex_groups}
    scores = np.zeros((len(mesh.data.vertices), 4), dtype=np.float64)
    for vertex in mesh.data.vertices:
        for membership in vertex.groups:
            owner = bone_owner.get(group_names.get(membership.group))
            if owner is not None:
                scores[vertex.index, owner] += float(membership.weight)
    return scores


def scene_summary():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    skinned = [
        obj
        for obj in meshes
        if any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
    ]
    return {
        "mesh_count": len(meshes),
        "skinned_mesh_count": len(skinned),
        "armature_count": len(armatures),
        "bone_count": sum(len(obj.data.bones) for obj in armatures),
        "material_count": len(bpy.data.materials),
        "image_count": len(bpy.data.images),
        "action_count": len(bpy.data.actions),
    }, skinned, armatures


def main():
    args = parse_argv()
    source = require_input(args.input, "heading-normalized rigged GLB")
    evidence = require_input(args.review_evidence, "heading/rig review evidence")
    output = require_new_output(args.output, "leveled GLB")
    manifest = require_new_output(args.manifest, "leveling manifest")
    if not 0.0 < args.maximum_tilt_deg <= 45.0:
        raise SystemExit("--maximum-tilt-deg must be in (0, 45]")
    if not 0.0 < args.maximum_foot_plane_residual_ratio <= 0.1:
        raise SystemExit("--maximum-foot-plane-residual-ratio must be in (0, 0.1]")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    before, skinned, armatures = scene_summary()
    if len(skinned) != 1 or len(armatures) != 1:
        raise RuntimeError(f"expected one skinned mesh and one armature: {before}")
    if before["action_count"] != 0:
        raise RuntimeError("support-plane leveling must run before animation")
    mesh = skinned[0]
    armature = armatures[0]
    minimum, extent = mesh_bbox(mesh)
    records = semantic_records(armature)
    semantics = infer_quadruped_semantics(
        records,
        bbox_min=minimum,
        bbox_extent=extent,
        front_axis=args.front_axis,
    )
    by_name = {record["name"]: record for record in records}
    world_matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
    local = np.empty((len(mesh.data.vertices), 3), dtype=np.float64)
    mesh.data.vertices.foreach_get("co", local.ravel())
    world_vertices = local @ world_matrix[:3, :3].T + world_matrix[:3, 3]
    segment_heads = np.asarray(
        [by_name[name]["head_world"] for name in semantics.foot_leaves],
        dtype=np.float64,
    )
    segment_tails = np.asarray(
        [by_name[name]["tail_world"] for name in semantics.foot_leaves],
        dtype=np.float64,
    )
    mesh_diagonal = float(np.linalg.norm(extent))
    dual_authority = evaluate_dual_authority_support_plane(
        world_vertices,
        segment_heads,
        segment_tails,
        distal_two_weight_scores(mesh, semantics),
        mesh_diagonal=mesh_diagonal,
        maximum_residual_ratio=args.maximum_foot_plane_residual_ratio,
        maximum_tilt_deg=args.maximum_tilt_deg,
    )
    primary = dual_authority["primary"]
    crosscheck = dual_authority["crosscheck"]
    primary_plane = primary["plane"]
    foot_points = np.asarray(primary["foot_points"], dtype=np.float64)
    crosscheck_foot_points = np.asarray(
        crosscheck["foot_points"], dtype=np.float64
    )
    normal = Vector(primary_plane["normal"])
    up = Vector((0.0, 0.0, 1.0))
    rotation = normal.rotation_difference(up).to_matrix().to_4x4()
    rotated_feet = np.asarray(
        [tuple(rotation @ Vector(point)) for point in foot_points],
        dtype=np.float64,
    )
    vertical_translation = -float(rotated_feet[:, 2].min())
    transform = Matrix.Translation((0.0, 0.0, vertical_translation)) @ rotation
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    if not roots:
        raise RuntimeError("imported scene has no root objects")
    for root in roots:
        root.matrix_world = transform @ root.matrix_world
    post_feet = np.asarray(
        [tuple(transform @ Vector(point)) for point in foot_points],
        dtype=np.float64,
    )
    post_crosscheck_feet = np.asarray(
        [tuple(transform @ Vector(point)) for point in crosscheck_foot_points],
        dtype=np.float64,
    )

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
    after, _skinned_after, _armatures_after = scene_summary()
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
            raise RuntimeError(f"rigid leveling changed {key}: {before} -> {after}")

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "technical_spike_only_pending_retarget_and_visual_qa",
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
        "support_plane": {
            "front_axis": args.front_axis,
            "foot_leaves": list(semantics.foot_leaves),
            "plane_source": args.plane_source,
            "dual_authority": dual_authority,
            "mesh_foot_capture_counts": primary["capture_counts"],
            "mesh_foot_contact_band_sizes": primary["contact_band_sizes"],
            "foot_points_before": foot_points.tolist(),
            "z_equals_ax_plus_by_plus_c": primary_plane[
                "z_equals_ax_plus_by_plus_c"
            ],
            "residual_z": primary_plane["residual_z"],
            "maximum_residual": primary_plane["maximum_residual"],
            "maximum_residual_ratio_of_mesh_diagonal": primary_plane[
                "maximum_residual_ratio_of_mesh_diagonal"
            ],
            "maximum_reviewed_residual_ratio_of_mesh_diagonal": (
                args.maximum_foot_plane_residual_ratio
            ),
            "normal_before": primary_plane["normal"],
            "tilt_deg": primary_plane["tilt_deg"],
            "maximum_tilt_deg": args.maximum_tilt_deg,
            "applied_vertical_translation": vertical_translation,
            "foot_points_after": post_feet.tolist(),
            "crosscheck_foot_points_after": post_crosscheck_feet.tolist(),
            "minimum_foot_z_after": float(post_feet[:, 2].min()),
            "policy": POLICY,
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
        "GENERATED_ANIMAL_SUPPORT_PLANE_LEVELING_OK "
        f"tilt_deg={primary_plane['tilt_deg']:.6f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
