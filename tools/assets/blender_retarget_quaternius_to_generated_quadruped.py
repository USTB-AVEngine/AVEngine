#!/usr/bin/env python3
"""Transfer Quaternius Walk/Idle to a generated, skinned quadruped rig.

The generated mesh, PBR material, skeleton, and skin weights remain the target
authority.  Target bone names are never assumed: a reviewed front axis plus
the rest hierarchy identifies the axial, head, tail, and four limb chains.
For genuinely fitted skeletons, source rotations are sampled in world space
and smoothly resampled by chain arc length.  When a generated asset retained a
uniformly scaled copy of the Quaternius deform skeleton, a stricter proof gate
allows the complete authored local pose (translation, rotation, and scale) to
be copied instead.  That path preserves the translation channels responsible
for stance-foot contact.
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
from mathutils import Matrix, Quaternion, Vector


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generated_quadruped_semantics import (  # noqa: E402
    SemanticRigError,
    infer_quadruped_semantics,
)


SCHEMA = "avengine_generated_quadruped_retarget_v5"
MOTION_BASIS_DECISION_SCHEMA = "generated_animal_motion_basis_manual_decision_v1"
ROTATION_TRANSFER_MODES = (
    "world-left-delta-v2",
    "legacy-rest-local-right-delta-v1",
)
CARDINAL_MOTION_BASIS_YAWS = (-90, 0, 90, 180)
POSE_TRANSFER_MODES = (
    "world-rotation-retarget-v2",
    "world-rotation-foot-ik-v3",
    "template-local-full-pose-v1",
)
SIDE_CHAIN_SWAP = {
    "front_side_negative": "front_side_positive",
    "front_side_positive": "front_side_negative",
    "hind_side_negative": "hind_side_positive",
    "hind_side_positive": "hind_side_negative",
}
SOURCE_CHAINS = {
    "axial_head": ("Bone.001", "Bone.002", "Bone.003"),
    "tail": ("Bone.004", "Bone.005", "Bone.006", "Bone.007"),
    "front_side_negative": ("Bone.014", "Bone.015", "Bone.016"),
    "front_side_positive": ("Bone.017", "Bone.018", "Bone.019"),
    "hind_side_negative": ("Bone.011", "Bone.012", "Bone.013"),
    "hind_side_positive": ("Bone.008", "Bone.009", "Bone.010"),
}
SOURCE_ROOT = "Bone"
ACTION_HINTS = {"Walking": "walk", "Idle": "idle"}


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-glb", type=Path, required=True)
    parser.add_argument("--source-rig-glb", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--motion-basis-decision",
        type=Path,
        help=(
            "Immutable human-approved pre-animation basis decision.  Target "
            "animation generation for registrable outputs is forbidden without "
            "this exact sidecar."
        ),
    )
    parser.add_argument(
        "--technical-spike-only",
        action="store_true",
        help=(
            "Render a non-registrable research candidate without claiming a "
            "human motion-basis approval.  This mode requires explicit cardinal "
            "--motion-basis-yaw-deg and --side-chain-mode values, forbids target "
            "derivations, and records target_animation_generation_authorized=false."
        ),
    )
    parser.add_argument(
        "--target-derivation-manifest",
        type=Path,
        help=(
            "Optional authenticated weight-only derivation from the exact "
            "human-reviewed target.  Geometry, skeleton, material, and front "
            "direction must remain unchanged."
        ),
    )
    parser.add_argument(
        "--target-front-axis",
        required=True,
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
    )
    parser.add_argument(
        "--motion-amplitude",
        type=float,
        default=1.0,
        help=(
            "Scale source rest-to-pose rotation deltas in [0, 1].  Batch "
            "selection can lower this value to keep generated skinning within "
            "deformation and foot-contact limits without changing the target rig."
        ),
    )
    parser.add_argument(
        "--rotation-transfer-mode",
        choices=ROTATION_TRANSFER_MODES,
        default=None,
        help=(
            "How source rest-to-pose rotations are mapped onto target rest axes. "
            "world-left-delta-v2 computes pose*rest^-1 in world coordinates and "
            "does not inherit source local bone roll.  The legacy mode is retained "
            "only to render an immutable A/B diagnostic candidate."
        ),
    )
    parser.add_argument(
        "--pose-transfer-mode",
        choices=POSE_TRANSFER_MODES,
        default="world-rotation-retarget-v2",
        help=(
            "Runtime pose transfer.  template-local-full-pose-v1 is allowed "
            "only when the target rest skeleton is deterministically proven to "
            "be a uniformly scaled copy of the species template; it preserves "
            "the source bone translations that maintain foot contact.  The "
            "world-rotation-foot-ik-v3 mode keeps a fitted target skeleton and "
            "pins its four ankle trajectories to the corresponding source gait."
        ),
    )
    parser.add_argument(
        "--motion-basis-yaw-deg",
        type=int,
        choices=CARDINAL_MOTION_BASIS_YAWS,
        default=None,
        help=(
            "Reviewer-approved cardinal rotation of source motion deltas around "
            "world up before they are applied to the target rest skeleton."
        ),
    )
    parser.add_argument(
        "--side-chain-mode",
        choices=("matched", "swapped"),
        default=None,
        help=(
            "Reviewer-approved mapping of source lateral limb chains onto target "
            "lateral chains.  This must be selected in the pre-animation basis gate."
        ),
    )
    parser.add_argument(
        "--disable-foot-grounding",
        action="store_true",
        help=(
            "Diagnostic only: do not vertically ground the lowest semantic "
            "foot against its own rest height on every sampled frame."
        ),
    )
    return parser.parse_args(argv)


def require_file(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe {label}: {path}")
    return path


def require_output(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def hash_without(value, key: str) -> str:
    return hashlib.sha256(
        canonical_json({name: item for name, item in value.items() if name != key}).encode(
            "utf-8"
        )
    ).hexdigest()


def _authenticated_file_record(path: Path, record, label: str):
    if (
        not isinstance(record, dict)
        or int(record.get("size_bytes", -1)) != path.stat().st_size
        or record.get("sha256") != sha256_file(path)
    ):
        raise SystemExit(f"motion-basis decision {label} identity mismatch")


def load_target_derivation_manifest(path, target, approved_target_record):
    path = require_file(path, "target derivation manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid target derivation manifest: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema")
        not in {
            "avengine_generated_quadruped_joint_weight_smoothing_v1",
            "avengine_generated_quadruped_joint_weight_smoothing_v2",
            "avengine_generated_quadruped_joint_weight_smoothing_v3",
        }
        or payload.get("animation_exported") is not False
        or payload.get("native_mesh_topology_preserved") is not True
        or payload.get("pbr_material_preserved") is not True
        or payload.get("fitted_skeleton_rest_matrices_preserved") is not True
        or payload.get("only_vertex_weights_modified") is not True
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise SystemExit("target derivation is not an approved weight-only transform")
    approved_path = require_file(
        Path(approved_target_record["path"]),
        "human-reviewed target recorded by the decision",
    )
    _authenticated_file_record(
        approved_path,
        approved_target_record,
        "human-reviewed target",
    )
    _authenticated_file_record(
        approved_path,
        payload.get("input"),
        "derivation input",
    )
    if Path(payload["input"].get("path", "")).absolute() != approved_path:
        raise SystemExit("target derivation input path does not match human decision")
    _authenticated_file_record(target, payload.get("output"), "derivation output")
    if Path(payload["output"].get("path", "")).absolute() != target:
        raise SystemExit("target derivation output path does not match CLI target")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": payload["schema"],
        "input_sha256": payload["input"]["sha256"],
        "output_sha256": payload["output"]["sha256"],
        "weight_only": True,
        "native_mesh_topology_preserved": True,
        "pbr_material_preserved": True,
        "fitted_skeleton_rest_matrices_preserved": True,
        "only_vertex_weights_modified": True,
    }


def load_motion_basis_decision(
    path: Path,
    target: Path,
    source: Path,
    front_axis: str,
    target_derivation_manifest=None,
):
    path = require_file(path, "motion-basis decision")
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid motion-basis decision: {error}") from error
    if (
        not isinstance(decision, dict)
        or decision.get("schema") != MOTION_BASIS_DECISION_SCHEMA
        or decision.get("status") != "motion_basis_approved"
        or decision.get("human_approved") is not True
        or decision.get("target_animation_generation_authorized") is not True
        or decision.get("decision_sha256") != hash_without(decision, "decision_sha256")
    ):
        raise SystemExit(
            "motion-basis decision is not an authenticated human approval"
        )
    target_record = decision.get("target")
    target_matches_decision = (
        isinstance(target_record, dict)
        and int(target_record.get("size_bytes", -1)) == target.stat().st_size
        and target_record.get("sha256") == sha256_file(target)
    )
    if target_matches_decision:
        if target_derivation_manifest is not None:
            raise SystemExit(
                "target derivation manifest is forbidden for the exact reviewed target"
            )
        decision["_authenticated_target_derivation"] = None
    else:
        if target_derivation_manifest is None:
            raise SystemExit(
                "CLI target differs from the human decision and has no authenticated derivation"
            )
        decision["_authenticated_target_derivation"] = (
            load_target_derivation_manifest(
                target_derivation_manifest,
                target,
                target_record,
            )
        )
    _authenticated_file_record(source, decision.get("source_motion"), "source motion")
    if decision["target"].get("reviewed_front_axis") != front_axis:
        raise SystemExit("motion-basis decision target front axis mismatch")
    try:
        yaw = int(decision["manual_cardinal_motion_basis_yaw_deg"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("motion-basis decision cardinal yaw is missing") from error
    side_chain_mode = decision.get("side_chain_mode")
    rotation_transfer_mode = decision.get("rotation_transfer_mode")
    if yaw not in CARDINAL_MOTION_BASIS_YAWS:
        raise SystemExit("motion-basis decision contains non-cardinal yaw")
    if side_chain_mode not in {"matched", "swapped"}:
        raise SystemExit("motion-basis decision contains invalid side-chain mapping")
    if rotation_transfer_mode != "world-left-delta-v2":
        raise SystemExit("motion-basis decision selected an unsupported rotation solver")
    return path, decision, yaw, side_chain_mode, rotation_transfer_mode


def hidden_objects():
    collection = bpy.data.collections.get("glTF_not_exported")
    return set(collection.objects) if collection is not None else set()


def imported_real_meshes(imported):
    hidden = hidden_objects()
    return [item for item in imported if item.type == "MESH" and item not in hidden]


def linked_armatures(mesh):
    result = set()
    if mesh.parent is not None and mesh.parent.type == "ARMATURE":
        result.add(mesh.parent)
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            result.add(modifier.object)
    return result


def import_target(path: Path):
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = tuple(item for item in bpy.data.objects if item not in before_objects)
    actions = [item for item in bpy.data.actions if item not in before_actions]
    armatures = [item for item in imported if item.type == "ARMATURE"]
    meshes = imported_real_meshes(imported)
    skinned = [item for item in meshes if linked_armatures(item)]
    if len(armatures) != 1 or len(skinned) != 1:
        raise RuntimeError(
            "target must contain one real skinned mesh and one armature; "
            f"meshes={[item.name for item in meshes]} "
            f"armatures={[item.name for item in armatures]}"
        )
    if actions:
        raise RuntimeError("generated target must not already contain actions")
    armature = armatures[0]
    mesh = skinned[0]
    if armature not in linked_armatures(mesh):
        raise RuntimeError("target mesh is linked to an unexpected armature")
    return armature, mesh, imported


def import_source(path: Path):
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = tuple(item for item in bpy.data.objects if item not in before_objects)
    actions = [item for item in bpy.data.actions if item not in before_actions]
    armatures = [item for item in imported if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError("source rig must contain exactly one armature")
    selected = {}
    for canonical, hint in ACTION_HINTS.items():
        matches = [item for item in actions if hint in item.name.lower()]
        if len(matches) != 1:
            raise RuntimeError(
                f"source action {canonical} is ambiguous: {[item.name for item in matches]}"
            )
        selected[canonical] = matches[0]
    required = {SOURCE_ROOT}
    for chain in SOURCE_CHAINS.values():
        required.update(chain)
    missing = sorted(required - set(armatures[0].data.bones.keys()))
    if missing:
        raise RuntimeError(f"source rig is missing required bones: {missing}")
    return armatures[0], imported, selected


def detach_armature_parent(armature):
    if armature.parent is None:
        return None
    parent_name = armature.parent.name
    world = armature.matrix_world.copy()
    armature.parent = None
    armature.matrix_world = world
    bpy.context.view_layer.update()
    return parent_name


def target_bbox(mesh):
    points = [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    extent = [maximum[index] - minimum[index] for index in range(3)]
    return minimum, maximum, extent


def evaluated_mesh_floor(mesh_object, depsgraph):
    evaluated = mesh_object.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh(
        preserve_all_data_layers=False, depsgraph=depsgraph
    )
    try:
        if not evaluated_mesh.vertices:
            raise RuntimeError("target mesh has no evaluated vertices")
        matrix = evaluated.matrix_world
        return min(float((matrix @ vertex.co).z) for vertex in evaluated_mesh.vertices)
    finally:
        evaluated.to_mesh_clear()


def target_semantic_records(armature):
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


def target_chains(armature, mesh, front_axis):
    minimum, _maximum, extent = target_bbox(mesh)
    semantics = infer_quadruped_semantics(
        target_semantic_records(armature),
        bbox_min=minimum,
        bbox_extent=extent,
        front_axis=front_axis,
    )
    axial_head = tuple(semantics.axial[1:]) + tuple(semantics.head_chain)
    if not axial_head:
        raise SemanticRigError("target axial/head chain is empty after root")
    chains = {
        "axial_head": axial_head,
        "tail": semantics.tail_chain,
        "front_side_negative": semantics.front_side_negative,
        "front_side_positive": semantics.front_side_positive,
        "hind_side_negative": semantics.hind_side_negative,
        "hind_side_positive": semantics.hind_side_positive,
    }
    target_names = set(armature.data.bones.keys())
    covered = set(semantics.all_bones())
    if covered != target_names:
        raise RuntimeError(
            "target semantics do not cover the complete skeleton: "
            f"missing={sorted(target_names - covered)} extra={sorted(covered - target_names)}"
        )
    return semantics, chains


def bone_world_rotation(armature, matrix) -> Quaternion:
    return (armature.matrix_world.to_quaternion() @ matrix.to_quaternion()).normalized()


def normalized_chain_centers(armature, names):
    lengths = [max(float(armature.data.bones[name].length), 1.0e-9) for name in names]
    total = sum(lengths)
    cursor = 0.0
    centers = []
    for length in lengths:
        centers.append((cursor + 0.5 * length) / total)
        cursor += length
    return centers


def source_sampling_plan(target, source, target_names, source_names):
    target_centers = normalized_chain_centers(target, target_names)
    source_centers = normalized_chain_centers(source, source_names)
    plan = []
    for target_name, fraction in zip(target_names, target_centers):
        if fraction <= source_centers[0]:
            first = second = 0
            blend = 0.0
        elif fraction >= source_centers[-1]:
            first = second = len(source_centers) - 1
            blend = 0.0
        else:
            second = next(
                index for index, value in enumerate(source_centers) if value >= fraction
            )
            first = second - 1
            span = source_centers[second] - source_centers[first]
            blend = (fraction - source_centers[first]) / max(span, 1.0e-12)
        source_first = source_names[first]
        source_second = source_names[second]
        source_rest_first = bone_world_rotation(
            source, source.data.bones[source_first].matrix_local
        )
        source_rest_second = bone_world_rotation(
            source, source.data.bones[source_second].matrix_local
        )
        source_rest = source_rest_first.slerp(source_rest_second, blend)
        target_rest = bone_world_rotation(
            target, target.data.bones[target_name].matrix_local
        )
        plan.append(
            {
                "target": target_name,
                "source_first": source_first,
                "source_second": source_second,
                "blend": float(blend),
                "source_rest_world": source_rest,
                "target_rest_world": target_rest,
                "target_chain_fraction": float(fraction),
            }
        )
    return plan


def full_sampling_plan(target, source, semantics, chains, side_chain_mode="matched"):
    if side_chain_mode not in {"matched", "swapped"}:
        raise RuntimeError(f"invalid side-chain mode: {side_chain_mode}")
    source_root_rest = bone_world_rotation(
        source, source.data.bones[SOURCE_ROOT].matrix_local
    )
    target_root_rest = bone_world_rotation(
        target, target.data.bones[semantics.root].matrix_local
    )
    result = [
        {
            "target": semantics.root,
            "source_first": SOURCE_ROOT,
            "source_second": SOURCE_ROOT,
            "blend": 0.0,
            "source_rest_world": source_root_rest,
            "target_rest_world": target_root_rest,
            "target_chain_fraction": 0.0,
            "semantic_chain": "root",
        }
    ]
    for semantic, target_names in chains.items():
        source_semantic = (
            SIDE_CHAIN_SWAP.get(semantic, semantic)
            if side_chain_mode == "swapped"
            else semantic
        )
        entries = source_sampling_plan(
            target, source, target_names, SOURCE_CHAINS[source_semantic]
        )
        for entry in entries:
            entry["semantic_chain"] = semantic
            entry["source_semantic_chain"] = source_semantic
        result.extend(entries)
    by_target = {entry["target"]: entry for entry in result}
    for branch in semantics.auxiliary_branches:
        for target_name in branch:
            bone = target.data.bones[target_name]
            if bone.parent is None or bone.parent.name not in by_target:
                raise RuntimeError(
                    "auxiliary target bone must follow an already mapped parent: "
                    f"{target_name}"
                )
            parent_entry = by_target[bone.parent.name]
            entry = {
                "target": target_name,
                "source_first": parent_entry["source_first"],
                "source_second": parent_entry["source_second"],
                "blend": parent_entry["blend"],
                "source_rest_world": parent_entry["source_rest_world"],
                "target_rest_world": bone_world_rotation(
                    target, bone.matrix_local
                ),
                "target_chain_fraction": parent_entry["target_chain_fraction"],
                "semantic_chain": "auxiliary_rigid_follow",
                "auxiliary_follow_parent": bone.parent.name,
            }
            result.append(entry)
            by_target[target_name] = entry
    if len({entry["target"] for entry in result}) != len(target.data.bones):
        raise RuntimeError("sampling plan must map every target bone exactly once")
    return result


def limb_ik_plan(target, source, chains, side_chain_mode):
    """Map source hip-to-ankle trajectories onto fitted target limb chains."""

    if side_chain_mode not in {"matched", "swapped"}:
        raise RuntimeError(f"invalid side-chain mode: {side_chain_mode}")
    result = []
    for semantic in (
        "front_side_negative",
        "front_side_positive",
        "hind_side_negative",
        "hind_side_positive",
    ):
        target_names = chains[semantic]
        source_semantic = (
            SIDE_CHAIN_SWAP.get(semantic, semantic)
            if side_chain_mode == "swapped"
            else semantic
        )
        source_names = SOURCE_CHAINS[source_semantic]
        if len(target_names) != 3 or len(source_names) != 3:
            raise RuntimeError(
                "foot IK requires three-bone upper/lower/foot chains: "
                f"{semantic} target={len(target_names)} source={len(source_names)}"
            )
        target_hip = target.matrix_world @ target.data.bones[target_names[0]].head_local
        target_ankle = target.matrix_world @ target.data.bones[target_names[2]].head_local
        source_hip = source.matrix_world @ source.data.bones[source_names[0]].head_local
        source_ankle = source.matrix_world @ source.data.bones[source_names[2]].head_local
        target_rest_vector = target_ankle - target_hip
        source_rest_vector = source_ankle - source_hip
        if target_rest_vector.length <= 1.0e-8 or source_rest_vector.length <= 1.0e-8:
            raise RuntimeError(f"foot IK chain has zero hip-to-ankle length: {semantic}")
        result.append(
            {
                "semantic_chain": semantic,
                "source_semantic_chain": source_semantic,
                "target_upper": target_names[0],
                "target_lower": target_names[1],
                "target_foot": target_names[2],
                "source_upper": source_names[0],
                "source_foot": source_names[2],
                "source_rest_foot_world": [
                    float(value) for value in source_ankle
                ],
                "target_rest_foot_world": [
                    float(value) for value in target_ankle
                ],
                "source_rest_hip_world": [
                    float(value) for value in source_hip
                ],
                "target_rest_hip_world": [
                    float(value) for value in target_hip
                ],
                "source_to_target_rest_rotation": source_rest_vector.rotation_difference(
                    target_rest_vector
                ),
                "source_foot_rest_world_rotation": bone_world_rotation(
                    source,
                    source.data.bones[source_names[2]].matrix_local,
                ),
                "target_foot_rest_world_rotation": bone_world_rotation(
                    target,
                    target.data.bones[target_names[2]].matrix_local,
                ),
                "hip_to_ankle_scale": float(
                    target_rest_vector.length / source_rest_vector.length
                ),
            }
        )
    return result


def root_motion_plan(target, source, target_root, limb_specs):
    scales = sorted(spec["hip_to_ankle_scale"] for spec in limb_specs)
    scale = scales[len(scales) // 2] if scales else 1.0
    source_rest = source.matrix_world @ source.data.bones[SOURCE_ROOT].head_local
    target_rest = target.matrix_world @ target.data.bones[target_root].head_local
    return {
        "source_root": SOURCE_ROOT,
        "target_root": target_root,
        "source_rest_root_world": [float(value) for value in source_rest],
        "target_rest_root_world": [float(value) for value in target_rest],
        "translation_scale": float(scale),
    }


def bone_world_endpoints(armature, bone):
    return (
        armature.matrix_world @ bone.head_local,
        armature.matrix_world @ bone.tail_local,
    )


def bone_armature_endpoints(bone):
    return bone.head_local.copy(), bone.tail_local.copy()


def template_local_pose_plan(target, source, semantics, chains, side_chain_mode):
    """Prove and map a target that retains the authored template rest skeleton.

    SkinTokens ``--use_skeleton`` can keep the Quaternius deform skeleton while
    replacing only the generated mesh weights.  In that case a second rotation-
    only retarget drops the authored local translations that keep stance feet on
    the floor.  This gate proves topology, rest rotations, uniform scale, and
    rest joint positions before full local pose matrices may be copied.
    """

    if side_chain_mode != "matched":
        raise RuntimeError(
            "template-local full-pose transfer requires reviewer-approved matched sides"
        )
    mapping = [(semantics.root, SOURCE_ROOT, "root")]
    for semantic, target_names in chains.items():
        source_names = SOURCE_CHAINS[semantic]
        if len(target_names) != len(source_names):
            raise RuntimeError(
                "template-local full-pose transfer requires equal chain lengths: "
                f"{semantic} target={len(target_names)} source={len(source_names)}"
            )
        mapping.extend(
            (target_name, source_name, semantic)
            for target_name, source_name in zip(target_names, source_names)
        )
    if len(mapping) != len(target.data.bones):
        raise RuntimeError(
            "template-local full-pose transfer forbids unmapped auxiliary bones"
        )

    target_by_source = {
        source_name: target_name for target_name, source_name, _ in mapping
    }
    for target_name, source_name, _ in mapping:
        target_parent = target.data.bones[target_name].parent
        source_parent = source.data.bones[source_name].parent
        expected_target_parent = (
            target_by_source.get(source_parent.name) if source_parent is not None else None
        )
        actual_target_parent = target_parent.name if target_parent is not None else None
        if actual_target_parent != expected_target_parent:
            raise RuntimeError(
                "template-local skeleton parent topology differs: "
                f"target={target_name}/{actual_target_parent} "
                f"source={source_name}/{expected_target_parent}"
            )

    # matrix_basis is expressed in armature/bone-local coordinates.  Proving
    # compatibility in world space would incorrectly reject the exact same
    # skeleton when generated-mesh fitting is represented by a non-uniform
    # armature object transform.
    target_root_head, _ = bone_armature_endpoints(
        target.data.bones[semantics.root]
    )
    source_root_head, _ = bone_armature_endpoints(source.data.bones[SOURCE_ROOT])
    target_root_rotation = (
        target.data.bones[semantics.root].matrix_local.to_quaternion().normalized()
    )
    source_root_rotation = (
        source.data.bones[SOURCE_ROOT].matrix_local.to_quaternion().normalized()
    )
    rest_alignment = (
        target_root_rotation @ source_root_rotation.inverted()
    ).normalized()

    length_ratios = []
    rotation_errors = []
    records = []
    for target_name, source_name, semantic in mapping:
        target_bone = target.data.bones[target_name]
        source_bone = source.data.bones[source_name]
        target_head, target_tail = bone_armature_endpoints(target_bone)
        source_head, source_tail = bone_armature_endpoints(source_bone)
        target_length = (target_tail - target_head).length
        source_length = (source_tail - source_head).length
        if target_length <= 1.0e-8 or source_length <= 1.0e-8:
            raise RuntimeError("template-local skeleton contains a zero-length bone")
        length_ratios.append(float(target_length / source_length))
        expected_rotation = (
            rest_alignment
            @ source_bone.matrix_local.to_quaternion().normalized()
        ).normalized()
        actual_rotation = target_bone.matrix_local.to_quaternion().normalized()
        rotation_error = shortest_quaternion_error(actual_rotation, expected_rotation)
        rotation_errors.append(rotation_error)
        records.append(
            {
                "target": target_name,
                "source": source_name,
                "semantic_chain": semantic,
                "armature_local_length_scale": float(target_length / source_length),
                "armature_local_rest_rotation_error_degrees": float(
                    math.degrees(rotation_error)
                ),
            }
        )

    ordered_ratios = sorted(length_ratios)
    uniform_scale = ordered_ratios[len(ordered_ratios) // 2]
    maximum_length_scale_error = max(
        abs(ratio / uniform_scale - 1.0) for ratio in length_ratios
    )
    skeleton_extent = max(
        (
            bone_armature_endpoints(target.data.bones[target_name])[0]
            - target_root_head
        ).length
        for target_name, _, _ in mapping
    )
    maximum_position_error = 0.0
    for target_name, source_name, _ in mapping:
        target_head, _ = bone_armature_endpoints(target.data.bones[target_name])
        source_head, _ = bone_armature_endpoints(source.data.bones[source_name])
        expected_head = target_root_head + rest_alignment @ (
            (source_head - source_root_head) * uniform_scale
        )
        maximum_position_error = max(
            maximum_position_error, (target_head - expected_head).length
        )
    position_error_ratio = maximum_position_error / max(skeleton_extent, 1.0e-8)
    maximum_rotation_error_degrees = math.degrees(max(rotation_errors))
    if maximum_length_scale_error > 0.05:
        raise RuntimeError(
            "template-local skeleton is not uniformly scaled: "
            f"maximum relative length error={maximum_length_scale_error:.6f}"
        )
    if maximum_rotation_error_degrees > 5.0:
        raise RuntimeError(
            "template-local skeleton rest rotations differ: "
            f"maximum error={maximum_rotation_error_degrees:.6f} degrees"
        )
    if position_error_ratio > 0.05:
        raise RuntimeError(
            "template-local skeleton rest joints differ: "
            f"maximum normalized position error={position_error_ratio:.6f}"
        )
    return records, {
        "proven": True,
        "method": "parent_topology_uniform_armature_local_scale_rest_matrix_v2",
        "mapped_bones": len(records),
        "uniform_armature_local_scale": float(uniform_scale),
        "uniform_scale": float(uniform_scale),
        "maximum_relative_armature_local_bone_length_scale_error": float(
            maximum_length_scale_error
        ),
        "maximum_armature_local_rest_rotation_error_degrees": float(
            maximum_rotation_error_degrees
        ),
        "maximum_armature_local_rest_joint_position_error_ratio": float(
            position_error_ratio
        ),
        "target_armature_object_world_scale": [
            float(value) for value in target.matrix_world.to_scale()
        ],
        "source_armature_object_world_scale": [
            float(value) for value in source.matrix_world.to_scale()
        ],
        "thresholds": {
            "relative_armature_local_bone_length_scale_error": 0.05,
            "armature_local_rest_rotation_error_degrees": 5.0,
            "armature_local_rest_joint_position_error_ratio": 0.05,
        },
    }


def source_action_sample_frames(action, source_bones):
    prefix = 'pose.bones["'
    frames = set()
    for curve in action.fcurves:
        if not curve.data_path.startswith(prefix):
            continue
        bone_name = curve.data_path[len(prefix) :].split('"', 1)[0]
        if bone_name in source_bones:
            frames.update(float(point.co.x) for point in curve.keyframe_points)
    ordered = sorted(frames)
    if len(ordered) < 2:
        raise RuntimeError(f"source action has too few samples: {action.name}")
    return ordered


def set_scene_time(value):
    base = math.floor(float(value))
    bpy.context.scene.frame_set(base, subframe=float(value) - base)


def cache_source_action(source, action, source_bones):
    source.animation_data_create()
    source.animation_data.action = action
    result = []
    for output_frame, source_frame in enumerate(
        source_action_sample_frames(action, source_bones)
    ):
        set_scene_time(source_frame)
        bpy.context.view_layer.update()
        result.append(
            {
                "frame": output_frame,
                "source_frame": source_frame,
                "rotations": {
                    name: bone_world_rotation(source, source.pose.bones[name].matrix)
                    for name in source_bones
                },
                "matrix_basis": {
                    name: source.pose.bones[name].matrix_basis.copy()
                    for name in source_bones
                },
                "heads_world": {
                    name: source.matrix_world @ source.pose.bones[name].head
                    for name in source_bones
                },
            }
        )
    return result


def target_parent_first(target):
    def depth(name):
        value = 0
        bone = target.data.bones[name]
        while bone.parent is not None:
            value += 1
            bone = bone.parent
        return value

    return sorted(target.data.bones.keys(), key=lambda name: (depth(name), name))


def parent_local_rest(bone):
    if bone.parent is None:
        return bone.matrix_local.copy()
    return bone.parent.matrix_local.inverted() @ bone.matrix_local


def keyframe_pose_bone(pose_bone, frame):
    pose_bone.keyframe_insert(data_path="location", frame=frame, group=pose_bone.name)
    pose_bone.keyframe_insert(
        data_path="rotation_quaternion", frame=frame, group=pose_bone.name
    )
    pose_bone.keyframe_insert(data_path="scale", frame=frame, group=pose_bone.name)


def shortest_quaternion_error(first, second):
    dot = min(1.0, max(-1.0, abs(float(first.normalized().dot(second.normalized())))))
    return 2.0 * math.acos(dot)


def rotate_pose_joint_toward(target, joint_name, end_name, desired_end):
    joint = target.pose.bones[joint_name]
    current_end = target.pose.bones[end_name].head.copy()
    pivot = joint.head.copy()
    current_vector = current_end - pivot
    desired_vector = desired_end - pivot
    if current_vector.length <= 1.0e-10 or desired_vector.length <= 1.0e-10:
        return
    delta = current_vector.rotation_difference(desired_vector)
    joint.matrix = (
        Matrix.Translation(pivot)
        @ delta.to_matrix().to_4x4()
        @ Matrix.Translation(-pivot)
        @ joint.matrix
    )
    bpy.context.view_layer.update()


def solve_two_bone_ankle_ik(target, spec, desired_world, desired_foot_rotation):
    desired = target.matrix_world.inverted() @ desired_world
    upper = target.pose.bones[spec["target_upper"]]
    lower = target.pose.bones[spec["target_lower"]]
    foot = target.pose.bones[spec["target_foot"]]
    hip = upper.head.copy()
    desired_vector = desired - hip
    # Autoregressive fitted skeletons often retain parent/child offsets instead
    # of connecting each child head to its parent tail.  Reach must therefore
    # use actual adjacent joint-head distances, not EditBone.length.
    upper_segment_length = (lower.head - upper.head).length
    lower_segment_length = (foot.head - lower.head).length
    maximum_reach = max(
        float(upper_segment_length + lower_segment_length) * 0.999,
        1.0e-8,
    )
    minimum_reach = max(
        abs(float(upper_segment_length - lower_segment_length)) * 1.001,
        0.0,
    )
    requested_distance = desired_vector.length
    clamped = False
    if requested_distance > maximum_reach:
        desired = hip + desired_vector.normalized() * maximum_reach
        clamped = True
    elif 0.0 < requested_distance < minimum_reach:
        desired = hip + desired_vector.normalized() * minimum_reach
        clamped = True

    for _iteration in range(12):
        for joint_name in (spec["target_lower"], spec["target_upper"]):
            rotate_pose_joint_toward(
                target,
                joint_name,
                spec["target_foot"],
                desired,
            )
        if (target.pose.bones[spec["target_foot"]].head - desired).length <= 1.0e-6:
            break

    # IK controls the ankle position through upper/lower rotations.  Restore
    # the reviewed world-retargeted foot orientation without moving its head.
    foot = target.pose.bones[spec["target_foot"]]
    foot.matrix = Matrix.LocRotScale(
        foot.matrix.translation.copy(),
        desired_foot_rotation,
        Vector((1.0, 1.0, 1.0)),
    )
    bpy.context.view_layer.update()
    residual_armature = (foot.head - desired).length
    residual_world = (
        (target.matrix_world @ foot.head) - (target.matrix_world @ desired)
    ).length
    return {
        "requested_distance_armature": float(requested_distance),
        "upper_joint_segment_length_armature": float(upper_segment_length),
        "lower_joint_segment_length_armature": float(lower_segment_length),
        "maximum_reach_armature": float(maximum_reach),
        "clamped_to_reachable_range": bool(clamped),
        "residual_armature": float(residual_armature),
        "residual_world": float(residual_world),
    }


def apply_source_root_motion(target, cached_frame, spec, motion_amplitude):
    source_delta = (
        cached_frame["heads_world"][spec["source_root"]]
        - Vector(spec["source_rest_root_world"])
    )
    desired_world = Vector(spec["target_rest_root_world"]) + (
        source_delta * spec["translation_scale"] * motion_amplitude
    )
    root = target.pose.bones[spec["target_root"]]
    current_world = target.matrix_world @ root.head
    correction_world = desired_world - current_world
    correction_armature = target.matrix_world.to_3x3().inverted() @ correction_world
    corrected = root.matrix.copy()
    corrected.translation += correction_armature
    root.matrix = corrected
    bpy.context.view_layer.update()
    residual_world = (target.matrix_world @ root.head - desired_world).length
    return {
        "source_delta_world": [float(value) for value in source_delta],
        "applied_correction_world": [float(value) for value in correction_world],
        "residual_world": float(residual_world),
    }


def bake_action(
    source,
    target,
    target_mesh,
    source_action,
    canonical_name,
    plan,
    motion_amplitude,
    root_name,
    foot_leaves,
    ground_feet,
    rest_mesh_floor,
    rotation_transfer_mode,
    motion_basis_yaw_deg,
    limb_ik_specs=None,
    root_motion_spec=None,
    semantic_foot_grounding=False,
):
    source_bones = sorted(
        {
            entry["source_first"]
            for entry in plan
        }
        | {entry["source_second"] for entry in plan}
    )
    cached = cache_source_action(source, source_action, source_bones)
    action = bpy.data.actions.new(name=canonical_name)
    target.animation_data_create()
    target.animation_data.action = action
    target.data.pose_position = "POSE"
    rest_locals = {
        name: parent_local_rest(target.data.bones[name]) for name in target.data.bones.keys()
    }
    by_target = {entry["target"]: entry for entry in plan}
    order = target_parent_first(target)
    target_object_rotation = target.matrix_world.to_quaternion().normalized()
    max_error = 0.0
    rest_foot_world_z = {
        name: float((target.matrix_world @ target.data.bones[name].head_local).z)
        for name in foot_leaves
    }
    grounding_frames = []
    limb_ik_frames = []
    root_trajectory_alignment_frames = []
    root_motion_frames = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    basis_rotation = Quaternion(
        Vector((0.0, 0.0, 1.0)), math.radians(float(motion_basis_yaw_deg))
    ).normalized()
    for cached_frame in cached:
        frame = cached_frame["frame"]
        bpy.context.scene.frame_set(frame)
        for pose_bone in target.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        requested = {}
        for target_name in order:
            entry = by_target[target_name]
            pose_bone = target.pose.bones[target_name]
            rest_local = rest_locals[target_name]
            if pose_bone.parent is None:
                translation = rest_local.translation.copy()
            else:
                translation = pose_bone.parent.matrix @ rest_local.translation
            source_first = cached_frame["rotations"][entry["source_first"]]
            source_second = cached_frame["rotations"][entry["source_second"]]
            source_pose_world = source_first.slerp(source_second, entry["blend"])
            if rotation_transfer_mode == "world-left-delta-v2":
                # A world-space delta premultiplies the target rest orientation.
                # This is independent of the source bone's authored local roll.
                source_delta_world = (
                    source_pose_world @ entry["source_rest_world"].inverted()
                ).normalized()
                source_delta_world = (
                    basis_rotation
                    @ source_delta_world
                    @ basis_rotation.inverted()
                ).normalized()
            elif rotation_transfer_mode == "legacy-rest-local-right-delta-v1":
                # Preserved only for reviewer-visible A/B evidence.  This is a
                # rest-local delta despite the historical variable name, and can
                # turn fore/aft limb motion sideways when target bone rolls differ.
                source_delta_world = (
                    entry["source_rest_world"].inverted() @ source_pose_world
                ).normalized()
            else:
                raise RuntimeError(
                    f"unsupported rotation transfer mode: {rotation_transfer_mode}"
                )
            scaled_delta_world = Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(
                source_delta_world,
                motion_amplitude,
            )
            if rotation_transfer_mode == "world-left-delta-v2":
                desired_world = (
                    scaled_delta_world @ entry["target_rest_world"]
                ).normalized()
            else:
                desired_world = (
                    entry["target_rest_world"] @ scaled_delta_world
                ).normalized()
            desired_armature = (
                target_object_rotation.inverted() @ desired_world
            ).normalized()
            desired = Matrix.LocRotScale(
                translation, desired_armature, Vector((1.0, 1.0, 1.0))
            )
            pose_bone.matrix = desired
            bpy.context.view_layer.update()
            requested[target_name] = desired_armature
        if root_motion_spec is not None:
            root_result = apply_source_root_motion(
                target,
                cached_frame,
                root_motion_spec,
                motion_amplitude,
            )
            root_result["frame"] = frame
            root_motion_frames.append(root_result)
        if limb_ik_specs:
            hip_alignment_deltas = []
            for spec in limb_ik_specs:
                source_hip_delta = (
                    cached_frame["heads_world"][spec["source_upper"]]
                    - Vector(spec["source_rest_hip_world"])
                )
                desired_hip_world = Vector(spec["target_rest_hip_world"]) + (
                    spec["source_to_target_rest_rotation"] @ source_hip_delta
                ) * spec["hip_to_ankle_scale"]
                current_hip_world = (
                    target.matrix_world
                    @ target.pose.bones[spec["target_upper"]].head
                )
                hip_alignment_deltas.append(desired_hip_world - current_hip_world)
            root_alignment_world = Vector(
                tuple(
                    sum(
                        sorted(delta[axis] for delta in hip_alignment_deltas)[
                            len(hip_alignment_deltas) // 2 - 1 :
                            len(hip_alignment_deltas) // 2 + 1
                        ]
                    )
                    / 2.0
                    for axis in range(3)
                )
            )
            root_alignment_armature = (
                target.matrix_world.to_3x3().inverted() @ root_alignment_world
            )
            root_pose = target.pose.bones[root_name]
            aligned_root = root_pose.matrix.copy()
            aligned_root.translation += root_alignment_armature
            root_pose.matrix = aligned_root
            bpy.context.view_layer.update()
            root_trajectory_alignment_frames.append(
                {
                    "frame": frame,
                    "world_translation": [
                        float(value) for value in root_alignment_world
                    ],
                    "maximum_hip_residual_world": max(
                        (
                            Vector(spec["target_rest_hip_world"])
                            + (
                                spec["source_to_target_rest_rotation"]
                                @ (
                                    cached_frame["heads_world"][spec["source_upper"]]
                                    - Vector(spec["source_rest_hip_world"])
                                )
                            )
                            * spec["hip_to_ankle_scale"]
                            - (
                                target.matrix_world
                                @ target.pose.bones[spec["target_upper"]].head
                            )
                        ).length
                        for spec in limb_ik_specs
                    ),
                }
            )
        frame_ik = []
        for spec in limb_ik_specs or ():
            source_foot_delta = (
                cached_frame["heads_world"][spec["source_foot"]]
                - Vector(spec["source_rest_foot_world"])
            )
            desired_world = Vector(spec["target_rest_foot_world"]) + (
                spec["source_to_target_rest_rotation"] @ source_foot_delta
            ) * spec["hip_to_ankle_scale"]
            # SkinTokens' generated terminal bone roll is not guaranteed to
            # match the authored template foot roll.  Pinning the ankle
            # trajectory while retaining the target's own rest-world foot
            # orientation prevents paws from folding beneath the ground.
            desired_foot_world_rotation = spec[
                "target_foot_rest_world_rotation"
            ].normalized()
            desired_foot_armature_rotation = (
                target.matrix_world.to_quaternion().normalized().inverted()
                @ desired_foot_world_rotation
            ).normalized()
            ik_result = solve_two_bone_ankle_ik(
                target,
                spec,
                desired_world,
                desired_foot_armature_rotation,
            )
            ik_result["semantic_chain"] = spec["semantic_chain"]
            frame_ik.append(ik_result)
        if frame_ik:
            limb_ik_frames.append({"frame": frame, "limbs": frame_ik})
        before_grounding = {
            name: float(
                (target.matrix_world @ target.pose.bones[name].head).z
                - rest_foot_world_z[name]
            )
            for name in foot_leaves
        }
        mesh_floor_before = evaluated_mesh_floor(target_mesh, depsgraph)
        mesh_floor_delta_before = mesh_floor_before - rest_mesh_floor
        correction_world_z = 0.0
        if ground_feet:
            if semantic_foot_grounding:
                correction_world_z = -min(before_grounding.values())
            else:
                correction_world_z = max(
                    -min(before_grounding.values()),
                    -mesh_floor_delta_before,
                )
            correction_armature = (
                target.matrix_world.to_3x3().inverted()
                @ Vector((0.0, 0.0, correction_world_z))
            )
            root_pose = target.pose.bones[root_name]
            corrected = root_pose.matrix.copy()
            corrected.translation += correction_armature
            root_pose.matrix = corrected
            bpy.context.view_layer.update()
        after_grounding = {
            name: float(
                (target.matrix_world @ target.pose.bones[name].head).z
                - rest_foot_world_z[name]
            )
            for name in foot_leaves
        }
        grounding_frames.append(
            {
                "frame": frame,
                "correction_world_z": float(correction_world_z),
                "minimum_before": min(before_grounding.values()),
                "minimum_after": min(after_grounding.values()),
                "mesh_floor_delta_before": float(mesh_floor_delta_before),
                "mesh_floor_delta_after": float(
                    mesh_floor_delta_before + correction_world_z
                ),
            }
        )
        for pose_bone in target.pose.bones:
            keyframe_pose_bone(pose_bone, frame)
        for name, desired in requested.items():
            actual = target.pose.bones[name].matrix.to_quaternion()
            max_error = max(max_error, shortest_quaternion_error(actual, desired))
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    action.use_fake_user = True
    return action, {
        "source_action": source_action.name,
        "output_action": canonical_name,
        "source_frame_range": [
            cached[0]["source_frame"],
            cached[-1]["source_frame"],
        ],
        "frame_range": [0, len(cached) - 1],
        "sampled_frames": len(cached),
        "motion_amplitude": float(motion_amplitude),
        "rotation_transfer_mode": rotation_transfer_mode,
        "motion_basis_yaw_deg": int(motion_basis_yaw_deg),
        "root_motion": {
            "enabled": root_motion_spec is not None,
            "method": (
                "source_root_head_delta_scaled_to_target_v1"
                if root_motion_spec is not None
                else None
            ),
            "translation_scale": (
                float(root_motion_spec["translation_scale"])
                if root_motion_spec is not None
                else None
            ),
            "maximum_source_delta_world": max(
                (
                    Vector(item["source_delta_world"]).length
                    for item in root_motion_frames
                ),
                default=0.0,
            ),
            "maximum_applied_correction_world": max(
                (
                    Vector(item["applied_correction_world"]).length
                    for item in root_motion_frames
                ),
                default=0.0,
            ),
            "maximum_residual_world": max(
                (item["residual_world"] for item in root_motion_frames),
                default=0.0,
            ),
        },
        "limb_ik": {
            "enabled": bool(limb_ik_specs),
            "method": (
                "source_rest_relative_ankle_trajectory_ccd_v2"
                if limb_ik_specs
                else None
            ),
            "foot_orientation_policy": (
                "lock_target_rest_world_v1" if limb_ik_specs else None
            ),
            "maximum_residual_world": max(
                (
                    limb["residual_world"]
                    for frame in limb_ik_frames
                    for limb in frame["limbs"]
                ),
                default=0.0,
            ),
            "clamped_limb_frames": sum(
                limb["clamped_to_reachable_range"]
                for frame in limb_ik_frames
                for limb in frame["limbs"]
            ),
            "sampled_limb_frames": sum(
                len(frame["limbs"]) for frame in limb_ik_frames
            ),
            "root_trajectory_alignment": {
                "enabled": bool(limb_ik_specs),
                "method": (
                    "componentwise_median_source_hip_trajectory_v1"
                    if limb_ik_specs
                    else None
                ),
                "maximum_translation_world": max(
                    (
                        Vector(frame["world_translation"]).length
                        for frame in root_trajectory_alignment_frames
                    ),
                    default=0.0,
                ),
                "translation_world_z_range": [
                    min(
                        (
                            frame["world_translation"][2]
                            for frame in root_trajectory_alignment_frames
                        ),
                        default=0.0,
                    ),
                    max(
                        (
                            frame["world_translation"][2]
                            for frame in root_trajectory_alignment_frames
                        ),
                        default=0.0,
                    ),
                ],
                "maximum_hip_residual_world": max(
                    (
                        frame["maximum_hip_residual_world"]
                        for frame in root_trajectory_alignment_frames
                    ),
                    default=0.0,
                ),
            },
        },
        "foot_grounding": {
            "enabled": bool(ground_feet),
            "reference": (
                "minimum_semantic_foot_delta_from_rest_v3"
                if semantic_foot_grounding
                else "minimum_semantic_foot_or_evaluated_mesh_floor_delta_from_rest_v2"
            ),
            "evaluated_mesh_floor_is_diagnostic_only": bool(
                semantic_foot_grounding
            ),
            "maximum_absolute_correction_world": max(
                abs(item["correction_world_z"]) for item in grounding_frames
            ),
            "correction_world_range": [
                min(item["correction_world_z"] for item in grounding_frames),
                max(item["correction_world_z"] for item in grounding_frames),
            ],
            "maximum_absolute_residual_minimum_world": max(
                abs(item["minimum_after"]) for item in grounding_frames
            ),
            "maximum_absolute_residual_mesh_floor_world": max(
                abs(item["mesh_floor_delta_after"]) for item in grounding_frames
            ),
        },
        "maximum_requested_rotation_error_degrees": math.degrees(max_error),
    }


def bake_template_local_action(
    source,
    target,
    target_mesh,
    source_action,
    canonical_name,
    plan,
    compatibility,
    motion_amplitude,
    root_name,
    foot_leaves,
    ground_feet,
    rest_mesh_floor,
):
    """Copy a proven template skeleton's complete authored local pose.

    Unlike ``bake_action``, this intentionally preserves per-bone translation
    and scale channels as well as rotations.  It is only callable after
    ``template_local_pose_plan`` has proven one-to-one topology and compatible
    rest matrices, so a generated fitted skeleton cannot silently enter this
    path.
    """

    source_bones = sorted({entry["source"] for entry in plan})
    cached = cache_source_action(source, source_action, source_bones)
    action = bpy.data.actions.new(name=canonical_name)
    target.animation_data_create()
    target.animation_data.action = action
    target.data.pose_position = "POSE"
    uniform_scale = float(compatibility["uniform_scale"])
    identity_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    identity_scale = Vector((1.0, 1.0, 1.0))
    rest_foot_world_z = {
        name: float((target.matrix_world @ target.data.bones[name].head_local).z)
        for name in foot_leaves
    }
    grounding_frames = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    by_target = {entry["target"]: entry for entry in plan}
    order = target_parent_first(target)

    for cached_frame in cached:
        frame = cached_frame["frame"]
        bpy.context.scene.frame_set(frame)
        for pose_bone in target.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.matrix_basis = Matrix.Identity(4)
        for target_name in order:
            entry = by_target[target_name]
            source_basis = cached_frame["matrix_basis"][entry["source"]]
            location, rotation, scale = source_basis.decompose()
            scaled_location = location * (uniform_scale * motion_amplitude)
            scaled_rotation = identity_rotation.slerp(
                rotation.normalized(), motion_amplitude
            )
            scaled_scale = identity_scale.lerp(scale, motion_amplitude)
            target.pose.bones[target_name].matrix_basis = Matrix.LocRotScale(
                scaled_location,
                scaled_rotation,
                scaled_scale,
            )
        bpy.context.view_layer.update()

        before_grounding = {
            name: float(
                (target.matrix_world @ target.pose.bones[name].head).z
                - rest_foot_world_z[name]
            )
            for name in foot_leaves
        }
        mesh_floor_before = evaluated_mesh_floor(target_mesh, depsgraph)
        mesh_floor_delta_before = mesh_floor_before - rest_mesh_floor
        correction_world_z = 0.0
        if ground_feet:
            # Semantic feet are the grounding authority.  Generated belly,
            # tail, or stretched-skin vertices may sit below the rest floor and
            # must not override the actual stance-foot contact signal.
            correction_world_z = -min(before_grounding.values())
            correction_armature = (
                target.matrix_world.to_3x3().inverted()
                @ Vector((0.0, 0.0, correction_world_z))
            )
            root_pose = target.pose.bones[root_name]
            corrected = root_pose.matrix.copy()
            corrected.translation += correction_armature
            root_pose.matrix = corrected
            bpy.context.view_layer.update()
        after_grounding = {
            name: float(
                (target.matrix_world @ target.pose.bones[name].head).z
                - rest_foot_world_z[name]
            )
            for name in foot_leaves
        }
        mesh_floor_after = evaluated_mesh_floor(target_mesh, depsgraph)
        grounding_frames.append(
            {
                "frame": frame,
                "correction_world_z": float(correction_world_z),
                "minimum_before": min(before_grounding.values()),
                "minimum_after": min(after_grounding.values()),
                "mesh_floor_delta_before": float(mesh_floor_delta_before),
                "mesh_floor_delta_after": float(mesh_floor_after - rest_mesh_floor),
            }
        )
        for pose_bone in target.pose.bones:
            keyframe_pose_bone(pose_bone, frame)

    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    action.use_fake_user = True
    return action, {
        "source_action": source_action.name,
        "output_action": canonical_name,
        "source_frame_range": [
            cached[0]["source_frame"],
            cached[-1]["source_frame"],
        ],
        "frame_range": [0, len(cached) - 1],
        "sampled_frames": len(cached),
        "motion_amplitude": float(motion_amplitude),
        "pose_transfer_mode": "template-local-full-pose-v1",
        "channels_copied": ["translation", "rotation", "scale"],
        "translation_scale": uniform_scale,
        "foot_grounding": {
            "enabled": bool(ground_feet),
            "reference": "minimum_semantic_foot_delta_from_rest_v3",
            "evaluated_mesh_floor_is_diagnostic_only": True,
            "maximum_absolute_correction_world": max(
                abs(item["correction_world_z"]) for item in grounding_frames
            ),
            "correction_world_range": [
                min(item["correction_world_z"] for item in grounding_frames),
                max(item["correction_world_z"] for item in grounding_frames),
            ],
            "maximum_absolute_residual_minimum_world": max(
                abs(item["minimum_after"]) for item in grounding_frames
            ),
            "maximum_absolute_residual_mesh_floor_world": max(
                abs(item["mesh_floor_delta_after"]) for item in grounding_frames
            ),
        },
    }


def remove_source(imported, actions):
    for item in imported:
        if item.name in bpy.data.objects:
            bpy.data.objects.remove(item, do_unlink=True)
    for action in actions:
        if action.name in bpy.data.actions:
            bpy.data.actions.remove(action)


def remove_export_extras(target, mesh):
    removed = []
    for item in list(bpy.data.objects):
        if item not in {target, mesh}:
            removed.append(item.name)
            bpy.data.objects.remove(item, do_unlink=True)
    return sorted(removed)


def add_nla_tracks(target, actions):
    target.animation_data_create()
    target.animation_data.action = None
    while target.animation_data.nla_tracks:
        target.animation_data.nla_tracks.remove(target.animation_data.nla_tracks[0])
    for action in actions:
        start, end = [int(round(value)) for value in action.frame_range]
        track = target.animation_data.nla_tracks.new()
        track.name = action.name
        strip = track.strips.new(action.name, start, action)
        strip.name = action.name
        strip.action_frame_start = start
        strip.action_frame_end = end


def export_target(target, mesh, actions, output, target_front_axis):
    add_nla_tracks(target, actions)
    canonical_yaw = {
        "positive-x": 0.0,
        "negative-x": 180.0,
        "positive-y": -90.0,
        "negative-y": 90.0,
    }[target_front_axis]
    if canonical_yaw:
        target.matrix_world = (
            Matrix.Rotation(math.radians(canonical_yaw), 4, "Z")
            @ target.matrix_world
        )
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_nla_strips=True,
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    return canonical_yaw


def serializable_plan(plan):
    return [
        {
            key: value
            for key, value in entry.items()
            if key not in {"source_rest_world", "target_rest_world"}
        }
        for entry in plan
    ]


def serializable_limb_ik_plan(plan):
    return [
        {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "source_to_target_rest_rotation",
                "source_foot_rest_world_rotation",
                "target_foot_rest_world_rotation",
            }
        }
        for entry in plan
    ]


def main():
    args = parse_argv()
    if not 0.0 < args.motion_amplitude <= 1.0:
        raise SystemExit("--motion-amplitude must be in (0, 1]")
    target_path = require_file(args.target_glb, "generated target GLB")
    source_path = require_file(args.source_rig_glb, "source rig GLB")
    if args.technical_spike_only:
        if args.motion_basis_decision is not None:
            raise SystemExit(
                "--motion-basis-decision is incompatible with --technical-spike-only"
            )
        if args.target_derivation_manifest is not None:
            raise SystemExit(
                "--target-derivation-manifest is forbidden for a technical spike"
            )
        if args.motion_basis_yaw_deg is None or args.side_chain_mode is None:
            raise SystemExit(
                "--technical-spike-only requires --motion-basis-yaw-deg and "
                "--side-chain-mode"
            )
        if args.rotation_transfer_mode not in {None, "world-left-delta-v2"}:
            raise SystemExit(
                "technical spikes support only world-left-delta-v2 rotation transfer"
            )
        motion_basis_decision_path = None
        motion_basis_decision = None
        approved_motion_basis_yaw = args.motion_basis_yaw_deg
        approved_side_chain_mode = args.side_chain_mode
        approved_rotation_transfer_mode = "world-left-delta-v2"
    else:
        if args.motion_basis_decision is None:
            raise SystemExit(
                "--motion-basis-decision is required unless --technical-spike-only "
                "is selected"
            )
        (
            motion_basis_decision_path,
            motion_basis_decision,
            approved_motion_basis_yaw,
            approved_side_chain_mode,
            approved_rotation_transfer_mode,
        ) = load_motion_basis_decision(
            args.motion_basis_decision,
            target_path,
            source_path,
            args.target_front_axis,
            target_derivation_manifest=args.target_derivation_manifest,
        )
        if (
            args.motion_basis_yaw_deg is not None
            and args.motion_basis_yaw_deg != approved_motion_basis_yaw
        ):
            raise SystemExit("CLI motion-basis yaw differs from the human decision")
        if (
            args.side_chain_mode is not None
            and args.side_chain_mode != approved_side_chain_mode
        ):
            raise SystemExit("CLI side-chain mode differs from the human decision")
        if (
            args.rotation_transfer_mode is not None
            and args.rotation_transfer_mode != approved_rotation_transfer_mode
        ):
            raise SystemExit("CLI rotation solver differs from the human decision")
    output = require_output(args.output_glb, "animated output GLB")
    manifest_path = require_output(args.manifest, "retarget manifest")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    target, mesh, target_imported = import_target(target_path)
    detached_parent = detach_armature_parent(target)
    target.data.pose_position = "REST"
    bpy.context.view_layer.update()
    rest_mesh_floor = evaluated_mesh_floor(
        mesh, bpy.context.evaluated_depsgraph_get()
    )
    semantics, chains = target_chains(target, mesh, args.target_front_axis)
    source, source_imported, source_actions = import_source(source_path)
    template_compatibility = None
    limb_ik_specs = []
    root_motion_spec = None
    if args.pose_transfer_mode == "template-local-full-pose-v1":
        if approved_motion_basis_yaw != 0:
            raise RuntimeError(
                "template-local full-pose transfer currently requires an "
                "approved 0-degree motion basis"
            )
        plan, template_compatibility = template_local_pose_plan(
            target,
            source,
            semantics,
            chains,
            side_chain_mode=approved_side_chain_mode,
        )
    else:
        if (
            args.pose_transfer_mode == "world-rotation-foot-ik-v3"
            and approved_motion_basis_yaw != 0
        ):
            raise RuntimeError(
                "world-rotation foot IK currently requires an approved "
                "0-degree motion basis"
            )
        plan = full_sampling_plan(
            target,
            source,
            semantics,
            chains,
            side_chain_mode=approved_side_chain_mode,
        )
        if args.pose_transfer_mode == "world-rotation-foot-ik-v3":
            limb_ik_specs = limb_ik_plan(
                target,
                source,
                chains,
                approved_side_chain_mode,
            )
            root_motion_spec = root_motion_plan(
                target,
                source,
                semantics.root,
                limb_ik_specs,
            )
    action_results = []
    output_actions = []
    for canonical in ("Walking", "Idle"):
        if args.pose_transfer_mode == "template-local-full-pose-v1":
            action, result = bake_template_local_action(
                source,
                target,
                mesh,
                source_actions[canonical],
                canonical,
                plan,
                template_compatibility,
                args.motion_amplitude,
                semantics.root,
                semantics.foot_leaves,
                not args.disable_foot_grounding,
                rest_mesh_floor,
            )
        else:
            action, result = bake_action(
                source,
                target,
                mesh,
                source_actions[canonical],
                canonical,
                plan,
                args.motion_amplitude,
                semantics.root,
                semantics.foot_leaves,
                not args.disable_foot_grounding,
                rest_mesh_floor,
                approved_rotation_transfer_mode,
                approved_motion_basis_yaw,
                limb_ik_specs=limb_ik_specs,
                root_motion_spec=root_motion_spec,
                semantic_foot_grounding=(
                    args.pose_transfer_mode == "world-rotation-foot-ik-v3"
                ),
            )
        output_actions.append(action)
        action_results.append(result)
    remove_source(source_imported, source_actions.values())
    removed = remove_export_extras(target, mesh)
    canonical_yaw = export_target(
        target, mesh, output_actions, output, args.target_front_axis
    )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "path": str(target_path),
            "sha256": sha256_file(target_path),
            "mesh_pbr_skeleton_and_weights_authority": True,
            "reviewed_front_axis": args.target_front_axis,
        },
        "source_motion": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "geometry_used": False,
            "weights_used": False,
            "animation_channels_used": ["translation", "rotation", "scale"],
        },
        "motion_basis_gate": (
            {
                "mode": "technical_spike_only_unreviewed",
                "decision_path": None,
                "human_approved": False,
                "target_animation_generation_authorized": False,
                "candidate_id": (
                    f"yaw_{approved_motion_basis_yaw:+d}_side_"
                    f"{approved_side_chain_mode}"
                ),
                "selected_motion_basis_yaw_deg": int(approved_motion_basis_yaw),
                "selected_side_chain_mode": approved_side_chain_mode,
                "selected_rotation_solver": approved_rotation_transfer_mode,
                "formal_dataset_registration_authorized": False,
            }
            if args.technical_spike_only
            else {
                "mode": "authenticated_human_approval",
                "decision_path": str(motion_basis_decision_path),
                "decision_sha256": motion_basis_decision["decision_sha256"],
                "decision_file_sha256": sha256_file(motion_basis_decision_path),
                "preview_sha256": motion_basis_decision["preview_sha256"],
                "human_approved": True,
                "human_approved_by": motion_basis_decision["human_approved_by"],
                "target_animation_generation_authorized": True,
                "candidate_id": motion_basis_decision["candidate_id"],
                "approved_motion_basis_yaw_deg": int(approved_motion_basis_yaw),
                "approved_side_chain_mode": approved_side_chain_mode,
                "approved_preview_rotation_solver": approved_rotation_transfer_mode,
                "authenticated_target_derivation": motion_basis_decision.get(
                    "_authenticated_target_derivation"
                ),
            }
        ),
        "semantic_inference": {
            "method": "one_root_four_low_leaf_segment_geometry_hierarchy_v3",
            "bone_name_independent_target": True,
            "root": semantics.root,
            "chains": {name: list(value) for name, value in chains.items()},
            "auxiliary_branches": [
                list(branch) for branch in semantics.auxiliary_branches
            ],
            "auxiliary_motion_policy": "rigid_follow_nearest_semantic_parent",
            "foot_leaves": list(semantics.foot_leaves),
            "complete_target_bone_coverage": True,
        },
        "runtime_pose_transfer": {
            "mode": args.pose_transfer_mode,
            "template_compatibility_proof": template_compatibility,
            "full_local_translation_rotation_scale_copied": (
                args.pose_transfer_mode == "template-local-full-pose-v1"
            ),
            "world_rotation_resampling_used": (
                args.pose_transfer_mode
                in {"world-rotation-retarget-v2", "world-rotation-foot-ik-v3"}
            ),
            "sampling_plan": serializable_plan(plan),
            "limb_ik_plan": serializable_limb_ik_plan(limb_ik_specs),
            "root_motion_plan": root_motion_spec,
        },
        "rotation_transfer": {
            "method": (
                approved_rotation_transfer_mode
                if args.pose_transfer_mode
                in {"world-rotation-retarget-v2", "world-rotation-foot-ik-v3"}
                else None
            ),
            "approved_preview_method": approved_rotation_transfer_mode,
            "used_for_runtime": (
                args.pose_transfer_mode
                in {"world-rotation-retarget-v2", "world-rotation-foot-ik-v3"}
            ),
            "local_bone_roll_copied": (
                args.pose_transfer_mode
                in {"world-rotation-retarget-v2", "world-rotation-foot-ik-v3"}
                and approved_rotation_transfer_mode
                == "legacy-rest-local-right-delta-v1"
            ),
            "motion_amplitude": float(args.motion_amplitude),
            "motion_basis_yaw_deg": int(approved_motion_basis_yaw),
            "side_chain_mode": approved_side_chain_mode,
            "foot_grounding": {
                "enabled": not args.disable_foot_grounding,
                "method": (
                    "minimum_semantic_foot_delta_from_rest_v3"
                    if args.pose_transfer_mode
                    in {"template-local-full-pose-v1", "world-rotation-foot-ik-v3"}
                    else "minimum_semantic_foot_or_evaluated_mesh_floor_delta_from_rest_v2"
                ),
            },
        },
        "actions": action_results,
        "export": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            "canonical_front_axis": "positive-x",
            "canonical_yaw_degrees": canonical_yaw,
            "detached_target_parent": detached_parent,
            "removed_export_extras": removed,
            "action_names": ["Walking", "Idle"],
        },
        "status": (
            "technical_spike_only_pending_deformation_and_visual_qa"
            if args.technical_spike_only
            else "research_candidate_pending_deformation_and_visual_qa"
        ),
        "formal_dataset_registration_authorized": False,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_QUADRUPED_RETARGET_OK "
        f"bones={len(target.data.bones)} actions=2 output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
