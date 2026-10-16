#!/usr/bin/env python3
"""Retarget audited GLB motion onto an audited target GLB in Blender.

Run with Blender, for example::

    blender --background --python tools/motion/retarget_blender.py -- \
      --source-glb source.glb --target-glb target.glb --profile profile.json \
      --output-glb output.glb --report retarget.json

The target mesh, UVs, skeleton and weights remain authoritative.  Only
profile-mapped joint rotations are compiled.  Unmapped target joints retain
their authored rest-local pose, and target root translation remains static.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from avengine.motion.math import (  # noqa: E402
    retarget_world_rotation_xyzw,
)
from avengine.motion.profiles import (  # noqa: E402
    MotionRetargetProfile,
    load_motion_retarget_profile,
)


REPORT_SCHEMA = "avengine_motion_retarget_evidence_v1"
QUALIFICATION_STATE = "research_candidate"


def parse_argv() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--target-glb", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def require_input(path: Path, label: str, suffixes: set[str]) -> Path:
    source = path.resolve()
    if (
        source.is_symlink()
        or not source.is_file()
        or source.stat().st_size <= 0
        or source.suffix.lower() not in suffixes
    ):
        raise SystemExit(f"missing or unsafe {label}: {source}")
    return source


def require_output(path: Path, label: str) -> Path:
    output = path.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def quaternion_xyzw(value: Quaternion) -> tuple[float, float, float, float]:
    item = value.normalized()
    return (float(item.x), float(item.y), float(item.z), float(item.w))


def blender_quaternion(value) -> Quaternion:
    x, y, z, w = map(float, value)
    return Quaternion((w, x, y, z)).normalized()


def shortest_quaternion_error_degrees(first: Quaternion, second: Quaternion) -> float:
    dot = min(1.0, max(-1.0, abs(float(first.normalized().dot(second.normalized())))))
    return math.degrees(2.0 * math.acos(dot))


def bone_world_rotation(armature, matrix: Matrix) -> Quaternion:
    return (armature.matrix_world.to_quaternion() @ matrix.to_quaternion()).normalized()


def mesh_skin_sha256(meshes) -> str:
    digest = hashlib.sha256()
    for mesh in sorted(meshes, key=lambda item: item.name):
        digest.update(mesh.name.encode("utf-8"))
        for vertex in mesh.data.vertices:
            digest.update(struct.pack("<3d", *map(float, vertex.co)))
            for group in sorted(vertex.groups, key=lambda value: value.group):
                digest.update(struct.pack("<Id", int(group.group), float(group.weight)))
        for polygon in mesh.data.polygons:
            digest.update(struct.pack("<I", len(polygon.vertices)))
            digest.update(struct.pack(f"<{len(polygon.vertices)}I", *polygon.vertices))
        for layer in mesh.data.uv_layers:
            digest.update(layer.name.encode("utf-8"))
            for item in layer.data:
                digest.update(struct.pack("<2d", *map(float, item.uv)))
        for group in mesh.vertex_groups:
            digest.update(group.name.encode("utf-8"))
    return digest.hexdigest()


def skeleton_contract(armature) -> dict:
    bones = []
    for bone in armature.data.bones:
        bones.append(
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent is not None else None,
                "matrix_local": [
                    [float(bone.matrix_local[row][column]) for column in range(4)]
                    for row in range(4)
                ],
            }
        )
    return {
        "armature_name": armature.name,
        "bones": bones,
    }


def linked_to_armature(mesh, armature) -> bool:
    if mesh.parent == armature:
        return True
    return any(
        modifier.type == "ARMATURE" and modifier.object == armature
        for modifier in mesh.modifiers
    )


def import_target(path: Path):
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = tuple(item for item in bpy.data.objects if item not in before_objects)
    actions = tuple(item for item in bpy.data.actions if item not in before_actions)
    armatures = [item for item in objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError("target GLB must contain exactly one armature")
    armature = armatures[0]
    meshes = tuple(
        item
        for item in objects
        if item.type == "MESH" and linked_to_armature(item, armature)
    )
    if not meshes:
        raise RuntimeError("target GLB must contain at least one skinned mesh")
    return armature, meshes, objects, actions


def import_source(path: Path, profile: MotionRetargetProfile):
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = tuple(item for item in bpy.data.objects if item not in before_objects)
    actions = tuple(item for item in bpy.data.actions if item not in before_actions)
    armatures = [item for item in objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError("source GLB must contain exactly one armature")
    selected = {}
    for action_mapping in profile.actions:
        hint = action_mapping.source_action_hint.lower()
        matches = [item for item in actions if hint in item.name.lower()]
        if len(matches) != 1:
            raise RuntimeError(
                f"source action hint {hint!r} is ambiguous: "
                f"{[item.name for item in matches]}"
            )
        selected[action_mapping.semantic_action_id] = matches[0]
    return armatures[0], objects, actions, selected


def validate_mapping(source, target, profile: MotionRetargetProfile) -> None:
    source_names = set(source.data.bones.keys())
    target_names = set(target.data.bones.keys())
    missing_source = sorted(
        {mapping.source_joint_id for mapping in profile.joint_mappings} - source_names
    )
    missing_target = sorted(
        {mapping.target_joint_id for mapping in profile.joint_mappings} - target_names
    )
    if missing_source or missing_target:
        raise RuntimeError(
            f"motion profile joint mapping is incomplete: "
            f"source={missing_source} target={missing_target}"
        )


def parent_local_rest(bone) -> Matrix:
    if bone.parent is None:
        return bone.matrix_local.copy()
    return bone.parent.matrix_local.inverted() @ bone.matrix_local


def target_parent_first(target) -> list[str]:
    def depth(name: str) -> int:
        result = 0
        bone = target.data.bones[name]
        while bone.parent is not None:
            result += 1
            bone = bone.parent
        return result

    return sorted(target.data.bones.keys(), key=lambda name: (depth(name), name))


def set_scene_time(value: float) -> None:
    base = math.floor(float(value))
    bpy.context.scene.frame_set(base, subframe=float(value) - base)


def cache_source_action(
    source,
    action,
    source_bones: tuple[str, ...],
    *,
    output_sample_rate_hz: int,
) -> tuple[list[dict], dict]:
    source.animation_data_create()
    source.animation_data.action = action
    start, end = map(float, action.frame_range)
    source_rate = float(bpy.context.scene.render.fps) / float(
        bpy.context.scene.render.fps_base
    )
    duration_seconds = (end - start) / source_rate
    output_intervals = int(round(duration_seconds * output_sample_rate_hz))
    if output_intervals < 1:
        raise RuntimeError(f"source action {action.name!r} has zero duration")
    reconstructed_duration = output_intervals / float(output_sample_rate_hz)
    if abs(reconstructed_duration - duration_seconds) > 1.0e-6:
        raise RuntimeError(
            f"source action {action.name!r} duration cannot be represented exactly "
            f"at {output_sample_rate_hz} Hz"
        )
    result = []
    for output_frame in range(output_intervals + 1):
        source_seconds = output_frame / float(output_sample_rate_hz)
        source_frame = start + source_seconds * source_rate
        set_scene_time(source_frame)
        bpy.context.view_layer.update()
        result.append(
            {
                "frame": output_frame,
                "source_frame": float(source_frame),
                "source_seconds": float(source_seconds),
                "rotations": {
                    name: bone_world_rotation(source, source.pose.bones[name].matrix)
                    for name in source_bones
                },
            }
        )
    return result, {
        "source_frame_range": [start, end],
        "source_scene_rate_hz": source_rate,
        "source_duration_seconds": duration_seconds,
        "output_frame_range": [0, output_intervals],
        "output_sample_rate_hz": output_sample_rate_hz,
        "sampled_frames_including_loop_endpoint": output_intervals + 1,
    }


def keyframe_pose_bone(pose_bone, frame: int) -> None:
    pose_bone.keyframe_insert(data_path="location", frame=frame, group=pose_bone.name)
    pose_bone.keyframe_insert(
        data_path="rotation_quaternion", frame=frame, group=pose_bone.name
    )
    pose_bone.keyframe_insert(data_path="scale", frame=frame, group=pose_bone.name)


def bake_action(
    source,
    target,
    source_action,
    action_mapping,
    profile: MotionRetargetProfile,
) -> tuple[object, dict]:
    mappings_by_target = {
        mapping.target_joint_id: mapping for mapping in profile.joint_mappings
    }
    source_bones = tuple(
        sorted({mapping.source_joint_id for mapping in profile.joint_mappings})
    )
    cached, timing = cache_source_action(
        source,
        source_action,
        source_bones,
        output_sample_rate_hz=profile.output_sample_rate_hz,
    )
    target_rest_locals = {
        bone.name: parent_local_rest(bone) for bone in target.data.bones
    }
    source_rest_world = {
        mapping.source_joint_id: bone_world_rotation(
            source, source.data.bones[mapping.source_joint_id].matrix_local
        )
        for mapping in profile.joint_mappings
    }
    target_rest_world = {
        mapping.target_joint_id: bone_world_rotation(
            target, target.data.bones[mapping.target_joint_id].matrix_local
        )
        for mapping in profile.joint_mappings
    }
    root_mapping = next(
        mapping
        for mapping in profile.joint_mappings
        if mapping.semantic_joint_id == profile.root_joint_semantic_id
    )
    order = target_parent_first(target)
    target_object_rotation = target.matrix_world.to_quaternion().normalized()
    target_object_rotation_inverse = target_object_rotation.inverted()
    action = bpy.data.actions.new(name=action_mapping.output_action_name)
    target.animation_data_create()
    target.animation_data.action = action
    target.data.pose_position = "POSE"
    maximum_rotation_error_degrees = 0.0
    maximum_root_rotation_error_degrees = 0.0
    for cached_frame in cached:
        frame = int(cached_frame["frame"])
        bpy.context.scene.frame_set(frame)
        for pose_bone in target.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()

        requested_world = {}
        for target_name in order:
            mapping = mappings_by_target.get(target_name)
            if mapping is None:
                continue
            if (
                mapping.target_joint_id == root_mapping.target_joint_id
                and profile.root_rotation_policy == "target_rest"
            ):
                continue
            rest_local = target_rest_locals[target_name]
            pose_bone = target.pose.bones[target_name]
            if pose_bone.parent is None:
                translation = rest_local.translation.copy()
            else:
                translation = pose_bone.parent.matrix @ rest_local.translation
            desired_xyzw = retarget_world_rotation_xyzw(
                source_pose_world_xyzw=quaternion_xyzw(
                    cached_frame["rotations"][mapping.source_joint_id]
                ),
                source_rest_world_xyzw=quaternion_xyzw(
                    source_rest_world[mapping.source_joint_id]
                ),
                target_rest_world_xyzw=quaternion_xyzw(
                    target_rest_world[mapping.target_joint_id]
                ),
                motion_basis_xyzw=profile.motion_basis_xyzw,
                motion_amplitude=profile.motion_amplitude,
            )
            desired_world = blender_quaternion(desired_xyzw)
            desired_armature = (
                target_object_rotation_inverse @ desired_world
            ).normalized()
            pose_bone.matrix = Matrix.LocRotScale(
                translation,
                desired_armature,
                Vector((1.0, 1.0, 1.0)),
            )
            bpy.context.view_layer.update()
            requested_world[target_name] = desired_world

        for pose_bone in target.pose.bones:
            keyframe_pose_bone(pose_bone, frame)
        for target_name, requested in requested_world.items():
            actual = bone_world_rotation(target, target.pose.bones[target_name].matrix)
            error = shortest_quaternion_error_degrees(actual, requested)
            maximum_rotation_error_degrees = max(maximum_rotation_error_degrees, error)
        root_actual = bone_world_rotation(
            target, target.pose.bones[root_mapping.target_joint_id].matrix
        )
        root_expected = target_rest_world[root_mapping.target_joint_id]
        maximum_root_rotation_error_degrees = max(
            maximum_root_rotation_error_degrees,
            shortest_quaternion_error_degrees(root_actual, root_expected),
        )

    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    action.use_fake_user = True
    return action, {
        "semantic_action_id": action_mapping.semantic_action_id,
        "source_action_name": source_action.name,
        "output_action_name": action.name,
        **timing,
        "maximum_requested_world_rotation_error_degrees": (
            maximum_rotation_error_degrees
        ),
        "maximum_root_rest_rotation_error_degrees": (
            maximum_root_rotation_error_degrees
        ),
    }


def clear_animation(armature) -> None:
    if armature.animation_data is not None:
        armature.animation_data_clear()


def remove_source(objects, actions) -> None:
    for item in objects:
        if item.name in bpy.data.objects:
            bpy.data.objects.remove(item, do_unlink=True)
    for action in actions:
        if action.name in bpy.data.actions:
            bpy.data.actions.remove(action)


def remove_actions(actions) -> None:
    for action in actions:
        if action.name in bpy.data.actions:
            bpy.data.actions.remove(action)


def add_nla_tracks(target, actions) -> None:
    target.animation_data_create()
    target.animation_data.action = None
    while target.animation_data.nla_tracks:
        target.animation_data.nla_tracks.remove(target.animation_data.nla_tracks[0])
    for action in actions:
        start_value, end_value = map(float, action.frame_range)
        start, end = int(round(start_value)), int(round(end_value))
        if abs(start_value - start) > 1.0e-6 or abs(end_value - end) > 1.0e-6:
            raise RuntimeError(
                f"output action {action.name!r} has non-integral frame bounds"
            )
        track = target.animation_data.nla_tracks.new()
        track.name = action.name
        strip = track.strips.new(action.name, start, action)
        strip.name = action.name
        strip.action_frame_start = start
        strip.action_frame_end = end


def export_target(
    target, meshes, actions, output: Path, *, output_rate_hz: int
) -> None:
    add_nla_tracks(target, actions)
    bpy.context.scene.render.fps = output_rate_hz
    bpy.context.scene.render.fps_base = 1.0
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    for mesh in meshes:
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
        export_image_format="AUTO",
    )


def write_report(path: Path, payload: dict) -> None:
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    args = parse_argv()
    source_path = require_input(args.source_glb, "source GLB", {".glb", ".gltf"})
    target_path = require_input(args.target_glb, "target GLB", {".glb", ".gltf"})
    profile_path = require_input(args.profile, "motion profile", {".json"})
    output_path = require_output(args.output_glb, "output GLB")
    report_path = require_output(args.report, "retarget report")
    if output_path == report_path:
        raise SystemExit("output GLB and report paths must differ")

    profile = load_motion_retarget_profile(profile_path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = profile.output_sample_rate_hz
    bpy.context.scene.render.fps_base = 1.0

    target, target_meshes, _target_objects, target_actions = import_target(target_path)
    target_mesh_contract_before = mesh_skin_sha256(target_meshes)
    target_skeleton = skeleton_contract(target)
    target_skeleton_sha256 = canonical_json_sha256(target_skeleton)
    clear_animation(target)

    source, source_objects, source_actions, selected_actions = import_source(
        source_path, profile
    )
    source_skeleton = skeleton_contract(source)
    source_skeleton_sha256 = canonical_json_sha256(source_skeleton)
    validate_mapping(source, target, profile)

    output_actions = []
    action_records = []
    for action_mapping in profile.actions:
        action, record = bake_action(
            source,
            target,
            selected_actions[action_mapping.semantic_action_id],
            action_mapping,
            profile,
        )
        output_actions.append(action)
        action_records.append(record)

    clear_animation(source)
    remove_source(source_objects, source_actions)
    remove_actions(target_actions)
    target_mesh_contract_after = mesh_skin_sha256(target_meshes)
    if target_mesh_contract_after != target_mesh_contract_before:
        raise RuntimeError("retarget changed target topology, UVs, or skin weights")
    if canonical_json_sha256(skeleton_contract(target)) != target_skeleton_sha256:
        raise RuntimeError("retarget changed target rest skeleton")

    export_target(
        target,
        target_meshes,
        output_actions,
        output_path,
        output_rate_hz=profile.output_sample_rate_hz,
    )
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "qualification_state": QUALIFICATION_STATE,
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "profile": {
            "path": str(profile_path),
            "sha256": sha256_file(profile_path),
            "profile_id": profile.profile_id,
            "adapter_id": profile.adapter_id,
            "body_plan_id": profile.body_plan_id,
            "motion_family_id": profile.motion_family_id,
            "source_skeleton_id": profile.source_skeleton_id,
            "target_template_id": profile.target_template_id,
            "attribute_domain": {
                "size": list(profile.attribute_domain.size),
                "body_build": list(profile.attribute_domain.body_build),
                "coat_profile_id": profile.attribute_domain.coat_profile_id,
                "coat_values": list(profile.attribute_domain.coat_values),
                "life_stage": list(profile.attribute_domain.life_stage),
            },
        },
        "solver": {
            "solver_id": profile.solver_id,
            "equation": (
                "target_pose_world = basis * source_pose_world * "
                "inverse(source_rest_world) * inverse(basis) * target_rest_world"
            ),
            "motion_basis_xyzw": list(profile.motion_basis_xyzw),
            "motion_amplitude": profile.motion_amplitude,
            "time_mapping": profile.time_mapping,
            "output_sample_rate_hz": profile.output_sample_rate_hz,
            "root_rotation_policy": profile.root_rotation_policy,
            "root_translation_policy": profile.root_translation_policy,
            "unmapped_target_joint_policy": profile.unmapped_target_joint_policy,
        },
        "source_motion": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
            "skeleton_sha256": source_skeleton_sha256,
        },
        "target_authority": {
            "path": str(target_path),
            "sha256": sha256_file(target_path),
            "size_bytes": target_path.stat().st_size,
            "skeleton_sha256": target_skeleton_sha256,
            "mesh_uv_skin_sha256_before": target_mesh_contract_before,
            "mesh_uv_skin_sha256_after": target_mesh_contract_after,
            "mesh_uv_skin_unchanged": True,
            "rest_skeleton_unchanged": True,
        },
        "semantic_chains": [
            {
                "chain_id": chain.chain_id,
                "chain_kind": chain.chain_kind,
                "side": chain.side,
                "semantic_joint_ids": list(chain.semantic_joint_ids),
                "end_effector_role": chain.end_effector_role,
                "target_end_effector_joint_id": (chain.target_end_effector_joint_id),
            }
            for chain in profile.semantic_chains
        ],
        "joint_mappings": [
            {
                "semantic_joint_id": mapping.semantic_joint_id,
                "source_joint_id": mapping.source_joint_id,
                "target_joint_id": mapping.target_joint_id,
            }
            for mapping in profile.joint_mappings
        ],
        "actions": action_records,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "actions": [action.name for action in output_actions],
        },
        "required_followup": [
            "GLB round-trip and M2 action-contract validation",
            "quadruped limb-chain and contact QA",
            "Habitat fixed-state review capture",
            "hash-bound human visual review",
        ],
    }
    write_report(report_path, payload)
    print(
        f"AVENGINE_MOTION_RETARGET_OK output={output_path} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
