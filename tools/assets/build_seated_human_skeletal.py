"""Build a seated Rocketbox human skeletal asset for AVEngine SPEAR research runs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    raw = os.sys.argv
    argv = raw[raw.index("--") + 1 :] if "--" in raw else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-glb", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--display-label", required=True)
    parser.add_argument("--color-name", required=True)
    parser.add_argument("--shirt-color-rgb", default=None)
    parser.add_argument("--emitter-offset-blender-m", required=True)
    parser.add_argument("--seat-anchor-id", required=True)
    parser.add_argument("--seat-top-m", type=float, default=0.53)
    parser.add_argument("--chair-center-blender-m", default="0,0.18,0")
    parser.add_argument("--floor-correction-m", type=float, default=-0.01)
    parser.add_argument("--animation-name", default="Seated_Idle")
    return parser.parse_args(argv)


def vec3(text: str, owner: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in text.split(","))
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{owner} must contain three finite values")
    return values


def reset_pose(armature: Any) -> None:
    if armature.animation_data:
        armature.animation_data_clear()
    for bone in armature.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def world_point(armature: Any, point: Vector) -> Vector:
    return armature.matrix_world @ point


def rotate_joint_toward(
    armature: Any, joint_name: str, child_name: str, target: tuple[float, float, float]
) -> dict[str, Any]:
    joint = armature.pose.bones[joint_name]
    child = armature.pose.bones[child_name]
    joint_world = world_point(armature, joint.head)
    child_world = world_point(armature, child.head)
    current = child_world - joint_world
    requested = Vector(target) - joint_world
    if current.length < 1.0e-7 or requested.length < 1.0e-7:
        raise ValueError((joint_name, child_name, current.length, requested.length))
    desired = requested.normalized() * current.length
    delta = current.rotation_difference(desired)
    transformed = (
        Matrix.Translation(joint_world)
        @ delta.to_matrix().to_4x4()
        @ Matrix.Translation(-joint_world)
        @ (armature.matrix_world @ joint.matrix)
    )
    joint.matrix = armature.matrix_world.inverted() @ transformed
    bpy.context.view_layer.update()
    return {
        "joint": joint_name,
        "child": child_name,
        "requested_child_world": list(target),
        "actual_child_world": list(world_point(armature, child.head)),
        "segment_length": float(current.length),
    }


def skinned_meshes(scene: Any, armature: Any) -> list[Any]:
    values = [
        obj
        for obj in scene.objects
        if obj.type == "MESH"
        and any(
            modifier.type == "ARMATURE" and modifier.object == armature
            for modifier in obj.modifiers
        )
    ]
    if not values:
        raise RuntimeError("source GLB contains no armature-skinned mesh")
    return values


def body_material_and_image() -> tuple[Any, Any]:
    material = next(
        (item for item in bpy.data.materials if item.name.endswith("_body") and item.use_nodes),
        None,
    )
    if material is None:
        raise RuntimeError("no *_body material found")
    image = next(
        (
            node.image
            for node in material.node_tree.nodes
            if node.type == "TEX_IMAGE" and node.image and node.image.name.endswith("_body_color")
        ),
        None,
    )
    if image is None:
        raise RuntimeError(f"no *_body_color image found in {material.name}")
    return material, image


def tint_shirt(meshes: list[Any], rgb_text: str | None, color_name: str) -> dict[str, Any]:
    base, source = body_material_and_image()
    report = {
        "color_name": color_name,
        "source_material": base.name,
        "source_texture": source.name,
        "mode": "original_texture" if rgb_text is None else "uv_sampled_face_selection_with_luminance_texture_tint",
        "changed_faces": 0,
    }
    if rgb_text is None:
        return report
    target = np.asarray(vec3(rgb_text, "--shirt-color-rgb"), dtype=np.float32)
    width, height = source.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    source.pixels.foreach_get(pixels)
    pixels = pixels.reshape(height, width, 4)
    source_pixels = pixels.copy()
    luminance = pixels[:, :, :3].mean(axis=2)
    pixels[:, :, :3] = np.clip(
        0.18 * pixels[:, :, :3]
        + 0.82 * target[None, None, :] * np.clip(0.42 + 0.86 * luminance[:, :, None], 0.0, 1.25),
        0.0,
        1.0,
    )
    image = bpy.data.images.new(f"seated_{color_name}_body_color", width=width, height=height, alpha=True)
    image.pixels.foreach_set(pixels.reshape(-1))
    image.pack()
    tinted = base.copy()
    tinted.name = f"{base.name.rsplit('_body', 1)[0]}_seated_{color_name}"
    for node in tinted.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image == source:
            node.image = image
    changed = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for mesh_object in meshes:
        body_index = next(
            (index for index, slot in enumerate(mesh_object.data.materials) if slot == base),
            None,
        )
        if body_index is None or mesh_object.data.uv_layers.active is None:
            continue
        tinted_index = len(mesh_object.data.materials)
        mesh_object.data.materials.append(tinted)
        uv_data = mesh_object.data.uv_layers.active.data
        evaluated = mesh_object.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        centers = [evaluated.matrix_world @ polygon.center for polygon in evaluated_mesh.polygons]
        for polygon_index, polygon in enumerate(mesh_object.data.polygons):
            if polygon.material_index != body_index:
                continue
            center = centers[polygon_index]
            if not 0.72 <= center.z <= 1.42:
                continue
            samples = []
            for loop_index in polygon.loop_indices:
                uv = uv_data[loop_index].uv
                x = int(round((float(uv.x) % 1.0) * (width - 1)))
                y = int(round((float(uv.y) % 1.0) * (height - 1)))
                samples.append(source_pixels[y, x, :3])
            color = np.mean(samples, axis=0)
            if float(color.max() - color.min()) < 0.20 and float(color.mean()) > 0.23:
                polygon.material_index = tinted_index
                changed += 1
        evaluated.to_mesh_clear()
    report.update({"changed_faces": changed, "tinted_material": tinted.name})
    return report


def make_constant_action(armature: Any, name: str) -> dict[str, Any]:
    old = bpy.data.actions.get(name)
    if old is not None:
        bpy.data.actions.remove(old)
    action = bpy.data.actions.new(name)
    armature.animation_data_create()
    armature.animation_data.action = action
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 2
    for frame in (1, 2):
        bpy.context.scene.frame_set(frame)
        for bone in armature.pose.bones:
            bone.rotation_mode = "QUATERNION"
            bone.keyframe_insert(data_path="location", frame=frame, group=bone.name)
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone.name)
            bone.keyframe_insert(data_path="scale", frame=frame, group=bone.name)
    for group in action.groups:
        for channel in group.channels:
            for key in channel.keyframe_points:
                key.interpolation = "CONSTANT"
    action.frame_start = 1.0
    action.frame_end = 2.0
    return {"name": name, "frame_start": 1, "frame_end": 2, "sample_count": 2}


def bounds(armature: Any, meshes: list[Any]) -> tuple[list[float], list[float], int]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        points.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
        evaluated.to_mesh_clear()
    if not points:
        raise RuntimeError("seated pose has no evaluated vertices")
    array = np.asarray([[float(point[axis]) for axis in range(3)] for point in points], dtype=np.float64)
    return array.min(axis=0).tolist(), array.max(axis=0).tolist(), len(points)


def export_skeletal(armature: Any, meshes: list[Any], destination: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=str(destination),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_nla_strips=False,
        export_frame_range=True,
        export_force_sampling=True,
        export_skins=True,
        export_materials="EXPORT",
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "asset").mkdir(parents=True)
    source = args.human_glb.resolve(strict=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    scene = bpy.context.scene
    scene.frame_set(1)
    armature = next(obj for obj in scene.objects if obj.type == "ARMATURE")
    meshes = skinned_meshes(scene, armature)
    reset_pose(armature)
    color_report = tint_shirt(meshes, args.shirt_color_rgb, args.color_name)
    armature.location += Vector((0.0, 0.10, -0.225))
    bpy.context.view_layer.update()
    operations = []
    for side in ("L", "R"):
        sign = 1.0 if side == "L" else -1.0
        thigh, calf, foot, toe = (f"Bip01 {side} {part}" for part in ("Thigh", "Calf", "Foot", "Toe0"))
        hip = world_point(armature, armature.pose.bones[thigh].head)
        operations.append(rotate_joint_toward(armature, thigh, calf, (hip.x, -0.284, 0.515)))
        operations.append(rotate_joint_toward(armature, calf, foot, (sign * 0.12, -0.264, 0.118)))
        ankle = world_point(armature, armature.pose.bones[foot].head)
        operations.append(rotate_joint_toward(armature, foot, toe, (ankle.x, ankle.y - 0.152, ankle.z - 0.076)))
        upper, forearm, hand = (f"Bip01 {side} {part}" for part in ("UpperArm", "Forearm", "Hand"))
        shoulder = world_point(armature, armature.pose.bones[upper].head)
        operations.append(rotate_joint_toward(armature, upper, forearm, (shoulder.x + sign * 0.06, shoulder.y - 0.10, shoulder.z - 0.26)))
        elbow = world_point(armature, armature.pose.bones[forearm].head)
        operations.append(rotate_joint_toward(armature, forearm, hand, (elbow.x - sign * 0.05, elbow.y - 0.18, elbow.z - 0.20)))
    action_report = make_constant_action(armature, args.animation_name)
    emitter_offset = vec3(args.emitter_offset_blender_m, "--emitter-offset-blender-m")
    emitter_world = list(world_point(armature, Vector(emitter_offset)))
    minimum, maximum, vertex_count = bounds(armature, meshes)
    glb_path = output / "asset" / f"{args.asset_id}.glb"
    export_skeletal(armature, meshes, glb_path)
    report = {
        "schema": "avengine_seated_skeletal_asset_report_v1",
        "status": "research_only",
        "asset_id": args.asset_id,
        "display_label": args.display_label,
        "source_human_glb": str(source),
        "static_pose_reused_from": "avengine_life_rooms_v1/render_seated_pose_candidate.py",
        "color_variant": color_report,
        "operations": operations,
        "action": action_report,
        "seat_reference": {
            "anchor_id": args.seat_anchor_id,
            "seat_top_m": float(args.seat_top_m),
            "chair_center_blender_m": list(vec3(args.chair_center_blender_m, "chair_center_blender_m")),
            "floor_correction_m": float(args.floor_correction_m),
            "reference_is_not_actor_root": True,
        },
        "emitter": {
            "anchor_id": "mouth",
            "anchor_type": "mouth",
            "offset_space": "seated_asset_local_root_blender_xyz_m",
            "offset_blender_m": list(emitter_offset),
            "world_readback_blender_m": emitter_world,
            "avengine_world_offset_m": [emitter_offset[0], emitter_offset[2], -emitter_offset[1]],
        },
        "bounds_blender_m": {"minimum": minimum, "maximum": maximum},
        "vertex_count": vertex_count,
        "skeletal_mesh": {"glb": str(glb_path), "animation": args.animation_name},
        "claim_boundary": "static seated idle only; no sit transition, lip animation, or furniture interaction",
    }
    (output / "seated_pose_report.json").write_text(json.dumps(report, indent=2) + chr(10), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "seated_pose_source.blend"))
    print("SEATED_SKELETAL_ASSET_COMPLETE", output)


if __name__ == "__main__":
    main()
