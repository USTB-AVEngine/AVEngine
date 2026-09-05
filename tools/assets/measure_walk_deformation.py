"""How badly a rigged animal's surface tears, over the whole walk cycle.

Two flaws in the earlier single-pose version of this measurement let a visibly
torn asset through. It sampled one frame at 35 percent through the action, and
the tearing moves around the cycle - a mild pose says nothing about the worst
one. And it weighted everything by area, while the artifact that dominates what
a viewer sees is a shard: a triangle stretched into a long thin sliver, which
fans open at an armpit or a hip and carries almost no area at all.

So this sweeps the cycle and reports the worst frame, and it measures shards by
how far a face's longest edge grew rather than by its area. Both readings are
kept per frame so a threshold can be argued against the frame it came from.

Where a shard sits matters as much as how large it is. Owner judgement on these
assets is explicit about it: tearing under the belly "is not a big problem",
while the same amount on a flank is what gets an asset rejected. Faces that point
downward and sit low on the body are therefore counted separately, so the
headline number describes the tearing a viewer can actually see.

Growth is measured against the rest pose with the action detached, in world
space, on triangulated geometry.

Example::

  blender -b --python tools/assets/measure_walk_deformation.py -- \\
    animated.glb report.json Walking 16
"""

from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
src, out_json, action_name = argv[0], argv[1], argv[2]
sample_count = int(argv[3]) if len(argv) > 3 else 16
shard_edge_growth = float(argv[4]) if len(argv) > 4 else 3.0
underside_height_fraction = float(argv[5]) if len(argv) > 5 else 0.45
max_abs_position = float(argv[6]) if len(argv) > 6 else 1.0e6
max_abs_scale = float(argv[7]) if len(argv) > 7 else 1.0e4

bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    result = bpy.ops.import_scene.gltf(filepath=src)
except Exception as error:
    raise SystemExit(f"Blender could not import animated GLB: {error}") from error
if "FINISHED" not in result:
    raise SystemExit(f"Blender could not import animated GLB: {result}")

objects = list(bpy.context.scene.objects)
armature = next((obj for obj in objects if obj.type == "ARMATURE"), None)
if armature is None or not armature.data.bones:
    raise SystemExit("animated GLB has no armature with bones")
meshes = [obj for obj in objects if obj.type == "MESH"]
if not meshes:
    raise SystemExit("animated GLB has no mesh")
skinned_meshes = [
    obj for obj in meshes
    if any(
        modifier.type == "ARMATURE" and modifier.object == armature
        for modifier in obj.modifiers
    )
]
if not skinned_meshes:
    raise SystemExit("animated GLB has no valid armature skinning")
for obj in meshes:
    if not obj.data.vertices or not obj.data.polygons:
        raise SystemExit(f"mesh {obj.name} has no valid faces or vertices")
    coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", coords)
    if not np.isfinite(coords).all():
        raise SystemExit(f"mesh {obj.name} contains NaN or Inf coordinates")
for obj in skinned_meshes:
    for vertex in obj.data.vertices:
        if not vertex.groups:
            raise SystemExit(f"vertex in {obj.name} has no skinning weights")
        total = 0.0
        for group in vertex.groups:
            weight = float(group.weight)
            if not math.isfinite(weight) or weight < 0.0:
                raise SystemExit(f"vertex in {obj.name} has invalid skinning weight")
            total += weight
        if not math.isfinite(total) or total <= 0.0:
            raise SystemExit(f"vertex in {obj.name} has no finite skinning weight")


def triangle_metrics():
    """Per-triangle area, longest edge, centroid height and normal tilt."""
    graph = bpy.context.evaluated_depsgraph_get()
    areas, longest, heights, downness = [], [], [], []
    for obj in bpy.data.objects:
        if obj not in skinned_meshes:
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        count = len(mesh.vertices)
        coords = np.empty(count * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(count, 3)
        matrix = np.array(obj.matrix_world.to_4x4())
        coords = coords @ matrix[:3, :3].T + matrix[:3, 3]
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(-1, 3)
        a, b, c = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
        normals = np.cross(b - a, c - a)
        lengths = np.linalg.norm(normals, axis=1)
        areas.append(0.5 * lengths)
        edges = np.stack([
            np.linalg.norm(b - a, axis=1),
            np.linalg.norm(c - b, axis=1),
            np.linalg.norm(a - c, axis=1),
        ], axis=1)
        longest.append(edges.max(axis=1))
        heights.append(((a + b + c) / 3.0)[:, 2])
        safe = np.where(lengths > 1e-12, lengths, 1.0)
        downness.append(normals[:, 2] / safe)
        evaluated.to_mesh_clear()
    if not areas:
        raise SystemExit("no skinned mesh in the file")
    return (np.concatenate(areas), np.concatenate(longest),
            np.concatenate(heights), np.concatenate(downness))


def numeric_bounds():
    """Return finite world coordinates and bone/object scales for this sample."""
    graph = bpy.context.evaluated_depsgraph_get()
    maximum_position = 0.0
    maximum_scale = 0.0
    for obj in objects:
        matrix = np.array(obj.matrix_world.to_4x4(), dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise SystemExit(f"{obj.name} transform contains NaN or Inf")
        location = obj.matrix_world.to_translation()
        maximum_position = max(
            maximum_position, *(abs(float(value)) for value in location)
        )
        maximum_scale = max(
            maximum_scale,
            *(abs(float(value)) for value in obj.matrix_world.to_scale()),
        )
    for pose_bone in armature.pose.bones:
        matrix = np.array(pose_bone.matrix, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise SystemExit(f"bone {pose_bone.name} transform contains NaN or Inf")
        maximum_scale = max(
            maximum_scale, *(abs(float(value)) for value in pose_bone.matrix.to_scale())
        )
    for obj in skinned_meshes:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            coords = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", coords)
            coords = coords.reshape(-1, 3)
            matrix = np.array(obj.matrix_world.to_4x4(), dtype=np.float64)
            world = coords @ matrix[:3, :3].T + matrix[:3, 3]
            if not np.isfinite(world).all():
                raise SystemExit(f"{obj.name} evaluated coordinates contain NaN or Inf")
            maximum_position = max(maximum_position, float(np.abs(world).max()))
        finally:
            evaluated.to_mesh_clear()
    if (
        not math.isfinite(maximum_position)
        or not math.isfinite(maximum_scale)
        or maximum_position > max_abs_position
        or maximum_scale > max_abs_scale
    ):
        raise SystemExit(
            "animation scale or position exceeds configured finite bounds: "
            f"position={maximum_position}, scale={maximum_scale}"
        )
    return maximum_position, maximum_scale


if armature.animation_data:
    armature.animation_data.action = None
bpy.context.view_layer.update()
rest_position, rest_scale = numeric_bounds()
rest_area, rest_edge, rest_height, _ = triangle_metrics()
maximum_position, maximum_scale = rest_position, rest_scale

action = next((a for a in bpy.data.actions
               if action_name.lower() in a.name.lower()), None)
if action is None:
    raise SystemExit(f"action {action_name} not in {[a.name for a in bpy.data.actions]}")
if armature.animation_data is None:
    armature.animation_data_create()
armature.animation_data.action = action
start, end = (int(round(v)) for v in action.frame_range)
frames = sorted({int(round(start + (end - start) * i / max(1, sample_count - 1)))
                 for i in range(sample_count)})

live = rest_area > 1e-12
series = []
for frame in frames:
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    current_position, current_scale = numeric_bounds()
    maximum_position = max(maximum_position, current_position)
    maximum_scale = max(maximum_scale, current_scale)
    area, edge, height, down = triangle_metrics()
    n = min(len(area), len(rest_area))
    growth = np.ones(n)
    growth[live[:n]] = area[:n][live[:n]] / rest_area[:n][live[:n]]
    edge_growth = np.ones(n)
    ok = rest_edge[:n] > 1e-9
    edge_growth[ok] = edge[:n][ok] / rest_edge[:n][ok]
    posed_total = float(area[:n].sum()) or 1.0
    shard = edge_growth > shard_edge_growth
    # The underside: pointing down, and low enough on the body to be the belly
    # rather than a flank that happens to face slightly downward.
    low, high = float(height[:n].min()), float(height[:n].max())
    cutoff = low + underside_height_fraction * (high - low)
    underside = (down[:n] < -0.2) & (height[:n] < cutoff)
    series.append({
        "frame": frame,
        "share_area_over_2x": round(float(area[:n][growth > 2].sum() / posed_total), 5),
        "share_area_over_4x": round(float(area[:n][growth > 4].sum() / posed_total), 5),
        "share_area_over_10x": round(float(area[:n][growth > 10].sum() / posed_total), 5),
        "share_area_shards": round(float(area[:n][shard].sum() / posed_total), 5),
        "share_area_shards_visible": round(
            float(area[:n][shard & ~underside].sum() / posed_total), 5),
        "share_area_shards_underside": round(
            float(area[:n][shard & underside].sum() / posed_total), 5),
        "shard_faces": int(shard.sum()),
        "max_area_growth": round(float(growth.max()), 1),
        "max_edge_growth": round(float(edge_growth.max()), 1),
    })

def worst(key):
    return max(series, key=lambda row: row[key])

report = {
    "schema": "avengine_generated_animal_walk_deformation_v2",
    "input": os.path.abspath(src),
    "action": action.name,
    "faces": int(len(rest_area)),
    "frames_sampled": frames,
    "mesh": {
        "valid": True,
        "finite_coordinates": True,
        "mesh_objects": len(meshes),
        "vertices": sum(len(obj.data.vertices) for obj in meshes),
        "faces": int(len(rest_area)),
    },
    "armature": {
        "present": True,
        "bones": len(armature.data.bones),
    },
    "skinning": {
        "valid": True,
        "finite_weights": True,
        "skinned_meshes": len(skinned_meshes),
        "vertex_groups": sum(len(obj.vertex_groups) for obj in skinned_meshes),
    },
    "animation_numeric_bounds": {
        "max_abs_position": maximum_position,
        "max_abs_scale": maximum_scale,
        "limits": {
            "maximum_abs_position": max_abs_position,
            "maximum_abs_scale": max_abs_scale,
        },
        "exploded": False,
    },
    "shard_edge_growth_threshold": shard_edge_growth,
    "underside_height_fraction": underside_height_fraction,
    "worst_frame_by_shards": worst("share_area_shards_visible")["frame"],
    "worst_share_area_shards": worst("share_area_shards")["share_area_shards"],
    "worst_share_area_shards_visible":
        worst("share_area_shards_visible")["share_area_shards_visible"],
    "worst_share_area_shards_underside":
        worst("share_area_shards_underside")["share_area_shards_underside"],
    "worst_shard_faces": worst("shard_faces")["shard_faces"],
    "worst_share_area_over_2x": worst("share_area_over_2x")["share_area_over_2x"],
    "worst_share_area_over_4x": worst("share_area_over_4x")["share_area_over_4x"],
    "worst_share_area_over_10x": worst("share_area_over_10x")["share_area_over_10x"],
    "worst_max_edge_growth": worst("max_edge_growth")["max_edge_growth"],
    "per_frame": series,
}
with open(out_json, "x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=1, allow_nan=False)
    handle.write("\n")
print("WALK_DEFORMATION_OK " + json.dumps(
    {k: v for k, v in report.items() if k != "per_frame"}, ensure_ascii=False))
