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
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
src, out_json, action_name = argv[0], argv[1], argv[2]
sample_count = int(argv[3]) if len(argv) > 3 else 16
shard_edge_growth = float(argv[4]) if len(argv) > 4 else 3.0
underside_height_fraction = float(argv[5]) if len(argv) > 5 else 0.45

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)


def triangle_metrics():
    """Per-triangle area, longest edge, centroid height and normal tilt."""
    graph = bpy.context.evaluated_depsgraph_get()
    areas, longest, heights, downness = [], [], [], []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.vertex_groups:
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


armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if armature is None:
    raise SystemExit("no armature in the file")
if armature.animation_data:
    armature.animation_data.action = None
bpy.context.view_layer.update()
rest_area, rest_edge, rest_height, _ = triangle_metrics()

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
    "input": src,
    "action": action.name,
    "faces": int(len(rest_area)),
    "frames_sampled": frames,
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
with open(out_json, "w", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=1)
print("WALK_DEFORMATION_OK " + json.dumps(
    {k: v for k, v in report.items() if k != "per_frame"}, ensure_ascii=False))
