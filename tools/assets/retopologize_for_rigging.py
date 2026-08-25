"""Rebuild a generated mesh as a manifold surface, then reduce it for rigging.

A reconstruction arrives near a million triangles carrying thousands of
non-manifold edges, and collapse decimation cannot touch a non-manifold edge.
Reducing such a mesh directly forces the whole budget onto the clean regions:
measured on a generated Burmese, the head end kept 0.51x its fair share of
triangles while the tail end kept 1.64x, which is what "the head is completely
deformed" looks like as a number.  Curvature-weighted collapse does not help,
because the blocker is topology rather than detail.

Voxel remeshing replaces the surface with one that has no non-manifold edges
and no boundary edges at all.  After that a face-count target is reachable
exactly, the loss spreads evenly, and geometric fidelity improves: p99
deviation drops from 0.0058 to 0.0017 of the bounding diagonal against a
weld-and-collapse of the same mesh.

The remesh discards the uv layout, so the original colour is baked onto a fresh
unwrap.  Both meshes stay in one session - the bake needs the dense original as
its source, and reloading it from a file that no longer carries the texture is
the failure this avoids.

Every count here is taken after welding.  A glTF file splits vertices at each
uv and normal seam, so a boundary or island count read off a freshly imported
file measures the file format instead of the surface.

Example::

  blender -b --python tools/assets/retopologize_for_rigging.py -- \
    --input raw.glb --output retopo.glb --report retopo.json \
    --target-faces 80000 --voxel-divisor 800
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

SCHEMA = "avengine_generated_mesh_retopology_v1"


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-faces", type=int, default=80000)
    parser.add_argument(
        "--voxel-divisor", type=float, default=800.0,
        help="voxel size is the bounding diagonal over this; resolution "
             "dominates fidelity, 400 gives p99 0.0217 where 800 gives 0.0017")
    parser.add_argument("--bake-size", type=int, default=2048)
    parser.add_argument(
        "--bake-ray-fraction", type=float, default=0.02,
        help="bake ray length as a fraction of the diagonal; too generous and a "
             "ray crosses a thin wall and samples the unlit inside of the far "
             "surface, which reads as dark speckle over a pale coat")
    parser.add_argument("--dilate-passes", type=int, default=4)
    parser.add_argument(
        "--skip-remesh", action="store_true",
        help="weld and collapse without remeshing: the previous recipe, kept so "
             "the same measurement can produce the control this one is judged "
             "against")
    parser.add_argument(
        "--front-yaw-deg", type=float, default=None,
        help="the reviewed forward direction of this mesh, in the world xy "
             "plane. Without it the survival census is blind to which end of "
             "the animal is the head, and a starved head reads the same as a "
             "starved tail - 0.51 on a mesh whose face collapsed, 0.507 on one "
             "that only thinned its tail")
    parser.add_argument(
        "--relief-ratio-limit", type=float, default=6.0,
        help="remeshed faces over target faces above which the surface carries "
             "more micro-relief than the target budget can hold; the excess "
             "survives reduction as visible faceting, worst on a pale coat")
    parser.add_argument(
        "--relief-smooth-iterations", type=int, default=-1,
        help="smoothing passes over the remesh before reduction; -1 derives "
             "them from the measured relief ratio, 0 disables")
    return parser.parse_args(argv)


def welded_topology(mesh, weld):
    """Surface counts, measured on a welded copy so seams do not read as holes."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
    stat = {
        "verts": len(bm.verts),
        "faces": len(bm.faces),
        "boundary": sum(1 for edge in bm.edges if len(edge.link_faces) == 1),
        "nonmanifold": sum(1 for edge in bm.edges if len(edge.link_faces) > 2),
    }
    seen = set()
    sizes = []
    for vert in bm.verts:
        if vert in seen:
            continue
        stack = [vert]
        seen.add(vert)
        count = 0
        while stack:
            current = stack.pop()
            count += 1
            for edge in current.link_edges:
                other = edge.other_vert(current)
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        sizes.append(count)
    sizes.sort(reverse=True)
    stat["islands"] = len(sizes)
    stat["share_in_largest"] = round(sizes[0] / max(1, stat["verts"]), 4)
    bm.free()
    return stat


def main_island_coords(mesh, weld):
    """Vertex positions belonging to the source's largest connected component.

    A reconstruction ships with detached debris - the Jack Russell has 74
    islands holding 1.3 percent of its vertices.  The remesh drops that debris,
    which is correct, but a fidelity percentile computed over every original
    vertex reads the dropped debris as surface error and crosses into it exactly
    when the debris share exceeds one percent.  Measuring the body against the
    body keeps the number about the surface.
    """
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
    seen = set()
    islands = []
    for vert in bm.verts:
        if vert in seen:
            continue
        stack = [vert]
        seen.add(vert)
        group = []
        while stack:
            current = stack.pop()
            group.append(current)
            for edge in current.link_edges:
                other = edge.other_vert(current)
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        islands.append(group)
    islands.sort(key=len, reverse=True)
    total = sum(len(group) for group in islands)
    coords = [vert.co.copy() for vert in islands[0]] if islands else []
    debris = 1.0 - (len(coords) / max(1, total))
    bm.free()
    return coords, round(debris, 5)


def faceting(mesh, weld):
    """How sharply adjacent triangles meet: the stipple, measured directly.

    Faceting on a pale coat is what a reviewer sees as dark speckle.  The share
    of edges bending past thirty degrees tracks that judgement across assets
    where the relief ratio does not, because the ratio moves with both the
    remesh resolution and the face budget while this does not.

    Welding first is not optional.  A uv unwrap seams almost every triangle and
    glTF splits a vertex at every seam, so on unwelded geometry hardly any edge
    has two faces at all - 80,000 faces once yielded five usable edges.
    """
    bm = bmesh.new()
    bm.from_mesh(mesh)
    coords = [vert.co for vert in bm.verts]
    low = [min(point[i] for point in coords) for i in range(3)]
    high = [max(point[i] for point in coords) for i in range(3)]
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=math.dist(low, high) * weld)
    bm.normal_update()
    angles = []
    for edge in bm.edges:
        if len(edge.link_faces) == 2:
            first, second = edge.link_faces
            angles.append(math.degrees(math.acos(
                max(-1.0, min(1.0, first.normal.dot(second.normal))))))
    bm.free()
    if not angles:
        return {"edges": 0}
    values = np.array(angles)
    return {
        "edges": len(angles),
        "mean_deg": round(float(values.mean()), 2),
        "p95_deg": round(float(np.percentile(values, 95)), 2),
        "share_over_30deg": round(float((values > 30.0).mean()), 4),
    }


def octant_census(mesh, centre):
    """Face counts per bounding-box octant, as an axis-free fallback."""
    counts = {}
    for poly in mesh.polygons:
        mid = poly.center
        key = "".join("+" if mid[i] >= centre[i] else "-" for i in range(3))
        counts[key] = counts.get(key, 0) + 1
    return counts


def band_census(mesh, forward, low, high):
    """Face counts in the front, middle and rear thirds along the body axis.

    A reduction that treats the animal evenly loses about the same share
    everywhere.  Which end lost it is the whole question: a head reduced to flat
    facets is a ruined asset, a tail carrying fewer triangles is not.  Splitting
    along the reviewed forward direction is what tells those apart, and an
    axis-free census cannot.
    """
    span_low = min(forward.dot(low), forward.dot(high))
    span_high = max(forward.dot(low), forward.dot(high))
    for corner in ((low.x, high.y, low.z), (low.x, low.y, high.z),
                   (high.x, low.y, low.z), (high.x, high.y, low.z),
                   (high.x, low.y, high.z), (low.x, high.y, high.z)):
        value = forward.dot(Vector(corner))
        span_low = min(span_low, value)
        span_high = max(span_high, value)
    extent = max(1e-9, span_high - span_low)
    counts = {"front": 0, "middle": 0, "rear": 0}
    for poly in mesh.polygons:
        position = (forward.dot(poly.center) - span_low) / extent
        if position >= 2.0 / 3.0:
            counts["front"] += 1
        elif position >= 1.0 / 3.0:
            counts["middle"] += 1
        else:
            counts["rear"] += 1
    return counts


def survival(before, after):
    overall = sum(after.values()) / max(1, sum(before.values()))
    return {key: round((after.get(key, 0) / value) / max(1e-9, overall), 3)
            for key, value in sorted(before.items())}


def load_single_mesh(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in the input file")
    meshes.sort(key=lambda obj: len(obj.data.vertices), reverse=True)
    for extra in meshes[1:]:
        bpy.data.objects.remove(extra, do_unlink=True)
    return meshes[0]


def bounds(obj):
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    low = Vector([min(point[i] for point in points) for i in range(3)])
    high = Vector([max(point[i] for point in points) for i in range(3)])
    return low, high


def deviation_from(source_coords, target_obj, diagonal):
    """How far the original surface moved, as a fraction of the diagonal."""
    from mathutils.bvhtree import BVHTree

    bm = bmesh.new()
    bm.from_mesh(target_obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    tree = BVHTree.FromBMesh(bm)
    step = max(1, len(source_coords) // 60000)
    values = sorted(
        (tree.find_nearest(point)[3] or 0.0) for point in source_coords[::step])
    bm.free()

    def at(share):
        return round(values[min(len(values) - 1, int(len(values) * share))] / diagonal, 5)

    return {"sampled": len(values), "p50": at(0.50), "p95": at(0.95),
            "p99": at(0.99), "max": round(values[-1] / diagonal, 5)}


def dilate_atlas(image, passes):
    """Grow colour outward past each island edge.

    A texel just outside an island is transparent, and bilinear filtering pulls
    it into the visible surface as a pale rim along every seam.
    """
    width, height = image.size
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    buffer = buffer.reshape(height, width, 4)
    for _ in range(passes):
        filled = buffer[:, :, 3] > 0.0
        if filled.all():
            break
        total = np.zeros((height, width, 3), dtype=np.float32)
        hits = np.zeros((height, width), dtype=np.float32)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            neighbour = np.roll(buffer, shift, axis=axis)
            mask = np.roll(filled, shift, axis=axis)
            total += neighbour[:, :, :3] * mask[:, :, None]
            hits += mask
        grow = (~filled) & (hits > 0)
        buffer[:, :, :3] = np.where(
            grow[:, :, None], total / np.maximum(hits, 1)[:, :, None], buffer[:, :, :3])
        buffer[:, :, 3] = np.where(grow, 1.0, buffer[:, :, 3])
    image.pixels.foreach_set(buffer.reshape(-1))
    image.update()


def bake_albedo(source, target, diagonal, ray_fraction):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = diagonal * ray_fraction * 0.2
    scene.render.bake.max_ray_distance = diagonal * ray_fraction
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_clear=True)


def main():
    args = parse_argv()
    source = load_single_mesh(args.input)
    low, high = bounds(source)
    diagonal = (high - low).length
    centre = (low + high) / 2.0
    weld = diagonal * 1e-4
    coords, debris_share = main_island_coords(source.data, weld)

    report = {
        "schema": SCHEMA,
        "input": args.input,
        "diagonal": round(diagonal, 5),
        "target_faces": args.target_faces,
        "voxel_divisor": args.voxel_divisor,
        "voxel_size": round(diagonal / args.voxel_divisor, 6),
        "bake_ray_fraction": args.bake_ray_fraction,
        "stages": {"source": welded_topology(source.data, weld)},
        "source_debris_share": debris_share,
        "formal_dataset_registration_authorized": False,
    }
    octants_before = octant_census(source.data, centre)
    forward = None
    if args.front_yaw_deg is not None:
        yaw = math.radians(args.front_yaw_deg)
        forward = Vector((math.cos(yaw), math.sin(yaw), 0.0))
        bands_before = band_census(source.data, forward, low, high)

    bpy.context.view_layer.objects.active = source
    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    bpy.ops.object.duplicate()
    target = bpy.context.view_layer.objects.active
    target.name = "retopologized"

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=weld)
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    if args.skip_remesh:
        report["remeshed_skipped"] = True
    else:
        modifier = target.modifiers.new("remesh", "REMESH")
        modifier.mode = "VOXEL"
        modifier.voxel_size = diagonal / args.voxel_divisor
        modifier.adaptivity = 0.0
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    report["stages"]["remeshed"] = welded_topology(target.data, weld)

    # A reconstruction whose surface carries fur-scale relief remeshes into far
    # more faces than a smooth one of the same size.  That relief is sub-pixel
    # at a million faces and reads as fur; at the rigging budget it survives as
    # irregular facets, which on a pale coat reads as dark stipple.  The ratio
    # says how much of it the target cannot hold, so it also says how much to
    # smooth away first.
    relief_ratio = report["stages"]["remeshed"]["faces"] / max(1, args.target_faces)
    if args.relief_smooth_iterations >= 0:
        smoothing = args.relief_smooth_iterations
    elif relief_ratio > args.relief_ratio_limit:
        smoothing = int(round(4.0 * (relief_ratio - args.relief_ratio_limit)))
    else:
        smoothing = 0
    report["relief_ratio"] = round(relief_ratio, 3)
    report["relief_smooth_iterations"] = smoothing
    if smoothing:
        modifier = target.modifiers.new("relief_smooth", "SMOOTH")
        modifier.factor = 0.5
        modifier.iterations = smoothing
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        report["stages"]["relief_smoothed"] = welded_topology(target.data, weld)

    for _ in range(6):
        current = len(target.data.polygons)
        if current <= args.target_faces * 1.05:
            break
        modifier = target.modifiers.new("decimate", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.use_collapse_triangulate = True
        modifier.ratio = min(1.0, args.target_faces / current)
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        if len(target.data.polygons) >= current:
            break
    report["stages"]["decimated"] = welded_topology(target.data, weld)

    octant_survival = survival(octants_before, octant_census(target.data, centre))
    report["octant_survival"] = octant_survival
    report["octant_survival_note"] = (
        "1.0 means the octant lost the same share as the mesh overall; blind to "
        "which end of the animal it is, so it informs rather than decides")
    report["octant_survival_span"] = [min(octant_survival.values()),
                                      max(octant_survival.values())]
    if forward is not None:
        report["band_survival"] = survival(
            bands_before, band_census(target.data, forward, low, high))
        report["band_survival_note"] = (
            "thirds along the reviewed forward direction; the front third holds "
            "the head, and it is the one that must not starve")
    report["fidelity_over_diagonal"] = deviation_from(coords, target, diagonal)
    report["faceting"] = faceting(target.data, 1e-4)
    report["fidelity_note"] = (
        "sampled from the source's largest island only; detached debris the "
        "remesh drops is reported as source_debris_share instead")

    target.data.materials.clear()
    material = bpy.data.materials.new("retopologized_albedo")
    material.use_nodes = True
    target.data.materials.append(material)
    image = bpy.data.images.new("albedo", args.bake_size, args.bake_size)
    nodes = material.node_tree.nodes
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    material.node_tree.links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    nodes.active = texture

    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")

    bake_albedo(source, target, diagonal, args.bake_ray_fraction)
    dilate_atlas(image, args.dilate_passes)
    report["atlas_dilate_passes"] = args.dilate_passes

    bpy.data.objects.remove(source, do_unlink=True)
    image.pack()
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.ops.export_scene.gltf(filepath=args.output, export_format="GLB", use_selection=True)
    report["output"] = args.output
    report["output_bytes"] = os.path.getsize(args.output)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print("RETOPOLOGY_OK " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
