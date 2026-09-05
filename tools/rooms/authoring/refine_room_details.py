#!/usr/bin/env python3
"""Refine a furnished room with exportable, static close-inspection details.

This is an authoring-only pass.  It opens an existing Blender room, adds
small-scale furniture, cabinet, sink and tableware geometry, bakes the few
procedural-looking appearance patterns to ordinary PNG images, and emits a
fresh GLB/USD/Blend package plus review-only previews.  AVEngine still owns
production cameras, actors, routes and episode semantics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import bmesh
import bpy
from mathutils import Vector


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--blend", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", default="room_b_detail_v4")
    parser.add_argument("--room-id")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_collection(name: str) -> bpy.types.Collection:
    value = bpy.data.collections.get(name)
    if value is None:
        value = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(value)
    return value


def local_dimensions(obj: bpy.types.Object) -> tuple[float, float, float]:
    if obj.type != "MESH" or not obj.data.vertices:
        return tuple(float(v) for v in obj.dimensions)
    coords = [v.co for v in obj.data.vertices]
    return (
        max(float(v.x) for v in coords) - min(float(v.x) for v in coords),
        max(float(v.y) for v in coords) - min(float(v.y) for v in coords),
        max(float(v.z) for v in coords) - min(float(v.z) for v in coords),
    )


def world_from_local(obj: bpy.types.Object, point: Vector) -> Vector:
    return obj.matrix_world @ point


def create_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.name = "Material Output"
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.name = "Principled BSDF"
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    mat.diffuse_color = color
    return mat


def create_box(
    name: str,
    center: tuple[float, float, float] | Vector,
    size: tuple[float, float, float],
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
    *,
    yaw: float = 0.0,
    bevel: float = 0.0,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "_Mesh")
    builder = bmesh.new()
    bmesh.ops.create_cube(builder, size=1.0)
    for vertex in builder.verts:
        vertex.co.x *= float(size[0])
        vertex.co.y *= float(size[1])
        vertex.co.z *= float(size[2])
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.location = tuple(float(v) for v in center)
    obj.rotation_euler.z = float(yaw)
    obj.data.materials.append(mat)
    if bevel > 0.0 and min(size) > 0.0:
        modifier = obj.modifiers.new("DetailSoftEdges", "BEVEL")
        modifier.width = min(float(bevel), min(float(v) for v in size) * 0.36)
        modifier.segments = 4
        modifier.limit_method = "ANGLE"
    return obj


def create_box_local(
    name: str,
    parent: bpy.types.Object,
    local_center: tuple[float, float, float],
    size: tuple[float, float, float],
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
    *,
    bevel: float = 0.0,
) -> bpy.types.Object:
    return create_box(
        name,
        world_from_local(parent, Vector(local_center)),
        size,
        mat,
        coll,
        yaw=float(parent.rotation_euler.z),
        bevel=bevel,
    )


def create_mesh(
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def create_curve_mesh(
    name: str,
    points: list[tuple[float, float, float]],
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
    *,
    bevel_depth: float = 0.006,
    cyclic: bool = False,
    center: Vector | None = None,
    yaw: float = 0.0,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name + "_Curve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for spline_point, xyz in zip(spline.points, points):
        spline_point.co = (*xyz, 1.0)
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve)
    coll.objects.link(obj)
    if center is not None:
        obj.location = center
    obj.rotation_euler.z = yaw
    obj.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")
    obj = bpy.context.object
    obj.name = name
    return obj


def create_pipe_loop(
    name: str,
    parent: bpy.types.Object,
    points: list[tuple[float, float, float]],
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
    *,
    bevel_depth: float = 0.006,
) -> bpy.types.Object:
    center = parent.matrix_world @ Vector((0.0, 0.0, 0.0))
    # The points are in parent-local coordinates, so a yaw-only object keeps
    # the output independent of the room's world orientation.
    return create_curve_mesh(
        name,
        points,
        mat,
        coll,
        bevel_depth=bevel_depth,
        cyclic=True,
        center=center,
        yaw=float(parent.rotation_euler.z),
    )


def create_cylinder(
    name: str,
    center: tuple[float, float, float] | Vector,
    radius: float,
    depth: float,
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
    *,
    vertices: int = 32,
    yaw: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=tuple(float(v) for v in center),
    )
    obj = bpy.context.object
    obj.name = name
    # Move the object into the requested collection and remove the implicit
    # collection link created by the operator.
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    coll.objects.link(obj)
    obj.rotation_euler.z = yaw
    obj.data.materials.append(mat)
    return obj


def create_torus(
    name: str,
    center: Vector,
    major_radius: float,
    minor_radius: float,
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
    *,
    yaw: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=32,
        minor_segments=8,
        location=tuple(float(v) for v in center),
        rotation=(math.pi / 2.0, 0.0, yaw),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_Mesh"
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    coll.objects.link(obj)
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def make_textured_image(
    name: str,
    path: Path,
    generator: Callable[[float, float], tuple[float, float, float]],
    *,
    size: int = 256,
) -> bpy.types.Image:
    image = bpy.data.images.get(name)
    if image is None or image.size[0] != size or image.size[1] != size:
        if image is not None:
            bpy.data.images.remove(image)
        image = bpy.data.images.new(name, width=size, height=size, alpha=False)
    pixels: list[float] = []
    for y in range(size):
        v = y / float(size - 1)
        for x in range(size):
            u = x / float(size - 1)
            r, g, b = generator(u, v)
            pixels.extend((max(0.0, min(1.0, r)),
                           max(0.0, min(1.0, g)),
                           max(0.0, min(1.0, b)), 1.0))
    image.pixels = pixels
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    return image


def wood_pattern(
    u: float,
    v: float,
    base: tuple[float, float, float],
    contrast: float,
) -> tuple[float, float, float]:
    # Variation is mostly perpendicular to local U, so projected local UVs
    # read as long grain.  The coordinate is deliberately deterministic and
    # baked to PNG rather than left as a Blender Noise node.
    wave = 0.5 + 0.5 * math.sin(
        v * 2.0 * math.pi * 18.0 + 0.6 * math.sin(u * 2.0 * math.pi * 4.0)
    )
    fine = 0.5 + 0.5 * math.sin(
        v * 2.0 * math.pi * 71.0 + u * 2.0 * math.pi * 2.3
    )
    scale = 1.0 + contrast * ((wave - 0.5) * 0.75 + (fine - 0.5) * 0.18)
    return tuple(max(0.0, min(1.0, value * scale)) for value in base)


def fabric_pattern(u: float, v: float) -> tuple[float, float, float]:
    weave_x = 0.5 + 0.5 * math.sin(u * 2.0 * math.pi * 128.0)
    weave_y = 0.5 + 0.5 * math.sin(v * 2.0 * math.pi * 128.0 + 0.7)
    value = 0.93 + 0.045 * ((weave_x + weave_y) * 0.5 - 0.5)
    return (0.35 * value, 0.46 * value, 0.42 * value)


def roughness_pattern(u: float, v: float, value: float) -> tuple[float, float, float]:
    grain = 0.5 + 0.5 * math.sin((u + v) * 2.0 * math.pi * 23.0)
    result = max(0.0, min(1.0, value + 0.035 * (grain - 0.5)))
    return result, result, result


def projected_uvs(obj: bpy.types.Object, tile_m: float) -> None:
    if obj.type != "MESH":
        return
    mesh = obj.data
    uv = mesh.uv_layers.active or mesh.uv_layers.new(name="DetailUV")
    for polygon in mesh.polygons:
        normal = polygon.normal
        axis = max(range(3), key=lambda item: abs(float(normal[item])))
        for loop_index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            if axis == 2:
                values = (float(vertex.x), float(vertex.y))
            elif axis == 1:
                values = (float(vertex.x), float(vertex.z))
            else:
                values = (float(vertex.y), float(vertex.z))
            uv.data[loop_index].uv = (values[0] / tile_m, values[1] / tile_m)


def connect_baked_texture(
    mat: bpy.types.Material,
    image: bpy.types.Image,
    *,
    roughness: float,
    roughness_image: bpy.types.Image | None = None,
    metallic: float = 0.0,
) -> None:
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.name = "Material Output"
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.name = "Principled BSDF"
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    texcoord = nodes.new("ShaderNodeTexCoord")
    color = nodes.new("ShaderNodeTexImage")
    color.name = "BakedBaseColor"
    color.image = image
    color.extension = "REPEAT"
    color.interpolation = "Linear"
    links.new(texcoord.outputs["UV"], color.inputs["Vector"])
    links.new(color.outputs["Color"], shader.inputs["Base Color"])
    if roughness_image is not None:
        rough = nodes.new("ShaderNodeTexImage")
        rough.name = "BakedRoughness"
        rough.image = roughness_image
        rough.image.colorspace_settings.name = "Non-Color"
        rough.extension = "REPEAT"
        links.new(texcoord.outputs["UV"], rough.inputs["Vector"])
        links.new(rough.outputs["Color"], shader.inputs["Roughness"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])


def add_detail_materials(output: Path) -> dict[str, Any]:
    texture_root = output / "textures"
    texture_root.mkdir(parents=True, exist_ok=True)
    details: dict[str, Any] = {"textures": [], "materials": {}}

    def register_image(
        image: bpy.types.Image,
        relative: str,
        kind: str,
    ) -> bpy.types.Image:
        details["textures"].append({"path": relative, "kind": kind, "exists": Path(image.filepath_raw).is_file()})
        return image

    walnut = register_image(
        make_textured_image(
            "RoomB_Walnut_Baked",
            texture_root / "room_b_walnut_grain.png",
            lambda u, v: wood_pattern(u, v, (0.30, 0.14, 0.055), 0.42),
        ),
        "textures/room_b_walnut_grain.png",
        "wood_grain",
    )
    oak = register_image(
        make_textured_image(
            "RoomB_Oak_Baked",
            texture_root / "room_b_oak_grain.png",
            lambda u, v: wood_pattern(u, v, (0.57, 0.34, 0.16), 0.30),
        ),
        "textures/room_b_oak_grain.png",
        "wood_grain",
    )
    sage = register_image(
        make_textured_image(
            "RoomB_Sage_Baked",
            texture_root / "room_b_sage_grain.png",
            lambda u, v: wood_pattern(u, v, (0.22, 0.34, 0.26), 0.18),
        ),
        "textures/room_b_sage_grain.png",
        "painted_wood_grain",
    )
    fabric = register_image(
        make_textured_image(
            "RoomB_Fabric_Baked",
            texture_root / "room_b_fabric_weave.png",
            fabric_pattern,
        ),
        "textures/room_b_fabric_weave.png",
        "fabric_weave",
    )
    walnut_rough = register_image(
        make_textured_image(
            "RoomB_Walnut_Roughness",
            texture_root / "room_b_wood_roughness.png",
            lambda u, v: roughness_pattern(u, v, 0.52),
        ),
        "textures/room_b_wood_roughness.png",
        "roughness_noncolor",
    )
    fabric_rough = register_image(
        make_textured_image(
            "RoomB_Fabric_Roughness",
            texture_root / "room_b_fabric_roughness.png",
            lambda u, v: roughness_pattern(u, v, 0.82),
        ),
        "textures/room_b_fabric_roughness.png",
        "roughness_noncolor",
    )

    wood_specs = {
        "RoomB_Walnut": (walnut, walnut_rough, 0.50, 0.0, 0.38),
        "RoomB_OakFloor": (oak, walnut_rough, 0.64, 0.0, 0.30),
        "RoomB_SageCabinet": (sage, walnut_rough, 0.58, 0.0, 0.18),
    }
    for name, (image, rough, value, metallic, tile) in wood_specs.items():
        mat = bpy.data.materials.get(name)
        if mat is None:
            continue
        connect_baked_texture(mat, image, roughness=value, roughness_image=rough, metallic=metallic)
        for obj in bpy.data.objects:
            if obj.type == "MESH" and any(slot.material and slot.material.name == name for slot in obj.material_slots):
                projected_uvs(obj, tile)
        details["materials"][name] = {
            "base_color_texture": str(image.filepath_raw),
            "roughness_texture": str(rough.filepath_raw),
            "uv_tile_m": tile,
            "roughness_default": value,
            "metallic": metallic,
        }

    fabric_names = {
        "RoomB_WarmFabric": 0.84,
        "Polish_Sofa_Seat": 0.90,
        "Polish_Sofa_Back": 0.92,
        "Polish_Muted_Pillow": 0.88,
        "Polish_Warm_Pillow": 0.88,
    }
    for name, value in fabric_names.items():
        mat = bpy.data.materials.get(name)
        if mat is None:
            continue
        connect_baked_texture(mat, fabric, roughness=value, roughness_image=fabric_rough)
        for obj in bpy.data.objects:
            if obj.type == "MESH" and any(slot.material and slot.material.name == name for slot in obj.material_slots):
                projected_uvs(obj, 0.12)
        details["materials"][name] = {
            "base_color_texture": str(fabric.filepath_raw),
            "roughness_texture": str(fabric_rough.filepath_raw),
            "uv_tile_m": 0.12,
            "roughness_default": value,
            "metallic": 0.0,
        }

    for name, rough, metallic in (
        ("RoomB_BrushedSteel", 0.28, 0.86),
        ("RoomB_DarkMetal", 0.34, 0.78),
        ("RoomB_Ceramic", 0.32, 0.0),
        ("RoomB_HonedStone", 0.46, 0.0),
    ):
        mat = bpy.data.materials.get(name)
        if mat is None:
            continue
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        if shader is not None:
            shader.inputs["Roughness"].default_value = rough
            shader.inputs["Metallic"].default_value = metallic
        details["materials"].setdefault(name, {
            "base_color_texture": None,
            "roughness_texture": None,
            "uv_tile_m": None,
            "roughness_default": rough,
            "metallic": metallic,
        })
    return details


def remove_object(obj: bpy.types.Object) -> None:
    bpy.data.objects.remove(obj, do_unlink=True)


def add_curved_chair_details(
    furniture: bpy.types.Collection,
    fabric: bpy.types.Material,
    piping: bpy.types.Material,
    frame: bpy.types.Material,
    metal: bpy.types.Material,
    changes: list[str],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seats = sorted(
        (
            obj for obj in bpy.data.objects
            if obj.type == "MESH"
            and re.search(r"dining_table_chair_\d+_seat$", obj.name)
        ),
        key=lambda obj: obj.name,
    )
    require(len(seats) == 4, f"expected four dining seats, found {len(seats)}")
    for index, seat in enumerate(seats):
        prefix = seat.name.rsplit("_seat", 1)[0]
        back = bpy.data.objects.get(prefix + "_back")
        require(back is not None and back.type == "MESH", f"missing back for {seat.name}")
        seat_w, seat_d, seat_h = local_dimensions(seat)
        back_w, back_d, back_h = local_dimensions(back)
        seat_top = float(seat.matrix_world.translation.z) + seat_h * 0.5
        # Keep the source seat and all its world orientation unchanged.  These
        # pipes sit on top of it and do not become new seating surfaces.
        loop = [
            (-seat_w * 0.46, -seat_d * 0.43, seat_h * 0.5 + 0.008),
            (seat_w * 0.46, -seat_d * 0.43, seat_h * 0.5 + 0.008),
            (seat_w * 0.46, seat_d * 0.43, seat_h * 0.5 + 0.008),
            (-seat_w * 0.46, seat_d * 0.43, seat_h * 0.5 + 0.008),
        ]
        create_pipe_loop(
            f"RoomB_Detail_FramePiping_{index}",
            seat,
            loop,
            piping,
            furniture,
            bevel_depth=0.008,
        )
        # A small under-seat rail gives the chair a visible joinery layer.
        create_box_local(
            f"RoomB_Detail_UnderRailFront_{index}",
            seat,
            (0.0, -seat_d * 0.35, -0.23),
            (seat_w * 0.78, 0.035, 0.045),
            metal,
            furniture,
            bevel=0.008,
        )
        create_box_local(
            f"RoomB_Detail_UnderRailSide_{index}",
            seat,
            (seat_w * 0.36, 0.0, -0.23),
            (0.035, seat_d * 0.68, 0.045),
            metal,
            furniture,
            bevel=0.008,
        )

        # Slightly bowed backboard: front/back surfaces are closed and the
        # curve is visible without changing the measured source back object.
        half_w = min(back_w * 0.47, 0.27)
        half_h = min(back_h * 0.45, 0.31)
        thickness = min(back_d * 0.45, 0.035)
        segments = 8
        vertices: list[tuple[float, float, float]] = []
        for y_sign in (-1.0, 1.0):
            for row in range(segments + 1):
                x = -half_w + (2.0 * half_w * row / segments)
                curve = 0.024 * (1.0 - (x / half_w) ** 2)
                y = y_sign * thickness * 0.5 + curve
                vertices.extend([(x, y, -half_h), (x, y, half_h)])
        faces: list[tuple[int, ...]] = []
        stride = 2
        row_stride = (segments + 1) * stride
        for row in range(segments):
            a = row * stride
            b = (row + 1) * stride
            c = row_stride + (row + 1) * stride
            d = row_stride + row * stride
            # Front/back vertical surfaces, top/bottom faces, and no
            # self-intersecting n-gons.  This keeps each curved panel a
            # valid closed mesh for both GLB and USD reimport.
            faces.extend([
                (a, b, b + 1, a + 1),
                (d, d + 1, c + 1, c),
                (a, d, c, b),
                (a + 1, b + 1, c + 1, d + 1),
            ])
        left = 0
        right = segments * stride
        back_left = row_stride
        back_right = row_stride + segments * stride
        faces.extend([
            (left, left + 1, back_left + 1, back_left),
            (right, back_right, back_right + 1, right + 1),
        ])
        panel_center = world_from_local(back, Vector((0.0, 0.062, 0.0)))
        panel = create_mesh(
            f"RoomB_Detail_CurvedBackboard_{index}",
            [(float(x), float(y), float(z)) for x, y, z in vertices],
            faces,
            fabric,
            furniture,
        )
        panel.location = panel_center
        panel.rotation_euler.z = float(back.rotation_euler.z)
        # Pipe the front silhouette; this is a separate evaluated mesh in
        # both GLB and USD rather than a Blender-only bevel.
        front_y = -thickness * 0.5 + 0.024
        create_pipe_loop(
            f"RoomB_Detail_BackboardPiping_{index}",
            panel,
            [
                (-half_w, front_y, -half_h),
                (half_w, front_y, -half_h),
                (half_w, front_y, half_h),
                (-half_w, front_y, half_h),
            ],
            piping,
            furniture,
            bevel_depth=0.006,
        )
        # Narrow wood/metal side posts are deliberately separate pieces so
        # close views reveal assembly instead of a single slab.
        for side in (-1.0, 1.0):
            create_box_local(
                f"RoomB_Detail_BackPost_{index}_{'L' if side < 0 else 'R'}",
                back,
                (side * half_w * 0.98, 0.0, 0.0),
                (0.035, max(0.06, back_d * 0.82), back_h * 0.90),
                frame,
                furniture,
                bevel=0.007,
            )
        changes.extend([
            f"curved_backboard_{seat.name}",
            f"seat_piping_{seat.name}",
            f"chair_joinery_rails_{seat.name}",
        ])
        references.append({
            "seat_object_id": seat.name,
            "back_object_id": back.name,
            "seat_top_m": round(seat_top, 6),
            "source_front_preserved": True,
            "backboard_object_id": panel.name,
        })
    return references


def plate_mesh(
    name: str,
    center: tuple[float, float, float],
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
    *,
    outer_radius: float = 0.115,
    inner_radius: float = 0.087,
    base_z: float = 0.0,
    rim_z: float = 0.032,
) -> bpy.types.Object:
    segments = 48
    rings = [
        (outer_radius, base_z),
        (outer_radius, base_z + rim_z * 0.80),
        (inner_radius, base_z + rim_z),
        (inner_radius * 0.84, base_z + rim_z * 0.27),
    ]
    vertices: list[tuple[float, float, float]] = []
    for radius, z in rings:
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append((
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
                z,
            ))
    faces: list[tuple[int, ...]] = []
    for ring in range(len(rings) - 1):
        for index in range(segments):
            nxt = (index + 1) % segments
            a = ring * segments + index
            b = ring * segments + nxt
            c = (ring + 1) * segments + nxt
            d = (ring + 1) * segments + index
            faces.append((a, b, c, d))
    center_top = len(vertices)
    vertices.append((center[0], center[1], rings[-1][1]))
    center_bottom = len(vertices)
    vertices.append((center[0], center[1], base_z))
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.append((center_top, 3 * segments + index, 3 * segments + nxt))
        faces.append((center_bottom, nxt, index))
    return create_mesh(name, vertices, faces, mat, coll)


def cup_mesh(
    name: str,
    center: tuple[float, float, float],
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
    *,
    outer_radius: float = 0.045,
    inner_radius: float = 0.036,
    height: float = 0.095,
) -> bpy.types.Object:
    segments = 32
    z0 = center[2]
    z1 = z0 + height
    rings = [
        (outer_radius, z0),
        (outer_radius, z1),
        (inner_radius, z1),
        (inner_radius, z0 + 0.012),
    ]
    vertices: list[tuple[float, float, float]] = []
    for radius, z in rings:
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append((
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
                z,
            ))
    faces: list[tuple[int, ...]] = []
    for ring in range(len(rings) - 1):
        for index in range(segments):
            nxt = (index + 1) % segments
            a = ring * segments + index
            b = ring * segments + nxt
            c = (ring + 1) * segments + nxt
            d = (ring + 1) * segments + index
            faces.append((a, b, c, d))
    # Close the base annulus only; the top remains open and the inner wall is
    # visible in close-up.
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.append((index, nxt, 3 * segments + nxt, 3 * segments + index))
    return create_mesh(name, vertices, faces, mat, coll)


def add_hollow_tableware(
    props: bpy.types.Collection,
    ceramic: bpy.types.Material,
    metal: bpy.types.Material,
    changes: list[str],
) -> list[str]:
    old_props = sorted(
        (obj for obj in bpy.data.objects if re.match(r"dining_prop_\d+$", obj.name)),
        key=lambda obj: obj.name,
    )
    positions = [
        (2.10, -1.75),
        (3.25, -2.25),
        (2.75, -1.45),
    ]
    for obj in old_props:
        remove_object(obj)
    ids: list[str] = []
    for index, (x, y) in enumerate(positions):
        plate = plate_mesh(
            f"RoomB_Tableware_Plate_{index}",
            (x, y, 0.835),
            ceramic,
            props,
        )
        cup = cup_mesh(
            f"RoomB_Tableware_Cup_{index}",
            (x + 0.018, y + 0.012, 0.866),
            ceramic,
            props,
        )
        create_torus(
            f"RoomB_Tableware_CupHandle_{index}",
            Vector((x + 0.066, y + 0.012, 0.912)),
            0.030,
            0.006,
            ceramic,
            props,
        )
        ids.extend([plate.name, cup.name, f"RoomB_Tableware_CupHandle_{index}"])
    changes.append("replaced_solid_dining_props_with_hollow_plate_cup_sets")
    return ids


def add_rect_basin(
    name: str,
    center: tuple[float, float, float],
    outer: tuple[float, float],
    mat: bpy.types.Material,
    coll: bpy.types.Collection,
    *,
    depth: float = 0.14,
    wall: float = 0.055,
) -> bpy.types.Object:
    w, d = outer
    iw, id_ = w - 2.0 * wall, d - 2.0 * wall
    z_top = center[2]
    z_bottom = z_top - depth
    z_inner = z_bottom + 0.018
    loops = [
        [(-w / 2, -d / 2, z_top), (w / 2, -d / 2, z_top),
         (w / 2, d / 2, z_top), (-w / 2, d / 2, z_top)],
        [(-w / 2, -d / 2, z_bottom), (w / 2, -d / 2, z_bottom),
         (w / 2, d / 2, z_bottom), (-w / 2, d / 2, z_bottom)],
        [(-iw / 2, -id_ / 2, z_top), (iw / 2, -id_ / 2, z_top),
         (iw / 2, id_ / 2, z_top), (-iw / 2, id_ / 2, z_top)],
        [(-iw / 2, -id_ / 2, z_inner), (iw / 2, -id_ / 2, z_inner),
         (iw / 2, id_ / 2, z_inner), (-iw / 2, id_ / 2, z_inner)],
    ]
    vertices = [
        (center[0] + x, center[1] + y, z)
        for loop in loops for x, y, z in loop
    ]
    faces = [
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        (0, 4, 8, 9, 1), (1, 9, 10, 2), (2, 10, 11, 3), (3, 11, 8, 0),
        (8, 11, 10, 9),
        (12, 13, 14, 15),
    ]
    obj = create_mesh(name, vertices, faces, mat, coll)
    bevel = obj.modifiers.new("SinkRoundedEdges", "BEVEL")
    bevel.width = 0.018
    bevel.segments = 3
    bevel.limit_method = "ANGLE"
    return obj


def apply_countertop_sink_cutout(
    furniture: bpy.types.Collection,
    stone: bpy.types.Material,
    metal: bpy.types.Material,
    changes: list[str],
) -> dict[str, Any]:
    top = bpy.data.objects.get("kitchen_counter_run_top")
    require(top is not None and top.type == "MESH", "missing kitchen countertop")
    center = Vector((4.10, 3.22, 1.03))
    cutter = create_box(
        "RoomB_TemporarySinkCutter",
        center,
        (0.82, 0.48, 0.30),
        stone,
        furniture,
        bevel=0.035,
    )
    bpy.context.view_layer.objects.active = cutter
    cutter.select_set(True)
    bpy.context.view_layer.objects.active = top
    top.select_set(True)
    modifier = top.modifiers.new("RoomB_ActualSinkOpening", "BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.solver = "EXACT"
    modifier.object = cutter
    modifier_index = top.modifiers.find(modifier.name)
    if modifier_index > 0:
        top.modifiers.move(modifier_index, 0)
    bpy.context.view_layer.objects.active = top
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    except RuntimeError as exc:
        remove_object(cutter)
        raise RuntimeError(f"countertop sink cutout failed: {exc}") from exc
    remove_object(cutter)
    old_basin = bpy.data.objects.get("kitchen_sink_basin")
    if old_basin is not None:
        remove_object(old_basin)
    basin = add_rect_basin(
        "kitchen_sink_basin",
        (center.x, center.y, 1.005),
        (0.82, 0.48),
        metal,
        ensure_collection("Props"),
        depth=0.15,
    )
    changes.extend([
        "rounded_kitchen_countertop",
        "cut_real_sink_opening",
        "added_hollow_sink_basin",
    ])
    return {
        "countertop_object_id": top.name,
        "opening_center_m": [round(float(v), 6) for v in center],
        "opening_size_m": [0.82, 0.48, 0.30],
        "basin_object_id": basin.name,
        "cavity_status": "geometry_opening_with_hollow_basin",
    }


def add_cabinet_panels(
    furniture: bpy.types.Collection,
    props: bpy.types.Collection,
    cabinet: bpy.types.Material,
    handle: bpy.types.Material,
    dark: bpy.types.Material,
    changes: list[str],
) -> list[str]:
    ids: list[str] = []
    body = bpy.data.objects.get("kitchen_counter_run_body")
    if body is not None:
        sx, sy, sz = local_dimensions(body)
        x0 = body.location.x - sx / 2.0
        front_y = body.location.y - sy / 2.0 - 0.018
        gap = 0.026
        door_h = 0.50
        door_z = 0.43
        usable = sx - 0.22
        widths = [usable / 3.0] * 3
        x = x0 + 0.11
        for index, width in enumerate(widths):
            panel = create_box(
                f"RoomB_KitchenDoor_{index}",
                (x + width / 2.0, front_y, door_z),
                (width - gap, 0.038, door_h),
                cabinet,
                furniture,
                bevel=0.012,
            )
            ids.append(panel.name)
            handle_obj = create_box(
                f"RoomB_KitchenHandle_{index}",
                (x + width / 2.0, front_y - 0.032, door_z + 0.02),
                (0.13, 0.024, 0.024),
                handle,
                props,
                bevel=0.008,
            )
            ids.append(handle_obj.name)
            x += width
        drawer = create_box(
            "RoomB_KitchenDrawer",
            (body.location.x + 0.32, front_y - 0.003, 0.79),
            (0.60, 0.038, 0.13),
            cabinet,
            furniture,
            bevel=0.010,
        )
        ids.append(drawer.name)
        create_box(
            "RoomB_KitchenToeKick",
            (body.location.x, body.location.y - sy / 2.0 + 0.035, 0.10),
            (sx - 0.16, 0.035, 0.14),
            dark,
            furniture,
            bevel=0.008,
        )
        changes.append("added_kitchen_door_drawer_seams_and_metal_handles")
    sideboard = bpy.data.objects.get("dining_sideboard")
    if sideboard is not None:
        sx, sy, sz = local_dimensions(sideboard)
        front_x = sideboard.location.x - sx / 2.0 - 0.018
        for index, y in enumerate((-1.35, -0.85, -0.35)):
            panel = create_box(
                f"RoomB_SideboardDoor_{index}",
                (front_x, y, 0.51),
                (0.038, 0.42, 0.54),
                cabinet,
                furniture,
                bevel=0.010,
            )
            ids.append(panel.name)
            handle_obj = create_box(
                f"RoomB_SideboardHandle_{index}",
                (front_x - 0.032, y, 0.54),
                (0.024, 0.12, 0.024),
                handle,
                props,
                bevel=0.006,
            )
            ids.append(handle_obj.name)
        changes.append("added_sideboard_door_seams_and_handles")
    media = bpy.data.objects.get("living_media_shelf")
    if media is not None:
        sx, sy, sz = local_dimensions(media)
        front_y = media.location.y - sy / 2.0 - 0.018
        for index, x in enumerate((-2.15, -1.50, -0.85)):
            panel = create_box(
                f"RoomB_MediaPanel_{index}",
                (x, front_y, 0.52),
                (0.52, 0.035, 0.48),
                cabinet,
                furniture,
                bevel=0.009,
            )
            ids.append(panel.name)
        changes.append("added_living_media_panel_seams")
    return ids


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def add_review_cameras(cameras_coll: bpy.types.Collection) -> list[dict[str, Any]]:
    specs = [
        {
            "camera_id": "REVIEW_OVERALL_ROOM_B",
            "position_m": [0.0, 0.0, 7.50],
            "target_m": [0.0, 0.0, 0.0],
            "focal_length_mm": 22.0,
        },
        {
            "camera_id": "REVIEW_CLOSE_DINING_DETAIL",
            "position_m": [4.55, -3.55, 1.26],
            "target_m": [2.70, -2.00, 0.90],
            "focal_length_mm": 52.0,
        },
        {
            "camera_id": "REVIEW_CLOSE_KITCHEN_DETAIL",
            "position_m": [5.05, 2.18, 1.38],
            "target_m": [4.10, 3.20, 1.04],
            "focal_length_mm": 55.0,
        },
        {
            "camera_id": "REVIEW_CLOSE_LIVING_DETAIL",
            "position_m": [-1.20, -3.65, 1.34],
            "target_m": [-3.65, -2.55, 1.02],
            "focal_length_mm": 55.0,
        },
    ]
    result: list[dict[str, Any]] = []
    for raw in specs:
        data = bpy.data.cameras.new(raw["camera_id"] + "_Data")
        data.lens = float(raw["focal_length_mm"])
        data.sensor_width = 36.0
        camera = bpy.data.objects.new(raw["camera_id"], data)
        cameras_coll.objects.link(camera)
        camera.location = tuple(raw["position_m"])
        look_at(camera, Vector(raw["target_m"]))
        bpy.context.view_layer.update()
        forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        result.append({
            "camera_id": raw["camera_id"],
            "position_m": [round(float(v), 6) for v in camera.matrix_world.translation],
            "target_m": [round(float(v), 6) for v in raw["target_m"]],
            "forward_world": [round(float(v), 6) for v in forward],
            "focal_length_mm": round(float(data.lens), 6),
            "sensor_width_mm": round(float(data.sensor_width), 6),
            "fov_horizontal_deg": round(math.degrees(float(data.angle)), 6),
            "purpose": "review_only_asset_closeup" if "CLOSE" in raw["camera_id"] else "review_only_overall",
            "visibility_note": "ceiling_hidden_for_topdown_authoring_preview_only" if "CLOSE" not in raw["camera_id"] else "production_geometry_visible",
            "production_camera": False,
        })
    return result


def render_review_cameras(scene: bpy.types.Scene, cameras: list[dict[str, Any]], output: Path) -> None:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 576
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    ceiling = bpy.data.objects.get("Ceiling_Slab")
    original_ceiling_visibility = ceiling.hide_render if ceiling is not None else None
    for record in cameras:
        camera = bpy.data.objects.get(record["camera_id"])
        require(camera is not None, "review camera missing: " + record["camera_id"])
        # The overall authoring preview is a high view.  Hide the ceiling only
        # while that one review frame is rendered; the source object is
        # restored before Blend/GLB/USD export so production geometry is intact.
        if ceiling is not None:
            ceiling.hide_render = record["purpose"] == "review_only_overall"
        scene.camera = camera
        scene.render.filepath = str(output / "renders" / (record["camera_id"] + ".png"))
        bpy.ops.render.render(write_still=True)
    if ceiling is not None and original_ceiling_visibility is not None:
        ceiling.hide_render = original_ceiling_visibility


def load_source(input_root: Path, blend_arg: Path | None) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_root = input_root.expanduser().resolve(strict=True)
    if blend_arg is not None:
        blend = blend_arg.expanduser().resolve(strict=True)
    else:
        candidates = sorted(source_root.glob("*.blend"))
        require(len(candidates) == 1, f"expected one blend under {source_root}, got {candidates}")
        blend = candidates[0]
    semantics_path = source_root / "object_semantics.json"
    anchors_path = source_root / "functional_anchors.json"
    report_path = source_root / "polish_report.json"
    require(semantics_path.is_file(), f"missing {semantics_path}")
    require(anchors_path.is_file(), f"missing {anchors_path}")
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    return blend, semantics, anchors, report


def preserve_seat_references(
    seats: list[dict[str, Any]],
    source_seats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_anchor = {}
    for source in source_seats:
        key = source.get("anchor_id") or source.get("source_anchor_id")
        if key:
            by_anchor[str(key)] = source
    result: list[dict[str, Any]] = []
    for seat in seats:
        record = dict(seat)
        key = record.get("source_anchor_id") or record.get("anchor_id")
        source = by_anchor.get(str(key))
        if source is not None:
            source_top = source.get("seat_top_m", source.get("surface_height_m", source.get("support_height_m")))
            if source_top is not None:
                record["seat_top_m"] = float(source_top)
                record["support_height_m"] = float(source.get("support_height_m", source_top))
            if source.get("facing_vector_world") is not None:
                record["front_world"] = list(source["facing_vector_world"])
                record["facing_vector_world"] = list(source["facing_vector_world"])
                record["facing_yaw_world_deg"] = source.get("facing_yaw_world_deg")
            for field in ("table_group_id", "table_center_m", "table_distance_xy_m"):
                if field in source:
                    record[field] = source[field]
                else:
                    record.pop(field, None)
            # The v3 sidecar did not carry facing vectors for the two
            # perpendicular dining seats.  Recover only those missing fronts
            # from the unchanged chair-to-table geometry; existing vectors are
            # preserved byte-for-value.
            if record.get("front_world") is None and str(record.get("anchor_id", "")).startswith("dining_"):
                seat_obj = bpy.data.objects.get(str(record.get("scene_object_id", "")))
                table_obj = bpy.data.objects.get("dining_table_top")
                if seat_obj is not None and table_obj is not None:
                    delta = table_obj.matrix_world.translation - seat_obj.matrix_world.translation
                    delta.z = 0.0
                    norm = math.hypot(float(delta.x), float(delta.y))
                    if norm > 1e-6:
                        front = [round(float(delta.x) / norm, 6), round(float(delta.y) / norm, 6), 0.0]
                        record["front_world"] = front
                        record["facing_vector_world"] = front
                        record["facing_yaw_world_deg"] = round(math.degrees(math.atan2(front[1], front[0])), 6)
                        record["front_recovery_status"] = "derived_from_unchanged_table_geometry"
            record["seat_reference_status"] = "preserved_from_v3_sidecar"
        else:
            record["seat_top_m"] = record.get("surface_height_m")
            if record.get("facing_vector_world") is not None:
                record["front_world"] = list(record["facing_vector_world"])
            record["seat_reference_status"] = "geometry_derived_fallback"
        result.append(record)
    return result


def main() -> int:
    args = cli()
    input_root = args.input_root.expanduser().resolve(strict=True)
    blend, source_semantics, source_anchors, source_report = load_source(input_root, args.blend)
    output = args.output_root.expanduser().resolve()
    require(not output.exists(), "refusing to replace existing detail output: " + str(output))
    output.mkdir(parents=True)
    (output / "renders").mkdir()
    (output / "visual").mkdir()
    (output / "usd").mkdir()

    bpy.ops.wm.open_mainfile(filepath=str(blend))
    furniture = ensure_collection("Furniture")
    props = ensure_collection("Props")
    cameras_coll = ensure_collection("Cameras")
    architectural = ensure_collection("ArchitecturalDetails")
    # Architectural details must be present for the shared USD exporter, even
    # though these additions are static and have no production camera binding.
    _ = architectural

    changes: list[str] = []
    material_report = add_detail_materials(output)
    fabric = bpy.data.materials.get("RoomB_WarmFabric") or create_material("RoomB_WarmFabric", (0.35, 0.46, 0.42, 1.0), roughness=0.84)
    piping = create_material("RoomB_FabricPiping", (0.11, 0.16, 0.14, 1.0), roughness=0.74)
    frame = create_material("RoomB_ChairFrame", (0.19, 0.10, 0.045, 1.0), roughness=0.46)
    metal = create_material("RoomB_ChairMetal", (0.08, 0.09, 0.085, 1.0), roughness=0.34, metallic=0.78)
    ceramic = bpy.data.materials.get("RoomB_Ceramic") or create_material(
        "RoomB_Ceramic", (0.84, 0.86, 0.83, 1.0), roughness=0.30
    )
    stone = bpy.data.materials.get("RoomB_HonedStone") or create_material(
        "RoomB_HonedStone", (0.60, 0.62, 0.58, 1.0), roughness=0.46
    )
    cabinet = bpy.data.materials.get("RoomB_SageCabinet") or create_material(
        "RoomB_SageCabinet", (0.22, 0.34, 0.26, 1.0), roughness=0.58
    )
    handle = bpy.data.materials.get("RoomB_BrushedSteel") or create_material(
        "RoomB_BrushedSteel", (0.28, 0.30, 0.29, 1.0), roughness=0.28, metallic=0.86
    )
    dark = bpy.data.materials.get("RoomB_DarkMetal") or create_material(
        "RoomB_DarkMetal", (0.08, 0.09, 0.085, 1.0), roughness=0.34, metallic=0.78
    )

    chair_refs = add_curved_chair_details(
        furniture, fabric, piping, frame, metal, changes
    )
    tableware_ids = add_hollow_tableware(props, ceramic, metal, changes)
    cabinet_ids = add_cabinet_panels(furniture, props, cabinet, handle, dark, changes)
    sink_ref = apply_countertop_sink_cutout(furniture, stone, metal, changes)
    review_cameras = add_review_cameras(cameras_coll)

    # Generate all preview images after all evaluated geometry and materials are
    # present.  The camera objects remain review-only and are excluded from
    # both GLB and USD below.
    render_review_cameras(bpy.context.scene, review_cameras, output)

    room_id = args.room_id or source_semantics.get("room_spec_id") or source_semantics.get("room_id") or output.name
    room_id = str(room_id).replace(" ", "_")
    out_blend = output / f"{room_id}_detailed_v4.blend"

    # External PNGs are intentionally kept beside the output package.  GLB
    # embeds their image payload; USD receives relative texture copies.
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))

    helper = load_module(
        Path(__file__).with_name("polish_furnished_room.py"),
        "avengine_room_detail_shared",
    )
    glb = helper.export_selection(output)
    usd_module = helper.load_usd_exporter()
    usd = output / "usd" / f"{room_id}_detailed_v4.usda"
    usd_record = usd_module.export_static_usd(bpy.context.scene, usd)

    source_assemblies = source_semantics.get("furniture_assemblies", [])
    assemblies = helper.semantic_records({"furnishing_assemblies": source_assemblies})
    assemblies = helper.geometry_seat_semantics(assemblies, bpy.context.scene)
    scene_objects = helper.scene_static_mesh_records(
        bpy.context.scene,
        f"visual/{output.name}.glb",
        f"usd/{usd.name}",
    )
    seat_points = helper.complete_seat_points(assemblies, scene_objects, bpy.context.scene)
    seat_points = preserve_seat_references(
        seat_points, source_semantics.get("seat_points", [])
    )
    for assembly in assemblies:
        assembly["seat_points"] = [
            seat for seat in seat_points
            if seat.get("assembly_id") == assembly.get("object_id")
        ]
    lighting = helper.scene_lighting_records(bpy.context.scene)
    anchors = source_anchors.get("anchors", {})
    source_room_id = source_semantics.get("room_id") or source_semantics.get("room_spec_id")

    write_json(output / "review_camera_specs.json", {
        "kind": "avengine_room_detail_review_cameras",
        "status": "review_only",
        "production_camera": "deferred_to_AVEngine",
        "room_id": room_id,
        "cameras": review_cameras,
    })
    write_json(output / "object_semantics.json", {
        "kind": "avengine_complete_furniture_semantics",
        "room_id": room_id,
        "room_spec_id": source_semantics.get("room_spec_id") or source_room_id,
        "source_room_id": source_room_id,
        "source_blend": str(blend),
        "source_reference_status": "v3_final_geometry_refined_detail_candidate",
        "static_scene": True,
        "geometry_scope": "all_static_scene_meshes",
        "furniture_assemblies": assemblies,
        "objects": scene_objects,
        "furniture_objects": [
            record for record in scene_objects
            if record["category"] in {"furniture", "prop", "cabinet", "appliance"}
        ],
        "seat_points": seat_points,
        "detail_profile": args.profile,
        "detail_objects": {
            "chair_parts": chair_refs,
            "tableware_object_ids": tableware_ids,
            "cabinet_detail_object_ids": cabinet_ids,
            "sink": sink_ref,
        },
    })
    write_json(output / "functional_anchors.json", {
        "kind": "avengine_complete_functional_anchors",
        "room_id": room_id,
        "room_spec_id": source_semantics.get("room_spec_id") or source_room_id,
        "source_room_id": source_room_id,
        "source_blend": str(blend),
        "source_reference_status": "v3_final_geometry_refined_detail_candidate",
        "coordinate_system": "Blender +Z up metres; exported GLB +Y up",
        "anchors": anchors,
        "seat_points": seat_points,
        "scene_object_ids": [record["object_id"] for record in scene_objects],
        "scene_object_categories": sorted({record["category"] for record in scene_objects}),
        "wall_segment_object_ids": [
            record["object_id"] for record in scene_objects
            if record["category"] == "wall"
        ],
    })
    write_json(output / "lighting.json", {
        "kind": "avengine_scene_lighting_manifest",
        "room_id": room_id,
        "source_blend": str(blend),
        "lights": lighting,
        "world_color_rgb": [
            round(float(value), 6) for value in bpy.context.scene.world.color[:3]
        ] if bpy.context.scene.world else None,
    })
    write_json(output / "detail_report.json", {
        "kind": "avengine_room_detail_report",
        "status": "research_candidate",
        "qualification_claim": False,
        "profile": args.profile,
        "source_root": str(input_root),
        "source_blend": str(blend),
        "source_room_id": source_room_id,
        "changes": changes,
        "counts": {
            "chair_detail_groups": len(chair_refs),
            "hollow_tableware_sets": len(tableware_ids) // 3,
            "cabinet_detail_objects": len(cabinet_ids),
        },
        "seat_reference_policy": "seat_top_m_and_front_world_preserved_from_v3_sidecar",
        "material_export_policy": "procedural_looking_patterns_baked_to_png",
        "materials": material_report,
        "sink": sink_ref,
        "review_cameras": review_cameras,
        "artifacts": {
            "blend": str(out_blend),
            "visual_glb": str(glb),
            "usd": usd_record,
            "renders": str(output / "renders"),
            "textures": str(output / "textures"),
        },
        "native_execution": "pending_root_spear_ue",
    })
    # Retain the existing report shape while making the stale source paths
    # explicit and linking the detailed sidecar.
    write_json(output / "polish_report.json", {
        "kind": "avengine_polished_room_report",
        "status": "research_candidate",
        "qualification_claim": False,
        "source_blend": str(blend),
        "source_report": str(input_root / "polish_report.json"),
        "changes": changes,
        "anchors": anchors,
        "furniture_semantics": assemblies,
        "scene_object_count": len(scene_objects),
        "scene_object_categories": sorted({record["category"] for record in scene_objects}),
        "seat_point_count": len(seat_points),
        "lighting_count": len(lighting),
        "source_reference_status": "v3_final_geometry_refined_detail_candidate",
        "detail_profile": args.profile,
        "artifacts": {
            "blend": str(out_blend),
            "visual_glb": str(glb),
            "usd": usd_record,
            "renders": str(output / "renders"),
            "review_camera_specs": str(output / "review_camera_specs.json"),
            "detail_report": str(output / "detail_report.json"),
        },
        "native_execution": "pending_root_spear_ue",
    })
    print(json.dumps({
        "status": "research_candidate",
        "output": str(output),
        "blend": str(out_blend),
        "glb": str(glb),
        "usd": str(usd),
        "review_cameras": len(review_cameras),
        "seat_points": len(seat_points),
        "changes": len(changes),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
