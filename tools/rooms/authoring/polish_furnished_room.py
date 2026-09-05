#!/usr/bin/env python3
"""Improve furnished room presentation while preserving room geometry and anchors.

The tool opens an existing research Blender room, adds reversible-looking
furniture/detail meshes to dedicated existing Furniture/Props collections,
adjusts material response, and exports a fresh Blender/GLB/USD package.
Production cameras, actors, routes and QA remain deferred to AVEngine.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Vector


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--room-id")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collection(name: str) -> bpy.types.Collection:
    value = bpy.data.collections.get(name)
    if value is None:
        value = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(value)
    return value


def material(name: str, color: tuple[float, float, float, float],
             roughness: float = 0.55, metallic: float = 0.0):
    value = bpy.data.materials.get(name)
    if value is None:
        value = bpy.data.materials.new(name)
    value.use_nodes = True
    value.diffuse_color = color
    bsdf = value.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if any(token in name.casefold() for token in (
            "polish_sofa", "polish_muted_pillow", "polish_warm_pillow",
        )):
            nodes = value.node_tree.nodes
            links = value.node_tree.links
            noise = nodes.get("PolishFabricNoise")
            if noise is None:
                noise = nodes.new("ShaderNodeTexNoise")
                noise.name = "PolishFabricNoise"
            noise.inputs["Scale"].default_value = 62.0
            noise.inputs["Detail"].default_value = 2.0
            noise.inputs["Roughness"].default_value = 0.7
            bump = nodes.get("PolishFabricBump")
            if bump is None:
                bump = nodes.new("ShaderNodeBump")
                bump.name = "PolishFabricBump"
            bump.inputs["Strength"].default_value = 0.08
            bump.inputs["Distance"].default_value = 0.018
            if not any(link.to_node == bump and link.to_socket.name == "Height"
                       for link in links):
                links.new(noise.outputs["Fac"], bump.inputs["Height"])
            if not any(link.to_node == bsdf and link.to_socket.name == "Normal"
                       for link in links):
                links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return value


def cube(name: str, center: tuple[float, float, float],
         size: tuple[float, float, float], mat, coll,
         *, bevel: float = 0.025, yaw: float = 0.0):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    builder = bmesh.new()
    bmesh.ops.create_cube(builder, size=1.0)
    for vertex in builder.verts:
        vertex.co.x *= size[0]
        vertex.co.y *= size[1]
        vertex.co.z *= size[2]
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.location = center
    obj.rotation_euler.z = yaw
    obj.data.materials.append(mat)
    if bevel:
        modifier = obj.modifiers.new("PolishSoftEdges", "BEVEL")
        modifier.width = min(bevel, min(size) * 0.38)
        modifier.segments = 6
        modifier.limit_method = "ANGLE"
    return obj


def round_edges(obj: bpy.types.Object, width: float, segments: int = 8) -> None:
    for modifier in obj.modifiers:
        if modifier.type == "BEVEL":
            modifier.width = width
            modifier.segments = segments
            modifier.limit_method = "ANGLE"
            modifier.affect = "EDGES"


def cyl(name: str, center: tuple[float, float, float], radius: float,
        depth: float, mat, coll, *, vertices: int = 32):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    builder = bmesh.new()
    bmesh.ops.create_cone(builder, cap_ends=True, cap_tris=False,
                          segments=vertices, radius1=radius, radius2=radius,
                          depth=depth)
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.location = center
    obj.data.materials.append(mat)
    return obj


def local_point(obj: bpy.types.Object, dx: float, dy: float, dz: float = 0.0) -> Vector:
    yaw = float(obj.rotation_euler.z)
    return obj.location + Vector((
        math.cos(yaw) * dx - math.sin(yaw) * dy,
        math.sin(yaw) * dx + math.cos(yaw) * dy,
        dz,
    ))


def soften_materials() -> list[str]:
    changed: list[str] = []
    for value in bpy.data.materials:
        name = value.name.casefold()
        if not any(token in name for token in ("oakfloor", "woodfloor", "darkoak", "lightoak", "walnut")):
            continue
        if not value.use_nodes:
            continue
        bsdf = value.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            continue
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = max(
                float(bsdf.inputs["Roughness"].default_value), 0.62
            )
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = min(
                float(bsdf.inputs["Metallic"].default_value), 0.05
            )
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.28
        rough_links = [
            link for link in value.node_tree.links
            if link.to_node == bsdf and link.to_socket.name == "Roughness"
        ]
        if rough_links:
            ramp = value.node_tree.nodes.get("PolishFloorRoughnessRemap")
            if ramp is None:
                ramp = value.node_tree.nodes.new("ShaderNodeValToRGB")
                ramp.name = "PolishFloorRoughnessRemap"
            ramp.color_ramp.elements[0].position = 0.0
            ramp.color_ramp.elements[0].color = (0.48, 0.48, 0.48, 1.0)
            ramp.color_ramp.elements[1].position = 1.0
            ramp.color_ramp.elements[1].color = (0.86, 0.86, 0.86, 1.0)
            source = rough_links[0].from_socket
            for link in rough_links:
                value.node_tree.links.remove(link)
            value.node_tree.links.new(source, ramp.inputs["Fac"])
            value.node_tree.links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])
        changed.append(value.name)
    return sorted(set(changed))


def polish_sofas(furniture, props, changes: list[str]) -> None:
    pillow_mat = material("Polish_Muted_Pillow", (0.42, 0.53, 0.51, 1.0), 0.88)
    pillow_alt = material("Polish_Warm_Pillow", (0.46, 0.28, 0.18, 1.0), 0.88)
    seat_mat = material("Polish_Sofa_Seat", (0.37, 0.47, 0.44, 1.0), 0.92)
    back_mat = material("Polish_Sofa_Back", (0.33, 0.42, 0.40, 1.0), 0.94)
    leg_mat = material("Polish_Sofa_Leg", (0.07, 0.055, 0.045, 1.0), 0.48, 0.15)
    def is_sofa_part(obj: bpy.types.Object, part: str) -> bool:
        name = obj.name.casefold()
        return obj.type == "MESH" and "sofa" in name and part in name

    back_objects = [obj for obj in furniture.objects if is_sofa_part(obj, "back")]
    bases = [obj for obj in furniture.objects if is_sofa_part(obj, "base")]
    for index, base in enumerate(sorted(bases, key=lambda obj: obj.name)):
        base.data.materials.clear()
        base.data.materials.append(back_mat)
        if not any(mod.name == "PolishSofaBodyRoundover" for mod in base.modifiers):
            modifier = base.modifiers.new("PolishSofaBodyRoundover", "BEVEL")
            modifier.width = 0.085
            modifier.segments = 6
            modifier.limit_method = "ANGLE"
        dims = base.dimensions
        for leg_index, (dx, dy) in enumerate((
            (-0.38 * dims.x, -0.28 * dims.y),
            (0.38 * dims.x, -0.28 * dims.y),
            (-0.38 * dims.x, 0.28 * dims.y),
            (0.38 * dims.x, 0.28 * dims.y),
        )):
            point = local_point(base, dx, dy, -0.03)
            cube(f"PolishSofaLeg_{index}_{leg_index}",
                 (point.x, point.y, max(0.09, point.z - 0.08)),
                 (0.09, 0.09, 0.16), leg_mat, furniture,
                 bevel=0.018, yaw=float(base.rotation_euler.z))
        changes.append(f"softened_sofa_{base.name}")
    for index, back in enumerate(sorted(back_objects, key=lambda obj: obj.name)):
        back.data.materials.clear()
        back.data.materials.append(back_mat)
        if not any(mod.name == "PolishSofaBodyRoundover" for mod in back.modifiers):
            modifier = back.modifiers.new("PolishSofaBodyRoundover", "BEVEL")
            modifier.width = 0.075
            modifier.segments = 6
            modifier.limit_method = "ANGLE"
        dims = back.dimensions
        for pillow_index, dx in enumerate((-0.31 * dims.x, 0.31 * dims.x)):
            point = local_point(back, dx, -0.16, 0.08)
            pillow = cube(
                f"PolishSofaPillow_{index}_{pillow_index}",
                (point.x, point.y, point.z + 0.18),
                (0.46, 0.26, 0.40),
                pillow_mat if pillow_index == 0 else pillow_alt,
                furniture,
                bevel=0.10,
                yaw=float(back.rotation_euler.z),
            )
            round_edges(pillow, 0.085, 8)
        changes.append(f"added_soft_pillows_{back.name}")
    for obj in furniture.objects:
        if obj.type != "MESH" or "cushion" not in obj.name.casefold():
            continue
        obj.data.materials.clear()
        obj.data.materials.append(seat_mat)
        original_top = obj.location.z + (obj.dimensions.z * 0.5)
        obj.scale.z *= 1.45
        bpy.context.view_layer.update()
        obj.location.z += original_top - (obj.location.z + obj.dimensions.z * 0.5)
        modifier = obj.modifiers.new("PolishCushionRoundover", "BEVEL")
        modifier.width = 0.105
        modifier.segments = 8
        modifier.limit_method = "ANGLE"
        modifier.affect = "EDGES"
        changes.append(f"rounded_cushion_{obj.name}")
    for index, base in enumerate(sorted(bases, key=lambda obj: obj.name)):
        for back_index, dx in enumerate((-0.78, 0.0, 0.78)):
            point = local_point(base, dx, 0.34, 0.55)
            back_cushion = cube(
                f"PolishSofaBackCushion_{index}_{back_index}",
                (point.x, point.y + 0.02, point.z + 0.31),
                (0.70, 0.30, 0.60),
                back_mat,
                furniture,
                bevel=0.14,
                yaw=float(base.rotation_euler.z),
            )
            round_edges(back_cushion, 0.105, 8)
        changes.append(f"segmented_sofa_back_{base.name}")


def leaf_blade(name: str, center: Vector, width: float, height: float,
               mat, coll, *, yaw: float, tilt_x: float, tilt_y: float):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    half = width * 0.5
    verts = [
        (0.0, 0.0, 0.0),
        (-half, 0.0, height * 0.28),
        (0.0, 0.0, height),
        (half, 0.0, height * 0.28),
        (0.0, 0.028, height * 0.32),
        (0.0, 0.036, height * 0.70),
    ]
    faces = [
        (0, 1, 4),
        (1, 2, 5, 4),
        (2, 3, 5),
        (3, 0, 4, 5),
    ]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.location = center
    obj.rotation_euler = (tilt_x, tilt_y, yaw)
    obj.data.materials.append(mat)
    solidify = obj.modifiers.new("PolishLeafThickness", "SOLIDIFY")
    solidify.thickness = 0.009
    solidify.offset = 0.0
    return obj


def polish_plants(furniture, props, changes: list[str]) -> None:
    leaf_mat = material("Polish_Leaf_Green", (0.08, 0.29, 0.13, 1.0), 0.82)
    leaf_light = material("Polish_Leaf_Light", (0.18, 0.43, 0.20, 1.0), 0.84)
    stem_mat = material("Polish_Stem", (0.08, 0.16, 0.06, 1.0), 0.92)
    pots = [obj for obj in props.objects
            if obj.type == "MESH" and "plantpot" in obj.name.casefold()]
    if not pots:
        pots = [obj for obj in furniture.objects
                if obj.type == "MESH" and "plantpot" in obj.name.casefold()]
    old_leaves = [
        obj for coll in (furniture, props)
        for obj in coll.objects
        if obj.type == "MESH" and (
            "plantleaf" in obj.name.casefold() or "plant_leaf" in obj.name.casefold()
        )
    ]
    for old in old_leaves:
        old.hide_render = True
        old.hide_set(True)
    for plant_index, pot in enumerate(sorted(pots, key=lambda obj: obj.name)):
        base = pot.location.copy() + Vector((0.0, 0.0, 0.38))
        for leaf_index in range(7):
            angle = (2.0 * math.pi * leaf_index / 7.0) + 0.21 * plant_index
            radius = 0.045 + 0.018 * (leaf_index % 2)
            point = base + Vector((
                math.cos(angle) * radius,
                math.sin(angle) * radius,
                0.0,
            ))
            leaf = leaf_blade(
                f"PolishPlantLeaf_{plant_index}_{leaf_index}",
                point,
                0.13 + 0.018 * (leaf_index % 3),
                0.52 + 0.08 * (leaf_index % 3),
                leaf_mat if leaf_index % 2 else leaf_light,
                props,
                yaw=angle,
                tilt_x=math.radians(12.0 + 5.0 * math.sin(angle)),
                tilt_y=math.radians(18.0 * math.cos(angle)),
            )
            changes.append(f"leaf_blade_{leaf.name}")
        for stem_index in range(4):
            angle = 2.0 * math.pi * stem_index / 4.0
            point = pot.location + Vector((
                math.cos(angle) * 0.035,
                math.sin(angle) * 0.035,
                0.26,
            ))
            cyl(
                f"PolishPlantStem_{plant_index}_{stem_index}",
                (point.x, point.y, point.z),
                0.009, 0.52, stem_mat, props,
            )
        changes.append(f"replaced_rod_plant_{pot.name}")

def repair_compact_dining_chairs(scene: bpy.types.Scene,
                                  changes: list[str]) -> None:
    table = next((
        obj for obj in scene.objects
        if obj.type == "MESH" and "dining_table_top" in obj.name.casefold()
    ), None)
    if table is None:
        return
    table_point = table.matrix_world.translation
    for seat in sorted((
        obj for obj in scene.objects
        if obj.type == "MESH"
        and "dining_table_chair" in obj.name.casefold()
        and obj.name.casefold().endswith("_seat")
    ), key=lambda obj: obj.name):
        prefix = seat.name[:-5]
        back = bpy.data.objects.get(prefix + "_back")
        if back is None:
            continue
        point = seat.matrix_world.translation
        dx = float(point.x - table_point.x)
        dy = float(point.y - table_point.y)
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            continue
        outward_x, outward_y = dx / distance, dy / distance
        back.location = (
            float(point.x) + outward_x * 0.24,
            float(point.y) + outward_y * 0.24,
            float(back.location.z),
        )
        back.rotation_euler.z = math.atan2(outward_y, outward_x) - math.pi / 2.0
        changes.append(f"aligned_compact_dining_back_{back.name}")


def fix_lighting(scene: bpy.types.Scene, changes: list[str]) -> None:
    for light in scene.objects:
        if light.type != "LIGHT" or "ceilingfill" not in light.name.casefold():
            continue
        light.rotation_euler = (0.0, 0.0, 0.0)
        light.data.energy = float(light.data.energy) * 1.15
        changes.append(f"aimed_downward_{light.name}")
    if bpy.data.objects.get("Ceiling_great_room") is None:
        return
    if bpy.data.objects.get("PolishRearWallWash") is not None:
        return
    data = bpy.data.lights.new("PolishRearWallWashData", "AREA")
    obj = bpy.data.objects.new("PolishRearWallWash", data)
    collection("Lights").objects.link(obj)
    obj.location = (-1.4, 4.25, 2.45)
    obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    data.energy = 260.0
    data.shape = "RECTANGLE"
    data.size = 3.5
    data.size_y = 1.0
    data.color = (1.0, 0.84, 0.72)
    changes.append("added_rear_wall_wash_for_great_room")


def add_wall_art_and_decor(furniture, props, changes: list[str]) -> None:
    frame_mat = material("Polish_Wall_Frame", (0.10, 0.07, 0.045, 1.0), 0.44, 0.08)
    art_mats = [
        material("Polish_Art_Clay", (0.58, 0.28, 0.18, 1.0), 0.76),
        material("Polish_Art_Sage", (0.25, 0.40, 0.32, 1.0), 0.76),
        material("Polish_Art_Ochre", (0.70, 0.46, 0.16, 1.0), 0.76),
    ]
    backs = [obj for obj in furniture.objects
             if obj.type == "MESH"
             and "sofa" in obj.name.casefold()
             and "back" in obj.name.casefold()]
    for index, back in enumerate(sorted(backs, key=lambda obj: obj.name)):
        point = local_point(back, 0.0, 0.55, 1.08)
        yaw = float(back.rotation_euler.z)
        cube(
            f"PolishWallArtFrame_{index}",
            (point.x, point.y, 1.92),
            (1.18, 0.055, 0.74),
            frame_mat,
            props,
            bevel=0.018,
            yaw=yaw,
        )
        inner = local_point(back, 0.0, 0.51, 1.08)
        cube(
            f"PolishWallArt_{index}",
            (inner.x, inner.y, 1.92),
            (0.94, 0.025, 0.50),
            art_mats[index % len(art_mats)],
            props,
            bevel=0.008,
            yaw=yaw,
        )
        changes.append(f"added_wall_art_{back.name}")
    # A small, low-contrast object cluster keeps long walls from reading empty.
    shelf_mat = material("Polish_Decor_Shelf", (0.30, 0.16, 0.08, 1.0), 0.56)
    if backs:
        back = sorted(backs, key=lambda obj: obj.name)[0]
        point = local_point(back, -0.65 * back.dimensions.x, 0.48, 1.15)
        cube("PolishWallShelf", (point.x, point.y, 1.38),
             (0.85, 0.20, 0.07), shelf_mat, props, bevel=0.018,
             yaw=float(back.rotation_euler.z))
        changes.append("added_wall_shelf")


def export_selection(output: Path) -> Path:
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"
              and not obj.hide_render]
    for obj in meshes:
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]
    glb = output / "visual" / (output.name + ".glb")
    glb.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.gltf(
        filepath=str(glb), check_existing=False, export_format="GLB",
        use_selection=True, use_visible=False, export_apply=True,
        export_yup=True, export_animations=False, export_cameras=False,
        export_lights=False, export_materials="EXPORT", export_extras=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB export failed: {result}")
    return glb


def load_usd_exporter():
    path = Path(__file__).with_name("semantic_household_builder.py")
    spec = importlib.util.spec_from_file_location("avengine_room_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import shared USD exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def semantic_records(spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (
        spec.get("semantic_objects")
        or spec.get("furnishing_assemblies")
        or spec.get("furniture")
        or []
    )
    records: list[dict[str, Any]] = []
    for index, assembly in enumerate(raw):
        if not isinstance(assembly, dict):
            continue
        kind = assembly.get("kind") or assembly.get("category") or "furniture"
        center = (
            assembly.get("center_xy_m")
            or assembly.get("sofa_center_xy_m")
            or assembly.get("desk_center_xy_m")
            or assembly.get("bench_center_xy_m")
            or assembly.get("island_center_xy_m")
            or assembly.get("run_center_xy_m")
        )
        records.append({
            "object_id": assembly.get("object_id") or f"{kind}_{index}",
            "kind": kind,
            "zone_id": assembly.get("zone_id"),
            "center_xy_m": center,
            "size_xyz_m": assembly.get("size_xyz_m"),
            "static": bool(assembly.get("static", True)),
            "navigation_role": assembly.get("navigation_role"),
            "seat_points": assembly.get("seat_points", []),
        })
    return records


def object_surface_height(obj: bpy.types.Object) -> float:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return max(float(corner.z) for corner in corners)


def seat_backrest(scene: bpy.types.Scene, seat_obj: bpy.types.Object):
    seat_name = seat_obj.name.casefold()
    if "sofa" in seat_name or "cushion" in seat_name:
        tokens = ("sofa", "back")
    elif "bench" in seat_name:
        tokens = ("bench", "back")
    else:
        tokens = ("chair", "back")
    candidates: list[tuple[float, bpy.types.Object]] = []
    seat_point = seat_obj.matrix_world.translation
    for obj in scene.objects:
        name = obj.name.casefold()
        if obj.type != "MESH" or obj.hide_render:
            continue
        if not all(token in name for token in tokens):
            continue
        dx = float(obj.matrix_world.translation.x - seat_point.x)
        dy = float(obj.matrix_world.translation.y - seat_point.y)
        distance = math.hypot(dx, dy)
        if distance <= 0.9:
            candidates.append((distance, obj))
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[0][1] if candidates else None


def nearest_table(scene: bpy.types.Scene, seat_obj: bpy.types.Object):
    if "chair" not in seat_obj.name.casefold():
        return None
    point = seat_obj.matrix_world.translation
    candidates: list[tuple[float, bpy.types.Object]] = []
    for obj in scene.objects:
        name = obj.name.casefold()
        if obj.type != "MESH" or obj.hide_render:
            continue
        if not any(token in name for token in (
            "table", "diningtop", "breakfasttop", "tabletop",
        )):
            continue
        if any(token in name for token in (
            "coffee", "side", "console", "chair", "seat", "back", "leg",
        )):
            continue
        dx = float(obj.matrix_world.translation.x - point.x)
        dy = float(obj.matrix_world.translation.y - point.y)
        distance = math.hypot(dx, dy)
        if distance <= 2.2:
            candidates.append((distance, obj))
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[0][1] if candidates else None


def canonical_anchor_id(obj_name: str, fallback: str) -> str:
    name = obj_name.casefold()
    if "dining" in name and "chair" in name:
        match = re.search(r"(\d+)", name)
        return f"dining_chair_{match.group(1)}_sit" if match else "dining_chair_sit"
    if "sofa" in name and "cushion" in name:
        match = re.search(r"(\d+)", name)
        return f"sofa_seat_{match.group(1)}_sit" if match else "sofa_seat_sit"
    if "study" in name and "chair" in name:
        return "study_chair_sit"
    if "bench" in name:
        return "bench_sit"
    return fallback


def inferred_scene_seats(scene: bpy.types.Scene) -> list[dict[str, Any]]:
    seats: list[dict[str, Any]] = []
    collections = [bpy.data.collections.get(name) for name in ("Furniture", "Props")]
    for coll in collections:
        if coll is None:
            continue
        for obj in sorted(coll.objects, key=lambda value: value.name):
            if obj.type != "MESH" or obj.hide_render:
                continue
            name = obj.name.casefold()
            if "backcushion" in name or "back_pillow" in name:
                continue
            kind = None
            if "cushion" in name and "sofa" in name:
                kind = "living"
            elif "chair" in name and "seat" in name:
                kind = "dining" if "dining" in name else "study"
            elif "bench" in name and (
                "seat" in name or name.endswith("bench_mesh")
            ):
                kind = "entry"
            if kind is None:
                continue
            back = seat_backrest(scene, obj)
            point = obj.matrix_world.translation
            surface_height = object_surface_height(obj)
            record: dict[str, Any] = {
                "anchor_id": canonical_anchor_id(
                    obj.name,
                    f"geometry_{obj.name.casefold().replace(' ', '_')}_seat",
                ),
                "assembly_kind": kind,
                "scene_object_id": obj.name,
                "position_m": [
                    round(float(point.x), 6),
                    round(float(point.y), 6),
                    0.0,
                ],
                "support_height_m": round(surface_height, 6),
                "surface_height_m": round(surface_height, 6),
                "surface_object_id": obj.name,
                "surface_status": "geometry_derived",
                "geometry_source": "seat_surface_and_backrest",
            }
            if back is not None:
                back_point = back.matrix_world.translation
                vx = float(point.x - back_point.x)
                vy = float(point.y - back_point.y)
                norm = math.hypot(vx, vy)
                if norm > 1e-6:
                    record["facing_vector_world"] = [
                        round(vx / norm, 6), round(vy / norm, 6), 0.0
                    ]
                    record["facing_yaw_world_deg"] = round(
                        math.degrees(math.atan2(vy, vx)), 6
                    )
                record["backrest_object_id"] = back.name
                record["backrest_yaw_deg"] = round(
                    math.degrees(float(back.rotation_euler.z)), 6
                )
            table = nearest_table(scene, obj)
            if table is not None:
                table_point = table.matrix_world.translation
                record["table_group_id"] = table.name
                record["table_center_m"] = [
                    round(float(table_point.x), 6),
                    round(float(table_point.y), 6),
                    0.0,
                ]
                record["table_distance_xy_m"] = round(
                    math.hypot(
                        float(point.x - table_point.x),
                        float(point.y - table_point.y),
                    ), 6
                )
            seats.append(record)
    return seats


def assembly_center(assembly: dict[str, Any]):
    for key in (
        "center_xy_m", "sofa_center_xy_m", "desk_center_xy_m",
        "bench_center_xy_m", "island_center_xy_m", "run_center_xy_m",
    ):
        center = assembly.get(key)
        if center:
            return center
    return None


def geometry_seat_semantics(semantics: list[dict[str, Any]],
                            scene: bpy.types.Scene) -> list[dict[str, Any]]:
    actual = inferred_scene_seats(scene)
    if not actual:
        return semantics
    assignments: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(semantics))
    }
    for seat in actual:
        candidates = [
            index for index, assembly in enumerate(semantics)
            if str(assembly.get("kind", "")).casefold()
            == str(seat.get("assembly_kind", "")).casefold()
        ]
        if not candidates:
            candidates = list(range(len(semantics)))
        px, py, _ = seat["position_m"]
        def distance(index: int) -> float:
            center = assembly_center(semantics[index])
            if not center:
                return 1e9
            return math.hypot(float(center[0]) - px, float(center[1]) - py)
        assignments[min(candidates, key=distance)].append(seat)
    used_anchor_ids: set[str] = set()
    for index, assembly in enumerate(semantics):
        physical = sorted(
            assignments[index],
            key=lambda seat: (seat["position_m"][0], seat["position_m"][1],
                              seat["scene_object_id"]),
        )
        templates = list(assembly.get("seat_points", []))
        template_for_physical: dict[int, dict[str, Any]] = {}
        used_templates: set[int] = set()
        used_physical: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for template_index, template in enumerate(templates):
            position = template.get("position_m")
            if not position:
                continue
            for physical_index, seat in enumerate(physical):
                px, py, _ = seat["position_m"]
                distance = math.hypot(
                    float(position[0]) - px, float(position[1]) - py
                )
                pairs.append((distance, template_index, physical_index))
        for _, template_index, physical_index in sorted(pairs):
            if template_index in used_templates or physical_index in used_physical:
                continue
            used_templates.add(template_index)
            used_physical.add(physical_index)
            template_for_physical[physical_index] = templates[template_index]
        updated: list[dict[str, Any]] = []
        for seat_index, seat in enumerate(physical):
            record = dict(seat)
            template = template_for_physical.get(seat_index)
            if template is not None:
                record["source_anchor_id"] = template.get("anchor_id")
                record["anchor_id"] = template.get(
                    "anchor_id", record["anchor_id"]
                )
            if record["anchor_id"] in used_anchor_ids:
                record["anchor_id"] = (
                    f"geometry_{record['scene_object_id'].casefold().replace(' ', '_')}_seat"
                )
            used_anchor_ids.add(record["anchor_id"])
            record["assembly_id"] = assembly.get("object_id")
            record["assembly_kind"] = assembly.get("kind")
            record["zone_id"] = assembly.get("zone_id")
            updated.append(record)
        assembly["seat_points"] = updated
        assembly["geometry_seat_count"] = len(updated)
        assembly["source_seat_count"] = len(templates)
    return semantics

def classify_static_mesh(obj: bpy.types.Object, collection_name: str) -> str:
    name = obj.name.casefold()
    if "doorframe" in name or name.startswith("door_") or "door_" in name:
        return "doorframe"
    if "windowframe" in name:
        return "window_frame"
    if "glass" in name or collection_name == "Glazing":
        return "glazing"
    if "ceilingfixture" in name or "lightfixture" in name:
        return "light_fixture"
    if "ceiling" in name:
        return "ceiling"
    if "floor" in name or "ground" in name:
        return "floor"
    if "wall" in name or name.startswith("wall_"):
        return "wall"
    if any(token in name for token in (
        "counter", "kitchenlower", "kitchenisland", "islandpanel",
        "islandhandle", "kitchenpanel", "kitchenpull", "cabinet",
    )):
        return "cabinet"
    if any(token in name for token in (
        "fridge", "oven", "microwave", "cooktop", "sink", "faucet",
        "kettle", "trash", "hood",
    )):
        return "appliance"
    if collection_name == "Furniture":
        return "furniture"
    if collection_name == "Props":
        return "prop"
    if collection_name == "ArchitecturalDetails":
        return "architectural_detail"
    return "static_geometry"


def world_bounds_record(obj: bpy.types.Object, scene: bpy.types.Scene) -> dict[str, Any]:
    try:
        evaluated = obj.evaluated_get(scene.evaluated_depsgraph_get())
        corners = [evaluated.matrix_world @ Vector(corner)
                   for corner in evaluated.bound_box]
    except (AttributeError, RuntimeError):
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lower = [min(float(corner[i]) for corner in corners) for i in range(3)]
    upper = [max(float(corner[i]) for corner in corners) for i in range(3)]
    return {
        "min_m": [round(value, 6) for value in lower],
        "max_m": [round(value, 6) for value in upper],
        "size_xyz_m": [round(upper[i] - lower[i], 6) for i in range(3)],
    }


def scene_static_mesh_records(scene: bpy.types.Scene, visual_glb_ref: str,
                              usd_ref: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    ordered_collections = (
        "Architecture", "ArchitecturalDetails", "Glazing", "Furniture", "Props",
    )
    for collection_name in ordered_collections:
        coll = bpy.data.collections.get(collection_name)
        if coll is None:
            continue
        for obj in sorted(coll.objects, key=lambda value: value.name):
            if obj.type != "MESH" or obj.as_pointer() in seen:
                continue
            seen.add(obj.as_pointer())
            memberships = sorted(
                owner.name for owner in obj.users_collection
                if owner.name in ordered_collections
            )
            bounds = world_bounds_record(obj, scene)
            category = classify_static_mesh(obj, collection_name)
            records.append({
                "object_id": obj.name,
                "mesh_data_name": obj.data.name,
                "collection": collection_name,
                "collection_memberships": memberships,
                "category": category,
                "kind": obj.name.rsplit("_Mesh", 1)[0],
                "world_bounds_m": bounds,
                "location_m": [round(float(v), 6) for v in obj.matrix_world.translation],
                "size_xyz_m": bounds["size_xyz_m"],
                "materials": [mat.name for mat in obj.data.materials if mat is not None],
                "vertex_count": len(obj.data.vertices),
                "polygon_count": len(obj.data.polygons),
                "static": True,
                "visible_render": not bool(obj.hide_render),
                "geometry_ref": {
                    "blend_object": obj.name,
                    "mesh_data": obj.data.name,
                    "visual_glb": visual_glb_ref,
                    "usd_prim_hint": "/" + obj.name.replace(" ", "_"),
                    "usd_file": usd_ref,
                },
            })
    return records


def scene_lighting_records(scene: bpy.types.Scene) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for obj in sorted((value for value in scene.objects if value.type == "LIGHT"),
                      key=lambda value: value.name):
        direction = obj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        records.append({
            "light_id": obj.name,
            "type": obj.data.type,
            "position_m": [round(float(v), 6) for v in obj.matrix_world.translation],
            "direction_world": [round(float(v), 6) for v in direction],
            "rotation_euler_rad": [round(float(v), 6) for v in obj.rotation_euler],
            "energy": round(float(obj.data.energy), 6),
            "color_rgb": [round(float(v), 6) for v in obj.data.color[:3]],
            "size_m": round(float(getattr(obj.data, "size", 0.0)), 6),
            "static": True,
        })
    return records

def seat_surface_record(seat: dict[str, Any], scene_objects: list[dict[str, Any]],
                        scene: bpy.types.Scene) -> dict[str, Any]:
    position = seat.get("position_m") or [None, None, None]
    px, py = float(position[0]), float(position[1])
    candidates: list[tuple[float, bpy.types.Object]] = []
    for collection_name in ("Furniture", "Props"):
        coll = bpy.data.collections.get(collection_name)
        if coll is None:
            continue
        for obj in coll.objects:
            name = obj.name.casefold()
            if obj.type != "MESH" or obj.hide_render:
                continue
            if not any(token in name for token in ("seat", "cushion", "bench", "sofa")):
                continue
            dx = float(obj.matrix_world.translation.x) - px
            dy = float(obj.matrix_world.translation.y) - py
            distance = math.hypot(dx, dy)
            if distance <= 0.9:
                candidates.append((distance, obj))
    candidates.sort(key=lambda item: (item[0], item[1].name))
    record = dict(seat)
    record["position_m"] = list(position)
    if not candidates:
        record.update({
            "surface_object_id": None,
            "surface_height_m": None,
            "surface_status": "unmatched",
        })
        return record
    distance, obj = candidates[0]
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    surface_height = max(float(corner.z) for corner in corners)
    expected_height = seat.get("support_height_m")
    record.update({
        "surface_object_id": obj.name,
        "surface_height_m": round(surface_height, 6),
        "surface_distance_xy_m": round(distance, 6),
        "surface_delta_m": (
            round(surface_height - float(expected_height), 6)
            if expected_height is not None else None
        ),
        "surface_status": "matched",
    })
    return record


def complete_seat_points(semantics: list[dict[str, Any]],
                        scene_objects: list[dict[str, Any]],
                        scene: bpy.types.Scene) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for assembly in semantics:
        for seat in assembly.get("seat_points", []):
            record = dict(seat)
            record["assembly_id"] = assembly.get("object_id")
            record["assembly_kind"] = assembly.get("kind")
            record["zone_id"] = assembly.get("zone_id")
            points.append(record)
    return points


def main() -> int:
    args = cli()
    blend = args.blend.expanduser().resolve(strict=True)
    spec_path = args.spec.expanduser().resolve(strict=True) if args.spec else None
    spec = json.loads(spec_path.read_text()) if spec_path else {}
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to replace existing polish output: " + str(output))
    (output / "renders").mkdir(parents=True)
    (output / "visual").mkdir()
    (output / "usd").mkdir()
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    furniture = collection("Furniture")
    props = collection("Props")
    changes: list[str] = []
    changes.extend(soften_materials())
    polish_sofas(furniture, props, changes)
    polish_plants(furniture, props, changes)
    add_wall_art_and_decor(furniture, props, changes)

    scene = bpy.context.scene
    repair_compact_dining_chairs(scene, changes)
    fix_lighting(scene, changes)
    for camera in sorted((obj for obj in scene.objects if obj.type == "CAMERA"), key=lambda obj: obj.name):
        scene.camera = camera
        scene.render.resolution_x = max(960, int(scene.render.resolution_x or 1280))
        scene.render.resolution_y = max(540, int(scene.render.resolution_y or 720))
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(output / "renders" / f"{camera.name}.png")
        bpy.ops.render.render(write_still=True)
    bpy.ops.file.pack_all()
    base_name = (args.room_id or scene.name).replace(" ", "_")
    out_blend = output / (base_name + ".blend")
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    glb = export_selection(output)
    usd_module = load_usd_exporter()
    usd = output / "usd" / (base_name + ".usda")
    usd_record = usd_module.export_static_usd(scene, usd)
    anchors = spec.get("anchors", {}) if isinstance(spec, dict) else {}
    semantics = semantic_records(spec) if isinstance(spec, dict) else []
    semantics = geometry_seat_semantics(semantics, scene)
    scene_objects = scene_static_mesh_records(
        scene,
        f"visual/{output.name}.glb",
        f"usd/{usd.name}",
    )
    seat_points = complete_seat_points(semantics, scene_objects, scene)
    lighting_records = scene_lighting_records(scene)
    source_reference_status = (
        "reference-informed_only"
        if isinstance(spec, dict) and str(spec.get("room_spec_id", "")).startswith("aea_loc3")
        else "authored_static_room"
    )
    write_json(output / "object_semantics.json", {
        "kind": "avengine_complete_furniture_semantics",
        "room_id": base_name,
        "room_spec_id": spec.get("room_spec_id") if isinstance(spec, dict) else None,
        "source_blend": str(blend),
        "source_reference_status": source_reference_status,
        "static_scene": True,
        "geometry_scope": "all_static_scene_meshes",
        "furniture_assemblies": semantics,
        "objects": scene_objects,
        "furniture_objects": [
            record for record in scene_objects
            if record["category"] in {"furniture", "prop", "cabinet", "appliance"}
        ],
        "seat_points": seat_points,
    })
    write_json(output / "functional_anchors.json", {
        "kind": "avengine_complete_functional_anchors",
        "room_id": base_name,
        "room_spec_id": spec.get("room_spec_id") if isinstance(spec, dict) else None,
        "source_blend": str(blend),
        "source_reference_status": source_reference_status,
        "anchors": anchors,
        "seat_points": seat_points,
        "scene_object_ids": [record["object_id"] for record in scene_objects],
        "scene_object_categories": sorted({
            record["category"] for record in scene_objects
        }),
        "wall_segment_object_ids": [
            record["object_id"] for record in scene_objects
            if record["category"] == "wall"
        ],
    })
    write_json(output / "lighting.json", {
        "kind": "avengine_scene_lighting_manifest",
        "room_id": base_name,
        "source_blend": str(blend),
        "lights": lighting_records,
        "world_color_rgb": [
            round(float(value), 6) for value in scene.world.color[:3]
        ] if scene.world else None,
    })
    write_json(output / "polish_report.json", {
        "kind": "avengine_polished_room_report",
        "status": "research_candidate",
        "qualification_claim": False,
        "source_blend": str(blend),
        "changes": changes,
        "anchors": anchors,
        "furniture_semantics": semantics,
        "scene_object_count": len(scene_objects),
        "scene_object_categories": sorted({
            record["category"] for record in scene_objects
        }),
        "wall_segment_count": sum(
            record["category"] == "wall" for record in scene_objects
        ),
        "seat_point_count": len(seat_points),
        "lighting_count": len(lighting_records),
        "source_reference_status": source_reference_status,
        "artifacts": {
            "blend": str(out_blend),
            "visual_glb": str(glb),
            "usd": usd_record,
            "renders": str(output / "renders"),
        },
        "native_execution": "pending_root_spear_ue",
    })
    print(json.dumps({"status":"research_candidate","changes":len(changes),"blend":str(out_blend),"glb":str(glb),"usd":str(usd)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
