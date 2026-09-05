#!/usr/bin/env python3
"""Build Room B: a compact separated home with static affordance semantics.

This builder is intentionally self-contained. It emits Blender/GLB geometry and
ordinary semantic records for a later AVEngine SPEAR/UE handoff. It does not
author actors, routes, sounds, doors, drawers, grasp, or release behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import bmesh
import bpy
from mathutils import Vector


def cli() -> argparse.Namespace:
    raw = os.sys.argv
    argv = raw[raw.index("--") + 1 :] if "--" in raw else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mat(name: str, color: tuple[float, float, float], roughness: float = 0.55,
        metallic: float = 0.0, transmission: float = 0.0,
        emission: tuple[float, float, float] | None = None):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    value.diffuse_color = (*color, 1.0)
    bsdf = value.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    if emission is not None:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 1.8
    return value


def cube(name: str, center: tuple[float, float, float],
         size: tuple[float, float, float], material, collection,
         *, bevel: float = 0.015, parent=None):
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
    collection.objects.link(obj)
    obj.location = center
    obj.data.materials.append(material)
    if parent is not None:
        obj.parent = parent
    if bevel:
        modifier = obj.modifiers.new("SoftEdges", "BEVEL")
        modifier.width = min(bevel, min(size) * 0.22)
        modifier.segments = 3
        modifier.limit_method = "ANGLE"
    return obj


def cyl(name: str, center: tuple[float, float, float], radius: float,
        depth: float, material, collection, *, parent=None, vertices: int = 32):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    builder = bmesh.new()
    bmesh.ops.create_cone(builder, cap_ends=True, cap_tris=False,
                          segments=vertices, radius1=radius, radius2=radius,
                          depth=depth)
    builder.to_mesh(mesh)
    builder.free()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = center
    obj.data.materials.append(material)
    if parent is not None:
        obj.parent = parent
    return obj


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def register_object(records: list[dict[str, Any]], object_id: str, category: str,
                    zone_id: str, center_xy: tuple[float, float],
                    size_xyz: tuple[float, float, float], *,
                    navigation_role: str = "ground_blocker",
                    static: bool = True, seat_points=None, source: str = "builder"):
    records.append({
        "object_id": object_id,
        "category": category,
        "zone_id": zone_id,
        "bounds_xy_m": [
            center_xy[0] - size_xyz[0] / 2.0,
            center_xy[1] - size_xyz[1] / 2.0,
            center_xy[0] + size_xyz[0] / 2.0,
            center_xy[1] + size_xyz[1] / 2.0,
        ],
        "height_m": size_xyz[2],
        "navigation_role": navigation_role,
        "static": bool(static),
        "seat_points": list(seat_points or []),
        "source": source,
        "interaction": "none",
    })


def wall_piece(axis: str, coordinate: float, start: float, end: float,
               thickness: float, z0: float, z1: float, material, collection,
               name: str):
    if end <= start or z1 <= z0:
        return
    if axis == "h":
        center = ((start + end) / 2.0, coordinate, (z0 + z1) / 2.0)
        size = (end - start, thickness, z1 - z0)
    else:
        center = (coordinate, (start + end) / 2.0, (z0 + z1) / 2.0)
        size = (thickness, end - start, z1 - z0)
    cube(name, center, size, material, collection, bevel=0.01)


def wall_with_openings(axis: str, coordinate: float, start: float, end: float,
                       thickness: float, height: float, openings: list[Mapping[str, Any]],
                       material, collection, name: str):
    cursor = start
    for index, opening in enumerate(sorted(openings, key=lambda row: float(row["center"]))):
        center = float(opening["center"])
        width = float(opening["width"])
        lo, hi = center - width / 2.0, center + width / 2.0
        wall_piece(axis, coordinate, cursor, lo, thickness, 0.0, height,
                   material, collection, f"{name}_pier_{index}")
        sill, head = float(opening.get("sill", 0.0)), float(opening["head"])
        if sill > 0.0:
            wall_piece(axis, coordinate, lo, hi, thickness, 0.0, sill,
                       material, collection, f"{name}_sill_{index}")
        wall_piece(axis, coordinate, lo, hi, thickness, head, height,
                   material, collection, f"{name}_head_{index}")
        cursor = hi
    wall_piece(axis, coordinate, cursor, end, thickness, 0.0, height,
               material, collection, f"{name}_end")


def build_architecture(spec, collections, materials, semantics):
    envelope = spec["envelope"]
    x0, y0, x1, y1 = envelope["bounds_xy_m"]
    height = float(envelope["wall_height_m"])
    ext_t = float(envelope["exterior_wall_thickness_m"])
    int_t = float(envelope["interior_wall_thickness_m"])
    zones = {row["zone_id"]: row for row in spec["zones"]}
    arch, glaze = collections["Architecture"], collections["Glazing"]
    floor_mats = {"oak": materials["floor"], "terrazzo": materials["terrazzo"]}

    for zone_id, zone in zones.items():
        a, b, c, d = zone["bounds_xy_m"]
        cube(f"Floor_{zone_id}", ((a + c) / 2.0, (b + d) / 2.0, -0.06),
             (c - a, d - b, 0.12), floor_mats[zone["floor_material"]], arch,
             bevel=0.004)
        register_object(semantics, f"floor_{zone_id}", "floor", zone_id,
                        ((a + c) / 2.0, (b + d) / 2.0), (c - a, d - b, 0.12),
                        navigation_role="walkable_surface")

    openings = spec["exterior_openings"]
    by_side = {side: [row for row in openings if row["side"] == side]
               for side in ("N", "S", "E", "W")}
    wall_with_openings("h", y0, x0, x1, ext_t, height,
                       [{"center": row["center_xy_m"][0], "width": row["width_m"],
                         "sill": row["sill_m"], "head": row["head_m"]}
                        for row in by_side["S"]], materials["wall"], arch, "Wall_S")
    wall_with_openings("h", y1, x0, x1, ext_t, height,
                       [{"center": row["center_xy_m"][0], "width": row["width_m"],
                         "sill": row["sill_m"], "head": row["head_m"]}
                        for row in by_side["N"]], materials["wall"], arch, "Wall_N")
    wall_with_openings("v", x0, y0, y1, ext_t, height,
                       [{"center": row["center_xy_m"][1], "width": row["width_m"],
                         "sill": row["sill_m"], "head": row["head_m"]}
                        for row in by_side["W"]], materials["wall"], arch, "Wall_W")
    wall_with_openings("v", x1, y0, y1, ext_t, height,
                       [{"center": row["center_xy_m"][1], "width": row["width_m"],
                         "sill": row["sill_m"], "head": row["head_m"]}
                        for row in by_side["E"]], materials["wall"], arch, "Wall_E")

    horizontal = []
    vertical = []
    for link in spec["links"]:
        center = link["center_xy_m"]
        opening = {"center": center[0], "width": link["width_m"],
                   "sill": 0.0, "head": link["head_m"]}
        if abs(center[1] - 0.22) < 0.04:
            horizontal.append(opening)
        elif abs(center[0] + 0.25) < 0.04:
            vertical.append({"center": center[1], "width": link["width_m"],
                             "sill": 0.0, "head": link["head_m"]})
    wall_with_openings("h", 0.22, x0 + 0.25, x1 - 0.25, int_t, height,
                       horizontal, materials["wall"], arch, "Wall_Internal_H")
    wall_with_openings("v", -0.25, y0 + 0.25, y1 - 0.25, int_t, height,
                       vertical, materials["wall"], arch, "Wall_Internal_V")

    cube("Ceiling_Slab", ((x0 + x1) / 2.0, (y0 + y1) / 2.0, height + 0.05),
         (x1 - x0, y1 - y0, 0.10), materials["ceiling"], arch, bevel=0.004)
    cube("ExteriorGround", ((x0 + x1) / 2.0, (y0 + y1) / 2.0, -0.19),
         (x1 - x0 + 5.0, y1 - y0 + 5.0, 0.10), materials["ground"], arch,
         bevel=0.004)

    for row in openings:
        cx, cy = row["center_xy_m"]
        sill, head, width = float(row["sill_m"]), float(row["head_m"]), float(row["width_m"])
        side = row["side"]
        if side in ("N", "S"):
            glass_center = (cx, cy, (sill + head) / 2.0)
            glass_size = (width - 0.10, 0.035, head - sill - 0.10)
            frame_size = ((width, 0.05, 0.06), (width, 0.05, 0.06))
            frame_axis = "x"
        else:
            glass_center = (cx, cy, (sill + head) / 2.0)
            glass_size = (0.035, width - 0.10, head - sill - 0.10)
            frame_size = ((0.05, width, 0.06), (0.05, width, 0.06))
            frame_axis = "y"
        cube("Glass_" + row["opening_id"], glass_center, glass_size,
             materials["glass"], glaze, bevel=0.003)
        if frame_axis == "x":
            for z in (sill + 0.05, head - 0.05):
                cube(f"WindowFrame_{row['opening_id']}_{z:.2f}",
                     (cx, cy, z), (width, 0.06, 0.06), materials["trim"], glaze,
                     bevel=0.006)
        else:
            for z in (sill + 0.05, head - 0.05):
                cube(f"WindowFrame_{row['opening_id']}_{z:.2f}",
                     (cx, cy, z), (0.06, width, 0.06), materials["trim"], glaze,
                     bevel=0.006)


def chair_parts(root, x, y, yaw, collections, materials):
    furniture = collections["Furniture"]
    seat = cube(root + "_seat", (x, y, 0.47), (0.55, 0.52, 0.10),
                materials["fabric"], furniture, bevel=0.045)
    back = cube(root + "_back", (x, y + 0.20 * math.cos(yaw), 0.86),
                (0.55, 0.10, 0.70), materials["fabric"], furniture, bevel=0.045)
    seat.rotation_euler.z = yaw
    back.rotation_euler.z = yaw
    for index, dx in enumerate((-0.21, 0.21)):
        for j, dy in enumerate((-0.18, 0.18)):
            leg = cube(f"{root}_leg_{index}_{j}", (x + dx, y + dy, 0.23),
                       (0.055, 0.055, 0.46), materials["dark_metal"],
                       furniture, bevel=0.012)
            leg.rotation_euler.z = yaw


def build_furniture(spec, collections, materials, semantics):
    furniture, props = collections["Furniture"], collections["Props"]
    for item in spec["furniture"]:
        object_id = item["object_id"]
        x, y = item["center_xy_m"]
        sx, sy, sz = item["size_xyz_m"]
        category, zone = item["category"], item["zone_id"]
        root = bpy.data.objects.new("SEM_" + object_id, None)
        furniture.objects.link(root)
        # Child meshes below are authored in room/world coordinates; keep the
        # semantic root at identity to avoid applying the furniture center twice.
        root.location = (0.0, 0.0, 0.0)
        root["semantic_id"] = object_id
        root["category"] = category
        root["zone_id"] = zone
        if category == "sofa":
            cube(object_id + "_base", (x, y, 0.26), (sx, sy, 0.42),
                 materials["sofa"], furniture, bevel=0.10, parent=root)
            cube(object_id + "_back", (x, y + 0.38, 0.78), (sx, 0.24, 1.0),
                 materials["sofa"], furniture, bevel=0.10, parent=root)
            for idx, px in enumerate((x - 0.82, x, x + 0.82)):
                cube(f"{object_id}_cushion_{idx}", (px, y - 0.12, 0.52),
                     (0.70, 0.66, 0.13), materials["fabric"], furniture,
                     bevel=0.06, parent=root)
        elif category == "table":
            top_z = sz
            cube(object_id + "_top", (x, y, top_z), (sx, sy, 0.10),
                 materials["wood"], furniture, bevel=0.06, parent=root)
            for i, dx in enumerate((-sx * 0.40, sx * 0.40)):
                for j, dy in enumerate((-sy * 0.36, sy * 0.36)):
                    cube(f"{object_id}_leg_{i}_{j}", (x + dx, y + dy, 0.38),
                         (0.10, 0.10, 0.76), materials["dark_metal"],
                         furniture, bevel=0.015, parent=root)
        elif category == "counter":
            cube(object_id + "_body", (x, y, sz / 2.0), (sx, sy, sz),
                 materials["cabinet"], furniture, bevel=0.04, parent=root)
            cube(object_id + "_top", (x, y - 0.01, sz + 0.04),
                 (sx + 0.05, sy + 0.04, 0.08), materials["stone"],
                 furniture, bevel=0.02, parent=root)
        elif category == "bench":
            cube(object_id + "_seat", (x, y, 0.49), (sx, sy, 0.12),
                 materials["wood"], furniture, bevel=0.04, parent=root)
            cube(object_id + "_back", (x, y + 0.25, 0.82),
                 (sx, 0.08, 0.68), materials["wood"], furniture,
                 bevel=0.03, parent=root)
            for dx in (-sx * 0.32, sx * 0.32):
                cube(object_id + "_leg_" + str(dx), (x + dx, y, 0.24),
                     (0.08, sy * 0.65, 0.48), materials["dark_metal"],
                     furniture, bevel=0.012, parent=root)
        else:
            cube(object_id + "_body", (x, y, sz / 2.0),
                 (sx, sy, sz), materials["wood"], furniture, bevel=0.03, parent=root)

        register_object(semantics, object_id, category, zone, (x, y),
                        (sx, sy, sz),
                        navigation_role=item.get("navigation_role", "ground_blocker"),
                        static=True, seat_points=item.get("seat_points", []))
        for index, seat in enumerate(item.get("seat_points", [])):
            if category == "table":
                px, py, _ = seat["position_m"]
                chair_parts(f"{object_id}_chair_{index}", px, py,
                            math.radians(float(seat["facing_yaw_deg"])),
                            collections, materials)
                register_object(
                    semantics, f"{object_id}_chair_{index}", "chair", zone,
                    (px, py), (0.58, 0.56, 0.92),
                    navigation_role="ground_blocker", static=True,
                    seat_points=[seat],
                )

    def prop_box(object_id, category, zone, center, size, material):
        obj = cube(object_id, center, size, material, props, bevel=0.025)
        register_object(semantics, object_id, category, zone,
                        (center[0], center[1]), size,
                        navigation_role="elevated_object", static=True)
        return obj

    prop_box("living_media_shelf", "media_shelf", "living_room",
             (-1.50, 0.08, 0.55), (2.4, 0.38, 0.80), materials["cabinet"])
    prop_box("living_rug", "rug", "living_room",
             (-3.00, -1.30, 0.025), (3.2, 2.0, 0.035), materials["rug"])
    prop_box("dining_sideboard", "sideboard", "dining_room",
             (4.85, -0.85, 0.60), (0.42, 1.65, 0.95), materials["cabinet"])
    prop_box("hall_mirror_console", "console", "corner_hall",
             (-4.85, 2.90, 0.48), (0.45, 1.55, 0.78), materials["wood"])
    prop_box("kitchen_sink_basin", "sink", "kitchen",
             (4.10, 3.03, 1.03), (0.82, 0.48, 0.10), materials["dark_metal"])
    prop_box("kitchen_fridge", "appliance_static", "kitchen",
             (0.15, 3.35, 1.38), (0.78, 0.74, 2.70), materials["steel"])
    cyl("kitchen_faucet", (4.35, 3.03, 1.36), 0.025, 0.64,
        materials["steel"], props)
    prop_box("living_plant_pot", "plant", "living_room",
             (-0.95, -0.15, 0.30), (0.44, 0.44, 0.58), materials["ceramic"])
    for i, offset in enumerate((-0.18, 0.0, 0.18)):
        leaf = prop_box(f"living_plant_leaf_{i}", "plant_leaf", "living_room",
                        (-0.95 + offset, -0.15, 0.90 + 0.10 * i),
                        (0.12, 0.06, 0.85), materials["leaf"])
        leaf.rotation_euler.y = math.radians(-18 + 18 * i)
    for i, (px, py) in enumerate(((2.10, -1.75), (3.25, -2.25), (2.75, -1.45))):
        cyl(f"dining_prop_{i}", (px, py, 0.88), 0.08, 0.10,
            materials["ceramic"], props)
        register_object(semantics, f"dining_prop_{i}", "tabletop_prop",
                        "dining_room", (px, py), (0.16, 0.16, 0.10),
                        navigation_role="elevated_object", static=True)


def add_anchors(spec, collections, semantics):
    anchors_collection = collections["Anchors"]
    for anchor_id, position in spec["anchors"].items():
        empty = bpy.data.objects.new("ANCHOR_" + anchor_id, None)
        anchors_collection.objects.link(empty)
        empty.location = tuple(position)
        empty["anchor_id"] = anchor_id
        empty["kind"] = "static_affordance_anchor"
        semantics.append({
            "object_id": anchor_id,
            "category": "anchor",
            "zone_id": None,
            "position_m": [float(value) for value in position],
            "navigation_role": "semantic_point",
            "static": True,
            "seat_points": [],
            "source": "room_spec"
        })


def add_camera(name, location, target, collection):
    data = bpy.data.cameras.new(name + "_Data")
    data.lens = 30.0
    data.sensor_width = 36.0
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.location = location
    aim(obj, target)
    return obj


def add_lights(scene, collections, materials):
    lights = collections["Lights"]
    for name, location, target, energy, size, color in (
        ("LivingArea", (-2.7, -1.8, 2.65), (-2.4, -2.0, 0.0), 780.0, 3.2, (1.0, 0.80, 0.66)),
        ("DiningArea", (2.7, -2.0, 2.65), (2.7, -2.0, 0.0), 820.0, 3.0, (1.0, 0.86, 0.70)),
        ("KitchenArea", (2.6, 2.8, 2.65), (2.6, 2.0, 0.0), 850.0, 3.0, (0.90, 0.96, 1.0)),
        ("HallArea", (-4.2, 2.25, 2.60), (-3.8, 1.9, 0.0), 600.0, 2.4, (0.85, 0.90, 1.0)),
    ):
        data = bpy.data.lights.new(name + "_Data", "AREA")
        data.energy, data.shape, data.size, data.color = energy, "DISK", size, color
        obj = bpy.data.objects.new(name, data)
        lights.objects.link(obj)
        obj.location = location
        aim(obj, target)
    sun_data = bpy.data.lights.new("DaySun_Data", "SUN")
    sun_data.energy = 1.1
    sun_data.color = (0.75, 0.85, 1.0)
    sun = bpy.data.objects.new("DaySun", sun_data)
    lights.objects.link(sun)
    sun.rotation_euler = (math.radians(32.0), math.radians(-18.0), math.radians(-35.0))
    scene.world.color = (0.045, 0.055, 0.075)


def main():
    args = cli()
    spec = load_json(args.spec)
    output = args.output_root.expanduser().resolve()
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"output root must be fresh and empty: {output}")
    else:
        output.mkdir(parents=True)
    for subdir in ("preview_stills", "ue_import"):
        (output / subdir).mkdir(parents=True, exist_ok=True)

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    scene.name = "RoomB_CompactSeparatedHome"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x, scene.render.resolution_y = 960, 540
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = 15
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    collections = {}
    for name in ("Architecture", "Glazing", "Furniture", "Props", "Anchors", "Lights", "Cameras"):
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)
        collections[name] = collection
    materials = {
        "wall": mat("RoomB_WarmPlaster", (0.66, 0.60, 0.53), 0.78),
        "ceiling": mat("RoomB_SoftCeiling", (0.82, 0.80, 0.76), 0.82),
        "floor": mat("RoomB_OakFloor", (0.36, 0.21, 0.11), 0.46),
        "terrazzo": mat("RoomB_Terrazzo", (0.42, 0.44, 0.43), 0.62),
        "ground": mat("RoomB_ExteriorPaving", (0.18, 0.20, 0.22), 0.78),
        "glass": mat("RoomB_WindowGlass", (0.52, 0.70, 0.82), 0.10, transmission=0.55),
        "trim": mat("RoomB_WindowTrim", (0.18, 0.13, 0.09), 0.45),
        "wood": mat("RoomB_Walnut", (0.30, 0.15, 0.07), 0.42),
        "cabinet": mat("RoomB_SageCabinet", (0.19, 0.29, 0.24), 0.50),
        "stone": mat("RoomB_HonedStone", (0.58, 0.56, 0.51), 0.34),
        "steel": mat("RoomB_BrushedSteel", (0.38, 0.41, 0.44), 0.28, metallic=0.82),
        "dark_metal": mat("RoomB_DarkMetal", (0.05, 0.06, 0.07), 0.36, metallic=0.45),
        "fabric": mat("RoomB_WarmFabric", (0.46, 0.25, 0.18), 0.82),
        "sofa": mat("RoomB_SofaTeal", (0.08, 0.22, 0.24), 0.88),
        "rug": mat("RoomB_Rug", (0.28, 0.18, 0.13), 0.92),
        "ceramic": mat("RoomB_Ceramic", (0.82, 0.77, 0.66), 0.20),
        "leaf": mat("RoomB_Leaf", (0.12, 0.30, 0.18), 0.76),
    }
    semantics: list[dict[str, Any]] = []
    build_architecture(spec, collections, materials, semantics)
    build_furniture(spec, collections, materials, semantics)
    add_anchors(spec, collections, semantics)

    camera_definitions = (
        ("CAM_LIVING", (-4.75, -3.45, 1.55), (-2.7, -2.0, 1.0)),
        ("CAM_DINING", (4.90, -3.35, 1.55), (2.7, -1.8, 1.0)),
        ("CAM_KITCHEN", (4.75, 2.90, 1.55), (2.5, 2.0, 1.0)),
        ("CAM_HALL", (-4.55, 2.75, 1.55), (-3.6, 1.0, 1.0)),
        ("CAM_CORNER", (-1.30, 0.75, 1.55), (0.7, -1.5, 1.0)),
        ("CAM_REVERSE", (4.85, 3.20, 1.55), (2.1, 1.65, 1.0)),
    )
    cameras = [add_camera(name, location, target, collections["Cameras"])
               for name, location, target in camera_definitions]
    add_lights(scene, collections, materials)
    scene.camera = cameras[0]

    blend_path = output / "ue_import" / "authored_compact_home_room_b_v1.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    for camera in cameras:
        scene.camera = camera
        scene.render.filepath = str(output / "preview_stills" / (camera.name + ".png"))
        bpy.ops.render.render(write_still=True)

    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    exportables = [obj for obj in bpy.context.scene.objects
                   if obj.type == "MESH" and obj.name not in {"ANCHORS"}]
    for obj in exportables:
        obj.select_set(True)
    if exportables:
        bpy.context.view_layer.objects.active = exportables[0]
    glb_path = output / "ue_import" / "authored_compact_home_room_b_v1.glb"
    result = bpy.ops.export_scene.gltf(
        filepath=str(glb_path), check_existing=False, export_format="GLB",
        use_selection=True, export_apply=True, export_animations=False,
        export_cameras=False, export_lights=False, export_materials="EXPORT",
        export_extras=True, export_yup=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB export failed: {result}")

    functional_anchors = {
        "room_spec_id": spec["room_spec_id"],
        "coordinate_system": "Blender +Z up metres; exported GLB +Y up",
        "anchors": spec["anchors"],
        "seat_points": [
            seat
            for item in spec["furniture"]
            for seat in item.get("seat_points", [])
        ],
    }
    object_semantics = {
        "room_spec_id": spec["room_spec_id"],
        "coordinate_system": "room_local_xy_plus_z",
        "objects": semantics,
    }
    room_manifest = {
        "kind": "avengine_authored_room_b_manifest",
        "status": "research_candidate",
        "qualification_claim": False,
        "room_id": spec["room_spec_id"],
        "room_family_id": spec["room_family_id"],
        "backend_intent": spec["backend_intent"],
        "coordinate_system": spec["coordinate_system"],
        "envelope": spec["envelope"],
        "zones": spec["zones"],
        "links": spec["links"],
        "exterior_openings": spec["exterior_openings"],
        "capabilities": spec["capabilities"],
        "artifacts": {
            "blend": str(blend_path),
            "visual_glb": str(glb_path),
            "functional_anchors": str(output / "functional_anchors.json"),
            "object_semantics": str(output / "object_semantics.json"),
            "preview_stills": str(output / "preview_stills"),
        },
        "native_execution": {
            "status": "pending_root_execution",
            "backend": "spear_unreal",
            "map_or_stage": None,
            "camera": "planned_by_avengine",
            "actors": "planned_by_avengine",
            "audio": "planned_by_avengine",
            "qa": "planned_by_avengine",
        },
    }
    ue_import_manifest = {
        "kind": "avengine_room_b_ue_import_manifest",
        "status": "research_candidate",
        "qualification_claim": False,
        "source_glb": str(glb_path),
        "unit_scale": 1.0,
        "up_axis": "+Y",
        "import_as_static_meshes": True,
        "preserve_object_names": True,
        "preserve_material_names": True,
        "semantic_sidecar": str(output / "object_semantics.json"),
        "static_interaction_boundary": {
            "seat_points": True,
            "grasp_release": False,
            "door_drawer_interaction": False,
        },
    }
    write_json(output / "functional_anchors.json", functional_anchors)
    write_json(output / "object_semantics.json", object_semantics)
    write_json(output / "room_manifest.json", room_manifest)
    write_json(output / "ue_import_manifest.json", ue_import_manifest)
    write_json(output / "build_report.json", {
        "kind": "avengine_room_b_build_report",
        "status": "research_candidate",
        "qualification_claim": False,
        "room_spec_id": spec["room_spec_id"],
        "mesh_object_count": len(exportables),
        "semantic_object_count": len(semantics),
        "preview_cameras": [camera.name for camera in cameras],
        "blend": str(blend_path),
        "visual_glb": str(glb_path),
    })
    print(f"ROOM_B_BUILD_COMPLETE output={output} glb={glb_path}")


if __name__ == "__main__":
    main()
