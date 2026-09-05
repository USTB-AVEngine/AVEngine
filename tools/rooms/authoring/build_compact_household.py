import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


def args_from_cli():
    raw = os.sys.argv
    argv = raw[raw.index("--") + 1 :] if "--" in raw else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def material(name, base, rough=0.5, metallic=0.0, transmission=0.0, emission=None):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    value.diffuse_color = (*base, 1.0)
    bsdf = value.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 2.0
    return value


def box(name, center, dimensions, mat, collection, bevel=0.02):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for vertex in bm.verts:
        vertex.co.x *= dimensions[0]
        vertex.co.y *= dimensions[1]
        vertex.co.z *= dimensions[2]
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = center
    obj.data.materials.append(mat)
    if bevel:
        modifier = obj.modifiers.new("MicroBevel", "BEVEL")
        modifier.width = min(bevel, min(dimensions) * 0.25)
        modifier.segments = 3
        modifier.limit_method = "ANGLE"
    return obj


def cylinder(name, center, radius, depth, mat, collection, vertices=48):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=vertices,
        radius1=radius, radius2=radius, depth=depth,
    )
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = center
    obj.data.materials.append(mat)
    return obj


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def layout_point(spec, point):
    value = Vector(point)
    transform = spec.get("layout_transform", {"kind": "identity"})
    if transform.get("kind") == "identity":
        return value
    if transform.get("kind") == "mirror_x":
        axis_x = float(transform.get("axis_x_m", 0.0))
        value.x = 2.0 * axis_x - value.x
        return value
    raise ValueError(f"unsupported layout transform: {transform}")


def transform_furniture_for_layout(spec, collections):
    transform = spec.get("layout_transform", {"kind": "identity"})
    if transform.get("kind") == "identity":
        return
    if transform.get("kind") != "mirror_x":
        raise ValueError(f"unsupported furniture layout transform: {transform}")
    axis_x = float(transform.get("axis_x_m", 0.0))
    for collection_name in ("Furniture", "Props"):
        for value in collections[collection_name].objects:
            value.location.x = 2.0 * axis_x - value.location.x
            value.rotation_euler.z = -value.rotation_euler.z


def add_area(name, location, target, energy, size, color, collection):
    data = bpy.data.lights.new(name + "_Data", "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.location = location
    aim(obj, target)
    return obj


def derive_segments(zones):
    horizontal = defaultdict(list)
    vertical = defaultdict(list)
    for zone in zones:
        x0, y0, x1, y1 = zone["bounds_xy_m"]
        name = zone["zone_id"]
        horizontal[y0].append((x0, x1, "hi", name))
        horizontal[y1].append((x0, x1, "lo", name))
        vertical[x0].append((y0, y1, "hi", name))
        vertical[x1].append((y0, y1, "lo", name))
    result = []
    for axis, table in (("h", horizontal), ("v", vertical)):
        for coordinate, entries in table.items():
            points = sorted({value for start, end, _side, _zone in entries for value in (start, end)})
            for start, end in zip(points, points[1:]):
                midpoint = (start + end) / 2.0
                lo = [zone for a, b, side, zone in entries if side == "lo" and a < midpoint < b]
                hi = [zone for a, b, side, zone in entries if side == "hi" and a < midpoint < b]
                if lo or hi:
                    result.append({
                        "axis": axis, "coordinate": coordinate,
                        "start": start, "end": end,
                        "lo": lo, "hi": hi, "openings": [],
                    })
    return result


def attach_openings(segments, spec):
    for link in spec["links"]:
        wanted = set(link["zones"])
        cx, cy = link["center_xy_m"]
        for segment in segments:
            zones = set(segment["lo"] + segment["hi"])
            coordinate_match = (
                segment["axis"] == "h" and abs(segment["coordinate"] - cy) < 1.0e-6
                or segment["axis"] == "v" and abs(segment["coordinate"] - cx) < 1.0e-6
            )
            along = cx if segment["axis"] == "h" else cy
            if zones == wanted and coordinate_match and segment["start"] <= along <= segment["end"]:
                segment["openings"].append({
                    "id": link["link_id"], "kind": link["kind"],
                    "center": along, "width": link["width_m"],
                    "sill": 0.0, "head": link["head_m"],
                    "door_leaf": link.get("door_leaf"),
                    "window_detail": None,
                })
                break
        else:
            raise RuntimeError(f"unable to attach internal opening {link['link_id']}")
    for opening in spec["exterior_openings"]:
        cx, cy = opening["center_xy_m"]
        wanted = opening["zone"]
        for segment in segments:
            zones = segment["lo"] + segment["hi"]
            coordinate_match = (
                segment["axis"] == "h" and abs(segment["coordinate"] - cy) < 1.0e-6
                or segment["axis"] == "v" and abs(segment["coordinate"] - cx) < 1.0e-6
            )
            along = cx if segment["axis"] == "h" else cy
            if zones == [wanted] and coordinate_match and segment["start"] <= along <= segment["end"]:
                segment["openings"].append({
                    "id": opening["opening_id"], "kind": opening["kind"],
                    "center": along, "width": opening["width_m"],
                    "sill": opening["sill_m"], "head": opening["head_m"],
                    "door_leaf": opening.get("door_leaf"),
                    "window_detail": opening.get("window_detail"),
                })
                break
        else:
            raise RuntimeError(f"unable to attach exterior opening {opening['opening_id']}")


def emit_architecture(spec, collections, mats):
    architecture = collections["Architecture"]
    glazing = collections["Glazing"]
    height = spec["envelope"]["wall_height_m"]
    zone_by_id = {zone["zone_id"]: zone for zone in spec["zones"]}
    for zone in spec["zones"]:
        x0, y0, x1, y1 = zone["bounds_xy_m"]
        floor_mat = mats["entry_floor"] if zone["floor_material"] == "stone" else mats["wood_floor"]
        box(f"Floor_{zone['zone_id']}", ((x0+x1)/2, (y0+y1)/2, -0.04), (x1-x0, y1-y0, 0.08), floor_mat, architecture, 0.004)
        if zone.get("has_ceiling", True):
            box(f"Ceiling_{zone['zone_id']}", ((x0+x1)/2, (y0+y1)/2, height+0.05), (x1-x0, y1-y0, 0.10), mats["ceiling"], architecture, 0.006)

    segments = derive_segments(spec["zones"])
    attach_openings(segments, spec)
    wall_count = 0
    for index, segment in enumerate(segments):
        exterior = not segment["lo"] or not segment["hi"]
        thickness = spec["envelope"]["exterior_wall_thickness_m"] if exterior else spec["envelope"]["interior_wall_thickness_m"]
        coordinate = segment["coordinate"]
        openings = sorted(segment["openings"], key=lambda item: item["center"])

        def panel(start, end, z0, z1, suffix):
            nonlocal wall_count
            if end - start <= 1.0e-4 or z1 - z0 <= 1.0e-4:
                return
            if segment["axis"] == "h":
                center = ((start+end)/2, coordinate, (z0+z1)/2)
                dimensions = (end-start, thickness, z1-z0)
            else:
                center = (coordinate, (start+end)/2, (z0+z1)/2)
                dimensions = (thickness, end-start, z1-z0)
            box(f"Wall_{index:03d}_{suffix}", center, dimensions, mats["wall"], architecture, 0.012)
            wall_count += 1

        cursor = segment["start"]
        for opening_index, opening in enumerate(openings):
            start = opening["center"] - opening["width"] / 2
            end = opening["center"] + opening["width"] / 2
            panel(cursor, start, 0.0, height, f"pier_{opening_index}")
            if opening["sill"] > 0:
                panel(start, end, 0.0, opening["sill"], f"sill_{opening_index}")
            panel(start, end, opening["head"], height, f"head_{opening_index}")
            if opening["kind"] == "window":
                if segment["axis"] == "h":
                    center = (opening["center"], coordinate, (opening["sill"]+opening["head"])/2)
                    dimensions = (opening["width"]-0.08, 0.024, opening["head"]-opening["sill"]-0.08)
                else:
                    center = (coordinate, opening["center"], (opening["sill"]+opening["head"])/2)
                    dimensions = (0.024, opening["width"]-0.08, opening["head"]-opening["sill"]-0.08)
                box(f"Glass_{opening['id']}", center, dimensions, mats["glass"], glazing, 0.004)
            cursor = end
        panel(cursor, segment["end"], 0.0, height, "pier_end")

    exterior = spec.get("exterior_ground", {"enabled": True})
    if exterior.get("enabled", True):
        x0, y0, x1, y1 = spec["envelope"]["bounds_xy_m"]
        margin = float(exterior.get("margin_m", 2.5))
        box(
            "ExteriorGround",
            ((x0 + x1) / 2, (y0 + y1) / 2, -0.13),
            (x1 - x0 + 2 * margin, y1 - y0 + 2 * margin, 0.10),
            mats["ground"], architecture, 0.005,
        )
    return segments, wall_count


def add_chair(name, x, y, rotation, collections, mats):
    furniture = collections["Furniture"]
    box(name+"_Seat", (x, y, 0.48), (0.52, 0.50, 0.10), mats["fabric"], furniture, 0.045).rotation_euler.z = rotation
    box(name+"_Back", (x, y+0.21*math.cos(rotation), 0.87), (0.52, 0.10, 0.72), mats["fabric"], furniture, 0.045).rotation_euler.z = rotation
    for dx in (-0.20, 0.20):
        for dy in (-0.18, 0.18):
            leg = box(name+f"_Leg_{dx:+.2f}_{dy:+.2f}", (x+dx, y+dy, 0.23), (0.055, 0.055, 0.46), mats["dark_metal"], furniture, 0.012)
            leg.rotation_euler.z = rotation


def emit_furniture(spec, collections, mats):
    furniture = collections["Furniture"]
    props = collections["Props"]
    # Entry joinery.
    box("EntryBench", (-5.35, -1.8, 0.43), (1.10, 0.48, 0.12), mats["light_wood"], furniture, 0.045)
    for x in (-5.75, -5.35, -4.95):
        box(f"EntryShoe_{x:.2f}", (x, -1.78, 0.18), (0.32, 0.42, 0.18), mats["dark_fabric"], props, 0.04)
    box("EntrySlatWall", (-5.78, -0.65, 1.45), (0.10, 1.45, 2.25), mats["light_wood"], furniture, 0.018)
    box("EntryMirror", (-5.70, -2.85, 1.45), (0.04, 0.90, 1.55), mats["glass"], props, 0.012)
    box("EntryRail", (-5.67, -0.65, 1.85), (0.05, 1.20, 0.05), mats["dark_metal"], props, 0.012)
    for index, z in enumerate((1.15, 1.45, 1.75)):
        cylinder(f"CoatHook_{index}", (-5.68, -0.65, z), 0.035, 0.16, mats["brass"], props)

    # Kitchen run and work surfaces.
    box("KitchenLowerRun", (0.35, -4.55, 0.47), (5.6, 0.72, 0.94), mats["cabinet"], furniture, 0.035)
    box("KitchenCounter", (0.35, -4.48, 0.99), (5.8, 0.86, 0.09), mats["stone"], furniture, 0.025)
    box("KitchenBacksplash", (0.35, -4.91, 1.65), (5.8, 0.04, 1.25), mats["tile"], furniture, 0.005)
    for index, x in enumerate((-1.45, -0.55, 0.35, 1.25, 2.15)):
        box(f"KitchenDoorSeam_{index}", (x, -4.17, 0.49), (0.018, 0.018, 0.82), mats["dark_metal"], props, 0.002)
        box(f"KitchenPull_{index}", (x+0.24, -4.14, 0.63), (0.24, 0.028, 0.025), mats["brass"], props, 0.006)
    box("KitchenOven", (0.35, -4.14, 0.48), (0.72, 0.08, 0.66), mats["dark_metal"], props, 0.025)
    box("KitchenOvenHandle", (0.35, -4.06, 0.72), (0.48, 0.035, 0.035), mats["steel"], props, 0.010)
    box("Fridge", (-2.35, -4.40, 1.36), (1.05, 0.82, 2.72), mats["steel"], furniture, 0.055)
    box("FridgeSeam", (-2.35, -3.98, 1.36), (0.018, 0.025, 2.45), mats["dark_metal"], furniture, 0.002)
    box("SinkBasin", (2.15, -4.47, 1.03), (0.92, 0.54, 0.12), mats["dark_metal"], props, 0.055)
    cylinder("FaucetStem", (2.55, -4.63, 1.38), 0.025, 0.68, mats["steel"], props)
    box("FaucetSpout", (2.38, -4.58, 1.67), (0.38, 0.045, 0.045), mats["steel"], props, 0.012)
    box("KitchenIsland", (0.2, -2.35, 0.49), (3.3, 1.10, 0.98), mats["cabinet"], furniture, 0.035)
    box("KitchenIslandTop", (0.2, -2.35, 1.03), (3.5, 1.25, 0.10), mats["stone"], furniture, 0.035)
    for index, x in enumerate((-0.85, 0.20, 1.25)):
        cylinder(f"Pendant_{index}", (x, -2.35, 2.55), 0.17, 0.22, mats["warm_light"], props)

    # Dining table and four chairs.
    box("DiningTop", (3.85, -1.30, 0.78), (2.25, 1.15, 0.10), mats["light_wood"], furniture, 0.055)
    for x in (3.0, 4.7):
        for y in (-1.72, -0.88):
            box(f"DiningLeg_{x}_{y}", (x, y, 0.38), (0.09, 0.09, 0.76), mats["dark_metal"], furniture, 0.016)
    # Chair local forward is -Y.  Both rows face the table, not away from it.
    for index, (x, y, rotation) in enumerate(((2.85,-2.15,math.pi),(4.85,-2.15,math.pi),(2.85,-0.42,0),(4.85,-0.42,0))):
        add_chair(f"DiningChair_{index}", x, y, rotation, collections, mats)
    place_settings = ((3.30,-1.58),(4.40,-1.58),(3.30,-1.02),(4.40,-1.02))
    for index, (x, y) in enumerate(place_settings):
        cylinder(f"DiningPlate_{index}", (x, y, 0.844), 0.17, 0.025, mats["ceramic"], props)
        cup_x = x + 0.28 if y < -1.30 else x - 0.28
        cylinder(f"DiningCup_{index}", (cup_x, y, 0.885), 0.055, 0.11, mats["ceramic"], props, 32)
        cylinder(f"DiningCupInset_{index}", (cup_x, y, 0.941), 0.042, 0.004, mats["dark_metal"], props, 32)
    cylinder("DiningSharedBowl", (3.85, -1.30, 0.885), 0.20, 0.10, mats["ceramic"], props, 48)
    cylinder("DiningSharedBowlInset", (3.85, -1.30, 0.937), 0.16, 0.004, mats["rug"], props, 48)

    # Living room.
    box("SofaBase", (-3.15, 3.75, 0.25), (3.0, 1.05, 0.40), mats["dark_fabric"], furniture, 0.10)
    box("SofaBack", (-3.15, 4.18, 0.78), (3.0, 0.25, 1.00), mats["dark_fabric"], furniture, 0.10)
    for index, x in enumerate((-4.05, -3.15, -2.25)):
        box(f"SofaCushion_{index}", (x, 3.60, 0.51), (0.78, 0.72, 0.12), mats["fabric"], furniture, 0.06)
    box("LivingRug", (-1.2, 2.35, 0.025), (3.4, 2.2, 0.035), mats["rug"], furniture, 0.008)
    box("CoffeeTable", (-1.2, 2.35, 0.42), (1.55, 0.78, 0.10), mats["light_wood"], furniture, 0.04)
    box("DogBed", (1.75, 4.0, 0.13), (1.05, 0.78, 0.26), mats["dog_bed"], furniture, 0.12)
    box("LivingMediaCabinet", (-0.9, 4.62, 0.48), (2.6, 0.45, 0.78), mats["cabinet"], furniture, 0.04)
    cylinder("LivingLampPole", (1.25, 3.20, 0.95), 0.025, 1.90, mats["dark_metal"], props)
    cylinder("LivingLampShade", (1.25, 3.20, 1.86), 0.22, 0.32, mats["warm_light"], props)
    cylinder("LivingPlantPot", (1.90, 2.70, 0.28), 0.22, 0.55, mats["ceramic"], props)
    for index, offset in enumerate((-0.18, 0.0, 0.18)):
        leaf = box(f"LivingPlantLeaf_{index}", (1.90+offset, 2.70, 0.95+0.12*index), (0.12, 0.06, 0.95), mats["dog_bed"], props, 0.04)
        leaf.rotation_euler.y = math.radians(-18+18*index)

    # Study.
    box("StudyDesk", (4.65, 4.18, 0.76), (2.05, 0.75, 0.09), mats["light_wood"], furniture, 0.04)
    box("StudyMonitor", (4.65, 4.36, 1.30), (0.86, 0.08, 0.54), mats["dark_metal"], props, 0.018)
    box("StudyMonitorStand", (4.65, 4.25, 1.01), (0.10, 0.18, 0.35), mats["steel"], props, 0.010)
    box("StudyKeyboard", (4.65, 3.94, 0.84), (0.72, 0.22, 0.035), mats["dark_metal"], props, 0.008)
    for x in (3.78, 5.52):
        box(f"StudyDeskLeg_{x}", (x, 4.18, 0.37), (0.08, 0.62, 0.74), mats["dark_metal"], furniture, 0.012)
    add_chair("StudyChair", 4.65, 3.45, 0.0, collections, mats)
    box("StudyBookcase", (5.66, 2.15, 1.35), (0.55, 2.30, 2.70), mats["wood"], furniture, 0.035)
    for row in range(5):
        box(f"StudyShelf_{row}", (5.34, 2.15, 0.30+row*0.52), (0.62, 2.20, 0.045), mats["light_wood"], furniture, 0.008)
    for row in range(4):
        for column in range(7):
            color = mats["book_warm"] if (row+column)%2 else mats["book_cool"]
            box(f"Book_{row}_{column}", (5.26, 1.30+column*0.27, 0.54+row*0.52), (0.14, 0.18, 0.36), color, props, 0.006)


def add_cameras(collections, spec):
    cameras = collections["Cameras"]
    definitions = [
        ("CAM_01_Entry", (-4.45,-4.25,1.55), (-5.05,-1.35,1.10), 30),
        ("CAM_02_Kitchen", (-2.55,0.15,1.65), (0.45,-3.55,1.15), 29),
        ("CAM_03_Dining", (5.35,-4.2,1.55), (3.8,-1.25,0.95), 40),
        ("CAM_04_Living", (-4.8,0.9,1.55), (-1.6,3.0,0.95), 34),
        ("CAM_05_Study", (3.30,1.45,1.55), (4.65,3.65,1.05), 32),
        ("CAM_06_ReverseAudit", (0.0,4.45,1.65), (0.0,-2.70,1.05), 30),
    ]
    result = []
    for name, location, target, lens in definitions:
        data = bpy.data.cameras.new(name+"_Data")
        data.lens = lens
        data.sensor_width = 36
        obj = bpy.data.objects.new(name, data)
        cameras.objects.link(obj)
        obj.location = layout_point(spec, location)
        aim(obj, layout_point(spec, target))
        result.append(obj)
    return result


def configure_scene(scene, collections, spec):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.fps = 15
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.world.color = (0.035, 0.045, 0.06)
    lights = collections["Lights"]
    for name, location, target, energy, size, color in (
        ("EntryLight", (-4.5,-2.0,2.65), (-4.5,-2.0,0), 500, 2.0, (1.0,0.82,0.66)),
        ("KitchenLight", (0.5,-2.5,2.70), (0.5,-2.5,0), 820, 3.0, (1.0,0.88,0.74)),
        ("LivingLight", (-1.5,2.5,2.70), (-1.5,2.5,0), 650, 3.0, (1.0,0.85,0.72)),
        ("StudyLight", (4.6,3.2,2.65), (4.6,3.2,0), 500, 2.0, (0.78,0.88,1.0)),
    ):
        add_area(name, layout_point(spec, location), layout_point(spec, target), energy, size, color, lights)
    sun_data = bpy.data.lights.new("Sun_Data", "SUN")
    sun_data.energy = 1.3
    sun_data.color = (0.72, 0.82, 1.0)
    sun = bpy.data.objects.new("Sun", sun_data)
    lights.objects.link(sun)
    sun.rotation_euler = (math.radians(35), math.radians(-20), math.radians(-35))
    if spec.get("layout_transform", {}).get("kind") == "mirror_x":
        sun.rotation_euler.z = -sun.rotation_euler.z


def main():
    args = args_from_cli()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"fresh output required: {root}")
    for path in (root/"renders/preview_cycle_00", root/"visual", root/"qa"):
        path.mkdir(parents=True, exist_ok=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    scene.name = "CompactHousehold"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    collections = {}
    for name in ("Architecture","Glazing","Furniture","Props","Anchors","Lights","Cameras"):
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)
        collections[name] = collection
    mats = {
        "wall": material("WarmPlaster", (0.64,0.58,0.50), 0.76),
        "ceiling": material("SoftCeiling", (0.78,0.76,0.72), 0.80),
        "wood_floor": material("OakFloor", (0.34,0.20,0.10), 0.45),
        "stone": material("EntryStone", (0.34,0.35,0.36), 0.65),
        "ground": material("ExteriorPaving", (0.18,0.20,0.22), 0.75),
        "glass": material("WindowGlass", (0.65,0.78,0.88), 0.08, transmission=0.6),
        "light_wood": material("LightOak", (0.58,0.34,0.16), 0.42),
        "wood": material("DarkOak", (0.26,0.12,0.055), 0.38),
        "cabinet": material("SageCabinet", (0.18,0.24,0.20), 0.50),
        "stone": material("HonedStone", (0.58,0.55,0.50), 0.34),
        "tile": material("BacksplashTile", (0.72,0.71,0.67), 0.26),
        "steel": material("BrushedSteel", (0.38,0.41,0.44), 0.30, metallic=0.82),
        "dark_metal": material("DarkMetal", (0.045,0.05,0.055), 0.38, metallic=0.45),
        "brass": material("AgedBrass", (0.38,0.18,0.04), 0.28, metallic=0.75),
        "fabric": material("WarmFabric", (0.48,0.28,0.18), 0.78),
        "dark_fabric": material("DeepBlueFabric", (0.06,0.10,0.16), 0.80),
        "rug": material("Rug", (0.32,0.20,0.14), 0.92),
        "dog_bed": material("DogBed", (0.18,0.30,0.26), 0.88),
        "ceramic": material("Ceramic", (0.80,0.76,0.68), 0.18),
        "book_warm": material("BookWarm", (0.55,0.18,0.08), 0.62),
        "book_cool": material("BookCool", (0.10,0.24,0.34), 0.62),
        "warm_light": material("WarmEmitter", (0.9,0.55,0.20), 0.20, emission=(1.0,0.45,0.15)),
    }
    # Restore the entry floor material after the common stone counter material.
    mats["entry_floor"] = material("EntryStoneFloor", (0.28,0.29,0.30), 0.72)
    segments, wall_count = emit_architecture(spec, collections, mats)
    emit_furniture(spec, collections, mats)
    transform_furniture_for_layout(spec, collections)

    for anchor_id, position in spec["anchors"].items():
        obj = bpy.data.objects.new("ANCHOR_"+anchor_id, None)
        collections["Anchors"].objects.link(obj)
        obj.location = position
        obj["anchor_id"] = anchor_id
        obj.hide_render = True
    cameras = add_cameras(collections, spec)
    configure_scene(scene, collections, spec)
    scene.camera = cameras[0]

    scene_id = spec["room_spec_id"]
    blend = root / f"{scene_id}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    for camera in cameras:
        scene.camera = camera
        scene.render.filepath = str(root/"renders/preview_cycle_00"/(camera.name+".png"))
        bpy.ops.render.render(write_still=True)

    collections["Anchors"].hide_viewport = True
    visual_glb = root / f"visual/{scene_id}.glb"
    result = bpy.ops.export_scene.gltf(
        filepath=str(visual_glb), check_existing=False, export_format="GLB",
        use_active_scene=True, use_visible=True, export_apply=True,
        export_yup=True, export_animations=False, export_cameras=False,
        export_lights=False, export_materials="EXPORT", export_extras=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB export failed: {result}")
    write_json(root/"functional_anchors.json", {
        "room_spec_id": spec["room_spec_id"],
        "coordinate_system": "Blender +Z up metres; exported GLB +Y up",
        "anchors": spec["anchors"],
    })
    write_json(root/"qa/build_report.json", {
        "schema": "avengine_life_room_build_report_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "zones": len(spec["zones"]),
        "links": len(spec["links"]),
        "exterior_openings": len(spec["exterior_openings"]),
        "wall_panel_count": wall_count,
        "object_count": len(scene.objects),
        "review_cameras": [camera.name for camera in cameras],
        "visual_glb": str(visual_glb),
        "blend": str(blend),
    })
    print("LIFE_ROOM_BUILD_COMPLETE", root)


if __name__ == "__main__":
    main()
