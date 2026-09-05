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
    return parser.parse_args()


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
        modifier.width = min(bevel, min(size) * 0.24)
        modifier.segments = 4
        modifier.limit_method = "ANGLE"
    return obj


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
        changed.append(value.name)
    return sorted(set(changed))


def polish_sofas(furniture, props, changes: list[str]) -> None:
    pillow_mat = material("Polish_Muted_Pillow", (0.42, 0.53, 0.51, 1.0), 0.88)
    pillow_alt = material("Polish_Warm_Pillow", (0.67, 0.43, 0.28, 1.0), 0.88)
    leg_mat = material("Polish_Sofa_Leg", (0.07, 0.055, 0.045, 1.0), 0.48, 0.15)
    back_objects = [obj for obj in furniture.objects if obj.type == "MESH"
                    and obj.name.casefold().startswith("sofaback")]
    # The source names are stable in the shared builder; tolerate the Unicode
    # typo guard above by also matching the ordinary ASCII prefix.
    back_objects = [obj for obj in furniture.objects if obj.type == "MESH"
                    and obj.name.casefold().startswith("sofaback".replace("​​", ""))]
    bases = [obj for obj in furniture.objects if obj.type == "MESH"
             and obj.name.casefold().startswith("sofabase".replace("​​", ""))]
    for index, base in enumerate(sorted(bases, key=lambda obj: obj.name)):
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
        dims = back.dimensions
        for pillow_index, dx in enumerate((-0.31 * dims.x, 0.31 * dims.x)):
            point = local_point(back, dx, -0.16, 0.08)
            cube(
                f"PolishSofaPillow_{index}_{pillow_index}",
                (point.x, point.y, point.z + 0.18),
                (0.52, 0.18, 0.34),
                pillow_mat if pillow_index == 0 else pillow_alt,
                furniture,
                bevel=0.10,
                yaw=float(back.rotation_euler.z),
            )
        changes.append(f"added_soft_pillows_{back.name}")
    for obj in furniture.objects:
        if obj.type != "MESH" or "cushion" not in obj.name.casefold():
            continue
        obj.scale.z *= 1.28
        modifier = obj.modifiers.new("PolishCushionRoundover", "BEVEL")
        modifier.width = 0.065
        modifier.segments = 4
        modifier.limit_method = "ANGLE"
        changes.append(f"rounded_cushion_{obj.name}")


def polish_plants(furniture, props, changes: list[str]) -> None:
    leaf_mat = material("Polish_Leaf_Green", (0.08, 0.29, 0.13, 1.0), 0.78)
    leaf_light = material("Polish_Leaf_Light", (0.18, 0.43, 0.20, 1.0), 0.80)
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
        base = pot.location.copy()
        for leaf_index in range(8):
            angle = (2.0 * math.pi * leaf_index / 8.0) + 0.18 * plant_index
            radius = 0.12 + 0.035 * (leaf_index % 3)
            point = base + Vector((
                math.cos(angle) * radius,
                math.sin(angle) * radius,
                0.54 + 0.08 * (leaf_index % 3),
            ))
            leaf = bpy.ops.mesh.primitive_uv_sphere_add(
                segments=24, ring_count=12, radius=1.0,
                location=(point.x, point.y, point.z),
            )
            obj = bpy.context.object
            obj.name = f"PolishPlantLeaf_{plant_index}_{leaf_index}"
            for owner in list(obj.users_collection):
                owner.objects.unlink(obj)
            props.objects.link(obj)
            obj.scale = (0.11, 0.045, 0.33 + 0.05 * (leaf_index % 2))
            obj.rotation_euler = (
                math.radians(18.0 * math.sin(angle)),
                math.radians(-22.0 * math.cos(angle)),
                angle,
            )
            obj.data.materials.append(leaf_mat if leaf_index % 2 else leaf_light)
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
        for stem_index in range(4):
            angle = 2.0 * math.pi * stem_index / 4.0
            point = base + Vector((math.cos(angle) * 0.05, math.sin(angle) * 0.05, 0.43))
            cyl(
                f"PolishPlantStem_{plant_index}_{stem_index}",
                (point.x, point.y, point.z),
                0.014, 0.70, stem_mat, props,
            )
        changes.append(f"replaced_rod_plant_{pot.name}")


def add_wall_art_and_decor(furniture, props, changes: list[str]) -> None:
    frame_mat = material("Polish_Wall_Frame", (0.10, 0.07, 0.045, 1.0), 0.44, 0.08)
    art_mats = [
        material("Polish_Art_Clay", (0.58, 0.28, 0.18, 1.0), 0.76),
        material("Polish_Art_Sage", (0.25, 0.40, 0.32, 1.0), 0.76),
        material("Polish_Art_Ochre", (0.70, 0.46, 0.16, 1.0), 0.76),
    ]
    backs = [obj for obj in furniture.objects if obj.type == "MESH"
             and obj.name.casefold().startswith("sofaback".replace("​​", ""))]
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
    raw = spec.get("furnishing_assemblies") or spec.get("furniture") or []
    records: list[dict[str, Any]] = []
    for index, assembly in enumerate(raw):
        if not isinstance(assembly, dict):
            continue
        kind = assembly.get("kind") or assembly.get("category") or "furniture"
        center = (
            assembly.get("center_xy_m")
            or assembly.get("sofa_center_xy_m")
            or assembly.get("desk_center_xy_m")
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
    write_json(output / "object_semantics.json", {
        "kind": "avengine_polished_object_semantics",
        "room_id": base_name,
        "room_spec_id": spec.get("room_spec_id") if isinstance(spec, dict) else None,
        "source_blend": str(blend),
        "static_scene": True,
        "objects": semantics,
    })
    write_json(output / "functional_anchors.json", {
        "kind": "avengine_polished_functional_anchors",
        "room_id": base_name,
        "room_spec_id": spec.get("room_spec_id") if isinstance(spec, dict) else None,
        "source_blend": str(blend),
        "anchors": anchors,
    })
    write_json(output / "polish_report.json", {
        "kind": "avengine_polished_room_report",
        "status": "research_candidate",
        "qualification_claim": False,
        "source_blend": str(blend),
        "changes": changes,
        "anchors": anchors,
        "furniture_semantics": semantics,
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
