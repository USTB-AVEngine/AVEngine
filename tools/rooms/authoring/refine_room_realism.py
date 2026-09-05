#!/usr/bin/env python3
"""Create an appearance-only realism pass for the accepted household room."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import bpy
from mathutils import Vector


TEXTURE_ROOT = Path(
    os.environ.get(
        "AVENGINE_AUTHORING_TEXTURE_ROOT",
        "/data/jzy/blender_projects/chef_home_test_kitchen_vfx_v1/assets/source_4k",
    )
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def image(path: Path, *, noncolor: bool = False) -> bpy.types.Image:
    require(path.is_file(), f"missing texture: {path}")
    value = bpy.data.images.load(str(path), check_existing=True)
    if noncolor:
        value.colorspace_settings.name = "Non-Color"
    return value


def pbr_material(
    name: str,
    *,
    diffuse: Path | None,
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
    roughness: Path | None = None,
    normal: Path | None = None,
    roughness_default: float = 0.45,
    metallic: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = roughness_default
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Base Color"].default_value = base_color
    if diffuse is not None:
        color = nodes.new("ShaderNodeTexImage")
        color.image = image(diffuse)
        color.extension = "REPEAT"
        color.interpolation = "Linear"
        links.new(color.outputs["Color"], principled.inputs["Base Color"])
    if roughness is not None:
        rough = nodes.new("ShaderNodeTexImage")
        rough.image = image(roughness, noncolor=True)
        rough.extension = "REPEAT"
        links.new(rough.outputs["Color"], principled.inputs["Roughness"])
    if normal is not None:
        normal_tex = nodes.new("ShaderNodeTexImage")
        normal_tex.image = image(normal, noncolor=True)
        normal_tex.extension = "REPEAT"
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = 0.38
        links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return mat


def principled_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
    coat: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Coat Weight"].default_value = coat
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return mat


def glass_material() -> bpy.types.Material:
    mat = bpy.data.materials.get("Real_WindowGlass") or bpy.data.materials.new(
        "Real_WindowGlass"
    )
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (0.72, 0.86, 0.94, 1.0)
    shader.inputs["Roughness"].default_value = 0.08
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Transmission Weight"].default_value = 0.82
    shader.inputs["IOR"].default_value = 1.45
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    mat.surface_render_method = "DITHERED"
    return mat


def projected_uvs(obj: bpy.types.Object, tile_m: float) -> None:
    mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh):
        return
    uv = mesh.uv_layers.active or mesh.uv_layers.new(name="RealismUV")
    sx, sy, sz = (float(value) for value in obj.scale)
    for polygon in mesh.polygons:
        normal = polygon.normal
        axis = max(range(3), key=lambda item: abs(normal[item]))
        for loop_index in polygon.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            point = (co.x * sx, co.y * sy, co.z * sz)
            if axis == 2:
                values = (point[0], point[1])
            elif axis == 1:
                values = (point[0], point[2])
            else:
                values = (point[1], point[2])
            uv.data[loop_index].uv = (values[0] / tile_m, values[1] / tile_m)


def add_bevel(obj: bpy.types.Object, width: float, segments: int = 3) -> None:
    if obj.type != "MESH" or min(float(value) for value in obj.dimensions) <= 0.02:
        return
    modifier = obj.modifiers.new("RealismBevel", "BEVEL")
    modifier.width = width
    modifier.segments = segments
    modifier.limit_method = "ANGLE"
    modifier.angle_limit = math.radians(28.0)


def add_box(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    bevel: float = 0.008,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel > 0.0:
        add_bevel(obj, bevel)
    return obj


def add_sphere(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=32, ring_count=16, radius=radius, location=location
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    (output / "renders").mkdir()

    plaster = pbr_material(
        "Real_WarmPlaster",
        diffuse=None,
        base_color=(0.58, 0.52, 0.44, 1.0),
        roughness=TEXTURE_ROOT / "white_plaster_02/white_plaster_02_rough_4k.jpg",
        normal=TEXTURE_ROOT / "white_plaster_02/white_plaster_02_nor_gl_4k.exr",
        roughness_default=0.72,
    )
    ceiling = principled_material(
        "Real_CleanCeiling", (0.76, 0.72, 0.66, 1.0), roughness=0.82
    )
    wood = pbr_material(
        "Real_NaturalWood",
        diffuse=TEXTURE_ROOT / "lacquered_cherry_wood/lacquered_cherry_wood_diff_4k.jpg",
        roughness=TEXTURE_ROOT / "lacquered_cherry_wood/lacquered_cherry_wood_rough_4k.exr",
        normal=TEXTURE_ROOT / "lacquered_cherry_wood/lacquered_cherry_wood_nor_gl_4k.exr",
        roughness_default=0.42,
    )
    stone = pbr_material(
        "Real_HonedStone",
        diffuse=TEXTURE_ROOT / "marble_01/marble_01_diff_4k.jpg",
        roughness=TEXTURE_ROOT / "marble_01/marble_01_rough_4k.jpg",
        normal=TEXTURE_ROOT / "marble_01/marble_01_nor_gl_4k.exr",
        roughness_default=0.28,
    )
    terrazzo = pbr_material(
        "Real_Terrazzo",
        diffuse=TEXTURE_ROOT / "terrazzo_tiles/terrazzo_tiles_diff_4k.jpg",
        roughness=TEXTURE_ROOT / "terrazzo_tiles/terrazzo_tiles_rough_4k.exr",
        normal=TEXTURE_ROOT / "terrazzo_tiles/terrazzo_tiles_nor_gl_4k.exr",
        roughness_default=0.5,
    )
    sage = principled_material(
        "Real_SageCabinet", (0.22, 0.34, 0.27, 1.0), roughness=0.31, coat=0.18
    )
    sage_panel = principled_material(
        "Real_SageCabinetPanel", (0.17, 0.28, 0.22, 1.0), roughness=0.34, coat=0.12
    )
    brass = principled_material(
        "Real_AgedBrass", (0.42, 0.20, 0.055, 1.0), roughness=0.27, metallic=0.86
    )
    steel = principled_material(
        "Real_BrushedSteel", (0.43, 0.47, 0.50, 1.0), roughness=0.24, metallic=0.92
    )
    dark_metal = principled_material(
        "Real_DarkMetal", (0.018, 0.021, 0.024, 1.0), roughness=0.3, metallic=0.84
    )
    ceramic = principled_material(
        "Real_Ceramic", (0.91, 0.88, 0.80, 1.0), roughness=0.18, coat=0.22
    )
    fabric = principled_material(
        "Real_WarmFabric", (0.31, 0.12, 0.055, 1.0), roughness=0.74
    )
    glass = glass_material()
    fruit_red = principled_material(
        "Real_FruitRed", (0.48, 0.018, 0.012, 1.0), roughness=0.28, coat=0.25
    )
    fruit_yellow = principled_material(
        "Real_FruitYellow", (0.62, 0.25, 0.015, 1.0), roughness=0.3, coat=0.2
    )

    replacements = {
        "WarmPlaster": (plaster, 1.8),
        "SoftCeiling": (ceiling, None),
        "OakFloor": (wood, 1.15),
        "LightOak": (wood, 0.85),
        "DarkOak": (wood, 0.85),
        "HonedStone": (stone, 1.2),
        "BacksplashTile": (terrazzo, 0.75),
        "EntryStoneFloor": (terrazzo, 0.9),
        "ExteriorPaving": (terrazzo, 1.0),
        "SageCabinet": (sage, None),
        "AgedBrass": (brass, None),
        "BrushedSteel": (steel, None),
        "DarkMetal": (dark_metal, None),
        "Ceramic": (ceramic, None),
        "WarmFabric": (fabric, None),
        "WindowGlass": (glass, None),
    }
    textured_objects = 0
    replaced_slots = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        tile = None
        for slot in obj.material_slots:
            if slot.material is None or slot.material.name not in replacements:
                continue
            replacement, candidate_tile = replacements[slot.material.name]
            slot.material = replacement
            replaced_slots += 1
            tile = candidate_tile or tile
        if tile is not None:
            projected_uvs(obj, tile)
            textured_objects += 1

    no_bevel_prefixes = (
        "Floor_",
        "Ceiling_",
        "Wall_",
        "Glass_",
        "ExteriorGround",
        "KitchenBacksplash",
    )
    beveled_objects = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name.startswith(no_bevel_prefixes):
            continue
        minimum = min(float(value) for value in obj.dimensions)
        if minimum <= 0.025:
            continue
        width = min(0.018, 0.10 * minimum)
        add_bevel(obj, width, 3)
        beveled_objects += 1

    # Appearance-only cabinet joinery on the camera-facing island side.
    for index in range(5):
        x = -1.16 + index * 0.68
        add_box(
            f"Realism_IslandPanel_{index}",
            (x, -1.785, 0.53),
            (0.60, 0.035, 0.72),
            sage_panel,
            bevel=0.012,
        )
        add_box(
            f"Realism_IslandHandle_{index}",
            (x, -1.752, 0.79),
            (0.22, 0.025, 0.026),
            brass,
            bevel=0.006,
        )
    add_box(
        "Realism_IslandToeKick",
        (0.2, -1.765, 0.105),
        (3.08, 0.055, 0.16),
        dark_metal,
        bevel=0.004,
    )

    # Back-run cabinet seams remain behind the existing island and do not
    # modify any navigation or acoustic authority.
    for index in range(7):
        x = -2.0 + index * 0.78
        add_box(
            f"Realism_BackPanel_{index}",
            (x, -4.175, 0.50),
            (0.70, 0.032, 0.73),
            sage_panel,
            bevel=0.01,
        )
        add_box(
            f"Realism_BackHandle_{index}",
            (x, -4.145, 0.76),
            (0.20, 0.022, 0.024),
            brass,
            bevel=0.005,
        )

    # Window frame and a restrained set of occupied-home countertop props.
    for name, location, dimensions in (
        ("Realism_WindowTop", (0.5, -4.965, 2.445), (2.68, 0.055, 0.07)),
        ("Realism_WindowBottom", (0.5, -4.965, 1.055), (2.68, 0.055, 0.07)),
        ("Realism_WindowLeft", (-0.805, -4.965, 1.75), (0.07, 0.055, 1.46)),
        ("Realism_WindowRight", (1.805, -4.965, 1.75), (0.07, 0.055, 1.46)),
    ):
        add_box(name, location, dimensions, dark_metal, bevel=0.006)

    board = add_box(
        "Realism_CuttingBoard",
        (-0.88, -2.28, 1.095),
        (0.62, 0.34, 0.026),
        wood,
        bevel=0.018,
    )
    projected_uvs(board, 0.55)
    add_sphere("Realism_AppleRed", (-0.98, -2.28, 1.19), 0.075, fruit_red)
    add_sphere("Realism_AppleYellow", (-0.80, -2.22, 1.18), 0.07, fruit_yellow)
    add_box(
        "Realism_SoapDispenser",
        (2.63, -4.34, 1.17),
        (0.10, 0.10, 0.24),
        ceramic,
        bevel=0.014,
    )
    add_box(
        "Realism_SoapPump",
        (2.63, -4.34, 1.31),
        (0.08, 0.05, 0.035),
        steel,
        bevel=0.006,
    )

    camera_data = bpy.data.cameras.new("CAM_Realism_Kitchen_AVEngine")
    camera = bpy.data.objects.new("CAM_Realism_Kitchen_AVEngine", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = (0.0, 0.8, 1.75)
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = 18.0
    look_at(camera, Vector((0.8, -2.2, 0.8)))
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output / "renders/kitchen_realism_preview.png")
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"
    if scene.world is not None:
        scene.world.color = (0.055, 0.07, 0.09)

    blend_path = output / "compact_household_realism_v1.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.render.render(write_still=True)
    write_json(
        output / "realism_report.json",
        {
            "status": "research_candidate",
            "qualification_claim": False,
            "appearance_only": True,
            "source_room": "compact_household_cycle03",
            "texture_source": "Poly Haven CC0 assets registered in chef_home_test_kitchen_vfx_v1",
            "textured_object_count": textured_objects,
            "replaced_material_slot_count": replaced_slots,
            "beveled_existing_object_count": beveled_objects,
            "new_visual_detail_object_count": 5 * 2 + 1 + 7 * 2 + 4 + 5,
            "room_spec_geometry_changed": False,
            "navigation_changed": False,
            "acoustic_proxy_changed": False,
            "artifacts": {
                "blend": str(blend_path),
                "preview": str(output / "renders/kitchen_realism_preview.png"),
            },
        },
    )


if __name__ == "__main__":
    main()
