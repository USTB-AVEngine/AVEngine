"""Blender-side builder for an inward-facing, textured exterior sphere."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-hdri", required=True, type=Path)
    parser.add_argument("--output-glb", required=True, type=Path)
    parser.add_argument("--visual-profile", required=True, type=Path)
    parser.add_argument("--texture-width", type=int, default=2048)
    return parser.parse_args(argv)


def _habitat_to_blender(value: list[float]) -> Vector:
    """Convert Habitat/glTF Y-up coordinates to Blender Z-up coordinates."""

    return Vector((value[0], -value[2], value[1]))


def _add_window_panel(panel: dict, material: bpy.types.Material) -> bpy.types.Object:
    center = _habitat_to_blender(panel["center_from_listener_m"])
    width = _habitat_to_blender(panel["width_axis"])
    height = _habitat_to_blender(panel["height_axis"])
    half_w = float(panel["size_wh_m"][0]) / 2.0
    half_h = float(panel["size_wh_m"][1]) / 2.0
    vertices = [
        center - width * half_w - height * half_h,
        center + width * half_w - height * half_h,
        center + width * half_w + height * half_h,
        center - width * half_w + height * half_h,
    ]
    mesh = bpy.data.meshes.new(f"{panel['panel_id']}_mesh")
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.materials.append(material)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    u0, v0, u1, v1 = (float(item) for item in panel["uv_rect"])
    uv_by_vertex = ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uv_by_vertex[vertex_index]
    mesh.update()
    result = bpy.data.objects.new(panel["panel_id"], mesh)
    bpy.context.collection.objects.link(result)
    return result


def main() -> None:
    args = parse_args()
    hdri = args.input_hdri.resolve()
    output = args.output_glb.resolve()
    visual_profile_path = args.visual_profile.resolve()
    if not hdri.is_file():
        raise RuntimeError(f"input HDRI is missing: {hdri}")
    if not visual_profile_path.is_file():
        raise RuntimeError(f"visual profile is missing: {visual_profile_path}")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite exterior proxy: {output}")
    if args.texture_width < 512 or args.texture_width % 2:
        raise RuntimeError("texture width must be an even integer >= 512")
    output.parent.mkdir(parents=True, exist_ok=True)
    visual_profile = json.loads(visual_profile_path.read_text(encoding="utf-8"))
    exterior = visual_profile["exterior_proxy"]
    if exterior["proxy_kind"] != "inward_uv_sphere_with_fixed_window_panels":
        raise RuntimeError("visual profile does not declare the fixed window panels")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    image = bpy.data.images.load(str(hdri), check_existing=False)
    texture_height = args.texture_width // 2
    image.scale(args.texture_width, texture_height)
    texture = output.with_name(f"{output.stem}_tonemapped.png")
    scene = bpy.context.scene
    scene.view_settings.look = "AgX - Medium High Contrast"
    image.filepath_raw = str(texture)
    image.file_format = "PNG"
    image.save_render(str(texture), scene=scene)
    if not texture.is_file() or texture.stat().st_size == 0:
        raise RuntimeError("Blender did not create the tonemapped exterior texture")

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=96,
        ring_count=48,
        radius=float(exterior["sphere_radius_m"]),
    )
    sphere = bpy.context.active_object
    sphere.name = "approaching_storm_visual_only_exterior"
    for polygon in sphere.data.polygons:
        polygon.flip()
    sphere.data.update()

    material = bpy.data.materials.new("approaching_storm_unlit_exterior")
    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    if principled is None:
        raise RuntimeError("Blender Principled BSDF node is unavailable")
    texture_node = nodes.new("ShaderNodeTexImage")
    texture_node.image = bpy.data.images.load(str(texture), check_existing=False)
    texture_node.interpolation = "Linear"
    links.new(texture_node.outputs["Color"], principled.inputs["Base Color"])
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Metallic"].default_value = 0.0
    sphere.data.materials.append(material)

    objects = [sphere]
    objects.extend(
        _add_window_panel(panel, material) for panel in exterior["window_panels"]
    )
    bpy.ops.object.select_all(action="DESELECT")
    for item in objects:
        item.select_set(True)
    bpy.context.view_layer.objects.active = sphere
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_yup=True,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Blender did not create the exterior proxy GLB")


main()
