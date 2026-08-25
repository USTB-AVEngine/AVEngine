#!/usr/bin/env python3
"""Build a clearly labelled 3D-FRONT Toolbox sample review proxy in USD.

This script runs inside Blender (``blender --background --python ... --``).
The public official Toolbox sample contains five posed furniture meshes and a
single reference rendering, but no complete 3D-FRONT house shell or texture
library.  The generated USD therefore remains a local, non-redistributable
comparison proxy and must never be reported as a full 3D-FRONT scene.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


SCENE_METADATA_SCHEMA = "avengine_optional_residential_scene_metadata_v1"


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-usd", type=Path)
    parser.add_argument("--output-glb", type=Path)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for value in list(datablocks):
            if value.users == 0:
                datablocks.remove(value)


def _material(name: str, color: tuple[float, float, float], roughness: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Roughness"].default_value = roughness
    return material


def _projected_material(image_path: Path) -> bpy.types.Material:
    material = bpy.data.materials.new("official_raw_scene_projected")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    principled.inputs["Roughness"].default_value = 0.48
    texture = nodes.new("ShaderNodeTexImage")
    texture.name = "OfficialToolboxRawScene"
    texture.image = bpy.data.images.load(str(image_path.resolve()), check_existing=True)
    texture.interpolation = "Linear"
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    return material


def _backplate_material(image_path: Path) -> bpy.types.Material:
    material = bpy.data.materials.new("official_raw_scene_backplate")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    texture = nodes.new("ShaderNodeTexImage")
    texture.name = "OfficialToolboxRawSceneBackplate"
    texture.image = bpy.data.images.load(str(image_path.resolve()), check_existing=True)
    texture.interpolation = "Linear"
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    principled.inputs["Roughness"].default_value = 1.0
    # Blender 4.x exports these Principled inputs to UsdPreviewSurface.  The
    # low emissive contribution keeps the official reference readable without
    # making it an unlit white billboard in UE.
    if "Emission Color" in principled.inputs:
        links.new(texture.outputs["Color"], principled.inputs["Emission Color"])
        principled.inputs["Emission Strength"].default_value = 0.35
    return material


def _make_quad(
    *, name: str, vertices: list[tuple[float, float, float]], material: bpy.types.Material
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop, uv in zip(mesh.loops, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=True):
        uv_layer.data[loop.index].uv = uv
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _sample_to_room(point: np.ndarray, camera_height: float) -> np.ndarray:
    # Official sample camera: +X right, +Y up, looks down -Z.  Review room:
    # +X forward, +Y right, +Z up.
    return np.asarray([-point[2], point[0], point[1] + camera_height], dtype=np.float64)


def _import_projected_object(
    *,
    obj_path: Path,
    pose: dict,
    material: bpy.types.Material,
    fov_degrees: float,
    camera_height: float,
) -> dict:
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(obj_path.resolve()), use_split_objects=True, use_split_groups=False)
    imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"OBJ import produced no mesh: {obj_path}")
    rotation = np.asarray(pose["rotation"], dtype=np.float64)
    translation = np.asarray(pose["translation"], dtype=np.float64)
    tangent = math.tan(math.radians(float(fov_degrees)) / 2.0)
    room_points: list[np.ndarray] = []
    for part_index, obj in enumerate(imported):
        obj.name = f"future_{obj_path.stem}_{part_index:02d}"
        obj.data.name = f"{obj.name}_mesh"
        sample_by_vertex: list[np.ndarray] = []
        for vertex in obj.data.vertices:
            local = np.asarray(vertex.co[:], dtype=np.float64)
            sample = rotation @ local + translation
            room = _sample_to_room(sample, camera_height)
            vertex.co = Vector(room.tolist())
            sample_by_vertex.append(sample)
            room_points.append(room)
        uv_layer = obj.data.uv_layers.get("UVMap") or obj.data.uv_layers.new(name="UVMap")
        for loop in obj.data.loops:
            sample = sample_by_vertex[loop.vertex_index]
            depth = max(1.0e-6, -float(sample[2]))
            u = 0.5 * (float(sample[0]) / depth / tangent + 1.0)
            v = 0.5 * (float(sample[1]) / depth / tangent + 1.0)
            uv_layer.data[loop.index].uv = (u, v)
        obj.data.materials.clear()
        obj.data.materials.append(material)
        obj.matrix_world = Matrix.Identity(4)
    points = np.asarray(room_points, dtype=np.float64)
    return {
        "object_id": obj_path.stem,
        "bounds_xyz_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "navigation_role": (
            "elevated_object" if float(points.min(axis=0)[2]) >= 1.75 else "ground_blocker"
        ),
    }


def main() -> int:
    args = _args()
    root = args.sample_root.expanduser().resolve()
    output_usd = args.output_usd.expanduser().resolve() if args.output_usd else None
    output_glb = args.output_glb.expanduser().resolve() if args.output_glb else None
    if output_usd is None and output_glb is None:
        raise ValueError("at least one of --output-usd or --output-glb is required")
    metadata_path = args.output_metadata.expanduser().resolve()
    if not (root / "scene_pose_info.npy").is_file():
        raise FileNotFoundError(root / "scene_pose_info.npy")
    if not (root / "images/raw_scene.png").is_file():
        raise FileNotFoundError(root / "images/raw_scene.png")
    for path in tuple(value for value in (output_usd, output_glb, metadata_path) if value):
        if path.exists() and not args.replace:
            raise FileExistsError(f"refusing to replace output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.replace:
            path.unlink(missing_ok=True)

    _clear()
    raw_scene = root / "images/raw_scene.png"
    projected = _projected_material(raw_scene)
    backplate_material = _backplate_material(raw_scene)
    floor_material = _material("warm_neutral_floor", (0.29, 0.25, 0.21), 0.62)
    side_material = _material("warm_neutral_wall", (0.74, 0.70, 0.64), 0.78)

    poses = np.load(root / "scene_pose_info.npy", allow_pickle=True).tolist()
    camera_height = 1.17
    fov = float(poses[0]["fov"])
    objects = []
    for pose in poses:
        model = root / "models" / f"{pose['shape_id']}.obj"
        objects.append(
            _import_projected_object(
                obj_path=model,
                pose=pose,
                material=projected,
                fov_degrees=fov,
                camera_height=camera_height,
            )
        )

    # The original Toolbox sample does not ship a house shell.  A simple
    # bounded review shell catches actor shadows; the exact official reference
    # image remains visible as a far backplate from the sample camera.
    half = 8.0 * math.tan(math.radians(fov) / 2.0)
    _make_quad(
        name="official_sample_backplate",
        vertices=[
            (8.0, -half, camera_height - half),
            (8.0, half, camera_height - half),
            (8.0, half, camera_height + half),
            (8.0, -half, camera_height + half),
        ],
        material=backplate_material,
    )
    _make_quad(
        name="review_floor",
        vertices=[(0.5, -3.2, 0.0), (8.0, -3.2, 0.0), (8.0, 3.2, 0.0), (0.5, 3.2, 0.0)],
        material=floor_material,
    )
    _make_quad(
        name="review_left_wall",
        vertices=[(0.5, -3.2, 0.0), (8.0, -3.2, 0.0), (8.0, -3.2, 3.4), (0.5, -3.2, 3.4)],
        material=side_material,
    )
    _make_quad(
        name="review_right_wall",
        vertices=[(8.0, 3.2, 0.0), (0.5, 3.2, 0.0), (0.5, 3.2, 3.4), (8.0, 3.2, 3.4)],
        material=side_material,
    )

    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    if output_usd is not None:
        bpy.ops.wm.usd_export(
            filepath=str(output_usd),
            selected_objects_only=False,
            export_animation=False,
            export_materials=True,
            export_textures=False,
            relative_paths=False,
            convert_orientation=True,
            export_global_forward_selection="X",
            export_global_up_selection="Z",
            evaluation_mode="RENDER",
        )
    if output_glb is not None:
        # The UE USD stage loader creates texture/material assets in a transient
        # cache.  Those assets are not guaranteed to survive a saved-map reload
        # in game mode.  GLB embeds the official sample image and lets the UE
        # editor import persistent StaticMesh/Material/Texture assets instead.
        bpy.ops.export_scene.gltf(
            filepath=str(output_glb),
            export_format="GLB",
            use_selection=False,
            export_yup=True,
            export_materials="EXPORT",
            export_cameras=False,
            export_lights=False,
            export_animations=False,
        )

    metadata = {
        "schema": SCENE_METADATA_SCHEMA,
        "dataset_id": "3D-FRONT-FUTURE/official-toolbox-sample",
        "scene_id": "3d_front_official_toolbox_sample_proxy",
        "room_id": "3d_front_official_toolbox_sample_proxy_room",
        "room_type": "living_room_proxy",
        "room_polygon_xy_m": [[0.5, -3.2], [8.0, -3.2], [8.0, 3.2], [0.5, 3.2]],
        "floor_z_m": 0.0,
        "objects": objects,
        "source_reference": str(root),
        "official_sample_fov_deg": fov,
        "official_sample_camera_xyz_m": [0.0, 0.0, camera_height],
        "claim_boundary": (
            "official 3D-FUTURE Toolbox five-object sample proxy with a generated "
            "review shell and official raw-scene projection; not a complete "
            "3D-FRONT house, room JSON, texture library or dataset qualification"
        ),
        "redistribution": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if output_usd is not None:
        print(output_usd)
    if output_glb is not None:
        print(output_glb)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
