"""Build the controlled two-zone Blender room used by the M1 canary.

Run with Blender, not the system Python:

    blender --background --python tools/blender/build_custom_room.py -- \
      --output-dir examples/rooms/blender_custom/visual

The helper accepts coordinates in the Habitat convention (+Y up, -Z camera
forward) and maps them to Blender before glTF export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(args)


def habitat_to_blender(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y_up, z = vector
    return (x, -z, y_up)


def material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    result = bpy.data.materials.new(name=name)
    result.diffuse_color = color
    result.use_nodes = True
    principled = result.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.75
    return result


def add_box(
    name: str,
    center_habitat: tuple[float, float, float],
    size_habitat: tuple[float, float, float],
    surface: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=habitat_to_blender(center_habitat))
    obj = bpy.context.object
    obj.name = name
    sx, sy, sz = habitat_to_blender(size_habitat)
    obj.scale = (abs(sx) / 2.0, abs(sy) / 2.0, abs(sz) / 2.0)
    obj.data.materials.append(surface)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def add_marker(name: str, color: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2, radius=0.3, location=(0, 0, 0)
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(color)
    return obj


def export_selected(path: Path, objects: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_materials="EXPORT",
        export_cameras=False,
        export_lights=False,
        export_animations=False,
    )


def triangle_count(objects: list[bpy.types.Object]) -> int:
    total = 0
    for obj in objects:
        obj.data.calc_loop_triangles()
        total += len(obj.data.loop_triangles)
    return total


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    stages = output / "stages"
    objects_dir = output / "objects"
    stages.mkdir(parents=True, exist_ok=True)
    objects_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    floor_material = material("FloorWarmGray", (0.32, 0.29, 0.25, 1.0))
    wall_material = material("WallSoftBlue", (0.58, 0.68, 0.78, 1.0))
    ceiling_material = material("CeilingOffWhite", (0.84, 0.84, 0.78, 1.0))
    accent_material = material("DoorFrameAccent", (0.18, 0.23, 0.28, 1.0))
    source0_material = material("Source0Orange", (0.95, 0.22, 0.05, 1.0))
    source1_material = material("Source1Green", (0.05, 0.75, 0.22, 1.0))

    stage: list[bpy.types.Object] = []
    stage.append(add_box("Floor", (0.0, -0.1, 0.0), (8.0, 0.2, 6.0), floor_material))
    stage.append(add_box("Ceiling", (0.0, 3.1, 0.0), (8.0, 0.2, 6.0), ceiling_material))
    stage.append(add_box("WestWall", (-4.1, 1.5, 0.0), (0.2, 3.0, 6.0), wall_material))
    stage.append(add_box("EastWall", (4.1, 1.5, 0.0), (0.2, 3.0, 6.0), wall_material))
    stage.append(add_box("SouthWall", (0.0, 1.5, 3.1), (8.0, 3.0, 0.2), wall_material))

    # North wall with a real 1.5 m x 1.3 m window opening centered at x=2.
    stage.append(
        add_box("NorthWallLeft", (-1.375, 1.5, -3.1), (5.25, 3.0, 0.2), wall_material)
    )
    stage.append(
        add_box("NorthWallRight", (3.375, 1.5, -3.1), (1.25, 3.0, 0.2), wall_material)
    )
    stage.append(
        add_box("WindowSill", (2.0, 0.4, -3.1), (1.5, 0.8, 0.2), accent_material)
    )
    stage.append(
        add_box("WindowLintel", (2.0, 2.55, -3.1), (1.5, 0.9, 0.2), accent_material)
    )

    # Partition at x=0 with a floor-to-2.1 m door opening, 1.2 m wide in Z.
    stage.append(
        add_box("PartitionNorth", (0.0, 1.5, -1.8), (0.2, 3.0, 2.4), wall_material)
    )
    stage.append(
        add_box("PartitionSouth", (0.0, 1.5, 1.8), (0.2, 3.0, 2.4), wall_material)
    )
    stage.append(
        add_box("DoorLintel", (0.0, 2.55, 0.0), (0.2, 0.9, 1.2), accent_material)
    )

    source0 = add_marker("SourceMarker0", source0_material)
    source1 = add_marker("SourceMarker1", source1_material)

    stage_path = stages / "m1_custom_room.glb"
    source0_path = objects_dir / "source_marker_0.glb"
    source1_path = objects_dir / "source_marker_1.glb"
    export_selected(stage_path, stage)
    export_selected(source0_path, [source0])
    export_selected(source1_path, [source1])

    report = {
        "schema": "avengine_blender_room_build_report_v1",
        "blender_version": bpy.app.version_string,
        "coordinate_mapping": "Habitat (X,Y,Z) -> Blender (X,-Z,Y); glTF export restores +Y up",
        "geometry_representation": "real_surface_mesh",
        "openings": ["door_main", "window_north"],
        "stage_object_count": len(stage),
        "stage_triangle_count": triangle_count(stage),
        "outputs": {
            path.relative_to(output).as_posix(): {
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (stage_path, source0_path, source1_path)
        },
    }
    with (output / "build_report.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
