#!/usr/bin/env python3
"""Build one data-driven household room with semantic furnishing assemblies."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import bpy
from mathutils import Vector


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))
from build_compact_household import (  # noqa: E402
    add_area,
    aim,
    box,
    cylinder,
    emit_architecture,
)


TEXTURE_ROOT = Path(
    os.environ.get(
        "AVENGINE_AUTHORING_TEXTURE_ROOT",
        "/data/jzy/blender_projects/chef_home_test_kitchen_vfx_v1/assets/source_4k",
    )
)
DEFAULT_WALL_MOUNT_CLEARANCE_M = 0.01
MINIMUM_WALL_MOUNT_CLEARANCE_M = 0.005
DEFAULT_WALL_SEARCH_DISTANCE_M = 1.5
WALL_MOUNT_OMISSIONS: list[dict[str, Any]] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_image(path: Path, *, noncolor: bool = False) -> bpy.types.Image:
    require(path.is_file(), f"missing texture: {path}")
    value = bpy.data.images.load(str(path), check_existing=True)
    if noncolor:
        value.colorspace_settings.name = "Non-Color"
    return value


def pbr_material(
    name: str,
    *,
    diffuse: Path | None,
    base_color: tuple[float, float, float, float],
    roughness: Path | None = None,
    normal: Path | None = None,
    normal_strength: float = 0.34,
    roughness_default: float = 0.5,
    metallic: float = 0.0,
    transmission: float = 0.0,
    emission: tuple[float, float, float, float] | None = None,
    emission_strength: float = 2.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = base_color
    shader.inputs["Roughness"].default_value = roughness_default
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Transmission Weight"].default_value = transmission
    if emission is not None:
        shader.inputs["Emission Color"].default_value = emission
        shader.inputs["Emission Strength"].default_value = emission_strength
    if diffuse is not None:
        node = nodes.new("ShaderNodeTexImage")
        node.image = load_image(diffuse)
        node.extension = "REPEAT"
        links.new(node.outputs["Color"], shader.inputs["Base Color"])
    if roughness is not None:
        node = nodes.new("ShaderNodeTexImage")
        node.image = load_image(roughness, noncolor=True)
        node.extension = "REPEAT"
        links.new(node.outputs["Color"], shader.inputs["Roughness"])
    if normal is not None:
        node = nodes.new("ShaderNodeTexImage")
        node.image = load_image(normal, noncolor=True)
        node.extension = "REPEAT"
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = normal_strength
        links.new(node.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return mat


STYLE_PALETTES = {
    "warm_scandinavian": {
        "cabinet": (0.48, 0.56, 0.46, 1.0),
        "cabinet_panel": (0.37, 0.47, 0.38, 1.0),
        "fabric": (0.42, 0.24, 0.14, 1.0),
        "dark_fabric": (0.055, 0.09, 0.15, 1.0),
        "rug": (0.32, 0.20, 0.14, 1.0),
        "wall": (0.68, 0.63, 0.56, 1.0),
    },
    "japanese_walnut": {
        "cabinet": (0.23, 0.13, 0.07, 1.0),
        "cabinet_panel": (0.16, 0.085, 0.04, 1.0),
        "fabric": (0.36, 0.25, 0.15, 1.0),
        "dark_fabric": (0.10, 0.13, 0.12, 1.0),
        "rug": (0.43, 0.38, 0.28, 1.0),
        "wall": (0.62, 0.58, 0.50, 1.0),
    },
    "urban_contemporary": {
        "cabinet": (0.12, 0.15, 0.16, 1.0),
        "cabinet_panel": (0.075, 0.09, 0.10, 1.0),
        "fabric": (0.45, 0.16, 0.07, 1.0),
        "dark_fabric": (0.08, 0.09, 0.11, 1.0),
        "rug": (0.18, 0.20, 0.22, 1.0),
        "wall": (0.52, 0.51, 0.48, 1.0),
    },
    "soft_contemporary": {
        "cabinet": (0.43, 0.50, 0.46, 1.0),
        "cabinet_panel": (0.31, 0.39, 0.35, 1.0),
        "fabric": (0.52, 0.32, 0.22, 1.0),
        "dark_fabric": (0.11, 0.16, 0.18, 1.0),
        "rug": (0.42, 0.36, 0.29, 1.0),
        "wall": (0.72, 0.68, 0.61, 1.0),
    },
}


def materials_for_style(style_id: str) -> dict[str, bpy.types.Material]:
    palette = STYLE_PALETTES.get(style_id)
    require(palette is not None, f"unsupported style: {style_id}")
    plaster_rough = TEXTURE_ROOT / "white_plaster_02/white_plaster_02_rough_4k.jpg"
    plaster_normal = TEXTURE_ROOT / "white_plaster_02/white_plaster_02_nor_gl_4k.exr"
    wood_diff = TEXTURE_ROOT / "lacquered_cherry_wood/lacquered_cherry_wood_diff_4k.jpg"
    wood_rough = TEXTURE_ROOT / "lacquered_cherry_wood/lacquered_cherry_wood_rough_4k.exr"
    wood_normal = TEXTURE_ROOT / "lacquered_cherry_wood/lacquered_cherry_wood_nor_gl_4k.exr"
    marble_diff = TEXTURE_ROOT / "marble_01/marble_01_diff_4k.jpg"
    marble_rough = TEXTURE_ROOT / "marble_01/marble_01_rough_4k.jpg"
    marble_normal = TEXTURE_ROOT / "marble_01/marble_01_nor_gl_4k.exr"
    tile_diff = TEXTURE_ROOT / "terrazzo_tiles/terrazzo_tiles_diff_4k.jpg"
    tile_rough = TEXTURE_ROOT / "terrazzo_tiles/terrazzo_tiles_rough_4k.exr"
    tile_normal = TEXTURE_ROOT / "terrazzo_tiles/terrazzo_tiles_nor_gl_4k.exr"

    def solid(
        name: str,
        color: tuple[float, float, float, float],
        rough: float,
        metal: float = 0.0,
        transmission: float = 0.0,
        emission: tuple[float, float, float, float] | None = None,
        emission_strength: float = 2.0,
    ) -> bpy.types.Material:
        return pbr_material(
            name,
            diffuse=None,
            base_color=color,
            roughness_default=rough,
            metallic=metal,
            transmission=transmission,
            emission=emission,
            emission_strength=emission_strength,
        )

    mats = {
        "wall": pbr_material("WarmPlaster", diffuse=None, base_color=palette["wall"], roughness=plaster_rough, normal=plaster_normal, roughness_default=0.76),
        "ceiling": solid(
            "SoftCeiling", (0.78, 0.75, 0.70, 1.0), 0.82,
            emission=(0.78, 0.75, 0.70, 1.0), emission_strength=0.22,
        ),
        "wood_floor": pbr_material("OakFloor", diffuse=wood_diff, base_color=(0.6, 0.6, 0.6, 1.0), roughness=wood_rough, normal=wood_normal, roughness_default=0.44),
        "entry_floor": pbr_material("EntryStoneFloor", diffuse=tile_diff, base_color=(0.7, 0.7, 0.7, 1.0), roughness=tile_rough, normal=tile_normal, roughness_default=0.58),
        "ground": solid("ExteriorPaving", (0.16, 0.18, 0.20, 1.0), 0.78),
        "glass": solid("WindowGlass", (0.66, 0.80, 0.90, 1.0), 0.08, transmission=0.78),
        "light_wood": pbr_material("LightOak", diffuse=wood_diff, base_color=(0.7, 0.7, 0.7, 1.0), roughness=wood_rough, normal=wood_normal, roughness_default=0.42),
        "wood": pbr_material("DarkOak", diffuse=wood_diff, base_color=(0.5, 0.5, 0.5, 1.0), roughness=wood_rough, normal=wood_normal, roughness_default=0.38),
        "cabinet": solid("Cabinet", palette["cabinet"], 0.33),
        "cabinet_panel": solid("CabinetPanel", palette["cabinet_panel"], 0.35),
        "stone": pbr_material("HonedStone", diffuse=marble_diff, base_color=(0.75, 0.75, 0.75, 1.0), roughness=marble_rough, normal=marble_normal, roughness_default=0.29),
        "tile": pbr_material(
            "BacksplashTile",
            # A broad marble slab stays legible at 1080p.  The former tightly
            # repeated terrazzo normal/diffuse maps produced sub-pixel sparkle
            # in UE even after the coplanar geometry was repaired.
            diffuse=marble_diff,
            base_color=(0.75, 0.75, 0.75, 1.0),
            roughness=marble_rough,
            normal=None,
            roughness_default=0.40,
        ),
        "steel": solid("BrushedSteel", (0.42, 0.46, 0.50, 1.0), 0.24, 0.92),
        "dark_metal": solid("DarkMetal", (0.02, 0.024, 0.028, 1.0), 0.30, 0.82),
        "brass": solid("AgedBrass", (0.42, 0.20, 0.055, 1.0), 0.27, 0.86),
        "fabric": solid("WarmFabric", palette["fabric"], 0.76),
        "dark_fabric": solid("DeepFabric", palette["dark_fabric"], 0.82),
        "rug": solid("Rug", palette["rug"], 0.91),
        "dog_bed": solid("DogBed", (0.16, 0.28, 0.23, 1.0), 0.88),
        "ceramic": solid("Ceramic", (0.90, 0.87, 0.80, 1.0), 0.18),
        "porcelain": solid("Porcelain", (0.82, 0.84, 0.82, 1.0), 0.34),
        "bedding": solid("Bedding", (0.82, 0.78, 0.70, 1.0), 0.86),
        "bedding_accent": solid("BeddingAccent", (0.26, 0.42, 0.47, 1.0), 0.84),
        "curtain": solid("Curtain", (0.72, 0.68, 0.61, 1.0), 0.92),
        "mirror": solid("Mirror", (0.72, 0.82, 0.88, 1.0), 0.04, 0.72),
        "screen": solid("Screen", (0.008, 0.012, 0.016, 1.0), 0.12),
        "appliance": solid("ApplianceWhite", (0.72, 0.76, 0.76, 1.0), 0.28, 0.38),
        "towel": solid("Towel", (0.56, 0.64, 0.63, 1.0), 0.96),
        "paper": solid("Paper", (0.82, 0.78, 0.66, 1.0), 0.88),
        "book_warm": solid("BookWarm", (0.50, 0.14, 0.055, 1.0), 0.62),
        "book_cool": solid("BookCool", (0.08, 0.22, 0.32, 1.0), 0.62),
        "warm_light": solid("WarmEmitter", (0.9, 0.55, 0.20, 1.0), 0.18, emission=(1.0, 0.42, 0.12, 1.0)),
        "green": solid("PlantGreen", (0.04, 0.20, 0.07, 1.0), 0.82),
        "fruit_red": solid("FruitRed", (0.48, 0.018, 0.012, 1.0), 0.28),
        "fruit_yellow": solid("FruitYellow", (0.62, 0.25, 0.015, 1.0), 0.30),
    }
    return mats


def world_point(center: tuple[float, float], offset: tuple[float, float], yaw: float) -> tuple[float, float]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (
        center[0] + cosine * offset[0] - sine * offset[1],
        center[1] + sine * offset[0] + cosine * offset[1],
    )


def oriented_box(
    name: str,
    center_xy: tuple[float, float],
    z: float,
    dimensions: tuple[float, float, float],
    yaw: float,
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
    *,
    bevel: float = 0.02,
    offset_xy: tuple[float, float] = (0.0, 0.0),
) -> bpy.types.Object:
    x, y = world_point(center_xy, offset_xy, yaw)
    obj = box(name, (x, y, z), dimensions, mat, collection, bevel)
    obj.rotation_euler.z = yaw
    return obj


def wall_mounted_box(
    name: str,
    anchor_xy: tuple[float, float],
    z: float,
    dimensions: tuple[float, float, float],
    yaw: float,
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
    *,
    clearance_m: float = DEFAULT_WALL_MOUNT_CLEARANCE_M,
    search_distance_m: float = DEFAULT_WALL_SEARCH_DISTANCE_M,
    bevel: float = 0.0,
    omit_at_opening: bool = True,
) -> bpy.types.Object | None:
    """Mount a thin box in front of the nearest backing wall.

    Furnishing assemblies use local +Y as their room-facing direction.  A
    ray cast in local -Y resolves the actual architecture surface instead of
    deriving it from cabinet depth.  The panel's rear face is then placed at
    ``clearance_m`` in front of that surface.  This avoids same-facing
    coplanar faces, which produce depth-buffer flicker after USD import.
    """

    width, depth, height = (float(value) for value in dimensions)
    require(width > 0.0 and depth > 0.0 and height > 0.0, f"{name} dimensions must be positive")
    require(
        clearance_m >= MINIMUM_WALL_MOUNT_CLEARANCE_M,
        f"{name} wall clearance {clearance_m} is below "
        f"{MINIMUM_WALL_MOUNT_CLEARANCE_M} m",
    )
    require(search_distance_m > 0.0, f"{name} wall search distance must be positive")

    interior = Vector((-math.sin(yaw), math.cos(yaw), 0.0))
    origin = Vector((float(anchor_xy[0]), float(anchor_xy[1]), float(z)))
    bpy.context.view_layer.update()
    hit, location, normal, _, backing, _ = bpy.context.scene.ray_cast(
        bpy.context.evaluated_depsgraph_get(),
        origin,
        -interior,
        distance=search_distance_m,
    )
    require(hit and backing is not None, f"{name} could not resolve a backing wall")
    if omit_at_opening and backing.name.startswith("Glass_"):
        WALL_MOUNT_OMISSIONS.append(
            {
                "object": name,
                "reason": "backing_surface_is_an_architectural_opening",
                "observed_object": backing.name,
                "status": "omitted",
            }
        )
        return None
    require(
        backing.name.startswith("Wall_"),
        f"{name} backing ray hit {backing.name!r}, not architecture",
    )
    require(
        Vector(normal).normalized().dot(interior) >= 0.95,
        f"{name} backing wall normal does not face the room",
    )

    center = Vector(location) + interior * (clearance_m + depth / 2.0)
    obj = box(
        name,
        (float(center.x), float(center.y), float(z)),
        (width, depth, height),
        mat,
        collection,
        bevel,
    )
    obj.rotation_euler.z = yaw
    obj["avengine_mount_kind"] = "wall_surface"
    obj["avengine_backing_object"] = backing.name
    obj["avengine_surface_clearance_m"] = float(clearance_m)
    obj["avengine_surface_depth_m"] = depth
    obj["avengine_interior_normal_xy"] = [float(interior.x), float(interior.y)]
    obj["avengine_backing_point_xyz_m"] = [
        float(location.x),
        float(location.y),
        float(location.z),
    ]
    return obj


def validate_wall_mounts(scene: bpy.types.Scene) -> list[dict[str, Any]]:
    """Verify authored wall-mounted surfaces retain a visible depth gap."""

    records: list[dict[str, Any]] = []
    bpy.context.view_layer.update()
    for obj in sorted(scene.objects, key=lambda value: value.name):
        if obj.get("avengine_mount_kind") != "wall_surface":
            continue
        interior_xy = obj["avengine_interior_normal_xy"]
        interior = Vector((float(interior_xy[0]), float(interior_xy[1]), 0.0))
        backing_point = Vector(obj["avengine_backing_point_xyz_m"])
        depth = float(obj["avengine_surface_depth_m"])
        rear_face = obj.matrix_world.translation - interior * (depth / 2.0)
        observed = float((rear_face - backing_point).dot(interior))
        declared = float(obj["avengine_surface_clearance_m"])
        require(
            observed >= MINIMUM_WALL_MOUNT_CLEARANCE_M - 1.0e-6,
            f"{obj.name} rear face is coplanar with or behind its backing surface: "
            f"observed={observed:.9f} declared={declared:.9f}",
        )
        require(
            abs(observed - declared) <= 1.0e-6,
            f"{obj.name} wall clearance changed after placement",
        )
        records.append(
            {
                "object": obj.name,
                "backing_object": str(obj["avengine_backing_object"]),
                "declared_clearance_m": declared,
                "observed_clearance_m": observed,
                "status": "pass",
            }
        )
    return records


def _rotate_xy(vector: tuple[float, float], angle: float) -> tuple[float, float]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
    )


def _opening_frame(
    *,
    name: str,
    segment: dict[str, Any],
    opening: dict[str, Any],
    collection: bpy.types.Collection,
    mats: dict[str, bpy.types.Material],
) -> None:
    """Add visible trim around one wall opening without changing the wall."""

    coordinate = float(segment["coordinate"])
    center = float(opening["center"])
    width = float(opening["width"])
    sill = float(opening["sill"])
    head = float(opening["head"])
    trim = 0.065
    depth = 0.055
    vertical_height = max(0.1, head - sill)
    if segment["axis"] == "h":
        for side in (-1.0, 1.0):
            box(
                name + f"_Jamb_{side:+.0f}",
                (center + side * (width / 2.0 - trim / 2.0), coordinate, (sill + head) / 2.0),
                (trim, depth, vertical_height),
                mats["light_wood"], collection, 0.008,
            )
        box(
            name + "_HeadTrim", (center, coordinate, head - trim / 2.0),
            (width, depth, trim), mats["light_wood"], collection, 0.008,
        )
        if sill > 0.0:
            box(
                name + "_SillTrim", (center, coordinate, sill + trim / 2.0),
                (width, depth + 0.10, trim), mats["light_wood"], collection, 0.008,
            )
    else:
        for side in (-1.0, 1.0):
            box(
                name + f"_Jamb_{side:+.0f}",
                (coordinate, center + side * (width / 2.0 - trim / 2.0), (sill + head) / 2.0),
                (depth, trim, vertical_height),
                mats["light_wood"], collection, 0.008,
            )
        box(
            name + "_HeadTrim", (coordinate, center, head - trim / 2.0),
            (depth, width, trim), mats["light_wood"], collection, 0.008,
        )
        if sill > 0.0:
            box(
                name + "_SillTrim", (coordinate, center, sill + trim / 2.0),
                (depth + 0.10, width, trim), mats["light_wood"], collection, 0.008,
            )


def _door_leaf(
    *,
    segment: dict[str, Any],
    opening: dict[str, Any],
    collection: bpy.types.Collection,
    mats: dict[str, bpy.types.Material],
) -> dict[str, Any] | None:
    config = opening.get("door_leaf")
    if not isinstance(config, dict):
        return None
    width = float(opening["width"]) - 0.06
    height = min(float(opening["head"]) - 0.06, float(config.get("height_m", 2.08)))
    angle = math.radians(float(config.get("angle_deg", 70.0)))
    angle *= 1.0 if float(config.get("swing_sign", 1.0)) >= 0.0 else -1.0
    along = (1.0, 0.0) if segment["axis"] == "h" else (0.0, 1.0)
    center_point = (
        (float(opening["center"]), float(segment["coordinate"]))
        if segment["axis"] == "h"
        else (float(segment["coordinate"]), float(opening["center"]))
    )
    hinge_sign = -1.0 if config.get("hinge", "start") == "start" else 1.0
    hinge = (
        center_point[0] + along[0] * hinge_sign * width / 2.0,
        center_point[1] + along[1] * hinge_sign * width / 2.0,
    )
    leaf_direction = _rotate_xy(along, angle)
    if hinge_sign > 0.0:
        leaf_direction = (-leaf_direction[0], -leaf_direction[1])
    leaf_center = (
        hinge[0] + leaf_direction[0] * width / 2.0,
        hinge[1] + leaf_direction[1] * width / 2.0,
    )
    leaf_yaw = math.atan2(leaf_direction[1], leaf_direction[0])
    leaf = oriented_box(
        "DoorLeaf_" + str(opening["id"]), leaf_center, height / 2.0 + 0.03,
        (width, 0.045, height), leaf_yaw, mats["wood"], collection, bevel=0.015,
    )
    handle_offset = 0.36 * width
    handle_xy = world_point(leaf_center, (handle_offset, -0.05), leaf_yaw)
    handle = cylinder(
        "DoorHandle_" + str(opening["id"]),
        (handle_xy[0], handle_xy[1], 1.02), 0.025, 0.13,
        mats["brass"], collection, vertices=24,
    )
    handle.rotation_euler.x = math.pi / 2.0
    return {
        "opening_id": str(opening["id"]),
        "angle_deg": math.degrees(angle),
        "object": leaf.name,
    }


def _window_details(
    *,
    segment: dict[str, Any],
    opening: dict[str, Any],
    collection: bpy.types.Collection,
    mats: dict[str, bpy.types.Material],
) -> None:
    config = opening.get("window_detail")
    if config is False:
        return
    _opening_frame(
        name="WindowFrame_" + str(opening["id"]), segment=segment,
        opening=opening, collection=collection, mats=mats,
    )
    center = float(opening["center"])
    coordinate = float(segment["coordinate"])
    sill = float(opening["sill"])
    head = float(opening["head"])
    width = float(opening["width"])
    if segment["axis"] == "h":
        box(
            "WindowMullion_" + str(opening["id"]),
            (center, coordinate, (sill + head) / 2.0),
            (0.045, 0.060, head - sill - 0.10),
            mats["light_wood"], collection, 0.006,
        )
    else:
        box(
            "WindowMullion_" + str(opening["id"]),
            (coordinate, center, (sill + head) / 2.0),
            (0.060, 0.045, head - sill - 0.10),
            mats["light_wood"], collection, 0.006,
        )
    if not isinstance(config, dict) or not config.get("curtains", False):
        return
    if segment["axis"] == "h":
        inward = 1.0 if segment["hi"] else -1.0
        curtain_y = coordinate + inward * 0.11
        for side in (-1.0, 1.0):
            box(
                "Curtain_" + str(opening["id"]) + f"_{side:+.0f}",
                (center + side * width * 0.37, curtain_y, (sill + head) / 2.0),
                (width * 0.22, 0.035, head - sill + 0.18),
                mats["curtain"], collection, 0.025,
            )
    else:
        inward = 1.0 if segment["hi"] else -1.0
        curtain_x = coordinate + inward * 0.11
        for side in (-1.0, 1.0):
            box(
                "Curtain_" + str(opening["id"]) + f"_{side:+.0f}",
                (curtain_x, center + side * width * 0.37, (sill + head) / 2.0),
                (0.035, width * 0.22, head - sill + 0.18),
                mats["curtain"], collection, 0.025,
            )


def emit_architecture_details(
    segments: list[dict[str, Any]],
    collections: dict[str, bpy.types.Collection],
    mats: dict[str, bpy.types.Material],
) -> dict[str, Any]:
    collection = collections["ArchitecturalDetails"]
    doors: list[dict[str, Any]] = []
    windows = 0
    for segment in segments:
        for opening in segment["openings"]:
            if opening["kind"] == "window":
                _window_details(
                    segment=segment, opening=opening, collection=collection, mats=mats
                )
                windows += 1
            else:
                _opening_frame(
                    name="DoorFrame_" + str(opening["id"]), segment=segment,
                    opening=opening, collection=collection, mats=mats,
                )
                record = _door_leaf(
                    segment=segment, opening=opening,
                    collection=collection, mats=mats,
                )
                if record is not None:
                    doors.append(record)
    return {"door_leaves": doors, "window_detail_count": windows}


def projected_uvs(obj: bpy.types.Object, tile_m: float) -> None:
    if obj.type != "MESH":
        return
    mesh = obj.data
    uv = mesh.uv_layers.active or mesh.uv_layers.new(name="SemanticRoomUV")
    for polygon in mesh.polygons:
        axis = max(range(3), key=lambda item: abs(polygon.normal[item]))
        for loop_index in polygon.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            if axis == 2:
                values = (co.x, co.y)
            elif axis == 1:
                values = (co.x, co.z)
            else:
                values = (co.y, co.z)
            uv.data[loop_index].uv = (values[0] / tile_m, values[1] / tile_m)


def add_chair(
    name: str,
    center: tuple[float, float],
    yaw: float,
    mats: dict[str, bpy.types.Material],
    collection: bpy.types.Collection,
) -> None:
    oriented_box(name + "_Seat", center, 0.48, (0.52, 0.50, 0.10), yaw, mats["fabric"], collection, bevel=0.045)
    oriented_box(name + "_Back", center, 0.87, (0.52, 0.10, 0.72), yaw, mats["fabric"], collection, bevel=0.045, offset_xy=(0.0, 0.21))
    for dx in (-0.20, 0.20):
        for dy in (-0.18, 0.18):
            oriented_box(name + f"_Leg_{dx:+.2f}_{dy:+.2f}", center, 0.23, (0.055, 0.055, 0.46), yaw, mats["dark_metal"], collection, bevel=0.012, offset_xy=(dx, dy))


def emit_entry(plan: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> None:
    center = tuple(plan["bench_center_xy_m"])
    yaw = math.radians(float(plan.get("yaw_deg", 0.0)))
    furniture, props = collections["Furniture"], collections["Props"]
    oriented_box("EntryBench", center, 0.43, (1.10, 0.48, 0.12), yaw, mats["light_wood"], furniture, bevel=0.045)
    for dx in (-0.42, 0.42):
        oriented_box(
            f"EntryBenchLeg_{dx:+.2f}",
            center,
            0.21,
            (0.08, 0.36, 0.42),
            yaw,
            mats["dark_metal"],
            furniture,
            bevel=0.012,
            offset_xy=(dx, 0.0),
        )
    for index, dx in enumerate((-0.38, 0.0, 0.38)):
        oriented_box(f"EntryShoe_{index}", center, 0.18, (0.30, 0.40, 0.18), yaw, mats["dark_fabric"], props, bevel=0.04, offset_xy=(dx, 0.0))
    slat_center = tuple(plan["slat_center_xy_m"])
    oriented_box("EntrySlatWall", slat_center, 1.45, (1.45, 0.10, 2.25), yaw, mats["light_wood"], furniture, bevel=0.018)
    for index, z in enumerate((1.15, 1.45, 1.75)):
        x, y = world_point(slat_center, (-0.35 + 0.35 * index, 0.08), yaw)
        cylinder(f"CoatHook_{index}", (x, y, z), 0.035, 0.16, mats["brass"], props)
    if "mirror_center_xy_m" in plan:
        mirror_center = tuple(plan["mirror_center_xy_m"])
        mirror_yaw = math.radians(float(plan.get("mirror_yaw_deg", plan.get("yaw_deg", 0.0))))
        oriented_box(
            "EntryMirror", mirror_center, 1.45, (0.72, 0.035, 1.45),
            mirror_yaw, mats["mirror"], props, bevel=0.018,
        )
    shelf = tuple(plan.get("drop_shelf_center_xy_m", center))
    oriented_box(
        "EntryDropShelf", shelf, 0.91, (0.72, 0.28, 0.08), yaw,
        mats["light_wood"], furniture, bevel=0.025,
    )
    oriented_box(
        "EntryKeyTray", shelf, 0.975, (0.24, 0.15, 0.035), yaw,
        mats["ceramic"], props, bevel=0.018, offset_xy=(0.16, 0.0),
    )
    umbrella_xy = world_point(center, (0.55, 0.18), yaw)
    cylinder(
        "EntryUmbrellaStand", (umbrella_xy[0], umbrella_xy[1], 0.28),
        0.14, 0.55, mats["dark_metal"], props, vertices=32,
    )


def emit_kitchen(plan: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> None:
    furniture, props = collections["Furniture"], collections["Props"]
    center = tuple(plan["run_center_xy_m"])
    yaw = math.radians(float(plan.get("yaw_deg", 0.0)))
    length = float(plan["run_length_m"])
    oriented_box("KitchenLowerRun", center, 0.47, (length, 0.72, 0.94), yaw, mats["cabinet"], furniture, bevel=0.035)
    oriented_box("KitchenCounter", center, 0.99, (length + 0.20, 0.86, 0.09), yaw, mats["stone"], furniture, bevel=0.025, offset_xy=(0.0, 0.07))
    wall_mounted_box(
        "KitchenBacksplash",
        center,
        1.65,
        (length + 0.20, 0.04, 1.25),
        yaw,
        mats["tile"],
        furniture,
        clearance_m=float(
            plan.get("backsplash_wall_clearance_m", DEFAULT_WALL_MOUNT_CLEARANCE_M)
        ),
        search_distance_m=float(
            plan.get("backing_wall_search_distance_m", DEFAULT_WALL_SEARCH_DISTANCE_M)
        ),
        bevel=0.005,
    )
    panel_count = max(3, int(plan.get("panel_count", round(length / 0.82))))
    panel_width = (length - 0.18 * (panel_count + 1)) / panel_count
    for index in range(panel_count):
        local_x = -length / 2 + 0.18 + panel_width / 2 + index * (panel_width + 0.18)
        oriented_box(f"KitchenPanel_{index}", center, 0.51, (panel_width, 0.035, 0.72), yaw, mats["cabinet_panel"], props, bevel=0.012, offset_xy=(local_x, 0.38))
        oriented_box(f"KitchenPull_{index}", center, 0.78, (0.20, 0.025, 0.024), yaw, mats["brass"], props, bevel=0.005, offset_xy=(local_x, 0.415))
    oven_x = float(plan.get("oven_local_x_m", 0.0))
    oriented_box("KitchenOven", center, 0.48, (0.72, 0.08, 0.66), yaw, mats["dark_metal"], props, bevel=0.025, offset_xy=(oven_x, 0.42))
    oriented_box("KitchenOvenHandle", center, 0.72, (0.48, 0.035, 0.035), yaw, mats["steel"], props, bevel=0.01, offset_xy=(oven_x, 0.48))
    fridge_x = float(plan.get("fridge_local_x_m", -length / 2 + 0.52))
    oriented_box("Fridge", center, 1.36, (1.05, 0.82, 2.72), yaw, mats["steel"], furniture, bevel=0.055, offset_xy=(fridge_x, 0.10))
    sink_x = float(plan.get("sink_local_x_m", length / 2 - 0.80))
    oriented_box("SinkBasin", center, 1.03, (0.92, 0.54, 0.12), yaw, mats["dark_metal"], props, bevel=0.055, offset_xy=(sink_x, 0.10))
    stem_xy = world_point(center, (sink_x + 0.35, -0.03), yaw)
    cylinder("FaucetStem", (stem_xy[0], stem_xy[1], 1.38), 0.025, 0.68, mats["steel"], props)
    oriented_box("FaucetSpout", center, 1.67, (0.38, 0.045, 0.045), yaw, mats["steel"], props, bevel=0.012, offset_xy=(sink_x + 0.18, 0.02))
    cooktop_x = float(plan.get("cooktop_local_x_m", 0.15 * length))
    oriented_box(
        "KitchenCooktop", center, 1.055, (0.72, 0.48, 0.025), yaw,
        mats["screen"], props, bevel=0.018, offset_xy=(cooktop_x, 0.12),
    )
    oriented_box(
        "KitchenHood", center, 2.18, (0.86, 0.42, 0.30), yaw,
        mats["steel"], furniture, bevel=0.045, offset_xy=(cooktop_x, -0.08),
    )
    microwave_x = float(plan.get("microwave_local_x_m", cooktop_x - 0.95))
    oriented_box(
        "KitchenMicrowave", center, 1.33, (0.62, 0.38, 0.36), yaw,
        mats["appliance"], props, bevel=0.045, offset_xy=(microwave_x, 0.08),
    )
    oriented_box(
        "KitchenMicrowaveDoor", center, 1.33, (0.48, 0.025, 0.24), yaw,
        mats["screen"], props, bevel=0.018, offset_xy=(microwave_x, 0.285),
    )
    kettle_x = float(plan.get("kettle_local_x_m", sink_x - 0.72))
    cylinder_xy = world_point(center, (kettle_x, 0.16), yaw)
    cylinder(
        "KitchenKettle", (cylinder_xy[0], cylinder_xy[1], 1.19),
        0.13, 0.27, mats["appliance"], props, vertices=32,
    )
    trash_x = float(plan.get("trash_local_x_m", length / 2 - 0.24))
    oriented_box(
        "KitchenTrashBin", center, 0.33, (0.34, 0.34, 0.66), yaw,
        mats["dark_metal"], props, bevel=0.055, offset_xy=(trash_x, 0.55),
    )
    if "island_center_xy_m" in plan:
        island = tuple(plan["island_center_xy_m"])
        island_yaw = math.radians(float(plan.get("island_yaw_deg", plan.get("yaw_deg", 0.0))))
        island_length = float(plan.get("island_length_m", 2.8))
        island_width = float(plan.get("island_width_m", 1.05))
        oriented_box("KitchenIsland", island, 0.49, (island_length, island_width, 0.98), island_yaw, mats["cabinet"], furniture, bevel=0.035)
        oriented_box("KitchenIslandTop", island, 1.03, (island_length + 0.20, island_width + 0.15, 0.10), island_yaw, mats["stone"], furniture, bevel=0.035)
        count = max(3, round(island_length / 0.65))
        width = (island_length - 0.12 * (count + 1)) / count
        for index in range(count):
            local_x = -island_length / 2 + 0.12 + width / 2 + index * (width + 0.12)
            oriented_box(f"IslandPanel_{index}", island, 0.53, (width, 0.035, 0.72), island_yaw, mats["cabinet_panel"], props, bevel=0.012, offset_xy=(local_x, island_width / 2 + 0.02))
            oriented_box(f"IslandHandle_{index}", island, 0.79, (0.20, 0.025, 0.024), island_yaw, mats["brass"], props, bevel=0.005, offset_xy=(local_x, island_width / 2 + 0.05))
        oriented_box("IslandToeKick", island, 0.105, (island_length - 0.22, 0.055, 0.16), island_yaw, mats["dark_metal"], props, bevel=0.004, offset_xy=(0.0, island_width / 2 + 0.04))
        board_x = -0.25 * island_length
        oriented_box("CuttingBoard", island, 1.095, (0.58, 0.34, 0.026), island_yaw, mats["light_wood"], props, bevel=0.018, offset_xy=(board_x, 0.02))
        for index, (dx, material) in enumerate(((board_x - 0.09, mats["fruit_red"]), (board_x + 0.09, mats["fruit_yellow"]))):
            x, y = world_point(island, (dx, 0.02), island_yaw)
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.07, location=(x, y, 1.19))
            fruit = bpy.context.object
            fruit.name = f"KitchenFruit_{index}"
            fruit.data.materials.append(material)
            for polygon in fruit.data.polygons:
                polygon.use_smooth = True


def emit_dining(plan: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> None:
    furniture, props = collections["Furniture"], collections["Props"]
    center = tuple(plan["center_xy_m"])
    yaw = math.radians(float(plan.get("yaw_deg", 0.0)))
    length = float(plan.get("length_m", 2.1))
    width = float(plan.get("width_m", 1.1))
    oriented_box("DiningTop", center, 0.78, (length, width, 0.10), yaw, mats["light_wood"], furniture, bevel=0.055)
    for dx in (-0.38 * length, 0.38 * length):
        for dy in (-0.34 * width, 0.34 * width):
            oriented_box(f"DiningLeg_{dx:+.2f}_{dy:+.2f}", center, 0.38, (0.09, 0.09, 0.76), yaw, mats["dark_metal"], furniture, bevel=0.016, offset_xy=(dx, dy))
    chair_count = int(plan.get("chair_count", 4))
    if chair_count == 2:
        seats = [(0.0, -0.82 * width, math.pi), (0.0, 0.82 * width, 0.0)]
    elif chair_count == 4:
        seats = [(-0.28 * length, -0.82 * width, math.pi), (0.28 * length, -0.82 * width, math.pi), (-0.28 * length, 0.82 * width, 0.0), (0.28 * length, 0.82 * width, 0.0)]
    else:
        raise ValueError(f"dining chair_count must be 2 or 4, got {chair_count}")
    for index, (dx, dy, local_yaw) in enumerate(seats):
        seat_center = world_point(center, (dx, dy), yaw)
        add_chair(f"DiningChair_{index}", seat_center, yaw + local_yaw, mats, furniture)
        setting_center = world_point(center, (dx, 0.38 * dy / abs(dy)), yaw)
        cylinder(f"DiningPlate_{index}", (setting_center[0], setting_center[1], 0.845), 0.17, 0.025, mats["ceramic"], props)
        glass_center = world_point(center, (dx + 0.12, 0.30 * dy / abs(dy)), yaw)
        cylinder(
            f"DiningGlass_{index}", (glass_center[0], glass_center[1], 0.94),
            0.045, 0.18, mats["glass"], props, vertices=24,
        )
    cylinder("DiningSharedBowl", (center[0], center[1], 0.885), 0.20, 0.10, mats["ceramic"], props)
    oriented_box(
        "DiningNapkin", center, 0.848, (0.24, 0.18, 0.012), yaw,
        mats["towel"], props, bevel=0.006, offset_xy=(0.26, 0.0),
    )


def emit_living(plan: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> None:
    furniture, props = collections["Furniture"], collections["Props"]
    sofa = tuple(plan["sofa_center_xy_m"])
    yaw = math.radians(float(plan.get("sofa_yaw_deg", 0.0)))
    width = float(plan.get("sofa_width_m", 2.8))
    oriented_box("SofaBase", sofa, 0.25, (width, 1.02, 0.40), yaw, mats["dark_fabric"], furniture, bevel=0.10)
    oriented_box("SofaBack", sofa, 0.78, (width, 0.25, 1.00), yaw, mats["dark_fabric"], furniture, bevel=0.10, offset_xy=(0.0, 0.42))
    for side in (-1.0, 1.0):
        oriented_box(
            f"SofaArm_{side:+.0f}", sofa, 0.51, (0.18, 0.94, 0.58), yaw,
            mats["dark_fabric"], furniture, bevel=0.08,
            offset_xy=(side * (width / 2.0 - 0.10), -0.02),
        )
    for index in range(3):
        dx = (-0.30 + 0.30 * index) * width
        oriented_box(f"SofaCushion_{index}", sofa, 0.51, (0.27 * width, 0.70, 0.12), yaw, mats["fabric"], furniture, bevel=0.06, offset_xy=(dx, -0.12))
    rug = tuple(plan["rug_center_xy_m"])
    rug_yaw = math.radians(float(plan.get("rug_yaw_deg", 0.0)))
    oriented_box("LivingRug", rug, 0.025, tuple(plan.get("rug_dimensions_m", [3.2, 2.1, 0.035])), rug_yaw, mats["rug"], furniture, bevel=0.008)
    coffee_table_dimensions = tuple(plan.get("coffee_table_dimensions_m", [1.45, 0.75, 0.10]))
    table_length = float(coffee_table_dimensions[0])
    table_width = float(coffee_table_dimensions[1])
    table_height = float(coffee_table_dimensions[2])
    oriented_box("CoffeeTable", rug, 0.42, coffee_table_dimensions, rug_yaw, mats["light_wood"], furniture, bevel=0.04)
    for dx in (-0.39 * table_length, 0.39 * table_length):
        for dy in (-0.33 * table_width, 0.33 * table_width):
            oriented_box(
                f"CoffeeTableLeg_{dx:+.2f}_{dy:+.2f}",
                rug,
                (0.42 - table_height / 2.0) / 2.0,
                (0.07, 0.07, 0.42 - table_height / 2.0),
                rug_yaw,
                mats["dark_metal"],
                furniture,
                bevel=0.012,
                offset_xy=(dx, dy),
            )
    dog_bed = tuple(plan["dog_bed_center_xy_m"])
    dog_bed_dimensions = tuple(plan.get("dog_bed_dimensions_m", [1.05, 0.78, 0.26]))
    oriented_box(
        "DogBed", dog_bed, float(dog_bed_dimensions[2]) / 2.0,
        dog_bed_dimensions, rug_yaw, mats["dog_bed"], furniture, bevel=0.12,
    )
    plant = tuple(plan.get("plant_center_xy_m", [rug[0] + 1.5, rug[1]]))
    cylinder("LivingPlantPot", (plant[0], plant[1], 0.28), 0.22, 0.55, mats["ceramic"], props)
    for index, dx in enumerate((-0.16, 0.0, 0.16)):
        oriented_box(f"LivingPlantLeaf_{index}", plant, 0.85 + index * 0.10, (0.10, 0.05, 0.82), -0.3 + index * 0.3, mats["green"], props, bevel=0.025, offset_xy=(dx, 0.0))
    if "tv_center_xy_m" in plan:
        television = tuple(plan["tv_center_xy_m"])
        tv_yaw = math.radians(float(plan.get("tv_yaw_deg", 0.0)))
        console_width = float(plan.get("tv_console_width_m", 1.65))
        tv_width = float(plan.get("tv_width_m", 1.45))
        oriented_box(
            "LivingTVConsole", television, 0.37, (console_width, 0.42, 0.58),
            tv_yaw, mats["wood"], furniture, bevel=0.055,
        )
        oriented_box(
            "LivingTV", television, 1.18, (tv_width, 0.10, 0.82),
            tv_yaw, mats["screen"], furniture, bevel=0.035,
            offset_xy=(0.0, 0.20),
        )
    oriented_box(
        "LivingRemote", rug, 0.49, (0.18, 0.055, 0.025), rug_yaw,
        mats["dark_metal"], props, bevel=0.008, offset_xy=(-0.30, 0.04),
    )
    mug_xy = world_point(rug, (0.32, -0.02), rug_yaw)
    cylinder("LivingMug", (mug_xy[0], mug_xy[1], 0.54), 0.055, 0.12, mats["ceramic"], props, vertices=24)
    oriented_box(
        "LivingThrow", sofa, 0.61, (0.62, 0.54, 0.055), yaw,
        mats["bedding_accent"], props, bevel=0.025, offset_xy=(-0.25 * width, -0.18),
    )


def emit_study(plan: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> None:
    furniture, props = collections["Furniture"], collections["Props"]
    center = tuple(plan["desk_center_xy_m"])
    yaw = math.radians(float(plan.get("yaw_deg", 0.0)))
    desk_width = float(plan.get("desk_width_m", 2.0))
    desk_depth = float(plan.get("desk_depth_m", 0.75))
    oriented_box("StudyDesk", center, 0.76, (desk_width, desk_depth, 0.09), yaw, mats["light_wood"], furniture, bevel=0.04)
    for dx in (-0.42 * desk_width, 0.42 * desk_width):
        oriented_box(
            f"StudyDeskLeg_{dx:+.2f}",
            center,
            0.37,
            (0.08, max(0.42, desk_depth - 0.13), 0.74),
            yaw,
            mats["dark_metal"],
            furniture,
            bevel=0.012,
            offset_xy=(dx, 0.0),
        )
    monitor_width = min(0.86, 0.58 * desk_width)
    keyboard_width = min(0.72, 0.52 * desk_width)
    oriented_box("StudyMonitor", center, 1.30, (monitor_width, 0.08, 0.54), yaw, mats["dark_metal"], props, bevel=0.018, offset_xy=(0.0, 0.22 * desk_depth))
    oriented_box("StudyKeyboard", center, 0.84, (keyboard_width, 0.22, 0.035), yaw, mats["dark_metal"], props, bevel=0.008, offset_xy=(0.0, -0.28 * desk_depth))
    chair_center = world_point(
        center, (0.0, float(plan.get("chair_local_y_m", -0.82))), yaw
    )
    add_chair("StudyChair", chair_center, yaw, mats, furniture)
    if "bookcase_center_xy_m" in plan:
        bookcase = tuple(plan["bookcase_center_xy_m"])
        book_yaw = math.radians(float(plan.get("bookcase_yaw_deg", yaw)))
        oriented_box("StudyBookcase", bookcase, 1.35, (0.55, 2.20, 2.70), book_yaw, mats["wood"], furniture, bevel=0.035)
        for row in range(5):
            oriented_box(f"StudyShelf_{row}", bookcase, 0.30 + row * 0.52, (0.62, 2.10, 0.045), book_yaw, mats["light_wood"], furniture, bevel=0.008)
        for row in range(4):
            for column in range(6):
                material = mats["book_warm"] if (row + column) % 2 else mats["book_cool"]
                oriented_box(f"Book_{row}_{column}", bookcase, 0.54 + row * 0.52, (0.14, 0.18, 0.36), book_yaw, material, props, bevel=0.006, offset_xy=(-0.33, -0.70 + column * 0.28))


def emit_bathroom(plan: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> None:
    furniture, props = collections["Furniture"], collections["Props"]
    center = tuple(plan["center_xy_m"])
    yaw = math.radians(float(plan.get("yaw_deg", 0.0)))
    vanity_x = float(plan.get("vanity_local_x_m", -0.53))
    toilet_x = float(plan.get("toilet_local_x_m", 0.53))
    back_y = float(plan.get("back_wall_local_y_m", 0.72))
    oriented_box(
        "BathroomVanity", center, 0.43, (0.78, 0.46, 0.86), yaw,
        mats["cabinet"], furniture, bevel=0.045, offset_xy=(vanity_x, back_y),
    )
    oriented_box(
        "BathroomVanityTop", center, 0.89, (0.84, 0.50, 0.07), yaw,
        mats["stone"], furniture, bevel=0.025, offset_xy=(vanity_x, back_y - 0.02),
    )
    cylinder_xy = world_point(center, (vanity_x, back_y - 0.04), yaw)
    cylinder(
        "BathroomSink", (cylinder_xy[0], cylinder_xy[1], 0.94),
        0.23, 0.10, mats["porcelain"], props, vertices=40,
    )
    faucet_xy = world_point(center, (vanity_x + 0.20, back_y - 0.02), yaw)
    cylinder(
        "BathroomFaucet", (faucet_xy[0], faucet_xy[1], 1.12),
        0.022, 0.30, mats["steel"], props, vertices=24,
    )
    oriented_box(
        "BathroomMirror", center, 1.62, (0.76, 0.035, 0.92), yaw,
        mats["mirror"], props, bevel=0.018, offset_xy=(vanity_x, back_y + 0.25),
    )
    oriented_box(
        "BathroomToiletBase", center, 0.24, (0.42, 0.68, 0.48), yaw,
        mats["porcelain"], furniture, bevel=0.12, offset_xy=(toilet_x, back_y - 0.04),
    )
    oriented_box(
        "BathroomToiletTank", center, 0.64, (0.48, 0.25, 0.66), yaw,
        mats["porcelain"], furniture, bevel=0.055, offset_xy=(toilet_x, back_y + 0.20),
    )
    oriented_box(
        "BathroomToiletSeat", center, 0.51, (0.38, 0.47, 0.06), yaw,
        mats["porcelain"], props, bevel=0.10, offset_xy=(toilet_x, back_y - 0.14),
    )
    shower_x = float(plan.get("shower_local_x_m", 0.48))
    shower_y = float(plan.get("shower_local_y_m", -0.60))
    oriented_box(
        "BathroomShowerTray", center, 0.055, (0.92, 0.92, 0.11), yaw,
        mats["entry_floor"], furniture, bevel=0.035,
        offset_xy=(shower_x, shower_y),
    )
    oriented_box(
        "BathroomShowerGlassA", center, 1.05, (0.92, 0.025, 2.05), yaw,
        mats["glass"], furniture, bevel=0.012,
        offset_xy=(shower_x, shower_y - 0.45),
    )
    oriented_box(
        "BathroomShowerGlassB", center, 1.05, (0.025, 0.72, 2.05), yaw,
        mats["glass"], furniture, bevel=0.012,
        offset_xy=(shower_x + 0.45, shower_y - 0.08),
    )
    bath_mat_x = float(plan.get("bath_mat_local_x_m", -0.48))
    bath_mat_y = float(plan.get("bath_mat_local_y_m", -0.55))
    oriented_box(
        "BathroomBathMat", center, 0.018, (0.58, 0.42, 0.025), yaw,
        mats["towel"], props, bevel=0.016, offset_xy=(bath_mat_x, bath_mat_y),
    )
    oriented_box(
        "BathroomTowel", center, 1.34, (0.48, 0.035, 0.62), yaw,
        mats["towel"], props, bevel=0.018, offset_xy=(-0.70, 0.94),
    )
    for index, dx in enumerate((-0.12, 0.02, 0.15)):
        bottle_xy = world_point(center, (vanity_x + dx, back_y - 0.08), yaw)
        cylinder(
            f"BathroomBottle_{index}", (bottle_xy[0], bottle_xy[1], 1.03),
            0.035 + 0.006 * index, 0.19 + 0.03 * index,
            mats["book_cool"] if index % 2 else mats["book_warm"], props,
            vertices=20,
        )


def emit_laundry(plan: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> None:
    furniture, props = collections["Furniture"], collections["Props"]
    center = tuple(plan["center_xy_m"])
    yaw = math.radians(float(plan.get("yaw_deg", 0.0)))
    oriented_box(
        "LaundryWasher", center, 0.46, (0.66, 0.66, 0.92), yaw,
        mats["appliance"], furniture, bevel=0.055,
    )
    door_xy = world_point(center, (0.0, -0.345), yaw)
    door = cylinder(
        "LaundryWasherDoor", (door_xy[0], door_xy[1], 0.48),
        0.20, 0.055, mats["screen"], props, vertices=40,
    )
    door.rotation_euler = (math.pi / 2.0, 0.0, yaw)
    oriented_box(
        "LaundryCounter", center, 0.98, (0.78, 0.76, 0.08), yaw,
        mats["light_wood"], furniture, bevel=0.025,
    )
    oriented_box(
        "LaundryWallCabinet", center, 1.92, (0.82, 0.38, 0.72), yaw,
        mats["cabinet"], furniture, bevel=0.035, offset_xy=(0.0, 0.20),
    )
    basket_xy = world_point(center, (0.58, -0.05), yaw)
    cylinder(
        "LaundryBasket", (basket_xy[0], basket_xy[1], 0.27),
        0.23, 0.54, mats["rug"], props, vertices=32,
    )
    for index, dx in enumerate((-0.13, 0.0, 0.13)):
        bottle_xy = world_point(center, (dx, -0.08), yaw)
        cylinder(
            f"LaundryBottle_{index}", (bottle_xy[0], bottle_xy[1], 1.15),
            0.04, 0.24 + 0.04 * index, mats["book_cool"], props, vertices=20,
        )


def emit_storage(plan: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> None:
    furniture, props = collections["Furniture"], collections["Props"]
    center = tuple(plan["center_xy_m"])
    yaw = math.radians(float(plan.get("yaw_deg", 0.0)))
    width = float(plan.get("width_m", 1.4))
    depth = float(plan.get("depth_m", 0.58))
    height = float(plan.get("height_m", 2.35))
    oriented_box(
        str(plan.get("name", "StorageWardrobe")), center, height / 2.0,
        (width, depth, height), yaw, mats["wood"], furniture, bevel=0.045,
    )
    for side in (-1.0, 1.0):
        oriented_box(
            str(plan.get("name", "StorageWardrobe")) + f"_Door_{side:+.0f}",
            center, height / 2.0, (width / 2.0 - 0.035, 0.025, height - 0.12),
            yaw, mats["cabinet_panel"], props, bevel=0.010,
            offset_xy=(side * width / 4.0, -depth / 2.0 - 0.014),
        )
        oriented_box(
            str(plan.get("name", "StorageWardrobe")) + f"_Pull_{side:+.0f}",
            center, 1.16, (0.022, 0.025, 0.34), yaw, mats["brass"], props,
            bevel=0.005,
            offset_xy=(side * 0.07, -depth / 2.0 - 0.04),
        )


def emit_bedroom(plan: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> None:
    furniture, props = collections["Furniture"], collections["Props"]
    center = tuple(plan["bed_center_xy_m"])
    yaw = math.radians(float(plan.get("yaw_deg", 0.0)))
    width = float(plan.get("width_m", 1.65))
    length = float(plan.get("length_m", 2.05))
    oriented_box("BedFrame", center, 0.26, (width, length, 0.32), yaw, mats["wood"], furniture, bevel=0.06)
    oriented_box("Mattress", center, 0.52, (width - 0.08, length - 0.08, 0.28), yaw, mats["bedding"], furniture, bevel=0.10)
    oriented_box("BedHeadboard", center, 0.93, (width, 0.16, 1.10), yaw, mats["fabric"], furniture, bevel=0.08, offset_xy=(0.0, length / 2 - 0.02))
    oriented_box(
        "BedDuvet", center, 0.71, (width - 0.12, 0.58 * length, 0.11), yaw,
        mats["bedding_accent"], props, bevel=0.055,
        offset_xy=(0.0, -0.16 * length),
    )
    for index, dx in enumerate((-0.28 * width, 0.28 * width)):
        oriented_box(f"Pillow_{index}", center, 0.75, (0.44 * width, 0.42, 0.14), yaw, mats["bedding"], props, bevel=0.08, offset_xy=(dx, 0.28 * length))
    nightstand_count = int(plan.get("nightstand_count", 2))
    if nightstand_count <= 0:
        sides = ()
    elif nightstand_count == 1:
        sides = (float(plan.get("nightstand_side", 1.0)),)
    else:
        sides = (-1.0, 1.0)
    for side in sides:
        offset_x = side * (width / 2.0 + 0.28)
        oriented_box(
            f"Nightstand_{side:+.0f}", center, 0.31, (0.44, 0.42, 0.62), yaw,
            mats["light_wood"], furniture, bevel=0.045,
            offset_xy=(offset_x, 0.30 * length),
        )
        lamp_xy = world_point(center, (offset_x, 0.30 * length), yaw)
        cylinder(
            f"BedsideLamp_{side:+.0f}", (lamp_xy[0], lamp_xy[1], 0.78),
            0.13, 0.30, mats["warm_light"], props, vertices=32,
        )
    if sides:
        book_side = sides[-1]
        oriented_box(
            "BedsideBook", center, 0.66, (0.28, 0.18, 0.035), yaw,
            mats["book_warm"], props, bevel=0.008,
            offset_xy=(book_side * (width / 2.0 + 0.28), 0.30 * length),
        )
    if "wardrobe_center_xy_m" in plan:
        emit_storage(
            {
                "center_xy_m": plan["wardrobe_center_xy_m"],
                "yaw_deg": plan.get("wardrobe_yaw_deg", plan.get("yaw_deg", 0.0)),
                "width_m": plan.get("wardrobe_width_m", 1.30),
                "depth_m": 0.58,
                "height_m": 2.35,
                "name": "BedroomWardrobe",
            },
            collections,
            mats,
        )


ASSEMBLY_EMITTERS = {
    "entry": emit_entry,
    "kitchen": emit_kitchen,
    "dining": emit_dining,
    "living": emit_living,
    "study": emit_study,
    "bedroom": emit_bedroom,
    "bathroom": emit_bathroom,
    "laundry": emit_laundry,
    "storage": emit_storage,
}


def emit_furnishings(spec: dict[str, Any], collections: dict[str, bpy.types.Collection], mats: dict[str, bpy.types.Material]) -> list[str]:
    emitted = []
    for assembly in spec.get("furnishing_assemblies", []):
        kind = assembly.get("kind")
        require(kind in ASSEMBLY_EMITTERS, f"unsupported furnishing assembly: {kind}")
        ASSEMBLY_EMITTERS[kind](assembly, collections, mats)
        emitted.append(str(kind))
    return emitted


def add_review_cameras(spec: dict[str, Any], collection: bpy.types.Collection) -> list[bpy.types.Object]:
    cameras = []
    for raw in spec["review_cameras"]:
        data = bpy.data.cameras.new(raw["camera_id"] + "_Data")
        data.sensor_width = 36.0
        hfov = math.radians(float(raw["hfov_deg"]))
        data.lens = 36.0 / (2.0 * math.tan(hfov / 2.0))
        camera = bpy.data.objects.new(raw["camera_id"], data)
        collection.objects.link(camera)
        camera.location = raw["position_xyz_m"]
        aim(camera, raw["target_xyz_m"])
        cameras.append(camera)
    require(cameras, "at least one review camera is required")
    return cameras


def add_lights(
    spec: dict[str, Any],
    collection: bpy.types.Collection,
    detail_collection: bpy.types.Collection,
    mats: dict[str, bpy.types.Material],
) -> None:
    for raw in spec["lights"]:
        add_area(
            raw["light_id"],
            raw["position_xyz_m"],
            raw["target_xyz_m"],
            float(raw["energy"]),
            float(raw["size_m"]),
            tuple(raw["color_rgb"]),
            collection,
        )
        ceiling_fill_energy = float(raw.get("ceiling_fill_energy", 0.0))
        if ceiling_fill_energy > 0.0:
            position = tuple(float(value) for value in raw["position_xyz_m"])
            add_area(
                str(raw["light_id"]) + "_CeilingFill",
                (position[0], position[1], min(2.18, position[2] - 0.35)),
                (position[0], position[1], 2.92),
                ceiling_fill_energy,
                float(raw.get("ceiling_fill_size_m", raw["size_m"])),
                tuple(raw["color_rgb"]),
                collection,
            )
        position = tuple(float(value) for value in raw["position_xyz_m"])
        fixture = cylinder(
            "CeilingFixture_" + str(raw["light_id"]),
            (position[0], position[1], min(position[2] + 0.12, 2.84)),
            min(0.26, 0.16 + 0.03 * float(raw["size_m"])), 0.08,
            mats["warm_light"], detail_collection, vertices=40,
        )
        fixture["avengine_visual_fixture_for"] = str(raw["light_id"])
    sun = spec.get("sun")
    if sun:
        data = bpy.data.lights.new("Sun_Data", "SUN")
        data.energy = float(sun["energy"])
        data.color = tuple(sun["color_rgb"])
        actor = bpy.data.objects.new("Sun", data)
        collection.objects.link(actor)
        actor.rotation_euler = tuple(math.radians(float(value)) for value in sun["rotation_deg"])


def export_static_usd(scene: bpy.types.Scene, output_path: Path) -> dict[str, Any]:
    """Export static geometry, materials/textures and lights for UE/SPEAR."""
    allowed_collections = {
        "Architecture", "ArchitecturalDetails", "Glazing",
        "Furniture", "Props", "Lights",
    }
    bpy.ops.object.select_all(action="DESELECT")
    selected: list[bpy.types.Object] = []
    for obj in scene.objects:
        names = {owner.name for owner in obj.users_collection}
        if not names.intersection(allowed_collections):
            continue
        if obj.type not in {"MESH", "LIGHT"}:
            continue
        obj.hide_set(False)
        obj.select_set(True)
        selected.append(obj)
    require(selected, "USD export selected no static geometry or lights")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.wm.usd_export(
        filepath=str(output_path),
        check_existing=False,
        selected_objects_only=True,
        visible_objects_only=False,
        export_animation=False,
        export_materials=True,
        generate_preview_surface=True,
        generate_materialx_network=False,
        export_textures=True,
        export_textures_mode="NEW",
        overwrite_textures=False,
        relative_paths=True,
        xform_op_mode="TRS",
        root_prim_path="/RoomA",
        export_custom_properties=True,
        custom_properties_namespace="AVEngine",
        export_lights=True,
        export_cameras=False,
        export_meshes=True,
        export_curves=True,
        export_points=False,
        export_volumes=False,
        export_normals=True,
        export_uvmaps=True,
        export_mesh_colors=True,
        evaluation_mode="RENDER",
        convert_orientation=False,
        convert_scene_units="METERS",
        meters_per_unit=1.0,
    )
    require("FINISHED" in result, "USD export failed: " + str(result))
    for obj in selected:
        obj.select_set(False)
    return {
        "path": str(output_path),
        "format": "usda",
        "selected_object_count": len(selected),
        "selected_collections": sorted(allowed_collections),
        "excluded_collections": ["Cameras", "Anchors"],
        "exported_materials": True,
        "exported_textures": True,
        "exported_lights": True,
        "exported_cameras": False,
        "production_camera": "deferred_to_AVEngine",
    }

def args_from_cli() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--usd-output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = args_from_cli()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = args.output_root.expanduser().resolve()
    require(not output.exists(), f"refusing to replace output: {output}")
    for path in (output / "renders", output / "visual", output / "qa"):
        path.mkdir(parents=True, exist_ok=True)

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    scene.name = str(spec["room_spec_id"])
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    collections = {}
    for name in (
        "Architecture", "ArchitecturalDetails", "Glazing", "Furniture", "Props",
        "Anchors", "Lights", "Cameras",
    ):
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)
        collections[name] = collection

    mats = materials_for_style(str(spec["style_id"]))
    segments, wall_count = emit_architecture(spec, collections, mats)
    WALL_MOUNT_OMISSIONS.clear()
    emitted_assemblies = emit_furnishings(spec, collections, mats)
    wall_mounts = validate_wall_mounts(scene)
    architecture_details = emit_architecture_details(segments, collections, mats)
    for anchor_id, position in spec["anchors"].items():
        anchor = bpy.data.objects.new("ANCHOR_" + anchor_id, None)
        collections["Anchors"].objects.link(anchor)
        anchor.location = position
        anchor["anchor_id"] = anchor_id
        anchor.hide_render = True
    cameras = add_review_cameras(spec, collections["Cameras"])
    add_lights(
        spec, collections["Lights"], collections["ArchitecturalDetails"], mats
    )

    textured_tiles = {
        "WarmPlaster": 1.8,
        "OakFloor": 1.15,
        "EntryStoneFloor": 0.9,
        "LightOak": 0.85,
        "DarkOak": 0.85,
        "HonedStone": 1.2,
        "BacksplashTile": 2.4,
    }
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        names = {slot.material.name for slot in obj.material_slots if slot.material}
        for material_name, tile_m in textured_tiles.items():
            if material_name in names:
                projected_uvs(obj, tile_m)
                break

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = 15
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.world.color = (0.04, 0.05, 0.065)
    scene.camera = cameras[0]

    blend_path = output / f"{spec['room_spec_id']}.blend"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    for camera in cameras:
        scene.camera = camera
        scene.render.filepath = str(output / "renders" / f"{camera.name}.png")
        bpy.ops.render.render(write_still=True)

    collections["Anchors"].hide_viewport = True
    visual_glb = output / "visual" / f"{spec['room_spec_id']}.glb"
    result = bpy.ops.export_scene.gltf(
        filepath=str(visual_glb),
        check_existing=False,
        export_format="GLB",
        use_active_scene=True,
        use_visible=True,
        export_apply=True,
        export_yup=True,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_materials="EXPORT",
        export_extras=True,
    )
    require("FINISHED" in result, f"GLB export failed: {result}")

    # Navigation must not reuse the complete render mesh.  In particular, the
    # upper face of every ceiling becomes a large walkable island in Habitat.
    # Keep room boundaries, floors, open door leaves and major furniture, but
    # omit ceilings, glazing, fixtures and small props from the collision GLB.
    collision_objects = []
    for obj in scene.objects:
        collection_names = {collection.name for collection in obj.users_collection}
        include = (
            obj.type == "MESH"
            and (
                ("Architecture" in collection_names and not obj.name.startswith("Ceiling_"))
                or "Furniture" in collection_names
            )
        )
        obj.select_set(include)
        if include:
            collision_objects.append(obj)
    require(collision_objects, "collision export selected no mesh objects")
    collision_glb = output / "visual" / f"{spec['room_spec_id']}_collision.glb"
    result = bpy.ops.export_scene.gltf(
        filepath=str(collision_glb),
        check_existing=False,
        export_format="GLB",
        use_active_scene=True,
        use_selection=True,
        use_visible=False,
        export_apply=True,
        export_yup=True,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_materials="NONE",
        export_extras=False,
    )
    require("FINISHED" in result, f"collision GLB export failed: {result}")
    usd_path = args.usd_output.expanduser().resolve() if args.usd_output is not None else output / "usd" / (str(spec["room_spec_id"]) + ".usda")
    usd_record = export_static_usd(scene, usd_path)
    for obj in collision_objects:
        obj.select_set(False)
    write_json(output / "functional_anchors.json", {
        "room_spec_id": spec["room_spec_id"],
        "coordinate_system": "Blender +Z up metres; exported GLB +Y up",
        "anchors": spec["anchors"],
    })
    write_json(output / "qa" / "build_report.json", {
        "status": "research_candidate",
        "qualification_claim": False,
        "room_spec_id": spec["room_spec_id"],
        "topology_family": spec["topology_family"],
        "style_id": spec["style_id"],
        "zone_count": len(spec["zones"]),
        "link_count": len(spec["links"]),
        "exterior_opening_count": len(spec["exterior_openings"]),
        "wall_panel_count": wall_count,
        "architecture_details": architecture_details,
        "collision_mesh_object_count": len(collision_objects),
        "object_count": len(scene.objects),
        "furnishing_assemblies": emitted_assemblies,
        "review_cameras": [camera.name for camera in cameras],
        "wall_mounts": wall_mounts,
        "wall_mount_omissions": list(WALL_MOUNT_OMISSIONS),
        "blend": str(blend_path),
        "visual_glb": str(visual_glb),
        "collision_glb": str(collision_glb),
        "usd": usd_record,
        "no_qa_generated": True,
        "formal_dataset_count": 0,
    })
    print(f"SEMANTIC_HOUSEHOLD_BUILD_OK room={spec['room_spec_id']} output={output}")


if __name__ == "__main__":
    main()
