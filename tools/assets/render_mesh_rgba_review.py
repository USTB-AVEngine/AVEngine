"""Render a mesh to a transparent RGBA review preview.

This is a review aid for image-conditioned 3D tools. It reads an existing mesh,
does not generate or register an asset, and writes a fresh PNG plus a sidecar
record identifying the mesh source. Registered assets require registry and
source-manifest provenance; generated candidates require an explicit Pixal3D
receipt. Cycles is explicitly configured for CPU rendering with a small sample
count and bounded render threads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import bpy
from mathutils import Vector


REPOSITORY = Path(__file__).resolve().parents[2]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="mesh GLB")
    parser.add_argument("--output", type=Path, required=True, help="fresh RGBA PNG")
    parser.add_argument(
        "--source-kind",
        choices=("registered_asset", "generated_candidate"),
        default="registered_asset",
        help=(
            "source provenance: registered_asset requires registry/source_manifest; "
            "generated_candidate requires an explicit Pixal3D receipt"
        ),
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help=(
            "registered source asset ID (required for registered_asset; optional "
            "label for generated_candidate)"
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("examples/runtime/source_asset_runtime_profiles.json"),
        help="runtime registry containing the source asset ID",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="prepared-input source_manifest.json; defaults to a sibling file",
    )
    parser.add_argument(
        "--pixal-receipt",
        type=Path,
        default=None,
        help=(
            "explicit AVEngine Pixal3D receipt whose output.path must equal "
            "the generated candidate input"
        ),
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--azimuth-deg", type=float, default=25.0)
    parser.add_argument("--elevation-deg", type=float, default=12.0)
    parser.add_argument("--focal-mm", type=float, default=50.0)
    parser.add_argument("--margin", type=float, default=1.20)
    if argv is None:
        raw = sys.argv
        argv = raw[raw.index("--") + 1:] if "--" in raw else []
    return parser.parse_args(argv)


def _load_unique_registry_record(registry_path: Path, asset_id: str) -> dict[str, Any]:
    if not registry_path.is_file():
        raise FileNotFoundError(f"source registry is missing: {registry_path}")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source registry must be a JSON object: {registry_path}")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"source registry has no assets list: {registry_path}")
    matches = [
        record
        for record in assets
        if isinstance(record, dict) and record.get("asset_id") == asset_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"source registry must contain exactly one record for {asset_id!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _resolve_pixal_receipt(input_path: Path, requested: Path | None) -> dict[str, Any]:
    if requested is None:
        raise ValueError(
            "--pixal-receipt is required when --source-kind=generated_candidate"
        )
    receipt_path = requested.expanduser().resolve()
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FileNotFoundError(f"Pixal3D receipt is missing or unsafe: {receipt_path}")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Pixal3D receipt: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Pixal3D receipt must be a JSON object")
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema.startswith("avengine_pixal3d_"):
        raise ValueError("Pixal3D receipt has no explicit AVEngine Pixal3D schema")
    if payload.get("status") != "passed":
        raise ValueError(
            f"Pixal3D receipt status must be 'passed', got {payload.get('status')!r}"
        )
    output = payload.get("output")
    output_path_value = output.get("path") if isinstance(output, dict) else None
    if not isinstance(output_path_value, str) or not output_path_value:
        raise ValueError("Pixal3D receipt has no explicit output.path")
    receipt_output = Path(output_path_value).expanduser().resolve()
    if receipt_output != input_path:
        raise ValueError(
            f"Pixal3D receipt output.path differs from input: "
            f"expected {input_path}, got {receipt_output}"
        )
    return {
        "verified": True,
        "status": "verified_pixal3d_receipt",
        "reason": None,
        "receipt_path": str(receipt_path),
        "receipt_schema": schema,
        "receipt_status": payload["status"],
        "output_path": str(receipt_output),
    }


def _resolve_source_manifest(
    input_path: Path,
    registry_path: Path,
    asset_id: str,
    registry_record: dict[str, Any],
    requested: Path | None,
) -> dict[str, Any]:
    manifest_path = (
        requested.expanduser().resolve()
        if requested is not None
        else input_path.parent / "source_manifest.json"
    )
    if not manifest_path.is_file():
        return {
            "verified": False,
            "status": "caller_declared_unverified",
            "reason": f"source manifest is missing: {manifest_path}",
            "manifest_path": str(manifest_path),
            "ue_object_path": None,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("source_manifest must be a JSON object")
        rows = [
            row
            for row in manifest.get("source_objects", [])
            if isinstance(row, dict) and row.get("asset_id") == asset_id
        ]
        if len(rows) != 1:
            raise ValueError(
                f"source_manifest must contain exactly one source_objects row for "
                f"{asset_id!r}; found {len(rows)}"
            )
        row = rows[0]
        ue_object_path = row.get("ue_object_path")
        if not isinstance(ue_object_path, str) or not ue_object_path:
            raise ValueError("source_manifest row has no ue_object_path")
        static_name = row.get("static_mesh_input")
        if not isinstance(static_name, str) or not static_name:
            raise ValueError("source_manifest row has no static_mesh_input")
        expected_input = (manifest_path.parent / static_name).resolve()
        if expected_input != input_path:
            raise ValueError(
                f"source_manifest mesh binding differs: expected {expected_input}, "
                f"got {input_path}"
            )
        source_registry = manifest.get("source_registry")
        if not isinstance(source_registry, dict):
            raise ValueError("source_manifest has no source_registry object")
        manifest_registry = source_registry.get("path")
        if not isinstance(manifest_registry, str):
            raise ValueError("source_manifest has no source_registry.path")
        manifest_registry_path = Path(manifest_registry)
        if not manifest_registry_path.is_absolute():
            manifest_registry_path = (REPOSITORY / manifest_registry_path).resolve()
        if manifest_registry_path != registry_path:
            raise ValueError(
                f"source_manifest registry differs: expected {registry_path}, "
                f"got {manifest_registry_path}"
            )
        uri_records = source_registry.get("mesh_uri_records")
        if not isinstance(uri_records, dict):
            raise ValueError("source_manifest has no source_registry.mesh_uri_records")
        uri = uri_records.get(asset_id)
        registry_uri = registry_record.get("geometry", {}).get("source_mesh_uri")
        if not isinstance(uri, str) or uri != registry_uri:
            raise ValueError(
                f"source_manifest mesh URI does not match registry: {uri!r} "
                f"vs {registry_uri!r}"
            )
        return {
            "verified": True,
            "status": "verified_registry_and_source_manifest",
            "reason": None,
            "manifest_path": str(manifest_path),
            "ue_object_path": ue_object_path,
            "stage_file": row.get("stage_file"),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "verified": False,
            "status": "caller_declared_unverified",
            "reason": f"source manifest binding failed: {error}",
            "manifest_path": str(manifest_path),
            "ue_object_path": None,
        }


def _bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, float]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points: list[Vector] = []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        points.extend(
            evaluated.matrix_world @ Vector(corner)
            for corner in evaluated.bound_box
        )
    if not points:
        raise RuntimeError("mesh has no bounding-box points")
    minimum = Vector((
        min(point.x for point in points),
        min(point.y for point in points),
        min(point.z for point in points),
    ))
    maximum = Vector((
        max(point.x for point in points),
        max(point.y for point in points),
        max(point.z for point in points),
    ))
    center = (minimum + maximum) * 0.5
    span = max(maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z)
    if span <= 1e-8:
        raise RuntimeError("mesh bounding box is degenerate")
    return center, span


def _make_camera(center: Vector, span: float, args: argparse.Namespace) -> bpy.types.Object:
    data = bpy.data.cameras.new("registered_mesh_preview_camera")
    data.lens = float(args.focal_mm)
    data.sensor_fit = "HORIZONTAL"
    camera = bpy.data.objects.new("registered_mesh_preview_camera", data)
    bpy.context.scene.collection.objects.link(camera)

    azimuth = math.radians(float(args.azimuth_deg))
    elevation = math.radians(float(args.elevation_deg))
    direction = Vector((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ))
    half_fov = math.atan(32.0 / (2.0 * float(args.focal_mm)))
    distance = float(args.margin) * span / (2.0 * math.tan(half_fov))
    camera.location = center + direction * distance
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    return camera


def _apply_preview_material(meshes: list[bpy.types.Object]) -> None:
    # The recovered UE SkeletalMesh GLB carries a neutral material without
    # textures. Use a clearly labelled in-memory gray preview material so the
    # CPU Cycles image remains readable; this never writes back to the source.
    material = bpy.data.materials.new("registered_mesh_preview_neutral_gray")
    material.diffuse_color = (0.42, 0.46, 0.52, 1.0)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (0.42, 0.46, 0.52, 1.0)
    shader.inputs["Roughness"].default_value = 0.82
    if "Emission Color" in shader.inputs:
        shader.inputs["Emission Color"].default_value = (0.05, 0.06, 0.08, 1.0)
    if "Emission Strength" in shader.inputs:
        shader.inputs["Emission Strength"].default_value = 0.12
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def _make_lights(center: Vector, span: float) -> None:
    for index, (offset, energy, size) in enumerate((
        ((2.0, -2.0, 2.5), 180.0, 2.0),
        ((-2.0, -1.0, 1.5), 120.0, 2.5),
        ((0.0, 2.0, 3.0), 160.0, 2.0),
    )):
        data = bpy.data.lights.new(f"registered_mesh_preview_light_{index}", "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size * span
        light = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(light)
        light.location = center + Vector(offset) * span
        light.rotation_euler = (
            center - light.location
        ).to_track_quat("-Z", "Y").to_euler()
    sun_data = bpy.data.lights.new(
        "registered_mesh_preview_sun", "SUN"
    )
    sun_data.energy = 0.35
    sun = bpy.data.objects.new(sun_data.name, sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = center + Vector((-2.0, -3.0, 4.0)) * span
    sun.rotation_euler = (
        center - sun.location
    ).to_track_quat("-Z", "Y").to_euler()


def _write_record(
    output: Path,
    args: argparse.Namespace,
    input_path: Path,
    registry_path: Path | None,
    binding: dict[str, Any],
    center: Vector,
    span: float,
    scene: bpy.types.Scene,
) -> Path:
    record_path = output.with_suffix(".json")
    record = {
        "schema": "avengine_mesh_rgba_review_v2",
        "status": "review_only",
        "source_kind": args.source_kind,
        "rendered_from_registered_mesh": (
            args.source_kind == "registered_asset" and bool(binding["verified"])
        ),
        "rendered_from_generated_candidate": (
            args.source_kind == "generated_candidate" and bool(binding["verified"])
        ),
        "asset_id": args.asset_id,
        "source_mesh": str(input_path),
        "source_registry": str(registry_path) if registry_path is not None else None,
        "source_binding": binding,
        "original_canonical_input": False,
        "canonical_image_replacement": False,
        "new_asset_registration": False,
        "purpose": "mesh RGBA review for image-conditioned 3D tools",
        "renderer": {
            "engine": str(scene.render.engine),
            "device": str(scene.cycles.device),
            "cpu_only": (
                scene.render.engine == "CYCLES"
                and scene.cycles.device == "CPU"
            ),
            "threads": int(scene.render.threads),
            "samples": int(scene.cycles.samples),
            "animation": False,
            "transparent_film": bool(scene.render.film_transparent),
            "image_format": str(scene.render.image_settings.file_format),
            "color_mode": str(scene.render.image_settings.color_mode),
            "material_override": "neutral_gray_preview_only",
        },
        "resolution": {"width": int(args.width), "height": int(args.height)},
        "camera": {
            "azimuth_deg": float(args.azimuth_deg),
            "elevation_deg": float(args.elevation_deg),
            "focal_mm": float(args.focal_mm),
            "margin": float(args.margin),
            "bounds_center": [float(center.x), float(center.y), float(center.z)],
            "bounds_span": float(span),
        },
        "output": str(output),
    }
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    registry_path = args.registry.expanduser().resolve()
    record_path = output_path.with_suffix(".json")
    if not input_path.is_file():
        raise FileNotFoundError(f"mesh input is missing: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite preview: {output_path}")
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite preview record: {record_path}")
    if output_path.suffix.lower() != ".png":
        raise ValueError("preview output must be a .png path")
    if args.width < 1 or args.height < 1:
        raise ValueError("width and height must be positive")
    if args.threads < 1 or args.samples < 1:
        raise ValueError("threads and samples must be positive")
    if args.focal_mm <= 0 or args.margin <= 0:
        raise ValueError("focal length and margin must be positive")

    if args.source_kind == "registered_asset":
        if not args.asset_id:
            raise ValueError("--asset-id is required when --source-kind=registered_asset")
        if args.pixal_receipt is not None:
            raise ValueError("--pixal-receipt is only valid for generated_candidate")
        registry_path = args.registry.expanduser().resolve()
        registry_record = _load_unique_registry_record(registry_path, args.asset_id)
        binding = _resolve_source_manifest(
            input_path,
            registry_path,
            args.asset_id,
            registry_record,
            args.source_manifest,
        )
        if not binding["verified"]:
            raise ValueError(
                "registered_asset source binding failed: "
                f"{binding.get('reason') or 'unknown source-manifest error'}"
            )
    else:
        if args.source_manifest is not None:
            raise ValueError("--source-manifest is only valid for registered_asset")
        registry_path = None
        binding = _resolve_pixal_receipt(input_path, args.pixal_receipt)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("mesh input contains no mesh objects")
    center, span = _bounds(meshes)
    camera = _make_camera(center, span, args)
    _apply_preview_material(meshes)
    _make_lights(center, span)

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(args.samples)
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.use_denoising = False
    scene.render.resolution_x = int(args.width)
    scene.render.resolution_y = int(args.height)
    scene.render.resolution_percentage = 100
    scene.render.threads_mode = "FIXED"
    scene.render.threads = int(args.threads)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(output_path)
    scene.view_settings.view_transform = "Standard"
    # Keep the explicitly created lights renderable. The factory GLB import
    # has no other non-mesh scene content that needs to be rendered; hiding
    # lights here would leave only the preview material emission and produce
    # an unnecessarily dark silhouette.
    for obj in bpy.context.scene.objects:
        if obj.type not in {"MESH", "LIGHT"} and obj is not camera:
            obj.hide_render = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"preview was not created: {output_path}")
    record_path = _write_record(
        output_path, args, input_path, registry_path, binding, center, span, scene
    )
    print(
        f"MESH_RGBA_REVIEW_OK {output_path} "
        f"engine={scene.render.engine} device={scene.cycles.device} "
        f"samples={scene.cycles.samples} threads={scene.render.threads}"
    )
    print(f"MESH_RGBA_REVIEW_RECORD {record_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"mesh RGBA review refused: {error}", file=sys.stderr)
        raise SystemExit(2)
