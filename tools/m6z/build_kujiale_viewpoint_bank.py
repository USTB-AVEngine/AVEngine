#!/usr/bin/env python3
"""Build a fresh CPU-only Kujiale viewpoint and source-route request bank."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import load_json, write_json  # noqa: E402
from avengine.m6x.raster_pathfinder import (  # noqa: E402
    build_polygon_raster_obstacle_map,
)
from avengine.m6x.room_feasibility import RoomFeasibilityCompiler  # noqa: E402
from avengine.optional_backends.kujiale_viewpoints import (  # noqa: E402
    CPU_PREFILTER_AUTHORITY,
    CPU_PREFILTER_BOUNDARY,
    build_viewpoint_plan,
    generate_polyline_corridors,
    generate_straight_corridors,
)
from avengine.optional_backends.kujiale_scene_binding import (  # noqa: E402
    validate_kujiale_scene_binding as _validate_scene_binding,
)
from tools.m6x.compile_kujiale_feasibility_bank import (  # noqa: E402
    build_kujiale_runtime_obstacles,
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_topdown(
    *,
    output: Path,
    obstacle_map: Any,
    corridors: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Any],
) -> Path:
    width, height = 1200, 900
    header = 78
    margin = 34
    legend_width = 320
    nav = np.asarray(obstacle_map.binary_navmesh, dtype=np.bool_)
    rgb = np.empty((*nav.shape, 3), dtype=np.uint8)
    rgb[:] = (33, 38, 45)
    rgb[nav] = (178, 218, 184)
    available_w = width - 2 * margin - legend_width
    available_h = height - header - 2 * margin
    scale = min(available_w / nav.shape[1], available_h / nav.shape[0])
    draw_w = max(2, int(round(nav.shape[1] * scale)))
    draw_h = max(2, int(round(nav.shape[0] * scale)))
    left = margin + (available_w - draw_w) // 2
    top = header + (height - header - draw_h) // 2
    image = Image.new("RGB", (width, height), (20, 23, 28))
    map_image = Image.fromarray(rgb, mode="RGB").resize(
        (draw_w, draw_h), Image.Resampling.NEAREST
    )
    image.paste(map_image, (left, top))
    draw = ImageDraw.Draw(image, "RGBA")
    bounds = np.asarray(obstacle_map.bounds_m, dtype=np.float64)

    def point(value: Sequence[float]) -> tuple[float, float]:
        x_fraction = (float(value[0]) - bounds[0, 0]) / (bounds[1, 0] - bounds[0, 0])
        z_fraction = (float(value[1]) - bounds[0, 2]) / (bounds[1, 2] - bounds[0, 2])
        return left + x_fraction * draw_w, top + z_fraction * draw_h

    for corridor in corridors[:128]:
        waypoints = corridor.get(
            "waypoints_xy_m",
            [corridor["start_xy_m"], corridor["end_xy_m"]],
        )
        draw.line(
            [point(value) for value in waypoints],
            fill=(55, 106, 165, 50),
            width=1,
        )
    palette = (
        (230, 74, 70, 255),
        (55, 126, 230, 255),
        (241, 164, 61, 255),
        (146, 84, 205, 255),
        (32, 159, 112, 255),
        (225, 95, 157, 255),
        (80, 180, 210, 255),
        (190, 190, 70, 255),
    )
    ray_length = 0.9
    for index, view in enumerate(bank["viewpoints"]):
        color = palette[index % len(palette)]
        camera_xy = view["position_xyz_m"][:2]
        camera_px = point(camera_xy)
        yaw = math.radians(float(view["yaw_ue_deg"]))
        endpoint = (
            float(camera_xy[0]) + math.cos(yaw) * ray_length,
            float(camera_xy[1]) + math.sin(yaw) * ray_length,
        )
        draw.line((*camera_px, *point(endpoint)), fill=color, width=5)
        draw.ellipse(
            (
                camera_px[0] - 7,
                camera_px[1] - 7,
                camera_px[0] + 7,
                camera_px[1] + 7,
            ),
            fill=color,
            outline=(0, 0, 0, 255),
            width=2,
        )
        draw.text(
            (camera_px[0] + 9, camera_px[1] - 15),
            f"V{index + 1}",
            font=_font(12),
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
        legend_y = header + 28 + index * 72
        draw.text(
            (width - legend_width + 18, legend_y),
            f"V{index + 1}  {view['viewpoint_id']}",
            font=_font(13),
            fill=color,
        )
        draw.text(
            (width - legend_width + 18, legend_y + 23),
            (
                "xyz="
                + ",".join(f"{float(value):.2f}" for value in view["position_xyz_m"])
                + "  "
                + f"yaw={view['yaw_ue_deg']:.1f} deg"
            ),
            font=_font(11),
            fill=(220, 220, 220, 255),
        )
        draw.text(
            (width - legend_width + 18, legend_y + 43),
            (
                f"corridors={view['eligible_corridor_count']} | "
                f"pairs described={view['described_pair_count']}/"
                f"{view['directed_pair_combination_count']} | pixels pending"
            ),
            font=_font(11),
            fill=(180, 180, 180, 255),
        )
    header_draw = ImageDraw.Draw(image)
    header_draw.rectangle((0, 0, width, header), fill=(0, 0, 0))
    header_draw.text(
        (20, 12),
        f"{bank['room_id']} AUTONOMOUS VIEWPOINT PREFILTER",
        font=_font(24),
        fill=(255, 255, 255),
    )
    header_draw.text(
        (20, 46),
        "CPU 2D PREFILTER ONLY | UE/SPEAR + NATIVE PIXELS + MINI-PILOT PENDING",
        font=_font(15),
        fill=(255, 199, 71),
    )
    path = output / "viewpoint_topdown.png"
    image.save(path)
    return path


def run(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        metadata_path = args.scene_metadata.expanduser().resolve()
        profile_path = args.source_profile.expanduser().resolve()
        visual_profile_path = args.visual_profile.expanduser().resolve()
        motion_path = args.motion_profiles.expanduser().resolve()
        camera_catalog_path = args.camera_catalog.expanduser().resolve(strict=True)
        metadata = load_json(metadata_path)
        profile = load_json(profile_path)
        visual_profile = load_json(visual_profile_path)
        motion_profiles = load_json(motion_path)
        camera_catalog = load_json(camera_catalog_path)
        scene_binding = _validate_scene_binding(
            metadata=metadata,
            source_profile=profile,
            visual_profile=visual_profile,
        )
        polygon = metadata.get("room_polygon_xy_m")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise RuntimeError("scene metadata lacks room_polygon_xy_m")
        raw_heights = profile.get("emitter_heights_m")
        if not isinstance(raw_heights, Mapping) or set(raw_heights) != {
            "human0",
            "dog0",
        }:
            raise RuntimeError(
                "source profile must declare human0/dog0 emitter heights"
            )
        emitter_heights = {
            "source1": float(raw_heights["human0"]),
            "source2": float(raw_heights["dog0"]),
        }
        floor = float(metadata.get("floor_z_m", 0.0))
        clearance = (
            float(args.minimum_clearance_m)
            if args.minimum_clearance_m is not None
            else float(profile.get("source_center_margin_m", 0.03))
        )
        obstacles, obstacle_statistics = build_kujiale_runtime_obstacles(
            metadata, emitter_heights
        )
        _pathfinder, obstacle_map = build_polygon_raster_obstacle_map(
            polygon_xz_m=polygon,
            rigid_obstacles=obstacles,
            floor_height_m=floor,
            meters_per_pixel=args.meters_per_pixel,
            padding_m=max(
                args.meters_per_pixel * 2.0,
                clearance + args.meters_per_pixel,
            ),
            minimum_clearance_m=clearance,
            authority=(
                "kujiale_room_polygon_plus_usd_descendant_mesh_blocking_footprints"
            ),
            claim_boundary=CPU_PREFILTER_BOUNDARY,
        )
        compiler = RoomFeasibilityCompiler(obstacle_map)
        regions = {
            slot: compiler.compile(
                source_center_height_m=floor + emitter_heights[slot],
                minimum_navmesh_clearance_m=0.0,
                minimum_rigid_clearance_m=0.0,
                sample_spacing_m=args.sample_spacing_m,
            )
            for slot in ("source1", "source2")
        }
        if args.route_geometry == "straight":
            corridors = generate_straight_corridors(
                region_by_source=regions,
                motion_profiles=motion_profiles,
                species=args.species,
                maximum_corridors=args.maximum_corridors,
            )
        else:
            corridors = generate_polyline_corridors(
                region_by_source=regions,
                motion_profiles=motion_profiles,
                species=args.species,
                maximum_corridors=args.maximum_corridors,
                waypoint_spacing_m=args.waypoint_spacing_m,
                maximum_segments=args.maximum_route_segments,
                maximum_segment_length_m=args.maximum_segment_length_m,
            )
        plan = build_viewpoint_plan(
            room_id=str(metadata.get("room_id")),
            room_type=str(metadata.get("room_type")),
            scene_binding=scene_binding,
            camera_catalog=camera_catalog,
            region_by_source=regions,
            corridors=corridors,
            view_count=args.view_count,
            hfov_degrees=args.hfov_degrees,
            side_margin=args.side_margin,
            screening_corridor_count=args.screening_corridor_count,
            minimum_pair_separation_m=args.minimum_pair_separation_m,
            maximum_pair_records=args.maximum_pair_records,
        )
        bank = plan["bank"]
        bank["inputs"] = {
            "scene_metadata": str(metadata_path),
            "source_profile": str(profile_path),
            "visual_profile": str(visual_profile_path),
            "motion_profiles": str(motion_path),
            "camera_catalog": str(camera_catalog_path),
        }
        bank["scene_binding"] = scene_binding
        bank["obstacle_statistics"] = obstacle_statistics
        corridors_document = {
            "status": "cpu_prefilter_complete",
            "authority": CPU_PREFILTER_AUTHORITY,
            "claim_boundary": CPU_PREFILTER_BOUNDARY,
            "room_id": metadata.get("room_id"),
            "room_type_annotation": metadata.get("room_type"),
            "room_type_role": "post_generation_annotation_only",
            "route_geometry": args.route_geometry,
            "corridor_count": len(corridors),
            "corridors": corridors,
        }
        write_json(output / "viewpoint_bank.json", bank)
        route_filename = f"{args.route_geometry}_corridors.json"
        write_json(output / route_filename, corridors_document)
        corridor_root = output / "corridors"
        corridor_root.mkdir()
        for viewpoint_id, subset in plan["corridor_subsets"].items():
            write_json(corridor_root / f"{viewpoint_id}.json", subset)
        topdown = _render_topdown(
            output=output,
            obstacle_map=obstacle_map,
            corridors=corridors,
            bank=bank,
        )
        write_json(
            output / "delivery.json",
            {
                "status": "cpu_prefilter_complete",
                "research_only": True,
                "episode_counted": False,
                "formal_dataset_count": 0,
                "qualification_claim": False,
                "authority": CPU_PREFILTER_AUTHORITY,
                "claim_boundary": CPU_PREFILTER_BOUNDARY,
                "room_id": metadata.get("room_id"),
                "route_geometry": args.route_geometry,
                "route_fallback_used": False,
                "camera_catalog": str(camera_catalog_path),
                "camera_catalog_placement_count": bank[
                    "camera_pool_placement_count"
                ],
                "scene_binding": scene_binding,
                "viewpoint_count": bank["viewpoint_count"],
                "corridor_count": len(corridors),
                "native_pixel_validation_status": "pending",
                "ue_spear_certification_status": "pending",
                "mini_pilot_status": "pending",
                "outputs": {
                    "viewpoint_bank": "viewpoint_bank.json",
                    "source_corridors": route_filename,
                    "corridor_subsets": "corridors/",
                    "topdown": topdown.name,
                },
            },
        )
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {
                "status": "fail",
                "error_type": type(error).__name__,
                "error": str(error),
                "claim_boundary": CPU_PREFILTER_BOUNDARY,
            },
        )
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--visual-profile", type=Path, required=True)
    parser.add_argument(
        "--motion-profiles",
        type=Path,
        default=REPOSITORY / "examples/qa_v2/motion_profiles_v1.json",
    )
    parser.add_argument("--species", nargs="+", default=["human", "dog"])
    parser.add_argument(
        "--camera-catalog",
        type=Path,
        required=True,
        help="complete target-agnostic room camera placement catalog",
    )
    parser.add_argument(
        "--route-geometry",
        choices=("straight", "polyline"),
        required=True,
        help="explicit source-route geometry; failures never fall back",
    )
    parser.add_argument("--view-count", type=int, default=4)
    parser.add_argument("--hfov-degrees", type=float, default=105.0)
    parser.add_argument("--meters-per-pixel", type=float, default=0.05)
    parser.add_argument("--minimum-clearance-m", type=float)
    parser.add_argument("--sample-spacing-m", type=float, default=0.25)
    parser.add_argument("--side-margin", type=float, default=0.06)
    parser.add_argument("--maximum-corridors", type=int, default=256)
    parser.add_argument("--waypoint-spacing-m", type=float, default=0.25)
    parser.add_argument("--maximum-route-segments", type=int, default=4)
    parser.add_argument("--maximum-segment-length-m", type=float, default=1.80)
    parser.add_argument("--screening-corridor-count", type=int, default=64)
    parser.add_argument("--minimum-pair-separation-m", type=float, default=0.60)
    parser.add_argument("--maximum-pair-records", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": "cpu_prefilter_complete",
                "output": str(output),
                "native_pixel_validation_status": "pending",
                "ue_spear_certification_status": "pending",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
