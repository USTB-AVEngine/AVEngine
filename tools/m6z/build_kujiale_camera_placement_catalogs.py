#!/usr/bin/env python3
"""Build complete target-agnostic CPU camera catalogs for Kujiale rooms."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.camera_placement import (  # noqa: E402
    DEFAULT_YAW_DEGREES,
    CameraPlacementError,
    iter_feasible_camera_placements,
)
from avengine.contracts.json_io import load_json, write_json  # noqa: E402
from avengine.m6x.raster_pathfinder import (  # noqa: E402
    build_polygon_raster_obstacle_map,
)
from avengine.optional_backends.kujiale_scene_binding import (  # noqa: E402
    validate_kujiale_scene_binding,
)
from avengine.security import (  # noqa: E402
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
)
from tools.m6x.compile_kujiale_feasibility_bank import (  # noqa: E402
    build_kujiale_camera_obstacles,
)


AUTHORITY = "kujiale_room_polygon_plus_usd_descendant_footprint_cpu_raster"
CLAIM_BOUNDARY = (
    "camera-center floor/free-space planning from the declared room-instance "
    "polygon and descendant-mesh ground-footprint CPU raster only; not a native "
    "navmesh, UE/SPEAR pixel or readback certification, body-volume collision, "
    "M3/RLR qualification, viewpoint selection, mini-pilot or dataset admission"
)
DEFAULT_CAMERA_HEIGHTS = ("low=1.05", "base=1.50", "high=1.85")


class KujialeCameraCatalogError(ValueError):
    """Kujiale CPU room inputs cannot produce one fresh camera catalog bundle."""


def _required_file(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{owner} is not a file: {resolved}")
    return resolved


def _fresh_output(path: Path) -> tuple[Path, WorkspacePathPolicy]:
    raw = path.expanduser()
    if not raw.is_absolute():
        raise KujialeCameraCatalogError("--output must be an absolute path")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise KujialeCameraCatalogError(
            f"--output parent does not exist: {raw.parent}"
        ) from exc
    output = parent / raw.name
    if os.path.lexists(output):
        raise KujialeCameraCatalogError(f"refusing to replace output: {output}")
    try:
        policy = WorkspacePathPolicy.from_roots((parent,))
    except PathPolicyError as exc:
        raise KujialeCameraCatalogError(str(exc)) from exc
    return output, policy


def _positive(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KujialeCameraCatalogError(f"{owner} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise KujialeCameraCatalogError(f"{owner} must be a positive finite number")
    return result


def _nonnegative(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KujialeCameraCatalogError(
            f"{owner} must be a nonnegative finite number"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise KujialeCameraCatalogError(
            f"{owner} must be a nonnegative finite number"
        )
    return result


def _camera_heights(values: Sequence[str] | None) -> tuple[tuple[str, float], ...]:
    raw = DEFAULT_CAMERA_HEIGHTS if values is None else tuple(values)
    if not raw:
        raise KujialeCameraCatalogError("at least one --camera-height is required")
    parsed = []
    seen = set()
    for index, value in enumerate(raw):
        if not isinstance(value, str) or value.count("=") != 1:
            raise KujialeCameraCatalogError(
                f"camera height {index} must use HEIGHT_ID=METERS"
            )
        height_id, raw_height = (item.strip() for item in value.split("=", 1))
        if not height_id or height_id in seen:
            raise KujialeCameraCatalogError("camera height ids must be nonempty and unique")
        seen.add(height_id)
        try:
            height = float(raw_height)
        except ValueError as exc:
            raise KujialeCameraCatalogError(
                f"camera height {height_id!r} must be numeric"
            ) from exc
        parsed.append((height_id, _positive(height, owner=f"camera height {height_id}")))
    return tuple(sorted(parsed, key=lambda item: (item[1], item[0].encode("utf-8"))))


def _room_annotation(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("room_type")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise KujialeCameraCatalogError("metadata.room_type must be null or nonempty")
    return value.strip()


def _build_room_catalog(
    *,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    binding: Mapping[str, Any],
    source_profile_path: Path,
    visual_profile_path: Path,
    relative_heights: tuple[tuple[str, float], ...],
    meters_per_pixel: float,
    camera_spacing_m: float,
    minimum_clearance_m: float,
    room_output: Path,
) -> dict[str, Any]:
    polygon = metadata.get("room_polygon_xy_m")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise KujialeCameraCatalogError("scene metadata lacks room_polygon_xy_m")
    floor = metadata.get("floor_z_m", 0.0)
    if isinstance(floor, bool) or not isinstance(floor, (int, float)):
        raise KujialeCameraCatalogError("metadata.floor_z_m must be finite")
    floor = float(floor)
    if not math.isfinite(floor):
        raise KujialeCameraCatalogError("metadata.floor_z_m must be finite")
    obstacles, obstacle_statistics = build_kujiale_camera_obstacles(metadata)
    _pathfinder, obstacle_map = build_polygon_raster_obstacle_map(
        polygon_xz_m=polygon,
        rigid_obstacles=obstacles,
        floor_height_m=floor,
        meters_per_pixel=meters_per_pixel,
        padding_m=max(
            meters_per_pixel * 2.0,
            minimum_clearance_m + meters_per_pixel,
        ),
        minimum_clearance_m=minimum_clearance_m,
        authority=AUTHORITY,
        claim_boundary=CLAIM_BOUNDARY,
    )
    absolute_heights = tuple(
        (height_id, floor + relative_height)
        for height_id, relative_height in relative_heights
    )
    try:
        placements = tuple(
            iter_feasible_camera_placements(
                obstacle_map,
                camera_center_heights_m=absolute_heights,
                minimum_navmesh_clearance_m=0.0,
                minimum_rigid_clearance_m=0.0,
                sample_spacing_m=camera_spacing_m,
                yaw_degrees=DEFAULT_YAW_DEGREES,
            )
        )
    except CameraPlacementError as exc:
        raise KujialeCameraCatalogError(str(exc)) from exc
    if not placements:
        raise KujialeCameraCatalogError(
            f"room {binding['room_id']} has no CPU camera-center placements"
        )

    height_above_floor = dict(relative_heights)
    records = []
    for index, placement in enumerate(placements, start=1):
        records.append(
            {
                "placement_id": (
                    f"{binding['room_scope']}_camera_placement_{index:06d}"
                ),
                **placement.to_record(),
                "camera_height_above_floor_m": height_above_floor[
                    placement.height_id
                ],
                "room_instance_id": binding["room_id"],
                "room_scope": binding["room_scope"],
                "room_type_annotation": _room_annotation(metadata),
                "semantic_target": None,
                "source_path_join_status": "pending",
                "native_dynamic_visibility_status": "pending",
                "difficulty_status": "not_evaluated",
            }
        )
    floor_positions = {
        (
            placement.raster_pixel_rc,
            placement.position_m[0],
            placement.position_m[2],
        )
        for placement in placements
    }
    navmesh = np.asarray(obstacle_map.binary_navmesh, dtype=np.uint8)
    sampled_pixels = np.asarray(
        sorted({placement.raster_pixel_rc for placement in placements}),
        dtype=np.int32,
    )
    room_output.mkdir()
    np.savez_compressed(
        room_output / "camera_free_space_raster.npz",
        free_space_mask=navmesh,
        bounds_m=np.asarray(obstacle_map.bounds_m, dtype=np.float64),
        floor_height_m=np.asarray(floor, dtype=np.float64),
        sampled_pixels_rc=sampled_pixels,
    )
    catalog = {
        "artifact_status": "complete_target_agnostic_cpu_camera_placement_catalog",
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification_claim": False,
        "candidate_filtering": "none",
        "studio_used": False,
        "room_id": binding["room_id"],
        "room_scope": binding["room_scope"],
        "room_type_annotation": _room_annotation(metadata),
        "room_polygon_role": "geometric_region_boundary_only",
        "scene_binding": dict(binding),
        "inputs": {
            "scene_metadata": str(metadata_path),
            "source_profile": str(source_profile_path),
            "visual_profile": str(visual_profile_path),
        },
        "authority": AUTHORITY,
        "claim_boundary": CLAIM_BOUNDARY,
        "obstacle_map": obstacle_map.summary(),
        "obstacle_statistics": obstacle_statistics,
        "meters_per_pixel": meters_per_pixel,
        "camera_position_sampling_spacing_m": camera_spacing_m,
        "minimum_camera_center_clearance_m": minimum_clearance_m,
        "camera_height_variants_above_floor_m": dict(relative_heights),
        "yaw_degrees": list(DEFAULT_YAW_DEGREES),
        "free_space_raster_pixel_count": int(np.count_nonzero(navmesh)),
        "sampled_floor_position_count": len(floor_positions),
        "placement_count": len(records),
        "placements": records,
        "source_path_join_status": "pending",
        "native_ue_spear_certification_status": "pending",
        "m3_status": "not_run",
        "rlr_status": "not_run",
        "mini_pilot_status": "pending",
    }
    write_json(room_output / "camera_placement_catalog.json", catalog)
    return {
        "room_id": binding["room_id"],
        "room_scope": binding["room_scope"],
        "room_type_annotation": _room_annotation(metadata),
        "free_space_raster_pixel_count": catalog["free_space_raster_pixel_count"],
        "sampled_floor_position_count": catalog["sampled_floor_position_count"],
        "placement_count": catalog["placement_count"],
        "catalog": f"{binding['room_scope']}/camera_placement_catalog.json",
        "free_space_raster": f"{binding['room_scope']}/camera_free_space_raster.npz",
    }


def run(args: argparse.Namespace) -> Path:
    metadata_paths = tuple(
        _required_file(path, owner="--scene-metadata")
        for path in args.scene_metadata
    )
    source_profile_path = _required_file(
        args.source_profile, owner="--source-profile"
    )
    visual_profile_path = _required_file(
        args.visual_profile, owner="--visual-profile"
    )
    output, policy = _fresh_output(args.output)
    if not metadata_paths:
        raise KujialeCameraCatalogError("at least one --scene-metadata is required")
    source_profile = load_json(source_profile_path)
    visual_profile = load_json(visual_profile_path)
    prepared = []
    seen_room_ids = set()
    seen_scopes = set()
    for metadata_path in metadata_paths:
        metadata = load_json(metadata_path)
        binding = validate_kujiale_scene_binding(
            metadata=metadata,
            source_profile=source_profile,
            visual_profile=visual_profile,
        )
        if binding["room_id"] in seen_room_ids or binding["room_scope"] in seen_scopes:
            raise KujialeCameraCatalogError(
                "scene metadata room_id and room_scope values must be unique"
            )
        seen_room_ids.add(binding["room_id"])
        seen_scopes.add(binding["room_scope"])
        prepared.append((metadata_path, metadata, binding))
    prepared.sort(key=lambda item: item[2]["room_scope"].encode("utf-8"))

    heights = _camera_heights(args.camera_heights)
    meters_per_pixel = _positive(args.meters_per_pixel, owner="meters_per_pixel")
    spacing = _positive(args.camera_spacing_m, owner="camera_spacing_m")
    clearance = _nonnegative(
        args.minimum_clearance_m, owner="minimum_clearance_m"
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        summaries = [
            _build_room_catalog(
                metadata_path=metadata_path,
                metadata=metadata,
                binding=binding,
                source_profile_path=source_profile_path,
                visual_profile_path=visual_profile_path,
                relative_heights=heights,
                meters_per_pixel=meters_per_pixel,
                camera_spacing_m=spacing,
                minimum_clearance_m=clearance,
                room_output=staging / binding["room_scope"],
            )
            for metadata_path, metadata, binding in prepared
        ]
        write_json(
            staging / "delivery.json",
            {
                "artifact_status": "complete_multiroom_target_agnostic_cpu_catalogs",
                "research_only": True,
                "episode_counted": False,
                "formal_dataset_count": 0,
                "qualification_claim": False,
                "candidate_filtering": "none",
                "studio_used": False,
                "authority": AUTHORITY,
                "claim_boundary": CLAIM_BOUNDARY,
                "room_instance_count": len(summaries),
                "total_free_space_raster_pixel_count": sum(
                    item["free_space_raster_pixel_count"] for item in summaries
                ),
                "total_sampled_floor_position_count": sum(
                    item["sampled_floor_position_count"] for item in summaries
                ),
                "total_placement_count": sum(
                    item["placement_count"] for item in summaries
                ),
                "camera_height_variants_above_floor_m": dict(heights),
                "yaw_degrees": list(DEFAULT_YAW_DEGREES),
                "room_instances": summaries,
                "native_package_status": "pending",
                "native_ue_spear_certification_status": "pending",
                "m3_status": "not_run",
                "rlr_status": "not_run",
                "mini_pilot_status": "pending",
            },
        )
        (staging / "README.md").write_text(
            "# Kujiale multi-room target-agnostic CPU camera catalogs\n\n"
            "Each room-instance catalog retains every sampled polygon/ground-"
            "footprint free-space camera center crossed with every declared height "
            "and yaw 0,30,...,330 degrees. Room type is annotation only. No "
            "source/path, 60 cm, side, visibility, composition, difficulty or "
            "selection filter is applied. Native UE/SPEAR, M3, RLR and mini-pilot "
            "evidence remains pending.\n",
            encoding="utf-8",
        )
        try:
            atomic_publish_directory(policy, staging, output)
        except (FileExistsError, PathPolicyError) as exc:
            raise KujialeCameraCatalogError(str(exc)) from exc
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-metadata", nargs="+", required=True, type=Path)
    parser.add_argument("--source-profile", required=True, type=Path)
    parser.add_argument("--visual-profile", required=True, type=Path)
    parser.add_argument(
        "--camera-height",
        action="append",
        dest="camera_heights",
        metavar="HEIGHT_ID=METERS",
    )
    parser.add_argument("--meters-per-pixel", type=float, default=0.05)
    parser.add_argument("--camera-spacing-m", type=float, default=0.25)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
