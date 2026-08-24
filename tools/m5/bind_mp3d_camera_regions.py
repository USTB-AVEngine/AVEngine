#!/usr/bin/env python3
"""Publish an additive MP3D camera-to-house-region membership sidecar."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from avengine.contracts.json_io import load_json, write_json
from avengine.m5.mp3d_regions import (
    MP3DRegionError,
    bind_floor_points_to_regions,
    parse_mp3d_house,
)
from avengine.security import (
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
)


_CATALOG_STATUS = "complete_target_agnostic_camera_placement_catalog"
_FLOOR_AUTHORITY = "live_habitat_pathfinder_declared_navmesh_export"
_PLACEMENT_STATUS = "declared_floor_position_crossed_with_height_and_yaw"
_MEMBERSHIP_STATUSES = ("unique", "boundary", "unknown", "ambiguous")


class MP3DCameraRegionBindingError(ValueError):
    """Inputs cannot produce a truthful additive region sidecar."""


def _required_file(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{owner} is not a file: {resolved}")
    return resolved


def _fresh_output(path: Path) -> tuple[Path, WorkspacePathPolicy]:
    raw = path.expanduser()
    if not raw.is_absolute():
        raise MP3DCameraRegionBindingError("--output must be an absolute path")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise MP3DCameraRegionBindingError(
            f"--output parent does not exist: {raw.parent}"
        ) from exc
    output = parent / raw.name
    if os.path.lexists(output):
        raise MP3DCameraRegionBindingError(f"refusing to replace output: {output}")
    try:
        policy = WorkspacePathPolicy.from_roots((parent,))
    except PathPolicyError as exc:
        raise MP3DCameraRegionBindingError(str(exc)) from exc
    return output, policy


def _finite_float(value: object, *, owner: str) -> float:
    if isinstance(value, bool):
        raise MP3DCameraRegionBindingError(f"{owner} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DCameraRegionBindingError(f"{owner} must be a finite number") from exc
    if not math.isfinite(result):
        raise MP3DCameraRegionBindingError(f"{owner} must be a finite number")
    return result


def _point3(value: object, *, owner: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MP3DCameraRegionBindingError(f"{owner} must contain three finite numbers")
    if len(value) != 3:
        raise MP3DCameraRegionBindingError(f"{owner} must contain three finite numbers")
    return tuple(
        _finite_float(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _floor_authority(
    value: Any,
) -> tuple[str, str, tuple[tuple[float, float, float], ...], int]:
    if not isinstance(value, Mapping):
        raise MP3DCameraRegionBindingError("floor artifact must be an object")
    room_id = value.get("room_id")
    authority = value.get("authority")
    raw_samples = value.get("camera_floor_samples_m")
    if (
        not isinstance(room_id, str)
        or not room_id
        or authority != _FLOOR_AUTHORITY
        or value.get("coordinate_space") != "habitat_world_xyz_m"
        or value.get("native_habitat_floor_export") is not True
        or value.get("research_only") is not True
        or value.get("episode_counted") is not False
        or not isinstance(raw_samples, list)
        or not raw_samples
    ):
        raise MP3DCameraRegionBindingError(
            "floor artifact is not one current native Habitat research export"
        )
    samples = tuple(
        _point3(point, owner=f"camera_floor_samples_m[{index}]")
        for index, point in enumerate(raw_samples)
    )
    canonical = tuple(sorted(set(samples)))
    if not canonical:
        raise MP3DCameraRegionBindingError("floor artifact has no canonical samples")
    return room_id, authority, canonical, len(samples)


def _catalog_placements(
    value: Any,
    *,
    room_id: str,
    authority: str,
    floor_samples: tuple[tuple[float, float, float], ...],
    raw_floor_sample_count: int,
) -> tuple[tuple[Mapping[str, Any], ...], dict[int, tuple[str, ...]]]:
    if not isinstance(value, Mapping):
        raise MP3DCameraRegionBindingError("camera catalog must be an object")
    placements = value.get("placements")
    heights = value.get("camera_height_variants_m")
    yaws = value.get("yaw_degrees")
    if (
        value.get("artifact_status") != _CATALOG_STATUS
        or value.get("research_only") is not True
        or value.get("episode_counted") is not False
        or value.get("candidate_filtering") != "none"
        or value.get("room_id") != room_id
        or value.get("floor_position_authority") != authority
        or value.get("input_floor_sample_count") != raw_floor_sample_count
        or value.get("canonical_floor_sample_count") != len(floor_samples)
        or not isinstance(placements, list)
        or not isinstance(heights, Mapping)
        or not isinstance(yaws, list)
    ):
        raise MP3DCameraRegionBindingError(
            "camera catalog does not match the complete unfiltered floor authority"
        )
    height_ids = tuple(sorted(heights))
    if any(not isinstance(value, str) or not value for value in height_ids):
        raise MP3DCameraRegionBindingError("camera height ids must be nonempty strings")
    for height_id in height_ids:
        _finite_float(heights[height_id], owner=f"camera height {height_id}")
    yaw_values = tuple(_finite_float(value, owner="catalog yaw") for value in yaws)
    if len(set(yaw_values)) != len(yaw_values):
        raise MP3DCameraRegionBindingError("catalog yaw values must be unique")
    expected_combinations = {
        (height_id, yaw) for height_id in height_ids for yaw in yaw_values
    }
    if len(expected_combinations) != 36:
        raise MP3DCameraRegionBindingError(
            "camera catalog must declare three heights and twelve yaw values"
        )
    if value.get("placement_count") != len(placements):
        raise MP3DCameraRegionBindingError("camera catalog placement_count drift")

    placement_ids: set[str] = set()
    combinations: dict[int, set[tuple[str, float]]] = defaultdict(set)
    ids_by_floor: dict[int, list[str]] = defaultdict(list)
    records = []
    for ordinal, placement in enumerate(placements):
        owner = f"placements[{ordinal}]"
        if not isinstance(placement, Mapping):
            raise MP3DCameraRegionBindingError(f"{owner} must be an object")
        placement_id = placement.get("placement_id")
        floor_index = placement.get("floor_sample_index")
        height_id = placement.get("height_id")
        if not isinstance(placement_id, str) or not placement_id:
            raise MP3DCameraRegionBindingError(f"{owner} placement_id is invalid")
        if placement_id in placement_ids:
            raise MP3DCameraRegionBindingError("camera placement IDs must be unique")
        placement_ids.add(placement_id)
        if isinstance(floor_index, bool) or not isinstance(floor_index, int):
            raise MP3DCameraRegionBindingError(
                f"{owner} floor_sample_index must be an integer"
            )
        if floor_index < 0 or floor_index >= len(floor_samples):
            raise MP3DCameraRegionBindingError(
                f"{owner} floor_sample_index is out of range"
            )
        floor_position = _point3(
            placement.get("floor_position_m"), owner=f"{owner} floor_position_m"
        )
        if floor_position != floor_samples[floor_index]:
            raise MP3DCameraRegionBindingError(
                f"{owner} floor_sample_index/floor_position_m drift"
            )
        if placement.get("placement_status") != _PLACEMENT_STATUS:
            raise MP3DCameraRegionBindingError(
                f"{owner} lacks declared floor placement authority"
            )
        if not isinstance(height_id, str) or height_id not in height_ids:
            raise MP3DCameraRegionBindingError(f"{owner} height_id is invalid")
        yaw = _finite_float(placement.get("yaw_deg"), owner=f"{owner} yaw_deg")
        combination = (height_id, yaw)
        if combination not in expected_combinations:
            raise MP3DCameraRegionBindingError(
                f"{owner} height/yaw is not catalog-declared"
            )
        if combination in combinations[floor_index]:
            raise MP3DCameraRegionBindingError(
                f"floor sample {floor_index} repeats a height/yaw combination"
            )
        combinations[floor_index].add(combination)
        ids_by_floor[floor_index].append(placement_id)
        records.append(placement)
    if len(records) != len(floor_samples) * 36:
        raise MP3DCameraRegionBindingError(
            "camera catalog must contain exactly 36 placements per floor sample"
        )
    for floor_index in range(len(floor_samples)):
        if combinations[floor_index] != expected_combinations:
            raise MP3DCameraRegionBindingError(
                f"floor sample {floor_index} does not have all 36 placements"
            )
    return tuple(records), {
        index: tuple(ids_by_floor[index]) for index in range(len(floor_samples))
    }


def run(args: argparse.Namespace) -> Path:
    house_path = _required_file(args.house, owner="--house")
    floor_path = _required_file(
        args.camera_floor_samples, owner="--camera-floor-samples"
    )
    catalog_path = _required_file(args.camera_catalog, owner="--camera-catalog")
    output, policy = _fresh_output(args.output)
    room_id, authority, floor_samples, raw_floor_count = _floor_authority(
        load_json(floor_path)
    )
    catalog = load_json(catalog_path)
    placements, placement_ids_by_floor = _catalog_placements(
        catalog,
        room_id=room_id,
        authority=authority,
        floor_samples=floor_samples,
        raw_floor_sample_count=raw_floor_count,
    )
    try:
        plan = parse_mp3d_house(house_path)
        memberships = bind_floor_points_to_regions(
            floor_samples,
            plan,
            boundary_descriptor_threshold_m=args.boundary_descriptor_threshold_m,
        )
    except MP3DRegionError as exc:
        raise MP3DCameraRegionBindingError(str(exc)) from exc
    floor_records = []
    by_floor = {}
    for membership in memberships:
        record = membership.to_record(plan)
        record["placement_count"] = len(
            placement_ids_by_floor[membership.floor_sample_index]
        )
        record["placement_ids"] = list(
            placement_ids_by_floor[membership.floor_sample_index]
        )
        floor_records.append(record)
        by_floor[membership.floor_sample_index] = record

    placement_records = []
    for placement in placements:
        floor_record = by_floor[int(placement["floor_sample_index"])]
        placement_records.append(
            {
                "placement_id": placement["placement_id"],
                "floor_sample_index": placement["floor_sample_index"],
                "floor_position_m": placement["floor_position_m"],
                "height_id": placement["height_id"],
                "position_m": placement["position_m"],
                "yaw_deg": placement["yaw_deg"],
                "membership_status": floor_record["membership_status"],
                "member_region_indices": floor_record["member_region_indices"],
                "member_region_instance_ids": floor_record[
                    "member_region_instance_ids"
                ],
                "member_regions": floor_record["member_regions"],
                "primary_region_index": floor_record["primary_region_index"],
                "primary_region_instance_id": floor_record[
                    "primary_region_instance_id"
                ],
                "primary_region_category_code": floor_record[
                    "primary_region_category_code"
                ],
                "primary_region_category_name": floor_record[
                    "primary_region_category_name"
                ],
                "distance_to_nearest_polygon_boundary_m": floor_record[
                    "distance_to_nearest_polygon_boundary_m"
                ],
                "on_polygon_boundary": floor_record["on_polygon_boundary"],
                "near_boundary_descriptor": floor_record["near_boundary_descriptor"],
                "candidate_legality_effect": "none",
            }
        )

    floor_status_counts = Counter(
        record["membership_status"] for record in floor_records
    )
    placement_status_counts = Counter(
        record["membership_status"] for record in placement_records
    )
    member_floor_counts = Counter(
        region_index
        for record in floor_records
        for region_index in record["member_region_indices"]
    )
    member_placement_counts = Counter(
        region_index
        for record in placement_records
        for region_index in record["member_region_indices"]
    )
    region_records = []
    for region in plan.regions:
        item = region.label_record()
        item.update(
            {
                "habitat_bbox_min_xyz_m": list(region.habitat_bbox_min_xyz_m),
                "habitat_bbox_max_xyz_m": list(region.habitat_bbox_max_xyz_m),
                "floor_surface_indices": [
                    polygon.surface_index for polygon in region.floor_polygons
                ],
                "floor_polygon_count": len(region.floor_polygons),
                "floor_sample_count": member_floor_counts[region.region_index],
                "placement_count": member_placement_counts[region.region_index],
            }
        )
        region_records.append(item)

    threshold = _finite_float(
        args.boundary_descriptor_threshold_m,
        owner="--boundary-descriptor-threshold-m",
    )
    sidecar = {
        "artifact_status": "complete_additive_mp3d_camera_region_membership_sidecar",
        "research_only": True,
        "episode_counted": False,
        "formal_release_status": "not_run",
        "candidate_filtering": "none",
        "candidate_legality_effect": "none",
        "studio_used": False,
        "room_id": room_id,
        "inputs": {
            "house": str(house_path),
            "camera_floor_samples": str(floor_path),
            "camera_catalog": str(catalog_path),
        },
        "coordinate_transform": {
            "profile": "mp3d_z_up_y_front_to_habitat",
            "source": "MP3D +Z up/+Y front",
            "target": "Habitat +Y up/-Z front",
            "mapping": "[x,y,z] -> [x,z,-y]",
        },
        "membership_axes": "habitat_xz_with_region_bbox_y_membership",
        "source_house_id": plan.house_id,
        "house_record_counts": {
            "declared": dict(plan.declared_counts),
            "parsed_panorama_records": plan.parsed_panorama_count,
            "parsed_portal_records": plan.parsed_portal_count,
        },
        "boundary_descriptor_threshold_m": threshold,
        "boundary_descriptor_is_legality_gate": False,
        "floor_sample_count": len(floor_records),
        "placement_count": len(placement_records),
        "placements_per_floor_sample": 36,
        "region_instance_count": len(region_records),
        "floor_membership_status_counts": {
            status: floor_status_counts[status] for status in _MEMBERSHIP_STATUSES
        },
        "placement_membership_status_counts": {
            status: placement_status_counts[status] for status in _MEMBERSHIP_STATUSES
        },
        "near_boundary_floor_sample_count": sum(
            bool(record["near_boundary_descriptor"]) for record in floor_records
        ),
        "regions": region_records,
        "floor_sample_memberships": floor_records,
        "placement_memberships": placement_records,
        "claim_boundary": (
            "additive static MP3D .house floor-polygon membership only; room labels "
            "do not generate candidates, unknown/ambiguous/boundary points are "
            "retained, and no camera legality, native visibility, audio, RIR, "
            "viewpoint certification, ranking or formal Episode is claimed"
        ),
    }

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        write_json(staging / "camera_region_sidecar.json", sidecar)
        (staging / "README.md").write_text(
            "# MP3D camera region sidecar\n\n"
            "This additive artifact binds every existing floor point and each of "
            "its 36 camera placements to static `.house` floor polygons. It does "
            "not modify or filter the source catalog. Unknown, ambiguous and "
            "boundary memberships are retained. The 5 cm boundary flag is a "
            "descriptor, never a legality gate.\n",
            encoding="utf-8",
        )
        try:
            atomic_publish_directory(policy, staging, output)
        except (FileExistsError, PathPolicyError) as exc:
            raise MP3DCameraRegionBindingError(str(exc)) from exc
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", required=True, type=Path)
    parser.add_argument("--camera-floor-samples", required=True, type=Path)
    parser.add_argument("--camera-catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--boundary-descriptor-threshold-m", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
