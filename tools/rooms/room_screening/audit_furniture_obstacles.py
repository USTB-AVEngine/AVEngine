#!/usr/bin/env python3
"""Resumable, provisional HM3D furniture/navmesh overlap audit.

This tool writes derived diagnostics only. It never edits source assets, review
verdicts, the review site, or the authoritative Plan/Rules documents.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import compute_semantic_region_candidates as base
from .geometry import (
    conservative_projected_footprint,
    load_semantic_ground_and_instances,
    projected_unmapped_surface_union,
    shape_preserving_projected_footprint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "tmp/room_screening/furniture_area_audit"
TOLERANCE_RULE = "one serialized navmesh cell area: cell_size_m ** 2"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def script_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("geometry.py"),
        Path(__file__).with_name("compute_semantic_region_candidates.py"),
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def navmesh_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def area(value) -> float:
    return float(value.area) if value is not None and not value.is_empty else 0.0


def issue_set(raw_value) -> set[str]:
    if not raw_value:
        return set()
    if isinstance(raw_value, list):
        return {str(x) for x in raw_value if x}
    return {part.strip() for part in str(raw_value).split(";") if part.strip()}


def object_bbox_intersects(instance: dict, candidate_bounds, low: float, high: float) -> bool:
    vertices = instance["triangles"].reshape((-1, 3))
    ymin, ymax = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    if ymax < low or ymin > high:
        return False
    xmin, xmax = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    bxmin, bzmin, bxmax, bzmax = candidate_bounds
    return not (xmax < bxmin or xmin > bxmax or zmax < bzmin or zmin > bzmax)


def fingerprint_for_scene(scene_record: dict, scene_dir: Path, code_hash: str, geometry_policy: dict) -> dict:
    scene_id = scene_record["house"].split("_", 3)[-1]
    semantic_glb = scene_dir / f"{scene_id}.semantic.glb"
    semantic_txt = scene_dir / f"{scene_id}.semantic.txt"
    navmesh_path = scene_dir / f"{scene_id}.basis.navmesh"
    return {
        "semantic_glb_size": semantic_glb.stat().st_size,
        "semantic_txt_size": semantic_txt.stat().st_size,
        "navmesh_size": navmesh_path.stat().st_size,
        "navmesh_sha256": navmesh_hash(navmesh_path),
        "implementation_sha256": code_hash,
        "geometry_policy": geometry_policy,
    }


def process_house(
    hs,
    scene_record: dict,
    raw_scene: dict,
    dataset_root: Path,
    footprint_method: str = "convex",
    subtract_cross_region_blockers: bool = False,
) -> dict:
    house = scene_record["house"]
    split, index, scene_id = house.split("_", 3)[1:]
    scene_dir = dataset_root / split / f"{index}-{scene_id}"
    settings = scene_record["navmesh"]["navmesh_settings"]
    agent_height = float(settings["agent_height"])
    agent_radius = float(settings["agent_radius"])
    cell_size = float(settings["cell_size"])
    tolerance_m2 = max(cell_size * cell_size, 1e-9)
    level_gap = 2.0 * float(settings["cell_height"])

    (
        room_ground,
        mesh_regions,
        instances,
        unmapped_triangles,
        unmapped_face_count,
        unmapped_colour_count,
        semantic_hashes,
        label_count,
        total_triangle_count,
    ) = load_semantic_ground_and_instances(scene_dir, scene_id)
    nav_path = scene_dir / f"{scene_id}.basis.navmesh"
    pf, nav_polygons, nav_ys = base.navmesh_triangles(hs, nav_path)
    nav_levels = base.cluster_levels(nav_ys, level_gap)
    nav_membership = np.argmin(np.abs(nav_ys[:, None] - np.asarray(nav_levels)[None, :]), axis=1) if nav_levels else np.zeros(len(nav_ys), dtype=int)
    nav_geometries = []
    for level_index, level_y in enumerate(nav_levels):
        polygons = [poly for poly, membership in zip(nav_polygons, nav_membership) if membership == level_index]
        nav_geometries.append(unary_union(polygons) if polygons else Polygon())

    raw_regions = {int(row["region_id"]): row for row in raw_scene.get("regions", []) if row.get("region_id") is not None}
    candidate_rows = []
    object_rows = []
    category_counts = Counter(item["category"] for item in instances if item["role"] == "blocker")
    review_category_counts = Counter(item["category"] for item in instances if item["role"] == "review")
    role_counts = Counter(item["role"] for item in instances)

    for region_id in sorted(raw_regions):
        source_row = raw_regions[region_id]
        ground_clusters = base.collect_clusters(room_ground.get(region_id, []), level_gap)
        floor_index = source_row.get("floor_id_candidate")
        issue_codes = issue_set(source_row.get("issue_code"))
        if floor_index is None or int(floor_index) >= len(ground_clusters):
            candidate_rows.append({
                "house": house,
                "region_id": region_id,
                "room_label": f"R{region_id}",
                "floor_id_candidate": floor_index,
                "floor_y_m": None,
                "raw_candidate_area_m2": source_row.get("candidate_semantic_region_nav_projected_area_m2"),
                "semantic_ground_projected_area_m2": None,
                "semantic_ground_outside_navmesh_area_m2": None,
                "semantic_ground_navmesh_coverage_fraction": None,
                "trial_adjusted_candidate_area_m2": None,
                "blocker_direct_overlap_area_m2": None,
                "blocker_clearance_overlap_area_m2": None,
                "blocker_hull_residual_area_m2": None,
                "unmapped_semantic_candidate_overlap_area_m2": None,
                "unknown_object_clearance_overlap_area_m2": None,
                "adjustment_area_m2": None,
                "navmesh_cell_area_tolerance_m2": tolerance_m2,
                "audit_status": "unassessable",
                "issue_codes": sorted(issue_codes | {"no_semantic_ground_faces"}),
                "blocker_instances": [],
                "review_instances": [],
                "mismatch_geometry_wkt": None,
                "review_geometry_wkt": None,
            })
            continue

        floor_index = int(floor_index)
        cluster = ground_clusters[floor_index]
        floor_y = float(cluster["median_y_m"])
        semantic_ground_area = area(cluster["ground_union"])
        nav_index = source_row.get("navmesh_floor_cluster_index")
        if nav_index is None or int(nav_index) >= len(nav_geometries):
            candidate_geometry = None
            semantic_ground_outside_navmesh = None
            semantic_ground_outside_navmesh_geometry = None
        else:
            nav_geometry = nav_geometries[int(nav_index)]
            candidate_geometry = cluster["ground_union"].intersection(nav_geometry)
            semantic_ground_outside_navmesh_geometry = cluster["ground_union"].difference(nav_geometry)
            semantic_ground_outside_navmesh = area(semantic_ground_outside_navmesh_geometry)
        raw_area = area(candidate_geometry)
        semantic_ground_coverage = (
            float(raw_area / semantic_ground_area) if semantic_ground_area > 0 else None
        )
        raw_json_area = source_row.get("candidate_semantic_region_nav_projected_area_m2")
        if raw_json_area is not None and abs(float(raw_json_area) - raw_area) > max(1e-5, tolerance_m2 / 100):
            issue_codes.add("candidate_recompute_disagreement")

        if candidate_geometry is None or candidate_geometry.is_empty:
            candidate_rows.append({
                "house": house,
                "region_id": region_id,
                "room_label": f"R{region_id}",
                "floor_id_candidate": floor_index,
                "floor_y_m": floor_y,
                "raw_candidate_area_m2": raw_json_area,
                "semantic_ground_projected_area_m2": semantic_ground_area,
                "semantic_ground_outside_navmesh_area_m2": semantic_ground_outside_navmesh,
                "semantic_ground_navmesh_coverage_fraction": semantic_ground_coverage,
                "semantic_ground_outside_navmesh_geometry_wkt": (
                    semantic_ground_outside_navmesh_geometry.wkt
                    if semantic_ground_outside_navmesh_geometry is not None
                    and not semantic_ground_outside_navmesh_geometry.is_empty else None
                ),
                "trial_adjusted_candidate_area_m2": None,
                "blocker_direct_overlap_area_m2": None,
                "blocker_clearance_overlap_area_m2": None,
                "blocker_hull_residual_area_m2": None,
                "unmapped_semantic_candidate_overlap_area_m2": None,
                "unknown_object_clearance_overlap_area_m2": None,
                "adjustment_area_m2": None,
                "navmesh_cell_area_tolerance_m2": tolerance_m2,
                "audit_status": "unassessable",
                "issue_codes": sorted(issue_codes | {"empty_room_navmesh_intersection"}),
                "blocker_instances": [],
                "review_instances": [],
                "mismatch_geometry_wkt": None,
                "review_geometry_wkt": None,
            })
            continue

        low, high = floor_y, floor_y + agent_height
        query_bounds = candidate_geometry.buffer(agent_radius).bounds
        blocker_masks = []
        blocker_direct_masks = []
        footprint_uncertainty_masks = []
        blocker_details = []
        review_masks = []
        review_items = []
        for instance in instances:
            if not object_bbox_intersects(instance, query_bounds, low, high):
                continue
            convex_reference = conservative_projected_footprint(
                instance["triangles"], floor_y, agent_height
            )
            if footprint_method == "shape_preserving":
                footprint = shape_preserving_projected_footprint(
                    instance["triangles"], floor_y, agent_height
                )
                if not convex_reference.is_empty and not footprint.is_empty:
                    convex_clearance = convex_reference.buffer(agent_radius, quad_segs=8)
                    shape_clearance = footprint.buffer(agent_radius, quad_segs=8)
                    footprint_uncertainty_masks.append(
                        convex_clearance.difference(shape_clearance).intersection(candidate_geometry)
                    )
            else:
                footprint = convex_reference
            if footprint.is_empty:
                continue
            expanded = footprint.buffer(agent_radius, quad_segs=8)
            expanded_overlap = expanded.intersection(candidate_geometry)
            clearance_overlap = area(expanded_overlap)
            if clearance_overlap <= 1e-10:
                continue
            direct_overlap = area(footprint.intersection(candidate_geometry))
            same_region = instance.get("region_id") == region_id
            unassigned_region = instance.get("region_id") is None or int(instance.get("region_id", -1)) < 0
            detail = {
                "instance_id": instance.get("instance_id"),
                "rgb": instance.get("rgb"),
                "category": instance.get("category"),
                "role": instance.get("role"),
                "semantic_region_id": instance.get("region_id"),
                "same_region": same_region,
                "unassigned_region": unassigned_region,
                "face_count": int(instance.get("face_count", 0)),
                "footprint_area_m2": area(footprint),
                "footprint_centroid_xz_m": [float(footprint.centroid.x), float(footprint.centroid.y)],
                "footprint_centroid_in_raw_candidate": bool(candidate_geometry.covers(footprint.centroid)),
                "direct_overlap_area_m2": direct_overlap,
                "direct_overlap_fraction_of_footprint": (
                    float(direct_overlap / area(footprint)) if area(footprint) > 0 else None
                ),
                "clearance_overlap_area_m2": clearance_overlap,
                "footprint_bounds_xz_m": [float(v) for v in footprint.bounds],
            }
            if instance["role"] == "blocker" and (
                same_region or unassigned_region or subtract_cross_region_blockers
            ):
                blocker_masks.append(expanded_overlap)
                blocker_direct_masks.append(footprint.intersection(candidate_geometry))
                if clearance_overlap > tolerance_m2 / 10:
                    blocker_details.append(detail)
                    object_rows.append({"house": house, "region_id": region_id, "floor_id_candidate": floor_index, **detail})
            else:
                review_masks.append(expanded_overlap)
                review_items.append((instance, footprint, direct_overlap, same_region, unassigned_region))

        blocker_union = unary_union(blocker_masks) if blocker_masks else Polygon()
        blocker_direct_union = unary_union(blocker_direct_masks) if blocker_direct_masks else Polygon()
        footprint_uncertainty = (
            unary_union(footprint_uncertainty_masks)
            if footprint_uncertainty_masks else Polygon()
        )
        review_union_raw = unary_union(review_masks) if review_masks else Polygon()
        review_union = review_union_raw.difference(blocker_union)
        unmapped_surface = projected_unmapped_surface_union(
            unmapped_triangles,
            floor_y,
            agent_height,
            query_bounds,
        )
        unmapped_overlap = (
            unmapped_surface.buffer(agent_radius, quad_segs=8).intersection(candidate_geometry)
            if not unmapped_surface.is_empty else Polygon()
        )
        unmapped_overlap_area = area(unmapped_overlap)
        if unmapped_overlap_area > tolerance_m2:
            issue_codes.add("unmapped_semantic_geometry_overlap")
            review_union = unary_union([review_union, unmapped_overlap])
        review_details = []
        for instance, footprint, direct_overlap, same_region, unassigned_region in review_items:
            expanded_overlap = footprint.buffer(agent_radius, quad_segs=8).intersection(candidate_geometry)
            residual_overlap = expanded_overlap.difference(blocker_union)
            residual_area = area(residual_overlap)
            if residual_area <= tolerance_m2 / 10:
                continue
            detail = {
                "instance_id": instance.get("instance_id"),
                "rgb": instance.get("rgb"),
                "category": instance.get("category"),
                "role": instance.get("role"),
                "semantic_region_id": instance.get("region_id"),
                "same_region": same_region,
                "unassigned_region": unassigned_region,
                "face_count": int(instance.get("face_count", 0)),
                "footprint_area_m2": area(footprint),
                "footprint_centroid_xz_m": [float(footprint.centroid.x), float(footprint.centroid.y)],
                "footprint_centroid_in_raw_candidate": bool(candidate_geometry.covers(footprint.centroid)),
                "direct_overlap_area_m2": direct_overlap,
                "direct_overlap_fraction_of_footprint": (
                    float(direct_overlap / area(footprint)) if area(footprint) > 0 else None
                ),
                "clearance_overlap_area_m2": residual_area,
                "footprint_bounds_xz_m": [float(v) for v in footprint.bounds],
                "reason": "cross_region_object_overlap" if not same_region and not unassigned_region else "unclassified_object_overlap",
            }
            review_details.append(detail)
            object_rows.append({"house": house, "region_id": region_id, "floor_id_candidate": floor_index, **detail})
        blocker_clearance_area = area(blocker_union)
        blocker_direct_area = area(blocker_direct_union)
        footprint_uncertainty_area = area(footprint_uncertainty)
        review_area = area(review_union)
        mismatch = blocker_clearance_area > tolerance_m2
        if mismatch:
            issue_codes.add("navmesh_static_obstacle_mismatch")
            adjusted_geometry = candidate_geometry.difference(blocker_union)
        else:
            adjusted_geometry = candidate_geometry
        if review_area > tolerance_m2:
            issue_codes.add("semantic_region_incomplete")
        if footprint_uncertainty_area > tolerance_m2:
            issue_codes.add("footprint_extent_uncertainty")
        if semantic_ground_outside_navmesh is not None and semantic_ground_outside_navmesh > tolerance_m2:
            issue_codes.add("semantic_ground_outside_navmesh")
        if any(obj["footprint_area_m2"] > raw_area for obj in blocker_details):
            issue_codes.add("obstacle_hull_exceeds_candidate_area")
        if any(not obj["same_region"] and not obj["unassigned_region"] for obj in blocker_details):
            issue_codes.add("cross_region_blocker_subtracted")
        if any(obj["reason"] == "cross_region_object_overlap" for obj in review_details):
            issue_codes.add("semantic_region_mismatch")

        row_area = area(adjusted_geometry)
        confirmed_geometry = adjusted_geometry.difference(footprint_uncertainty).difference(review_union)
        confirmed_area = area(confirmed_geometry)
        # Avoid a second overlay of (candidate \ clearance) ∩ direct footprint:
        # the installed GEOS build can misclassify that expression for some
        # MultiPolygon scan boundaries. The equivalent coverage test is the
        # direct footprint mask minus the clearance mask; both were already
        # clipped to this candidate before union/subtraction.
        blocker_hull_residual_area = area(blocker_direct_union.difference(blocker_union))
        if mismatch and blocker_hull_residual_area > tolerance_m2 / 100:
            issue_codes.add("blocker_footprint_not_fully_removed")
        if row_area > raw_area + 1e-8 or row_area < -1e-8:
            issue_codes.add("area_invariant_failed")
        source_codes = issue_set(source_row.get("issue_code"))
        unassessable_source_codes = {"floor_alignment_needs_review", "no_navmesh_floor_cluster"}
        if issue_codes & unassessable_source_codes:
            audit_status = "unassessable"
        elif (
            review_area > tolerance_m2
            or unmapped_overlap_area > tolerance_m2
            or footprint_uncertainty_area > tolerance_m2
            or (semantic_ground_outside_navmesh is not None and semantic_ground_outside_navmesh > tolerance_m2)
            or "obstacle_hull_exceeds_candidate_area" in issue_codes
        ):
            audit_status = "needs_manual_review"
        elif mismatch:
            audit_status = "corrected_provisional"
        else:
            audit_status = "provisionally_clear"

        candidate_rows.append({
            "house": house,
            "region_id": region_id,
            "room_label": f"R{region_id}",
            "floor_id_candidate": floor_index,
            "floor_y_m": floor_y,
            "navmesh_floor_cluster_index": nav_index,
            "agent_height_m": agent_height,
            "agent_radius_m": agent_radius,
            "navmesh_cell_size_m": cell_size,
            "navmesh_cell_area_tolerance_m2": tolerance_m2,
            "tolerance_rule": TOLERANCE_RULE,
            "raw_candidate_area_m2": raw_area,
            "semantic_ground_projected_area_m2": semantic_ground_area,
            "semantic_ground_outside_navmesh_area_m2": semantic_ground_outside_navmesh,
            "semantic_ground_navmesh_coverage_fraction": semantic_ground_coverage,
            "semantic_ground_outside_navmesh_geometry_wkt": (
                semantic_ground_outside_navmesh_geometry.wkt
                if semantic_ground_outside_navmesh_geometry is not None
                and not semantic_ground_outside_navmesh_geometry.is_empty else None
            ),
            "source_candidate_area_m2": raw_json_area,
            "blocker_direct_overlap_area_m2": blocker_direct_area,
            "blocker_clearance_overlap_area_m2": blocker_clearance_area,
            "blocker_hull_residual_area_m2": blocker_hull_residual_area,
            "unmapped_semantic_candidate_overlap_area_m2": unmapped_overlap_area,
            "unknown_object_clearance_overlap_area_m2": review_area,
            "footprint_extent_uncertainty_area_m2": footprint_uncertainty_area,
            "confirmed_candidate_area_m2": confirmed_area,
            "adjustment_area_m2": blocker_clearance_area if mismatch else 0.0,
            "trial_adjusted_candidate_area_m2": row_area,
            "obstacle_instance_count": len(blocker_details),
            "cross_region_blocker_instance_count": sum(
                not obj["same_region"] and not obj["unassigned_region"] for obj in blocker_details
            ),
            "obstacle_categories": sorted({obj["category"] for obj in blocker_details}),
            "blocker_instances": blocker_details,
            "review_instances": review_details,
            "audit_status": audit_status,
            "issue_codes": sorted(issue_codes),
            "mismatch_geometry_wkt": blocker_union.wkt if mismatch else None,
            "footprint_extent_uncertainty_geometry_wkt": (
                footprint_uncertainty.wkt if footprint_uncertainty_area > tolerance_m2 / 10 else None
            ),
            "review_geometry_wkt": review_union.wkt if review_area > tolerance_m2 else None,
        })

    return {
        "house": house,
        "status": "complete",
        "scene_dir": str(scene_dir),
        "semantic_hashes": semantic_hashes,
        "navmesh_sha256": navmesh_hash(nav_path),
        "semantic_label_count": label_count,
        "semantic_region_ids": sorted(mesh_regions),
        "unmapped_semantic_face_count": int(unmapped_face_count),
        "unmapped_semantic_colour_count": int(unmapped_colour_count),
        "semantic_total_triangle_count": int(total_triangle_count),
        "unmapped_semantic_face_fraction": (
            float(unmapped_face_count / total_triangle_count) if total_triangle_count else None
        ),
        "agent_height_m": agent_height,
        "agent_radius_m": agent_radius,
        "navmesh_cell_size_m": cell_size,
        "navmesh_cell_area_tolerance_m2": tolerance_m2,
        "footprint_method": footprint_method,
        "subtract_cross_region_blockers": subtract_cross_region_blockers,
        "category_instance_counts": dict(sorted(category_counts.items())),
        "review_category_instance_counts": dict(sorted(review_category_counts.items())),
        "category_role_instance_counts": dict(sorted(role_counts.items())),
        "region_floor_rows": candidate_rows,
        "overlapping_object_rows": object_rows,
        "completed_at_utc": now_utc(),
    }


def flatten_rows(house_results: list[dict]) -> tuple[list[dict], list[dict]]:
    rooms, objects = [], []
    for house_result in house_results:
        rooms.extend(house_result.get("region_floor_rows", []))
        objects.extend(house_result.get("overlapping_object_rows", []))
    return rooms, objects


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    formatted[key] = json.dumps(value, ensure_ascii=False)
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def build_summary(
    house_results: list[dict], failed: list[dict], scope: list[str],
    run_id: str = "room-screening-furniture-audit",
) -> dict:
    room_rows, object_rows = flatten_rows(house_results)
    status_counts = Counter(row.get("audit_status") for row in room_rows)
    issue_counts = Counter(code for row in room_rows for code in row.get("issue_codes", []))
    category_instances = Counter()
    category_overlap_rooms = Counter()
    for result in house_results:
        category_instances.update(result.get("category_instance_counts", {}))
    for obj in object_rows:
        category_overlap_rooms[obj.get("category", "unknown")] += 1
    raw = [row["raw_candidate_area_m2"] for row in room_rows if row.get("raw_candidate_area_m2") is not None]
    adjusted = [row["trial_adjusted_candidate_area_m2"] for row in room_rows if row.get("trial_adjusted_candidate_area_m2") is not None]
    return {
        "run_id": run_id,
        "created_at_utc": now_utc(),
        "scope": "provisional furniture/navmesh obstacle audit for downloaded HM3D rooms; not a screening verdict",
        "requested_houses": len(scope),
        "completed_houses": len(house_results),
        "failed_houses": len(failed),
        "region_floor_rows": len(room_rows),
        "rows_with_raw_candidate_area": len(raw),
        "row_status_counts": dict(sorted(status_counts.items())),
        "issue_code_counts": dict(sorted(issue_counts.items())),
        "semantic_blocker_instance_counts": dict(sorted(category_instances.items())),
        "unmapped_semantic_face_count": int(sum(h.get("unmapped_semantic_face_count", 0) for h in house_results)),
        "unmapped_semantic_colour_count": int(sum(h.get("unmapped_semantic_colour_count", 0) for h in house_results)),
        "semantic_total_triangle_count": int(sum(h.get("semantic_total_triangle_count", 0) for h in house_results)),
        "overlapping_object_row_counts_by_category": dict(sorted(category_overlap_rooms.items())),
        "overlapping_object_records": len(object_rows),
        "raw_candidate_area_sum_m2_descriptive_only": float(sum(raw)),
        "trial_adjusted_area_sum_m2_descriptive_only": float(sum(adjusted)),
        "failed_house_details": failed,
        "interpretation": (
            "The trial adjusted area is a conservative semantic-instance projection proxy. "
            "It is not the Plan's frozen metric, not validated ground truth, and must not be used "
            "for thresholds, verdicts, or automatic discards."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="JSON with scenes and region records; see manifest schema")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="External HM3D root containing train/ and val/ scene folders")
    parser.add_argument("--runtime-prefix", default=os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX"),
                        help="Installed Habitat prefix (or AVENGINE_HABITAT_RUNTIME_PREFIX)")
    parser.add_argument("--magnum-python-site", default=os.environ.get("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"),
                        help="Magnum Python site (or AVENGINE_HABITAT_MAGNUM_PYTHON_SITE)")
    parser.add_argument("--mp3d-root", default=os.environ.get("AVENGINE_MP3D_ROOT"),
                        help="External licensed MP3D root (or AVENGINE_MP3D_ROOT)")
    parser.add_argument("--house", action="append", help="repeatable house ID; default is all inventory scenes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Derived output directory; default is under repository tmp/")
    parser.add_argument("--run-id", default="room-screening-furniture-audit")
    parser.add_argument(
        "--raw-area-path",
        type=Path,
        required=True,
        help="candidate-area inventory generated with the same semantic ground-category definition",
    )
    parser.add_argument("--no-resume", action="store_true", help="recompute requested houses despite checkpoints")
    parser.add_argument(
        "--footprint-method",
        choices=("convex", "shape_preserving"),
        default="convex",
        help="convex is the original baseline; shape_preserving uses a concave projected hull",
    )
    parser.add_argument(
        "--subtract-cross-region-blockers",
        action="store_true",
        help="subtract mapped blockers on physical overlap even when semantic region ids differ",
    )
    args = parser.parse_args()
    if not args.runtime_prefix or not args.magnum_python_site or not args.mp3d_root:
        parser.error("Habitat runtime, Magnum Python site, and MP3D root must be supplied explicitly")
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {dataset_root}")
    out_dir = args.output_dir
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    preflight = json.loads(args.inventory.read_text())
    raw_doc = json.loads(args.raw_area_path.read_text())
    raw_scenes = {row["house"]: row for row in raw_doc["scenes"]}
    scene_records = preflight["scenes"]
    if args.house:
        requested = set(args.house)
        scene_records = [row for row in scene_records if row["house"] in requested]
        missing = requested - {row["house"] for row in scene_records}
        if missing:
            raise SystemExit(f"Unknown houses: {sorted(missing)}")
    if args.limit:
        scene_records = scene_records[:args.limit]
    code_hash = script_fingerprint()
    geometry_policy = {
        "footprint_method": args.footprint_method,
        "subtract_cross_region_blockers": bool(args.subtract_cross_region_blockers),
    }
    failed_current = []
    completed_now = 0
    total = len(scene_records)

    write_json_atomic(out_dir / "run_state.json", {
        "status": "running",
        "started_at_utc": now_utc(),
        "requested_house_count": total,
        "requested_houses": [row["house"] for row in scene_records],
        "implementation_sha256": code_hash,
        "geometry_policy": geometry_policy,
        "last_completed_house": None,
    })
    hs = base.runtime(
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_python_site,
        mp3d_root=args.mp3d_root,
    ).habitat_sim
    for number, record in enumerate(scene_records, 1):
        house = record["house"]
        split, index, scene_id = house.split("_", 3)[1:]
        scene_dir = dataset_root / split / f"{index}-{scene_id}"
        checkpoint_path = checkpoint_dir / f"{house}.json"
        try:
            if record.get("input_status", "complete") != "complete":
                failure = {
                    "house": house,
                    "status": "failed",
                    "error_type": "UnassessableInput",
                    "error": ";".join(record.get("issue_codes") or ["input_unassessable"]),
                    "failed_at_utc": now_utc(),
                }
                failed_current.append(failure)
                write_json_atomic(checkpoint_path, failure)
                print(f"[{number}/{total}] {house} UNASSESSABLE {failure['error']}", flush=True)
                continue
            fingerprint = fingerprint_for_scene(record, scene_dir, code_hash, geometry_policy)
            if not args.no_resume and checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text())
                if checkpoint.get("status") == "complete" and checkpoint.get("input_fingerprint") == fingerprint:
                    print(f"[{number}/{total}] {house} checkpoint=valid rows={len(checkpoint.get('region_floor_rows', []))}", flush=True)
                    continue
            result = process_house(
                hs,
                record,
                raw_scenes.get(house, {}),
                dataset_root,
                footprint_method=args.footprint_method,
                subtract_cross_region_blockers=args.subtract_cross_region_blockers,
            )
            result["input_fingerprint"] = fingerprint
            write_json_atomic(checkpoint_path, result)
            completed_now += 1
            state = {
                "status": "running",
                "started_at_utc": json.loads((out_dir / "run_state.json").read_text())["started_at_utc"],
                "last_updated_at_utc": now_utc(),
                "requested_house_count": total,
                "requested_houses": [row["house"] for row in scene_records],
                "implementation_sha256": code_hash,
                "geometry_policy": geometry_policy,
                "last_completed_house": house,
                "completed_in_this_invocation": completed_now,
            }
            write_json_atomic(out_dir / "run_state.json", state)
            scene_rows = result["region_floor_rows"]
            mismatch_count = sum("navmesh_static_obstacle_mismatch" in row["issue_codes"] for row in scene_rows)
            review_count = sum(row["audit_status"] == "needs_manual_review" for row in scene_rows)
            print(f"[{number}/{total}] {house} rows={len(scene_rows)} mismatches={mismatch_count} manual_review={review_count}", flush=True)
        except Exception as exc:  # keep the batch moving; preserve exception evidence
            failure = {
                "house": house,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": now_utc(),
            }
            failed_current.append(failure)
            write_json_atomic(checkpoint_path, failure)
            print(f"[{number}/{total}] {house} FAILED {type(exc).__name__}: {exc}", flush=True)

    all_house_results = []
    all_failed = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text())
        except Exception:
            continue
        if item.get("status") == "complete":
            all_house_results.append(item)
        elif item.get("status") == "failed":
            all_failed.append({k: item.get(k) for k in ("house", "error_type", "error", "failed_at_utc")})

    room_rows, object_rows = flatten_rows(all_house_results)
    summary = build_summary(
        all_house_results, all_failed, [row["house"] for row in scene_records], args.run_id
    )
    write_json_atomic(out_dir / "furniture_area_audit.json", {"summary": summary, "houses": all_house_results})
    write_csv(out_dir / "region_floor_audit.csv", room_rows, [
        "house", "region_id", "room_label", "floor_id_candidate", "floor_y_m",
        "agent_height_m", "agent_radius_m", "navmesh_cell_size_m", "navmesh_cell_area_tolerance_m2",
        "raw_candidate_area_m2", "semantic_ground_projected_area_m2",
        "semantic_ground_outside_navmesh_area_m2", "semantic_ground_navmesh_coverage_fraction",
        "semantic_ground_outside_navmesh_geometry_wkt",
        "source_candidate_area_m2", "blocker_direct_overlap_area_m2",
        "blocker_clearance_overlap_area_m2", "unknown_object_clearance_overlap_area_m2",
        "blocker_hull_residual_area_m2", "unmapped_semantic_candidate_overlap_area_m2",
        "footprint_extent_uncertainty_area_m2", "confirmed_candidate_area_m2",
        "adjustment_area_m2", "trial_adjusted_candidate_area_m2", "obstacle_instance_count",
        "cross_region_blocker_instance_count", "obstacle_categories", "audit_status", "issue_codes",
    ])
    write_csv(out_dir / "object_overlap_audit.csv", object_rows, [
        "house", "region_id", "floor_id_candidate", "instance_id", "rgb", "category", "role",
        "semantic_region_id", "same_region", "unassigned_region", "reason", "face_count",
        "footprint_area_m2", "footprint_centroid_xz_m", "footprint_centroid_in_raw_candidate",
        "direct_overlap_area_m2", "direct_overlap_fraction_of_footprint", "clearance_overlap_area_m2", "footprint_bounds_xz_m",
    ])
    requested_houses = {row["house"] for row in scene_records}
    requested_complete = requested_houses & {row["house"] for row in all_house_results}
    requested_failed = requested_houses & {row.get("house") for row in all_failed}
    final_state = {
        "status": "complete_with_failures" if requested_failed else ("complete" if len(requested_complete) == len(requested_houses) else "partial"),
        "started_at_utc": json.loads((out_dir / "run_state.json").read_text()).get("started_at_utc"),
        "finished_at_utc": now_utc(),
        "requested_house_count": total,
        "valid_checkpoint_count": len(all_house_results),
        "failed_checkpoint_count": len(all_failed),
        "implementation_sha256": code_hash,
        "geometry_policy": geometry_policy,
        "last_completed_house": json.loads((out_dir / "run_state.json").read_text()).get("last_completed_house"),
    }
    write_json_atomic(out_dir / "run_state.json", final_state)
    print(
        f"FINAL status={final_state['status']} requested={total} completed={len(requested_complete)} "
        f"failed={len(requested_failed)} rows={summary['region_floor_rows']} "
        f"mismatches={summary['issue_code_counts'].get('navmesh_static_obstacle_mismatch', 0)} "
        f"manual_review={summary['row_status_counts'].get('needs_manual_review', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
