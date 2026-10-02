#!/usr/bin/env python3
"""Compute provisional navmesh/floor-mesh overlap areas; never writes verdicts."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from avengine.acoustics.gltf import (
    extract_triangle_scene_document,
    load_glb_bytes,
    triangle_vertex_colours,
)
from avengine.acoustics.semantic import _linear_to_srgb_bytes
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "tmp/room_screening/candidate_area_inventory.json"
GROUND_CATEGORIES = {
    "floor", "carpet", "rug", "flooring", "floor mat", "mat", "doormat",
    "shower floor", "bathroom floor", "bath floor",
    "bath mat", "bathmat", "bathroom mat", "shower mat",
    "bathroom rug", "bath carpet", "bathroom carpet",
}


def runtime(
    *,
    runtime_prefix: str,
    magnum_python_site: str,
    mp3d_root: str,
    rlr_sdk_root: str | None = None,
):
    """Load the explicitly installed Habitat runtime and external MP3D root."""
    return prepare_installed_habitat_runtime(
        runtime_prefix=runtime_prefix,
        magnum_python_site=magnum_python_site,
        mp3d_root=mp3d_root,
        rlr_sdk_root=rlr_sdk_root,
        allow_mp3d_environment=False,
    )


def colour_to_region(scene_id: str, semantic_txt: Path):
    mapping = {}
    for line in semantic_txt.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 4 or not parts[0].strip().isdigit():
            continue
        colour = parts[1].strip().upper()
        if len(colour) != 6 or any(c not in "0123456789ABCDEF" for c in colour):
            continue
        key = (int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16))
        region_text = parts[3].strip()
        region = int(region_text) if region_text.lstrip("-").isdigit() else -1
        mapping[key] = (parts[2].strip().strip('"').lower(), region)
    return mapping


def semantic_ground_faces(scene_dir: Path, scene_id: str):
    semantic_glb = scene_dir / f"{scene_id}.semantic.glb"
    text_path = scene_dir / f"{scene_id}.semantic.txt"
    colour_map = colour_to_region(scene_id, text_path)
    document = load_glb_bytes(semantic_glb.read_bytes(), source_path=str(semantic_glb))
    scene = extract_triangle_scene_document(document)
    linear, _mixed = triangle_vertex_colours(document, scene)
    face_colours = _linear_to_srgb_bytes(linear)
    corners = scene.vertices[scene.triangles.astype(np.int64)].astype(np.float64)
    # Same HM3D Z-up to Habitat +Y-up transform as the shared inventory tool.
    corners = np.stack((corners[..., 0], corners[..., 2], -corners[..., 1]), axis=-1)
    result = defaultdict(list)
    all_regions = set()
    for i, colour in enumerate(map(tuple, face_colours)):
        item = colour_map.get(colour)
        if item is None:
            continue
        category, region = item
        if region >= 0:
            all_regions.add(region)
        if region < 0 or category not in GROUND_CATEGORIES:
            continue
        face = corners[i]
        poly = Polygon(face[:, [0, 2]])
        if poly.is_empty or poly.area <= 1e-10:
            continue
        result[region].append(
            {
                "polygon": poly,
                "ys": float(face[:, 1].mean()),
                "category": category,
            }
        )
    return result, all_regions


def navmesh_triangles(hs, nav_path: Path):
    pf = hs.PathFinder()
    if not pf.load_nav_mesh(str(nav_path)):
        raise RuntimeError(f"Habitat could not load {nav_path}")
    vertices = np.asarray(
        [[float(v[0]), float(v[1]), float(v[2])] for v in pf.build_navmesh_vertices()],
        dtype=np.float64,
    )
    indices = np.asarray(pf.build_navmesh_vertex_indices(), dtype=np.int64)
    if indices.size % 3 or (indices.size and (indices.min() < 0 or indices.max() >= len(vertices))):
        raise RuntimeError(f"Invalid triangulated navmesh output for {nav_path}")
    tri = vertices[indices].reshape((-1, 3, 3))
    polys = []
    ys = []
    for face in tri:
        poly = Polygon(face[:, [0, 2]])
        if poly.is_empty or poly.area <= 1e-10:
            continue
        polys.append(poly)
        ys.append(float(face[:, 1].mean()))
    return pf, polys, np.asarray(ys, dtype=np.float64)


def cluster_levels(ys: np.ndarray, gap_m: float):
    if ys.size == 0:
        return []
    order = np.argsort(ys)
    sorted_ys = ys[order]
    cuts = np.flatnonzero(np.diff(sorted_ys) > gap_m) + 1
    chunks = np.split(sorted_ys, cuts)
    return [float(np.median(c)) for c in chunks]


def collect_clusters(faces, gap_m):
    if not faces:
        return []
    face_ys = np.asarray([f["ys"] for f in faces], dtype=np.float64)
    centers = cluster_levels(face_ys, gap_m)
    clusters = []
    for center in centers:
        # The same level split rule is applied to semantic ground and navmesh
        # triangles; membership is assigned to the closest derived level.
        nearest = np.argmin(np.abs(face_ys[:, None] - np.asarray(centers)[None, :]), axis=1)
        selected = [f for f, idx in zip(faces, nearest) if idx == centers.index(center)]
        if not selected:
            continue
        polys = [f["polygon"] for f in selected]
        ground_union = unary_union(polys)
        yvals = np.asarray([f["ys"] for f in selected], dtype=np.float64)
        mad = float(np.median(np.abs(yvals - np.median(yvals))))
        clusters.append(
            {
                "median_y_m": float(np.median(yvals)),
                "face_count": len(selected),
                "categories": sorted({f["category"] for f in selected}),
                "ground_union": ground_union,
                "ground_projected_area_m2": float(ground_union.area),
                "ground_y_span_m": float(np.percentile(yvals, 95) - np.percentile(yvals, 5)) if len(yvals) > 1 else 0.0,
                "ground_y_mad_m": mad,
            }
        )
    return sorted(clusters, key=lambda c: c["median_y_m"])


def process_house(hs, scene_record, region_records, dataset_root: Path):
    house = scene_record["house"]
    split, index, scene_id = house.split("_", 3)[1:]
    scene_dir = dataset_root / split / f"{index}-{scene_id}"
    nav_path = scene_dir / f"{scene_id}.basis.navmesh"
    settings = scene_record["navmesh"]["navmesh_settings"]
    cell_height = float(settings["cell_height"])
    # The gap is derived from the saved vertical voxel resolution. It is a
    # provisional way to group candidate surfaces, not a fixed room-height test.
    level_gap_m = 2.0 * cell_height

    room_ground, mesh_regions = semantic_ground_faces(scene_dir, scene_id)
    pf, nav_polys, nav_ys = navmesh_triangles(hs, nav_path)
    nav_levels = cluster_levels(nav_ys, level_gap_m)
    nav_geometries = []
    for floor_y in nav_levels:
        memberships = np.argmin(np.abs(nav_ys[:, None] - np.asarray(nav_levels)[None, :]), axis=1)
        level_index = nav_levels.index(floor_y)
        polys = [poly for poly, idx in zip(nav_polys, memberships) if idx == level_index]
        geometry = unary_union(polys) if polys else None
        nav_geometries.append(
            {
                "navmesh_level_index": level_index,
                "median_y_m": floor_y,
                "triangle_count": len(polys),
                "projected_area_m2": float(geometry.area) if geometry is not None else 0.0,
                "geometry": geometry,
            }
        )

    old_regions = {
        int(r["region_id"]): r
        for r in region_records
        if r["house"] == house and r.get("region_id") is not None
    }
    region_rows = []
    flags = []
    for region_id in sorted(old_regions):
        source = old_regions[region_id]
        faces = room_ground.get(region_id, [])
        clusters = collect_clusters(faces, level_gap_m)
        if not clusters:
            region_rows.append(
                {
                    "house": house,
                    "region_id": region_id,
                    "room_label": f"R{region_id}",
                    "scope_status": "pending_verification",
                    "candidate_method": "semantic ground polygons intersected with Habitat PathFinder triangulated navmesh",
                    "ground_categories": sorted(GROUND_CATEGORIES),
                    "floor_id_candidate": None,
                    "floor_y_median_m": None,
                    "ground_triangle_count": 0,
                    "ground_projected_union_area_m2": 0.0,
                    "navmesh_floor_cluster_index": None,
                    "navmesh_floor_y_median_m": None,
                    "navmesh_floor_y_difference_m": None,
                    "navmesh_triangle_count": 0,
                    "navmesh_projected_union_area_m2": 0.0,
                    "candidate_semantic_region_nav_projected_area_m2": None,
                    "candidate_coverage_of_ground_footprint": None,
                    "issue_code": "no_semantic_ground_faces",
                }
            )
            continue
        if len(clusters) > 1:
            flags.append({"house": house, "region_id": region_id, "issue_code": "multiple_ground_height_clusters"})
        for floor_index, cluster in enumerate(clusters):
            if not nav_geometries:
                matched = None
            else:
                matched = min(nav_geometries, key=lambda item: abs(item["median_y_m"] - cluster["median_y_m"]))
            if matched is None or matched["geometry"] is None:
                inter = None
                floor_diff = None
                nav_area = 0.0
                nav_count = 0
                nav_level_index = None
                nav_level_y = None
            else:
                floor_diff = abs(matched["median_y_m"] - cluster["median_y_m"])
                inter = cluster["ground_union"].intersection(matched["geometry"])
                nav_area = matched["projected_area_m2"]
                nav_count = matched["triangle_count"]
                nav_level_index = matched["navmesh_level_index"]
                nav_level_y = matched["median_y_m"]
            candidate_area = float(inter.area) if inter is not None else None
            issue_codes = []
            if floor_diff is None:
                issue_codes.append("no_navmesh_floor_cluster")
            elif floor_diff > max(cell_height, cluster["ground_y_mad_m"] * 4.4478):
                issue_codes.append("floor_alignment_needs_review")
            if candidate_area is not None:
                if candidate_area > cluster["ground_projected_area_m2"] + 1e-5:
                    issue_codes.append("candidate_area_exceeds_ground_footprint")
                if candidate_area > nav_area + 1e-5:
                    issue_codes.append("candidate_area_exceeds_navmesh_layer")
            region_rows.append(
                {
                    "house": house,
                    "region_id": region_id,
                    "room_label": f"R{region_id}",
                    "scope_status": "pending_verification",
                    "candidate_method": "semantic ground polygons intersected with Habitat PathFinder triangulated navmesh",
                    "ground_categories": cluster["categories"],
                    "floor_id_candidate": floor_index,
                    "floor_y_median_m": cluster["median_y_m"],
                    "ground_triangle_count": cluster["face_count"],
                    "ground_projected_union_area_m2": cluster["ground_projected_area_m2"],
                    "ground_y_span_p95_p5_m": cluster["ground_y_span_m"],
                    "navmesh_floor_cluster_index": nav_level_index,
                    "navmesh_floor_y_median_m": nav_level_y,
                    "navmesh_floor_y_difference_m": floor_diff,
                    "navmesh_triangle_count": nav_count,
                    "navmesh_projected_union_area_m2": nav_area,
                    "candidate_semantic_region_nav_projected_area_m2": candidate_area,
                    "candidate_coverage_of_ground_footprint": (
                        candidate_area / cluster["ground_projected_area_m2"]
                        if candidate_area is not None and cluster["ground_projected_area_m2"] > 0
                        else None
                    ),
                    "issue_code": ";".join(issue_codes) if issue_codes else None,
                }
            )
    return {
        "house": house,
        "region_count_from_preflight": len(old_regions),
        "region_count_with_ground_faces": len(room_ground),
        "scene_navmesh_total_area_m2": float(pf.navigable_area),
        "navmesh_cell_height_m": cell_height,
        "derived_floor_cluster_gap_m": level_gap_m,
        "navmesh_floor_cluster_count": len(nav_levels),
        "navmesh_floor_cluster_heights_m": nav_levels,
        "regions": region_rows,
        "flags": flags,
    }


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main():
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
    parser.add_argument("--rlr-sdk-root", default=os.environ.get("AVENGINE_RLR_SDK_ROOT"),
                        help="Optional external RLR SDK root (or AVENGINE_RLR_SDK_ROOT)")
    parser.add_argument("--house", action="append", help="repeatable HM3D house id; default is all inventory scenes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Derived output path; default is under repository tmp/")
    parser.add_argument("--run-id", default="room-screening-area-provisional")
    args = parser.parse_args()

    if not args.runtime_prefix or not args.magnum_python_site or not args.mp3d_root:
        parser.error("Habitat runtime, Magnum Python site, and MP3D root must be supplied explicitly")
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {dataset_root}")
    data = json.loads(args.inventory.read_text())
    scene_records = data["scenes"]
    region_records = data["regions"]
    if args.house:
        selected = set(args.house)
        scene_records = [s for s in scene_records if s["house"] in selected]
        missing = selected - {s["house"] for s in scene_records}
        if missing:
            raise SystemExit(f"Unknown houses: {sorted(missing)}")
    if args.limit:
        scene_records = scene_records[:args.limit]

    hs = runtime(
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_python_site,
        mp3d_root=args.mp3d_root,
        rlr_sdk_root=args.rlr_sdk_root,
    ).habitat_sim
    results = []
    failed_scenes = []
    for number, scene in enumerate(scene_records, 1):
        house = scene["house"]
        if scene.get("input_status", "complete") != "complete":
            issue_codes = scene.get("issue_codes") or ["input_unassessable"]
            rows = [
                {
                    "house": house,
                    "region_id": region.get("region_id"),
                    "room_label": f"R{region.get('region_id')}",
                    "scope_status": "unresolved",
                    "candidate_semantic_region_nav_projected_area_m2": None,
                    "issue_code": ";".join(issue_codes),
                }
                for region in region_records if region.get("house") == house
            ]
            row = {"house": house, "status": "unassessable", "regions": rows, "flags": issue_codes}
        else:
            try:
                row = process_house(hs, scene, region_records, dataset_root)
                row["status"] = "complete"
            except Exception as error:
                row = {"house": house, "status": "failed", "regions": [], "flags": [type(error).__name__]}
                failed_scenes.append({"house": house, "error_type": type(error).__name__, "error": str(error)})
        results.append(row)
        areas = [
            r["candidate_semantic_region_nav_projected_area_m2"]
            for r in row["regions"]
            if r.get("candidate_semantic_region_nav_projected_area_m2") is not None
        ]
        print(
            f"[{number}/{len(scene_records)}] {scene['house']} "
            f"status={row['status']} regions={len(row['regions'])} with_area={len(areas)}",
            flush=True,
        )
        write_json_atomic(args.output, {
            "run_id": args.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provisional_only": True,
            "area_field": "candidate_semantic_region_nav_projected_area_m2",
            "shapely_version": shapely.__version__,
            "floor_level_gap_rule": "2 x serialized navmesh cell_height",
            "houses_processed": [x["house"] for x in results],
            "failed_scenes": failed_scenes,
            "unassessable_scenes": [x["house"] for x in results if x["status"] == "unassessable"],
            "scenes": results,
        })


if __name__ == "__main__":
    main()
