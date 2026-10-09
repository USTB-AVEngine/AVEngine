#!/usr/bin/env python3
"""Measure retained polygon rooms with the original CPU whole-house escape rays.

Only polygon membership changes: each Polygon uses its exterior, without
interior rings. Original navmesh grid, pose search, heights, four ray origins,
512 spherical directions and whole-house acoustic mesh remain in use.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np
import shapely
from shapely.geometry import Polygon, shape
import yaml

from tools.acoustics.room_split_escape import (
    alternate_listener, ray_checks, room_manifest, save, strict_zero_scene,
)
from tools.rooms.room_selection.geometry import collision_mesh
from tools.rooms.room_selection.navigation import sample_navigation, placement


def read(path):
    return json.loads(Path(path).read_text())


def exterior_scope(geojson):
    """Fill each Polygon's holes while preserving concavity and separate parts."""
    original = shape(geojson)
    if original.geom_type not in ("Polygon", "MultiPolygon") or original.is_empty:
        raise ValueError("floor_polygon_xz_m must be a nonempty Polygon/MultiPolygon")
    if not original.is_valid:
        raise ValueError("Invalid floor polygon; automatic repair is not authorized")
    parts = list(original.geoms) if original.geom_type == "MultiPolygon" else [original]
    outer = shapely.union_all([Polygon(p.exterior) for p in parts])
    return outer, {
        "membership_rule": "union of each Polygon exterior; interior holes filled; no inward buffer",
        "polygon_count": len(parts),
        "internal_hole_count": sum(len(p.interiors) for p in parts),
        "original_area_m2": float(original.area),
        "filled_exterior_area_m2": float(outer.area),
        "added_area_m2": float(outer.area - original.area),
    }


def origins_for(label, witness):
    return [(label, role, witness[role]) for role in ("camera_m", "source_1_m", "source_2_m")] + [
        (label, "camera_alt_m", alternate_listener(witness))
    ]


def band(fraction):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Invalid escape fraction")
    return "测试" if fraction <= .05 else "只训练" if fraction <= .15 else "不用"


def choose_retained(delivery, limit=None):
    rooms = []
    for path in sorted((delivery / "rooms").glob("*.json")):
        row = read(path)
        if row.get("decision") == "retain":
            row["input_room_json"] = str(path)
            if path.stem != row["id"] or Path(row["id"]).name != row["id"]:
                raise ValueError("Room ID/path mismatch")
            rooms.append(row)
    if not rooms:
        raise ValueError("No retained rooms in delivery/rooms")
    if len({r["id"] for r in rooms}) != len(rooms):
        raise ValueError("Duplicate room IDs")
    if limit is None:
        return rooms, len(rooms)
    # Premeasurement, deterministic round-robin across houses for the pilot.
    grouped = defaultdict(list)
    for row in rooms:
        grouped[row["house"]].append(row)
    ordered = []
    while any(grouped.values()):
        for house in sorted(grouped):
            if grouped[house]:
                ordered.append(grouped[house].pop(0))
    return ordered[:limit], len(rooms)


def make_renderer(house_input, first_witness, settings, destination):
    from avengine.acoustics.compiler import compile_hm3d_semantic_research_scene
    from avengine.acoustics.runtime import load_compiled_acoustic_scene, RLRSimulationConfig
    from avengine.acoustics.rir_cache import _NativeRIRBatchRenderer

    stamp = time.monotonic()
    destination.mkdir(parents=True, exist_ok=False)
    h = dict(house_input)
    h["acoustic_rooms"] = [{"placement": first_witness}]
    package = h.get("acoustic_package_manifest")
    compile_mode = "reused_whole_house_package"
    if package is None:
        # A new package uses the original compiler, material seed and raw house;
        # the split polygon never enters acoustic geometry compilation.
        config = destination / "scene_dataset_config.json"
        save(config, {"stages": {"paths": {".glb": [str(Path(h["scene_directory"]) / (h["scan_id"] + ".glb"))]}}})
        manifest = destination / "room_manifest.json"
        save(manifest, room_manifest(h, config))
        package, _ = compile_hm3d_semantic_research_scene(
            room_manifest=manifest, material_rules=settings["material_rules"]["hm3d"],
            output=destination / "package", seed=settings["material_seeds"]["hm3d"],
            probe_origins=[first_witness["camera_m"]], probe_direction_count=16,
        )
        compile_mode = "compiled_original_whole_house"
    scene = load_compiled_acoustic_scene(package, allow_nonpassing_research_qa=True)
    if scene.manifest["source_room"]["room_id"] != h["house"]:
        raise ValueError("Acoustic package belongs to a different house")
    scene, geometry = strict_zero_scene(scene, destination / "derived_acoustic_geometry")
    renderer = _NativeRIRBatchRenderer(
        scene, RLRSimulationConfig.from_mapping(settings["simulation"]), batch_size=1,
        initial_positions_m=[first_witness["source_1_m"]], listener_position_m=first_witness["camera_m"],
        listener_orientation_wxyz=[1., 0., 0., 0.], layout_type="ambisonics", channel_count=4,
        hrtf_file_path="", source_radius_m=settings["source_radius_m"],
        listener_radius_m=settings["listener_radius_m"], **settings["runtime"],
    )
    warmup = renderer.render([first_witness["source_1_m"]])
    vertices = np.concatenate([o["vertices"] for o in scene.objects])
    bounds = np.stack((vertices.min(0), vertices.max(0)))
    setup = {
        "house": h["house"], "package_manifest": str(package), "compile_mode": compile_mode,
        "geometry_scope": "entire original physical house; split polygon never clips the mesh",
        "geometry": geometry, "native_setup": renderer.setup_report,
        "bounds_m": bounds.tolist(), "warmup_seconds": warmup.wall_seconds,
        "wall_seconds": time.monotonic() - stamp,
    }
    save(destination / "native_setup.json", setup)
    return renderer, bounds, setup


def polygon_pose_checks(scope, original, witness, pathfinder, parameters):
    records = []
    for _, role, origin in origins_for("check", witness):
        xyz = np.asarray(origin, dtype=float)
        item = {
            "role": role, "origin_m": xyz.tolist(),
            "inside_filled_exterior": bool(shapely.contains_xy(scope, xyz[0], xyz[2])),
            "inside_original_with_holes": bool(shapely.contains_xy(original, xyz[0], xyz[2])),
            "exterior_boundary_distance_m": float(scope.boundary.distance(shapely.Point(xyz[0], xyz[2]))),
        }
        if role != "camera_alt_m":
            height = parameters["camera_height_m"] if role == "camera_m" else parameters["source_height_m"]
            support = xyz - [0, height, 0]
            snapped = np.asarray(pathfinder.snap_point(support), dtype=float)
            if not np.isfinite(snapped).all():
                raise ValueError("Selected witness has no navmesh support")
            item.update(
                nav_support_m=support.tolist(), navmesh_snap_m=snapped.tolist(),
                navmesh_resnap_distance_m=float(np.linalg.norm(snapped - support)),
                height_above_navmesh_m=float(xyz[1] - snapped[1]),
                expected_height_m=height,
                clearance_m=float(pathfinder.distance_to_closest_obstacle(support, parameters["clearance_query_radius_m"])),
            )
            if not item["inside_filled_exterior"]:
                raise ValueError("Original placement selected a pose outside the exterior")
        records.append(item)
    return records


def measure_new_room(row, renderer, bounds, mesh, pf, hs, parameters, settings, output):
    started = time.monotonic()
    scope, polygon_metadata = exterior_scope(row["floor_polygon_xz_m"])
    nav, points, clearance, adj, _ = sample_navigation(pf, hs, scope, row["floor_y_m"], parameters)
    witness = placement(mesh, points, clearance, adj, parameters)
    if not witness["found"]:
        raise ValueError(f"No placement witness within original frozen budget: {row['id']}")
    checks = polygon_pose_checks(scope, shape(row["floor_polygon_xz_m"]), witness, pf, parameters)
    placement_seconds = time.monotonic() - started
    ray_started = time.monotonic()
    leaks = ray_checks(renderer.context, origins_for(row["id"], witness), bounds, settings["criteria"]["ray_count_per_origin"])
    ray_seconds = time.monotonic() - ray_started
    fraction = max(x["escape_fraction"] for x in leaks)
    result = {
        "room": row["id"], "house": row["house"], "source_region": row["source_region"],
        "input_room_json": row["input_room_json"], "floor_y_m": row["floor_y_m"],
        "floor_area_m2": row["floor_area_m2"], "max_escape_fraction": fraction, "band": band(fraction),
        "measurement_scope": "ray escape only; other acoustic/visual/split eligibility gates not evaluated",
        "polygon_membership": polygon_metadata, "pose_checks": checks, "placement_witness": witness,
        "sources_inside_filled_exterior": all(c["inside_filled_exterior"] for c in checks if c["role"].startswith("source_")),
        "all_ray_origins_inside_filled_exterior": all(c["inside_filled_exterior"] for c in checks),
        "navigation": nav, "leakage": leaks, "placement_seconds": placement_seconds,
        "ray_seconds": ray_seconds, "wall_seconds": time.monotonic() - started,
        "compute_device": "CPU", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(output / "rooms" / (row["id"] + ".json"), result)
    return result


def write_csv(output, rows):
    columns = ["room", "house", "source_region", "max_escape_fraction", "band", "sources_inside_filled_exterior",
               "all_ray_origins_inside_filled_exterior", "placement_seconds", "ray_seconds", "wall_seconds",
               "house_setup_seconds_per_room", "wall_seconds_including_amortized_setup", "input_room_json"]
    with (output / "room_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' before CPU-only execution")
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    settings = read(args.settings)
    if settings["criteria"]["ray_count_per_origin"] != 512 or settings["criteria"]["leak_pass_max"] != .05 or settings["criteria"]["leak_fail_above"] != .15:
        raise ValueError("The supplied settings differ from the requested frozen measurement")
    spec = yaml.safe_load(args.thresholds.read_text())
    parameters = {k: v["value"] for k, v in spec["parameters"].items()}
    house_inputs = read(args.house_inputs)["houses"]
    if not (args.delivery / "summary.json").is_file():
        raise FileNotFoundError(args.delivery / "summary.json")
    rooms, total = choose_retained(args.delivery, args.limit)
    by_house = defaultdict(list)
    for row in rooms:
        by_house[row["house"]].append(row)
        h = house_inputs[row["house"]]
        if h["house"] != row["house"]:
            raise ValueError("House input ID mismatch")
        for key in ("navmesh_source", "semantic_source", "annotation_source"):
            if not Path(h[key]).is_file():
                raise FileNotFoundError(h[key])
        if h.get("acoustic_package_manifest") and not Path(h["acoustic_package_manifest"]).is_file():
            raise FileNotFoundError(h["acoustic_package_manifest"])
    output = args.output.resolve()
    delivery = args.delivery.resolve()
    if output == delivery or delivery in output.parents or output in delivery.parents:
        raise ValueError("Output must be separate from delivery inputs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "rooms").mkdir()
    save(output / "selected_rooms.json", {"method": "all retained rooms" if args.limit is None else "deterministic round-robin by house before acoustics", "rooms": [r["id"] for r in rooms]})
    save(output / "settings.used.json", settings)
    (output / "placement_thresholds.used.yaml").write_bytes(args.thresholds.read_bytes())
    save(output / "run.json", {"pid": os.getpid(), "argv": sys.argv, "machine": os.uname().nodename,
                               "nice": os.getpriority(os.PRIO_PROCESS, 0), "worker_process_count": 1,
                               "rlr_thread_count": settings["simulation"]["thread_count"], "cpu_only": True,
                               "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    results = []
    setups = []
    error = None
    try:
        for house, house_rooms in sorted(by_house.items()):
            h = house_inputs[house]
            first = house_rooms[0]["placement_witness"]
            if not first.get("found"):
                raise ValueError("Input delivery room lacks its original placement witness")
            renderer, bounds, setup = make_renderer(h, first, settings, output / "houses" / house)
            hs = sys.modules["habitat_sim"]
            pf = hs.PathFinder()
            if not pf.load_nav_mesh(h["navmesh_source"]):
                raise ValueError("Native PathFinder failed to load original navmesh")
            load_started = time.monotonic()
            mesh, raw_path = collision_mesh(h["scene_directory"])
            setup["placement_raw_mesh_path"] = raw_path
            setup["placement_mesh_load_seconds"] = time.monotonic() - load_started
            setup["total_setup_seconds"] = setup["wall_seconds"] + setup["placement_mesh_load_seconds"]
            setups.append(setup)
            for row in house_rooms:
                result = measure_new_room(row, renderer, bounds, mesh, pf, hs, parameters, settings, output)
                result["house_setup_seconds_per_room"] = setup["total_setup_seconds"] / len(house_rooms)
                result["wall_seconds_including_amortized_setup"] = result["wall_seconds"] + result["house_setup_seconds_per_room"]
                # Result was saved before house timing attribution; keep the timing in CSV and summary.
                results.append(result)
                write_csv(output, results)
                print("ROOM_DONE", result["room"], result["max_escape_fraction"], result["band"],
                      "sources_inside", result["sources_inside_filled_exterior"], "seconds", round(result["wall_seconds"], 3), flush=True)
            del renderer, mesh, pf, bounds
            gc.collect()
    except Exception as exc:
        error = {"exception": type(exc).__name__, "reason": str(exc), "traceback": traceback.format_exc()}
        save(output / "failure.json", error)
    write_csv(output, results)
    summary = {
        "status": "complete" if error is None and len(results) == len(rooms) else "stopped",
        "delivery": str(delivery), "total_retained_input_rooms": total, "selected_rooms": len(rooms),
        "measured_rooms": len(results), "measured_houses": len(setups), "bands": dict(Counter(r["band"] for r in results)),
        "all_sources_inside_filled_exterior": all(r["sources_inside_filled_exterior"] for r in results),
        "all_ray_origins_inside_filled_exterior": all(r["all_ray_origins_inside_filled_exterior"] for r in results),
        "cpu_only": True, "worker_process_count": 1, "rlr_thread_count": settings["simulation"]["thread_count"],
        "ray_count_per_origin": 512, "origins_per_room": 4, "thresholds": {"test_max": .05, "training_max": .15},
        "scope": "leakage-only bands; no complete test/production admission", "wall_seconds": time.monotonic() - started,
        "settings_path": str(args.settings), "placement_thresholds_path": str(args.thresholds),
        "house_inputs_path": str(args.house_inputs), "house_setups": setups,
        "code_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "code_worktree_dirty": bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "error": error,
    }
    save(output / "summary.json", summary)
    if error:
        raise RuntimeError("Stopped after failure; no further houses started: " + error["reason"])
    print("COMPLETE", len(results), "wall_seconds", round(summary["wall_seconds"], 3), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--house-inputs", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="pilot only: deterministic selection across houses; omit to run all retained rooms")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
