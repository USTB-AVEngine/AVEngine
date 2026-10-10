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
import multiprocessing
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
from tools.acoustics.split_house_inputs import placement_mesh, require_houses, validate_house_input
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
    from avengine.acoustics.compiler import compile_hm3d_semantic_research_scene, compile_mp3d_semantic_research_scene
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
        compiler = {"hm3d": compile_hm3d_semantic_research_scene, "mp3d": compile_mp3d_semantic_research_scene}.get(h["family"])
        if compiler is None:
            raise ValueError("Kujiale requires its historical package_rlr; it cannot be compiled as MP3D")
        package, _ = compiler(
            room_manifest=manifest, material_rules=settings["material_rules"][h["family"]],
            output=destination / "package", seed=settings["material_seeds"][h["family"]],
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
        "house": h["house"], "family": h["family"], "package_manifest": str(package), "compile_mode": compile_mode,
        "source_to_canonical": scene.manifest["geometry"]["source_to_canonical"],
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
        "room": row["id"], "house": row["house"], "family": row["house"].split("_")[0], "source_region": row["source_region"],
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
    columns = ["room", "house", "family", "source_region", "max_escape_fraction", "band", "sources_inside_filled_exterior",
               "all_ray_origins_inside_filled_exterior", "placement_seconds", "ray_seconds", "wall_seconds",
               "house_setup_seconds_per_room", "wall_seconds_including_amortized_setup", "input_room_json"]
    with (output / "room_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in sorted(rows, key=lambda r: r["room"]))


def partition_houses(by_house, jobs):
    """Deterministic house assignments; a house is never split between workers."""
    if jobs < 1:
        raise ValueError("--jobs must be positive")
    houses = sorted(by_house)
    count = min(jobs, len(houses))
    return [houses[i::count] for i in range(count)] if count else []


def merge_worker_results(chunks):
    rows = sorted((r for chunk in chunks for r in chunk["results"]), key=lambda r: r["room"])
    if len({r["room"] for r in rows}) != len(rows):
        raise ValueError("Workers returned duplicate room IDs")
    setups = sorted((s for chunk in chunks for s in chunk["setups"]), key=lambda s: s["house"])
    return rows, setups


def cpu_guard():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' before CPU-only execution")
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))


def _measure_house_batch(index, houses, by_house, house_inputs, settings, parameters, output):
    cpu_guard()
    started = time.monotonic()
    result = {"worker_index": index, "pid": os.getpid(), "houses": houses, "results": [], "setups": [],
              "nice": os.getpriority(os.PRIO_PROCESS, 0), "rlr_thread_count": 8, "cpu_only": True, "error": None}
    print("WORKER_START", index, "pid", os.getpid(), "houses", houses, flush=True)
    try:
        for house in houses:
            if (output / "failure_requested.json").exists():
                break  # A different worker failed; finish current work, start no further house.
            house_started = time.monotonic()
            h, house_rooms = house_inputs[house], by_house[house]
            first = house_rooms[0]["placement_witness"]
            if not first.get("found"):
                raise ValueError("Input delivery room lacks its original placement witness")
            renderer, bounds, setup = make_renderer(h, first, settings, output / "houses" / house)
            hs = sys.modules["habitat_sim"]
            pf = hs.PathFinder()
            nav_started = time.monotonic()
            if not pf.load_nav_mesh(h["navmesh_source"]):
                raise ValueError("Native PathFinder failed to load original navmesh")
            setup["navmesh_load_seconds"] = time.monotonic() - nav_started
            load_started = time.monotonic()
            mesh, mesh_receipt = placement_mesh(h)
            setup["placement_raw_mesh_path"] = mesh_receipt["path"]
            setup["placement_mesh"] = mesh_receipt
            setup["placement_mesh_load_seconds"] = time.monotonic() - load_started
            setup["total_setup_seconds"] = time.monotonic() - house_started
            setup["worker_index"] = index
            save(output / "houses" / house / "placement_setup.json", setup)
            result["setups"].append(setup)
            print("HOUSE_READY", house, "setup_seconds", setup["total_setup_seconds"], flush=True)
            for row in house_rooms:
                measured = measure_new_room(row, renderer, bounds, mesh, pf, hs, parameters, settings, output)
                measured["house_setup_seconds_per_room"] = setup["total_setup_seconds"] / len(house_rooms)
                measured["wall_seconds_including_amortized_setup"] = measured["wall_seconds"] + measured["house_setup_seconds_per_room"]
                result["results"].append(measured)
                print("ROOM_DONE", measured["room"], measured["max_escape_fraction"], measured["band"],
                      "sources_inside", measured["sources_inside_filled_exterior"], "seconds", round(measured["wall_seconds"], 3), flush=True)
            del renderer, mesh, pf, bounds
            gc.collect()
    except Exception as exc:
        result["error"] = {"exception": type(exc).__name__, "reason": str(exc), "traceback": traceback.format_exc(), "worker_index": index}
        try:
            save(output / "failure_requested.json", result["error"])
        except FileExistsError:
            pass
        print("WORKER_FAILED", result["error"]["traceback"], flush=True)
    result["wall_seconds"] = time.monotonic() - started
    result["status"] = "stopped" if result["error"] else "complete"
    save(output / "workers" / f"result_{index:03d}.json", result)
    return result


def _worker_entry(index, houses, by_house, house_inputs, settings, parameters, output):
    # Redirect file descriptors too: the native RLR library writes outside Python's stdout.
    sys.stdout.flush()
    sys.stderr.flush()
    previous = (os.dup(1), os.dup(2))
    try:
        with (output / "logs" / f"worker_{index:03d}.log").open("x") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            result = _measure_house_batch(index, houses, by_house, house_inputs, settings, parameters, output)
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(previous[0], 1)
        os.dup2(previous[1], 2)
        os.close(previous[0])
        os.close(previous[1])
    return result


def family_timings(results, setups):
    timings = {}
    for family in sorted({r["family"] for r in results}):
        rs = [r for r in results if r["family"] == family]
        ss = [s["total_setup_seconds"] for s in setups if s["family"] == family]
        timings[family] = {
            "houses": len(ss), "rooms": len(rs),
            "initialization_seconds_mean": float(np.mean(ss)), "initialization_seconds_median": float(np.median(ss)),
            "initialization_seconds_range": [min(ss), max(ss)],
            "room_seconds_mean": float(np.mean([r["wall_seconds"] for r in rs])),
            "room_seconds_median": float(np.median([r["wall_seconds"] for r in rs])),
            "ray_seconds_mean": float(np.mean([r["ray_seconds"] for r in rs])),
            "placement_seconds_mean": float(np.mean([r["placement_seconds"] for r in rs])),
            "worker_measured_seconds_sum": sum(ss) + sum(r["wall_seconds"] for r in rs),
        }
    return timings


def run(args):
    cpu_guard()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    settings = read(args.settings)
    if (settings["criteria"]["ray_count_per_origin"] != 512 or settings["criteria"]["leak_pass_max"] != .05
            or settings["criteria"]["leak_fail_above"] != .15 or settings["simulation"]["thread_count"] != 8):
        raise ValueError("The supplied settings differ from the requested frozen measurement (512 rays, 5/15%, RLR 8 threads)")
    spec = yaml.safe_load(args.thresholds.read_text())
    parameters = {k: v["value"] for k, v in spec["parameters"].items()}
    house_inputs = read(args.house_inputs)["houses"]
    if not (args.delivery / "summary.json").is_file():
        raise FileNotFoundError(args.delivery / "summary.json")
    rooms, total = choose_retained(args.delivery, args.limit)
    by_house = defaultdict(list)
    for row in rooms:
        by_house[row["house"]].append(row)
    require_houses(house_inputs, by_house)
    input_checks = []
    for house in sorted(by_house):
        h = house_inputs[house]
        if h["house"] != house:
            raise ValueError("House input ID mismatch")
        input_checks.append(validate_house_input(h))
    output = args.output.resolve()
    delivery = args.delivery.resolve()
    if output == delivery or delivery in output.parents or output in delivery.parents:
        raise ValueError("Output must be separate from delivery inputs")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("rooms", "workers", "logs"):
        (output / directory).mkdir()
    assignments = partition_houses(by_house, args.jobs)
    save(output / "selected_rooms.json", {"method": "all retained rooms" if args.limit is None else "deterministic round-robin by house before acoustics", "rooms": sorted(r["id"] for r in rooms)})
    save(output / "settings.used.json", settings)
    save(output / "house_inputs.validation.json", {"status": "pass", "houses": input_checks})
    (output / "placement_thresholds.used.yaml").write_bytes(args.thresholds.read_bytes())
    save(output / "run.json", {"pid": os.getpid(), "argv": sys.argv, "machine": os.uname().nodename,
                               "nice": os.getpriority(os.PRIO_PROCESS, 0), "worker_process_count": len(assignments),
                               "jobs_requested": args.jobs, "house_assignments": assignments,
                               "rlr_thread_count": 8, "cpu_only": True,
                               "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    processes = []
    if len(assignments) == 1:
        chunks = [_worker_entry(0, assignments[0], by_house, house_inputs, settings, parameters, output)]
        process_records = [{"worker_index": 0, "pid": os.getpid(), "houses": assignments[0], "exitcode": 0}]
    else:
        context = multiprocessing.get_context("spawn")
        for index, houses in enumerate(assignments):
            process = context.Process(target=_worker_entry, args=(index, houses, by_house, house_inputs, settings, parameters, output), name=f"escape-worker-{index}")
            process.start()
            processes.append(process)
        save(output / "workers" / "processes.json", {"coordinator_pid": os.getpid(), "workers": [
            {"worker_index": i, "pid": p.pid, "houses": assignments[i], "log": str(output / "logs" / f"worker_{i:03d}.log")}
            for i, p in enumerate(processes)]})
        for process in processes:
            process.join()
        chunks, process_records = [], []
        for index, process in enumerate(processes):
            process_records.append({"worker_index": index, "pid": process.pid, "houses": assignments[index], "exitcode": process.exitcode})
            response = output / "workers" / f"result_{index:03d}.json"
            if response.is_file():
                chunk = read(response)
            else:
                chunk = {"results": [], "setups": [], "error": {"reason": f"Worker {index} exited without its result file", "worker_index": index, "exitcode": process.exitcode}}
            if process.exitcode and not chunk["error"]:
                chunk["error"] = {"reason": f"Worker {index} exited with code {process.exitcode}", "worker_index": index}
            chunks.append(chunk)
    results, setups = merge_worker_results(chunks)
    errors = [chunk["error"] for chunk in chunks if chunk["error"]]
    if not errors and len(results) != len(rooms):
        errors.append({"reason": "Workers returned fewer rooms than selected; consult worker logs"})
    error = errors or None
    if error:
        save(output / "failure.json", {"errors": error})
    write_csv(output, results)
    summary = {
        "status": "complete" if error is None else "stopped",
        "delivery": str(delivery), "total_retained_input_rooms": total, "selected_rooms": len(rooms),
        "measured_rooms": len(results), "measured_houses": len(setups), "bands": dict(Counter(r["band"] for r in results)),
        "all_sources_inside_filled_exterior": bool(results) and all(r["sources_inside_filled_exterior"] for r in results),
        "all_ray_origins_inside_filled_exterior": bool(results) and all(r["all_ray_origins_inside_filled_exterior"] for r in results),
        "cpu_only": True, "worker_process_count": len(assignments), "jobs_requested": args.jobs, "workers": process_records,
        "rlr_thread_count": 8, "ray_count_per_origin": 512, "origins_per_room": 4,
        "thresholds": {"test_max": .05, "training_max": .15}, "csv_row_order": "lexicographic room ID",
        "scope": "leakage-only bands; no complete test/production admission", "wall_seconds": time.monotonic() - started,
        "settings_path": str(args.settings), "placement_thresholds_path": str(args.thresholds),
        "house_inputs_path": str(args.house_inputs), "house_setups": setups, "family_timings": family_timings(results, setups),
        "code_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "code_worktree_dirty": bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "error": error,
    }
    save(output / "summary.json", summary)
    if error:
        raise RuntimeError("Stopped after worker failure: " + "; ".join(x["reason"] for x in error))
    print("COMPLETE", len(results), "workers", len(assignments), "wall_seconds", round(summary["wall_seconds"], 3), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--house-inputs", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="CPU worker processes, assigned by whole house; each RLR context uses the original 8 threads")
    parser.add_argument("--limit", type=int, help="pilot only: deterministic selection across houses; omit to run all retained rooms")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
