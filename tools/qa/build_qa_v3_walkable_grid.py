#!/usr/bin/env python3
"""Build a scene's walkable-floor grid (see walkable_grid.py for what it is for).

Two sources, chosen explicitly on the command line:

``--from-ue-navmesh``
    Launch the packaged stage once and draw random points from the map's own
    RecastNavMesh (the same navigation data the apartment route bank was
    queried from), then rasterise them.  A cell is walkable when at least
    ``--min-points-per-cell`` points fell into it; with the default two million
    points and 10 cm cells the expected count per navigable cell is in the
    hundreds, so a miss is not a sampling accident but a sliver of navmesh
    smaller than a cell.

``--from-feasible-region``
    Re-express the scene config's ``line_of_sight_grid`` raster (schema
    ``avengine_room_feasible_region_v1``, Habitat (x, z) metres) on this
    schema through its declared coordinate contract.  No engine is launched.

The scene's own navigation points (every route-bank sample and every solver
camera point) are then stamped walkable: they come from the same navigation
authority, and a navigation path may clip a polygon corner that random points
rarely reach (58 of 144,975 apartment samples on 2026-09-03).  The number of
cells added this way is recorded.

Validation (always run, recorded in the product; a low bank-inside fraction
fails the build and keeps the evidence): every bank sample and every camera
point must lie in a walkable cell.  The fraction of bank samples that would
also keep the consumer margin is reported so the strictness of the margin
relative to the bank is visible.

Fresh outputs only; nothing is overwritten.  Research product.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import scene_sampler as SS  # noqa: E402
from walkable_grid import (  # noqa: E402
    SCHEMA,
    WalkableGrid,
    rasterize_points,
    write_walkable_grid,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_worktree_state(repo: Path = REPO) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                              capture_output=True).stdout
    status = run("status", "--short").splitlines()
    return {"revision": run("rev-parse", "HEAD").strip(), "dirty": bool(status),
            "status": status}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-ue-navmesh", action="store_true")
    source.add_argument("--from-feasible-region", action="store_true")
    parser.add_argument("--cell-cm", type=float, default=10.0,
                        help="navmesh mode only; the feasible-region mode keeps its pixel size")
    parser.add_argument("--stage-root")
    parser.add_argument("--spear-executable")
    parser.add_argument("--samples-per-call", type=int, default=20000)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--min-points-per-cell", type=int, default=2)
    parser.add_argument("--report-margin-m", type=float, action="append",
                        help="margins whose bank pass fraction is reported (default 0.2 0.3)")
    parser.add_argument("--min-bank-inside-fraction", type=float, default=0.99)
    parser.add_argument("--rpc-port", type=int, default=39571)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# source A: the stage's navmesh
# --------------------------------------------------------------------------

def navmesh_random_points(args, native_map: str) -> tuple[np.ndarray, dict]:
    from avengine.backends.spear_ue.research_runtime import (
        launch_external_game_instance,
        run_frame_transaction,
    )
    from avengine.timeline import current_apartment_visual as VISUAL

    executable = Path(args.spear_executable)
    _require(executable.is_file(), f"missing SpearSim executable: {executable}")
    _require(Path(args.stage_root).is_dir(), f"missing stage root: {args.stage_root}")
    started = time.time()
    instance = launch_external_game_instance(
        spear_executable=executable, native_map=native_map,
        frame_rate_hz=VISUAL.FRAME_RATE_HZ, rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter)
    launch_seconds = time.time() - started
    chunks: list[np.ndarray] = []
    zero_rows = 0
    try:
        game = instance.get_game()
        unreal = game.unreal_service
        actors = run_frame_transaction(instance, apply=lambda: None,
                                       readback=lambda: unreal.find_actors_as_dict())
        navigation_actors = {name: handle for name, handle in actors.items()
                             if "nav" in str(name).lower()}
        _require(bool(navigation_actors), "the cooked map has no navigation actor")
        data_name = sorted(navigation_actors,
                           key=lambda name: (0 if "recast" in name.lower() else 1, name))[0]
        _require("recast" in data_name.lower(),
                 f"only {data_name!r} is available; AbstractNavData returns zeroed points")
        navigation = game.navigation_service
        navigation_data = navigation_actors[data_name]
        for call in range(args.calls):
            points = run_frame_transaction(
                instance, apply=lambda: None,
                readback=lambda: navigation.get_random_points(
                    navigation_data=navigation_data, num_points=args.samples_per_call))
            block = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            keep = (block != 0).any(axis=1)
            zero_rows += int((~keep).sum())
            chunks.append(block[keep])
            if (call + 1) % 10 == 0 or call + 1 == args.calls:
                total = sum(len(c) for c in chunks)
                print(f"navmesh points: {call + 1}/{args.calls} calls, {total} points, "
                      f"{time.time() - started:.1f} s", flush=True)
    finally:
        try:
            instance.close(force=True)
        except Exception:
            pass
    sampled = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    _require(sampled.shape[0] >= 1000, "navigation returned too few usable points")
    facts = {"kind": "ue_navmesh_random_points", "navigation_data_actor": data_name,
             "native_map": native_map, "spear_executable": str(executable),
             "stage_root": str(args.stage_root), "calls": int(args.calls),
             "samples_per_call": int(args.samples_per_call),
             "points_returned": int(sampled.shape[0]), "zero_rows_dropped": zero_rows,
             "launch_seconds": round(launch_seconds, 1),
             "engine_wall_clock_seconds": round(time.time() - started, 1),
             "point_z_ue_cm": {"min": float(sampled[:, 2].min()),
                               "max": float(sampled[:, 2].max()),
                               "median": float(np.median(sampled[:, 2]))},
             "min_points_per_cell": int(args.min_points_per_cell)}
    return sampled, facts


def grid_from_navmesh_points(points_xyz: np.ndarray, cell_cm: float,
                             min_points: int) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    counts, origin = rasterize_points(points_xyz[:, :2], cell_cm)
    walkable = counts >= int(min_points)
    return walkable, origin, counts


def stamp_scene_points(walkable: np.ndarray, origin: tuple[float, float], cell_cm: float,
                       points: Sequence[Sequence[float]]) -> dict:
    """Mark the cells of the scene's own navigation points walkable (in place).

    Returns how many cells were added and how many points fell outside the
    raster altogether (those cannot be stamped and will fail validation)."""
    added = 0
    outside = 0
    rows, cols = walkable.shape
    for xy in points:
        col = math.floor((float(xy[0]) - origin[0]) / cell_cm)
        row = math.floor((float(xy[1]) - origin[1]) / cell_cm)
        if not (0 <= row < rows and 0 <= col < cols):
            outside += 1
            continue
        if not walkable[row, col]:
            walkable[row, col] = True
            added += 1
    return {"cells_added_from_scene_points": added,
            "scene_points_outside_raster": outside}


# --------------------------------------------------------------------------
# source B: an existing feasible-region raster
# --------------------------------------------------------------------------

def grid_from_feasible_region(config: dict) -> tuple[np.ndarray, tuple[float, float], float, dict]:
    required = ("metadata", "metadata_key", "arrays", "mask_key", "coordinate_contract")
    missing = [key for key in required if key not in config]
    _require(not missing, f"line_of_sight_grid missing keys: {missing}")
    _require(config["coordinate_contract"] == "habitat_xz_m_to_ue_xy_cm_v1",
             "unsupported feasible-region coordinate contract")
    metadata_doc = json.loads(Path(config["metadata"]).read_text())
    metadata = metadata_doc.get(config["metadata_key"])
    _require(isinstance(metadata, dict)
             and metadata.get("schema") == "avengine_room_feasible_region_v1",
             "feasible-region metadata has the wrong schema")
    with np.load(Path(config["arrays"])) as arrays:
        _require(config["mask_key"] in arrays.files, "feasible-region mask key is absent")
        mask = np.asarray(arrays[config["mask_key"]], dtype=bool).copy()
    expected = tuple(int(v) for v in metadata["mask_shape_hw"])
    _require(mask.shape == expected, "feasible-region mask shape differs from metadata")
    pixel_x = float(metadata["pixel_size_x_m"])
    pixel_z = float(metadata["pixel_size_z_m"])
    _require(pixel_x > 0 and math.isclose(pixel_x, pixel_z, rel_tol=1e-6),
             "feasible-region pixels must be square and positive")
    bounds = metadata["bounds_m"]
    # rows index Habitat z (-> UE y), columns index Habitat x (-> UE x), as in
    # scene_sampler.line_of_sight_from_feasible_grid
    origin = (float(bounds[0][0]) * 100.0, float(bounds[0][2]) * 100.0)
    cell_cm = pixel_x * 100.0
    facts = {"kind": "feasible_region_mask", "metadata": str(config["metadata"]),
             "metadata_key": config["metadata_key"], "arrays": str(config["arrays"]),
             "mask_key": config["mask_key"],
             "coordinate_contract": config["coordinate_contract"],
             "region_schema": metadata.get("schema"),
             "region_claim_boundary": metadata.get("claim_boundary"),
             "region_feasible_pixel_count": int(metadata.get("feasible_pixel_count", mask.sum())),
             "region_minimum_rigid_clearance_m": metadata.get("minimum_rigid_clearance_m")}
    return mask, origin, cell_cm, facts


# --------------------------------------------------------------------------
# validation against the scene's own navigation products
# --------------------------------------------------------------------------

def validate_against_scene(grid: WalkableGrid, scene: SS.SceneInputs,
                           margins_m: Sequence[float]) -> dict:
    samples = [xy for route in scene.routes for xy in route.samples_xy]
    clearances = np.asarray([grid.clearance_at(xy) if grid.is_walkable(xy) else -1.0
                             for xy in samples])
    inside = float((clearances >= 0.0).mean()) if len(samples) else float("nan")
    per_margin = {}
    for margin in margins_m:
        margin_cm = float(margin) * 100.0
        per_margin[f"{margin:g}"] = {
            "bank_samples_pass_fraction": float((clearances >= margin_cm).mean()),
            "bank_routes_fully_pass": int(sum(
                1 for route in scene.routes
                if all(grid.is_walkable(xy, margin_cm) for xy in route.samples_xy))),
            "cells_with_margin": int(grid.cells_with_clearance(margin_cm).size),
            "area_with_margin_m2": round(
                grid.cells_with_clearance(margin_cm).size * (grid.cell_cm / 100.0) ** 2, 3)}
    return {"routes_loaded": len(scene.routes), "bank_samples": len(samples),
            "bank_samples_inside_fraction": inside,
            "bank_samples_clearance_cm_percentiles": {
                str(p): float(np.percentile(clearances[clearances >= 0], p))
                for p in (5, 25, 50, 75, 95)} if (clearances >= 0).any() else None,
            "camera_points": len(scene.camera_points),
            "camera_points_inside_fraction": grid.fraction_inside(scene.camera_points),
            "by_margin_m": per_margin}


def run(args: argparse.Namespace) -> dict:
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    config = SS.read_scene_config(args.scene_config)
    scene = SS.load_scene(config)
    if args.from_ue_navmesh:
        render = config.get("render") or {}
        _require("native_map" in render, "scene config render.native_map is required")
        _require(args.stage_root and args.spear_executable,
                 "--stage-root and --spear-executable are required for the navmesh source")
        points, source = navmesh_random_points(args, str(render["native_map"]))
        walkable, origin, counts = grid_from_navmesh_points(points, args.cell_cm,
                                                            args.min_points_per_cell)
        cell_cm = float(args.cell_cm)
        occupied = counts[walkable]
        source["points_per_walkable_cell"] = {
            "min": int(occupied.min()), "median": float(np.median(occupied)),
            "max": int(occupied.max())}
        source["cells_dropped_below_min_points"] = int(((counts > 0) & ~walkable).sum())
        extra = {"point_counts": counts.astype(np.int32),
                 "navmesh_points_xyz_cm": points.astype(np.float32)}
    else:
        _require("line_of_sight_grid" in config,
                 "scene config declares no line_of_sight_grid to convert")
        walkable, origin, cell_cm, source = grid_from_feasible_region(config["line_of_sight_grid"])
        extra = None
    scene_points = [xy for route in scene.routes for xy in route.samples_xy]
    scene_points.extend(scene.camera_points)
    source.update(stamp_scene_points(walkable, origin, cell_cm, scene_points))
    source["scene_points_stamped"] = len(scene_points)
    args.output.mkdir(parents=True)
    write_walkable_grid(args.output, scene_id=str(config["scene_id"]), cell_cm=cell_cm,
                        origin_xy_cm=origin, walkable=walkable, source=source,
                        code=git_worktree_state(), validation=None, extra_arrays=extra)
    grid = WalkableGrid.load(args.output)
    margins = args.report_margin_m or [0.2, 0.3]
    validation = validate_against_scene(grid, scene, margins)
    validation["min_bank_inside_fraction_required"] = float(args.min_bank_inside_fraction)
    validation["passed"] = bool(
        validation["bank_samples_inside_fraction"] >= args.min_bank_inside_fraction
        and validation["camera_points_inside_fraction"] >= args.min_bank_inside_fraction)
    index_path = args.output / "walkable_grid.json"
    index = json.loads(index_path.read_text())
    index["validation"] = validation
    index["scene_config"] = str(args.scene_config.resolve())
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1))
    summary = {"output": str(args.output), "scene_id": grid.scene_id, "schema": SCHEMA,
               "cell_cm": grid.cell_cm, "shape_hw": list(grid.shape),
               "walkable_area_m2": round(grid.walkable_area_m2, 2),
               "source_kind": source["kind"], "validation": validation}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    _require(validation["passed"],
             "validation failed: bank samples or camera points fall outside the grid "
             "(evidence kept in the output directory)")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
