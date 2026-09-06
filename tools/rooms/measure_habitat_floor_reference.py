#!/usr/bin/env python3
"""Measure Habitat floor levels from a declared scene and navmesh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

SCHEMA = "avengine_qa_habitat_floor_reference_v1"


def _write_fresh(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")


def measure(args: argparse.Namespace) -> dict[str, Any]:
    scene = args.scene_glb.expanduser().resolve()
    navmesh = args.navmesh.expanduser().resolve()
    dataset = args.dataset_config.expanduser().resolve()
    for path in (scene, navmesh, dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_python_site,
        rlr_sdk_root=args.rlr_sdk_root,
        mp3d_root=args.mp3d_root,
        allow_mp3d_environment=False,
    )
    hs = runtime.habitat_sim
    sim_cfg = hs.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene)
    sim_cfg.scene_dataset_config_file = str(dataset)
    sim_cfg.load_semantic_mesh = False
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = int(args.gpu_device_id)
    simulator = hs.Simulator(hs.Configuration(sim_cfg, [hs.AgentConfiguration()]))
    try:
        pathfinder = simulator.pathfinder
        loaded = bool(pathfinder.is_loaded) or bool(pathfinder.load_nav_mesh(str(navmesh)))
        if not loaded or not pathfinder.is_loaded:
            raise RuntimeError(f"declared Habitat navmesh did not load: {navmesh}")
        pathfinder.seed(int(args.seed))
        rows = []
        for index in range(int(args.samples)):
            query = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
            snapped = np.asarray(pathfinder.snap_point(query), dtype=np.float64)
            rows.append({
                "index": index,
                "query_m": query.tolist(),
                "snap_m": snapped.tolist(),
                "floor_height_m": float(snapped[1]),
                "is_navigable": bool(pathfinder.is_navigable(query)),
                "snap_distance_m": float(np.linalg.norm(snapped - query)),
                "island": int(pathfinder.get_island(query)),
            })
    finally:
        simulator.close()
    heights = np.asarray([row["floor_height_m"] for row in rows], dtype=np.float64)
    if heights.size == 0 or not np.all(np.isfinite(heights)):
        raise RuntimeError("no finite Habitat floor samples")
    values, counts = np.unique(np.round(heights, 4), return_counts=True)
    levels = [
        {"floor_height_m": float(value), "sample_count": int(count), "share": float(count / heights.size)}
        for value, count in zip(values, counts, strict=True)
    ]
    output = args.output.expanduser().resolve()
    result = {
        "schema": SCHEMA,
        "status": "measured",
        "room_id": args.room_id,
        "scene_glb": str(scene),
        "navmesh": str(navmesh),
        "dataset_config": str(dataset),
        "method": {
            "kind": "habitat_pathfinder_snap_point_v1",
            "runtime_prefix": str(Path(args.runtime_prefix).expanduser().resolve()),
            "gpu_device_id": int(args.gpu_device_id),
            "declared_navmesh_loaded": True,
            "seed": int(args.seed),
            "samples": int(args.samples),
            "coordinate_frame": {"linear_unit": "meter", "up_axis": "+Y", "handedness": "right"},
        },
        "summary": {
            "sample_count": int(heights.size),
            "navigable_count": int(sum(row["is_navigable"] for row in rows)),
            "height_min_m": float(heights.min()),
            "height_median_m": float(np.median(heights)),
            "height_max_m": float(heights.max()),
            "levels": levels,
        },
        "floor_height_m": float(np.median(heights)),
        "floor_levels_m": [float(value) for value in values],
        "claim_boundary": "PathFinder snap/readback on the declared Habitat navmesh; multi-level scenes retain the observed levels",
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_fresh(output / "floor_reference.json", result)
    _write_fresh(output / "floor_trace_rows.json", {"schema": SCHEMA, "room_id": args.room_id, "rows": rows})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--scene-glb", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-prefix", type=Path, required=True)
    parser.add_argument("--magnum-python-site", type=Path, required=True)
    parser.add_argument("--rlr-sdk-root", type=Path, required=True)
    parser.add_argument("--mp3d-root", type=Path, required=True)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    result = measure(args)
    print(json.dumps({"status": result["status"], "room_id": result["room_id"], "floor_height_m": result["floor_height_m"], "floor_levels_m": result["floor_levels_m"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
