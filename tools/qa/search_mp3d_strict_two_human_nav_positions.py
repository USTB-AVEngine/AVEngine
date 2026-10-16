#!/usr/bin/env python3
"""Search the real MP3D navmesh for a safer two-adult static probe pair."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CAMERA = [-4.1499128342, 1.572447, -1.2454376221]
ROOT_Y = 0.072447
MOUTH_HEIGHTS = [1.61, 1.569012451171875]


def _project(root: Sequence[float], mouth_height: float) -> list[float]:
    depth = 100.0 * (CAMERA[2] - root[2])
    if depth <= 0.0:
        return [math.nan, math.nan]
    horizontal = 100.0 * (root[0] - CAMERA[0])
    vertical = 100.0 * (root[1] + mouth_height - CAMERA[1])
    return [639.5 + 640.0 * horizontal / depth, 359.5 - 640.0 * vertical / depth]


def search(navmesh: Path, *, step_m: float) -> dict[str, Any]:
    import habitat_sim
    import numpy as np

    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh)):
        raise RuntimeError(f"could not load navmesh: {navmesh}")
    candidates = []
    for x in np.arange(-5.8, -2.5 + step_m / 2.0, step_m):
        for z in np.arange(-3.5, -2.05 + step_m / 2.0, step_m):
            requested = np.asarray([x, ROOT_Y, z], dtype=np.float32)
            if not pathfinder.is_navigable(requested):
                continue
            snapped = pathfinder.snap_point(requested)
            error = float(np.linalg.norm(snapped - requested))
            if error > 1.0e-4 or int(pathfinder.get_island(snapped)) != 1:
                continue
            clearance = float(pathfinder.distance_to_closest_obstacle(snapped, 10.0))
            if clearance < 0.50:
                continue
            positions = [float(value) for value in snapped]
            projected = [_project(positions, height) for height in MOUTH_HEIGHTS]
            if not all(
                48.0 <= uv[0] < 1280.0 - 48.0 and 48.0 <= uv[1] < 720.0 - 48.0
                for uv in projected
            ):
                continue
            candidates.append(
                {
                    "requested_m": [float(value) for value in requested],
                    "snap_point_m": positions,
                    "snap_error_m": error,
                    "clearance_m": clearance,
                    "island_id": 1,
                }
            )
    pairs = []
    for first in candidates:
        for second in candidates:
            if first["snap_point_m"][0] >= second["snap_point_m"][0]:
                continue
            dx = first["snap_point_m"][0] - second["snap_point_m"][0]
            dz = first["snap_point_m"][2] - second["snap_point_m"][2]
            separation = math.hypot(dx, dz)
            if separation < 1.30:
                continue
            uv1 = _project(first["snap_point_m"], MOUTH_HEIGHTS[0])
            uv2 = _project(second["snap_point_m"], MOUTH_HEIGHTS[1])
            pixel_separation = uv2[0] - uv1[0]
            if (
                pixel_separation < 320.0
                or uv1[0] > 0.42 * 1280.0
                or uv2[0] < 0.58 * 1280.0
            ):
                continue
            min_clearance = min(first["clearance_m"], second["clearance_m"])
            midpoint_x = (first["snap_point_m"][0] + second["snap_point_m"][0]) / 2.0
            depth_delta = abs(first["snap_point_m"][2] - second["snap_point_m"][2])
            score = (
                4.0 * min_clearance
                - abs(separation - 1.5)
                - 0.8 * abs(midpoint_x - CAMERA[0])
                - depth_delta
            )
            pairs.append(
                {
                    "score": score,
                    "horizontal_separation_m": separation,
                    "mouth_pixel_separation": pixel_separation,
                    "source1": {**first, "mouth_pixel_uv": uv1},
                    "source2": {**second, "mouth_pixel_uv": uv2},
                }
            )
    pairs.sort(key=lambda item: item["score"], reverse=True)
    if not pairs:
        raise RuntimeError("no pair passed the strict search gates")
    return {
        "schema": "avengine_mp3d_strict_two_human_navmesh_pair_search_v1",
        "status": "pass",
        "navmesh_path": str(navmesh),
        "grid_step_m": step_m,
        "candidate_count": len(candidates),
        "passing_pair_count": len(pairs),
        "requirements": {
            "same_island_id": 1,
            "minimum_clearance_m": 0.5,
            "minimum_horizontal_separation_m": 1.3,
            "minimum_mouth_pixel_separation": 320.0,
            "maximum_source1_mouth_x_fraction": 0.42,
            "minimum_source2_mouth_x_fraction": 0.58,
            "mouth_envelope_radius_px": 48.0,
        },
        "selected": pairs[0],
        "top_pairs": pairs[:20],
        "semantics": "navmesh root clearance plus planned mouth-proxy projection only",
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navmesh", required=True, type=Path)
    parser.add_argument("--step-m", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.step_m <= 0.0:
        parser.error("--step-m must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(search(args.navmesh, step_m=args.step_m), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
