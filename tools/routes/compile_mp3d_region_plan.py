#!/usr/bin/env python3
"""Build a bounded CPU MP3D .house region/camera/source-route plan.

This entrypoint only reads external room/camera/navmesh inputs and writes a
research-only planning artifact. It does not start a Simulator, GPU capture,
RLR, audio rendering or QuestionSpec generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
from avengine.rooms.mp3d_regions import parse_mp3d_house
from avengine.routes.mp3d_region_planner import build_region_route_plan
from avengine.routes.mp3d_region_views import select_region_cameras


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", required=True, type=Path)
    parser.add_argument(
        "--camera-sidecar",
        required=True,
        type=Path,
        help="existing camera placement membership/bank JSON",
    )
    parser.add_argument("--navmesh", required=True, type=Path)
    parser.add_argument("--runtime-prefix", required=True, type=Path)
    parser.add_argument("--magnum-python-site", required=True, type=Path)
    parser.add_argument("--mp3d-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--region-index", type=int, action="append")
    parser.add_argument("--cameras-per-region", type=int, default=1)
    parser.add_argument("--route-families-per-region", type=int, default=1)
    parser.add_argument("--motion-case", action="append", dest="motion_cases")
    parser.add_argument("--frame-count", type=int, default=75)
    parser.add_argument("--frame-rate-hz", type=int, default=15)
    parser.add_argument("--sample-spacing-m", type=float, default=0.50)
    parser.add_argument("--maximum-candidate-points", type=int, default=64)
    parser.add_argument("--maximum-route-attempts", type=int, default=256)
    parser.add_argument("--minimum-pair-separation-m", type=float, default=0.0)
    return parser


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    house_path = args.house.expanduser().resolve(strict=True)
    sidecar_path = args.camera_sidecar.expanduser().resolve(strict=True)
    navmesh_path = args.navmesh.expanduser().resolve(strict=True)
    output = args.output.expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output exists; refusing to replace: {output}")
    if not navmesh_path.is_file():
        raise ValueError(f"navmesh is not a regular file: {navmesh_path}")

    house_plan = parse_mp3d_house(house_path)
    sidecar = _json(sidecar_path)
    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=args.runtime_prefix,
        mp3d_root=args.mp3d_root,
        magnum_python_site=args.magnum_python_site,
        rlr_sdk_root=None,
    )
    pathfinder = runtime.habitat_sim.PathFinder()
    if not bool(pathfinder.load_nav_mesh(str(navmesh_path))):
        raise RuntimeError(f"Habitat PathFinder could not load navmesh: {navmesh_path}")
    if not bool(pathfinder.is_loaded):
        raise RuntimeError("Habitat PathFinder reports is_loaded=false")

    cameras = select_region_cameras(
        house_plan,
        sidecar,
        region_indices=args.region_index,
        cameras_per_region=args.cameras_per_region,
        seed=args.seed,
    )
    motion_cases = (
        tuple(args.motion_cases)
        if args.motion_cases
        else None
    )
    kwargs: dict[str, Any] = {
        "region_indices": args.region_index,
        "route_families_per_region": args.route_families_per_region,
        "frame_count": args.frame_count,
        "frame_rate_hz": args.frame_rate_hz,
        "sample_spacing_m": args.sample_spacing_m,
        "maximum_candidate_points": args.maximum_candidate_points,
        "maximum_route_attempts": args.maximum_route_attempts,
        "minimum_pair_separation_m": args.minimum_pair_separation_m,
        "seed": args.seed,
    }
    if motion_cases is not None:
        kwargs["motion_cases"] = motion_cases
    plan = build_region_route_plan(
        house_plan,
        cameras,
        pathfinder,
        runtime.habitat_sim.ShortestPath,
        **kwargs,
    )
    plan["inputs"] = {
        "house": str(house_path),
        "camera_sidecar": str(sidecar_path),
        "navmesh": str(navmesh_path),
        "runtime_prefix": str(runtime.prefix),
        "magnum_python_site": str(runtime.magnum_python_site),
        "mp3d_root": None if runtime.mp3d_root is None else str(runtime.mp3d_root),
    }
    plan["runtime"] = {
        "pathfinder_loaded": True,
        "planner_execution": "CPU_PathFinder_only",
        "native_simulator_started": False,
        "gpu_started": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_plan(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc), "output_written": False}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "regions": plan["region_count"],
                "route_families": plan["route_family_count"],
                "cases": plan["case_count"],
                "frame_count": plan["parameters"]["frame_count"],
                "frame_rate_hz": plan["parameters"]["frame_rate_hz"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
