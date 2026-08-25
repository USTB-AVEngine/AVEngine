#!/usr/bin/env python3
"""Build the CPU MP3D region source-route mini-pilot.

This command loads a real Habitat PathFinder against the explicitly supplied
MP3D navmesh, then generates source routes inside each .house region.  It never
starts a Simulator, renderer, GPU job, UE process or RLR context.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import numpy as np

from avengine.m1.habitat_capture import prepare_installed_habitat_runtime
from avengine.m5.mp3d_region_source_planner import (
    MP3DRegionSourcePlannerError,
    load_region_source_inputs,
)


class BuildError(ValueError):
    """The explicit CPU region route build cannot be published."""


def _required_file(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise BuildError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise BuildError(f"{owner} is not a regular file: {resolved}")
    return resolved


def _required_dir(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise BuildError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_dir():
        raise BuildError(f"{owner} is not a directory: {resolved}")
    return resolved


def _fresh_output(path: Path) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute():
        raise BuildError("--output must be absolute")
    parent = raw.parent.resolve(strict=True)
    output = parent / raw.name
    if os.path.lexists(output):
        raise BuildError(f"refusing to replace output: {output}")
    return output


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> Path:
    bank_path = _required_file(
        args.region_viewpoint_bank, owner="--region-viewpoint-bank"
    )
    sidecar_path = _required_file(
        args.camera_region_sidecar, owner="--camera-region-sidecar"
    )
    house_path = _required_file(args.house, owner="--house")
    navmesh_path = _required_file(args.navmesh, owner="--navmesh")
    runtime_prefix = _required_dir(args.runtime_prefix, owner="--runtime-prefix")
    mp3d_root = _required_dir(args.mp3d_root, owner="--mp3d-root")
    magnum_site = _required_dir(args.magnum_python_site, owner="--magnum-python-site")
    rlr_sdk_root = _required_dir(args.rlr_sdk_root, owner="--rlr-sdk-root")
    output = _fresh_output(args.output)

    try:
        runtime = prepare_installed_habitat_runtime(
            runtime_prefix=runtime_prefix,
            mp3d_root=mp3d_root,
            magnum_python_site=magnum_site,
            rlr_sdk_root=rlr_sdk_root,
            allow_mp3d_environment=False,
        )
        habitat_sim = runtime.habitat_sim
        pathfinder = habitat_sim.PathFinder()
        loaded = bool(pathfinder.load_nav_mesh(str(navmesh_path)))
        if not loaded or not bool(pathfinder.is_loaded):
            raise BuildError(
                f"Habitat PathFinder could not load declared navmesh: {navmesh_path}"
            )
        bounds = np.asarray(pathfinder.get_bounds(), dtype=np.float64)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            raise BuildError("loaded PathFinder returned invalid bounds")
        record = load_region_source_inputs(
            bank_path=bank_path,
            sidecar_path=sidecar_path,
            house_path=house_path,
            pathfinder=pathfinder,
            shortest_path_factory=habitat_sim.ShortestPath,
            seed=args.seed,
            sample_spacing_m=args.sample_spacing_m,
            maximum_y_delta_m=args.maximum_y_delta_m,
        )
        record["inputs"] = {
            "region_viewpoint_bank": str(bank_path),
            "camera_region_sidecar": str(sidecar_path),
            "house": str(house_path),
            "navmesh": str(navmesh_path),
            "runtime_prefix": str(runtime_prefix),
            "mp3d_root": str(mp3d_root),
            "magnum_python_site": str(magnum_site),
            "rlr_sdk_root": str(rlr_sdk_root),
        }
        record["pathfinder_authority"] = {
            "backend": "Habitat PathFinder",
            "execution": "standalone_cpu_only",
            "navmesh_loaded": True,
            "navmesh_path": str(navmesh_path),
            "bounds_m": bounds.tolist(),
            "simulator_started": False,
            "renderer_started": False,
            "gpu_started": False,
            "ue_started": False,
        }
    except (
        BuildError,
        MP3DRegionSourcePlannerError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise BuildError(str(exc)) from exc

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        _write_json(staging / "region_source_route_bank.json", record)
        (staging / "README.md").write_text(
            "# MP3D region source-route CPU mini-pilot\n\n"
            "This artifact uses a standalone Habitat PathFinder loaded from the "
            "declared MP3D navmesh and .house region polygons. Camera requests are "
            "metadata bindings only. All 40 bindings and 160 75-frame motion cases "
            "are retained; native actor collision, visibility, audio/RIR and formal "
            "Episode status remain pending.\n",
            encoding="utf-8",
        )
        if os.path.lexists(output):
            raise BuildError(f"output appeared during build: {output}")
        os.rename(staging, output)
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-viewpoint-bank", required=True, type=Path)
    parser.add_argument("--camera-region-sidecar", required=True, type=Path)
    parser.add_argument("--house", required=True, type=Path)
    parser.add_argument("--navmesh", required=True, type=Path)
    parser.add_argument("--runtime-prefix", required=True, type=Path)
    parser.add_argument("--mp3d-root", required=True, type=Path)
    parser.add_argument("--magnum-python-site", required=True, type=Path)
    parser.add_argument("--rlr-sdk-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20_260_824)
    parser.add_argument("--sample-spacing-m", type=float, default=0.10)
    parser.add_argument("--maximum-y-delta-m", type=float, default=0.30)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output = run(parse_args(argv))
    except (BuildError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "pass", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
