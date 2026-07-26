#!/usr/bin/env python3
"""Select balanced generic source trajectories for one room evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from uuid import uuid4

from avengine.contracts.json_io import load_json, write_json
from avengine.m7.room_evaluation import build_room_evaluation_plan
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-bank", type=Path, required=True)
    parser.add_argument("--template-rir-plan", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=100)
    parser.add_argument("--sound-class", action="append", dest="sound_classes")
    parser.add_argument("--listener-position-m", type=float, nargs=3)
    parser.add_argument("--listener-orientation-wxyz", type=float, nargs=4)
    parser.add_argument("--minimum-listener-source-distance-m", type=float, default=0.0)
    parser.add_argument("--balance-azimuth-regions", action="store_true")
    parser.add_argument("--minimum-azimuth-region-fraction", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory_bank = load_json(args.trajectory_bank.resolve())
    template = load_json(args.template_rir_plan.resolve())
    sound_classes = args.sound_classes or None
    kwargs = {
        "listener_position_m": args.listener_position_m
        or template.get("listener_position_m"),
        "listener_orientation_wxyz": args.listener_orientation_wxyz
        or template.get("listener_orientation_wxyz"),
        "stride_frames": template.get("stride_frames"),
        "episode_count": args.episode_count,
        "minimum_listener_source_distance_m": args.minimum_listener_source_distance_m,
        "balance_azimuth_regions": args.balance_azimuth_regions,
        "minimum_azimuth_region_fraction": args.minimum_azimuth_region_fraction,
    }
    if sound_classes is not None:
        kwargs["sound_classes"] = sound_classes
    plan = build_room_evaluation_plan(trajectory_bank, **kwargs)
    unresolved = args.output.expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    if os.path.lexists(unresolved):
        raise FileExistsError(f"refusing to replace output: {unresolved}")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    output_parent = unresolved.parent.resolve(strict=True)
    policy = WorkspacePathPolicy.from_roots([output_parent])
    output = policy.resolve_output(
        output_parent / unresolved.name,
        owner="room evaluation plan",
    )
    staging = policy.resolve_output(
        output.with_name(f".{output.name}.staging-{uuid4().hex}"),
        owner="room evaluation plan staging directory",
    )
    try:
        staging.mkdir()
        write_json(staging / "trajectory_bank.json", plan.trajectory_bank)
        write_json(staging / "rir_job_plan.json", plan.rir_job_plan)
        write_json(staging / "sound_assignments.json", plan.sound_assignments)
        write_json(staging / "delivery.json", plan.summary)
        output = atomic_publish_directory(policy, staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"ROOM_EVALUATION_PLAN_OK output={output} "
        f"episodes={plan.summary['episode_count']} "
        f"rirs={plan.summary['unique_rir_job_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
