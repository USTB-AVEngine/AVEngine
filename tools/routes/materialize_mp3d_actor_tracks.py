#!/usr/bin/env python3
"""Build CPU Habitat apply tracks for one planned MP3D region case.

The command validates each explicit M2 asset package and base request, samples
its baked action loops, and writes planned root/joint targets. It never starts
Habitat, GPU, UE, or RLR; native emitter/support/collision readback remains
pending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.assets.mp3d_region_actor_tracks import (
    MP3DRegionActorTrackError,
    materialize_region_actor_tracks,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-plan", required=True, type=Path)
    parser.add_argument("--planned-timeline", required=True, type=Path)
    parser.add_argument("--room-manifest", required=True, type=Path)
    parser.add_argument("--m1-request", required=True, type=Path)
    parser.add_argument("--actor-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--frame-rate-hz", type=float)
    parser.add_argument("--time-base-hz", type=int)
    parser.add_argument("--ticks-per-frame", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = materialize_region_actor_tracks(
            region_plan_path=args.region_plan,
            planned_timeline_path=args.planned_timeline,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            actor_config=args.actor_config,
            output_directory=args.output,
            frame_count=args.frame_count,
            frame_rate_hz=args.frame_rate_hz,
            time_base_hz=args.time_base_hz,
            ticks_per_frame=args.ticks_per_frame,
        )
    except (MP3DRegionActorTrackError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "output_written": False}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "region_instance_id": receipt["region_instance_id"],
                "route_family_id": receipt["route_family_id"],
                "motion_case": receipt["motion_case"],
                "actor_count": len(receipt["actors"]),
                "frame_count": receipt["clock"]["frame_count"],
                "frame_rate_hz": receipt["clock"]["frame_rate_hz"],
                "native_capture": receipt["native_capture"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
