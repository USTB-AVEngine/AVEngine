#!/usr/bin/env python3
"""Materialize one MP3D region route case into current AVEngine inputs.

This CPU tool writes a fresh research-only directory containing an M1 request,
actor/endpoint mappings, explicitly planned frame records, and an optional
AudioProgram. Planned records are not native observations. The tool does not
start Habitat, native capture, GPU work, or RLR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.routes.mp3d_region_materializer import (
    DEFAULT_TIME_BASE_HZ,
    MP3DRegionMaterializationError,
    materialize_region_case,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-plan", required=True, type=Path)
    parser.add_argument("--room-manifest", required=True, type=Path)
    parser.add_argument("--m1-request", required=True, type=Path)
    parser.add_argument("--actor-selection", required=True, type=Path)
    parser.add_argument("--source-endpoint-registry", required=True, type=Path)
    parser.add_argument("--audio-program", type=Path)
    parser.add_argument("--sound-asset-registry", type=Path)
    parser.add_argument("--region-index", required=True, type=int)
    parser.add_argument("--route-family-id")
    parser.add_argument("--motion-case", required=True)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--frame-rate-hz", type=float)
    parser.add_argument("--time-base-hz", type=int, default=DEFAULT_TIME_BASE_HZ)
    parser.add_argument("--ticks-per-frame", type=int)
    parser.add_argument("--request-id")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sound_asset_registry is not None and args.audio_program is None:
        raise SystemExit("--sound-asset-registry requires --audio-program")
    try:
        receipt = materialize_region_case(
            args.region_plan,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            actor_selection_path=args.actor_selection,
            source_endpoint_registry_path=args.source_endpoint_registry,
            audio_program_path=args.audio_program,
            sound_asset_registry_path=args.sound_asset_registry,
            output_directory=args.output,
            region_index=args.region_index,
            route_family_id=args.route_family_id,
            motion_case=args.motion_case,
            frame_count=args.frame_count,
            frame_rate_hz=args.frame_rate_hz,
            time_base_hz=args.time_base_hz,
            ticks_per_frame=args.ticks_per_frame,
            request_id=args.request_id,
        )
    except (MP3DRegionMaterializationError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "output_written": False}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "house_id": receipt["room"]["house_id"],
                "region_index": receipt["region"]["region_index"],
                "route_family_id": receipt["route"]["route_family_id"],
                "motion_case": receipt["route"]["motion_case"],
                "frame_count": receipt["planned_clock"]["frame_count"],
                "frame_rate_hz": receipt["planned_clock"]["frame_rate_hz"],
                "sample_count": receipt["planned_clock"]["sample_count"],
                "audio_status": receipt["audio"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
