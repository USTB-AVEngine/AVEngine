#!/usr/bin/env python3
"""Materialize CPU research actor tracks from an MP3D region route bank."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from avengine.m5.mp3d_region_actor_track_materializer import (
    MP3DRegionActorTrackError,
    materialize_region_actor_tracks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-bank", required=True)
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--base-m2-request", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = materialize_region_actor_tracks(
            route_bank_path=args.route_bank,
            region_instance_id=args.region_id,
            asset_manifest_path=args.asset_manifest,
            base_m2_request_path=args.base_m2_request,
            output_root=args.output_root,
        )
    except (MP3DRegionActorTrackError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "output_written": False}))
        return 2
    print(json.dumps({"output_root": str(output), "output_written": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
