#!/usr/bin/env python3
"""Build all MP3D region camera/route mini-pilot bindings on CPU."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from avengine.m5.mp3d_region_capture_bindings import (
    MP3DRegionCaptureBindingError,
    build_region_capture_binding_bank,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewpoint-bank", required=True)
    parser.add_argument("--route-bank", required=True)
    parser.add_argument("--actor-tracks-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = build_region_capture_binding_bank(
            viewpoint_bank_path=args.viewpoint_bank,
            route_bank_path=args.route_bank,
            actor_tracks_root=args.actor_tracks_root,
            output_directory=args.output,
        )
    except (MP3DRegionCaptureBindingError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "artifact": str(output / "capture_binding_bank.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
