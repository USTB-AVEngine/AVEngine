#!/usr/bin/env python3
"""Capture the committed 270-frame Rocketbox-human + Beagle legacy route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.capture.mixed_capture import MixedCaptureError, capture_legacy_route


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--m1-request", type=Path, required=True)
    parser.add_argument("--human-runtime-glb", type=Path, required=True)
    parser.add_argument("--beagle-manifest", type=Path, required=True)
    parser.add_argument("--beagle-m2-request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = capture_legacy_route(
            route_manifest_path=args.route_manifest,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            human_runtime_glb_path=args.human_runtime_glb,
            beagle_animal_manifest_path=args.beagle_manifest,
            beagle_m2_request_path=args.beagle_m2_request,
            output_dir=args.output,
            runtime_root=args.runtime_root,
        )
    except (MixedCaptureError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": result.evidence["status"],
                "output": str(result.output_dir),
                "frame_count": result.evidence["frame_count"],
                "semantic_visible_frame_count": result.evidence["readback"][
                    "semantic_visible_frame_count"
                ],
                "evidence_content_sha256": result.evidence[
                    "evidence_content_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
