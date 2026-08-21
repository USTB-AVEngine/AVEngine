#!/usr/bin/env python3
"""Probe a static-camera capture for scheduled-but-unrendered walk animation.

Room-agnostic: needs only frame_records.json (declared walk phases, static
camera) and arrays/rgb.npy. Pass the slots in left-to-right image order for
the probed route. Exit code 2 when any slot slides without animation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.m7.animation_probe import (  # noqa: E402
    AnimationProbeError,
    probe_capture_animation_playback,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-capture-dir", required=True, type=Path)
    parser.add_argument(
        "--slot-order-left-to-right",
        required=True,
        help="comma-separated slot ids in left-to-right image order",
    )
    parser.add_argument("--sliding-max", type=float, default=7.5)
    parser.add_argument("--animated-min", type=float, default=9.5)
    args = parser.parse_args()
    try:
        report = probe_capture_animation_playback(
            args.visual_capture_dir,
            slot_order_left_to_right=tuple(
                part.strip()
                for part in args.slot_order_left_to_right.split(",")
                if part.strip()
            ),
            sliding_max=args.sliding_max,
            animated_min=args.animated_min,
        )
    except (AnimationProbeError, OSError, ValueError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
