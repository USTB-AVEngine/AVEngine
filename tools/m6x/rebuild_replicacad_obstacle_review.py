#!/usr/bin/env python3
"""Rebuild the retained ReplicaCAD review with live furniture obstacles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from avengine.m6x.replicacad import (  # noqa: E402
    rebuild_replicacad_obstacle_review,
)


def _default(relative: str) -> Path:
    return REPOSITORY_ROOT / relative


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse retained ReplicaCAD RGB/audio while rebuilding its Topdown and "
            "source-center gate from the live apt_0 navmesh and 113 rigid OBBs."
        )
    )
    parser.add_argument(
        "--replicacad-root",
        type=Path,
        default=_default("tmp/m6x/datasets/replica_cad"),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=REPOSITORY_ROOT.parent / "habitat-sim-AVEngine",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=_default("tmp/m5_1/replicacad_mixed_20260719_04"),
    )
    parser.add_argument(
        "--delivery-dir",
        type=Path,
        default=_default("tmp/m5_1/replicacad_delivery_20260719_03"),
    )
    parser.add_argument(
        "--room-manifest",
        type=Path,
        default=_default(
            "examples/m5_1/replicacad_articulated_review/room_manifest.json"
        ),
    )
    parser.add_argument(
        "--m1-request",
        type=Path,
        default=_default(
            "examples/m5_1/replicacad_articulated_review/capture_request.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default("tmp/m6x/replicacad_obstacle_review_20260719_01"),
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=0.02,
        help="Live PathFinder Topdown resolution in meters per pixel.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    status = rebuild_replicacad_obstacle_review(
        replicacad_root=args.replicacad_root,
        runtime_root=args.runtime_root,
        capture_dir=args.capture_dir,
        delivery_dir=args.delivery_dir,
        room_manifest_path=args.room_manifest,
        m1_request_path=args.m1_request,
        output_dir=args.output,
        meters_per_pixel=args.meters_per_pixel,
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
