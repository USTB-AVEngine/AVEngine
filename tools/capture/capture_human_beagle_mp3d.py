#!/usr/bin/env python3
"""Run the real-navmesh 270-frame MP3D human + Beagle visual canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.capture.mp3d_capture import MP3DCaptureError, capture_mp3d_route


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--m1-request", type=Path, required=True)
    parser.add_argument("--human-runtime-glb", type=Path, required=True)
    parser.add_argument("--beagle-manifest", type=Path, required=True)
    parser.add_argument("--beagle-m2-request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument(
        "--runtime-prefix",
        type=Path,
        help="Non-Git installed Habitat runtime prefix",
    )
    runtime.add_argument(
        "--runtime-root",
        type=Path,
        help="Compatibility alias for --runtime-prefix; Git checkouts are rejected",
    )
    parser.add_argument(
        "--mp3d-root",
        type=Path,
        help="External MP3D data root containing scene_datasets",
    )
    parser.add_argument(
        "--pbr-asset-root",
        type=Path,
        required=True,
        help="External non-Git Brown Photostudio PBR IBL asset root",
    )
    parser.add_argument(
        "--magnum-python-site",
        type=Path,
        help="External Corrade/Magnum Python site; otherwise AVENGINE_HABITAT_MAGNUM_PYTHON_SITE",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = capture_mp3d_route(
            route_manifest_path=args.route_manifest,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            human_runtime_glb_path=args.human_runtime_glb,
            beagle_animal_manifest_path=args.beagle_manifest,
            beagle_m2_request_path=args.beagle_m2_request,
            output_dir=args.output,
            runtime_prefix=args.runtime_prefix,
            runtime_root=args.runtime_root,
            mp3d_root=args.mp3d_root,
            pbr_asset_root=args.pbr_asset_root,
            magnum_python_site=args.magnum_python_site,
        )
    except (MP3DCaptureError, OSError, ValueError) as exc:
        parser.error(str(exc))
    evidence = result.gate_evidence
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "output": str(result.capture.output_dir),
                "frame_count": evidence["frame_count"],
                "visible_frames": evidence["mixed_capture"][
                    "semantic_visible_frame_count"
                ],
                "minimum_center_separation_m": evidence["pathfinder"][
                    "geometry"
                ]["minimum_center_separation_m"],
                "gate_evidence_content_sha256": evidence[
                    "evidence_content_sha256"
                ],
                "contact_sheet": str(result.contact_sheet_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
