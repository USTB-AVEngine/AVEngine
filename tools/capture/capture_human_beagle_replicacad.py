#!/usr/bin/env python3
"""Run the real ReplicaCAD apt_0 human + Beagle visual/placement review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.m1.habitat_capture import prepare_installed_habitat_runtime
from avengine.capture.mixed_capture import MixedCaptureError
from avengine.capture.replicacad_capture import (
    ReplicaCADCaptureError,
    capture_replicacad_route,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--m1-request", type=Path, required=True)
    parser.add_argument("--human-runtime-glb", type=Path, required=True)
    parser.add_argument("--beagle-manifest", type=Path, required=True)
    parser.add_argument("--beagle-m2-request", type=Path, required=True)
    parser.add_argument("--replicacad-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--runtime-prefix", type=Path)
    parser.add_argument("--magnum-python-site", type=Path)
    parser.add_argument("--rlr-sdk-root", type=Path)
    parser.add_argument("--pbr-asset-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    installed_runtime = None
    if args.runtime_prefix is not None:
        if args.runtime_root is not None:
            parser.error("--runtime-prefix does not accept --runtime-root")
        installed_runtime = prepare_installed_habitat_runtime(
            runtime_prefix=args.runtime_prefix,
            pbr_asset_root=args.pbr_asset_root,
            magnum_python_site=args.magnum_python_site,
            rlr_sdk_root=args.rlr_sdk_root,
        )
    try:
        result = capture_replicacad_route(
            route_manifest_path=args.route_manifest,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            human_runtime_glb_path=args.human_runtime_glb,
            beagle_animal_manifest_path=args.beagle_manifest,
            beagle_m2_request_path=args.beagle_m2_request,
            output_dir=args.output,
            replicacad_root=args.replicacad_root,
            runtime_root=args.runtime_root,
            installed_runtime=installed_runtime,
        )
    except (ReplicaCADCaptureError, MixedCaptureError, OSError, ValueError) as exc:
        parser.error(str(exc))
    evidence = result.gate_evidence
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "research_only": evidence["research_only"],
                "output": str(result.capture.output_dir),
                "frame_count": evidence["frame_count"],
                "passed_gate_count": evidence["passed_gate_count"],
                "gate_count": evidence["gate_count"],
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
