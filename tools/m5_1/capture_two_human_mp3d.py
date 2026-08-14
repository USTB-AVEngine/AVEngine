#!/usr/bin/env python3
"""Run the Habitat-native MP3D two-human production visual capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.m5_1.two_human_capture import (
    TwoHumanCaptureError,
    capture_two_human_mp3d,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-request", type=Path, required=True)
    parser.add_argument("--suite-plan", type=Path, required=True)
    parser.add_argument("--sensor-rig", type=Path, required=True)
    parser.add_argument("--trajectory-bank", type=Path, required=True)
    parser.add_argument("--rir-plan", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--m1-request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = capture_two_human_mp3d(
            atom_request_path=args.atom_request,
            suite_plan_path=args.suite_plan,
            sensor_rig_path=args.sensor_rig,
            trajectory_bank_path=args.trajectory_bank,
            rir_plan_path=args.rir_plan,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            output_dir=args.output,
            runtime_root=args.runtime_root,
        )
    except (TwoHumanCaptureError, RuntimeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    evidence = result.evidence
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "status_scope": evidence["status_scope"],
                "backend_role": evidence["backend_role"],
                "output": str(result.output_dir),
                "evidence": str(result.output_dir / "evidence.json"),
                "frame_count": evidence["frame_count"],
                "research_only": evidence["research_only"],
                "manual_review_status": evidence["manual_review_status"],
                "formal_dataset_count": evidence["formal_dataset_count"],
                "semantic_visible_frame_count": evidence[
                    "semantic_visible_frame_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
