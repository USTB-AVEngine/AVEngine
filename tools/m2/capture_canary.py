#!/usr/bin/env python3
"""Run one formal 75-state M2 canary capture in Habitat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m2.contracts import load_and_validate_inputs as load_m2_inputs
from avengine.m2.habitat_capture import capture_m2_habitat


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--room-request", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = load_m2_inputs(args.asset_manifest, args.request)
    room_inputs = load_m1_inputs(args.room_manifest, args.room_request)
    output = args.output.resolve()
    evidence = capture_m2_habitat(
        inputs,
        room_inputs,
        output,
        runtime_root=args.runtime_root,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "evidence_kind": evidence["evidence_kind"],
                "review_only": evidence["review_only"],
                "asset_admission_state": evidence["asset_admission_state"],
                "formal_view_ids": evidence["formal_view_ids"],
                "review_view_ids": evidence["review_view_ids"],
                "formal_modalities": evidence["formal_modalities"],
                "frame_count": len(evidence["frames"]),
                "world_time_seconds": [
                    evidence["runtime_application"]["initial_world_time_seconds"],
                    evidence["runtime_application"]["final_world_time_seconds"],
                ],
                "evidence": str(output / "evidence.json"),
                "evidence_content_sha256": evidence["evidence_content_sha256"],
                "array_artifacts": {
                    modality: record["artifact"]
                    for modality, record in evidence["array_artifacts"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
