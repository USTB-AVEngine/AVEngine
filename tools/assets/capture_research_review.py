#!/usr/bin/env python3
"""Run the 75-state single-view M2 research-review capture in Habitat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.assets.habitat_capture import (
    capture_m2_research_review,
    load_research_review_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--room-request", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = load_research_review_inputs(args.asset_manifest, args.request)
    room_inputs = load_m1_inputs(args.room_manifest, args.room_request)
    evidence = capture_m2_research_review(
        inputs,
        room_inputs,
        args.output,
        runtime_root=args.runtime_root,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "review_only": evidence["review_only"],
                "qualification_claim": evidence["qualification_claim"],
                "formal_view_ids": evidence["formal_view_ids"],
                "review_view_ids": evidence["review_view_ids"],
                "frame_count": len(evidence["frames"]),
                "world_time_seconds": [
                    evidence["runtime_application"]["initial_world_time_seconds"],
                    evidence["runtime_application"]["final_world_time_seconds"],
                ],
                "evidence": str(args.output.resolve() / "evidence.json"),
                "evidence_content_sha256": evidence["evidence_content_sha256"],
                "review_videos": {
                    modality: record["artifact"]
                    for modality, record in evidence["review_media"]["videos"].items()
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
