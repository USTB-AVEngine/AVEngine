#!/usr/bin/env python3
"""Promote one hash-closed M2 research candidate to a new canary package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.assets.admission import (
    ExpectedArtifact,
    promote_research_candidate,
)


def _expected_artifact(value: str) -> ExpectedArtifact:
    try:
        raw_path, digest = value.rsplit("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "artifact must be PATH=LOWERCASE_SHA256"
        ) from error
    if (
        not raw_path
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise argparse.ArgumentTypeError("artifact must be PATH=LOWERCASE_SHA256")
    return ExpectedArtifact(path=Path(raw_path), sha256=digest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--human-review-decision",
        type=Path,
        required=True,
        help="Explicit no-replace, content-authenticated human decision JSON.",
    )
    parser.add_argument("--review-request", type=Path, required=True)
    parser.add_argument("--capture-evidence", type=Path, required=True)
    parser.add_argument(
        "--world-contact-audit",
        type=_expected_artifact,
        required=True,
        help="PATH=SHA256 for the passing world-contact/root-trajectory audit.",
    )
    parser.add_argument(
        "--diagnostic-video",
        type=_expected_artifact,
        action="append",
        required=True,
        help="Repeatable PATH=SHA256 for each diagnostic video the user reviewed.",
    )
    parser.add_argument(
        "--rocketbox-root",
        type=Path,
        default=Path("/data/datasets/rocketbox/Microsoft-Rocketbox"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = promote_research_candidate(
        candidate_manifest=args.candidate_manifest,
        human_review_decision=args.human_review_decision,
        review_request=args.review_request,
        capture_evidence=args.capture_evidence,
        world_contact_audit=args.world_contact_audit,
        diagnostic_videos=args.diagnostic_video,
        rocketbox_root=args.rocketbox_root,
        output_directory=args.output,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "admission_state": "canary_qualified",
                "formal_dataset_registration_authorized": False,
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "human_visual_review": str(result.human_review_path),
                "human_visual_review_sha256": result.human_review_sha256,
                "provenance": str(result.provenance_path),
                "provenance_sha256": result.provenance_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
