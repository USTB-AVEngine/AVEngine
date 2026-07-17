#!/usr/bin/env python3
"""Render a non-qualifying 75-frame Habitat local-TR v2 review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.m2.local_tr_review import capture_local_tr_habitat_review


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument(
        "--rebase-report",
        type=Path,
        required=True,
        help=(
            "Local-TR-preserving canonicalization report binding the visual to "
            "actor_from_skin_root."
        ),
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        required=True,
        help=(
            "Existing 75-state M2 request used only for root/action/tick scheduling; "
            "its rotation-only joint poses and hashes are ignored."
        ),
    )
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--room-request", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-id", type=int, default=200)
    parser.add_argument("--shader-type", choices=("phong", "pbr"), default="pbr")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = capture_local_tr_habitat_review(
        visual_glb=args.visual_glb,
        actions_npz=args.actions_npz,
        rebase_report=args.rebase_report,
        schedule_path=args.schedule,
        room_manifest=args.room_manifest,
        room_request=args.room_request,
        output_dir=args.output,
        runtime_root=args.runtime_root,
        semantic_id=args.semantic_id,
        shader_type=args.shader_type,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "qualification_claim": evidence["qualification_claim"],
                "formal_view_ids": evidence["formal_view_ids"],
                "output": str(args.output.resolve()),
                "evidence_content_sha256": evidence["evidence_content_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
