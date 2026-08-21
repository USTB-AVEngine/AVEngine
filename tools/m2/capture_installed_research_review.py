#!/usr/bin/env python3
"""Run the current installed-prefix M2 Blender-room research review.

The result is research-only and is not historical M2 formal v1 evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import resolve_installed_runtime_prefix
from avengine.m2.habitat_capture import (
    capture_m2_installed_research_review,
    load_research_review_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--room-request", type=Path, required=True)
    parser.add_argument(
        "--runtime-prefix",
        type=Path,
        required=True,
        help="Non-Git installed Habitat runtime prefix",
    )
    parser.add_argument(
        "--magnum-python-site",
        type=Path,
        required=True,
        help="External Corrade/Magnum Python site for this interpreter",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        # Reject an old checkout before loading a research candidate or starting
        # native imports.  The public capture entry repeats this via prepare().
        runtime_prefix = resolve_installed_runtime_prefix(args.runtime_prefix)
        inputs = load_research_review_inputs(args.asset_manifest, args.request)
        room_inputs = load_m1_inputs(args.room_manifest, args.room_request)
        output = args.output.resolve()
        receipt = capture_m2_installed_research_review(
            inputs,
            room_inputs,
            output,
            runtime_prefix=runtime_prefix,
            magnum_python_site=args.magnum_python_site,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "research_only": receipt["research_only"],
                "qualification_claim": receipt["qualification_claim"],
                "formal_admission": receipt["formal_admission"],
                "output": str(output),
                "receipt": str(output / "research_receipt.json"),
                "frame_count": receipt["capture"]["frame_count"],
                "review_view_ids": receipt["capture"]["review_view_ids"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
