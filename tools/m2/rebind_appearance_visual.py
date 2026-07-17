#!/usr/bin/env python3
"""Preserve a source M2 package actor frame on one appearance realization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.contracts.json_io import sha256_file
from avengine.m2.appearance_visual import (
    AppearanceVisualError,
    build_canonical_appearance_visual,
    write_canonical_appearance_visual,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package-manifest", type=Path, required=True)
    parser.add_argument("--appearance-report", type=Path, required=True)
    parser.add_argument("--normalized-visual-glb", type=Path, required=True)
    parser.add_argument("--normalized-rebase-report", type=Path, required=True)
    parser.add_argument("--visual-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_canonical_appearance_visual(
            source_package_manifest=args.source_package_manifest,
            appearance_report=args.appearance_report,
            normalized_visual_glb=args.normalized_visual_glb,
            normalized_rebase_report=args.normalized_rebase_report,
        )
        visual, report = write_canonical_appearance_visual(
            result,
            visual_output=args.visual_output,
            report_output=args.report_output,
        )
    except (OSError, ValueError, AppearanceVisualError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "pass",
                "qualification_state": "research_candidate",
                "qualification_claim": False,
                "visual": str(visual),
                "visual_sha256": sha256_file(visual),
                "rebase_report": str(report),
                "rebase_report_sha256": sha256_file(report),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
