#!/usr/bin/env python3
"""Reuse one validated M2 package action set on a compatible appearance rig."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.contracts.json_io import sha256_file
from avengine.assets.action_rebind import (
    ActionRebindError,
    build_action_rebind,
    write_action_rebind,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package-manifest", type=Path, required=True)
    parser.add_argument("--appearance-report", type=Path, required=True)
    parser.add_argument("--target-visual-glb", type=Path, required=True)
    parser.add_argument("--target-rebase-report", type=Path, required=True)
    parser.add_argument("--target-rebase-deformation-report", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_action_rebind(
            source_package_manifest=args.source_package_manifest,
            appearance_report=args.appearance_report,
            target_visual_glb=args.target_visual_glb,
            target_rebase_report=args.target_rebase_report,
            target_rebase_deformation_report=(args.target_rebase_deformation_report),
        )
        artifact, report = write_action_rebind(
            result,
            output_npz=args.output_npz,
            report_output=args.report_output,
        )
    except (OSError, ValueError, ActionRebindError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "pass",
                "qualification_state": "research_candidate",
                "qualification_claim": False,
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "report": str(report),
                "report_sha256": sha256_file(report),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
