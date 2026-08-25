#!/usr/bin/env python3
"""Run body-plan-neutral automatic M2 QA using explicit variant anchors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from avengine.contracts.json_io import load_json, sha256_file
from avengine.assets.actions import read_baked_actions_npz
from avengine.assets.glb import load_glb
from avengine.assets.habitat import build_habitat_asset_mapping_from_rebase_report
from avengine.assets.qa import audit_m2_candidate
from avengine.assets.variant_package import load_variant_package_spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--rebase-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_once(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"refusing to replace QA output: {output}")
    spec = load_variant_package_spec(args.spec)
    document = load_glb(args.visual_glb)
    actions = read_baked_actions_npz(args.actions_npz)
    mapping = build_habitat_asset_mapping_from_rebase_report(
        document,
        load_json(args.rebase_report),
    )
    result = audit_m2_candidate(
        document,
        actions,
        mapping,
        semantic_joint_map=spec.semantic_joint_map,
    )
    reports = {
        "static_geometry": result.static_geometry,
        "deformation": result.deformation,
        "animation": result.animation,
    }
    if any(report.get("status") != "pass" for report in reports.values()):
        raise ValueError("automatic M2 audit did not pass; refusing to write QA")
    output.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for report_id, report in reports.items():
        path = output / f"{report_id}.json"
        _write_once(path, report)
        paths[report_id] = path
    print(
        json.dumps(
            {
                "status": "pass",
                "qualification_state": "research_candidate",
                "qualification_claim": False,
                "human_visual_review_required": True,
                "asset_id": spec.identity.asset_id,
                "semantic_joint_map": spec.semantic_joint_map,
                "reports": {
                    report_id: {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for report_id, path in paths.items()
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
