#!/usr/bin/env python3
"""Assemble a generic M2 animal research package from real QA evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from avengine.contracts.json_io import sha256_file  # noqa: E402
from avengine.assets.contracts import validate_animal_asset_package  # noqa: E402
from avengine.assets.variant_package import (  # noqa: E402
    VariantPackageEvidence,
    assemble_variant_package,
    load_variant_package_spec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Explicit taxonomy, identity, appearance and semantic-anchor spec",
    )
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--rebase-report", type=Path, required=True)
    parser.add_argument("--rebase-deformation-report", type=Path, required=True)
    parser.add_argument("--action-report", type=Path, required=True)
    parser.add_argument(
        "--static-qa",
        type=Path,
        required=True,
        help="Passing static_geometry.json from the generic M2 candidate audit",
    )
    parser.add_argument(
        "--deformation-qa",
        type=Path,
        required=True,
        help="Passing deformation.json from the generic M2 candidate audit",
    )
    parser.add_argument(
        "--animation-qa",
        type=Path,
        required=True,
        help="Passing animation.json from the generic M2 candidate audit",
    )
    parser.add_argument("--habitat-static-probe", type=Path, required=True)
    parser.add_argument("--habitat-animation-review", type=Path, required=True)
    parser.add_argument("--contact-phases", type=Path, required=True)
    parser.add_argument(
        "--appearance-lineage",
        type=Path,
        required=True,
        help=(
            "Passing Beagle L9 or cross-species diagnostic appearance lineage "
            "whose authenticated upstream source is the exact --source-manifest "
            "and whose visual chain closes through rebase to --visual-glb"
        ),
    )
    parser.add_argument(
        "--material-normalization-report",
        type=Path,
        required=True,
        help=(
            "Passing material-normalization v2 report whose output is the exact "
            "--visual-glb bytes and whose policy forces opaque matte materials"
        ),
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--license-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = load_variant_package_spec(args.spec)
    evidence = VariantPackageEvidence(
        visual_glb=args.visual_glb,
        rebase_report=args.rebase_report,
        rebase_deformation_report=args.rebase_deformation_report,
        action_report=args.action_report,
        static_qa=args.static_qa,
        deformation_qa=args.deformation_qa,
        animation_qa=args.animation_qa,
        habitat_static_probe=args.habitat_static_probe,
        habitat_animation_review=args.habitat_animation_review,
        baked_actions=args.actions_npz,
        contacts=args.contact_phases,
        appearance_lineage=args.appearance_lineage,
        material_normalization_report=args.material_normalization_report,
        source_manifest=args.source_manifest,
        license_snapshot=args.license_snapshot,
    )
    manifest = assemble_variant_package(
        spec=spec,
        evidence=evidence,
        output_directory=args.output,
    )
    errors = validate_animal_asset_package(
        json.loads(manifest.read_text(encoding="utf-8")),
        manifest_path=manifest,
    )
    if errors:
        raise RuntimeError("assembled package failed readback: " + "; ".join(errors))
    value = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "pass",
                "admission_state": value["admission_state"],
                "qualification": value["qualification"],
                "asset_id": value["asset_id"],
                "body_plan_id": value["body_plan_id"],
                "morphotype_id": value["morphotype_id"],
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "spec": str(spec.path),
                "spec_sha256": spec.sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
