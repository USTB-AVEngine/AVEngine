#!/usr/bin/env python3
"""Build deterministic pre-evaluation option permutations and private mappings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.evaluation_permutations import (  # noqa: E402
    MANIFEST_SCHEMA,
    EvaluationPermutationError,
    build_evaluation_permutations,
    load_json_documents,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--items-per-source", type=int)
    args = parser.parse_args(argv)
    output = args.output_root.expanduser().resolve()
    if output.exists() or output.is_symlink():
        print(f"error: refusing existing output root: {output}", file=sys.stderr)
        return 2
    try:
        documents = load_json_documents(args.questions)
        result = build_evaluation_permutations(
            documents,
            items_per_source=args.items_per_source,
        )
        output.mkdir(parents=True)
        _write_json(output / "public_permutations.json", result["public"])
        _write_json(
            output / "private_gold_mappings.json",
            {
                key: value
                for key, value in result["private"].items()
                if key != "claim_boundary"
            }
            | {"claim_boundary": result["private"]["claim_boundary"]},
        )
        _write_json(output / "permutation_report.json", result["report"])
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "ready_for_pre_evaluation",
            "source_questions": [str(path.expanduser().resolve()) for path in args.questions],
            "items_per_source": args.items_per_source,
            "counts": {
                "source_question_count": result["report"]["source_question_count"],
                "permutation_count": result["report"]["permutation_count"],
                "skipped_count": result["report"]["skipped_count"],
            },
            "outputs": {
                "public_permutations": "public_permutations.json",
                "private_gold_mappings": "private_gold_mappings.json",
                "permutation_report": "permutation_report.json",
            },
            "position_distribution": result["report"]["position_distribution"],
            "consistency_summary": result["report"]["consistency_summary"],
            "public_claim_boundary": result["public"]["claim_boundary"],
            "private_claim_boundary": result["private"]["claim_boundary"],
            "qualification_claim": False,
        }
        _write_json(output / "manifest.json", manifest)
    except (OSError, json.JSONDecodeError, EvaluationPermutationError) as exc:
        if output.exists() and not any(output.iterdir()):
            output.rmdir()
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
