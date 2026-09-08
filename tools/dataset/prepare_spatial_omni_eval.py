#!/usr/bin/env python3
"""Prepare the private QA root consumed by the installed Spatial-Omni bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.model_evaluation import (  # noqa: E402
    ModelEvaluationAdapterError,
    prepare_spatial_omni_qa_root,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-inputs", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_spatial_omni_qa_root(
            model_inputs_path=args.model_inputs,
            answers_path=args.answers,
            output_root=args.output_root,
        )
    except ModelEvaluationAdapterError as exc:
        print(f"Spatial-Omni adapter failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
