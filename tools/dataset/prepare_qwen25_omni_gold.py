#!/usr/bin/env python3
"""Build the private gold sidecar for the existing Qwen2.5-Omni pilot scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.model_evaluation import (  # noqa: E402
    ModelEvaluationAdapterError,
    prepare_qwen25_omni_pilot_gold,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-inputs", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_qwen25_omni_pilot_gold(
            model_inputs_path=args.model_inputs,
            answers_path=args.answers,
            output_path=args.output,
        )
    except ModelEvaluationAdapterError as exc:
        print(f"Qwen2.5-Omni gold preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
