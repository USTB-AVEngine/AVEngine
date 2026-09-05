#!/usr/bin/env python3
"""Prepare a truth-free request for the local Whisper speech review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.model_evaluation import (  # noqa: E402
    ModelEvaluationAdapterError,
    prepare_whisper_review_request,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-inputs", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_whisper_review_request(
            model_inputs_path=args.model_inputs,
            answers_path=args.answers,
            output_path=args.output,
            model_path=args.model,
            device=args.device,
        )
    except ModelEvaluationAdapterError as exc:
        print(f"Whisper adapter failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
