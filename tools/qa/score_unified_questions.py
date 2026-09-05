#!/usr/bin/env python3
"""Score model answers for a generated unified QA question set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.unified_scoring import (  # noqa: E402
    UnifiedScoreError,
    score_unified_question_set,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--form", choices=("mcq", "open"), default="open")
    parser.add_argument("--params", type=Path, help="optional explicit scorer parameters")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"refusing to overwrite existing output: {args.out}", file=sys.stderr)
        return 2
    try:
        question_set = _read(args.questions.resolve())
        answers = _read(args.answers.resolve())
        params = _read(args.params.resolve()) if args.params else {}
        result = score_unified_question_set(
            question_set,
            answers,
            form=args.form,
            params=params,
        )
        args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.out.resolve().write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, UnifiedScoreError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "form": result["form"],
                "counts": result["counts"],
                "out": str(args.out.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
