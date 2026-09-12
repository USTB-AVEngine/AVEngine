#!/usr/bin/env python3
"""Score private AVEngine instance-binding groups against model answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.binding_group_scoring import (  # noqa: E402
    BindingGroupScoreError,
    score_binding_groups,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        "--input",
        dest="groups",
        required=True,
        type=Path,
        help="private avengine_binding_groups_v1 JSON",
    )
    parser.add_argument(
        "--answers",
        required=True,
        type=Path,
        help="JSON object mapping sample_id to model answer",
    )
    parser.add_argument("--form", choices=("mcq", "open"), default="open")
    parser.add_argument("--params", type=Path, help="optional explicit scorer parameters")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.out.expanduser().resolve()
    if output.exists():
        print(f"refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2
    try:
        document = _read(args.groups.expanduser().resolve())
        answers = _read(args.answers.expanduser().resolve())
        params = _read(args.params.expanduser().resolve()) if args.params else {}
        result = score_binding_groups(
            document,
            answers,
            form=args.form,
            params=params,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, BindingGroupScoreError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "form": result["form"],
                "counts": result["counts"],
                "out": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
