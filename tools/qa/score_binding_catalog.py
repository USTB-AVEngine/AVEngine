#!/usr/bin/env python3
"""Score full-catalog predictions through the AVEngine unified scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.binding_catalog_scoring import (  # noqa: E402
    BindingCatalogScoreError,
    score_binding_catalog,
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BindingCatalogScoreError(f"cannot read JSON: {path}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        "--catalog-index",
        "--input",
        dest="catalog",
        required=True,
        type=Path,
        help="full-catalog catalog_index.json",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="JSON list of {question_id, prediction} records",
    )
    parser.add_argument("--form", choices=("mcq", "open"), default="open")
    parser.add_argument("--params", type=Path, help="optional unified scorer parameters")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    catalog_path = args.catalog.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if output.exists():
        print(f"refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2
    try:
        catalog = _read_json(catalog_path)
        predictions = _read_json(predictions_path)
        params = _read_json(args.params.expanduser().resolve()) if args.params else {}
        result = score_binding_catalog(
            catalog,
            predictions,
            form=args.form,
            params=params,
            input_base=catalog_path.parent,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, BindingCatalogScoreError) as error:
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

