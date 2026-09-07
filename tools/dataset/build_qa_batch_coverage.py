#!/usr/bin/env python3
"""Build full-scope five-state QA coverage and structural baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.batch_coverage import (  # noqa: E402
    build_batch_coverage,
    write_batch_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repository",
        type=Path,
        default=REPOSITORY,
        help="repository root used for default inventory and room catalog paths",
    )
    args = parser.parse_args()
    result = build_batch_coverage(
        args.input_manifest,
        repository=args.repository,
    )
    paths = write_batch_coverage(result, args.output_dir)
    summary = {
        "status": result["status"],
        "schema": result["schema"],
        "denominator": result["denominator"],
        "episode_count": len(result["episode_outcomes"]),
        "global_outcome_count": len(result["global_outcomes"]),
        "output_paths": paths,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
