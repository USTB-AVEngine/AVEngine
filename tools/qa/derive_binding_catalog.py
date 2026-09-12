#!/usr/bin/env python3
"""Derive all 25 catalog types for validated paired AV samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY/"src"))
from avengine.qa.binding_catalog import derive_binding_catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-groups", type=Path, action="append", required=True)
    parser.add_argument("--qa-sampling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items-per-type", type=int, default=1)
    parser.add_argument("--seed", default="binding-catalog")
    args = parser.parse_args(argv)
    result = derive_binding_catalog(args.binding_groups, output=args.output,
        qa_sampling=json.loads(args.qa_sampling.read_text()), items_per_type=args.items_per_type, seed=args.seed)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
