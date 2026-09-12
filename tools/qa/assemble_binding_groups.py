#!/usr/bin/env python3
"""Assemble controlled binding groups from completed native episode variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.binding_groups import assemble_binding_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="native group member specification")
    parser.add_argument("--output", required=True, type=Path, help="fresh output directory")
    parser.add_argument("--seed", default="binding-dataset")
    parser.add_argument("--coverage-output", type=Path, default=None,
                        help="write the declared task/room coverage to this JSON file")
    args = parser.parse_args(argv)
    path = args.input.resolve()
    result = assemble_binding_dataset(
        json.loads(path.read_text(encoding="utf-8")),
        input_base=path.parent, output=args.output.resolve(), seed=args.seed,
    )
    coverage = result["source_identity_coverage"]
    summary = {key: result[key] for key in
               ("status", "group_count", "sample_count", "validation")}
    summary["public_payload_check"] = result["public_payload_check"]["status"]
    summary["task_families_present"] = coverage["task_families_present"]
    summary["task_families_absent"] = coverage["task_families_absent"]
    summary["unknown_source_identity_count"] = len(coverage["unknown_source_identity"])
    summary["task_family_by_room_family"] = coverage["task_family_by_room_family"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.coverage_output is not None:
        target = args.coverage_output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            json.dump(coverage, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
