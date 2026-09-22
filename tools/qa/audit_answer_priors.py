#!/usr/bin/env python3
"""Audit a retained question bank and optional splits into a new private receipt."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avengine.qa.prior_audit import write_prior_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--split-views", type=Path, help="Directory with train/valid/test JSONL; repeated forms/presentations are deduplicated")
    parser.add_argument("--output", type=Path, required=True, help="New receipt file; existing files are never replaced")
    parser.add_argument("--thresholds", type=Path, help="Explicit JSON threshold overrides")
    parser.add_argument("--fail-on-flags", action="store_true", help="Opt-in nonzero exit on audit flags; default is report only")
    args = parser.parse_args()
    thresholds = json.loads(args.thresholds.read_text()) if args.thresholds else None
    result = write_prior_receipt(args.bank, output=args.output, split_views=args.split_views, thresholds=thresholds)
    print(json.dumps({"output": str(args.output), "status": result["status"],
        "question_count": result.get("question_count"), "split_status": result.get("split_status"),
        "under_powered_qa_ids": [qa for qa,r in result.get("by_qa", {}).items() if r["status"] == "under_powered"]}))
    return 2 if args.fail_on_flags and result["status"] != "pass" else 0

if __name__ == "__main__":
    raise SystemExit(main())
