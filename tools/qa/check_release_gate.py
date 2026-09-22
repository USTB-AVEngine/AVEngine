#!/usr/bin/env python3
"""Decide whether one generated bank meets the declared benchmark standard.

Exits non-zero when the bank is blocked, so a production run can gate on it without
anyone reading the receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avengine.qa.release_gate import write_release_receipt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--policy", type=Path, help="a release policy JSON; the built-in default otherwise")
    parser.add_argument("--split-views", type=Path, help="a directory of split view files, when the bank has no splits.jsonl")
    parser.add_argument("--output", type=Path, help="where to write the decision; private/release_gate.json otherwise")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else None
    result = write_release_receipt(
        args.bank, output=args.output, policy=policy, split_views=args.split_views
    )
    print(json.dumps({"status": result["status"], "blocked_rules": result["blocked_rules"]},
                     ensure_ascii=False))
    return 0 if result["status"] == "release" else 1


if __name__ == "__main__":
    raise SystemExit(main())
