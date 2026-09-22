#!/usr/bin/env python3
"""Append QA-26 to QA-28 from semantic builder runs to a copy of a question bank."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from avengine.qa.semantic_bank_merge import merge_semantic_into_bank  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-in", type=Path, required=True)
    p.add_argument("--bank-out", type=Path, required=True)
    p.add_argument("--single", type=Path, nargs="*", default=())
    p.add_argument("--paired", type=Path, nargs="*", default=())
    a = p.parse_args()
    summary = merge_semantic_into_bank(a.bank_in, a.bank_out, single_runs=a.single, paired_runs=a.paired)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("public_export_file_check",)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
