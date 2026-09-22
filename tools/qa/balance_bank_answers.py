#!/usr/bin/env python3
"""Thin over-represented answers per split; see avengine.qa.answer_balance."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from avengine.qa.answer_balance import balance_bank  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-in", type=Path, required=True)
    p.add_argument("--bank-out", type=Path, required=True)
    p.add_argument("--question-manifest", type=Path, required=True,
                   help="question_manifest.jsonl of a split run over --bank-in")
    p.add_argument("--max-share", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=20260924)
    a = p.parse_args()
    splits = {}
    for line in a.question_manifest.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            splits[row["question_id"]] = row.get("split")
    summary = balance_bank(a.bank_in, a.bank_out, splits, max_share=a.max_share, seed=a.seed)
    print(json.dumps({k: summary[k] for k in ("questions_before", "questions_after", "dropped", "by_qa_after")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
