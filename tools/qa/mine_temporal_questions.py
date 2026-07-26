#!/usr/bin/env python3
"""Mine temporal (B-group) and numeric questions over intermittent fact tables.

Reads a fact-table compilation produced with ``--intermittent-batch`` (whose
sound events carry declared multi-window truth), mines the temporal and
numeric question types, balances MCQ answer histograms deterministically
(numeric-banded questions pass through unbalanced with their bands recorded)
and writes the question set plus a stratified review sheet linking each
question to its gated mixture.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import (  # noqa: E402
    file_record,
    load_json,
    write_json,
)
from avengine.qa.fact_table import FACT_TABLE_SCHEMA  # noqa: E402
from avengine.qa.miner import answer_histogram, balance_answer_histogram  # noqa: E402
from avengine.qa.miner_temporal import mine_temporal_fact_table  # noqa: E402

SET_SCHEMA = "avengine_qa_temporal_question_set_v1"


class TemporalMiningError(RuntimeError):
    pass


def _stratified_sample(
    questions: list[Mapping[str, Any]], count: int, *, seed: str
) -> list[Mapping[str, Any]]:
    def digest(question: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            f"{seed}\0sample\0{question['question_id']}".encode("utf-8")
        ).hexdigest()

    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for question in questions:
        by_type.setdefault(question["type_id"], []).append(question)
    per_type = max(1, math.ceil(count / len(by_type))) if by_type else 0
    sample: list[Mapping[str, Any]] = []
    for type_id in sorted(by_type):
        ordered = sorted(by_type[type_id], key=digest)
        picked: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for question in ordered:
            if len(picked) == per_type:
                break
            if question["episode_id"] in seen:
                continue
            seen.add(question["episode_id"])
            picked.append(question)
        sample.extend(picked)
    return sample[:count] if len(sample) > count else sample


def _review_sheet(
    sample: list[Mapping[str, Any]], *, intermittent_batch: Path | None
) -> str:
    lines = [
        "# Temporal-question review sample",
        "",
        "Windows are declared truth (fade-gated dry before convolution).",
        "Listen to the gated mixture; the original review video shows the",
        "same visuals but its audio is the continuous realization.",
        "",
    ]
    for index, question in enumerate(sample, start=1):
        episode_id = question["episode_id"]
        lines.append(f"## {index}. [{question['type_id']}] {episode_id}")
        lines.append(f"- EN: {question['question_en']}")
        lines.append(f"- ZH: {question['question_zh']}")
        if question.get("format") == "numeric_banded":
            lines.append(
                f"- Answer: {question['answer_numeric']} {question['unit']} "
                f"(bands ±{question['scoring_bands'][0]}, "
                f"±{question['scoring_bands'][1]})"
            )
        else:
            for option_index, option in enumerate(question["options"]):
                marker = (
                    " **<- answer**"
                    if option_index == question["answer_index"]
                    else ""
                )
                lines.append(
                    f"  {chr(ord('a') + option_index)}) {option['label_en']} / "
                    f"{option['label_zh']}{marker}"
                )
        lines.append(f"- Evidence: `{question['evidence']}`")
        lines.append(f"- Modality note: {question['modality_note']}")
        if intermittent_batch is not None:
            wav = (
                intermittent_batch
                / "audio"
                / "binaural"
                / f"{episode_id}__int00.wav"
            )
            if not wav.is_file():
                raise TemporalMiningError(f"missing gated mixture: {wav}")
            lines.append(f"- Gated audio (authoritative): {wav}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact-tables", type=Path, required=True)
    parser.add_argument("--intermittent-batch", type=Path)
    parser.add_argument("--seed", default="qa_temporal_v1_20260727")
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    fact_root = args.fact_tables.resolve()
    index = load_json(fact_root / "fact_table_index.json")
    if index.get("fact_table_schema") != FACT_TABLE_SCHEMA:
        raise TemporalMiningError("index does not reference fact table v1 documents")
    if index.get("realization") != "intermittent_declared_windows":
        raise TemporalMiningError(
            "temporal mining requires an intermittent-realization compilation"
        )
    if not index.get("complete"):
        raise TemporalMiningError("refusing to mine an incomplete compilation")

    mined: list[dict[str, Any]] = []
    for entry in index["episodes"]:
        fact_table = load_json(fact_root / entry["fact_table"]["path"])
        mined.extend(mine_temporal_fact_table(fact_table))

    mcq = [question for question in mined if question.get("format") == "mcq"]
    numeric = [
        question for question in mined if question.get("format") == "numeric_banded"
    ]
    balanced_mcq = balance_answer_histogram(mcq, seed=args.seed)
    questions = sorted(
        balanced_mcq + numeric, key=lambda question: question["question_id"]
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    question_set = {
        "schema": SET_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "Temporal/numeric questions mined from declared intermittent "
            "windows; certification pending and no dataset admission"
        ),
        "seed": args.seed,
        "fact_table_index": file_record(
            fact_root / "fact_table_index.json", relative_to=fact_root
        ),
        "episode_count": len(index["episodes"]),
        "mined_question_count": len(mined),
        "balanced_question_count": len(questions),
        "mcq_histogram_before_balancing": answer_histogram(mcq),
        "mcq_histogram_after_balancing": answer_histogram(balanced_mcq),
        "numeric_question_count": len(numeric),
        "questions": questions,
    }
    write_json(output / "questions.json", question_set)

    sample = _stratified_sample(questions, args.sample, seed=args.seed)
    sheet = _review_sheet(
        sample,
        intermittent_batch=(
            args.intermittent_batch.resolve() if args.intermittent_batch else None
        ),
    )
    (output / "review_sample.md").write_text(sheet, encoding="utf-8")

    type_counts: dict[str, int] = {}
    for question in questions:
        type_counts[question["type_id"]] = type_counts.get(question["type_id"], 0) + 1
    print(
        f"QA_TEMPORAL_QUESTIONS_OK output={output} mined={len(mined)} "
        f"kept={len(questions)} sample={len(sample)} types={type_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
