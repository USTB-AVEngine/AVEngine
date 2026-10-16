#!/usr/bin/env python3
"""Mine simple (A-group) questions from compiled QA fact tables.

Reads a fact-table index produced by ``compile_apartment_fact_tables.py``,
mines every v1-mineable simple question type, balances the per-type answer
histograms deterministically, and writes the balanced question set plus a
stratified human-review sample sheet. Deferred question types are reported
explicitly; nothing here claims dataset admission or modality-necessity
certification (certificates are attached in a later phase).
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
from avengine.qa.miner import (  # noqa: E402
    DEFERRED_TYPES,
    MINER_SCHEMA,
    answer_histogram,
    balance_answer_histogram,
    mine_fact_table,
)
from avengine.qa.fact_table import FACT_TABLE_SCHEMA  # noqa: E402


class SimpleQuestionMiningError(RuntimeError):
    pass


def _stratified_sample(
    questions: list[Mapping[str, Any]], count: int, *, seed: str
) -> list[Mapping[str, Any]]:
    """Deterministic per-type sample that prefers distinct episodes."""

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
        seen_episodes: set[str] = set()
        for question in ordered:
            if len(picked) == per_type:
                break
            if question["episode_id"] in seen_episodes:
                continue
            seen_episodes.add(question["episode_id"])
            picked.append(question)
        for question in ordered:
            if len(picked) == per_type:
                break
            if question not in picked:
                picked.append(question)
        sample.extend(picked)
    return sample[:count] if len(sample) > count else sample


def _review_sheet(
    sample: list[Mapping[str, Any]],
    *,
    ue_bundle_dir: Path | None,
    audio_batch_dir: Path | None,
) -> str:
    lines = [
        "# Simple-question review sample",
        "",
        "Answer options are shown in their shuffled presentation order; the",
        "correct option is marked. The bundled review video predates the",
        "semantic-v2 acoustic re-render, so judge visuals from the video and",
        "audio from the linked semantic-v2 wav.",
        "",
    ]
    for index, question in enumerate(sample, start=1):
        episode_id = question["episode_id"]
        lines.append(
            f"## {index}. [{question['type_id']}] {episode_id}"
        )
        lines.append(f"- EN: {question['question_en']}")
        lines.append(f"- ZH: {question['question_zh']}")
        for option_index, option in enumerate(question["options"]):
            marker = " **<- answer**" if option_index == question["answer_index"] else ""
            lines.append(
                f"  {chr(ord('a') + option_index)}) {option['label_en']} / "
                f"{option['label_zh']}{marker}"
            )
        lines.append(f"- Evidence: `{question['evidence']}`")
        lines.append(f"- Modality note: {question['modality_note']}")
        if ue_bundle_dir is not None:
            video = ue_bundle_dir / episode_id / "ue_clean_binaural.mp4"
            if not video.is_file():
                raise SimpleQuestionMiningError(f"missing review video: {video}")
            lines.append(f"- Video (visuals authoritative): {video}")
        if audio_batch_dir is not None:
            wav = audio_batch_dir / "audio" / "binaural" / f"{episode_id}__v00.wav"
            if not wav.is_file():
                raise SimpleQuestionMiningError(f"missing semantic-v2 wav: {wav}")
            lines.append(f"- Audio (semantic-v2, authoritative): {wav}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fact-tables",
        type=Path,
        required=True,
        help="Output directory of compile_apartment_fact_tables.py",
    )
    parser.add_argument("--seed", default="qa_simple_v1_20260727")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument(
        "--ue-bundle-dir",
        type=Path,
        help="Per-episode UE review bundle root (adds video links to the sheet)",
    )
    parser.add_argument(
        "--audio-batch-dir",
        type=Path,
        help="Asset-bound binaural batch root (adds semantic-v2 wav links)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    fact_root = args.fact_tables.resolve()
    index_path = fact_root / "fact_table_index.json"
    index = load_json(index_path)
    if index.get("fact_table_schema") != FACT_TABLE_SCHEMA:
        raise SimpleQuestionMiningError(
            "index does not reference avengine_qa_fact_table_v1 documents"
        )
    if not index.get("complete"):
        raise SimpleQuestionMiningError(
            "refusing to mine from an incomplete fact-table compilation"
        )

    mined: list[dict[str, Any]] = []
    for entry in index["episodes"]:
        fact_table = load_json(fact_root / entry["fact_table"]["path"])
        mined.extend(mine_fact_table(fact_table))

    pre_histogram = answer_histogram(mined)
    balanced = balance_answer_histogram(mined, seed=args.seed)
    post_histogram = answer_histogram(balanced)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    question_set = {
        "schema": MINER_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "Mining-first simple questions over compiled fact tables; no "
            "modality-necessity certification is attached yet and no dataset "
            "admission is granted"
        ),
        "seed": args.seed,
        "fact_table_index": file_record(index_path, relative_to=fact_root),
        "episode_count": len(index["episodes"]),
        "deferred_types": dict(DEFERRED_TYPES),
        "mined_question_count": len(mined),
        "balanced_question_count": len(balanced),
        "answer_histogram_before_balancing": pre_histogram,
        "answer_histogram_after_balancing": post_histogram,
        "questions": balanced,
    }
    write_json(output / "questions.json", question_set)

    sample = _stratified_sample(balanced, args.sample, seed=args.seed)
    sheet = _review_sheet(
        sample,
        ue_bundle_dir=args.ue_bundle_dir.resolve() if args.ue_bundle_dir else None,
        audio_batch_dir=(
            args.audio_batch_dir.resolve() if args.audio_batch_dir else None
        ),
    )
    (output / "review_sample.md").write_text(sheet, encoding="utf-8")

    type_counts: dict[str, int] = {}
    for question in balanced:
        type_counts[question["type_id"]] = type_counts.get(question["type_id"], 0) + 1
    print(
        f"QA_SIMPLE_QUESTIONS_OK output={output} mined={len(mined)} "
        f"balanced={len(balanced)} sample={len(sample)} types={type_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
