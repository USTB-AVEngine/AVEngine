#!/usr/bin/env python3
"""Fact-level axis-1 (route-swap) certification for mined simple questions.

For every episode hosting mined questions this builds the route-swap twin
fact table, re-answers each question's predicate on both tables and grants
an axis-1 certificate only when the answer actually flips. Twin fact tables
are retained (schema-validated) for the later twin-audio render; refusals
and not-applicable types are reported, never silently dropped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import (  # noqa: E402
    file_record,
    load_json,
    write_json,
)
from avengine.qa.certify import certify_axis1, twin_fact_table  # noqa: E402
from avengine.qa.fact_table import FACT_TABLE_SCHEMA  # noqa: E402

CERTIFICATES_SCHEMA = "avengine_qa_axis1_certificates_v1"


class Axis1CertificationError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact-tables", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPOSITORY / "schemas/avengine_qa_fact_table_v1.schema.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import jsonschema  # noqa: PLC0415

    fact_root = args.fact_tables.resolve()
    index = load_json(fact_root / "fact_table_index.json")
    if index.get("fact_table_schema") != FACT_TABLE_SCHEMA or not index.get("complete"):
        raise Axis1CertificationError(
            "fact-table index is not a complete avengine_qa_fact_table_v1 compilation"
        )
    fact_paths = {
        entry["episode_id"]: fact_root / entry["fact_table"]["path"]
        for entry in index["episodes"]
    }

    question_set = load_json(args.questions)
    questions = question_set["questions"]
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        by_episode.setdefault(question["episode_id"], []).append(question)
    unknown = sorted(set(by_episode) - set(fact_paths))
    if unknown:
        raise Axis1CertificationError(
            f"questions reference episodes without fact tables: {unknown[:3]}"
        )

    validator = jsonschema.Draft202012Validator(load_json(args.schema))
    output = args.output.resolve()
    twins_dir = output / "facts_twin"
    twins_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    status_counts: dict[str, dict[str, int]] = {}
    twin_entries: list[dict[str, Any]] = []
    certified_questions: list[dict[str, Any]] = []
    for episode_id in sorted(by_episode):
        original = load_json(fact_paths[episode_id])
        twin = twin_fact_table(original)
        errors = sorted(
            validator.iter_errors(twin), key=lambda err: list(err.absolute_path)
        )
        if errors:
            raise Axis1CertificationError(
                f"{twin['episode_id']}: twin violates the fact-table schema: "
                f"{errors[0].message}"
            )
        twin_path = twins_dir / f"{twin['episode_id']}.json"
        write_json(twin_path, twin)
        twin_entries.append(
            {
                "episode_id": episode_id,
                "twin_episode_id": twin["episode_id"],
                "twin_fact_table": file_record(twin_path, relative_to=output),
            }
        )
        for question in by_episode[episode_id]:
            record = certify_axis1(question, original, twin)
            records.append(record)
            per_type = status_counts.setdefault(question["type_id"], {})
            per_type[record["status"]] = per_type.get(record["status"], 0) + 1
            certified = dict(question)
            certified["certification"] = {
                "status": record["status"],
                "axis1": record,
            }
            certified_questions.append(certified)

    certificates = {
        "schema": CERTIFICATES_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "Fact-level route-swap flip verification; twin audio has not been "
            "rendered and no dataset admission is granted"
        ),
        "question_set": file_record(args.questions.resolve(), relative_to=args.questions.resolve().parent),
        "fact_table_index": file_record(
            fact_root / "fact_table_index.json", relative_to=fact_root
        ),
        "question_count": len(records),
        "status_counts_by_type": status_counts,
        "twins": twin_entries,
        "certificates": records,
    }
    write_json(output / "certificates.json", certificates)
    write_json(
        output / "questions_certified.json",
        {
            **{
                key: value
                for key, value in question_set.items()
                if key != "questions"
            },
            "axis1_certificates": file_record(
                output / "certificates.json", relative_to=output
            ),
            "questions": certified_questions,
        },
    )
    granted = sum(
        counts.get("granted", 0) for counts in status_counts.values()
    )
    print(
        f"QA_AXIS1_CERTIFY_OK output={output} questions={len(records)} "
        f"granted={granted} by_type={status_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
