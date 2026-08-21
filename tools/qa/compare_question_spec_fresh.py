#!/usr/bin/env python3
"""Fresh QuestionSpec re-evaluation versus retained bind-time records.

For every episode in the native question catalog, load the verified native
closure (facts with pixel truth, registries, sound bindings), re-evaluate the
retained bind-time QuestionSpecs with the current evaluator, and compare
status and answer against the retained question_evaluations.json and
expected_question_status.json. Writes one no-clobber report directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.qa.question_protocol import _load_native_episode
from avengine.qa.question_spec import evaluate_question_specs

REPOSITORY = Path(__file__).resolve().parents[2]

DEFAULT_EPISODES = REPOSITORY / "examples/qa/native_question_episode_catalog_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to replace output: {output}")

    catalog = json.loads(args.episodes.read_text())

    per_episode = []
    total = matches = 0
    for record in catalog["episodes"]:
        episode = _load_native_episode(REPOSITORY, record)
        binding_dir = (REPOSITORY / record["binding_manifest_path"]).parent
        specs = json.loads((binding_dir / "question_specs.json").read_text())
        retained = json.loads((binding_dir / "question_evaluations.json").read_text())
        expected_path = binding_dir / "expected_question_status.json"
        expected = (
            json.loads(expected_path.read_text()) if expected_path.is_file() else {}
        )
        retained_by_id = {
            r["spec_id"]: r
            for r in (retained if isinstance(retained, list) else retained.values())
        }
        fresh = evaluate_question_specs(
            specs,
            facts=episode["facts"],
            asset_registry=episode["asset_registry"],
            sound_registry=episode["sound_registry"],
            event_sound_bindings=episode["event_sound_bindings"],
        )
        rows = []
        for spec, evaluation in zip(specs, fresh):
            spec_id = spec["spec_id"]
            old = retained_by_id.get(spec_id)
            row = {
                "spec_id": spec_id,
                "fresh_status": evaluation.get("status"),
                "retained_status": old.get("status") if old else None,
                "expected_status": expected.get(spec_id),
                "fresh_answer": evaluation.get("answer"),
                "retained_answer": old.get("answer") if old else None,
            }
            row["status_match"] = row["fresh_status"] == row["retained_status"]
            row["answer_match"] = row["fresh_answer"] == row["retained_answer"]
            row["expected_match"] = (
                row["expected_status"] is None
                or row["fresh_status"] == row["expected_status"]
            )
            rows.append(row)
            total += 1
            if row["status_match"] and row["answer_match"] and row["expected_match"]:
                matches += 1
        per_episode.append(
            {
                "episode_key": record["episode_key"],
                "spec_count": len(rows),
                "exact_matches": sum(
                    1
                    for r in rows
                    if r["status_match"] and r["answer_match"] and r["expected_match"]
                ),
                "rows": rows,
            }
        )

    summary = {
        "schema": "avengine_questionspec_fresh_comparison_v1",
        "claim_boundary": (
            "Fresh re-evaluation of retained bind-time QuestionSpecs with the "
            "current evaluator over the verified native closure (facts include "
            "pixel truth). Research comparison only; no admission claim."
        ),
        "episode_count": len(per_episode),
        "spec_total": total,
        "spec_exact_matches": matches,
        "all_match": matches == total,
        "episodes": per_episode,
    }
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "episode_count": len(per_episode),
                "spec_total": total,
                "spec_exact_matches": matches,
                "all_match": matches == total,
                "output": str(output),
            }
        )
    )
    return 0 if matches == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
