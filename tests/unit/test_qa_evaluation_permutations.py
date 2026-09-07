from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools" / "dataset"))

from avengine.qa.evaluation_permutations import (  # noqa: E402
    EvaluationPermutationError,
    build_evaluation_permutations,
    consistency_summary,
    validate_public_document,
)
from run_qwen_content_controls import (  # noqa: E402
    build_question_text,
    load_permutation_map,
    load_completed,
    load_legacy_unpermuted_question_ids,
    model_questions,
    parse_answer,
)


def _item(question_id: str, qa_id: str, labels: str, gold: int) -> dict:
    options = [
        {"label_en": label, "label_zh": label, "value": label}
        for label in labels
    ]
    return {
        "question_id": question_id,
        "qa_id": qa_id,
        "status": "pass",
        "forms": {
            "mcq": {
                "question_en": f"Pick {question_id}.",
                "question_zh": f"Question {question_id}.",
                "options": options,
                "gold": {"correct_index": gold, "value": labels[gold]},
            }
        },
    }


def test_cyclic_rotations_keep_semantic_id_and_move_three_and_five_gold() -> None:
    result = build_evaluation_permutations(
        [{"items": [_item("q3", "QA-03", "ABC", 1), _item("q5", "QA-05", "ABCDE", 2)]}]
    )
    public = result["public"]
    private = result["private"]
    validate_public_document(public)
    assert len(public["items"]) == 8
    assert len({item["permutation_id"] for item in public["items"]}) == 8
    assert {item["question_id"] for item in public["items"] if item["question_id"] == "q3"} == {"q3"}
    assert sum(item["gold_moved"] for item in private["items"] if item["option_count"] == 3) == 2
    assert sum(item["gold_moved"] for item in private["items"] if item["option_count"] == 5) == 4
    assert result["report"]["position_distribution"]["QA-03::K3"]["gold_position_counts"] == {
        "A": 1, "B": 1, "C": 1
    }


def test_six_option_public_document_is_whitelisted() -> None:
    result = build_evaluation_permutations(
        [{"items": [_item("q6", "QA-06", "ABCDEF", 5)]}]
    )
    validate_public_document(result["public"])
    assert len(result["public"]["items"]) == 6
    assert [item["letter"] for item in result["public"]["items"][0]["options"]] == list("ABCDEF")


def test_public_and_private_gold_are_separate() -> None:
    result = build_evaluation_permutations(
        [{"items": [_item("q2", "QA-02", "AB", 0)]}]
    )
    public_text = json.dumps(result["public"], ensure_ascii=False).casefold()
    assert "gold" not in public_text
    assert "truth" not in public_text
    assert "profile" not in public_text
    assert "gold_original_index" in json.dumps(result["private"], ensure_ascii=False)
    with pytest.raises(EvaluationPermutationError):
        validate_public_document(
            {
                "schema": "avengine_qa_evaluation_permutations_public_v1",
                "items": [{"question_id": "q", "permutation_id": "p", "gold": 0}],
            }
        )


def test_no_model_outputs_means_unmeasured_consistency() -> None:
    summary = consistency_summary(None)
    assert summary["status"] == "unmeasured"
    assert summary["model_outputs_present"] is False
    assert summary["consistency_rate"] is None
    assert summary["evaluated_accuracy"] is None


def test_six_option_answer_parser_has_a_to_f_bounds() -> None:
    options = list("abcdef")
    assert parse_answer("F", options) == ("parsed_letter", "F", 5)
    assert parse_answer("G", options) == ("parse_invalid", None, None)
    assert parse_answer("answer: f", options) == ("parsed_explicit_letter", "F", 5)


def test_runner_consumes_precomputed_map_and_preserves_semantic_id(tmp_path: Path) -> None:
    source = _item("semantic_q", "QA-10", "ABC", 1)
    result = build_evaluation_permutations([{"items": [source]}])
    public_path = tmp_path / "public.json"
    public_path.write_text(json.dumps(result["public"]), encoding="utf-8")
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"placeholder")
    inputs_path = tmp_path / "inputs.json"
    options = [
        {"index": index, "letter": chr(ord("A") + index), "label_en": label}
        for index, label in enumerate("ABC")
    ]
    document = {
        "schema": "avengine_pilot_model_inputs_v1",
        "items": [
            {
                "sample_id": "semantic_q",
                "input_id": "semantic_q__full_av",
                "condition": "full_av",
                "question_en": "Pick semantic_q.",
                "options": options,
                "media_path": "clip.mp4",
            }
        ],
    }
    inputs_path.write_text(json.dumps(document), encoding="utf-8")
    questions = model_questions(
        document,
        inputs_path,
        "full_av",
        permutation_map=load_permutation_map(public_path),
    )
    assert len(questions) == 3
    assert {question["semantic_question_id"] for question in questions} == {"semantic_q"}
    assert len({question["input_id"] for question in questions}) == 3
    assert all(question["actual_model_prompt"] == build_question_text(question) for question in questions)
    assert all("gold" not in json.dumps(question).casefold() for question in questions)


def test_consistency_requires_two_distinct_rotations_and_reports_missing() -> None:
    result = build_evaluation_permutations([{"items": [_item("q3", "QA-03", "ABC", 1)]}])
    private = {row["permutation_id"]: row for row in result["private"]["items"]}
    pids = sorted(private)
    stable = [
        {
            "question_id": "q3",
            "semantic_question_id": "q3",
            "permutation_id": pid,
            "parsed_answer_index": row["gold_permuted_index"],
        }
        for pid, row in ((pids[0], private[pids[0]]), (pids[1], private[pids[1]]))
    ]
    measured = consistency_summary(stable, private)
    assert measured["status"] == "partial"
    assert measured["consistent_semantic_question_count"] == 1
    assert measured["missing_permutation_count"] == 1
    assert measured["per_question"]["q3"]["observed_permutation_count"] == 2
    all_stable = [
        {
            "question_id": "q3",
            "semantic_question_id": "q3",
            "permutation_id": pid,
            "parsed_answer_index": row["gold_permuted_index"],
        }
        for pid, row in private.items()
    ]
    complete = consistency_summary(all_stable, private)
    assert complete["status"] == "measured"
    assert complete["consistency_rate"] == 1.0

    one = consistency_summary(stable[:1], private)
    assert one["status"] == "partial"
    assert one["consistent_semantic_question_count"] == 0
    assert one["partial_semantic_question_count"] == 1

    duplicate = consistency_summary(stable[:1] * 2, private)
    assert duplicate["status"] == "partial"
    assert duplicate["duplicate_prediction_count"] == 1
    assert duplicate["observed_permutation_count"] == 1


def test_consistency_reports_inconsistent_and_accepts_runner_predicted_index() -> None:
    result = build_evaluation_permutations([{"items": [_item("q3", "QA-03", "ABC", 1)]}])
    private = {row["permutation_id"]: row for row in result["private"]["items"]}
    pids = sorted(private)
    first = private[pids[0]]
    second = private[pids[1]]
    first_original = first["permuted_to_original"][0]
    second_index = next(
        index
        for index, original in enumerate(second["permuted_to_original"])
        if original != first_original
    )
    records = [
        {
            "question_id": "q3",
            "permutation_id": pids[0],
            "parsed_answer_index": 0,
        },
        {
            "question_id": "q3",
            "permutation_id": pids[1],
            "predicted_index": second_index,
        },
    ]
    summary = consistency_summary(records, private)
    assert summary["status"] == "partial"
    assert summary["inconsistent_semantic_question_count"] == 1
    assert summary["consistent_semantic_question_count"] == 0
    assert summary["observed_permutation_count"] == 2


def test_consistency_rejects_private_question_id_mismatch_and_bool_index() -> None:
    result = build_evaluation_permutations([{"items": [_item("q3", "QA-03", "ABC", 1)]}])
    private = {row["permutation_id"]: row for row in result["private"]["items"]}
    pid = sorted(private)[0]
    with pytest.raises(EvaluationPermutationError):
        consistency_summary(
            [{"semantic_question_id": "other", "permutation_id": pid, "parsed_answer_index": 0}],
            private,
        )
    summary = consistency_summary(
        [{"semantic_question_id": "q3", "permutation_id": pid, "parsed_answer_index": True}],
        private,
    )
    assert summary["invalid_prediction_count"] == 1
    assert summary["observed_permutation_count"] == 0


def test_permutation_rejects_replaced_base_question_or_option_multiset(tmp_path: Path) -> None:
    source = _item("semantic_q", "QA-10", "ABC", 1)
    result = build_evaluation_permutations([{"items": [source]}])
    public_path = tmp_path / "public.json"
    public_path.write_text(json.dumps(result["public"]), encoding="utf-8")
    (tmp_path / "clip.mp4").write_bytes(b"placeholder")
    inputs_path = tmp_path / "inputs.json"
    base_options = [
        {"index": index, "letter": chr(ord("A") + index), "label_en": label}
        for index, label in enumerate("ABC")
    ]
    base = {
        "schema": "avengine_pilot_model_inputs_v1",
        "items": [{
            "sample_id": "semantic_q",
            "input_id": "semantic_q__full_av",
            "condition": "full_av",
            "question_en": "Pick semantic_q.",
            "options": base_options,
            "media_path": "clip.mp4",
        }],
    }
    inputs_path.write_text(json.dumps(base), encoding="utf-8")
    mapping = load_permutation_map(public_path)
    replaced = json.loads(json.dumps(base))
    replaced["items"][0]["question_en"] = "Different question."
    with pytest.raises(RuntimeError, match="replaces the base question"):
        model_questions(replaced, inputs_path, "full_av", permutation_map=mapping)
    changed_options = json.loads(json.dumps(base))
    changed_options["items"][0]["options"][0]["label_en"] = "different"
    with pytest.raises(RuntimeError, match="option multiset"):
        model_questions(changed_options, inputs_path, "full_av", permutation_map=mapping)


def test_public_whitelist_requires_nonempty_string_fields() -> None:
    result = build_evaluation_permutations([{"items": [_item("q2", "QA-02", "AB", 0)]}])
    bad = json.loads(json.dumps(result["public"]))
    bad["items"][0]["question_id"] = ""
    with pytest.raises(EvaluationPermutationError):
        validate_public_document(bad)
    bad = json.loads(json.dumps(result["public"]))
    bad["items"][0]["options"][0]["label_en"] = {"nested": "no"}
    with pytest.raises(EvaluationPermutationError):
        validate_public_document(bad)
    bad = json.loads(json.dumps(result["public"]))
    bad["items"][0]["actual_model_prompt"] = {"nested": "no"}
    with pytest.raises(EvaluationPermutationError):
        validate_public_document(bad)


def test_explicit_answer_boundary_and_legacy_resume_keys(tmp_path: Path) -> None:
    options = list("abcdef")
    assert parse_answer("answer: F", options)[1:] == ("F", 5)
    assert parse_answer("answer: finally", options) == ("parse_invalid", None, None)
    unpermuted = tmp_path / "legacy.jsonl"
    unpermuted.write_text(json.dumps({"question_id": "q"}) + "\n", encoding="utf-8")
    assert load_completed(unpermuted) == {"q"}
    assert load_legacy_unpermuted_question_ids(unpermuted) == {"q"}
    permuted = tmp_path / "permuted.jsonl"
    permuted.write_text(
        json.dumps({"question_id": "q", "permutation_id": "q__perm"}) + "\n",
        encoding="utf-8",
    )
    assert load_completed(permuted) == set()
    assert load_legacy_unpermuted_question_ids(permuted) == set()
