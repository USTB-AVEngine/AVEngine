from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from avengine.qa.binding_group_scoring import (
    BindingGroupScoreError,
    score_binding_groups,
)


def _mcq_item(
    question_id: str,
    gold: str,
    options: tuple[str, ...],
) -> dict:
    option_rows = [
        {"value": value, "label_en": value, "label_zh": value}
        for value in options
    ]
    return {
        "schema": "avengine_qa_unified_question_v1",
        "status": "pass",
        "qa_id": "QA-20",
        "question_id": question_id,
        "forms": {
            "mcq": {
                "question_en": "Which answer?",
                "question_zh": "哪个答案？",
                "answer_type": "choice",
                "options": option_rows,
                "gold": {
                    "correct_index": options.index(gold),
                    "value": gold,
                },
            }
        },
    }


def _open_item(
    question_id: str,
    truth: str,
    *,
    include_mcq: bool = False,
    angle: bool = False,
) -> dict:
    if angle:
        open_form = {
            "question_en": "What angle?",
            "question_zh": "角度是多少？",
            "answer_type": "angle_deg",
            "truth": truth,
        }
    else:
        open_form = {
            "question_en": "Is it yes?",
            "question_zh": "是否为是？",
            "answer_type": "closed_set",
            "truth": truth,
            "classes": {"yes": ["yes", "是"], "no": ["no", "否"]},
        }
    forms = {"open": open_form}
    if include_mcq:
        forms["mcq"] = {
            "question_en": "Which answer?",
            "question_zh": "哪个答案？",
            "answer_type": "choice",
            "options": [
                {"value": "yes", "label_en": "yes", "label_zh": "是"},
                {"value": "no", "label_en": "no", "label_zh": "否"},
            ],
            "gold": {"correct_index": 0 if truth == "yes" else 1, "value": truth},
        }
    return {
        "schema": "avengine_qa_unified_question_v1",
        "status": "pass",
        "qa_id": "QA-20",
        "question_id": question_id,
        "forms": forms,
    }


def _member(sample_id: str, question: dict) -> dict:
    return {
        "sample_id": sample_id,
        "question": question,
        "media": {"video_path": f"video/{sample_id}.mp4", "audio_path": f"audio/{sample_id}.wav"},
    }


def _group(group_id: str, members: list[dict], comparisons: list[dict]) -> dict:
    return {
        "group_id": group_id,
        "task_family": "visible_binding",
        "room_family": "Apartment",
        "room_id": "apartment_0000",
        "split": "test",
        "members": members,
        "comparisons": comparisons,
    }


def _document(groups: list[dict]) -> dict:
    return {
        "schema": "avengine_binding_groups_v1",
        "status": "research_candidate",
        "groups": groups,
    }


def test_mcq_relations_use_semantic_option_values_and_are_separate_from_all_correct():
    groups = [
        _group(
            "g1",
            [
                _member("s1", _mcq_item("q1", "red", ("red", "blue"))),
                _member("s2", _mcq_item("q2", "blue", ("blue", "red"))),
                _member("s3", _mcq_item("q3", "green", ("green", "red"))),
                _member("s4", _mcq_item("q4", "green", ("red", "green"))),
            ],
            [
                {
                    "members": ["s1", "s2"],
                    "shared_modality": "audio",
                    "answer_relation": "different",
                    "kind": "necessity",
                },
                {
                    "members": ["s3", "s4"],
                    "shared_modality": "video",
                    "answer_relation": "same",
                    "kind": "invariance",
                },
            ],
        )
    ]
    # Both answers in the necessity pair are wrong, but B means blue in the
    # first form and red in the second form. The semantic outputs differ as
    # expected. The invariance pair answers are both green via different
    # option positions.
    result = score_binding_groups(
        _document(groups),
        {"s1": "B", "s2": "B", "s3": "A", "s4": "B"},
        form="mcq",
    )
    assert result["counts"]["members_total"] == 4
    assert result["counts"]["members_correct"] == 2
    assert result["group_all_correct"]["all_correct"] == 0
    assert result["relation_metrics"]["correct"] == 2
    assert result["relation_metrics"]["by_kind"]["necessity"]["correct"] == 1
    assert result["relation_metrics"]["by_kind"]["invariance"]["correct"] == 1
    assert result["gold_selfcheck"]["status"] == "pass"
    assert "flip-rate" not in result["claim_boundary"]


def test_missing_and_invalid_answers_count_as_failures_unavailable_form_is_separate():
    g1 = _group(
        "available",
        [
            _member("a", _open_item("qa", "yes")),
            _member("b", _open_item("qb", "yes")),
        ],
        [
            {
                "members": ["a", "b"],
                "shared_modality": None,
                "answer_relation": "same",
                "kind": "invariance",
            }
        ],
    )
    g2 = _group(
        "mixed",
        [
            _member("c", _open_item("qc", "yes")),
            _member("d", _open_item("qd", "yes", include_mcq=True)),
        ],
        [
            {
                "members": ["c", "d"],
                "shared_modality": "video",
                "answer_relation": "same",
                "kind": "invariance",
            }
        ],
    )
    # Remove the requested open form only from d; a missing answer for c still
    # belongs to the open denominator.
    g2["members"][1]["question"]["forms"].pop("open")
    result = score_binding_groups(
        _document([g1, g2]),
        {"a": "yes", "b": "yes and no"},
        form="open",
    )
    assert result["counts"]["members_total"] == 4
    assert result["counts"]["members_form_available"] == 3
    assert result["counts"]["members_unavailable_form"] == 1
    assert result["counts"]["members_invalid"] == 2
    assert result["counts"]["members_missing"] == 1
    assert result["item_metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert result["counts"]["groups_form_available"] == 1
    assert result["counts"]["groups_unavailable_form"] == 1
    assert result["group_all_correct"]["accuracy"] == 0.0
    assert result["counts"]["comparisons_form_available"] == 1
    assert result["counts"]["comparisons_invalid"] == 1
    assert result["counts"]["comparisons_unavailable_form"] == 1
    assert result["relation_metrics"]["agreement_accuracy"] == 0.0
    assert result["groups"][0]["comparisons"][0]["status"] == "invalid"
    assert result["groups"][1]["comparisons"][0]["status"] == "unavailable_form"


def test_gold_selfcheck_flags_declared_relation_mismatch_without_changing_model_metric():
    group = _group(
        "mismatch",
        [
            _member("s1", _open_item("q1", "yes")),
            _member("s2", _open_item("q2", "no")),
        ],
        [
            {
                "members": ["s1", "s2"],
                "shared_modality": None,
                "answer_relation": "same",
                "kind": "necessity",
            }
        ],
    )
    result = score_binding_groups(
        _document([group]),
        {"s1": "yes", "s2": "no"},
        form="open",
    )
    assert result["relation_metrics"]["scored"] == 1
    assert result["relation_metrics"]["correct"] == 0
    assert result["groups"][0]["comparisons"][0]["gold_selfcheck"]["gold_relation"] == "different"
    assert result["gold_selfcheck"]["status"] == "mismatch"
    assert result["gold_selfcheck"]["mode"] == "software_consistency_only"


def test_explicit_angle_member_is_scored_once_without_catalog_followup_expansion():
    question = _open_item("angle", "0", angle=True)
    question["question_kind"] = "angle_followup"
    question["parent_question_id"] = "parent"
    result = score_binding_groups(
        _document([_group("angle", [_member("s1", question)], [])]),
        {"s1": "0 degrees"},
        form="open",
    )
    assert result["counts"]["members_total"] == 1
    assert result["counts"]["members_scored"] == 1
    assert result["counts"]["comparisons_total"] == 0
    assert result["groups"][0]["all_correct"] is True


def test_validation_rejects_unknown_answers_and_duplicate_comparisons():
    member_a = _member("a", _open_item("qa", "yes"))
    member_b = _member("b", _open_item("qb", "yes"))
    comparison = {
        "members": ["a", "b"],
        "shared_modality": None,
        "answer_relation": "same",
        "kind": "invariance",
    }
    group = _group("g", [member_a, member_b], [comparison, deepcopy(comparison)])
    with pytest.raises(BindingGroupScoreError, match="repeats comparison"):
        score_binding_groups(_document([group]), {"a": "yes", "b": "yes"})

    group["comparisons"] = [comparison]
    with pytest.raises(BindingGroupScoreError, match="unknown sample_ids"):
        score_binding_groups(
            _document([group]), {"a": "yes", "b": "yes", "unknown": "yes"}
        )


def test_cli_writes_score_and_refuses_to_clobber(tmp_path: Path):
    group = _group(
        "cli",
        [_member("s1", _open_item("q1", "yes"))],
        [],
    )
    groups_path = tmp_path / "groups.json"
    answers_path = tmp_path / "answers.json"
    output_path = tmp_path / "score.json"
    groups_path.write_text(json.dumps(_document([group])), encoding="utf-8")
    answers_path.write_text(json.dumps({"s1": "yes"}), encoding="utf-8")
    repository = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(repository / "tools/qa/score_binding_groups.py"),
        "--groups",
        str(groups_path),
        "--answers",
        str(answers_path),
        "--form",
        "open",
        "--out",
        str(output_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["counts"]["members_correct"] == 1
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["status"]
        == "research_candidate"
    )
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 2


def test_angle_group_uses_declared_tolerance_and_reports_signed_change():
    left = _open_item("q1", 179, angle=True)
    right = _open_item("q2", -149, angle=True)
    for item in (left, right):
        item["forms"]["open"].update(scoring_mode="continuous", theta_full_deg=1, theta_half_deg=3)
    group = _group("angle_change", [_member("a", left), _member("b", right)], [{
        "members": ["a", "b"], "shared_modality": "audio",
        "answer_relation": "different", "kind": "necessity",
    }])
    group["angle_tolerance_deg"] = 5
    result = score_binding_groups(_document([group]), {"a": -179, "b": -147})
    assert result["item_metrics"]["correct"] == 2
    assert result["group_all_correct"]["accuracy"] == 1.0
    assert result["angle_group_accuracy_at_deg"]["1"]["accuracy"] == 0.0
    assert result["angle_group_accuracy_at_deg"]["3"]["accuracy"] == 1.0
    relation = result["groups"][0]["comparisons"][0]
    assert relation["observed_relation"] == "circular_change"
    assert relation["relation_correct"] is True
    # Constant and reversed changes cannot earn agreement for a 32-degree change.
    constant = score_binding_groups(_document([group]), {"a": 0, "b": 0})
    reversed_change = score_binding_groups(_document([group]), {"a": -149, "b": 179})
    assert constant["relation_metrics"]["correct"] == 0
    assert reversed_change["relation_metrics"]["correct"] == 0
