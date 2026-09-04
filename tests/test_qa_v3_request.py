"""Tests for the QA-v3 request budget planner."""

from __future__ import annotations


import pytest

from tools.qa.qa_v3_request import QARequestError, normalize_answer_forms, plan_room_questions


def test_normalize_answer_forms_accepts_v10_aliases() -> None:
    assert normalize_answer_forms(["equal_bands", "open_degrees"]) == ["mcq", "open"]
    assert normalize_answer_forms("mcq") == ["mcq"]


def test_dual_forms_budget_counts_final_items_not_profile_cells() -> None:
    plan = plan_room_questions(
        ["card1", "card2", "card3"],
        {"ITEMS_PER_ROOM_DEFAULT": 300, "ANSWER_FORMS_DEFAULT": ["mcq", "open"]},
    )
    assert plan["forms_per_candidate"] == 2
    assert plan["planned_candidates"] == 150
    assert plan["planned_question_count"] == 300
    assert sum(plan["cells"].values()) == 150
    assert set(plan) == {
        "cells",
        "profile_weights",
        "answer_forms",
        "forms_per_candidate",
        "planned_candidates",
        "planned_question_count",
        "requested_budget",
        "unallocated_budget",
    }


def test_single_form_has_one_item_per_candidate() -> None:
    plan = plan_room_questions(
        ["a", "b"],
        {"ITEMS_PER_ROOM_DEFAULT": 5, "ANSWER_FORMS_DEFAULT": ["open"]},
    )
    assert plan["answer_forms"] == ["open"]
    assert plan["planned_candidates"] == 5
    assert plan["planned_question_count"] == 5
    assert plan["unallocated_budget"] == 0


def test_non_divisible_dual_budget_reports_remainder() -> None:
    plan = plan_room_questions(
        ["a", "b"],
        {"ITEMS_PER_ROOM_DEFAULT": 5, "ANSWER_FORMS_DEFAULT": ["mcq", "open"]},
    )
    assert plan["planned_candidates"] == 2
    assert plan["planned_question_count"] == 4
    assert plan["unallocated_budget"] == 1


def test_largest_remainder_weights_and_zero_profile() -> None:
    plan = plan_room_questions(
        ["a", "b", "empty"],
        {"ITEMS_PER_ROOM_DEFAULT": 8, "ANSWER_FORMS_DEFAULT": ["mcq", "open"]},
        profile_weights={"a": 1.0, "b": 3.0, "empty": 0.0},
    )
    assert plan["cells"] == {"a": 1, "b": 3, "empty": 0}
    assert plan["cells"]["empty"] == 0


def test_explicit_arguments_override_params() -> None:
    plan = plan_room_questions(
        ["a", "b"],
        {
            "ITEMS_PER_ROOM_DEFAULT": 300,
            "ANSWER_FORMS_DEFAULT": ["mcq", "open"],
        },
        question_budget=5,
        answer_forms=["open"],
        profile_weights={"a": 0.0, "b": 1.0},
    )
    assert plan["requested_budget"] == 5
    assert plan["answer_forms"] == ["open"]
    assert plan["planned_candidates"] == 5
    assert plan["cells"] == {"a": 0, "b": 5}


def test_extreme_finite_weights_allocate_without_sum_overflow() -> None:
    plan = plan_room_questions(
        ["a", "b"],
        {"ITEMS_PER_ROOM_DEFAULT": 3, "ANSWER_FORMS_DEFAULT": ["open"]},
        profile_weights={"a": 1e308, "b": 1e308},
    )
    assert plan["cells"] == {"a": 2, "b": 1}
    assert plan["planned_candidates"] == 3


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(QARequestError, match="missing required"):
        plan_room_questions(["a"], {})
    with pytest.raises(QARequestError, match="duplicate profile"):
        plan_room_questions(
            ["a", "a"], {"ITEMS_PER_ROOM_DEFAULT": 2, "ANSWER_FORMS_DEFAULT": ["mcq"]}
        )
    with pytest.raises(QARequestError, match="smaller than one candidate"):
        plan_room_questions(
            ["a"], {"ITEMS_PER_ROOM_DEFAULT": 1, "ANSWER_FORMS_DEFAULT": ["mcq", "open"]}
        )
    with pytest.raises(QARequestError, match="finite and non-negative"):
        plan_room_questions(
            ["a", "b"],
            {"ITEMS_PER_ROOM_DEFAULT": 2, "ANSWER_FORMS_DEFAULT": ["mcq"]},
            profile_weights={"a": float("nan"), "b": 1.0},
        )


def test_written_questions_follow_request_without_exposing_internal_facts(tmp_path):
    import json
    from qa_v3_request import write_requested_questions
    point = tmp_path / "p1"
    point.mkdir()
    fact = {"point_id": "p1", "scene_id": "room", "profile_id": "card8",
            "truth": {"hidden_timeline_position": [1, 2, 3]},
            "open": {"stem": "When?", "truth_value": 1.5, "scoring": "absolute_time"},
            "mcq": {"stem": "Choose when", "truth_option": 0, "options_space": ["early", "late"]}}
    for name in ("fact_record.json", "fact_record_gateA.json"):
        (point / name).write_text(json.dumps(fact))
    result = write_requested_questions(tmp_path, [point / "fact_record.json"],
                                       {"ANSWER_FORMS_DEFAULT": ["open"]})
    assert result["designed_question_count"] == 1
    assert result["counterfactual_question_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "questions.jsonl").read_text().splitlines()]
    assert [row["form"] for row in rows] == ["open"]
    assert rows[0]["answer"]["truth_value"] == 1.5
    assert "hidden_timeline_position" not in json.dumps(rows)
    assert "mcq" not in rows[0]
