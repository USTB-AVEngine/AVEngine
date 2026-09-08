from __future__ import annotations

from avengine.qa.unified_scoring import (
    circular_distance_deg,
    score_closed,
    score_counts,
    score_angle,
    score_unified_item,
    score_time_range,
)


def _item(answer_type: str, truth, *, classes=None, options=None, correct=0, **extra):
    open_form = {"answer_type": answer_type, "truth": truth, **extra}
    if classes is not None:
        open_form["classes"] = classes
    return {
        "status": "pass",
        "qa_id": "QA-TEST",
        "question_id": "qa_test__0",
        "forms": {
            "open": open_form,
            "mcq": {
                "options": options
                or [{"value": "yes", "label_en": "yes", "label_zh": "是"}],
                "gold": {"correct_index": correct, "value": "yes"},
            },
        },
    }


def test_closed_set_prefers_longest_negated_term_and_rejects_conflict() -> None:
    classes = {"moving": ["moving", "动"], "still": ["still", "不动", "静止"]}
    assert score_closed("不动", "still", classes)["score"] == 1.0
    assert score_closed("moving and still", "moving", classes)["status"] == "invalid"


def test_numeric_and_count_rules_are_strict() -> None:
    assert circular_distance_deg(179, -179) == 2.0
    assert score_counts("2人，3人", [2, 3])["score"] == 1.0
    assert score_counts("大约 2 人，也可能 3 人", [2])["status"] == "invalid"


def test_unified_item_scores_mcq_and_open_forms() -> None:
    item = _item(
        "closed_set",
        "left",
        classes={"left": ["left", "左侧"], "right": ["right", "右侧"]},
        options=[
            {"value": "left", "label_en": "left", "label_zh": "左侧"},
            {"value": "right", "label_en": "right", "label_zh": "右侧"},
        ],
        correct=0,
    )
    assert score_unified_item(item, "left", form="open")["score"] == 1.0
    assert score_unified_item(item, "A", form="mcq")["score"] == 1.0


def test_refusal_is_zero_and_separate_from_invalid() -> None:
    result = score_closed(
        "无法判断",
        "none",
        {"none": ["都不是"]},
        refusal_allowed=True,
    )
    assert result["status"] == "abstained"
    assert result["score"] == 0.0


def test_transcript_scorer_rejects_multi_statement_sweep() -> None:
    item = _item(
        "transcript_wer",
        "hello world",
        classes={
            "hello world": ["hello world"],
            "goodbye": ["goodbye"],
        },
        options=[
            {"value": "hello world", "label_en": "hello world", "label_zh": "hello world"},
            {"value": "goodbye", "label_en": "goodbye", "label_zh": "goodbye"},
        ],
        correct=0,
        reject_multiple_statements=True,
    )
    result = score_unified_item(item, "hello world; goodbye", form="open")
    assert result["status"] == "invalid"


def test_negative_angle_with_matching_direction_word_is_accepted() -> None:
    result = score_angle(
        "-30 degrees, on the left",
        -30.0,
        full_tolerance_deg=1.0,
        half_tolerance_deg=2.0,
        convention="right_positive",
    )
    assert result["status"] == "scored"
    assert result["parsed"] == -30.0
    assert result["score"] == 1.0

    conflict = score_angle(
        "-30 degrees, on the right",
        -30.0,
        full_tolerance_deg=1.0,
        half_tolerance_deg=2.0,
        convention="right_positive",
    )
    assert conflict["status"] == "invalid"


def test_negated_chinese_direction_alias_is_not_positive_farther() -> None:
    classes = {
        "nearer": ["nearer", "更近", "近"],
        "farther": ["farther", "更远", "远"],
    }
    result = score_closed("并不远", "farther", classes)
    assert result["status"] == "invalid"
    assert result["score"] == 0.0



def test_time_range_scoring_accepts_natural_range_separators() -> None:
    truth = [4.0, 8.0]
    for answer in ("4-8", "4～8", "4到8", "4–8 seconds", "4 to 8 seconds"):
        result = score_time_range(answer, truth)
        assert result["status"] == "scored"
        assert result["score"] == 1.0
    assert score_time_range("4-8 and 9", truth)["status"] == "invalid"


def test_time_range_scoring_accepts_declared_labels_and_rejects_other_intervals() -> None:
    form = {
        "answer_type": "time_range_s",
        "truth": [0.0, 2.0],
        "classes": {
            "band_0": ["first quarter of the clip", "片段前四分之一"],
            "band_1": ["second quarter of the clip", "片段第二个四分之一"],
        },
        "time_ranges_s": [[0.0, 2.0], [2.0, 4.0]],
        "time_range_index": 0,
    }
    assert score_time_range(
        "first quarter of the clip", form["truth"], form=form
    )["score"] == 1.0
    assert score_time_range("[0, 2) seconds", form["truth"], form=form)["score"] == 1.0
    assert score_time_range(
        "second quarter of the clip", form["truth"], form=form
    )["score"] == 0.0
    assert score_time_range("at 1 second", form["truth"], form=form)["status"] == "invalid"
