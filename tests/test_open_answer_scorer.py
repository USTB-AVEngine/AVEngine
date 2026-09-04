"""Unit tests for the open-form answer scorer (pilot item 1.5).

覆盖:环形角距(含 ±180 绕算)、带宽三档、方向词换算与冲突、复述题干的
多数字防线、时刻带、闭集最长词条优先("没动"压制"动"这一历史陷阱)、
跨类冲突判 invalid、拒答三分、计数全对制、参数缺失拒绝、输出 no-clobber。
阳性对照:每类判分器都有"必须拒/必须零分"的坏对象用例。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from score_open_answers import (  # noqa: E402
    DEFAULT_VOCAB,
    circular_deg,
    main,
    score_angle,
    score_closed,
    score_counts,
    score_time,
    score_item,
    scorer_params,
    angle_credit_radius,
    resolve_angle_policy,
)

TF, TH = 15.0, 30.0  # 角度带占位
SF, SH = 0.3, 1.0    # 时刻带占位


# ---------- 角度 ----------

def test_circular_wraparound():
    assert circular_deg(-179.0, 179.0) == pytest.approx(2.0)
    assert circular_deg(0.0, 360.0) == pytest.approx(0.0)


def test_angle_full_half_zero():
    assert score_angle("-85°", -85.0, TF, TH)["score"] == 1.0
    assert score_angle("大约 -70 度", -85.0, TF, TH)["score"] == 1.0   # 误差 15 = 满分带边界
    assert score_angle("-60°", -85.0, TF, TH)["score"] == 0.5          # 误差 25 → 半分
    assert score_angle("40 度", -85.0, TF, TH)["score"] == 0.0


def test_angle_wrap_scoring():
    assert score_angle("179°", -175.0, TF, TH)["score"] == 1.0  # 环形误差 6°


def test_angle_direction_words():
    rec = score_angle("在左边大约 85 度", -85.0, TF, TH)
    assert rec["status"] == "scored" and rec["score"] == 1.0
    rec = score_angle("right 85 degrees", 85.0, TF, TH)
    assert rec["score"] == 1.0


def test_angle_direction_conflicts_invalid():
    assert score_angle("left -85°", -85.0, TF, TH)["status"] == "invalid"
    assert score_angle("可能在左边也可能在右边 85 度", -85.0, TF, TH)["status"] == "invalid"


def test_angle_policy_strict_keeps_half_credit_as_diagnostic_only():
    strict = score_angle(
        "20°", 0.0, TF, TH,
        certification_policy="strict_full_credit_only")
    assert strict["score"] == 0.0
    assert strict["diagnostic_two_tier_score"] == 0.5
    assert strict["certification_policy"] == "strict_full_credit_only"

    legacy = score_angle(
        "20°", 0.0, TF, TH,
        certification_policy="legacy_two_tier")
    assert legacy["score"] == 0.5
    assert "diagnostic_two_tier_score" not in legacy


def test_angle_direction_words_follow_the_declared_convention():
    # DCASE publishes left as positive. Numeric and directional answers use
    # the same sign when they state the same direction.
    numeric = score_angle(
        "85°", 85.0, TF, TH, convention="dcase_foa_left_positive")
    directional = score_angle(
        "left 85°", 85.0, TF, TH, convention="dcase_foa_left_positive")
    assert numeric["parsed"] == directional["parsed"] == 85.0
    assert numeric["score"] == directional["score"] == 1.0
    assert score_angle(
        "right 85°", -85.0, TF, TH,
        convention="dcase_foa_left_positive")["parsed"] == -85.0

    # Existing right-positive items retain their old direction-word meaning.
    old_numeric = score_angle("-85°", -85.0, TF, TH,
                              convention="right_positive_deg")
    old_directional = score_angle("left 85°", -85.0, TF, TH,
                                  convention="right_positive_deg")
    assert old_numeric["parsed"] == old_directional["parsed"] == -85.0


def test_angle_circular_wraparound_survives_strict_policy():
    result = score_angle(
        "-179°", 179.0, TF, TH,
        certification_policy="strict_full_credit_only",
        convention="dcase_foa_left_positive")
    assert result["circular_error_deg"] == pytest.approx(2.0)
    assert result["score"] == 1.0


def test_score_item_uses_question_convention_and_params_policy():
    result = score_item(
        {
            "answer_type": "angle_deg",
            "question": "Use DCASE FOA: positive values to the left.",
            "model_answer": "left 20°",
            "truth": 0.0,
        },
        {
            "THETA_FULL": TF,
            "THETA_HALF": TH,
            "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only",
        },
        DEFAULT_VOCAB,
    )
    assert result["parsed"] == 20.0
    assert result["score"] == 0.0
    assert result["diagnostic_two_tier_score"] == 0.5


def test_scorer_params_executes_declared_angle_policy_without_faking_legacy():
    base = {"THETA_FULL": TF, "THETA_HALF": TH,
            "T_FULL": SF, "T_HALF": SH}
    strict = scorer_params({
        **base, "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only"})
    assert strict["ANGLE_CERTIFICATION_POLICY"] == "strict_full_credit_only"
    assert "ANGLE_CERTIFICATION_POLICY" not in scorer_params(base)
    assert resolve_angle_policy(base) == "legacy_two_tier"
    assert resolve_angle_policy(
        base, "strict_full_credit_only") == "strict_full_credit_only"
    with pytest.raises(ValueError, match="ANGLE_CERTIFICATION_POLICY"):
        scorer_params({**base, "ANGLE_CERTIFICATION_POLICY": "future_policy"})


def test_score_item_policy_precedes_params_and_conflicts_are_explicit():
    item = {
        "answer_type": "angle_deg",
        "model_answer": "20°",
        "truth": 0.0,
        "certification_policy": "strict_full_credit_only",
    }
    params = {"THETA_FULL": TF, "THETA_HALF": TH}
    result = score_item(item, params, DEFAULT_VOCAB)
    assert result["score"] == 0.0
    assert result["diagnostic_two_tier_score"] == 0.5

    conflict = score_item(
        {**item, "certification_policy": "legacy_two_tier"},
        {**params, "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only"},
        DEFAULT_VOCAB,
    )
    assert conflict["status"] == "invalid"
    assert "policy conflict" in conflict["reason"]


def test_cli_preserves_item_strict_policy_with_old_four_key_params(tmp_path):
    items_p = _write(tmp_path / "items.json", [{
        "question_id": "strict-angle",
        "answer_type": "angle_deg",
        "model_answer": "20°",
        "truth": 0.0,
        "certification_policy": "strict_full_credit_only",
    }])
    params_p = _write(tmp_path / "params.json", {
        "THETA_FULL": TF, "THETA_HALF": TH,
        "T_FULL": SF, "T_HALF": SH,
    })
    out = tmp_path / "scores.json"
    assert main(["--items", items_p, "--params", params_p, "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert "ANGLE_CERTIFICATION_POLICY" not in doc["parameters"]
    record = doc["records"][0]
    assert record["score"] == 0.0
    assert record["diagnostic_two_tier_score"] == 0.5


@pytest.mark.parametrize(
    ("theta_full", "theta_half"),
    [(float("nan"), 30.0), (15.0, float("inf")), (-1.0, 30.0), (30.0, 15.0)],
)
def test_angle_tolerances_are_finite_and_ordered(theta_full, theta_half):
    with pytest.raises(ValueError, match="THETA_FULL"):
        score_angle("20°", 0.0, theta_full, theta_half)
    with pytest.raises(ValueError, match="THETA_FULL"):
        angle_credit_radius({
            "THETA_FULL": theta_full,
            "THETA_HALF": theta_half,
        })


def test_angle_interval_is_authoritative_across_the_wrap():
    # The interval [150, 180] crosses the representation boundary. An answer
    # at -179 is one degree from the arc, while the midpoint 165 is 16 degrees
    # away and would incorrectly lose strict credit.
    result = score_angle(
        "-179°", 165.0, TF, TH,
        certification_policy="strict_full_credit_only",
        convention="dcase_foa_left_positive",
        truth_interval_deg=[150.0, 180.0])
    assert result["truth_mode"] == "interval"
    assert result["circular_error_deg"] == pytest.approx(1.0)
    assert result["score"] == 1.0


def test_angle_interval_can_encode_a_forward_arc_across_180():
    result = score_angle(
        "179°", 0.0, TF, TH,
        certification_policy="strict_full_credit_only",
        convention="dcase_foa_left_positive",
        truth_interval_deg=[170.0, -170.0])
    assert result["truth_mode"] == "interval"
    assert result["circular_error_deg"] == pytest.approx(0.0)
    assert result["score"] == 1.0


def test_score_item_uses_truth_interval_instead_of_its_midpoint():
    result = score_item(
        {
            "answer_type": "angle_deg",
            "question": "positive values to the left",
            "model_answer": "-179°",
            "truth": 165.0,
            "truth_interval_deg": [150.0, 180.0],
            "convention": "dcase_foa_left_positive",
        },
        {
            "THETA_FULL": TF,
            "THETA_HALF": TH,
            "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only",
        },
        DEFAULT_VOCAB,
    )
    assert result["truth_mode"] == "interval"
    assert result["score"] == 1.0
    assert result["circular_error_deg"] == pytest.approx(1.0)


def test_angle_credit_radius_uses_policy_and_unknown_policy_is_an_error():
    params = {"THETA_FULL": TF, "THETA_HALF": TH,
              "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only"}
    assert angle_credit_radius(params) == TF
    assert angle_credit_radius(params, "legacy_two_tier") == TH
    with pytest.raises(ValueError, match="ANGLE_CERTIFICATION_POLICY"):
        angle_credit_radius({**params, "ANGLE_CERTIFICATION_POLICY": "bogus"})
    with pytest.raises(ValueError, match="ANGLE_CERTIFICATION_POLICY"):
        score_angle("20°", 0.0, TF, TH,
                    certification_policy="bogus")
    with pytest.raises(ValueError, match="ANGLE_CERTIFICATION_POLICY"):
        angle_credit_radius({**params, "ANGLE_CERTIFICATION_POLICY": None})


def test_angle_marked_number_beats_bare_and_restated_stem():
    # 模型复述题干"第二声"后给角度:带角度记号的唯一数字胜出
    rec = score_angle("第2声响起后,它在片尾大约 -85 度", -85.0, TF, TH)
    assert rec["status"] == "scored" and rec["parsed"] == -85.0


def test_angle_positive_controls_invalid():
    assert score_angle("它在左前方", -85.0, TF, TH)["status"] == "invalid"          # 无数字
    assert score_angle("-85° 或 -60°", -85.0, TF, TH)["status"] == "invalid"        # 撒网
    assert score_angle("坐标 (3, 4) 附近往左", -85.0, TF, TH)["status"] == "invalid"  # 多裸数字


# ---------- 时刻 ----------

def test_time_bands():
    assert score_time("2.4 秒", 2.4, SF, SH)["score"] == 1.0
    assert score_time("2.0s", 2.4, SF, SH)["score"] == 0.5   # 误差 0.4 ∈ (0.3, 1.0]
    assert score_time("4.0 秒", 2.4, SF, SH)["score"] == 0.0


def test_time_restated_stem_is_invalid():
    assert score_time("第 3 秒时?它第一次叫在 2.4 秒", 2.4, SF, SH)["status"] == "invalid"


def test_time_single_bare_number_ok():
    assert score_time("2.5", 2.4, SF, SH)["score"] == 1.0


def test_card8_strict_certification_drops_wide_partial_credit():
    rec = score_time(
        "2.0s", 2.4, SF, SH, strict_certification=True)
    assert rec["score"] == 0.0
    assert rec["diagnostic_two_tier_score"] == 0.5
    assert rec["certification_policy"] == "strict_full_credit_only"


# ---------- 闭集 ----------

def test_closed_longest_term_beats_substring_trap():
    # 历史陷阱:"没动"包含"动"——最长词条优先必须判 still
    rec = score_closed("它没动", "still", DEFAULT_VOCAB["motion"], refusal_allowed=False)
    assert rec["status"] == "scored" and rec["parsed"] == "still" and rec["score"] == 1.0
    rec = score_closed("it is not moving", "still", DEFAULT_VOCAB["motion"], refusal_allowed=False)
    assert rec["parsed"] == "still"


def test_closed_plain_hit_and_wrong_answer():
    rec = score_closed("正在走动", "still", DEFAULT_VOCAB["motion"], refusal_allowed=False)
    assert rec["status"] == "scored" and rec["score"] == 0.0
    rec = score_closed("黑白花的那只", "black_white", DEFAULT_VOCAB["coat"], refusal_allowed=False)
    assert rec["score"] == 1.0


def test_closed_cross_class_conflict_invalid():
    rec = score_closed("先走动后来静止", "still", DEFAULT_VOCAB["motion"], refusal_allowed=False)
    assert rec["status"] == "invalid"


def test_closed_no_hit_invalid():
    rec = score_closed("看不清", "still", DEFAULT_VOCAB["motion"], refusal_allowed=False)
    assert rec["status"] == "invalid"


def test_occlusion_four_state_and_offscreen():
    rec = score_closed("它已出画,不在画面里了", "out_of_view",
                       DEFAULT_VOCAB["occlusion4"], refusal_allowed=True)
    assert rec["status"] == "scored" and rec["score"] == 1.0
    rec = score_closed("被完全挡住了", "fully_occluded",
                       DEFAULT_VOCAB["occlusion4"], refusal_allowed=True)
    assert rec["score"] == 1.0


def test_refusal_three_way_abstained_scores_zero():
    rec = score_closed("无法判断", "offscreen_source",
                       DEFAULT_VOCAB["presence"], refusal_allowed=True)
    assert rec["status"] == "abstained" and rec["score"] == 0.0


def test_refusal_correct_assertion_scores():
    rec = score_closed("声音来自画面外,都不是", "offscreen_source",
                       DEFAULT_VOCAB["presence"], refusal_allowed=True)
    assert rec["status"] == "scored" and rec["score"] == 1.0


# ---------- 计数 ----------

def test_count_pair_all_or_nothing():
    assert score_counts("2 只;3 声", [2, 3])["score"] == 1.0
    assert score_counts("2 只;4 声", [2, 3])["score"] == 0.0
    assert score_counts("大概 2 只吧,叫了 3 声,也可能 4 声", [2, 3])["status"] == "invalid"
    assert score_counts("好几只", [2])["status"] == "invalid"


# ---------- 主流程 ----------

def _write(p: Path, obj) -> str:
    p.write_text(json.dumps(obj, ensure_ascii=False))
    return str(p)


def test_main_executes_strict_angle_policy(tmp_path):
    items_p = _write(tmp_path / "items.json", [{
        "question_id": "strict-angle",
        "answer_type": "angle_deg",
        "question": "DCASE FOA: positive values to the left.",
        "model_answer": "20°",
        "truth": 0.0,
    }])
    params_p = _write(tmp_path / "params.json", {
        "THETA_FULL": TF,
        "THETA_HALF": TH,
        "T_FULL": SF,
        "T_HALF": SH,
        "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only",
    })
    out = tmp_path / "scores.json"
    assert main(["--items", items_p, "--params", params_p, "--out", str(out)]) == 0
    record = json.loads(out.read_text())["records"][0]
    assert record["angle_convention"] == "left_positive"
    assert record["score"] == 0.0
    assert record["diagnostic_two_tier_score"] == 0.5


def test_main_end_to_end_and_no_clobber(tmp_path):
    items = [
        {"question_id": "q1", "answer_type": "angle_deg", "model_answer": "-85°", "truth": -85},
        {"question_id": "q2", "answer_type": "closed_set", "vocab_key": "motion",
         "model_answer": "没动", "truth": "still"},
        {"question_id": "q3", "answer_type": "closed_set", "vocab_key": "presence",
         "model_answer": "无法判断", "truth": "offscreen_source", "refusal_truth": True},
        {"question_id": "q4", "answer_type": "time_s", "model_answer": "怎么会知道呢", "truth": 2.4},
        {"question_id": "q5", "answer_type": "time_s", "model_answer": "2.0s",
         "truth": 2.4, "certification_policy": "strict_full_credit_only"},
    ]
    params = {"THETA_FULL": TF, "THETA_HALF": TH, "T_FULL": SF, "T_HALF": SH,
              "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only",
              "BANDS_CARD8": [0.35, 1.1, 1.85, 2.6]}
    items_p = _write(tmp_path / "items.json", items)
    params_p = _write(tmp_path / "params.json", params)
    out = tmp_path / "scores.json"
    assert main(["--items", items_p, "--params", params_p, "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["counts"] == {"total": 5, "scored": 3, "invalid": 1, "abstained": 1}
    assert doc["mean_score_over_scored"] == 0.6667
    assert doc["mean_score_over_all"] == 0.4
    q5 = next(record for record in doc["records"]
              if record["question_id"] == "q5")
    assert q5["score"] == 0.0
    assert q5["diagnostic_two_tier_score"] == 0.5
    assert doc["parameters"] == {
        "THETA_FULL": TF, "THETA_HALF": TH, "T_FULL": SF, "T_HALF": SH,
        "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only"}
    # no-clobber
    assert main(["--items", items_p, "--params", params_p, "--out", str(out)]) == 2


def test_main_rejects_missing_band_params(tmp_path):
    items_p = _write(tmp_path / "i.json", [])
    params_p = _write(tmp_path / "p.json", {"THETA_FULL": 15})
    assert main(["--items", items_p, "--params", params_p,
                 "--out", str(tmp_path / "o.json")]) == 2
