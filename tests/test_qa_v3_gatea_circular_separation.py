"""Gate A 的区间分离必须在圆上算,而不是拿端点相减。

claude-d3 2026-09-04 在实跑中指出:1f3ecd5 给这一处加的守卫判的是单个区间自己
hi < lo,而 [172,178] 与 [-178,-172] 各自都是正序的、谁都不 wrap,所以守卫不响,
线性相减照旧给 344 度、越过 2*THETA_HALF=60、判为已分离。圆上真实间隙是 4 度。

这是认证的假通过:两个金标叠在一起,一个完全不听音频、两边都报同一个角度的模型
可以全中,却算"通过了必要性认证"。这个文件在 audit_gatea_pair 这一层钉住它,不只在
弧模块那一层。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import design_qa_v3_scene_batch as SB  # noqa: E402

PARAMS = {"THETA_FULL": 15.0, "THETA_HALF": 30.0}


def _pair(main_interval, gate_interval):
    """两份只含 open 区间的最小 answer,其余字段取相同值以隔离被测的那一项。"""

    def answer(interval):
        return {"open": {"truth_interval_deg": list(interval),
                         "truth_value": sum(interval) / 2.0,
                         "scoring": "circular_deg_interval"}}

    return answer(main_interval), answer(gate_interval)


def _separation(main_interval, gate_interval):
    """只跑 audit_gatea_pair 里那段 open 分离判定,拿到它记录的数与结论。"""

    main_answer, gate_answer = _pair(main_interval, gate_interval)
    import qa_v3_arc as AR
    main_arc = AR.Arc.from_bounds(*main_interval)
    gate_arc = AR.Arc.from_bounds(*gate_interval)
    return (AR.circular_gap_deg(main_arc, gate_arc),
            AR.wide_credit_regions_disjoint(main_arc, gate_arc,
                                            PARAMS["THETA_HALF"]))


def test_the_reproduced_false_pass_is_now_refused():
    gap, separated = _separation((172.0, 178.0), (-178.0, -172.0))
    linear = max(0.0, -178.0 - 178.0, 172.0 - (-172.0))
    assert linear == pytest.approx(344.0)      # 旧判据看到的数
    assert gap == pytest.approx(4.0)           # 圆上真相
    assert separated is False                  # 结论从假通过翻成不通过


def test_both_intervals_hugging_the_seam_is_also_refused():
    gap, separated = _separation((-179.0, -170.0), (170.0, 179.0))
    assert gap == pytest.approx(2.0)
    assert separated is False


def test_a_genuinely_separated_pair_is_unaffected():
    """诚实对照:这一对本来就该过,换判据之后数和结论都不能变。"""
    gap, separated = _separation((0.0, 10.0), (130.0, 140.0))
    assert gap == pytest.approx(120.0)
    assert separated is True


def test_separation_just_above_and_just_below_the_threshold():
    # 阈值是 2*THETA_HALF = 60
    _, above = _separation((0.0, 5.0), (66.0, 71.0))    # 间隙 61
    _, below = _separation((0.0, 5.0), (64.0, 69.0))    # 间隙 59
    assert above is True
    assert below is False


def test_a_wrapping_interval_is_still_refused_outright():
    """单个有序对表达不了跨界楔形,这一层仍然拒绝而不是猜。"""
    import qa_v3_arc as AR
    with pytest.raises(ValueError, match="ambiguous on a circle"):
        AR.Arc.from_bounds(172.0, -178.0)


def test_the_audit_reports_the_circular_gap_not_the_endpoint_difference():
    """记录进 fact 的那个数也必须是环形的——原来那个 344 连证据本身都是错的。"""
    gap, _ = _separation((172.0, 178.0), (-178.0, -172.0))
    assert gap < 10.0
