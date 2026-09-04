"""弧表示的四条不变量,以及那个假通过必须在这里变成真判定。

四条来自 pilot 2026-09-04,刻意不依赖最终选哪种表示。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from qa_v3_arc import (  # noqa: E402
    Arc, arc_overlap_width_deg, arcs_intersect, normalize_deg, signed_delta_deg,
    wide_credit_regions_disjoint,
)


# ── 不变量 1：楔形与它的补集必须不同 ────────────────────────────────────────
def test_a_wedge_and_its_complement_are_not_equal():
    behind = Arc(start_deg=170.0, sweep_deg=20.0)     # 身后 20 度
    front = Arc(start_deg=-170.0, sweep_deg=340.0)    # 身前 340 度
    assert behind != front
    assert behind.width_deg == pytest.approx(20.0)
    assert front.width_deg == pytest.approx(340.0)
    # 旧的有序表示把两者压成同一对数,这是它非要被换掉的原因
    assert sorted((behind.start_deg, behind.end_deg)) == \
        pytest.approx(sorted((front.start_deg, front.end_deg)))


# ── 不变量 2：扫角保号，且可以超过 180 度 ──────────────────────────────────
def test_sweep_keeps_its_sign():
    right = Arc(start_deg=0.0, sweep_deg=40.0)
    left = Arc(start_deg=0.0, sweep_deg=-40.0)
    assert right != left
    assert right.end_deg == pytest.approx(40.0)
    assert left.end_deg == pytest.approx(-40.0)


def test_a_sweep_wider_than_180_is_not_folded_back():
    wide = Arc(start_deg=-100.0, sweep_deg=200.0)
    assert wide.sweep_deg == pytest.approx(200.0)   # 不是 160
    assert wide.width_deg == pytest.approx(200.0)
    assert wide.contains(0.0)
    assert wide.contains(99.0)
    assert not wide.contains(-140.0)


# ── 不变量 3：分离按圆上集合算，不是端点相减 ────────────────────────────────
def test_the_false_pass_that_this_module_exists_to_kill():
    """pilot 在真实路径上复现的那一对:端点差 344,圆上真实间隙 4。

    旧判据 max(0, gate_lo - main_hi, main_lo - gate_hi) 给 344,越过 2*THETA_HALF=60,
    判为已分离;而两个金标各外扩 30 度之后重重叠在一起。
    """
    main = Arc.from_bounds(172.0, 178.0)
    gatea = Arc(start_deg=-178.0, sweep_deg=6.0)
    endpoint_arithmetic = max(0.0, gatea.start_deg - 178.0, 172.0 - gatea.end_deg)
    assert endpoint_arithmetic == pytest.approx(344.0)      # 旧判据看到的
    assert not wide_credit_regions_disjoint(main, gatea, 30.0)  # 真相


def test_two_genuinely_separated_golds_still_pass():
    main = Arc.from_bounds(0.0, 10.0)
    gatea = Arc.from_bounds(130.0, 140.0)
    assert wide_credit_regions_disjoint(main, gatea, 30.0)


def test_separation_across_the_wrap_is_measured_on_the_circle():
    # 圆上相隔 100 度,跨过 ±180
    main = Arc.from_bounds(140.0, 150.0)
    gatea = Arc(start_deg=-110.0, sweep_deg=10.0)
    assert wide_credit_regions_disjoint(main, gatea, 30.0)
    # 收紧到相隔 40 度,外扩 30 之后就该相交
    near = Arc(start_deg=-170.0, sweep_deg=10.0)
    assert not wide_credit_regions_disjoint(main, near, 30.0)


def test_a_full_circle_arc_intersects_everything():
    everything = Arc(start_deg=0.0, sweep_deg=360.0)
    assert arcs_intersect(everything, Arc.from_bounds(10.0, 20.0))


def test_arc_overlap_width_handles_a_seam_band():
    rear = Arc(start_deg=90.0, sweep_deg=180.0)
    seam = Arc(start_deg=162.0, sweep_deg=36.0)
    assert arc_overlap_width_deg(rear, seam) == pytest.approx(36.0)
    assert arc_overlap_width_deg(rear, Arc.from_bounds(-45.0, 45.0)) == pytest.approx(0.0)




# ── 不变量 4：编码解码往返 ──────────────────────────────────────────────────
@pytest.mark.parametrize("start,sweep", [
    (0.0, 40.0), (0.0, -40.0), (175.0, 10.0), (-175.0, -10.0),
    (-100.0, 200.0), (180.0, 5.0), (170.0, 20.0), (-170.0, 340.0),
])
def test_round_trip_keeps_start_and_signed_sweep(start, sweep):
    arc = Arc(start_deg=start, sweep_deg=sweep)
    back = Arc.from_dict(arc.as_dict())
    assert back == arc
    assert back.sweep_deg == pytest.approx(sweep)


def test_round_trip_rejects_a_foreign_schema():
    with pytest.raises(ValueError, match="expected avengine_qa_v3_arc_v1"):
        Arc.from_dict({"schema": "something_else", "start_deg": 0, "sweep_deg": 1})


# ── from_samples 取代 min/max ──────────────────────────────────────────────
def test_from_samples_recovers_a_sweep_across_the_wrap():
    """min/max 会把这串读成 358 度;真实扫角是 10 度。"""
    samples = [175.0, 178.0, -179.0, -177.0, -175.0]
    assert max(samples) - min(samples) == pytest.approx(357.0)  # 线性读法
    arc = Arc.from_samples(samples)
    assert arc.width_deg == pytest.approx(10.0)
    assert arc.start_deg == pytest.approx(175.0)
    assert arc.wraps


def test_from_samples_on_a_plain_sweep_matches_min_max():
    samples = [10.0, 12.0, 14.0, 16.0, 18.0]
    arc = Arc.from_samples(samples)
    assert (arc.start_deg, arc.end_deg) == pytest.approx((10.0, 18.0))
    assert not arc.wraps


def test_from_bounds_refuses_an_ambiguous_pair():
    with pytest.raises(ValueError, match="ambiguous on a circle"):
        Arc.from_bounds(170.0, -170.0)


# ── 折叠与转角的基础 ───────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,folded", [
    (0.0, 0.0), (180.0, 180.0), (-180.0, 180.0), (190.0, -170.0),
    (-190.0, 170.0), (540.0, 180.0), (360.0, 0.0),
])
def test_normalize_folds_into_minus180_exclusive_180_inclusive(raw, folded):
    assert normalize_deg(raw) == pytest.approx(folded)


@pytest.mark.parametrize("frm,to,delta", [
    (175.0, -175.0, 10.0), (-175.0, 175.0, -10.0),
    (0.0, 90.0, 90.0), (0.0, -90.0, -90.0), (10.0, 10.0, 0.0),
])
def test_signed_delta_takes_the_short_way(frm, to, delta):
    assert signed_delta_deg(frm, to) == pytest.approx(delta)
