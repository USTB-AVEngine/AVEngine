"""查询窗口的方位扫角是圆上的量，min/max 是线性读法，跨 +-180 会读成补集。

2026-09-03 owner 把答案范围放开到整圈之前，答案一直锁在相机视锥内，所以扫角
不可能跨 +-180，这个洞是睡着的。放开之后它就能醒，而且醒的方式是安静的：
带包含判定和 Gate A 分离判定都会在错的区间上算，不报任何错。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import design_qa_v3_scene_batch as SB  # noqa: E402


def _timeline(azimuths):
    """一条只有目标 slot、逐帧方位由参数给定的最小时间线。"""

    return {"frames": [{"actor_states": [{"slot": "source1",
                                          "azimuth_deg": float(a)}]}
                       for a in azimuths]}


def _patch_recompute(monkeypatch, azimuths):
    monkeypatch.setattr(
        SB, "recompute_azimuth",
        lambda timeline, slot, frame: float(azimuths[frame]))


def test_a_sweep_that_stays_on_one_side_is_reported_linearly(monkeypatch):
    azimuths = [10.0, 12.0, 14.0, 16.0, 18.0]
    _patch_recompute(monkeypatch, azimuths)
    lo, hi, frames = SB.azimuth_sweep_engine_frame(
        _timeline(azimuths), "source1", (0.0, 0.5), 8.0)
    assert (lo, hi) == pytest.approx((10.0, 18.0))
    assert frames == (0, 4)


def test_a_sweep_across_plus_minus_180_keeps_an_ordered_arc(monkeypatch):
    # 真实扫角是 10 度（175 -> -175），不能被线性 min/max 读成补集。
    azimuths = [175.0, 178.0, -179.0, -177.0, -175.0]
    _patch_recompute(monkeypatch, azimuths)
    lo, hi, frames = SB.azimuth_sweep_engine_frame(
        _timeline(azimuths), "source1", (0.0, 0.5), 8.0)
    assert (lo, hi) == pytest.approx((175.0, -175.0))
    assert frames == (0, 4)
    arc, arc_frames = SB.azimuth_sweep_engine_arc(
        _timeline(azimuths), "source1", (0.0, 0.5), 8.0)
    assert arc.sweep_deg == pytest.approx(10.0)
    assert arc_frames == frames
