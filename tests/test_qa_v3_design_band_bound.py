"""设计点的方位上界成了参数,而默认值必须跟以前逐位相同。

owner 2026-09-04 的约束是不许影响现有求解器和分配器的行为。这个文件钉的就是那件事:
不传 bound_deg 时,_design_band 的输出跟加参数之前一模一样;传 180 时画外的方位才可用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import scene_sampler as SS  # noqa: E402

M = SS.DESIGN_EDGE_MARGIN_DEG


@pytest.mark.parametrize("band,half_fov,expected", [
    (None, 47.5, (-47.5 + M, 47.5 - M)),
    ((17.5, 52.5), 47.5, (17.5 + M, 47.5 - M)),      # 老行为:先被视锥夹掉外缘,再两端各收边距
    ((-52.5, -17.5), 47.5, (-47.5 + M, -17.5 - M)),
    ((-17.5, 17.5), 47.5, (-17.5 + M, 17.5 - M)),
    (None, 55.0, (-55.0 + M, 55.0 - M)),             # 换镜头
])
def test_the_default_bound_is_the_field_of_view_exactly_as_before(
        band, half_fov, expected):
    assert SS._design_band(band, half_fov) == pytest.approx(expected)


def test_a_narrow_band_is_returned_untrimmed():
    """带宽不足两倍边距时原样返回,这条老行为也不许动。"""
    narrow = (10.0, 10.0 + M)
    assert SS._design_band(narrow, 47.5) == pytest.approx(narrow)


@pytest.mark.parametrize("band,expected", [
    ((60.0, 120.0), (60.0 + M, 120.0 - M)),          # 侧方画外
    ((132.5, 180.0), (132.5 + M, 180.0 - M)),        # 相机背后
    ((-180.0, -132.5), (-180.0 + M, -132.5 - M)),
])
def test_a_full_circle_bound_lets_off_screen_bearings_through(band, expected):
    assert SS._design_band(band, 47.5, bound_deg=180.0) == pytest.approx(expected)


def test_without_the_wider_bound_an_off_screen_band_inverts():
    """画外题以前抽不出来的原因,比"夹到边上"更难看:夹完之后 lo > hi。

    (132.5, 180) 被 47.5 的上界夹成 (132.5, 47.5)——一个**倒序**的对,而且因为
    hi - lo 是负的、小于两倍边距,函数原样把它返回,不报错。这条老行为一直没发作
    只是因为没有任何调用方传过视锥外的带。传了 bound_deg=180 之后才有意义。
    """
    lo, hi = SS._design_band((132.5, 180.0), 47.5)
    assert (lo, hi) == pytest.approx((132.5, 47.5))
    assert lo > hi, "夹完之后是倒序的,不是退化成一点"
