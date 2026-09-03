"""发布出去的方位一律 DCASE 左为正，引擎内部一律右为正，两者不许混。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import qa_v3_azimuth as AZ  # noqa: E402


def test_publishing_mirrors_the_engine_frame():
    # 2026-09-03 实测的一对真值：card1F_397 的目标狗在引擎帧 +37.101（偏右），
    # 同一集另一只在 -34.375（偏左）。发布出去必须各自反过来。
    assert AZ.to_published_deg(37.101) == pytest.approx(-37.101)
    assert AZ.to_published_deg(-34.375) == pytest.approx(34.375)
    assert AZ.to_published_deg(0.0) == pytest.approx(0.0)


def test_the_two_ends_never_both_appear():
    # -180 与 +180 是同一个方向；发布侧只留 +180，免得同一方向出现两种写法。
    assert AZ.to_published_deg(180.0) == pytest.approx(180.0)
    assert AZ.to_published_deg(-180.0) == pytest.approx(180.0)


def test_a_wedge_keeps_lo_le_x_lt_hi_after_publishing():
    # 引擎帧最右那一楔 [17.5, 52.5) 发布成 [-52.5, -17.5)，
    # 而且仍然是有序的 [lo, hi)，消费方不用记住取负会换端。
    assert AZ.to_published_band((17.5, 52.5)) == pytest.approx((-52.5, -17.5))
    assert AZ.to_published_band((-52.5, -17.5)) == pytest.approx((17.5, 52.5))
    lo, hi = AZ.to_published_band((-17.5, 17.5))
    assert lo < hi


def test_side_words_follow_the_published_convention():
    # 发布侧左为正，所以正数是左。card3 的答案就靠这一条。
    assert AZ.side_word(30.0) == "left"
    assert AZ.side_word(-30.0) == "right"
    # 引擎帧 +30 是偏右，发布出去应当答 right
    assert AZ.side_word(AZ.to_published_deg(30.0)) == "right"


def test_the_stem_names_every_landmark_angle():
    """owner 2026-09-03:"在 prompt 里，就把所有的主要度数说清楚分别在哪个角度就行"。

    他试做时两次写了 105，而 105 在正后方——题面不点明 ±90 与 ±180 在哪，
    这个答案就排除不掉。
    """

    sentence = AZ.landmark_sentence(52.5)
    for needed in ("DCASE FOA", "positive", "left", "0 degrees is straight ahead",
                   "+90", "-90", "180", "+52.5", "-52.5"):
        assert needed in sentence, needed


def test_a_published_number_never_travels_without_its_convention():
    block = AZ.published_block(37.101)
    assert block["azimuth_deg"] == pytest.approx(-37.101)
    assert block["convention"] == "dcase_foa_left_positive"
    assert "positive to the left" in block["convention_note"]


def test_the_engine_frame_note_says_it_is_internal():
    assert "never published" in AZ.ENGINE_FRAME_NOTE
