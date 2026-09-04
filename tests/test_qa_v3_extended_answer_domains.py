"""extended 里的答案带也能按场景相机推导,而 card17 原样不动。

card17 现在写死 location_bands_deg 到 ±52.5,带着跟基础题型一样的死区(可见性闸停在
47.5,外侧带各有 5.0 度到不了),而且每加一个房间就要手改一次表。画外那一族要八个扇区,
按房间手写更不现实。

这里钉的是:声明了域的 profile 从场景相机推导,没声明的一行行为不变。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import design_qa_v3_extended_profile as EX  # noqa: E402

PARAMS = {"VISUAL_FOV_MARGIN_DEG": 5.0}


def scene(hfov=105.0):
    return SimpleNamespace(hfov_deg=float(hfov))


def test_a_written_table_is_used_exactly_as_before():
    profile = {"id": "card17",
               "location_bands_deg": [[-52.5, -17.5], [-17.5, 17.5], [17.5, 52.5]],
               "location_band_labels": ["left", "center", "right"]}
    bands, labels = EX._location_bands_and_labels(profile, scene(), PARAMS)
    assert bands == profile["location_bands_deg"]
    assert labels == ["left", "center", "right"]


def test_a_written_table_without_labels_is_still_refused():
    profile = {"id": "card17",
               "location_bands_deg": [[-47.5, 0.0], [0.0, 47.5]]}
    with pytest.raises(ValueError, match="required"):
        EX._location_bands_and_labels(profile, scene(), PARAMS)


def test_a_domain_profile_derives_its_bands_from_the_camera():
    profile = {"id": "f1x", "answer_domain": "camera_cone",
               "answer_shape": {"equal_bands": 3}}
    bands, labels = EX._location_bands_and_labels(profile, scene(105.0), PARAMS)
    assert bands[0][0] == pytest.approx(-47.5)
    assert bands[-1][1] == pytest.approx(47.5)
    assert [b[1] - b[0] for b in bands] == pytest.approx([95.0 / 3] * 3)
    assert labels == ["sector_0", "sector_1", "sector_2"]
    # 换镜头,同一个 profile 自己跟着变
    wide, _ = EX._location_bands_and_labels(profile, scene(120.0), PARAMS)
    assert wide[-1][1] == pytest.approx(55.0)


def test_the_off_screen_family_gets_eight_sectors_without_writing_degrees():
    profile = {"id": "f2a", "answer_domain": "full_circle",
               "answer_shape": {"equal_bands": 8}}
    bands, labels = EX._location_bands_and_labels(profile, scene(), PARAMS)
    assert len(bands) == 8 and len(labels) == 8
    assert [round(b[1] - b[0], 6) for b in bands] == [45.0] * 8
    assert bands[0][0] == pytest.approx(-180.0)
    assert bands[-1][1] == pytest.approx(180.0)


def test_declared_labels_must_match_the_derived_count():
    profile = {"id": "f2a", "answer_domain": "full_circle",
               "answer_shape": {"equal_bands": 8},
               "location_band_labels": ["left", "right"]}
    with pytest.raises(ValueError, match="2 labels for 8 bands"):
        EX._location_bands_and_labels(profile, scene(), PARAMS)


def test_declared_labels_of_the_right_count_are_kept():
    profile = {"id": "f1x", "answer_domain": "camera_cone",
               "answer_shape": {"equal_bands": 3},
               "location_band_labels": ["left", "center", "right"]}
    _, labels = EX._location_bands_and_labels(profile, scene(), PARAMS)
    assert labels == ["left", "center", "right"]


def test_a_domain_profile_without_params_is_refused_not_defaulted():
    """四个既有调用点都不传 params。域 profile 走到那里会拿到空字典,
    边距变 0,推出 ±52.5 而不是 ±47.5——正好是这套机制要消灭的那个死区,
    而且不会报错。所以拒绝。
    """
    profile = {"id": "f1x", "answer_domain": "camera_cone",
               "answer_shape": {"equal_bands": 3}}
    for bad in ({}, None, {"SOMETHING_ELSE": 1}):
        with pytest.raises(ValueError, match="VISUAL_FOV_MARGIN_DEG"):
            EX._location_bands_and_labels(profile, scene(), bad)


def test_a_written_table_still_works_without_params():
    """card17 那条路不需要 params,不能被上面那道守卫连累。"""
    profile = {"id": "card17",
               "location_bands_deg": [[-47.5, 0.0], [0.0, 47.5]],
               "location_band_labels": ["left", "right"]}
    bands, labels = EX._location_bands_and_labels(profile, scene(), {})
    assert labels == ["left", "right"]
