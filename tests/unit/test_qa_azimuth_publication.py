"""Published azimuths are DCASE left positive; the engine frame stays right positive."""
import re

import pytest

from avengine.qa import unified_catalog as u
from avengine.qa.azimuth_publication import PUBLISHED_CONVENTION, publish_azimuth_deg
from avengine.qa.unified_scoring import score_open_form


def test_engine_bearings_are_negated_into_the_published_range():
    assert publish_azimuth_deg(30.0) == -30.0      # 30 degrees right
    assert publish_azimuth_deg(-90.0) == 90.0      # directly left
    assert publish_azimuth_deg(0.0) == 0.0
    assert publish_azimuth_deg(180.0) == -180.0 and publish_azimuth_deg(-180.0) == -180.0


def _stated_ranges():
    _en, zh = u.sector_range_text()
    body = zh.split("：", 1)[1].rstrip("。")
    ranges = {}
    for part in body.split("；"):
        name = re.match(r"^[^0-9+-]+", part).group(0)
        numbers = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", part)]
        ranges[name] = numbers
    return ranges


def test_every_stated_range_is_where_the_sector_judge_puts_it():
    names = {label_zh: value for value, _en, label_zh in u._DIRECTION_SECTORS}
    ranges = _stated_ranges()
    assert set(ranges) == set(names)
    for name, numbers in ranges.items():
        if len(numbers) == 4:     # behind: +157.5..180 and -180..-157.5
            probes = [(numbers[0] + numbers[1]) / 2, (numbers[2] + numbers[3]) / 2]
        else:
            probes = [(numbers[0] + numbers[1]) / 2]
        for published in probes:
            engine = publish_azimuth_deg(published)   # the flip is its own inverse
            assert u._azimuth_sector(engine) == names[name], (name, published)


def test_left_sectors_are_stated_with_positive_angles():
    ranges = _stated_ranges()
    assert ranges["正左方"] == [67.5, 112.5] and ranges["正右方"] == [-112.5, -67.5]
    assert ranges["左前方"] == [22.5, 67.5] and ranges["右前方"] == [-67.5, -22.5]
    en = u.sector_range_text()[0]
    assert "left positive, right negative" in en


def test_a_direction_word_is_read_in_the_published_convention():
    form = {"answer_type": "angle_deg", "truth": 30, "convention": PUBLISHED_CONVENTION,
            "theta_full_deg": 15.0, "theta_half_deg": 30.0}
    assert score_open_form(form, "30 degrees to the left")["score"] == 1.0
    assert score_open_form(form, "30 degrees to the right")["score"] == 0.0
    assert score_open_form(form, "+30")["score"] == 1.0


def test_band_labels_state_left_as_positive():
    labels = {o["value"]: o["label_en"] for o in u._fov_band_options()}
    assert "+13.5°, +40.44°" in labels["fov_band_0"] and "left" in labels["fov_band_0"]
    assert "-40.44°, -13.5°" in labels["fov_band_2"] and "right" in labels["fov_band_2"]
