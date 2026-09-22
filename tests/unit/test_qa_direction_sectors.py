# -*- coding: utf-8 -*-
"""The direction answer domain: eight sectors, right positive, boundaries refused."""
import pytest

from avengine.qa.unified_catalog import (
    _DIRECTION_SECTORS,
    _SECTOR_DEAD_ZONE_DEG,
    _azimuth_sector,
    _choice_aliases,
    _option,
)


def test_the_domain_is_eight_named_sectors():
    values = [value for value, _en, _zh in _DIRECTION_SECTORS]
    assert len(values) == 8 and len(set(values)) == 8
    # guessing one of eight is 12.5%, against 50% for the two-way question this replaced
    assert 1 / len(values) == pytest.approx(0.125)


@pytest.mark.parametrize(
    "azimuth,expected",
    [
        (0.0, "front"),
        (45.0, "front-right"),
        (90.0, "right"),
        (135.0, "back-right"),
        (179.0, "back"),
        (-179.0, "back"),
        (-135.0, "back-left"),
        (-90.0, "left"),
        (-45.0, "front-left"),
        # the two bearings that motivated the change, read off the real bank
        (-28.2, "front-left"),
        (-53.3, "front-left"),
        (147.1, "back-right"),
    ],
)
def test_right_is_positive_and_each_sector_covers_its_own_45_degrees(azimuth, expected):
    assert _azimuth_sector(azimuth) == expected


@pytest.mark.parametrize("boundary", [22.5, 67.5, 112.5, 157.5, -22.5, -67.5, -112.5, -157.5])
def test_a_bearing_on_a_boundary_has_no_defensible_answer(boundary):
    assert _azimuth_sector(boundary) is None
    assert _azimuth_sector(boundary + _SECTOR_DEAD_ZONE_DEG * 0.5) is None
    assert _azimuth_sector(boundary - _SECTOR_DEAD_ZONE_DEG * 0.5) is None
    # just outside the dead zone the sector resolves again
    assert _azimuth_sector(boundary + _SECTOR_DEAD_ZONE_DEG * 1.01) is not None


def test_a_non_finite_bearing_is_refused_rather_than_labelled():
    assert _azimuth_sector(float("nan")) is None
    assert _azimuth_sector(float("inf")) is None


def test_compound_sector_names_win_over_the_plain_ones_they_contain():
    """"front left" contains "left" and 左后方 contains 左.

    The closed-set matcher keeps the longest hit, so the compound name has to be the one
    that survives; otherwise every diagonal answer would be scored as a cardinal one.
    """
    options = [_option(value, label_en) for value, label_en, _zh in _DIRECTION_SECTORS]
    aliases = _choice_aliases(options)
    for compound, plain in (("front-left", "left"), ("front-right", "right"),
                            ("back-left", "left"), ("back-right", "right")):
        assert compound in aliases and plain in aliases
        longest_compound = max(aliases[compound], key=len)
        assert any(term in longest_compound for term in aliases[plain]), (
            "the compound name must contain the plain one for the longest-hit rule to apply"
        )
