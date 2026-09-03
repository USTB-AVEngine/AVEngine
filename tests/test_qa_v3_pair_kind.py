"""PAIR_KIND must match the selected actor assets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from design_qa_v3_scene_batch import (  # noqa: E402
    COAT_WORDS,
    assert_assets_match_pair_kind,
)


def test_pair_kind_cat_with_dog_assets_fails_closed():
    dogs = list(COAT_WORDS)
    assert_assets_match_pair_kind({"PAIR_KIND": "dog"}, dogs)
    with pytest.raises(ValueError, match="PAIR_KIND"):
        assert_assets_match_pair_kind({"PAIR_KIND": "cat"}, dogs)
    with pytest.raises(ValueError, match="no pair_kind mapping"):
        assert_assets_match_pair_kind({"PAIR_KIND": "dog"}, ["not_a_dog_asset"])
