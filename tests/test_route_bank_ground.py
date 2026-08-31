"""UE navigation-point ground evidence used by the debug scene probe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "routes"))

from build_apartment_route_bank import summarize_navigation_ground  # noqa: E402


def test_navigation_ground_summary_uses_z_axis_in_ue_centimetres():
    summary = summarize_navigation_ground(np.asarray([
        [0.0, 0.0, 125.0],
        [100.0, 0.0, 126.0],
        [0.0, 100.0, 124.0],
    ]))
    assert summary == {
        "sampled_ground_z_min_ue_cm": 124.0,
        "sampled_ground_z_median_ue_cm": 125.0,
        "sampled_ground_z_max_ue_cm": 126.0,
        "sampled_ground_z_span_ue_cm": 2.0,
    }


def test_navigation_ground_summary_rejects_invalid_samples():
    with pytest.raises(ValueError):
        summarize_navigation_ground(np.empty((0, 3)))
    with pytest.raises(ValueError):
        summarize_navigation_ground(np.asarray([[0.0, 0.0, float("nan")]]))
