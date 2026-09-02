"""Pure parts of the camera clearance depth preflight (no SPEAR launch)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from preflight_camera_clearance_depth import (  # noqa: E402
    clearance_statistics,
    poses_from_facts,
    target_band_rows,
    verdict,
)


def _depth(width=64, height=36, background=6.0):
    return np.full((height, width), background, dtype=np.float32)


def test_clear_view_has_no_blocked_columns():
    stats = clearance_statistics(_depth(), (1.0, 1.5, 2.5), hfov_deg=105.0)
    assert stats["near_fraction"]["1.5"] == 0.0
    assert stats["eye_band_blocked_column_fraction"]["1.5"] == 0.0
    assert stats["blocked_azimuth_spans_deg"]["1.5"] == []
    assert verdict(stats, near_m=1.5, blocked_fraction_max=0.3)["camera_view_clear"]


def test_lamp_shade_in_front_of_the_lens_blocks_the_eye_band():
    depth = _depth()
    depth[:, 20:52] = 0.25              # a wide object 0.25 m away, all rows
    stats = clearance_statistics(depth, (1.0, 1.5, 2.5), hfov_deg=105.0)
    assert stats["eye_band_blocked_column_fraction"]["1.5"] == pytest.approx(0.5)
    spans = stats["blocked_azimuth_spans_deg"]["1.5"]
    assert len(spans) == 1 and spans[0][0] < 0 < spans[0][1]
    result = verdict(stats, near_m=1.5, blocked_fraction_max=0.3,
                     metric="eye_band_blocked_column_fraction")
    assert result["camera_view_clear"] is False
    assert result["status"] == "placeholder_research_not_human_calibrated"
    assert stats["target_band_blocked_column_fraction"]["1.5"] == pytest.approx(0.5)


def test_sofa_back_below_eye_height_blocks_the_target_band_not_the_eye_band():
    """Sofa backs and kitchen islands 0.4-0.7 m from a 1.47 m camera leave the
    eye band clear yet hide a 0.5 m dog standing behind them."""
    depth = _depth(width=128, height=72)
    rows = target_band_rows(72, hfov_deg=105.0, aspect=128 / 72,
                            camera_height_m=1.47, target_height_m=0.5,
                            distance_range_m=(2.5, 10.0))
    assert 36 < rows[0] < rows[1] <= 72          # lower half of the frame
    depth[rows[0]:, :] = 0.5                     # everything below the horizon is near
    stats = clearance_statistics(depth, (1.0, 1.5, 2.5), hfov_deg=105.0)
    assert stats["eye_band_blocked_column_fraction"]["1.5"] == 0.0
    assert stats["target_band_blocked_column_fraction"]["1.5"] == 1.0
    assert stats["near_fraction"]["1.5"] > 0.3
    assert verdict(stats, near_m=1.5, blocked_fraction_max=0.2,
                   metric="near_fraction")["camera_view_clear"] is False
    assert verdict(stats, near_m=1.5, blocked_fraction_max=0.2,
                   metric="target_band_blocked_column_fraction")["camera_view_clear"] is False
    assert verdict(stats, near_m=1.5, blocked_fraction_max=0.2,
                   metric="eye_band_blocked_column_fraction")["camera_view_clear"] is True


def test_far_scene_geometry_does_not_count_as_blockage():
    depth = _depth()
    depth[30:, :] = 3.2                 # floor and furniture beyond the near range
    stats = clearance_statistics(depth, (1.0, 1.5, 2.5), hfov_deg=105.0)
    assert stats["near_fraction"]["1.5"] == 0.0
    assert stats["target_band_blocked_column_fraction"]["2.5"] == 0.0
    assert verdict(stats, near_m=1.5, blocked_fraction_max=0.2)["camera_view_clear"]
    assert verdict(stats, near_m=1.5, blocked_fraction_max=0.2)["metric"] == \
        "target_band_blocked_column_fraction"


def test_poses_from_facts_reads_the_recorded_camera(tmp_path):
    fact = tmp_path / "fact_record.json"
    fact.write_text(json.dumps({
        "scene_id": "room", "point_id": "card1F_001",
        "camera": {"ue_cm": [1.0, 2.0, 147.0], "ue_yaw_deg": 30.5}}))
    poses = poses_from_facts([fact])
    assert poses[0]["pose_id"] == "room::card1F_001"
    assert poses[0]["translation_ue_cm"] == [1.0, 2.0, 147.0]
    assert poses[0]["yaw_ue_deg"] == 30.5
    # the same point id from another batch keeps a distinct pose id
    other = tmp_path / "other" / "fact_record.json"
    other.parent.mkdir()
    other.write_text(fact.read_text())
    twice = poses_from_facts([fact, other])
    assert [pose["pose_id"] for pose in twice] == [
        "room::card1F_001", "room::card1F_001#2"]
    assert twice[1]["source_fact"] == str(other.resolve())
