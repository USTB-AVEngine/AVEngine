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


def _scene_config(tmp_path, *, hfov=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base_request.json"
    base.write_text(json.dumps({
        "primary_camera_rig": {"shared_calibration": {"hfov_degrees": 98.0}}}))
    config = {"scene_id": "room", "camera_base_request": str(base)}
    if hfov is not None:
        config["hfov_deg"] = hfov
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(config))
    return path


def test_camera_hfov_comes_from_the_scene_config_not_a_constant(tmp_path):
    from argparse import Namespace
    from preflight_camera_clearance_depth import resolve_camera_hfov
    # explicit scene key wins, and the source is recorded
    contract = resolve_camera_hfov(Namespace(
        hfov_deg=None, scene_config=_scene_config(tmp_path, hfov=105.0)))
    assert contract["hfov_deg"] == 105.0
    assert contract["source"].startswith("scene_config:")
    # without the key the camera base request calibration is used
    contract = resolve_camera_hfov(Namespace(
        hfov_deg=None, scene_config=_scene_config(tmp_path / "b")))
    assert contract["hfov_deg"] == 98.0
    # an explicit flag overrides both and says so
    contract = resolve_camera_hfov(Namespace(
        hfov_deg=90.0, scene_config=_scene_config(tmp_path / "c", hfov=105.0)))
    assert contract == {"hfov_deg": 90.0, "source": "cli:--hfov-deg"}
    with pytest.raises(RuntimeError, match="HFOV must come from"):
        resolve_camera_hfov(Namespace(hfov_deg=None, scene_config=None))


def test_cli_refuses_to_run_without_a_camera_contract(tmp_path):
    from preflight_camera_clearance_depth import parse_args
    common = ["--stage-root", "s", "--spear-executable", "e", "--native-map",
              "/Game/m", "--poses", "p.json", "--output", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        parse_args(common)
    args = parse_args(common + ["--hfov-deg", "90"])
    assert args.hfov_deg == 90.0
    with pytest.raises(SystemExit):
        parse_args(common + ["--hfov-deg", "0"])
