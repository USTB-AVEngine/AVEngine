"""Scene inputs keep their meaning across working directories and run copies."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "qa"))
import scene_sampler as SS


def test_relative_resources_follow_the_config_not_the_process(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "room.json"
    doc = {
        "scene_id": "room",
        "route_bank": "../data/routes.json",
        "camera_base_request": "camera.json",
        "camera_clearance_table": "../data/clearance",
        "floor_reference": {"path": "../data/floor"},
        "walkable_grid": "../data/walkable",
        "line_of_sight_grid": {"arrays": "../data/los.npz", "metadata": "../data/los.json"},
    }
    path.write_text(json.dumps(doc))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    got = SS.read_scene_config(path)
    assert got["camera_base_request"] == str(configs / "camera.json")
    assert got["route_bank"] == str(tmp_path / "data/routes.json")
    assert got["floor_reference"]["path"] == str(tmp_path / "data/floor")
    assert got["walkable_grid"] == str(tmp_path / "data/walkable")
    assert got["line_of_sight_grid"]["arrays"] == str(tmp_path / "data/los.npz")
    assert json.loads(path.read_text()) == doc
    recorded = elsewhere / "room.json"
    recorded.write_text(json.dumps(got))
    assert SS.read_scene_config(recorded) == got


def test_invalid_scene_payload_is_not_treated_as_a_config(tmp_path):
    path = tmp_path / "room.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        SS.read_scene_config(path)


def test_checked_in_camera_requests_come_from_this_avengine_checkout():
    for path in (REPO / "examples/qa/scenes").glob("*.json"):
        got = SS.read_scene_config(path)
        camera = Path(got["camera_base_request"])
        assert camera.is_file()
        assert camera.is_relative_to(REPO)
