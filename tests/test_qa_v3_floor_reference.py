"""Floor reference: the measured floor height a room's render facts must agree with.

Background (2026-09-03): the Apartment scene config carried ground_z_ue_cm 0.0
while the cooked floor sits at about +27 cm, so every Apartment render stood
27 cm too low.  These tests pin the rule that render facts need a measured
floor reference and that the declared ground must equal it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from floor_reference import (  # noqa: E402
    FloorReference,
    FloorReferenceError,
    STATUS_INCONSISTENT,
    STATUS_MEASURED,
    summarize_floor_hits,
    write_floor_reference,
)
from scene_sampler import load_scene  # noqa: E402

THRESHOLDS = {"min_hits": 200, "min_hit_fraction": 0.98, "min_within_fraction": 0.95}


def _rows(z_values):
    return [{"index": i, "origin": "camera_point", "xy_ue_cm": [float(i), 0.0], "hit": True,
             "floor_z_ue_cm": float(z)} for i, z in enumerate(z_values)]


def _reference(root, scene_id="room_a", z_values=None, total=None, **extra):
    z_values = [27.1] * 300 if z_values is None else z_values
    summary = summarize_floor_hits(z_values, total_traces=total or len(z_values))
    return write_floor_reference(root, scene_id=scene_id, native_map="/Game/Test/Map",
                                 method={"kind": "test_trace"}, summary=summary,
                                 rows=_rows(z_values), thresholds=THRESHOLDS, **extra)


def test_summary_is_robust_to_furniture_hits():
    values = [27.1] * 95 + [102.0] * 5          # five traces landed on a table top
    summary = summarize_floor_hits(values, total_traces=100)
    assert summary["median_cm"] == pytest.approx(27.1)
    assert summary["hit_fraction"] == 1.0
    assert summary["within_fraction"] == pytest.approx(0.95)
    assert summary["max_cm"] == 102.0 and summary["mad_cm"] == 0.0
    empty = summarize_floor_hits([], total_traces=10)
    assert empty["hit_count"] == 0 and empty["hit_fraction"] == 0.0
    with pytest.raises(FloorReferenceError):
        summarize_floor_hits([1.0], total_traces=0)


def test_write_load_identity_and_tamper_refusal(tmp_path):
    root = _reference(tmp_path / "floor", z_values=[27.1] * 290 + [28.0] * 10)
    reference = FloorReference.load(root)
    assert reference.status == STATUS_MEASURED
    assert reference.ground_z_ue_cm == pytest.approx(27.1)
    assert reference.matches(27.3) and not reference.matches(0.0) and not reference.matches(28.0)
    identity = reference.identity
    assert identity["scene_id"] == "room_a" and identity["hit_count"] == 300
    assert identity["rows_sha256"] and identity["method"] == "test_trace"
    with pytest.raises(FloorReferenceError, match="refusing to overwrite"):
        _reference(tmp_path / "floor")
    # tampered rows are refused
    rows_path = root / "floor_trace_rows.json"
    rows_path.write_text(rows_path.read_text().replace("27.1", "0.0", 1))
    with pytest.raises(FloorReferenceError, match="sha256"):
        FloorReference.load(root)


def test_inconsistent_measurements_never_feed_render_facts(tmp_path):
    # all traces missed (Kujiale baked-lit floor has no collision for line traces)
    root = _reference(tmp_path / "misses", z_values=[], total=500)
    index = json.loads((root / "floor_reference.json").read_text())
    assert index["status"] == STATUS_INCONSISTENT and index["ground_z_ue_cm"] is None
    with pytest.raises(FloorReferenceError, match="status"):
        FloorReference.load(root)
    # a split-level room: half the hits on another level
    root2 = _reference(tmp_path / "split", z_values=[0.0] * 150 + [27.1] * 150)
    assert json.loads((root2 / "floor_reference.json").read_text())["status"] == STATUS_INCONSISTENT


def _bank_and_request(tmp_path):
    routes = []
    for k in range(4):
        x0 = 300.0 + 10 * k
        routes.append({"route_id": f"r{k}",
                       "samples_ue_cm": [[x0, -200.0 + 400.0 * i / 74.0, 0.0] for i in range(75)],
                       "implied_speed_mps": 0.7})
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"schema": "avengine_apartment_route_bank_v1", "routes": routes}))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "primary_camera_rig": {"world_from_rig": {"translation_m": [0, 1.47, 0]},
                               "shared_calibration": {"hfov_degrees": 105.0}},
        "listener": {"rig_from_listener": {"translation_m": [0, 0, 0]}}}))
    return {"scene_id": "room_a", "backend": "ue_spear", "route_bank": str(bank),
            "camera_base_request": str(request)}


def test_load_scene_requires_a_measured_floor_for_render_facts(tmp_path):
    config = _bank_and_request(tmp_path)
    render = {"native_map": "/Game/Test/Map", "room_profile_id": "room",
              "world_transform": "ue_xyz_cm_to_xzy_m_v1", "ground_z_ue_cm": 27.1}
    # no render facts: nothing to verify, loads as before
    assert load_scene(config).floor is None
    # render facts without a floor reference: refused, and the message names the constant
    with pytest.raises(ValueError, match="ground_z_ue_cm"):
        load_scene(dict(config, render=render))
    floor = _reference(tmp_path / "floor")
    scene = load_scene(dict(config, render=render, floor_reference=str(floor)))
    assert scene.floor is not None and scene.floor.ground_z_ue_cm == pytest.approx(27.1)
    assert scene.provenance["floor_reference"]["status"] == STATUS_MEASURED
    assert scene.render_config["ground_z_ue_cm"] == 27.1
    # the hand-written 0.0 of the 2026-09-03 incident is refused against a measured floor
    with pytest.raises(ValueError, match="disagrees"):
        load_scene(dict(config, render=dict(render, ground_z_ue_cm=0.0),
                        floor_reference=str(floor)))
    with pytest.raises(ValueError, match="disagrees"):
        load_scene(dict(config, render={k: v for k, v in render.items() if k != "ground_z_ue_cm"},
                        floor_reference=str(floor)))
    # another room's floor is not this room's floor
    other = _reference(tmp_path / "other", scene_id="room_b")
    with pytest.raises(ValueError, match="belongs to"):
        load_scene(dict(config, render=render, floor_reference=str(other)))
    # a floor reference may also be given as an object with a path
    scene2 = load_scene(dict(config, render=render, floor_reference={"path": str(floor)}))
    assert scene2.provenance["floor_reference"]["path"] == str(floor)
