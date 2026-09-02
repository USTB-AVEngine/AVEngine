"""Predicted visibility in the scene batch designer: recorded for every
candidate, decisive only when a profile declares a rejecting requirement."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from camera_clearance import (  # noqa: E402
    FACE_YAWS_DEG,
    CameraClearanceTable,
    face_ray_directions,
    min_pool,
    point_key,
)
from design_qa_v3_scene_batch import (  # noqa: E402
    PredictedVisibilityRejection,
    predicted_tier_distribution,
    predicted_visibility_block,
)
from scene_sampler import Route, SceneInputs  # noqa: E402

CAMERA_HEIGHT_M = 1.47
CAM = (100.0, -250.0)
PLANES = [(0, 6.0), (0, -6.0), (1, 6.0), (1, -6.0), (2, -CAMERA_HEIGHT_M), (2, 1.5)]
ISLAND = (np.array([0.8, -0.8, -CAMERA_HEIGHT_M]), np.array([1.4, 0.8, -0.27]))


def _raycast_t(dirs):
    dirs = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    t = np.full(dirs.shape[:-1], np.inf)
    for axis, value in PLANES:
        tt = value / dirs[..., axis]
        t = np.where((tt > 0) & (tt < t), tt, t)
    lo, hi = ISLAND
    t1, t2 = lo / dirs, hi / dirs
    tmin = np.minimum(t1, t2).max(axis=-1)
    tmax = np.maximum(t1, t2).min(axis=-1)
    hit = (tmax >= tmin) & (tmin > 0)
    return np.where(hit & (tmin < t), tmin, t)


def _table(root):
    faces = np.stack([_raycast_t(face_ray_directions(128, 192, yaw)).astype(np.float32)
                      for yaw in FACE_YAWS_DEG])
    root.mkdir()
    (root / "faces").mkdir()
    yaws = np.arange(0, 360, 2.0)
    np.savez_compressed(root / "faces" / "shard_0000.npz",
                        radial_m=np.stack([min_pool(f, 2) for f in faces])[None].astype(np.float16),
                        point_index=np.asarray([0], np.int32), height_index=np.asarray([0], np.int32))
    np.savez_compressed(root / "summaries.npz",
                        target_band_blocked_column_fraction=np.zeros((1, 1, 3, 3, 180), np.float16),
                        eye_band_blocked_column_fraction=np.zeros((1, 1, 3, 180), np.float16),
                        clear_default_rule=np.ones((1, 1, 180), bool),
                        points_xy_cm=np.asarray([CAM], np.float32),
                        camera_heights_m=np.asarray([CAMERA_HEIGHT_M], np.float32),
                        yaws_deg=yaws.astype(np.float32),
                        nears_m=np.asarray([1.0, 1.5, 2.5], np.float32),
                        target_heights_m=np.asarray([0.5, 1.0, 1.7], np.float32),
                        point_seconds=np.zeros((1, 1), np.float32))
    (root / "camera_clearance_table.json").write_text(json.dumps({
        "schema": "qa_v3_camera_clearance_table_v1", "scene_id": "synthetic_room",
        "code": {"revision": "0" * 40, "dirty": False},
        "summaries": {"path": "summaries.npz", "yaw_step_deg": 2.0},
        "faces": {"shards": [{"path": "faces/shard_0000.npz", "count": 1}]},
        "points": {"keys": [point_key(CAM)]}, "stage": {"native_map": "/Game/synthetic"}}))
    return CameraClearanceTable.load(root)


def _scene(table):
    route = Route("r0", [CAM] * 75, 0.0)
    return SceneInputs(scene_id="synthetic_room", backend="synthetic", routes=[route],
                       stand_points=[CAM], camera_points=[CAM],
                       camera_height_m=CAMERA_HEIGHT_M, hfov_deg=105.0, clearance=table,
                       render_config={"native_map": "/Game/synthetic", "room_profile_id": "r",
                                      "world_transform": "ue_xyz_cm_to_xzy_m_v1",
                                      "ground_z_ue_cm": 0.0})


def _at(bearing_deg, distance_m):
    return [CAM[0] + 100.0 * distance_m * math.cos(math.radians(bearing_deg)),
            CAM[1] + 100.0 * distance_m * math.sin(math.radians(bearing_deg)), 0.0]


def _timeline(target_positions, other_positions, yaw):
    frames = []
    for index in range(75):
        frames.append({"frame_index": index,
                       "camera": {"translation_ue_cm": [CAM[0], CAM[1], 147.0], "yaw_ue_deg": yaw},
                       "actor_states": [
                           {"source_slot_id": "source1", "translation_ue_cm": target_positions[index]},
                           {"source_slot_id": "source2", "translation_ue_cm": other_positions[index]}]})
    return {"frames": frames}


PARAMS = {"PIXEL_TIER_VISIBLE_FRACTION_EDGES": [0.5, 0.2]}


def test_block_records_tiers_statistics_and_declarations(tmp_path):
    scene = _scene(_table(tmp_path / "table"))
    # target hides behind the island (bearing 0) until frame 40, then walks
    # into the open; the other dog stays in the open on the left
    target = [_at(0.0, 3.0)] * 40 + [_at(180.0 - 1.0 * k, 3.0) for k in range(35)]
    other = [_at(200.0, 3.5)] * 75
    profile = {"id": "card1F", "visual_requirements": [
        {"referent": "target", "frames": ["anchor", "query"], "mode": "tier"},
        {"referent": "other", "frames": ["query"], "min_predicted_visible_fraction": 0.5,
         "mode": "tier"}]}
    block, failures = predicted_visibility_block(
        scene, PARAMS, profile, _timeline(target, other, yaw=180.0),
        target_slot="source1", other_slot="source2", camera_height_m=CAMERA_HEIGHT_M,
        instants={"anchor": 20, "query": 74})
    assert failures == []
    assert block["status"] == "predicted"
    assert block["authority"].startswith("prediction_from_camera_clearance_table")
    assert block["at_instants"]["target"]["anchor"]["tier"] == "out_of_view"
    assert block["at_instants"]["target"]["query"]["tier"] in ("light", "medium")
    assert block["at_instants"]["other"]["query"]["tier"] == "light"
    assert block["statistics"]["target"]["visible_near_instant"] == {"anchor": False, "query": True}
    assert block["statistics"]["other"]["never_visible"] is False
    assert len(block["per_frame"]["target"]) == 75
    assert all(d["satisfied"] for d in block["declarations"] if d["referent"] == "other")
    dist = predicted_tier_distribution([{"profile_id": "card1F", "predicted_visibility": block}])
    assert dist["card1F"]["at_instants"]["target@anchor"] == {"out_of_view": 1}
    assert dist["card1F"]["never_visible"] == {}


def test_rejecting_declaration_fails_the_candidate_with_evidence(tmp_path):
    scene = _scene(_table(tmp_path / "table"))
    target = [_at(0.0, 3.0)] * 75            # behind the island the whole clip
    other = [_at(200.0, 3.5)] * 75
    profile = {"id": "card7", "visual_requirements": [
        {"referent": "target", "frames": [30], "min_predicted_visible_fraction": 0.5,
         "mode": "reject"}]}
    block, failures = predicted_visibility_block(
        scene, PARAMS, profile, _timeline(target, other, yaw=0.0),
        target_slot="source1", other_slot="source2", camera_height_m=CAMERA_HEIGHT_M,
        instants={"anchor": 30, "query": 30})
    assert block["at_instants"]["target"]["query"]["tier"] == "hidden"
    assert len(failures) == 1 and failures[0]["frame"] == 30
    assert block["reject_failures"] == 1
    assert block["statistics"]["target"]["never_visible"] is True
    tier_only = dict(profile, visual_requirements=[dict(profile["visual_requirements"][0], mode="tier")])
    _, none = predicted_visibility_block(
        scene, PARAMS, tier_only, _timeline(target, other, yaw=0.0),
        target_slot="source1", other_slot="source2", camera_height_m=CAMERA_HEIGHT_M,
        instants={"anchor": 30, "query": 30})
    assert none == []
    with pytest.raises(ValueError, match="mode"):
        predicted_visibility_block(
            scene, PARAMS, {"id": "x", "visual_requirements": [
                {"referent": "target", "frames": [1], "mode": "maybe"}]},
            _timeline(target, other, yaw=0.0), target_slot="source1", other_slot="source2",
            camera_height_m=CAMERA_HEIGHT_M, instants={"anchor": 1, "query": 1})
    assert issubclass(PredictedVisibilityRejection, ValueError)


def test_without_a_table_nothing_is_predicted_and_nothing_rejects(tmp_path):
    scene = _scene(None)
    profile = {"id": "card7", "visual_requirements": [
        {"referent": "target", "frames": [30], "min_predicted_visible_fraction": 0.5,
         "mode": "reject"}]}
    block, failures = predicted_visibility_block(
        scene, PARAMS, profile, _timeline([_at(0.0, 3.0)] * 75, [_at(200.0, 3.5)] * 75, yaw=0.0),
        target_slot="source1", other_slot="source2", camera_height_m=CAMERA_HEIGHT_M,
        instants={"anchor": 30, "query": 30})
    assert block["status"] == "not_predicted" and failures == []
    assert block["declarations"] == profile["visual_requirements"]
