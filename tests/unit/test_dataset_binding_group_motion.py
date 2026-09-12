from copy import deepcopy
import math

import numpy as np
import pytest

from avengine.dataset.binding_group_motion import _root_state, _settings, select_early_audio
from avengine.dataset.binding_group_native import BindingNativeError


def settings():
    return {"binding_motion": {"walk_speed_range_mps": [0.5, 0.8],
        "minimum_motion_s": 2.0, "end_hold_s": 0.5,
        "minimum_entity_separation_m": 0.95, "angle_tolerance_deg": 10,
        "source_start_s": 0.1}}


@pytest.mark.parametrize("speeds", ([0, 1], [1, 0.5], [0.5, math.inf], [math.nan, 1]))
def test_invalid_motion_speeds_rejected(speeds):
    request = settings()
    request["binding_motion"]["walk_speed_range_mps"] = speeds
    with pytest.raises(BindingNativeError):
        _settings(request)


def test_missing_explicit_motion_budget_rejected():
    request = settings()
    del request["binding_motion"]["minimum_motion_s"]
    with pytest.raises(BindingNativeError):
        _settings(request)


def test_rotation_camera_rejected_before_asset_loading():
    request = settings()
    request["camera"] = {"motion": "pan"}
    with pytest.raises(BindingNativeError, match="moving cameras"):
        select_early_audio({"clock": {}}, request, np.random.default_rng(1))


def test_native_walk_period_and_initial_silent_pose_preserved():
    actor = {"timeline": {"walking_action_id": "native_walk", "idle_action_id": "native_idle",
             "walk_phase_period_frames": 8, "local_anatomical_forward_axis": [1, 0, 0]},
             "emitter_binding": {"emitter_offset_m": [0.1, 1.5, 0]}}
    initial = {"root_transform": {"translation_m": [0, 0, 0],
               "rotation_xyzw": [0, 0, 0, 1], "scale": [1, 1, 1]}}
    original = deepcopy(initial)
    path = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0.05, 0, 0],
                     [0.1, 0, 0], [0.1, 0, 0]], dtype=float)
    states = _root_state(actor, initial, path, 15, 3200)
    assert initial == original
    assert [s["action_id"] for s in states] == [
        "native_idle", "native_idle", "native_walk", "native_walk", "native_idle", "native_idle"]
    assert states[0]["root_transform"] == initial["root_transform"]
    assert states[2]["action_phase"] == 1/8
    assert states[3]["action_phase"] == 2/8
    np.testing.assert_allclose(states[-1]["planned_emitter_m"], [0.2, 1.5, 0])


def test_path_triangle_prefilter_preserves_real_occlusion():
    from avengine.dataset.binding_group_motion import _visible_path
    from avengine.qa.answerability import MeshHandle
    vertices = np.array([[-2, -2, -1], [2, -2, -1], [2, 2, -1], [-2, 2, -1],
                         [100, 100, 100], [101, 100, 100], [100, 101, 100]], dtype=float)
    mesh = MeshHandle(vertices, np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]]))
    camera = {"position_m": [0, 0, 0], "horizontal_fov_deg": 90,
              "basis": {"right": [1, 0, 0], "forward": [0, 0, -1]}}
    clear = [{"planned_emitter_m": [0, 0, -0.5]}] * 3
    blocked = [{"planned_emitter_m": [0, 0, -2]}] * 3
    assert _visible_path(clear, camera, mesh, 5) is None
    assert _visible_path(blocked, camera, mesh, 5) == "blocked"
    assert len(mesh.triangles) == 3


def test_native_segments_start_at_retained_endpoint_without_timewarp():
    from types import SimpleNamespace
    from avengine.dataset.binding_group_motion import _native_segments
    points = np.column_stack([np.arange(75)*0.04, np.zeros(75), np.zeros(75)])
    space = SimpleNamespace(frame_rate_hz=15, metadata={"route_authority": "native_recast"},
                            route_bank=lambda: [{"route_id": "route", "points_m": points}])
    choices = _native_segments(space, points[-1], 31, 40, 15)
    assert len(choices) == 10
    for selected, metadata in choices:
        np.testing.assert_array_equal(selected, points[metadata["source_frame_indices"]])
        np.testing.assert_array_equal(selected[0], points[-1])
        np.testing.assert_allclose(np.linalg.norm(np.diff(selected, axis=0), axis=1)*15, 0.6)
        assert metadata["direction"] == "reverse_retained_geometry"
        assert metadata["time_stretch_applied"] is False
    with pytest.raises(BindingNativeError, match="clock"):
        _native_segments(space, points[-1], 31, 40, 30)


def test_native_identity_paths_share_endpoint_without_resampling():
    from types import SimpleNamespace
    from avengine.dataset.binding_group_motion import native_common_endpoint_paths
    points = np.column_stack([np.arange(51) * 0.7/15, np.zeros(51), np.zeros(51)])
    space = SimpleNamespace(frame_rate_hz=15, metadata={"route_authority": "native_recast"},
                            route_bank=lambda: [{"route_id": "route", "points_m": points}])
    candidates = native_common_endpoint_paths(
        space, {"a": points[0], "b": points[-1]}, frame_rate_hz=15,
        minimum_motion_s=0.5, path_length_range_m=[0.8, 2.0],
        walk_speed_range_mps=[0.65, 0.8], minimum_entity_separation_m=0.95,
        same_floor_tolerance_m=0.35)
    assert candidates
    for candidate in candidates:
        assert candidate["native_timing_preserved"] is True
        np.testing.assert_array_equal(candidate["paths"]["a"][-1], candidate["paths"]["b"][-1])
        for actor in ("a", "b"):
            record = candidate["route_records"][actor]
            np.testing.assert_array_equal(candidate["paths"][actor], points[record["source_frame_indices"]])
            assert candidate["native_frame_counts"][actor] == len(record["source_frame_indices"])
            assert record["time_stretch_applied"] is False
            assert record["coordinate_interpolation_applied"] is False
    other = points.copy()
    other[:, 2] = 1.0
    space.route_bank = lambda: [{"route_id": "a", "points_m": points},
                               {"route_id": "b", "points_m": other}]
    assert not native_common_endpoint_paths(
        space, {"a": points[0], "b": other[0]}, frame_rate_hz=15,
        minimum_motion_s=0.5, path_length_range_m=[0.8, 2.0],
        walk_speed_range_mps=[0.65, 0.8], minimum_entity_separation_m=0.95,
        same_floor_tolerance_m=0.35)


def test_native_visual_answer_preflight_uses_centroid_and_integer_tolerance(tmp_path):
    import json
    from avengine.dataset.binding_group_motion import native_state_visual_angle_preflight
    facts = {"time": {"frame_count": 2}, "camera_calibration": {
        "projection": "pinhole", "public": True, "width_px": 200, "height_px": 200,
        "fx_px": 100, "cx_px": 100}}
    captured = {}
    documents = {}
    for visual, angles in (("v0", (17, -16)), ("v1", (-24, 4))):
        folder = tmp_path / visual
        folder.mkdir()
        captured[visual] = {"capture": str(folder)}
        documents[visual] = {"resolution_hw": [200, 200], "per_instance": {
            actor: {"frames": [{"frame_index": 1, "state": "visible_occluded",
                    "visible_centroid_xy_px": [100+100*math.tan(math.radians(angle)), 100]}]}
            for actor, angle in zip(("source1", "source2"), angles)}}
        (folder/"pixel_visibility_truth.json").write_text(json.dumps(documents[visual]))
    result = native_state_visual_angle_preflight(captured, facts, ["source1", "source2"], 10)
    assert result["status"] == "fail"
    assert [row["comparison"] for row in result["comparisons"] if not row["pass"]] == ["audio_a1"]
    documents["v1"]["per_instance"]["source2"]["frames"][0]["visible_centroid_xy_px"][0] = 100+100*math.tan(math.radians(6))
    (tmp_path/"v1/pixel_visibility_truth.json").write_text(json.dumps(documents["v1"]))
    assert native_state_visual_angle_preflight(captured, facts, ["source1", "source2"], 10)["status"] == "pass"


def test_late_path_does_not_mistake_emitter_rotation_for_visual_displacement(monkeypatch):
    from types import SimpleNamespace
    from avengine.dataset import binding_group_motion as motion
    request = settings()
    request.update(seed=7, profile={"retry_budget_within_profile": 1})
    request["binding_motion"]["minimum_motion_s"] = 1.0
    actor = {"actor_id": "source1", "timeline": {
        "walking_action_id": "walk", "idle_action_id": "idle", "walk_phase_period_frames": 8,
        "local_anatomical_forward_axis": [1, 0, 0]},
        "emitter_binding": {"emitter_offset_m": [5, 0, 0]}}
    initial = {"actor_id": "source1", "root_transform": {
        "translation_m": [0, 0, -10], "rotation_xyzw": [0, 1, 0, 0], "scale": [1, 1, 1]},
        "planned_emitter_m": [-5, 0, -10]}
    camera = {"position_m": [0, 0, 0], "horizontal_fov_deg": 85,
              "basis": {"right": [1, 0, 0], "forward": [0, 0, -1]}}
    space = SimpleNamespace(metadata={"authority": "test_native_pathfinder"},
        route_bank=lambda: None,
        bounds=lambda: np.array([[-20., -1., -20.], [20., 1., 20.]]),
        sample_navigable=lambda rng, bounds: np.array([1., 0., -10.]),
        shortest_path=lambda start, end: np.stack([start, end]),
        is_navigable=lambda point: True)
    monkeypatch.setattr(motion, "load_planning_resources", lambda resources, req: (space, None, None))
    plan = {"resources": {}, "clock": {"frame_rate_hz": 15, "frame_count": 150,
            "ticks_per_frame": 3200},
            "visual_plan": {"camera": camera, "actors": [actor],
                            "frames": [{"actor_states": [initial]}]}}
    # Its acoustic marker turns through over 50 degrees, while the body root
    # moves only about 6 degrees: it cannot certify the 20-degree visual change.
    with pytest.raises(BindingNativeError, match="answer_change_too_small"):
        motion.sample_late_paths(plan, request, 5, 140)


def test_native_state_history_rejects_middle_occlusion_but_ignores_pre_anchor_frames(tmp_path, monkeypatch):
    import json
    from avengine.qa import binding_questions
    from avengine.dataset.binding_group_motion import native_state_visual_history_preflight
    monkeypatch.setattr(binding_questions, "_event", lambda facts, number: {})
    monkeypatch.setattr(binding_questions, "_at_event", lambda facts, event: (1, {"source1": None}))
    path = tmp_path / "pixel_visibility_truth.json"
    rows = [{"frame_index": frame, "state": "visible_clear"} for frame in range(4)]
    rows[0]["state"] = "fully_occluded"
    def write():
        path.write_text(json.dumps({"per_instance": {"source1": {"frames": rows}}}))
    write()
    captures = {"v1": {"capture": str(tmp_path)}}
    facts = {"a0": {"time": {"frame_count": 4}}}
    assert native_state_visual_history_preflight(captures, facts)["status"] == "pass"
    rows[2]["state"] = "fully_occluded"
    write()
    result = native_state_visual_history_preflight(captures, facts)
    assert result["status"] == "fail"
    assert result["checks"][0]["invalid_frames"] == [2]
