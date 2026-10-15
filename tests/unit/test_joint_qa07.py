import math
import numpy as np
import pytest

from avengine.construction.joint.families.qa07 import (
    EntryEnvelope, entry_prediction_certificate, requested_side, tangent_entry_anchors,
)

pytestmark = pytest.mark.fast_unit


def test_requested_side_uses_actual_counts_and_alternates_ties():
    assert requested_side({}, 0) == "left"
    assert requested_side({}, 1) == "right"
    assert requested_side({"answer_counts": {"QA-07": {"left": 4, "right": 9}}}, 1) == "left"
    assert requested_side({"family_answer_counts": {"QA-07": {"left": 8, "right": 5}}}, 0) == "right"


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("bearing", [0., 45., 179., -135.])
def test_tangent_inversion_whole_body_side_and_endpoint(side, bearing):
    yaw = math.radians(bearing)
    camera = np.array([1., 1.25, -2.])
    endpoint = camera + [4.2*math.sin(yaw), -1.25, -4.2*math.cos(yaw)]
    result = tangent_entry_anchors(camera, endpoint, side)
    env = EntryEnvelope(); sign = -1 if side == "left" else 1
    forward = np.array(result["camera_forward"]); right = np.array(result["camera_right"])
    for dx in (-env.radius_m, env.radius_m):
        for dz in (-env.radius_m, env.radius_m):
            for dy in (0., env.height_m):
                start = np.array(result["start_m"]) + right*dx + forward*dz + [0, dy, 0] - camera
                end = endpoint + right*dx + forward*dz + [0, dy, 0] - camera
                assert sign*float(start@right) > float(start@forward)*math.tan(math.radians(42.5))
                assert abs(float(end@right)) < float(end@forward)*math.tan(math.radians(42.5))
                assert abs(float(end[1])) < float(end@forward)*math.tan(math.radians(42.5))/(16/9)
    assert result["planning_length_m"] > 0
    assert not result["capture_camera_moves"]
    assert not result["native_pixel_truth_verified"]


def test_vertical_crop_is_not_certified_from_horizontal_azimuth():
    with pytest.raises(ValueError, match="vertically"):
        tangent_entry_anchors([0, 1.5, 0], [0, 0, -2.5], "left")


def predicted_rows(side="left"):
    x = 30 if side == "left" else 609
    return [{"frame_index": i, "state": "out_of_view" if i < 15 else "visible_clear",
             "target_centroid_xy_px": None if i < 15 else [x, 180]} for i in range(60)]


@pytest.mark.parametrize("side", ["left", "right"])
def test_certificate_calls_original_predicate_and_never_claims_native_truth(side):
    certificate = entry_prediction_certificate(predicted_rows(side), [360, 640], side)
    assert certificate["status"] == "pass"
    assert certificate["transitions"][0]["entry_frame"] == 15
    assert certificate["transitions"][0]["stable_s"] == 3
    assert certificate["evidence_kind"] == "geometry_prediction_only"
    assert not certificate["native_pixel_truth_verified"]


@pytest.mark.parametrize("defect", ["wrong_side", "missing_frame", "unknown", "central", "too_short"])
def test_certificate_rejects_bad_or_uncertain_entry(defect):
    rows = predicted_rows()
    if defect == "wrong_side": rows = predicted_rows("right")
    if defect == "missing_frame": rows.pop(20)
    if defect == "unknown": rows[14]["state"] = "unknown"
    if defect == "central":
        for r in rows[15:]: r["target_centroid_xy_px"] = [320, 180]
    if defect == "too_short": rows = rows[:16]
    assert entry_prediction_certificate(rows, [360, 640], "left")["status"] == "fail"


def test_unpublishable_subsecond_interval_is_rejected():
    rows = [{"frame_index": i, "state": "out_of_view" if i < 18 else "visible_clear",
             "target_centroid_xy_px": None if i < 18 else [30, 180]} for i in range(27)]
    assert entry_prediction_certificate(rows, [360, 640], "left")["status"] == "fail"


def native_fixture(monkeypatch, length_extra=.3):
    from types import SimpleNamespace
    from shapely.geometry import box, mapping
    from avengine.rooms import walkable_map
    route = tangent_entry_anchors([0, 1.25, 0], [0, 0, -4.2], "left")
    route.update(room_id="test_room", walkable_map_path="map.npz", native_gait_extension_reserve_m=.65)
    fake_map = SimpleNamespace(contains_many=lambda p: np.ones(len(p), bool),
                               segment_free=lambda a, b: True)
    monkeypatch.setattr(walkable_map, "load_room_map", lambda *args, **kwargs: fake_map)
    def find_path(path):
        path.geodesic_distance = float(np.linalg.norm(path.requested_end-path.requested_start))
        return True
    entry = {"applied_root_input": np.eye(4), "declaration": {"kind": "human"}}
    video = SimpleNamespace(actors=[entry], pathfinder=SimpleNamespace(snap_point=lambda p: p, find_path=find_path),
                            hs=SimpleNamespace(ShortestPath=SimpleNamespace))
    scene = {"listener_m": route["listener_m"], "camera_forward": route["camera_forward"],
             "production_scenario": {"motion_floor_polygon": mapping(box(-20, -20, 20, 20))}}
    distance = np.clip((np.arange(901)/30-3)/3, 0, 1)*(route["planning_length_m"]+length_extra)
    return video, entry, scene, distance, {"joint_entry_route": route}


def test_native_route_preserves_completed_gait_and_solved_endpoint(monkeypatch):
    from avengine.construction.joint.families.qa07 import native_entry_route_proposals
    video, entry, scene, distance, request = native_fixture(monkeypatch)
    before = distance.copy()
    route = next(native_entry_route_proposals(video, entry, scene, distance, [0, 0, -1], request))
    assert np.array_equal(distance, before)
    assert np.allclose(route.positions[-1], request["joint_entry_route"]["endpoint_m"])
    traveled = np.linalg.norm(np.diff(route.positions, axis=0), axis=1)
    assert np.allclose(traveled, np.diff(distance))
    assert route.receipt["proposal_clock_changed"] is False


@pytest.mark.parametrize("defect", ["camera", "too_long", "backwards"])
def test_native_route_rejects_changed_camera_or_uncertified_clock(monkeypatch, defect):
    from avengine.construction.joint.families.qa07 import native_entry_route_proposals
    args = list(native_fixture(monkeypatch, length_extra=1. if defect == "too_long" else .3))
    video, entry, scene, distance, request = args
    if defect == "camera": scene["listener_m"] = [0, 1.25, .2]
    if defect == "backwards": distance[200] = distance[199]-.1
    with pytest.raises(ValueError):
        list(native_entry_route_proposals(video, entry, scene, distance, [0, 0, -1], request))


def test_native_probe_restores_original_pose_when_capture_raises(monkeypatch):
    from types import SimpleNamespace
    from avengine.construction.joint.families.qa07 import native_entry_preflight
    from avengine.construction.joint.families import qa07
    monkeypatch.setattr(qa07, "_native_rest_framing", lambda *args: {"state": "outside"})
    from avengine.capture import continuous_native_visibility, v7_offscreen_layout
    calls = []
    class FailingStream:
        def __init__(self, video): pass
        def capture(self, *args): raise ValueError("native probe failure")
        def close(self): calls.append("closed")
    monkeypatch.setattr(continuous_native_visibility, "NativeVisibilityStream", FailingStream)
    monkeypatch.setattr(v7_offscreen_layout, "native_projection_proof", lambda *args: {"status": "pass"})
    entry = {"endpoint_id": "person", "declaration": {"native_instance_id": "person", "asset_id": "person"},
             "motion_episode": {"complete": True}}
    video = SimpleNamespace(actors=[entry],
        _apply_motion_entry=lambda e, frame: calls.append(frame),
        _native_pair_pixels=lambda scene, entries: {"person": 0},
        sim=SimpleNamespace(get_sensor_observations=lambda: {"semantic": np.zeros((2, 2), int)}))
    scene = {"production_scenario": {"motion_requests": {"person": {"joint_entry_route": {"branch": "left"}}}}}
    with pytest.raises(ValueError, match="native probe failure"):
        native_entry_preflight(video, scene, [entry])
    assert calls == [0, 0, "closed"]


@pytest.mark.parametrize("length", [1.5, 2.3, 3.5, 5.])
def test_entry_timing_keeps_the_complete_stop_away_from_crop_boundary(length):
    from avengine.construction.joint.families.qa07 import entry_timing
    timing = entry_timing(length, 1.05)
    assert timing["start_s"] >= .75
    assert timing["requested_interval_s"][1] == pytest.approx(7.62)
    assert timing["latest_reserved_stop_s"] < 9.
    assert timing["start_s"]+length/1.05+.25 == pytest.approx(7.6)


def test_proxy_lower_bound_zero_does_not_discard_named_inner_proxy_evidence():
    from types import SimpleNamespace
    from avengine.construction.joint import ActorPlan, BodySpec, CameraPose, FamilyRequest, ScenePlan
    from avengine.construction.joint.geometry import VisibilityPrediction, framing
    from avengine.construction.joint.families.qa07 import QA07Plugin
    camera = CameraPose((0., 1.25, 0.), 0., width=1280, height=720)
    weight = np.clip((np.arange(901)/30-3)/3, 0, 1)
    roots = np.c_[-5.+3.*weight, np.zeros(901), np.full(901, -4.)]
    body = BodySpec("human", .35, 1.9, .05)
    actor = ActorPlan("target", roots, body)
    def predict(camera, root, body, **kwargs):
        return VisibilityPrediction(
            "out_of_view" if framing(camera, root, body)[0] == "outside" else "visible_clear",
            visible_pixels_lower=0, evidence={"inner_proxy_visible_pixels": 10000})
    checks = list(QA07Plugin({"target": body}).verify(SimpleNamespace(predict=predict),
        FamilyRequest("QA-07", "left", ("target",)), ScenePlan(camera, {"target": actor})))
    assert all(c.status == "pass" for c in checks)
    assert checks[-1].evidence["final_inner_proxy_pixels"] == 10000
    assert checks[-1].evidence["native_preflight_required"] is True
