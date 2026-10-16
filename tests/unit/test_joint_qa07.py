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
    direction = np.array([route["inward_unit_xz"][0], 0., route["inward_unit_xz"][1]])
    positions = np.array(route["endpoint_m"])+(distance-distance[-1])[:, None]*direction
    heading = math.atan2(float(direction[2]), float(direction[0]))
    route.update(planned_distance_m=distance.tolist(), planned_positions_m=positions.tolist(),
                 planned_native_yaws_rad=np.full(901, -math.pi/2-heading).tolist())
    entry["applied_root_input"][:3, 3] = positions[0]
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
    args = list(native_fixture(monkeypatch))
    video, entry, scene, distance, request = args
    if defect == "camera": scene["listener_m"] = [0, 1.25, .2]
    if defect == "too_long": distance *= 1.2
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
             "motion_episode": {"complete": True, "world_roots_m": np.c_[np.arange(901)/30, np.zeros(901), np.zeros(901)].tolist()}}
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
    from avengine.construction.joint.geometry import DepthBuffer, predict_visibility
    from avengine.construction.joint.families.qa07 import QA07Plugin
    camera = CameraPose((0., 1.25, 0.), 0., width=1280, height=720)
    weight = np.clip((np.arange(901)/30-3)/3, 0, 1)
    roots = np.c_[-5.+3.*weight, np.zeros(901), np.full(901, -4.)]
    body = BodySpec("human", .35, 1.9, .05)
    actor = ActorPlan("target", roots, body)
    depth = DepthBuffer(camera, np.full((camera.height, camera.width), np.inf))
    def predict(camera, root, body, **kwargs):
        return predict_visibility(camera, root, body, depth=depth, **kwargs)
    checks = list(QA07Plugin({"target": body}).verify(SimpleNamespace(predict=predict),
        FamilyRequest("QA-07", "left", ("target",)), ScenePlan(camera, {"target": actor})))
    assert all(c.status == "pass" for c in checks)
    assert checks[-1].evidence["final_inner_proxy_pixels"] >= 200
    assert checks[-1].evidence["native_preflight_required"] is True


def test_qa07_sound_constraints_use_measured_activity_and_keep_complete_clip_start():
    from avengine.construction.joint.families.qa07 import qa07_sound_requirements
    from avengine.construction.joint.scheduling import schedule_sound_events
    scene = {"events": [{"event_id": "speech_1", "endpoint_id": "voice1", "start_tick": 25}]}
    program = {"events": [{"event_id": "speech_1", "sound_asset_id": "recording"}],
               "records": [{"sound_asset_id": "recording", "sound_class": "speech_playback",
                            "header": {"sample_rate_hz": 16000, "sample_count": 32000},
                            "screens": {"automatic_activity": {"activity_intervals_samples": [[3200, 8000], [9600, 24000]]}}}]}
    constraints = qa07_sound_requirements(scene, program, [{"endpoint_id": "voice1", "asset_id": "real_person"}])
    result = schedule_sound_events(constraints)
    assert result.status == "solved"
    assert result.starts_s == (2.5,)
    assert [(e.start_s, e.end_s) for e in result.events] == [(2.7, 3.), (3.1, 4.)]
    assert all(e.entity_id == "real_person" and e.sound_id == "recording" for e in result.events)
    program["records"][0]["screens"] = {}
    with pytest.raises(ValueError, match="measured activity"):
        qa07_sound_requirements(scene, program, [{"endpoint_id": "voice1", "asset_id": "real_person"}])


@pytest.mark.parametrize("gap,accepted", [(1e-8, True), (.01, False)])
def test_native_region_closes_numeric_seams_but_keeps_real_floor_holes(monkeypatch, gap, accepted):
    from shapely.geometry import box, mapping
    from avengine.construction.joint.families.qa07 import native_entry_route_proposals
    video, entry, scene, distance, request = native_fixture(monkeypatch)
    region = box(-20, -20, -1-gap/2, 20).union(box(-1+gap/2, -20, 20, 20))
    scene["production_scenario"]["motion_floor_polygon"] = mapping(region)
    if accepted:
        assert len(list(native_entry_route_proposals(video, entry, scene, distance, [0, 0, -1], request))) == 1
    else:
        with pytest.raises(ValueError, match="floor polygon"):
            list(native_entry_route_proposals(video, entry, scene, distance, [0, 0, -1], request))


def test_native_execution_does_not_reconstruct_missing_plan(monkeypatch):
    from avengine.construction.joint.families.qa07 import native_entry_route_proposals
    video, entry, scene, distance, request = native_fixture(monkeypatch)
    del request["joint_entry_route"]["planned_positions_m"]
    with pytest.raises(ValueError, match="lacks solved"):
        list(native_entry_route_proposals(video, entry, scene, distance, [0, 0, -1], request))


def test_native_execution_never_moves_the_solved_root_to_the_placement(monkeypatch):
    from avengine.construction.joint.families.qa07 import native_entry_route_proposals
    video, entry, scene, distance, request = native_fixture(monkeypatch)
    entry["applied_root_input"][1, 3] += .01
    with pytest.raises(ValueError, match="root differs"):
        list(native_entry_route_proposals(video, entry, scene, distance, [0, 0, -1], request))


def test_entry_certificate_keeps_absolute_frame_clock_for_a_contiguous_probe_window():
    rows = predicted_rows()
    for row in rows:
        row["frame_index"] += 45
    certificate = entry_prediction_certificate(rows, [360, 640], "left")
    assert certificate["status"] == "pass"
    assert certificate["transitions"][0]["entry_frame"] == 60
    assert certificate["transitions"][0]["stable_frames"] == [60, 105]


def test_probe_samples_every_entry_motion_frame_with_outside_and_stable_margins():
    from avengine.construction.joint.families.qa07 import entry_preflight_frame_indices
    times = np.arange(901)/30
    p = np.c_[np.clip(times-4.3, 0, 3.4), np.zeros(901), np.zeros(901)]
    frames = entry_preflight_frame_indices(p)
    assert frames == list(range(frames[0], frames[-1]+1))
    assert frames[0]/15 <= 4.3-.6
    assert frames[-1]/15 >= 7.7+1.2
    assert len(frames) < 100
    with pytest.raises(ValueError, match="no actual motion"):
        entry_preflight_frame_indices(np.zeros((901, 3)))


def test_original_qa07_witness_requires_a_real_review_in_addition_to_visible_pixels():
    from avengine.construction.joint.families.qa07 import original_qa07_witness
    rows = predicted_rows()
    assets = [{"asset_id": "registered_person", "native_instance_id": 7, "label_color": "blue"}]
    review = {"7": {"status": "reviewed", "value": "blue", "frame_refs": [55]}}
    result = original_qa07_witness(rows, review, assets, "7", [360, 640], "left", "fixture")
    assert result["status"] == "pass"
    assert result["question"]["qa_id"] == "QA-07"
    assert result["question"]["truth"]["value"] == "left"
    assert original_qa07_witness(rows, {}, assets, "7", [360, 640], "left", "fixture")["status"] == "fail"
    assert original_qa07_witness(rows, review, assets, "7", [360, 640], "right", "fixture")["status"] == "fail"


def test_proxy_occlusion_fraction_is_not_an_extra_answer_gate():
    from types import SimpleNamespace
    from avengine.construction.joint.families.qa07 import entry_edge_prediction_certificate
    predictions = [SimpleNamespace(state=r["state"], occluded_fraction=0.,
        evidence={"target_centroid_xy_px": r["target_centroid_xy_px"],
                  "target_proxy_pixels": 0 if r["state"] == "out_of_view" else 1000})
        for r in predicted_rows()]
    clear = entry_edge_prediction_certificate(predictions, [360, 640], "left", first_inside_frame=35)
    assert clear["status"] == "pass"
    assert clear["native_pixel_truth_verified"] is False
    # Visible pixels still give the same original entry answer. A high proxy
    # occlusion percentage cannot reject it under the owner's phase-2 rule C.
    predictions[16].occluded_fraction = .9
    observed = entry_edge_prediction_certificate(predictions, [360, 640], "left", first_inside_frame=35)
    assert observed["status"] == "pass"
    assert observed["maximum_edge_proxy_occlusion"] == .9
    predictions[15].state = "fully_occluded"
    blocked = entry_edge_prediction_certificate(predictions, [360, 640], "left", first_inside_frame=35)
    assert blocked["status"] == "fail"


def test_entry_helpers_are_owned_by_shared_core():
    from avengine.construction.joint import entry
    from avengine.construction.joint.families import qa07
    assert qa07.EntryEnvelope is entry.EntryEnvelope
    assert qa07.tangent_entry_anchors is entry.tangent_entry_anchors
    assert qa07.entry_timing is entry.entry_timing


def test_declared_answer_branch_does_not_change_after_other_parent_finishes():
    state = {"answer_counts": {"QA-07": {"left": 7, "right": 0}}}
    assert requested_side(state, 1, declared_side="left") == "left"
    assert requested_side(state, 0, declared_side="right") == "right"
    with pytest.raises(ValueError, match="declared QA-07"):
        requested_side(state, 0, declared_side="up")


def test_visual_entry_silence_preserves_real_bodies_and_complete_companion_binding():
    from avengine.construction.joint.families.qa07 import declare_silent_entry_target
    row = {"asset_ids": ["moving", "static"], "silent_asset_ids": [],
           "semantic_human_asset_ids": ["moving", "static"],
           "semantic_sound_ids": ["utterance0", "utterance1"],
           "semantic_start_offsets_s": [0., 2.6],
           "source_classes": {"moving": ["speech_playback"], "static": ["speech_playback"]},
           "phase2_authored_speech_schedule_s": {"moving": [0., 10., 20.], "static": [2.6, 12.6, 22.6]}}
    declare_silent_entry_target(row, "moving")
    assert row["asset_ids"] == ["moving", "static"]
    assert row["silent_asset_ids"] == ["moving"]
    assert row["semantic_sound_ids"] == ["utterance1"]
    assert row["semantic_start_offsets_s"] == [2.6]
    assert row["source_classes"] == {"static": ["speech_playback"]}
    assert row["phase2_authored_speech_schedule_s"] == {"static": [2.6, 12.6, 22.6]}
    row["joint_prepared_scene_required"] = True
    with pytest.raises(ValueError, match="before native"):
        declare_silent_entry_target(row, "moving")


def test_silent_walker_has_no_moving_audio_negative_control_obligation():
    from types import SimpleNamespace
    from avengine.construction.joint.families.qa07 import native_entry_counterfactual_checks
    actors = [{"endpoint_id": "moving", "motion_episode": {"status": "pass"}},
              {"endpoint_id": "static"}]
    video = SimpleNamespace(actors=actors,
        _apply_entry=lambda *args, **kwargs: {"emitter_position_m": [0., 1.5, -3.]})
    scene = {"listener_m": [0., 1.1, 0.], "events": [{"endpoint_id": "static"}],
             "sources": {"moving": {}, "static": {}}}
    assert list(native_entry_counterfactual_checks(video, scene)) == []
    # An actually sounding mover still needs the original independent control.
    scene["events"].append({"endpoint_id": "moving"})
    checks = list(native_entry_counterfactual_checks(video, scene))
    assert len(checks) == 1 and checks[0].status == "fail"


def test_obstructed_stage_entry_does_not_spend_the_companion_layout_beam():
    from types import SimpleNamespace
    from avengine.construction.joint import BodySpec, CameraPose, ActorPlan
    from avengine.construction.joint.geometry import DepthBuffer, predict_visibility
    from avengine.construction.joint.families.qa07 import entry_stage_seed_preview
    camera = CameraPose((0., 1.25, 0.), 0., width=640, height=360)
    weight = np.clip((np.arange(901)/30-3)/3, 0, 1)
    roots = np.c_[-5.+3.*weight, np.zeros(901), np.full(901, -4.)]
    actor = ActorPlan("moving", roots, BodySpec("human", .35, 1.9, .05))
    for depth_value, expected in [(np.inf, True), (1., False)]:
        depth = DepthBuffer(camera, np.full((camera.height, camera.width), depth_value))
        context = SimpleNamespace(predict=lambda cam, root, body, **kwargs:
            predict_visibility(cam, root, body, depth=depth, **kwargs))
        assert entry_stage_seed_preview(context, camera, actor) is expected


def test_registered_companion_cannot_hide_the_entry_after_stage_preview_passes():
    from types import SimpleNamespace
    from avengine.construction.joint import BodySpec, CameraPose, ActorPlan
    from avengine.construction.joint.geometry import DepthBuffer, predict_visibility
    from avengine.construction.joint.families.qa07 import entry_stage_seed_preview
    camera = CameraPose((0., 1.25, 0.), 0., width=640, height=360)
    weight = np.clip((np.arange(901)/30-3)/3, 0, 1)
    roots = np.c_[-5.+3.*weight, np.zeros(901), np.full(901, -4.)]
    body = BodySpec("human", .35, 1.9, .05)
    actor = ActorPlan("moving", roots, body)
    depth = DepthBuffer(camera, np.full((camera.height, camera.width), np.inf))
    context = SimpleNamespace(predict=lambda cam, root, spec, **kwargs:
        predict_visibility(cam, root, spec, depth=depth, **kwargs))
    assert entry_stage_seed_preview(context, camera, actor)
    assert not entry_stage_seed_preview(context, camera, actor,
        occluders=(("real_companion", np.array([-1.7, 0., -3.]), body),))


def test_partial_proxy_visibility_does_not_certify_an_actual_occlusion_to_visible_event():
    from types import SimpleNamespace
    from avengine.construction.joint.families.qa07 import entry_edge_prediction_certificate, original_qa07_witness
    rows = predicted_rows()
    predictions = [SimpleNamespace(state=r["state"], occluded_fraction=.5,
        evidence={"target_centroid_xy_px":r["target_centroid_xy_px"],
                  "target_proxy_pixels":0 if r["state"]=="out_of_view" else 1000})
        for r in rows]
    assert entry_edge_prediction_certificate(predictions, [360,640], "left", first_inside_frame=35)["status"] == "pass"
    rows[15]["state"] = "fully_occluded"
    assets = [{"asset_id":"real_person", "native_instance_id":7, "label_color":"blue"}]
    review = {"7":{"status":"reviewed", "value":"blue", "frame_refs":[55]}}
    assert original_qa07_witness(rows, review, assets, "7", [360,640], "left", "fixture")["status"] == "fail"

def endpoint_test_context():
    from avengine.rooms.walkable_map import WalkableMap, VERSION, SPECIES_RADII_M
    from avengine.construction.joint import RoomContext, CameraPose
    class FlatNav:
        def snap_point(self, p):
            return np.asarray(p)
    mask = np.ones((400, 400), bool)
    meta = {"version": VERSION, "species_radii_m": SPECIES_RADII_M,
            "footprint_policy": "independent-radius-coverage-v2", "room_id": "endpoint-fixture"}
    walkable = WalkableMap([-10., -10.], .05, 0., {k: mask.copy() for k in SPECIES_RADII_M}, meta)
    room = {"room_label": "endpoint-fixture", "floor_polygon": {"type": "Polygon",
            "coordinates": [[[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]]},
            "measured_floor_y_m": 0., "placement": {"camera_m": [0.,1.25,0.]}}
    context = RoomContext(room, walkable_map=walkable, pathfinder=FlatNav(),
                          camera_limit=6, grid_step_m=.5, width=640, height=360)
    context.cameras = (CameraPose((0.,1.25,0.),0.,width=640,height=360),
                       CameraPose((0.,1.25,0.),45.,width=640,height=360))
    return context


@pytest.mark.parametrize("side", ["left", "right"])
def test_endpoint_domains_cover_hundreds_of_pairs_on_the_requested_side(side):
    from avengine.construction.joint import BodySpec
    from avengine.construction.joint.families.qa07 import ranked_entry_endpoint_pairs, entry_keypoint_roles
    context = endpoint_test_context()
    camera = context.cameras[0]; body = BodySpec("human", .35, 1.9, .05)
    pairs, census = ranked_entry_endpoint_pairs(context, camera, body, side)
    assert len(pairs) == 512 and census["available_pairs"] > 512
    assert entry_keypoint_roles(side) == {"start": "outside_"+side, "end": "inside_full_body"}
    grid = context.grid(camera, body)
    for _, a, b in pairs:
        assert grid.frustum_state[a] == "outside" and grid.frustum_state[b] == "inside"
        assert grid.azimuth_deg[a] * (-1 if side == "left" else 1) > 0
        assert min(grid.distances_m[a],grid.distances_m[b]) >= 1.5


def test_endpoint_domains_do_not_impose_the_old_seven_metre_distance_ring():
    from avengine.construction.joint import BodySpec
    from avengine.construction.joint.families.qa07 import ranked_entry_endpoint_pairs
    context = endpoint_test_context()
    context.points_m = np.array([[-9.,0.,-7.], [0.,0.,-8.]])
    context._grids.clear()
    pairs, census = ranked_entry_endpoint_pairs(context, context.cameras[0], BodySpec("human",.35,1.9,.05), "left")
    assert census["available_pairs"] == 1
    assert [(a,b) for _,a,b in pairs] == [(0,1)]


def test_endpoint_clock_is_consumed_by_original_native_clock_without_refitting():
    from avengine.construction.joint import BodySpec
    from avengine.construction.joint.families.qa07 import endpoint_route
    from avengine.assets.roster_motion.navmesh_execution import human_distance_driven_clock
    context = endpoint_test_context()
    actor = endpoint_route(context, context.cameras[0], [-6.,0.,-5.], [0.,0.,-5.],
                           BodySpec("human",.35,1.9,.05), "target")
    transport = actor.metadata["native_distance_clock_transport"]
    result = human_distance_driven_clock(np.array([0.,1.]), [0.], actor.metadata["speed_m_s"],
        actor.metadata["walk_intervals_s"], timed_route=transport)
    assert np.array_equal(result[3], np.asarray(actor.metadata["distance_clock_m"]))
    assert np.allclose(actor.positions_m[[0,-1]], [[-6.,0.,-5.],[0.,0.,-5.]])
    assert .8 <= actor.metadata["speed_m_s"] <= 1.4
    assert len(np.unique(actor.headings_rad)) == 1


def test_entry_may_finish_in_the_centre_after_a_valid_left_entry():
    from types import SimpleNamespace
    from avengine.construction.joint import BodySpec, FamilyRequest, ScenePlan
    from avengine.construction.joint.geometry import DepthBuffer, predict_visibility
    from avengine.construction.joint.families.qa07 import QA07Plugin, endpoint_route
    context = endpoint_test_context(); camera = context.cameras[0]
    body = BodySpec("human",.35,1.9,.05)
    actor = endpoint_route(context, camera, [-6.,0.,-5.],[0.,0.,-5.],body,"target")
    depth = DepthBuffer(camera,np.full((360,640),np.inf))
    proxy = SimpleNamespace(predict=lambda c,r,b,**kw:predict_visibility(c,r,b,depth=depth,**kw))
    plan = ScenePlan(camera,{"target":actor})
    checks = list(QA07Plugin({"target":body}).verify(proxy,FamilyRequest("QA-07","left",("target",)),plan))
    assert all(c.status == "pass" for c in checks)
    assert plan.metadata["qa07_window_starts_s"]
    observation = next(c for c in checks if c.name == "entry_projection_observation")
    assert observation.evidence["all_projected_body_points_on_side"] is False


def test_endpoint_generator_uses_each_camera_yaw_even_at_the_same_position():
    from avengine.construction.joint import BodySpec, FamilyRequest
    from avengine.construction.joint.families.qa07 import QA07Plugin
    context = endpoint_test_context(); body = BodySpec("human",.35,1.9,.05)
    plugin = QA07Plugin({"target":body})
    stream = plugin.candidates(context,FamilyRequest("QA-07","left",("target",),{"pairs_per_camera":512}))
    next(stream)
    assert plugin.seed_diagnostics["camera_candidates"] == 2
    assert plugin.seed_diagnostics["cameras_with_endpoint_pairs"] == 2
    assert plugin.seed_diagnostics["ranked_endpoint_pairs"] == 1024
    stream.close()
