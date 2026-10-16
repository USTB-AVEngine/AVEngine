"""Synthetic preflight regressions; only native production counts as delivery."""
from copy import deepcopy

import pytest

from avengine.construction.joint.families.qa09 import (
    ReappearanceDemand, certify_mask_sequence, requested_answer,
)

pytestmark = pytest.mark.fast_unit


def sequence(answer="yes"):
    states = ["visible_clear"] * 45 + ["fully_occluded"] * 30
    states += ["visible_clear" if answer == "yes" else "fully_occluded"] * 75
    frames = [{"frame_index": i, "state": state, "target_pixels": 1000,
               "visible_pixels": 0 if state == "fully_occluded" else 1000}
              for i, state in enumerate(states)]
    occ = [{"frame_index": f["frame_index"], "target_instance_id": "person_id",
            "occluder_instance_ids": ["speaker_id"], "unknown_occluder_pixels": 0,
            "target_only_pixels": 1000, "context_pixels": 0, "occluded_pixels": 1000}
           for f in frames if f["state"] == "fully_occluded"]
    return frames, occ


def certify(frames, occ, answer="yes", **kwargs):
    return certify_mask_sequence(frames, occ, ReappearanceDemand("person", "speaker", answer, **kwargs),
        target_instance_id="person_id", occluder_instance_id="speaker_id",
        registered_instance_ids=["person_id", "speaker_id"])


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_complete_registered_sequence_certifies_its_actual_branch(answer):
    f, o = sequence(answer)
    result = certify(f, o, answer)
    assert result["status"] == "pass"
    assert result["observed_answer"] == answer
    assert result["clear_before_intervals_frames"] == [[0, 44]]


def test_no_branch_rejects_even_one_late_partial_visible_frame():
    f, o = sequence("no")
    f[149].update(state="visible_occluded", visible_pixels=1)
    result = certify(f, o, "no")
    assert result["status"] == "fail"
    assert result["observed_answer"] == "yes"
    assert "observed_branch_differs_from_requested" in result["errors"]


def test_transient_early_hide_already_changes_the_original_no_answer():
    f, o = sequence("no")
    f[1].update(state="fully_occluded", visible_pixels=0)
    result = certify(f, o, "no")
    assert result["observed_answer"] == "yes"
    assert result["status"] == "fail"


@pytest.mark.parametrize("change", ["missing_end", "duplicate", "reordered"])
def test_sparse_or_duplicate_stream_cannot_certify_absence(change):
    f, o = sequence("no")
    if change == "missing_end":
        f.pop()
    elif change == "duplicate":
        f[60] = deepcopy(f[59])
    else:
        f[59], f[60] = f[60], f[59]
    result = certify(f, o, "no")
    assert result["status"] == "fail"
    assert result["observed_answer"] is None


@pytest.mark.parametrize("field,value", [
    ("occluder_instance_ids", []), ("occluder_instance_ids", ["stage"]),
    ("occluder_instance_ids", ["speaker_id", "other"]),
    ("unknown_occluder_pixels", 1), ("target_only_pixels", 999),
    ("context_pixels", 1), ("occluded_pixels", 999),
])
def test_full_occlusion_needs_unique_registered_mask_evidence(field, value):
    f, o = sequence()
    for r in o:
        r[field] = value
    result = certify(f, o)
    assert result["status"] == "fail"
    assert "no_stable_registered_full_occlusion" in result["errors"]


def test_out_of_view_is_never_full_occlusion():
    f, o = sequence()
    for r in f[45:75]:
        r.update(state="out_of_view", target_pixels=0)
    result = certify(f, o)
    assert "no_fully_occluded_frame" in result["errors"]


def test_optional_construction_duration_is_explicit_and_default_matches_existence():
    f, o = sequence()
    for r in f[60:75]:
        r.update(state="visible_clear", visible_pixels=1000)
    assert certify(f, o, minimum_hidden_s=1.)["status"] == "pass"
    f[59].update(state="visible_clear", visible_pixels=1000)
    assert "no_stable_registered_full_occlusion" in certify(f, o, minimum_hidden_s=1.)["errors"]
    assert certify(f, o)["status"] == "pass"


def test_one_actual_visible_pixel_satisfies_original_reappearance_predicate():
    f, o = sequence()
    for r in f[75:]:
        r.update(state="visible_occluded", visible_pixels=1)
    result = certify(f, o)
    assert result["observed_answer"] == "yes"
    assert result["status"] == "pass"


def test_no_branch_must_have_been_seen_before_becoming_hidden():
    f, o = sequence("no")
    for r in f[:45]:
        r.update(state="out_of_view", visible_pixels=0, target_pixels=0)
    assert "no_stable_visibility_before_first_occlusion" in certify(f, o, "no")["errors"]


def test_no_branch_can_leave_view_after_registered_full_occlusion():
    f, o = sequence("no")
    f[149].update(state="out_of_view", visible_pixels=0, target_pixels=0)
    result = certify(f, o, "no")
    assert result["observed_answer"] == "no"
    assert result["status"] == "pass"


def test_requests_balance_actual_counts_and_keep_negative_branch_available():
    assert requested_answer({}, 0) == "yes"
    assert requested_answer({}, 1) == "no"
    assert requested_answer({"answer_counts": {"QA-09": {"yes": 8, "no": 2}}}) == "no"
    assert requested_answer({"family_answer_counts": {"QA-09": {"yes": 0, "no": 2}}}) == "yes"
    state = {"answer_branch_distribution": {"families": {"QA-09": {"answers": {"yes": 4, "no": 1}}}}}
    assert requested_answer(state) == "no"


def test_negative_relation_does_not_require_a_later_clear_frame():
    relation = ReappearanceDemand("person", "speaker", "no").relation()
    assert relation["qa09_demand"]["answer"] == "no"
    assert "later_clear_time_s" not in relation


@pytest.mark.parametrize("kwargs", [{"window_start_s": 21}, {"window_start_s": .01},
    {"minimum_hidden_s": 0}, {"frame_rate_hz": 0}, {"window_duration_s": float("nan")}])
def test_invalid_request_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ReappearanceDemand("person", "speaker", "no", **kwargs)


def final_clock(answer="yes"):
    import numpy as np
    times = np.arange(901) / 30
    # Deliberately 1.8m first travel: a nominal 1.5m plan must not displace shadow.
    d = np.clip((times - 1.5) / 1.8, 0, 1) * 1.8
    if answer == "yes":
        d += np.clip((times - 6.0) / 1.9, 0, 1) * 1.9
    return d


def test_final_gait_distance_is_anchored_at_shadow_without_rescaling():
    import numpy as np
    from avengine.construction.joint.families.qa09 import clock_anchored_positions
    route = {"answer": "yes", "cross_direction_xz": [1., 0.], "shadow_root_m": [3., .2, -5.]}
    d = final_clock()
    p = clock_anchored_positions(d, route)
    np.testing.assert_allclose(p[135:166], np.broadcast_to([3., .2, -5.], (31, 3)))
    np.testing.assert_allclose(p[0], [1.2, .2, -5.])
    np.testing.assert_allclose(p[-1], [4.9, .2, -5.])
    np.testing.assert_allclose(np.linalg.norm(np.diff(p, axis=0), axis=1), np.diff(d), atol=1e-12)


def test_negative_gait_cannot_leave_its_hidden_stop_later_in_the_parent():
    from avengine.construction.joint.families.qa09 import clock_anchored_positions
    route = {"answer": "no", "cross_direction_xz": [1., 0.], "shadow_root_m": [3., .2, -5.]}
    assert clock_anchored_positions(final_clock("no"), route).shape == (901, 3)
    with pytest.raises(ValueError, match="stay at the hidden stop"):
        clock_anchored_positions(final_clock("yes"), route)


def test_complete_step_extension_into_hidden_interval_is_not_silently_accepted():
    import numpy as np
    from avengine.construction.joint.families.qa09 import clock_anchored_positions
    route = {"answer": "no", "cross_direction_xz": [1., 0.], "shadow_root_m": [3., .2, -5.]}
    bad = np.clip(np.arange(901) / 30 - 1.5, 0, 3.5)
    with pytest.raises(ValueError, match="did not stop"):
        clock_anchored_positions(bad, route)


def test_qa09_reappearance_accepts_stably_visible_occluded_as_original_predicate_does():
    f, o = sequence()
    for r in f[75:]:
        r.update(state="visible_occluded", visible_pixels=400)
    result = certify(f, o)
    assert result["status"] == "pass"
    assert result["observed_answer"] == "yes"
    assert result["visible_after_intervals_frames"] == [[75, 149]]
    assert result["clear_after_intervals_frames"] == []


def test_native_execution_rejects_new_distance_instead_of_solving_another_start():
    import numpy as np
    from avengine.construction.joint.families.qa09 import native_qa09_route_proposals
    d = final_clock()
    request = {"joint_qa09_route": {"planned_distance_m": d.tolist(), "positions_m": np.zeros((901, 3)).tolist()}}
    with pytest.raises(ValueError, match="clock differs"):
        next(native_qa09_route_proposals(None, {}, {}, d + np.linspace(0, .01, 901), [0, 0, -1], request))


def test_native_execution_rejects_relocated_start_instead_of_applying_the_plan_elsewhere():
    import numpy as np
    from avengine.construction.joint.families.qa09 import native_qa09_route_proposals
    d = final_clock();root=np.eye(4);root[0,3]=.1
    request = {"joint_qa09_route": {"planned_distance_m": d.tolist(), "positions_m": np.zeros((901, 3)).tolist()}}
    with pytest.raises(ValueError, match="no execution relocation"):
        next(native_qa09_route_proposals(None, {"applied_root_input": root}, {}, d, [0, 0, -1], request))


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_branch_requests_only_registered_qa09_pipeline_conditions(answer):
    from avengine.timeline.v7_demand import FAMILY_SCENE_REQUIREMENTS
    demand = ReappearanceDemand("target", "occluder", answer)
    assert set(demand.requested_conditions()) <= set(FAMILY_SCENE_REQUIREMENTS["QA-09"]["conditions"])
    assert ("later_visible_reappearance" in demand.requested_conditions()) == (answer == "yes")


def test_pending_map_fallback_requires_explicit_native_path_and_never_hides_other_failures():
    from avengine.construction.joint import Certificate, CheckResult
    from avengine.construction.joint.families.qa09 import planning_admissible
    pending = CheckResult("target:route", "unknown", "native_full_pose_required_for_map_fallback")
    certificate = Certificate((CheckResult("camera", "pass"), pending))
    assert not certificate.passed
    assert not planning_admissible(certificate)
    assert planning_admissible(certificate, native_sweep_pending=True)
    for failure in (CheckResult("spacing", "fail"), CheckResult("floor", "unknown", "missing_mesh")):
        assert not planning_admissible(Certificate(certificate.checks + (failure,)), native_sweep_pending=True)
    assert not planning_admissible(Certificate(()), native_sweep_pending=True)


def test_stationary_turn_preserves_the_promised_registered_hidden_probe_interval():
    from avengine.construction.joint.families.qa09 import gait_intervals, TURNAROUND_INTERVAL_S
    relation = ReappearanceDemand("target", "occluder", "yes").relation()
    start, end = TURNAROUND_INTERVAL_S
    assert max(relation["query_times_s"]) + 1 / 15 < start
    assert end < gait_intervals("yes", "out_and_back")[1][0]
    assert gait_intervals("no", "out_and_back") == [[1.5, 3.3]]


@pytest.mark.parametrize("sign", [0, 2, float("nan"), float("inf")])
def test_arc_turn_sign_rejects_invalid_input_before_asset_loading(sign):
    from avengine.construction.joint.families.qa09 import prepare_gait_geometry
    with pytest.raises(ValueError, match="turn sign"):
        prepare_gait_geometry({}, {"qa09_arc_turn_sign": sign}, {}, "yes")


def test_opposite_arc_has_separate_pose_cache_and_preserves_the_admitted_clock(monkeypatch, tmp_path):
    import numpy as np
    import avengine.assets.roster_motion.episode as episode
    import avengine.assets.roster_motion.contact as contact
    import avengine.construction.joint as joint
    from avengine.construction.joint.families.qa09 import prepare_gait_geometry
    lineage = {"master_sha256": "test-master", "admitted_clip": "test-clip"}
    times = np.arange(901) / 30
    distance = final_clock()
    captures = []
    class RecordedClock:
        def __init__(self, d, b, action, phases, positions, yaws, *args):
            captures.append((np.asarray(phases), np.asarray(positions), np.asarray(yaws)))
        def frame(self, index):
            return None, None, None
    monkeypatch.setattr(episode, "admitted_master", lambda *a: ({}, b"", {}, lineage))
    monkeypatch.setattr(episode, "triangle_mesh", lambda *a: (np.zeros((3, 3)), np.array([[0, 1, 2]])))
    monkeypatch.setattr(contact, "FootPlant", RecordedClock)
    monkeypatch.setattr(joint, "admitted_human_clock_source", lambda *a: {
        "source_lineage": lineage, "source_times_s": np.array([0., 1.])})
    monkeypatch.setattr(joint, "complete_human_gait_clock", lambda *a: {
        "times_s": times, "distance_m": distance, "phase_s": times, "certificate": {}})
    asset = {"asset_id": "target", "habitat_glb": "unused"}
    runtime = {"joint_cache_directory": str(tmp_path), "qa09_arc_radius_m": .5}
    positive = prepare_gait_geometry(asset, runtime, {}, "yes")
    negative = prepare_gait_geometry(asset, dict(runtime, qa09_arc_turn_sign=-1), {}, "yes")
    cached = prepare_gait_geometry(asset, runtime, {}, "yes")
    assert len(captures) == 2  # Opposite direction must not load the forward pose cache.
    assert len(list(tmp_path.glob("qa09_gait_*.npz"))) == 2
    assert negative["metadata"]["turn_sign"] == -1
    assert positive["metadata"] == cached["metadata"]
    np.testing.assert_array_equal(captures[0][0], captures[1][0])
    np.testing.assert_array_equal(negative["distance"], positive["distance"])
    np.testing.assert_allclose(captures[1][1][:, 0], captures[0][1][:, 0])
    np.testing.assert_allclose(captures[1][1][:, 2], -captures[0][1][:, 2])
    np.testing.assert_allclose(captures[1][2] + captures[0][2], -np.pi)


def test_query_times_are_derived_from_the_solved_event_not_a_fixed_stop():
    demand=ReappearanceDemand("a","b","yes",window_start_s=20.)
    assert demand.relation([22.2,22.4])["query_times_s"]==[22.2,22.4]
    assert "foreground_depth_fraction" not in demand.relation()


def test_independent_late_window_keeps_local_mask_indices():
    f,o=sequence()
    result=certify(f,o,window_start_s=20.)
    assert result["status"]=="pass"
    assert result["demand"]["window_start_s"]==20.


@pytest.mark.parametrize("answer,offset",[("yes",0.),("no",20.)])
def test_endpoint_clock_reaches_requested_shadow_and_preserves_owner_speed(answer,offset):
    import numpy as np
    from types import SimpleNamespace
    from avengine.construction.joint import BodySpec
    from avengine.construction.joint.families.qa09 import endpoint_event_route
    class Floor:
        floor_reference=0.
        def contains_many(self,points,species):return np.ones(len(points),bool)
        def segment_free(self,a,b,species):return True
    class Nav:
        def snap_point(self,p):return np.asarray(p)
    ctx=SimpleNamespace(walkable_map=Floor(),pathfinder=Nav(),allow_native_map_fallback=False)
    a=np.array([0.,0.,0.]);h=np.array([1.7,0.,.2]);b=np.array([3.3,0.,-.2]) if answer=="yes" else h
    actor=endpoint_event_route(ctx,a,h,b,entity_id="target",body=BodySpec(),answer=answer,window_start_s=offset)
    np.testing.assert_allclose(actor.positions_m[0],a)
    np.testing.assert_allclose(actor.positions_m[-1],b)
    lo,hi=actor.metadata["qa09_hidden_interval_s"]
    assert offset<lo<hi<=offset+10.
    np.testing.assert_allclose(actor.positions_m[round((lo+hi)/2*30)],h)
    velocity=np.linalg.norm(np.diff(actor.positions_m[:,[0,2]],axis=0),axis=1)*30
    assert velocity.max()<=1.4+1e-6
    # Native roots are float32; the original strict speed certifier must still
    # accept their measured clock, including the lower-bound regression.
    from avengine.timeline.v7_motion_speed import certify_moving_root_speed
    native=(actor.positions_m+np.array([3.225,4.239589,-5.775])).astype(np.float32).astype(float)
    moving=np.r_[velocity>1e-8,False]
    report=certify_moving_root_speed(native,np.arange(901)/30,moving,"human")
    assert report["status"]=="pass"
    assert .8<actor.metadata["speed_m_s"]<1.4
    if offset:np.testing.assert_allclose(actor.positions_m[:int(offset*30)],np.tile(a,(int(offset*30),1)))


def test_geometry_budget_never_certifies_an_incomplete_prediction(monkeypatch):
    from types import SimpleNamespace
    import time
    from avengine.construction.joint.families.qa09 import QA09Plugin
    monkeypatch.setattr(time,"monotonic",lambda:2.)
    plugin=object.__new__(QA09Plugin)
    context=SimpleNamespace(depth=lambda camera:None)
    assert plugin._predict(context,SimpleNamespace(camera=None),ReappearanceDemand("a","b","yes"),{},deadline=1.) is None


def test_second_event_budget_failure_preserves_certified_first_event():
    import numpy as np
    from types import SimpleNamespace
    from avengine.construction.joint import BodySpec,CameraPose
    from avengine.construction.joint.families.qa09 import QA09Plugin
    plugin=object.__new__(QA09Plugin)
    plugin.foreground={"asset_id":"fore"};plugin.target={"asset_id":"a"};plugin.second={"asset_id":"b"}
    plugin.bodies={"fore":BodySpec()};plugin.geometry_budget_s=30.;plugin.seed=1
    plugin.diagnostics={"multi_event_attempts":0}
    first=SimpleNamespace(camera=CameraPose((0.,1.25,0.),0.),metadata={})
    calls=[]
    def events(*args,existing=None,**kwargs):
        calls.append((args[3]["asset_id"],args[4],args[5],existing is not None))
        if existing is None:yield first
    plugin._events=events
    context=SimpleNamespace(points_m=np.array([[0.,0.,-3.]]),
        walkable_map=SimpleNamespace(contains_many=lambda *a:np.array([True])),
        solve_cameras=lambda roles:SimpleNamespace(solutions=[SimpleNamespace(camera=first.camera)],diagnostics={}))
    stream=plugin.candidates(context,SimpleNamespace(branch="yes"))
    assert next(stream) is first
    assert calls==[("a","yes",0.,False),("b","no",20.,True)]
    assert "second event" in first.metadata["multi_event_downgrade"]
    stream.close()


@pytest.mark.parametrize("fraction",[.43,.88])
def test_native_ranking_compatibility_uses_solved_geometry_without_fixed_ratio(fraction):
    import numpy as np
    from avengine.construction.joint import CameraPose,ScenePlan,static_actor
    from avengine.construction.joint.families.qa09 import solved_occlusion_relation
    from avengine.capture.v7_dynamic_occlusion_proposals import dynamic_proposal_score
    target=static_actor("target",[0.,0.,-10.])
    fore=static_actor("fore",[0.,0.,-10.*fraction])
    plan=ScenePlan(CameraPose((0.,1.25,0.),0.),{"target":target,"fore":fore})
    states=["visible_clear"]*150;states[60]="fully_occluded"
    demand=ReappearanceDemand("target","fore","yes",window_start_s=20.)
    relation=solved_occlusion_relation(demand,plan,states)
    assert relation["query_times_s"]==[24.]
    assert relation["foreground_depth_fraction"]==pytest.approx(fraction)
    assert relation["preferred_lateral_m"]==0.
    native_target={"declaration":{"asset_id":"target"},
                   "motion_episode":{"world_roots_m":target.positions_m.tolist()}}
    scene={"listener_m":plan.camera.position_m,"native_dynamic_occlusion_relations":[relation]}
    result=dynamic_proposal_score(scene,{"declaration":{"asset_id":"fore"}},fore.positions_m[0],[native_target])
    assert result[1:]==pytest.approx((0.,0.))
