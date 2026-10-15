"""QA-10 construction regressions; no replacement for native rendering."""
import pytest

from avengine.construction.joint.families.qa10 import (
    OcclusionDemand, certify_native_relation,
)

pytestmark = pytest.mark.fast_unit


def native_probe():
    return {"relations": [{"target_asset_id": "green_person", "occluder_asset_id": "white_speaker",
                           "native_state": "visible_occluded", "satisfies_request": True,
                           "occluder_instance_ids": ["3"], "unknown_occluder_pixels": 0,
                           "context_pixels": 2000,
                           "native_mask_metrics": {"occlusion_fraction": .35}}],
            "native_control_restoration": {k: 0 for k in
                ("root_max_error", "joint_max_error", "camera_max_error", "link_max_error")}}


def test_near_foreground_feet_are_rejected_before_render():
    from avengine.construction.joint import CameraPose, BodySpec, framing
    body = BodySpec("human", .4, 1.85, 0.)
    assert framing(CameraPose((0,1.25,0),0), [.2,0,-1.8], body)[0] == "edge"
    camera = CameraPose((0,1.1,0),0)
    assert framing(camera, [0,0,-3.1], BodySpec("slide",.4,1.,0.))[0] == "inside"
    assert framing(camera, [0,0,-4.2], body)[0] == "inside"
    assert framing(CameraPose((0,1.1,0),180), [0,0,-4.2], body)[0] == "outside"


def test_registered_low_object_uses_shared_partial_visibility_prediction():
    import numpy as np
    from avengine.construction.joint import CameraPose, BodySpec, DepthBuffer, predict_visibility
    camera = CameraPose((0,1.1,0),0,width=128,height=72)
    depth = DepthBuffer(camera,np.full((72,128),np.inf),None,"visual_mesh",{})
    prediction = predict_visibility(camera,[0,0,-4.2],BodySpec("human",.4,1.85,0.),
        depth=depth,occluders=(("speaker",[0,0,-3.1],BodySpec("slide",.3,.95,0.)),))
    assert prediction.state == "visible_occluded"
    assert prediction.occluder_ids == ("speaker",)
    assert .25 < prediction.occluded_fraction < .65
    assert prediction.visible_pixels_lower == 0  # proxy is not native truth


def test_native_identity_must_be_the_requested_unique_registered_occluder():
    demand = OcclusionDemand("green_person", "white_speaker")
    assert certify_native_relation(native_probe(), demand, "3")["status"] == "pass"
    for ids in (["2"], ["3", "4"], []):
        witness = native_probe(); witness["relations"][0]["occluder_instance_ids"] = ids
        assert certify_native_relation(witness, demand, "3")["status"] == "fail"
    witness = native_probe(); witness["relations"][0]["unknown_occluder_pixels"] = 1
    assert certify_native_relation(witness, demand, "3")["status"] == "fail"


@pytest.mark.parametrize("field,value", [("context_pixels", 99), ("native_state", "fully_occluded"),
                                        ("satisfies_request", False)])
def test_hidden_or_unreviewable_target_is_not_certified(field, value):
    witness = native_probe(); witness["relations"][0][field] = value
    assert certify_native_relation(witness, OcclusionDemand("green_person", "white_speaker"), "3")["status"] == "fail"


def test_native_fraction_and_restoration_have_to_pass_independently():
    for fraction in (.10, .9):
        witness = native_probe(); witness["relations"][0]["native_mask_metrics"]["occlusion_fraction"] = fraction
        assert certify_native_relation(witness, OcclusionDemand("green_person", "white_speaker"), "3")["status"] == "fail"
    witness = native_probe(); witness["native_control_restoration"]["camera_max_error"] = 1e-7
    assert certify_native_relation(witness, OcclusionDemand("green_person", "white_speaker"), "3")["status"] == "fail"


def row_inputs():
    from types import SimpleNamespace
    import numpy as np
    room = {"key": "train_house::room", "house": "train_house", "room_id": "room",
            "family": "hm3d", "categories": ["standing_human", "floor_object"]}
    house = {"house": "train_house", "checked_room_ids": ["room"], "checked_rooms": [
        {"room_label": "room", "measured_floor_y_m": 0.}]}
    people = [{"asset_id": aid, "kind": "human", "species_id": "human", "controlled_shirt": True,
               "attachment_surface": "floor", "color": colour, "visual_gender": "male",
               "allowed_classes": ["speech_playback"]}
              for aid, colour in (("target", "green"), ("companion", "blue"))]
    occluder = {"asset_id": "speaker", "kind": "object", "attachment_surface": "floor",
                "label_color": "white", "registered_label_color": "white",
                "object_type": "floorstanding_speaker", "geometry": {"dimensions_m": [.49, .95, .42]}}
    assets = [*people, occluder]
    contract = {"schema": "v7-production-inputs-v1", "rooms": [room], "test_houses": [], "policy": {}}
    semantic = {"sounds": [{
        "sound_asset_id": "whole", "scenario_id": "complete", "role_id": "one",
        "transcript": "Please put the package beside the door.",
        "synthesis_readback": {"duration_s": 1.5},
        "voice_reference": {"speaker_id": "male1", "gender": "male"}}]}
    positions = {"target": [0.,0.,-4.2], "speaker": [0.,0.,-3.1], "companion": [1.5,0.,-3.2]}
    pins = {aid: {"surface": "floor", "original_surface_point_m": p, "normal": [0.,1.,0.],
                  "floor_y": 0., "_outward_clearance_m": 0., "yaw_rad": 0.,
                  "joint_solved_pose": True, "joint_expected_root_m": p}
            for aid, p in positions.items()}
    plan = SimpleNamespace(
        metadata={"source_playback_starts_s": {"companion": [0., 10., 20.]},
                  "native_bindings": {"native_attachment_supports": pins,
                  "native_placement_preferences_m": positions, "joint_strict_placement": True}},
        camera=SimpleNamespace(position_m=[0., 1.1, 0.], yaw_deg=0.),
        sounds=tuple(SimpleNamespace(entity_id="companion", sound_id="whole", sound_class="speech_playback",
                                     start_s=t, end_s=t+1.5, wet_end_s=t+2.) for t in (0., 10., 20.)),
        actors={aid: SimpleNamespace(positions_m=np.repeat(np.array(p)[None], 901, axis=0))
                for aid, p in (("target", [0., 0., -4.2]), ("speaker", [0., 0., -3.1]),
                               ("companion", [1.5, 0., -3.2]))})
    return room, house, assets, semantic, contract, plan, SimpleNamespace(passed=True)


def test_solved_row_keeps_answer_identity_native_support_and_real_single_voice():
    from avengine.construction.joint.families.qa10 import build_qa10_row
    room, house, assets, semantic, contract, plan, cert = row_inputs()
    row = build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                         assets[0], assets[2], assets[1], scene_id="new", seed=8)["row"]
    assert row["silent_asset_ids"] == ["target", "speaker"]
    assert row["source_classes"] == {"companion": ["speech_playback"]}
    assert row["semantic_human_asset_ids"] == ["companion"]
    assert row["semantic_sound_ids"] == ["whole"]
    assert row["phase2_authored_speech_schedule_s"]["companion"] == [0., 10., 20.]
    assert row["native_occlusion_relations"][0]["target_asset_id"] == "target"
    assert row["native_occlusion_relations"][0]["occluder_asset_id"] == "speaker"
    assert row["native_attachment_supports"]["target"]["original_surface_point_m"] == [0., 0., -4.2]
    assert all(k["yaw_deg"] == 0 for k in row["camera_yaw_knots"])
    assert row["scene_demand"]["required_independent_family_counts"] == {"QA-10": 1}
    assert row["automatic_construction"]["native_identity_certification_required"]
    assert row["joint_prepared_scene_required"] is True


def test_row_translation_rejects_missing_roles_motion_and_colour_collisions():
    from avengine.construction.joint.families.qa10 import build_qa10_row
    room, house, assets, semantic, contract, plan, cert = row_inputs()
    plan.actors["target"].positions_m[10, 0] = .01
    with pytest.raises(ValueError, match="stationary"):
        build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                       assets[0], assets[2], assets[1], scene_id="bad", seed=8)
    room, house, assets, semantic, contract, plan, cert = row_inputs()
    assets[1]["color"] = "green"
    with pytest.raises(ValueError, match="distinct colour"):
        build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                       assets[0], assets[2], assets[1], scene_id="bad", seed=8)


def test_original_production_contract_still_excludes_test_houses():
    from avengine.construction.joint.families.qa10 import build_qa10_row
    room, house, assets, semantic, contract, plan, cert = row_inputs()
    contract["test_houses"] = ["train_house"]
    with pytest.raises(ValueError, match="test house"):
        build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                       assets[0], assets[2], assets[1], scene_id="bad", seed=8)


def test_support_site_keeps_the_walkable_maps_measured_floor_height():
    from avengine.construction.joint.families.qa10 import build_qa10_row
    room, house, assets, semantic, contract, plan, cert = row_inputs()
    plan.actors["target"].positions_m[:, 1] = .017
    plan.metadata["native_bindings"]["native_attachment_supports"]["target"]["original_surface_point_m"][1] = .017
    row = build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                         assets[0], assets[2], assets[1], scene_id="terrain", seed=8)["row"]
    assert row["native_attachment_supports"]["target"]["original_surface_point_m"][1] == .017
    assert row["native_attachment_supports"]["target"]["floor_y"] == 0.



def test_native_program_uses_the_core_solved_event_times():
    from avengine.construction.joint.families.qa10 import build_qa10_row
    room, house, assets, semantic, contract, plan, cert = row_inputs()
    for e in plan.sounds:
        e.start_s += 1.; e.end_s += 1.; e.wet_end_s += 1.
    plan.metadata["source_playback_starts_s"]["companion"] = [1., 11., 21.]
    row = build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                         assets[0], assets[2], assets[1], scene_id="clock", seed=8)["row"]
    assert row["phase2_authored_speech_schedule_s"] == {"companion": [1., 11., 21.]}
    assert row["semantic_start_offsets_s"] == [1.]



def test_measured_activity_does_not_repeat_the_whole_utterance_at_each_syllable():
    from types import SimpleNamespace
    from avengine.construction.joint.families.qa10 import build_qa10_row
    room, house, assets, semantic, contract, plan, cert = row_inputs()
    plan.sounds = tuple(SimpleNamespace(entity_id="companion", sound_id="whole",
        sound_class="speech_playback", start_s=t+a, end_s=t+b, wet_end_s=t+b+.5)
        for t in (0.,10.,20.) for a,b in ((.2,.7),(.9,1.4)))
    row = build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                         assets[0], assets[2], assets[1], scene_id="activity", seed=8)["row"]
    assert row["phase2_authored_speech_schedule_s"]["companion"] == [0.,10.,20.]


def test_native_binding_rejects_post_solver_position_changes():
    from avengine.construction.joint.families.qa10 import build_qa10_row
    room, house, assets, semantic, contract, plan, cert = row_inputs()
    plan.actors["target"].positions_m[:,0] += .1
    with pytest.raises(ValueError, match="binding poses differ"):
        build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                       assets[0], assets[2], assets[1], scene_id="changed", seed=8)



def test_solver_transfers_the_same_live_native_world_to_render():
    from types import SimpleNamespace
    from avengine.construction.joint import Certificate, CheckResult, PreparedCandidate
    from avengine.construction.joint.families.qa10 import finalize_qa10_solution
    provenance = {"solver_entry": "avengine.construction.joint.solve",
                  "geometry_certificate_status": "pass", "native_preflight_status": "pass"}
    row = {"automatic_construction": {}, "joint_solution": provenance}
    bound = []
    payload = SimpleNamespace(row=row, bind_solution=lambda record: bound.append(record))
    solved = SimpleNamespace(status="certified", prepared=PreparedCandidate(payload,(CheckResult("native","pass"),)),
                             diagnostics={"candidates":1},
                             certificate=Certificate((CheckResult("native", "pass"),)))
    result = finalize_qa10_solution(solved, SimpleNamespace(branch="visible_registered_partial"))
    assert result["prepared_native"] is payload
    assert result["row"] is row
    assert not bound  # the core's immutable binding must not be written twice
    assert result["row"]["joint_solution"] is provenance
    assert row["automatic_construction"]["joint_solver"]["native_execution_must_not_replan"]
