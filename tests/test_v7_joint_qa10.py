"""QA-10 construction regressions; no replacement for native rendering."""
import pytest

from avengine.construction.joint.families.qa10 import (
    OcclusionDemand, certify_native_relation, full_body_in_view,
    projected_lower_body_coverage,
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
    assert not full_body_in_view([0, 1.25, 0], 0, [.2, 0, -1.8])
    assert full_body_in_view([0, 1.10, 0], 0, [0, 0, -3.1], top_m=1.0)
    assert full_body_in_view([0, 1.10, 0], 0, [0, 0, -4.2])
    assert not full_body_in_view([0, 1.10, 0], 180, [0, 0, -4.2])


def test_registered_low_object_hides_legs_while_torso_can_be_reviewed():
    height = projected_lower_body_coverage(1.1, 4.2, 3.1, .95)
    assert .85 < height < .95
    with pytest.raises(ValueError, match="in front"):
        projected_lower_body_coverage(1.1, 3.1, 4.2, .95)


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
    plan = SimpleNamespace(
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
    row = build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                         assets[0], assets[2], assets[1], scene_id="terrain", seed=8)["row"]
    assert row["native_attachment_supports"]["target"]["original_surface_point_m"][1] == .017
    assert row["native_attachment_supports"]["target"]["floor_y"] == 0.



def test_native_program_uses_the_core_solved_event_times():
    from avengine.construction.joint.families.qa10 import build_qa10_row
    room, house, assets, semantic, contract, plan, cert = row_inputs()
    for e in plan.sounds:
        e.start_s += 1.; e.end_s += 1.; e.wet_end_s += 1.
    row = build_qa10_row(room, house, {}, assets, semantic, contract, plan, cert,
                         assets[0], assets[2], assets[1], scene_id="clock", seed=8)["row"]
    assert row["phase2_authored_speech_schedule_s"] == {"companion": [1., 11., 21.]}
    assert row["semantic_start_offsets_s"] == [1.]
