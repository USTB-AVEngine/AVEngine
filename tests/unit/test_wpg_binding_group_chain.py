"""WP-G: the core-group recipe on the unified question chain.

Six things this file pins down, each one of them something that was wrong or
missing when the group recipe was pointed at a request from the new chain:

* a request that pins an asset per entity instance has those pins moved with
  the selected asset order, so the compiled condition profile and the planned
  actors cannot disagree about what a slot holds;
* which registered assets may take part in a controlled visual swap is read
  from the registry body fields, not from asset names;
* the members of one group are measured to plan the same picture, and the
  measurement notices a per-frame channel that could show who is speaking;
* a question's group recipe says what its answer is and which interventions
  move it, so the recipe is not one hard-coded question type;
* comparisons are derived from the members' declared interventions and their
  predicted answers, including the invariance rows an answer-preserving
  intervention will need;
* a question type whose builder does not exist yet is refused by name rather
  than silently replaced.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.dataset import binding_group_native as native
from avengine.qa import binding_conditions as conditions


BODY = {
    "timeline": {"template_id": "body_one", "body_plan_id": "biped_human",
                 "walk_phase_period_frames": 16, "idle_action_id": "idle",
                 "walking_action_id": "walk",
                 "local_anatomical_forward_axis": [0.0, 0.0, 1.0]},
    "geometry": {"rig_authority": "rig_v3"},
    "emitter_anchors": [{"anchor_id": "mouth", "offset_m": [0.0, 1.6, 0.0]}],
    "default_emitter_anchor_id": "mouth",
    "runtime_backends": {"spear_unreal": {"actor_scale": 1.0},
                         "habitat": {"asset_kind": "articulated_m2_package"}},
}


def _asset(asset_id, colour, *, body=None, anchor=None):
    record = json.loads(json.dumps(body or BODY))
    if anchor is not None:
        record["emitter_anchors"][0]["offset_m"] = anchor
    record.update(asset_id=asset_id, entity_class="articulated_human",
                  identity={"species_id": "human"},
                  realized_attributes={"top_color": colour})
    return record


REGISTRY = {"assets": [
    _asset("one_blue", "blue"),
    _asset("one_green", "green"),
    _asset("one_burgundy", "burgundy"),
    _asset("one_pink", "pink"),
    _asset("other_yellow", "yellow", anchor=[0.0, 1.5, 0.0]),
]}


def _request():
    instances = [
        {"asset_id": "one_blue", "instance_id": "human_a", "role": "anchor",
         "source_class": "articulated_human", "speaking": True},
        {"asset_id": "one_green", "instance_id": "human_b", "role": "competitor",
         "source_class": "articulated_human", "speaking": True},
    ]
    return {
        "room_id": "room_x", "qa_ids": ["QA-20"],
        "source_asset_ids": ["one_blue", "one_green"],
        "entities": {"total_count": 2, "silent_count": 0,
                     "instances": json.loads(json.dumps(instances))},
        "entity_instances": json.loads(json.dumps(instances)),
        "qa_targets": [{"qa_id": "QA-20", "target_instance_ids": ["human_a"],
                        "branch": "visible_candidate"}],
    }


def _instance_assets(request, owner):
    rows = request[owner]["instances"] if owner == "entities" else request[owner]
    return [row["asset_id"] for row in rows]


def test_swapping_the_selection_moves_the_declared_instance_assets():
    swapped = native.build_variant_request(
        _request(), episode_id="g_v1", source_asset_ids=("one_green", "one_blue"))
    assert swapped["source_asset_ids"] == ["one_green", "one_blue"]
    # Both places the request pins an instance move together; leaving either
    # behind is what made the two variants compare as different worlds.
    assert _instance_assets(swapped, "entity_instances") == ["one_green", "one_blue"]
    assert _instance_assets(swapped, "entities") == ["one_green", "one_blue"]
    moved = swapped["binding_variant"]["declared_instance_assets"]
    assert moved["status"] == "applied"
    assert {row["instance_id"] for row in moved["moved"]} == {"human_a", "human_b"}


def test_an_unswapped_variant_leaves_the_declared_instances_alone():
    same = native.build_variant_request(
        _request(), episode_id="g_v0", source_asset_ids=("one_blue", "one_green"))
    assert _instance_assets(same, "entity_instances") == ["one_blue", "one_green"]
    assert same["binding_variant"]["declared_instance_assets"]["moved"] == []


def test_a_request_without_instance_pins_still_builds():
    request = _request()
    request.pop("entity_instances")
    request["entities"] = {"total_count": 2, "silent_count": 0}
    swapped = native.build_variant_request(
        request, episode_id="g_v1", source_asset_ids=("one_green", "one_blue"))
    assert swapped["binding_variant"]["declared_instance_assets"]["status"] == (
        "no_declared_instance_assets")


def test_a_controlled_swap_needs_one_body_and_two_colour_families():
    picked = native.select_controlled_swap_assets(REGISTRY)
    assert len(picked["asset_ids"]) == 2
    keys = {native.controlled_swap_body_key(record) for record in REGISTRY["assets"]
            if record["asset_id"] in picked["asset_ids"]}
    assert len(keys) == 1
    families = set(picked["appearance_families"].values())
    assert len(families) == 2


def test_a_swap_across_two_bodies_is_refused():
    with pytest.raises(native.BindingNativeError, match="one body"):
        native.select_controlled_swap_assets(
            REGISTRY, prefer=["one_blue", "other_yellow"])


def test_a_swap_inside_one_colour_family_is_refused():
    # burgundy and pink are a shade and a tint of one hue, so a viewer cannot
    # tell the two members apart by looking.
    with pytest.raises(native.BindingNativeError, match="colour family"):
        native.select_controlled_swap_assets(
            REGISTRY, prefer=["one_burgundy", "one_pink"])


def _plan(tmp_path, name, *, slot_assets, events, extra_state=None):
    frames = []
    for index in range(3):
        states = []
        for slot, asset_id in slot_assets.items():
            state = {"actor_id": slot, "entity_instance_id": f"inst_{slot}",
                     "frame_index": index, "moving": False, "action_id": "idle",
                     "action_phase": 0.0, "action_time_ticks": 0,
                     "root_transform": {"translation_m": [0.0, 0.0, float(index)]}}
            state.update(extra_state or {})
            states.append(state)
        frames.append({"frame_index": index, "pts_ticks": index * 10,
                       "camera_state": {"position_m": [0.0, 1.5, 0.0]},
                       "actor_states": states})
    plan = {
        "clock": {"frame_count": 3, "frame_rate_hz": 15.0},
        "scene": {"room_id": "room_x"},
        "condition_profile": {"instances": [
            {"entity_instance_id": f"inst_{slot}", "asset_id": asset_id,
             "source_slot_id": slot}
            for slot, asset_id in slot_assets.items()]},
        "camera_condition_sampling": {},
        "request": {"sampling_candidate_index": 0},
        "visual_plan": {
            "camera": {"resolution_hw": [720, 1280]},
            "actors": [{"actor_id": slot, "entity_instance_id": f"inst_{slot}",
                        "asset_id": asset_id}
                       for slot, asset_id in slot_assets.items()],
            "frames": frames,
        },
        "audio_events": [
            {"event_id": event_id, "actor_id": slot, "start_sample": start,
             "end_sample": start + 100, "sound_asset_id": f"clip_{event_id}"}
            for event_id, (slot, start) in events.items()
        ],
        "audio_assignment_targets": {event_id: slot
                                     for event_id, (slot, _s) in events.items()},
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _four_plans(tmp_path):
    visual = {"v0": {"source1": "one_blue", "source2": "one_green"},
              "v1": {"source1": "one_green", "source2": "one_blue"}}
    audio = {"a0": {"event_001": ("source1", 100), "event_002": ("source2", 900)},
             "a1": {"event_001": ("source2", 100), "event_002": ("source1", 900)}}
    return {f"{v}_{a}": _plan(tmp_path, f"{v}_{a}", slot_assets=visual[v], events=audio[a])
            for v in visual for a in audio}


def test_every_member_of_a_group_plans_the_same_picture(tmp_path):
    measured = native.measure_group_visual_invariance(_four_plans(tmp_path))
    assert measured["status"] == "pass"
    assert measured["frames_compared_per_member"] == 3
    assert measured["actor_states_compared_per_member"] == 6
    assert measured["differing_entries"] == []
    assert "moving" in measured["motion_fields_present"]
    assert measured["speech_animation_channels_found"] == []


def test_a_per_frame_speech_channel_is_reported(tmp_path):
    plans = _four_plans(tmp_path)
    leaking = _plan(tmp_path, "leaking",
                    slot_assets={"source1": "one_blue", "source2": "one_green"},
                    events={"event_001": ("source1", 100), "event_002": ("source2", 900)},
                    extra_state={"viseme_weight": 0.0})
    plans["v0_a0"] = leaking
    measured = native.measure_group_visual_invariance(plans)
    assert measured["status"] == "fail"
    assert any("viseme" in name for name in measured["speech_animation_channels_found"])


def test_a_moved_actor_is_a_difference(tmp_path):
    plans = _four_plans(tmp_path)
    moved = json.loads(Path(plans["v1_a0"]).read_text())
    moved["visual_plan"]["frames"][1]["actor_states"][0]["root_transform"][
        "translation_m"] = [9.0, 0.0, 1.0]
    Path(plans["v1_a0"]).write_text(json.dumps(moved), encoding="utf-8")
    measured = native.measure_group_visual_invariance(plans)
    assert measured["status"] == "fail"
    assert measured["differing_entries"]


def test_world_bindings_come_from_the_member_plan(tmp_path):
    path = _plan(tmp_path, "one", slot_assets={"source1": "one_blue", "source2": "one_green"},
                 events={"event_002": ("source2", 900), "event_001": ("source1", 100)})
    bindings = native.member_world_bindings(json.loads(path.read_text()), registry=REGISTRY)
    assert bindings["slot_appearances"] == {"source1": "blue", "source2": "green"}
    # Ordered by when the event actually starts, not by how the plan listed it.
    assert bindings["event_order"] == ["event_001", "event_002"]
    assert bindings["event_slots"]["event_001"] == "source1"


def _members():
    visual = {"v0": {"source1": "blue", "source2": "green"},
              "v1": {"source1": "green", "source2": "blue"}}
    audio = {"a0": {"event_001": "source1", "event_002": "source2"},
             "a1": {"event_001": "source2", "event_002": "source1"}}
    return [
        {"member_id": f"{v}_{a}",
         "factor_levels": {"visual_appearance_slots": v,
                           "audio_event_slot_assignment": a},
         "bindings": {"slot_appearances": visual[v], "event_slots": audio[a],
                      "event_order": ["event_001", "event_002"]}}
        for v in visual for a in audio
    ]


def test_comparisons_are_derived_from_the_declared_interventions():
    rows = conditions.derive_group_comparisons(_members(), qa_id="QA-20")
    necessity = [row for row in rows if row["kind"] == "necessity"]
    invariance = [row for row in rows if row["kind"] == "invariance"]
    assert len(necessity) == 4 and len(invariance) == 2
    assert {row["shared_modality"] for row in necessity} == {"audio", "video"}
    assert all(row["answer_relation"] == "different" for row in necessity)
    assert all(row["answer_relation"] == "same" for row in invariance)
    # Every member is one endpoint of an audio-shared and a video-shared row.
    for member_id in (member["member_id"] for member in _members()):
        shared = {row["shared_modality"] for row in necessity
                  if member_id in row["members"]}
        assert shared == {"audio", "video"}


def test_an_answer_preserving_intervention_becomes_an_invariance_row():
    # Nothing produces this member yet; the schema and the derivation accept it
    # so that adding one later is a producer change and not a format change.
    members = _members()[:2]
    control = json.loads(json.dumps(members[0]))
    control["member_id"] = "v0_a0_clip2"
    control["factor_levels"]["queried_clip_within_class"] = "c1"
    for member in members:
        member["factor_levels"]["queried_clip_within_class"] = "c0"
    rows = conditions.derive_group_comparisons(members + [control], qa_id="QA-20")
    control_rows = [row for row in rows if "v0_a0_clip2" in row["members"]]
    assert any(row["kind"] == "invariance"
               and row["intervention_type"] == "answer_irrelevant_change"
               for row in control_rows)


def test_a_declared_answer_changing_factor_that_does_not_change_the_answer_stops_the_build():
    members = _members()
    for member in members:
        member["bindings"]["slot_appearances"] = {"source1": "blue", "source2": "blue"}
    with pytest.raises(conditions.BindingConditionError, match="answer-changing"):
        conditions.derive_group_comparisons(members, qa_id="QA-20")


def test_each_member_states_what_was_done_to_it_and_what_it_should_do():
    record = conditions.member_intervention_record(
        {"visual_appearance_slots": "v1", "audio_event_slot_assignment": "a0"},
        qa_id="QA-20")
    assert record["factor_levels"]["visual_appearance_slots"] == "v1"
    assert {row["expected_answer_relation"] for row in record["applied"]} == {"different"}
    assert record["answer_relevance"]["visual_appearance_slots"] == "answer_changing"
    control = conditions.member_intervention_record(
        {"queried_clip_within_class": "c1"}, qa_id="QA-20")
    assert control["applied"][0]["expected_answer_relation"] == "same"
    assert control["answer_relevance"]["queried_clip_within_class"] == "answer_preserving"


def test_the_group_question_is_a_recipe_not_a_constant():
    recipe = conditions.group_question_recipe("QA-20")
    assert recipe["task_family"] == "visible_binding"
    assert set(recipe["flipped_by"]) == {"visual_appearance_slots",
                                         "audio_event_slot_assignment"}
    assert conditions.group_question_recipe("QA-19")["answer_type"] == "time_range_s"
    with pytest.raises(conditions.BindingConditionError, match="no core-group recipe"):
        conditions.group_question_recipe("QA-99")


def test_a_declared_question_without_a_builder_is_refused_by_name():
    with pytest.raises(conditions.BindingConditionError, match="no builder yet"):
        conditions.implemented_group_question_recipe("QA-19")
    assert conditions.implemented_group_question_recipe("QA-20")["qa_id"] == "QA-20"


def test_a_request_naming_two_questions_cannot_be_one_group():
    request = _request()
    request["qa_targets"].append({"qa_id": "QA-02", "target_instance_ids": ["human_b"]})
    with pytest.raises(native.BindingNativeError, match="one question"):
        native._requested_group_question(request)


def _review(status_by_actor):
    return {"actors": {
        actor: {"status": status, "value": value,
                "checks": [{"status": "pass" if status == "reviewed" else "not_observable",
                            "observed_value": observed,
                            "reason": None if status == "reviewed" else "target_is_too_dark"}
                           for _ in range(3)]}
        for actor, (status, value, observed) in status_by_actor.items()}}


def test_an_appearance_review_is_read_in_either_shape():
    pack = _review({"source1": ("reviewed", "blue", "blue"),
                    "source2": ("reviewed", "green", "green")})
    rows, unreviewed = native.appearance_review_rows(pack, ["source1", "source2"])
    assert unreviewed == []
    assert rows["source1"]["registered_value"] == "blue"
    assert rows["source1"]["checked_frames"] == 3
    # The delivered facts hoist the same rows to the top level.
    hoisted, hoisted_unreviewed = native.appearance_review_rows(
        pack["actors"], ["source1", "source2"])
    assert (hoisted, hoisted_unreviewed) == (rows, unreviewed)


def test_a_candidate_nobody_can_name_is_reported_with_the_reason():
    pack = _review({"source1": ("not_observable", "blue", "black"),
                    "source2": ("reviewed", "green", "green")})
    rows, unreviewed = native.appearance_review_rows(pack, ["source1", "source2"])
    assert unreviewed == ["source1"]
    assert rows["source1"]["observed_values"] == ["black"]
    assert rows["source1"]["failing_frames"] == 3
    assert rows["source1"]["reasons"] == ["target_is_too_dark"]


def test_a_slot_missing_from_the_review_counts_as_unreviewed():
    pack = _review({"source2": ("reviewed", "green", "green")})
    rows, unreviewed = native.appearance_review_rows(pack, ["source1", "source2"])
    assert unreviewed == ["source1"]
    assert rows["source1"]["status"] is None
