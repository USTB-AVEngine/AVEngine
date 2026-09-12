from copy import deepcopy

import pytest

from avengine.qa.binding_questions import (
    TASK_FAMILIES,
    TASK_QA_IDS,
    BindingQuestionError,
    generate_binding_question,
)
from avengine.qa.binding_conditions import task_family_requirements
from avengine.qa import binding_groups
from avengine.qa.binding_groups import BindingGroupError, align_question_forms, validate_group
from avengine.qa.unified_catalog import UNIFIED_FACT_SCHEMA
from avengine.qa.unified_scoring import score_unified_item


def _facts():
    colors = ["blue", "green", "pink"]
    actors = {
        f"a{i}": {
            "actor_id": f"a{i}", "asset_id": f"asset{i}", "species_id": "human",
            "display_label": f"{color} shirt",
            "appearance": {"field": "top_color", "value": color, "label": f"{color} shirt"},
            "root_positions_m": [[i - 1, 0, -3]] * 12,
            "emitter_positions_m": [[i - 1, 1, -3]] * 12,
        } for i, color in enumerate(colors)
    }
    events = []
    for i, (actor, start, end) in enumerate([("a0", 1, 2), ("a1", 4, 6), ("a2", 5, 5.5)]):
        activity = [{"start_sample": int(start * 100), "end_sample_exclusive": int(end * 100)}]
        events.append({
            "event_id": f"e{i}", "actor_id": actor, "sound_asset_id": f"sound{i}",
            "start_s": start, "end_s": end, "start_frame": int(start), "end_frame": int(end),
            "sound_class": "speech", "sound_class_explicit": True,
            "source_activity_intervals_samples": activity, "source_activity_evidence_status": "observed",
        })
    return {
        "schema": UNIFIED_FACT_SCHEMA, "status": "research_candidate", "episode_id": "fixture",
        "time": {"frame_count": 12, "frame_rate_hz": 1.0, "sample_rate_hz": 100,
                 "sample_count": 1200, "duration_seconds": 12.0, "time_base_hz": 100},
        "actors": actors, "events": events, "listener": {},
        "audio": {"status": "pass", "channel_count": 2, "wet_tail_intervals": [
            {"event_id": e["event_id"], "start_s": e["start_s"], "end_s": e["end_s"] + 0.25}
            for e in events]},
        "source_activity_evidence_present": True,
        "source_activity_intervals_samples": {e["event_id"]: e["source_activity_intervals_samples"] for e in events},
        "visibility": {
            actor: {frame: {"state": "visible_clear", "frame_index": frame,
                            "visible_fraction": 1.0, "visible_pixels": 600,
                            "target_centroid_xy_px": [30 + i * 20, 50]}
                    for frame in range(12)} for i, actor in enumerate(actors)},
        "visibility_meta": {"resolution_hw": [100, 100]},
        "camera_calibration": {"projection": "pinhole", "public": True, "fx_px": 50.0, "cx_px": 50.0,
                               "width_px": 100, "height_px": 100},
        "appearance_review": {actor: {"status": "reviewed", "value": colors[i], "frame_refs": list(range(12))}
                              for i, actor in enumerate(actors)},
        "sampling": {}, "input_summary": {},
    }


def test_visible_binding_uses_appearance_answer_not_private_actor_id():
    facts = _facts()
    first = generate_binding_question(facts, "visible_binding", {"event_number": 1})
    changed = deepcopy(facts)
    for actor in ("a0", "a1"):
        other = "a1" if actor == "a0" else "a0"
        changed["actors"][actor]["appearance"] = deepcopy(facts["actors"][other]["appearance"])
        changed["appearance_review"][actor] = deepcopy(facts["appearance_review"][other])
    second = generate_binding_question(changed, "visible_binding", {"event_number": 1})
    assert first["truth"]["value"] == "blue"
    assert second["truth"]["value"] == "green"
    assert first["evidence"]["target_actor_id"] == second["evidence"]["target_actor_id"]
    align_question_forms([first, second], "paired")
    assert first["model_input"] == second["model_input"]
    assert score_unified_item(first, "blue", form="open")["score"] == 1.0


def test_conditioned_overlap_selects_visual_subset_and_measured_activity():
    facts = _facts()
    query = {"appearance_values": ["green", "pink"], "window_s": [4, 6]}
    yes = generate_binding_question(facts, "visual_conditioned_relation", query)
    no = generate_binding_question(facts, "visual_conditioned_relation",
                                   {**query, "appearance_values": ["blue", "green"]})
    assert yes["truth"]["value"] == "yes"
    assert no["truth"]["value"] == "no"
    assert yes["qa_id"] == "QA-05"
    assert yes["required_modalities"] == ["audio", "video"]
    # Declared durations overlap; actual emissions do not after this change.
    facts["events"][2]["source_activity_intervals_samples"] = []
    with pytest.raises(BindingQuestionError, match="all visible pairs"):
        generate_binding_question(facts, "visual_conditioned_relation", query)


def test_conditioned_overlap_rejects_two_candidate_global_relation():
    facts = _facts()
    facts["visibility"]["a2"] = {frame: {"state": "out_of_view"} for frame in range(12)}
    with pytest.raises(BindingQuestionError, match="third candidate"):
        generate_binding_question(facts, "visual_conditioned_relation",
                                  {"appearance_values": ["blue", "green"], "window_s": [4, 6]})


def test_cross_event_identity_uses_persistent_instance_not_appearance_or_clip():
    facts = _facts()
    query = {"event_numbers": [1, 2]}
    before = generate_binding_question(facts, "cross_event_identity", query)
    recolored = deepcopy(facts)
    recolored["actors"]["a0"]["appearance"], recolored["actors"]["a1"]["appearance"] = (
        deepcopy(facts["actors"]["a1"]["appearance"]), deepcopy(facts["actors"]["a0"]["appearance"]))
    recolored["appearance_review"]["a0"], recolored["appearance_review"]["a1"] = (
        deepcopy(facts["appearance_review"]["a1"]), deepcopy(facts["appearance_review"]["a0"]))
    after_color = generate_binding_question(recolored, "cross_event_identity", query)
    assert before["truth"]["value"] == after_color["truth"]["value"] == "no"
    facts["events"][1]["actor_id"] = "a0"
    assert generate_binding_question(facts, "cross_event_identity", query)["truth"]["value"] == "yes"


def test_cross_event_identity_rejects_unobservable_history_and_tautology():
    facts = _facts()
    facts["visibility"]["a0"][3]["state"] = "fully_occluded"
    with pytest.raises(BindingQuestionError, match="observable"):
        generate_binding_question(facts, "cross_event_identity", {"event_numbers": [1, 2]})
    with pytest.raises(BindingQuestionError, match="tautologies"):
        generate_binding_question(_facts(), "cross_event_identity", {"event_numbers": [1, 1]})


def test_cross_time_bearing_exposes_calibration_and_integer_query_only():
    facts = _facts()
    facts["events"] = facts["events"][:1]
    facts["audio"]["wet_tail_intervals"] = facts["audio"]["wet_tail_intervals"][:1]
    facts["source_activity_intervals_samples"] = {"e0": facts["events"][0]["source_activity_intervals_samples"]}
    q = generate_binding_question(facts, "cross_time_state", {"event_number": 1, "query_time_s": 9})
    assert q["truth"]["value"] == -22
    assert q["evidence"]["observation_cutoff_s"] == 9
    assert q["model_input"]["open"]["camera_calibration"]["fx_px"] == 50
    assert "mcq" not in q["forms"]
    with pytest.raises(BindingQuestionError, match="integer"):
        generate_binding_question(_facts(), "cross_time_state", {"query_time_s": 9.5})


def _group():
    members = []
    for row in range(2):
        for column in range(2):
            facts = _facts()
            facts["events"][0]["actor_id"] = f"a{column}"
            if row:
                for actor in ("a0", "a1"):
                    other = "a1" if actor == "a0" else "a0"
                    original = _facts()
                    facts["actors"][actor]["appearance"] = deepcopy(original["actors"][other]["appearance"])
                    facts["appearance_review"][actor] = deepcopy(original["appearance_review"][other])
            members.append({"sample_id": f"s{row}{column}",
                            "question": generate_binding_question(facts, "visible_binding", {"event_number": 1}),
                            "media": {"video_path": "unused", "audio_path": "unused"}})
    align_question_forms([m["question"] for m in members], "group")
    comparisons = []
    for index in range(2):
        comparisons.append({"members": [f"s0{index}", f"s1{index}"], "shared_modality": "audio",
                            "answer_relation": "different", "kind": "necessity"})
        comparisons.append({"members": [f"s{index}0", f"s{index}1"], "shared_modality": "video",
                            "answer_relation": "different", "kind": "necessity"})
    return {"members": members, "comparisons": comparisons}


def test_structure_only_group_is_not_reported_as_media_verified(tmp_path):
    result = validate_group(_group(), base=tmp_path, verify_media=False)
    assert result["status"] == "structure_only"
    assert result["human_answerability"] == "not_run"


def test_group_rejects_single_direction_witness_and_question_leak(tmp_path):
    group = _group()
    group["comparisons"] = group["comparisons"][:2]
    with pytest.raises(BindingGroupError, match="every member"):
        validate_group(group, base=tmp_path, verify_media=False)
    group = _group()
    group["members"][0]["question"]["model_input"]["open"]["question_en"] += " left blue source"
    with pytest.raises(BindingGroupError, match="inputs differ"):
        validate_group(group, base=tmp_path, verify_media=False)


def test_group_rejects_missing_answer_change(tmp_path):
    group = _group()
    group["members"][0]["question"]["forms"]["open"]["truth"] = "green"
    with pytest.raises(BindingGroupError, match="not distinguishable"):
        validate_group(group, base=tmp_path, verify_media=False)


def test_binding_rejects_missing_activity_instead_of_treating_it_as_silence():
    facts = _facts()
    facts["events"][2]["source_activity_evidence_status"] = "missing"
    facts["events"][2]["source_activity_intervals_samples"] = []
    with pytest.raises(BindingQuestionError, match="activity"):
        generate_binding_question(facts, "visual_conditioned_relation", {
            "appearance_values": ["blue", "green"], "window_s": [4, 6],
        })


def test_cross_time_rejects_other_event_wet_tail_at_query():
    facts = _facts()
    facts["events"] = facts["events"][:2]
    facts["events"][1].update(start_s=0, end_s=0.5, start_frame=0, end_frame=0)
    facts["events"][1]["source_activity_intervals_samples"] = [{"start_sample": 0, "end_sample_exclusive": 50}]
    facts["audio"]["wet_tail_intervals"] = [
        {"event_id": "e0", "start_s": 1, "end_s": 2.25},
        {"event_id": "e1", "start_s": 0, "end_s": 9.5},
    ]
    with pytest.raises(BindingQuestionError, match="wet"):
        generate_binding_question(facts, "cross_time_state", {"event_number": 2, "query_time_s": 9})


def test_group_rejects_duplicate_edges_and_unknown_relation_kind(tmp_path):
    group = _group()
    group["comparisons"].append(deepcopy(group["comparisons"][0]))
    with pytest.raises(BindingGroupError, match="duplicate"):
        validate_group(group, base=tmp_path, verify_media=False)
    group = _group()
    group["comparisons"][0]["kind"] = "unknown"
    with pytest.raises(BindingGroupError, match="kind"):
        validate_group(group, base=tmp_path, verify_media=False)


def test_cross_time_clip_end_uses_actual_last_frame_and_full_observation():
    facts = _facts()
    facts["events"] = facts["events"][:1]
    facts["audio"]["wet_tail_intervals"] = facts["audio"]["wet_tail_intervals"][:1]
    question = generate_binding_question(facts, "cross_time_state", {"event_number": 1, "query_anchor": "clip_end"})
    assert question["evidence"]["query_frame"] == facts["time"]["frame_count"] - 1
    assert question["evidence"]["observation_cutoff_s"] is None
    assert "片尾" in question["forms"]["open"]["question_zh"]
    with pytest.raises(BindingQuestionError, match="either"):
        generate_binding_question(facts, "cross_time_state", {"query_anchor": "clip_end", "query_time_s": 9})


def test_cross_time_rejects_hidden_identity_between_sound_and_endpoint():
    facts = _facts()
    facts["events"] = facts["events"][:1]
    facts["audio"]["wet_tail_intervals"] = facts["audio"]["wet_tail_intervals"][:1]
    facts["visibility"]["a0"][8]["state"] = "fully_occluded"
    with pytest.raises(BindingQuestionError, match="identity history"):
        generate_binding_question(facts, "cross_time_state", {"query_anchor": "clip_end"})


def test_relation_counts_visible_unnamed_third_candidate():
    facts = _facts()
    facts["appearance_review"]["a2"]["status"] = "not_observable"
    question = generate_binding_question(facts, "visual_conditioned_relation", {
        "appearance_values": ["blue", "green"], "window_s": [4, 6]})
    assert question["truth"]["value"] == "no"
    assert set(question["evidence"]["candidate_actor_ids"]) == {"a0", "a1", "a2"}
    facts["appearance_review"]["a1"]["status"] = "not_observable"
    with pytest.raises(BindingQuestionError, match="uniquely visible"):
        generate_binding_question(facts, "visual_conditioned_relation", {
            "appearance_values": ["blue", "green"], "window_s": [4, 6]})


def test_cross_time_pixel_bearing_does_not_require_color_naming():
    facts = _facts()
    facts["events"] = facts["events"][:1]
    facts["audio"]["wet_tail_intervals"] = facts["audio"]["wet_tail_intervals"][:1]
    for value in facts["appearance_review"].values():
        value["status"] = "not_observable"
    question = generate_binding_question(facts, "cross_time_state", {"query_anchor": "clip_end"})
    assert question["truth"]["value"] == -22


def test_reference_geometry_selects_pair_without_color_names():
    facts = _facts()
    facts["actors"]["a3"] = deepcopy(facts["actors"]["a0"])
    facts["actors"]["a3"]["actor_id"] = "a3"
    facts["visibility"]["a3"] = deepcopy(facts["visibility"]["a0"])
    centers = {"a0": [40, 10], "a1": [40, 50], "a2": [40, 90], "a3": [5, 10]}
    for actor, center in centers.items():
        for row in facts["visibility"][actor].values():
            row["visible_centroid_xy_px"] = center
    facts["appearance_review"] = {}
    query = {"reference_time_s": 0, "window_s": [4, 6], "visual_selector": {
        "kind": "two_nearest_to_leftmost", "minimum_margin_px": 10,
        "candidate_scope_en": "the four people", "candidate_scope_zh": "画面中的四个人"}}
    first = generate_binding_question(facts, "visual_conditioned_relation", query)
    assert first["truth"]["value"] == "no"
    assert set(first["evidence"]["selected_actor_ids"]) == {"a0", "a1"}
    facts["visibility"]["a3"][0]["visible_centroid_xy_px"] = [5, 90]
    second = generate_binding_question(facts, "visual_conditioned_relation", query)
    assert second["truth"]["value"] == "yes"
    assert set(second["evidence"]["selected_actor_ids"]) == {"a1", "a2"}
    assert first["model_input"] == second["model_input"]
    facts["visibility"]["a3"][0]["visible_centroid_xy_px"] = [5, 50]
    with pytest.raises(BindingQuestionError, match="nearest pair"):
        generate_binding_question(facts, "visual_conditioned_relation", query)
    bad = deepcopy(query)
    del bad["visual_selector"]["candidate_scope_zh"]
    with pytest.raises(BindingQuestionError, match="candidate scope"):
        generate_binding_question(facts, "visual_conditioned_relation", bad)

def test_named_reference_can_switch_neighbors_without_being_leftmost():
    facts = _facts()
    facts["actors"]["a0"]["appearance"] = {"field": "top_color", "value": "yellow", "label": "yellow shirt"}
    facts["appearance_review"]["a0"]["value"] = "yellow"
    facts["actors"]["a3"] = deepcopy(facts["actors"]["a0"])
    facts["actors"]["a3"].update(actor_id="a3", asset_id="asset3")
    facts["actors"]["a3"]["appearance"] = {"field": "top_color", "value": "blue", "label": "blue shirt"}
    facts["appearance_review"]["a3"] = {"status": "reviewed", "value": "blue", "frame_refs": [0]}
    facts["visibility"]["a3"] = deepcopy(facts["visibility"]["a0"])
    centers = {"a0": [20, 50], "a1": [50, 50], "a2": [80, 50], "a3": [40, 30]}
    for actor, center in centers.items():
        for row in facts["visibility"][actor].values():
            row["visible_centroid_xy_px"] = center
    query = {"reference_time_s": 0, "window_s": [4, 6], "visual_selector": {
        "kind": "two_nearest_to_named_reference",
        "reference_appearance": {"field": "top_color", "value": "blue"},
        "minimum_margin_px": 10,
        "candidate_scope_en": "the four people", "candidate_scope_zh": "画面中的四个人"}}
    first = generate_binding_question(facts, "visual_conditioned_relation", query)
    assert first["truth"]["value"] == "no"
    assert set(first["evidence"]["selected_actor_ids"]) == {"a0", "a1"}
    assert first["evidence"]["visual_selection"]["reference_actor_id"] == "a3"
    facts["visibility"]["a3"][0]["visible_centroid_xy_px"] = [60, 30]
    second = generate_binding_question(facts, "visual_conditioned_relation", query)
    assert second["truth"]["value"] == "yes"
    assert set(second["evidence"]["selected_actor_ids"]) == {"a1", "a2"}
    assert first["model_input"] == second["model_input"]
    facts["appearance_review"]["a3"]["status"] = "not_observable"
    with pytest.raises(BindingQuestionError, match="named reference appearance"):
        generate_binding_question(facts, "visual_conditioned_relation", query)


# --------------------------------------------------------------------------- conditions


def _single_event_facts():
    facts = _facts()
    facts["events"] = facts["events"][:1]
    facts["audio"]["wet_tail_intervals"] = facts["audio"]["wet_tail_intervals"][:1]
    facts["source_activity_intervals_samples"] = {
        "e0": facts["events"][0]["source_activity_intervals_samples"]}
    return facts


_QUERIES = {
    "visible_binding": (lambda: _facts(), {"event_number": 1}),
    "visual_conditioned_relation": (
        lambda: _facts(), {"appearance_values": ["green", "pink"], "window_s": [4, 6]}),
    "cross_event_identity": (lambda: _facts(), {"event_numbers": [1, 2]}),
    "cross_time_state": (_single_event_facts, {"event_number": 1, "query_anchor": "clip_end"}),
}


@pytest.mark.parametrize("family", TASK_FAMILIES)
def test_every_family_records_a_reading_for_each_declared_condition(family):
    build, query = _QUERIES[family]
    item = generate_binding_question(build(), family, query)
    conditions = item["binding_conditions"]
    declared = task_family_requirements(family)
    assert item["qa_id"] == TASK_QA_IDS[family] == conditions["qa_id"]
    assert [row["key"] for row in conditions["question_conditions"]] == [
        row["key"] for row in declared["question_conditions"]]
    for row in conditions["question_conditions"]:
        assert row["status"] == "pass"
        assert isinstance(row["reading"], dict) and row["reading"]
    assert [row["key"] for row in conditions["recipe_conditions"]] == [
        row["key"] for row in declared["recipe_conditions"]]
    assert conditions["core_group_valid"] == "not_run"
    assert conditions["dual_modality_necessity"] == "not_run"
    assert conditions["human_answerability"] == "not_run"
    assert item["modality_necessity"]["status"] == "not_run"


def test_condition_readings_quote_the_actual_episode_not_a_template():
    item = generate_binding_question(_facts(), "visible_binding", {"event_number": 1})
    readings = {row["key"]: row["reading"] for row in
                item["binding_conditions"]["question_conditions"]}
    anchor = readings["visible_active_anchor"]
    assert anchor["anchor_frame"] == item["evidence"]["anchor_frame"]
    assert anchor["emitter_actor_id"] == item["evidence"]["target_actor_id"]
    assert item["evidence"]["event_id"] in anchor["active_event_ids"]
    assert readings["identified_sound_event"]["coincident_onset_count"] == 1
    assert readings["distinct_reviewed_candidates"]["appearance_values"] == sorted(
        set(readings["distinct_reviewed_candidates"]["appearance_values"]))


def test_cross_time_state_reading_comes_from_the_catalog_silence_predicate():
    item = generate_binding_question(
        _single_event_facts(), "cross_time_state", {"event_number": 1, "query_anchor": "clip_end"})
    readings = {row["key"]: row["reading"] for row in
                item["binding_conditions"]["question_conditions"]}
    judges = {row["key"]: row["judge"] for row in
              item["binding_conditions"]["question_conditions"]}
    post_sound = readings["query_after_event_and_measured_wet_tail"]
    assert judges["query_after_event_and_measured_wet_tail"].endswith("unified_catalog.py:_silent_after")
    assert post_sound["query_frame"] == item["evidence"]["query_frame"]
    assert post_sound["wet_tail_end_s"] == pytest.approx(2.25)
    assert post_sound["query_time_s"] > post_sound["wet_tail_end_s"]
    assert readings["no_other_source_wet_tail_at_query"]["other_event_wet_tails"] == []
    history = readings["continuous_visual_identity_history"]
    assert history["frame_range"] == [item["evidence"]["anchor_frame"],
                                      item["evidence"]["query_frame"]]


def test_relation_reading_shows_the_answer_is_not_a_clip_wide_property():
    item = generate_binding_question(
        _facts(), "visual_conditioned_relation",
        {"appearance_values": ["green", "pink"], "window_s": [4, 6]})
    readings = {row["key"]: row["reading"] for row in
                item["binding_conditions"]["question_conditions"]}
    assert readings["not_a_global_overlap_property"]["distinct_pair_answers"] == [False, True]
    assert readings["integer_second_activity_window"]["window_s"] == [4, 6]
    assert set(readings["emission_readback_available"]["interval_counts"]) == set(
        item["evidence"]["selected_actor_ids"])


# --------------------------------------------------------------------------- group identity


def _clip(prepared, actor, start=0, end=100):
    return {"sound_asset_id": prepared, "actor_id": actor,
            "rendered_slice": {"start_sample": 0, "end_sample_exclusive": 100},
            "prepared_audio_id": prepared, "source_relative": "speech/original.wav",
            "source_sha256": "b" * 64, "source_crop_start_sample": start,
            "source_crop_end_sample_exclusive": end, "crop_provenance": "sound_pool_row"}


def _group_with_audio_inputs(identities):
    group = _group()
    for member in group["members"]:
        member["episode_conditions"] = {
            "audio_input_identity": deepcopy(identities[member["sample_id"]])}
    return group


def _paired_identities(second_pair):
    first = {"e0": _clip("pA", "a0")}
    return {"s00": first, "s10": deepcopy(first),
            "s01": deepcopy(second_pair), "s11": deepcopy(second_pair)}


def test_shared_audio_must_be_one_crop_not_one_window_per_member(tmp_path):
    reassigned = {"e0": _clip("pA", "a1")}
    result = validate_group(_group_with_audio_inputs(_paired_identities(reassigned)),
                            base=tmp_path, verify_media=False)
    assert result["shared_audio_crop_check"] == "pass"
    assert result["audio_intervention_kinds"] == {
        "same_audio": 2, "reassignment_of_same_crops": 2}

    identities = _paired_identities(reassigned)
    # One member of the shared-audio pair silently picked a different window of
    # the same recording.
    identities["s10"]["e0"]["source_crop_start_sample"] = 4800
    identities["s10"]["e0"]["source_crop_end_sample_exclusive"] = 4900
    with pytest.raises(BindingGroupError, match="different clip or crop per member"):
        validate_group(_group_with_audio_inputs(identities), base=tmp_path, verify_media=False)


def test_audio_intervention_kind_separates_reassignment_from_other_clips(tmp_path):
    other = {"e0": _clip("pB", "a1", start=9600, end=9700)}
    result = validate_group(_group_with_audio_inputs(_paired_identities(other)),
                            base=tmp_path, verify_media=False)
    assert result["audio_intervention_kinds"] == {
        "same_audio": 2, "different_clip_inventory": 2}
    # Reported, never fatal: a different clip is still a real audio change.
    assert result["status"] == "structure_only"


def test_missing_audio_input_identity_is_reported_rather_than_assumed(tmp_path):
    result = validate_group(_group(), base=tmp_path, verify_media=False)
    assert result["shared_audio_crop_check"] == "not_run"
    assert result["audio_intervention_kinds"] == {}
    assert all(row["audio_inputs"]["status"] == "not_run" for row in result["comparisons"])


def test_declared_identifiers_carry_their_origin_and_absent_ones_stay_unknown():
    group = {"group_id": "g1", "task_family": "visible_binding",
             "room_family": "apartment", "room_id": "legacy_ue_apartment_0000_v1",
             "world_id": "w1"}
    identity = binding_groups._source_identity(group)
    assert identity["task_family"] == {
        "value": "visible_binding", "status": "declared",
        "provenance": "group_spec.task_family"}
    without_room = binding_groups._source_identity({k: v for k, v in group.items()
                                                    if k != "room_id"})
    assert without_room["room_id"]["status"] == "unknown"
    assert without_room["room_id"]["value"] is None
    assert "g1" in without_room["room_id"]["provenance"]


@pytest.mark.parametrize("field", ["task_family", "room_family", "world_id"])
def test_a_required_identifier_is_never_inferred(field):
    group = {"group_id": "g1", "task_family": "visible_binding",
             "room_family": "apartment", "world_id": "w1"}
    del group[field]
    with pytest.raises(BindingGroupError, match=f"must declare a nonempty {field}"):
        binding_groups._source_identity(group)
    with pytest.raises(BindingGroupError, match="not one of"):
        binding_groups._source_identity({**group, field: "x", "task_family": "invented"})


def test_identity_matrix_counts_real_declarations_and_names_the_absent_family():
    packed = [
        {"group_id": "g1", "members": [{}, {}, {}, {}],
         "source_identity": binding_groups._source_identity(
             {"group_id": "g1", "task_family": "visible_binding",
              "room_family": "apartment", "world_id": "w1"})},
        {"group_id": "g2", "members": [{}, {}, {}, {}],
         "source_identity": binding_groups._source_identity(
             {"group_id": "g2", "task_family": "visible_binding",
              "room_family": "apartment", "room_id": "r", "world_id": "w2"})},
    ]
    matrix = binding_groups._identity_matrix(packed)
    assert matrix["task_family_by_room_family"]["visible_binding"]["apartment"] == {
        "group_count": 2, "sample_count": 8}
    assert matrix["task_families_absent"] == [
        "visual_conditioned_relation", "cross_event_identity", "cross_time_state"]
    assert [row["field"] for row in matrix["unknown_source_identity"]] == ["room_id"]


def test_public_rows_may_not_carry_grouping_or_intervention_identity():
    packed = [{
        "group_id": "g1", "task_family": "visible_binding", "world_id": "w1",
        "members": [{"member_id": "v0_a0", "native_episode_id": "ep_v0",
                     "facts_path": "/tmp/facts.json",
                     "interventions": {"visual_variant": "v0", "audio_assignment": "a0"}}],
    }]
    clean = [{"sample_id": "sample_000001", "question_id": "sample_000001",
              "media": {"video_path": "media/video_3.mp4", "audio_path": "media/audio_3.wav"},
              "forms": {"mcq": {"question_en": "Which visual appearance ...",
                                "options": [{"option": "A", "label_en": "blue-shirt person"}]}}}]
    assert binding_groups._check_public_payload(clean, packed)["status"] == "pass"
    with pytest.raises(BindingGroupError, match="values=..v0_a0"):
        binding_groups._check_public_payload(
            [{**clean[0], "routing": "v0_a0"}], packed)
    with pytest.raises(BindingGroupError, match="keys="):
        binding_groups._check_public_payload(
            [{**clean[0], "group_id": "anything"}], packed)
    # An option label that happens to be the gold answer is public by design.
    assert binding_groups._check_public_payload(
        [{**clean[0], "forms": {"mcq": {"options": [{"label_en": "blue-shirt person"}]}}}],
        packed)["status"] == "pass"
