"""WP-G round two: more questions per world, and interventions that must not move the answer.

A core group's necessity claim ("change this and the answer changes") only says
something next to its opposite ("change that and it does not"). These tests pin
down the second half, and the two extra questions the same controlled world can
now be asked.
"""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from avengine.dataset import binding_group_native as native
from avengine.qa import binding_conditions as conditions
from avengine.qa.binding_questions import BindingQuestionError, generate_binding_question
from avengine.qa.binding_groups import align_question_forms
from avengine.qa.unified_catalog import UNIFIED_FACT_SCHEMA


def _facts(colors=("blue", "green")):
    actors = {
        f"a{index}": {
            "actor_id": f"a{index}", "asset_id": f"asset{index}", "species_id": "human",
            "display_label": f"{color} shirt",
            "appearance": {"field": "top_color", "value": color, "label": f"{color} shirt"},
            "root_positions_m": [[index - 1, 0, -3]] * 12,
            "emitter_positions_m": [[index - 1, 1, -3]] * 12,
        } for index, color in enumerate(colors)
    }
    rows = [("a0", 1, 2), ("a1", 4, 6)][:len(colors)]
    events = []
    for index, (actor, start, end) in enumerate(rows):
        events.append({
            "event_id": f"e{index}", "actor_id": actor, "sound_asset_id": f"sound{index}",
            "start_s": start, "end_s": end, "start_frame": int(start), "end_frame": int(end),
            "sound_class": "speech", "sound_class_explicit": True,
            "source_activity_intervals_samples": [
                {"start_sample": int(start * 100), "end_sample_exclusive": int(end * 100)}],
            "source_activity_evidence_status": "observed",
        })
    return {
        "schema": UNIFIED_FACT_SCHEMA, "status": "research_candidate", "episode_id": "fixture",
        "time": {"frame_count": 12, "frame_rate_hz": 1.0, "sample_rate_hz": 100,
                 "sample_count": 1200, "duration_seconds": 12.0, "time_base_hz": 100},
        "actors": actors, "events": events, "listener": {},
        "audio": {"status": "pass", "channel_count": 2, "wet_tail_intervals": [
            {"event_id": row["event_id"], "start_s": row["start_s"], "end_s": row["end_s"] + 0.25}
            for row in events]},
        "source_activity_evidence_present": True,
        "source_activity_intervals_samples": {
            row["event_id"]: row["source_activity_intervals_samples"] for row in events},
        "visibility": {
            actor: {frame: {"state": "visible_clear", "frame_index": frame,
                            "visible_fraction": 1.0, "visible_pixels": 600,
                            "target_centroid_xy_px": [30 + index * 20, 50]}
                    for frame in range(12)} for index, actor in enumerate(actors)},
        "visibility_meta": {"resolution_hw": [100, 100]},
        "camera_calibration": {"projection": "pinhole", "public": True, "fx_px": 50.0,
                               "cx_px": 50.0, "width_px": 100, "height_px": 100},
        "appearance_review": {
            actor: {"status": "reviewed", "value": colors[index], "frame_refs": list(range(12))}
            for index, actor in enumerate(actors)},
        "sampling": {}, "input_summary": {},
    }


def _swap_appearances(facts):
    """The visual intervention: the two bodies exchange shirts, nothing moves."""
    swapped = deepcopy(facts)
    for actor, other in (("a0", "a1"), ("a1", "a0")):
        swapped["actors"][actor]["appearance"] = deepcopy(facts["actors"][other]["appearance"])
        swapped["appearance_review"][actor] = deepcopy(facts["appearance_review"][other])
    return swapped


def _swap_events(facts):
    """The audio intervention: the two slots exchange which event they emit."""
    swapped = deepcopy(facts)
    for row in swapped["events"]:
        row["actor_id"] = "a1" if row["actor_id"] == "a0" else "a0"
    return swapped


# --------------------------------------------------------------------------- QA-02

def test_qa02_answers_with_the_appearance_of_whoever_made_the_sound():
    item = generate_binding_question(_facts(), "visible_binding", {"event_number": 1},
                                     qa_id="QA-02")
    assert item["qa_id"] == "QA-02"
    assert item["truth"]["value"] == "blue"
    assert generate_binding_question(_swap_appearances(_facts()), "visible_binding",
                                     {"event_number": 1}, qa_id="QA-02")["truth"]["value"] == "green"
    assert generate_binding_question(_swap_events(_facts()), "visible_binding",
                                     {"event_number": 1}, qa_id="QA-02")["truth"]["value"] == "green"


def test_qa02_offers_every_reviewed_appearance_not_only_the_visible_ones():
    facts = _facts()
    facts["visibility"]["a1"] = {frame: {"state": "out_of_view", "frame_index": frame,
                                         "visible_fraction": 0.0, "visible_pixels": 0}
                                 for frame in range(12)}
    # a1 is off screen at the anchor, so QA-20 cannot offer it as a visible
    # candidate; QA-02's domain is the reviewed appearances, so it still can.
    item = generate_binding_question(facts, "visible_binding", {"event_number": 1},
                                     qa_id="QA-02")
    assert sorted(option["value"] for option in item["forms"]["mcq"]["options"]) == ["blue", "green"]
    assert item["evidence"]["visible_candidate_actor_ids"] == ["a0"]


def test_qa02_refuses_a_sound_two_sources_share():
    facts = _facts()
    # a1 sounds across the whole clip while a0 sounds inside it: the two onsets
    # are still separable, but at every frame a0 is audible so is a1, and "the
    # object that produced the second sound" names no single emitter.
    facts["events"][0].update(start_s=3, end_s=5, start_frame=3, end_frame=5)
    facts["events"][0]["source_activity_intervals_samples"] = [
        {"start_sample": 300, "end_sample_exclusive": 500}]
    facts["events"][1].update(start_s=0.5, end_s=8, start_frame=0, end_frame=8)
    facts["events"][1]["source_activity_intervals_samples"] = [
        {"start_sample": 50, "end_sample_exclusive": 800}]
    for row in facts["events"]:
        facts["source_activity_intervals_samples"][row["event_id"]] = row[
            "source_activity_intervals_samples"]
    facts["audio"]["wet_tail_intervals"] = [
        {"event_id": row["event_id"], "start_s": row["start_s"], "end_s": row["end_s"] + 0.25}
        for row in facts["events"]]
    with pytest.raises(BindingQuestionError, match="attributable"):
        generate_binding_question(facts, "visible_binding", {"event_number": 2}, qa_id="QA-02")


# --------------------------------------------------------------------------- QA-19

def test_qa19_answers_with_the_interval_holding_the_named_target_first_sound():
    query = {"appearance_value": "blue"}
    base = generate_binding_question(_facts(), "visible_binding", query, qa_id="QA-19")
    assert base["qa_id"] == "QA-19"
    assert base["truth"]["value"] == [0.0, 3.0]
    # Both interventions move it: the shirt moves to the other body, and the
    # other body emits at a different time.
    assert generate_binding_question(_swap_appearances(_facts()), "visible_binding", query,
                                     qa_id="QA-19")["truth"]["value"] == [3.0, 6.0]
    assert generate_binding_question(_swap_events(_facts()), "visible_binding", query,
                                     qa_id="QA-19")["truth"]["value"] == [3.0, 6.0]


def test_qa19_asks_all_members_the_same_sentence():
    query = {"appearance_value": "blue"}
    items = [generate_binding_question(facts, "visible_binding", query, qa_id="QA-19")
             for facts in (_facts(), _swap_appearances(_facts()), _swap_events(_facts()),
                           _swap_events(_swap_appearances(_facts())))]
    align_question_forms(items, "qa19")
    assert len({json.dumps(item["model_input"], sort_keys=True) for item in items}) == 1
    assert [item["truth"]["value"] for item in items] == [
        [0.0, 3.0], [3.0, 6.0], [3.0, 6.0], [0.0, 3.0]]


def test_qa19_refuses_an_appearance_two_candidates_share():
    facts = _facts(colors=("blue", "blue"))
    with pytest.raises(BindingQuestionError, match="distinct reviewed appearances"):
        generate_binding_question(facts, "visible_binding", {"appearance_value": "blue"},
                                  qa_id="QA-19")


def test_qa19_answer_domain_does_not_depend_on_the_unqueried_actor():
    # This is why QA-19 can carry an unqueried_actor_appearance control and a
    # closed set drawn from the candidates cannot: recolouring the other actor
    # leaves both the options and the answer alone.
    query = {"appearance_value": "blue"}
    base = generate_binding_question(_facts(), "visible_binding", query, qa_id="QA-19")
    recoloured = _facts()
    recoloured["actors"]["a1"]["appearance"] = {"field": "top_color", "value": "burgundy",
                                                "label": "burgundy shirt"}
    recoloured["appearance_review"]["a1"] = {"status": "reviewed", "value": "burgundy",
                                             "frame_refs": list(range(12))}
    control = generate_binding_question(recoloured, "visible_binding", query, qa_id="QA-19")
    assert control["truth"]["value"] == base["truth"]["value"]
    assert control["model_input"] == base["model_input"]


def test_a_family_refuses_a_question_it_cannot_build():
    with pytest.raises(BindingQuestionError, match="no builder"):
        generate_binding_question(_facts(), "visible_binding", {"event_number": 1},
                                  qa_id="QA-17")


# --------------------------------------------------------------------------- clip control

def _clip_plan():
    def event(event_id, actor, start, count, asset, identity):
        return {"event_id": event_id, "actor_id": actor, "entity_instance_id": f"inst_{actor}",
                "gender": "M", "is_vktk": False,
                "source_endpoint_id": f"{actor}_mouth", "start_sample": start,
                "end_sample": start + count, "start_tick": start * 3, "end_tick": (start + count) * 3,
                "sample_count": count, "sample_rate_hz": 16000, "sound_class": "speech_playback",
                "sound_asset_id": asset, "prepared_audio_id": asset,
                "sound_identity_id": identity, "transcript": f"line for {asset}",
                "audible_start_sample": 0, "audible_end_sample_exclusive": count,
                "clip_length_bound_samples": 80000,
                "clip_selection_deadline_samples": 112000,
                "compatible_asset_ids": ["one_blue", "one_green"],
                "linear_gain": 1.0, "source_asset_id": "one_blue"}
    return {
        "clock": {"sample_rate_hz": 16000, "time_base_hz": 48000, "sample_count": 160000,
                  "frame_count": 150, "frame_rate_hz": 15.0},
        "visual_plan": {"actors": [
            {"actor_id": "source1", "entity_instance_id": "inst_source1",
             "asset_id": "one_blue", "entity_class": "articulated_human",
             "identity": {"species_id": "human"},
             "realized_attributes": {"sex_or_gender_label": "male", "life_stage": "adult"}},
            {"actor_id": "source2", "entity_instance_id": "inst_source2",
             "asset_id": "one_green", "entity_class": "articulated_human",
             "identity": {"species_id": "human"},
             "realized_attributes": {"sex_or_gender_label": "male", "life_stage": "adult"}}]},
        "audio_events": [event("event_001", "source1", 8000, 40000, "clip_a", "speaker:p1"),
                         event("event_002", "source2", 64000, 40000, "clip_b", "speaker:p2")],
        "voice_bindings": [event("event_001", "source1", 8000, 40000, "clip_a", "speaker:p1"),
                           event("event_002", "source2", 64000, 40000, "clip_b", "speaker:p2")],
    }


def _pool(rows):
    return {"sounds": [
        {"sound_asset_id": asset, "prepared_audio_id": asset, "sound_class": klass,
         "sound_identity_id": identity, "sample_count": count, "sample_rate_hz": 16000,
         "audible_start_sample": 0, "audible_end_sample_exclusive": count,
         "transcript": f"line for {asset}", "prepared_sha256": asset,
         "gender": "M", "is_vctk": False,
         "compatible_asset_ids": ["one_blue", "one_green"]}
        for asset, klass, identity, count in rows]}


def test_the_room_for_another_recording_stops_at_the_next_event():
    plan = _clip_plan()
    room = native.available_clip_room_samples(
        plan, "event_001", {"profile": {"min_gap_between_audible_windows_s": 0.5}})
    # 64000 - 8000 - 0.5 s of gap, not the 80000 sample clip bound.
    assert room == 64000 - 8000 - 8000


def test_a_control_clip_stays_in_class_and_changes_the_recording():
    plan = _clip_plan()
    pool = _pool([("clip_a", "speech_playback", "speaker:p1", 40000),
                  ("clip_other_class", "dog_bark", "dog:1", 20000),
                  ("clip_too_long", "speech_playback", "speaker:p3", 90000),
                  ("clip_c", "speech_playback", "speaker:p9", 30000)])
    chosen = native.select_clip_within_class(
        pool, plan["audio_events"][0], exclude_sound_asset_ids=["clip_a", "clip_b"],
        sample_rate_hz=16000, maximum_sample_count=48000, seed="unit")
    assert chosen["sound_asset_id"] == "clip_c"
    assert chosen["_control_selection"]["candidate_count"] == 1


def test_a_control_clip_keeps_the_onset_and_the_slot_and_recomputes_the_rest():
    plan = _clip_plan()
    pool = _pool([("clip_c", "speech_playback", "speaker:p9", 30000)])
    chosen = native.select_clip_within_class(
        pool, plan["audio_events"][0], sample_rate_hz=16000,
        maximum_sample_count=48000, seed="unit")
    updated, request = native.build_clip_variant_plan(
        plan, {"sound_pool": "x"}, event_id="event_001", replacement=chosen, level="c1")
    first = updated["audio_events"][0]
    assert first["event_id"] == "event_001" and first["actor_id"] == "source1"
    assert first["start_sample"] == 8000 and first["source_endpoint_id"] == "source1_mouth"
    assert first["sound_asset_id"] == "clip_c"
    # end, ticks and the audible interval follow the sampler's own rule.
    assert first["end_sample"] == 8000 + 30000
    assert first["start_tick"] == int(round(8000 * 48000 / 16000))
    assert first["end_tick"] == int(round(38000 * 48000 / 16000))
    assert first["planned_audible_interval_samples"] == [8000, 38000]
    assert updated["audio_events"][1]["sound_asset_id"] == "clip_b"
    assert updated["control_intervention"]["factor"] == "queried_clip_within_class"
    assert request["control_intervention"]["replacement_sound_asset_id"] == "clip_c"


def test_a_control_clip_that_would_push_a_neighbour_is_refused():
    plan = _clip_plan()
    chosen = _pool([("clip_long", "speech_playback", "speaker:p9", 70000)])["sounds"][0]
    with pytest.raises(native.BindingNativeError, match="overlap"):
        native.build_clip_variant_plan(plan, {}, event_id="event_001",
                                       replacement=chosen, level="c1")


def test_a_control_clip_may_not_leave_its_sound_class():
    plan = _clip_plan()
    chosen = _pool([("clip_bark", "dog_bark", "dog:1", 20000)])["sounds"][0]
    with pytest.raises(native.BindingNativeError, match="sound class"):
        native.build_clip_variant_plan(plan, {}, event_id="event_001",
                                       replacement=chosen, level="c1")


# --------------------------------------------------------------------------- control members in the table

def _levels(visual, audio, **extra):
    return {"visual_appearance_slots": visual, "audio_event_slot_assignment": audio, **extra}


def _crossed_with_clip_control():
    visual = {"v0": {"source1": "blue", "source2": "green"},
              "v1": {"source1": "green", "source2": "blue"}}
    audio = {"a0": {"event_001": "source1", "event_002": "source2"},
             "a1": {"event_001": "source2", "event_002": "source1"}}
    members = []
    for v in ("v0", "v1"):
        for a in ("a0", "a1"):
            members.append({"member_id": f"{v}_{a}",
                            "factor_levels": _levels(v, a, queried_clip_within_class="c0"),
                            "bindings": {"slot_appearances": visual[v], "event_slots": audio[a],
                                         "event_order": ["event_001", "event_002"]}})
    for v in ("v0", "v1"):
        members.append({"member_id": f"{v}_a0c1",
                        "factor_levels": _levels(v, "a0", queried_clip_within_class="c1"),
                        "bindings": {"slot_appearances": visual[v], "event_slots": audio["a0"],
                                     "event_order": ["event_001", "event_002"]}})
    return members


def test_a_clip_control_pair_keeps_every_member_with_both_witnesses():
    rows = conditions.derive_group_comparisons(_crossed_with_clip_control(), qa_id="QA-20")
    necessity = [row for row in rows if row["kind"] == "necessity"]
    invariance = [row for row in rows if row["kind"] == "invariance"]
    for member_id in [member["member_id"] for member in _crossed_with_clip_control()]:
        shared = {row["shared_modality"] for row in necessity if member_id in row["members"]}
        assert shared == {"audio", "video"}, (member_id, shared)
    # The control claim itself: same video, different recording, same answer.
    control = [row for row in invariance
               if row["intervention_factors"] == ["queried_clip_within_class"]]
    assert len(control) == 2
    assert all(row["shared_modality"] == "video" for row in control)
    assert all(row["changed_modalities"] == ["audio"] for row in control)


def _member_plan(tmp_path, member_id, slot_assets, event_slots):
    """A member root shaped the way the producer writes one."""
    root = tmp_path / member_id
    (root / "plan").mkdir(parents=True)
    (root / "delivery").mkdir(parents=True)
    plan = {
        "clock": {"frame_count": 150, "frame_rate_hz": 15.0, "sample_rate_hz": 16000,
                  "sample_count": 160000, "time_base_hz": 48000},
        "visual_plan": {"actors": [{"actor_id": slot, "entity_instance_id": f"inst_{slot}",
                                    "asset_id": asset}
                                   for slot, asset in slot_assets.items()]},
        "audio_events": [{"event_id": event_id, "actor_id": slot,
                          "start_sample": 8000 if event_id == "event_001" else 64000,
                          "end_sample": 40000 if event_id == "event_001" else 96000}
                         for event_id, slot in event_slots.items()],
        "audio_assignment_targets": dict(event_slots),
    }
    (root / "plan/episode_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    facts = root / "delivery/facts.json"
    facts.write_text("{}", encoding="utf-8")
    return {"facts": str(facts), "audio": str(facts)}


def test_a_group_may_carry_more_than_the_crossed_four(tmp_path):
    # _group_spec used to demand exactly four; a control pair is extra rows in
    # the same table, not a different shape.
    assets = {"v0": {"source1": "one_blue", "source2": "one_green"},
              "v1": {"source1": "one_green", "source2": "one_blue"}}
    slots = {"a0": {"event_001": "source1", "event_002": "source2"},
             "a1": {"event_001": "source2", "event_002": "source1"}}
    visual = {"v0": {"capture": "/tmp/v0", "visual_video": "/tmp/v0.mp4"},
              "v1": {"capture": "/tmp/v1", "visual_video": "/tmp/v1.mp4"}}
    variants, units, levels = {}, [], {}
    for member_id, visual_id, audio, clip in (
            ("v0_a0", "v0", "a0", "c0"), ("v0_a1", "v0", "a1", "c0"),
            ("v1_a0", "v1", "a0", "c0"), ("v1_a1", "v1", "a1", "c0"),
            ("v0_a0c1", "v0", "a0", "c1"), ("v1_a0c1", "v1", "a0", "c1")):
        variants[member_id] = _member_plan(tmp_path, member_id, assets[visual_id], slots[audio])
        units.append((member_id, visual_id, member_id))
        levels[member_id] = _levels(visual_id, audio, queried_clip_within_class=clip)
    spec = native._group_spec("g", "w", "apartment", "room", visual, variants,
                              member_units=units, member_factor_levels=levels,
                              qa_id="QA-20", source_registry=_registry())
    group = spec["groups"][0]
    assert [member["member_id"] for member in group["members"]] == [
        "v0_a0", "v0_a1", "v1_a0", "v1_a1", "v0_a0c1", "v1_a0c1"]
    assert group["members"][4]["interventions"]["factor_levels"][
        "queried_clip_within_class"] == "c1"
    assert group["members"][4]["planned_answer"]["value"] == "blue"
    control = [row for row in group["comparisons"]
               if row["intervention_factors"] == ["queried_clip_within_class"]]
    assert len(control) == 2 and all(row["kind"] == "invariance" for row in control)


# --------------------------------------------------------------------------- appearance control

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


def _registry():
    rows = []
    for asset_id, colour in (("one_blue", "blue"), ("one_green", "green"),
                             ("one_burgundy", "burgundy"), ("one_pink", "pink")):
        record = json.loads(json.dumps(BODY))
        record.update(asset_id=asset_id, entity_class="articulated_human",
                      identity={"species_id": "human"},
                      realized_attributes={"top_color": colour})
        rows.append(record)
    other = json.loads(json.dumps(BODY))
    other["emitter_anchors"][0]["offset_m"] = [0.0, 1.5, 0.0]
    other.update(asset_id="other_yellow", entity_class="articulated_human",
                 identity={"species_id": "human"},
                 realized_attributes={"top_color": "yellow"})
    rows.append(other)
    return {"assets": rows}


def test_the_third_appearance_shares_the_body_and_a_new_colour_family():
    chosen = native.select_appearance_substitute(
        _registry(), body_of="one_blue", avoid_families=["blue", "green"])
    assert chosen["asset_id"] == "one_burgundy"
    assert chosen["appearance_family"] == "red"
    with pytest.raises(native.BindingNativeError, match="colour family outside"):
        native.select_appearance_substitute(
            _registry(), body_of="one_blue",
            avoid_families=["blue", "green", "red", "yellow"])


def test_substituting_an_asset_is_not_a_permutation():
    request = {"source_asset_ids": ["one_blue", "one_green"],
               "entity_instances": [{"instance_id": "human_a", "asset_id": "one_blue"},
                                    {"instance_id": "human_b", "asset_id": "one_green"}],
               "entities": {"instances": [{"instance_id": "human_a", "asset_id": "one_blue"},
                                          {"instance_id": "human_b", "asset_id": "one_green"}]}}
    result = native.substitute_declared_instance_asset(
        request, replace="one_green", with_asset="one_burgundy")
    assert request["source_asset_ids"] == ["one_blue", "one_burgundy"]
    assert [row["asset_id"] for row in request["entity_instances"]] == ["one_blue", "one_burgundy"]
    assert result["moved"]
    with pytest.raises(native.BindingNativeError, match="already selected"):
        native.substitute_declared_instance_asset(
            request, replace="one_blue", with_asset="one_burgundy")


def test_an_appearance_control_pair_is_an_invariance_on_a_question_that_allows_it():
    visual = {"v0": {"source1": "blue", "source2": "green"},
              "v1": {"source1": "green", "source2": "blue"},
              "v2": {"source1": "blue", "source2": "burgundy"}}
    audio = {"a0": {"event_001": "source1", "event_002": "source2"},
             "a1": {"event_001": "source2", "event_002": "source1"}}
    starts = {"event_001": 0.5, "event_002": 4.0}
    bands = [[0, 2], [2, 5], [5, 8], [8, 10]]
    members = []
    for visual_id in ("v0", "v1", "v2"):
        for assignment in ("a0", "a1"):
            members.append({
                "member_id": f"{visual_id}_{assignment}",
                "factor_levels": _levels("v0" if visual_id == "v2" else visual_id, assignment,
                                         unqueried_actor_appearance="u1" if visual_id == "v2" else "u0"),
                "bindings": {"slot_appearances": visual[visual_id],
                             "event_slots": audio[assignment],
                             "event_order": ["event_001", "event_002"],
                             "event_start_s": starts, "time_bands": bands},
            })
    rows = conditions.derive_group_comparisons(
        members, qa_id="QA-19", query={"appearance_value": "blue"})
    control = [row for row in rows
               if row["intervention_factors"] == ["unqueried_actor_appearance"]]
    assert len(control) == 2
    assert all(row["kind"] == "invariance" and row["shared_modality"] == "audio"
               for row in control)
    assert all(row["changed_modalities"] == ["video"] for row in control)
    for member in members:
        shared = {row["shared_modality"] for row in rows
                  if row["kind"] == "necessity" and member["member_id"] in row["members"]}
        assert shared == {"audio", "video"}, (member["member_id"], shared)
