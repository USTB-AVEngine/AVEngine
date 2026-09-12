from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
from avengine.qa import unified_catalog as u
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms.qa_episode import compile_question_conditions

def fixture_module(name):
    p=Path(__file__).with_name(name)
    spec=importlib.util.spec_from_file_location(name.replace(".py",""),p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def ordinary(facts):
    facts=deepcopy(facts)
    facts.setdefault("sampling",{})["acceptance_policy"]={
        "policy_id":"ordinary_test","question_mode":"ordinary_observation",
        "post_sound_distance_query":"integer_timepoint",
        "sound_class_options":["speech_playback","dog_bark"]}
    return facts

def test_same_answer_is_legal_for_ordinary_observation():
    facts=u.normalize_episode_bundle(fixture_module("test_qa_unified_catalog.py")._fixture())
    for actor in facts["actors"].values():
        actor["moving"]=[False]*facts["time"]["frame_count"]
    strict=u.generate_unified_questions(facts,qa_ids=["QA-06"],include_angle_followups=False)
    simple=u.generate_unified_questions(ordinary(facts),qa_ids=["QA-06"],include_angle_followups=False)
    assert not strict["items"]
    assert simple["items"]
    assert all(i["truth"]["value"]=="still" for i in simple["items"])
    assert all(i["cross_modal_necessity_claim"] is False for i in simple["items"])

def test_registered_global_sound_options_allow_single_class_scene():
    facts=u.normalize_episode_bundle(fixture_module("test_qa_unified_catalog.py")._fixture())
    for event in facts["events"]:
        event["sound_class"]="speech_playback";event["sound_class_explicit"]=True
    strict=u.generate_unified_questions(facts,qa_ids=["QA-21"],include_angle_followups=False)
    simple=u.generate_unified_questions(ordinary(facts),qa_ids=["QA-21"],include_angle_followups=False)
    assert not strict["items"] and simple["items"]
    item=simple["items"][0]
    assert item["truth"]["value"]=="speech_playback"
    assert item["evidence"]["option_domain_source"]=="configured_registered_sound_class_catalog"

def test_timepoint_distance_answer_uses_its_published_integer_frame():
    facts=ordinary(u.normalize_episode_bundle(fixture_module("test_qa_unified_catalog.py")._fixture()))
    facts["sampling"]["time_display_precision"]=0
    # The shared fixture has other sounds at the integer instants. Keep its
    # first event and its own tail so the test has genuine silent timepoints.
    facts["events"]=facts["events"][:1]
    facts["audio"]["wet_tail_intervals"]=facts["audio"]["wet_tail_intervals"][:1]
    result=u.generate_unified_questions(facts,qa_ids=["QA-16"],items_per_type=20,include_angle_followups=False)
    assert result["items"]
    for item in result["items"]:
        e=item["evidence"]
        assert e["query_scope"]=="explicit_integer_timepoint"
        assert e["query_time_s"]==int(e["query_time_s"])
        assert e["query_frame"]==int(e["query_time_s"]*facts["time"]["frame_rate_hz"])
        assert e["distance_delta_m"]==pytest.approx(e["query_distance_m"]-e["reference_distance_m"])

def test_planner_also_removes_only_answer_distinguishability():
    registry=fixture_module("test_v1_takeover_motion_solver.py")._registry()
    instances=[{"entity_instance_id":"target","source_slot_id":"source1","asset_id":"human_0",
                "source_class":"articulated_human","role":"anchor"},
               {"entity_instance_id":"foil","source_slot_id":"source2","asset_id":"dog_0",
                "source_class":"articulated_animal","role":"competitor"}]
    kwargs={"targets":[{"qa_id":"QA-06","branch":"still","target_instance_ids":["target"]}],
            "registry":registry,"generator":cs}
    strict=compile_question_conditions(["QA-06"],instances,**kwargs)
    simple=compile_question_conditions(["QA-06"],instances,question_mode="ordinary_observation",**kwargs)
    old=strict["compiled"][0]["conditions"];new=simple["compiled"][0]["conditions"]
    assert any(c["kind"]=="answer_distinguishable" for c in old)
    assert new==[c for c in old if c["kind"]!="answer_distinguishable"]
