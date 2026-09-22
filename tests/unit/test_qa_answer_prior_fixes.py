"""Regressions for shortcuts observed in retained bank questions."""
from avengine.qa.unified_catalog import _named_alternatives


def test_reused_event_ids_do_not_fix_wording_across_episodes():
    choices=(("nearer","nearer","更近"),("farther","farther","更远"))
    seed="claude-constructive-render-20260912"
    orders=[_named_alternatives(seed,"QA-15","event_001",choices,episode_id=f"episode_{i}")[2] for i in range(32)]
    assert {tuple(x) for x in orders} == {("nearer","farther"),("farther","nearer")}
    assert orders[0] == _named_alternatives(seed,"QA-15","event_001",choices,episode_id="episode_0")[2]

from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
from avengine.qa import unified_catalog as u


def _facts():
    path=Path(__file__).with_name("test_qa_unified_catalog.py")
    spec=importlib.util.spec_from_file_location("prior_fixture",path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    return u.normalize_episode_bundle(fixture._fixture())


def test_qa02_can_name_a_real_non_speech_event():
    facts=_facts()
    event=facts["events"][0]
    event["transcript"]=None
    event["sound_class"]="laughter";event["sound_class_explicit"]=True
    event["sound_asset_id"]="registered_laughter"
    pool=u._P8_CANDIDATES["QA-02"](facts)
    candidate=next(c for c in pool if c["event_id"]==event["event_id"])
    item=u._P8_EMITTERS["QA-02"](facts,candidate,"non-speech")
    assert item["evidence"]["event_id"]==event["event_id"]
    assert not item["evidence"]["transcript_quoted"]
    assert item["truth"]["value"]==facts["actors"][event["actor_id"]]["appearance"]["value"]


def test_qa18_drops_only_impossible_multiple_and_keeps_one_actor_none():
    facts=_facts()
    reviewed=u._reviewed_appearances(facts)
    one=dict(list(reviewed.items())[:1])
    options=u._qa18_actor_options(facts,one)
    assert {o["value"] for o in options}==set(one)|{"none"}
    assert "multiple" in {o["value"] for o in u._qa18_actor_options(facts,reviewed)}
    with pytest.raises(u._Deferred,match="at least one identifiable"):
        u._qa18_actor_options(facts,{})

from avengine.qa.choice_support import apply_choice_support


def _choice_item(values, *, open_form=True):
    forms={"mcq":{"options":[{"value":v,"label_en":v} for v in values],"gold":{"correct_index":1,"value":values[1]}}}
    if open_form:forms["open"]={"answer_type":"closed_set","truth":values[1]}
    return {"forms":forms,"model_input":{"mcq":{"options":values},"open":{"question_en":"Who?"}},"truth":{"value":values[1]},"evidence":{}}


def test_bank_policy_keeps_open_truth_instead_of_inventing_two_actors():
    item=_choice_item(["actor1","actor2"])
    fresh=apply_choice_support(item)
    assert "mcq" not in fresh["forms"] and "mcq" not in fresh["model_input"]
    assert fresh["forms"]["open"]==item["forms"]["open"]
    assert fresh["truth"]["value"]==item["truth"]["value"]
    assert fresh["evidence"]["mcq_support"]["action"]=="open_only"
    assert "mcq" in item["forms"]  # Retained source candidates are untouched.
    assert "mcq" in apply_choice_support(item,minimum=2)["forms"]


@pytest.mark.parametrize("values", [["yes","no"],["fov_band_0","fov_band_1","fov_band_2"],["a","b","c","d"]])
def test_bank_policy_retains_real_four_options_and_intrinsic_domains(values):
    item=_choice_item(values)
    fresh=apply_choice_support(item)
    assert fresh["forms"]["mcq"]==item["forms"]["mcq"]
    assert fresh["evidence"]["mcq_support"]["chance_baseline"]==1/len(values)
