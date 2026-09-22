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
