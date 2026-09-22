from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
from avengine.qa.semantic_questions import generate_semantic_questions
from avengine.qa import unified_catalog as u
from avengine.qa.unified_scoring import score_unified_item

def fixture():
    path=Path(__file__).with_name("test_qa_unified_catalog.py")
    spec=importlib.util.spec_from_file_location("semantic_fixture",path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    raw=m._fixture()
    raw["audio_readback"]["source_activity_intervals_samples"]=[
        {"event_id":e["event_id"],"start_sample":e["start_sample"],"end_sample_exclusive":e["end_sample_exclusive"]}
        for e in raw["audio_program"]["events"]]
    facts=u.normalize_episode_bundle(raw)
    events=facts["events"][:2]
    scenario={"id":"request","slot":"object","question_en":"What did the {appearance} ask for?","question_zh":"{appearance}想要什么？"}
    sounds=[]
    for e,answer in zip(events,["cup","book"]):
        sounds.append({"sound_asset_id":e["sound_asset_id"],"scenario_id":"request","transcript":e["transcript"],
                       "semantic_answer":{"answer":answer,"label_en":answer,"label_zh":{"cup":"杯子","book":"书"}[answer],"aliases":[answer]}})
    facts["events"]=events
    return facts,{"sounds":sounds,"dialogues":{"scenarios":[scenario]}}

def test_meaning_questions_have_two_content_candidates_and_no_hidden_actor_answer():
    facts,manifest=fixture()
    result=generate_semantic_questions(facts,manifest)
    assert len(result["items"])==4
    for item in result["items"]:
        assert item["required_modalities"]==["audio","video"]
        assert item["cross_modal_necessity_claim"] is False
        if item["qa_id"]=="QA-26":assert "mcq" not in item["forms"]
        elif "mcq" in item["forms"]:assert len(item["forms"]["mcq"]["options"])>=4
        assert len(item["evidence"]["candidate_semantic_answers"])==2
        form=item["forms"]["open"]
        assert score_unified_item(item,form["classes"][str(form["truth"])][0])["score"]==1

def test_ambiguous_or_mismatched_meaning_is_not_invented_from_text():
    facts,manifest=fixture()
    mismatch=deepcopy(manifest);mismatch["sounds"][0]["transcript"]="unrendered words"
    with pytest.raises(ValueError,match="differs"):
        generate_semantic_questions(facts,mismatch)
    same=deepcopy(manifest);same["sounds"][1]["semantic_answer"]=same["sounds"][0]["semantic_answer"]
    assert not generate_semantic_questions(facts,same)["items"]
    one=deepcopy(facts);one["events"]=one["events"][:1]
    assert not generate_semantic_questions(one,manifest)["items"]

