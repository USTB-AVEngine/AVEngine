"""QA-26 to QA-28: speech meaning bound to appearance and to direction."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from avengine.qa import unified_catalog as u
from avengine.qa.semantic_questions import generate_semantic_questions


def _base():
    path = Path(__file__).with_name("test_qa_semantic_questions.py")
    spec = importlib.util.spec_from_file_location("semantic_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture()


FOUR = [("cup", "杯子"), ("book", "书"), ("remote", "遥控器"), ("towel", "毛巾")]


def _four_answer_fixture():
    facts, manifest = _base()
    scenario = manifest["dialogues"]["scenarios"][0]
    scenario["utterances"] = [{"text": f"please bring the {en}", "answer": en, "label_en": en,
                               "label_zh": zh, "aliases": [en]} for en, zh in FOUR]
    scenario["direction_question_zh"] = "请别人拿{answer}的那个人，在你的哪个方向？"
    return facts, manifest


def _by(result, qa_id):
    return [item for item in result["items"] if item["qa_id"] == qa_id]


def _sectors(monkeypatch, by_event):
    def window(facts, event):
        sector = by_event.get(event["event_id"])
        return None if sector is None else ((0, 1), sector, 30.0)
    monkeypatch.setattr(u, "_event_start_sector_window", window)


def test_extension_types_stay_out_of_the_scene_driven_list():
    assert len(u.QA_IDS) == 25
    assert not set(u.EXTENSION_QA_IDS) & set(u.QA_IDS)
    assert u.ALL_QA_IDS[-3:] == ("QA-26", "QA-27", "QA-28")
    assert u.get_requirements("QA-28")["required_modalities"] == ["binaural_audio"]
    with pytest.raises(u.UnifiedQAError, match="QA-01 through QA-28"):
        u._canonical_qa_id("QA-29")


def test_appearance_to_meaning_offers_every_answer_of_the_scenario():
    facts, manifest = _four_answer_fixture()
    items = _by(generate_semantic_questions(facts, manifest), "QA-26")
    assert len(items) == 2
    for item in items:
        values = [o["value"] for o in item["forms"]["mcq"]["options"]]
        assert sorted(values) == sorted(en for en, _ in FOUR)
        heard = item["evidence"]["heard_answers"]
        # Audio alone narrows four options to the two that were said.
        assert len(heard) == 2 and set(heard) <= set(values)
        assert item["truth"]["value"] in heard
        assert item["required_modalities"] == ["audio", "video"]


def test_reverse_is_withheld_when_only_one_speaker_is_visible(monkeypatch):
    facts, manifest = _four_answer_fixture()
    everyone = u._reviewed_appearances(facts)
    keep = sorted(everyone)[0]
    monkeypatch.setattr(u, "_reviewed_appearances", lambda f: {keep: everyone[keep]})
    result = generate_semantic_questions(facts, manifest)
    assert [i["evidence"]["target_actor_id"] for i in _by(result, "QA-26")] == [keep]
    assert not _by(result, "QA-27")
    assert any("elimination" in d.get("reason", "") for d in result["deferred"])


def test_direction_needs_speakers_in_different_sectors(monkeypatch):
    facts, manifest = _four_answer_fixture()
    first, second = (e["event_id"] for e in facts["events"])
    _sectors(monkeypatch, {first: "front-left", second: "back-right"})
    items = _by(generate_semantic_questions(facts, manifest), "QA-28")
    assert len(items) == 2
    assert {i["truth"]["value"] for i in items} == {"front-left", "back-right"}
    for item in items:
        assert len(item["forms"]["mcq"]["options"]) == 8
        assert item["required_modalities"] == ["audio"]
        assert item["evidence"]["answer_domain"] == "eight_45_degree_sectors"

    _sectors(monkeypatch, {first: "front-left", second: "front-left"})
    result = generate_semantic_questions(facts, manifest)
    assert not _by(result, "QA-28")
    assert any("share a sector" in d.get("reason", "") for d in result["deferred"])


def test_mono_render_withholds_direction_but_not_appearance(monkeypatch):
    facts, manifest = _four_answer_fixture()
    first, second = (e["event_id"] for e in facts["events"])
    _sectors(monkeypatch, {first: "front-left", second: "back-right"})

    def mono(_facts):
        raise u.UnifiedQAError("mono")
    monkeypatch.setattr(u, "_require_stereo", mono)
    result = generate_semantic_questions(facts, manifest)
    assert not _by(result, "QA-28") and _by(result, "QA-26")


def test_stems_never_say_when_the_utterance_happened(monkeypatch):
    facts, manifest = _four_answer_fixture()
    first, second = (e["event_id"] for e in facts["events"])
    _sectors(monkeypatch, {first: "front-left", second: "back-right"})
    for item in generate_semantic_questions(facts, manifest)["items"]:
        text = item["forms"]["open"]["question_zh"] + item["forms"]["open"]["question_en"]
        assert "查询区间" not in text and "query interval" not in text


def test_voice_relation_is_recorded_for_analysis():
    facts, manifest = _four_answer_fixture()
    for sound, preset in zip(manifest["sounds"], ["中文男", "英文男"]):
        sound["voice_preset"] = preset
    item = _by(generate_semantic_questions(facts, manifest), "QA-26")[0]
    assert item["evidence"]["voice_relation"] == "different"
    same = deepcopy(manifest)
    for sound in same["sounds"]:
        sound["voice_preset"] = "中文男"
    item = _by(generate_semantic_questions(facts, same), "QA-26")[0]
    assert item["evidence"]["voice_relation"] == "same"


def test_direction_wording_follows_the_meaning_not_the_speaker(monkeypatch):
    """Swapping who said what must keep each stem, so the paired check can compare it."""
    facts, manifest = _four_answer_fixture()
    first, second = (e["event_id"] for e in facts["events"])
    _sectors(monkeypatch, {first: "front-left", second: "back-right"})
    before = {i["evidence"]["authored_semantic_answer"]["answer"]: i["forms"]["open"]["question_zh"]
              for i in _by(generate_semantic_questions(facts, manifest), "QA-28")}
    swapped = deepcopy(manifest)
    a, b = swapped["sounds"]
    a["semantic_answer"], b["semantic_answer"] = b["semantic_answer"], a["semantic_answer"]
    after = {i["evidence"]["authored_semantic_answer"]["answer"]: i["forms"]["open"]["question_zh"]
             for i in _by(generate_semantic_questions(facts, swapped), "QA-28")}
    assert before == after
