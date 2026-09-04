"""Tests for Gate-B gold and representative precert rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from finalize_qa_v3_gateb_precheck import audio_pair, pixel_case  # noqa: E402
from recompute_qa_v3_gateb_gold import compute  # noqa: E402


def test_event_count_gateb_preserves_gold():
    fact = {
        "mcq": {"truth_option": 4},
        "open": {"truth_value": 4},
    }
    program = {"events": [{}, {}, {}, {}]}
    result = compute(
        "card15b", {"id": "card15b"}, fact,
        {}, {}, program, {})
    assert result["status"] == "pass"
    assert result["gateb_gold"] == 4


def test_card15a_gateb_pixel_rejects_gold_outside_main_options(tmp_path):
    pixel = {
        "per_instance": {
            "source1": {"frames": [{"frame_index": 30, "state": "visible_clear"}]},
            "source2": {"frames": [{"frame_index": 30, "state": "visible_clear"}]},
            "source3": {"frames": [{"frame_index": 30, "state": "visible_clear"}]},
            "source4": {"frames": [{"frame_index": 30, "state": "out_of_view"}]},
        }
    }
    fact = {
        "open": {"truth_value": [4, 2]},
        "mcq": {"options_space": [[4, 1], [4, 2], [4, 3], [4, 4]]},
    }
    pixel_path = tmp_path / "pixel.json"
    fact_path = tmp_path / "fact.json"
    pixel_path.write_text(json.dumps(pixel))
    fact_path.write_text(json.dumps(fact))
    result = pixel_case(
        "card15a",
        {"pixel_truth": str(pixel_path), "main_fact": str(fact_path)})
    assert result["status"] == "reject"
    assert result["gateb_gold"] == [3, 2]
    assert result["rejection_reasons"] == [
        "gateb_gold_outside_main_mcq_option_space"]


def _audio_root(path, payload):
    mixture = path / "audio" / "binaural" / "mixture.wav"
    mixture.parent.mkdir(parents=True)
    mixture.write_bytes(payload)
    (path / "research_receipt.json").write_text(json.dumps({
        "status": "pass", "research_only": True,
    }))
    return path


def test_canonical_appearance_audio_requires_identical_rerenders(tmp_path):
    main = _audio_root(tmp_path / "main", b"same")
    gateb = _audio_root(tmp_path / "gateb", b"same")
    result = audio_pair({
        "main_audio": str(main), "gateb_audio": str(gateb),
        "policy": "appearance_canonical_anchor_audio_must_be_identical",
    })
    assert result["rerender_mixtures_identical"] is True
    assert result["decision"] == "pass_canonical_anchor_audio_identical"
    (gateb / "audio" / "binaural" / "mixture.wav").write_bytes(b"changed")
    import pytest
    with pytest.raises(RuntimeError, match="unexpectedly changed"):
        audio_pair({
            "main_audio": str(main), "gateb_audio": str(gateb),
            "policy": "appearance_canonical_anchor_audio_must_be_identical",
        })



def _speech_endpoint(endpoint_id, instance_id, asset_id):
    return {
        "source_endpoint_id": endpoint_id,
        "binding": {
            "entity_instance_id": instance_id,
            "entity_asset_id": asset_id,
            "emitter_anchor_id": "voice",
        },
    }


def _speech_gateb_fixture():
    main_selection = {
        "actors": [
            {"source_slot_id": "source1", "asset_id": "human_blue",
             "entity_instance_id": "person1"},
            {"source_slot_id": "source2", "asset_id": "human_red",
             "entity_instance_id": "person2"},
        ]
    }
    # The visible assets swap while the explicit entity instance bindings stay
    # attached to the source endpoints.  Endpoint list order is intentionally
    # unrelated to actor/source-slot order.
    gateb_selection = {
        "actors": [
            {"source_slot_id": "source2", "asset_id": "human_blue",
             "entity_instance_id": "person2"},
            {"source_slot_id": "source1", "asset_id": "human_red",
             "entity_instance_id": "person1"},
        ]
    }
    main_endpoints = {
        "source_endpoints": [
            _speech_endpoint("main_ep_2", "person2", "human_red"),
            _speech_endpoint("main_ep_1", "person1", "human_blue"),
        ]
    }
    gateb_endpoints = {
        "source_endpoints": [
            _speech_endpoint("gate_ep_2", "person2", "human_blue"),
            _speech_endpoint("gate_ep_1", "person1", "human_red"),
        ]
    }
    program = {
        # Candidate order is deliberately reversed; events still name the
        # concrete main endpoint they were authored against.
        "candidate_source_endpoint_ids": ["main_ep_2", "main_ep_1"],
        "events": [
            {"event_id": "event_a", "source_endpoint_id": "main_ep_1",
             "sound_asset_id": "speech_a", "start_sample": 100,
             "end_sample_exclusive": 200},
            {"event_id": "event_b", "source_endpoint_id": "main_ep_2",
             "sound_asset_id": "speech_b", "start_sample": 400,
             "end_sample_exclusive": 500},
        ],
    }
    timeline = {
        "frames": [{
            "actor_states": [
                {"source_slot_id": "source2", "asset_id": "human_blue",
                 "actor_id": "source2_actor"},
                {"source_slot_id": "source1", "asset_id": "human_red",
                 "actor_id": "source1_actor"},
            ]
        }]
    }
    rows = [
        {"slot": "source2", "sound_asset_id": "speech_b", "speaker_id": "sp",
         "utterance_id": "b", "transcript": "Beta", "colour": "red",
         "start_sample": 400, "duration_samples": 100},
        {"slot": "source1", "sound_asset_id": "speech_a", "speaker_id": "sp",
         "utterance_id": "a", "transcript": "Alpha", "colour": "blue",
         "start_sample": 100, "duration_samples": 100},
    ]
    return main_selection, gateb_selection, main_endpoints, gateb_endpoints, program, timeline, rows


def test_speech_gateb_recompute_uses_explicit_bindings():
    (main_selection, gateb_selection, main_endpoints, gateb_endpoints,
     program, timeline, rows) = _speech_gateb_fixture()
    card13 = {
        "speech_bindings": rows,
        "mcq": {"stem": "What did the person in blue say?",
                "options_space": ["Alpha", "Beta"], "truth_option": "Alpha"},
        "open": {"stem": "What did the person in blue say?",
                  "truth_value": "Alpha"},
        "question_target_colour": "blue",
    }
    card14 = {
        "speech_bindings": rows,
        "mcq": {"stem": "What colour was the person who said 'Alpha'?",
                "options_space": ["blue", "red"], "truth_option": "blue"},
        "open": {"stem": "What colour was the person who said 'Alpha'?",
                  "truth_value": "blue"},
        "question_target_transcript": "Alpha",
    }
    for pid, fact, expected in (
            ("card13", card13, "Beta"), ("card14", card14, "red")):
        result = compute(
            pid, {"id": pid}, fact, gateb_selection, timeline, program, {},
            main_selection=main_selection,
            main_endpoint_registry=main_endpoints,
            gateb_endpoint_registry=gateb_endpoints,
        )
        assert result["status"] == "pass"
        assert result["gateb_gold"] == expected
        assert result["gateb_open_gold"] == expected
        assert result["question_stem_preserved"] is True
        assert result["question_options_preserved"] is True
        assert result["speech_identity_join"]["join_key"] == (
            "binding.entity_instance_id")


def test_speech_gateb_recompute_rejects_missing_identity_join():
    (main_selection, gateb_selection, main_endpoints, gateb_endpoints,
     program, timeline, rows) = _speech_gateb_fixture()
    gateb_endpoints["source_endpoints"][0]["binding"]["entity_instance_id"] = (
        "unrelated")
    fact = {
        "speech_bindings": rows,
        "mcq": {"stem": "What did the person in blue say?",
                "options_space": ["Alpha", "Beta"], "truth_option": "Alpha"},
        "open": {"stem": "What did the person in blue say?",
                  "truth_value": "Alpha"},
        "question_target_colour": "blue",
    }
    import pytest
    with pytest.raises(ValueError, match="Gate-B"):
        compute(
            "card13", {"id": "card13"}, fact, gateb_selection, timeline,
            program, {}, main_selection=main_selection,
            main_endpoint_registry=main_endpoints,
            gateb_endpoint_registry=gateb_endpoints,
        )
