"""Focused tests for the QA-v3 extended profile execution layer."""

from __future__ import annotations

import json

import numpy as np
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from design_qa_v3_extended_profile import (  # noqa: E402
    SUPPORTED,
    CARD11_BINDING_FRAME,
    CARD11_EVENT_START_SAMPLE,
    _assert_gateb_visual_change,
    _facts,
    _find_gateb_out_of_view_route,
    _program_events,
    _resource_inventory,
    _speech_program_events,
    _speech_question_context,
    audio_program_mode,
)


def _repo_json(path):
    return json.loads((Path(__file__).resolve().parents[1] / path).read_text())


def test_catalog_contains_all_twenty_one_profiles_once():
    profiles = _repo_json("examples/qa/qa_v3_current_profiles_v1.json")
    ids = [item["id"] for item in profiles]
    assert len(ids) == len(set(ids)) == 21
    assert SUPPORTED <= set(ids)
    for profile in profiles:
        if profile["id"] in SUPPORTED:
            assert profile["execution_backend"] == "extended"


def test_current_registry_exposes_semantic_asset_shortfalls():
    assets = _repo_json(
        "examples/runtime/source_asset_runtime_profiles.json")["assets"]
    sounds = _repo_json(
        "examples/registry/registries/sound_assets_v1.json")["sound_assets"]
    assert _resource_inventory("card11", assets, sounds)["missing"] == []
    assert _resource_inventory("card15a", assets, sounds)["missing"] == []
    assert _resource_inventory("card16", assets, sounds)["missing"] == []
    assert _resource_inventory("card17", assets, sounds)["missing"] == []
    assert _resource_inventory("card12", assets, sounds)["missing"] == [
        "four_registered_semantic_sound_types"]
    speech_missing = _resource_inventory("card13", assets, sounds)["missing"]
    assert speech_missing == ["transcribed_speech_assets"]


def test_card15a_gatea_changes_distinct_callers_not_event_times():
    main, gatea, truth = _program_events(
        "card15a", 0, [{"sound_asset_id": "dog_beagle_v2_scheduled_dry"}])
    assert len({event[0] for event in main}) == 1
    assert len({event[0] for event in gatea}) == 4
    assert [event[1] for event in main] == [event[1] for event in gatea]
    assert truth["distinct_callers"] == 1
    assert truth["gatea_distinct_callers"] == 4


def test_card11_audio_spans_the_native_pixel_binding_frame():
    main, gatea, _ = _program_events(
        "card11", 0, [{"sound_asset_id": "dog_beagle_v2_scheduled_dry"}])
    assert main[0][1] == gatea[0][1] == CARD11_EVENT_START_SAMPLE
    samples_per_frame = 16000 / 15
    first_frame = int(CARD11_EVENT_START_SAMPLE // samples_per_frame)
    last_frame_exclusive = int(
        -(-(CARD11_EVENT_START_SAMPLE + 4800) // samples_per_frame))
    assert first_frame <= CARD11_BINDING_FRAME < last_frame_exclusive


def test_card16_gatea_flips_first_caller_with_structure_preserved():
    main, gatea, truth = _program_events(
        "card16", 0, [{"sound_asset_id": "dog_beagle_v2_scheduled_dry"}])
    assert [event[1:] for event in main] == [event[1:] for event in gatea]
    assert main[0][0] == "source1"
    assert gatea[0][0] == "source2"
    assert truth["first_caller_slot"] == "source1"
    assert truth["gatea_first_caller_slot"] == "source2"


def _synthetic_sound(index, *, speech=False):
    value = {
        "sound_asset_id": f"sound_{index}",
        "semantic_sound_class": "human_speech" if speech else "test_signal",
        "taxonomy_path": ["synthetic", f"class_{index}"],
    }
    if speech:
        value["transcript"] = f"sentence {index}"
    return value


def test_asset_ready_card12_builds_four_sound_mcq_and_gatea_swap():
    inventory = {
        "sound_types": [_synthetic_sound(index) for index in range(4)],
        "speech": [], "humans": [],
        "dogs": [{"display_label": f"dog {index}"} for index in range(4)],
        "missing": [],
    }
    main, gatea, truth = _program_events(
        "card12", 0, inventory["sound_types"])
    assert sorted(event[2] for event in main) == sorted(
        event[2] for event in gatea)
    assert main[0][2] == gatea[1][2]
    fact = _facts("card12", inventory, truth, None, None)
    assert len(fact["mcq"]["options_space"]) == 4
    assert fact["mcq"]["truth_option"] == "class_0"


def test_asset_ready_speech_profiles_use_four_unique_colours_and_transcripts():
    humans = [
        {"realized_attributes": {"top_color": colour}}
        for colour in ["blue", "burgundy", "green", "yellow"]
    ]
    speech = [_synthetic_sound(index, speech=True) for index in range(4)]
    inventory = {
        "sound_types": [], "speech": speech, "humans": humans,
        "dogs": [], "missing": [],
    }
    for profile_id in ["card13", "card14"]:
        main, gatea, truth = _program_events(profile_id, 0, speech)
        assert sorted(event[2] for event in main) == sorted(
            event[2] for event in gatea)
        fact = _facts(profile_id, inventory, truth, None, None)
        assert len(fact["mcq"]["options_space"]) == 4
        assert fact["mcq"]["truth_option"] in fact["mcq"]["options_space"]


def test_resource_inventory_deduplicates_semantic_values():
    humans = [
        {
            "asset_id": f"human_{index}",
            "identity": {"species_id": "human"},
            "realized_attributes": {"top_color": colour},
        }
        for index, colour in enumerate(["blue", "blue", "green", "yellow"])
    ]
    speech = [_synthetic_sound(0, speech=True) for _ in range(4)]
    result = _resource_inventory("card13", humans, speech)
    assert result["missing"] == [
        "controlled_human_top_colours",
        "transcribed_speech_assets",
    ]


def _selection_for_tracks(first_asset, second_asset):
    return {
        "actors": [
            {"source_slot_id": "source1", "asset_id": first_asset},
            {"source_slot_id": "source2", "asset_id": second_asset},
        ]
    }


def _timeline_for_tracks(first_x, second_x):
    return {
        "frames": [
            {
                "actor_states": [
                    {"source_slot_id": "source1",
                     "translation_ue_cm": [first_x, 0.0, 0.0]},
                    {"source_slot_id": "source2",
                     "translation_ue_cm": [second_x, 0.0, 0.0]},
                ]
            }
            for _ in range(2)
        ]
    }


def test_gateb_rejects_slot_relabel_noop_and_accepts_visual_change():
    main_selection = _selection_for_tracks("asset_a", "asset_b")
    main_timeline = _timeline_for_tracks(1.0, 2.0)
    reversed_selection = _selection_for_tracks("asset_b", "asset_a")
    reversed_routes = _timeline_for_tracks(2.0, 1.0)
    import pytest
    with pytest.raises(RuntimeError, match="only slot labels"):
        _assert_gateb_visual_change(
            main_selection, main_timeline,
            reversed_selection, reversed_routes)
    changed = _assert_gateb_visual_change(
        main_selection, main_timeline,
        reversed_selection, main_timeline)
    assert changed["per_asset_tracks_changed"] is True


def test_extended_targets_rotate_across_visible_candidates():
    bark = [{"sound_asset_id": "dog_beagle_v2_scheduled_dry"}]
    card11_targets = [
        _program_events("card11", index, bark)[2]["target_slot"]
        for index in range(6)
    ]
    assert card11_targets == [
        "source1", "source1", "source2",
        "source2", "source3", "source3"]
    sounds = [_synthetic_sound(index) for index in range(4)]
    for profile_id in ["card12", "card13", "card14"]:
        targets = [
            _program_events(profile_id, index, sounds)[2]["target_index"]
            for index in range(4)
        ]
        assert targets == [0, 1, 2, 3]


def _manifest_code_is_traceable(manifest):
    code = manifest["code"]
    assert len(code["revision"]) == 40 and int(code["revision"], 16) >= 0
    assert isinstance(code["dirty"], bool)
    assert isinstance(code["status"], list)


def test_unavailable_manifest_records_the_code_revision(tmp_path):
    from types import SimpleNamespace
    from design_qa_v3_extended_profile import _write_unavailable
    manifest = _write_unavailable(
        tmp_path / "batch", {"id": "card12"}, SimpleNamespace(scene_id="room"),
        ["four_registered_semantic_sound_types"], 3)
    _manifest_code_is_traceable(manifest)
    on_disk = json.loads((tmp_path / "batch" / "batch_manifest.json").read_text())
    assert on_disk["code"] == manifest["code"]
    assert on_disk["evidence_class"] == "resource_unavailable"


def test_failed_manifest_records_code_and_the_error(tmp_path):
    from design_qa_v3_extended_profile import _write_failed
    out = tmp_path / "batch"
    out.mkdir()
    manifest = _write_failed(
        out, "card16", "room", RuntimeError("engine handshake lost"),
        cells_requested=4, completed=1)
    _manifest_code_is_traceable(manifest)
    on_disk = json.loads((out / "batch_manifest.json").read_text())
    assert on_disk["status"] == "failed"
    assert on_disk["evidence_class"] == "runner_failure"
    assert on_disk["failure"] == {"type": "RuntimeError",
                                  "detail": "engine handshake lost"}
    assert on_disk["counts"] == {"cells_requested": 4,
                                 "geometry_candidates": 1, "rejected": 0}
    assert on_disk["qualification_claim"] is False


def test_gateb_search_requires_min_camera_distance():
    with pytest.raises(ValueError, match="MIN_CAMERA_DISTANCE_CM"):
        _find_gateb_out_of_view_route(
            object(), {}, {"routes": [], "camera_xy": (0.0, 0.0),
                           "camera_yaw_deg": 0.0})

def test_speech_program_events_keep_full_windows_and_swap_gatea_slots():
    from audio_profiles import schedule_speech_utterances
    from avengine.assets.sound_pool import PoolClip

    clips = [
        PoolClip(
            sound_asset_id=f"speech_{index}",
            event_class="speech_playback",
            duration_samples=16000 + index * 1000,
            sample_rate_hz=16000,
            source_start_sample=0,
            source_end_sample_exclusive=16000 + index * 1000,
            speaker_id=f"p{index}",
            utterance_id=f"{index:03d}",
            transcript=f"sentence {index}",
            split="train",
        )
        for index in range(4)
    ]

    class Source:
        def select_distinct_speech_clips(self, count=4, *, split="train", **selection_options):
            assert split == "train"
            return clips[:count]

    params = {
        "CLIP_SECONDS": 10.0,
        "SAMPLE_RATE_HZ": 16000,
        "FRAME_COUNT": 150,
        "TICKS_PER_SAMPLE": 3,
        "TICKS_PER_FRAME": 3200,
        "GAP_MIN_S": 0.3,
    }
    schedule = schedule_speech_utterances(
        np.random.default_rng(1),
        params=params,
        clip_source=Source(),
        roles=["source1", "source2", "source3", "source4"],
    )
    main, gatea, truth = _speech_program_events(schedule, 0)
    assert audio_program_mode(main) == "sequential_sources"
    assert [row["start_sample"] for row in gatea] == [
        row["start_sample"] for row in main
    ]
    assert [row["duration_samples"] for row in gatea] == [
        row["duration_samples"] for row in main
    ]
    assert [row["sound_asset_id"] for row in gatea] == [
        row["sound_asset_id"] for row in main
    ]
    assert [row["slot"] for row in gatea] != [row["slot"] for row in main]
    assert truth["target_speech_utterance_id"] == main[0]["utterance_id"]


def test_speech_facts_bind_transcript_to_colour_and_preserve_metadata():
    events = [
        {
            "slot": f"source{index + 1}",
            "sound_asset_id": f"speech_{index}",
            "speaker_id": f"p{index}",
            "utterance_id": f"{index:03d}",
            "transcript": f"sentence {index}",
            "split": "train",
            "start_sample": index * 20000,
            "duration_samples": 16000,
        }
        for index in range(4)
    ]
    colours = {
        "source1": "blue",
        "source2": "burgundy",
        "source3": "green",
        "source4": "yellow",
    }
    inventory = {"humans": [], "speech": [], "dogs": [], "sound_types": []}
    for profile_id in ("card13", "card14"):
        fact = _facts(
            {"id": profile_id},
            inventory,
            {"target_index": 2},
            None,
            None,
            speech_events=events,
            colour_by_slot=colours,
        )
        assert fact["target_speaker_id"] == "p2"
        assert fact["target_utterance_id"] == "002"
        assert fact["speech_bindings"][2]["colour"] == "green"
        assert fact["speech_bindings"][2]["split"] == "train"
        assert fact["mcq"]["truth_option"] in fact["mcq"]["options_space"]
        assert fact["open"]["truth_value"] == fact["mcq"]["truth_option"]


def test_speech_pool_inventory_uses_train_pool_rows():
    rows = [
        {
            "event_class": "speech_playback",
            "speaker_id": f"p{index}",
            "utterance_id": f"{index:03d}",
            "transcript": f"sentence {index}",
            "split": "train",
            "sound_asset_id": f"speech_{index}",
        }
        for index in range(4)
    ]
    rows.append({
        "event_class": "speech_playback",
        "speaker_id": "eval",
        "utterance_id": "999",
        "transcript": "eval sentence",
        "split": "eval",
        "sound_asset_id": "eval_speech",
    })
    inventory = _resource_inventory(
        "card13",
        [
            {"identity": {"species_id": "human"},
             "realized_attributes": {"top_color": colour}}
            for colour in ("blue", "burgundy", "green", "yellow")
        ],
        [],
        speech_pool=rows,
    )
    assert len(inventory["speech"]) == 4
    assert {row["split"] for row in inventory["speech"]} == {"train"}
    assert inventory["missing"] == []



def _speech_test_schedule():
    from audio_profiles import Schedule, ScheduledEvent

    events = [
        ScheduledEvent(
            role=f"source{index + 1}",
            start_sample=index * 20000,
            end_sample_exclusive=index * 20000 + 8000,
            purpose="answer_evidence",
            sample_rate_hz=16000,
            ticks_per_sample=3,
            ticks_per_frame=3200,
            frame_count=75,
            sound_asset_id=f"speech_{index}",
            source_start_sample=0,
            source_end_sample_exclusive=8000,
            speaker_id=f"p{index}",
            utterance_id=f"{index:03d}",
            transcript=f"sentence {index}",
            split="train",
        )
        for index in range(4)
    ]
    return Schedule(
        "speech_utterances",
        events,
        0,
        {"role_order": [f"source{index + 1}" for index in range(4)]},
    )


def test_speech_gatea_preserves_question_and_recomputes_binding_gold():
    colours = {
        "source1": "red",
        "source2": "blue",
        "source3": "green",
        "source4": "yellow",
    }
    inventory = {"humans": [], "speech": [], "dogs": [], "sound_types": []}
    schedule = _speech_test_schedule()
    observed = {}
    for profile_id in ("card13", "card14"):
        profile_observed = []
        for seed in (11, 12):
            target_rng = np.random.default_rng(seed)
            option_rng = np.random.default_rng(seed + 100)
            main, gatea, truth = _speech_program_events(
                schedule,
                0,
                target_rng=target_rng,
                option_rng=option_rng,
            )
            truth["speech_question"] = _speech_question_context(
                main, colours, truth
            )
            main_fact = _facts(
                {"id": profile_id},
                inventory,
                truth,
                None,
                None,
                speech_events=main,
                colour_by_slot=colours,
            )
            gatea_fact = _facts(
                {"id": profile_id},
                inventory,
                truth,
                None,
                None,
                speech_events=gatea,
                colour_by_slot=colours,
            )
            assert main_fact["mcq"]["stem"] == gatea_fact["mcq"]["stem"]
            assert (
                main_fact["mcq"]["options_space"]
                == gatea_fact["mcq"]["options_space"]
            )
            assert main_fact["mcq"]["truth_option"] != gatea_fact["mcq"]["truth_option"]
            assert main_fact["open"]["truth_value"] != gatea_fact["open"]["truth_value"]
            if profile_id == "card13":
                assert main_fact["question_target_slot"] == gatea_fact["question_target_slot"]
                assert main_fact["target_slot"] == gatea_fact["target_slot"]
                assert main_fact["target_utterance_id"] != gatea_fact["target_utterance_id"]
            else:
                assert (
                    main_fact["question_target_transcript"]
                    == gatea_fact["question_target_transcript"]
                )
                assert (
                    main_fact["target_utterance_id"]
                    == gatea_fact["target_utterance_id"]
                )
                assert main_fact["target_slot"] != gatea_fact["target_slot"]
            position = main_fact["mcq"]["options_space"].index(
                main_fact["mcq"]["truth_option"]
            )
            profile_observed.append(
                (
                    truth["target_index"],
                    tuple(main_fact["mcq"]["options_space"]),
                    position,
                )
            )
        assert len({row[0] for row in profile_observed}) > 1
        assert len({row[2] for row in profile_observed}) > 1
        observed[profile_id] = profile_observed
    assert observed["card13"] != observed["card14"]


def test_speech_fact_rejects_same_text_under_different_utterance_ids():
    schedule = _speech_test_schedule()
    main, _, truth = _speech_program_events(schedule, 0)
    main[1]["transcript"] = main[0]["transcript"]
    colours = {f"source{i+1}": f"colour{i}" for i in range(4)}
    for profile, reason in (("card13", "duplicate transcript"),
                            ("card14", "not unique among speakers")):
        with pytest.raises(ValueError, match=reason):
            _facts({"id": profile}, {}, truth, None, None,
                   speech_events=main, colour_by_slot=colours)


def test_audio_search_exhaustion_does_not_leave_partial_point(tmp_path, monkeypatch):
    import design_qa_v3_extended_profile as module
    from audio_profiles import AudioProfileSearchExhausted
    def exhausted(*args, **kwargs):
        raise AudioProfileSearchExhausted("bounded audio search ended", attempts=1)
    monkeypatch.setattr(module, "_speech_schedule", exhausted)
    with pytest.raises(AudioProfileSearchExhausted):
        module._realise_cell(tmp_path, {"id": "card13"}, 0, None, {}, {}, {},
                             tmp_path / "registry.json", {}, "unused", "seed")
    assert not list(tmp_path.iterdir())


def test_speech_inventory_uses_profile_actor_count():
    inventory = _resource_inventory("card13", [
        {"identity": {"species_id": "human"}, "realized_attributes": {"top_color": c}}
        for c in ("blue", "green")], [], speech_pool=[
        {"event_class": "speech_playback", "speaker_id": f"p{i}",
         "utterance_id": str(i), "transcript": f"sentence {i}", "split": "train"}
        for i in range(2)], profile={"id": "card13", "actor_count": 2})
    assert inventory["missing"] == []
    assert inventory["requirements"]["required_transcripts"] == 2


def test_speech_frame_clock_does_not_round_away_a_duration_conflict():
    import design_qa_v3_extended_profile as module
    with pytest.raises(ValueError, match="FRAME_COUNT"):
        module._timeline_dimensions({"CLIP_SECONDS": 5.01, "VIDEO_FPS": 15, "FRAME_COUNT": 75})
    assert module._timeline_dimensions({"CLIP_SECONDS": 5, "VIDEO_FPS": 15, "FRAME_COUNT": 75}) == (75, 15)
    assert module._timeline_dimensions({"CLIP_SECONDS": 10, "VIDEO_FPS": 15, "FRAME_COUNT": 150}) == (150, 15)
