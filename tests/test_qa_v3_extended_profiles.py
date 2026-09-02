"""Focused tests for the QA-v3 extended profile execution layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from design_qa_v3_extended_profile import (  # noqa: E402
    SUPPORTED,
    CARD11_BINDING_FRAME,
    CARD11_EVENT_START_SAMPLE,
    _assert_gateb_visual_change,
    _facts,
    _program_events,
    _resource_inventory,
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
    assert speech_missing == [
        "four_controlled_human_top_colours",
        "four_transcribed_speech_assets",
    ]


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
        "four_controlled_human_top_colours",
        "four_transcribed_speech_assets",
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
