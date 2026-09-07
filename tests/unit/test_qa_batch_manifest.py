from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from avengine.qa.batch_manifest import (
    collect_batch_outcomes, grouped_splits, prepare_batch_manifest, sound_identity,
)
from avengine.rooms.conditioned_sampler import (
    CandidateFailure, neutral_source_declaration, resolve_condition_profile, select_sounds,
)


def asset(asset_id, color, *, surface="floor"):
    return {"asset_id": asset_id, "revision": "v1", "entity_class": "articulated_human",
            "identity": {"species_id": "human"},
            "realized_attributes": {"sex_or_gender_label": "male", "top_color": color},
            "display_label": asset_id, "default_emitter_anchor_id": "mouth",
            "emitter_anchors": [{"anchor_id": "mouth", "offset_m": [0, 1.6, 0],
                                 "offset_space": "final_scaled_asset_root"}],
            "runtime_backends": {"spear_unreal": {"binding": "existing"},
                                 "habitat": {"resting_pose": {"attachment_surface": surface,
                                                             "base_plane_offset_m": 0}}}}


def sound(sound_id, identity, *, count=32000):
    return {"sound_asset_id": sound_id, "sound_identity_id": identity,
            "source_pcm_path": f"/source/{identity}.wav", "sound_class": "speech",
            "gender": "male", "sample_rate_hz": 16000, "sample_count": count,
            "active_duration_s": 2, "audible_start_sample": 0, "audible_end_sample_exclusive": count,
            "transcript": sound_id, "path": f"/prepared/{sound_id}.wav"}


@pytest.fixture
def inputs():
    registry = {"assets": [asset("red", "red"), asset("blue", "blue"), asset("green", "green")]}
    rooms = {"rooms": [{"room_id": "a", "family": "authored", "renderer": "ue_spear"},
                        {"room_id": "b", "family": "authored", "renderer": "ue_spear"}]}
    sounds = [sound("one", "speaker1"), sound("one_alt", "speaker1"),
              sound("two", "speaker2"), sound("three", "speaker3")]
    config = {"batch_id": "batch", "seed": 10,
              "base_request": {"camera": {"fov_deg": 85},
                               "profile": {"separation_bin_deg": [30, 60], "anchor_count": 1}},
              "slots": [{"room_id": "a", "source_classes": ["articulated_human"] * 2,
                          "condition_group": "identity"},
                        {"room_id": "b", "source_classes": ["articulated_human"] * 2,
                          "condition_group": "after_sound", "silent_count": 1}]}
    return config, registry, rooms, sounds


def test_preallocation_is_deterministic_independent_and_preserves_inputs(inputs):
    before = deepcopy(inputs)
    first = prepare_batch_manifest(*inputs)
    second = prepare_batch_manifest(*inputs)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert inputs == before
    assert first["runtime_shared_counters"] is False
    for row in first["episodes"]:
        assert resolve_condition_profile(row["request"], inputs[1]) == row["requested_profile"]
        speakers = [actor for actor in row["source_assignments"] if actor["speaking"]]
        assert len({actor["sound_identity_id"] for actor in speakers}) == len(speakers)
        assert len(row["request"]["qa_ids"]) == 24
        assert row["achieved_conditions"] is None
    silent = [a for a in first["episodes"][1]["source_assignments"] if not a["speaking"]]
    assert silent[0]["sound_asset_ids"] == []


def test_preallocated_sound_mapping_is_actually_consumed_and_never_falls_back(inputs):
    config, registry, rooms, sounds = inputs
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    records = {a["asset_id"]: a for a in registry["assets"]}
    actors = [neutral_source_declaration(records[a["asset_id"]], a["actor_id"])
              for a in row["source_assignments"]]
    clock = {"sample_rate_hz": 16000, "sample_count": 256000}
    allowed = row["request"]["sound_selection"]["preallocated_sound_asset_ids_by_actor"]
    for seed in range(12):
        selected = select_sounds(actors, sounds, row["requested_profile"], clock,
                                 row["request"], np.random.default_rng(seed))
        assert all(s["sound_asset_id"] in allowed[actors[i]["actor_id"]] for i, s in selected.items())
    missing = deepcopy(row["request"])
    missing["sound_selection"]["preallocated_sound_asset_ids_by_actor"]["source1"] = ["absent"]
    with pytest.raises(CandidateFailure, match="no_compatible_prepared_sound_source1"):
        select_sounds(actors, sounds, row["requested_profile"], clock, missing, np.random.default_rng(1))
    del missing["sound_selection"]["preallocated_sound_asset_ids_by_actor"]["source1"]
    with pytest.raises(ValueError, match="every speaking actor"):
        select_sounds(actors, sounds, row["requested_profile"], clock, missing, np.random.default_rng(1))


def test_missing_sound_identity_retains_quota_and_failure(inputs):
    config, registry, rooms, sounds = inputs
    sounds = [sound("one", "same"), sound("two", "same")]
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    assert len(result["episodes"]) == 2
    assert result["preallocation_gap_counts"]["no_distinct_compatible_sound_identity"] == 1
    outcomes = collect_batch_outcomes(result, [{"episode_id": "batch_001", "status": "planning_failed",
                                               "failure_histogram": {"sounds:no_distinct": 200}}])
    assert outcomes["episode_denominator"] == 2
    assert outcomes["outcome_counts"] == {"planning_failed": 1, "not_run": 1}
    assert sum(sum(row["unmet_quota_by_qa"].values()) for row in outcomes["episodes"]) == 48


def test_wall_asset_remains_in_manifest_with_explicit_interface_gap(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    wall = asset("wall", None, surface="wall")
    wall["entity_class"] = "rigid_object"
    wall["identity"] = {"category": "audio_playback", "object_type": "speaker"}
    registry["assets"].append(wall)
    config["slots"][0].update(source_classes=["rigid_static_object", "articulated_human"],
                              source_asset_ids=["wall", "red"])
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    row = result["episodes"][0]
    assert row["source_assignments"][0]["appearance"]["status"] == "unknown"
    assert any(gap["state"] == "interface_not_implemented" for gap in row["preallocation_gaps"])
    assert "wall" in result["asset_inventory"]
    assert len(row["requested_quota_by_qa"]) == 24


def test_profile_substitution_does_not_satisfy_requested_quota(inputs):
    manifest = prepare_batch_manifest(*inputs)
    row = manifest["episodes"][0]
    profile = deepcopy(row["requested_profile"])
    profile["separation_bin_deg"] = [15, 30]
    outcome = {"episode_id": row["episode_id"], "status": "delivered", "facts_path": "/facts.json",
               "questions_path": "/questions.json", "condition_profile": profile,
               "produced_count_by_qa": {"QA-01": 1}}
    result = collect_batch_outcomes(manifest, [outcome])
    assert result["episodes"][0]["profile_matches_request"] is False
    assert result["episodes"][0]["unmet_quota_by_qa"]["QA-01"] == 1


def test_planned_conditions_cannot_be_reported_as_achieved_without_evidence(inputs):
    manifest = prepare_batch_manifest(*inputs)
    with pytest.raises(ValueError, match="actual evidence source"):
        collect_batch_outcomes(manifest, [{"episode_id": "batch_001", "status": "capture_failed",
                                           "achieved_conditions": {"separation": 40}}])


def test_sound_identity_groups_original_files_and_speakers():
    assert sound_identity({"speaker_id": "p123", "prepared_audio_id": "different_crop"}) == "speaker:p123"
    assert sound_identity({"source_pcm_path": "/original.wav", "sound_asset_id": "crop1"}) == "source:/original.wav"
    assert sound_identity({"sound_asset_id": "crop_without_lineage"}) is None


def row(record_id, *, episode=None, visual=None, room=None, route=None, sound_id=None, **extra):
    return {"record_id": record_id, "episode_id": episode or record_id,
            "visual_episode_id": visual or episode or record_id,
            "room_id": room or record_id, "route_ids": [route] if route else [],
            "sound_identity_ids": [sound_id or record_id], **extra}


def test_split_transitive_union_covers_all_group_axes_and_audio_variants():
    records = [
        row("a", episode="e1", room="room1", sound_id="voice1"),
        row("b", visual="e1", room="room2", sound_id="voice2"),
        row("c", room="room2", route="routeX", sound_id="voice3"),
        row("d", room="room3", route="routeX", sound_id="voice4"),
        row("e", room="room4", sound_id="voice4"),
        row("f", room="room5", sound_id="voice5"),
    ]
    result = grouped_splits(records, ratios={"train": 0.8, "eval": 0.2}, seed=8)
    assert result["record_denominator"] == 6
    by_id = {record["record_id"]: record for record in result["records"]}
    assert len({by_id[name]["group_id"] for name in "abcde"}) == 1
    assert len({by_id[name]["split"] for name in "abcde"}) == 1
    assert by_id["f"]["group_id"] != by_id["a"]["group_id"]
    assert result == grouped_splits(records, ratios={"train": 0.8, "eval": 0.2}, seed=8)


def test_split_keeps_unknown_identity_records_and_their_connected_group_unassigned():
    records = [row("a", room="shared"), row("b", room="shared"), row("c")]
    records[1].pop("sound_identity_ids")
    result = grouped_splits(records, ratios={"train": 1, "eval": 1})
    assert result["unassigned_count"] == 2
    assert result["record_denominator"] == 3
    assert sum(result["actual_counts"].values()) == 1


def test_split_does_not_force_requested_proportions_by_breaking_a_group():
    result = grouped_splits([row("a", room="shared"), row("b", room="shared")],
                            ratios={"train": 1, "eval": 1})
    assert sorted(result["actual_counts"].values()) == [0, 2]


def test_split_rejects_duplicate_record_ids_and_invalid_ratios():
    with pytest.raises(ValueError, match="duplicate"):
        grouped_splits([row("a"), row("a")], ratios={"train": 1})
    with pytest.raises(ValueError, match="positive"):
        grouped_splits([row("a")], ratios={"train": 1, "eval": 0})


def test_fixed_repeat_identities_keep_their_unmet_budget_instead_of_substitution(inputs):
    config, registry, rooms, _ = deepcopy(inputs)
    config["slots"] = [config["slots"][0]]
    config["slots"][0]["profile"] = {"event_relation": "repeat"}
    sounds = [sound("long_one", "speaker1", count=64000),
              sound("long_two", "speaker2", count=70528)]
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    episode = result["episodes"][0]
    assert episode["requested_profile"]["event_relation"] == "repeat"
    assert {row["sound_identity_id"] for row in episode["source_assignments"]} == {"speaker1", "speaker2"}
    gap = next(row for row in episode["preallocation_gaps"]
               if row["code"] == "fixed_sound_identities_exceed_profile_clip_budget")
    assert gap["minimum_program_seconds_under_sampler_clip_budget"] > 13
    assert gap["identity_substitution_applied"] is False
    assert len(episode["requested_quota_by_qa"]) == 24
