from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.qa.batch_manifest import (
    CLASS_PAIRS, CONDITION_GROUP_BY_TASK_FAMILY, CONDITION_GROUPS, build_scaleup_slots,
    class_pair_condition_group_crosstab, collect_batch_outcomes, core_group_from_manifest,
    entity_instances_for_slot, format_class_pair_condition_group_crosstab,
    group_blockers_for_group, grouped_splits, merge_request_overrides,
    prepare_batch_manifest, production_config_slots, program_seconds_for_durations,
    resolve_qa_plan, resolve_qa_targets, scatter_condition_groups, shared_asset_instance_gap,
    sound_identity, stage_work_items_for_group, stage_work_items_for_row,
)
from avengine.dataset.production_spec import SCHEMA as PRODUCTION_SPEC_SCHEMA
from avengine.qa.unified_catalog import QA_IDS
from avengine.rooms.conditioned_sampler import histogram_separation_5deg
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
        assert len(row["request"]["qa_ids"]) == 25
        assert row["achieved_conditions"] is None
    silent = [a for a in first["episodes"][1]["source_assignments"] if not a["speaking"]]
    assert silent[0]["sound_asset_ids"] == []


def test_request_overrides_merge_replaces_lists_and_scalars_without_mutating_base():
    base = {
        "runtime": {
            "uproject": "base.uproject",
            "path_bindings": {"COMMON": "/common", "BASE_ONLY": "/base"},
        },
        "camera": {"fov_deg": 85, "resolution_hw": [720, 1280]},
        "sound_selection": {"max_clip_s": 5.0, "classes": ["speech", "event"]},
    }
    overrides = {
        "runtime": {"path_bindings": {"SLOT_ONLY": "/slot"}},
        "camera": {"resolution_hw": [480, 640]},
        "sound_selection": {"classes": ["animal"]},
    }
    before = deepcopy(base)
    merged = merge_request_overrides(base, overrides)

    assert merged["runtime"] == {
        "uproject": "base.uproject",
        "path_bindings": {
            "COMMON": "/common", "BASE_ONLY": "/base", "SLOT_ONLY": "/slot"
        },
    }
    assert merged["camera"] == {"fov_deg": 85, "resolution_hw": [480, 640]}
    assert merged["sound_selection"] == {"max_clip_s": 5.0, "classes": ["animal"]}
    merged["runtime"]["path_bindings"]["COMMON"] = "/changed"
    assert base == before


def test_request_overrides_merge_keeps_base_siblings_per_slot(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["base_request"].update({
        "runtime": {
            "uproject": "base.uproject",
            "runtime_prefix": "base-prefix",
            "graphics_adapter": 3,
            "path_bindings": {"COMMON": "/common", "BASE_ONLY": "/base"},
        },
        "camera": {"fov_deg": 85, "resolution_hw": [720, 1280]},
        "sound_selection": {"max_clip_s": 5.0, "classes": ["speech", "event"]},
        "qa_sampling": {
            "candidate_policy": "uniform_over_legal",
            "query_time_policy": "uniform",
            "bins": [1, 2],
        },
        "profile": {"separation_bin_deg": [30, 60], "anchor_count": 1},
    })
    config["slots"] = [
        {
            "room_id": "a",
            "source_classes": ["articulated_human"] * 2,
            "condition_group": "identity",
            "request_overrides": {
                "runtime": {
                    "graphics_adapter": 1,
                    "path_bindings": {"SLOT_ONLY": "/slot"},
                },
                "camera": {"fov_deg": 70, "resolution_hw": [480, 640]},
                "sound_selection": {"max_clip_s": 4.0},
                "qa_sampling": {"candidate_policy": "slot"},
                "profile": {"separation_bin_deg": [15, 30]},
            },
        },
        {
            "room_id": "b",
            "source_classes": ["articulated_human"] * 2,
            "condition_group": "identity",
            "request_overrides": {"runtime": {"rpc_port": 40002}},
        },
    ]
    before = deepcopy(config)
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    first, second = result["episodes"]

    assert first["request"]["runtime"] == {
        "uproject": "base.uproject",
        "runtime_prefix": "base-prefix",
        "graphics_adapter": 1,
        "path_bindings": {
            "COMMON": "/common", "BASE_ONLY": "/base", "SLOT_ONLY": "/slot"
        },
    }
    assert second["request"]["runtime"] == {
        "uproject": "base.uproject",
        "runtime_prefix": "base-prefix",
        "graphics_adapter": 3,
        "rpc_port": 40002,
        "path_bindings": {"COMMON": "/common", "BASE_ONLY": "/base"},
    }
    assert first["request"]["camera"] == {
        "fov_deg": 70, "resolution_hw": [480, 640], "motion": "static"
    }
    assert second["request"]["camera"] == {
        "fov_deg": 85, "resolution_hw": [720, 1280], "motion": "static"
    }
    assert first["request"]["sound_selection"]["max_clip_s"] == 4.0
    assert second["request"]["sound_selection"]["max_clip_s"] == 5.0
    assert first["request"]["sound_selection"]["classes"] == ["speech", "event"]
    assert second["request"]["sound_selection"]["classes"] == ["speech", "event"]
    assert first["request"]["qa_sampling"] == {
        "candidate_policy": "slot", "query_time_policy": "uniform", "bins": [1, 2],
        "items_per_type": 1,
    }
    assert second["request"]["qa_sampling"] == {
        "candidate_policy": "uniform_over_legal", "query_time_policy": "uniform",
        "bins": [1, 2], "items_per_type": 1,
    }
    assert first["request"]["profile"]["separation_bin_deg"] == [15, 30]
    assert second["request"]["profile"]["separation_bin_deg"] == [30, 60]
    assert config == before


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
    assert sum(sum(row["unmet_quota_by_qa"].values()) for row in outcomes["episodes"]) == 54


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
    assert len(row["requested_quota_by_qa"]) == 25


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
    assert len(episode["requested_quota_by_qa"]) == 25


def test_repeat_legal_pair_is_accepted_without_deficit(inputs):
    config, registry, rooms, _ = deepcopy(inputs)
    config["slots"] = [config["slots"][0]]
    config["slots"][0]["profile"] = {"event_relation": "repeat"}
    sounds = [sound("short_one", "speaker1", count=32000), sound("short_two", "speaker2", count=32000)]
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    episode = result["episodes"][0]
    assert episode["requested_profile"]["event_relation"] == "repeat"
    assert not any(gap["code"] == "fixed_sound_identities_exceed_profile_clip_budget"
                   for gap in episode["preallocation_gaps"])
    assert {row["sound_identity_id"] for row in episode["source_assignments"]} == {"speaker1", "speaker2"}
    assert episode["request"]["sound_selection"].get("repeat_actor_id") in {"source1", "source2"}


def test_repeat_over_budget_pair_is_replaced_by_legal_identities(inputs):
    config, registry, rooms, _ = deepcopy(inputs)
    registry["assets"].append(asset("yellow", "yellow"))
    config["slots"] = [
        {"room_id": "a", "source_classes": ["articulated_human"] * 2, "condition_group": "identity_binding"},
        {"room_id": "a", "source_classes": ["articulated_human"] * 2, "condition_group": "identity_binding",
         "profile": {"event_relation": "repeat"}},
    ]
    sounds = [
        sound("short_a", "speaker_short_a", count=32000),
        sound("short_b", "speaker_short_b", count=32000),
        sound("long_red", "speaker_long_red", count=72000),
        sound("long_blue", "speaker_long_blue", count=72000),
    ]
    sounds[2]["compatible_asset_ids"] = ["red"]
    sounds[3]["compatible_asset_ids"] = ["blue"]
    config["slots"][0]["source_asset_ids"] = ["green", "yellow"]
    config["slots"][1]["source_asset_ids"] = ["red", "blue"]
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    first, second = result["episodes"]
    assert first["requested_profile"]["event_relation"] != "repeat"
    assert {row["sound_identity_id"] for row in first["source_assignments"]} == {
        "speaker_short_a", "speaker_short_b"}
    assert second["requested_profile"]["event_relation"] == "repeat"
    assert not any(gap["code"] == "fixed_sound_identities_exceed_profile_clip_budget"
                   for gap in second["preallocation_gaps"])
    assigned = [row["sound_identity_id"] for row in second["source_assignments"]]
    assert set(assigned) != {"speaker_long_red", "speaker_long_blue"}
    assert second["request"]["sound_selection"].get("identity_substitution_applied") is True
    duration = {"speaker_short_a": 2.0, "speaker_short_b": 2.0,
                "speaker_long_red": 4.5, "speaker_long_blue": 4.5}
    durs = [duration[name] for name in assigned]
    legal = [program_seconds_for_durations(durs, gap_s=0.5, relation="repeat", repeat_index=i) <= 13 + 1e-9
             for i in range(len(durs))]
    assert any(legal)


def test_sequential_over_budget_replaces_only_sound_identities_with_legal_pair(inputs):
    config, registry, rooms, _ = deepcopy(inputs)
    config["seed"] = 1
    config["base_request"]["frame_count"] = 150
    config["base_request"]["frame_rate_hz"] = 15
    config["slots"] = [{
        "room_id": "a",
        "source_classes": ["articulated_human"] * 2,
        "source_asset_ids": ["red", "blue"],
        "condition_group": "identity",
        "profile": {
            "event_relation": "sequential",
            "reserve_tail_s": 3.0,
            "min_gap_between_audible_windows_s": 0.5,
        },
    }]
    sounds = [
        sound("long_red", "long_red", count=76800),
        sound("short_red", "short_red", count=32000),
        sound("long_blue", "long_blue", count=76800),
        sound("short_blue", "short_blue", count=32000),
    ]
    sounds[0]["compatible_asset_ids"] = ["red"]
    sounds[1]["compatible_asset_ids"] = ["red"]
    sounds[2]["compatible_asset_ids"] = ["blue"]
    sounds[3]["compatible_asset_ids"] = ["blue"]

    result = prepare_batch_manifest(config, registry, rooms, sounds)
    episode = result["episodes"][0]
    assert episode["request"]["source_asset_ids"] == ["red", "blue"]
    assert [row["sound_identity_id"] for row in episode["source_assignments"]] == [
        "short_red", "short_blue"
    ]
    assert not any(gap["code"] == "fixed_sound_identities_exceed_profile_clip_budget"
                   for gap in episode["preallocation_gaps"])
    assert episode["request"]["sound_selection"]["identity_substitution_applied"] is True

    records = {record["asset_id"]: record for record in registry["assets"]}
    actors = [neutral_source_declaration(records[asset_id], f"source{index + 1}")
              for index, asset_id in enumerate(("red", "blue"))]
    selected = select_sounds(
        actors,
        sounds,
        episode["requested_profile"],
        {"sample_rate_hz": 16000, "sample_count": 160000},
        episode["request"],
        np.random.default_rng(2),
    )
    assert [selected[index]["sound_asset_id"] for index in range(2)] == [
        "short_red", "short_blue"
    ]


def test_class_pair_crosstab_covers_at_least_three_groups():
    rooms = [{"room_id": f"room_{i}", "family": fam}
             for i, fam in enumerate(("authored", "apartment", "hm3d", "mp3d"))]
    slots = build_scaleup_slots(rooms, seed=20260907, episodes_per_room=20)
    table = class_pair_condition_group_crosstab(slots)
    assert set(table["class_pairs"]) >= {
        "human-human", "animal-human", "device-human", "animal-animal", "animal-device", "device-device"}
    assert table["min_distinct_groups_per_class_pair"] >= 3
    assert table["meets_acceptance"] is True
    text = format_class_pair_condition_group_crosstab(table)
    assert "human-human" in text and "identity_binding" in text


def test_collect_outcomes_histograms_achieved_5deg_not_requested_bin(inputs):
    manifest = prepare_batch_manifest(*inputs)
    row = manifest["episodes"][0]
    outcome = {
        "episode_id": row["episode_id"], "status": "delivered",
        "facts_path": "/facts.json", "questions_path": "/questions.json",
        "condition_profile": deepcopy(row["requested_profile"]),
        "achieved_conditions_source": "native_pixel_and_pcm",
        "achieved_conditions": {"separation": {"status": "measured", "min": 60.7}},
        "produced_count_by_qa": {qa: 1 for qa in row["requested_quota_by_qa"]},
    }
    result = collect_batch_outcomes(manifest, [outcome])
    hist = result["achieved_separation_histogram_5deg"]
    assert hist["requested_bin_is_not_coverage"] is True
    assert result["separation_coverage_unit"] == "achieved_angle_5deg_bins"
    occupied = {item["lo_deg"]: item["count"] for item in hist["occupied_bins"]}
    assert occupied == {60: 1}
    assert 60.7 not in [row["requested_profile"]["separation_bin_deg"][0]
                        for row in manifest["episodes"]]


def device_asset(asset_id, color="black"):
    record = asset(asset_id, color)
    record["entity_class"] = "rigid_object"
    record["identity"] = {"category": "appliance", "object_type": "speaker"}
    record["realized_attributes"] = {"finish": color}
    return record


def animal_asset(asset_id, coat="brown"):
    record = asset(asset_id, coat)
    record["entity_class"] = "articulated_animal"
    record["identity"] = {"species_id": "dog"}
    record["realized_attributes"] = {"coat_profile": {"value": coat}}
    return record


def test_rescatter_keeps_scaleup_distance_range_m(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["scatter_condition_groups"] = True
    config["scaleup"] = {"distance_range_m": [1.5, 6.0]}
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    for row in result["episodes"]:
        assert row["request"]["profile"]["distance_range_m"] == [1.5, 6.0]
        assert row["requested_profile"]["distance_range_m"] == [1.5, 6.0]


def test_rescatter_does_not_drop_existing_slot_distance_range(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["scatter_condition_groups"] = True
    for slot in config["slots"]:
        slot["profile"] = {**(slot.get("profile") or {}), "distance_range_m": [1.5, 6.0]}
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    for row in result["episodes"]:
        assert row["request"]["profile"]["distance_range_m"] == [1.5, 6.0]
        assert row["requested_profile"]["distance_range_m"] == [1.5, 6.0]


def test_zero_compatible_sounds_is_reported_as_no_sounds(inputs):
    config, registry, rooms, _ = deepcopy(inputs)
    sounds = [{
        "sound_asset_id": "ac_hum", "sound_identity_id": "ac1",
        "sound_class": "air_conditioning", "sample_rate_hz": 16000, "sample_count": 16000,
        "compatible_asset_ids": ["missing_portable_ac"],
        "path": "/prepared/ac_hum.wav",
    }]
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    assert result["preallocation_gap_counts"].get("no_compatible_sounds", 0) >= 1
    assert "no_distinct_compatible_sound_identity" not in result["preallocation_gap_counts"]
    gap = next(item for row in result["episodes"] for item in row["preallocation_gaps"]
               if item["code"] == "no_compatible_sounds")
    assert gap["compatible_sound_count"] == 0
    assert gap["state"] == "evidence_missing_or_unsampled"


def test_collect_outcomes_keeps_source_classes_and_old_profile_defaults(inputs):
    manifest = prepare_batch_manifest(*deepcopy(inputs))
    row = manifest["episodes"][0]
    old_profile = deepcopy(row["requested_profile"])
    for key in ("competitor_visibility", "distance_range_m", "separation_target_policy"):
        old_profile.pop(key, None)
    manifest["episodes"][0]["requested_profile"] = old_profile
    observed = deepcopy(row["requested_profile"])
    assert "competitor_visibility" not in old_profile
    outcome = {
        "episode_id": row["episode_id"], "status": "delivered",
        "facts_path": "/facts.json", "questions_path": "/questions.json",
        "condition_profile": observed,
        "produced_count_by_qa": {qa: 1 for qa in row["requested_quota_by_qa"]},
    }
    result = collect_batch_outcomes(manifest, [outcome])
    collected = result["episodes"][0]
    assert collected["requested_source_classes"] == row["requested_source_classes"]
    assert collected["profile_matches_request"] is True
    pairs = result["class_pair_condition_group_crosstab"]["class_pairs"]
    assert pairs and "" not in pairs
    quota = next(item for item in result["quota_by_condition_group"]
                 if item["room_id"] == row["room_id"] and item["condition_group"] == row["condition_group"])
    assert quota["delivered"] == 1
    assert quota["unmet"] == 0


def test_scaleup_dry_run_cli_writes_sound_pool(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    catalog = {
        "path_bindings": {"AVENGINE_TEST_ROOT": "/data/test"},
        "rooms": [
            {"room_id": "room_a", "family": "authored", "renderer": "ue_spear"},
            {"room_id": "room_b", "family": "apartment", "renderer": "ue_spear"},
        ],
    }
    registry = {"assets": [
        asset("red", "red"), asset("blue", "blue"), asset("green", "green"),
        animal_asset("dog_a"), animal_asset("dog_b"),
        device_asset("dev_a"), device_asset("dev_b"),
    ]}
    sounds = {
        "sounds": [
            sound("one", "speaker1"), sound("two", "speaker2"), sound("three", "speaker3"),
            {"sound_asset_id": "bark1", "sound_identity_id": "dog:a", "sound_class": "dog_bark",
             "species_id": "dog", "sample_rate_hz": 16000, "sample_count": 32000,
             "active_duration_s": 2, "path": "/prepared/bark1.wav",
             "compatible_asset_ids": ["dog_a", "dog_b"]},
            {"sound_asset_id": "hum1", "sound_identity_id": "dev:a", "sound_class": "blender",
             "sample_rate_hz": 16000, "sample_count": 32000, "active_duration_s": 2,
             "path": "/prepared/hum1.wav", "compatible_asset_ids": ["dev_a", "dev_b"]},
        ]
    }
    config = {
        "base_request": {
            "camera": {"fov_deg": 85, "motion": "static"},
            "room_catalog": "relative/catalog.json",
            "source_registry": "relative/registry.json",
            "runtime": {"graphics_adapter": 0},
            "sound_selection": {"prepared_set": "/should/be/removed.json", "max_clip_s": 5.0},
        }
    }
    catalog_path = tmp_path / "catalog.json"
    registry_path = tmp_path / "registry.json"
    sounds_path = tmp_path / "sounds.json"
    config_path = tmp_path / "config.json"
    catalog_path.write_text(json.dumps(catalog))
    registry_path.write_text(json.dumps(registry))
    sounds_path.write_text(json.dumps(sounds))
    config_path.write_text(json.dumps(config))
    spec = importlib.util.spec_from_file_location(
        "build_qa_batch_manifest_h7", repo / "tools/dataset/build_qa_batch_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_source_asset_runtime_registry",
                        lambda path: json.loads(Path(path).read_text()))
    output = tmp_path / "dryrun"
    mod.main([
        "scaleup-dry-run",
        "--config", str(config_path),
        "--catalog", str(catalog_path),
        "--registry", str(registry_path),
        "--sounds", str(sounds_path),
        "--output", str(output),
        "--seed", "20260907",
        "--episodes-per-room", "7",
        "--batch-id", "qa_h7_cli",
    ])
    manifest = json.loads((output / "batch_manifest.json").read_text())
    scaleup = json.loads((output / "scaleup_config.json").read_text())
    assert manifest["requested_episode_count"] == 14
    distances = {tuple(row["requested_profile"]["distance_range_m"]) for row in manifest["episodes"]}
    assert distances == {(1.5, 6.0)}
    for row, slot in zip(manifest["episodes"], scaleup["slots"]):
        assert row["request"]["profile"]["distance_range_m"] == slot["profile"]["distance_range_m"] == [1.5, 6.0]
        assert row["request"]["sound_pool"] == str(sounds_path.resolve())
        assert Path(row["request"]["room_catalog"]).is_absolute()
        assert row["request"]["runtime"]["path_bindings"]["AVENGINE_TEST_ROOT"] == "/data/test"
        assert "prepared_set" not in row["request"].get("sound_selection", {})
    producer = manifest["producer"]
    assert producer["room_catalog"] == str(catalog_path.resolve())
    assert producer["source_registry"] == str(registry_path.resolve())
    assert producer["sound_pool"] == str(sounds_path.resolve())
    for key, path in (("room_catalog", catalog_path), ("source_registry", registry_path),
                      ("sound_pool", sounds_path)):
        assert producer["inputs"][key]["path"] == str(path.resolve())
        assert producer["inputs"][key]["size_bytes"] == path.stat().st_size
    assert producer["effective_runtime_path_inputs"]["path_bindings"] == {
        "AVENGINE_TEST_ROOT": "/data/test"
    }



def test_collect_outcomes_uses_shared_failure_classifier_for_each_failure_stage():
    manifest = {
        "batch_id": "failure_classifier",
        "episodes": [
            {
                "episode_id": "capture_slot",
                "room_id": "room_a",
                "room_family": "authored",
                "condition_group": "identity_binding",
                "requested_source_classes": ["articulated_human", "articulated_human"],
                "requested_profile": {},
                "source_assignments": [],
                "requested_quota_by_qa": {"QA-01": 1},
                "preallocation_gaps": [],
            },
            {
                "episode_id": "delivery_slot",
                "room_id": "room_a",
                "room_family": "authored",
                "condition_group": "audio_event_relations",
                "requested_source_classes": ["articulated_human", "articulated_human"],
                "requested_profile": {},
                "source_assignments": [],
                "requested_quota_by_qa": {"QA-01": 1},
                "preallocation_gaps": [],
            },
            {
                "episode_id": "review_slot",
                "room_id": "room_a",
                "room_family": "authored",
                "condition_group": "visibility_occlusion",
                "requested_source_classes": ["articulated_human", "articulated_human"],
                "requested_profile": {},
                "source_assignments": [],
                "requested_quota_by_qa": {"QA-01": 1},
                "preallocation_gaps": [],
            },
        ],
    }
    outcomes = collect_batch_outcomes(manifest, [
        {
            "episode_id": "capture_slot",
            "status": "capture_failed",
            "failure_reason": "renderer returned an unknown native status 17",
        },
        {
            "episode_id": "delivery_slot",
            "status": "delivery_failed",
            "failure_reason": "ValueError: delivery output is malformed",
        },
        {
            "episode_id": "review_slot",
            "status": "review_failed",
            "failure_reason": "review gate returned an unrecognised result",
        },
    ])
    by_id = {row["episode_id"]: row for row in outcomes["episodes"]}
    assert by_id["capture_slot"]["failure_stage"] == "capture"
    assert by_id["capture_slot"]["gap_state"] == "evidence_missing_or_unsampled"
    assert by_id["capture_slot"]["outcome"]["reason_code"] == "unclassified_failure"
    assert by_id["capture_slot"]["outcome"]["diagnostic"]["classification"] == "unclassified"
    assert by_id["capture_slot"]["outcome"]["diagnostic"]["failure_stage"] == "capture"
    assert by_id["capture_slot"]["outcome"]["diagnostic"]["failure_reason"] == (
        "renderer returned an unknown native status 17"
    )
    assert by_id["delivery_slot"]["failure_stage"] == "finalize"
    assert by_id["delivery_slot"]["gap_state"] == "interface_not_implemented"
    assert by_id["review_slot"]["failure_stage"] == "finalize"
    assert by_id["review_slot"]["gap_state"] == "evidence_missing_or_unsampled"
    assert by_id["review_slot"]["outcome"]["reason_code"] == "unclassified_failure"


def test_collect_outcomes_marks_explicit_clip_rejection_as_evidence_gap():
    manifest = {
        "batch_id": "clip_classifier",
        "episodes": [{
            "episode_id": "clip_slot",
            "room_id": "room_a",
            "room_family": "authored",
            "condition_group": "identity_binding",
            "requested_source_classes": ["rigid_static_object", "rigid_static_object"],
            "requested_profile": {},
            "source_assignments": [],
            "requested_quota_by_qa": {"QA-01": 1},
            "preallocation_gaps": [],
        }],
    }
    outcomes = collect_batch_outcomes(manifest, [{
        "episode_id": "clip_slot",
        "status": "audio_failed",
        "failure_reason": (
            "CurrentMP3DDynamicAudioError: audio output would clip "
            "without normalization/limiting: peak=1.138"
        ),
    }])
    row = outcomes["episodes"][0]
    assert row["gap_state"] == "evidence_missing_or_unsampled"
    assert row["outcome"]["reason_code"] == "clip_overflow_rejected"
    assert row["outcome"]["diagnostic"]["classification_reason"] == "clip_overflow_rejection"


# ---------------------------------------------------------------------------
# P01: one production request, QA quota from configuration, entity instances
# and the minimal stage protocol.
# ---------------------------------------------------------------------------


def production_defaults():
    return {
        "clock": {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000},
        "rig": {"resolution_hw": [720, 1280], "fov_deg": 85},
        "reserve_tail_s": 3.0,
        "profile": {"separation_bin_deg": [30, 60], "anchor_count": 1},
        "sound": {
            "pool": "/pool/batch_sounds.json",
            "selection": {
                "preallocated_sound_asset_ids_by_actor": {
                    "source1": ["one", "one_alt"],
                    "source2": ["two", "three"],
                }
            },
        },
        "request_extras": {"binding_motion": dict(BINDING_MOTION)},
    }


BINDING_MOTION = {"minimum_motion_s": 2.0, "end_hold_s": 0.5, "angle_tolerance_deg": 10,
                  "minimum_entity_separation_m": 0.95, "source_start_s": 0.1,
                  "walk_speed_range_mps": [0.5, 0.8]}


def production_member(request_id, assets=("red", "blue")):
    return {"request_id": request_id,
            "instances": [{"instance_id": f"source{index + 1}", "asset_id": asset,
                           "source_class": "articulated_human"}
                          for index, asset in enumerate(assets)]}


def production_group(group_id, task_family, room_id):
    return {"group_id": group_id, "task_family": task_family, "room_id": room_id,
            "members": [production_member(f"{group_id}_m{index + 1}") for index in range(4)]}


def production_config(rooms):
    return {
        "schema": PRODUCTION_SPEC_SCHEMA,
        "batch_id": "p01_batch",
        "seed": 7,
        "defaults": production_defaults(),
        "episodes": [{**production_member("p01_episode_01"), "room_id": rooms[0],
                      "condition_group": "identity_binding"}],
        "core_groups": [
            production_group("p01_visible", "visible_binding", rooms[0]),
            production_group("p01_relation", "visual_conditioned_relation", rooms[1]),
            production_group("p01_identity", "cross_event_identity", rooms[0]),
            production_group("p01_state", "cross_time_state", rooms[1]),
        ],
        "coverage_quota": {"min_main_questions_per_qa_id": 8},
    }


def test_production_config_yields_an_episode_and_four_core_groups(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_batch", "seed": 7, "base_request": config["base_request"],
         "production": production_config(room_ids)},
        registry, rooms, sounds)
    assert manifest["production"]["episode_count"] == 1
    assert manifest["production"]["core_group_count"] == 4
    assert manifest["production"]["core_member_count"] == 16
    assert manifest["production"]["derived_slot_count"] == 17
    assert manifest["production"]["coverage_quota"] == {"min_main_questions_per_qa_id": 8}
    assert manifest["requested_episode_count"] == 17

    rows = {row["episode_id"]: row for row in manifest["episodes"]}
    assert rows["p01_episode_01"].get("task_family") is None
    assert rows["p01_episode_01"]["condition_group_source"] == "config"
    member = rows["p01_state_m1"]
    assert member["task_family"] == "cross_time_state"
    assert member["group_id"] == "p01_state"
    assert member.get("member_role") is None or isinstance(member["member_role"], str)
    assert member["condition_group"] == CONDITION_GROUP_BY_TASK_FAMILY["cross_time_state"]
    assert member["condition_group_source"] == "task_family_default"
    families = {row.get("task_family") for row in manifest["episodes"]}
    assert families == {None, "visible_binding", "visual_conditioned_relation",
                        "cross_event_identity", "cross_time_state"}
    assert rows["p01_episode_01"]["production_request"]["stage_plan"] == [
        "plan", "capture", "audio", "delivery"]
    assert rows["p01_episode_01"]["stage_scope"] == {
        "kind": "episode", "scope_id": "p01_episode_01"}
    # A member has no plan of its own: its stages are the group's shared units.
    assert member["production_request"]["stage_plan"] is None
    assert member["production_request"]["stage_scope"] == "core_group"
    assert member["stage_work_items"] == []
    scope = member["stage_scope"]
    assert scope["kind"] == "core_group"
    assert scope["group_id"] == "p01_state"
    assert scope["delivering_unit_id"] == "v0_a0"
    assert scope["consumes_visual_unit_id"] == "v0_capture"
    assert scope["entry_point"].endswith("stage_work_items_for_group")
    state = next(entry for entry in manifest["production"]["core_groups"]
                 if entry["group_id"] == "p01_state")
    assert [row["unit_id"] for row in state["stage_units"]] == [
        "v0", "v0_capture", "v0_a0", "v0_a1", "v1", "v1_capture", "v1_a0", "v1_a1", "group"]
    assert state["recipe"]["motion_timing"] == "after_wet_tail"
    assert [item["unit_id"] for item in state["initial_work_items"]] == ["v0"]
    assert manifest["production"]["shared_unit_count"] == 40
    identity_units = next(group for group in manifest["production"]["core_groups"]
                          if group["task_family"] == "cross_event_identity")["stage_units"]
    assert [unit["unit_id"] for unit in identity_units if unit.get("internal_only")] == [
        "identity_probe_plan", "identity_probe_capture", "identity_probe_audio", "identity_topology"]



def test_shared_production_audio_preallocation_intersects_and_rejects_empty(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    production = production_config(room_ids)
    production["episodes"] = []
    production["core_groups"] = [production["core_groups"][0]]
    production["core_groups"][0]["members"][2]["sound"] = {
        "selection": {
            "preallocated_sound_asset_ids_by_actor": {
                "source1": ["one"],
                "source2": ["two"],
            }
        }
    }
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_shared_audio", "seed": 7,
         "base_request": config["base_request"], "production": production},
        registry, rooms, sounds)
    rows = {row["episode_id"]: row for row in manifest["episodes"]}
    expected = {"source1": ["one"], "source2": ["two"]}
    assert rows["p01_visible_m1"]["request"]["sound_selection"][
        "preallocated_sound_asset_ids_by_actor"
    ] == expected
    assert rows["p01_visible_m3"]["request"]["sound_selection"][
        "preallocated_sound_asset_ids_by_actor"
    ] == expected

    broken = production_config(room_ids)
    broken["episodes"] = []
    broken["core_groups"] = [broken["core_groups"][0]]
    broken["core_groups"][0]["members"][2]["sound"] = {
        "selection": {
            "preallocated_sound_asset_ids_by_actor": {
                "source1": ["three"],
                "source2": ["one"],
            }
        }
    }
    with pytest.raises(ValueError, match="no common legal sound candidate"):
        prepare_batch_manifest(
            {"batch_id": "p01_shared_audio_broken", "seed": 7,
             "base_request": config["base_request"], "production": broken},
            registry, rooms, sounds)


def test_production_and_slots_together_are_refused(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    with pytest.raises(ValueError, match="declares both production and slots"):
        prepare_batch_manifest(
            {**config, "production": production_config(room_ids)}, registry, rooms, sounds)


def test_explicit_clock_resolution_and_fov_reach_the_production_request(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    production = production_config(room_ids)
    production["core_groups"] = []
    production["episodes"][0].update({
        "clock": {"frame_count": 300, "frame_rate_hz": 30, "sample_rate_hz": 48000},
        "rig": {"resolution_hw": [1080, 1920], "fov_deg": 60},
        "reserve_tail_s": 4.0,
        "resources": {"capture": {"graphics_adapter": 1, "min_free_vram_mb": 24000}},
    })
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_explicit", "seed": 7, "base_request": config["base_request"],
         "production": production}, registry, rooms, sounds)
    row = manifest["episodes"][0]
    assert row["request"]["frame_count"] == 300
    assert row["request"]["frame_rate_hz"] == 30.0
    assert row["request"]["sample_rate_hz"] == 48000
    assert row["request"]["camera"]["resolution_hw"] == [1080, 1920]
    assert row["request"]["camera"]["fov_deg"] == 60.0
    assert row["request"]["profile"]["reserve_tail_s"] == 4.0
    assert row["production_request"]["duration_seconds"] == pytest.approx(10.0)
    assert row["production_request"]["available_program_seconds"] == pytest.approx(6.0)
    plan = row["stage_work_items"][0]
    assert plan["payload"]["clock"]["frame_count"] == 300
    assert plan["payload"]["rig"]["fov_deg"] == 60.0
    assert plan["payload"]["reserve_tail_s"] == 4.0


def test_illegal_production_values_fail_before_any_row_exists(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    production = production_config(room_ids)
    production["core_groups"] = []
    production["episodes"][0]["rig"] = {"resolution_hw": [720, 1280], "fov_deg": 85,
                                        "motion": "orbit"}
    with pytest.raises(ValueError, match="fixes the camera"):
        prepare_batch_manifest(
            {"batch_id": "p01_bad", "seed": 7, "base_request": config["base_request"],
             "production": production}, registry, rooms, sounds)


def test_qa_ids_items_and_quota_come_from_configuration(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["qa"] = {"qa_ids": ["QA-05", "QA-20", "QA-25"], "items_per_type": 2,
                    "quota_by_qa": {"QA-05": 8, "QA-20": 8, "QA-25": 3}}
    config["slots"][1]["qa"] = {"qa_ids": ["QA-13"], "quota_by_qa": {"QA-13": 4}}
    manifest = prepare_batch_manifest(config, registry, rooms, sounds)
    first, second = manifest["episodes"]
    assert first["requested_qa_ids"] == ["QA-05", "QA-20", "QA-25"]
    assert first["request"]["qa_ids"] == ["QA-05", "QA-20", "QA-25"]
    assert first["request"]["qa_sampling"]["items_per_type"] == 2
    assert first["items_per_type"] == 2
    assert first["items_per_type_source"] == "config"
    assert first["requested_quota_by_qa"] == {"QA-05": 8, "QA-20": 8, "QA-25": 3}
    assert first["requested_quota_source"] == "config"
    assert first["qa_ids_source"] == "config"
    assert second["requested_qa_ids"] == ["QA-13"]
    assert second["requested_quota_by_qa"] == {"QA-13": 4}
    assert second["request"]["qa_sampling"]["items_per_type"] == 2


def test_a_config_without_a_qa_block_keeps_the_old_rows_and_says_so(inputs):
    manifest = prepare_batch_manifest(*inputs)
    row = manifest["episodes"][0]
    assert row["requested_qa_ids"] == list(QA_IDS)
    assert row["qa_ids_source"] == "unified_catalog_all"
    assert row["requested_quota_by_qa"] == {qa: (3 if qa == "QA-25" else 1) for qa in QA_IDS}
    assert row["requested_quota_source"] == "legacy_default"
    assert row["items_per_type"] == 1
    assert row["items_per_type_source"] == "legacy_default"


@pytest.mark.parametrize("block,message", [
    ({"qa_ids": ["QA-99"]}, "not in the unified catalog"),
    ({"qa_ids": []}, "nonempty list"),
    ({"qa_ids": ["QA-05", "QA-05"]}, "must be distinct"),
    ({"items_per_type": 0}, "positive integer"),
    ({"qa_ids": ["QA-05"], "quota_by_qa": {"QA-06": 1}}, "outside qa_ids"),
    ({"qa_ids": ["QA-05"], "quota_by_qa": {"QA-05": 0}}, "positive integer"),
    ({"qa_ids": ["QA-05"], "quota_by_qa": {}}, "nonempty mapping"),
])
def test_illegal_qa_blocks_fail_explicitly(inputs, block, message):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["qa"] = block
    with pytest.raises(ValueError, match=message):
        prepare_batch_manifest(config, registry, rooms, sounds)


def test_derived_targets_name_the_anchor_entity_not_the_first_sound(inputs):
    manifest = prepare_batch_manifest(*inputs)
    row = manifest["episodes"][0]
    profile = row["requested_profile"]
    instance_ids = [instance["instance_id"] for instance in row["entity_instances"]]
    expected = [instance_ids[index] for index in profile["anchor_indices"]]
    assert row["qa_targets"]
    for target in row["qa_targets"]:
        assert target["target_instance_ids"] == expected
        assert target["event"] == {"kind": "target_audible_window"}
        assert target["target_source"] == "resolved_anchor_entities"
        assert "ordinal" not in target["event"]
        assert target["items"] == row["requested_quota_by_qa"][target["qa_id"]]
    assert "qa_targets" not in row["request"]


def test_declared_targets_are_kept_and_validated(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["qa"] = {
        "qa_ids": ["QA-05"],
        "quota_by_qa": {"QA-05": 2},
        "qa_targets": [{"qa_id": "QA-05", "target_instance_ids": ["source2"],
                        "event": {"kind": "event_ordinal", "ordinal": 3}, "items": 2}],
    }
    manifest = prepare_batch_manifest(config, registry, rooms, sounds)
    target = manifest["episodes"][0]["qa_targets"][0]
    assert target["target_instance_ids"] == ["source2"]
    assert target["event"] == {"kind": "event_ordinal", "ordinal": 3}
    assert target["target_source"] == "config"
    assert manifest["episodes"][0]["request"]["qa_targets"] == manifest["episodes"][0]["qa_targets"]

    broken = deepcopy(config)
    broken["qa"]["qa_targets"][0]["target_instance_ids"] = ["source9"]
    with pytest.raises(ValueError, match="are not in this episode"):
        prepare_batch_manifest(broken, registry, rooms, sounds)

    broken = deepcopy(config)
    broken["qa"]["qa_targets"][0].pop("event")
    with pytest.raises(ValueError, match="event must state its kind"):
        prepare_batch_manifest(broken, registry, rooms, sounds)

    broken = deepcopy(config)
    broken["qa"]["qa_targets"][0]["qa_id"] = "QA-13"
    with pytest.raises(ValueError, match="outside qa_ids"):
        prepare_batch_manifest(broken, registry, rooms, sounds)


def test_resolve_qa_targets_without_a_resolved_profile_says_so():
    plan = resolve_qa_plan({"qa": {"qa_ids": ["QA-05"], "quota_by_qa": {"QA-05": 1}}}, {})
    targets = resolve_qa_targets(
        plan, instances=[{"instance_id": "source1"}, {"instance_id": "source2"}], condition=None)
    assert targets[0]["target_source"] == "unresolved_all_instances"
    assert targets[0]["target_instance_ids"] == ["source1", "source2"]


def test_entity_instance_count_is_not_the_asset_count(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0]["source_asset_ids"] = ["red", "red"]
    manifest = prepare_batch_manifest(config, registry, rooms, sounds)
    row = manifest["episodes"][0]
    assert [instance["instance_id"] for instance in row["entity_instances"]] == ["source1", "source2"]
    assert [instance["asset_id"] for instance in row["entity_instances"]] == ["red", "red"]
    assert row["entity_instance_count"] == 2
    assert row["distinct_asset_count"] == 1
    assert [actor["instance_id"] for actor in row["source_assignments"]] == ["source1", "source2"]
    assert [actor["actor_id"] for actor in row["source_assignments"]] == ["source1", "source2"]
    gap = next(item for item in row["preallocation_gaps"]
               if item["code"] == "repeated_asset_across_entity_instances")
    assert gap["state"] == "interface_not_implemented"
    assert gap["asset_ids"] == ["red"]
    assert gap["file"] == "src/avengine/rooms/conditioned_sampler.py"
    assert gap["functions"] == ["resolve_condition_profile", "select_entities"]
    # The shared asset is not silently planned: the row keeps its whole quota.
    assert row["requested_profile"] is None
    assert manifest["preallocation_gap_counts"]["repeated_asset_across_entity_instances"] == 1


def test_distinct_assets_produce_no_shared_asset_gap(inputs):
    manifest = prepare_batch_manifest(*inputs)
    row = manifest["episodes"][0]
    assert row["entity_instance_count"] == row["distinct_asset_count"] == 2
    assert not [item for item in row["preallocation_gaps"]
                if item["code"] == "repeated_asset_across_entity_instances"]
    assert shared_asset_instance_gap(row["entity_instances"]) is None


def test_named_entity_instances_and_roles_survive_into_the_row(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0]["entity_instances"] = [
        {"instance_id": "speaker_left", "asset_id": "red", "role": "anchor"},
        {"instance_id": "speaker_right", "asset_id": "blue", "role": "competitor"},
    ]
    config["slots"][0]["source_asset_ids"] = ["red", "blue"]
    manifest = prepare_batch_manifest(config, registry, rooms, sounds)
    row = manifest["episodes"][0]
    assert [instance["instance_id"] for instance in row["entity_instances"]] == [
        "speaker_left", "speaker_right"]
    assert [actor["actor_id"] for actor in row["source_assignments"]] == [
        "speaker_left", "speaker_right"]
    assert [actor["role"] for actor in row["source_assignments"]] == ["anchor", "competitor"]
    assert row["request"]["entity_instances"][0]["role"] == "anchor"
    assert set(row["qa_targets"][0]["target_instance_ids"]) <= {"speaker_left", "speaker_right"}


@pytest.mark.parametrize("declared,message", [
    ([{"instance_id": "a"}], "one entry per source class"),
    ([{"instance_id": "a"}, {"instance_id": "a"}], "must be distinct"),
    ([{"instance_id": "a"}, {"instance_id": "b", "source_class": "rigid_static_object"}],
     "disagrees with source_classes"),
])
def test_illegal_entity_instances_fail_explicitly(declared, message):
    with pytest.raises(ValueError, match=message):
        entity_instances_for_slot({"entity_instances": declared},
                                  ["articulated_human", "articulated_human"], None)


def test_episode_row_stage_items_start_at_planning_and_grow_from_results(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_stages", "seed": 7, "base_request": config["base_request"],
         "production": production_config(room_ids)}, registry, rooms, sounds)
    plain = next(row for row in manifest["episodes"] if row["episode_id"] == "p01_episode_01")
    assert [item["stage"] for item in plain["stage_work_items"]] == ["plan"]
    assert plain["stage_work_items"][0]["resource"]["kind"] == "cpu"
    assert plain["stage_work_items"][0]["resource"]["execution"] == "cpu"
    assert plain["stage_work_items"][0]["fresh_output_relative"] == (
        "p01_episode_01/plan/attempt_01")
    plan_pass = {"work_item_id": plain["stage_work_items"][0]["work_item_id"], "stage": "plan",
                 "request_id": "p01_episode_01", "status": "pass",
                 "facts": {"episode_plan_path": "plan/episode_plan.json", "renderer": "ue_spear",
                           "clock": plain["stage_work_items"][0]["payload"]["clock"]}}
    after = stage_work_items_for_row(plain, results=[plan_pass])
    assert [item["stage"] for item in after] == ["capture"]
    assert after[0]["resource"]["execution"] == "gpu"


def test_a_member_row_refuses_a_row_scoped_schedule(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_member", "seed": 7, "base_request": config["base_request"],
         "production": production_config(room_ids)}, registry, rooms, sounds)
    member = next(row for row in manifest["episodes"] if row["episode_id"] == "p01_state_m1")
    with pytest.raises(ValueError, match="stage_work_items_for_group"):
        stage_work_items_for_row(member)


def test_group_units_follow_the_real_cross_time_dependency(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_group", "seed": 7, "base_request": config["base_request"],
         "production": production_config(room_ids)}, registry, rooms, sounds)
    clock = next(row for row in manifest["episodes"]
                 if row["episode_id"] == "p01_state_m1")["production_request"]["clock"]

    first = stage_work_items_for_group(manifest, "p01_state")
    assert [item["unit_id"] for item in first] == ["v0"]
    assert first[0]["member_request_ids"] == ["p01_state_m1", "p01_state_m2"]

    def result(item, facts):
        return {"work_item_id": item["work_item_id"], "stage": item["stage"],
                "request_id": item["scope_id"], "status": "pass", "facts": facts,
                "depends_on": item["depends_on"]}

    results = [result(first[0], {"episode_plan_path": "v0/plan.json", "renderer": "habitat",
                                 "clock": clock})]
    capture = stage_work_items_for_group(manifest, "p01_state", results=results)
    assert [item["unit_id"] for item in capture] == ["v0_capture"]
    assert capture[0]["resource"]["execution"] == "gpu"
    assert capture[0]["member_request_ids"] == ["p01_state_m1", "p01_state_m2"]
    results.append(result(capture[0], {"capture_receipt_path": "v0_capture.json",
                                       "captured_frame_count": 150}))

    columns = stage_work_items_for_group(manifest, "p01_state", results=results)
    assert sorted(item["unit_id"] for item in columns) == ["v0_a0", "v0_a1"]
    assert all(item["resource"]["execution"] == "cpu" for item in columns)
    assert all(item["resource"]["runtime_context"] == "rlr_native" for item in columns)
    for index, item in enumerate(columns):
        results.append(result(item, {
            "facts_path": f"{item['unit_id']}/facts.json",
            "audio_report_path": f"{item['unit_id']}/report.json",
            "wet_tail_intervals": [{"start_s": 1.0, "end_s": 3.0 + index * 0.4}]}))

    late = stage_work_items_for_group(manifest, "p01_state", results=results)
    assert [item["unit_id"] for item in late] == ["v1"]
    assert late[0]["stage"] == "late_plan"
    window = late[0]["payload"]["measured_motion_window"]
    assert window["measured_wet_end_s"] == pytest.approx(3.4)
    assert window["sufficient"] is True
    assert window["boundary_formula"] == "ceil(max(wet_tail end_s) * fps) + 1"
    assert len(window["wet_tail_source_work_item_ids"]) == 2
    assert group_blockers_for_group(manifest, "p01_state", results=results) == []


def test_a_group_whose_tail_fills_the_clip_is_blocked_with_a_reason(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_blocked", "seed": 7, "base_request": config["base_request"],
         "production": production_config(room_ids)}, registry, rooms, sounds)
    clock = next(row for row in manifest["episodes"]
                 if row["episode_id"] == "p01_state_m1")["production_request"]["clock"]
    scope = "p01_state/{}"
    results = [
        {"work_item_id": f"{scope.format('v0')}:plan:01", "stage": "plan",
         "request_id": scope.format("v0"), "status": "pass",
         "facts": {"episode_plan_path": "v0/plan.json", "renderer": "habitat", "clock": clock}},
        {"work_item_id": f"{scope.format('v0_capture')}:capture:01", "stage": "capture",
         "request_id": scope.format("v0_capture"), "status": "pass",
         "facts": {"capture_receipt_path": "v0.json", "captured_frame_count": 150},
         "depends_on": [f"{scope.format('v0')}:plan:01"]},
    ]
    for unit in ("v0_a0", "v0_a1"):
        results.append({
            "work_item_id": f"{scope.format(unit)}:audio:01", "stage": "audio",
            "request_id": scope.format(unit), "status": "pass",
            "facts": {"facts_path": "f.json", "audio_report_path": "r.json",
                      "wet_tail_intervals": [{"start_s": 8.0, "end_s": 9.4}]},
            "depends_on": [f"{scope.format('v0_capture')}:capture:01"]})
    assert stage_work_items_for_group(manifest, "p01_state", results=results) == []
    blockers = group_blockers_for_group(manifest, "p01_state", results=results)
    assert [row["code"] for row in blockers] == [
        "measured_reverberation_leaves_insufficient_movement_time"]


def test_declared_resources_and_retry_survive_the_group_round_trip(inputs):
    """A work item rebuilt from a saved request keeps the declared resources."""
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    production = production_config(room_ids)
    production["defaults"]["resources"] = {
        "graphics_adapter": 2,
        "capture": {"min_free_vram_mb": 20000, "rpc_port": 40100},
        "audio": {"rlr_threads": 1},
        "audio_tail_probe": {"rlr_threads": 1},
    }
    production["defaults"]["retry"] = {"attempts_per_stage": 3, "attempts_within_profile": 50}
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_resources", "seed": 7, "base_request": config["base_request"],
         "production": production}, registry, rooms, sounds)
    rows = {row["episode_id"]: row for row in manifest["episodes"]}
    state = rows["p01_state_m1"]
    assert state["request"]["production"]["stage_resources"]["capture"] == {
        "kind": "gpu_native_visual", "execution": "gpu", "runtime_context": "renderer_native",
        "graphics_adapter": 2, "min_free_vram_mb": 20000, "rpc_port": 40100}
    assert state["request"]["production"]["stage_resources"]["audio"] == {
        "kind": "cpu_native_acoustic", "execution": "cpu", "runtime_context": "rlr_native",
        "graphics_adapter": 2, "rlr_threads": 1}
    assert state["request"]["production"]["retry"] == {"attempts_per_stage": 3,
                                                       "attempts_within_profile": 50}
    plan_item = stage_work_items_for_group(manifest, "p01_state")[0]
    assert plan_item["resource"] == {"kind": "cpu", "execution": "cpu",
                                     "runtime_context": "pure_python", "graphics_adapter": 2}
    assert plan_item["payload"]["retry"] == {"attempts_per_stage": 3,
                                             "attempts_within_profile": 50}
    plan_pass = {"work_item_id": plan_item["work_item_id"], "stage": "plan",
                 "request_id": plan_item["scope_id"], "status": "pass",
                 "facts": {"episode_plan_path": "p/plan.json", "renderer": "habitat",
                           "clock": plan_item["payload"]["clock"]}}
    capture = stage_work_items_for_group(manifest, "p01_state", results=[plan_pass])[0]
    assert capture["resource"] == {"kind": "gpu_native_visual", "execution": "gpu",
                                  "runtime_context": "renderer_native", "graphics_adapter": 2,
                                  "min_free_vram_mb": 20000, "rpc_port": 40100}
    capture_pass = {"work_item_id": capture["work_item_id"], "stage": "capture",
                    "request_id": capture["scope_id"], "status": "pass",
                    "facts": {"capture_receipt_path": "v0.json", "captured_frame_count": 150},
                    "depends_on": capture["depends_on"]}
    columns = stage_work_items_for_group(manifest, "p01_state",
                                          results=[plan_pass, capture_pass])
    assert all(item["resource"] == {"kind": "cpu_native_acoustic", "execution": "cpu",
                                    "runtime_context": "rlr_native", "graphics_adapter": 2,
                                    "rlr_threads": 1} for item in columns)


def test_manifest_declares_the_stage_protocol_and_its_boundary(inputs):
    manifest = prepare_batch_manifest(*inputs)
    protocol = manifest["stage_protocol"]
    assert protocol["stages"] == ["plan", "capture", "audio", "late_plan", "assembly", "delivery"]
    assert protocol["resource_kind_by_stage"]["capture"] == "gpu_native_visual"
    assert protocol["resource_kind_by_stage"]["audio"] == "cpu_native_acoustic"
    assert "audio_tail_probe" in protocol["removed_stages"]
    assert protocol["group_recipes"]["cross_time_state"]["units"][4]["unit_id"] == "v1"
    assert "not an executed stage" in protocol["claim_boundary"]
    assert manifest["production"] is None
    assert manifest["executed_episode_count"] == 0


def test_a_batch_quota_still_reaches_a_production_config_row(inputs):
    """The spec fills a unit quota only as a placeholder; config.qa still wins."""
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    production = production_config(room_ids)
    production["core_groups"] = []
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_quota", "seed": 7, "base_request": config["base_request"],
         "qa": {"quota_by_qa": {qa: 8 for qa in QA_IDS}},
         "production": production}, registry, rooms, sounds)
    row = manifest["episodes"][0]
    assert row["requested_quota_by_qa"] == {qa: 8 for qa in QA_IDS}
    assert row["requested_quota_source"] == "config"
    assert all(target["items"] == 8 for target in row["qa_targets"])
    slots, _summary = production_config_slots(production)
    assert "quota_by_qa" not in slots[0]["qa"]
    assert "qa_targets" not in slots[0]["qa"]


def test_a_production_declared_quota_beats_the_batch_block(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    production = production_config(room_ids)
    production["core_groups"] = []
    production["episodes"][0].update({"qa_ids": ["QA-05"], "quota_by_qa": {"QA-05": 12}})
    manifest = prepare_batch_manifest(
        {"batch_id": "p01_quota2", "seed": 7, "base_request": config["base_request"],
         "qa": {"quota_by_qa": {qa: 8 for qa in QA_IDS}},
         "production": production}, registry, rooms, sounds)
    row = manifest["episodes"][0]
    assert row["requested_qa_ids"] == ["QA-05"]
    assert row["requested_quota_by_qa"] == {"QA-05": 12}
    slots, _summary = production_config_slots(production)
    assert slots[0]["qa"]["quota_by_qa"] == {"QA-05": 12}


def test_production_slots_are_derived_without_touching_the_registry(inputs):
    _config, _registry, rooms, _sounds = deepcopy(inputs)
    room_ids = [room["room_id"] for room in rooms["rooms"]]
    slots, summary = production_config_slots(production_config(room_ids))
    assert len(slots) == 17
    assert summary["core_member_count"] == 16
    assert [pair for group in summary["core_groups"] for pair in group["member_request_ids"]][:4] == [
        "p01_visible_m1", "p01_visible_m2", "p01_visible_m3", "p01_visible_m4"]
    assert [slot.get("member_index") for slot in slots[1:5]] == [0, 1, 2, 3]
    episode = slots[0]
    assert episode.get("member_index") is None
    assert episode["source_classes"] == ["articulated_human", "articulated_human"]
    assert episode["source_asset_ids"] == ["red", "blue"]
    assert episode["request_overrides"]["schema"] == "avengine_native_qa_room_request_v1"
    assert episode["qa"]["qa_ids"] == list(QA_IDS)
    assert len({slot["seed"] for slot in slots[1:5]}) == 1
    assert len({slot["seed"] for slot in slots[5:9]}) == 1
    assert len({slot["seed"] for slot in slots[9:13]}) == 1
    assert len({slot["seed"] for slot in slots[13:17]}) == 1
    assert len({slot["seed"] for slot in slots[1:]}) == 4


def test_prepare_preserves_runner_controls_without_putting_them_in_episode_inputs(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    controls = {
        "p19_coverage": {"enabled": True, "cached_survey": "retained.json",
                         "append_requests": True, "max_new_ordinary": 3},
        "resource_policy": {"gpu": {"allow_shared_device": False},
                            "cpu": {"max_workers": 2}},
    }
    config.update(deepcopy(controls))
    manifest = prepare_batch_manifest(config, registry, rooms, sounds)
    for key, value in controls.items():
        assert manifest[key] == value
        assert all(key not in row["request"] for row in manifest["episodes"])
    manifest["resource_policy"]["cpu"]["max_workers"] = 1
    assert config["resource_policy"]["cpu"]["max_workers"] == 2
    manifest["p19_coverage"]["max_new_ordinary"] = 0
    assert config["p19_coverage"]["max_new_ordinary"] == 3


@pytest.mark.parametrize("key", ["p19_coverage", "resource_policy"])
def test_prepare_rejects_nonmapping_runner_controls(inputs, key):
    config, registry, rooms, sounds = deepcopy(inputs)
    config[key] = []
    with pytest.raises(ValueError, match=key):
        prepare_batch_manifest(config, registry, rooms, sounds)


def test_production_explicit_targets_survive_nonconfig_entity_origin():
    production = {
        "schema": PRODUCTION_SPEC_SCHEMA, "batch_id": "explicit_origin", "seed": 1,
        "defaults": {**production_defaults(), "qa_ids": ["QA-06"]},
        "episodes": [{
            "request_id": "episode", "room_id": "a",
            "instances": [
                {"instance_id": "left", "source_class": "articulated_human", "asset_id": "red"},
                {"instance_id": "right", "source_class": "articulated_human", "asset_id": "blue"},
            ],
            "qa_targets": [{
                "qa_id": "QA-06", "branch": "moving",
                "target_instance_ids": ["left"],
                "event": {"kind": "target_audible_window"},
                "target_source": "derived_from_speaking_instances",
            }],
        }],
    }
    slots, _ = production_config_slots(production)
    assert slots[0]["qa"]["qa_targets"][0]["branch"] == "moving"
    assert slots[0]["request_overrides"]["qa_targets"][0]["target_instance_ids"] == ["left"]


def test_explicit_request_override_targets_remain_execution_inputs(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    target = {"qa_id": "QA-05", "target_instance_ids": ["source2"],
              "event": {"kind": "target_audible_window"}}
    config["slots"][0]["request_overrides"] = {"qa_targets": [target]}
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    assert row["request"]["qa_targets"][0]["target_instance_ids"] == ["source2"]
    assert row["qa_targets"] == row["request"]["qa_targets"]


def test_formal_static_placement_input_bypasses_legacy_floor_gap(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    wall = asset("wall", None, surface="wall")
    wall["entity_class"] = "rigid_object"
    wall["identity"] = {"category": "audio_playback", "object_type": "speaker"}
    registry["assets"].append(wall)
    config["slots"][0].update(
        source_classes=["rigid_static_object", "articulated_human"],
        source_asset_ids=["wall", "red"],
    )
    config["base_request"]["static_source_placement"] = {
        "catalog_path": "/catalog/support.json",
        "config": {
            "normal_tolerance_deg": 8.0,
            "plane_tolerance_m": 0.03,
            "candidate_search": {
                "grid_step_m": 0.1,
                "max_candidates": 4,
                "edge_margin_m": 0.01,
            },
        },
        "requests": [{
            "instance_id": "source1",
            "asset_id": "wall",
            "support_surface_id": "wall_surface",
        }],
    }
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    assert not any(
        gap["state"] == "interface_not_implemented"
        for gap in row["preallocation_gaps"]
    )
    assert not any(
        gap["code"] == "floor_only_placement_interface"
        for gap in row["preallocation_gaps"]
    )


def test_malformed_formal_static_placement_is_evidence_gap_not_backend_gap(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    wall = asset("wall", None, surface="wall")
    wall["entity_class"] = "rigid_object"
    wall["identity"] = {"category": "audio_playback", "object_type": "speaker"}
    registry["assets"].append(wall)
    config["slots"][0].update(
        source_classes=["rigid_static_object", "articulated_human"],
        source_asset_ids=["wall", "red"],
    )
    config["base_request"]["static_source_placement"] = {
        "catalog_path": "/catalog/support.json",
        "config": {},
        "requests": [],
    }
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    gap = next(
        gap for gap in row["preallocation_gaps"]
        if gap["code"] == "static_placement_input_missing"
    )
    assert gap["state"] == "evidence_missing_or_unsampled"
    assert not any(
        gap["state"] == "interface_not_implemented"
        for gap in row["preallocation_gaps"]
    )


def test_explicit_sound_allowlist_and_clip_fit_do_not_fallback_to_full_pool(inputs):
    config, registry, rooms, _ = deepcopy(inputs)
    config["slots"] = [{
        "room_id": "a",
        "source_classes": ["articulated_human", "articulated_human"],
        "source_asset_ids": ["red", "blue"],
        "condition_group": "identity",
        "profile": {
            "event_relation": "sequential",
            "reserve_tail_s": 3.0,
            "min_gap_between_audible_windows_s": 0.5,
        },
    }]
    sounds = [
        sound("long_red", "long_red", count=76800),
        sound("short_red", "short_red", count=32000),
        sound("long_blue", "long_blue", count=76800),
        sound("short_blue", "short_blue", count=32000),
    ]
    config["base_request"]["sound_selection"] = {
        "max_clip_s": 4.0,
        "preallocated_sound_asset_ids_by_actor": {
            "source1": ["long_red", "short_red"],
            "source2": ["long_blue", "short_blue"],
        },
    }
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    row = result["episodes"][0]
    assert [assignment["sound_asset_ids"] for assignment in row["source_assignments"]] == [
        ["short_red"], ["short_blue"]
    ]
    assert not any(
        gap["code"] == "fixed_sound_identities_exceed_profile_clip_budget"
        for gap in row["preallocation_gaps"]
    )


def test_explicit_long_sound_allowlist_keeps_real_duration_gap_without_pool_fallback(inputs):
    config, registry, rooms, _ = deepcopy(inputs)
    config["slots"] = [{
        "room_id": "a",
        "source_classes": ["articulated_human", "articulated_human"],
        "source_asset_ids": ["red", "blue"],
        "condition_group": "identity",
        "profile": {
            "event_relation": "sequential",
            "reserve_tail_s": 3.0,
            "min_gap_between_audible_windows_s": 0.5,
        },
    }]
    sounds = [
        sound("long_red", "long_red", count=76800),
        sound("short_red", "short_red", count=32000),
        sound("long_blue", "long_blue", count=76800),
        sound("short_blue", "short_blue", count=32000),
    ]
    config["base_request"]["frame_count"] = 150
    config["base_request"]["frame_rate_hz"] = 15
    config["base_request"]["sample_rate_hz"] = 16000
    config["base_request"]["sound_selection"] = {
        "max_clip_s": 7.0,
        "preallocated_sound_asset_ids_by_actor": {
            "source1": ["long_red"],
            "source2": ["long_blue"],
        },
    }
    result = prepare_batch_manifest(config, registry, rooms, sounds)
    row = result["episodes"][0]
    assert [assignment["sound_asset_ids"] for assignment in row["source_assignments"]] == [
        ["long_red"], ["long_blue"]
    ]
    gap = next(
        gap for gap in row["preallocation_gaps"]
        if gap["code"] == "fixed_sound_identities_exceed_profile_clip_budget"
    )
    assert gap["identity_substitution_applied"] is False


def test_articulated_actor_is_not_asked_for_a_support_placement_request(inputs):
    """A human anchor stands on the navmesh, so a support request for it can
    never exist. Reporting one as a gap makes a world look unfixable when the
    static devices beside it are fully specified."""
    config, registry, rooms, sounds = deepcopy(inputs)
    wall = asset("wall", None, surface="wall")
    wall["entity_class"] = "rigid_object"
    wall["identity"] = {"category": "audio_playback", "object_type": "speaker"}
    registry["assets"].append(wall)
    config["slots"][0].update(
        source_classes=["rigid_static_object", "articulated_human"],
        source_asset_ids=["wall", "red"],
    )
    config["base_request"]["static_source_placement"] = {
        "catalog_path": "/catalog/support.json",
        "config": {
            "normal_tolerance_deg": 8.0,
            "plane_tolerance_m": 0.03,
            "candidate_search": {
                "grid_step_m": 0.1,
                "max_candidates": 4,
                "edge_margin_m": 0.01,
            },
        },
        "requests": [{
            "instance_id": "source1",
            "asset_id": "wall",
            "support_surface_id": "wall_surface",
        }],
    }
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    assert not [
        gap for gap in row["preallocation_gaps"]
        if gap["code"] == "static_placement_input_missing"
    ]


def test_a_static_device_without_its_own_support_request_is_still_a_gap(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    wall = asset("wall", None, surface="wall")
    wall["entity_class"] = "rigid_object"
    wall["identity"] = {"category": "audio_playback", "object_type": "speaker"}
    registry["assets"].append(wall)
    config["slots"][0].update(
        source_classes=["rigid_static_object", "articulated_human"],
        source_asset_ids=["wall", "red"],
    )
    config["base_request"]["static_source_placement"] = {
        "catalog_path": "/catalog/support.json",
        "config": {
            "normal_tolerance_deg": 8.0,
            "plane_tolerance_m": 0.03,
            "candidate_search": {
                "grid_step_m": 0.1,
                "max_candidates": 4,
                "edge_margin_m": 0.01,
            },
        },
        "requests": [{
            "instance_id": "source9",
            "asset_id": "some_other_device",
            "support_surface_id": "wall_surface",
        }],
    }
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    gap = next(
        gap for gap in row["preallocation_gaps"]
        if gap["code"] == "static_placement_input_missing"
    )
    assert gap["asset_id"] == "wall"
    assert gap["missing_fields"] == "requests[wall]"


def test_a_static_device_with_a_blank_support_surface_id_is_still_a_gap(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    wall = asset("wall", None, surface="wall")
    wall["entity_class"] = "rigid_object"
    wall["identity"] = {"category": "audio_playback", "object_type": "speaker"}
    registry["assets"].append(wall)
    config["slots"][0].update(
        source_classes=["rigid_static_object", "articulated_human"],
        source_asset_ids=["wall", "red"],
    )
    config["base_request"]["static_source_placement"] = {
        "catalog_path": "/catalog/support.json",
        "config": {
            "normal_tolerance_deg": 8.0,
            "plane_tolerance_m": 0.03,
            "candidate_search": {
                "grid_step_m": 0.1,
                "max_candidates": 4,
                "edge_margin_m": 0.01,
            },
        },
        "requests": [{
            "instance_id": "source1",
            "asset_id": "wall",
            "support_surface_id": "   ",
        }],
    }
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    gap = next(
        gap for gap in row["preallocation_gaps"]
        if gap["code"] == "static_placement_input_missing"
    )
    assert gap["missing_fields"] == "support_surface_id[wall]"


def test_a_malformed_spec_is_still_reported_for_an_articulated_actor(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0].update(
        source_classes=["articulated_human", "articulated_human"],
        source_asset_ids=["red", "blue"],
    )
    config["base_request"]["static_source_placement"] = {
        "catalog_path": "",
        "config": {},
    }
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    gap = next(
        gap for gap in row["preallocation_gaps"]
        if gap["code"] == "static_placement_input_missing"
    )
    assert "catalog_path" in gap["missing_fields"]
    assert "config" in gap["missing_fields"]


def _speaking_by_instance(row):
    return {a["instance_id"]: a["speaking"] for a in row["source_assignments"]}


def test_a_declared_silent_competitor_silences_that_instance_and_no_other(inputs):
    """The flag names which instance is silent; a count only says how many.

    Before this was carried through, entity_instances_for_slot dropped `speaking`,
    conditioned_sampler saw no declaration and drew the silent actor at random, so
    the declared-speaking target could come back silent.
    """
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0]["entity_instances"] = [
        {"instance_id": "human_target", "role": "anchor", "speaking": True},
        {"instance_id": "dog_competitor", "speaking": False},
    ]
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    assert _speaking_by_instance(row) == {"human_target": True, "dog_competitor": False}
    silent = next(a for a in row["source_assignments"] if a["instance_id"] == "dog_competitor")
    assert silent["sound_status"] == "silent_by_request"
    assert silent["sound_asset_ids"] == []
    speaking = next(a for a in row["source_assignments"] if a["instance_id"] == "human_target")
    assert speaking["sound_asset_ids"]
    # the silent instance is still in the scene, with its own asset and actor slot
    assert [i["instance_id"] for i in row["entity_instances"]] == ["human_target", "dog_competitor"]
    assert silent["asset_id"] and silent["asset_id"] != speaking["asset_id"]
    assert row["request"]["entities"]["silent_count"] == 1
    assert row["request"]["entities"]["total_count"] == 2


def test_declared_speaking_flags_reach_the_sampler_request(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0]["entity_instances"] = [
        {"instance_id": "human_target", "role": "anchor", "speaking": True},
        {"instance_id": "dog_competitor", "speaking": False},
    ]
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    for block in (row["request"]["entity_instances"], row["request"]["entities"]["instances"]):
        assert [i.get("speaking") for i in block] == [True, False]


def test_a_silent_target_is_refused_rather_than_silencing_the_other_instance(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0]["entity_instances"] = [
        {"instance_id": "human_target", "role": "anchor", "speaking": False},
        {"instance_id": "dog_competitor", "speaking": True},
    ]
    with pytest.raises(ValueError, match="target instance must be a speaking instance"):
        prepare_batch_manifest(config, registry, rooms, sounds)


def test_a_stated_silent_count_that_contradicts_the_flags_is_named(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0]["entity_instances"] = [
        {"instance_id": "human_target", "role": "anchor", "speaking": True},
        {"instance_id": "dog_competitor", "speaking": False},
    ]
    config["slots"][0]["silent_count"] = 0
    with pytest.raises(ValueError) as caught:
        prepare_batch_manifest(config, registry, rooms, sounds)
    message = str(caught.value)
    assert "entities.silent_count is 0" in message
    assert "1 instance(s) silent" in message
    assert "dog_competitor" in message


def test_flags_that_leave_nobody_speaking_are_refused(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0]["entity_instances"] = [
        {"instance_id": "one", "speaking": False},
        {"instance_id": "two", "speaking": False},
    ]
    with pytest.raises(ValueError):
        prepare_batch_manifest(config, registry, rooms, sounds)


def test_a_non_boolean_speaking_flag_is_refused(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0]["entity_instances"] = [
        {"instance_id": "one", "speaking": "yes"},
        {"instance_id": "two"},
    ]
    with pytest.raises(ValueError, match="speaking must be true or false"):
        prepare_batch_manifest(config, registry, rooms, sounds)


def test_an_episode_that_declares_no_flags_keeps_the_stated_count(inputs):
    config, registry, rooms, sounds = deepcopy(inputs)
    config["slots"][0]["silent_count"] = 1
    row = prepare_batch_manifest(config, registry, rooms, sounds)["episodes"][0]
    assert row["request"]["entities"]["silent_count"] == 1
    assert sum(1 for a in row["source_assignments"] if not a["speaking"]) == 1


def test_production_episode_contradicting_its_own_instances_is_refused():
    production = {
        "schema": PRODUCTION_SPEC_SCHEMA,
        "batch_id": "contradiction",
        "seed": 1,
        "defaults": {
            "qa_ids": ["QA-01"], "items_per_type": 1,
            "clock": {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000},
            "rig": {"fov_deg": 85.0, "height_above_floor_m": 1.55, "motion": "static",
                    "resolution_hw": [720, 1280]},
            "audio_layouts": [{"role": "primary", "type": "binaural"}],
            "reserve_tail_s": 3.0,
            "post_assembly_convolution_gain": 0.5,
        },
        "episodes": [{
            "request_id": "ep",
            "room_id": "a",
            "entities": {"silent_count": 1},
            "instances": [
                {"instance_id": "one", "source_class": "articulated_human",
                 "asset_id": "red", "role": "anchor", "speaking": True},
                {"instance_id": "two", "source_class": "articulated_human",
                 "asset_id": "blue", "speaking": True},
            ],
        }],
        "core_groups": [],
    }
    with pytest.raises(ValueError) as caught:
        production_config_slots(production)
    assert "entities.silent_count is 1" in str(caught.value)


def test_entity_instances_for_slot_carries_the_declared_speaking_flag():
    slot = {"entity_instances": [
        {"instance_id": "a", "speaking": True},
        {"instance_id": "b", "speaking": False},
        {"instance_id": "c"},
    ]}
    rows = entity_instances_for_slot(slot, ["articulated_human"] * 3, None)
    assert [row.get("speaking") for row in rows] == [True, False, None]
