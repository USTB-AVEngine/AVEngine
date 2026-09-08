from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.qa.batch_manifest import (
    CLASS_PAIRS, CONDITION_GROUPS, build_scaleup_slots, class_pair_condition_group_crosstab,
    collect_batch_outcomes, format_class_pair_condition_group_crosstab, grouped_splits,
    prepare_batch_manifest, program_seconds_for_durations, scatter_condition_groups, sound_identity,
)
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
