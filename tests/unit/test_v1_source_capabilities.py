"""Capability resolution contracts for registered source assets and the final pool.

Low-level probes use hand-built records; anything that resolves through the registry
uses the checked-in runtime registry, so the six-combination and gap assertions are
grounded in real registered metadata rather than a convenient fixture.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from avengine.dataset import source_capabilities as mod
from avengine.runtime_profiles import load_source_asset_runtime_registry

pytestmark = pytest.mark.fast_unit

REPOSITORY = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"

CONFIG = {
    "species_sound_classes": {"dog": ["dog_bark"], "cat": ["cat_meow"]},
    "object_sound_classes": {
        "television": ["speech_playback", "music_playback"],
        "smart_speaker": ["speech_playback", "music_playback"],
        "soundbar": ["speech_playback", "music_playback"],
        "bookshelf_speaker": ["speech_playback", "music_playback"],
        "floorstanding_speaker": ["speech_playback", "music_playback"],
        "toilet": ["toilet_flush"],
        "printer": ["printer"],
        "air_conditioner": ["air_conditioning"],
    },
    "speech_playback_categories": ["audio_playback"],
}


@pytest.fixture(scope="module")
def registry():
    return load_source_asset_runtime_registry(REGISTRY_PATH)


def pool_sounds():
    """A pool shaped like the produced batch pool, without touching any real PCM."""
    return [
        {"sound_asset_id": "speech_m", "sound_class": "speech_playback", "gender": "M",
         "speaker_id": "p226", "sound_identity_id": "speaker:p226",
         "metadata_status": "bridged", "human_review": {"status": "pending_human"},
         "transcript": "hello there"},
        {"sound_asset_id": "speech_f", "sound_class": "speech_playback", "gender": "F",
         "speaker_id": "p225", "sound_identity_id": "speaker:p225",
         "metadata_status": "bridged", "human_review": {"status": "pending_human"}},
        {"sound_asset_id": "speech_unknown", "sound_class": "speech_playback",
         "speaker_id": "p999", "sound_identity_id": "speaker:p999"},
        {"sound_asset_id": "bark_1", "sound_class": "dog_bark", "species_id": "dog",
         "sound_identity_id": "source:/library/bark_1.wav", "linear_gain": 1.0,
         "activity_calibration": "placeholder", "human_review": {"status": "not_measured"}},
        {"sound_asset_id": "meow_1", "sound_class": "cat_meow", "species_id": "cat",
         "sound_identity_id": "source:/library/meow_1.wav", "linear_gain": 1.0},
        {"sound_asset_id": "flush_1", "sound_class": "toilet_flush",
         "sound_identity_id": "source:/library/flush_1.wav"},
        {"sound_asset_id": "printer_1", "sound_class": "printer",
         "sound_identity_id": "source:/library/printer_1.wav"},
        {"sound_asset_id": "music_1", "sound_class": "music_playback",
         "sound_identity_id": "source:/library/music_1.wav"},
    ]


def record(entity_class, **overrides):
    base = {
        "asset_id": "asset_x", "revision": "r1", "display_label": "X",
        "entity_class": entity_class, "identity": {}, "realized_attributes": {},
        "default_emitter_anchor_id": "mouth",
        "emitter_anchors": [{"anchor_id": "mouth", "anchor_type": "mouth",
                             "offset_m": [0.0, 1.6, 0.0],
                             "offset_space": "final_scaled_asset_root"}],
        "runtime_backends": {"habitat": {"resting_pose": {"attachment_surface": "floor",
                                                          "base_plane_offset_m": 0.0,
                                                          "verdict": "level"}}},
        "timeline": {"template_id": "t", "body_plan_id": "biped_human",
                     "local_anatomical_forward_axis": [0.0, 0.0, 1.0],
                     "walk_phase_period_frames": 16,
                     "idle_action_id": "idle", "walking_action_id": "walk"},
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ families and IDs


def test_three_families_and_six_retained_two_entity_combinations():
    assert mod.SOURCE_FAMILIES == ("human", "animal", "device")
    combinations = mod.entity_combinations()
    assert len(combinations) == 6
    assert len({mod.combination_key(*pair) for pair in combinations}) == 6
    # order-insensitive
    assert mod.combination_key("device", "human") == mod.combination_key("human", "device")


def test_family_and_class_tokens_cover_every_registered_entity_class(registry):
    families = {mod.source_family(r) for r in registry["assets"]}
    assert families == {"human", "animal", "device"}
    tokens = {mod.source_class_token(r) for r in registry["assets"]}
    assert tokens == {"articulated_human", "articulated_animal", "rigid_static_object"}
    with pytest.raises(mod.SourceCapabilityError):
        mod.source_family({"asset_id": "z", "entity_class": "spaceship"})


def test_asset_instance_sound_event_and_sound_identity_are_five_distinct_ids(registry):
    asset_id = "rocketbox_human_male_adult_01_m5_1_candidate"
    plan = mod.plan_source_binding(
        registry, pool_sounds(),
        [{"asset_id": asset_id, "sound_asset_ids": ["speech_m", "speech_m"]}],
        CONFIG)
    event = plan["events"][0]
    values = {event["asset_id"], event["entity_instance_id"], event["event_id"],
              event["sound_asset_id"], event["sound_identity_id"]}
    assert len(values) == 5
    assert event["entity_instance_id"] != event["asset_id"]
    assert event["event_id"] != event["entity_instance_id"]


# ------------------------------------------------------------------------ locomotion


def test_device_locomotion_is_inapplicable_by_definition_not_a_missing_measurement():
    device = record("rigid_object", identity={"object_type": "toilet",
                                              "category": "plumbing_fixture"},
                    timeline=None)
    del device["timeline"]
    capability = mod.locomotion_capability(device)
    assert capability["state"] == mod.STATE_NOT_APPLICABLE
    assert capability["state"] != mod.STATE_EVIDENCE_MISSING
    assert "never a self-locomotion target" in capability["reason"]
    with pytest.raises(mod.SourceCapabilityError, match="cannot be a self-locomotion"):
        mod.assert_motion_target(device)


def test_articulated_locomotion_reports_registered_actions_and_missing_timeline():
    human = record("articulated_human", identity={"species_id": "human"})
    available = mod.locomotion_capability(human)
    assert available["state"] == mod.STATE_AVAILABLE
    assert available["basis"]["walking_action_id"] == "walk"

    incomplete = record("articulated_animal", identity={"species_id": "dog"})
    incomplete["timeline"] = dict(incomplete["timeline"], walking_action_id=None)
    gap = mod.locomotion_capability(incomplete)
    assert gap["state"] == mod.STATE_EVIDENCE_MISSING
    assert gap["basis"]["missing_fields"] == ["walking_action_id"]


def test_every_registered_device_refuses_self_motion_and_every_articulated_allows_it(registry):
    report = mod.capability_report(registry, pool_sounds(), CONFIG)
    by_family = {"human": set(), "animal": set(), "device": set()}
    for entry in report["assets"]:
        by_family[entry["family"]].add(entry["capability_states"]["locomotion"])
    assert by_family["device"] == {mod.STATE_NOT_APPLICABLE}
    assert by_family["human"] == {mod.STATE_AVAILABLE}
    assert by_family["animal"] == {mod.STATE_AVAILABLE}


def test_walk_request_on_a_registered_device_is_rejected(registry):
    device = next(r["asset_id"] for r in registry["assets"]
                  if mod.source_family(r) == "device")
    with pytest.raises(mod.SourceCapabilityError, match="self-locomotion"):
        mod.declare_instances(registry, [{"asset_id": device, "motion": "walk"}])
    # the same device is still declarable as a static source
    instances = mod.declare_instances(registry, [{"asset_id": device, "motion": "static"}])
    assert instances[0]["locomotion"]["state"] == mod.STATE_NOT_APPLICABLE
    assert instances[0]["motion"] == "static"


# ------------------------------------------------------------- repeated instances


def test_one_asset_backs_several_distinct_physical_instances(registry):
    asset_id = next(r["asset_id"] for r in registry["assets"]
                    if mod.source_family(r) == "animal")
    instances = mod.declare_instances(
        registry, [{"asset_id": asset_id}, {"asset_id": asset_id}])
    assert len({i["entity_instance_id"] for i in instances}) == 2
    assert {i["asset_id"] for i in instances} == {asset_id}
    assert [i["instance_ordinal"] for i in instances] == [1, 2]
    assert [i["source_slot_id"] for i in instances] == ["source1", "source2"]

    plan = mod.plan_source_binding(
        registry, pool_sounds(),
        [{"asset_id": asset_id}, {"asset_id": asset_id}], CONFIG)
    assert plan["repeated_assets"] == [asset_id]


def test_duplicate_slot_and_explicit_instance_collision_are_refused(registry):
    asset_id = next(r["asset_id"] for r in registry["assets"])
    with pytest.raises(mod.SourceCapabilityError, match="duplicate source slot"):
        mod.declare_instances(registry, [{"asset_id": asset_id, "source_slot_id": "source1"},
                                         {"asset_id": asset_id, "source_slot_id": "source1"}])
    with pytest.raises(mod.SourceCapabilityError, match="must be distinct"):
        mod.declare_instances(
            registry,
            [{"asset_id": asset_id, "entity_instance_id": "same"},
             {"asset_id": asset_id, "entity_instance_id": "same", "source_slot_id": "source2"}])


# ------------------------------------------------------------------- multiple events


def test_one_entity_may_own_several_events(registry):
    asset_id = next(r["asset_id"] for r in registry["assets"]
                    if r["identity"].get("species_id") == "dog")
    instance = mod.declare_instances(registry, [{"asset_id": asset_id}])[0]
    sounds = [s for s in pool_sounds() if s["sound_class"] == "dog_bark"] * 3
    events = mod.declare_events(registry, instance, sounds, CONFIG)
    assert len(events) == 3
    assert len({e["event_id"] for e in events}) == 3
    assert {e["entity_instance_id"] for e in events} == {instance["entity_instance_id"]}
    assert [e["event_ordinal"] for e in events] == [1, 2, 3]

    report = mod.instance_event_identity_report(events)
    assert report["grouping_key"] == "entity_instance_id"
    assert report["multi_event_instances"] == [instance["entity_instance_id"]]
    assert report["event_count"] == 3


def test_incompatible_reassignment_to_an_instance_is_refused(registry):
    dog = next(r["asset_id"] for r in registry["assets"]
               if r["identity"].get("species_id") == "dog")
    instance = mod.declare_instances(registry, [{"asset_id": dog}])[0]
    meow = [s for s in pool_sounds() if s["sound_class"] == "cat_meow"]
    with pytest.raises(mod.SourceCapabilityError, match="cannot bind to instance"):
        mod.declare_events(registry, instance, meow, CONFIG)


def test_shared_sound_identity_is_reported_and_never_becomes_instance_evidence(registry):
    dog = next(r["asset_id"] for r in registry["assets"]
               if r["identity"].get("species_id") == "dog")
    plan = mod.plan_source_binding(
        registry, pool_sounds(),
        [{"asset_id": dog, "sound_asset_ids": ["bark_1"]},
         {"asset_id": dog, "sound_asset_ids": ["bark_1"]}], CONFIG)
    identity = plan["identity"]
    assert identity["grouping_key"] == "entity_instance_id"
    assert identity["instance_count"] == 2
    shared = identity["sound_identities_shared_across_instances"]
    assert list(shared) == ["source:/library/bark_1.wav"]
    assert len(shared["source:/library/bark_1.wav"]) == 2
    assert "not instance evidence" in identity["claim_boundary"]


def test_an_event_without_an_instance_id_is_refused():
    with pytest.raises(mod.SourceCapabilityError, match="must name its entity_instance_id"):
        mod.instance_event_identity_report([{"event_id": "e1", "sound_identity_id": "s"}])


def test_instance_and_event_ids_reject_invalid_ordinals():
    assert mod.make_instance_id("a", 2) != mod.make_instance_id("a", 1)
    assert mod.make_event_id("a#instance01", 2).endswith("event02")
    for bad in (0, -1, True):
        with pytest.raises(mod.SourceCapabilityError):
            mod.make_instance_id("a", bad)


# --------------------------------------------------------------------- placement


def test_placement_states_separate_measured_unmeasured_and_unmountable():
    floor = mod.placement_capability(record("rigid_object",
                                            identity={"object_type": "toilet"}))
    assert floor["state"] == mod.STATE_AVAILABLE
    assert floor["basis"]["floor_reference_owner"] == "room_runtime_profile"

    unmountable = record("rigid_object", identity={"object_type": "air_conditioner"})
    unmountable["runtime_backends"]["habitat"]["resting_pose"] = {
        "attachment_surface": "wall", "verdict": "no_mounting_plane_found"}
    result = mod.placement_capability(unmountable)
    assert result["state"] == mod.STATE_NOT_IMPLEMENTED
    assert "no measurable mounting plane" in result["reason"]

    unmeasured = record("rigid_object", identity={"object_type": "toilet"})
    unmeasured["runtime_backends"]["habitat"] = {}
    assert mod.placement_capability(unmeasured)["state"] == mod.STATE_EVIDENCE_MISSING

    articulated = record("articulated_animal", identity={"species_id": "dog"})
    articulated["runtime_backends"]["habitat"]["resting_pose"] = {
        "attachment_surface": "floor", "base_plane_offset_m": 0.0}
    native = mod.placement_capability(articulated)
    assert native["state"] == mod.STATE_AVAILABLE
    assert native["basis"]["locomotion_placement_authority"] == "native_navigation"


def test_registered_unmountable_assets_are_reported_not_implemented(registry):
    expected = {
        r["asset_id"] for r in registry["assets"]
        if (r["runtime_backends"]["habitat"].get("resting_pose") or {}).get("verdict")
        == mod.NO_MOUNTING_PLANE_VERDICT
    }
    report = mod.capability_report(registry, pool_sounds(), CONFIG)
    reported = {e["asset_id"] for e in report["assets"]
                if e["capability_states"]["placement"] == mod.STATE_NOT_IMPLEMENTED}
    assert reported == expected
    assert expected, "the registry is expected to contain unmountable declared assets"


# --------------------------------------------------------------------- appearance


def test_missing_appearance_is_a_metadata_gap_and_never_an_absent_object():
    named = record("articulated_human", realized_attributes={"top_color": "blue"})
    assert mod.appearance_capability(named)["state"] == mod.STATE_AVAILABLE

    unnamed = record("articulated_human", realized_attributes={"life_stage": "adult"})
    result = mod.appearance_capability(unnamed)
    assert result["state"] == mod.STATE_EVIDENCE_MISSING
    assert result["state"] != mod.STATE_NOT_APPLICABLE
    assert "may still be visible and counted" in result["reason"]
    assert result["basis"]["pixel_visibility_authority"] == "native_pixel_readback"


def test_appearance_field_follows_the_family(registry):
    report = mod.capability_report(registry, pool_sounds(), CONFIG)
    fields = {}
    for entry in report["assets"]:
        fields.setdefault(entry["family"], set()).add(
            entry["capabilities"]["appearance"]["basis"]["field"])
    assert fields["human"] == {"top_color"}
    assert fields["animal"] == {"coat_profile.value"}
    assert fields["device"] <= {"finish", "body_color"}


# ------------------------------------------------------------- sound compatibility


def test_human_speech_binds_on_registered_gender_and_separates_missing_metadata():
    male = record("articulated_human", realized_attributes={"sex_or_gender_label": "male"})
    sounds = {s["sound_asset_id"]: s for s in pool_sounds()}

    ok = mod.sound_compatibility(male, sounds["speech_m"], CONFIG)
    assert ok["compatible"] and ok["state"] == mod.STATE_AVAILABLE

    mismatch = mod.sound_compatibility(male, sounds["speech_f"], CONFIG)
    assert not mismatch["compatible"]
    assert mismatch["state"] == mod.STATE_NOT_APPLICABLE

    unknown = mod.sound_compatibility(male, sounds["speech_unknown"], CONFIG)
    assert not unknown["compatible"]
    assert unknown["state"] == mod.STATE_EVIDENCE_MISSING
    assert unknown["reason"] == "speech_sound_declares_no_usable_gender_metadata"

    unlabelled = record("articulated_human", realized_attributes={})
    missing = mod.sound_compatibility(unlabelled, sounds["speech_m"], CONFIG)
    assert missing["state"] == mod.STATE_EVIDENCE_MISSING
    assert missing["reason"] == "asset_declares_no_usable_sex_or_gender_label"


def test_animal_binding_uses_species_and_refuses_another_species():
    dog = record("articulated_animal", identity={"species_id": "dog", "breed_id": "beagle"})
    sounds = {s["sound_asset_id"]: s for s in pool_sounds()}
    assert mod.sound_compatibility(dog, sounds["bark_1"], CONFIG)["compatible"]
    refused = mod.sound_compatibility(dog, sounds["meow_1"], CONFIG)
    assert not refused["compatible"]
    assert refused["state"] == mod.STATE_NOT_APPLICABLE
    assert refused["basis"]["species_id"] == "dog"


def test_device_binding_uses_object_type_and_honours_pool_allowlists():
    toilet = record("rigid_object", identity={"object_type": "toilet",
                                              "category": "plumbing_fixture"})
    sounds = {s["sound_asset_id"]: s for s in pool_sounds()}
    assert mod.sound_compatibility(toilet, sounds["flush_1"], CONFIG)["compatible"]
    wrong = mod.sound_compatibility(toilet, sounds["printer_1"], CONFIG)
    assert wrong["state"] == mod.STATE_NOT_APPLICABLE
    assert wrong["reason"] == "sound_class_is_not_declared_for_this_object_type"

    restricted = dict(sounds["flush_1"], compatible_asset_ids=["some_other_asset"])
    blocked = mod.sound_compatibility(toilet, restricted, CONFIG)
    assert not blocked["compatible"]
    assert blocked["basis"]["decided_by"] == "sound.compatible_asset_ids"


def test_a_declared_class_with_no_pool_entry_is_an_evidence_gap_not_inapplicable(registry):
    """The registered air conditioners are the real case: mapping declared, pool empty."""
    air = next((r["asset_id"] for r in registry["assets"]
                if r["identity"].get("object_type") == "air_conditioner"), None)
    assert air is not None
    result = mod.compatible_sounds(registry, air, pool_sounds(), CONFIG)
    assert result["state"] == mod.STATE_EVIDENCE_MISSING
    assert result["reason"] == "declared_sound_classes_have_no_usable_entry_in_this_pool"
    assert result["declared_sound_classes"] == ["air_conditioning"]
    assert result["candidate_count"] == 0


def test_recognizability_is_reported_as_declared_and_never_certified():
    sounds = {s["sound_asset_id"]: s for s in pool_sounds()}
    speech = mod.sound_recognizability(sounds["speech_m"])
    assert speech["certified"] is False
    assert speech["human_review_status"] == "pending_human"
    assert speech["transcript_available"] is True
    assert speech["sound_identity_id"] == "speaker:p226"
    event = mod.sound_recognizability(sounds["bark_1"])
    assert event["activity_calibration"] == "placeholder"
    assert event["transcript_available"] is False


def test_sound_identity_prefers_the_original_over_a_crop():
    assert mod.sound_identity_of({"sound_identity_id": "speaker:p1"}) == "speaker:p1"
    assert mod.sound_identity_of({"speaker_id": "p2"}) == "speaker:p2"
    assert mod.sound_identity_of({"source_origin": "/a.wav"}) == "source:/a.wav"
    assert mod.sound_identity_of({"sound_asset_id": "crop_1"}) is None


# ------------------------------------------------------------------ combinations


def test_all_six_combinations_resolve_candidates_from_real_metadata(registry):
    pool = pool_sounds()
    for pair in mod.entity_combinations():
        result = mod.combination_candidates(registry, pool, pair, CONFIG)
        assert result["state"] == mod.STATE_AVAILABLE, (pair, result["reason"])
        for slot in result["slots"]:
            assert slot["eligible_count"] > 0
            for entry in slot["assets"]:
                if entry["eligible"]:
                    assert entry["sound_candidate_count"] > 0
                    assert entry["sound_state"] == mod.STATE_AVAILABLE


def test_a_mixed_combination_keeps_its_applicable_slot_when_the_other_cannot_move(registry):
    """A device slot that is no motion target must not exclude human+device as a class."""
    result = mod.combination_candidates(
        registry, pool_sounds(), ("human", "device"), CONFIG, require_motion_target=True)
    slots = {slot["family"]: slot for slot in result["slots"]}
    assert slots["human"]["state"] == mod.STATE_AVAILABLE
    assert slots["human"]["eligible_count"] > 0
    assert slots["device"]["state"] == mod.STATE_NOT_APPLICABLE
    assert result["usable_motion_target_families"] == ["human"]
    # the combination is reported with its per-slot detail, not dropped
    assert result["state"] != mod.STATE_NOT_APPLICABLE
    assert "not excluded as a class" in result["reason"]


def test_device_only_combination_has_no_motion_target_but_keeps_its_candidates(registry):
    plain = mod.combination_candidates(registry, pool_sounds(), ("device", "device"), CONFIG)
    assert plain["state"] == mod.STATE_AVAILABLE
    assert plain["slots"][0]["eligible_count"] > 0

    motion = mod.combination_candidates(
        registry, pool_sounds(), ("device", "device"), CONFIG, require_motion_target=True)
    assert motion["usable_motion_target_families"] == []
    assert all(slot["state"] == mod.STATE_NOT_APPLICABLE for slot in motion["slots"])


def test_combination_requires_exactly_two_families(registry):
    with pytest.raises(mod.SourceCapabilityError, match="exactly two families"):
        mod.combination_candidates(registry, pool_sounds(), ("human",), CONFIG)
    with pytest.raises(mod.SourceCapabilityError, match="unknown source family"):
        mod.combination_key("human", "robot")


# ---------------------------------------------------------------------- reporting


def test_report_keeps_every_registered_asset_in_the_denominator(registry):
    report = mod.capability_report(registry, pool_sounds(), CONFIG)
    assert report["registered_asset_count"] == len(registry["assets"])
    assert {e["asset_id"] for e in report["assets"]} == {
        r["asset_id"] for r in registry["assets"]}
    assert sum(report["family_counts"].values()) == len(registry["assets"])
    for dimension, counts in report["capability_state_counts"].items():
        assert sum(counts.values()) == len(registry["assets"]), dimension
        assert set(counts) <= set(mod.CAPABILITY_STATES), dimension


def test_report_lists_gaps_without_presenting_them_as_implemented(registry):
    report = mod.capability_report(registry, pool_sounds(), CONFIG)
    for entry in report["gap_assets"]:
        assert entry["gaps"]
        assert all(state in mod.GAP_STATES for state in entry["gaps"].values())
        # a gap asset is still present in the full asset list
        assert entry["asset_id"] in {e["asset_id"] for e in report["assets"]}
    gap_ids = {e["asset_id"] for e in report["gap_assets"]}
    for entry in report["assets"]:
        non_motion = {name: state for name, state in entry["capability_states"].items()
                      if name != "locomotion"}
        if any(state in mod.GAP_STATES for state in non_motion.values()):
            assert entry["asset_id"] in gap_ids


def test_capability_gap_vocabulary_matches_the_coverage_accounting():
    from avengine.qa.batch_coverage import COVERAGE_STATES
    assert set(mod.GAP_STATES) <= set(COVERAGE_STATES)
    assert mod.STATE_AVAILABLE not in COVERAGE_STATES


def test_music_playback_binds_while_the_packaged_suite_stays_unimplemented(registry):
    result = mod.music_suite_capability(registry, pool_sounds(), CONFIG)
    assert result["state"] == mod.STATE_AVAILABLE
    assert result["pool_entry_count"] == 1
    assert result["accepting_asset_ids"]
    assert result["packaged_suite"]["state"] == mod.STATE_NOT_IMPLEMENTED
    assert "extension_point" in result["packaged_suite"]

    without_music = [s for s in pool_sounds() if s["sound_class"] != "music_playback"]
    empty = mod.music_suite_capability(registry, without_music, CONFIG)
    assert empty["state"] == mod.STATE_EVIDENCE_MISSING


# ------------------------------------------------------------- config and read-only


def test_sound_class_index_preserves_registry_order_and_is_idempotent(registry):
    index = mod.sound_class_asset_index(registry, CONFIG)
    order = [r["asset_id"] for r in registry["assets"]]
    for asset_ids in index.values():
        positions = [order.index(a) for a in asset_ids]
        assert positions == sorted(positions)
    resolved = mod.normalize_sound_class_config(CONFIG)
    assert mod.normalize_sound_class_config(resolved) is resolved
    assert mod.sound_class_asset_index(registry, resolved) == index


def test_declared_classes_separate_an_empty_human_declaration_from_a_mismatch():
    human = record("articulated_human", realized_attributes={"sex_or_gender_label": "male"})
    bark = {"sound_asset_id": "b", "sound_class": "dog_bark", "species_id": "dog"}
    silent_config = dict(CONFIG)
    silent_config.pop("human_nonverbal_sound_classes", None)
    result = mod.sound_compatibility(human, bark, silent_config)
    assert result["state"] == mod.STATE_EVIDENCE_MISSING
    assert result["reason"] == "no_human_nonverbal_sound_classes_are_configured"

    declared = dict(CONFIG, human_nonverbal_sound_classes=["laughter"])
    mismatch = mod.sound_compatibility(human, bark, declared)
    assert mismatch["state"] == mod.STATE_NOT_APPLICABLE
    assert mismatch["reason"] == "sound_class_is_not_declared_for_humans"
    laughter = {"sound_asset_id": "l", "sound_class": "laughter"}
    assert mod.sound_compatibility(human, laughter, declared)["compatible"]


def test_resolution_never_mutates_the_registry_or_the_pool(registry):
    pool = pool_sounds()
    registry_before = json.dumps(registry, sort_keys=True)
    pool_before = json.dumps(pool, sort_keys=True)
    report = mod.capability_report(registry, pool, CONFIG)
    mod.plan_source_binding(
        registry, pool,
        [{"asset_id": report["assets"][0]["asset_id"]}], CONFIG)
    assert json.dumps(registry, sort_keys=True) == registry_before
    assert json.dumps(pool, sort_keys=True) == pool_before


def test_report_detaches_its_output_from_the_registry_record(registry):
    report = mod.capability_report(registry, pool_sounds(), CONFIG)
    entry = report["assets"][0]
    entry["identity"]["species_id"] = "mutated"
    entry["realized_attributes"]["life_stage"] = "mutated"
    source = next(r for r in registry["assets"] if r["asset_id"] == entry["asset_id"])
    assert source["identity"].get("species_id") != "mutated"
    assert source["realized_attributes"].get("life_stage") != "mutated"


def test_unknown_capability_state_is_refused():
    with pytest.raises(mod.SourceCapabilityError, match="unknown capability state"):
        mod._state("probably_fine", "no")


def test_cli_reports_the_registry_without_a_pool(registry, capsys, tmp_path):
    out = tmp_path / "report.json"
    assert mod.main(["--source-registry", str(REGISTRY_PATH), "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"registered assets: {len(registry['assets'])}" in printed
    written = json.loads(out.read_text())
    assert written["registered_asset_count"] == len(registry["assets"])
    assert written["combinations"] == []
    assert written["music_suite"] is None


def test_placement_available_is_a_resting_pose_record_not_a_native_pass(registry):
    """placement available means the measurement is readable, nothing about backends."""
    report = mod.capability_report(registry, pool_sounds(), CONFIG)
    available = [entry for entry in report["assets"]
                 if entry["capability_states"]["placement"] == mod.STATE_AVAILABLE]
    assert available
    for entry in available:
        basis = entry["capabilities"]["placement"]["basis"]
        assert basis["evidence_layer"] == "registry_resting_pose_record"
        assert basis["native_placement_verified_backends"] == []
        assert "not evidence that native placement passed" in basis["claim_boundary"]
