from __future__ import annotations

from copy import deepcopy

import numpy as np

from avengine.qa.unified_catalog import (
    derive_sound_class_answer_domain,
    with_derived_sound_class_answer_domain,
)
from avengine.rooms.qa_evidence import inspect_registered_appearance
from avengine.runtime_profiles import (
    load_default_source_asset_runtime_registry,
    validate_new_source_asset_runtime_profile,
    validate_source_asset_runtime_registry,
)


SOUND_POOL = [
    "air_conditioning",
    "alarm_beep",
    "alarm_bell",
    "alarm_clock",
    "any_audioset_class_playback",
    "bathtub_filling_washing",
    "blender",
    "busy_signal",
    "cat_meow",
    "cellphone_vibration_alert",
    "chime",
    "clock_tick",
    "crackle",
    "ding_dong",
    "dog_bark",
    "doorbell",
    "doorbell_chime",
    "drip",
    "fire",
    "fire_alarm",
    "gurgling",
    "microwave_beep",
    "microwave_hum",
    "music_playback",
    "phone_ring",
    "printer",
    "ringtone",
    "sink_filling_washing",
    "smoke_alarm",
    "speech_playback",
    "telephone",
    "telephone_bell_ringing",
    "telephone_dialing_dtmf",
    "toilet_flush",
    "water_tap_faucet",
]


def test_canonical_registry_retains_d2_fields_without_global_rejection() -> None:
    registry = load_default_source_asset_runtime_registry()
    assert len(registry["assets"]) == 59
    assert validate_source_asset_runtime_registry(registry) == []
    for record in registry["assets"]:
        assert record["display_label"]
        assert record["display_label_zh"]
        attributes = record["realized_attributes"]
        assert any(
            attributes.get(field)
            for field in ("finish", "surface_finish", "body_color", "top_color")
        ) or attributes.get("coat_profile", {}).get("value")
        pose = record["runtime_backends"]["habitat"]["resting_pose"]
        assert pose["attachment_surface"] in {"floor", "wall", "ceiling"}
        assert isinstance(pose["attachment_surface_assumed"], bool)


def test_registry_rejects_missing_or_unvocabular_appearance_declarations() -> None:
    registry = load_default_source_asset_runtime_registry()
    human = next(
        item
        for item in registry["assets"]
        if item["asset_id"] == "rocketbox_human_male_adult_01_m5_1_candidate"
    )

    missing = deepcopy(registry)
    missing_human = next(
        item
        for item in missing["assets"]
        if item["asset_id"] == human["asset_id"]
    )
    missing_human["realized_attributes"].pop("top_color")
    assert any(
        "one of" in error
        for error in validate_new_source_asset_runtime_profile(missing_human)
    )

    unknown = deepcopy(registry)
    unknown_human = next(
        item
        for item in unknown["assets"]
        if item["asset_id"] == human["asset_id"]
    )
    unknown_human["realized_attributes"]["top_color"] = "chartreuse"
    assert any(
        "outside the native RGB appearance vocabulary" in error
        for error in validate_new_source_asset_runtime_profile(unknown_human)
    )

    missing_pose = deepcopy(registry)
    pose = next(
        item
        for item in missing_pose["assets"]
        if item["asset_id"] == human["asset_id"]
    )["runtime_backends"]["habitat"]["resting_pose"]
    pose.pop("attachment_surface_assumed")
    assert any(
        "must set attachment_surface_assumed=false" in error
        for error in validate_new_source_asset_runtime_profile(
            next(
                item
                for item in missing_pose["assets"]
                if item["asset_id"] == human["asset_id"]
            )
        )
    )


def test_legacy_registry_keeps_assumed_rigid_assets_loadable() -> None:
    registry = load_default_source_asset_runtime_registry()
    true_rigid = [
        item
        for item in registry["assets"]
        if item["entity_class"] == "rigid_object"
        and item["runtime_backends"]["habitat"]["resting_pose"].get(
            "attachment_surface_assumed"
        )
        is True
    ]
    assert len(true_rigid) == 22
    assert validate_source_asset_runtime_registry(registry) == []

    # A legacy record may omit the new-only metadata and still be usable by
    # the compatibility loader.
    legacy = deepcopy(registry)
    record = next(
        item for item in legacy["assets"] if item["asset_id"] == true_rigid[0]["asset_id"]
    )
    record.pop("display_label_zh")
    record["realized_attributes"].pop("finish")
    record["runtime_backends"]["habitat"]["resting_pose"].pop(
        "attachment_surface_assumed"
    )
    assert validate_source_asset_runtime_registry(legacy) == []


def test_new_asset_contract_accepts_the_now_observed_standard_seal_point() -> None:
    """A pointed coat has a colour-family predicate, so it may be registered."""
    registry = load_default_source_asset_runtime_registry()
    record = next(
        item
        for item in registry["assets"]
        if item["asset_id"] == "generated_siamese_standard_seal_point_research_v1"
    )
    errors = validate_new_source_asset_runtime_profile(record)
    assert not any("outside the native RGB appearance vocabulary" in error for error in errors)


def test_new_asset_contract_still_rejects_a_value_with_no_classifier() -> None:
    from copy import deepcopy

    registry = load_default_source_asset_runtime_registry()
    record = deepcopy(next(
        item
        for item in registry["assets"]
        if item["asset_id"] == "generated_siamese_standard_seal_point_research_v1"
    ))
    record["realized_attributes"] = {"finish": "iridescent_teal_flake"}
    errors = validate_new_source_asset_runtime_profile(record)
    assert any("outside the native RGB appearance vocabulary" in error for error in errors)


def test_sound_pool_labels_and_shared_domain_respect_explicit_pool_boundary() -> None:
    domain = derive_sound_class_answer_domain(
        SOUND_POOL,
        observed_events=[{"sound_class": "outside_pool"}],
    )
    assert len(domain["values"]) == 34
    assert "any_audioset_class_playback" not in domain["values"]
    assert {"alarm_beep", "fire_alarm", "ringtone"} <= set(domain["values"])
    assert domain["boundary"] == "explicit_configured_pool"
    assert domain["excluded"] == [{
        "sound_class": "any_audioset_class_playback",
        "reason": "owner_policy_exclusion_any_audioset_class_playback",
    }]
    assert "outside_pool" not in domain["values"]

    facts = {
        "sampling": {
            "acceptance_policy": {
                "question_mode": "ordinary_observation",
                "sound_class_options": SOUND_POOL,
            }
        },
        "events": [
            {"sound_class": "fire_alarm"},
            {"sound_class": "outside_pool"},
        ],
    }
    derived = with_derived_sound_class_answer_domain(facts)
    policy = derived["sampling"]["acceptance_policy"]
    assert policy["sound_class_options"] == domain["values"]
    assert derived["sound_class_answer_domain"] == domain

    observed_only = derive_sound_class_answer_domain(
        None,
        observed_events=[
            {"sound_class": "fire_alarm"},
            {"sound_class": "any_audioset_class_playback"},
        ],
    )
    assert observed_only["boundary"] == "observed_explicit_events"
    assert observed_only["values"] == ["fire_alarm"]


def test_new_sable_and_light_gray_classifier_rules_are_observable() -> None:
    sable = np.full((64, 64, 3), [80, 50, 25], dtype=np.uint8)
    gray = np.full((64, 64, 3), [150, 150, 150], dtype=np.uint8)
    assert inspect_registered_appearance(
        sable,
        np.ones((64, 64), dtype=bool),
        "standard_sable",
        entity_kind="animal",
    )["status"] == "pass"
    assert inspect_registered_appearance(
        gray,
        np.ones((64, 64), dtype=bool),
        "light_gray",
        entity_kind="device",
    )["status"] == "pass"
