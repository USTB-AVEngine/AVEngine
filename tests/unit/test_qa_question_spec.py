from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema

from avengine.qa.intermittent import event_records
from avengine.qa.question_spec import (
    QUESTION_SPEC_SCHEMA,
    evaluate_question_spec,
    evaluate_question_specs,
    question_type_catalog,
    scenario_requirements,
)

from test_qa_fact_table import (
    SHA_B,
    _compile,
    _moving_listener_trajectory,
    _pixel_truth_for_episode,
    _registry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "avengine_qa_question_spec_v1.schema.json"


def _controlled_inputs() -> tuple[dict, dict, dict, dict]:
    trajectory = _moving_listener_trajectory()
    declared = {
        "source1": event_records(slot_id="source1", windows=[(8000, 64000)]),
        "source2": event_records(slot_id="source2", windows=[(16000, 72000)]),
    }
    facts = _compile(
        registry=_registry(),
        declared_events_by_slot=declared,
        sensor_rig_trajectory=trajectory,
        pixel_visibility_truth=_pixel_truth_for_episode(trajectory),
        camera={"hfov_degrees": 105.0, "resolution_hw": [8, 10]},
    )
    asset_registry = _registry()
    sound_registry = {
        "registry_id": "question_spec_sound_canary_v1",
        "sounds": [
            {
                "sound_asset_id": "dog_bark",
                "species": "dog",
                "path": "/dry/dog.wav",
                "sha256": SHA_B,
            },
            {
                "sound_asset_id": "cat_meow",
                "species": "cat",
                "path": "/dry/cat.wav",
                "sha256": SHA_B,
            },
        ],
    }
    bindings = {
        event["event_id"]: (
            "dog_bark" if event["source_slot_id"] == "source1" else "cat_meow"
        )
        for event in facts["sound_events"]
    }
    return facts, asset_registry, sound_registry, bindings


def _specs() -> list[dict]:
    return [
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-001",
            "question_type": "appearance_to_speaking",
            "selectors": {
                "appearance_field": "breed_id",
                "appearance_value": "dog_breed",
            },
        },
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-002",
            "question_type": "sound_to_appearance",
            "selectors": {
                "sound_asset_id": "dog_bark",
                "appearance_field": "coat_value",
            },
        },
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-003",
            "question_type": "who_spoke_first",
            "selectors": {},
        },
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-004",
            "question_type": "speaker_side",
            "selectors": {"sound_asset_id": "dog_bark", "frame_index": 50},
        },
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-005",
            "question_type": "overlapping_speech",
            "selectors": {"sound_asset_ids": ["dog_bark", "cat_meow"]},
        },
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-006",
            "question_type": "speaking_while_moving",
            "selectors": {"sound_asset_id": "cat_meow"},
        },
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-007",
            "question_type": "offscreen_to_onscreen",
            "selectors": {"target_instance_id": "source1"},
        },
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-008",
            "question_type": "occlusion_while_speaking",
            "selectors": {"sound_asset_id": "dog_bark", "frame_index": 35},
        },
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-009",
            "question_type": "reappeared_after_occlusion",
            "selectors": {"target_instance_id": "source1"},
        },
        {
            "schema": QUESTION_SPEC_SCHEMA,
            "spec_id": "QS-010",
            "question_type": "occluder_identity",
            "selectors": {"target_instance_id": "source1", "frame_index": 35},
        },
    ]


def test_fixed_catalog_has_ten_plain_language_types() -> None:
    catalog = question_type_catalog()
    assert [entry["index"] for entry in catalog] == list(range(1, 11))
    assert [entry["name_zh"] for entry in catalog] == [
        "外貌→是否发声",
        "声音/内容→外貌",
        "谁先发声",
        "发声者左右",
        "重叠发声",
        "发声时是否运动",
        "画外→入画",
        "发声时遮挡状态",
        "遮挡后重新出现",
        "遮挡者身份",
    ]


def test_specs_validate_and_only_emit_registry_requirements() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for spec in _specs():
        jsonschema.validate(spec, schema)
        assert set(spec) == {"schema", "spec_id", "question_type", "selectors"}
        assert "question" not in spec
        requirements = scenario_requirements(spec)
        assert requirements["generation_policy"] == "select_from_controlled_registries_only"
        assert requirements["required_facts"]
        assert requirements["required_modalities"]


def test_dynamic_pixel_canary_passes_first_nine_and_truthfully_rejects_tenth() -> None:
    facts, assets, sounds, bindings = _controlled_inputs()
    results = evaluate_question_specs(
        _specs(),
        facts=facts,
        asset_registry=assets,
        sound_registry=sounds,
        event_sound_bindings=bindings,
    )
    assert [result["status"] for result in results] == ["pass"] * 9 + ["unsupported"]
    assert [result["answer"]["value"] for result in results[:9]] == [
        "yes",
        "black",
        "source1",
        "left",
        "yes",
        "yes",
        "yes",
        "fully_occluded",
        "yes",
    ]
    assert results[3]["evidence"]["azimuth_deg"] < -5.0
    assert results[6]["evidence"]["entry_frame_indices"] == [60]
    assert results[7]["evidence"]["frame_index"] == 35
    assert results[8]["evidence"]["reappeared_frames"] == [60]
    assert results[9]["reason"]["code"] == "missing_occluder_identity"
    assert all(
        "not a model-ablation claim"
        in next(
            check["detail"]
            for check in result["checks"]
            if check["name"] == "modality_necessity"
        )
        for result in results[:9]
    )


def test_simultaneous_first_speakers_are_rejected_instead_of_guessed() -> None:
    facts, assets, sounds, _ = _controlled_inputs()
    continuous = _compile(
        sensor_rig_trajectory=_moving_listener_trajectory(),
    )
    bindings = {
        event["event_id"]: (
            "dog_bark" if event["source_slot_id"] == "source1" else "cat_meow"
        )
        for event in continuous["sound_events"]
    }
    result = evaluate_question_spec(
        _specs()[2],
        facts=continuous,
        asset_registry=assets,
        sound_registry=sounds,
        event_sound_bindings=bindings,
    )
    assert result["status"] == "rejected"
    assert result["reason"]["code"] == "answer_not_unique"
    assert result["evidence"]["source_slots"] == ["source1", "source2"]


def test_unregistered_sound_and_missing_pixel_truth_are_executable_rejections() -> None:
    facts, assets, sounds, bindings = _controlled_inputs()
    unregistered = copy.deepcopy(_specs()[1])
    unregistered["selectors"]["sound_asset_id"] = "wolf_howl"
    result = evaluate_question_spec(
        unregistered,
        facts=facts,
        asset_registry=assets,
        sound_registry=sounds,
        event_sound_bindings=bindings,
    )
    assert result["status"] == "rejected"
    assert result["reason"]["code"] == "registry_reference_missing"

    no_pixels = copy.deepcopy(facts)
    del no_pixels["visibility"]["pixel_truth"]
    result = evaluate_question_spec(
        _specs()[7],
        facts=no_pixels,
        asset_registry=assets,
        sound_registry=sounds,
        event_sound_bindings=bindings,
    )
    assert result["status"] == "rejected"
    assert result["reason"]["code"] == "facts_not_observable"


def test_registry_provenance_mismatch_and_hash_like_id_are_rejected() -> None:
    facts, assets, sounds, bindings = _controlled_inputs()
    tampered = copy.deepcopy(sounds)
    tampered["sounds"][0]["sha256"] = "c" * 64
    result = evaluate_question_spec(
        _specs()[0],
        facts=facts,
        asset_registry=assets,
        sound_registry=tampered,
        event_sound_bindings=bindings,
    )
    assert result["status"] == "rejected"
    assert result["reason"]["code"] == "sound_provenance_mismatch"

    bad_id = copy.deepcopy(_specs()[0])
    bad_id["spec_id"] = "a" * 64
    result = evaluate_question_spec(
        bad_id,
        facts=facts,
        asset_registry=assets,
        sound_registry=sounds,
        event_sound_bindings=bindings,
    )
    assert result["status"] == "rejected"
    assert result["reason"]["code"] == "invalid_question_spec"
