from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from avengine.contracts.json_io import load_json
from avengine.registry.entities import (
    load_animal_template_registry,
    load_entity_asset_registry,
    select_animal_template,
    validate_animal_template_registry,
    validate_entity_template_bindings,
)
from avengine.registry.registry import M6RegistryError, bind_content_hash, schema_path
from avengine.registry.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
    resolve_source_endpoint_bindings,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRIES = ROOT / "examples" / "m6" / "registries"


def test_registry_registry_schemas_are_valid_draft_2020_12() -> None:
    for schema_name in (
        "avengine_m6_entity_asset_registry_v1",
        "avengine_m6_static_object_marker_visual_approval_v1",
        "avengine_m6_animal_template_registry_v1",
        "avengine_m6_source_endpoint_registry_v1",
        "avengine_m6_sound_asset_registry_v1",
        "avengine_m6_flag_definition_registry_v1",
    ):
        Draft202012Validator.check_schema(load_json(schema_path(schema_name)))


def test_checked_in_registries_validate_and_resolve_legacy_beagle_instances() -> None:
    entities = load_entity_asset_registry(REGISTRIES / "entity_assets_v1.json")
    templates = load_animal_template_registry(REGISTRIES / "animal_templates_v1.json")
    endpoints = load_source_endpoint_registry(REGISTRIES / "source_endpoints_v1.json")
    sounds = load_sound_asset_registry(REGISTRIES / "sound_assets_v1.json")

    assert validate_entity_template_bindings(entities, templates) == []
    resolved = resolve_source_endpoint_bindings(endpoints, entities)
    resolved_by_id = {item.source_endpoint_id: item for item in resolved}
    assert tuple(
        resolved_by_id[source_id].entity_instance_id
        for source_id in ("beagle_0_muzzle", "beagle_1_muzzle")
    ) == ("beagle_0", "beagle_1")
    assert all(
        resolved_by_id[source_id].emitter_anchor_id == "muzzle"
        for source_id in ("beagle_0_muzzle", "beagle_1_muzzle")
    )
    assert all(item.persistent_when_silent for item in resolved)
    assert all(
        resolved_by_id[source_id].source_visibility_mode == "logical_point"
        for source_id in (
            "m6x_marker_front_speaker",
            "m6x_marker_rear_speaker",
        )
    )
    dog_sound = next(
        item
        for item in sounds["sound_assets"]
        if item["sound_asset_id"] == "dog_beagle_v2_scheduled_dry"
    )
    assert dog_sound["admissibility"] == "research"
    assert dog_sound["provenance"]["rights_status"] == "unresolved"
    assert dog_sound["permitted_event_usage"] == [
        "counterfactual_route_swap",
        "intermittent_events",
        "one_active_of_n",
        "sequential_sources",
        "simultaneous_subset",
    ]


def test_beagle_appearance_domains_are_data_driven_and_breed_scoped() -> None:
    templates = load_animal_template_registry(REGISTRIES / "animal_templates_v1.json")
    domains = templates["templates"][0]["appearance_domains"]
    assert domains["size"] == ["small", "medium", "large"]
    assert domains["body_build"] == ["slim", "standard", "stocky"]
    assert domains["life_stage"] == ["young", "adult", "senior"]
    assert domains["coat_profile"] == {
        "profile_id": "dog_beagle_tricolor_v1",
        "values": ["light_tricolor", "standard_tricolor", "dark_tricolor"],
    }


def test_template_selection_accepts_in_range_beagle() -> None:
    templates = load_animal_template_registry(REGISTRIES / "animal_templates_v1.json")
    result = select_animal_template(
        templates,
        {
            "body_plan_id": "quadruped_mammal_canid_v1",
            "morphotype_id": "beagle",
            "measurements": {
                "bbox_height_to_length": 0.91,
                "bbox_width_to_length": 0.32,
                "shoulder_height_m": 0.36,
            },
        },
    )
    assert result.status == "selected"
    assert result.selected_template_id == "rocketbox_dog_beagle_01"
    assert result.rejection_code is None


def test_template_selection_rejects_ood_without_silent_dog_fallback() -> None:
    templates = load_animal_template_registry(REGISTRIES / "animal_templates_v1.json")
    result = select_animal_template(
        templates,
        {
            "body_plan_id": "quadruped_mammal_canid_v1",
            "morphotype_id": "beagle",
            "measurements": {
                "bbox_height_to_length": 1.6,
                "bbox_width_to_length": 0.8,
                "shoulder_height_m": 0.8,
            },
        },
    )
    assert result.status == "rejected"
    assert result.selected_template_id is None
    assert result.nearest_template_id == "rocketbox_dog_beagle_01"
    assert result.rejection_code == "morphology_out_of_distribution"
    assert set(result.candidates[0].exceeded_dimensions) == {
        "bbox_height_to_length",
        "bbox_width_to_length",
        "shoulder_height_m",
    }


def test_unknown_morphotype_is_a_structured_rejection_not_generic_dog() -> None:
    templates = load_animal_template_registry(REGISTRIES / "animal_templates_v1.json")
    result = select_animal_template(
        templates,
        {
            "body_plan_id": "quadruped_mammal_canid_v1",
            "morphotype_id": "unknown_canid",
            "measurements": {
                "bbox_height_to_length": 0.9,
                "bbox_width_to_length": 0.3,
                "shoulder_height_m": 0.36,
            },
        },
    )
    assert result.status == "rejected"
    assert result.selected_template_id is None
    assert result.nearest_template_id is None
    assert result.rejection_code == "no_morphotype_candidate"


def test_registry_hash_drift_fails_closed() -> None:
    templates = load_json(REGISTRIES / "animal_templates_v1.json")
    templates["templates"][0]["status"] = "research"
    errors = validate_animal_template_registry(templates)
    assert "registry_content_sha256 does not match canonical content" in errors
    rebound = bind_content_hash(templates)
    assert (
        "registry_content_sha256 does not match canonical content"
        not in validate_animal_template_registry(rebound)
    )


def test_unregistered_anchor_binding_fails_closed() -> None:
    entities = load_entity_asset_registry(REGISTRIES / "entity_assets_v1.json")
    endpoints = load_source_endpoint_registry(REGISTRIES / "source_endpoints_v1.json")
    broken = deepcopy(endpoints)
    broken["source_endpoints"][0]["binding"]["emitter_anchor_id"] = "tail_speaker"
    broken = bind_content_hash(broken)
    with pytest.raises(M6RegistryError, match="unregistered emitter anchor"):
        resolve_source_endpoint_bindings(broken, entities)
