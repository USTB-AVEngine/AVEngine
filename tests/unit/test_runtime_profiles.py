from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.runtime_profiles import (
    RuntimeProfileError,
    build_asset_emitter_binding,
    load_default_room_runtime_profile_registry,
    load_default_source_asset_runtime_registry,
    resolve_room_runtime_profile,
    resolve_source_asset_alias,
    source_timeline_profiles,
    spear_actor_bindings,
    validate_room_runtime_links,
    validate_room_runtime_profile_registry,
    validate_source_asset_runtime_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def test_default_room_and_source_runtime_registries_are_independent_and_valid():
    sources = load_default_source_asset_runtime_registry()
    rooms = load_default_room_runtime_profile_registry()

    assert validate_source_asset_runtime_registry(sources) == []
    assert validate_room_runtime_profile_registry(rooms) == []
    assert validate_room_runtime_links(
        rooms, load_json(ROOT / "examples/m6/rooms/room_registry.json")
    ) == []
    assert "sounds" not in sources
    assert "assets" not in rooms


def test_source_alias_resolves_timeline_emitter_and_ue_from_one_asset_record():
    registry = load_default_source_asset_runtime_registry()
    cat = resolve_source_asset_alias(registry, "current_generated_cat")
    asset_id = cat["asset_id"]
    timeline = source_timeline_profiles(registry)[asset_id]
    ue = spear_actor_bindings(registry)[asset_id]
    emitter = build_asset_emitter_binding(
        registry, source_slot_id="source2", asset_id=asset_id
    )

    assert cat["identity"] == {"species_id": "cat", "breed_id": "abyssinian"}
    assert cat["realized_attributes"]["coat_profile"]["value"] == "ruddy"
    assert cat["geometry"]["mesh_authority"] == "generated_pixel3d_target_native"
    assert timeline["body_plan_id"] == "quadruped_mammal_felid_v1"
    assert ue["walking_animation"].endswith("/Walking.Walking")
    assert ue["skeletal_mesh_binding"] == "blueprint_component"
    assert ue["skeletal_mesh_path"] is None
    assert ue["floor_contact_gate"] is True
    assert emitter["asset_revision"] == cat["revision"]
    assert emitter["semantic_anchor_id"] == "muzzle"
    assert emitter["emitter_offset_m"] == [
        0.38869346364905827,
        0.16641961991328985,
        0.0,
    ]


def test_new_labrador_is_an_independent_generated_runtime_asset():
    registry = load_default_source_asset_runtime_registry()
    labrador = resolve_source_asset_alias(registry, "runtime_interface_labrador")
    border_collie = resolve_source_asset_alias(registry, "current_generated_dog")
    asset_id = labrador["asset_id"]
    ue = spear_actor_bindings(registry)[asset_id]
    emitter = build_asset_emitter_binding(
        registry, source_slot_id="source1", asset_id=asset_id
    )

    assert labrador["identity"] == {
        "species_id": "dog",
        "breed_id": "labrador_retriever",
    }
    assert labrador["realized_attributes"] == {
        "size": "medium",
        "body_build": "standard",
        "life_stage": "adult",
        "coat_profile": {
            "profile_id": "dog_labrador_retriever_coat_v1",
            "value": "yellow",
        },
    }
    assert labrador["geometry"]["mesh_authority"] == (
        "generated_pixel3d_target_native"
    )
    assert labrador["geometry"]["source_mesh_uri"] != (
        border_collie["geometry"]["source_mesh_uri"]
    )
    assert ue["blueprint_class_path"] != (
        spear_actor_bindings(registry)[border_collie["asset_id"]][
            "blueprint_class_path"
        ]
    )
    assert ue["skeletal_mesh_binding"] == "blueprint_component"
    assert ue["idle_animation"].endswith("/Idle.Idle")
    assert ue["walking_animation"].endswith("/Walking.Walking")
    assert emitter["emitter_offset_m"] == [0.454, 0.585, 0.0]


def test_runtime_selection_fails_closed_for_unknown_or_wrong_revision():
    registry = load_default_source_asset_runtime_registry()
    with pytest.raises(RuntimeProfileError, match="unregistered source asset"):
        build_asset_emitter_binding(
            registry, source_slot_id="source1", asset_id="future_unknown_cat"
        )
    cat = resolve_source_asset_alias(registry, "current_generated_cat")
    with pytest.raises(RuntimeProfileError, match="revision does not resolve"):
        build_asset_emitter_binding(
            registry,
            source_slot_id="source1",
            asset_id=cat["asset_id"],
            revision="wrong_revision",
        )


def test_skeletal_mesh_binding_policy_is_explicit_and_consistent():
    registry = load_default_source_asset_runtime_registry()
    invalid = deepcopy(registry)
    unreal = invalid["assets"][0]["runtime_backends"]["spear_unreal"]
    unreal["skeletal_mesh_binding"] = "explicit_path"
    assert any(
        "skeletal_mesh_path" in error
        for error in validate_source_asset_runtime_registry(invalid)
    )


def test_registry_rejects_duplicate_asset_ids_and_unresolved_default_room():
    sources = load_default_source_asset_runtime_registry()
    duplicate = deepcopy(sources["assets"][0])
    duplicate["revision"] = "second_runtime_revision"
    sources["assets"].append(duplicate)
    assert any(
        "one selected runtime revision per asset ID" in error
        for error in validate_source_asset_runtime_registry(sources)
    )

    rooms = load_default_room_runtime_profile_registry()
    rooms["default_profile_id"] = "missing_room"
    assert any(
        "default_profile_id does not resolve" in error
        for error in validate_room_runtime_profile_registry(rooms)
    )
    with pytest.raises(RuntimeProfileError, match="unknown room runtime profile"):
        resolve_room_runtime_profile(
            load_default_room_runtime_profile_registry(), "missing_room"
        )


def test_room_registry_can_describe_a_future_adapter_without_weakening_current_runner():
    rooms = load_default_room_runtime_profile_registry()
    future = deepcopy(rooms["profiles"][0])
    future["profile_id"] = "future_habitat_room"
    future["revision"] = "scene_v1"
    future["backend_id"] = "habitat_sim"
    future["adapter_id"] = "habitat_room_v1"
    future["scene"]["map_path"] = "artifact://rooms/future_room/scene.glb"
    future["supported_input_layouts"] = ["asset-bound-batch"]
    rooms["profiles"].append(future)

    assert validate_room_runtime_profile_registry(rooms) == []
