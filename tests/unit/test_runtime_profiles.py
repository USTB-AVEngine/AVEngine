from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.runtime_profiles import (
    RuntimeProfileError,
    build_asset_emitter_binding,
    build_exact_asset_bound_runtime_binding,
    load_default_room_runtime_profile_registry,
    load_default_source_asset_runtime_registry,
    resolve_room_runtime_profile,
    resolve_source_asset_alias,
    resolve_source_asset_runtime_profile,
    source_timeline_profiles,
    spear_actor_bindings,
    validate_room_runtime_links,
    validate_room_runtime_profile_registry,
    validate_source_asset_runtime_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def _artifact_ref(name: str, digest_character: str, *, formal: bool) -> dict:
    root = "release/source_assets" if formal else "tmp/source_assets"
    return {
        "root_id": "spear_repo",
        "path": f"{root}/{name}",
        "sha256": digest_character * 64,
        "size_bytes": 128,
    }


def _exact_generated_registry(*, formal: bool = False) -> dict:
    registry = deepcopy(load_default_source_asset_runtime_registry())
    asset = registry["assets"][0]
    timeline = asset["timeline"]
    unreal = asset["runtime_backends"]["spear_unreal"]
    unreal["skeletal_mesh_path"] = (
        "/Game/Test/GeneratedAnimal/SK_GeneratedAnimal.SK_GeneratedAnimal"
    )
    unreal["actor_scale"] = 0.875
    unreal["animation_paths_by_action_id"] = {
        timeline["idle_action_id"]: unreal["idle_animation"],
        timeline["walking_action_id"]: unreal["walking_animation"],
    }
    for anchor in asset["emitter_anchors"]:
        anchor["local_basis"] = {
            "schema": "avengine_asset_local_basis_v1",
            "forward_axis": [1.0, 0.0, 0.0],
            "up_axis": [0.0, 1.0, 0.0],
            "right_axis": [0.0, 0.0, 1.0],
            "handedness": "forward_cross_up_equals_right",
        }
    admission_state = "formal" if formal else "research"
    source_state = "formal_dataset_asset" if formal else "research_candidate"
    asset["admission_state"] = admission_state
    asset["asset_bound_lineage"] = {
        "schema": "avengine_spear_source_asset_v2_runtime_lineage_v1",
        "runtime_asset_id": asset["asset_id"],
        "runtime_revision": asset["revision"],
        "source_asset_v2": {
            "schema": "source_asset_v2",
            "asset_id": "spear_generated_animal_fixture_v1",
            "profile_schema_id": "generated_animal_fixture_profile_v1",
            "profile_sha256": "1" * 64,
            "request_sha256": "2" * 64,
            "lineage_group_id": "generated_animal_fixture_lineage_v1",
            "state_classification": source_state,
            "record": _artifact_ref(
                "source_asset_v2.json", "3", formal=formal
            ),
            "registry": _artifact_ref(
                "registry_manifest.json", "4", formal=formal
            ),
        },
        "geometry": {
            "lineage_kind": "bounded_same_pixel3d_mesh_repair",
            "raw_pixel3d_glb": _artifact_ref(
                "raw_pixel3d.glb", "5", formal=formal
            ),
            "tokenrig_input_glb": _artifact_ref(
                "bounded_repair.glb", "6", formal=formal
            ),
            "runtime_mesh_uri": asset["geometry"]["source_mesh_uri"],
            "runtime_glb": _artifact_ref(
                "animated_runtime.glb", "7", formal=formal
            ),
            "repair_evidence": _artifact_ref(
                "bounded_repair_evidence.json", "8", formal=formal
            ),
        },
        "tokenrig_animation_closure": _artifact_ref(
            "tokenrig_animation_closure.json", "9", formal=formal
        ),
        "ue_asset_bound_import_evidence": _artifact_ref(
            "ue_asset_bound_import.json", "a", formal=formal
        ),
        "ue_runtime_readback_evidence": _artifact_ref(
            "ue_runtime_readback.json", "b", formal=formal
        ),
        "emitter_measurement_evidence": _artifact_ref(
            "emitter_measurement.json", "c", formal=formal
        ),
        "admission": {
            "state": admission_state,
            "formal_dataset_registration_authorized": formal,
            "evidence": [
                _artifact_ref("admission_evidence.json", "d", formal=formal)
            ],
        },
    }
    return registry


def test_default_room_and_source_runtime_registries_are_independent_and_valid():
    sources = load_default_source_asset_runtime_registry()
    rooms = load_default_room_runtime_profile_registry()

    assert validate_source_asset_runtime_registry(sources) == []
    assert validate_room_runtime_profile_registry(rooms) == []
    assert validate_room_runtime_links(
        rooms, load_json(ROOT / "examples/registry/rooms/room_registry.json")
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

    assert cat["identity"] == {
        "species_id": "cat",
        "breed_id": "british_shorthair",
    }
    assert cat["realized_attributes"]["coat_profile"]["value"] == "standard_blue"
    assert cat["geometry"]["mesh_authority"] == "generated_pixel3d_target_native"
    assert timeline["body_plan_id"] == "quadruped_mammal_felid_v1"
    assert ue["walking_animation"].endswith("/Walking.Walking")
    assert ue["skeletal_mesh_binding"] == "explicit_path"
    assert ue["skeletal_mesh_path"].endswith(
        "target_animated_repaired_low_slice_edge_average_residual_continuation"
        ".target_animated_repaired_low_slice_edge_average_residual_continuation"
    )
    assert ue["floor_contact_gate"] is True
    assert emitter["asset_revision"] == cat["revision"]
    assert emitter["semantic_anchor_id"] == "muzzle"
    assert emitter["emitter_offset_m"] == [
        0.2503672200051257,
        0.23847342533548063,
        0.0,
    ]


def test_retired_labrador_record_remains_an_independent_runtime_asset():
    registry = load_default_source_asset_runtime_registry()
    labrador = resolve_source_asset_runtime_profile(
        registry,
        "generated_labrador_yellow_medium_standard_adult_research_v1",
        "pixel3d_tokenrig_ue_v1",
    )
    current_dog = resolve_source_asset_alias(registry, "current_generated_dog")
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
            "value": "standard_yellow",
        },
    }
    assert labrador["generation_request_attributes"] == {
        "size": "medium",
        "body_build": "standard",
        "life_stage": "adult",
    }
    assert labrador["geometry"]["mesh_authority"] == (
        "generated_pixel3d_target_native"
    )
    assert labrador["geometry"]["source_mesh_uri"] != (
        current_dog["geometry"]["source_mesh_uri"]
    )
    assert ue["blueprint_class_path"] != (
        spear_actor_bindings(registry)[current_dog["asset_id"]][
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


def test_habitat_native_mp3d_profile_matches_apartment_75_frame_contract():
    rooms = load_default_room_runtime_profile_registry()
    by_id = {profile["profile_id"]: profile for profile in rooms["profiles"]}
    apartment = by_id["spear_apartment_0000"]
    mp3d = by_id["habitat_mp3d_17DRP5sb8fy"]

    assert mp3d["backend_id"] == "habitat_native"
    assert mp3d["room_ref"]["room_id"] == "habitat_mp3d_example_17DRP5sb8fy"
    assert mp3d["scene"]["map_path"].endswith("room_manifest.json")
    # The dataset render contract (resolution, 75 frames, 15 Hz, HFOV) is
    # shared across backends; only warmup handling is backend-specific.
    for field in ("width", "height", "frame_count", "frame_rate_hz", "horizontal_fov_deg"):
        assert mp3d["render"][field] == apartment["render"][field]
    assert mp3d["render"]["streaming_warmup_frames"] == 0
    assert mp3d["render"]["camera_warmup_frames"] == 0
    assert "m5_1-mixed-route" in mp3d["supported_input_layouts"]


def test_habitat_native_profile_rejects_ue_map_paths():
    rooms = deepcopy(load_default_room_runtime_profile_registry())
    by_id = {profile["profile_id"]: profile for profile in rooms["profiles"]}
    by_id["habitat_mp3d_17DRP5sb8fy"]["scene"]["map_path"] = (
        "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
    )
    errors = validate_room_runtime_profile_registry(rooms)
    assert any("habitat_native map_path" in error for error in errors)


def test_unregistered_coat_profile_fails_closed():
    registry = load_default_source_asset_runtime_registry()
    broken = deepcopy(registry)
    cat = next(
        record
        for record in broken["assets"]
        if record["identity"] == {"species_id": "cat", "breed_id": "abyssinian"}
    )
    cat["realized_attributes"]["coat_profile"]["profile_id"] = (
        "cat_abyssinian_unreviewed_coat_v9"
    )
    errors = validate_source_asset_runtime_registry(broken)
    assert any("not registered in the appearance contract" in error for error in errors)


def test_coat_value_outside_registered_domain_fails_closed():
    registry = load_default_source_asset_runtime_registry()
    broken = deepcopy(registry)
    cat = next(
        record
        for record in broken["assets"]
        if record["identity"] == {"species_id": "cat", "breed_id": "abyssinian"}
    )
    cat["realized_attributes"]["coat_profile"]["value"] = "blue"
    errors = validate_source_asset_runtime_registry(broken)
    assert any("outside the registered domain" in error for error in errors)


def test_generation_request_provenance_is_separate_from_instance_baseline():
    registry = load_default_source_asset_runtime_registry()
    cat = resolve_source_asset_runtime_profile(
        registry,
        "generated_abyssinian_ruddy_medium_standard_adult_research_v1",
        "pixel3d_tokenrig_ue_v1",
    )

    # The Abyssinian breed base mesh was generated from a `slim` request
    # (breed-accurate morphology) while the instance-variation baseline is the
    # neutral `standard` level of this asset's own realizer domain.  Both are
    # recorded explicitly instead of colliding in one field.
    assert cat["generation_request_attributes"]["body_build"] == "slim"
    assert cat["realized_attributes"]["body_build"] == "standard"

    broken = deepcopy(registry)
    target = next(
        record
        for record in broken["assets"]
        if record["identity"] == {"species_id": "cat", "breed_id": "abyssinian"}
    )
    target["generation_request_attributes"]["body_build"] = "athletic"
    assert validate_source_asset_runtime_registry(broken) != []


def test_exact_asset_bound_binding_carries_generic_scale_basis_actions_and_lineage():
    registry = _exact_generated_registry()
    assert validate_source_asset_runtime_registry(registry) == []
    asset = registry["assets"][0]

    binding = build_exact_asset_bound_runtime_binding(
        registry,
        source_slot_id="source2",
        asset_id=asset["asset_id"],
        revision=asset["revision"],
    )

    assert binding["actor_scale"] == pytest.approx(0.875)
    assert binding["emitter"]["local_basis"] == {
        "schema": "avengine_asset_local_basis_v1",
        "forward_axis": [1.0, 0.0, 0.0],
        "up_axis": [0.0, 1.0, 0.0],
        "right_axis": [0.0, 0.0, 1.0],
        "handedness": "forward_cross_up_equals_right",
    }
    assert set(binding["timeline"]["animation_paths_by_action_id"]) == {
        "idle",
        "walk",
    }
    assert binding["spear_unreal"]["skeletal_mesh_path"].startswith("/Game/")
    assert (
        binding["asset_bound_lineage"]["geometry"]["lineage_kind"]
        == "bounded_same_pixel3d_mesh_repair"
    )
    assert (
        binding["asset_bound_lineage"]["source_asset_v2"]["schema"]
        == "source_asset_v2"
    )
    assert binding["admission_state"] == "research"


def test_exact_binding_is_opt_in_for_historical_research_profiles():
    registry = load_default_source_asset_runtime_registry()
    asset = registry["assets"][0]
    with pytest.raises(RuntimeProfileError, match="no exact asset-bound lineage"):
        build_exact_asset_bound_runtime_binding(
            registry,
            source_slot_id="source1",
            asset_id=asset["asset_id"],
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda asset: asset["runtime_backends"]["spear_unreal"].update(
                {"actor_scale": 0.0}
            ),
            "actor_scale",
        ),
        (
            lambda asset: asset["runtime_backends"]["spear_unreal"].update(
                {"skeletal_mesh_path": None}
            ),
            "skeletal_mesh_path",
        ),
        (
            lambda asset: asset["runtime_backends"]["spear_unreal"][
                "animation_paths_by_action_id"
            ].update({"walk": "/Game/Test/Wrong.Wrong"}),
            "animation_paths_by_action_id",
        ),
        (
            lambda asset: asset["emitter_anchors"][0]["local_basis"].update(
                {"right_axis": [0.0, 0.0, -1.0]}
            ),
            "forward cross up equals right",
        ),
        (
            lambda asset: asset["asset_bound_lineage"]["geometry"].update(
                {"lineage_kind": "template_geometry_replacement"}
            ),
            "lineage_kind",
        ),
        (
            lambda asset: asset["asset_bound_lineage"]["geometry"].update(
                {"runtime_mesh_uri": "artifact://detached/runtime.glb"}
            ),
            "runtime_mesh_uri",
        ),
    ],
)
def test_exact_asset_bound_contract_fails_closed(mutation, expected):
    registry = _exact_generated_registry()
    mutation(registry["assets"][0])
    assert any(
        expected in error
        for error in validate_source_asset_runtime_registry(registry)
    )


def test_formal_admission_requires_a_matching_immutable_exact_closure():
    missing = load_default_source_asset_runtime_registry()
    missing["assets"][0]["admission_state"] = "formal"
    assert any(
        "asset_bound_lineage" in error or "asset-bound lineage" in error
        for error in validate_source_asset_runtime_registry(missing)
    )

    mismatched = _exact_generated_registry()
    mismatched["assets"][0]["admission_state"] = "formal"
    assert any(
        "state_classification" in error
        or "admission state" in error
        or "authorization" in error
        for error in validate_source_asset_runtime_registry(mismatched)
    )

    formal = _exact_generated_registry(formal=True)
    assert validate_source_asset_runtime_registry(formal) == []

    formal["assets"][0]["asset_bound_lineage"]["admission"]["evidence"][0][
        "path"
    ] = "tmp/formal_claim.json"
    assert any(
        "immutable relative non-tmp paths" in error
        for error in validate_source_asset_runtime_registry(formal)
    )


def test_registry_validation_is_remembered_per_content(monkeypatch):
    """Every lookup re-validated the whole registry: 28 full passes over the same
    14-asset document in one two-cell design, 1.75 s of a 3.05 s run.  The result
    is now remembered per content hash, which took that run to 1.23 s."""
    from avengine import runtime_profiles as RP

    registry = load_default_source_asset_runtime_registry()
    passes = {"n": 0}
    original = RP._validate_source_asset_runtime_registry_uncached

    def counted(value):
        passes["n"] += 1
        return original(value)

    monkeypatch.setattr(RP, "_validate_source_asset_runtime_registry_uncached", counted)
    RP._VALIDATED_SOURCE_REGISTRIES.clear()

    first = RP.validate_source_asset_runtime_registry(registry)
    for _ in range(5):
        assert RP.validate_source_asset_runtime_registry(registry) == first
    assert passes["n"] == 1, "the same document must be validated once"

    # the caller may mutate its own copy of the errors without poisoning anyone
    again = RP.validate_source_asset_runtime_registry(registry)
    again.append("caller scribble")
    assert "caller scribble" not in RP.validate_source_asset_runtime_registry(registry)

    # a changed document is a different key, so it is validated again
    changed = deepcopy(registry)
    changed["assets"] = list(changed["assets"])[:-1]
    RP.validate_source_asset_runtime_registry(changed)
    assert passes["n"] == 2

    # and a broken document still reports its errors through the cache
    broken = deepcopy(registry)
    broken["assets"][0].pop("emitter_anchors", None)
    errors = RP.validate_source_asset_runtime_registry(broken)
    assert errors and RP.validate_source_asset_runtime_registry(broken) == errors
    assert passes["n"] == 3
