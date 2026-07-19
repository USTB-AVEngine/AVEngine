from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from avengine.contracts.json_io import load_json
from avengine.m6.room_providers import (
    LegacyUEApartmentRoomProvider,
    Matterport3DRoomProvider,
    ReplicaCADRoomProvider,
    provider_for_id,
    providers_from_registry,
)
from avengine.m6.rooms import find_room_record


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = load_json(REPOSITORY_ROOT / "examples/m6/rooms/room_registry.json")


def test_provider_registry_exposes_all_four_adapters() -> None:
    providers = providers_from_registry(REGISTRY)

    assert {provider.provider_id for provider in providers} == {
        "blender_custom",
        "replica_cad",
        "legacy_ue_apartment",
        "matterport3d",
    }


def test_custom_provider_resolves_checked_in_inputs_but_not_generated_audio() -> None:
    record = find_room_record(REGISTRY, "blender_custom_two_zone_v1")
    provider = provider_for_id(record["provider_id"])

    resolution = provider.resolve_room(
        record, repository_root=REPOSITORY_ROOT, environment={}
    )
    acoustic = provider.acoustic_representation(
        record,
        "custom_two_zone_acoustic_v1",
        repository_root=REPOSITORY_ROOT,
        environment={},
    )

    assert resolution.resources["custom_visual_surface"].status == "pass"
    assert resolution.resources["custom_navmesh"].status == "pass"
    assert resolution.resources["custom_acoustic_package"].status == "not_run"
    assert acoustic.status == "not_run"
    assert acoustic.producer == "avengine.m3.compiler:compile_custom_acoustic_scene"


def test_compiled_representation_preserves_output_hash_failure() -> None:
    record = deepcopy(find_room_record(REGISTRY, "blender_custom_two_zone_v1"))
    output = next(
        resource
        for resource in record["resources"]
        if resource["resource_id"] == "custom_acoustic_package"
    )
    output["location"] = {
        "kind": "repository_relative",
        "path": "README.md",
    }
    output["sha256"] = "0" * 64
    provider = provider_for_id(record["provider_id"])

    acoustic = provider.acoustic_representation(
        record,
        "custom_two_zone_acoustic_v1",
        repository_root=REPOSITORY_ROOT,
        environment={},
    )

    assert acoustic.status == "fail"
    assert "SHA-256 mismatch" in (acoustic.reason or "")


def test_replicacad_provider_requires_declared_environment_root() -> None:
    record = find_room_record(REGISTRY, "replicacad_apt_0")
    provider = ReplicaCADRoomProvider()

    resolution = provider.resolve_room(
        record, repository_root=REPOSITORY_ROOT, environment={}
    )

    assert resolution.status == "blocked"
    assert all(
        "/data/datasets" not in (resource.reason or "")
        for resource in resolution.resources.values()
    )
    assert any("AVENGINE_REPLICACAD_ROOT" in blocker for blocker in resolution.blockers)


def test_legacy_provider_keeps_aabb_diagnostic_non_authoritative() -> None:
    record = find_room_record(REGISTRY, "legacy_ue_apartment_0000_v1")
    provider = LegacyUEApartmentRoomProvider()

    aabb = provider.acoustic_representation(
        record,
        "legacy_route_aabb_diagnostic",
        repository_root=REPOSITORY_ROOT,
        environment={},
    )
    real_surface = provider.acoustic_representation(
        record,
        "legacy_real_surface_acoustic_v1",
        repository_root=REPOSITORY_ROOT,
        environment={},
    )

    assert aabb.status == "pass"
    assert next(
        item
        for item in record["acoustic_representations"]
        if item["representation_id"] == aabb.representation_id
    )["role"] == "diagnostic_only"
    assert real_surface.status == "blocked"
    assert "AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT" in (real_surface.reason or "")


def test_mp3d_provider_never_conflates_raw_and_derived_availability() -> None:
    record = find_room_record(REGISTRY, "habitat_mp3d_example_17DRP5sb8fy")
    provider = Matterport3DRoomProvider()

    raw = provider.acoustic_representation(
        record,
        "mp3d_17DRP5sb8fy_raw_source_v1",
        repository_root=REPOSITORY_ROOT,
        environment={},
    )
    derived = provider.acoustic_representation(
        record,
        "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
        repository_root=REPOSITORY_ROOT,
        environment={},
    )

    assert raw.representation_id != derived.representation_id
    assert raw.build_mode == "reference"
    assert derived.build_mode == "derive"
    assert raw.status == "blocked"
    assert derived.status == "blocked"
