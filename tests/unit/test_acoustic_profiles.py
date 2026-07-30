from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from avengine.acoustic_profiles import (
    SOLVER_BACKEND_ID,
    AcousticProfileError,
    load_default_acoustic_profile_registry,
    resolve_acoustic_profile,
    select_acoustic_profile_for_room_source,
    validate_acoustic_profile_links,
    validate_acoustic_profile_registry,
)
from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m3.compiler import compile_custom_acoustic_scene


ROOT = Path(__file__).resolve().parents[2]
ROOM_REGISTRY_PATH = ROOT / "examples/m6/rooms/room_registry.json"
CUSTOM_ROOM_PATH = ROOT / "examples/m1/rooms/blender_custom/room_manifest.json"
CUSTOM_MATERIALS = ROOT / "examples/m3/blender_custom"


def _default_inputs() -> tuple[dict, dict]:
    return (
        load_default_acoustic_profile_registry(),
        load_json(ROOM_REGISTRY_PATH),
    )


def _binding(registry: dict, binding_id: str) -> dict:
    return next(
        binding
        for binding in registry["bindings"]
        if binding["binding_id"] == binding_id
    )


def _profile(registry: dict, profile_id: str) -> dict:
    return next(
        profile
        for profile in registry["profiles"]
        if profile["profile_id"] == profile_id
    )


def _rewrite_package_manifest(path: Path, manifest: dict) -> None:
    manifest["package_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "package_content_sha256"
        }
    )
    write_json(path, manifest)


def _valid_mp3d_package(
    tmp_path: Path,
    *,
    room_registry: dict,
    binding: dict,
    profile: dict,
) -> Path:
    package_path = compile_custom_acoustic_scene(
        room_manifest=CUSTOM_ROOM_PATH,
        material_mapping=CUSTOM_MATERIALS / "mapping.json",
        material_database=CUSTOM_MATERIALS / "materials_low.json",
        output=tmp_path / "mp3d-package",
    )
    room = next(
        record
        for record in room_registry["records"]
        if record["room_id"] == binding["room_ref"]["room_id"]
        and record["revision"] == binding["room_ref"]["revision"]
    )
    resources = {
        resource["resource_id"]: resource for resource in room["resources"]
    }
    manifest = load_json(package_path)
    manifest["source_room"] = {
        "room_id": binding["room_ref"]["room_id"],
        "manifest_sha256": resources["mp3d_room_manifest"]["sha256"],
        "source_revision": binding["room_ref"]["revision"],
        "geometry_asset_sha256": resources[binding["geometry_resource_id"]][
            "sha256"
        ],
    }
    manifest["materials"]["database_id"] = profile["profile_id"]
    manifest["materials"]["acoustic_profile_binding"] = {
        "schema": "avengine_m3_acoustic_profile_binding_v1",
        "profile_id": binding["profile_ref"]["profile_id"],
        "profile_revision": binding["profile_ref"]["revision"],
        "adapter_id": profile["material_binding"]["adapter_id"],
        "resources": [
            {
                "role": resource["role"],
                "sha256": resource["path"]["sha256"],
            }
            for resource in profile["material_binding"]["resources"]
        ],
    }
    _rewrite_package_manifest(package_path, manifest)
    return package_path


def test_default_registry_covers_three_origins_with_one_solver() -> None:
    registry, room_registry = _default_inputs()

    assert validate_acoustic_profile_registry(registry) == []
    assert validate_acoustic_profile_links(registry, room_registry) == []
    assert registry["solver_backend_id"] == SOLVER_BACKEND_ID
    assert {
        profile["origin"]["kind"] for profile in registry["profiles"]
    } == {
        "soundspaces2_public",
        "habitat_scene",
        "spear_ue_authored",
    }
    assert {
        profile["solver_backend_id"] for profile in registry["profiles"]
    } == {SOLVER_BACKEND_ID}
    for binding in registry["bindings"]:
        room = next(
            record
            for record in room_registry["records"]
            if record["room_id"] == binding["room_ref"]["room_id"]
            and record["revision"] == binding["room_ref"]["revision"]
        )
        selected = select_acoustic_profile_for_room_source(registry, room)
        assert selected["profile_id"] == binding["profile_ref"]["profile_id"]
        assert selected["revision"] == binding["profile_ref"]["revision"]


def test_source_selection_fails_closed_on_provider_ambiguity() -> None:
    registry, room_registry = _default_inputs()
    habitat_profile = _profile(
        registry,
        "replicacad_controlled_approximation_pending_v1",
    )
    habitat_profile["source_selection"]["provider_ids"].append("matterport3d")
    mp3d_room = next(
        record
        for record in room_registry["records"]
        if record["provider_id"] == "matterport3d"
    )

    with pytest.raises(AcousticProfileError, match="match_count=2"):
        select_acoustic_profile_for_room_source(registry, mp3d_room)
    errors = validate_acoustic_profile_links(registry, room_registry)
    assert any(
        "provider_id does not select exactly one acoustic profile" in error
        for error in errors
    )


def test_exact_mp3d_selection_keeps_production_and_reference_separate() -> None:
    registry, room_registry = _default_inputs()
    binding = _binding(registry, "mp3d_17DRP5sb8fy_soundspaces2_v1")

    selected = resolve_acoustic_profile(
        registry,
        room_registry,
        binding["room_ref"],
        repository_root=ROOT,
        environment={},
        verify_paths=False,
    )

    assert selected.profile["profile_id"] == (
        "soundspaces2_mp3d_public_materials_v1"
    )
    assert selected.binding is binding
    assert selected.representation["representation_id"] == (
        "mp3d_17DRP5sb8fy_soundspaces2_acoustic_v1"
    )
    assert selected.resource["resource_id"] == (
        "mp3d_soundspaces2_acoustic_package_v1"
    )
    assert selected.acoustic_package_manifest_path is None
    assert selected.simulation_request_path.name == (
        "rir_cache_simulation_request_v2.json"
    )
    assert selected.simulation_path("reference").name == (
        "rir_cache_simulation_request_soundspaces2_public_reference_v1.json"
    )
    receipt = selected.receipt("production")
    assert receipt["verification_status"] == "not_verified"
    assert receipt["paths"]["acoustic_package_manifest"]["resolved_path"] is None
    assert receipt["paths"]["selected_simulation_request"][
        "verification_status"
    ] == "not_verified"
    assert len(receipt["selection_content_sha256"]) == 64


def test_unverified_selection_does_not_read_acoustic_package(
    tmp_path: Path,
) -> None:
    registry, room_registry = _default_inputs()
    binding = _binding(registry, "mp3d_17DRP5sb8fy_soundspaces2_v1")
    package_root = tmp_path / "invalid-package"
    package_root.mkdir()
    package_path = package_root / "manifest.json"
    package_path.write_text("not JSON\n", encoding="utf-8")

    selected = resolve_acoustic_profile(
        registry,
        room_registry,
        binding["room_ref"],
        repository_root=ROOT,
        environment={
            "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT": str(package_root),
        },
        verify_paths=False,
    )

    assert selected.acoustic_package_manifest_path == package_path.resolve()
    assert selected.verification_status == "not_verified"


def test_exact_selection_fails_when_room_lineage_changes() -> None:
    registry, room_registry = _default_inputs()
    binding = _binding(registry, "mp3d_17DRP5sb8fy_soundspaces2_v1")
    room = next(
        record
        for record in room_registry["records"]
        if record["room_id"] == binding["room_ref"]["room_id"]
    )
    room["lineage"]["acoustic_profile_id"] = "different_material_lineage_v1"

    with pytest.raises(
        AcousticProfileError,
        match="lineage.acoustic_profile_id",
    ):
        resolve_acoustic_profile(
            registry,
            room_registry,
            binding["room_ref"],
            repository_root=ROOT,
            verify_paths=False,
        )


def test_verified_selection_checks_files_hash_and_package_room(
    tmp_path: Path,
) -> None:
    registry, room_registry = _default_inputs()
    binding = _binding(registry, "mp3d_17DRP5sb8fy_soundspaces2_v1")
    profile = _profile(registry, "soundspaces2_mp3d_public_materials_v1")

    soundspaces_root = tmp_path / "sound-spaces"
    material_path = soundspaces_root / "data/mp3d_material_config.json"
    material_path.parent.mkdir(parents=True)
    write_json(material_path, {"materials": [{"name": "fixture"}]})
    profile["material_binding"]["resources"][0]["path"]["sha256"] = sha256_file(
        material_path
    )

    package_root = tmp_path / "mp3d-package"
    package_path = _valid_mp3d_package(
        tmp_path,
        room_registry=room_registry,
        binding=binding,
        profile=profile,
    )
    environment = {
        "AVENGINE_SOUNDSPACES_ROOT": str(soundspaces_root),
        "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT": str(package_root),
    }

    selected = resolve_acoustic_profile(
        registry,
        room_registry,
        binding["room_ref"],
        repository_root=ROOT,
        environment=environment,
        verify_paths=True,
    )

    assert selected.verification_status == "verified"
    assert selected.acoustic_package_manifest_path == package_path.resolve()
    assert selected.material_paths["soundspaces2_public_material_config"] == (
        material_path.resolve()
    )
    receipt = selected.receipt()
    assert receipt["paths"]["acoustic_package_manifest"]["sha256"] == (
        sha256_file(package_path)
    )
    assert receipt["paths"]["simulation_production"]["exists"] is True

    manifest = load_json(package_path)
    manifest["source_room"]["room_id"] = "wrong_visual_room"
    _rewrite_package_manifest(package_path, manifest)
    with pytest.raises(
        AcousticProfileError,
        match="source_room.room_id does not match",
    ):
        resolve_acoustic_profile(
            registry,
            room_registry,
            binding["room_ref"],
            repository_root=ROOT,
            environment=environment,
            verify_paths=True,
        )


def test_verified_selection_rejects_package_geometry_outside_representation_inputs(
    tmp_path: Path,
) -> None:
    registry, room_registry = _default_inputs()
    binding = _binding(registry, "mp3d_17DRP5sb8fy_soundspaces2_v1")
    profile = _profile(registry, "soundspaces2_mp3d_public_materials_v1")
    soundspaces_root = tmp_path / "sound-spaces"
    material_path = soundspaces_root / "data/mp3d_material_config.json"
    material_path.parent.mkdir(parents=True)
    write_json(material_path, {"materials": [{"name": "fixture"}]})
    profile["material_binding"]["resources"][0]["path"]["sha256"] = sha256_file(
        material_path
    )
    package_path = _valid_mp3d_package(
        tmp_path,
        room_registry=room_registry,
        binding=binding,
        profile=profile,
    )
    manifest = load_json(package_path)
    manifest["source_room"]["geometry_asset_sha256"] = "f" * 64
    _rewrite_package_manifest(package_path, manifest)

    with pytest.raises(
        AcousticProfileError,
        match="geometry_asset_sha256 does not match.*exact geometry_resource_id",
    ):
        resolve_acoustic_profile(
            registry,
            room_registry,
            binding["room_ref"],
            repository_root=ROOT,
            environment={
                "AVENGINE_SOUNDSPACES_ROOT": str(soundspaces_root),
                "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT": str(
                    package_path.parent
                ),
            },
            verify_paths=True,
        )


def test_verified_selection_rejects_other_representation_input_as_geometry(
    tmp_path: Path,
) -> None:
    registry, room_registry = _default_inputs()
    binding = _binding(registry, "mp3d_17DRP5sb8fy_soundspaces2_v1")
    profile = _profile(registry, "soundspaces2_mp3d_public_materials_v1")
    soundspaces_root = tmp_path / "sound-spaces"
    material_path = soundspaces_root / "data/mp3d_material_config.json"
    material_path.parent.mkdir(parents=True)
    write_json(material_path, {"materials": [{"name": "fixture"}]})
    profile["material_binding"]["resources"][0]["path"]["sha256"] = sha256_file(
        material_path
    )
    package_path = _valid_mp3d_package(
        tmp_path,
        room_registry=room_registry,
        binding=binding,
        profile=profile,
    )
    room = next(
        item
        for item in room_registry["records"]
        if item["room_id"] == binding["room_ref"]["room_id"]
        and item["revision"] == binding["room_ref"]["revision"]
    )
    resources = {
        resource["resource_id"]: resource for resource in room["resources"]
    }
    assert (
        "mp3d_house_descriptor"
        in next(
            item
            for item in room["acoustic_representations"]
            if item["representation_id"]
            == binding["acoustic_representation_id"]
        )["input_resource_ids"]
    )
    manifest = load_json(package_path)
    manifest["source_room"]["geometry_asset_sha256"] = resources[
        "mp3d_house_descriptor"
    ]["sha256"]
    _rewrite_package_manifest(package_path, manifest)

    with pytest.raises(
        AcousticProfileError,
        match="geometry_asset_sha256 does not match.*exact geometry_resource_id",
    ):
        resolve_acoustic_profile(
            registry,
            room_registry,
            binding["room_ref"],
            repository_root=ROOT,
            environment={
                "AVENGINE_SOUNDSPACES_ROOT": str(soundspaces_root),
                "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT": str(
                    package_path.parent
                ),
            },
            verify_paths=True,
        )


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        (
            "adapter",
            "material adapter does not match selected profile",
        ),
        (
            "material_resource",
            "material resource identities do not match the selected profile",
        ),
        (
            "profile_revision",
            "material profile identity does not match binding.profile_ref",
        ),
    ],
)
def test_verified_selection_rejects_wrong_package_profile_binding(
    tmp_path: Path,
    tamper: str,
    expected_error: str,
) -> None:
    registry, room_registry = _default_inputs()
    binding = _binding(registry, "mp3d_17DRP5sb8fy_soundspaces2_v1")
    profile = _profile(registry, "soundspaces2_mp3d_public_materials_v1")
    soundspaces_root = tmp_path / "sound-spaces"
    material_path = soundspaces_root / "data/mp3d_material_config.json"
    material_path.parent.mkdir(parents=True)
    write_json(material_path, {"materials": [{"name": "fixture"}]})
    profile["material_binding"]["resources"][0]["path"]["sha256"] = sha256_file(
        material_path
    )
    package_path = _valid_mp3d_package(
        tmp_path,
        room_registry=room_registry,
        binding=binding,
        profile=profile,
    )
    manifest = load_json(package_path)
    package_binding = manifest["materials"]["acoustic_profile_binding"]
    if tamper == "adapter":
        package_binding["adapter_id"] = "different_adapter_v1"
    elif tamper == "material_resource":
        package_binding["resources"][0]["sha256"] = "f" * 64
    else:
        package_binding["profile_revision"] = "different_revision_v1"
    _rewrite_package_manifest(package_path, manifest)

    with pytest.raises(AcousticProfileError, match=expected_error):
        resolve_acoustic_profile(
            registry,
            room_registry,
            binding["room_ref"],
            repository_root=ROOT,
            environment={
                "AVENGINE_SOUNDSPACES_ROOT": str(soundspaces_root),
                "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT": str(
                    package_path.parent
                ),
            },
            verify_paths=True,
        )


def test_verified_selection_rejects_package_content_hash_mismatch(
    tmp_path: Path,
) -> None:
    registry, room_registry = _default_inputs()
    binding = _binding(registry, "mp3d_17DRP5sb8fy_soundspaces2_v1")
    profile = _profile(registry, "soundspaces2_mp3d_public_materials_v1")
    soundspaces_root = tmp_path / "sound-spaces"
    material_path = soundspaces_root / "data/mp3d_material_config.json"
    material_path.parent.mkdir(parents=True)
    write_json(material_path, {"materials": [{"name": "fixture"}]})
    profile["material_binding"]["resources"][0]["path"]["sha256"] = sha256_file(
        material_path
    )
    package_path = _valid_mp3d_package(
        tmp_path,
        room_registry=room_registry,
        binding=binding,
        profile=profile,
    )
    manifest = load_json(package_path)
    manifest["package_id"] = "tampered_after_hash"
    write_json(package_path, manifest)

    with pytest.raises(
        AcousticProfileError,
        match="package_content_sha256 does not match canonical manifest content",
    ):
        resolve_acoustic_profile(
            registry,
            room_registry,
            binding["room_ref"],
            repository_root=ROOT,
            environment={
                "AVENGINE_SOUNDSPACES_ROOT": str(soundspaces_root),
                "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT": str(
                    package_path.parent
                ),
            },
            verify_paths=True,
        )


def test_registry_rejects_duplicate_binding_and_profile_lineage_mismatch() -> None:
    registry, _ = _default_inputs()
    duplicate = deepcopy(registry["bindings"][0])
    registry["bindings"].append(duplicate)
    errors = validate_acoustic_profile_registry(registry)
    assert any("duplicate binding_id" in error for error in errors)
    assert any("duplicate exact room/lineage binding" in error for error in errors)

    registry, _ = _default_inputs()
    registry["bindings"][0]["profile_ref"]["profile_id"] = (
        "replicacad_controlled_approximation_pending_v1"
    )
    errors = validate_acoustic_profile_registry(registry)
    assert any(
        "profile_id must equal the room lineage acoustic_profile_id" in error
        for error in errors
    )


def test_reference_mode_fails_closed_when_profile_has_no_reference() -> None:
    registry, room_registry = _default_inputs()
    binding = _binding(registry, "replicacad_apt_0_habitat_scene_v1")
    selected = resolve_acoustic_profile(
        registry,
        room_registry,
        binding["room_ref"],
        repository_root=ROOT,
        environment={},
        verify_paths=False,
    )

    with pytest.raises(AcousticProfileError, match="has no reference"):
        selected.simulation_path("reference")
