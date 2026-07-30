from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m6 import rooms
from tools.m6x import render_rir_cache as rir_tool
from tools.m7 import run_habitat_room_batch as habitat_batch


def _arguments(**overrides):
    values = {
        "acoustic_package_manifest": None,
        "simulation_request": None,
        "room_id": None,
        "room_revision": None,
        "room_registry": Path("room_registry.json"),
        "acoustic_profile_registry": None,
        "simulation_profile": "production",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_fake_acoustic_profiles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selection: SimpleNamespace,
    calls: list[tuple],
) -> None:
    module = ModuleType("avengine.acoustic_profiles")

    def load_default():
        return {"registry_id": "default"}

    def load_explicit(path):
        calls.append(("load_explicit", Path(path)))
        return {"registry_id": "explicit"}

    def resolve(
        acoustic_registry,
        room_registry,
        room_ref,
        *,
        repository_root,
        verify_paths,
    ):
        calls.append(
            (
                "resolve",
                acoustic_registry,
                room_registry,
                room_ref,
                Path(repository_root),
                verify_paths,
            )
        )
        return selection

    module.load_default_acoustic_profile_registry = load_default
    module.load_acoustic_profile_registry = load_explicit
    module.resolve_acoustic_profile = resolve
    monkeypatch.setitem(sys.modules, "avengine.acoustic_profiles", module)


def test_rir_tool_preserves_explicit_legacy_cli_paths_and_request_default(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package.json"
    simulation = tmp_path / "simulation.json"
    write_json(package, {"schema": "fixture"})
    write_json(simulation, {"simulation": {}})

    parsed = rir_tool.parse_args(
        [
            "--rir-job-plan",
            str(tmp_path / "plan.json"),
            "--acoustic-package-manifest",
            str(package),
            "--simulation-request",
            str(simulation),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    resolved = rir_tool.resolve_effective_acoustic_inputs(parsed)

    assert resolved.selection_mode == "explicit"
    assert resolved.acoustic_package_manifest == package.resolve()
    assert resolved.simulation_request == simulation.resolve()
    assert resolved.profile_selection_receipt is None
    receipt = resolved.receipt()
    assert receipt["selection_mode"] == "explicit"
    assert receipt["registry_resolution"] is None
    assert receipt["registry_selection_applied_to_effective_inputs"] == {
        "acoustic_package_manifest": False,
        "simulation_request": False,
    }

    legacy_default = rir_tool.resolve_effective_acoustic_inputs(
        _arguments(acoustic_package_manifest=package)
    )
    assert (
        legacy_default.simulation_request
        == rir_tool.LEGACY_SIMULATION_REQUEST.resolve()
    )


def test_rir_tool_selects_reference_request_and_allows_equivalent_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_package = tmp_path / "selected_manifest.json"
    override_package = tmp_path / "override_manifest.json"
    production = tmp_path / "production.json"
    reference = tmp_path / "reference.json"
    for path in (selected_package, production, reference):
        write_json(path, {"path": path.name})
    override_package.write_bytes(selected_package.read_bytes())
    selection_receipt = {"schema": "fixture_selection", "profile_id": "ss2"}
    receipt_modes: list[str] = []
    selection = SimpleNamespace(
        acoustic_package_manifest_path=selected_package,
        simulation_request_path=production,
        reference_simulation_request_path=reference,
        simulation_path=lambda mode="production": (
            production if mode == "production" else reference
        ),
        receipt=lambda mode="production": (
            receipt_modes.append(mode) or selection_receipt
        ),
    )
    calls: list[tuple] = []
    _install_fake_acoustic_profiles(
        monkeypatch,
        selection=selection,
        calls=calls,
    )
    monkeypatch.setattr(
        rooms,
        "load_room_registry",
        lambda _path: {
            "registry_id": "rooms",
            "records": [],
        },
    )

    resolved = rir_tool.resolve_effective_acoustic_inputs(
        _arguments(
            acoustic_package_manifest=override_package,
            room_id="room",
            room_revision="rev",
            acoustic_profile_registry=tmp_path / "profiles.json",
            simulation_profile="reference",
        )
    )

    assert (
        resolved.selection_mode
        == "registry_with_verified_equivalent_overrides"
    )
    assert resolved.acoustic_package_manifest == override_package.resolve()
    assert resolved.simulation_request == reference.resolve()
    assert resolved.profile_selection_receipt == selection_receipt
    receipt = resolved.receipt()
    assert receipt["registry_resolution"] == selection_receipt
    assert receipt["registry_selection_applied_to_effective_inputs"] == {
        "acoustic_package_manifest": False,
        "simulation_request": True,
    }
    assert receipt["effective_inputs"]["acoustic_package_manifest"]["path"] == str(
        override_package.resolve()
    )
    receipt_identity = receipt.pop("effective_selection_content_sha256")
    assert receipt_identity == canonical_json_sha256(receipt)
    assert calls[0] == ("load_explicit", tmp_path / "profiles.json")
    assert calls[1][3] == {
        "registry_id": "rooms",
        "room_id": "room",
        "revision": "rev",
    }
    assert calls[1][-1] is True
    assert receipt_modes == ["reference"]


@pytest.mark.parametrize(
    ("argument_name", "option_name"),
    (
        ("acoustic_package_manifest", "--acoustic-package-manifest"),
        ("simulation_request", "--simulation-request"),
    ),
)
def test_rir_tool_registry_selection_rejects_non_equivalent_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument_name: str,
    option_name: str,
) -> None:
    selected_package = tmp_path / "selected_manifest.json"
    arbitrary_override = tmp_path / "arbitrary_override.json"
    production = tmp_path / "production.json"
    for path, marker in (
        (selected_package, "selected"),
        (arbitrary_override, "arbitrary"),
        (production, "production"),
    ):
        write_json(path, {"marker": marker})
    selection = SimpleNamespace(
        acoustic_package_manifest_path=selected_package,
        simulation_request_path=production,
        reference_simulation_request_path=None,
        simulation_path=lambda _mode="production": production,
        receipt=lambda _mode="production": {"schema": "fixture_selection"},
    )
    _install_fake_acoustic_profiles(
        monkeypatch,
        selection=selection,
        calls=[],
    )
    monkeypatch.setattr(
        rooms,
        "load_room_registry",
        lambda _path: {"registry_id": "rooms", "records": []},
    )

    with pytest.raises(
        ValueError,
        match=(
            f"{option_name} override SHA-256 differs from the "
            "registry-selected physical file"
        ),
    ):
        rir_tool.resolve_effective_acoustic_inputs(
            _arguments(
                room_id="room",
                room_revision="rev",
                **{argument_name: arbitrary_override},
            )
        )


def test_rir_tool_automatically_applies_selected_package_and_production_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "manifest.json"
    production = tmp_path / "production.json"
    reference = tmp_path / "reference.json"
    for path in (package, production, reference):
        write_json(path, {"path": path.name})
    selection_receipt = {"schema": "fixture_selection", "profile_id": "ss2"}
    selection = SimpleNamespace(
        acoustic_package_manifest_path=package,
        simulation_request_path=production,
        reference_simulation_request_path=reference,
        simulation_path=lambda mode="production": (
            production if mode == "production" else reference
        ),
        receipt=lambda _mode="production": selection_receipt,
    )
    calls: list[tuple] = []
    _install_fake_acoustic_profiles(
        monkeypatch,
        selection=selection,
        calls=calls,
    )
    monkeypatch.setattr(
        rooms,
        "load_room_registry",
        lambda _path: {"registry_id": "rooms", "records": []},
    )

    resolved = rir_tool.resolve_effective_acoustic_inputs(
        _arguments(room_id="room", room_revision="rev")
    )

    assert resolved.selection_mode == "registry"
    assert resolved.acoustic_package_manifest == package.resolve()
    assert resolved.simulation_request == production.resolve()
    assert resolved.receipt()["registry_selection_applied_to_effective_inputs"] == {
        "acoustic_package_manifest": True,
        "simulation_request": True,
    }


def test_rir_tool_run_passes_registry_selected_inputs_to_existing_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "manifest.json"
    write_json(package, {"schema": "fixture"})
    production = (
        rir_tool.REPOSITORY
        / "examples/runtime/rir_cache_simulation_request_v2.json"
    )
    selection = SimpleNamespace(
        acoustic_package_manifest_path=package,
        simulation_request_path=production,
        reference_simulation_request_path=None,
        simulation_path=lambda _mode="production": production,
        receipt=lambda _mode="production": {"schema": "fixture_selection"},
    )
    _install_fake_acoustic_profiles(
        monkeypatch,
        selection=selection,
        calls=[],
    )
    monkeypatch.setattr(
        rooms,
        "load_room_registry",
        lambda _path: {"registry_id": "rooms", "records": []},
    )
    loaded: dict[str, object] = {}
    scene = object()

    def load_scene(path, *, allow_nonpassing_research_qa):
        loaded["scene_path"] = path
        loaded["allow_nonpassing_research_qa"] = allow_nonpassing_research_qa
        return scene

    def render(**kwargs):
        loaded["render"] = kwargs
        Path(kwargs["output"]).mkdir()
        return SimpleNamespace(
            output=Path(kwargs["output"]).resolve(),
            receipt={
                "selected_job_count": 3,
                "full_plan_complete": True,
            },
        )

    monkeypatch.setattr(rir_tool, "load_compiled_acoustic_scene", load_scene)
    monkeypatch.setattr(rir_tool, "render_rir_cache", render)
    output = tmp_path / "cache"
    args = rir_tool.parse_args(
        [
            "--rir-job-plan",
            str(tmp_path / "plan.json"),
            "--room-id",
            "room",
            "--room-revision",
            "rev",
            "--output",
            str(output),
        ]
    )

    result = rir_tool.run(args)

    assert result == output.resolve()
    assert loaded["scene_path"] == package.resolve()
    assert loaded["allow_nonpassing_research_qa"] is True
    assert loaded["render"]["scene"] is scene
    assert loaded["render"]["simulation_request_path"] == production.resolve()
    forwarded = loaded["render"]["acoustic_selection_receipt"]
    assert forwarded["selection_mode"] == "registry"
    assert forwarded["effective_inputs"]["acoustic_package_manifest"]["path"] == str(
        package.resolve()
    )


def test_m7_uses_room_profile_ref_for_the_same_acoustic_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "manifest.json"
    production = tmp_path / "production.json"
    reference = tmp_path / "reference.json"
    room_registry_path = tmp_path / "rooms.json"
    for path in (package, production, reference, room_registry_path):
        write_json(path, {"path": path.name})
    selection_receipt = {
        "schema": "fixture_selection",
        "profile_id": "soundspaces2_mp3d",
    }
    receipt_modes: list[str] = []
    selection = SimpleNamespace(
        acoustic_package_manifest_path=package,
        simulation_request_path=production,
        reference_simulation_request_path=reference,
        simulation_path=lambda mode="production": (
            production if mode == "production" else reference
        ),
        receipt=lambda mode="production": (
            receipt_modes.append(mode) or selection_receipt
        ),
    )
    calls: list[tuple] = []
    _install_fake_acoustic_profiles(
        monkeypatch,
        selection=selection,
        calls=calls,
    )
    monkeypatch.setattr(
        rooms,
        "load_room_registry",
        lambda _path: {
            "registry_id": "rooms",
            "records": [],
        },
    )
    room_ref = {
        "registry_id": "rooms",
        "room_id": "room",
        "revision": "rev",
    }

    binding = habitat_batch._resolve_acoustic_selection_binding(
        room_profile={"room_ref": room_ref},
        room_registry_path=room_registry_path,
        acoustic_profile_registry_path=None,
        simulation_profile="reference",
    )

    assert binding["room_ref"] == room_ref
    assert binding["simulation_profile"] == "reference"
    assert binding["selected_simulation_request"]["path"] == str(reference.resolve())
    assert binding["acoustic_profile_selection"] == selection_receipt
    assert calls[0][3] == room_ref
    assert calls[0][-1] is False
    assert receipt_modes == ["reference"]


def test_m7_real_default_mp3d_profile_binds_without_requiring_audio_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AVENGINE_SOUNDSPACES_ROOT", raising=False)
    monkeypatch.delenv(
        "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT",
        raising=False,
    )
    runtime_profiles = load_json(
        habitat_batch.REPOSITORY
        / "examples/runtime/room_runtime_profiles.json"
    )
    profile = next(
        item
        for item in runtime_profiles["profiles"]
        if item["profile_id"] == "habitat_mp3d_17DRP5sb8fy"
    )

    binding = habitat_batch._resolve_acoustic_selection_binding(
        room_profile=profile,
        room_registry_path=habitat_batch.DEFAULT_ROOM_REGISTRY,
        acoustic_profile_registry_path=None,
        simulation_profile="production",
    )

    selection = binding["acoustic_profile_selection"]
    assert selection["verification_status"] == "not_verified"
    assert selection["room_ref"] == profile["room_ref"]
    assert selection["profile_ref"]["profile_id"] == (
        "soundspaces2_mp3d_public_materials_v1"
    )
    assert selection["paths"]["acoustic_package_manifest"]["resolved_path"] is None
    assert binding["selected_simulation_request"]["sha256"]


def test_m7_room_manifest_binding_verifies_exact_registry_room() -> None:
    runtime_profiles = load_json(
        habitat_batch.REPOSITORY
        / "examples/runtime/room_runtime_profiles.json"
    )
    profile = next(
        item
        for item in runtime_profiles["profiles"]
        if item["profile_id"] == "habitat_mp3d_17DRP5sb8fy"
    )
    room_manifest = (
        habitat_batch.REPOSITORY
        / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
    )

    binding = habitat_batch._verify_room_manifest_binding(
        room_profile=profile,
        room_registry_path=habitat_batch.DEFAULT_ROOM_REGISTRY,
        room_manifest_path=room_manifest,
    )

    assert binding["status"] == "pass"
    assert binding["room_ref"] == profile["room_ref"]
    assert binding["room_manifest"]["path"] == str(room_manifest.resolve())
    assert binding["room_manifest"]["sha256"] == sha256_file(room_manifest)
    assert binding["checks"]["manifest_contract"] == "pass"
    assert binding["checks"]["room_id_matches_room_ref"] == "pass"
    hash_check = binding["checks"]["registry_declared_hash"]
    assert hash_check["status"] == "pass"
    assert hash_check["declared_sha256"] == binding["room_manifest"]["sha256"]


def test_m7_room_manifest_binding_rejects_wrong_room_id(tmp_path: Path) -> None:
    runtime_profiles = load_json(
        habitat_batch.REPOSITORY
        / "examples/runtime/room_runtime_profiles.json"
    )
    profile = next(
        item
        for item in runtime_profiles["profiles"]
        if item["profile_id"] == "habitat_mp3d_17DRP5sb8fy"
    )
    manifest = load_json(
        habitat_batch.REPOSITORY
        / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
    )
    manifest["room_id"] = "wrong_visual_room"
    room_manifest = tmp_path / "wrong_room.json"
    write_json(room_manifest, manifest)

    with pytest.raises(
        habitat_batch.HabitatRoomBatchError,
        match="room_id does not match",
    ):
        habitat_batch._verify_room_manifest_binding(
            room_profile=profile,
            room_registry_path=habitat_batch.DEFAULT_ROOM_REGISTRY,
            room_manifest_path=room_manifest,
        )


def test_m7_room_manifest_binding_rejects_wrong_declared_hash(
    tmp_path: Path,
) -> None:
    runtime_profiles = load_json(
        habitat_batch.REPOSITORY
        / "examples/runtime/room_runtime_profiles.json"
    )
    profile = next(
        item
        for item in runtime_profiles["profiles"]
        if item["profile_id"] == "habitat_mp3d_17DRP5sb8fy"
    )
    manifest = load_json(
        habitat_batch.REPOSITORY
        / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
    )
    room_manifest = tmp_path / "same_room_different_bytes.json"
    write_json(room_manifest, manifest)

    with pytest.raises(
        habitat_batch.HabitatRoomBatchError,
        match="sha256 does not match",
    ):
        habitat_batch._verify_room_manifest_binding(
            room_profile=profile,
            room_registry_path=habitat_batch.DEFAULT_ROOM_REGISTRY,
            room_manifest_path=room_manifest,
        )


def test_m7_room_manifest_binding_rejects_ambiguous_registry_resource(
    tmp_path: Path,
) -> None:
    runtime_profiles = load_json(
        habitat_batch.REPOSITORY
        / "examples/runtime/room_runtime_profiles.json"
    )
    profile = next(
        item
        for item in runtime_profiles["profiles"]
        if item["profile_id"] == "habitat_mp3d_17DRP5sb8fy"
    )
    registry = load_json(habitat_batch.DEFAULT_ROOM_REGISTRY)
    record = next(
        item
        for item in registry["records"]
        if item["room_id"] == profile["room_ref"]["room_id"]
        and item["revision"] == profile["room_ref"]["revision"]
    )
    declared = next(
        item
        for item in record["resources"]
        if item["resource_type"] == "room_manifest"
    )
    record["resources"].append(
        {
            **declared,
            "resource_id": "ambiguous_second_room_manifest",
        }
    )
    room_registry = tmp_path / "ambiguous_rooms.json"
    write_json(room_registry, registry)

    with pytest.raises(
        habitat_batch.HabitatRoomBatchError,
        match="exactly one sha256-bound room_manifest resource",
    ):
        habitat_batch._verify_room_manifest_binding(
            room_profile=profile,
            room_registry_path=room_registry,
            room_manifest_path=(
                habitat_batch.REPOSITORY
                / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
            ),
        )


def test_m7_room_manifest_binding_rejects_missing_registry_resource(
    tmp_path: Path,
) -> None:
    runtime_profiles = load_json(
        habitat_batch.REPOSITORY
        / "examples/runtime/room_runtime_profiles.json"
    )
    profile = next(
        item
        for item in runtime_profiles["profiles"]
        if item["profile_id"] == "habitat_mp3d_17DRP5sb8fy"
    )
    registry = load_json(habitat_batch.DEFAULT_ROOM_REGISTRY)
    record = next(
        item
        for item in registry["records"]
        if item["room_id"] == profile["room_ref"]["room_id"]
        and item["revision"] == profile["room_ref"]["revision"]
    )
    record["resources"] = [
        item
        for item in record["resources"]
        if item["resource_type"] != "room_manifest"
    ]
    room_registry = tmp_path / "missing_room_manifest_resource.json"
    write_json(room_registry, registry)

    with pytest.raises(
        habitat_batch.HabitatRoomBatchError,
        match="exactly one sha256-bound room_manifest resource",
    ):
        habitat_batch._verify_room_manifest_binding(
            room_profile=profile,
            room_registry_path=room_registry,
            room_manifest_path=(
                habitat_batch.REPOSITORY
                / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
            ),
        )


def test_m7_room_manifest_binding_rejects_invalid_manifest(
    tmp_path: Path,
) -> None:
    runtime_profiles = load_json(
        habitat_batch.REPOSITORY
        / "examples/runtime/room_runtime_profiles.json"
    )
    profile = next(
        item
        for item in runtime_profiles["profiles"]
        if item["profile_id"] == "habitat_mp3d_17DRP5sb8fy"
    )
    room_manifest = tmp_path / "invalid_room.json"
    write_json(room_manifest, {"room_id": profile["room_ref"]["room_id"]})

    with pytest.raises(
        habitat_batch.HabitatRoomBatchError,
        match="room manifest contract failed",
    ):
        habitat_batch._verify_room_manifest_binding(
            room_profile=profile,
            room_registry_path=habitat_batch.DEFAULT_ROOM_REGISTRY,
            room_manifest_path=room_manifest,
        )


def test_m7_batch_manifest_binds_resolved_acoustic_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_registry = tmp_path / "runtime_registry.json"
    route = tmp_path / "route.json"
    write_json(runtime_registry, {"schema": "fixture"})
    write_json(route, {"schema": "fixture"})
    output = tmp_path / "batch"
    evidence = output / "episodes/episode0/mp3d_gate_evidence.json"
    write_json(
        evidence,
        {
            "status": "pass",
            "frame_count": 75,
            "frame_rate_hz": 15,
            "route_id": "route0",
            "gates": [{"gate_id": "readback", "status": "pass"}],
        },
    )
    profile = {
        "profile_id": "habitat_room",
        "revision": "v1",
        "backend_id": "habitat_native",
        "room_ref": {
            "registry_id": "rooms",
            "room_id": "room",
            "revision": "rev",
        },
        "render": {"frame_count": 75},
    }
    acoustic_binding = {
        "schema": "avengine_m7_acoustic_selection_binding_v1",
        "simulation_profile": "production",
    }
    room_manifest_binding = {
        "schema": "avengine_m7_room_manifest_binding_v1",
        "status": "pass",
        "room_ref": profile["room_ref"],
    }
    monkeypatch.setattr(
        habitat_batch,
        "_select_profile",
        lambda _path, _profile_id: profile,
    )
    monkeypatch.setattr(
        habitat_batch,
        "_verify_room_manifest_binding",
        lambda **_kwargs: room_manifest_binding,
    )
    monkeypatch.setattr(
        habitat_batch,
        "_resolve_acoustic_selection_binding",
        lambda **_kwargs: acoustic_binding,
    )
    capture_module = ModuleType("avengine.m5_1.mp3d_capture")
    capture_module.capture_mp3d_route = lambda **_kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "avengine.m5_1.mp3d_capture",
        capture_module,
    )

    result = habitat_batch.main(
        [
            "--room-runtime-registry",
            str(runtime_registry),
            "--room-profile",
            "habitat_room",
            "--episode",
            f"episode0={route}",
            "--room-manifest",
            str(tmp_path / "room.json"),
            "--m1-request",
            str(tmp_path / "m1.json"),
            "--human-runtime-glb",
            str(tmp_path / "human.glb"),
            "--beagle-manifest",
            str(tmp_path / "beagle.json"),
            "--beagle-m2-request",
            str(tmp_path / "beagle_m2.json"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    manifest = load_json(output / "batch_manifest.json")
    assert manifest["room_manifest_binding"] == room_manifest_binding
    assert manifest["acoustic_selection"] == acoustic_binding
