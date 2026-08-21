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


def test_rir_tool_semantic_mode_uses_only_explicit_structural_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan.json"
    package = tmp_path / "manifest.json"
    simulation = tmp_path / "simulation.json"
    hrtf = tmp_path / "fixture.sofa"
    write_json(plan, {"schema": "fixture_plan"})
    write_json(package, {"schema": "fixture_package"})
    write_json(
        simulation,
        load_json(
            rir_tool.REPOSITORY
            / "examples/runtime/rir_cache_simulation_request_v2.json"
        ),
    )
    hrtf.write_bytes(b"fixture")
    scene = object()
    calls: dict[str, object] = {}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("semantic CLI entered a legacy file-evidence path")

    def load_scene(path):
        calls["scene_path"] = path
        return scene

    def render(**kwargs):
        calls["render"] = kwargs
        return SimpleNamespace(
            output=Path(kwargs["output"]),
            receipt={"selected_job_count": 2, "full_plan_complete": True},
        )

    for name in (
        "resolve_effective_acoustic_inputs",
        "sha256_file",
        "canonical_json_sha256",
        "load_compiled_acoustic_scene",
        "render_rir_cache",
    ):
        monkeypatch.setattr(rir_tool, name, forbidden)
    monkeypatch.setattr(rir_tool, "load_semantic_acoustic_scene", load_scene)
    monkeypatch.setattr(rir_tool, "render_semantic_rir_cache", render)
    output = tmp_path / "cache"
    args = rir_tool.parse_args(
        [
            "--semantic-no-file-evidence",
            "--rir-job-plan",
            str(plan),
            "--acoustic-package-manifest",
            str(package),
            "--simulation-request",
            str(simulation),
            "--hrtf",
            str(hrtf),
            "--output",
            str(output),
        ]
    )

    assert rir_tool.run(args) == output
    assert calls["scene_path"] == package.resolve()
    render_call = calls["render"]
    assert isinstance(render_call, dict)
    assert render_call["scene"] is scene
    assert render_call["plan_path"] == plan.resolve()
    assert render_call["simulation_request_path"] == simulation.resolve()
    assert render_call["acoustic_selection"] == {
        "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
        "selection_mode": "explicit_legacy_unbound",
        "registry_selection_applied": False,
        "room_ref": None,
        "profile_ref": None,
        "binding_id": None,
    }


@pytest.mark.parametrize("failure", ["registry", "profile", "partial", "symlink"])
def test_rir_tool_semantic_mode_rejects_mixed_or_aliased_inputs(
    tmp_path: Path, failure: str
) -> None:
    plan = tmp_path / "plan.json"
    package = tmp_path / "manifest.json"
    simulation = tmp_path / "simulation.json"
    hrtf = tmp_path / "fixture.sofa"
    for path in (plan, package, simulation):
        write_json(path, {"simulation": {}} if path == simulation else {})
    hrtf.write_bytes(b"fixture")
    arguments = [
        "--semantic-no-file-evidence",
        "--rir-job-plan",
        str(plan),
        "--acoustic-package-manifest",
        str(package),
        "--simulation-request",
        str(simulation),
        "--hrtf",
        str(hrtf),
        "--output",
        str(tmp_path / "cache"),
    ]
    if failure == "registry":
        arguments.extend(["--room-id", "room", "--room-revision", "v1"])
    elif failure == "profile":
        arguments.extend(["--simulation-profile", "reference"])
    elif failure == "partial":
        arguments.extend(["--job-limit", "1"])
    else:
        target = tmp_path / "real_manifest.json"
        package.rename(target)
        package.symlink_to(target)
    with pytest.raises(ValueError):
        rir_tool.run(rir_tool.parse_args(arguments))


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


def test_m7_current_room_manifest_and_mp3d_templates_match_registry() -> None:
    runtime_profiles = load_json(
        habitat_batch.REPOSITORY / "examples/runtime/room_runtime_profiles.json"
    )
    profile = next(
        item
        for item in runtime_profiles["profiles"]
        if item["profile_id"] == "habitat_mp3d_17DRP5sb8fy"
    )
    binding = habitat_batch._verify_room_manifest_binding(
        room_profile=profile,
        room_registry_path=habitat_batch.DEFAULT_ROOM_REGISTRY,
        room_manifest_path=(
            habitat_batch.REPOSITORY
            / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
        ),
    )
    assert binding["status"] == "pass"

    registry = load_json(habitat_batch.DEFAULT_ROOM_REGISTRY)
    record = next(
        item
        for item in registry["records"]
        if item["room_id"] == "habitat_mp3d_example_17DRP5sb8fy"
    )
    expected_resource_ids = {
        "mp3d_dataset_config",
        "mp3d_raw_visual_surface",
        "mp3d_navmesh",
        "mp3d_house_descriptor",
        "mp3d_semantic_mesh",
    }
    external_mp3d = [
        resource
        for resource in record["resources"]
        if resource["resource_id"] in expected_resource_ids
    ]
    assert len(external_mp3d) == 5
    assert all(
        resource["location"]["environment_variable"] == "AVENGINE_MP3D_ROOT"
        and resource["location"]["path_template"].startswith(
            "${AVENGINE_MP3D_ROOT}/scene_datasets/"
        )
        for resource in external_mp3d
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
    habitat_module = ModuleType("avengine.m1.habitat_capture")

    def fail_if_resume_prepares_runtime(**_kwargs):
        raise AssertionError("resume-only batch must not prepare native runtime")

    habitat_module.prepare_installed_habitat_runtime = fail_if_resume_prepares_runtime
    monkeypatch.setitem(sys.modules, "avengine.m1.habitat_capture", habitat_module)
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
            "--runtime-prefix",
            str(tmp_path / "runtime-prefix"),
            "--mp3d-root",
            str(tmp_path / "mp3d"),
            "--magnum-python-site",
            str(tmp_path / "magnum"),
            "--rlr-sdk-root",
            str(tmp_path / "rlr"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    manifest = load_json(output / "batch_manifest.json")
    assert manifest["room_manifest_binding"] == room_manifest_binding
    assert manifest["acoustic_selection"] == acoustic_binding


def test_m7_batch_prepares_one_explicit_runtime_for_all_rendered_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_registry = tmp_path / "runtime_registry.json"
    routes = [tmp_path / "route0.json", tmp_path / "route1.json"]
    write_json(runtime_registry, {"schema": "fixture"})
    for route in routes:
        write_json(route, {"schema": "fixture"})
    output = tmp_path / "batch"
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
    monkeypatch.setattr(
        habitat_batch,
        "_select_profile",
        lambda _path, _profile_id: profile,
    )
    monkeypatch.setattr(
        habitat_batch,
        "_verify_room_manifest_binding",
        lambda **_kwargs: {"schema": "room-binding", "status": "pass"},
    )
    monkeypatch.setattr(
        habitat_batch,
        "_resolve_acoustic_selection_binding",
        lambda **_kwargs: {"schema": "acoustic-binding"},
    )

    runtime = SimpleNamespace(mp3d_root=tmp_path / "mp3d")
    prepare_calls: list[dict[str, object]] = []
    capture_calls: list[dict[str, object]] = []

    def fake_prepare(**kwargs):
        prepare_calls.append(kwargs)
        return runtime

    def fake_capture(**kwargs):
        capture_calls.append(kwargs)
        write_json(
            Path(kwargs["output_dir"]) / "mp3d_gate_evidence.json",
            {
                "status": "pass",
                "frame_count": 270,
                "frame_rate_hz": 15,
                "route_id": Path(kwargs["route_manifest_path"]).stem,
                "gates": [{"gate_id": "readback", "status": "pass"}],
            },
        )

    habitat_module = ModuleType("avengine.m1.habitat_capture")
    habitat_module.prepare_installed_habitat_runtime = fake_prepare
    capture_module = ModuleType("avengine.m5_1.mp3d_capture")
    capture_module.capture_mp3d_route = fake_capture
    monkeypatch.setitem(sys.modules, "avengine.m1.habitat_capture", habitat_module)
    monkeypatch.setitem(sys.modules, "avengine.m5_1.mp3d_capture", capture_module)

    result = habitat_batch.main(
        [
            "--room-runtime-registry",
            str(runtime_registry),
            "--room-profile",
            "habitat_room",
            "--episode",
            f"episode0={routes[0]}",
            "--episode",
            f"episode1={routes[1]}",
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
            "--runtime-prefix",
            str(tmp_path / "runtime-prefix"),
            "--mp3d-root",
            str(tmp_path / "mp3d"),
            "--magnum-python-site",
            str(tmp_path / "magnum"),
            "--rlr-sdk-root",
            str(tmp_path / "rlr"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert prepare_calls == [
        {
            "runtime_prefix": tmp_path / "runtime-prefix",
            "runtime_root": None,
            "mp3d_root": tmp_path / "mp3d",
            "magnum_python_site": tmp_path / "magnum",
            "rlr_sdk_root": tmp_path / "rlr",
            "allow_mp3d_environment": False,
        }
    ]
    assert len(capture_calls) == 2
    assert all(call["installed_runtime"] is runtime for call in capture_calls)
    assert all("runtime_root" not in call for call in capture_calls)
    manifest = load_json(output / "batch_manifest.json")
    assert [item["episode_role"] for item in manifest["episodes"]] == [
        "review_only",
        "review_only",
    ]
    assert all(
        item["frame_count_matches_profile_contract"] is False
        for item in manifest["episodes"]
    )
    assert capsys.readouterr().out.splitlines() == [
        "HABITAT_BATCH_EPISODE_OK id=episode0 resumed=False frames=270 gates=1",
        "HABITAT_BATCH_EPISODE_OK id=episode1 resumed=False frames=270 gates=1",
        f"HABITAT_BATCH_OK output={output} selected=2 of 2 episodes",
    ]
