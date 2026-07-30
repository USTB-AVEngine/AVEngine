from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avengine.acoustic_profiles import (
    SOLVER_BACKEND_ID,
    AcousticProfileError,
    AcousticProfileSelection,
    acoustic_profile_package_binding,
)
from avengine.contracts.json_io import load_json, sha256_file
from avengine.m3.compiler import AcousticSceneCompileError
from avengine.m3.contracts import load_and_validate_acoustic_scene_package
from avengine.m3.profiled_compiler import compile_profiled_acoustic_scene


ROOT = Path(__file__).resolve().parents[2]
ROOM_MANIFEST = ROOT / "examples/m1/rooms/blender_custom/room_manifest.json"
ROOM_GEOMETRY = (
    ROOT
    / "examples/m1/rooms/blender_custom/visual/stages/m1_custom_room.glb"
)
MATERIAL_RULES = (
    ROOT / "examples/m3/semantic_materials/residential_material_rules.json"
)
SIMULATION_REQUEST = (
    ROOT / "examples/runtime/rir_cache_simulation_request_v2.json"
)

_VISUAL_ROUTES = {
    "habitat_scene": (
        "habitat_visual_material_slots_residential_v1",
        "habitat_visual_material_rules",
        "habitat_visual_slots_to_m3_v1",
    ),
    "spear_ue_authored": (
        "spear_ue_material_slot_authored_v1",
        "spear_ue_authored_material_rules",
        "spear_ue_export_visual_slots_to_m3_v1",
    ),
}


def _selection(
    *,
    origin_kind: str,
    adapter_id: str,
    material_role: str,
    room_id: str | None = None,
    geometry_sha256: str | None = None,
    material_sha256: str | None = None,
) -> AcousticProfileSelection:
    room = load_json(ROOM_MANIFEST)
    selected_room_id = room_id or room["room_id"]
    profile_id = f"fixture_{origin_kind}_profile_v1"
    profile_revision = "fixture_material_revision_v1"
    room_revision = "fixture_room_revision_v1"
    room_ref = {
        "registry_id": "fixture_room_registry_v1",
        "room_id": selected_room_id,
        "revision": room_revision,
    }
    effective_geometry_sha256 = (
        geometry_sha256
        if geometry_sha256 is not None
        else sha256_file(ROOM_GEOMETRY)
    )
    effective_material_sha256 = (
        material_sha256
        if material_sha256 is not None
        else sha256_file(MATERIAL_RULES)
    )
    acoustic_resource = {
        "resource_id": "fixture_acoustic_package",
        "resource_type": "acoustic_package",
    }
    representation = {
        "representation_id": "fixture_acoustic_representation_v1",
        "resource_id": acoustic_resource["resource_id"],
        "input_resource_ids": ["fixture_room_geometry"],
    }
    profile = {
        "profile_id": profile_id,
        "revision": profile_revision,
        "solver_backend_id": SOLVER_BACKEND_ID,
        "origin": {
            "kind": origin_kind,
            "project": "AVEngine profiled compiler test fixture",
            "source_revision": "fixture_source_revision_v1",
            "qualification_claim": False,
        },
        "material_binding": {
            "adapter_id": adapter_id,
            "resources": [
                {
                    "role": material_role,
                    "path": {
                        "kind": "repository_relative",
                        "path": str(MATERIAL_RULES.relative_to(ROOT)),
                        "sha256": effective_material_sha256,
                    },
                }
            ],
        },
        "simulation": {
            "production": {
                "kind": "repository_relative",
                "path": str(SIMULATION_REQUEST.relative_to(ROOT)),
                "sha256": sha256_file(SIMULATION_REQUEST),
            }
        },
    }
    binding = {
        "binding_id": f"fixture_{origin_kind}_binding_v1",
        "room_ref": room_ref,
        "lineage_acoustic_profile_id": profile_id,
        "profile_ref": {
            "profile_id": profile_id,
            "revision": profile_revision,
        },
        "acoustic_representation_id": representation["representation_id"],
        "acoustic_resource_id": acoustic_resource["resource_id"],
        "geometry_resource_id": "fixture_room_geometry",
    }
    room_record = {
        "room_id": selected_room_id,
        "revision": room_revision,
        "coordinate_system": dict(room["coordinate_system"]),
        "resources": [
            {
                "resource_id": "fixture_room_manifest",
                "resource_type": "room_manifest",
                "sha256": sha256_file(ROOM_MANIFEST),
            },
            {
                "resource_id": "fixture_room_geometry",
                "resource_type": "visual_scene",
                "sha256": effective_geometry_sha256,
            },
            acoustic_resource,
        ],
        "acoustic_representations": [representation],
    }
    return AcousticProfileSelection(
        solver_backend_id=SOLVER_BACKEND_ID,
        profile=profile,
        binding=binding,
        room_record=room_record,
        representation=representation,
        resource=acoustic_resource,
        acoustic_package_manifest_path=None,
        material_paths={material_role: MATERIAL_RULES},
        simulation_request_path=SIMULATION_REQUEST,
        reference_simulation_request_path=None,
        verification_status="verified",
        acoustic_profile_registry_sha256="1" * 64,
        room_registry_sha256="2" * 64,
        _path_records={},
    )


@pytest.fixture(
    scope="module",
    params=tuple(_VISUAL_ROUTES),
    ids=("habitat", "spear_ue"),
)
def compiled_visual_profile(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
):
    origin_kind = str(request.param)
    adapter_id, material_role, _expected_route = _VISUAL_ROUTES[origin_kind]
    selection = _selection(
        origin_kind=origin_kind,
        adapter_id=adapter_id,
        material_role=material_role,
    )
    output = tmp_path_factory.mktemp(f"profiled-{origin_kind}") / "package"
    result = compile_profiled_acoustic_scene(
        selection,
        room_manifest=ROOM_MANIFEST,
        output=output,
        seed=917,
        probe_origins=[],
        probe_direction_count=4,
    )
    return result


def test_habitat_and_spear_routes_emit_strict_profile_bound_packages(
    compiled_visual_profile,
) -> None:
    result = compiled_visual_profile
    origin_kind = result.selection.profile["origin"]["kind"]
    _adapter_id, _material_role, expected_route = _VISUAL_ROUTES[origin_kind]
    expected_binding = acoustic_profile_package_binding(result.selection)

    package = load_and_validate_acoustic_scene_package(result.manifest_path)

    assert result.compiler_route == expected_route
    assert (
        package.manifest["materials"]["acoustic_profile_binding"]
        == expected_binding
    )
    assert package.manifest["source_room"]["room_id"] == (
        result.selection.binding["room_ref"]["room_id"]
    )


def test_soundspaces_origin_and_adapter_select_exact_compiler_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import avengine.m3.profiled_compiler as profiled_compiler

    selection = _selection(
        origin_kind="soundspaces2_public",
        adapter_id="soundspaces2_mp3d_semantic_labels_v1",
        material_role="soundspaces2_public_material_config",
    )
    expected_binding = acoustic_profile_package_binding(selection)
    calls: list[dict] = []

    def fake_soundspaces_compile(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output"])
        output.mkdir()
        manifest = output / "manifest.json"
        coverage = output / "coverage.json"
        manifest.write_text(
            json.dumps(
                {
                    "materials": {
                        "acoustic_profile_binding": expected_binding,
                    }
                }
            ),
            encoding="utf-8",
        )
        coverage.write_text("{}\n", encoding="utf-8")
        return manifest, coverage

    monkeypatch.setattr(
        profiled_compiler,
        "compile_mp3d_soundspaces_research_scene",
        fake_soundspaces_compile,
    )
    monkeypatch.setattr(
        profiled_compiler,
        "load_and_validate_acoustic_scene_package",
        lambda _path: SimpleNamespace(
            manifest={"materials": {"acoustic_profile_binding": expected_binding}}
        ),
    )
    monkeypatch.setattr(
        profiled_compiler,
        "verify_acoustic_package_for_selection",
        lambda _selection, _path: None,
    )

    result = compile_profiled_acoustic_scene(
        selection,
        room_manifest=ROOM_MANIFEST,
        output=tmp_path / "soundspaces-package",
        probe_origins=[],
        probe_direction_count=4,
    )

    assert result.compiler_route == (
        "soundspaces2_mp3d_public_materials_to_m3_v1"
    )
    assert len(calls) == 1
    assert calls[0]["database_id"] == selection.profile["profile_id"]
    assert calls[0]["material_config"] == MATERIAL_RULES


def test_unknown_origin_adapter_combination_fails_closed(
    tmp_path: Path,
) -> None:
    selection = _selection(
        origin_kind="habitat_scene",
        adapter_id="spear_ue_material_slot_authored_v1",
        material_role="spear_ue_authored_material_rules",
    )

    with pytest.raises(
        AcousticProfileError,
        match="no fail-closed M3 compiler route.*habitat_scene.*spear_ue",
    ):
        compile_profiled_acoustic_scene(
            selection,
            room_manifest=ROOM_MANIFEST,
            output=tmp_path / "must-not-exist",
            probe_origins=[],
            probe_direction_count=4,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_material_hash_mismatch_is_rejected_before_compilation(
    tmp_path: Path,
) -> None:
    adapter_id, material_role, _route = _VISUAL_ROUTES["habitat_scene"]
    selection = _selection(
        origin_kind="habitat_scene",
        adapter_id=adapter_id,
        material_role=material_role,
        material_sha256="f" * 64,
    )

    with pytest.raises(
        AcousticProfileError,
        match="material resource .* sha256 mismatch",
    ):
        compile_profiled_acoustic_scene(
            selection,
            room_manifest=ROOM_MANIFEST,
            output=tmp_path / "must-not-exist",
            probe_origins=[],
            probe_direction_count=4,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_room_identity_mismatch_is_rejected_before_compilation(
    tmp_path: Path,
) -> None:
    adapter_id, material_role, _route = _VISUAL_ROUTES["habitat_scene"]
    selection = _selection(
        origin_kind="habitat_scene",
        adapter_id=adapter_id,
        material_role=material_role,
        room_id="different_room_identity_v1",
    )

    with pytest.raises(
        AcousticSceneCompileError,
        match="room_id does not match exact room_ref",
    ):
        compile_profiled_acoustic_scene(
            selection,
            room_manifest=ROOM_MANIFEST,
            output=tmp_path / "must-not-exist",
            probe_origins=[],
            probe_direction_count=4,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_geometry_identity_mismatch_rejects_generated_package(
    tmp_path: Path,
) -> None:
    adapter_id, material_role, _route = _VISUAL_ROUTES["habitat_scene"]
    selection = _selection(
        origin_kind="habitat_scene",
        adapter_id=adapter_id,
        material_role=material_role,
        geometry_sha256="f" * 64,
    )

    with pytest.raises(
        AcousticProfileError,
        match="exact geometry_resource_id",
    ):
        compile_profiled_acoustic_scene(
            selection,
            room_manifest=ROOM_MANIFEST,
            output=tmp_path / "package",
            probe_origins=[],
            probe_direction_count=4,
        )
    assert not (tmp_path / "package").exists()


def test_compile_receipt_closes_room_profile_geometry_and_solver_identity(
    compiled_visual_profile,
) -> None:
    result = compiled_visual_profile
    receipt = result.receipt()
    selection = result.selection

    assert receipt["solver_backend_id"] == SOLVER_BACKEND_ID
    assert receipt["room_ref"] == selection.binding["room_ref"]
    assert receipt["lineage_acoustic_profile_id"] == (
        selection.binding["lineage_acoustic_profile_id"]
    )
    assert receipt["profile_ref"] == selection.binding["profile_ref"]
    assert receipt["binding_id"] == selection.binding["binding_id"]
    assert receipt["geometry_resource_id"] == (
        selection.binding["geometry_resource_id"]
    )
    assert receipt["acoustic_representation_id"] == (
        selection.representation["representation_id"]
    )
    assert receipt["acoustic_resource_id"] == selection.resource["resource_id"]
    assert receipt["origin"]["kind"] == selection.profile["origin"]["kind"]
    assert receipt["room_manifest"]["room_id"] == (
        selection.binding["room_ref"]["room_id"]
    )
    assert receipt["package"]["acoustic_profile_binding"] == (
        acoustic_profile_package_binding(selection)
    )
    assert len(receipt["receipt_content_sha256"]) == 64
