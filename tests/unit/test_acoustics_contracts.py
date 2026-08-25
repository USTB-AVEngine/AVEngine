from __future__ import annotations

import json
from pathlib import Path

from avengine.contracts.json_io import canonical_json_sha256, load_json
from avengine.acoustics.compiler import compile_custom_acoustic_scene
from avengine.acoustics.contracts import (
    json_schema_errors,
    validate_canary_request,
    validate_package,
)
from avengine.acoustics.materials import MATERIAL_PROFILE_SCHEMA


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY_ROOT / "examples/acoustics/blender_custom"
ROOM = REPOSITORY_ROOT / "examples/rooms/blender_custom/room_manifest.json"


def _compiled_manifest(tmp_path: Path) -> Path:
    return compile_custom_acoustic_scene(
        room_manifest=ROOM,
        material_mapping=EXAMPLES / "mapping.json",
        material_database=EXAMPLES / "materials_low.json",
        output=tmp_path / "package",
    )


def _rewrite_manifest(path: Path, value: dict) -> None:
    value["package_content_sha256"] = canonical_json_sha256(
        {key: item for key, item in value.items() if key != "package_content_sha256"}
    )
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_checked_in_canary_request_declares_runtime_and_metric_thresholds() -> None:
    request = load_json(EXAMPLES / "canary_request.json")

    assert validate_canary_request(request) == []
    assert request["repeat_count"] >= 3
    assert request["simulation"]["temporal_coherence"] is False
    assert request["simulation"]["mesh_simplification"] is False
    assert request["source"]["radius_m"] == 0
    assert request["listener"]["radius_m"] == 0
    assert request["thresholds"]["ray_checks"]["require_nonempty"] is True
    assert set(request["thresholds"]["metrics"]) == {
        "edt_seconds",
        "drr_db",
        "late_energy_ratio",
    }


def test_checked_in_material_profile_passes_registered_schema() -> None:
    profile = load_json(EXAMPLES / "material_profile_example.json")

    assert json_schema_errors(profile, MATERIAL_PROFILE_SCHEMA) == []
    profile["unexpected"] = True
    assert json_schema_errors(profile, MATERIAL_PROFILE_SCHEMA)


def test_canary_request_rejects_hidden_temporal_state_and_coincident_anchors() -> None:
    request = load_json(EXAMPLES / "canary_request.json")
    request["simulation"]["temporal_coherence"] = True
    request["source"]["position_m"] = request["listener"]["position_m"]

    errors = validate_canary_request(request)

    assert any("temporal_coherence" in error for error in errors)
    assert any("distinct" in error for error in errors)


def test_canary_request_rejects_weakened_admission_thresholds() -> None:
    request = load_json(EXAMPLES / "canary_request.json")
    request["repeat_count"] = 2
    request["simulation"]["indirect_ray_count"] = 4999
    request["thresholds"]["ray_checks"][
        "maximum_first_hit_distance_error_m"
    ] = 0.001
    request["thresholds"]["metrics"]["edt_seconds"][
        "minimum_fit_r2"
    ] = 0.8
    request["thresholds"]["metrics"]["drr_db"][
        "minimum_absolute_effect"
    ] = 0.5

    errors = validate_canary_request(request)

    assert any("repeat_count" in error for error in errors)
    assert any("indirect_ray_count" in error for error in errors)
    assert any("maximum_first_hit_distance_error_m" in error for error in errors)
    assert any("minimum_fit_r2" in error for error in errors)
    assert any("minimum_absolute_effect" in error for error in errors)


def test_package_schema_accepts_extended_array_descriptors(tmp_path: Path) -> None:
    manifest_path = _compiled_manifest(tmp_path)
    manifest = load_json(manifest_path)

    assert json_schema_errors(manifest, "avengine_acoustic_scene_package_v1") == []
    assert validate_package(manifest_path) == []


def test_production_package_requires_reviewed_source_transform(tmp_path: Path) -> None:
    manifest_path = _compiled_manifest(tmp_path)
    manifest = load_json(manifest_path)
    manifest["geometry"]["source_to_canonical"]["reviewed"] = False
    _rewrite_manifest(manifest_path, manifest)

    errors = validate_package(manifest_path)

    assert any("explicitly reviewed" in error for error in errors)


def test_compiler_identity_is_a_composite_hash(tmp_path: Path) -> None:
    manifest_path = _compiled_manifest(tmp_path)
    manifest = load_json(manifest_path)
    manifest["compiler"]["components"]["qa.py"] = "0" * 64
    _rewrite_manifest(manifest_path, manifest)

    errors = validate_package(manifest_path)

    assert any("component hash" in error for error in errors)


def test_package_artifact_paths_cannot_escape_root(tmp_path: Path) -> None:
    manifest_path = _compiled_manifest(tmp_path)
    manifest = load_json(manifest_path)
    manifest["debug_mesh"]["path"] = "../outside.obj"
    _rewrite_manifest(manifest_path, manifest)

    errors = validate_package(manifest_path)

    assert any("confined" in error for error in errors)
