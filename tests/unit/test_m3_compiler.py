from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m3.compiler import (
    AcousticSceneCompileError,
    compile_canary_request,
    compile_custom_acoustic_scene,
    compile_explicit_glb_research_scene,
    compile_visual_slot_semantic_research_scene,
    propose_visual_slot_research_materials,
)
from avengine.m3.contracts import (
    AcousticSceneContractError,
    load_and_validate_acoustic_scene_package,
)
from avengine.m3.evidence import (
    load_and_verify_compile_evidence,
    verify_compile_evidence,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOM = REPOSITORY_ROOT / "examples/m1/rooms/blender_custom/room_manifest.json"
EXAMPLES = REPOSITORY_ROOT / "examples/m3/blender_custom"
MAPPING = EXAMPLES / "mapping.json"
LOW = EXAMPLES / "materials_low.json"
REQUEST = EXAMPLES / "canary_request.json"


def test_custom_compiler_emits_strict_144_triangle_package(tmp_path: Path) -> None:
    manifest = compile_custom_acoustic_scene(
        room_manifest=ROOM,
        material_mapping=MAPPING,
        material_database=LOW,
        output=tmp_path / "package",
    )
    package = load_and_validate_acoustic_scene_package(manifest)

    assert package.vertex_count == 288
    assert package.triangle_count == 144
    assert package.object_count == 12
    assert package.material_category_count == 4
    assert sorted(package.category_triangle_counts.values()) == [12, 12, 36, 84]
    assert package.manifest["package_mode"] == "production"
    assert package.manifest["geometry"]["source_to_canonical"]["reviewed"] is True
    assert all(report["status"] == "pass" for report in package.qa_reports.values())
    assert (
        package.qa_reports["geometry_report"]["topology"]
        ["global_nonmanifold_is_inter_object_junction_diagnostic"]
        is True
    )
    assert (
        package.qa_reports["geometry_report"]["topology"]
        ["per_object_nonmanifold_edge_count_after_exact_weld"]
        == 0
    )
    assert package.qa_reports["ray_leakage"]["declared_check_count"] == 4
    assert package.qa_reports["ray_leakage"]["rlr_runtime_ray_check_status"] == "not_run"


def test_low_high_compile_evidence_freezes_geometry_and_verifies(tmp_path: Path) -> None:
    evidence_path = compile_canary_request(REQUEST, tmp_path / "canary")

    status, checks = verify_compile_evidence(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert status == "pass"
    assert all(check["status"] == "pass" for check in checks)
    assert all(
        record["identical"]
        for record in evidence["frozen_variable_proof"].values()
    )
    assert evidence["runtime_material_activation"]["status"] == "not_run"
    low = load_and_validate_acoustic_scene_package(
        tmp_path / "canary/low_absorption/manifest.json"
    )
    high = load_and_validate_acoustic_scene_package(
        tmp_path / "canary/high_absorption/manifest.json"
    )
    assert low.rlr_material_database != high.rlr_material_database


def test_compiler_is_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "package"
    compile_custom_acoustic_scene(
        room_manifest=ROOM,
        material_mapping=MAPPING,
        material_database=LOW,
        output=output,
    )

    with pytest.raises(AcousticSceneCompileError, match="already exists"):
        compile_custom_acoustic_scene(
            room_manifest=ROOM,
            material_mapping=MAPPING,
            material_database=LOW,
            output=output,
        )


def test_strict_loader_rejects_tampered_array(tmp_path: Path) -> None:
    manifest = compile_custom_acoustic_scene(
        room_manifest=ROOM,
        material_mapping=MAPPING,
        material_database=LOW,
        output=tmp_path / "package",
    )
    vertices = manifest.parent / "acoustic/vertices.npy"
    payload = bytearray(vertices.read_bytes())
    payload[-1] ^= 1
    vertices.write_bytes(payload)

    with pytest.raises(AcousticSceneContractError, match="sha256"):
        load_and_validate_acoustic_scene_package(manifest)


def test_production_loader_replays_qa_status_not_only_file_hash(tmp_path: Path) -> None:
    manifest_path = compile_custom_acoustic_scene(
        room_manifest=ROOM,
        material_mapping=MAPPING,
        material_database=LOW,
        output=tmp_path / "package",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage_path = manifest_path.parent / manifest["qa"]["material_coverage"]["path"]
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["status"] = "fail"
    coverage_path.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["qa"]["material_coverage"]["sha256"] = sha256_file(coverage_path)
    manifest["qa"]["material_coverage"]["byte_size"] = coverage_path.stat().st_size
    manifest["package_content_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "package_content_sha256"}
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(AcousticSceneContractError, match="must have status pass"):
        load_and_validate_acoustic_scene_package(manifest_path)


def test_generic_visual_slot_path_is_explicitly_research_only(tmp_path: Path) -> None:
    mapping, materials, report = propose_visual_slot_research_materials(
        room_manifest=ROOM,
        output=tmp_path / "proposal",
        transform_profile="identity_y_up",
    )
    proposal = json.loads(report.read_text(encoding="utf-8"))
    mapping_value = json.loads(mapping.read_text(encoding="utf-8"))

    assert proposal["qualification_claim"] is False
    assert proposal["physical_acoustic_material_claim"] is False
    assert mapping_value["mapping_source_kind"] == "visual_material_slot_proposal"
    assert mapping_value["source_to_canonical"]["reviewed"] is False

    manifest = compile_explicit_glb_research_scene(
        room_manifest=ROOM,
        material_mapping=mapping,
        material_database=materials,
        output=tmp_path / "research_package",
    )
    package = load_and_validate_acoustic_scene_package(manifest)
    assert package.manifest["package_mode"] == "research_candidate"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_manifest_record(
    manifest_path: Path, manifest: dict, record: dict, artifact_path: Path
) -> None:
    record["sha256"] = sha256_file(artifact_path)
    record["byte_size"] = artifact_path.stat().st_size
    manifest["package_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "package_content_sha256"
        }
    )
    _write_json(manifest_path, manifest)


def test_production_compiler_rejects_visual_placeholder_privilege_escalation(
    tmp_path: Path,
) -> None:
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    mapping["mapping_source_kind"] = "visual_material_slot_proposal"
    for entry in mapping["entries"]:
        entry["mapping_confidence"] = 0.0
        entry["human_override"] = False
    mapping_path = tmp_path / "visual_mapping.json"
    _write_json(mapping_path, mapping)

    with pytest.raises(
        AcousticSceneCompileError, match="production material admission rejected"
    ):
        compile_custom_acoustic_scene(
            room_manifest=ROOM,
            material_mapping=mapping_path,
            material_database=LOW,
            output=tmp_path / "forbidden_production",
        )


def test_canary_compiler_rejects_non_absorption_counterfactual_change(
    tmp_path: Path,
) -> None:
    high = json.loads((EXAMPLES / "materials_high.json").read_text(encoding="utf-8"))
    high["materials"][0]["scattering"][0] = 0.02
    high_path = tmp_path / "high_changed.json"
    _write_json(high_path, high)
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request["room_manifest"] = str(ROOM)
    request["material_mapping"] = str(MAPPING)
    request["material_databases"] = {
        "low_absorption": str(LOW),
        "high_absorption": str(high_path),
    }
    request_path = tmp_path / "request.json"
    _write_json(request_path, request)

    with pytest.raises(AcousticSceneCompileError, match="absorption-only"):
        compile_canary_request(request_path, tmp_path / "canary")


def test_strict_loader_recompiles_hash_bound_source_mapping(tmp_path: Path) -> None:
    manifest_path = compile_custom_acoustic_scene(
        room_manifest=ROOM,
        material_mapping=MAPPING,
        material_database=LOW,
        output=tmp_path / "package",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping_record = manifest["materials"]["source_mapping"]
    mapping_path = manifest_path.parent / mapping_record["path"]
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["entries"][0]["material_id"] = 1
    mapping["entries"][1]["material_id"] = 0
    _write_json(mapping_path, mapping)
    manifest["materials"]["mapping_sha256"] = sha256_file(mapping_path)
    _rewrite_manifest_record(manifest_path, manifest, mapping_record, mapping_path)

    with pytest.raises(AcousticSceneContractError, match="source-input replay"):
        load_and_validate_acoustic_scene_package(manifest_path)


def test_compile_evidence_replays_hash_bound_source_glb(tmp_path: Path) -> None:
    evidence_path = compile_canary_request(REQUEST, tmp_path / "canary")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    geometry_record = evidence["source_inputs"]["source_geometry"]
    geometry_path = evidence_path.parent / geometry_record["path"]
    payload = bytearray(geometry_path.read_bytes())
    payload[-1] ^= 1
    geometry_path.write_bytes(payload)
    geometry_record["sha256"] = sha256_file(geometry_path)
    geometry_record["byte_size"] = geometry_path.stat().st_size
    evidence["evidence_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in evidence.items()
            if key != "evidence_content_sha256"
        }
    )
    _write_json(evidence_path, evidence)

    status, checks = verify_compile_evidence(evidence_path)

    assert status == "fail"
    assert any(
        check["check_id"] in {
            "compile_source_input_contracts",
            "compile_source_glb_to_package_replay",
        }
        and check["status"] == "fail"
        for check in checks
    )


def test_strict_loader_rejects_rehashed_parity_claim_tamper(tmp_path: Path) -> None:
    manifest_path = compile_custom_acoustic_scene(
        room_manifest=ROOM,
        material_mapping=MAPPING,
        material_database=LOW,
        output=tmp_path / "package",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["qa"]["compiler_source_to_package_parity"]
    report_path = manifest_path.parent / record["path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["array_hashes"]["canonical_expected_vertices"] = "0" * 64
    _write_json(report_path, report)
    _rewrite_manifest_record(manifest_path, manifest, record, report_path)

    with pytest.raises(AcousticSceneContractError, match="array hashes"):
        load_and_validate_acoustic_scene_package(manifest_path)


def test_strict_loader_reads_each_package_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = compile_custom_acoustic_scene(
        room_manifest=ROOM,
        material_mapping=MAPPING,
        material_database=LOW,
        output=tmp_path / "package",
    )
    expected_manifest_sha256 = sha256_file(manifest_path)
    original_read_bytes = Path.read_bytes
    read_counts: dict[Path, int] = {}

    def unstable_second_read(path: Path) -> bytes:
        resolved = path.resolve()
        read_counts[resolved] = read_counts.get(resolved, 0) + 1
        payload = original_read_bytes(path)
        return payload if read_counts[resolved] == 1 else b"changed-on-second-read"

    monkeypatch.setattr(Path, "read_bytes", unstable_second_read)

    package = load_and_validate_acoustic_scene_package(manifest_path)

    assert package.manifest_file_sha256 == expected_manifest_sha256
    assert package.source_material_database["database_id"] == (
        "m3_controlled_low_absorption_v1"
    )
    assert read_counts
    assert set(read_counts.values()) == {1}


def test_compile_evidence_reuses_one_snapshot_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path = compile_canary_request(REQUEST, tmp_path / "canary")
    original_read_bytes = Path.read_bytes
    read_counts: dict[Path, int] = {}

    def unstable_second_read(path: Path) -> bytes:
        resolved = path.resolve()
        read_counts[resolved] = read_counts.get(resolved, 0) + 1
        payload = original_read_bytes(path)
        return payload if read_counts[resolved] == 1 else b"changed-on-second-read"

    monkeypatch.setattr(Path, "read_bytes", unstable_second_read)

    result = load_and_verify_compile_evidence(evidence_path)

    assert result.status == "pass", result.checks
    assert set(result.packages) == {"low_absorption", "high_absorption"}
    assert result.evidence_snapshot is not None
    assert read_counts
    assert set(read_counts.values()) == {1}


def test_visual_slot_semantic_compile_is_deterministic_research_candidate(
    tmp_path: Path,
) -> None:
    rules = (
        REPOSITORY_ROOT
        / "examples/m3/semantic_materials/residential_material_rules.json"
    )
    manifest_path, coverage_path = compile_visual_slot_semantic_research_scene(
        room_manifest=ROOM,
        material_rules=rules,
        output=tmp_path / "package_a",
        seed=917,
        transform_profile="identity_y_up",
        transform_reviewed=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))

    assert manifest["package_mode"] == "research_candidate"
    assert coverage["source_kind"] == "visual_material_slots"
    assert coverage["physical_material_claim"] is False
    assert coverage["qualification_claim"] is False
    assert coverage["source_material_slot_count"] > 0
    assert coverage["source_to_canonical_reviewed"] is True
    for decision in coverage["decisions"]:
        assert decision["semantic_category"] == "unknown"
        assert decision["resolution"] in {
            "explicit_override",
            "name_hint",
            "default_candidate",
        }
    total = sum(coverage["resolution_counts"].values())
    assert total == coverage["source_material_slot_count"]

    # The strict production loader must replay the generated package.
    load_and_validate_acoustic_scene_package(manifest_path)

    # Same seed and inputs must reproduce identical material decisions.
    manifest_path_b, coverage_path_b = compile_visual_slot_semantic_research_scene(
        room_manifest=ROOM,
        material_rules=rules,
        output=tmp_path / "package_b",
        seed=917,
        transform_profile="identity_y_up",
        transform_reviewed=True,
    )
    coverage_b = json.loads(coverage_path_b.read_text(encoding="utf-8"))
    assert coverage_b["decisions"] == coverage["decisions"]
    manifest_b = json.loads(manifest_path_b.read_text(encoding="utf-8"))
    assert (
        manifest_b["materials"]["source_database"]["sha256"]
        == manifest["materials"]["source_database"]["sha256"]
    )
