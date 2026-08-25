from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.contracts.json_io import file_record
from avengine.rooms.feasibility_canary import (
    M6CanaryError,
    _artifact_errors,
    _controlled_execution_contract,
    _load_registries,
    _materialization_runtime_report,
    _retained_materialization_claim_errors,
    _retained_materialization_room_report,
    _runtime_checks,
    _schema_errors,
    _spatial_format,
    _validate_entity_visual_authority,
    bind_controlled_canary_request_hash,
    load_controlled_canary_request,
    run_controlled_canary,
    validate_controlled_canary_request,
    verify_controlled_canary_evidence,
)


ROOT = Path(__file__).resolve().parents[2]
REQUEST = ROOT / "examples" / "m6" / "canary" / "controlled_one_active_of_two_request.json"


def test_controlled_request_is_hash_bound_and_research_only() -> None:
    request = load_controlled_canary_request(REQUEST)
    assert request["research_only"] is True
    assert request["qualification_claim"] is False
    assert request["upstream_evidence"]["episode_variant"] == "A"
    assert validate_controlled_canary_request(request) == []


def test_controlled_request_requires_exact_endpoint_mapping() -> None:
    request = load_json(REQUEST)
    request["endpoint_to_upstream_source_id"].pop("beagle_1_muzzle")
    request = bind_controlled_canary_request_hash(request)
    errors = validate_controlled_canary_request(request)
    assert any("cover exactly" in error for error in errors)


def test_controlled_request_hash_detects_semantic_mutation() -> None:
    request = load_json(REQUEST)
    request["listener"]["observer_yaw_deg"] = 90
    assert any(
        "request_content_sha256" in error
        for error in validate_controlled_canary_request(request)
    )
    rebound = bind_controlled_canary_request_hash(request)
    assert validate_controlled_canary_request(rebound) == []


def test_controlled_request_rejects_variant_b_and_more_than_two_endpoints() -> None:
    request = load_json(REQUEST)
    request["upstream_evidence"]["episode_variant"] = "B"
    request = bind_controlled_canary_request_hash(request)
    assert any(
        "'A' was expected" in error
        for error in validate_controlled_canary_request(request)
    )

    request = load_json(REQUEST)
    request["source_endpoint_ids"].append("beagle_2_muzzle")
    request["endpoint_to_upstream_source_id"]["beagle_2_muzzle"] = "source2"
    request = bind_controlled_canary_request_hash(request)
    errors = validate_controlled_canary_request(request)
    assert any("too long" in error or "more than 2" in error for error in errors)


def test_malformed_entity_instance_returns_errors_instead_of_raising() -> None:
    request = load_json(REQUEST)
    request["entity_instances"].append({})
    request = bind_controlled_canary_request_hash(request)
    errors = validate_controlled_canary_request(request)
    assert errors
    assert any("entity_instances.2" in error for error in errors)


def test_run_rejects_non_commit_before_materialization(tmp_path: Path) -> None:
    with pytest.raises(M6CanaryError, match="implementation_commit"):
        run_controlled_canary(
            request_path=REQUEST,
            upstream_evidence_path=tmp_path / "absent.json",
            output_directory=tmp_path / "output",
            implementation_commit="not-a-commit",
        )


def test_retained_materialization_pass_never_promotes_native_episode(
    tmp_path: Path,
) -> None:
    upstream = tmp_path / "upstream_m5_evidence.json"
    upstream.write_text("{}\n", encoding="utf-8")
    historical = load_json(
        ROOT
        / "examples"
        / "m6"
        / "rooms"
        / "qualification"
        / "blender_custom_two_zone.json"
    )
    runtime = _materialization_runtime_report(
        [
            {
                "check_id": "retained_materialization_fixture",
                "status": "pass",
                "measured": {"native_rir_rerun": False},
            }
        ]
    )
    room_report = _retained_materialization_room_report(
        historical_report=historical,
        upstream_evidence=upstream,
        materialization_status=runtime["overall_status"],
    )
    execution = _controlled_execution_contract()

    assert runtime["overall_status"] == "pass"
    assert {key: runtime[key] for key in execution} == execution
    assert room_report["evidence_basis"] == (
        "verified_retained_evidence_materialization"
    )
    episode = room_report["dimensions"]["episode_feasibility_status"]
    assert episode["status"] == "not_run"
    assert episode["blocker_code"] == "m6_current_native_episode_not_run"
    assert "materialization semantic verifier status is pass" in episode["summary"]
    assert "episode_feasibility_status=not_run" in room_report["admission_blockers"]

    provenance = {
        "derivation": {
            "native_rir_rerun": False,
            "current_native_episode_status": "not_run",
            "semantic_materialization_verifier_status": "pass",
        }
    }
    evidence = {**execution, "overall_status": "pass"}
    assert (
        _retained_materialization_claim_errors(
            evidence=evidence,
            provenance=provenance,
            room_report=room_report,
        )
        == []
    )

    contradictory = deepcopy(room_report)
    contradictory["evidence_basis"] = "current_execution"
    contradictory["dimensions"]["episode_feasibility_status"] = {
        "status": "pass",
        "summary": "incorrectly promoted retained evidence",
        "evidence_refs": ["evidence.json"],
    }
    contradiction_errors = _retained_materialization_claim_errors(
        evidence=evidence,
        provenance=provenance,
        room_report=contradictory,
    )
    assert any("evidence basis" in error for error in contradiction_errors)
    assert any("episode must remain not_run" in error for error in contradiction_errors)


def test_controlled_evidence_schema_requires_explicit_non_native_scope() -> None:
    artifact = {"path": "payload.json", "byte_size": 2, "sha256": "0" * 64}
    evidence = {
        "schema": "avengine_m6_canary_evidence_v1",
        "run_id": "controlled_fixture",
        "evidence_kind": "controlled_one_active_of_n",
        **_controlled_execution_contract(),
        "research_only": True,
        "qualification_claim": False,
        "dataset_admission": False,
        "implementation_commit": "1" * 40,
        "request": artifact,
        "release_manifest_ref": artifact,
        "upstream_evidence": {
            "kind": "verified_m5_controlled_bundle",
            "status": "pass",
            "path": "upstream.json",
            "sha256": "2" * 64,
        },
        "artifacts": {"payload.json": artifact},
        "checks": [
            {"check_id": "semantic_materialization", "status": "pass", "measured": {}}
        ],
        "overall_status": "pass",
        "evidence_content_sha256": "3" * 64,
    }
    assert _schema_errors(evidence, "m6_canary_evidence_v1.schema.json") == []

    missing_scope = deepcopy(evidence)
    missing_scope.pop("status_scope")
    assert any(
        "status_scope" in error
        for error in _schema_errors(
            missing_scope, "m6_canary_evidence_v1.schema.json"
        )
    )

    false_native_pass = deepcopy(evidence)
    false_native_pass["native_execution"]["habitat_sim"] = "pass"
    assert any(
        "not_run" in error
        for error in _schema_errors(
            false_native_pass, "m6_canary_evidence_v1.schema.json"
        )
    )


def test_artifact_closure_rejects_symlink_components(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(target.name)
    evidence = {
        "artifacts": {
            "target.json": file_record(target, relative_to=tmp_path),
            "linked.json": file_record(linked, relative_to=tmp_path),
        }
    }
    errors = _artifact_errors(tmp_path, evidence)
    assert any("symlink" in item and "linked.json" in item for item in errors)


def test_artifact_closure_rejects_undeclared_symlink_directory(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text("{}\n", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    (outside / "hidden.json").write_text("{}\n", encoding="utf-8")
    linked_directory = tmp_path / "extra_directory"
    linked_directory.symlink_to(outside, target_is_directory=True)
    evidence = {
        "artifacts": {
            "payload.json": file_record(payload, relative_to=tmp_path),
        }
    }
    errors = _artifact_errors(tmp_path, evidence)
    assert any("symlink" in item and "extra_directory" in item for item in errors)


def test_artifact_closure_does_not_ignore_nested_entry_filename(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text("{}\n", encoding="utf-8")
    nested = tmp_path / "nested" / "evidence.json"
    nested.parent.mkdir()
    nested.write_text("{}\n", encoding="utf-8")
    evidence = {
        "artifacts": {
            "payload.json": file_record(payload, relative_to=tmp_path),
        }
    }
    assert "artifact index is not an exact retained-file closure" in _artifact_errors(
        tmp_path, evidence
    )


def test_entity_registry_visual_is_the_mesh_rendered_by_retained_m5() -> None:
    request = load_controlled_canary_request(REQUEST)
    registries = _load_registries(ROOT / "examples" / "m6" / "registries")
    upstream = load_json(
        ROOT
        / "examples"
        / "m5"
        / "blender_custom"
        / "two_dog_simultaneous_counterfactual_request.json"
    )
    _validate_entity_visual_authority(
        request=request,
        registries=registries,
        upstream_request=upstream,
    )

    mismatched_upstream = deepcopy(upstream)
    mismatched_upstream["actors"][0]["mesh_sha256"] = "0" * 64
    with pytest.raises(M6CanaryError, match="visual mesh differs"):
        _validate_entity_visual_authority(
            request=request,
            registries=registries,
            upstream_request=mismatched_upstream,
        )


def test_runtime_check_rejects_wrong_foa_order_and_contract_is_n3d_world() -> None:
    checks = _runtime_checks(
        upstream_checks=[{"check_id": "fixture", "status": "pass"}],
        audio_records={
            "layouts": {
                "binaural": {
                    "channel_count": 2,
                    "channel_labels": ["left", "right"],
                },
                "foa": {
                    "channel_count": 4,
                    "channel_labels": ["W", "X", "Y", "Z"],
                },
            }
        },
        mux_reports={
            "primary": {"video_stream_copy_verified": True},
            "topdown": {"video_stream_copy_verified": True},
        },
        aac_diagnostics={
            "presentation_sample_count_matches": True,
            "lr_swap_suspected": False,
            "minimum_correlation": 1.0,
            "minimum_snr_db": 100.0,
        },
        active_endpoint_count=1,
        silent_endpoint_count=1,
        event_count=1,
    )
    layout_check = next(
        item
        for item in checks
        if item["check_id"] == "foa_and_360_binaural_retained"
    )
    assert layout_check["status"] == "fail"
    foa = _spatial_format("foa")
    assert foa["raw_channel_order"] == ["W", "Y", "Z", "X"]
    assert foa["normalization"] == "N3D"
    assert foa["coordinate_frame"] == "avengine_world"


@pytest.mark.release_canary
@pytest.mark.canary
def test_local_formal_controlled_bundle_if_explicitly_enabled() -> None:
    if os.environ.get("AVENGINE_RUN_LOCAL_M6_CANARY_TEST") != "1":
        pytest.skip("set AVENGINE_RUN_LOCAL_M6_CANARY_TEST=1 for retained-evidence readback")
    evidence = ROOT / "tmp" / "m6" / "formal_controlled_v1" / "evidence.json"
    if not evidence.is_file():
        pytest.skip("formal controlled M6 evidence has not been materialized")
    status, checks = verify_controlled_canary_evidence(evidence)
    assert status == "pass", [item for item in checks if item["status"] != "pass"]
