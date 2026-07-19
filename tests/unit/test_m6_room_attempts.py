from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m6.room_attempts import (
    ATTEMPT_CASE_IDS,
    _declared_derivation_assessment,
    _formal_registry_git_binding,
    _git_provenance_observation,
    _proxy_descriptor_assessment,
    run_room_qualification_attempt,
    verify_room_qualification_attempt,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _rewrite_manifest(path: Path, value: dict) -> None:
    core = dict(value)
    core.pop("content_sha256", None)
    value["content_sha256"] = canonical_json_sha256(core)
    write_json(path, value)


def _rehash_attempt_bundle(manifest_path: Path) -> None:
    manifest = load_json(manifest_path)
    for record in manifest["reports"]:
        report_path = manifest_path.parent / record["path"]
        report = load_json(report_path)
        for artifact in report["evidence_artifacts"]:
            artifact_path = manifest_path.parent / artifact["path"]
            artifact["sha256"] = sha256_file(artifact_path)
        write_json(report_path, report)
        record["byte_size"] = report_path.stat().st_size
        record["sha256"] = sha256_file(report_path)
    for record in manifest["artifacts"]:
        artifact_path = manifest_path.parent / record["path"]
        record["byte_size"] = artifact_path.stat().st_size
        record["sha256"] = sha256_file(artifact_path)
    _rewrite_manifest(manifest_path, manifest)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_git_repository(path: Path) -> tuple[str, bytes]:
    path.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(path)],
        check=True,
        capture_output=True,
    )
    _git(path, "config", "user.email", "m6-room-tests@example.invalid")
    _git(path, "config", "user.name", "M6 Room Tests")
    registry_bytes = (
        REPOSITORY_ROOT / "examples/m6/rooms/room_registry.json"
    ).read_bytes()
    registry = path / "examples/m6/rooms/room_registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_bytes(registry_bytes)
    (path / ".gitignore").write_text("tmp/\n", encoding="utf-8")
    _git(path, "add", ".gitignore", "examples/m6/rooms/room_registry.json")
    _git(path, "commit", "-q", "-m", "canonical room registry")
    return _git(path, "rev-parse", "HEAD"), registry_bytes


def _run_minimal(tmp_path: Path) -> Path:
    return run_room_qualification_attempt(
        registry_path="examples/m6/rooms/room_registry.json",
        corrupted_fixture_path=(
            "tests/fixtures/m6/corrupted_acoustic_package/fixture.json"
        ),
        output_directory=tmp_path / "attempt",
        repository_root=REPOSITORY_ROOT,
        environment={},
        attempt_id="unit_minimal_room_attempt",
    )


def test_minimal_attempt_is_complete_honest_and_fail_closed(tmp_path: Path) -> None:
    manifest_path = _run_minimal(tmp_path)

    status, checks = verify_room_qualification_attempt(manifest_path)
    manifest = load_json(manifest_path)

    # A clean checkout promotes this same minimal fixture into formal scope,
    # where the intentionally absent materialized MP3D proxy must fail closed.
    # In an ordinary dirty development checkout it remains a valid diagnostic
    # attempt. Keep both outcomes explicit instead of depending on how the
    # surrounding repository happened to be invoked by pytest.
    if manifest["code_provenance"]["worktree_clean"]:
        assert status == "fail", checks
        binding_check = next(
            check
            for check in checks
            if check["check_id"] == "mp3d_materialized_proxy_binding"
        )
        assert binding_check["status"] == "fail"
    else:
        assert status == "pass", checks
    assert manifest["case_ids"] == list(ATTEMPT_CASE_IDS)
    assert len(manifest["reports"]) == 6
    assert manifest["claims"] == {
        "current_native_runtime_pass": False,
        "dataset_admission_count": 0,
        "historical_artifact_statuses_promoted_to_current_native_pass": False,
        "mp3d_raw_modified": False,
    }
    assert all(record["dataset_admission"] is False for record in manifest["reports"])

    fixture = load_json(
        manifest_path.parent / "reports/independent_corrupted_fixture.json"
    )
    assert fixture["dimensions"]["acoustic_geometry_status"]["status"] == "fail"
    assert fixture["dimensions"]["material_binding_status"]["status"] == "fail"
    assert fixture["dimensions"]["ray_leakage_status"]["status"] == "fail"
    derived_observation = load_json(
        manifest_path.parent / "observations/mp3d_17DRP5sb8fy_derived.json"
    )
    derived_report = load_json(
        manifest_path.parent / "reports/mp3d_17DRP5sb8fy_derived.json"
    )
    binding_status = derived_observation["materialized_proxy_binding"]["status"]
    assert binding_status == "blocked"
    assert (
        derived_report["dimensions"]["acoustic_geometry_status"]["status"]
        == binding_status
    )
    assert "not assessed" in derived_report["dimensions"][
        "acoustic_geometry_status"
    ]["summary"]


def test_attempt_verifier_detects_report_tamper(tmp_path: Path) -> None:
    manifest_path = _run_minimal(tmp_path)
    report_path = manifest_path.parent / "reports/replicacad_apt_0.json"
    report = load_json(report_path)
    report["dimensions"]["visual_runtime_status"]["summary"] = "tampered"
    write_json(report_path, report)

    status, checks = verify_room_qualification_attempt(manifest_path)

    assert status == "fail"
    artifact_check = next(
        check for check in checks if check["check_id"] == "artifact_hashes"
    )
    assert artifact_check["status"] == "fail"


def test_attempt_verifier_rejects_rehashed_top_level_native_and_claim_attack(
    tmp_path: Path,
) -> None:
    manifest_path = _run_minimal(tmp_path)
    manifest = load_json(manifest_path)
    manifest["execution_mode"] = "current_native_execution"
    manifest["native_execution"]["habitat_sim"] = "pass"
    manifest["claims"]["current_native_runtime_pass"] = True
    manifest["claims"]["dataset_admission_count"] = 1
    manifest["claims"]["historical_artifact_statuses_promoted_to_current_native_pass"] = True
    manifest["claims"]["mp3d_raw_modified"] = True
    manifest["claims"]["undeclared_claim"] = False
    _rewrite_manifest(manifest_path, manifest)

    status, checks = verify_room_qualification_attempt(manifest_path)

    assert status == "fail"
    assert next(
        check for check in checks if check["check_id"] == "manifest_schema"
    )["status"] == "fail"
    assert next(
        check for check in checks if check["check_id"] == "no_native_pass_claim"
    )["status"] == "fail"
    assert next(
        check
        for check in checks
        if check["check_id"] == "claims_report_consistency"
    )["status"] == "fail"


def test_attempt_verifier_rejects_fully_rehashed_observation_native_pass_attack(
    tmp_path: Path,
) -> None:
    manifest_path = _run_minimal(tmp_path)
    observation_path = (
        manifest_path.parent / "observations/replicacad_apt_0.json"
    )
    observation = load_json(observation_path)
    observation["execution_mode"] = "current_native_execution"
    observation["native_execution"]["habitat_sim"] = "pass"
    observation["native_execution"]["undeclared_runtime"] = "not_run"
    _rewrite_manifest(observation_path, observation)
    _rehash_attempt_bundle(manifest_path)

    status, checks = verify_room_qualification_attempt(manifest_path)

    assert status == "fail"
    report_check = next(
        check for check in checks if check["check_id"] == "qualification_reports"
    )
    assert report_check["status"] == "fail"
    assert any(
        "observation" in error
        and (
            "execution_mode" in error
            or "native_execution" in error
            or "schema" in error
        )
        for error in report_check["measured"]["errors"]
    )


def test_attempt_verifier_rejects_fully_rehashed_case_identity_attack(
    tmp_path: Path,
) -> None:
    manifest_path = _run_minimal(tmp_path)
    report_path = manifest_path.parent / "reports/blender_custom_two_zone.json"
    report = load_json(report_path)
    report["report_id"] = "unit_minimal_room_attempt_replicacad_apt_0"
    report["subject"]["room_id"] = "replicacad_apt_0"
    write_json(report_path, report)
    observation_path = (
        manifest_path.parent / "observations/blender_custom_two_zone.json"
    )
    observation = load_json(observation_path)
    observation["case_id"] = "replicacad_apt_0"
    observation["room_key"] = "replicacad_apt_0@wrong_revision"
    _rewrite_manifest(observation_path, observation)
    manifest = load_json(manifest_path)
    manifest["reports"][0]["admission_blockers"] = ["wrong_outer_blocker"]
    _rewrite_manifest(manifest_path, manifest)
    _rehash_attempt_bundle(manifest_path)

    status, checks = verify_room_qualification_attempt(manifest_path)

    assert status == "fail"
    report_check = next(
        check for check in checks if check["check_id"] == "qualification_reports"
    )
    assert report_check["status"] == "fail"
    errors = "\n".join(report_check["measured"]["errors"])
    assert "report_id differs" in errors
    assert "report subject differs" in errors
    assert "observation case_id differs" in errors
    assert "observation room_key differs" in errors
    assert "outer admission_blockers differ" in errors


def test_attempt_verifier_rejects_self_claimed_commit_rebind(tmp_path: Path) -> None:
    manifest_path = _run_minimal(tmp_path)
    manifest = load_json(manifest_path)
    manifest["code_provenance"]["commit"] = "0" * 40
    _rewrite_manifest(manifest_path, manifest)

    status, checks = verify_room_qualification_attempt(manifest_path)

    assert status == "fail"
    commit_check = next(
        check for check in checks if check["check_id"] == "code_provenance_commit"
    )
    report_check = next(
        check for check in checks if check["check_id"] == "qualification_reports"
    )
    assert commit_check["status"] == "fail"
    assert report_check["status"] == "fail"
    assert any(
        "observation code provenance differs" in error
        for error in report_check["measured"]["errors"]
    )


def test_attempt_verifier_rejects_self_claimed_registry_rebind(
    tmp_path: Path,
) -> None:
    manifest_path = _run_minimal(tmp_path)
    manifest = load_json(manifest_path)
    manifest["registry"]["sha256"] = "f" * 64
    _rewrite_manifest(manifest_path, manifest)

    status, checks = verify_room_qualification_attempt(manifest_path)

    assert status == "fail"
    report_check = next(
        check for check in checks if check["check_id"] == "qualification_reports"
    )
    assert report_check["status"] == "fail"
    assert any(
        "observation registry SHA-256 differs" in error
        for error in report_check["measured"]["errors"]
    )


def test_formal_attempt_requires_materialized_mp3d_proxy_binding(
    tmp_path: Path,
) -> None:
    manifest_path = _run_minimal(tmp_path)
    manifest = load_json(manifest_path)
    manifest["code_provenance"]["worktree_clean"] = True
    _rewrite_manifest(manifest_path, manifest)

    status, checks = verify_room_qualification_attempt(manifest_path)

    assert status == "fail"
    binding_check = next(
        check
        for check in checks
        if check["check_id"] == "mp3d_materialized_proxy_binding"
    )
    assert binding_check["status"] == "fail"
    assert binding_check["measured"]["exact_binding_pass"] is False
    assert any(
        "requires an exact materialized" in error
        for error in binding_check["measured"]["errors"]
    )


def test_formal_registry_binding_uses_exact_canonical_git_blob(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "formal-repository"
    commit, registry_bytes = _init_git_repository(repository)
    manifest = {
        "code_provenance": {"commit": commit, "worktree_clean": True},
        "registry": {
            "kind": "repository_relative",
            "path": "examples/m6/rooms/room_registry.json",
            "byte_size": len(registry_bytes),
            "sha256": hashlib.sha256(registry_bytes).hexdigest(),
        },
    }
    observation = {"repository": repository, "commit": commit}

    passed, measured = _formal_registry_git_binding(
        manifest, git_observation=observation
    )

    assert passed is True
    assert measured["git_blob_available"] is True
    assert measured["fixed_repository_locator"] is True

    manifest["registry"]["path"] = "examples/m6/rooms/rebound.json"
    assert _formal_registry_git_binding(
        manifest, git_observation=observation
    )[0] is False
    manifest["registry"]["path"] = "examples/m6/rooms/room_registry.json"
    manifest["registry"]["sha256"] = "0" * 64
    assert _formal_registry_git_binding(
        manifest, git_observation=observation
    )[0] is False
    manifest["registry"]["sha256"] = hashlib.sha256(registry_bytes).hexdigest()
    manifest["registry"]["byte_size"] += 1
    assert _formal_registry_git_binding(
        manifest, git_observation=observation
    )[0] is False


def test_git_provenance_requires_lowercase_existing_ancestor_commit(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "provenance-repository"
    commit, _ = _init_git_repository(repository)
    bundle_root = repository / "tmp/attempt"
    bundle_root.mkdir(parents=True)
    manifest = {
        "code_provenance": {"commit": commit, "worktree_clean": True}
    }

    observed = _git_provenance_observation(manifest, bundle_root=bundle_root)

    assert observed["commit_format_valid"] is True
    assert observed["commit_exists"] is True
    assert observed["commit_is_ancestor_of_head"] is True
    assert observed["current_worktree_clean"] is True

    manifest["code_provenance"]["commit"] = commit.upper()
    uppercase = _git_provenance_observation(manifest, bundle_root=bundle_root)
    assert uppercase["commit_format_valid"] is False

    _git(repository, "switch", "-q", "--orphan", "unrelated")
    _git(repository, "commit", "-q", "--allow-empty", "-m", "unrelated root")
    unrelated = _git(repository, "rev-parse", "HEAD")
    _git(repository, "switch", "-q", "main")
    manifest["code_provenance"]["commit"] = unrelated
    nonancestor = _git_provenance_observation(manifest, bundle_root=bundle_root)
    assert nonancestor["commit_exists"] is True
    assert nonancestor["commit_is_ancestor_of_head"] is False


def test_attempt_verifier_rejects_symlinked_artifact_component(
    tmp_path: Path,
) -> None:
    manifest_path = _run_minimal(tmp_path)
    reports = manifest_path.parent / "reports"
    external_reports = tmp_path / "external-reports"
    shutil.move(str(reports), external_reports)
    reports.symlink_to(external_reports, target_is_directory=True)

    status, checks = verify_room_qualification_attempt(manifest_path)

    assert status == "fail"
    artifact_check = next(
        check for check in checks if check["check_id"] == "artifact_hashes"
    )
    assert artifact_check["status"] == "fail"
    assert any("symlink component" in error for error in artifact_check["measured"])


def test_attempt_verifier_rejects_symlinked_manifest_component(
    tmp_path: Path,
) -> None:
    manifest_path = _run_minimal(tmp_path)
    alias = tmp_path / "attempt-alias"
    alias.symlink_to(manifest_path.parent, target_is_directory=True)

    status, checks = verify_room_qualification_attempt(
        alias / "attempt_manifest.json"
    )

    assert status == "fail"
    assert checks[0]["check_id"] == "bundle_path_no_symlinks"


def test_nested_attempt_manifest_is_not_excluded_from_exact_closure(
    tmp_path: Path,
) -> None:
    manifest_path = _run_minimal(tmp_path)
    nested = manifest_path.parent / "nested/attempt_manifest.json"
    nested.parent.mkdir()
    nested.write_text("{}\n", encoding="utf-8")

    status, checks = verify_room_qualification_attempt(manifest_path)

    assert status == "fail"
    artifact_check = next(
        check for check in checks if check["check_id"] == "artifact_hashes"
    )
    assert artifact_check["status"] == "fail"
    assert "nested/attempt_manifest.json" in " ".join(artifact_check["measured"])


def test_attempt_output_is_immutable_no_clobber(tmp_path: Path) -> None:
    _run_minimal(tmp_path)

    try:
        _run_minimal(tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("room attempt must not replace an existing bundle")


def test_declared_derivation_is_not_misclassified_by_raw_byte_parity() -> None:
    raw_geometry = {
        "source_geometry_sha256": "a" * 64,
        "source_to_canonical": {"matrix_row_major": list(range(16))},
        "bounds_m": {"min": [0, 0, 0], "max": [1, 1, 1]},
        "array_hashes": {"vertices": "b" * 64, "triangles": "c" * 64},
    }
    derived_geometry = {
        **raw_geometry,
        "array_hashes": {"vertices": "d" * 64, "triangles": "e" * 64},
        "research_cleanup": {
            "source_arrays": {"vertices": "b" * 64, "triangles": "c" * 64},
            "derived_arrays": {"vertices": "d" * 64, "triangles": "e" * 64},
            "removed_triangle_count": 2,
            "removed_vertex_count": 1,
            "removed_triangle_area_max_m2": 0,
            "qualification_claim": False,
        },
    }
    raw = SimpleNamespace(
        triangle_count=10,
        vertex_count=8,
        qa_reports={"geometry_report": raw_geometry},
    )
    derived = SimpleNamespace(
        triangle_count=8,
        vertex_count=7,
        qa_reports={
            "geometry_report": derived_geometry,
            "compiler_source_to_package_parity": {"status": "fail"},
        },
    )

    result = _declared_derivation_assessment(raw, derived)

    assert result["status"] == "pass"
    assert result["legacy_byte_parity_status"] == "fail"
    assert all(result["checks"].values())


def test_materialized_proxy_descriptor_binds_exact_package_closure(
    tmp_path: Path,
) -> None:
    raw_manifest = tmp_path / "raw/manifest.json"
    write_json(raw_manifest, {"schema": "raw_fixture"})
    package_root = tmp_path / "proxy"
    manifest = package_root / "manifest.json"
    cleanup = {
        "policy": "m3_research_remove_geometry_qa_degenerate_triangles_v1",
        "record_content_sha256": "1" * 64,
        "removed_triangle_count": 1,
        "removed_triangle_indices_sha256": "2" * 64,
        "removed_triangle_area_max_m2": 0.0,
        "source_triangle_count": 3,
        "derived_triangle_count": 2,
        "source_arrays": {
            "vertices": "3" * 64,
            "triangles": "4" * 64,
            "triangle_material_ids": "5" * 64,
        },
        "derived_arrays": {
            "vertices": "6" * 64,
            "triangles": "7" * 64,
            "triangle_material_ids": "8" * 64,
        },
    }
    write_json(
        manifest,
        {
            "package_id": "proxy_fixture",
            "package_content_sha256": "9" * 64,
            "materials": {"material_semantics": "research_placeholder"},
            "source_room": {
                "room_id": "habitat_mp3d_example_17DRP5sb8fy",
                "geometry_asset_sha256": "a" * 64,
            },
        },
    )
    write_json(
        package_root / "qa/geometry_report.json",
        {"research_cleanup": cleanup},
    )
    records = [
        file_record(path, relative_to=package_root)
        for path in sorted(path for path in package_root.rglob("*") if path.is_file())
    ]
    descriptor = {
        "schema": "avengine_m6_derived_acoustic_proxy_v1",
        "proxy_id": "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
        "revision": "fixture_v2",
        "room_id": "habitat_mp3d_example_17DRP5sb8fy",
        "representation_id": "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
        "artifact_uri": "artifact://fixture/proxy",
        "source": {
            "resource_id": "mp3d_raw_visual_surface",
            "representation_id": "mp3d_17DRP5sb8fy_raw_source_v1",
            "sha256": "a" * 64,
            "raw_package_manifest_sha256": sha256_file(raw_manifest),
        },
        "derivation": {
            "producer": "fixture:derive",
            **{
                key: cleanup[key]
                for key in (
                    "policy",
                    "record_content_sha256",
                    "removed_triangle_count",
                    "removed_triangle_indices_sha256",
                    "removed_triangle_area_max_m2",
                    "source_triangle_count",
                    "derived_triangle_count",
                )
            },
            "source_array_hashes": cleanup["source_arrays"],
            "derived_array_hashes": cleanup["derived_arrays"],
        },
        "package": {
            "resource_id": "mp3d_declared_proxy_v2",
            "package_id": "proxy_fixture",
            "package_content_sha256": "9" * 64,
            "manifest": file_record(manifest, relative_to=package_root),
            "artifacts": records,
            "artifact_set_sha256": canonical_json_sha256(records),
        },
        "qualification_claim": False,
        "dataset_admission": False,
    }
    descriptor["descriptor_content_sha256"] = canonical_json_sha256(descriptor)
    descriptor_path = tmp_path / "descriptor.json"
    write_json(descriptor_path, descriptor)
    room_record = {
        "room_id": "habitat_mp3d_example_17DRP5sb8fy",
        "resources": [
            {
                "resource_id": "mp3d_proxy_v2_descriptor",
                "resource_type": "derived_proxy_descriptor",
            }
        ],
        "acoustic_representations": [
            {
                "representation_id": "mp3d_17DRP5sb8fy_raw_source_v1",
                "role": "raw_source",
                "resource_id": "mp3d_raw_visual_surface",
            },
            {
                "representation_id": "mp3d_17DRP5sb8fy_acoustic_proxy_v2",
                "role": "derived_proxy",
                "resource_id": "mp3d_declared_proxy_v2",
                "producer": "fixture:derive",
                "derived_from": "mp3d_17DRP5sb8fy_raw_source_v1",
                "input_resource_ids": [
                    "mp3d_raw_visual_surface",
                    "mp3d_proxy_v2_descriptor",
                ],
                "material_semantics": "research_placeholder",
            },
        ],
    }

    result = _proxy_descriptor_assessment(
        descriptor_path,
        manifest,
        raw_manifest,
        provider_manifest_path=manifest,
        provider_manifest_status="pass",
        provider_representation_status="pass",
        room_record=room_record,
    )
    assert result["status"] == "pass", result
    assert result["artifact_count"] == 2
    assert result["same_materialized_manifest"] is True

    other_manifest = tmp_path / "other-proxy/manifest.json"
    write_json(other_manifest, load_json(manifest))
    split_root = _proxy_descriptor_assessment(
        descriptor_path,
        manifest,
        raw_manifest,
        provider_manifest_path=other_manifest,
        provider_manifest_status="pass",
        provider_representation_status="pass",
        room_record=room_record,
    )
    assert split_root["status"] == "fail"
    assert any("different manifests" in error for error in split_root["errors"])

    wrong_contract = deepcopy(room_record)
    wrong_contract["acoustic_representations"][1]["producer"] = "wrong:producer"
    contract_result = _proxy_descriptor_assessment(
        descriptor_path,
        manifest,
        raw_manifest,
        provider_manifest_path=manifest,
        provider_manifest_status="pass",
        provider_representation_status="pass",
        room_record=wrong_contract,
    )
    assert contract_result["status"] == "fail"
    assert any("producer differs" in error for error in contract_result["errors"])

    (package_root / "qa/geometry_report.json").write_text(
        "{}\n", encoding="utf-8"
    )
    tampered = _proxy_descriptor_assessment(
        descriptor_path,
        manifest,
        raw_manifest,
    )
    assert tampered["status"] == "fail"
    assert any("closure differs" in error for error in tampered["errors"])


def test_proxy_resolution_failure_is_not_downgraded_to_not_run() -> None:
    result = _proxy_descriptor_assessment(
        None,
        None,
        None,
        descriptor_resolution_status="fail",
        provider_manifest_status="fail",
        provider_representation_status="fail",
    )

    assert result["status"] == "fail"
