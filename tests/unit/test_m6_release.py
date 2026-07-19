from __future__ import annotations

import base64
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from avengine.release import (
    ReleaseManifestError,
    build_file_record,
    canonical_file_record_set_sha256,
    load_json_strict,
    require_verified_release_manifest,
    validate_release_attestation_document,
    validate_test_execution_receipt_document,
    verify_release_attestation,
    verify_release_manifest,
    write_release_attestation,
)
from tools.release.build_manifest import main as release_tool_main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RELEASE_SCHEMA = (
    REPOSITORY_ROOT / "schemas" / "avengine_release_manifest_v1.schema.json"
)
AVENGINE_URL = "https://github.com/Eastforward/AVEngine.git"
HABITAT_URL = "https://github.com/Eastforward/habitat-sim-AVEngine.git"
UPSTREAM_URL = "https://github.com/facebookresearch/habitat-sim.git"
RLR_URL = "https://github.com/facebookresearch/rlr-audio-propagation.git"


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repository(path: Path, *, origin: str) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "AVEngine release fixture")
    _git(path, "config", "user.email", "release-fixture@example.invalid")
    _git(path, "remote", "add", "origin", origin)


def _commit_all(path: Path, message: str) -> str:
    _git(path, "add", "--all")
    _git(path, "commit", "--quiet", "-m", message)
    return _git(path, "rev-parse", "HEAD")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _fixture_receipt_execution(declared_path: str) -> dict[str, Any]:
    junit = b'<testsuite name="fixture"><testcase name="passes"/></testsuite>'
    return {
        "execution_cwd": ".",
        "captured_output": {
            "encoding": "base64",
            "stdout_base64": base64.b64encode(b"fixture pass\n").decode("ascii"),
            "stderr_base64": "",
        },
        "junit_xml": {
            "declared_path": declared_path,
            "encoding": "base64",
            "raw_bytes_base64": base64.b64encode(junit).decode("ascii"),
        },
        "result_totals": {
            "executed": 1,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        },
    }


@dataclass(frozen=True)
class ReleaseFixture:
    avengine: Path
    habitat: Path
    manifest_path: Path
    release_tag: str
    implementation_commit: str
    metadata_commit: str
    upstream_commit: str
    habitat_commit: str
    rlr_commit: str


def _make_release_fixture(tmp_path: Path) -> ReleaseFixture:
    rlr_source = tmp_path / "rlr-source"
    _init_repository(rlr_source, origin=RLR_URL)
    (rlr_source / "README.md").write_text("fixture RLR\n", encoding="utf-8")
    rlr_commit = _commit_all(rlr_source, "fixture RLR")

    habitat = tmp_path / "habitat-runtime"
    _init_repository(habitat, origin=HABITAT_URL)
    _git(habitat, "remote", "add", "upstream", UPSTREAM_URL)
    (habitat / "README.md").write_text("upstream fixture\n", encoding="utf-8")
    upstream_commit = _commit_all(habitat, "upstream baseline")
    _git(
        habitat,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        str(rlr_source),
        "src/deps/rlr-audio-propagation",
    )
    rlr_checkout = habitat / "src/deps/rlr-audio-propagation"
    _git(rlr_checkout, "remote", "set-url", "origin", RLR_URL)
    (habitat / "build").mkdir()
    habitat_binding = habitat / "build" / "habitat_sim_fixture.so"
    rlr_binary = habitat / "build" / "librlr_audio_fixture.so"
    habitat_binding.write_bytes(b"habitat native binding fixture\n")
    rlr_binary.write_bytes(b"RLR native binary fixture\n")
    habitat_commit = _commit_all(habitat, "fork implementation")

    avengine = tmp_path / "avengine"
    _init_repository(avengine, origin=AVENGINE_URL)
    (avengine / ".gitignore").write_text("tmp/\n", encoding="utf-8")
    release_tool = avengine / "tools" / "release" / "build_manifest.py"
    release_tool.parent.mkdir(parents=True)
    release_tool.write_text("# fixture release verifier\n", encoding="utf-8")
    schemas = avengine / "schemas"
    schemas.mkdir()
    _write_json(
        schemas / "fixture_v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://avengine.local/schema/fixture_v1.schema.json",
            "type": "object",
        },
    )
    implementation_commit = _commit_all(avengine, "AVEngine implementation")
    controlled_root = avengine / "tmp" / "formal" / "controlled"
    release_reference = controlled_root / "release_manifest_ref.json"
    _write_json(
        release_reference,
        {
            "schema": "avengine_release_manifest_ref_v1",
            "release_id": "avengine-m6-fixture",
            "expected_tag": "v0.1.0-m6-fixture",
            "repository_path": "release/avengine_release_manifest_v1.json",
            "implementation_commit": implementation_commit,
        },
    )
    controlled_entry = controlled_root / "evidence.json"
    _write_json(
        controlled_entry,
        {
            "schema": "avengine_m6_canary_evidence_v1",
            "implementation_commit": implementation_commit,
            "overall_status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "release_manifest_ref": build_file_record(
                release_reference,
                root=controlled_root,
                root_id="avengine",
            )
            | {"path": "release_manifest_ref.json"},
        },
    )
    room_entry = avengine / "tmp" / "formal" / "room" / "attempt_manifest.json"
    _write_json(
        room_entry,
        {
            "schema": "avengine_m6_room_qualification_attempt_v1",
            "code_provenance": {
                "commit": implementation_commit,
                "worktree_clean": True,
            },
            "reports": [{"case": index} for index in range(6)],
            "case_ids": [
                "blender_custom_two_zone",
                "replicacad_apt_0",
                "legacy_ue_apartment",
                "mp3d_17DRP5sb8fy_raw",
                "mp3d_17DRP5sb8fy_derived",
                "independent_corrupted_fixture",
            ],
        },
    )
    junit_path = "tmp/formal/fast-unit.junit.xml"
    layer_command = [
        "python",
        "-m",
        "pytest",
        "tests/unit",
        "--junitxml",
        junit_path,
    ]
    evidence = avengine / "tmp" / "formal" / "fast-unit.json"
    _write_json(
        evidence,
        {
            "schema": "avengine_m6_test_execution_receipt_v1",
            "receipt_id": "fast-unit-fixture-receipt",
            "test_layer_id": "fast-unit",
            "status": "pass",
            "command": layer_command,
            "exit_code": 0,
            "implementation_commit": implementation_commit,
            "habitat_runtime_commit": habitat_commit,
            "rlr_commit": rlr_commit,
            **_fixture_receipt_execution(junit_path),
        },
    )

    schema_records = [
        build_file_record(path, root=avengine, root_id="avengine")
        for path in sorted(schemas.rglob("*.json"))
    ]
    evidence_records = [
        build_file_record(evidence, root=avengine, root_id="avengine")
    ]
    controlled_record = build_file_record(
        controlled_entry, root=avengine, root_id="avengine"
    )
    room_record = build_file_record(room_entry, root=avengine, root_id="avengine")
    evidence_id = "fast-unit-fixture"
    release_tag = "v0.1.0-m6-fixture"
    manifest: dict[str, Any] = {
        "schema": "avengine_release_manifest_v1",
        "release": {
            "release_id": "avengine-m6-fixture",
            "tag": release_tag,
            "state": "candidate",
            "current_milestone": "M6",
            "manifest_path": "release/avengine_release_manifest_v1.json",
            "metadata_commit_policy": {
                "mode": "direct_child_of_implementation",
                "allowed_changed_paths": [
                    "release/avengine_release_manifest_v1.json"
                ],
                "require_clean_worktrees": True,
                "require_annotated_tag": True,
            },
        },
        "repositories": {
            "avengine": {
                "repository": AVENGINE_URL,
                "implementation_commit": implementation_commit,
            },
            "habitat_runtime": {
                "repository": HABITAT_URL,
                "commit": habitat_commit,
                "upstream_repository": UPSTREAM_URL,
                "upstream_commit": upstream_commit,
                "rlr_repository": RLR_URL,
                "rlr_commit": rlr_commit,
                "rlr_submodule_path": "src/deps/rlr-audio-propagation",
            },
        },
        "schemas": {
            "directory": "schemas",
            "algorithm": "sha256_canonical_file_records_v1",
            "files": schema_records,
            "set_sha256": canonical_file_record_set_sha256(schema_records),
        },
        "native_artifacts": {
            "habitat_sim_binding": build_file_record(
                habitat_binding,
                root=habitat,
                root_id="habitat_runtime",
            ),
            "rlr_binary": build_file_record(
                rlr_binary,
                root=habitat,
                root_id="habitat_runtime",
            ),
        },
        "environment": {
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "compiler": {
                "id": "fixture-version-command",
                "command": sys.executable,
                "version": platform.python_version(),
            },
            "python_dependencies": [
                {
                    "distribution": "jsonschema",
                    "version": importlib_metadata.version("jsonschema"),
                }
            ],
        },
        "evidence_bundles": [
            {
                "evidence_id": evidence_id,
                "status_scope": "test_execution",
                "status": "pass",
                "artifacts": evidence_records,
                "bundle_sha256": canonical_file_record_set_sha256(evidence_records),
            },
            {
                "evidence_id": "m6-controlled-fixture",
                "status_scope": "controlled_canary_verifier",
                "status": "pass",
                "artifacts": [controlled_record],
                "bundle_sha256": canonical_file_record_set_sha256(
                    [controlled_record]
                ),
            },
            {
                "evidence_id": "m6-room-fixture",
                "status_scope": "room_attempt_verifier",
                "status": "pass",
                "artifacts": [room_record],
                "bundle_sha256": canonical_file_record_set_sha256([room_record]),
            },
        ],
        "m6_evidence": {
            "controlled_canary_bundle_id": "m6-controlled-fixture",
            "controlled_canary_entry": controlled_record,
            "room_qualification_bundle_id": "m6-room-fixture",
            "room_qualification_entry": room_record,
        },
        "test_layers": {
            "fast-unit": {
                "status": "pass",
                "command": layer_command,
                "evidence_bundle_ids": [evidence_id],
                "receipt_artifacts": evidence_records,
                "summary": "Fixture fast unit evidence.",
            },
            **{
                layer: {
                    "status": "not_run",
                    "command": [],
                    "evidence_bundle_ids": [],
                    "receipt_artifacts": [],
                    "reason": "Not exercised by the hermetic release fixture.",
                }
                for layer in (
                    "slow-hermetic",
                    "native-habitat",
                    "rlr-audio",
                    "blender-assets",
                    "media-readback",
                    "release-canary",
                )
            },
        },
    }
    manifest["test_layers"]["release-canary"] = {
        "status": "not_run",
        "command": [
            "python",
            "tools/release/build_manifest.py",
            "verify",
            "--manifest",
            "release/avengine_release_manifest_v1.json",
            "--avengine-root",
            str(avengine),
            "--habitat-runtime-root",
            str(habitat),
            "--output",
            "tmp/m6/release_attestation.json",
        ],
        "evidence_bundle_ids": [
            evidence_id,
            "m6-controlled-fixture",
            "m6-room-fixture",
        ],
        "receipt_artifacts": [],
        "reason": "Post-tag final attestation is performed by the live verifier.",
    }
    manifest_path = avengine / manifest["release"]["manifest_path"]
    _write_json(manifest_path, manifest)
    metadata_commit = _commit_all(avengine, "M6 release metadata")
    _git(avengine, "tag", "-a", release_tag, "-m", "M6 release fixture")

    return ReleaseFixture(
        avengine=avengine,
        habitat=habitat,
        manifest_path=manifest_path,
        release_tag=release_tag,
        implementation_commit=implementation_commit,
        metadata_commit=metadata_commit,
        upstream_commit=upstream_commit,
        habitat_commit=habitat_commit,
        rlr_commit=rlr_commit,
    )


def _check_errors(report: dict[str, Any], check_id: str) -> list[str]:
    check = next(item for item in report["checks"] if item["check_id"] == check_id)
    return check["errors"]


def _verify(fixture: ReleaseFixture, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "avengine_root": fixture.avengine,
        "habitat_runtime_root": fixture.habitat,
        "schema_path": RELEASE_SCHEMA,
        "verify_m6_evidence": False,
    }
    arguments.update(overrides)
    return verify_release_manifest(fixture.manifest_path, **arguments)


def _rewrite_receipt_and_manifest(
    fixture: ReleaseFixture, receipt: dict[str, Any]
) -> None:
    receipt_path = fixture.avengine / "tmp" / "formal" / "fast-unit.json"
    _write_json(receipt_path, receipt)
    manifest = load_json_strict(fixture.manifest_path)
    record = build_file_record(
        receipt_path, root=fixture.avengine, root_id="avengine"
    )
    bundle = next(
        item
        for item in manifest["evidence_bundles"]
        if item["evidence_id"] == "fast-unit-fixture"
    )
    bundle["artifacts"] = [record]
    bundle["bundle_sha256"] = canonical_file_record_set_sha256([record])
    manifest["test_layers"]["fast-unit"]["receipt_artifacts"] = [record]
    _write_json(fixture.manifest_path, manifest)


def test_release_schema_is_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(load_json_strict(RELEASE_SCHEMA))


def test_receipt_cli_writes_schema_valid_no_clobber_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _make_release_fixture(tmp_path)
    output = fixture.avengine / "tmp" / "receipts" / "fast-unit.json"
    junit_path = "tmp/receipts/fast-unit.junit.xml"
    junit = b'<testsuite><testcase name="pass"/></testsuite>'
    script = (
        "from pathlib import Path; import sys; "
        f"Path(sys.argv[1]).write_bytes({junit!r}); "
        "print('receipt pass'); raise SystemExit(0)"
    )
    command = [sys.executable, "-c", script, junit_path]
    arguments = [
        "receipt",
        "--output",
        str(output),
        "--workspace-root",
        str(fixture.avengine),
        "--habitat-runtime-root",
        str(fixture.habitat),
        "--receipt-id",
        "fast-unit-test-receipt",
        "--layer-id",
        "fast-unit",
        "--junit-xml",
        junit_path,
        "--",
        *command,
    ]
    assert release_tool_main(arguments) == 0
    capsys.readouterr()
    receipt = load_json_strict(output)
    assert receipt["command"] == command
    assert receipt["status"] == "pass"
    assert validate_test_execution_receipt_document(receipt) == []
    assert not (fixture.avengine / junit_path).exists()
    assert release_tool_main(arguments) == 2
    assert "refusing to replace" in capsys.readouterr().out

    failed_output = fixture.avengine / "tmp" / "receipts" / "rlr-audio-fail.json"
    failed_junit_path = "tmp/receipts/rlr-audio-fail.junit.xml"
    failed_junit = (
        b'<testsuite><testcase name="fail"><failure/></testcase></testsuite>'
    )
    failed_script = (
        "from pathlib import Path; import sys; "
        f"Path(sys.argv[1]).write_bytes({failed_junit!r}); "
        "raise SystemExit(1)"
    )
    failed_command = [sys.executable, "-c", failed_script, failed_junit_path]
    failed_arguments = [
        "receipt",
        "--output",
        str(failed_output),
        "--workspace-root",
        str(fixture.avengine),
        "--habitat-runtime-root",
        str(fixture.habitat),
        "--receipt-id",
        "rlr-audio-failed-receipt",
        "--layer-id",
        "rlr-audio",
        "--junit-xml",
        failed_junit_path,
        "--",
        *failed_command,
    ]
    assert release_tool_main(failed_arguments) == 1
    capsys.readouterr()
    failed_receipt = load_json_strict(failed_output)
    assert failed_receipt["status"] == "fail"
    assert failed_receipt["exit_code"] == 1
    assert validate_test_execution_receipt_document(failed_receipt) == []


def test_release_manifest_verifies_hashes_git_tag_and_environment(
    tmp_path: Path,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    report = _verify(fixture)
    assert report["status"] == "pass", report
    assert report["observed"]["avengine_metadata_commit"] == fixture.metadata_commit
    assert report["observed"]["avengine_metadata_parent"] == (
        fixture.implementation_commit
    )
    assert report["observed"]["habitat_runtime_commit"] == fixture.habitat_commit
    assert report["observed"]["rlr_commit"] == fixture.rlr_commit


def test_release_verifier_runs_m6_semantic_roles_by_default(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    report = verify_release_manifest(
        fixture.manifest_path,
        avengine_root=fixture.avengine,
        habitat_runtime_root=fixture.habitat,
        schema_path=RELEASE_SCHEMA,
        verify_git=False,
        verify_environment=False,
    )
    assert report["status"] == "fail"
    assert any(
        "authoritative verifier failed checks" in error
        for error in _check_errors(report, "m6_evidence")
    )


def test_release_m6_role_entry_must_belong_to_named_bundle(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["m6_evidence"]["controlled_canary_entry"] = manifest[
        "m6_evidence"
    ]["room_qualification_entry"]
    _write_json(fixture.manifest_path, manifest)
    report = _verify(
        fixture,
        verify_git=False,
        verify_environment=False,
    )
    assert report["status"] == "fail"
    assert any(
        "controlled canary entry is not in its exact evidence closure" in error
        for error in _check_errors(report, "m6_evidence")
    )


def test_release_manifest_detects_tampered_evidence(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    (fixture.avengine / "tmp" / "formal" / "fast-unit.json").write_text(
        '{"status":"tampered"}\n', encoding="utf-8"
    )
    report = _verify(fixture, verify_git=False, verify_environment=False)
    assert report["status"] == "fail"
    errors = _check_errors(report, "evidence_bundles")
    assert any("SHA-256 mismatch" in error for error in errors)


def test_release_manifest_rejects_receipt_outside_referenced_bundle(
    tmp_path: Path,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["test_layers"]["fast-unit"]["receipt_artifacts"] = [
        manifest["m6_evidence"]["controlled_canary_entry"]
    ]
    _write_json(fixture.manifest_path, manifest)
    report = _verify(fixture, verify_git=False, verify_environment=False)
    assert report["status"] == "fail"
    assert any(
        "not a member of a referenced evidence bundle" in error
        for error in _check_errors(report, "test_layers")
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("test_layer_id", "slow-hermetic", "test_layer_id mismatch"),
        (
            "command",
            ["python", "wrong-command"],
            "does not reference junit_xml.declared_path",
        ),
        ("status", "fail", "status differs from exit_code"),
        ("implementation_commit", "0" * 40, "implementation_commit mismatch"),
        ("habitat_runtime_commit", "1" * 40, "habitat_runtime_commit mismatch"),
        ("rlr_commit", "2" * 40, "rlr_commit mismatch"),
    ],
)
def test_release_manifest_rejects_semantically_mismatched_receipt(
    tmp_path: Path, field: str, value: Any, expected: str
) -> None:
    fixture = _make_release_fixture(tmp_path)
    receipt_path = fixture.avengine / "tmp" / "formal" / "fast-unit.json"
    receipt = load_json_strict(receipt_path)
    receipt[field] = value
    _rewrite_receipt_and_manifest(fixture, receipt)
    report = _verify(fixture, verify_git=False, verify_environment=False)
    assert report["status"] == "fail"
    assert any(expected in error for error in _check_errors(report, "test_layers"))


def test_release_manifest_rejects_receipt_totals_not_derived_from_junit(
    tmp_path: Path,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    receipt_path = fixture.avengine / "tmp" / "formal" / "fast-unit.json"
    receipt = load_json_strict(receipt_path)
    receipt["result_totals"]["executed"] = 2
    _rewrite_receipt_and_manifest(fixture, receipt)
    report = _verify(fixture, verify_git=False, verify_environment=False)
    errors = _check_errors(report, "test_layers")
    assert report["status"] == "fail"
    assert any("result_totals" in error for error in errors)


def test_release_manifest_rejects_receipt_status_not_derived_from_exit_code(
    tmp_path: Path,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    receipt_path = fixture.avengine / "tmp" / "formal" / "fast-unit.json"
    receipt = load_json_strict(receipt_path)
    receipt["exit_code"] = 7
    _rewrite_receipt_and_manifest(fixture, receipt)
    report = _verify(fixture, verify_git=False, verify_environment=False)
    errors = _check_errors(report, "test_layers")
    assert report["status"] == "fail"
    assert any("status differs from exit_code" in error for error in errors)


def test_post_tag_attestation_binds_manifest_tag_and_full_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    monkeypatch.setattr(
        "avengine.release._require_current_m6_evidence_verifiers",
        lambda **kwargs: None,
    )
    manifest = load_json_strict(fixture.manifest_path)
    command = manifest["test_layers"]["release-canary"]["command"]
    output = fixture.avengine / "tmp" / "m6" / "release_attestation.json"
    published, attestation = write_release_attestation(
        output,
        manifest_path=fixture.manifest_path,
        avengine_root=fixture.avengine,
        habitat_runtime_root=fixture.habitat,
        verification_command=command,
    )
    assert published == output
    assert attestation["status"] == "pass"
    assert attestation["release_tag_commit"] == fixture.metadata_commit
    assert attestation["implementation_commit"] == fixture.implementation_commit
    assert attestation["manifest"] == build_file_record(
        fixture.manifest_path, root=fixture.avengine, root_id="avengine"
    )
    assert attestation["verification_report"]["status"] == "pass"
    assert [
        check["check_id"] for check in attestation["verification_report"]["checks"]
    ] == [
        "manifest_json",
        "manifest_schema",
        "schema_set",
        "native_artifacts",
        "evidence_bundles",
        "m6_evidence",
        "test_layers",
        "environment",
        "git_identity",
    ]
    assert attestation["verification_command"] == command
    assert validate_release_attestation_document(attestation) == []
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_release_attestation(
            output,
            manifest_path=fixture.manifest_path,
            avengine_root=fixture.avengine,
            habitat_runtime_root=fixture.habitat,
            verification_command=command,
        )


def test_post_tag_attestation_rejects_current_manifest_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    monkeypatch.setattr(
        "avengine.release._require_current_m6_evidence_verifiers",
        lambda **kwargs: None,
    )
    manifest = load_json_strict(fixture.manifest_path)
    command = manifest["test_layers"]["release-canary"]["command"]
    manifest["evidence_bundles"][0]["bundle_sha256"] = "0" * 64
    _write_json(fixture.manifest_path, manifest)
    output = fixture.avengine / "tmp" / "m6" / "release_attestation.json"
    with pytest.raises(ReleaseManifestError, match="must pass"):
        write_release_attestation(
            output,
            manifest_path=fixture.manifest_path,
            avengine_root=fixture.avengine,
            habitat_runtime_root=fixture.habitat,
            verification_command=command,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "command",
    [
        ["echo", "verify"],
        ["python", "definitely_not_release_tool.py", "verify"],
        [
            "python",
            "tools/release/build_manifest.py",
            "verify",
            "--manifest=release/avengine_release_manifest_v1.json",
        ],
    ],
)
def test_post_tag_attestation_rejects_noncanonical_actual_command(
    tmp_path: Path, command: list[str]
) -> None:
    fixture = _make_release_fixture(tmp_path)
    output = fixture.avengine / "tmp" / "m6" / "release_attestation.json"
    with pytest.raises(ReleaseManifestError, match="release-canary|invocation|command"):
        write_release_attestation(
            output,
            manifest_path=fixture.manifest_path,
            avengine_root=fixture.avengine,
            habitat_runtime_root=fixture.habitat,
            verification_command=command,
        )
    assert not output.exists()


def test_post_tag_attestation_rejects_lightweight_or_moved_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_release_fixture(tmp_path)
    monkeypatch.setattr(
        "avengine.release._require_current_m6_evidence_verifiers",
        lambda **kwargs: None,
    )
    manifest = load_json_strict(fixture.manifest_path)
    command = manifest["test_layers"]["release-canary"]["command"]
    _git(fixture.avengine, "tag", "-d", fixture.release_tag)
    _git(
        fixture.avengine,
        "tag",
        fixture.release_tag,
        fixture.implementation_commit,
    )
    output = fixture.avengine / "tmp" / "m6" / "release_attestation.json"
    with pytest.raises(ReleaseManifestError, match="must pass"):
        write_release_attestation(
            output,
            manifest_path=fixture.manifest_path,
            avengine_root=fixture.avengine,
            habitat_runtime_root=fixture.habitat,
            verification_command=command,
        )
    assert not output.exists()


def test_release_attestation_readback_reruns_live_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_release_fixture(tmp_path)
    monkeypatch.setattr(
        "avengine.release._require_current_m6_evidence_verifiers",
        lambda **kwargs: None,
    )
    manifest = load_json_strict(fixture.manifest_path)
    command = manifest["test_layers"]["release-canary"]["command"]
    output = fixture.avengine / "tmp" / "m6" / "release_attestation.json"
    write_release_attestation(
        output,
        manifest_path=fixture.manifest_path,
        avengine_root=fixture.avengine,
        habitat_runtime_root=fixture.habitat,
        verification_command=command,
    )
    report = verify_release_attestation(
        output,
        avengine_root=fixture.avengine,
        habitat_runtime_root=fixture.habitat,
    )
    assert report["status"] == "pass", report

    _git(fixture.avengine, "tag", "-d", fixture.release_tag)
    _git(
        fixture.avengine,
        "tag",
        fixture.release_tag,
        fixture.implementation_commit,
    )
    stale = verify_release_attestation(
        output,
        avengine_root=fixture.avengine,
        habitat_runtime_root=fixture.habitat,
    )
    assert stale["status"] == "fail"
    assert any(
        "fresh full verification" in error or "must pass" in error
        for check in stale["checks"]
        for error in check["errors"]
    )


def test_release_manifest_rejects_released_state(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["release"]["state"] = "released"
    _write_json(fixture.manifest_path, manifest)
    report = _verify(fixture, verify_git=False, verify_environment=False)
    assert report["status"] == "fail"
    assert any(
        "'candidate' was expected" in error
        for error in _check_errors(report, "manifest_schema")
    )


def test_release_manifest_detects_schema_set_hash_mismatch(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["schemas"]["set_sha256"] = "0" * 64
    _write_json(fixture.manifest_path, manifest)
    report = _verify(fixture, verify_git=False, verify_environment=False)
    assert report["status"] == "fail"
    assert "schema set SHA-256 mismatch" in _check_errors(report, "schema_set")


def test_release_manifest_detects_runtime_commit_drift(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    (fixture.habitat / "runtime-drift.txt").write_text("drift\n", encoding="utf-8")
    _commit_all(fixture.habitat, "runtime drift")
    report = _verify(fixture, verify_environment=False)
    assert report["status"] == "fail"
    assert any(
        "Habitat runtime HEAD mismatch" in error
        for error in _check_errors(report, "git_identity")
    )


def test_release_manifest_requires_annotated_tag(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    _git(fixture.avengine, "tag", "-d", fixture.release_tag)
    _git(fixture.avengine, "tag", fixture.release_tag)
    report = _verify(fixture, verify_environment=False)
    assert report["status"] == "fail"
    assert any(
        "is not annotated" in error
        for error in _check_errors(report, "git_identity")
    )


def test_release_manifest_rejects_tracked_symlink_to_ignored_manifest(
    tmp_path: Path,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    _git(fixture.avengine, "tag", "-d", fixture.release_tag)
    _git(fixture.avengine, "switch", "--detach", fixture.implementation_commit)

    ignored_target = fixture.avengine / "tmp" / "ignored" / "manifest.json"
    _write_json(ignored_target, manifest)
    fixture.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fixture.manifest_path.symlink_to(Path("../tmp/ignored/manifest.json"))
    _commit_all(fixture.avengine, "release metadata symlink B")
    _git(
        fixture.avengine,
        "tag",
        "-a",
        fixture.release_tag,
        "-m",
        "symlink release fixture",
    )
    assert _git(
        fixture.avengine, "status", "--porcelain", "--untracked-files=all"
    ) == ""

    report = _verify(fixture, verify_environment=False)

    assert report["status"] == "fail"
    assert any(
        "must not be or traverse a symlink" in error
        for error in _check_errors(report, "manifest_json")
    )


def test_release_metadata_commit_rejects_non_release_changes(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    _git(fixture.avengine, "tag", "-d", fixture.release_tag)
    (fixture.avengine / "README.md").write_text("not release metadata\n", encoding="utf-8")
    _commit_all(fixture.avengine, "disallowed metadata change")
    _git(
        fixture.avengine,
        "tag",
        "-a",
        fixture.release_tag,
        "-m",
        "moved fixture tag",
    )
    report = _verify(fixture, verify_environment=False)
    assert report["status"] == "fail"
    assert any(
        "metadata commit changes non-release paths: README.md" in error
        for error in _check_errors(report, "git_identity")
    )


@pytest.mark.parametrize("self_authorized_path", ["README.md", "src/"])
def test_release_manifest_cannot_self_authorize_non_release_metadata_changes(
    tmp_path: Path, self_authorized_path: str
) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["release"]["metadata_commit_policy"]["allowed_changed_paths"].append(
        self_authorized_path
    )
    _write_json(fixture.manifest_path, manifest)

    report = _verify(fixture, verify_git=False, verify_environment=False)

    assert report["status"] == "fail"
    assert any(
        "must remain beneath release/" in error
        for error in _check_errors(report, "manifest_schema")
    )


def test_release_manifest_path_must_remain_release_metadata(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["release"]["manifest_path"] = "README.md"
    manifest["release"]["metadata_commit_policy"]["allowed_changed_paths"] = [
        "README.md"
    ]
    _write_json(fixture.manifest_path, manifest)

    report = _verify(fixture, verify_git=False, verify_environment=False)

    assert report["status"] == "fail"
    assert any(
        "release.manifest_path must remain beneath release/" in error
        for error in _check_errors(report, "manifest_schema")
    )


def test_release_metadata_allowlist_must_include_manifest_path(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["release"]["metadata_commit_policy"]["allowed_changed_paths"] = [
        "release/M6_FINAL_REPORT.md"
    ]
    _write_json(fixture.manifest_path, manifest)

    report = _verify(fixture, verify_git=False, verify_environment=False)

    assert report["status"] == "fail"
    assert any(
        "must allow release.manifest_path" in error
        for error in _check_errors(report, "manifest_schema")
    )


def test_require_verified_release_manifest_raises_structured_error(
    tmp_path: Path,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["test_layers"]["fast-unit"]["evidence_bundle_ids"] = ["missing"]
    _write_json(fixture.manifest_path, manifest)
    with pytest.raises(ReleaseManifestError, match="unknown evidence bundles"):
        require_verified_release_manifest(
            fixture.manifest_path,
            avengine_root=fixture.avengine,
            habitat_runtime_root=fixture.habitat,
            schema_path=RELEASE_SCHEMA,
            verify_git=False,
            verify_environment=False,
            verify_m6_evidence=False,
        )


def test_build_file_record_rejects_root_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(ReleaseManifestError, match="escapes root"):
        build_file_record(outside, root=root, root_id="avengine")


def test_build_file_record_rejects_symlinked_input(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.bin"
    target.write_bytes(b"target")
    linked = root / "linked.bin"
    linked.symlink_to(target.name)
    with pytest.raises(ReleaseManifestError, match="symlink"):
        build_file_record(linked, root=root, root_id="avengine")
