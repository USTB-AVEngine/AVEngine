from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import avengine.release as release_module
import avengine.release_receipt as receipt_module
from avengine.release import (
    ReleaseManifestError,
    build_release_manifest,
    load_release_manifest,
    prepare_release_manifest,
    require_verified_release_manifest,
    validate_release_attestation_document,
    validate_release_manifest_document,
    validate_test_execution_receipt_document,
    verify_release_attestation,
    verify_release_manifest,
    write_release_attestation,
)
from avengine.release_receipt import (
    RELEASE_V1_ARCHIVAL_ERROR,
    TestReceiptError as ReceiptError,
    derive_junit_totals,
    execute_test_receipt,
    verify_receipt_payload,
)
from tools.release.build_manifest import main as release_tool_main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("command", "expected_option"),
    (
        ("receipt", "--junit-xml"),
        ("prepare", "--request"),
        ("verify", "--manifest"),
        ("verify-attestation", "--attestation"),
    ),
)
def test_archived_v1_cli_help_remains(
    command: str,
    expected_option: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        release_tool_main([command, "--help"])

    assert raised.value.code == 0
    assert expected_option in capsys.readouterr().out


def test_archived_v1_cli_fails_before_paths_or_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = tmp_path / "marker.txt"
    receipt = tmp_path / "receipt.json"
    junit = tmp_path / "receipt.junit.xml"
    attestation = tmp_path / "attestation.json"
    marker_script = (
        "from pathlib import Path; "
        f"Path({str(marker)!r}).write_text('ran', encoding='utf-8'); "
        f"Path({str(junit)!r}).write_text('<testsuite/>', encoding='utf-8')"
    )
    cases = (
        [
            "receipt",
            "--output",
            str(receipt),
            "--workspace-root",
            "/definitely/missing/workspace",
            "--habitat-runtime-root",
            "/definitely/missing/habitat",
            "--receipt-id",
            "archive-probe",
            "--layer-id",
            "fast-unit",
            "--junit-xml",
            str(junit),
            "--",
            sys.executable,
            "-c",
            marker_script,
        ],
        [
            "prepare",
            "--request",
            "/definitely/missing/request.json",
            "--avengine-root",
            "/definitely/missing/workspace",
            "--habitat-runtime-root",
            "/definitely/missing/habitat",
        ],
        [
            "verify",
            "--manifest",
            "/definitely/missing/manifest.json",
            "--avengine-root",
            "/definitely/missing/workspace",
            "--habitat-runtime-root",
            "/definitely/missing/habitat",
            "--output",
            str(attestation),
        ],
        [
            "verify-attestation",
            "--attestation",
            "/definitely/missing/attestation.json",
            "--avengine-root",
            "/definitely/missing/workspace",
            "--habitat-runtime-root",
            "/definitely/missing/habitat",
        ],
    )

    for arguments in cases:
        assert release_tool_main(arguments) == 2
        result = json.loads(capsys.readouterr().out)
        assert result == {
            "schema": "avengine_release_tool_error_v1",
            "status": "fail",
            "error": RELEASE_V1_ARCHIVAL_ERROR,
        }

    assert not marker.exists()
    assert not receipt.exists()
    assert not junit.exists()
    assert not attestation.exists()


def test_archived_v1_direct_apis_fail_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("archived v1 attempted Git, subprocess, or output I/O")

    monkeypatch.setattr(release_module.subprocess, "run", forbidden)
    monkeypatch.setattr(release_module, "write_bytes_no_clobber", forbidden)
    monkeypatch.setattr(receipt_module.subprocess, "run", forbidden)
    monkeypatch.setattr(receipt_module, "write_bytes_no_clobber", forbidden)

    missing = tmp_path / "missing"
    receipt_output = tmp_path / "receipt.json"
    attestation_output = tmp_path / "attestation.json"
    calls = (
        lambda: execute_test_receipt(
            receipt_output,
            workspace_root=missing,
            habitat_runtime_root=missing,
            receipt_id="archive-probe",
            test_layer_id="fast-unit",
            junit_xml="tmp/archive-probe.xml",
            command=["marker-command"],
        ),
        lambda: build_release_manifest(
            {},
            avengine_root=missing,
            habitat_runtime_root=missing,
        ),
        lambda: prepare_release_manifest(
            missing / "request.json",
            avengine_root=missing,
            habitat_runtime_root=missing,
        ),
        lambda: verify_release_manifest(
            missing / "manifest.json",
            avengine_root=missing,
            habitat_runtime_root=missing,
        ),
        lambda: write_release_attestation(
            attestation_output,
            manifest_path=missing / "manifest.json",
            avengine_root=missing,
            habitat_runtime_root=missing,
            verification_command=["marker-command"],
        ),
        lambda: verify_release_attestation(
            missing / "attestation.json",
            avengine_root=missing,
            habitat_runtime_root=missing,
        ),
        lambda: require_verified_release_manifest(
            missing / "manifest.json",
            avengine_root=missing,
            habitat_runtime_root=missing,
        ),
    )

    for call in calls:
        with pytest.raises(
            (ReleaseManifestError, ReceiptError),
            match="archived reader-only",
        ) as raised:
            call()
        assert str(raised.value) == RELEASE_V1_ARCHIVAL_ERROR

    assert not receipt_output.exists()
    assert not attestation_output.exists()


def test_archived_v1_document_receipt_and_junit_readers_remain() -> None:
    manifest = load_release_manifest(
        REPOSITORY_ROOT / "release" / "avengine_release_manifest_v1.json"
    )
    assert manifest["schema"] == "avengine_release_manifest_v1"
    assert manifest["release"]["state"] == "candidate"

    junit = b'<testsuite><testcase name="pass"/></testsuite>'
    declared_junit = "tmp/historical/fast-unit.junit.xml"
    receipt = {
        "schema": "avengine_m6_test_execution_receipt_v1",
        "receipt_id": "historical-fast-unit",
        "test_layer_id": "fast-unit",
        "status": "pass",
        "command": ["python", "-m", "pytest", f"--junitxml={declared_junit}"],
        "exit_code": 0,
        "implementation_commit": "1" * 40,
        "habitat_runtime_commit": "2" * 40,
        "rlr_commit": "3" * 40,
        "execution_cwd": ".",
        "captured_output": {
            "encoding": "base64",
            "stdout_base64": "",
            "stderr_base64": "",
        },
        "junit_xml": {
            "declared_path": declared_junit,
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
    assert validate_test_execution_receipt_document(receipt) == []
    assert verify_receipt_payload(receipt) == []
    assert derive_junit_totals(junit) == receipt["result_totals"]

    totals_tamper = deepcopy(receipt)
    totals_tamper["result_totals"]["passed"] = 0
    assert any(
        "result_totals differ" in error
        for error in verify_receipt_payload(totals_tamper)
    )

    status_tamper = deepcopy(receipt)
    status_tamper["status"] = "fail"
    assert any(
        "status differs" in error
        for error in verify_receipt_payload(status_tamper)
    )

    base64_tamper = deepcopy(receipt)
    base64_tamper["captured_output"]["stdout_base64"] = "not base64"
    assert any(
        "base64" in error for error in verify_receipt_payload(base64_tamper)
    )

    junit_tamper = deepcopy(receipt)
    junit_tamper["junit_xml"]["raw_bytes_base64"] = base64.b64encode(
        b'<testsuite><testcase name="fail"><failure/></testcase></testsuite>'
    ).decode("ascii")
    assert any(
        "result_totals differ" in error
        for error in verify_receipt_payload(junit_tamper)
    )


def test_archived_v1_manifest_reader_rejects_metadata_policy_escape() -> None:
    manifest = load_release_manifest(
        REPOSITORY_ROOT / "release" / "avengine_release_manifest_v1.json"
    )

    outside_allowlist = deepcopy(manifest)
    outside_allowlist["release"]["metadata_commit_policy"][
        "allowed_changed_paths"
    ].append("README.md")
    assert any(
        "must remain beneath release/" in error
        for error in validate_release_manifest_document(outside_allowlist)
    )

    outside_manifest = deepcopy(manifest)
    outside_manifest["release"]["manifest_path"] = "README.md"
    outside_manifest["release"]["metadata_commit_policy"][
        "allowed_changed_paths"
    ] = ["README.md"]
    assert any(
        "release.manifest_path must remain beneath release/" in error
        for error in validate_release_manifest_document(outside_manifest)
    )

    incomplete_allowlist = deepcopy(manifest)
    incomplete_allowlist["release"]["metadata_commit_policy"][
        "allowed_changed_paths"
    ] = ["release/M6_FINAL_REPORT.md"]
    assert any(
        "must allow release.manifest_path" in error
        for error in validate_release_manifest_document(incomplete_allowlist)
    )


def test_archived_v1_attestation_schema_reader_remains() -> None:
    file_record = {
        "root_id": "avengine",
        "path": "release/avengine_release_manifest_v1.json",
        "byte_size": 1,
        "sha256": "a" * 64,
    }
    attestation = {
        "schema": "avengine_release_attestation_v1",
        "status": "pass",
        "release_id": "historical-release",
        "release_tag": "historical-release-tag",
        "release_tag_commit": "4" * 40,
        "implementation_commit": "5" * 40,
        "manifest": file_record,
        "verification_command": [
            "python",
            "tools/release/build_manifest.py",
            "verify",
        ],
        "verification_report": {
            "schema": "avengine_release_verification_v1",
            "status": "pass",
            "checks": [
                {"check_id": f"check-{index}", "status": "pass", "errors": []}
                for index in range(9)
            ],
            "observed": {
                "manifest_file_record": file_record,
                "avengine_metadata_commit": "4" * 40,
                "avengine_metadata_parent": "5" * 40,
                "avengine_metadata_parent_count": 1,
                "release_tag_commit": "4" * 40,
                "habitat_runtime_commit": "6" * 40,
                "rlr_commit": "7" * 40,
            },
        },
    }
    assert validate_release_attestation_document(attestation) == []

    invalid = deepcopy(attestation)
    invalid["verification_report"]["status"] = "fail"
    assert validate_release_attestation_document(invalid)
