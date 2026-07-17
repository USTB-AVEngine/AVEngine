from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from avengine.cli import build_parser, main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUEST = REPOSITORY_ROOT / "examples/m4/blender_custom/multi_source_canary_request.json"
DEFAULT_RUNTIME_LOCK = REPOSITORY_ROOT / "locks/m4_runtime_v1.json"


def _output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    value = json.loads(capsys.readouterr().out)
    assert isinstance(value, dict)
    return value


def test_m4_parser_exposes_commands_and_pins_default_runtime_lock() -> None:
    parser = build_parser()

    validate = parser.parse_args(["m4", "validate-request", "request.json"])
    run = parser.parse_args(
        [
            "m4",
            "run-canary",
            "--request",
            "request.json",
            "--package-manifest",
            "manifest.json",
            "--output",
            "/tmp/m4-canary",
        ]
    )
    verify = parser.parse_args(["m4", "verify-canary", "evidence.json"])
    bundle = parser.parse_args(["m4", "verify-bundle", "bundle.json"])

    assert validate.m4_command == "validate-request"
    assert run.m4_command == "run-canary"
    assert Path(run.runtime_lock) == DEFAULT_RUNTIME_LOCK
    assert DEFAULT_RUNTIME_LOCK.is_file()
    assert verify.m4_command == "verify-canary"
    assert bundle.m4_command == "verify-bundle"


def test_m4_validate_request_passes_for_checked_in_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["m4", "validate-request", str(REQUEST)]) == 0

    rendered = _output(capsys)
    assert rendered == {
        "all_m2_anchor_evidence_available": False,
        "canonical_source_ids": ["source0", "source1"],
        "identity_position_authority": "formal_m1_source_pose",
        "listener_id": "listener0",
        "request": str(REQUEST.resolve()),
        "request_id": "m4_blender_custom_two_source_foa_v1",
        "status": "pass",
    }


def test_m4_run_canary_blocks_before_native_when_hrtf_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_hrtf = tmp_path / "missing.sofa"
    license_path = tmp_path / "COPYING"
    license_path.write_text("fixture license", encoding="utf-8")

    exit_code = main(
        [
            "m4",
            "run-canary",
            "--request",
            "unused-request.json",
            "--package-manifest",
            "unused-package.json",
            "--hrtf",
            str(missing_hrtf),
            "--hrtf-license",
            str(license_path),
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 3
    assert _output(capsys) == {
        "missing": [str(missing_hrtf.resolve())],
        "reason": "explicit HRTF or its license evidence is unavailable",
        "status": "blocked",
    }


def test_m4_run_canary_mocked_pass_does_not_invoke_native(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hrtf = tmp_path / "fixture.sofa"
    license_path = tmp_path / "COPYING"
    hrtf.write_bytes(b"fixture hrtf")
    license_path.write_text("fixture license", encoding="utf-8")
    evidence = tmp_path / "canary" / "m4_canary_evidence.json"
    calls: dict[str, Any] = {}

    def fake_run(
        request: str,
        package_manifest: str,
        runtime_lock: str,
        output: Path,
        *,
        hrtf_path: str,
        hrtf_license_path: str,
    ) -> Path:
        calls["run"] = (
            request,
            package_manifest,
            runtime_lock,
            output,
            hrtf_path,
            hrtf_license_path,
        )
        return evidence

    def fake_verify(path: str | Path) -> tuple[str, tuple[dict[str, Any], ...]]:
        calls["verify"] = path
        return "pass", ({"check_id": "contract", "status": "pass"},)

    monkeypatch.setattr("avengine.cli.run_m4_canary", fake_run)
    monkeypatch.setattr("avengine.cli.verify_m4_canary_evidence", fake_verify)

    output = tmp_path / "canary"
    exit_code = main(
        [
            "m4",
            "run-canary",
            "--request",
            "request.json",
            "--package-manifest",
            "manifest.json",
            "--hrtf",
            str(hrtf),
            "--hrtf-license",
            str(license_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert calls["run"] == (
        "request.json",
        "manifest.json",
        str(DEFAULT_RUNTIME_LOCK),
        output.resolve(),
        str(hrtf),
        str(license_path),
    )
    assert calls["verify"] == evidence
    assert _output(capsys) == {
        "canary_evidence": str(evidence),
        "failed_checks": [],
        "status": "pass",
    }


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [("pass", 0), ("fail", 1)],
)
def test_m4_verify_canary_maps_verifier_status_to_stable_exit(
    status: str,
    expected_exit: int,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = ({"check_id": "contract", "status": status},)
    monkeypatch.setattr(
        "avengine.cli.verify_m4_canary_evidence",
        lambda _path: (status, checks),
    )

    assert main(["m4", "verify-canary", "evidence.json"]) == expected_exit
    assert _output(capsys) == {"checks": list(checks), "status": status}


def test_m4_verify_canary_malformed_json_is_a_stable_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = tmp_path / "malformed.json"
    evidence.write_text("{not-json", encoding="utf-8")

    assert main(["m4", "verify-canary", str(evidence)]) == 1
    rendered = _output(capsys)
    assert rendered["status"] == "fail"
    assert rendered["checks"][0]["check_id"] == "evidence_json"
    assert rendered["checks"][0]["status"] == "fail"


@pytest.mark.parametrize(
    ("errors", "expected_status", "expected_exit"),
    [([], "pass", 0), (["pair closure failed"], "fail", 1)],
)
def test_m4_verify_bundle_maps_contract_result_to_stable_exit(
    errors: list[str],
    expected_status: str,
    expected_exit: int,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = {"schema": "avengine_m4_audio_bundle_v1"}
    calls: dict[str, Any] = {}

    monkeypatch.setattr("avengine.cli.load_json", lambda _path: bundle)

    def fake_validate(value: Any, *, bundle_path: str | Path) -> list[str]:
        calls["validation"] = (value, bundle_path)
        return errors

    monkeypatch.setattr("avengine.cli.validate_audio_bundle", fake_validate)

    assert main(["m4", "verify-bundle", "bundle.json"]) == expected_exit
    assert calls["validation"] == (bundle, "bundle.json")
    assert _output(capsys) == {"errors": errors, "status": expected_status}
