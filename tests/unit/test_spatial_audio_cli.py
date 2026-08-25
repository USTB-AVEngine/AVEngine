from __future__ import annotations

import json
import os
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


def test_spatial_audio_parser_exposes_commands_and_pins_default_runtime_lock() -> None:
    parser = build_parser()

    validate = parser.parse_args(["spatial-audio", "validate-request", "request.json"])
    run = parser.parse_args(
        [
            "spatial-audio",
            "run-canary",
            "--request",
            "request.json",
            "--package-manifest",
            "manifest.json",
            "--output",
            "/tmp/m4-canary",
        ]
    )
    verify = parser.parse_args(["spatial-audio", "verify-canary", "evidence.json"])
    bundle = parser.parse_args(["spatial-audio", "verify-bundle", "bundle.json"])

    assert validate.m4_command == "validate-request"
    assert run.m4_command == "run-canary"
    assert Path(run.runtime_lock) == DEFAULT_RUNTIME_LOCK
    assert DEFAULT_RUNTIME_LOCK.is_file()
    assert verify.m4_command == "verify-canary"
    assert bundle.m4_command == "verify-bundle"


def test_spatial_audio_current_installed_help_requires_explicit_runtime_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["spatial-audio", "run-canary", "--help"])
    assert exit_info.value.code == 0
    rendered = " ".join(capsys.readouterr().out.split())
    assert (
        "Explicit installed non-checkout Habitat prefix required with "
        "--runtime-mode current-installed"
    ) in rendered
    assert (
        "Explicit external non-checkout RLRAudioPropagationPkg required with "
        "--runtime-mode current-installed"
    ) in rendered
    assert "AVENGINE_HABITAT_RUNTIME_PREFIX" not in rendered
    assert "AVENGINE_RLR_SDK_ROOT" not in rendered


def test_spatial_audio_validate_request_passes_for_checked_in_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["spatial-audio", "validate-request", str(REQUEST)]) == 0

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


def test_spatial_audio_run_canary_blocks_before_native_when_hrtf_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_hrtf = tmp_path / "missing.sofa"
    license_path = tmp_path / "COPYING"
    license_path.write_text("fixture license", encoding="utf-8")

    exit_code = main(
        [
            "spatial-audio",
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


def test_spatial_audio_run_canary_mocked_pass_does_not_invoke_native(
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
        calls["external_env"] = (
            os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX"),
            os.environ.get("AVENGINE_RLR_SDK_ROOT"),
        )
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
    monkeypatch.delenv("AVENGINE_HABITAT_RUNTIME_PREFIX", raising=False)
    monkeypatch.delenv("AVENGINE_RLR_SDK_ROOT", raising=False)

    output = tmp_path / "canary"
    exit_code = main(
        [
            "spatial-audio",
            "run-canary",
            "--request",
            "request.json",
            "--package-manifest",
            "manifest.json",
            "--hrtf",
            str(hrtf),
            "--hrtf-license",
            str(license_path),
            "--runtime-prefix",
            "/external/habitat-prefix",
            "--rlr-sdk-root",
            "/external/rlr-sdk",
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
    assert calls["external_env"] == (
        "/external/habitat-prefix",
        "/external/rlr-sdk",
    )
    assert os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX") is None
    assert os.environ.get("AVENGINE_RLR_SDK_ROOT") is None
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
def test_spatial_audio_verify_canary_maps_verifier_status_to_stable_exit(
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

    assert main(["spatial-audio", "verify-canary", "evidence.json"]) == expected_exit
    assert _output(capsys) == {"checks": list(checks), "status": status}


def test_spatial_audio_verify_canary_malformed_json_is_a_stable_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = tmp_path / "malformed.json"
    evidence.write_text("{not-json", encoding="utf-8")

    assert main(["spatial-audio", "verify-canary", str(evidence)]) == 1
    rendered = _output(capsys)
    assert rendered["status"] == "fail"
    assert rendered["checks"][0]["check_id"] == "evidence_json"
    assert rendered["checks"][0]["status"] == "fail"


@pytest.mark.parametrize(
    ("errors", "expected_status", "expected_exit"),
    [([], "pass", 0), (["pair closure failed"], "fail", 1)],
)
def test_spatial_audio_verify_bundle_maps_contract_result_to_stable_exit(
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

    assert main(["spatial-audio", "verify-bundle", "bundle.json"]) == expected_exit
    assert calls["validation"] == (bundle, "bundle.json")
    assert _output(capsys) == {"errors": errors, "status": expected_status}

def test_spatial_audio_current_installed_cli_requires_and_forwards_explicit_runtime_inputs(
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
        runtime_lock: str | None,
        output: Path,
        *,
        hrtf_path: str,
        hrtf_license_path: str,
        **kwargs: Any,
    ) -> Path:
        calls["run"] = (
            request,
            package_manifest,
            runtime_lock,
            output,
            hrtf_path,
            hrtf_license_path,
            kwargs,
        )
        calls["environment"] = (
            os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX"),
            os.environ.get("AVENGINE_RLR_SDK_ROOT"),
            os.environ.get("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"),
        )
        return evidence

    monkeypatch.setattr("avengine.cli.run_m4_canary", fake_run)
    monkeypatch.setattr(
        "avengine.cli.verify_m4_canary_evidence",
        lambda _path: ("pass", ({"check_id": "contract", "status": "pass"},)),
    )

    output = tmp_path / "canary"
    command = [
        "spatial-audio",
        "run-canary",
        "--request",
        "request.json",
        "--package-manifest",
        "manifest.json",
        "--hrtf",
        str(hrtf),
        "--hrtf-license",
        str(license_path),
        "--runtime-mode",
        "current-installed",
        "--runtime-prefix",
        "/current/habitat",
        "--rlr-sdk-root",
        "/current/sdk",
        "--magnum-python-site",
        "/current/magnum/python",
        "--current-hrtf-sample-rate-hz",
        "48000",
        "--current-hrtf-license-id",
        "fixture-license",
        "--current-hrtf-citation",
        "fixture citation",
        "--output",
        str(output),
    ]
    assert main(command) == 0
    assert calls["run"] == (
        "request.json",
        "manifest.json",
        None,
        output.resolve(),
        str(hrtf),
        str(license_path),
        {
            "runtime_mode": "current-installed",
            "runtime_prefix": "/current/habitat",
            "rlr_sdk_root": "/current/sdk",
            "magnum_python_site": "/current/magnum/python",
            "current_hrtf_sample_rate_hz": 48000,
            "current_hrtf_license_id": "fixture-license",
            "current_hrtf_citation": "fixture citation",
        },
    )
    assert calls["environment"] == (
        "/current/habitat",
        "/current/sdk",
        "/current/magnum/python",
    )
    assert _output(capsys) == {
        "canary_evidence": str(evidence),
        "failed_checks": [],
        "status": "pass",
    }

    incomplete = [
        "spatial-audio",
        "run-canary",
        "--request",
        "request.json",
        "--package-manifest",
        "manifest.json",
        "--hrtf",
        str(hrtf),
        "--hrtf-license",
        str(license_path),
        "--runtime-mode",
        "current-installed",
        "--output",
        str(tmp_path / "second-output"),
    ]
    assert main(incomplete) == 2
    rendered = _output(capsys)
    assert rendered["status"] == "fail"
    assert "--runtime-prefix" in rendered["error"]

def test_spatial_audio_current_foa_parser_requires_only_current_runtime_inputs() -> None:
    command = [
        "spatial-audio",
        "run-current-foa",
        "--request",
        "request.json",
        "--package-manifest",
        "manifest.json",
        "--runtime-prefix",
        "/current/habitat",
        "--rlr-sdk-root",
        "/current/sdk",
        "--magnum-python-site",
        "/current/magnum/python",
        "--output",
        "/tmp/current-foa",
    ]
    parser = build_parser()
    parsed = parser.parse_args(command)

    assert parsed.m4_command == "run-current-foa"
    assert parsed.runtime_mode == "current-installed"
    assert not hasattr(parsed, "hrtf")
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args([*command, "--hrtf", "unexpected.sofa"])
    assert exit_info.value.code == 2


def test_spatial_audio_current_foa_cli_forwards_explicit_inputs_without_hrtf(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def fake_run(
        request: str,
        package_manifest: str,
        output: Path,
        *,
        runtime_prefix: str,
        rlr_sdk_root: str,
        magnum_python_site: str,
    ) -> dict[str, Any]:
        calls["run"] = (
            request,
            package_manifest,
            output,
            runtime_prefix,
            rlr_sdk_root,
            magnum_python_site,
        )
        calls["environment"] = (
            os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX"),
            os.environ.get("AVENGINE_RLR_SDK_ROOT"),
            os.environ.get("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"),
        )
        return {
            "status": "pass",
            "research_status": "research_candidate",
            "qualification_claim": False,
            "binaural": "not_requested",
            "pairs": [{"source_id": "source0"}, {"source_id": "source1"}],
        }

    monkeypatch.setattr("avengine.cli.run_current_foa", fake_run)
    monkeypatch.delenv("AVENGINE_HABITAT_RUNTIME_PREFIX", raising=False)
    monkeypatch.delenv("AVENGINE_RLR_SDK_ROOT", raising=False)
    monkeypatch.delenv("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", raising=False)
    output = tmp_path / "current-foa"

    assert main(
        [
            "spatial-audio",
            "run-current-foa",
            "--request",
            "request.json",
            "--package-manifest",
            "manifest.json",
            "--runtime-prefix",
            "/current/habitat",
            "--rlr-sdk-root",
            "/current/sdk",
            "--magnum-python-site",
            "/current/magnum/python",
            "--output",
            str(output),
        ]
    ) == 0

    assert calls["run"] == (
        "request.json",
        "manifest.json",
        output.resolve(),
        "/current/habitat",
        "/current/sdk",
        "/current/magnum/python",
    )
    assert calls["environment"] == (
        "/current/habitat",
        "/current/sdk",
        "/current/magnum/python",
    )
    assert os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX") is None
    assert os.environ.get("AVENGINE_RLR_SDK_ROOT") is None
    assert os.environ.get("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE") is None
    assert _output(capsys) == {
        "binaural": "not_requested",
        "foa_pair_count": 2,
        "output": str(output.resolve()),
        "qualification_claim": False,
        "research_status": "research_candidate",
        "status": "pass",
    }
