from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from avengine.cli import build_parser, main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUEST = (
    REPOSITORY_ROOT
    / "examples/m5/blender_custom/two_dog_simultaneous_counterfactual_request.json"
)


def _output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    value = json.loads(capsys.readouterr().out)
    assert isinstance(value, dict)
    return value


def test_m5_parser_exposes_validate_run_and_verify() -> None:
    parser = build_parser()
    validate = parser.parse_args(["m5", "validate-request", "request.json"])
    run = parser.parse_args(
        [
            "m5",
            "run-canary",
            "--request",
            "request.json",
            "--animal-manifest",
            "animal.json",
            "--m2-request",
            "m2.json",
            "--room-manifest",
            "room.json",
            "--m1-request",
            "m1.json",
            "--acoustic-package-manifest",
            "acoustic.json",
            "--m4-request",
            "m4.json",
            "--beagle-dry",
            "beagle.wav",
            "--golden-dry",
            "golden.wav",
            "--sensor-rig-trajectory",
            "rig.json",
            "--output",
            "/tmp/m5-output",
        ]
    )
    verify = parser.parse_args(["m5", "verify-canary", "evidence.json"])
    assert validate.m5_command == "validate-request"
    assert run.m5_command == "run-canary"
    assert run.hrtf == "/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"
    assert run.runtime_root is None
    assert run.runtime_prefix is None
    assert run.mp3d_root is None
    assert run.magnum_python_site is None
    assert run.rlr_sdk_root is None
    assert run.sensor_rig_trajectory == "rig.json"
    assert verify.m5_command == "verify-canary"


def test_m5_checked_in_request_validates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["m5", "validate-request", str(REQUEST)]) == 0
    rendered = _output(capsys)
    assert rendered["status"] == "pass"
    assert rendered["formal_view_ids"] == ["view0"]
    assert rendered["simultaneous_source_count"] == 2
    assert rendered["errors"] == []


def test_m5_run_canary_mock_forwards_explicit_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    evidence = tmp_path / "result" / "evidence.json"

    def fake_run(**kwargs: Any) -> Path:
        calls["run"] = kwargs
        return evidence

    def fake_verify(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
        calls["verify"] = path
        return "pass", [{"check_id": "m5", "status": "pass"}]

    monkeypatch.setattr("avengine.cli.run_m5_canary", fake_run)
    monkeypatch.setattr("avengine.cli.verify_m5_canary_evidence", fake_verify)
    output = tmp_path / "result"
    argv = [
        "m5",
        "run-canary",
        "--request",
        "request.json",
        "--animal-manifest",
        "animal.json",
        "--m2-request",
        "m2.json",
        "--room-manifest",
        "room.json",
        "--m1-request",
        "m1.json",
        "--acoustic-package-manifest",
        "acoustic.json",
        "--m4-request",
        "m4.json",
        "--runtime-root",
        "/runtime",
        "--mp3d-root",
        "/mp3d",
        "--magnum-python-site",
        "/magnum",
        "--rlr-sdk-root",
        "/rlr",
        "--hrtf",
        "hrtf.sofa",
        "--hrtf-license",
        "COPYING",
        "--beagle-dry",
        "beagle.wav",
        "--golden-dry",
        "golden.wav",
        "--sensor-rig-trajectory",
        "rig.json",
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    assert calls["run"]["output_directory"] == output.resolve()
    assert calls["run"]["runtime_root"] == "/runtime"
    assert calls["run"]["runtime_prefix"] is None
    assert calls["run"]["mp3d_root"] == "/mp3d"
    assert calls["run"]["magnum_python_site"] == "/magnum"
    assert calls["run"]["rlr_sdk_root"] == "/rlr"
    assert calls["run"]["beagle_dry_path"] == "beagle.wav"
    assert calls["run"]["sensor_rig_trajectory_path"] == "rig.json"
    assert calls["verify"] == evidence
    assert _output(capsys) == {
        "canary_evidence": str(evidence),
        "failed_checks": [],
        "status": "pass",
    }


@pytest.mark.parametrize(("status", "exit_code"), [("pass", 0), ("fail", 1)])
def test_m5_verify_status_mapping(
    status: str,
    exit_code: int,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = [{"check_id": "evidence", "status": status}]
    monkeypatch.setattr(
        "avengine.cli.verify_m5_canary_evidence", lambda _path: (status, checks)
    )
    assert main(["m5", "verify-canary", "evidence.json"]) == exit_code
    assert _output(capsys) == {"checks": checks, "status": status}
