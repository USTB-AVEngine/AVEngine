from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from avengine.cli import build_parser, main
from avengine.spatial_audio.current_request_pair_ir import CurrentM1PairIRBlockedError


def _output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def _common_command(name: str, output: Path) -> list[str]:
    return [
        "m4",
        name,
        "--m1-request",
        "m1.json",
        "--simulation-request",
        "simulation.json",
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


def _binaural_command(output: Path) -> list[str]:
    return [
        *_common_command("run-current-m1-binaural", output),
        "--hrtf",
        "/external/hrtf/derived-16k.sofa",
        "--hrtf-sha256",
        "a" * 64,
        "--hrtf-sample-rate-hz",
        "16000",
        "--hrtf-license",
        "/external/hrtf/LICENSE.txt",
        "--hrtf-license-sha256",
        "b" * 64,
        "--hrtf-license-id",
        "MIT-KEMAR-license",
        "--hrtf-citation",
        "MIT KEMAR citation",
    ]


def _receipt() -> dict[str, Any]:
    return {
        "status": "pass",
        "research_status": "research_candidate",
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification": False,
        "qualification_claim": False,
        "pairs": [{"source_id": "source0"}, {"source_id": "source1"}],
    }


def test_current_m1_foa_parser_has_no_source_or_hrtf_override(tmp_path: Path) -> None:
    command = _common_command("run-current-m1-foa", tmp_path / "foa")
    parser = build_parser()
    parsed = parser.parse_args(command)

    assert parsed.m4_command == "run-current-m1-foa"
    assert parsed.runtime_mode == "current-installed"
    assert not hasattr(parsed, "source")
    assert not hasattr(parsed, "hrtf")
    with pytest.raises(SystemExit) as source_exit:
        parser.parse_args([*command, "--source", "0,0,0"])
    assert source_exit.value.code == 2
    with pytest.raises(SystemExit) as hrtf_exit:
        parser.parse_args([*command, "--hrtf", "unexpected.sofa"])
    assert hrtf_exit.value.code == 2


def test_current_m1_binaural_parser_requires_explicit_hrtf_rights(
    tmp_path: Path,
) -> None:
    command = _binaural_command(tmp_path / "binaural")
    parser = build_parser()
    parsed = parser.parse_args(command)

    assert parsed.m4_command == "run-current-m1-binaural"
    assert parsed.runtime_mode == "current-installed"
    assert parsed.hrtf_sample_rate_hz == 16_000
    assert parsed.hrtf_sha256 == "a" * 64
    assert parsed.hrtf_license_sha256 == "b" * 64
    assert not hasattr(parsed, "source")
    citation_index = command.index("--hrtf-citation")
    incomplete = command[:citation_index]
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(incomplete)
    assert exit_info.value.code == 2


def test_current_m1_foa_cli_forwards_only_m1_and_current_runtime_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def fake_run(
        m1_request: str,
        simulation_request: str,
        package_manifest: str,
        output: Path,
        *,
        runtime_prefix: str,
        rlr_sdk_root: str,
        magnum_python_site: str,
    ) -> dict[str, Any]:
        calls["run"] = (
            m1_request,
            simulation_request,
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
        return _receipt()

    monkeypatch.setattr("avengine.cli.run_current_m1_foa", fake_run)
    monkeypatch.delenv("AVENGINE_HABITAT_RUNTIME_PREFIX", raising=False)
    monkeypatch.delenv("AVENGINE_RLR_SDK_ROOT", raising=False)
    monkeypatch.delenv("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", raising=False)
    output = tmp_path / "foa"

    assert main(_common_command("run-current-m1-foa", output)) == 0

    assert calls["run"] == (
        "m1.json",
        "simulation.json",
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
        "episode_counted": False,
        "foa_pair_count": 2,
        "formal_dataset_count": 0,
        "output": str(output.resolve()),
        "qualification_claim": False,
        "research_only": True,
        "research_status": "research_candidate",
        "status": "pass",
    }


def test_current_m1_binaural_cli_forwards_explicit_hrtf_and_rights(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def fake_run(
        m1_request: str,
        simulation_request: str,
        package_manifest: str,
        output: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls["run"] = (
            m1_request,
            simulation_request,
            package_manifest,
            output,
            kwargs,
        )
        return _receipt()

    monkeypatch.setattr("avengine.cli.run_current_m1_binaural", fake_run)
    output = tmp_path / "binaural"

    assert main(_binaural_command(output)) == 0

    request, simulation, package, observed_output, kwargs = calls["run"]
    assert (request, simulation, package) == (
        "m1.json",
        "simulation.json",
        "manifest.json",
    )
    assert observed_output == output.resolve()
    assert kwargs == {
        "runtime_prefix": "/current/habitat",
        "rlr_sdk_root": "/current/sdk",
        "magnum_python_site": "/current/magnum/python",
        "hrtf_path": "/external/hrtf/derived-16k.sofa",
        "expected_hrtf_sha256": "a" * 64,
        "hrtf_sample_rate_hz": 16_000,
        "hrtf_license_path": "/external/hrtf/LICENSE.txt",
        "expected_hrtf_license_sha256": "b" * 64,
        "hrtf_license_id": "MIT-KEMAR-license",
        "hrtf_citation": "MIT KEMAR citation",
    }
    assert _output(capsys) == {
        "binaural_pair_count": 2,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "output": str(output.resolve()),
        "qualification_claim": False,
        "research_only": True,
        "research_status": "research_candidate",
        "status": "pass",
    }


def test_current_m1_binaural_cli_maps_missing_external_dependency_to_blocked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise CurrentM1PairIRBlockedError("explicit HRTF is unavailable")

    monkeypatch.setattr("avengine.cli.run_current_m1_binaural", blocked)
    assert main(_binaural_command(tmp_path / "binaural")) == 3
    assert _output(capsys) == {
        "status": "blocked",
        "error": "explicit HRTF is unavailable",
    }
