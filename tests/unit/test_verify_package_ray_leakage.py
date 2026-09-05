from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.acoustics import verify_package_ray_leakage as verifier


def _automatic_report() -> dict:
    return {
        "schema": "avengine_m3_ray_leakage_v1",
        "status": "not_run",
        "declared_check_count": 0,
        "checks": [],
        "automatic_enclosure_probe": {
            "schema": "avengine_m3_automatic_mesh_leakage_diagnostic_v1",
            "status": "diagnostic_complete",
            "maximum_distance_m": 10.0,
            "directions": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            "origins": [
                {
                    "origin_index": 0,
                    "origin_m": [0.0, 1.0, 0.0],
                    "escaped_direction_indices": [1],
                }
            ],
        },
    }


def test_automatic_declarations_replay_cpu_hit_expectations() -> None:
    declarations = verifier._automatic_declarations(_automatic_report())
    assert len(declarations) == 2
    assert declarations[0]["expectation"] == "hit_within_m"
    assert declarations[1]["expectation"] == "clear_until_m"
    assert all(item["distance_m"] == 10.0 for item in declarations)


def test_missing_runtime_is_explicitly_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scene = SimpleNamespace(qa_reports={"ray_leakage": _automatic_report()})
    monkeypatch.setattr(verifier, "load_compiled_acoustic_scene", lambda *a, **k: scene)
    report = verifier.verify_package_ray_leakage(
        package_manifest=tmp_path / "manifest.json",
        room_manifest=None,
        runtime_prefix=None,
        magnum_site=None,
        rlr_sdk_root=None,
    )
    assert report["status"] == "unavailable"
    assert report["rlr_runtime_ray_check_status"] == "unavailable"
    assert report["rlr_runtime_ray_check_count"] == 2
    assert "runtime_prefix argument is missing" in report[
        "rlr_runtime_unavailable_reason"
    ]


def test_cli_refuses_to_overwrite_existing_report(tmp_path: Path) -> None:
    output = tmp_path / "ray.json"
    output.write_text("{}\n", encoding="utf-8")
    assert verifier.main([
        "--package-manifest", str(tmp_path / "manifest.json"),
        "--output", str(output),
    ]) == 2
    assert output.read_text(encoding="utf-8") == "{}\n"


def _runtime_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = SimpleNamespace(qa_reports={"ray_leakage": _automatic_report()})
    monkeypatch.setattr(verifier, "load_compiled_acoustic_scene", lambda *a, **k: scene)
    monkeypatch.setattr(verifier, "_dependency_errors", lambda *a, **k: [])
    monkeypatch.setattr(
        verifier.RLRSimulationConfig, "from_mapping", lambda value: object()
    )


def test_runtime_contract_error_is_not_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _runtime_ready(monkeypatch)
    def fail(*args, **kwargs):
        raise verifier.RuntimeContractError("invalid TraceRay result")
    monkeypatch.setattr(verifier, "simulate_compiled_acoustic_scene", fail)
    report = verifier.verify_package_ray_leakage(
        package_manifest=tmp_path / "manifest.json",
        room_manifest=None,
        runtime_prefix=tmp_path,
        magnum_site=tmp_path,
        rlr_sdk_root=tmp_path,
    )
    assert report["status"] == "error"
    assert report["rlr_runtime_ray_check_status"] == "error"
    assert "RuntimeContractError" in report["rlr_runtime_error"]
    assert report["rlr_runtime_unavailable_reason"] is None


def test_runtime_success_requires_every_trace_ray_check_to_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _runtime_ready(monkeypatch)
    monkeypatch.setattr(
        verifier,
        "simulate_compiled_acoustic_scene",
        lambda *args, **kwargs: SimpleNamespace(
            ray_checks=({"check_id": "one", "passed": True},)
        ),
    )
    passed = verifier.verify_package_ray_leakage(
        package_manifest=tmp_path / "manifest.json",
        room_manifest=None,
        runtime_prefix=tmp_path,
        magnum_site=tmp_path,
        rlr_sdk_root=tmp_path,
    )
    assert passed["status"] == "pass"

    monkeypatch.setattr(
        verifier,
        "simulate_compiled_acoustic_scene",
        lambda *args, **kwargs: SimpleNamespace(
            ray_checks=({"check_id": "one", "passed": False},)
        ),
    )
    failed = verifier.verify_package_ray_leakage(
        package_manifest=tmp_path / "manifest.json",
        room_manifest=None,
        runtime_prefix=tmp_path,
        magnum_site=tmp_path,
        rlr_sdk_root=tmp_path,
    )
    assert failed["status"] == "fail"
