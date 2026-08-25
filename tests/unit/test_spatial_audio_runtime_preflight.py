from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avengine.acoustics.runtime import RuntimeUnavailableError
from avengine.contracts.json_io import canonical_json_sha256
from avengine.spatial_audio import canary
from avengine.spatial_audio import evidence as m4_evidence
from avengine.spatial_audio.contracts import CURRENT_INSTALLED_EVIDENCE_SCHEMA


def _lock(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "avengine_m4_runtime_lock_v1",
                "native_binaries": {
                    "habitat_sim_bindings": {"byte_size": 10, "sha256": "old"},
                    "rlr_audio_propagation": {"byte_size": 20, "sha256": "old"},
                },
                "hrtf": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_m4_preflight_stops_before_any_rir_job_on_historical_lock_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(tmp_path / "historical-lock.json")
    package = tmp_path / "package" / "manifest.json"
    package.parent.mkdir()
    package.write_text("{}", encoding="utf-8")
    calls: list[object] = []

    monkeypatch.setattr(
        canary,
        "load_and_validate_multi_source_canary_request",
        lambda _path: SimpleNamespace(request={}),
    )
    monkeypatch.setattr(
        canary,
        "load_compiled_acoustic_scene",
        lambda _path: SimpleNamespace(manifest_path=package),
    )
    monkeypatch.setattr(
        canary,
        "load_habitat_runtime",
        lambda: (
            object(),
            {
                "native_binaries": {
                    "habitat_sim_bindings": {"byte_size": 11, "sha256": "new"},
                    "rlr_audio_propagation": {"byte_size": 21, "sha256": "new"},
                }
            },
        ),
    )
    monkeypatch.setattr(
        canary,
        "render_named_sources",
        lambda *_args, **_kwargs: calls.append("rir") or None,
    )

    output = tmp_path / "fresh-output"
    with pytest.raises(RuntimeUnavailableError, match="before executing an RLR job"):
        canary.run_m4_canary(
            tmp_path / "request.json",
            package,
            lock,
            output,
            hrtf_path=tmp_path / "hrtf.sofa",
            hrtf_license_path=tmp_path / "COPYING",
        )

    assert calls == []
    assert not output.exists()


def test_m4_preflight_accepts_the_existing_lock_binary_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = json.loads(_lock(tmp_path / "historical-lock.json").read_text())
    monkeypatch.setattr(
        canary,
        "load_habitat_runtime",
        lambda: (object(), {"native_binaries": lock["native_binaries"]}),
    )

    canary._preflight_runtime_binary_lock(lock)

def _current_identity(root: Path) -> dict[str, object]:
    habitat = root / "habitat"
    sdk = root / "sdk"
    magnum_site = root / "magnum/python"
    module = habitat / "python/habitat_sim/__init__.py"
    binding = habitat / "python/habitat_sim/_ext.so"
    header = sdk / "include/RLRAcousticContext.h"
    library = sdk / "lib/libRLR.so"
    for path in (module, binding, header, library):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    magnum_site.mkdir(parents=True, exist_ok=True)
    return {
        "identity_schema": "avengine_current_installed_rlr_runtime_v1",
        "mode": "current-installed",
        "habitat_runtime_prefix": str(habitat.resolve()),
        "habitat_sim_module": str(module.resolve()),
        "habitat_sim_binding": str(binding.resolve()),
        "magnum_python_site": str(magnum_site.resolve()),
        "rlr_sdk_root": str(sdk.resolve()),
        "rlr_sdk_header": str(header.resolve()),
        "rlr_sdk_library": str(library.resolve()),
        "rlr_adapter_enabled": True,
        "binding_api": "habitat_sim.RLRAcousticContext_v1",
    }


def test_current_m4_identity_requires_same_fresh_runtime_every_repeat(
    tmp_path: Path,
) -> None:
    first = _current_identity(tmp_path / "first")
    passed, summary, unique_count = canary._current_installed_identity_summary(
        [first, dict(first), dict(first)]
    )
    assert passed is True
    assert summary == first
    assert unique_count == 1

    different = _current_identity(tmp_path / "different")
    passed, summary, unique_count = canary._current_installed_identity_summary(
        [first, different]
    )
    assert passed is False
    assert summary is None
    assert unique_count == 2


def test_current_m4_never_parses_historical_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_calls: list[object] = []

    def forbidden_lock(_path: Path) -> dict[str, object]:
        lock_calls.append(_path)
        raise AssertionError("current-installed must not parse a v1 lock")

    monkeypatch.setattr(canary, "_runtime_lock", forbidden_lock)
    monkeypatch.setattr(
        canary,
        "load_and_validate_multi_source_canary_request",
        lambda _path: SimpleNamespace(request={}),
    )
    monkeypatch.setattr(
        canary,
        "load_compiled_acoustic_scene",
        lambda _path: (_ for _ in ()).throw(RuntimeUnavailableError("stop before run")),
    )

    with pytest.raises(RuntimeUnavailableError, match="stop before run"):
        canary.run_m4_canary(
            tmp_path / "request.json",
            tmp_path / "package.json",
            tmp_path / "old-v1-lock.json",
            tmp_path / "fresh-output",
            hrtf_path=tmp_path / "hrtf.sofa",
            hrtf_license_path=tmp_path / "COPYING",
            runtime_mode="current-installed",
            runtime_prefix="/current/habitat",
            rlr_sdk_root="/current/sdk",
            magnum_python_site="/current/magnum/python",
            current_hrtf_sample_rate_hz=48000,
            current_hrtf_license_id="fixture-license",
            current_hrtf_citation="fixture citation",
        )

    assert lock_calls == []


def test_m4_current_v2_reader_dispatch_skips_historical_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path = tmp_path / "m4-v2.json"
    evidence_path.write_text(
        json.dumps({"schema": CURRENT_INSTALLED_EVIDENCE_SCHEMA}),
        encoding="utf-8",
    )
    expected = ("pass", ({"check_id": "v2", "status": "pass"},))
    monkeypatch.setattr(
        m4_evidence,
        "_verify_current_installed_m4_canary_evidence",
        lambda _path, _evidence: expected,
    )
    monkeypatch.setattr(
        m4_evidence,
        "_verify_historical_m4_canary_evidence",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("v2 must not enter the historical lock reader")
        ),
    )

    assert m4_evidence.verify_m4_canary_evidence(evidence_path) == expected


def _reader_evidence(
    records: list[dict[str, object]], *, identity_passed: bool
) -> dict[str, object]:
    unique: list[dict[str, object]] = []
    for record in records:
        if record not in unique:
            unique.append(record)
    failure_reason = "current runtime identity changed"
    identity_check: dict[str, object] = {
        "check_id": "runtime_current_installed_identity",
        "required": True,
        "status": "pass" if identity_passed else "fail",
        "measured": {
            "record_count": len(records),
            "unique_identity_count": len(unique),
            "identities": records,
        },
        "threshold": {
            "same_runtime_identity_every_native_call": True,
            "runtime_mode": "current-installed",
        },
    }
    if not identity_passed:
        identity_check["failure_reason"] = failure_reason
    evidence: dict[str, object] = {
        "schema": CURRENT_INSTALLED_EVIDENCE_SCHEMA,
        "inputs": {
            "hrtf_role": "hrtf",
            "hrtf_license_role": "license",
        },
        "execution": {"runtime_mode": "current-installed"},
        "performance": {
            "one_source": {"runs": [{"runtime_identity": records[7]}]},
            "multi_source": {"runs": [{"runtime_identity": records[8]}]},
        },
        "runtime": {
            "runtime_mode": "current-installed",
            "current_installed_identity": records[0] if identity_passed else None,
            "current_installed_identity_records": records,
        },
        "audio_contracts": {
            "binaural": {
                "hrtf": {"artifact_role": "hrtf"},
                "rights": {"license_artifact_role": "license"},
                "sample_rate_binding": {
                    "policy": "strict_match",
                    "native_rate_adaptation": "not_required",
                },
            }
        },
        "checks": [identity_check],
        "overall_status": "pass" if identity_passed else "fail",
        "failure_reasons": [] if identity_passed else [failure_reason],
        "evidence_content_sha256": "",
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in evidence.items()
            if key != "evidence_content_sha256"
        }
    )
    return evidence


def test_m4_current_reader_rejects_identity_only_receipt(tmp_path: Path) -> None:
    identity = _current_identity(tmp_path / "identity")
    evidence = _reader_evidence(
        [dict(identity) for _ in range(9)], identity_passed=True
    )
    evidence_path = tmp_path / "identity-only.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    status, checks = m4_evidence.verify_m4_canary_evidence(evidence_path)

    assert status == "fail"
    assert any(
        check["check_id"] == "evidence_contract" and check["status"] == "fail"
        for check in checks
    )
