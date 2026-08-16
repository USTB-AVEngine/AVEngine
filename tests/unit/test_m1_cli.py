from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from avengine.cli import _aggregate, _capture, build_parser
from avengine.m1.contracts import EVIDENCE_SCHEMA_V2
from avengine.m1.evidence import verify_evidence_artifacts


ROOM_KINDS = [
    "habitat_native",
    "blender_custom",
    "legacy_ue_real_surface_export",
]


def _write_evidence(
    path: Path,
    *,
    room_kind: str,
    room_id: str,
    request_id: str,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "room_kind": room_kind,
                "room_id": room_id,
                "request_id": request_id,
                "overall_status": "pass",
                "evidence_content_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_aggregate(
    paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_path: Path | None = None,
) -> int:
    def fake_verify(path: str | Path) -> tuple[str, list[dict]]:
        status = (
            "fail" if failed_path is not None and Path(path) == failed_path else "pass"
        )
        return status, []

    monkeypatch.setattr("avengine.cli.verify_evidence_artifacts", fake_verify)
    return _aggregate(
        argparse.Namespace(evidence=[str(path) for path in paths], output=None)
    )


def _valid_three(tmp_path: Path) -> list[Path]:
    return [
        _write_evidence(
            tmp_path / f"{index}.json",
            room_kind=room_kind,
            room_id=f"room-{index}",
            request_id=f"request-{index}",
        )
        for index, room_kind in enumerate(ROOM_KINDS)
    ]


def test_m1_aggregate_accepts_exactly_one_verified_entry_per_room_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run_aggregate(_valid_three(tmp_path), monkeypatch) == 0


@pytest.mark.parametrize("entry_count", [2, 4])
def test_m1_aggregate_rejects_wrong_entry_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_count: int,
) -> None:
    paths = _valid_three(tmp_path)
    if entry_count == 2:
        paths = paths[:2]
    else:
        paths.append(
            _write_evidence(
                tmp_path / "extra.json",
                room_kind="habitat_native",
                room_id="room-extra",
                request_id="request-extra",
            )
        )

    assert _run_aggregate(paths, monkeypatch) == 1


@pytest.mark.parametrize("duplicated_field", ["room_kind", "room_id", "request_id"])
def test_m1_aggregate_rejects_duplicate_identity_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duplicated_field: str,
) -> None:
    paths = _valid_three(tmp_path)
    second = json.loads(paths[1].read_text(encoding="utf-8"))
    first = json.loads(paths[0].read_text(encoding="utf-8"))
    second[duplicated_field] = first[duplicated_field]
    paths[1].write_text(json.dumps(second), encoding="utf-8")

    assert _run_aggregate(paths, monkeypatch) == 1


def test_m1_aggregate_rejects_one_failed_room_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _valid_three(tmp_path)

    assert _run_aggregate(paths, monkeypatch, failed_path=paths[1]) == 1


def test_m1_capture_uses_runtime_prefix_and_writes_v2_blocked_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room_path = tmp_path / "room.json"
    request_path = tmp_path / "request.json"
    room_path.write_text("{}", encoding="utf-8")
    request_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "blocked"
    prefix = tmp_path / "installed_prefix"
    prefix.mkdir()
    inputs = type(
        "Inputs",
        (),
        {
            "room": {
                "room_id": "room0",
                "room_kind": "habitat_native",
            },
            "request": {"request_id": "request0"},
            "room_path": room_path,
            "request_path": request_path,
        },
    )()
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "avengine.cli.load_and_validate_inputs", lambda *_: inputs
    )

    def fail_capture(_inputs, capture_output, **kwargs):
        seen.update(kwargs)
        Path(capture_output).mkdir(parents=True, exist_ok=True)
        raise RuntimeError("prefix import fixture failure")

    monkeypatch.setattr("avengine.cli.capture_m1", fail_capture)
    exit_code = _capture(
        argparse.Namespace(
            room=str(room_path),
            request=str(request_path),
            output=str(output),
            runtime_prefix=str(prefix),
            repeat=2,
            reference_evidence=None,
        )
    )

    evidence_path = output / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    status, checks = verify_evidence_artifacts(evidence_path)

    assert exit_code == 3
    assert seen["runtime_prefix"] == str(prefix)
    assert evidence["schema"] == EVIDENCE_SCHEMA_V2
    assert status == "blocked"
    assert _checks_by_id(checks)["evidence_json_schema"]["status"] == "pass"


def test_m1_parser_requires_runtime_prefix_and_rejects_legacy_runtime_root() -> None:
    parser = build_parser()

    capture = parser.parse_args(
        [
            "m1",
            "capture",
            "--room",
            "room.json",
            "--request",
            "request.json",
            "--output",
            "out",
            "--runtime-prefix",
            "/installed/prefix",
        ]
    )
    navmesh = parser.parse_args(
        [
            "m1",
            "build-navmesh",
            "--room",
            "room.json",
            "--request",
            "request.json",
            "--runtime-prefix",
            "/installed/prefix",
        ]
    )

    assert capture.runtime_prefix == "/installed/prefix"
    assert navmesh.runtime_prefix == "/installed/prefix"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "m1",
                "capture",
                "--room",
                "room.json",
                "--request",
                "request.json",
                "--output",
                "out",
                "--runtime-root",
                "/legacy/checkout",
            ]
        )


def _checks_by_id(checks: list[dict]) -> dict[str, dict]:
    return {check["check_id"]: check for check in checks}
