from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from avengine.cli import _aggregate


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
