from __future__ import annotations

import math
from pathlib import Path

import pytest

from avengine.assets.world_contact import (
    WorldContactError,
    scaled_contact_gate,
)
from tools.assets import audit_world_contacts


@pytest.mark.parametrize(
    ("scale", "expected"),
    ((0.82, 0.0123), (1.0, 0.015), (1.18, 0.0177)),
)
def test_contact_gate_scales_with_uniform_animal_size(
    scale: float, expected: float
) -> None:
    assert scaled_contact_gate(scale) == pytest.approx(expected)


@pytest.mark.parametrize("scale", (0.0, 0.09, 10.01, math.inf, math.nan))
def test_contact_gate_rejects_unsafe_scale(scale: float) -> None:
    with pytest.raises(WorldContactError, match="within"):
        scaled_contact_gate(scale)


def test_output_pair_is_exclusive_and_writes_both_payloads(tmp_path: Path) -> None:
    contacts = tmp_path / "out" / "contacts.json"
    audit = tmp_path / "out" / "audit.json"

    emitted = audit_world_contacts._write_output_pair(
        contacts, {"kind": "contacts"}, audit, {"kind": "audit"}
    )

    assert emitted == (contacts, audit)
    assert contacts.read_text(encoding="utf-8") == '{"kind":"contacts"}\n'
    assert audit.read_text(encoding="utf-8") == '{"kind":"audit"}\n'
    with pytest.raises(
        audit_world_contacts.WorldContactCliError, match="refusing to replace"
    ):
        audit_world_contacts._write_output_pair(
            contacts, {"changed": True}, tmp_path / "other.json", {"audit": True}
        )
    assert contacts.read_text(encoding="utf-8") == '{"kind":"contacts"}\n'


def test_output_pair_rejects_same_path_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "same.json"

    with pytest.raises(
        audit_world_contacts.WorldContactCliError, match="must be different paths"
    ):
        audit_world_contacts._write_output_pair(
            output, {"contacts": True}, tmp_path / "." / "same.json", {"audit": True}
        )

    assert not output.exists()


@pytest.mark.parametrize("symlink_index", (0, 1))
def test_output_pair_rejects_terminal_symlinks(
    tmp_path: Path, symlink_index: int
) -> None:
    target = tmp_path / "owned.json"
    target.write_text("owned\n", encoding="utf-8")
    paths = [tmp_path / "contacts.json", tmp_path / "audit.json"]
    paths[symlink_index].symlink_to(target)

    with pytest.raises(
        audit_world_contacts.WorldContactCliError, match="terminal symbolic link"
    ):
        audit_world_contacts._write_output_pair(
            paths[0], {"contacts": True}, paths[1], {"audit": True}
        )

    assert target.read_text(encoding="utf-8") == "owned\n"
    assert paths[symlink_index].is_symlink()
    assert not paths[1 - symlink_index].exists()


def test_output_pair_removes_first_output_when_second_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contacts = tmp_path / "contacts.json"
    audit = tmp_path / "audit.json"
    real_write = audit_world_contacts._exclusive_write_payload
    calls = 0

    def fail_second(path: Path, payload: bytes) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise audit_world_contacts.WorldContactCliError("fixture second failure")
        return real_write(path, payload)

    monkeypatch.setattr(audit_world_contacts, "_exclusive_write_payload", fail_second)

    with pytest.raises(
        audit_world_contacts.WorldContactCliError, match="fixture second failure"
    ):
        audit_world_contacts._write_output_pair(
            contacts, {"contacts": True}, audit, {"audit": True}
        )

    assert not contacts.exists()
    assert not audit.exists()
