from __future__ import annotations

from pathlib import Path

import pytest

from avengine.runtime_lock import RuntimeLockError, resolve_runtime_profile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_root_runtime_lock_is_a_minimal_profile_index() -> None:
    text = (REPOSITORY_ROOT / "runtime.lock.yaml").read_text(encoding="utf-8")
    assert "schema_version: 2" in text
    assert "role: runtime_profile_index" in text
    assert "verification:" not in text
    assert "_sha256:" not in text
    for profile_id in ("m1", "m2", "m3", "m4"):
        assert f"  {profile_id}:\n" in text


@pytest.mark.parametrize(
    ("profile_id", "filename"),
    [
        ("m1", "m1_runtime_v1.yaml"),
        ("m2", "m2_runtime_v1.yaml"),
        ("m3", "m3_runtime_v1.yaml"),
        ("m4", "m4_runtime_v1.json"),
    ],
)
def test_profiles_resolve_to_confined_historical_files(
    profile_id: str, filename: str
) -> None:
    selected = resolve_runtime_profile(REPOSITORY_ROOT, profile_id)
    assert selected == REPOSITORY_ROOT / "locks" / filename
    assert selected.is_file()


def test_legacy_runtime_lock_fixture_resolves_to_itself(tmp_path: Path) -> None:
    lock = tmp_path / "runtime.lock.yaml"
    lock.write_text(
        "runtime_test_environment:\n  required_m2_native_binding_sha256: "
        + "a" * 64
        + "\n",
        encoding="utf-8",
    )
    assert resolve_runtime_profile(tmp_path, "m2") == lock


def test_profile_index_rejects_root_escape(tmp_path: Path) -> None:
    (tmp_path / "runtime.lock.yaml").write_text(
        "schema_version: 2\n"
        "role: runtime_profile_index\n"
        "profiles:\n"
        "  m3:\n"
        "    path: ../outside.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeLockError, match="not confined"):
        resolve_runtime_profile(tmp_path, "m3")


def test_profile_index_rejects_missing_profile(tmp_path: Path) -> None:
    (tmp_path / "runtime.lock.yaml").write_text(
        "schema_version: 2\n"
        "role: runtime_profile_index\n"
        "profiles:\n"
        "  m1:\n"
        "    path: locks/m1.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeLockError, match="not declared"):
        resolve_runtime_profile(tmp_path, "m3")
