from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from avengine.security.path_policy import (
    STRICT_UNTRUSTED_LINUX,
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
    write_bytes_no_clobber,
)


def test_policy_rejects_missing_input_and_root_escape(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    policy = WorkspacePathPolicy.from_roots([root])

    with pytest.raises(PathPolicyError, match="does not exist"):
        policy.resolve_input(root / "missing.json")
    secret = outside / "secret.json"
    secret.write_text("outside\n", encoding="utf-8")
    with pytest.raises(PathPolicyError, match="escapes declared"):
        policy.resolve_input(secret)


def test_internal_symlink_is_allowed_but_escape_symlink_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    real = root / "versioned" / "dataset"
    real.mkdir(parents=True)
    value = real / "value.bin"
    value.write_bytes(b"inside")
    (root / "active").symlink_to(real, target_is_directory=True)
    policy = WorkspacePathPolicy.from_roots([root])
    assert policy.resolve_input(root / "active" / "value.bin") == value.resolve()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "value.bin").write_bytes(b"outside")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathPolicyError, match="escapes declared"):
        policy.resolve_input(root / "escape" / "value.bin")


def test_file_hash_is_checked_against_real_bytes(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "source.bin"
    source.write_bytes(b"authenticated")
    policy = WorkspacePathPolicy.from_roots([root])
    expected = hashlib.sha256(b"authenticated").hexdigest()
    assert policy.resolve_input(source, expected_sha256=expected) == source
    with pytest.raises(PathPolicyError, match="SHA-256 mismatch"):
        policy.resolve_input(source, expected_sha256="0" * 64)


def test_output_is_no_clobber_and_atomic_file_write_reads_back(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    policy = WorkspacePathPolicy.from_roots([root])
    output = write_bytes_no_clobber(policy, root / "evidence" / "record.json", b"{}\n")
    assert output.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_bytes_no_clobber(policy, output, b"replacement")
    assert output.read_bytes() == b"{}\n"


def test_directory_publish_is_sibling_atomic_no_replace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    policy = WorkspacePathPolicy.from_roots([root])
    staging = root / ".run.staging"
    staging.mkdir()
    (staging / "final_status.json").write_text("{}\n", encoding="utf-8")
    published = atomic_publish_directory(policy, staging, root / "run")
    assert (published / "final_status.json").is_file()
    another = root / ".run2.staging"
    another.mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        atomic_publish_directory(policy, another, root / "run")


def test_strict_untrusted_linux_is_explicitly_unimplemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="openat2"):
        WorkspacePathPolicy.from_roots([tmp_path], mode=STRICT_UNTRUSTED_LINUX)
