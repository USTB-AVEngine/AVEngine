#!/usr/bin/env python3
"""Fail-closed archival for prepared attempts that never reached a GPU launch.

Launchers supply their schema, exact request shape, and path bindings through
``PreparedAttemptSpec``.  This module deliberately knows nothing about rooms,
revisions, or episode IDs, so old and new launchers can share one ledger rule.
"""

from __future__ import annotations

import json
import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avengine.security.path_policy import (
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
)

ARCHIVE_RECEIPT_SCHEMA = "avengine_pre_gpu_prepared_attempt_archive_v1"
ARCHIVE_RECEIPT_STATUS = "pre_gpu_archive_intent"
DEFAULT_FORBIDDEN_ENTRY_NAMES = frozenset(
    {
        "dry_run_receipt.json",
        "running_receipt.json",
        "final_receipt.json",
        "capture_stdout.log",
        "capture_stderr.log",
        "stdout.log",
        "stderr.log",
    }
)


class PreGpuLaunchLedgerError(RuntimeError):
    """Raised when a prepared attempt is not safe to archive."""


@dataclass(frozen=True)
class PreservedFileIdentity:
    """Expected immutable bytes for one launcher-specific CPU artifact."""

    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        if self.byte_size < 0:
            raise ValueError("preserved file byte_size must be non-negative")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("preserved file sha256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class PreparedAttemptSpec:
    """Launcher-owned contract for one unlaunched prepared request.

    ``request_keys`` closes the JSON shape exactly.  ``expected_fields`` binds
    non-path values, while ``expected_paths`` compares canonical absolute paths.
    Capture directories and any other evidence outside the attempt directory
    belong in ``forbidden_paths``.
    """

    request_schema: str
    request_keys: frozenset[str]
    workspace_roots: tuple[Path, ...]
    expected_fields: Mapping[str, Any] = field(default_factory=dict)
    expected_paths: Mapping[str, Path] = field(default_factory=dict)
    forbidden_paths: tuple[Path, ...] = ()
    preserved_files: Mapping[str, PreservedFileIdentity] = field(default_factory=dict)
    request_status: str = "prepared_not_launched"
    request_filename: str = "request.json"
    receipt_filename: str = "pre_gpu_archive_receipt.json"
    forbidden_entry_names: frozenset[str] = DEFAULT_FORBIDDEN_ENTRY_NAMES

    def __post_init__(self) -> None:
        if not self.request_schema:
            raise ValueError("request_schema must be non-empty")
        if not self.workspace_roots:
            raise ValueError("workspace_roots must be non-empty")
        if not self.request_keys:
            raise ValueError("request_keys must close a non-empty request shape")
        if "schema" not in self.request_keys or "status" not in self.request_keys:
            raise ValueError("request_keys must include schema and status")
        for name in (self.request_filename, self.receipt_filename):
            if not name or Path(name).name != name:
                raise ValueError("ledger filenames must be plain basenames")
        if self.request_filename == self.receipt_filename:
            raise ValueError("request and receipt filenames must differ")
        for name in self.preserved_files:
            if (
                not name
                or Path(name).name != name
                or name in {self.request_filename, self.receipt_filename}
            ):
                raise ValueError("preserved file names must be distinct basenames")
        unknown = (set(self.expected_fields) | set(self.expected_paths)) - set(
            self.request_keys
        )
        if unknown:
            raise ValueError(
                f"bindings are absent from request_keys: {sorted(unknown)}"
            )
        overlap = set(self.expected_fields) & set(self.expected_paths)
        if overlap:
            raise ValueError(
                f"fields cannot be both scalar and path bindings: {sorted(overlap)}"
            )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreGpuLaunchLedgerError(message)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _exists_or_symlink(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _load_request(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreGpuLaunchLedgerError(f"cannot read prepared request: {exc}") from exc
    _require(isinstance(value, dict), "prepared request must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_preserved_files(root: Path, spec: PreparedAttemptSpec) -> None:
    for name, identity in spec.preserved_files.items():
        path = root / name
        _require(path.is_file(), f"preserved file is missing or not regular: {name}")
        _require(
            path.stat().st_size == identity.byte_size,
            f"preserved file byte size drift: {name}",
        )
        _require(_sha256(path) == identity.sha256, f"preserved file SHA drift: {name}")


def _path_policy(spec: PreparedAttemptSpec) -> WorkspacePathPolicy:
    try:
        return WorkspacePathPolicy.from_roots(spec.workspace_roots)
    except (FileNotFoundError, PathPolicyError, ValueError) as exc:
        raise PreGpuLaunchLedgerError(f"invalid ledger workspace roots: {exc}") from exc


def _resolve_existing_directory(
    policy: WorkspacePathPolicy, path: Path, *, owner: str
) -> Path:
    if path.is_symlink():
        raise PreGpuLaunchLedgerError(f"{owner} must not be a symlink")
    try:
        return policy.resolve_input(path, owner=owner, kind="directory")
    except (FileNotFoundError, PathPolicyError, ValueError) as exc:
        raise PreGpuLaunchLedgerError(f"invalid {owner}: {exc}") from exc


def _resolve_absent_output(
    policy: WorkspacePathPolicy, path: Path, *, owner: str
) -> Path:
    if path.is_symlink():
        raise PreGpuLaunchLedgerError(f"{owner} must not be a symlink")
    try:
        return policy.resolve_output(path, owner=owner)
    except FileExistsError as exc:
        raise PreGpuLaunchLedgerError(f"{owner} destination already exists") from exc
    except (PathPolicyError, ValueError) as exc:
        raise PreGpuLaunchLedgerError(f"invalid {owner}: {exc}") from exc


def _validate_request(request: Mapping[str, Any], spec: PreparedAttemptSpec) -> None:
    _require(
        set(request) == set(spec.request_keys),
        "prepared request keys do not match the exact contract",
    )
    _require(request.get("schema") == spec.request_schema, "request schema drift")
    _require(request.get("status") == spec.request_status, "request status drift")

    for key, expected in spec.expected_fields.items():
        _require(request.get(key) == expected, f"request field drift: {key}")
    for key, expected in spec.expected_paths.items():
        value = request.get(key)
        _require(isinstance(value, str) and value, f"request path is invalid: {key}")
        _require(
            Path(value).expanduser().resolve(strict=False)
            == Path(expected).expanduser().resolve(strict=False),
            f"request path drift: {key}",
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_prepared_attempt(
    *, attempt_root: Path, spec: PreparedAttemptSpec
) -> dict[str, Any]:
    """Validate that an attempt contains only one exact, unlaunched request."""

    policy = _path_policy(spec)
    attempt_root = _resolve_existing_directory(
        policy, _absolute(attempt_root), owner="prepared attempt directory"
    )

    entries = list(attempt_root.iterdir())
    symlinks = sorted(entry.name for entry in entries if entry.is_symlink())
    _require(not symlinks, f"prepared attempt contains symlinks: {symlinks}")
    names = {entry.name for entry in entries}
    forbidden = sorted(names & set(spec.forbidden_entry_names))
    _require(not forbidden, f"launch evidence exists: {forbidden}")
    expected_names = {spec.request_filename, *spec.preserved_files}
    _require(
        names == expected_names,
        f"prepared attempt entries are not closed: {sorted(names)}",
    )

    request_path = attempt_root / spec.request_filename
    _require(request_path.is_file(), "prepared request is not a regular file")
    request = _load_request(request_path)
    _validate_request(request, spec)
    _verify_preserved_files(attempt_root, spec)

    for forbidden_path in spec.forbidden_paths:
        forbidden_path = _absolute(forbidden_path)
        if not _exists_or_symlink(forbidden_path):
            _resolve_absent_output(
                policy, forbidden_path, owner="forbidden launch or capture path"
            )
        _require(
            not _exists_or_symlink(forbidden_path),
            f"launch or capture evidence exists: {_absolute(forbidden_path)}",
        )
    return request


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PreGpuLaunchLedgerError(
            f"archive receipt already exists: {path}"
        ) from exc


def archive_prepared_attempt(
    *,
    attempt_root: Path,
    archive_root: Path,
    spec: PreparedAttemptSpec,
    reason: str,
) -> Path:
    """Archive one verified request via a no-clobber sibling rename.

    The receipt names both the original path and the post-rename archive path.
    It explicitly records that neither a launch GPU query nor a GPU child began.
    """

    _require(bool(reason.strip()), "archive reason must be non-empty")
    policy = _path_policy(spec)
    requested_attempt_root = _absolute(attempt_root)
    requested_archive_root = _absolute(archive_root)
    _require(
        requested_attempt_root.parent == requested_archive_root.parent,
        "archive must be a sibling of the prepared attempt",
    )
    attempt_root = _resolve_existing_directory(
        policy, requested_attempt_root, owner="prepared attempt directory"
    )
    archive_root = _resolve_absent_output(
        policy, requested_archive_root, owner="prepared attempt archive"
    )
    _require(attempt_root != archive_root, "archive path must differ from attempt path")
    _require(
        attempt_root.parent == archive_root.parent,
        "archive must be a sibling of the prepared attempt",
    )
    request = verify_prepared_attempt(attempt_root=attempt_root, spec=spec)

    original_request = attempt_root / spec.request_filename
    archived_request = archive_root / spec.request_filename
    receipt_path = attempt_root / spec.receipt_filename
    archived_receipt = archive_root / spec.receipt_filename
    payload = {
        "schema": ARCHIVE_RECEIPT_SCHEMA,
        "status": ARCHIVE_RECEIPT_STATUS,
        "archive_publication_state_at_write": "pending_atomic_no_replace",
        "archive_publication_required": True,
        "request_schema": request["schema"],
        "request_status": request["status"],
        "reason": reason,
        "requested_attempt_root": str(requested_attempt_root),
        "original_attempt_root": str(attempt_root),
        "archive_root": str(archive_root),
        "original_request_path": str(original_request),
        "archived_request_path": str(archived_request),
        "preserved_file_records": {
            name: {
                "path": str(archive_root / name),
                "byte_size": identity.byte_size,
                "sha256": identity.sha256,
            }
            for name, identity in sorted(spec.preserved_files.items())
        },
        "preserved_embedded_paths_rehomed_by_archive": bool(spec.preserved_files),
        "embedded_paths_non_authoritative_after_archive": bool(spec.preserved_files),
        "capture_launch_gpu_query_started": False,
        "gpu_query_started": False,
        "gpu_started": False,
        "attempt_consumed": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_exclusive(receipt_path, payload)
    _fsync_directory(attempt_root)
    try:
        atomic_publish_directory(policy, attempt_root, archive_root)
    except (FileExistsError, PathPolicyError, OSError) as exc:
        raise PreGpuLaunchLedgerError(
            f"atomic archive publication failed: {exc}"
        ) from exc
    _fsync_directory(archive_root.parent)
    verify_preparation_archive(
        archive_root=archive_root,
        original_attempt_root=attempt_root,
        spec=spec,
        require_original_absent=True,
    )
    return archived_receipt


def verify_preparation_archive(
    *,
    archive_root: Path,
    original_attempt_root: Path,
    spec: PreparedAttemptSpec,
    require_original_absent: bool = False,
) -> dict[str, Any]:
    """Reopen a published archive and prove its request/receipt closure."""

    policy = _path_policy(spec)
    archive_root = _resolve_existing_directory(
        policy, _absolute(archive_root), owner="prepared attempt archive"
    )
    original_attempt_root = _absolute(original_attempt_root)
    if require_original_absent:
        _require(
            not _exists_or_symlink(original_attempt_root),
            "original prepared attempt still exists after archival",
        )
    entries = list(archive_root.iterdir())
    _require(
        not any(entry.is_symlink() for entry in entries),
        "prepared attempt archive contains symlinks",
    )
    _require(
        {entry.name for entry in entries}
        == {spec.request_filename, spec.receipt_filename, *spec.preserved_files},
        "prepared attempt archive entries are not closed",
    )
    request_path = archive_root / spec.request_filename
    receipt_path = archive_root / spec.receipt_filename
    request = _load_request(request_path)
    _validate_request(request, spec)
    _verify_preserved_files(archive_root, spec)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreGpuLaunchLedgerError(f"cannot read archive receipt: {exc}") from exc
    _require(isinstance(receipt, dict), "archive receipt must be a JSON object")
    expected = {
        "schema": ARCHIVE_RECEIPT_SCHEMA,
        "status": ARCHIVE_RECEIPT_STATUS,
        "archive_publication_state_at_write": "pending_atomic_no_replace",
        "archive_publication_required": True,
        "original_attempt_root": str(original_attempt_root),
        "archive_root": str(archive_root),
        "original_request_path": str(original_attempt_root / spec.request_filename),
        "archived_request_path": str(request_path),
        "preserved_file_records": {
            name: {
                "path": str(archive_root / name),
                "byte_size": identity.byte_size,
                "sha256": identity.sha256,
            }
            for name, identity in sorted(spec.preserved_files.items())
        },
        "preserved_embedded_paths_rehomed_by_archive": bool(spec.preserved_files),
        "embedded_paths_non_authoritative_after_archive": bool(spec.preserved_files),
        "capture_launch_gpu_query_started": False,
        "gpu_query_started": False,
        "gpu_started": False,
        "attempt_consumed": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    exact_keys = set(expected) | {
        "request_schema",
        "request_status",
        "reason",
        "requested_attempt_root",
        "archived_at_utc",
    }
    _require(set(receipt) == exact_keys, "archive receipt keys are not closed")
    for key, value in expected.items():
        _require(receipt.get(key) == value, f"archive receipt field drift: {key}")
    _require(
        receipt.get("request_schema") == request["schema"]
        and receipt.get("request_status") == request["status"],
        "archive receipt request identity drift",
    )
    _require(
        isinstance(receipt.get("reason"), str) and bool(receipt["reason"].strip()),
        "archive receipt reason is invalid",
    )
    requested_root = receipt.get("requested_attempt_root")
    _require(
        isinstance(requested_root, str)
        and Path(requested_root).expanduser().resolve(strict=False)
        == original_attempt_root.resolve(strict=False),
        "archive receipt requested attempt path drift",
    )
    archived_at = receipt.get("archived_at_utc")
    _require(isinstance(archived_at, str), "archive receipt timestamp is invalid")
    try:
        timestamp = datetime.fromisoformat(archived_at)
    except ValueError as exc:
        raise PreGpuLaunchLedgerError("archive receipt timestamp is invalid") from exc
    _require(timestamp.tzinfo is not None, "archive receipt timestamp lacks timezone")
    return receipt
