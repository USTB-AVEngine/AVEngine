"""Central path policy for the controlled AVEngine research workspace.

The default policy prevents accidental root escape and replacement of
immutable evidence.  It deliberately does not claim to defend against a
malicious local process racing path resolution.  A future untrusted Linux mode
is named here so callers cannot accidentally infer that it already exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Iterable, Literal


TRUSTED_RESEARCH_WORKSPACE = "trusted_research_workspace"
STRICT_UNTRUSTED_LINUX = "strict_untrusted_linux"
TrustMode = Literal["trusted_research_workspace"]


class PathPolicyError(ValueError):
    """A path violates the declared workspace or immutable-output policy."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class FileSnapshot:
    """Exact bytes and identity checked at one trusted-workspace instant."""

    path: Path
    payload: bytes
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class WorkspacePathPolicy:
    """Canonical containment policy for one controlled research workspace."""

    roots: tuple[Path, ...]
    mode: TrustMode = TRUSTED_RESEARCH_WORKSPACE

    @classmethod
    def from_roots(
        cls,
        roots: Iterable[str | Path],
        *,
        mode: str = TRUSTED_RESEARCH_WORKSPACE,
    ) -> "WorkspacePathPolicy":
        if mode != TRUSTED_RESEARCH_WORKSPACE:
            if mode == STRICT_UNTRUSTED_LINUX:
                raise NotImplementedError(
                    "strict_untrusted_linux is reserved until an openat2-based "
                    "implementation and Linux integration suite exist"
                )
            raise PathPolicyError(f"unknown filesystem trust mode: {mode}")
        resolved: list[Path] = []
        for raw in roots:
            path = Path(raw).expanduser().resolve(strict=True)
            if not path.is_dir():
                raise PathPolicyError(f"workspace root is not a directory: {path}")
            if path not in resolved:
                resolved.append(path)
        if not resolved:
            raise PathPolicyError("at least one workspace root is required")
        return cls(tuple(resolved))

    def _absolute_candidate(self, raw: str | Path) -> Path:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.roots[0] / candidate
        return candidate

    def _require_contained(self, candidate: Path, *, owner: str) -> Path:
        if not any(_contains(root, candidate) for root in self.roots):
            roots = ", ".join(str(root) for root in self.roots)
            raise PathPolicyError(
                f"{owner} escapes declared workspace roots ({roots}): {candidate}"
            )
        return candidate

    def resolve_input(
        self,
        raw: str | Path,
        *,
        owner: str = "input",
        kind: Literal["file", "directory", "any"] = "file",
        expected_sha256: str | None = None,
        allow_empty: bool = False,
    ) -> Path:
        """Resolve an existing input and optionally authenticate file bytes."""

        try:
            candidate = self._absolute_candidate(raw).resolve(strict=True)
        except FileNotFoundError as exc:
            raise PathPolicyError(f"{owner} does not exist: {raw}") from exc
        self._require_contained(candidate, owner=owner)
        if kind == "file" and not candidate.is_file():
            raise PathPolicyError(f"{owner} is not a regular file: {candidate}")
        if kind == "directory" and not candidate.is_dir():
            raise PathPolicyError(f"{owner} is not a directory: {candidate}")
        if kind not in {"file", "directory", "any"}:
            raise PathPolicyError(f"unsupported input kind: {kind}")
        if kind == "file" and not allow_empty and candidate.stat().st_size == 0:
            raise PathPolicyError(f"{owner} is empty: {candidate}")
        if expected_sha256 is not None:
            if kind != "file":
                raise PathPolicyError("expected_sha256 is valid only for file inputs")
            if len(expected_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            ):
                raise PathPolicyError(f"{owner} expected SHA-256 is malformed")
            actual = _sha256_file(candidate)
            if actual != expected_sha256:
                raise PathPolicyError(
                    f"{owner} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
                )
        return candidate

    def snapshot_file(
        self,
        raw: str | Path,
        *,
        owner: str = "input",
        expected_sha256: str | None = None,
    ) -> FileSnapshot:
        path = self.resolve_input(
            raw,
            owner=owner,
            kind="file",
            expected_sha256=expected_sha256,
        )
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise PathPolicyError(f"{owner} changed while being snapshotted")
        return FileSnapshot(path, payload, len(payload), digest)

    def resolve_output(
        self,
        raw: str | Path,
        *,
        owner: str = "output",
        create_parent: bool = False,
    ) -> Path:
        """Resolve a new output name without replacing an existing entry."""

        unresolved = self._absolute_candidate(raw)
        if unresolved.exists() or unresolved.is_symlink():
            raise FileExistsError(f"refusing to replace immutable {owner}: {unresolved}")
        parent = unresolved.parent
        if create_parent:
            # Resolve the nearest existing ancestor first, then create only a
            # path already proven to remain under one declared root.
            pending: list[str] = []
            cursor = parent
            while not cursor.exists() and not cursor.is_symlink():
                pending.append(cursor.name)
                cursor = cursor.parent
            try:
                resolved_cursor = cursor.resolve(strict=True)
            except FileNotFoundError as exc:
                raise PathPolicyError(f"{owner} has no existing ancestor") from exc
            self._require_contained(resolved_cursor, owner=f"{owner} parent")
            for component in reversed(pending):
                resolved_cursor = resolved_cursor / component
            self._require_contained(resolved_cursor, owner=f"{owner} parent")
            resolved_cursor.mkdir(parents=True, exist_ok=True)
        try:
            resolved_parent = parent.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PathPolicyError(f"{owner} parent does not exist: {parent}") from exc
        self._require_contained(resolved_parent, owner=f"{owner} parent")
        candidate = resolved_parent / unresolved.name
        self._require_contained(candidate, owner=owner)
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(f"refusing to replace immutable {owner}: {candidate}")
        return candidate


def _rename_directory_no_replace(staging: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PathPolicyError(
            "atomic directory no-replace publication is unavailable on this platform"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            f"refusing to replace immutable evidence: {destination}"
        )
    raise PathPolicyError(
        f"atomic directory publication failed: {os.strerror(error)}"
    )


def atomic_publish_directory(
    policy: WorkspacePathPolicy,
    staging: str | Path,
    destination: str | Path,
) -> Path:
    """Publish a complete sibling staging directory with no replacement."""

    source = policy.resolve_input(staging, owner="staging directory", kind="directory")
    target = policy.resolve_output(destination, owner="evidence directory")
    if source.parent != target.parent:
        raise PathPolicyError(
            "staging and destination must be siblings on the same filesystem"
        )
    _rename_directory_no_replace(source, target)
    return target


def write_bytes_no_clobber(
    policy: WorkspacePathPolicy,
    destination: str | Path,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> Path:
    """Write immutable bytes via a sibling temporary file and hard-link commit."""

    target = policy.resolve_output(
        destination, owner="evidence file", create_parent=True
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("temporary evidence write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, target, follow_symlinks=False)
        return target
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to replace immutable evidence: {target}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
