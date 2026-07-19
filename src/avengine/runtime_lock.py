"""Resolve milestone-specific historical runtime profiles.

The repository root ``runtime.lock.yaml`` is intentionally only a lightweight
Git-tracked index.  M1--M4 compatibility profiles retain the exact historical
bytes needed to verify already-recorded native evidence without making their
old test summaries the current project state.
"""

from __future__ import annotations

from pathlib import Path
import re


class RuntimeLockError(ValueError):
    """The runtime profile index or selected profile is invalid."""


_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _load_index(path: Path) -> dict[str, str] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeLockError(f"unable to read runtime lock {path}: {exc}") from exc
    schema_lines = [line for line in text.splitlines() if line.startswith("schema_version:")]
    if schema_lines != ["schema_version: 2"]:
        # Unit fixtures and checkouts before the profile-index migration are
        # complete legacy locks and remain valid inputs in their own right.
        return None
    role_lines = [line for line in text.splitlines() if line.startswith("role:")]
    if role_lines != ["role: runtime_profile_index"]:
        raise RuntimeLockError(
            "schema_version 2 runtime lock must use role runtime_profile_index"
        )
    lines = text.splitlines()
    if lines.count("profiles:") != 1:
        raise RuntimeLockError("runtime profile index must contain one profiles mapping")
    in_profiles = False
    current_profile: str | None = None
    profiles: dict[str, str] = {}
    for line in lines:
        if line == "profiles:":
            in_profiles = True
            current_profile = None
            continue
        if in_profiles and line and not line[0].isspace():
            break
        if not in_profiles or not line or line.lstrip().startswith("#"):
            continue
        profile_match = re.fullmatch(r"  ([a-z0-9][a-z0-9._-]*):", line)
        if profile_match is not None:
            current_profile = profile_match.group(1)
            continue
        path_match = re.fullmatch(r"    path: (\S+)", line)
        if path_match is not None:
            if current_profile is None or current_profile in profiles:
                raise RuntimeLockError("runtime profile index has a misplaced/duplicate path")
            profiles[current_profile] = path_match.group(1)
    if not profiles:
        raise RuntimeLockError("runtime profile index lacks profile paths")
    return profiles


def resolve_runtime_profile(
    repository_root: str | Path,
    profile_id: str,
) -> Path:
    """Return the confined historical profile selected by the root index.

    A legacy, non-index ``runtime.lock.yaml`` resolves to itself.  This keeps
    hermetic fixtures and older checkouts compatible while current code uses
    explicit milestone profiles.
    """

    root = Path(repository_root).resolve(strict=True)
    index_path = root / "runtime.lock.yaml"
    if not index_path.is_file():
        raise RuntimeLockError(f"runtime lock is missing: {index_path}")
    index = _load_index(index_path)
    if index is None:
        return index_path

    if _PROFILE_ID.fullmatch(profile_id) is None:
        raise RuntimeLockError(f"runtime profile ID is invalid: {profile_id!r}")
    relative = index.get(profile_id)
    if relative is None:
        raise RuntimeLockError(f"runtime profile is not declared: {profile_id}")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or "\\" in relative:
        raise RuntimeLockError(
            f"runtime profile {profile_id} path is not confined: {relative}"
        )

    cursor = root
    for part in raw.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeLockError(
                f"runtime profile {profile_id} path traverses a symlink: {relative}"
            )
    try:
        selected = (root / raw).resolve(strict=True)
        selected.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RuntimeLockError(
            f"unable to resolve runtime profile {profile_id}: {exc}"
        ) from exc
    if not selected.is_file():
        raise RuntimeLockError(
            f"runtime profile {profile_id} is not a regular file: {selected}"
        )
    return selected
