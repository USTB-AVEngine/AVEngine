from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


_UNEXPANDED_ENV = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable UTF-8 representation used by AVEngine hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {resolved}")
    return value


def write_json(path: str | Path, value: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, *, relative_to: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    base = Path(relative_to).resolve()
    return {
        "path": resolved.relative_to(base).as_posix(),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def resolve_declared_path(
    raw_path: str,
    *,
    manifest_dir: str | Path,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a manifest path without silently accepting an unset variable.

    Relative paths are confined to the directory containing the declaring
    manifest. Absolute paths are allowed for pinned external datasets.
    """

    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Declared path must be a non-empty string")

    env = dict(os.environ if environment is None else environment)
    expanded = raw_path
    for name, value in env.items():
        expanded = expanded.replace(f"${{{name}}}", value)
        expanded = re.sub(rf"\${re.escape(name)}(?![A-Za-z0-9_])", value, expanded)

    unresolved = _UNEXPANDED_ENV.search(expanded)
    if unresolved:
        raise ValueError(
            f"Environment variable in path is not set: {unresolved.group(0)}"
        )

    candidate = Path(expanded).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    base = Path(manifest_dir).resolve()
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"Relative path escapes manifest directory: {raw_path}"
        ) from exc
    return resolved
