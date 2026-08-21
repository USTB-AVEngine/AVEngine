"""Resolve the versioned Habitat-native workspace configuration.

Environment values override matching entries in ``paths.yaml``. Relative
defaults are resolved from the repository root, never from the caller's cwd.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "paths.yaml"


def _load_document(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("schema") != "avengine_workspace_paths_v2":
        raise ValueError("paths config must use avengine_workspace_paths_v2")
    entries = value.get("paths")
    if not isinstance(entries, dict):
        raise ValueError("paths config must contain a paths mapping")
    return value


def load_paths_dict(
    yaml_path: str | Path = DEFAULT_CONFIG,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return configured environment variables as canonical absolute paths."""

    document = _load_document(yaml_path)
    env = dict(os.environ if environment is None else environment)
    result: dict[str, str] = {}
    for name, record in document["paths"].items():
        if not isinstance(name, str) or not name.startswith("AVENGINE_"):
            raise ValueError(f"invalid path environment name: {name!r}")
        if not isinstance(record, dict):
            raise ValueError(f"path record must be an object: {name}")
        raw = env.get(name, record.get("default"))
        if raw is None or raw == "":
            continue
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = REPOSITORY_ROOT / candidate
        result[name] = str(candidate.resolve(strict=False))
    return result


def _git_checkout_ancestor(path: Path) -> Path | None:
    """Return a containing Git marker without invoking Git itself."""

    candidate = path.resolve(strict=False)
    while True:
        marker = candidate / ".git"
        if marker.exists() or marker.is_symlink():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent


def load_paths_env(
    yaml_path: str | Path = DEFAULT_CONFIG,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    resolved = load_paths_dict(yaml_path, environment=environment)
    os.environ.update(resolved)
    return resolved


def validate_paths(
    layers: list[str],
    yaml_path: str | Path = DEFAULT_CONFIG,
) -> tuple[bool, list[dict[str, Any]]]:
    document = _load_document(yaml_path)
    resolved = load_paths_dict(yaml_path)
    checks: list[dict[str, Any]] = []
    for name, record in document["paths"].items():
        required = bool(set(record.get("required_for", [])) & set(layers))
        path_value = resolved.get(name)
        exists = bool(path_value and Path(path_value).exists())
        kind = record.get("kind")
        kind_ok = exists and (
            kind == "output_directory" or Path(path_value).is_dir()
        )
        must_not_be_git_checkout = record.get("must_not_be_git_checkout", False)
        if not isinstance(must_not_be_git_checkout, bool):
            raise ValueError(
                f"must_not_be_git_checkout must be boolean for path record: {name}"
            )
        checkout_root = (
            _git_checkout_ancestor(Path(path_value))
            if path_value and must_not_be_git_checkout
            else None
        )
        outside_checkout = checkout_root is None
        status = "pass" if (not required or kind_ok) and outside_checkout else "fail"
        reason = None
        if checkout_root is not None:
            reason = "inside_git_checkout"
        elif required and not kind_ok:
            reason = "missing_or_wrong_kind"
        checks.append(
            {
                "name": name,
                "path": path_value,
                "required": required,
                "kind": kind,
                "must_not_be_git_checkout": must_not_be_git_checkout,
                "checkout_root": str(checkout_root) if checkout_root else None,
                "reason": reason,
                "status": status,
            }
        )
    return all(item["status"] == "pass" for item in checks), checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--layer",
        action="append",
        default=[],
        help="validate paths required by this layer (default: fast_unit)",
    )
    args = parser.parse_args()
    if args.validate:
        ok, checks = validate_paths(args.layer or ["fast_unit"], args.config)
        for check in checks:
            if check["status"] == "pass":
                marker = "OK"
            elif check["reason"] == "inside_git_checkout":
                marker = "REJECTED"
            else:
                marker = "MISSING"
            requirement = "required" if check["required"] else "optional"
            detail = (
                f" reason={check['reason']} checkout_root={check['checkout_root']}"
                if check["reason"] == "inside_git_checkout"
                else ""
            )
            print(
                f"[paths] {marker} {check['name']}={check['path']} "
                f"({requirement}){detail}"
            )
        return 0 if ok else 1
    values = load_paths_dict(args.config)
    for name, value in values.items():
        if args.export:
            print(f"export {name}={shlex.quote(value)}")
        else:
            print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
