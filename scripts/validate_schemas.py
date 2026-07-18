#!/usr/bin/env python3
"""Validate every versioned AVEngine JSON Schema and hash the schema set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

from avengine.release import (
    ReleaseManifestError,
    build_file_record,
    canonical_file_record_set_sha256,
    load_json_strict,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_ROOT = REPOSITORY_ROOT / "schemas"


def validate_schema_directory(schema_root: str | Path) -> dict[str, Any]:
    """Return a deterministic validation report for all ``*.json`` schemas."""

    root = Path(schema_root)
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    schema_ids: dict[str, str] = {}

    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        errors.append(f"unable to resolve schema directory {root}: {exc}")
        resolved_root = root.resolve(strict=False)

    if not resolved_root.is_dir():
        errors.append(f"schema root is not a directory: {resolved_root}")
        paths: list[Path] = []
    else:
        paths = sorted(
            path for path in resolved_root.rglob("*.json") if path.is_file()
        )
        if not paths:
            errors.append(f"schema directory contains no JSON schemas: {resolved_root}")

    for path in paths:
        relative = path.relative_to(resolved_root).as_posix()
        try:
            schema = load_json_strict(path)
        except ReleaseManifestError as exc:
            errors.extend(f"{relative}: {error}" for error in exc.errors)
            continue

        dialect = schema.get("$schema")
        if dialect != "https://json-schema.org/draft/2020-12/schema":
            errors.append(
                f"{relative}: $schema must be the Draft 2020-12 canonical URI"
            )
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id:
            errors.append(f"{relative}: missing non-empty $id")
        elif schema_id in schema_ids:
            errors.append(
                f"{relative}: duplicate $id {schema_id!r} also used by "
                f"{schema_ids[schema_id]}"
            )
        else:
            schema_ids[schema_id] = relative

        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:  # jsonschema exposes several schema exceptions
            errors.append(f"{relative}: invalid Draft 2020-12 schema: {exc}")

        try:
            records.append(
                build_file_record(path, root=resolved_root, root_id="avengine")
            )
        except ReleaseManifestError as exc:
            errors.extend(f"{relative}: {error}" for error in exc.errors)

    set_sha256: str | None = None
    if records:
        try:
            set_sha256 = canonical_file_record_set_sha256(records)
        except ReleaseManifestError as exc:
            errors.extend(exc.errors)

    return {
        "schema": "avengine_schema_validation_v1",
        "status": "pass" if not errors else "fail",
        "schema_root": str(resolved_root),
        "schema_count": len(records),
        "set_sha256": set_sha256,
        "files": records,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-root", type=Path, default=DEFAULT_SCHEMA_ROOT)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit one-line JSON instead of an indented report",
    )
    args = parser.parse_args(argv)
    report = validate_schema_directory(args.schema_root)
    print(
        json.dumps(
            report,
            indent=None if args.compact else 2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
