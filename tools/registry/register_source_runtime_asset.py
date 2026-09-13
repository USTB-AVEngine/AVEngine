#!/usr/bin/env python3
"""Register one new source runtime asset into a fresh registry copy.

The existing registry and the single asset JSON are read-only inputs. The
command validates the new record through the opt-in D1 contract, rejects a
duplicate asset_id, validates the merged registry through the compatibility
validator, and emits a new output file without modifying either input.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from avengine import runtime_profiles


class SourceRuntimeAssetRegistrationError(ValueError):
    """The new source runtime asset cannot be registered safely."""


def _read_json(path: Path, *, owner: str) -> Any:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise SourceRuntimeAssetRegistrationError(
            f"{owner} must be a regular file: {raw}"
        )
    try:
        return json.loads(raw.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceRuntimeAssetRegistrationError(
            f"{owner} is not valid JSON: {raw}: {error}"
        ) from error


def _read_mapping(path: Path, *, owner: str) -> dict[str, Any]:
    value = _read_json(path, owner=owner)
    if not isinstance(value, Mapping):
        raise SourceRuntimeAssetRegistrationError(
            f"{owner} must contain one JSON object"
        )
    return dict(value)


def merge_source_runtime_asset(
    registry: Mapping[str, Any],
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a detached registry with one new asset appended."""

    if not isinstance(registry, Mapping):
        raise SourceRuntimeAssetRegistrationError(
            "existing source registry must be a JSON object"
        )
    assets = registry.get("assets")
    if not isinstance(assets, list):
        raise SourceRuntimeAssetRegistrationError(
            "existing source registry must contain an assets list"
        )
    if not isinstance(asset, Mapping):
        raise SourceRuntimeAssetRegistrationError(
            "new source runtime asset must be a JSON object"
        )

    new_errors = runtime_profiles.validate_new_source_asset_runtime_profile(asset)
    if new_errors:
        raise SourceRuntimeAssetRegistrationError(
            "new asset D1 validation failed: " + "; ".join(new_errors)
        )
    asset_id = asset.get("asset_id")
    existing_ids = {
        str(row.get("asset_id"))
        for row in assets
        if isinstance(row, Mapping) and row.get("asset_id") is not None
    }
    if str(asset_id) in existing_ids:
        raise SourceRuntimeAssetRegistrationError(
            f"duplicate source asset_id is already registered: {asset_id!r}"
        )

    merged = deepcopy(dict(registry))
    merged["assets"] = [deepcopy(row) for row in assets]
    merged["assets"].append(deepcopy(dict(asset)))
    errors = runtime_profiles.validate_source_asset_runtime_registry(merged)
    if errors:
        raise SourceRuntimeAssetRegistrationError(
            "merged source registry compatibility validation failed: "
            + "; ".join(errors)
        )
    return merged


def _write_fresh_json(path: Path, value: Mapping[str, Any]) -> None:
    output = Path(path).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fresh output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp_{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"fresh output already exists: {output}")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def register_source_runtime_asset(
    registry_path: str | Path,
    asset_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Register one asset from files and write one fresh merged registry."""

    output = Path(output_path).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fresh output already exists: {output}")
    registry_file = Path(registry_path).expanduser().resolve()
    asset_file = Path(asset_path).expanduser().resolve()
    registry = runtime_profiles.load_source_asset_runtime_registry(registry_file)
    asset = _read_mapping(asset_file, owner="new source runtime asset")
    merged = merge_source_runtime_asset(registry, asset)
    _write_fresh_json(output, merged)
    return {
        "status": "pass",
        "output": str(output.resolve()),
        "input_registry": str(registry_file),
        "input_asset": str(asset_file),
        "registry_id": merged.get("registry_id"),
        "registry_revision": merged.get("revision"),
        "asset_id": asset.get("asset_id"),
        "asset_count": len(merged["assets"]),
        "input_asset_count": len(registry["assets"]),
        "claim_boundary": (
            "Fresh derived registry output only. Input registry and asset files "
            "were not modified."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--asset",
        "--asset-record",
        dest="asset_path",
        type=Path,
        required=True,
        help="JSON file containing exactly one new source runtime asset object",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = register_source_runtime_asset(
            args.registry,
            args.asset_path,
            args.output,
        )
    except (
        FileExistsError,
        OSError,
        SourceRuntimeAssetRegistrationError,
        runtime_profiles.RuntimeProfileError,
    ) as error:
        print(f"SOURCE_RUNTIME_ASSET_REGISTRATION_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "SOURCE_RUNTIME_ASSET_REGISTRATION_OK "
        + json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
