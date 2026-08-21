#!/usr/bin/env python3
"""Finalize and verify a native SPEAR pixel-capture artifact inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "avengine_qa_native_spear_pixel_episode_v1"
DEPTH_KEYS = {
    "normal_depth_m",
    "target_only_source1_depth_m",
    "target_only_source2_depth_m",
}
MASK_KEYS = {
    "depth_derived_modal_semantic",
    "modal_visible_source1",
    "modal_visible_source2",
    "target_only_source1",
    "target_only_source2",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"artifact file is missing: {path}")
    return {
        "kind": "file",
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _directory_record(path: Path) -> dict[str, Any]:
    _require(path.is_dir(), f"artifact directory is missing: {path}")
    inventory = [
        {
            "relative_path": str(file.relative_to(path)),
            "size_bytes": file.stat().st_size,
            "sha256": _sha256(file),
        }
        for file in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    _require(inventory, f"artifact directory is empty: {path}")
    return {
        "kind": "directory",
        "path": str(path.resolve()),
        "file_count": len(inventory),
        "total_size_bytes": sum(item["size_bytes"] for item in inventory),
        "inventory": inventory,
        "inventory_root_sha256": _canonical_json_sha256(inventory),
    }


def finalize(manifest_path: Path) -> Mapping[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema") == SCHEMA and manifest.get("status") == "pass",
        "capture manifest is not a passing native SPEAR pixel Episode",
    )
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, Mapping), "capture manifest lacks artifacts")
    paths = {name: Path(path).resolve() for name, path in artifacts.items()}
    _require(
        {"metric_depth", "pixel_masks", "rgb_frames", "object_id_descriptors"}
        <= set(paths),
        "capture manifest lacks required native artifact paths",
    )
    with np.load(paths["metric_depth"]) as payload:
        _require(
            set(payload.files) == DEPTH_KEYS,
            "metric-depth NPZ does not have the exact contracted keys",
        )
    with np.load(paths["pixel_masks"]) as payload:
        _require(
            set(payload.files) == MASK_KEYS,
            "pixel-mask NPZ does not have the exact contracted keys",
        )
    records = {
        name: (_directory_record(path) if path.is_dir() else _file_record(path))
        for name, path in paths.items()
    }
    manifest["artifact_records"] = records
    manifest["sha256"] = {
        name: record["sha256"]
        for name, record in records.items()
        if record["kind"] == "file"
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = finalize(args.manifest.resolve())
    print(
        json.dumps(
            {
                "status": "pass",
                "artifact_count": len(manifest["artifact_records"]),
                "file_hash_count": len(manifest["sha256"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
