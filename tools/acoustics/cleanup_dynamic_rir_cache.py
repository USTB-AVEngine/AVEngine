#!/usr/bin/env python3
"""Clear completed numeric RIR payloads while retaining replayable evidence.

This explicit lifecycle operation accepts one cache path and one or more
durable reports that reference it, verifies the cache is complete, and by
default performs a dry run. With execute enabled it removes only numeric RIR
payloads, then writes cleanup_record.json beside the retained manifest,
index, receipt, and historical references. An active or incomplete dependency
blocks cleanup; an old completed report reference is retained as history.
Geometry OBJ files and failed caches are never selected by filename or deleted
by this tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.acoustics.dynamic_cache import (
    DYNAMIC_RIR_CACHE_SCHEMA,
    DynamicRIRCacheError,
    load_dynamic_rir_cache,
)
from avengine.acoustics.rir_cache import (
    RIR_CACHE_INDEX_SCHEMA,
    RIR_CACHE_RECEIPT_SCHEMA,
    RIR_CACHE_REQUEST_SCHEMA,
)


SCHEMA = "avengine_dynamic_rir_cache_cleanup_v2"


class DynamicRIRCacheCleanupError(ValueError):
    """The requested cache cleanup is not safe to perform."""


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DynamicRIRCacheCleanupError(f"cannot read JSON {path}: {exc}") from exc


def _contains_path(value: Any, target: str) -> bool:
    if isinstance(value, str):
        try:
            return str(Path(value).expanduser().resolve()) == target
        except (OSError, RuntimeError):
            return value == target
    if isinstance(value, Mapping):
        return any(_contains_path(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_path(item, target) for item in value)
    return False



_INCOMPLETE_STATUS_VALUES = {
    "active",
    "in_progress",
    "incomplete",
    "pending",
    "queued",
    "running",
    "started",
}


def _has_incomplete_dependency(value: Any) -> bool:
    """Return true only for explicit active/incomplete dependency markers."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).lower()
            if (
                normalized_key in {
                    "status",
                    "state",
                    "dependency_status",
                    "lifecycle_status",
                    "cache_status",
                }
                and isinstance(item, str)
                and item.strip().lower() in _INCOMPLETE_STATUS_VALUES
            ):
                return True
            if (
                normalized_key
                in {"complete", "completed", "dependency_complete", "full_plan_complete", "ready"}
                and item is False
            ):
                return True
            if _has_incomplete_dependency(item):
                return True
    elif isinstance(value, list):
        return any(_has_incomplete_dependency(item) for item in value)
    return False


def _owned_cache_path(cache_root: str | Path, owned_root: str | Path) -> tuple[Path, Path]:
    requested_cache = Path(cache_root).expanduser()
    requested_owned = Path(owned_root).expanduser()
    if requested_cache.is_symlink():
        raise DynamicRIRCacheCleanupError("cache root must not be a symlink")
    cache = requested_cache.resolve()
    owned = requested_owned.resolve()
    if not owned.is_dir():
        raise DynamicRIRCacheCleanupError(f"owned root is unavailable: {owned}")
    try:
        relative = cache.relative_to(owned)
    except ValueError as exc:
        raise DynamicRIRCacheCleanupError(
            "cache root is outside the explicitly owned task root"
        ) from exc
    if not relative.parts:
        raise DynamicRIRCacheCleanupError("refusing to clean the owned root itself")
    if not cache.is_dir():
        raise DynamicRIRCacheCleanupError(f"cache root is unavailable: {cache}")
    return cache, owned


def _file_bytes(paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += int(path.stat().st_size)
        except OSError as exc:
            raise DynamicRIRCacheCleanupError(f"cannot stat cache file {path}: {exc}") from exc
    return total


def _cache_payload(cache: Path) -> tuple[str, list[Path], list[Path], dict[str, Any]]:
    """Return format, numeric payload files, preserved files, and metadata."""

    if (cache / "FAILED.json").exists():
        raise DynamicRIRCacheCleanupError(
            "cache has FAILED.json; failed现场 must be retained and not cleaned"
        )
    pair_index_path = cache / "pair_sequence_index.json"
    if pair_index_path.is_file():
        index = _load(pair_index_path)
        pairs = index.get("pairs") if isinstance(index, Mapping) else None
        if (
            not isinstance(index, Mapping)
            or index.get("kind") != "dynamic_rir_existing_cache_pair_sequence"
            or index.get("status") != "pass"
            or not isinstance(pairs, list)
            or not pairs
        ):
            raise DynamicRIRCacheCleanupError("pair cache sequence index is incomplete")
        payload: list[Path] = []
        preserved: list[Path] = [pair_index_path]
        pair_metadata: list[dict[str, Any]] = []
        for pair_index, pair in enumerate(pairs):
            if (
                not isinstance(pair, Mapping)
                or pair.get("cache_path") != f"pair_{pair_index:02d}"
                or not isinstance(pair.get("source_ids"), list)
            ):
                raise DynamicRIRCacheCleanupError("pair cache sequence index has an invalid pair")
            pair_root = cache / pair["cache_path"]
            pair_format, pair_payload, pair_preserved, pair_info = _cache_payload(pair_root)
            if pair_format != "avengine_rlr_rir_cache_v1":
                raise DynamicRIRCacheCleanupError("pair cache must use the established RIR cache format")
            payload.extend(pair_payload)
            preserved.extend(pair_preserved)
            pair_metadata.append({"source_ids": pair["source_ids"], **pair_info})
        all_files = [path for path in cache.rglob("*") if path.is_file()]
        preserved = [path for path in all_files if path not in payload]
        return (
            "avengine_rlr_rir_cache_v1_pair_shards",
            payload,
            preserved,
            {"pair_count": len(pair_metadata), "pairs": pair_metadata},
        )
    custom_manifest = cache / "manifest.json"
    custom_payload = cache / "sequence.npz"
    if custom_manifest.is_file() and custom_payload.is_file():
        try:
            payload = load_dynamic_rir_cache(cache)
        except DynamicRIRCacheError as exc:
            raise DynamicRIRCacheCleanupError(
                f"custom dynamic cache is not a complete validated cache: {exc}"
            ) from exc
        preserved = [
            path
            for path in cache.rglob("*")
            if path.is_file() and path != custom_payload
        ]
        return (
            "avengine_dynamic_rir_sequence_cache_v1",
            [custom_payload],
            preserved,
            {
                "manifest_schema": DYNAMIC_RIR_CACHE_SCHEMA,
                "cache_identity_sha256": payload.metadata.get("cache_identity_sha256"),
            },
        )

    request_path = cache / "request.json"
    index_path = cache / "index.json"
    receipt_path = cache / "receipt.json"
    shards_root = cache / "shards"
    if request_path.is_file() and index_path.is_file() and receipt_path.is_file() and shards_root.is_dir():
        request = _load(request_path)
        index = _load(index_path)
        receipt = _load(receipt_path)
        if (
            request.get("schema") != RIR_CACHE_REQUEST_SCHEMA
            or index.get("schema") != RIR_CACHE_INDEX_SCHEMA
            or receipt.get("schema") != RIR_CACHE_RECEIPT_SCHEMA
            or request.get("status") == "fail"
            or index.get("status") != "pass"
            or receipt.get("status") != "pass"
            or receipt.get("full_plan_complete") is not True
        ):
            raise DynamicRIRCacheCleanupError(
                "established RIR cache request/index/receipt is not complete"
            )
        entries = index.get("entries")
        if not isinstance(entries, list) or not entries:
            raise DynamicRIRCacheCleanupError("established RIR cache index has no entries")
        payload: list[Path] = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("shard"), str):
                raise DynamicRIRCacheCleanupError("established RIR cache index has a malformed shard entry")
            shard = (cache / entry["shard"]).resolve()
            try:
                shard.relative_to(shards_root.resolve())
            except ValueError as exc:
                raise DynamicRIRCacheCleanupError("established RIR shard escapes shards/") from exc
            if shard.suffix != ".npz" or not shard.is_file():
                raise DynamicRIRCacheCleanupError(f"established RIR shard is unavailable: {shard}")
            if shard not in payload:
                payload.append(shard)
        preserved = [
            path
            for path in cache.rglob("*")
            if path.is_file() and path not in payload
        ]
        return (
            "avengine_rlr_rir_cache_v1",
            payload,
            preserved,
            {
                "request_identity_sha256": request.get("request_identity_sha256"),
                "index_schema": RIR_CACHE_INDEX_SCHEMA,
                "receipt_schema": RIR_CACHE_RECEIPT_SCHEMA,
            },
        )
    raise DynamicRIRCacheCleanupError(
        "cache is neither a complete dynamic sequence cache nor an established RIR cache"
    )


def cleanup(
    *,
    cache_root: str | Path,
    owned_root: str | Path,
    reports: Sequence[str | Path],
    execute: bool = False,
) -> dict[str, Any]:
    """Validate and optionally clear one explicitly owned completed cache."""

    cache, owned = _owned_cache_path(cache_root, owned_root)
    report_paths = [Path(path).expanduser().resolve() for path in reports]
    if not report_paths:
        raise DynamicRIRCacheCleanupError(
            "at least one durable report is required to establish cache ownership"
        )
    target = str(cache)
    for report in report_paths:
        if not report.is_file():
            raise DynamicRIRCacheCleanupError(f"durable report is unavailable: {report}")
        report_value = _load(report)
        if not _contains_path(report_value, target):
            raise DynamicRIRCacheCleanupError(
                f"durable report does not reference the selected cache: {report}"
            )
        if _has_incomplete_dependency(report_value):
            raise DynamicRIRCacheCleanupError(
                f"durable report has an active or incomplete dependency: {report}"
            )

    # Other reports may retain historical references to this cache. Preserve
    # and record those references instead of treating every old path as an
    # active dependency. Only an explicit active/incomplete marker blocks.
    known_reports = {path.resolve() for path in report_paths}
    unlisted_references: list[Path] = []
    incomplete_references: list[Path] = []
    for candidate in owned.rglob("*.json"):
        candidate_resolved = candidate.resolve()
        if candidate_resolved in known_reports or candidate_resolved.is_relative_to(cache):
            continue
        try:
            candidate_value = _load(candidate)
        except DynamicRIRCacheCleanupError:
            # Unparseable sidecars cannot establish a dependency for this
            # lifecycle operation and remain untouched.
            continue
        if _contains_path(candidate_value, target):
            unlisted_references.append(candidate_resolved)
            if _has_incomplete_dependency(candidate_value):
                incomplete_references.append(candidate_resolved)
    if incomplete_references:
        raise DynamicRIRCacheCleanupError(
            "cache has active or incomplete durable references: "
            + "; ".join(str(path) for path in incomplete_references)
        )
    cache_format, payload_files, preserved_files, cache_metadata = _cache_payload(cache)
    before_bytes = _file_bytes(
        [path for path in cache.rglob("*") if path.is_file()]
    )
    payload_bytes = _file_bytes(payload_files)
    metadata_bytes = before_bytes - payload_bytes
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "planned" if not execute else "pass",
        "cache_root": str(cache),
        "owned_root": str(owned),
        "cache_format": cache_format,
        "cache_metadata": cache_metadata,
        "durable_reports": [str(path) for path in report_paths],
        "historical_durable_references": [str(path) for path in unlisted_references],
        "unlisted_reference_count": len(unlisted_references),
        "numeric_payload_lifecycle": (
            "planned" if not execute else "numeric_payload_cleaned"
        ),
        "historical_cache_path": str(cache),
        "numeric_payload_files": [str(path) for path in payload_files],
        "preserved_files": [str(path.relative_to(cache)) for path in preserved_files],
        "storage": {
            "cache_bytes_before": before_bytes,
            "numeric_payload_bytes_before": payload_bytes,
            "preserved_metadata_bytes": metadata_bytes,
            "peak_bytes": before_bytes,
        },
        "action": "remove_numeric_rir_payloads_only",
        "claim_boundary": (
            "Only the explicitly selected completed task-owned numeric RIR payloads "
            "are eligible; reports, indexes, receipts, historical references, "
            "geometry OBJ and failure现场 remain."
        ),
    }
    if not execute:
        return result
    cleanup_record = cache / "cleanup_record.json"
    pending = {
        **result,
        "status": "pending",
        "numeric_payload_lifecycle": "pending",
        "pending_action": "remove_numeric_rir_payloads_only",
    }
    cleanup_record.write_text(
        json.dumps(pending, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for payload in payload_files:
        payload.unlink()
    if any(path.exists() for path in payload_files):
        raise DynamicRIRCacheCleanupError(
            "one or more numeric RIR payloads remain after cleanup"
        )
    after_bytes = _file_bytes([path for path in cache.rglob("*") if path.is_file()])
    result["storage"].update(
        {
            "cache_bytes_after": after_bytes,
            "numeric_payload_bytes_cleared": payload_bytes,
        }
    )
    result["payload_bytes_cleared"] = payload_bytes
    result["cleanup_record"] = str(cleanup_record)
    cleanup_record.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--owned-root", required=True, type=Path)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise DynamicRIRCacheCleanupError(f"refusing to overwrite: {args.output}")
    result = cleanup(
        cache_root=args.cache_root,
        owned_root=args.owned_root,
        reports=args.report,
        execute=args.execute,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "cache_format": result["cache_format"], "storage": result["storage"], "output": str(args.output.resolve())}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DynamicRIRCacheCleanupError as error:
        print(f"DYNAMIC_RIR_CACHE_CLEANUP_FAILED {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
