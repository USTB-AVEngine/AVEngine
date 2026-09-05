#!/usr/bin/env python3
"""Load a task-configured sound event pool and require each audio file to exist.

The three-asset canary registry in examples/registry stays untouched.  A task
points at an external pool or registry through params; this module resolves
each clip to a real file and reports missing files instead of silently
substituting another asset.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


class EventPoolError(ValueError):
    """A configured event pool cannot be used as written."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EventPoolError(f"cannot read event pool {path}: {exc}") from exc


def _library_root(params: Mapping[str, Any], pool_path: Path | None = None) -> Path | None:
    raw = params.get("SOUND_EVENT_LIBRARY_ROOT")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    if pool_path is None:
        return None
    payload = _read_json(pool_path)
    if isinstance(payload, Mapping):
        for key in ("library_root", "output_root"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value).expanduser().resolve()
        manifest = payload.get("source_manifest")
        if isinstance(manifest, str) and manifest.strip():
            source = Path(manifest).expanduser()
            if source.is_file():
                recorded = _read_json(source)
                if isinstance(recorded, Mapping):
                    for key in ("output_root", "library_root"):
                        value = recorded.get(key)
                        if isinstance(value, str) and value.strip():
                            return Path(value).expanduser().resolve()
    return pool_path.parent.resolve()


def audio_path_from_row(
    row: Mapping[str, Any],
    *,
    library_root: Path | None = None,
) -> Path | None:
    """Return the on-disk wav for one pool or registry row, if declared."""

    dry = row.get("dry_audio")
    if isinstance(dry, Mapping):
        uri = dry.get("uri") or dry.get("path")
        if isinstance(uri, str) and uri.strip():
            text = uri.strip()
            if text.startswith("file://"):
                parsed = urlparse(text)
                return Path(unquote(parsed.path))
            path = Path(text).expanduser()
            if path.is_absolute():
                return path
            if library_root is not None:
                return (library_root / path).resolve()
            return path
    for key in ("prepared", "path", "audio_path", "wav"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value.strip()).expanduser()
        if path.is_absolute():
            return path
        if library_root is not None:
            return (library_root / path).resolve()
        return path
    return None


def _as_sound_type(row: Mapping[str, Any]) -> dict[str, Any]:
    taxonomy = row.get("taxonomy_path")
    label = None
    if isinstance(taxonomy, list) and taxonomy:
        label = taxonomy[-1]
    if not isinstance(label, str) or not label.strip():
        label = row.get("semantic_sound_class") or row.get("event_class")
    if not isinstance(label, str) or not label.strip():
        raise EventPoolError("event-pool row has no semantic sound class")
    asset_id = row.get("sound_asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise EventPoolError("event-pool row has no sound_asset_id")
    result = dict(row)
    result["sound_asset_id"] = asset_id.strip()
    result["semantic_sound_class"] = str(
        row.get("semantic_sound_class") or row.get("event_class") or label
    ).strip()
    result["taxonomy_path"] = list(taxonomy) if isinstance(taxonomy, list) and taxonomy else [label.strip()]
    return result


def rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        if isinstance(payload.get("clips"), list):
            rows = payload["clips"]
        elif isinstance(payload.get("sound_assets"), list):
            rows = payload["sound_assets"]
        else:
            raise EventPoolError("event pool has neither clips nor sound_assets")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise EventPoolError("event pool must be an object or a list")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise EventPoolError(f"event pool row {index} is not an object")
        result.append(dict(row))
    return result


def load_event_pool_rows(params: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Path | None]:
    """Load the task-configured pool.  Empty params yield an empty row list."""

    pool_raw = params.get("SOUND_EVENT_POOL")
    registry_raw = params.get("SOUND_EVENT_REGISTRY") or params.get("TASK_SOUND_ASSET_REGISTRY")
    path = None
    if isinstance(pool_raw, str) and pool_raw.strip():
        path = Path(pool_raw).expanduser().resolve()
    elif isinstance(registry_raw, str) and registry_raw.strip():
        path = Path(registry_raw).expanduser().resolve()
    if path is None:
        return [], None
    if not path.is_file():
        raise EventPoolError(f"configured event pool is missing: {path}")
    rows = rows_from_payload(_read_json(path))
    return rows, path


def require_existing_audio_files(
    rows: Sequence[Mapping[str, Any]],
    *,
    library_root: Path | None,
    owner: str,
) -> list[dict[str, Any]]:
    """Return copies with audio_path set; raise if any file is missing."""

    missing: list[str] = []
    resolved: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        path = audio_path_from_row(item, library_root=library_root)
        asset_id = str(item.get("sound_asset_id") or "<unknown>")
        if path is None:
            missing.append(f"{asset_id}: no audio path")
            continue
        if not path.is_file():
            missing.append(f"{asset_id}: {path}")
            continue
        item["audio_path"] = str(path)
        resolved.append(item)
    if missing:
        preview = ", ".join(missing[:8])
        extra = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise EventPoolError(
            f"{owner} audio files are missing ({len(missing)}): {preview}{extra}"
        )
    return resolved


def distinct_sound_types(
    rows: Sequence[Mapping[str, Any]],
    *,
    required: int,
    preferred_classes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Pick one existing clip per semantic class, preferring an explicit class list."""

    if isinstance(required, bool) or not isinstance(required, int) or required < 1:
        raise EventPoolError("required sound type count must be a positive integer")
    by_label: dict[str, dict[str, Any]] = {}
    for row in rows:
        typed = _as_sound_type(row)
        label = typed["taxonomy_path"][-1]
        by_label.setdefault(label, typed)
    order: list[str] = []
    if preferred_classes:
        for name in preferred_classes:
            if not isinstance(name, str) or not name.strip():
                raise EventPoolError("preferred sound classes must be non-empty strings")
            order.append(name.strip())
    for label in by_label:
        if label not in order:
            order.append(label)
    selected: list[dict[str, Any]] = []
    for label in order:
        item = by_label.get(label)
        if item is None:
            continue
        selected.append(item)
        if len(selected) >= required:
            return selected
    raise EventPoolError(
        f"event pool has {len(by_label)} semantic sound types, need {required}"
    )


def configured_sound_types(params: Mapping[str, Any], *, required: int) -> list[dict[str, Any]]:
    """Task-configured card12 types, each with a real audio file."""

    rows, pool_path = load_event_pool_rows(params)
    if not rows:
        return []
    library_root = _library_root(params, pool_path)
    existing = require_existing_audio_files(
        rows, library_root=library_root, owner="event pool"
    )
    preferred = params.get("CARD12_SOUND_EVENT_CLASSES") or params.get("SOUND_EVENT_CLASSES")
    if preferred is not None and (
        not isinstance(preferred, Sequence) or isinstance(preferred, (str, bytes))
    ):
        raise EventPoolError("CARD12_SOUND_EVENT_CLASSES must be a list of class names")
    return distinct_sound_types(
        existing,
        required=required,
        preferred_classes=list(preferred) if preferred is not None else None,
    )
