"""Class-level sound event pools for QA scheduling.

A pool is a catalog of already-cut events (one pulse per clip). The
scheduler draws from it per event. Clip duration and source window come
from the catalog, not from constants in this module.

This module does not split libraries and does not choose a source mode.
The caller reads SOUND_SOURCE_MODE from params.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "avengine_sound_event_pool_v1"


class SoundPoolError(ValueError):
    """The catalog or params cannot produce a clip."""


@dataclass(frozen=True)
class PoolClip:
    sound_asset_id: str
    event_class: str
    duration_samples: int
    sample_rate_hz: int
    source_start_sample: int
    source_end_sample_exclusive: int


def _int_field(row: Mapping[str, Any], key: str, *, owner: str) -> int:
    if key not in row:
        raise SoundPoolError(f"{owner} missing {key}")
    return int(row[key])


class SoundEventPool:
    def __init__(self, clips: list[PoolClip], *, source: str) -> None:
        by_class: dict[str, list[PoolClip]] = {}
        for clip in clips:
            by_class.setdefault(clip.event_class, []).append(clip)
        self._by_class = by_class
        self.source = source

    @classmethod
    def from_catalog(cls, path: str | Path) -> "SoundEventPool":
        catalog_path = Path(path)
        if not catalog_path.is_file():
            raise SoundPoolError(f"sound event catalog missing: {catalog_path}")
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        schema = payload.get("schema")
        if schema != SCHEMA:
            raise SoundPoolError(
                f"{catalog_path} schema {schema!r} is not {SCHEMA}")
        rows = payload.get("clips")
        if not isinstance(rows, list) or not rows:
            raise SoundPoolError(f"{catalog_path} has no clips")
        clips: list[PoolClip] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise SoundPoolError(f"{catalog_path} clips[{index}] is not an object")
            owner = f"{catalog_path} clips[{index}]"
            if "sound_asset_id" not in row or "event_class" not in row:
                raise SoundPoolError(
                    f"{owner} needs sound_asset_id and event_class")
            asset_id = str(row["sound_asset_id"])
            event_class = str(row["event_class"])
            if not asset_id or not event_class:
                raise SoundPoolError(
                    f"{owner} needs sound_asset_id and event_class")
            rate = _int_field(row, "sample_rate_hz", owner=owner)
            duration = _int_field(row, "duration_samples", owner=owner)
            start = _int_field(row, "source_start_sample", owner=owner)
            end = _int_field(row, "source_end_sample_exclusive", owner=owner)
            if duration <= 0 or rate <= 0 or end <= start:
                raise SoundPoolError(f"{owner} has an empty source window")
            clips.append(
                PoolClip(
                    sound_asset_id=asset_id,
                    event_class=event_class,
                    duration_samples=duration,
                    sample_rate_hz=rate,
                    source_start_sample=start,
                    source_end_sample_exclusive=end,
                )
            )
        return cls(clips, source=str(catalog_path))

    def clips_for(self, event_class: str) -> list[PoolClip]:
        if event_class not in self._by_class:
            raise SoundPoolError(
                f"sound event class {event_class!r} is empty in {self.source}")
        found = self._by_class[event_class]
        if not found:
            raise SoundPoolError(
                f"sound event class {event_class!r} is empty in {self.source}")
        return list(found)

    def draw(self, rng: Any, event_class: str) -> PoolClip:
        clips = self.clips_for(event_class)
        index = int(rng.integers(0, len(clips)))
        return clips[index]


class ClassClipSource:
    def __init__(self, pool: SoundEventPool, event_class: str, rng: Any) -> None:
        self.pool = pool
        self.event_class = event_class
        self.rng = rng

    def next(self) -> PoolClip:
        return self.pool.draw(self.rng, self.event_class)


def event_class_for_pair_kind(pair_kind: str, params: Mapping[str, Any]) -> str:
    mapping = params.get("SOUND_EVENT_CLASS_BY_PAIR_KIND")
    if not isinstance(mapping, Mapping):
        raise SoundPoolError("params missing SOUND_EVENT_CLASS_BY_PAIR_KIND")
    if pair_kind not in mapping:
        raise SoundPoolError(
            f"SOUND_EVENT_CLASS_BY_PAIR_KIND has no entry for {pair_kind!r}")
    value = str(mapping[pair_kind])
    if not value:
        raise SoundPoolError(f"empty event class for pair_kind {pair_kind!r}")
    return value


def clip_source_from_params(
    params: Mapping[str, Any], rng: Any, *, pair_kind: str
) -> ClassClipSource | None:
    if "SOUND_SOURCE_MODE" not in params:
        raise SoundPoolError("params missing SOUND_SOURCE_MODE")
    mode = str(params["SOUND_SOURCE_MODE"])
    if mode == "dry_canvas_window":
        for key in (
            "SOUND_ASSET",
            "EVENT_SECONDS",
            "SAMPLE_RATE_HZ",
            "DRY_CANVAS_SOURCE_START_SAMPLE",
            "DRY_CANVAS_SOURCE_END_SAMPLE_EXCLUSIVE",
        ):
            if key not in params:
                raise SoundPoolError(
                    f"SOUND_SOURCE_MODE=dry_canvas_window requires {key}")
        return None
    if mode != "event_pool":
        raise SoundPoolError(f"unknown SOUND_SOURCE_MODE {mode!r}")
    if "SOUND_EVENT_POOL" not in params or not params["SOUND_EVENT_POOL"]:
        raise SoundPoolError(
            "SOUND_SOURCE_MODE=event_pool requires SOUND_EVENT_POOL")
    path = params["SOUND_EVENT_POOL"]
    pool = SoundEventPool.from_catalog(path)
    return ClassClipSource(
        pool, event_class_for_pair_kind(pair_kind, params), rng
    )
