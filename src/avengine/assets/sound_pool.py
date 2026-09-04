"""Class-level sound event pools for QA scheduling.

A pool is a catalog of already-cut events (one pulse per clip). Clip
duration and source window come from the catalog, not from constants in
this module.

catalog 里的 sound_asset_id 必须是已注册的声资产，否则 program 在渲染前的
校验会被拒（events[i].sound_asset_id is not registered）。本模块不负责
登记，调用方在切库/转 catalog 时必须先把每一声写进声资产注册表。

This module does not split libraries and does not choose a source mode.
The caller reads SOUND_SOURCE_MODE from params.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from avengine.assets.sound_harvest import speech_metadata_from_mapping

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
    speaker_id: str | None = None
    utterance_id: str | None = None
    transcript: str | None = None
    split: str | None = None


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
            window = end - start
            if duration != window:
                raise SoundPoolError(
                    f"{owner} duration_samples={duration} != "
                    f"source window {end}-{start}={window}")
            metadata = speech_metadata_from_mapping(row)
            clips.append(
                PoolClip(
                    sound_asset_id=asset_id,
                    event_class=event_class,
                    duration_samples=duration,
                    sample_rate_hz=rate,
                    source_start_sample=start,
                    source_end_sample_exclusive=end,
                    speaker_id=metadata.get("speaker_id"),
                    utterance_id=metadata.get("utterance_id"),
                    transcript=metadata.get("transcript"),
                    split=metadata.get("split"),
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

    def select_distinct_speech_clips(
        self,
        rng: Any,
        count: int = 4,
        *,
        split: str | None = "train",
    ) -> list[PoolClip]:
        """Select a bounded set of complete speech clips with explicit identity.

        The pool is sorted once by duration, with seeded tie order. A selected
        clip must declare a split, speaker, utterance, and transcript, and neither
        speaker nor utterance may repeat. This is a finite filter, not a
        search over combinations.
        """

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise SoundPoolError("speech clip count must be a positive integer")
        if split is not None and (not isinstance(split, str) or not split.strip()):
            raise SoundPoolError("speech split must be a non-empty string or None")
        candidates = self.clips_for("speech_playback")
        eligible = [
            clip
            for clip in candidates
            if (split is None or clip.split == split)
            and clip.speaker_id
            and clip.utterance_id
            and clip.transcript
        ]
        # Duration-first ordering finds a feasible short set with one bounded
        # pass; it never searches combinations across the full pool.
        try:
            permutation = rng.permutation(len(eligible))
        except (AttributeError, TypeError, ValueError) as exc:
            raise SoundPoolError(
                "speech clip selection needs an RNG with permutation()"
            ) from exc
        tie_rank = {
            int(index): rank for rank, index in enumerate(permutation)
        }
        eligible = [
            clip
            for index, clip in sorted(
                enumerate(eligible),
                key=lambda pair: (
                    pair[1].duration_samples,
                    tie_rank[pair[0]],
                    pair[1].speaker_id,
                    pair[1].utterance_id,
                    pair[1].sound_asset_id,
                ),
            )
        ]
        edges: dict[str, list[tuple[str, PoolClip]]] = {}
        clip_by_pair: dict[tuple[str, str], PoolClip] = {}
        for clip in eligible:
            pair = (clip.speaker_id, clip.utterance_id)
            if pair in clip_by_pair:
                continue
            clip_by_pair[pair] = clip
            edges.setdefault(clip.speaker_id, []).append(
                (clip.utterance_id, clip)
            )

        # Match speakers to utterances with augmenting paths. Processing
        # low-degree speakers first handles constrained speakers without
        # enumerating candidate combinations.
        speaker_order = sorted(
            edges,
            key=lambda speaker: (
                len(edges[speaker]),
                min(clip.duration_samples for _, clip in edges[speaker]),
                speaker,
            ),
        )
        matched_by_utterance: dict[str, str] = {}

        def augment(speaker: str, seen: set[str]) -> bool:
            for utterance, _ in edges[speaker]:
                if utterance in seen:
                    continue
                seen.add(utterance)
                owner = matched_by_utterance.get(utterance)
                if owner is None or augment(owner, seen):
                    matched_by_utterance[utterance] = speaker
                    return True
            return False

        for speaker in speaker_order:
            if len(matched_by_utterance) == count:
                break
            augment(speaker, set())

        if len(matched_by_utterance) < count:
            raise SoundPoolError(
                f"speech pool has fewer than {count} distinct clips with explicit "
                f"speaker/utterance/transcript metadata in split={split!r}"
            )
        selected = [
            clip_by_pair[(speaker, utterance)]
            for utterance, speaker in matched_by_utterance.items()
        ]
        selected.sort(
            key=lambda clip: (
                clip.duration_samples,
                clip.speaker_id,
                clip.utterance_id,
                clip.sound_asset_id,
            )
        )
        return selected

    def draw(self, rng: Any, event_class: str) -> PoolClip:
        clips = self.clips_for(event_class)
        index = int(rng.integers(0, len(clips)))
        return clips[index]


class BoundRoleClipSource:
    """One clip per semantic role for the whole episode."""

    def __init__(self, by_role: Mapping[str, PoolClip]) -> None:
        self.by_role = dict(by_role)

    def for_role(self, role: str) -> PoolClip:
        if role not in self.by_role:
            raise SoundPoolError(f"no clip bound for role {role!r}")
        return self.by_role[role]


class ClassClipSource:
    def __init__(
        self,
        pool: SoundEventPool,
        event_class: str,
        rng: Any,
        *,
        sample_rate_hz: int,
    ) -> None:
        self.pool = pool
        self.event_class = event_class
        self.rng = rng
        self.sample_rate_hz = sample_rate_hz
        for clip in pool.clips_for(event_class):
            if clip.sample_rate_hz != sample_rate_hz:
                raise SoundPoolError(
                    f"clip {clip.sound_asset_id} sample_rate_hz="
                    f"{clip.sample_rate_hz} != SAMPLE_RATE_HZ={sample_rate_hz}")

    def next(self) -> PoolClip:
        return self.pool.draw(self.rng, self.event_class)

    def select_distinct_speech_clips(
        self,
        count: int = 4,
        *,
        split: str | None = "train",
    ) -> list[PoolClip]:
        if self.event_class != "speech_playback":
            raise SoundPoolError(
                "distinct speech selection requires event_class='speech_playback'"
            )
        return self.pool.select_distinct_speech_clips(
            self.rng, count=count, split=split
        )

    def bind_distinct_roles(self, roles: tuple[str, ...]) -> BoundRoleClipSource:
        clips = self.pool.clips_for(self.event_class)
        by_id: dict[str, PoolClip] = {}
        for clip in clips:
            by_id.setdefault(clip.sound_asset_id, clip)
        if len(by_id) < len(roles):
            raise SoundPoolError(
                f"event class {self.event_class!r} has {len(by_id)} distinct "
                f"clips, need {len(roles)} for roles {list(roles)}")
        ids = list(by_id)
        order = [ids[int(index)] for index in self.rng.permutation(len(ids))]
        chosen = order[: len(roles)]
        return BoundRoleClipSource(
            {role: by_id[clip_id] for role, clip_id in zip(roles, chosen)}
        )


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
    if "SAMPLE_RATE_HZ" not in params:
        raise SoundPoolError(
            "SOUND_SOURCE_MODE=event_pool requires SAMPLE_RATE_HZ")
    pool = SoundEventPool.from_catalog(path)
    return ClassClipSource(
        pool,
        event_class_for_pair_kind(pair_kind, params),
        rng,
        sample_rate_hz=int(params["SAMPLE_RATE_HZ"]),
    )
