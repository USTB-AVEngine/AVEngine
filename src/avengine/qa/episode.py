"""Unified QA Episode data structure wrapping Timeline v2 with structured facts.

An Episode records everything needed to verify a QA pair:
- which entities and sounds were used
- the room and furniture configuration
- the authoritative Timeline v2
- per-frame sound, spatial, motion, and visibility facts
- visibility/occlusion events (enter_frustum, occlusion_start, reappear, etc.)
- validated question/answer pairs
- sidecar file references

The module is pure: builders return detached JSON-compatible values and never
write files.  It follows the same fail-closed, content-hash-bound contract
patterns used by M5/M6.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import datetime
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import canonical_json_sha256, load_json
from avengine.m5.timeline import (
    FRAME_COUNT,
    AUDIO_SAMPLE_COUNT,
    DURATION_TICKS,
    TIME_BASE_HZ,
    TICKS_PER_FRAME,
    validate_timeline_semantics,
)

QA_EPISODE_SCHEMA = "avengine_qa_episode_v1"

# ── visibility states ────────────────────────────────────────────────────────

VISIBILITY_OUT_OF_VIEW = "out_of_view"
VISIBILITY_CLEAR = "visible_clear"
VISIBILITY_OCCLUDED = "visible_occluded"
VISIBILITY_FULLY_OCCLUDED = "fully_occluded"
VISIBILITY_STATES = (
    VISIBILITY_OUT_OF_VIEW,
    VISIBILITY_CLEAR,
    VISIBILITY_OCCLUDED,
    VISIBILITY_FULLY_OCCLUDED,
)

# ── event types ──────────────────────────────────────────────────────────────

EVENT_ENTER_FRUSTUM = "enter_frustum"
EVENT_EXIT_FRUSTUM = "exit_frustum"
EVENT_BECOME_VISIBLE = "become_visible"
EVENT_OCCLUSION_START = "occlusion_start"
EVENT_FULLY_OCCLUDED = "fully_occluded"
EVENT_REAPPEAR = "reappear"
EVENT_TYPES = (
    EVENT_ENTER_FRUSTUM,
    EVENT_EXIT_FRUSTUM,
    EVENT_BECOME_VISIBLE,
    EVENT_OCCLUSION_START,
    EVENT_FULLY_OCCLUDED,
    EVENT_REAPPEAR,
)

# ── motion states ────────────────────────────────────────────────────────────

MOTION_IDLE = "idle"
MOTION_WALK = "walk"
MOTION_OTHER = "other"
MOTION_STATES = (MOTION_IDLE, MOTION_WALK, MOTION_OTHER)

# ── occluder types ───────────────────────────────────────────────────────────

OCCLUDER_ACTOR = "actor"
OCCLUDER_FURNITURE = "furniture"
OCCLUDER_UNKNOWN = "unknown_static"
OCCLUDER_TYPES = (OCCLUDER_ACTOR, OCCLUDER_FURNITURE, OCCLUDER_UNKNOWN)

# ── thresholds ───────────────────────────────────────────────────────────────

DEFAULT_CLEAR_THRESHOLD = 0.90
DEFAULT_VISIBLE_THRESHOLD = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VisibilityRecord:
    """Per-frame, per-actor visibility measurement."""

    amodal_pixels: int
    visible_pixels: int
    visible_fraction: float
    visibility_state: str
    touches_frame_border: bool
    bbox_visible: tuple[int, int, int, int] | None = None
    occluders: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "amodal_pixels": self.amodal_pixels,
            "visible_pixels": self.visible_pixels,
            "visible_fraction": round(self.visible_fraction, 6),
            "visibility_state": self.visibility_state,
            "touches_frame_border": self.touches_frame_border,
        }
        if self.bbox_visible is not None:
            result["bbox_visible"] = {
                "x_min": self.bbox_visible[0],
                "y_min": self.bbox_visible[1],
                "x_max": self.bbox_visible[2],
                "y_max": self.bbox_visible[3],
            }
        if self.occluders:
            result["occluders"] = list(self.occluders)
        return result


@dataclass(frozen=True)
class EpisodeEvent:
    """A discrete visibility/occlusion event at a specific frame."""

    event_type: str
    frame_index: int
    actor_id: str
    occluder: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_type": self.event_type,
            "frame_index": self.frame_index,
            "actor_id": self.actor_id,
        }
        if self.occluder is not None:
            result["occluder"] = deepcopy(self.occluder)
        return result


@dataclass(frozen=True)
class QAPair:
    """A validated question/answer pair with provenance."""

    question_id: str
    question_type: str
    question_text: str
    answer_text: str
    answer_unique: bool
    fact_observable: bool
    choices: tuple[str, ...] | None = None
    answer_source: dict[str, Any] | None = None
    distractor_check: str | None = None
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "answer_text": self.answer_text,
            "validation": {
                "answer_unique": self.answer_unique,
                "fact_observable": self.fact_observable,
            },
        }
        if self.choices:
            result["choices"] = list(self.choices)
        if self.answer_source is not None:
            result["answer_source"] = deepcopy(self.answer_source)
        if self.distractor_check:
            result["validation"]["distractor_check"] = self.distractor_check
        if self.rejection_reason:
            result["validation"]["rejection_reason"] = self.rejection_reason
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Schema helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _schema_path() -> Path:
    source = _repository_root() / "src" / "avengine" / "qa" / "schemas" / "qa_episode_v1.schema.json"
    if source.is_file():
        return source
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / "qa_episode_v1.schema.json"
    if installed.is_file():
        return installed
    raise FileNotFoundError("qa_episode_v1 schema is unavailable")


def _load_schema() -> dict[str, Any]:
    return load_json(_schema_path())


def validate_qa_episode_schema(value: Any) -> list[str]:
    """Validate that *value* conforms to the qa_episode_v1 JSON Schema."""
    schema = _load_schema()
    errors: list[str] = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# Content hash
# ═══════════════════════════════════════════════════════════════════════════════


def _all_numbers_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    return False


def _bind_content_hash(episode: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(episode)
    result.pop("episode_content_sha256", None)
    result["episode_content_sha256"] = canonical_json_sha256(result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Episodic class
# ═══════════════════════════════════════════════════════════════════════════════


class EpisodeError(ValueError):
    """One or more Episode invariants failed."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass
class Episode:
    """A mutable builder for a QA Episode.

    Call :meth:`build` to produce the validated, hash-bound JSON document.
    """

    episode_id: str
    created: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    question_spec_id: str | None = None

    # asset references
    actors: list[dict[str, Any]] = field(default_factory=list)
    sounds: list[dict[str, Any]] = field(default_factory=list)

    # scene
    room_id: str = "apartment_0000"
    room_provider: str = "replicacad"
    furniture_occluders: list[dict[str, Any]] = field(default_factory=list)

    # timeline (populated at build time or injected)
    timeline: dict[str, Any] | None = None

    # facts (populated incrementally)
    sound_facts: list[dict[str, Any]] = field(default_factory=list)
    spatial_frames: list[dict[str, Any]] = field(default_factory=list)
    motion_frames: list[dict[str, Any]] = field(default_factory=list)
    visibility_frames: list[dict[str, Any]] = field(default_factory=list)
    events: list[EpisodeEvent] = field(default_factory=list)

    # qa
    qa_pairs: list[QAPair] = field(default_factory=list)

    # sidecars
    rgb_video: str = ""
    semantic_video: str = ""
    depth_frames: str = ""
    target_only_masks: str = ""
    audio_mix_binaural: str = ""
    audio_mix_foa: str = ""
    visibility_overlay: str = ""

    # provenance
    seed: int = 0
    avengine_commit: str = ""
    habitat_commit: str = ""
    asset_registry_sha256: str = ""
    sound_registry_sha256: str = ""

    # ── actor management ───────────────────────────────────────────────

    def add_actor(
        self,
        actor_id: str,
        entity_asset_id: str,
        species_id: str,
        semantic_id: int,
        *,
        breed_id: str | None = None,
        size: str | None = None,
        body_build: str | None = None,
        life_stage: str | None = None,
        coat_profile_id: str | None = None,
        coat_value: str | None = None,
        top_color: str | None = None,
        emitter_anchor_id: str = "muzzle",
        emitter_anchor_type: str = "muzzle",
        emitter_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Episode:
        identity: dict[str, Any] = {"species_id": species_id}
        if breed_id is not None:
            identity["breed_id"] = breed_id

        attrs: dict[str, Any] = {}
        if size is not None:
            attrs["size"] = size
        if body_build is not None:
            attrs["body_build"] = body_build
        if life_stage is not None:
            attrs["life_stage"] = life_stage
        if coat_profile_id is not None and coat_value is not None:
            attrs["coat_profile"] = {"profile_id": coat_profile_id, "value": coat_value}
        if top_color is not None:
            attrs["clothing"] = {"top_color": top_color}

        actor: dict[str, Any] = {
            "actor_id": actor_id,
            "entity_asset_id": entity_asset_id,
            "identity": identity,
            "semantic_id": semantic_id,
            "emitter_anchor": {
                "anchor_id": emitter_anchor_id,
                "anchor_type": emitter_anchor_type,
                "offset_m": list(emitter_offset_m),
            },
        }
        if attrs:
            actor["realized_visual_attributes"] = attrs

        self.actors.append(actor)
        return self

    # ── sound management ───────────────────────────────────────────────

    def add_sound(
        self,
        sound_asset_id: str,
        sound_class: str,
        bound_to_actor: str,
        *,
        sound_category: str = "",
        transcript: str = "",
        audio_program_id: str = "",
        duration_samples: int = 0,
    ) -> Episode:
        sound: dict[str, Any] = {
            "sound_asset_id": sound_asset_id,
            "semantic_sound_class": sound_class,
            "bound_to_actor": bound_to_actor,
        }
        if sound_category:
            sound["sound_category"] = sound_category
        if transcript:
            sound["transcript"] = transcript
        if audio_program_id:
            sound["audio_program_id"] = audio_program_id
        if duration_samples > 0:
            sound["duration_samples"] = duration_samples
        self.sounds.append(sound)
        return self

    # ── sound facts ────────────────────────────────────────────────────

    def add_sound_fact(
        self,
        event_id: str,
        actor_id: str,
        sound_asset_id: str,
        start_tick: int,
        end_tick: int,
        *,
        transcript: str = "",
    ) -> Episode:
        start_frame = start_tick // TICKS_PER_FRAME
        end_frame = (end_tick + TICKS_PER_FRAME - 1) // TICKS_PER_FRAME
        start_sample = (start_tick * AUDIO_SAMPLE_COUNT + DURATION_TICKS // 2) // DURATION_TICKS
        end_sample = (end_tick * AUDIO_SAMPLE_COUNT + DURATION_TICKS // 2) // DURATION_TICKS
        fact: dict[str, Any] = {
            "event_id": event_id,
            "actor_id": actor_id,
            "sound_asset_id": sound_asset_id,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "start_frame": max(0, min(FRAME_COUNT - 1, start_frame)),
            "end_frame": max(1, min(FRAME_COUNT, end_frame)),
            "start_sample": max(0, min(AUDIO_SAMPLE_COUNT - 1, start_sample)),
            "end_sample": max(1, min(AUDIO_SAMPLE_COUNT, end_sample)),
        }
        if transcript:
            fact["transcript"] = transcript
        self.sound_facts.append(fact)
        return self

    # ── spatial facts ──────────────────────────────────────────────────

    def set_spatial_frames(self, frames: list[dict[str, Any]]) -> Episode:
        self.spatial_frames = list(frames)
        return self

    # ── motion facts ───────────────────────────────────────────────────

    def set_motion_frames(self, frames: list[dict[str, Any]]) -> Episode:
        self.motion_frames = list(frames)
        return self

    # ── visibility facts ───────────────────────────────────────────────

    def set_visibility_frames(self, frames: list[dict[str, Any]]) -> Episode:
        self.visibility_frames = list(frames)
        return self

    def add_visibility_record(
        self,
        frame_index: int,
        actor_id: str,
        record: VisibilityRecord,
    ) -> Episode:
        while len(self.visibility_frames) <= frame_index:
            self.visibility_frames.append({"frame_index": len(self.visibility_frames), "actor_visibility": {}})
        self.visibility_frames[frame_index]["actor_visibility"][actor_id] = record.as_dict()
        return self

    # ── events ─────────────────────────────────────────────────────────

    def add_event(self, event: EpisodeEvent) -> Episode:
        self.events.append(event)
        return self

    # ── QA ─────────────────────────────────────────────────────────────

    def add_qa(self, qa: QAPair) -> Episode:
        self.qa_pairs.append(qa)
        return self

    # ── furniture ──────────────────────────────────────────────────────

    def add_furniture_occluder(
        self,
        instance_id: str,
        semantic_label: str,
        semantic_id: int,
        *,
        reliable_semantic_mapping: bool = True,
    ) -> Episode:
        self.furniture_occluders.append({
            "instance_id": instance_id,
            "semantic_label": semantic_label,
            "semantic_id": semantic_id,
            "reliable_semantic_mapping": reliable_semantic_mapping,
        })
        return self

    # ── build ──────────────────────────────────────────────────────────

    def build(self) -> dict[str, Any]:
        """Produce the validated, hash-bound QA Episode document.

        Returns:
            A detached JSON-compatible ``dict``.

        Raises:
            EpisodeError: if any structural or semantic invariant fails.
        """
        if self.timeline is None:
            raise EpisodeError(["timeline must be set before build()"])

        if not self.actors:
            raise EpisodeError(["at least one actor is required"])

        doc: dict[str, Any] = {
            "schema": QA_EPISODE_SCHEMA,
            "episode_id": self.episode_id,
            "created": self.created,
            "assets_used": {
                "actors": deepcopy(self.actors),
                "sounds": deepcopy(self.sounds),
            },
            "scene": {
                "room_id": self.room_id,
                "room_provider": self.room_provider,
            },
            "timeline": deepcopy(self.timeline),
            "facts": {
                "sound_facts": deepcopy(self.sound_facts),
                "spatial_facts": {"per_frame": deepcopy(self.spatial_frames)},
                "motion_facts": {"per_frame": deepcopy(self.motion_frames)},
                "visibility_facts": {"per_frame": deepcopy(self.visibility_frames)},
                "events": [e.as_dict() for e in self.events],
            },
            "qa_pairs": [qa.as_dict() for qa in self.qa_pairs],
            "sidecars": {
                "rgb_video": self.rgb_video,
                "semantic_video": self.semantic_video,
                "depth_frames": self.depth_frames,
                "target_only_masks": self.target_only_masks,
                "audio_mix_binaural": self.audio_mix_binaural,
                "audio_mix_foa": self.audio_mix_foa,
                "visibility_overlay": self.visibility_overlay,
            },
            "provenance": {
                "seed": self.seed,
            },
        }

        if self.question_spec_id:
            doc["question_spec_id"] = self.question_spec_id

        if self.furniture_occluders:
            doc["scene"]["furniture_occluders"] = deepcopy(self.furniture_occluders)

        if self.avengine_commit:
            doc["provenance"]["avengine_commit"] = self.avengine_commit
        if self.habitat_commit:
            doc["provenance"]["habitat_commit"] = self.habitat_commit
        if self.asset_registry_sha256:
            doc["provenance"]["asset_registry_sha256"] = self.asset_registry_sha256
        if self.sound_registry_sha256:
            doc["provenance"]["sound_registry_sha256"] = self.sound_registry_sha256

        doc = _bind_content_hash(doc)

        errors = validate_qa_episode(doc)
        if errors:
            raise EpisodeError(errors)

        return doc


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════


def validate_qa_episode(episode: Any) -> list[str]:
    """Return every structural and semantic error in *episode*.

    This covers JSON Schema validation plus cross-field checks that schema
    alone cannot express (timeline compatibility, actor/sound cross-references,
    frame counts, uniqueness, finiteness).
    """
    if not isinstance(episode, Mapping):
        return ["QA Episode must be a mapping"]

    errors: list[str] = []
    errors.extend(validate_qa_episode_schema(episode))
    if not _all_numbers_finite(episode):
        errors.append("QA Episode contains a non-finite number")

    # content hash
    payload = {k: v for k, v in episode.items() if k != "episode_content_sha256"}
    declared = episode.get("episode_content_sha256")
    expected = canonical_json_sha256(payload)
    if declared != expected:
        errors.append("episode_content_sha256 does not match canonical content")

    # timeline cross-validation
    timeline = episode.get("timeline")
    if isinstance(timeline, Mapping):
        timeline_errors = validate_timeline_semantics(timeline)
        errors.extend(f"timeline: {e}" for e in timeline_errors)

        # frame count agreement
        tv = timeline.get("video")
        timeline_frames = timeline.get("frames")
        if isinstance(tv, Mapping) and isinstance(timeline_frames, list):
            tc = tv.get("frame_count")
            if tc != FRAME_COUNT:
                errors.append(f"timeline video.frame_count must be {FRAME_COUNT}")

            # spatial facts frame count
            sf = episode.get("facts", {}).get("spatial_facts", {}).get("per_frame")
            if isinstance(sf, list) and len(sf) != FRAME_COUNT:
                errors.append(f"spatial_facts.per_frame must have {FRAME_COUNT} entries")

            # motion facts frame count
            mf = episode.get("facts", {}).get("motion_facts", {}).get("per_frame")
            if isinstance(mf, list) and len(mf) != FRAME_COUNT:
                errors.append(f"motion_facts.per_frame must have {FRAME_COUNT} entries")

            # visibility facts frame count
            vf = episode.get("facts", {}).get("visibility_facts", {}).get("per_frame")
            if isinstance(vf, list) and len(vf) != FRAME_COUNT:
                errors.append(f"visibility_facts.per_frame must have {FRAME_COUNT} entries")

    # actor cross-references
    assets = episode.get("assets_used")
    if isinstance(assets, Mapping):
        actor_list = assets.get("actors")
        if isinstance(actor_list, list):
            actor_ids = {a.get("actor_id") for a in actor_list if isinstance(a, Mapping)}
            sound_list = assets.get("sounds")
            if isinstance(sound_list, list):
                for i, snd in enumerate(sound_list):
                    if isinstance(snd, Mapping) and snd.get("bound_to_actor") not in actor_ids:
                        errors.append(f"assets_used.sounds[{i}].bound_to_actor does not resolve")

    # QA validation
    qa_pairs = episode.get("qa_pairs")
    if isinstance(qa_pairs, list):
        seen_ids: set[str] = set()
        for i, qa in enumerate(qa_pairs):
            if not isinstance(qa, Mapping):
                continue
            qid = qa.get("question_id")
            if isinstance(qid, str):
                if qid in seen_ids:
                    errors.append(f"qa_pairs[{i}].question_id is not unique")
                seen_ids.add(qid)
            val = qa.get("validation")
            if isinstance(val, Mapping):
                if val.get("answer_unique") is not True:
                    errors.append(f"qa_pairs[{i}] answer is not unique: {val.get('rejection_reason', 'no reason')}")
                if val.get("fact_observable") is not True:
                    errors.append(f"qa_pairs[{i}] fact is not observable: {val.get('rejection_reason', 'no reason')}")

    return _dedupe(errors)


def _dedupe(errors: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(errors))


# ═══════════════════════════════════════════════════════════════════════════════
# Visibility classification
# ═══════════════════════════════════════════════════════════════════════════════


def classify_visibility(
    amodal_pixels: int,
    visible_pixels: int,
    in_frustum: bool,
    *,
    clear_threshold: float = DEFAULT_CLEAR_THRESHOLD,
    visible_threshold: float = DEFAULT_VISIBLE_THRESHOLD,
) -> str:
    """Classify visibility into one of four canonical states.

    Args:
        amodal_pixels: pixel count from the target-only (unoccluded) pass.
        visible_pixels: pixel count from the normal multi-object semantic pass.
        in_frustum: whether the target's bounding sphere intersects the frustum.
        clear_threshold: fraction above which the target is ``visible_clear``.
        visible_threshold: fraction below which (but still in frustum) the
            target is ``fully_occluded``.

    Returns:
        One of ``"out_of_view"``, ``"visible_clear"``, ``"visible_occluded"``,
        or ``"fully_occluded"``.
    """
    if not in_frustum:
        return VISIBILITY_OUT_OF_VIEW
    if amodal_pixels == 0:
        return VISIBILITY_OUT_OF_VIEW
    fraction = visible_pixels / amodal_pixels
    if fraction >= clear_threshold:
        return VISIBILITY_CLEAR
    if fraction >= visible_threshold:
        return VISIBILITY_OCCLUDED
    return VISIBILITY_FULLY_OCCLUDED


def make_visibility_record(
    amodal_pixels: int,
    visible_pixels: int,
    in_frustum: bool,
    *,
    touches_frame_border: bool = False,
    bbox_visible: tuple[int, int, int, int] | None = None,
    occluders: Sequence[dict[str, Any]] = (),
    clear_threshold: float = DEFAULT_CLEAR_THRESHOLD,
    visible_threshold: float = DEFAULT_VISIBLE_THRESHOLD,
) -> VisibilityRecord:
    """Create a :class:`VisibilityRecord` with automatic state classification."""
    fraction = visible_pixels / amodal_pixels if amodal_pixels > 0 else 0.0
    state = classify_visibility(
        amodal_pixels, visible_pixels, in_frustum,
        clear_threshold=clear_threshold, visible_threshold=visible_threshold,
    )
    return VisibilityRecord(
        amodal_pixels=amodal_pixels,
        visible_pixels=visible_pixels,
        visible_fraction=round(fraction, 6),
        visibility_state=state,
        touches_frame_border=touches_frame_border,
        bbox_visible=bbox_visible,
        occluders=tuple(occluders),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Event detection from per-frame visibility
# ═══════════════════════════════════════════════════════════════════════════════


def detect_visibility_events(
    visibility_frames: Sequence[dict[str, Any]],
    actor_id: str,
) -> list[EpisodeEvent]:
    """Detect visibility/occlusion events from a sequence of per-frame records.

    Args:
        visibility_frames: ordered per-frame dicts (one per frame_index).
        actor_id: the actor whose visibility records to inspect.

    Returns:
        A chronological list of :class:`EpisodeEvent` instances.
    """
    events: list[EpisodeEvent] = []
    prev_state: str | None = None
    prev_fraction: float = 1.0

    for frame in visibility_frames:
        if not isinstance(frame, Mapping):
            continue
        fi = frame.get("frame_index")
        if not isinstance(fi, int):
            continue
        av = frame.get("actor_visibility", {})
        rec = av.get(actor_id) if isinstance(av, Mapping) else None
        if not isinstance(rec, Mapping):
            continue

        state = rec.get("visibility_state", VISIBILITY_OUT_OF_VIEW)
        fraction = rec.get("visible_fraction", 1.0)

        if prev_state is not None:
            # enter / exit frustum
            if prev_state == VISIBILITY_OUT_OF_VIEW and state != VISIBILITY_OUT_OF_VIEW:
                events.append(EpisodeEvent(EVENT_ENTER_FRUSTUM, fi, actor_id))
            elif prev_state != VISIBILITY_OUT_OF_VIEW and state == VISIBILITY_OUT_OF_VIEW:
                events.append(EpisodeEvent(EVENT_EXIT_FRUSTUM, fi, actor_id))

            # become visible
            if prev_state in (VISIBILITY_OUT_OF_VIEW, VISIBILITY_FULLY_OCCLUDED) and state in (
                VISIBILITY_CLEAR,
                VISIBILITY_OCCLUDED,
            ):
                events.append(EpisodeEvent(EVENT_BECOME_VISIBLE, fi, actor_id))

            # occlusion start
            if prev_fraction >= DEFAULT_CLEAR_THRESHOLD and fraction < DEFAULT_CLEAR_THRESHOLD and state != VISIBILITY_OUT_OF_VIEW:
                occluders = rec.get("occluders")
                occluder = occluders[0] if isinstance(occluders, list) and occluders else None
                events.append(EpisodeEvent(EVENT_OCCLUSION_START, fi, actor_id, occluder=occluder))

            # fully occluded
            if prev_state != VISIBILITY_FULLY_OCCLUDED and state == VISIBILITY_FULLY_OCCLUDED:
                events.append(EpisodeEvent(EVENT_FULLY_OCCLUDED, fi, actor_id))

            # reappear
            if prev_state == VISIBILITY_FULLY_OCCLUDED and state in (
                VISIBILITY_OCCLUDED,
                VISIBILITY_CLEAR,
            ):
                events.append(EpisodeEvent(EVENT_REAPPEAR, fi, actor_id))

        prev_state = state
        prev_fraction = fraction

    return events


__all__ = [
    "Episode",
    "EpisodeError",
    "EpisodeEvent",
    "QAPair",
    "VisibilityRecord",
    "QA_EPISODE_SCHEMA",
    "VISIBILITY_OUT_OF_VIEW",
    "VISIBILITY_CLEAR",
    "VISIBILITY_OCCLUDED",
    "VISIBILITY_FULLY_OCCLUDED",
    "VISIBILITY_STATES",
    "EVENT_ENTER_FRUSTUM",
    "EVENT_EXIT_FRUSTUM",
    "EVENT_BECOME_VISIBLE",
    "EVENT_OCCLUSION_START",
    "EVENT_FULLY_OCCLUDED",
    "EVENT_REAPPEAR",
    "EVENT_TYPES",
    "MOTION_IDLE",
    "MOTION_WALK",
    "MOTION_OTHER",
    "MOTION_STATES",
    "OCCLUDER_ACTOR",
    "OCCLUDER_FURNITURE",
    "OCCLUDER_UNKNOWN",
    "OCCLUDER_TYPES",
    "classify_visibility",
    "detect_visibility_events",
    "make_visibility_record",
    "validate_qa_episode",
    "validate_qa_episode_schema",
]
