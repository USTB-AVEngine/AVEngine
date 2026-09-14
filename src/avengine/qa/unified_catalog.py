"""Unified QA-01..QA-25 conditions, episode evidence and question mining.

The twelve historical QuestionSpec classes remain the semantic base for the
older protocol. This module is the small adapter used by the real-room
continuation: it consumes the native episode readbacks that the room runner
already writes and derives the remaining time, motion, count and combination
questions from the same episode.

The module intentionally keeps three layers separate:

* CATALOG describes potential scene requirements before an episode is
  generated;
* normalize_episode_bundle validates and normalizes actual readbacks;
* generate_unified_questions emits a research candidate only when its
  concrete evidence is present and unique.

No answer is inferred from a missing field or from an actor filename. A
question with missing evidence is returned as deferred with a precise reason,
which lets a room scheduler distinguish an unsuitable episode from an
implementation failure.

The canonical input is a JSON object with episode_id and any combination of
plan/visual_plan, frame_readbacks, pixel_visibility_truth, audio_program and
audio_readback. The raw native shapes are intentionally accepted directly so
a caller does not need to manufacture the older bank/fact-table wrapper first.
"""

from __future__ import annotations

import copy
import math
import random
import re
import wave
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any


UNIFIED_INPUT_SCHEMA = "avengine_qa_unified_episode_input_v1"
UNIFIED_FACT_SCHEMA = "avengine_qa_unified_episode_facts_v1"
UNIFIED_ITEM_SCHEMA = "avengine_qa_unified_question_v1"
UNIFIED_OUTPUT_SCHEMA = "avengine_qa_unified_question_set_v1"
CATALOG_VERSION = "20260909"

VISIBLE_STATES = {"visible_clear", "visible_occluded"}
VISIBILITY_STATES = (
    "out_of_view",
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
)
_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_APPEARANCE_VERSION_SUFFIX = re.compile(
    r"(?:[\s._-]+(?:v|ver|version)[\s._-]*\d+)\s*$",
    re.IGNORECASE,
)


def _strip_display_version_suffix(value: Any) -> str:
    """Remove an internal version suffix from a human-facing display label."""

    if not isinstance(value, str):
        return ""
    text = value.strip()
    return _APPEARANCE_VERSION_SUFFIX.sub("", text).strip(" ._-\t")


class UnifiedQAError(ValueError):
    """A unified QA input or request cannot be represented safely."""


def _req(
    *,
    min_entities: int = 1,
    events: Mapping[str, Any] | None = None,
    appearance: Mapping[str, Any] | None = None,
    speech_content: Mapping[str, Any] | None = None,
    motion: Mapping[str, Any] | None = None,
    after_sound: Mapping[str, Any] | None = None,
    pixel_visibility: Mapping[str, Any] | None = None,
    occlusion: Mapping[str, Any] | None = None,
    entry: Mapping[str, Any] | None = None,
    distinct_sound_classes: int = 0,
    required_modalities: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one explicit potential/evidence requirement record."""

    return {
        "min_entities": int(min_entities),
        "events": dict(events or {}),
        "appearance": dict(appearance or {}),
        "speech_content": dict(speech_content or {}),
        "motion": dict(motion or {}),
        "after_sound": dict(after_sound or {}),
        "pixel_visibility": dict(pixel_visibility or {}),
        "occlusion": dict(occlusion or {}),
        "entry": dict(entry or {}),
        "distinct_sound_classes": int(distinct_sound_classes),
        "required_modalities": list(required_modalities),
    }


# The keys are deliberately the public QA numbers. Potential requirements are
# useful before capture; evidence requirements are the concrete fields a
# readback must expose before a row can be emitted.
CATALOG: tuple[dict[str, Any], ...] = (
    {
        "qa_id": "QA-01",
        "title": "外观→是否发声",
        "question_family": "appearance_to_speaking",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            events={"min_count": 0, "target_binding": True},
            appearance={"required": True, "unique_target_value": True},
            required_modalities=("video", "binaural_audio"),
        ),
        "evidence_requirements": _req(
            events={"target_event_membership": True},
            appearance={"field_and_value": True},
            required_modalities=("video", "binaural_audio"),
        ),
    },
    {
        "qa_id": "QA-02",
        "title": "声音/台词→外观",
        "question_family": "sound_to_appearance",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"min_count": 1, "unique_sound_binding": True},
            appearance={"required": True, "visible_target": True},
            speech_content={"unique_sound_or_transcript": True},
            required_modalities=("video", "binaural_audio"),
        ),
        "evidence_requirements": _req(
            events={"unique_source": True},
            appearance={"field_and_value": True},
            speech_content={"sound_or_transcript": True},
            pixel_visibility={"target_frame": True},
            required_modalities=("video", "binaural_audio"),
        ),
    },
    {
        "qa_id": "QA-03",
        "title": "谁先发声",
        "question_family": "who_spoke_first",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"min_count": 2, "unique_earliest": True},
            appearance={"candidate_labels": True},
            required_modalities=("video", "binaural_audio"),
        ),
        "evidence_requirements": _req(
            events={"min_count": 2, "unique_earliest": True},
            appearance={"candidate_labels": True},
            required_modalities=("video", "binaural_audio"),
        ),
    },
    {
        "qa_id": "QA-04",
        "title": "发声者左右",
        "question_family": "speaker_side",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            events={"min_count": 1, "query_frame": True},
            required_modalities=("binaural_audio",),
        ),
        "evidence_requirements": _req(
            events={"active_at_query": True},
            required_modalities=("binaural_audio",),
        ),
    },
    {
        "qa_id": "QA-05",
        "title": "发声区间重叠",
        "question_family": "overlapping_speech",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"min_count": 2, "intervals": True},
            required_modalities=("audio",),
        ),
        "evidence_requirements": _req(
            events={"pairwise_intervals": True},
            required_modalities=("audio",),
        ),
    },
    {
        "qa_id": "QA-06",
        "title": "发声期间是否移动",
        "question_family": "speaking_while_moving",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            events={"min_count": 1, "target_binding": True},
            motion={"stable_state_during_event": True},
            required_modalities=("video", "binaural_audio"),
        ),
        "evidence_requirements": _req(
            events={"bound_event": True},
            motion={"per_frame_moving": True, "stable_state_during_event": True},
            required_modalities=("video", "binaural_audio"),
        ),
    },
    {
        "qa_id": "QA-07",
        "title": "从哪侧入画",
        "question_family": "offscreen_to_onscreen",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            pixel_visibility={"transition": "out_of_view_to_visible"},
            entry={"side": True},
            required_modalities=("video",),
        ),
        "evidence_requirements": _req(
            pixel_visibility={"transition": "out_of_view_to_visible", "centroid": True},
            entry={"side": True},
            required_modalities=("video",),
        ),
    },
    {
        "qa_id": "QA-08",
        "title": "发声时的可见状态",
        "question_family": "occlusion_while_speaking",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            events={"min_count": 1, "query_frame": True},
            pixel_visibility={"target_frame": True},
            occlusion={"four_state": True},
            required_modalities=("video", "binaural_audio", "pixel_visibility"),
        ),
        "evidence_requirements": _req(
            events={"active_at_query": True},
            pixel_visibility={"target_frame": True, "four_state": True},
            required_modalities=("video", "binaural_audio", "pixel_visibility"),
        ),
    },
    {
        "qa_id": "QA-09",
        "title": "完全遮挡后是否重现",
        "question_family": "reappeared_after_occlusion",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            pixel_visibility={"fully_occluded_then_visible": True},
            occlusion={"four_state": True},
            required_modalities=("video", "pixel_visibility"),
        ),
        "evidence_requirements": _req(
            pixel_visibility={"fully_occluded_then_visible": True},
            required_modalities=("video", "pixel_visibility"),
        ),
    },
    {
        "qa_id": "QA-10",
        "title": "遮挡物身份",
        "question_family": "occluder_identity",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            pixel_visibility={"target_frame": True},
            occlusion={"unique_registered_occluder": True},
            required_modalities=("video", "pixel_instance_visibility"),
        ),
        "evidence_requirements": _req(
            pixel_visibility={"target_frame": True, "occluder_ids": True},
            occlusion={"unique_registered_occluder": True},
            required_modalities=("video", "pixel_instance_visibility"),
        ),
    },
    {
        "qa_id": "QA-11",
        "title": "部分遮挡→完全可见",
        "question_family": "became_clear_after_partial_occlusion",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            pixel_visibility={"partial_to_clear": True},
            occlusion={"four_state": True},
            required_modalities=("video", "pixel_visibility"),
        ),
        "evidence_requirements": _req(
            pixel_visibility={"partial_to_clear": True},
            required_modalities=("video", "pixel_visibility"),
        ),
    },
    {
        "qa_id": "QA-12",
        "title": "外观→说了什么",
        "question_family": "appearance_to_spoken_content",
        "answer_type": "transcript_wer",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"min_count": 1, "target_binding": True},
            appearance={"required": True, "unique_target_value": True},
            speech_content={"transcript": True},
            required_modalities=("video", "binaural_audio"),
        ),
        "evidence_requirements": _req(
            events={"unique_target_statement": True},
            appearance={"field_and_value": True},
            speech_content={"transcript": True},
            required_modalities=("video", "binaural_audio"),
        ),
    },
    {
        "qa_id": "QA-13",
        "title": "发声结束后的方位",
        "question_family": "post_sound_azimuth",
        "answer_type": "angle_deg",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"anchor_event": True, "post_sound_query": True},
            after_sound={"silent_query_window": True, "target_continues": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
        "evidence_requirements": _req(
            events={"anchor_event": True},
            after_sound={"silent_query_window": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
    },
    {
        "qa_id": "QA-14",
        "title": "给定时刻的距离比较",
        "question_family": "distance_comparison_at_time",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            appearance={"two_unique_targets": True},
            motion={"query_frame_positions": True},
            required_modalities=("video", "time"),
        ),
        "evidence_requirements": _req(
            appearance={"two_unique_targets": True},
            motion={"query_frame_positions": True, "distance_margin": True},
            pixel_visibility={"targets_observable": True},
            required_modalities=("video", "time"),
        ),
    },
    {
        "qa_id": "QA-15",
        "title": "发声期间靠近/远离",
        "question_family": "distance_trend_during_sound",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            events={"bound_event": True},
            motion={"distance_trend_during_event": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
        "evidence_requirements": _req(
            events={"bound_event": True},
            motion={"distance_trend_during_event": True, "distance_margin": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
    },
    {
        "qa_id": "QA-16",
        "title": "声停后的距离变化",
        "question_family": "distance_trend_after_sound",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            events={"anchor_event": True, "post_sound_query": True},
            after_sound={"silent_query_window": True},
            motion={"distance_change_after_event": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
        "evidence_requirements": _req(
            events={"anchor_event": True},
            after_sound={"silent_query_window": True},
            motion={"distance_change_after_event": True, "distance_margin": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
    },
    {
        "qa_id": "QA-17",
        "title": "声停后是否移动",
        "question_family": "motion_after_sound",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            events={"anchor_event": True, "post_sound_query": True},
            after_sound={"silent_query_window": True},
            motion={"post_sound_motion": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
        "evidence_requirements": _req(
            events={"anchor_event": True},
            after_sound={"silent_query_window": True},
            motion={"post_sound_motion": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
    },
    {
        "qa_id": "QA-18",
        "title": "指定时刻谁在发声",
        "question_family": "speaker_at_time",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"query_time": True},
            appearance={"candidate_labels": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
        "evidence_requirements": _req(
            events={"active_at_query": True},
            appearance={"candidate_labels": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
    },
    {
        "qa_id": "QA-19",
        "title": "目标首次发声时刻",
        "question_family": "first_sound_time",
        "answer_type": "time_range_s",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            events={"target_binding": True},
            appearance={"required": True, "unique_target_value": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
        "evidence_requirements": _req(
            events={"target_binding": True, "first_event": True},
            appearance={"field_and_value": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
    },
    {
        "qa_id": "QA-20",
        "title": "可见候选谁发声/都不是",
        "question_family": "visible_candidate_or_none",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"anchor_event": True},
            pixel_visibility={"candidate_frame": True},
            required_modalities=("video", "binaural_audio"),
        ),
        "evidence_requirements": _req(
            events={"bound_event": True},
            pixel_visibility={"candidate_frame": True},
            required_modalities=("video", "binaural_audio"),
        ),
    },
    {
        "qa_id": "QA-21",
        "title": "外观→声音类别",
        "question_family": "appearance_to_sound_class",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"min_count": 2, "target_binding": True},
            appearance={"required": True, "unique_target_value": True},
            distinct_sound_classes=2,
            required_modalities=("video", "binaural_audio"),
        ),
        "evidence_requirements": _req(
            events={"target_binding": True, "explicit_sound_class": True},
            appearance={"field_and_value": True},
            distinct_sound_classes=2,
            required_modalities=("video", "binaural_audio"),
        ),
    },
    {
        "qa_id": "QA-22",
        "title": "出现过的实体数/发声个体数",
        "question_family": "entity_and_speaking_counts",
        "answer_type": "count_pair",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"statistics_window": "whole_clip"},
            required_modalities=("video", "binaural_audio", "time"),
        ),
        "evidence_requirements": _req(
            events={"statistics_window": "whole_clip"},
            required_modalities=("video", "binaural_audio", "time"),
        ),
    },
    {
        "qa_id": "QA-23",
        "title": "发声事件次数",
        "question_family": "sound_event_count",
        "answer_type": "count_single",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=1,
            events={"statistics_window": "whole_clip", "event_definition": True},
            required_modalities=("audio", "time"),
        ),
        "evidence_requirements": _req(
            events={"statistics_window": "whole_clip", "event_ids": True},
            required_modalities=("audio", "time"),
        ),
    },
    {
        "qa_id": "QA-24",
        "title": "先发声者在片尾的可见状态",
        "question_family": "first_speaker_final_visibility",
        "answer_type": "closed_set",
        "forms": ["mcq", "open"],
        "potential_requirements": _req(
            min_entities=2,
            events={"unique_earliest": True},
            pixel_visibility={"final_state": True},
            occlusion={"four_state": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
        "evidence_requirements": _req(
            events={"unique_earliest": True},
            pixel_visibility={"final_state": True, "four_state": True},
            required_modalities=("video", "binaural_audio", "time"),
        ),
    },
    {
        "qa_id": "QA-25",
        "title": "实例级连续水平角度",
        "question_family": "instance_continuous_bearing",
        "answer_type": "angle_deg",
        "forms": ["open"],
        "subsets": {
            "A": {"required_modalities": ["audio"], "target": "audible_event_emitter"},
            "V": {"required_modalities": ["video"], "target": "visible_pixel_centroid", "public_camera_calibration": True},
            "AV": {"required_modalities": ["audio", "video"], "target": "visually_anchored_hidden_emitter", "min_query_sources": 2},
        },
        "potential_requirements": _req(events={"source_activity": True},
            appearance={"reviewed": True}, motion={"hidden_change_for_av": True},
            pixel_visibility={"query_state": True}, required_modalities=("video", "binaural_audio", "time")),
        "evidence_requirements": _req(events={"source_activity": True},
            pixel_visibility={"query_state": True}, required_modalities=("time",)),
    },
)

_CATALOG_BY_ID = {item["qa_id"]: item for item in CATALOG}
QA_IDS = tuple(_CATALOG_BY_ID)


def _canonical_qa_id(value: Any) -> str:
    if not isinstance(value, str):
        raise UnifiedQAError("qa_id must be a string such as QA-01")
    match = re.fullmatch(r"QA[-_]?(\d{1,2})", value.strip().upper())
    if not match:
        raise UnifiedQAError(f"unknown qa_id {value!r}")
    number = int(match.group(1))
    if f"QA-{number:02d}" not in _CATALOG_BY_ID:
        raise UnifiedQAError(f"qa_id must be QA-01 through QA-25, got {value!r}")
    return f"QA-{number:02d}"


def get_requirements(qa_id: str) -> dict[str, Any]:
    """Return a copy of one pre-capture requirement record."""

    result = copy.deepcopy(_CATALOG_BY_ID[_canonical_qa_id(qa_id)])
    # Keep a flat compatibility view for planners that consume a single
    # requirement mapping, while retaining the explicit potential/evidence
    # split for callers that need to distinguish planning from readback proof.
    for key, value in result["potential_requirements"].items():
        result.setdefault(key, copy.deepcopy(value))
    return result


def question_catalog() -> list[dict[str, Any]]:
    """Return the complete stable catalog in numeric order."""

    return copy.deepcopy(list(CATALOG))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnifiedQAError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise UnifiedQAError(f"{name} must be a finite number")
    return number


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnifiedQAError(f"{name} must be a positive integer")
    return int(value)


def _as_vec(value: Any, *, name: str) -> list[float]:
    if not _is_sequence(value) or len(value) != 3:
        raise UnifiedQAError(f"{name} must be a three-vector")
    return [
        _finite_number(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _safe_slug(value: str) -> str:
    result = _ID_SAFE.sub("_", str(value)).strip("_.")
    return result or "item"


def _position_from_record(
    record: Mapping[str, Any],
    *,
    default_ue_cm: bool,
) -> list[float] | None:
    for key in ("position_m", "translation_m", "world_position_m", "location_m"):
        if key in record and record[key] is not None:
            return _as_vec(record[key], name=key)
    for key in ("location_cm", "translation_cm", "position_cm"):
        if key not in record or record[key] is None:
            continue
        vector = _as_vec(record[key], name=key)
        space = str(_first(record, "coordinate_system", "space", "units") or "").casefold()
        ue = default_ue_cm or "ue" in space or "unreal" in space
        if ue and "xyz" not in space:
            # Native readbacks use Unreal X/Y/Z centimetres while AVEngine
            # analytic coordinates are [x, vertical-y, z]. Plan records with
            # position_m already use analytic order and bypass this path.
            return [vector[0] / 100.0, vector[2] / 100.0, vector[1] / 100.0]
        return [component / 100.0 for component in vector]
    return None


def _records_for_actor(
    container: Any,
    actor_id: str,
    *,
    frame_count: int,
) -> list[Mapping[str, Any]] | None:
    if not isinstance(container, Mapping):
        return None
    value = container.get(actor_id)
    if isinstance(value, Mapping):
        value = value.get("frames", value.get("readbacks"))
    if value is None:
        return None
    if not _is_sequence(value):
        raise UnifiedQAError(f"{actor_id}: actor frame records must be a list")
    records = [record for record in value if isinstance(record, Mapping)]
    if len(records) != len(value):
        raise UnifiedQAError(f"{actor_id}: actor frame records must be objects")
    if len(records) == 1:
        records = [dict(records[0], frame_index=index) for index in range(frame_count)]
    if len(records) != frame_count:
        return None
    by_index: dict[int, Mapping[str, Any]] = {}
    for ordinal, record in enumerate(records):
        frame = record.get("frame_index", ordinal)
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise UnifiedQAError(f"{actor_id}: frame_index must be an integer")
        if frame in by_index:
            raise UnifiedQAError(f"{actor_id}: duplicate frame_index {frame}")
        by_index[frame] = record
    if set(by_index) != set(range(frame_count)):
        return None
    return [by_index[index] for index in range(frame_count)]


def _plan_actor_records(
    plan_frames: Any,
    actor_id: str,
    *,
    frame_count: int,
) -> list[Mapping[str, Any]] | None:
    if not _is_sequence(plan_frames):
        return None
    records: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(plan_frames):
        if not isinstance(frame, Mapping):
            continue
        states = frame.get("actor_states")
        if not _is_sequence(states):
            continue
        match = next(
            (
                state
                for state in states
                if isinstance(state, Mapping) and state.get("actor_id") == actor_id
            ),
            None,
        )
        if match is None:
            continue
        record = dict(match)
        record.setdefault("frame_index", frame.get("frame_index", frame_index))
        records.append(record)
    if len(records) != frame_count:
        return None
    return records


def _position_series(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    default_ue_cm: bool,
) -> list[list[float]] | None:
    if records is None:
        return None
    result: list[list[float]] = []
    for record in records:
        position = _position_from_record(record, default_ue_cm=default_ue_cm)
        if position is None:
            return None
        result.append(position)
    return result


def _yaw_from_record(record: Mapping[str, Any]) -> float | None:
    raw = _first(record, "yaw_deg", "heading_deg")
    if raw is not None:
        return _finite_number(raw, name="yaw_deg")
    rotation = _first(record, "rotation_deg", "euler_deg")
    if _is_sequence(rotation) and len(rotation) >= 2:
        return _finite_number(rotation[1], name="rotation_deg[1]")
    return None


def _yaw_series(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    frame_count: int,
) -> list[float] | None:
    if records is None:
        return None
    values: list[float] = []
    for record in records:
        yaw = _yaw_from_record(record)
        if yaw is None:
            return None
        values.append(yaw)
    if len(values) != frame_count:
        return None
    return values


def _normalize_vec(value: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(component) ** 2 for component in value))
    if norm <= 1.0e-12:
        raise UnifiedQAError("listener basis vector must be non-zero")
    return [float(component) / norm for component in value]


def _cross(first: Sequence[float], second: Sequence[float]) -> list[float]:
    return [
        float(first[1]) * float(second[2])
        - float(first[2]) * float(second[1]),
        float(first[2]) * float(second[0])
        - float(first[0]) * float(second[2]),
        float(first[0]) * float(second[1])
        - float(first[1]) * float(second[0]),
    ]


def _ue_rotator_to_m3_basis(
    rotation: Sequence[float],
) -> dict[str, list[float]]:
    """Use the authoritative SPEAR/UE X-forward,Z-up to M3 Y-up exchange.

    This is the same basis contract used by the native RLR frame-readback
    renderer: rotation is [roll, pitch, yaw], UE optical forward is +X, right
    is +Y, up is +Z, and the canonical AVEngine frame is [x, vertical-y, z].
    Keeping the basis instead of reducing it to a negated UE yaw avoids the
    historical 90-degree listener error.
    """

    if not _is_sequence(rotation) or len(rotation) != 3:
        raise UnifiedQAError("UE listener rotation must be [roll,pitch,yaw]")
    roll, pitch, yaw = (
        math.radians(float(value)) for value in rotation
    )
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    forward_ue = [cp * cy, cp * sy, sp]
    right0_ue = [-sy, cy, 0.0]
    up0_ue = [-sp * cy, -sp * sy, cp]
    cr, sr = math.cos(roll), math.sin(roll)
    right_ue = [
        right0_ue[index] * cr + up0_ue[index] * sr
        for index in range(3)
    ]
    up_ue = [
        -right0_ue[index] * sr + up0_ue[index] * cr
        for index in range(3)
    ]
    # UE [X,Y,Z] -> AVEngine [x, vertical-y, z].
    forward = [forward_ue[0], forward_ue[2], forward_ue[1]]
    right = [right_ue[0], right_ue[2], right_ue[1]]
    up = [up_ue[0], up_ue[2], up_ue[1]]
    right = _normalize_vec(_cross(forward, up))
    forward = _normalize_vec(forward)
    up = _normalize_vec(_cross(right, forward))
    return {"forward": forward, "right": right, "up": up}


def _basis_from_record(record: Mapping[str, Any]) -> dict[str, list[float]] | None:
    raw_basis = _first(record, "basis_m3", "listener_basis_m3", "basis")
    if isinstance(raw_basis, Mapping):
        if all(
            _is_sequence(raw_basis.get(key)) and len(raw_basis[key]) == 3
            for key in ("forward", "right", "up")
        ):
            return {
                key: _normalize_vec(_as_vec(raw_basis[key], name=f"basis.{key}"))
                for key in ("forward", "right", "up")
            }
    rotation = _first(record, "rotation_deg", "euler_deg")
    if _is_sequence(rotation) and len(rotation) == 3:
        return _ue_rotator_to_m3_basis(rotation)
    raw_yaw = _first(record, "yaw_deg", "heading_deg")
    if isinstance(raw_yaw, (int, float)) and not isinstance(raw_yaw, bool):
        yaw = math.radians(float(raw_yaw))
        return {
            "forward": [math.sin(yaw), 0.0, -math.cos(yaw)],
            "right": [math.cos(yaw), 0.0, math.sin(yaw)],
            "up": [0.0, 1.0, 0.0],
        }
    return None


def _basis_series(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    frame_count: int,
) -> list[dict[str, list[float]]] | None:
    if records is None:
        return None
    values: list[dict[str, list[float]]] = []
    for record in records:
        basis = _basis_from_record(record)
        if basis is None:
            return None
        values.append(basis)
    if len(values) != frame_count:
        return None
    return values


def _appearance_from_record(record: Mapping[str, Any]) -> dict[str, str] | None:
    """Read only explicit realized appearance fields.

    Actor IDs are never parsed here: foo_blue_v1 is an identifier, not pixel
    evidence. The room planner should provide realized_attributes or a
    top-level appearance binding.
    """

    candidates: list[tuple[str, Any]] = []
    for owner_key in ("appearance", "realized_attributes", "attributes"):
        owner = record.get(owner_key)
        if not isinstance(owner, Mapping):
            continue
        if owner_key == "appearance":
            if owner.get("value") is not None:
                candidates.append(
                    (str(owner.get("field") or "appearance_value"), owner.get("value"))
                )
            keys = (
                "shirt_color",
                "top_color",
                "color",
                "color_name",
                "coat_value",
                "appearance_value",
            )
        else:
            keys = (
                "shirt_color",
                "top_color",
                "color",
                "color_name",
                "coat_value",
                "appearance_value",
            )
        for key in keys:
            if owner.get(key) is not None:
                candidates.append((key, owner.get(key)))
        coat = owner.get("coat_profile")
        if isinstance(coat, Mapping) and coat.get("value") is not None:
            candidates.append(("coat_value", coat.get("value")))
    for key in (
        "shirt_color",
        "top_color",
        "color",
        "color_name",
        "coat_value",
        "appearance_value",
    ):
        if record.get(key) is not None:
            candidates.append((key, record.get(key)))
    for field, value in candidates:
        if isinstance(value, str) and value.strip():
            label = _first(record, "appearance_label", "display_label", "label")
            if not isinstance(label, str) or not label.strip():
                nested_appearance = record.get("appearance")
                if isinstance(nested_appearance, Mapping):
                    label = _first(
                        nested_appearance, "appearance_label", "display_label", "label"
                    )
            label_text = _strip_display_version_suffix(label)
            return {
                "field": field,
                "value": value.strip(),
                "label": label_text or value.strip(),
            }
    return None


def _actor_label(
    actor_id: str,
    record: Mapping[str, Any],
    appearance: Mapping[str, Any] | None,
) -> str:
    value = _first(record, "display_label", "label")
    value_text = _strip_display_version_suffix(value)
    if value_text:
        return value_text
    if appearance and isinstance(appearance.get("label"), str):
        appearance_label = _strip_display_version_suffix(appearance["label"])
        if appearance_label:
            return appearance_label
    if appearance and isinstance(appearance.get("value"), str):
        return str(appearance["value"])
    return actor_id


def _extract_registry_records(
    registry: Any,
    *,
    families: Sequence[str],
) -> list[Mapping[str, Any]]:
    if not isinstance(registry, Mapping):
        return []
    for family in families:
        records = registry.get(family)
        if _is_sequence(records):
            return [record for record in records if isinstance(record, Mapping)]
    return []


def _sound_registry_index(root: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = _extract_registry_records(
        root.get("sound_registry"),
        families=("sounds", "sound_assets", "assets", "samples"),
    )
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sound_id = _first(record, "sound_asset_id", "sound_id", "sound_key", "id")
        if isinstance(sound_id, str) and sound_id:
            result[sound_id] = record
    return result


def _voice_binding_index(
    root: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    value = _first(root, "voice_bindings", "voice_binding")
    records: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if _is_sequence(value.get("bindings")):
            records = [
                record for record in value["bindings"] if isinstance(record, Mapping)
            ]
        else:
            records = [
                dict(record, actor_id=key)
                for key, record in value.items()
                if isinstance(record, Mapping)
            ]
    elif _is_sequence(value):
        records = [record for record in value if isinstance(record, Mapping)]
    by_sound: dict[str, Mapping[str, Any]] = {}
    by_actor: dict[str, Mapping[str, Any]] = {}
    for record in records:
        actor = _first(
            record, "actor_id", "source_slot_id", "instance_id", "entity_instance_id"
        )
        sound = _first(record, "sound_asset_id", "sound_id")
        if isinstance(actor, str) and actor:
            by_actor[actor] = record
        if isinstance(sound, str) and sound:
            by_sound[sound] = record
    return by_sound, by_actor


def _endpoint_index(root: Mapping[str, Any]) -> dict[str, str]:
    values: list[Any] = [
        root.get("source_endpoint_bindings"),
        root.get("endpoint_bindings"),
        root.get("source_endpoint_registry"),
    ]
    result: dict[str, str] = {}
    for value in values:
        if isinstance(value, Mapping):
            records = value.get("source_endpoints")
            if _is_sequence(records):
                iterable = records
            else:
                iterable = [
                    dict(record, source_endpoint_id=key)
                    for key, record in value.items()
                    if isinstance(record, Mapping)
                ]
        elif _is_sequence(value):
            iterable = value
        else:
            continue
        for record in iterable:
            if not isinstance(record, Mapping):
                continue
            endpoint = _first(record, "source_endpoint_id", "endpoint_id", "id")
            binding = record.get("binding") if isinstance(record.get("binding"), Mapping) else record
            actor = _first(
                binding,
                "actor_id",
                "source_slot_id",
                "instance_id",
                "entity_instance_id",
                "target_actor_id",
            )
            if isinstance(endpoint, str) and isinstance(actor, str):
                result[endpoint] = actor
    return result


def _resolve_event_actor(
    event: Mapping[str, Any],
    *,
    actor_ids: Sequence[str],
    endpoint_index: Mapping[str, str],
    by_sound: Mapping[str, Mapping[str, Any]],
) -> str | None:
    for key in (
        "actor_id",
        "source_slot_id",
        "source_instance_id",
        "instance_id",
        "target_actor_id",
    ):
        value = event.get(key)
        if isinstance(value, str) and value in actor_ids:
            return value
    endpoint = _first(event, "source_endpoint_id", "endpoint_id")
    if isinstance(endpoint, str):
        bound = endpoint_index.get(endpoint)
        if bound in actor_ids:
            return bound
        matches = [actor for actor in actor_ids if actor in endpoint]
        if len(matches) == 1:
            return matches[0]
    sound_id = _first(event, "sound_asset_id", "sound_id")
    voice = by_sound.get(sound_id) if isinstance(sound_id, str) else None
    if isinstance(voice, Mapping):
        value = _first(
            voice, "actor_id", "source_slot_id", "instance_id", "entity_instance_id"
        )
        if isinstance(value, str) and value in actor_ids:
            return value
    return None


def _event_source_field(
    event: Mapping[str, Any],
    *keys: str,
    sound_record: Mapping[str, Any] | None,
    voice_record: Mapping[str, Any] | None,
) -> Any:
    """One per-event field, from the event or from the clip it plays.

    An M6 AudioProgram event carries scheduling only - which clip, from which
    sample, at which gain - so facts about the *recording* (its QC record, the
    event segmentation that says how many sounds it holds) live on the pool
    row, which reaches delivery as the rendered sound registry.  Reading the
    event first keeps an explicitly stated value authoritative; falling back to
    the registry is what lets a field added to the pool arrive in the facts
    without every producer in between being taught to copy it.
    """

    for owner in (event, voice_record, sound_record):
        if not isinstance(owner, Mapping):
            continue
        found = _first(owner, *keys)
        if found is not None:
            return found
    return None


def _content_from_event(
    event: Mapping[str, Any],
    *,
    sound_record: Mapping[str, Any] | None,
    voice_record: Mapping[str, Any] | None,
) -> tuple[str | None, str | None, str | None, str | None, bool]:
    """Return transcript, statement id, language, sound class and explicitness.

    A generic playback capability is not an observed event class. If the
    event carries that capability while its bound sound registry identifies a
    concrete class, the concrete registry class is authoritative.
    """

    owners = [event, voice_record or {}, sound_record or {}]
    transcript: str | None = None
    statement_id: str | None = None
    language: str | None = None
    sound_class: str | None = None
    explicit = False
    for owner in owners:
        content = (
            owner.get("content")
            if isinstance(owner.get("content"), Mapping)
            else owner
        )
        if transcript is None:
            candidate = _first(content, "transcript", "text", "utterance")
            if isinstance(candidate, str) and candidate.strip():
                transcript = candidate.strip()
        if statement_id is None:
            candidate = _first(content, "statement_id", "utterance_id")
            if isinstance(candidate, str) and candidate.strip():
                statement_id = candidate.strip()
        if language is None:
            candidate = content.get("language")
            if isinstance(candidate, str) and candidate.strip():
                language = candidate.strip()
        if sound_class is None:
            candidate = _first(
                content,
                "sound_class",
                "sound_category",
                "sound_type",
                "semantic_sound_class",
                "event_class",
                "category",
            )
            if isinstance(candidate, str) and candidate.strip():
                sound_class = candidate.strip()
                explicit = True
            elif _is_sequence(content.get("event_classes")):
                candidates = [
                    value.strip()
                    for value in content["event_classes"]
                    if isinstance(value, str) and value.strip()
                ]
                if len(candidates) == 1:
                    sound_class = candidates[0]
                    explicit = True
    capability_key = (
        re.sub(r"\s+", "_", sound_class.casefold())
        if sound_class is not None
        else None
    )
    if capability_key in _CAPABILITY_SOUND_CLASSES:
        for owner in (sound_record or {}, voice_record or {}):
            content = (
                owner.get("content")
                if isinstance(owner.get("content"), Mapping)
                else owner
            )
            candidate = _first(
                content,
                "semantic_sound_class",
                "sound_class",
                "sound_category",
                "sound_type",
                "event_class",
                "category",
            )
            if isinstance(candidate, str) and candidate.strip() and candidate.casefold() not in _CAPABILITY_SOUND_CLASSES:
                sound_class = candidate.strip()
                explicit = True
                break
            if _is_sequence(content.get("event_classes")):
                candidates = [
                    value.strip()
                    for value in content["event_classes"]
                    if isinstance(value, str)
                    and value.strip()
                    and value.casefold() not in _CAPABILITY_SOUND_CLASSES
                ]
                if len(candidates) == 1:
                    sound_class = candidates[0]
                    explicit = True
                    break
    if transcript is not None and sound_class is None:
        sound_class = "speech"
        explicit = True
    if sound_class is not None:
        sound_class = re.sub(r"\s+", "_", sound_class.casefold())
    return transcript, statement_id, language, sound_class, explicit


def _event_time(
    event: Mapping[str, Any],
    *,
    clock: Mapping[str, Any],
    timeline: Mapping[str, Any],
) -> tuple[float, float, int | None, int | None]:
    frame_rate = float(clock["frame_rate_hz"])
    sample_rate = int(clock["sample_rate_hz"])
    time_base = int(timeline.get("time_base_hz", clock.get("time_base_hz", 48000)))
    start_tick = _first(event, "start_tick", "start_ticks")
    end_tick = _first(event, "end_tick_exclusive", "end_tick", "end_ticks")
    start_sample = _first(event, "start_sample", "start_sample_index")
    end_sample = _first(event, "end_sample_exclusive", "end_sample", "end_sample_index")
    start_frame = _first(event, "start_frame")
    end_frame = _first(event, "end_frame")
    start_s = _first(event, "start_s", "start_seconds", "time_start_s")
    end_s = _first(event, "end_s", "end_seconds", "time_end_s")
    if start_s is None and isinstance(start_tick, (int, float)) and not isinstance(start_tick, bool):
        start_s = float(start_tick) / time_base
    if end_s is None and isinstance(end_tick, (int, float)) and not isinstance(end_tick, bool):
        end_s = float(end_tick) / time_base
    if start_s is None and isinstance(start_sample, (int, float)) and not isinstance(start_sample, bool):
        start_s = float(start_sample) / sample_rate
    if end_s is None and isinstance(end_sample, (int, float)) and not isinstance(end_sample, bool):
        end_s = float(end_sample) / sample_rate
    if start_s is None and isinstance(start_frame, (int, float)) and not isinstance(start_frame, bool):
        start_s = float(start_frame) / frame_rate
    if end_s is None and isinstance(end_frame, (int, float)) and not isinstance(end_frame, bool):
        end_s = float(end_frame) / frame_rate
    if start_s is None or end_s is None:
        raise UnifiedQAError(f"event {event.get('event_id')!r} has no complete time interval")
    start = _finite_number(start_s, name="event start_s")
    end = _finite_number(end_s, name="event end_s")
    if end <= start:
        raise UnifiedQAError(f"event {event.get('event_id')!r} must have end after start")
    start_frame_norm = int(math.floor(start * frame_rate)) if start_frame is None else int(start_frame)
    end_frame_norm = int(math.ceil(end * frame_rate)) if end_frame is None else int(end_frame)
    return start, end, start_frame_norm, end_frame_norm


def _camera_fallback(
    frame_readbacks: Mapping[str, Any],
    *,
    frame_count: int,
) -> list[Mapping[str, Any]] | None:
    camera = frame_readbacks.get("camera")
    if not _is_sequence(camera) or len(camera) != frame_count:
        return None
    if not all(isinstance(record, Mapping) for record in camera):
        return None
    return list(camera)


def _listener_block(
    root: Mapping[str, Any],
    frame_readbacks: Mapping[str, Any],
    audio_readback: Mapping[str, Any] | None,
    plan: Mapping[str, Any],
    *,
    frame_count: int,
) -> dict[str, Any]:
    candidates: list[tuple[str, Any, bool]] = [
        ("listener", root.get("listener"), False),
        ("frame_readbacks.listener", frame_readbacks.get("listener"), True),
        (
            "audio_readback.listener",
            audio_readback.get("listener") if audio_readback else None,
            True,
        ),
        ("plan.listener", plan.get("listener"), False),
    ]
    camera = _camera_fallback(frame_readbacks, frame_count=frame_count)
    if camera is not None:
        candidates.append(("frame_readbacks.camera", camera, True))
    for source, value, ue_default in candidates:
        if value is None:
            continue
        if isinstance(value, Mapping):
            records = value.get("frames", value.get("poses"))
            if records is None:
                records = [value]
        elif _is_sequence(value):
            records = value
        else:
            continue
        if not _is_sequence(records):
            continue
        records = [record for record in records if isinstance(record, Mapping)]
        if len(records) == 1:
            records = [dict(records[0], frame_index=index) for index in range(frame_count)]
        if len(records) != frame_count:
            continue
        positions = _position_series(records, default_ue_cm=ue_default)
        yaws = _yaw_series(records, frame_count=frame_count)
        basis = _basis_series(records, frame_count=frame_count)
        if positions is None:
            continue
        if yaws is None:
            yaws = [0.0] * frame_count
        if basis is None:
            basis = [
                {
                    "forward": [
                        math.sin(math.radians(yaw)),
                        0.0,
                        -math.cos(math.radians(yaw)),
                    ],
                    "right": [
                        math.cos(math.radians(yaw)),
                        0.0,
                        math.sin(math.radians(yaw)),
                    ],
                    "up": [0.0, 1.0, 0.0],
                }
                for yaw in yaws
            ]
        return {
            "status": "pass",
            "source": source,
            "positions_m": positions,
            "yaw_deg": yaws,
            "basis_m3": basis,
            "declared": source != "frame_readbacks.camera",
        }
    return {
        "status": "not_run",
        "source": None,
        "positions_m": None,
        "yaw_deg": None,
        "declared": False,
    }


def _validate_audio_readback(
    root: Mapping[str, Any],
    *,
    clock: Mapping[str, Any],
) -> dict[str, Any]:
    value = _first(root, "audio_readback", "audio_validation", "audio")
    if not isinstance(value, Mapping):
        report = root.get("research_report") or root.get("audio_report")
        value = report.get("audio") if isinstance(report, Mapping) else None
    if not isinstance(value, Mapping):
        return {
            "status": "not_run",
            "reason": "audio_readback/audio validation is missing",
            "channel_count": None,
            "sample_rate_hz": None,
            "sample_count": None,
        }
    report = root.get("research_report") or root.get("audio_report")
    report_events = report.get("events") if isinstance(report, Mapping) else None
    path_value = _first(value, "mixture_path", "audio_path", "path", "file")
    path = Path(path_value).expanduser() if isinstance(path_value, str) and path_value else None
    declared_channels = _first(value, "channel_count", "channels")
    declared_rate = _first(value, "sample_rate_hz", "sample_rate")
    declared_count = _first(value, "sample_count", "frames")
    observed: dict[str, Any] = {}
    if path is not None and path.is_file():
        try:
            with wave.open(str(path), "rb") as stream:
                observed = {
                    "channel_count": stream.getnchannels(),
                    "sample_rate_hz": stream.getframerate(),
                    "sample_count": stream.getnframes(),
                    "sample_width": stream.getsampwidth(),
                }
        except (OSError, wave.Error) as error:
            return {
                "status": "fail",
                "reason": f"cannot read audio mixture: {error}",
                "path": str(path),
            }
    channels = observed.get("channel_count", declared_channels)
    rate = observed.get("sample_rate_hz", declared_rate)
    count = observed.get("sample_count", declared_count)
    if isinstance(channels, bool) or not isinstance(channels, int):
        return {"status": "not_run", "reason": "audio channel_count is unavailable"}
    if channels != 2:
        return {
            "status": "fail",
            "reason": f"final mixture has {channels} channels; exactly two are required",
            "channel_count": channels,
            "sample_rate_hz": rate,
            "sample_count": count,
        }
    if rate is not None and int(rate) != int(clock["sample_rate_hz"]):
        return {
            "status": "fail",
            "reason": "final mixture sample rate disagrees with episode clock",
            "channel_count": channels,
            "sample_rate_hz": rate,
            "sample_count": count,
        }
    if count is not None and int(count) != int(clock["sample_count"]):
        return {
            "status": "fail",
            "reason": "final mixture sample count disagrees with episode clock",
            "channel_count": channels,
            "sample_rate_hz": rate,
            "sample_count": count,
        }
    proof = "wave_readback" if observed else _first(value, "proof", "readback_proof")
    if not proof:
        return {
            "status": "not_run",
            "reason": "two channels are declared without an audio readback proof",
            "channel_count": channels,
            "sample_rate_hz": rate,
            "sample_count": count,
        }
    wet_tail_intervals: list[dict[str, Any]] = []
    raw_intervals = value.get("wet_tail_intervals")
    if _is_sequence(raw_intervals):
        for interval in raw_intervals:
            if not isinstance(interval, Mapping):
                continue
            start = _first(interval, "start_s", "start_seconds")
            end = _first(interval, "end_s", "end_seconds")
            if start is None or end is None:
                start_sample = _first(interval, "start_sample", "start_sample_index")
                end_sample = _first(
                    interval, "end_sample_exclusive", "end_sample", "end_sample_index"
                )
                if isinstance(start_sample, (int, float)) and isinstance(
                    end_sample, (int, float)
                ):
                    start = float(start_sample) / int(clock["sample_rate_hz"])
                    end = float(end_sample) / int(clock["sample_rate_hz"])
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                wet_tail_intervals.append(
                    {
                        "event_id": _first(interval, "event_id", "id"),
                        "start_s": float(start),
                        "end_s": float(end),
                        "source": "audio_readback",
                    }
                )
    event_records = report_events
    if not _is_sequence(event_records):
        event_records = value.get("events")
    if _is_sequence(event_records):
        for interval in event_records:
            if not isinstance(interval, Mapping):
                continue
            raw = _first(
                interval,
                "wet_render_interval",
                "wet_float_nonzero_interval",
                "rendered_interval",
            )
            if not _is_sequence(raw) or len(raw) != 2:
                continue
            wet_tail_intervals.append(
                {
                    "event_id": _first(interval, "event_id", "id"),
                    "start_s": float(raw[0]) / int(clock["sample_rate_hz"]),
                    "end_s": float(raw[1]) / int(clock["sample_rate_hz"]),
                    "source": "research_report",
                }
            )
    return {
        "status": "pass",
        "proof": proof,
        "path": str(path) if path is not None else None,
        "channel_count": channels,
        "sample_rate_hz": int(rate) if rate is not None else None,
        "sample_count": int(count) if count is not None else None,
        "channel_order": _first(value, "channel_order", "channels_order"),
        "hrtf_id": _first(value, "hrtf_id", "hrtf_profile_id"),
        "source_mix": _first(value, "source_mix", "mixing", "mixture"),
        "wet_tail_intervals": wet_tail_intervals,
        "event_segmentation": _first(
            value, "event_segmentation", "event_segmentation_review"
        ),
    }


def _source_activity_index(
    values: Sequence[Any],
    *,
    sample_count: int,
) -> tuple[dict[str, list[dict[str, int]]], bool, set[str]]:
    """Normalize P6 episode-sample source activity without inferring it."""

    result: dict[str, list[dict[str, int]]] = {}
    observed_event_ids: set[str] = set()
    present = False
    known_keys = {
        "event_id",
        "id",
        "source_activity_intervals_samples",
        "intervals",
        "source_activity",
        "start_sample",
        "start_sample_index",
        "end_sample_exclusive",
        "end_sample",
        "end_sample_index",
    }
    metadata_keys = {
        "schema",
        "status",
        "coordinate_space",
        "sample_rate_hz",
        "sample_count",
        "metadata",
        "provenance",
    }

    def integer(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            return None
        return int(number)

    def mark(event_id: Any) -> None:
        if isinstance(event_id, str) and event_id.strip():
            observed_event_ids.add(event_id.strip())

    def add(event_id: Any, start: Any, end: Any) -> None:
        if not isinstance(event_id, str) or not event_id.strip():
            return
        start_i, end_i = integer(start), integer(end)
        if start_i is None or end_i is None:
            return
        start_i = max(0, start_i)
        end_i = min(sample_count, end_i)
        if end_i <= start_i:
            return
        event_key = event_id.strip()
        mark(event_key)
        result.setdefault(event_key, []).append(
            {
                "start_sample": start_i,
                "end_sample_exclusive": end_i,
            }
        )

    def visit(value: Any, event_id: str | None = None) -> None:
        nonlocal present
        if value is None:
            return
        if isinstance(value, Mapping):
            local_event = _first(value, "event_id", "id")
            if not isinstance(local_event, str) or not local_event.strip():
                local_event = event_id
            nested = _first(
                value,
                "source_activity_intervals_samples",
                "intervals",
                "source_activity",
            )
            if nested is not None:
                present = True
                if isinstance(nested, Mapping):
                    nested_intervals = _first(
                        nested,
                        "source_activity_intervals_samples",
                        "intervals",
                    )
                    if nested_intervals is not None:
                        visit(nested_intervals, local_event)
                    else:
                        visit(nested, local_event)
                else:
                    visit(nested, local_event)
                return
            start = _first(value, "start_sample", "start_sample_index")
            end = _first(value, "end_sample_exclusive", "end_sample", "end_sample_index")
            if start is not None or end is not None:
                present = True
                add(local_event, start, end)
                return
            for key, nested_value in value.items():
                if key in known_keys:
                    continue
                if key == "events":
                    present = True
                    visit(nested_value)
                    continue
                if key in metadata_keys:
                    continue
                if isinstance(key, str):
                    if nested_value is None:
                        continue
                    present = True
                    visit(nested_value, key)
            return
        if _is_sequence(value):
            if not value:
                # An explicitly empty per-event list is measured silence.
                # Malformed non-empty intervals are missing evidence instead
                # of an observed empty activity set.
                mark(event_id)
                return
            if len(value) == 2:
                start, end = value
                if integer(start) is not None and integer(end) is not None:
                    present = True
                    add(event_id, start, end)
                    return
            for nested_value in value:
                visit(nested_value, event_id)

    for value in values:
        visit(value)
    for event_id, rows in result.items():
        result[event_id] = list(
            dict.fromkeys(
                (row["start_sample"], row["end_sample_exclusive"])
                for row in rows
            )
        )
        result[event_id] = [
            {"start_sample": start, "end_sample_exclusive": end}
            for start, end in result[event_id]
        ]
    return result, present, observed_event_ids




def _visibility_index(
    value: Any,
    *,
    actor_ids: Sequence[str],
    frame_count: int,
) -> dict[str, dict[int, Mapping[str, Any]]]:
    result: dict[str, dict[int, Mapping[str, Any]]] = {}
    if not isinstance(value, Mapping):
        return result
    if value.get("status") != "computed_modal_target_only_v1":
        return result
    per_instance = value.get("per_instance")
    if not isinstance(per_instance, Mapping):
        return result
    for actor_id in actor_ids:
        entry = per_instance.get(actor_id)
        frames = entry.get("frames") if isinstance(entry, Mapping) else None
        if not _is_sequence(frames) or not frames:
            continue
        by_frame: dict[int, Mapping[str, Any]] = {}
        valid = True
        for ordinal, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                valid = False
                break
            # A complete legacy array has an implicit ordinal frame clock.
            # Sparse probes need explicit indices; never infer or fill gaps.
            if len(frames) != frame_count and "frame_index" not in frame:
                valid = False
                break
            index = frame.get("frame_index", ordinal)
            state = frame.get("state")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= frame_count
                or state not in VISIBILITY_STATES
                or index in by_frame
            ):
                valid = False
                break
            normalized_frame = dict(frame)
            by_frame[index] = normalized_frame
        if valid:
            result[actor_id] = by_frame
    return result


def _visibility_is_complete(facts: Mapping[str, Any], actor_id: str) -> bool:
    """Return whether one actor has explicit visibility for every frame."""

    frames = facts.get("visibility", {}).get(actor_id)
    time = facts.get("time")
    if not isinstance(frames, Mapping) or not isinstance(time, Mapping):
        return False
    frame_count = time.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        return False
    try:
        indices = {int(frame) for frame in frames}
    except (TypeError, ValueError):
        return False
    return indices == set(range(frame_count))


def visibility_state_census(
    facts: Mapping[str, Any],
    actor_id: str | None = None,
) -> dict[str, Any]:
    """Count the visibility states actually observed, per actor or overall.

    A family that needs an out-of-view or fully-occluded state produces
    nothing when the episode never records one. That is a scene the producer
    has to generate, not a judging failure, so the absent state is named and
    counted instead of collapsing into one generic miss.
    """

    rows = facts.get("visibility")
    states: dict[str, int] = {}
    complete: list[str] = []
    incomplete: list[str] = []
    if not isinstance(rows, Mapping):
        return {"states": states, "complete_actors": complete,
                "incomplete_actors": incomplete, "actor_count": 0}
    selected = (
        {actor_id: rows.get(actor_id)}
        if actor_id is not None
        else rows
    )
    for key, frames in selected.items():
        if not isinstance(frames, Mapping):
            incomplete.append(str(key))
            continue
        for row in frames.values():
            if isinstance(row, Mapping):
                name = str(row.get("state"))
                states[name] = states.get(name, 0) + 1
        (complete if _visibility_is_complete(facts, str(key)) else incomplete).append(
            str(key)
        )
    return {
        "states": dict(sorted(states.items())),
        "complete_actors": sorted(complete),
        "incomplete_actors": sorted(incomplete),
        "actor_count": len(selected),
    }


def _motion_series(
    positions: list[list[float]] | None,
    *,
    frame_rate_hz: float,
    explicit: Sequence[Any] | None = None,
) -> tuple[list[float] | None, list[bool] | None]:
    expected = len(positions) if positions is not None else len(explicit or ())
    if explicit is not None and len(explicit) == expected and all(
        isinstance(value, bool) for value in explicit
    ):
        return None, [bool(value) for value in explicit]
    if positions is None or len(positions) < 2:
        return None, None
    speeds: list[float] = []
    for left, right in zip(positions, positions[1:]):
        distance = math.sqrt(
            sum((right[index] - left[index]) ** 2 for index in range(3))
        )
        speeds.append(distance * frame_rate_hz)
    speeds.append(speeds[-1] if speeds else 0.0)
    return speeds, [speed > 0.05 for speed in speeds]


def _listener_azimuth(
    source: Sequence[float],
    listener: Mapping[str, Any],
    frame: int,
) -> float | None:
    positions = listener.get("positions_m")
    if not _is_sequence(positions) or len(positions) <= frame:
        return None
    point = positions[frame]
    if not _is_sequence(point) or len(point) != 3:
        return None
    vector = [float(source[index]) - float(point[index]) for index in range(3)]
    basis = listener.get("basis_m3")
    if (
        not _is_sequence(basis)
        or len(basis) <= frame
        or not isinstance(basis[frame], Mapping)
    ):
        yaws = listener.get("yaw_deg")
        yaw = (
            float(yaws[frame])
            if _is_sequence(yaws) and len(yaws) > frame
            else 0.0
        )
        forward = [math.sin(math.radians(yaw)), 0.0, -math.cos(math.radians(yaw))]
        right = [math.cos(math.radians(yaw)), 0.0, math.sin(math.radians(yaw))]
    else:
        forward = basis[frame]["forward"]
        right = basis[frame]["right"]
    forward_dot = vector[0] * forward[0] + vector[2] * forward[2]
    right_dot = vector[0] * right[0] + vector[2] * right[2]
    if math.isclose(forward_dot, 0.0, abs_tol=1.0e-12) and math.isclose(
        right_dot, 0.0, abs_tol=1.0e-12
    ):
        return None
    angle = math.degrees(math.atan2(right_dot, forward_dot))
    return float((angle + 180.0) % 360.0 - 180.0)


def _distance(
    source: Sequence[float],
    listener: Mapping[str, Any],
    frame: int,
) -> float | None:
    positions = listener.get("positions_m")
    if not _is_sequence(positions) or len(positions) <= frame:
        return None
    point = positions[frame]
    if not _is_sequence(point) or len(point) != 3:
        return None
    return math.sqrt(
        sum((float(source[index]) - float(point[index])) ** 2 for index in range(3))
    )


def normalize_episode_bundle(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize native plan/readbacks into deterministic episode facts."""

    if not isinstance(raw, Mapping):
        raise UnifiedQAError("episode bundle must be an object")
    root: Mapping[str, Any] = raw
    nested = raw.get("episode")
    if isinstance(nested, Mapping):
        root = nested
    plan_value = root.get("plan") or root.get("visual_plan") or {}
    plan_meta = plan_value if isinstance(plan_value, Mapping) else {}
    episode_id = _first(root, "episode_id", "scene_id", "id")
    if episode_id is None:
        episode_id = _first(plan_meta, "episode_id", "scene_id", "id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise UnifiedQAError("episode bundle requires a readable episode_id")
    episode_id = episode_id.strip()
    plan = plan_value
    if isinstance(plan, Mapping) and isinstance(plan.get("visual_plan"), Mapping):
        plan = plan["visual_plan"]
    if not isinstance(plan, Mapping):
        raise UnifiedQAError("plan/visual_plan must be an object when supplied")
    frame_readbacks = root.get("frame_readbacks") or {}
    if not isinstance(frame_readbacks, Mapping):
        raise UnifiedQAError("frame_readbacks must be an object when supplied")
    audio_program = root.get("audio_program")
    if audio_program is None and (
        root.get("audio_events") is not None
        or plan_meta.get("audio_events") is not None
        or plan.get("audio_events") is not None
    ):
        audio_program = {
            "events": root.get(
                "audio_events",
                plan_meta.get("audio_events", plan.get("audio_events")),
            ),
            "timeline": root.get("clock") or plan_meta.get("clock") or {},
        }
    audio_program = audio_program or {}
    if not isinstance(audio_program, Mapping):
        raise UnifiedQAError("audio_program must be an object when supplied")
    timeline = audio_program.get("timeline")
    if not isinstance(timeline, Mapping):
        timeline = {}
    clock = (
        root.get("clock")
        or frame_readbacks.get("clock")
        or plan_meta.get("clock")
        or plan.get("clock")
        or timeline
    )
    if not isinstance(clock, Mapping):
        raise UnifiedQAError("episode bundle requires a clock")
    frame_count = _positive_int(
        _first(clock, "frame_count", "frames"), name="clock.frame_count"
    )
    frame_rate = _finite_number(
        _first(clock, "frame_rate_hz", "video_fps", "fps"),
        name="clock.frame_rate_hz",
    )
    if frame_rate <= 0.0:
        raise UnifiedQAError("clock.frame_rate_hz must be positive")
    sample_rate = _positive_int(
        _first(clock, "sample_rate_hz", "audio_sample_rate_hz") or 16000,
        name="clock.sample_rate_hz",
    )
    sample_count_raw = _first(clock, "sample_count", "audio_sample_count")
    sample_count = (
        int(round(frame_count / frame_rate * sample_rate))
        if sample_count_raw is None
        else _positive_int(sample_count_raw, name="clock.sample_count")
    )
    duration = frame_count / frame_rate

    actor_container = (
        root.get("actors")
        or plan.get("actors")
        or frame_readbacks.get("actors")
        or {}
    )
    actor_records: dict[str, Mapping[str, Any]] = {}
    if isinstance(actor_container, Mapping):
        for actor_id, value in actor_container.items():
            if isinstance(actor_id, str) and isinstance(value, Mapping):
                actor_records[actor_id] = value
            elif isinstance(actor_id, str):
                actor_records[actor_id] = {}
    elif _is_sequence(actor_container):
        for value in actor_container:
            if not isinstance(value, Mapping):
                continue
            actor_id = _first(value, "actor_id", "instance_id", "source_slot_id")
            if isinstance(actor_id, str) and actor_id:
                actor_records[actor_id] = value
    actor_ids = list(actor_records)
    if not actor_ids and isinstance(frame_readbacks.get("actors"), Mapping):
        actor_ids = [
            actor_id
            for actor_id in frame_readbacks["actors"]
            if isinstance(actor_id, str)
        ]
        actor_records = {actor_id: {} for actor_id in actor_ids}
    if not actor_ids:
        raise UnifiedQAError("episode bundle has no actor records")

    plan_frames = plan.get("frames")
    actors: dict[str, dict[str, Any]] = {}
    for actor_id in actor_ids:
        record = actor_records.get(actor_id, {})
        actor_readbacks = _records_for_actor(
            frame_readbacks.get("actors"), actor_id, frame_count=frame_count
        )
        emitter_readbacks = _records_for_actor(
            frame_readbacks.get("emitters"), actor_id, frame_count=frame_count
        )
        plan_actor_frames = _plan_actor_records(
            plan_frames, actor_id, frame_count=frame_count
        )
        root_positions = _position_series(actor_readbacks, default_ue_cm=True)
        if root_positions is None:
            root_positions = _position_series(plan_actor_frames, default_ue_cm=False)
        emitter_positions = _position_series(emitter_readbacks, default_ue_cm=True)
        if emitter_positions is None:
            emitter_positions = _position_series(actor_readbacks, default_ue_cm=True)
        if emitter_positions is None:
            emitter_positions = root_positions
        explicit_motion: Sequence[Any] | None = None
        if plan_actor_frames is not None:
            candidate_motion = [
                _first(frame, "moving", "is_moving") for frame in plan_actor_frames
            ]
            if all(isinstance(value, bool) for value in candidate_motion):
                explicit_motion = candidate_motion
        speeds, moving = _motion_series(
            root_positions or emitter_positions,
            frame_rate_hz=frame_rate,
            explicit=explicit_motion,
        )
        appearance = _appearance_from_record(record)
        if appearance is None and plan_actor_frames:
            for frame in plan_actor_frames:
                appearance = _appearance_from_record(frame)
                if appearance is not None:
                    break
        actors[actor_id] = {
            "actor_id": actor_id,
            "asset_id": _first(record, "asset_id", "entity_asset_id"),
            "display_label": _actor_label(actor_id, record, appearance),
            "appearance": appearance,
            "species_id": _first(record, "species_id", "entity_class"),
            "root_positions_m": root_positions,
            "emitter_positions_m": emitter_positions,
            "speed_mps": speeds,
            "moving": moving,
            "source": {
                "actor_binding": "explicit" if record else "frame_readbacks",
                "root_readbacks": actor_readbacks is not None,
                "emitter_readbacks": emitter_readbacks is not None,
                "plan_frames": plan_actor_frames is not None,
            },
        }

    audio_readback_value = _first(root, "audio_readback", "audio_validation", "audio")
    audio_readback = (
        audio_readback_value if isinstance(audio_readback_value, Mapping) else None
    )
    listener = _listener_block(
        root, frame_readbacks, audio_readback, plan, frame_count=frame_count
    )
    audio = _validate_audio_readback(
        root,
        clock={
            "frame_count": frame_count,
            "frame_rate_hz": frame_rate,
            "sample_rate_hz": sample_rate,
            "sample_count": sample_count,
        },
    )
    binding_root = dict(root)
    for key in (
        "voice_bindings",
        "voice_binding",
        "source_endpoint_bindings",
        "source_endpoint_registry",
        "sound_registry",
    ):
        if key not in binding_root and key in plan_meta:
            binding_root[key] = plan_meta[key]
    endpoint_index = _endpoint_index(binding_root)
    by_sound, by_actor = _voice_binding_index(binding_root)
    sound_registry = _sound_registry_index(binding_root)
    event_values = audio_program.get(
        "events",
        root.get("events", plan_meta.get("audio_events", [])),
    )
    if not _is_sequence(event_values):
        event_values = []
    events: list[dict[str, Any]] = []
    unresolved_event_ids: list[str] = []
    for ordinal, value in enumerate(event_values):
        if not isinstance(value, Mapping):
            raise UnifiedQAError("audio_program events must be objects")
        event_id = _first(value, "event_id", "id") or f"event_{ordinal:04d}"
        if not isinstance(event_id, str) or not event_id:
            raise UnifiedQAError("audio event ids must be readable strings")
        sound_id = _first(value, "sound_asset_id", "sound_id")
        if not isinstance(sound_id, str) or not sound_id:
            voice_candidate = by_actor.get(
                _first(value, "actor_id", "source_slot_id")
            )
            sound_id = _first(voice_candidate or {}, "sound_asset_id", "sound_id")
        if not isinstance(sound_id, str) or not sound_id:
            sound_id = None
        actor_id = _resolve_event_actor(
            value,
            actor_ids=actor_ids,
            endpoint_index=endpoint_index,
            by_sound=by_sound,
        )
        if actor_id is None:
            unresolved_event_ids.append(event_id)
        sound_record = sound_registry.get(sound_id) if sound_id else None
        voice_record = by_sound.get(sound_id) if sound_id else None
        transcript, statement_id, language, sound_class, class_explicit = _content_from_event(
            value, sound_record=sound_record, voice_record=voice_record
        )
        if sound_class is None and actor_id:
            species = actors.get(actor_id, {}).get("species_id")
            if isinstance(species, str) and species:
                sound_class = f"{species.casefold()}_vocalization"
        start_s, end_s, start_frame, end_frame = _event_time(
            value,
            clock={
                "frame_rate_hz": frame_rate,
                "sample_rate_hz": sample_rate,
                "time_base_hz": clock.get("time_base_hz", 48000),
            },
            timeline=timeline,
        )
        start_frame = max(0, min(frame_count, start_frame))
        end_frame = max(0, min(frame_count, end_frame))
        events.append(
            {
                "event_id": event_id,
                "actor_id": actor_id,
                "sound_asset_id": sound_id,
                "start_s": start_s,
                "end_s": end_s,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_tick": _first(value, "start_tick", "start_ticks"),
                "end_tick": _first(value, "end_tick_exclusive", "end_tick", "end_ticks"),
                "transcript": transcript,
                "statement_id": statement_id,
                "language": language,
                "sound_class": sound_class,
                "sound_class_explicit": class_explicit,
                "source_endpoint_id": _first(value, "source_endpoint_id", "endpoint_id"),
                "event_unit": value.get("event_unit"),
                "event_segmentation": copy.deepcopy(
                    _event_source_field(
                        value,
                        "event_segmentation",
                        "event_segmentation_status",
                        sound_record=sound_record,
                        voice_record=voice_record,
                    )
                ),
                "source_qc": copy.deepcopy(
                    _event_source_field(
                        value,
                        "source_qc",
                        sound_record=sound_record,
                        voice_record=voice_record,
                    )
                ),
                "source_activity_intervals_samples": copy.deepcopy(
                    value.get("source_activity_intervals_samples")
                ),
                "source_record": "audio_program",
            }
        )
    events.sort(key=lambda item: (item["start_s"], item["event_id"]))

    activity_values: list[Any] = [
        root.get("source_activity_intervals_samples"),
        audio_program.get("source_activity_intervals_samples"),
        audio_readback.get("source_activity_intervals_samples")
        if isinstance(audio_readback, Mapping)
        else None,
    ]
    activity_values.extend(
        {
            "event_id": event["event_id"],
            "source_activity_intervals_samples": event.get(
                "source_activity_intervals_samples"
            ),
        }
        for event in events
        if event.get("source_activity_intervals_samples") is not None
    )
    for report_value in (
        root.get("audio_readback"),
        root.get("research_report"),
        root.get("audio_report"),
    ):
        if isinstance(report_value, Mapping):
            activity_values.append(
                report_value.get("source_activity_intervals_samples")
            )
            report_events = report_value.get("events")
            if _is_sequence(report_events):
                activity_values.extend(report_events)
    (
        source_activity_by_event,
        source_activity_present,
        source_activity_event_ids,
    ) = _source_activity_index(
        activity_values,
        sample_count=sample_count,
    )
    for event_id in source_activity_event_ids:
        source_activity_by_event.setdefault(event_id, [])
    source_activity_evidence_complete = bool(events) and (
        source_activity_present
        and all(
            event["event_id"] in source_activity_event_ids
            for event in events
        )
    )
    for event in events:
        event_id = event["event_id"]
        if event_id in source_activity_event_ids:
            event["source_activity_intervals_samples"] = copy.deepcopy(
                source_activity_by_event.get(event_id, [])
            )
            event["source_activity_evidence_status"] = "observed"
        else:
            event.pop("source_activity_intervals_samples", None)
            event["source_activity_evidence_status"] = "missing"

    visibility_value = root.get("pixel_visibility_truth") or root.get("pixel_truth")
    visibility = _visibility_index(
        visibility_value, actor_ids=actor_ids, frame_count=frame_count
    )
    visibility_meta = {
        "status": (
            visibility_value.get("status")
            if isinstance(visibility_value, Mapping)
            else "not_run"
        ),
        "resolution_hw": (
            copy.deepcopy(visibility_value.get("resolution_hw"))
            if isinstance(visibility_value, Mapping)
            else None
        ),
        "camera_pose_ids": (
            copy.deepcopy(visibility_value.get("camera_pose_ids"))
            if isinstance(visibility_value, Mapping)
            else None
        ),
        "schema": (
            visibility_value.get("schema")
            if isinstance(visibility_value, Mapping)
            else None
        ),
    }
    appearance_review_value = (
        root.get("appearance_review")
        or plan_meta.get("appearance_review")
        or plan.get("appearance_review")
    )
    appearance_review: dict[str, Any] = {}
    if isinstance(appearance_review_value, Mapping):
        default_review_status = appearance_review_value.get("status")
        records = appearance_review_value.get(
            "actors",
            appearance_review_value.get(
                "items", appearance_review_value.get("actor_reviews")
            ),
        )
        if isinstance(records, Mapping):
            for actor_id, record in records.items():
                if isinstance(actor_id, str) and isinstance(record, Mapping):
                    normalized = dict(record)
                    if "status" not in normalized and isinstance(
                        default_review_status, str
                    ):
                        normalized["status"] = default_review_status
                    appearance_review[actor_id] = normalized
        elif _is_sequence(records):
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                actor_id = _first(record, "actor_id", "instance_id")
                if isinstance(actor_id, str):
                    normalized = dict(record)
                    if "status" not in normalized and isinstance(
                        default_review_status, str
                    ):
                        normalized["status"] = default_review_status
                    appearance_review[actor_id] = normalized
        else:
            for actor_id, record in appearance_review_value.items():
                if isinstance(actor_id, str) and isinstance(record, Mapping):
                    appearance_review[actor_id] = dict(record)
    elif _is_sequence(appearance_review_value):
        for record in appearance_review_value:
            if not isinstance(record, Mapping):
                continue
            actor_id = _first(record, "actor_id", "instance_id")
            if isinstance(actor_id, str):
                appearance_review[actor_id] = dict(record)
    for actor_id, record in actor_records.items():
        if actor_id in appearance_review:
            continue
        candidate = record.get("appearance_review")
        if isinstance(candidate, Mapping):
            appearance_review[actor_id] = dict(candidate)
    sampling_value = root.get("sampling") or root.get("qa_sampling") or {}
    sampling = (
        copy.deepcopy(dict(sampling_value))
        if isinstance(sampling_value, Mapping)
        else {}
    )
    public_entities = root.get("entities")
    if not isinstance(public_entities, Mapping):
        request_value = plan_meta.get("request")
        if isinstance(request_value, Mapping):
            public_entities = request_value.get("entities")
    if isinstance(public_entities, Mapping):
        # Keep the existing public/qa_sampling shape while making the
        # request's entity-count choices available to post-capture QA.
        sampling.setdefault("entities", copy.deepcopy(dict(public_entities)))

    facts = {
        "schema": UNIFIED_FACT_SCHEMA,
        "status": "pass",
        "claim_boundary": (
            "Facts are deterministic research evidence derived from supplied "
            "native readbacks; no formal dataset admission or learned-model "
            "modality claim is made."
        ),
        "episode_id": episode_id,
        "catalog_version": CATALOG_VERSION,
        "time": {
            "frame_count": frame_count,
            "frame_rate_hz": frame_rate,
            "sample_rate_hz": sample_rate,
            "sample_count": sample_count,
            "duration_seconds": duration,
            "time_base_hz": int(clock.get("time_base_hz", 48000)),
        },
        "actors": actors,
        "events": events,
        "listener": listener,
        "audio": {
            **audio,
            **(
                {
                    "source_activity_intervals_samples": copy.deepcopy(
                        source_activity_by_event
                    ),
                    "source_activity_coordinate_space": "episode_sample_clock",
                }
                if source_activity_present
                else {}
            ),
        },
        "source_activity_intervals_samples": (
            copy.deepcopy(source_activity_by_event)
            if source_activity_present
            else None
        ),
        "source_activity_evidence_present": source_activity_present,
        "source_activity_evidence_complete": source_activity_evidence_complete,
        "source_activity_evidence_by_event": {
            event["event_id"]: (
                "observed"
                if event["event_id"] in source_activity_event_ids
                else "missing"
            )
            for event in events
        },
        "visibility": visibility,
        "visibility_meta": visibility_meta,
        "camera_calibration": copy.deepcopy(root.get("camera_calibration")),
        "appearance_review": appearance_review,
        "input_summary": {
            "plan_present": bool(plan),
            "frame_readbacks_present": bool(frame_readbacks),
            "pixel_visibility_truth_present": isinstance(visibility_value, Mapping),
            "audio_program_present": bool(audio_program),
            "source_activity_intervals_samples_present": source_activity_present,
            "source_activity_evidence_complete": source_activity_evidence_complete,
            "unresolved_event_ids": unresolved_event_ids,
        },
        "source_paths": {
            "plan": _first(root, "plan_path"),
            "video": _first(root, "video_path", "video_uri"),
            "mixture_audio": audio.get("path"),
            "frame_readbacks": _first(root, "frame_readbacks_path"),
            "pixel_visibility_truth": _first(root, "pixel_visibility_truth_path"),
            "audio_program": _first(root, "audio_program_path"),
            "audio_readback": _first(root, "audio_readback_path"),
            "research_report": _first(root, "research_report_path"),
            "appearance_review": _first(root, "appearance_review_path"),
            "occluder_evidence": _first(root, "occluder_evidence_path"),
            "occluder_registry": _first(root, "occluder_registry_path"),
        },
        "sampling": sampling,
        "sampling_policy": _first(root, "sampling_policy") or _first(plan_meta, "sampling_policy"),
        "occluder_registry": root.get("occluder_registry") or {},
        "occluder_evidence": root.get("occluder_evidence") or {},
    }
    return facts


class _Deferred(Exception):
    """Internal control flow for a question whose evidence is unavailable."""

    def __init__(self, code: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.extra = extra


def _defer(code: str, detail: str, **extra: Any) -> None:
    raise _Deferred(code, detail, **extra)


def _actor(facts: Mapping[str, Any], actor_id: str | None) -> Mapping[str, Any]:
    actors = facts.get("actors")
    value = actors.get(actor_id) if isinstance(actors, Mapping) else None
    if not isinstance(value, Mapping):
        _defer("unknown_actor", f"event target {actor_id!r} is not a registered actor")
    return value


def _bound_events(facts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = facts.get("events")
    if not _is_sequence(events):
        _defer("missing_events", "audio_program has no event list")
    result = [
        event
        for event in events
        if isinstance(event, Mapping) and isinstance(event.get("actor_id"), str)
    ]
    if len(result) != len(events):
        # Unresolved events cannot silently become a negative answer or a
        # smaller event sequence. QA-23 counts raw event IDs separately;
        # attribution questions must stop here.
        _defer(
            "unresolved_event_attribution",
            "a source event is not bound to a registered actor",
        )
    return result


def _event_frame(event: Mapping[str, Any], key: str) -> int:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _defer("missing_event_frame", f"event {event.get('event_id')!r} has no {key}")
    return value


def _event_for_actor(
    facts: Mapping[str, Any],
    actor_id: str,
    *,
    require_content: bool = False,
) -> list[Mapping[str, Any]]:
    events = [
        event
        for event in _bound_events(facts)
        if event.get("actor_id") == actor_id
    ]
    if require_content:
        events = [
            event
            for event in events
            if isinstance(event.get("transcript"), str)
            and bool(event.get("transcript", "").strip())
        ]
    ordered = sorted(
        events,
        key=lambda event: (float(event["start_s"]), event["event_id"]),
    )
    candidate = facts.get("_p8_candidate")
    preferred_id = (
        candidate.get("event_id")
        if isinstance(candidate, Mapping)
        else None
    )
    if preferred_id is not None:
        ordered = [
            *[
                event
                for event in ordered
                if event.get("event_id") == preferred_id
            ],
            *[
                event
                for event in ordered
                if event.get("event_id") != preferred_id
            ],
        ]
    return ordered


def _first_event(facts: Mapping[str, Any], actor_id: str) -> Mapping[str, Any]:
    events = _event_for_actor(facts, actor_id)
    if not events:
        _defer("target_has_no_event", f"actor {actor_id!r} has no bound sound event")
    return min(events, key=lambda event: (float(event["start_s"]), event["event_id"]))


def _earliest_event(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    events = _bound_events(facts)
    if not events:
        _defer("missing_bound_events", "no sound event resolves to an actor")
    events = sorted(events, key=lambda event: (float(event["start_s"]), event["event_id"]))
    earliest = float(events[0]["start_s"])
    ties = [event for event in events if math.isclose(float(event["start_s"]), earliest, abs_tol=1.0e-9)]
    if len({event.get("actor_id") for event in ties}) != 1:
        _defer(
            "non_unique_earliest",
            "multiple actors share the earliest sound-event start",
            event_ids=[event.get("event_id") for event in ties],
        )
    return ties[0]


def _appearance_candidates(
    facts: Mapping[str, Any],
    *,
    require_unique: bool = True,
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    actors = facts.get("actors")
    if not isinstance(actors, Mapping):
        _defer("missing_actors", "episode has no actor table")
    values: dict[str, list[str]] = {}
    result: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for actor_id, actor in actors.items():
        if not isinstance(actor, Mapping):
            continue
        appearance = actor.get("appearance")
        if not isinstance(appearance, Mapping):
            continue
        value = appearance.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        review = facts.get("appearance_review", {}).get(actor_id)
        if not isinstance(review, Mapping) or review.get("status") not in {
            "pass",
            "reviewed",
            "human_reviewed",
            "astra_reviewed",
        }:
            continue
        reviewed_value = review.get(
            "value",
            review.get("attribute_value", review.get("reviewed_value")),
        )
        frame_refs = review.get("frame_refs", review.get("frames"))
        normalized_refs = []
        if _is_sequence(frame_refs):
            for reference in frame_refs:
                if isinstance(reference, Mapping):
                    reference = _first(reference, "frame_index", "frame")
                if isinstance(reference, bool):
                    normalized_refs.append(None)
                elif isinstance(reference, int):
                    normalized_refs.append(reference)
                else:
                    normalized_refs.append(reference)
        if (
            not isinstance(reviewed_value, str)
            or reviewed_value.strip().casefold() != value.strip().casefold()
            or not _is_sequence(frame_refs)
            or not frame_refs
            or any(
                isinstance(reference, int)
                and not 0 <= reference < int(facts["time"]["frame_count"])
                for reference in normalized_refs
            )
        ):
            continue
        values.setdefault(_appearance_family(appearance) or value.strip(), []).append(str(actor_id))
        result.append((str(actor_id), actor, appearance))
    if not result:
        _defer(
            "appearance_review_missing",
            "no actor has a matching reviewed appearance value and frame reference; "
            "a native mask alone does not certify fine-grained appearance",
        )
    if require_unique:
        duplicate = {value: ids for value, ids in values.items() if len(ids) > 1}
        if duplicate:
            _defer(
                "appearance_not_unique",
                "appearance selector matches more than one actor; two registered "
                "values in one colour family are one colour in the rendered frame",
                duplicate_colour_families=duplicate,
            )
        labels: dict[str, list[str]] = {}
        for actor_id, _actor, appearance in result:
            label = _strip_display_version_suffix(appearance.get("label"))
            if label:
                labels.setdefault(label.casefold(), []).append(actor_id)
        duplicate_labels = {
            label: ids for label, ids in labels.items() if len(ids) > 1
        }
        if duplicate_labels:
            _defer(
                "appearance_display_labels_not_unique",
                "reviewed appearance labels do not distinguish the target actors",
                duplicate_labels=duplicate_labels,
            )
    return result


def _actor_options(
    facts: Mapping[str, Any],
    actor_ids: Sequence[str],
) -> list[dict[str, str]]:
    actors = facts["actors"]
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    for actor_id in actor_ids:
        actor = actors.get(actor_id)
        if not isinstance(actor, Mapping):
            continue
        appearance = actor.get("appearance")
        if isinstance(appearance, Mapping):
            label_en, label_zh = _appearance_phrases(appearance)
        else:
            fallback = _strip_display_version_suffix(actor.get("display_label"))
            label_en = label_zh = fallback or str(actor_id)
        value = str(actor_id)
        if value in seen:
            continue
        seen.add(value)
        options.append(
            {
                "value": value,
                "label_en": label_en,
                "label_zh": label_zh,
                "allow_value": False,
            }
        )
    labels = [option["label_en"] for option in options]
    if len(labels) != len(set(labels)):
        _defer(
            "appearance_display_labels_not_unique",
            "reviewed appearance labels do not distinguish the answer options",
            labels=labels,
        )
    families: dict[str, list[str]] = {}
    for option in options:
        actor = actors.get(option["value"])
        appearance = actor.get("appearance") if isinstance(actor, Mapping) else None
        family = _appearance_family(appearance) if isinstance(appearance, Mapping) else ""
        if family:
            families.setdefault(family, []).append(option["value"])
    shared = {family: members for family, members in families.items() if len(members) > 1}
    if shared:
        _defer(
            "appearance_options_share_a_colour_family",
            "two answer options are the same colour in the rendered frame",
            colour_families=shared,
        )
    return options


_APPEARANCE_WORDS = {
    "blue": ("blue-shirt person", "蓝色上衣的人"),
    "pink": ("pink-shirt person", "粉色上衣的人"),
    "green": ("green-shirt person", "绿色上衣的人"),
    "white": ("white-shirt person", "白色上衣的人"),
    "burgundy": ("burgundy-shirt person", "酒红色上衣的人"),
    "yellow": ("yellow-shirt person", "黄色上衣的人"),
    "black_white": ("black-and-white individual", "黑白外观个体"),
    "black": ("black individual", "黑色外观个体"),
    "ruddy": ("ruddy individual", "红棕色外观个体"),
}


def _appearance_family(appearance: Any) -> str:
    """The colour family a registered appearance value can be told apart by.

    Two values in one family are one colour as far as a rendered frame is
    concerned: pink and burgundy are a tint and a shade of the same hue, and no
    participant looking at the video could be expected to separate them. A
    question that asks who wore which colour therefore has to be built on values
    that land in different families, which is stricter than the plain string
    inequality this used to be. A value with no family falls back to itself, so
    an unrecognised value is still only equal to an identical one.
    """
    from avengine.rooms.appearance_color import appearance_distinction_family

    if isinstance(appearance, Mapping):
        value = appearance.get("value", appearance.get("attribute_value"))
        kind = str(appearance.get("entity_kind") or "")
    else:
        value, kind = appearance, ""
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    return appearance_distinction_family(text, kind) or text


def _appearance_phrases(appearance: Mapping[str, Any]) -> tuple[str, str]:
    value = str(appearance.get("value", "")).strip()
    label = appearance.get("label")
    label_text = _strip_display_version_suffix(label)
    mapped = _APPEARANCE_WORDS.get(value.casefold())
    if mapped is not None and (
        not label_text
        or label_text.casefold() == value.casefold()
        or any(
            token in label_text.casefold()
            for token in ("human", "person", "actor", "shirt", "top")
        )
    ):
        return mapped
    if label_text and label_text.casefold() != value.casefold():
        return label_text, label_text
    if mapped is not None:
        return mapped
    _defer(
        "missing_appearance_display_label",
        "reviewed appearance has no human-readable display label",
        appearance_value=value,
    )


def _event_number(facts: Mapping[str, Any], event: Mapping[str, Any]) -> int:
    events = sorted(
        _bound_events(facts),
        key=lambda item: (float(item["start_s"]), item["event_id"]),
    )
    for index, candidate in enumerate(events, start=1):
        if candidate.get("event_id") == event.get("event_id"):
            return index
    _defer(
        "event_not_in_episode",
        f"event {event.get('event_id')!r} is not in the episode event table",
    )


def _event_anchor(facts: Mapping[str, Any], event: Mapping[str, Any]) -> tuple[str, str]:
    number = _event_number(facts, event)
    return (
        f"the {number}{'st' if number == 1 else 'nd' if number == 2 else 'rd' if number == 3 else 'th'} independent sound event",
        f"第{number}个独立发声事件",
    )


def _reviewed_appearances(
    facts: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        actor_id: appearance
        for actor_id, _actor_record, appearance in _appearance_candidates(
            facts, require_unique=False
        )
    }


def _appearance_options(
    facts: Mapping[str, Any],
    *,
    include_values: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    candidates = _appearance_candidates(facts, require_unique=False)
    wanted = set(include_values) if include_values is not None else None
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    for _actor_id, _actor, appearance in candidates:
        value = str(appearance["value"])
        if wanted is not None and value not in wanted:
            continue
        if value in seen:
            continue
        label_en, label_zh = _appearance_phrases(appearance)
        options.append(
            {
                "value": value,
                "label_en": label_en,
                "label_zh": label_zh,
                "allow_value": False,
            }
        )
        seen.add(value)
    labels = [option["label_en"] for option in options]
    if len(labels) != len(set(labels)):
        _defer(
            "appearance_display_labels_not_unique",
            "reviewed appearance labels do not distinguish the answer options",
            labels=labels,
        )
    families: dict[str, list[str]] = {}
    for option in options:
        families.setdefault(_appearance_family(option["value"]), []).append(option["value"])
    shared = {family: members for family, members in families.items() if len(members) > 1}
    if shared:
        _defer(
            "appearance_options_share_a_colour_family",
            "two answer options are the same colour in the rendered frame",
            colour_families=shared,
        )
    return options


def _appearance_review_for(
    facts: Mapping[str, Any],
    actor_id: str,
) -> dict[str, Any] | None:
    value = facts.get("appearance_review", {}).get(actor_id)
    if not isinstance(value, Mapping):
        return None
    result = {
        "actor_id": actor_id,
        "status": value.get("status"),
        "value": value.get("value", value.get("attribute_value")),
        "attribute_field": value.get(
            "attribute_field", value.get("field")
        ),
        "frame_refs": copy.deepcopy(
            value.get("frame_refs", value.get("frames", []))
        ),
    }
    source_path = facts.get("source_paths", {}).get("appearance_review")
    if source_path:
        result["source_path"] = source_path
    return result


def _state(facts: Mapping[str, Any], actor_id: str, frame: int) -> Mapping[str, Any]:
    frames = facts.get("visibility", {}).get(actor_id)
    if not isinstance(frames, Mapping):
        _defer(
            "missing_pixel_visibility",
            f"pixel visibility truth is unavailable for {actor_id!r} at frame {frame}",
            actor_id=actor_id,
            frame=frame,
        )
    missing = object()
    value = frames.get(frame, missing)
    if value is missing:
        value = frames.get(str(frame), missing)
    if value is missing:
        _defer(
            "missing_pixel_visibility",
            f"pixel visibility truth is unavailable for {actor_id!r} at frame {frame}",
            actor_id=actor_id,
            frame=frame,
        )
    if not isinstance(value, Mapping) or value.get("state") not in VISIBILITY_STATES:
        _defer(
            "invalid_pixel_visibility",
            f"pixel visibility state is unavailable for {actor_id!r} at frame {frame}",
            actor_id=actor_id,
            frame=frame,
        )
    return value


def _source_activity_for_event(
    facts: Mapping[str, Any],
    event_id: str,
) -> list[Mapping[str, Any]]:
    events = facts.get("events")
    if _is_sequence(events):
        for event in events:
            if (
                isinstance(event, Mapping)
                and event.get("event_id") == event_id
                and _is_sequence(event.get("source_activity_intervals_samples"))
            ):
                return [
                    row
                    for row in event["source_activity_intervals_samples"]
                    if isinstance(row, Mapping)
                ]
    value = facts.get("source_activity_intervals_samples")
    if not isinstance(value, Mapping):
        audio = facts.get("audio")
        value = (
            audio.get("source_activity_intervals_samples")
            if isinstance(audio, Mapping)
            else None
        )
    rows = value.get(event_id) if isinstance(value, Mapping) else None
    return [
        row
        for row in rows
        if isinstance(row, Mapping)
    ] if _is_sequence(rows) else []


def _source_activity_present(facts: Mapping[str, Any]) -> bool:
    if facts.get("source_activity_evidence_present") is True:
        return True
    if facts.get("source_activity_intervals_samples") is not None:
        return True
    audio = facts.get("audio")
    return isinstance(audio, Mapping) and (
        audio.get("source_activity_intervals_samples") is not None
    )


def _active_at(
    facts: Mapping[str, Any],
    frame: int,
    *,
    require_source_activity: bool = False,
) -> list[Mapping[str, Any]]:
    if require_source_activity and (
        not _source_activity_present(facts)
        or facts.get("source_activity_evidence_complete") is False
    ):
        _defer(
            "missing_source_activity_readback",
            "QA-18 requires episode source_activity_intervals_samples",
        )
    sample = int(
        round(
            float(frame)
            * float(facts["time"]["sample_rate_hz"])
            / float(facts["time"]["frame_rate_hz"])
        )
    )
    active: list[Mapping[str, Any]] = []
    for event in _bound_events(facts):
        rows = _source_activity_for_event(facts, str(event["event_id"]))
        if any(
            int(row.get("start_sample", 0)) <= sample
            < int(row.get("end_sample_exclusive", 0))
            for row in rows
        ):
            active.append(event)
    return active


def _sampling_value(
    facts: Mapping[str, Any],
    qa_id: str,
    *keys: str,
) -> Any:
    sampling = facts.get("sampling")
    if not isinstance(sampling, Mapping):
        return None
    nested_sampling = sampling.get("qa_sampling")
    if isinstance(nested_sampling, Mapping):
        sampling = {**sampling, **nested_sampling}
    qa_keys = (
        qa_id,
        qa_id.lower(),
        qa_id.replace("-", "_"),
        qa_id.lower().replace("-", "_"),
    )
    for key in keys:
        value = sampling.get(key)
        if isinstance(value, Mapping):
            for candidate_key in qa_keys:
                if candidate_key in value:
                    return value[candidate_key]
        elif value is not None:
            return value
    queries = sampling.get("queries")
    if isinstance(queries, Mapping):
        query = None
        for candidate_key in qa_keys:
            if candidate_key in queries:
                query = queries[candidate_key]
                break
        if isinstance(query, Mapping):
            for key in keys:
                if key in query:
                    return query[key]
    return None


def _query_frame(
    facts: Mapping[str, Any],
    qa_id: str,
    *,
    event: Mapping[str, Any] | None = None,
    after_event: bool = False,
    require_declared: bool = False,
) -> tuple[int, str]:
    frame_count = int(facts["time"]["frame_count"])
    if frame_count <= 0:
        _defer("invalid_frame_clock", "frame_count must be positive")
    fields = (
        ("post_sound_query_frame", "query_frame")
        if after_event
        else ("query_frame", "at_frame")
    )
    if event is not None:
        for field in fields:
            if field in event and event[field] is not None:
                frame = _resolve_query_frame_spec(
                    facts, qa_id, event[field], source=f"event.{field}"
                )
                return frame, f"event.{field}"
    value = _sampling_value(
        facts, qa_id, "query_frame_by_qa", "query_frames", *fields
    )
    if value is not None:
        frame = _resolve_query_frame_spec(
            facts, qa_id, value, source="sampling"
        )
        return frame, "sampling"
    sampling = facts.get("sampling")
    if isinstance(sampling, Mapping):
        nested = sampling.get("qa_sampling")
        if isinstance(nested, Mapping):
            sampling = {**sampling, **nested}
        policy = sampling.get("query_time_policy") or sampling.get("policy")
        if policy == "uniform_in_legal_window":
            window = _sampling_value(
                facts, qa_id, "legal_window_by_qa", "legal_windows",
                "query_windows",
            )
            if window is None:
                derived = _derived_legal_query_windows(
                    facts, qa_id, event=event
                )
                if derived is None:
                    _defer(
                        "sampling_window_missing",
                        f"{qa_id} uses uniform_in_legal_window without a legal window",
                    )
                if not derived:
                    _defer(
                        "no_valid_post_sound_window",
                        f"{qa_id} has no legal query frame",
                    )
                _record_derived_query_windows(
                    facts, qa_id, derived, event=event
                )
                frame = _sample_frame_from_windows(facts, qa_id, derived)
                return frame, "derived_uniform_in_legal_window"
            frame = _resolve_query_frame_spec(
                facts,
                qa_id,
                {"policy": policy, "window": window},
                source="uniform_in_legal_window",
            )
            return frame, "uniform_in_legal_window"

    if require_declared:
        _defer("missing_query_frame", f"{qa_id} requires an explicit query frame")
    if after_event and event is not None:
        end_frame = _event_frame(event, "end_frame")
        return min(frame_count - 1, max(end_frame, end_frame + 1)), "derived_after_event"
    return min(frame_count - 1, max(0, frame_count // 2)), "derived_midpoint"


def _query_time(
    facts: Mapping[str, Any],
    qa_id: str,
    *,
    event: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    value = _sampling_value(facts, qa_id, "query_time_s_by_qa", "query_times_s")
    if value is not None:
        return (
            _resolve_query_time_spec(
                facts, qa_id, value, source="sampling"
            ),
            "sampling",
        )
    sampling = facts.get("sampling")
    if isinstance(sampling, Mapping):
        nested = sampling.get("qa_sampling")
        if isinstance(nested, Mapping):
            sampling = {**sampling, **nested}
        policy = sampling.get("query_time_policy") or sampling.get("policy")
        if policy == "uniform_in_legal_window":
            window = _sampling_value(
                facts, qa_id, "legal_window_by_qa", "legal_windows",
                "query_windows",
            )
            if window is None:
                derived = _derived_legal_query_windows(facts, qa_id)
                if derived is None:
                    _defer(
                        "sampling_window_missing",
                        f"{qa_id} uses uniform_in_legal_window without a legal window",
                    )
                if not derived:
                    _defer(
                        "sampling_window_missing",
                        f"{qa_id} has no safe query frame",
                    )
                _record_derived_query_windows(facts, qa_id, derived)
                frame = _sample_frame_from_windows(facts, qa_id, derived)
                return (
                    frame / float(facts["time"]["frame_rate_hz"]),
                    "derived_uniform_in_legal_window",
                )
            return (
                _resolve_query_time_spec(
                    facts,
                    qa_id,
                    {"policy": policy, "window": window},
                    source="uniform_in_legal_window",
                ),
                "uniform_in_legal_window",
            )

    if event is not None:
        return (float(event["start_s"]) + float(event["end_s"])) / 2.0, "event_midpoint"
    duration = float(facts["time"]["duration_seconds"])
    return duration / 2.0, "derived_midpoint"


def _silent_after(
    facts: Mapping[str, Any],
    event: Mapping[str, Any],
    query_frame: int,
) -> dict[str, Any]:
    end_frame = _event_frame(event, "end_frame")
    if query_frame <= end_frame:
        _defer(
            "query_not_after_sound",
            "the query frame is not after the anchor event",
            event_id=event.get("event_id"),
            query_frame=query_frame,
            end_frame=end_frame,
        )
    audio = facts.get("audio")
    wet_tails = audio.get("wet_tail_intervals") if isinstance(audio, Mapping) else None
    if not _is_sequence(wet_tails) or not wet_tails:
        _defer(
            "missing_wet_tail_readback",
            "post-sound silence is not proven without final wet-tail intervals",
            event_id=event.get("event_id"),
        )
    query_time = query_frame / float(facts["time"]["frame_rate_hz"])
    end_time = float(event["end_s"])
    overlaps = [
        other.get("event_id")
        for other in facts.get("events", [])
        if isinstance(other, Mapping)
        and other.get("event_id") != event.get("event_id")
        and float(other.get("start_s", 0.0)) <= query_time
        and float(other.get("end_s", 0.0)) > end_time
    ]
    if overlaps:
        _defer(
            "post_sound_window_not_silent",
            "another programmed event overlaps the post-sound query window",
            overlapping_event_ids=overlaps,
        )
    matching_tails = [
        interval
        for interval in wet_tails
        if isinstance(interval, Mapping)
        and interval.get("event_id") == event.get("event_id")
    ]
    if not matching_tails:
        _defer(
            "missing_event_wet_tail",
            "the anchor event has no final wet-tail readback interval",
            event_id=event.get("event_id"),
        )
    tail_end = max(float(interval["end_s"]) for interval in matching_tails)
    if query_time <= tail_end:
        _defer(
            "post_sound_window_in_wet_tail",
            "query frame falls inside the anchor event's measured wet tail",
            event_id=event.get("event_id"),
            wet_tail_end_s=tail_end,
            query_time_s=query_time,
        )
    return {
        "anchor_end_frame": end_frame,
        "query_frame": query_frame,
        "query_time_s": query_time,
        "overlapping_event_ids": [],
        "wet_tail_end_s": tail_end,
        "proof": "audio_program_gap_and_wet_tail_readback_v1",
    }


def _anchor_pre_silence(
    facts: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    minimum_seconds: float = 0.1,
) -> dict[str, Any]:
    start_time = float(event["start_s"])
    prior_end = 0.0
    for other in facts.get("events", []):
        if not isinstance(other, Mapping) or other.get("event_id") == event.get("event_id"):
            continue
        if float(other.get("end_s", 0.0)) <= start_time:
            prior_end = max(prior_end, float(other["end_s"]))
    audio = facts.get("audio")
    tails = audio.get("wet_tail_intervals") if isinstance(audio, Mapping) else []
    if _is_sequence(tails):
        for interval in tails:
            if not isinstance(interval, Mapping):
                continue
            if float(interval.get("end_s", 0.0)) <= start_time:
                prior_end = max(prior_end, float(interval["end_s"]))
    gap = start_time - prior_end
    if gap < minimum_seconds:
        _defer(
            "insufficient_anchor_pre_silence",
            "the sound anchor is too close to a previous event or clip start",
            anchor_start_s=start_time,
            preceding_audio_end_s=prior_end,
            required_silence_s=minimum_seconds,
        )
    return {
        "anchor_start_s": start_time,
        "preceding_audio_end_s": prior_end,
        "pre_silence_s": gap,
        "minimum_required_s": minimum_seconds,
    }


def _azimuth(
    facts: Mapping[str, Any],
    actor_id: str,
    frame: int,
) -> float:
    actor = _actor(facts, actor_id)
    positions = actor.get("emitter_positions_m") or actor.get("root_positions_m")
    listener = facts.get("listener")
    if not _is_sequence(positions) or frame >= len(positions):
        _defer("missing_source_position", f"actor {actor_id!r} has no position at frame {frame}")
    if not isinstance(listener, Mapping) or listener.get("status") != "pass":
        _defer("missing_listener_readback", "listener/camera pose readback is unavailable")
    value = _listener_azimuth(positions[frame], listener, frame)
    if value is None:
        _defer("missing_azimuth", f"cannot derive listener-relative azimuth at frame {frame}")
    return value


def _distance_at(
    facts: Mapping[str, Any],
    actor_id: str,
    frame: int,
) -> float:
    actor = _actor(facts, actor_id)
    positions = actor.get("emitter_positions_m") or actor.get("root_positions_m")
    listener = facts.get("listener")
    if not _is_sequence(positions) or frame >= len(positions):
        _defer("missing_source_position", f"actor {actor_id!r} has no position at frame {frame}")
    if not isinstance(listener, Mapping) or listener.get("status") != "pass":
        _defer("missing_listener_readback", "listener/camera pose readback is unavailable")
    value = _distance(positions[frame], listener, frame)
    if value is None:
        _defer("missing_distance", f"cannot derive listener-relative distance at frame {frame}")
    return value


def _motion_at(
    facts: Mapping[str, Any],
    actor_id: str,
    frame: int,
) -> bool:
    actor = _actor(facts, actor_id)
    moving = actor.get("moving")
    if not _is_sequence(moving) or frame >= len(moving):
        _defer("missing_motion_readback", f"motion truth is unavailable for {actor_id!r}")
    if not isinstance(moving[frame], bool):
        _defer("invalid_motion_readback", f"motion truth is not boolean for {actor_id!r}")
    return bool(moving[frame])


def _stable_motion_window(
    facts: Mapping[str, Any],
    actor_id: str,
    start_frame: int,
    end_frame: int,
) -> bool:
    if end_frame <= start_frame:
        _defer("empty_motion_window", "event does not cover a video frame")
    values = [_motion_at(facts, actor_id, frame) for frame in range(start_frame, end_frame)]
    if len(set(values)) != 1:
        _defer(
            "motion_state_changes",
            "motion state changes within the question window",
            frame_range=[start_frame, end_frame],
        )
    return values[0]


DISTANCE_TREND_DEFAULTS = {
    "min_net_change_m": 0.2,
    "reversal_tolerance_m": 0.05,
    "reversal_fraction": 0.25,
}


def _policy_number(
    facts: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: float,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    """Read one numeric policy value from the episode sampling policy.

    New thresholds arrive through configuration like every other runtime
    parameter; nothing here keys off a room, source or asset name.
    """

    owners: list[Mapping[str, Any]] = []
    sampling = facts.get("sampling")
    if isinstance(sampling, Mapping):
        nested = sampling.get("qa_sampling")
        if isinstance(nested, Mapping):
            owners.append(nested)
        acceptance = sampling.get("acceptance_policy")
        if isinstance(acceptance, Mapping):
            # The acceptance policy a bank run injects carries the same
            # numeric thresholds, so a retained-media rerun reads them too.
            owners.append(acceptance)
        owners.append(sampling)
    policy = facts.get("sampling_policy")
    if isinstance(policy, Mapping):
        owners.append(policy)
    value: Any = None
    for owner in owners:
        for key in keys:
            if key in owner:
                value = owner[key]
                break
        if value is not None:
            break
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _defer("policy_number_invalid", f"{name} must be a finite number", value=value)
    value = float(value)
    if not math.isfinite(value) or value < minimum or (
        maximum is not None and value > maximum
    ):
        _defer("policy_number_invalid", f"{name} is outside its allowed range", value=value)
    return value


def audible_frame_window(
    facts: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """The frame span over which this event's own source activity is present.

    A question about what happened "while the source was sounding" is about
    the measured activity span, not about the declared event bounds. The
    current audio policy keeps natural short pauses inside one segment, so
    the span runs from the first to the last active frame and the gaps are
    reported rather than treated as the end of the sound.
    """

    frame_count = int(facts["time"]["frame_count"])
    declared_start = max(0, _event_frame(event, "start_frame"))
    declared_end = min(
        frame_count, max(declared_start + 1, _event_frame(event, "end_frame"))
    )
    declared = [declared_start, declared_end]
    rows = _source_activity_for_event(facts, str(event.get("event_id")))
    record: dict[str, Any] = {
        "declared_event_frames": declared,
        "activity_readback_present": bool(rows),
    }
    if not rows:
        record.update({"frames": declared, "window_source": "declared_event_frames"})
        return record
    sample_rate = float(facts["time"]["sample_rate_hz"])
    frame_rate = float(facts["time"]["frame_rate_hz"])
    active = [
        frame
        for frame in range(frame_count)
        if any(
            int(row.get("start_sample", 0))
            <= int(round(frame * sample_rate / frame_rate))
            < int(row.get("end_sample_exclusive", 0))
            for row in rows
        )
    ]
    if not active:
        record.update({
            "frames": declared,
            "window_source": "declared_event_frames",
            "audible_frame_count": 0,
            "detail": "the activity readback covers no video frame",
        })
        return record
    record.update({
        "frames": [active[0], active[-1] + 1],
        "window_source": "source_activity_readback",
        "audible_frame_count": len(active),
        "audible_span_frame_count": active[-1] - active[0] + 1,
        "contiguous": active[-1] - active[0] + 1 == len(active),
        "internal_pause_frame_count": active[-1] - active[0] + 1 - len(active),
    })
    return record


def distance_trend_during_window(
    facts: Mapping[str, Any],
    actor_id: str,
    window: Sequence[int],
    *,
    min_net_change_m: float | None = None,
    reversal_tolerance_m: float | None = None,
    reversal_fraction: float | None = None,
) -> dict[str, Any]:
    """Decide nearer or farther over a window and prove the trend holds.

    The published answer is a direction, so the difference between the first
    and last frame cannot carry it on its own: a path that approaches and
    then recedes has the same endpoint difference as one that only recedes.
    The trend qualifies when the net change clears ``min_net_change_m`` and
    the largest excursion against the net direction stays inside both an
    absolute tolerance and a fraction of that net change. Both readings are
    reported, so a caller can see the endpoint figure and the excursion that
    actually decided the verdict.
    """

    start, end = int(window[0]), int(window[1])
    minimum = (
        _policy_number(
            facts,
            ("qa15_min_net_change_m", "distance_net_change_min_m"),
            default=DISTANCE_TREND_DEFAULTS["min_net_change_m"],
            name="distance net change margin",
        )
        if min_net_change_m is None
        else float(min_net_change_m)
    )
    tolerance = (
        _policy_number(
            facts,
            ("qa15_reversal_tolerance_m", "distance_reversal_tolerance_m"),
            default=DISTANCE_TREND_DEFAULTS["reversal_tolerance_m"],
            name="distance reversal tolerance",
        )
        if reversal_tolerance_m is None
        else float(reversal_tolerance_m)
    )
    fraction = (
        _policy_number(
            facts,
            ("qa15_reversal_fraction", "distance_reversal_fraction"),
            default=DISTANCE_TREND_DEFAULTS["reversal_fraction"],
            name="distance reversal fraction",
            maximum=1.0,
        )
        if reversal_fraction is None
        else float(reversal_fraction)
    )
    criteria = {
        "min_net_change_m": minimum,
        "reversal_tolerance_m": tolerance,
        "reversal_fraction": fraction,
        "definition": (
            "net change clears the margin and no excursion against the net "
            "direction exceeds the absolute tolerance or the allowed "
            "fraction of that net change"
        ),
    }
    record: dict[str, Any] = {
        "window_frames": [start, end],
        "criteria": criteria,
        "verdict": None,
        "reason": None,
    }
    if end <= start + 1:
        record.update({
            "reason": "distance_window_too_short",
            "detail": "a distance trend needs at least two readback frames",
        })
        return record
    try:
        series = [_distance_at(facts, actor_id, frame) for frame in range(start, end)]
    except _Deferred as error:
        record.update({"reason": error.code, "detail": error.detail})
        return record
    net = series[-1] - series[0]
    direction = "nearer" if net < 0.0 else "farther"
    extreme = series[0]
    counter = 0.0
    for value in series:
        if net < 0.0:
            extreme = min(extreme, value)
            counter = max(counter, value - extreme)
        else:
            extreme = max(extreme, value)
            counter = max(counter, extreme - value)
    allowed = min(tolerance, fraction * abs(net)) if abs(net) > 0.0 else tolerance
    record.update({
        "distance_series_m": series,
        "distance_start_m": series[0],
        "distance_end_m": series[-1],
        "endpoint_delta_m": net,
        "net_direction": direction,
        "distance_span_m": max(series) - min(series),
        "total_variation_m": sum(abs(b - a) for a, b in zip(series, series[1:])),
        "max_counter_trend_m": counter,
        "allowed_counter_trend_m": allowed,
        "monotone_within_tolerance": counter <= allowed,
        "endpoint_delta_is_not_sufficient": True,
    })
    if abs(net) < minimum:
        record.update({
            "reason": "distance_net_change_below_margin",
            "detail": (
                "the distance changes by less than the configured margin over "
                "the window"
            ),
        })
        return record
    if counter > allowed:
        record.update({
            "reason": "distance_trend_reverses",
            "detail": (
                "the path moves back against its net direction by more than "
                "the configured reversal allowance"
            ),
        })
        return record
    record["verdict"] = direction
    return record


def _ordinary_observation_questions(facts):
    return ((facts.get("sampling") or {}).get("acceptance_policy") or {}).get(
        "question_mode") == "ordinary_observation"


def _generate_qa16_timepoint(facts, seed):
    """Compare distance at an explicit whole-second point, using native positions."""
    for event, query_frame, silence in _after_event_candidates(facts, qa_id="QA-16"):
        seconds = query_frame / float(facts["time"]["frame_rate_hz"])
        if abs(seconds - round(seconds)) > 1e-8:
            continue
        try:
            anchor_frame = _event_frame(event, "end_frame")
            anchor_distance = _distance_at(facts, event["actor_id"], anchor_frame)
            query_distance = _distance_at(facts, event["actor_id"], query_frame)
            _silent_after(facts, event, query_frame)
        except _Deferred:
            continue
        margin = _policy_number(facts, ("qa16_distance_margin_m",), default=0.2,
                                name="post-sound distance margin")
        delta = query_distance - anchor_distance
        if abs(delta) < margin:
            continue
        trend = "nearer" if delta < 0 else "farther"
        anchor_en, anchor_zh = _event_anchor(facts, event)
        return _question_item(
            qa_id="QA-16", facts=facts, seed=seed,
            question_en=f"At {int(round(seconds))} seconds, compared with its position at the end of {anchor_en}, was the source nearer or farther from the listener?",
            question_zh=f"与{anchor_zh}结束时的声源位置相比，在第{int(round(seconds))}秒，声源离听者更近还是更远？",
            open_answer_type="closed_set", open_truth=trend, truth_label=trend,
            options=[_option("nearer", "nearer"), _option("farther", "farther")],
            evidence={**_event_evidence(event), "query_frame":query_frame,
                      "query_time_s":float(round(seconds)), "reference_frame":anchor_frame,
                      "reference_distance_m":anchor_distance,"query_distance_m":query_distance,
                      "distance_delta_m":delta,"distance_margin_m":margin,
                      "query_scope":"explicit_integer_timepoint",
                      "silence_evidence":silence},
            slug=f"{event['event_id']}_post_distance_frame_{query_frame}")
    _defer("no_valid_post_sound_timepoint",
           "no silent whole-second query has a measured distance change above the margin")


def _noticeable_motion_policy(facts):
    policy = (facts.get("sampling") or {}).get("acceptance_policy") or {}
    motion = policy.get("motion")
    if not motion:
        return None
    if motion.get("mode") != "noticeable_motion":
        raise UnifiedQAError("unknown configured motion acceptance mode")
    result = dict(motion)
    result.setdefault("speed_threshold_mps", 0.05)
    for key in ("min_moving_duration_s", "min_travel_m", "max_still_travel_m", "speed_threshold_mps"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise UnifiedQAError(f"invalid motion acceptance parameter: {key}")
    if result["max_still_travel_m"] >= result["min_travel_m"]:
        raise UnifiedQAError("moving travel must exceed the still allowance")
    return result


def _noticeable_motion_window(facts, actor_id, start, end, policy):
    actor = _actor(facts, actor_id)
    positions = actor.get("root_positions_m")
    if not _is_sequence(positions) or end > len(positions) or end <= start + 1:
        return {"moving": None, "reason": "missing_motion_position_readback"}
    points = positions[start:end]
    if any(not _is_sequence(p) or len(p) != 3 or
           any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
           for p in points):
        return {"moving": None, "reason": "invalid_motion_position_readback"}
    rate = float(facts["time"]["frame_rate_hz"])
    distances = [math.dist(a, b) for a, b in zip(points, points[1:])]
    travel = sum(distances)
    moving_s = sum(d * rate > policy["speed_threshold_mps"] for d in distances) / rate
    value = (True if travel >= policy["min_travel_m"] and moving_s >= policy["min_moving_duration_s"]
             else False if travel <= policy["max_still_travel_m"] else None)
    return {"moving": value, "reason": None if value is not None else "motion_between_noticeability_thresholds",
            "measurement": {"window_frames": [start, end], "travel_m": travel,
                            "moving_duration_s": moving_s, "position_source": "root_positions_m",
                            "criteria": dict(policy)}}


def _tag_question_tolerance(item, facts, qa_id):
    acceptance = (facts.get("sampling") or {}).get("acceptance_policy") or {}
    active = (_ordinary_observation_questions(facts) or
              (qa_id == "QA-06" and _noticeable_motion_policy(facts) is not None) or
              (qa_id in {"QA-07", "QA-09"} and facts.get("visibility_interpretation")))
    if not active:
        return item
    policy_id = str(acceptance.get("policy_id") or "configured_question_tolerance")
    item["question_id"] += "__policy_" + policy_id
    item["acceptance_policy"] = copy.deepcopy(acceptance)
    item["evidence"]["acceptance_policy"] = copy.deepcopy(acceptance)
    if facts.get("visibility_interpretation"):
        item["evidence"]["visibility_interpretation"] = copy.deepcopy(facts["visibility_interpretation"])
    item["truth"]["source"] = "native_measurements_with_configured_question_tolerance"
    item["truth"]["evidence"] = copy.deepcopy(item["evidence"])
    item["claim_boundary"] = "Research question under the stated tolerance, not strict pixel visibility or original V1 admission."
    if _ordinary_observation_questions(facts):
        item["question_mode"] = "ordinary_observation"
        item["cross_modal_necessity_claim"] = False
        item["claim_boundary"] = "Ordinary observation question from native evidence; cross-modal necessity and original paired-research admission are not claimed."
    return item


def motion_state_during_audible_window(
    facts: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Whether the emitter held one motion state across its sounding span."""

    audible = audible_frame_window(facts, event)
    record: dict[str, Any] = {"audible_window": audible, "moving": None, "reason": None}
    policy = _noticeable_motion_policy(facts)
    if policy is not None:
        record.update(_noticeable_motion_window(
            facts, str(event["actor_id"]), int(audible["frames"][0]), int(audible["frames"][1]), policy))
        return record
    try:
        record["moving"] = _stable_motion_window(
            facts,
            str(event["actor_id"]),
            int(audible["frames"][0]),
            int(audible["frames"][1]),
        )
    except _Deferred as error:
        record.update({"reason": error.code, "detail": error.detail})
    return record


def _defer_with_reasons(
    code: str,
    detail: str,
    reasons: Sequence[Mapping[str, Any]],
    **extra: Any,
) -> None:
    """Defer with one primary code while keeping every candidate reason."""

    _defer(
        code,
        detail,
        candidate_reasons=[dict(reason) for reason in reasons],
        candidate_reason_codes=sorted(
            {str(reason.get("code")) for reason in reasons if reason.get("code")}
        ),
        **extra,
    )


def _option(value: Any, label: str | None = None) -> dict[str, str]:
    text = str(label if label is not None else value)
    return {"value": str(value), "label_en": text, "label_zh": text}


def _choice_aliases(options: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    builtin = {
        "yes": ["yes", "是", "有", "true"],
        "no": ["no", "否", "没有", "无", "false"],
        "moving": ["moving", "walking", "在移动", "在走动", "动"],
        "still": ["still", "static", "staying still", "静止", "不动", "没动"],
        "left": ["left", "左", "左侧", "左边"],
        "right": ["right", "右", "右侧", "右边"],
        "nearer": ["nearer", "closer", "更近", "靠近", "近"],
        "farther": ["farther", "further", "更远", "远离", "远"],
        "multiple": ["multiple", "多人", "都在发声", "多人同时"],
        "none": ["none", "无人", "都没有", "都没发声"],
        "none_of_visible": ["none", "none of them", "都不是", "画外", "来自画面外"],
    }
    for option in options:
        value = str(option["value"])
        values = (
            [value] if bool(option.get("allow_value", True)) else []
        )
        values.extend(
            [str(option.get("label_en", "")), str(option.get("label_zh", ""))]
        )
        values.extend(builtin.get(value, []))
        aliases[value] = list(dict.fromkeys(item for item in values if item))
    return aliases


def _question_item(
    *,
    qa_id: str,
    facts: Mapping[str, Any],
    seed: str,
    question_en: str,
    question_zh: str,
    open_answer_type: str,
    open_truth: Any,
    truth_label: str,
    evidence: Mapping[str, Any],
    options: Sequence[Mapping[str, Any]] | None = None,
    mcq_truth: str | None = None,
    mcq_question_en: str | None = None,
    mcq_question_zh: str | None = None,
    open_extra: Mapping[str, Any] | None = None,
    mcq_optional: bool = False,
    mcq_deferred_reason: Mapping[str, Any] | None = None,
    open_deferred_reason: Mapping[str, Any] | None = None,
    slug: str,
) -> dict[str, Any]:
    canonical = _canonical_qa_id(qa_id)
    if open_answer_type == "angle_deg" or (
        open_answer_type in {"time_s", "time_range_s"} and _time_display_precision(facts) == 0
    ):
        evidence = copy.deepcopy(dict(evidence))
        evidence["answer_full_precision"] = copy.deepcopy(open_truth)
        if open_answer_type == "angle_deg":
            open_truth = (int(round(float(open_truth))) + 180) % 360 - 180
            truth_label = f"{open_truth}°"
            if "integer" not in question_en:
                question_en += " Answer in whole degrees."
                question_zh += " 请回答整数角度。"
        elif open_answer_type == "time_s":
            open_truth = int(round(float(open_truth)))
            truth_label = f"{open_truth} s"
            question_en += " Answer in whole seconds."
            question_zh += " 请回答整数秒。"
        else:
            open_truth = [int(round(float(value))) for value in open_truth]
            truth_label = f"[{open_truth[0]}, {open_truth[1]}) s"
            question_en += " Answer in whole seconds."
            question_zh += " 请回答整数秒。"
    question_id = _question_id_for(
        canonical,
        facts,
        slug,
        evidence,
    )
    option_values: list[dict[str, str]] = []
    for option in options or []:
        value = str(option.get("value"))
        if value in {item["value"] for item in option_values}:
            continue
        option_values.append(
            {
                "value": value,
                "label_en": str(option.get("label_en", value)),
                "label_zh": str(option.get("label_zh", option.get("label_en", value))),
                "allow_value": bool(option.get("allow_value", True)),
            }
        )
    mcq_deferred: dict[str, Any] | None = None
    if mcq_deferred_reason is not None:
        mcq_deferred = copy.deepcopy(dict(mcq_deferred_reason))
        mcq_deferred["status"] = "deferred"
    if option_values and len(option_values) < 2:
        if not mcq_optional and mcq_deferred is None:
            _defer(
                "mcq_option_domain_too_small",
                f"{canonical} has fewer than two MCQ options",
            )
        if mcq_deferred is None:
            mcq_deferred = {
                "status": "deferred",
                "code": "mcq_option_domain_too_small",
                "detail": f"{canonical} has fewer than two MCQ options",
            }
    if not option_values and mcq_optional and mcq_deferred is None:
        mcq_deferred = {
            "status": "deferred",
            "code": "mcq_options_missing",
            "detail": f"{canonical} has no MCQ option domain",
        }
    selected_mcq_truth = str(mcq_truth if mcq_truth is not None else open_truth)
    if options and not mcq_deferred and selected_mcq_truth not in {
        item["value"] for item in option_values
    }:
        _defer("truth_not_in_option_domain", f"{canonical} truth is absent from its options")
    if options and not mcq_deferred:
        order = list(option_values)
        random.Random(f"{seed}\0{question_id}").shuffle(order)
        correct_index = [option["value"] for option in order].index(selected_mcq_truth)
    else:
        order = []
        correct_index = None
    classes = (
        _choice_aliases(order or option_values)
        if (order or option_values)
        else None
    )
    open_form: dict[str, Any] = {
        "question_en": question_en,
        "question_zh": question_zh,
        "answer_type": open_answer_type,
        "truth": copy.deepcopy(open_truth),
    }
    if classes is not None:
        open_form["classes"] = classes
    if open_extra:
        open_form.update(copy.deepcopy(dict(open_extra)))
    if open_answer_type == "transcript_wer" and "normalization" not in open_form:
        open_form["normalization"] = {
            "unicode_form": "NFKC",
            "casefold": True,
            "punctuation": "space",
        }
    if open_answer_type == "transcript_wer":
        open_form.setdefault("reject_multiple_statements", True)
    forms: dict[str, Any] = {} if open_deferred_reason else {"open": open_form}
    mcq_text_en = mcq_question_en if mcq_question_en is not None else question_en
    mcq_text_zh = mcq_question_zh if mcq_question_zh is not None else question_zh
    if order:
        forms["mcq"] = {
            "question_en": mcq_text_en,
            "question_zh": mcq_text_zh,
            "answer_type": "choice",
            "options": order,
            "gold": {"correct_index": correct_index, "value": selected_mcq_truth},
        }
    elif not mcq_optional and mcq_deferred is None:
        _defer("mcq_options_missing", f"{canonical} cannot construct an MCQ option set")
    form_status = {
        "open": ({**copy.deepcopy(dict(open_deferred_reason)), "status": "deferred"}
                 if open_deferred_reason else {"status": "pass"}),
        "mcq": (
            mcq_deferred
            if mcq_deferred is not None
            else {"status": "pass"}
        ),
    }
    model_input: dict[str, Any] = {} if open_deferred_reason else {
        "open": {"question_en": question_en, "question_zh": question_zh}}
    if order:
        model_input["mcq"] = {
            "question_en": mcq_text_en,
            "question_zh": mcq_text_zh,
            "options": [
                {
                    "option": chr(ord("A") + index),
                    "label_en": option["label_en"],
                    "label_zh": option["label_zh"],
                }
                for index, option in enumerate(order)
            ],
        }
    return {
        "schema": UNIFIED_ITEM_SCHEMA,
        "status": "pass",
        "research_only": True,
        "qa_id": canonical,
        "question_id": question_id,
        "episode_id": facts["episode_id"],
        "question": {"en": mcq_text_en if open_deferred_reason else question_en,
                     "zh": mcq_text_zh if open_deferred_reason else question_zh},
        "model_input": model_input,
        "forms": forms,
        "form_status": form_status,
        "truth": {
            "answer_type": open_answer_type,
            "value": copy.deepcopy(open_truth),
            "label": truth_label,
            "mcq_value": selected_mcq_truth,
            "source": "native_engine_readbacks",
            "evidence": copy.deepcopy(dict(evidence)),
        },
        "evidence": copy.deepcopy(dict(evidence)),
        "certification": {
            "status": "not_run",
            "reason": "research question generation does not certify modality necessity",
        },
        "claim_boundary": (
            "Question and answer are derived from supplied native facts for "
            "research use; this is not model evaluation or dataset admission."
        ),
    }


def _require_actor_count(facts: Mapping[str, Any], minimum: int) -> None:
    actors = facts.get("actors")
    count = len(actors) if isinstance(actors, Mapping) else 0
    if count < minimum:
        _defer(
            "insufficient_entities",
            f"requires at least {minimum} actors, observed {count}",
            observed=count,
        )


def _require_stereo(facts: Mapping[str, Any]) -> None:
    audio = facts.get("audio")
    if not isinstance(audio, Mapping) or audio.get("status") != "pass":
        _defer(
            "stereo_audio_not_validated",
            "a verified two-channel final audio readback is required",
            audio_status=audio.get("status") if isinstance(audio, Mapping) else None,
        )


def _require_visibility(facts: Mapping[str, Any], actor_id: str, frame: int) -> Mapping[str, Any]:
    if actor_id not in facts.get("visibility", {}):
        _defer(
            "missing_pixel_visibility",
            f"pixel visibility truth is unavailable for {actor_id!r}",
            actor_id=actor_id,
        )
    return _state(facts, actor_id, frame)


def _event_evidence(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "actor_id": event.get("actor_id"),
        "sound_asset_id": event.get("sound_asset_id"),
        "start_s": event.get("start_s"),
        "end_s": event.get("end_s"),
        "start_frame": event.get("start_frame"),
        "end_frame": event.get("end_frame"),
    }


def _event_segmentation_proven(
    facts: Mapping[str, Any],
    event: Mapping[str, Any],
) -> tuple[bool, Any]:
    """Return whether a non-speech event count was audibly/externally checked."""

    values: list[Any] = [event.get("event_segmentation")]
    qc = event.get("source_qc")
    if isinstance(qc, Mapping) and any(
        key in qc for key in ("event_count", "burst_count", "segmentation_status")
    ):
        values.append(qc)
    audio = facts.get("audio")
    if isinstance(audio, Mapping):
        values.append(audio.get("event_segmentation"))
    sampling = facts.get("sampling")
    if isinstance(sampling, Mapping):
        values.append(sampling.get("event_segmentation"))
    for value in values:
        if isinstance(value, str) and value.casefold() in {
            "pass",
            "reviewed",
            "audited",
            "verified",
        }:
            return True, value
        if isinstance(value, Mapping):
            status = value.get("status", value.get("verdict"))
            if isinstance(status, str) and status.casefold() in {
                "pass",
                "reviewed",
                "audited",
                "verified",
            }:
                return True, value
            count = value.get("event_count", value.get("burst_count"))
            if count == 1 and status in {None, "pass", "reviewed"}:
                return True, value
    return False, None


def _sector_of(angle: float) -> str:
    """Map azimuth to equal-width, half-open sectors."""

    value = (float(angle) + 180.0) % 360.0 - 180.0
    if -45.0 <= value < 45.0:
        return "front"
    if 45.0 <= value < 135.0:
        return "right"
    if -135.0 <= value < -45.0:
        return "left"
    return "back"


def _sector_options() -> list[dict[str, str]]:
    return [
        {
            "value": "front",
            "label_en": "front [-45°, 45°)",
            "label_zh": "前方 [-45°, 45°)",
        },
        {
            "value": "right",
            "label_en": "right [45°, 135°)",
            "label_zh": "右方 [45°, 135°)",
        },
        {
            "value": "back",
            "label_en": "back [135°, 180°) ∪ [-180°, -135°)",
            "label_zh": "后方 [135°, 180°) ∪ [-180°, -135°)",
        },
        {
            "value": "left",
            "label_en": "left [-135°, -45°)",
            "label_zh": "左方 [-135°, -45°)",
        },
    ]


_STATE_LABELS = {
    "visible_clear": "clearly visible",
    "visible_occluded": "partially occluded",
    "fully_occluded": "fully occluded",
    "out_of_view": "out of view",
}


def _state_label(state: Any) -> str:
    value = str(state)
    label = _STATE_LABELS.get(value)
    if label is None:
        _defer(
            "missing_visibility_display_label",
            "visibility state has no human-readable display label",
            state=value,
        )
    return label


def _state_options() -> list[dict[str, str]]:
    return [
        _option(value, label)
        for value, label in _STATE_LABELS.items()
    ]


_CAPABILITY_SOUND_CLASSES = frozenset({
    "any_audioset_class_playback",
    "audio_playback",
    "sound_playback",
})

SOUND_CLASS_ANSWER_DOMAIN_EXCLUSIONS = frozenset({"any_audioset_class_playback"})

_SOUND_CLASS_LABELS = {
    "speech": ("speech", "语音"),
    "speech_playback": ("speech", "语音"),
    "dog_bark": ("dog bark", "狗叫声"),
    "bark": ("bark", "吠声"),
    "cat_meow": ("cat meow", "猫叫声"),
    "laugh": ("laughter", "笑声"),
    "whistle": ("whistle", "口哨声"),
    "music_playback": ("music", "音乐"),
    "bathtub_filling_washing": ("bathtub filling or washing", "浴缸进水或冲洗声"),
    "sink_filling_washing": ("sink filling or washing", "水槽进水或冲洗声"),
    "blender": ("blender", "搅拌机声"),
    "drip": ("dripping water", "滴水声"),
    "fire": ("fire", "火焰声"),
    "microwave_beep": ("microwave beep", "微波炉提示音"),
    "printer": ("printer", "打印机声"),
    "alarm_bell": ("alarm bell", "警铃声"),
    "toilet_flush": ("toilet flush", "冲马桶声"),
    "phone_ring": ("phone ringing", "电话铃声"),
    "air_conditioning": ("air conditioning", "空调声"),
    "alarm_beep": ("alarm beep", "报警提示音"),
    "alarm_clock": ("alarm clock", "闹钟声"),
    "busy_signal": ("busy signal", "占线音"),
    "cellphone_vibration_alert": ("cellphone vibration alert", "手机振动提示音"),
    "chime": ("chime", "提示音"),
    "clock_tick": ("clock ticking", "时钟滴答声"),
    "crackle": ("crackling", "噼啪声"),
    "ding_dong": ("ding-dong chime", "叮咚声"),
    "doorbell": ("doorbell", "门铃声"),
    "doorbell_chime": ("doorbell chime", "门铃提示音"),
    "fire_alarm": ("fire alarm", "火灾报警声"),
    "gurgling": ("gurgling water", "咕噜水声"),
    "microwave_hum": ("microwave hum", "微波炉嗡鸣声"),
    "ringtone": ("ringtone", "手机铃声"),
    "smoke_alarm": ("smoke alarm", "烟雾报警声"),
    "telephone": ("telephone", "电话声"),
    "telephone_bell_ringing": ("telephone bell ringing", "电话铃响"),
    "telephone_dialing_dtmf": ("telephone dialing tones", "电话拨号音"),
    "water_tap_faucet": ("running tap water", "水龙头流水声"),
}


def _sound_class_phrases(
    sound_class: Any,
    *,
    event: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    value = str(sound_class).strip().casefold()
    if event is not None:
        for key in (
            "sound_class_label",
            "sound_category_label",
            "sound_type_label",
            "display_label",
        ):
            label = event.get(key)
            if isinstance(label, str) and label.strip() and label.strip().casefold() != value:
                return label.strip(), label.strip()
    mapped = _SOUND_CLASS_LABELS.get(value)
    if mapped is not None:
        return mapped
    _defer(
        "missing_sound_class_display_label",
        "sound class has no human-readable display label",
        sound_class=value,
    )


def derive_sound_class_answer_domain(
    configured_values: Sequence[Any] | None = None,
    *,
    observed_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Derive one public sound-class domain from a configured pool.

    When a caller supplies a pool, that pool is the domain boundary. Observed
    events may provide an event-owned display label for a class already in the
    pool, but they cannot add a new answer option. If no pool is supplied,
    explicit observed event classes are the only source of the derived
    domain. Capability-only classes keep their existing exclusion semantics;
    the owner-specific any_audioset_class_playback removal is recorded
    separately.
    """

    event_by_class: dict[str, Mapping[str, Any]] = {}
    observed_values: list[str] = []
    for event in observed_events:
        if not isinstance(event, Mapping):
            continue
        raw = event.get("sound_class")
        if not isinstance(raw, str) or not raw.strip():
            continue
        value = re.sub(r"\s+", "_", raw.strip().casefold())
        observed_values.append(value)
        event_by_class.setdefault(value, event)

    if configured_values is None:
        candidates = observed_values
        boundary = "observed_explicit_events"
        source = (
            "observed_explicit_events_with_display_labels; "
            "no configured pool supplied"
        )
    else:
        candidates = []
        for raw in configured_values:
            if not isinstance(raw, str) or not raw.strip():
                raise UnifiedQAError(
                    "ordinary sound_class_options must contain non-empty strings"
                )
            candidates.append(re.sub(r"\s+", "_", raw.strip().casefold()))
        boundary = "explicit_configured_pool"
        source = (
            "explicit_configured_pool_with_display_labels; observed events "
            "supply label metadata only"
        )

    values: list[str] = []
    excluded: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in candidates:
        if value in seen:
            continue
        seen.add(value)
        if value in SOUND_CLASS_ANSWER_DOMAIN_EXCLUSIONS:
            excluded.append({
                "sound_class": value,
                "reason": "owner_policy_exclusion_any_audioset_class_playback",
            })
            continue
        if value in _CAPABILITY_SOUND_CLASSES:
            excluded.append({
                "sound_class": value,
                "reason": "existing_capability_only_class",
            })
            continue
        try:
            _sound_class_phrases(value, event=event_by_class.get(value))
        except _Deferred:
            excluded.append({
                "sound_class": value,
                "reason": "missing_sound_class_display_label",
            })
            continue
        values.append(value)
    return {
        "values": values,
        "excluded": excluded,
        "boundary": boundary,
        "source": source,
    }


def with_derived_sound_class_answer_domain(
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    """Return facts with the ordinary policy's sound domain derived once."""

    result = copy.deepcopy(dict(facts))
    sampling = result.get("sampling")
    if not isinstance(sampling, MutableMapping):
        return result
    policy = sampling.get("acceptance_policy")
    policy_owner: MutableMapping[str, Any] | None = sampling
    nested = sampling.get("qa_sampling")
    if isinstance(nested, MutableMapping) and isinstance(
        nested.get("acceptance_policy"), Mapping
    ):
        policy = nested.get("acceptance_policy")
        policy_owner = nested
    if not isinstance(policy, Mapping):
        return result
    if policy.get("question_mode") != "ordinary_observation":
        return result

    configured = policy.get("sound_class_options")
    if configured is not None and (
        not isinstance(configured, Sequence)
        or isinstance(configured, (str, bytes))
    ):
        raise UnifiedQAError(
            "ordinary sound_class_options must be a list of registered classes"
        )
    events = result.get("events")
    events = (
        events
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes))
        else ()
    )
    domain = derive_sound_class_answer_domain(
        configured,
        observed_events=events,
    )
    updated_policy = copy.deepcopy(dict(policy))
    updated_policy["sound_class_options"] = list(domain["values"])
    updated_policy["sound_class_options_source"] = domain["source"]
    updated_policy["sound_class_options_excluded"] = copy.deepcopy(
        domain["excluded"]
    )
    if policy_owner is not None:
        policy_owner["acceptance_policy"] = updated_policy
    result["sampling"] = sampling
    result["sound_class_answer_domain"] = domain
    return result


def _event_pair(facts: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    events = sorted(_bound_events(facts), key=lambda event: (float(event["start_s"]), event["event_id"]))
    if len(events) < 2:
        _defer("insufficient_events", "this question requires at least two bound events")
    candidate = facts.get("_p8_candidate", {})
    ids = candidate.get("event_ids") if isinstance(candidate, Mapping) else None
    if ids is not None:
        selected = [event for event in events if event["event_id"] in ids]
        if len(ids) != 2 or len(selected) != 2:
            _defer("event_pair_missing", "selected event pair does not resolve uniquely")
        return selected[0], selected[1]
    return events[0], events[1]

def _event_overlap_intervals(
    facts: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[list[list[float]], str]:
    """Return measured source-activity overlaps for one event pair.

    The event `start_s`/`end_s` fields describe placement on the episode
    clock.  When the native source-activity readback is present, it is the
    answer authority for QA-05; an event with no measured active interval
    therefore contributes no overlap.  Older retained bundles without that
    readback keep the placement interval as an explicitly reported fallback.
    """

    sample_rate = float(facts["time"]["sample_rate_hz"])
    use_activity = _source_activity_present(facts)
    basis = (
        "source_activity_intervals_samples"
        if use_activity
        else "event_program_interval"
    )

    if use_activity and any(
        event.get("source_activity_evidence_status") == "missing"
        for event in (first, second)
    ):
        _defer(
            "missing_source_activity_readback",
            "QA-05 cannot classify an event pair with partial source-activity evidence",
            event_ids=[first.get("event_id"), second.get("event_id")],
        )

    def intervals(event: Mapping[str, Any]) -> list[tuple[int, int]]:
        if use_activity:
            result: list[tuple[int, int]] = []
            for row in _source_activity_for_event(
                facts, str(event.get("event_id"))
            ):
                start = row.get("start_sample")
                end = row.get("end_sample_exclusive")
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, (int, float))
                    or not isinstance(end, (int, float))
                    or not math.isfinite(float(start))
                    or not math.isfinite(float(end))
                    or not float(start).is_integer()
                    or not float(end).is_integer()
                ):
                    continue
                start_i, end_i = int(start), int(end)
                if end_i > start_i:
                    result.append((start_i, end_i))
            return result

        start = float(event["start_s"]) * sample_rate
        end = float(event["end_s"]) * sample_rate
        return (
            [(round(start), round(end))]
            if math.isfinite(start) and math.isfinite(end) and end > start
            else []
        )

    overlaps: list[tuple[int, int]] = []
    for first_start, first_end in intervals(first):
        for second_start, second_end in intervals(second):
            start = max(first_start, second_start)
            end = min(first_end, second_end)
            if end > start:
                overlaps.append((start, end))

    merged: list[list[int]] = []
    for start, end in sorted(overlaps):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return (
        [[start / sample_rate, end / sample_rate] for start, end in merged],
        basis,
    )


def _after_event_candidates(
    facts: Mapping[str, Any],
    *,
    qa_id: str,
) -> Any:
    events = sorted(
        _bound_events(facts),
        key=lambda event: (float(event["end_s"]), event["event_id"]),
    )
    candidate = facts.get("_p8_candidate")
    preferred_id = (
        candidate.get("event_id")
        if isinstance(candidate, Mapping)
        else None
    )
    if preferred_id is not None:
        events = [event for event in events if event.get("event_id") == preferred_id]
    for event in events:
        try:
            pre_silence = _anchor_pre_silence(facts, event)
        except _Deferred:
            continue
        try:
            query_frame, query_source = _query_frame(
                facts, qa_id, event=event, after_event=True
            )
        except _Deferred as error:
            if error.code not in {"no_valid_post_sound_window", "sampling_window_missing"}:
                raise
            continue
        if query_frame <= _event_frame(event, "end_frame"):
            continue
        if query_frame >= int(facts["time"]["frame_count"]):
            continue
        # A derived query can be moved later if the first frame still lies in
        # the measured reverberation tail. An explicitly authored query is
        # checked at exactly that frame.
        candidates = (
            [query_frame]
            if query_source not in {"derived_after_event"}
            else list(range(query_frame, int(facts["time"]["frame_count"])))
        )
        for candidate_frame in candidates:
            try:
                silence = _silent_after(facts, event, candidate_frame)
            except _Deferred:
                continue
            evidence = {
                **silence,
                "query_source": query_source,
                "pre_silence": pre_silence,
            }
            candidate = facts.get("_p8_candidate")
            legal_windows = (
                candidate.get("legal_query_windows")
                if isinstance(candidate, Mapping)
                else None
            )
            legal_authority = (
                candidate.get("legal_window_authority")
                if isinstance(candidate, Mapping)
                else None
            )
            if legal_windows is None and query_source == "derived_uniform_in_legal_window":
                legal_windows = _derived_legal_query_windows(
                    facts, qa_id, event=event
                )
            if legal_windows is not None:
                evidence["legal_query_windows"] = copy.deepcopy(legal_windows)
                evidence["legal_window_authority"] = (
                    legal_authority
                    or (
                        _derived_query_window_authority(qa_id)
                        if query_source == "derived_uniform_in_legal_window"
                        else "caller_declared_sampling_window"
                    )
                )
            yield event, candidate_frame, evidence


def _after_event_candidate(
    facts: Mapping[str, Any],
    *,
    qa_id: str,
) -> tuple[Mapping[str, Any], int, dict[str, Any]]:
    for candidate in _after_event_candidates(facts, qa_id=qa_id):
        return candidate
    _defer(
        "no_valid_post_sound_window",
        "no bound event has a later silent query frame",
    )


def _target_with_event(
    facts: Mapping[str, Any],
    *,
    require_content: bool = False,
    require_visible: bool = False,
) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    for actor_id, actor, appearance in _appearance_candidates(facts):
        events = _event_for_actor(facts, actor_id, require_content=require_content)
        for event in events:
            if require_visible:
                frame = max(
                    0,
                    min(
                        int(facts["time"]["frame_count"]) - 1,
                        _event_frame(event, "start_frame"),
                    ),
                )
                try:
                    if _require_visibility(facts, actor_id, frame).get("state") not in VISIBLE_STATES:
                        continue
                except _Deferred:
                    continue
            return actor_id, actor, event
    _defer(
        "no_unique_appearance_target",
        "no actor has an explicit unique appearance and a matching event",
    )


def _generate_qa_01(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 1)
    targets = _appearance_candidates(facts)
    if not targets:
        _defer("missing_appearance", "QA-01 needs an explicit appearance value")
    actor_id, actor, appearance = targets[0]
    truth = "yes" if _event_for_actor(facts, actor_id) else "no"
    appearance_en, appearance_zh = _appearance_phrases(appearance)
    return _question_item(
        qa_id="QA-01",
        facts=facts,
        seed=seed,
        question_en=f"Did the {appearance_en} make a sound during the clip?",
        question_zh=f"{appearance_zh}在片段中发过声吗？",
        open_answer_type="closed_set",
        open_truth=truth,
        truth_label="yes" if truth == "yes" else "no",
        options=[_option("yes", "yes"), _option("no", "no")],
        evidence={
            "target_actor_id": actor_id,
            "appearance": dict(appearance),
            "appearance_review": _appearance_review_for(facts, actor_id),
            "event_ids": [event["event_id"] for event in _event_for_actor(facts, actor_id)],
        },
        slug=actor_id,
    )


def _generate_qa_02(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    _require_stereo(facts)
    actor_id, actor, event = _target_with_event(
        facts, require_content=True, require_visible=True
    )
    appearance = actor["appearance"]
    anchor_en, anchor_zh = _event_anchor(facts, event)
    frame = max(
        0,
        min(
            int(facts["time"]["frame_count"]) - 1,
            _event_frame(event, "start_frame"),
        ),
    )
    if not isinstance(event.get("transcript"), str) and not event.get("sound_asset_id"):
        _defer("missing_sound_identity", "QA-02 needs a sound id or transcript")
    options = _appearance_options(facts)
    if len(options) < 2:
        _defer("appearance_option_domain_too_small", "QA-02 needs at least two appearance values")
    return _question_item(
        qa_id="QA-02",
        facts=facts,
        seed=seed,
        question_en=(
            f"Which described appearance belongs to the actor of {anchor_en}"
            + (
                f" (the recorded utterance is {event['transcript']!r})?"
                if event.get("transcript")
                else "?"
            )
        ),
        question_zh=(
            f"{anchor_zh}对应的个体具有什么已核验外观？"
            + (
                f"（录音台词为“{event['transcript']}”）"
                if event.get("transcript")
                else ""
            )
        ),
        open_answer_type="closed_set",
        open_truth=appearance["value"],
        truth_label=appearance["label"],
        options=options,
        evidence={
            **_event_evidence(event),
            "target_actor_id": actor_id,
            "appearance": dict(appearance),
            "appearance_review": _appearance_review_for(facts, actor_id),
            "target_frame": frame,
        },
        slug=event["event_id"],
    )


def _generate_qa_03(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    _require_stereo(facts)
    event = _earliest_event(facts)
    actor_id = event["actor_id"]
    actor = _actor(facts, actor_id)
    reviewed = _reviewed_appearances(facts)
    candidates = [
        actor_key
        for actor_key in facts["actors"]
        if actor_key in reviewed and _event_for_actor(facts, actor_key)
    ]
    options = _actor_options(facts, candidates)
    if len(options) < 2:
        _defer("speaker_candidate_domain_too_small", "QA-03 needs at least two speaking actors")
    labels = [option["value"] for option in options]
    if len(labels) != len(set(labels)):
        _defer("speaker_labels_not_unique", "QA-03 actor labels are ambiguous")
    return _question_item(
        qa_id="QA-03",
        facts=facts,
        seed=seed,
        question_en="Which reviewed appearance made the first sound?",
        question_zh="哪个已核验外观的个体最先发声？",
        open_answer_type="closed_set",
        open_truth=actor_id,
        truth_label=str(
            reviewed.get(actor_id, {}).get(
                "label", actor.get("display_label", actor_id)
            )
        ),
        options=options,
        evidence={
            "first_event": _event_evidence(event),
            "candidate_actor_ids": candidates,
            "appearance_reviews": {
                candidate: _appearance_review_for(facts, candidate)
                for candidate in candidates
            },
        },
        slug="first_sound",
    )


def _generate_qa_04(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    events = _bound_events(facts)
    for event in events:
        result = _event_start_side_window(facts, event)
        if result is None:
            continue
        window, side, angle = result
        anchor_en, anchor_zh = _event_anchor(facts, event)
        window_fields = _query_window_fields(facts, window)
        display = _display_time_range(facts, window)
        if display is None:
            _defer("query_interval_too_short_for_display", "the stable onset interval has no public range")
        display_en, display_zh = display
        return _question_item(
            qa_id="QA-04",
            facts=facts,
            seed=seed,
            question_en=f"At the onset of {anchor_en} (query interval {display_en}), was the source on your left or right?",
            question_zh=f"在{anchor_zh}的查询区间{display_zh}开始阶段，声源在听者左侧还是右侧？",
            open_answer_type="closed_set",
            open_truth=side,
            truth_label=side,
            options=[_option("left", "left"), _option("right", "right")],
            evidence={
                **_event_evidence(event),
                "query_frame": window[0],
                **window_fields,
                "azimuth_deg": angle,
            },
            slug=event["event_id"],
        )
    _defer("front_dead_zone", "no sound event has a stable left/right side at onset")

def _generate_qa_05(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    first, second = _event_pair(facts)
    overlap_intervals, overlap_basis = _event_overlap_intervals(
        facts, first, second
    )
    truth = "yes" if overlap_intervals else "no"
    first_anchor_en, first_anchor_zh = _event_anchor(facts, first)
    second_anchor_en, second_anchor_zh = _event_anchor(facts, second)
    return _question_item(
        qa_id="QA-05",
        facts=facts,
        seed=seed,
        question_en=f"Did {first_anchor_en} and {second_anchor_en} overlap?",
        question_zh=f"{first_anchor_zh}和{second_anchor_zh}有重叠吗？",
        open_answer_type="closed_set",
        open_truth=truth,
        truth_label=truth,
        options=[_option("yes", "yes"), _option("no", "no")],
        evidence={
            "event_ids": [first["event_id"], second["event_id"]],
            "overlap_interval_s": (
                overlap_intervals[0] if overlap_intervals else None
            ),
            "overlap_intervals_s": overlap_intervals,
            "overlap_basis": overlap_basis,
        },
        slug=f"{first['event_id']}_{second['event_id']}",
    )


def _generate_qa_06(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    events = _bound_events(facts)
    reasons: list[dict[str, Any]] = []
    for event in events:
        # The question is about the sounding span, so the motion state is read
        # over the measured activity window rather than the declared event
        # bounds. Both windows stay in the evidence.
        state = motion_state_during_audible_window(facts, event)
        if state["moving"] is None:
            reasons.append({
                "event_id": event.get("event_id"),
                "actor_id": event.get("actor_id"),
                "code": state["reason"],
                "detail": state.get("detail"),
                "audible_window": state["audible_window"],
            })
            continue
        moving = bool(state["moving"])
        audible = state["audible_window"]
        anchor_en, anchor_zh = _event_anchor(facts, event)
        return _question_item(
            qa_id="QA-06",
            facts=facts,
            seed=seed,
            question_en=(f"Did the source move noticeably during {anchor_en}?"
                         if _noticeable_motion_policy(facts) is not None else
                         f"Was the source moving while making {anchor_en}?"),
            question_zh=(f"{anchor_zh}期间，声源有没有明显移动？"
                         if _noticeable_motion_policy(facts) is not None else
                         f"{anchor_zh}期间，声源在运动吗？"),
            open_answer_type="closed_set",
            open_truth="moving" if moving else "still",
            truth_label="moving" if moving else "still",
            options=[_option("moving", "moving"), _option("still", "staying still")],
            evidence={
                **_event_evidence(event),
                "moving": moving,
                "motion_window_frames": list(audible["frames"]),
                "motion_window_source": audible["window_source"],
                "audible_window": audible,
                **({"motion_measurement": state["measurement"]} if "measurement" in state else {}),
            },
            slug=event["event_id"],
        )
    if reasons:
        _defer_with_reasons(
            "no_stable_motion_event",
            "no bound event holds one motion state across its sounding span",
            reasons,
        )
    _defer("no_stable_motion_event", "no bound event has a stable motion state")


def _generate_qa_07(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    meta = facts.get("visibility_meta", {})
    resolution = meta.get("resolution_hw") if isinstance(meta, Mapping) else None
    if not _is_sequence(resolution) or len(resolution) != 2:
        _defer("missing_visibility_resolution", "QA-07 needs pixel-truth resolution")
    width = float(resolution[1])
    center = (width - 1.0) / 2.0
    dead_zone = max(1.0, width * 0.02)
    reviewed = _reviewed_appearances(facts)
    preferred = facts.get("_p8_candidate")
    preferred_actor = (
        preferred.get("actor_id")
        if isinstance(preferred, Mapping)
        else None
    )
    preferred_frame = (
        preferred.get("query_frame")
        if isinstance(preferred, Mapping)
        else None
    )
    census = visibility_state_census(facts)
    reasons: list[dict[str, Any]] = []
    for actor_id, frames in facts.get("visibility", {}).items():
        if preferred_actor is not None and actor_id != preferred_actor:
            continue
        if not isinstance(frames, Mapping):
            reasons.append({
                "actor_id": str(actor_id),
                "code": "visibility_rows_missing",
                "detail": "this actor has no per-frame visibility readback",
            })
            continue
        ordered = [frames[index] for index in sorted(frames)]
        transitions: list[tuple[int, float]] = []
        ambiguous: list[int] = []
        for previous, current in zip(ordered, ordered[1:]):
            if int(current.get("frame_index", -1)) != int(previous.get("frame_index", -2)) + 1:
                continue
            if previous.get("state") != "out_of_view" or current.get("state") not in VISIBLE_STATES:
                continue
            centroid = current.get("target_centroid_xy_px")
            if not _is_sequence(centroid) or len(centroid) != 2:
                ambiguous.append(int(current.get("frame_index", -1)))
                continue
            offset = float(centroid[0]) - center
            if abs(offset) <= dead_zone:
                ambiguous.append(int(current["frame_index"]))
                continue
            transitions.append((int(current["frame_index"]), offset))
        if not transitions:
            if ambiguous:
                reasons.append({
                    "actor_id": str(actor_id),
                    "code": "entry_side_ambiguous",
                    "detail": (
                        "the entry frame centroid is missing or sits inside "
                        "the centre dead zone, so no side can be named"
                    ),
                    "entry_frames": ambiguous,
                    "side_dead_zone_px": dead_zone,
                })
            elif not census["states"].get("out_of_view"):
                reasons.append({
                    "actor_id": str(actor_id),
                    "code": "no_out_of_view_state_observed",
                    "detail": (
                        "the episode never records an out-of-view frame, so "
                        "an entry into view cannot exist in it"
                    ),
                    "observed_states": census["states"],
                })
            else:
                reasons.append({
                    "actor_id": str(actor_id),
                    "code": "no_entry_transition",
                    "detail": "this actor never crosses from out of view into view",
                    "observed_states": visibility_state_census(
                        facts, str(actor_id)
                    )["states"],
                })
            continue
        if actor_id not in reviewed:
            # The entry is observed. What is missing is a reviewed appearance
            # to name the target with, which is not the same as the target
            # being invisible.
            reasons.append({
                "actor_id": str(actor_id),
                "code": "appearance_review_missing_for_entry",
                "detail": (
                    "the entry into view is observed but this target has no "
                    "reviewed appearance to name it in the question"
                ),
                "entry_frames": [frame for frame, _offset in transitions],
            })
            continue
        for entry_frame, offset in transitions:
            if preferred_frame is not None and entry_frame != int(preferred_frame):
                continue
            window = _entry_transition_window(
                facts,
                str(actor_id),
                entry_frame,
                center=center,
                dead_zone=dead_zone,
            )
            if window is None:
                reasons.append({
                    "actor_id": str(actor_id),
                    "code": "entry_transition_window_not_stable",
                    "detail": "the entry side is not held across a usable interval",
                    "entry_frame": entry_frame,
                })
                continue
            side = "right" if offset > 0.0 else "left"
            appearance_en, appearance_zh = _appearance_phrases(reviewed[actor_id])
            window_fields = _query_window_fields(
                facts, window, start_rounding="floor"
            )
            display = _display_time_range(
                facts, window, start_rounding="floor"
            )
            if display is None:
                _defer("query_interval_too_short_for_display", "the entry interval has no public range")
            display_en, display_zh = display
            return _question_item(
                qa_id="QA-07",
                facts=facts,
                seed=seed,
                question_en=(
                    f"Did the {appearance_en} {'clearly enter' if facts.get('visibility_interpretation') else 'enter'} from the left or right side "
                    f"of the frame during the transition into view {display_en}?"
                ),
                question_zh=(
                    f"在入画过渡时段{display_zh}内，{appearance_zh}是从左侧还是右侧{'清晰进入' if facts.get('visibility_interpretation') else '进入'}画面的？"
                ),
                open_answer_type="closed_set",
                open_truth=side,
                truth_label=side,
                options=[_option("left", "left"), _option("right", "right")],
                evidence={
                    "target_actor_id": actor_id,
                    "entry_frame": entry_frame,
                    "query_frame": entry_frame,
                    **window_fields,
                    "centroid_xy_px": list(centroid),
                    "side_dead_zone_px": dead_zone,
                },
                slug=f"{actor_id}_entry_{entry_frame}",
            )
    if reasons:
        codes = {str(reason.get("code")) for reason in reasons}
        primary = next(
            (
                code
                for code in (
                    "entry_transition_window_not_stable",
                    "appearance_review_missing_for_entry",
                    "entry_side_ambiguous",
                    "no_out_of_view_state_observed",
                    "visibility_rows_missing",
                )
                if code in codes
            ),
            "no_entry_transition",
        )
        _defer_with_reasons(
            primary,
            "no out_of_view to visible transition yields a nameable, publishable side",
            reasons,
            observed_visibility_states=census["states"],
        )
    _defer("no_entry_transition", "no out_of_view to visible transition with an unambiguous side")

def _generate_qa_08(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    for event in _bound_events(facts):
        window_and_state = _event_start_visibility_window(facts, event)
        if window_and_state is None:
            continue
        window, state = window_and_state
        anchor_en, anchor_zh = _event_anchor(facts, event)
        window_fields = _query_window_fields(facts, window)
        display = _display_time_range(facts, window)
        if display is None:
            _defer("query_interval_too_short_for_display", "the visibility interval has no public range")
        display_en, display_zh = display
        return _question_item(
            qa_id="QA-08",
            facts=facts,
            seed=seed,
            question_en=(
                f"At the beginning of {anchor_en} within {display_en}, what was "
                "the source's visibility state?"
            ),
            question_zh=f"在{anchor_zh}对应的{display_zh}开始阶段，声源处于什么可见状态？",
            open_answer_type="closed_set",
            open_truth=state,
            truth_label=_state_label(state),
            options=_state_options(),
            evidence={
                **_event_evidence(event),
                "query_frame": window[0],
                **window_fields,
                "visibility_state": state,
            },
            slug=event["event_id"],
        )
    _defer("no_event_visibility", "no sound event has a stable pixel visibility state at its onset")

def _generate_qa_09(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    reviewed = _reviewed_appearances(facts)
    preferred = facts.get("_p8_candidate")
    preferred_actor = (
        preferred.get("actor_id")
        if isinstance(preferred, Mapping)
        else None
    )
    preferred_frame = (
        preferred.get("query_frame")
        if isinstance(preferred, Mapping)
        else None
    )
    preferred_occluded = (
        preferred.get("occluded_frame")
        if isinstance(preferred, Mapping)
        else None
    )
    incomplete_negative = False
    census = visibility_state_census(facts)
    reasons: list[dict[str, Any]] = []
    for actor_id, frames in facts.get("visibility", {}).items():
        if preferred_actor is not None and actor_id != preferred_actor:
            continue
        ordered = [frames[index] for index in sorted(frames)] if isinstance(frames, Mapping) else []
        fully = [frame.get("frame_index") for frame in ordered if frame.get("state") == "fully_occluded"]
        if fully and actor_id not in reviewed:
            # Full occlusion is observed; the target simply has no reviewed
            # appearance to name it. That is not evidence of invisibility.
            reasons.append({
                "actor_id": str(actor_id),
                "code": "appearance_review_missing_for_occlusion",
                "detail": (
                    "full occlusion is observed but this target has no "
                    "reviewed appearance to name it in the question"
                ),
                "fully_occluded_frames": fully,
            })
            continue
        if actor_id not in reviewed:
            continue
        if preferred_occluded is not None:
            fully = [
                frame for frame in fully
                if int(frame) == int(preferred_occluded)
            ]
        visible_after = [
            frame.get("frame_index")
            for frame in ordered
            if frame.get("state") in VISIBLE_STATES
            and any(int(previous) < int(frame.get("frame_index", 0)) for previous in fully)
        ]
        if preferred_frame is not None:
            visible_after = [
                frame for frame in visible_after
                if int(frame) == int(preferred_frame)
            ]
        if fully:
            truth = "yes" if visible_after else "no"
            if not visible_after and not _visibility_is_complete(facts, actor_id):
                incomplete_negative = True
                continue
            if not visible_after and not any(
                frame.get("state") in VISIBLE_STATES
                and int(frame.get("frame_index", -1)) < min(int(value) for value in fully)
                for frame in ordered
            ):
                # "Did X reappear?" presupposes X was seen before it was hidden.
                reasons.append({
                    "actor_id": str(actor_id),
                    "code": "target_never_visible_before_occlusion",
                    "detail": (
                        "a negative reappearance answer presupposes the target was "
                        "seen before it was hidden; this actor is never visible "
                        "before its first fully occluded frame"
                    ),
                    "fully_occluded_frames": fully,
                })
                continue
            appearance_en, appearance_zh = _appearance_phrases(reviewed[actor_id])
            return _question_item(
                qa_id="QA-09",
                facts=facts,
                seed=seed,
                question_en=(
                    f"Did the {appearance_en} become visible again after being mostly occluded before the end of the clip?"
                    if facts.get("visibility_interpretation") else
                    f"Did the {appearance_en} reappear after being fully occluded before the end of the clip?"
                ),
                question_zh=(
                    f"整段视频中，{appearance_zh}{'基本被遮挡后又重新出现' if facts.get('visibility_interpretation') else '完全遮挡后又重新出现'}了吗？"
                ),
                open_answer_type="closed_set",
                open_truth=truth,
                truth_label=truth,
                options=[_option("yes", "yes"), _option("no", "no")],
                evidence={
                    "target_actor_id": actor_id,
                    "fully_occluded_frames": fully,
                    "reappeared_frames": visible_after,
                    "query_frame": visible_after[0] if visible_after else None,
                    "observation_window": [0, int(facts["time"]["frame_count"]) - 1],
                },
                slug=(
                    f"{actor_id}_reappearance_"
                    f"{visible_after[0] if visible_after else 'none'}"
                ),
            )
    if incomplete_negative:
        _defer(
            "incomplete_visibility_for_negative",
            "cannot emit a negative reappearance answer without complete visibility coverage",
            candidate_reasons=[dict(reason) for reason in reasons],
            observed_visibility_states=census["states"],
        )
    if reasons:
        codes = {str(reason.get("code")) for reason in reasons}
        only_presupposition = codes == {"target_never_visible_before_occlusion"}
        _defer_with_reasons(
            "target_never_visible_before_occlusion"
            if only_presupposition
            else "appearance_review_missing_for_occlusion",
            "the hidden target was never visible before its occlusion"
            if only_presupposition
            else "full occlusion is observed but no occluded target can be named",
            reasons,
            observed_visibility_states=census["states"],
        )
    if not census["states"].get("fully_occluded"):
        _defer(
            "no_fully_occluded_state_observed",
            "the episode never records a fully occluded frame, so a "
            "reappearance question cannot exist in it",
            observed_visibility_states=census["states"],
            complete_visibility_actors=census["complete_actors"],
        )
    _defer("no_reappearance_transition", "no fully_occluded to visible transition is present")


def _occluder_ids(facts: Mapping[str, Any], actor_id: str, frame: int) -> list[str]:
    value = facts.get("visibility", {}).get(actor_id, {}).get(frame)
    if isinstance(value, Mapping) and _is_sequence(value.get("occluder_instance_ids")):
        return [str(item) for item in value["occluder_instance_ids"]]
    evidence = facts.get("occluder_evidence")
    records = evidence.get("frame_records") if isinstance(evidence, Mapping) else None
    if _is_sequence(records):
        matches = [
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("target_instance_id") == actor_id
            and record.get("frame_index") == frame
        ]
        if len(matches) == 1 and _is_sequence(matches[0].get("occluder_instance_ids")):
            return [str(item) for item in matches[0]["occluder_instance_ids"]]
    return []


def _occluder_label(facts: Mapping[str, Any], occluder_id: str) -> str:
    registry = facts.get("occluder_registry")
    if isinstance(registry, Mapping):
        value = registry.get(occluder_id)
        label: Any = None
        if isinstance(value, Mapping):
            label = value.get("display_label") or value.get("label") or value.get("category")
        elif isinstance(value, str):
            label = value
        label_text = _strip_display_version_suffix(label)
        if label_text:
            # Registry entries such as occluder_17/source3 are identifiers,
            # not labels a participant can use to answer the question.
            lower = label_text.casefold()
            if not re.fullmatch(r"(?:source|actor|occluder|instance)[_.-]?\d*", lower):
                return label_text
    _defer(
        "missing_occluder_display_label",
        "occluder identity has no human-readable display label",
        occluder_instance_id=occluder_id,
    )


def _generate_qa_10(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    reviewed = _reviewed_appearances(facts)
    preferred = facts.get("_p8_candidate")
    preferred_actor = (
        preferred.get("actor_id")
        if isinstance(preferred, Mapping)
        else None
    )
    preferred_frame = (
        preferred.get("query_frame")
        if isinstance(preferred, Mapping)
        else None
    )
    for actor_id, frames in facts.get("visibility", {}).items():
        if actor_id not in reviewed or (
            preferred_actor is not None and actor_id != preferred_actor
        ):
            continue
        if not isinstance(frames, Mapping):
            continue
        for frame, value in frames.items():
            frame = int(frame)
            if preferred_frame is not None and frame != int(preferred_frame):
                continue
            if value.get("state") not in {"visible_occluded", "fully_occluded"}:
                continue
            ids = _occluder_ids(facts, actor_id, frame)
            if len(ids) != 1:
                continue
            window = _occlusion_interval_window(facts, str(actor_id), frame)
            if window is None:
                continue
            observed_ids: list[str] = []
            for other_actor_id, other_frames in facts.get("visibility", {}).items():
                if isinstance(other_frames, Mapping):
                    for other_frame in other_frames:
                        observed_ids.extend(
                            _occluder_ids(
                                facts, str(other_actor_id), int(other_frame)
                            )
                        )
            observed_ids = list(dict.fromkeys(observed_ids))
            registry = facts.get("occluder_registry")
            if not isinstance(registry, Mapping):
                _defer(
                    "occluder_registry_missing",
                    "QA-10 requires a registry for the occluder instance IDs",
                )
            if any(item not in registry for item in observed_ids):
                _defer(
                    "occluder_registry_incomplete",
                    "one or more pixel occluder IDs are not registered",
                    occluder_instance_ids=observed_ids,
                )
            candidate_ids = [item for item in observed_ids if item != actor_id]
            candidate_ids.extend(
                item
                for item in registry
                if item != actor_id and item not in candidate_ids
            )
            if len(candidate_ids) < 2:
                candidate_ids.extend(
                    item
                    for item in facts.get("actors", {})
                    if item != actor_id
                    and item in registry
                    and item not in candidate_ids
                )
            for occluder_id in observed_ids:
                if occluder_id in facts.get("actors", {}) and occluder_id not in reviewed:
                    _defer(
                        "occluder_appearance_review_missing",
                        f"occluder actor {occluder_id!r} has no reviewed appearance label",
                    )
            options = [
                {
                    **_option(item, _occluder_label(facts, item)),
                    "allow_value": False,
                }
                for item in candidate_ids
            ]
            target_family = _appearance_family(reviewed[actor_id])
            clash = {}
            for item in candidate_ids:
                entry = registry.get(item)
                if not isinstance(entry, Mapping):
                    continue
                family = _appearance_family(entry.get("appearance_value"))
                if family and family == target_family:
                    clash[item] = entry.get("appearance_value")
            if clash:
                _defer(
                    "occluder_shares_the_target_colour_family",
                    "an answer option is the same colour as the occluded actor in "
                    "the rendered frame, so the question cannot be answered by looking",
                    target_appearance=reviewed[actor_id].get("value"),
                    clashing_occluders=clash,
                )
            option_families: dict[str, list[str]] = {}
            for item in candidate_ids:
                entry = registry.get(item)
                if not isinstance(entry, Mapping):
                    continue
                family = _appearance_family(entry.get("appearance_value"))
                if family:
                    option_families.setdefault(family, []).append(item)
            shared_options = {
                family: members for family, members in option_families.items() if len(members) > 1
            }
            if shared_options:
                _defer(
                    "occluder_options_share_a_colour_family",
                    "two answer options are the same colour in the rendered frame",
                    colour_families=shared_options,
                )
            appearance_en, appearance_zh = _appearance_phrases(reviewed[actor_id])
            window_fields = _query_window_fields(facts, window)
            display = _display_time_range(facts, window)
            if display is None:
                _defer("query_interval_too_short_for_display", "the occlusion interval has no public range")
            display_en, display_zh = display
            return _question_item(
                qa_id="QA-10",
                facts=facts,
                seed=seed,
                question_en=(
                    f"Which visible object or person occluded the {appearance_en} "
                    f"during the occlusion interval {display_en}?"
                ),
                question_zh=(
                    f"在遮挡时段{display_zh}内，哪个可见物体或人物遮挡了{appearance_zh}？"
                ),
                open_answer_type="closed_set",
                open_truth=ids[0],
                truth_label=_occluder_label(facts, ids[0]),
                options=options,
                evidence={
                    "target_actor_id": actor_id,
                    "frame": frame,
                    "query_frame": frame,
                    **window_fields,
                    "occluder_instance_ids": ids,
                    "option_instance_ids": candidate_ids,
                },
                mcq_optional=len(candidate_ids) < 2,
                slug=f"{actor_id}_occluder_{frame}",
            )
    _defer("missing_occluder_identity", "pixel visibility contains no unique occluder identity over a stable interval")

def _generate_qa_11(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    reviewed = _reviewed_appearances(facts)
    preferred = facts.get("_p8_candidate")
    preferred_actor = (
        preferred.get("actor_id")
        if isinstance(preferred, Mapping)
        else None
    )
    preferred_frame = (
        preferred.get("query_frame")
        if isinstance(preferred, Mapping)
        else None
    )
    preferred_partial = (
        preferred.get("partial_frame")
        if isinstance(preferred, Mapping)
        else None
    )
    incomplete_negative = False
    for actor_id, frames in facts.get("visibility", {}).items():
        if actor_id not in reviewed or (
            preferred_actor is not None and actor_id != preferred_actor
        ):
            continue
        if not isinstance(frames, Mapping):
            continue
        ordered = [frames[index] for index in sorted(frames)]
        transitions = [
            current.get("frame_index")
            for previous, current in zip(ordered, ordered[1:])
            if previous.get("state") == "visible_occluded"
            and current.get("state") == "visible_clear"
            and (
                preferred_frame is None
                or int(current.get("frame_index", -1)) == int(preferred_frame)
            )
            and (
                preferred_partial is None
                or int(previous.get("frame_index", -1)) == int(preferred_partial)
            )
        ]
        partial_frames = [
            frame.get("frame_index")
            for frame in ordered
            if frame.get("state") == "visible_occluded"
        ]
        if partial_frames:
            if not transitions and not _visibility_is_complete(facts, actor_id):
                incomplete_negative = True
                continue
            truth = "yes" if transitions else "no"
            appearance_en, appearance_zh = _appearance_phrases(reviewed[actor_id])
            return _question_item(
                qa_id="QA-11",
                facts=facts,
                seed=seed,
                question_en=(
                    f"Did the {appearance_en} become "
                    "clearly visible after partial occlusion before the end of the clip?"
                ),
                question_zh=(
                    f"整段视频中，{appearance_zh}是否曾从部分遮挡变为清晰可见？"
                ),
                open_answer_type="closed_set",
                open_truth=truth,
                truth_label=truth,
                options=[_option("yes", "yes"), _option("no", "no")],
                evidence={
                    "target_actor_id": actor_id,
                    "partial_occlusion_frames": partial_frames,
                    "transition_frames": transitions,
                    "query_frame": transitions[0] if transitions else None,
                    "observation_window": [0, int(facts["time"]["frame_count"]) - 1],
                },
                slug=(
                    f"{actor_id}_clear_"
                    f"{transitions[0] if transitions else 'none'}"
                ),
            )
    if incomplete_negative:
        _defer(
            "incomplete_visibility_for_negative",
            "cannot emit a negative partial-to-clear answer without complete visibility coverage",
        )
    _defer("no_partial_to_clear_transition", "no adjacent partial-occlusion to clear transition is present")


def _statement_ordinal(facts: Mapping[str, Any], event: Mapping[str, Any]) -> int:
    statements = sorted(
        [row for row in facts.get("events", [])
         if row.get("actor_id") == event.get("actor_id")
         and isinstance(row.get("transcript"), str) and row["transcript"].strip()],
        key=lambda row: (float(row["start_s"]), str(row["event_id"])))
    return next(index + 1 for index, row in enumerate(statements)
                if row["event_id"] == event["event_id"])


def _generate_qa_12(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    _require_stereo(facts)
    actor_id, actor, event = _target_with_event(
        facts, require_content=True, require_visible=True
    )
    appearance_en, appearance_zh = _appearance_phrases(actor["appearance"])
    frame = max(
        0,
        min(
            int(facts["time"]["frame_count"]) - 1,
            _event_frame(event, "start_frame"),
        ),
    )
    candidates = [
        candidate
        for other_id in facts["actors"]
        for candidate in _event_for_actor(facts, other_id, require_content=True)
        if isinstance(candidate.get("transcript"), str)
    ]
    transcripts = list(dict.fromkeys(str(candidate["transcript"]) for candidate in candidates))
    if len(transcripts) < 2:
        _defer("transcript_option_domain_too_small", "QA-12 needs at least two distinct recorded transcripts")
    return _question_item(
        qa_id="QA-12",
        facts=facts,
        seed=seed,
        question_en=f"What did the {appearance_en} say in their spoken statement {_statement_ordinal(facts, event)}?",
        question_zh=f"{appearance_zh}在其第{_statement_ordinal(facts, event)}次说话时说了什么？",
        open_answer_type="transcript_wer",
        open_truth=event["transcript"],
        truth_label=str(event["transcript"]),
        options=[_option(text, text) for text in transcripts],
        open_extra={
            "transcript_attribution_policy": "candidate_match_reported_separately",
            "wer_metric": "word_error_rate",
        },
        evidence={
            "target_actor_id": actor_id,
            "appearance": dict(actor["appearance"]),
            "transcript_attribution": {
                "target_event_id": event.get("event_id"),
                "candidate_transcripts": transcripts,
                "match_required": True,
                "ambiguity_policy": "defer_if_no_unique_candidate",
            },
            "wer": {
                "metric": "word_error_rate",
                "reference": event["transcript"],
            },
            "appearance_review": _appearance_review_for(facts, actor_id),
            "event": _event_evidence(event),
            "statement_id": event.get("statement_id"),
        },
        slug=actor_id,
    )


def _generate_qa_13(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    _require_actor_count(facts, 2)
    saw_window = False
    saw_unobservable = False
    saw_unstable_interval = False
    last_reasons = {}
    for event, query_frame, silence in _after_event_candidates(facts, qa_id="QA-13"):
        saw_window = True
        legal_windows = _post_sound_window(
            facts, event, silence, qa_id="QA-13"
        )
        if not legal_windows:
            saw_unstable_interval = True
            continue
        try:
            query_visibility = _require_visibility(
                facts, event["actor_id"], query_frame
            )
            query_angle = _azimuth(facts, event["actor_id"], query_frame)
        except _Deferred:
            saw_unobservable = True
            continue
        if (
            query_visibility.get("state") not in VISIBLE_STATES
            or _fov_band(query_angle) is None
        ):
            saw_unobservable = True
            continue

        def value_for_frame(frame: int) -> dict[str, Any] | None:
            try:
                _silent_after(facts, event, frame)
                visibility = _require_visibility(facts, event["actor_id"], frame)
                angle = _azimuth(facts, event["actor_id"], frame)
            except _Deferred:
                return None
            band = _fov_band(angle)
            if visibility.get("state") not in VISIBLE_STATES or band is None:
                return None
            return {"angle": float(angle), "band": band}

        def stable(values: Sequence[Any]) -> bool:
            if not values or any(not isinstance(value, Mapping) for value in values):
                return False
            bands = {value.get("band") for value in values}
            if len(bands) != 1:
                return False
            angles = [float(value["angle"]) for value in values]
            reference = float(sorted(angles)[len(angles) // 2])
            return max(abs(angle - reference) for angle in angles) <= 15.0

        window = _stable_frame_window(
            facts,
            legal_windows,
            query_frame,
            value_for_frame,
            stable,
        )
        if window is None:
            saw_unstable_interval = True
            continue
        values = [value_for_frame(frame) for frame in range(window[0], window[1])]
        if any(not isinstance(value, Mapping) for value in values):
            saw_unstable_interval = True
            continue
        angles = [float(value["angle"]) for value in values]
        stable_angle = float(sorted(angles)[len(angles) // 2])
        query_value = value_for_frame(query_frame)
        if not isinstance(query_value, Mapping):
            saw_unobservable = True
            continue
        angle_at_query = float(query_value["angle"])
        band = str(query_value["band"])
        try:
            other_angles = {
                actor_id: _azimuth(facts, actor_id, query_frame)
                for actor_id in facts["actors"]
                if actor_id != event["actor_id"]
            }
        except _Deferred:
            continue
        gaps = [
            abs((angle_at_query - value + 180.0) % 360.0 - 180.0)
            for value in other_angles.values()
        ]
        open_reason = None if gaps and min(gaps) > 60.0 else {
            "code": "open_numeric_candidate_gap_too_small",
            "detail": "Open candidates overlap the existing 30-degree partial-credit tolerance",
            "minimum_gap_deg": min(gaps) if gaps else None,
            "minimum_required_gap_deg": 60.0,
            "calibration": "placeholder",
        }
        other_bands = {
            actor_id: _fov_band(value)
            for actor_id, value in other_angles.items()
        }
        missing = [
            actor_id for actor_id, value in other_bands.items() if value is None
        ]
        equal = [
            actor_id for actor_id, value in other_bands.items() if value == band
        ]
        boundary_distance = min(
            abs(stable_angle - boundary) for boundary in _FOV_BAND_BOUNDARIES_DEG
        )
        mcq_reason = None
        available = [value for value in other_bands.values() if value is not None]
        if not available:
            mcq_reason = {
                "code": "candidate_value_missing",
                "detail": "offscreen competitors have no in-view band and are not counted as different",
                "actor_ids": missing,
            }
        elif len(equal) == len(available):
            mcq_reason = {
                "code": "distractors_equal_gold",
                "detail": "all available competitors occupy the same MCQ band as gold",
                "actor_ids": equal,
            }
        elif boundary_distance < 5.0:
            mcq_reason = {
                "code": "mcq_band_boundary_margin",
                "detail": "target is within the placeholder 5-degree band-boundary margin",
                "distance_to_boundary_deg": boundary_distance,
                "required_margin_deg": 5.0,
                "calibration": "placeholder",
            }
        if open_reason is not None and mcq_reason is not None:
            last_reasons = {"open": open_reason, "mcq": mcq_reason}
            continue
        anchor_en, anchor_zh = _event_anchor(facts, event)
        window_fields = _query_window_fields(facts, window)
        display = _display_time_range(facts, window)
        if display is None:
            _defer("query_interval_too_short_for_display", "the post-sound interval has no public range")
        display_en, display_zh = display
        silence = copy.deepcopy(silence)
        silence.update(window_fields)
        evidence = {
            **_event_evidence(event),
            "post_sound": silence,
            "query_frame": query_frame,
            "query_time_s": query_frame / float(facts["time"]["frame_rate_hz"]),
            **window_fields,
            "query_visibility_state": query_value.get("band"),
            "azimuth_deg": stable_angle,
            "azimuth_at_query_deg": angle_at_query,
            "azimuth_interval_deg": [min(angles), max(angles)],
            "distractor_azimuths_deg": other_angles,
            "target_fov_band": band,
            "distractor_fov_bands": other_bands,
            "fov_half_angle_deg": _FOV_HALF_DEG,
            "fov_band_boundaries_deg": list(_FOV_BAND_BOUNDARIES_DEG),
            "fov_band_calibration": "placeholder",
            "band_boundary_margin_deg": 5.0,
            "target_unobservable_at_query": False,
        }
        return _question_item(
            qa_id="QA-13",
            facts=facts,
            seed=seed,
            question_en=(
                f"After {anchor_en} ended, during the silent interval {display_en}, "
                "what approximate numeric azimuth did the source maintain? "
                "Report degrees: front is 0°, right is positive, range [-180°, 180°)."
            ),
            question_zh=(
                f"{anchor_zh}结束后的静音时段{display_zh}内，声源大致保持在什么数值方位角？"
                "正前方为0°，右侧为正，范围[-180°，180°）。"
            ),
            mcq_question_en=(
                f"After {anchor_en} ended, during the silent interval {display_en}, "
                "which in-view horizontal band contained the source?"
            ),
            mcq_question_zh=(
                f"{anchor_zh}结束后的静音时段{display_zh}内，声源位于哪个视野内水平角带？"
            ),
            open_answer_type="angle_deg",
            open_truth=stable_angle,
            truth_label=f"{stable_angle:.1f}°",
            mcq_truth=band,
            open_extra={
                "convention": "right_positive",
                "convention_description": "azimuth_deg; front=0°, right_positive, range=[-180°,180°)",
                "theta_full_deg": 15.0,
                "theta_half_deg": 30.0,
            },
            open_deferred_reason=open_reason,
            mcq_deferred_reason=mcq_reason,
            options=_fov_band_options(),
            evidence=evidence,
            slug=f"{event['event_id']}_post_direction",
        )
    if saw_unobservable:
        _defer(
            "target_unobservable_at_query",
            "the target is not observable in the declared query view",
            open_and_mcq_deferred=True,
        )
    if saw_unstable_interval:
        _defer(
            "post_sound_query_interval_not_stable",
            "no legal post-sound interval keeps the target answer within the established scoring margin",
        )
    if not saw_window:
        _defer(
            "no_valid_post_sound_window",
            "no bound event has a measured silent query window",
        )
    _defer(
        "post_sound_angle_not_separated",
        "no query candidate supports either answer form",
        form_reasons=last_reasons,
    )

def _generate_qa_14(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    candidates = _appearance_candidates(facts)
    if len(candidates) < 2:
        _defer("distance_candidate_domain_too_small", "QA-14 needs two explicit appearance targets")
    frame_count = int(facts["time"]["frame_count"])
    frame_rate = float(facts["time"]["frame_rate_hz"])
    sampled_frame = _sampling_value(
        facts, "QA-14", "query_frame_by_qa", "query_frames"
    )
    sampled_time = _sampling_value(
        facts, "QA-14", "query_time_s_by_qa", "query_times_s"
    )
    if sampled_frame is not None:
        frame_candidates = [
            _resolve_query_frame_spec(
                facts, "QA-14", sampled_frame, source="sampling_frame"
            )
        ]
        query_source = "sampling_frame"
    elif sampled_time is not None:
        time_s = _resolve_query_time_spec(
            facts, "QA-14", sampled_time, source="sampling_time"
        )
        frame_candidates = [
            _resolve_query_frame_spec(
                facts,
                "QA-14",
                int(round(time_s * frame_rate)),
                source="sampling_time",
            )
        ]
        query_source = "sampling_time"
    else:
        frame_candidates = list(range(frame_count))
        query_source = "first_valid_frame_search"
    declared_windows = _sampling_value(
        facts, "QA-14", "legal_window_by_qa", "legal_windows", "query_windows"
    )
    legal_windows = declared_windows if declared_windows is not None else [[0, frame_count]]
    selected: tuple[
        str, Mapping[str, Any], str, Mapping[str, Any], float, float, int, list[int]
    ] | None = None
    for frame in frame_candidates:
        for first_index, (first_id, first_actor, _first_appearance) in enumerate(candidates):
            for second_id, second_actor, _second_appearance in candidates[first_index + 1 :]:
                def value_for_frame(query: int) -> dict[str, Any] | None:
                    try:
                        first_state = _require_visibility(facts, first_id, query)
                        second_state = _require_visibility(facts, second_id, query)
                        first_distance = _distance_at(facts, first_id, query)
                        second_distance = _distance_at(facts, second_id, query)
                    except _Deferred:
                        return None
                    if (
                        first_state.get("state") not in VISIBLE_STATES
                        or second_state.get("state") not in VISIBLE_STATES
                        or abs(first_distance - second_distance) < 0.5
                    ):
                        return None
                    return {
                        "closer": first_id if first_distance < second_distance else second_id,
                        "first_distance": float(first_distance),
                        "second_distance": float(second_distance),
                    }

                def stable(values: Sequence[Any]) -> bool:
                    if not values or any(not isinstance(value, Mapping) for value in values):
                        return False
                    closer = {value.get("closer") for value in values}
                    return len(closer) == 1 and all(
                        abs(float(value["first_distance"]) - float(value["second_distance"])) >= 0.5
                        for value in values
                    )

                window = _stable_frame_window(
                    facts,
                    legal_windows,
                    frame,
                    value_for_frame,
                    stable,
                )
                if window is None:
                    continue
                value = value_for_frame(frame)
                if not isinstance(value, Mapping):
                    continue
                selected = (
                    first_id,
                    first_actor,
                    second_id,
                    second_actor,
                    float(value["first_distance"]),
                    float(value["second_distance"]),
                    frame,
                    window,
                )
                break
            if selected is not None:
                break
        if selected is not None:
            break
    if selected is None:
        _defer(
            "no_valid_distance_query",
            "no sampled frame has a stable interval with two reviewed visible targets and a 0.5 m distance margin",
        )
    (
        first_id,
        first_actor,
        second_id,
        second_actor,
        first_distance,
        second_distance,
        frame,
        window,
    ) = selected
    values = []
    for query in range(window[0], window[1]):
        try:
            values.append(
                {
                    "first_distance": _distance_at(facts, first_id, query),
                    "second_distance": _distance_at(facts, second_id, query),
                }
            )
        except _Deferred:
            continue
    truth = first_id if first_distance < second_distance else second_id
    options = _actor_options(facts, [first_id, second_id])
    window_fields = _query_window_fields(facts, window)
    display = _display_time_range(facts, window)
    if display is None:
        _defer("query_interval_too_short_for_display", "the distance interval has no public range")
    display_en, display_zh = display
    return _question_item(
        qa_id="QA-14",
        facts=facts,
        seed=seed,
        question_en=(
            f"During {display_en}, which actor was closer to the listener?"
        ),
        question_zh=f"{display_zh}内，哪个个体离听者更近？",
        open_answer_type="closed_set",
        open_truth=truth,
        truth_label=_appearance_phrases(facts["actors"][truth]["appearance"])[0],
        options=options,
        evidence={
            "query_time_s": frame / frame_rate,
            "query_source": query_source,
            "query_frame": frame,
            **window_fields,
            "distances_m": {first_id: first_distance, second_id: second_distance},
            "distance_interval_m": {
                first_id: [min(value["first_distance"] for value in values), max(value["first_distance"] for value in values)]
                if values else [first_distance, first_distance],
                second_id: [min(value["second_distance"] for value in values), max(value["second_distance"] for value in values)]
                if values else [second_distance, second_distance],
            },
            "appearance_reviews": {
                first_id: _appearance_review_for(facts, first_id),
                second_id: _appearance_review_for(facts, second_id),
            },
        },
        slug=f"{first_id}_{second_id}_{frame}",
    )

def _generate_qa_15(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    """Ask whether the source approached or receded while it was sounding.

    The answer is a direction held across the whole sounding span, so the
    check is ``distance_trend_during_window``: a net change past the margin
    plus a bounded excursion against that direction. The first-to-last
    difference is kept as a reading, never as the proof.
    """

    _require_stereo(facts)
    reasons: list[dict[str, Any]] = []
    for event in _bound_events(facts):
        actor_id = str(event["actor_id"])
        audible = audible_frame_window(facts, event)
        window = [int(audible["frames"][0]), int(audible["frames"][1])]
        trend = distance_trend_during_window(facts, actor_id, window)
        if trend["verdict"] is None:
            reasons.append({
                "event_id": event.get("event_id"),
                "actor_id": actor_id,
                "code": trend["reason"],
                "detail": trend.get("detail"),
                "endpoint_delta_m": trend.get("endpoint_delta_m"),
                "max_counter_trend_m": trend.get("max_counter_trend_m"),
                "window_frames": window,
                "window_source": audible["window_source"],
            })
            continue
        # A change in listener distance under a fixed camera means the emitter
        # moved. If the per-frame motion readback denies that, the two
        # readbacks disagree and neither may be published as the answer.
        motion = motion_state_during_audible_window(facts, event)
        if motion["reason"] is None and motion["moving"] is False:
            reasons.append({
                "event_id": event.get("event_id"),
                "actor_id": actor_id,
                "code": "distance_and_motion_readbacks_disagree",
                "detail": (
                    "listener distance changes past the margin while the "
                    "per-frame motion readback reports a still emitter"
                ),
                "endpoint_delta_m": trend.get("endpoint_delta_m"),
                "window_frames": window,
            })
            continue
        truth = str(trend["verdict"])
        anchor_en, anchor_zh = _event_anchor(facts, event)
        return _question_item(
            qa_id="QA-15",
            facts=facts,
            seed=seed,
            question_en=(
                f"During {anchor_en}, did the source move nearer or farther "
                "from the listener?"
            ),
            question_zh=f"{anchor_zh}期间，声源是在靠近还是远离听者？",
            open_answer_type="closed_set",
            open_truth=truth,
            truth_label=truth,
            options=[_option("nearer", "nearer"), _option("farther", "farther")],
            evidence={
                **_event_evidence(event),
                "distance_start_m": trend["distance_start_m"],
                "distance_end_m": trend["distance_end_m"],
                "delta_m": trend["endpoint_delta_m"],
                "distance_trend": trend,
                "audible_window": audible,
                "motion_readback": {
                    "moving": motion.get("moving"),
                    "reason": motion.get("reason"),
                },
            },
            slug=event["event_id"],
        )
    if reasons:
        _defer_with_reasons(
            "no_distance_trend_during_event",
            "no event has a proven approach or recession across its sounding span",
            reasons,
        )
    _defer("no_distance_trend_during_event", "no event has a measurable distance trend")


def _generate_qa_16(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    acceptance = (facts.get("sampling") or {}).get("acceptance_policy") or {}
    if _ordinary_observation_questions(facts) and acceptance.get("post_sound_distance_query") == "integer_timepoint":
        return _generate_qa16_timepoint(facts, seed)
    candidate_seen = False
    unstable_interval = False
    for event, query_frame, silence in _after_event_candidates(
        facts, qa_id="QA-16"
    ):
        candidate_seen = True
        legal_windows = _post_sound_window(
            facts, event, silence, qa_id="QA-16"
        )
        if not legal_windows:
            unstable_interval = True
            continue
        try:
            anchor_frame = _event_frame(event, "end_frame")
            anchor_distance = _distance_at(
                facts, event["actor_id"], anchor_frame
            )
        except _Deferred:
            continue

        def value_for_frame(frame: int) -> dict[str, Any] | None:
            try:
                _silent_after(facts, event, frame)
                query_distance = _distance_at(
                    facts, event["actor_id"], frame
                )
            except _Deferred:
                return None
            delta = float(query_distance) - float(anchor_distance)
            if abs(delta) < 0.2:
                return None
            return {
                "trend": "nearer" if delta < 0.0 else "farther",
                "query_distance": float(query_distance),
                "delta": delta,
            }

        def stable(values: Sequence[Any]) -> bool:
            if not values or any(not isinstance(value, Mapping) for value in values):
                return False
            trends = {value.get("trend") for value in values}
            return len(trends) == 1 and all(
                abs(float(value["delta"])) >= 0.2 for value in values
            )

        window = _stable_frame_window(
            facts,
            legal_windows,
            query_frame,
            value_for_frame,
            stable,
        )
        if window is None:
            unstable_interval = True
            continue
        query_value = value_for_frame(query_frame)
        if not isinstance(query_value, Mapping):
            unstable_interval = True
            continue
        trend = str(query_value["trend"])
        query_distance = float(query_value["query_distance"])
        interval_values = [
            value_for_frame(frame)
            for frame in range(window[0], window[1])
        ]
        distances = [
            float(value["query_distance"])
            for value in interval_values
            if isinstance(value, Mapping)
        ]
        anchor_en, anchor_zh = _event_anchor(facts, event)
        window_fields = _query_window_fields(facts, window)
        display = _display_time_range(facts, window)
        if display is None:
            _defer("query_interval_too_short_for_display", "the post-sound interval has no public range")
        display_en, display_zh = display
        silence = copy.deepcopy(silence)
        silence.update(window_fields)
        return _question_item(
            qa_id="QA-16",
            facts=facts,
            seed=seed,
            question_en=(
                f"Compared with the source position at the end of {anchor_en}, "
                f"was the source nearer or farther throughout the silent "
                f"interval {display_en} afterward?"
            ),
            question_zh=(
                f"与{anchor_zh}结束时的声源位置相比，在其后的静音时段{display_zh}内，"
                "声源整体更近还是更远？"
            ),
            open_answer_type="closed_set",
            open_truth=trend,
            truth_label=trend,
            options=[_option("nearer", "nearer"), _option("farther", "farther")],
            evidence={
                **_event_evidence(event),
                "post_sound": silence,
                "distance_anchor_m": anchor_distance,
                "distance_query_m": query_distance,
                "distance_query_interval_m": [min(distances), max(distances)] if distances else [query_distance, query_distance],
                "delta_m": float(query_value["delta"]),
                "query_time_s": query_frame / float(facts["time"]["frame_rate_hz"]),
                "query_frame": query_frame,
                **window_fields,
            },
            slug=f"{event['event_id']}_post_distance",
        )
    if not candidate_seen:
        _defer(
            "no_valid_post_sound_window",
            "no bound event has a later silent query frame",
        )
    if unstable_interval:
        _defer(
            "post_sound_query_interval_not_stable",
            "no legal post-sound interval keeps the distance answer within the established margin",
        )
    _defer(
        "no_distance_change_after_event",
        "no legal post-sound event has a distance change above the research margin",
    )

def _generate_qa_17(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    unstable_interval = False
    first_valid: tuple[Mapping[str, Any], int, dict[str, Any], list[int], list[bool]] | None = None
    for event, query_frame, silence in _after_event_candidates(
        facts, qa_id="QA-17"
    ):
        legal_windows = _post_sound_window(
            facts, event, silence, qa_id="QA-17"
        )
        if not legal_windows:
            unstable_interval = True
            continue
        anchor_frame = _event_frame(event, "end_frame")

        def value_for_frame(frame: int) -> bool | None:
            try:
                _silent_after(facts, event, frame)
                values = [
                    _motion_at(facts, event["actor_id"], index)
                    for index in range(max(0, anchor_frame), frame + 1)
                ]
            except _Deferred:
                return None
            # The question asks whether any movement occurred in the interval;
            # it does not require every frame to be moving.
            return any(values)

        window = _stable_frame_window(
            facts,
            legal_windows,
            query_frame,
            value_for_frame,
            lambda values: bool(values) and len(set(values)) == 1,
        )
        if window is None:
            unstable_interval = True
            continue
        truth = value_for_frame(query_frame)
        if truth is None:
            unstable_interval = True
            continue
        values = [
            bool(value_for_frame(frame))
            for frame in range(window[0], window[1])
        ]
        if first_valid is None or truth:
            first_valid = (event, query_frame, copy.deepcopy(silence), window, values)
        if truth:
            break
    if first_valid is None:
        _defer(
            "no_valid_post_sound_window",
            "no bound event has a later silent query frame with a stable motion answer",
        )
    event, query_frame, silence, window, values = first_valid
    anchor_en, anchor_zh = _event_anchor(facts, event)
    chosen_anchor_frame = _event_frame(event, "end_frame")

    def chosen_value_for_frame(frame: int) -> bool | None:
        """Re-read the answer for the selected event, not the last one tried."""
        try:
            _silent_after(facts, event, frame)
            readings = [
                _motion_at(facts, event["actor_id"], index)
                for index in range(max(0, chosen_anchor_frame), frame + 1)
            ]
        except _Deferred:
            return None
        return any(readings)

    truth_value = bool(values[0]) if values else None
    # The reader is handed whole seconds, which is a different set of instants
    # from the proven frame window, so the answer is measured again on exactly
    # those instants rather than restated from the finer window.
    publication = verify_published_query_window(
        facts, window, chosen_value_for_frame, truth_value, qa_id="QA-17"
    )
    window_fields = _query_window_fields(facts, window)
    display = _display_time_range(facts, window)
    if display is None:
        _defer("query_interval_too_short_for_display", "the post-sound interval has no public range")
    display_en, display_zh = display
    silence.update(window_fields)
    truth = "yes" if any(values) else "no"
    return _question_item(
        qa_id="QA-17",
        facts=facts,
        seed=seed,
        question_en=(
            f"Within the silent interval {display_en} after {anchor_en}, did the "
            "source move at any point between the event's end and the end of "
            "that interval?"
        ),
        question_zh=(
            f"在{anchor_zh}结束后的静音时段{display_zh}内，直到该时段结束前，"
            "声源是否曾移动过？"
        ),
        open_answer_type="closed_set",
        open_truth=truth,
        truth_label=truth,
        options=[_option("yes", "yes"), _option("no", "no")],
        evidence={
            **_event_evidence(event),
            "post_sound": silence,
            "motion_frames": [event["end_frame"], window[1] - 1],
            "moving_values": values,
            "query_frame": query_frame,
            "query_time_s": query_frame / float(facts["time"]["frame_rate_hz"]),
            "published_window_verification": publication,
            **window_fields,
        },
        slug=f"{event['event_id']}_post_motion",
    )

def _generate_qa_18(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    _require_stereo(facts)
    try:
        reviewed = _reviewed_appearances(facts)
    except _Deferred:
        # A known all-silent query has the closed-set answer "none" without
        # selecting a visual actor. Require appearance review only when an
        # active source must be named.
        reviewed = {}
    query_time, query_source = _query_time(facts, "QA-18")
    candidate = facts.get("_p8_candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    legal_windows = candidate.get("legal_query_windows")
    legal_windows_source = "candidate" if legal_windows is not None else None
    if legal_windows is None:
        legal_windows = _sampling_value(
            facts, "QA-18", "legal_window_by_qa", "legal_windows", "query_windows"
        )
        if legal_windows is not None:
            legal_windows_source = "caller"
    if legal_windows is None:
        legal_windows = _derived_legal_query_windows(facts, "QA-18")
        if legal_windows is not None:
            legal_windows_source = "derived"
    frame = _resolve_query_frame_spec(
        facts,
        "QA-18",
        int(round(query_time * float(facts["time"]["frame_rate_hz"]))),
        source="query_time",
    )
    if not _source_activity_present(facts):
        _defer(
            "missing_source_activity_readback",
            "QA-18 needs source activity readback to prove its interval",
        )
    if _time_display_precision(facts) == 0:
        return _generate_qa18_integer_window(
            facts,
            seed,
            query_time=query_time,
            query_source=query_source,
            candidate=candidate,
            frame=frame,
            reviewed=reviewed,
        )
    wet_tails = (
        facts.get("audio", {}).get("wet_tail_intervals", [])
        if isinstance(facts.get("audio"), Mapping)
        else []
    )
    active = _active_at(
        facts,
        frame,
        require_source_activity=True,
    )
    wet_tail_events = [
        interval.get("event_id")
        for interval in wet_tails
        if isinstance(interval, Mapping)
        and interval.get("event_id") is not None
        and float(interval.get("start_s", 0.0)) <= query_time
        < float(interval.get("end_s", 0.0))
    ]
    if wet_tail_events and not active:
        _defer(
            "query_inside_wet_tail",
            "QA-18 query has no active source and is inside a measured listener-side wet-tail interval",
            query_time_s=query_time,
            event_ids=wet_tail_events,
        )

    def active_signature(query_frame: int) -> tuple[str, ...] | None:
        try:
            return tuple(
                str(event.get("event_id"))
                for event in _active_at(
                    facts, query_frame, require_source_activity=True
                )
            )
        except _Deferred:
            return None

    query_signature = active_signature(frame)
    if query_signature is None:
        _defer(
            "missing_source_activity_readback",
            "QA-18 needs source activity readback to prove its interval",
        )
    if legal_windows is None:
        _defer(
            "query_interval_missing",
            "QA-18 requires a concrete legal query interval",
        )
    window = _stable_frame_window(
        facts,
        legal_windows,
        frame,
        active_signature,
        lambda values: bool(values) and len(set(values)) == 1,
    )
    if window is None:
        _defer(
            "query_interval_not_stable",
            "no legal query interval keeps the active source set unchanged",
        )
    legal_authority = candidate.get("legal_window_authority")
    if legal_authority is None:
        legal_authority = (
            _derived_query_window_authority("QA-18")
            if legal_windows_source == "derived"
            else "caller_declared_sampling_window"
        )
    active_actor_ids = list(
        dict.fromkeys(
            event["actor_id"]
            for event in active
            if event.get("actor_id")
        )
    )
    activity_class = (
        "empty"
        if not active_actor_ids
        else "active"
        if len(active_actor_ids) == 1
        else "multiple"
    )
    if any(actor_id not in reviewed for actor_id in active_actor_ids):
        _defer(
            "speaker_appearance_review_missing",
            "the specified-time speaker has no reviewed appearance label",
        )
    if len(active_actor_ids) == 1:
        truth = active_actor_ids[0]
    elif len(active_actor_ids) > 1:
        truth = "multiple"
    else:
        truth = "none"
    options = _actor_options(facts, list(reviewed))
    options.extend([_option("multiple", "multiple actors"), _option("none", "no actor")])
    if truth not in {option["value"] for option in options}:
        _defer("speaker_at_time_truth_missing", "query truth is absent from its option domain")
    window_fields = _query_window_fields(facts, window)
    display = _display_time_range(facts, window)
    if display is None:
        _defer("query_interval_too_short_for_display", "the active-source interval has no public range")
    display_en, display_zh = display
    legal_windows_copy = copy.deepcopy(legal_windows)
    return _question_item(
        qa_id="QA-18",
        facts=facts,
        seed=seed,
        question_en=f"During {display_en}, who was making a sound?",
        question_zh=f"{display_zh}内，谁在发声？",
        open_answer_type="closed_set",
        open_truth=truth,
        truth_label=(
            "multiple actors"
            if truth == "multiple"
            else "no actor"
            if truth == "none"
            else _appearance_phrases(facts["actors"][truth]["appearance"])[0]
        ),
        options=options,
        evidence={
            "query_time_s": query_time,
            "query_source": query_source,
            "query_frame": frame,
            **window_fields,
            "source_activity_coordinate_space": "episode_sample_clock",
            "source_activity_event_ids": [event["event_id"] for event in active],
            "wet_tail_event_ids": wet_tail_events,
            "wet_tail_boundary_policy": "measured_interval_only_for_empty_branch",
            "legal_query_windows": legal_windows_copy,
            "activity_class": activity_class,
            "legal_window_authority": legal_authority,
            "active_event_ids": [event["event_id"] for event in active],
            "active_actor_ids": active_actor_ids,
            "appearance_reviews": {
                actor_id: _appearance_review_for(facts, actor_id)
                for actor_id in reviewed
            },
        },
        slug=f"frame_{frame}",
    )

def _time_band_count(facts: Mapping[str, Any]) -> int:
    sampling = facts.get("sampling")
    owners: list[Mapping[str, Any]] = []
    if isinstance(sampling, Mapping):
        nested = sampling.get("qa_sampling")
        if isinstance(nested, Mapping):
            owners.append(nested)
        owners.append(sampling)
    policy = facts.get("sampling_policy")
    if isinstance(policy, Mapping):
        owners.append(policy)
    value = None
    for owner in owners:
        for key in ("time_band_count", "time_interval_count", "qa19_time_band_count"):
            if key in owner:
                value = owner[key]
                break
        if value is not None:
            break
    if value is None:
        return 4
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        _defer(
            "time_band_config_invalid",
            "time interval count must be an integer greater than one",
            value=value,
        )
    return int(value)


def _time_bands(facts: Mapping[str, Any]) -> list[tuple[float, float]]:
    duration = float(facts["time"]["duration_seconds"])
    if not math.isfinite(duration) or duration <= 0.0:
        _defer("invalid_duration", "time duration must be positive")
    count = _time_band_count(facts)
    step = duration / float(count)
    if _time_display_precision(facts) == 0:
        boundaries = [round(index * step) for index in range(count)] + [math.ceil(duration)]
        if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
            _defer("integer_time_bands_too_short", "the requested time bands need distinct whole-second boundaries")
        return list(zip(boundaries, boundaries[1:]))
    return [(index * step, (index + 1) * step) for index in range(count)]


def _format_public_seconds(value: float, *, precision: int = 6) -> str:
    precision = max(0, int(precision))
    text = f"{float(value):.{precision}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _time_band_label(
    facts: Mapping[str, Any],
    index: int,
    bands: Sequence[Sequence[float]],
) -> tuple[str, str]:
    if not 0 <= int(index) < len(bands):
        _defer(
            "time_band_invalid",
            "time band index is outside the declared interval domain",
            index=index,
        )
    start, end = bands[int(index)]
    start_text = _format_public_seconds(float(start))
    end_text = _format_public_seconds(float(end))
    return (
        f"[{start_text}, {end_text}) seconds",
        f"第{start_text}至第{end_text}秒的时间段",
    )


def _generate_qa_19(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    actor_id, actor, event = _target_with_event(facts, require_visible=True)
    appearance_en, appearance_zh = _appearance_phrases(actor["appearance"])
    first = _first_event(facts, actor_id)
    time_s = float(first["start_s"])
    bands = _time_bands(facts)
    band_index = next(
        (index for index, (lo, hi) in enumerate(bands) if lo <= time_s < hi),
        len(bands) - 1,
    )
    labels = [_time_band_label(facts, index, bands) for index in range(len(bands))]
    band_en, band_zh = labels[band_index]
    options = [
        {
            "value": f"band_{index}",
            "label_en": labels[index][0],
            "label_zh": labels[index][1],
            "allow_value": False,
        }
        for index in range(len(bands))
    ]
    truth_range = [float(bands[band_index][0]), float(bands[band_index][1])]
    domain_en = ", ".join(label[0] for label in labels)
    domain_zh = "、".join(label[1] for label in labels)
    return _question_item(
        qa_id="QA-19",
        facts=facts,
        seed=seed,
        question_en=(
            f"Which time interval contained the first sound from {appearance_en}? "
            f"The clip is divided into {len(bands)} time intervals: {domain_en}."
        ),
        question_zh=(
            f"{appearance_zh}第一次发声落在哪个时间段？"
            f"片段按时间划分为{len(bands)}段：{domain_zh}。"
        ),
        open_answer_type="time_range_s",
        open_truth=truth_range,
        truth_label=band_en,
        options=options,
        mcq_truth=f"band_{band_index}",
        open_extra={
            "time_ranges_s": [list(band) for band in bands],
            "time_range_labels_en": [label[0] for label in labels],
            "time_range_labels_zh": [label[1] for label in labels],
            "time_range_index": band_index,
        },
        evidence={
            "target_actor_id": actor_id,
            "appearance": dict(actor["appearance"]),
            "appearance_review": _appearance_review_for(facts, actor_id),
            "first_event": _event_evidence(first),
            "first_sound_interval_s": truth_range,
            "time_band_index": band_index,
            "time_band_count": len(bands),
            "time_bands_s": [list(band) for band in bands],
        },
        slug=f"{actor_id}_first_time",
    )

def _generate_qa_20(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    _require_stereo(facts)
    try:
        reviewed = _reviewed_appearances(facts)
    except _Deferred:
        # Appearance review controls which visible identities can be named;
        # it must never decide whether a pixel-visible actor exists.
        reviewed = {}
    events = _bound_events(facts)
    for event in events:
        frame = max(0, min(int(facts["time"]["frame_count"]) - 1, _event_frame(event, "start_frame")))
        missing_visibility_ids: list[str] = []
        visible_ids: list[str] = []
        for actor_id in facts["actors"]:
            if actor_id not in facts.get("visibility", {}):
                missing_visibility_ids.append(str(actor_id))
                continue
            try:
                state = _state(facts, str(actor_id), frame)
            except _Deferred:
                missing_visibility_ids.append(str(actor_id))
                continue
            if state.get("state") in VISIBLE_STATES:
                visible_ids.append(str(actor_id))
        if missing_visibility_ids:
            _defer(
                "missing_pixel_visibility",
                "QA-20 cannot classify visible candidates without a state for every actor",
                actor_ids=sorted(set(missing_visibility_ids)),
            )
        if not visible_ids:
            continue
        unreviewed_visible_ids = [
            actor_id for actor_id in visible_ids if actor_id not in reviewed
        ]
        if unreviewed_visible_ids:
            _defer(
                "speaker_appearance_review_missing",
                "a pixel-visible candidate has no reviewed appearance label",
                actor_ids=sorted(unreviewed_visible_ids),
            )
        target_visible = event["actor_id"] in visible_ids
        truth = event["actor_id"] if target_visible else "none_of_visible"
        options = _actor_options(facts, visible_ids)
        options.append(_option("none_of_visible", "none of the visible actors"))
        anchor_en, anchor_zh = _event_anchor(facts, event)
        return _question_item(
            qa_id="QA-20",
            facts=facts,
            seed=seed,
            question_en=(
                f"Which visible actor made {anchor_en}, or did none of them make it?"
            ),
            question_zh=f"{anchor_zh}是画面中哪位发出的，还是都不是？",
            open_answer_type="closed_set",
            open_truth=truth,
            truth_label=(
                "none of the visible actors"
                if truth == "none_of_visible"
                else str(
                    facts["actors"][truth].get("appearance", {}).get(
                        "label", facts["actors"][truth]["display_label"]
                    )
                )
            ),
            options=options,
            evidence={
                **_event_evidence(event),
                "query_frame": frame,
                "visible_candidate_actor_ids": visible_ids,
                "appearance_reviews": {
                    actor_id: _appearance_review_for(facts, actor_id)
                    for actor_id in visible_ids
                },
                "target_visible": target_visible,
                "matched_doa_status": facts.get("sampling", {}).get("matched_doa_status")
                if isinstance(facts.get("sampling"), Mapping)
                else None,
            },
            slug=f"{event['event_id']}_visible_candidate",
        )
    _defer("no_visible_candidate_frame", "no event has at least one visible candidate")


def _generate_qa_21(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    _require_stereo(facts)
    targets: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for actor_id, actor, appearance in _appearance_candidates(facts):
        events = [
            event
            for event in _event_for_actor(facts, actor_id)
            if event.get("sound_class") and event.get("sound_class_explicit")
        ]
        if events:
            targets.append((actor_id, actor, events[0]))
    if not targets:
        _defer("missing_explicit_sound_class", "QA-21 requires an explicit registered sound class")
    target_id, target, event = targets[0]
    target_sound_class = str(event["sound_class"]).casefold()
    if target_sound_class in _CAPABILITY_SOUND_CLASSES:
        _defer(
            "sound_class_capability_only",
            "the target event carries a playback capability rather than an observed sound class",
            sound_class=target_sound_class,
            target_actor_id=target_id,
        )
    target_classes = {
        str(candidate.get("sound_class"))
        for candidate in _event_for_actor(facts, target_id)
        if candidate.get("sound_class")
        and candidate.get("sound_class_explicit")
        and str(candidate.get("sound_class")).casefold() not in _CAPABILITY_SOUND_CLASSES
    }
    if len(target_classes) != 1:
        _defer(
            "target_sound_class_not_unique",
            "QA-21 target appearance maps to more than one sound class",
            target_actor_id=target_id,
            sound_classes=sorted(target_classes),
        )
    observed_classes = list(
        dict.fromkeys(
            str(other.get("sound_class"))
            for other in facts["events"]
            if isinstance(other, Mapping)
            and other.get("sound_class")
            and other.get("sound_class_explicit")
            and str(other.get("sound_class")).casefold() not in _CAPABILITY_SOUND_CLASSES
        )
    )
    classes = observed_classes
    answer_domain: Mapping[str, Any] | None = None
    option_domain_source = "observed_classes_in_this_episode"
    if _ordinary_observation_questions(facts):
        recorded_domain = facts.get("sound_class_answer_domain")
        if (
            isinstance(recorded_domain, Mapping)
            and isinstance(recorded_domain.get("values"), list)
        ):
            answer_domain = recorded_domain
        else:
            sampling = facts.get("sampling")
            policy = sampling.get("acceptance_policy") if isinstance(sampling, Mapping) else None
            configured = policy.get("sound_class_options") if isinstance(policy, Mapping) else None
            if configured is not None and (
                not isinstance(configured, list)
                or any(not isinstance(value, str) for value in configured)
            ):
                raise UnifiedQAError(
                    "ordinary sound_class_options must be a list of registered classes"
                )
            answer_domain = derive_sound_class_answer_domain(
                configured,
                observed_events=facts.get("events", ()),
            )
        classes = list(answer_domain.get("values", ()))
        option_domain_source = (
            "configured_registered_sound_class_catalog"
            if answer_domain.get("boundary") == "explicit_configured_pool"
            else "observed_classes_in_this_episode"
        )
    if len(classes) < 2:
        _defer("sound_class_option_domain_too_small", "QA-21 needs at least two explicit sound classes")
    if event["sound_class"] not in classes:
        _defer("sound_class_truth_missing", "target sound class is absent from the class domain")
    appearance_en, appearance_zh = _appearance_phrases(target["appearance"])
    sound_options = []
    for value in classes:
        label_en, label_zh = _sound_class_phrases(value)
        sound_options.append(
            {
                "value": value,
                "label_en": label_en,
                "label_zh": label_zh,
                "allow_value": True,
            }
        )
    target_sound_label, _target_sound_label_zh = _sound_class_phrases(
        event["sound_class"], event=event
    )
    return _question_item(
        qa_id="QA-21",
        facts=facts,
        seed=seed,
        question_en=(
            f"What sound category did the {appearance_en} make?"
        ),
        question_zh=f"{appearance_zh}发出什么声音类别？",
        open_answer_type="closed_set",
        open_truth=event["sound_class"],
        truth_label=target_sound_label,
        options=sound_options,
        evidence={
            "target_actor_id": target_id,
            "appearance": dict(target["appearance"]),
            "appearance_review": _appearance_review_for(facts, target_id),
            "event": _event_evidence(event),
            "sound_class": event["sound_class"],
            "explicit_class": True,
            "distinct_sound_classes": classes,
            "sound_class_answer_domain": (
                copy.deepcopy(dict(answer_domain))
                if isinstance(answer_domain, Mapping)
                else None
            ),
            **({"option_domain_source": option_domain_source} if _ordinary_observation_questions(facts) else {}),
        },
        slug=target_id,
    )


def _public_entity_count_values(facts: Mapping[str, Any]) -> list[int] | None:
    """Return the public visible-entity count domain for QA-22.

    Explicit visible-count choices in the existing sampling configuration
    are authoritative.  When only ``entities.total_count`` is configured,
    that field is a scene upper bound and the public visible domain is every
    integer from zero through its largest configured total.  An observed
    roster is never promoted into a counterfactual option domain.
    """

    def parse(value: Any) -> list[int] | None:
        if isinstance(value, Mapping):
            for key in ("QA-22", "qa-22", "qa_22", "QA_22"):
                if key in value:
                    parsed = parse(value[key])
                    if parsed is not None:
                        return parsed
            for key in (
                "public_visible_count_domain",
                "visible_count_domain",
                "visible_entity_count_domain",
                "public_visible_count_values",
                "visible_count_values",
                "visible_entity_count_values",
                # Compatibility aliases used by earlier QA sampling drafts.
                "public_entity_count_domain",
                "entity_count_domain",
                "entity_count_values",
                "public_entity_count_values",
                "qa_sampling",
                "entities",
            ):
                if key in value:
                    parsed = parse(value[key])
                    if parsed is not None:
                        return parsed
            for key in ("choices", "values", "allowed", "domain"):
                if key in value:
                    parsed = parse(value[key])
                    if parsed is not None:
                        return parsed
            for key in ("total_count", "visible_count", "entity_count"):
                if key in value:
                    parsed = parse(value[key])
                    if parsed is not None:
                        return parsed
            if "min" in value or "max" in value:
                lower, upper = value.get("min"), value.get("max")
                if (
                    isinstance(lower, bool)
                    or isinstance(upper, bool)
                    or not isinstance(lower, int)
                    or not isinstance(upper, int)
                    or upper < lower
                ):
                    _defer(
                        "invalid_public_entity_count_domain",
                        "QA-22 public entity count range must have integer min <= max",
                    )
                return list(range(lower, upper + 1))
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return [value]
        if _is_sequence(value):
            result: list[int] = []
            for item in value:
                if isinstance(item, bool) or not isinstance(item, int):
                    _defer(
                        "invalid_public_entity_count_domain",
                        "QA-22 public entity count choices must be integers",
                    )
                result.append(int(item))
            return result
        return None

    def normalize(values: list[int], *, owner: str) -> list[int]:
        normalized = sorted(set(values))
        if any(value < 0 for value in normalized):
            _defer(
                "invalid_public_entity_count_domain",
                f"QA-22 {owner} cannot contain negative counts",
            )
        return normalized

    sampling = facts.get("sampling")
    containers: list[Mapping[str, Any]] = []
    if isinstance(sampling, Mapping):
        nested = sampling.get("qa_sampling")
        if isinstance(nested, Mapping):
            containers.append(nested)
        containers.append(sampling)
    # These fields are retained for normalized facts produced by callers that
    # already copied the public request configuration into the facts object.
    if isinstance(facts, Mapping):
        containers.append(facts)

    explicit_keys = (
        "public_visible_count_domain",
        "visible_count_domain",
        "visible_entity_count_domain",
        "public_visible_count_values",
        "visible_count_values",
        "visible_entity_count_values",
        # Compatibility aliases used by earlier QA sampling drafts.
        "public_entity_count_domain",
        "entity_count_domain",
        "entity_count_values",
        "public_entity_count_values",
    )
    total_candidates: list[Any] = []
    for container in containers:
        for key in explicit_keys:
            if key not in container:
                continue
            raw_value = container[key]
            values = parse(raw_value)
            if values is None:
                _defer(
                    "invalid_public_entity_count_domain",
                    f"QA-22 {key} must provide integer visible-count choices",
                )
            return normalize(values, owner=key)
        entities = container.get("entities")
        if isinstance(entities, Mapping):
            for key in (
                "public_visible_count_domain",
                "visible_count_domain",
                "visible_entity_count_domain",
                "public_visible_count_values",
                "visible_count_values",
                "visible_entity_count_values",
                "visible_count",
            ):
                if key not in entities:
                    continue
                raw_value = entities[key]
                values = parse(raw_value)
                if values is None:
                    _defer(
                        "invalid_public_entity_count_domain",
                        f"QA-22 entities.{key} must provide integer visible-count choices",
                    )
                return normalize(values, owner=f"entities.{key}")
            if "total_count" in entities:
                total_candidates.append(entities["total_count"])
        if "total_count" in container and container.get("total_count") is not None:
            total_candidates.append(container["total_count"])

    if total_candidates:
        upper_values: list[int] = []
        for raw_value in total_candidates:
            values = parse(raw_value)
            if values is None or not values:
                _defer(
                    "invalid_public_entity_count_domain",
                    "QA-22 entities.total_count must provide a nonempty integer domain",
                )
            upper_values.extend(values)
        upper_values = normalize(upper_values, owner="entities.total_count")
        if not upper_values:
            _defer(
                "invalid_public_entity_count_domain",
                "QA-22 entities.total_count must provide a nonempty integer domain",
            )
        return list(range(0, max(upper_values) + 1))
    return None


def _generate_qa_22(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    events = facts.get("events")
    if not _is_sequence(events):
        _defer("missing_events", "QA-22 needs a whole-clip event table")
    if any(event.get("actor_id") is None for event in events if isinstance(event, Mapping)):
        _defer("unresolved_event_attribution", "QA-22 cannot count speaking individuals while an event source is unresolved")
    actors = facts.get("actors")
    visibility = facts.get("visibility")
    if not isinstance(actors, Mapping) or not isinstance(visibility, Mapping) or not visibility:
        _defer(
            "missing_visibility_for_entity_count",
            "QA-22 needs explicit pixel visibility to count entities that appear",
        )
    visible_actor_ids: set[str] = set()
    unknown_actor_ids: list[str] = []
    for actor_id in actors:
        frames = visibility.get(actor_id)
        if not isinstance(frames, Mapping) or not frames:
            unknown_actor_ids.append(str(actor_id))
            continue
        states = [
            frame.get("state")
            for frame in frames.values()
            if isinstance(frame, Mapping)
        ]
        if any(state in VISIBLE_STATES for state in states):
            visible_actor_ids.add(str(actor_id))
        elif not _visibility_is_complete(facts, str(actor_id)):
            unknown_actor_ids.append(str(actor_id))
    if unknown_actor_ids:
        _defer(
            "incomplete_visibility_for_entity_count",
            "cannot count entities that may appear in unobserved frames",
            actor_ids=sorted(unknown_actor_ids),
        )
    speaking_actor_ids = {
        str(event["actor_id"])
        for event in events
        if str(event["actor_id"]) in visible_actor_ids
    }
    entity_count = len(visible_actor_ids)
    speaking_count = len(speaking_actor_ids)
    truth = [entity_count, speaking_count]

    configured_counts = _public_entity_count_values(facts)
    option_pairs: list[tuple[int, int]] = []
    mcq_deferred_reason: dict[str, Any] | None = None
    if configured_counts is None:
        mcq_deferred_reason = {
            "code": "missing_public_entity_count_domain",
            "detail": "QA-22 needs configured public entity count choices for distinct MCQ distractors",
        }
    elif entity_count not in configured_counts:
        mcq_deferred_reason = {
            "code": "entity_count_outside_public_domain",
            "detail": "observed QA-22 entity count is absent from the configured public count domain",
            "observed_entity_count": entity_count,
            "configured_entity_count_values": configured_counts,
        }
    else:
        # Every pair is a legal count pair for a configured visible count.
        # Retaining all legal speaking values avoids manufacturing an
        # impossible pair and keeps the observed truth untouched.
        option_pairs = [
            (visible_count, candidate_speaking_count)
            for visible_count in configured_counts
            for candidate_speaking_count in range(visible_count + 1)
        ]
        if not any(visible_count != entity_count for visible_count, _ in option_pairs):
            mcq_deferred_reason = {
                "code": "count_option_domain_too_small",
                "detail": "QA-22 public entity count domain has no alternative visible count",
                "configured_entity_count_values": configured_counts,
            }
    options = [
        {
            **_option(
                f"{visible_count}|{candidate_speaking_count}",
                f"{visible_count} entities, {candidate_speaking_count} speaking",
            ),
            "allow_value": False,
        }
        for visible_count, candidate_speaking_count in option_pairs
    ]
    if mcq_deferred_reason is not None:
        # Keep a deferred one-count domain out of the Open form's choice
        # aliases; its pair values are not a usable public MCQ.
        options = []
    pair_values = [f"{visible_count}|{speaking}" for visible_count, speaking in option_pairs]
    return _question_item(
        qa_id="QA-22",
        facts=facts,
        seed=seed,
        question_en="How many entities appear in the clip, and how many of them make a sound?",
        question_zh="整段中出现过多少个体，其中多少个体发过声？",
        open_answer_type="count_pair",
        open_truth=truth,
        truth_label=f"{entity_count}, {speaking_count}",
        options=options,
        mcq_truth=f"{entity_count}|{speaking_count}",
        mcq_deferred_reason=mcq_deferred_reason,
        evidence={
            "statistics_window": [0.0, float(facts["time"]["duration_seconds"])],
            "entity_count": entity_count,
            "appeared_actor_ids": sorted(visible_actor_ids),
            "speaking_actor_ids": sorted(speaking_actor_ids),
            "speaking_count": speaking_count,
            "option_domain": {
                "entity_count_values": configured_counts,
                "pair_values": pair_values,
                "speaking_count_values_by_entity_count": (
                    {str(value): list(range(value + 1)) for value in configured_counts}
                    if configured_counts is not None
                    else {}
                ),
            },
            "answer_prior_diagnostic": {
                "all_appeared_entities_speak": speaking_count == entity_count,
                "no_appeared_entities_speak": speaking_count == 0,
                "speaking_count_is_boundary": speaking_count in {0, entity_count},
            },
        },
        slug="whole_clip",
    )


def _generate_qa_23(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    events = facts.get("events")
    if not _is_sequence(events):
        _defer("missing_events", "QA-23 needs a whole-clip event table")
    count = len(events)
    if count < 1:
        _defer("missing_events", "QA-23 needs at least one event")
    unreviewed_non_speech = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        sound_class = str(event.get("sound_class") or "").casefold()
        is_speech = bool(event.get("transcript")) or sound_class in {
            "speech",
            "speech_playback",
            "utterance",
            "human_speech",
        }
        if not is_speech:
            proven, _detail = _event_segmentation_proven(facts, event)
            if not proven:
                unreviewed_non_speech.append(event.get("event_id"))
    if unreviewed_non_speech:
        _defer(
            "event_segmentation_not_reviewed",
            "non-speech source clips need a source/final-audio event segmentation review before counting starts",
            event_ids=unreviewed_non_speech,
        )
    option_count = max(4, count + 1)
    options = [_option(str(value), str(value)) for value in range(option_count)]
    return _question_item(
        qa_id="QA-23",
        facts=facts,
        seed=seed,
        question_en="How many independent sound events begin during the clip?",
        question_zh="整段中有多少次独立发声事件开始？",
        open_answer_type="count_single",
        open_truth=[count],
        truth_label=str(count),
        options=options,
        mcq_truth=str(count),
        evidence={
            "statistics_window": [0.0, float(facts["time"]["duration_seconds"])],
            "event_definition": "one audio_program event id",
            "event_ids": [event.get("event_id") for event in events],
            "non_speech_segmentation_review": {
                event.get("event_id"): _event_segmentation_proven(facts, event)[1]
                for event in events
                if isinstance(event, Mapping)
                and not event.get("transcript")
                and str(event.get("sound_class") or "").casefold()
                not in {"speech", "speech_playback", "utterance", "human_speech"}
            },
        },
        slug="whole_clip_event_count",
    )


def _generate_qa_24(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    event = _earliest_event(facts)
    actor_id = event["actor_id"]
    final_frame = int(facts["time"]["frame_count"]) - 1
    state = _require_visibility(facts, actor_id, final_frame).get("state")
    return _question_item(
        qa_id="QA-24",
        facts=facts,
        seed=seed,
        question_en="What visibility state does the first speaker have at the end of the clip?",
        question_zh="最先发声的个体在片尾处于什么可见状态？",
        open_answer_type="closed_set",
        open_truth=state,
        truth_label=_state_label(state),
        options=_state_options(),
        evidence={
            "anchor_event": _event_evidence(event),
            "target_actor_id": actor_id,
            "final_frame": final_frame,
            "final_visibility_state": state,
        },
        slug="first_speaker_final_state",
    )


def _generate_qa_25(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    from avengine.qa.angular_questions import candidates, emit
    rows = candidates(facts)
    if not rows:
        _defer("no_continuous_bearing_candidate", "no QA-25 subset has sufficient native evidence")
    return emit(facts, rows[0], seed)


_GENERATORS = {
    "QA-01": _generate_qa_01,
    "QA-02": _generate_qa_02,
    "QA-03": _generate_qa_03,
    "QA-04": _generate_qa_04,
    "QA-05": _generate_qa_05,
    "QA-06": _generate_qa_06,
    "QA-07": _generate_qa_07,
    "QA-08": _generate_qa_08,
    "QA-09": _generate_qa_09,
    "QA-10": _generate_qa_10,
    "QA-11": _generate_qa_11,
    "QA-12": _generate_qa_12,
    "QA-13": _generate_qa_13,
    "QA-14": _generate_qa_14,
    "QA-15": _generate_qa_15,
    "QA-16": _generate_qa_16,
    "QA-17": _generate_qa_17,
    "QA-18": _generate_qa_18,
    "QA-19": _generate_qa_19,
    "QA-20": _generate_qa_20,
    "QA-21": _generate_qa_21,
    "QA-22": _generate_qa_22,
    "QA-23": _generate_qa_23,
    "QA-24": _generate_qa_24,
    "QA-25": _generate_qa_25,
}


def _restore_normalized_frame_keys(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Restore integer frame keys lost when normalized facts go through JSON."""

    visibility = facts.get("visibility")
    if not isinstance(visibility, Mapping):
        return dict(facts)
    restored: dict[str, dict[Any, Any]] = {}
    for actor_id, frames in visibility.items():
        if not isinstance(frames, Mapping):
            continue
        by_frame: dict[Any, Any] = {}
        for frame, value in frames.items():
            key: Any = frame
            if isinstance(frame, str):
                try:
                    key = int(frame)
                except ValueError:
                    pass
            by_frame[key] = value
        restored[str(actor_id)] = by_frame
    result = dict(facts)
    result["visibility"] = restored
    return result


__all__ = [
    "CATALOG",
    "CATALOG_VERSION",
    "QUESTION_CATALOG",
    "UNIFIED_FACT_SCHEMA",
    "UNIFIED_INPUT_SCHEMA",
    "UNIFIED_ITEM_SCHEMA",
    "UNIFIED_OUTPUT_SCHEMA",
    "UnifiedQAError",
    "generate_unified_questions",
    "generate_questions",
    "get_requirements",
    "get_question_requirements",
    "build_unified_episode_facts",
    "normalize_episode_bundle",
    "question_catalog",
    "derive_sound_class_answer_domain",
    "with_derived_sound_class_answer_domain",
]

# Small descriptive aliases for callers that do not need the longer function
# names. They are aliases, not additional registries or question protocols.
QUESTION_CATALOG = CATALOG
get_question_requirements = get_requirements
build_unified_episode_facts = normalize_episode_bundle
def _sampling_window_bounds(value: Any) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        pairs = (
            ("start_frame", "end_frame_exclusive"),
            ("start", "end"),
            ("lo", "hi"),
        )
        for start_key, end_key in pairs:
            if start_key in value and end_key in value:
                value = (value[start_key], value[end_key])
                break
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        start, end = value
        if (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
        ):
            return int(start), int(end)
    return None



def _window_bounds_list(value: Any, *, qa_id: str) -> list[tuple[int, int]]:
    """Normalize one or more half-open frame windows."""
    bounds = _sampling_window_bounds(value)
    if bounds is not None:
        return [bounds]
    if _is_sequence(value):
        windows: list[tuple[int, int]] = []
        for part in value:
            part_bounds = _sampling_window_bounds(part)
            if part_bounds is None:
                _defer(
                    "sampling_window_invalid",
                    f"{qa_id} legal windows must be half-open frame intervals",
                )
            windows.append(part_bounds)
        if windows:
            return windows
    _defer(
        "sampling_window_invalid",
        f"{qa_id} legal windows must be half-open frame intervals",
    )


def _compress_frame_windows(frames: Sequence[int]) -> list[list[int]]:
    ordered = sorted({int(frame) for frame in frames})
    if not ordered:
        return []
    windows: list[list[int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            windows.append([start, previous + 1])
            start = frame
        previous = frame
    windows.append([start, previous + 1])
    return windows


def _frame_window_seconds(
    facts: Mapping[str, Any],
    window: Sequence[int],
) -> list[float]:
    start, end = (int(window[0]), int(window[1]))
    fps = float(facts["time"]["frame_rate_hz"])
    return [start / fps, end / fps]


def _time_display_precision(facts: Mapping[str, Any]) -> int:
    """Read the public interval precision from the episode sampling policy."""

    owners: list[Mapping[str, Any]] = []
    sampling = facts.get("sampling")
    if isinstance(sampling, Mapping):
        nested = sampling.get("qa_sampling")
        if isinstance(nested, Mapping):
            owners.append(nested)
        owners.append(sampling)
    policy = facts.get("sampling_policy")
    if isinstance(policy, Mapping):
        owners.append(policy)
    value: Any = None
    for owner in owners:
        for key in (
            "time_display_precision",
            "time_interval_precision",
            "query_time_precision",
            "time_range_precision",
            "time_precision",
            "public_time_precision",
        ):
            if key in owner:
                value = owner[key]
                break
        if value is not None:
            break
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9:
        _defer(
            "time_display_precision_invalid",
            "public time interval precision must be an integer from zero through nine",
            value=value,
        )
    return int(value)


def publishable_query_window(
    facts: Mapping[str, Any],
    window: Sequence[int],
    *,
    start_rounding: str = "ceil",
) -> dict[str, Any]:
    """Report whether a proven frame interval survives public quantization.

    A public question states whole seconds, so the readable interval is the
    proven interval quantized inward: the start rounds up and the end rounds
    down. A window that is shorter than one public step therefore has no
    publishable form even though the underlying evidence is sound. That is a
    different rejection from an interval whose answer is not distinguishable,
    so this returns the measured numbers instead of a bare boolean and never
    widens the interval to make it expressible.
    """

    exact_start, exact_end = _frame_window_seconds(facts, window)
    precision = _time_display_precision(facts)
    scale = float(10**precision)
    step = 1.0 / scale
    # Epsilon only removes binary representation noise at an exact decimal
    # boundary. QA-07 can deliberately floor the public start to contain its
    # entry frame; all other callers retain the inward ceil.
    if start_rounding == "ceil":
        lower_units = math.ceil(exact_start * scale - 1.0e-9)
    elif start_rounding == "floor":
        lower_units = math.floor(exact_start * scale + 1.0e-9)
    else:
        raise ValueError("start_rounding must be 'ceil' or 'floor'")
    upper_units = math.floor(exact_end * scale + 1.0e-9)
    record: dict[str, Any] = {
        "query_window_frames": [int(window[0]), int(window[1])],
        "query_window_exact_s": [exact_start, exact_end],
        "query_window_precision": precision,
        "public_time_step_s": step,
    }
    if upper_units > lower_units:
        record["publishable"] = True
        record["query_window_s"] = [lower_units / scale, upper_units / scale]
        return record
    record["publishable"] = False
    record["reason"] = "query_interval_too_short_for_display"
    record["proven_span_s"] = exact_end - exact_start
    record["shortfall_s"] = max(0.0, step - (exact_end - exact_start))
    record["inward_lower_s"] = lower_units / scale
    record["inward_upper_s"] = upper_units / scale
    return record


def _display_time_bounds(
    facts: Mapping[str, Any],
    window: Sequence[int],
    *,
    start_rounding: str = "ceil",
) -> tuple[float, float] | None:
    """Quantize a proven frame interval inward for a readable public range."""

    record = publishable_query_window(
        facts, window, start_rounding=start_rounding
    )
    if not record["publishable"]:
        return None
    bounds = record["query_window_s"]
    return float(bounds[0]), float(bounds[1])


def published_window_frames(
    facts: Mapping[str, Any],
    published_s: Sequence[float],
) -> list[int]:
    """Frames whose own instants fall inside the published half-open interval."""

    fps = float(facts["time"]["frame_rate_hz"])
    lower = math.ceil(float(published_s[0]) * fps - 1.0e-9)
    upper = math.ceil(float(published_s[1]) * fps - 1.0e-9)
    frame_count = int(facts["time"]["frame_count"])
    lower = max(0, lower)
    upper = min(frame_count, upper)
    return list(range(lower, upper))


def verify_published_query_window(
    facts: Mapping[str, Any],
    window: Sequence[int],
    value_for_frame: Any,
    expected: Any,
    *,
    qa_id: str,
) -> dict[str, Any]:
    """Recompute the answer on the published interval, not the proven one.

    Quantizing the public text is not a cosmetic step: the readable interval
    is a different set of instants from the proven frame window, so the truth
    is measured again on exactly the instants a reader is given. This refuses
    to publish a rounded restatement of an answer that was established
    somewhere else.
    """

    record = publishable_query_window(facts, window)
    if not record["publishable"]:
        _defer(
            "query_interval_too_short_for_display",
            f"{qa_id} has a proven interval that no public whole step can express",
            **{key: value for key, value in record.items() if key != "publishable"},
        )
    frames = published_window_frames(facts, record["query_window_s"])
    record["published_window_frames"] = [
        (frames[0], frames[-1] + 1) if frames else []
    ][0]
    if not frames:
        _defer(
            "published_window_has_no_frame",
            f"{qa_id} published interval contains no readback frame",
            query_window_s=record["query_window_s"],
            query_window_frames=record["query_window_frames"],
        )
    outside = [
        frame for frame in frames
        if not int(window[0]) <= frame < int(window[1])
    ]
    if outside:
        _defer(
            "published_window_outside_proven_window",
            f"{qa_id} published interval reaches frames the evidence never proved",
            frames_outside=outside,
            query_window_frames=record["query_window_frames"],
            query_window_s=record["query_window_s"],
        )
    values = [value_for_frame(frame) for frame in frames]
    record["published_window_values"] = copy.deepcopy(values)
    changed = [
        frame for frame, value in zip(frames, values) if value != expected
    ]
    if changed:
        _defer(
            "published_window_truth_changed",
            f"{qa_id} answer does not hold on every instant of the published interval",
            frames_with_other_answer=changed,
            expected=copy.deepcopy(expected),
            query_window_s=record["query_window_s"],
        )
    record["published_window_recomputed"] = True
    return record


def _query_window_fields(
    facts: Mapping[str, Any],
    window: Sequence[int],
    *,
    start_rounding: str = "ceil",
) -> dict[str, Any]:
    record = publishable_query_window(
        facts, window, start_rounding=start_rounding
    )
    if not record["publishable"]:
        _defer(
            "query_interval_too_short_for_display",
            "the stable query interval cannot be expressed at the configured public precision",
            **{key: value for key, value in record.items() if key != "publishable"},
        )
    return {
        "query_window_frames": record["query_window_frames"],
        "query_window_s": [float(value) for value in record["query_window_s"]],
        "query_window_exact_s": record["query_window_exact_s"],
        "query_window_precision": record["query_window_precision"],
        "public_time_step_s": record["public_time_step_s"],
    }


def _display_time_range(
    facts: Mapping[str, Any],
    window: Sequence[int],
    *,
    start_rounding: str = "ceil",
) -> tuple[str, str] | None:
    """Describe a proven frame interval with explicit half-open boundaries."""

    display = _display_time_bounds(
        facts, window, start_rounding=start_rounding
    )
    if display is None:
        return None
    precision = _time_display_precision(facts)
    start_text = _format_public_seconds(display[0], precision=precision)
    end_text = _format_public_seconds(display[1], precision=precision)
    return (
        f"[{start_text}, {end_text}) seconds",
        f"第{start_text}至第{end_text}秒的时间段（左闭右开）",
    )


def _stable_frame_window(
    facts: Mapping[str, Any],
    windows: Any,
    query_frame: int,
    value_for_frame: Any,
    values_are_stable: Any,
    *,
    min_frames: int = 2,
) -> list[int] | None:
    """Find the longest legal interval containing a query frame with one answer."""
    normalized = _window_bounds_list(windows, qa_id="query")
    query_frame = int(query_frame)
    for start, end in normalized:
        if not start <= query_frame < end:
            continue
        values: dict[int, Any] = {}
        for frame in range(start, end):
            try:
                values[frame] = value_for_frame(frame)
            except _Deferred:
                return None
        best: tuple[int, int] | None = None
        for lower in range(query_frame, start - 1, -1):
            for upper in range(query_frame + 1, end + 1):
                if upper - lower < int(min_frames):
                    continue
                if not values_are_stable(
                    [values[frame] for frame in range(lower, upper)]
                ):
                    continue
                if best is None or upper - lower > best[1] - best[0]:
                    best = (lower, upper)
        if best is not None:
            return [best[0], best[1]]
    return None


def _event_start_side_window(
    facts: Mapping[str, Any],
    event: Mapping[str, Any],
) -> tuple[list[int], str, float] | None:
    frame_count = int(facts["time"]["frame_count"])
    start = max(0, min(frame_count - 1, _event_frame(event, "start_frame")))
    stop = min(frame_count, max(start + 1, _event_frame(event, "end_frame") + 1))

    def side_and_angle(frame: int) -> tuple[str, float] | None:
        try:
            angle = float(_azimuth(facts, str(event["actor_id"]), frame))
        except _Deferred:
            return None
        if abs(angle) < 5.0:
            return None
        return ("right" if angle > 0.0 else "left", angle)

    values = [side_and_angle(frame) for frame in range(start, stop)]
    if not values or values[0] is None:
        return None
    side = values[0][0]
    end = start + 1
    while end < stop and values[end - start] is not None and values[end - start][0] == side:
        end += 1
    if end - start < 2:
        return None
    return [start, end], side, values[0][1]


def _event_start_visibility_window(
    facts: Mapping[str, Any],
    event: Mapping[str, Any],
) -> tuple[list[int], str] | None:
    frame_count = int(facts["time"]["frame_count"])
    start = max(0, min(frame_count - 1, _event_frame(event, "start_frame")))
    stop = min(frame_count, max(start + 1, _event_frame(event, "end_frame") + 1))
    try:
        target_state = str(_state(facts, str(event["actor_id"]), start).get("state"))
    except _Deferred:
        return None
    end = start + 1
    while end < stop:
        try:
            state = str(_state(facts, str(event["actor_id"]), end).get("state"))
        except _Deferred:
            break
        if state != target_state:
            break
        end += 1
    if end - start < 2:
        return None
    return [start, end], target_state


def _entry_transition_window(
    facts: Mapping[str, Any],
    actor_id: str,
    entry_frame: int,
    *,
    center: float,
    dead_zone: float,
) -> list[int] | None:
    frames = facts.get("visibility", {}).get(actor_id)
    if not isinstance(frames, Mapping):
        return None

    def side(frame: int) -> str | None:
        try:
            row = _state(facts, actor_id, frame)
        except _Deferred:
            return None
        if row.get("state") not in VISIBLE_STATES:
            return None
        centroid = row.get("target_centroid_xy_px")
        if not _is_sequence(centroid) or len(centroid) != 2:
            return None
        offset = float(centroid[0]) - center
        if abs(offset) <= dead_zone:
            return None
        return "right" if offset > 0.0 else "left"

    target = side(int(entry_frame))
    if target is None:
        return None
    end = int(entry_frame) + 1
    frame_count = int(facts["time"]["frame_count"])
    while end < frame_count and side(end) == target:
        end += 1
    if end - int(entry_frame) < 2:
        return None
    return [int(entry_frame), end]


def _occlusion_interval_window(
    facts: Mapping[str, Any],
    actor_id: str,
    query_frame: int,
) -> list[int] | None:
    frame_count = int(facts["time"]["frame_count"])

    def occluder(frame: int) -> str | None:
        try:
            state = _state(facts, actor_id, frame)
        except _Deferred:
            return None
        if state.get("state") not in {"visible_occluded", "fully_occluded"}:
            return None
        ids = _occluder_ids(facts, actor_id, frame)
        return ids[0] if len(ids) == 1 else None

    return _stable_frame_window(
        facts,
        [[0, frame_count]],
        int(query_frame),
        occluder,
        lambda values: bool(values) and len(set(values)) == 1,
    )


def _post_sound_window(
    facts: Mapping[str, Any],
    event: Mapping[str, Any],
    silence: Mapping[str, Any],
    *,
    qa_id: str,
) -> Any:
    windows = silence.get("legal_query_windows")
    if windows is not None:
        return windows
    return _derived_legal_query_windows(
        facts,
        qa_id,
        event=event,
    )


def _query_time_policy(facts: Mapping[str, Any]) -> Any:
    sampling = facts.get("sampling")
    if not isinstance(sampling, Mapping):
        return None
    nested = sampling.get("qa_sampling")
    nested = nested if isinstance(nested, Mapping) else {}
    return _first(nested, "query_time_policy", "policy") or _first(
        sampling, "query_time_policy", "policy"
    )


def _derived_query_window_authority(qa_id: str) -> str:
    if qa_id in {"QA-13", "QA-16", "QA-17"}:
        return "native_audio_event_gap_and_wet_tail_readback_v1"
    if qa_id == "QA-18":
        return "native_wet_tail_complement_frame_clock_v1"
    return "derived_native_query_window"


def _record_derived_query_windows(
    facts: Mapping[str, Any],
    qa_id: str,
    windows: Sequence[Sequence[int]],
    *,
    event: Mapping[str, Any] | None = None,
) -> None:
    sampling = facts.get("sampling")
    if not isinstance(sampling, MutableMapping):
        return
    derived = sampling.setdefault("derived_legal_windows_by_qa", {})
    if not isinstance(derived, MutableMapping):
        return
    normalized = [[int(start), int(end)] for start, end in windows]
    if qa_id in {"QA-13", "QA-16", "QA-17"} and event is not None:
        by_event = derived.setdefault(qa_id, {})
        if isinstance(by_event, MutableMapping):
            event_id = event.get("event_id")
            if event_id is not None:
                by_event[str(event_id)] = copy.deepcopy(normalized)
    else:
        derived[qa_id] = copy.deepcopy(normalized)


def _derived_legal_query_windows(
    facts: Mapping[str, Any],
    qa_id: str,
    *,
    event: Mapping[str, Any] | None = None,
) -> list[list[int]] | None:
    """Derive legal query windows from native audio predicates when absent."""
    frame_count = int(facts["time"]["frame_count"])
    if qa_id in {"QA-13", "QA-16", "QA-17"}:
        if event is None:
            return None
        legal_frames: list[int] = []
        missing_readback = False
        for frame in range(frame_count):
            try:
                _silent_after(facts, event, frame)
            except _Deferred as error:
                if error.code in {"missing_wet_tail_readback", "missing_event_wet_tail"}:
                    missing_readback = True
                continue
            legal_frames.append(frame)
        if missing_readback and not legal_frames:
            return None
        return _compress_frame_windows(legal_frames)
    if qa_id == "QA-18":
        audio = facts.get("audio")
        tails = audio.get("wet_tail_intervals") if isinstance(audio, Mapping) else None
        if not _is_sequence(tails) or not tails:
            return None
        normalized_tails: list[tuple[float, float]] = []
        for interval in tails:
            if not isinstance(interval, Mapping):
                return None
            try:
                start = float(interval["start_s"])
                end = float(interval["end_s"])
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                return None
            normalized_tails.append((start, end))
        fps = float(facts["time"]["frame_rate_hz"])
        safe_frames = {
            frame for frame in range(frame_count)
            if not any(start <= frame / fps < end for start, end in normalized_tails)
        }
        # A source-active frame is the intended positive branch of QA-18 and
        # must remain eligible even though its own listener-side wet tail
        # necessarily overlaps the source activity.  Empty frames still need
        # the wet-tail complement so "no actor" does not mean "only a tail".
        active_frames: set[int] = set()
        if _source_activity_present(facts):
            for frame in range(frame_count):
                if _active_at(facts, frame, require_source_activity=True):
                    active_frames.add(frame)
        return _compress_frame_windows(sorted(safe_frames | active_frames))
    return None


def _sample_frame_from_windows(
    facts: Mapping[str, Any],
    qa_id: str,
    windows: Sequence[Sequence[int]],
) -> int:
    frames = sorted({
        frame
        for start, end in _window_bounds_list(windows, qa_id=qa_id)
        for frame in range(start, end)
    })
    if not frames:
        _defer(
            "no_valid_post_sound_window",
            f"{qa_id} has no legal query frame",
        )
    generation_seed = str(facts.get("_generation_seed", ""))
    window_key = repr(tuple(tuple(int(value) for value in window) for window in windows))
    return random.Random(f"{generation_seed}\0{qa_id}\0{window_key}").choice(frames)

def _resolve_query_frame_spec(
    facts: Mapping[str, Any],
    qa_id: str,
    value: Any,
    *,
    source: str,
) -> int:
    frame_count = int(facts["time"]["frame_count"])
    if isinstance(value, int) and not isinstance(value, bool):
        frame = int(value)
    elif isinstance(value, Mapping):
        policy = value.get("policy") or value.get("query_time_policy")
        if policy == "uniform_in_legal_window":
            window = (
                value.get("window_frames")
                or value.get("frame_window")
                or value.get("legal_window")
                or value.get("window")
            )
            windows = _window_bounds_list(window, qa_id=qa_id)
            for start, end in windows:
                if start < 0 or end > frame_count or start >= end:
                    _defer(
                        "sampling_window_invalid",
                        f"{qa_id} legal frame window {(start, end)} is outside the frame clock",
                        window=[start, end],
                    )
            generation_seed = str(facts.get("_generation_seed", ""))
            if len(windows) == 1:
                start, end = windows[0]
                frame = random.Random(
                    f"{generation_seed}\\0{qa_id}\\0{start}\\0{end}"
                ).randrange(start, end)
            else:
                frame = _sample_frame_from_windows(facts, qa_id, windows)
        else:
            nested = next(
                (
                    value[key]
                    for key in ("frame", "query_frame", "at_frame")
                    if key in value
                ),
                None,
            )
            if nested is None:
                _defer(
                    "invalid_query_frame",
                    f"{qa_id} has an unrecognized query frame specification",
                    source=source,
                )
            return _resolve_query_frame_spec(
                facts, qa_id, nested, source=source
            )
    else:
        _defer(
            "invalid_query_frame",
            f"{qa_id} has an unrecognized query frame specification",
            source=source,
        )
    if frame < 0 or frame >= frame_count:
        _defer(
            "query_frame_out_of_range",
            f"{qa_id} query frame {frame} is outside [0, {frame_count})",
            frame=frame,
            source=source,
        )
    return frame


def _resolve_query_time_spec(
    facts: Mapping[str, Any],
    qa_id: str,
    value: Any,
    *,
    source: str,
) -> float:
    duration = float(facts["time"]["duration_seconds"])
    if isinstance(value, Mapping):
        policy = value.get("policy") or value.get("query_time_policy")
        if policy == "uniform_in_legal_window":
            frame = _resolve_query_frame_spec(
                facts, qa_id, value, source=source
            )
            return frame / float(facts["time"]["frame_rate_hz"])
        nested = next(
            (
                value[key]
                for key in ("time_s", "query_time_s", "seconds")
                if key in value
            ),
            None,
        )
        if nested is None:
            _defer(
                "invalid_query_time",
                f"{qa_id} has an unrecognized query time specification",
                source=source,
            )
        value = nested
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _defer(
            "invalid_query_time",
            f"{qa_id} query time is not numeric",
            source=source,
        )
    time_s = _finite_number(value, name="query_time_s")
    if time_s < 0.0 or time_s > duration:
        _defer(
            "query_time_out_of_range",
            f"{qa_id} query time {time_s} is outside [0, {duration}]",
            query_time_s=time_s,
            source=source,
        )
    return time_s


def _nested_evidence_value(
    evidence: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key in evidence and evidence[key] is not None:
            return evidence[key]
    for parent in ("event", "first_event", "anchor_event", "post_sound"):
        nested = evidence.get(parent)
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested and nested[key] is not None:
                    return nested[key]
    return None


def _question_id_for(
    canonical: str,
    facts: Mapping[str, Any],
    slug: str,
    evidence: Mapping[str, Any],
) -> str:
    """Include every fact that can distinguish a sampled question."""

    parts = [
        canonical.lower().replace("-", "_"),
        _safe_slug(str(facts["episode_id"])),
    ]
    target = _nested_evidence_value(
        evidence,
        "target_actor_id",
        "actor_id",
        "candidate_actor_id",
    )
    event = _nested_evidence_value(evidence, "event_id")
    if event is None:
        event_ids = _nested_evidence_value(evidence, "event_ids")
        if isinstance(event_ids, Sequence) and not isinstance(event_ids, (str, bytes)):
            event = ",".join(str(value) for value in event_ids)
    frame = _nested_evidence_value(evidence, "query_frame", "frame")
    query_time = _nested_evidence_value(evidence, "query_time_s")
    query_window = _nested_evidence_value(
        evidence,
        "query_window",
        "window",
        "motion_frames",
        "statistics_window",
        "observation_window",
    )
    if target is not None:
        parts.append(f"target_{_safe_slug(str(target))}")
    if event is not None:
        parts.append(f"event_{_safe_slug(str(event))}")
    if frame is not None:
        parts.append(f"frame_{_safe_slug(str(frame))}")
    if query_time is not None:
        parts.append(f"time_{_safe_slug(str(query_time))}")
    if query_window is not None:
        parts.append(f"window_{_safe_slug(str(query_window))}")
    parts.append(_safe_slug(str(slug)))
    return "__".join(parts)


def structural_baselines(candidate_values: Mapping[str, Any] | Sequence[Any],
                         gold_actor: str | None, *, answer_domain_size: int | None = None) -> dict[str, Any]:
    from avengine.qa.answerability import structural_baselines as shared_baselines
    values = candidate_values if isinstance(candidate_values, Mapping) else {str(i): v for i, v in enumerate(candidate_values)}
    return shared_baselines(values, gold_actor, answer_domain_size=answer_domain_size)


def distractors_equal_gold(
    candidate_values: Mapping[str, Any],
    gold_actor: str,
) -> bool:
    """Return the diagnostic only; this function never rejects a question."""

    gold = candidate_values.get(gold_actor)
    if gold is None:
        return False
    distractors = [
        value
        for actor_id, value in candidate_values.items()
        if actor_id != gold_actor and value is not None
    ]
    return bool(distractors) and all(value == gold for value in distractors)


def _candidate_actor_values(
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    actors = facts.get("actors")
    if not isinstance(actors, Mapping):
        return values
    for actor_id, actor in actors.items():
        if not isinstance(actor, Mapping):
            continue
        appearance = actor.get("appearance")
        if isinstance(appearance, Mapping):
            value = appearance.get("value", appearance.get("label"))
        else:
            value = actor.get("display_label")
        values[str(actor_id)] = value if value is not None and str(value).strip() else None
    return values


def _p8_actor_ids(facts: Mapping[str, Any]) -> list[str]:
    actors = facts.get("actors")
    return [
        str(actor_id)
        for actor_id, actor in actors.items()
        if isinstance(actor, Mapping)
    ] if isinstance(actors, Mapping) else []


def _p8_ordered_visibility(
    facts: Mapping[str, Any],
    actor_id: str,
) -> list[Mapping[str, Any]]:
    frames = facts.get("visibility", {}).get(actor_id)
    if not isinstance(frames, Mapping):
        return []
    return [
        value
        for index, value in sorted(frames.items(), key=lambda pair: int(pair[0]))
        if isinstance(value, Mapping)
    ]


def _p8_entry_transition_candidates(
    facts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reviewed = _reviewed_appearances(facts)
    result: list[dict[str, Any]] = []
    resolution = facts.get("visibility_meta", {}).get("resolution_hw")
    if not _is_sequence(resolution) or len(resolution) != 2:
        return result
    center = (float(resolution[1]) - 1.0) / 2.0
    dead_zone = max(1.0, float(resolution[1]) * 0.02)
    for actor_id in _p8_actor_ids(facts):
        if actor_id not in reviewed:
            continue
        ordered = _p8_ordered_visibility(facts, actor_id)
        for previous, current in zip(ordered, ordered[1:]):
            previous_index = previous.get("frame_index")
            current_index = current.get("frame_index")
            centroid = current.get("target_centroid_xy_px")
            if (
                not isinstance(previous_index, int)
                or not isinstance(current_index, int)
                or current_index != previous_index + 1
                or previous.get("state") != "out_of_view"
                or current.get("state") not in VISIBLE_STATES
                or not _is_sequence(centroid)
                or len(centroid) != 2
            ):
                continue
            offset = float(centroid[0]) - center
            if abs(offset) <= dead_zone:
                continue
            result.append(
                {
                    "candidate_id": f"QA-07:actor:{actor_id}:frame:{current_index}",
                    "kind": "entry_transition",
                    "actor_id": actor_id,
                    "query_frame": current_index,
                    "entry_frame": current_index,
                    "entry_side": "right" if offset > 0 else "left",
                }
            )
    return result


def _p8_reappearance_candidates(facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    # The question asks an existential property of the whole clip. Each actor
    # contributes one candidate, regardless of how long it stays occluded.
    reviewed = _reviewed_appearances(facts)
    result = []
    for actor_id in _p8_actor_ids(facts):
        if actor_id not in reviewed:
            continue
        rows = _p8_ordered_visibility(facts, actor_id)
        fully = [int(row["frame_index"]) for row in rows
                 if row.get("state") == "fully_occluded"]
        if not fully:
            continue
        positive = any(row.get("state") in VISIBLE_STATES
                       and int(row["frame_index"]) > min(fully) for row in rows)
        seen_before = any(row.get("state") in VISIBLE_STATES
                          and int(row["frame_index"]) < min(fully) for row in rows)
        if positive or (seen_before and _visibility_is_complete(facts, actor_id)):
            result.append({"candidate_id": f"QA-09:actor:{actor_id}:whole-clip",
                           "kind": "whole_clip_reappearance", "actor_id": actor_id})
    return result


def _p8_partial_clear_candidates(facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    reviewed = _reviewed_appearances(facts)
    result = []
    for actor_id in _p8_actor_ids(facts):
        if actor_id not in reviewed:
            continue
        rows = _p8_ordered_visibility(facts, actor_id)
        if not any(row.get("state") == "visible_occluded" for row in rows):
            continue
        positive = any(previous.get("state") == "visible_occluded"
                       and current.get("state") == "visible_clear"
                       and int(current["frame_index"]) == int(previous["frame_index"]) + 1
                       for previous, current in zip(rows, rows[1:]))
        if positive or _visibility_is_complete(facts, actor_id):
            result.append({"candidate_id": f"QA-11:actor:{actor_id}:whole-clip",
                           "kind": "whole_clip_partial_clear", "actor_id": actor_id})
    return result


def _p8_occlusion_candidates(
    facts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reviewed = _reviewed_appearances(facts)
    result: list[dict[str, Any]] = []
    for actor_id in _p8_actor_ids(facts):
        if actor_id not in reviewed:
            continue
        for row in _p8_ordered_visibility(facts, actor_id):
            frame = row.get("frame_index")
            if (
                not isinstance(frame, int)
                or row.get("state") not in {"visible_occluded", "fully_occluded"}
            ):
                continue
            ids = _occluder_ids(facts, actor_id, frame)
            if len(ids) != 1:
                continue
            result.append(
                {
                    "candidate_id": (
                        f"QA-10:actor:{actor_id}:frame:{frame}:"
                        f"occluder:{ids[0]}"
                    ),
                    "kind": "occlusion_instance",
                    "actor_id": actor_id,
                    "query_frame": frame,
                    "occlusion_frame": frame,
                    "occluder_id": ids[0],
                }
            )
    return result


def _p8_window_frames(
    facts: Mapping[str, Any],
    qa_id: str,
    value: Any,
) -> list[int] | None:
    if isinstance(value, Mapping):
        policy = value.get("policy") or value.get("query_time_policy")
        if policy == "uniform_in_legal_window":
            value = _first(
                value,
                "window_frames",
                "frame_window",
                "legal_window",
                "window",
            )
    if isinstance(value, Mapping):
        bounds = _sampling_window_bounds(value)
        if bounds is None:
            return None
        windows: list[tuple[int, int]] = [bounds]
    elif _is_sequence(value) and len(value) == 2:
        if all(isinstance(part, int) and not isinstance(part, bool) for part in value):
            windows = [(int(value[0]), int(value[1]))]
        else:
            windows = []
            for part in value:
                bounds = _sampling_window_bounds(part)
                if bounds is None:
                    return None
                windows.append(bounds)
    else:
        return None
    frame_count = int(facts["time"]["frame_count"])
    if any(start < 0 or start >= end or end > frame_count for start, end in windows):
        _defer(
            "sampling_window_invalid",
            f"{qa_id} legal frame window is outside the frame clock",
            windows=[list(window) for window in windows],
        )
    return sorted({frame for start, end in windows for frame in range(start, end)})


def _p8_qa14_frames(facts: Mapping[str, Any]) -> list[int]:
    value = _sampling_value(
        facts,
        "QA-14",
        "query_frame_by_qa",
        "query_frames",
        "query_frame",
        "at_frame",
    )
    time_value = _sampling_value(
        facts,
        "QA-14",
        "query_time_s_by_qa",
        "query_times_s",
    )
    if value is not None:
        if isinstance(value, Mapping) and (
            value.get("policy") or value.get("query_time_policy")
        ) == "uniform_in_legal_window":
            frames = _p8_window_frames(facts, "QA-14", value)
            if frames is None:
                _defer("sampling_window_invalid", "QA-14 legal frame window is invalid")
            return frames
        return [_resolve_query_frame_spec(facts, "QA-14", value, source="sampling")]
    if time_value is not None:
        if isinstance(time_value, Mapping) and (
            time_value.get("policy") or time_value.get("query_time_policy")
        ) == "uniform_in_legal_window":
            frames = _p8_window_frames(facts, "QA-14", time_value)
            if frames is None:
                _defer("sampling_window_invalid", "QA-14 legal frame window is invalid")
            return frames
        time_s = _resolve_query_time_spec(
            facts, "QA-14", time_value, source="sampling_time"
        )
        return [
            _resolve_query_frame_spec(
                facts,
                "QA-14",
                int(round(time_s * float(facts["time"]["frame_rate_hz"]))),
                source="sampling_time",
            )
        ]
    window = _sampling_value(
        facts,
        "QA-14",
        "legal_window_by_qa",
        "legal_windows",
        "query_windows",
    )
    if window is not None:
        frames = _p8_window_frames(facts, "QA-14", window)
        if frames is None:
            _defer("sampling_window_invalid", "QA-14 legal frame window is invalid")
        return frames
    return list(range(int(facts["time"]["frame_count"])))


def _p8_candidate_pool(
    facts: Mapping[str, Any],
    qa_id: str,
) -> list[dict[str, Any]]:
    """Enumerate cheap legal actor/event candidates before the expensive emit."""

    if qa_id == "QA-25":
        from avengine.qa.angular_questions import candidates
        return candidates(facts)
    actors = facts.get("actors")
    actor_ids = [
        str(actor_id)
        for actor_id, actor in actors.items()
        if isinstance(actor, Mapping)
    ] if isinstance(actors, Mapping) else []
    events = facts.get("events")
    event_rows = [
        event
        for event in events
        if isinstance(event, Mapping) and isinstance(event.get("actor_id"), str)
    ] if isinstance(events, Sequence) and not isinstance(events, (str, bytes)) else []
    values = _candidate_actor_values(facts)
    fixed = {"QA-03", "QA-22", "QA-23", "QA-24"}
    if qa_id in fixed:
        if qa_id == "QA-03" and len(event_rows) < 2:
            return []
        if qa_id == "QA-22" and len(actor_ids) < 2:
            return []
        if qa_id in {"QA-23", "QA-24"} and not event_rows:
            return []
        return [
            {
                "candidate_id": f"{qa_id}:semantic_fixed",
                "kind": "semantic_fixed",
                "candidate_values": values,
            }
        ]
    if qa_id == "QA-05":
        return [{"candidate_id": f"{qa_id}:pair:{a['event_id']}:{b['event_id']}",
                 "kind": "event_pair", "event_ids": [a["event_id"], b["event_id"]]}
                for index, a in enumerate(event_rows) for b in event_rows[index + 1:]]
    if qa_id == "QA-14":
        candidates = _appearance_candidates(facts)
        if len(candidates) < 2:
            return []
        result: list[dict[str, Any]] = []
        for frame in _p8_qa14_frames(facts):
            for first_index, (first_id, _first_actor, _first_appearance) in enumerate(candidates):
                for second_id, _second_actor, _second_appearance in candidates[first_index + 1:]:
                    try:
                        first_state = _require_visibility(facts, first_id, frame)
                        second_state = _require_visibility(facts, second_id, frame)
                        first_distance = _distance_at(facts, first_id, frame)
                        second_distance = _distance_at(facts, second_id, frame)
                    except _Deferred:
                        continue
                    if (
                        first_state.get("state") in VISIBLE_STATES
                        and second_state.get("state") in VISIBLE_STATES
                        and abs(first_distance - second_distance) >= 0.5
                    ):
                        result.append(
                            {
                                "candidate_id": (
                                    f"{qa_id}:pair:{first_id}:{second_id}:"
                                    f"frame:{frame}"
                                ),
                                "kind": "actor_pair",
                                "actor_ids": [first_id, second_id],
                                "actor_id": first_id,
                                "query_frame": frame,
                                "candidate_values": values,
                            }
                        )
        return result
    if qa_id in {"QA-07", "QA-09", "QA-10", "QA-11"}:
        return {
            "QA-07": _p8_entry_transition_candidates,
            "QA-09": _p8_reappearance_candidates,
            "QA-10": _p8_occlusion_candidates,
            "QA-11": _p8_partial_clear_candidates,
        }[qa_id](facts)
    if qa_id in {"QA-13", "QA-16", "QA-17"}:
        return _post_sound_candidates(facts, qa_id, event_rows)
    if qa_id == "QA-18":
        return _query_time_candidates(facts, qa_id)
    if qa_id in {"QA-19", "QA-21"}:
        return [{"candidate_id": f"{qa_id}:actor:{actor_id}", "kind": "actor",
                 "actor_id": actor_id, "candidate_values": values}
                for actor_id in actor_ids if any(e["actor_id"] == actor_id for e in event_rows)]
    actor_qas = {"QA-01"}
    if qa_id in actor_qas:
        return [
            {
                "candidate_id": f"{qa_id}:actor:{actor_id}",
                "kind": "actor",
                "actor_id": actor_id,
                "candidate_values": values,
            }
            for actor_id in actor_ids
        ]
    if qa_id == "QA-02":
        event_rows = [
            event
            for event in event_rows
            if isinstance(event.get("transcript"), str)
            and bool(event.get("transcript", "").strip())
        ]
    if qa_id == "QA-12":
        event_rows = [
            event
            for event in event_rows
            if (
                isinstance(event.get("transcript"), str)
                and bool(event.get("transcript", "").strip())
            )
            or _p8_actor_kind(
                facts["actors"].get(event.get("actor_id"), {})
            ) == "articulated_animal"
        ]
    if qa_id == "QA-21":
        event_rows = [
            event
            for event in event_rows
            if event.get("sound_class")
            and event.get("sound_class_explicit")
        ]
    if qa_id == "QA-06":
        # The pool asks the same predicate the emitter asks, so a candidate is
        # never enumerated on a looser rule than the one that judges it.
        event_rows = [
            event
            for event in event_rows
            if motion_state_during_audible_window(facts, event)["moving"] is not None
        ]
    if qa_id == "QA-15":
        legal_events = []
        for event in event_rows:
            window = audible_frame_window(facts, event)["frames"]
            trend = distance_trend_during_window(
                facts, str(event["actor_id"]), window
            )
            if trend["verdict"] is not None:
                legal_events.append(event)
        event_rows = legal_events
    return [
        {
            "candidate_id": (
                f"{qa_id}:event:{event.get('event_id', index)}:"
                f"{event.get('actor_id')}"
            ),
            "kind": "event",
            "event_id": event.get("event_id"),
            "actor_id": str(event["actor_id"]),
            "candidate_values": values,
        }
        for index, event in enumerate(event_rows)
    ]

def _p8_facts_for_candidate(
    facts: Mapping[str, Any],
    candidate: Mapping[str, Any],
    seed: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(facts))
    result["_generation_seed"] = str(seed)
    result["_p8_candidate"] = copy.deepcopy(dict(candidate))
    if candidate.get("kind") == "query_time":
        sampling = copy.deepcopy(result.get("sampling", {}))
        sampling.setdefault("query_time_s_by_qa", {})["QA-18"] = float(candidate["query_time_s"])
        result["sampling"] = sampling
    if candidate.get("kind") == "actor_pair" and candidate.get("query_frame") is not None:
        sampling = copy.deepcopy(result.get("sampling", {}))
        sampling.setdefault("query_frame_by_qa", {})["QA-14"] = int(
            candidate["query_frame"]
        )
        result["sampling"] = sampling
    actor_ids = candidate.get("actor_ids")
    if not isinstance(actor_ids, Sequence) or isinstance(actor_ids, (str, bytes)):
        actor_ids = [candidate.get("actor_id")]
    actor_ids = [
        actor_id for actor_id in actor_ids if actor_id is not None
    ]
    actors = result.get("actors")
    if isinstance(actors, Mapping) and actor_ids:
        selected = {
            actor_id: actors[actor_id]
            for actor_id in actor_ids
            if actor_id in actors
        }
        result["actors"] = {
            **selected,
            **{
                key: value
                for key, value in actors.items()
                if key not in selected
            },
        }
    actor_id = actor_ids[0] if actor_ids else None
    visibility = result.get("visibility")
    if isinstance(visibility, Mapping) and actor_ids:
        selected_visibility = {
            actor_id: visibility[actor_id]
            for actor_id in actor_ids
            if actor_id in visibility
        }
        result["visibility"] = {
            **selected_visibility,
            **{
                key: value
                for key, value in visibility.items()
                if key not in selected_visibility
            },
        }
    event_id = candidate.get("event_id")
    event_rows = result.get("events")
    if candidate.get("kind") == "post_event_query" and isinstance(event_rows, list):
        for event in event_rows:
            if event.get("event_id") == event_id:
                event["post_sound_query_frame"] = int(candidate["query_frame"])
    if event_id is not None and isinstance(event_rows, list):
        result["events"] = [
            *[
                event
                for event in event_rows
                if isinstance(event, Mapping)
                and event.get("event_id") == event_id
            ],
            *[
                event
                for event in event_rows
                if not (
                    isinstance(event, Mapping)
                    and event.get("event_id") == event_id
                )
            ],
        ]
    return result


def _p8_form_candidate_values(
    qa_id: str,
    item: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return per-entity values in each form's actual answer domain."""

    actor_ids = _p8_actor_ids(facts)
    open_values: dict[str, Any] = {actor_id: None for actor_id in actor_ids}
    mcq_values: dict[str, Any] = {actor_id: None for actor_id in actor_ids}
    forms = item.get("forms")
    mcq_form = forms.get("mcq") if isinstance(forms, Mapping) else None
    options = mcq_form.get("options", []) if isinstance(mcq_form, Mapping) else []
    option_values = {
        str(option.get("value"))
        for option in options
        if isinstance(option, Mapping) and option.get("value") is not None
    }
    has_mcq_form = isinstance(mcq_form, Mapping) and bool(mcq_form)
    evidence = item.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    candidate = facts.get("_p8_candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    selected_event_id = candidate.get("event_id") or _nested_evidence_value(
        evidence, "event_id"
    )
    target_actor_id = _nested_evidence_value(
        evidence,
        "target_actor_id",
        "actor_id",
        "candidate_actor_id",
    )
    query_frame = _nested_evidence_value(
        evidence,
        "query_frame",
        "frame",
        "entry_frame",
        "transition_frame",
        "final_frame",
    )
    if query_frame is None:
        query_frame = candidate.get("query_frame")
    if isinstance(query_frame, bool) or not isinstance(query_frame, int):
        query_frame = None
    final_frame = _nested_evidence_value(evidence, "final_frame")
    if isinstance(final_frame, bool) or not isinstance(final_frame, int):
        final_frame = int(facts["time"]["frame_count"]) - 1

    events = facts.get("events")
    event_rows = [
        event
        for event in events
        if isinstance(event, Mapping) and isinstance(event.get("actor_id"), str)
    ] if _is_sequence(events) else []

    anchor_event = next((row for row in event_rows
                         if row.get("event_id") == selected_event_id), None)

    def actor_events(actor_id: str) -> list[Mapping[str, Any]]:
        rows = [
            event for event in event_rows
            if str(event.get("actor_id")) == actor_id
        ]
        rows.sort(key=lambda event: (float(event.get("start_s", 0.0)), str(event.get("event_id", ""))))
        if selected_event_id is not None:
            rows = [
                *[
                    event for event in rows
                    if event.get("event_id") == selected_event_id
                ],
                *[
                    event for event in rows
                    if event.get("event_id") != selected_event_id
                ],
            ]
        return rows

    def set_value(actor_id: str, open_value: Any, mcq_value: Any = None) -> None:
        if actor_id not in open_values or open_value is None:
            return
        open_values[actor_id] = open_value
        value = open_value if mcq_value is None else mcq_value
        if (
            has_mcq_form
            and value is not None
            and (not option_values or str(value) in option_values)
        ):
            mcq_values[actor_id] = value

    def set_closed(actor_id: str, value: Any) -> None:
        set_value(actor_id, value, value)

    def visibility_state(actor_id: str, frame: int) -> str | None:
        try:
            return str(_state(facts, actor_id, frame).get("state"))
        except _Deferred:
            return None

    def event_motion(actor_id: str, event: Mapping[str, Any]) -> str | None:
        if _noticeable_motion_policy(facts) is not None:
            response = motion_state_during_audible_window(facts, {**event, "actor_id": actor_id})
            return None if response["moving"] is None else ("moving" if response["moving"] else "still")
        try:
            moving = _stable_motion_window(
                facts,
                actor_id,
                max(0, _event_frame(event, "start_frame")),
                min(
                    int(facts["time"]["frame_count"]),
                    _event_frame(event, "end_frame"),
                ),
            )
        except _Deferred:
            return None
        return "moving" if moving else "still"

    def event_distance_trend(actor_id: str, event: Mapping[str, Any]) -> str | None:
        try:
            start_frame = max(0, _event_frame(event, "start_frame"))
            end_frame = min(
                int(facts["time"]["frame_count"]) - 1,
                max(start_frame + 1, _event_frame(event, "end_frame") - 1),
            )
            delta = _distance_at(facts, actor_id, end_frame) - _distance_at(
                facts, actor_id, start_frame
            )
        except _Deferred:
            return None
        if abs(delta) < 0.2:
            return None
        return "nearer" if delta < 0 else "farther"

    def post_distance_trend(actor_id: str, event: Mapping[str, Any]) -> str | None:
        if query_frame is None:
            return None
        post_sound = evidence.get("post_sound")
        anchor_frame = (
            post_sound.get("anchor_end_frame")
            if isinstance(post_sound, Mapping)
            else _event_frame(event, "end_frame")
        )
        if isinstance(anchor_frame, bool) or not isinstance(anchor_frame, int):
            return None
        try:
            delta = _distance_at(facts, actor_id, anchor_frame) - _distance_at(
                facts, actor_id, query_frame
            )
        except _Deferred:
            return None
        delta = -delta
        if abs(delta) < 0.2:
            return None
        return "nearer" if delta < 0 else "farther"

    def post_motion(actor_id: str, event: Mapping[str, Any]) -> str | None:
        if query_frame is None:
            return None
        end_frame = min(query_frame, _event_frame(event, "end_frame"))
        if query_frame < end_frame:
            return None
        try:
            values = [
                _motion_at(facts, actor_id, frame)
                for frame in range(max(0, end_frame), query_frame + 1)
            ]
        except _Deferred:
            return None
        return "yes" if any(values) else "no"

    def entry_side_at_query(actor_id: str) -> str | None:
        for candidate_row in _p8_entry_transition_candidates(facts):
            if (candidate_row.get("actor_id") == actor_id
                    and candidate_row.get("query_frame") == query_frame):
                return str(candidate_row["entry_side"])
        return None

    def reappeared(actor_id: str) -> str | None:
        rows = _p8_ordered_visibility(facts, actor_id)
        full_frames = [
            int(row["frame_index"])
            for row in rows
            if isinstance(row.get("frame_index"), int)
            and row.get("state") == "fully_occluded"
        ]
        if not full_frames:
            return None
        visible_after = [
            int(row["frame_index"])
            for row in rows
            if isinstance(row.get("frame_index"), int)
            and row.get("state") in VISIBLE_STATES
            and any(full < int(row["frame_index"]) for full in full_frames)
        ]
        if visible_after:
            return "yes"
        return "no" if _visibility_is_complete(facts, actor_id) else None

    def partial_clear(actor_id: str) -> str | None:
        rows = _p8_ordered_visibility(facts, actor_id)
        partial = any(row.get("state") == "visible_occluded" for row in rows)
        if not partial:
            return None
        transition = any(
            previous.get("state") == "visible_occluded"
            and current.get("state") == "visible_clear"
            and isinstance(previous.get("frame_index"), int)
            and isinstance(current.get("frame_index"), int)
            and current["frame_index"] == previous["frame_index"] + 1
            for previous, current in zip(rows, rows[1:])
        )
        if transition:
            return "yes"
        return "no" if _visibility_is_complete(facts, actor_id) else None

    def occluder_at(actor_id: str) -> str | None:
        if query_frame is None:
            return None
        ids = _occluder_ids(facts, actor_id, query_frame)
        return ids[0] if len(ids) == 1 else None

    def transcript_value(actor_id: str) -> str | None:
        if anchor_event is None:
            return None
        ordinal = _statement_ordinal(facts, anchor_event)
        rows = sorted([row for row in actor_events(actor_id)
                       if isinstance(row.get("transcript"), str) and row["transcript"].strip()],
                      key=lambda row: (float(row["start_s"]), str(row["event_id"])))
        return str(rows[ordinal - 1]["transcript"]).strip() if len(rows) >= ordinal else None

    def sound_class_value(actor_id: str) -> str | None:
        values = {
            str(event["sound_class"])
            for event in actor_events(actor_id)
            if event.get("sound_class") and event.get("sound_class_explicit")
        }
        return next(iter(values)) if len(values) == 1 else None

    def time_value(actor_id: str) -> tuple[list[float] | None, str | None]:
        rows = actor_events(actor_id)
        if not rows:
            return None, None
        value = float(rows[0]["start_s"])
        bands = _time_bands(facts)
        index = next(
            (index for index, (lo, hi) in enumerate(bands) if lo <= value < hi),
            len(bands) - 1,
        )
        return [float(bands[index][0]), float(bands[index][1])], f"band_{index}"

    if qa_id == "QA-01":
        for actor_id in actor_ids:
            set_closed(actor_id, "yes" if actor_events(actor_id) else "no")
    elif qa_id == "QA-02":
        for actor_id, value in _candidate_actor_values(facts).items():
            set_closed(actor_id, value)
    elif qa_id == "QA-03":
        candidate_ids = evidence.get("candidate_actor_ids")
        candidate_ids = (
            [str(value) for value in candidate_ids]
            if _is_sequence(candidate_ids)
            else []
        )
        for actor_id in candidate_ids:
            set_closed(actor_id, actor_id)
    elif qa_id == "QA-04":
        if anchor_event is not None:
            for actor_id in actor_ids:
                try:
                    angle = _azimuth(facts, actor_id, _event_frame(anchor_event, "start_frame"))
                except _Deferred:
                    continue
                set_closed(actor_id, "right" if angle > 0 else "left")
    elif qa_id in {"QA-06", "QA-15"}:
        if anchor_event is not None:
            for actor_id in actor_ids:
                value = (event_motion(actor_id, anchor_event) if qa_id == "QA-06"
                         else event_distance_trend(actor_id, anchor_event))
                if value is not None:
                    set_closed(actor_id, value)
    elif qa_id == "QA-07":
        for actor_id in actor_ids:
            set_closed(actor_id, entry_side_at_query(actor_id))
    elif qa_id == "QA-08":
        if anchor_event is not None:
            for actor_id in actor_ids:
                set_closed(actor_id, visibility_state(actor_id, _event_frame(anchor_event, "start_frame")))
    elif qa_id == "QA-09":
        for actor_id in actor_ids:
            set_closed(actor_id, reappeared(actor_id))
    elif qa_id == "QA-10":
        for actor_id in actor_ids:
            set_closed(actor_id, occluder_at(actor_id))
    elif qa_id == "QA-11":
        for actor_id in actor_ids:
            set_closed(actor_id, partial_clear(actor_id))
    elif qa_id == "QA-12":
        for actor_id in actor_ids:
            set_closed(actor_id, transcript_value(actor_id))
    elif qa_id == "QA-13":
        for actor_id in actor_ids:
            if query_frame is None:
                continue
            try:
                angle = _azimuth(facts, actor_id, query_frame)
            except _Deferred:
                continue
            set_value(actor_id, angle, _fov_band(angle))
    elif qa_id == "QA-14":
        pair = evidence.get("distances_m")
        pair_ids = (
            {str(actor_id) for actor_id in pair}
            if isinstance(pair, Mapping)
            else set()
        )
        for actor_id in actor_ids:
            if actor_id in pair_ids:
                set_closed(actor_id, actor_id)
    elif qa_id == "QA-16":
        if anchor_event is not None:
            for actor_id in actor_ids:
                set_closed(actor_id, post_distance_trend(actor_id, anchor_event))
    elif qa_id == "QA-17":
        if anchor_event is not None:
            for actor_id in actor_ids:
                set_closed(actor_id, post_motion(actor_id, anchor_event))
    elif qa_id == "QA-18":
        active_ids = {
            str(value)
            for value in evidence.get("active_actor_ids", [])
            if value is not None
        }
        answer = (
            "multiple"
            if len(active_ids) > 1
            else next(iter(active_ids), "none")
        )
        for actor_id in actor_ids:
            set_closed(
                actor_id,
                answer if actor_id in active_ids else (
                    "multiple" if len(active_ids) > 1 else "none"
                ),
            )
    elif qa_id == "QA-19":
        for actor_id in actor_ids:
            value, band = time_value(actor_id)
            if value is not None:
                set_value(actor_id, value, band)
    elif qa_id == "QA-20":
        visible_ids = {
            str(value)
            for value in evidence.get("visible_candidate_actor_ids", [])
            if value is not None
        }
        for actor_id in actor_ids:
            if actor_id in visible_ids:
                set_closed(actor_id, actor_id)
    elif qa_id == "QA-21":
        for actor_id in actor_ids:
            set_closed(actor_id, sound_class_value(actor_id))
    elif qa_id == "QA-24":
        if event_rows:
            for actor_id in actor_ids:
                if actor_events(actor_id):
                    set_closed(actor_id, visibility_state(actor_id, final_frame))
    # QA-05, QA-22 and QA-23 have whole-episode answers, not an entity
    # counterfactual domain. Keep their entity maps empty.
    if qa_id in {"QA-05", "QA-22", "QA-23"}:
        return {"open": {}, "mcq": {}}
    return {"open": open_values, "mcq": mcq_values}


def _attach_p8_structure(
    item: dict[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    form_candidate_values = candidate.get("form_candidate_values")
    if isinstance(form_candidate_values, Mapping):
        open_values = form_candidate_values.get("open", {})
        mcq_values = form_candidate_values.get("mcq", {})
    else:
        open_values = candidate.get("candidate_values", {})
        mcq_values = {}
    if not isinstance(open_values, Mapping):
        open_values = {}
    if not isinstance(mcq_values, Mapping):
        mcq_values = {}
    gold_actor = candidate.get("gold_actor")
    open_baseline = structural_baselines(open_values, gold_actor)
    mcq_form = item.get("forms", {}).get("mcq", {})
    mcq_baseline = structural_baselines(mcq_values, gold_actor,
        answer_domain_size=len(mcq_form.get("options", [])) if mcq_form else None)
    structure = {
        "open": open_baseline,
        "mcq": mcq_baseline,
        "distractors_equal_gold_open": (
            distractors_equal_gold(open_values, str(gold_actor))
            if gold_actor is not None
            else False
        ),
        "majority_refusal_applied": False,
    }
    item["structure"] = structure
    item["candidate_id"] = candidate.get("candidate_id")
    item["candidate_value_multiplicity"] = {
        "open": open_baseline["candidate_value_multiplicity"],
        "mcq": mcq_baseline["candidate_value_multiplicity"],
    }
    item["gold_is_majority"] = {
        "open": open_baseline["gold_is_majority"],
        "mcq": mcq_baseline["gold_is_majority"],
    }
    item["gold_is_unique_minority"] = {
        "open": open_baseline["gold_is_unique_minority"],
        "mcq": mcq_baseline["gold_is_unique_minority"],
    }
    return item


_P8_BASE_GENERATORS = dict(_GENERATORS)


def _p8_candidates_for(qa_id: str):
    def candidates(facts: Mapping[str, Any]) -> list[dict[str, Any]]:
        return _p8_candidate_pool(facts, qa_id)
    candidates.__name__ = f"_candidates_{qa_id.lower().replace('-', '_')}"
    return candidates


_P8_DISTRACTOR_GATE_EXEMPT = {
    "QA-05",
    "QA-18",
    "QA-22",
    "QA-23",
}


def _p8_apply_distractor_gate(
    qa_id: str,
    item: dict[str, Any],
    candidate: MutableMapping[str, Any],
) -> None:
    """Defer only the form whose real actor values collapse onto its gold."""

    if qa_id in _P8_DISTRACTOR_GATE_EXEMPT:
        return
    gold_actor = candidate.get("gold_actor")
    if gold_actor is None:
        return
    values = candidate.get("form_candidate_values")
    if not isinstance(values, MutableMapping):
        return
    rejected: list[str] = []
    for form in ("open", "mcq"):
        form_values = values.get(form)
        if not isinstance(form_values, Mapping):
            continue
        if form not in item.get("forms", {}):
            continue
        # QA-01 negative rows have no target event by definition, so the
        # absence answer is a legal whole-clip negative even when other
        # entities also remain silent. Positive QA-01 rows use the normal
        # target/event separation check.
        if qa_id == "QA-01" and str(form_values.get(str(gold_actor))) == "no":
            continue
        if not distractors_equal_gold(
            {str(actor_id): value for actor_id, value in form_values.items()},
            str(gold_actor),
        ):
            continue
        reason = {
            "status": "deferred",
            "code": "distractors_equal_gold",
            "detail": "all available real distractors have the same answer value as gold",
            "form": form,
            "gold_actor": str(gold_actor),
        }
        item.setdefault("form_status", {})[form] = reason
        item.setdefault("forms", {}).pop(form, None)
        item.setdefault("model_input", {}).pop(form, None)
        values[form] = {}
        rejected.append(form)
    if rejected and not item.get("forms"):
        _defer(
            "distractors_equal_gold",
            f"{qa_id} has no answer form with a distinct real distractor value",
            forms=rejected,
        )


def _p8_emit_for(qa_id: str):
    def emit(
        facts: Mapping[str, Any],
        candidate: Mapping[str, Any],
        seed: str,
    ) -> dict[str, Any]:
        if qa_id == "QA-25":
            from avengine.qa.angular_questions import emit as emit_bearing
            return emit_bearing(facts, candidate, seed)
        candidate_facts = _p8_facts_for_candidate(facts, candidate, seed)
        applicability = _p8_applicability_reason(
            candidate_facts, qa_id, candidate
        )
        if applicability is not None:
            code, detail = applicability
            raise _Deferred(code, detail)
        item = _P8_BASE_GENERATORS[qa_id](candidate_facts, seed)
        if not _p8_candidate_matches_item(item, candidate):
            raise _Deferred(
                "candidate_not_emitted",
                "the selected candidate did not produce the emitted target/event",
            )
        metadata_candidate = _p8_metadata_candidate(qa_id, item, candidate)
        metadata_candidate["form_candidate_values"] = _p8_form_candidate_values(
            qa_id, item, candidate_facts
        )
        if not _ordinary_observation_questions(candidate_facts):
            _p8_apply_distractor_gate(qa_id, item, metadata_candidate)
        return _tag_question_tolerance(
            _attach_p8_structure(item, metadata_candidate), candidate_facts, qa_id)
    emit.__name__ = f"_emit_{qa_id.lower().replace('-', '_')}"
    return emit


_P8_CANDIDATES = {
    qa_id: _p8_candidates_for(qa_id)
    for qa_id in _P8_BASE_GENERATORS
}
_P8_EMITTERS = {
    qa_id: _p8_emit_for(qa_id)
    for qa_id in _P8_BASE_GENERATORS
}
for _qa_id, _candidate_fn in _P8_CANDIDATES.items():
    globals()[_candidate_fn.__name__] = _candidate_fn
for _qa_id, _emit_fn in _P8_EMITTERS.items():
    globals()[_emit_fn.__name__] = _emit_fn


# Evidence-side states reuse the shared four-word vocabulary. The catalog can
# only speak for what the readbacks prove; whether the sampler can point at a
# scene is the planning side of the same question and lives in
# ``avengine.qa.generation_conditions``.
EVIDENCE_STATE_AVAILABLE = "available"
EVIDENCE_STATE_NOT_APPLICABLE = "not_applicable_by_definition"
EVIDENCE_STATE_MISSING = "evidence_missing_or_unsampled"


def _evidence_state(emitted_count: int, codes: Sequence[str]) -> str:
    if emitted_count:
        return EVIDENCE_STATE_AVAILABLE
    distinct = {str(code) for code in codes if code}
    if distinct and distinct == {EVIDENCE_STATE_NOT_APPLICABLE}:
        return EVIDENCE_STATE_NOT_APPLICABLE
    return EVIDENCE_STATE_MISSING


def _form_coverage(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-form validity and the denominator a scorer divides by.

    MCQ and open are accepted or refused independently, so a missing answer
    rate has to be counted against the items that actually offered that form.
    Merging the two produces a denominator no form ever had.
    """

    coverage: dict[str, Any] = {}
    for form in ("mcq", "open"):
        passed = 0
        deferred: dict[str, int] = {}
        for item in items:
            status = item.get("form_status")
            row = status.get(form) if isinstance(status, Mapping) else None
            if not isinstance(row, Mapping):
                continue
            if row.get("status") == "pass":
                passed += 1
            else:
                code = str(row.get("code", "deferred_without_code"))
                deferred[code] = deferred.get(code, 0) + 1
        coverage[form] = {
            "answerable_item_count": passed,
            "deferred_item_count": sum(deferred.values()),
            "deferred_codes": dict(sorted(deferred.items())),
            "scoring_denominator": passed,
        }
    coverage["item_count"] = len(items)
    return coverage


# How a published answer token maps onto a declared key branch, for the types
# whose answer vocabulary is not the branch vocabulary. This is metadata only:
# the published truth, the options and the scoring keep the answer token.
#
# QA-20 publishes an actor id when a visible candidate made the sound and
# ``none_of_visible`` when none of them did (see ``_generate_qa_20``); the
# branch owner calls that second case ``none_of_them``. Without this map a
# world that really produced the none_of_them question reported
# ``branches_seen=['visible_candidate']``, so the branch looked unmet while the
# question existed.
#
# ``avengine.qa.batch_delivery.BRANCH_OBSERVATION_RULES`` carries the same
# statement for the delivery side. The two are asserted to agree in
# ``tests/unit/test_qa_unified_branch_reporting.py``; this module is the lower
# layer and must not import that one.
ANSWER_TOKEN_BRANCH_MAP: dict[str, dict[str, str]] = {
    "QA-20": {"none_of_visible": "none_of_them", "none_of_them": "none_of_them"},
}
ANSWER_TOKEN_BRANCH_DEFAULT: dict[str, str] = {"QA-20": "visible_candidate"}


def _emitted_branch(qa_id: str, item: Mapping[str, Any]) -> str | None:
    """Read which key branch an emitted item landed on.

    The branch is not always the answer value. QA-25 branches by modality
    subset and QA-20 branches by whether any candidate was named at all, so
    each shape is read from the field that actually carries it rather than
    from ``truth.value`` for every type. Where the answer vocabulary differs
    from the branch vocabulary, ``ANSWER_TOKEN_BRANCH_MAP`` names the mapping
    instead of the answer token being compared to a branch name it never uses.
    """

    if qa_id == "QA-25":
        subset = item.get("angle_subset")
        return str(subset) if subset is not None else None
    truth = item.get("truth")
    value = truth.get("value") if isinstance(truth, Mapping) else None
    if value is None:
        return None
    mapped = ANSWER_TOKEN_BRANCH_MAP.get(qa_id)
    if mapped is not None:
        return mapped.get(str(value), ANSWER_TOKEN_BRANCH_DEFAULT[qa_id])
    return str(value)


def _branch_authority_branches(qa_id: str) -> tuple[tuple[str, ...], str]:
    """Ask the branch owner, so the branch list has one definition."""

    try:
        from avengine.qa.generation_conditions import branches_for
    except Exception:  # noqa: BLE001 - report the absence, never invent a list
        return (), "unavailable"
    try:
        return tuple(branches_for(qa_id)), "avengine.qa.generation_conditions.branches_for"
    except Exception:  # noqa: BLE001
        return (), "unavailable"


def _qa_targets_for(facts: Mapping[str, Any], qa_id: str) -> list[Mapping[str, Any]]:
    sampling = facts.get("sampling") or {}
    return [target for target in sampling.get("qa_targets", ())
            if isinstance(target, Mapping) and _canonical_qa_id(target.get("qa_id")) == qa_id]


def _qa_target_actor_matches(target: Mapping[str, Any], value: Mapping[str, Any]) -> bool:
    ids = target.get("target_actor_ids", target.get("target_instance_ids")) or ()
    evidence = value.get("evidence") or {}
    actor = value.get("actor_id") or evidence.get("actor_id") or evidence.get("target_actor_id")
    # Clip-level candidates do not carry a single primary actor.
    return not ids or actor is None or str(actor) in {str(item) for item in ids}


def _qa_target_item_matches(target: Mapping[str, Any], qa_id: str,
                            item: Mapping[str, Any], facts: Mapping[str, Any],
                            *, accept_observed: bool = False) -> bool:
    if not _qa_target_actor_matches(target, item):
        return False
    branch = target.get("branch")
    if not branch:
        return True
    observed = _emitted_branch(qa_id, item)
    if observed == str(branch):
        return True
    if accept_observed:
        policy = (facts.get("sampling") or {}).get("acceptance_policy") or {}
        allowed = (policy.get("accept_observed_branches") or {}).get(qa_id, ())
        return observed in allowed
    return False


def generate_unified_questions(
    raw_or_facts: Mapping[str, Any],
    *,
    qa_ids: Sequence[str] | None = None,
    seed: str = "avengine-qa-20260906",
    items_per_type: int = 1,
    include_angle_followups: bool = True,
) -> dict[str, Any]:
    """Enumerate legal candidates, sample without replacement, then emit."""

    if not isinstance(raw_or_facts, Mapping):
        raise UnifiedQAError("episode input must be an object")
    if isinstance(items_per_type, bool) or items_per_type <= 0:
        raise UnifiedQAError("items_per_type must be a positive integer")
    facts = (
        _restore_normalized_frame_keys(raw_or_facts)
        if raw_or_facts.get("schema") == UNIFIED_FACT_SCHEMA
        else normalize_episode_bundle(raw_or_facts)
    )
    facts = with_derived_sound_class_answer_domain(facts)
    requested = (
        [_canonical_qa_id(value) for value in qa_ids]
        if qa_ids is not None
        else [item["qa_id"] for item in CATALOG]
    )
    if len(requested) != len(set(requested)):
        raise UnifiedQAError("qa_ids must be unique")
    items: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    item_groups: dict[str, list[dict[str, Any]]] = {}
    deferred_groups: dict[str, list[dict[str, Any]]] = {}
    # One row per attempted candidate. A type that emits one question still
    # rejected the others for their own reasons, and those reasons are what a
    # producer needs: "no publishable whole-second interval" and "every real
    # distractor shares the gold answer" call for different scenes.
    candidate_attempts: list[dict[str, Any]] = []
    base_facts = facts
    from avengine.qa.visibility_interpretation import prepare_facts_for_qa
    for qa_id in requested:
        facts = prepare_facts_for_qa(base_facts, qa_id)
        try:
            candidates = _P8_CANDIDATES[qa_id](facts)
        except _Deferred as error:
            row = {"qa_id": qa_id, "status": "deferred", "code": error.code,
                   "detail": error.detail, **error.extra, "requirements": get_requirements(qa_id)}
            deferred.append(row)
            deferred_groups.setdefault(qa_id, []).append(row)
            continue
        quota = (1 if qa_id in {"QA-03", "QA-22", "QA-23", "QA-24"}
                 else int(items_per_type) * (3 if qa_id == "QA-25" else 1))
        if not candidates:
            try:
                item = _P8_BASE_GENERATORS[qa_id](facts, seed)
            except _Deferred as error:
                row = {
                    "qa_id": qa_id,
                    "status": "deferred",
                    "code": error.code,
                    "detail": error.detail,
                    **error.extra,
                    "requirements": get_requirements(qa_id),
                }
                deferred.append(row)
                deferred_groups.setdefault(qa_id, []).append(row)
            else:
                row = {"qa_id": qa_id, "status": "deferred", "code": "insufficient_candidates",
                       "detail": "no legal candidate can be enumerated", "requirements": get_requirements(qa_id)}
                deferred.append(row)
                deferred_groups.setdefault(qa_id, []).append(row)
            continue
        rng = random.Random(f"{seed}\\0{qa_id}")
        order = list(candidates)
        rng.shuffle(order)
        targets = _qa_targets_for(facts, qa_id)
        if targets:
            order.sort(key=lambda candidate: not any(
                _qa_target_actor_matches(target, candidate) for target in targets))
            quota = max(quota, sum(int(target.get("items", 1)) for target in targets))
        side_questions: list[tuple[dict[str, Any], dict[str, Any]]] = []
        emitted: list[dict[str, Any]] = []
        last_error: _Deferred | None = None
        for candidate in order:
            if qa_id == "QA-25" and sum(item.get("angle_subset") == candidate["subset"] for item in emitted) >= items_per_type:
                candidate_attempts.append({
                    "qa_id": qa_id,
                    "candidate_id": candidate.get("candidate_id"),
                    "status": "not_attempted",
                    "code": "subset_quota_already_met",
                    "detail": "this angle subset already reached its requested count",
                    "angle_subset": candidate.get("subset"),
                })
                continue
            attempt: dict[str, Any] = {
                "qa_id": qa_id,
                "candidate_id": candidate.get("candidate_id"),
                "actor_id": candidate.get("actor_id"),
                "event_id": candidate.get("event_id"),
            }
            if candidate.get("subset") is not None:
                attempt["angle_subset"] = candidate.get("subset")
            try:
                item = _P8_EMITTERS[qa_id](facts, candidate, seed)
            except _Deferred as error:
                last_error = error
                attempt.update({
                    "status": "candidate_rejected",
                    "code": error.code,
                    "detail": error.detail,
                    **{
                        key: value
                        for key, value in error.extra.items()
                        if key not in {"qa_id", "candidate_id", "status", "code", "detail"}
                    },
                })
                candidate_attempts.append(attempt)
                continue
            if any(previous["question_id"] == item["question_id"] for previous in emitted):
                attempt.update({
                    "status": "candidate_rejected",
                    "code": "duplicate_question_id",
                    "detail": "another candidate already produced this question",
                    "question_id": item["question_id"],
                })
                candidate_attempts.append(attempt)
                continue
            if targets and not any(_qa_target_item_matches(target, qa_id, item, facts)
                                   for target in targets):
                attempt.update({"status": "not_selected", "code": "valid_side_question",
                                "question_id": item["question_id"],
                                "truth_value": copy.deepcopy(item.get("truth", {}).get("value")),
                                "forms": sorted(item.get("forms", {}))})
                candidate_attempts.append(attempt)
                side_questions.append((item, attempt))
                continue
            attempt.update({
                "status": "pass",
                "question_id": item["question_id"],
                "truth_value": copy.deepcopy(item.get("truth", {}).get("value")),
                "forms": sorted(item.get("forms", {})),
            })
            candidate_attempts.append(attempt)
            emitted.append(item)
            items.append(item)
            if len(emitted) >= quota:
                break
        # Preserve valid questions when the named target is unavailable. The
        # explicit target result below keeps them from disguising a missed goal.
        if len(emitted) < quota and side_questions:
            side_questions.sort(key=lambda pair: not any(
                _qa_target_item_matches(target, qa_id, pair[0], facts, accept_observed=True)
                for target in targets))
            for item, attempt in side_questions:
                if len(emitted) >= quota:
                    break
                if any(previous["question_id"] == item["question_id"] for previous in emitted):
                    continue
                attempt.update(status="pass", target_relation="valid_side_question")
                attempt.pop("code", None)
                emitted.append(item)
                items.append(item)
        if emitted:
            item_groups[qa_id] = emitted
        elif last_error is not None:
            row = {
                "qa_id": qa_id,
                "status": "deferred",
                "code": last_error.code,
                "detail": last_error.detail,
                **last_error.extra,
                "requirements": get_requirements(qa_id),
            }
            deferred.append(row)
            deferred_groups.setdefault(qa_id, []).append(row)
        else:
            try:
                _P8_BASE_GENERATORS[qa_id](facts, seed)
            except _Deferred as error:
                row = {
                    "qa_id": qa_id,
                    "status": "deferred",
                    "code": error.code,
                    "detail": error.detail,
                    **error.extra,
                    "requirements": get_requirements(qa_id),
                }
                deferred.append(row)
                deferred_groups.setdefault(qa_id, []).append(row)
    facts = base_facts
    rejection_codes_by_qa: dict[str, dict[str, int]] = {}
    for attempt in candidate_attempts:
        if attempt.get("status") != "candidate_rejected":
            continue
        counts = rejection_codes_by_qa.setdefault(str(attempt["qa_id"]), {})
        code = str(attempt.get("code", "candidate_rejected_without_code"))
        counts[code] = counts.get(code, 0) + 1
    unmet_quota = {}
    for qa_id in requested:
        quota = (1 if qa_id in {"QA-03", "QA-22", "QA-23", "QA-24"}
                 else int(items_per_type) * (3 if qa_id == "QA-25" else 1))
        available = len(item_groups.get(qa_id, []))
        if available < quota:
            unmet_quota[qa_id] = {"requested": quota, "valid": available, "missing": quota - available,
                                 "code": "insufficient_candidates",
                                 "rejection_codes": dict(sorted(
                                     rejection_codes_by_qa.get(qa_id, {}).items()))}
            if available:
                deferred_groups.setdefault(qa_id, []).append({"qa_id": qa_id, "status": "insufficient_candidates",
                    "code": "insufficient_candidates", **unmet_quota[qa_id]})
    coverage: list[dict[str, Any]] = []
    coverage_by_qa: dict[str, list[dict[str, Any]]] = {}
    for qa_id in requested:
        records: list[dict[str, Any]] = []
        for item in item_groups.get(qa_id, []):
            record = {
                "qa_id": qa_id,
                "status": "pass",
                "question_id": item["question_id"],
                "candidate_id": item.get("candidate_id"),
                "requirements": get_requirements(qa_id),
            }
            records.append(record)
            coverage.append(record)
        for row in deferred_groups.get(qa_id, []):
            records.append(dict(row))
            coverage.append(dict(row))
        # Rejected candidates carry their own status so existing consumers that
        # select "pass" or "deferred" rows keep the counts they had.
        for attempt in candidate_attempts:
            if attempt.get("qa_id") != qa_id or attempt.get("status") not in {
                "candidate_rejected", "not_attempted"
            }:
                continue
            row = dict(attempt)
            row["requirements"] = get_requirements(qa_id)
            records.append(row)
            coverage.append(dict(row))
        coverage_by_qa[qa_id] = records
    from avengine.qa.angular_questions import followups
    angle_followups, angle_deferred = followups(facts, items, seed) if include_angle_followups else ([], [])
    angle_subset_reasons: dict[str, Any] = {}
    if "QA-25" in requested:
        from avengine.qa.angular_questions import subset_diagnostics
        try:
            angle_subset_reasons = subset_diagnostics(facts)
        except _Deferred as error:
            angle_subset_reasons = {
                subset: {"candidate_count": 0, "code": error.code,
                         "detail": error.detail}
                for subset in ("A", "V", "AV")
            }
    branch_state_by_qa: dict[str, Any] = {}
    for qa_id in requested:
        expected, authority = _branch_authority_branches(qa_id)
        observed = {
            branch
            for branch in (
                _emitted_branch(qa_id, item)
                for item in item_groups.get(qa_id, [])
            )
            if branch is not None
        }
        # A branch list exists to be checked against. A value outside it is
        # reported as unmapped, and a type the branch owner declares no
        # branches for reports none rather than promoting its answer values
        # into a branch list nobody defined.
        answer_values = sorted(observed)
        seen = sorted(observed & set(expected))
        unmapped = sorted(observed - set(expected)) if expected else []
        codes = sorted(rejection_codes_by_qa.get(qa_id, {}))
        deferred_codes = sorted({
            str(row.get("code"))
            for row in deferred_groups.get(qa_id, [])
            if row.get("code")
        })
        branch_state_by_qa[qa_id] = {
            "branches_expected": list(expected),
            "branch_authority": authority,
            "branches_seen": seen,
            "branches_missing": [
                branch for branch in expected if branch not in set(seen)
            ],
            "branch_values_unmapped": unmapped,
            "answer_values_seen": answer_values,
            "emitted_item_count": len(item_groups.get(qa_id, [])),
            "evidence_state": _evidence_state(
                len(item_groups.get(qa_id, [])), codes + deferred_codes
            ),
            "evidence_state_authority": "unified_catalog_evidence",
            "planning_state_authority": (
                "avengine.qa.generation_conditions.compile_generation_conditions"
            ),
            "rejection_codes": dict(sorted(
                rejection_codes_by_qa.get(qa_id, {}).items())),
            "deferred_codes": deferred_codes,
        }
    qa_target_results = []
    for qa_id in requested:
        for target in _qa_targets_for(base_facts, qa_id):
            exact = [item["question_id"] for item in item_groups.get(qa_id, ())
                     if _qa_target_item_matches(target, qa_id, item, base_facts)]
            accepted = [item["question_id"] for item in item_groups.get(qa_id, ())
                        if _qa_target_item_matches(target, qa_id, item, base_facts,
                                                   accept_observed=True)]
            count = int(target.get("items", 1))
            qa_target_results.append({"qa_id": qa_id,
                "target_actor_ids": list(target.get("target_actor_ids", target.get("target_instance_ids")) or ()),
                "requested_branch": target.get("branch"), "requested_items": count,
                "requested_branch_question_ids": exact,
                "policy_accepted_target_question_ids": accepted,
                "status": "met" if len(accepted) >= count else "unmet",
                "claim_boundary": "Emitted target questions under the existing policy. This does not measure predicted-frame agreement."})
    return {
        "schema": UNIFIED_OUTPUT_SCHEMA,
        "qa_target_results": qa_target_results,
        "status": "research_candidate",
        "qualification_claim": False,
        "catalog_version": CATALOG_VERSION,
        "episode_id": facts["episode_id"],
        "seed": seed,
        "items_per_type": int(items_per_type),
        "unmet_quota_by_qa": unmet_quota,
        "coverage_summary": {"requested_type_count": len(requested), "covered_type_count": len(item_groups),
                             "valid_item_count": len(items), "unmet_item_count": sum(x["missing"] for x in unmet_quota.values())},
        "candidate_counts": {
            qa_id: _safe_candidate_count(facts, qa_id)
            for qa_id in requested
        },
        "input_facts": facts,
        "counts": {
            "requested": len(requested),
            "valid": len(items),
            "deferred": len(deferred),
        },
        "coverage": coverage,
        "coverage_by_qa": coverage_by_qa,
        "candidate_attempts": candidate_attempts,
        "rejection_codes_by_qa": {
            qa_id: dict(sorted(codes.items()))
            for qa_id, codes in sorted(rejection_codes_by_qa.items())
        },
        "public_time_precision": _time_display_precision(facts),
        "form_coverage": {
            "main": _form_coverage(items),
            "angle_followup": _form_coverage(angle_followups),
        },
        "branch_state_by_qa": branch_state_by_qa,
        "items": items,
        "angle_followups": angle_followups,
        "angle_followup_deferred": angle_deferred,
        "angle_followup_counts": {"valid": len(angle_followups), "deferred": len(angle_deferred)},
        "angle_subset_coverage": {
            subset: {
                "requested": int(items_per_type),
                "valid": sum(
                    item.get("angle_subset") == subset
                    for item in item_groups.get("QA-25", [])
                ),
                # A subset that produced nothing says why here. Without it a
                # zero looks the same whether the predicate never fired or
                # the episode simply has no such scene.
                **{
                    key: value
                    for key, value in angle_subset_reasons.get(subset, {}).items()
                    if key != "candidate_count"
                },
                "candidate_count": angle_subset_reasons.get(subset, {}).get(
                    "candidate_count"
                ),
            }
            for subset in ("A", "V", "AV")
        } if "QA-25" in requested else {},
        "deferred": deferred,
        "actual_evidence_summary": {
            "actor_count": len(facts.get("actors", {})),
            "event_count": len(facts.get("events", [])),
            "bound_event_count": sum(
                1
                for event in facts.get("events", [])
                if isinstance(event, Mapping)
                and isinstance(event.get("actor_id"), str)
            ),
            "unresolved_event_ids": list(
                facts.get("input_summary", {}).get("unresolved_event_ids", [])
            ),
            "reviewed_appearance_actor_count": len(
                facts.get("appearance_review", {})
            ),
            "pixel_visibility_actor_count": len(facts.get("visibility", {})),
            "audio_validation_status": facts.get("audio", {}).get("status"),
        },
        "claim_boundary": (
            "Rows are deterministic research candidates derived from native "
            "readbacks. They are not model outcomes, formal admission or "
            "modality-necessity certificates."
        ),
    }


generate_questions = generate_unified_questions




_FOV_HALF_DEG = 40.44
_FOV_BAND_BOUNDARIES_DEG = (-40.44, -13.5, 13.5, 40.44)


def _fov_band(angle: float) -> str | None:
    value = float(angle)
    if value < _FOV_HALF_DEG * -1.0 or value > _FOV_HALF_DEG:
        return None
    if value < _FOV_BAND_BOUNDARIES_DEG[1]:
        return "fov_band_0"
    if value < _FOV_BAND_BOUNDARIES_DEG[2]:
        return "fov_band_1"
    return "fov_band_2"


def _fov_band_options() -> list[dict[str, Any]]:
    labels = (
        ("fov_band_0", "in-view left band [-40.44°, -13.5°)", "视野内左带[-40.44°，-13.5°)"),
        ("fov_band_1", "in-view center band [-13.5°, 13.5°)", "视野内中带[-13.5°，13.5°)"),
        ("fov_band_2", "in-view right band [13.5°, 40.44°]", "视野内右带[13.5°，40.44°]"),
    )
    return [
        {
            "value": value,
            "label_en": label,
            "label_zh": label_zh,
            "allow_value": False,
        }
        for value, label, label_zh in labels
    ]


def _p8_candidate_matches_item(
    item: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    evidence = item.get("evidence")
    if not isinstance(evidence, Mapping):
        return True
    candidate_actor = candidate.get("actor_id")
    actual_actor = _nested_evidence_value(
        evidence,
        "target_actor_id",
        "actor_id",
        "candidate_actor_id",
    )
    if candidate_actor is not None and candidate.get("kind") != "actor_pair":
        if actual_actor is not None and str(actual_actor) != str(candidate_actor):
            return False
    if candidate.get("kind") == "event_pair":
        if set(evidence.get("event_ids", [])) != set(candidate.get("event_ids", [])):
            return False
    if candidate.get("query_frame") is not None:
        actual_frame = _nested_evidence_value(evidence, "query_frame", "frame")
        if actual_frame != candidate.get("query_frame"):
            return False
    candidate_event = candidate.get("event_id")
    if candidate_event is not None:
        actual_event = _nested_evidence_value(evidence, "event_id")
        if actual_event is None:
            event_ids = _nested_evidence_value(evidence, "event_ids")
            if isinstance(event_ids, Sequence) and not isinstance(
                event_ids, (str, bytes)
            ):
                if candidate_event not in event_ids:
                    return False
        elif str(actual_event) != str(candidate_event):
            return False
    if candidate.get("kind") == "actor_pair":
        distances = evidence.get("distances_m")
        pair = candidate.get("actor_ids")
        if isinstance(distances, Mapping) and isinstance(pair, Sequence):
            return set(str(value) for value in distances) == set(
                str(value) for value in pair
            )
    return True

def _p8_metadata_candidate(
    qa_id: str,
    item: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = item.get("evidence")
    if not isinstance(evidence, Mapping):
        return dict(candidate)
    actual_target = _nested_evidence_value(
        evidence,
        "target_actor_id",
        "actor_id",
        "candidate_actor_id",
    )
    actual_event = _nested_evidence_value(evidence, "event_id")
    if actual_event is None:
        nested_event = _nested_evidence_value(evidence, "event")
        if isinstance(nested_event, Mapping):
            actual_event = nested_event.get("event_id")
    actual_frame = _nested_evidence_value(evidence, "query_frame", "frame")
    if actual_target is None and qa_id == "QA-14":
        truth = item.get("truth")
        if isinstance(truth, Mapping):
            value = truth.get("value")
            if isinstance(value, str):
                actual_target = value
    if actual_target is None and actual_event is None and actual_frame is None:
        return dict(candidate)
    metadata_candidate = dict(candidate)
    parts = [qa_id]
    if actual_target is not None:
        parts.append(f"target:{actual_target}")
        metadata_candidate["gold_actor"] = str(actual_target)
    if candidate.get("kind") == "actor_pair":
        pair = candidate.get("actor_ids")
        if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)):
            parts.append("pair:" + ",".join(str(value) for value in pair))
    if actual_event is not None:
        parts.append(f"event:{actual_event}")
    if actual_frame is not None:
        parts.append(f"frame:{actual_frame}")
    metadata_candidate["candidate_id"] = ":".join(parts)
    return metadata_candidate

def _p8_actor_kind(actor: Mapping[str, Any]) -> str:
    raw = " ".join(
        str(actor.get(key) or "")
        for key in ("entity_class", "source_class", "class", "species_id", "asset_type")
    ).casefold()
    if any(token in raw for token in ("rigid", "static", "device", "object")):
        return "rigid_static_object"
    if any(token in raw for token in ("animal", "dog", "cat", "beagle")):
        return "articulated_animal"
    if any(token in raw for token in ("human", "person", "adult")):
        return "articulated_human"
    return "unknown"


def _p8_applicability_reason(
    facts: Mapping[str, Any],
    qa_id: str,
    candidate: Mapping[str, Any],
) -> tuple[str, str] | None:
    actor_id = candidate.get("actor_id")
    actors = facts.get("actors")
    actor = actors.get(actor_id) if isinstance(actors, Mapping) else None
    if not isinstance(actor, Mapping):
        return None
    kind = _p8_actor_kind(actor)
    if qa_id == "QA-12" and kind == "articulated_animal":
        return (
            "not_applicable_by_definition",
            "animal vocalizations do not have a transcript question target",
        )
    if qa_id in {"QA-06", "QA-15", "QA-16", "QA-17"} and kind == "rigid_static_object":
        return (
            "not_applicable_by_definition",
            "a static device is not a movement-question target",
        )
    if qa_id == "QA-07" and kind == "rigid_static_object":
        # Under a fixed camera a device that cannot move itself can never
        # cross into view, so this is a semantic mismatch and not a gap in
        # the evidence.
        return (
            "not_applicable_by_definition",
            "a static device cannot enter the frame under a fixed camera",
        )
    return None
__all__ = list(dict.fromkeys([
    *__all__,
    "structural_baselines",
    "distractors_equal_gold",
    "resolve_query_frame",
    *[
        f"_candidates_qa_{index:02d}"
        for index in range(1, 26)
    ],
    *[
        f"_emit_qa_{index:02d}"
        for index in range(1, 26)
    ],
]))

resolve_query_frame = _resolve_query_frame_spec



def _qa18_frame_activity(
    facts: Mapping[str, Any],
    frame: int,
) -> dict[str, Any]:
    """Return the measured activity set and public branch for one frame."""

    active = _active_at(facts, int(frame), require_source_activity=True)
    event_ids = sorted(str(event.get("event_id")) for event in active)
    actor_ids = sorted(
        {
            str(event.get("actor_id"))
            for event in active
            if event.get("actor_id") is not None
        }
    )
    return {
        "activity_class": (
            "empty"
            if not actor_ids
            else "active"
            if len(actor_ids) == 1
            else "multiple"
        ),
        "active_event_ids": event_ids,
        "active_actor_ids": actor_ids,
        "active_event_count": len(event_ids),
        "active_actor_count": len(actor_ids),
    }


def _qa18_stable_activity_windows(
    facts: Mapping[str, Any],
    legal_windows: Sequence[Sequence[int]],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]]]:
    """Group legal frames by a stable measured active-event set."""

    frame_data: dict[int, dict[str, Any]] = {}
    stable_by_frame: dict[int, list[int]] = {}
    for start, end in _window_bounds_list(legal_windows, qa_id="QA-18"):
        run_start: int | None = None
        run_signature: tuple[str, ...] | None = None
        previous: int | None = None
        for frame in range(int(start), int(end)):
            info = _qa18_frame_activity(facts, frame)
            frame_data[frame] = info
            signature = tuple(info["active_event_ids"])
            if (
                run_start is None
                or previous is None
                or frame != previous + 1
                or signature != run_signature
            ):
                if run_start is not None and previous is not None:
                    run = [run_start, previous + 1]
                    if run[1] - run[0] >= 2:
                        for member in range(run[0], run[1]):
                            stable_by_frame[member] = list(run)
                run_start = frame
                run_signature = signature
            previous = frame
        if run_start is not None and previous is not None:
            run = [run_start, previous + 1]
            if run[1] - run[0] >= 2:
                for member in range(run[0], run[1]):
                    stable_by_frame[member] = list(run)
    return frame_data, stable_by_frame


def _qa18_legal_windows(
    facts: Mapping[str, Any],
    requested: Any,
) -> tuple[list[list[int]] | None, str | None]:
    """Intersect a caller window with native activity/tail legal frames."""

    legal = _derived_legal_query_windows(facts, "QA-18")
    if legal is None:
        return None, None
    if requested is None:
        return copy.deepcopy(legal), _derived_query_window_authority("QA-18")
    allowed = _window_bounds_list(requested, qa_id="QA-18")
    count = int(facts["time"]["frame_count"])
    if any(start < 0 or start >= end or end > count for start, end in allowed):
        _defer(
            "sampling_window_invalid",
            "QA-18 legal frame windows must be inside the frame clock",
        )
    allowed_frames = {
        frame for start, end in allowed for frame in range(start, end)
    }
    legal_frames = {
        frame for start, end in legal for frame in range(start, end)
    }
    return (
        _compress_frame_windows(sorted(allowed_frames & legal_frames)),
        "caller_declared_sampling_window",
    )


def _qa18_wet_tail_intervals(
    facts: Mapping[str, Any],
) -> list[tuple[float, float, str | None]] | None:
    audio = facts.get("audio")
    tails = audio.get("wet_tail_intervals") if isinstance(audio, Mapping) else None
    if not _is_sequence(tails) or not tails:
        return None
    result: list[tuple[float, float, str | None]] = []
    for interval in tails:
        if not isinstance(interval, Mapping):
            return None
        try:
            start = float(interval["start_s"])
            end = float(interval["end_s"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            return None
        result.append((start, end, interval.get("event_id")))
    return result


def _qa18_integer_window_activity(
    facts: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
    tails: Sequence[tuple[float, float, str | None]],
) -> dict[str, Any]:
    fps = float(facts["time"]["frame_rate_hz"])
    sample_rate = float(facts["time"]["sample_rate_hz"])
    start_sample = int(round(start_frame * sample_rate / fps))
    end_sample = int(round(end_frame * sample_rate / fps))
    active_events = [
        event
        for event in _bound_events(facts)
        if any(
            int(row.get("start_sample", 0)) < end_sample
            and int(row.get("end_sample_exclusive", 0)) > start_sample
            for row in _source_activity_for_event(
                facts, str(event.get("event_id"))
            )
        )
    ]
    active_event_ids = sorted(
        str(event.get("event_id")) for event in active_events
    )
    active_actor_ids = sorted(
        {
            str(event.get("actor_id"))
            for event in active_events
            if event.get("actor_id") is not None
        }
    )
    start_s = start_frame / fps
    end_s = end_frame / fps
    wet_tail_event_ids = [
        str(event_id)
        for tail_start, tail_end, event_id in tails
        if tail_start < end_s and tail_end > start_s and event_id is not None
    ]
    return {
        "activity_class": (
            "empty"
            if not active_actor_ids
            else "active"
            if len(active_actor_ids) == 1
            else "multiple"
        ),
        "active_event_ids": active_event_ids,
        "active_actor_ids": active_actor_ids,
        "active_event_count": len(active_event_ids),
        "active_actor_count": len(active_actor_ids),
        "wet_tail_event_ids": list(dict.fromkeys(wet_tail_event_ids)),
    }


def _qa18_integer_bins(
    facts: Mapping[str, Any],
    requested: Any,
) -> list[dict[str, Any]]:
    if not _source_activity_present(facts):
        _defer(
            "missing_source_activity_readback",
            "QA-18 needs source activity readback before selecting a query window",
        )
    if facts.get("source_activity_evidence_complete") is False:
        _defer(
            "missing_source_activity_readback",
            "QA-18 cannot select a query window from partial source activity evidence",
        )
    tails = _qa18_wet_tail_intervals(facts)
    if tails is None:
        _defer(
            "sampling_window_missing",
            "QA-18 needs measured wet-tail intervals before selecting a query window",
        )
    count = int(facts["time"]["frame_count"])
    fps = float(facts["time"]["frame_rate_hz"])
    duration = float(facts["time"]["duration_seconds"])
    allowed_frames: set[int] | None = None
    if requested is not None:
        allowed = _window_bounds_list(requested, qa_id="QA-18")
        if any(start < 0 or start >= end or end > count for start, end in allowed):
            _defer(
                "sampling_window_invalid",
                "QA-18 legal frame windows must be inside the frame clock",
            )
        allowed_frames = {
            frame for start, end in allowed for frame in range(start, end)
        }
    result: list[dict[str, Any]] = []
    for second in range(int(math.floor(duration + 1.0e-9))):
        start_frame = int(round(second * fps))
        end_frame = int(round((second + 1) * fps))
        if end_frame <= start_frame or end_frame > count:
            continue
        window_frames = list(range(start_frame, end_frame))
        if allowed_frames is not None and not set(window_frames) <= allowed_frames:
            continue
        activity = _qa18_integer_window_activity(
            facts, start_frame, end_frame, tails
        )
        if activity["active_actor_ids"] or not activity["wet_tail_event_ids"]:
            result.append(
                {
                    "window_frames": [start_frame, end_frame],
                    "window_seconds": [float(second), float(second + 1)],
                    **activity,
                }
            )
    return result


def _qa18_integer_query_candidates(
    facts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fps = float(facts["time"]["frame_rate_hz"])
    declared_time = _sampling_value(
        facts, "QA-18", "query_time_s_by_qa", "query_times_s"
    )
    declared_frame = _sampling_value(
        facts,
        "QA-18",
        "query_frame_by_qa",
        "query_frames",
        "query_frame",
        "at_frame",
    )

    def uniform(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and value.get("policy", value.get("query_time_policy"))
            == "uniform_in_legal_window"
        )

    uniform_time = uniform(declared_time)
    uniform_frame = uniform(declared_frame)
    exact = (
        declared_time is not None and not uniform_time
    ) or (
        declared_frame is not None and not uniform_frame
    )
    requested = (
        _first(
            declared_time,
            "window_frames",
            "frame_window",
            "legal_window",
            "window",
        )
        if uniform_time
        else _first(
            declared_frame,
            "window_frames",
            "frame_window",
            "legal_window",
            "window",
        )
        if uniform_frame
        else None
    )
    if requested is None:
        requested = _sampling_value(
            facts,
            "QA-18",
            "legal_window_by_qa",
            "legal_windows",
            "query_windows",
        )

    if exact:
        if declared_time is not None:
            time_s = _resolve_query_time_spec(
                facts, "QA-18", declared_time, source="sampling"
            )
            query_frames = [
                _resolve_query_frame_spec(
                    facts,
                    "QA-18",
                    int(round(time_s * fps)),
                    source="sampling",
                )
            ]
        else:
            query_frames = [
                _resolve_query_frame_spec(
                    facts, "QA-18", declared_frame, source="sampling"
                )
            ]
        if not _source_activity_present(facts):
            return [
                {
                    "candidate_id": f"QA-18:frame:{frame}:time:{frame / fps:.9f}",
                    "kind": "query_time",
                    "query_frame": frame,
                    "query_time_s": frame / fps,
                }
                for frame in query_frames
            ]
    else:
        query_frames = []

    bins = _qa18_integer_bins(facts, requested)
    if not bins:
        _defer(
            "sampling_window_missing",
            "QA-18 has no whole-second window with observed activity or proven silence",
        )
    legal_windows = [entry["window_frames"] for entry in bins]
    authority = (
        "caller_declared_sampling_window"
        if requested is not None
        else _derived_query_window_authority("QA-18")
    )
    if exact:
        frame = query_frames[0]
        selected = next(
            (
                entry
                for entry in bins
                if entry["window_frames"][0] <= frame < entry["window_frames"][1]
            ),
            None,
        )
        if selected is None:
            _defer(
                "sampling_window_missing",
                "QA-18 query frame is not inside a legal whole-second window",
            )
        return [
            {
                "candidate_id": f"QA-18:frame:{frame}:time:{frame / fps:.9f}",
                "kind": "query_time",
                "query_frame": frame,
                "query_time_s": frame / fps,
                **selected,
                "legal_query_windows": copy.deepcopy(legal_windows),
                "legal_window_authority": authority,
            }
        ]

    return [
        {
            "candidate_id": (
                f"QA-18:frame:{(entry['window_frames'][0] + entry['window_frames'][1] - 1) // 2}:"
                f"time:{((entry['window_frames'][0] + entry['window_frames'][1] - 1) // 2) / fps:.9f}"
            ),
            "kind": "query_time",
            "query_frame": (
                entry["window_frames"][0] + entry["window_frames"][1] - 1
            ) // 2,
            "query_time_s": (
                entry["window_frames"][0] + entry["window_frames"][1] - 1
            ) / (2 * fps),
            **entry,
            "legal_query_windows": copy.deepcopy(legal_windows),
            "legal_window_authority": authority,
        }
        for entry in bins
    ]


def _generate_qa18_integer_window(
    facts: Mapping[str, Any],
    seed: str,
    *,
    query_time: float,
    query_source: str,
    candidate: Mapping[str, Any],
    frame: int,
    reviewed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if facts.get("source_activity_evidence_complete") is False:
        _defer(
            "missing_source_activity_readback",
            "QA-18 cannot answer from partial source activity evidence",
        )
    tails = _qa18_wet_tail_intervals(facts)
    if tails is None:
        _defer(
            "sampling_window_missing",
            "QA-18 needs measured wet-tail intervals before selecting a query window",
        )
    window = candidate.get("activity_query_window")
    if (
        not _is_sequence(window)
        or len(window) != 2
        or isinstance(window[0], bool)
        or isinstance(window[1], bool)
        or not isinstance(window[0], int)
        or not isinstance(window[1], int)
    ):
        bins = _qa18_integer_bins(facts, None)
        fps = float(facts["time"]["frame_rate_hz"])
        selected = next(
            (
                entry
                for entry in bins
                if entry["window_frames"][0] <= frame < entry["window_frames"][1]
            ),
            None,
        )
        if selected is None:
            _defer(
                "sampling_window_missing",
                "QA-18 query frame is not inside a legal whole-second window",
            )
        window = selected["window_frames"]
    start_frame, end_frame = int(window[0]), int(window[1])
    activity = _qa18_integer_window_activity(facts, start_frame, end_frame, tails)
    active_actor_ids = activity["active_actor_ids"]
    if not active_actor_ids and activity["wet_tail_event_ids"]:
        _defer(
            "query_inside_wet_tail",
            "QA-18 empty query window overlaps measured listener-side wet tails",
            query_time_s=query_time,
            event_ids=activity["wet_tail_event_ids"],
        )
    if any(actor_id not in reviewed for actor_id in active_actor_ids):
        _defer(
            "speaker_appearance_review_missing",
            "the specified-time speaker has no reviewed appearance label",
            actor_ids=active_actor_ids,
        )
    truth = (
        active_actor_ids[0]
        if len(active_actor_ids) == 1
        else "multiple"
        if len(active_actor_ids) > 1
        else "none"
    )
    options = _actor_options(facts, list(reviewed))
    options.extend(
        [_option("multiple", "multiple actors"), _option("none", "no actor")]
    )
    if truth not in {option["value"] for option in options}:
        _defer(
            "speaker_at_time_truth_missing",
            "query truth is absent from its option domain",
        )
    window_fields = _query_window_fields(facts, [start_frame, end_frame])
    display = _display_time_range(facts, [start_frame, end_frame])
    if display is None:
        _defer(
            "query_interval_too_short_for_display",
            "the QA-18 whole-second query window has no public range",
        )
    display_en, display_zh = display
    legal_windows = candidate.get("legal_query_windows")
    legal_windows = (
        copy.deepcopy(legal_windows)
        if _is_sequence(legal_windows)
        else [[start_frame, end_frame]]
    )
    legal_authority = candidate.get("legal_window_authority")
    return _question_item(
        qa_id="QA-18",
        facts=facts,
        seed=seed,
        question_en=(
            f"Which actor(s) made a sound at any point during {display_en}?"
        ),
        question_zh=(
            f"{display_zh}内，哪些个体曾经发声（至少在某一时刻）？"
        ),
        open_answer_type="closed_set",
        open_truth=truth,
        truth_label=(
            "multiple actors"
            if truth == "multiple"
            else "no actor"
            if truth == "none"
            else _appearance_phrases(
                facts["actors"][truth]["appearance"]
            )[0]
        ),
        options=options,
        evidence={
            "query_time_s": query_time,
            "query_source": query_source,
            "query_frame": frame,
            **window_fields,
            "source_activity_coordinate_space": "episode_sample_clock",
            "source_activity_event_ids": activity["active_event_ids"],
            "active_event_ids": activity["active_event_ids"],
            "active_actor_ids": active_actor_ids,
            "activity_class": activity["activity_class"],
            "activity_semantics": "union_any_time_within_query_window_v1",
            "wet_tail_event_ids": activity["wet_tail_event_ids"],
            "wet_tail_boundary_policy": "measured_interval_only_for_empty_branch",
            "legal_query_windows": legal_windows,
            "legal_window_authority": (
                legal_authority
                or _derived_query_window_authority("QA-18")
            ),
            "appearance_reviews": {
                actor_id: _appearance_review_for(facts, actor_id)
                for actor_id in reviewed
            },
        },
        slug=f"frame_{frame}",
    )


def _qa18_query_candidates(
    facts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Enumerate legal QA-18 frames and preserve activity branch diversity."""
    if _time_display_precision(facts) == 0:
        return _qa18_integer_query_candidates(facts)

    fps = float(facts["time"]["frame_rate_hz"])
    declared_time = _sampling_value(
        facts, "QA-18", "query_time_s_by_qa", "query_times_s"
    )
    declared_frame = _sampling_value(
        facts,
        "QA-18",
        "query_frame_by_qa",
        "query_frames",
        "query_frame",
        "at_frame",
    )

    def uniform(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and value.get("policy", value.get("query_time_policy"))
            == "uniform_in_legal_window"
        )

    uniform_time = uniform(declared_time)
    uniform_frame = uniform(declared_frame)
    exact = (
        declared_time is not None and not uniform_time
    ) or (
        declared_frame is not None and not uniform_frame
    )
    requested_window = (
        _first(
            declared_time,
            "window_frames",
            "frame_window",
            "legal_window",
            "window",
        )
        if uniform_time
        else _first(
            declared_frame,
            "window_frames",
            "frame_window",
            "legal_window",
            "window",
        )
        if uniform_frame
        else None
    )
    if requested_window is None:
        requested_window = _sampling_value(
            facts,
            "QA-18",
            "legal_window_by_qa",
            "legal_windows",
            "query_windows",
        )

    if exact:
        if declared_time is not None:
            time_s = _resolve_query_time_spec(
                facts, "QA-18", declared_time, source="sampling"
            )
            query_frames = [
                _resolve_query_frame_spec(
                    facts,
                    "QA-18",
                    int(round(time_s * fps)),
                    source="sampling",
                )
            ]
        else:
            query_frames = [
                _resolve_query_frame_spec(
                    facts, "QA-18", declared_frame, source="sampling"
                )
            ]
    else:
        if not _source_activity_present(facts):
            code = (
                "sampling_window_missing"
                if _query_time_policy(facts) == "uniform_in_legal_window"
                else "missing_source_activity_readback"
            )
            _defer(
                code,
                "QA-18 needs source activity readback before selecting a query frame",
            )
        legal_windows, authority = _qa18_legal_windows(
            facts, requested_window
        )
        if legal_windows is None:
            _defer(
                "sampling_window_missing",
                "QA-18 needs measured wet-tail intervals before selecting a query frame",
            )
        if not legal_windows:
            _defer(
                "sampling_window_missing",
                "QA-18 has no legal query frame outside measured wet tails",
            )
        frame_data, stable_by_frame = _qa18_stable_activity_windows(
            facts, legal_windows
        )
        query_frames = sorted(stable_by_frame)
        if not query_frames:
            _defer(
                "sampling_window_missing",
                "QA-18 has no stable active, multiple-active or empty query window",
            )

    if _source_activity_present(facts):
        if exact:
            legal_windows, authority = _qa18_legal_windows(
                facts, requested_window
            )
            frame_data = {}
            stable_by_frame = {}
            if legal_windows:
                frame_data, stable_by_frame = _qa18_stable_activity_windows(
                    facts, legal_windows
                )
            for frame in query_frames:
                frame_data.setdefault(
                    frame, _qa18_frame_activity(facts, frame)
                )
    else:
        frame_data = {}
        stable_by_frame = {}
        authority = None
        legal_windows = None

    result: list[dict[str, Any]] = []
    for frame in query_frames:
        item = {
            "candidate_id": f"QA-18:frame:{frame}:time:{frame / fps:.9f}",
            "kind": "query_time",
            "query_frame": frame,
            "query_time_s": frame / fps,
        }
        info = frame_data.get(frame)
        if info is not None:
            item.update(info)
        if legal_windows:
            item["legal_query_windows"] = copy.deepcopy(legal_windows)
            item["legal_window_authority"] = authority
        if frame in stable_by_frame:
            item["activity_query_window"] = list(stable_by_frame[frame])
        result.append(item)
    return result




def _query_time_candidates(
    facts: Mapping[str, Any],
    qa_id: str,
    *,
    event: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate query instants, not one duplicate per unrelated sound event."""
    if qa_id == "QA-18":
        return _qa18_query_candidates(facts)
    count = int(facts['time']['frame_count'])
    fps = float(facts['time']['frame_rate_hz'])
    declared_time = _sampling_value(facts, qa_id, 'query_time_s_by_qa', 'query_times_s')
    declared_frame = _sampling_value(facts, qa_id, 'query_frame_by_qa', 'query_frames', 'query_frame', 'at_frame')
    window = None
    derived_windows = None
    if isinstance(declared_time, Mapping) or isinstance(declared_frame, Mapping):
        spec = declared_time if isinstance(declared_time, Mapping) else declared_frame
        if spec.get('policy', spec.get('query_time_policy')) == 'uniform_in_legal_window':
            window = _first(spec, 'window_frames', 'frame_window', 'legal_window', 'window')
    if declared_time is not None and window is None:
        time_s = _resolve_query_time_spec(facts, qa_id, declared_time, source='sampling')
        frame = _resolve_query_frame_spec(facts, qa_id, int(round(time_s * fps)), source='sampling')
        candidates = [(frame, time_s)]
    elif declared_frame is not None and window is None:
        frame = _resolve_query_frame_spec(facts, qa_id, declared_frame, source='sampling')
        candidates = [(frame, frame / fps)]
    else:
        sampling = facts.get('sampling', {})
        nested = sampling.get('qa_sampling', {}) if isinstance(sampling, Mapping) else {}
        policy = _first(nested, 'query_time_policy', 'policy') or _first(sampling, 'query_time_policy', 'policy')
        derived_windows = None
        if window is None:
            window = _sampling_value(facts, qa_id, 'legal_window_by_qa', 'legal_windows', 'query_windows')
        if policy == 'uniform_in_legal_window' and window is None:
            derived_windows = _derived_legal_query_windows(
                facts, qa_id, event=event
            )
            if derived_windows is None:
                _defer('sampling_window_missing', f'{qa_id} needs concrete legal query windows')
            if not derived_windows:
                _defer('sampling_window_missing', f'{qa_id} has no legal query frame')
            _record_derived_query_windows(
                facts, qa_id, derived_windows, event=event
            )
            window = derived_windows
        if window is None:
            frames = range(count)
        else:
            windows = _window_bounds_list(window, qa_id=qa_id)
            if any(not 0 <= start < end <= count for start, end in windows):
                _defer('sampling_window_invalid', f'{qa_id} legal windows must be half-open frame intervals')
            frames = sorted({f for start, end in windows for f in range(start, end)})
        candidates = [(frame, frame / fps) for frame in frames]
    result = []
    for frame, time_s in candidates:
        item = {
            'candidate_id': f'{qa_id}:frame:{frame}:time:{time_s:.9f}',
            'kind': 'query_time',
            'query_frame': frame,
            'query_time_s': time_s,
        }
        if derived_windows is not None:
            item['legal_query_windows'] = copy.deepcopy(derived_windows)
            item['legal_window_authority'] = _derived_query_window_authority(qa_id)
        result.append(item)
    return result

def _safe_candidate_count(facts: Mapping[str, Any], qa_id: str) -> int:
    try:
        return len(_P8_CANDIDATES[qa_id](facts))
    except _Deferred:
        return 0



def _post_sound_candidates(facts: Mapping[str, Any], qa_id: str, events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    '''Enumerate real silent query frames; do not always choose the earliest one.'''
    result = []
    for event in events:
        try:
            _anchor_pre_silence(facts, event)
        except _Deferred:
            continue
        specific = dict(facts)
        declared = _first(event, 'post_sound_query_frame', 'query_frame')
        if declared is None:
            declared = _sampling_value(facts, qa_id, 'post_sound_query_frame')
        explicit_window = _sampling_value(
            facts, qa_id, 'legal_window_by_qa', 'legal_windows', 'query_windows'
        )
        derived_windows = None
        if (
            declared is None
            and explicit_window is None
            and _query_time_policy(facts) == 'uniform_in_legal_window'
        ):
            derived_windows = _derived_legal_query_windows(
                facts, qa_id, event=event
            )
            if derived_windows is None or not derived_windows:
                continue
            _record_derived_query_windows(
                facts, qa_id, derived_windows, event=event
            )
            sampling = copy.deepcopy(facts.get('sampling', {}))
            sampling.setdefault('legal_window_by_qa', {})[qa_id] = derived_windows
            specific['sampling'] = sampling
        if declared is not None:
            sampling = copy.deepcopy(facts.get('sampling', {}))
            sampling.setdefault('query_frame_by_qa', {})[qa_id] = declared
            specific['sampling'] = sampling
        try:
            queries = _query_time_candidates(
                specific, qa_id, event=event
            )
        except _Deferred:
            if derived_windows is not None:
                continue
            raise
        for query in queries:
            frame = query['query_frame']
            acceptance = (facts.get('sampling') or {}).get('acceptance_policy') or {}
            if (qa_id == 'QA-16' and _ordinary_observation_questions(facts)
                    and acceptance.get('post_sound_distance_query') == 'integer_timepoint'):
                seconds = frame / float(facts['time']['frame_rate_hz'])
                if abs(seconds - round(seconds)) > 1e-8:
                    continue
            try:
                _silent_after(facts, event, frame)
            except _Deferred:
                continue
            item = {
                'candidate_id': f"{qa_id}:event:{event['event_id']}:frame:{frame}",
                'kind': 'post_event_query', 'actor_id': event['actor_id'],
                'event_id': event['event_id'],
                'query_frame': frame, 'query_time_s': query['query_time_s'],
            }
            legal_windows = query.get(
                'legal_query_windows',
                derived_windows if derived_windows is not None else explicit_window,
            )
            if legal_windows is not None:
                if derived_windows is None and explicit_window is not None:
                    legal_windows = _window_bounds_list(
                        legal_windows, qa_id=qa_id
                    )
                item['legal_query_windows'] = copy.deepcopy(legal_windows)
                item['legal_window_authority'] = (
                    _derived_query_window_authority(qa_id)
                    if derived_windows is not None
                    else 'caller_declared_sampling_window'
                )
            result.append(item)
    return result


def iter_unified_items(question_set: Mapping[str, Any], *, include_angle_followups: bool = True):
    """Yield main questions and explicit linked angle questions exactly once."""
    seen = set()
    for key in (("items", "angle_followups") if include_angle_followups else ("items",)):
        for item in question_set.get(key, []):
            if not isinstance(item, Mapping):
                continue
            question_id = item.get("question_id")
            if question_id in seen:
                continue
            seen.add(question_id)
            yield item


def model_input_questions(question_set: Mapping[str, Any]) -> dict[str, Any]:
    """Export public questions with IDs scoped to this set, never hidden actor IDs."""
    rows = []
    for index, item in enumerate(iter_unified_items(question_set)):
        rows.append({"question_id": f"question_{index + 1:06d}", "qa_id": item["qa_id"],
                     "forms": copy.deepcopy(item.get("model_input", {})),
                     "required_modalities": copy.deepcopy(item.get("required_modalities"))})
    return {"schema": "avengine_qa_public_questions_v1", "episode_id": question_set.get("episode_id"),
            "items": rows, "count": len(rows)}

__all__.extend(["QA_IDS", "iter_unified_items", "model_input_questions"])
