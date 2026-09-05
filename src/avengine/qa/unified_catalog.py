"""Unified QA-01..QA-24 conditions, episode evidence and question mining.

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


UNIFIED_INPUT_SCHEMA = "avengine_qa_unified_episode_input_v1"
UNIFIED_FACT_SCHEMA = "avengine_qa_unified_episode_facts_v1"
UNIFIED_ITEM_SCHEMA = "avengine_qa_unified_question_v1"
UNIFIED_OUTPUT_SCHEMA = "avengine_qa_unified_question_set_v1"
CATALOG_VERSION = "20260906"

VISIBLE_STATES = {"visible_clear", "visible_occluded"}
VISIBILITY_STATES = (
    "out_of_view",
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
)
_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


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
        "answer_type": "time_s",
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
)

_CATALOG_BY_ID = {item["qa_id"]: item for item in CATALOG}


def _canonical_qa_id(value: Any) -> str:
    if not isinstance(value, str):
        raise UnifiedQAError("qa_id must be a string such as QA-01")
    match = re.fullmatch(r"QA[-_]?(\d{1,2})", value.strip().upper())
    if not match:
        raise UnifiedQAError(f"unknown qa_id {value!r}")
    number = int(match.group(1))
    if not 1 <= number <= 24:
        raise UnifiedQAError(f"qa_id must be QA-01 through QA-24, got {value!r}")
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
            return {
                "field": field,
                "value": value.strip(),
                "label": (
                    str(label).strip()
                    if isinstance(label, str) and label.strip()
                    else value.strip()
                ),
            }
    return None


def _actor_label(
    actor_id: str,
    record: Mapping[str, Any],
    appearance: Mapping[str, Any] | None,
) -> str:
    value = _first(record, "display_label", "label")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if appearance and isinstance(appearance.get("label"), str):
        return str(appearance["label"])
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


def _content_from_event(
    event: Mapping[str, Any],
    *,
    sound_record: Mapping[str, Any] | None,
    voice_record: Mapping[str, Any] | None,
) -> tuple[str | None, str | None, str | None, str | None, bool]:
    """Return transcript, statement id, language, sound class and explicitness."""

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
                "category",
            )
            if isinstance(candidate, str) and candidate.strip():
                sound_class = candidate.strip()
                explicit = True
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
        if not _is_sequence(frames) or len(frames) != frame_count:
            continue
        by_frame: dict[int, Mapping[str, Any]] = {}
        valid = True
        for ordinal, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
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
            ):
                valid = False
                break
            normalized_frame = dict(frame)
            normalized_frame.setdefault("frame_index", index)
            by_frame[index] = normalized_frame
        if valid and set(by_frame) == set(range(frame_count)):
            result[actor_id] = by_frame
    return result


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
                "event_segmentation": value.get(
                    "event_segmentation", value.get("event_segmentation_status")
                ),
                "source_qc": copy.deepcopy(value.get("source_qc")),
                "source_record": "audio_program",
            }
        )
    events.sort(key=lambda item: (item["start_s"], item["event_id"]))

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
        "audio": audio,
        "visibility": visibility,
        "visibility_meta": visibility_meta,
        "appearance_review": appearance_review,
        "input_summary": {
            "plan_present": bool(plan),
            "frame_readbacks_present": bool(frame_readbacks),
            "pixel_visibility_truth_present": isinstance(visibility_value, Mapping),
            "audio_program_present": bool(audio_program),
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
        "sampling": root.get("sampling") or root.get("qa_sampling") or {},
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
    return sorted(events, key=lambda event: (float(event["start_s"]), event["event_id"]))


def _first_event(facts: Mapping[str, Any], actor_id: str) -> Mapping[str, Any]:
    events = _event_for_actor(facts, actor_id)
    if not events:
        _defer("target_has_no_event", f"actor {actor_id!r} has no bound sound event")
    return events[0]


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
        values.setdefault(value.strip(), []).append(str(actor_id))
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
                "appearance selector matches more than one actor",
                duplicate_values=duplicate,
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
            label_en = label_zh = str(actor.get("display_label") or actor_id)
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


def _appearance_phrases(appearance: Mapping[str, Any]) -> tuple[str, str]:
    value = str(appearance.get("value", "")).strip()
    return _APPEARANCE_WORDS.get(
        value.casefold(),
        (value, value),
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
    values = [str(item[2]["value"]) for item in candidates]
    if include_values is not None:
        wanted = set(include_values)
        values = [value for value in values if value in wanted]
    values = list(dict.fromkeys(values))
    return [
        {"value": value, "label_en": value, "label_zh": value}
        for value in values
    ]


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
    if not isinstance(frames, Mapping) or frame not in frames:
        _defer(
            "missing_pixel_visibility",
            f"pixel visibility truth is unavailable for {actor_id!r} at frame {frame}",
            actor_id=actor_id,
            frame=frame,
        )
    value = frames[frame]
    if not isinstance(value, Mapping) or value.get("state") not in VISIBILITY_STATES:
        _defer(
            "invalid_pixel_visibility",
            f"pixel visibility state is unavailable for {actor_id!r} at frame {frame}",
            actor_id=actor_id,
            frame=frame,
        )
    return value


def _active_at(facts: Mapping[str, Any], frame: int) -> list[Mapping[str, Any]]:
    return [
        event
        for event in _bound_events(facts)
        if _event_frame(event, "start_frame") <= frame < _event_frame(event, "end_frame")
    ]


def _sampling_value(
    facts: Mapping[str, Any],
    qa_id: str,
    *keys: str,
) -> Any:
    sampling = facts.get("sampling")
    if not isinstance(sampling, Mapping):
        return None
    for key in keys:
        value = sampling.get(key)
        if isinstance(value, Mapping):
            candidate = value.get(qa_id) or value.get(qa_id.lower())
            if candidate is None:
                candidate = value.get(qa_id.replace("-", "_"))
            if candidate is not None:
                return candidate
        elif value is not None:
            return value
    queries = sampling.get("queries")
    if isinstance(queries, Mapping):
        query = queries.get(qa_id) or queries.get(qa_id.lower())
        if isinstance(query, Mapping):
            return _first(query, *keys)
    return None


def _query_frame(
    facts: Mapping[str, Any],
    qa_id: str,
    *,
    event: Mapping[str, Any] | None = None,
    after_event: bool = False,
    require_declared: bool = False,
) -> tuple[int, str]:
    fields = (
        ("post_sound_query_frame", "query_frame")
        if after_event
        else ("query_frame", "at_frame")
    )
    if event is not None:
        for field in fields:
            value = event.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                return value, f"event.{field}"
    value = _sampling_value(
        facts, qa_id, "query_frame_by_qa", "query_frames", *fields
    )
    if isinstance(value, int) and not isinstance(value, bool):
        return value, "sampling"
    if require_declared:
        _defer("missing_query_frame", f"{qa_id} requires an explicit query frame")
    frame_count = int(facts["time"]["frame_count"])
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
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite_number(value, name="query_time_s"), "sampling"
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
    slug: str,
) -> dict[str, Any]:
    canonical = _canonical_qa_id(qa_id)
    question_id = (
        f"{canonical.lower().replace('-', '_')}__"
        f"{_safe_slug(str(facts['episode_id']))}__{_safe_slug(slug)}"
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
    forms: dict[str, Any] = {
        "open": open_form,
    }
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
        "open": {"status": "pass"},
        "mcq": (
            mcq_deferred
            if mcq_deferred is not None
            else {"status": "pass"}
        ),
    }
    model_input: dict[str, Any] = {
        "open": {"question_en": question_en, "question_zh": question_zh},
    }
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
        "question": {"en": question_en, "zh": question_zh},
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


def _state_options() -> list[dict[str, str]]:
    return [
        _option("visible_clear", "clearly visible"),
        _option("visible_occluded", "partially occluded"),
        _option("fully_occluded", "fully occluded"),
        _option("out_of_view", "out of view"),
    ]


def _event_pair(facts: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    events = sorted(_bound_events(facts), key=lambda event: (float(event["start_s"]), event["event_id"]))
    if len(events) < 2:
        _defer("insufficient_events", "this question requires at least two bound events")
    return events[0], events[1]


def _after_event_candidates(
    facts: Mapping[str, Any],
    *,
    qa_id: str,
) -> Any:
    events = sorted(_bound_events(facts), key=lambda event: (float(event["end_s"]), event["event_id"]))
    for event in events:
        try:
            pre_silence = _anchor_pre_silence(facts, event)
        except _Deferred:
            continue
        query_frame, query_source = _query_frame(
            facts, qa_id, event=event, after_event=True
        )
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
            yield event, candidate_frame, {
                **silence,
                "query_source": query_source,
                "pre_silence": pre_silence,
            }


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
            f"What appearance value belongs to the actor of {anchor_en}"
            + (
                f" (the recorded utterance is {event['transcript']!r})?"
                if event.get("transcript")
                else "?"
            )
        ),
        question_zh=(
            f"{anchor_zh}对应的个体是什么外观属性？"
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
    event = _bound_events(facts)[0] if _bound_events(facts) else None
    if event is None:
        _defer("missing_bound_events", "QA-04 needs a bound sound event")
    frame = _event_frame(event, "start_frame")
    angle = _azimuth(facts, event["actor_id"], max(0, min(frame, facts["time"]["frame_count"] - 1)))
    if abs(angle) < 5.0:
        _defer("front_dead_zone", "speaker angle is inside the left/right dead zone")
    side = "right" if angle > 0 else "left"
    anchor_en, anchor_zh = _event_anchor(facts, event)
    return _question_item(
        qa_id="QA-04",
        facts=facts,
        seed=seed,
        question_en=f"At the onset of {anchor_en}, was the source on your left or right?",
        question_zh=f"{anchor_zh}开始时，声源在听者左侧还是右侧？",
        open_answer_type="closed_set",
        open_truth=side,
        truth_label=side,
        options=[_option("left", "left"), _option("right", "right")],
        evidence={**_event_evidence(event), "query_frame": frame, "azimuth_deg": angle},
        slug=event["event_id"],
    )


def _generate_qa_05(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    first, second = _event_pair(facts)
    overlap_start = max(float(first["start_s"]), float(second["start_s"]))
    overlap_end = min(float(first["end_s"]), float(second["end_s"]))
    truth = "yes" if overlap_start < overlap_end else "no"
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
            "overlap_interval_s": [overlap_start, overlap_end] if truth == "yes" else None,
        },
        slug=f"{first['event_id']}_{second['event_id']}",
    )


def _generate_qa_06(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    events = _bound_events(facts)
    for event in events:
        actor_id = event["actor_id"]
        try:
            moving = _stable_motion_window(
                facts,
                actor_id,
                max(0, _event_frame(event, "start_frame")),
                min(int(facts["time"]["frame_count"]), _event_frame(event, "end_frame")),
            )
        except _Deferred:
            continue
        anchor_en, anchor_zh = _event_anchor(facts, event)
        return _question_item(
            qa_id="QA-06",
            facts=facts,
            seed=seed,
            question_en=f"Was the source moving while making {anchor_en}?",
            question_zh=f"{anchor_zh}期间，声源在运动吗？",
            open_answer_type="closed_set",
            open_truth="moving" if moving else "still",
            truth_label="moving" if moving else "still",
            options=[_option("moving", "moving"), _option("still", "staying still")],
            evidence={**_event_evidence(event), "moving": moving},
            slug=event["event_id"],
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
    for actor_id, frames in facts.get("visibility", {}).items():
        if actor_id not in reviewed:
            continue
        if not isinstance(frames, Mapping):
            continue
        ordered = [frames[index] for index in sorted(frames)]
        for previous, current in zip(ordered, ordered[1:]):
            if previous.get("state") != "out_of_view" or current.get("state") not in VISIBLE_STATES:
                continue
            centroid = current.get("target_centroid_xy_px")
            if not _is_sequence(centroid) or len(centroid) != 2:
                continue
            offset = float(centroid[0]) - center
            if abs(offset) <= dead_zone:
                continue
            side = "right" if offset > 0 else "left"
            appearance_en, appearance_zh = _appearance_phrases(reviewed[actor_id])
            return _question_item(
                qa_id="QA-07",
                facts=facts,
                seed=seed,
                question_en=(
                    f"Did the {appearance_en} enter from the "
                    "left or right side of the frame during the clip?"
                ),
                question_zh=f"{appearance_zh}从画面左侧还是右侧入画？",
                open_answer_type="closed_set",
                open_truth=side,
                truth_label=side,
                options=[_option("left", "left"), _option("right", "right")],
                evidence={
                    "target_actor_id": actor_id,
                    "entry_frame": current.get("frame_index"),
                    "centroid_xy_px": list(centroid),
                    "side_dead_zone_px": dead_zone,
                },
                slug=f"{actor_id}_entry",
            )
    _defer("no_entry_transition", "no out_of_view to visible transition with an unambiguous side")


def _generate_qa_08(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    for event in _bound_events(facts):
        frame = max(0, min(int(facts["time"]["frame_count"]) - 1, _event_frame(event, "start_frame")))
        try:
            state = _require_visibility(facts, event["actor_id"], frame).get("state")
        except _Deferred:
            continue
        anchor_en, anchor_zh = _event_anchor(facts, event)
        return _question_item(
            qa_id="QA-08",
            facts=facts,
            seed=seed,
            question_en=f"What was the source's visibility state during {anchor_en}?",
            question_zh=f"{anchor_zh}期间，声源处于什么可见状态？",
            open_answer_type="closed_set",
            open_truth=state,
            truth_label=state,
            options=_state_options(),
            evidence={**_event_evidence(event), "query_frame": frame, "visibility_state": state},
            slug=event["event_id"],
        )
    _defer("no_event_visibility", "no sound event has pixel visibility at its onset")


def _generate_qa_09(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    reviewed = _reviewed_appearances(facts)
    for actor_id, frames in facts.get("visibility", {}).items():
        if actor_id not in reviewed:
            continue
        ordered = [frames[index] for index in sorted(frames)] if isinstance(frames, Mapping) else []
        fully = [frame.get("frame_index") for frame in ordered if frame.get("state") == "fully_occluded"]
        visible_after = [
            frame.get("frame_index")
            for frame in ordered
            if frame.get("state") in VISIBLE_STATES
            and any(int(previous) < int(frame.get("frame_index", 0)) for previous in fully)
        ]
        if fully:
            truth = "yes" if visible_after else "no"
            appearance_en, appearance_zh = _appearance_phrases(reviewed[actor_id])
            return _question_item(
                qa_id="QA-09",
                facts=facts,
                seed=seed,
                question_en=(
                    f"Did the {appearance_en} reappear "
                    "after being fully occluded before the end of the clip?"
                ),
                question_zh=(
                    f"{appearance_zh}完全遮挡后又重新出现了吗？"
                ),
                open_answer_type="closed_set",
                open_truth=truth,
                truth_label=truth,
                options=[_option("yes", "yes"), _option("no", "no")],
                evidence={
                    "target_actor_id": actor_id,
                    "fully_occluded_frames": fully,
                    "reappeared_frames": visible_after,
                    "observation_window": [0, int(facts["time"]["frame_count"]) - 1],
                },
                slug=f"{actor_id}_reappearance",
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
        if isinstance(value, Mapping):
            label = value.get("display_label") or value.get("label") or value.get("category")
            if isinstance(label, str) and label.strip():
                return label.strip()
        if isinstance(value, str) and value.strip():
            return value.strip()
    return occluder_id


def _generate_qa_10(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    reviewed = _reviewed_appearances(facts)
    for actor_id, frames in facts.get("visibility", {}).items():
        if actor_id not in reviewed:
            continue
        if not isinstance(frames, Mapping):
            continue
        for frame, value in frames.items():
            if value.get("state") not in {"visible_occluded", "fully_occluded"}:
                continue
            ids = _occluder_ids(facts, actor_id, int(frame))
            if len(ids) != 1:
                continue
            appearance_en, appearance_zh = _appearance_phrases(reviewed[actor_id])
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
            candidate_ids = list(observed_ids)
            candidate_ids.extend(
                item for item in registry if item not in candidate_ids
            )
            if len(candidate_ids) < 2:
                candidate_ids.extend(
                    item
                    for item in facts.get("actors", {})
                    if item in registry and item not in candidate_ids
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
            return _question_item(
                qa_id="QA-10",
                facts=facts,
                seed=seed,
                question_en=(
                    f"Which visible object or person occluded the "
                    f"{appearance_en} at frame {frame}?"
                ),
                question_zh=(
                    f"这一帧中哪个可见的物体或人物遮挡了"
                    f"{appearance_zh}？（帧{frame}）"
                ),
                open_answer_type="closed_set",
                open_truth=ids[0],
                truth_label=_occluder_label(facts, ids[0]),
                options=options,
                evidence={
                    "target_actor_id": actor_id,
                    "frame": frame,
                    "occluder_instance_ids": ids,
                    "option_instance_ids": candidate_ids,
                },
                mcq_optional=len(candidate_ids) < 2,
                slug=f"{actor_id}_occluder_{frame}",
            )
    _defer("missing_occluder_identity", "pixel visibility contains no unique occluder identity")


def _generate_qa_11(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    reviewed = _reviewed_appearances(facts)
    for actor_id, frames in facts.get("visibility", {}).items():
        if actor_id not in reviewed:
            continue
        if not isinstance(frames, Mapping):
            continue
        ordered = [frames[index] for index in sorted(frames)]
        transitions = [
            current.get("frame_index")
            for previous, current in zip(ordered, ordered[1:])
            if previous.get("state") == "visible_occluded"
            and current.get("state") == "visible_clear"
        ]
        partial_frames = [
            frame.get("frame_index")
            for frame in ordered
            if frame.get("state") == "visible_occluded"
        ]
        if partial_frames:
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
                    f"{appearance_zh}是否从部分遮挡变为清晰可见？"
                ),
                open_answer_type="closed_set",
                open_truth=truth,
                truth_label=truth,
                options=[_option("yes", "yes"), _option("no", "no")],
                evidence={
                    "target_actor_id": actor_id,
                    "partial_occlusion_frames": partial_frames,
                    "transition_frames": transitions,
                    "observation_window": [0, int(facts["time"]["frame_count"]) - 1],
                },
                slug=f"{actor_id}_clear",
            )
    _defer("no_partial_to_clear_transition", "no adjacent partial-occlusion to clear transition is present")


def _generate_qa_12(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    _require_stereo(facts)
    actor_id, actor, event = _target_with_event(
        facts, require_content=True, require_visible=True
    )
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
        question_en=f"What did the {actor['appearance']['value']} actor say?",
        question_zh=f"{actor['appearance']['value']}的个体说了什么？",
        open_answer_type="transcript_wer",
        open_truth=event["transcript"],
        truth_label=str(event["transcript"]),
        options=[_option(text, text) for text in transcripts],
        evidence={
            "target_actor_id": actor_id,
            "appearance": dict(actor["appearance"]),
            "appearance_review": _appearance_review_for(facts, actor_id),
            "event": _event_evidence(event),
            "statement_id": event.get("statement_id"),
        },
        slug=actor_id,
    )


def _generate_qa_13(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    candidate_seen = False
    first_open_candidate: dict[str, Any] | None = None
    for event, query_frame, silence in _after_event_candidates(
        facts, qa_id="QA-13"
    ):
        candidate_seen = True
        try:
            query_visibility = _require_visibility(
                facts, event["actor_id"], query_frame
            )
            if query_visibility.get("state") not in VISIBLE_STATES:
                continue
            angle = _azimuth(facts, event["actor_id"], query_frame)
            distractor_angles = {
                actor_id: _azimuth(facts, actor_id, query_frame)
                for actor_id in facts["actors"]
                if actor_id != event["actor_id"]
            }
        except _Deferred:
            continue
        others = list(distractor_angles.values())
        if others and min(
            abs(((angle - other + 180.0) % 360.0) - 180.0)
            for other in others
        ) <= 60.0:
            continue
        anchor_en, anchor_zh = _event_anchor(facts, event)
        sector = _sector_of(angle)
        query_time = query_frame / float(facts["time"]["frame_rate_hz"])
        distractor_sectors = {
            actor_id: _sector_of(other_angle)
            for actor_id, other_angle in distractor_angles.items()
        }
        same_sector = {
            actor_id: distractor_sectors[actor_id]
            for actor_id in distractor_sectors
            if distractor_sectors[actor_id] == sector
        }
        options = _sector_options()
        candidate = {
            "question_en": (
                f"After {anchor_en} ended, what was the source's numeric "
                f"azimuth at {query_time:.3f} seconds (video frame {query_frame})? "
                "Report one angle in degrees: front is 0°, right is positive, "
                "and the range is [-180°, 180°)."
            ),
            "question_zh": (
                f"{anchor_zh}结束后，在第{query_time:.3f}秒（视频帧{query_frame}）"
                "声源的数值方位角是多少？请用度数回答：正前方为0°，右侧为正，"
                "范围为[-180°，180°）。"
            ),
            "mcq_question_en": (
                f"After {anchor_en} ended, which 90-degree sector contains the "
                f"source at {query_time:.3f} seconds (video frame {query_frame})? "
                "Choose one: front [-45°, 45°), right [45°, 135°), "
                "left [-135°, -45°), or back [135°, 180°) ∪ [-180°, -135°)."
            ),
            "mcq_question_zh": (
                f"{anchor_zh}结束后，在第{query_time:.3f}秒（视频帧{query_frame}）"
                "声源属于哪个90°扇区？请选择：前方[-45°，45°)、右方[45°，135°)、"
                "左方[-135°，-45°)，或后方[135°，180°) ∪ [-180°，-135°)。"
            ),
            "open_truth": angle,
            "truth_label": sector,
            "evidence": {
                **_event_evidence(event),
                "post_sound": silence,
                "query_frame": query_frame,
                "query_time_s": query_time,
                "query_visibility_state": query_visibility.get("state"),
                "azimuth_deg": angle,
                "distractor_azimuths_deg": distractor_angles,
                "target_sector": sector,
                "distractor_sectors": distractor_sectors,
                "sector_boundary_convention": (
                    "equal_width_half_open: front[-45,45), right[45,135), "
                    "back[135,180)U[-180,-135), left[-135,-45)"
                ),
            },
            "slug": f"{event['event_id']}_post_direction",
            "same_sector": same_sector,
        }
        if not same_sector:
            return _question_item(
                qa_id="QA-13",
                facts=facts,
                seed=seed,
                question_en=candidate["question_en"],
                question_zh=candidate["question_zh"],
                open_answer_type="angle_deg",
                open_truth=candidate["open_truth"],
                truth_label=candidate["truth_label"],
                options=options,
                mcq_truth=sector,
                mcq_question_en=candidate["mcq_question_en"],
                mcq_question_zh=candidate["mcq_question_zh"],
                open_extra={
                    "convention": "right_positive",
                    "convention_description": (
                        "azimuth_deg; front=0°, right_positive, range=[-180°,180°)"
                    ),
                    "theta_full_deg": 15.0,
                    "theta_half_deg": 30.0,
                },
                evidence=candidate["evidence"],
                slug=candidate["slug"],
            )
        if first_open_candidate is None:
            first_open_candidate = candidate
    if first_open_candidate is not None:
        same_sector = first_open_candidate["same_sector"]
        return _question_item(
            qa_id="QA-13",
            facts=facts,
            seed=seed,
            question_en=first_open_candidate["question_en"],
            question_zh=first_open_candidate["question_zh"],
            open_answer_type="angle_deg",
            open_truth=first_open_candidate["open_truth"],
            truth_label=first_open_candidate["truth_label"],
            options=_sector_options(),
            mcq_truth=first_open_candidate["truth_label"],
            mcq_question_en=first_open_candidate["mcq_question_en"],
            mcq_question_zh=first_open_candidate["mcq_question_zh"],
            open_extra={
                "convention": "right_positive",
                "convention_description": (
                    "azimuth_deg; front=0°, right_positive, range=[-180°,180°)"
                ),
                "theta_full_deg": 15.0,
                "theta_half_deg": 30.0,
            },
            evidence=first_open_candidate["evidence"],
            mcq_deferred_reason={
                "code": "mcq_same_sector",
                "detail": (
                    "MCQ requires every distractor to occupy a different "
                    "equal-width half-open sector"
                ),
                "target_sector": first_open_candidate["truth_label"],
                "conflicting_distractors": same_sector,
            },
            slug=first_open_candidate["slug"],
        )
    if not candidate_seen:
        _defer(
            "no_valid_post_sound_window",
            "no bound event has a later silent query frame",
        )
    _defer(
        "post_sound_angle_not_separated",
        "no legal post-sound query has target and distractor angles separated by more than 2x the 30 degree placeholder band",
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
    if isinstance(sampled_frame, int) and not isinstance(sampled_frame, bool):
        frame_candidates = [
            max(0, min(frame_count - 1, int(sampled_frame)))
        ]
        query_source = "sampling_frame"
    elif isinstance(sampled_time, (int, float)) and not isinstance(sampled_time, bool):
        frame_candidates = [
            max(
                0,
                min(
                    frame_count - 1,
                    int(round(float(sampled_time) * frame_rate)),
                ),
            )
        ]
        query_source = "sampling_time"
    else:
        frame_candidates = list(range(frame_count))
        query_source = "first_valid_frame_search"
    selected: tuple[str, Mapping[str, Any], str, Mapping[str, Any], float, float, int] | None = None
    for frame in frame_candidates:
        for first_index, (first_id, first_actor, _first_appearance) in enumerate(candidates):
            for second_id, second_actor, _second_appearance in candidates[first_index + 1 :]:
                try:
                    first_state = _require_visibility(facts, first_id, frame)
                    second_state = _require_visibility(facts, second_id, frame)
                except _Deferred:
                    continue
                if (
                    first_state.get("state") not in VISIBLE_STATES
                    or second_state.get("state") not in VISIBLE_STATES
                ):
                    continue
                try:
                    first_distance = _distance_at(facts, first_id, frame)
                    second_distance = _distance_at(facts, second_id, frame)
                except _Deferred:
                    continue
                if abs(first_distance - second_distance) >= 0.5:
                    selected = (
                        first_id,
                        first_actor,
                        second_id,
                        second_actor,
                        first_distance,
                        second_distance,
                        frame,
                    )
                    break
            if selected is not None:
                break
        if selected is not None:
            break
    if selected is None:
        _defer(
            "no_valid_distance_query",
            "no sampled frame has two reviewed visible targets with a 0.5 m distance margin",
        )
    first_id, first_actor, second_id, second_actor, first_distance, second_distance, frame = selected
    query_time = frame / frame_rate
    truth = first_id if first_distance < second_distance else second_id
    options = _actor_options(facts, [first_id, second_id])
    return _question_item(
        qa_id="QA-14",
        facts=facts,
        seed=seed,
        question_en=(
            f"At {query_time:.3f} seconds (video frame {frame}), which actor "
            "is closer to the listener?"
        ),
        question_zh=f"第{query_time:.3f}秒（视频帧{frame}）时，哪个个体离听者更近？",
        open_answer_type="closed_set",
        open_truth=truth,
        truth_label=str(
            facts["actors"][truth].get("appearance", {}).get(
                "label", facts["actors"][truth]["display_label"]
            )
        ),
        options=options,
        evidence={
            "query_time_s": query_time,
            "query_source": query_source,
            "query_frame": frame,
            "distances_m": {first_id: first_distance, second_id: second_distance},
            "appearance_reviews": {
                first_id: _appearance_review_for(facts, first_id),
                second_id: _appearance_review_for(facts, second_id),
            },
        },
        slug=f"{first_id}_{second_id}_{frame}",
    )


def _generate_qa_15(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    for event in _bound_events(facts):
        start_frame = max(0, _event_frame(event, "start_frame"))
        end_frame = min(int(facts["time"]["frame_count"]) - 1, max(start_frame + 1, _event_frame(event, "end_frame") - 1))
        try:
            start_distance = _distance_at(facts, event["actor_id"], start_frame)
            end_distance = _distance_at(facts, event["actor_id"], end_frame)
        except _Deferred:
            continue
        delta = end_distance - start_distance
        if abs(delta) < 0.2:
            continue
        truth = "nearer" if delta < 0 else "farther"
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
                "distance_start_m": start_distance,
                "distance_end_m": end_distance,
                "delta_m": delta,
            },
            slug=event["event_id"],
        )
    _defer("no_distance_trend_during_event", "no event has a measurable distance trend")


def _generate_qa_16(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    candidate_seen = False
    for event, query_frame, silence in _after_event_candidates(
        facts, qa_id="QA-16"
    ):
        candidate_seen = True
        try:
            start_frame = max(0, _event_frame(event, "start_frame"))
            anchor_frame = max(
                start_frame, _event_frame(event, "end_frame") - 1
            )
            start_distance = _distance_at(
                facts, event["actor_id"], anchor_frame
            )
            query_distance = _distance_at(
                facts, event["actor_id"], query_frame
            )
        except _Deferred:
            continue
        delta = query_distance - start_distance
        if abs(delta) < 0.2:
            continue
        anchor_en, anchor_zh = _event_anchor(facts, event)
        query_time = query_frame / float(facts["time"]["frame_rate_hz"])
        truth = "nearer" if delta < 0 else "farther"
        return _question_item(
            qa_id="QA-16",
            facts=facts,
            seed=seed,
            question_en=(
                f"After {anchor_en} ended, was the source nearer or farther "
                f"at {query_time:.3f} seconds (video frame {query_frame})?"
            ),
            question_zh=(
                f"{anchor_zh}结束后，在第{query_time:.3f}秒（视频帧{query_frame}）"
                "比发声时更近还是更远？"
            ),
            open_answer_type="closed_set",
            open_truth=truth,
            truth_label=truth,
            options=[_option("nearer", "nearer"), _option("farther", "farther")],
            evidence={
                **_event_evidence(event),
                "post_sound": silence,
                "distance_anchor_m": start_distance,
                "distance_query_m": query_distance,
                "delta_m": delta,
                "query_time_s": query_time,
            },
            slug=f"{event['event_id']}_post_distance",
        )
    if not candidate_seen:
        _defer(
            "no_valid_post_sound_window",
            "no bound event has a later silent query frame",
        )
    _defer(
        "no_distance_change_after_event",
        "no legal post-sound event has a distance change above the research margin",
    )


def _generate_qa_17(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    first_valid: tuple[Mapping[str, Any], int, dict[str, Any], list[bool]] | None = None
    for event, query_frame, silence in _after_event_candidates(
        facts, qa_id="QA-17"
    ):
        try:
            end_frame = min(query_frame, _event_frame(event, "end_frame"))
            values = [
                _motion_at(facts, event["actor_id"], frame)
                for frame in range(max(0, end_frame), query_frame + 1)
            ]
        except _Deferred:
            continue
        if first_valid is None:
            first_valid = (event, query_frame, silence, values)
        if any(values):
            first_valid = (event, query_frame, silence, values)
            break
    if first_valid is None:
        _defer(
            "no_valid_post_sound_window",
            "no bound event has a later silent query frame with motion readback",
        )
    event, query_frame, silence, values = first_valid
    anchor_en, anchor_zh = _event_anchor(facts, event)
    end_frame = min(query_frame, _event_frame(event, "end_frame"))
    truth = "yes" if any(values) else "no"
    return _question_item(
        qa_id="QA-17",
        facts=facts,
        seed=seed,
        question_en=f"Did the source move after {anchor_en} ended?",
        question_zh=f"{anchor_zh}结束后声源还移动过吗？",
        open_answer_type="closed_set",
        open_truth=truth,
        truth_label=truth,
        options=[_option("yes", "yes"), _option("no", "no")],
        evidence={
            **_event_evidence(event),
            "post_sound": silence,
            "motion_frames": [end_frame, query_frame],
            "moving_values": values,
        },
        slug=f"{event['event_id']}_post_motion",
    )


def _generate_qa_18(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    _require_stereo(facts)
    reviewed = _reviewed_appearances(facts)
    query_time, query_source = _query_time(facts, "QA-18")
    frame = max(
        0,
        min(
            int(facts["time"]["frame_count"]) - 1,
            int(round(query_time * float(facts["time"]["frame_rate_hz"]))),
        ),
    )
    active = _active_at(facts, frame)
    active_actor_ids = list(dict.fromkeys(event["actor_id"] for event in active if event.get("actor_id")))
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
    return _question_item(
        qa_id="QA-18",
        facts=facts,
        seed=seed,
        question_en=f"At {query_time:.3f} seconds, who is making a sound?",
        question_zh=f"第{query_time:.3f}秒时谁在发声？",
        open_answer_type="closed_set",
        open_truth=truth,
        truth_label=(
            "multiple actors"
            if truth == "multiple"
            else "no actor"
            if truth == "none"
            else str(
                facts["actors"][truth].get("appearance", {}).get(
                    "label", facts["actors"][truth]["display_label"]
                )
            )
        ),
        options=options,
        evidence={
            "query_time_s": query_time,
            "query_source": query_source,
            "query_frame": frame,
            "active_event_ids": [event["event_id"] for event in active],
            "active_actor_ids": active_actor_ids,
            "appearance_reviews": {
                actor_id: _appearance_review_for(facts, actor_id)
                for actor_id in reviewed
            },
        },
        slug=f"frame_{frame}",
    )


def _time_bands(facts: Mapping[str, Any]) -> list[tuple[float, float]]:
    duration = float(facts["time"]["duration_seconds"])
    step = duration / 4.0
    return [(index * step, (index + 1) * step) for index in range(4)]


def _generate_qa_19(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_stereo(facts)
    actor_id, actor, event = _target_with_event(facts, require_visible=True)
    appearance = actor["appearance"]
    first = _first_event(facts, actor_id)
    time_s = float(first["start_s"])
    bands = _time_bands(facts)
    band_index = next(
        (index for index, (lo, hi) in enumerate(bands) if lo <= time_s < hi),
        len(bands) - 1,
    )
    options = [
        {
            **_option(f"band_{index}", f"[{lo:.2f}, {hi:.2f}) s"),
            "allow_value": False,
        }
        for index, (lo, hi) in enumerate(bands)
    ]
    return _question_item(
        qa_id="QA-19",
        facts=facts,
        seed=seed,
        question_en=f"At what time did the {appearance['value']} actor first make a sound?",
        question_zh=f"{appearance['value']}的个体第一次发声是在第几秒？",
        open_answer_type="time_s",
        open_truth=time_s,
        truth_label=f"{time_s:.3f} s",
        options=options,
        mcq_truth=f"band_{band_index}",
        open_extra={"t_full_s": 0.3, "t_half_s": 1.0},
        evidence={
            "target_actor_id": actor_id,
            "appearance": dict(appearance),
            "appearance_review": _appearance_review_for(facts, actor_id),
            "first_event": _event_evidence(first),
            "time_bands_s": bands,
        },
        slug=f"{actor_id}_first_time",
    )


def _generate_qa_20(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    _require_stereo(facts)
    reviewed = _reviewed_appearances(facts)
    events = _bound_events(facts)
    for event in events:
        frame = max(0, min(int(facts["time"]["frame_count"]) - 1, _event_frame(event, "start_frame")))
        visible_ids = [
            actor_id
            for actor_id in facts["actors"]
            if actor_id in reviewed
            if actor_id in facts.get("visibility", {})
            and _state(facts, actor_id, frame).get("state") in VISIBLE_STATES
        ]
        if not visible_ids:
            continue
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
    target_classes = {
        str(candidate.get("sound_class"))
        for candidate in _event_for_actor(facts, target_id)
        if candidate.get("sound_class") and candidate.get("sound_class_explicit")
    }
    if len(target_classes) != 1:
        _defer(
            "target_sound_class_not_unique",
            "QA-21 target appearance maps to more than one sound class",
            target_actor_id=target_id,
            sound_classes=sorted(target_classes),
        )
    classes = list(
        dict.fromkeys(
            str(other.get("sound_class"))
            for other in facts["events"]
            if isinstance(other, Mapping)
            and other.get("sound_class")
            and other.get("sound_class_explicit")
        )
    )
    if len(classes) < 2:
        _defer("sound_class_option_domain_too_small", "QA-21 needs at least two explicit sound classes")
    if event["sound_class"] not in classes:
        _defer("sound_class_truth_missing", "target sound class is absent from the class domain")
    appearance_en, appearance_zh = _appearance_phrases(target["appearance"])
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
        truth_label=event["sound_class"],
        options=[_option(value, value) for value in classes],
        evidence={
            "target_actor_id": target_id,
            "appearance": dict(target["appearance"]),
            "appearance_review": _appearance_review_for(facts, target_id),
            "event": _event_evidence(event),
            "sound_class": event["sound_class"],
            "explicit_class": True,
            "distinct_sound_classes": classes,
        },
        slug=target_id,
    )


def _generate_qa_22(facts: Mapping[str, Any], seed: str) -> dict[str, Any]:
    _require_actor_count(facts, 2)
    events = facts.get("events")
    if not _is_sequence(events):
        _defer("missing_events", "QA-22 needs a whole-clip event table")
    if any(event.get("actor_id") is None for event in events if isinstance(event, Mapping)):
        _defer("unresolved_event_attribution", "QA-22 cannot count speaking individuals while an event source is unresolved")
    entity_count = len(facts["actors"])
    speaking_count = len({event["actor_id"] for event in events})
    truth = [entity_count, speaking_count]
    candidates = {(entity_count, speaking_count)}
    for entity_delta, speaker_delta in (
        (0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1)
    ):
        pair = (entity_count + entity_delta, speaking_count + speaker_delta)
        if pair[0] >= 0 and 0 <= pair[1] <= pair[0]:
            candidates.add(pair)
    pairs = sorted(candidates)
    while len(pairs) < 4:
        pair = (entity_count, max(0, min(entity_count, speaking_count + len(pairs))))
        if pair not in candidates:
            candidates.add(pair)
            pairs = sorted(candidates)
        else:
            break
    if len(pairs) < 2:
        _defer("count_option_domain_too_small", "QA-22 cannot construct distinct count options")
    options = [
        {
            **_option(
                f"{pair[0]}|{pair[1]}",
                f"{pair[0]} entities, {pair[1]} speaking",
            ),
            "allow_value": False,
        }
        for pair in pairs
    ]
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
        evidence={
            "statistics_window": [0.0, float(facts["time"]["duration_seconds"])],
            "entity_count": entity_count,
            "speaking_actor_ids": sorted({event["actor_id"] for event in events}),
            "speaking_count": speaking_count,
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
        truth_label=state,
        options=_state_options(),
        evidence={
            "anchor_event": _event_evidence(event),
            "target_actor_id": actor_id,
            "final_frame": final_frame,
            "final_visibility_state": state,
        },
        slug="first_speaker_final_state",
    )


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


def generate_unified_questions(
    raw_or_facts: Mapping[str, Any],
    *,
    qa_ids: Sequence[str] | None = None,
    seed: str = "avengine-qa-20260906",
) -> dict[str, Any]:
    """Generate valid rows and per-QA deferred records from one Episode.

    Passing the output of normalize_episode_bundle is supported for callers
    that want to retain the normalized facts. Raw native input is preferred.
    """

    if not isinstance(raw_or_facts, Mapping):
        raise UnifiedQAError("episode input must be an object")
    facts = (
        _restore_normalized_frame_keys(raw_or_facts)
        if raw_or_facts.get("schema") == UNIFIED_FACT_SCHEMA
        else normalize_episode_bundle(raw_or_facts)
    )
    requested = (
        [_canonical_qa_id(value) for value in qa_ids]
        if qa_ids is not None
        else [item["qa_id"] for item in CATALOG]
    )
    if len(requested) != len(set(requested)):
        raise UnifiedQAError("qa_ids must be unique")
    items: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for qa_id in requested:
        generator = _GENERATORS[qa_id]
        try:
            item = generator(facts, seed)
        except _Deferred as error:
            deferred.append(
                {
                    "qa_id": qa_id,
                    "status": "deferred",
                    "code": error.code,
                    "detail": error.detail,
                    **error.extra,
                    "requirements": get_requirements(qa_id),
                }
            )
            continue
        items.append(item)
    coverage: list[dict[str, Any]] = []
    item_by_qa = {item["qa_id"]: item for item in items}
    deferred_by_qa = {item["qa_id"]: item for item in deferred}
    for qa_id in requested:
        if qa_id in item_by_qa:
            coverage.append(
                {
                    "qa_id": qa_id,
                    "status": "pass",
                    "question_id": item_by_qa[qa_id]["question_id"],
                    "requirements": get_requirements(qa_id),
                }
            )
        else:
            coverage.append(dict(deferred_by_qa[qa_id]))
    return {
        "schema": UNIFIED_OUTPUT_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "catalog_version": CATALOG_VERSION,
        "episode_id": facts["episode_id"],
        "seed": seed,
        "input_facts": facts,
        "counts": {
            "requested": len(requested),
            "valid": len(items),
            "deferred": len(deferred),
        },
        "actual_evidence_summary": {
            "actor_count": len(facts.get("actors", {})),
            "event_count": len(facts.get("events", [])),
            "bound_event_count": sum(
                1
                for event in facts.get("events", [])
                if isinstance(event, Mapping) and isinstance(event.get("actor_id"), str)
            ),
            "unresolved_event_ids": list(
                facts.get("input_summary", {}).get("unresolved_event_ids", [])
            ),
            "reviewed_appearance_actor_count": len(
                facts.get("appearance_review", {})
            ),
            "pixel_visibility_actor_count": len(facts.get("visibility", {})),
            "audio_validation_status": facts.get("audio", {}).get("status"),
            "moving_actor_count": sum(
                1
                for actor in facts.get("actors", {}).values()
                if isinstance(actor, Mapping)
                and any(value is True for value in actor.get("moving", []) or [])
            ),
        },
        "coverage": coverage,
        "items": items,
        "deferred": deferred,
        "claim_boundary": (
            "Rows are deterministic research candidates derived from native "
            "readbacks. They are not model outcomes, formal admission or "
            "modality-necessity certificates."
        ),
    }


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
]

# Small descriptive aliases for callers that do not need the longer function
# names. They are aliases, not additional registries or question protocols.
QUESTION_CATALOG = CATALOG
get_question_requirements = get_requirements
build_unified_episode_facts = normalize_episode_bundle
generate_questions = generate_unified_questions
