"""A small, registry-bound QuestionSpec layer over QA Facts.

QuestionSpec deliberately does not generate assets, sounds, dialogue or scene
content.  A spec may only select values already present in the supplied asset
and sound registries.  It emits the scene requirements for that question and
then queries one compiled Fact table.  Missing or ambiguous evidence is
reported as ``rejected`` or ``unsupported``; it is never guessed.

The modality check below is a structural dependency check over Facts.  It is
not a claim that a learned model has passed a modality-ablation experiment.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


QUESTION_SPEC_SCHEMA = "avengine_qa_question_spec_v1"
QUESTION_SPEC_EVALUATION_SCHEMA = "avengine_qa_question_spec_evaluation_v1"
CLAIM_BOUNDARY = (
    "Deterministic registry-bound QuestionSpec evaluation over compiled Facts; "
    "modality necessity is structural, not a learned-model ablation result"
)
_SPEC_ID = re.compile(r"^QS-[0-9]{3}$")
_HASH_LIKE_ID = re.compile(r"^[0-9a-fA-F]{32,}$")
_SIDE_DEAD_ZONE_DEG = 5.0
_VISIBLE_STATES = {"visible_clear", "visible_occluded"}
_OCCLUDED_STATES = {"visible_occluded", "fully_occluded"}
_APPEARANCE_FIELDS = {
    "breed_id",
    "size",
    "body_build",
    "life_stage",
    "coat_value",
}


QUESTION_TYPES: tuple[dict[str, Any], ...] = (
    {
        "index": 1,
        "question_type": "appearance_to_speaking",
        "name_zh": "外貌→是否发声",
        "required_facts": ["instances", "sound_events"],
        "required_modalities": ["video", "audio"],
        "scene_constraints": [
            "一个受控外貌值只匹配一个实例",
            "实例与受控声音事件绑定",
        ],
    },
    {
        "index": 2,
        "question_type": "sound_to_appearance",
        "name_zh": "声音/内容→外貌",
        "required_facts": ["sound_events", "instances.attributes"],
        "required_modalities": ["audio", "video"],
        "scene_constraints": [
            "受控声音只对应一个发声实例",
            "被询问外貌字段已观测",
        ],
    },
    {
        "index": 3,
        "question_type": "who_spoke_first",
        "name_zh": "谁先发声",
        "required_facts": ["sound_events.start_tick"],
        "required_modalities": ["audio"],
        "scene_constraints": ["最早发声开始时刻只有一个实例"],
    },
    {
        "index": 4,
        "question_type": "speaker_side",
        "name_zh": "发声者左右",
        "required_facts": ["sound_events", "tracks.instances.*.doa.azimuth_deg"],
        "required_modalities": ["binaural_audio"],
        "scene_constraints": [
            "指定帧只有一个匹配声音正在发声",
            "方位角不落在正前方死区",
        ],
    },
    {
        "index": 5,
        "question_type": "overlapping_speech",
        "name_zh": "重叠发声",
        "required_facts": ["sound_events.start_tick", "sound_events.end_tick"],
        "required_modalities": ["audio"],
        "scene_constraints": ["两个受控声音均出现且时间窗可比较"],
    },
    {
        "index": 6,
        "question_type": "speaking_while_moving",
        "name_zh": "发声时是否运动",
        "required_facts": ["sound_events", "tracks.instances.*.moving"],
        "required_modalities": ["audio", "video"],
        "scene_constraints": [
            "受控声音只对应一个实例",
            "发声覆盖帧的运动状态一致",
        ],
    },
    {
        "index": 7,
        "question_type": "offscreen_to_onscreen",
        "name_zh": "画外→入画",
        "required_facts": ["visibility.pixel_truth.per_instance.*.frames"],
        "required_modalities": ["video", "pixel_visibility"],
        "scene_constraints": ["target-only 像素轨迹先画外、后可见"],
    },
    {
        "index": 8,
        "question_type": "occlusion_while_speaking",
        "name_zh": "发声时遮挡状态",
        "required_facts": [
            "sound_events",
            "visibility.pixel_truth.per_instance.*.frames.state",
        ],
        "required_modalities": ["audio", "video", "pixel_visibility"],
        "scene_constraints": ["指定帧只有一个匹配发声者且像素状态可用"],
    },
    {
        "index": 9,
        "question_type": "reappeared_after_occlusion",
        "name_zh": "遮挡后重新出现",
        "required_facts": ["visibility.pixel_truth.per_instance.*.frames"],
        "required_modalities": ["video", "pixel_visibility"],
        "scene_constraints": ["实例完全遮挡后存在后续可见帧"],
    },
    {
        "index": 10,
        "question_type": "occluder_identity",
        "name_zh": "遮挡者身份",
        "required_facts": [
            "visibility.pixel_truth.per_instance.*.frames.occluder_instance_ids"
        ],
        "required_modalities": ["video", "pixel_instance_visibility"],
        "scene_constraints": ["遮挡像素必须绑定到唯一受控实例 ID"],
    },
)

_TYPE_BY_ID = {item["question_type"]: item for item in QUESTION_TYPES}
_SELECTOR_KEYS = {
    "appearance_to_speaking": {"appearance_field", "appearance_value"},
    "sound_to_appearance": {"sound_asset_id", "appearance_field"},
    "who_spoke_first": set(),
    "speaker_side": {"sound_asset_id", "frame_index"},
    "overlapping_speech": {"sound_asset_ids"},
    "speaking_while_moving": {"sound_asset_id"},
    "offscreen_to_onscreen": {"target_instance_id"},
    "occlusion_while_speaking": {"sound_asset_id", "frame_index"},
    "reappeared_after_occlusion": {"target_instance_id"},
    "occluder_identity": {"target_instance_id", "frame_index"},
}


class QuestionSpecError(ValueError):
    """A QuestionSpec or one of its controlled inputs is invalid."""


@dataclass(frozen=True)
class _StopEvaluation(Exception):
    status: str
    code: str
    detail: str
    check: str
    evidence: Mapping[str, Any] | None = None


def question_type_catalog() -> list[dict[str, Any]]:
    """Return the fixed ten-type catalog in stable user-facing order."""

    return copy.deepcopy(list(QUESTION_TYPES))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _readable_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise QuestionSpecError(f"{name} must be a non-empty string")
    if _HASH_LIKE_ID.fullmatch(value):
        raise QuestionSpecError(f"{name} must be readable, not a hash-like id")
    return value


def _validate_spec(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(spec, Mapping):
        raise QuestionSpecError("QuestionSpec must be an object")
    allowed = {"schema", "spec_id", "question_type", "selectors"}
    extra = set(spec) - allowed
    if extra:
        raise QuestionSpecError(f"QuestionSpec has unsupported fields: {sorted(extra)}")
    if spec.get("schema") != QUESTION_SPEC_SCHEMA:
        raise QuestionSpecError(f"QuestionSpec schema must be {QUESTION_SPEC_SCHEMA!r}")
    spec_id = spec.get("spec_id")
    if not isinstance(spec_id, str) or not _SPEC_ID.fullmatch(spec_id):
        raise QuestionSpecError("spec_id must use the readable sequential form QS-001")
    question_type = spec.get("question_type")
    definition = _TYPE_BY_ID.get(question_type)
    if definition is None:
        raise QuestionSpecError("question_type is not one of the fixed ten types")
    selectors = spec.get("selectors")
    if not isinstance(selectors, Mapping):
        raise QuestionSpecError("selectors must be an object")
    expected = _SELECTOR_KEYS[question_type]
    if set(selectors) != expected:
        raise QuestionSpecError(
            f"{question_type} selectors must be exactly {sorted(expected)}"
        )

    for key in ("sound_asset_id", "target_instance_id"):
        if key in selectors:
            _readable_id(selectors[key], name=key)
    if "sound_asset_ids" in selectors:
        values = selectors["sound_asset_ids"]
        if not _is_sequence(values) or len(values) != 2:
            raise QuestionSpecError("sound_asset_ids must contain exactly two ids")
        readable = [_readable_id(value, name="sound_asset_ids item") for value in values]
        if len(set(readable)) != 2:
            raise QuestionSpecError("sound_asset_ids must be distinct")
    if "appearance_field" in selectors:
        if selectors["appearance_field"] not in _APPEARANCE_FIELDS:
            raise QuestionSpecError(
                f"appearance_field must be one of {sorted(_APPEARANCE_FIELDS)}"
            )
    if "appearance_value" in selectors:
        value = selectors["appearance_value"]
        if not isinstance(value, str) or not value:
            raise QuestionSpecError("appearance_value must be a non-empty registry value")
    if "frame_index" in selectors:
        frame = selectors["frame_index"]
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise QuestionSpecError("frame_index must be a non-negative integer")
    return dict(selectors), definition


def scenario_requirements(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one spec into requirements without inventing scene content."""

    selectors, definition = _validate_spec(spec)
    constraints = list(definition["scene_constraints"])
    for key, value in selectors.items():
        constraints.append({"selector": key, "registry_value": copy.deepcopy(value)})
    return {
        "question_type_index": definition["index"],
        "question_type_name_zh": definition["name_zh"],
        "required_facts": list(definition["required_facts"]),
        "required_modalities": list(definition["required_modalities"]),
        "constraints": constraints,
        "generation_policy": "select_from_controlled_registries_only",
    }


def _asset_records(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if not isinstance(registry, Mapping):
        raise _StopEvaluation(
            "rejected", "invalid_asset_registry", "asset registry must be an object", "registry_references"
        )
    records = registry.get("assets")
    id_field = "asset_id"
    if not _is_sequence(records):
        records = registry.get("entities")
        id_field = "entity_asset_id"
    if not _is_sequence(records):
        raise _StopEvaluation(
            "rejected",
            "invalid_asset_registry",
            "asset registry needs an assets or entities list",
            "registry_references",
        )
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise _StopEvaluation(
                "rejected", "invalid_asset_registry", "asset entries must be objects", "registry_references"
            )
        try:
            asset_id = _readable_id(record.get(id_field), name=id_field)
        except QuestionSpecError as error:
            raise _StopEvaluation(
                "rejected", "invalid_asset_registry", str(error), "registry_references"
            ) from error
        if asset_id in result:
            raise _StopEvaluation(
                "rejected", "invalid_asset_registry", f"duplicate asset id {asset_id!r}", "registry_references"
            )
        result[asset_id] = record
    return result


def _sound_records(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if not isinstance(registry, Mapping):
        raise _StopEvaluation(
            "rejected", "invalid_sound_registry", "sound registry must be an object", "registry_references"
        )
    families = (
        ("sounds", ("sound_asset_id", "sound_id", "sound_key")),
        ("sound_assets", ("sound_asset_id",)),
        ("samples", ("category",)),
    )
    records = None
    id_fields: tuple[str, ...] = ()
    for family, fields in families:
        candidate = registry.get(family)
        if _is_sequence(candidate):
            records = candidate
            id_fields = fields
            break
    if records is None:
        raise _StopEvaluation(
            "rejected",
            "invalid_sound_registry",
            "sound registry needs a sounds, sound_assets or samples list",
            "registry_references",
        )
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise _StopEvaluation(
                "rejected", "invalid_sound_registry", "sound entries must be objects", "registry_references"
            )
        raw_id = next((record.get(field) for field in id_fields if record.get(field)), None)
        try:
            sound_id = _readable_id(raw_id, name="sound registry id")
        except QuestionSpecError as error:
            raise _StopEvaluation(
                "rejected", "invalid_sound_registry", str(error), "registry_references"
            ) from error
        if sound_id in result:
            raise _StopEvaluation(
                "rejected", "invalid_sound_registry", f"duplicate sound id {sound_id!r}", "registry_references"
            )
        result[sound_id] = record
    return result


def _registry_asset_value(record: Mapping[str, Any], field: str) -> Any:
    identity = record.get("identity")
    realized = record.get("realized_attributes")
    if not isinstance(realized, Mapping):
        realized = record.get("realized_visual_attributes")
    if field == "breed_id":
        if isinstance(identity, Mapping) and identity.get(field) is not None:
            return identity.get(field)
        return realized.get(field) if isinstance(realized, Mapping) else None
    if not isinstance(realized, Mapping):
        return None
    if field == "coat_value":
        coat = realized.get("coat_profile")
        return coat.get("value") if isinstance(coat, Mapping) else None
    return realized.get(field)


def _fact_asset_value(instance: Mapping[str, Any], field: str) -> Any:
    if field == "breed_id":
        return instance.get(field)
    attributes = instance.get("attributes")
    return attributes.get(field) if isinstance(attributes, Mapping) else None


def _sound_binding_id(binding: Any) -> str:
    if isinstance(binding, str):
        value = binding
    elif isinstance(binding, Mapping):
        value = binding.get("sound_asset_id") or binding.get("sound_id")
    else:
        value = None
    try:
        return _readable_id(value, name="event sound binding")
    except QuestionSpecError as error:
        raise _StopEvaluation(
            "rejected", "invalid_event_sound_binding", str(error), "registry_references"
        ) from error


def _sound_provenance(record: Mapping[str, Any]) -> tuple[Any, Any]:
    path = record.get("path")
    sha256 = record.get("sha256")
    dry = record.get("dry_audio")
    if isinstance(dry, Mapping):
        path = path or dry.get("uri")
        sha256 = sha256 or dry.get("sha256")
    return path, sha256


def _prepare_context(
    facts: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    sound_registry: Mapping[str, Any],
    event_sound_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(facts, Mapping) or facts.get("schema") != "avengine_qa_fact_table_v1":
        raise _StopEvaluation(
            "rejected", "invalid_facts", "Facts must use avengine_qa_fact_table_v1", "observable"
        )
    if facts.get("status") != "pass":
        raise _StopEvaluation(
            "rejected", "invalid_facts", "Facts status is not pass", "observable"
        )
    instances = facts.get("instances")
    events = facts.get("sound_events")
    if not _is_sequence(instances) or not _is_sequence(events):
        raise _StopEvaluation(
            "rejected", "invalid_facts", "Facts instances and sound_events must be lists", "observable"
        )
    assets = _asset_records(asset_registry)
    sounds = _sound_records(sound_registry)
    by_instance: dict[str, Mapping[str, Any]] = {}
    by_slot: dict[str, Mapping[str, Any]] = {}
    for instance in instances:
        if not isinstance(instance, Mapping):
            raise _StopEvaluation(
                "rejected", "invalid_facts", "Facts instance entries must be objects", "observable"
            )
        instance_id = instance.get("instance_id")
        slot = instance.get("source_slot_id")
        asset_id = instance.get("asset_id")
        if not all(isinstance(value, str) and value for value in (instance_id, slot, asset_id)):
            raise _StopEvaluation(
                "rejected", "invalid_facts", "Facts instance ids are incomplete", "observable"
            )
        if instance_id in by_instance or slot in by_slot:
            raise _StopEvaluation(
                "rejected", "invalid_facts", "Facts instance or slot ids are duplicated", "observable"
            )
        record = assets.get(asset_id)
        if record is None:
            raise _StopEvaluation(
                "rejected",
                "registry_reference_missing",
                f"Facts asset {asset_id!r} is not in the controlled asset registry",
                "registry_references",
            )
        for field in _APPEARANCE_FIELDS:
            registered = _registry_asset_value(record, field)
            observed = _fact_asset_value(instance, field)
            if registered is not None and observed != registered:
                raise _StopEvaluation(
                    "rejected",
                    "asset_registry_mismatch",
                    f"{asset_id!r} field {field!r} differs between registry and Facts",
                    "registry_references",
                )
        by_instance[instance_id] = instance
        by_slot[slot] = instance

    if not isinstance(event_sound_bindings, Mapping):
        raise _StopEvaluation(
            "rejected",
            "invalid_event_sound_binding",
            "event_sound_bindings must be an event-id mapping",
            "registry_references",
        )
    event_ids = {event.get("event_id") for event in events if isinstance(event, Mapping)}
    extra_bindings = set(event_sound_bindings) - event_ids
    if extra_bindings:
        raise _StopEvaluation(
            "rejected",
            "invalid_event_sound_binding",
            f"bindings reference unknown events: {sorted(extra_bindings)}",
            "registry_references",
        )
    sound_id_by_event: dict[str, str] = {}
    normalized_events: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise _StopEvaluation(
                "rejected", "invalid_facts", "Facts sound events must be objects", "observable"
            )
        event_id = event.get("event_id")
        slot = event.get("source_slot_id")
        if not isinstance(event_id, str) or slot not in by_slot:
            raise _StopEvaluation(
                "rejected", "invalid_facts", "Facts sound event has an unknown source", "observable"
            )
        if event_id not in event_sound_bindings:
            raise _StopEvaluation(
                "rejected",
                "registry_reference_missing",
                f"event {event_id!r} has no controlled sound binding",
                "registry_references",
            )
        sound_id = _sound_binding_id(event_sound_bindings[event_id])
        sound_record = sounds.get(sound_id)
        if sound_record is None:
            raise _StopEvaluation(
                "rejected",
                "registry_reference_missing",
                f"event {event_id!r} selects unregistered sound {sound_id!r}",
                "registry_references",
            )
        species = sound_record.get("species")
        sound_class = event.get("sound_class")
        event_species = sound_class.get("species_id") if isinstance(sound_class, Mapping) else None
        if species is not None and species != event_species:
            raise _StopEvaluation(
                "rejected",
                "sound_registry_mismatch",
                f"sound {sound_id!r} species differs from event {event_id!r}",
                "registry_references",
            )
        registered_path, registered_sha = _sound_provenance(sound_record)
        dry_variant = event.get("dry_variant")
        event_path = dry_variant.get("input_path") if isinstance(dry_variant, Mapping) else None
        event_sha = dry_variant.get("input_sha256") if isinstance(dry_variant, Mapping) else None
        if registered_sha is not None and event_sha != registered_sha:
            raise _StopEvaluation(
                "rejected",
                "sound_provenance_mismatch",
                f"sound {sound_id!r} bytes do not match event {event_id!r}",
                "registry_references",
            )
        if isinstance(registered_path, str) and registered_path.startswith("/") and event_path != registered_path:
            raise _StopEvaluation(
                "rejected",
                "sound_provenance_mismatch",
                f"sound {sound_id!r} path does not match event {event_id!r}",
                "registry_references",
            )
        sound_id_by_event[event_id] = sound_id
        normalized_events.append(event)
    return {
        "facts": facts,
        "instances": by_instance,
        "instances_by_slot": by_slot,
        "events": normalized_events,
        "sound_records": sounds,
        "sound_id_by_event": sound_id_by_event,
    }


def _events_for_sound(context: Mapping[str, Any], sound_id: str) -> list[Mapping[str, Any]]:
    if sound_id not in context["sound_records"]:
        raise _StopEvaluation(
            "rejected",
            "registry_reference_missing",
            f"QuestionSpec selects unregistered sound {sound_id!r}",
            "registry_references",
        )
    events = [
        event
        for event in context["events"]
        if context["sound_id_by_event"][event["event_id"]] == sound_id
    ]
    if not events:
        raise _StopEvaluation(
            "rejected",
            "facts_not_observable",
            f"registered sound {sound_id!r} does not occur in this Episode",
            "observable",
        )
    return events


def _one_source_for_events(
    events: Sequence[Mapping[str, Any]], *, detail: str
) -> str:
    slots = {event["source_slot_id"] for event in events}
    if len(slots) != 1:
        raise _StopEvaluation(
            "rejected", "answer_not_unique", detail, "answer_unique", {"source_slots": sorted(slots)}
        )
    return next(iter(slots))


def _frame_count(context: Mapping[str, Any]) -> int:
    time = context["facts"].get("time")
    value = time.get("frame_count") if isinstance(time, Mapping) else None
    if not isinstance(value, int) or value <= 0:
        raise _StopEvaluation(
            "rejected", "invalid_facts", "Facts frame_count is unavailable", "observable"
        )
    return value


def _validate_frame(context: Mapping[str, Any], frame: int) -> None:
    if frame >= _frame_count(context):
        raise _StopEvaluation(
            "rejected",
            "facts_not_observable",
            f"frame {frame} lies outside this Episode",
            "observable",
        )


def _pixel_frames(context: Mapping[str, Any], instance_id: str) -> list[Mapping[str, Any]]:
    if instance_id not in context["instances"]:
        raise _StopEvaluation(
            "rejected",
            "registry_reference_missing",
            f"target instance {instance_id!r} is not registry-bound in this Episode",
            "registry_references",
        )
    visibility = context["facts"].get("visibility")
    truth = visibility.get("pixel_truth") if isinstance(visibility, Mapping) else None
    per_instance = truth.get("per_instance") if isinstance(truth, Mapping) else None
    track = per_instance.get(instance_id) if isinstance(per_instance, Mapping) else None
    frames = track.get("frames") if isinstance(track, Mapping) else None
    if not _is_sequence(frames) or len(frames) != _frame_count(context):
        raise _StopEvaluation(
            "rejected",
            "facts_not_observable",
            f"pixel truth is unavailable for {instance_id!r}",
            "observable",
        )
    return list(frames)


def _active_events(events: Sequence[Mapping[str, Any]], frame: int) -> list[Mapping[str, Any]]:
    return [
        event
        for event in events
        if isinstance(event.get("start_frame"), int)
        and isinstance(event.get("end_frame"), int)
        and event["start_frame"] <= frame < event["end_frame"]
    ]


def _question_and_answer(
    question_type: str,
    selectors: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if question_type == "appearance_to_speaking":
        field = selectors["appearance_field"]
        value = selectors["appearance_value"]
        matches = [
            instance
            for instance in context["instances"].values()
            if _fact_asset_value(instance, field) == value
        ]
        if len(matches) != 1:
            raise _StopEvaluation(
                "rejected",
                "answer_not_unique",
                f"appearance selector {field}={value!r} matches {len(matches)} instances",
                "answer_unique",
            )
        instance = matches[0]
        spoke = any(
            event["source_slot_id"] == instance["source_slot_id"]
            for event in context["events"]
        )
        return (
            f"外貌属性“{field}={value}”的个体是否发过声？",
            {"value": "yes" if spoke else "no", "label_zh": "是" if spoke else "否"},
            {"instance_id": instance["instance_id"], "matched_event_count": sum(
                event["source_slot_id"] == instance["source_slot_id"] for event in context["events"]
            )},
        )

    if question_type == "sound_to_appearance":
        sound_id = selectors["sound_asset_id"]
        events = _events_for_sound(context, sound_id)
        slot = _one_source_for_events(events, detail="the selected sound is emitted by multiple instances")
        instance = context["instances_by_slot"][slot]
        field = selectors["appearance_field"]
        value = _fact_asset_value(instance, field)
        if value is None:
            raise _StopEvaluation(
                "rejected", "facts_not_observable", f"appearance field {field!r} is unknown", "observable"
            )
        return (
            f"发出受控声音“{sound_id}”的个体，其 {field} 是什么？",
            {"value": value, "label_zh": str(value)},
            {"instance_id": instance["instance_id"], "event_ids": [event["event_id"] for event in events]},
        )

    if question_type == "who_spoke_first":
        if not context["events"]:
            raise _StopEvaluation(
                "rejected", "facts_not_observable", "the Episode has no sound events", "observable"
            )
        start = min(event["start_tick"] for event in context["events"])
        first = [event for event in context["events"] if event["start_tick"] == start]
        slots = {event["source_slot_id"] for event in first}
        if len(slots) != 1:
            raise _StopEvaluation(
                "rejected",
                "answer_not_unique",
                "multiple instances share the earliest start_tick",
                "answer_unique",
                {"start_tick": start, "source_slots": sorted(slots)},
            )
        instance = context["instances_by_slot"][next(iter(slots))]
        return (
            "谁先发声？",
            {"value": instance["instance_id"], "label_zh": instance["display_label"]},
            {"start_tick": start, "event_ids": [event["event_id"] for event in first]},
        )

    if question_type == "speaker_side":
        sound_id = selectors["sound_asset_id"]
        frame = selectors["frame_index"]
        _validate_frame(context, frame)
        active = _active_events(_events_for_sound(context, sound_id), frame)
        if not active:
            raise _StopEvaluation(
                "rejected", "facts_not_observable", f"{sound_id!r} is inactive at frame {frame}", "observable"
            )
        slot = _one_source_for_events(active, detail="multiple matching speakers are active at the requested frame")
        tracks = context["facts"].get("tracks")
        instance_tracks = tracks.get("instances") if isinstance(tracks, Mapping) else None
        track = instance_tracks.get(slot) if isinstance(instance_tracks, Mapping) else None
        doa = track.get("doa") if isinstance(track, Mapping) else None
        azimuths = doa.get("azimuth_deg") if isinstance(doa, Mapping) else None
        if not _is_sequence(azimuths) or len(azimuths) <= frame:
            raise _StopEvaluation(
                "rejected", "facts_not_observable", "per-frame Listener-relative DOA is unavailable", "observable"
            )
        azimuth = float(azimuths[frame])
        if abs(azimuth) < _SIDE_DEAD_ZONE_DEG:
            raise _StopEvaluation(
                "rejected",
                "answer_not_unique",
                f"azimuth {azimuth:.3f} is inside the ±{_SIDE_DEAD_ZONE_DEG:g}° front dead zone",
                "answer_unique",
            )
        side = "right" if azimuth > 0.0 else "left"
        return (
            f"第 {frame} 帧发出“{sound_id}”的个体在 Listener 左侧还是右侧？",
            {"value": side, "label_zh": "右侧" if side == "right" else "左侧"},
            {"source_slot_id": slot, "frame_index": frame, "azimuth_deg": azimuth},
        )

    if question_type == "overlapping_speech":
        first_id, second_id = selectors["sound_asset_ids"]
        first = _events_for_sound(context, first_id)
        second = _events_for_sound(context, second_id)
        overlaps = [
            {
                "first_event_id": left["event_id"],
                "second_event_id": right["event_id"],
                "start_tick": max(left["start_tick"], right["start_tick"]),
                "end_tick": min(left["end_tick"], right["end_tick"]),
            }
            for left in first
            for right in second
            if max(left["start_tick"], right["start_tick"])
            < min(left["end_tick"], right["end_tick"])
        ]
        answer = bool(overlaps)
        return (
            f"受控声音“{first_id}”与“{second_id}”是否发生过重叠？",
            {"value": "yes" if answer else "no", "label_zh": "是" if answer else "否"},
            {"overlaps": overlaps},
        )

    if question_type == "speaking_while_moving":
        sound_id = selectors["sound_asset_id"]
        events = _events_for_sound(context, sound_id)
        slot = _one_source_for_events(events, detail="the selected sound is emitted by multiple instances")
        tracks = context["facts"].get("tracks")
        instance_tracks = tracks.get("instances") if isinstance(tracks, Mapping) else None
        track = instance_tracks.get(slot) if isinstance(instance_tracks, Mapping) else None
        moving = track.get("moving") if isinstance(track, Mapping) else None
        if not _is_sequence(moving) or len(moving) != _frame_count(context):
            raise _StopEvaluation(
                "rejected", "facts_not_observable", "per-frame moving truth is unavailable", "observable"
            )
        frames = sorted(
            {
                frame
                for event in events
                for frame in range(max(0, event["start_frame"]), min(len(moving), event["end_frame"]))
            }
        )
        states = {bool(moving[frame]) for frame in frames}
        if not frames:
            raise _StopEvaluation(
                "rejected", "facts_not_observable", "the selected sound covers no video frame", "observable"
            )
        if len(states) != 1:
            raise _StopEvaluation(
                "rejected",
                "answer_not_unique",
                "the speaker changes motion state during the selected sound",
                "answer_unique",
                {"frame_indices": frames},
            )
        answer = next(iter(states))
        return (
            f"发出受控声音“{sound_id}”时，发声个体是否在运动？",
            {"value": "yes" if answer else "no", "label_zh": "是" if answer else "否"},
            {"source_slot_id": slot, "frame_indices": frames},
        )

    if question_type == "offscreen_to_onscreen":
        instance_id = selectors["target_instance_id"]
        frames = _pixel_frames(context, instance_id)
        transitions = []
        previous_state = None
        for frame in frames:
            state = frame.get("state")
            if previous_state == "out_of_view" and state in _VISIBLE_STATES:
                transitions.append(frame["frame_index"])
            previous_state = state
        answer = bool(transitions)
        return (
            f"受控实例“{instance_id}”是否经历过画外到入画？",
            {"value": "yes" if answer else "no", "label_zh": "是" if answer else "否"},
            {"entry_frame_indices": transitions},
        )

    if question_type == "occlusion_while_speaking":
        sound_id = selectors["sound_asset_id"]
        frame = selectors["frame_index"]
        _validate_frame(context, frame)
        active = _active_events(_events_for_sound(context, sound_id), frame)
        if not active:
            raise _StopEvaluation(
                "rejected", "facts_not_observable", f"{sound_id!r} is inactive at frame {frame}", "observable"
            )
        slot = _one_source_for_events(active, detail="multiple matching speakers are active at the requested frame")
        instance = context["instances_by_slot"][slot]
        frames = _pixel_frames(context, instance["instance_id"])
        state = frames[frame].get("state")
        if state not in _VISIBLE_STATES | {"fully_occluded", "out_of_view"}:
            raise _StopEvaluation(
                "rejected", "facts_not_observable", "pixel occlusion state is unknown", "observable"
            )
        labels = {
            "visible_clear": "清晰可见",
            "visible_occluded": "部分遮挡",
            "fully_occluded": "完全遮挡",
            "out_of_view": "画外",
        }
        return (
            f"第 {frame} 帧发出“{sound_id}”的个体处于什么遮挡状态？",
            {"value": state, "label_zh": labels[state]},
            {"instance_id": instance["instance_id"], "frame_index": frame},
        )

    if question_type == "reappeared_after_occlusion":
        instance_id = selectors["target_instance_id"]
        frames = _pixel_frames(context, instance_id)
        fully_occluded_at: list[int] = []
        reappeared_at: list[int] = []
        awaiting_reappearance = False
        for frame in frames:
            state = frame.get("state")
            if state == "fully_occluded":
                awaiting_reappearance = True
                fully_occluded_at.append(frame["frame_index"])
            elif awaiting_reappearance and state in _VISIBLE_STATES:
                reappeared_at.append(frame["frame_index"])
                awaiting_reappearance = False
        answer = bool(reappeared_at)
        return (
            f"受控实例“{instance_id}”是否在完全遮挡后重新出现？",
            {"value": "yes" if answer else "no", "label_zh": "是" if answer else "否"},
            {"fully_occluded_frames": fully_occluded_at, "reappeared_frames": reappeared_at},
        )

    if question_type == "occluder_identity":
        instance_id = selectors["target_instance_id"]
        frame_index = selectors["frame_index"]
        _validate_frame(context, frame_index)
        frame = _pixel_frames(context, instance_id)[frame_index]
        occluders = frame.get("occluder_instance_ids")
        if not _is_sequence(occluders):
            raise _StopEvaluation(
                "unsupported",
                "missing_occluder_identity",
                "modal/target-only pixels measure how much is hidden but do not identify which instance hid it",
                "observable",
                {"target_instance_id": instance_id, "frame_index": frame_index, "pixel_state": frame.get("state")},
            )
        known = [value for value in occluders if value in context["instances"]]
        if len(known) != 1 or len(known) != len(occluders):
            raise _StopEvaluation(
                "rejected",
                "answer_not_unique",
                "occluder identity is absent, unregistered or non-unique",
                "answer_unique",
                {"occluder_instance_ids": list(occluders)},
            )
        occluder = context["instances"][known[0]]
        return (
            f"第 {frame_index} 帧是谁遮挡了受控实例“{instance_id}”？",
            {"value": known[0], "label_zh": occluder["display_label"]},
            {"target_instance_id": instance_id, "frame_index": frame_index},
        )

    raise AssertionError(f"unhandled question type {question_type!r}")


def _base_evaluation(spec: Mapping[str, Any]) -> dict[str, Any]:
    spec_id = spec.get("spec_id") if isinstance(spec, Mapping) else None
    question_type = spec.get("question_type") if isinstance(spec, Mapping) else None
    definition = _TYPE_BY_ID.get(question_type)
    return {
        "schema": QUESTION_SPEC_EVALUATION_SCHEMA,
        "spec_id": spec_id,
        "question_type": question_type,
        "question_type_index": definition["index"] if definition else None,
        "question_type_name_zh": definition["name_zh"] if definition else None,
        "status": "rejected",
        "scenario_requirements": None,
        "question": None,
        "answer": None,
        "checks": [],
        "evidence": {},
        "reason": None,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def evaluate_question_spec(
    spec: Mapping[str, Any],
    *,
    facts: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    sound_registry: Mapping[str, Any],
    event_sound_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one registry-bound spec against one Episode Fact table."""

    result = _base_evaluation(spec)
    try:
        selectors, definition = _validate_spec(spec)
        result["scenario_requirements"] = scenario_requirements(spec)
        context = _prepare_context(
            facts, asset_registry, sound_registry, event_sound_bindings
        )
        result["checks"].append(
            {
                "name": "registry_references",
                "status": "pass",
                "detail": "all Episode instances and sound events resolve to controlled registries",
            }
        )
        question, answer, evidence = _question_and_answer(
            definition["question_type"], selectors, context
        )
        result.update(
            {
                "status": "pass",
                "question": question,
                "answer": answer,
                "evidence": evidence,
            }
        )
        result["checks"].extend(
            [
                {
                    "name": "answer_unique",
                    "status": "pass",
                    "detail": "the Facts query returned one deterministic answer",
                },
                {
                    "name": "observable",
                    "status": "pass",
                    "detail": "all facts required by this concrete QuestionSpec are present",
                },
                {
                    "name": "modality_necessity",
                    "status": "pass",
                    "detail": (
                        "structural dependency: "
                        + ", ".join(definition["required_modalities"])
                        + "; not a model-ablation claim"
                    ),
                },
            ]
        )
    except QuestionSpecError as error:
        result["reason"] = {"code": "invalid_question_spec", "detail": str(error)}
        result["checks"].append(
            {"name": "question_spec", "status": "rejected", "detail": str(error)}
        )
    except _StopEvaluation as stop:
        result["status"] = stop.status
        result["reason"] = {"code": stop.code, "detail": stop.detail}
        result["evidence"] = dict(stop.evidence or {})
        result["checks"].append(
            {"name": stop.check, "status": stop.status, "detail": stop.detail}
        )
    return result


def evaluate_question_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    facts: Mapping[str, Any],
    asset_registry: Mapping[str, Any],
    sound_registry: Mapping[str, Any],
    event_sound_bindings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate a stable ordered list of specs with shared controlled inputs."""

    if not _is_sequence(specs):
        raise QuestionSpecError("specs must be a sequence")
    seen: set[str] = set()
    results = []
    for spec in specs:
        spec_id = spec.get("spec_id") if isinstance(spec, Mapping) else None
        if spec_id in seen:
            raise QuestionSpecError(f"duplicate QuestionSpec id {spec_id!r}")
        if isinstance(spec_id, str):
            seen.add(spec_id)
        results.append(
            evaluate_question_spec(
                spec,
                facts=facts,
                asset_registry=asset_registry,
                sound_registry=sound_registry,
                event_sound_bindings=event_sound_bindings,
            )
        )
    return results
