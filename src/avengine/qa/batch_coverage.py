"""Batch coverage accounting for the full QA scope.

This module consumes an explicit manifest of generated facts/questions and
keeps the full asset-union x room-catalog x QA catalog denominator visible.
It records generated target-scoped evidence separately from global question
outcomes so an item for one actor is never multiplied over every actor in a
mixed pair.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.qa.answerability import structural_baselines


COVERAGE_SCHEMA = "avengine_qa_batch_coverage_v1"
INPUT_MANIFEST_SCHEMA = "avengine_qa_batch_episode_input_manifest_v1"
COVERAGE_STATES = (
    "produced",
    "deferred_by_rule",
    "not_applicable_by_definition",
    "interface_not_implemented",
    "evidence_missing_or_unsampled",
)
QA_IDS = tuple(f"QA-{index:02d}" for index in range(1, 25))
MOTION_TARGET_QA_IDS = frozenset({"QA-06", "QA-15", "QA-16", "QA-17"})
TRANSCRIPT_QA_ID = "QA-12"
STATIC_CLASS_NAMES = frozenset({"rigid_object", "rigid_static_object"})
REQUIRED_FIVE_MANIFEST_FIELDS = frozenset({"episode_id", "facts", "questions", "room_id", "family"})


class BatchCoverageError(ValueError):
    """Input or generated batch coverage is invalid."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BatchCoverageError(f"cannot read JSON input: {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise BatchCoverageError(f"invalid JSON input: {path}: {error}") from error


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _document(value: Any, *, base: Path, label: str) -> tuple[dict[str, Any], str]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value)), f"{label}:inline"
    if not isinstance(value, str) or not value:
        raise BatchCoverageError(f"{label} must be a JSON path or object")
    path = _resolve_path(value, base=base)
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise BatchCoverageError(f"{label} must contain an object: {path}")
    return dict(payload), str(path)


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BatchCoverageError(f"{label} must be a non-empty string")
    return value


def _asset_class(value: Any) -> str:
    value = str(value or "")
    if value == "rigid_object":
        return "rigid_static_object"
    return value


def _asset_inventory(
    value: Any,
    *,
    base: Path,
) -> tuple[list[dict[str, Any]], str]:
    payload, source = _document(value, base=base, label="asset_inventory")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise BatchCoverageError("asset_inventory.assets must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(assets):
        if not isinstance(raw, Mapping):
            raise BatchCoverageError(f"asset_inventory.assets[{index}] must be an object")
        asset_id = _nonempty(raw.get("asset_id"), f"asset_inventory.assets[{index}].asset_id")
        if asset_id in seen:
            raise BatchCoverageError(f"duplicate asset_id in inventory: {asset_id}")
        seen.add(asset_id)
        record = deepcopy(dict(raw))
        record["asset_id"] = asset_id
        record["entity_class"] = _asset_class(record.get("entity_class"))
        result.append(record)
    return result, source


def _room_catalog(
    value: Any,
    *,
    base: Path,
) -> tuple[list[dict[str, Any]], str]:
    payload, source = _document(value, base=base, label="room_catalog")
    rooms = payload.get("rooms")
    if not isinstance(rooms, list) or not rooms:
        raise BatchCoverageError("room_catalog.rooms must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rooms):
        if not isinstance(raw, Mapping):
            raise BatchCoverageError(f"room_catalog.rooms[{index}] must be an object")
        room_id = _nonempty(raw.get("room_id"), f"room_catalog.rooms[{index}].room_id")
        family = _nonempty(raw.get("family"), f"room_catalog.rooms[{index}].family")
        if room_id in seen:
            raise BatchCoverageError(f"duplicate room_id in catalog: {room_id}")
        seen.add(room_id)
        record = deepcopy(dict(raw))
        record["room_id"] = room_id
        record["family"] = family
        record["renderer"] = str(record.get("renderer") or "")
        result.append(record)
    return result, source


def _runtime_registry(
    value: Any,
    *,
    base: Path,
) -> tuple[dict[str, dict[str, Any]], str]:
    payload, source = _document(value, base=base, label="runtime_registry")
    records = payload.get("assets")
    if not isinstance(records, list):
        raise BatchCoverageError("runtime_registry.assets must be a list")
    result: dict[str, dict[str, Any]] = {}
    for raw in records:
        if isinstance(raw, Mapping) and isinstance(raw.get("asset_id"), str):
            result[raw["asset_id"]] = dict(raw)
    return result, source


def load_batch_input_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate an explicit facts/questions input manifest."""

    manifest_path = Path(path).expanduser().resolve()
    payload = _read_json(manifest_path)
    if payload.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise BatchCoverageError(
            f"unsupported input manifest schema: {payload.get('schema')!r}"
        )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise BatchCoverageError("input manifest episodes must be a list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(episodes):
        if not isinstance(raw, Mapping):
            raise BatchCoverageError(f"episodes[{index}] must be an object")
        missing = REQUIRED_FIVE_MANIFEST_FIELDS - set(raw)
        if missing:
            raise BatchCoverageError(
                f"episodes[{index}] missing required fields: {sorted(missing)}"
            )
        episode_id = _nonempty(raw["episode_id"], f"episodes[{index}].episode_id")
        if episode_id in seen:
            raise BatchCoverageError(f"duplicate episode_id: {episode_id}")
        seen.add(episode_id)
        facts = _resolve_path(raw["facts"], base=manifest_path.parent)
        questions = _resolve_path(raw["questions"], base=manifest_path.parent)
        if not facts.is_file() or not questions.is_file():
            raise BatchCoverageError(
                f"episode {episode_id} facts/questions inputs must exist"
            )
        room_id = _nonempty(raw["room_id"], f"episodes[{index}].room_id")
        family = _nonempty(raw["family"], f"episodes[{index}].family")
        source_room_id = str(raw.get("source_room_id") or room_id)
        normalized.append(
            {
                **deepcopy(dict(raw)),
                "episode_id": episode_id,
                "facts": str(facts),
                "questions": str(questions),
                "room_id": room_id,
                "source_room_id": source_room_id,
                "family": family,
                "source_refs": deepcopy(dict(raw.get("source_refs", {})))
                if isinstance(raw.get("source_refs"), Mapping)
                else {},
            }
        )
    result = deepcopy(dict(payload))
    result["episodes"] = normalized
    result["_manifest_path"] = str(manifest_path)
    return result


def _actor_map(facts: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = facts.get("actors", {})
    if isinstance(raw, Mapping):
        result = {}
        for actor_id, actor in raw.items():
            if isinstance(actor, Mapping):
                result[str(actor_id)] = deepcopy(dict(actor))
        return result
    if isinstance(raw, list):
        return {
            str(actor["actor_id"]): deepcopy(dict(actor))
            for actor in raw
            if isinstance(actor, Mapping) and isinstance(actor.get("actor_id"), str)
        }
    return {}


def _event_map(facts: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = facts.get("events", [])
    if isinstance(raw, Mapping):
        return {
            str(event_id): dict(event)
            for event_id, event in raw.items()
            if isinstance(event, Mapping)
        }
    return {
        str(event.get("event_id")): dict(event)
        for event in raw
        if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
    } if isinstance(raw, list) else {}


def _actor_appearance_value(actor: Mapping[str, Any]) -> Any:
    appearance = actor.get("appearance")
    if isinstance(appearance, Mapping) and appearance.get("value") is not None:
        return appearance.get("value")
    realized = actor.get("realized_attributes")
    if isinstance(realized, Mapping):
        top = realized.get("top_color")
        if top is not None:
            return top
        coat = realized.get("coat_profile")
        if isinstance(coat, Mapping) and coat.get("value") is not None:
            return coat.get("value")
        finish = realized.get("finish")
        if finish is not None:
            return finish
    return None


def _recursive_ids(value: Any, *, key_names: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in key_names:
                if isinstance(item, str):
                    found.add(item)
                elif isinstance(item, list):
                    found.update(str(v) for v in item if isinstance(v, str))
            found.update(_recursive_ids(item, key_names=key_names))
    elif isinstance(value, list):
        for item in value:
            found.update(_recursive_ids(item, key_names=key_names))
    return found


def _scope_for_item(
    evidence: Mapping[str, Any],
    *,
    actor_ids: set[str],
) -> tuple[str, set[str], set[str]]:
    explicit = set()
    for key in ("target_actor_id", "actor_id"):
        value = evidence.get(key)
        if isinstance(value, str) and value in actor_ids:
            explicit.add(value)
    if explicit:
        return "target", explicit, set()
    broad = _recursive_ids(
        evidence,
        key_names={
            "candidate_actor_ids",
            "appeared_actor_ids",
            "speaking_actor_ids",
            "actor_ids",
        },
    )
    if broad:
        return "global", set(), broad & actor_ids
    reference = _recursive_ids(
        evidence,
        key_names={
            "first_event",
            "reference_actor_id",
            "occluder_actor_id",
            "anchor_actor_id",
        },
    )
    if reference:
        return "reference", set(), reference & actor_ids
    return "global", set(), set()


def _supporting_evidence(
    item: Mapping[str, Any],
    *,
    actors: Mapping[str, Mapping[str, Any]],
    events: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, str]:
    if item.get("status") != "pass":
        return False, "item_status_not_pass"
    if not isinstance(item.get("question_id"), str) or not item.get("question_id"):
        return False, "missing_question_id"
    if not isinstance(item.get("forms"), Mapping) or not item["forms"]:
        return False, "missing_generated_forms"
    evidence = item.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        return False, "missing_item_evidence"
    ids = _recursive_ids(
        evidence,
        key_names={
            "actor_id",
            "target_actor_id",
            "candidate_actor_ids",
            "appeared_actor_ids",
            "speaking_actor_ids",
        },
    )
    missing_actors = sorted(actor_id for actor_id in ids if actor_id not in actors)
    if missing_actors:
        return False, f"evidence_actor_not_in_facts:{','.join(missing_actors)}"
    event_ids = _recursive_ids(evidence, key_names={"event_id", "event_ids"})
    missing_events = sorted(event_id for event_id in event_ids if event_id not in events)
    if missing_events:
        return False, f"evidence_event_not_in_facts:{','.join(missing_events)}"
    return True, "supported_by_facts_and_generated_item"


def _episode_context(
    episode: Mapping[str, Any],
    *,
    asset_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    facts = _read_json(Path(episode["facts"]))
    questions = _read_json(Path(episode["questions"]))
    if not isinstance(facts, Mapping) or not isinstance(questions, Mapping):
        raise BatchCoverageError(f"episode {episode['episode_id']} facts/questions must be objects")
    expected_episode_id = str(episode["episode_id"])
    if facts.get("episode_id") != expected_episode_id:
        raise BatchCoverageError(
            f"episode {expected_episode_id} facts episode_id does not agree"
        )
    if questions.get("episode_id") != expected_episode_id:
        raise BatchCoverageError(
            f"episode {expected_episode_id} questions episode_id does not agree"
        )
    if facts.get("status") != "pass":
        raise BatchCoverageError(
            f"episode {expected_episode_id} facts status is not pass"
        )
    actors = _actor_map(facts)
    events = _event_map(facts)
    # Optional source records come from the actual audio receipt. Keep stage
    # paths distinct and never attribute another actor's sound to an unsampled
    # asset merely because they share a room.
    for source in episode.get("sound_events", []):
        if not isinstance(source, Mapping) or source.get("event_id") not in events:
            raise BatchCoverageError("sound source record references an unknown event")
        event = events[source["event_id"]]
        for key in ("actor_id", "sound_asset_id"):
            if source.get(key) is not None and source[key] != event.get(key):
                raise BatchCoverageError("sound source identity disagrees with factual event")
        dry = source.get("dry_audio_origin")
        dry = deepcopy(dict(dry)) if isinstance(dry, Mapping) else None
        original = source.get("original_source_path")
        speaker = source.get("speaker_id")
        dry_path = dry.get("path") if dry else None
        verified_dry = isinstance(dry_path, str) and Path(dry_path).is_file()
        verified_original = isinstance(original, str) and Path(original).is_file()
        if speaker is None and verified_dry:
            sidecar = Path(dry_path).with_name("clip.json")
            if sidecar.is_file():
                metadata = _read_json(sidecar)
                if isinstance(metadata, Mapping):
                    speaker = metadata.get("speaker_id")
        event.update(dry_audio_origin=dry, original_source_path=original,
                     speaker_id=speaker, source_status=(
                         "dry_source_file_verified" if verified_dry else
                         "original_source_file_verified_dry_path_unmeasured" if verified_original else "unmeasured"))
    actor_assets = {
        actor_id: actor.get("asset_id")
        for actor_id, actor in actors.items()
        if isinstance(actor.get("asset_id"), str)
    }
    actor_classes = {
        actor_id: _asset_class(asset_by_id.get(asset_id, {}).get("entity_class"))
        for actor_id, asset_id in actor_assets.items()
    }
    coverage_entries = questions.get("coverage", [])
    coverage_by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if isinstance(coverage_entries, list):
        for entry in coverage_entries:
            if isinstance(entry, Mapping) and isinstance(entry.get("qa_id"), str):
                coverage_by_qa[entry["qa_id"]].append(dict(entry))
    items_by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in questions.get("items", []) if isinstance(questions.get("items"), list) else []:
        if isinstance(item, Mapping) and isinstance(item.get("qa_id"), str):
            items_by_qa[item["qa_id"]].append(dict(item))
    outputs: list[dict[str, Any]] = []
    global_outputs: list[dict[str, Any]] = []
    reference_outputs: list[dict[str, Any]] = []
    invalid_items: list[dict[str, Any]] = []
    for qa_id, items in items_by_qa.items():
        for item in items:
            evidence = item.get("evidence")
            evidence = evidence if isinstance(evidence, Mapping) else {}
            supported, support_reason = _supporting_evidence(
                item, actors=actors, events=events
            )
            scope, targets, refs = _scope_for_item(
                evidence, actor_ids=set(actors)
            )
            outcome = {
                "episode_id": episode["episode_id"],
                "qa_id": qa_id,
                "question_id": item.get("question_id"),
                "scope": scope,
                "target_actor_ids": sorted(targets),
                "reference_actor_ids": sorted(refs),
                "target_asset_ids": sorted(
                    actor_assets.get(actor_id)
                    for actor_id in targets
                    if actor_assets.get(actor_id)
                ),
                "reference_asset_ids": sorted(
                    actor_assets.get(actor_id)
                    for actor_id in refs
                    if actor_assets.get(actor_id)
                ),
                "source_refs": {
                    "facts": episode["facts"],
                    "questions": episode["questions"],
                },
                "sound_origins": [
                    {
                        key: event.get(key)
                        for key in (
                            "event_id",
                            "actor_id",
                            "source_endpoint_id",
                            "sound_asset_id",
                            "sound_class",
                        )
                        if event.get(key) is not None
                    }
                    for event in events.values()
                    if isinstance(event, Mapping)
                ],
            }
            if supported:
                outcome.update({"state": "produced", "reason_code": support_reason})
                outputs.append(outcome)
                if scope == "global":
                    global_outputs.append(outcome)
                elif scope == "reference":
                    reference_outputs.append(outcome)
            else:
                outcome.update(
                    {
                        "state": "evidence_missing_or_unsampled",
                        "reason_code": support_reason,
                    }
                )
                invalid_items.append(outcome)
    origin_context = {"episode": episode, "facts": facts, "events": events, "actor_assets": actor_assets}
    for outcome in outputs:
        origins = _sound_origins_for_context(origin_context, outcomes=[outcome])
        outcome["context_sound_origins"] = origins
        scoped_assets = set(outcome["target_asset_ids"]) | set(outcome["reference_asset_ids"])
        outcome["sound_origins"] = [origin for origin in origins if
            outcome["scope"] == "global" or origin["asset_id"] in scoped_assets]
    return {
        "episode": dict(episode),
        "facts": facts,
        "questions": questions,
        "actors": actors,
        "events": events,
        "actor_assets": actor_assets,
        "actor_classes": actor_classes,
        "coverage_by_qa": dict(coverage_by_qa),
        "items_by_qa": dict(items_by_qa),
        "outputs": outputs,
        "global_outputs": global_outputs,
        "reference_outputs": reference_outputs,
        "invalid_items": invalid_items,
    }




def _failed_episode_index(payload: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index failed episode gap_state by (room_id, asset_id)."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    raw = payload.get("failed_episodes")
    if not isinstance(raw, list):
        return index
    rank = {"interface_not_implemented": 2, "evidence_missing_or_unsampled": 1}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        room_id = item.get("room_id")
        gap_state = item.get("gap_state")
        if not isinstance(room_id, str) or gap_state not in rank:
            continue
        asset_ids = item.get("asset_ids")
        if not isinstance(asset_ids, list) or not asset_ids:
            continue
        record = {
            "episode_id": item.get("episode_id"),
            "room_id": room_id,
            "gap_state": gap_state,
            "failure_stage": item.get("failure_stage") or "unknown",
            "failure_reason": item.get("failure_reason") or item.get("reason") or gap_state,
        }
        for asset_id in asset_ids:
            if not isinstance(asset_id, str) or not asset_id:
                continue
            key = (room_id, asset_id)
            current = index.get(key)
            if current is None or rank[gap_state] > rank[current["gap_state"]]:
                index[key] = record
    return index


APPEARANCE_CLASSIFIER_GAP_REASON = (
    "registered_appearance_value_classifier_not_implemented"
)
APPEARANCE_DEFER_CODES = frozenset(
    {
        "appearance_review_missing",
        "appearance_not_unique",
        "missing_appearance",
        APPEARANCE_CLASSIFIER_GAP_REASON,
    }
)


def _appearance_classifier_gap_reason(
    contexts: Sequence[Mapping[str, Any]],
    asset_id: str,
) -> str | None:
    """Return the classifier-gap reason if this asset's appearance review recorded one."""
    for context in contexts:
        actors = context.get("actors") if isinstance(context.get("actors"), Mapping) else {}
        facts = context.get("facts") if isinstance(context.get("facts"), Mapping) else {}
        review = facts.get("appearance_review") if isinstance(facts, Mapping) else None
        review = review if isinstance(review, Mapping) else {}
        for actor_id, actor in actors.items():
            if not isinstance(actor, Mapping) or actor.get("asset_id") != asset_id:
                continue
            row = review.get(actor_id)
            if not isinstance(row, Mapping):
                continue
            if row.get("reason") == APPEARANCE_CLASSIFIER_GAP_REASON:
                return APPEARANCE_CLASSIFIER_GAP_REASON
            checks = row.get("checks")
            if not isinstance(checks, list):
                continue
            for check in checks:
                if (
                    isinstance(check, Mapping)
                    and check.get("reason") == APPEARANCE_CLASSIFIER_GAP_REASON
                ):
                    return APPEARANCE_CLASSIFIER_GAP_REASON
    return None


def _deferred_state(reason_code: Any) -> str:
    code = str(reason_code or "deferred_by_rule")
    if code in {APPEARANCE_CLASSIFIER_GAP_REASON, "interface_not_implemented"}:
        return "interface_not_implemented"
    if code.startswith("missing_") or code == "event_segmentation_not_reviewed":
        return "evidence_missing_or_unsampled"
    return "deferred_by_rule"

def _asset_interface(
    asset: Mapping[str, Any],
    room: Mapping[str, Any],
    runtime_by_id: Mapping[str, Mapping[str, Any]],
    *,
    runtime_path: str,
) -> dict[str, Any] | None:
    asset_id = str(asset["asset_id"])
    record = runtime_by_id.get(asset_id)
    renderer = room.get("renderer")
    backend_key = "spear_unreal" if renderer == "ue_spear" else "habitat"
    if not isinstance(record, Mapping):
        return {
            "reason_code": "missing_renderer_binding",
            "reason": f"{asset_id} has no runtime registry record",
            "files": [runtime_path],
        }
    backends = record.get("runtime_backends")
    binding = backends.get(backend_key) if isinstance(backends, Mapping) else None
    if not isinstance(binding, Mapping):
        return {
            "reason_code": "missing_renderer_binding",
            "reason": f"{asset_id} has no {backend_key} binding for {room['room_id']}",
            "files": [runtime_path],
        }
    if _asset_class(asset.get("entity_class")) in STATIC_CLASS_NAMES:
        habitat = backends.get("habitat") if isinstance(backends, Mapping) else None
        resting = habitat.get("resting_pose") if isinstance(habitat, Mapping) else None
        surface = resting.get("attachment_surface") if isinstance(resting, Mapping) else "unknown"
        if surface in {"wall", "ceiling"}:
            return {
                "reason_code": "static_attachment_surface_not_implemented",
                "reason": (
                    f"{asset_id} requires {surface} placement; current static "
                    "placement interface is floor-only"
                ),
                "files": [
                    "src/avengine/rooms/qa_episode.py",
                    "tools/rooms/run_spear_apartment_canary.py",
                ],
            }
    return None


def _not_applicable_reason(
    asset: Mapping[str, Any],
    qa_id: str,
    *,
    speech_asset_ids: set[str] | None = None,
) -> str | None:
    asset_id = str(asset.get("asset_id"))
    asset_class = _asset_class(asset.get("entity_class"))
    if asset_class in STATIC_CLASS_NAMES and qa_id in MOTION_TARGET_QA_IDS:
        return "static_device_is_not_a_motion_target_by_definition"
    if qa_id == TRANSCRIPT_QA_ID and asset_class == "articulated_animal":
        return "transcript_question_is_not_applicable_to_animals"
    if (
        qa_id == TRANSCRIPT_QA_ID
        and asset_class in STATIC_CLASS_NAMES
        and asset_id not in (speech_asset_ids or set())
    ):
        allowed = asset.get("allowed_event_classes")
        allowed = allowed if isinstance(allowed, (list, tuple, set)) else []
        category = str(asset.get("category") or "")
        known_speech_device = (
            category == "audio_playback"
            or any(
                isinstance(value, str)
                and any(
                    token in value.lower()
                    for token in ("speech", "spoken", "voice")
                )
                for value in allowed
            )
        )
        if not known_speech_device and allowed:
            return "transcript_question_is_not_applicable_to_this_source_class"
    return None


def _flatten_source_refs(
    episode_contexts: Sequence[Mapping[str, Any]],
    *,
    asset_id: str,
    room_id: str,
    qa_id: str,
) -> dict[str, Any]:
    refs = []
    for context in episode_contexts:
        episode = context["episode"]
        refs.append(
            {
                "episode_id": episode["episode_id"],
                "facts": episode["facts"],
                "questions": episode["questions"],
                "source_room_id": episode.get("source_room_id", room_id),
            }
        )
    return {
        "asset_id": asset_id,
        "room_id": room_id,
        "qa_id": qa_id,
        "episodes": refs,
    }


def _sound_origins_for_context(
    context: Mapping[str, Any],
    *,
    outcomes: Sequence[Mapping[str, Any]],
    asset_id: str | None = None,
) -> list[dict[str, Any]]:
    events = context.get("events", {})
    facts = context.get("facts", {})
    audio = facts.get("audio") if isinstance(facts, Mapping) else {}
    audio = audio if isinstance(audio, Mapping) else {}
    ids = {
        event_id
        for outcome in outcomes
        for event_id in _recursive_ids(
            outcome,
            key_names={"event_id", "event_ids"},
        )
    }
    if not ids:
        ids = set(events)
    audio_refs = {
        "mixture_path": audio.get("path"),
        "source_mix": audio.get("source_mix"),
        "hrtf_id": audio.get("hrtf_id"),
    }
    return [
        {
            "episode_id": context.get("episode", {}).get("episode_id"),
            "event_id": event_id,
            "actor_id": event.get("actor_id"),
            "asset_id": context.get("actor_assets", {}).get(event.get("actor_id")),
            "dry_audio_origin": event.get("dry_audio_origin"),
            "original_source_path": event.get("original_source_path"),
            "speaker_id": event.get("speaker_id"),
            "source_status": event.get("source_status", "unmeasured"),
            "source_endpoint_id": event.get("source_endpoint_id"),
            "sound_asset_id": event.get("sound_asset_id"),
            "sound_class": event.get("sound_class"),
            "source_record": event.get("source_record"),
            "audio_refs": audio_refs,
        }
        for event_id, event in events.items()
        if event_id in ids and (asset_id is None or
            context.get("actor_assets", {}).get(event.get("actor_id")) == asset_id)
    ]


def _structural_report(
    contexts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate the exact per-form structure emitted by P8.

    The P8 generator already computed candidate counterfactual values for each
    QA/form. Recomputing from appearance would silently apply one visual
    attribute to direction, motion, or timing questions, so missing structure
    remains unmeasured.
    """

    item_rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    validation_errors: list[dict[str, Any]] = []
    for context in contexts:
        for output in context["outputs"]:
            qa_id = output["qa_id"]
            item = next(
                (
                    candidate
                    for candidate in context["items_by_qa"].get(qa_id, [])
                    if candidate.get("question_id") == output.get("question_id")
                ),
                None,
            )
            if not isinstance(item, Mapping):
                continue
            structure = item.get("structure")
            forms = item.get("forms")
            for form in ("mcq", "open"):
                baseline = structure.get(form) if isinstance(structure, Mapping) else None
                effective_k: int | None = None
                if form == "mcq" and isinstance(forms, Mapping):
                    payload = forms.get("mcq")
                    options = payload.get("options", []) if isinstance(payload, Mapping) else []
                    effective_k = len(options) if isinstance(options, list) else None
                reason = None
                if not isinstance(baseline, Mapping):
                    reason = "missing_p8_structure"
                elif baseline.get("status") != "measured":
                    reason = str(baseline.get("reason") or "p8_structure_unmeasured")
                elif form == "mcq" and baseline.get("k") != effective_k:
                    reason = "p8_mcq_k_does_not_match_actual_options"
                    validation_errors.append(
                        {
                            "episode_id": output["episode_id"],
                            "qa_id": qa_id,
                            "question_id": output.get("question_id"),
                            "form": form,
                            "structure_k": baseline.get("k"),
                            "actual_k": effective_k,
                        }
                    )
                status = "measured" if reason is None else "unmeasured"
                effective_baseline = deepcopy(dict(baseline)) if isinstance(baseline, Mapping) else {}
                effective_baseline["effective_k"] = effective_k
                effective_baseline["chance_calibration"] = (
                    "explicit_mcq_answer_domain"
                    if form == "mcq"
                    else "uncalibrated_open_candidate_entity"
                )
                item_rows.append(
                    {
                        "episode_id": output["episode_id"],
                        "qa_id": qa_id,
                        "question_id": output.get("question_id"),
                        "target_asset_id": (
                            output.get("target_asset_ids", [None])[0]
                            if output.get("target_asset_ids")
                            else None
                        ),
                        "form": form,
                        "effective_k": effective_k,
                        "status": status,
                        "reason": reason,
                        "baseline": effective_baseline,
                    }
                )
                key = (qa_id, form, effective_k)
                summary = aggregate.setdefault(
                    key,
                    {
                        "qa_id": qa_id,
                        "form": form,
                        "k": effective_k,
                        "rows": 0,
                        "measured_rows": 0,
                        "majority_available_rows": 0,
                        "unique_minority_available_rows": 0,
                        "majority_hits_all": 0.0,
                        "unique_minority_hits_all": 0.0,
                        "random_expected_hits_all": 0.0,
                        "majority_hits_available": 0.0,
                        "unique_minority_hits_available": 0.0,
                    },
                )
                summary["rows"] += 1
                if status != "measured":
                    continue
                summary["measured_rows"] += 1
                for name, all_key, available_key in (
                    (
                        "majority",
                        "majority_hits_all",
                        "majority_hits_available",
                    ),
                    (
                        "unique_minority",
                        "unique_minority_hits_all",
                        "unique_minority_hits_available",
                    ),
                ):
                    available = bool(baseline.get(f"{name}_available"))
                    value = baseline.get(f"{name}_hits")
                    if isinstance(value, (int, float)):
                        summary[all_key] += float(value)
                        if available:
                            summary[available_key] += float(value)
                    if available:
                        summary[f"{name}_available_rows"] += 1
                random_value = baseline.get("random_hits")
                if isinstance(random_value, (int, float)):
                    summary["random_expected_hits_all"] += float(random_value)
    for summary in aggregate.values():
        measured = int(summary["measured_rows"])
        for name, all_key in (
            ("majority", "majority_hits_all"),
            ("unique_minority", "unique_minority_hits_all"),
        ):
            available_rows = int(summary[f"{name}_available_rows"])
            summary[f"{name}_rate_all"] = (
                summary[all_key] / measured if measured else None
            )
            summary[f"{name}_rate_available"] = (
                summary[f"{name}_hits_available"] / available_rows
                if available_rows
                else None
            )
        summary["random_rate_all"] = (
            summary["random_expected_hits_all"] / measured if measured else None
        )
        summary["random_rate_calibration"] = (
            "explicit_mcq_answer_domain"
            if summary["form"] == "mcq"
            else "uncalibrated_open_candidate_entity"
        )
    return {
        "schema": "avengine_qa_structural_baselines_v1",
        "structure_source": "questions.items[].structure",
        "by_qa_form_k": sorted(
            aggregate.values(),
            key=lambda value: (
                value["qa_id"],
                value["form"],
                -1 if value["k"] is None else value["k"],
            ),
        ),
        "items": item_rows,
        "validation_errors": validation_errors,
        "claim_boundary": (
            "Structural values are batch diagnostics. They are not model "
            "outcomes. MCQ random rates use the actual option domain; Open "
            "random rates are uncalibrated and are not a 25 percent claim."
        ),
    }


def _axis_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    by_qa: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (
            str(row["asset_class"]),
            str(row["room_family"]),
            str(row["room_id"]),
        )
        grouped[key][str(row["state"])] += 1
        by_qa[str(row["qa_id"])][str(row["state"])] += 1
    def materialize(grouped_value: Mapping[Any, Counter[str]]) -> list[dict[str, Any]]:
        result = []
        def sort_key(item: tuple[Any, Counter[str]]) -> tuple[str, ...]:
            key = item[0]
            return tuple(str(value) for value in key) if isinstance(key, tuple) else (str(key),)
        for key, counts in sorted(grouped_value.items(), key=sort_key):
            if isinstance(key, tuple):
                entry = {
                    "asset_class": key[0],
                    "room_family": key[1],
                    "room_id": key[2],
                }
            else:
                entry = {"qa_id": str(key)}
            entry["row_count"] = int(sum(counts.values()))
            entry["states"] = {
                state: int(counts.get(state, 0)) for state in COVERAGE_STATES
            }
            result.append(entry)
        return result
    return {
        "by_asset_class_family_room": materialize(grouped),
        "by_qa": materialize(by_qa),
    }


def _joint_summary(
    rows: Sequence[Mapping[str, Any]],
    assets: Sequence[Mapping[str, Any]],
    rooms: Sequence[Mapping[str, Any]],
    global_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the complete class x room x QA joint grid (3 x 7 x 24)."""

    classes = sorted({str(asset.get("entity_class")) for asset in assets})
    room_family = {str(room["room_id"]): str(room["family"]) for room in rooms}
    grouped: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[
            (
                str(row["asset_class"]),
                str(row["room_id"]),
                str(row["qa_id"]),
            )
        ][str(row["state"])] += 1
    global_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for outcome in global_outcomes:
        room_id = outcome.get("room_id")
        qa_id = outcome.get("qa_id")
        if not isinstance(room_id, str) or not isinstance(qa_id, str):
            continue
        question_id = outcome.get("question_id")
        participants = outcome.get("participating_asset_classes", [])
        participants = (
            [str(value) for value in participants]
            if isinstance(participants, list)
            else []
        )
        for asset_class in set(participants):
            key = (asset_class, room_id, qa_id)
            entry = global_by_key.setdefault(
                key,
                {
                    "global_produced": 0,
                    "global_question_ids": set(),
                    "global_episode_ids": set(),
                },
            )
            entry["global_produced"] += 1
            if isinstance(question_id, str):
                entry["global_question_ids"].add(question_id)
            if isinstance(outcome.get("episode_id"), str):
                entry["global_episode_ids"].add(outcome["episode_id"])
    cells = []
    for asset_class in classes:
        for room in rooms:
            room_id = str(room["room_id"])
            for qa_id in QA_IDS:
                counts = grouped[(asset_class, room_id, qa_id)]
                global_entry = global_by_key.get(
                    (asset_class, room_id, qa_id),
                    {
                        "global_produced": 0,
                        "global_question_ids": set(),
                        "global_episode_ids": set(),
                    },
                )
                cells.append(
                    {
                        "asset_class": asset_class,
                        "room_family": room_family[room_id],
                        "room_id": room_id,
                        "qa_id": qa_id,
                        "row_count": int(sum(counts.values())),
                        "states": {
                            state: int(counts.get(state, 0))
                            for state in COVERAGE_STATES
                        },
                        "global_produced": int(global_entry["global_produced"]),
                        "global_question_ids": sorted(
                            global_entry["global_question_ids"]
                        ),
                        "global_episode_ids": sorted(
                            global_entry["global_episode_ids"]
                        ),
                    }
                )
    return {
        "row_count": len(cells),
        "asset_class_count": len(classes),
        "room_count": len(rooms),
        "qa_type_count": len(QA_IDS),
        "cells": cells,
        "claim_boundary": (
            "Joint cells aggregate target/reference asset rows by class and "
            "room, while global questions are counted once per participating "
            "class and semantic question_id."
        ),
    }


def validate_batch_coverage(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate state exclusivity, stable denominator, and produced evidence."""

    if result.get("schema") != COVERAGE_SCHEMA:
        raise BatchCoverageError("unsupported coverage schema")
    rows = result.get("rows")
    denominator = result.get("denominator")
    if not isinstance(rows, list) or not isinstance(denominator, Mapping):
        raise BatchCoverageError("coverage must contain rows and denominator")
    expected = (
        int(denominator.get("asset_count", 0))
        * int(denominator.get("room_count", 0))
        * int(denominator.get("qa_type_count", 0))
    )
    if len(rows) != expected or int(denominator.get("row_count", -1)) != expected:
        raise BatchCoverageError(
            f"coverage denominator mismatch: rows={len(rows)} expected={expected}"
        )
    expected_asset_ids = set(str(value) for value in denominator.get("asset_ids", []))
    expected_room_ids = set(str(value) for value in denominator.get("room_ids", []))
    expected_qa_ids = set(str(value) for value in denominator.get("qa_ids", []))
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise BatchCoverageError(f"coverage row {index} must be an object")
        state = row.get("state")
        if state not in COVERAGE_STATES:
            raise BatchCoverageError(f"coverage row {index} has invalid state: {state!r}")
        key = (
            str(row.get("asset_id")),
            str(row.get("room_id")),
            str(row.get("qa_id")),
        )
        if expected_asset_ids and key[0] not in expected_asset_ids:
            raise BatchCoverageError(f"row {index} references an unknown asset_id")
        if expected_room_ids and key[1] not in expected_room_ids:
            raise BatchCoverageError(f"row {index} references an unknown room_id")
        if expected_qa_ids and key[2] not in expected_qa_ids:
            raise BatchCoverageError(f"row {index} references an unknown qa_id")
        if key in seen:
            raise BatchCoverageError(f"duplicate detailed coverage row: {key}")
        seen.add(key)
        if state == "produced":
            if not row.get("question_ids") or not row.get("evidence_refs"):
                raise BatchCoverageError(
                    f"produced row {key} lacks generated item evidence"
                )
    if len(seen) != expected:
        raise BatchCoverageError("coverage rows do not form a complete Cartesian denominator")
    return dict(result)


def build_batch_coverage(
    manifest: str | Path | Mapping[str, Any],
    *,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Build the five-state full-scope coverage table."""

    if isinstance(manifest, (str, Path)):
        payload = load_batch_input_manifest(manifest)
        manifest_path = Path(payload["_manifest_path"])
    elif isinstance(manifest, Mapping):
        payload = deepcopy(dict(manifest))
        manifest_path = Path.cwd()
        payload.setdefault("schema", INPUT_MANIFEST_SCHEMA)
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise BatchCoverageError("inline manifest episodes must be a list")
        normalized = []
        for raw in episodes:
            if not isinstance(raw, Mapping):
                raise BatchCoverageError("inline episode must be an object")
            copy = deepcopy(dict(raw))
            copy["facts"] = str(copy["facts"])
            copy["questions"] = str(copy["questions"])
            normalized.append(copy)
        payload["episodes"] = normalized
    else:
        raise BatchCoverageError("manifest must be a path or object")
    repository_path = Path(repository or manifest_path.parent).expanduser().resolve()
    inventory_value = payload.get(
        "asset_inventory",
        repository_path / "tmp/qa_generalized_sampler_review_20260906_v1/full_source_scope_inventory.json",
    )
    catalog_value = payload.get(
        "room_catalog",
        repository_path / "examples/rooms/packages/catalog.json",
    )
    registry_value = payload.get(
        "runtime_registry",
        repository_path / "examples/runtime/source_asset_runtime_profiles.json",
    )
    assets, inventory_source = _asset_inventory(inventory_value, base=manifest_path.parent)
    rooms, catalog_source = _room_catalog(catalog_value, base=manifest_path.parent)
    runtime_by_id, runtime_source = _runtime_registry(
        registry_value, base=manifest_path.parent
    )
    # The review inventory is the external 44-ID slice. The shipped runtime
    # registry carries the 17 runtime IDs. Merge them into the required 59-ID
    # union without inventing attributes for the runtime-only records.
    known_asset_ids = {asset["asset_id"] for asset in assets}
    for asset_id, record in sorted(runtime_by_id.items()):
        if asset_id in known_asset_ids:
            continue
        identity = record.get("identity") if isinstance(record, Mapping) else {}
        attributes = (
            record.get("realized_attributes")
            if isinstance(record, Mapping)
            else {}
        )
        category = None
        if isinstance(identity, Mapping):
            category = identity.get("category") or identity.get("object_type")
        if category is None and isinstance(attributes, Mapping):
            category = attributes.get("category") or attributes.get("object_type")
        assets.append(
            {
                "asset_id": asset_id,
                "entity_class": _asset_class(record.get("entity_class")),
                "category": category or record.get("display_label") or "runtime_asset",
                "allowed_event_classes": [],
                "in_runtime_registry": True,
            }
        )
    assets.sort(key=lambda value: str(value["asset_id"]))
    room_by_id = {room["room_id"]: room for room in rooms}
    episodes = []
    for raw in payload["episodes"]:
        copy = deepcopy(dict(raw))
        copy["facts"] = str(_resolve_path(copy["facts"], base=manifest_path.parent))
        copy["questions"] = str(_resolve_path(copy["questions"], base=manifest_path.parent))
        room_id = str(copy["room_id"])
        room = room_by_id.get(room_id)
        if room is None:
            raise BatchCoverageError(
                f"episode {copy.get('episode_id')} references unknown room_id {room_id}"
            )
        if str(copy.get("family")) != str(room.get("family")):
            raise BatchCoverageError(
                f"episode {copy.get('episode_id')} family does not agree with room catalog"
            )
        episodes.append(copy)
    asset_by_id = {a["asset_id"]: a for a in assets}
    contexts = [
        _episode_context(episode, asset_by_id=asset_by_id)
        for episode in episodes
    ]
    failed_index = _failed_episode_index(payload)
    episodes_by_room: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        episodes_by_room[context["episode"]["room_id"]].append(context)
    rows: list[dict[str, Any]] = []
    global_outcomes: list[dict[str, Any]] = []
    reference_outcomes: list[dict[str, Any]] = []
    for context in contexts:
        episode = context["episode"]
        participating_classes = sorted(
            {
                context["actor_classes"].get(actor_id)
                for actor_id in context["actor_assets"]
                if context["actor_classes"].get(actor_id)
            }
        )
        for outcome in context["global_outputs"]:
            global_outcomes.append(
                {
                    **outcome,
                    "room_id": episode["room_id"],
                    "room_family": episode["family"],
                    "participating_asset_classes": participating_classes,
                }
            )
        for outcome in context["reference_outputs"]:
            reference_outcomes.append(
                {
                    **outcome,
                    "room_id": episode["room_id"],
                    "room_family": episode["family"],
                }
            )
    for asset in assets:
        asset_id = asset["asset_id"]
        for room in rooms:
            room_id = room["room_id"]
            room_contexts = episodes_by_room.get(room_id, [])
            room_sound_origins = [
                origin
                for context in room_contexts
                for origin in _sound_origins_for_context(context, outcomes=[])
            ]
            for qa_id in QA_IDS:
                source_refs = _flatten_source_refs(
                    room_contexts,
                    asset_id=asset_id,
                    room_id=room_id,
                    qa_id=qa_id,
                )
                interface = _asset_interface(
                    asset,
                    room,
                    runtime_by_id,
                    runtime_path=runtime_source,
                )
                speech_asset_ids: set[str] = set()
                for context in room_contexts:
                    for event in context["events"].values():
                        if not isinstance(event, Mapping):
                            continue
                        sound_class = str(event.get("sound_class") or "").lower()
                        if sound_class == "speech" or "transcript" in event:
                            actor_id = event.get("actor_id")
                            event_asset_id = context["actor_assets"].get(actor_id)
                            if isinstance(event_asset_id, str):
                                speech_asset_ids.add(event_asset_id)
                not_applicable = _not_applicable_reason(
                    asset,
                    qa_id,
                    speech_asset_ids=speech_asset_ids,
                )
                if not_applicable is not None:
                    rows.append(
                        {
                            "asset_id": asset_id,
                            "asset_class": asset.get("entity_class"),
                            "asset_category": asset.get("category"),
                            "room_id": room_id,
                            "room_family": room["family"],
                            "renderer": room.get("renderer"),
                            "qa_id": qa_id,
                            "state": "not_applicable_by_definition",
                            "reason_code": not_applicable,
                            "reason": not_applicable,
                            "scope": "target",
                            "target_asset_ids": [asset_id],
                            "reference_asset_ids": [],
                            "episode_ids": [],
                            "question_ids": [],
                            "evidence_refs": [],
                            "sound_origins": [],
                            "context_sound_origins": room_sound_origins,
                            "interface_gap": interface,
                            "source_refs": source_refs,
                        }
                    )
                    continue
                if interface is not None:
                    rows.append(
                        {
                            "asset_id": asset_id,
                            "asset_class": asset.get("entity_class"),
                            "asset_category": asset.get("category"),
                            "room_id": room_id,
                            "room_family": room["family"],
                            "renderer": room.get("renderer"),
                            "qa_id": qa_id,
                            "state": "interface_not_implemented",
                            "reason_code": interface["reason_code"],
                            "reason": interface["reason"],
                            "scope": "none",
                            "target_asset_ids": [],
                            "reference_asset_ids": [],
                            "episode_ids": [],
                            "question_ids": [],
                            "evidence_refs": [],
                            "sound_origins": [],
                            "context_sound_origins": room_sound_origins,
                            "interface_gap": interface,
                            "source_refs": {
                                **source_refs,
                                "interface_files": interface["files"],
                            },
                        }
                    )
                    continue
                produced: list[dict[str, Any]] = []
                deferred: list[dict[str, Any]] = []
                missing_reason = "no_episode_input_for_room"
                for context in room_contexts:
                    actor_asset_ids = set(context["actor_assets"].values())
                    if asset_id not in actor_asset_ids:
                        missing_reason = "asset_not_in_episode"
                        continue
                    candidates = [
                        outcome
                        for outcome in context["outputs"]
                        if outcome["qa_id"] == qa_id
                        and (
                            (
                                outcome.get("scope") == "target"
                                and asset_id in outcome.get("target_asset_ids", [])
                            )
                            or (
                                outcome.get("scope") == "reference"
                                and asset_id in outcome.get("reference_asset_ids", [])
                            )
                        )
                    ]
                    produced.extend(candidates)
                    if candidates:
                        continue
                    coverage = context["coverage_by_qa"].get(qa_id, [])
                    deferred_entries = [
                        entry for entry in coverage if entry.get("status") == "deferred"
                    ]
                    if deferred_entries:
                        deferred.extend(deferred_entries)
                        missing_reason = "deferred_by_rule"
                    elif context["global_outputs"] and any(
                        outcome["qa_id"] == qa_id for outcome in context["global_outputs"]
                    ):
                        missing_reason = "global_question_not_target_asset"
                    else:
                        invalid = [
                            item
                            for item in context["invalid_items"]
                            if item["qa_id"] == qa_id
                        ]
                        missing_reason = (
                            invalid[0]["reason_code"]
                            if invalid
                            else "question_not_generated_for_target_asset"
                        )
                if produced:
                    first = produced[0]
                    rows.append(
                        {
                            "asset_id": asset_id,
                            "asset_class": asset.get("entity_class"),
                            "asset_category": asset.get("category"),
                            "room_id": room_id,
                            "room_family": room["family"],
                            "renderer": room.get("renderer"),
                            "qa_id": qa_id,
                            "state": "produced",
                            "reason_code": "generated_item_with_supporting_evidence",
                            "reason": "generated target-scoped item and supporting facts exist",
                            "scope": first["scope"],
                            "target_asset_ids": first.get("target_asset_ids", []),
                            "reference_asset_ids": first.get("reference_asset_ids", []),
                            "episode_ids": sorted({item["episode_id"] for item in produced}),
                            "question_ids": sorted(
                                {
                                    item["question_id"]
                                    for item in produced
                                    if item.get("question_id")
                                }
                            ),
                            "evidence_refs": produced,
                            "sound_origins": [
                                origin
                                for context in room_contexts
                                for origin in _sound_origins_for_context(
                                    context, outcomes=produced, asset_id=asset_id
                                )
                            ],
                            "interface_gap": None,
                            "source_refs": source_refs,
                        }
                    )
                elif deferred:
                    first = deferred[0]
                    reason_code = first.get("code", "deferred_by_rule")
                    deferred_state = _deferred_state(reason_code)
                    classifier_reason = None
                    if str(reason_code) in APPEARANCE_DEFER_CODES:
                        classifier_reason = _appearance_classifier_gap_reason(
                            room_contexts, asset_id
                        )
                        if classifier_reason:
                            deferred_state = "interface_not_implemented"
                            reason_code = classifier_reason
                    rows.append(
                        {
                            "asset_id": asset_id,
                            "asset_class": asset.get("entity_class"),
                            "asset_category": asset.get("category"),
                            "room_id": room_id,
                            "room_family": room["family"],
                            "renderer": room.get("renderer"),
                            "qa_id": qa_id,
                            "state": deferred_state,
                            "reason_code": reason_code,
                            "reason": (
                                classifier_reason
                                if classifier_reason
                                else first.get("detail", "question was legally deferred")
                            ),
                            "scope": "target",
                            "target_asset_ids": [asset_id],
                            "reference_asset_ids": [],
                            "episode_ids": sorted(
                                {
                                    context["episode"]["episode_id"]
                                    for context in room_contexts
                                    if asset_id in set(context["actor_assets"].values())
                                }
                            ),
                            "question_ids": [],
                            "evidence_refs": [],
                            "sound_origins": [
                                origin
                                for context in room_contexts
                                for origin in _sound_origins_for_context(
                                    context, outcomes=[], asset_id=asset_id
                                )
                            ],
                            "interface_gap": None,
                            "source_refs": source_refs,
                        }
                    )
                else:
                    failed = failed_index.get((room_id, asset_id))
                    if failed is not None:
                        stage = str(failed.get("failure_stage") or "unknown")
                        reason_text = str(failed.get("failure_reason") or failed["gap_state"])
                        rows.append(
                            {
                                "asset_id": asset_id,
                                "asset_class": asset.get("entity_class"),
                                "asset_category": asset.get("category"),
                                "room_id": room_id,
                                "room_family": room["family"],
                                "renderer": room.get("renderer"),
                                "qa_id": qa_id,
                                "state": failed["gap_state"],
                                "reason_code": f"episode_{stage}_failed",
                                "reason": f"{stage}: {reason_text}",
                                "scope": "none",
                                "target_asset_ids": [asset_id],
                                "reference_asset_ids": [],
                                "episode_ids": [failed["episode_id"]] if failed.get("episode_id") else [],
                                "question_ids": [],
                                "evidence_refs": [],
                                "sound_origins": [],
                                "context_sound_origins": room_sound_origins,
                                "interface_gap": None,
                                "source_refs": source_refs,
                                "failed_episode": deepcopy(failed),
                            }
                        )
                    else:
                        rows.append(
                            {
                                "asset_id": asset_id,
                                "asset_class": asset.get("entity_class"),
                                "asset_category": asset.get("category"),
                                "room_id": room_id,
                                "room_family": room["family"],
                                "renderer": room.get("renderer"),
                                "qa_id": qa_id,
                                "state": "evidence_missing_or_unsampled",
                                "reason_code": missing_reason,
                                "reason": (
                                    "No target-scoped generated item and supporting "
                                    f"evidence for {asset_id} in {room_id}/{qa_id}"
                                ),
                                "scope": "none",
                                "target_asset_ids": [],
                                "reference_asset_ids": [],
                                "episode_ids": sorted(
                                    {
                                        context["episode"]["episode_id"]
                                        for context in room_contexts
                                        if asset_id in set(context["actor_assets"].values())
                                    }
                                ),
                                "question_ids": [],
                                "evidence_refs": [],
                                "sound_origins": [],
                                "context_sound_origins": room_sound_origins,
                                "interface_gap": None,
                                "source_refs": source_refs,
                            }
                        )
    denominator = {
        "asset_count": len(assets),
        "room_count": len(rooms),
        "qa_type_count": len(QA_IDS),
        "row_count": len(assets) * len(rooms) * len(QA_IDS),
        "asset_ids": [asset["asset_id"] for asset in assets],
        "room_ids": [room["room_id"] for room in rooms],
        "qa_ids": list(QA_IDS),
        "asset_union": (
            "full_source_scope_inventory.unique_ids_union "
            "union runtime_registry.assets"
        ),
        "room_catalog": "examples/rooms/packages/catalog.json",
        "qa_catalog": "QA-01..QA-24",
    }
    summary = _axis_summary(rows)
    summary["joint_asset_class_family_room_qa"] = _joint_summary(
        rows, assets, rooms, global_outcomes
    )
    interface_counter = Counter(
        (
            row["asset_id"],
            row["room_id"],
            row.get("reason_code"),
        )
        for row in rows
        if row.get("interface_gap") is not None
    )
    summary["interface_gaps"] = [
        {
            "asset_id": key[0],
            "room_id": key[1],
            "reason_code": key[2],
            "row_count": count,
            "files": next(
                (
                    row["interface_gap"].get("files", [])
                    for row in rows
                    if row.get("interface_gap") is not None
                    and row["asset_id"] == key[0]
                    and row["room_id"] == key[1]
                    and row.get("reason_code") == key[2]
                ),
                [],
            ),
        }
        for key, count in sorted(interface_counter.items())
    ]
    result = {
        "schema": COVERAGE_SCHEMA,
        "status": "research_only",
        "denominator": denominator,
        "states": list(COVERAGE_STATES),
        "rows": rows,
        "global_outcomes": global_outcomes,
        "reference_outcomes": reference_outcomes,
        "episode_outcomes": [
            {
                "episode_id": context["episode"]["episode_id"],
                "room_id": context["episode"]["room_id"],
                "family": context["episode"]["family"],
                "facts": context["episode"]["facts"],
                "questions": context["episode"]["questions"],
                "counts": context["questions"].get("counts"),
                "coverage_summary": context["questions"].get("coverage_summary"),
                "coverage_by_qa": context["questions"].get("coverage_by_qa"),
                "candidate_counts": context["questions"].get("candidate_counts"),
                "unmet_quota_by_qa": context["questions"].get(
                    "unmet_quota_by_qa"
                ),
                "actual_evidence_summary": context["questions"].get(
                    "actual_evidence_summary"
                ),
                "invalid_items": context["invalid_items"],
            }
            for context in contexts
        ],
        "summaries": summary,
        "structural_baselines": _structural_report(contexts),
        "provenance": {
            "input_manifest": payload.get("_manifest_path"),
            "asset_inventory": inventory_source,
            "room_catalog": catalog_source,
            "runtime_registry": runtime_source,
            "inventory_asset_count": len(
                _asset_inventory(inventory_value, base=manifest_path.parent)[0]
            ),
            "runtime_asset_count": len(runtime_by_id),
            "merged_asset_count": len(assets),
            "episode_count": len(episodes),
            "episode_ids": [episode["episode_id"] for episode in episodes],
            "claim_boundary": (
                "Coverage is an evidence and interface accounting artifact. "
                "It is not a model evaluation, formal admission, or modality "
                "necessity certification."
            ),
        },
    }
    return validate_batch_coverage(result)


def write_batch_coverage(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON, CSV, structural baseline, and provenance artifacts."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing coverage output: {output}")
    output.mkdir(parents=True)
    coverage = output / "coverage.json"
    baseline = output / "structural_baselines.json"
    provenance = output / "provenance.json"
    csv_path = output / "coverage.csv"
    coverage.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(result["structural_baselines"], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    provenance.write_text(
        json.dumps(result["provenance"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    columns = (
        "asset_id",
        "asset_class",
        "asset_category",
        "room_id",
        "room_family",
        "renderer",
        "qa_id",
        "state",
        "reason_code",
        "reason",
        "scope",
        "target_asset_ids",
        "reference_asset_ids",
        "episode_ids",
        "question_ids",
        "evidence_refs",
        "sound_origins",
        "context_sound_origins",
        "source_refs",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(
                {
                    key: (
                        json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                        if isinstance(row.get(key), (list, dict))
                        else row.get(key)
                    )
                    for key in columns
                }
            )
    return {
        "coverage": str(coverage),
        "csv": str(csv_path),
        "structural_baselines": str(baseline),
        "provenance": str(provenance),
    }


__all__ = [
    "BatchCoverageError",
    "COVERAGE_SCHEMA",
    "COVERAGE_STATES",
    "INPUT_MANIFEST_SCHEMA",
    "QA_IDS",
    "build_batch_coverage",
    "load_batch_input_manifest",
    "validate_batch_coverage",
    "write_batch_coverage",
]
