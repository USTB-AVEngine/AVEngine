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
from avengine.qa.generation_conditions import SELF_MOTION_QA_IDS
from avengine.qa.unified_catalog import QA_IDS

# One predicate for "this question asks about the target's own motion". It used
# to be a local set that left QA-07 out, so a device that cannot walk was
# recorded as missing evidence for "which side did it enter from" instead of
# inapplicable by definition. The condition compiler owns this set now.
MOTION_TARGET_QA_IDS = SELF_MOTION_QA_IDS
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


CODEX_WORKTREE_PREFIX = "/data/jzy/tmp/wt-multi-home-activity-integration"


def rewrite_codex_worktree_path(
    value: Any,
    *,
    repository: Path,
    fallback: Path | None = None,
) -> Any:
    """Point Codex worktree inventory/catalog paths at the production repository."""
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value:
        return str(fallback) if fallback is not None else value
    if CODEX_WORKTREE_PREFIX not in value:
        return value
    suffix = value.split(CODEX_WORKTREE_PREFIX, 1)[1].lstrip("/")
    rewritten = (Path(repository) / suffix).resolve()
    if rewritten.exists():
        return str(rewritten)
    if fallback is not None:
        return str(Path(fallback).resolve())
    return str(rewritten)


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
    default_inventory = repository_path / "tmp/qa_generalized_sampler_review_20260906_v1/full_source_scope_inventory.json"
    default_catalog = repository_path / "examples/rooms/packages/catalog.json"
    default_registry = repository_path / "examples/runtime/source_asset_runtime_profiles.json"
    inventory_value = rewrite_codex_worktree_path(
        payload.get("asset_inventory", default_inventory),
        repository=repository_path,
        fallback=default_inventory,
    )
    catalog_value = rewrite_codex_worktree_path(
        payload.get("room_catalog", default_catalog),
        repository=repository_path,
        fallback=default_catalog,
    )
    registry_value = rewrite_codex_worktree_path(
        payload.get("runtime_registry", default_registry),
        repository=repository_path,
        fallback=default_registry,
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
        "qa_catalog": "QA-01..QA-25",
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


# ---------------------------------------------------------------------------
# V1 coverage targets and shortfall feedback
#
# The table above is the batch denominator over asset x room x QA type. This
# section is the V1 question-bank accounting the production runner consumes:
# declared targets on one side, real achieved rows from
# ``avengine.qa.batch_delivery`` on the other, and the condition compiler's
# per-candidate applicability in between. A cell that produced nothing keeps
# the reason it produced nothing; "not applicable by definition", "interface
# not implemented", "never sampled" and "generation failed" are four different
# answers and are never merged.
# ---------------------------------------------------------------------------

V1_TARGETS_SCHEMA = "avengine_qa_v1_coverage_targets_v1"
V1_FEEDBACK_SCHEMA = "avengine_qa_v1_coverage_feedback_v1"
V1_SOUND_INPUT_SCHEMA = "avengine_qa_v1_sound_input_accounting_v1"
V1_SOURCE_TYPE_SCHEMA = "avengine_qa_v1_source_type_accounting_v1"
V1_SOURCE_TYPE_MATRIX_SCHEMA = "avengine_source_asset_qualification_matrix_v1"

# The eight dimensions one representative asset has to pass before its fine
# type counts as covered. This module does not measure them and does not get
# to choose them: the list is the qualification builder's own
# REQUIRED_DIMENSIONS, imported so the two cannot drift apart and so a type
# can never be retired by asking a shorter question.
from avengine.dataset.source_asset_qualification import (
    REQUIRED_DIMENSIONS as V1_SOURCE_TYPE_CHECKS,
)

# The verdict vocabulary. "pass" and "fail" are results; "not_run" is the
# absence of a result and is the default for an asset nobody has measured;
# "error" is a producer that broke, which is a defect to fix and never a
# statement about the asset.
V1_SOURCE_TYPE_VERDICTS = ("pass", "fail", "not_run", "error")

V1_SHORTFALL_STATES = (
    "met",
    "short_of_target",
    "not_applicable_by_definition",
    "interface_not_implemented",
    "evidence_missing_or_unsampled",
    "generation_failed",
)

# Room families and core task families are read from the configuration, never
# from a name list in this module. These are the required declaration keys.
V1_REQUIRED_TARGET_KEYS = (
    "min_valid_main_questions_per_qa_id",
    "min_distinct_worlds_per_qa_id",
    "min_valid_main_questions_per_branch",
    "min_distinct_worlds_per_branch",
    "min_core_groups_per_task_family_and_room_family",
    "target_core_group_count",
    "min_distinct_worlds_per_entity_combination",
    "room_families",
    "core_task_families",
    "entity_combinations",
)

# A sound class whose semantics the library evidence does not settle is declared
# undetermined and stays out of the schedulable pool. It is not silently bound
# to a device because a directory or class token is named after one.
V1_UNDETERMINED_SEMANTICS_STATE = "semantics_undetermined_pending_owner_decision"


def _positive_int_target(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BatchCoverageError(f"{owner} must be a positive integer, got {value!r}")
    return int(value)


def load_v1_coverage_targets(document: Mapping[str, Any]) -> dict[str, Any]:
    """Read and check the V1 coverage targets of one production configuration.

    ``document`` is either the configuration itself or its target block. The
    check is a completeness check on the declaration, not a claim about
    achieved coverage: every QA type must be named, every key branch of those
    types must be named, and the room families, core task families and
    two-entity combinations must be declared explicitly so a missing axis
    cannot silently shrink the denominator later.
    """
    if not isinstance(document, Mapping):
        raise BatchCoverageError("coverage targets must be an object")
    block = document
    for key in ("v1_coverage_targets", "coverage_quota"):
        candidate = document.get(key)
        if isinstance(candidate, Mapping) and candidate.get("schema") == V1_TARGETS_SCHEMA:
            block = candidate
            break
    if block.get("schema") != V1_TARGETS_SCHEMA:
        raise BatchCoverageError(
            f"coverage targets must declare schema {V1_TARGETS_SCHEMA!r}"
        )
    missing = [key for key in V1_REQUIRED_TARGET_KEYS if key not in block]
    if missing:
        raise BatchCoverageError(f"coverage targets are missing {missing}")
    from avengine.qa.generation_conditions import branches_for

    declared_qa_ids = [str(value) for value in block.get("qa_ids") or []]
    if sorted(declared_qa_ids) != sorted(QA_IDS):
        absent = sorted(set(QA_IDS) - set(declared_qa_ids))
        unknown = sorted(set(declared_qa_ids) - set(QA_IDS))
        raise BatchCoverageError(
            f"coverage targets must name all {len(QA_IDS)} QA types; "
            f"absent={absent} unknown={unknown}"
        )
    declared_branches = block.get("branches_by_qa_id")
    if not isinstance(declared_branches, Mapping):
        raise BatchCoverageError("coverage targets must declare branches_by_qa_id")
    branch_problems = []
    for qa_id in QA_IDS:
        expected = sorted(branches_for(qa_id))
        found = sorted(str(value) for value in declared_branches.get(qa_id, []) or [])
        if found != expected:
            branch_problems.append({"qa_id": qa_id, "expected": expected, "declared": found})
    if branch_problems:
        raise BatchCoverageError(
            f"coverage targets disagree with the shared branch table: {branch_problems}"
        )
    room_families = [str(value) for value in block["room_families"]]
    task_families = [str(value) for value in block["core_task_families"]]
    combinations = [str(value) for value in block["entity_combinations"]]
    for label, values in (
        ("room_families", room_families),
        ("core_task_families", task_families),
        ("entity_combinations", combinations),
    ):
        if not values or len(set(values)) != len(values):
            raise BatchCoverageError(f"coverage targets {label} must be a non-empty unique list")
    from avengine.dataset.production_spec import GROUP_RECIPES

    unknown_families = sorted(set(task_families) - set(GROUP_RECIPES))
    if unknown_families:
        raise BatchCoverageError(
            f"coverage targets name core task families with no shared-unit recipe: {unknown_families}"
        )
    from avengine.dataset.source_capabilities import combination_key, entity_combinations

    known_combinations = {combination_key(*pair) for pair in entity_combinations()}
    unknown_combinations = sorted(set(combinations) - known_combinations)
    if unknown_combinations:
        raise BatchCoverageError(
            f"coverage targets name unknown two-entity combinations: {unknown_combinations}"
        )
    resolved = {
        "schema": V1_TARGETS_SCHEMA,
        "qa_ids": list(QA_IDS),
        "branches_by_qa_id": {qa_id: sorted(branches_for(qa_id)) for qa_id in QA_IDS},
        "room_families": room_families,
        "core_task_families": task_families,
        "entity_combinations": combinations,
        "claim_boundary": (
            "Declared targets only. Achieved coverage, human answerability and "
            "paper admission are separate facts measured elsewhere."
        ),
    }
    for key in (
        "min_valid_main_questions_per_qa_id",
        "min_distinct_worlds_per_qa_id",
        "min_valid_main_questions_per_branch",
        "min_distinct_worlds_per_branch",
        "min_core_groups_per_task_family_and_room_family",
        "target_core_group_count",
        "min_distinct_worlds_per_entity_combination",
    ):
        resolved[key] = _positive_int_target(block[key], f"coverage_targets.{key}")
    expected_groups = (
        resolved["min_core_groups_per_task_family_and_room_family"]
        * len(task_families)
        * len(room_families)
    )
    if resolved["target_core_group_count"] < expected_groups:
        raise BatchCoverageError(
            "target_core_group_count "
            f"{resolved['target_core_group_count']} cannot satisfy "
            f"{resolved['min_core_groups_per_task_family_and_room_family']} groups in each of "
            f"{len(task_families)}x{len(room_families)} cells ({expected_groups})"
        )
    for key in ("notes", "claim_boundary", "first_version_cap_source"):
        if key in block:
            resolved.setdefault("declared_" + key, deepcopy(block[key]))
    return resolved


def source_family_applicability(
    registry: Mapping[str, Any],
    *,
    speech_asset_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Which source families each QA type applies to, from the same predicate.

    A family is applicable when at least one registered asset of that family is
    applicable; a family is inapplicable by definition only when every
    registered asset of it is. That is why a static device inapplicable to a
    self-motion question never collapses the whole QA type: the human and animal
    families of the same type stay applicable and stay in the denominator.
    """
    from avengine.dataset.source_capabilities import SOURCE_FAMILIES, source_family

    by_family: dict[str, list[Mapping[str, Any]]] = {name: [] for name in SOURCE_FAMILIES}
    for record in registry.get("assets") or []:
        if not isinstance(record, Mapping):
            continue
        by_family.setdefault(source_family(record), []).append(record)
    speech_ids = set(str(value) for value in speech_asset_ids or ())
    result: dict[str, dict[str, Any]] = {}
    for qa_id in QA_IDS:
        row: dict[str, Any] = {}
        for family, records in sorted(by_family.items()):
            if not records:
                row[family] = {
                    "state": "evidence_missing_or_unsampled",
                    "reason": f"no registered asset belongs to the {family} family",
                }
                continue
            reasons = []
            applicable = 0
            for record in records:
                asset = {
                    "asset_id": str(record.get("asset_id")),
                    "entity_class": _asset_class(record.get("entity_class")),
                    "category": (record.get("identity") or {}).get("category")
                    or record.get("category"),
                    "allowed_event_classes": record.get("allowed_event_classes") or [],
                }
                reason = _not_applicable_reason(asset, qa_id, speech_asset_ids=speech_ids)
                if reason is None:
                    applicable += 1
                else:
                    reasons.append(reason)
            if applicable:
                row[family] = {
                    "state": "available",
                    "applicable_asset_count": applicable,
                    "inapplicable_asset_count": len(reasons),
                }
            else:
                row[family] = {
                    "state": "not_applicable_by_definition",
                    "reason": sorted(set(reasons))[0] if reasons else "not_applicable_by_definition",
                    "inapplicable_asset_count": len(reasons),
                }
        result[qa_id] = row
    return result


# How a blocked planning state is ranked when one cell has several candidates.
# Higher wins, so an inapplicable device never speaks for a movable source.
_PLANNING_RANK = {
    "not_applicable_by_definition": 1,
    "evidence_missing_or_unsampled": 2,
    "interface_not_implemented": 3,
    "available": 4,
}


def _planning_states(planning: Mapping[str, Any] | None) -> dict[tuple[str, str | None], dict[str, Any]]:
    """Index one ``generation_conditions.condition_report`` by QA type and branch."""
    if not planning:
        return {}
    rows = planning.get("qa_ids")
    if not isinstance(rows, Mapping):
        raise BatchCoverageError("planning input must be a condition_report document")
    index: dict[tuple[str, str | None], dict[str, Any]] = {}
    for qa_id, row in rows.items():
        if not isinstance(row, Mapping):
            continue
        for candidate in row.get("candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            key = (str(qa_id), candidate.get("branch"))
            current = index.get(key)
            state = str(candidate.get("state"))
            # One available candidate makes the cell plannable. Among blocked
            # candidates the least excusing reason wins: a report covering
            # several source pairs contains a device pair for which a
            # self-motion branch is inapplicable by definition, and letting
            # that reason represent the cell would excuse the human and animal
            # pairs whose real blocker is a missing solver.
            if current is None or _PLANNING_RANK.get(
                state, 0
            ) > _PLANNING_RANK.get(str(current["state"]), 0):
                index[key] = {
                    "state": state,
                    "reason": candidate.get("reason"),
                    "targets": candidate.get("targets"),
                }
    return index


def _cell_state(
    *,
    produced_main: int,
    produced_worlds: int,
    required_main: int,
    required_worlds: int,
    planning: Mapping[str, Any] | None,
    applicability: Mapping[str, Any] | None,
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if produced_main >= required_main and produced_worlds >= required_worlds:
        return {"state": "met", "reason": None}
    if produced_main > 0:
        return {
            "state": "short_of_target",
            "reason": (
                f"produced {produced_main} valid main questions in {produced_worlds} worlds; "
                f"target is {required_main} in {required_worlds}"
            ),
        }
    if applicability is not None and applicability.get("state") == "not_applicable_by_definition":
        return {
            "state": "not_applicable_by_definition",
            "reason": applicability.get("reason"),
        }
    if planning is not None and planning.get("state") in {
        "interface_not_implemented",
        "not_applicable_by_definition",
        "evidence_missing_or_unsampled",
    }:
        return {"state": str(planning["state"]), "reason": planning.get("reason")}
    if failures:
        return {
            "state": "generation_failed",
            "reason": f"{len(failures)} member(s) failed question generation",
        }
    return {
        "state": "evidence_missing_or_unsampled",
        "reason": "no valid main question was produced and no blocking condition was reported",
    }


def build_v1_coverage_feedback(
    *,
    targets: Mapping[str, Any],
    achieved: Mapping[str, Any],
    planning: Mapping[str, Any] | None = None,
    applicability: Mapping[str, Any] | None = None,
    sound_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join declared V1 targets with achieved rows into per-cell shortfall.

    ``achieved`` is an ``avengine.qa.batch_delivery.achieved_coverage_table``
    result over real artifacts. ``planning`` is an optional
    ``generation_conditions.condition_report``. Nothing here lowers a target,
    drops a QA type from the denominator or turns an available interface into a
    produced question.
    """
    # ``targets`` is either the target block itself or a whole production
    # configuration carrying it; the loader resolves both.
    resolved = load_v1_coverage_targets(targets)
    if achieved.get("schema") != "avengine_qa_v1_achieved_coverage_v1":
        raise BatchCoverageError("achieved input must be an achieved coverage table")
    planning_index = _planning_states(planning)
    by_qa = achieved.get("by_qa_id") or {}
    by_branch = achieved.get("by_qa_branch") or {}
    failures = list(achieved.get("generation_failures") or [])
    qa_rows: dict[str, Any] = {}
    branch_rows: dict[str, Any] = {}
    for qa_id in resolved["qa_ids"]:
        produced = by_qa.get(qa_id) or {}
        main = int(produced.get("valid_main_questions", 0) or 0)
        worlds = int(produced.get("distinct_worlds_with_main", 0) or 0)
        family_row = (applicability or {}).get(qa_id) or {}
        # The QA type is inapplicable only when every family is; a device-only
        # inapplicability leaves the type in the denominator.
        family_states = {
            family: row.get("state") for family, row in family_row.items()
        }
        aggregate_applicability = None
        if family_states and all(
            state == "not_applicable_by_definition" for state in family_states.values()
        ):
            aggregate_applicability = {
                "state": "not_applicable_by_definition",
                "reason": "every registered source family is inapplicable by definition",
            }
        verdict = _cell_state(
            produced_main=main,
            produced_worlds=worlds,
            required_main=resolved["min_valid_main_questions_per_qa_id"],
            required_worlds=resolved["min_distinct_worlds_per_qa_id"],
            planning=planning_index.get((qa_id, None)),
            applicability=aggregate_applicability,
            failures=failures,
        )
        qa_rows[qa_id] = {
            "qa_id": qa_id,
            "state": verdict["state"],
            "reason": verdict["reason"],
            "valid_main_questions": main,
            "valid_angle_followups": int(produced.get("valid_angle_followups", 0) or 0),
            "distinct_worlds_with_main": worlds,
            "remaining_valid_main_questions": max(
                0, resolved["min_valid_main_questions_per_qa_id"] - main
            ),
            "remaining_distinct_worlds": max(
                0, resolved["min_distinct_worlds_per_qa_id"] - worlds
            ),
            "form_counts": deepcopy(dict(produced.get("form_counts") or {})),
            "branch_unobservable_main": int(produced.get("branch_unobservable_main", 0) or 0),
            "source_family_applicability": deepcopy(family_row),
            "room_families_seen": sorted((produced.get("room_families") or {})),
            "core_task_families_seen": sorted((produced.get("task_families") or {})),
            "entity_combinations_seen": sorted((produced.get("entity_combinations") or {})),
            "deferred_codes": deepcopy(
                dict((achieved.get("deferred_codes_by_qa_id") or {}).get(qa_id) or {})
            ),
        }
        for branch in resolved["branches_by_qa_id"][qa_id]:
            cell = by_branch.get(f"{qa_id}:{branch}") or {}
            branch_main = int(cell.get("valid_main_questions", 0) or 0)
            branch_worlds = int(cell.get("distinct_worlds_with_main", 0) or 0)
            branch_verdict = _cell_state(
                produced_main=branch_main,
                produced_worlds=branch_worlds,
                required_main=resolved["min_valid_main_questions_per_branch"],
                required_worlds=resolved["min_distinct_worlds_per_branch"],
                planning=planning_index.get((qa_id, branch)),
                applicability=aggregate_applicability,
                failures=failures,
            )
            branch_rows[f"{qa_id}:{branch}"] = {
                "qa_id": qa_id,
                "branch": branch,
                "state": branch_verdict["state"],
                "reason": branch_verdict["reason"],
                "valid_main_questions": branch_main,
                "distinct_worlds_with_main": branch_worlds,
                "remaining_valid_main_questions": max(
                    0, resolved["min_valid_main_questions_per_branch"] - branch_main
                ),
                "remaining_distinct_worlds": max(
                    0, resolved["min_distinct_worlds_per_branch"] - branch_worlds
                ),
            }
    matrix_counts = achieved.get("core_task_by_room_family_member_counts") or {}
    matrix_groups = achieved.get("core_task_by_room_family_group_counts") or {}
    matrix_rows = {}
    for task_family in resolved["core_task_families"]:
        for room_family in resolved["room_families"]:
            key = f"{task_family}|{room_family}"
            members = int(matrix_counts.get(key, 0) or 0)
            # Distinct groups when the survey reports them; a member count alone
            # cannot say whether four members are one group or four partial ones.
            groups = int(matrix_groups.get(key, 0) or 0)
            required = resolved["min_core_groups_per_task_family_and_room_family"]
            matrix_rows[key] = {
                "core_task_family": task_family,
                "room_family": room_family,
                "member_count": members,
                "complete_group_count": groups,
                "required_group_count": required,
                "state": "met" if groups >= required else (
                    "short_of_target" if groups else "evidence_missing_or_unsampled"
                ),
                "remaining_group_count": max(0, required - groups),
            }
    combination_rows = {}
    combination_worlds = achieved.get("world_ids_by_entity_combination")
    for combination in resolved["entity_combinations"]:
        questions_by_qa = {
            qa_id: int(((by_qa.get(qa_id) or {}).get("entity_combinations") or {}).get(combination, 0) or 0)
            for qa_id in resolved["qa_ids"]
        }
        produced = sum(questions_by_qa.values())
        required = resolved["min_distinct_worlds_per_entity_combination"]
        if isinstance(combination_worlds, Mapping):
            values = combination_worlds.get(combination, [])
            if (not isinstance(values, (list, tuple))
                    or any(not isinstance(value, str) or not value.strip() for value in values)):
                raise BatchCoverageError(
                    f"world identities for {combination} must be a list of nonempty strings"
                )
            world_ids = sorted(set(values))
            world_count = len(world_ids)
            verdict = _cell_state(
                produced_main=produced, produced_worlds=world_count,
                required_main=1, required_worlds=required,
                planning=None, applicability=None, failures=failures,
            )
        else:
            # Legacy aggregate tables contain question counts per pair, not
            # pair-specific world identities. Re-aggregate their saved survey;
            # neither a question count nor a global world count proves this quota.
            world_ids = None
            world_count = None
            verdict = {
                "state": "evidence_missing_or_unsampled",
                "reason": "distinct world identities by entity combination are missing; rebuild the achieved table from its saved survey",
            }
        combination_rows[combination] = {
            "entity_combination": combination,
            "valid_main_questions": produced,
            "distinct_worlds_with_main": world_count,
            "world_ids": world_ids,
            "required_distinct_worlds": required,
            "remaining_distinct_worlds": (
                None if world_count is None else max(0, required - world_count)
            ),
            "state": verdict["state"],
            "reason": verdict["reason"],
            "qa_ids_with_main": sorted(k for k, v in questions_by_qa.items() if v),
        }
    state_counts = Counter(row["state"] for row in qa_rows.values())
    branch_state_counts = Counter(row["state"] for row in branch_rows.values())
    result = {
        "schema": V1_FEEDBACK_SCHEMA,
        "states": list(V1_SHORTFALL_STATES),
        "targets": resolved,
        "achieved_source_kind": achieved.get("source_kind"),
        "achieved_member_count": achieved.get("member_count"),
        "achieved_group_count": achieved.get("group_count"),
        "achieved_world_count": achieved.get("world_count"),
        "planning_report_present": planning is not None,
        "by_qa_id": qa_rows,
        "by_qa_branch": branch_rows,
        "core_task_by_room_family": matrix_rows,
        "by_entity_combination": combination_rows,
        "qa_state_counts": dict(sorted(state_counts.items())),
        "branch_state_counts": dict(sorted(branch_state_counts.items())),
        "generation_failures": deepcopy(failures),
        "source_families_unresolved": list(achieved.get("source_families_unresolved") or []),
        "counting_note": achieved.get("counting_note"),
        "model_evaluation": "not_run",
        "human_answerability": "not_run",
        "claim_boundary": (
            "Declared targets against achieved rows. An implemented interface is "
            "not a produced question, and a met target is not paper admission."
        ),
    }
    if sound_inputs is not None:
        if sound_inputs.get("schema") != V1_SOUND_INPUT_SCHEMA:
            raise BatchCoverageError("sound_inputs must be a sound input accounting document")
        result["sound_inputs"] = deepcopy(dict(sound_inputs))
    return result


def outstanding_production_requests(feedback: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What the production runner still has to make, with the reason per row.

    Only cells the runner can act on are returned. A cell that is inapplicable
    by definition, or blocked on an unimplemented interface, is reported by
    ``build_v1_coverage_feedback`` but is not handed to the runner as work,
    because retrying it would only burn a budget.
    """
    if feedback.get("schema") != V1_FEEDBACK_SCHEMA:
        raise BatchCoverageError("outstanding requests need a V1 coverage feedback document")
    actionable = {"short_of_target", "evidence_missing_or_unsampled", "generation_failed"}
    rows: list[dict[str, Any]] = []
    for qa_id, row in sorted((feedback.get("by_qa_id") or {}).items()):
        if row["state"] not in actionable:
            continue
        rows.append(
            {
                "kind": "qa_id",
                "qa_id": qa_id,
                "branch": None,
                "state": row["state"],
                "reason": row["reason"],
                "remaining_valid_main_questions": row["remaining_valid_main_questions"],
                "remaining_distinct_worlds": row["remaining_distinct_worlds"],
            }
        )
    for key, row in sorted((feedback.get("by_qa_branch") or {}).items()):
        if row["state"] not in actionable:
            continue
        rows.append(
            {
                "kind": "qa_branch",
                "qa_id": row["qa_id"],
                "branch": row["branch"],
                "state": row["state"],
                "reason": row["reason"],
                "remaining_valid_main_questions": row["remaining_valid_main_questions"],
                "remaining_distinct_worlds": row["remaining_distinct_worlds"],
            }
        )
    for key, row in sorted((feedback.get("core_task_by_room_family") or {}).items()):
        if row["state"] not in actionable:
            continue
        rows.append(
            {
                "kind": "core_group_cell",
                "core_task_family": row["core_task_family"],
                "room_family": row["room_family"],
                "state": row["state"],
                "reason": None,
                "remaining_group_count": row["remaining_group_count"],
            }
        )
    for key, row in sorted((feedback.get("by_entity_combination") or {}).items()):
        if row["state"] not in actionable:
            continue
        rows.append(
            {
                "kind": "entity_combination",
                "entity_combination": row["entity_combination"],
                "state": row["state"],
                "reason": row.get("reason"),
                "distinct_worlds_with_main": row.get("distinct_worlds_with_main"),
                "required_distinct_worlds": row.get("required_distinct_worlds"),
                "remaining_distinct_worlds": row.get("remaining_distinct_worlds"),
            }
        )
    return rows


def sound_input_accounting(
    *,
    registry: Mapping[str, Any],
    sound_class_config: Mapping[str, Any],
    registered_event_class_counts: Mapping[str, int] | None = None,
    library_denominators: Mapping[str, Any] | None = None,
    segment_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Account for the sound input side with its denominators kept apart.

    Library inventory, byte-identical carry-over, truly new relative paths,
    cropped candidates and rows actually schedulable into one Episode are five
    different numbers. They are reported separately here because a single
    "available sounds" figure hides which of them a shortfall belongs to.

    ``registered_event_class_counts`` is the registered event library counted by
    class. A class with registered events but no registered device that declares
    it is reported as an unbound class; it is never made schedulable by letting
    some device accept anything.
    """
    from avengine.dataset.source_capabilities import (
        normalize_sound_class_config,
        sound_class_asset_index,
    )

    resolved_config = normalize_sound_class_config(sound_class_config)
    index = sound_class_asset_index(registry, resolved_config)
    declared_undetermined = {
        str(row.get("sound_class")): deepcopy(dict(row))
        for row in (sound_class_config.get("undetermined_sound_class_semantics") or [])
        if isinstance(row, Mapping)
    }
    unbound: dict[str, Any] = {}
    bound: dict[str, Any] = {}
    for sound_class, count in sorted((registered_event_class_counts or {}).items()):
        accepting = list(index.get(str(sound_class), []))
        row = {
            "sound_class": str(sound_class),
            "registered_event_count": int(count),
            "accepting_asset_count": len(accepting),
            "accepting_asset_ids": accepting,
        }
        if accepting:
            bound[str(sound_class)] = row
        else:
            undetermined = declared_undetermined.get(str(sound_class))
            row["state"] = (
                V1_UNDETERMINED_SEMANTICS_STATE
                if undetermined is not None
                else "no_registered_device_declares_this_sound_class"
            )
            if undetermined is not None:
                row["undetermined_semantics"] = undetermined
            unbound[str(sound_class)] = row
    unbound_events = sum(row["registered_event_count"] for row in unbound.values())
    segment_summary = None
    if segment_rows is not None:
        authorized = [
            row for row in segment_rows
            if isinstance(row, Mapping) and row.get("selection_authorized") is True
        ]
        segment_summary = {
            "cropped_candidate_rows": len(list(segment_rows)),
            "authorized_rows": len(authorized),
            "rows_without_named_authorization": len(list(segment_rows)) - len(authorized),
            "claim_boundary": (
                "A verified crop record is a structural fact about the selection; "
                "it is not a PCM audition, not an AV answerability verdict and it "
                "does not upgrade the original recording's human review."
            ),
        }
    result = {
        "schema": V1_SOUND_INPUT_SCHEMA,
        "denominators": deepcopy(dict(library_denominators or {})),
        "denominator_note": (
            "Library inventory, byte-identical carry-over, truly new paths, "
            "cropped candidates, pool admissions and schedulable rows are "
            "separate denominators and must not be compared to each other."
        ),
        "sound_classes_bound_to_a_device": bound,
        "sound_classes_without_accepting_device": unbound,
        "unbound_registered_event_count": unbound_events,
        "undetermined_sound_class_semantics": declared_undetermined,
        "segment_candidates": segment_summary,
        "declared_sound_class_count": len(index),
    }
    return result


def source_type_accounting(
    *,
    inventory: Mapping[str, Any],
    qualification_matrix: Mapping[str, Any] | None = None,
    candidate_asset_ids_by_class: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Account for the fine source types against the measured 8-dimension matrix.

    The verdicts are not this module's to invent. The qualification builder
    observes registry, geometry scale, support, placement, clearance,
    emitter, visibility and sound PCM per asset and decides each one; this
    reads what it decided and does the arithmetic. There is deliberately no
    way to hand a type five hand-written passes, and no way to shorten the
    required dimension list: retiring a type by asking a smaller question is
    the failure mode this replaced.

    The denominator is the declared type list, not the types that happened to
    appear. A type nobody has attempted is short by exactly as much as a type
    that was attempted and failed, and a type the matrix does not mention at
    all is short and says so rather than vanishing from the count.

    Registry presence is not a verdict. An asset the inventory lists but that
    has no observation stays ``not_run`` and keeps its type short, with the
    dimensions still missing named so the next round knows what to measure.
    """

    types = inventory.get("types")
    if not isinstance(types, Mapping) or not types:
        raise BatchCoverageError(
            "source type inventory declares no types; the 31-type denominator "
            "comes from the declaration and cannot be inferred from deliveries"
        )
    required = list(V1_SOURCE_TYPE_CHECKS)
    matrix_rows: dict[str, Mapping[str, Any]] = {}
    matrix_assets: dict[str, Mapping[str, Any]] = {}
    matrix_status = None
    if qualification_matrix is not None:
        schema = str(qualification_matrix.get("schema") or "")
        if schema != V1_SOURCE_TYPE_MATRIX_SCHEMA:
            raise BatchCoverageError(
                "source type qualification must be a "
                f"{V1_SOURCE_TYPE_MATRIX_SCHEMA} document built by the "
                f"qualification builder, got schema {schema!r}. A map of "
                "hand-written verdicts is not a measurement."
            )
        declared_dimensions = (
            (qualification_matrix.get("api") or {}).get("required_dimensions")
        )
        if declared_dimensions and list(declared_dimensions) != required:
            raise BatchCoverageError(
                "the qualification matrix was built against dimensions "
                f"{list(declared_dimensions)}, which are not the required "
                f"{required}; a type cannot be retired by asking less"
            )
        matrix_status = qualification_matrix.get("status")
        for row in qualification_matrix.get("type_status_rows") or ():
            if isinstance(row, Mapping) and row.get("type"):
                matrix_rows[str(row["type"])] = row
        for row in qualification_matrix.get("asset_status_rows") or ():
            if isinstance(row, Mapping) and row.get("asset_id"):
                matrix_assets[str(row["asset_id"])] = row

    in_candidate_scope: set[str] = set()
    for value in (candidate_asset_ids_by_class or {}).values():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            in_candidate_scope.update(str(item) for item in value)

    def asset_row(asset_id: str) -> dict[str, Any]:
        measured = matrix_assets.get(asset_id)
        dimensions = (measured or {}).get("dimensions") or {}
        seen: dict[str, str] = {}
        for name in required:
            entry = dimensions.get(name)
            value = entry.get("status") if isinstance(entry, Mapping) else entry
            seen[name] = str(value) if value in V1_SOURCE_TYPE_VERDICTS else (
                "not_run" if value is None else "error"
            )
        if measured is None:
            state = "not_run"
            reason = "the qualification matrix carries no row for this asset"
        else:
            state = str(measured.get("status") or "not_run")
            reason = None
            if state not in ("pass", "fail", "not_run"):
                state = "error"
                reason = f"unrecognised asset status {measured.get('status')!r}"
        return {
            "asset_id": asset_id,
            "state": {"pass": "qualified", "fail": "refused"}.get(state, state),
            "reason": reason,
            "dimensions": seen,
            "missing_dimensions": sorted(
                name for name, value in seen.items() if value != "pass"
            ),
            "in_candidate_scope": (
                asset_id in in_candidate_scope if in_candidate_scope else None
            ),
        }

    rows: dict[str, Any] = {}
    for fine_type, candidates in sorted(types.items()):
        listed = (
            candidates
            if isinstance(candidates, Sequence)
            and not isinstance(candidates, (str, bytes))
            else []
        )
        assets: list[dict[str, Any]] = []
        for entry in listed:
            asset_id = entry.get("asset_id") if isinstance(entry, Mapping) else entry
            if not isinstance(asset_id, str) or not asset_id.strip():
                continue
            assets.append(asset_row(asset_id.strip()))
        measured_type = matrix_rows.get(str(fine_type))
        qualified = [row for row in assets if row["state"] == "qualified"]
        errored = [row for row in assets if row["state"] == "error"]
        refused = [row for row in assets if row["state"] == "refused"]
        untested = [row for row in assets if row["state"] == "not_run"]
        if qualification_matrix is None:
            state = "evidence_missing_or_unsampled"
            reason = "no qualification matrix was supplied for this inventory"
        elif measured_type is None:
            state = "evidence_missing_or_unsampled"
            reason = (
                "the inventory declares this type but the qualification "
                "matrix has no row for it"
            )
        elif not assets:
            state = "evidence_missing_or_unsampled"
            reason = "the inventory declares this type with no candidate asset"
        elif str(measured_type.get("status")) == "pass" and qualified:
            state = "met"
            reason = None
        elif errored:
            state = "generation_failed"
            reason = (
                "at least one candidate carries an unreadable verdict rather "
                "than a measurement"
            )
        else:
            state = "evidence_missing_or_unsampled"
            reason = (
                f"{len(refused)} of {len(assets)} candidates were refused and "
                f"{len(untested)} have no measurement yet"
            )
        rows[str(fine_type)] = {
            "fine_type": str(fine_type),
            "state": state,
            "reason": reason,
            "matrix_status": (
                None if measured_type is None else measured_type.get("status")
            ),
            "candidate_count": len(assets),
            "qualified_asset_ids": [row["asset_id"] for row in qualified],
            "refused_asset_ids": [row["asset_id"] for row in refused],
            "errored_asset_ids": [row["asset_id"] for row in errored],
            "not_run_asset_ids": [row["asset_id"] for row in untested],
            "blocking_dimensions": sorted(
                (measured_type or {}).get("blocking_dimensions")
                or {
                    name
                    for row in assets
                    for name in row["missing_dimensions"]
                }
            ),
            "remaining_qualified_assets": 0 if qualified else 1,
            "candidates": assets,
        }

    state_counts: dict[str, int] = {}
    for row in rows.values():
        state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
    blocking_totals: dict[str, int] = {}
    for row in rows.values():
        if row["state"] == "met":
            continue
        for name in row["blocking_dimensions"]:
            blocking_totals[name] = blocking_totals.get(name, 0) + 1
    return {
        "schema": V1_SOURCE_TYPE_SCHEMA,
        "required_dimensions": required,
        "qualification_matrix_schema": (
            None if qualification_matrix is None
            else qualification_matrix.get("schema")
        ),
        "qualification_matrix_status": matrix_status,
        "declared_fine_type_count": len(rows),
        "declared_candidate_asset_count": sum(
            row["candidate_count"] for row in rows.values()),
        "met_fine_type_count": state_counts.get("met", 0),
        "state_counts": dict(sorted(state_counts.items())),
        "blocking_dimension_counts": dict(sorted(blocking_totals.items())),
        "by_fine_type": rows,
        "qualification_supplied": qualification_matrix is not None,
        "claim_boundary": (
            "Declared types against the measured qualification matrix. Being "
            "in the registry is not a verdict, a declared runtime backend is "
            "not native acceptance, and a met type is one representative that "
            "passed every required dimension, not every variant of that type."
        ),
    }


def source_type_shortfall_requests(
    accounting: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Turn every uncovered fine type into one request that says what to make.

    A type that has never been attempted produces a request just as a type
    whose candidates failed does, and the request names the assets still
    without a measurement and the dimensions still missing, so the next round
    has somewhere to start rather than only a count.
    """

    requests: list[dict[str, Any]] = []
    for fine_type, row in sorted((accounting.get("by_fine_type") or {}).items()):
        if row.get("state") == "met":
            continue
        requests.append({
            "kind": "source_fine_type",
            "fine_type": str(fine_type),
            "state": row.get("state"),
            "reason": row.get("reason"),
            "matrix_status": row.get("matrix_status"),
            "remaining_qualified_assets": int(
                row.get("remaining_qualified_assets") or 1),
            "candidate_asset_ids": [
                candidate["asset_id"]
                for candidate in row.get("candidates") or ()
            ],
            "untested_asset_ids": list(row.get("not_run_asset_ids") or ()),
            "refused_asset_ids": list(row.get("refused_asset_ids") or ()),
            "errored_asset_ids": list(row.get("errored_asset_ids") or ()),
            "blocking_dimensions": list(row.get("blocking_dimensions") or ()),
            "missing_dimensions_by_asset": {
                candidate["asset_id"]: list(candidate.get("missing_dimensions") or ())
                for candidate in row.get("candidates") or ()
                if candidate.get("missing_dimensions")
            },
        })
    return requests


def write_v1_coverage_feedback(
    result: Mapping[str, Any], output_dir: str | Path
) -> dict[str, str]:
    """Write the feedback document, its CSV view and the outstanding work list."""
    if result.get("schema") != V1_FEEDBACK_SCHEMA:
        raise BatchCoverageError("only a V1 coverage feedback document is written here")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing coverage feedback output: {output}")
    output.mkdir(parents=True)
    feedback = output / "v1_coverage_feedback.json"
    outstanding = output / "v1_outstanding_requests.json"
    csv_path = output / "v1_coverage_feedback.csv"
    feedback.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = outstanding_production_requests(result)
    outstanding.write_text(
        json.dumps(
            {
                "schema": "avengine_qa_v1_outstanding_requests_v1",
                "count": len(rows),
                "requests": rows,
                "claim_boundary": (
                    "Cells a program worker can act on. Inapplicable and "
                    "unimplemented cells are excluded here and kept in the "
                    "feedback document with their reason."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    columns = (
        "scope",
        "qa_id",
        "branch",
        "state",
        "valid_main_questions",
        "distinct_worlds_with_main",
        "remaining_valid_main_questions",
        "remaining_distinct_worlds",
        "reason",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for qa_id, row in sorted((result.get("by_qa_id") or {}).items()):
            writer.writerow(
                {
                    "scope": "qa_id",
                    "qa_id": qa_id,
                    "branch": "",
                    "state": row["state"],
                    "valid_main_questions": row["valid_main_questions"],
                    "distinct_worlds_with_main": row["distinct_worlds_with_main"],
                    "remaining_valid_main_questions": row["remaining_valid_main_questions"],
                    "remaining_distinct_worlds": row["remaining_distinct_worlds"],
                    "reason": row["reason"] or "",
                }
            )
        for key, row in sorted((result.get("by_qa_branch") or {}).items()):
            writer.writerow(
                {
                    "scope": "qa_branch",
                    "qa_id": row["qa_id"],
                    "branch": row["branch"],
                    "state": row["state"],
                    "valid_main_questions": row["valid_main_questions"],
                    "distinct_worlds_with_main": row["distinct_worlds_with_main"],
                    "remaining_valid_main_questions": row["remaining_valid_main_questions"],
                    "remaining_distinct_worlds": row["remaining_distinct_worlds"],
                    "reason": row["reason"] or "",
                }
            )
    return {
        "feedback": str(feedback),
        "outstanding": str(outstanding),
        "csv": str(csv_path),
    }


__all__ = [
    "MOTION_TARGET_QA_IDS",
    "V1_FEEDBACK_SCHEMA",
    "V1_SHORTFALL_STATES",
    "V1_SOUND_INPUT_SCHEMA",
    "V1_SOURCE_TYPE_CHECKS",
    "V1_SOURCE_TYPE_MATRIX_SCHEMA",
    "V1_SOURCE_TYPE_SCHEMA",
    "V1_SOURCE_TYPE_VERDICTS",
    "V1_TARGETS_SCHEMA",
    "build_v1_coverage_feedback",
    "load_v1_coverage_targets",
    "outstanding_production_requests",
    "sound_input_accounting",
    "source_family_applicability",
    "source_type_accounting",
    "source_type_shortfall_requests",
    "write_v1_coverage_feedback",
    "BatchCoverageError",
    "CODEX_WORKTREE_PREFIX",
    "COVERAGE_SCHEMA",
    "COVERAGE_STATES",
    "INPUT_MANIFEST_SCHEMA",
    "QA_IDS",
    "build_batch_coverage",
    "load_batch_input_manifest",
    "rewrite_codex_worktree_path",
    "validate_batch_coverage",
    "write_batch_coverage",
]
