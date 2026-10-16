"""Axis-1 (route-swap) counterfactual twins and answer-flip verification.

A route-swap twin keeps every visual byte and every trajectory identical and
swaps the dry-audio routing between the two source slots: the sound that
played from slot A's route now plays from slot B's route. Because the RIR
cache is dry-audio independent, the twin's audio is a pure re-mix of cached
impulse responses; nothing is re-simulated.

Certification is fact-level and per question: the verifier re-answers the
question's underlying predicate on the twin fact table and grants an axis-1
certificate only when the answer actually flips. Question types whose answer
does not depend on audio routing (counts, distances, motion) are recorded as
``not_applicable`` rather than silently certified.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

TWIN_SUFFIX = "__cf_route_swap"
COUNTERFACTUAL_KIND = "axis1_route_swap"

AXIS1_FLIP_TYPES = ("Q-ATTR", "Q-AVREL")
AXIS1_NOT_APPLICABLE_TYPES = ("Q-ACT", "Q-CNT", "Q-CMP", "Q-SRC")


class QACertifyError(ValueError):
    """A twin construction or flip verification contract is violated."""


def _two_slots(fact_table: Mapping[str, Any]) -> tuple[str, str]:
    instances = fact_table.get("instances")
    if not isinstance(instances, list) or len(instances) != 2:
        raise QACertifyError("axis-1 twins require exactly two instances")
    return instances[0]["instance_id"], instances[1]["instance_id"]


def twin_fact_table(fact_table: Mapping[str, Any]) -> dict[str, Any]:
    """Build the route-swap twin of a compiled fact table.

    Poses, tracks, visibility and instances are unchanged (the visuals are
    byte-identical); each sound event keeps its slot/route but carries the
    other slot's voice. ``voice_asset_id`` records whose recording is heard.
    """

    slot_a, slot_b = _two_slots(fact_table)
    twin = copy.deepcopy(dict(fact_table))
    events = twin.get("sound_events")
    if not isinstance(events, list) or len(events) != 2:
        raise QACertifyError("axis-1 twins require exactly one event per slot")
    by_slot = {event["source_slot_id"]: event for event in events}
    if set(by_slot) != {slot_a, slot_b}:
        raise QACertifyError("sound events do not cover both source slots")

    def voice_fields(event: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "sound_class": copy.deepcopy(event["sound_class"]),
            "dry_variant": copy.deepcopy(event["dry_variant"]),
            "voice_asset_id": event.get("voice_asset_id", event["asset_id"]),
        }

    voice_a = voice_fields(by_slot[slot_a])
    voice_b = voice_fields(by_slot[slot_b])
    by_slot[slot_a].update(voice_b)
    by_slot[slot_b].update(voice_a)

    twin["episode_id"] = f"{fact_table['episode_id']}{TWIN_SUFFIX}"
    twin["counterfactual"] = {
        "kind": COUNTERFACTUAL_KIND,
        "twin_of": fact_table["episode_id"],
        "visual_bytes_identical": True,
        "rir_cache_reused": True,
    }
    # The twin's mixture has not been rendered at fact level; drop the
    # original mixture identity so nothing can bind the twin to the
    # original waveform by mistake.
    twin["audio"] = {
        **twin["audio"],
        "mixture_path": f"pending_render/{twin['episode_id']}.wav",
        "mixture_sha256": "0" * 64,
        "peak_absolute": 0.0,
    }
    return twin


def _instance_by_id(fact_table: Mapping[str, Any], instance_id: str) -> Mapping[str, Any]:
    for instance in fact_table["instances"]:
        if instance["instance_id"] == instance_id:
            return instance
    raise QACertifyError(f"instance {instance_id!r} is missing from the fact table")


def _slot_hosting_species_voice(
    fact_table: Mapping[str, Any], species_id: str
) -> str:
    matches = [
        event["source_slot_id"]
        for event in fact_table["sound_events"]
        if event["sound_class"]["species_id"] == species_id
    ]
    if len(matches) != 1:
        raise QACertifyError(
            f"species voice {species_id!r} does not resolve to exactly one slot"
        )
    return matches[0]


def reanswer_on_fact_table(
    question: Mapping[str, Any], fact_table: Mapping[str, Any]
) -> str | None:
    """Re-answer a mined question's predicate on an arbitrary fact table.

    Questions reference sounds, not slots: "the coat of the animal that is
    barking" resolves through the slot that hosts the bark. Returns the
    answer value, or ``None`` when the predicate has no unambiguous answer
    on this fact table (for example the referring expression loses its
    unique referent after the swap).
    """

    type_id = question["type_id"]
    if type_id == "Q-ATTR":
        species = _instance_by_id(fact_table, question["evidence"]["instance_id"])
        # The question asked about the animal heard vocalizing as this
        # species; resolve the slot hosting that voice now.
        voiced_slot = _slot_hosting_species_voice(fact_table, species["species_id"])
        host = _instance_by_id(fact_table, voiced_slot)
        return host["attributes"].get("coat_value")
    if type_id == "Q-AVREL":
        original_host = _instance_by_id(
            fact_table, question["evidence"]["instance_id"]
        )
        voiced_slot = _slot_hosting_species_voice(
            fact_table, original_host["species_id"]
        )
        track = fact_table["tracks"]["instances"][voiced_slot]
        azimuth = track["doa"]["azimuth_deg"]
        right = sum(1 for value in azimuth if value > 0.0) / len(azimuth)
        if right >= 0.9:
            return "right"
        if right <= 0.1:
            return "left"
        return None
    raise QACertifyError(f"no axis-1 re-answer rule for question type {type_id!r}")


def certify_axis1(
    question: Mapping[str, Any],
    original: Mapping[str, Any],
    twin: Mapping[str, Any],
) -> dict[str, Any]:
    """Grant or refuse an axis-1 certificate for one mined question."""

    type_id = question["type_id"]
    if type_id in AXIS1_NOT_APPLICABLE_TYPES:
        return {
            "question_id": question["question_id"],
            "certificate": "axis1_route_swap",
            "status": "not_applicable",
            "reason": "answer does not depend on dry-audio routing",
        }
    if type_id not in AXIS1_FLIP_TYPES:
        raise QACertifyError(f"unknown question type {type_id!r}")

    original_answer = reanswer_on_fact_table(question, original)
    if original_answer != question["answer_value"]:
        return {
            "question_id": question["question_id"],
            "certificate": "axis1_route_swap",
            "status": "refused",
            "reason": "re-answer on the original fact table disagrees with the mined answer",
        }
    twin_answer = reanswer_on_fact_table(question, twin)
    granted = twin_answer is not None and twin_answer != original_answer
    record = {
        "question_id": question["question_id"],
        "certificate": "axis1_route_swap",
        "status": "granted" if granted else "refused",
        "original_answer": original_answer,
        "twin_answer": twin_answer,
        "twin_episode_id": twin["episode_id"],
    }
    if not granted:
        record["reason"] = (
            "twin answer is ambiguous"
            if twin_answer is None
            else "answer does not flip under route swap"
        )
    return record
