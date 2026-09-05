#!/usr/bin/env python3
"""Adapt four-speaker SPEAR/audio research inputs to existing QuestionSpec Facts.

This helper does not add question types or a second scheduler. It verifies the
complete PCM clips, the sequential audio-program/research report, and emitter/
listener frame readbacks, then builds the existing FactTable-shaped inputs for
QuestionSpec types already present in the repository:

* card 13 semantics -> appearance_to_spoken_content;
* card 14 semantics -> sound_to_appearance;
* speaking order -> who_spoke_first.

The output remains research_only. Pixel visibility is applied only to the
specific card-13/card-14 all-speaker condition when supplied; an out-of-view
non-target actor never rejects the room.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import wave
from pathlib import Path
from typing import Any, Mapping

from avengine.qa.question_spec import evaluate_question_spec


FRAME_TIME_BASE_HZ = 48_000
LANGUAGE = "en"
VISIBLE_STATES = {"visible_clear", "visible_occluded"}
COLOR_RE = re.compile(r"_(blue|pink|green|white)_v1$", re.IGNORECASE)


class AdapterError(ValueError):
    """Input or evidence is not sufficient for the adapter."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AdapterError(f"PCM clip is missing: {path}")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        payload = handle.readframes(frames)
    if channels != 1 or width != 2 or rate != 16_000:
        raise AdapterError(
            f"PCM must be mono int16 16 kHz: {path} "
            f"(channels={channels}, width={width}, rate={rate})"
        )
    values = struct.iter_unpack("<h", payload)
    nonzero = sum(1 for (value,) in values if value != 0)
    if frames <= 0 or nonzero <= 0:
        raise AdapterError(f"PCM is empty or silent: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "sample_rate_hz": rate,
        "sample_count": frames,
        "nonzero_sample_count": nonzero,
        "duration_seconds": frames / rate,
    }


def actor_color(actor_id: str, record: Mapping[str, Any]) -> str:
    value = record.get("color") or record.get("color_name")
    if isinstance(value, str) and value:
        return value.casefold()
    match = COLOR_RE.search(actor_id)
    if match:
        return match.group(1).casefold()
    raise AdapterError(f"actor {actor_id!r} has no controlled shirt color")


def validate_voice_binding(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = load_json(path)
    if isinstance(raw, Mapping) and isinstance(raw.get("bindings"), list):
        raw = raw["bindings"]
    if not isinstance(raw, list) or len(raw) != 4:
        raise AdapterError("voice_binding must contain exactly four records")
    records: list[dict[str, Any]] = []
    actor_ids: set[str] = set()
    sound_ids: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise AdapterError("voice_binding entries must be objects")
        actor_id = item.get("actor_id")
        sound_id = item.get("sound_asset_id")
        transcript = item.get("transcript")
        wav_path = item.get("path")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (actor_id, sound_id, transcript, wav_path)
        ):
            raise AdapterError("voice_binding needs actor_id, sound_asset_id, transcript and path")
        if actor_id in actor_ids or sound_id in sound_ids:
            raise AdapterError("voice_binding actor_id and sound_asset_id must be unique")
        actor_ids.add(actor_id)
        sound_ids.add(sound_id)
        record = dict(item)
        record["color"] = actor_color(actor_id, record)
        record["pcm"] = pcm_evidence(Path(wav_path).expanduser().resolve())
        records.append(record)
    return records, {
        "status": "pass",
        "path": str(path),
        "actor_count": len(records),
        "actor_ids": [item["actor_id"] for item in records],
        "sound_asset_ids": [item["sound_asset_id"] for item in records],
        "pcm": [
            {
                "actor_id": item["actor_id"],
                "sound_asset_id": item["sound_asset_id"],
                **item["pcm"],
            }
            for item in records
        ],
    }


def validate_frame_readbacks(
    path: Path, actor_ids: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = load_json(path)
    if not isinstance(raw, Mapping):
        raise AdapterError("frame_readbacks root must be an object")
    clock = raw.get("clock")
    camera = raw.get("camera")
    emitters = raw.get("emitters")
    if not isinstance(clock, Mapping) or not isinstance(camera, list):
        raise AdapterError("frame_readbacks needs clock and camera")
    frame_count = clock.get("frame_count")
    frame_rate = clock.get("frame_rate_hz")
    sample_rate = clock.get("sample_rate_hz")
    sample_count = clock.get("sample_count")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (frame_count, frame_rate, sample_rate, sample_count)
    ):
        raise AdapterError("frame_readbacks clock has invalid numeric fields")
    frame_count = int(frame_count)
    if frame_count <= 0 or len(camera) != frame_count:
        raise AdapterError("camera readback length disagrees with clock.frame_count")
    if int(sample_rate) != 16_000 or float(frame_rate) <= 0:
        raise AdapterError("current adapter requires positive frame rate and 16 kHz clock")
    if not isinstance(emitters, Mapping) or set(emitters) != actor_ids:
        raise AdapterError("emitter readbacks must exactly cover the four bound actors")
    for actor_id in actor_ids:
        records = emitters[actor_id]
        if not isinstance(records, list) or len(records) != frame_count:
            raise AdapterError(f"emitter readback length mismatch for {actor_id}")
        for record in records:
            if not isinstance(record, Mapping):
                raise AdapterError(f"invalid emitter frame for {actor_id}")
            if not isinstance(record.get("location_cm"), list) or len(record["location_cm"]) != 3:
                raise AdapterError(f"emitter frame lacks location_cm for {actor_id}")
    for record in camera:
        if not isinstance(record, Mapping):
            raise AdapterError("camera readback frame must be an object")
        if not isinstance(record.get("location_cm"), list) or len(record["location_cm"]) != 3:
            raise AdapterError("camera frame lacks location_cm")
    return raw, {
        "status": "pass",
        "path": str(path),
        "frame_count": frame_count,
        "frame_rate_hz": float(frame_rate),
        "sample_rate_hz": int(sample_rate),
        "sample_count": int(sample_count),
        "actor_ids": sorted(actor_ids),
        "camera_frame_count": len(camera),
        "emitter_frame_counts": {
            actor_id: len(emitters[actor_id]) for actor_id in sorted(actor_ids)
        },
    }


def _event_sample_bounds(event: Mapping[str, Any]) -> tuple[int, int]:
    start = event.get("start_sample")
    end = event.get("end_sample_exclusive")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        raise AdapterError(f"invalid event sample bounds: {event.get('event_id')!r}")
    return start, end


def validate_audio_artifacts(
    audio_program_path: Path | None,
    research_report_path: Path | None,
    bindings: list[dict[str, Any]],
    clock: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    if audio_program_path is None or research_report_path is None:
        missing = []
        if audio_program_path is None:
            missing.append("audio_program")
        if research_report_path is None:
            missing.append("research_report")
        reason = "required native/audio artifact is not supplied"
        details: dict[str, Any] = {}
        if audio_program_path is not None and audio_program_path.is_file():
            program = load_json(audio_program_path)
            events = program.get("events") if isinstance(program, Mapping) else None
            clock_samples = int(clock.get("sample_count", 0))
            ends = [
                int(item.get("end_sample_exclusive"))
                for item in events
                if isinstance(item, Mapping)
                and isinstance(item.get("end_sample_exclusive"), int)
            ] if isinstance(events, list) else []
            if ends and max(ends) > clock_samples:
                reason = "audio_program_exceeds_frame_readback_clock"
                details = {
                    "audio_program_max_end_sample": max(ends),
                    "frame_readback_sample_count": clock_samples,
                }
        return None, None, {
            "status": "not_run",
            "reason": reason,
            "missing": missing,
            **details,
        }
    if not audio_program_path.is_file() or not research_report_path.is_file():
        raise AdapterError("audio_program and research_report paths must exist together")
    program = load_json(audio_program_path)
    report = load_json(research_report_path)
    events = program.get("events") if isinstance(program, Mapping) else None
    report_events = report.get("events") if isinstance(report, Mapping) else None
    if not isinstance(events, list) or len(events) != 4:
        raise AdapterError("audio_program must contain four events")
    if not isinstance(report, Mapping) or report.get("status") not in {"research", "pass"}:
        raise AdapterError("research_report must be a research/pass report")
    if report.get("complete_sentences_preserved") is not True:
        raise AdapterError("research_report must assert complete_sentences_preserved")
    if not isinstance(report_events, list) or len(report_events) != 4:
        raise AdapterError("research_report must contain four events")
    by_actor = {item["actor_id"]: item for item in bindings}
    by_sound = {item["sound_asset_id"]: item for item in bindings}
    report_by_actor: dict[str, Mapping[str, Any]] = {}
    report_by_event: dict[str, Mapping[str, Any]] = {}
    for item in report_events:
        actor_id = item.get("actor_id")
        sound_id = item.get("sound_asset_id")
        transcript = item.get("transcript")
        if actor_id not in by_actor or sound_id not in by_sound:
            raise AdapterError("research_report event does not resolve to voice_binding")
        event_id = item.get("event_id")
        if actor_id in report_by_actor or event_id in report_by_event:
            raise AdapterError("research_report repeats an actor or event")
        if transcript != by_actor[actor_id]["transcript"]:
            raise AdapterError(f"transcript mismatch for {actor_id}")
        pcm_interval = item.get("pcm_output_nonzero_interval")
        if not isinstance(pcm_interval, list) or len(pcm_interval) != 2:
            raise AdapterError(f"PCM output nonzero interval missing for {actor_id}")
        start, end = _event_sample_bounds(item)
        if end > int(clock.get("sample_count", 0)):
            raise AdapterError(f"research_report event exceeds frame-readback clock for {actor_id}")
        if int(pcm_interval[0]) < start or int(pcm_interval[1]) > end:
            raise AdapterError(f"PCM nonzero interval escapes event for {actor_id}")
        clip_count = by_actor[actor_id]["pcm"]["sample_count"]
        source_end = item.get("source_end_sample_exclusive")
        if source_end is not None and int(source_end) != clip_count:
            raise AdapterError(f"research_report clip boundary differs from voice binding for {actor_id}")
        report_by_actor[actor_id] = item
        report_by_event[str(event_id)] = item
    ordered_report = sorted(
        (_event_sample_bounds(item) for item in report_events),
        key=lambda value: value[0],
    )
    if any(left[1] > right[0] for left, right in zip(ordered_report, ordered_report[1:])):
        raise AdapterError("research_report speech events overlap")
    program_by_sound = {item.get("sound_asset_id"): item for item in events}
    program_by_event = {item.get("event_id"): item for item in events}
    if set(program_by_sound) != set(by_sound):
        raise AdapterError("audio_program sound IDs differ from voice_binding")
    for actor_id, binding in by_actor.items():
        event = report_by_actor.get(actor_id)
        if event is None:
            raise AdapterError(f"research_report has no event for {actor_id}")
        program_event = program_by_sound[binding["sound_asset_id"]]
        endpoint = program_event.get("source_endpoint_id")
        if endpoint not in {actor_id, f"{actor_id}_mouth"}:
            raise AdapterError(f"audio_program endpoint mismatch for {actor_id}")
        report_event = report_by_actor[actor_id]
        paired_program = program_by_event.get(report_event.get("event_id"))
        if paired_program is None:
            raise AdapterError(f"audio_program has no event for {actor_id}")
        for field in ("start_sample", "end_sample_exclusive", "sound_asset_id"):
            if paired_program.get(field) != report_event.get(field):
                raise AdapterError(f"audio_program/research_report mismatch for {actor_id}: {field}")
    return program, report, {
        "status": "pass",
        "audio_program": str(audio_program_path),
        "research_report": str(research_report_path),
        "event_count": 4,
        "complete_sentences_preserved": True,
        "nonzero_pcm_events": 4,
    }


def visibility_at(truth: Mapping[str, Any] | None, actor_id: str, frame: int) -> str | None:
    if not isinstance(truth, Mapping):
        return None
    per_instance = truth.get("per_instance")
    if not isinstance(per_instance, Mapping):
        return None
    record = per_instance.get(actor_id)
    frames = record.get("frames") if isinstance(record, Mapping) else None
    if not isinstance(frames, list):
        return None
    for item in frames:
        if isinstance(item, Mapping) and item.get("frame_index") == frame:
            value = item.get("state")
            return value if isinstance(value, str) else None
    return None


def card_visibility(
    truth: Mapping[str, Any] | None,
    actor_ids: list[str],
    frame: int,
    *,
    required: bool,
) -> dict[str, Any]:
    if truth is None:
        return {
            "status": "not_run",
            "reason": "pixel_visibility_truth is missing",
            "frame_index": frame,
        }
    states = {actor_id: visibility_at(truth, actor_id, frame) for actor_id in actor_ids}
    if any(value is None for value in states.values()):
        return {
            "status": "not_run",
            "reason": "pixel_visibility_truth lacks one or more target actor frames",
            "frame_index": frame,
            "states": states,
        }
    if required and not all(value in VISIBLE_STATES for value in states.values()):
        return {
            "status": "unsupported",
            "reason": "all-speaker visibility condition is not satisfied at the target utterance frame",
            "frame_index": frame,
            "states": states,
        }
    return {
        "status": "pass",
        "frame_index": frame,
        "states": states,
    }


def _color_asset_record(binding: Mapping[str, Any]) -> dict[str, Any]:
    color = binding["color"]
    actor_id = binding["actor_id"]
    return {
        "asset_id": actor_id,
        "display_label": f"{color} shirt speaker",
        "entity_class": "articulated_human",
        "identity": {
            "species_id": "human",
            "breed_id": "rocketbox_human",
        },
        "realized_attributes": {
            "size": "medium",
            "body_build": "standard",
            "life_stage": "adult",
            "sex_or_gender_label": str(binding.get("gender", "unknown")).casefold(),
            "coat_profile": {
                "profile_id": f"shirt_{color}",
                "value": color,
            },
        },
    }


def _build_facts(
    bindings: list[dict[str, Any]],
    report: Mapping[str, Any],
    readbacks: Mapping[str, Any],
    truth: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    clock = readbacks["clock"]
    frame_count = int(clock["frame_count"])
    frame_rate = float(clock["frame_rate_hz"])
    ticks_per_frame = int(FRAME_TIME_BASE_HZ // int(frame_rate))
    by_actor = {item["actor_id"]: item for item in bindings}
    event_bindings: dict[str, dict[str, str]] = {}
    events: list[dict[str, Any]] = []
    asset_records = [_color_asset_record(item) for item in bindings]
    sound_records: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    for item in bindings:
        actor_id = item["actor_id"]
        instances.append(
            {
                "instance_id": actor_id,
                "source_slot_id": actor_id,
                "asset_id": actor_id,
                "display_label": f"{item['color']} shirt speaker",
                "entity_class": "articulated_human",
                "breed_id": "rocketbox_human",
                "attributes": {
                    "size": "medium",
                    "body_build": "standard",
                    "life_stage": "adult",
                    "sex_or_gender_label": str(item.get("gender", "unknown")).casefold(),
                    "coat_value": item["color"],
                },
            }
        )
        sound_records.append(
            {
                "sound_asset_id": item["sound_asset_id"],
                "path": item["pcm"]["path"],
                "sha256": item["pcm"]["sha256"],
                "species": "human",
                "content": {
                    "species": "human",
                    "statement_id": f"statement_{item['sound_asset_id']}",
                    "transcript": item["transcript"],
                    "language": LANGUAGE,
                },
            }
        )
    for index, event in enumerate(report["events"]):
        actor_id = event["actor_id"]
        item = by_actor[actor_id]
        start, end = _event_sample_bounds(event)
        start_frame = int(math.floor(start / float(clock["sample_rate_hz"]) * frame_rate))
        end_frame = int(math.ceil(end / float(clock["sample_rate_hz"]) * frame_rate))
        event_id = str(event["event_id"])
        sound_id = item["sound_asset_id"]
        event_bindings[event_id] = {"sound_asset_id": sound_id}
        events.append(
            {
                "event_id": event_id,
                "source_slot_id": actor_id,
                "asset_id": actor_id,
                "sound_asset_id": sound_id,
                "sound_class": {"species_id": "human", "display_label": "human speech"},
                "statement_id": f"statement_{sound_id}",
                "transcript": item["transcript"],
                "language": LANGUAGE,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_tick": start_frame * ticks_per_frame,
                "end_tick": end_frame * ticks_per_frame,
                "start_sample": start,
                "end_sample": end,
                "dry_variant": {
                    "input_path": item["pcm"]["path"],
                    "input_sha256": item["pcm"]["sha256"],
                },
            }
        )
    moving = [False] * frame_count
    tracks = {
        "instances": {
            item["actor_id"]: {
                "moving": list(moving),
                "doa": {
                    "azimuth_deg": [0.0] * frame_count,
                    "distance_m": [1.0] * frame_count,
                },
            }
            for item in bindings
        }
    }
    facts = {
        "schema": "avengine_qa_fact_table_v1",
        "status": "pass",
        "episode_id": "four_speaker_vctk_research",
        "time": {
            "frame_count": frame_count,
            "frame_rate_hz": frame_rate,
            "sample_rate_hz": int(clock["sample_rate_hz"]),
            "sample_count": int(clock["sample_count"]),
            "ticks_per_frame": ticks_per_frame,
        },
        "instances": instances,
        "sound_events": events,
        "tracks": tracks,
        "visibility": {"pixel_truth": truth},
    }
    asset_registry = {
        "schema": "avengine_research_asset_registry_v1",
        "assets": asset_records,
    }
    sound_registry = {
        "schema": "avengine_m6_sound_asset_registry_v1",
        "sounds": sound_records,
    }
    return facts, asset_registry, sound_registry, event_bindings


def build(
    *,
    frame_readbacks: Path,
    voice_binding: Path,
    output: Path,
    audio_program: Path | None = None,
    research_report: Path | None = None,
    pixel_visibility_truth: Path | None = None,
    require_all_speaker_visible: bool = True,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"fresh output required: {output}")
    bindings, binding_report = validate_voice_binding(voice_binding)
    readbacks, readback_report = validate_frame_readbacks(
        frame_readbacks, {item["actor_id"] for item in bindings}
    )
    program, research, audio_report = validate_audio_artifacts(
        audio_program,
        research_report,
        bindings,
        readbacks["clock"],
    )
    truth = load_json(pixel_visibility_truth) if pixel_visibility_truth else None
    input_report = {
        "status": "pass",
        "voice_binding": binding_report,
        "frame_readbacks": readback_report,
        "audio": audio_report,
        "pixel_visibility": {
            "status": "pass" if truth is not None else "not_run",
            "path": str(pixel_visibility_truth) if pixel_visibility_truth else None,
        },
    }
    output.mkdir(parents=True)
    write_json(output / "input_validation_report.json", input_report)
    result: dict[str, Any] = {
        "kind": "four_speaker_research_qa_examples",
        "status": "not_run" if audio_report["status"] != "pass" else "research_only",
        "claim_boundary": (
            "Reuses existing QuestionSpec/Facts evaluation. This is a research "
            "example adapter, not formal QA admission or a new question protocol."
        ),
        "input_validation_report": str(output / "input_validation_report.json"),
        "question_type_mapping": {
            "card13": "appearance_to_spoken_content",
            "card14": "sound_to_appearance",
            "speaker_order": "who_spoke_first",
        },
        "visibility_policy": {
            "card13_card14_target_condition": "all four speakers visible at target utterance frame",
            "required": require_all_speaker_visible,
            "out_of_view_non_target_is_not_room_failure": True,
        },
        "samples": [],
        "deferred_samples": [],
    }
    if audio_report["status"] != "pass":
        write_json(output / "research_qa_examples.json", result)
        return result
    facts, asset_registry, sound_registry, event_bindings = _build_facts(
        bindings,
        research,
        readbacks,
        truth,
    )
    write_json(output / "facts.json", facts)
    write_json(output / "asset_registry.json", asset_registry)
    write_json(output / "sound_registry.json", sound_registry)
    write_json(output / "event_sound_bindings.json", event_bindings)
    by_color = {item["color"]: item for item in bindings}
    target = by_color.get("blue", bindings[0])
    target_event = next(event for event in research["events"] if event["actor_id"] == target["actor_id"])
    target_frame = int(
        math.floor(
            int(target_event["start_sample"])
            / float(readbacks["clock"]["sample_rate_hz"])
            * float(readbacks["clock"]["frame_rate_hz"])
        )
    )
    visibility = card_visibility(
        truth,
        [item["actor_id"] for item in bindings],
        target_frame,
        required=require_all_speaker_visible,
    )
    samples: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    order_spec = {
        "schema": "avengine_qa_question_spec_v1",
        "spec_id": "QS-003",
        "question_type": "who_spoke_first",
        "selectors": {},
    }
    order_eval = evaluate_question_spec(
        order_spec,
        facts=facts,
        asset_registry=asset_registry,
        sound_registry=sound_registry,
        event_sound_bindings=event_bindings,
    )
    samples.append({"card": "speaker_order", "question_spec": order_spec, "evaluation": order_eval})
    card13_spec = {
        "schema": "avengine_qa_question_spec_v1",
        "spec_id": "QS-013",
        "question_type": "appearance_to_spoken_content",
        "selectors": {"appearance_field": "coat_value", "appearance_value": target["color"]},
    }
    card14_spec = {
        "schema": "avengine_qa_question_spec_v1",
        "spec_id": "QS-014",
        "question_type": "sound_to_appearance",
        "selectors": {"sound_asset_id": target["sound_asset_id"], "appearance_field": "coat_value"},
    }
    if visibility["status"] == "pass" or not require_all_speaker_visible:
        samples.extend([
            {"card": "card13", "question_spec": card13_spec,
             "visibility": visibility,
             "evaluation": evaluate_question_spec(
                 card13_spec,
                 facts=facts,
                 asset_registry=asset_registry,
                 sound_registry=sound_registry,
                 event_sound_bindings=event_bindings,
             )},
            {"card": "card14", "question_spec": card14_spec,
             "visibility": visibility,
             "evaluation": evaluate_question_spec(
                 card14_spec,
                 facts=facts,
                 asset_registry=asset_registry,
                 sound_registry=sound_registry,
                 event_sound_bindings=event_bindings,
             )},
        ])
    else:
        deferred.extend([
            {"card": "card13", "question_spec": card13_spec, "visibility": visibility},
            {"card": "card14", "question_spec": card14_spec, "visibility": visibility},
        ])
    result["status"] = "research_only"
    result["samples"] = samples
    result["deferred_samples"] = deferred
    result["facts"] = str(output / "facts.json")
    result["asset_registry"] = str(output / "asset_registry.json")
    result["sound_registry"] = str(output / "sound_registry.json")
    result["event_sound_bindings"] = str(output / "event_sound_bindings.json")
    write_json(output / "research_qa_examples.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-readbacks", required=True, type=Path)
    parser.add_argument("--voice-binding", required=True, type=Path)
    parser.add_argument("--audio-program", type=Path)
    parser.add_argument("--research-report", type=Path)
    parser.add_argument("--pixel-visibility-truth", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-partial-speaker-visibility",
        action="store_true",
        help="emit card13/card14 research rows without the all-four visible condition",
    )
    args = parser.parse_args()
    result = build(
        frame_readbacks=args.frame_readbacks,
        voice_binding=args.voice_binding,
        audio_program=args.audio_program,
        research_report=args.research_report,
        pixel_visibility_truth=args.pixel_visibility_truth,
        output=args.output,
        require_all_speaker_visible=not args.allow_partial_speaker_visibility,
    )
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "sample_count": len(result["samples"]),
        "deferred_count": len(result["deferred_samples"]),
    }, indent=2))


if __name__ == "__main__":
    main()
