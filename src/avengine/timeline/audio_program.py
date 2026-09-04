"""Data-driven source activation programs on the frozen M5/M5.1 timeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from avengine.contracts.json_io import canonical_json_sha256, load_json
from avengine.capture.source_contracts import sample_boundary
from avengine.registry.registry import (
    AUDIO_PROGRAM_SCHEMA,
    M6RegistryError,
    all_numbers_finite,
    json_schema_errors,
)
from avengine.registry.sources import (
    endpoint_index,
    sound_index,
    source_sound_compatibility_errors,
    validate_sound_asset_registry,
    validate_source_endpoint_registry,
)


TIME_BASE_HZ = 48_000
VIDEO_FPS = 15
TICKS_PER_FRAME = 3_200
AUDIO_SAMPLE_RATE_HZ = 16_000
TICKS_PER_SAMPLE = 3


class AudioProgramError(M6RegistryError):
    pass


def bind_audio_program_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result.pop("program_content_sha256", None)
    result["program_content_sha256"] = canonical_json_sha256(result)
    return result


def validate_audio_program(
    value: Any,
    *,
    source_endpoint_registry: Mapping[str, Any] | None = None,
    sound_asset_registry: Mapping[str, Any] | None = None,
) -> list[str]:
    errors = json_schema_errors(value, AUDIO_PROGRAM_SCHEMA)
    if not isinstance(value, Mapping):
        return errors
    if not all_numbers_finite(value):
        errors.append("audio program must contain only finite JSON numbers")
    declared_hash = value.get("program_content_sha256")
    expected_hash = canonical_json_sha256(
        {key: item for key, item in value.items() if key != "program_content_sha256"}
    )
    if declared_hash != expected_hash:
        errors.append("program_content_sha256 does not match canonical content")

    candidates = value.get("candidate_source_endpoint_ids")
    events = value.get("events")
    if not isinstance(candidates, list) or not isinstance(events, list):
        return errors
    if candidates != sorted(set(candidates)):
        errors.append("candidate_source_endpoint_ids must be unique and canonical")
    timeline = value.get("timeline")
    frame_count = timeline.get("frame_count") if isinstance(timeline, Mapping) else None
    sample_count = timeline.get("sample_count") if isinstance(timeline, Mapping) else None
    if isinstance(frame_count, int) and not isinstance(frame_count, bool):
        expected_sample_count = sample_boundary(frame_count)
        if sample_count != expected_sample_count:
            errors.append(
                f"timeline.sample_count must equal exact boundary {expected_sample_count}"
            )
    event_keys: list[tuple[int, str, str]] = []
    seen_event_ids: set[str] = set()
    active_endpoint_ids: set[str] = set()
    intervals_by_endpoint: dict[str, list[tuple[int, int, str]]] = {}
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        owner = f"events[{index}]"
        event_id = str(event.get("event_id", ""))
        source_endpoint_id = str(event.get("source_endpoint_id", ""))
        if event_id in seen_event_ids:
            errors.append(f"{owner}.event_id is duplicated")
        seen_event_ids.add(event_id)
        active_endpoint_ids.add(source_endpoint_id)
        if source_endpoint_id not in candidates:
            errors.append(f"{owner}.source_endpoint_id is not a candidate endpoint")
        start_sample = event.get("start_sample")
        end_sample = event.get("end_sample_exclusive")
        if isinstance(start_sample, int) and not isinstance(start_sample, bool) and isinstance(end_sample, int) and not isinstance(end_sample, bool):
            if not start_sample < end_sample:
                errors.append(f"{owner} requires start_sample < end_sample_exclusive")
            if isinstance(sample_count, int) and end_sample > sample_count:
                errors.append(f"{owner}.end_sample_exclusive exceeds timeline.sample_count")
            expected = {
                "start_tick": start_sample * TICKS_PER_SAMPLE,
                "end_tick_exclusive": end_sample * TICKS_PER_SAMPLE,
            }
            for field, expected_value in expected.items():
                if event.get(field) != expected_value:
                    errors.append(f"{owner}.{field} must equal {expected_value}")
            event_keys.append((start_sample, source_endpoint_id, event_id))
            intervals_by_endpoint.setdefault(source_endpoint_id, []).append(
                (start_sample, end_sample, event_id)
            )
            source_start = event.get("source_start_sample")
            source_end = event.get("source_end_sample_exclusive")
            if (
                isinstance(source_start, int)
                and not isinstance(source_start, bool)
                and isinstance(source_end, int)
                and not isinstance(source_end, bool)
            ):
                if not 0 <= source_start < source_end:
                    errors.append(
                        f"{owner} requires source_start_sample < source_end_sample_exclusive"
                    )
                if source_end - source_start != end_sample - start_sample:
                    errors.append(
                        f"{owner} scheduled duration must equal exact source slice duration"
                    )
                fade_samples = event.get("fade_samples")
                if (
                    isinstance(fade_samples, int)
                    and not isinstance(fade_samples, bool)
                    and 2 * fade_samples > source_end - source_start
                ):
                    errors.append(f"{owner}.fade_samples does not fit the source slice")
    if event_keys != sorted(event_keys):
        errors.append("events must use canonical sample/source/event order")
    for source_endpoint_id, intervals in intervals_by_endpoint.items():
        previous_end = -1
        for start, end, event_id in sorted(intervals):
            if start < previous_end:
                errors.append(
                    f"events for {source_endpoint_id!r} overlap at {event_id!r}"
                )
            previous_end = max(previous_end, end)

    all_intervals = sorted(
        (
            event["start_sample"],
            event["end_sample_exclusive"],
            event["source_endpoint_id"],
            event["event_id"],
        )
        for event in events
        if isinstance(event, Mapping)
        and isinstance(event.get("start_sample"), int)
        and not isinstance(event.get("start_sample"), bool)
        and isinstance(event.get("end_sample_exclusive"), int)
        and not isinstance(event.get("end_sample_exclusive"), bool)
        and isinstance(event.get("source_endpoint_id"), str)
        and isinstance(event.get("event_id"), str)
    )
    cross_source_overlap = any(
        left_source != right_source
        and max(left_start, right_start) < min(left_end, right_end)
        for left_start, left_end, left_source, _ in all_intervals
        for right_start, right_end, right_source, _ in all_intervals
    )
    any_overlap = any(
        max(left_start, right_start) < min(left_end, right_end)
        for left_index, (left_start, left_end, _, _) in enumerate(all_intervals)
        for right_start, right_end, _, _ in all_intervals[left_index + 1 :]
    )
    has_intermittent_gap = any(
        right_start > left_end
        for intervals in intervals_by_endpoint.values()
        for (left_start, left_end, _), (right_start, _, _) in zip(
            sorted(intervals), sorted(intervals)[1:]
        )
    )

    mode = value.get("mode")
    counterfactual = value.get("counterfactual")
    if mode == "one_active_of_n":
        if len(candidates) < 2:
            errors.append("one_active_of_n requires at least two candidate endpoints")
        if not events or len(active_endpoint_ids) != 1:
            errors.append("one_active_of_n requires events on exactly one candidate endpoint")
    elif mode == "simultaneous_subset":
        if len(active_endpoint_ids) < 2 or not cross_source_overlap:
            errors.append(
                "simultaneous_subset requires overlapping events on at least two endpoints"
            )
    elif mode == "sequential_sources":
        if len(active_endpoint_ids) < 2:
            errors.append("sequential_sources requires at least two active endpoints")
        if any_overlap:
            errors.append("sequential_sources events must not overlap")
    elif mode == "intermittent_events":
        if not events or not has_intermittent_gap:
            errors.append(
                "intermittent_events requires a positive silent gap between events "
                "on at least one endpoint"
            )
    elif mode == "counterfactual_route_swap":
        if not events:
            errors.append("counterfactual_route_swap requires at least one event")
        if not isinstance(counterfactual, Mapping):
            errors.append("counterfactual_route_swap requires counterfactual metadata")
        else:
            permutation = counterfactual.get("endpoint_permutation")
            if not isinstance(permutation, Mapping):
                errors.append(
                    "counterfactual.endpoint_permutation must be an object"
                )
            else:
                keys = set(permutation)
                values = list(permutation.values())
                if keys != set(candidates) or set(values) != set(candidates):
                    errors.append(
                        "counterfactual.endpoint_permutation must be a bijection over "
                        "exactly the candidate endpoints"
                    )
                if len(values) != len(set(values)):
                    errors.append(
                        "counterfactual.endpoint_permutation values must be unique"
                    )
                if all(permutation.get(item) == item for item in candidates):
                    errors.append(
                        "counterfactual.endpoint_permutation must change routing"
                    )
    elif mode == "silent_negative":
        if events:
            errors.append("silent_negative must not contain events")
    if mode != "counterfactual_route_swap" and counterfactual is not None:
        errors.append(
            "counterfactual metadata is only valid for counterfactual_route_swap"
        )

    endpoint_records: dict[str, Mapping[str, Any]] | None = None
    sound_records: dict[str, Mapping[str, Any]] | None = None
    if source_endpoint_registry is not None:
        endpoint_errors = validate_source_endpoint_registry(source_endpoint_registry)
        if endpoint_errors:
            errors.extend(f"source endpoint registry: {item}" for item in endpoint_errors)
        else:
            endpoint_records = endpoint_index(source_endpoint_registry)
            for endpoint_id in candidates:
                if endpoint_id not in endpoint_records:
                    errors.append(f"candidate endpoint {endpoint_id!r} is not registered")
    if sound_asset_registry is not None:
        sound_errors = validate_sound_asset_registry(sound_asset_registry)
        if sound_errors:
            errors.extend(f"sound asset registry: {item}" for item in sound_errors)
        else:
            sound_records = sound_index(sound_asset_registry)
    if endpoint_records is not None and sound_records is not None:
        for index, event in enumerate(events):
            endpoint = endpoint_records.get(event["source_endpoint_id"])
            sound = sound_records.get(event["sound_asset_id"])
            if sound is None:
                errors.append(f"events[{index}].sound_asset_id is not registered")
            elif endpoint is not None:
                errors.extend(
                    source_sound_compatibility_errors(
                        endpoint, sound, owner=f"events[{index}]"
                    )
                )
                if mode not in sound["permitted_event_usage"]:
                    errors.append(
                        f"events[{index}] sound asset {sound['sound_asset_id']!r} "
                        f"does not permit AudioProgram mode {mode!r}"
                    )
    return errors


def load_audio_program(
    path: str | Path,
    *,
    source_endpoint_registry: Mapping[str, Any] | None = None,
    sound_asset_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = load_json(path)
    errors = validate_audio_program(
        value,
        source_endpoint_registry=source_endpoint_registry,
        sound_asset_registry=sound_asset_registry,
    )
    if errors:
        raise AudioProgramError(errors)
    return value


def materialize_audio_program_variant(
    value: Mapping[str, Any],
    variant_id: str,
    *,
    source_endpoint_registry: Mapping[str, Any] | None = None,
    sound_asset_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize A/B routing while changing only endpoint route fields.

    Non-counterfactual programs have only the canonical ``A`` representation.
    A counterfactual program keeps all event timing, sound, gain, fade and stem
    policy fixed; variant ``B`` applies the declared endpoint permutation.
    """

    errors = validate_audio_program(
        value,
        source_endpoint_registry=source_endpoint_registry,
        sound_asset_registry=sound_asset_registry,
    )
    if errors:
        raise AudioProgramError(errors)
    result = deepcopy(dict(value))
    if value["mode"] != "counterfactual_route_swap":
        if variant_id != "A":
            raise AudioProgramError([
                "non-counterfactual AudioProgram supports only variant 'A'"
            ])
        return result
    counterfactual = value["counterfactual"]
    if variant_id not in counterfactual["variants"]:
        raise AudioProgramError([
            f"unknown counterfactual AudioProgram variant {variant_id!r}"
        ])
    if variant_id == counterfactual["mapped_variant"]:
        permutation = counterfactual["endpoint_permutation"]
        for event in result["events"]:
            event["source_endpoint_id"] = permutation[event["source_endpoint_id"]]
        result = bind_audio_program_hash(result)
        rebound_errors = validate_audio_program(
            result,
            source_endpoint_registry=source_endpoint_registry,
            sound_asset_registry=sound_asset_registry,
        )
        if rebound_errors:
            raise AudioProgramError(rebound_errors)
    return result


@dataclass(frozen=True)
class CompiledAudioEvent:
    event_id: str
    source_endpoint_id: str
    sound_asset_id: str
    start_tick: int
    end_tick_exclusive: int
    start_sample: int
    end_sample_exclusive: int
    source_start_sample: int
    source_end_sample_exclusive: int
    linear_gain: float
    fade_samples: int
    render_source_stem: bool


@dataclass(frozen=True)
class CompiledAudioProgram:
    program_id: str
    revision: str
    mode: str
    frame_count: int
    candidate_source_endpoint_ids: tuple[str, ...]
    active_source_endpoint_ids: tuple[str, ...]
    silent_source_endpoint_ids: tuple[str, ...]
    events: tuple[CompiledAudioEvent, ...]

    def current_event_by_source(self, frame_index: int) -> Mapping[str, str | None]:
        """Derive one frame state on demand; no dense episode schema is frozen."""

        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < self.frame_count
        ):
            raise ValueError(f"frame_index must be in 0..{self.frame_count - 1}")
        result: dict[str, str | None] = {
            source_id: None for source_id in self.candidate_source_endpoint_ids
        }
        frame_sample = sample_boundary(frame_index)
        for event in self.events:
            if event.start_sample <= frame_sample < event.end_sample_exclusive:
                result[event.source_endpoint_id] = event.event_id
        return result


def compile_audio_program(
    value: Mapping[str, Any],
    *,
    source_endpoint_registry: Mapping[str, Any] | None = None,
    sound_asset_registry: Mapping[str, Any] | None = None,
) -> CompiledAudioProgram:
    errors = validate_audio_program(
        value,
        source_endpoint_registry=source_endpoint_registry,
        sound_asset_registry=sound_asset_registry,
    )
    if errors:
        raise AudioProgramError(errors)
    events = tuple(
        CompiledAudioEvent(
            event_id=item["event_id"],
            source_endpoint_id=item["source_endpoint_id"],
            sound_asset_id=item["sound_asset_id"],
            start_tick=item["start_tick"],
            end_tick_exclusive=item["end_tick_exclusive"],
            start_sample=item["start_sample"],
            end_sample_exclusive=item["end_sample_exclusive"],
            source_start_sample=item["source_start_sample"],
            source_end_sample_exclusive=item["source_end_sample_exclusive"],
            linear_gain=float(item["linear_gain"]),
            fade_samples=item["fade_samples"],
            render_source_stem=item["render_source_stem"],
        )
        for item in value["events"]
    )
    active = tuple(sorted({item.source_endpoint_id for item in events}))
    candidates = tuple(value["candidate_source_endpoint_ids"])
    silent = tuple(item for item in candidates if item not in active)
    return CompiledAudioProgram(
        program_id=value["program_id"],
        revision=value["revision"],
        mode=value["mode"],
        frame_count=value["timeline"]["frame_count"],
        candidate_source_endpoint_ids=candidates,
        active_source_endpoint_ids=active,
        silent_source_endpoint_ids=silent,
        events=events,
    )


def compile_audio_program_variant(
    value: Mapping[str, Any],
    variant_id: str,
    *,
    source_endpoint_registry: Mapping[str, Any] | None = None,
    sound_asset_registry: Mapping[str, Any] | None = None,
) -> CompiledAudioProgram:
    materialized = materialize_audio_program_variant(
        value,
        variant_id,
        source_endpoint_registry=source_endpoint_registry,
        sound_asset_registry=sound_asset_registry,
    )
    return compile_audio_program(
        materialized,
        source_endpoint_registry=source_endpoint_registry,
        sound_asset_registry=sound_asset_registry,
    )
