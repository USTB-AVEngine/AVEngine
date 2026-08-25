"""Render validated M6 AudioPrograms into exact named dry-audio buses.

This module is the shared bridge between the data-driven M6 scheduling
contract and the deterministic M5.1 dry-audio assembler.  Callers resolve
registry URIs to local audio bindings; the bridge materializes the requested
routing variant, validates both registries, and projects the compiled events
without introducing a second scheduler.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m5_1.dry_audio import (
    DryAudioAssembly,
    DryAudioClipSpec,
    SemanticDryAudioAssembly,
    assemble_dry_audio_buses,
    assemble_semantic_dry_audio_buses,
)
from avengine.timeline.audio_program import (
    AudioProgramError,
    CompiledAudioEvent,
    CompiledAudioProgram,
    compile_audio_program,
    materialize_audio_program_variant,
)
from avengine.m6.sources import sound_index


@dataclass(frozen=True)
class AudioProgramDryAssembly:
    """One materialized program and its exact deterministic dry buses."""

    materialized_program: Mapping[str, Any]
    compiled_program: CompiledAudioProgram
    dry_audio: DryAudioAssembly | SemanticDryAudioAssembly


def _require_mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AudioProgramError([f"{owner} is required and must be a mapping"])
    return value


def _stable_string(value: Any, *, owner: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "_.-") for character in value)
    ):
        raise AudioProgramError([f"{owner} must be a stable semantic ID"])
    return value


def assemble_audio_program_dry_buses(
    program: Mapping[str, Any],
    variant_id: str,
    *,
    source_endpoint_registry: Mapping[str, Any],
    sound_asset_registry: Mapping[str, Any],
    asset_bindings: Mapping[str, Any],
) -> AudioProgramDryAssembly:
    """Materialize, compile, and assemble one AudioProgram routing variant."""

    endpoints = _require_mapping(
        source_endpoint_registry, owner="source_endpoint_registry"
    )
    sounds = _require_mapping(sound_asset_registry, owner="sound_asset_registry")
    bindings = _require_mapping(asset_bindings, owner="asset_bindings")
    materialized = materialize_audio_program_variant(
        program,
        variant_id,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    compiled = compile_audio_program(
        materialized,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    timeline = materialized["timeline"]
    clip = DryAudioClipSpec.from_values(
        frame_count=timeline["frame_count"],
        fps_numerator=timeline["video_fps"],
        sample_rate_hz=timeline["sample_rate_hz"],
    )
    if clip.sample_count != timeline["sample_count"]:
        raise AudioProgramError(
            [
                (
                    "AudioProgram timeline.sample_count differs from the exact "
                    "dry-audio clip boundary"
                )
            ]
        )

    sound_records = sound_index(sounds)
    event_mappings = []
    for event in compiled.events:
        dry = sound_records[event.sound_asset_id]["dry_audio"]
        if (
            dry["sample_rate_hz"] != timeline["sample_rate_hz"]
            or dry["channel_count"] != 1
        ):
            raise AudioProgramError(
                [
                    (
                        f"sound asset {event.sound_asset_id!r} must be canonical mono "
                        f"{timeline['sample_rate_hz']} Hz for AudioProgram v1 dry assembly"
                    )
                ]
            )
        event_mappings.append(
            {
                "event_id": event.event_id,
                "source_id": event.source_endpoint_id,
                "start_sample": event.start_sample,
                "end_sample_exclusive": event.end_sample_exclusive,
                "dry_asset_id": event.sound_asset_id,
                "dry_asset_sha256": dry["sha256"],
                "dry_clip_start_sample": event.source_start_sample,
                "dry_clip_end_sample_exclusive": event.source_end_sample_exclusive,
                "linear_gain": event.linear_gain,
                "fade_samples": event.fade_samples,
            }
        )
    dry_audio = assemble_dry_audio_buses(
        event_mappings,
        source_ids=compiled.candidate_source_endpoint_ids,
        clip=clip,
        asset_bindings=bindings,
    )
    return AudioProgramDryAssembly(
        materialized_program=materialized,
        compiled_program=compiled,
        dry_audio=dry_audio,
    )


def assemble_semantic_audio_program_dry_buses(
    program: Mapping[str, Any],
    variant_id: str,
    *,
    source_endpoint_ids: Mapping[str, str],
    semantic_content_registry: Mapping[str, Any],
    content_bindings: Mapping[str, Mapping[str, Any]],
) -> AudioProgramDryAssembly:
    """Compile one planning AudioProgram through semantic content bindings.

    ``source_endpoint_ids`` maps each candidate endpoint to its semantic slot.
    The values are evidence only; the AudioProgram endpoint IDs remain the bus
    IDs.  No sound-registry or local file digest is accepted by this branch.
    """

    endpoints = _require_mapping(source_endpoint_ids, owner="source_endpoint_ids")
    registry = _require_mapping(
        semantic_content_registry, owner="semantic_content_registry"
    )
    bindings = _require_mapping(content_bindings, owner="content_bindings")
    if variant_id != "A":
        raise AudioProgramError(
            ["semantic planning AudioProgram currently supports only variant 'A'"]
        )
    if not isinstance(program, Mapping):
        raise AudioProgramError(["program is required and must be a mapping"])
    program_fields = {
        "schema",
        "program_id",
        "revision",
        "mode",
        "timeline",
        "candidate_source_endpoint_ids",
        "events",
        "source_specific_stems",
        "admission_state",
        "program_content_sha256",
    }
    if (
        set(program) != program_fields
        or program.get("schema") != "avengine_semantic_audio_program_v1"
        or program.get("mode") != "one_active_of_n"
        or program.get("source_specific_stems") is not True
        or program.get("admission_state") != "research"
    ):
        raise AudioProgramError(["semantic planning AudioProgram header is invalid"])
    program_id = _stable_string(program.get("program_id"), owner="program_id")
    revision = _stable_string(program.get("revision"), owner="revision")
    declared_content = program.get("program_content_sha256")
    expected_content = canonical_json_sha256(
        {
            key: value
            for key, value in program.items()
            if key != "program_content_sha256"
        }
    )
    if declared_content != expected_content:
        raise AudioProgramError(
            ["program_content_sha256 does not match canonical content"]
        )
    candidates = program.get("candidate_source_endpoint_ids")
    events = program.get("events")
    timeline = program.get("timeline")
    if (
        not isinstance(candidates, list)
        or candidates != sorted(set(candidates))
        or set(candidates) != set(endpoints)
        or any(
            _stable_string(endpoint_id, owner="candidate endpoint") != endpoint_id
            for endpoint_id in candidates
        )
        or any(
            _stable_string(slot, owner="semantic source slot") != slot
            for slot in endpoints.values()
        )
        or len(set(endpoints.values())) != len(endpoints)
        or not isinstance(events, list)
        or len(events) != 1
        or not isinstance(timeline, Mapping)
    ):
        raise AudioProgramError(["semantic planning AudioProgram structure is invalid"])
    records = registry.get("contents")
    if (
        registry.get("schema") != "avengine_semantic_sound_content_registry_v1"
        or set(registry) != {"schema", "registry_id", "revision", "contents"}
        or not isinstance(registry.get("registry_id"), str)
        or not registry.get("registry_id")
        or not isinstance(registry.get("revision"), str)
        or not registry.get("revision")
        or not isinstance(records, list)
    ):
        raise AudioProgramError(["semantic content registry structure is invalid"])
    record_fields = {
        "content_id",
        "sound_asset_id",
        "voice_id",
        "source_audio_uri",
        "sample_rate_hz",
        "channel_count",
        "sample_count",
    }
    if any(
        not isinstance(record, Mapping)
        or set(record) != record_fields
        or not isinstance(record.get("source_audio_uri"), str)
        or not record.get("source_audio_uri")
        for record in records
    ):
        raise AudioProgramError(["semantic content record structure is invalid"])
    by_id = {
        str(record.get("content_id")): record
        for record in records
        if isinstance(record, Mapping)
    }
    if len(by_id) != len(records):
        raise AudioProgramError(["semantic content IDs must be unique"])
    if set(bindings) != set(by_id):
        raise AudioProgramError(
            ["semantic content bindings must exactly cover the content registry"]
        )
    binding_fields = {
        "content_id",
        "path",
        "sample_rate_hz",
        "channel_count",
        "sample_count",
    }
    for index, record in enumerate(records):
        content_id = _stable_string(
            record.get("content_id"), owner=f"contents[{index}].content_id"
        )
        _stable_string(
            record.get("sound_asset_id"),
            owner=f"contents[{index}].sound_asset_id",
        )
        _stable_string(record.get("voice_id"), owner=f"contents[{index}].voice_id")
        expected_binding = {
            "content_id": content_id,
            "sample_rate_hz": record.get("sample_rate_hz"),
            "channel_count": record.get("channel_count"),
            "sample_count": record.get("sample_count"),
        }
        binding = bindings.get(content_id)
        if (
            any(
                type(value) is not int or value < 1
                for key, value in expected_binding.items()
                if key != "content_id"
            )
            or not isinstance(binding, Mapping)
            or set(binding) != binding_fields
            or not isinstance(binding.get("path"), str)
            or not binding.get("path")
            or any(binding.get(key) != value for key, value in expected_binding.items())
        ):
            raise AudioProgramError(
                [f"semantic content {content_id!r} binding structure or metadata drift"]
            )
    required_timeline = {
        "time_base_hz": 48_000,
        "ticks_per_frame": 3_200,
        "video_fps": 15,
        "frame_count": 75,
        "sample_rate_hz": 16_000,
        "ticks_per_sample": 3,
        "sample_count": 80_000,
    }
    if dict(timeline) != required_timeline:
        raise AudioProgramError(["semantic planning AudioProgram timeline drift"])
    event_mappings = []
    active: set[str] = set()
    compiled_values = []
    event_fields = {
        "event_id",
        "source_endpoint_id",
        "content_id",
        "start_tick",
        "end_tick_exclusive",
        "start_sample",
        "end_sample_exclusive",
        "source_start_sample",
        "source_end_sample_exclusive",
        "source_sample_rate_hz",
        "source_channel_count",
        "source_sample_count",
        "linear_gain",
        "fade_samples",
        "render_source_stem",
    }

    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or set(event) != event_fields:
            raise AudioProgramError([f"events[{index}] must be an object"])
        endpoint_id = event.get("source_endpoint_id")
        content_id = event.get("content_id")
        if endpoint_id not in endpoints or content_id not in by_id:
            raise AudioProgramError([f"events[{index}] semantic binding is unresolved"])
        record = by_id[str(content_id)]
        expected = {
            "content_id": content_id,
            "sample_rate_hz": event.get("source_sample_rate_hz"),
            "channel_count": event.get("source_channel_count"),
            "sample_count": event.get("source_sample_count"),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise AudioProgramError(
                [f"events[{index}] semantic content metadata drift"]
            )
        binding = bindings.get(str(content_id))
        if (
            not isinstance(binding, Mapping)
            or binding.get("content_id") != content_id
            or any(binding.get(key) != value for key, value in expected.items())
        ):
            raise AudioProgramError([f"events[{index}] semantic local binding drift"])
        start = event.get("start_sample")
        end = event.get("end_sample_exclusive")
        source_start = event.get("source_start_sample")
        source_end = event.get("source_end_sample_exclusive")
        if not all(
            type(value) is int for value in (start, end, source_start, source_end)
        ):
            raise AudioProgramError([f"events[{index}] sample bounds are invalid"])
        if (
            event.get("start_tick") != start * 3
            or event.get("end_tick_exclusive") != end * 3
            or not 0 <= start < end <= 80_000
            or not 0 <= source_start < source_end
            or end - start != source_end - source_start
        ):
            raise AudioProgramError([f"events[{index}] sample/tick closure drift"])
        fade = event.get("fade_samples", 0)
        gain = event.get("linear_gain", 1.0)
        if (
            type(fade) is not int
            or fade < 0
            or isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(float(gain))
            or float(gain) < 0.0
        ):
            raise AudioProgramError([f"events[{index}] gain/fade is invalid"])
        active.add(str(endpoint_id))
        event_id = _stable_string(
            event.get("event_id"), owner=f"events[{index}].event_id"
        )
        event_mappings.append(
            {
                "event_id": event_id,
                "source_id": endpoint_id,
                "start_sample": start,
                "end_sample_exclusive": end,
                "content_id": content_id,
                "dry_clip_start_sample": source_start,
                "dry_clip_end_sample_exclusive": source_end,
                "linear_gain": float(gain),
                "fade_samples": fade,
            }
        )
        compiled_values.append(
            CompiledAudioEvent(
                event_id=event_id,
                source_endpoint_id=str(endpoint_id),
                sound_asset_id=str(content_id),
                start_tick=start * 3,
                end_tick_exclusive=end * 3,
                start_sample=start,
                end_sample_exclusive=end,
                source_start_sample=source_start,
                source_end_sample_exclusive=source_end,
                linear_gain=float(gain),
                fade_samples=fade,
                render_source_stem=bool(event.get("render_source_stem", True)),
            )
        )
    compiled = CompiledAudioProgram(
        program_id=program_id,
        revision=revision,
        mode="one_active_of_n",
        frame_count=75,
        candidate_source_endpoint_ids=tuple(candidates),
        active_source_endpoint_ids=tuple(sorted(active)),
        silent_source_endpoint_ids=tuple(
            endpoint for endpoint in candidates if endpoint not in active
        ),
        events=tuple(compiled_values),
    )
    clip = DryAudioClipSpec.from_values(
        frame_count=75, fps_numerator=15, sample_rate_hz=16_000
    )
    dry_audio = assemble_semantic_dry_audio_buses(
        event_mappings,
        source_ids=tuple(candidates),
        clip=clip,
        content_bindings=bindings,
    )
    return AudioProgramDryAssembly(
        materialized_program=dict(program),
        compiled_program=compiled,
        dry_audio=dry_audio,
    )


__all__ = [
    "AudioProgramDryAssembly",
    "assemble_audio_program_dry_buses",
    "assemble_semantic_audio_program_dry_buses",
]
