"""Render validated M6 AudioPrograms into exact named dry-audio buses.

This module is the shared bridge between the data-driven M6 scheduling
contract and the deterministic M5.1 dry-audio assembler.  Callers resolve
registry URIs to local audio bindings; the bridge materializes the requested
routing variant, validates both registries, and projects the compiled events
without introducing a second scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from avengine.m5_1.dry_audio import (
    DryAudioAssembly,
    DryAudioClipSpec,
    assemble_dry_audio_buses,
)
from avengine.m6.audio_program import (
    AudioProgramError,
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
    dry_audio: DryAudioAssembly


def _require_mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AudioProgramError([f"{owner} is required and must be a mapping"])
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
                "AudioProgram timeline.sample_count differs from the exact "
                "dry-audio clip boundary"
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
                    f"sound asset {event.sound_asset_id!r} must be canonical mono "
                    f"{timeline['sample_rate_hz']} Hz for AudioProgram v1 dry assembly"
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


__all__ = [
    "AudioProgramDryAssembly",
    "assemble_audio_program_dry_buses",
]
