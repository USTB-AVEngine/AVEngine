from __future__ import annotations

import wave
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import load_json, sha256_file
from avengine.m4.audio import AudioContractError
from avengine.m6.audio_program import AudioProgramError, bind_audio_program_hash
from avengine.m6.audio_render import (
    assemble_audio_program_dry_buses,
    assemble_semantic_audio_program_dry_buses,
)
from avengine.m6.registry import bind_content_hash
from avengine.m6.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRIES = ROOT / "examples/m6/registries"
PROGRAMS = ROOT / "examples/m6x/fixed_apartment/audio_programs"


def _semantic_fixture(tmp_path: Path) -> tuple[dict, dict, dict, dict]:
    path = tmp_path / "speech.wav"
    samples = np.full(45_912, 8_000, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())
    endpoints = {
        "source1_emitter": "source1",
        "source2_emitter": "source2",
    }
    contents = {
        "schema": "avengine_semantic_sound_content_registry_v1",
        "registry_id": "semantic_contents",
        "revision": "planning_v1",
        "contents": [
            {
                "content_id": "speech_content",
                "sound_asset_id": "speech_asset",
                "voice_id": "speaker",
                "source_audio_uri": "semantic://speech_content",
                "sample_rate_hz": 16_000,
                "channel_count": 1,
                "sample_count": 45_912,
            }
        ],
    }
    bindings = {
        "speech_content": {
            "content_id": "speech_content",
            "path": str(path),
            "sample_rate_hz": 16_000,
            "channel_count": 1,
            "sample_count": 45_912,
        }
    }
    program = {
        "schema": "avengine_semantic_audio_program_v1",
        "program_id": "episode_audio",
        "revision": "planning_v1",
        "mode": "one_active_of_n",
        "timeline": {
            "time_base_hz": 48_000,
            "ticks_per_frame": 3_200,
            "video_fps": 15,
            "frame_count": 75,
            "sample_rate_hz": 16_000,
            "ticks_per_sample": 3,
            "sample_count": 80_000,
        },
        "candidate_source_endpoint_ids": sorted(endpoints),
        "events": [
            {
                "event_id": "target_speech",
                "source_endpoint_id": "source1_emitter",
                "content_id": "speech_content",
                "start_tick": 7_467 * 3,
                "end_tick_exclusive": 53_379 * 3,
                "start_sample": 7_467,
                "end_sample_exclusive": 53_379,
                "source_start_sample": 0,
                "source_end_sample_exclusive": 45_912,
                "source_sample_rate_hz": 16_000,
                "source_channel_count": 1,
                "source_sample_count": 45_912,
                "linear_gain": 1.0,
                "fade_samples": 0,
                "render_source_stem": True,
            }
        ],
        "source_specific_stems": True,
        "admission_state": "research",
        "program_content_sha256": "PENDING",
    }
    return bind_audio_program_hash(program), endpoints, contents, bindings


def _write_pcm16(path: Path, value: int) -> None:
    samples = np.full(80_000, value, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


def _registries_and_bindings(
    tmp_path: Path,
) -> tuple[dict, dict, dict[str, dict[str, str]]]:
    endpoints = load_source_endpoint_registry(REGISTRIES / "source_endpoints_v1.json")
    sounds = deepcopy(load_sound_asset_registry(REGISTRIES / "sound_assets_v1.json"))
    required = {
        "dog_beagle_v2_scheduled_dry": 12_000,
        "human_speech_libritts_1594_16k_v1": 8_000,
    }
    bindings: dict[str, dict[str, str]] = {}
    for sound_id, value in required.items():
        path = tmp_path / f"{sound_id}.wav"
        _write_pcm16(path, value)
        digest = sha256_file(path)
        record = next(
            item
            for item in sounds["sound_assets"]
            if item["sound_asset_id"] == sound_id
        )
        record["dry_audio"].update(
            {
                "sha256": digest,
                "sample_rate_hz": 16_000,
                "channel_count": 1,
                "sample_count": 80_000,
            }
        )
        bindings[sound_id] = {"path": str(path), "sha256": digest}
    return endpoints, bind_content_hash(sounds), bindings


def test_s4_uses_compiled_events_and_authenticated_asset_bindings(
    tmp_path: Path,
) -> None:
    endpoints, sounds, bindings = _registries_and_bindings(tmp_path)
    program = load_json(PROGRAMS / "m6x_s4_overlapping_sources_v1.json")

    result = assemble_audio_program_dry_buses(
        program,
        "A",
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
        asset_bindings=bindings,
    )

    assert result.materialized_program == program
    assert result.compiled_program.mode == "simultaneous_subset"
    assert result.compiled_program.active_source_endpoint_ids == (
        "m6x_dog0_muzzle",
        "m6x_human0_mouth",
    )
    assert result.dry_audio.source_ids == tuple(
        program["candidate_source_endpoint_ids"]
    )
    assert len(result.dry_audio.placement_receipts) == len(program["events"])
    dog = result.dry_audio.buses["m6x_dog0_muzzle"]
    human = result.dry_audio.buses["m6x_human0_mouth"]
    assert not np.any(dog[:16_000])
    assert np.any(dog[16_080:20_720])
    assert not np.any(dog[20_800:32_000])
    assert not np.any(human[:8_000])
    assert np.any(human[8_080:55_920])
    assert not np.any(human[56_000:])
    assert np.any(dog[16_080:20_720]) and np.any(human[16_080:20_720])
    metadata = result.dry_audio.metadata()
    assert (
        metadata["assembly_content_sha256"] == result.dry_audio.assembly_content_sha256
    )
    assert {
        receipt["dry_asset"]["sha256"]
        for receipt in result.dry_audio.placement_receipts
    } == {record["sha256"] for record in bindings.values()}


def test_silent_program_retains_candidate_zero_buses(tmp_path: Path) -> None:
    endpoints, sounds, _bindings = _registries_and_bindings(tmp_path)
    program = load_json(PROGRAMS / "m6x_s2_silent_negative_v1.json")

    result = assemble_audio_program_dry_buses(
        program,
        "A",
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
        asset_bindings={},
    )

    assert result.compiled_program.active_source_endpoint_ids == ()
    assert result.compiled_program.silent_source_endpoint_ids == tuple(
        program["candidate_source_endpoint_ids"]
    )
    assert result.dry_audio.placement_receipts == ()
    assert all(
        np.array_equal(bus, np.zeros(80_000, dtype=np.float64))
        for bus in result.dry_audio.buses.values()
    )


def test_asset_binding_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    endpoints, sounds, bindings = _registries_and_bindings(tmp_path)
    program = load_json(PROGRAMS / "m6x_s4_overlapping_sources_v1.json")
    bad_bindings = deepcopy(bindings)
    bad_bindings["dog_beagle_v2_scheduled_dry"]["sha256"] = "0" * 64

    with pytest.raises(AudioContractError, match="hash conflicts with asset binding"):
        assemble_audio_program_dry_buses(
            program,
            "A",
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
            asset_bindings=bad_bindings,
        )


def test_stem_export_flags_do_not_block_internal_source_bus_assembly(
    tmp_path: Path,
) -> None:
    endpoints, sounds, bindings = _registries_and_bindings(tmp_path)
    program = load_json(PROGRAMS / "m6x_s4_overlapping_sources_v1.json")
    program["source_specific_stems"] = False
    for event in program["events"]:
        event["render_source_stem"] = False
    program = bind_audio_program_hash(program)

    result = assemble_audio_program_dry_buses(
        program,
        "A",
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
        asset_bindings=bindings,
    )

    assert set(result.dry_audio.buses) == set(program["candidate_source_endpoint_ids"])
    assert all(np.any(bus) for bus in result.dry_audio.buses.values())


def test_registries_are_required() -> None:
    sounds = load_sound_asset_registry(REGISTRIES / "sound_assets_v1.json")
    program = load_json(PROGRAMS / "m6x_s2_silent_negative_v1.json")

    with pytest.raises(AudioProgramError, match="source_endpoint_registry"):
        assemble_audio_program_dry_buses(
            program,
            "A",
            source_endpoint_registry=None,  # type: ignore[arg-type]
            sound_asset_registry=sounds,
            asset_bindings={},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("sample_rate_hz", 44_100), ("channel_count", 2)),
)
def test_audio_program_v1_requires_canonical_mono_registry_audio(
    field: str,
    value: int,
) -> None:
    endpoints = load_source_endpoint_registry(REGISTRIES / "source_endpoints_v1.json")
    sounds = deepcopy(load_sound_asset_registry(REGISTRIES / "sound_assets_v1.json"))
    human = next(
        item
        for item in sounds["sound_assets"]
        if item["sound_asset_id"] == "human_speech_libritts_1594_16k_v1"
    )
    human["dry_audio"][field] = value
    sounds = bind_content_hash(sounds)
    program = load_json(PROGRAMS / "m6x_s4_overlapping_sources_v1.json")

    with pytest.raises(AudioProgramError, match="canonical mono 16000 Hz"):
        assemble_audio_program_dry_buses(
            program,
            "A",
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
            asset_bindings={},
        )


def test_semantic_audio_program_keeps_exact_non_frame_aligned_event(
    tmp_path: Path,
) -> None:
    program, endpoints, contents, bindings = _semantic_fixture(tmp_path)

    result = assemble_semantic_audio_program_dry_buses(
        program,
        "A",
        source_endpoint_ids=endpoints,
        semantic_content_registry=contents,
        content_bindings=bindings,
    )

    event = result.compiled_program.events[0]
    assert (event.start_sample, event.end_sample_exclusive) == (7_467, 53_379)
    assert not np.any(result.dry_audio.buses["source1_emitter"][:7_467])
    assert np.any(result.dry_audio.buses["source1_emitter"][7_467:53_379])
    assert not np.any(result.dry_audio.buses["source1_emitter"][53_379:])
    assert not np.any(result.dry_audio.buses["source2_emitter"])
    metadata = result.dry_audio.metadata()
    assert metadata["binding_mode"] == (
        "semantic_content_id_and_declared_audio_metadata_v1"
    )
    assert "bus_content_sha256" in metadata
    assert "bus_float64_le_sha256" not in metadata


@pytest.mark.parametrize("mutation", ("metadata", "gain", "binding_coverage"))
def test_semantic_audio_program_fails_closed_on_semantic_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    program, endpoints, contents, bindings = _semantic_fixture(tmp_path)
    if mutation == "metadata":
        contents["contents"][0]["sample_count"] += 1
    elif mutation == "gain":
        program["events"][0]["linear_gain"] = -1.0
        program = bind_audio_program_hash(program)
    else:
        bindings["unused"] = dict(bindings["speech_content"])
        bindings["unused"]["content_id"] = "unused"

    with pytest.raises((AudioProgramError, AudioContractError)):
        assemble_semantic_audio_program_dry_buses(
            program,
            "A",
            source_endpoint_ids=endpoints,
            semantic_content_registry=contents,
            content_bindings=bindings,
        )


@pytest.mark.parametrize(
    ("scope", "key", "value"),
    [
        ("program", "file_sha256", "0" * 64),
        ("event", "byte_size", 123),
    ],
)
def test_semantic_audio_program_rejects_unknown_fields_after_hash_rebind(
    tmp_path: Path, scope: str, key: str, value: object
) -> None:
    program, endpoints, contents, bindings = _semantic_fixture(tmp_path)
    if scope == "program":
        program[key] = value
    else:
        program["events"][0][key] = value
    program = bind_audio_program_hash(program)
    with pytest.raises(AudioProgramError):
        assemble_semantic_audio_program_dry_buses(
            program,
            "A",
            source_endpoint_ids=endpoints,
            semantic_content_registry=contents,
            content_bindings=bindings,
        )
