from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import wave

import numpy as np
import pytest

from avengine.contracts.json_io import load_json, sha256_file
from avengine.m4.audio import AudioContractError
from avengine.m6.audio_program import AudioProgramError, bind_audio_program_hash
from avengine.m6.audio_render import assemble_audio_program_dry_buses
from avengine.m6.registry import bind_content_hash
from avengine.m6.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRIES = ROOT / "examples/m6/registries"
PROGRAMS = ROOT / "examples/m6x/fixed_apartment/audio_programs"


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
    sounds = deepcopy(
        load_sound_asset_registry(REGISTRIES / "sound_assets_v1.json")
    )
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
            item for item in sounds["sound_assets"] if item["sound_asset_id"] == sound_id
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
        metadata["assembly_content_sha256"]
        == result.dry_audio.assembly_content_sha256
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

    assert set(result.dry_audio.buses) == set(
        program["candidate_source_endpoint_ids"]
    )
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
    sounds = deepcopy(
        load_sound_asset_registry(REGISTRIES / "sound_assets_v1.json")
    )
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
