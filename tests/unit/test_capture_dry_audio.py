from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.spatial_audio.audio import AudioContractError
from avengine.capture.dry_audio import (
    RESAMPLING_ALGORITHM,
    DryAudioClipSpec,
    assemble_dry_audio_buses,
    assemble_semantic_dry_audio_buses,
    deterministic_resample_mono,
    parse_dry_audio_events,
    read_authenticated_mono_pcm_wav,
)


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, (list, tuple)):
        return {key for item in value for key in _keys(item)}
    return set()


def _write_pcm16(
    path: Path,
    samples: list[int],
    *,
    sample_rate_hz: int = 4,
    channel_count: int = 1,
) -> None:
    payload = np.asarray(samples, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channel_count)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(payload.tobytes())


def _event(
    *,
    event_id: str,
    source_id: str,
    start: int,
    end: int,
    asset_id: str,
    digest: str,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "source_id": source_id,
        "start_sample": start,
        "end_sample_exclusive": end,
        "dry_audio_asset_id": asset_id,
        "dry_audio_asset_sha256": digest,
    }


def _float64_hash(samples: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(samples, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def test_arbitrary_clip_frame_rate_has_exact_rounded_boundaries() -> None:
    clip = DryAudioClipSpec.from_values(
        frame_count=7,
        fps_numerator=4,
        sample_rate_hz=8,
    )
    assert clip.sample_count == 14
    assert [clip.sample_boundary(index) for index in range(8)] == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
    ]

    fractional = DryAudioClipSpec.from_values(
        frame_count=3,
        fps_numerator=3,
        fps_denominator=2,
        sample_rate_hz=5,
    )
    assert fractional.sample_count == 10
    assert [fractional.sample_boundary(index) for index in range(4)] == [0, 3, 7, 10]


def test_authenticated_mono_read_and_deterministic_linear_resample(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.wav"
    _write_pcm16(source_path, [0, 16384, -16384, 8192])
    digest = sha256_file(source_path)

    decoded = read_authenticated_mono_pcm_wav(source_path, expected_sha256=digest)
    assert decoded.sample_rate_hz == 4
    assert decoded.sample_width_bytes == 2
    assert np.array_equal(decoded.samples, [0.0, 0.5, -0.5, 0.25])

    result = deterministic_resample_mono(
        decoded.samples,
        source_sample_rate_hz=4,
        target_sample_rate_hz=8,
    )
    assert np.array_equal(
        result,
        [0.0, 0.25, 0.5, 0.0, -0.5, -0.125, 0.25, 0.25],
    )
    assert result.dtype == np.float64
    assert result.flags.c_contiguous


def test_assembly_supports_simultaneous_overlap_crop_padding_gain_and_fade(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "voice.wav"
    _write_pcm16(source_path, [0, 8192, 16384, 24576])
    digest = sha256_file(source_path)
    clip = DryAudioClipSpec.from_values(
        frame_count=4,
        fps_numerator=4,
        sample_rate_hz=8,
    )
    bindings = {"voice_asset": {"path": "voice.wav", "sha256": digest}}
    events = [
        {
            **_event(
                event_id="dog_primary",
                source_id="dog_source",
                start=0,
                end=6,
                asset_id="voice_asset",
                digest=digest,
            ),
            "start_frame": 0,
            "end_frame_exclusive": 3,
            "fade_in_samples": 2,
            "fade_out_samples": 2,
        },
        {
            "event_id": "human_simultaneous",
            "source_id": "human_source",
            "start_sample": 0,
            "end_sample_exclusive": 8,
            "start_frame": 0,
            "end_frame_exclusive": 4,
            "dry_asset_id": "voice_asset",
            "dry_asset_sha256": digest,
            "gain": 0.5,
        },
        {
            "event_id": "dog_layer",
            "source_id": "dog_source",
            "start_sample": 2,
            "end_sample_exclusive": 8,
            "start_frame": 1,
            "end_frame_exclusive": 4,
            "dry_audio_asset_path": str(source_path),
            "dry_audio_asset_sha256": digest,
            "dry_clip": {"start_sample": 2, "end_sample_exclusive": 4},
            "linear_gain": 0.5,
        },
    ]

    result = assemble_dry_audio_buses(
        events,
        source_ids=("dog_source", "human_source"),
        clip=clip,
        asset_bindings=bindings,
        asset_root=tmp_path,
    )

    assert result.source_ids == ("dog_source", "human_source")
    assert all(bus.shape == (8,) for bus in result.buses.values())
    assert np.allclose(
        result.buses["dog_source"],
        [0.0, 0.125, 0.5, 0.6875, 0.875, 0.375, 0.0, 0.0],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.allclose(
        result.buses["human_source"],
        [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.375],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert [receipt["event_id"] for receipt in result.placement_receipts] == [
        "dog_primary",
        "human_simultaneous",
        "dog_layer",
    ]
    primary, _human, layer = result.placement_receipts
    assert primary["fit"] == {
        "copied_sample_count": 6,
        "cropped_tail_sample_count": 2,
        "zero_padded_tail_sample_count": 0,
    }
    assert layer["fit"] == {
        "copied_sample_count": 4,
        "cropped_tail_sample_count": 0,
        "zero_padded_tail_sample_count": 2,
    }
    assert primary["resampling"]["algorithm"] == RESAMPLING_ALGORITHM
    assert primary["resampling"]["performed"] is True
    assert result.bus_float64_le_sha256 == {
        source_id: _float64_hash(bus) for source_id, bus in result.buses.items()
    }
    assert len(result.assembly_content_sha256) == 64
    metadata = result.metadata()
    declared_content_hash = metadata.pop("assembly_content_sha256")
    assert canonical_json_sha256(metadata) == declared_content_hash
    assert metadata["qualification_claim"] is False
    assert all(not bus.flags.writeable for bus in result.buses.values())

    # Input mapping order cannot alter float addition order or receipts.
    reversed_result = assemble_dry_audio_buses(
        list(reversed(events)),
        source_ids=("dog_source", "human_source"),
        clip=clip,
        asset_bindings=bindings,
        asset_root=tmp_path,
    )
    assert all(
        np.array_equal(result.buses[source_id], reversed_result.buses[source_id])
        for source_id in result.source_ids
    )
    assert result.placement_receipts == reversed_result.placement_receipts
    assert result.assembly_content_sha256 == reversed_result.assembly_content_sha256


def test_silent_declared_source_still_has_exact_length_hashed_bus(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "call.wav"
    _write_pcm16(source_path, [8192, 8192], sample_rate_hz=8)
    digest = sha256_file(source_path)
    clip = DryAudioClipSpec.from_values(
        frame_count=2,
        fps_numerator=2,
        sample_rate_hz=8,
    )
    result = assemble_dry_audio_buses(
        [
            {
                **_event(
                    event_id="dog_call",
                    source_id="dog_source",
                    start=2,
                    end=6,
                    asset_id="call",
                    digest=digest,
                ),
                "fade_samples": 1,
            }
        ],
        source_ids=("dog_source", "human_source"),
        clip=clip,
        asset_bindings={"call": source_path},
    )
    assert result.buses["human_source"].shape == (8,)
    assert np.count_nonzero(result.buses["human_source"]) == 0
    assert result.bus_float64_le_sha256["human_source"] == _float64_hash(np.zeros(8))


def test_hash_and_mono_contract_fail_closed(tmp_path: Path) -> None:
    mono = tmp_path / "mono.wav"
    stereo = tmp_path / "stereo.wav"
    _write_pcm16(mono, [1, 2, 3, 4])
    _write_pcm16(stereo, [1, 2, 3, 4], channel_count=2)

    with pytest.raises(AudioContractError, match="SHA-256 differs"):
        read_authenticated_mono_pcm_wav(mono, expected_sha256="0" * 64)
    with pytest.raises(AudioContractError, match="exactly one channel"):
        read_authenticated_mono_pcm_wav(stereo, expected_sha256=sha256_file(stereo))


def test_parser_rejects_alias_binding_boundary_and_identity_conflicts(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_pcm16(first, [1, 2, 3, 4], sample_rate_hz=8)
    _write_pcm16(second, [4, 3, 2, 1], sample_rate_hz=8)
    first_hash = sha256_file(first)
    second_hash = sha256_file(second)
    clip = DryAudioClipSpec.from_values(
        frame_count=4,
        fps_numerator=4,
        sample_rate_hz=8,
    )
    base = _event(
        event_id="event0",
        source_id="dog_source",
        start=0,
        end=4,
        asset_id="asset0",
        digest=first_hash,
    )

    conflicting_hash = {
        **base,
        "dry_asset_sha256": second_hash,
    }
    with pytest.raises(AudioContractError, match="conflicting aliases"):
        parse_dry_audio_events(
            [conflicting_hash],
            source_ids=("dog_source", "human_source"),
            clip=clip,
            asset_bindings={"asset0": first},
        )

    with pytest.raises(AudioContractError, match="path conflicts"):
        parse_dry_audio_events(
            [{**base, "dry_asset_path": str(second)}],
            source_ids=("dog_source", "human_source"),
            clip=clip,
            asset_bindings={"asset0": first},
        )

    with pytest.raises(AudioContractError, match="hash conflicts"):
        parse_dry_audio_events(
            [base],
            source_ids=("dog_source", "human_source"),
            clip=clip,
            asset_bindings={"asset0": {"path": first, "sha256": second_hash}},
        )

    with pytest.raises(AudioContractError, match="start_frame conflicts"):
        parse_dry_audio_events(
            [{**base, "start_frame": 1}],
            source_ids=("dog_source", "human_source"),
            clip=clip,
            asset_bindings={"asset0": first},
        )

    with pytest.raises(AudioContractError, match="must satisfy"):
        parse_dry_audio_events(
            [{**base, "end_sample_exclusive": 9}],
            source_ids=("dog_source", "human_source"),
            clip=clip,
            asset_bindings={"asset0": first},
        )

    with pytest.raises(AudioContractError, match="duplicate event_id"):
        parse_dry_audio_events(
            [base, base],
            source_ids=("dog_source", "human_source"),
            clip=clip,
            asset_bindings={"asset0": first},
        )


def test_assembly_rejects_fake_event_hash_and_invalid_native_clip(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    _write_pcm16(source, [1, 2, 3, 4], sample_rate_hz=8)
    digest = sha256_file(source)
    clip = DryAudioClipSpec.from_values(
        frame_count=2,
        fps_numerator=2,
        sample_rate_hz=8,
    )
    base = _event(
        event_id="event0",
        source_id="dog_source",
        start=0,
        end=4,
        asset_id="asset0",
        digest=digest,
    )

    with pytest.raises(AudioContractError, match="SHA-256 differs"):
        assemble_dry_audio_buses(
            [{**base, "dry_audio_asset_sha256": "f" * 64}],
            source_ids=("dog_source", "human_source"),
            clip=clip,
            asset_bindings={"asset0": source},
        )

    with pytest.raises(AudioContractError, match="dry clip escapes"):
        assemble_dry_audio_buses(
            [{**base, "dry_clip_end_sample_exclusive": 9}],
            source_ids=("dog_source", "human_source"),
            clip=clip,
            asset_bindings={"asset0": source},
        )


def test_semantic_assembly_uses_declared_audio_shape_without_file_digest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "semantic.wav"
    _write_pcm16(source, [100, 200, 300, 400], sample_rate_hz=8)
    clip = DryAudioClipSpec.from_values(
        frame_count=2,
        fps_numerator=2,
        sample_rate_hz=8,
    )
    bindings = {
        "spoken_content": {
            "content_id": "spoken_content",
            "path": str(source),
            "sample_rate_hz": 8,
            "channel_count": 1,
            "sample_count": 4,
        }
    }
    event = {
        "event_id": "event0",
        "source_id": "source1",
        "start_sample": 2,
        "end_sample_exclusive": 6,
        "content_id": "spoken_content",
        "dry_clip_start_sample": 0,
        "dry_clip_end_sample_exclusive": 4,
        "linear_gain": 1.0,
        "fade_samples": 0,
    }

    first = assemble_semantic_dry_audio_buses(
        [event],
        source_ids=("source1", "source2"),
        clip=clip,
        content_bindings=bindings,
    )
    assert not np.any(first.buses["source1"][:2])
    assert np.any(first.buses["source1"][2:6])
    assert not np.any(first.buses["source1"][6:])
    assert not np.any(first.buses["source2"])
    assert first.metadata()["schema"] == (
        "avengine_m5_1_semantic_dry_audio_assembly_v1"
    )
    metadata = first.metadata()
    declared_content_id = metadata.pop("assembly_content_sha256")
    assert canonical_json_sha256(metadata) == declared_content_id
    forbidden = {
        "sha256",
        "file_sha256",
        "input_sha256",
        "byte_size",
        "audio_byte_size",
        "bus_float64_le_sha256",
    }
    assert not (_keys(first.metadata()) & forbidden)

    _write_pcm16(source, [-100, -200, -300, -400], sample_rate_hz=8)
    second = assemble_semantic_dry_audio_buses(
        [event],
        source_ids=("source1", "source2"),
        clip=clip,
        content_bindings=bindings,
    )
    assert np.array_equal(second.buses["source1"], -first.buses["source1"])


def test_semantic_assembly_fails_closed_on_declared_metadata_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "semantic.wav"
    _write_pcm16(source, [1, 2, 3, 4], sample_rate_hz=8)
    clip = DryAudioClipSpec.from_values(
        frame_count=2,
        fps_numerator=2,
        sample_rate_hz=8,
    )
    event = {
        "event_id": "event0",
        "source_id": "source1",
        "start_sample": 0,
        "end_sample_exclusive": 4,
        "content_id": "spoken_content",
    }
    binding = {
        "content_id": "spoken_content",
        "path": str(source),
        "sample_rate_hz": 8,
        "channel_count": 1,
        "sample_count": 5,
    }
    with pytest.raises(AudioContractError, match="metadata differs"):
        assemble_semantic_dry_audio_buses(
            [event],
            source_ids=("source1", "source2"),
            clip=clip,
            content_bindings={"spoken_content": binding},
        )
