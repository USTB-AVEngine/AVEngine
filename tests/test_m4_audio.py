from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from avengine.m4.audio import (
    AudioContractError,
    canonical_source_ids,
    generate_sine_wave,
    linear_convolve,
    read_float32_wav,
    render_stems_and_mix,
    sum_stems_canonical,
    write_float32_wav,
)


def test_exact_length_sine_has_no_fade_or_normalization() -> None:
    first = generate_sine_wave(8, 8, 1.0, amplitude=2.0)
    second = generate_sine_wave(8, 8, 1.0, amplitude=2.0)

    assert first.dtype == np.float64
    assert np.array_equal(first, second)
    assert first.shape == (8,)
    assert first[0] == 0.0
    assert first[2] == pytest.approx(2.0)
    assert np.max(np.abs(first)) == pytest.approx(2.0)

    with pytest.raises(AudioContractError, match="Nyquist"):
        generate_sine_wave(8, 8, 4.0)


def test_source_ids_are_portable_unique_and_byte_lexical() -> None:
    assert canonical_source_ids(["source_b", "source.A", "source-a"]) == (
        "source-a",
        "source.A",
        "source_b",
    )
    with pytest.raises(AudioContractError, match="unique"):
        canonical_source_ids(["dog", "dog"])
    with pytest.raises(AudioContractError, match="must match"):
        canonical_source_ids(["猫"])


def test_linear_convolution_is_full_float64_and_gain_preserving() -> None:
    result = linear_convolve(
        np.asarray([1.0, 2.0]),
        np.asarray([[1.0, 0.5], [0.0, 1.0]]),
    )

    assert result.dtype == np.float64
    assert result.shape == (2, 3)
    assert np.array_equal(
        result,
        np.asarray([[1.0, 2.5, 1.0], [0.0, 1.0, 2.0]], dtype=np.float64),
    )
    assert np.max(np.abs(result)) == 2.5


def test_stems_and_mix_use_canonical_ids_and_zero_pad_full_tails() -> None:
    dry_ab = {
        "source_b": np.asarray([1.0, -1.0, 0.5]),
        "source_a": np.asarray([0.5, 0.25]),
    }
    rir_ab = {
        "source_b": np.asarray([[1.0], [2.0]]),
        "source_a": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
    }
    stems_first, mixture_first = render_stems_and_mix(
        dry_ab,
        rir_ab,
        linear_gain_by_source={"source_b": 0.5, "source_a": 2.0},
    )
    stems_second, mixture_second = render_stems_and_mix(
        dict(reversed(tuple(dry_ab.items()))),
        dict(reversed(tuple(rir_ab.items()))),
        linear_gain_by_source={"source_a": 2.0, "source_b": 0.5},
    )

    assert tuple(stems_first) == ("source_a", "source_b")
    assert tuple(stems_second) == tuple(stems_first)
    assert np.array_equal(mixture_first, mixture_second)
    assert mixture_first.shape == (2, 3)
    expected = np.zeros_like(mixture_first)
    for source_id in ("source_a", "source_b"):
        stem = stems_first[source_id]
        expected[:, : stem.shape[1]] += stem
    assert np.array_equal(mixture_first, expected)


def test_canonical_sum_rejects_shape_and_sample_tampering() -> None:
    with pytest.raises(AudioContractError, match="channel count"):
        sum_stems_canonical(
            {"a": np.ones((4, 10)), "b": np.ones((2, 10))}
        )
    with pytest.raises(AudioContractError, match="finite"):
        linear_convolve(np.asarray([1.0, np.nan]), np.ones((4, 2)))
    with pytest.raises(AudioContractError, match="dimensions"):
        linear_convolve(np.ones((1, 2)), np.ones((4, 2)))


def test_float32_wav_roundtrip_is_deterministic_and_does_not_clip(tmp_path) -> None:
    samples = np.asarray(
        [
            [0.0, 1.5, -2.0, 0.25],
            [0.5, -0.5, 0.75, -0.75],
            [0.125, 0.25, 0.5, 1.0],
            [-1.25, 0.0, 1.25, 2.5],
        ],
        dtype=np.float64,
    )
    metadata = {
        "spatial_format": {
            "format_id": "rlr_foa_acn_n3d_world_v1",
            "channel_order": ["W", "Y", "Z", "X"],
        }
    }
    first = write_float32_wav(
        tmp_path / "first" / "authority.wav",
        samples,
        16_000,
        metadata=metadata,
    )
    second = write_float32_wav(
        tmp_path / "second" / "authority.wav",
        samples,
        16_000,
        metadata=metadata,
    )

    assert first.audio_path.read_bytes() == second.audio_path.read_bytes()
    assert first.sidecar_path.read_bytes() == second.sidecar_path.read_bytes()
    payload = first.audio_path.read_bytes()
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"
    assert struct.unpack_from("<H", payload, 20)[0] == 3

    decoded = read_float32_wav(first.audio_path)
    assert decoded.sample_rate_hz == 16_000
    assert decoded.samples.shape == samples.shape
    assert decoded.samples.dtype == np.dtype("float32")
    assert np.array_equal(decoded.samples, samples.astype(np.float32))
    assert float(np.max(np.abs(decoded.samples))) == 2.5
    assert decoded.sidecar is not None
    assert decoded.sidecar["metadata"] == metadata

    with pytest.raises(AudioContractError, match="overwrite"):
        write_float32_wav(first.audio_path, samples, 16_000, metadata=metadata)


def test_float32_wav_reader_detects_audio_sidecar_and_header_tampering(tmp_path) -> None:
    samples = np.arange(24, dtype=np.float64).reshape(4, 6) / 10.0

    audio_tamper = write_float32_wav(tmp_path / "audio_tamper.wav", samples, 16_000)
    payload = bytearray(audio_tamper.audio_path.read_bytes())
    payload[-1] ^= 1
    audio_tamper.audio_path.write_bytes(payload)
    with pytest.raises(AudioContractError, match="sidecar does not match"):
        read_float32_wav(audio_tamper.audio_path)

    sidecar_tamper = write_float32_wav(
        tmp_path / "sidecar_tamper.wav", samples, 16_000
    )
    sidecar = json.loads(sidecar_tamper.sidecar_path.read_text(encoding="utf-8"))
    sidecar["channel_count"] = 2
    sidecar_tamper.sidecar_path.write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    with pytest.raises(AudioContractError, match="content hash"):
        read_float32_wav(sidecar_tamper.audio_path)

    header_tamper = write_float32_wav(tmp_path / "header_tamper.wav", samples, 16_000)
    payload = bytearray(header_tamper.audio_path.read_bytes())
    struct.pack_into("<H", payload, 22, 3)
    header_tamper.audio_path.write_bytes(payload)
    with pytest.raises(AudioContractError, match="alignment"):
        read_float32_wav(header_tamper.audio_path, verify_sidecar=False)


def test_float32_wav_rejects_ambiguous_shape_and_invalid_metadata(tmp_path) -> None:
    with pytest.raises(AudioContractError, match="channel_axis"):
        write_float32_wav(
            tmp_path / "axis.wav",
            np.ones((4, 8)),
            16_000,
            channel_axis=2,
        )
    with pytest.raises(AudioContractError, match="real numeric"):
        write_float32_wav(
            tmp_path / "complex.wav",
            np.ones((2, 8), dtype=np.complex128),
            16_000,
        )
    with pytest.raises(AudioContractError, match="finite JSON"):
        write_float32_wav(
            tmp_path / "metadata.wav",
            np.ones((2, 8)),
            16_000,
            metadata={"bad": float("nan")},
        )
