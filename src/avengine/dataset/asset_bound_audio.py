"""Prepare real dry recordings and assemble them with asset-bound RIRs.

The reusable route bank contains actor-root motion, while the asset-bound RIR
cache contains concrete emitter positions.  This module keeps dry sound
selection separate: a human, dog, cat, or future small-animal recording can
be exchanged only after the matching asset-bound cache has been selected.

Input WAVE files are deliberately limited to uncompressed PCM16 with one or
two channels.  Stereo inputs require an explicit equal-weight downmix policy;
there is no implicit channel selection, loudness normalization, limiter, or
hidden RLR work in this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import wave

import numpy as np

from avengine.spatial_audio.audio import AudioContractError, canonical_source_ids
from avengine.timeline.audio import (
    M5_AUDIO_SAMPLE_COUNT,
    M5_AUDIO_SAMPLE_RATE_HZ,
    DynamicStemResult,
    render_dynamic_stems_and_mix,
)
from avengine.capture.dry_audio import deterministic_resample_mono


ASSET_BOUND_AUDIO_SCHEMA = "avengine_m7_asset_bound_binaural_audio_v1"


class AssetBoundAudioError(AudioContractError):
    """An external dry recording cannot be used in a bound-RIR episode."""


@dataclass(frozen=True)
class PreparedDryAudio:
    """One exact five-second mono bus and the input transformation receipt."""

    samples: np.ndarray
    record: Mapping[str, Any]


def bind_endpoint_buses_to_source_slots(
    endpoint_buses: Mapping[str, Any],
    *,
    endpoint_to_source_slot: Mapping[str, str],
    source_slots: Sequence[str],
) -> dict[str, np.ndarray]:
    """Bind validated M6 endpoint buses to the concrete M7 RIR source slots.

    AudioProgram endpoint identity and asset-bound RIR slot identity are
    intentionally different contracts.  The bridge between them must be an
    explicit bijection; positional or lexical inference is not permitted.
    """

    canonical_slots = canonical_source_ids(source_slots)
    if tuple(source_slots) != canonical_slots:
        raise AssetBoundAudioError("source slots must already be canonical")
    if set(endpoint_to_source_slot) != set(endpoint_buses):
        raise AssetBoundAudioError(
            "endpoint-to-slot mapping keys must exactly match AudioProgram endpoints"
        )
    mapped_slots = tuple(endpoint_to_source_slot.values())
    if (
        len(mapped_slots) != len(set(mapped_slots))
        or set(mapped_slots) != set(canonical_slots)
    ):
        raise AssetBoundAudioError(
            "endpoint-to-slot mapping must be a bijection over the RIR source slots"
        )
    result: dict[str, np.ndarray] = {}
    for endpoint_id, slot in endpoint_to_source_slot.items():
        bus = np.asarray(endpoint_buses[endpoint_id])
        if bus.shape != (M5_AUDIO_SAMPLE_COUNT,) or bus.dtype.kind not in "iuf":
            raise AssetBoundAudioError(
                f"AudioProgram endpoint {endpoint_id!r} has an invalid dry bus"
            )
        bus = np.ascontiguousarray(bus, dtype=np.float64)
        if not np.all(np.isfinite(bus)):
            raise AssetBoundAudioError(
                f"AudioProgram endpoint {endpoint_id!r} dry bus is not finite"
            )
        result[slot] = bus
    return {slot: result[slot] for slot in canonical_slots}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonnegative_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AssetBoundAudioError(f"{owner} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise AssetBoundAudioError(f"{owner} must be a non-negative integer")
    return result


def _finite_gain(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise AssetBoundAudioError("linear_gain must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AssetBoundAudioError("linear_gain must be finite and non-negative")
    return result


def _decode_pcm16_wave(
    path: str | Path,
    *,
    channel_policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode an explicit mono input or an equal-weight stereo downmix."""

    source = Path(path).resolve()
    if not source.is_file():
        raise AssetBoundAudioError(f"dry audio is not a regular file: {source}")
    if channel_policy not in {"require_mono", "equal_weight_downmix"}:
        raise AssetBoundAudioError(
            "channel_policy must be require_mono or equal_weight_downmix"
        )
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate_hz = handle.getframerate()
            compression = handle.getcomptype()
            frame_count = handle.getnframes()
            payload = handle.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise AssetBoundAudioError(f"cannot read PCM WAVE {source}: {exc}") from exc
    if sample_width != 2 or compression != "NONE" or channels not in {1, 2}:
        raise AssetBoundAudioError(
            "dry audio must be uncompressed PCM16 WAVE with one or two channels"
        )
    if channels == 2 and channel_policy != "equal_weight_downmix":
        raise AssetBoundAudioError("stereo dry audio requires equal_weight_downmix")
    samples = np.frombuffer(payload, dtype="<i2")
    if samples.size != frame_count * channels:
        raise AssetBoundAudioError("PCM WAVE payload differs from its header")
    samples = samples.reshape(frame_count, channels).astype(np.float64) / 32768.0
    mono = samples[:, 0] if channels == 1 else np.mean(samples, axis=1)
    mono = np.ascontiguousarray(mono, dtype=np.float64)
    if not np.all(np.isfinite(mono)):
        raise AssetBoundAudioError("decoded dry audio is not finite")
    return mono, {
        "path": str(source),
        "sha256": _file_sha256(source),
        "byte_size": source.stat().st_size,
        "container": "WAVE",
        "encoding": "PCM16",
        "source_channel_count": channels,
        "source_sample_rate_hz": sample_rate_hz,
        "source_sample_count": frame_count,
        "channel_policy": channel_policy,
    }


def prepare_dry_audio(
    path: str | Path,
    *,
    channel_policy: str,
    source_start_sample: int = 0,
    linear_gain: float = 1.0,
    fade_samples: int = 80,
    output_sample_rate_hz: int = M5_AUDIO_SAMPLE_RATE_HZ,
    output_sample_count: int = M5_AUDIO_SAMPLE_COUNT,
) -> PreparedDryAudio:
    """Return one exact-length source bus with no normalization or looping."""

    source, source_record = _decode_pcm16_wave(path, channel_policy=channel_policy)
    start = _nonnegative_integer(source_start_sample, owner="source_start_sample")
    fade = _nonnegative_integer(fade_samples, owner="fade_samples")
    if start >= source.size:
        raise AssetBoundAudioError("source_start_sample lies outside the recording")
    if (
        isinstance(output_sample_rate_hz, bool)
        or not isinstance(output_sample_rate_hz, int)
        or output_sample_rate_hz < 1
        or isinstance(output_sample_count, bool)
        or not isinstance(output_sample_count, int)
        or output_sample_count < 1
    ):
        raise AssetBoundAudioError("output audio duration is invalid")
    gain = _finite_gain(linear_gain)
    selected = source[start:]
    resampled = deterministic_resample_mono(
        selected,
        source_sample_rate_hz=int(source_record["source_sample_rate_hz"]),
        target_sample_rate_hz=output_sample_rate_hz,
    )
    copied = min(int(resampled.size), output_sample_count)
    if fade * 2 > copied:
        raise AssetBoundAudioError("fade_samples does not fit the retained audio")
    result = np.zeros(output_sample_count, dtype=np.float64)
    result[:copied] = resampled[:copied] * gain
    if fade:
        envelope = np.linspace(0.0, 1.0, fade, endpoint=True, dtype=np.float64)
        result[:fade] *= envelope
        result[copied - fade : copied] *= envelope[::-1]
    if not np.any(result):
        raise AssetBoundAudioError("prepared dry audio is silent")
    if not np.all(np.isfinite(result)):
        raise AssetBoundAudioError("prepared dry audio is not finite")
    record = {
        "schema": ASSET_BOUND_AUDIO_SCHEMA,
        "status": "pass",
        "input": source_record,
        "channel_policy": channel_policy,
        "source_start_sample": start,
        "source_selected_sample_count": int(selected.size),
        "target_sample_rate_hz": output_sample_rate_hz,
        "target_sample_count": output_sample_count,
        "resampling": "float64_linear_source_time_grid_v1",
        "copied_target_sample_count": copied,
        "zero_padded_target_sample_count": output_sample_count - copied,
        "linear_gain": gain,
        "fade_samples": fade,
        "normalization": False,
        "limiting": False,
        "looping": False,
        "bus_peak_absolute": float(np.max(np.abs(result))),
    }
    return PreparedDryAudio(samples=np.ascontiguousarray(result), record=record)


def render_asset_bound_binaural(
    dry_by_source: Mapping[str, Any],
    *,
    rir_samples: Any,
    rir_lengths: Any,
    source_ids: Sequence[str],
    keyframe_samples: Sequence[int],
    partition_weights: Any | None = None,
) -> tuple[dict[str, DynamicStemResult], np.ndarray]:
    """Render independent source buses through a completed binaural RIR grid."""

    canonical = canonical_source_ids(source_ids)
    if tuple(source_ids) != canonical:
        raise AssetBoundAudioError("source IDs must already be canonical")
    if set(dry_by_source) != set(canonical):
        raise AssetBoundAudioError("dry source IDs differ from the RIR source slots")
    prepared = {
        source_id: np.ascontiguousarray(dry_by_source[source_id], dtype=np.float64)
        for source_id in canonical
    }
    if any(value.shape != (M5_AUDIO_SAMPLE_COUNT,) for value in prepared.values()):
        raise AssetBoundAudioError("each dry bus must be exactly 80,000 samples")
    rirs = np.asarray(rir_samples)
    if rirs.ndim != 4 or rirs.shape[1] != len(canonical) or rirs.shape[2] != 2:
        raise AssetBoundAudioError(
            "asset-bound preview requires [keyframe, source, left-right, sample] RIRs"
        )
    stems, mixture = render_dynamic_stems_and_mix(
        prepared,
        rirs,
        rir_lengths,
        source_ids=canonical,
        keyframe_samples=keyframe_samples,
        partition_weights=partition_weights,
    )
    expected = sum(stems[source_id].episode for source_id in canonical)
    if not np.array_equal(mixture, expected):
        raise AssetBoundAudioError("mixture differs from the exact source-stem sum")
    if not np.all(np.isfinite(mixture)):
        raise AssetBoundAudioError("binaural mixture is not finite")
    return stems, np.ascontiguousarray(mixture, dtype=np.float64)


def float32_stems_and_exact_mix(
    stems: Mapping[str, DynamicStemResult], *, source_ids: Sequence[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Quantize stems once, then form the persisted mixture from those stems.

    The acoustic arithmetic is float64.  The WAVE delivery contract is
    float32, so the stored mixture must be formed from the same float32 stem
    values that are written independently; otherwise a reader sees a tiny
    rounding difference despite a float64 in-memory exact sum.
    """

    canonical = canonical_source_ids(source_ids)
    if tuple(source_ids) != canonical or set(stems) != set(canonical):
        raise AssetBoundAudioError("stem IDs differ from the canonical source order")
    stored: dict[str, np.ndarray] = {}
    for source_id in canonical:
        stem = np.asarray(stems[source_id].episode)
        if stem.shape != (2, M5_AUDIO_SAMPLE_COUNT) or not np.all(np.isfinite(stem)):
            raise AssetBoundAudioError("binaural stem has an invalid persisted shape")
        stored[source_id] = np.ascontiguousarray(stem, dtype=np.float32)
    mixture = np.zeros((2, M5_AUDIO_SAMPLE_COUNT), dtype=np.float32)
    for source_id in canonical:
        mixture += stored[source_id]
    if not np.array_equal(mixture, sum(stored[source_id] for source_id in canonical)):
        raise AssetBoundAudioError("float32 mixture differs from the stored stem sum")
    return stored, mixture


__all__ = [
    "ASSET_BOUND_AUDIO_SCHEMA",
    "AssetBoundAudioError",
    "PreparedDryAudio",
    "bind_endpoint_buses_to_source_slots",
    "float32_stems_and_exact_mix",
    "prepare_dry_audio",
    "render_asset_bound_binaural",
]
