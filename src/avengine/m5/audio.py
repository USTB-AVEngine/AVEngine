"""Exact-length, gain-preserving dynamic audio assembly for M5.

The authoritative path is deliberately small: mono dry events are placed on
named source buses, adjacent RIR keyframes form a raised-cosine partition of
unity, and full linear convolutions are overlap-added before an explicit
``[0, 80000)`` episode crop.  Nothing in this module resamples, normalizes,
limits, or infers a source from a mixture.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import wave
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.m4.audio import AudioContractError, canonical_source_ids, sum_stems_canonical


M5_AUDIO_SAMPLE_RATE_HZ = 16_000
M5_AUDIO_SAMPLE_COUNT = 80_000


@dataclass(frozen=True)
class DynamicStemResult:
    """One full-tail dynamic convolution and its exact episode crop."""

    full_tail: np.ndarray
    episode: np.ndarray
    keyframe_samples: tuple[int, ...]
    maximum_partition_error: float
    algorithm: str = "raised_cosine_source_time_partition_v1"


def read_pcm16_mono_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read one explicit PCM16 mono dry asset as owned float64 samples."""

    source = Path(path).resolve()
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            compression = handle.getcomptype()
            frame_count = handle.getnframes()
            payload = handle.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise AudioContractError(f"cannot read dry PCM asset {source}: {exc}") from exc
    if channels != 1 or sample_width != 2 or compression != "NONE":
        raise AudioContractError("M5 dry assets must be uncompressed PCM16 mono WAVE")
    if sample_rate != M5_AUDIO_SAMPLE_RATE_HZ:
        raise AudioContractError(
            f"M5 dry asset rate must be {M5_AUDIO_SAMPLE_RATE_HZ} Hz"
        )
    samples = np.frombuffer(payload, dtype="<i2")
    if samples.size != frame_count:
        raise AudioContractError("dry PCM payload length differs from WAVE header")
    result = np.ascontiguousarray(samples.astype(np.float64) / 32768.0)
    if not np.all(np.isfinite(result)):
        raise AudioContractError("dry PCM asset contains non-finite samples")
    return result, sample_rate


def extract_faded_clip(
    samples: Any,
    *,
    start_sample: int,
    end_sample: int,
    fade_samples: int = 80,
) -> np.ndarray:
    """Extract one immutable half-open clip with an explicit linear edge fade."""

    source = np.asarray(samples)
    if source.ndim != 1 or source.dtype.kind not in "iuf":
        raise AudioContractError("clip source must be a one-dimensional real array")
    source = np.ascontiguousarray(source, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise AudioContractError("clip source must be finite")
    if not (0 <= start_sample < end_sample <= source.size):
        raise AudioContractError("clip interval is outside the dry asset")
    if isinstance(fade_samples, bool) or not isinstance(fade_samples, int):
        raise AudioContractError("fade_samples must be an integer")
    length = end_sample - start_sample
    if fade_samples < 0 or fade_samples * 2 > length:
        raise AudioContractError("fade_samples must fit twice inside the clip")
    result = source[start_sample:end_sample].copy()
    if fade_samples:
        # The first and last retained samples are exactly zero.  The same
        # envelope is applied to both dry assets before any source routing.
        ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float64)
        result[:fade_samples] *= ramp
        result[-fade_samples:] *= ramp[::-1]
    return result


def place_simultaneous_events(
    clips_by_asset_id: Mapping[str, Any],
    asset_id_by_source: Mapping[str, str],
    *,
    start_samples: Sequence[int],
    output_sample_count: int = M5_AUDIO_SAMPLE_COUNT,
    linear_gain: float = 0.18,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Place the same event windows on every named source bus.

    Each source may route a different dry asset, but every source receives an
    event at every declared start.  This is the M5 simultaneous-vocalization
    invariant used by both counterfactual episodes.
    """

    if not isinstance(clips_by_asset_id, Mapping) or not clips_by_asset_id:
        raise AudioContractError("clips_by_asset_id must be a non-empty mapping")
    if not isinstance(asset_id_by_source, Mapping):
        raise AudioContractError("asset_id_by_source must be a mapping")
    source_ids = canonical_source_ids(asset_id_by_source.keys())
    if isinstance(output_sample_count, bool) or not isinstance(output_sample_count, int):
        raise AudioContractError("output_sample_count must be an integer")
    if output_sample_count <= 0:
        raise AudioContractError("output_sample_count must be positive")
    if isinstance(linear_gain, bool) or not isinstance(linear_gain, (int, float)):
        raise AudioContractError("linear_gain must be numeric")
    gain = float(linear_gain)
    if not math.isfinite(gain) or gain < 0.0:
        raise AudioContractError("linear_gain must be finite and non-negative")

    clips: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for asset_id, value in clips_by_asset_id.items():
        if not isinstance(asset_id, str) or not asset_id:
            raise AudioContractError("dry asset IDs must be non-empty strings")
        clip = np.asarray(value)
        if clip.ndim != 1 or clip.dtype.kind not in "iuf":
            raise AudioContractError(f"dry clip {asset_id!r} must be one-dimensional")
        clip = np.ascontiguousarray(clip, dtype=np.float64)
        if clip.size < 1 or not np.all(np.isfinite(clip)):
            raise AudioContractError(f"dry clip {asset_id!r} must be finite and non-empty")
        clips[asset_id] = clip
        lengths.add(int(clip.size))
    if len(lengths) != 1:
        raise AudioContractError("simultaneous dry clips must have exactly equal lengths")
    clip_length = next(iter(lengths))

    starts = tuple(int(value) for value in start_samples)
    if not starts or any(isinstance(value, bool) for value in start_samples):
        raise AudioContractError("start_samples must contain integer event starts")
    if tuple(sorted(set(starts))) != starts:
        raise AudioContractError("event starts must be strictly increasing")
    if starts[0] < 0 or starts[-1] + clip_length > output_sample_count:
        raise AudioContractError("simultaneous event window escapes the episode")

    buses = {
        source_id: np.zeros(output_sample_count, dtype=np.float64)
        for source_id in source_ids
    }
    events: list[dict[str, Any]] = []
    for event_index, start in enumerate(starts):
        for source_id in source_ids:
            asset_id = asset_id_by_source[source_id]
            if asset_id not in clips:
                raise AudioContractError(
                    f"source {source_id!r} routes undeclared dry asset {asset_id!r}"
                )
            stop = start + clip_length
            buses[source_id][start:stop] += clips[asset_id] * gain
            events.append(
                {
                    "event_id": f"event{event_index}_{source_id}",
                    "source_id": source_id,
                    "dry_asset_id": asset_id,
                    "start_sample": start,
                    "end_sample": stop,
                    "linear_gain": gain,
                    "simultaneous_group_id": f"simultaneous{event_index}",
                }
            )
    if any(not np.any(bus) for bus in buses.values()):
        raise AudioContractError("every M5 source bus must contain audible dry samples")
    return buses, events


def raised_cosine_partition(
    keyframe_samples: Sequence[int],
    sample_count: int,
) -> np.ndarray:
    """Return ``[keyframes, samples]`` weights that sum to one at every sample."""

    keys = np.asarray(keyframe_samples)
    if keys.ndim != 1 or keys.dtype.kind not in "iu" or keys.size < 1:
        raise AudioContractError("keyframe_samples must be a non-empty integer vector")
    keys = keys.astype(np.int64, copy=False)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise AudioContractError("sample_count must be a positive integer")
    if keys[0] != 0 or np.any(np.diff(keys) <= 0) or keys[-1] >= sample_count:
        raise AudioContractError(
            "keyframe samples must start at zero, increase, and lie inside the episode"
        )
    weights = np.zeros((keys.size, sample_count), dtype=np.float64)
    if keys.size == 1:
        weights[0] = 1.0
        return weights
    weights[0, : keys[0] + 1] = 1.0
    for index in range(keys.size - 1):
        left = int(keys[index])
        right = int(keys[index + 1])
        positions = np.arange(left, right + 1, dtype=np.float64)
        u = (positions - left) / float(right - left)
        left_weight = 0.5 * (1.0 + np.cos(np.pi * u))
        weights[index, left : right + 1] = left_weight
        weights[index + 1, left : right + 1] = 1.0 - left_weight
    weights[-1, int(keys[-1]) :] = 1.0
    error = float(np.max(np.abs(np.sum(weights, axis=0) - 1.0)))
    if error > 2.0e-15:
        raise AudioContractError(f"RIR partition does not sum to one: {error:.9g}")
    return weights


def _fft_convolve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output_count = left.size + right.size - 1
    fft_count = 1 << (output_count - 1).bit_length()
    spectrum = np.fft.rfft(left, fft_count) * np.fft.rfft(right, fft_count)
    result = np.fft.irfft(spectrum, fft_count)[:output_count]
    return np.ascontiguousarray(result, dtype=np.float64)


def time_varying_convolve(
    dry_samples: Any,
    rir_keyframes: Any,
    keyframe_samples: Sequence[int],
    *,
    rir_lengths: Sequence[int] | None = None,
    output_sample_count: int = M5_AUDIO_SAMPLE_COUNT,
    partition_weights: Any | None = None,
) -> DynamicStemResult:
    """Render one named source through a deterministic sequence of RIRs."""

    dry = np.asarray(dry_samples)
    rirs = np.asarray(rir_keyframes)
    if dry.ndim != 1 or dry.dtype.kind not in "iuf":
        raise AudioContractError("dynamic dry samples must be a real one-dimensional array")
    if rirs.ndim != 3 or rirs.dtype.kind not in "iuf":
        raise AudioContractError("rir_keyframes must have shape [keyframe, channel, sample]")
    dry = np.ascontiguousarray(dry, dtype=np.float64)
    rirs = np.ascontiguousarray(rirs, dtype=np.float64)
    if dry.size != output_sample_count:
        raise AudioContractError("dynamic dry bus must already have exact episode length")
    if not np.all(np.isfinite(dry)) or not np.all(np.isfinite(rirs)):
        raise AudioContractError("dynamic dry/RIR samples must be finite")
    keys = tuple(int(value) for value in keyframe_samples)
    if len(keys) != rirs.shape[0]:
        raise AudioContractError("one RIR is required for every keyframe sample")
    if rir_lengths is None:
        lengths = np.full(rirs.shape[0], rirs.shape[2], dtype=np.int64)
    else:
        lengths = np.asarray(rir_lengths)
        if lengths.shape != (rirs.shape[0],) or lengths.dtype.kind not in "iu":
            raise AudioContractError("rir_lengths must be an integer per-keyframe vector")
        lengths = lengths.astype(np.int64, copy=False)
        if np.any(lengths < 1) or np.any(lengths > rirs.shape[2]):
            raise AudioContractError("rir_lengths lies outside the padded RIR extent")
    if partition_weights is None:
        weights = raised_cosine_partition(keys, output_sample_count)
    else:
        weights = np.asarray(partition_weights)
        if (
            weights.shape != (len(keys), output_sample_count)
            or weights.dtype.kind not in "f"
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
        ):
            raise AudioContractError(
                "partition_weights must be finite [keyframe, sample] float weights"
            )
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        error = float(np.max(np.abs(weights.sum(axis=0) - 1.0)))
        if error > 2.0e-15:
            raise AudioContractError(
                f"partition_weights do not sum to one: {error:.9g}"
            )
    maximum_ir = int(np.max(lengths))
    full = np.zeros(
        (rirs.shape[1], output_sample_count + maximum_ir - 1), dtype=np.float64
    )
    for keyframe_index in range(rirs.shape[0]):
        weighted = dry * weights[keyframe_index]
        support = np.flatnonzero(weighted != 0.0)
        if not support.size:
            continue
        first = int(support[0])
        last = int(support[-1]) + 1
        segment = weighted[first:last]
        ir_length = int(lengths[keyframe_index])
        for channel_index in range(rirs.shape[1]):
            convolution = _fft_convolve(
                segment, rirs[keyframe_index, channel_index, :ir_length]
            )
            full[channel_index, first : first + convolution.size] += convolution
    if not np.all(np.isfinite(full)):
        raise AudioContractError("dynamic convolution overflowed float64")
    partition_error = float(np.max(np.abs(weights.sum(axis=0) - 1.0)))
    return DynamicStemResult(
        full_tail=np.ascontiguousarray(full),
        episode=np.ascontiguousarray(full[:, :output_sample_count]),
        keyframe_samples=keys,
        maximum_partition_error=partition_error,
    )


def render_dynamic_stems_and_mix(
    dry_by_source: Mapping[str, Any],
    rir_samples: Any,
    rir_lengths: Any,
    *,
    source_ids: Sequence[str],
    keyframe_samples: Sequence[int],
    output_sample_count: int = M5_AUDIO_SAMPLE_COUNT,
    partition_weights: Any | None = None,
) -> tuple[dict[str, DynamicStemResult], np.ndarray]:
    """Render ``[K,S,C,L]`` RIRs into named episode stems and one mixture."""

    canonical = canonical_source_ids(source_ids)
    if tuple(source_ids) != canonical:
        raise AudioContractError("dynamic source_ids must already be canonical")
    if set(dry_by_source) != set(canonical):
        raise AudioContractError("dry buses and dynamic source IDs differ")
    rirs = np.asarray(rir_samples)
    lengths = np.asarray(rir_lengths)
    if rirs.ndim != 4 or rirs.shape[1] != len(canonical):
        raise AudioContractError("dynamic RIR samples must have shape [K,S,C,L]")
    if lengths.shape != rirs.shape[:2]:
        raise AudioContractError("dynamic RIR lengths must have shape [K,S]")
    stems: dict[str, DynamicStemResult] = {}
    for source_index, source_id in enumerate(canonical):
        stems[source_id] = time_varying_convolve(
            dry_by_source[source_id],
            rirs[:, source_index],
            keyframe_samples,
            rir_lengths=lengths[:, source_index],
            output_sample_count=output_sample_count,
            partition_weights=partition_weights,
        )
    mixture = sum_stems_canonical(
        {source_id: stems[source_id].episode for source_id in canonical}
    )
    if mixture.shape[1] != output_sample_count:
        raise AudioContractError("dynamic mixture lost exact episode length")
    return stems, mixture


__all__ = [
    "DynamicStemResult",
    "M5_AUDIO_SAMPLE_COUNT",
    "M5_AUDIO_SAMPLE_RATE_HZ",
    "extract_faded_clip",
    "place_simultaneous_events",
    "raised_cosine_partition",
    "read_pcm16_mono_wav",
    "render_dynamic_stems_and_mix",
    "time_varying_convolve",
]
