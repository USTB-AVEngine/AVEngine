"""Deterministic, gain-preserving audio primitives for M4.

This module deliberately has no normalization, resampling, limiting, or
codec-facing convenience path.  A caller supplies mono dry samples, explicit
per-pair impulse responses, and (optionally) an explicit linear gain.  The
authoritative arithmetic stays in ``float64`` until a caller deliberately
writes the resulting samples as IEEE ``float32`` WAVE data.

Arrays are channel-major throughout the Python API.  WAVE interleaving is an
I/O detail and is never inferred from array dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)


FLOAT32_WAV_SIDECAR_SCHEMA = "avengine_float32_wav_sidecar_v1"
IEEE_FLOAT_WAVE_FORMAT_TAG = 3
_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1


class AudioContractError(ValueError):
    """Audio samples, identities, or serialized bytes violate M4 rules."""


@dataclass(frozen=True)
class Float32WavArtifact:
    """Authenticated record returned after writing one WAVE + sidecar pair."""

    audio_path: Path
    sidecar_path: Path
    audio_sha256: str
    sidecar_sha256: str
    sample_rate_hz: int
    frame_count: int
    channel_count: int


@dataclass(frozen=True)
class Float32Wav:
    """One strictly decoded IEEE-float WAVE file.

    ``samples`` is always a newly owned, channel-major ``float32`` array with
    shape ``[channel_count, frame_count]``.
    """

    path: Path
    samples: np.ndarray
    sample_rate_hz: int
    sidecar: Mapping[str, Any] | None

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[1])

    @property
    def channel_count(self) -> int:
        return int(self.samples.shape[0])


def canonical_source_ids(source_ids: Iterable[str]) -> tuple[str, ...]:
    """Validate and return the one portable, byte-lexical source order.

    Restricting IDs to ASCII makes Python ordering, UTF-8 byte ordering, file
    naming, and the native RLR registration order agree without locale or
    Unicode-normalization ambiguity.
    """

    if isinstance(source_ids, (str, bytes)):
        raise AudioContractError("source_ids must be an iterable of IDs, not text")
    values = tuple(source_ids)
    if not values:
        raise AudioContractError("at least one source_id is required")
    invalid = [value for value in values if not isinstance(value, str) or not _SOURCE_ID.fullmatch(value)]
    if invalid:
        raise AudioContractError(
            "source_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}: "
            f"{invalid!r}"
        )
    if len(set(values)) != len(values):
        raise AudioContractError("source_ids must be unique")
    # The accepted character set is ASCII, so Unicode-codepoint, UTF-8 byte,
    # and C-locale lexical order are identical.
    return tuple(sorted(values, key=lambda value: value.encode("ascii")))


def generate_sine_wave(
    sample_rate_hz: int,
    sample_count: int,
    frequency_hz: float,
    *,
    amplitude: float = 1.0,
    phase_radians: float = 0.0,
) -> np.ndarray:
    """Generate an exact-length, unfaded ``float64`` canary sine.

    This helper performs no resampling, envelope, normalization, or endpoint
    adjustment.  ``sample_count`` is authoritative and frequency must be
    strictly below Nyquist.
    """

    rate = _positive_sample_rate(sample_rate_hz)
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or int(sample_count) <= 0
    ):
        raise AudioContractError("sample_count must be a positive integer")
    numeric_values = {
        "frequency_hz": frequency_hz,
        "amplitude": amplitude,
        "phase_radians": phase_radians,
    }
    converted: dict[str, float] = {}
    for name, value in numeric_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise AudioContractError(f"{name} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise AudioContractError(f"{name} must be a finite number")
        converted[name] = number
    if not 0.0 < converted["frequency_hz"] < rate / 2.0:
        raise AudioContractError("frequency_hz must be strictly between 0 and Nyquist")
    if converted["amplitude"] < 0.0:
        raise AudioContractError("amplitude must be non-negative")
    indices = np.arange(int(sample_count), dtype=np.float64)
    result = converted["amplitude"] * np.sin(
        (2.0 * math.pi * converted["frequency_hz"] / rate) * indices
        + converted["phase_radians"]
    )
    return np.ascontiguousarray(result, dtype=np.float64)


def _float64_samples(
    value: Any,
    *,
    owner: str,
    dimensions: tuple[int, ...],
) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise AudioContractError(f"{owner} must contain real numeric samples")
    if source.ndim not in dimensions:
        expected = " or ".join(str(item) for item in dimensions)
        raise AudioContractError(f"{owner} must have {expected} dimensions")
    try:
        result = np.ascontiguousarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AudioContractError(f"{owner} cannot be represented as float64") from exc
    if any(size < 1 for size in result.shape):
        raise AudioContractError(f"{owner} must not contain an empty axis")
    if not np.all(np.isfinite(result)):
        raise AudioContractError(f"{owner} must contain only finite samples")
    return result


def _finite_gain(value: Any, *, source_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise AudioContractError(f"linear gain for {source_id!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AudioContractError(f"linear gain for {source_id!r} must be finite")
    return result


def linear_convolve(dry_samples: Any, impulse_response: Any) -> np.ndarray:
    """Convolve one mono dry signal with one channel-major IR in ``float64``.

    A one-dimensional IR is accepted as explicitly mono.  A two-dimensional
    IR must already be ``[channels, samples]``.  The returned full convolution
    has shape ``[channels, len(dry) + len(ir) - 1]``; no tail is cropped.
    """

    dry = _float64_samples(
        dry_samples,
        owner="dry_samples",
        dimensions=(1,),
    )
    ir = _float64_samples(
        impulse_response,
        owner="impulse_response",
        dimensions=(1, 2),
    )
    if ir.ndim == 1:
        ir = ir[np.newaxis, :]

    sample_count = dry.size + ir.shape[1] - 1
    result = np.empty((ir.shape[0], sample_count), dtype=np.float64)
    for channel_index in range(ir.shape[0]):
        # numpy.convolve is a deterministic direct linear convolution for
        # these one-dimensional float64 operands.  Avoiding an FFT here also
        # avoids backend- and plan-dependent round-off in the authority path.
        result[channel_index] = np.convolve(
            dry,
            ir[channel_index],
            mode="full",
        )
    if not np.all(np.isfinite(result)):
        raise AudioContractError("linear convolution overflowed float64")
    return result


def render_stems(
    dry_by_source: Mapping[str, Any],
    rir_by_source: Mapping[str, Any],
    *,
    linear_gain_by_source: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Render one independent wet stem per source in canonical ID order.

    The default gain is exactly ``1.0``.  Providing gains is an explicit
    operation; there is no peak normalization or mixture-dependent scaling.
    """

    if not isinstance(dry_by_source, Mapping) or not isinstance(rir_by_source, Mapping):
        raise AudioContractError("dry_by_source and rir_by_source must be mappings")
    source_ids = canonical_source_ids(dry_by_source.keys())
    if set(rir_by_source) != set(source_ids):
        raise AudioContractError(
            "dry_by_source and rir_by_source must have the exact same source IDs"
        )
    if linear_gain_by_source is not None:
        if not isinstance(linear_gain_by_source, Mapping):
            raise AudioContractError("linear_gain_by_source must be a mapping")
        if set(linear_gain_by_source) != set(source_ids):
            raise AudioContractError(
                "linear_gain_by_source must have the exact same source IDs"
            )

    stems: dict[str, np.ndarray] = {}
    for source_id in source_ids:
        gain = (
            1.0
            if linear_gain_by_source is None
            else _finite_gain(linear_gain_by_source[source_id], source_id=source_id)
        )
        stem = linear_convolve(
            dry_by_source[source_id],
            rir_by_source[source_id],
        )
        if gain != 1.0:
            stem = stem * gain
            if not np.all(np.isfinite(stem)):
                raise AudioContractError(
                    f"linear gain for {source_id!r} overflowed float64"
                )
        stems[source_id] = stem
    return stems


def sum_stems_canonical(stems_by_source: Mapping[str, Any]) -> np.ndarray:
    """Zero-pad and compensated-sum stems in canonical source-ID order.

    All stems must have the same channel count.  Different full-convolution
    lengths are padded with exact zeros at the end.  Vectorized Kahan
    compensation makes the declared order stable without silently changing
    source gains.
    """

    if not isinstance(stems_by_source, Mapping):
        raise AudioContractError("stems_by_source must be a mapping")
    source_ids = canonical_source_ids(stems_by_source.keys())
    stems: dict[str, np.ndarray] = {}
    channel_count: int | None = None
    maximum_samples = 0
    for source_id in source_ids:
        stem = _float64_samples(
            stems_by_source[source_id],
            owner=f"stem[{source_id!r}]",
            dimensions=(2,),
        )
        if channel_count is None:
            channel_count = int(stem.shape[0])
        elif stem.shape[0] != channel_count:
            raise AudioContractError("all stems must have the same channel count")
        maximum_samples = max(maximum_samples, int(stem.shape[1]))
        stems[source_id] = stem

    assert channel_count is not None
    total = np.zeros((channel_count, maximum_samples), dtype=np.float64)
    compensation = np.zeros_like(total)
    for source_id in source_ids:
        stem = stems[source_id]
        width = stem.shape[1]
        current = total[:, :width]
        correction = compensation[:, :width]
        adjusted = stem - correction
        updated = current + adjusted
        compensation[:, :width] = (updated - current) - adjusted
        total[:, :width] = updated
    if not np.all(np.isfinite(total)):
        raise AudioContractError("canonical stem sum overflowed float64")
    return total


def render_stems_and_mix(
    dry_by_source: Mapping[str, Any],
    rir_by_source: Mapping[str, Any],
    *,
    linear_gain_by_source: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return independent stems and their canonical, gain-preserving mixture."""

    stems = render_stems(
        dry_by_source,
        rir_by_source,
        linear_gain_by_source=linear_gain_by_source,
    )
    return stems, sum_stems_canonical(stems)


def _positive_sample_rate(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AudioContractError("sample_rate_hz must be a positive integer")
    result = int(value)
    if result <= 0 or result > _UINT32_MAX:
        raise AudioContractError("sample_rate_hz must be a positive uint32 integer")
    return result


def _channel_major(value: Any, *, channel_axis: int) -> np.ndarray:
    samples = _float64_samples(value, owner="samples", dimensions=(2,))
    if channel_axis in (0, -2):
        result = samples
    elif channel_axis in (1, -1):
        result = samples.T
    else:
        raise AudioContractError("channel_axis must explicitly identify axis 0 or 1")
    return np.ascontiguousarray(result, dtype=np.float64)


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    if len(chunk_id) != 4:
        raise AssertionError("RIFF chunk IDs contain four bytes")
    padding = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding


def _float32_wav_bytes(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
) -> bytes:
    channel_count, frame_count = samples.shape
    block_align = channel_count * 4
    byte_rate = sample_rate_hz * block_align
    if channel_count > _UINT16_MAX or block_align > _UINT16_MAX:
        raise AudioContractError("channel count exceeds canonical WAVE limits")
    if byte_rate > _UINT32_MAX or frame_count > _UINT32_MAX:
        raise AudioContractError("audio dimensions exceed canonical WAVE limits")

    converted = np.ascontiguousarray(samples.T, dtype="<f4")
    if not np.all(np.isfinite(converted)):
        raise AudioContractError("samples overflow float32 during WAVE serialization")
    data = converted.tobytes(order="C")
    fmt = struct.pack(
        "<HHIIHH",
        IEEE_FLOAT_WAVE_FORMAT_TAG,
        channel_count,
        sample_rate_hz,
        byte_rate,
        block_align,
        32,
    )
    fact = struct.pack("<I", frame_count)
    body = b"WAVE" + _chunk(b"fmt ", fmt) + _chunk(b"fact", fact) + _chunk(b"data", data)
    if len(body) > _UINT32_MAX:
        raise AudioContractError("WAVE RIFF payload exceeds uint32 size")
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _validated_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise AudioContractError("metadata must be a JSON object")
    value = dict(metadata)
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise AudioContractError("metadata must contain finite JSON values") from exc
    return value


def _sidecar_path(audio_path: Path, sidecar_path: str | Path | None) -> Path:
    result = (
        audio_path.with_suffix(audio_path.suffix + ".json")
        if sidecar_path is None
        else Path(sidecar_path).absolute()
    )
    if result.parent != audio_path.parent:
        raise AudioContractError("WAVE sidecar must be next to its audio file")
    if result == audio_path:
        raise AudioContractError("WAVE and sidecar paths must differ")
    return result


def write_float32_wav(
    path: str | Path,
    samples: Any,
    sample_rate_hz: int,
    *,
    channel_axis: int = 0,
    metadata: Mapping[str, Any] | None = None,
    sidecar_path: str | Path | None = None,
) -> Float32WavArtifact:
    """Write canonical IEEE-float WAVE bytes and an authenticated sidecar.

    Existing files are never overwritten.  Serialization only casts to
    ``float32`` and interleaves channels; it never rescales, clips, resamples,
    limits, or changes the sample count.
    """

    audio_path = Path(path).absolute()
    if audio_path.suffix.casefold() != ".wav":
        raise AudioContractError("float32 WAVE path must end in .wav")
    sidecar = _sidecar_path(audio_path, sidecar_path)
    if os.path.lexists(audio_path) or os.path.lexists(sidecar):
        raise AudioContractError("refusing to overwrite WAVE or sidecar output")
    rate = _positive_sample_rate(sample_rate_hz)
    channel_major = _channel_major(samples, channel_axis=channel_axis)
    metadata_value = _validated_metadata(metadata)
    payload = _float32_wav_bytes(channel_major, sample_rate_hz=rate)

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_created = False
    sidecar_created = False
    try:
        with audio_path.open("xb") as handle:
            audio_created = True
            handle.write(payload)
        audio_sha256 = sha256_file(audio_path)
        sidecar_value: dict[str, Any] = {
            "schema": FLOAT32_WAV_SIDECAR_SCHEMA,
            "audio_file": audio_path.name,
            "audio_sha256": audio_sha256,
            "audio_byte_size": len(payload),
            "container": "RIFF/WAVE",
            "format_tag": IEEE_FLOAT_WAVE_FORMAT_TAG,
            "sample_encoding": "IEEE_FLOAT",
            "bits_per_sample": 32,
            "endianness": "little",
            "sample_rate_hz": rate,
            "frame_count": int(channel_major.shape[1]),
            "channel_count": int(channel_major.shape[0]),
            "file_interleave": "frame_major",
            "api_array_layout": "channel_major",
            "metadata": metadata_value,
        }
        sidecar_value["sidecar_content_sha256"] = canonical_json_sha256(
            sidecar_value
        )
        sidecar_bytes = (
            json.dumps(
                sidecar_value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with sidecar.open("xb") as handle:
            sidecar_created = True
            handle.write(sidecar_bytes)
    except Exception:
        # Remove only entries this call successfully created.  In particular,
        # never unlink a sidecar that another process won in an O_EXCL race.
        for output, created in (
            (sidecar, sidecar_created),
            (audio_path, audio_created),
        ):
            if not created:
                continue
            try:
                output.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return Float32WavArtifact(
        audio_path=audio_path,
        sidecar_path=sidecar,
        audio_sha256=audio_sha256,
        sidecar_sha256=sha256_file(sidecar),
        sample_rate_hz=rate,
        frame_count=int(channel_major.shape[1]),
        channel_count=int(channel_major.shape[0]),
    )


def _parse_canonical_float32_wav(payload: bytes) -> tuple[np.ndarray, int]:
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise AudioContractError("audio file is not RIFF/WAVE")
    declared_size = struct.unpack_from("<I", payload, 4)[0]
    if declared_size + 8 != len(payload):
        raise AudioContractError("RIFF byte size does not match the file")

    chunks: list[tuple[bytes, bytes]] = []
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise AudioContractError("truncated RIFF chunk header")
        chunk_id = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        start = offset + 8
        end = start + chunk_size
        if end > len(payload):
            raise AudioContractError("truncated RIFF chunk payload")
        chunks.append((chunk_id, payload[start:end]))
        offset = end + (chunk_size % 2)
        if chunk_size % 2 and (offset > len(payload) or payload[end:offset] != b"\x00"):
            raise AudioContractError("invalid RIFF padding byte")
    if offset != len(payload):
        raise AudioContractError("RIFF chunks do not consume the file")
    if [item[0] for item in chunks] != [b"fmt ", b"fact", b"data"]:
        raise AudioContractError(
            "WAVE must contain exactly the canonical fmt, fact, data chunks"
        )

    fmt, fact, data = (item[1] for item in chunks)
    if len(fmt) != 16:
        raise AudioContractError("canonical IEEE-float fmt chunk must be 16 bytes")
    (
        format_tag,
        channel_count,
        sample_rate_hz,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack("<HHIIHH", fmt)
    if format_tag != IEEE_FLOAT_WAVE_FORMAT_TAG or bits_per_sample != 32:
        raise AudioContractError("WAVE samples must use 32-bit IEEE float format tag 3")
    if channel_count < 1 or sample_rate_hz < 1:
        raise AudioContractError("WAVE channel count and sample rate must be positive")
    if block_align != channel_count * 4 or byte_rate != sample_rate_hz * block_align:
        raise AudioContractError("WAVE byte-rate or block alignment is inconsistent")
    if len(data) == 0 or len(data) % block_align:
        raise AudioContractError("WAVE data size is not a positive whole frame count")
    frame_count = len(data) // block_align
    if len(fact) != 4 or struct.unpack("<I", fact)[0] != frame_count:
        raise AudioContractError("WAVE fact sample count does not match data")

    frame_major = np.frombuffer(data, dtype="<f4").reshape(frame_count, channel_count)
    samples = np.ascontiguousarray(frame_major.T)
    if not np.all(np.isfinite(samples)):
        raise AudioContractError("WAVE data contains non-finite samples")
    return samples, int(sample_rate_hz)


def _load_verified_sidecar(
    path: Path,
    *,
    audio_path: Path,
    audio_payload: bytes,
    samples: np.ndarray,
    sample_rate_hz: int,
) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AudioContractError(f"WAVE sidecar is missing or invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise AudioContractError("WAVE sidecar root must be an object")
    declared_content_hash = value.get("sidecar_content_sha256")
    content = {key: item for key, item in value.items() if key != "sidecar_content_sha256"}
    try:
        actual_content_hash = canonical_json_sha256(content)
    except (TypeError, ValueError) as exc:
        raise AudioContractError("WAVE sidecar contains invalid JSON values") from exc
    if declared_content_hash != actual_content_hash:
        raise AudioContractError("WAVE sidecar content hash mismatch")

    expected = {
        "schema": FLOAT32_WAV_SIDECAR_SCHEMA,
        "audio_file": audio_path.name,
        "audio_sha256": hashlib.sha256(audio_payload).hexdigest(),
        "audio_byte_size": len(audio_payload),
        "container": "RIFF/WAVE",
        "format_tag": IEEE_FLOAT_WAVE_FORMAT_TAG,
        "sample_encoding": "IEEE_FLOAT",
        "bits_per_sample": 32,
        "endianness": "little",
        "sample_rate_hz": sample_rate_hz,
        "frame_count": int(samples.shape[1]),
        "channel_count": int(samples.shape[0]),
        "file_interleave": "frame_major",
        "api_array_layout": "channel_major",
    }
    mismatches = [
        key for key, expected_value in expected.items() if value.get(key) != expected_value
    ]
    if mismatches:
        raise AudioContractError(
            "WAVE sidecar does not match decoded audio fields: "
            + ", ".join(mismatches)
        )
    if not isinstance(value.get("metadata"), dict):
        raise AudioContractError("WAVE sidecar metadata must be an object")
    return value


def read_float32_wav(
    path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    verify_sidecar: bool = True,
) -> Float32Wav:
    """Strictly read canonical IEEE-float WAVE bytes as channel-major samples."""

    audio_path = Path(path).absolute()
    try:
        payload = audio_path.read_bytes()
    except OSError as exc:
        raise AudioContractError(f"WAVE file is missing or unreadable: {exc}") from exc
    samples, rate = _parse_canonical_float32_wav(payload)
    sidecar_value: Mapping[str, Any] | None = None
    if verify_sidecar:
        sidecar = _sidecar_path(audio_path, sidecar_path)
        sidecar_value = _load_verified_sidecar(
            sidecar,
            audio_path=audio_path,
            audio_payload=payload,
            samples=samples,
            sample_rate_hz=rate,
        )
    return Float32Wav(
        path=audio_path,
        samples=samples,
        sample_rate_hz=rate,
        sidecar=sidecar_value,
    )
