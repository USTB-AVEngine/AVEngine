"""Deterministic exact-length dry-audio bus assembly for M5.1 review clips.

The source-contract schema identifies dry assets by ID and SHA-256, while an
executable local review also needs a concrete file path.  This module keeps
that boundary explicit: an event may declare its path directly, or a caller
may provide an ``asset_bindings`` mapping from the schema's asset ID to a
local path.  Conflicting aliases, hashes, or bindings fail closed.

Only uncompressed mono integer-PCM WAVE files are decoded.  Source-native
clips are deterministically resampled on a float64 linear time grid, fitted to
the declared half-open event window by tail cropping or zero padding, faded,
gained, and summed into canonical named-source buses.  No normalization,
limiting, network access, or implicit codec conversion occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import wave

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m4.audio import AudioContractError, canonical_source_ids


DRY_AUDIO_ASSEMBLY_SCHEMA = "avengine_m5_1_dry_audio_assembly_v1"
RESAMPLING_ALGORITHM = "float64_linear_source_time_grid_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MISSING = object()
_NO_DEFAULT = object()


@dataclass(frozen=True)
class DryAudioClipSpec:
    """Exact video/audio duration contract for one review clip."""

    frame_count: int
    fps_numerator: int
    fps_denominator: int
    sample_rate_hz: int
    sample_count: int

    @classmethod
    def from_values(
        cls,
        *,
        frame_count: int,
        fps_numerator: int,
        fps_denominator: int = 1,
        sample_rate_hz: int,
    ) -> "DryAudioClipSpec":
        frames = _positive_integer(frame_count, owner="frame_count")
        fps_num = _positive_integer(fps_numerator, owner="fps_numerator")
        fps_den = _positive_integer(fps_denominator, owner="fps_denominator")
        rate = _positive_integer(sample_rate_hz, owner="sample_rate_hz")
        sample_count = _round_nonnegative_fraction(
            Fraction(frames * rate * fps_den, fps_num)
        )
        if sample_count < 1:
            raise AudioContractError("clip duration rounds to zero audio samples")
        return cls(
            frame_count=frames,
            fps_numerator=fps_num,
            fps_denominator=fps_den,
            sample_rate_hz=rate,
            sample_count=sample_count,
        )

    def sample_boundary(self, frame_index: int) -> int:
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, (int, np.integer))
            or not 0 <= int(frame_index) <= self.frame_count
        ):
            raise AudioContractError(
                f"frame boundary must lie in [0,{self.frame_count}]"
            )
        return _round_nonnegative_fraction(
            Fraction(
                int(frame_index)
                * self.sample_rate_hz
                * self.fps_denominator,
                self.fps_numerator,
            )
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "frame_count": self.frame_count,
            "fps_numerator": self.fps_numerator,
            "fps_denominator": self.fps_denominator,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class DecodedMonoAsset:
    """Owned float64 samples and authenticated WAVE metadata."""

    path: Path
    sha256: str
    samples: np.ndarray
    sample_rate_hz: int
    sample_width_bytes: int
    frame_count: int
    byte_size: int


@dataclass(frozen=True)
class DryAudioEvent:
    """One fully resolved dry placement declaration."""

    event_id: str
    source_id: str
    start_sample: int
    end_sample_exclusive: int
    dry_asset_id: str | None
    dry_asset_path: Path
    dry_asset_sha256: str
    dry_clip_start_sample: int
    dry_clip_end_sample_exclusive: int | None
    linear_gain: float
    fade_in_samples: int
    fade_out_samples: int


@dataclass(frozen=True)
class DryAudioAssembly:
    """Exact named buses plus replayable placement evidence."""

    clip: DryAudioClipSpec
    source_ids: tuple[str, ...]
    buses: Mapping[str, np.ndarray]
    placement_receipts: tuple[Mapping[str, Any], ...]
    bus_float64_le_sha256: Mapping[str, str]
    assembly_content_sha256: str

    def metadata(self) -> dict[str, Any]:
        content = {
            "schema": DRY_AUDIO_ASSEMBLY_SCHEMA,
            "qualification_claim": False,
            "clip": self.clip.to_dict(),
            "source_ids": list(self.source_ids),
            "arithmetic": _arithmetic_record(),
            "placement_receipts": [dict(value) for value in self.placement_receipts],
            "bus_float64_le_sha256": dict(self.bus_float64_le_sha256),
        }
        return {**content, "assembly_content_sha256": self.assembly_content_sha256}


def _positive_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AudioContractError(f"{owner} must be a positive integer")
    result = int(value)
    if result < 1:
        raise AudioContractError(f"{owner} must be a positive integer")
    return result


def _nonnegative_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AudioContractError(f"{owner} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise AudioContractError(f"{owner} must be a non-negative integer")
    return result


def _round_nonnegative_fraction(value: Fraction) -> int:
    if value < 0:
        raise AudioContractError("sample boundary cannot be negative")
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(remainder * 2 >= value.denominator)


def _float64_le_sha256(samples: np.ndarray) -> str:
    payload = np.ascontiguousarray(samples, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _arithmetic_record() -> dict[str, Any]:
    return {
        "bus_dtype": "float64_le",
        "resampling_algorithm": RESAMPLING_ALGORITHM,
        "resampling_output_length_rounding": "nearest_half_up_minimum_one",
        "resampling_anti_alias_filter": False,
        "placement": "half_open_crop_or_zero_pad_then_sum_v1",
        "overlap": "canonical_event_order_float64_sum_v1",
        "normalization": False,
        "limiting": False,
    }


def _coalesce(
    value: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    owner: str,
    default: Any = _NO_DEFAULT,
) -> Any:
    found = [(name, value[name]) for name in aliases if name in value]
    if not found:
        if default is _NO_DEFAULT:
            raise AudioContractError(
                f"{owner} requires exactly one of {', '.join(aliases)}"
            )
        return default
    first_name, first_value = found[0]
    for name, observed in found[1:]:
        if observed != first_value:
            raise AudioContractError(
                f"{owner} has conflicting aliases {first_name!r} and {name!r}"
            )
    return first_value


def _stable_id(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not _STABLE_ID.fullmatch(value):
        raise AudioContractError(
            f"{owner} must match [A-Za-z0-9][A-Za-z0-9_.-]{{0,127}}"
        )
    return value


def _sha256(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise AudioContractError(f"{owner} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative_gain(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, np.number)):
        raise AudioContractError(f"{owner} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AudioContractError(f"{owner} must be a finite non-negative number")
    return result


def _resolved_local_path(value: Any, *, asset_root: Path | None, owner: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise AudioContractError(f"{owner} must be a non-empty local path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        if asset_root is None:
            raise AudioContractError(f"relative {owner} requires explicit asset_root")
        path = asset_root / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise AudioContractError(f"{owner} is not a regular file: {resolved}")
    return resolved


def read_authenticated_mono_pcm_wav(
    path: str | Path,
    *,
    expected_sha256: str,
) -> DecodedMonoAsset:
    """Authenticate and decode one mono integer-PCM WAVE into owned float64."""

    digest = _sha256(expected_sha256, owner="expected_sha256")
    resolved = _resolved_local_path(path, asset_root=None, owner="dry asset path")
    actual = sha256_file(resolved)
    if actual != digest:
        raise AudioContractError(
            f"dry asset SHA-256 differs for {resolved}: expected {digest}, got {actual}"
        )
    try:
        with wave.open(str(resolved), "rb") as handle:
            channel_count = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
            payload = handle.readframes(frame_count)
            trailing = handle.readframes(1)
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioContractError(f"cannot decode dry WAVE {resolved}: {exc}") from exc
    if channel_count != 1:
        raise AudioContractError("dry WAVE must contain exactly one channel")
    if compression != "NONE" or sample_width not in {1, 2, 3, 4}:
        raise AudioContractError(
            "dry WAVE must be uncompressed 8/16/24/32-bit integer PCM"
        )
    if sample_rate < 1 or frame_count < 1:
        raise AudioContractError("dry WAVE must have positive rate and frame count")
    expected_bytes = frame_count * sample_width
    if len(payload) != expected_bytes or trailing:
        raise AudioContractError("dry WAVE payload length differs from its header")

    if sample_width == 1:
        integers = np.frombuffer(payload, dtype=np.uint8).astype(np.float64)
        samples = (integers - 128.0) / 128.0
    elif sample_width == 2:
        integers = np.frombuffer(payload, dtype="<i2").astype(np.float64)
        samples = integers / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        unsigned = (
            packed[:, 0].astype(np.int64)
            | (packed[:, 1].astype(np.int64) << 8)
            | (packed[:, 2].astype(np.int64) << 16)
        )
        signed = np.where(unsigned & 0x800000, unsigned - 0x1000000, unsigned)
        samples = signed.astype(np.float64) / 8388608.0
    else:
        integers = np.frombuffer(payload, dtype="<i4").astype(np.float64)
        samples = integers / 2147483648.0
    samples = np.ascontiguousarray(samples, dtype=np.float64)
    if samples.shape != (frame_count,) or not np.all(np.isfinite(samples)):
        raise AudioContractError("decoded dry WAVE samples are malformed")
    return DecodedMonoAsset(
        path=resolved,
        sha256=actual,
        samples=samples,
        sample_rate_hz=int(sample_rate),
        sample_width_bytes=int(sample_width),
        frame_count=int(frame_count),
        byte_size=resolved.stat().st_size,
    )


def deterministic_resample_mono(
    samples: Any,
    *,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> np.ndarray:
    """Resample finite mono samples on one explicit linear source-time grid."""

    source_rate = _positive_integer(
        source_sample_rate_hz, owner="source_sample_rate_hz"
    )
    target_rate = _positive_integer(
        target_sample_rate_hz, owner="target_sample_rate_hz"
    )
    source = np.asarray(samples)
    if source.ndim != 1 or source.dtype.kind not in "iuf" or source.size < 1:
        raise AudioContractError("resampling input must be a non-empty real mono array")
    source = np.ascontiguousarray(source, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise AudioContractError("resampling input must contain only finite samples")
    if source_rate == target_rate:
        return source.copy()
    output_count = max(
        1,
        _round_nonnegative_fraction(
            Fraction(source.size * target_rate, source_rate)
        ),
    )
    output_indices = np.arange(output_count, dtype=np.int64)
    numerators = output_indices * source_rate
    left = numerators // target_rate
    remainder = numerators % target_rate
    left = np.minimum(left, source.size - 1)
    right = np.minimum(left + 1, source.size - 1)
    fraction = remainder.astype(np.float64) / float(target_rate)
    result = source[left] * (1.0 - fraction) + source[right] * fraction
    if not np.all(np.isfinite(result)):
        raise AudioContractError("deterministic resampling overflowed float64")
    return np.ascontiguousarray(result, dtype=np.float64)


def _binding_record(
    asset_bindings: Mapping[str, Any] | None,
    asset_id: str | None,
) -> tuple[Any, Any]:
    if asset_id is None or asset_bindings is None or asset_id not in asset_bindings:
        return _MISSING, _MISSING
    binding = asset_bindings[asset_id]
    if isinstance(binding, Mapping):
        unknown = set(binding) - {"path", "sha256"}
        if unknown:
            raise AudioContractError(
                f"asset binding {asset_id!r} has unknown keys: {sorted(unknown)}"
            )
        return binding.get("path", _MISSING), binding.get("sha256", _MISSING)
    return binding, _MISSING


def _optional_clip_value(
    event: Mapping[str, Any],
    nested: Mapping[str, Any] | None,
    aliases: Sequence[str],
    nested_name: str,
    *,
    owner: str,
    default: Any,
) -> Any:
    direct = _coalesce(event, aliases, owner=owner, default=_MISSING)
    nested_value = (
        nested[nested_name]
        if nested is not None and nested_name in nested
        else _MISSING
    )
    if direct is not _MISSING and nested_value is not _MISSING and direct != nested_value:
        raise AudioContractError(f"{owner} conflicts with dry_clip.{nested_name}")
    if direct is not _MISSING:
        return direct
    if nested_value is not _MISSING:
        return nested_value
    return default


def parse_dry_audio_events(
    event_mappings: Sequence[Mapping[str, Any]],
    *,
    source_ids: Sequence[str],
    clip: DryAudioClipSpec,
    asset_bindings: Mapping[str, Any] | None = None,
    asset_root: str | Path | None = None,
) -> tuple[DryAudioEvent, ...]:
    """Resolve schema-compatible event mappings into canonical placements.

    Accepted compatibility aliases are deliberately finite:

    * ``dry_audio_asset_id`` or ``dry_asset_id``;
    * ``dry_audio_asset_path`` or ``dry_asset_path``;
    * ``dry_audio_asset_sha256`` or ``dry_asset_sha256``;
    * source-native clip bounds as direct ``dry_clip_*`` fields or a
      ``dry_clip`` mapping;
    * ``linear_gain`` or ``gain`` and symmetric ``fade_samples`` or explicit
      ``fade_in_samples`` / ``fade_out_samples``.
    """

    if not isinstance(clip, DryAudioClipSpec):
        raise AudioContractError("clip must be DryAudioClipSpec")
    canonical = canonical_source_ids(source_ids)
    if tuple(source_ids) != canonical:
        raise AudioContractError("source_ids must already be in canonical order")
    if isinstance(event_mappings, (str, bytes)) or not isinstance(
        event_mappings, Sequence
    ):
        raise AudioContractError("event_mappings must be a sequence of mappings")
    if asset_bindings is not None and not isinstance(asset_bindings, Mapping):
        raise AudioContractError("asset_bindings must be a mapping when provided")
    root = Path(asset_root).expanduser().resolve() if asset_root is not None else None
    parsed: list[DryAudioEvent] = []
    observed_event_ids: set[str] = set()
    for index, raw in enumerate(event_mappings):
        owner = f"event_mappings[{index}]"
        if not isinstance(raw, Mapping):
            raise AudioContractError(f"{owner} must be a mapping")
        event_id = _stable_id(raw.get("event_id"), owner=f"{owner}.event_id")
        if event_id in observed_event_ids:
            raise AudioContractError(f"duplicate event_id {event_id!r}")
        observed_event_ids.add(event_id)
        source_id = _stable_id(raw.get("source_id"), owner=f"{owner}.source_id")
        if source_id not in canonical:
            raise AudioContractError(f"{owner}.source_id is not declared")
        start = _nonnegative_integer(raw.get("start_sample"), owner=f"{owner}.start_sample")
        end = _positive_integer(
            raw.get("end_sample_exclusive"),
            owner=f"{owner}.end_sample_exclusive",
        )
        if not start < end <= clip.sample_count:
            raise AudioContractError(
                f"{owner} must satisfy 0 <= start_sample < end_sample_exclusive "
                f"<= {clip.sample_count}"
            )
        if "start_frame" in raw:
            start_frame = _nonnegative_integer(
                raw["start_frame"], owner=f"{owner}.start_frame"
            )
            if start_frame >= clip.frame_count or clip.sample_boundary(start_frame) != start:
                raise AudioContractError(f"{owner}.start_frame conflicts with start_sample")
        if "end_frame_exclusive" in raw:
            end_frame = _positive_integer(
                raw["end_frame_exclusive"], owner=f"{owner}.end_frame_exclusive"
            )
            if (
                end_frame > clip.frame_count
                or clip.sample_boundary(end_frame) != end
            ):
                raise AudioContractError(
                    f"{owner}.end_frame_exclusive conflicts with end_sample_exclusive"
                )

        raw_asset_id = _coalesce(
            raw,
            ("dry_audio_asset_id", "dry_asset_id"),
            owner=f"{owner}.dry_asset_id",
            default=None,
        )
        asset_id = (
            _stable_id(raw_asset_id, owner=f"{owner}.dry_asset_id")
            if raw_asset_id is not None
            else None
        )
        binding_path, binding_hash = _binding_record(asset_bindings, asset_id)
        direct_path = _coalesce(
            raw,
            ("dry_audio_asset_path", "dry_asset_path"),
            owner=f"{owner}.dry_asset_path",
            default=_MISSING,
        )
        if direct_path is _MISSING and binding_path is _MISSING:
            raise AudioContractError(
                f"{owner} must declare a dry asset path or resolve asset_id in asset_bindings"
            )
        resolved_direct = (
            _resolved_local_path(
                direct_path, asset_root=root, owner=f"{owner}.dry_asset_path"
            )
            if direct_path is not _MISSING
            else None
        )
        resolved_binding = (
            _resolved_local_path(
                binding_path,
                asset_root=root,
                owner=f"asset binding {asset_id!r} path",
            )
            if binding_path is not _MISSING
            else None
        )
        if (
            resolved_direct is not None
            and resolved_binding is not None
            and resolved_direct != resolved_binding
        ):
            raise AudioContractError(f"{owner} path conflicts with asset binding")
        resolved_path = resolved_direct or resolved_binding
        assert resolved_path is not None

        event_hash = _sha256(
            _coalesce(
                raw,
                ("dry_audio_asset_sha256", "dry_asset_sha256"),
                owner=f"{owner}.dry_asset_sha256",
            ),
            owner=f"{owner}.dry_asset_sha256",
        )
        if binding_hash is not _MISSING:
            normalized_binding_hash = _sha256(
                binding_hash, owner=f"asset binding {asset_id!r} sha256"
            )
            if normalized_binding_hash != event_hash:
                raise AudioContractError(f"{owner} hash conflicts with asset binding")

        nested_clip = raw.get("dry_clip")
        if nested_clip is not None:
            if not isinstance(nested_clip, Mapping):
                raise AudioContractError(f"{owner}.dry_clip must be a mapping")
            unknown = set(nested_clip) - {"start_sample", "end_sample_exclusive"}
            if unknown:
                raise AudioContractError(
                    f"{owner}.dry_clip has unknown keys: {sorted(unknown)}"
                )
        clip_start = _nonnegative_integer(
            _optional_clip_value(
                raw,
                nested_clip,
                ("dry_clip_start_sample", "dry_clip_start"),
                "start_sample",
                owner=f"{owner}.dry_clip_start_sample",
                default=0,
            ),
            owner=f"{owner}.dry_clip_start_sample",
        )
        raw_clip_end = _optional_clip_value(
            raw,
            nested_clip,
            (
                "dry_clip_end_sample_exclusive",
                "dry_clip_end_exclusive",
                "dry_clip_end",
            ),
            "end_sample_exclusive",
            owner=f"{owner}.dry_clip_end_sample_exclusive",
            default=None,
        )
        clip_end = (
            _positive_integer(
                raw_clip_end, owner=f"{owner}.dry_clip_end_sample_exclusive"
            )
            if raw_clip_end is not None
            else None
        )
        if clip_end is not None and clip_start >= clip_end:
            raise AudioContractError(
                f"{owner} dry clip must satisfy start < end_exclusive"
            )
        gain = _finite_nonnegative_gain(
            _coalesce(
                raw,
                ("linear_gain", "gain"),
                owner=f"{owner}.linear_gain",
                default=1.0,
            ),
            owner=f"{owner}.linear_gain",
        )
        symmetric_fade = _coalesce(
            raw,
            ("fade_samples",),
            owner=f"{owner}.fade_samples",
            default=0,
        )
        symmetric_fade = _nonnegative_integer(
            symmetric_fade, owner=f"{owner}.fade_samples"
        )
        fade_in = _nonnegative_integer(
            raw.get("fade_in_samples", symmetric_fade),
            owner=f"{owner}.fade_in_samples",
        )
        fade_out = _nonnegative_integer(
            raw.get("fade_out_samples", symmetric_fade),
            owner=f"{owner}.fade_out_samples",
        )
        if "fade_samples" in raw and (
            ("fade_in_samples" in raw and fade_in != symmetric_fade)
            or ("fade_out_samples" in raw and fade_out != symmetric_fade)
        ):
            raise AudioContractError(
                f"{owner}.fade_samples conflicts with explicit fade endpoints"
            )
        parsed.append(
            DryAudioEvent(
                event_id=event_id,
                source_id=source_id,
                start_sample=start,
                end_sample_exclusive=end,
                dry_asset_id=asset_id,
                dry_asset_path=resolved_path,
                dry_asset_sha256=event_hash,
                dry_clip_start_sample=clip_start,
                dry_clip_end_sample_exclusive=clip_end,
                linear_gain=gain,
                fade_in_samples=fade_in,
                fade_out_samples=fade_out,
            )
        )
    return tuple(
        sorted(
            parsed,
            key=lambda event: (
                event.start_sample,
                event.end_sample_exclusive,
                event.source_id.encode("ascii"),
                event.event_id.encode("ascii"),
            ),
        )
    )


def _fade_envelope(
    sample_count: int,
    *,
    fade_in_samples: int,
    fade_out_samples: int,
) -> np.ndarray:
    if fade_in_samples + fade_out_samples > sample_count:
        raise AudioContractError(
            "fade_in_samples + fade_out_samples exceeds retained dry clip"
        )
    envelope = np.ones(sample_count, dtype=np.float64)
    if fade_in_samples:
        envelope[:fade_in_samples] *= np.linspace(
            0.0, 1.0, fade_in_samples, endpoint=True, dtype=np.float64
        )
    if fade_out_samples:
        envelope[-fade_out_samples:] *= np.linspace(
            1.0, 0.0, fade_out_samples, endpoint=True, dtype=np.float64
        )
    return envelope


def assemble_dry_audio_buses(
    event_mappings: Sequence[Mapping[str, Any]],
    *,
    source_ids: Sequence[str],
    clip: DryAudioClipSpec,
    asset_bindings: Mapping[str, Any] | None = None,
    asset_root: str | Path | None = None,
) -> DryAudioAssembly:
    """Build exact-length canonical buses and authenticated placement receipts."""

    events = parse_dry_audio_events(
        event_mappings,
        source_ids=source_ids,
        clip=clip,
        asset_bindings=asset_bindings,
        asset_root=asset_root,
    )
    canonical = tuple(source_ids)
    buses = {
        source_id: np.zeros(clip.sample_count, dtype=np.float64)
        for source_id in canonical
    }
    assets: dict[tuple[Path, str], DecodedMonoAsset] = {}
    receipts: list[dict[str, Any]] = []
    for event in events:
        cache_key = (event.dry_asset_path, event.dry_asset_sha256)
        if cache_key not in assets:
            assets[cache_key] = read_authenticated_mono_pcm_wav(
                event.dry_asset_path,
                expected_sha256=event.dry_asset_sha256,
            )
        asset = assets[cache_key]
        clip_end = (
            asset.frame_count
            if event.dry_clip_end_sample_exclusive is None
            else event.dry_clip_end_sample_exclusive
        )
        if not 0 <= event.dry_clip_start_sample < clip_end <= asset.frame_count:
            raise AudioContractError(
                f"event {event.event_id!r} dry clip escapes source-native asset"
            )
        selected = asset.samples[event.dry_clip_start_sample:clip_end]
        resampled = deterministic_resample_mono(
            selected,
            source_sample_rate_hz=asset.sample_rate_hz,
            target_sample_rate_hz=clip.sample_rate_hz,
        )
        placement_count = event.end_sample_exclusive - event.start_sample
        copied_count = min(placement_count, int(resampled.size))
        cropped_count = max(0, int(resampled.size) - placement_count)
        zero_padded_count = max(0, placement_count - int(resampled.size))
        envelope = _fade_envelope(
            copied_count,
            fade_in_samples=event.fade_in_samples,
            fade_out_samples=event.fade_out_samples,
        )
        contribution = np.zeros(placement_count, dtype=np.float64)
        contribution[:copied_count] = (
            resampled[:copied_count] * envelope * event.linear_gain
        )
        if not np.all(np.isfinite(contribution)):
            raise AudioContractError(
                f"event {event.event_id!r} contribution overflowed float64"
            )
        target = buses[event.source_id][
            event.start_sample : event.end_sample_exclusive
        ]
        target += contribution
        if not np.all(np.isfinite(target)):
            raise AudioContractError(
                f"source {event.source_id!r} bus overflowed float64"
            )
        receipts.append(
            {
                "event_id": event.event_id,
                "source_id": event.source_id,
                "target_interval": {
                    "start_sample": event.start_sample,
                    "end_sample_exclusive": event.end_sample_exclusive,
                    "sample_count": placement_count,
                },
                "dry_asset": {
                    "asset_id": event.dry_asset_id,
                    "path": event.dry_asset_path.as_posix(),
                    "sha256": asset.sha256,
                    "byte_size": asset.byte_size,
                    "sample_rate_hz": asset.sample_rate_hz,
                    "sample_width_bytes": asset.sample_width_bytes,
                    "frame_count": asset.frame_count,
                },
                "dry_clip_source_native_interval": {
                    "start_sample": event.dry_clip_start_sample,
                    "end_sample_exclusive": clip_end,
                    "sample_count": clip_end - event.dry_clip_start_sample,
                },
                "resampling": {
                    "algorithm": RESAMPLING_ALGORITHM,
                    "output_length_rounding": "nearest_half_up_minimum_one",
                    "anti_alias_filter": False,
                    "source_sample_rate_hz": asset.sample_rate_hz,
                    "target_sample_rate_hz": clip.sample_rate_hz,
                    "input_sample_count": int(selected.size),
                    "output_sample_count": int(resampled.size),
                    "performed": asset.sample_rate_hz != clip.sample_rate_hz,
                },
                "fit": {
                    "copied_sample_count": copied_count,
                    "cropped_tail_sample_count": cropped_count,
                    "zero_padded_tail_sample_count": zero_padded_count,
                },
                "linear_gain": event.linear_gain,
                "fade_in_samples": event.fade_in_samples,
                "fade_out_samples": event.fade_out_samples,
                "contribution_float64_le_sha256": _float64_le_sha256(
                    contribution
                ),
            }
        )
    bus_hashes = {
        source_id: _float64_le_sha256(buses[source_id])
        for source_id in canonical
    }
    for bus in buses.values():
        bus.setflags(write=False)
    content = {
        "schema": DRY_AUDIO_ASSEMBLY_SCHEMA,
        "qualification_claim": False,
        "clip": clip.to_dict(),
        "source_ids": list(canonical),
        "arithmetic": _arithmetic_record(),
        "placement_receipts": receipts,
        "bus_float64_le_sha256": bus_hashes,
    }
    return DryAudioAssembly(
        clip=clip,
        source_ids=canonical,
        buses=buses,
        placement_receipts=tuple(receipts),
        bus_float64_le_sha256=bus_hashes,
        assembly_content_sha256=canonical_json_sha256(content),
    )


__all__ = [
    "DRY_AUDIO_ASSEMBLY_SCHEMA",
    "RESAMPLING_ALGORITHM",
    "DecodedMonoAsset",
    "DryAudioAssembly",
    "DryAudioClipSpec",
    "DryAudioEvent",
    "assemble_dry_audio_buses",
    "deterministic_resample_mono",
    "parse_dry_audio_events",
    "read_authenticated_mono_pcm_wav",
]
