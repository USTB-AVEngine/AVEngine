"""Structural validator for shared P6 acoustic render receipts."""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from avengine.capture.neutral_readback import validate_clock
from avengine.spatial_audio.audio import read_float32_wav

UNIFIED_AUDIO_RECEIPT_SCHEMA = "avengine_unified_audio_receipt_v1"


class UnifiedAudioReceiptError(ValueError):
    """A shared P6 receipt is malformed or does not match its media."""


def _required_mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UnifiedAudioReceiptError(f"{owner} must be an object")
    return value


def _required_nonnegative_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UnifiedAudioReceiptError(f"{owner} must be a non-negative integer")
    return int(value)


def _required_positive_int(value: Any, owner: str) -> int:
    result = _required_nonnegative_int(value, owner)
    if result < 1:
        raise UnifiedAudioReceiptError(f"{owner} must be positive")
    return result


def _required_gain(value: Any, owner: str) -> float:
    """Require a finite non-negative render gain without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise UnifiedAudioReceiptError(
            f"{owner} must be a finite non-negative real"
        )
    try:
        gain = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise UnifiedAudioReceiptError(
            f"{owner} must be a finite non-negative real"
        ) from error
    if not math.isfinite(gain) or gain < 0.0:
        raise UnifiedAudioReceiptError(
            f"{owner} must be a finite non-negative real"
        )
    return gain


def validate_unified_audio_receipt(
    receipt: Mapping[str, Any],
    *,
    require_files: bool = True,
) -> dict[str, Any]:
    """Validate P6 fields and, optionally, the emitted float32 WAV boundaries."""
    if not isinstance(receipt, Mapping):
        raise UnifiedAudioReceiptError("receipt must be an object")
    if receipt.get("schema") != UNIFIED_AUDIO_RECEIPT_SCHEMA:
        raise UnifiedAudioReceiptError("unsupported unified audio receipt schema")
    for key in (
        "clock",
        "audio",
        "events",
        "wet_tail_intervals",
        "peak_dbfs",
        "gain_application",
        "propagation",
        "hrtf",
        "input_neutral_readback",
    ):
        if key not in receipt:
            raise UnifiedAudioReceiptError(f"receipt lacks {key}")
    clock = _required_mapping(receipt["clock"], "clock")
    try:
        validated_clock = validate_clock(clock)
    except (TypeError, ValueError) as error:
        raise UnifiedAudioReceiptError(f"clock validation failed: {error}") from error
    frame_count = _required_positive_int(validated_clock.get("frame_count"), "clock.frame_count")
    sample_rate = _required_positive_int(validated_clock.get("sample_rate_hz"), "clock.sample_rate_hz")
    sample_count = _required_positive_int(validated_clock.get("sample_count"), "clock.sample_count")
    audio = _required_mapping(receipt["audio"], "audio")
    if audio.get("sample_rate_hz") != sample_rate or audio.get("sample_count") != sample_count:
        raise UnifiedAudioReceiptError("audio clock metadata differs from receipt clock")
    layouts = audio.get("layouts")
    if not isinstance(layouts, list) or not layouts or any(
        not isinstance(layout, str) or not layout for layout in layouts
    ):
        raise UnifiedAudioReceiptError("audio.layouts must be a nonempty list of layout names")
    if "binaural" not in layouts:
        raise UnifiedAudioReceiptError(
            "unified neutral output must include an actual binaural layout"
        )
    hrtf = receipt.get("hrtf")
    if not isinstance(hrtf, Mapping) or not isinstance(hrtf.get("id"), str) or not hrtf.get("id"):
        raise UnifiedAudioReceiptError("binaural receipt requires an HRTF identifier")
    if audio.get("layout_type") != "binaural":
        raise UnifiedAudioReceiptError("primary unified neutral layout must be binaural")
    if audio.get("channel_labels") != ["left", "right"]:
        raise UnifiedAudioReceiptError(
            "binaural unified neutral output must declare left/right channels"
        )
    by_layout = audio.get("by_layout")
    binaural_record = by_layout.get("binaural") if isinstance(by_layout, Mapping) else None
    if not isinstance(binaural_record, Mapping):
        raise UnifiedAudioReceiptError("receipt lacks binaural layout metadata")
    if (
        binaural_record.get("channel_count") != 2
        or binaural_record.get("channel_labels") != ["left", "right"]
        or binaural_record.get("sample_rate_hz") != sample_rate
        or binaural_record.get("sample_count") != sample_count
    ):
        raise UnifiedAudioReceiptError(
            "binaural layout metadata must declare two left/right channels and match the clock"
        )
    mixture = audio.get("mixture_path") or receipt.get("mixture_path")
    stems = audio.get("stems") or receipt.get("stems")
    if not isinstance(mixture, str) or not mixture:
        raise UnifiedAudioReceiptError("receipt lacks a mixture path")
    if not isinstance(stems, Mapping) or not stems:
        raise UnifiedAudioReceiptError("receipt lacks source stem paths")
    outputs_by_layout = receipt.get("outputs_by_layout")
    binaural_outputs = (
        outputs_by_layout.get("binaural")
        if isinstance(outputs_by_layout, Mapping)
        else None
    )
    if not isinstance(binaural_outputs, Mapping):
        raise UnifiedAudioReceiptError("receipt lacks binaural output paths")
    if binaural_outputs.get("mixture") != mixture:
        raise UnifiedAudioReceiptError("primary mixture path is not the binaural output")
    binaural_stems = binaural_outputs.get("stems")
    if not isinstance(binaural_stems, Mapping) or dict(binaural_stems) != dict(stems):
        raise UnifiedAudioReceiptError("primary stem paths are not the binaural outputs")
    if require_files:
        mixture_wav = read_float32_wav(Path(mixture))
        if (
            mixture_wav.sample_rate_hz != sample_rate
            or mixture_wav.frame_count != sample_count
            or mixture_wav.channel_count != 2
        ):
            raise UnifiedAudioReceiptError(
                "binaural mixture WAVE must be two-channel and match receipt clock"
            )
        for source_id, path in stems.items():
            if not isinstance(path, str) or not path:
                raise UnifiedAudioReceiptError(f"stem path is invalid for {source_id!r}")
            stem_wav = read_float32_wav(Path(path))
            if (
                stem_wav.sample_rate_hz != sample_rate
                or stem_wav.frame_count != sample_count
                or stem_wav.channel_count != 2
            ):
                raise UnifiedAudioReceiptError(
                    f"binaural stem WAVE must be two-channel and match clock for {source_id!r}"
                )
    neutral = _required_mapping(receipt["input_neutral_readback"], "input_neutral_readback")
    if "path" not in neutral:
        raise UnifiedAudioReceiptError("input_neutral_readback lacks path provenance")
    event_post_gains: list[float] = []
    events = receipt["events"]
    if not isinstance(events, list):
        raise UnifiedAudioReceiptError("receipt.events must be a list")
    event_ids: set[str] = set()
    for index, event in enumerate(events):
        row = _required_mapping(event, f"events[{index}]")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise UnifiedAudioReceiptError(f"events[{index}] has an invalid or duplicate event_id")
        event_ids.add(event_id)
        for key in ("source_activity_intervals_samples", "wet_tail_intervals", "gain_application"):
            if key not in row:
                raise UnifiedAudioReceiptError(f"events[{index}] lacks {key}")
        for interval_key in ("source_activity_intervals_samples", "wet_tail_intervals"):
            intervals = row[interval_key]
            if not isinstance(intervals, list):
                raise UnifiedAudioReceiptError(f"events[{index}].{interval_key} must be a list")
            for interval in intervals:
                if isinstance(interval, Mapping):
                    start_value = interval.get("start_sample")
                    end_value = interval.get("end_sample_exclusive")
                elif isinstance(interval, (list, tuple)) and len(interval) == 2:
                    start_value, end_value = interval
                else:
                    raise UnifiedAudioReceiptError(
                        f"events[{index}].{interval_key} interval must be an object or pair"
                    )
                start = _required_nonnegative_int(start_value, "interval.start_sample")
                end = _required_positive_int(end_value, "interval.end_sample_exclusive")
                if end <= start or end > sample_count:
                    raise UnifiedAudioReceiptError(f"events[{index}] interval escapes the episode clock")
        gain = _required_mapping(row["gain_application"], f"events[{index}].gain_application")
        if gain.get("application_count") != 1:
            raise UnifiedAudioReceiptError(f"events[{index}] does not prove one-time gain application")
        post_gain = _required_gain(
            gain.get("post_assembly_convolution_gain"),
            f"events[{index}].gain_application.post_assembly_convolution_gain",
        )
        if gain.get("post_assembly_convolution_gain_application_count", 1 if post_gain == 1.0 else None) != 1:
            raise UnifiedAudioReceiptError(
                f"events[{index}] does not prove one-time post-assembly gain application"
            )
        if "normalization" in gain and gain.get("normalization") is not False:
            raise UnifiedAudioReceiptError(f"events[{index}] enables normalization")
        event_post_gains.append(post_gain)
    wet = receipt["wet_tail_intervals"]
    if not isinstance(wet, list) or {row.get("event_id") for row in wet if isinstance(row, Mapping)} != event_ids:
        raise UnifiedAudioReceiptError("top-level wet_tail_intervals do not cover events exactly")
    for wet_index, row in enumerate(wet):
        if not isinstance(row, Mapping):
            raise UnifiedAudioReceiptError(f"wet_tail_intervals[{wet_index}] must be an object")
        start = _required_nonnegative_int(row.get("start_sample"), "wet_tail.start_sample")
        end = _required_positive_int(row.get("end_sample_exclusive"), "wet_tail.end_sample_exclusive")
        if end <= start or end > sample_count:
            raise UnifiedAudioReceiptError(
                f"wet_tail_intervals[{wet_index}] escapes the episode clock"
            )
    propagation = _required_mapping(receipt["propagation"], "propagation")
    if not isinstance(propagation.get("diffraction"), bool):
        raise UnifiedAudioReceiptError("propagation.diffraction must be boolean")
    _required_nonnegative_int(propagation.get("max_diffraction_order"), "propagation.max_diffraction_order")
    overall_gain = _required_mapping(receipt["gain_application"], "gain_application")
    if overall_gain.get("applied_once_per_event") is not True or overall_gain.get("normalization") is not False:
        raise UnifiedAudioReceiptError("receipt gain proof is incomplete")
    overall_post_gain = _required_gain(
        overall_gain.get("post_assembly_convolution_gain", 1.0),
        "gain_application.post_assembly_convolution_gain",
    )
    if overall_gain.get("post_assembly_convolution_gain_application_count", 1 if overall_post_gain == 1.0 else None) != 1:
        raise UnifiedAudioReceiptError(
            "receipt does not prove one-time post-assembly gain application"
        )
    if "limiting" in overall_gain and overall_gain.get("limiting") is not False:
        raise UnifiedAudioReceiptError("receipt enables limiting")
    if "post_assembly_convolution_gain" in audio:
        audio_post_gain = _required_gain(
            audio["post_assembly_convolution_gain"],
            "audio.post_assembly_convolution_gain",
        )
        if audio_post_gain != overall_post_gain:
            raise UnifiedAudioReceiptError("audio and batch post-assembly gains differ")
    if any(gain != overall_post_gain for gain in event_post_gains):
        raise UnifiedAudioReceiptError("event and batch post-assembly gains differ")
    return {
        "status": "pass",
        "frame_count": frame_count,
        "sample_rate_hz": sample_rate,
        "sample_count": sample_count,
        "event_count": len(events),
        "source_count": len(stems),
    }


__all__ = ["UNIFIED_AUDIO_RECEIPT_SCHEMA", "UnifiedAudioReceiptError", "validate_unified_audio_receipt"]
