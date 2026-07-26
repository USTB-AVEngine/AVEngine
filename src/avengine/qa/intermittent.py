"""Declared intermittent sound-event windows for the asset-bound batch line.

Truth by declaration, not detection: sub-windows are planned
deterministically, declared in the AudioProgram event vocabulary
(``start_tick = start_sample * 3`` at the 48 kHz timeline base over 16 kHz
audio) and enforced on the waveform by a raised-cosine gating envelope
applied to the dry clip before convolution. The declared window IS the
ground truth; the RIR cache is reused untouched because cache keys are
dry-audio independent.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

TICKS_PER_SAMPLE = 3
SAMPLE_RATE_HZ = 16_000
SAMPLE_COUNT = 80_000
DEFAULT_FADE_SAMPLES = 80
WINDOW_AUTHORITY = "declared_intermittent_program_v1"

MIN_EVENT_SAMPLES = 8_000  # 0.5 s
MAX_EVENT_SAMPLES = 24_000  # 1.5 s
MIN_GAP_SAMPLES = 6_400  # 0.4 s
EDGE_MARGIN_SAMPLES = 1_600  # 0.1 s away from the clip edges


class QAIntermittentError(ValueError):
    """An intermittent window plan violates its declared contract."""


def _digest_stream(*parts: str):
    """Yield deterministic floats in [0, 1) from a hash counter stream."""

    counter = 0
    while True:
        digest = hashlib.sha256(
            "\0".join((*parts, str(counter))).encode("utf-8")
        ).digest()
        yield int.from_bytes(digest[:8], "big") / float(1 << 64)
        counter += 1


def plan_slot_windows(
    *,
    seed: str,
    episode_id: str,
    slot_id: str,
    event_count_choices: Sequence[int] = (2, 3),
    sample_count: int = SAMPLE_COUNT,
) -> list[tuple[int, int]]:
    """Plan non-overlapping [start, end) sample windows for one source slot."""

    if sample_count <= 2 * EDGE_MARGIN_SAMPLES + MIN_EVENT_SAMPLES:
        raise QAIntermittentError("sample_count is too small for one window")
    stream = _digest_stream(seed, episode_id, slot_id)
    count = event_count_choices[
        int(next(stream) * len(event_count_choices)) % len(event_count_choices)
    ]
    for _ in range(64):
        durations = [
            MIN_EVENT_SAMPLES
            + int(next(stream) * (MAX_EVENT_SAMPLES - MIN_EVENT_SAMPLES))
            for _ in range(count)
        ]
        occupied = sum(durations) + MIN_GAP_SAMPLES * (count - 1)
        free = sample_count - 2 * EDGE_MARGIN_SAMPLES - occupied
        if free < 0:
            count = max(1, count - 1)
            continue
        cuts = sorted(next(stream) for _ in range(count))
        shares = [cuts[0]] + [
            cuts[index] - cuts[index - 1] for index in range(1, count)
        ]
        windows: list[tuple[int, int]] = []
        cursor = EDGE_MARGIN_SAMPLES
        for duration, share in zip(durations, shares):
            cursor += int(share * free / max(sum(shares), 1.0e-9))
            start = cursor
            end = start + duration
            windows.append((start, end))
            cursor = end + MIN_GAP_SAMPLES
        if windows[-1][1] <= sample_count - EDGE_MARGIN_SAMPLES:
            return windows
    raise QAIntermittentError(
        f"could not place {count} windows for {episode_id}/{slot_id}"
    )


def validate_windows(
    windows: Sequence[tuple[int, int]],
    *,
    sample_count: int = SAMPLE_COUNT,
    fade_samples: int = DEFAULT_FADE_SAMPLES,
) -> None:
    if not windows:
        raise QAIntermittentError("at least one window is required")
    previous_end = None
    for start, end in windows:
        if not 0 <= start < end <= sample_count:
            raise QAIntermittentError(f"window [{start}, {end}) is out of range")
        if end - start < 2 * fade_samples:
            raise QAIntermittentError("window is shorter than its fades")
        if previous_end is not None and start - previous_end < MIN_GAP_SAMPLES:
            raise QAIntermittentError("windows must keep the declared minimum gap")
        previous_end = end


def gating_envelope(
    windows: Sequence[tuple[int, int]],
    *,
    sample_count: int = SAMPLE_COUNT,
    fade_samples: int = DEFAULT_FADE_SAMPLES,
) -> np.ndarray:
    """Raised-cosine 0/1 envelope; fades live inside each declared window."""

    validate_windows(
        windows, sample_count=sample_count, fade_samples=fade_samples
    )
    envelope = np.zeros(sample_count, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(
        np.pi * (np.arange(fade_samples, dtype=np.float64) + 0.5) / fade_samples
    )
    for start, end in windows:
        envelope[start:end] = 1.0
        envelope[start : start + fade_samples] = ramp
        envelope[end - fade_samples : end] = ramp[::-1]
    return envelope


def event_records(
    *,
    slot_id: str,
    windows: Sequence[tuple[int, int]],
    fade_samples: int = DEFAULT_FADE_SAMPLES,
) -> list[dict[str, Any]]:
    """Events in the AudioProgram vocabulary (ticks derived, gating slice)."""

    validate_windows(windows, fade_samples=fade_samples)
    records = []
    for index, (start, end) in enumerate(windows):
        records.append(
            {
                "event_id": f"{slot_id}_event_{index:03d}",
                "source_slot_id": slot_id,
                "start_sample": int(start),
                "end_sample_exclusive": int(end),
                "start_tick": int(start) * TICKS_PER_SAMPLE,
                "end_tick_exclusive": int(end) * TICKS_PER_SAMPLE,
                "source_start_sample": int(start),
                "source_end_sample_exclusive": int(end),
                "fade_samples": int(fade_samples),
                "gating": "raised_cosine_envelope_on_continuous_dry",
            }
        )
    return records


def frame_window(
    start_tick: int, end_tick_exclusive: int, *, ticks_per_frame: int = 3200
) -> tuple[int, int]:
    """Video frames [start, end) that overlap an event's tick window."""

    if not 0 <= start_tick < end_tick_exclusive:
        raise QAIntermittentError("tick window must be non-empty and non-negative")
    start_frame = start_tick // ticks_per_frame
    end_frame = -(-end_tick_exclusive // ticks_per_frame)
    return int(start_frame), int(end_frame)
