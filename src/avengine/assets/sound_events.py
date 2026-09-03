"""Cut dry clips into one sounding event per burst.

The QA scheduler treats a bark as a 0.3 s event on a 5 s clip. A 9 s
file that still contains several barks cannot sit on that schedule:
later pulses leak after the intended onset. Head-and-tail trim does
not fix this, because it keeps everything between the first and last
sound.

Pulse classes (dog bark, cat meow, doorbell, alarms) are split with a
hysteresis energy gate. Continuous classes (speech, music, appliance
hum) stay one event: first sound to last sound, with a short guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Policy = Literal["pulse", "continuous"]

FRAME_S = 0.010
GUARD_S = 0.030
MIN_EVENT_S = 0.080
MAX_EVENT_S = 1.500
MERGE_GAP_S = 0.060
HANGOVER_S = 0.040
ENTER_DB_ABOVE_NOISE = 12.0
EXIT_DB_ABOVE_NOISE = 6.0
NOISE_PERCENTILE = 20.0
ABS_NOISE_FLOOR = 1e-5

PULSE_CLASSES = frozenset(
    {
        "dog_bark",
        "cat_meow",
        "doorbell",
        "doorbell_chime",
        "ding_dong",
        "chime",
        "alarm_bell",
        "alarm_beep",
        "alarm_clock",
        "buzzer",
        "ringtone",
        "phone_ring",
        "telephone",
        "telephone_bell_ringing",
        "telephone_dialing_dtmf",
        "busy_signal",
        "microwave_beep",
        "smoke_alarm",
        "fire_alarm",
    }
)

_PULSE_TOKENS = (
    "bark",
    "meow",
    "beep",
    "bell",
    "chime",
    "ring",
    "alarm",
    "buzzer",
    "doorbell",
    "ding",
)


class SoundEventError(ValueError):
    """A clip has no usable sounding event."""


@dataclass(frozen=True)
class SoundEvent:
    """One contiguous burst, sample-accurate in the source clip."""

    start_sample: int
    end_sample_exclusive: int
    purpose: str
    truncated: bool = False
    untruncated_end_sample_exclusive: int | None = None

    def duration_s(self, rate: int) -> float:
        return (self.end_sample_exclusive - self.start_sample) / rate


def event_policy_for_class(event_class: str | None) -> Policy:
    """Pulse classes split; everything else stays one trimmed span."""

    name = (event_class or "").strip().lower().replace("-", "_").replace(" ", "_")
    if name in PULSE_CLASSES:
        return "pulse"
    if any(token in name for token in _PULSE_TOKENS):
        return "pulse"
    return "continuous"


def _rms_frames(samples: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    hop = max(1, int(round(FRAME_S * rate)))
    usable = len(samples) // hop * hop
    if usable == 0:
        return np.array([], dtype=np.float64), hop
    frames = samples[:usable].reshape(-1, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-20)
    return rms, hop


def _noise_floor(rms: np.ndarray) -> float:
    if rms.size == 0:
        return ABS_NOISE_FLOOR
    return float(max(np.percentile(rms, NOISE_PERCENTILE), ABS_NOISE_FLOOR))


def _span_with_guard(
    start: int, end: int, length: int, rate: int
) -> tuple[int, int]:
    guard = int(round(GUARD_S * rate))
    return max(0, start - guard), min(length, end + guard)


def _continuous_span(samples: np.ndarray, rate: int) -> SoundEvent:
    rms, hop = _rms_frames(samples, rate)
    if rms.size == 0:
        raise SoundEventError("clip is too short to measure")
    peak = float(rms.max())
    if peak <= ABS_NOISE_FLOOR:
        raise SoundEventError("clip is digital silence")
    floor = max(peak * 10 ** (-40.0 / 20.0), ABS_NOISE_FLOOR)
    loud = np.flatnonzero(rms > floor)
    if loud.size == 0:
        raise SoundEventError("clip has no sounding frames")
    start, end = _span_with_guard(
        int(loud[0]) * hop, (int(loud[-1]) + 1) * hop, len(samples), rate
    )
    if end - start < int(round(MIN_EVENT_S * rate)):
        raise SoundEventError("sounding span is shorter than the minimum event")
    return SoundEvent(start, end, "continuous")


def _pulse_spans(samples: np.ndarray, rate: int) -> list[SoundEvent]:
    rms, hop = _rms_frames(samples, rate)
    if rms.size == 0:
        raise SoundEventError("clip is too short to measure")
    peak = float(rms.max())
    if peak <= ABS_NOISE_FLOOR:
        raise SoundEventError("clip is digital silence")

    noise = _noise_floor(rms)
    enter = noise * 10 ** (ENTER_DB_ABOVE_NOISE / 20.0)
    exit_ = noise * 10 ** (EXIT_DB_ABOVE_NOISE / 20.0)
    # A file that is almost all sound has a high noise estimate; the
    # gate would never open. Fall back to one trimmed span.
    if enter >= peak * 0.9:
        return [_continuous_span(samples, rate)]

    hangover_frames = max(1, int(round(HANGOVER_S / FRAME_S)))
    min_frames = max(1, int(round(MIN_EVENT_S / FRAME_S)))
    max_frames = max(min_frames, int(round(MAX_EVENT_S / FRAME_S)))
    merge_gap_frames = max(1, int(round(MERGE_GAP_S / FRAME_S)))

    raw: list[tuple[int, int]] = []
    i = 0
    n = int(rms.size)
    while i < n:
        if rms[i] < enter:
            i += 1
            continue
        start = i
        last_loud = i
        i += 1
        while i < n:
            if rms[i] >= exit_:
                last_loud = i
                i += 1
                continue
            quiet = 0
            while i < n and rms[i] < exit_ and quiet < hangover_frames:
                quiet += 1
                i += 1
            if quiet >= hangover_frames or i >= n:
                break
        end = last_loud + 1
        if end - start >= min_frames:
            capped = min(end, start + max_frames)
            raw.append((start, capped, capped < end, end))

    if not raw:
        return [_continuous_span(samples, rate)]

    merged: list[tuple[int, int, bool, int]] = [raw[0]]
    for start, end, truncated, uncapped in raw[1:]:
        prev_s, prev_e, prev_t, prev_u = merged[-1]
        if start - prev_e <= merge_gap_frames:
            merged[-1] = (
                prev_s,
                max(prev_e, end),
                prev_t or truncated,
                max(prev_u, uncapped),
            )
        else:
            merged.append((start, end, truncated, uncapped))

    events: list[SoundEvent] = []
    for start_f, end_f, truncated, uncapped_f in merged:
        start, end = _span_with_guard(
            start_f * hop, end_f * hop, len(samples), rate
        )
        _, untruncated_end = _span_with_guard(
            start_f * hop, uncapped_f * hop, len(samples), rate
        )
        if end - start < int(round(MIN_EVENT_S * rate)):
            continue
        events.append(SoundEvent(
            start, end, "pulse",
            truncated=truncated,
            untruncated_end_sample_exclusive=(
                untruncated_end if truncated else None),
        ))
    if not events:
        raise SoundEventError("no pulse survived the minimum duration")
    return events


def extract_sound_events(
    samples: np.ndarray,
    rate: int,
    *,
    event_class: str | None = None,
    policy: Policy | None = None,
) -> list[SoundEvent]:
    """Return sounding events for one mono clip."""

    if samples.ndim != 1:
        raise SoundEventError("samples must be a mono vector")
    if rate <= 0:
        raise SoundEventError("sample rate must be positive")
    chosen = policy or event_policy_for_class(event_class)
    if chosen == "continuous":
        return [_continuous_span(samples, rate)]
    return _pulse_spans(samples, rate)


def slice_event(samples: np.ndarray, event: SoundEvent) -> np.ndarray:
    return np.asarray(samples[event.start_sample : event.end_sample_exclusive])
