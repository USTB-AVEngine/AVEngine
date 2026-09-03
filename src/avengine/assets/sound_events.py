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
# Per-family split parameters.  A single global pair was wrong: 2026-09-03
# measurement showed every doorbell event landing at 1.53-1.56 s, which is the
# old 1.5 s cap plus the guard, so real single rings were being cut off; and one
# chime file produced eleven 0.2-0.5 s fragments because a 60 ms merge gap
# cannot hold a multi-tone ring together.
HANGOVER_S = 0.040
ENTER_DB_ABOVE_NOISE = 12.0
EXIT_DB_ABOVE_NOISE = 6.0
NOISE_PERCENTILE = 20.0
ABS_NOISE_FLOOR = 1e-5

# What counts as one occurrence decides whether a clip may be cut, not whether
# the waveform looks pulsatile.  owner 2026-09-03: "人声可千万不能和狗一样切掉
# 中间的声音" -- and the same holds for a ringtone (a melody), a DTMF dial (a
# digit sequence) and an alarm (one sustained ringing episode).  Cutting the
# middle out of any of those destroys the occurrence exactly as it would a
# sentence.  An unknown class therefore falls through to "continuous": never
# cut the middle of something whose occurrence you have not characterised.
#
# ATOMIC: one occurrence is a single short burst, self-contained and countable.
ATOMIC_PULSE_CLASSES = frozenset({"dog_bark", "cat_meow"})
# GROUPED: one occurrence is a short run of tones that must stay whole, but
# separate occurrences are separable.  A doorbell press is one occurrence even
# though it is two tones.
GROUPED_PULSE_CLASSES = frozenset(
    {"doorbell", "doorbell_chime", "ding_dong", "chime"}
)
PULSE_CLASSES = ATOMIC_PULSE_CLASSES | GROUPED_PULSE_CLASSES

# Classes whose occurrence is a sustained episode.  Listed rather than inferred
# so the reason survives: one beep of a smoke alarm is not "one alarm", and one
# tone of a busy signal is not "one busy signal".  The 2026-09-03 split put all
# fifty hysteresis-fallback spans in exactly these classes, which is the gate
# reporting that these files are continuous rather than a train of bursts.
SUSTAINED_ALERT_CLASSES = frozenset(
    {
        "alarm_bell", "alarm_beep", "alarm_clock", "buzzer", "ringtone",
        "phone_ring", "telephone", "telephone_bell_ringing",
        "telephone_dialing_dtmf", "busy_signal", "microwave_beep",
        "smoke_alarm", "fire_alarm", "cellphone_vibration_alert",
    }
)

_SPLIT_PARAMS = {
    "atomic": {"merge_gap_s": 0.060, "max_event_s": 2.000},
    # 0.8 s holds a ding and its dong together; 4 s lets a full chime ring out.
    "grouped": {"merge_gap_s": 0.800, "max_event_s": 4.000},
}


def split_family(event_class: str | None) -> str:
    """atomic | grouped | sustained — what one occurrence of this class is."""

    name = _normalised(event_class)
    if name in ATOMIC_PULSE_CLASSES:
        return "atomic"
    if name in GROUPED_PULSE_CLASSES:
        return "grouped"
    return "sustained"


def split_params_for_class(event_class: str | None) -> dict:
    family = split_family(event_class)
    if family == "sustained":
        raise SoundEventError(
            f"class {event_class!r} is a sustained occurrence; it is trimmed, "
            "never split, so it has no burst-splitting parameters")
    return dict(_SPLIT_PARAMS[family])


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


def _normalised(event_class: str | None) -> str:
    return (event_class or "").strip().lower().replace("-", "_").replace(" ", "_")


def event_policy_for_class(event_class: str | None) -> Policy:
    """Pulse classes split into occurrences; everything else is trimmed whole.

    There is no token heuristic any more.  The old one matched "ring", "bell",
    "beep" and "alarm", which is how ringtone, phone_ring, alarm_* and buzzer
    came to be split into fragments.  Membership is now declared, and an
    unrecognised class is treated as continuous.
    """

    return "pulse" if _normalised(event_class) in PULSE_CLASSES else "continuous"


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


def _pulse_spans(samples: np.ndarray, rate: int, *,
                 merge_gap_s: float, max_event_s: float) -> list[SoundEvent]:
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
    max_frames = max(min_frames, int(round(max_event_s / FRAME_S)))
    merge_gap_frames = max(1, int(round(merge_gap_s / FRAME_S)))

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
    return _pulse_spans(samples, rate, **split_params_for_class(event_class))


def slice_event(samples: np.ndarray, event: SoundEvent) -> np.ndarray:
    return np.asarray(samples[event.start_sample : event.end_sample_exclusive])
