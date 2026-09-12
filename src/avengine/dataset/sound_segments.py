"""Choose the part of a long recording that is actually sounding, and cut it.

The owner authorised this on 2026-09-10: a ten second Episode may use a
piece of a longer original recording, provided the piece that survives is
genuinely sounding.  That authorisation replaces the older "never cut the
original recording" rule.  It does not authorise the three cheap ways of
pretending, so this module refuses all of them:

* **A non-zero sample is not sound.**  Every decision here is made on
  window RMS, never on ``samples != 0``.
* **A whole-clip peak is not sound either.**  One loud doorbell press in
  eight seconds of digital silence has an excellent peak and 0.8 s of
  sound; the peak says nothing about the other 7.2 s.
* **An event bounding box is not sound.**  ``extract_sound_events`` adds a
  30 ms guard, merges pulses across a gap and caps a burst at 2-4 s, so
  its span deliberately covers material that is not sounding and
  deliberately stops before material that is.  Its islands are quoted here
  only as a cross-reference, never as the activity proof and never as a QA
  event count.

What replaces them is a per-window measurement plus four numbers that are
recorded for every segment: activity coverage, leading and trailing
inactive time, and the longest silence inside the segment.  Those, with
the per-class threshold that produced them, are the evidence that "what is
left is sounding".

Two class facts drive everything, and neither is inferred from the
waveform:

* An air conditioner is a noise source.  Gating it against its own noise
  floor - the standard speech VAD move - deletes it, because for a steady
  hum the noise floor *is* the signal.  Continuous device classes are
  therefore gated relative to the clip peak and an absolute floor, and
  come out with coverage near 1.0, which is correct.
* Unvoiced consonants are quiet and aperiodic.  Nothing here tests for
  periodicity, and nothing cuts the inside of a selected region: a
  segment is one contiguous span, so an /s/ that dips below the gate for
  60 ms stays in the audio and is only counted as a short internal
  silence.

The other half of the job is arithmetic on time.  A caller - the ten
second Episode planner - has a budget.  This module returns the longest
contiguous region that fits the budget, starts and ends on sound, and does
not contain a silence longer than the class allows.  It never stitches two
separated bursts into a fake continuous sound, never time-stretches, and
never levels loudness sample by sample.

Activity is judged on the band the dataset actually ships.  The recording
is resampled to the delivery rate first, then measured, then sliced - not
the other way round.  One alarm clock here passed a 44.1 kHz gate and had
no window left above that gate once the resampler removed the energy above
8 kHz that had carried it; the plan promised sound the delivered file did
not have.  See ``analysis_signal``.

Selection and materialisation are separate on purpose.  ``plan_segment``
reads the original PCM read-only and returns coordinates; ``materialize_segment``
writes PCM for one chosen plan.  An upstream sampler can index a whole
library and then cut only the handful of segments an Episode actually
uses, instead of pre-rendering every window it might have wanted.

Nothing here certifies that a human listened.  Every record carries
``acoustic_detector_candidate_not_human_certified``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy.signal import resample_poly

from avengine.assets.sound_events import (
    ABS_NOISE_FLOOR,
    ENTER_DB_ABOVE_NOISE,
    EXIT_DB_ABOVE_NOISE,
    HANGOVER_S,
    MIN_EVENT_S,
    NOISE_PERCENTILE,
    SoundEventError,
    extract_sound_events,
)
from avengine.assets.sound_prepare import (
    SPEECH_DETECTOR_DEFAULTS,
    SPEECH_FILTER_DEFAULTS,
    SPEECH_MIN_AUDIBLE_S,
    TARGET_RATE_HZ,
    activity_profile_for_class,
    make_prepared_audio_id,
)
# Reuse of the module that already owns clip preparation, rather than a second
# copy of WAV writing, mono coercion and zero-phase filtering.
from avengine.assets.sound_prepare import (
    as_mono_float,
    write_wav_mono_no_clobber,
    zero_phase_filter,
)
from avengine.assets.sound_qc import SoundQCError, read_mono

SEGMENT_SCHEMA = "avengine_sound_segment_v1"
PLAN_SCHEMA = "avengine_sound_segment_plan_v1"
INDEX_SCHEMA = "avengine_sound_segment_index_v1"
ACTIVITY_SCHEMA = "avengine_sound_segment_activity_v1"
VERIFICATION_SCHEMA = "avengine_sound_segment_verification_v1"

OPERATION = "activity_selected_contiguous_segment_v1"
CROP_AUTHORIZATION = "owner_authorized_activity_segment_selection_20260910"
CERTIFICATION_STATUS = "acoustic_detector_candidate_not_human_certified"

# One analysis grid for every class, so coverage numbers from a bark, a hum
# and a sentence are comparable.  The values are the ones the existing speech
# detector already uses (sound_prepare.SPEECH_DETECTOR_DEFAULTS).
WINDOW_S = float(SPEECH_DETECTOR_DEFAULTS["window_s"])
HOP_S = float(SPEECH_DETECTOR_DEFAULTS["hop_s"])

# A gap this short is a dip inside one sound, not a silence.  Taken from the
# hangover the existing burst gate already uses for the same purpose, and it is
# also what keeps a zero crossing or a single quiet window from being reported
# as internal silence.
SILENCE_RUN_MERGE_S = float(HANGOVER_S)

# Below this a window is not sound at any relative threshold.  sound_qc warns
# at a whole-clip RMS of -45 dBFS ("raising this raises the noise floor with
# it") and fails a clip whose peak is under 1e-4 (-80 dBFS).  -60 dBFS on a
# 20 ms window sits between the two: it never rejects material sound_qc would
# merely warn about, and it does reject a recording that is only room tone at
# an inaudible level.
ABSOLUTE_ACTIVE_FLOOR_DBFS = -60.0
# Mirrors sound_qc's calibrated quiet-clip warning.  A segment below it is
# reported qualified, not rejected - the same severity sound_qc assigns.
SEGMENT_QUIET_RMS_DBFS = -45.0

# Anti-click only.  5 ms of raised cosine at each edge, applied inside the
# 30 ms crop guard so it never touches a sounding sample.  Fixed, recorded on
# every segment, and switchable off with edge_fade_s=0.
# Budget fields that can change the samples written, once the crop bounds are
# fixed.  Every one of them goes into the segment id.  The 2026-09-10 defect
# was exactly this list being empty: asking for a 5 ms fade returned the
# no-fade file already on disk, because nothing in the name depended on the
# fade.  ``test_every_budget_field_is_classified`` fails if a field is added to
# SegmentBudget and not listed in one of these two tuples.
PCM_DETERMINING_BUDGET_FIELDS = (
    "target_rate_hz",
    "remove_dc",
    "edge_fade_s",
    "normalize_peak",
    "target_peak_dbfs",
    "allow_resample_headroom",
)
# Budget fields that choose *which* region is cut.  They do not need to be in
# the id separately: the region they chose is in it, as the crop bounds.
BOUND_DETERMINING_BUDGET_FIELDS = (
    "max_duration_s",
    "min_duration_s",
    "crop_guard_s",
    "selection",
    "temporal_policy",
)

DEFAULT_EDGE_FADE_S = 0.005
DEFAULT_CROP_GUARD_S = 0.030
DEFAULT_MAX_EDGE_SILENCE_S = 0.100

# Per-family selection policy.
#
# gate:
#   peak_relative        - threshold is a fixed number of dB under the clip's
#                          own peak window.  The only correct choice for a
#                          sound that never pauses.
#   noise_floor_hysteresis - enter/exit thresholds above the clip's 20th
#                          percentile window level, with the same fallback to
#                          peak_relative the burst detector already uses when
#                          the clip is almost all sound.  For sounds that do
#                          pause between occurrences.
#
# min_active_duration_s is taken from the existing activity profile for the
# class where that profile states one; MIN_EVENT_S is the fallback, because it
# is already the project's answer to "how short is too short to be an event".
#
# min_activity_coverage and max_internal_silence_s are new numbers introduced
# by this task, listed here with the reason each was chosen and overridable per
# call.  They are policy, not measurement.
_FAMILY_POLICY: dict[str, dict[str, Any]] = {
    "speech": {
        "band": "speech_band",
        "gate": "peak_relative",
        "relative_peak_db": float(SPEECH_DETECTOR_DEFAULTS["relative_peak_db"]),
        # 1.5 s of speech in at most 5 s is what the existing detector already
        # accepts by default, so 0.30 is that same rule expressed as coverage
        # rather than a new, stricter one.
        "min_activity_coverage": 0.30,
        # A pause longer than this is a gap between two utterances, not a
        # breath inside one.
        "max_internal_silence_s": 0.60,
        "min_active_duration_s": float(SPEECH_MIN_AUDIBLE_S),
    },
    "animal_call": {
        "band": "full_band",
        "gate": "noise_floor_hysteresis",
        "relative_peak_db": -25.0,
        # Two barks with a pause between them are one usable segment; demanding
        # half of it be sound would reject the normal case.
        "min_activity_coverage": 0.25,
        "max_internal_silence_s": 0.50,
        "min_active_duration_s": None,
    },
    "short_prompt": {
        "band": "full_band",
        "gate": "noise_floor_hysteresis",
        "relative_peak_db": -25.0,
        "min_activity_coverage": 0.25,
        # A doorbell's ding and dong, and consecutive DTMF digits, sit well
        # inside half a second; a longer gap separates two presses.
        "max_internal_silence_s": 0.50,
        "min_active_duration_s": None,
    },
    "vocal_non_speech": {
        "band": "full_band",
        "gate": "noise_floor_hysteresis",
        "relative_peak_db": -25.0,
        "min_activity_coverage": 0.25,
        "max_internal_silence_s": 0.50,
        "min_active_duration_s": None,
    },
    "device_continuous": {
        "band": "full_band",
        # Never noise-floor gated: for a steady hum the noise floor is the
        # signal, and this gate is what stops an air conditioner being deleted
        # for sounding like noise.
        "gate": "peak_relative",
        "relative_peak_db": float(SPEECH_DETECTOR_DEFAULTS["relative_peak_db"]),
        # A continuous device that is quiet for a seventh of the window is not
        # running continuously.
        "min_activity_coverage": 0.85,
        "max_internal_silence_s": 0.25,
        "min_active_duration_s": None,
    },
    "unknown": {
        "band": "full_band",
        # An unrecognised class falls through to the never-cut-the-middle
        # treatment, exactly as sound_events does for an unrecognised class.
        "gate": "peak_relative",
        "relative_peak_db": float(SPEECH_DETECTOR_DEFAULTS["relative_peak_db"]),
        "min_activity_coverage": 0.50,
        "max_internal_silence_s": 0.30,
        "min_active_duration_s": None,
    },
}

# Duration and coverage thresholds that match a *measured* shape, as opposed
# to the shape a class is assumed to have.  Only these two numbers; the band,
# the gate and the semantic family stay with the declared class, because the
# form was measured through that gate and re-picking the gate from the result
# would be circular.
#
# Off by default.  Turning it on raises how many recordings qualify, and a
# mechanism that raises pass rates must be asked for explicitly and recorded
# when used, never applied quietly.  The 2026-09-10 measurement is the reason
# it exists: in clock_tick, crackle, drip, fire and gurgling every rejection
# was a pulse-like recording and every continuous-like recording already
# passed, so the mismatch is per recording and renaming the five classes
# would have re-aimed 27 recordings that were already fine.
TEMPORAL_FORM_POLICY: dict[str, dict[str, float]] = {
    "continuous": {"min_activity_coverage": 0.85, "max_internal_silence_s": 0.25},
    "intermittent_continuous": {
        "min_activity_coverage": 0.70, "max_internal_silence_s": 0.40},
    "periodic_pulse_train": {
        "min_activity_coverage": 0.25, "max_internal_silence_s": 0.60},
    "sparse_bursts": {
        "min_activity_coverage": 0.25, "max_internal_silence_s": 0.50},
    "single_burst": {
        "min_activity_coverage": 0.25, "max_internal_silence_s": 0.50},
}

TEMPORAL_POLICY_MODES = frozenset({"declared", "measured"})

_POLICY_KEYS = frozenset(
    {
        "band",
        "gate",
        "relative_peak_db",
        "enter_db_above_noise",
        "exit_db_above_noise",
        "noise_percentile",
        "absolute_active_floor_dbfs",
        "window_s",
        "hop_s",
        "silence_run_merge_s",
        "min_activity_coverage",
        "max_internal_silence_s",
        "min_active_duration_s",
        "max_edge_silence_s",
    }
)


class SoundSegmentError(ValueError):
    """A recording cannot yield a segment under the requested policy."""


def _db(value: float) -> float:
    return float(20.0 * math.log10(max(float(value), 1e-12)))


def _amp(dbfs: float) -> float:
    return float(10.0 ** (float(dbfs) / 20.0))


def segment_policy_for_class(
    sound_class: str | None,
    *,
    overrides: Mapping[str, Any] | None = None,
    temporal_form: str | None = None,
) -> dict[str, Any]:
    """Detector and selection policy for one sound class.

    The family comes from the existing ``activity_profile_for_class``; this
    module does not keep a second class list.  An unmapped class resolves to
    ``unknown`` and is treated as never-cut-the-middle, which is why a caller
    that knows better can override the family explicitly instead of a class
    list growing here.
    """

    profile = activity_profile_for_class(sound_class)
    family = str(profile.get("activity_family") or "unknown")
    overrides = dict(overrides or {})
    forced_family = overrides.pop("activity_family", None)
    if forced_family is not None:
        family = str(forced_family)
    if family not in _FAMILY_POLICY:
        raise SoundSegmentError(
            f"no segment policy for activity family {family!r}; known families "
            f"are {sorted(_FAMILY_POLICY)}"
        )
    unknown = set(overrides) - _POLICY_KEYS
    if unknown:
        raise SoundSegmentError(f"unknown policy override(s): {sorted(unknown)}")

    policy: dict[str, Any] = {
        "schema": ACTIVITY_SCHEMA,
        "sound_class": str(sound_class or ""),
        "activity_family": family,
        "activity_family_source": (
            "explicit_override" if forced_family is not None
            else "avengine.assets.sound_prepare.activity_profile_for_class"
        ),
        "window_s": WINDOW_S,
        "hop_s": HOP_S,
        "level": "frame_rms",
        "silence_run_merge_s": SILENCE_RUN_MERGE_S,
        "enter_db_above_noise": float(ENTER_DB_ABOVE_NOISE),
        "exit_db_above_noise": float(EXIT_DB_ABOVE_NOISE),
        "noise_percentile": float(NOISE_PERCENTILE),
        "absolute_active_floor_dbfs": ABSOLUTE_ACTIVE_FLOOR_DBFS,
        "max_edge_silence_s": DEFAULT_MAX_EDGE_SILENCE_S,
        "calibration": "placeholder",
        "status": CERTIFICATION_STATUS,
    }
    policy.update(_FAMILY_POLICY[family])
    if policy.get("min_active_duration_s") is None:
        declared = profile.get("minimum_audible_duration_s")
        policy["min_active_duration_s"] = (
            float(declared) if declared is not None else float(MIN_EVENT_S)
        )
        policy["min_active_duration_source"] = (
            "activity_profile_minimum_audible_duration_s" if declared is not None
            else "avengine.assets.sound_events.MIN_EVENT_S"
        )
    else:
        policy["min_active_duration_source"] = "family_policy"
    if policy["band"] == "speech_band":
        policy["filter"] = dict(SPEECH_FILTER_DEFAULTS)
    else:
        policy["filter"] = {"mode": "full_band", "highpass_hz": None,
                            "bandpass_hz": None}
    policy["temporal_form_applied"] = None
    if temporal_form is not None:
        adjusted = TEMPORAL_FORM_POLICY.get(str(temporal_form))
        if adjusted is None:
            raise SoundSegmentError(
                f"no temporal policy for measured form {temporal_form!r}; "
                f"known: {sorted(TEMPORAL_FORM_POLICY)}"
            )
        policy["declared_thresholds"] = {
            "min_activity_coverage": policy["min_activity_coverage"],
            "max_internal_silence_s": policy["max_internal_silence_s"],
        }
        policy.update(adjusted)
        policy["temporal_form_applied"] = str(temporal_form)
        policy["temporal_policy_note"] = (
            "coverage and internal-silence thresholds come from the measured "
            "temporal form; band, gate and semantic family stay with the "
            "declared class"
        )
    policy.update(overrides)
    for key in ("min_activity_coverage", "max_internal_silence_s",
                "min_active_duration_s", "max_edge_silence_s",
                "relative_peak_db", "absolute_active_floor_dbfs",
                "window_s", "hop_s", "silence_run_merge_s"):
        policy[key] = float(policy[key])
    if not 0.0 <= policy["min_activity_coverage"] <= 1.0:
        raise SoundSegmentError("min_activity_coverage must be within [0, 1]")
    if policy["window_s"] <= 0 or policy["hop_s"] <= 0:
        raise SoundSegmentError("window_s and hop_s must be positive")
    return policy


def _frame_levels(
    samples: np.ndarray, rate: int, *, window_s: float, hop_s: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Window RMS on a sliding grid, plus each window's start sample."""

    window = max(1, int(round(window_s * rate)))
    hop = max(1, int(round(hop_s * rate)))
    if samples.size < window:
        raise SoundSegmentError(
            f"audio is {samples.size} samples, shorter than one "
            f"{window}-sample detector window"
        )
    starts = np.arange(0, samples.size - window + 1, hop, dtype=np.int64)
    cumulative = np.concatenate(
        [np.array([0.0]), np.cumsum(samples.astype(np.float64) ** 2)]
    )
    levels = np.sqrt(
        np.maximum(0.0, (cumulative[starts + window] - cumulative[starts]) / window)
    )
    return levels, starts, window, hop


def _runs_to_intervals(
    active: np.ndarray, starts: np.ndarray, window: int, length: int
) -> list[list[int]]:
    if not active.any():
        return []
    ids = np.flatnonzero(active)
    cuts = np.r_[0, np.flatnonzero(np.diff(ids) > 1) + 1, len(ids)]
    intervals: list[list[int]] = []
    for index in range(len(cuts) - 1):
        first = int(starts[ids[cuts[index]]])
        last = min(length, int(starts[ids[cuts[index + 1] - 1]]) + window)
        intervals.append([first, last])
    return intervals


def _merge_short_gaps(
    intervals: Sequence[Sequence[int]], merge_samples: int
) -> list[list[int]]:
    """Absorb dips shorter than one merge window into the sound around them.

    A zero crossing, a stop consonant and the trough of a slow tremolo all
    look like silence for a few milliseconds.  Treating them as silence would
    both understate coverage and invent internal silences that no listener
    would hear.
    """

    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= merge_samples:
            merged[-1][1] = max(merged[-1][1], int(end))
        else:
            merged.append([int(start), int(end)])
    return merged


def _intersect(
    intervals: Sequence[Sequence[int]], start: int, end: int
) -> list[list[int]]:
    out: list[list[int]] = []
    for a, b in intervals:
        lo, hi = max(int(a), start), min(int(b), end)
        if hi > lo:
            out.append([lo, hi])
    return out


def _span_stats(
    intervals: Sequence[Sequence[int]], start: int, end: int, rate: int
) -> dict[str, Any]:
    """Coverage, edge silence and longest internal silence for one span."""

    inside = _intersect(intervals, start, end)
    total = max(0, end - start)
    active = sum(b - a for a, b in inside)
    if not inside:
        return {
            "duration_s": total / rate,
            "active_duration_s": 0.0,
            "activity_coverage": 0.0,
            "activity_interval_count": 0,
            "leading_inactive_s": total / rate,
            "trailing_inactive_s": total / rate,
            "max_internal_silence_s": 0.0,
            "activity_intervals_samples": [],
        }
    gaps = [inside[i + 1][0] - inside[i][1] for i in range(len(inside) - 1)]
    return {
        "duration_s": total / rate,
        "active_duration_s": active / rate,
        "activity_coverage": (active / total) if total else 0.0,
        "activity_interval_count": len(inside),
        "leading_inactive_s": (inside[0][0] - start) / rate,
        "trailing_inactive_s": (end - inside[-1][1]) / rate,
        "max_internal_silence_s": (max(gaps) / rate) if gaps else 0.0,
        "activity_intervals_samples": [[a - start, b - start] for a, b in inside],
    }


def pause_structure(
    intervals: Sequence[Sequence[int]], start: int, end: int, rate: int
) -> dict[str, Any]:
    """The shape of the sound and the silence, not just the ratio between them.

    Coverage on its own decides nothing.  A dialled telephone number is 40-49%
    sounding because a dialled number *has* gaps between digits, and the owner
    has said to keep that rhythm; a recording that is 45% sounding because it
    trails off into two seconds of nothing is a different object with the same
    coverage.  What separates them is here: how long each sounding stretch is,
    how long each pause is, whether the pauses are regular, and where the
    silence sits.
    """

    inside = _intersect(intervals, start, end)
    if not inside:
        return {
            "sounding_interval_count": 0,
            "sounding_durations_s": [],
            "pause_durations_s": [],
            "temporal_form": "silent",
            "coverage_alone_is_not_a_verdict": True,
        }
    sounding = [(b - a) / rate for a, b in inside]
    pauses = [(inside[i + 1][0] - inside[i][1]) / rate
              for i in range(len(inside) - 1)]
    onsets = [a / rate for a, _b in inside]
    periods = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    total = max(1, end - start)
    coverage = sum(b - a for a, b in inside) / total

    def _cv(values: Sequence[float]) -> float | None:
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        if mean <= 0:
            return None
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return float(math.sqrt(variance) / mean)

    period_cv = _cv(periods)
    # A train is regular; "regular" here means the onset-to-onset spacing
    # varies by less than a quarter of its own mean.  Three onsets is the
    # fewest that can show a spacing is repeated rather than coincidental.
    regular = period_cv is not None and len(periods) >= 2 and period_cv < 0.25
    if len(inside) == 1:
        form = "continuous" if coverage >= 0.95 else "single_burst"
    elif regular:
        form = "periodic_pulse_train"
    elif coverage >= 0.70:
        form = "intermittent_continuous"
    else:
        form = "sparse_bursts"
    return {
        "sounding_interval_count": len(inside),
        "sounding_durations_s": [round(v, 4) for v in sounding],
        "sounding_duration_median_s": float(sorted(sounding)[len(sounding) // 2]),
        "sounding_duration_min_s": float(min(sounding)),
        "sounding_duration_max_s": float(max(sounding)),
        "pause_count": len(pauses),
        "pause_durations_s": [round(v, 4) for v in pauses],
        "pause_duration_median_s": (
            float(sorted(pauses)[len(pauses) // 2]) if pauses else None),
        "pause_duration_max_s": float(max(pauses)) if pauses else None,
        "onset_period_s": [round(v, 4) for v in periods],
        "onset_period_coefficient_of_variation": period_cv,
        "onset_spacing_is_regular": bool(regular),
        "temporal_form": form,
        "temporal_form_is_measured_not_declared": True,
        "coverage_alone_is_not_a_verdict": True,
    }


def measure_activity(
    samples: np.ndarray | Sequence[float],
    rate: int,
    *,
    sound_class: str | None = None,
    policy: Mapping[str, Any] | None = None,
    threshold_frame_rms: float | tuple[float, float] | None = None,
    include_frame_levels: bool = False,
) -> dict[str, Any]:
    """Per-window activity for one mono recording, under one class policy.

    ``threshold_frame_rms`` replays an already-derived gate instead of deriving
    a new one: pass one level for a plain threshold, or ``(enter, exit)`` to
    reproduce a hysteresis gate exactly.  Read-back verification uses it,
    because a threshold re-derived from the cut piece is a *different*
    definition of sounding - the piece has a different peak and a different
    noise percentile than the recording it came from - and a segment must be
    checked against the rule that selected it.
    """

    active_policy = dict(policy) if policy is not None else segment_policy_for_class(
        sound_class
    )
    x = as_mono_float(samples)
    rate = int(rate)
    if rate <= 0:
        raise SoundSegmentError("sample rate must be positive")
    if active_policy["band"] == "speech_band":
        band_filter = active_policy["filter"]
        analysed = zero_phase_filter(
            x, rate, kind="highpass",
            cutoff=float(band_filter["highpass_hz"]),
            order=int(band_filter["order"]),
        )
        analysed = zero_phase_filter(
            analysed, rate, kind="bandpass",
            cutoff=(float(band_filter["bandpass_low_hz"]),
                    float(band_filter["bandpass_high_hz"])),
            order=int(band_filter["order"]),
        )
    else:
        analysed = x

    levels, starts, window, hop = _frame_levels(
        analysed, rate,
        window_s=active_policy["window_s"], hop_s=active_policy["hop_s"],
    )
    peak = float(levels.max()) if levels.size else 0.0
    noise = float(max(np.percentile(levels, active_policy["noise_percentile"]),
                      ABS_NOISE_FLOOR)) if levels.size else ABS_NOISE_FLOOR
    absolute_floor = _amp(active_policy["absolute_active_floor_dbfs"])
    gate_mode = str(active_policy["gate"])
    relative = peak * _amp(active_policy["relative_peak_db"])
    noise_enter = noise * _amp(active_policy["enter_db_above_noise"])
    noise_exit = noise * _amp(active_policy["exit_db_above_noise"])
    # No gate is ever allowed below the peak-relative floor.  Without it a clip
    # whose gaps are digital silence pushes the noise estimate to the clamp and
    # the gate down to the absolute floor, at which point the decay tail of a
    # bark - fifty decibels under the bark - counts as sounding and coverage
    # reads 0.97 for material that is mostly ring-out.  The floor is the same
    # "25 dB under the loudest window" rule the speech detector already uses,
    # which also makes coverage comparable between a bark, a hum and a
    # sentence instead of meaning something different in each.
    gate_floor = max(relative, absolute_floor)
    gate_fallback = None

    def _hysteresis(enter_level: float, exit_level: float) -> np.ndarray:
        flags = np.zeros(levels.shape, dtype=bool)
        open_gate = False
        for index in range(levels.size):
            if open_gate:
                open_gate = levels[index] >= exit_level
            else:
                open_gate = levels[index] >= enter_level
            flags[index] = open_gate
        return flags

    if threshold_frame_rms is not None:
        gate_mode = "replayed_gate"
        if isinstance(threshold_frame_rms, (tuple, list)):
            enter, exit_ = (float(value) for value in threshold_frame_rms)
        else:
            enter = exit_ = float(threshold_frame_rms)
        threshold = enter
        active = _hysteresis(enter, exit_) if exit_ < enter else levels >= enter
    elif gate_mode == "peak_relative":
        threshold = enter = exit_ = gate_floor
        active = levels >= threshold
    elif gate_mode == "noise_floor_hysteresis":
        # The same guard the burst detector uses: a clip that is almost all
        # sound has a noise estimate close to its peak, so the gate would
        # never open.  Fall back to the relative gate and say so.
        if noise_enter >= peak * 0.9:
            gate_fallback = "clip_is_almost_all_sound_relative_gate_used"
            gate_mode = "peak_relative"
            threshold = enter = exit_ = gate_floor
            active = levels >= threshold
        else:
            enter = max(noise_enter, gate_floor)
            exit_ = max(noise_exit, gate_floor)
            threshold = enter
            active = _hysteresis(enter, exit_)
    else:
        raise SoundSegmentError(f"unknown gate {gate_mode!r}")

    raw_intervals = _runs_to_intervals(active, starts, window, int(x.size))
    merge_samples = int(round(active_policy["silence_run_merge_s"] * rate))
    intervals = _merge_short_gaps(raw_intervals, merge_samples)
    stats = _span_stats(intervals, 0, int(x.size), rate)
    whole_rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0

    facts: dict[str, Any] = {
        "schema": ACTIVITY_SCHEMA,
        "sound_class": active_policy.get("sound_class"),
        "activity_family": active_policy["activity_family"],
        "source_rate_hz": rate,
        "source_sample_count": int(x.size),
        "source_duration_s": float(x.size / rate),
        "policy": active_policy,
        "gate": {
            "mode": gate_mode,
            "fallback": gate_fallback,
            "threshold_frame_rms": float(threshold),
            "threshold_dbfs": _db(threshold),
            "enter_frame_rms": float(enter),
            "exit_frame_rms": float(exit_),
            "peak_frame_rms": peak,
            "peak_frame_dbfs": _db(peak),
            "noise_floor_frame_rms": noise,
            "noise_floor_frame_dbfs": _db(noise),
            "absolute_floor_frame_rms": absolute_floor,
            "absolute_floor_dbfs": _db(absolute_floor),
            "peak_relative_floor_frame_rms": relative,
            "peak_relative_floor_dbfs": _db(relative),
            "noise_referenced_enter_frame_rms": noise_enter,
            "noise_referenced_exit_frame_rms": noise_exit,
            "gate_floor_frame_rms": gate_floor,
            "gate_floor_dbfs": _db(gate_floor),
            "gate_contrast_db": _db(peak) - _db(noise),
        },
        "frame": {
            "window_s": active_policy["window_s"],
            "hop_s": active_policy["hop_s"],
            "window_samples": window,
            "hop_samples": hop,
            "window_count": int(levels.size),
            "active_window_count": int(np.count_nonzero(active)),
            "active_window_ratio": (
                float(np.count_nonzero(active) / levels.size) if levels.size else 0.0
            ),
            "level": "frame_rms",
            "active_duration": "union_of_active_windows",
        },
        "frame_level_dbfs_percentiles": {
            str(int(p)): _db(float(np.percentile(levels, p)))
            for p in (5, 20, 50, 80, 95)
        } if levels.size else {},
        "source_activity_intervals_samples": [list(pair) for pair in intervals],
        "activity_intervals_before_gap_merge": [list(p) for p in raw_intervals],
        "activity_interval_count": stats["activity_interval_count"],
        "active_duration_s": stats["active_duration_s"],
        "activity_coverage": stats["activity_coverage"],
        "leading_inactive_s": stats["leading_inactive_s"],
        "trailing_inactive_s": stats["trailing_inactive_s"],
        "max_internal_silence_s": stats["max_internal_silence_s"],
        "structure": pause_structure(intervals, 0, int(x.size), rate),
        "audible_start_sample": intervals[0][0] if intervals else None,
        "audible_end_sample_exclusive": intervals[-1][1] if intervals else None,
        "whole_clip_rms_dbfs": _db(whole_rms),
        "whole_clip_peak_dbfs": _db(float(np.abs(x).max()) if x.size else 0.0),
        "measurement": "avengine_sound_segment_window_rms_activity_v1",
        "activity_is_qa_event_count": False,
        "activity_guard_included": False,
        "calibration": "placeholder",
        "status": CERTIFICATION_STATUS,
    }
    if include_frame_levels:
        facts["frame_level_dbfs"] = [_db(float(value)) for value in levels]
        facts["frame_active"] = [bool(value) for value in active]
        facts["frame_start_samples"] = [int(value) for value in starts]
    return facts


def cross_reference_sound_events(
    samples: np.ndarray | Sequence[float],
    rate: int,
    *,
    sound_class: str | None,
) -> dict[str, Any]:
    """Run the existing burst detector purely as a cross-reference.

    Its spans include a 30 ms guard and stop at a 2-4 s cap, so they are not
    the activity proof and its island count is not the QA event count.  It is
    recorded because a large disagreement between the two is worth seeing.
    """

    try:
        events = extract_sound_events(
            as_mono_float(samples), int(rate), event_class=sound_class
        )
    except (SoundEventError, ValueError) as error:
        return {
            "status": "not_run",
            "reason": f"{type(error).__name__}: {error}",
            "island_count": None,
            "is_activity_proof": False,
            "is_qa_event_count": False,
        }
    return {
        "status": "measured",
        "island_count": len(events),
        "detector_cap_truncated": any(event.truncated for event in events),
        "spans_samples": [
            [event.start_sample, event.end_sample_exclusive] for event in events
        ],
        "purpose": events[0].purpose if events else None,
        "guard_included": True,
        "is_activity_proof": False,
        "is_qa_event_count": False,
        "note": (
            "extract_sound_events extents carry a 30 ms guard and a per-family "
            "max_event_s cap; quoted for comparison only"
        ),
    }


@dataclass(frozen=True)
class SegmentBudget:
    """What the caller has room for, in its own clock.

    ``max_duration_s`` is the only required field: it is the caller's time
    budget, for example what is left of a ten second Episode after the start
    offset and the three second wet-tail reserve.  Nothing here knows about
    Episodes, tails or renderers - it returns the longest sounding region that
    fits the number it was given.
    """

    max_duration_s: float
    min_duration_s: float | None = None
    crop_guard_s: float = DEFAULT_CROP_GUARD_S
    edge_fade_s: float = DEFAULT_EDGE_FADE_S
    target_rate_hz: int | None = TARGET_RATE_HZ
    remove_dc: bool = False
    normalize_peak: bool = False
    target_peak_dbfs: float | None = None
    # A polyphase resampler overshoots: one library clip is peak-normalised to
    # exactly 1.0 and comes out of the 44.1 -> 16 kHz conversion at 1.26.
    # 16-bit PCM has no room for that, and clipping it would be distortion, so
    # by default the whole segment is attenuated by one constant factor, which
    # is recorded.  Recording the factor recovers the original *scale*; it does
    # not undo the 16-bit quantisation, so the stored audio is not a lossless
    # round trip and must not be described as one.  This is an overflow guard,
    # not per-clip loudness matching: it fires only when the peak would exceed
    # full scale, and a quiet clip and a loud one both keep gain 1.0.  Set
    # False to make the overshoot an error instead.
    allow_resample_headroom: bool = True
    selection: str = "longest_active"
    # "declared": the class's own coverage/silence thresholds.  "measured":
    # the thresholds matching the shape this recording actually has.  Default
    # declared, because the second one qualifies more recordings and that has
    # to be an explicit choice.
    temporal_policy: str = "declared"

    def validated(self) -> "SegmentBudget":
        if not math.isfinite(self.max_duration_s) or self.max_duration_s <= 0:
            raise SoundSegmentError("max_duration_s must be positive and finite")
        if self.min_duration_s is not None:
            if self.min_duration_s < 0:
                raise SoundSegmentError("min_duration_s must not be negative")
            if self.min_duration_s > self.max_duration_s:
                raise SoundSegmentError(
                    "min_duration_s must not exceed max_duration_s"
                )
        if self.crop_guard_s < 0 or self.edge_fade_s < 0:
            raise SoundSegmentError("crop guard and edge fade must not be negative")
        if self.edge_fade_s > self.crop_guard_s and self.edge_fade_s > 0:
            # The fade exists to stop a click at a cut that lands mid-waveform.
            # Longer than the guard it would start eating sounding samples,
            # which is loudness editing, not a cut.
            raise SoundSegmentError(
                "edge_fade_s must not exceed crop_guard_s; a longer fade would "
                "attenuate sounding samples"
            )
        if self.target_rate_hz is not None and self.target_rate_hz <= 0:
            raise SoundSegmentError("target_rate_hz must be positive or None")
        if self.normalize_peak and self.target_peak_dbfs is None:
            raise SoundSegmentError(
                "normalize_peak requires an explicit target_peak_dbfs"
            )
        if self.selection not in _SELECTIONS:
            raise SoundSegmentError(
                f"unknown selection {self.selection!r}; known: {sorted(_SELECTIONS)}"
            )
        if self.temporal_policy not in TEMPORAL_POLICY_MODES:
            raise SoundSegmentError(
                f"unknown temporal_policy {self.temporal_policy!r}; "
                f"known: {sorted(TEMPORAL_POLICY_MODES)}"
            )
        return self

    def processing_identity(self) -> dict[str, Any]:
        """The settings that change the written samples, for the segment id.

        Requested settings only.  A gain that is only known after the samples
        have been processed cannot take part in the name those samples are
        looked up by, which is how the peak target came to be invisible to the
        id even though ``normalization_applied`` was in it.
        """

        identity: dict[str, Any] = {}
        for name in PCM_DETERMINING_BUDGET_FIELDS:
            value = getattr(self, name)
            identity[name] = (
                None if value is None
                else bool(value) if isinstance(value, bool)
                else int(value) if isinstance(value, int)
                else float(value)
            )
        return identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_duration_s": float(self.max_duration_s),
            "min_duration_s": (
                None if self.min_duration_s is None else float(self.min_duration_s)
            ),
            "crop_guard_s": float(self.crop_guard_s),
            "edge_fade_s": float(self.edge_fade_s),
            "target_rate_hz": self.target_rate_hz,
            "remove_dc": bool(self.remove_dc),
            "normalize_peak": bool(self.normalize_peak),
            "target_peak_dbfs": self.target_peak_dbfs,
            "allow_resample_headroom": bool(self.allow_resample_headroom),
            "selection": self.selection,
            "temporal_policy": self.temporal_policy,
        }


# How the winner is chosen among the regions that satisfy every constraint.
# "longest_active" is the default because it is the least selective thing that
# still answers the caller's question: of the regions that fit, keep the one
# carrying the most sound.  It is deterministic, and every rejected candidate
# is written to the plan, so nothing is quietly discarded.
_SELECTIONS = frozenset({"longest_active", "earliest", "highest_coverage"})


def _rank_key(candidate: Mapping[str, Any], selection: str) -> tuple:
    if selection == "earliest":
        return (candidate["source_crop_start_sample"],)
    if selection == "highest_coverage":
        return (
            -candidate["activity_coverage"],
            -candidate["active_duration_s"],
            candidate["source_crop_start_sample"],
        )
    return (
        -candidate["active_duration_s"],
        -candidate["activity_coverage"],
        candidate["source_crop_start_sample"],
    )


def select_segment(
    activity: Mapping[str, Any],
    budget: SegmentBudget,
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick one contiguous region of the source that fits the budget.

    The region always starts at the beginning of a sounding interval and ends
    at the end of one, so no candidate ever begins or ends in silence beyond
    the crop guard.  Extending across a gap is allowed only while the gap is
    within the class's ``max_internal_silence_s``; a longer gap ends the
    region rather than being bridged, because bridging two separated bursts
    would manufacture a sound that was never recorded.

    A region is never cut in the middle of a sounding interval *unless* that
    single interval is by itself longer than the budget - a fifteen second
    hum with a five second budget has no other answer, and clamping a
    continuous sound is not the same as cutting an occurrence in half.
    """

    budget = budget.validated()
    active_policy = dict(policy or activity["policy"])
    rate = int(activity["source_rate_hz"])
    length = int(activity["source_sample_count"])
    intervals = [list(pair) for pair in activity["source_activity_intervals_samples"]]
    max_len = int(round(budget.max_duration_s * rate))
    min_len = (
        0 if budget.min_duration_s is None
        else int(round(budget.min_duration_s * rate))
    )
    max_gap = int(round(float(active_policy["max_internal_silence_s"]) * rate))
    guard = int(round(budget.crop_guard_s * rate))
    min_active = int(round(float(active_policy["min_active_duration_s"]) * rate))
    min_coverage = float(active_policy["min_activity_coverage"])
    max_edge = float(active_policy["max_edge_silence_s"])

    result: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "budget": budget.to_dict(),
        "policy": active_policy,
        "constraints": {
            "max_duration_samples": max_len,
            "min_duration_samples": min_len,
            "max_internal_silence_samples": max_gap,
            "min_active_duration_samples": min_active,
            "min_activity_coverage": min_coverage,
            "max_edge_silence_s": max_edge,
            "crop_guard_samples": guard,
        },
        "candidates": [],
        "selected": None,
        "status": "rejected",
        "reason": None,
    }
    if not intervals:
        result["reason"] = "no_active_window_above_gate"
        return result
    if max_len <= 0:
        result["reason"] = "budget_shorter_than_one_sample"
        return result

    seen: set[tuple[int, int]] = set()
    for index, (first_start, first_end) in enumerate(intervals):
        start = int(first_start)
        stopped = "single_interval_exceeds_budget"
        if first_end - start > max_len:
            end = start + max_len
        else:
            end = int(first_end)
            cursor = index
            stopped = "no_further_interval"
            while cursor + 1 < len(intervals):
                gap = intervals[cursor + 1][0] - intervals[cursor][1]
                if gap > max_gap:
                    stopped = "gap_exceeds_max_internal_silence"
                    break
                if intervals[cursor + 1][1] - start > max_len:
                    stopped = "next_interval_would_exceed_budget"
                    break
                cursor += 1
                end = int(intervals[cursor][1])

        crop_start = max(0, start - guard)
        crop_end = min(length, end + guard)
        applied_guard_s = budget.crop_guard_s
        if crop_end - crop_start > max_len:
            # Drop the guard rather than the sound, exactly as the existing
            # speech crop does when the guard is what breaks the limit.
            crop_start, crop_end = start, end
            applied_guard_s = 0.0
        key = (crop_start, crop_end)
        if key in seen:
            continue
        seen.add(key)

        stats = _span_stats(intervals, crop_start, crop_end, rate)
        failures: list[str] = []
        if crop_end - crop_start < min_len:
            failures.append("segment_shorter_than_requested_minimum")
        if stats["active_duration_s"] * rate < min_active - 0.5:
            failures.append("active_duration_below_minimum")
        if stats["activity_coverage"] < min_coverage:
            failures.append("activity_coverage_below_minimum")
        if stats["max_internal_silence_s"] > float(
            active_policy["max_internal_silence_s"]
        ) + 1e-9:
            failures.append("internal_silence_exceeds_maximum")
        if max(stats["leading_inactive_s"], stats["trailing_inactive_s"]) > (
            max_edge + 1e-9
        ):
            failures.append("edge_silence_exceeds_maximum")

        source_active = float(activity["active_duration_s"])
        candidate = {
            "source_crop_start_sample": int(crop_start),
            "source_crop_end_sample_exclusive": int(crop_end),
            "source_offset_s": float(crop_start / rate),
            "source_crop_duration_s": float((crop_end - crop_start) / rate),
            "crop_guard_s_requested": float(budget.crop_guard_s),
            "crop_guard_s_applied": float(applied_guard_s),
            "first_active_sample": int(start),
            "last_active_sample_exclusive": int(end),
            "extension_stopped_because": stopped,
            "covers_all_source_activity": bool(
                source_active > 0
                and abs(stats["active_duration_s"] - source_active)
                <= 1.0 / rate
            ),
            **{key: value for key, value in stats.items()
               if key != "activity_intervals_samples"},
            "segment_activity_intervals_samples": stats["activity_intervals_samples"],
            "pause_structure": pause_structure(
                intervals, crop_start, crop_end, rate
            ),
            "constraint_failures": failures,
            "feasible": not failures,
        }
        result["candidates"].append(candidate)

    feasible = [row for row in result["candidates"] if row["feasible"]]
    if not feasible:
        reasons = [
            reason
            for row in result["candidates"]
            for reason in row["constraint_failures"]
        ]
        # Report the constraint that blocked the most candidates, and keep the
        # full per-candidate list so a caller can see the rest.
        result["reason"] = (
            max(set(reasons), key=reasons.count) if reasons
            else "no_contiguous_region_fits_budget"
        )
        result["rejection_reasons"] = sorted(set(reasons))
        return result

    winner = min(feasible, key=lambda row: _rank_key(row, budget.selection))
    result["selected"] = winner
    result["status"] = "selected"
    result["selection_strategy"] = budget.selection
    result["feasible_candidate_count"] = len(feasible)
    return result


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceRecording:
    """One original recording, with the digest of the bytes that were decoded.

    Passing a bare sample array between planning and cutting is how a segment
    ends up carrying the wrong provenance: the array says nothing about which
    file it came from, and a same-length rewrite of the original slips through
    a sample-count check untouched.  This handle keeps the path, the digest
    and the samples together, so cutting can refuse an array that does not
    belong to the plan it is cutting for.
    """

    path: str
    sha256: str
    rate_hz: int
    samples: np.ndarray

    @classmethod
    def read(cls, path: str | Path) -> "SourceRecording":
        target = Path(path)
        before = _sha256(target)
        samples, rate = read_source_pcm(target)
        after = _sha256(target)
        if before != after:
            raise SoundSegmentError(
                f"original recording changed while it was being read: {target}"
            )
        return cls(str(target), before, int(rate), samples)

    def mismatches(self, plan: Mapping[str, Any]) -> list[str]:
        """Every way this recording fails to be the one the plan was made from."""

        problems: list[str] = []
        if str(self.path) != str(plan.get("source_path")):
            problems.append("source_path_differs")
        if self.sha256 != plan.get("source_sha256"):
            problems.append("source_sha256_differs")
        if self.rate_hz != int(plan.get("source_rate_hz", -1)):
            problems.append("source_rate_differs")
        if int(self.samples.size) != int(plan.get("source_sample_count", -1)):
            problems.append("source_sample_count_differs")
        return problems


def read_source_pcm(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float samples and rate from one original WAV, read-only.

    Uses the reader the QC stage already uses, so 8/16/24/32-bit and
    multi-channel deliveries behave here exactly as they did when the clip was
    measured.  The original file is never opened for writing anywhere in this
    module.
    """

    samples, rate, _channels, _width = read_mono(Path(path))
    return np.asarray(samples, dtype=np.float64), int(rate)


def analysis_signal(
    samples: np.ndarray, source_rate: int, target_rate: int | None
) -> tuple[np.ndarray, int, str, bool]:
    """Resample the whole recording once, before anything is measured or cut.

    Order matters and this is the order: resample, then measure, then slice.

    Measuring at 44.1 kHz and delivering at 16 kHz asks two different
    questions.  An alarm clock in this library whose energy sits mostly above
    8 kHz passed a 44.1 kHz gate at -19 dBFS and then, once the resampler had
    removed the band that carried it, had no window left above that level at
    all - the plan promised sound the delivered file did not have.  Judging
    activity on the band the dataset actually ships removes that whole class
    of disagreement, and it is the honest question anyway: what a listener
    gets is the 16 kHz file.

    Resampling before slicing rather than after also makes two overlapping
    segments of one recording agree sample for sample where they overlap,
    which slicing first does not.
    """

    source_rate = int(source_rate)
    if target_rate is None or int(target_rate) == source_rate:
        return np.asarray(samples, dtype=np.float64), source_rate, "1/1", False
    target_rate = int(target_rate)
    ratio = Fraction(target_rate, source_rate).limit_denominator(10000)
    resampled = np.asarray(
        resample_poly(np.asarray(samples, dtype=np.float64),
                      ratio.numerator, ratio.denominator),
        dtype=np.float64,
    )
    return resampled, target_rate, f"{ratio.numerator}/{ratio.denominator}", True


def read_clip_metadata(wav_path: str | Path) -> dict[str, Any]:
    """The two sidecars the sound library already writes beside each clip."""

    wav_path = Path(wav_path)
    out: dict[str, Any] = {"clip_json": None, "clip_qc_json": None}
    for key, name in (
        ("clip_json", wav_path.with_suffix(".json")),
        ("clip_qc_json", Path(str(wav_path.with_suffix("")) + ".qc.json")),
    ):
        if name.is_file():
            try:
                out[key] = json.loads(name.read_text(encoding="utf-8"))
                out[f"{key}_path"] = str(name)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                out[key] = None
                out[f"{key}_error"] = f"{type(error).__name__}: {error}"
    return out


def iter_library_clips(
    library_root: str | Path,
    *,
    classes: Iterable[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Walk the ``<class>/<clip_id>/clip.wav`` layout the sound library uses."""

    root = Path(library_root)
    if not root.is_dir():
        raise SoundSegmentError(f"sound library root is not a directory: {root}")
    wanted = None if classes is None else {str(name) for name in classes}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if wanted is not None and class_dir.name not in wanted:
            continue
        for clip_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            wav = clip_dir / "clip.wav"
            if not wav.is_file():
                continue
            metadata = read_clip_metadata(wav)
            declared = (metadata.get("clip_json") or {}).get("event_classes") or []
            yield {
                "source_path": str(wav),
                "relative_path": str(wav.relative_to(root)),
                "source_asset_id": f"{class_dir.name}/{clip_dir.name}",
                "sound_class": str(declared[0]) if declared else class_dir.name,
                "directory_class": class_dir.name,
                "declared_event_classes": [str(name) for name in declared],
                **metadata,
            }


def _transcript_scope(
    clip_json: Mapping[str, Any] | None, covers_all: bool
) -> dict[str, Any]:
    """What may still be said about the words after a crop.

    A transcript describes the whole recording.  Once part of it is gone the
    old transcript is a claim about audio that is no longer there, so it is
    carried forward only when the segment kept every sounding window.
    """

    metadata = dict(clip_json or {})
    transcript = metadata.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        return {
            "transcript": None,
            "transcript_status": "absent_in_source_metadata",
            "covers_all_source_activity": bool(covers_all),
        }
    if covers_all:
        return {
            "transcript": transcript,
            "transcript_status": "inherited_full_source_activity_retained",
            "covers_all_source_activity": True,
        }
    return {
        "transcript": None,
        "transcript_status": "unknown_after_crop",
        "source_transcript_not_reused": transcript,
        "covers_all_source_activity": False,
    }


# Which measured shapes are consistent with each declared family.  Used only
# to report agreement; nothing here changes a policy on its own.
_FAMILY_EXPECTED_FORMS = {
    "speech": {"intermittent_continuous", "continuous", "single_burst",
               "sparse_bursts"},
    "animal_call": {"single_burst", "sparse_bursts", "periodic_pulse_train"},
    "short_prompt": {"single_burst", "sparse_bursts", "periodic_pulse_train",
                     "intermittent_continuous"},
    "vocal_non_speech": {"single_burst", "sparse_bursts"},
    "device_continuous": {"continuous", "intermittent_continuous"},
    "unknown": set(),
}


def _form_matches_family(family: str, form: str) -> bool | None:
    expected = _FAMILY_EXPECTED_FORMS.get(family)
    if not expected:
        return None
    return form in expected


def plan_segment(
    source_path: str | Path,
    *,
    budget: SegmentBudget,
    sound_class: str | None = None,
    policy_overrides: Mapping[str, Any] | None = None,
    source_asset_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_frame_levels: bool = False,
) -> dict[str, Any]:
    """Measure one original recording and choose a segment, writing nothing.

    Returns a plan whose ``status`` is ``selected`` or ``rejected``.  A
    rejected plan still carries the full measurement and every candidate that
    was considered, because "this clip has no usable segment" is a result the
    caller has to be able to argue with.
    """

    path = Path(source_path)
    record: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "source_path": str(path),
        "source_asset_id": str(source_asset_id or path.parent.name),
        "sound_class": sound_class,
        "budget": budget.to_dict(),
        "status": "rejected",
        "reason": None,
        "crop_authorization": CROP_AUTHORIZATION,
        "selection_authorized": True,
        "certification": CERTIFICATION_STATUS,
    }
    sidecars = dict(metadata) if metadata is not None else read_clip_metadata(path)
    record["source_metadata"] = sidecars
    if sound_class is None:
        declared = (sidecars.get("clip_json") or {}).get("event_classes") or []
        sound_class = str(declared[0]) if declared else path.parent.parent.name
        record["sound_class"] = sound_class
        record["sound_class_source"] = "clip_json_event_classes"
    else:
        record["sound_class_source"] = "caller"

    try:
        samples, rate = read_source_pcm(path)
    except (SoundQCError, OSError) as error:
        record["reason"] = "source_read_failed"
        record["detail"] = f"{type(error).__name__}: {error}"
        return record
    record["source_sha256"] = _sha256(path)
    record["source_rate_hz"] = rate
    record["source_sample_count"] = int(samples.size)
    record["source_duration_s"] = float(samples.size / rate)

    analysis, analysis_rate, resample_ratio, antialiased = analysis_signal(
        samples, rate, budget.target_rate_hz
    )
    record.update({
        "analysis_rate_hz": analysis_rate,
        "analysis_sample_count": int(analysis.size),
        "resample_ratio": resample_ratio,
        "antialiased": antialiased,
        "resample_method": "scipy.signal.resample_poly",
        "measurement_band": (
            "delivered_rate" if analysis_rate != rate else "source_rate"
        ),
    })
    policy = segment_policy_for_class(sound_class, overrides=policy_overrides)
    record["policy"] = policy
    try:
        activity = measure_activity(
            analysis, analysis_rate, policy=policy,
            include_frame_levels=include_frame_levels,
        )
    except SoundSegmentError as error:
        record["reason"] = "activity_measurement_failed"
        record["detail"] = str(error)
        return record
    record["activity"] = activity
    record["source_temporal_form"] = activity["structure"]["temporal_form"]
    record["temporal_form_vs_declared_family"] = {
        "declared_activity_family": policy["activity_family"],
        "measured_source_temporal_form": activity["structure"]["temporal_form"],
        "agrees": _form_matches_family(
            policy["activity_family"], activity["structure"]["temporal_form"]
        ),
        "note": (
            "a disagreement is a fact about this recording, not licence to "
            "relabel the class; pass policy_overrides if the class should be "
            "treated differently"
        ),
    }
    record["sound_events_cross_reference"] = cross_reference_sound_events(
        analysis, analysis_rate, sound_class=sound_class
    )
    if activity["gate"]["peak_frame_rms"] <= _amp(
        float(policy["absolute_active_floor_dbfs"])
    ):
        record["reason"] = "no_active_window_above_absolute_floor"
        record["selection"] = {"status": "rejected", "candidates": []}
        return record

    applied_policy = policy
    if budget.temporal_policy == "measured":
        # Same measurement, same gate, same intervals: only the coverage and
        # internal-silence thresholds are re-picked from the shape observed.
        applied_policy = segment_policy_for_class(
            sound_class, overrides=policy_overrides,
            temporal_form=activity["structure"]["temporal_form"],
        )
        record["declared_policy"] = policy
        record["policy"] = applied_policy
    selection = select_segment(activity, budget, policy=applied_policy)
    record["selection"] = selection
    if selection["status"] != "selected":
        record["reason"] = selection["reason"]
        return record

    chosen = dict(selection["selected"])
    # The selector worked in analysis coordinates; keep those, and record the
    # original recording's own sample numbers beside them so the crop can
    # always be pointed at in the file it came from.
    analysis_start = int(chosen.pop("source_crop_start_sample"))
    analysis_end = int(chosen.pop("source_crop_end_sample_exclusive"))
    chosen["analysis_crop_start_sample"] = analysis_start
    chosen["analysis_crop_end_sample_exclusive"] = analysis_end
    back = rate / analysis_rate
    chosen["source_crop_start_sample"] = max(
        0, int(math.floor(analysis_start * back))
    )
    chosen["source_crop_end_sample_exclusive"] = min(
        int(samples.size), int(math.ceil(analysis_end * back))
    )
    chosen["crop_coordinate_note"] = (
        "analysis_* are exact and are what the PCM is cut at; source_* are the "
        "same region expressed in the original recording's samples"
    )
    record.update(chosen)
    record["status"] = "selected"
    # A class name says what the thing is; the temporal form says how its
    # energy is laid out in time.  Different questions, and this module never
    # lets the second one silently rewrite the first.
    record["segment_temporal_form"] = chosen["pause_structure"]["temporal_form"]
    record.update(
        _transcript_scope(
            sidecars.get("clip_json"), bool(chosen["covers_all_source_activity"])
        )
    )
    human_review = (sidecars.get("clip_json") or {}).get("human_review")
    record["source_human_review"] = (
        dict(human_review) if isinstance(human_review, Mapping)
        else {"status": "not_provided"}
    )
    # A source-level listening note says a person heard the original file.  It
    # is not a verdict on a piece of it that no one has heard yet.
    record["segment_human_review"] = {
        "status": "not_requested",
        "inherited_source_status": record["source_human_review"].get("status"),
        "inheritance": "source_level_only_not_a_segment_verdict",
    }
    record["source_qc"] = {
        "verdict": (sidecars.get("clip_qc_json") or {}).get("verdict"),
        "measured": (sidecars.get("clip_qc_json") or {}).get("measured"),
    }
    return record


def _merged_overrides(
    sound_class: Any,
    policy_overrides: Mapping[str, Any] | None,
    class_policy_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Global overrides, then the per-class entry for this clip on top.

    Per-class is the shape a configuration needs.  ``clock_tick`` measuring as
    a pulse train while ``blender`` measures as a hum is a fact about those two
    recordings; expressing it must not require either renaming the class or
    forcing one temporal policy onto every class in the run.
    """

    merged = dict(policy_overrides or {})
    if class_policy_overrides:
        merged.update(dict(class_policy_overrides.get(str(sound_class), {})))
    return merged


def plan_segments(
    sources: Iterable[Mapping[str, Any]],
    *,
    budget: SegmentBudget,
    policy_overrides: Mapping[str, Any] | None = None,
    class_policy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    include_frame_levels: bool = False,
) -> dict[str, Any]:
    """Index many recordings without writing any PCM.

    This is the half a sampler runs over a whole library.  Materialisation is
    a separate call, so indexing 1190 recordings does not put 1190 - let alone
    every sliding window of 1190 - WAV files on disk.
    """

    budget = budget.validated()
    plans: list[dict[str, Any]] = []
    for entry in sources:
        plans.append(
            plan_segment(
                entry["source_path"],
                budget=budget,
                sound_class=entry.get("sound_class"),
                source_asset_id=entry.get("source_asset_id"),
                metadata=entry.get("metadata"),
                policy_overrides=_merged_overrides(
                    entry.get("sound_class"), policy_overrides,
                    class_policy_overrides,
                ),
                include_frame_levels=include_frame_levels,
            )
        )
    selected = [row for row in plans if row["status"] == "selected"]
    rejected = [row for row in plans if row["status"] != "selected"]
    by_class: dict[str, dict[str, Any]] = {}
    for row in plans:
        bucket = by_class.setdefault(
            str(row.get("sound_class")),
            {"considered": 0, "selected": 0, "rejected": 0, "reasons": {}},
        )
        bucket["considered"] += 1
        if row["status"] == "selected":
            bucket["selected"] += 1
        else:
            bucket["rejected"] += 1
            reason = str(row.get("reason"))
            bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1
    forms: dict[str, dict[str, int]] = {}
    for row in plans:
        form = row.get("source_temporal_form")
        if form is None:
            continue
        bucket = forms.setdefault(str(row.get("sound_class")), {})
        bucket[str(form)] = bucket.get(str(form), 0) + 1
    return {
        "schema": INDEX_SCHEMA,
        "budget": budget.to_dict(),
        "policy_overrides": dict(policy_overrides or {}),
        "class_policy_overrides": {
            key: dict(value) for key, value in (class_policy_overrides or {}).items()
        },
        "measured_temporal_form_by_class": forms,
        "plans": plans,
        "counts": {
            "considered": len(plans),
            "selected": len(selected),
            "rejected": len(rejected),
        },
        "by_sound_class": by_class,
        "status": CERTIFICATION_STATUS,
        "claim_boundary": (
            "Segment coordinates and activity measurement only. No PCM was "
            "written and no human listened to any segment."
        ),
    }


def segment_id(
    source_asset_id: str,
    *,
    source_sha256: str,
    facts: Mapping[str, Any],
    processing: Mapping[str, Any],
) -> str:
    """Deterministic id from source identity plus every result-changing setting.

    The digest comes from the function that already derives prepared-audio ids,
    given the same identity dictionary plus this producer's own processing
    settings, so two callers asking for the same crop of the same recording
    under the same settings get the same id and can share one file - and two
    callers asking for *different* processing never do.

    ``processing`` has no default on purpose.  It is the argument that was
    missing on 2026-09-10, and a caller that forgets it should get a
    TypeError, not a colliding id.
    """

    if not processing:
        raise SoundSegmentError(
            "segment_id needs the processing settings that change the samples; "
            f"expected at least {sorted(PCM_DETERMINING_BUDGET_FIELDS)}"
        )
    return make_prepared_audio_id(
        source_asset_id,
        source_sha256=source_sha256,
        facts=facts,
        processing=processing,
        prefix="sound_segment",
    )


def _processing_identity(
    plan: Mapping[str, Any], budget: SegmentBudget
) -> dict[str, Any]:
    """Everything outside the historical identity that changes these samples."""

    return {
        **budget.processing_identity(),
        # The coordinates the PCM is literally sliced at.  ``source_crop_*``
        # is these rounded back into the original recording's samples, so two
        # different cuts can in principle round to the same source pair.
        "analysis_crop_start_sample": int(plan["analysis_crop_start_sample"]),
        "analysis_crop_end_sample_exclusive": int(
            plan["analysis_crop_end_sample_exclusive"]
        ),
        "analysis_rate_hz": int(plan["analysis_rate_hz"]),
        "operation": OPERATION,
    }


def _identity_facts(plan: Mapping[str, Any], budget: SegmentBudget) -> dict[str, Any]:
    policy = plan["policy"]
    return {
        "operation": OPERATION,
        "source_crop_start_sample": int(plan["source_crop_start_sample"]),
        "source_crop_end_sample_exclusive": int(
            plan["source_crop_end_sample_exclusive"]
        ),
        "analysis_crop_start_sample": int(plan["analysis_crop_start_sample"]),
        "analysis_crop_end_sample_exclusive": int(
            plan["analysis_crop_end_sample_exclusive"]
        ),
        "filter": dict(policy["filter"]),
        "activity_filter": dict(policy["filter"]),
        "detector": {
            "gate": policy["gate"],
            "band": policy["band"],
            "window_s": policy["window_s"],
            "hop_s": policy["hop_s"],
            "relative_peak_db": policy["relative_peak_db"],
            "enter_db_above_noise": policy["enter_db_above_noise"],
            "exit_db_above_noise": policy["exit_db_above_noise"],
            "absolute_active_floor_dbfs": policy["absolute_active_floor_dbfs"],
            "silence_run_merge_s": policy["silence_run_merge_s"],
        },
        "target_rate_hz": budget.target_rate_hz,
        # Processing settings live in the separate ``processing`` block, which
        # the id actually reads.  Putting them here as well would look like
        # they were bound when they were not - the historical identity picks a
        # fixed set of keys and silently drops the rest.
    }


def _edge_fade(samples: np.ndarray, rate: int, fade_s: float) -> tuple[np.ndarray, int]:
    """Raised-cosine ramp at both ends, to stop a click at the cut.

    Not loudness editing: the ramp is shorter than the crop guard, so it lands
    on the inactive margin the selector already put around the sound.
    """

    length = int(round(float(fade_s) * rate))
    if length <= 0 or samples.size < 2 * length:
        return samples, 0
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, length + 1) / (length + 1)))
    out = samples.copy()
    out[:length] *= ramp
    out[-length:] *= ramp[::-1]
    return out, length


def materialize_segment(
    plan: Mapping[str, Any],
    output_root: str | Path,
    *,
    budget: SegmentBudget | None = None,
    reuse_existing: bool = True,
    source: SourceRecording | None = None,
    sidecar_wait_s: float = 10.0,
) -> dict[str, Any]:
    """Cut the planned region out of the original and write it as fresh PCM.

    The original file is opened read-only; the segment goes to a new path under
    ``output_root``.  Two callers that planned the same crop of the same
    recording under the same settings resolve to the same id and, with
    ``reuse_existing``, to the same file - which is what "the four members of
    one group share this audio" has to mean in the filesystem.  Two callers
    asking for *different* processing resolve to different ids and different
    files; ``reuse_existing`` never hands back audio that was made under other
    settings.

    Reuse is checked against **this request**, not only against the stored
    record's internal consistency, and the stored PCM is re-hashed before it
    is handed back.  An artifact that is present but incomplete - a clip
    without its sidecar, or bytes that no longer match the sidecar - is left
    exactly where it is and reported, because deleting it would destroy the
    evidence of whatever interrupted it; cut it again under a fresh
    ``output_root``.
    """

    if plan.get("status") != "selected":
        raise SoundSegmentError(
            f"plan is {plan.get('status')!r}, not selected: {plan.get('reason')}"
        )
    budget = (
        budget if budget is not None
        else SegmentBudget(**{
            key: value for key, value in plan["budget"].items()
        })
    ).validated()

    source_path = Path(plan["source_path"])
    crop_start = int(plan["source_crop_start_sample"])
    crop_end = int(plan["source_crop_end_sample_exclusive"])
    analysis_start = int(plan["analysis_crop_start_sample"])
    analysis_end = int(plan["analysis_crop_end_sample_exclusive"])
    source_rate = int(plan["source_rate_hz"])
    target_rate = int(budget.target_rate_hz or source_rate)

    facts = _identity_facts(plan, budget)
    processing = _processing_identity(plan, budget)
    identifier = segment_id(
        str(plan["source_asset_id"]),
        source_sha256=str(plan["source_sha256"]),
        facts=facts,
        processing=processing,
    )
    relative = Path(str(plan.get("sound_class") or "unknown")) / identifier / "clip.wav"
    target = Path(output_root) / relative
    sidecar = target.with_suffix(".json")

    if target.exists() or sidecar.exists():
        if not reuse_existing:
            raise FileExistsError(target)
        return _reuse_existing_segment(
            target, sidecar, identifier=identifier, plan=plan, budget=budget,
            processing=processing, sidecar_wait_s=sidecar_wait_s,
        )

    if source is None:
        source = SourceRecording.read(source_path)
    problems = source.mismatches(plan)
    if problems:
        raise SoundSegmentError(
            f"the recording handed to materialize_segment is not the one the "
            f"plan was made from ({source.path}): {', '.join(problems)}"
        )
    source_samples = np.asarray(source.samples, dtype=np.float64)
    # Same order as planning: resample the whole recording, then slice.  The
    # segment is therefore bit-for-bit the region that was measured, not a
    # separately resampled copy of it.
    analysis, analysis_rate, resample_ratio, antialiased = analysis_signal(
        source_samples, source_rate, budget.target_rate_hz
    )
    if analysis_rate != target_rate:
        raise SoundSegmentError(
            f"analysis rate {analysis_rate} does not match target {target_rate}"
        )
    if analysis.size != int(plan["analysis_sample_count"]):
        raise SoundSegmentError(
            "analysis sample count changed since planning: "
            f"{analysis.size} != {plan['analysis_sample_count']}"
        )
    cut = analysis[analysis_start:analysis_end].copy()
    if cut.size == 0:
        raise SoundSegmentError("planned crop is empty")

    dc_offset_removed = 0.0
    if budget.remove_dc:
        dc_offset_removed = float(cut.mean())
        cut = cut - dc_offset_removed

    faded, fade_samples = _edge_fade(cut, target_rate, budget.edge_fade_s)

    applied_gain_db = 0.0
    linear_gain = 1.0
    if budget.normalize_peak:
        peak = float(np.abs(faded).max())
        if peak <= 0.0:
            raise SoundSegmentError("segment is digital silence; cannot normalise")
        linear_gain = _amp(float(budget.target_peak_dbfs)) / peak
        faded = faded * linear_gain
        applied_gain_db = _db(linear_gain)
    if not np.isfinite(faded).all():
        raise SoundSegmentError("segment contains non-finite samples")
    overshoot_peak_dbfs = None
    headroom_gain_db = 0.0
    peak_abs = float(np.abs(faded).max())
    if peak_abs > 1.0:
        if not budget.allow_resample_headroom:
            raise SoundSegmentError(
                f"segment exceeds PCM full scale: peak_abs={peak_abs:.12g}"
            )
        overshoot_peak_dbfs = _db(peak_abs)
        headroom = 1.0 / peak_abs
        faded = faded * headroom
        headroom_gain_db = _db(headroom)
        linear_gain *= headroom
        applied_gain_db += headroom_gain_db
        peak_abs = float(np.abs(faded).max())
    if peak_abs > 1.0:
        raise SoundSegmentError(
            f"segment exceeds PCM full scale: peak_abs={peak_abs:.12g}"
        )

    try:
        prepared_sha = write_wav_mono_no_clobber(target, faded, target_rate)
    except FileExistsError:
        # Another worker created this exact segment between our existence check
        # and now.  Both of us computed the same id from the same inputs, so
        # the right answer is to verify theirs and use it - not to overwrite it
        # and then claim the bytes happened to match.
        if not reuse_existing:
            raise
        return _reuse_existing_segment(
            target, sidecar, identifier=identifier, plan=plan, budget=budget,
            processing=processing, waited_for_writer=True,
            sidecar_wait_s=sidecar_wait_s,
        )
    facts["applied_gain_db"] = applied_gain_db

    # Already in delivered-rate samples relative to the segment start: the
    # selector measured on this very signal.
    segment_intervals = [
        [max(0, int(start)), min(int(faded.size), int(end))]
        for start, end in plan["segment_activity_intervals_samples"]
    ]
    record: dict[str, Any] = {
        "schema": SEGMENT_SCHEMA,
        "segment_id": identifier,
        "prepared_audio_id": identifier,
        "operation": OPERATION,
        "crop_authorization": CROP_AUTHORIZATION,
        "selection_authorized": True,
        # This segment was chosen, not clipped short by a detector cap.  The
        # older "truncated" rejection in the batch pool is about the second
        # thing; these two must not be conflated.
        "truncated": False,
        "truncation_reason": None,
        "detector_cap_truncated": bool(
            (plan.get("sound_events_cross_reference") or {}).get(
                "detector_cap_truncated"
            )
        ),
        "sound_class": plan.get("sound_class"),
        "activity_family": plan["policy"]["activity_family"],
        "source_asset_id": plan["source_asset_id"],
        "source_path": str(source_path),
        "source_sha256": plan["source_sha256"],
        "source_rate_hz": source_rate,
        "source_sample_count": int(plan["source_sample_count"]),
        "source_duration_s": float(plan["source_duration_s"]),
        "source_crop_start_sample": crop_start,
        "source_crop_end_sample_exclusive": crop_end,
        "analysis_crop_start_sample": analysis_start,
        "analysis_crop_end_sample_exclusive": analysis_end,
        "analysis_rate_hz": analysis_rate,
        "analysis_sample_count": int(analysis.size),
        "crop_coordinate_note": plan.get("crop_coordinate_note"),
        "source_offset_s": float(crop_start / source_rate),
        "source_crop_duration_s": float((crop_end - crop_start) / source_rate),
        "crop_guard_s_requested": plan["crop_guard_s_requested"],
        "crop_guard_s_applied": plan["crop_guard_s_applied"],
        "source_activity_intervals_samples": [
            list(pair) for pair in plan["activity"]["source_activity_intervals_samples"]
        ],
        "segment_activity_intervals_samples": segment_intervals,
        "activity_interval_coordinates": {
            "source_activity_intervals_samples": (
                "whole_recording_samples_at_analysis_rate"
            ),
            "segment_activity_intervals_samples": (
                "prepared_segment_samples_at_target_rate"
            ),
        },
        "prepared": relative.as_posix(),
        "prepared_path": str(target),
        "prepared_sha256": prepared_sha,
        "prepared_sample_count": int(faded.size),
        "prepared_duration_s": float(faded.size / target_rate),
        "prepared_peak_dbfs": _db(peak_abs),
        "target_rate_hz": target_rate,
        "resample_ratio": resample_ratio,
        "antialiased": antialiased,
        "resample_method": "scipy.signal.resample_poly",
        "edge_fade_s": float(budget.edge_fade_s),
        "edge_fade_samples": int(fade_samples),
        "edge_fade_shape": "raised_cosine",
        "time_stretched": False,
        "concatenated_from_multiple_regions": False,
        "remove_dc": bool(budget.remove_dc),
        "dc_offset_removed": dc_offset_removed,
        "normalization_applied": bool(budget.normalize_peak),
        "applied_gain_db": applied_gain_db,
        "linear_gain": float(linear_gain),
        "resample_overshoot_peak_dbfs": overshoot_peak_dbfs,
        "resample_headroom_gain_db": headroom_gain_db,
        "headroom_is_overflow_guard": True,
        "headroom_fired": overshoot_peak_dbfs is not None,
        "gain_is_uniform_over_segment": True,
        "gain_recovers_scale_not_quantisation": (
            "the recorded factor recovers the original scale; 16-bit "
            "quantisation is not undone and this is not a lossless round trip"
        ),
        "per_sample_loudness_levelling": False,
        "planned_activity": {
            key: plan[key]
            for key in (
                "duration_s", "active_duration_s", "activity_coverage",
                "activity_interval_count", "leading_inactive_s",
                "trailing_inactive_s", "max_internal_silence_s",
                "covers_all_source_activity", "extension_stopped_because",
            )
        },
        "pause_structure": plan["pause_structure"],
        "source_temporal_form": plan.get("source_temporal_form"),
        "segment_temporal_form": plan.get("segment_temporal_form"),
        "temporal_form_vs_declared_family": plan.get(
            "temporal_form_vs_declared_family"
        ),
        "policy": plan["policy"],
        "budget": budget.to_dict(),
        "source_activity_measurement": {
            key: plan["activity"][key]
            for key in (
                "gate", "frame", "measurement", "activity_coverage",
                "active_duration_s", "max_internal_silence_s",
                "whole_clip_rms_dbfs", "whole_clip_peak_dbfs",
                "frame_level_dbfs_percentiles",
            )
        },
        "sound_events_cross_reference": plan.get("sound_events_cross_reference"),
        "transcript": plan.get("transcript"),
        "transcript_status": plan.get("transcript_status"),
        "source_transcript_not_reused": plan.get("source_transcript_not_reused"),
        "source_human_review": plan.get("source_human_review"),
        "segment_human_review": plan.get("segment_human_review"),
        "source_qc": plan.get("source_qc"),
        "source_metadata_paths": {
            key: plan.get("source_metadata", {}).get(key)
            for key in ("clip_json_path", "clip_qc_json_path")
        },
        "activity_measurement": "avengine_sound_segment_window_rms_activity_v1",
        "activity_calibration": "placeholder",
        "activity_guard_included": False,
        "activity_is_qa_event_count": False,
        "certification": CERTIFICATION_STATUS,
        "reused_existing_artifact": False,
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with sidecar.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except BaseException:
        # We created the clip in this call, so removing it is removing our own
        # half-finished work, not somebody else's artifact.
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    record["sidecar_path"] = str(sidecar)
    return record


def _reuse_existing_segment(
    target: Path,
    sidecar: Path,
    *,
    identifier: str,
    plan: Mapping[str, Any],
    budget: SegmentBudget,
    processing: Mapping[str, Any],
    waited_for_writer: bool = False,
    sidecar_wait_s: float = 10.0,
) -> dict[str, Any]:
    """Hand back an artifact already on disk, or refuse and say why.

    Three things have to hold, and the old version checked only the first
    half of the first one: the record has to be the segment this request asks
    for, the bytes on disk have to be the bytes that record describes, and the
    processing settings recorded there have to be the ones being requested
    now.
    """

    # A clip with no sidecar yet is either an abandoned half-publish or a
    # worker that is still finishing one.  Those look identical from here, so
    # give the second case a bounded chance to become the first case's
    # opposite before calling the artifact incomplete.
    deadline = time.monotonic() + max(0.0, float(sidecar_wait_s))
    waited = 0.0
    while target.is_file() and not sidecar.is_file():
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
        waited = time.monotonic() - (deadline - max(0.0, float(sidecar_wait_s)))

    incomplete = []
    if not target.is_file():
        incomplete.append(f"clip missing: {target}")
    if not sidecar.is_file():
        incomplete.append(
            f"sidecar missing after waiting {waited:.2f}s: {sidecar}")
    if incomplete:
        raise SoundSegmentError(
            "found an incomplete segment artifact and left it in place ("
            + "; ".join(incomplete)
            + "). Nothing here was deleted; cut this segment again under a "
            "fresh output root rather than repairing in place."
        )
    try:
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SoundSegmentError(
            f"segment sidecar cannot be read and was left in place ({sidecar}): "
            f"{type(error).__name__}: {error}. Cut again under a fresh output root."
        ) from error

    differences: list[str] = []
    for field, expected in (
        ("segment_id", identifier),
        ("source_sha256", plan["source_sha256"]),
        ("source_path", str(plan["source_path"])),
        ("source_crop_start_sample", int(plan["source_crop_start_sample"])),
        ("source_crop_end_sample_exclusive",
         int(plan["source_crop_end_sample_exclusive"])),
        ("analysis_crop_start_sample", int(plan["analysis_crop_start_sample"])),
        ("analysis_crop_end_sample_exclusive",
         int(plan["analysis_crop_end_sample_exclusive"])),
    ):
        if existing.get(field) != expected:
            differences.append(f"{field}: stored {existing.get(field)!r} != "
                               f"requested {expected!r}")
    stored_budget = existing.get("budget") or {}
    for name, value in budget.processing_identity().items():
        if stored_budget.get(name) != value:
            differences.append(f"budget.{name}: stored "
                               f"{stored_budget.get(name)!r} != requested {value!r}")
    if differences:
        raise SoundSegmentError(
            f"the segment already at {target} was made under different settings "
            "and was left in place: " + "; ".join(differences)
            + ". Cut this request under a fresh output root."
        )

    actual = _sha256(target)
    if actual != existing.get("prepared_sha256"):
        raise SoundSegmentError(
            f"the PCM at {target} does not match the sidecar it was published "
            f"with (on disk {actual}, recorded {existing.get('prepared_sha256')}). "
            "Left in place; cut again under a fresh output root."
        )

    record = dict(existing)
    record["reused_existing_artifact"] = True
    record["reuse_verified_against_request"] = True
    record["reuse_checked"] = {
        "identity_fields": True,
        "requested_processing": dict(processing),
        "prepared_sha256_rechecked": actual,
        "lost_write_race_to_another_worker": bool(waited_for_writer),
        "waited_for_sidecar_s": round(waited, 3),
    }
    record["prepared_path"] = str(target)
    record["sidecar_path"] = str(sidecar)
    return record


def verify_segment_artifact(
    record: Mapping[str, Any],
    *,
    coverage_tolerance: float = 0.05,
    duration_tolerance_s: float = 0.002,
) -> dict[str, Any]:
    """Read the written segment back off disk and re-measure it.

    The plan's numbers describe the original recording.  These describe the
    file that now exists, at the rate it was written, after resampling, the
    anti-click fade and 16-bit quantisation.  Two measurements are made and
    both are reported, but only one of them is the pass criterion:

    * **Pass criterion** - the written segment measured against the gate that
      selected it, scaled by whatever uniform gain was applied.  This is the
      one that answers "does the file on disk still carry the sound the plan
      promised, in the same places".
    * **Reported alongside** - the segment measured against its own peak and
      its own noise percentile, which is what a consumer reading only this
      file would compute.  It is deliberately *not* the criterion: a cut piece
      has a different peak and a different noise floor from the recording it
      came from, so re-deriving a gate from it asks a different question and
      will disagree, sometimes by a lot, on material whose level varies.
    """

    path = Path(record["prepared_path"])
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, **detail: Any) -> None:
        checks.append({"check": name, "status": "pass" if ok else "fail", **detail})

    if not path.is_file():
        return {
            "schema": VERIFICATION_SCHEMA,
            "segment_id": record.get("segment_id"),
            "status": "fail",
            "checks": [{"check": "artifact_exists", "status": "fail",
                        "path": str(path)}],
        }
    samples, rate, channels, width = read_mono(path)
    samples = np.asarray(samples, dtype=np.float64)
    check("artifact_exists", True, path=str(path))
    check("channel_count_is_mono", channels == 1, channels=channels)
    check(
        "sample_rate_matches_record",
        rate == int(record["target_rate_hz"]),
        read=rate, recorded=int(record["target_rate_hz"]),
    )
    check(
        "sample_count_matches_record",
        samples.size == int(record["prepared_sample_count"]),
        read=int(samples.size), recorded=int(record["prepared_sample_count"]),
    )
    check("samples_are_finite", bool(np.isfinite(samples).all()))
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    check("peak_within_full_scale", peak <= 1.0, peak_dbfs=_db(peak))
    check(
        "prepared_sha256_matches",
        _sha256(path) == record.get("prepared_sha256"),
    )

    policy = dict(record["policy"])
    policy["max_edge_silence_s"] = float(policy["max_edge_silence_s"])
    local = measure_activity(samples, rate, policy=policy)
    source_gate = record["source_activity_measurement"]["gate"]
    gain = float(record.get("linear_gain", 1.0))
    replayed_gate = (
        float(source_gate["enter_frame_rms"]) * gain,
        float(source_gate["exit_frame_rms"]) * gain,
    )
    replayed = measure_activity(
        samples, rate, policy=policy, threshold_frame_rms=replayed_gate
    )

    check(
        "readback_duration_matches_record",
        abs(samples.size / rate - float(record["prepared_duration_s"]))
        <= duration_tolerance_s,
        read_s=float(samples.size / rate),
        recorded_s=float(record["prepared_duration_s"]),
    )
    check(
        "readback_activity_coverage_meets_policy",
        replayed["activity_coverage"] + coverage_tolerance
        >= float(policy["min_activity_coverage"]),
        measured=replayed["activity_coverage"],
        required=float(policy["min_activity_coverage"]),
    )
    check(
        "readback_active_duration_meets_policy",
        replayed["active_duration_s"] + duration_tolerance_s
        >= float(policy["min_active_duration_s"]),
        measured=replayed["active_duration_s"],
        required=float(policy["min_active_duration_s"]),
    )
    check(
        "readback_internal_silence_within_policy",
        replayed["max_internal_silence_s"]
        <= float(policy["max_internal_silence_s"]) + 1e-9,
        measured=replayed["max_internal_silence_s"],
        allowed=float(policy["max_internal_silence_s"]),
    )
    check(
        "readback_edge_silence_within_policy",
        max(replayed["leading_inactive_s"], replayed["trailing_inactive_s"])
        <= float(policy["max_edge_silence_s"]) + 1e-9,
        leading_s=replayed["leading_inactive_s"],
        trailing_s=replayed["trailing_inactive_s"],
        allowed=float(policy["max_edge_silence_s"]),
    )
    check(
        "readback_coverage_matches_plan",
        abs(replayed["activity_coverage"]
            - float(record["planned_activity"]["activity_coverage"]))
        <= coverage_tolerance,
        readback=replayed["activity_coverage"],
        planned=float(record["planned_activity"]["activity_coverage"]),
        tolerance=coverage_tolerance,
    )
    check(
        "readback_starts_and_ends_on_sound",
        bool(replayed["source_activity_intervals_samples"]),
        interval_count=replayed["activity_interval_count"],
    )

    qualifications: list[dict[str, Any]] = []
    if local["whole_clip_rms_dbfs"] < SEGMENT_QUIET_RMS_DBFS:
        qualifications.append({
            "name": "segment_is_quiet",
            "measured_rms_dbfs": local["whole_clip_rms_dbfs"],
            "warn_below_dbfs": SEGMENT_QUIET_RMS_DBFS,
            "note": "same severity sound_qc assigns to a quiet clip: warn, not fail",
        })
    if abs(local["activity_coverage"] - replayed["activity_coverage"]) > 0.10:
        qualifications.append({
            "name": "segment_local_gate_disagrees_with_selecting_gate",
            "coverage_against_selecting_gate": replayed["activity_coverage"],
            "coverage_against_segment_own_peak": local["activity_coverage"],
            "selecting_gate_dbfs": _db(replayed_gate[0]),
            "segment_local_gate_dbfs": local["gate"]["threshold_dbfs"],
            "note": (
                "the cut piece has a different peak and noise percentile from "
                "the recording, so a gate re-derived from it answers a "
                "different question; the selecting gate is the criterion"
            ),
        })
    if record["activity_family"] in {"device_continuous", "unknown"}:
        qualifications.append({
            "name": "level_contrast_cannot_separate_device_from_room_tone",
            "gate_contrast_db": record["source_activity_measurement"]["gate"][
                "gate_contrast_db"
            ],
            "note": (
                "a steady device sound and a steady room tone have the same "
                "shape; only the absolute level floor and the class label "
                "distinguish them here, and no one has listened"
            ),
        })

    failed = [row for row in checks if row["status"] == "fail"]
    return {
        "schema": VERIFICATION_SCHEMA,
        "segment_id": record.get("segment_id"),
        "prepared_path": str(path),
        "status": "fail" if failed else ("qualified" if qualifications else "pass"),
        "checks": checks,
        "failed_checks": [row["check"] for row in failed],
        "qualifications": qualifications,
        "readback_segment_local_activity": {
            key: local[key]
            for key in (
                "activity_coverage", "active_duration_s",
                "activity_interval_count", "leading_inactive_s",
                "trailing_inactive_s", "max_internal_silence_s",
                "whole_clip_rms_dbfs", "whole_clip_peak_dbfs",
            )
        },
        "readback_selecting_gate_activity": {
            key: replayed[key]
            for key in (
                "activity_coverage", "active_duration_s",
                "activity_interval_count", "leading_inactive_s",
                "trailing_inactive_s", "max_internal_silence_s",
            )
        },
        "readback_selecting_gate_frame_rms": list(replayed_gate),
        "readback_selecting_gate_dbfs": [
            _db(replayed_gate[0]), _db(replayed_gate[1])
        ],
        "readback_frame": local["frame"],
        "readback_sample_width_bytes": width,
        "certification": CERTIFICATION_STATUS,
        "claim_boundary": (
            "Machine read-back of the written PCM. Not a listening test and "
            "not an acoustic answerability certificate."
        ),
    }


def pool_row(
    record: Mapping[str, Any],
    *,
    compatible_asset_ids: Sequence[str] | None = None,
    compatible_object_categories: Sequence[str] | None = None,
    species_id: str | None = None,
) -> dict[str, Any]:
    """One candidate in the shape ``build_batch_sound_pool`` already emits.

    ``compatible_asset_ids`` stays the caller's business: the registry mapping
    from a sound class to the assets that accept it lives in the pool builder,
    not here.

    Coordinate note for the consumer: ``source_activity_intervals_samples``
    here is in the coordinates of the file at ``path`` - that is, the prepared
    segment - because that is what every existing reader of this field assumes.
    The original recording's coordinates are kept beside it under
    ``origin_activity_intervals_samples``.
    """

    # Older prepared records predate rhythm metadata, but retain the actual
    # measured intervals. Reconstruct only the summary; never recut the PCM.
    structure = record.get("pause_structure")
    structure_source = "segment_record"
    if structure is None:
        structure = pause_structure(
            record["segment_activity_intervals_samples"], 0,
            int(record["prepared_sample_count"]), int(record["target_rate_hz"]),
        )
        structure_source = "prepared_segment_activity_intervals_samples"
    if not isinstance(structure, Mapping):
        raise SoundSegmentError("segment pause_structure must be an object")
    coordinates = record["activity_interval_coordinates"]
    row_coordinates = {
        "source_activity_intervals_samples": coordinates[
            "segment_activity_intervals_samples"],
        "origin_activity_intervals_samples": coordinates[
            "source_activity_intervals_samples"],
    }

    return {
        "sound_asset_id": record["segment_id"],
        "path": record["prepared_path"],
        "sound_class": record["sound_class"],
        "event_class": record["sound_class"],
        "species_id": species_id,
        "source_origin": record["source_path"],
        "source_origin_aliases": [record["source_path"]],
        "sound_identity_id": "source:" + str(record["source_path"]),
        "sound_identity_keys": [
            "source:" + str(record["source_path"]),
            "segment:" + str(record["segment_id"]),
        ],
        "compatible_asset_ids": sorted(compatible_asset_ids or []),
        "compatible_object_categories": list(compatible_object_categories or []),
        "sample_count": int(record["prepared_sample_count"]),
        "sample_rate_hz": int(record["target_rate_hz"]),
        "active_duration_s": float(record["planned_activity"]["active_duration_s"]),
        "activity_coverage": float(record["planned_activity"]["activity_coverage"]),
        "max_internal_silence_s": float(
            record["planned_activity"]["max_internal_silence_s"]
        ),
        "leading_inactive_s": float(record["planned_activity"]["leading_inactive_s"]),
        "trailing_inactive_s": float(record["planned_activity"]["trailing_inactive_s"]),
        # Coverage on its own settles nothing.  A dialled number is 40-49%
        # sounding because it has gaps between digits and the owner asked for
        # that rhythm to be kept; a clip that trails off into silence can have
        # the same number.  These are what tell them apart, and the thresholds
        # that were actually applied are beside them.
        "pause_structure": dict(structure),
        "pause_structure_source": structure_source,
        "temporal_form": record.get("segment_temporal_form") or structure.get("temporal_form"),
        "source_temporal_form": record.get("source_temporal_form"),
        "temporal_form_vs_declared_family": record.get(
            "temporal_form_vs_declared_family"),
        "applied_thresholds": {
            "min_activity_coverage": record["policy"]["min_activity_coverage"],
            "max_internal_silence_s": record["policy"]["max_internal_silence_s"],
            "min_active_duration_s": record["policy"]["min_active_duration_s"],
            "max_edge_silence_s": record["policy"]["max_edge_silence_s"],
            "temporal_form_applied": record["policy"].get("temporal_form_applied"),
            "temporal_policy": record["budget"].get("temporal_policy"),
        },
        "coverage_alone_is_not_a_verdict": True,
        "source_activity_intervals_samples": [
            list(pair) for pair in record["segment_activity_intervals_samples"]
        ],
        "origin_activity_intervals_samples": [
            list(pair) for pair in record["source_activity_intervals_samples"]
        ],
        "activity_interval_coordinates": row_coordinates,
        # Keep exact bounds with their coordinate system for the existing
        # source-capability validator; missing legacy bounds stay unknown.
        **{key: record[key] for key in (
            "source_sample_count", "analysis_sample_count", "analysis_rate_hz",
        ) if key in record},
        "audible_start_sample": (
            record["segment_activity_intervals_samples"][0][0]
            if record["segment_activity_intervals_samples"] else None
        ),
        "audible_end_sample_exclusive": (
            record["segment_activity_intervals_samples"][-1][1]
            if record["segment_activity_intervals_samples"] else None
        ),
        "activity_measurement": record["activity_measurement"],
        "activity_calibration": record["activity_calibration"],
        "activity_guard_included": record["activity_guard_included"],
        "activity_is_qa_event_count": record["activity_is_qa_event_count"],
        "linear_gain": float(record["linear_gain"]),
        "normalization_applied": bool(record["normalization_applied"]),
        # Distinguishes an authorised crop from the older "a detector cut this
        # short and nobody knows why" rejection the batch pool already has.
        "selection_authorized": True,
        "crop_authorization": record["crop_authorization"],
        "truncated": False,
        "truncation_reason": None,
        "detector_cap_truncated": bool(record.get("detector_cap_truncated")),
        "source_asset_id": record["source_asset_id"],
        "source_sha256": record["source_sha256"],
        "source_crop_start_sample": int(record["source_crop_start_sample"]),
        "source_crop_end_sample_exclusive": int(
            record["source_crop_end_sample_exclusive"]
        ),
        "source_rate_hz": int(record["source_rate_hz"]),
        "transcript": record.get("transcript"),
        "transcript_status": record.get("transcript_status"),
        "human_review": record.get("segment_human_review")
        or {"status": "not_requested"},
        "source_human_review": record.get("source_human_review"),
        "source_metadata_manifest": (
            record.get("source_metadata_paths", {}).get("clip_json_path")
        ),
    }


def prepare_segments(
    sources: Iterable[Mapping[str, Any]],
    output_root: str | Path,
    *,
    budget: SegmentBudget,
    policy_overrides: Mapping[str, Any] | None = None,
    class_policy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    verify: bool = True,
    include_frame_levels: bool = False,
) -> dict[str, Any]:
    """Index, cut and read back in one call - the whole loop, no agent in it.

    Every stage is a plain function on plain data, so a normal worker process
    runs this to completion: nothing waits for a person or a model to judge a
    clip in the middle.
    """

    budget = budget.validated()
    index = plan_segments(
        sources, budget=budget, policy_overrides=policy_overrides,
        class_policy_overrides=class_policy_overrides,
        include_frame_levels=include_frame_levels,
    )
    segments: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for plan in index["plans"]:
        if plan["status"] != "selected":
            continue
        try:
            record = materialize_segment(plan, output_root, budget=budget)
        except (SoundSegmentError, OSError, FileExistsError) as error:
            failures.append({
                "source_path": plan["source_path"],
                "sound_class": plan.get("sound_class"),
                "stage": "materialize",
                "error": f"{type(error).__name__}: {error}",
            })
            continue
        segments.append(record)
        if verify:
            verification = verify_segment_artifact(record)
            verifications.append(verification)
            if verification["status"] == "fail":
                failures.append({
                    "source_path": plan["source_path"],
                    "sound_class": plan.get("sound_class"),
                    "stage": "verify",
                    "error": verification["failed_checks"],
                })
    by_status: dict[str, int] = {}
    for verification in verifications:
        by_status[verification["status"]] = by_status.get(verification["status"], 0) + 1
    return {
        "schema": INDEX_SCHEMA,
        "output_root": str(output_root),
        "budget": budget.to_dict(),
        "index": index,
        "segments": segments,
        "verifications": verifications,
        "failures": failures,
        "counts": {
            **index["counts"],
            "materialized": len(segments),
            "verification_by_status": by_status,
            "failures": len(failures),
        },
        "status": CERTIFICATION_STATUS,
        "claim_boundary": (
            "Automatic activity measurement, contiguous crop and machine "
            "read-back. No listening test, no acoustic answerability claim, "
            "and no change to any original recording."
        ),
    }
