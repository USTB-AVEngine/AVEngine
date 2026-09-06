"""Turn collected dry clips into pipeline-ready material, reproducibly.

The person collecting sound material was told, in writing, not to edit
anything by hand: drop the files in, and the machine does the format work.
This module is that promise. It reads a clip exactly as delivered and
produces the 16 kHz mono form the acoustic pipeline consumes, recording
every number it changed so the result can be argued with rather than
merely trusted.

Four operations, in this order, each for a stated reason:

* **Skip anything QC called unusable.** Preparation is not repair. A clip
  with a third of its samples pinned at full scale is distorted, and
  resampling distortion produces resampled distortion. Clips that were
  never checked are skipped too, because a silent pass over unmeasured
  material is how bad audio reaches a dataset.
* **Remove the DC offset** by subtracting the mean. A constant offset
  costs headroom and survives every later gain stage; one delivered clip
  sat at 0.36 of full scale.
* **Resample with a polyphase anti-aliasing filter**, not by picking
  every third sample. Going from 44.1 kHz to 16 kHz discards everything
  above 8 kHz, and without the filter those frequencies do not vanish -
  they fold back down into the band the spatial cues live in.
* **Trim and normalise** to a stated peak, keeping a short guard before
  the first sound so an event onset is never clipped off, and recording
  the applied gain so the original loudness is recoverable.

Byte-identical clips filed under several event classes are prepared once
and recorded as aliases: two sound sources in one room must never be
handed the same waveform and asked which of them is sounding.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import wave
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

from avengine.assets.sound_harvest import (
    SPEECH_METADATA_FIELDS,
    speech_metadata_from_mapping,
)

PREPARED_SPEECH_SCHEMA = "avengine_prepared_speech_set_v1"
ACTIVITY_PROFILE_SCHEMA = "avengine_sound_activity_profile_v1"
SCHEMA = "avengine_prepared_sound_clip_v1"

TARGET_RATE_HZ = 16000
TARGET_PEAK_DBFS = -3.0
# Content is anything within this much of the clip's own peak; quieter
# head and tail is silence to trim.
_TRIM_FLOOR_DB = 40.0
_TRIM_GUARD_S = 0.030
_FRAME_S = 0.010

# The speech detector is intentionally separate from the generic head/tail
# trimmer above. It detects a speech-band pre-roll without changing the
# generic preparation contract. These values mirror the read-only measurement
# in tmp/qa_generalized_sampler_review_20260906_v1/measure_audio_pool.py.
SPEECH_FILTER_DEFAULTS = {
    "highpass_hz": 80.0,
    "bandpass_low_hz": 300.0,
    "bandpass_high_hz": 3400.0,
    "order": 4,
    "phase": "zero_phase",
}
SPEECH_DETECTOR_DEFAULTS = {
    "window_s": 0.020,
    "hop_s": 0.010,
    "relative_peak_db": -25.0,
    "level": "frame_rms",
    "active_duration": "union_of_active_windows",
}
SPEECH_MIN_AUDIBLE_S = 1.5
SPEECH_MAX_CLIP_S = 5.0
SPEECH_CROP_GUARD_S = 0.030


class PrepareError(ValueError):
    pass


@dataclass
class PreparedClip:
    source: str
    prepared: str | None
    status: str
    reason_zh: str
    facts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_source_metadata(wav_path: Path) -> dict[str, Any]:
    """Read only explicit optional metadata beside one source WAV."""

    sidecar = wav_path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return speech_metadata_from_mapping(payload)


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(frames)
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 1:
        samples = (
            np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
        ) / 128.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = (
            packed[:, 0]
            | (packed[:, 1] << 8)
            | (packed[:, 2].astype(np.int8).astype(np.int32) << 16)
        )
        samples = value.astype(np.float64) / 8388608.0
    else:
        raise PrepareError(f"unhandled sample width {width}")
    if channels > 1:
        usable = len(samples) // channels * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return samples, rate


def _write_wav_mono(path: Path, samples: np.ndarray, rate: int) -> str:
    ints = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trim_bounds(samples: np.ndarray, rate: int) -> tuple[int, int]:
    hop = max(1, int(_FRAME_S * rate))
    usable = len(samples) // hop * hop
    if usable == 0:
        return 0, len(samples)
    frames = samples[:usable].reshape(-1, hop)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-20)
    floor = rms.max() * 10 ** (-_TRIM_FLOOR_DB / 20)
    loud = np.flatnonzero(rms > floor)
    if loud.size == 0:
        return 0, len(samples)
    guard = int(_TRIM_GUARD_S * rate)
    start = max(0, int(loud[0]) * hop - guard)
    end = min(len(samples), (int(loud[-1]) + 1) * hop + guard)
    return start, end


def prepare_samples(
    samples: np.ndarray,
    rate: int,
    *,
    target_rate_hz: int = TARGET_RATE_HZ,
    target_peak_dbfs: float = TARGET_PEAK_DBFS,
    normalize_peak: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """DC removal, anti-aliased resample, trim, normalise - with a record."""

    facts: dict[str, Any] = {
        "source_rate_hz": rate,
        "source_sample_count": int(len(samples)),
    }
    offset = float(samples.mean())
    facts["removed_dc_offset"] = round(offset, 6)
    work = samples - offset

    if rate != target_rate_hz:
        ratio = Fraction(target_rate_hz, rate).limit_denominator(10000)
        work = resample_poly(work, ratio.numerator, ratio.denominator)
        facts["resample_ratio"] = f"{ratio.numerator}/{ratio.denominator}"
        facts["antialiased"] = True
    else:
        facts["resample_ratio"] = "1/1"
        facts["antialiased"] = False

    start, end = _trim_bounds(work, target_rate_hz)
    facts["trimmed_head_s"] = round(start / target_rate_hz, 3)
    facts["trimmed_tail_s"] = round((len(work) - end) / target_rate_hz, 3)
    work = work[start:end]
    if work.size == 0:
        raise PrepareError("nothing left after trimming; the clip is silent")

    peak = float(np.abs(work).max())
    if peak <= 0.0:
        raise PrepareError("clip is digital silence")
    gain = (10 ** (target_peak_dbfs / 20)) / peak if normalize_peak else 1.0
    facts["peak_normalization"] = bool(normalize_peak)
    facts["applied_gain_db"] = round(20 * np.log10(gain), 2)
    work = work * gain

    facts["prepared_sample_count"] = int(len(work))
    facts["prepared_duration_s"] = round(len(work) / target_rate_hz, 3)
    facts["prepared_peak_dbfs"] = round(
        20 * np.log10(float(np.abs(work).max())), 2
    )
    return work, facts


def prepare_library(
    library_root: Path,
    output_root: Path,
    *,
    target_rate_hz: int = TARGET_RATE_HZ,
    accept_warn: bool = True,
) -> dict[str, Any]:
    """Prepare every usable clip; skip the unusable and say which.

    ``accept_warn`` keeps clips QC merely warned about - a long tail or a
    quiet passage is a judgement call for a person, not grounds for the
    machine to drop material silently.
    """

    results: list[PreparedClip] = []
    prepared_by_digest: dict[str, str] = {}

    for wav_path in sorted(library_root.rglob("*.wav")):
        relative = wav_path.relative_to(library_root).as_posix()
        metadata = _read_source_metadata(wav_path)
        digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()

        qc_path = wav_path.with_suffix(".qc.json")
        if not qc_path.is_file():
            results.append(
                PreparedClip(
                    relative, None, "skipped",
                    "还没做过质检,先跑 qc_sound_library.py",
                    metadata=metadata,
                )
            )
            continue
        try:
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
        except ValueError as error:
            results.append(
                PreparedClip(
                    relative, None, "skipped",
                    f"质检报告读不了:{error}",
                    metadata=metadata,
                )
            )
            continue
        verdict = str(qc.get("verdict"))
        if verdict == "fail" or (verdict == "warn" and not accept_warn):
            reasons = "；".join(
                f.get("reason_zh", "") for f in qc.get("findings") or []
                if f.get("severity") == "fail"
            )
            results.append(
                PreparedClip(
                    relative, None, "skipped",
                    f"质检判为{verdict},不做处理:{reasons}",
                    metadata=metadata,
                )
            )
            continue

        if digest in prepared_by_digest:
            results.append(
                PreparedClip(
                    relative, prepared_by_digest[digest], "alias",
                    f"与 {prepared_by_digest[digest]} 是同一段音频,不重复处理",
                    {"source_sha256": digest},
                    metadata,
                )
            )
            continue

        try:
            samples, rate = _read_wav_mono(wav_path)
            work, facts = prepare_samples(
                samples, rate, target_rate_hz=target_rate_hz
            )
        except (PrepareError, wave.Error, OSError, ValueError) as error:
            results.append(
                PreparedClip(
                    relative, None, "failed",
                    f"处理失败:{error}",
                    metadata=metadata,
                )
            )
            continue

        target = output_root / relative
        prepared_sha = _write_wav_mono(target, work, target_rate_hz)
        prepared_by_digest[digest] = relative
        facts.update(
            {
                "source_sha256": digest,
                "prepared_sha256": prepared_sha,
                "qc_verdict": verdict,
            }
        )
        results.append(
            PreparedClip(
                relative, relative, "prepared",
                f"已转 {target_rate_hz} Hz 单声道,去直流、去首尾静音、峰值归一",
                facts,
                metadata,
            )
        )

    counts: dict[str, int] = {}
    for row in results:
        counts[row.status] = counts.get(row.status, 0) + 1
    return {
        "schema": SCHEMA,
        "library_root": str(library_root),
        "output_root": str(output_root),
        "target_rate_hz": target_rate_hz,
        "target_peak_dbfs": TARGET_PEAK_DBFS,
        "counts": counts,
        "clips": [
            {
                "source": row.source,
                "prepared": row.prepared,
                "status": row.status,
                "reason_zh": row.reason_zh,
                **({"facts": row.facts} if row.facts else {}),
                **({key: row.metadata[key] for key in SPEECH_METADATA_FIELDS
                    if key in row.metadata} if row.metadata else {}),
            }
            for row in results
        ],
    }
_ANIMAL_SOUND_CLASSES = frozenset(
    {
        "dog_bark",
        "cat_meow",
        "animal_call",
        "animal_vocalization",
    }
)
_DEVICE_SOUND_CLASSES = frozenset(
    {
        "air_conditioning",
        "any_audioset_class_playback",
        "audio_playback",
        "bathtub_filling_washing",
        "blender",
        "clock_tick",
        "crackle",
        "drip",
        "fire",
        "gurgling",
        "microwave_hum",
        "music_playback",
        "printer",
        "sink_filling_washing",
        "toilet_flush",
        "water_tap_faucet",
    }
)
_SHORT_PROMPT_CLASSES = frozenset(
    {
        "alarm_bell",
        "alarm_beep",
        "alarm_clock",
        "buzzer",
        "cellphone_vibration_alert",
        "chime",
        "ding_dong",
        "doorbell",
        "doorbell_chime",
        "fire_alarm",
        "microwave_beep",
        "phone_ring",
        "ringtone",
        "smoke_alarm",
        "telephone",
        "telephone_bell_ringing",
        "telephone_dialing_dtmf",
        "busy_signal",
    }
)
_VOCAL_NON_SPEECH_CLASSES = frozenset({"laughter", "cough", "sneeze"})


def _normalise_sound_class(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def activity_profile_for_class(sound_class: str | None) -> dict[str, Any]:
    """Return the detector contract for one class without inventing thresholds."""

    original = str(sound_class or "")
    name = _normalise_sound_class(sound_class)
    if name == "speech_playback":
        return {
            "schema": ACTIVITY_PROFILE_SCHEMA,
            "sound_class": original,
            "activity_family": "speech",
            "energy_domain": "speech_band",
            "filter": dict(SPEECH_FILTER_DEFAULTS),
            "detector": dict(SPEECH_DETECTOR_DEFAULTS),
            "minimum_audible_duration_s": SPEECH_MIN_AUDIBLE_S,
            "maximum_clip_duration_s": SPEECH_MAX_CLIP_S,
            "repeat_policy": "first_utterance_transcript_unique",
            "threshold": {
                "relative_peak_db": -25.0,
                "calibration": "placeholder",
                "status": "acoustic_detector_candidate_not_human_certified",
            },
        }
    if name in _ANIMAL_SOUND_CLASSES or name.startswith(("dog_", "cat_")):
        return {
            "schema": ACTIVITY_PROFILE_SCHEMA,
            "sound_class": original,
            "activity_family": "animal_call",
            "energy_domain": "full_band",
            "filter": {
                "mode": "full_band",
                "highpass_hz": None,
                "bandpass_hz": None,
            },
            "detector": {
                "level": "frame_energy",
                "threshold_db": None,
                "relative_to": "noise_floor",
                "calibration": "placeholder",
            },
            "minimum_audible_duration_s": 0.5,
            "maximum_clip_duration_s": None,
            "repeat_policy": "event_count",
            "threshold": {
                "value": None,
                "units": "dB",
                "calibration": "placeholder",
                "status": "not_measured",
            },
        }
    if name in _SHORT_PROMPT_CLASSES:
        return {
            "schema": ACTIVITY_PROFILE_SCHEMA,
            "sound_class": original,
            "activity_family": "short_prompt",
            "energy_domain": "full_band",
            "filter": {
                "mode": "full_band",
                "highpass_hz": None,
                "bandpass_hz": None,
            },
            "detector": {
                "level": "frame_energy",
                "threshold_db": None,
                "relative_to": "noise_floor",
                "calibration": "placeholder",
            },
            "minimum_audible_duration_s": None,
            "maximum_clip_duration_s": None,
            "repeat_policy": "allowed_and_counted_per_event",
            "threshold": {
                "value": None,
                "units": "dB",
                "calibration": "placeholder",
                "status": "not_measured",
            },
        }
    if name in _DEVICE_SOUND_CLASSES:
        return {
            "schema": ACTIVITY_PROFILE_SCHEMA,
            "sound_class": original,
            "activity_family": "device_continuous",
            "energy_domain": "full_band",
            "filter": {
                "mode": "full_band",
                "highpass_hz": None,
                "bandpass_hz": None,
            },
            "detector": {
                "level": "frame_energy",
                "threshold_db": None,
                "relative_to": "noise_floor",
                "calibration": "placeholder",
            },
            "minimum_audible_duration_s": None,
            "maximum_clip_duration_s": None,
            "coverage_requirement": "source_activity_covers_query_window",
            "repeat_policy": "event_count_if_short_prompt",
            "threshold": {
                "value": None,
                "units": "dB",
                "calibration": "placeholder",
                "status": "not_measured",
            },
        }
    if name in _VOCAL_NON_SPEECH_CLASSES:
        return {
            "schema": ACTIVITY_PROFILE_SCHEMA,
            "sound_class": original,
            "activity_family": "vocal_non_speech",
            "energy_domain": "full_band",
            "filter": {
                "mode": "full_band",
                "highpass_hz": None,
                "bandpass_hz": None,
            },
            "detector": {
                "level": "frame_energy",
                "threshold_db": None,
                "relative_to": "noise_floor",
                "calibration": "placeholder",
            },
            "minimum_audible_duration_s": 0.5,
            "maximum_clip_duration_s": None,
            "repeat_policy": "event_count",
            "threshold": {
                "value": None,
                "units": "dB",
                "calibration": "placeholder",
                "status": "not_measured",
            },
        }
    return {
        "schema": ACTIVITY_PROFILE_SCHEMA,
        "sound_class": original or "unknown",
        "activity_family": "unknown",
        "energy_domain": "unknown",
        "filter": None,
        "detector": None,
        "minimum_audible_duration_s": None,
        "maximum_clip_duration_s": None,
        "coverage_requirement": None,
        "repeat_policy": None,
        "threshold": {
            "value": None,
            "units": None,
            "calibration": "placeholder",
            "status": "unknown",
        },
    }


def _as_mono_float(samples: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[1] == 0:
            raise PrepareError("audio has zero channels")
        array = array.mean(axis=1)
    if array.ndim != 1:
        raise PrepareError("audio must be one-dimensional or [frames, channels]")
    if array.size == 0:
        raise PrepareError("audio is empty")
    if not np.isfinite(array).all():
        raise PrepareError("audio contains non-finite samples")
    return array


def _zero_phase_filter(
    samples: np.ndarray,
    rate: int,
    *,
    kind: str,
    cutoff: float | tuple[float, float],
    order: int,
) -> np.ndarray:
    if samples.size < 3:
        raise PrepareError("audio is too short for speech-band filtering")
    try:
        sos = butter(order, cutoff, fs=rate, btype=kind, output="sos")
    except ValueError as error:
        raise PrepareError(f"invalid {kind} filter: {error}") from error
    try:
        return sosfiltfilt(sos, samples)
    except ValueError:
        # Very short synthetic fixtures can be shorter than scipy's default
        # padding. Keep the real zero-phase filter whenever possible.
        padlen = min(samples.size - 1, max(0, 3 * (2 * len(sos) + 1)))
        if padlen > 0:
            try:
                return sosfiltfilt(sos, samples, padlen=padlen)
            except ValueError:
                pass
        return sosfilt(sos, samples)


def _speech_band_analysis(
    samples: np.ndarray | Sequence[float],
    rate: int,
    *,
    highpass_hz: float = SPEECH_FILTER_DEFAULTS["highpass_hz"],
    bandpass_low_hz: float = SPEECH_FILTER_DEFAULTS["bandpass_low_hz"],
    bandpass_high_hz: float = SPEECH_FILTER_DEFAULTS["bandpass_high_hz"],
    filter_order: int = int(SPEECH_FILTER_DEFAULTS["order"]),
    window_s: float = SPEECH_DETECTOR_DEFAULTS["window_s"],
    hop_s: float = SPEECH_DETECTOR_DEFAULTS["hop_s"],
    relative_peak_db: float = SPEECH_DETECTOR_DEFAULTS["relative_peak_db"],
) -> tuple[np.ndarray, dict[str, Any]]:
    x = _as_mono_float(samples)
    rate = int(rate)
    if rate <= 0:
        raise PrepareError("sample rate must be positive")
    if not 0 < highpass_hz < rate / 2:
        raise PrepareError("highpass cutoff must be below Nyquist")
    if not 0 < bandpass_low_hz < bandpass_high_hz < rate / 2:
        raise PrepareError("speech band must be inside Nyquist")
    if window_s <= 0 or hop_s <= 0:
        raise PrepareError("detector window and hop must be positive")
    window_samples = max(1, int(round(window_s * rate)))
    hop_samples = max(1, int(round(hop_s * rate)))
    if x.size < window_samples:
        raise PrepareError("audio is shorter than one detector window")
    filter_config = {
        "highpass_hz": float(highpass_hz),
        "bandpass_low_hz": float(bandpass_low_hz),
        "bandpass_high_hz": float(bandpass_high_hz),
        "order": int(filter_order),
        "phase": "zero_phase",
    }
    detector_config = {
        "window_s": float(window_s),
        "hop_s": float(hop_s),
        "window_samples": window_samples,
        "hop_samples": hop_samples,
        "relative_peak_db": float(relative_peak_db),
        "level": "frame_rms",
        "active_duration": "union_of_active_windows",
    }
    highpassed = _zero_phase_filter(
        x, rate, kind="highpass", cutoff=highpass_hz, order=filter_order
    )
    band = _zero_phase_filter(
        highpassed,
        rate,
        kind="bandpass",
        cutoff=(bandpass_low_hz, bandpass_high_hz),
        order=filter_order,
    )
    starts = np.arange(
        0, x.size - window_samples + 1, hop_samples, dtype=np.int64
    )
    cumulative = np.concatenate(
        [np.array([0.0]), np.cumsum(band * band, dtype=np.float64)]
    )
    levels = np.sqrt(
        np.maximum(
            0.0,
            (cumulative[starts + window_samples] - cumulative[starts])
            / window_samples,
        )
    )
    peak = float(levels.max()) if levels.size else 0.0
    threshold = (
        peak * 10 ** (float(relative_peak_db) / 20.0)
        if peak > 1e-12
        else 0.0
    )
    active = levels >= threshold if peak > 1e-12 else np.zeros_like(levels, dtype=bool)
    base: dict[str, Any] = {
        "source_rate_hz": rate,
        "source_sample_count": int(x.size),
        "filter": filter_config,
        "detector": detector_config,
        "peak_frame_rms": peak,
        "threshold_frame_rms": float(threshold),
        "threshold_db_relative_peak": float(relative_peak_db),
        "source_activity_intervals_s": [],
        "source_activity_duration_s": 0.0,
        "source_activity_interval_count": 0,
        "active_interval_count": 0,
        "active_window_count": 0,
        "audible_start_sample": None,
        "audible_end_sample_exclusive": None,
        "audible_start_s": None,
        "audible_end_s": None,
        "audible_span_s": 0.0,
        "accepted_by_default_detector": False,
        "calibration": "placeholder",
    }
    if not active.any():
        return highpassed, base
    ids = np.flatnonzero(active)
    cuts = np.r_[0, np.flatnonzero(np.diff(ids) > 1) + 1, len(ids)]
    intervals: list[tuple[int, int]] = []
    for index in range(len(cuts) - 1):
        start = int(starts[ids[cuts[index]]])
        end = min(
            int(x.size),
            int(starts[ids[cuts[index + 1] - 1]]) + window_samples,
        )
        intervals.append((start, end))
    first, last = intervals[0][0], intervals[-1][1]
    active_duration_s = sum(end - start for start, end in intervals) / rate
    audible_span_s = (last - first) / rate
    source_intervals = [
        {
            "start_sample": int(start),
            "end_sample_exclusive": int(end),
            "start_s": float(start / rate),
            "end_s": float(end / rate),
        }
        for start, end in intervals
    ]
    pre_roll = x[:first]
    pre_roll_band = band[:first]
    db = lambda value: float(20.0 * np.log10(max(float(value), 1e-12)))
    base.update(
        {
            "source_activity_intervals_s": source_intervals,
            "source_activity_duration_s": float(active_duration_s),
            "active_duration_s": float(active_duration_s),
            "source_activity_interval_count": len(intervals),
            "active_interval_count": len(intervals),
            "active_window_count": int(ids.size),
            "audible_start_sample": first,
            "audible_end_sample_exclusive": last,
            "audible_start_s": float(first / rate),
            "audible_end_s": float(last / rate),
            "audible_span_s": float(audible_span_s),
            "accepted_by_default_detector": bool(
                active_duration_s >= SPEECH_MIN_AUDIBLE_S
                and audible_span_s <= SPEECH_MAX_CLIP_S
            ),
            "pre_roll_rms_dbfs": db(
                np.sqrt(np.mean(pre_roll * pre_roll)) if pre_roll.size else 0.0
            ),
            "pre_roll_speech_band_rms_dbfs": db(
                np.sqrt(np.mean(pre_roll_band * pre_roll_band))
                if pre_roll_band.size
                else 0.0
            ),
        }
    )
    return highpassed, base


def measure_speech_band(
    samples: np.ndarray | Sequence[float],
    rate: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Measure source activity after the speech highpass and bandpass."""

    _highpassed, facts = _speech_band_analysis(samples, rate, **kwargs)
    return facts


def prepare_speech_clip(
    samples: np.ndarray | Sequence[float],
    rate: int,
    *,
    target_rate_hz: int = TARGET_RATE_HZ,
    min_audible_s: float = SPEECH_MIN_AUDIBLE_S,
    max_clip_s: float = SPEECH_MAX_CLIP_S,
    max_duration_s: float | None = None,
    crop_guard_s: float = SPEECH_CROP_GUARD_S,
    normalize_peak: bool = False,
    target_peak_dbfs: float = TARGET_PEAK_DBFS,
    **detector_kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Filter and crop one speech clip while preserving internal pauses."""

    if max_duration_s is not None:
        max_clip_s = float(max_duration_s)
    if min_audible_s < 0 or max_clip_s <= 0 or crop_guard_s < 0:
        raise PrepareError("invalid speech duration or crop guard")
    x = _as_mono_float(samples)
    highpassed, measurement = _speech_band_analysis(x, rate, **detector_kwargs)
    first = measurement.get("audible_start_sample")
    last = measurement.get("audible_end_sample_exclusive")
    active_duration_s = float(measurement.get("source_activity_duration_s") or 0.0)
    audible_span_s = float(measurement.get("audible_span_s") or 0.0)
    if first is None or last is None:
        raise PrepareError("speech band has no active frame")
    if active_duration_s < float(min_audible_s):
        raise PrepareError(
            f"speech active duration {active_duration_s:.3f}s is below "
            f"minimum {float(min_audible_s):.3f}s"
        )
    if audible_span_s > float(max_clip_s):
        raise PrepareError(
            f"speech audible span {audible_span_s:.3f}s exceeds "
            f"maximum {float(max_clip_s):.3f}s"
        )
    source_rate = int(rate)
    requested_guard_samples = int(round(float(crop_guard_s) * source_rate))
    crop_start = max(0, int(first) - requested_guard_samples)
    crop_end = min(x.size, int(last) + requested_guard_samples)
    unguarded_duration_s = (int(last) - int(first)) / source_rate
    effective_guard_s = float(crop_guard_s)
    if (crop_end - crop_start) / source_rate > float(max_clip_s):
        # Keep all detected speech and remove only the optional guard when it
        # would be the part that violates the hard five-second limit.
        if unguarded_duration_s <= float(max_clip_s):
            crop_start, crop_end = int(first), int(last)
            effective_guard_s = 0.0
        else:
            raise PrepareError(
                f"speech crop {(crop_end - crop_start) / source_rate:.3f}s "
                f"exceeds maximum {float(max_clip_s):.3f}s"
            )
    cropped = highpassed[crop_start:crop_end]
    if cropped.size == 0:
        raise PrepareError("speech crop is empty")
    target_rate_hz = int(target_rate_hz)
    if target_rate_hz <= 0:
        raise PrepareError("target sample rate must be positive")
    output = cropped
    resample_ratio = "1/1"
    antialiased = False
    if source_rate != target_rate_hz:
        ratio = Fraction(target_rate_hz, source_rate).limit_denominator(10000)
        output = resample_poly(output, ratio.numerator, ratio.denominator)
        resample_ratio = f"{ratio.numerator}/{ratio.denominator}"
        antialiased = True
    output = np.asarray(output, dtype=np.float64)
    if normalize_peak:
        peak = float(np.abs(output).max())
        if peak <= 0.0:
            raise PrepareError("filtered speech is digital silence")
        gain = (10 ** (float(target_peak_dbfs) / 20.0)) / peak
        output = output * gain
        applied_gain_db = float(20.0 * np.log10(gain))
    else:
        applied_gain_db = 0.0
    peak_abs = float(np.max(np.abs(output)))
    if peak_abs > 1.0:
        raise PrepareError(
            f"highpass output exceeds PCM full scale without normalization: peak_abs={peak_abs:.12g}"
        )
    facts = dict(measurement)
    facts.update(
        {
            "source_crop_start_sample": int(crop_start),
            "source_crop_end_sample_exclusive": int(crop_end),
            "source_offset_s": float(crop_start / source_rate),
            "source_crop_duration_s": float((crop_end - crop_start) / source_rate),
            "crop_guard_s_requested": float(crop_guard_s),
            "crop_guard_s_applied": effective_guard_s,
            "target_rate_hz": target_rate_hz,
            "resample_ratio": resample_ratio,
            "antialiased": antialiased,
            "normalization_applied": bool(normalize_peak),
            "applied_gain_db": applied_gain_db,
            "prepared_sample_count": int(output.size),
            "prepared_duration_s": float(output.size / target_rate_hz),
            "prepared_peak_dbfs": float(
                20.0 * np.log10(max(float(np.abs(output).max()), 1e-12))
            ),
            "operation": "speech_highpass_with_band_detected_crop_v1",
            "activity_filter": dict(measurement["filter"]),
            "filter": {key: measurement["filter"][key] for key in
                       ("highpass_hz", "order", "phase")},
        }
    )
    if facts["prepared_duration_s"] > float(max_clip_s) + 1e-6:
        raise PrepareError(
            f"prepared duration {facts['prepared_duration_s']:.3f}s exceeds "
            f"maximum {float(max_clip_s):.3f}s"
        )
    return output, facts


def measure_speech_activity(
    samples: np.ndarray | Sequence[float],
    rate: int,
    **kwargs: Any,
) -> dict[str, Any]:
    return measure_speech_band(samples, rate, **kwargs)


def make_prepared_audio_id(
    source_asset_id: str,
    *,
    source_sha256: str,
    facts: Mapping[str, Any],
) -> str:
    """Derive a new ID from source identity and all result-changing settings."""

    identity = {
        "source_asset_id": str(source_asset_id),
        "source_sha256": str(source_sha256),
        "operation": facts.get("operation"),
        "source_crop_start_sample": facts.get("source_crop_start_sample"),
        "source_crop_end_sample_exclusive": facts.get(
            "source_crop_end_sample_exclusive"
        ),
        "filter": facts.get("filter"),
        "activity_filter": facts.get("activity_filter"),
        "detector": facts.get("detector"),
        "target_rate_hz": facts.get("target_rate_hz"),
    }
    if facts.get("normalization_applied"):
        identity["normalization"] = {"applied_gain_db": facts.get("applied_gain_db")}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"prepared_speech_band_{digest}_v1"


prepared_speech_id = make_prepared_audio_id


def _registry_source_value(row: Mapping[str, Any]) -> str | None:
    provenance = row.get("provenance")
    origin = provenance.get("origin") if isinstance(provenance, Mapping) else None
    match = re.search(r"(?:^|;)\s*source=([^;]+)", str(origin or ""))
    if match:
        return match.group(1).strip()
    for key in ("source_pcm_path", "source_path", "source"):
        value = row.get(key)
        if value:
            return str(value)
    dry = row.get("dry_audio")
    if isinstance(dry, Mapping) and dry.get("uri"):
        return str(dry["uri"])
    return None


def _file_uri_path(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(value)


def bridge_speech_metadata(
    registry_row: Mapping[str, Any],
    source_library_root: str | Path,
) -> dict[str, Any]:
    """Bridge only explicit fields from the original clip.json sidecar."""

    root = Path(source_library_root)
    raw_source = _registry_source_value(registry_row)
    source_value = raw_source or ""
    source_path = _file_uri_path(source_value)
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()
    try:
        source_relative = source_path.relative_to(root.resolve()).as_posix()
    except ValueError:
        source_relative = None
    metadata_path = source_path.with_suffix(".json")
    metadata: Mapping[str, Any] = {}
    metadata_status = "missing"
    if metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, Mapping):
            metadata = loaded
            metadata_status = "bridged"
        else:
            metadata_status = "invalid"
    def explicit_text(key: str) -> str | None:
        value = metadata.get(key)
        if value is None:
            return None
        value = str(value).strip()
        return value or None
    gender = explicit_text("gender") or "unknown"
    transcript = explicit_text("transcript")
    speaker_id = explicit_text("speaker_id")
    utterance_id = explicit_text("utterance_id")
    split = explicit_text("split")
    return {
        "source_pcm_path": str(source_path),
        "source_relative": source_relative,
        "source_metadata_path": str(metadata_path),
        "metadata_status": metadata_status,
        "metadata_fields": sorted(str(key) for key in metadata),
        "is_vctk": any(
            part.lower().startswith("vctk_") for part in source_path.parts
        ),
        "gender": gender,
        "gender_source": (
            f"{metadata_path}#/gender" if "gender" in metadata else None
        ),
        "transcript": transcript,
        "transcript_source": (
            f"{metadata_path}#/transcript" if "transcript" in metadata else None
        ),
        "speaker_id": speaker_id,
        "utterance_id": utterance_id,
        "split": split,
    }


def _registry_payload(
    registry: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if isinstance(registry, (str, Path)):
        path = Path(registry).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise PrepareError("sound registry must be a JSON object")
        return dict(payload), str(path)
    if not isinstance(registry, Mapping):
        raise PrepareError("sound registry must be a path or mapping")
    return dict(registry), None


def _write_wav_no_clobber(
    path: Path,
    samples: np.ndarray,
    rate: int,
) -> str:
    if path.exists():
        raise FileExistsError(path)
    return _write_wav_mono(path, samples, rate)


def prepare_speech_registry(
    registry: str | Path | Mapping[str, Any],
    output_root: str | Path,
    *,
    source_library_root: str | Path,
    prepared_set_id: str = "speech_band_prepared_20260906_v1",
    vctk_only: bool = True,
    target_rate_hz: int = TARGET_RATE_HZ,
    min_audible_s: float = SPEECH_MIN_AUDIBLE_S,
    max_clip_s: float = SPEECH_MAX_CLIP_S,
    crop_guard_s: float = SPEECH_CROP_GUARD_S,
    normalize_peak: bool = False,
    no_clobber: bool = True,
) -> dict[str, Any]:
    """Build a fresh prepared speech set without touching source or registry."""

    payload, registry_path = _registry_payload(registry)
    assets = payload.get("sound_assets")
    if not isinstance(assets, list):
        raise PrepareError("sound registry has no sound_assets list")
    output = Path(output_root).resolve()
    manifest_path = output / "prepared_manifest.json"
    if no_clobber and manifest_path.exists():
        raise FileExistsError(manifest_path)
    source_root = Path(source_library_root).resolve()
    speech_rows = [
        row
        for row in assets
        if isinstance(row, Mapping)
        and _normalise_sound_class(row.get("semantic_sound_class"))
        == "speech_playback"
    ]
    clips: list[dict[str, Any]] = []
    to_write: list[tuple[Path, np.ndarray, dict[str, Any]]] = []
    for index, row in enumerate(speech_rows):
        bridge = bridge_speech_metadata(row, source_root)
        source_asset_id = str(
            row.get("sound_asset_id") or f"speech_registry_row_{index:04d}"
        )
        base = {
            "source_asset_id": source_asset_id,
            "source_registry_revision": row.get("revision"),
            "source_pcm_path": bridge["source_pcm_path"],
            "source_relative": bridge["source_relative"],
            "source_metadata_path": bridge["source_metadata_path"],
            "metadata_status": bridge["metadata_status"],
            "is_vctk": bridge["is_vctk"],
            "gender": bridge["gender"],
            "gender_source": bridge["gender_source"],
            "transcript": bridge["transcript"],
            "transcript_source": bridge["transcript_source"],
            "speaker_id": bridge["speaker_id"],
            "utterance_id": bridge["utterance_id"],
            "split": bridge["split"],
            "activity_profile": activity_profile_for_class("speech_playback"),
        }
        if vctk_only and not bridge["is_vctk"]:
            clips.append(
                {
                    **base,
                    "status": "excluded",
                    "reason": "non_vctk_source_excluded_by_candidate_policy",
                    "human_review": {"status": "not_requested"},
                }
            )
            continue
        source_path = Path(bridge["source_pcm_path"])
        if not source_path.is_file():
            clips.append(
                {
                    **base,
                    "status": "rejected",
                    "reason": "source_pcm_missing",
                    "human_review": {"status": "not_requested"},
                }
            )
            continue
        source_bytes = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        try:
            samples, source_rate = _read_wav_mono(source_path)
            measurement = measure_speech_band(samples, source_rate)
        except (OSError, ValueError, wave.Error, PrepareError) as error:
            clips.append(
                {
                    **base,
                    "source_sha256": source_sha256,
                    "status": "rejected",
                    "reason": f"measurement_failed:{error}",
                    "human_review": {"status": "not_requested"},
                }
            )
            continue
        try:
            prepared, facts = prepare_speech_clip(
                samples,
                source_rate,
                target_rate_hz=target_rate_hz,
                min_audible_s=min_audible_s,
                max_clip_s=max_clip_s,
                crop_guard_s=crop_guard_s,
                normalize_peak=normalize_peak,
            )
        except (PrepareError, ValueError) as error:
            clips.append(
                {
                    **base,
                    "source_sha256": source_sha256,
                    "status": "rejected",
                    "reason": f"qualification_failed:{error}",
                    "measurement": measurement,
                    "source_activity": {
                        "audible_span_s": measurement.get("audible_span_s"),
                        "active_duration_s": measurement.get(
                            "source_activity_duration_s"
                        ),
                        "active_interval_count": measurement.get(
                            "source_activity_interval_count"
                        ),
                        "intervals": measurement.get(
                            "source_activity_intervals_s", []
                        ),
                    },
                    "human_review": {"status": "not_requested"},
                }
            )
            continue
        prepared_id = make_prepared_audio_id(
            source_asset_id,
            source_sha256=source_sha256,
            facts=facts,
        )
        relative = Path("speech_playback") / prepared_id / "clip.wav"
        target = output / relative
        source_activity = {
            "audible_span_s": facts.get("audible_span_s"),
            "active_duration_s": facts.get("source_activity_duration_s"),
            "active_interval_count": facts.get("source_activity_interval_count"),
            "intervals": facts.get("source_activity_intervals_s", []),
        }
        record = {
            **base,
            "prepared_audio_id": prepared_id,
            "prepared": relative.as_posix(),
            "source_sha256": source_sha256,
            "source_crop_start_sample": facts["source_crop_start_sample"],
            "source_crop_end_sample_exclusive": facts[
                "source_crop_end_sample_exclusive"
            ],
            "source_offset_s": facts["source_offset_s"],
            "source_crop_duration_s": facts["source_crop_duration_s"],
            "filter_parameters": facts["filter"],
            "activity_filter_parameters": facts["activity_filter"],
            "detector_parameters": facts["detector"],
            "source_activity": source_activity,
            "facts": facts,
            "status": "prepared",
            "reason": "machine_detector_candidate_pending_human_review",
            "human_review": {
                "status": "pending_human",
                "reviewer": None,
                "heard": None,
                "consonant_preserved": None,
                "notes": None,
            },
        }
        clips.append(record)
        to_write.append((target, prepared, record))
    collisions = [str(path) for path, _samples, _record in to_write if path.exists()]
    if no_clobber and collisions:
        raise FileExistsError("prepared output already exists: " + ", ".join(collisions[:5]))
    output.mkdir(parents=True, exist_ok=True)
    for target, prepared, record in to_write:
        record["prepared_sha256"] = _write_wav_no_clobber(
            target, prepared, int(target_rate_hz)
        )
    counts: dict[str, int] = {}
    gender_counts: dict[str, int] = {}
    for clip in clips:
        status = str(clip.get("status"))
        counts[status] = counts.get(status, 0) + 1
        if status == "prepared":
            gender = str(clip.get("gender") or "unknown")
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
    manifest = {
        "schema": PREPARED_SPEECH_SCHEMA,
        "prepared_set_id": prepared_set_id,
        "status": "research_candidate",
        "source_registry_path": registry_path,
        "source_registry_schema": payload.get("schema"),
        "source_library_root": str(source_root),
        "output_root": str(output),
        "target_rate_hz": int(target_rate_hz),
        "candidate_policy": {
            "vctk_only": bool(vctk_only),
            "max_clip_s": float(max_clip_s),
            "min_audible_s": float(min_audible_s),
        },
        "filter": {key: SPEECH_FILTER_DEFAULTS[key] for key in ("highpass_hz", "order", "phase")},
        "activity_filter": dict(SPEECH_FILTER_DEFAULTS),
        "detector": dict(SPEECH_DETECTOR_DEFAULTS),
        "qualification": {
            "max_clip_s": float(max_clip_s),
            "min_audible_s": float(min_audible_s),
            "human_certification": "pending",
            "calibration": "placeholder",
        },
        "counts": counts,
        "counts_by_gender": gender_counts,
        "measurement_summary": {
            "speech_registry_rows": len(speech_rows),
            "prepared_rows": counts.get("prepared", 0),
            "detector_candidates_are_not_human_review": True,
            "human_review_status": "pending",
        },
        "non_vctk_excluded_count": counts.get("excluded", 0),
        "claims": {
            "original_pcm_untouched": True,
            "old_episode_offsets_untouched": True,
            "registry_untouched": True,
            "source_activity_only": True,
            "listener_audibility_not_measured_here": True,
            "wet_tail_not_measured_here": True,
        },
        "clips": clips,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_nonverbal_source_inventory(
    csv_path: str | Path,
    *,
    labels: Sequence[str] = ("Sneeze", "Laughter", "Cough"),
) -> dict[str, Any]:
    """Count independent source names for the non-speech inventory."""

    path = Path(csv_path).resolve()
    counts: dict[str, int] = {}
    unique_source_names: dict[str, list[str]] = {}
    if not path.is_file():
        return {
            "schema": "avengine_nonverbal_source_inventory_v1",
            "path": str(path),
            "status": "missing",
            "label_exact_match_counts": {},
            "unique_source_names": {},
        }
    rows_by_label = {str(label): set() for label in labels}
    row_counts = {str(label): 0 for label in labels}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("labels") or "").strip()
            if label not in rows_by_label:
                continue
            row_counts[label] += 1
            source = str(row.get("fname") or "").strip()
            if source:
                rows_by_label[label].add(source)
    for label, names in rows_by_label.items():
        counts[label] = row_counts[label]
        unique_source_names[label] = sorted(names)
    return {
        "schema": "avengine_nonverbal_source_inventory_v1",
        "path": str(path),
        "status": "measured",
        "labels": list(labels),
        "label_exact_match_counts": counts,
        "independent_source_counts": {
            label: len(names) for label, names in unique_source_names.items()
        },
        "unique_source_names": unique_source_names,
        "note": "Counts are independent source files, not repeated event uses.",
    }


def load_prepared_speech_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PREPARED_SPEECH_SCHEMA:
        raise PrepareError(
            f"unexpected prepared speech schema {payload.get('schema')!r}"
        )
    return payload


def iter_prepared_speech_clips(
    manifest: str | Path | Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(manifest, (str, Path)):
        payload = load_prepared_speech_manifest(manifest)
    else:
        payload = dict(manifest)
        if payload.get("schema") != PREPARED_SPEECH_SCHEMA:
            raise PrepareError(
                f"unexpected prepared speech schema {payload.get('schema')!r}"
            )
    return [
        dict(row)
        for row in payload.get("clips", [])
        if isinstance(row, Mapping) and row.get("status") == "prepared"
    ]


def listening_sample_records(
    manifest: str | Path | Mapping[str, Any],
    *,
    count: int = 10,
    seed: int = 20260906,
) -> list[dict[str, Any]]:
    """Select reviewable samples; every human field stays explicitly pending."""

    clips = iter_prepared_speech_clips(manifest)
    if count < 0 or count > len(clips):
        raise PrepareError(f"requested {count} samples from {len(clips)} clips")
    ordered = sorted(clips, key=lambda row: str(row["prepared_audio_id"]))
    rng = random.Random(seed)
    selected = rng.sample(ordered, count)
    selected.sort(key=lambda row: str(row["prepared_audio_id"]))
    rows = []
    for index, clip in enumerate(selected, 1):
        rows.append(
            {
                "sample_index": index,
                "prepared_audio_id": clip["prepared_audio_id"],
                "prepared_path": clip["prepared"],
                "gender": clip.get("gender", "unknown"),
                "speaker_id": clip.get("speaker_id"),
                "utterance_id": clip.get("utterance_id"),
                "transcript": clip.get("transcript"),
                "source_activity_duration_s": clip.get(
                    "source_activity", {}
                ).get("active_duration_s"),
                "source_audible_span_s": clip.get("source_activity", {}).get(
                    "audible_span_s"
                ),
                "review_status": "pending_human",
                "reviewer": None,
                "heard": None,
                "consonant_preserved": None,
                "notes": None,
            }
        )
    return rows
