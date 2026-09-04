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

import hashlib
import json
import wave
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from avengine.assets.sound_harvest import (
    SPEECH_METADATA_FIELDS,
    speech_metadata_from_mapping,
)

SCHEMA = "avengine_prepared_sound_clip_v1"

TARGET_RATE_HZ = 16000
TARGET_PEAK_DBFS = -3.0
# Content is anything within this much of the clip's own peak; quieter
# head and tail is silence to trim.
_TRIM_FLOOR_DB = 40.0
_TRIM_GUARD_S = 0.030
_FRAME_S = 0.010


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
    gain = (10 ** (target_peak_dbfs / 20)) / peak
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
