"""Measure one dry-sound clip and say, in words, what is wrong with it.

The sound library is the one place in this pipeline where material comes
from outside, so it is the one place where a person could quietly hand us
a clip that breaks a benchmark we cannot rebuild cheaply. This module
turns "is this clip usable" into arithmetic: level, clipping, DC offset,
background noise, and the decay tail that betrays a recording made in a
room rather than a dry one.

Every threshold below is calibrated against the first real delivery -
140 clips from FSD50K, measured 2026-08-30 - not invented, and the
calibration also killed two heuristics that looked reasonable and were
not:

* A flat "low dynamic range means noisy" rule flagged nineteen clips.
  Nearly all were continuous sounds - air conditioning, a running
  blender, a busy signal - which have no quiet passage by nature. The
  rule now applies only to clips whose energy actually pauses (the
  active-frame ratio separates them cleanly: continuous clips sit at
  0.93-1.00, impulsive ones far below).
* "Peak equals full scale means clipped" flagged eleven clips, of which
  the majority touch full scale on one or two samples - that is peak
  normalisation, not distortion. Clipping is now judged by how much of
  the clip sits at the ceiling and in how many separate runs.

The decay check is deliberately advisory. A doorbell chime and a bell
telephone ring for a second on their own, so a long tail is evidence of
reverberation, not proof of it; the report says "listen to this one"
rather than failing it, and a human decides.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "avengine_sound_clip_qc_v1"

# levels
_QUIET_RMS_DBFS = -45.0
_NOISE_FLOOR_DBFS = -40.0
# clipping: fraction of samples pinned at full scale, and how many runs
_CLIP_FRACTION_SERIOUS = 0.001
_CLIP_RUNS_MILD = 10
# a constant offset this large distorts every convolution downstream
_DC_OFFSET_MILD = 0.01
_DC_OFFSET_SERIOUS = 0.05
# an impulsive clip still ringing this long after its peak was probably
# recorded in a room, or is a resonant object - either way, a human listens
_DECAY_T20_SUSPECT_S = 1.0
# clips whose energy never pauses are continuous by nature, not defective
_CONTINUOUS_ACTIVE_RATIO = 0.8
_DURATION_MIN_S = 0.3
_DURATION_MAX_S = 30.0


class SoundQCError(ValueError):
    pass


def read_mono(path: Path) -> tuple[np.ndarray, int, int, int]:
    """Mono float samples, sample rate, channel count, sample width.

    Handles the PCM widths the wave module exposes, including 24-bit,
    which it hands over as raw bytes rather than refusing outright.
    """

    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(frames)
    except (wave.Error, EOFError, OSError) as error:
        raise SoundQCError(f"{type(error).__name__}: {error}") from error

    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        samples = (
            np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = (
            packed[:, 0]
            | (packed[:, 1] << 8)
            | (packed[:, 2].astype(np.int8).astype(np.int32) << 16)
        )
        samples = value.astype(np.float32) / 8388608.0
    else:
        raise SoundQCError(f"unhandled sample width {width}")

    if channels > 1:
        usable = len(samples) // channels * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return samples, rate, channels, width


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def measure(samples: np.ndarray, rate: int, width: int) -> dict[str, Any]:
    hop = max(1, int(0.020 * rate))
    usable = len(samples) // hop * hop
    if usable == 0:
        raise SoundQCError("clip is shorter than one analysis frame")
    frames = samples[:usable].reshape(-1, hop)
    frame_rms = np.sqrt((frames**2).mean(axis=1) + 1e-20)
    loudest = float(frame_rms.max())

    # A clip whose energy never drops far below its peak is a continuous
    # sound; quiet-passage statistics do not apply to it.
    active_ratio = float((frame_rms > loudest * 0.1).mean())
    peak = float(np.abs(samples).max())

    ceiling = 1.0 - 1.0 / (2 ** (8 * width - 1))
    at_ceiling = np.abs(samples) >= ceiling
    pinned = int(at_ceiling.sum())
    runs = 0
    if pinned:
        indices = np.flatnonzero(at_ceiling)
        runs = int((np.diff(indices) > 1).sum() + 1)

    decay_t20_s = None
    if active_ratio < _CONTINUOUS_ACTIVE_RATIO:
        peak_frame = int(np.argmax(frame_rms))
        after = frame_rms[peak_frame:]
        target = loudest * 10 ** (-20 / 20)
        below = np.flatnonzero(after < target)
        if below.size:
            decay_t20_s = round(float(below[0] * hop / rate), 3)

    return {
        "duration_s": round(len(samples) / rate, 3),
        "peak": round(peak, 4),
        "rms_dbfs": round(_db(float(np.sqrt((samples**2).mean()))), 1),
        "noise_floor_dbfs": round(
            _db(float(np.percentile(frame_rms, 10))), 1
        ),
        "dc_offset": round(float(samples.mean()), 5),
        "active_frame_ratio": round(active_ratio, 3),
        "continuous": active_ratio >= _CONTINUOUS_ACTIVE_RATIO,
        "samples_at_full_scale": pinned,
        "full_scale_runs": runs,
        "full_scale_fraction": round(pinned / max(len(samples), 1), 5),
        "decay_to_minus20db_s": decay_t20_s,
    }


def judge(measured: dict[str, Any], *, channels: int) -> list[dict[str, str]]:
    """Findings in words. severity is 'fail' (unusable) or 'warn' (look)."""

    findings: list[dict[str, str]] = []

    def add(severity: str, name: str, reason: str) -> None:
        findings.append({"severity": severity, "name": name, "reason_zh": reason})

    if channels != 1:
        add("warn", "channels", f"{channels} 声道,管线要单声道(入库后我统一转)")
    duration = measured["duration_s"]
    if duration < _DURATION_MIN_S:
        add("fail", "duration", f"只有 {duration}s,太短了")
    elif duration > _DURATION_MAX_S:
        add("warn", "duration", f"{duration}s 偏长,建议剪到 10s 以内")

    if measured["peak"] < 1e-4:
        add("fail", "silent", "整条几乎没有声音")
    if measured["full_scale_fraction"] >= _CLIP_FRACTION_SERIOUS:
        add(
            "fail",
            "clipping",
            f"削波严重:{measured['full_scale_fraction']*100:.1f}% 的采样点顶到最大值"
            f"({measured['full_scale_runs']} 段),声音已经失真,建议换一条",
        )
    elif measured["full_scale_runs"] >= _CLIP_RUNS_MILD:
        add(
            "warn",
            "clipping",
            f"轻微削波:有 {measured['full_scale_runs']} 段顶到最大值,能换就换",
        )

    if measured["rms_dbfs"] < _QUIET_RMS_DBFS:
        add(
            "warn",
            "level",
            f"整体音量很小({measured['rms_dbfs']} dBFS),放大后底噪会跟着上来",
        )
    offset = abs(measured["dc_offset"])
    if offset >= _DC_OFFSET_SERIOUS:
        add(
            "fail",
            "dc_offset",
            f"直流偏移 {measured['dc_offset']}(整条波形整体偏离零点),这条文件是坏的",
        )
    elif offset >= _DC_OFFSET_MILD:
        add("warn", "dc_offset", f"有直流偏移 {measured['dc_offset']},建议做个高通")

    if not measured["continuous"]:
        if measured["noise_floor_dbfs"] > _NOISE_FLOOR_DBFS:
            add(
                "warn",
                "noise_floor",
                f"安静段的底噪有 {measured['noise_floor_dbfs']} dBFS,背景不够干净",
            )
        decay = measured["decay_to_minus20db_s"]
        if decay is not None and decay > _DECAY_T20_SUSPECT_S:
            add(
                "warn",
                "reverb_tail",
                f"声音停下后还拖了 {decay}s 才衰减 20 dB,可能是在房间里录的(带混响),"
                f"也可能是这个东西本身余音长——听一下再定",
            )
    return findings


def audit_clip(path: Path) -> dict[str, Any]:
    try:
        samples, rate, channels, width = read_mono(path)
        measured = measure(samples, rate, width)
    except SoundQCError as error:
        return {
            "schema": SCHEMA,
            "verdict": "fail",
            "findings": [
                {"severity": "fail", "name": "unreadable",
                 "reason_zh": f"这个 wav 读不了:{error}"}
            ],
            "measured": {},
            "sample_rate_hz": None,
            "channel_count": None,
        }
    findings = judge(measured, channels=channels)
    verdict = "fail" if any(f["severity"] == "fail" for f in findings) else (
        "warn" if findings else "pass"
    )
    return {
        "schema": SCHEMA,
        "verdict": verdict,
        "findings": findings,
        "measured": measured,
        "sample_rate_hz": rate,
        "channel_count": channels,
    }


def write_clip_qc(path: Path) -> dict[str, Any]:
    """Audit one clip and drop the report beside it as clip.qc.json."""

    report = audit_clip(path)
    report["clip"] = path.name
    target = path.with_suffix(".qc.json")
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
