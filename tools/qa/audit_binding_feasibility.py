#!/usr/bin/env python3
"""绑定可行性与逐题难度画像审计（research_only）。

这个工具回答的问题
------------------
一道"先用声音认出是谁，再问它怎样"的题，到底能不能答、有多难。2026-09-06 的审阅对
74 题批次逐题复算后发现：四段自建房间里说话人之间在听者处的夹角只有 2 到 15 度，
成片双耳音轨里几乎量不出左右差；大多数"AV 核心"题其他候选的取值和金标一样，
不需要绑定；声停后的答案常常等于最后听到的状态。这些都不是引擎真值算错，而是
布景、出题条件和渲染配置的问题，所以要一个从**交付产物**直接算出来的审计表，
每道题带一份难度画像，供分层报告和准入使用。

输入
----
- ``--facts``：一段 Episode 的归一化事实 ``facts.json``（``unified_catalog.normalize_episode_bundle``
  的输出，delivery 目录里那份）。
- ``--wav``：成片双耳 WAV；缺省时取 ``facts.audio.path``。
- ``--questions``：同一 delivery 目录的 ``questions.json``（可选；给了才做逐题分歧审计）。
- ``--out``：输出 JSON，no-clobber。

输出里有什么
------------
- ``audio``：成片峰值/有效值 dBFS、精确零采样比例、片尾一秒有效值、左右声道相关。
- ``events``：每个声音事件的几何（听者相对方位、距离、与最近竞争者的夹角在说话期间的
  起止/最小/最大）、说话期间谁在动、说话人自身方位扫过多少度、与其他事件的重叠、
  以及**从成片实测**的左右能量差（全带 / 0.5–1.5 kHz / 2–6 kHz）、起点直达声窗内的双耳
  时差与其相关系数、按人头几何算的时差预期。
- ``questions``：每道题的四组难度画像（听 / 看 / 时间推理 / 是否需要绑定）和分歧审计：
  其他候选在被查属性上的取值、金标是否为多数、是否退化、声停后答案能否由声停前趋势外推、
  QA-13 金标离扇区边界多近。
- ``summary``：各标记的计数。

所有阈值都是占位值，等人工校准；输出里 ``thresholds.calibration`` 明确写着这一点。
方位一律是**引擎帧、右为正**（和 ``unified_catalog._listener_azimuth`` 同一公式、同一基向量），
并带 ``azimuth_convention`` 字段；能导入 ``avengine.qa.unified_catalog`` 时会逐帧对账，
把最大差值写进 ``azimuth_formula_crosscheck``。这里的数不是模型成绩，也不是正式准入。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "avengine_qa_binding_feasibility_audit_v1"
AZIMUTH_CONVENTION = "engine_right_positive_deg"
VISIBLE_STATES = ("visible_clear", "visible_occluded")
SECTOR_BOUNDARIES_DEG = (-135.0, -45.0, 45.0, 135.0)

# 题型分组（与 2026-09-06 审阅第 4 节一致）。
BINDING_TYPES = {"QA-01", "QA-02", "QA-03", "QA-08", "QA-12", "QA-13", "QA-16", "QA-17",
                 "QA-19", "QA-20", "QA-21", "QA-22", "QA-24"}  # 13 类，与审阅第 4 节一致
CONDITIONAL_TYPES = {"QA-06", "QA-15", "QA-18"}
AUDIO_CONTROL_TYPES = {"QA-04", "QA-05", "QA-23"}
VISUAL_CONTROL_TYPES = {"QA-07", "QA-09", "QA-10", "QA-11", "QA-14"}

DEFAULT_THRESHOLDS = {
    "theta_static_deg": 30.0,   # 说话期间某一刻与最近竞争者的夹角至少这么大
    "theta_motion_deg": 10.0,   # 或者说话期间这个夹角至少变化这么多
    "ild_min_db": 3.0,          # 成片 2–6 kHz 左右能量差
    "itd_min_ms": 0.2,          # 或者起点直达声窗内的双耳时差
    "sector_margin_deg": 5.0,   # QA-13 金标离扇区边界小于这个值就标出来
    "post_sound_min_displacement_deg": 15.0,  # QA-13 声停后至少要挪这么多才不是"最后听到的方位"
    "distance_margin_m": 0.2,   # 与 unified_catalog 的 QA-15/16 研究边距一致
}
CALIBRATION_NOTE = "placeholder_pending_human_calibration"


# ── 基础工具 ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))


def circular_diff_deg(first: float, second: float) -> float:
    """圆上两个方位的最短夹角，落在 [0, 180]。"""

    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def signed_circular_delta_deg(frm: float, to: float) -> float:
    return (float(to) - float(frm) + 180.0) % 360.0 - 180.0


def sector_of(angle: float) -> str:
    value = (float(angle) + 180.0) % 360.0 - 180.0
    if -45.0 <= value < 45.0:
        return "front"
    if 45.0 <= value < 135.0:
        return "right"
    if -135.0 <= value < -45.0:
        return "left"
    return "back"


def sector_boundary_margin_deg(angle: float) -> float:
    value = (float(angle) + 180.0) % 360.0 - 180.0
    margins = [circular_diff_deg(value, boundary) for boundary in SECTOR_BOUNDARIES_DEG]
    return min(margins)


def woodworth_itd_ms(azimuth_deg: float, head_radius_m: float = 0.0875, c_mps: float = 343.0) -> float:
    """Woodworth 球头模型给出的 |ITD| 预期，毫秒。"""

    theta = math.radians(abs(circular_diff_deg(azimuth_deg, 0.0)))
    if theta > math.pi / 2:
        theta = math.pi - theta  # 后半球按镜像角
    return head_radius_m / c_mps * (theta + math.sin(theta)) * 1000.0


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    """读 PCM WAV，返回 (采样率, float64 数组 [n, channels])，幅度归一到 ±1。"""

    with wave.open(str(path), "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width {width} bytes: {path}")
    return sr, data.reshape(-1, channels)


# ── 几何 ────────────────────────────────────────────────────────────────────

def listener_basis(listener: Mapping[str, Any], frame: int) -> tuple[list[float], list[float]]:
    """和 ``unified_catalog._listener_azimuth`` 一样：优先 basis_m3，退回 yaw_deg。"""

    basis = listener.get("basis_m3")
    if isinstance(basis, Sequence) and len(basis) > frame and isinstance(basis[frame], Mapping):
        return list(basis[frame]["forward"]), list(basis[frame]["right"])
    yaws = listener.get("yaw_deg")
    yaw = float(yaws[frame]) if isinstance(yaws, Sequence) and len(yaws) > frame else 0.0
    forward = [math.sin(math.radians(yaw)), 0.0, -math.cos(math.radians(yaw))]
    right = [math.cos(math.radians(yaw)), 0.0, math.sin(math.radians(yaw))]
    return forward, right


def listener_azimuth_deg(source: Sequence[float], listener: Mapping[str, Any], frame: int) -> float | None:
    positions = listener.get("positions_m")
    if not isinstance(positions, Sequence) or len(positions) <= frame:
        return None
    point = positions[frame]
    vector = [float(source[i]) - float(point[i]) for i in range(3)]
    forward, right = listener_basis(listener, frame)
    forward_dot = vector[0] * forward[0] + vector[2] * forward[2]
    right_dot = vector[0] * right[0] + vector[2] * right[2]
    if math.isclose(forward_dot, 0.0, abs_tol=1e-12) and math.isclose(right_dot, 0.0, abs_tol=1e-12):
        return None
    angle = math.degrees(math.atan2(right_dot, forward_dot))
    return float((angle + 180.0) % 360.0 - 180.0)


def listener_distance_m(source: Sequence[float], listener: Mapping[str, Any], frame: int) -> float | None:
    positions = listener.get("positions_m")
    if not isinstance(positions, Sequence) or len(positions) <= frame:
        return None
    point = positions[frame]
    return math.sqrt(sum((float(source[i]) - float(point[i])) ** 2 for i in range(3)))


class EpisodeFacts:
    """facts.json 的薄封装，只暴露审计要用的读法。"""

    def __init__(self, facts: Mapping[str, Any]):
        self.raw = facts
        self.time = facts["time"]
        self.frame_count = int(self.time["frame_count"])
        self.frame_rate = float(self.time["frame_rate_hz"])
        self.duration_s = float(self.time.get("duration_seconds", self.frame_count / self.frame_rate))
        self.actors: dict[str, Mapping[str, Any]] = {
            str(k): v for k, v in (facts.get("actors") or {}).items() if isinstance(v, Mapping)
        }
        self.listener = facts.get("listener") or {}
        self.events = sorted(
            [e for e in (facts.get("events") or []) if isinstance(e, Mapping)],
            key=lambda e: (float(e.get("start_s", 0.0)), str(e.get("event_id"))),
        )
        self.visibility = facts.get("visibility") or {}
        self.appearance_review = facts.get("appearance_review") or {}

    def clamp(self, frame: int) -> int:
        return max(0, min(self.frame_count - 1, int(frame)))

    def appearance(self, actor_id: str) -> str | None:
        actor = self.actors.get(actor_id) or {}
        appearance = actor.get("appearance")
        return appearance.get("value") if isinstance(appearance, Mapping) else None

    def position(self, actor_id: str, frame: int) -> Sequence[float] | None:
        actor = self.actors.get(actor_id) or {}
        positions = actor.get("emitter_positions_m") or actor.get("root_positions_m")
        if not isinstance(positions, Sequence) or frame >= len(positions):
            return None
        return positions[frame]

    def azimuth(self, actor_id: str, frame: int) -> float | None:
        pos = self.position(actor_id, frame)
        return None if pos is None else listener_azimuth_deg(pos, self.listener, frame)

    def distance(self, actor_id: str, frame: int) -> float | None:
        pos = self.position(actor_id, frame)
        return None if pos is None else listener_distance_m(pos, self.listener, frame)

    def moving(self, actor_id: str, frame: int) -> bool | None:
        flags = (self.actors.get(actor_id) or {}).get("moving")
        if not isinstance(flags, Sequence) or frame >= len(flags):
            return None
        value = flags[frame]
        return bool(value) if isinstance(value, bool) else None

    def moving_any(self, actor_id: str, start: int, end_inclusive: int) -> bool | None:
        values = [self.moving(actor_id, f) for f in range(max(0, start), min(self.frame_count - 1, end_inclusive) + 1)]
        if not values or any(v is None for v in values):
            return None
        return any(values)

    def vis_record(self, actor_id: str, frame: int) -> Mapping[str, Any] | None:
        frames = self.visibility.get(actor_id)
        if not isinstance(frames, Mapping):
            return None
        record = frames.get(str(frame), frames.get(frame))
        return record if isinstance(record, Mapping) else None

    def vis_state(self, actor_id: str, frame: int) -> str | None:
        record = self.vis_record(actor_id, frame)
        return record.get("state") if record else None

    def appearance_review_frames(self, actor_id: str) -> int:
        review = self.appearance_review.get(actor_id) or {}
        refs = review.get("frame_refs") or review.get("frames") or []
        return len(refs) if isinstance(refs, Sequence) else 0

    def event_by_id(self, event_id: str | None) -> Mapping[str, Any] | None:
        for event in self.events:
            if event.get("event_id") == event_id:
                return event
        return None

    def first_event_of(self, actor_id: str) -> Mapping[str, Any] | None:
        for event in self.events:
            if event.get("actor_id") == actor_id:
                return event
        return None


# ── 成片双耳测量 ─────────────────────────────────────────────────────────────

def bandpass(signal: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    spectrum[(freqs < lo_hz) | (freqs > hi_hz)] = 0.0
    return np.fft.irfft(spectrum, len(signal))


def band_ild_db(left: np.ndarray, right: np.ndarray, sr: int, lo_hz: float | None, hi_hz: float | None) -> float | None:
    if left.size < 8 or right.size < 8:
        return None
    if lo_hz is not None and hi_hz is not None:
        left = bandpass(left, sr, lo_hz, hi_hz)
        right = bandpass(right, sr, lo_hz, hi_hz)
    left_rms, right_rms = rms(left), rms(right)
    if left_rms <= 0.0 or right_rms <= 0.0:
        return None
    return db(left_rms) - db(right_rms)


def onset_itd_ms(left: np.ndarray, right: np.ndarray, sr: int, *, window_s: float = 0.05,
                 max_lag_ms: float = 1.0, upsample: int = 8) -> dict[str, Any]:
    """在事件起点的直达声窗内量双耳时差。

    起点用 |mono| 首次超过事件峰值 10% 的采样定，避免把片头静音当直达声。返回的 ``itd_ms``
    为正表示**左耳先到**（声源在左），负表示右耳先到。这在混响混合声里本来就不稳，
    所以一起给出相关系数 ``cc``；cc 低的时候别单独相信这个数。
    """

    mono = (left + right) / 2.0
    if mono.size == 0:
        return {"itd_ms": None, "cc": None, "onset_offset_ms": None}
    peak = float(np.max(np.abs(mono)))
    if peak <= 0.0:
        return {"itd_ms": None, "cc": None, "onset_offset_ms": None, "note": "event is digital silence"}
    onset = int(np.argmax(np.abs(mono) > 0.1 * peak))
    window = int(window_s * sr)
    lw = left[onset:onset + window].astype(np.float64)
    rw = right[onset:onset + window].astype(np.float64)
    if lw.size < 16 or rw.size < 16:
        return {"itd_ms": None, "cc": None, "onset_offset_ms": onset / sr * 1000.0, "note": "onset window too short"}
    lw = lw - lw.mean()
    rw = rw - rw.mean()
    grid = np.arange(0, lw.size, 1.0 / upsample)
    lu = np.interp(grid, np.arange(lw.size), lw)
    ru = np.interp(grid, np.arange(rw.size), rw)
    norm = float(np.linalg.norm(lu) * np.linalg.norm(ru))
    if norm <= 0.0:
        return {"itd_ms": None, "cc": None, "onset_offset_ms": onset / sr * 1000.0, "note": "flat window"}
    max_lag = int(round(max_lag_ms / 1000.0 * sr * upsample))
    best_lag, best_cc = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = lu[lag:], ru[:lu.size - lag]
        else:
            a, b = lu[:lag], ru[-lag:]
        cc = float(np.dot(a, b) / norm)
        if cc > best_cc:
            best_lag, best_cc = lag, cc
    # lag > 0 表示 lu 要往后挪才对得上 ru，即左耳信号更晚到 → 右耳先到 → 负 ITD。
    itd_ms = -best_lag / upsample / sr * 1000.0
    return {"itd_ms": itd_ms, "cc": best_cc, "onset_offset_ms": onset / sr * 1000.0}


def audio_summary(data: np.ndarray, sr: int) -> dict[str, Any]:
    left, right = data[:, 0], data[:, 1]
    mono = (left + right) / 2.0
    tail = mono[-sr:] if mono.size >= sr else mono
    corr = float(np.corrcoef(left, right)[0, 1]) if left.size > 1 and np.std(left) > 0 and np.std(right) > 0 else None
    return {
        "sample_rate_hz": sr,
        "channel_count": int(data.shape[1]),
        "duration_s": data.shape[0] / sr,
        "peak_dbfs": db(float(np.max(np.abs(data)))) if data.size else None,
        "rms_dbfs": db(rms(mono)),
        "exact_zero_sample_fraction": float(np.mean(np.all(data == 0.0, axis=1))) if data.size else None,
        "last_second_rms_dbfs": db(rms(tail)),
        "left_right_correlation": corr,
        "rms_left_minus_right_over_rms_left": (rms(left - right) / rms(left)) if rms(left) > 0 else None,
    }


# ── 事件审计 ─────────────────────────────────────────────────────────────────

def event_frames(facts: EpisodeFacts, event: Mapping[str, Any]) -> tuple[int, int]:
    start = facts.clamp(int(event.get("start_frame", round(float(event["start_s"]) * facts.frame_rate))))
    end = facts.clamp(int(event.get("end_frame", round(float(event["end_s"]) * facts.frame_rate))))
    return start, max(start, end)


def event_overlaps(facts: EpisodeFacts, event: Mapping[str, Any]) -> dict[str, Any]:
    start, end = float(event["start_s"]), float(event["end_s"])
    overlaps = []
    total = 0.0
    for other in facts.events:
        if other.get("event_id") == event.get("event_id"):
            continue
        lo, hi = max(start, float(other["start_s"])), min(end, float(other["end_s"]))
        if lo < hi:
            overlaps.append({"event_id": other.get("event_id"), "actor_id": other.get("actor_id"), "overlap_s": hi - lo})
            total += hi - lo
    concurrent = 1 + len({o["actor_id"] for o in overlaps})
    return {"overlapping_events": overlaps, "overlap_total_s": total, "max_concurrent_speakers": concurrent}


def audit_event(facts: EpisodeFacts, event: Mapping[str, Any], audio: tuple[int, np.ndarray] | None,
                thresholds: Mapping[str, float]) -> dict[str, Any]:
    actor_id = str(event.get("actor_id"))
    start_f, end_f = event_frames(facts, event)
    competitors = [a for a in facts.actors if a != actor_id]

    separations: list[float] = []
    nearest_by_frame: list[str | None] = []
    speaker_azimuths: list[float] = []
    for frame in range(start_f, end_f + 1):
        target_az = facts.azimuth(actor_id, frame)
        if target_az is None:
            continue
        speaker_azimuths.append(target_az)
        best: tuple[float, str] | None = None
        for other in competitors:
            other_az = facts.azimuth(other, frame)
            if other_az is None:
                continue
            gap = circular_diff_deg(target_az, other_az)
            if best is None or gap < best[0]:
                best = (gap, other)
        if best is not None:
            separations.append(best[0])
            nearest_by_frame.append(best[1])

    onset_az = facts.azimuth(actor_id, start_f)
    end_az = facts.azimuth(actor_id, end_f)
    speaker_sweep = None
    if speaker_azimuths:
        # 说话期间说话人方位扫过的总角度（相邻帧带号转角的绝对值累加），以及起止净变化。
        speaker_sweep = {
            "net_deg": signed_circular_delta_deg(speaker_azimuths[0], speaker_azimuths[-1]),
            "path_deg": float(sum(abs(signed_circular_delta_deg(a, b)) for a, b in zip(speaker_azimuths, speaker_azimuths[1:]))),
        }

    record: dict[str, Any] = {
        "event_id": event.get("event_id"),
        "actor_id": actor_id,
        "appearance": facts.appearance(actor_id),
        "sound_class": event.get("sound_class"),
        "transcript": event.get("transcript"),
        "start_s": float(event["start_s"]),
        "end_s": float(event["end_s"]),
        "start_frame": start_f,
        "end_frame": end_f,
        "azimuth_convention": AZIMUTH_CONVENTION,
        "onset_azimuth_deg": onset_az,
        "end_azimuth_deg": end_az,
        "onset_distance_m": facts.distance(actor_id, start_f),
        "end_distance_m": facts.distance(actor_id, end_f),
        "onset_visibility_state": facts.vis_state(actor_id, start_f),
        "competitor_count": len(competitors),
        "separation_to_nearest_competitor_deg": {
            "onset": separations[0] if separations else None,
            "end": separations[-1] if separations else None,
            "min": min(separations) if separations else None,
            "max": max(separations) if separations else None,
            "change": (max(separations) - min(separations)) if separations else None,
            "nearest_competitor_at_onset": nearest_by_frame[0] if nearest_by_frame else None,
        },
        "speaker_azimuth_sweep_deg": speaker_sweep,
        "speaker_moving_during_event": facts.moving_any(actor_id, start_f, end_f),
        "competitors_moving_during_event": [
            other for other in competitors if facts.moving_any(other, start_f, end_f)
        ],
        "line_of_sight": None,  # 需要房间网格；由射线工具另行填写
        **event_overlaps(facts, event),
    }

    if audio is not None:
        sr, data = audio
        s = max(0, int(round(float(event["start_s"]) * sr)))
        e = min(data.shape[0], int(round(float(event["end_s"]) * sr)))
        left, right = data[s:e, 0], data[s:e, 1]
        seg = (left + right) / 2.0
        itd = onset_itd_ms(left, right, sr)
        record["measured"] = {
            "event_rms_dbfs": db(rms(seg)),
            "event_peak_dbfs": db(float(np.max(np.abs(seg)))) if seg.size else None,
            "ild_full_db": band_ild_db(left, right, sr, None, None),
            "ild_0p5_1p5k_db": band_ild_db(left, right, sr, 500.0, 1500.0),
            "ild_2_6k_db": band_ild_db(left, right, sr, 2000.0, 6000.0),
            "itd_onset_ms": itd.get("itd_ms"),
            "itd_onset_cc": itd.get("cc"),
            "itd_onset_offset_ms": itd.get("onset_offset_ms"),
            "itd_sign_convention": "positive_means_left_ear_leads",
            "expected_abs_itd_woodworth_ms": woodworth_itd_ms(onset_az) if onset_az is not None else None,
            "left_right_correlation": (
                float(np.corrcoef(left, right)[0, 1]) if left.size > 1 and np.std(left) > 0 and np.std(right) > 0 else None
            ),
        }
        if itd.get("note"):
            record["measured"]["itd_note"] = itd["note"]

    record["binding_feasibility"] = binding_feasibility(record, thresholds)
    return record


def binding_feasibility(event_record: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    """占位判据：几何上分得开（静态夹角够大，或说话期间夹角变化够大）且成片里量得到线索。"""

    sep = event_record.get("separation_to_nearest_competitor_deg") or {}
    measured = event_record.get("measured") or {}
    geometry_ok = None
    if sep.get("max") is not None:
        geometry_ok = bool(
            sep["max"] >= thresholds["theta_static_deg"]
            or (sep.get("change") or 0.0) >= thresholds["theta_motion_deg"]
        )
    cue_ok = None
    ild = measured.get("ild_2_6k_db")
    itd = measured.get("itd_onset_ms")
    if ild is not None or itd is not None:
        cue_ok = bool(
            (ild is not None and abs(ild) >= thresholds["ild_min_db"])
            or (itd is not None and abs(itd) >= thresholds["itd_min_ms"])
        )
    verdict = None
    if geometry_ok is not None:
        verdict = geometry_ok and (cue_ok if cue_ok is not None else True)
    reasons = []
    if geometry_ok is False:
        reasons.append("nearest competitor too close in azimuth for the whole event and no relative angular motion")
    if cue_ok is False:
        reasons.append("delivered audio carries neither ILD nor ITD above threshold")
    if event_record.get("competitor_count", 0) == 0:
        reasons.append("no competitor: binding is trivial")
    return {
        "geometry_separable": geometry_ok,
        "delivered_cues_present": cue_ok,
        "feasible": verdict,
        "reasons": reasons,
        "calibration": CALIBRATION_NOTE,
    }


# ── 逐题审计 ─────────────────────────────────────────────────────────────────

def anchor_event_id(evidence: Mapping[str, Any]) -> str | None:
    for key in ("event", "anchor_event", "first_event"):
        nested = evidence.get(key)
        if isinstance(nested, Mapping) and nested.get("event_id"):
            return str(nested["event_id"])
    if evidence.get("event_id"):
        return str(evidence["event_id"])
    ids = evidence.get("event_ids") or evidence.get("active_event_ids")
    if isinstance(ids, Sequence) and ids:
        return str(ids[0])
    return None


def gold_of(item: Mapping[str, Any]) -> Any:
    truth = item.get("truth")
    if isinstance(truth, Mapping):
        for key in ("mcq_value", "value", "label"):
            if truth.get(key) is not None:
                return truth[key]
    forms = item.get("forms") or {}
    mcq = forms.get("mcq") if isinstance(forms, Mapping) else None
    if isinstance(mcq, Mapping) and isinstance(mcq.get("gold"), Mapping):
        return mcq["gold"].get("value")
    return truth


def divergence(values: Mapping[str, Any], target: str | None, gold: Any) -> dict[str, Any]:
    """其他候选取值与金标的关系。"""

    others = {k: v for k, v in values.items() if k != target}
    same = [k for k, v in others.items() if v == gold]
    all_values = list(values.values())
    gold_count = sum(1 for v in all_values if v == gold)
    return {
        "candidate_values": values,
        "distractors_equal_to_gold": same,
        "degenerate_all_same": bool(others) and len(same) == len(others),
        "gold_is_majority": gold_count * 2 > len(all_values) if all_values else None,
    }


def seeing_profile(facts: EpisodeFacts, target: str | None, frames: Sequence[int]) -> dict[str, Any]:
    occlusion, pixels = [], []
    for frame in frames:
        record = facts.vis_record(target, frame) if target else None
        if record:
            if record.get("occlusion_fraction") is not None:
                occlusion.append(float(record["occlusion_fraction"]))
            if record.get("visible_pixels") is not None:
                pixels.append(int(record["visible_pixels"]))
    return {
        "target_actor_id": target,
        "frames": list(frames),
        "target_occlusion_fraction_max": max(occlusion) if occlusion else None,
        "target_visible_pixels_min": min(pixels) if pixels else None,
        "appearance_review_frame_count": facts.appearance_review_frames(target) if target else None,
        "candidate_count": len(facts.actors),
    }


def audit_question(facts: EpisodeFacts, item: Mapping[str, Any], events_by_id: Mapping[str, Mapping[str, Any]],
                   thresholds: Mapping[str, float]) -> dict[str, Any]:
    qa_id = str(item.get("qa_id"))
    evidence = item.get("evidence") or {}
    gold = gold_of(item)
    anchor_id = anchor_event_id(evidence)
    anchor = facts.event_by_id(anchor_id)
    target = evidence.get("target_actor_id") or (anchor.get("actor_id") if anchor else None)
    if target is None and qa_id in {"QA-14"}:
        target = None
    group = ("binding" if qa_id in BINDING_TYPES else "conditional" if qa_id in CONDITIONAL_TYPES
             else "audio_control" if qa_id in AUDIO_CONTROL_TYPES else "visual_control" if qa_id in VISUAL_CONTROL_TYPES
             else "unclassified")
    out: dict[str, Any] = {
        "qa_id": qa_id,
        "question_id": item.get("question_id"),
        "status": item.get("status"),
        "group": group,
        "gold": gold,
        "anchor_event_id": anchor_id,
        "target_actor_id": target,
        "target_appearance": facts.appearance(target) if target else None,
        "listening": events_by_id.get(anchor_id, {}).get("binding_feasibility") if anchor_id else None,
        "listening_event": (
            {k: events_by_id[anchor_id].get(k) for k in ("separation_to_nearest_competitor_deg", "measured", "max_concurrent_speakers", "speaker_moving_during_event", "competitors_moving_during_event")}
            if anchor_id in events_by_id else None
        ),
        "seeing": None,
        "temporal": None,
        "divergence": None,
        "flags": [],
    }
    last = facts.frame_count - 1
    actors = list(facts.actors)

    if qa_id == "QA-06" and anchor:
        s, e = event_frames(facts, anchor)
        values = {}
        for a in actors:
            flags = [facts.moving(a, f) for f in range(s, max(s + 1, e))]
            values[a] = ("moving" if all(flags) else "still" if not any(flags) else "mixed") if flags and None not in flags else None
        out["divergence"] = divergence(values, target, gold)
        out["seeing"] = seeing_profile(facts, target, [s, e])
    elif qa_id == "QA-08" and anchor:
        frame = int(evidence.get("query_frame", event_frames(facts, anchor)[0]))
        values = {a: facts.vis_state(a, frame) for a in actors}
        out["divergence"] = divergence(values, target, gold)
        out["seeing"] = seeing_profile(facts, target, [frame])
        out["flags"].append("truth_is_onset_frame_but_question_says_during")
    elif qa_id == "QA-24":
        frame = int(evidence.get("final_frame", last))
        values = {a: facts.vis_state(a, frame) for a in actors}
        out["divergence"] = divergence(values, target, gold)
        out["seeing"] = seeing_profile(facts, target, [frame])
    elif qa_id == "QA-13" and anchor and target:
        query = int(evidence.get("query_frame", (evidence.get("post_sound") or {}).get("query_frame", last)))
        _, end_f = event_frames(facts, anchor)
        az_end, az_query = facts.azimuth(target, end_f), facts.azimuth(target, query)
        others = {a: facts.azimuth(a, query) for a in actors if a != target}
        displacement = signed_circular_delta_deg(az_end, az_query) if az_end is not None and az_query is not None else None
        values = {a: (sector_of(v) if v is not None else None) for a, v in {**others, target: az_query}.items()}
        out["divergence"] = divergence(values, target, sector_of(az_query) if az_query is not None else None)
        out["temporal"] = {
            "azimuth_at_event_end_deg": az_end,
            "azimuth_at_query_deg": az_query,
            "post_sound_displacement_deg": displacement,
            "target_moved_between": facts.moving_any(target, end_f, query),
            "answer_equals_last_heard_direction": (
                abs(displacement) < thresholds["post_sound_min_displacement_deg"] if displacement is not None else None
            ),
            "nearest_distractor_gap_at_query_deg": (
                min(circular_diff_deg(az_query, v) for v in others.values() if v is not None) if az_query is not None and any(v is not None for v in others.values()) else None
            ),
            "gold_sector_boundary_margin_deg": sector_boundary_margin_deg(az_query) if az_query is not None else None,
            "query_frame": query,
        }
        if out["temporal"]["gold_sector_boundary_margin_deg"] is not None and out["temporal"]["gold_sector_boundary_margin_deg"] < thresholds["sector_margin_deg"]:
            out["flags"].append("mcq_gold_within_sector_boundary_margin")
        if out["temporal"]["answer_equals_last_heard_direction"]:
            out["flags"].append("post_sound_answer_equals_last_heard_direction")
        out["seeing"] = seeing_profile(facts, target, [end_f, query])
    elif qa_id == "QA-16" and anchor and target:
        post = evidence.get("post_sound") or {}
        query = int(post.get("query_frame", evidence.get("query_frame", last)))
        s, e = event_frames(facts, anchor)
        anchor_f = max(s, e - 1)
        deltas = {}
        for a in actors:
            d0, d1 = facts.distance(a, anchor_f), facts.distance(a, query)
            deltas[a] = (d1 - d0) if d0 is not None and d1 is not None else None
        values = {a: ("nearer" if d is not None and d < -thresholds["distance_margin_m"] else "farther" if d is not None and d > thresholds["distance_margin_m"] else "unchanged" if d is not None else None) for a, d in deltas.items()}
        during = None
        d_start, d_end = facts.distance(target, s), facts.distance(target, anchor_f)
        if d_start is not None and d_end is not None:
            during = d_end - d_start
        post_delta = deltas.get(target)
        extrapolable = (
            during is not None and post_delta is not None and abs(during) >= thresholds["distance_margin_m"] / 2
            and math.copysign(1, during) == math.copysign(1, post_delta) and bool(facts.moving(target, anchor_f))
        )
        out["divergence"] = divergence(values, target, gold)
        out["divergence"]["distance_delta_m"] = deltas
        out["temporal"] = {
            "during_event_distance_trend_m": during,
            "post_sound_distance_delta_m": post_delta,
            "target_moving_at_event_end": facts.moving(target, anchor_f),
            "extrapolable_from_pre_end_trend": extrapolable,
            "query_frame": query,
        }
        if extrapolable:
            out["flags"].append("post_sound_answer_extrapolable_from_trend")
        out["seeing"] = seeing_profile(facts, target, [anchor_f, query])
    elif qa_id == "QA-17" and anchor and target:
        frames = evidence.get("motion_frames")
        if isinstance(frames, Sequence) and len(frames) == 2:
            end_f, query = int(frames[0]), int(frames[1])
        else:
            _, end_f = event_frames(facts, anchor)
            query = int((evidence.get("post_sound") or {}).get("query_frame", last))
        values = {a: ("yes" if facts.moving_any(a, end_f, query) else "no" if facts.moving_any(a, end_f, query) is False else None) for a in actors}
        out["divergence"] = divergence(values, target, gold)
        out["temporal"] = {
            "window_frames": [end_f, query],
            "target_moving_at_event_end": facts.moving(target, end_f),
            "motion_continuity_predicts_answer": bool(facts.moving(target, end_f)) and gold == "yes",
        }
        if out["temporal"]["motion_continuity_predicts_answer"]:
            out["flags"].append("post_sound_answer_extrapolable_from_trend")
        out["seeing"] = seeing_profile(facts, target, [end_f, query])
    elif qa_id == "QA-15" and anchor:
        s, e = event_frames(facts, anchor)
        e2 = max(s + 1, e - 1)
        deltas = {}
        for a in actors:
            d0, d1 = facts.distance(a, s), facts.distance(a, e2)
            deltas[a] = (d1 - d0) if d0 is not None and d1 is not None else None
        values = {a: ("nearer" if d is not None and d < -thresholds["distance_margin_m"] else "farther" if d is not None and d > thresholds["distance_margin_m"] else "unchanged" if d is not None else None) for a, d in deltas.items()}
        out["divergence"] = divergence(values, target, gold)
        out["divergence"]["distance_delta_m"] = deltas
        out["seeing"] = seeing_profile(facts, target, [s, e2])
    elif qa_id == "QA-19" and target:
        bands = evidence.get("time_bands_s")
        if isinstance(bands, Sequence) and bands:
            def band_of(t: float) -> str:
                for index, (lo, hi) in enumerate(bands):
                    if lo <= t < hi:
                        return f"band_{index}"
                return f"band_{len(bands) - 1}"
            values = {}
            for a in actors:
                first = facts.first_event_of(a)
                values[a] = band_of(float(first["start_s"])) if first else "no_sound"
            out["divergence"] = divergence(values, target, gold)
        out["seeing"] = seeing_profile(facts, target, [event_frames(facts, anchor)[0]] if anchor else [])
    elif qa_id == "QA-14":
        frame = int(evidence.get("query_frame", 0))
        dists = {a: facts.distance(a, frame) for a in actors}
        out["divergence"] = {"candidate_values": dists, "note": "visual control; distances at the query frame"}
        out["seeing"] = seeing_profile(facts, gold if gold in facts.actors else None, [frame])
    elif qa_id == "QA-18":
        active = evidence.get("active_actor_ids") or []
        out["divergence"] = {
            "branch": "single_speaker" if len(active) == 1 else "none" if not active else "multiple",
            "speakers_in_clip": len({e.get("actor_id") for e in facts.events}),
            "query_source": evidence.get("query_source"),
        }
        if out["divergence"]["branch"] != "single_speaker":
            out["flags"].append("audio_only_branch")
    elif qa_id == "QA-20":
        out["divergence"] = {"target_visible": evidence.get("target_visible"), "visible_candidates": evidence.get("visible_candidate_actor_ids")}
        if evidence.get("target_visible"):
            out["flags"].append("positive_branch_only")
    elif qa_id == "QA-22":
        entity, speaking = evidence.get("entity_count"), evidence.get("speaking_count")
        out["divergence"] = {"entity_count": entity, "speaking_count": speaking, "all_appearing_entities_speak": entity == speaking}
        if entity == speaking:
            out["flags"].append("no_silent_entity")
    elif qa_id == "QA-01":
        silent = [a for a in actors if facts.first_event_of(a) is None]
        out["divergence"] = {"silent_actors": silent, "gold": gold}
        if not silent:
            out["flags"].append("no_silent_entity")
    elif qa_id in {"QA-02", "QA-03", "QA-12", "QA-21"} and anchor:
        s, _ = event_frames(facts, anchor)
        out["seeing"] = seeing_profile(facts, target, [s])
    if out["divergence"] and out["divergence"].get("degenerate_all_same"):
        out["flags"].append("degenerate_distractors_equal_gold")
    if out["divergence"] and out["divergence"].get("gold_is_majority") and group == "binding":
        out["flags"].append("gold_is_majority_among_candidates")
    if group == "binding" and out["listening"] and out["listening"].get("feasible") is False:
        out["flags"].append("binding_not_feasible_placeholder")
    return out


# ── 汇总 ─────────────────────────────────────────────────────────────────────

def crosscheck_azimuth_formula(facts: EpisodeFacts) -> dict[str, Any]:
    try:
        from avengine.qa.unified_catalog import _distance, _listener_azimuth  # type: ignore
    except Exception as exc:  # noqa: BLE001 - 对账是可选的
        return {"status": "unavailable", "detail": f"{type(exc).__name__}: {exc}"}
    max_az, max_dist, compared = 0.0, 0.0, 0
    for actor_id in facts.actors:
        for frame in range(facts.frame_count):
            pos = facts.position(actor_id, frame)
            if pos is None:
                continue
            ref_az = _listener_azimuth(pos, facts.listener, frame)
            ref_d = _distance(pos, facts.listener, frame)
            mine_az, mine_d = facts.azimuth(actor_id, frame), facts.distance(actor_id, frame)
            if ref_az is not None and mine_az is not None:
                max_az = max(max_az, circular_diff_deg(ref_az, mine_az))
            if ref_d is not None and mine_d is not None:
                max_dist = max(max_dist, abs(ref_d - mine_d))
            compared += 1
    status = "matched" if max_az < 1e-6 and max_dist < 1e-6 else "mismatch"
    return {"status": status, "frames_compared": compared, "max_abs_azimuth_diff_deg": max_az, "max_abs_distance_diff_m": max_dist}


def run_audit(facts_path: Path, wav_path: Path | None, questions_path: Path | None,
              thresholds: Mapping[str, float]) -> dict[str, Any]:
    facts_raw = load_json(facts_path)
    facts = EpisodeFacts(facts_raw)
    audio_path = wav_path or (Path(facts_raw.get("audio", {}).get("path")) if facts_raw.get("audio", {}).get("path") else None)
    audio = None
    audio_block: dict[str, Any] = {"path": str(audio_path) if audio_path else None, "status": "missing"}
    if audio_path and Path(audio_path).is_file():
        sr, data = read_wav(Path(audio_path))
        if data.shape[1] != 2:
            raise ValueError(f"expected a 2-channel WAV, got {data.shape[1]} channels: {audio_path}")
        audio = (sr, data)
        audio_block = {"path": str(audio_path), "status": "measured", **audio_summary(data, sr)}
        expected = int(facts.time.get("sample_count", 0))
        if expected and expected != data.shape[0]:
            audio_block["sample_count_mismatch"] = {"facts": expected, "wav": int(data.shape[0])}
    events = [audit_event(facts, event, audio, thresholds) for event in facts.events]
    events_by_id = {e["event_id"]: e for e in events}
    questions = []
    if questions_path:
        qset = load_json(questions_path)
        for item in qset.get("items", []):
            if isinstance(item, Mapping):
                questions.append(audit_question(facts, item, events_by_id, thresholds))
    flag_counts: dict[str, int] = {}
    for q in questions:
        for flag in q["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    feasible = [e["binding_feasibility"].get("feasible") for e in events]
    summary = {
        "event_count": len(events),
        "events_geometry_separable": sum(1 for e in events if e["binding_feasibility"].get("geometry_separable")),
        "events_with_delivered_cues": sum(1 for e in events if e["binding_feasibility"].get("delivered_cues_present")),
        "events_binding_feasible_placeholder": sum(1 for v in feasible if v),
        "events_speaker_moving_while_speaking": sum(1 for e in events if e.get("speaker_moving_during_event")),
        "question_count": len(questions),
        "questions_by_group": {g: sum(1 for q in questions if q["group"] == g) for g in ("binding", "conditional", "audio_control", "visual_control", "unclassified")},
        "flag_counts": flag_counts,
    }
    return {
        "schema": SCHEMA,
        "status": "research_only",
        "claim_boundary": (
            "Numbers are recomputed from delivered facts.json and the delivered stereo WAV. They describe "
            "answerability structure and difficulty; they are not model results, not human-answerability "
            "certificates and not formal admission. Thresholds are placeholders pending human calibration."
        ),
        "inputs": {"facts": str(facts_path), "wav": audio_block.get("path"), "questions": str(questions_path) if questions_path else None},
        "episode_id": facts_raw.get("episode_id"),
        "catalog_version": facts_raw.get("catalog_version"),
        "azimuth_convention": AZIMUTH_CONVENTION,
        "azimuth_formula_crosscheck": crosscheck_azimuth_formula(facts),
        "thresholds": {**thresholds, "calibration": CALIBRATION_NOTE},
        "audio": audio_block,
        "events": events,
        "questions": questions,
        "summary": summary,
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    a = payload["audio"]
    print(f"episode {payload.get('episode_id')}: {s['event_count']} events, "
          f"{s['events_geometry_separable']} geometry-separable, {s['events_with_delivered_cues']} with delivered cues, "
          f"{s['events_binding_feasible_placeholder']} binding-feasible (placeholder thresholds); "
          f"{s['events_speaker_moving_while_speaking']} speaker(s) moving while speaking")
    if a.get("status") == "measured":
        print(f"  audio: peak {a['peak_dbfs']:.1f} dBFS, rms {a['rms_dbfs']:.1f} dBFS, exact-zero fraction {a['exact_zero_sample_fraction']:.3f}")
    for e in payload["events"]:
        sep = e["separation_to_nearest_competitor_deg"]
        m = e.get("measured") or {}
        ild = m.get("ild_2_6k_db")
        itd = m.get("itd_onset_ms")
        print(f"  {e['event_id']} {e['actor_id']}({e['appearance']}) az {e['onset_azimuth_deg']:+.1f}° "
              f"sep onset/min/max {sep['onset']:.1f}/{sep['min']:.1f}/{sep['max']:.1f}° "
              f"ILD2-6k {ild:+.1f} dB ITD {itd:+.3f} ms -> feasible={e['binding_feasibility']['feasible']}"
              if sep.get("onset") is not None and ild is not None and itd is not None else
              f"  {e['event_id']} {e['actor_id']} (incomplete measurement)")
    if s["question_count"]:
        print(f"  questions: {s['question_count']} ({s['questions_by_group']}); flags: {s['flag_counts']}")
    xc = payload["azimuth_formula_crosscheck"]
    print(f"  azimuth formula crosscheck vs unified_catalog: {xc.get('status')}"
          + (f" (max diff {xc['max_abs_azimuth_diff_deg']:.2e}°)" if xc.get("status") in {"matched", "mismatch"} else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facts", required=True, type=Path, help="delivery 目录里的 facts.json")
    parser.add_argument("--wav", type=Path, default=None, help="成片双耳 WAV；缺省取 facts.audio.path")
    parser.add_argument("--questions", type=Path, default=None, help="同一 delivery 目录的 questions.json（可选）")
    parser.add_argument("--out", required=True, type=Path, help="输出 JSON，已存在则拒绝")
    for key, value in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=value, help=f"占位阈值，默认 {value}")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    thresholds = {key: float(getattr(args, key)) for key in DEFAULT_THRESHOLDS}
    payload = run_audit(args.facts, args.wav, args.questions, thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print_summary(payload)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
