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

v2（2026-09-06 晚，按 Codex 审核第 8 节修正）
--------------------------------------------
- 不再输出合并的 ``feasible`` 布尔值。几何和成片线索各自是三态：``candidate_pass`` /
  ``candidate_fail`` / ``unmeasured``（没有竞争者时是 ``not_applicable_no_competitor``）。
  没量到的东西永远是 ``unmeasured``，不会在汇总里变成通过。
- 几何判据不再看瞬时最大夹角，而是看整窗最小值和"连续达到阈值的时长"，并报告
  分位数、最近竞争者是否换人。
- 成片线索不再用绝对 ILD/ITD 阈值，而是比较目标与最近竞争者按几何算的预期时差之差，
  再看实测方向是否与预期一致。正前方的声源实测线索本来就接近零，不再因此判"无空间信息"。
- 有逐源湿声 stem（``<actor>_mouth_stem.wav`` 等）时优先在 stem 上量起点和线索，并从
  stem 读出可听窗；混合轨另报，与其他事件重叠的窗打 ``mixture_contaminated``。
- 最大并发发声人数改为按事件边界的时间扫描、按实体去重、半开区间。
- 逐题分歧按题目实际可用的形式分别算：Open 用数值间隙，MCQ 用该题的答案域。
- 可选 ``--acoustic-package``：用声学包里的静态三角面对发声点到听者做直达射线，填
  ``line_of_sight``；坐标系不是米、Y 向上时拒绝使用并说明原因，不静默换轴。

输入
----
- ``--facts``：一段 Episode 的归一化事实 ``facts.json``（``unified_catalog.normalize_episode_bundle``
  的输出，delivery 目录里那份）。
- ``--wav``：成片双耳 WAV；缺省时取 ``facts.audio.path``。
- ``--stems-dir``：逐源 stem 所在目录；缺省取成片 WAV 所在目录，找不到就只量混合轨。
- ``--acoustic-package``：RLR 声学包 manifest（``avengine_acoustic_scene_package_v1``），可选。
- ``--questions``：同一 delivery 目录的 ``questions.json``（可选；给了才做逐题分歧审计）。
- ``--out``：输出 JSON，no-clobber。

所有阈值都是占位值，等人工校准；输出里 ``thresholds.calibration`` 明确写着这一点。
方位一律是**引擎帧、右为正**（和 ``unified_catalog._listener_azimuth`` 同一公式、同一基向量），
并带 ``azimuth_convention`` 字段；能导入 ``avengine.qa.unified_catalog`` 时会逐帧对账，
把最大差值写进 ``azimuth_formula_crosscheck``。这里的数不是模型成绩，也不是正式准入。
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "avengine_qa_binding_feasibility_audit_v2"
AZIMUTH_CONVENTION = "engine_right_positive_deg"
VISIBLE_STATES = ("visible_clear", "visible_occluded")
SECTOR_BOUNDARIES_DEG = (-135.0, -45.0, 45.0, 135.0)
STATE_PASS = "candidate_pass"
STATE_FAIL = "candidate_fail"
STATE_UNMEASURED = "unmeasured"
STATE_NO_COMPETITOR = "not_applicable_no_competitor"

# 题型分组（与 2026-09-06 审阅第 4 节一致）。
BINDING_TYPES = {"QA-01", "QA-02", "QA-03", "QA-08", "QA-12", "QA-13", "QA-16", "QA-17",
                 "QA-19", "QA-20", "QA-21", "QA-22", "QA-24"}  # 13 类，与审阅第 4 节一致
CONDITIONAL_TYPES = {"QA-06", "QA-15", "QA-18"}
AUDIO_CONTROL_TYPES = {"QA-04", "QA-05", "QA-23"}
VISUAL_CONTROL_TYPES = {"QA-07", "QA-09", "QA-10", "QA-11", "QA-14"}

DEFAULT_THRESHOLDS = {
    "theta_static_deg": 30.0,   # 与最近竞争者的夹角至少这么大才算"分得开"
    "min_sustained_separation_s": 1.0,  # 夹角连续达到阈值至少持续这么久
    "theta_motion_deg": 10.0,   # 只作报告：说话期间这个夹角变化了多少
    "ild_min_db": 3.0,          # 只作报告：成片 2–6 kHz 左右能量差
    "itd_min_ms": 0.2,          # 目标与最近竞争者的预期时差之差至少这么大
    "itd_min_cc": 0.3,          # 起点窗互相关峰值低于这个数就不信 ITD 的符号
    "near_axis_deg": 10.0,      # 方位绝对值小于这个数时预期左右线索接近零
    "sector_margin_deg": 5.0,   # QA-13 金标离扇区边界小于这个值就标出来
    "post_sound_min_displacement_deg": 15.0,  # QA-13 声停后至少要挪这么多才不是"最后听到的方位"
    "open_angle_min_gap_deg": 10.0,  # Open 形式 QA-13：目标与最近干扰项数值方位至少差这么多
    "open_time_min_gap_s": 1.0,      # Open 形式 QA-19：首次发声时刻至少差这么多
    "distance_margin_m": 0.2,   # 与 unified_catalog 的 QA-15/16 研究边距一致
}
CALIBRATION_NOTE = "placeholder_pending_human_calibration"
AUDIBLE_WINDOW_METHOD = "stem_rms20ms_hop10ms_minus25dB_relative_to_peak_placeholder"


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


def expected_signed_itd_ms(azimuth_deg: float | None) -> float | None:
    """带符号的预期 ITD：正表示左耳先到，也就是声源在左（右为正约定下方位为负）。"""

    if azimuth_deg is None:
        return None
    value = (float(azimuth_deg) + 180.0) % 360.0 - 180.0
    if math.isclose(value, 0.0, abs_tol=1e-9) or math.isclose(abs(value), 180.0, abs_tol=1e-9):
        return 0.0
    return -math.copysign(woodworth_itd_ms(value), value)


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    """读 WAV（PCM 8/16/24/32 位或 IEEE float 32/64 位），返回 (采样率, float64 数组 [n, channels])，幅度归一到 ±1。

    自己解析 RIFF 块，因为标准库 ``wave`` 不认 IEEE_FLOAT（format_tag=3）的逐源 stem。
    """

    raw = Path(path).read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    offset = 12
    fmt: dict[str, int] | None = None
    data: bytes | None = None
    while offset + 8 <= len(raw):
        chunk_id = raw[offset:offset + 4]
        size = struct.unpack("<I", raw[offset + 4:offset + 8])[0]
        body = raw[offset + 8:offset + 8 + size]
        if chunk_id == b"fmt ":
            tag, channels, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = {"tag": tag, "channels": channels, "sr": sr, "bits": bits}
        elif chunk_id == b"data":
            data = body
        offset += 8 + size + (size % 2)
    if fmt is None or data is None:
        raise ValueError(f"WAV without fmt/data chunks: {path}")
    channels, bits, tag = fmt["channels"], fmt["bits"], fmt["tag"]
    if tag == 1:
        if bits == 8:
            values = (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        elif bits == 16:
            values = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
        elif bits == 24:
            frames = np.frombuffer(data[: len(data) - len(data) % 3], dtype=np.uint8).reshape(-1, 3)
            ints = (frames[:, 0].astype(np.int32) | (frames[:, 1].astype(np.int32) << 8)
                    | (frames[:, 2].astype(np.int32) << 16))
            ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
            values = ints.astype(np.float64) / float(1 << 23)
        elif bits == 32:
            values = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"unsupported PCM width {bits} bits: {path}")
    elif tag == 3:
        if bits == 32:
            values = np.frombuffer(data, dtype="<f4").astype(np.float64)
        elif bits == 64:
            values = np.frombuffer(data, dtype="<f8").astype(np.float64)
        else:
            raise ValueError(f"unsupported float width {bits} bits: {path}")
    else:
        raise ValueError(f"unsupported WAV format tag {tag}: {path}")
    usable = values.size - values.size % channels
    return int(fmt["sr"]), values[:usable].reshape(-1, channels)


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

    def frame_of_time(self, seconds: float) -> int:
        return self.clamp(int(math.floor(float(seconds) * self.frame_rate + 1e-9)))

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

    def listener_position(self, frame: int) -> Sequence[float] | None:
        positions = self.listener.get("positions_m")
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


# ── 静态几何与直达射线 ───────────────────────────────────────────────────────

ACCEPTED_UNITS = {"m", "meter", "metre", "meters", "metres"}
ACCEPTED_UP = {"y", "+y"}


class StaticGeometry:
    """声学包里的静态三角面，只用于发声点到听者的直达射线。"""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, source: str):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        tri = np.asarray(triangles).reshape(-1, 3).astype(np.int64)
        self.v0 = self.vertices[tri[:, 0]]
        self.v1 = self.vertices[tri[:, 1]]
        self.v2 = self.vertices[tri[:, 2]]
        self.lo = np.minimum(np.minimum(self.v0, self.v1), self.v2)
        self.hi = np.maximum(np.maximum(self.v0, self.v1), self.v2)
        self.source = source
        self.triangle_count = int(tri.shape[0])

    def segment_blocked(self, origin: Sequence[float], target: Sequence[float], *, clearance_m: float = 0.02) -> bool:
        """Möller–Trumbore：线段 origin→target 是否穿过任何三角面（两端各留 clearance_m 不算）。"""

        o = np.asarray(origin, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        d = t - o
        length = float(np.linalg.norm(d))
        if length <= 2 * clearance_m:
            return False
        direction = d / length
        seg_lo, seg_hi = np.minimum(o, t) - 1e-6, np.maximum(o, t) + 1e-6
        candidates = np.all(self.hi >= seg_lo, axis=1) & np.all(self.lo <= seg_hi, axis=1)
        if not np.any(candidates):
            return False
        v0, v1, v2 = self.v0[candidates], self.v1[candidates], self.v2[candidates]
        e1, e2 = v1 - v0, v2 - v0
        h = np.cross(direction, e2)
        a = np.einsum("ij,ij->i", e1, h)
        parallel = np.abs(a) < 1e-12
        a = np.where(parallel, 1.0, a)
        f = 1.0 / a
        s = o - v0
        u = f * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = f * (q @ direction)
        dist = f * np.einsum("ij,ij->i", e2, q)
        hit = (~parallel) & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0)
        hit &= (dist > clearance_m) & (dist < length - clearance_m)
        return bool(np.any(hit))


def load_static_geometry(manifest_path: Path) -> StaticGeometry | dict[str, Any]:
    """读 ``avengine_acoustic_scene_package_v1`` 的三角面；坐标不是米、Y 向上就拒绝，不静默换轴。"""

    manifest = load_json(manifest_path)
    coordinate = manifest.get("coordinate_system") or {}
    unit = str(coordinate.get("linear_unit", "")).lower()
    up = str(coordinate.get("up_axis", "")).lower()
    if unit not in ACCEPTED_UNITS or up not in ACCEPTED_UP:
        return {"status": "unusable", "reason": f"package coordinate system is not metres/Y-up: {coordinate}",
                "manifest": str(manifest_path)}
    arrays = manifest.get("arrays") or {}
    try:
        base = Path(manifest_path).parent
        vertices = np.load(base / arrays["vertices"]["path"])
        triangles = np.load(base / arrays["triangles"]["path"])
    except (KeyError, OSError, ValueError) as exc:
        return {"status": "unusable", "reason": f"cannot load package arrays: {type(exc).__name__}: {exc}",
                "manifest": str(manifest_path)}
    return StaticGeometry(vertices, triangles, source=str(manifest_path))


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


def measure_binaural_window(data: np.ndarray, sr: int, start_s: float, end_s: float) -> dict[str, Any]:
    """在 [start_s, end_s) 上量峰值、有效值、三个频带的 ILD 和起点窗 ITD。"""

    s = max(0, int(round(float(start_s) * sr)))
    e = min(data.shape[0], int(round(float(end_s) * sr)))
    left, right = data[s:e, 0], data[s:e, 1]
    seg = (left + right) / 2.0
    itd = onset_itd_ms(left, right, sr)
    out: dict[str, Any] = {
        "window_s": [s / sr, e / sr],
        "event_rms_dbfs": db(rms(seg)),
        "event_peak_dbfs": db(float(np.max(np.abs(seg)))) if seg.size else None,
        "ild_full_db": band_ild_db(left, right, sr, None, None),
        "ild_0p5_1p5k_db": band_ild_db(left, right, sr, 500.0, 1500.0),
        "ild_2_6k_db": band_ild_db(left, right, sr, 2000.0, 6000.0),
        "itd_onset_ms": itd.get("itd_ms"),
        "itd_onset_cc": itd.get("cc"),
        "itd_onset_offset_ms": itd.get("onset_offset_ms"),
        "itd_sign_convention": "positive_means_left_ear_leads",
        "left_right_correlation": (
            float(np.corrcoef(left, right)[0, 1]) if left.size > 1 and np.std(left) > 0 and np.std(right) > 0 else None
        ),
    }
    if itd.get("note"):
        out["itd_note"] = itd["note"]
    return out


def audible_window_from_stem(data: np.ndarray, sr: int, start_s: float, end_s: float, *,
                             rel_db: float = 25.0, win_s: float = 0.02, hop_s: float = 0.01) -> dict[str, Any]:
    """在 stem 的放置窗里按 20 ms 窗有效值找可听起止：相对该窗内最响一帧低 rel_db 以上算静音。

    这是占位检测器，不是人工核过的 VAD；只用来把"放置区间"和"听得到的区间"分开。
    """

    s = max(0, int(round(float(start_s) * sr)))
    e = min(data.shape[0], int(round(float(end_s) * sr)))
    mono = (data[s:e, 0] + data[s:e, 1]) / 2.0 if data.shape[1] >= 2 else data[s:e, 0]
    win, hop = max(1, int(win_s * sr)), max(1, int(hop_s * sr))
    if mono.size < win:
        return {"status": "unmeasured", "reason": "placement window shorter than one analysis frame"}
    starts = np.arange(0, mono.size - win + 1, hop)
    frame_rms = np.array([rms(mono[i:i + win]) for i in starts])
    peak = float(frame_rms.max()) if frame_rms.size else 0.0
    if peak <= 0.0:
        return {"status": "silent", "reason": "stem is digital silence inside the placement window"}
    active = frame_rms >= peak * 10 ** (-rel_db / 20.0)
    idx = np.flatnonzero(active)
    # 帧级找到首末活动帧后，在帧内按采样点细化到首个/末个超过该帧峰值一成的样本，
    # 避免 20 ms 窗刚沾到能量就把起点报早半帧。
    first = mono[int(starts[idx[0]]):int(starts[idx[0]]) + win]
    last_frame = mono[int(starts[idx[-1]]):int(starts[idx[-1]]) + win]
    first_peak, last_peak = float(np.max(np.abs(first))), float(np.max(np.abs(last_frame)))
    first_hit = int(np.argmax(np.abs(first) > 0.1 * first_peak)) if first_peak > 0 else 0
    last_hits = np.flatnonzero(np.abs(last_frame) > 0.1 * last_peak) if last_peak > 0 else np.array([win - 1])
    onset = (s + int(starts[idx[0]]) + first_hit) / sr
    offset = (s + int(starts[idx[-1]]) + int(last_hits[-1]) + 1) / sr
    return {"status": "measured", "onset_s": onset, "offset_s": offset, "span_s": offset - onset,
            "active_frame_fraction": float(active.mean()), "method": AUDIBLE_WINDOW_METHOD}


def discover_stems(mixture_path: Path | None, actor_ids: Sequence[str], stems_dir: Path | None) -> dict[str, Path]:
    """按 ``<actor>_*stem*.wav`` 找逐源湿声。

    给了 ``stems_dir`` 就只在那里找；否则依次看成片所在目录、它的 ``audio/binaural``、``binaural`` 子目录，
    以及成片目录下最多三层里的任何 ``*_stem*.wav``。找不到就没有，不猜。
    """

    candidates: list[Path] = []
    if stems_dir is not None:
        candidates = [Path(stems_dir)]
    elif mixture_path is not None:
        parent = Path(mixture_path).parent
        candidates = [parent, parent / "audio" / "binaural", parent / "binaural", parent.parent / "audio" / "binaural"]
    found: dict[str, Path] = {}
    for actor_id in actor_ids:
        matches: list[Path] = []
        for directory in candidates:
            if directory.is_dir():
                matches.extend(p for p in directory.glob(f"{actor_id}_*stem*.wav") if p.is_file())
        if not matches and stems_dir is None and mixture_path is not None:
            root = Path(mixture_path).parent
            for depth_pattern in (f"*/{actor_id}_*stem*.wav", f"*/*/{actor_id}_*stem*.wav", f"*/*/*/{actor_id}_*stem*.wav"):
                matches.extend(p for p in root.glob(depth_pattern) if p.is_file())
        if matches:
            found[actor_id] = sorted(set(matches))[0]
    return found


# ── 事件审计 ─────────────────────────────────────────────────────────────────

def event_frames(facts: EpisodeFacts, event: Mapping[str, Any]) -> tuple[int, int]:
    start = facts.clamp(int(event.get("start_frame", round(float(event["start_s"]) * facts.frame_rate))))
    end = facts.clamp(int(event.get("end_frame", round(float(event["end_s"]) * facts.frame_rate))))
    return start, max(start, end)


def max_concurrent_entities(intervals: Sequence[tuple[float, float, str]]) -> int:
    """时间扫描：半开区间 [start, end)，同一实体只算一次，返回任一时刻同时活跃的实体数最大值。"""

    best = 0
    for start, _, _ in intervals:
        active = {entity for s, e, entity in intervals if s <= start < e}
        best = max(best, len(active))
    return best


def event_overlaps(facts: EpisodeFacts, event: Mapping[str, Any]) -> dict[str, Any]:
    start, end = float(event["start_s"]), float(event["end_s"])
    overlaps = []
    total = 0.0
    intervals: list[tuple[float, float, str]] = [(start, end, str(event.get("actor_id")))]
    for other in facts.events:
        if other.get("event_id") == event.get("event_id"):
            continue
        lo, hi = max(start, float(other["start_s"])), min(end, float(other["end_s"]))
        if lo < hi:
            overlaps.append({"event_id": other.get("event_id"), "actor_id": other.get("actor_id"), "overlap_s": hi - lo})
            total += hi - lo
            intervals.append((lo, hi, str(other.get("actor_id"))))
    return {
        "overlapping_events": overlaps,
        "overlap_total_s": total,
        "max_concurrent_speakers": max_concurrent_entities(intervals),
        "concurrency_method": "time_sweep_half_open_dedup_by_entity",
    }


def separation_stats(facts: EpisodeFacts, actor_id: str, competitors: Sequence[str], start_f: int, end_f: int,
                     theta_deg: float) -> dict[str, Any]:
    """说话窗内每一帧与最近竞争者的夹角：起止、最小最大、分位数、连续达阈时长、最近竞争者是否换人。"""

    separations: list[float] = []
    nearest: list[str] = []
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
            nearest.append(best[1])
    if not separations:
        return {"frames_measured": 0, "onset": None, "end": None, "min": None, "max": None, "change": None,
                "p10": None, "p50": None, "sustained_s_above_theta": 0.0, "theta_deg": theta_deg,
                "nearest_competitor_at_onset": None, "nearest_competitor_ids": [], "nearest_competitor_changes": 0,
                "speaker_azimuths_deg": speaker_azimuths}
    arr = np.asarray(separations, dtype=np.float64)
    above = arr >= theta_deg
    best_run, run = 0, 0
    for flag in above:
        run = run + 1 if flag else 0
        best_run = max(best_run, run)
    changes = sum(1 for a, b in zip(nearest, nearest[1:]) if a != b)
    return {
        "frames_measured": int(arr.size),
        "onset": float(arr[0]),
        "end": float(arr[-1]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "change": float(arr.max() - arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "sustained_s_above_theta": best_run / facts.frame_rate,
        "theta_deg": theta_deg,
        "nearest_competitor_at_onset": nearest[0],
        "nearest_competitor_ids": sorted(set(nearest)),
        "nearest_competitor_changes": changes,
        "speaker_azimuths_deg": speaker_azimuths,
    }


def line_of_sight_record(facts: EpisodeFacts, actor_id: str, frames: Sequence[int],
                         geometry: StaticGeometry | dict[str, Any] | None) -> dict[str, Any]:
    if geometry is None:
        return {"state": STATE_UNMEASURED, "reason": "no static geometry supplied (--acoustic-package)"}
    if isinstance(geometry, dict):
        return {"state": STATE_UNMEASURED, "reason": geometry.get("reason"), "source": geometry.get("manifest")}
    per_frame: dict[str, str] = {}
    for frame in frames:
        emitter, listener = facts.position(actor_id, frame), facts.listener_position(frame)
        if emitter is None or listener is None:
            per_frame[str(frame)] = STATE_UNMEASURED
            continue
        per_frame[str(frame)] = "blocked" if geometry.segment_blocked(listener, emitter) else "clear"
    states = set(per_frame.values())
    if not states or states == {STATE_UNMEASURED}:
        state = STATE_UNMEASURED
    elif "blocked" in states and "clear" in states:
        state = "partially_blocked"
    elif "blocked" in states:
        state = "blocked"
    else:
        state = "clear"
    return {"state": state, "frames": per_frame, "source": geometry.source,
            "triangle_count": geometry.triangle_count, "ray": "listener_position_to_registered_emitter_static_mesh_only"}


def audit_event(facts: EpisodeFacts, event: Mapping[str, Any], audio: tuple[int, np.ndarray] | None,
                thresholds: Mapping[str, float], *, stems: Mapping[str, tuple[int, np.ndarray]] | None = None,
                geometry: StaticGeometry | dict[str, Any] | None = None) -> dict[str, Any]:
    actor_id = str(event.get("actor_id"))
    start_f, end_f = event_frames(facts, event)
    competitors = [a for a in facts.actors if a != actor_id]
    theta = float(thresholds["theta_static_deg"])
    start_s, end_s = float(event["start_s"]), float(event["end_s"])

    # 可听窗：有 stem 就从 stem 读；没有就退回放置区间，并如实标注。
    audible: dict[str, Any] = {"status": STATE_UNMEASURED, "reason": "no per-source stem found"}
    stem = (stems or {}).get(actor_id)
    if stem is not None:
        sr_stem, data_stem = stem
        audible = audible_window_from_stem(data_stem, sr_stem, start_s, end_s)
        audible["stem_path_used"] = True
    if audible.get("status") == "measured":
        a_start_f, a_end_f = facts.frame_of_time(audible["onset_s"]), facts.frame_of_time(audible["offset_s"])
        window_used = "audible_window_from_stem"
        s_start_f, s_end_f = a_start_f, max(a_start_f, a_end_f)
    else:
        window_used = "placement_window"
        s_start_f, s_end_f = start_f, end_f

    placement_sep = separation_stats(facts, actor_id, competitors, start_f, end_f, theta)
    states_sep = placement_sep if window_used == "placement_window" else separation_stats(
        facts, actor_id, competitors, s_start_f, s_end_f, theta)
    speaker_azimuths = placement_sep.pop("speaker_azimuths_deg")
    states_sep.pop("speaker_azimuths_deg", None)

    onset_az = facts.azimuth(actor_id, start_f)
    end_az = facts.azimuth(actor_id, end_f)
    speaker_sweep = None
    if speaker_azimuths:
        speaker_sweep = {
            "net_deg": signed_circular_delta_deg(speaker_azimuths[0], speaker_azimuths[-1]),
            "path_deg": float(sum(abs(signed_circular_delta_deg(a, b)) for a, b in zip(speaker_azimuths, speaker_azimuths[1:]))),
        }
    nearest_at_onset = states_sep.get("nearest_competitor_at_onset")
    competitor_az_at_onset = facts.azimuth(nearest_at_onset, s_start_f) if nearest_at_onset else None

    mid_f = (s_start_f + s_end_f) // 2
    record: dict[str, Any] = {
        "event_id": event.get("event_id"),
        "actor_id": actor_id,
        "appearance": facts.appearance(actor_id),
        "sound_class": event.get("sound_class"),
        "transcript": event.get("transcript"),
        "start_s": start_s,
        "end_s": end_s,
        "start_frame": start_f,
        "end_frame": end_f,
        "audible_window": audible,
        "window_used_for_states": window_used,
        "state_window_frames": [s_start_f, s_end_f],
        "azimuth_convention": AZIMUTH_CONVENTION,
        "onset_azimuth_deg": onset_az,
        "end_azimuth_deg": end_az,
        "state_window_onset_azimuth_deg": facts.azimuth(actor_id, s_start_f),
        "nearest_competitor_azimuth_at_state_onset_deg": competitor_az_at_onset,
        "onset_distance_m": facts.distance(actor_id, start_f),
        "end_distance_m": facts.distance(actor_id, end_f),
        "onset_visibility_state": facts.vis_state(actor_id, start_f),
        "competitor_count": len(competitors),
        "separation_to_nearest_competitor_deg": placement_sep,
        "separation_over_state_window_deg": states_sep,
        "speaker_azimuth_sweep_deg": speaker_sweep,
        "speaker_moving_during_event": facts.moving_any(actor_id, start_f, end_f),
        "competitors_moving_during_event": [
            other for other in competitors if facts.moving_any(other, start_f, end_f)
        ],
        "line_of_sight": line_of_sight_record(facts, actor_id, [s_start_f, mid_f, s_end_f], geometry),
        **event_overlaps(facts, event),
    }
    record["mixture_contaminated"] = bool(record["overlapping_events"])
    record["mixture_contaminating_event_ids"] = [o["event_id"] for o in record["overlapping_events"]]

    measured_source = None
    if stem is not None:
        sr_stem, data_stem = stem
        record["measured_stem"] = measure_binaural_window(data_stem, sr_stem, start_s, end_s)
        measured_source = "per_source_stem"
    if audio is not None:
        sr, data = audio
        record["measured_mixture"] = measure_binaural_window(data, sr, start_s, end_s)
        if measured_source is None:
            measured_source = "final_mixture"
    if measured_source is not None:
        chosen = record["measured_stem"] if measured_source == "per_source_stem" else record["measured_mixture"]
        record["measured"] = {**chosen, "expected_abs_itd_woodworth_ms": woodworth_itd_ms(onset_az) if onset_az is not None else None}
        record["measured_source"] = measured_source
        if measured_source == "final_mixture" and record["mixture_contaminated"]:
            record["measured"]["contamination_note"] = "window overlaps other events in the final mixture; cues are not attributable to this source alone"

    record["binding_feasibility"] = binding_diagnostics(record, thresholds)
    return record


def binding_diagnostics(event_record: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    """几何与成片线索各给一个三态，绝不合并成一个通过布尔值。占位阈值，等人工校准。"""

    sep = event_record.get("separation_over_state_window_deg") or {}
    measured = event_record.get("measured") or {}
    reasons: list[str] = []
    no_competitor = int(event_record.get("competitor_count", 0)) == 0

    # 几何：整窗最小值和连续达阈时长。
    geometry_basis = {
        "min_deg": sep.get("min"),
        "p10_deg": sep.get("p10"),
        "sustained_s_above_theta": sep.get("sustained_s_above_theta"),
        "theta_static_deg": thresholds["theta_static_deg"],
        "min_sustained_separation_s": thresholds["min_sustained_separation_s"],
        "relative_motion_deg": sep.get("change"),
        "nearest_competitor_changes": sep.get("nearest_competitor_changes"),
        "window": event_record.get("window_used_for_states"),
    }
    if no_competitor:
        geometry_state = STATE_NO_COMPETITOR
        reasons.append("no competitor: binding is trivial")
    elif sep.get("min") is None:
        geometry_state = STATE_UNMEASURED
        reasons.append("no frames with both target and competitor positions")
    elif float(sep.get("sustained_s_above_theta") or 0.0) >= float(thresholds["min_sustained_separation_s"]):
        geometry_state = STATE_PASS
    else:
        geometry_state = STATE_FAIL
        reasons.append("nearest competitor stays within theta_static for the whole window or the separated stretch is too short")

    # 线索：目标与最近竞争者的预期时差之差，加实测方向一致性。
    target_az = event_record.get("state_window_onset_azimuth_deg")
    competitor_az = event_record.get("nearest_competitor_azimuth_at_state_onset_deg")
    exp_target = expected_signed_itd_ms(target_az)
    exp_comp = expected_signed_itd_ms(competitor_az)
    pairwise = abs(exp_target - exp_comp) if exp_target is not None and exp_comp is not None else None
    itd, cc, ild = measured.get("itd_onset_ms"), measured.get("itd_onset_cc"), measured.get("ild_2_6k_db")
    near_axis = target_az is not None and abs((float(target_az) + 180.0) % 360.0 - 180.0) < float(thresholds["near_axis_deg"])
    itd_consistent = None
    if itd is not None and exp_target is not None:
        if near_axis:
            itd_consistent = abs(float(itd)) < float(thresholds["itd_min_ms"])
        elif cc is not None and float(cc) < float(thresholds["itd_min_cc"]):
            itd_consistent = None  # 相关太低，符号不可信
        else:
            itd_consistent = math.copysign(1.0, float(itd)) == math.copysign(1.0, exp_target)
    ild_consistent = None
    if ild is not None and target_az is not None:
        value = (float(target_az) + 180.0) % 360.0 - 180.0
        if near_axis:
            ild_consistent = abs(float(ild)) < float(thresholds["ild_min_db"])
        else:
            # 右为正约定：声源在右 → 右耳更响 → 左减右为负。
            ild_consistent = (float(ild) < 0.0) == (value > 0.0) if abs(float(ild)) >= 0.5 else None
    cue_basis = {
        "measured_source": event_record.get("measured_source"),
        "mixture_contaminated": event_record.get("mixture_contaminated"),
        "measured_itd_onset_ms": itd,
        "measured_itd_cc": cc,
        "measured_ild_2_6k_db": ild,
        "expected_itd_target_ms": exp_target,
        "expected_itd_nearest_competitor_ms": exp_comp,
        "pairwise_expected_itd_diff_ms": pairwise,
        "itd_min_ms": thresholds["itd_min_ms"],
        "target_near_axis": near_axis,
        "itd_direction_consistent": itd_consistent,
        "ild_direction_consistent": ild_consistent,
    }
    if no_competitor:
        cue_state = STATE_NO_COMPETITOR
    elif not measured or (itd is None and ild is None):
        cue_state = STATE_UNMEASURED
        reasons.append("no delivered audio measured for this event")
    elif pairwise is None:
        cue_state = STATE_UNMEASURED
        reasons.append("cannot compute expected cue difference (missing azimuths)")
    elif pairwise < float(thresholds["itd_min_ms"]):
        cue_state = STATE_FAIL
        reasons.append("target and nearest competitor would produce nearly the same interaural time difference")
    elif itd_consistent is True or ild_consistent is True:
        cue_state = STATE_PASS
    elif itd_consistent is None and ild_consistent is None:
        cue_state = STATE_UNMEASURED
        reasons.append("measured cues too weak or too decorrelated to check direction")
    else:
        cue_state = STATE_FAIL
        reasons.append("measured interaural cues point the wrong way relative to the registered geometry")

    return {
        "geometry_state": geometry_state,
        "geometry_basis": geometry_basis,
        "delivered_cue_state": cue_state,
        "cue_basis": cue_basis,
        # 兼容旧读法的三态别名：True / False / None；None 永远不等于通过。
        "geometry_separable": True if geometry_state == STATE_PASS else False if geometry_state == STATE_FAIL else None,
        "delivered_cues_present": True if cue_state == STATE_PASS else False if cue_state == STATE_FAIL else None,
        "reasons": reasons,
        "calibration": CALIBRATION_NOTE,
        "note": "no combined verdict on purpose: geometry candidates and delivered-audio evidence are reported separately",
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


def numeric_gold_of(item: Mapping[str, Any]) -> float | None:
    truth = item.get("truth")
    if isinstance(truth, Mapping) and isinstance(truth.get("value"), (int, float)) and not isinstance(truth.get("value"), bool):
        return float(truth["value"])
    return None


def available_forms(item: Mapping[str, Any]) -> list[str]:
    """题目实际可用的形式。显式 ``available_forms`` 优先；其次 ``form_status``/``forms`` 里没被 deferred 的；都没有就当两种都在。"""

    declared = item.get("available_forms")
    if isinstance(declared, Sequence) and not isinstance(declared, str):
        return [str(f) for f in declared]
    status = item.get("form_status")
    if isinstance(status, Mapping) and status:
        return [str(f) for f, rec in status.items() if not (isinstance(rec, Mapping) and rec.get("status") == "deferred")]
    forms = item.get("forms")
    if isinstance(forms, Mapping) and forms:
        return [str(f) for f, rec in forms.items() if not (isinstance(rec, Mapping) and rec.get("status") == "deferred")]
    return ["open", "mcq"]


def divergence(values: Mapping[str, Any], target: str | None, gold: Any) -> dict[str, Any]:
    """其他候选取值与金标的关系（离散答案域）。"""

    others = {k: v for k, v in values.items() if k != target}
    same = [k for k, v in others.items() if v == gold]
    all_values = list(values.values())
    gold_count = sum(1 for v in all_values if v == gold)
    counts: dict[str, int] = {}
    for v in all_values:
        counts[json.dumps(v, ensure_ascii=False, sort_keys=True)] = counts.get(json.dumps(v, ensure_ascii=False, sort_keys=True), 0) + 1
    return {
        "form": "discrete",
        "candidate_values": values,
        "candidate_value_multiplicity": counts,
        "distractors_equal_to_gold": same,
        "degenerate_all_same": bool(others) and len(same) == len(others),
        "gold_is_majority": gold_count * 2 > len(all_values) if all_values else None,
        "gold_is_unique_minority": (gold_count == 1 and len(all_values) >= 3) if all_values else None,
    }


def numeric_divergence(values: Mapping[str, float | None], target: str | None, min_gap: float, *,
                       circular: bool, unit: str) -> dict[str, Any]:
    """数值答案域（Open 形式）：目标与最近干扰项的数值间隙。"""

    gold = values.get(target) if target else None
    gaps = {}
    for actor, value in values.items():
        if actor == target or value is None or gold is None:
            continue
        gaps[actor] = circular_diff_deg(gold, value) if circular else abs(float(gold) - float(value))
    nearest = min(gaps.values()) if gaps else None
    return {
        "form": "open",
        "candidate_values": values,
        "gold_value": gold,
        f"gap_to_nearest_distractor_{unit}": nearest,
        f"min_gap_{unit}": min_gap,
        "degenerate_all_same": (nearest is not None and nearest < min_gap),
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


def _attach_form_divergences(out: dict[str, Any], by_form: Mapping[str, Mapping[str, Any]]) -> None:
    out["divergence_by_form"] = dict(by_form)
    primary = by_form.get("mcq") or by_form.get("open") or next(iter(by_form.values()), None)
    out["divergence"] = dict(primary) if primary else None


def audit_question(facts: EpisodeFacts, item: Mapping[str, Any], events_by_id: Mapping[str, Mapping[str, Any]],
                   thresholds: Mapping[str, float]) -> dict[str, Any]:
    qa_id = str(item.get("qa_id"))
    evidence = item.get("evidence") or {}
    gold = gold_of(item)
    forms = available_forms(item)
    anchor_id = anchor_event_id(evidence)
    anchor = facts.event_by_id(anchor_id)
    target = evidence.get("target_actor_id") or (anchor.get("actor_id") if anchor else None)
    group = ("binding" if qa_id in BINDING_TYPES else "conditional" if qa_id in CONDITIONAL_TYPES
             else "audio_control" if qa_id in AUDIO_CONTROL_TYPES else "visual_control" if qa_id in VISUAL_CONTROL_TYPES
             else "unclassified")
    listening = events_by_id.get(anchor_id, {}).get("binding_feasibility") if anchor_id else None
    out: dict[str, Any] = {
        "qa_id": qa_id,
        "question_id": item.get("question_id"),
        "status": item.get("status"),
        "group": group,
        "available_forms": forms,
        "gold": gold,
        "anchor_event_id": anchor_id,
        "target_actor_id": target,
        "target_appearance": facts.appearance(target) if target else None,
        "silent_target": bool(target) and facts.first_event_of(target) is None,
        "listening": listening,
        "listening_event": (
            {k: events_by_id[anchor_id].get(k) for k in (
                "separation_over_state_window_deg", "measured", "measured_source", "mixture_contaminated",
                "max_concurrent_speakers", "speaker_moving_during_event", "competitors_moving_during_event",
                "line_of_sight", "audible_window")}
            if anchor_id in events_by_id else None
        ),
        "seeing": None,
        "temporal": None,
        "divergence": None,
        "divergence_by_form": None,
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
        _attach_form_divergences(out, {"mcq": divergence(values, target, gold)})
        out["seeing"] = seeing_profile(facts, target, [s, e])
    elif qa_id == "QA-08" and anchor:
        frame = int(evidence.get("query_frame", event_frames(facts, anchor)[0]))
        values = {a: facts.vis_state(a, frame) for a in actors}
        _attach_form_divergences(out, {"mcq": divergence(values, target, gold)})
        out["seeing"] = seeing_profile(facts, target, [frame])
        out["flags"].append("truth_is_onset_frame_but_question_says_during")
    elif qa_id == "QA-24":
        frame = int(evidence.get("final_frame", last))
        values = {a: facts.vis_state(a, frame) for a in actors}
        _attach_form_divergences(out, {"mcq": divergence(values, target, gold)})
        out["seeing"] = seeing_profile(facts, target, [frame])
    elif qa_id == "QA-13" and anchor and target:
        query = int(evidence.get("query_frame", (evidence.get("post_sound") or {}).get("query_frame", last)))
        _, end_f = event_frames(facts, anchor)
        az_end, az_query = facts.azimuth(target, end_f), facts.azimuth(target, query)
        others = {a: facts.azimuth(a, query) for a in actors if a != target}
        displacement = signed_circular_delta_deg(az_end, az_query) if az_end is not None and az_query is not None else None
        by_form: dict[str, Mapping[str, Any]] = {}
        if "mcq" in forms:
            values = {a: (sector_of(v) if v is not None else None) for a, v in {**others, target: az_query}.items()}
            by_form["mcq"] = divergence(values, target, sector_of(az_query) if az_query is not None else None)
            by_form["mcq"]["answer_domain"] = "equal_width_half_open_sectors"
        if "open" in forms:
            by_form["open"] = numeric_divergence({**others, target: az_query}, target, thresholds["open_angle_min_gap_deg"],
                                                 circular=True, unit="deg")
        _attach_form_divergences(out, by_form)
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
            "target_visibility_at_query": facts.vis_state(target, query),
            "query_frame": query,
        }
        if "mcq" in forms and out["temporal"]["gold_sector_boundary_margin_deg"] is not None and out["temporal"]["gold_sector_boundary_margin_deg"] < thresholds["sector_margin_deg"]:
            out["flags"].append("mcq_gold_within_sector_boundary_margin")
        if out["temporal"]["answer_equals_last_heard_direction"]:
            out["flags"].append("post_sound_answer_equals_last_heard_direction")
        if out["temporal"]["target_visibility_at_query"] is not None and out["temporal"]["target_visibility_at_query"] not in VISIBLE_STATES:
            out["flags"].append("target_unobservable_at_query")
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
        div = divergence(values, target, gold)
        div["distance_delta_m"] = deltas
        _attach_form_divergences(out, {"mcq": div})
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
        _attach_form_divergences(out, {"mcq": divergence(values, target, gold)})
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
        div = divergence(values, target, gold)
        div["distance_delta_m"] = deltas
        _attach_form_divergences(out, {"mcq": div})
        out["seeing"] = seeing_profile(facts, target, [s, e2])
    elif qa_id == "QA-19" and target:
        first_times = {}
        for a in actors:
            first = facts.first_event_of(a)
            first_times[a] = float(first["start_s"]) if first else None
        by_form = {}
        bands = evidence.get("time_bands_s")
        if "mcq" in forms and isinstance(bands, Sequence) and bands:
            def band_of(t: float) -> str:
                for index, (lo, hi) in enumerate(bands):
                    if lo <= t < hi:
                        return f"band_{index}"
                return f"band_{len(bands) - 1}"
            values = {a: (band_of(t) if t is not None else "no_sound") for a, t in first_times.items()}
            by_form["mcq"] = divergence(values, target, gold)
            by_form["mcq"]["answer_domain"] = "time_bands"
        if "open" in forms:
            by_form["open"] = numeric_divergence(first_times, target, thresholds["open_time_min_gap_s"], circular=False, unit="s")
            by_form["open"]["note"] = "first placement starts; audible onsets need the stem-based audible window"
        if by_form:
            _attach_form_divergences(out, by_form)
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
        if out["silent_target"]:
            out["flags"].append("negative_item_silent_target_not_binding_gated")
    elif qa_id in {"QA-02", "QA-03", "QA-12", "QA-21"} and anchor:
        s, _ = event_frames(facts, anchor)
        out["seeing"] = seeing_profile(facts, target, [s])

    by_form = out.get("divergence_by_form") or {}
    degenerate_forms = [f for f, d in by_form.items() if isinstance(d, Mapping) and d.get("degenerate_all_same")]
    if degenerate_forms:
        out["flags"].append("degenerate_distractors_equal_gold")
        out["degenerate_forms"] = degenerate_forms
    elif out["divergence"] and out["divergence"].get("degenerate_all_same") and not by_form:
        out["flags"].append("degenerate_distractors_equal_gold")
    mcq_div = by_form.get("mcq") if isinstance(by_form.get("mcq"), Mapping) else None
    if mcq_div and mcq_div.get("gold_is_majority") and group == "binding":
        out["flags"].append("gold_is_majority_among_candidates")
    if mcq_div and mcq_div.get("gold_is_unique_minority") and group == "binding":
        out["flags"].append("gold_is_unique_minority_among_candidates")
    if group == "binding" and listening and not out["silent_target"]:
        if listening.get("geometry_state") == STATE_FAIL:
            out["flags"].append("binding_geometry_candidate_fail")
        if listening.get("delivered_cue_state") == STATE_FAIL:
            out["flags"].append("binding_cue_candidate_fail")
        elif listening.get("delivered_cue_state") == STATE_UNMEASURED:
            out["flags"].append("binding_cue_unmeasured")
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


def structural_baselines(questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """只看视觉结构的两种常数策略在本批绑定组 MCQ 题上的命中数：选多数值、选唯一少数值。"""

    applicable = majority_hits = minority_hits = 0
    for q in questions:
        if q.get("group") != "binding":
            continue
        by_form = q.get("divergence_by_form") or {}
        mcq = by_form.get("mcq") if isinstance(by_form, Mapping) else None
        if not isinstance(mcq, Mapping) or mcq.get("gold_is_majority") is None:
            continue
        applicable += 1
        majority_hits += 1 if mcq.get("gold_is_majority") else 0
        minority_hits += 1 if mcq.get("gold_is_unique_minority") else 0
    return {
        "applicable_binding_mcq_questions": applicable,
        "majority_value_strategy_hits": majority_hits,
        "unique_minority_strategy_hits": minority_hits,
        "note": "structure-only baselines; compare against chance for the form's option count before drawing conclusions",
    }


def wet_tail_intervals(facts_raw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    audio = facts_raw.get("audio") or {}
    out: dict[str, Mapping[str, Any]] = {}
    for item in audio.get("wet_tail_intervals") or []:
        if isinstance(item, Mapping) and item.get("event_id"):
            out[str(item["event_id"])] = item
    return out


def run_audit(facts_path: Path, wav_path: Path | None, questions_path: Path | None,
              thresholds: Mapping[str, float], *, stems_dir: Path | None = None,
              acoustic_package: Path | None = None) -> dict[str, Any]:
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
    stem_paths = discover_stems(audio_path, list(facts.actors), stems_dir)
    stems: dict[str, tuple[int, np.ndarray]] = {}
    for actor_id, path in stem_paths.items():
        sr_s, data_s = read_wav(path)
        if data_s.shape[1] == 1:
            data_s = np.repeat(data_s, 2, axis=1)
        stems[actor_id] = (sr_s, data_s)
    audio_block["per_source_stems"] = {a: str(p) for a, p in stem_paths.items()}
    geometry: StaticGeometry | dict[str, Any] | None = None
    geometry_block: dict[str, Any] = {"status": "not_supplied"}
    if acoustic_package is not None:
        geometry = load_static_geometry(Path(acoustic_package))
        geometry_block = (geometry if isinstance(geometry, dict)
                          else {"status": "loaded", "manifest": geometry.source, "triangle_count": geometry.triangle_count})
    tails = wet_tail_intervals(facts_raw)
    events = []
    for event in facts.events:
        record = audit_event(facts, event, audio, thresholds, stems=stems, geometry=geometry)
        tail = tails.get(str(event.get("event_id")))
        record["wet_tail_interval_from_facts"] = dict(tail) if tail else None
        events.append(record)
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

    def count_states(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in events:
            state = str(e["binding_feasibility"].get(key))
            counts[state] = counts.get(state, 0) + 1
        return counts

    summary = {
        "event_count": len(events),
        "events_geometry_state_counts": count_states("geometry_state"),
        "events_delivered_cue_state_counts": count_states("delivered_cue_state"),
        "events_line_of_sight_counts": {
            state: sum(1 for e in events if (e.get("line_of_sight") or {}).get("state") == state)
            for state in sorted({(e.get("line_of_sight") or {}).get("state") for e in events} - {None})
        },
        "events_measured_on_stem": sum(1 for e in events if e.get("measured_source") == "per_source_stem"),
        "events_mixture_contaminated": sum(1 for e in events if e.get("mixture_contaminated")),
        "events_speaker_moving_while_speaking": sum(1 for e in events if e.get("speaker_moving_during_event")),
        "question_count": len(questions),
        "questions_by_group": {g: sum(1 for q in questions if q["group"] == g) for g in ("binding", "conditional", "audio_control", "visual_control", "unclassified")},
        "flag_counts": flag_counts,
        "structural_baselines": structural_baselines(questions),
        "note": "no combined feasibility count: unmeasured never counts as pass",
    }
    return {
        "schema": SCHEMA,
        "status": "research_only",
        "claim_boundary": (
            "Numbers are recomputed from delivered facts.json, per-source stems when present and the delivered stereo WAV. "
            "They describe answerability structure and difficulty; they are not model results, not human-answerability "
            "certificates and not formal admission. Thresholds are placeholders pending human calibration; "
            "geometry and delivered-cue states are reported separately and never merged."
        ),
        "inputs": {"facts": str(facts_path), "wav": audio_block.get("path"),
                   "questions": str(questions_path) if questions_path else None,
                   "stems_dir": str(stems_dir) if stems_dir else None,
                   "acoustic_package": str(acoustic_package) if acoustic_package else None},
        "episode_id": facts_raw.get("episode_id"),
        "catalog_version": facts_raw.get("catalog_version"),
        "azimuth_convention": AZIMUTH_CONVENTION,
        "azimuth_formula_crosscheck": crosscheck_azimuth_formula(facts),
        "thresholds": {**thresholds, "calibration": CALIBRATION_NOTE},
        "audio": audio_block,
        "static_geometry": geometry_block,
        "events": events,
        "questions": questions,
        "summary": summary,
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    a = payload["audio"]
    print(f"episode {payload.get('episode_id')}: {s['event_count']} events; geometry states {s['events_geometry_state_counts']}; "
          f"delivered-cue states {s['events_delivered_cue_state_counts']}; line of sight {s['events_line_of_sight_counts']}; "
          f"{s['events_measured_on_stem']} measured on stems, {s['events_mixture_contaminated']} mixture windows contaminated; "
          f"{s['events_speaker_moving_while_speaking']} speaker(s) moving while speaking")
    if a.get("status") == "measured":
        print(f"  audio: peak {a['peak_dbfs']:.1f} dBFS, rms {a['rms_dbfs']:.1f} dBFS, exact-zero fraction {a['exact_zero_sample_fraction']:.3f}; stems for {sorted(a.get('per_source_stems', {}))}")
    for e in payload["events"]:
        sep = e["separation_over_state_window_deg"]
        m = e.get("measured") or {}
        bf = e["binding_feasibility"]
        aw = e.get("audible_window") or {}
        onset_txt = f"audible {aw['onset_s']:.3f}-{aw['offset_s']:.3f}s" if aw.get("status") == "measured" else "audible window unmeasured"
        if sep.get("onset") is not None:
            print(f"  {e['event_id']} {e['actor_id']}({e['appearance']}) az {e['onset_azimuth_deg']:+.1f}° {onset_txt} "
                  f"sep min/p50/max {sep['min']:.1f}/{sep['p50']:.1f}/{sep['max']:.1f}° sustained {sep['sustained_s_above_theta']:.2f}s "
                  f"ILD2-6k {m.get('ild_2_6k_db') if m.get('ild_2_6k_db') is None else round(m['ild_2_6k_db'], 1)} dB "
                  f"ITD {m.get('itd_onset_ms') if m.get('itd_onset_ms') is None else round(m['itd_onset_ms'], 3)} ms "
                  f"[{e.get('measured_source')}] -> geometry={bf['geometry_state']} cue={bf['delivered_cue_state']} los={(e.get('line_of_sight') or {}).get('state')}")
        else:
            print(f"  {e['event_id']} {e['actor_id']} (no competitor frames) -> geometry={bf['geometry_state']} cue={bf['delivered_cue_state']}")
    if s["question_count"]:
        print(f"  questions: {s['question_count']} ({s['questions_by_group']}); flags: {s['flag_counts']}; structural baselines: {s['structural_baselines']}")
    xc = payload["azimuth_formula_crosscheck"]
    print(f"  azimuth formula crosscheck vs unified_catalog: {xc.get('status')}"
          + (f" (max diff {xc['max_abs_azimuth_diff_deg']:.2e}°)" if xc.get("status") in {"matched", "mismatch"} else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--facts", required=True, type=Path, help="delivery 目录里的 facts.json")
    parser.add_argument("--wav", type=Path, default=None, help="成片双耳 WAV；缺省取 facts.audio.path")
    parser.add_argument("--stems-dir", type=Path, default=None, help="逐源 stem 目录；缺省在成片目录及其 audio/binaural 子目录里找")
    parser.add_argument("--acoustic-package", type=Path, default=None, help="RLR 声学包 manifest，用于直达射线（可选）")
    parser.add_argument("--questions", type=Path, default=None, help="同一 delivery 目录的 questions.json（可选）")
    parser.add_argument("--out", required=True, type=Path, help="输出 JSON，已存在则拒绝")
    for key, value in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=value, help=f"占位阈值，默认 {value}")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    thresholds = {key: float(getattr(args, key)) for key in DEFAULT_THRESHOLDS}
    payload = run_audit(args.facts, args.wav, args.questions, thresholds, stems_dir=args.stems_dir,
                        acoustic_package=args.acoustic_package)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print_summary(payload)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
