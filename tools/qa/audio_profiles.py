#!/usr/bin/env python3
"""Question-type audio profiles: one schedule per question type, not one for all.

run01 的 card8 被压扁,是因为它继承了 card1 需要的片尾静默 —— 一套声音
时间表不该同时承担所有题型。这里每种题型声明自己的事件调度:什么时候
必须有声、什么时候必须静、哪一声承担"身份锚"、哪一声只是对照。

三条硬约定:
  1. 发声者用**语义角色**(target_actor / non_target_actor),不写死
     source1/source2;绑定到槽位是后一步的事,由批次的反平衡决定。
  2. 时间窗按**片长比例**声明,不绑定某个房间或某个绝对秒数;需要秒
     的地方由 clip_seconds 算出来。
  3. 每次调度完**自检**:profile 声明的性质必须在产出的事件表上成立,
     不成立就抛错,而不是让一条不满足约束的题混进批次。

窗内时刻随机(不固定在某一秒),否则"最后一声总是在第 X 秒"本身就是
可学的模板。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SAMPLE_RATE = 16000
CLIP_SECONDS = 5.0
FRAME_COUNT = 75
EVENT_SECONDS = 0.3

TARGET = "target_actor"
OTHER = "non_target_actor"


class AudioProfileError(ValueError):
    """A schedule cannot satisfy the profile's declared constraints."""


@dataclass
class ScheduledEvent:
    role: str
    start_sample: int
    end_sample_exclusive: int
    purpose: str          # identity_anchor | answer_evidence | control_sound

    @property
    def start_seconds(self) -> float:
        return self.start_sample / SAMPLE_RATE

    def frame_span(self) -> tuple[int, int]:
        ticks_per_frame = 3200
        t0 = self.start_sample * 3
        t1 = self.end_sample_exclusive * 3
        return (t0 // ticks_per_frame,
                min(FRAME_COUNT, -(-t1 // ticks_per_frame)))


@dataclass
class Schedule:
    profile_id: str
    events: list[ScheduledEvent]
    anchor_index: int
    declared: dict = field(default_factory=dict)

    @property
    def anchor(self) -> ScheduledEvent:
        return self.events[self.anchor_index]

    def as_role_events(self) -> list[tuple[str, int]]:
        return [(e.role, e.start_sample) for e in self.events]

    def bind(self, role_to_slot: dict[str, str]) -> list[tuple[str, int]]:
        """语义角色 → 槽位。批次的反平衡决定这张映射表。"""
        missing = {e.role for e in self.events} - set(role_to_slot)
        if missing:
            raise AudioProfileError(f"unbound roles: {sorted(missing)}")
        return [(role_to_slot[e.role], e.start_sample) for e in self.events]


def _event_len() -> int:
    return int(EVENT_SECONDS * SAMPLE_RATE)


def _clip_samples() -> int:
    return int(CLIP_SECONDS * SAMPLE_RATE)


def _fraction_to_sample(fraction: float) -> int:
    return int(round(fraction * CLIP_SECONDS * SAMPLE_RATE))


def schedule_forward_anchor(rng, *, params, anchor_frame: int) -> Schedule:
    """①F 正向错时:身份锚在前,查询在片尾,锚后到片尾必须静。

    锚是**最后一声**;片尾静默长度按片长比例声明(tail_silence_fraction)。
    前面还要有至少两声,让"哪一声是第二声"这类指代有意义,且两个角色
    都发过声(锚定不是唯一线索)。
    """
    tail_fraction = float(params["TAIL_SILENCE_FRACTION"])
    gap = int(float(params["GAP_MIN_S"]) * SAMPLE_RATE)
    first_min = int(float(params["FIRST_MIN_S"]) * SAMPLE_RATE)
    event_len = _event_len()
    anchor_start = int(round(anchor_frame / FRAME_COUNT * _clip_samples()))
    tail = _clip_samples() - (anchor_start + event_len)
    if tail < _fraction_to_sample(tail_fraction):
        raise AudioProfileError(
            f"anchor at frame {anchor_frame} leaves {tail / SAMPLE_RATE:.2f}s "
            f"of tail silence, below the declared "
            f"{tail_fraction * CLIP_SECONDS:.2f}s")
    step = event_len + gap
    n_before = int(rng.integers(2, 4))          # 锚之前 2 或 3 声
    latest_first = anchor_start - n_before * step
    if latest_first < first_min:
        n_before = 2
        latest_first = anchor_start - n_before * step
        if latest_first < first_min:
            raise AudioProfileError(
                "no room for the pre-anchor calls before the anchor instant")
    starts: list[int] = []
    cursor = first_min
    for index in range(n_before):
        remaining = n_before - 1 - index
        hi = anchor_start - (remaining + 1) * step
        starts.append(int(rng.integers(cursor, hi + 1)))
        cursor = starts[-1] + step
    # 锚定者与前面的声音交替,保证两个角色都发过声
    roles = [TARGET if i % 2 == 0 else OTHER for i in range(n_before)]
    if roles.count(TARGET) == 0 or roles.count(OTHER) == 0:
        roles[-1] = OTHER if roles[0] == TARGET else TARGET
    events = [ScheduledEvent(role, start, start + event_len, "control_sound")
              for role, start in zip(roles, starts)]
    events.append(ScheduledEvent(TARGET, anchor_start,
                                 anchor_start + event_len, "identity_anchor"))
    schedule = Schedule("card1F", events, len(events) - 1,
                        {"tail_silence_seconds": tail / SAMPLE_RATE,
                         "anchor_relation": "anchor_before_query",
                         "query_frame": FRAME_COUNT - 1})
    _self_check_forward(schedule, params)
    return schedule


def schedule_backward_anchor(rng, *, params, anchor_frame: int,
                             query_frame: int) -> Schedule:
    """①B 反向错时:视觉查询在前,身份锚在末段。

    要求(与正向相反):**查询时刻附近目标必须静**,否则听声即可定位,
    题退化成即时 DoA;锚在查询之后,由末段发声确定身份。
    """
    if query_frame >= anchor_frame:
        raise AudioProfileError(
            "backward cross-time needs the visual query before the audio anchor")
    silence_fraction = float(params["QUERY_SILENCE_FRACTION"])
    guard = _fraction_to_sample(silence_fraction)
    event_len = _event_len()
    gap = int(float(params["GAP_MIN_S"]) * SAMPLE_RATE)
    first_min = int(float(params["FIRST_MIN_S"]) * SAMPLE_RATE)
    query_sample = int(round(query_frame / FRAME_COUNT * _clip_samples()))
    anchor_start = int(round(anchor_frame / FRAME_COUNT * _clip_samples()))
    if anchor_start + event_len > _clip_samples():
        raise AudioProfileError("the anchor event runs past the clip")
    window = (query_sample - guard, query_sample + guard)
    # 锚之前安排 1-2 声对照;它们既不能落进查询静默窗,也不能压到锚上
    starts: list[int] = []
    roles: list[str] = []
    n_before = int(rng.integers(1, 3))
    cursor = first_min
    for index in range(n_before):
        hi = anchor_start - (n_before - index) * (event_len + gap)
        if cursor > hi:
            break
        for _attempt in range(40):
            candidate = int(rng.integers(cursor, hi + 1))
            if candidate + event_len <= window[0] or candidate >= window[1]:
                starts.append(candidate)
                roles.append(OTHER if index % 2 == 0 else TARGET)
                cursor = candidate + event_len + gap
                break
        else:
            continue
    events = [ScheduledEvent(role, start, start + event_len, "control_sound")
              for role, start in zip(roles, starts)]
    events.append(ScheduledEvent(TARGET, anchor_start,
                                 anchor_start + event_len, "identity_anchor"))
    schedule = Schedule("card1B", events, len(events) - 1,
                        {"anchor_relation": "anchor_after_query",
                         "query_frame": query_frame,
                         "query_silence_window_samples": list(window),
                         "query_silence_seconds": guard / SAMPLE_RATE})
    _self_check_backward(schedule, params, window)
    return schedule


def schedule_first_call_bands(rng, *, params, target_bands: tuple[int, int],
                              band_edges: list[float] | None = None,
                              first_caller_role: str = TARGET) -> Schedule:
    """⑧ 首叫时间带:两个角色的首叫落进**预先声明**的不同带。

    不继承 card1 的片尾静默 —— 那正是 run01 把首叫压进前 2.6 秒、后两带
    结构性为空的原因。带边由题型配置声明,窗内时刻随机。
    """
    bands = ([float(b) for b in band_edges] if band_edges
             else [float(b) for b in params["BANDS_CARD8"]])
    b1, b2 = target_bands
    if not 0 <= b1 < b2 <= len(bands) - 2:
        raise AudioProfileError(
            f"unreachable band pair {target_bands}: events alternate in time "
            "so the first caller's band index must be the smaller one")
    event_len = _event_len()
    gap = int(float(params["GAP_MIN_S"]) * SAMPLE_RATE)
    min_first_gap = int(float(params["T_HALF"]) * SAMPLE_RATE)
    second_role = OTHER if first_caller_role == TARGET else TARGET
    lo1, hi1 = int(bands[b1] * SAMPLE_RATE), int(bands[b1 + 1] * SAMPLE_RATE)
    lo2, hi2 = int(bands[b2] * SAMPLE_RATE), int(bands[b2 + 1] * SAMPLE_RATE)
    limit = _clip_samples() - event_len
    for _attempt in range(400):
        t1 = int(rng.integers(lo1, min(hi1, limit)))
        t2_lo = max(t1 + event_len + gap, lo2, t1 + min_first_gap + 1)
        t2_hi = min(hi2, limit)
        if t2_lo >= t2_hi:
            continue
        t2 = int(rng.integers(t2_lo, t2_hi))
        events = [
            ScheduledEvent(first_caller_role, t1, t1 + event_len,
                           "answer_evidence"),
            ScheduledEvent(second_role, t2, t2 + event_len, "answer_evidence"),
        ]
        # 第三声让"每集≥3 声"成立,且不改变任一方的首叫
        third_lo = t2 + event_len + gap
        if third_lo <= limit:
            t3 = int(rng.integers(third_lo, limit + 1))
            events.append(ScheduledEvent(first_caller_role, t3,
                                         t3 + event_len, "control_sound"))
        schedule = Schedule("card8", events, len(events) - 1,
                            {"target_bands": [b1, b2],
                             "band_edges_seconds": bands,
                             "first_caller_role": first_caller_role})
        _self_check_first_call_bands(schedule, params, target_bands,
                                     band_edges=bands)
        return schedule
    raise AudioProfileError(
        f"no onset layout lands the first calls in bands {target_bands}")


def schedule_exactly_one_calling(rng, *, params, query_frame: int) -> Schedule:
    """⑦ 指定时刻恰好一只在叫:围绕查询帧安排唯一发声者。"""
    event_len = _event_len()
    gap = int(float(params["GAP_MIN_S"]) * SAMPLE_RATE)
    query_sample = int(round(query_frame / FRAME_COUNT * _clip_samples()))
    start = max(0, min(query_sample - event_len // 2,
                       _clip_samples() - event_len))
    events = [ScheduledEvent(TARGET, start, start + event_len,
                             "answer_evidence")]
    # 另一角色的声音必须完全避开查询帧所在事件窗
    limit = _clip_samples() - event_len
    for _attempt in range(60):
        other = int(rng.integers(0, limit + 1))
        if other + event_len + gap <= start or other >= start + event_len + gap:
            events.append(ScheduledEvent(OTHER, other, other + event_len,
                                         "control_sound"))
            break
    events.sort(key=lambda e: e.start_sample)
    schedule = Schedule("card7", events,
                        max(range(len(events)),
                            key=lambda i: events[i].start_sample),
                        {"query_frame": query_frame,
                         "exactly_one_calling_at_query": True})
    _self_check_exactly_one(schedule, query_frame)
    return schedule


def schedule_first_sound_at_frame(rng, *, params, query_frame: int) -> Schedule:
    """Card3 control: the target makes the first sound at a declared frame."""
    event_len = _event_len()
    gap = int(float(params["GAP_MIN_S"]) * SAMPLE_RATE)
    first_start = int(round(query_frame * 3200 / 3))
    limit = _clip_samples() - event_len
    if first_start < 0 or first_start > limit:
        raise AudioProfileError(
            f"query frame {query_frame} cannot host the first event")
    second_min = first_start + event_len + gap
    third_min = second_min + event_len + gap
    if third_min > limit:
        raise AudioProfileError(
            "no room for three separated events after the first sound")
    second_start = int(rng.integers(second_min, limit - event_len - gap + 1))
    third_start = int(rng.integers(
        second_start + event_len + gap, limit + 1))
    events = [
        ScheduledEvent(TARGET, first_start, first_start + event_len,
                       "answer_evidence"),
        ScheduledEvent(OTHER, second_start, second_start + event_len,
                       "control_sound"),
        ScheduledEvent(TARGET, third_start, third_start + event_len,
                       "control_sound"),
    ]
    schedule = Schedule(
        "card3", events, 0,
        {"first_sound_frame": query_frame,
         "first_sound_role": TARGET,
         "event_count": len(events)})
    _self_check_first_sound(schedule, query_frame)
    return schedule


def _self_check_first_sound(schedule: Schedule, query_frame: int) -> None:
    ordered = sorted(schedule.events, key=lambda event: event.start_sample)
    if ordered[0] is not schedule.anchor or schedule.anchor.role != TARGET:
        raise AudioProfileError("card3 first sound must belong to target")
    first_frame, _ = schedule.anchor.frame_span()
    if first_frame != query_frame:
        raise AudioProfileError(
            f"card3 first sound starts at frame {first_frame}, not {query_frame}")
    if len(ordered) < 3:
        raise AudioProfileError("card3 requires at least three events")
    _assert_no_overlap(schedule)


def _self_check_forward(schedule: Schedule, params) -> None:
    anchor = schedule.anchor
    if anchor is not schedule.events[-1]:
        raise AudioProfileError("card1F anchor must be the last event")
    if anchor.role != TARGET:
        raise AudioProfileError("card1F anchor must be the target actor")
    tail = (_clip_samples() - anchor.end_sample_exclusive) / SAMPLE_RATE
    declared = float(params["TAIL_SILENCE_FRACTION"]) * CLIP_SECONDS
    if tail + 1e-9 < declared:
        raise AudioProfileError(f"tail silence {tail:.3f}s < declared {declared:.3f}s")
    roles = {e.role for e in schedule.events}
    if roles != {TARGET, OTHER}:
        raise AudioProfileError("card1F needs both roles to have sounded")
    _assert_no_overlap(schedule)


def _self_check_backward(schedule: Schedule, params, window) -> None:
    anchor = schedule.anchor
    if anchor.role != TARGET or anchor is not schedule.events[-1]:
        raise AudioProfileError("card1B anchor must be the target's last event")
    for event in schedule.events:
        if event.role != TARGET:
            continue
        if event is anchor:
            continue
        if event.end_sample_exclusive > window[0] and \
                event.start_sample < window[1]:
            raise AudioProfileError(
                "the target sounds inside the query silence window; the "
                "earlier position would be audible")
    _assert_no_overlap(schedule)


def _self_check_first_call_bands(schedule: Schedule, params, target_bands,
                                 band_edges=None) -> None:
    bands = ([float(b) for b in band_edges] if band_edges
             else [float(b) for b in params["BANDS_CARD8"]])
    firsts: dict[str, float] = {}
    for event in sorted(schedule.events, key=lambda e: e.start_sample):
        firsts.setdefault(event.role, event.start_seconds)
    if len(firsts) != 2:
        raise AudioProfileError("card8 needs a first call from each role")
    ordered = sorted(firsts.values())
    if ordered[1] - ordered[0] <= float(params["T_HALF"]):
        raise AudioProfileError("the two first calls are closer than T_HALF")
    got = tuple(_band_index(value, bands) for value in ordered)
    if got != tuple(target_bands):
        raise AudioProfileError(
            f"first calls landed in bands {got}, not the assigned {target_bands}")
    _assert_no_overlap(schedule)


def _self_check_exactly_one(schedule: Schedule, query_frame: int) -> None:
    calling = [e for e in schedule.events
               if e.frame_span()[0] <= query_frame < e.frame_span()[1]]
    if len(calling) != 1:
        raise AudioProfileError(
            f"{len(calling)} actors sound at the query frame; card7 needs one")
    _assert_no_overlap(schedule)


def _assert_no_overlap(schedule: Schedule) -> None:
    ordered = sorted(schedule.events, key=lambda e: e.start_sample)
    for earlier, later in zip(ordered, ordered[1:]):
        if later.start_sample < earlier.end_sample_exclusive:
            raise AudioProfileError(
                "sequential_sources programs cannot carry overlapping events")


def _band_index(value: float, bands: list[float]) -> int | None:
    for index in range(len(bands) - 1):
        if bands[index] <= value < bands[index + 1]:
            return index
    if math.isclose(value, bands[-1]):
        return len(bands) - 2
    return None

def card8_feasible_interval(params, *, min_events: int = 3) -> tuple[float, float]:
    """⑧ 的首叫可行域,由片长与事件约束**推**出来,不读任何批次的答案分布。

    推导(全部是声明量,与房间无关):
      末事件必须在片内放完      t_last <= clip - event
      每个事件之后要留间隔      t_k    <= t_{k+1} - (event + gap)
      两只的首叫要可分辨        t2     >  t1 + T_HALF
      每集至少 min_events 声
    于是 ⑧ 的**目标首叫**(可能是第一声也可能是第二声)落在
      [first_min, clip - event - (min_events - 2) * (event + gap)]

    关键:这里**不含**任何片尾静默 —— ①F 需要片尾静默,⑧ 不需要。
    run01 把 ⑧ 的带按 ①F 的 1.5 秒片尾静默切出来,首叫因此被压进前
    2.6 秒、后段结构性为空;那条边界不能带进新方案。
    """
    clip = float(params.get("CLIP_SECONDS", CLIP_SECONDS))
    event = float(params.get("EVENT_SECONDS", EVENT_SECONDS))
    gap = float(params["GAP_MIN_S"])
    first_min = float(params["FIRST_MIN_S"])
    last_start = clip - event
    latest_second_call = last_start - max(0, min_events - 2) * (event + gap)
    if latest_second_call <= first_min:
        raise AudioProfileError(
            "no feasible first-call interval: the clip cannot hold "
            f"{min_events} spaced events")
    return (first_min, latest_second_call)


def card8_band_edges(params, *, n_bands: int = 4,
                     min_events: int = 3) -> list[float]:
    """可行域内**等宽半开**的 MCQ 时间带;边界在生成数据前锁定。"""
    lo, hi = card8_feasible_interval(params, min_events=min_events)
    width = (hi - lo) / n_bands
    t_half = float(params["T_HALF"])
    if width <= 0:
        raise AudioProfileError("degenerate band width")
    edges = [round(lo + width * i, 6) for i in range(n_bands + 1)]
    # 相邻带对必须仍能满足"两只首叫相隔超过 T_HALF",否则带对退化成
    # 确定性模板(听到第一声就知道第二只在哪带)。
    if hi - lo <= t_half:
        raise AudioProfileError(
            f"feasible interval {hi - lo:.2f}s cannot host two first calls "
            f"separated by more than T_HALF={t_half}s")
    return edges

