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
    sample_rate_hz: int
    ticks_per_sample: int
    ticks_per_frame: int
    frame_count: int
    sound_asset_id: str | None = None
    source_start_sample: int | None = None
    source_end_sample_exclusive: int | None = None

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate_hz

    @property
    def duration_samples(self) -> int:
        return self.end_sample_exclusive - self.start_sample

    def frame_span(self) -> tuple[int, int]:
        t0 = self.start_sample * self.ticks_per_sample
        t1 = self.end_sample_exclusive * self.ticks_per_sample
        return (t0 // self.ticks_per_frame,
                min(self.frame_count, -(-t1 // self.ticks_per_frame)))


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

    def program_events(self, role_to_slot: dict[str, str]) -> list[dict]:
        """Bind roles and keep per-event duration / clip identity.

        Dry-canvas schedules have no clip identity on the events; call
        ``bind()`` and put the canvas window on the program request.
        """
        missing = {e.role for e in self.events} - set(role_to_slot)
        if missing:
            raise AudioProfileError(f"unbound roles: {sorted(missing)}")
        rows = []
        for event in self.events:
            if event.sound_asset_id is None:
                raise AudioProfileError(
                    "program_events needs a sound_asset_id and source window "
                    "on every event; dry_canvas_window schedules must use bind()")
            if (event.source_start_sample is None
                    or event.source_end_sample_exclusive is None):
                raise AudioProfileError(
                    f"{event.sound_asset_id} is missing source window")
            rows.append({
                "slot": role_to_slot[event.role],
                "start_sample": event.start_sample,
                "duration_samples": event.duration_samples,
                "sound_asset_id": event.sound_asset_id,
                "source_start_sample": int(event.source_start_sample),
                "source_end_sample_exclusive": int(
                    event.source_end_sample_exclusive
                ),
            })
        return rows


def _require(params, key):
    if key not in params:
        raise AudioProfileError(f"params missing {key}")
    return params[key]


def _positive_int(params, key) -> int:
    value = int(_require(params, key))
    if value <= 0:
        raise AudioProfileError(f"{key} must be positive")
    return value


def _sample_rate(params) -> int:
    return _positive_int(params, "SAMPLE_RATE_HZ")


def _frame_count(params) -> int:
    return _positive_int(params, "FRAME_COUNT")


def _ticks_per_sample(params) -> int:
    return _positive_int(params, "TICKS_PER_SAMPLE")


def _ticks_per_frame(params) -> int:
    return _positive_int(params, "TICKS_PER_FRAME")


def _event_len(params, clip=None) -> int:
    if clip is not None:
        duration = int(clip.duration_samples)
        if duration <= 0:
            raise AudioProfileError("clip duration_samples must be positive")
        return duration
    seconds = float(_require(params, "EVENT_SECONDS"))
    if seconds <= 0:
        raise AudioProfileError("EVENT_SECONDS must be positive")
    return int(round(seconds * _sample_rate(params)))


def _draw_clip(clip_source):
    if clip_source is None:
        return None
    return clip_source.next()


def _stamp(role: str, start: int, purpose: str, params, clip=None) -> ScheduledEvent:
    duration = _event_len(params, clip)
    event = ScheduledEvent(
        role, start, start + duration, purpose,
        sample_rate_hz=_sample_rate(params),
        ticks_per_sample=_ticks_per_sample(params),
        ticks_per_frame=_ticks_per_frame(params),
        frame_count=_frame_count(params),
    )
    if clip is not None:
        if not getattr(clip, "sound_asset_id", None):
            raise AudioProfileError("clip is missing sound_asset_id")
        start_src = getattr(clip, "source_start_sample", None)
        end_src = getattr(clip, "source_end_sample_exclusive", None)
        if start_src is None or end_src is None:
            raise AudioProfileError(
                f"clip {clip.sound_asset_id} is missing source window")
        event.sound_asset_id = clip.sound_asset_id
        event.source_start_sample = int(start_src)
        event.source_end_sample_exclusive = int(end_src)
    return event


def _clip_samples(params) -> int:
    seconds = float(_require(params, "CLIP_SECONDS"))
    if seconds <= 0:
        raise AudioProfileError("CLIP_SECONDS must be positive")
    return int(round(seconds * _sample_rate(params)))


def _fraction_to_sample(params, fraction: float) -> int:
    return int(round(
        fraction * float(_require(params, "CLIP_SECONDS")) * _sample_rate(params)))


def _frame_to_sample(params, frame: int) -> int:
    return int(round(frame / _frame_count(params) * _clip_samples(params)))


def _place_sequential(rng, durs: list[int], gap: int, lo: int, hi_end: int):
    """Place events of known lengths so each ends before the next starts.

    ``hi_end`` is the exclusive sample limit for the last event's end.
    """
    n = len(durs)
    starts: list[int] = []
    cursor = lo
    for index in range(n):
        later = 0
        for later_index in range(index + 1, n):
            later += gap + durs[later_index]
        hi = hi_end - later - durs[index]
        if cursor > hi:
            return None
        starts.append(int(rng.integers(cursor, hi + 1)))
        cursor = starts[-1] + durs[index] + gap
    return starts


CARD8_CERTIFICATION_POLICY = "strict_full_credit_only"
CARD8_WIDE_TOLERANCE_ROLE = "diagnostic_only"
CARD8_MIN_SEPARATION_RULE = "first_call_onset_gap > max(T_HALF, 2 * T_FULL)"


def card8_scoring_params(params, *, historical_record: bool = False) -> dict:
    """⑧ 首叫链的显式评分参数;缺 T_FULL 直接 fail-closed。

    正式 Open 按 strict T_FULL 判满分(宽带 T_HALF 只作诊断),所以生成端的
    最小首叫间隔必须同时盖住两件事:两只的首叫在 T_HALF 下可分辨,且"报两声
    中点"这类 A-only 策略在 strict 评分下拿不到满分 —— 后者要求间隔严格大于
    2 * T_FULL。T_FULL 的终值等人类校准:这里只记录参数文件给出的值与它的
    状态字段,不替它写 final。

    生成路径 (historical_record=False) 必填 T_FULL_status。审计/复核历史
    批次时传 historical_record=True:允许缺这个键,输出写成
    unspecified_in_historical_record,一眼能看出是老数据。
    """
    missing = [key for key in ("T_FULL", "T_HALF") if key not in params]
    if missing:
        raise AudioProfileError(
            f"card8 requires explicit scoring params {missing}; the minimum "
            "first-call separation cannot be derived without them")
    if not historical_record and "T_FULL_status" not in params:
        raise AudioProfileError(
            "card8 requires explicit scoring params ['T_FULL_status']; the "
            "minimum first-call separation cannot be derived without them")
    t_full = float(params["T_FULL"])
    t_half = float(params["T_HALF"])
    if historical_record and "T_FULL_status" not in params:
        status = "unspecified_in_historical_record"
    else:
        status = str(params["T_FULL_status"])
        if not status:
            raise AudioProfileError("T_FULL_status is empty")
    if not (math.isfinite(t_full) and math.isfinite(t_half)):
        raise AudioProfileError("T_FULL and T_HALF must be finite seconds")
    if t_full <= 0.0 or t_half <= 0.0:
        raise AudioProfileError("T_FULL and T_HALF must be positive seconds")
    if t_half < t_full:
        raise AudioProfileError(
            f"T_HALF={t_half} must not be narrower than T_FULL={t_full}")
    min_sep = max(t_half, 2.0 * t_full)
    return {
        "T_FULL": t_full,
        "T_HALF": t_half,
        "T_FULL_status": status,
        "min_first_call_separation_s": min_sep,
        "min_first_call_separation_samples": int(round(min_sep * _sample_rate(params))),
        "min_first_call_separation_rule": CARD8_MIN_SEPARATION_RULE,
        "certification_policy": CARD8_CERTIFICATION_POLICY,
        "wide_tolerance_role": CARD8_WIDE_TOLERANCE_ROLE,
    }


def schedule_forward_anchor(rng, *, params, anchor_frame: int,
                            clip_source=None) -> Schedule:
    """①F 正向错时:身份锚在前,查询在片尾,锚后到片尾必须静。

    锚是**最后一声**;片尾静默长度按片长比例声明(tail_silence_fraction)。
    前面还要有至少两声,让"最后一声"的事件选择不是唯一事件检测。
    但目标全片只在锚发声一次:否则 A-only 可按目标此前的 DoA 轨迹猜
    查询时刻方位,绕过声后视觉追踪。
    """
    tail_fraction = float(params["TAIL_SILENCE_FRACTION"])
    gap = int(float(params["GAP_MIN_S"]) * _sample_rate(params))
    first_min = int(float(params["FIRST_MIN_S"]) * _sample_rate(params))
    n_before = int(rng.integers(2, 4))          # 锚之前 2 或 3 声
    pre_clips = [_draw_clip(clip_source) for _ in range(3)]
    anchor_clip = _draw_clip(clip_source)
    pre_durs = [_event_len(params, clip) for clip in pre_clips]
    anchor_dur = _event_len(params, anchor_clip)
    anchor_start = int(round(anchor_frame / _frame_count(params) * _clip_samples(params)))
    tail = _clip_samples(params) - (anchor_start + anchor_dur)
    if tail < _fraction_to_sample(params, tail_fraction):
        raise AudioProfileError(
            f"anchor at frame {anchor_frame} leaves {tail / _sample_rate(params):.2f}s "
            f"of tail silence, below the declared "
            f"{tail_fraction * float(params['CLIP_SECONDS']):.2f}s")

    def _fits(count: int) -> bool:
        return first_min + sum(pre_durs[:count]) + count * gap <= anchor_start

    if n_before == 3 and not _fits(3):
        n_before = 2
    if not _fits(n_before):
        raise AudioProfileError(
            "no room for the pre-anchor calls before the anchor instant")
    pre_clips = pre_clips[:n_before]
    pre_durs = pre_durs[:n_before]
    starts = _place_sequential(
        rng, pre_durs, gap, first_min, anchor_start - gap)
    if starts is None:
        raise AudioProfileError(
            "no room for the pre-anchor calls before the anchor instant")
    events = [
        _stamp(OTHER, start, "control_sound", params, clip)
        for start, clip in zip(starts, pre_clips)
    ]
    events.append(_stamp(TARGET, anchor_start, "identity_anchor", params, anchor_clip))
    schedule = Schedule("card1F", events, len(events) - 1,
                        {"tail_silence_seconds": tail / _sample_rate(params),
                         "anchor_relation": "anchor_before_query",
                         "query_frame": _frame_count(params) - 1})
    _self_check_forward(schedule, params)
    return schedule


def schedule_backward_anchor(rng, *, params, anchor_frame: int,
                             query_frame: int, clip_source=None) -> Schedule:
    """①B 反向错时:视觉查询在前,身份锚在末段。

    要求(与正向相反):**查询时刻附近目标必须静**,否则听声即可定位,
    题退化成即时 DoA;锚在查询之后,由末段发声确定身份。
    """
    if query_frame >= anchor_frame:
        raise AudioProfileError(
            "backward cross-time needs the visual query before the audio anchor")
    silence_fraction = float(params["QUERY_SILENCE_FRACTION"])
    guard = _fraction_to_sample(params, silence_fraction)
    gap = int(float(params["GAP_MIN_S"]) * _sample_rate(params))
    first_min = int(float(params["FIRST_MIN_S"]) * _sample_rate(params))
    query_sample = int(round(query_frame / _frame_count(params) * _clip_samples(params)))
    anchor_start = int(round(anchor_frame / _frame_count(params) * _clip_samples(params)))
    n_before = int(rng.integers(1, 3))
    pre_clips = [_draw_clip(clip_source) for _ in range(n_before)]
    anchor_clip = _draw_clip(clip_source)
    pre_durs = [_event_len(params, clip) for clip in pre_clips]
    anchor_dur = _event_len(params, anchor_clip)
    if anchor_start + anchor_dur > _clip_samples(params):
        raise AudioProfileError("the anchor event runs past the clip")
    window = (query_sample - guard, query_sample + guard)
    # 锚之前安排 1-2 声对照;它们既不能落进查询静默窗,也不能压到锚上
    starts: list[int] = []
    kept_clips: list = []
    cursor = first_min
    for index, (clip, duration) in enumerate(zip(pre_clips, pre_durs)):
        remaining = n_before - 1 - index
        later = sum(pre_durs[index + 1:]) + remaining * gap
        hi = anchor_start - gap - later - duration
        if cursor > hi:
            break
        for _attempt in range(40):
            candidate = int(rng.integers(cursor, hi + 1))
            if candidate + duration <= window[0] or candidate >= window[1]:
                starts.append(candidate)
                kept_clips.append(clip)
                cursor = candidate + duration + gap
                break
        else:
            continue
    events = [
        _stamp(OTHER, start, "control_sound", params, clip)
        for start, clip in zip(starts, kept_clips)
    ]
    events.append(_stamp(TARGET, anchor_start, "identity_anchor", params, anchor_clip))
    schedule = Schedule("card1B", events, len(events) - 1,
                        {"anchor_relation": "anchor_after_query",
                         "query_frame": query_frame,
                         "query_silence_window_samples": list(window),
                         "query_silence_seconds": guard / _sample_rate(params)})
    _self_check_backward(schedule, params, window)
    return schedule


def schedule_first_call_bands(rng, *, params, target_bands: tuple[int, int],
                              band_edges: list[float] | None = None,
                              first_caller_role: str = TARGET,
                              clip_source=None) -> Schedule:
    """⑧ 首叫时间带:两个角色的首叫落进**预先声明**的不同带。

    不继承 card1 的片尾静默 —— 那正是 run01 把首叫压进前 2.6 秒、后两带
    结构性为空的原因。带边由题型配置声明,窗内时刻随机。

    两只的首叫间隔必须**严格大于** max(T_HALF, 2 * T_FULL)(样本域整数比较):
    正式 Open 按 strict T_FULL 判分,只查 T_HALF 会让 1.1 s 间隔的题在
    T_FULL=0.6 下被"报中点"拿满分。缺 T_FULL 直接拒绝调度。
    """
    bands = ([float(b) for b in band_edges] if band_edges
             else [float(b) for b in params["BANDS_CARD8"]])
    b1, b2 = target_bands
    if not 0 <= b1 < b2 <= len(bands) - 2:
        raise AudioProfileError(
            f"unreachable band pair {target_bands}: events alternate in time "
            "so the first caller's band index must be the smaller one")
    clip1 = _draw_clip(clip_source)
    clip2 = _draw_clip(clip_source)
    clip3 = _draw_clip(clip_source)
    dur1, dur2, dur3 = _event_len(params, clip1), _event_len(params, clip2), _event_len(params, clip3)
    gap = int(float(params["GAP_MIN_S"]) * _sample_rate(params))
    scoring = card8_scoring_params(params)
    min_first_gap = scoring["min_first_call_separation_samples"]
    second_role = OTHER if first_caller_role == TARGET else TARGET
    lo1, hi1 = int(bands[b1] * _sample_rate(params)), int(bands[b1 + 1] * _sample_rate(params))
    lo2, hi2 = int(bands[b2] * _sample_rate(params)), int(bands[b2 + 1] * _sample_rate(params))
    limit1 = _clip_samples(params) - dur1
    limit2 = _clip_samples(params) - dur2
    for _attempt in range(400):
        t1 = int(rng.integers(lo1, min(hi1, limit1)))
        t2_lo = max(t1 + dur1 + gap, lo2, t1 + min_first_gap + 1)
        t2_hi = min(hi2, limit2)
        if t2_lo >= t2_hi:
            continue
        t2 = int(rng.integers(t2_lo, t2_hi))
        events = [
            _stamp(first_caller_role, t1, "answer_evidence", params, clip1),
            _stamp(second_role, t2, "answer_evidence", params, clip2),
        ]
        # 第三声让"每集≥3 声"成立,且不改变任一方的首叫
        third_lo = t2 + dur2 + gap
        if third_lo + dur3 <= _clip_samples(params):
            t3 = int(rng.integers(third_lo, _clip_samples(params) - dur3 + 1))
            events.append(_stamp(first_caller_role, t3, "control_sound", params, clip3))
        schedule = Schedule("card8", events, len(events) - 1,
                            {"target_bands": [b1, b2],
                             "band_edges_seconds": bands,
                             "first_caller_role": first_caller_role,
                             "first_call_scoring": dict(scoring)})
        _self_check_first_call_bands(schedule, params, target_bands,
                                     band_edges=bands)
        return schedule
    raise AudioProfileError(
        f"no onset layout lands the first calls in bands {target_bands}")


def schedule_exactly_one_calling(rng, *, params, query_frame: int,
                                 clip_source=None) -> Schedule:
    """⑦ 指定时刻恰好一只在叫:围绕查询帧安排唯一发声者。"""
    target_clip = _draw_clip(clip_source)
    other_clip = _draw_clip(clip_source)
    event_len = _event_len(params, target_clip)
    other_len = _event_len(params, other_clip)
    gap = int(float(params["GAP_MIN_S"]) * _sample_rate(params))
    query_sample = int(round(query_frame / _frame_count(params) * _clip_samples(params)))
    start = max(0, min(query_sample - event_len // 2,
                       _clip_samples(params) - event_len))
    events = [_stamp(TARGET, start, "answer_evidence", params, target_clip)]
    # 另一角色的声音必须完全避开查询帧所在事件窗
    limit = _clip_samples(params) - other_len
    for _attempt in range(60):
        other = int(rng.integers(0, limit + 1))
        if other + other_len + gap <= start or other >= start + event_len + gap:
            events.append(_stamp(OTHER, other, "control_sound", params, other_clip))
            break
    events.sort(key=lambda e: e.start_sample)
    schedule = Schedule("card7", events,
                        max(range(len(events)),
                            key=lambda i: events[i].start_sample),
                        {"query_frame": query_frame,
                         "exactly_one_calling_at_query": True})
    _self_check_exactly_one(schedule, query_frame)
    return schedule


def schedule_first_sound_at_frame(rng, *, params, query_frame: int,
                                  clip_source=None) -> Schedule:
    """Card3 control: the target makes the first sound at a declared frame."""
    clips = [_draw_clip(clip_source) for _ in range(3)]
    durs = [_event_len(params, clip) for clip in clips]
    gap = int(float(params["GAP_MIN_S"]) * _sample_rate(params))
    first_start = _frame_to_sample(params, query_frame)
    if first_start < 0 or first_start + durs[0] > _clip_samples(params):
        raise AudioProfileError(
            f"query frame {query_frame} cannot host the first event")
    rest = _place_sequential(
        rng, durs[1:], gap, first_start + durs[0] + gap, _clip_samples(params))
    if rest is None:
        raise AudioProfileError(
            "no room for three separated events after the first sound")
    events = [
        _stamp(TARGET, first_start, "answer_evidence", params, clips[0]),
        _stamp(OTHER, rest[0], "control_sound", params, clips[1]),
        _stamp(TARGET, rest[1], "control_sound", params, clips[2]),
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




def schedule_event_count(rng, *, params, event_count: int,
                         clip_source=None) -> Schedule:
    """Audio-count control with randomized, separated event times."""
    if event_count < 2:
        raise AudioProfileError("event-count profile needs at least two events")
    clips = [_draw_clip(clip_source) for _ in range(event_count)]
    durs = [_event_len(params, clip) for clip in clips]
    gap = int(float(params["GAP_MIN_S"]) * _sample_rate(params))
    first_min = int(float(params["FIRST_MIN_S"]) * _sample_rate(params))
    starts = _place_sequential(rng, durs, gap, first_min, _clip_samples(params))
    if starts is None:
        raise AudioProfileError(
            f"{event_count} separated events do not fit in the clip")
    events = [
        _stamp(TARGET if index % 2 == 0 else OTHER,
               start, "answer_evidence", params, clip)
        for index, (start, clip) in enumerate(zip(starts, clips))
    ]
    schedule = Schedule(
        "card15b", events, 0,
        {"event_count": event_count, "count_is_gatea_invariant": True})
    _self_check_event_count(schedule, event_count)
    return schedule


def _self_check_event_count(schedule: Schedule, event_count: int) -> None:
    if len(schedule.events) != event_count:
        raise AudioProfileError(


            f"event count {len(schedule.events)} != declared {event_count}")
    if {event.role for event in schedule.events} != {TARGET, OTHER}:
        raise AudioProfileError("event-count profile must use both roles")
    _assert_no_overlap(schedule)


def schedule_second_sound_at_frame(rng, *, params,
                                   query_frame: int,
                                   clip_source=None) -> Schedule:
    """Card6 family: the target owns the second event at a declared frame."""
    clips = [_draw_clip(clip_source) for _ in range(3)]
    durs = [_event_len(params, clip) for clip in clips]
    gap = int(float(params["GAP_MIN_S"]) * _sample_rate(params))
    second_start = _frame_to_sample(params, query_frame)
    first_latest = second_start - durs[0] - gap
    if first_latest < 0:
        raise AudioProfileError(
            f"query frame {query_frame} leaves no room for a first event")
    first_start = int(rng.integers(0, first_latest + 1))
    third_min = second_start + durs[1] + gap
    if third_min + durs[2] > _clip_samples(params):
        raise AudioProfileError(
            f"query frame {query_frame} leaves no room for a third event")
    third_start = int(rng.integers(third_min, _clip_samples(params) - durs[2] + 1))
    events = [
        _stamp(OTHER, first_start, "control_sound", params, clips[0]),
        _stamp(TARGET, second_start, "answer_evidence", params, clips[1]),
        _stamp(OTHER, third_start, "control_sound", params, clips[2]),
    ]
    schedule = Schedule(
        "card6", events, 1,
        {"second_sound_frame": query_frame,
         "second_sound_role": TARGET,
         "event_count": len(events)})
    _self_check_second_sound(schedule, query_frame)
    return schedule


def _self_check_second_sound(schedule: Schedule, query_frame: int) -> None:
    ordered = sorted(schedule.events, key=lambda event: event.start_sample)
    if ordered[1] is not schedule.anchor or schedule.anchor.role != TARGET:
        raise AudioProfileError("card6 second sound must belong to target")
    second_frame, _ = schedule.anchor.frame_span()
    if second_frame != query_frame:
        raise AudioProfileError(
            f"card6 second sound starts at frame {second_frame}, not {query_frame}")
    if len(ordered) != 3:
        raise AudioProfileError("card6 schedule must contain three events")
    _assert_no_overlap(schedule)


def _self_check_forward(schedule: Schedule, params) -> None:
    anchor = schedule.anchor
    if anchor is not schedule.events[-1]:
        raise AudioProfileError("card1F anchor must be the last event")
    if anchor.role != TARGET:
        raise AudioProfileError("card1F anchor must be the target actor")
    tail = (_clip_samples(params) - anchor.end_sample_exclusive) / _sample_rate(params)
    declared = float(params["TAIL_SILENCE_FRACTION"]) * float(params["CLIP_SECONDS"])
    if tail + 1e-9 < declared:
        raise AudioProfileError(f"tail silence {tail:.3f}s < declared {declared:.3f}s")
    target_events = [event for event in schedule.events if event.role == TARGET]
    if target_events != [anchor]:
        raise AudioProfileError(
            "card1F target must sound exactly once, at the identity anchor")
    if not any(event.role == OTHER for event in schedule.events):
        raise AudioProfileError("card1F needs non-target control sounds")
    _assert_no_overlap(schedule)


def _self_check_backward(schedule: Schedule, params, window) -> None:
    anchor = schedule.anchor
    if anchor.role != TARGET or anchor is not schedule.events[-1]:
        raise AudioProfileError("card1B anchor must be the target's last event")
    target_events = [event for event in schedule.events if event.role == TARGET]
    if target_events != [anchor]:
        raise AudioProfileError(
            "card1B target must sound exactly once, at the identity anchor; "
            "earlier target audio would expose another DoA observation")
    _assert_no_overlap(schedule)


def _self_check_first_call_bands(schedule: Schedule, params, target_bands,
                                 band_edges=None) -> None:
    bands = ([float(b) for b in band_edges] if band_edges
             else [float(b) for b in params["BANDS_CARD8"]])
    scoring = card8_scoring_params(params)
    firsts: dict[str, float] = {}
    first_samples: dict[str, int] = {}
    for event in sorted(schedule.events, key=lambda e: e.start_sample):
        firsts.setdefault(event.role, event.start_seconds)
        first_samples.setdefault(event.role, event.start_sample)
    if len(firsts) != 2:
        raise AudioProfileError("card8 needs a first call from each role")
    ordered_samples = sorted(first_samples.values())
    separation = ordered_samples[1] - ordered_samples[0]
    if separation <= scoring["min_first_call_separation_samples"]:
        raise AudioProfileError(
            f"the two first calls are {separation / _sample_rate(params):.4f}s apart; "
            "card8 requires strictly more than max(T_HALF, 2*T_FULL)="
            f"{scoring['min_first_call_separation_s']:.4f}s")
    ordered = sorted(firsts.values())
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
      两只的首叫要可分辨        t2     >  t1 + max(T_HALF, 2 * T_FULL)
      每集至少 min_events 声
    于是 ⑧ 的**目标首叫**(可能是第一声也可能是第二声)落在
      [first_min, clip - event - (min_events - 2) * (event + gap)]

    关键:这里**不含**任何片尾静默 —— ①F 需要片尾静默,⑧ 不需要。
    run01 把 ⑧ 的带按 ①F 的 1.5 秒片尾静默切出来,首叫因此被压进前
    2.6 秒、后段结构性为空;那条边界不能带进新方案。
    """
    clip = float(_require(params, "CLIP_SECONDS"))
    event = float(_require(params, "EVENT_SECONDS"))
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
    # 缺 T_FULL 时这里就失败:带边是可行域的一部分,而可行域取决于两只
    # 首叫必须相隔多远。
    scoring = card8_scoring_params(params)
    min_sep = scoring["min_first_call_separation_s"]
    if width <= 0:
        raise AudioProfileError("degenerate band width")
    edges = [round(lo + width * i, 6) for i in range(n_bands + 1)]
    # 可行域必须仍能满足"两只首叫相隔严格超过 max(T_HALF, 2*T_FULL)",
    # 否则带对退化成确定性模板(听到第一声就知道第二只在哪带)。
    if hi - lo <= min_sep:
        raise AudioProfileError(
            f"feasible interval {hi - lo:.2f}s cannot host two first calls "
            f"separated by more than max(T_HALF, 2*T_FULL)={min_sep}s")
    return edges
