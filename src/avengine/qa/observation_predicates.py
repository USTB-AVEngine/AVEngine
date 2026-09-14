"""出题用的观测判据，从 unified_catalog 的出题函数里拆出来的纯函数。

拆出来的理由不是好看。这些判据（动没动、离得更近还是更远、被挡住之后有没有重新露出来）原本长在
出题函数里，谁想在出题之外再问一次同样的问题——比如问「只看画面能不能唯一确定这个答案」——就只能
照着再写一遍，而照着再写一遍必然会慢慢走样，两边都绿、结论却不一样。所以这里只放纯函数：输入是
已经取好的读数（一串位置、一串距离、一串可见状态），输出是判定和它的量度。它不认识 facts 的结构、
不抛出题流程的异常、也不导入 avengine 的任何东西，出题函数负责把读数取出来递进来。

行为必须跟拆之前一个字节不差：``tools/dataset/replay_question_catalog.py`` 拿现成题库的
checkpoint 回放对拍，同一份 facts 必须出一模一样的题和答案。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

VISIBLE_STATES = frozenset({"visible_clear", "visible_occluded"})


def visibility_series(
    frames: Any, *, require_mapping: bool = False
) -> list[Any]:
    """把一个按帧号索引的字典摊成一串，按帧号的**数值**排序。

    这件事值得有个共用函数。帧号是 JSON 对象的键，也就是字符串，所以 ``sorted(frames)``
    给的是字典序：0, 1, 10, 100, 101, …, 2, 20, …。凡是靠“相邻两帧”判转场的地方，
    字典序会把不相邻的两帧凑成一对（比如第 1 帧跟第 10 帧），也会让真正相邻的两帧
    永远不相邻（第 9 帧跟第 10 帧）。

    2026-09-14 实测：QA-11 的 144 道题里有 14 道因此判反，另有 503 个报出来的转场帧
    在数值序下根本不存在。按键直接取值的路径（QA-08、QA-24 那种 ``frames[str(k)]``）
    不受影响，因为它们压根不遍历。

    修的是帧键排序，不是放宽判据：什么叫转场、什么叫遗挡，一个字没改。
    """

    if not isinstance(frames, Mapping):
        return []

    def order(key: Any) -> int:
        try:
            return int(key)
        except (TypeError, ValueError):
            return -1

    keys = sorted(frames, key=order)
    if require_mapping:
        return [frames[key] for key in keys if isinstance(frames[key], Mapping)]
    return [frames[key] for key in keys]


def _is_point(value: Any) -> bool:
    # 判定跟拆出来之前逐字相同，包括这里不把 bool 单独排除——改了就不是同一个判据了。
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
        and all(isinstance(axis, (int, float)) and math.isfinite(axis) for axis in value)
    )


# --------------------------------------------------------------------------
# 运动
# --------------------------------------------------------------------------


def noticeable_motion(
    points: Sequence[Sequence[float]],
    frame_rate_hz: float,
    policy: Mapping[str, Any],
    *,
    window_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """一段位置读数算不算「明显动过」。

    三个门限都来自 policy，这里一个都不自己定：走过的总路程要够长、其中速度超过门限的时间要够久，
    才算动；总路程小到 ``max_still_travel_m`` 以内才算没动；夹在中间的两样都不算，返回 None 和原因，
    因为那种情况说不清楚，不该硬判一个。
    """

    if any(not _is_point(point) for point in points):
        return {"moving": None, "reason": "invalid_motion_position_readback"}
    rate = float(frame_rate_hz)
    distances = [math.dist(a, b) for a, b in zip(points, points[1:])]
    travel = sum(distances)
    moving_s = sum(d * rate > policy["speed_threshold_mps"] for d in distances) / rate
    value = (
        True
        if travel >= policy["min_travel_m"] and moving_s >= policy["min_moving_duration_s"]
        else False
        if travel <= policy["max_still_travel_m"]
        else None
    )
    measurement = {
        "window_frames": list(window_frames) if window_frames is not None else None,
        "travel_m": travel,
        "moving_duration_s": moving_s,
        "position_source": "root_positions_m",
        "criteria": dict(policy),
    }
    return {
        "moving": value,
        "reason": None if value is not None else "motion_between_noticeability_thresholds",
        "measurement": measurement,
    }


def motion_reachable_verdicts(
    tracks: Mapping[str, Sequence[Sequence[float]]],
    frame_rate_hz: float,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """整段里「动过」和「没动」这两个判定各自够不够得着，用来算只看画面剩几个候选。

    只看画面的人知道每个人每一帧在哪，却不知道题问的是哪一段窗口，所以他能排除的只有「整段任何人
    任何窗口都做不出来」的那个判定。这两件事都不用穷举窗口就能定死：

    * 路程和运动时长都随窗口变长单调不减，所以「动过」够得着，当且仅当某个人整段那一个窗口就够；
    * 「没动」只要求总路程小于门限，路程随窗口变短单调不增，所以它够得着，当且仅当某个人某两个
      相邻帧之间的位移就已经小于门限。

    这是精确判定，不是抽样，也没有新门限。
    """

    moving_reachable = False
    still_reachable = False
    for points in tracks.values():
        if len(points) < 2 or any(not _is_point(point) for point in points):
            continue
        whole = noticeable_motion(points, frame_rate_hz, policy)
        if whole["moving"] is True:
            moving_reachable = True
        steps = [math.dist(a, b) for a, b in zip(points, points[1:])]
        if any(step <= policy["max_still_travel_m"] for step in steps):
            still_reachable = True
    return {
        "moving_reachable": moving_reachable,
        "still_reachable": still_reachable,
        "basis": "路程与运动时长对窗口单调，所以两个判定的可达性不用穷举窗口就能定死",
    }


def moved_at_any_frame(flags: Sequence[Any]) -> bool:
    """一段逐帧运动读数里有没有任何一帧在动。"""

    return any(bool(flag) for flag in flags)


# --------------------------------------------------------------------------
# 距离
# --------------------------------------------------------------------------


def distance_trend(
    series: Sequence[float],
    *,
    min_net_change_m: float,
    reversal_tolerance_m: float,
    reversal_fraction: float,
) -> dict[str, Any]:
    """一串到听者的距离是在靠近还是远离。

    首尾之差一个人扛不住这个答案：先走近再走远和一路走远的首尾差可以一模一样。所以净变化要过门限，
    而且逆着净方向的最大回撤要同时小于绝对容差和净变化的那个比例。两个读数都返回，调用方能看见
    首尾差，也能看见真正定判的那个回撤。
    """

    net = series[-1] - series[0]
    direction = "nearer" if net < 0.0 else "farther"
    extreme = series[0]
    counter = 0.0
    for value in series:
        if net < 0.0:
            extreme = min(extreme, value)
            counter = max(counter, value - extreme)
        else:
            extreme = max(extreme, value)
            counter = max(counter, extreme - value)
    allowed = min(reversal_tolerance_m, reversal_fraction * abs(net)) if abs(net) > 0.0 else reversal_tolerance_m
    result: dict[str, Any] = {
        "distance_series_m": list(series),
        "distance_start_m": series[0],
        "distance_end_m": series[-1],
        "endpoint_delta_m": net,
        "net_direction": direction,
        "distance_span_m": max(series) - min(series),
        "total_variation_m": sum(abs(b - a) for a, b in zip(series, series[1:])),
        "max_counter_trend_m": counter,
        "allowed_counter_trend_m": allowed,
        "monotone_within_tolerance": counter <= allowed,
        "endpoint_delta_is_not_sufficient": True,
    }
    if abs(net) < min_net_change_m:
        result["verdict"] = None
        result["reason"] = "distance_net_change_below_margin"
        result["detail"] = "the distance changes by less than the configured margin over the window"
        return result
    if counter > allowed:
        result["verdict"] = None
        result["reason"] = "distance_trend_reverses"
        result["detail"] = (
            "the path moves back against its net direction by more than the "
            "configured reversal allowance"
        )
        return result
    result["verdict"] = direction
    result["reason"] = None
    return result


def distance_step_verdict(
    anchor_distance: float, query_distance: float, margin_m: float
) -> dict[str, Any]:
    """某一刻的距离跟锚点比是更近还是更远；差值不到门限就不判。"""

    delta = float(query_distance) - float(anchor_distance)
    if abs(delta) < margin_m:
        return {"trend": None, "delta": delta, "reason": "distance_change_below_margin"}
    return {"trend": "nearer" if delta < 0.0 else "farther", "delta": delta, "reason": None}


def distance_direction_reachable(
    series_by_actor: Mapping[str, Sequence[float]], *, min_net_change_m: float
) -> dict[str, Any]:
    """整段里「更近」和「更远」各自能不能在某个窗口上做出来。

    这里给的是**可达的必要条件**，不是充分条件：某个方向要能在某段窗口上成立，那一串距离至少得在
    某处朝那个方向走够门限那么多。反过来不一定成立（回撤可能把它顶掉），所以判定只用来**排除**：
    朝某个方向一步都没走够的，那个方向可以排除；走够了的，那个选项就留着，不敢说它一定能成立。
    这条方向是安全的——它只会多留选项，不会把某个模态说成比实际更行。
    """

    nearer = farther = False
    for series in series_by_actor.values():
        if len(series) < 2:
            continue
        lowest = highest = series[0]
        for value in series:
            if value - lowest >= min_net_change_m:
                farther = True
            if highest - value >= min_net_change_m:
                nearer = True
            lowest = min(lowest, value)
            highest = max(highest, value)
    return {
        "nearer_reachable": nearer,
        "farther_reachable": farther,
        "basis": "某个方向可达的必要条件是距离在某处朝该方向走够了门限；只用来排除，不用来断言唯一",
    }


# --------------------------------------------------------------------------
# 遮挡转场
# --------------------------------------------------------------------------


def full_occlusion_frames(ordered: Sequence[Mapping[str, Any]]) -> list[Any]:
    """一条可见状态轨迹里被完全遮住的那些帧。"""

    return [frame.get("frame_index") for frame in ordered if frame.get("state") == "fully_occluded"]


def partial_occlusion_frames(ordered: Sequence[Mapping[str, Any]]) -> list[Any]:
    """一条可见状态轨迹里被部分遮住的那些帧。"""

    return [frame.get("frame_index") for frame in ordered if frame.get("state") == "visible_occluded"]


def reappeared_frames(
    ordered: Sequence[Mapping[str, Any]], occluded_frames: Sequence[Any]
) -> list[Any]:
    """完全被遮住之后又重新看得见的那些帧。"""

    return [
        frame.get("frame_index")
        for frame in ordered
        if frame.get("state") in VISIBLE_STATES
        and any(int(previous) < int(frame.get("frame_index", 0)) for previous in occluded_frames)
    ]


def clear_after_partial_transitions(
    ordered: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """从部分遮挡直接变成清晰可见的那些相邻帧对，返回（前一帧记录, 当前帧记录）。

    返回的是帧记录本身而不是帧号，调用方怎么取帧号、怎么再按自己的偏好过滤都跟拆之前一样；
    判据本身只认相邻两帧的状态。
    """

    return [
        (previous, current)
        for previous, current in zip(ordered, ordered[1:])
        if previous.get("state") == "visible_occluded" and current.get("state") == "visible_clear"
    ]


def occlusion_recovery_reachable(
    tracks: Mapping[str, Sequence[Mapping[str, Any]]], *, kind: str
) -> dict[str, Any]:
    """整段里 yes / no 这两个答案各自够不够得着，用来算只看画面剩几个候选。

    ``kind`` 是 ``full``（QA-09：完全被遮住之后有没有重新露出来）或 ``partial``
    （QA-11：部分遮挡之后有没有变清晰）。只有满足前提的演员才算候选：QA-09 要有被完全遮住的帧，
    QA-11 要有部分遮挡的帧。这是精确判定，逐条轨迹跑一遍就完了。
    """

    if kind not in ("full", "partial"):
        raise ValueError("kind must be full or partial")
    yes = no = False
    candidates = 0
    for ordered in tracks.values():
        if kind == "full":
            occluded = full_occlusion_frames(ordered)
            if not occluded:
                continue
            verdict = bool(reappeared_frames(ordered, occluded))
        else:
            if not partial_occlusion_frames(ordered):
                continue
            verdict = bool(clear_after_partial_transitions(ordered))
        candidates += 1
        yes = yes or verdict
        no = no or not verdict
    return {
        "yes_reachable": yes,
        "no_reachable": no,
        "candidate_track_count": candidates,
        "basis": "逐条可见状态轨迹跑一遍转场判据，够得着的答案就是只看画面排除不掉的那些",
    }
