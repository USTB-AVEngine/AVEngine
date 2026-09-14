"""题库导出时算出来的两个私有分层字段：查询时刻的可见状态，和单模态候选数收据。

这两个字段都只写进 private，public 里一个字节都不加。它们是记录不是闸门：谁也不该因为
``unimodal_candidates`` 里某个数是 1 就把题删掉或者把政策改松。

**字段一 visibility_at_query**：评测的时候想按「被问的那个演员在查询时刻是什么可见状态」
分层报分，就得先把「查询时刻」这件事按题型定下来。取帧规则写在 ``QUERY_FRAME_RULES`` 这张
表里，每一种都说得出依据；取不到的记 ``state: unknown`` 加原因，不猜。可见状态本身来自
``facts["visibility"]``，也就是 ``capture/pixel_visibility_truth.json`` 的逐帧四态。

**字段二 unimodal_candidates**：竞品 ST-OmniQA 把必要性规则（|Ca|>1、|Cv|>1、|Cav|=1）
写死在生成规则里，我们反过来，在真实媒体上给每道题一张收据：只听能剩几个候选答案、只看能
剩几个。数法是把选项集合分别按两个视角裁一遍——

* **只听（audio）**：能听到的是各个独立发声事件的起止时刻、声音类别、以及相对听者的方位和
  距离。听不出哪个事件属于画面里的哪个个体，也听不出任何外观。
* **只看（video）**：能看到的是逐帧每个演员的可见状态和屏幕位置、世界坐标与距离、已登记并
  复核过的外观、以及运动。看不到事件表、声音类别，也看不到谁在发声。

一个选项只有被某个视角**严格排除**才不计入那一侧的候选数。推不出来的题型写 ``null`` 加
一句原因，不许估、不许为了填满改口径。``joint`` 按构造恒为 1（导出保留的每道题只有一个被
接受的金标），记下来是为了凑齐 |Cav|=1 那一条，它不是一次独立测量。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from avengine.qa.observation_predicates import (
    clear_after_partial_transitions,
    distance_direction_reachable,
    full_occlusion_frames,
    motion_reachable_verdicts,
    noticeable_motion,
    occlusion_recovery_reachable,
    partial_occlusion_frames,
    reappeared_frames,
    visibility_series,
)

PIXEL_STATES: tuple[str, ...] = (
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
    "out_of_view",
)
VISIBLE_STATES = frozenset({"visible_clear", "visible_occluded"})

#: 取帧规则表。每道题按它的题型在这里查一次，查不到的走 ``FALLBACK_FRAME_FIELDS``。
QUERY_FRAME_RULES: dict[str, str] = {
    "QA-01": "first_event_start_frame",
    "QA-02": "target_frame",
    "QA-03": "first_event_start_frame",
    "QA-04": "query_frame",
    "QA-05": "whole_clip",
    "QA-06": "event_window",
    "QA-07": "entry_frame",
    "QA-08": "query_frame",
    "QA-09": "query_frame_else_observation_window",
    "QA-10": "query_frame",
    "QA-11": "query_frame_else_observation_window",
    "QA-12": "first_event_start_frame",
    "QA-13": "query_frame",
    "QA-14": "query_frame",
    "QA-15": "event_window",
    "QA-16": "query_frame",
    "QA-17": "query_frame",
    "QA-18": "query_frame",
    "QA-19": "first_event_start_frame",
    "QA-20": "query_frame",
    "QA-21": "first_event_start_frame",
    "QA-22": "whole_clip",
    "QA-23": "whole_clip",
    "QA-24": "final_frame",
    "QA-25": "query_frame",
}

#: 表里没有的题型（以后新加的）按这个顺序找一个查询时刻，找到哪个就记哪个规则名。
FALLBACK_FRAME_FIELDS: tuple[str, ...] = (
    "query_frame",
    "target_frame",
    "entry_frame",
    "final_frame",
)

#: 每条取帧规则一句人话，报告和汇总工具直接印这张表。
FRAME_RULE_NOTES: dict[str, str] = {
    "query_frame": "题面自己给了查询时刻，用证据里的 query_frame。",
    "target_frame": "题问的是某个事件的发声者，用证据里的 target_frame。",
    "entry_frame": "题问的是入画方向，用入画那一帧 entry_frame。",
    "final_frame": "题问的是片尾状态，用证据里的 final_frame。",
    "first_event_start_frame": "题面锚在第一个（或指定的那个）发声事件上，用该事件的起始帧。",
    "query_frame_else_observation_window": "转场题：发生过那次转场就用 query_frame；"
    "答案是没发生过时证据里的 query_frame 写的是 null（实测 47/47 如此），"
    "改用 observation_window 整个窗口逐帧取状态。",
    "event_window": "答案是发声窗口上的一个统计量，整个窗口逐帧取状态：窗口内不变就记那个状态，"
    "变了记 window_mixed。",
    "whole_clip": "答案是整段的统计量，没有单一查询时刻，记 whole_clip，分层看直方图。",
}


_COLOCATION_CACHE: dict[str, dict[str, Any]] = {}

#: “同位”按逐帧位置完全相等算。实测两批题库都是正好 0，所以没必要给一个容差去掩护什么。
COLOCATION_TOLERANCE_M = 0.0


def listener_camera_colocation(facts: Mapping[str, Any]) -> dict[str, Any]:
    """听者和相机是不是同一个点。量出来才声明，量不到就写 null 加理由。

    QA-14 问的是“谁离听者更近”。只看画面能不能答，完全取决于听者是不是就坐在相机那个点上；
    第一轮没有这条事实，所以那 22 题写的是 null。相机位姿不在 facts 里，但 facts 自己记了
    ``source_paths.frame_readbacks``，顺着它读就行；读法用仓库自己那份（``camera_position_series``），
    不另写一套单位与坐标转换。结果按读数文件路径缓存，一段只读一次。
    """

    listener = facts.get("listener")
    positions = listener.get("positions_m") if isinstance(listener, Mapping) else None
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)) or not positions:
        return {"colocated": None, "reason": "facts 里没有听者的逐帧位置",
                "max_separation_m": None, "frames_compared": 0,
                "camera_pose_path": None, "camera_pose_source": None}
    path = (facts.get("source_paths") or {}).get("frame_readbacks")
    if not path:
        return {"colocated": None,
                "reason": "facts 的 source_paths 里没有 frame_readbacks，取不到相机位姿",
                "max_separation_m": None, "frames_compared": 0,
                "camera_pose_path": None, "camera_pose_source": None}
    key = str(path)
    if key not in _COLOCATION_CACHE:
        _COLOCATION_CACHE[key] = _measure_colocation(key, positions)
    return dict(_COLOCATION_CACHE[key])


def _measure_colocation(path: str, positions: Sequence[Any]) -> dict[str, Any]:
    from avengine.qa.unified_catalog import camera_position_series

    blank = {"colocated": None, "max_separation_m": None, "frames_compared": 0,
             "camera_pose_path": path, "camera_pose_source": None}
    try:
        readbacks = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        return {**blank, "reason": f"读不到相机位姿读数：{type(error).__name__}"}
    if not isinstance(readbacks, Mapping):
        return {**blank, "reason": "相机位姿读数不是一个对象"}
    camera = camera_position_series(readbacks, frame_count=len(positions))
    if camera is None:
        return {**blank, "reason": "这份读数里没有认得出来的逐帧相机位置"}
    series = camera["positions_m"]
    worst = max(math.dist(a, b) for a, b in zip(positions, series))
    colocated = worst <= COLOCATION_TOLERANCE_M
    return {"colocated": colocated, "max_separation_m": worst,
            "frames_compared": len(series), "camera_pose_path": path,
            "camera_pose_source": camera["source"],
            "reason": None if colocated else
            f"听者与相机最远差 {worst} m，不是同一个点"}


class _NotDerivable(Exception):
    """某道题的单模态候选数推不出来，带上一句原因。"""


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _evidence(item: Mapping[str, Any]) -> Mapping[str, Any]:
    truth = item.get("truth")
    if isinstance(truth, Mapping) and isinstance(truth.get("evidence"), Mapping):
        return truth["evidence"]
    if isinstance(item.get("evidence"), Mapping):
        return item["evidence"]
    return {}


def _options(item: Mapping[str, Any]) -> list[str] | None:
    """MCQ 选项的 value 列表；只有开放式答案的题返回 None。"""

    forms = item.get("forms")
    if not isinstance(forms, Mapping):
        return None
    mcq = forms.get("mcq")
    if not isinstance(mcq, Mapping):
        return None
    options = mcq.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)) or not options:
        return None
    values: list[str] = []
    for option in options:
        if isinstance(option, Mapping) and "value" in option:
            values.append(str(option["value"]))
    return values or None


def _frame_count(facts: Mapping[str, Any]) -> int:
    clock = facts.get("time")
    if isinstance(clock, Mapping):
        count = _int_or_none(clock.get("frame_count"))
        if count:
            return count
    visibility = facts.get("visibility")
    if isinstance(visibility, Mapping):
        longest = 0
        for frames in visibility.values():
            if isinstance(frames, Mapping):
                longest = max(longest, len(frames))
        if longest:
            return longest
    return 0


def _state_at(facts: Mapping[str, Any], actor_id: str, frame: int) -> str | None:
    visibility = facts.get("visibility")
    if not isinstance(visibility, Mapping):
        return None
    frames = visibility.get(actor_id)
    if not isinstance(frames, Mapping):
        return None
    entry = frames.get(str(frame))
    if entry is None:
        entry = frames.get(frame)
    if not isinstance(entry, Mapping):
        return None
    state = entry.get("state")
    return str(state) if state is not None else None


def visibility_histogram(facts: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """整段每个演员四态各多少帧；出现表外的状态记进 ``other``。"""

    histogram: dict[str, dict[str, int]] = {}
    visibility = facts.get("visibility")
    if not isinstance(visibility, Mapping):
        return histogram
    for actor_id, frames in visibility.items():
        if not isinstance(frames, Mapping):
            continue
        counts = {state: 0 for state in PIXEL_STATES}
        counts["other"] = 0
        for entry in frames.values():
            state = entry.get("state") if isinstance(entry, Mapping) else None
            key = str(state) if str(state) in PIXEL_STATES else "other"
            counts[key] += 1
        counts["frames"] = len(frames)
        histogram[str(actor_id)] = counts
    return histogram


def _events(facts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = facts.get("events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return []
    return [event for event in events if isinstance(event, Mapping)]


def _first_event_start_frame(
    evidence: Mapping[str, Any], facts: Mapping[str, Any]
) -> tuple[int | None, str | None]:
    for key in ("first_event", "event", "anchor_event"):
        block = evidence.get(key)
        if isinstance(block, Mapping):
            frame = _int_or_none(block.get("start_frame"))
            if frame is not None:
                return frame, None
    frame = _int_or_none(evidence.get("start_frame"))
    if frame is not None:
        return frame, None
    wanted = evidence.get("event_ids")
    events = _events(facts)
    if isinstance(wanted, Sequence) and not isinstance(wanted, (str, bytes)):
        chosen = [event for event in events if event.get("event_id") in set(wanted)]
        frames = [
            value
            for value in (_int_or_none(event.get("start_frame")) for event in chosen)
            if value is not None
        ]
        if frames:
            return min(frames), None
    return None, "证据里没有可用的事件起始帧（first_event/event/anchor_event/start_frame 都没有）"


def resolve_query_frames(
    qa_id: str, evidence: Mapping[str, Any], facts: Mapping[str, Any]
) -> dict[str, Any]:
    """按取帧规则表定出这道题的查询时刻。返回规则名、单帧、窗口和取不到时的原因。"""

    rule = QUERY_FRAME_RULES.get(qa_id)
    if rule is None:
        for field in FALLBACK_FRAME_FIELDS:
            frame = _int_or_none(evidence.get(field))
            if frame is not None:
                return {
                    "rule": f"unlisted_type_{field}",
                    "frame": frame,
                    "window_frames": None,
                    "reason": None,
                }
        frame, reason = _first_event_start_frame(evidence, facts)
        if frame is not None:
            return {
                "rule": "unlisted_type_first_event_start_frame",
                "frame": frame,
                "window_frames": None,
                "reason": None,
            }
        return {
            "rule": "unlisted_type",
            "frame": None,
            "window_frames": None,
            "reason": f"{qa_id} 不在取帧规则表里，证据里也找不到查询时刻：{reason}",
        }

    if rule == "whole_clip":
        return {"rule": rule, "frame": None, "window_frames": None, "reason": None}

    if rule == "query_frame_else_observation_window":
        frame = _int_or_none(evidence.get("query_frame"))
        if frame is not None:
            return {"rule": rule, "frame": frame, "window_frames": None, "reason": None}
        window = evidence.get("observation_window")
        if (
            isinstance(window, Sequence)
            and not isinstance(window, (str, bytes))
            and len(window) == 2
        ):
            start, end = _int_or_none(window[0]), _int_or_none(window[1])
            if start is not None and end is not None:
                return {
                    "rule": rule,
                    "frame": None,
                    "window_frames": [min(start, end), max(start, end)],
                    "reason": None,
                }
        return {
            "rule": rule,
            "frame": None,
            "window_frames": None,
            "reason": "证据里 query_frame 是 null，observation_window 也不可用",
        }

    if rule == "event_window":
        window = evidence.get("motion_window_frames")
        start = end = None
        if isinstance(window, Sequence) and not isinstance(window, (str, bytes)) and len(window) == 2:
            start, end = _int_or_none(window[0]), _int_or_none(window[1])
        if start is None or end is None:
            start = _int_or_none(evidence.get("start_frame"))
            end = _int_or_none(evidence.get("end_frame"))
        if start is None or end is None:
            return {
                "rule": rule,
                "frame": None,
                "window_frames": None,
                "reason": "证据里没有发声窗口（motion_window_frames 和 start/end_frame 都没有）",
            }
        return {
            "rule": rule,
            "frame": None,
            "window_frames": [min(start, end), max(start, end)],
            "reason": None,
        }

    if rule == "first_event_start_frame":
        frame, reason = _first_event_start_frame(evidence, facts)
        return {"rule": rule, "frame": frame, "window_frames": None, "reason": reason}

    field = {
        "query_frame": "query_frame",
        "target_frame": "target_frame",
        "entry_frame": "entry_frame",
        "final_frame": "final_frame",
    }[rule]
    frame = _int_or_none(evidence.get(field))
    if frame is None:
        reason = (
            f"证据里没有 {field}"
            if field not in evidence
            else f"证据里的 {field} 写的是 null，这道题没有可用的查询时刻"
        )
        return {"rule": rule, "frame": None, "window_frames": None, "reason": reason}
    return {"rule": rule, "frame": frame, "window_frames": None, "reason": None}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key) for key in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(entry) for entry in value if entry is not None]
    return []


def resolve_target_actors(
    qa_id: str, item: Mapping[str, Any], facts: Mapping[str, Any]
) -> tuple[list[str], str | None]:
    """这道题问的是哪几个演员。多目标题（QA-03 候选、QA-14 两人）逐个记。"""

    evidence = _evidence(item)
    if qa_id == "QA-03":
        actors = _string_list(evidence.get("candidate_actor_ids"))
    elif qa_id == "QA-14":
        actors = _string_list(evidence.get("appearance_reviews")) or _string_list(
            evidence.get("distances_m")
        )
    elif qa_id == "QA-18":
        actors = _string_list(evidence.get("active_actor_ids"))
        if not actors:
            return [], "查询窗口里没有任何声源在发声，这道题没有被问到的演员"
    elif qa_id == "QA-22":
        actors = _string_list(evidence.get("appeared_actor_ids"))
    elif qa_id in ("QA-05", "QA-23"):
        wanted = set(_string_list(evidence.get("event_ids")))
        actors = [
            str(event["actor_id"])
            for event in _events(facts)
            if event.get("event_id") in wanted and event.get("actor_id")
        ]
    else:
        actors = []

    if not actors:
        for key in ("target_actor_id", "actor_id"):
            value = evidence.get(key)
            if value:
                actors = [str(value)]
                break
    if not actors:
        for key in ("first_event", "event", "anchor_event"):
            block = evidence.get(key)
            if isinstance(block, Mapping) and block.get("actor_id"):
                actors = [str(block["actor_id"])]
                break
    seen: set[str] = set()
    ordered = [a for a in actors if not (a in seen or seen.add(a))]
    if not ordered:
        return [], "证据里没有指名任何演员"
    return ordered, None


def _window_states(
    facts: Mapping[str, Any], actor_id: str, window: Sequence[int]
) -> tuple[dict[str, int], str | None]:
    start, end = int(window[0]), int(window[1])
    counts: dict[str, int] = {}
    missing = 0
    for frame in range(start, end + 1):
        state = _state_at(facts, actor_id, frame)
        if state is None:
            missing += 1
            continue
        counts[state] = counts.get(state, 0) + 1
    if not counts:
        return counts, None
    if len(counts) == 1 and missing == 0:
        return counts, next(iter(counts))
    return counts, "window_mixed"


def _cross_check(
    qa_id: str, evidence: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """题目证据里已经写着可见状态的三种题，用来对拍本模块的取帧与查表。"""

    if not targets:
        return None
    observed = targets[0].get("state")
    if qa_id == "QA-08" and "visibility_state" in evidence:
        expected = evidence.get("visibility_state")
        return {
            "field": "visibility_state",
            "expected": expected,
            "observed": observed,
            "agrees": expected == observed,
        }
    if qa_id == "QA-24" and "final_visibility_state" in evidence:
        expected = evidence.get("final_visibility_state")
        return {
            "field": "final_visibility_state",
            "expected": expected,
            "observed": observed,
            "agrees": expected == observed,
        }
    if qa_id == "QA-20" and "target_visible" in evidence:
        expected = bool(evidence.get("target_visible"))
        return {
            "field": "target_visible",
            "expected": expected,
            "observed": observed in VISIBLE_STATES,
            "agrees": expected == (observed in VISIBLE_STATES),
        }
    return None


def visibility_at_query(item: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    """字段一：被问演员在查询时刻的可见状态，多目标逐个记。"""

    qa_id = str(item.get("qa_id"))
    evidence = _evidence(item)
    resolved = resolve_query_frames(qa_id, evidence, facts)
    actors, actor_reason = resolve_target_actors(qa_id, item, facts)
    rule = resolved["rule"]
    reasons = [text for text in (resolved["reason"], actor_reason) if text]

    targets: list[dict[str, Any]] = []
    for actor_id in actors:
        record: dict[str, Any] = {
            "actor_id": actor_id,
            "frame": resolved["frame"],
            "state": "unknown",
            "source": "facts.visibility",
        }
        if rule == "whole_clip":
            record["frame"] = None
            record["state"] = "whole_clip"
        elif resolved["window_frames"]:
            counts, state = _window_states(facts, actor_id, resolved["window_frames"])
            record["frame"] = None
            record["window_frames"] = list(resolved["window_frames"])
            record["window_histogram"] = counts
            if state is None:
                record["reason"] = f"{actor_id} 在 facts.visibility 里没有这个窗口的逐帧状态"
            else:
                record["state"] = state
        elif resolved["frame"] is not None:
            state = _state_at(facts, actor_id, resolved["frame"])
            if state is None:
                record["reason"] = (
                    f"{actor_id} 在 facts.visibility 里没有第 {resolved['frame']} 帧的状态"
                )
            else:
                record["state"] = state
        else:
            record["reason"] = resolved["reason"] or "没有定出查询时刻"
        targets.append(record)

    if not targets:
        state = "unknown"
    elif rule == "whole_clip":
        state = "whole_clip"
    else:
        distinct = {record["state"] for record in targets}
        state = distinct.pop() if len(distinct) == 1 else "mixed"

    if state == "unknown" and not reasons:
        reasons.append("没有取到可见状态")
    return {
        "rule": rule,
        "rule_note": FRAME_RULE_NOTES.get(rule.replace("unlisted_type_", ""), ""),
        "frame": resolved["frame"],
        "window_frames": resolved["window_frames"],
        "state": state,
        "targets": targets,
        "reason": "；".join(reasons) or None,
        "cross_check": _cross_check(qa_id, evidence, targets),
    }


# --------------------------------------------------------------------------
# 字段二：单模态候选数
# --------------------------------------------------------------------------


class _Context:
    def __init__(self, item: Mapping[str, Any], facts: Mapping[str, Any], options: list[str],
                 option_domain_source: str = "mcq"):
        self.item = item
        self.facts = facts
        self.evidence = _evidence(item)
        self.options = options
        self.option_count = len(options)
        self.option_domain_source = option_domain_source

    @property
    def actors(self) -> Mapping[str, Any]:
        actors = self.facts.get("actors")
        return actors if isinstance(actors, Mapping) else {}

    @property
    def events(self) -> list[Mapping[str, Any]]:
        return _events(self.facts)

    @property
    def sounding_actor_ids(self) -> set[str]:
        return {str(event["actor_id"]) for event in self.events if event.get("actor_id")}

    @property
    def appeared_actor_ids(self) -> set[str]:
        appeared = set()
        for actor_id, counts in visibility_histogram(self.facts).items():
            if counts.get("frames", 0) - counts.get("out_of_view", 0) > 0:
                appeared.add(actor_id)
        return appeared

    @property
    def reviewed_actor_ids(self) -> set[str]:
        review = self.facts.get("appearance_review")
        if not isinstance(review, Mapping):
            return set()
        return {
            str(actor_id)
            for actor_id, record in review.items()
            if isinstance(record, Mapping) and record.get("status") == "reviewed"
        }

    @property
    def sound_classes(self) -> set[str]:
        return {
            str(event["sound_class"]) for event in self.events if event.get("sound_class")
        }

    def appearance_value_owners(self) -> dict[str, set[str]]:
        owners: dict[str, set[str]] = {}
        for actor_id, actor in self.actors.items():
            appearance = actor.get("appearance") if isinstance(actor, Mapping) else None
            if isinstance(appearance, Mapping) and appearance.get("value") is not None:
                owners.setdefault(str(appearance["value"]), set()).add(str(actor_id))
        return owners

    @property
    def frame_rate_hz(self) -> float:
        return float(self.facts["time"]["frame_rate_hz"])

    def world_tracks(self) -> dict[str, list[Any]]:
        tracks = {}
        for actor_id, actor in self.actors.items():
            points = actor.get("root_positions_m") if isinstance(actor, Mapping) else None
            if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
                tracks[str(actor_id)] = list(points)
        return tracks

    def listener_positions(self) -> list[Any] | None:
        listener = self.facts.get("listener")
        points = listener.get("positions_m") if isinstance(listener, Mapping) else None
        if isinstance(points, Sequence) and not isinstance(points, (str, bytes)) and points:
            return list(points)
        return None

    def listener_relative_track(self, actor_id: str) -> list[list[float]] | None:
        """同一条轨迹换到听者坐标里。只听的人拿到的就是这个。"""

        actor = self.actors.get(actor_id)
        points = actor.get("root_positions_m") if isinstance(actor, Mapping) else None
        listener = self.listener_positions()
        if not points or not listener or len(listener) < len(points):
            return None
        try:
            return [[float(point[axis]) - float(listener[index][axis]) for axis in range(3)]
                    for index, point in enumerate(points)]
        except (TypeError, ValueError, IndexError):
            return None

    def distance_series_by_actor(self) -> dict[str, list[float]]:
        listener = self.listener_positions()
        if not listener:
            return {}
        series = {}
        for actor_id, actor in self.actors.items():
            if not isinstance(actor, Mapping):
                continue
            points = actor.get("emitter_positions_m") or actor.get("root_positions_m")
            if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
                continue
            count = min(len(points), len(listener))
            try:
                series[str(actor_id)] = [math.dist(points[i], listener[i]) for i in range(count)]
            except (TypeError, ValueError):
                continue
        return series

    def visibility_tracks(self) -> dict[str, list[Any]]:
        """逐条可见状态轨迹，按帧号数值排序，跟出题函数走同一个共用函数。

        这两边必须是同一种遍历顺序，否则收据描述的判定跟题库里真正发生的判定不是一回事。
        """

        tracks = {}
        for actor_id, frames in (self.facts.get("visibility") or {}).items():
            series = visibility_series(frames)
            if series:
                tracks[str(actor_id)] = series
        return tracks

    def moving_flags(self, actor_id: str) -> list[Any] | None:
        actor = self.actors.get(actor_id)
        flags = actor.get("moving") if isinstance(actor, Mapping) else None
        if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)):
            return list(flags)
        return None

    def event_window(self) -> tuple[int, int]:
        # motion_window_frames 本来就是半开区间（出题函数按 [start, end) 调用判据），直接用。
        window = self.evidence.get("motion_window_frames")
        if isinstance(window, Sequence) and not isinstance(window, (str, bytes)) and len(window) == 2:
            low, high = _int_or_none(window[0]), _int_or_none(window[1])
            if low is not None and high is not None:
                return min(low, high), max(low, high)
        start = end = None
        if start is None or end is None:
            start = _int_or_none(self.evidence.get("start_frame"))
            end = _int_or_none(self.evidence.get("end_frame"))
        if start is None or end is None:
            raise _NotDerivable("证据里没有发声窗口")
        # start_frame / end_frame 是事件的首末帧（闭区间），而判据要的是半开区间。
        return min(start, end), max(start, end) + 1

    def count_reachable(self, reachable: Mapping[str, bool]) -> int:
        """选项里有几个是画面排除不掉的。认不出来的选项一律算排除不掉。"""

        total = sum(1 for value in self.options if reachable.get(value, True))
        if total == 0:
            raise _NotDerivable("没有任何一个选项是画面做得出来的，这不可能，不报数")
        return total

    def require(self, key: str) -> Any:
        if key not in self.evidence or self.evidence.get(key) is None:
            raise _NotDerivable(f"证据里没有 {key}")
        return self.evidence[key]


def _qa_01(ctx: _Context) -> tuple[int, int, str]:
    audio = 1 if not ctx.events else ctx.option_count
    return (
        audio,
        ctx.option_count,
        "问的是某个按外观点名的个体有没有发过声。画面里没有任何「谁在发声」的证据，两个选项都"
        "留着；只听能确定「整段一声没有」这一种情况（那时只剩 no），否则听到的声音绑不到这个"
        "外观上，也是两个都留着——片中一共有几个个体是视觉事实。",
    )


def _qa_02(ctx: _Context) -> tuple[int, int, str]:
    owners = ctx.appearance_value_owners()
    sounding = ctx.sounding_actor_ids
    identifiable = ctx.reviewed_actor_ids & ctx.appeared_actor_ids
    audio = video = 0
    for value in ctx.options:
        holders = owners.get(value)
        if not holders:
            audio += 1
            video += 1
            continue
        if holders & sounding:
            audio += 1
        if holders & identifiable:
            video += 1
    return (
        audio,
        video,
        "选项是各个已登记外观。只听能把发声者缩到「确实发过声的那些源」，一个整段没出过声的个体"
        "被排除；只看能确认「这个外观确实在画面里且复核通过」，认不出来的那个被排除。对不上任何"
        "演员的选项两边都不排除。",
    )


def _qa_03(ctx: _Context) -> tuple[int, int, str]:
    sounding = ctx.sounding_actor_ids
    identifiable = ctx.reviewed_actor_ids & ctx.appeared_actor_ids
    audio = video = 0
    for value in ctx.options:
        if value not in ctx.actors:
            audio += 1
            video += 1
            continue
        if value in sounding:
            audio += 1
        if value in identifiable:
            video += 1
    return (
        audio,
        video,
        "选项是几个候选演员。只听把候选缩到发过声的那些，只看把候选缩到画面里认得出来的那些。"
        "这一类题的候选本来就是「复核过外观且发过声」的人，所以两边通常都等于选项数。",
    )


def _qa_04(ctx: _Context) -> tuple[int, int, str]:
    azimuth = float(ctx.require("azimuth_deg"))
    folded = abs(((azimuth + 180.0) % 360.0) - 180.0)
    on_median = folded < 1e-9 or abs(folded - 180.0) < 1e-9
    return (
        ctx.option_count if on_median else 1,
        ctx.option_count,
        "问的是发声瞬间声源在左还是在右。方位角是听得出来的：只要它不是正好 0 度或正好 180 度，"
        "就严格落在一边；只看不行，因为画面里挑不出是哪个源、也挑不出是哪一刻。",
    )


def _qa_05(ctx: _Context) -> tuple[int, int, str]:
    return (
        1,
        ctx.option_count,
        "问的是两个独立发声事件有没有重叠。两个事件的起止都在音频里，重不重叠听得出来；画面里"
        "没有事件表，两个选项都留着。",
    )


def _qa_08(ctx: _Context) -> tuple[int, int, str]:
    reachable = {"out_of_view"}
    visibility = ctx.facts.get("visibility")
    if not isinstance(visibility, Mapping) or not visibility:
        raise _NotDerivable("facts 里没有逐帧可见状态")
    for frames in visibility.values():
        if isinstance(frames, Mapping):
            for entry in frames.values():
                if isinstance(entry, Mapping) and entry.get("state") is not None:
                    reachable.add(str(entry["state"]))
    video = sum(
        1 for value in ctx.options if value not in PIXEL_STATES or value in reachable
    )
    return (
        ctx.option_count,
        video,
        "问的是某个发声事件开始时声源的可见状态。声音里没有遮挡信息，四个状态一个都排除不掉；"
        "只看能排除的只有「整段任何演员任何一帧都没出现过」的那个状态，而且必须保留 out_of_view，"
        "因为声源可能是一个镜头里从没见过的个体。",
    )


def _qa_12(ctx: _Context) -> tuple[int, int, str]:
    spoken = {
        str(event["transcript"]) for event in ctx.events if isinstance(event.get("transcript"), str)
    }
    if not spoken:
        raise _NotDerivable("facts 的事件里没有任何台词文本")
    audio = sum(1 for value in ctx.options if value in spoken)
    return (
        audio,
        ctx.option_count,
        "问的是某个人说了哪一句。只听能排除「整段根本没被说出来过」的句子，说出来过的都留着——"
        "声音绑不到外观上；只看一句话也听不见，选项全留。",
    )


def _qa_13(ctx: _Context) -> tuple[int, int, str]:
    """哪一条视野横向带装着声源——**查询时刻声源是静音的**。

    第一轮我按「方位角落在哪一条带里」判只听唯一，那是错的：这道题问的是事件结束之后那段静音里的
    方位，声源那时候不出声，只听拿不到它当时在哪。只听真正握着的是事件结束那一刻的方位；
    要把它延到查询时刻，必须声源在这段时间里没动过，而「动没动」正是拆出来那条判据能判的。
    """

    if ctx.option_domain_source == "derived":
        # 这一支的题面要的是一个整数度数，不是带；静音时刻两边都定不了它。
        return (
            ctx.option_count,
            ctx.option_count,
            "开放作答那一支问的是静音时段里的整数方位角。声源那时不出声，只听拿不到；"
            "画面里挑不出是哪个源，只看也拿不到。这里的两个数是上界——没有把"
            "「答案必然是画面里那几个人的方位之一」这条再用上去，所以它们只用来说明两边都不唯一。",
        )
    boundaries = ctx.require("fov_band_boundaries_deg")
    azimuth = float(ctx.require("azimuth_at_query_deg"))
    if not isinstance(boundaries, Sequence) or len(boundaries) < 2:
        raise _NotDerivable("fov_band_boundaries_deg 不是一串可用的边界")
    edges = [float(value) for value in boundaries]
    bands: dict[str, tuple[float, float]] = {
        f"fov_band_{index}": (edges[index], edges[index + 1])
        for index in range(len(edges) - 1)
    }

    def band_of(value: float) -> str | None:
        for name, (low, high) in bands.items():
            last = name == f"fov_band_{len(edges) - 2}"
            if low <= value < high or (last and value == high):
                return name
        return None

    still = _still_between_event_end_and_query(ctx)
    query_band = band_of(azimuth)
    if still and query_band is not None:
        audio = sum(1 for value in ctx.options if value not in bands or value == query_band)
        heard = "声源在事件结束到查询时刻之间没有明显移动，所以事件结束时听到的那个方位一直有效"
    else:
        audio = ctx.option_count
        heard = ("声源在这段静音里动过（或者动没动判不出来），事件结束时听到的方位延不到查询时刻，"
                 "所以每条带都排除不掉")
    return (
        audio,
        ctx.option_count,
        "问的是静音时段里声源落在哪条视野横向带里。" + heard + "；只看不行，因为画面里挑不出是哪个源。",
    )


def _still_between_event_end_and_query(ctx: _Context) -> bool:
    """事件结束到查询时刻之间，声源有没有明显移动。判不出来一律当成动过。"""

    try:
        policy = _motion_policy(ctx)
    except _NotDerivable:
        return False
    actor_id = str(ctx.evidence.get("actor_id") or ctx.evidence.get("target_actor_id") or "")
    end_frame = _int_or_none(ctx.evidence.get("end_frame"))
    query_frame = _int_or_none(ctx.evidence.get("query_frame"))
    relative = ctx.listener_relative_track(actor_id)
    if not actor_id or end_frame is None or query_frame is None or relative is None:
        return False
    low, high = min(end_frame, query_frame), max(end_frame, query_frame) + 1
    if high > len(relative) or high <= low + 1:
        return False
    return noticeable_motion(relative[low:high], ctx.frame_rate_hz, policy)["moving"] is False

def _qa_18(ctx: _Context) -> tuple[int, int, str]:
    active = ctx.evidence.get("active_actor_ids")
    if active is None:
        raise _NotDerivable("证据里没有 active_actor_ids")
    actor_options = [value for value in ctx.options if value in ctx.actors]
    if len(active) == 0 or len(active) >= 2:
        audio = 1
    else:
        audio = len(actor_options) or ctx.option_count
    return (
        audio,
        ctx.option_count,
        "问的是查询窗口里谁在发声。只听能数出窗口里有几个源在响：零个就只剩 none，两个以上就只剩"
        " multiple，正好一个时 none 和 multiple 都被排除、只剩那几个演员选项；只看完全听不见，"
        "选项全留。",
    )


def _qa_19(ctx: _Context) -> tuple[int, int, str]:
    bands = ctx.require("time_bands_s")
    if not isinstance(bands, Sequence) or not bands:
        raise _NotDerivable("time_bands_s 不是一串可用的时间段")
    onsets = [
        float(event["start_s"]) for event in ctx.events if event.get("start_s") is not None
    ]
    if not onsets:
        raise _NotDerivable("facts 的事件里没有起始时刻")
    audio = 0
    for value in ctx.options:
        index = None
        if value.startswith("band_"):
            index = _int_or_none(value[len("band_") :])
        if index is None or not 0 <= index < len(bands):
            audio += 1
            continue
        low, high = float(bands[index][0]), float(bands[index][1])
        last = index == len(bands) - 1
        if any(low <= onset < high or (last and onset == high) for onset in onsets):
            audio += 1
    return (
        audio,
        ctx.option_count,
        "问的是某个人的第一声落在哪个时间段。那一声的起始必定是整段里某个事件的起始，所以只听"
        "能排除「里面一个事件起始都没有」的时间段；只看没有任何发声时刻，选项全留。",
    )


def _qa_20(ctx: _Context) -> tuple[int, int, str]:
    sounding = ctx.sounding_actor_ids
    audio = 0
    for value in ctx.options:
        if value not in ctx.actors or value in sounding:
            audio += 1
    return (
        audio,
        ctx.option_count,
        "问的是画面里哪个可见演员发的这一声，或者都不是。只听能排除「整段没出过声」的可见演员，"
        "但排除不掉「都不是」这个选项；只看能确认谁可见，却不知道谁发的声，选项全留。",
    )


def _qa_21(ctx: _Context) -> tuple[int, int, str]:
    heard = ctx.sound_classes
    if not heard:
        raise _NotDerivable("facts 的事件里没有声音类别")
    audio = sum(1 for value in ctx.options if value in heard)
    return (
        audio,
        ctx.option_count,
        "问的是某个个体发的是哪一类声音。选项是整套类别表，只听能排除整段根本没出现过的类别；"
        "只看听不见类别，选项全留。这里没有用「可见发声者数」，因为画面本身给不出声音类别。",
    )


def _qa_22(ctx: _Context) -> tuple[int, int, str]:
    entity_count = _int_or_none(ctx.require("entity_count"))
    speaking_count = _int_or_none(ctx.require("speaking_count"))
    if entity_count is None or speaking_count is None:
        raise _NotDerivable("entity_count 或 speaking_count 不是整数")
    audio = video = 0
    for value in ctx.options:
        parts = value.split("|")
        if len(parts) != 2:
            audio += 1
            video += 1
            continue
        entities, speakers = _int_or_none(parts[0]), _int_or_none(parts[1])
        if entities is None or speakers is None:
            audio += 1
            video += 1
            continue
        if speakers == speaking_count:
            audio += 1
        if entities == entity_count:
            video += 1
    return (
        audio,
        video,
        "选项是「几个个体|其中几个发声」的组合。只看数得出出场个体数，留下前一半对上的；只听数得出"
        "发声源数，留下后一半对上的。两边各裁一刀，合起来才唯一。",
    )


def _qa_23(ctx: _Context) -> tuple[int, int, str]:
    return (
        1,
        ctx.option_count,
        "问的是整段有几个独立发声事件开始。这是纯听觉的计数，听得出来；画面里没有事件表，选项全留。",
    )


def _qa_24(ctx: _Context) -> tuple[int, int, str]:
    final_frame = _int_or_none(ctx.require("final_frame"))
    if final_frame is None:
        raise _NotDerivable("final_frame 不是整数")
    reachable = {"out_of_view"}
    visibility = ctx.facts.get("visibility")
    if not isinstance(visibility, Mapping) or not visibility:
        raise _NotDerivable("facts 里没有逐帧可见状态")
    for actor_id in visibility:
        state = _state_at(ctx.facts, str(actor_id), final_frame)
        if state is not None:
            reachable.add(state)
    video = sum(
        1 for value in ctx.options if value not in PIXEL_STATES or value in reachable
    )
    return (
        ctx.option_count,
        video,
        "问的是最先发声的那个个体在片尾的可见状态。声音里没有遮挡信息，四态都留着；片尾是哪一帧"
        "看画面就知道，所以只看能排除「片尾没有任何演员处在」的状态，但要保留 out_of_view，因为"
        "那个个体可能镜头里从没出现过。",
    )



def _motion_policy(ctx: _Context) -> Mapping[str, Any]:
    from avengine.qa.unified_catalog import _noticeable_motion_policy

    policy = _noticeable_motion_policy(ctx.facts)
    if policy is None:
        raise _NotDerivable("这段的验收政策里没有 noticeable_motion 门限")
    return policy


def _qa_06(ctx: _Context) -> tuple[int, int, str]:
    policy = _motion_policy(ctx)
    actor_id = str(ctx.require("actor_id"))
    start, end = ctx.event_window()
    tracks = ctx.world_tracks()
    world = tracks.get(actor_id)
    relative = ctx.listener_relative_track(actor_id)
    if world is None or relative is None:
        raise _NotDerivable("取不到这个演员的位置轨迹或听者位置")
    if end > len(world) or end > len(relative):
        raise _NotDerivable("发声窗口超出了位置读数的长度")
    world_verdict = noticeable_motion(world[start:end], ctx.frame_rate_hz, policy)["moving"]
    heard = noticeable_motion(relative[start:end], ctx.frame_rate_hz, policy)["moving"]
    audio = 1 if (heard is not None and heard == world_verdict) else ctx.option_count
    reach = motion_reachable_verdicts(tracks, ctx.frame_rate_hz, policy)
    video = ctx.count_reachable({"moving": reach["moving_reachable"],
                                 "still": reach["still_reachable"]})
    return (
        audio,
        video,
        "问的是发声期间声源动没动。只听拿得到的是声源相对听者的位置，把同一套门限套在这条"
        "相对轨迹上，判得出来且跟世界坐标的判定一致就算唯一；只看知道每个人每一帧在哪，"
        "却不知道问的是哪一段窗口，所以只能排除整段任何人任何窗口都做不出来的那个判定。",
    )


def _qa_15(ctx: _Context) -> tuple[int, int, str]:
    trend = ctx.evidence.get("distance_trend")
    verdict = trend.get("verdict") if isinstance(trend, Mapping) else None
    if verdict is None:
        raise _NotDerivable("证据里没有已判定的 distance_trend")
    margin = ((trend.get("criteria") or {}).get("min_net_change_m"))
    if not isinstance(margin, (int, float)):
        raise _NotDerivable("distance_trend 里没有 min_net_change_m")
    reach = distance_direction_reachable(ctx.distance_series_by_actor(),
                                         min_net_change_m=float(margin))
    video = ctx.count_reachable({"nearer": reach["nearer_reachable"],
                                 "farther": reach["farther_reachable"]})
    return (
        1,
        video,
        "问的是发声期间声源在靠近还是远离。已发布的判决就是从到听者的距离序列算出来的，而距离正是只听"
        "拿得到的东西，所以只听唯一；只看算得出距离却不知道问的是哪一段，只能排除整段朝那个方向"
        "一步都没走够过的选项。",
    )


def _qa_16(ctx: _Context) -> tuple[int, int, str]:
    margin = ctx.evidence.get("distance_margin_m")
    if not isinstance(margin, (int, float)):
        raise _NotDerivable("证据里没有 distance_margin_m")
    reach = distance_direction_reachable(ctx.distance_series_by_actor(),
                                         min_net_change_m=float(margin))
    video = ctx.count_reachable({"nearer": reach["nearer_reachable"],
                                 "farther": reach["farther_reachable"]})
    return (
        ctx.option_count,
        video,
        "问的是某一刻跟事件结束时相比距离更近还是更远。那一刻声源是静音的，只听根本拿不到它那时的"
        "距离，两个选项都留；只看知道秒数却不知道锚点是哪一帧、声源是谁，同样只能按方向可达性排除。",
    )


def _qa_17(ctx: _Context) -> tuple[int, int, str]:
    yes = no = False
    seen = False
    for actor_id in ctx.actors:
        flags = ctx.moving_flags(str(actor_id))
        if not flags:
            continue
        seen = True
        yes = yes or any(bool(flag) for flag in flags)
        no = no or any(not bool(flag) for flag in flags)
    if not seen:
        raise _NotDerivable("facts 里没有逐帧运动读数")
    video = ctx.count_reachable({"yes": yes, "no": no})
    return (
        ctx.option_count,
        video,
        "问的是静音段里声源动没动。那一段它不出声，只听什么也拿不到，两个选项都留；"
        "只看有逐帧运动读数，却不知道问的是谁、从哪一帧算起，只能排除整段没有任何人做得出来的那个答案。",
    )


def _occlusion_recovery(ctx: _Context, kind: str, subject: str) -> tuple[int, int, str]:
    reach = occlusion_recovery_reachable(ctx.visibility_tracks(), kind=kind)
    if not reach["candidate_track_count"]:
        raise _NotDerivable("没有任何一条轨迹满足这道题的前提")
    video = ctx.count_reachable({"yes": reach["yes_reachable"], "no": reach["no_reachable"]})
    return (
        ctx.option_count,
        video,
        f"问的是{subject}。声音里没有遮挡信息，两个选项都留；"
        "只看把转场判据对每条满足前提的轨迹跑一遍，够得着的答案就是排除不掉的那些。",
    )


def _qa_09(ctx: _Context) -> tuple[int, int, str]:
    return _occlusion_recovery(ctx, "full", "被完全遮住之后有没有重新露出来")


def _qa_11(ctx: _Context) -> tuple[int, int, str]:
    return _occlusion_recovery(ctx, "partial", "部分遮挡之后有没有变得清晰")


def _qa_14(ctx: _Context) -> tuple[int, int, str]:
    colocation = listener_camera_colocation(ctx.facts)
    if not colocation.get("colocated"):
        raise _NotDerivable(
            "听者与相机同位这件事没量出来：" + str(colocation.get("reason"))
        )
    return (
        ctx.option_count,
        1,
        "问的是谁离听者更近。实测听者逐帧位置与相机逐帧位置完全相等（最大差 "
        f"{colocation['max_separation_m']} m，比了 {colocation['frames_compared']} 帧，读数来源 "
        f"{colocation['camera_pose_source']}），所以“离听者更近”就是“离镜头更近”，"
        "画面自己就能定；只听能听出两个源的距离，却对不上选项上的外观，选项全留。",
    )


def _qa_10(ctx: _Context) -> tuple[int, int, str]:
    return (
        ctx.option_count,
        1,
        "问的是画面里哪个可见物体或人挡住了目标。声音里没有遮挡信息，候选一个都排除不掉；"
        "遮挡是像素事实，目标按外观点名、时刻写在题面上，所以只看就能定。",
    )


def _qa_25(ctx: _Context) -> tuple[int, int, str]:
    target = str(ctx.require("angle_target"))
    if target == "visible_pixel_centroid":
        return (
            ctx.option_count,
            1,
            "答案是可见像素质心的相机方位角，题面还给了针孔标定，所以只看就能算出唯一一个度数；"
            "声音里没有像素质心这回事，整个度数域都留着。",
        )
    if target == "sound_emitter":
        return (
            1,
            ctx.option_count,
            "答案是某个发声事件的方位角，方位是听得出来的，所以只听唯一；"
            "画面里挑不出是哪个事件、也没有事件表，整个度数域都留着。",
        )
    raise _NotDerivable(f"还没有写过 angle_target={target!r} 这一支的口径")


#: 能严格推的题型 → 计算函数。不在这张表里的写 null 和原因。
CANDIDATE_RULES = {
    "QA-01": _qa_01,
    "QA-02": _qa_02,
    "QA-03": _qa_03,
    "QA-04": _qa_04,
    "QA-05": _qa_05,
    "QA-06": _qa_06,
    "QA-08": _qa_08,
    "QA-09": _qa_09,
    "QA-10": _qa_10,
    "QA-11": _qa_11,
    "QA-12": _qa_12,
    "QA-13": _qa_13,
    "QA-14": _qa_14,
    "QA-15": _qa_15,
    "QA-16": _qa_16,
    "QA-17": _qa_17,
    "QA-18": _qa_18,
    "QA-19": _qa_19,
    "QA-20": _qa_20,
    "QA-21": _qa_21,
    "QA-22": _qa_22,
    "QA-23": _qa_23,
    "QA-24": _qa_24,
    "QA-25": _qa_25,
}

#: 推不出来的题型，各写一句为什么。写在这里比写在代码里好找。
#: 推不出来的题型，各写一句为什么。写在这里比写在代码里好找。
UNDERIVED_REASONS: dict[str, str] = {
    "QA-07": "答案是入画方向。只看知道每个入画的人从哪边进来，却不知道问的是谁；"
    "要把这变成候选数，得先定「只看的观察者认得出哪些人」，那是外观复核之外的另一套判定，本模块不做。",
}

def _qa_10_domain(item: Mapping[str, Any], facts: Mapping[str, Any]) -> list[str]:
    """QA-10 是开放作答，候选域取“查询时刻画面里看得见的、不是目标自己的个体”。"""

    evidence = _evidence(item)
    frame = _int_or_none(evidence.get("query_frame"))
    if frame is None:
        raise _NotDerivable("证据里没有 query_frame，圈不出候选域")
    target = str(evidence.get("target_actor_id") or "")
    domain = {str(value) for value in _string_list(evidence.get("option_instance_ids"))}
    for actor_id in (facts.get("visibility") or {}):
        if str(actor_id) == target:
            continue
        if _state_at(facts, str(actor_id), frame) in VISIBLE_STATES:
            domain.add(str(actor_id))
    domain.discard(target)
    if not domain:
        raise _NotDerivable("查询时刻画面里没有其他可见个体，候选域是空的")
    return sorted(domain)


#: 整数度数答案域：题面写的就是 [-180, 180) 里的一个整数度数。
WHOLE_DEGREE_DOMAIN_SIZE = 360


def _qa_25_domain(item: Mapping[str, Any], facts: Mapping[str, Any]) -> list[str]:
    return [str(value) for value in range(-180, 180)]


#: 没有 MCQ 选项、但答案域可以从证据或题面严格推出来的题型。
EVIDENCE_OPTION_DOMAINS = {
    "QA-10": _qa_10_domain,
    "QA-13": _qa_25_domain,
    "QA-25": _qa_25_domain,
}

_OPEN_ONLY_REASON = "这道题导出时没有封闭选项集合（只有开放式作答），候选数无从数起。"


def unimodal_candidates(item: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    """字段二：只听 / 只看 各剩几个候选答案。推不出来的写 null 和原因。"""

    qa_id = str(item.get("qa_id"))
    options = _options(item)
    blank: dict[str, Any] = {
        "audio_only": None,
        "video_only": None,
        "joint": None,
        "option_count": len(options) if options else None,
        "rule": None,
        "basis": None,
        "reason": None,
        "necessary_multimodal": None,
    }
    domain_source = "mcq" if options is not None else "derived"
    rule = CANDIDATE_RULES.get(qa_id)
    if rule is None:
        blank["reason"] = UNDERIVED_REASONS.get(
            qa_id, f"{qa_id} 没有写过严格的候选数口径，本轮不猜。"
        )
        return blank
    if options is None:
        derive = EVIDENCE_OPTION_DOMAINS.get(qa_id)
        if derive is None:
            blank["reason"] = _OPEN_ONLY_REASON
            return blank
        try:
            options = derive(item, facts)
        except _NotDerivable as error:
            blank["rule"] = qa_id
            blank["reason"] = str(error)
            return blank
        blank["option_count"] = len(options)
    try:
        audio, video, basis = rule(_Context(item, facts, options, domain_source))
    except _NotDerivable as error:
        blank["rule"] = qa_id
        blank["reason"] = str(error)
        return blank
    return {
        "audio_only": int(audio),
        "video_only": int(video),
        "joint": 1,
        "option_count": len(options),
        "option_domain_source": domain_source,
        "rule": qa_id,
        "basis": basis,
        "reason": None,
        "necessary_multimodal": bool(audio > 1 and video > 1),
    }


def question_strata(
    question_id: str,
    item: Mapping[str, Any],
    facts: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """一道题的私有分层记录，导出时按 question_id 一行写进 private/strata.jsonl。"""

    source = source or {}
    row: dict[str, Any] = {
        "question_id": question_id,
        "qa_id": str(item.get("qa_id")),
        "episode_id": source.get("episode_id"),
        "room_family": source.get("room_family"),
        "room_id": source.get("room_id"),
        "facts_path": source.get("facts_path"),
        "visibility_at_query": visibility_at_query(item, facts),
        "visibility_histogram": visibility_histogram(facts),
        "unimodal_candidates": unimodal_candidates(item, facts),
        "research_only": True,
    }
    return row


def summarize_strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """题型 × 房间 × 可见状态的题数表，和 unimodal_candidates 的可用率表。"""

    by_state: dict[str, dict[str, dict[str, int]]] = {}
    state_totals: dict[str, int] = {}
    availability: dict[str, dict[str, int]] = {}
    cross_check: dict[str, dict[str, int]] = {}
    necessary = {
        "counted": 0,
        "necessary_multimodal": 0,
        "audio_only_sufficient": 0,
        "video_only_sufficient": 0,
        "either_modality_sufficient": 0,
    }
    for row in rows:
        qa_id = str(row.get("qa_id"))
        room = str(row.get("room_family"))
        visibility = row.get("visibility_at_query") or {}
        state = str(visibility.get("state"))
        by_state.setdefault(qa_id, {}).setdefault(room, {})
        by_state[qa_id][room][state] = by_state[qa_id][room].get(state, 0) + 1
        state_totals[state] = state_totals.get(state, 0) + 1

        counts = row.get("unimodal_candidates") or {}
        bucket = availability.setdefault(qa_id, {"total": 0, "available": 0, "null": 0})
        bucket["total"] += 1
        if counts.get("audio_only") is None:
            bucket["null"] += 1
        else:
            bucket["available"] += 1
            necessary["counted"] += 1
            audio, video = counts.get("audio_only"), counts.get("video_only")
            if audio > 1 and video > 1:
                necessary["necessary_multimodal"] += 1
            elif audio == 1 and video == 1:
                necessary["either_modality_sufficient"] += 1
            elif audio == 1:
                necessary["audio_only_sufficient"] += 1
            elif video == 1:
                necessary["video_only_sufficient"] += 1

        check = visibility.get("cross_check")
        if isinstance(check, Mapping):
            field = str(check.get("field"))
            entry = cross_check.setdefault(field, {"checked": 0, "agrees": 0})
            entry["checked"] += 1
            entry["agrees"] += 1 if check.get("agrees") else 0
    return {
        "question_count": len(rows),
        "visibility_state_counts": dict(sorted(state_totals.items())),
        "visibility_by_qa_room_state": by_state,
        "unimodal_availability_by_qa": availability,
        "unimodal_necessity": necessary,
        "visibility_cross_checks": cross_check,
    }
