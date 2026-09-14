"""拆出来的观测判据：门限语义钉死，并且拆之前那份实现算出来的还是同一个数。

这些判据原来长在出题函数里。拆出来最大的风险不是拆坏，是拆完之后两份实现慢慢走样——出题按一套
门限判，收据按另一套门限判，两边都绿，结论却对不上。所以这里一半用例钉判据本身的语义（什么叫动过、
什么叫趋势成立、什么叫可达），另一半用 unified_catalog 的调用入口再跑一遍，确认它拿到的结果跟直接
调纯函数完全一样。整批题库级别的逐字节对拍在 tools/dataset/replay_question_catalog.py。
"""

from __future__ import annotations

import pytest

from avengine.qa.observation_predicates import (
    clear_after_partial_transitions,
    distance_direction_reachable,
    distance_step_verdict,
    distance_trend,
    full_occlusion_frames,
    motion_reachable_verdicts,
    moved_at_any_frame,
    noticeable_motion,
    occlusion_recovery_reachable,
    partial_occlusion_frames,
    reappeared_frames,
)

POLICY = {
    "mode": "noticeable_motion",
    "min_moving_duration_s": 0.8,
    "min_travel_m": 0.2,
    "max_still_travel_m": 0.05,
    "speed_threshold_mps": 0.05,
}
RATE = 15.0


def walk(step_m, frames=30):
    return [[index * step_m, 0.0, 0.0] for index in range(frames)]


# --------------------------------------------------------------------------
# 运动
# --------------------------------------------------------------------------


def test_走够了路又走够了时间才算动过():
    got = noticeable_motion(walk(0.05), RATE, POLICY, window_frames=[0, 30])
    assert got["moving"] is True
    assert got["measurement"]["travel_m"] == pytest.approx(0.05 * 29)
    assert got["measurement"]["window_frames"] == [0, 30]


def test_几乎没挪动算没动过():
    assert noticeable_motion(walk(0.001, frames=10), RATE, POLICY)["moving"] is False


def test_夹在两个门限中间的两样都不算():
    # 总路程 0.1 m：超过 0.05 的静止上限，够不着 0.2 的移动下限。
    got = noticeable_motion(walk(0.01, frames=11), RATE, POLICY)
    assert got["moving"] is None
    assert got["reason"] == "motion_between_noticeability_thresholds"


def test_走得够远但持续时间不够也不算动过():
    # 一帧跳 0.3 m 然后不动：路程够，但超过速度门限的时间只有 1/15 秒。
    points = [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]] + [[0.3, 0.0, 0.0]] * 28
    got = noticeable_motion(points, RATE, POLICY)
    assert got["moving"] is None


def test_读数坏了就说读数坏了而不是判一个():
    got = noticeable_motion([[0.0, 0.0, 0.0], [0.0, float("nan"), 0.0]], RATE, POLICY)
    assert got == {"moving": None, "reason": "invalid_motion_position_readback"}


def test_可达判定不靠穷举窗口():
    tracks = {"a": walk(0.05), "b": [[0.0, 0.0, 0.0]] * 30}
    got = motion_reachable_verdicts(tracks, RATE, POLICY)
    # a 整段走了 1.45 m，所以「动过」够得着；b 一直不动，所以「没动」也够得着
    assert got["moving_reachable"] is True
    assert got["still_reachable"] is True


def test_谁都没动过时动过就够不着():
    got = motion_reachable_verdicts({"a": [[0.0, 0.0, 0.0]] * 30}, RATE, POLICY)
    assert got["moving_reachable"] is False
    assert got["still_reachable"] is True


def test_每一帧都在猛走时没动就够不着():
    got = motion_reachable_verdicts({"a": walk(0.5)}, RATE, POLICY)
    assert got["moving_reachable"] is True
    assert got["still_reachable"] is False


def test_任意一帧动过就算动过():
    assert moved_at_any_frame([False, False, True]) is True
    assert moved_at_any_frame([False, False]) is False
    assert moved_at_any_frame([]) is False


# --------------------------------------------------------------------------
# 距离
# --------------------------------------------------------------------------


MARGINS = {"min_net_change_m": 0.2, "reversal_tolerance_m": 0.15, "reversal_fraction": 0.4}


def test_一路走远判远离():
    got = distance_trend([1.0, 1.2, 1.5, 2.0], **MARGINS)
    assert got["verdict"] == "farther"
    assert got["reason"] is None
    assert got["max_counter_trend_m"] == pytest.approx(0.0)


def test_净变化不够就不判():
    got = distance_trend([1.0, 1.05, 1.1], **MARGINS)
    assert got["verdict"] is None
    assert got["reason"] == "distance_net_change_below_margin"


def test_回撤太大就不判():
    # 净变化 +0.5，但中途回撤 0.4，超过 0.15 的绝对容差和 0.4×0.5=0.2 的比例容差
    got = distance_trend([1.0, 1.8, 1.4, 1.5], **MARGINS)
    assert got["verdict"] is None
    assert got["reason"] == "distance_trend_reverses"
    assert got["max_counter_trend_m"] == pytest.approx(0.4)


def test_首尾之差不是证据这件事写在返回值里():
    assert distance_trend([1.0, 2.0], **MARGINS)["endpoint_delta_is_not_sufficient"] is True


def test_某一刻跟锚点比的判定():
    assert distance_step_verdict(1.0, 1.5, 0.2)["trend"] == "farther"
    assert distance_step_verdict(1.5, 1.0, 0.2)["trend"] == "nearer"
    below = distance_step_verdict(1.0, 1.1, 0.2)
    assert below["trend"] is None and below["reason"] == "distance_change_below_margin"


def test_方向可达只用来排除():
    got = distance_direction_reachable({"a": [1.0, 1.1, 1.2, 1.3]}, min_net_change_m=0.2)
    assert got["farther_reachable"] is True
    assert got["nearer_reachable"] is False


def test_距离几乎不变时两个方向都够不着():
    got = distance_direction_reachable({"a": [1.0, 1.01, 1.0]}, min_net_change_m=0.2)
    assert got["farther_reachable"] is False
    assert got["nearer_reachable"] is False


# --------------------------------------------------------------------------
# 遮挡转场
# --------------------------------------------------------------------------


def track(*states):
    return [{"frame_index": index, "state": state} for index, state in enumerate(states)]


def test_完全遮挡之后重新看得见():
    ordered = track("visible_clear", "fully_occluded", "fully_occluded", "visible_clear")
    fully = full_occlusion_frames(ordered)
    assert fully == [1, 2]
    assert reappeared_frames(ordered, fully) == [3]


def test_没被完全遮住过就没有重新露出来这回事():
    ordered = track("visible_clear", "visible_occluded", "visible_clear")
    assert full_occlusion_frames(ordered) == []
    assert reappeared_frames(ordered, []) == []


def test_部分遮挡变清晰只认相邻两帧的状态():
    ordered = track("visible_occluded", "visible_clear", "visible_occluded", "out_of_view")
    assert partial_occlusion_frames(ordered) == [0, 2]
    pairs = clear_after_partial_transitions(ordered)
    assert [(p["frame_index"], c["frame_index"]) for p, c in pairs] == [(0, 1)]


def test_转场可达按每条轨迹各判一次():
    tracks = {
        "yes": track("visible_occluded", "visible_clear"),
        "no": track("visible_occluded", "visible_occluded"),
        "irrelevant": track("visible_clear", "visible_clear"),
    }
    got = occlusion_recovery_reachable(tracks, kind="partial")
    assert got["yes_reachable"] is True
    assert got["no_reachable"] is True
    # 从没被部分遮挡过的那条不算候选
    assert got["candidate_track_count"] == 2


def test_完全遮挡那一族的可达判定():
    tracks = {
        "recovers": track("fully_occluded", "visible_clear"),
        "stays_hidden": track("fully_occluded", "fully_occluded"),
    }
    got = occlusion_recovery_reachable(tracks, kind="full")
    assert (got["yes_reachable"], got["no_reachable"], got["candidate_track_count"]) == (True, True, 2)


def test_只认两种族别():
    with pytest.raises(ValueError):
        occlusion_recovery_reachable({}, kind="sideways")


# --------------------------------------------------------------------------
# 出题入口拿到的还是同一个结果
# --------------------------------------------------------------------------


def _facts(points, policy=POLICY):
    return {
        "time": {"frame_count": len(points), "frame_rate_hz": RATE},
        "actors": {"a1": {"actor_id": "a1", "root_positions_m": points}},
        "sampling": {"acceptance_policy": {"motion": dict(policy)}},
    }


def test_出题入口和纯函数算出来完全一样():
    from avengine.qa import unified_catalog as catalog

    points = walk(0.05)
    facts = _facts(points)
    policy = catalog._noticeable_motion_policy(facts)
    through_catalog = catalog._noticeable_motion_window(facts, "a1", 0, 30, policy)
    direct = noticeable_motion(points[0:30], RATE, policy, window_frames=[0, 30])
    assert through_catalog == direct


def test_窗口越界时出题入口仍旧报缺读数():
    from avengine.qa import unified_catalog as catalog

    facts = _facts(walk(0.05, frames=5))
    policy = catalog._noticeable_motion_policy(facts)
    assert catalog._noticeable_motion_window(facts, "a1", 0, 99, policy) == {
        "moving": None, "reason": "missing_motion_position_readback"}


def test_距离趋势入口保留窗口和判据两栏():
    from avengine.qa import unified_catalog as catalog

    facts = {
        "time": {"frame_count": 4, "frame_rate_hz": RATE},
        "actors": {"a1": {"actor_id": "a1", "root_positions_m": [[x, 0.0, 0.0] for x in (1.0, 1.2, 1.5, 2.0)]}},
        "listener": {"status": "pass", "positions_m": [[0.0, 0.0, 0.0]] * 4},
        "sampling": {"acceptance_policy": {}},
    }
    record = catalog.distance_trend_during_window(facts, "a1", [0, 4])
    assert record["verdict"] == "farther"
    assert record["window_frames"] == [0, 4]
    assert set(record["criteria"]) >= {"min_net_change_m", "reversal_tolerance_m", "reversal_fraction"}
    # 纯函数拿同一串距离得到同一个判决
    direct = distance_trend(
        record["distance_series_m"],
        min_net_change_m=record["criteria"]["min_net_change_m"],
        reversal_tolerance_m=record["criteria"]["reversal_tolerance_m"],
        reversal_fraction=record["criteria"]["reversal_fraction"],
    )
    assert direct["verdict"] == record["verdict"]
    assert direct["max_counter_trend_m"] == record["max_counter_trend_m"]
