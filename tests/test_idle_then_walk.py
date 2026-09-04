"""Unit tests for the idle-then-walk timeline transform (items 1.2/1.7 支撑件).

阳性对照:合法的任意帧数都按声明时钟变换;越界 K、缺角色帧仍必须拒;
变换自带的验证器
必须能抓出人为注入的边界跳变(直接调用 _verify 对坏文档断言)。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from make_idle_then_walk_timeline import (  # noqa: E402
    FRAME_COUNT,
    _verify,
    main,
    resample_route_samples,
    transform_idle_then_walk,
    transform_to_solved_routes,
)


def _mini_timeline(n=FRAME_COUNT):
    frames = []
    for i in range(n):
        frames.append({
            "frame_index": i,
            "pts_ticks": i * 100,
            "camera": {"yaw": -145.0},
            "actor_states": [
                {"source_slot_id": "source1", "actor_id": "human_1",
                 "action_id": "walk", "action_phase": i / n,
                 "translation_ue_cm": [-100.0 + 2.0 * i, 50.0 + 1.0 * i, 27.1],
                 "yaw_ue_deg": 10.0 + i, "walk_phase_period_frames": 16},
                {"source_slot_id": "source2", "actor_id": "human_2",
                 "action_id": "idle", "action_phase": 0.0,
                 "translation_ue_cm": [-200.0, 80.0, 27.1],
                 "yaw_ue_deg": -30.0, "walk_phase_period_frames": 16},
            ],
        })
    return {"kind": "test", "render": {"frame_count": n, "walk_start_frame": 0},
            "frames": frames}


def _pos(doc, slot, i):
    for s in doc["frames"][i]["actor_states"]:
        if s["source_slot_id"] == slot:
            return tuple(s["translation_ue_cm"]), s["action_id"]
    raise KeyError(slot)


def test_transform_pins_idle_then_walks_at_original_speed():
    doc = _mini_timeline()
    k = 20
    out = transform_idle_then_walk(doc, "source1", k)
    start, _ = _pos(doc, "source1", 0)
    for i in range(k):
        p, act = _pos(out, "source1", i)
        assert p == start and act == "idle"
    p_k, act_k = _pos(out, "source1", k)
    assert p_k == start and act_k == "walk"          # 边界连续
    p_last, _ = _pos(out, "source1", FRAME_COUNT - 1)
    orig_mid, _ = _pos(doc, "source1", FRAME_COUNT - 1 - k)
    assert p_last == orig_mid                          # 终点=原路径中途点
    assert out["render"]["walk_start_frame"] == k      # 顶层元数据同步事实
    # 逐帧位移与原速一致
    for i in range(k + 1, FRAME_COUNT):
        a1, _ = _pos(out, "source1", i)
        a0, _ = _pos(out, "source1", i - 1)
        b1, _ = _pos(doc, "source1", i - k)
        b0, _ = _pos(doc, "source1", i - k - 1)
        assert tuple(x - y for x, y in zip(a1, a0)) == \
               tuple(x - y for x, y in zip(b1, b0))


def test_other_actor_untouched_and_input_not_mutated():
    doc = _mini_timeline()
    snapshot = json.dumps(doc, sort_keys=True)
    out = transform_idle_then_walk(doc, "source1", 15)
    assert json.dumps(doc, sort_keys=True) == snapshot  # 输入只读
    for i in range(FRAME_COUNT):
        assert _pos(out, "source2", i) == _pos(doc, "source2", i)


def test_generic_frame_count_and_positive_control_rejects_bad_inputs():
    sixty = _mini_timeline(n=60)
    adapted = transform_idle_then_walk(sixty, "source1", 10)
    assert len(adapted["frames"]) == 60
    assert _pos(adapted, "source1", 10)[0] == _pos(sixty, "source1", 0)[0]
    assert _pos(adapted, "source1", 59)[0] == _pos(sixty, "source1", 49)[0]

    with pytest.raises(ValueError):
        transform_idle_then_walk(_mini_timeline(), "source1", 0)       # K 越界
    with pytest.raises(ValueError):
        transform_idle_then_walk(_mini_timeline(), "source1", FRAME_COUNT - 1)
    broken = _mini_timeline()
    broken["frames"][3]["actor_states"] = [broken["frames"][3]["actor_states"][1]]
    with pytest.raises(ValueError):
        transform_idle_then_walk(broken, "source1", 10)                # 缺角色帧


def test_positive_control_verifier_catches_injected_discontinuity():
    doc = _mini_timeline()
    out = transform_idle_then_walk(doc, "source1", 20)
    bad = copy.deepcopy(out)
    for s in bad["frames"][20]["actor_states"]:
        if s["source_slot_id"] == "source1":
            s["translation_ue_cm"] = [999.0, 999.0, 27.1]  # 注入边界跳变
    with pytest.raises(AssertionError):
        _verify(doc, bad, "source1", 20)


def test_cli_end_to_end_and_no_clobber(tmp_path):
    src = tmp_path / "tl.json"
    src.write_text(json.dumps(_mini_timeline()))
    out = tmp_path / "tl_idle.json"
    assert main(["--timeline", str(src), "--slot", "source1",
                 "--idle-frames", "20", "--output", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["frames"][5]["actor_states"][0]["action_id"] == "idle"
    assert main(["--timeline", str(src), "--slot", "source1",
                 "--idle-frames", "20", "--output", str(out)]) == 2


def test_solved_route_transform_preserves_pause_and_updates_actions():
    doc = _mini_timeline()
    route1 = [(float(frame), 0.0) for frame in range(FRAME_COUNT)]
    route2 = [(0.0, float(frame)) for frame in range(FRAME_COUNT)]
    for frame in range(20, 31):
        route1[frame] = route1[20]
    out = transform_to_solved_routes(
        doc, {"source1": route1, "source2": route2})
    for frame in range(FRAME_COUNT):
        pos1, action1 = _pos(out, "source1", frame)
        pos2, _ = _pos(out, "source2", frame)
        assert pos1[:2] == route1[frame]
        assert pos2[:2] == route2[frame]
        if 21 <= frame <= 29:
            assert action1 == "idle"
    assert _pos(out, "source1", 19)[1] == "walk"
    assert _pos(out, "source1", 31)[1] == "walk"

def test_solved_routes_resample_legacy_75_points_to_timeline_clock():
    doc = _mini_timeline(n=150)
    source_route = [(float(frame), float(frame * 2)) for frame in range(FRAME_COUNT)]
    out = transform_to_solved_routes(
        doc,
        {"source1": source_route, "source2": source_route},
    )
    assert len(out["frames"]) == 150
    assert resample_route_samples(source_route, 150)[0] == [0.0, 0.0]
    assert resample_route_samples(source_route, 150)[-1] == [74.0, 148.0]
    state = out["frames"][-1]["actor_states"][0]
    assert state["translation_ue_cm"][:2] == [74.0, 148.0]
    assert state["route_waypoint_count"] == 150
    assert state["route_geometry"] == "solver_authoritative_150_frame"
