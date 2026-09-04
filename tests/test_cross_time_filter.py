"""Unit tests for the cross-time sampling filter (pilot item 1.7).

一个手工几何的"全通过"点位作基准,再逐项破坏它构造阳性对照:片尾角距
不足、锚后角位移不足、锚时方位分离不足、静默段动静相同、首叫同带、
锚不是最后事件、片尾出视锥——每一种都必须被对应的卡拒出。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from filter_cross_time_points import (  # noqa: E402
    FRAME_COUNT,
    PointView,
    azimuth_deg,
    circ_diff,
    evaluate_point,
    main,
)

PARAMS = {"THETA_FULL": 10.0, "THETA_HALF": 15.0, "T_HALF": 0.9,
          "TAIL_MIN_S": 1.5, "MIN_AZIMUTH_SEP": 25.0,
          "MIN_DIST_CHANGE_CM": 30.0, "MIN_CARD7_FRAMES": 8,
          "BANDS": [0.0, 1.25, 2.5, 3.75, 5.0]}

S1_POS = (300.0, 150.0)                  # 静止者
S2_START, S2_END = (300.0, -100.0), (150.0, -140.0)   # 移动者(锚定角色)


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def make_timeline(s1_pos=S1_POS, s2_start=S2_START, s2_end=S2_END,
                  s2_action="walk"):
    frames = []
    for i in range(FRAME_COUNT):
        t = i / (FRAME_COUNT - 1)
        p2 = _lerp(s2_start, s2_end, t)
        frames.append({
            "frame_index": i,
            "camera": {"translation_ue_cm": [0.0, 0.0, 147.0], "yaw_ue_deg": 0.0},
            "actor_states": [
                {"source_slot_id": "source1", "action_id": "idle",
                 "translation_ue_cm": [s1_pos[0], s1_pos[1], 27.0]},
                {"source_slot_id": "source2", "action_id": s2_action,
                 "translation_ue_cm": [p2[0], p2[1], 27.0]},
            ],
        })
    return {"render": {"hfov_degrees": 105.0}, "frames": frames}


def make_program_plan(event_list):
    """event_list: [(slot, start_sample)];事件长 4800。"""
    eps = ["ep_s1", "ep_s2"]
    events = []
    for slot, start in event_list:
        events.append({"source_endpoint_id": eps[0] if slot == "source1" else eps[1],
                       "start_sample": start, "end_sample_exclusive": start + 4800})
    program = {"candidate_source_endpoint_ids": eps, "events": events}
    a_slot, a_start = event_list[-1]
    plan = {"anchor_slot": a_slot, "anchor_start_sample": a_start,
            "anchor_end_sample": a_start + 4800,
            "tail_silence_samples": 80000 - (a_start + 4800)}
    return program, plan


BASE_EVENTS = [("source1", 8000), ("source2", 24000),
               ("source1", 36000), ("source2", 48000)]


def _eval(timeline=None, events=None, plan_override=None, params=None):
    program, plan = make_program_plan(events or BASE_EVENTS)
    if plan_override:
        plan.update(plan_override)
    view = PointView(timeline or make_timeline(), program, plan)
    return evaluate_point(view, params or PARAMS)


def test_baseline_point_admits_everywhere():
    res = _eval()
    for cardk in ("card1", "card5R", "card6R", "card7", "card8", "card9"):
        assert res[cardk]["admit"], (cardk, res[cardk]["reasons"])
    assert res["card6R"]["target_moving"] is True   # 锚定者=移动者
    assert res["card1"]["ending_gap_deg"] > 2 * PARAMS["THETA_HALF"]


def test_card1_rejects_small_ending_gap():
    # 移动者终点凑到静止者同侧近角
    res = _eval(timeline=make_timeline(s2_start=(320.0, 120.0), s2_end=(260.0, 170.0)))
    assert not res["card1"]["admit"]
    assert any("ending angular gap" in r for r in res["card1"]["reasons"])


def test_card1_rejects_static_anchor_target():
    # 锚定角色(source2)原地不动 → 角位移不足
    res = _eval(timeline=make_timeline(s2_start=(300.0, -100.0),
                                       s2_end=(300.0, -100.0), s2_action="idle"))
    assert not res["card1"]["admit"]
    assert any("angular travel" in r for r in res["card1"]["reasons"])


def test_common_rejects_small_anchor_separation():
    # 两角色锚定帧方位贴近 → C2 波及全部卡
    res = _eval(timeline=make_timeline(s1_pos=(300.0, -90.0)))
    for cardk in ("card1", "card8", "card9"):
        assert not res[cardk]["admit"]
        assert any(r.startswith("C2") for r in res[cardk]["reasons"])


def test_card6r_rejects_same_motion_state():
    # 静默段两角色都不动(s2 全程 idle 且不移动;锚定者仍是 s2)
    tl = make_timeline(s2_start=(300.0, -100.0), s2_end=(300.0, -100.0),
                       s2_action="idle")
    res = _eval(timeline=tl)
    assert not res["card6R"]["admit"]
    assert any("same motion state" in r for r in res["card6R"]["reasons"])


def test_card8_rejects_same_band_and_small_gap():
    events = [("source1", 8000), ("source2", 12800),   # 0.5s 与 0.8s:同带0,间隔0.3s
              ("source1", 30000), ("source2", 48000)]
    res = _eval(events=events)
    assert not res["card8"]["admit"]
    joined = " ".join(res["card8"]["reasons"])
    assert "band" in joined and "gap" in joined


def test_c1_positive_control_anchor_not_last():
    # 篡改 plan:把锚标成第 2 个事件 → C1 纵深复核必须抓
    res = _eval(plan_override={"anchor_start_sample": 24000,
                               "anchor_end_sample": 28800,
                               "anchor_slot": "source2",
                               "tail_silence_samples": 80000 - 28800})
    assert any(r.startswith("C1") for r in res["card1"]["reasons"])


def test_fov_rejection_at_final_frame():
    # 移动者走出视锥(方位 > 52.5°)
    res = _eval(timeline=make_timeline(s2_end=(60.0, -200.0)))
    assert not res["card1"]["admit"]
    assert any("outside FOV" in r for r in res["card1"]["reasons"])


def test_azimuth_sign_convention_matches_side_of():
    # 右为正:sin(rel)>0 ⇔ 在产 side_of 的叉积>0(right)
    import math
    cam, yaw = (0.0, 0.0), -145.0
    p = (-150.0, 30.0)
    rel = azimuth_deg(cam, yaw, p)
    fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    c = fx * (p[1] - cam[1]) - fy * (p[0] - cam[0])
    assert (rel > 0) == (c > 0)
    assert circ_diff(-179.0, 179.0) == 2.0


def test_cli_end_to_end_and_no_clobber(tmp_path):
    inputs = tmp_path / "inputs"
    programs = tmp_path / "programs"
    programs.mkdir()
    for pid in ("p001", "p002"):
        d = inputs / pid
        d.mkdir(parents=True)
        (d / "spec.json").write_text(json.dumps({"program_id": f"prog_{pid}"}))
        (d / "timeline.json").write_text(json.dumps(make_timeline()))
        program, plan = make_program_plan(BASE_EVENTS)
        program = dict(program, program_id=f"prog_{pid}")
        (programs / f"prog_{pid}.json").write_text(json.dumps(program))
        (programs / f"prog_{pid}.plan.json").write_text(json.dumps(plan))
    params_p = tmp_path / "params.json"
    params_p.write_text(json.dumps(PARAMS))
    out = tmp_path / "filter.json"
    assert main(["--inputs-root", str(inputs), "--programs-dir", str(programs),
                 "--params", str(params_p), "--out", str(out),
                 "--historical-reproduction"]) == 0
    doc = json.loads(out.read_text())
    assert doc["counts"]["points"] == 2
    assert doc["counts"]["admits"]["card1"] == 2
    assert doc["card6R_answer_counts"] == {"moving": 2}
    assert main(["--inputs-root", str(inputs), "--programs-dir", str(programs),
                 "--params", str(params_p), "--out", str(out),
                 "--historical-reproduction"]) == 2  # no-clobber
    bad = tmp_path / "bad_params.json"
    bad.write_text(json.dumps({"THETA_FULL": 10}))
    assert main(["--inputs-root", str(inputs), "--programs-dir", str(programs),
                 "--params", str(bad), "--out", str(tmp_path / "o2.json"),
                 "--historical-reproduction"]) == 2


def test_card1_end_gap_threshold_is_overridable():
    # END_GAP_MIN 覆盖:MCQ 口径(25°)下中等角距应过,开放版口径(60°)拒
    tl = make_timeline(s1_pos=(300.0, 150.0), s2_end=(240.0, -20.0))
    res_open = _eval(timeline=tl)
    res_mcq = _eval(timeline=tl, params={**PARAMS, "THETA_HALF": 30.0,
                                         "END_GAP_MIN": 25.0})
    gap = res_open["card1"]["ending_gap_deg"]
    assert 25.0 < gap <= 60.0, gap
    assert not _eval(timeline=tl, params={**PARAMS, "THETA_HALF": 30.0})["card1"]["admit"]
    assert res_mcq["card1"]["admit"] or any(
        not r.startswith("card1: ending angular gap")
        for r in res_mcq["card1"]["reasons"])
