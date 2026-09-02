#!/usr/bin/env python3
"""Cross-time sampling filter (pilot work order item 1.7).

设计批的总闸:逐点读 timeline(逐帧位置+相机)与 program 伴生 plan
(锚事件、尾窗、事件表),把全案 2.3 的准入条件与逐卡专用约束落成
机器判定,输出逐点×逐卡的 admit/reject(带原因与实测量)和批级均衡
报告。**分层诚实声明**:本闸做的是设计期的**程序复核+几何预检**
(方位分离、视锥内)——最终"提问帧可观察"以渲染后的像素判定为准
(工单 1.8),物理分类器与模型探针在认证阶段;本闸的 admit 只是
"可送渲染",不是题目准入。

方位角口径:相对相机朝向的环形角,**右为正**——与在产
generate_qa_v2_questions.py 的 side_of 叉积口径一致(sin(rel)>0 ⇔
叉积>0 ⇔ right),题面"右为正"的约定同源。视锥预检:|方位| ≤
hfov/2(从 timeline.render.hfov_degrees 读,不硬编码)。

逐卡检查(全部参数显式,快照进输出):
  通用   C1 锚后静默复核:program 事件表里锚事件必须是最后一个,尾窗
           ≥ TAIL_MIN_S(program 生成器已保证,这里是纵深复核);
         C2 锚时可绑定:锚定帧两角色方位分离 ≥ MIN_AZIMUTH_SEP 且
           双双在视锥内(几何预检)。
  card1  片尾帧两角色环形角距 > 2×THETA_HALF;目标锚定帧→片尾帧的
         角位移 > THETA_FULL;片尾目标在视锥内。
  card5R 目标锚定帧→片尾的相机距离变化 ≥ MIN_DIST_CHANGE_CM。
  card6R 静默段(锚事件结束帧→片尾)两角色动静互异(都以该段
         action_id 序列判:含 walk 帧=动),相同即拒(无区分力)。
  card7  逐帧可绑定性:输出"恰好一只在叫且两角色方位分离达标且双双
         在视锥内"的帧集合;可用帧数 ≥ MIN_CARD7_FRAMES 才 admit
         (查询时刻由出题器在该集合内选)。
  card8  两角色首叫落不同 MCQ 时间带(BANDS)且间隔 > T_HALF;各自
         首叫帧上两角色方位分离达标(绑定前提)。
  card9  首事件帧上两角色方位分离达标且双双在视锥内。

批级报告:各卡 admit 计数与拒因分布;card1 的"锚时目标方位左右 ×
片尾扇区"联合计数(条件均衡的证据,偏斜由设计器迭代,不在此硬挡);
card6R 动静答案分布。输出 no-clobber。research_candidate。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

FRAME_COUNT = 75
SAMPLE_COUNT = 80000


def azimuth_deg(cam_pos, cam_yaw_deg, p) -> float:
    """相对相机朝向的环形角,右为正(与 side_of 叉积口径同源)。"""
    dx, dy = float(p[0]) - float(cam_pos[0]), float(p[1]) - float(cam_pos[1])
    bearing = math.degrees(math.atan2(dy, dx))
    rel = (bearing - float(cam_yaw_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if rel == -180.0 else rel


def circ_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def sample_to_frame(sample: int) -> int:
    return min(FRAME_COUNT - 1, int(sample * FRAME_COUNT / SAMPLE_COUNT))


class PointView:
    """一个采样点的逐帧几何与程序事实。"""

    def __init__(self, timeline: dict, program: dict, plan: dict):
        self.plan = plan
        self.events = sorted(program["events"], key=lambda e: e["start_sample"])
        ep_order = program["candidate_source_endpoint_ids"]
        self.ep_to_slot = {ep_order[0]: "source1", ep_order[1]: "source2"}
        self.hfov = float(timeline.get("render", {}).get("hfov_degrees", 105.0))
        self.az = {"source1": [], "source2": []}
        self.act = {"source1": [], "source2": []}
        self.dist = {"source1": [], "source2": []}
        for fr in timeline["frames"]:
            cam = fr["camera"]
            for st in fr["actor_states"]:
                slot = st["source_slot_id"]
                if slot not in self.az:
                    continue
                self.az[slot].append(azimuth_deg(cam["translation_ue_cm"],
                                                 cam["yaw_ue_deg"],
                                                 st["translation_ue_cm"]))
                self.act[slot].append(st["action_id"])
                dx = st["translation_ue_cm"][0] - cam["translation_ue_cm"][0]
                dy = st["translation_ue_cm"][1] - cam["translation_ue_cm"][1]
                self.dist[slot].append(math.hypot(dx, dy))
        n = len(timeline["frames"])
        if n != FRAME_COUNT or any(len(self.az[s]) != n for s in self.az):
            raise ValueError(f"timeline must carry {FRAME_COUNT} frames per slot")

    def in_fov(self, slot: str, frame: int) -> bool:
        return abs(self.az[slot][frame]) <= self.hfov / 2.0

    def calling_slots_at(self, frame: int) -> set:
        out = set()
        for ev in self.events:
            if sample_to_frame(ev["start_sample"]) <= frame < sample_to_frame(
                    ev["end_sample_exclusive"]) + 1:
                out.add(self.ep_to_slot[ev["source_endpoint_id"]])
        return out

    def first_call_sample(self, slot: str):
        for ev in self.events:
            if self.ep_to_slot[ev["source_endpoint_id"]] == slot:
                return ev["start_sample"]
        return None

    def moving_in(self, slot: str, frame_a: int, frame_b: int) -> bool:
        return any(a == "walk" for a in self.act[slot][frame_a:frame_b + 1])


def _band_of(t_s: float, bands) -> int | None:
    if t_s < bands[0] or t_s > bands[-1]:
        return None
    for i in range(len(bands) - 1):
        if bands[i] <= t_s < bands[i + 1]:
            return i
    return len(bands) - 2


def evaluate_point(view: PointView, params: dict) -> dict:
    plan = view.plan
    anchor_slot = plan["anchor_slot"]
    other_slot = "source2" if anchor_slot == "source1" else "source1"
    anchor_frame = sample_to_frame(plan["anchor_start_sample"])
    anchor_end_frame = min(FRAME_COUNT - 1,
                           sample_to_frame(plan["anchor_end_sample"]) + 1)
    last = FRAME_COUNT - 1
    out: dict = {"anchor_slot": anchor_slot, "anchor_frame": anchor_frame}

    common: list[str] = []
    # C1 锚后静默复核
    if plan["anchor_start_sample"] != view.events[-1]["start_sample"]:
        common.append("C1: anchor is not the last program event")
    tail_s = plan["tail_silence_samples"] / 16000.0
    if tail_s < params["TAIL_MIN_S"]:
        common.append(f"C1: tail silence {tail_s:.2f}s < {params['TAIL_MIN_S']}s")
    # C2 锚时可绑定
    sep_anchor = circ_diff(view.az["source1"][anchor_frame],
                           view.az["source2"][anchor_frame])
    if sep_anchor < params["MIN_AZIMUTH_SEP"]:
        common.append(f"C2: anchor-frame separation {sep_anchor:.1f} < "
                      f"{params['MIN_AZIMUTH_SEP']}")
    for slot in ("source1", "source2"):
        if not view.in_fov(slot, anchor_frame):
            common.append(f"C2: {slot} outside FOV at the anchor frame")
    out["anchor_separation_deg"] = round(sep_anchor, 1)

    def card(reasons_extra):
        reasons = list(common) + reasons_extra
        return {"admit": not reasons, "reasons": reasons}

    # card1(END_GAP_MIN 可覆盖:开放版口径=2×THETA_HALF;MCQ 口径可放宽
    # ——占位带宽下开放版口径在本房间几何近乎不可产,见设计批 manifest 标注)
    r = []
    end_gap = circ_diff(view.az["source1"][last], view.az["source2"][last])
    end_gap_min = params.get("END_GAP_MIN", 2 * params["THETA_HALF"])
    if end_gap <= end_gap_min:
        r.append(f"card1: ending angular gap {end_gap:.1f} <= {end_gap_min}")
    move = circ_diff(view.az[anchor_slot][anchor_frame], view.az[anchor_slot][last])
    if move <= params["THETA_FULL"]:
        r.append(f"card1: target angular travel {move:.1f} <= THETA_FULL")
    if not view.in_fov(anchor_slot, last):
        r.append("card1: target outside FOV at the final frame")
    out["card1"] = card(r)
    out["card1"].update(ending_gap_deg=round(end_gap, 1),
                        target_travel_deg=round(move, 1),
                        ending_azimuth_deg=round(view.az[anchor_slot][last], 1))

    # card5R
    r = []
    dist_change = abs(view.dist[anchor_slot][last] - view.dist[anchor_slot][anchor_frame])
    if dist_change < params["MIN_DIST_CHANGE_CM"]:
        r.append(f"card5R: distance change {dist_change:.0f}cm < "
                 f"{params['MIN_DIST_CHANGE_CM']}cm")
    out["card5R"] = card(r)
    out["card5R"]["dist_change_cm"] = round(dist_change, 0)

    # card6R
    r = []
    m_target = view.moving_in(anchor_slot, anchor_end_frame, last)
    m_other = view.moving_in(other_slot, anchor_end_frame, last)
    if m_target == m_other:
        r.append("card6R: both actors share the same motion state in the silent tail")
    out["card6R"] = card(r)
    out["card6R"].update(target_moving=m_target, other_moving=m_other)

    # card7:逐帧可绑定集合
    usable = []
    for f in range(FRAME_COUNT):
        calling = view.calling_slots_at(f)
        if len(calling) != 1:
            continue
        sep = circ_diff(view.az["source1"][f], view.az["source2"][f])
        if sep >= params["MIN_AZIMUTH_SEP"] and view.in_fov("source1", f) \
                and view.in_fov("source2", f):
            usable.append(f)
    r = []
    if len(usable) < params["MIN_CARD7_FRAMES"]:
        r.append(f"card7: only {len(usable)} bindable exactly-one-calling frames "
                 f"< {params['MIN_CARD7_FRAMES']}")
    out["card7"] = card(r)
    out["card7"]["usable_frames"] = usable[:40]

    # card8
    r = []
    f1, f2 = view.first_call_sample("source1"), view.first_call_sample("source2")
    if f1 is None or f2 is None:
        r.append("card8: a slot never calls")
    else:
        t1, t2 = f1 / 16000.0, f2 / 16000.0
        b1, b2 = _band_of(t1, params["BANDS"]), _band_of(t2, params["BANDS"])
        if b1 == b2:
            r.append(f"card8: both first calls in band {b1}")
        if abs(t1 - t2) <= params["T_HALF"]:
            r.append(f"card8: first-call gap {abs(t1 - t2):.2f}s <= T_HALF")
        for slot, sample in (("source1", f1), ("source2", f2)):
            fr = sample_to_frame(sample)
            sep = circ_diff(view.az["source1"][fr], view.az["source2"][fr])
            if sep < params["MIN_AZIMUTH_SEP"]:
                r.append(f"card8: separation {sep:.1f} at {slot} first call")
    out["card8"] = card(r)

    # card9:首事件帧绑定
    r = []
    first_frame = sample_to_frame(view.events[0]["start_sample"])
    sep = circ_diff(view.az["source1"][first_frame], view.az["source2"][first_frame])
    if sep < params["MIN_AZIMUTH_SEP"]:
        r.append(f"card9: first-event separation {sep:.1f} < MIN_AZIMUTH_SEP")
    for slot in ("source1", "source2"):
        if not view.in_fov(slot, first_frame):
            r.append(f"card9: {slot} outside FOV at the first event")
    out["card9"] = card(r)
    return out


REQUIRED_PARAMS = ("THETA_FULL", "THETA_HALF", "T_HALF", "TAIL_MIN_S",
                   "MIN_AZIMUTH_SEP", "MIN_DIST_CHANGE_CM", "MIN_CARD7_FRAMES",
                   "BANDS")


HISTORICAL_NOTICE = (
    "this filter is historical: it predates the explicit T_FULL first-call "
    "chain and the room-centric scene batch (design_qa_v3_scene_batch.py). It "
    "only runs with --historical-reproduction, for reproducing old products.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs-root", required=True,
                        help="设计批根:每点一个目录,含 timeline.json 与 spec.json")
    parser.add_argument("--programs-dir", required=True,
                        help="build_qa_v3_programs 的输出目录(program+plan)")
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--historical-reproduction", action="store_true",
                        help=HISTORICAL_NOTICE)
    args = parser.parse_args(argv)

    if not args.historical_reproduction:
        print(f"refusing to run: {HISTORICAL_NOTICE}", file=sys.stderr)
        return 2
    if os.path.exists(args.out):
        print(f"refusing to overwrite existing output: {args.out}", file=sys.stderr)
        return 2
    params = json.load(open(args.params))
    missing = [k for k in REQUIRED_PARAMS if k not in params]
    if missing:
        print(f"params missing explicit keys: {missing}", file=sys.stderr)
        return 2

    points = sorted(d for d in os.listdir(args.inputs_root)
                    if os.path.isdir(os.path.join(args.inputs_root, d)))
    results, errors = {}, []
    admit_counts = defaultdict(int)
    reject_reasons = defaultdict(Counter)
    card1_joint = Counter()
    card6_answers = Counter()
    for pid in points:
        pdir = os.path.join(args.inputs_root, pid)
        try:
            spec = json.load(open(os.path.join(pdir, "spec.json")))
            timeline = json.load(open(os.path.join(pdir, "timeline.json")))
            prog_path = os.path.join(args.programs_dir, spec["program_id"] + ".json")
            plan_path = os.path.join(args.programs_dir, spec["program_id"] + ".plan.json")
            program = json.load(open(prog_path))
            plan = json.load(open(plan_path))
            view = PointView(timeline, program, plan)
            res = evaluate_point(view, params)
        except Exception as exc:
            errors.append({"point_id": pid, "error": repr(exc)})
            continue
        results[pid] = res
        for cardk in ("card1", "card5R", "card6R", "card7", "card8", "card9"):
            if res[cardk]["admit"]:
                admit_counts[cardk] += 1
            else:
                for reason in res[cardk]["reasons"]:
                    reject_reasons[cardk][reason.split(":")[0]] += 1
        if res["card1"]["admit"]:
            anchor_side = "R" if res["card1"]["ending_azimuth_deg"] >= 0 else "L"
            anchor_dir = "R" if view.az[res["anchor_slot"]][res["anchor_frame"]] >= 0 else "L"
            card1_joint[f"anchor_{anchor_dir}/end_{anchor_side}"] += 1
        if res["card6R"]["admit"]:
            card6_answers["moving" if res["card6R"]["target_moving"] else "still"] += 1

    if errors:
        print(f"FAIL: {len(errors)} point(s) unreadable; first: {errors[0]}",
              file=sys.stderr)
        return 1

    payload = {
        "schema": "avengine_cross_time_filter_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "parameters": params,
        "counts": {"points": len(points),
                   "admits": dict(admit_counts)},
        "reject_reasons": {k: dict(v) for k, v in reject_reasons.items()},
        "card1_anchor_end_joint": dict(card1_joint),
        "card6R_answer_counts": dict(card6_answers),
        "note": ("geometric precheck + program recheck only; pixel-level "
                 "observability (1.8) and probe certification decide final admission"),
        "results": results,
    }
    with open(args.out, "w") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=1)
    print(f"points={len(points)} admits={dict(admit_counts)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
