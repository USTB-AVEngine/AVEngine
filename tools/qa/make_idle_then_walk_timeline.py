#!/usr/bin/env python3
"""Idle-then-walk timeline transform (pilot work order items 1.2/1.7 支撑件).

首轮转折轨迹的实现:渲染器逐帧消费 timeline 的 action_id(idle/walk)与
位置,所以"静→走单次转折"是**纯数据构造**——本工具把 CLI 编制的匀速
timeline 变换为:指定角色前 K 帧钉在起点(action_id=idle、相位 0),
第 K 帧起按**原速**沿原路径行走(位置/朝向/步态相位取原 timeline 的
第 0..74−K 帧平移)——速度保持自然带(0.60–0.76 m/s 的设计带不变),
终点落在原路径 (75−K)/75 处,K−1→K 帧位置连续无跳变。另一角色一律
不动。owner 边界:首轮只允许这一种单次转折(idle→walk);walk/idle
多段切换是押后的中量级扩展,本工具拒绝任何多段请求形态。

用法:
  make_idle_then_walk_timeline.py --timeline IN.json --slot source1 \
      --idle-frames K --output OUT.json
库用法:transform_idle_then_walk(doc, slot, k) -> new_doc(不改输入)。

校验(变换前后都做,失败即停):帧数与帧序不变;目标角色 0..K−1 帧
位置全同且为原起点;K 帧位置 == 原第 0 帧位置(连续);K..74 帧逐帧
位移向量 == 原 0..74−K 帧的位移(速度不变);非目标角色逐字节不变;
输出 no-clobber。research_candidate。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

FRAME_COUNT = 75


def _states(doc: dict, slot: str) -> list[dict]:
    out = []
    for fr in doc["frames"]:
        matches = [s for s in fr["actor_states"] if s.get("source_slot_id") == slot]
        if len(matches) != 1:
            raise ValueError(f"frame {fr.get('frame_index')}: expected exactly one "
                             f"state for {slot}, got {len(matches)}")
        out.append(matches[0])
    return out


def transform_idle_then_walk(doc: dict, slot: str, idle_frames: int) -> dict:
    if not 1 <= idle_frames <= FRAME_COUNT - 2:
        raise ValueError(f"idle_frames must be in [1, {FRAME_COUNT - 2}], got {idle_frames}")
    frames = doc.get("frames", [])
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"expected the formal {FRAME_COUNT}-frame timeline, "
                         f"got {len(frames)} frames")
    new_doc = copy.deepcopy(doc)
    src_states = _states(doc, slot)          # 原始(只读)
    dst_states = _states(new_doc, slot)      # 就地改写副本
    carried_keys = ("translation_ue_cm", "yaw_ue_deg", "action_phase",
                    "walk_phase_period_frames")
    for i in range(FRAME_COUNT):
        if i < idle_frames:
            ref = src_states[0]
            dst_states[i]["action_id"] = "idle"
            dst_states[i]["action_phase"] = 0.0
            dst_states[i]["translation_ue_cm"] = list(ref["translation_ue_cm"])
            dst_states[i]["yaw_ue_deg"] = ref["yaw_ue_deg"]
        else:
            ref = src_states[i - idle_frames]
            dst_states[i]["action_id"] = ref["action_id"]
            for key in carried_keys:
                if key in ref:
                    value = ref[key]
                    dst_states[i][key] = list(value) if isinstance(value, list) else value
    # 元数据一致性:顶层 render.walk_start_frame 记录 walk 动作起始帧,
    # 变换后必须与逐帧 action 序列一致(查证:引擎内该键仅由编制器写入、
    # 无下游消费方,但留 0 就是对读者撒谎)。注意原生编制器的
    # walk_start_frame 是"压缩式"(仍走到原终点、速度放大),与本工具的
    # "平移式"(保速度、终点提前)语义不同——本工具只借该键记录事实。
    render = new_doc.get("render")
    if isinstance(render, dict) and "walk_start_frame" in render:
        render["walk_start_frame"] = idle_frames
    _verify(doc, new_doc, slot, idle_frames)
    return new_doc


def _pos(state: dict) -> tuple:
    return tuple(float(v) for v in state["translation_ue_cm"])


def _verify(old: dict, new: dict, slot: str, k: int) -> None:
    old_s, new_s = _states(old, slot), _states(new, slot)
    start = _pos(old_s[0])
    for i in range(k):
        if _pos(new_s[i]) != start or new_s[i]["action_id"] != "idle":
            raise AssertionError(f"idle segment broken at frame {i}")
    if _pos(new_s[k]) != start:
        raise AssertionError("discontinuity at the idle->walk boundary")
    for i in range(k, FRAME_COUNT):
        ref = old_s[i - k]
        if _pos(new_s[i]) != _pos(ref):
            raise AssertionError(f"walk segment mismatch at frame {i}")
    # 速度不变:逐帧位移向量等于原前段位移
    for i in range(k + 1, FRAME_COUNT):
        da = tuple(a - b for a, b in zip(_pos(new_s[i]), _pos(new_s[i - 1])))
        db = tuple(a - b for a, b in zip(_pos(old_s[i - k]), _pos(old_s[i - k - 1])))
        if any(abs(x - y) > 1e-9 for x, y in zip(da, db)):
            raise AssertionError(f"speed changed at frame {i}")
    # 非目标角色逐字节不变
    for fr_old, fr_new in zip(old["frames"], new["frames"]):
        others_old = [s for s in fr_old["actor_states"] if s.get("source_slot_id") != slot]
        others_new = [s for s in fr_new["actor_states"] if s.get("source_slot_id") != slot]
        if json.dumps(others_old, sort_keys=True) != json.dumps(others_new, sort_keys=True):
            raise AssertionError("non-target actor states were modified")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--slot", required=True, choices=["source1", "source2"])
    parser.add_argument("--idle-frames", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if os.path.exists(args.output):
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    doc = json.load(open(args.timeline))
    try:
        new_doc = transform_idle_then_walk(doc, args.slot, args.idle_frames)
    except (ValueError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    with open(args.output, "w") as fp:
        json.dump(new_doc, fp, ensure_ascii=False, indent=2)
    old_end = _states(doc, args.slot)[-1]["translation_ue_cm"]
    new_end = _states(new_doc, args.slot)[-1]["translation_ue_cm"]
    print(f"ok slot={args.slot} idle_frames={args.idle_frames} "
          f"end_moved_from={old_end} to={new_end} out={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
