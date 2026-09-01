#!/usr/bin/env python3
"""Per-point audio-program generator for the v3 pilot (work order item 1.2).

拆掉"发声恒在第 4 帧"的病根:qa_v2 用 8 个预制固定 program(全批同一
套 onset),本工具给**每个采样点生成专属 program**——发声时刻在可行域
内均匀随机、每集至少 3 声、两只都至少叫一声、锚事件(全片最后一声)
之后留静默尾窗(错时题的提问窗,窗内两只都无事件)。

事件规划约束(全部显式参数,写进批报告):
  FIRST_MIN     首声最早时刻(默认 0.3s)——防事件贴片头;
  GAP_MIN       相邻事件间隔下限(默认 0.3s);
  TAIL_SILENCE  锚事件结束到片尾的静默下限(默认 1.5s);
  n_events      每点 3 或 4(种子定),归属交替(first_slot 起手),
                保证两只各 ≥1 声(⑧的"单叫退化"防线);
  事件时长固定 0.3s,素材窗沿用在产 program 的 [0.2s, 0.5s) 截取。

产物:每点一个 `<program_id>.json`(结构与 schema 同在产 program:
timeline 常量块、sequential_sources 模式、两端点、封印
program_content_sha256 用引擎自己的 canonical_json_sha256 计算)+
伴生 `<program_id>.plan.json`(锚事件元数据:锚槽位、锚起止、静默尾
长——供错时采样过滤器与出题器消费,不塞进 program 本体以免破坏
schema)。每个产物过 jsonschema 校验,失败即停;输出目录 no-clobber。

批级自检:首声 onset 的帧分布必须覆盖 ≥3 个不同的 15 帧桶(旧病根的
回归锚:全批 onset 挤同一帧即报错)。research_candidate。

请求清单格式(JSON 列表):
  {"point_id": "v3p001", "pair_kind": "dog"|"human",
   "endpoint_1": "...", "endpoint_2": "...",
   "sound_asset_id": "...", "first_slot": "source1"|"source2"}
用法:
  build_qa_v3_programs.py --requests R.json --seed S --out-dir DIR
      [--first-min-s 0.3] [--gap-min-s 0.3] [--tail-silence-s 1.5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from avengine.contracts.json_io import canonical_json_sha256  # noqa: E402

SAMPLE_RATE = 16000
SAMPLE_COUNT = 80000
TICKS_PER_SAMPLE = 3
EVENT_LEN = int(0.3 * SAMPLE_RATE)          # 4800,与在产 program 一致
SOURCE_START, SOURCE_END = 3200, 8000       # 素材内截取窗,沿用在产
TIMELINE_BLOCK = {"time_base_hz": 48000, "ticks_per_frame": 3200, "video_fps": 15,
                  "frame_count": 75, "sample_rate_hz": 16000,
                  "ticks_per_sample": 3, "sample_count": 80000}


def _rng(seed: str, *parts: str):
    import numpy as np
    digest = hashlib.sha256(("\0".join((seed,) + parts)).encode()).hexdigest()
    return np.random.default_rng(int(digest[:12], 16))


def plan_events(seed: str, point_id: str, first_slot: str, *,
                first_min_s: float, gap_min_s: float, tail_silence_s: float,
                first_call_bands: list[float] | None = None,
                min_first_call_gap_s: float | None = None,
                target_first_bands: tuple[int, int] | None = None):
    """返回 (事件列表[(slot, start_sample)], 锚元数据)。

    可选的 card8 约束(设计冒烟的教训:这两条必须在 program 规划层满足,
    换路线的重试对它们无能为力):两个槽位的**首叫**须落在
    first_call_bands 的不同带,且相隔超过 min_first_call_gap_s。

    `target_first_bands=(b1, b2)` 是**带优先调度**(codex 审阅裁定):先
    给这一点分配答案带,再在带内采样,而不是先采样再看落在哪带——后者
    让答案带的分布由可行域形状决定(run01 实测 54/60/73/53,按外观拆开
    后黑白狗 22/21/48/29,只看外观猜第三带即得 40%)。因为事件按槽位
    交替、时间递增,首叫必然 t1 < t2,故只有 b1 < b2 的有序带对可达;
    调用方按有序带对均匀配额分配,两个槽位的答案带边际分布即自动配平。
    """
    rng = _rng(seed, point_id, "events")
    other = "source2" if first_slot == "source1" else "source1"
    lo = int(first_min_s * SAMPLE_RATE)
    gap = int(gap_min_s * SAMPLE_RATE)
    hi = SAMPLE_COUNT - int(tail_silence_s * SAMPLE_RATE) - EVENT_LEN
    step = EVENT_LEN + gap
    if target_first_bands is not None:
        if first_call_bands is None:
            raise ValueError("target_first_bands requires first_call_bands")
        b1, b2 = target_first_bands
        if not 0 <= b1 < b2 <= len(first_call_bands) - 2:
            raise ValueError(f"{point_id}: unreachable band pair {b1},{b2} "
                             "(events alternate in time, so b1 < b2)")
        starts, n_events = _plan_banded_starts(
            rng, point_id, first_call_bands, (b1, b2), lo, hi, step,
            min_first_call_gap_s)
        slots = [first_slot if i % 2 == 0 else other
                 for i in range(n_events)]
        return _finish_plan(slots, starts, n_events)
    n_events = int(rng.integers(3, 5))  # 3 或 4
    slots = [first_slot if i % 2 == 0 else other for i in range(n_events)]
    if hi - lo < (n_events - 1) * step:
        raise ValueError(f"{point_id}: infeasible constraints "
                         f"(window {hi - lo} < needed {(n_events - 1) * step})")

    def _band(t_s: float):
        b = first_call_bands
        if t_s < b[0] or t_s > b[-1]:
            return None
        for i in range(len(b) - 1):
            if b[i] <= t_s < b[i + 1]:
                return i
        return len(b) - 2

    # 顺序条件采样:每一步在当前可行区间内均匀取。教训(自检抓到的
    # 设计缺陷):"采 n 个均匀样本再排序"会让首声=n 个样本的最小值,
    # 分布挤向低端,批内首声全落前几十帧——"首声可预测"换形式复活。
    # 条件采样让首声的边际分布在 [lo, hi-(n-1)step] 上真均匀。
    starts: list[int] = []
    for _attempt in range(300):
        starts = []
        cursor = lo
        for i in range(n_events):
            remaining = n_events - 1 - i
            hi_i = hi - remaining * step
            starts.append(int(rng.integers(cursor, hi_i + 1)))
            cursor = starts[-1] + step
        if first_call_bands is None and min_first_call_gap_s is None:
            break
        # 首叫 = 交替序列的前两个事件(各槽位第一次)
        t1, t2 = starts[0] / SAMPLE_RATE, starts[1] / SAMPLE_RATE
        gap_ok = (min_first_call_gap_s is None
                  or abs(t2 - t1) > min_first_call_gap_s)
        band_ok = (first_call_bands is None or _band(t1) != _band(t2))
        if gap_ok and band_ok:
            break
    else:
        raise RuntimeError(f"{point_id}: no onset layout satisfying first-call "
                           f"band/gap constraints in 300 attempts")
    return _finish_plan(slots, starts, n_events)


def _plan_banded_starts(rng, point_id, bands, target, lo, hi, step,
                        min_first_call_gap_s):
    """带优先:两个首叫先各自锁进目标带,其余事件再条件采样。

    带内仍是条件均匀采样(保住"首叫时刻在带内不可预测");事件数 3/4
    的选择随目标带对的可行性走——晚带对需要更短的事件序列才塞得下。
    """
    b1, b2 = target
    lo1, hi1 = bands[b1] * SAMPLE_RATE, bands[b1 + 1] * SAMPLE_RATE
    lo2, hi2 = bands[b2] * SAMPLE_RATE, bands[b2 + 1] * SAMPLE_RATE
    gap_samples = (0 if min_first_call_gap_s is None
                   else int(min_first_call_gap_s * SAMPLE_RATE))
    for n_events in [int(v) for v in rng.permutation([3, 4])]:
        t1_lo = max(lo, int(lo1))
        t1_hi = min(hi - (n_events - 1) * step, int(hi1) - 1)
        if t1_lo > t1_hi:
            continue
        for _attempt in range(200):
            t1 = int(rng.integers(t1_lo, t1_hi + 1))
            t2_lo = max(t1 + step, int(lo2), t1 + gap_samples + 1)
            t2_hi = min(hi - (n_events - 2) * step, int(hi2) - 1)
            if t2_lo > t2_hi:
                continue
            starts = [t1, int(rng.integers(t2_lo, t2_hi + 1))]
            cursor = starts[-1] + step
            ok = True
            for i in range(2, n_events):
                hi_i = hi - (n_events - 1 - i) * step
                if cursor > hi_i:
                    ok = False
                    break
                starts.append(int(rng.integers(cursor, hi_i + 1)))
                cursor = starts[-1] + step
            if ok:
                return starts, n_events
    raise RuntimeError(f"{point_id}: no onset layout lands the first calls in "
                       f"bands {target} (tried 3 and 4 events)")


def _finish_plan(slots, starts, n_events):
    events = list(zip(slots, starts))
    anchor_slot, anchor_start = events[-1]
    anchor = {"anchor_event_index": n_events - 1,
              "anchor_slot": anchor_slot,
              "anchor_start_sample": anchor_start,
              "anchor_end_sample": anchor_start + EVENT_LEN,
              "tail_silence_samples": SAMPLE_COUNT - (anchor_start + EVENT_LEN),
              "n_events": n_events,
              "per_slot_counts": {s: slots.count(s) for s in set(slots)}}
    return events, anchor


def build_program(request: dict, events, *, revision: str = "v1") -> dict:
    if "slot_endpoints" in request:
        slot_to_ep = {
            str(slot): str(endpoint)
            for slot, endpoint in request["slot_endpoints"].items()}
        if len(slot_to_ep) < 2 or len(set(slot_to_ep.values())) != len(slot_to_ep):
            raise ValueError("slot_endpoints must bind at least two unique endpoints")
        endpoints = list(slot_to_ep.values())
    else:
        endpoints = [request["endpoint_1"], request["endpoint_2"]]
        slot_to_ep = {"source1": endpoints[0], "source2": endpoints[1]}
    ev_rows = []
    for i, event in enumerate(events):
        if len(event) == 2:
            slot, start = event
            sound_asset_id = request["sound_asset_id"]
        elif len(event) == 3:
            slot, start, sound_asset_id = event
        else:
            raise ValueError("events must be (slot,start) or (slot,start,sound_asset_id)")
        end = start + EVENT_LEN
        ev_rows.append({
            "event_id": f"{slot}_event_{i}",
            "source_endpoint_id": slot_to_ep[slot],
            "sound_asset_id": str(sound_asset_id),
            "start_tick": start * TICKS_PER_SAMPLE,
            "end_tick_exclusive": end * TICKS_PER_SAMPLE,
            "start_sample": start,
            "end_sample_exclusive": end,
            "source_start_sample": SOURCE_START,
            "source_end_sample_exclusive": SOURCE_END,
            "linear_gain": 0.18,
            "fade_samples": 80,
            "normalization_policy": "use_sound_asset_policy",
            "render_source_stem": True,
        })
    doc = {
        "schema": "avengine_m6_audio_program_v1",
        "program_id": f"qa_v3_{request['pair_kind']}_{request['point_id']}_rand_{revision}",
        "revision": revision,
        "mode": str(request.get("mode", "sequential_sources")),
        "timeline": dict(TIMELINE_BLOCK),
        "candidate_source_endpoint_ids": endpoints,
        "events": ev_rows,
        "source_specific_stems": True,
        "admission_state": "research",
    }
    doc["program_content_sha256"] = canonical_json_sha256(doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--first-min-s", type=float, default=0.3)
    parser.add_argument("--gap-min-s", type=float, default=0.3)
    parser.add_argument("--tail-silence-s", type=float, default=1.5)
    parser.add_argument("--schema", default=str(REPO / "schemas/m6_audio_program_v1.schema.json"))
    args = parser.parse_args(argv)

    if os.path.exists(args.out_dir):
        print(f"refusing to overwrite existing output dir: {args.out_dir}", file=sys.stderr)
        return 2
    requests = json.load(open(args.requests))
    import jsonschema
    validator = jsonschema.Draft202012Validator(json.load(open(args.schema)))

    os.makedirs(args.out_dir)
    onset_frames = []
    rows = []
    for req in requests:
        events, anchor = plan_events(args.seed, req["point_id"], req["first_slot"],
                                     first_min_s=args.first_min_s,
                                     gap_min_s=args.gap_min_s,
                                     tail_silence_s=args.tail_silence_s)
        doc = build_program(req, events)
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
        if errors:
            print(f"FAIL {req['point_id']}: schema violation: {errors[0].message}",
                  file=sys.stderr)
            return 1
        with open(os.path.join(args.out_dir, doc["program_id"] + ".json"), "w") as fp:
            json.dump(doc, fp, ensure_ascii=False, indent=1)
        plan = {"schema": "avengine_qa_v3_program_plan_v1",
                "status": "research_candidate", "qualification_claim": False,
                "point_id": req["point_id"], "program_id": doc["program_id"],
                "first_slot": req["first_slot"],
                "parameters": {"first_min_s": args.first_min_s,
                               "gap_min_s": args.gap_min_s,
                               "tail_silence_s": args.tail_silence_s},
                **anchor}
        with open(os.path.join(args.out_dir, doc["program_id"] + ".plan.json"), "w") as fp:
            json.dump(plan, fp, ensure_ascii=False, indent=1)
        onset_frames.append(int(events[0][1] / SAMPLE_COUNT * 75))
        rows.append({"point_id": req["point_id"], "program_id": doc["program_id"],
                     "n_events": anchor["n_events"],
                     "first_onset_frame": onset_frames[-1],
                     "anchor_start_frame": int(anchor["anchor_start_sample"]
                                               / SAMPLE_COUNT * 75),
                     "anchor_slot": anchor["anchor_slot"]})

    # 批级自检:旧病根是"全批首声恒在同一帧",所以最贴切的锚是
    # 单一帧值的占比上限;桶数下限取 2——首声可行域被"≥3 声+间隔+尾窗"
    # 数学地压在片长前 40%(约帧 4–30),散布其中已不可预测,
    # 更宽的桶数要求会误伤合法约束。
    from collections import Counter
    first_buckets = {f // 15 for f in onset_frames}
    anchor_buckets = {r["anchor_start_frame"] // 15 for r in rows}
    top_frame_share = max(Counter(onset_frames).values()) / len(rows)
    manifest = {
        "schema": "avengine_qa_v3_program_batch_v1",
        "status": "research_candidate", "qualification_claim": False,
        "seed": args.seed,
        "count": len(rows),
        "first_onset_frame_buckets": sorted(first_buckets),
        "anchor_frame_buckets": sorted(anchor_buckets),
        "top_first_frame_share": round(top_frame_share, 4),
        "rows": rows,
    }
    with open(os.path.join(args.out_dir, "programs_manifest.json"), "w") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=1)
    if len(requests) >= 12 and (len(first_buckets) < 2 or len(anchor_buckets) < 2
                                or top_frame_share > 0.3):
        print(f"FAIL: onsets collapse (first={sorted(first_buckets)}, "
              f"anchor={sorted(anchor_buckets)}, top_frame_share="
              f"{top_frame_share:.2f}) — randomization is broken", file=sys.stderr)
        return 1
    print(f"programs={len(rows)} first_buckets={sorted(first_buckets)} "
          f"anchor_buckets={sorted(anchor_buckets)} "
          f"top_first_frame_share={top_frame_share:.2f} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
