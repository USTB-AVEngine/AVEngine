#!/usr/bin/env python3
"""MCQ option builder for the dual-source five-card pilot (work order 1.4).

职责:把每题的事实记录变成选择题呈现——选项集合、呈现顺序、正确位置
——并把三类历史捷径制度化地封死:

1. **位置先验**(batch2d 教训:正确答案落在 A/B 的频率 0.34 高于 C/D 的
   0.16):对每个 题型×切分 桶做**正确位置轮转均衡**(各位置计数差 ≤1),
   轮转起点与题序由种子决定,可复现。
2. **上游约束复查**(纵深防线,采样器应拒而未拒的点在这里再拦一道):
   卡① 两只片尾同扇区 → 拒;卡⑧ 两只首叫同时间带 → 拒;卡⑯ 两只片尾
   同可见性状态 → 拒;卡⑦ 非"恰好一只在叫"的点必须带 negative_control
   标记,主集里混入未标记的负样本 → 拒。拒出率超过 --reject-threshold
   (默认 5%)则整批非零退出——说明上游采样坏了,停。
3. **负样本频率披露**:"都在叫/都没叫/已出画"这类选项作为正确答案的
   频率按 题型×切分 写进汇总,不藏进总分。

首轮五卡的选项规则(全部是固定闭集枚举,无自由干扰项抽取;内容题的
"候选须在片段内齐备"规则属多源族,留接口不在本轮):
  card1  四扇区:前 [−45°,45°) / 右 [45°,135°) / 后 [135°,180°]∪(−180°,−135°) /
         左 [−135°,−45°);正确项 = 真值角所在扇区(角度先归一到 (−180°,180])。
  card7  黑白毛 / 黄毛 / 都在叫 / 都没叫(负样本仅进对照桶)。
  card8  四个等宽半开时间带(默认 [0,1.25) [1.25,2.5) [2.5,3.75) [3.75,5.0],
         末带右闭),边界由 --bands 显式给出。
  card9  二选一毛色(在产形态)。
  card16 四态:visible_clear / visible_occluded / fully_occluded / out_of_view。

输出:逐题呈现记录 + 按 题型×切分 的位置分布、标签分布、负样本频率、
拒出清单;manifest no-clobber。research_candidate。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

SECTORS = ["front", "right", "back", "left"]
SECTOR_ZH = {"front": "前", "right": "右", "back": "后", "left": "左"}
CARD7_LABELS = ["black_white", "yellow", "both_calling", "none_calling"]
CARD9_LABELS = ["black_white", "yellow"]
CARD16_LABELS = ["visible_clear", "visible_occluded", "fully_occluded", "out_of_view"]
NEGATIVE_LABELS = {"both_calling", "none_calling", "out_of_view"}
DEFAULT_BANDS = [0.0, 1.25, 2.5, 3.75, 5.0]


def norm_deg(deg: float) -> float:
    """归一到 (−180, 180]。"""
    d = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def sector_of(deg: float) -> str:
    d = norm_deg(deg)
    if -45.0 <= d < 45.0:
        return "front"
    if 45.0 <= d < 135.0:
        return "right"
    if -135.0 <= d < -45.0:
        return "left"
    return "back"


def band_of(t: float, bands: list[float]) -> int | None:
    """半开带 [b_i, b_{i+1}),末带右闭;界外返回 None。"""
    if t < bands[0] or t > bands[-1]:
        return None
    for i in range(len(bands) - 1):
        if bands[i] <= t < bands[i + 1]:
            return i
    return len(bands) - 2  # t == bands[-1]


def _digest(seed: str, *parts: str) -> str:
    return hashlib.sha256(("\0".join((seed,) + parts)).encode()).hexdigest()


def _labels_and_truth(item: dict, bands: list[float]):
    """返回 (labels, truth_label, reject_reason)。"""
    card = item["card"]
    if card == "card1":
        truth = sector_of(item["truth_deg"])
        other = sector_of(item["other_ending_deg"])
        if truth == other:
            return None, None, "both endings in the same sector"
        return list(SECTORS), truth, None
    if card == "card7":
        truth = item["truth_label"]
        if truth not in CARD7_LABELS:
            return None, None, f"unknown card7 label {truth!r}"
        negative = truth in NEGATIVE_LABELS
        if negative and not item.get("negative_control"):
            return None, None, "negative anchor lacking negative_control mark"
        return list(CARD7_LABELS), truth, None
    if card == "card8":
        tb = band_of(item["truth_s"], bands)
        ob = band_of(item["other_first_bark_s"], bands)
        if tb is None or ob is None:
            return None, None, "first-bark time outside the clip"
        if tb == ob:
            return None, None, "both first barks in the same band"
        labels = [f"band_{i}" for i in range(len(bands) - 1)]
        return labels, f"band_{tb}", None
    if card == "card9":
        truth = item["truth_label"]
        if truth not in CARD9_LABELS:
            return None, None, f"unknown card9 label {truth!r}"
        return list(CARD9_LABELS), truth, None
    if card == "card16":
        truth = item["truth_label"]
        other = item["other_ending_state"]
        if truth not in CARD16_LABELS:
            return None, None, f"unknown card16 label {truth!r}"
        if truth == other:
            return None, None, "both endings share the same visibility state"
        return list(CARD16_LABELS), truth, None
    return None, None, f"unknown card {card!r}"


def build(items: list[dict], seed: str, bands: list[float], reject_threshold: float):
    accepted, rejected = [], []
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for item in items:
        labels, truth, why = _labels_and_truth(item, bands)
        if why:
            rejected.append({"question_id": item.get("question_id"), "card": item.get("card"),
                             "reason": why})
            continue
        bucket = "control" if item.get("negative_control") else "main"
        groups[(item["card"], item.get("split", "unsplit"), bucket, len(labels))].append(
            dict(item, _labels=labels, _truth=truth))

    presented = []
    balance_report = {}
    for key in sorted(groups, key=str):
        card, split, bucket, n_pos = key
        members = sorted(groups[key], key=lambda it: _digest(seed, it["question_id"]))
        # 正确位置轮转均衡:起点由种子定,逐题 +1
        start = int(_digest(seed, card, split, bucket), 16) % n_pos
        pos_counts = defaultdict(int)
        label_counts = defaultdict(int)
        for rank, it in enumerate(members):
            correct_index = (start + rank) % n_pos
            others = [l for l in it["_labels"] if l != it["_truth"]]
            # 非正确项顺序:种子化确定重排
            others = sorted(others, key=lambda l: _digest(seed, it["question_id"], l))
            options = list(others)
            options.insert(correct_index, it["_truth"])
            presented.append({
                "question_id": it["question_id"], "card": card, "split": split,
                "bucket": bucket, "options": options, "correct_index": correct_index,
                "truth_label": it["_truth"],
            })
            pos_counts[correct_index] += 1
            label_counts[it["_truth"]] += 1
        total = len(members)
        neg_true = sum(c for l, c in label_counts.items() if l in NEGATIVE_LABELS)
        balance_report[f"{card}/{split}/{bucket}"] = {
            "n": total,
            "correct_position_counts": {str(k): pos_counts[k] for k in sorted(pos_counts)},
            "truth_label_counts": dict(sorted(label_counts.items())),
            "negative_label_true_rate": round(neg_true / total, 4) if total else 0.0,
        }

    reject_rate = len(rejected) / len(items) if items else 0.0
    return presented, rejected, balance_report, reject_rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--bands", default=",".join(str(b) for b in DEFAULT_BANDS),
                        help="卡⑧时间带边界,显式参数")
    parser.add_argument("--reject-threshold", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if os.path.exists(args.out):
        print(f"refusing to overwrite existing output: {args.out}", file=sys.stderr)
        return 2
    items = json.load(open(args.items))
    bands = [float(x) for x in args.bands.split(",")]
    if sorted(bands) != bands or len(bands) < 3:
        print("bands must be an ascending list with >=3 edges", file=sys.stderr)
        return 2

    presented, rejected, balance, reject_rate = build(items, args.seed, bands,
                                                      args.reject_threshold)
    payload = {
        "schema": "avengine_mcq_presentation_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "seed": args.seed,
        "bands": bands,
        "counts": {"input": len(items), "presented": len(presented),
                   "rejected": len(rejected)},
        "reject_rate": round(reject_rate, 4),
        "balance": balance,
        "rejected": rejected,
        "presented": presented,
    }
    with open(args.out, "w") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=1)
    print(f"presented={len(presented)} rejected={len(rejected)} "
          f"reject_rate={reject_rate:.3f} out={args.out}")
    if reject_rate > args.reject_threshold:
        print(f"FAIL: reject rate {reject_rate:.3f} exceeds threshold "
              f"{args.reject_threshold} — upstream sampling is broken, stopping",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
