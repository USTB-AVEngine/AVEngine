#!/usr/bin/env python3
"""Glue: facts (generator) -> split plan (1.3) -> MCQ items (1.4 input).

三步(全部 no-clobber,失败即停):
1. 从设计批 spec + facts 汇出 split 点位文件(trajectory_id = 路线坐标
   sha,twin 沿用其主点、房间恒 apartment_0000 —— 单房软维度由隔离器
   诚实披露);
2. 调 split_isolator assign 切 train/eval(比例显式参数);
3. facts jsonl → build_mcq_options 的 items jsonl:标签词面映射
   (black-and-white→black_white 等)、card1 补 other_ending_deg、card8
   补同点对方首叫、card7 负样本带 negative_control 标记。

card16 不在本批(掩码通道未接)。research_candidate。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LABEL_MAP = {"black-and-white": "black_white", "yellow": "yellow",
             "both": "both_calling", "neither": "none_calling"}

CARD_FILES = ("card1", "card7_main", "card7_control_neither", "card8",
              "card9")
CARD_OF_VIEW = {"card7_main": "card7", "card7_control_neither": "card7"}


def _band_label(deg, edges):
    """另一只片尾方位所属的声明带标签;落在带域外返回 None(不同带即可)。"""
    for i in range(len(edges) - 1):
        if edges[i] <= deg < edges[i + 1]:
            return f"[{edges[i]:g}, {edges[i + 1]:g})"
    return None


def stratified_balanced(items, key_fields, truth_of, seed):
    """在 split × motion_class × truth 分层里等量选取(codex 审阅裁定)。

    全局 50:50 不够:run01 的全局均衡子集内,只按运动类猜多数类仍有
    60.6%。分层后每个 (split, motion_class) 单元内两类等量,单元内的
    多数类先验因此消失。少数类整组保留,多数类按确定性哈希序截齐。
    """
    cells: dict[tuple, dict[str, list]] = {}
    for item in items:
        cell = tuple(item.get(f) for f in key_fields)
        cells.setdefault(cell, {}).setdefault(truth_of(item), []).append(item)
    chosen = set()
    for cell, by_truth in cells.items():
        n = min(len(v) for v in by_truth.values()) if len(by_truth) > 1 else 0
        for truth, group in by_truth.items():
            keep = sorted(group, key=lambda it: hashlib.sha256(
                f"{seed}|bal|{it['question_id']}".encode()).hexdigest())[:n]
            chosen.update(id(it) for it in keep)
    for item in items:
        item["balanced_subset"] = id(item) in chosen


def route_sha(route) -> str:
    return hashlib.sha256(json.dumps(route).encode()).hexdigest()[:16]


def load_facts(qroot: Path, card: str) -> list[dict]:
    p = qroot / f"facts_{card}.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-root", required=True, type=Path)
    parser.add_argument("--questions-root", required=True, type=Path)
    parser.add_argument("--ratios", default="train=0.5,eval=0.5",
                        help="split 比例,显式参数")
    parser.add_argument("--mcq-cards",
                        default="card1,card7_main,card7_control_neither,"
                                "card8,card9",
                        help="进 MCQ 编排的卡(split 点位文件仍含全部卡的点)")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.out_root.exists():
        print(f"refusing to overwrite: {args.out_root}", file=sys.stderr)
        return 2
    repo = Path(__file__).resolve().parents[2]
    args.out_root.mkdir(parents=True)

    facts = {c: load_facts(args.questions_root, c) for c in CARD_FILES}
    point_ids = sorted({r["point_id"] for recs in facts.values()
                        for r in recs})

    # 1) split 点位文件(主点;孪生不进题池但同组维度已在 twin_group 上)
    points = []
    for pid in point_ids:
        spec = json.loads(
            (args.design_root / pid / "spec.json").read_text())
        troutes = sorted(route_sha(spec[k]) for k in ("s1_route", "s2_route")
                         if spec.get(k) is not None)
        points.append({
            "point_id": pid,
            "episode_id": pid,
            "twin_of": spec.get("twin_of"),
            "room_id": "apartment_0000",
            "trajectory_id": "+".join(troutes),
            "speaker_voice_ids": [],
            "transcript_ids": [],
        })
    points_path = args.out_root / "split_points.json"
    points_path.write_text(json.dumps(points, ensure_ascii=False, indent=1))

    plan_path = args.out_root / "split_plan.json"
    cmd = [args.python, str(repo / "tools/qa/split_isolator.py"), "assign",
           "--points", str(points_path), "--ratios", args.ratios,
           "--seed", args.seed, "--out", str(plan_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"split_isolator failed:\n{proc.stdout}\n{proc.stderr}",
              file=sys.stderr)
        return 1
    plan = json.loads(plan_path.read_text())
    assign = plan.get("assignment", plan)

    # 2) facts -> MCQ items
    items = []
    onsets_by_point: dict[str, dict[str, float]] = {}
    for rec in facts["card8"]:
        onsets_by_point.setdefault(rec["point_id"], {})[
            rec["target_slot"]] = rec["truth"]["first_onset_s"]
    mcq_cards = set(args.mcq_cards.split(","))
    for card, recs in facts.items():
        if card not in mcq_cards:
            continue
        for rec in recs:
            pid = rec["point_id"]
            split = assign.get(pid)
            if split is None:
                print(f"point {pid} missing from split plan", file=sys.stderr)
                return 1
            item = {"card": CARD_OF_VIEW.get(card, card), "view": card,
                    "point_id": pid, "split": split,
                    "motion_class": rec.get("motion_class"),
                    "question_id": f"{card}|{pid}" + (
                        f"|{rec['target_slot']}" if card == "card8" else "")}
            if card == "card1":
                blk = rec.get("azimuth_band") or {}
                if not blk.get("options_space"):
                    continue          # 越出声明带域的点不出 MCQ,开放版仍在
                item["card"] = "card1_az_band"
                item["band_labels"] = blk["options_space"]
                item["truth_band_label"] = blk["truth_option"]
                other = _band_label(rec["truth"]["other_slot_final_azimuth_deg"],
                                    blk["edges"])
                item["other_band_label"] = other
                item["truth_deg"] = rec["truth"]["final_azimuth_deg"]
                item["other_ending_deg"] = rec["truth"][
                    "other_slot_final_azimuth_deg"]
            elif card.startswith("card7"):
                item["truth_label"] = LABEL_MAP[rec["truth"]["calling_at_t"]]
                if rec["negative_sample"]:
                    item["negative_control"] = True
            elif card == "card8":
                if rec["mcq"] is None:
                    continue      # 越带题无 MCQ 形式,开放版另走
                slot = rec["target_slot"]
                other_slot = "source1" if slot == "source2" else "source2"
                item["truth_s"] = rec["truth"]["first_onset_s"]
                item["other_first_bark_s"] = onsets_by_point[pid][other_slot]
            elif card == "card9":
                item["truth_label"] = LABEL_MAP[rec["truth"]["first_to_bark"]]
            items.append(item)

    # 分层均衡子集标记(切分之后才做得了)
    stratified_balanced([i for i in items if i["view"] == "card7_main"],
                        ("split", "motion_class"),
                        lambda i: i["truth_label"], args.seed)
    stratified_balanced([i for i in items if i["card"] == "card9"],
                        ("split", "motion_class"),
                        lambda i: i["truth_label"], args.seed)
    for item in items:
        item.setdefault("balanced_subset", None)

    items_path = args.out_root / "mcq_items.jsonl"
    with open(items_path, "w") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    # build_mcq_options 读 JSON 数组
    (args.out_root / "mcq_items.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1))

    manifest = {
        "schema": "qa_v3_mcq_prep_manifest_v1",
        "questions_root": str(args.questions_root),
        "ratios": args.ratios,
        "seed": args.seed,
        "counts": {c: sum(1 for i in items if i["view"] == c)
                   for c in CARD_FILES},
        "balanced_subset_counts": {
            v: sum(1 for i in items
                   if i["view"] == v and i.get("balanced_subset"))
            for v in ("card7_main", "card9")},
        "split_sizes": {s: sum(1 for v in assign.values() if v == s)
                        for s in sorted(set(assign.values()))},
        "unisolated_dimensions": plan.get("unisolated"),
        "mcq_cards": sorted(mcq_cards),
        "status": "research_candidate",
        "qualification_claim": False,
    }
    (args.out_root / "prep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"out": str(args.out_root),
                      "counts": manifest["counts"],
                      "split_sizes": manifest["split_sizes"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
