#!/usr/bin/env python3
"""Split isolator for pilot batches (work order item 1.3).

把一批采样点(episode)切成 train/eval 等份,并保证六个维度的隔离:
同一**视觉片段、孪生组、房间、轨迹、说话人、台词**的值不得同时出现在
两个切分里(泄漏防线)。孪生永远与其主点同侧沿用 batch2d 既有规则。

原理:六维中任一维同值 ⇒ 两个点必须同侧。把"必须同侧"关系建成图,
取连通分量,分量整体分配;分配用大分量优先的贪心装箱逼近目标比例。

诚实边界:某维在整批里只有一个取值(例如 pilot 批全在同一房间)时,
该维**不可隔离**——工具不假装隔离成功,而是把该维列进
`single_value_dimensions` 显式报告,由读者判断是否可接受;**空值不
参与连边**(没有说话人的犬吠点不因"都没有说话人"被锁在一起)。

两种用法:
  分配:  split_isolator.py assign --points P.json --ratios train=0.8,eval=0.2 \
              --seed qa_v3_pilot --out plan.json
  扫描:  split_isolator.py check --points P.json --plan plan.json --out report.json
          (对已有切分找违规;有违规则非零退出——失败即停)

点位记录字段(缺失按空值处理):point_id 必填;episode_id、twin_of、
room_id、trajectory_id、speaker_voice_ids(列表)、transcript_ids(列表)。
输出一律 no-clobber。research_candidate;不构成 dataset admission。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

HARD_DIMENSIONS = ("episode_id", "twin_group", "trajectory_id",
                   "speaker_voice_ids", "transcript_ids")
SOFT_DIMENSIONS = ("room_id",)
DIMENSIONS = HARD_DIMENSIONS + SOFT_DIMENSIONS
# 硬维:同值必须同侧,跨集永远是违规(整组放同一侧总是可满足的——
#       代价最多是切分比例,不豁免;这是测试抓出的语义修正:早先按
#       "该维只有一个取值"豁免会漏报"两个点共享同一说话人"的真泄漏,
#       因为空值点不进桶)。
# 软维:值覆盖整批时(如 pilot 批全在同一房间)隔离在数学上不可达,
#       降级为 unisolated 显式标注;部分覆盖而跨集仍是违规。


class _DSU:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _twin_group(point: dict) -> str:
    """孪生组键:孪生指向主点,主点指向自己——同组必然同键。"""
    return str(point.get("twin_of") or point["point_id"])


def _dim_values(point: dict, dim: str) -> list[str]:
    if dim == "twin_group":
        return [_twin_group(point)]
    value = point.get(dim)
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def build_components(points: list[dict]):
    """返回 (dsu, single_value_dims):同维同值连边;单值维不连边并报告。"""
    ids = [p["point_id"] for p in points]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate point_id in input")
    dsu = _DSU(ids)
    unisolated: list[dict] = []
    total = len(points)
    for dim in DIMENSIONS:
        buckets: dict[str, list[str]] = defaultdict(list)
        for p in points:
            for v in _dim_values(p, dim):
                buckets[v].append(p["point_id"])
        for value, members in buckets.items():
            if dim in SOFT_DIMENSIONS and len(set(members)) == total:
                unisolated.append({"dimension": dim, "value": value,
                                   "reason": "value covers the whole batch"})
                continue  # 软维全覆盖:隔离不可达,显式标注,不连边
            for other in members[1:]:
                dsu.union(members[0], other)
    return dsu, unisolated


def assign(points: list[dict], ratios: dict[str, float], seed: str):
    dsu, unisolated = build_components(points)
    comps: dict[str, list[str]] = defaultdict(list)
    for p in points:
        comps[dsu.find(p["point_id"])].append(p["point_id"])
    # 确定性排序:大分量优先,同大小按内容哈希(种子participates)
    def comp_key(root):
        members = sorted(comps[root])
        digest = hashlib.sha256((seed + "\0" + "\0".join(members)).encode()).hexdigest()
        return (-len(members), digest)

    order = sorted(comps, key=comp_key)
    total = len(points)
    targets = {name: ratio * total for name, ratio in ratios.items()}
    filled = {name: 0 for name in ratios}
    assignment: dict[str, str] = {}
    for root in order:
        # 装到"距目标缺口最大"的切分
        gaps = {name: targets[name] - filled[name] for name in ratios}
        best = max(sorted(gaps), key=lambda n: gaps[n])
        for pid in comps[root]:
            assignment[pid] = best
        filled[best] += len(comps[root])
    return assignment, unisolated, {r: comps[r] for r in order}


def check(points: list[dict], plan: dict[str, str]):
    """对已有切分扫描六维违规。返回 (违规清单, 未隔离标注清单)。
    硬维跨集一律违规;软维的值覆盖全批时降级为 unisolated 标注。"""
    violations = []
    unisolated = []
    missing = [p["point_id"] for p in points if p["point_id"] not in plan]
    if missing:
        violations.append({"kind": "unassigned_points", "points": sorted(missing)[:20],
                           "count": len(missing)})
    assigned_total = sum(1 for p in points if p["point_id"] in plan)
    for dim in DIMENSIONS:
        buckets: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for p in points:
            split = plan.get(p["point_id"])
            if split is None:
                continue
            for v in _dim_values(p, dim):
                buckets[v][split].append(p["point_id"])
        for value, by_split in buckets.items():
            covered = sum(len(m) for m in by_split.values())
            if dim in SOFT_DIMENSIONS and covered == assigned_total:
                unisolated.append({"dimension": dim, "value": value,
                                   "reason": "value covers the whole batch"})
                continue
            if len(by_split) > 1:
                violations.append({
                    "kind": "cross_split_leak", "dimension": dim, "value": value,
                    "splits": {s: sorted(m)[:6] for s, m in by_split.items()},
                })
    return violations, unisolated


def _load(path):
    with open(path) as fp:
        return json.load(fp)


def _dump_no_clobber(path, payload) -> bool:
    if os.path.exists(path):
        print(f"refusing to overwrite existing output: {path}", file=sys.stderr)
        return False
    with open(path, "w") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=1)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("assign")
    pa.add_argument("--points", required=True)
    pa.add_argument("--ratios", required=True, help="如 train=0.8,eval=0.2")
    pa.add_argument("--seed", required=True)
    pa.add_argument("--out", required=True)
    pc = sub.add_parser("check")
    pc.add_argument("--points", required=True)
    pc.add_argument("--plan", required=True, help="assign 的输出,或 {point_id: split} 映射")
    pc.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    points = _load(args.points)

    if args.cmd == "assign":
        ratios = {}
        for part in args.ratios.split(","):
            name, _, val = part.partition("=")
            ratios[name.strip()] = float(val)
        if abs(sum(ratios.values()) - 1.0) > 1e-6:
            print("ratios must sum to 1", file=sys.stderr)
            return 2
        assignment, unisolated, comps = assign(points, ratios, args.seed)
        counts = defaultdict(int)
        for split in assignment.values():
            counts[split] += 1
        payload = {
            "schema": "avengine_split_plan_v1",
            "status": "research_candidate",
            "qualification_claim": False,
            "seed": args.seed,
            "ratios": ratios,
            "counts": dict(counts),
            "unisolated": unisolated,
            "component_sizes": sorted((len(v) for v in comps.values()), reverse=True)[:20],
            "assignment": assignment,
        }
        if not _dump_no_clobber(args.out, payload):
            return 2
        # 自检:分配完立即用 check 复核一遍(自己出的计划必须过自己的扫描)
        violations, _ = check(points, assignment)
        if violations:
            print(f"self-check FAILED with {len(violations)} violation(s)", file=sys.stderr)
            return 1
        print(f"assigned={dict(counts)} unisolated={unisolated} "
              f"largest_component={payload['component_sizes'][:3]} out={args.out}")
        return 0

    plan_doc = _load(args.plan)
    plan = plan_doc.get("assignment", plan_doc)
    violations, unisolated = check(points, plan)
    payload = {
        "schema": "avengine_split_check_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "violations": violations,
        "violation_count": len(violations),
        "unisolated": unisolated,
    }
    if not _dump_no_clobber(args.out, payload):
        return 2
    if violations:
        print(f"FAIL: {len(violations)} violation(s); first: {violations[0]}")
        return 1
    print(f"no cross-split leakage found; unisolated={unisolated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
