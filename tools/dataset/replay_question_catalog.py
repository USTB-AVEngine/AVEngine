#!/usr/bin/env python3
"""回放对拍：同一份 facts 重新出题，跟已导出题库的 checkpoint 逐字节比

判据从出题函数里拆出来之后，唯一能证明"行为没变"的办法是拿同一份输入再跑一遍，
跟拆之前的产物对。已经导出的题库每个源都留着 `sources/NNNN/checkpoint.json`，里面就是
那次出题的完整结果，所以这里不需要另存一份基线。用法：

    python tools/dataset/replay_question_catalog.py <bank_run> [--workers N] [--json out.json]

对不上就打印第一处差异的路径和两边的值，退出码非零。CPU 跑，不碰媒体、不写题库。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.unified_catalog import (  # noqa: E402
    generate_unified_questions,
    with_derived_sound_class_answer_domain,
)
from avengine.qa.binding_catalog import whole_degree_display  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def regenerate(facts_path, policy, items_per_type, seed):
    """完全照 generate_retained_qa_bank.py 的做法准备 facts 再出题。"""

    facts = deepcopy(read(facts_path))
    facts.setdefault("sampling", {})["acceptance_policy"] = policy
    facts = with_derived_sound_class_answer_domain(facts)
    facts["sampling"]["time_display_precision"] = 0
    facts["sampling"].setdefault("qa_sampling", {})["time_display_precision"] = 0
    return whole_degree_display(
        generate_unified_questions(
            facts, items_per_type=int(items_per_type), seed=str(seed),
            include_angle_followups=False,
        )
    )


def first_difference(left, right, path="questions"):
    """第一处不一样在哪，说得出具体路径，而不是只说"不相等"。"""

    if type(left) is not type(right):
        return f"{path}: 类型不同 {type(left).__name__} / {type(right).__name__}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                return f"{path}.{key}: 新产物少了这个键"
            if key not in right:
                return f"{path}.{key}: 新产物多了这个键"
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: 长度不同 {len(left)} / {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            found = first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def check_one(job):
    index, checkpoint_path, policy, items_per_type, seed = job
    checkpoint = read(checkpoint_path)
    source = checkpoint["source"]
    if checkpoint.get("status") != "pass":
        return {"index": index, "status": "skipped_failed_source",
                "facts_path": source.get("facts_path"), "reason": checkpoint.get("reason")}
    try:
        fresh = regenerate(source["facts_path"], policy, items_per_type, seed)
    except Exception as error:
        return {"index": index, "status": "regeneration_failed",
                "facts_path": source["facts_path"], "reason": f"{type(error).__name__}: {error}"}
    old = checkpoint["questions"]
    # 比较的是解析后的 JSON，所以键的书写顺序不算差异，值和结构必须一致。
    difference = first_difference(json.loads(json.dumps(old)), json.loads(json.dumps(fresh)))
    return {"index": index, "status": "identical" if difference is None else "differs",
            "facts_path": source["facts_path"],
            "item_count": len(fresh.get("items") or []),
            "difference": difference}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bank_run", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    bank = args.bank_run.resolve()
    config = read(bank / "run_config.json")
    policy = config["acceptance_policy"]
    items_per_type = config["config"]["items_per_type"]
    seed = config["config"]["seed"]
    checkpoints = sorted((bank / "sources").glob("*/checkpoint.json"))
    if not checkpoints:
        raise SystemExit(f"{bank} 下面没有 sources/*/checkpoint.json，没法回放对拍。")
    jobs = [(i, str(p), policy, items_per_type, seed) for i, p in enumerate(checkpoints)]

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(check_one, jobs, chunksize=1):
                results.append(result)
                print(".", end="", flush=True)
    else:
        for job in jobs:
            results.append(check_one(job))
            print(".", end="", flush=True)
    print()

    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    items = sum(r.get("item_count") or 0 for r in results)
    print(f"源 {len(results)} 个，题 {items} 道：{counts}")
    bad = [r for r in results if r["status"] in ("differs", "regeneration_failed")]
    for result in bad[:5]:
        print(f"  [{result['status']}] {result['facts_path']}")
        print(f"      {result.get('difference') or result.get('reason')}")
    summary = {"bank_run": str(bank), "source_count": len(results), "item_count": items,
               "status_counts": counts,
               "mismatches": [r for r in results if r["status"] != "identical"]}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print("对拍结果写到了", args.json)
    if bad:
        raise SystemExit(1)
    print("回放对拍通过：同一份 facts 出来的题和答案跟拆判据之前完全一致。")


if __name__ == "__main__":
    main()
