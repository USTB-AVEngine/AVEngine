#!/usr/bin/env python3
"""只读汇总题库的私有分层字段：题型 × 房间 × 可见状态的题数，和单模态候选数的可用率

读 <bank_run>/private/strata.jsonl，什么都不写（除非显式给 --json）。用法：

    python tools/dataset/summarize_question_strata.py <bank_run> [--json <out.json>]

第一张表回答"按查询时刻的可见状态分层，每个题型每个房间各有多少题"；第二张表回答
"unimodal_candidates 这一栏有多少题算得出来、算出来的长什么样"；第三张表是三种题型
自带可见状态证据时的对拍命中率，它是这套取帧规则有没有取对帧的直接证据。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.question_strata import (  # noqa: E402
    CANDIDATE_RULES,
    FRAME_RULE_NOTES,
    QUERY_FRAME_RULES,
    UNDERIVED_REASONS,
    summarize_strata,
)

STATE_COLUMNS = (
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
    "out_of_view",
    "window_mixed",
    "mixed",
    "whole_clip",
    "unknown",
)


def read_rows(bank_run: Path) -> list[dict]:
    path = bank_run / "private" / "strata.jsonl"
    if not path.is_file():
        raise SystemExit(f"没有找到 {path}；这个题库还没有导出私有分层字段。")
    return [json.loads(line) for line in path.open() if line.strip()]


def print_frame_rules() -> None:
    print("== 取帧规则表（每个题型的查询时刻怎么定）==")
    width = max(len(rule) for rule in set(QUERY_FRAME_RULES.values()))
    for qa_id in sorted(QUERY_FRAME_RULES):
        rule = QUERY_FRAME_RULES[qa_id]
        print(f"  {qa_id}  {rule:<{width}}  {FRAME_RULE_NOTES.get(rule, '')}")
    print()


def print_visibility_table(rows: list[dict]) -> None:
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    rooms = set()
    for row in rows:
        qa_id = str(row.get("qa_id"))
        room = str(row.get("room_family"))
        rooms.add(room)
        state = str((row.get("visibility_at_query") or {}).get("state"))
        counts[(qa_id, room)][state] += 1
    print("== 题型 × 房间 × 查询时刻可见状态 的题数 ==")
    header = f"  {'qa_id':<7}{'room':<11}" + "".join(f"{name:>18}" for name in STATE_COLUMNS) + f"{'total':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    totals: Counter = Counter()
    for qa_id, room in sorted(counts):
        bucket = counts[(qa_id, room)]
        line = f"  {qa_id:<7}{room:<11}" + "".join(f"{bucket.get(name, 0):>18}" for name in STATE_COLUMNS)
        print(line + f"{sum(bucket.values()):>8}")
        totals.update(bucket)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'all':<7}{'all':<11}" + "".join(f"{totals.get(name, 0):>18}" for name in STATE_COLUMNS)
          + f"{sum(totals.values()):>8}")
    other = set(totals) - set(STATE_COLUMNS)
    if other:
        print("  表外还出现了这些状态：", {name: totals[name] for name in sorted(other)})
    print()


def print_unimodal_table(rows: list[dict]) -> None:
    print("== unimodal_candidates 可用率，以及算出来的 (只听, 只看, 选项数) 分布 ==")
    print(f"  {'qa_id':<7}{'题数':>6}{'算得出':>8}{'可用率':>9}   (audio, video, options) -> 题数 / 算不出的原因")
    by_qa: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_qa[str(row.get("qa_id"))].append(row)
    for qa_id in sorted(by_qa):
        group = by_qa[qa_id]
        counted = [r for r in group if (r.get("unimodal_candidates") or {}).get("audio_only") is not None]
        share = f"{100.0 * len(counted) / len(group):.0f}%"
        shape = Counter(
            (
                r["unimodal_candidates"]["audio_only"],
                r["unimodal_candidates"]["video_only"],
                r["unimodal_candidates"]["option_count"],
            )
            for r in counted
        )
        if counted:
            tail = "  ".join(f"{key}:{value}" for key, value in sorted(shape.items()))
        else:
            reasons = Counter(str((r.get("unimodal_candidates") or {}).get("reason")) for r in group)
            tail = reasons.most_common(1)[0][0]
        print(f"  {qa_id:<7}{len(group):>6}{len(counted):>8}{share:>9}   {tail}")
    print()


def print_necessity(summary: dict) -> None:
    necessity = summary["unimodal_necessity"]
    counted = necessity["counted"] or 1
    print("== 竞品那条必要性规则（|Ca|>1、|Cv|>1、|Cav|=1）在算得出的题上的分布 ==")
    for key in ("necessary_multimodal", "audio_only_sufficient", "video_only_sufficient",
                "either_modality_sufficient"):
        value = necessity[key]
        print(f"  {key:<28}{value:>6}  ({100.0 * value / counted:.1f}% of {necessity['counted']})")
    print("  这是记录不是闸门：算出来 1 也不删题、不改政策。")
    print()
    checks = summary["visibility_cross_checks"]
    if checks:
        print("== 取帧与查表的对拍（题目证据自带可见状态的那几类）==")
        for field in sorted(checks):
            entry = checks[field]
            print(f"  {field:<24}{entry['agrees']:>6} / {entry['checked']:<6}命中")
        print()


def print_gaps() -> None:
    print("== 本轮没有算候选数的题型，各自的原因 ==")
    for qa_id in sorted(set(QUERY_FRAME_RULES) - set(CANDIDATE_RULES)):
        print(f"  {qa_id}  {UNDERIVED_REASONS.get(qa_id, '(没有写原因)')}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bank_run", type=Path, help="题库导出目录，里面要有 private/strata.jsonl")
    parser.add_argument("--json", type=Path, help="把汇总同时写成一个 JSON 文件（默认什么都不写）")
    args = parser.parse_args()
    rows = read_rows(args.bank_run.resolve())
    summary = summarize_strata(rows)
    print(f"题库 {args.bank_run}  共 {summary['question_count']} 题\n")
    print_frame_rules()
    print_visibility_table(rows)
    print_unimodal_table(rows)
    print_necessity(summary)
    print_gaps()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print("汇总也写到了", args.json)


if __name__ == "__main__":
    main()
