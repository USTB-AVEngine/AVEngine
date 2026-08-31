"""Stratified balanced-subset selection (codex 审阅裁定的 run01 修法)。

阳性对照:构造一个"全局 50:50 但分层内严重偏斜"的集合——正是 run01 的
形态(全局均衡子集内按运动类猜多数类仍有 60.6%)——证明全局配平放它
过去,而分层配平把每个单元内的多数类先验压到 50%。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from prepare_qa_v3_mcq import stratified_balanced  # noqa: E402


def make_items():
    """两个运动类,全局 12:12,但 A 类 10:2、B 类 2:10 —— 只看运动类
    猜多数类即得 20/24 = 83%。"""
    items = []
    n = 0
    for cls, (n_black, n_yellow) in (("A1", (10, 2)), ("B", (2, 10))):
        for truth, count in (("black_white", n_black), ("yellow", n_yellow)):
            for _ in range(count):
                n += 1
                items.append({"question_id": f"q{n}", "split": "train",
                              "motion_class": cls, "truth_label": truth})
    return items


def majority_rate(items, key):
    cells = {}
    for it in items:
        cells.setdefault(it[key], {}).setdefault(it["truth_label"], 0)
        cells[it[key]][it["truth_label"]] += 1
    hit = sum(max(c.values()) for c in cells.values())
    return hit / len(items)


def test_stratified_kills_within_cell_prior():
    items = make_items()
    assert abs(majority_rate(items, "motion_class") - 20 / 24) < 1e-9
    stratified_balanced(items, ("split", "motion_class"),
                        lambda i: i["truth_label"], "seed")
    kept = [i for i in items if i["balanced_subset"]]
    assert len(kept) == 8               # 每类取 min(10,2)=2 的两倍
    assert abs(majority_rate(kept, "motion_class") - 0.5) < 1e-9
    for cls in ("A1", "B"):
        cell = [i for i in kept if i["motion_class"] == cls]
        assert sum(1 for i in cell if i["truth_label"] == "black_white") == \
            sum(1 for i in cell if i["truth_label"] == "yellow")


def test_split_is_part_of_the_stratum():
    items = make_items()
    for i, item in enumerate(items):
        item["split"] = "train" if i % 2 == 0 else "eval"
    stratified_balanced(items, ("split", "motion_class"),
                        lambda i: i["truth_label"], "seed")
    kept = [i for i in items if i["balanced_subset"]]
    for split in ("train", "eval"):
        for cls in ("A1", "B"):
            cell = [i for i in kept
                    if i["split"] == split and i["motion_class"] == cls]
            counts = {}
            for i in cell:
                counts[i["truth_label"]] = counts.get(i["truth_label"], 0) + 1
            assert len(set(counts.values())) <= 1, (split, cls, counts)


def test_single_truth_cell_contributes_nothing():
    # 某单元内只有一类 → 该单元整体不入均衡子集(不能靠它凑数)
    items = [{"question_id": f"q{i}", "split": "train", "motion_class": "A1",
              "truth_label": "yellow"} for i in range(6)]
    stratified_balanced(items, ("split", "motion_class"),
                        lambda i: i["truth_label"], "seed")
    assert not any(i["balanced_subset"] for i in items)


def test_selection_is_deterministic():
    a, b = make_items(), make_items()
    for items in (a, b):
        stratified_balanced(items, ("split", "motion_class"),
                            lambda i: i["truth_label"], "seed")
    assert [i["balanced_subset"] for i in a] == [i["balanced_subset"] for i in b]
