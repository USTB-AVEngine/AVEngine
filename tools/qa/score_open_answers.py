#!/usr/bin/env python3
"""Open-form answer scorer (pilot work order item 1.5).

服务双源五题(①⑦⑧⑨⑯,外加⑮的计数规则)的开放问答版判分。设计原则
(全案 2.2):能机器判的绝不交裁判;判不动的宁可记 invalid 也不蒙对。

四类判分器(按题目元数据的 answer_type 分派):

  angle_deg     数值角度。**环形角距离** d = min(|a-b|, 360-|a-b|);
                d <= THETA_FULL 满分 1.0,<= THETA_HALF 半分 0.5,否则 0。
                解析规则(保守,防撒网):优先取带角度记号(° / 度 /
                deg / degree)的数字;若无记号数字**恰好一个**则用它;
                否则 invalid。支持"左/右/left/right + 数字"的方向词
                换算(题面约定右为正,"左 85"→ −85);同时出现方向词
                与负号视为冲突 → invalid。
  time_s        数值时刻。绝对差;<= T_FULL 满分,<= T_HALF 半分。解析
                同上(记号:秒 / s / sec;"第 N 秒"的 N 同样计入候选,
                因此多数字 → invalid,防模型复述题干蹭数字)。
  closed_set    闭集枚举。同义词表归一后匹配;**最长词条优先**(词表把
                "不动/没动"登记为 still 的词条,天然压制"动"的子串误
                中——绝不做裸子串匹配);检测到两个不同枚举类的独立命中
                → invalid(冲突);无命中 → invalid。
  count_pair /  计数。从文本抽非负整数,数量与位数须与真值完全一致,
  count_single  逐个相等才满分(计数不设容差);多余数字 → invalid。

拒答三分(适用于带 refusal_truth 的可回答负样本题,如⑪/⑯的"已出画"):
  - 命中真值枚举(正确断言)→ 按闭集规则得分;
  - 命中含混拒答词表("无法判断/不知道/说不准/cannot tell/not sure")
    → 得 0 分,单独计入 abstention(弃答率独立指标,不给部分分);
  - 其余 → 按闭集规则(通常 0 分)。

带宽与词表全部显式参数:CLI 读 --params JSON(THETA_FULL/HALF、
T_FULL/T_HALF 必填,占位值也要写在参数文件里,不藏进代码);词表内置
默认(毛色/动静/左右/遮挡四态/拒答),可用 --vocab JSON 覆盖。逐题输出
解析明细与判定;manifest no-clobber;出现 invalid 不算错也不算对,单独
计率(invalid 率高说明解析器要调,不吞进准确率)。

research_candidate;不构成 dataset admission。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

# ---------- 默认同义词表(可被 --vocab 覆盖;matching 一律最长词条优先) ----------
DEFAULT_VOCAB: dict[str, dict[str, list[str]]] = {
    "coat": {
        "black_white": ["黑白毛", "黑白色", "黑白花", "黑白", "black and white", "black-and-white"],
        "yellow": ["黄毛", "黄色", "yellow"],
    },
    "motion": {
        "moving": ["正在走动", "在走动", "走动", "在移动", "移动", "动了", "在走", "moving", "walking"],
        "still": ["站着不动", "静止不动", "没有动", "站在原地", "不动", "没动", "未动", "静止",
                  "staying still", "not moving", "stationary", "still"],
    },
    "side": {
        "left": ["左侧", "左边", "左", "left"],
        "right": ["右侧", "右边", "右", "right"],
    },
    "occlusion4": {
        "visible_clear": ["清晰可见", "完整可见", "完全可见", "没有被挡", "没被挡", "fully visible", "clearly visible"],
        "visible_occluded": ["部分遮挡", "被部分挡住", "挡住一部分", "partially occluded", "partly occluded"],
        "fully_occluded": ["完全被遮挡", "完全挡住", "全被挡住", "看不到了但还在画面方向", "fully occluded", "completely occluded"],
        "out_of_view": ["已出画", "出画", "离开画面", "不在画面里", "走出画面", "out of view", "left the frame", "off-screen", "offscreen"],
    },
    "presence": {
        "offscreen_source": ["来自画面外", "画面外", "画外", "都不是", "不是画面里的", "none of them", "off-screen", "from outside the view"],
    },
}

ABSTAIN_TERMS = ["无法判断", "不知道", "说不准", "不确定", "无法确定", "难以判断",
                 "cannot tell", "can't tell", "not sure", "cannot determine", "unable to determine"]

SCORER_PARAM_KEYS = ("THETA_FULL", "THETA_HALF", "T_FULL", "T_HALF")
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
ANGLE_MARK = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(?:°|度|deg(?:ree)?s?)", re.IGNORECASE)
TIME_MARK = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(?:秒|s\b|sec(?:ond)?s?)", re.IGNORECASE)
LEFT_WORDS = ("左", "left")
RIGHT_WORDS = ("右", "right")


def circular_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _parse_numeric(text: str, mark: re.Pattern) -> tuple[float | None, str | None]:
    """返回 (数值, invalid原因)。带记号数字优先且须唯一;否则裸数字须唯一。"""
    marked = [float(m.group(1)) for m in mark.finditer(text)]
    if len(marked) == 1:
        return marked[0], None
    if len(marked) > 1:
        return None, f"multiple marked numbers: {marked}"
    bare = [float(m.group(0)) for m in _NUM.finditer(text)]
    if len(bare) == 1:
        return bare[0], None
    if not bare:
        return None, "no number found"
    return None, f"multiple bare numbers: {bare}"


def _apply_direction_words(text: str, value: float) -> tuple[float | None, str | None]:
    """题面约定右为正:'左 85' → −85。方向词与显式负号并存 → 冲突。"""
    has_left = any(w in text.lower() for w in LEFT_WORDS)
    has_right = any(w in text.lower() for w in RIGHT_WORDS)
    if has_left and has_right:
        return None, "both left and right words present"
    if not has_left and not has_right:
        return value, None
    if value < 0:
        return None, "direction word combined with an explicit negative sign"
    return (-value, None) if has_left else (value, None)


def score_angle(answer: str, truth_deg: float, theta_full: float, theta_half: float) -> dict:
    value, why = _parse_numeric(answer, ANGLE_MARK)
    if value is None:
        return {"status": "invalid", "reason": why, "score": 0.0}
    value, why = _apply_direction_words(answer, value)
    if value is None:
        return {"status": "invalid", "reason": why, "score": 0.0}
    err = circular_deg(value, truth_deg)
    score = 1.0 if err <= theta_full else 0.5 if err <= theta_half else 0.0
    return {"status": "scored", "parsed": value, "circular_error_deg": round(err, 2), "score": score}


def score_time(answer: str, truth_s: float, t_full: float, t_half: float) -> dict:
    value, why = _parse_numeric(answer, TIME_MARK)
    if value is None:
        return {"status": "invalid", "reason": why, "score": 0.0}
    err = abs(value - truth_s)
    score = 1.0 if err <= t_full else 0.5 if err <= t_half else 0.0
    return {"status": "scored", "parsed": value, "abs_error_s": round(err, 3), "score": score}


def _match_closed(answer: str, classes: dict[str, list[str]]) -> tuple[str | None, str | None]:
    """最长词条优先的闭集匹配。命中≥2 个不同枚举 → 冲突。"""
    text = answer.lower()
    hits: dict[str, tuple[int, str]] = {}
    for label, terms in classes.items():
        best = ""
        for term in terms:
            t = term.lower()
            if t in text and len(t) > len(best):
                best = t
        if best:
            hits[label] = (len(best), best)
    if not hits:
        return None, "no vocabulary hit"
    if len(hits) == 1:
        return next(iter(hits)), None
    # 多枚举命中:若某命中词条是另一命中词条的子串(如"动"⊂"没动"),弃子串
    labels = sorted(hits, key=lambda k: -hits[k][0])
    top_len, top_term = hits[labels[0]]
    survivors = [l for l in labels if not (hits[l][1] in top_term and hits[l][1] != top_term)]
    if len(survivors) == 1:
        return survivors[0], None
    return None, f"conflicting hits: { {l: hits[l][1] for l in survivors} }"


def score_closed(answer: str, truth_label: str, classes: dict[str, list[str]],
                 refusal_allowed: bool) -> dict:
    low = answer.lower()
    if any(t.lower() in low for t in ABSTAIN_TERMS):
        # 含混拒答:零分,单独计弃答(可回答负样本不给部分分)
        return {"status": "abstained", "score": 0.0}
    label, why = _match_closed(answer, classes)
    if label is None:
        return {"status": "invalid", "reason": why, "score": 0.0}
    return {"status": "scored", "parsed": label, "score": 1.0 if label == truth_label else 0.0}


def score_counts(answer: str, truth: list[int]) -> dict:
    nums = [int(float(m.group(0))) for m in _NUM.finditer(answer)]
    nums = [n for n in nums if n >= 0]
    if len(nums) != len(truth):
        return {"status": "invalid",
                "reason": f"expected {len(truth)} number(s), found {len(nums)}: {nums}",
                "score": 0.0}
    ok = all(a == b for a, b in zip(nums, truth))
    return {"status": "scored", "parsed": nums, "score": 1.0 if ok else 0.0}


def scorer_params(params: dict) -> dict:
    """Keep only parameters that this scorer actually executes."""
    missing = [key for key in SCORER_PARAM_KEYS if key not in params]
    if missing:
        raise ValueError(f"params missing explicit {missing}")
    return {key: params[key] for key in SCORER_PARAM_KEYS}


def score_item(item: dict, params: dict, vocab: dict) -> dict:
    """item: question_id, answer_type, model_answer, truth(类型相关), vocab_key?"""
    at = item["answer_type"]
    ans = str(item.get("model_answer", ""))
    if at == "angle_deg":
        return score_angle(ans, float(item["truth"]), params["THETA_FULL"], params["THETA_HALF"])
    if at == "time_s":
        return score_time(ans, float(item["truth"]), params["T_FULL"], params["T_HALF"])
    if at == "closed_set":
        classes = vocab[item["vocab_key"]]
        return score_closed(ans, str(item["truth"]), classes,
                            refusal_allowed=bool(item.get("refusal_truth", False)))
    if at == "count_pair" or at == "count_single":
        truth = item["truth"] if isinstance(item["truth"], list) else [item["truth"]]
        return score_counts(ans, [int(x) for x in truth])
    return {"status": "invalid", "reason": f"unknown answer_type {at!r}", "score": 0.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True, help="JSON 列表:题目+模型答案+真值")
    parser.add_argument("--params", required=True,
                        help="JSON:THETA_FULL/THETA_HALF/T_FULL/T_HALF(显式占位值)")
    parser.add_argument("--vocab", default=None, help="可选:覆盖默认同义词表的 JSON")
    parser.add_argument("--out", required=True, help="输出 manifest(no-clobber)")
    args = parser.parse_args(argv)

    if os.path.exists(args.out):
        print(f"refusing to overwrite existing output: {args.out}", file=sys.stderr)
        return 2
    try:
        params = scorer_params(json.load(open(args.params)))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    vocab: dict[str, Any] = dict(DEFAULT_VOCAB)
    if args.vocab:
        vocab.update(json.load(open(args.vocab)))
    items = json.load(open(args.items))

    records = []
    for item in items:
        rec = dict(question_id=item.get("question_id"), answer_type=item["answer_type"])
        rec.update(score_item(item, params, vocab))
        records.append(rec)

    n = len(records)
    scored = [r for r in records if r["status"] == "scored"]
    summary = {
        "schema": "avengine_open_answer_scores_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "parameters": params,
        "counts": {
            "total": n,
            "scored": len(scored),
            "invalid": sum(r["status"] == "invalid" for r in records),
            "abstained": sum(r["status"] == "abstained" for r in records),
        },
        "mean_score_over_all": round(sum(r["score"] for r in records) / n, 4) if n else None,
        "mean_score_over_scored": (round(sum(r["score"] for r in scored) / len(scored), 4)
                                   if scored else None),
        "records": records,
    }
    with open(args.out, "w") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=1)
    print(f"scored={summary['counts']}  mean_all={summary['mean_score_over_all']}  out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
