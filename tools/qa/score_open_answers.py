#!/usr/bin/env python3
"""Open-form answer scorer (pilot work order item 1.5).

服务双源五题(①⑦⑧⑨⑯,外加⑮的计数规则)的开放问答版判分。设计原则
(全案 2.2):能机器判的绝不交裁判;判不动的宁可记 invalid 也不蒙对。

四类判分器(按题目元数据的 answer_type 分派):

  angle_deg     数值角度。**环形角距离** d = min(|a-b|, 360-|a-b|);
                THETA_FULL/THETA_HALF first define the full and diagnostic
                two-tier regions. ANGLE_CERTIFICATION_POLICY=strict_full_credit_only
                gives certified score 1.0 only inside THETA_FULL; a half-credit
                match remains in diagnostic_two_tier_score. Legacy two-tier
                scoring keeps the old 0.5 score. truth_interval_deg, when
                present, is authoritative and uses point-to-arc distance.
                解析规则(保守,防撒网):优先取带角度记号(° / 度 /
                deg / degree)的数字;若无记号数字**恰好一个**则用它;
                否则 invalid。支持"左/右/left/right + 数字"的方向词,
                using the item's explicit convention. An item without one
                keeps the existing right-positive compatibility convention;
                同时出现方向词与负号视为冲突 → invalid。
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
T_FULL/T_HALF 必填,ANGLE_CERTIFICATION_POLICY 可选以兼容旧 params;
提供时必须是已知策略,不藏进代码);词表内置默认(毛色/动静/左右/遮挡
四态/拒答),可用 --vocab JSON 覆盖。逐题输出
解析明细与判定;manifest no-clobber;出现 invalid 不算错也不算对,单独
计率(invalid 率高说明解析器要调,不吞进准确率)。

research_candidate;不构成 dataset admission。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping
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

# Existing numeric questions that do not carry a convention used the scorer's
# right-positive interpretation. Keep that compatibility path while making
# an explicit convention authoritative whenever a new question supplies one.
ANGLE_CONVENTION_ALIASES = {
    "right_positive": "right_positive",
    "right_positive_deg": "right_positive",
    "engine_right_positive": "right_positive",
    "left_positive": "left_positive",
    "left_positive_deg": "left_positive",
    "dcase_foa_left_positive": "left_positive",
}
ANGLE_CERTIFICATION_POLICIES = {
    "legacy": "legacy_two_tier",
    "legacy_two_tier": "legacy_two_tier",
    "two_tier": "legacy_two_tier",
    "strict_full_credit_only": "strict_full_credit_only",
}
DEFAULT_ANGLE_CONVENTION = "right_positive"
DEFAULT_ANGLE_CERTIFICATION_POLICY = "legacy_two_tier"

_CONVENTION_MARKERS = (
    ("right side is positive", "right_positive"),
    ("right is positive", "right_positive"),
    ("positive values to the right", "right_positive"),
    ("left side is positive", "left_positive"),
    ("left is positive", "left_positive"),
    ("positive values to the left", "left_positive"),
)


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


def _canonical_angle_convention(value: Any) -> str | None:
    if value is None:
        return DEFAULT_ANGLE_CONVENTION
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return ANGLE_CONVENTION_ALIASES.get(normalized)


def _convention_from_text(text: str) -> tuple[str | None, str | None]:
    lowered = str(text).lower()
    found = {canonical for marker, canonical in _CONVENTION_MARKERS
             if marker in lowered}
    if len(found) > 1:
        return None, f"conflicting azimuth conventions in question: {sorted(found)}"
    if not found:
        return None, None
    return next(iter(found)), None


def _angle_convention_for_item(
    item: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Resolve an item's stated convention, preserving old numeric items."""
    candidates: list[tuple[str, str]] = []
    for key in ("convention", "azimuth_convention"):
        if key not in item or item[key] is None:
            continue
        canonical = _canonical_angle_convention(item[key])
        if canonical is None:
            return None, f"unknown azimuth convention {item[key]!r} in {key}"
        candidates.append((key, canonical))
    truth = item.get("truth")
    if isinstance(truth, Mapping) and truth.get("convention") is not None:
        canonical = _canonical_angle_convention(truth["convention"])
        if canonical is None:
            return None, (
                f"unknown azimuth convention {truth['convention']!r} in truth"
            )
        candidates.append(("truth.convention", canonical))

    stem = item.get("question") or item.get("stem") or item.get("prompt")
    if stem:
        stem_convention, reason = _convention_from_text(str(stem))
        if reason:
            return None, reason
        if stem_convention is not None:
            candidates.append(("question", stem_convention))

    if not candidates:
        # Legacy numerical items had no convention field and were scored with
        # the historical right-positive direction-word interpretation.
        return DEFAULT_ANGLE_CONVENTION, None
    unique = {canonical for _, canonical in candidates}
    if len(unique) > 1:
        return None, f"conflicting azimuth conventions: {sorted(unique)}"
    return next(iter(unique)), None


def _canonical_angle_policy(value: Any) -> str | None:
    return ANGLE_CERTIFICATION_POLICIES.get(str(value).strip().lower())


def resolve_angle_policy(
    params: Mapping[str, Any] | None = None,
    certification_policy: str | None = None,
) -> str:
    if certification_policy is not None:
        raw = certification_policy
    elif params is not None and "ANGLE_CERTIFICATION_POLICY" in params:
        raw = params["ANGLE_CERTIFICATION_POLICY"]
    else:
        return DEFAULT_ANGLE_CERTIFICATION_POLICY
    policy = _canonical_angle_policy(raw)
    if policy is None:
        raise ValueError(f"unknown ANGLE_CERTIFICATION_POLICY {raw!r}")
    return policy


def _validate_angle_tolerances(
    theta_full: Any,
    theta_half: Any,
) -> tuple[float, float]:
    try:
        full = float(theta_full)
        half = float(theta_half)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "THETA_FULL and THETA_HALF must be finite numbers"
        ) from exc
    if not math.isfinite(full) or not math.isfinite(half):
        raise ValueError("THETA_FULL and THETA_HALF must be finite numbers")
    if full < 0.0 or half < 0.0 or full > half:
        raise ValueError(
            "angle tolerances require 0 <= THETA_FULL <= THETA_HALF"
        )
    return full, half


def angle_credit_radius(
    params: Mapping[str, Any],
    certification_policy: str | None = None,
) -> float:
    """Return the angle radius that can receive certified credit.

    An explicit function argument overrides params; otherwise the declared
    ANGLE_CERTIFICATION_POLICY is used. Missing policy means legacy scoring for
    old parameter files, while an unknown declared value raises ValueError.
    """
    policy = resolve_angle_policy(params, certification_policy)
    if "THETA_FULL" not in params or "THETA_HALF" not in params:
        raise ValueError(
            "params must contain explicit THETA_FULL and THETA_HALF"
        )
    full, half = _validate_angle_tolerances(
        params["THETA_FULL"], params["THETA_HALF"])
    return full if policy == "strict_full_credit_only" else half


def _arc_from_truth_interval(value: Any):
    from qa_v3_arc import Arc

    if isinstance(value, Mapping):
        if value.get("schema") == "avengine_qa_v3_arc_v1":
            return Arc.from_dict(dict(value))
        if "start_deg" in value and "sweep_deg" in value:
            return Arc(float(value["start_deg"]), float(value["sweep_deg"]))
        raise ValueError(
            "truth_interval_deg object must be an avengine_qa_v3_arc_v1 arc"
        )
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "truth_interval_deg must be [lo, hi] or an avengine_qa_v3_arc_v1 arc"
        )
    lo, hi = (float(v) for v in value)
    if hi < lo:
        # The ordered pair can represent the forward arc across +-180.  Keep
        # the orientation in Arc instead of sorting the endpoints into its
        # complement.
        return Arc(start_deg=lo, sweep_deg=(hi - lo) % 360.0)
    return Arc.from_bounds(lo, hi)


def _distance_to_arc(point_deg: float, arc) -> float:
    if arc.contains(float(point_deg)):
        return 0.0
    return min(circular_deg(point_deg, arc.start_deg),
               circular_deg(point_deg, arc.end_deg))


def _apply_direction_words(
    text: str,
    value: float,
    *,
    convention: str = DEFAULT_ANGLE_CONVENTION,
) -> tuple[float | None, str | None]:
    """Apply left/right words in the item's explicit azimuth convention."""
    has_left = any(w in text.lower() for w in LEFT_WORDS)
    has_right = any(w in text.lower() for w in RIGHT_WORDS)
    if has_left and has_right:
        return None, "both left and right words present"
    if not has_left and not has_right:
        return value, None
    if value < 0:
        return None, "direction word combined with an explicit negative sign"
    if has_left:
        return (value, None) if convention == "left_positive" else (-value, None)
    return (-value, None) if convention == "left_positive" else (value, None)


def score_angle(
    answer: str,
    truth_deg: float,
    theta_full: float,
    theta_half: float,
    *,
    certification_policy: str | None = None,
    convention: str | None = None,
    truth_interval_deg: Any = None,
) -> dict:
    canonical_convention = _canonical_angle_convention(convention)
    if canonical_convention is None:
        return {
            "status": "invalid",
            "reason": f"unknown azimuth convention {convention!r}",
            "score": 0.0,
        }
    canonical_policy = resolve_angle_policy(
        None, certification_policy
    )
    theta_full, theta_half = _validate_angle_tolerances(
        theta_full, theta_half)
    value, why = _parse_numeric(answer, ANGLE_MARK)
    if value is None:
        return {"status": "invalid", "reason": why, "score": 0.0}
    value, why = _apply_direction_words(
        answer, value, convention=canonical_convention
    )
    if value is None:
        return {"status": "invalid", "reason": why, "score": 0.0}
    try:
        if truth_interval_deg is None:
            err = circular_deg(value, float(truth_deg))
            truth_mode = "point"
        else:
            err = _distance_to_arc(
                value, _arc_from_truth_interval(truth_interval_deg))
            truth_mode = "interval"
    except (TypeError, ValueError) as exc:
        return {"status": "invalid", "reason": str(exc), "score": 0.0}
    diagnostic = 1.0 if err <= theta_full else 0.5 if err <= theta_half else 0.0
    if canonical_policy == "strict_full_credit_only":
        score = 1.0 if err <= theta_full else 0.0
    else:
        score = diagnostic
    result = {
        "status": "scored",
        "parsed": value,
        "circular_error_deg": round(err, 2),
        "score": score,
        "angle_convention": canonical_convention,
        "truth_mode": truth_mode,
    }
    if canonical_policy == "strict_full_credit_only":
        result.update({
            "certification_policy": "strict_full_credit_only",
            "diagnostic_two_tier_score": diagnostic,
        })
    return result


def score_time(answer: str, truth_s: float, t_full: float, t_half: float, *,
               strict_certification: bool = False) -> dict:
    value, why = _parse_numeric(answer, TIME_MARK)
    if value is None:
        return {"status": "invalid", "reason": why, "score": 0.0}
    err = abs(value - truth_s)
    diagnostic = 1.0 if err <= t_full else 0.5 if err <= t_half else 0.0
    score = (1.0 if err <= t_full else 0.0) if strict_certification else diagnostic
    result = {"status": "scored", "parsed": value,
              "abs_error_s": round(err, 3), "score": score}
    if strict_certification:
        result.update({
            "certification_policy": "strict_full_credit_only",
            "diagnostic_two_tier_score": diagnostic,
        })
    return result


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
    """Keep parameters that this scorer executes, including declared angle policy."""
    missing = [key for key in SCORER_PARAM_KEYS if key not in params]
    if missing:
        raise ValueError(f"params missing explicit {missing}")
    _validate_angle_tolerances(
        params["THETA_FULL"], params["THETA_HALF"])
    result = {key: params[key] for key in SCORER_PARAM_KEYS}
    if "ANGLE_CERTIFICATION_POLICY" in params:
        result["ANGLE_CERTIFICATION_POLICY"] = resolve_angle_policy(params)
    return result


def score_item(item: dict, params: dict, vocab: dict) -> dict:
    """item: question_id, answer_type, model_answer, truth(类型相关), vocab_key?"""
    at = item["answer_type"]
    ans = str(item.get("model_answer", ""))
    if at == "angle_deg":
        raw_truth = item.get("truth")
        interval = item.get("truth_interval_deg")
        if interval is not None:
            if isinstance(raw_truth, Mapping):
                truth_value = next(
                    (raw_truth[key] for key in
                     ("azimuth_deg", "final_azimuth_deg", "truth_value", "value")
                     if key in raw_truth),
                    0.0,
                )
            else:
                truth_value = 0.0 if raw_truth is None else raw_truth
        else:
            if raw_truth is None:
                return {
                    "status": "invalid",
                    "reason": "angle item missing truth",
                    "score": 0.0,
                }
            if isinstance(raw_truth, Mapping):
                truth_value = next(
                    (raw_truth[key] for key in
                     ("azimuth_deg", "final_azimuth_deg", "truth_value", "value")
                     if key in raw_truth),
                    None,
                )
                if truth_value is None:
                    return {
                        "status": "invalid",
                        "reason": "angle truth object has no numeric azimuth value",
                        "score": 0.0,
                    }
            else:
                truth_value = raw_truth
        convention, reason = _angle_convention_for_item(item)
        if reason:
            return {"status": "invalid", "reason": reason, "score": 0.0}
        params_has_policy = "ANGLE_CERTIFICATION_POLICY" in params
        params_policy = (
            resolve_angle_policy(params) if params_has_policy else None)
        item_has_policy = "certification_policy" in item
        item_policy = (
            resolve_angle_policy({}, item["certification_policy"])
            if item_has_policy and item["certification_policy"] is not None
            else None)
        if item_policy is not None and params_policy is not None:
            if item_policy != params_policy:
                return {
                    "status": "invalid",
                    "reason": (
                        "angle certification policy conflict: "
                        f"item={item_policy!r}, params={params_policy!r}"
                    ),
                    "score": 0.0,
                }
        policy = item_policy if item_policy is not None else params_policy
        return score_angle(
            ans,
            float(truth_value),
            params["THETA_FULL"],
            params["THETA_HALF"],
            certification_policy=policy,
            convention=convention,
            truth_interval_deg=interval,
        )
    if at == "time_s":
        return score_time(
            ans, float(item["truth"]), params["T_FULL"], params["T_HALF"],
            strict_certification=(
                item.get("certification_policy") ==
                "strict_full_credit_only"))
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
