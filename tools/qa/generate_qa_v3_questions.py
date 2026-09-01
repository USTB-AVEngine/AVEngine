#!/usr/bin/env python3
"""Generate qa-v3 pilot fact records + question candidates (cards ①⑦⑧⑨).

从设计批(timeline/program/plan/filter_report)生成四张卡的题目候选与
事实记录。**真值全部来自引擎侧记录**(时间线几何 + 事件表),不经过任何
模型;每条 fact record 携带来源链(输入文件 sha256、filter 判定、可见性
判定口径)。

卡⑯(两跳遮挡)不在本生成器:像素遮挡真值需要掩码通道(工单 1.8),
本批捕获只有 rgb;卡①⑦ 的可见性此处按几何视锥判定,fact record 记
`visibility_source=geometry_fov_only_pixel_check_pending`,掩码通道接入后
逐题像素复核。

输出(--out-root,fresh/no-clobber):
  facts_card1.jsonl / facts_card7.jsonl / facts_card8.jsonl / facts_card9.jsonl
  generation_manifest.json

题目形式:每条 fact 同时声明 MCQ 与开放两形式的题面与真值(共享事实、
分别认证);MCQ 选项**顺序不在此处编排**——正确位置轮转必须在 split 之后
由 build_mcq_options.py 做,这里只给候选集与真值。

research_candidate;qualification_claim=false;配额裁剪是显式未决参数,
本工具输出全量候选并在 manifest 里记 quota_note。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filter_cross_time_points import (  # noqa: E402
    FRAME_COUNT,
    azimuth_deg,
    circ_diff,
)

AUDIO_SAMPLE_RATE = 16000
TICKS_PER_SAMPLE = 3
TICKS_PER_FRAME = 3200
VIDEO_FPS = 15

# 外观词表:与 score_open_answers.DEFAULT_VOCAB 的 coat 词面一致
COAT_WORDS = {
    "generated_border_collie_black_white_medium_standard_adult_research_v1":
        "black-and-white",
    "generated_labrador_yellow_medium_standard_adult_research_v1": "yellow",
}
COAT_LABEL = {
    "black-and-white": "the black-and-white dog",
    "yellow": "the yellow dog",
}

CARD7_OPTIONS = ("black-and-white", "yellow", "both", "neither")

CARD9_STEMS = (
    "Which dog barked first, the black-and-white one or the yellow one?",
    "Which of the two dogs was the first to make a sound?",
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_pick(seed: str, items: list):
    """sha256 派生的确定性抽取(禁 builtin hash,PYTHONHASHSEED 陷阱)。"""
    if not items:
        raise ValueError("stable_pick from empty list")
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return items[int.from_bytes(digest[:8], "big") % len(items)]


def event_frame_span(event: dict) -> tuple[int, int]:
    """事件覆盖的帧域 [start, end);tick = sample*3,frame = tick//3200。"""
    t0 = event["start_sample"] * TICKS_PER_SAMPLE
    t1 = event["end_sample_exclusive"] * TICKS_PER_SAMPLE
    f0 = t0 // TICKS_PER_FRAME
    f1 = (t1 + TICKS_PER_FRAME - 1) // TICKS_PER_FRAME
    return max(0, f0), min(FRAME_COUNT, f1)


class PointBundle:
    """一个设计点的引擎侧事实视图(timeline + program + plan)。"""

    def __init__(self, pdir: Path, programs_dir: Path):
        self.pdir = pdir
        self.point_id = pdir.name
        self.spec = json.loads((pdir / "spec.json").read_text())
        self.timeline = json.loads((pdir / "timeline.json").read_text())
        prog_matches = sorted(programs_dir.glob(
            f"qa_v3_*_{self.point_id}_rand_v1.json"))
        if len(prog_matches) != 1:
            raise ValueError(f"{self.point_id}: expected one main program, "
                             f"found {len(prog_matches)}")
        self.program_path = prog_matches[0]
        self.program = json.loads(self.program_path.read_text())
        self.plan_path = programs_dir / (self.program_path.stem + ".plan.json")
        self.plan = json.loads(self.plan_path.read_text())
        sel = json.loads((pdir / "actor_selection.json").read_text())
        self.slot_asset = {a["source_slot_id"]: a["asset_id"]
                          for a in sel["actors"]}
        self.slot_coat = {s: COAT_WORDS[a] for s, a in self.slot_asset.items()}
        ep_ids = list(self.program["candidate_source_endpoint_ids"])
        # 事件 endpoint → slot:装配器按 [slot1, slot2] 顺序写候选
        self.ep_slot = {ep_ids[0]: "source1", ep_ids[1]: "source2"}
        self.hfov = float(self.timeline["render"]["hfov_degrees"])
        self.frames = self.timeline["frames"]

    def cam(self, f: int) -> tuple[tuple[float, float], float]:
        c = self.frames[f]["camera"]
        t = c["translation_ue_cm"]
        return (float(t[0]), float(t[1])), float(c["yaw_ue_deg"])

    def actor_xy(self, f: int, slot: str) -> tuple[float, float]:
        for st in self.frames[f]["actor_states"]:
            if st["source_slot_id"] == slot:
                t = st["translation_ue_cm"]
                return (float(t[0]), float(t[1]))
        raise ValueError(f"slot {slot} missing at frame {f}")

    def azimuth(self, f: int, slot: str) -> float:
        (cxy, yaw) = self.cam(f)
        return azimuth_deg(cxy, yaw, self.actor_xy(f, slot))

    def in_fov(self, f: int, slot: str) -> bool:
        return abs(self.azimuth(f, slot)) <= self.hfov / 2.0

    def calling_slots(self, f: int) -> set[str]:
        out = set()
        for e in self.program["events"]:
            f0, f1 = event_frame_span(e)
            if f0 <= f < f1:
                out.add(self.ep_slot[e["source_endpoint_id"]])
        return out

    def first_onsets(self) -> dict[str, float]:
        onsets: dict[str, float] = {}
        for e in sorted(self.program["events"], key=lambda x: x["start_sample"]):
            slot = self.ep_slot[e["source_endpoint_id"]]
            onsets.setdefault(slot, e["start_sample"] / AUDIO_SAMPLE_RATE)
        return onsets

    def provenance(self) -> dict:
        return {
            "timeline_sha256": sha256_path(self.pdir / "timeline.json"),
            "program_sha256": sha256_path(self.program_path),
            "plan_sha256": sha256_path(self.plan_path),
            "actor_selection_sha256": sha256_path(
                self.pdir / "actor_selection.json"),
            "visibility_source": "geometry_fov_only_pixel_check_pending",
        }


def base_record(bundle: PointBundle, card: str, filter_entry: dict) -> dict:
    return {
        "schema": "qa_v3_fact_record_v1",
        "card": card,
        "point_id": bundle.point_id,
        "episode_id": bundle.point_id,
        "twin_group": bundle.spec.get("twin_of") or bundle.point_id,
        "motion_class": bundle.spec.get("motion_class"),
        "slot_coat": bundle.slot_coat,
        "filter_admit": {k: v.get("admit") for k, v in filter_entry.items()
                        if isinstance(v, dict)},
        "provenance": bundle.provenance(),
        "status": "research_candidate",
        "qualification_claim": False,
    }


def gen_card1(bundle: PointBundle, fe: dict, params: dict) -> dict | None:
    if not fe.get("card1", {}).get("admit"):
        return None
    slot = bundle.plan["anchor_slot"]
    coat = bundle.slot_coat[slot]
    truth_deg = bundle.azimuth(FRAME_COUNT - 1, slot)
    other = "source1" if slot == "source2" else "source2"
    rec = base_record(bundle, "card1", fe)
    rec.update({
        "anchor": {"slot": slot, "kind": "last_bark",
                   "start_sample": bundle.plan["anchor_start_sample"],
                   "anchor_frame": (bundle.plan["anchor_start_sample"]
                                    * TICKS_PER_SAMPLE) // TICKS_PER_FRAME},
        "truth": {"final_azimuth_deg": round(truth_deg, 2),
                  "convention": "right_positive_deg",
                  "other_slot_final_azimuth_deg": round(
                      bundle.azimuth(FRAME_COUNT - 1, other), 2)},
        "mcq": _card1_band_mcq(truth_deg, params),
        "azimuth_band": _azimuth_band_block(truth_deg, params),
        "mcq_four_sector_deprecated": {
            "options_space": ["front", "right", "back", "left"],
            "truth_option": sector_name(truth_deg),
            "why_deprecated": (
                "run01 证伪:目标片尾必须可见 ⇒ 方位恒在相机视锥内 ⇒ 只有"
                "'前'扇区可达,40/40 被编排器上游复检拒出。保留仅作诊断。")},
        "open": {"stem": ("Roughly how many degrees from your facing "
                          "direction is the dog that barked last, at the end "
                          "of the video? Right side is positive."),
                 "truth_value": round(truth_deg, 2), "unit": "deg",
                 "scoring": "circular_deg"},
        "referral_coat_note": coat,
    })
    return rec


def _card1_band_mcq(truth_deg: float, params: dict) -> dict | None:
    """card1 的选择题形态(run02 起):预先声明的视锥内方位带。"""
    blk = _azimuth_band_block(truth_deg, params)
    if not blk or not blk.get("options_space"):
        return None
    return {"stem": ("At the end of the video, which azimuth band relative to "
                     "your facing direction is the dog that barked LAST in? "
                     "Right side is positive; the bands are given in degrees."),
            "options_space": blk["options_space"],
            "truth_option": blk["truth_option"],
            "band_index": blk["band_index"],
            "answer_space_note": (
                "bands are equal-width bins of the reachable azimuth window; "
                "the window's right edge is set by the joint requirement that "
                "the target stay visible at the end while still moving in "
                "azimuth after the anchor")}


def _azimuth_band_block(truth_deg: float, params: dict) -> dict | None:
    """预先声明的视锥等分方位带(run02 起的 card1 MCQ 答案空间)。

    四扇区(前/右/后/左)被 run01 证伪:目标片尾必须可见 ⇒ 方位恒在
    视锥内 ⇒ 只有"前"扇区可达,40/40 被编排器上游复检拒出。视锥等分带
    的答案空间是相机视野的属性,与走廊弦库无关。
    """
    edges = params.get("AZ_BANDS_CARD1")
    if not edges:
        return None
    idx = None
    for i in range(len(edges) - 1):
        if edges[i] <= truth_deg < edges[i + 1]:
            idx = i
            break
    if idx is None and abs(truth_deg - edges[-1]) < 1e-9:
        idx = len(edges) - 2
    if idx is None:
        return {"options_space": None, "truth_option": None,
                "note": f"final azimuth {truth_deg:.2f} outside declared bands"}
    labels = [f"[{edges[i]:g}, {edges[i + 1]:g})"
              for i in range(len(edges) - 1)]
    return {"options_space": labels, "truth_option": labels[idx],
            "band_index": idx, "edges": list(edges)}


def sector_name(deg: float) -> str:
    if -45.0 <= deg < 45.0:
        return "front"
    if 45.0 <= deg < 135.0:
        return "right"
    if -135.0 <= deg < -45.0:
        return "left"
    return "back"


def gen_card7(bundle: PointBundle, fe: dict, params: dict,
              negative: bool) -> dict | None:
    if not fe.get("card7", {}).get("admit"):
        return None
    sep = float(params["MIN_AZIMUTH_SEP"])
    ok_frames = []
    for f in range(FRAME_COUNT):
        calling = bundle.calling_slots(f)
        want = (len(calling) == 0) if negative else (len(calling) == 1)
        if not want:
            continue
        if negative and f >= bundle.plan["anchor_start_sample"] * \
                TICKS_PER_SAMPLE // TICKS_PER_FRAME:
            continue  # 负样本帧不取锚后尾静默(那是卡①的构造段)
        if f % 3 != 0:
            continue  # t=f/15 保持一位小数(0.2s 粒度)
        if not (bundle.in_fov(f, "source1") and bundle.in_fov(f, "source2")):
            continue
        gap = circ_diff(bundle.azimuth(f, "source1"),
                        bundle.azimuth(f, "source2"))
        if gap < sep:
            continue
        ok_frames.append(f)
    if not ok_frames:
        return None
    f = stable_pick(f"card7|{bundle.point_id}|{negative}", ok_frames)
    calling = bundle.calling_slots(f)
    truth = ("neither" if not calling
             else bundle.slot_coat[next(iter(calling))])
    t_s = f / VIDEO_FPS
    rec = base_record(bundle, "card7", fe)
    rec.update({
        "query_time": {"frame": f, "second": round(t_s, 4),
                       "stem_second": f"{t_s:.1f}"},
        "negative_sample": negative,
        "truth": {"calling_at_t": truth},
        "mcq": {"stem": (f"At {t_s:.1f} seconds on the video clock, which "
                         f"dog is barking?"),
                "options_space": list(CARD7_OPTIONS),
                "truth_option": truth},
        "open": {"stem": (f"Which dog, if any, is barking at {t_s:.1f} "
                          f"seconds — the black-and-white one, the yellow "
                          f"one, both, or neither?"),
                 "truth_value": truth, "scoring": "closed_set"},
    })
    return rec


def gen_card8(bundle: PointBundle, fe: dict, params: dict) -> list[dict]:
    if not fe.get("card8", {}).get("admit"):
        return []
    # MCQ 带用**预先声明**的 BANDS_CARD8(run02 起由装配器带优先调度
    # 填满,答案带按构造均匀);缺省回退 BANDS。
    bands_key = "BANDS_CARD8" if "BANDS_CARD8" in params else "BANDS"
    bands = [float(b) for b in params[bands_key]]
    out = []
    for slot, onset in sorted(bundle.first_onsets().items()):
        coat = bundle.slot_coat[slot]
        try:
            band_idx = band_of(onset, bands)
            mcq = {"stem": (f"When does {COAT_LABEL[coat]} bark for the "
                            f"FIRST time? Pick the time range (seconds, "
                            f"video clock)."),
                   "options_space": [
                       f"[{bands[i]:g}, {bands[i + 1]:g})"
                       for i in range(len(bands) - 1)],
                   "truth_option": (f"[{bands[band_idx]:g}, "
                                    f"{bands[band_idx + 1]:g})"),
                   "bands_key": bands_key}
        except ValueError:
            band_idx, mcq = None, None   # 越带:MCQ 缺席,开放版仍出
        rec = base_record(bundle, "card8", fe)
        rec.update({
            "target_slot": slot,
            "truth": {"first_onset_s": round(onset, 4),
                      "band_index": band_idx,
                      "band_bounds": (None if band_idx is None else
                                      [bands[band_idx], bands[band_idx + 1]])},
            "mcq": mcq,
            "open": {"stem": (f"At how many seconds into the video does "
                              f"{COAT_LABEL[coat]} bark for the first "
                              f"time?"),
                     "truth_value": round(onset, 4), "unit": "s",
                     "scoring": "absolute_time",
                     "certification_policy": "strict_full_credit_only",
                     "wide_tolerance_role": "diagnostic_only"},
        })
        out.append(rec)
    return out


def band_of(value: float, bands: list[float]) -> int:
    for i in range(len(bands) - 1):
        if bands[i] <= value < bands[i + 1]:
            return i
    raise ValueError(f"onset {value} outside bands {bands}")


def gen_card9(bundle: PointBundle, fe: dict) -> dict | None:
    if not fe.get("card9", {}).get("admit"):
        return None
    onsets = bundle.first_onsets()
    first_slot = min(onsets, key=onsets.get)
    truth = bundle.slot_coat[first_slot]
    stem = stable_pick(f"card9|{bundle.point_id}", list(CARD9_STEMS))
    rec = base_record(bundle, "card9", fe)
    rec.update({
        "truth": {"first_to_bark": truth,
                  "onset_gap_s": round(abs(
                      onsets["source1"] - onsets["source2"]), 4)},
        "mcq": {"stem": stem,
                "stem_variants": list(CARD9_STEMS),
                "options_space": ["black-and-white", "yellow"],
                "truth_option": truth},
        "open": {"stem": stem, "truth_value": truth,
                 "scoring": "closed_set"},
    })
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-root", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--card7-negative-share", type=float, default=0.2,
                        help="卡⑦负样本(都没叫)目标占比,显式参数")
    args = parser.parse_args(argv)

    if args.out_root.exists():
        print(f"refusing to overwrite existing out root: {args.out_root}",
              file=sys.stderr)
        return 2
    params = json.loads(args.params.read_text())
    freport = json.loads(
        (args.design_root / "filter_report.json").read_text())["results"]
    programs_dir = args.design_root / "programs"

    args.out_root.mkdir(parents=True)
    # 卡⑦按工单拆批(codex 审阅裁定):主集只出"恰好一只在叫",
    # "都没叫"是音频充分对照,单独成批、单独统计,不混进一个分布。
    # "都在叫"对照需要事件重叠(schema 的 simultaneous_subset 模式),
    # 本生成器的 sequential_sources program 造不出,不在本批。
    facts: dict[str, list[dict]] = {"card1": [], "card7_main": [],
                                    "card7_control_neither": [],
                                    "card8": [], "card9": []}
    skipped: list[str] = []
    # 主点(排除孪生:孪生用于 Gate B 对照,不进主题池)
    points = sorted(p for p in args.design_root.iterdir()
                    if p.is_dir() and p.name in freport
                    and "_tw" not in p.name)
    # 卡⑦负样本配额:按 sha 序取前 n 个点(确定性、份额精确、不聚在批首)
    n_neg_target = int(round(len(points) * args.card7_negative_share))
    neg_rank = sorted(points, key=lambda p: hashlib.sha256(
        f"c7neg|{p.name}".encode()).hexdigest())
    negative_points = {p.name for p in neg_rank[:n_neg_target]}
    neg_used = 0
    for pdir in points:
        try:
            bundle = PointBundle(pdir, programs_dir)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            skipped.append(f"{pdir.name}: {exc}")
            continue
        fe = freport[pdir.name]
        r1 = gen_card1(bundle, fe, params)
        if r1:
            facts["card1"].append(r1)
        want_negative = pdir.name in negative_points
        r7 = gen_card7(bundle, fe, params, negative=want_negative)
        if r7 is None and want_negative:
            r7 = gen_card7(bundle, fe, params, negative=False)
        if r7:
            key = ("card7_control_neither" if r7["negative_sample"]
                   else "card7_main")
            facts[key].append(r7)
            if r7["negative_sample"]:
                neg_used += 1
        facts["card8"].extend(gen_card8(bundle, fe, params))
        r9 = gen_card9(bundle, fe)
        if r9:
            facts["card9"].append(r9)

    # 均衡子集的标记不在这里做:codex 审阅指出全局 50:50 不够 ——
    # run01 的均衡子集内按运动类猜多数类仍有 60.6%。分层至少要到
    # split × motion_class × truth,而 split 在下游才分配,故标记
    # 移交 prepare_qa_v3_mcq.py(切分之后),这里不留第二个真相源。
    for card, records in facts.items():
        with open(args.out_root / f"facts_{card}.jsonl", "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "schema": "qa_v3_question_generation_manifest_v1",
        "design_root": str(args.design_root),
        "params_sha256": sha256_path(args.params),
        "parameters": params,
        "card7_negative_share": args.card7_negative_share,
        "counts": {card: len(records) for card, records in facts.items()},
        "card7_negatives": neg_used,
        "card7_views": {
            "main": "exactly one dog calling (main set)",
            "control_neither": "no dog calling (audio-sufficient control, "
                               "counted and reported separately)",
            "control_both": "not built in this batch: overlapping events need "
                            "the simultaneous_subset program mode",
        },
        "card16_note": ("card16 (two-hop occlusion) requires the pixel mask "
                        "channel (work-order item 1.8); not generated in "
                        "this batch"),
        "quota_note": ("full candidate pool emitted; per-card quotas remain "
                       "explicit undecided parameters (owner decision)"),
        "visibility_note": ("card1/card7 visibility judged by geometric FOV "
                            "only; pixel-level occlusion re-check pending "
                            "mask channel"),
        "card1_feasibility_finding": (
            "pilot01 量化:固定审阅相机(yaw -145°, hfov 105°)+走廊弦库+"
            "片尾可见约束 ⇒ card1 真值方位实测 [-31°,+8°](左右侧点数均衡"
            "21R/19L 但右侧幅值 ≤8°);四扇区 MCQ 退化为全 front,三带下"
            "right 空。根因与全案已知'±60–105° 近乎空'相同——方位可行域由"
            "相机与弦库几何决定。扩相机 pose 或扩弦库是 owner 决策项;"
            "本批 card1 两形式均受此限,证据入 pilot 报告"),
        "skipped": skipped,
        "status": "research_candidate",
        "qualification_claim": False,
    }
    (args.out_root / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"out": str(args.out_root),
                      "counts": manifest["counts"],
                      "card7_negatives": neg_used,
                      "skipped": len(skipped)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
