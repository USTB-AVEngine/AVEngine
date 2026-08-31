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


def gen_card1(bundle: PointBundle, fe: dict) -> dict | None:
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
        "mcq": {"stem": ("At the end of the video, in which direction from "
                         "you is the dog that barked LAST? Front is "
                         "[-45°,45°), right is [45°,135°), back and left are "
                         "the mirror sectors."),
                "options_space": ["front", "right", "back", "left"],
                "truth_option": sector_name(truth_deg),
                "degeneracy_note": (
                    "pilot01: 固定审阅相机(hfov 105°)+片尾须可见 ⇒ 真值方位"
                    "可行域实测 [-31°,+8°],四扇区退化(全 front);三带备选"
                    "见 three_band_visible;选项空间定版是 owner 决策项")},
        "three_band_visible": {
            "options_space": ["left_of_-15", "within_±15", "right_of_+15"],
            "truth_option": ("within_±15" if -15.0 <= truth_deg < 15.0
                             else ("right_of_+15" if truth_deg >= 15.0
                                   else "left_of_-15"))},
        "open": {"stem": ("Roughly how many degrees from your facing "
                          "direction is the dog that barked last, at the end "
                          "of the video? Right side is positive."),
                 "truth_value": round(truth_deg, 2), "unit": "deg",
                 "scoring": "circular_deg"},
        "referral_coat_note": coat,
    })
    return rec


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
    # MCQ 带优先用实测可行域带(BANDS_CARD8_MCQ,pilot01 发现:锚尾静默把
    # 首叫压到 ≤2.6s,原全域四带后两带结构性空);缺省回退 BANDS。
    bands_key = "BANDS_CARD8_MCQ" if "BANDS_CARD8_MCQ" in params else "BANDS"
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
                     "scoring": "absolute_time"},
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
    facts: dict[str, list[dict]] = {"card1": [], "card7": [], "card8": [],
                                    "card9": []}
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
        r1 = gen_card1(bundle, fe)
        if r1:
            facts["card1"].append(r1)
        want_negative = pdir.name in negative_points
        r7 = gen_card7(bundle, fe, params, negative=want_negative)
        if r7 is None and want_negative:
            r7 = gen_card7(bundle, fe, params, negative=False)
        if r7:
            facts["card7"].append(r7)
            if r7["negative_sample"]:
                neg_used += 1
        facts["card8"].extend(gen_card8(bundle, fe, params))
        r9 = gen_card9(bundle, fe)
        if r9:
            facts["card9"].append(r9)

    # 外观均衡子集标记(card7 正样本 / card9):少数类全保,多数类 sha 序
    # 等量抽取;认证时可并行跑全量与 balanced 两个视图(先验偏斜的对照)。
    def mark_balanced(records: list[dict], truth_key) -> None:
        groups: dict[str, list[dict]] = {}
        for rec in records:
            rec["balanced_subset"] = False
            groups.setdefault(truth_key(rec), []).append(rec)
        coats = [g for k, g in groups.items()
                 if k in ("black-and-white", "yellow")]
        if len(coats) == 2:
            n = min(len(g) for g in coats)
            for g in coats:
                keep = sorted(g, key=lambda r: hashlib.sha256(
                    f"bal|{r['point_id']}|{r['card']}".encode()).hexdigest())[:n]
                for rec in keep:
                    rec["balanced_subset"] = True
        for k, g in groups.items():
            if k not in ("black-and-white", "yellow"):
                for rec in g:      # 负样本(neither)整组保留
                    rec["balanced_subset"] = True

    mark_balanced(facts["card7"], lambda r: r["truth"]["calling_at_t"])
    mark_balanced(facts["card9"], lambda r: r["truth"]["first_to_bark"])

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
