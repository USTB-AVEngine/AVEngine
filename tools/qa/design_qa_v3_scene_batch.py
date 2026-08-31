#!/usr/bin/env python3
"""Integrated qa-v3 batch: generic scene solver + per-type audio + facts.

把三样东西接进同一条链:场景无关几何求解器(scene_sampler)、题型专用
音频调度(audio_profiles)、以及从**最终时间线**重算的题目事实。

链路:
    场景输入(导航路线库 + 相机基准请求)
      → 逐格分配答案(方位带 / 角色绑定 / 首叫角色)
      → 几何求解(解 yaw 落带,不枚举)
      → 相机与听者同一姿态结果(apply_camera_listener_pose_ue)
      → 题型 AudioProgram(语义角色 → 槽位绑定)
      → 时间线创作(相机与折线来自求解结果)
      → **在最终相机姿态下重算真值**,与分配的答案格逐条核对
      → 事实记录(MCQ 与 Open 引用同一条事实)

边界:没有像素证据之前一律标 geometry_candidate,不是题目准入。
房间 ID、固定路线名、固定 yaw 一概不出现;场景由 --scene-config 给。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import audio_profiles as AP  # noqa: E402
import scene_sampler as SS  # noqa: E402
from build_qa_v3_programs import build_program  # noqa: E402
# 选角文档的结构(蓝图/网格/动画的物理来源、UE 绑定)已在既有装配器里
# 验证过,直接复用它的构造函数,不在这里重写一份容易走样的副本。
from design_qa_v3_pilot_batch import _selection_doc  # noqa: E402
# 静→走用与旧管线**同一个**变换:创作函数按弧长把整条路线铺满 75 帧,
# 那是"压缩式";求解器用的是保速的"平移式"。两者不一致会让中途帧的
# 位置对不上 —— 集成冒烟里正是反向题(查询帧在中途)先露馅。
from make_idle_then_walk_timeline import transform_idle_then_walk  # noqa: E402
from avengine.camera_pose import apply_camera_listener_pose_ue  # noqa: E402
from avengine.dataset.apartment_dynamic_audio import (  # noqa: E402
    apartment_ue_point_to_world_m,
)
from avengine.timeline.current_apartment_visual import (  # noqa: E402
    author_current_apartment_visual_timeline,
)

COAT_WORDS = {
    "generated_border_collie_black_white_medium_standard_adult_research_v1":
        "black-and-white",
    "generated_labrador_yellow_medium_standard_adult_research_v1": "yellow",
}
EP_MAP = {
    "generated_border_collie_black_white_medium_standard_adult_research_v1":
        ("qa_v2_dog_1_collie_muzzle", "qa_v2_dog_2_collie_muzzle"),
    "generated_labrador_yellow_medium_standard_adult_research_v1":
        ("qa_v2_dog_1_labrador_muzzle", "qa_v2_dog_2_labrador_muzzle"),
}


def sha_rng(*parts) -> np.random.Generator:
    tag = "|".join(str(p) for p in parts).encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(tag).digest()[:8], "big") % 2**32)


def balanced(values, n, *seed_parts):
    reps = -(-n // len(values))
    pool = (list(values) * reps)[:n]
    order = sha_rng(*seed_parts).permutation(n)
    return [pool[int(i)] for i in order]


def recompute_azimuth(timeline, slot, frame):
    """从**最终时间线**重算方位:相机姿态已经应用,不能沿用旋转前的角度。"""
    record = timeline["frames"][frame]
    camera = record["camera"]
    cam_xy = (float(camera["translation_ue_cm"][0]),
              float(camera["translation_ue_cm"][1]))
    yaw = float(camera["yaw_ue_deg"])
    for state in record["actor_states"]:
        if state["source_slot_id"] == slot:
            xy = (float(state["translation_ue_cm"][0]),
                  float(state["translation_ue_cm"][1]))
            return SS.relative_azimuth_deg(cam_xy, yaw, xy)
    raise KeyError(f"slot {slot} missing at frame {frame}")


def band_of(value, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def solve_for_profile(profile, cell, scene, params, rng, ledger):
    """时间关系决定用哪个求解器 —— 题型只声明关系,不写房间分支。"""
    temporal = profile["temporal"]
    if temporal == "forward":
        return SS.solve_forward_cross_time(
            scene, params, answer_band=cell["answer_band"],
            anchor_frame=profile["anchor_frame"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            max_attempts=profile.get("max_attempts", 3000))
    if temporal == "backward":
        return SS.solve_backward_cross_time(
            scene, params, answer_band=cell["answer_band"],
            anchor_frame=profile["anchor_frame"],
            query_frame=profile["query_frame"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            max_attempts=profile.get("max_attempts", 3000))
    if temporal == "instant":
        return SS.solve_instant_binding(
            scene, params, instants=profile["binding_frames"],
            profile_id=profile["id"], idle_choices=profile["idle_choices"],
            rng=rng, ledger=ledger,
            max_attempts=profile.get("max_attempts", 3000))
    raise ValueError(f"unknown temporal relation {temporal!r}")


def build_cell_plan(cells, profiles, pair_assets, params, seed):
    """先分配答案与角色,再求解 —— 不是先采样后看落在哪。"""
    a1, a2 = pair_assets
    plan = []
    per_profile = {}
    for profile in profiles:
        # 答案格按题型的答案空间分配:方位带题分带,时间带题分有序带对,
        # 外观题分目标外观 —— 一律**先分配再求解**。
        kind = profile.get("answer_kind", "azimuth_band")
        if kind == "azimuth_band":
            cellsets = [tuple(b) for b in profile["answer_bands_deg"]]
        elif kind in ("time_band", "first_caller_coat"):
            cellsets = [tuple(p) for p in profile["answer_band_pairs"]]
        else:
            cellsets = list(profile.get("answer_labels",
                                        ["black-and-white", "yellow"]))
        n = cells
        per_profile[profile["id"]] = {
            "band": balanced(cellsets, n, seed, profile["id"], "band"),
            # 目标是先叫者还是后叫者:两半均衡 —— 固定当先叫者会让
            # 目标的答案带被带对的第一个分量决定,分布随之偏斜。
            "target_first": balanced([True, False], n, seed, profile["id"],
                                     "first"),
            "coat": balanced([(a1, a2), (a2, a1)], n, seed, profile["id"],
                             "coat"),
            "target_slot": balanced(["source1", "source2"], n, seed,
                                    profile["id"], "slot"),
        }
    for profile in profiles:
        alloc = per_profile[profile["id"]]
        for index in range(cells):
            plan.append({
                "profile": profile,
                "cell_index": index,
                "answer_band": alloc["band"][index],
                "call_bands": alloc["band"][index],
                "target_first": bool(alloc["target_first"][index]),
                "pair_assets": alloc["coat"][index],
                "target_slot": alloc["target_slot"][index],
            })
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--cells", type=int, default=6)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--snapshot-content", default=(
        "/data/avengine_external/ue-assets/"
        "actor_content_registry_v9_20260823T033709Z/cpp/unreal_projects/"
        "SpearSim/Content"))
    args = parser.parse_args(argv)

    if args.out_root.exists():
        print(f"refusing to overwrite: {args.out_root}", file=sys.stderr)
        return 2
    scene_cfg = json.loads(args.scene_config.read_text())
    profiles = json.loads(args.profiles.read_text())
    params = json.loads(args.params.read_text())
    scene = SS.load_scene(scene_cfg)
    base_request = json.loads(Path(scene_cfg["camera_base_request"]).read_text())
    registry = json.loads(
        (REPO / "examples/runtime/source_asset_runtime_profiles.json").read_text())
    by_id = {a["asset_id"]: a for a in registry["assets"]}
    pair = (list(COAT_WORDS)[0], list(COAT_WORDS)[1])

    args.out_root.mkdir(parents=True)
    programs_dir = args.out_root / "programs"
    programs_dir.mkdir()
    ledger = SS.RejectionLedger()
    cells = build_cell_plan(args.cells, profiles, pair, params, args.seed)

    made, rejected, records = [], [], []
    for cell in cells:
        profile = cell["profile"]
        pid = f"{profile['id']}_{cell['cell_index'] + 1:03d}"
        rng = sha_rng(args.seed, pid)
        outcome = solve_for_profile(profile, cell, scene, params, rng, ledger)
        if isinstance(outcome, SS.Rejection):
            rejected.append({"point_id": pid, "reason": outcome.reason,
                             "detail": outcome.detail})
            continue
        try:
            record = realise_point(pid, cell, outcome, scene, base_request,
                                   params, by_id, args, programs_dir, rng)
        except Exception as exc:            # 失败即停的证据,不静默跳过
            rejected.append({"point_id": pid,
                             "reason": f"realisation_failed:{type(exc).__name__}",
                             "detail": str(exc)[:240]})
            continue
        made.append(pid)
        records.append(record)

    write_outputs(args, scene, scene_cfg, profiles, params, ledger, made,
                  rejected, records)
    print(json.dumps({"out": str(args.out_root), "scene": scene.scene_id,
                      "geometry_candidates": len(made),
                      "cells_requested": len(cells),
                      "rejected": len(rejected),
                      "combinations_evaluated":
                          ledger.summary()["combinations_evaluated"],
                      "evidence_class": "geometry_candidate"},
                     ensure_ascii=False))
    return 0


def realise_point(pid, cell, plan, scene, base_request, params, by_id, args,
                  programs_dir, rng):
    profile = cell["profile"]
    pdir = args.out_root / pid
    pdir.mkdir()
    target_slot = cell["target_slot"]
    other_slot = "source2" if target_slot == "source1" else "source1"
    assets = cell["pair_assets"]
    selection = _selection_doc(assets[0], assets[1], by_id,
                               args.snapshot_content)
    (pdir / "actor_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2))
    slot_asset = {a["source_slot_id"]: a["asset_id"] for a in selection["actors"]}
    slot_coat = {s: COAT_WORDS[a] for s, a in slot_asset.items()}

    # 相机与听者:同一份姿态结果
    camera_ue_cm = [plan.camera_xy[0], plan.camera_xy[1],
                    scene.camera_height_m * 100.0]
    camera_world_m = apartment_ue_point_to_world_m(camera_ue_cm)
    m1_request = apply_camera_listener_pose_ue(
        base_request, request_id=f"qa_v3_{pid}", position_m=camera_world_m,
        ue_yaw_degrees=plan.camera_ue_yaw_deg,
        horizontal_fov_deg=scene.hfov_deg)
    (pdir / "m1_capture_request.json").write_text(
        json.dumps(m1_request, ensure_ascii=False, indent=2))

    # 题型专用音频调度:语义角色 → 槽位
    if profile["id"] == "card1F":
        schedule = AP.schedule_forward_anchor(
            rng, params=params, anchor_frame=plan.anchor_frame)
    elif profile["id"] == "card1B":
        schedule = AP.schedule_backward_anchor(
            rng, params=params, anchor_frame=plan.anchor_frame,
            query_frame=plan.query_frame)
    elif profile.get("answer_kind") in ("time_band", "first_caller_coat"):
        # ⑧⑨ 都要"两只都有首叫、且可分辨";谁先叫由本格分配,不写死。
        schedule = AP.schedule_first_call_bands(
            rng, params=params, target_bands=tuple(cell["call_bands"]),
            first_caller_role=(AP.TARGET if cell["target_first"] else AP.OTHER))
    else:
        schedule = AP.schedule_exactly_one_calling(
            rng, params=params, query_frame=plan.query_frame)
    slot_events = schedule.bind({AP.TARGET: target_slot, AP.OTHER: other_slot})

    request = {"pair_kind": "dog", "point_id": pid,
               "endpoint_1": EP_MAP[assets[0]][0],
               "endpoint_2": EP_MAP[assets[1]][1],
               "sound_asset_id": params["SOUND_ASSET"]}
    program = build_program(request, slot_events)
    (programs_dir / f"{program['program_id']}.json").write_text(
        json.dumps(program, ensure_ascii=False, indent=1))

    # 时间线:相机与折线都来自求解结果
    base_route = plan.base_route.samples_xy      # 未平移:创作用原路线
    z = 0.0
    routes = {target_slot: (base_route[0], base_route[-1], base_route),
              other_slot: (plan.other_point, plan.other_point, None)}
    s1, s2 = routes["source1"], routes["source2"]
    timeline = author_current_apartment_visual_timeline(
        actor_selection_path=pdir / "actor_selection.json",
        source_asset_registry_path=(
            REPO / "examples/runtime/source_asset_runtime_profiles.json"),
        output_path=pdir / "timeline_authored.json",
        camera_position_ue_cm=camera_ue_cm,
        camera_yaw_deg=plan.camera_ue_yaw_deg,
        human_start_ue_cm=[s1[0][0], s1[0][1], z],
        human_end_ue_cm=[s1[1][0], s1[1][1], z],
        beagle_start_ue_cm=[s2[0][0], s2[0][1], z],
        beagle_end_ue_cm=[s2[1][0], s2[1][1], z],
        human_waypoints_ue_cm=([[p[0], p[1], z] for p in s1[2]]
                               if s1[2] else None),
        beagle_waypoints_ue_cm=([[p[0], p[1], z] for p in s2[2]]
                                if s2[2] else None),
        hfov_degrees=scene.hfov_deg,
    )
    if plan.idle_frames:
        timeline = transform_idle_then_walk(timeline, target_slot,
                                            plan.idle_frames)
    (pdir / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=1))

    # 在最终相机姿态下重算真值,并与分配的答案格核对
    query_frame = plan.query_frame
    answer_kind = profile.get("answer_kind", "azimuth_band")
    # 真值一律在**最终时间线**上重算(相机姿态已应用),不沿用求解器角度
    truth_deg = recompute_azimuth(timeline, target_slot, query_frame)
    other_deg = recompute_azimuth(timeline, other_slot, query_frame)
    answer = build_answer(answer_kind, profile, cell, timeline, schedule,
                          slot_events, target_slot, other_slot, slot_coat,
                          truth_deg, query_frame, params)

    fact = {
        "schema": "qa_v3_fact_record_v2",
        "point_id": pid, "scene_id": scene.scene_id,
        "profile_id": profile["id"],
        "evidence_class": "geometry_candidate",
        "temporal_relation": ("anchor_before_query"
                              if profile["temporal"] == "forward"
                              else "anchor_after_query"),
        "anchor_frame": plan.anchor_frame, "query_frame": query_frame,
        "target_slot": target_slot, "target_coat": slot_coat[target_slot],
        "slot_coat": slot_coat,
        "camera": {"ue_cm": camera_ue_cm,
                   "ue_yaw_deg": plan.camera_ue_yaw_deg,
                   "listener_from_same_pose_result": True},
        "answer_kind": answer_kind,
        "truth": dict(answer["truth"],
                      query_azimuth_deg=round(truth_deg, 3),
                      other_slot_azimuth_deg=round(other_deg, 3),
                      recomputed_after_camera_pose=True),
        "mcq": answer["mcq"],
        "open": answer["open"],
        "audio": {"program_id": program["program_id"],
                  "anchor_role": schedule.anchor.role,
                  "anchor_slot": target_slot,
                  "declared": schedule.declared,
                  "events": [{"role": e.role, "purpose": e.purpose,
                              "start_sample": e.start_sample}
                             for e in schedule.events]},
        "search_attempts": plan.checks.get("search_attempts"),
        "line_of_sight_screened": plan.checks.get("line_of_sight_screened"),
        "status": "research_candidate", "qualification_claim": False,
    }
    # 事实与 program 逐条一致性:锚定角色必须绑到题目目标槽位
    # 锚定与题目目标的关系**因题型而异**,按 profile 声明检查:
    #   错时族的锚就是身份锚,必须绑到目标;
    #   ⑦ 的目标是查询时刻的发声者;⑨ 的目标由"谁先叫"决定。
    # 早先这里把错时族的假设硬套到 ⑦⑨ 上,把合格点全判失败了。
    binding = profile.get("anchor_binding", "target")
    anchor_start = schedule.anchor.start_sample
    anchor_slots = [slot for slot, start in slot_events if start == anchor_start]
    if binding == "target" and anchor_slots != [target_slot]:
        raise ValueError("the audio anchor is not bound to the question target")
    if binding == "query_caller":
        span = schedule.events[0].frame_span()
        calling = [slot for (slot, _), event in zip(slot_events, schedule.events)
                   if event.frame_span()[0] <= plan.query_frame
                   < event.frame_span()[1]]
        if calling != [target_slot]:
            raise ValueError("the query-instant caller is not the question "
                             f"target (span {span})")
    (pdir / "fact_record.json").write_text(
        json.dumps(fact, ensure_ascii=False, indent=2))
    return fact


def build_answer(kind, profile, cell, timeline, schedule, slot_events,
                 target_slot, other_slot, slot_coat, truth_deg, query_frame,
                 params):
    """按题型的答案空间造真值;MCQ 与 Open 引用**同一条**事实。"""
    coat = slot_coat[target_slot]
    if kind == "azimuth_band":
        bands = [tuple(b) for b in profile["answer_bands_deg"]]
        labels = [f"[{lo:g}, {hi:g})" for lo, hi in bands]
        got = next((i for i, (lo, hi) in enumerate(bands)
                    if lo <= truth_deg < hi), None)
        want = bands.index(tuple(cell["answer_band"]))
        if got != want:
            raise ValueError(
                f"recomputed truth {truth_deg:.2f} deg lands in band {got}, "
                f"not the assigned {want}: the final camera pose disagrees "
                "with the solver's geometry")
        return {"truth": {"band_index": got},
                "mcq": {"stem": ("Which azimuth band relative to your facing "
                                 f"direction is the {coat} dog in at the "
                                 "queried moment? Right is positive."),
                        "options_space": labels, "truth_option": labels[got]},
                "open": {"stem": ("Roughly how many degrees from your facing "
                                  "direction is that dog at the queried "
                                  "moment? Right positive."),
                         "truth_value": round(truth_deg, 3), "unit": "deg",
                         "scoring": "circular_deg"}}
    if kind == "coat_at_query":
        calling = [slot for slot, event in zip(
            [s for s, _ in slot_events], schedule.events)
            if event.frame_span()[0] <= query_frame < event.frame_span()[1]]
        if len(calling) != 1:
            raise ValueError(f"{len(calling)} actors sound at the query frame")
        truth = slot_coat[calling[0]]
        options = ["black-and-white", "yellow", "both", "neither"]
        seconds = query_frame / 15.0
        return {"truth": {"calling_at_query": truth,
                          "query_second": round(seconds, 4)},
                "mcq": {"stem": (f"At {seconds:.1f} seconds on the video "
                                 "clock, which dog is barking?"),
                        "options_space": options, "truth_option": truth},
                "open": {"stem": (f"Which dog, if any, is barking at "
                                  f"{seconds:.1f} seconds?"),
                         "truth_value": truth, "scoring": "closed_set"}}
    if kind == "time_band":
        edges = [float(b) for b in params["BANDS_CARD8"]]
        firsts = {}
        for (slot, start), event in zip(slot_events, schedule.events):
            firsts.setdefault(slot, start / AP.SAMPLE_RATE)
        onset = firsts[target_slot]
        got = next((i for i in range(len(edges) - 1)
                    if edges[i] <= onset < edges[i + 1]), None)
        pair = tuple(cell["call_bands"])
        want = pair[0] if cell["target_first"] else pair[1]
        if got != want:
            raise ValueError(f"first call landed in band {got}, assigned {want}")
        labels = [f"[{edges[i]:g}, {edges[i + 1]:g})"
                  for i in range(len(edges) - 1)]
        return {"truth": {"first_onset_s": round(onset, 4), "band_index": got},
                "mcq": {"stem": (f"When does the {coat} dog bark for the FIRST "
                                 "time? Pick the time range in seconds."),
                        "options_space": labels, "truth_option": labels[got]},
                "open": {"stem": (f"At how many seconds does the {coat} dog "
                                  "bark for the first time?"),
                         "truth_value": round(onset, 4), "unit": "s",
                         "scoring": "absolute_time"}}
    if kind == "first_caller_coat":
        firsts = {}
        for slot, start in slot_events:
            firsts.setdefault(slot, start)
        first_slot = min(firsts, key=firsts.get)
        truth = slot_coat[first_slot]
        expect_first = target_slot if cell["target_first"] else other_slot
        if first_slot != expect_first:
            raise ValueError(
                f"the first caller is {first_slot}, not the assigned "
                f"{expect_first}: the schedule disagrees with the cell plan")
        return {"truth": {"first_to_bark": truth,
                          "onset_gap_s": round(
                              abs(firsts[target_slot] - firsts[other_slot])
                              / AP.SAMPLE_RATE, 4)},
                "mcq": {"stem": ("Which dog barked first, the black-and-white "
                                 "one or the yellow one?"),
                        "options_space": ["black-and-white", "yellow"],
                        "truth_option": truth},
                "open": {"stem": "Which dog made the first sound?",
                         "truth_value": truth, "scoring": "closed_set"}}
    raise ValueError(f"unknown answer kind {kind!r}")


def write_outputs(args, scene, scene_cfg, profiles, params, ledger, made,
                  rejected, records):
    by_profile = Counter(r["profile_id"] for r in records)
    coat_of_slot1 = Counter(r["slot_coat"]["source1"] for r in records)
    target_slots = Counter(r["target_slot"] for r in records)
    # 答案分布按题型的答案空间统计:方位带/时间带看带号,外观题看标签
    def answer_key(record):
        truth = record["truth"]
        for field in ("band_index", "calling_at_query", "first_to_bark"):
            if field in truth:
                return f"{record['profile_id']}:{truth[field]}"
        return f"{record['profile_id']}:unknown"

    bands = Counter(answer_key(r) for r in records)
    manifest = {
        "schema": "qa_v3_scene_batch_manifest_v1",
        "scene": {"scene_id": scene.scene_id, "backend": scene.backend,
                  "scene_asset_id": scene_cfg.get("scene_asset_id",
                                                  scene.scene_id),
                  "route_domain": scene_cfg.get("route_domain"),
                  "bank_adapter": scene.provenance.get("bank_adapter"),
                  "routes_loaded": scene.provenance.get("routes_loaded"),
                  "line_of_sight_screened": scene.line_of_sight_screened},
        "evidence_class": "geometry_candidate",
        "boundary": ("no pixel or line-of-sight evidence yet; these are "
                     "pre-render candidates, not admitted questions"),
        "counts": {"cells_requested": args.cells * len(profiles),
                   "geometry_candidates": len(made),
                   "rejected": len(rejected),
                   "by_profile": dict(by_profile)},
        "search": {k: v for k, v in ledger.summary().items()
                   if k != "first_example"},
        "search_note": ("pass rates observed while filling the quota; "
                        "candidate order and the stopping rule affect them, "
                        "so they are not scene-wide admission probabilities"),
        "evaluated_until_quota_filled": True,
        "balance": {"source1_coat": dict(coat_of_slot1),
                    "target_slot": dict(target_slots),
                    "answer_by_profile": dict(sorted(bands.items()))},
        "rejections": rejected,
        "params": params,
        "status": "research_candidate", "qualification_claim": False,
    }
    (args.out_root / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    with open(args.out_root / "facts.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
