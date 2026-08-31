#!/usr/bin/env python3
"""Design one qa-v3 dual-source pilot batch (stage two assembler).

把阶段一组件串成设计批:每点先生成专属音频程序(随机 onset、锚事件+
静默尾),据锚槽位指派运动角色,从**经验证的走廊弦库**选路线,引擎 CLI
编制匀速 timeline,A 类点再做静→走平移变换,最后逐点过错时过滤总闸,
不过就换路线/转折帧重试。**不改在产 design_qa_batch.py 一行**;其中
actor_entry/selection_doc/mesh_package_for 与路线库逻辑复制自该文件并
标注(行为保持,新批投产不影响旧批可重现性)。

两类运动构成(答案均衡的结构来源):
  A 类  锚定角色 = 移动者(静→走,转折帧在锚定帧前,锚后仍在走)——
        供 card1(锚后角位移)、card5R(距离变化)、card6R=moving;
  B 类  锚定角色 = 站立者,另一只全程走——供 card6R=still 与 7/8/9;
        card1/5R 在 B 类点天然被过滤器拒(设计如此,报告可见)。

闸门孪生(设计层):
  Gate A  每个主点自动生成 program 的归属互换版(同 onset 序列、
          source1/source2 事件对调)——视觉复用主点字节,零渲染成本;
  Gate B  外观孪生(资产互换,复用 batch2d 机制,服务 card7/9)与
          轨迹孪生(两角色路线互换,服务 card1)按配额从 A 类主点派生,
          需要独立视觉渲染。

产物(fresh 根目录,拒绝覆盖):每点目录 spec.json / actor_selection.json
/ timeline.json(A 类另存 timeline_authored.json 供审计)+ programs/
(program、plan、Gate A 变体)+ filter_report.json + batch_manifest.json。
失败点如实记录,不硬凑。research_candidate。

用法:
  design_qa_v3_pilot_batch.py --output-root DIR --seed S \
      --class-a N --class-b M --twin-appearance K1 --twin-route K2 \
      --params PARAMS.json [--pair dog|human] [--max-retries 12]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from build_qa_v3_programs import build_program, plan_events  # noqa: E402
from filter_cross_time_points import (  # noqa: E402
    PointView,
    azimuth_deg,
    circ_diff,
    evaluate_point,
    sample_to_frame,
)
from make_idle_then_walk_timeline import transform_idle_then_walk  # noqa: E402

import hashlib  # noqa: E402

import jsonschema  # noqa: E402

# ---- 复制自 tools/qa/design_qa_batch.py(行为保持)----------------------
CAMERA_POS = [-70.0, 65.0, 147.1]
CAMERA_YAW = -145.0
Z = 27.1
HUMANS = {"blue": "rocketbox_human_male_adult_01_top_blue_research_v1",
          "green": "rocketbox_human_male_adult_01_top_green_research_v1"}
DOGS = {"collie": "generated_border_collie_black_white_medium_standard_adult_research_v1",
        "labrador": "generated_labrador_yellow_medium_standard_adult_research_v1"}


def _load_bank():
    """路线弦 + 站点。站点扩展到每条弦上的五分点(端点与中点都在
    unique1000 验证过的走廊几何上,家具安全性继承)。"""
    bank = json.load(open(REPO / "examples/qa_v2/straight_corridor_bank_v1.json"))
    directed, stands = [], set()
    for seg in bank["segments"]:
        a, b = tuple(seg["start"]), tuple(seg["end"])
        directed.append((a, b))
        directed.append((b, a))
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            stands.add((round(a[0] + (b[0] - a[0]) * t, 1),
                        round(a[1] + (b[1] - a[1]) * t, 1)))
    return directed, sorted(stands)


def _sub_chords(walk, speeds=(0.60, 0.68, 0.76), t0_steps=5):
    """弦的子段族:子段长 = CLI 速度 × 5s(平移变换保逐帧位移,行走
    速度即 CLI 速度,不因转折失真);起终点仍在原弦上。"""
    import math
    a, b = walk
    full = math.hypot(b[0] - a[0], b[1] - a[1])
    out = []
    for v in speeds:
        ls = v * 5.0 * 100.0
        if ls > full + 1e-6:
            continue
        dt = ls / full
        for i in range(t0_steps + 1):
            t0 = (1.0 - dt) * i / t0_steps if t0_steps else 0.0
            p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
            p1 = (a[0] + (b[0] - a[0]) * (t0 + dt), a[1] + (b[1] - a[1]) * (t0 + dt))
            out.append((p0, p1))
    return out or [walk]


def _mesh_package_for(asset, snap):
    su = asset["runtime_backends"]["spear_unreal"]
    mesh_dir_pkg = su["idle_animation"].split(".", 1)[0].rsplit("/", 1)[0]
    gate = mesh_dir_pkg.rsplit("/", 1)[-1]
    phys_dir = os.path.join(snap, "MyAssets/Audioset/Meshes", gate)
    names = [f[:-7] for f in os.listdir(phys_dir) if f.endswith(".uasset")]
    for n in names:
        if n + "_Skeleton" in names:
            return mesh_dir_pkg + "/" + n
    if "runtime" in names:
        return mesh_dir_pkg + "/runtime"
    raise RuntimeError(f"cannot identify skeletal mesh in {phys_dir}: {names}")


def _actor_entry(slot, asset_id, by_id, snap):
    rec = by_id[asset_id]
    su = rec["runtime_backends"]["spear_unreal"]
    bp = su["blueprint_class_path"]
    bp_pkg = bp.split(".", 1)[0]
    mesh_pkg = _mesh_package_for(rec, snap)
    mesh_name = mesh_pkg.rsplit("/", 1)[-1]

    def phys(package):
        p = os.path.join(snap, package.split("/Game/", 1)[1] + ".uasset")
        if not os.path.isfile(p):
            raise RuntimeError(f"missing physical source: {p}")
        return p

    return {
        "asset_id": asset_id,
        "legacy_timeline_actor_id": f"{rec['identity']['species_id']}_{slot[-1]}",
        "physical_authorized_internal_sources": {
            "blueprint": phys(bp_pkg),
            "graph_derived_mesh": phys(mesh_pkg),
            "idle": phys(su["idle_animation"].split(".", 1)[0]),
            "walking": phys(su["walking_animation"].split(".", 1)[0]),
        },
        "profile_alias": asset_id,
        "revision": rec["revision"],
        "source_slot_id": slot,
        "ue_binding": {
            "blueprint_object_path": bp,
            "blueprint_package": bp_pkg,
            "graph_derived_mesh": {
                "derivation": "direct graph dependency of the selected Blueprint; profile binds blueprint_component and declares no standalone mesh path",
                "object_path": f"{mesh_pkg}.{mesh_name}",
                "package": mesh_pkg,
            },
            "idle_object_path": su["idle_animation"],
            "idle_package": su["idle_animation"].split(".", 1)[0],
            "profile_skeletal_mesh_binding": su["skeletal_mesh_binding"],
            "profile_skeletal_mesh_path": su["skeletal_mesh_path"],
            "walking_object_path": su["walking_animation"],
            "walking_package": su["walking_animation"].split(".", 1)[0],
        },
    }


def _selection_doc(a1, a2, by_id, snap):
    return {
        "schema": "avengine_apartment_actor_selection_v1",
        "asset_authorization": "verified_internal",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": "QA v3 dual-source pilot batch; research only.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "actors": [_actor_entry("source1", a1, by_id, snap),
                   _actor_entry("source2", a2, by_id, snap)],
    }
# ---- 复制段结束 ---------------------------------------------------------


def _rng(seed: str, *parts: str):
    import numpy as np
    digest = hashlib.sha256(("\0".join((seed,) + parts)).encode()).hexdigest()
    return np.random.default_rng(int(digest[:12], 16))


def _author_timeline(py, selection_path, s1_start, s1_end, s2_start, s2_end, out_path):
    cmd = [py, "-m", "avengine.cli", "m5", "author-current-apartment-visual-timeline",
           "--actor-selection", str(selection_path),
           "--source-asset-registry", str(REPO / "examples/runtime/source_asset_runtime_profiles.json"),
           "--camera-position-ue-cm", *map(str, CAMERA_POS),
           "--camera-yaw-deg", str(CAMERA_YAW),
           "--human-start-ue-cm", *map(str, s1_start),
           "--human-end-ue-cm", *map(str, s1_end),
           "--beagle-start-ue-cm", *map(str, s2_start),
           "--beagle-end-ue-cm", *map(str, s2_end),
           "--output", str(out_path)]
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"timeline authoring failed: {(proc.stdout + proc.stderr)[-300:]}")


def _swap_events(events):
    other = {"source1": "source2", "source2": "source1"}
    return [(other[slot], start) for slot, start in events]


PRESCREEN_FOV_DEG = 52.5  # hfov 105° 的一半;终审仍以 timeline 实际 hfov 为准


def _walk_pos(walk, k_idle, frame):
    """A 类平移式变换后 mover 的解析位置(k_idle=0 即 CLI 原样匀速)。"""
    t = max(0, frame - k_idle) / 74.0
    return (walk[0][0] + (walk[1][0] - walk[0][0]) * t,
            walk[0][1] + (walk[1][1] - walk[0][1]) * t)


def _prescreen(sub_class, walk, stand, k_idle, anchor_frame,
               first_frames, params):
    """纯几何预筛(闭式解,不调 CLI);终审仍走 evaluate_point。

    sub_class:B(锚定者站立)只查通用绑定;A1(供 card1)查片尾角距
    (END_GAP_MIN 参数——占位带宽下开放版口径在本房间几何中近乎不可产,
    冒烟按 MCQ 口径传值并在 manifest 标注)与锚后角位移;A5(供 card5R)
    查距离变化。卡间解耦:不再要求单点全能。
    """
    import math
    cam, yaw = (CAMERA_POS[0], CAMERA_POS[1]), CAMERA_YAW
    az_stand = azimuth_deg(cam, yaw, stand)
    if abs(az_stand) > PRESCREEN_FOV_DEG:
        return False

    def az_mover(f):
        return azimuth_deg(cam, yaw, _walk_pos(walk, k_idle, f))

    def dist_mover(f):
        p = _walk_pos(walk, k_idle, f)
        return math.hypot(p[0] - cam[0], p[1] - cam[1])

    for f in [anchor_frame] + list(first_frames):
        if circ_diff(az_mover(f), az_stand) < params["MIN_AZIMUTH_SEP"]:
            return False
        if abs(az_mover(f)) > PRESCREEN_FOV_DEG:
            return False
    if sub_class in ("A1", "A5"):
        if abs(az_mover(74)) > PRESCREEN_FOV_DEG:
            return False
    if sub_class == "A1":
        if circ_diff(az_mover(74), az_stand) <= params["END_GAP_MIN"]:
            return False
        if circ_diff(az_mover(anchor_frame), az_mover(74)) <= params["THETA_FULL"]:
            return False
    if sub_class == "A5":
        if abs(dist_mover(74) - dist_mover(anchor_frame)) < params["MIN_DIST_CHANGE_CM"]:
            return False
    return True


def build_point(pid, pair_assets, sub_class, seed, params, py, by_id, snap,
                directed, stands, out_root, programs_dir, validator, max_retries):
    """组装一个主点;返回 (spec, filter_result) 或抛错。

    结构(首次冒烟的教训):card8 的首叫约束在 program 规划层满足(换
    路线救不了它);路线×站点×转折帧先过**解析几何预筛**,只有预筛通过
    的组合才花一次 CLI 调用,evaluate_point 终审兜底。program 本身也纳入
    重采(不同派生种子最多 3 版)。
    """
    a1, a2 = pair_assets
    pdir = Path(out_root) / pid
    request_base = {"pair_kind": "dog" if a1 in DOGS.values() else "human",
                    "endpoint_1": f"qa_v3_{pid}_s1", "endpoint_2": f"qa_v3_{pid}_s2",
                    "sound_asset_id": params["SOUND_ASSET"]}
    pdir.mkdir(parents=True)
    (pdir / "actor_selection.json").write_text(
        json.dumps(_selection_doc(a1, a2, by_id, snap), ensure_ascii=False, indent=2))

    last_reason = "no candidate passed prescreen"
    cli_budget = max_retries
    for prog_try in range(3):
        sub_pid = f"{pid}#p{prog_try}"
        first_slot = ("source1" if int(_rng(seed, sub_pid, "first").integers(0, 2))
                      else "source2")
        events, anchor = plan_events(seed, sub_pid, first_slot,
                                     first_min_s=params["FIRST_MIN_S"],
                                     gap_min_s=params["GAP_MIN_S"],
                                     tail_silence_s=params["TAIL_MIN_S"],
                                     first_call_bands=params["BANDS"],
                                     min_first_call_gap_s=params["T_HALF"])
        anchor_slot = anchor["anchor_slot"]
        other_slot = "source2" if anchor_slot == "source1" else "source1"
        is_a = sub_class.startswith("A")
        mover = anchor_slot if is_a else other_slot
        static = other_slot if is_a else anchor_slot
        anchor_frame = sample_to_frame(anchor["anchor_start_sample"])
        first_frames = [sample_to_frame(events[0][1]), sample_to_frame(events[1][1])]

        request = dict(request_base, point_id=pid, first_slot=first_slot)
        program = build_program(request, events)
        gate_a = build_program(dict(request, point_id=pid + "_gateA"),
                               _swap_events(events))
        for doc in (program, gate_a):
            errs = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
            if errs:
                raise RuntimeError(f"{pid}: program schema violation: {errs[0].message}")

        rng = _rng(seed, sub_pid, "route")
        if is_a:
            k_options = sorted({int(v) for v in
                                rng.integers(8, max(9, anchor_frame - 4) + 1,
                                             size=3)})
        else:
            k_options = [0]
        walks = []
        for w in directed:
            walks.extend(_sub_chords(w) if sub_class == "A1" else [w])
        combos = [(w, s, k) for w in walks for s in stands for k in k_options]
        order = rng.permutation(len(combos))
        candidates = [combos[i] for i in order
                      if _prescreen(sub_class, combos[i][0], combos[i][1],
                                    combos[i][2], anchor_frame, first_frames, params)]
        if not candidates:
            last_reason = f"prog_try {prog_try}: prescreen found no candidate"
            continue

        for walk, stand, k_idle in candidates[:cli_budget]:
            cli_budget -= 1
            stand_route = (stand, stand)
            routes = {mover: walk, static: stand_route}
            idle_frames = k_idle if is_a else None
            s1r, s2r = routes["source1"], routes["source2"]
            authored = pdir / ("timeline_authored.json" if is_a
                               else "timeline.json")
            if authored.exists():
                authored.unlink()
            if (pdir / "timeline.json").exists():
                (pdir / "timeline.json").unlink()
            _author_timeline(py, pdir / "actor_selection.json",
                             list(s1r[0]) + [Z], list(s1r[1]) + [Z],
                             list(s2r[0]) + [Z], list(s2r[1]) + [Z], authored)
            timeline = json.loads(authored.read_text())
            if is_a:
                timeline = transform_idle_then_walk(timeline, mover, idle_frames)
                (pdir / "timeline.json").write_text(
                    json.dumps(timeline, ensure_ascii=False, indent=1))
            view = PointView(timeline, program, {**anchor})
            result = evaluate_point(view, params)
            core = [k for k in ("card7", "card8", "card9")
                    if not result[k]["admit"]]
            need_map = {"A1": ("card1", "card6R"), "A5": ("card5R", "card6R"),
                        "B": ("card6R",)}
            class_need = [k for k in need_map[sub_class]
                          if not result[k]["admit"]]
            if not core and not class_need:
                spec = {"point_id": pid, "pair_kind": request["pair_kind"],
                        "source1_asset": a1, "source2_asset": a2,
                        "motion_class": sub_class, "mover_slot": mover,
                        "idle_frames": idle_frames, "first_slot": first_slot,
                        "anchor_slot": anchor_slot, "anchor_frame": anchor_frame,
                        "s1_route": [list(s1r[0]), list(s1r[1])],
                        "s2_route": [list(s2r[0]), list(s2r[1])],
                        "twin_of": None, "gate_a_program_id": gate_a["program_id"],
                        "program_id": program["program_id"],
                        "program_try": prog_try,
                        "cli_calls_left": cli_budget}
                (pdir / "spec.json").write_text(
                    json.dumps(spec, ensure_ascii=False, indent=2))
                for doc, name in ((program, program["program_id"]),
                                  (gate_a, gate_a["program_id"])):
                    (Path(programs_dir) / f"{name}.json").write_text(
                        json.dumps(doc, ensure_ascii=False, indent=1))
                plan_doc = {"schema": "avengine_qa_v3_program_plan_v1",
                            "status": "research_candidate",
                            "qualification_claim": False,
                            "point_id": pid, "program_id": program["program_id"],
                            "first_slot": first_slot, **anchor}
                (Path(programs_dir) / f"{program['program_id']}.plan.json").write_text(
                    json.dumps(plan_doc, ensure_ascii=False, indent=1))
                return spec, result
            last_reason = "; ".join(f"{k}: {result[k]['reasons'][:1]}"
                                    for k in core + class_need)
            if cli_budget <= 0:
                break
        if cli_budget <= 0:
            break
    raise RuntimeError(f"{pid}: no admissible layout (cli budget left {cli_budget}); "
                       f"last: {last_reason[:240]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--class-a", type=int, default=8)
    parser.add_argument("--class-b", type=int, default=4)
    parser.add_argument("--twin-appearance", type=int, default=2)
    parser.add_argument("--twin-route", type=int, default=2)
    parser.add_argument("--pair", choices=["dog", "human"], default="dog")
    parser.add_argument("--params", required=True)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--snapshot-content",
                        default="/data/avengine_external/ue-assets/actor_content_registry_v9_20260823T033709Z/cpp/unreal_projects/SpearSim/Content")
    args = parser.parse_args(argv)

    if os.path.exists(args.output_root):
        print(f"refusing to overwrite existing output root: {args.output_root}",
              file=sys.stderr)
        return 2
    params = json.load(open(args.params))
    needed = ("THETA_FULL", "THETA_HALF", "T_HALF", "TAIL_MIN_S", "MIN_AZIMUTH_SEP",
              "MIN_DIST_CHANGE_CM", "MIN_CARD7_FRAMES", "BANDS",
              "FIRST_MIN_S", "GAP_MIN_S", "SOUND_ASSET")
    missing = [k for k in needed if k not in params]
    if missing:
        print(f"params missing explicit keys: {missing}", file=sys.stderr)
        return 2

    reg = json.load(open(REPO / "examples/runtime/source_asset_runtime_profiles.json"))
    by_id = {a["asset_id"]: a for a in reg["assets"]}
    directed, stands = _load_bank()
    schema = json.load(open(REPO / "schemas/m6_audio_program_v1.schema.json"))
    validator = jsonschema.Draft202012Validator(schema)
    pair_assets = ((DOGS["collie"], DOGS["labrador"]) if args.pair == "dog"
                   else (HUMANS["blue"], HUMANS["green"]))

    os.makedirs(args.output_root)
    programs_dir = Path(args.output_root) / "programs"
    programs_dir.mkdir()

    specs, filter_results, failures = [], {}, []
    n_a1 = (args.class_a + 1) // 2
    plan_list = [("A1", i) for i in range(n_a1)] + \
                [("A5", i) for i in range(args.class_a - n_a1)] + \
                [("B", i) for i in range(args.class_b)]
    for cls, i in plan_list:
        pid = f"v3{cls.lower()}_{i + 1:03d}"
        try:
            spec, result = build_point(pid, pair_assets, cls, args.seed, params,
                                       args.python, by_id, args.snapshot_content,
                                       directed, stands, args.output_root,
                                       programs_dir, validator, args.max_retries)
            specs.append(spec)
            filter_results[pid] = {k: v for k, v in result.items()
                                   if k.startswith("card") or k.startswith("anchor")}
        except Exception as exc:
            failures.append({"point_id": pid, "error": str(exc)[:300]})

    # Gate B 孪生:外观孪生从任意 A 型派生;路线孪生只从 A1(card1 语义)
    twins = []
    a_specs = [s for s in specs if s["motion_class"].startswith("A")]
    a1_specs = [s for s in specs if s["motion_class"] == "A1"]
    for j in range(min(args.twin_appearance, len(a_specs))):
        src = a_specs[j]
        twins.append(dict(src, point_id=f"{src['point_id']}_twA",
                          source1_asset=src["source2_asset"],
                          source2_asset=src["source1_asset"],
                          twin_of=src["point_id"], twin_kind="appearance"))
    for j in range(min(args.twin_route, len(a1_specs))):
        src = a1_specs[-(j + 1)]
        twins.append(dict(src, point_id=f"{src['point_id']}_twR",
                          s1_route=src["s2_route"], s2_route=src["s1_route"],
                          mover_slot=("source2" if src["mover_slot"] == "source1"
                                      else "source1"),
                          twin_of=src["point_id"], twin_kind="route"))
    for tw in twins:
        tdir = Path(args.output_root) / tw["point_id"]
        tdir.mkdir()
        (tdir / "actor_selection.json").write_text(json.dumps(
            _selection_doc(tw["source1_asset"], tw["source2_asset"], by_id,
                           args.snapshot_content), ensure_ascii=False, indent=2))
        authored = tdir / ("timeline_authored.json"
                           if tw["motion_class"].startswith("A")
                           else "timeline.json")
        _author_timeline(args.python, tdir / "actor_selection.json",
                         list(tw["s1_route"][0]) + [Z], list(tw["s1_route"][1]) + [Z],
                         list(tw["s2_route"][0]) + [Z], list(tw["s2_route"][1]) + [Z],
                         authored)
        if tw["motion_class"].startswith("A"):
            timeline = json.loads(authored.read_text())
            timeline = transform_idle_then_walk(timeline, tw["mover_slot"],
                                                tw["idle_frames"])
            (tdir / "timeline.json").write_text(
                json.dumps(timeline, ensure_ascii=False, indent=1))
        (tdir / "spec.json").write_text(json.dumps(tw, ensure_ascii=False, indent=2))

    manifest = {
        "schema": "avengine_qa_v3_pilot_design_batch_v1",
        "status": "research_candidate", "qualification_claim": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "pair": args.pair, "parameters": params,
        "counts": {"class_a1": sum(1 for s in specs if s["motion_class"] == "A1"),
                   "class_a5": sum(1 for s in specs if s["motion_class"] == "A5"),
                   "class_b": sum(1 for s in specs if s["motion_class"] == "B"),
                   "twins": len(twins), "failures": len(failures)},
        "failures": failures,
        "card1_open_form_note": (
            "END_GAP_MIN in params decides the card1 ending-gap gate. The open-"
            "form bound (2 x THETA_HALF at the placeholder 30deg = 60deg) is "
            "nearly unsatisfiable in this room geometry (diagnostic: 4/1900 "
            "layouts); this batch uses the MCQ-form value and card1 open-form "
            "production is BLOCKED pending tolerance-band calibration."),
    }
    (Path(args.output_root) / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    (Path(args.output_root) / "filter_report.json").write_text(
        json.dumps({"schema": "avengine_cross_time_filter_v1",
                    "parameters": params, "results": filter_results},
                   ensure_ascii=False, indent=1))
    print(json.dumps({"root": args.output_root, "counts": manifest["counts"],
                      "failed": [f["point_id"] for f in failures]},
                     ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
