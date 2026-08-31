#!/usr/bin/env python3
"""Design-layer cross-scene smoke: one question-type config, several route domains.

三条措辞边界(codex 审阅裁定,必须原样体现在输出里):

1. **数场景资产,不数路线域**。同一栋房子的两个楼层是一个场景资产里的
   两个路线域,不能当作两个独立场景来证明场景多样性。报告分别给出
   scene_assets / route_domains 两个计数。
2. **没有视线与像素证据时不使用"准入"**。这里产出的是
   `geometry_candidate`(设计期/渲染前候选):它证明方位带、时间关系与
   几何约束在数学上可构造,**不能**证明目标没有被墙、家具或另一角色
   遮挡。题目准入要走实际渲染、像素可见性真值、相机—听者一致性、
   完整音视频可答性与单模态认证。
3. **"同一份配置"要分层说**。相同的是题型 profile、方位带、时间关系、
   采样参数、语义角色与拒绝规则;不同的是各后端的路线库 schema 适配器
   (UE 是 (x,y) 厘米,habitat 是 (x,z) 米),适配器名逐场景记录。

配额被填满不等于通过率 100%,所以每个 profile 都报搜索分母:读了多少
条原始路线、评估了多少个候选组合、每个合格点平均/最大尝试多少次、
有没有答案带把预算用光。

"仓库里有某套编译器"不算场景证据:只有实际载入并跑过的才计数。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scene_sampler import (  # noqa: E402
    RejectionLedger,
    Rejection,
    load_scene,
    solve_backward_cross_time,
    solve_forward_cross_time,
)


def run_profile(scene, params, profile, per_profile, seed_base, ledger):
    import hashlib
    tag = f"{scene.scene_id}|{profile['id']}|{seed_base}".encode()
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(tag).digest()[:8], "big") % 2**32)
    bands = [tuple(b) for b in profile["answer_bands_deg"]]
    plans, rejects = [], 0
    per_band: dict[tuple, dict] = {b: {"requested": 0, "candidates": 0,
                                       "exhausted": 0, "attempts": []}
                                   for b in bands}
    for index in range(per_profile):
        band = bands[index % len(bands)]      # 答案带轮流分配(先分配后求解)
        per_band[band]["requested"] += 1
        if profile["temporal"] == "forward":
            outcome = solve_forward_cross_time(
                scene, params, answer_band=band,
                anchor_frame=profile["anchor_frame"],
                idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
                max_attempts=profile.get("max_attempts", 3000))
        else:
            outcome = solve_backward_cross_time(
                scene, params, answer_band=band,
                anchor_frame=profile["anchor_frame"],
                query_frame=profile["query_frame"],
                idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
                max_attempts=profile.get("max_attempts", 3000))
        if isinstance(outcome, Rejection):
            rejects += 1
            per_band[band]["exhausted"] += 1
        else:
            plans.append(outcome)
            per_band[band]["candidates"] += 1
            per_band[band]["attempts"].append(
                outcome.checks.get("search_attempts"))
    return plans, rejects, per_band


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", required=True, type=Path,
                        help="场景配置列表(JSON 数组)")
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--per-profile", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--route-limit", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"refusing to overwrite: {args.out}", file=sys.stderr)
        return 2
    scenes_cfg = json.loads(args.scenes.read_text())
    profiles = json.loads(args.profiles.read_text())
    params = json.loads(args.params.read_text())

    report = {"schema": "qa_v3_scene_generalization_smoke_v2",
              "boundary": ("design/sampling layer only. Outputs are "
                           "geometry_candidates, not admitted questions: no "
                           "line-of-sight or pixel evidence is involved. This "
                           "does not render and does not demonstrate "
                           "end-to-end cross-scene generalization."),
              "shared_across_scenes": ["question-type profile", "answer bands",
                                       "temporal relation", "sampling params",
                                       "semantic roles", "rejection rules"],
              "differs_across_scenes": ["route-bank schema adapter "
                                        "(UE (x,y) cm vs habitat (x,z) m)"],
              "params": params, "cells_per_profile": args.per_profile,
              "scenes_loaded": [], "scenes_failed": [], "results": {}}

    for cfg in scenes_cfg:
        try:
            scene = load_scene(cfg, route_limit=args.route_limit)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            report["scenes_failed"].append(
                {"scene_id": cfg.get("scene_id"), "backend": cfg.get("backend"),
                 "error": f"{type(exc).__name__}: {exc}"})
            continue
        report["scenes_loaded"].append(
            {"scene_id": scene.scene_id, "backend": scene.backend,
             "scene_asset_id": cfg.get("scene_asset_id", scene.scene_id),
             "route_domain": cfg.get("route_domain"),
             "navigation_input": scene.provenance.get("route_bank_schema"),
             "bank_adapter": scene.provenance.get("bank_adapter"),
             "routes_loaded": scene.provenance.get("routes_loaded"),
             "navigable_points": scene.provenance.get("navigable_points"),
             "hfov_deg": scene.hfov_deg,
             "line_of_sight_screened": scene.line_of_sight_screened})
        per_scene = {}
        for profile in profiles:
            ledger = RejectionLedger()
            plans, rejects, per_band = run_profile(
                scene, params, profile, args.per_profile, args.seed, ledger)
            band_counts = Counter(
                tuple(p.answer_cell["band"]) for p in plans)
            attempts = [p.checks.get("search_attempts") for p in plans
                        if p.checks.get("search_attempts")]
            summary = ledger.summary()
            per_scene[profile["id"]] = {
                "requested_cells": args.per_profile,
                "geometry_candidates": len(plans),
                "cells_unfilled": rejects,
                "search": {
                    "routes_loaded": scene.provenance.get("routes_loaded"),
                    "navigable_points": scene.provenance.get("navigable_points"),
                    "combinations_evaluated": summary["combinations_evaluated"],
                    "stand_points_evaluated": summary["stand_points_evaluated"],
                    "budget_exhausted_calls": summary["budget_exhausted"],
                    "attempts_per_candidate_mean": (
                        round(sum(attempts) / len(attempts), 2)
                        if attempts else None),
                    "attempts_per_candidate_max": max(attempts) if attempts
                    else None,
                    "candidate_pass_rate": (
                        round(len(plans) / summary["combinations_evaluated"], 5)
                        if summary["combinations_evaluated"] else None),
                },
                "per_band": {str(k): {"requested": v["requested"],
                                      "candidates": v["candidates"],
                                      "budget_exhausted": v["exhausted"],
                                      "attempts_mean": (
                                          round(sum(a for a in v["attempts"]
                                                    if a) / len(v["attempts"]), 2)
                                          if v["attempts"] else None),
                                      "attempts_max": (
                                          max((a for a in v["attempts"] if a),
                                              default=None))}
                             for k, v in per_band.items()},
                "candidate_rejections": {k: v for k, v in summary.items()
                                         if k in ("total", "by_reason",
                                                  "first_example")},
                "answer_band_distribution": {str(k): v for k, v
                                             in sorted(band_counts.items())},
                "distinct_cameras": len({p.camera_xy for p in plans}),
                "distinct_routes": len({p.target_route.route_id.split("+")[0]
                                        for p in plans}),
                "camera_yaw_range_deg": ([round(min(p.camera_ue_yaw_deg
                                                    for p in plans), 1),
                                          round(max(p.camera_ue_yaw_deg
                                                    for p in plans), 1)]
                                         if plans else None),
                "evidence_class": "geometry_candidate",
            }
        report["results"][scene.scene_id] = per_scene

    # 场景资产 vs 路线域:同一栋房子的两个楼层是一个资产里的两个域
    assets = {}
    for entry in report["scenes_loaded"]:
        assets.setdefault(entry.get("scene_asset_id") or entry["scene_id"],
                          []).append(entry["scene_id"])
    report["scene_assets"] = {k: sorted(v) for k, v in sorted(assets.items())}
    report["counts"] = {"scene_assets": len(assets),
                        "route_domains": len(report["scenes_loaded"]),
                        "backends": len({e["backend"]
                                         for e in report["scenes_loaded"]})}
    report["status"] = "research_candidate"
    report["qualification_claim"] = False
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    summary = {sid: {p: {"candidates": r["geometry_candidates"],
                         "cells": r["requested_cells"],
                         "combos": r["search"]["combinations_evaluated"],
                         "pass_rate": r["search"]["candidate_pass_rate"]}
                     for p, r in per.items()}
               for sid, per in report["results"].items()}
    print(json.dumps({"scene_assets": report["counts"]["scene_assets"],
                      "route_domains": report["counts"]["route_domains"],
                      "backends": report["counts"]["backends"],
                      "scenes_failed": len(report["scenes_failed"]),
                      "evidence_class": "geometry_candidate",
                      "per_domain": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
