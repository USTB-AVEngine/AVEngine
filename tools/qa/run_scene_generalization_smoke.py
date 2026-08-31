#!/usr/bin/env python3
"""Design-layer cross-scene smoke: one config, several real scenes.

对每个场景用**完全相同**的题型配置跑候选搜索,报告场景来源、后端、
是否有真实导航输入、每题型的候选数/准入数/拒绝率/拒绝原因、答案带分布,
以及外观、source slot、答案带的配平。

边界(必须原样出现在报告里):本工具只验证**设计/采样层**的场景无关性。
它不渲染,也不证明端到端渲染泛化 —— 后者需要同一条生产渲染链在至少
两个场景上跑通,当前只有一个场景具备打包舞台。

"仓库里有某套编译器"不算场景证据:只有实际载入并跑过的场景才计数,
载入失败的场景带原因列在 scenes_failed 里。
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
    for index in range(per_profile):
        band = bands[index % len(bands)]      # 答案带轮流分配(先分配后求解)
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
        else:
            plans.append(outcome)
    return plans, rejects


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

    report = {"schema": "qa_v3_scene_generalization_smoke_v1",
              "boundary": ("design/sampling layer only; this does not render "
                           "and does not demonstrate end-to-end cross-scene "
                           "generalization"),
              "params": params, "per_profile": args.per_profile,
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
             "navigation_input": scene.provenance.get("route_bank_schema"),
             "bank_adapter": scene.provenance.get("bank_adapter"),
             "routes_loaded": scene.provenance.get("routes_loaded"),
             "navigable_points": scene.provenance.get("navigable_points"),
             "hfov_deg": scene.hfov_deg,
             "line_of_sight_screened": scene.line_of_sight_screened})
        per_scene = {}
        for profile in profiles:
            ledger = RejectionLedger()
            plans, rejects = run_profile(scene, params, profile,
                                         args.per_profile, args.seed, ledger)
            band_counts = Counter(
                tuple(p.answer_cell["band"]) for p in plans)
            per_scene[profile["id"]] = {
                "requested": args.per_profile,
                "admitted": len(plans),
                "point_rejects": rejects,
                "admission_rate": len(plans) / args.per_profile,
                "candidate_rejections": ledger.summary(),
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
            }
        report["results"][scene.scene_id] = per_scene

    report["status"] = "research_candidate"
    report["qualification_claim"] = False
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    summary = {sid: {p: (r["admitted"], r["requested"])
                     for p, r in per.items()}
               for sid, per in report["results"].items()}
    print(json.dumps({"scenes_loaded": len(report["scenes_loaded"]),
                      "scenes_failed": len(report["scenes_failed"]),
                      "admitted_over_requested": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
