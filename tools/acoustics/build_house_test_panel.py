#!/usr/bin/env python3
"""Build a whole-house test panel in a predeclared order from CPU acceptance.

Retain the initial train/development/test assignments. This consumes the
existing house acceptance collector without modifying its thresholds or
ranking on RT60, timing, model scores, or worker completion order.
Quadruplet seed rows are reservations for future production, not media.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "house_acceptance_summary", Path(__file__).with_name("summarize_house_acceptance.py"))
summary_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_module)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def eligible(result):
    # Use the original combined acoustic/visual decision and explicitly ensure
    # the task's minimum of six valid RIRs, not merely six stored probe rows.
    return (result.get("batch_eligible") is True
            and result.get("status") == "pass"
            and not result.get("known_exposure", True)
            and sum(r.get("status") == "valid" for r in result.get("rirs", [])) >= 6)


def build_recommendations(initial, results, frozen, target=12):
    by_house = {r["house"]: r for r in results}
    if len(by_house) != len(results):
        raise ValueError("Duplicate physical houses in acceptance results")
    retained = initial["houses"]
    retained_ids = [r["house"] for r in retained]
    if len(set(retained_ids)) != len(retained_ids):
        raise ValueError("Initial house assignments overlap")
    for h in retained_ids:
        if h not in by_house or not eligible(by_house[h]):
            raise ValueError("Retained house lacks a passing uniform remeasurement: " + h)
    rows = []
    for entry in retained:
        result = by_house[entry["house"]]
        row = copy.deepcopy(entry)
        row.update(rt60_mid_median_s=result["rt60_mid_median_s"],
                   result_path=str(Path(result["result_directory"]) / "result.json"),
                   visual_path=str(Path(result["result_directory"]) / "visual_checks.json"),
                   selection_reason="retained initial whole-house assignment")
        rows.append(row)
    test_count = sum(r["split"] == "test" for r in rows)
    if test_count > target:
        raise ValueError("Target is smaller than the retained test allocation")
    seen = set(retained_ids)
    candidates = [(tier["tier"], rank, entry) for tier in frozen["tiers"]
                  for rank, entry in enumerate(tier["order"], 1)]
    for tier, rank, entry in candidates:
        house = entry["house"]
        if test_count >= target:
            break
        if house in seen:
            raise ValueError("Frozen candidates overlap retained/earlier assignments: " + house)
        seen.add(house)
        result = by_house.get(house)
        if result is None:
            # A slower or unmeasured earlier candidate cannot be bypassed by
            # a completed later tier. Wait for the frozen prefix to finish.
            break
        if not eligible(result):
            continue
        h = result["input"]
        rows.append({"house": house, "split": "test",
                     "rt60_mid_median_s": result["rt60_mid_median_s"],
                     "candidate_area_sum_m2": h["candidate_area_sum_m2"],
                     "candidate_room_count": h["candidate_room_count"],
                     "room_types": h["room_types"], "original_split": h["original_split"],
                     "result_path": str(Path(result["result_directory"]) / "result.json"),
                     "visual_path": str(Path(result["result_directory"]) / "visual_checks.json"),
                     "selection_reason": f"first eligible in frozen tier {tier}, rank {rank}",
                     "frozen_tier": tier, "frozen_rank": rank})
        test_count += 1
    counts = {split: sum(r["split"] == split for r in rows)
              for split in ("train", "development", "test")}
    return {"target_met": test_count == target, "houses": rows, "house_counts": counts,
            "house_ratios": {k: v / len(rows) for k, v in counts.items()},
            "test_house_target": target, "test_house_shortfall": max(0, target - test_count),
            "independent_window_target": None,
            "window_targets": {"train": None, "development": None, "test": test_count * 60 * 4},
            "independent_quadruplets_per_test_house": 60,
            "independent_test_quadruplet_target": target * 60,
            "independent_test_quadruplets_reserved": test_count * 60,
            "independent_test_quadruplets_generated": 0,
            "known_exposure_excluded": True, "anonymous_exposure_verified": False,
            "native_habitat_rgb_verified": False, "production_readiness_verified": False,
            "target_met_scope": "whole-house CPU acoustic/visual panel allocation; quadruplets reserved, not produced",
            "method": "retain initial splits; admit passing candidates only in frozen tier/rank order; no model scores or RT60 ranking"}


def reserve_quadruplets(recommendations, seed_base):
    groups = []
    for i, row in enumerate(r for r in recommendations["houses"] if r["split"] == "test"):
        for j in range(60):
            group_id = f"{row['house']}__test_quad_{j:03d}"
            groups.append({"house": row["house"], "split": "test", "group_id": group_id,
                           "independent_scene_seed": seed_base + i * 60 + j,
                           "member_ids": [group_id + f"__{m}" for m in range(4)],
                           "status": "reserved_not_generated"})
    assert len({g["independent_scene_seed"] for g in groups}) == len(groups)
    return {"scope": "future independent test groups; no generated episodes or model evaluations",
            "quadruplets_per_house": 60, "quadruplets_reserved": len(groups),
            "quadruplets_generated": 0, "members_reserved": len(groups) * 4, "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, action="append", required=True)
    parser.add_argument("--frozen-order", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=2026100300)
    args = parser.parse_args()
    manifest, frozen = read(args.manifest), read(args.frozen_order)
    results = []
    for house in manifest["houses"]:
        matching = [p for p in args.measurements if (p / "houses" / house["house"] / "result.json").exists()]
        if len(matching) != 1:
            raise ValueError("Require one uniform result per house: " + house["house"])
        subset = dict(manifest, houses=[house])
        results.extend(summary_module.collect(subset, matching[0]))
    recommendations = build_recommendations(read(args.first_report / "recommended_houses.json"), results, frozen)
    args.report.mkdir(parents=True, exist_ok=False)
    save(args.report / "all_house_results.json", results)
    save(args.report / "recommended_houses.json", recommendations)
    save(args.report / "recommended_houses_v2.json", recommendations)
    save(args.report / "test_quadruplet_seed_plan.json", reserve_quadruplets(recommendations, args.seed_base))
    columns = ["house", "family", "original_split", "candidate_area_sum_m2", "candidate_room_count",
               "native_status", "visual_status", "status", "batch_eligible", "rt60_mid_median_s",
               "max_escape_fraction", "max_foa_angle_deg", "reasons", "result_directory"]
    with (args.report / "house_results.csv").open("x", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row = {k: result.get(k) for k in columns}
            row.update({k: result["input"].get(k) for k in
                        ("original_split", "candidate_area_sum_m2", "candidate_room_count")})
            row.update(visual_status=result["visual"]["status"], reasons="; ".join(result["reasons"]))
            writer.writerow(row)
    all_by_house = {r["house"]: r for r in results}
    rows = []
    for tier in frozen["tiers"]:
        for rank, h in enumerate(tier["order"], 1):
            r = all_by_house.get(h["house"])
            rows.append({"house": h["house"], "tier": tier["tier"], "rank": rank,
                         "status": r["status"] if r else "not_run",
                         "reasons": r["reasons"] if r else ["later tier not needed or not yet measured"],
                         "assigned_test": any(x["house"] == h["house"] and x["split"] == "test"
                                              for x in recommendations["houses"])})
    save(args.report / "candidate_disposition.json", rows)
    new = [r for r in results if r["house"] not in
           {h["house"] for h in read(args.first_report / "all_house_results.json")}]
    stats = {"generated_at": datetime.now().astimezone().isoformat(), "houses": len(results),
             "new_houses": len(new), "new_house_status_counts": dict(Counter(r["status"] for r in new)),
             "by_family_status": {family: dict(Counter(r["status"] for r in results if r["family"] == family))
                                  for family in ("hm3d", "mp3d")},
             "rirs": sum(len(r.get("rirs", [])) for r in results),
             "valid_rirs": sum(m.get("status") == "valid" for r in results for m in r.get("rirs", [])),
             "houses_with_rirs": sum(bool(r.get("rirs")) for r in results),
             "baseline_ray_count": sum(x["ray_count"] for r in results for x in r.get("leakage", [])),
             "cpu_views": sum(len(r["visual"].get("checks", [])) for r in results),
             "manifest_source": str(args.manifest), "measurement_roots": list(map(str, args.measurements)),
             "measurement_revision": manifest.get("measurement_revision"), "recommendations": recommendations,
             "complete_v7_readiness_verified": False, "compute_device": "CPU"}
    save(args.report / "summary.json", stats)
    save(args.report / "recommended_room_scope.json",
         {"scope": "only these sampled rooms/positions were acoustically checked; a house allocation does not certify every room",
          "houses": [{"house": h["house"], "split": h["split"],
                      "acoustic_rooms": all_by_house[h["house"]]["input"]["acoustic_rooms"]}
                     for h in recommendations["houses"]]})
    print(json.dumps({"house_counts": recommendations["house_counts"], "new_statuses": stats["new_house_status_counts"],
                      "shortfall": recommendations["test_house_shortfall"], "quadruplets_generated": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
