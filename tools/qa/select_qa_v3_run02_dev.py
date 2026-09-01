#!/usr/bin/env python3
"""Reproduce the QA-v3 run02-dev pixel-qualified 6-per-profile selection.

This versions the policy that originally existed only as prose.  It enumerates
all six-item combinations per profile, requires every observed answer and both
values of each binary construction factor, minimizes squared marginal counts,
then breaks ties by stable candidate identity.  Pixel evidence is eligibility
only: rendered audio, probes, model scores and downstream outcomes are unread.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

FACTOR_FIELDS = (
    "target_slot", "source1_coat", "target_coat", "target_moves_more",
    "first_caller_slot",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pixel_eligible(truth: dict, *, minimum_visible_fraction: float,
                   minimum_visible_pixels: int,
                   bbox_must_not_touch_frame_edge: bool) -> bool:
    height, width = [int(value) for value in truth["resolution_hw"]]
    for instance in truth["per_instance"].values():
        for frame in instance["frames"]:
            fraction = frame.get("visible_fraction")
            if fraction is None or float(fraction) < minimum_visible_fraction:
                return False
            if int(frame.get("visible_pixels", 0)) < minimum_visible_pixels:
                return False
            bbox = frame.get("target_bbox_xyxy_px")
            if bbox_must_not_touch_frame_edge:
                if bbox is None:
                    return False
                x0, y0, x1, y1 = [int(value) for value in bbox]
                if x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height:
                    return False
    return True


def candidate_from_pixel_dir(pixel_dir: Path) -> dict:
    evidence = _read_json(pixel_dir / "evidence.json")
    timeline_path = Path(evidence["inputs"]["timeline"])
    point_dir = timeline_path.parent
    fact = _read_json(point_dir / "fact_record.json")
    if fact["point_id"] != pixel_dir.name:
        raise ValueError(
            f"pixel point {pixel_dir.name} disagrees with fact "
            f"{fact['point_id']}")
    return {
        "answer": fact["mcq"]["truth_option"],
        "first_caller_slot": fact.get("first_caller_slot"),
        "pixel_evidence": str(pixel_dir.resolve()),
        "point_id": fact["point_id"],
        "profile_id": fact["profile_id"],
        "source1_coat": fact["slot_coat"]["source1"],
        "source_design_root": str(point_dir.parent.resolve()),
        "target_coat": fact["target_coat"],
        "target_moves_more": bool(fact["motion"]["target_moves_more"]),
        "target_slot": fact["target_slot"],
    }


def candidate_identity(candidate: dict) -> tuple[str, str, str]:
    return (candidate["point_id"], candidate["source_design_root"],
            candidate["pixel_evidence"])


def select_profile(candidates: list[dict], quota: int) -> tuple[list[dict], dict]:
    if len(candidates) < quota:
        raise ValueError(
            f"profile has {len(candidates)} candidates, below quota {quota}")
    candidates = sorted(candidates, key=candidate_identity)
    fields = ["answer"]
    for field in FACTOR_FIELDS:
        values = {candidate[field] for candidate in candidates
                  if candidate.get(field) is not None}
        if not values:
            continue
        if len(values) != 2:
            raise ValueError(
                f"{field} must expose both binary values in the eligible pool; "
                f"got {sorted(map(str, values))}")
        fields.append(field)
    domains = {field: {candidate[field] for candidate in candidates}
               for field in fields}

    best_key = None
    best = None
    for combination in itertools.combinations(candidates, quota):
        if any({candidate[field] for candidate in combination} != domains[field]
               for field in fields):
            continue
        objective = sum(
            sum(count * count for count in
                Counter(candidate[field] for candidate in combination).values())
            for field in fields)
        key = (objective, tuple(candidate_identity(candidate)
                                for candidate in combination))
        if best_key is None or key < best_key:
            best_key, best = key, list(combination)
    if best is None:
        raise ValueError(
            f"no {quota}-item combination covers fields {fields}")

    marginals = {
        field: {str(value): count for value, count in sorted(
            Counter(candidate[field] for candidate in best).items(),
            key=lambda item: str(item[0]))}
        for field in fields
    }
    return best, {
        "eligible_candidates": len(candidates),
        "fields": fields,
        "marginals": marginals,
        "squared_marginal_objective": best_key[0],
    }


def build_selection(pixel_roots: list[Path], *, quota: int,
                    minimum_visible_fraction: float,
                    minimum_visible_pixels: int,
                    bbox_must_not_touch_frame_edge: bool) -> dict:
    by_profile: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    evaluated = 0
    for root in pixel_roots:
        for pixel_dir in sorted(root.iterdir()):
            truth_path = pixel_dir / "pixel_visibility_truth.json"
            if not pixel_dir.is_dir() or not truth_path.is_file():
                continue
            evaluated += 1
            truth = _read_json(truth_path)
            if not pixel_eligible(
                    truth,
                    minimum_visible_fraction=minimum_visible_fraction,
                    minimum_visible_pixels=minimum_visible_pixels,
                    bbox_must_not_touch_frame_edge=bbox_must_not_touch_frame_edge):
                continue
            candidate = candidate_from_pixel_dir(pixel_dir)
            identity = (candidate["point_id"],
                        candidate["source_design_root"])
            bucket = by_profile[candidate["profile_id"]]
            previous = bucket.get(identity)
            if previous is not None:
                comparable = lambda item: {
                    key: value for key, value in item.items()
                    if key != "pixel_evidence"}
                if comparable(previous) != comparable(candidate):
                    raise ValueError(f"duplicate candidate differs: {identity}")
            if (previous is None
                    or candidate_identity(candidate)
                    < candidate_identity(previous)):
                bucket[identity] = candidate
    selected, diagnostics = [], {}
    for profile_id in sorted(by_profile):
        chosen, profile_diag = select_profile(
            list(by_profile[profile_id].values()), quota)
        selected.extend(chosen)
        diagnostics[profile_id] = profile_diag
    if not selected:
        raise ValueError("no pixel-eligible candidates found")
    return {
        "schema": "qa_v3_run02_dev_selection_v2",
        "status": "research_dev_selection",
        "qualification_claim": False,
        "inputs": {
            "pixel_roots": [str(path.resolve()) for path in pixel_roots],
            "evaluated_pixel_points": evaluated,
        },
        "pixel_criteria": {
            "minimum_visible_fraction": minimum_visible_fraction,
            "minimum_visible_pixels": minimum_visible_pixels,
            "bbox_must_not_touch_frame_edge": bbox_must_not_touch_frame_edge,
        },
        "policy": {
            "quota_per_profile": quota,
            "eligibility": "native pixel truth only",
            "coverage": (
                "every observed answer and both binary construction values"),
            "objective": "minimize sum of squared marginal counts",
            "tie_break": (
                "point_id, then source_design_root, then pixel_evidence path"),
            "forbidden_inputs": [
                "rendered audio", "physical probes", "model scores",
                "downstream question outcomes",
            ],
        },
        "counts_by_profile": dict(Counter(
            candidate["profile_id"] for candidate in selected)),
        "profile_diagnostics": diagnostics,
        "selected": selected,
    }


def _selected_identities(manifest: dict) -> set[tuple[str, str]]:
    return {(item["point_id"], item["source_design_root"])
            for item in manifest["selected"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-root", action="append", required=True,
                        type=Path)
    parser.add_argument("--quota", type=int, default=6)
    parser.add_argument("--minimum-visible-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-visible-pixels", type=int, default=1000)
    parser.add_argument("--allow-edge-touch", action="store_true")
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")

    manifest = build_selection(
        args.pixel_root, quota=args.quota,
        minimum_visible_fraction=args.minimum_visible_fraction,
        minimum_visible_pixels=args.minimum_visible_pixels,
        bbox_must_not_touch_frame_edge=not args.allow_edge_touch)
    if args.expected_manifest:
        expected = _read_json(args.expected_manifest)
        match = _selected_identities(manifest) == _selected_identities(expected)
        manifest["reproduction"] = {
            "expected_manifest": str(args.expected_manifest.resolve()),
            "selected_identities_match": match,
        }
        if not match:
            raise ValueError(
                "recomputed selection differs from expected manifest")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "selected": len(manifest["selected"]),
        "counts_by_profile": manifest["counts_by_profile"],
        "reproduction": manifest.get("reproduction"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
