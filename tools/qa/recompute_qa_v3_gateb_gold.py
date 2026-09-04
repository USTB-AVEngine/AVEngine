#!/usr/bin/env python3
"""Recompute Gate-B gold for every selected QA-v3 pilot candidate."""

from __future__ import annotations
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from design_qa_v3_scene_batch import COAT_WORDS, recompute_azimuth  # noqa:E402

PIXEL = {"card11", "card15a", "card16"}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n")


def state(timeline, slot, frame):
    return next(
        x
        for x in timeline["frames"][frame]["actor_states"]
        if x["source_slot_id"] == slot
    )


def distance(timeline, slot, frame):
    c = np.asarray(timeline["frames"][frame]["camera"]["translation_ue_cm"][:2], float)
    p = np.asarray(state(timeline, slot, frame)["translation_ue_cm"][:2], float)
    return float(np.linalg.norm(p - c))


def coat(selection, slot):
    asset = next(
        x["asset_id"] for x in selection["actors"] if x["source_slot_id"] == slot
    )
    return COAT_WORDS[asset]


def bands(profile):
    return [tuple(x) for x in profile.get("answer_bands_deg", [])]


def band_label(profile, value):
    bs = bands(profile)
    labels = [f"[{a:g}, {b:g})" for a, b in bs]
    matches = [labels[i] for i, (a, b) in enumerate(bs) if a <= value < b]
    if len(matches) != 1:
        raise ValueError(f"angle {value} outside bands")
    return matches[0]


def slot_events(program):
    candidates = program["candidate_source_endpoint_ids"]
    ep_to_slot = {candidates[0]: "source1", candidates[1]: "source2"}
    return [
        (ep_to_slot[e["source_endpoint_id"]], e["start_sample"] / 16000.0)
        for e in program["events"]
    ]


def compute(pid, profile, fact, selection, timeline, program, params):
    main = fact["mcq"]["truth_option"]
    if pid in PIXEL:
        return {
            "status": "pixel_pending",
            "main_gold": main,
            "gateb_gold": None,
            "reason": "native Gate-B pixel truth required",
        }
    if pid in {"card1F", "card1B", "card2"}:
        slot = fact["target_slot"]
        frame = fact["query_frame"]
        angle = recompute_azimuth(timeline, slot, frame)
        gold = band_label(profile, angle)
        open_gold = round(angle, 3)
        separated = abs(
            (float(fact["open"]["truth_value"]) - angle + 180) % 360 - 180
        ) > 2 * float(params["THETA_HALF"])
    elif pid == "card3":
        slot = min(slot_events(program), key=lambda x: x[1])[0]
        angle = recompute_azimuth(timeline, slot, fact["query_frame"])
        gold = "left" if angle < 0 else "right"
        open_gold = gold
        separated = gold != main
    elif pid == "card4R":
        frame = fact["query_frame"]
        ds = {s: distance(timeline, s, frame) for s in ("source1", "source2")}
        slot = min(ds, key=ds.get)
        gold = coat(selection, slot)
        open_gold = gold
        separated = gold != main
    elif pid in {"card5", "card5R"}:
        slot = fact["target_slot"]
        a, b = profile["relation_frames"]
        delta = distance(timeline, slot, b) - distance(timeline, slot, a)
        m = float(profile["min_distance_change_cm"])
        gold = "closer" if delta <= -m else "farther" if delta >= m else None
        open_gold = gold
        separated = gold is not None and gold != main
    elif pid in {"card6", "card6R", "card10"}:
        slot = fact["target_slot"]
        a, b = profile["motion_frames"]
        p = np.asarray(state(timeline, slot, b)["translation_ue_cm"], float)
        q = np.asarray(state(timeline, slot, a)["translation_ue_cm"], float)
        d = float(np.linalg.norm(p - q))
        m = float(profile["min_motion_cm"])
        gold = "moving" if d >= m else "still" if d <= 1e-6 else None
        open_gold = gold
        separated = gold is not None and gold != main
    elif pid == "card7":
        q = fact["truth"]["query_frame"]
        active = []
        for slot, e in slot_events(program):
            raw = next(x for x in program["events"] if x["start_sample"] / 16000.0 == e)
            if raw["start_sample"] <= q / 15 * 16000 < raw["end_sample_exclusive"]:
                active.append(slot)
        if len(active) != 1:
            gold = None
        else:
            gold = coat(selection, active[0])
        open_gold = gold
        separated = gold is not None and gold != main
    elif pid == "card8":
        target_coat = fact["target_coat"]
        target = next(
            s for s in ("source1", "source2") if coat(selection, s) == target_coat
        )
        first = min(t for s, t in slot_events(program) if s == target)
        options = fact["mcq"]["options_space"]
        parsed = [
            tuple(float(v.strip()) for v in option.strip("[]() ").split(","))
            for option in options
        ]
        idx = next((i for i, (lo, hi) in enumerate(parsed) if lo <= first < hi), None)
        gold = None if idx is None else options[idx]
        open_gold = first
        separated = idx is not None and abs(
            first - float(fact["open"]["truth_value"])
        ) > float(params["T_HALF"])
    elif pid == "card9":
        first = min(slot_events(program), key=lambda x: x[1])[0]
        gold = coat(selection, first)
        open_gold = gold
        separated = gold != main
    elif pid == "card15b":
        gold = len(program["events"])
        open_gold = gold
        separated = gold == main
    elif pid == "card17":
        target_asset = next(
            x["asset_id"]
            for x in read(fact["_main_selection"])["actors"]
            if x["source_slot_id"] == "source1"
        )
        target = next(
            x["source_slot_id"]
            for x in selection["actors"]
            if x["asset_id"] == target_asset
        )
        angle = recompute_azimuth(timeline, target, 40)
        bs = profile["location_bands_deg"]
        ls = profile["location_band_labels"]
        ms = [ls[i] for i, (a, b) in enumerate(bs) if a <= angle < b]
        gold = ms[0] if len(ms) == 1 else None
        open_gold = gold
        separated = gold is not None and gold != main
    else:
        raise ValueError(pid)
    relation = "preserve" if pid == "card15b" else "flip"
    ok = (gold == main) if relation == "preserve" else (gold != main and separated)
    return {
        "status": "pass" if ok else "reject",
        "main_gold": main,
        "gateb_gold": gold,
        "gateb_open_gold": open_gold,
        "expected_relation": relation,
        "relation_satisfied": bool(ok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-manifest", type=Path, required=True)
    ap.add_argument("--dual-gateb-manifest", type=Path, required=True)
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    a = ap.parse_args()
    if a.output_root.exists():
        return 2
    pilot = read(a.pilot_manifest)
    dual = read(a.dual_gateb_manifest)
    prof = {x["id"]: x for x in read(a.profiles)}
    params = read(a.params)
    dualmap = {x["pilot_id"]: x for x in dual["records"]}
    rows = []
    for room in pilot["rooms"].values():
        for pid, e in room["profiles"].items():
            for c in e.get("candidates", []):
                fact = read(c["artifacts"]["fact"])
                fact["_main_selection"] = c["artifacts"]["actor_selection"]
                if c["gateb_status"] == "materialized":
                    g = Path(c["artifacts"]["gateb"]).parent
                    manifest = read(c["artifacts"]["gateb"])
                else:
                    manifest = dualmap[c["pilot_id"]]
                    g = Path(manifest["artifacts"]["timeline"]).parent
                selection = read(g / "actor_selection_gateB.json")
                timeline = read(g / "timeline_gateB.json")
                program = read(
                    g
                    / (
                        "audio_program_gateB.json"
                        if (g / "audio_program_gateB.json").is_file()
                        else "audio_program.json"
                    )
                )
                result = compute(
                    pid, prof[pid], fact, selection, timeline, program, params
                )
                rows.append(
                    {
                        "pilot_id": c["pilot_id"],
                        "profile_id": pid,
                        "gateb_root": str(g),
                        **result,
                    }
                )
    counts = Counter(x["status"] for x in rows)
    a.output_root.mkdir(parents=True)
    out = {
        "schema": "qa_v3_gateb_gold_recompute_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "counts": dict(counts),
        "rows": rows,
    }
    write(a.output_root / "gateb_gold_manifest.json", out)
    rowmap = {row["pilot_id"]: row for row in rows}
    augmented = copy.deepcopy(pilot)
    augmented["schema"] = "qa_v3_room_centric_pilot_augmented_gateb_v1"
    augmented["gateb_gold_manifest"] = str(
        (a.output_root / "gateb_gold_manifest.json").resolve()
    )
    route_profiles = {
        "card1F",
        "card1B",
        "card2",
        "card3",
        "card5",
        "card5R",
        "card6",
        "card6R",
        "card10",
        "card15a",
        "card16",
    }
    for room in augmented["rooms"].values():
        for entry in room["profiles"].values():
            for candidate in entry.get("candidates", []):
                row = rowmap[candidate["pilot_id"]]
                candidate["gateb"] = {
                    "root": row["gateb_root"],
                    "gold_status": row["status"],
                    "main_gold": row["main_gold"],
                    "gateb_gold": row["gateb_gold"],
                    "audio_policy": (
                        "route_audio_must_change_consistently"
                        if row["profile_id"] in route_profiles
                        else "appearance_reuse_main_audio_no_rerender"
                    ),
                }
    write(a.output_root / "augmented_pilot_manifest.json", augmented)
    print(json.dumps({"counts": dict(counts), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
