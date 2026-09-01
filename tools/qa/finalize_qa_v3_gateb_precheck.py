#!/usr/bin/env python3
"""Finalize representative Gate-B audio and native-pixel precert evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from avengine.contracts.json_io import sha256_file


VISIBLE = {"visible_clear", "visible_occluded"}
HIDDEN = {"fully_occluded", "out_of_view"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def bound(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def audio_pair(spec):
    main = Path(spec["main_audio"]).resolve()
    gateb = Path(spec["gateb_audio"]).resolve()
    main_mix = main / "audio/binaural/mixture.wav"
    gateb_mix = gateb / "audio/binaural/mixture.wav"
    for root in (main, gateb):
        receipt = read(root / "research_receipt.json")
        if receipt.get("status") != "pass" or not receipt.get("research_only"):
            raise RuntimeError(f"audio receipt is not research pass: {root}")
    same = sha256_file(main_mix) == sha256_file(gateb_mix)
    policy = spec["policy"]
    if policy == "route_audio_must_change_consistently" and same:
        raise RuntimeError("route-swap audio unexpectedly remained identical")
    return {
        "policy": policy,
        "main_mixture": bound(main_mix),
        "gateb_mixture": bound(gateb_mix),
        "rerender_mixtures_identical": same,
        "decision": (
            "pass_route_audio_changed"
            if policy == "route_audio_must_change_consistently"
            else "reuse_main_audio_gateb_rerender_is_diagnostic_only"
        ),
    }


def pixel_states(path):
    value = read(path)
    return {
        slot: {
            int(frame["frame_index"]): frame["state"]
            for frame in record["frames"]
        }
        for slot, record in value["per_instance"].items()
    }


def pixel_case(profile_id, spec):
    states = pixel_states(spec["pixel_truth"])
    fact = read(spec["main_fact"])
    reasons = []
    gateb_gold = None
    if profile_id == "card11":
        visible = [states[f"source{i}"].get(30) for i in range(1, 4)]
        hidden = states["source4"].get(30)
        if any(state not in VISIBLE for state in visible):
            reasons.append("one_of_three_candidates_not_visible")
        if hidden not in HIDDEN:
            reasons.append("offscreen_source_is_visually_present")
        gateb_gold = fact.get("gatea_desired_truth")
    elif profile_id == "card15a":
        values = [states[f"source{i}"].get(30) for i in range(1, 5)]
        in_scene = sum(state in VISIBLE for state in values)
        callers = int(fact["open"]["truth_value"][1])
        gateb_gold = [in_scene, callers]
        if gateb_gold not in fact["mcq"]["options_space"]:
            reasons.append("gateb_gold_outside_main_mcq_option_space")
    elif profile_id == "card16":
        binding = [states[f"source{i}"].get(12) for i in range(1, 3)]
        final = [states[f"source{i}"].get(74) for i in range(1, 3)]
        if any(state not in VISIBLE for state in binding):
            reasons.append("gateb_first_caller_not_visible_at_binding")
        if final[0] == final[1]:
            reasons.append("gateb_final_states_not_distinct")
        gateb_gold = final[0]
        if gateb_gold == spec["main_gold"]:
            reasons.append("gateb_gold_did_not_flip")
    else:
        raise ValueError(profile_id)
    return {
        "status": "pass" if not reasons else "reject",
        "gateb_gold": gateb_gold,
        "rejection_reasons": reasons,
        "inputs": {
            "pixel_truth": bound(spec["pixel_truth"]),
            "main_fact": bound(spec["main_fact"]),
        },
        "states": states,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        print(f"refusing to overwrite: {args.output_root}", file=sys.stderr)
        return 2
    spec = read(args.spec)
    result = {
        "schema": "qa_v3_gateb_representative_precheck_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "audio": {
            name: audio_pair(value)
            for name, value in spec["audio"].items()
        },
        "pixel": {
            name: pixel_case(name, value)
            for name, value in spec["pixel"].items()
        },
        "boundary": (
            "Representative Gate-B precert evidence only; appearance twins "
            "reuse main audio, pixel-dependent twins require per-candidate "
            "native pixel joins before certification."),
    }
    args.output_root.mkdir(parents=True)
    output = args.output_root / "gateb_representative_manifest.json"
    write(output, result)
    print(json.dumps({
        "output": str(output),
        "audio": {
            key: value["decision"] for key, value in result["audio"].items()
        },
        "pixel": {
            key: value["status"] for key, value in result["pixel"].items()
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
