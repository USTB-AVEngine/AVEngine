#!/usr/bin/env python3
"""Revalidate an existing room-pilot manifest against prescale QA-v3 rules.

The audit is deliberately non-mutating: it classifies retained candidates by
the minimum next action after the 2026-09-02 modality-leakage fixes.  It does
not rewrite historical facts, relabel media, or promote research candidates.

Card1 angles are recomputed from the final timeline; solver plan values are
reported only as planning values and never decide a status.  The scoring
parameters that the audit executed (THETA_HALF, T_FULL, T_HALF and the derived
card8 minimum first-call separation) are embedded in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import audio_profiles as AP
from design_qa_v3_scene_batch import recompute_azimuth
from scene_sampler import circular_gap_deg, open_angle_candidate_scores_zero


APPEARANCE_GATEB = {"card4R", "card7", "card8", "card9", "card15b"}
CARD1 = {"card1F", "card1B"}
CORE = {"card1F", "card1B", "card5R", "card6R", "card7", "card8",
        "card9", "card11", "card16"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_span(event):
    start = int(event["start_sample"])
    end = int(event["end_sample_exclusive"])
    return start * 3 // 3200, min(75, -(-(end * 3) // 3200))


def _band_edges_from_labels(labels):
    """Parse '[lo, hi)' MCQ labels back into numeric half-open bands."""
    bands = []
    for label in labels:
        text = str(label).strip()
        if not (text.startswith("[") and text.endswith(")")):
            return None
        lo, hi = text[1:-1].split(",")
        bands.append((float(lo), float(hi)))
    return bands


def scoring_snapshot(params):
    """The parameters this audit actually executes; fail closed when absent."""
    if "THETA_HALF" not in params:
        raise ValueError("params missing explicit THETA_HALF")
    card8 = AP.card8_scoring_params(params)
    return {
        "THETA_FULL": (float(params["THETA_FULL"])
                       if "THETA_FULL" in params else None),
        "THETA_HALF": float(params["THETA_HALF"]),
        "T_FULL": card8["T_FULL"],
        "T_HALF": card8["T_HALF"],
        "T_FULL_status": card8["T_FULL_status"],
        "card8_min_first_call_separation_s": card8[
            "min_first_call_separation_s"],
        "card8_min_first_call_separation_rule": card8[
            "min_first_call_separation_rule"],
        "card8_certification_policy": card8["certification_policy"],
        "card8_wide_tolerance_role": card8["wide_tolerance_role"],
    }


def _profile_id(candidate):
    value = candidate.get("pilot_id", "").split("__")
    if len(value) >= 3:
        return value[1]
    return str(candidate.get("profile_id", ""))


def audit_candidate(candidate, params):
    profile = _profile_id(candidate)
    artifacts = candidate["artifacts"]
    fact = _read(artifacts["fact"])
    actions, reasons, checks = set(), [], {}

    if profile in CARD1:
        target_events = [event for event in fact["audio"]["events"]
                         if event.get("role") == "target_actor"]
        checks["target_event_count"] = len(target_events)
        if len(target_events) != 1:
            actions.add("regenerate_audio_and_fact")
            reasons.append("card1_target_sounded_outside_identity_anchor")
        timeline = _read(artifacts["timeline"])
        # Realized values come from the final timeline only.  The solver plan
        # (generation_checks.az_anchor_deg) is a planning value: recorded for
        # the deviation report, never trusted for a decision.
        anchor = recompute_azimuth(
            timeline, fact["target_slot"], int(fact["anchor_frame"]))
        query = recompute_azimuth(
            timeline, fact["target_slot"], int(fact["query_frame"]))
        gap = circular_gap_deg(anchor, query)
        zero = open_angle_candidate_scores_zero(
            anchor, query, float(params["THETA_HALF"]))
        planned = (fact.get("generation_checks") or {}).get("az_anchor_deg")
        checks.update({
            "angle_source": "final_timeline_recompute",
            "anchor_azimuth_deg": anchor,
            "query_azimuth_deg": query,
            "anchor_query_gap_deg": gap,
            "anchor_answer_scores_zero": zero,
            "planned_anchor_azimuth_deg_planning_value_only": planned,
            "planned_vs_realized_anchor_deviation_deg": (
                circular_gap_deg(float(planned), anchor)
                if planned is not None else None),
        })
        if not zero:
            actions.add("resample_geometry")
            reasons.append("audible_anchor_angle_receives_open_credit")
        allocated = fact.get("generation_checks", {}).get(
            "allocated_anchor_band")
        checks["allocated_anchor_band"] = allocated
        if allocated is None:
            actions.add("pool_joint_rebalance")
            reasons.append("anchor_band_query_band_stratum_not_recorded")
        else:
            inside = float(allocated[0]) <= anchor < float(allocated[1])
            checks["realized_anchor_in_allocated_band"] = inside
            if not inside:
                actions.add("resample_geometry")
                reasons.append("realized_anchor_outside_allocated_band")
        bands = _band_edges_from_labels(
            (fact.get("mcq") or {}).get("options_space") or [])
        gold = (fact.get("mcq") or {}).get("truth_option")
        if bands and gold is not None:
            labels = list(fact["mcq"]["options_space"])
            realized_band = next(
                (labels[index] for index, (lo, hi) in enumerate(bands)
                 if lo <= query < hi), None)
            checks["realized_query_band"] = realized_band
            if realized_band != gold:
                actions.add("resample_geometry")
                reasons.append("realized_query_outside_gold_answer_band")

    if profile == "card5R":
        target_events = [event for event in fact["audio"]["events"]
                         if event.get("role") == "target_actor"]
        checks["target_event_count"] = len(target_events)
        if len(target_events) != 1:
            actions.add("regenerate_audio_and_fact")
            reasons.append("card5R_target_has_preanchor_audio")

    if profile == "card8":
        target = float(fact["truth"]["first_onset_s"])
        other = float(fact["truth"]["non_target_first_onset_s"])
        separation = abs(target - other)
        scoring = AP.card8_scoring_params(params)
        strict_needed = scoring["min_first_call_separation_s"]
        checks.update({"first_onset_separation_s": separation,
                       "min_first_call_separation_s": strict_needed,
                       "min_first_call_separation_rule": scoring[
                           "min_first_call_separation_rule"],
                       "T_FULL": scoring["T_FULL"],
                       "T_HALF": scoring["T_HALF"]})
        if separation <= strict_needed:
            actions.add("regenerate_audio_and_fact")
            reasons.append("card8_first_call_separation_not_above_minimum")
        if fact["open"].get("certification_policy") != \
                "strict_full_credit_only":
            actions.add("rewrite_question_metadata")
            reasons.append("card8_strict_certification_metadata_missing")

    if profile == "card11":
        program = _read(artifacts["main_program"])
        spans = [_frame_span(event) for event in program["events"]]
        checks["event_frame_spans"] = spans
        if not spans or not all(lo <= 30 < hi for lo, hi in spans):
            actions.add("regenerate_audio_and_fact")
            reasons.append("card11_audio_does_not_span_pixel_binding_frame")
        acceptance = fact.get("pixel_acceptance", {})
        checks["source4_pixel_acceptance"] = acceptance.get("source4")
        if acceptance.get("source4") != "fully_occluded":
            actions.add("rerun_pixel_join")
            reasons.append("card11_out_of_view_negative_still_allowed")

    if profile == "card16":
        gateb = candidate.get("gateb", {}) or {}
        if gateb.get("gold_status") == "pixel_pending" or \
                fact.get("truth_status") == "pending_native_pixel_join":
            actions.add("render_pixel_then_quota_select")
            reasons.append("card16_gold_requires_native_pixel_join")

    if profile in APPEARANCE_GATEB:
        policy = (candidate.get("gateb", {}) or {}).get("audio_policy")
        checks["gateb_audio_policy"] = policy
        if policy != "appearance_canonical_anchor_audio_must_be_identical":
            actions.add("canonical_gateb_rerender")
            reasons.append("appearance_gateb_uses_precanonical_audio_policy")

    if profile == "card15a":
        actions.add("demote_from_main")
        reasons.append("card15a_visual_fact_and_gateb_option_space_degenerate")

    if profile == "card17":
        actions.add("keep_future_extension")
        reasons.append("card17_not_in_prescale_main_submission_scope")

    if not actions:
        status = "prescale_structure_pass"
    elif "resample_geometry" in actions:
        status = "geometry_resample_required"
    elif actions <= {"rewrite_question_metadata", "pool_joint_rebalance"}:
        status = "metadata_or_pool_reselection_required"
    elif actions == {"render_pixel_then_quota_select"}:
        status = "pixel_pending"
    elif actions == {"demote_from_main"}:
        status = "demoted_from_main"
    elif actions == {"keep_future_extension"}:
        status = "future_extension"
    else:
        status = "media_regeneration_required"
    return {
        "pilot_id": candidate["pilot_id"],
        "profile_id": profile,
        "is_core_profile": profile in CORE,
        "status": status,
        "actions": sorted(actions),
        "reasons": reasons,
        "checks": checks,
    }


def audit(pilot, params, params_source=None):
    scoring = scoring_snapshot(params)
    records = []
    for room in pilot["rooms"].values():
        for profile in room["profiles"].values():
            for candidate in profile.get("candidates", []):
                records.append(audit_candidate(candidate, params))
    status = Counter(record["status"] for record in records)
    by_profile = defaultdict(Counter)
    for record in records:
        by_profile[record["profile_id"]][record["status"]] += 1
    return {
        "schema": "qa_v3_prescale_candidate_revalidation_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "scoring_params": scoring,
        "params_source": params_source,
        "angle_policy": (
            "card1 anchor/query angles recomputed from the final timeline; "
            "solver plan values are planning values only"),
        "candidate_count": len(records),
        "counts_by_status": dict(sorted(status.items())),
        "counts_by_profile_and_status": {
            profile: dict(sorted(counts.items()))
            for profile, counts in sorted(by_profile.items())},
        "records": sorted(records, key=lambda row: row["pilot_id"]),
        "boundary": (
            "Non-mutating revalidation of historical research candidates. "
            "A pass is structural only and is not modality certification or "
            "dataset admission."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    params_path = args.params.resolve()
    params_source = {"path": str(params_path),
                     "sha256": _sha256(params_path)}
    try:
        result = audit(_read(args.pilot_manifest), _read(params_path),
                       params_source)
    except (ValueError, AP.AudioProfileError) as exc:
        print(f"audit refused: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "candidate_count": result["candidate_count"],
                      "counts_by_status": result["counts_by_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
