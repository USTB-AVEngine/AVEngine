#!/usr/bin/env python3
"""Card1 realized conditional tables and best-response unimodal baselines.

Read-only report over **generated** card1F / card1B manifests.  For every
profile x form x split x missing_modality it reports the anchor-band x
answer-band joint table, the structurally empty cells, the conditional answer
distribution, the best-response baseline an attacker with only the remaining
modality could reach, the per-room contribution and the largest single-room
share.  Nothing here is a gate: it describes the pool that was actually
generated so that the declared chance level for a single-modality probe is the
conditional one, not the nominal 1/k.

Inputs are generation artifacts only (facts, batch manifests, scheduler
matrices, the design-layer smoke report).  Model predictions, probe results
and scored outcomes are refused when they appear in an input record, and there
is no option to supply them.  Baselines are recomputed from the supplied
manifests every time; no benchmark constant is hardcoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_open_answers import circular_deg  # noqa: E402

CARD1 = ("card1F", "card1B")
MISSING_MODALITIES = ("video", "audio", "audio_and_video")
FORBIDDEN_KEYS = frozenset({
    "model_answer", "model_answers", "prediction", "predictions",
    "probe", "probe_result", "probe_results", "accuracy",
    "mean_scorer_score", "model_outcome", "model_score", "scores",
    "empirical_majority_baseline", "empirical_constant_baseline",
})
DEFAULT_GRID_STEP_DEG = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_band_label(label: str) -> tuple[float, float]:
    text = str(label).strip()
    if not (text.startswith("[") and text.endswith(")")):
        raise ValueError(f"not a half-open band label: {label!r}")
    lo, hi = text[1:-1].split(",")
    return float(lo), float(hi)


def band_label_for(value: float, labels: list[str]) -> str | None:
    for label in labels:
        lo, hi = parse_band_label(label)
        if lo <= value < hi:
            return label
    return None


def assert_no_model_outcomes(record: dict, owner: str) -> None:
    """Refuse any record that carries model predictions or probe results."""
    found = sorted(FORBIDDEN_KEYS & set(record))
    if found:
        raise ValueError(
            f"{owner}: input carries model outcome fields {found}; this "
            "report only reads generation artifacts")


def item_from_fact(fact: dict, *, room: str, split: str, source: str) -> dict | None:
    """Reduce one main fact record to the fields the report needs."""
    if fact.get("variant", "main") != "main":
        return None
    if fact.get("profile_id") not in CARD1:
        return None
    assert_no_model_outcomes(fact, f"{source}:{fact.get('point_id')}")
    labels = list(fact["mcq"]["options_space"])
    truth = fact["truth"]
    realized = fact.get("realized_generation_checks")
    if realized:
        anchor_deg = float(realized["main"]["anchor_azimuth_deg"])
        anchor_source = "realized_generation_checks"
    else:
        planned = (fact.get("generation_checks") or {}).get("az_anchor_deg")
        if planned is None:
            raise ValueError(
                f"{source}:{fact.get('point_id')}: no anchor azimuth recorded")
        anchor_deg = float(planned)
        anchor_source = "planned_solver_value_no_realized_record"
    motion = fact.get("motion") or {}
    return {
        "point_id": fact["point_id"],
        "room": room,
        "profile_id": fact["profile_id"],
        "split": split,
        "labels": labels,
        "anchor_azimuth_deg": anchor_deg,
        "anchor_source": anchor_source,
        "anchor_band": band_label_for(anchor_deg, labels)
        or "outside_declared_bands",
        "answer_band": str(fact["mcq"]["truth_option"]),
        "query_azimuth_deg": float(truth["query_azimuth_deg"]),
        "other_query_azimuth_deg": (
            float(truth["other_slot_azimuth_deg"])
            if truth.get("other_slot_azimuth_deg") is not None else None),
        "target_coat": fact.get("target_coat"),
        "target_moves_more": motion.get("target_moves_more"),
    }


def load_facts_file(path: Path, *, split_of, room_override=None) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fact = json.loads(line)
        room = room_override or str(fact.get("scene_id") or "unknown_room")
        item = item_from_fact(fact, room=room,
                              split=split_of(fact.get("point_id"), room),
                              source=str(path))
        if item is not None:
            items.append(item)
    return items


def load_scheduler_matrix(path: Path, *, split_of) -> list[dict]:
    matrix = _read(path)
    items = []
    for row in matrix.get("matrix", []):
        assert_no_model_outcomes(row, f"{path}:{row.get('profile_id')}")
        if row.get("profile_id") not in CARD1:
            continue
        manifest = row.get("batch_manifest")
        if not manifest:
            continue
        facts = Path(manifest).parent / "facts.jsonl"
        if not facts.is_file():
            continue
        items.extend(load_facts_file(
            facts, split_of=split_of, room_override=str(row["scene_id"])))
    return items


def circular_score(candidate_deg: float, truth_deg: float, theta_full: float,
                   theta_half: float) -> float:
    error = circular_deg(candidate_deg, truth_deg)
    return 1.0 if error <= theta_full else 0.5 if error <= theta_half else 0.0


def best_constant_angle(truths: list[float], theta_full: float,
                        theta_half: float, step: float) -> tuple[float, float]:
    """Grid-search the angle maximising the mean actual-scorer score."""
    if not truths:
        return float("nan"), 0.0
    best_angle, best_score = None, -1.0
    count = int(round(360.0 / step))
    for index in range(count):
        angle = -180.0 + index * step
        score = sum(circular_score(angle, truth, theta_full, theta_half)
                    for truth in truths) / len(truths)
        if score > best_score + 1e-12:
            best_angle, best_score = angle, score
    return round(best_angle, 6), best_score


def joint_table(items: list[dict], labels: list[str]) -> dict:
    table = {anchor: {answer: 0 for answer in labels} for anchor in labels}
    outside = Counter()
    for item in items:
        anchor = item["anchor_band"]
        if anchor not in table:
            outside[anchor] += 1
            continue
        table[anchor][item["answer_band"]] += 1
    empty = [{"anchor_band": anchor, "answer_band": answer,
              "same_band_diagonal": anchor == answer}
             for anchor in labels for answer in labels
             if table[anchor][answer] == 0]
    return {"counts": table, "structural_empty_cells": empty,
            "anchor_outside_declared_bands": dict(outside)}


def conditional_distribution(table_counts: dict) -> dict:
    result = {}
    for anchor, row in table_counts.items():
        total = sum(row.values())
        result[anchor] = {
            "n": total,
            "answer_fraction": ({answer: count / total for answer, count in
                                 row.items()} if total else {}),
        }
    return result


def audio_only_baselines(items, labels, theta_full, theta_half, step):
    """Attacker hears the identity anchor, so it knows the anchor band."""
    table = joint_table(items, labels)
    counts = table["counts"]
    n = len(items)
    by_anchor = defaultdict(list)
    for item in items:
        by_anchor[item["anchor_band"]].append(item)
    mcq_best = sum(max(row.values()) for row in counts.values()) + sum(
        table["anchor_outside_declared_bands"].values()) * 0  # outside rows do not add
    exclusion = 0.0
    open_expected = 0.0
    per_anchor = {}
    repeat_anchor_score = 0.0
    for anchor, rows in by_anchor.items():
        answers = Counter(item["answer_band"] for item in rows)
        non_empty = len([value for value in answers.values() if value > 0])
        exclusion += len(rows) / non_empty if non_empty else 0.0
        truths = [item["query_azimuth_deg"] for item in rows]
        angle, score = best_constant_angle(truths, theta_full, theta_half, step)
        open_expected += score * len(rows)
        repeat_anchor_score += sum(
            circular_score(item["anchor_azimuth_deg"], item["query_azimuth_deg"],
                           theta_full, theta_half) for item in rows)
        per_anchor[anchor] = {
            "n": len(rows),
            "mcq_best_answer_band": answers.most_common(1)[0][0],
            "mcq_best_answer_share": answers.most_common(1)[0][1] / len(rows),
            "open_best_constant_angle_deg": angle,
            "open_best_constant_expected_score": score,
        }
    return {
        "attacker_observes": "identity anchor azimuth band from binaural audio",
        "joint_table": table,
        "conditional_answer_distribution": conditional_distribution(counts),
        "mcq": {
            "best_response_conditional_baseline": mcq_best / n if n else None,
            "best_response_numerator": mcq_best,
            "best_response_denominator": n,
            "structural_exclusion_uniform_baseline": exclusion / n if n else None,
            "nominal_uniform_baseline": 1.0 / len(labels),
            "caveat": ("in-sample best response over the generated pool; "
                       "small strata make it an optimistic upper bound"),
        },
        "open": {
            "best_response_conditional_expected_score": (
                open_expected / n if n else None),
            "repeat_anchor_angle_expected_score": (
                repeat_anchor_score / n if n else None),
            "method": "grid search over candidate angles against the realized "
                      "query angles of each anchor stratum under the actual "
                      "two-tier circular scorer",
            "caveat": ("in-sample best response over the generated pool; with "
                       "a handful of items per anchor stratum this is an "
                       "optimistic upper bound, not an out-of-sample estimate"),
        },
        "per_anchor_band": per_anchor,
    }


def video_only_baselines(items, labels, theta_full, theta_half):
    """Attacker sees both dogs but cannot tell which one barked last."""
    n = len(items)

    def chosen_angle(item, rule):
        target = item["query_azimuth_deg"]
        other = item["other_query_azimuth_deg"]
        if other is None:
            return None
        if rule == "pick_yellow":
            return target if item["target_coat"] == "yellow" else other
        if rule == "pick_black_and_white":
            return target if item["target_coat"] == "black-and-white" else other
        if rule == "pick_dog_that_moves_more":
            return target if item["target_moves_more"] else other
        if rule == "pick_dog_that_moves_less":
            return other if item["target_moves_more"] else target
        if rule == "pick_more_lateral_dog":
            return target if abs(target) >= abs(other) else other
        if rule == "pick_more_central_dog":
            return target if abs(target) < abs(other) else other
        if rule == "pick_leftmost_dog":
            return min(target, other)
        if rule == "pick_rightmost_dog":
            return max(target, other)
        raise ValueError(rule)

    rules = ("pick_yellow", "pick_black_and_white", "pick_dog_that_moves_more",
             "pick_dog_that_moves_less", "pick_more_lateral_dog",
             "pick_more_central_dog", "pick_leftmost_dog", "pick_rightmost_dog")
    per_rule = {}
    for rule in rules:
        mcq_hits = open_total = 0.0
        usable = 0
        for item in items:
            angle = chosen_angle(item, rule)
            if angle is None:
                continue
            usable += 1
            chosen_band = band_label_for(angle, labels)
            mcq_hits += 1.0 if chosen_band == item["answer_band"] else 0.0
            open_total += circular_score(angle, item["query_azimuth_deg"],
                                         theta_full, theta_half)
        per_rule[rule] = {
            "n": usable,
            "mcq_accuracy": mcq_hits / usable if usable else None,
            "open_expected_score": open_total / usable if usable else None,
        }
    best_mcq = max((value["mcq_accuracy"] for value in per_rule.values()
                    if value["mcq_accuracy"] is not None), default=None)
    best_open = max((value["open_expected_score"] for value in per_rule.values()
                     if value["open_expected_score"] is not None), default=None)
    return {
        "attacker_observes": "both dogs' positions, coats and motion in the "
                             "video; not which one barked last",
        "candidate_set_size": 2,
        "structural_two_candidate_baseline": 0.5,
        "nominal_uniform_baseline": 1.0 / len(labels),
        "mcq": {"best_single_rule_accuracy": best_mcq},
        "open": {"best_single_rule_expected_score": best_open},
        "per_rule": per_rule,
        "covariate_balance": {
            "target_coat": dict(Counter(str(item["target_coat"])
                                        for item in items)),
            "target_moves_more": dict(Counter(str(item["target_moves_more"])
                                              for item in items)),
        },
        "n": n,
    }


def text_only_baselines(items, labels, theta_full, theta_half, step):
    n = len(items)
    answers = Counter(item["answer_band"] for item in items)
    angle, score = best_constant_angle(
        [item["query_azimuth_deg"] for item in items], theta_full, theta_half,
        step)
    return {
        "attacker_observes": "question text and option labels only",
        "answer_distribution": dict(answers),
        "mcq": {"majority_answer_baseline": (
            answers.most_common(1)[0][1] / n if n else None),
                "nominal_uniform_baseline": 1.0 / len(labels)},
        "open": {"best_constant_angle_deg": angle,
                 "best_constant_expected_score": score if n else None},
        "n": n,
    }


def room_contributions(items, labels):
    rooms = defaultdict(list)
    for item in items:
        rooms[item["room"]].append(item)
    n = len(items)
    per_room = {}
    for room, rows in sorted(rooms.items()):
        table = joint_table(rows, labels)["counts"]
        best = sum(max(row.values()) for row in table.values())
        per_room[room] = {
            "n": len(rows),
            "share": len(rows) / n if n else None,
            "joint_table_counts": table,
            "audio_only_mcq_best_response": best / len(rows) if rows else None,
        }
    return {
        "rooms": per_room,
        "room_count": len(rooms),
        "max_single_room_share": (max(value["share"] for value in per_room.values())
                                  if per_room else None),
    }


def analyse_group(items, *, theta_full, theta_half, step):
    labels = list(items[0]["labels"])
    for item in items:
        if list(item["labels"]) != labels:
            raise ValueError("answer band labels differ inside one group")
    anchor_sources = dict(Counter(item["anchor_source"] for item in items))
    audio = audio_only_baselines(items, labels, theta_full, theta_half, step)
    video = video_only_baselines(items, labels, theta_full, theta_half)
    text = text_only_baselines(items, labels, theta_full, theta_half, step)
    rooms = room_contributions(items, labels)
    return {
        "n": len(items),
        "answer_band_labels": labels,
        "anchor_angle_sources": anchor_sources,
        "joint_table_anchor_by_answer": audio["joint_table"],
        "conditional_answer_distribution": audio["conditional_answer_distribution"],
        "by_missing_modality": {
            "video": {"form": {"mcq": audio["mcq"], "open": audio["open"]},
                      "attacker_observes": audio["attacker_observes"],
                      "per_anchor_band": audio["per_anchor_band"]},
            "audio": {"form": {"mcq": video["mcq"], "open": video["open"]},
                      "attacker_observes": video["attacker_observes"],
                      "structural_two_candidate_baseline": video[
                          "structural_two_candidate_baseline"],
                      "per_rule": video["per_rule"],
                      "covariate_balance": video["covariate_balance"]},
            "audio_and_video": {"form": {"mcq": text["mcq"], "open": text["open"]},
                                "attacker_observes": text["attacker_observes"],
                                "answer_distribution": text["answer_distribution"]},
        },
        "rooms": rooms,
    }


def flatten_rows(profile, split, group):
    """One row per profile x form x split x missing_modality."""
    rows = []
    for missing in MISSING_MODALITIES:
        block = group["by_missing_modality"][missing]
        for form in ("mcq", "open"):
            metrics = block["form"][form]
            rows.append({
                "profile_id": profile, "form": form, "split": split,
                "missing_modality": missing, "n": group["n"],
                "metrics": metrics,
                "room_count": group["rooms"]["room_count"],
                "max_single_room_share": group["rooms"]["max_single_room_share"],
            })
    return rows


def analyse_items(items, *, theta_full, theta_half, step):
    groups = defaultdict(list)
    for item in items:
        groups[(item["profile_id"], item["split"])].append(item)
    detail = {}
    rows = []
    for (profile, split), rows_in in sorted(groups.items()):
        group = analyse_group(rows_in, theta_full=theta_full,
                              theta_half=theta_half, step=step)
        detail[f"{profile}|{split}"] = group
        rows.extend(flatten_rows(profile, split, group))
    return detail, rows


def analyse_smoke_report(report: dict) -> dict:
    """MCQ-only reproduction from the design-layer smoke's aggregated tables.

    The smoke report keeps anchor->answer band counts but no angles, so only
    the audio-only MCQ best response is computable here.
    """
    out = {}
    for scene_id, per_profile in report.get("results", {}).items():
        for profile, block in per_profile.items():
            if profile not in CARD1:
                continue
            assert_no_model_outcomes(block, f"{scene_id}:{profile}")
            table = defaultdict(Counter)
            labels = set()
            for key, count in block.get(
                    "anchor_answer_band_distribution", {}).items():
                anchor, answer = [part.strip() for part in key.split("->")]
                table[anchor][answer] += int(count)
                labels.update([anchor, answer])
            labels = sorted(labels, key=lambda text: float(
                text.strip("()").split(",")[0]))
            n = sum(sum(row.values()) for row in table.values())
            best = sum(max(row.values()) for row in table.values())
            counts = {anchor: {answer: table[anchor].get(answer, 0)
                               for answer in labels} for anchor in labels}
            empty = [{"anchor_band": anchor, "answer_band": answer,
                      "same_band_diagonal": anchor == answer}
                     for anchor in labels for answer in labels
                     if counts[anchor][answer] == 0]
            out[f"{scene_id}|{profile}"] = {
                "scene_id": scene_id, "profile_id": profile, "n": n,
                "joint_table_counts": counts,
                "structural_empty_cells": empty,
                "audio_only_mcq_best_response": best / n if n else None,
                "audio_only_mcq_best_response_fraction": f"{best}/{n}",
                "nominal_uniform_baseline": 1.0 / len(labels) if labels else None,
                "open_baselines": None,
                "open_baselines_reason": (
                    "aggregated smoke tables carry no realized angles"),
            }
    return out


def build_split_lookup(assignment: dict | None):
    def split_of(point_id, room):
        if not assignment:
            return "unsplit"
        return str(assignment.get(f"{room}/{point_id}",
                                  assignment.get(point_id, "unassigned")))
    return split_of


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", action="append", type=Path, default=[],
                        help="facts.jsonl of one scene batch (room from scene_id)")
    parser.add_argument("--scheduler-matrix", action="append", type=Path,
                        default=[], help="scene_profile_matrix.json")
    parser.add_argument("--smoke-report", action="append", type=Path,
                        default=[], help="design-layer scene generalization "
                                         "smoke JSON (MCQ reproduction only)")
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--split-assignment", type=Path,
                        help="optional JSON {point_id or room/point_id: split}")
    parser.add_argument("--grid-step-deg", type=float,
                        default=DEFAULT_GRID_STEP_DEG)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    if not (args.facts or args.scheduler_matrix or args.smoke_report):
        parser.error("supply --facts, --scheduler-matrix or --smoke-report")
    params = _read(args.params)
    missing = [key for key in ("THETA_FULL", "THETA_HALF") if key not in params]
    if missing:
        print(f"params missing {missing}", file=sys.stderr)
        return 2
    theta_full, theta_half = float(params["THETA_FULL"]), float(params["THETA_HALF"])
    assignment = _read(args.split_assignment) if args.split_assignment else None
    split_of = build_split_lookup(assignment)
    inputs = []
    items = []
    try:
        for path in args.facts:
            items.extend(load_facts_file(path, split_of=split_of))
            inputs.append({"kind": "facts", "path": str(path.resolve()),
                           "sha256": _sha256(path)})
        for path in args.scheduler_matrix:
            items.extend(load_scheduler_matrix(path, split_of=split_of))
            inputs.append({"kind": "scheduler_matrix", "path": str(path.resolve()),
                           "sha256": _sha256(path)})
        smoke = {}
        for path in args.smoke_report:
            smoke.update(analyse_smoke_report(_read(path)))
            inputs.append({"kind": "smoke_report", "path": str(path.resolve()),
                           "sha256": _sha256(path)})
        detail, rows = analyse_items(
            items, theta_full=theta_full, theta_half=theta_half,
            step=args.grid_step_deg) if items else ({}, [])
    except ValueError as exc:
        print(f"report refused: {exc}", file=sys.stderr)
        return 2
    result = {
        "schema": "qa_v3_card1_conditional_baseline_report_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "is_gate": False,
        "params_snapshot": {
            "THETA_FULL": theta_full, "THETA_HALF": theta_half,
            "grid_step_deg": args.grid_step_deg,
            "params_path": str(args.params.resolve()),
            "params_sha256": _sha256(args.params),
        },
        "inputs": inputs,
        "forbidden_inputs": sorted(FORBIDDEN_KEYS),
        "item_count": len(items),
        "rows": rows,
        "groups": detail,
        "smoke_report_reproduction": smoke,
        "boundary": (
            "Descriptive conditional baselines over generated candidates; "
            "recomputed from the supplied manifests, not benchmark constants. "
            "Not a gate, not modality certification, not dataset admission."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "items": len(items), "rows": len(rows),
                      "smoke_groups": len(smoke)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
