#!/usr/bin/env python3
"""Assemble one quota-complete room-centric QA-v3 research pilot manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from qa_v3_request import normalize_answer_forms


# Declared selection strata.  The answer option has always been balanced.  The
# camera height joined it on 2026-09-03: the solver falls back from the scene
# camera height to a taller pose when the clearance table calls the lower one
# blocked, so the taller height is not random - it concentrates in cluttered
# corners.  Owner kept that fallback, so the height is a nuisance variable that
# has to be balanced and declared instead of removed.  It is a secondary key:
# the answer balance still comes first and the height round-robins inside each
# answer group, because a small quota cannot always satisfy both exactly.
STRATA = ("mcq_truth_option", "camera_height_m")
STRATIFICATION_RULE = (
    "answer option balanced first (round robin across options), camera height "
    "round-robins inside each option; achieved counts are reported per profile "
    "and per room so any residual imbalance is visible rather than silent")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_forms(sources):
    """Normalize declared forms and refuse conflicting request metadata."""
    resolved = None
    resolved_source = None
    for source, value in sources:
        if value is None:
            continue
        forms = normalize_answer_forms(value)
        if resolved is None:
            resolved = forms
            resolved_source = source
        elif forms != resolved:
            raise ValueError(
                f"answer_forms conflict between {resolved_source}={resolved} "
                f"and {source}={forms}")
    return resolved


def _question_rows(path: Path, point_id: str, variant: str):
    """Read one point's rows from a generated request view, if present."""
    if not path.is_file():
        return None
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid question row {path}:{line_no}") from exc
            if not isinstance(value, Mapping):
                raise RuntimeError(
                    f"question row {path}:{line_no} is not an object")
            if str(value.get("point_id")) != str(point_id):
                continue
            if value.get("variant", "main") != variant:
                continue
            rows.append(dict(value))
    return rows


def _forms_from_question_rows(rows):
    if rows is None or not rows:
        return None
    forms = [row.get("form") for row in rows]
    if any(not isinstance(form, str) for form in forms):
        raise ValueError("question rows must carry string form values")
    return _resolve_forms([("questions.jsonl", forms)])


def _id_component(value):
    text = str(value)
    return f"{len(text)}:{text}"


def _scoped_key(prefix, *parts):
    """Length-prefix every component so the key is reversible and unambiguous."""
    return prefix + "".join(_id_component(part) for part in parts)


def _pilot_id(scene_id, profile_id, index):
    return _scoped_key("pilot:", scene_id, profile_id, f"{index:03d}")


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _point_signature(timeline):
    frames = timeline["frames"]
    indices = [0, 40, 74]
    camera = frames[0]["camera"]
    actors = []
    for frame_index in indices:
        actors.append(tuple(
            (
                state["source_slot_id"],
                tuple(float(value) for value in state["translation_ue_cm"]),
            )
            for state in frames[frame_index]["actor_states"]
        ))
    return (
        tuple(float(value) for value in camera["translation_ue_cm"]),
        float(camera["yaw_ue_deg"]),
        tuple(actors),
    )


def _program_path(point, batch, fact, *, gatea=False):
    if point.joinpath(
            "audio_program_gateA.json" if gatea else "audio_program.json").is_file():
        return point / ("audio_program_gateA.json" if gatea else "audio_program.json")
    owner = fact
    if gatea:
        owner = _read(point / "fact_record_gateA.json")
    program_id = owner["audio"]["program_id"]
    path = batch / "programs" / f"{program_id}.json"
    if not path.is_file():
        raise RuntimeError(f"missing AudioProgram for {point}: {path}")
    return path


def _candidate(point, scene_id, profile_id, answer_forms_hint=None):
    fact_path = point / "fact_record.json"
    timeline_path = point / "timeline.json"
    selection_path = point / "actor_selection.json"
    m1_path = point / "m1_capture_request.json"
    required = [fact_path, timeline_path, selection_path, m1_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"incomplete candidate {point}: {missing}")
    fact = _read(fact_path)
    if fact.get("scene_id") != scene_id or fact.get("profile_id") != profile_id:
        raise RuntimeError(f"candidate fact identity drift: {point}")
    timeline = _read(timeline_path)
    batch = point.parent
    main_program = _program_path(point, batch, fact)
    gatea_program = _program_path(point, batch, fact, gatea=True)
    gatea_fact = point / "fact_record_gateA.json"
    gateb = point / "gateB_intervention.json"

    questions_path = batch / "questions.jsonl"
    gatea_questions_path = batch / "questions_gateA.jsonl"
    main_rows = _question_rows(questions_path, point.name, "main")
    gatea_rows = _question_rows(gatea_questions_path, point.name, "gateA")
    if main_rows is not None and not main_rows:
        raise RuntimeError(
            f"candidate {point} has no main question row in {questions_path}")
    candidate_forms = _resolve_forms([
        ("fact.answer_forms", fact.get("answer_forms")),
        ("questions.jsonl", _forms_from_question_rows(main_rows)),
        ("matrix.answer_forms", answer_forms_hint),
    ])
    if candidate_forms is None:
        candidate_forms = [
            form for form in ("mcq", "open")
            if isinstance(fact.get(form), Mapping)
        ]
    if not candidate_forms:
        raise RuntimeError(f"candidate {point} declares no answer forms")
    for form in candidate_forms:
        if not isinstance(fact.get(form), Mapping):
            raise RuntimeError(
                f"candidate {point} is missing requested {form} fact")
    question_count = (
        len(main_rows) if main_rows is not None else len(candidate_forms)
    )
    gatea_forms = _forms_from_question_rows(gatea_rows)
    if gatea_forms is not None:
        _resolve_forms([
            ("candidate.answer_forms", candidate_forms),
            ("questions_gateA.jsonl", gatea_forms),
        ])
    gatea_question_count = 0 if gatea_rows is None else len(gatea_rows)
    mcq = fact.get("mcq") if isinstance(fact.get("mcq"), Mapping) else {}
    open_fact = fact.get("open") if isinstance(fact.get("open"), Mapping) else {}
    truth = mcq.get("truth_option")
    camera = fact.get("camera") or {}
    clearance = camera.get("clearance") or {}
    height = camera.get("height_m")
    return {
        "source_point": str(point.resolve()),
        "source_point_id": point.name,
        "camera_height_m": (None if height is None else float(height)),
        "camera_height_fallback_used": bool(clearance.get("fallback_used")),
        "scene_camera_height_m": (
            None if camera.get("scene_camera_height_m") is None
            else float(camera["scene_camera_height_m"])),
        "artifacts": {
            "actor_selection": str(selection_path.resolve()),
            "timeline": str(timeline_path.resolve()),
            "m1_request": str(m1_path.resolve()),
            "fact": str(fact_path.resolve()),
            "questions": (
                str(questions_path.resolve()) if questions_path.is_file()
                else None),
            "questions_gateA": (
                str(gatea_questions_path.resolve())
                if gatea_questions_path.is_file() else None),
            "gatea_fact": (
                str(gatea_fact.resolve()) if gatea_fact.is_file() else None),
            "main_program": str(main_program.resolve()),
            "gatea_program": str(gatea_program.resolve()),
            "gateb": str(gateb.resolve()) if gateb.is_file() else None,
        },
        "gateb_status": (
            "materialized" if gateb.is_file()
            else "existing_dual_source_twin_stage_not_materialized_here"),
        "answer_forms": candidate_forms,
        "question_count": question_count,
        "counterfactual_question_count": gatea_question_count,
        "mcq_truth_option": truth,
        "open_truth_value": open_fact.get("truth_value"),
        "geometry_signature": _point_signature(timeline),
    }


def _matrix_rows(root):
    matrix_path = root / "scene_profile_matrix.json"
    if not matrix_path.is_file():
        raise RuntimeError(f"missing matrix: {matrix_path}")
    matrix = _read(matrix_path)
    return matrix, matrix.get("matrix", [])


def _matrix_answer_forms(matrix, row, batch=None):
    sources = [
        ("row.answer_forms", row.get("answer_forms")),
        ("matrix.answer_forms", matrix.get("answer_forms")),
    ]
    request = matrix.get("question_request")
    if isinstance(request, Mapping):
        sources.append(("matrix.question_request.answer_forms",
                        request.get("answer_forms")))
        per_room = request.get("per_room")
        if isinstance(per_room, Mapping):
            sources.append(("matrix.question_request.per_room.answer_forms",
                            per_room.get("answer_forms")))
    if isinstance(batch, Mapping):
        batch_request = batch.get("question_request")
        if isinstance(batch_request, Mapping):
            sources.append(("batch.question_request.answer_forms",
                            batch_request.get("answer_forms")))
    return _resolve_forms(sources)


def collect_candidates(matrix_roots, *, include_requests=False):
    candidates = {}
    statuses = {}
    scenes = {}
    requests = {}
    for root in matrix_roots:
        matrix, rows = _matrix_rows(root)
        for scene in matrix.get("scenes", []):
            scenes[scene["scene_id"]] = scene
        for row in rows:
            key = (str(row["scene_id"]), str(row["profile_id"]))
            statuses.setdefault(key, []).append(row["attempt_status"])
            batch_manifest = row.get("batch_manifest")
            requested_cells = row.get("requested_cells")
            batch_doc = None
            if batch_manifest:
                batch_path = Path(batch_manifest).resolve()
                if batch_path.is_file():
                    batch_doc = _read(batch_path)
            answer_forms = _matrix_answer_forms(matrix, row, batch_doc)
            requests.setdefault(key, []).append({
                "requested_cells": requested_cells,
                "answer_forms": answer_forms,
                "attempt_status": row.get("attempt_status"),
                "matrix_root": str(Path(root).resolve()),
            })
            if not batch_manifest:
                continue
            batch = Path(batch_manifest).resolve().parent
            for point in sorted(batch.glob(f"{row['profile_id']}_*")):
                if not point.is_dir() or not (point / "fact_record.json").is_file():
                    continue
                candidates.setdefault(key, []).append(
                    _candidate(
                        point, row["scene_id"], row["profile_id"],
                        answer_forms_hint=answer_forms))
    if include_requests:
        return scenes, statuses, candidates, requests
    return scenes, statuses, candidates


def _height_key(item):
    """Secondary stratum: the camera height the candidate was rendered at."""
    height = item.get("camera_height_m")
    return "unknown" if height is None else f"{float(height):.3f}"


def _interleave_by_height(items):
    """Round-robin one answer group across its camera heights.

    Without this the first N candidates of an answer group can all share one
    height, which is exactly the concentration the fallback creates.
    """
    by_height = {}
    for item in items:
        by_height.setdefault(_height_key(item), []).append(item)
    if len(by_height) <= 1:
        return list(items)
    order = sorted(by_height)
    ordered = []
    offsets = {key: 0 for key in order}
    while len(ordered) < len(items):
        progressed = False
        for key in order:
            index = offsets[key]
            if index >= len(by_height[key]):
                continue
            ordered.append(by_height[key][index])
            offsets[key] += 1
            progressed = True
        if not progressed:
            break
    return ordered


def _balanced_choice(pool, quota):
    if len(pool) < quota:
        raise ValueError("candidate pool is smaller than quota")
    groups = {}
    for item in pool:
        key = json.dumps(item["mcq_truth_option"], sort_keys=True)
        groups.setdefault(key, []).append(item)
    groups = {key: _interleave_by_height(items) for key, items in groups.items()}
    if len(pool) == quota or len(groups) <= 1:
        flat = [item for key in sorted(groups) for item in groups[key]]
        return flat[:quota]
    selected = []
    offsets = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(selected) < quota:
        progressed = False
        for key in keys:
            index = offsets[key]
            if index >= len(groups[key]):
                continue
            selected.append(groups[key][index])
            offsets[key] += 1
            progressed = True
            if len(selected) == quota:
                break
        if not progressed:
            break
    if len(selected) != quota:
        raise RuntimeError(
            f"balanced selection stopped at {len(selected)} for quota {quota}")
    return selected


def _quota_for(request_rows, unique, explicit):
    if explicit is not None:
        return explicit, "explicit_per_profile", []
    requested = []
    for row in request_rows:
        value = row.get("requested_cells")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"requested_cells must be a non-negative integer: {value!r}")
        requested.append(value)
    if requested:
        # Supplementary matrix roots add candidates to the same room/profile;
        # each scheduler row states the target quota, so the largest explicit
        # target is the least surprising aggregate quota.
        return max(requested), "scheduler_requested_cells", requested
    return len(unique), "all_available_candidates", []


def _observed_status(statuses):
    for status in (
            "resource_unavailable", "profile_not_implemented", "pipeline_error",
            "pixel_rejected", "not_found_within_budget", "not_scheduled"):
        if status in statuses:
            return status
    return "not_found_within_budget"


def _common_forms(entries):
    values = {tuple(entry["answer_forms"])
             for entry in entries if entry.get("answer_forms")}
    return list(next(iter(values))) if len(values) == 1 else None


def assemble(*, matrix_roots, profiles, per_profile=None):
    if per_profile is not None:
        if (isinstance(per_profile, bool) or not isinstance(per_profile, int)
                or per_profile <= 0):
            raise ValueError("per_profile must be a positive integer")
    scenes, statuses, candidates, requests = collect_candidates(
        matrix_roots, include_requests=True)
    profile_ids = [profile["id"] for profile in profiles]
    room_manifests = {}
    global_truth = {}
    for scene_id, scene in sorted(scenes.items()):
        selected_profiles = {}
        room_count = 0
        for profile_id in profile_ids:
            key = (scene_id, profile_id)
            request_rows = requests.get(key, [])
            observed = statuses.get(key, [])
            pool = candidates.get(key, [])
            unique = []
            signatures = set()
            for item in pool:
                signature = json.dumps(
                    item["geometry_signature"], sort_keys=True)
                if signature in signatures:
                    continue
                signatures.add(signature)
                unique.append(item)
            quota, quota_source, requested_values = _quota_for(
                request_rows, unique, per_profile)
            forms = _resolve_forms([
                (f"request[{index}].answer_forms", row.get("answer_forms"))
                for index, row in enumerate(request_rows)
            ] + [
                (f"candidate[{index}].answer_forms",
                 item.get("answer_forms"))
                for index, item in enumerate(unique)
            ])
            if quota == 0:
                status = (_observed_status(observed)
                          if observed and any(value != "generated"
                                              for value in observed)
                          else "not_scheduled")
                selected_profiles[profile_id] = {
                    "status": status,
                    "selected_count": 0,
                    "available_unique_candidates": len(unique),
                    "observed_statuses": observed,
                    "requested_cells": quota,
                    "quota_source": quota_source,
                    "requested_cells_by_source": requested_values,
                    "answer_forms": forms,
                    "question_count": 0,
                    "counterfactual_question_count": 0,
                    "candidates": [],
                }
                continue
            if not unique:
                selected_profiles[profile_id] = {
                    "status": _observed_status(observed),
                    "selected_count": 0,
                    "available_unique_candidates": 0,
                    "observed_statuses": observed,
                    "requested_cells": quota,
                    "quota_source": quota_source,
                    "requested_cells_by_source": requested_values,
                    "answer_forms": forms,
                    "question_count": 0,
                    "counterfactual_question_count": 0,
                    "candidates": [],
                }
                continue
            if len(unique) < quota:
                raise RuntimeError(
                    f"{scene_id}/{profile_id}: only {len(unique)} unique "
                    f"candidates for quota {quota}")
            chosen = _balanced_choice(unique, quota)
            selected_forms = _resolve_forms([
                (f"request[{index}].answer_forms", row.get("answer_forms"))
                for index, row in enumerate(request_rows)
            ] + [
                (f"candidate[{index}].answer_forms",
                 item.get("answer_forms"))
                for index, item in enumerate(chosen)
            ])
            if selected_forms is None:
                selected_forms = forms
            for index, item in enumerate(chosen, start=1):
                item["pilot_id"] = _pilot_id(scene_id, profile_id, index)
            truth_counts = Counter(
                json.dumps(item["mcq_truth_option"], sort_keys=True)
                for item in chosen)
            height_counts = Counter(_height_key(item) for item in chosen)
            pool_height_counts = Counter(_height_key(item) for item in unique)
            question_count = sum(item["question_count"] for item in chosen)
            counterfactual_count = sum(
                item["counterfactual_question_count"] for item in chosen)
            selected_profiles[profile_id] = {
                "status": "selected",
                "selected_count": len(chosen),
                "available_unique_candidates": len(unique),
                "observed_statuses": observed,
                "requested_cells": quota,
                "quota_source": quota_source,
                "requested_cells_by_source": requested_values,
                "answer_forms": selected_forms,
                "question_count": question_count,
                "counterfactual_question_count": counterfactual_count,
                "mcq_truth_counts": dict(truth_counts),
                "camera_height_counts": dict(height_counts),
                "camera_height_counts_in_pool": dict(pool_height_counts),
                "camera_height_fallback_selected": sum(
                    1 for item in chosen if item["camera_height_fallback_used"]),
                "candidates": chosen,
            }
            global_truth[f"{scene_id}/{profile_id}"] = dict(truth_counts)
            room_count += len(chosen)
        room_entries = list(selected_profiles.values())
        room_manifests[scene_id] = {
            "schema": "qa_v3_room_pilot_manifest_v1",
            "status": "research_candidate",
            "qualification_claim": False,
            "scene": scene,
            "per_profile_quota": per_profile,
            "selected_candidate_count": room_count,
            "question_count": sum(
                entry.get("question_count", 0) for entry in room_entries),
            "counterfactual_question_count": sum(
                entry.get("counterfactual_question_count", 0)
                for entry in room_entries),
            "answer_forms": _common_forms(room_entries),
            "profiles": selected_profiles,
        }

    all_entries = [entry for room in room_manifests.values()
                   for entry in room["profiles"].values()]
    request_forms = _resolve_forms([
        (f"request[{index}].answer_forms", row.get("answer_forms"))
        for scene_requests in requests.values()
        for index, row in enumerate(scene_requests)
    ])
    manifest_forms = _common_forms(all_entries) or request_forms
    status_counts = Counter(entry["status"] for entry in all_entries)
    return {
        "schema": "qa_v3_room_centric_pilot_manifest_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "boundary": (
            "Quota-complete geometry/timeline/AudioProgram pilot selection; "
            "not pixel admission, missing-modality certification, human "
            "answerability, or formal dataset release."),
        "inputs": {
            "matrix_roots": [str(path.resolve()) for path in matrix_roots],
            "profile_ids": profile_ids,
            "per_profile_quota": per_profile,
            "answer_forms": manifest_forms,
        },
        "answer_forms": manifest_forms,
        "question_count": sum(
            room["question_count"] for room in room_manifests.values()),
        "counterfactual_question_count": sum(
            room["counterfactual_question_count"]
            for room in room_manifests.values()),
        "scene_count": len(room_manifests),
        "runnable_profile_count": status_counts["selected"],
        "resource_profile_count": status_counts["resource_unavailable"],
        "selected_candidate_count": sum(
            room["selected_candidate_count"]
            for room in room_manifests.values()),
        "per_scene_selected_counts": {
            scene_id: room["selected_candidate_count"]
            for scene_id, room in room_manifests.items()},
        "truth_counts": global_truth,
        "stratification": {
            "strata": list(STRATA),
            "rule": STRATIFICATION_RULE,
            "camera_height_counts": {
                f"{scene_id}/{profile_id}": counts
                for scene_id, room in room_manifests.items()
                for profile_id, entry in room["profiles"].items()
                for counts in [entry.get("camera_height_counts")] if counts},
            "status": "placeholder_research_declared_not_calibrated",
            "note": ("balancing the evaluation split by camera height is the "
                     "consumer's job; this manifest declares the strata and "
                     "reports what was achieved so the consumer can check it"),
        },
        "rooms": room_manifests,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--supplement-root", action="append", type=Path, default=[])
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--per-profile", type=int, default=None,
                        help="optional pilot subset; default uses each scheduler row requested_cells")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        print(f"refusing to overwrite: {args.output_root}", file=sys.stderr)
        return 2
    if args.per_profile is not None and args.per_profile <= 0:
        parser.error("--per-profile must be positive")
    profiles = _read(args.profiles)
    roots = [args.matrix_root, *args.supplement_root]
    manifest = assemble(
        matrix_roots=roots, profiles=profiles,
        per_profile=args.per_profile)
    args.output_root.mkdir(parents=True)
    _write(args.output_root / "pilot_manifest.json", manifest)
    for scene_id, room in manifest["rooms"].items():
        _write(
            args.output_root / "rooms" / scene_id / "room_pilot_manifest.json",
            room)
    print(json.dumps({
        "output": str(args.output_root),
        "scene_count": manifest["scene_count"],
        "selected_candidate_count": manifest["selected_candidate_count"],
        "per_scene_selected_counts": manifest["per_scene_selected_counts"],
        "question_count": manifest["question_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
