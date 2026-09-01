#!/usr/bin/env python3
"""Assemble one quota-complete room-centric QA-v3 research pilot manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


RESOURCE_PROFILES = {"card12", "card13", "card14"}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def _candidate(point, scene_id, profile_id):
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
    truth = fact.get("mcq", {}).get("truth_option")
    return {
        "source_point": str(point.resolve()),
        "source_point_id": point.name,
        "artifacts": {
            "actor_selection": str(selection_path.resolve()),
            "timeline": str(timeline_path.resolve()),
            "m1_request": str(m1_path.resolve()),
            "fact": str(fact_path.resolve()),
            "gatea_fact": (
                str(gatea_fact.resolve()) if gatea_fact.is_file() else None),
            "main_program": str(main_program.resolve()),
            "gatea_program": str(gatea_program.resolve()),
            "gateb": str(gateb.resolve()) if gateb.is_file() else None,
        },
        "gateb_status": (
            "materialized" if gateb.is_file()
            else "existing_dual_source_twin_stage_not_materialized_here"),
        "mcq_truth_option": truth,
        "open_truth_value": fact.get("open", {}).get("truth_value"),
        "geometry_signature": _point_signature(timeline),
    }


def _matrix_rows(root):
    matrix_path = root / "scene_profile_matrix.json"
    if not matrix_path.is_file():
        raise RuntimeError(f"missing matrix: {matrix_path}")
    matrix = _read(matrix_path)
    return matrix, matrix.get("matrix", [])


def collect_candidates(matrix_roots):
    candidates = {}
    statuses = {}
    scenes = {}
    for root in matrix_roots:
        matrix, rows = _matrix_rows(root)
        for scene in matrix.get("scenes", []):
            scenes[scene["scene_id"]] = scene
        for row in rows:
            key = (row["scene_id"], row["profile_id"])
            statuses.setdefault(key, []).append(row["attempt_status"])
            batch_manifest = row.get("batch_manifest")
            if not batch_manifest:
                continue
            batch = Path(batch_manifest).resolve().parent
            for point in sorted(batch.glob(f"{row['profile_id']}_*")):
                if not point.is_dir() or not (point / "fact_record.json").is_file():
                    continue
                candidates.setdefault(key, []).append(
                    _candidate(point, row["scene_id"], row["profile_id"]))
    return scenes, statuses, candidates


def _balanced_choice(pool, quota):
    if len(pool) < quota:
        raise ValueError("candidate pool is smaller than quota")
    groups = {}
    for item in pool:
        key = json.dumps(item["mcq_truth_option"], sort_keys=True)
        groups.setdefault(key, []).append(item)
    if len(pool) == quota or len(groups) <= 1:
        return pool[:quota]
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


def assemble(*, matrix_roots, profiles, per_profile):
    scenes, statuses, candidates = collect_candidates(matrix_roots)
    profile_ids = [profile["id"] for profile in profiles]
    room_manifests = {}
    global_truth = {}
    for scene_id, scene in sorted(scenes.items()):
        selected_profiles = {}
        room_count = 0
        for profile_id in profile_ids:
            key = (scene_id, profile_id)
            if profile_id in RESOURCE_PROFILES:
                selected_profiles[profile_id] = {
                    "status": "resource_unavailable",
                    "selected_count": 0,
                    "observed_statuses": statuses.get(key, []),
                }
                continue
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
            if len(unique) < per_profile:
                raise RuntimeError(
                    f"{scene_id}/{profile_id}: only {len(unique)} unique "
                    f"candidates for quota {per_profile}")
            chosen = _balanced_choice(unique, per_profile)
            truth_counts = Counter(
                json.dumps(item["mcq_truth_option"], sort_keys=True)
                for item in chosen)
            for index, item in enumerate(chosen, start=1):
                item["pilot_id"] = (
                    f"{scene_id}__{profile_id}__{index:03d}")
            selected_profiles[profile_id] = {
                "status": "selected",
                "selected_count": len(chosen),
                "available_unique_candidates": len(unique),
                "observed_statuses": statuses.get(key, []),
                "mcq_truth_counts": dict(truth_counts),
                "candidates": chosen,
            }
            global_truth[f"{scene_id}/{profile_id}"] = dict(truth_counts)
            room_count += len(chosen)
        room_manifests[scene_id] = {
            "schema": "qa_v3_room_pilot_manifest_v1",
            "status": "research_candidate",
            "qualification_claim": False,
            "scene": scene,
            "per_profile_quota": per_profile,
            "selected_candidate_count": room_count,
            "profiles": selected_profiles,
        }
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
        },
        "scene_count": len(room_manifests),
        "runnable_profile_count": len(profile_ids) - len(RESOURCE_PROFILES),
        "resource_profile_count": len(RESOURCE_PROFILES),
        "selected_candidate_count": sum(
            room["selected_candidate_count"]
            for room in room_manifests.values()),
        "per_scene_selected_counts": {
            scene_id: room["selected_candidate_count"]
            for scene_id, room in room_manifests.items()},
        "truth_counts": global_truth,
        "rooms": room_manifests,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--supplement-root", action="append", type=Path, default=[])
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--per-profile", type=int, default=6)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        print(f"refusing to overwrite: {args.output_root}", file=sys.stderr)
        return 2
    if args.per_profile <= 0:
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
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
