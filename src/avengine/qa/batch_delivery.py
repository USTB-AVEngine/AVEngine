"""Read completed native deliveries into batch diagnostics and review frames."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from avengine.capture.neutral_readback import validate_neutral_readback
from avengine.qa.answerability import (
    MeshHandle, line_of_sight, listener_azimuth_deg,
    max_concurrent_entities, separation_stats,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def achieved_from_facts(facts: Mapping[str, Any], profile: Mapping[str, Any] | None,
                        room_package: Mapping[str, Any]) -> dict[str, Any]:
    """Measure actual geometry/activity; unavailable comparisons stay explicit."""
    actors, events = facts["actors"], facts["events"]
    clock = facts["time"]
    count, fps, sr = int(clock["frame_count"]), float(clock["frame_rate_hz"]), int(clock["sample_rate_hz"])
    listener = facts["listener"]
    positions = np.asarray(listener["positions_m"], dtype=float)
    bases = listener["basis_m3"]
    if positions.shape != (count, 3) or len(bases) != count or not np.isfinite(positions).all():
        raise ValueError("actual listener readbacks do not cover the Episode clock")
    camera_static = bool(np.allclose(positions, positions[:1], atol=1e-8, rtol=0) and
                         all(basis == bases[0] for basis in bases))
    source_intervals = []
    spans = []
    missing_activity = []
    for event in events:
        intervals = event.get("source_activity_intervals_samples")
        if not intervals:
            missing_activity.append(event["event_id"])
            continue
        event_intervals = [(int(item["start_sample"]), int(item["end_sample_exclusive"]), event["actor_id"])
                           for item in intervals]
        source_intervals.extend(event_intervals)
        spans.append((min(item[0] for item in event_intervals),
                      max(item[1] for item in event_intervals), event))
    overlap = max_concurrent_entities(source_intervals) if source_intervals else None
    ordered_spans = sorted(spans, key=lambda span: (span[0], span[1], span[2]["event_id"]))
    gaps = [(right[0] - left[1]) / sr for left, right in zip(ordered_spans, ordered_spans[1:])]
    repeated = Counter((event["actor_id"], event.get("sound_asset_id")) for event in events)
    source_end = max((end for _, end, _ in source_intervals), default=None)
    wet_ends = [float(row["end_s"]) for row in facts.get("audio", {}).get("wet_tail_intervals", [])]
    result = {"authority": "actual_facts_listener_roots_emitters_pixel_visibility_and_audio_activity",
              "total_count": len(actors), "speaking_count": None if missing_activity else len({item[2] for item in source_intervals}),
              "actors_with_bound_events_count": len({event["actor_id"] for event in events}),
              "silent_actor_ids": sorted(set(actors) - {event["actor_id"] for event in events}),
              "camera_static": camera_static, "source_activity_missing_event_ids": missing_activity,
              "maximum_concurrent_source_activity": overlap,
              "minimum_gap_between_event_activity_spans_s": min(gaps) if gaps else None,
              "repeated_source_sound_pairs": [list(key) for key, value in repeated.items() if value > 1],
              "source_activity_tail_s": (int(clock["sample_count"]) - source_end) / sr if source_end is not None else None,
              "listener_wet_tail_s": float(clock["duration_seconds"]) - max(wet_ends) if wet_ends else None,
              "anchor_event_measurements": [], "requested_profile": deepcopy(profile),
              "profile_certified": False,
              "calibration": "placeholder; geometry/activity measurements are not human answerability"}
    if not profile:
        result["profile_comparison_status"] = "request_profile_missing"
        return result
    actor_order = [f"source{index + 1}" for index in range(int(profile["total_count"]))]
    if set(actor_order) != set(actors):
        raise ValueError("actual actors disagree with declared profile slots")
    geometry = room_package.get("static_geometry", {})
    mesh = None
    if geometry.get("vertices") and geometry.get("triangles"):
        mesh = MeshHandle.from_paths(geometry["vertices"], geometry["triangles"])
    azimuths = {}
    for actor_id, actor in actors.items():
        emitters = actor["emitter_positions_m"]
        if len(emitters) != count:
            raise ValueError(f"{actor_id} emitter readback is incomplete")
        azimuths[actor_id] = [listener_azimuth_deg({"position_m": positions[index], "basis": bases[index]}, point)
                              for index, point in enumerate(emitters)]
    los_cache = {}
    anchors = {actor_order[index] for index in profile["anchor_indices"]}
    for start_sample, end_sample, event in spans:
        actor_id = event["actor_id"]
        if actor_id not in anchors:
            continue
        start = int(math.floor(start_sample / sr * fps))
        end = int(math.ceil(end_sample / sr * fps))
        if not 0 <= start < end <= count:
            raise ValueError("actual source activity falls outside the native frame clock")
        values = azimuths[actor_id]
        other = {key: value for key, value in azimuths.items() if key != actor_id}
        if any(value is None for value in values) or any(v is None for array in other.values() for v in array):
            separation = {"status": "unmeasured", "reason": "horizontal_azimuth_undefined"}
        else:
            separation = separation_stats(values, other, [start, end], frame_rate_hz=fps,
                                          thresholds_deg=[profile["separation_floor_deg"]])
            lo, hi = profile["separation_bin_deg"]
            separation["requested_bin_deg"] = [lo, hi]
            separation["inside_requested_bin_all_frames"] = separation["min"] >= lo and separation["max"] < hi
        pixel_states, target_in_fov, los_states = [], [], []
        for frame in range(start, end):
            visible = facts.get("visibility", {}).get(actor_id, {}).get(str(frame))
            if visible is None:
                pixel_states.append("unmeasured")
                target_in_fov.append(None)
            else:
                pixel_states.append(visible["state"])
                target_in_fov.append(int(visible.get("target_pixels", 0)) > 0)
            emitter = actors[actor_id]["emitter_positions_m"][frame]
            key = tuple(positions[frame]) + tuple(emitter)
            if key not in los_cache:
                los_cache[key] = line_of_sight(mesh, positions[frame], emitter)
            los_states.append(los_cache[key])
        target_motion = [bool(value) for value in actors[actor_id]["moving"][start:end]]
        competitor_motion = [
            any(bool(actors[other_id]["moving"][frame]) for other_id in actors if other_id != actor_id)
            for frame in range(start, end)]
        result["anchor_event_measurements"].append({
            "event_id": event["event_id"], "actor_id": actor_id, "window_frames": [start, end],
            "window_source": "actual_audio_source_activity_extent", "separation": separation,
            "pixel_state_counts": dict(Counter(pixel_states)),
            "in_fov_frame_count": sum(value is True for value in target_in_fov),
            "missing_pixel_frames": sum(value is None for value in target_in_fov),
            "static_emitter_los_counts": dict(Counter(los_states)),
            "body_proxy_los": {"status": "unmeasured", "reason": "no native body-proxy point in facts"},
            "target_moving_frame_count": sum(target_motion),
            "competitor_moving_frame_count": sum(competitor_motion),
            "frame_count": end - start})
    result["profile_comparison_status"] = "actual_fields_measured_unavailable_fields_explicit"
    return result


def _review_frames(questions: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[int, list[str]]:
    frame_count = int(facts["time"]["frame_count"])
    selected = {0: ["episode_start"]}
    def visit(value, qid):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"query_frame", "query_start_frame", "query_end_frame", "anchor_end_frame",
                           "start_frame", "end_frame", "frame_index"} and isinstance(child, int) and not isinstance(child, bool):
                    if key == "query_frame" and not 0 <= child < frame_count:
                        raise ValueError("explicit question query frame is outside the Episode")
                    if child == frame_count and key in {"end_frame", "query_end_frame", "anchor_end_frame"}:
                        child -= 1
                    if 0 <= child < frame_count:
                        selected.setdefault(child, []).append(qid + ":" + key)
                elif isinstance(child, (Mapping, list)):
                    visit(child, qid)
        elif isinstance(value, list):
            for child in value:
                visit(child, qid)
    for question in questions["items"]:
        visit(question.get("evidence", {}), question["question_id"])
    return {frame: sorted(set(reasons)) for frame, reasons in sorted(selected.items())}


def finalize_batch_episode(episode_root: Path, manifest_entry: Mapping[str, Any], *,
                           repository: Path, review_root: Path | None = None) -> dict[str, Any]:
    """Validate one completed P9 delivery, run the existing auditor, save review frames."""
    episode_root, repository = Path(episode_root).resolve(), Path(repository).resolve()
    delivery = episode_root / "delivery"
    facts_path, questions_path = delivery / "facts.json", delivery / "questions.json"
    facts, questions = _read(facts_path), _read(questions_path)
    episode_id = manifest_entry["episode_id"]
    if facts["episode_id"] != episode_id or questions["episode_id"] != episode_id:
        raise ValueError("P9 facts/questions belong to another Episode")
    if facts.get("status") != "pass":
        raise ValueError("P9 facts validation did not pass")
    plan = _read(episode_root / "plan/episode_plan.json")
    package = _read(episode_root / "plan/room_package.json")
    if package["room_id"] != manifest_entry["room_id"]:
        raise ValueError("native room differs from preallocated room")
    expected_assets = {row["asset_id"] for row in manifest_entry["source_assignments"]}
    observed_assets = {row["asset_id"] for row in facts["actors"].values()}
    if expected_assets != observed_assets:
        raise ValueError("native source assets differ from preallocated assets")
    neutral = _read(episode_root / "capture/neutral_readback.json")
    validate_neutral_readback(neutral, plan=plan)
    refs = _read(delivery / "input_refs.json")
    if Path(refs["neutral_readback"]).resolve() != (episode_root / "capture/neutral_readback.json").resolve():
        raise ValueError("delivery consumed another native capture")
    wav_path = Path(facts["audio"]["path"]).resolve()
    pcm, rate = sf.read(wav_path, dtype="float64", always_2d=True)
    if pcm.shape != (facts["time"]["sample_count"], 2) or rate != facts["time"]["sample_rate_hz"]:
        raise ValueError("delivered lossless audio clock/channels disagree")
    if not np.isfinite(pcm).all():
        raise ValueError("nonfinite delivered PCM")
    peak = float(np.max(np.abs(pcm)))
    audio_level = {"path": str(wav_path), "sample_rate_hz": rate,
                   "shape": list(pcm.shape), "peak_abs": peak,
                   "peak_dbfs": 20 * math.log10(peak) if peak else None,
                   "rms_by_channel": np.sqrt(np.mean(pcm * pcm, axis=0)).tolist()}
    root = Path(review_root).resolve() if review_root else episode_root / "batch_review"
    root.mkdir(parents=True, exist_ok=False)
    profile = manifest_entry.get("requested_profile")
    achieved = achieved_from_facts(facts, profile, package)
    achieved_path = root / "achieved_conditions.json"
    _write(achieved_path, achieved)
    produced = Counter(item["qa_id"] for item in questions["items"])
    qa_ids = [f"QA-{index:02d}" for index in range(1, 25)]
    if set(questions.get("coverage_by_qa", {})) != set(qa_ids):
        raise ValueError("delivery is missing requested QA-type denominator entries")
    commands = [sys.executable, str(repository / "tools/qa/audit_binding_feasibility.py"),
                "--facts", str(facts_path), "--questions", str(questions_path),
                "--wav", str(wav_path), "--stems-dir", str(wav_path.parent),
                "--out", str(root / "audit_v2.json")]
    if package.get("acoustic_package"):
        commands.extend(["--acoustic-package", package["acoustic_package"]])
    with (root / "audit.log").open("x") as log:
        audit_run = subprocess.run(commands, cwd=repository, stdout=log, stderr=subprocess.STDOUT)
    audit = {"status": "completed" if audit_run.returncode == 0 else "failed",
             "returncode": audit_run.returncode, "command": commands,
             "path": str(root / "audit_v2.json"), "log": str(root / "audit.log")}
    # Frame extraction uses the delivered video; timestamps preserve the native
    # frame index. No fabricated image or planned appearance enters this bundle.
    preview = delivery / "preview.mp4"
    frames = _review_frames(questions, facts)
    image_root = root / "frames"
    image_root.mkdir()
    selector = "+".join(f"eq(n\\,{frame})" for frame in frames)
    extraction = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(preview),
                  "-vf", "select=" + selector, "-vsync", "0", "-start_number", "0",
                  str(image_root / "frame_%03d.png")]
    with (root / "frames.log").open("x") as log:
        subprocess.run(extraction, cwd=repository, stdout=log, stderr=subprocess.STDOUT, check=True)
    image_files = sorted(image_root.glob("frame_*.png"))
    if len(image_files) != len(frames) or any(path.stat().st_size == 0 for path in image_files):
        raise ValueError("review frame extraction count differs")
    frame_records = [{"frame_index": frame, "time_s": frame / facts["time"]["frame_rate_hz"],
                      "reasons": reasons, "path": str(image_files[index]), "source_video": str(preview)}
                     for index, (frame, reasons) in enumerate(frames.items())]
    _write(root / "frames.json", frame_records)
    assignments = manifest_entry["source_assignments"]
    allowed = {row["actor_id"]: set(row.get("sound_asset_ids", [])) for row in assignments}
    preallocation_present = all("sound_asset_ids" in row for row in assignments)
    sound_match = (all(event["actor_id"] in allowed and event["sound_asset_id"] in allowed[event["actor_id"]]
                       for event in facts["events"]) if preallocation_present else None)
    if sound_match is False:
        raise ValueError("delivered sound IDs differ from the preallocated actor identities")
    source_identity_keys = set()
    for binding in plan.get("voice_bindings", []):
        source_identity_keys.update(binding.get("sound_identity_keys", []))
        if binding.get("sound_identity_id"):
            source_identity_keys.add(binding["sound_identity_id"])
    result = {"episode_id": episode_id, "status": "delivered" if audit_run.returncode == 0 else "review_failed",
              "facts_path": str(facts_path), "questions_path": str(questions_path), "preview_path": str(preview),
              "room_id": package["room_id"], "room_family": package["family"], "review_root": str(root),
              "condition_profile": deepcopy(plan.get("condition_profile")),
              "profile_matches_request": plan.get("condition_profile") == profile,
              "sound_preallocation_matches": sound_match,
              "achieved_conditions": achieved, "achieved_conditions_source": str(achieved_path),
              "produced_count_by_qa": {qa: produced[qa] for qa in qa_ids},
              "unmet_quota_by_qa": {qa: max(0, 1 - produced[qa]) for qa in qa_ids},
              "audio_level": audio_level, "source_activity_intervals_samples": facts["audio"].get("source_activity_intervals_samples"),
              "wet_tail_intervals": facts["audio"].get("wet_tail_intervals"), "audit": audit,
              "review_frames": str(root / "frames.json"),
              "grouped_split_record": {"record_id": episode_id, "episode_id": episode_id,
                                      "visual_episode_id": episode_id, "room_id": package["room_id"],
                                      "route_ids": [], "sound_identity_ids": sorted(source_identity_keys)},
              "human_listening": {"status": "pending_human", "reviewer": None, "notes": None},
              "qualification_claim": False}
    _write(root / "review.json", result)
    return result


def finalize_batch_outputs(output_root: Path, manifest: Mapping[str, Any],
                           execution_summary: Mapping[str, Any], *,
                           repository: Path, summary_root: Path | None = None) -> dict[str, Any]:
    """Produce full-denominator batch artifacts after the background queue ends."""
    from avengine.qa.batch_coverage import build_batch_coverage, write_batch_coverage
    from avengine.qa.batch_manifest import collect_batch_outcomes, grouped_splits

    output_root, repository = Path(output_root).resolve(), Path(repository).resolve()
    summary_root = Path(summary_root).resolve() if summary_root else output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=False)
    entries = {row["episode_id"]: row for row in manifest["episodes"]}
    records, contexts, split_records, audio_rows, previews = [], [], [], [], []
    failures = []
    for raw in execution_summary["episodes"]:
        episode_id = raw["episode_id"]
        entry = entries[episode_id]
        episode_root = Path(raw.get("episode_output_root") or (Path(raw["attempt_root"]) / "episode"))
        review = raw.get("review")
        if isinstance(review, Mapping):
            record = deepcopy(dict(review))
            records.append(record)
            if review.get("grouped_split_record"):
                split_records.append(review["grouped_split_record"])
            audio_rows.append({"episode_id": episode_id, "audio_level": review.get("audio_level"),
                               "source_activity_intervals_samples": review.get("source_activity_intervals_samples"),
                               "wet_tail_intervals": review.get("wet_tail_intervals"),
                               "audit": review.get("audit")})
        else:
            if raw["status"] == "blocked":
                status = "preallocation_blocked" if raw.get("reason_code") == "preallocation_gap" else "resource_failed"
            elif raw["status"] == "review_failed":
                status = "review_failed"
            elif not (episode_root / "execution_commands.json").is_file():
                status = "planning_failed"
            elif not (episode_root / "capture/neutral_readback.json").is_file():
                status = "capture_failed"
            else:
                status = "delivery_failed"
            record = {"episode_id": episode_id, "status": status,
                      "failure_reason": raw.get("reason"), "failure_code": raw.get("reason_code"),
                      "failure_path": raw.get("stderr_log"), "executor_outcome": deepcopy(dict(raw)),
                      "stage_classification": "from_executor_result_and_existing_stage_artifacts"}
            plan_path = episode_root / "plan/episode_plan.json"
            if plan_path.is_file():
                record["condition_profile"] = _read(plan_path).get("condition_profile")
            records.append(record)
        if record["status"] != "delivered":
            failures.append(record)
        facts_path, questions_path = episode_root / "delivery/facts.json", episode_root / "delivery/questions.json"
        if not facts_path.is_file() or not questions_path.is_file():
            continue
        facts = _read(facts_path)
        if facts.get("status") != "pass" or facts.get("episode_id") != episode_id:
            continue
        package = _read(episode_root / "plan/room_package.json")
        plan = _read(episode_root / "plan/episode_plan.json")
        refs = _read(episode_root / "delivery/input_refs.json")
        bindings = {row["actor_id"]: row for row in plan.get("voice_bindings", [])}
        sound_events = []
        # Event IDs are from the actual P6 audio program, not new synthetic events.
        report = _read(Path(refs["audio_report"]))
        report_events = {row.get("event_id"): row for row in report.get("events", [])}
        for event in facts["events"]:
            binding = bindings.get(event["actor_id"], {})
            report_event = report_events.get(event["event_id"], {})
            sound_events.append({
                "event_id": event["event_id"], "actor_id": event["actor_id"],
                "sound_asset_id": event.get("sound_asset_id"),
                "source_endpoint_id": event.get("source_endpoint_id"),
                "sound_class": event.get("sound_class"),
                "dry_audio_origin": deepcopy(report_event.get("audio")),
                "original_source_path": binding.get("source_pcm_path") or binding.get("source_origin"),
                "speaker_id": binding.get("speaker_id"),
                "source_identity_keys": deepcopy(binding.get("sound_identity_keys", [])),
                "source_refs": {"audio_report": refs["audio_report"], "audio_program": refs["audio_program"]}})
        contexts.append({"episode_id": episode_id, "room_id": package["room_id"], "family": package["family"],
                         "facts": str(facts_path), "questions": str(questions_path),
                         "source_refs": refs, "sound_events": sound_events})
        preview = episode_root / "delivery/preview.mp4"
        if preview.is_file():
            previews.append({"episode_id": episode_id, "room_family": package["family"],
                             "preview_path": str(preview), "audio_path": facts["audio"].get("path")})
    request = manifest["episodes"][0]["request"]
    coverage_manifest = {"schema": "avengine_qa_batch_episode_input_manifest_v1",
                         "asset_inventory": request["source_registry"],
                         "runtime_registry": request["source_registry"],
                         "room_catalog": request["room_catalog"], "episodes": contexts}
    _write(summary_root / "coverage_inputs.json", coverage_manifest)
    coverage = build_batch_coverage(coverage_manifest, repository=repository)
    coverage_paths = write_batch_coverage(coverage, summary_root / "coverage")
    joined = collect_batch_outcomes(manifest, records)
    _write(summary_root / "batch_outcomes.json", joined)
    _write(summary_root / "failed_episodes.json", failures)
    _write(summary_root / "audio_levels_and_activity.json", audio_rows)
    splits = grouped_splits(split_records, ratios={"train": 0.8, "eval": 0.2}, seed=int(manifest["seed"]))
    _write(summary_root / "grouped_splits.json", splits)
    # Prefer one actual clip from each family, then fill to five with distinct
    # completed clips. This selects review material and never manufactures ratings.
    chosen, families = [], set()
    for row in previews:
        if row["room_family"] not in families:
            chosen.append(row)
            families.add(row["room_family"])
        if len(chosen) == 5:
            break
    selected_ids = {row["episode_id"] for row in chosen}
    for row in previews:
        if len(chosen) == 5:
            break
        if row["episode_id"] not in selected_ids:
            chosen.append(row)
            selected_ids.add(row["episode_id"])
    listening = {"requested_count": 5, "available_count": len(chosen),
                 "unmet_count": max(0, 5 - len(chosen)),
                 "records": [{**row, "status": "pending_human", "reviewer": None, "heard": None,
                              "source_assignment_clear": None, "notes": None} for row in chosen]}
    _write(summary_root / "five_clip_listening_pending.json", listening)
    result = {"status": "machine_artifacts_complete", "batch_id": manifest["batch_id"],
              "episode_denominator": len(manifest["episodes"]), "completed_fact_bundles": len(contexts),
              "outcome_counts": joined["outcome_counts"], "summary_root": str(summary_root),
              "coverage_outputs": coverage_paths, "coverage_denominator": coverage["denominator"],
              "failed_episode_count": len(failures), "grouped_split_counts": splits["actual_counts"],
              "grouped_split_unassigned_count": splits["unassigned_count"],
              "five_clip_listening": str(summary_root / "five_clip_listening_pending.json"),
              "human_listening_status": "pending_human", "qualification_claim": False}
    _write(summary_root / "summary.json", result)
    return result
