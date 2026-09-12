"""Read completed native deliveries into batch diagnostics and review frames."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from avengine.capture.neutral_readback import validate_neutral_readback
from avengine.rooms.qa_evidence import (
    annotate_achieved_conditions_visibility,
    annotate_pixel_visibility_semantics,
)
from avengine.qa.answerability import (
    MeshHandle, line_of_sight, listener_azimuth_deg,
    max_concurrent_entities, separation_stats,
)
from avengine.dataset.source_capabilities import combination_key, source_family
from avengine.qa.unified_catalog import iter_unified_items


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def attach_visibility_semantics(
    achieved: Mapping[str, Any],
    pixel_truth: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run pixel-visibility annotators after achieved_from_facts / compiled truth."""
    if not isinstance(pixel_truth, Mapping):
        return deepcopy(dict(achieved)), None
    annotated_truth = annotate_pixel_visibility_semantics(pixel_truth)
    annotated_achieved = annotate_achieved_conditions_visibility(
        achieved, annotated_truth
    )
    return annotated_achieved, annotated_truth


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
                           "start_frame", "end_frame", "frame_index", "frame"} and isinstance(child, int) and not isinstance(child, bool):
                    if key in {"query_frame", "frame"} and not 0 <= child < frame_count:
                        raise ValueError("explicit question query frame is outside the Episode")
                    if child == frame_count and key in {"end_frame", "query_end_frame", "anchor_end_frame"}:
                        child -= 1
                    if 0 <= child < frame_count:
                        selected.setdefault(child, []).append(qid + ":" + key)
                elif key == "query_window_frames" and isinstance(child, list) and len(child) == 2:
                    start, end = child
                    if not all(isinstance(v, int) and not isinstance(v, bool) for v in child) or not 0 <= start < end <= frame_count:
                        raise ValueError("explicit question query window is outside the Episode")
                    selected.setdefault(start, []).append(qid + ":query_window_start")
                    selected.setdefault(end - 1, []).append(qid + ":query_window_end")
                elif isinstance(child, (Mapping, list)):
                    visit(child, qid)
        elif isinstance(value, list):
            for child in value:
                visit(child, qid)
    for question in questions["items"]:
        visit(question.get("evidence", {}), question["question_id"])
    return {frame: sorted(set(reasons)) for frame, reasons in sorted(selected.items())}



def _failed_episode_asset_ids(episode_root: Path, entry: Mapping[str, Any], raw: Mapping[str, Any]) -> list[str]:
    """Collect intended asset IDs from the manifest row, request, or written plan."""
    seen: set[str] = set()
    ids: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            ids.append(value)

    for row in entry.get("source_assignments") or []:
        if isinstance(row, Mapping):
            add(row.get("asset_id"))
    request = entry.get("request")
    if not isinstance(request, Mapping):
        request = raw.get("request")
    if isinstance(request, Mapping):
        for item in request.get("source_asset_ids") or []:
            add(item)
    plan_path = Path(episode_root) / "plan" / "episode_plan.json"
    if plan_path.is_file():
        try:
            plan = _read(plan_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            plan = {}
        actors = plan.get("actors") if isinstance(plan, Mapping) else None
        if isinstance(actors, Mapping):
            actors = list(actors.values())
        visual = plan.get("visual_plan") if isinstance(plan, Mapping) else None
        if not actors and isinstance(visual, Mapping):
            actors = visual.get("actors")
        if isinstance(actors, list):
            for actor in actors:
                if isinstance(actor, Mapping):
                    add(actor.get("asset_id"))
    return ids


def _failed_episode_room_id(episode_root: Path, entry: Mapping[str, Any]) -> str | None:
    room_id = entry.get("room_id")
    if isinstance(room_id, str) and room_id:
        return room_id
    package_path = Path(episode_root) / "plan" / "room_package.json"
    if package_path.is_file():
        try:
            package = _read(package_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            package = {}
        if isinstance(package, Mapping) and isinstance(package.get("room_id"), str):
            return package["room_id"]
    return None


def _import_apply_exposure_gate():
    from avengine.qa.exposure_gate import apply_exposure_gate
    return apply_exposure_gate


def apply_review_exposure_gate(review: Mapping[str, Any], episode_root: Path) -> dict:
    """Run the exposure gate; ImportError is review_failed, not a silent skip."""
    try:
        apply_exposure_gate = _import_apply_exposure_gate()
    except ImportError as exc:
        result = deepcopy(dict(review))
        result["status"] = "review_failed"
        result["reason"] = f"exposure_gate.status=unavailable: cannot import avengine.qa.exposure_gate ({exc})"
        result["exposure_gate"] = {
            "status": "unavailable",
            "threshold_kind": "placeholder",
            "frames": [],
            "frame_source": {"kind": "unavailable", "path": None, "note": "import_error"},
        }
        result["frame_source"] = result["exposure_gate"]["frame_source"]
        return result
    gated = apply_exposure_gate(review, episode_root)
    if not isinstance(gated, Mapping):
        raise TypeError("apply_exposure_gate must return a mapping")
    result = dict(gated)
    block = result.get("exposure_gate")
    source = block.get("frame_source") if isinstance(block, Mapping) else None
    result["frame_source"] = source
    return result


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
    pixel_truth_path = Path(refs["pixel_visibility_truth"]) if isinstance(
        refs.get("pixel_visibility_truth"), str
    ) else (episode_root / "capture/pixel_visibility_truth.json")
    pixel_truth = _read(pixel_truth_path) if pixel_truth_path.is_file() else None
    achieved, _annotated_truth = attach_visibility_semantics(achieved, pixel_truth)
    achieved_path = root / "achieved_conditions.json"
    _write(achieved_path, achieved)
    produced = Counter(item["qa_id"] for item in questions["items"])
    from avengine.qa.unified_catalog import QA_IDS
    legacy_ids = [qa_id for qa_id in QA_IDS if qa_id != "QA-25"]
    default_ids = list(QA_IDS) if questions.get("catalog_version", "20260906") >= "20260909" else legacy_ids
    qa_ids = list(plan.get("request", {}).get("qa_ids") or default_ids)
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
    result = apply_review_exposure_gate(result, episode_root)
    _write(root / "review.json", result)
    return result


def finalize_batch_outputs(output_root: Path, manifest: Mapping[str, Any],
                           execution_summary: Mapping[str, Any], *,
                           repository: Path, summary_root: Path | None = None) -> dict[str, Any]:
    """Produce full-denominator batch artifacts after the background queue ends."""
    from avengine.qa.batch_coverage import build_batch_coverage, write_batch_coverage
    from avengine.qa.batch_manifest import collect_batch_outcomes, grouped_splits
    from avengine.qa.failure_accounting import classify_failure

    output_root, repository = Path(output_root).resolve(), Path(repository).resolve()
    summary_root = Path(summary_root).resolve() if summary_root else output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=False)
    entries = {row["episode_id"]: row for row in manifest["episodes"]}
    records, contexts, split_records, audio_rows, previews = [], [], [], [], []
    failures = []

    def classify_failed_record(
        raw: Mapping[str, Any],
        *,
        status: str | None = None,
        reason: str | None = None,
        reason_code: str | None = None,
    ) -> dict[str, Any]:
        status = status or raw.get("status")
        reason = reason if reason is not None else (
            raw.get("failure_reason") or raw.get("reason") or raw.get("review_error")
            or raw.get("failure_code") or ""
        )
        reason_code = reason_code if reason_code is not None else raw.get("reason_code")
        declared_gap = raw.get("gap_state")
        if reason_code == "unclassified_failure":
            declared_gap = None
        result = classify_failure(
            failure_stage=raw.get("failure_stage"),
            reason=str(reason),
            reason_code=reason_code if isinstance(reason_code, str) else None,
            status=status if isinstance(status, str) else None,
            declared_gap_state=declared_gap if isinstance(declared_gap, str) else None,
        )
        existing = raw.get("diagnostic")
        diagnostic = {
            **result["diagnostic"],
            **(deepcopy(dict(existing)) if isinstance(existing, Mapping) else {}),
        }
        return {
            "failure_stage": result["failure_stage"],
            "failure_reason": result["failure_reason"],
            "failure_code": result.get("reason_code") or reason_code,
            "gap_state": result["gap_state"],
            "diagnostic": diagnostic,
        }

    for raw in execution_summary["episodes"]:
        episode_id = raw["episode_id"]
        entry = entries[episode_id]
        episode_root = Path(raw.get("episode_output_root") or (Path(raw["attempt_root"]) / "episode"))
        review = raw.get("review")
        if isinstance(review, Mapping):
            record = deepcopy(dict(review))
            if review.get("status") != "delivered":
                review_reason = (
                    review.get("failure_reason")
                    or review.get("reason")
                    or review.get("error")
                )
                fields = classify_failed_record(
                    raw,
                    status="review_failed",
                    reason=str(review_reason) if review_reason else None,
                    reason_code=(
                        review.get("reason_code")
                        if isinstance(review.get("reason_code"), str)
                        else None
                    ),
                )
                record.update(fields)
                record.setdefault("episode_id", episode_id)
                record.setdefault("status", "review_failed")
                record.setdefault("failure_path", raw.get("stderr_log"))
                record.setdefault("room_id", _failed_episode_room_id(episode_root, entry))
                record.setdefault("asset_ids", _failed_episode_asset_ids(episode_root, entry, raw))
                record["executor_outcome"] = deepcopy(dict(raw))
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
            fields = classify_failed_record(raw, status=status)
            record = {
                "episode_id": episode_id,
                "status": status,
                **fields,
                "failure_path": raw.get("stderr_log"),
                "executor_outcome": deepcopy(dict(raw)),
                "stage_classification": "from_executor_result_and_existing_stage_artifacts",
                "room_id": _failed_episode_room_id(episode_root, entry),
                "asset_ids": _failed_episode_asset_ids(episode_root, entry, raw),
            }
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
    failed_coverage = []
    for record in failures:
        if not isinstance(record, Mapping):
            continue
        gap_state = record.get("gap_state")
        if gap_state not in {"interface_not_implemented", "evidence_missing_or_unsampled"}:
            continue
        room_id = record.get("room_id")
        if not isinstance(room_id, str) or not room_id:
            continue
        failed_coverage.append({
            "episode_id": record.get("episode_id"),
            "room_id": room_id,
            "asset_ids": list(record.get("asset_ids") or []),
            "gap_state": gap_state,
            "failure_stage": record.get("failure_stage"),
            "failure_reason": record.get("failure_reason") or record.get("failure_code"),
            "reason_code": record.get("failure_code"),
            "diagnostic": deepcopy(record.get("diagnostic")),
        })
    coverage_manifest = {"schema": "avengine_qa_batch_episode_input_manifest_v1",
                         "asset_inventory": request["source_registry"],
                         "runtime_registry": request["source_registry"],
                         "room_catalog": request["room_catalog"], "episodes": contexts,
                         "failed_episodes": failed_coverage}
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


# ---------------------------------------------------------------------------
# V1 achieved coverage
#
# The V1 question bank is counted here from artifacts that actually exist: a
# retained group library, a delivered export, or a fresh run's question sets.
# Every count uses the same publication predicate the delivery boundary uses
# (``binding_delivery`` item status plus per-form status), so a "valid main
# question" means the same thing in the feedback table and in the exported
# dataset. Requested quota never enters this side.
# ---------------------------------------------------------------------------

V1_ACHIEVED_SCHEMA = "avengine_qa_v1_achieved_coverage_v1"
V1_QUESTION_FORMS = ("mcq", "open")
V1_QUESTION_KINDS = ("main", "angle_followup")
UNRESOLVED_SOURCE_FAMILY = "unresolved_source_family"
UNOBSERVED_BRANCH = "branch_not_observable_from_published_answer"


class V1AchievedCoverageError(ValueError):
    """An achieved-coverage input is missing or contradicts itself."""


def valid_question_forms(item: Mapping[str, Any]) -> list[str]:
    """Return the forms the delivery boundary would actually publish.

    This mirrors ``binding_delivery`` exactly: a form counts only when the
    generator emitted it and its own ``form_status`` is ``pass``. A form the
    export would drop must not be counted here either.
    """
    declared = item.get("forms") or {}
    status = item.get("form_status") or {}
    return sorted(
        form
        for form in V1_QUESTION_FORMS
        if form in declared and (status.get(form) or {}).get("status") == "pass"
    )


# How a declared key branch shows up in a published answer. Three real shapes,
# measured against the retained catalog rather than assumed:
#
#  * identity      - the answer token is the branch token (QA-06/07/08/09/15/17/24).
#  * answer_map    - the answer vocabulary differs from the branch vocabulary.
#                    QA-05 publishes yes/no for "did they overlap", which is the
#                    overlap/disjoint branch under a different name; QA-20
#                    publishes an actor id or none_of_visible, so any actor
#                    id is the visible-candidate branch.
#  * modalities    - QA-25 asks the same bearing question of audio, video or
#                    both, so its branch is the item's required modalities and
#                    is not readable from the numeric answer at all.
#
# A branch that cannot be read this way is reported as unobservable; it is never
# assigned to a branch on the strength of the request that asked for it.
BRANCH_OBSERVATION_RULES = {
    "QA-05": {"kind": "answer_map", "map": {"yes": "overlap", "no": "disjoint"}},
    "QA-20": {
        "kind": "answer_map",
        "map": {"none_of_visible": "none_of_them", "none_of_them": "none_of_them"},
        "default": "visible_candidate",
    },
    "QA-25": {
        "kind": "modalities",
        "map": {"audio": "A", "video": "V", "audio+video": "AV"},
    },
}


def _answer_tokens(item: Mapping[str, Any]) -> list[str]:
    tokens = []
    forms = item.get("forms") or {}
    for form in V1_QUESTION_FORMS:
        block = forms.get(form)
        if not isinstance(block, Mapping) or "truth" not in block:
            continue
        value = block.get("truth")
        if isinstance(value, bool):
            tokens.append("yes" if value else "no")
        elif isinstance(value, (str, int)):
            tokens.append(str(value))
    return tokens


def observed_branch(
    item: Mapping[str, Any],
    branches: Sequence[str],
    *,
    qa_id: str | None = None,
    rules: Mapping[str, Any] | None = None,
) -> str | None:
    """Return which declared key branch this produced item actually realizes.

    The branch is read off what the item published, so a request that asked for
    a moving speaker but produced a still episode is not counted as branch
    coverage. When the published answer does not identify the branch this
    returns None and the caller records the item as branch-unobservable.
    """
    if not branches:
        return None
    branches = tuple(str(value) for value in branches)
    rules = BRANCH_OBSERVATION_RULES if rules is None else rules
    rule = rules.get(str(qa_id)) if qa_id is not None else None
    if rule is not None and rule.get("kind") == "modalities":
        modalities = item.get("required_modalities")
        if not modalities:
            return None
        key = "+".join(sorted(str(value) for value in modalities))
        branch = (rule.get("map") or {}).get(key)
        return branch if branch in branches else None
    tokens = _answer_tokens(item)
    if rule is not None and rule.get("kind") == "answer_map":
        mapping = rule.get("map") or {}
        for token in tokens:
            if token in mapping:
                branch = mapping[token]
                return branch if branch in branches else None
        default = rule.get("default")
        if tokens and default in branches:
            return default
        return None
    for token in tokens:
        if token in branches:
            return token
    return None


def question_set_rows(question_set: Mapping[str, Any], *, branches_for_qa=None) -> list[dict[str, Any]]:
    """Publishable question rows of one generated question set.

    ``branches_for_qa`` defaults to the shared branch table in
    ``avengine.qa.generation_conditions`` so the condition compiler and this
    counter never disagree about what a key branch is.
    """
    if branches_for_qa is None:
        from avengine.qa.generation_conditions import branches_for as branches_for_qa
    angle_ids = {
        str(entry.get("question_id"))
        for entry in question_set.get("angle_followups") or []
        if isinstance(entry, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for item in iter_unified_items(question_set):
        if not isinstance(item, Mapping):
            continue
        qa_id = str(item.get("qa_id"))
        if item.get("status") != "pass":
            continue
        forms = valid_question_forms(item)
        if not forms:
            continue
        branches = tuple(branches_for_qa(qa_id))
        rows.append(
            {
                "qa_id": qa_id,
                "question_id": str(item.get("question_id")),
                "kind": "angle_followup" if str(item.get("question_id")) in angle_ids else "main",
                "forms": forms,
                "branches_expected": list(branches),
                "branch": observed_branch(item, branches, qa_id=qa_id),
            }
        )
    return rows


def question_set_deferrals(question_set: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deferred rows of one generated question set, reason code kept per row."""
    rows = []
    for entry in question_set.get("deferred") or []:
        if not isinstance(entry, Mapping):
            continue
        rows.append(
            {
                "qa_id": str(entry.get("qa_id")),
                "code": str(entry.get("code") or "deferred_by_rule"),
                "detail": entry.get("detail"),
            }
        )
    return rows


def source_family_index(registry: Mapping[str, Any]) -> dict[str, str]:
    """asset_id -> human/animal/device, from the registry that owns that fact."""
    index: dict[str, str] = {}
    for record in registry.get("assets") or []:
        if not isinstance(record, Mapping):
            continue
        index[str(record["asset_id"])] = source_family(record)
    return index


def member_source_families(
    facts: Mapping[str, Any], *, family_by_asset: Mapping[str, str]
) -> dict[str, Any]:
    """Resolve the participating source families of one Episode's actors.

    Retained facts carry ``asset_id`` but leave ``entity_class`` null, so the
    family is resolved through the runtime registry that owns that attribute.
    An asset the registry does not know stays unresolved; it is never inferred
    from the identifier text.
    """
    actors = facts.get("actors")
    actors = actors if isinstance(actors, Mapping) else {}
    families: list[str] = []
    unresolved: list[str] = []
    for actor_id in sorted(actors):
        actor = actors.get(actor_id)
        asset_id = str((actor or {}).get("asset_id"))
        family = family_by_asset.get(asset_id)
        if family is None:
            unresolved.append(asset_id)
        else:
            families.append(family)
    # Coverage of the two-entity combinations is every unordered pair actually
    # present, not only the pair of a two-source Episode: a room holding a
    # human, a second human and a device covers human+human and human+device at
    # once. Counting only two-source Episodes hid every device combination in
    # the retained library, where 36 of 148 members carry three or four sources.
    counted = Counter(families)
    combinations = sorted({
        combination_key(first, second)
        for first in counted
        for second in counted
        if first != second or counted[first] >= 2
    }) if not unresolved else []
    combination = None
    if len(families) == 2 and not unresolved:
        combination = combination_key(families[0], families[1])
    elif unresolved:
        combination = UNRESOLVED_SOURCE_FAMILY
    return {
        "source_families": sorted(families),
        "entity_combination": combination,
        "entity_combinations": combinations,
        "asset_ids_absent_from_registry": sorted(set(unresolved)),
    }


def _member_facts_path(member: Mapping[str, Any], *, base: Path) -> Path:
    value = member.get("facts_path")
    if not value:
        raise V1AchievedCoverageError("group member has no facts_path")
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def survey_group_member(
    group: Mapping[str, Any],
    member: Mapping[str, Any],
    *,
    base: Path,
    family_by_asset: Mapping[str, str],
    generate,
    seed: str,
    items_per_type: int = 1,
) -> dict[str, Any]:
    """Survey one retained core-group member from its own facts.

    Only the question generator runs; no media is copied and no world is
    rendered, so this census can cover a whole retained library without
    spending a native budget.
    """
    facts_path = _member_facts_path(member, base=base)
    facts = _read(facts_path)
    sampling = dict(facts.get("sampling") or {})
    sampling["time_display_precision"] = 0
    sampling["qa_sampling"] = {
        **(sampling.get("qa_sampling") or {}),
        "time_display_precision": 0,
    }
    facts["sampling"] = sampling
    identity = member_source_families(facts, family_by_asset=family_by_asset)
    row = {
        "group_id": str(group.get("group_id")),
        "member_id": str(member.get("member_id")),
        "world_id": str(group.get("world_id")),
        "task_family": group.get("task_family"),
        "room_family": group.get("room_family"),
        "room_id": group.get("room_id"),
        "core_sample_id": member.get("sample_id"),
        "facts_path": str(facts_path),
        **identity,
    }
    try:
        questions = generate(facts, seed=seed, items_per_type=int(items_per_type))
    except Exception as error:  # a real generation failure stays one row, not a gap
        row["generation_status"] = "fail"
        row["generation_error"] = f"{type(error).__name__}: {error}"
        row["questions"] = []
        row["deferred"] = []
        return row
    row["generation_status"] = "pass"
    row["questions"] = question_set_rows(questions)
    row["deferred"] = question_set_deferrals(questions)
    return row


def survey_retained_group_library(
    snapshot: str | Path | Mapping[str, Any],
    *,
    registry: str | Path | Mapping[str, Any],
    generate=None,
    seed_prefix: str = "v1-coverage-census",
    items_per_type: int = 1,
    group_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Census every member of a retained group snapshot, one world counted once.

    A snapshot lists groups with their bundle path, world identity, task family
    and room family; each bundle carries the members and their facts. This walks
    that real structure instead of assuming a delivery subset is the library.
    """
    if generate is None:
        from avengine.qa.binding_catalog import whole_degree_display
        from avengine.qa.unified_catalog import QA_IDS, generate_unified_questions

        def generate(facts, *, seed, items_per_type):
            return whole_degree_display(
                generate_unified_questions(
                    facts, qa_ids=QA_IDS, items_per_type=items_per_type, seed=seed
                )
            )

    snapshot_path = None
    if isinstance(snapshot, (str, Path)):
        snapshot_path = Path(snapshot).expanduser().resolve()
        document = _read(snapshot_path)
    elif isinstance(snapshot, Mapping):
        document = deepcopy(dict(snapshot))
    else:
        raise V1AchievedCoverageError("snapshot must be a path or object")
    registry_path = None
    if isinstance(registry, (str, Path)):
        registry_path = Path(registry).expanduser().resolve()
        registry_document = _read(registry_path)
    elif isinstance(registry, Mapping):
        registry_document = registry
    else:
        raise V1AchievedCoverageError("registry must be a path or object")
    family_by_asset = source_family_index(registry_document)
    wanted = None if group_ids is None else {str(value) for value in group_ids}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in document.get("groups") or []:
        if not isinstance(entry, Mapping):
            raise V1AchievedCoverageError("snapshot group entry must be an object")
        if wanted is not None and str(entry.get("group_id")) not in wanted:
            continue
        bundle_path = Path(str(entry["bundle"])).expanduser()
        bundle = _read(bundle_path)
        for group in bundle.get("groups") or []:
            merged = {
                "group_id": group.get("group_id", entry.get("group_id")),
                "world_id": group.get("world_id", entry.get("world_id")),
                "task_family": group.get("task_family", entry.get("task_family")),
                "room_family": group.get("room_family", entry.get("room_family")),
                "room_id": group.get("room_id", entry.get("room_id")),
            }
            for member in group.get("members") or []:
                key = (str(merged["group_id"]), str(member.get("member_id")))
                if key in seen:
                    raise V1AchievedCoverageError(
                        f"snapshot supplies group/member {key} twice"
                    )
                seen.add(key)
                rows.append(
                    survey_group_member(
                        merged,
                        member,
                        base=bundle_path.parent,
                        family_by_asset=family_by_asset,
                        generate=generate,
                        seed=f"{seed_prefix}:{merged['group_id']}:{member.get('member_id')}",
                        items_per_type=items_per_type,
                    )
                )
    return {
        "schema": V1_ACHIEVED_SCHEMA,
        "source_kind": "retained_group_library_census",
        "snapshot": str(snapshot_path) if snapshot_path else "inline",
        "snapshot_status": document.get("status"),
        "registry": str(registry_path) if registry_path else "inline",
        "members": rows,
        "claim_boundary": (
            "Question generation over retained facts only. No new world, no media "
            "readback, no human answerability and no model evaluation."
        ),
    }


def survey_delivery_export(root: str | Path) -> dict[str, Any]:
    """Read achieved rows out of a delivered export instead of regenerating them.

    The public index owns world key, room family and split; the private gold
    index owns the per-sample facts, group identity and core task. Both are read
    as delivered, so an imported delivery contributes the counts it actually
    published.
    """
    root = Path(root).expanduser().resolve()
    public = _read(root / "public" / "dataset_index.json")
    private_path = root / "private" / "gold_index.json"
    private = _read(private_path) if private_path.exists() else {"records": []}
    records = [
        record for record in private.get("records") or []
        if isinstance(record, Mapping)
    ]
    public_samples = list(public.get("samples") or [])
    public_ids = {
        str(sample.get("sample_id")) for sample in public_samples
        if sample.get("sample_id") is not None
    }
    # The public index joins on the private record's own public `sample_id`.
    # `core_sample_id` is a *different* namespace and in a real export the two
    # are permuted: public sample_000001 carries core id sample_000003 while
    # public sample_000003 carries core id sample_000001. Folding both into
    # one map let one record's alias overwrite another record's primary key,
    # so two public samples resolved to the same member and a delivered
    # member disappeared from the survey entirely.
    by_public_id: dict[str, Mapping[str, Any]] = {}
    duplicate_public_ids: list[str] = []
    for record in records:
        key = record.get("sample_id")
        if not isinstance(key, str) or not key.strip():
            continue
        key = key.strip()
        if key in by_public_id:
            duplicate_public_ids.append(key)
            continue
        by_public_id[key] = record
    # An older export wrote only the core id. That alias is a fallback, never
    # a primary key: it is consulted only for a public sample no record
    # claims, only when exactly one record carries it, and only when that
    # record is not already joined through its own public id. Any of those
    # three failing leaves the sample unmatched and says so, because guessing
    # is what produced a wrong member list in the first place.
    alias_counts: dict[str, int] = {}
    by_core_alias: dict[str, Mapping[str, Any]] = {}
    for record in records:
        alias = record.get("core_sample_id")
        if not isinstance(alias, str) or not alias.strip():
            continue
        alias = alias.strip()
        alias_counts[alias] = alias_counts.get(alias, 0) + 1
        by_core_alias.setdefault(alias, record)
    ambiguous_core_aliases = sorted(
        alias for alias, count in alias_counts.items() if count > 1
    )
    rows = []
    world_identity_unknown = False
    unmatched_public_sample_ids: list[str] = []
    join_key_counts: dict[str, int] = {}
    for sample in public_samples:
        sample_id = str(sample.get("sample_id"))
        record = by_public_id.get(sample_id)
        join_key = "public_sample_id"
        if record is None:
            candidate = by_core_alias.get(sample_id)
            own_public_id = (
                None if candidate is None else candidate.get("sample_id")
            )
            already_joined = (
                isinstance(own_public_id, str)
                and own_public_id.strip() in public_ids
            )
            if (
                candidate is not None
                and alias_counts.get(sample_id) == 1
                and not already_joined
            ):
                record = candidate
                join_key = "core_sample_id_alias"
        if record is None:
            record = {}
            join_key = "unmatched"
            unmatched_public_sample_ids.append(sample_id)
        join_key_counts[join_key] = join_key_counts.get(join_key, 0) + 1
        private_world_id = record.get("world_id")
        if not isinstance(private_world_id, str) or not private_world_id.strip():
            world_identity_unknown = True
            private_world_id = None
        questions = []
        for entry in sample.get("questions") or []:
            questions.append(
                {
                    "qa_id": str(entry.get("qa_id")),
                    "question_id": str(entry.get("question_id")),
                    "kind": str(entry.get("kind")),
                    "forms": sorted(str(form) for form in entry.get("forms") or []),
                    "branches_expected": [],
                    "branch": None,
                }
            )
        rows.append(
            {
                "group_id": record.get("group_id"),
                "member_id": record.get("member_id"),
                "world_id": private_world_id,
                "task_family": record.get("core_task") or record.get("task_family"),
                "room_family": sample.get("room_family"),
                "room_id": None,
                "core_sample_id": record.get("core_sample_id") or sample_id,
                "facts_path": record.get("facts_path"),
                "source_families": [],
                "entity_combination": None,
                "asset_ids_absent_from_registry": [],
                "generation_status": "pass",
                "questions": questions,
                "deferred": [],
                "delivered_sample_id": sample_id,
                "private_join_key": join_key,
            }
        )
    return {
        "schema": V1_ACHIEVED_SCHEMA,
        "source_kind": "delivered_export",
        "export_root": str(root),
        "members": rows,
        "world_accounting": (
            "unknown_public_only" if world_identity_unknown else "private_metadata"
        ),
        "delivered_counts": deepcopy(dict(public.get("counts") or {})),
        "private_join": {
            "join_key_counts": dict(sorted(join_key_counts.items())),
            "unmatched_public_sample_ids": unmatched_public_sample_ids,
            "duplicate_public_sample_ids": sorted(set(duplicate_public_ids)),
            "ambiguous_core_aliases": ambiguous_core_aliases,
            "note": (
                "The public sample_id is the join key. core_sample_id is a "
                "separate namespace kept for older exports and is used only "
                "for a public sample no record claims, when it is unambiguous "
                "and its record is not already joined."
            ),
        },
        "claim_boundary": (
            "Counts as delivered. Branch and source family are absent from the "
            "public export, so this view reports them as unavailable rather than "
            "deriving them."
        ),
    }


def _delivered_sample_identity(row: Mapping[str, Any]) -> str | None:
    """Which delivered sample a survey row came from, if it says."""
    for field in ("delivered_sample_id", "core_sample_id", "sample_id"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def merge_achieved_surveys(*surveys: Mapping[str, Any]) -> dict[str, Any]:
    """Union several surveys, counting each core-group member slot once.

    Identity is the core-group member, falling back to the delivered sample
    when a survey has no group identity. Re-importing the same retained
    delivery therefore does not inflate a world or a question count.

    Two different things collide on that key and they are reported apart,
    because they mean opposite things about the data. The same member
    delivered twice is a re-import: nothing new arrived. A *different*
    delivered sample sitting in a member slot that is already filled is an
    extra sample -- an engineering replay of one member, which a real export
    in this programme contains. Both are kept out of the member count, so a
    four-member core group stays four members and never reads as five. Only
    the extra sample is a fact about the delivery rather than about the
    import, so it keeps its own sample id: calling it "already imported"
    named a sample that had never been imported at all and left no way to
    see that the export carried a replay.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    reimported: list[dict[str, Any]] = []
    extra_samples: list[dict[str, Any]] = []
    for survey in surveys:
        if survey.get("schema") != V1_ACHIEVED_SCHEMA:
            raise V1AchievedCoverageError("every merged survey must be an achieved survey")
        for row in survey.get("members") or []:
            group_id = row.get("group_id")
            member_id = row.get("member_id")
            if group_id and member_id:
                key = (str(group_id), str(member_id))
            else:
                key = ("delivered_sample", str(row.get("delivered_sample_id") or row.get("core_sample_id")))
            if key in merged:
                kept = merged[key]
                incoming_sample = _delivered_sample_identity(row)
                kept_sample = _delivered_sample_identity(kept)
                record = {
                    "key": list(key),
                    "kept_source": kept["_source_kind"],
                    "skipped_source": survey.get("source_kind"),
                }
                if (
                    incoming_sample is not None
                    and kept_sample is not None
                    and incoming_sample != kept_sample
                ):
                    # Only the extra sample needs the two ids: for a genuine
                    # re-import they are the same value, and the re-import
                    # record is an existing published shape.
                    record["kept_sample_id"] = kept_sample
                    record["skipped_sample_id"] = incoming_sample
                    record["world_id"] = row.get("world_id")
                    record["reason"] = (
                        "a second delivered sample occupies a member slot that "
                        "is already filled; it is an extra sample for that "
                        "member, not a re-import, and it does not add a member "
                        "or a world"
                    )
                    extra_samples.append(record)
                else:
                    reimported.append(record)
                continue
            merged[key] = {**deepcopy(dict(row)), "_source_kind": survey.get("source_kind")}
    rows = []
    for key in sorted(merged):
        row = merged[key]
        row.pop("_source_kind", None)
        rows.append(row)
    return {
        "schema": V1_ACHIEVED_SCHEMA,
        "source_kind": "merged",
        "merged_source_kinds": [survey.get("source_kind") for survey in surveys],
        "members": rows,
        "already_imported_members_counted_once": reimported,
        "extra_samples_for_a_filled_member_slot": extra_samples,
        "claim_boundary": (
            "Union of real surveys; no member, world or question counted "
            "twice. An extra delivered sample for a member slot is listed "
            "separately and counted in neither the member nor the world total."
        ),
    }


def achieved_coverage_table(survey: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate one survey into the axes the V1 targets are written against.

    One item carrying both mcq and open counts once as a main question; the form
    counts are a breakdown of the same items. Angle follow-ups are listed on
    their own line and never enter the main-question count.
    """
    from avengine.qa.unified_catalog import QA_IDS

    rows = survey.get("members") or []
    per_qa: dict[str, dict[str, Any]] = {}
    branch_rows: dict[tuple[str, str], dict[str, Any]] = {}
    combination_world_ids: dict[str, set[str]] = {}
    failures = []
    world_identity_unknown = False
    for row in rows:
        if row.get("generation_status") != "pass":
            failures.append(
                {
                    "group_id": row.get("group_id"),
                    "member_id": row.get("member_id"),
                    "world_id": row.get("world_id"),
                    "error": row.get("generation_error"),
                }
            )
            continue
        raw_world_id = row.get("world_id")
        world_id = (
            str(raw_world_id)
            if isinstance(raw_world_id, str) and raw_world_id.strip()
            else None
        )
        if world_id is None:
            world_identity_unknown = True
        for question in row.get("questions") or []:
            qa_id = str(question["qa_id"])
            bucket = per_qa.setdefault(
                qa_id,
                {
                    "valid_main_questions": 0,
                    "valid_angle_followups": 0,
                    "worlds": set(),
                    "form_counts": Counter(),
                    "task_families": Counter(),
                    "room_families": Counter(),
                    "entity_combinations": Counter(),
                    "source_families": Counter(),
                    "branch_unobservable_main": 0,
                },
            )
            if question["kind"] == "main":
                bucket["valid_main_questions"] += 1
                if world_id is not None:
                    bucket["worlds"].add(world_id)
                if row.get("task_family"):
                    bucket["task_families"][str(row["task_family"])] += 1
                if row.get("room_family"):
                    bucket["room_families"][str(row["room_family"])] += 1
                for combination in row.get("entity_combinations") or (
                    [row["entity_combination"]] if row.get("entity_combination") else []
                ):
                    bucket["entity_combinations"][str(combination)] += 1
                    # Questions, forms and core members from one world cannot
                    # satisfy a quota for a second independent world.
                    actual_world = row.get("world_id")
                    if isinstance(actual_world, str) and actual_world.strip():
                        combination_world_ids.setdefault(str(combination), set()).add(actual_world)
                for family in sorted(set(row.get("source_families") or [])):
                    bucket["source_families"][str(family)] += 1
                branch = question.get("branch")
                if question.get("branches_expected"):
                    if branch is None:
                        bucket["branch_unobservable_main"] += 1
                    else:
                        entry = branch_rows.setdefault(
                            (qa_id, str(branch)),
                            {"qa_id": qa_id, "branch": str(branch),
                             "valid_main_questions": 0, "worlds": set()},
                        )
                        entry["valid_main_questions"] += 1
                        if world_id is not None:
                            entry["worlds"].add(world_id)
            else:
                bucket["valid_angle_followups"] += 1
            for form in question.get("forms") or []:
                bucket["form_counts"][str(form)] += 1
    by_qa = {}
    for qa_id in QA_IDS:
        bucket = per_qa.get(qa_id)
        if bucket is None:
            by_qa[qa_id] = {
                "valid_main_questions": 0,
                "valid_angle_followups": 0,
                "distinct_worlds_with_main": 0,
                "form_counts": {},
                "task_families": {},
                "room_families": {},
                "entity_combinations": {},
                "source_families": {},
                "branch_unobservable_main": 0,
            }
            continue
        by_qa[qa_id] = {
            "valid_main_questions": bucket["valid_main_questions"],
            "valid_angle_followups": bucket["valid_angle_followups"],
            "distinct_worlds_with_main": (
                None if world_identity_unknown else len(bucket["worlds"])
            ),
            "form_counts": dict(sorted(bucket["form_counts"].items())),
            "task_families": dict(sorted(bucket["task_families"].items())),
            "room_families": dict(sorted(bucket["room_families"].items())),
            "entity_combinations": dict(sorted(bucket["entity_combinations"].items())),
            "source_families": dict(sorted(bucket["source_families"].items())),
            "branch_unobservable_main": bucket["branch_unobservable_main"],
        }
    branches = {
        f"{qa_id}:{branch}": {
            "qa_id": entry["qa_id"],
            "branch": entry["branch"],
            "valid_main_questions": entry["valid_main_questions"],
            "distinct_worlds_with_main": (
                None if world_identity_unknown else len(entry["worlds"])
            ),
        }
        for (qa_id, branch), entry in sorted(branch_rows.items())
    }
    deferred_by_qa: dict[str, Counter] = {}
    for row in rows:
        for entry in row.get("deferred") or []:
            deferred_by_qa.setdefault(str(entry["qa_id"]), Counter())[str(entry["code"])] += 1
    matrix: dict[str, int] = Counter()
    matrix_groups: dict[str, set] = {}
    for row in rows:
        if row.get("task_family") and row.get("room_family"):
            key = f"{row['task_family']}|{row['room_family']}"
            matrix[key] += 1
            matrix_groups.setdefault(key, set()).add(str(row.get("group_id")))
    actor_counts = Counter(len(row.get("source_families") or []) for row in rows)
    return {
        "schema": V1_ACHIEVED_SCHEMA,
        "source_kind": survey.get("source_kind"),
        "member_count": len(rows),
        "group_count": len({str(row.get("group_id")) for row in rows if row.get("group_id")}),
        "world_count": (
            None if world_identity_unknown
            else len({str(row.get("world_id")) for row in rows})
        ),
        "world_identity_status": (
            "unknown" if world_identity_unknown else "known"
        ),
        "generation_failures": failures,
        "by_qa_id": by_qa,
        "by_qa_branch": branches,
        "world_ids_by_entity_combination": {
            key: sorted(values) for key, values in sorted(combination_world_ids.items())
        },
        "core_task_by_room_family_member_counts": dict(sorted(matrix.items())),
        "core_task_by_room_family_group_counts": {
            key: len(values) for key, values in sorted(matrix_groups.items())
        },
        "resolved_source_count_per_member": dict(sorted(actor_counts.items())),
        "deferred_codes_by_qa_id": {
            qa_id: dict(sorted(counter.items()))
            for qa_id, counter in sorted(deferred_by_qa.items())
        },
        "source_families_unresolved": sorted(
            {
                asset_id
                for row in rows
                for asset_id in row.get("asset_ids_absent_from_registry") or []
            }
        ),
        "counting_note": (
            "One generated item with both forms is one valid main question; the "
            "form counts break the same items down. Angle follow-ups are counted "
            "separately. A world contributes once however many members it serves."
        ),
        "model_evaluation": "not_run",
        "human_answerability": "not_run",
    }

