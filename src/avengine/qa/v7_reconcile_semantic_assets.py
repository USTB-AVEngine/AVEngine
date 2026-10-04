"""Recompare retained blind-ASR hypotheses without synthesizing or rewriting audio.

This explicitly separates the initially executed gate from subsequent
value-preserving transcript normalization. Failed raw records remain immutable.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf

from avengine.qa.v7_batch_semantic_assets import digest, save_json
from avengine.qa.v7_recorded_audio import NATURAL_SPEECH_POLICY, semantic_asr_comparison


def _sound(task, attempt, proof, plan, original_sound):
    path = Path(attempt["path"])
    info = sf.info(path)
    if info.channels != 1 or info.samplerate != 16000 or info.frames < 1:
        raise ValueError("retained natural waveform must be complete mono 16k")
    if original_sound is not None and original_sound["path"] == str(path):
        sound = deepcopy(original_sound)
        if sound["synthesis_readback"]["sample_count"] != info.frames:
            raise ValueError("passing raw source count differs from retained waveform")
    else:
        receipt = plan["model_receipt"]
        reference = task["voice_reference"]
        sound = {**deepcopy(task), "script_author": "project_author",
            "speech_policy": NATURAL_SPEECH_POLICY,
            "voice_identity": "vctk:" + reference["speaker_id"],
            "voice_preset": "vctk:" + reference["speaker_id"],
            "emission_mechanism": "human_live_natural_tts", "content_class": "speech",
            "is_program_synthesized": True, "source_real_waveform": False,
            "source_dataset": "project_authored_CosyVoice2", "source_uri": receipt["code_source"],
            "path": str(path), "sha256": digest(path),
            "synthesis_config": dict(task["synthesis_config"], seed=attempt["seed"],
                model_license=receipt["model_license"], actual_waveform=str(path),
                zero_shot_reference_cache_in_memory=True),
            "synthesis_readback": {
                "path": str(path), "sample_rate_hz": info.samplerate, "sample_count": info.frames,
                "source_sample_rate_hz": None, "source_sample_count": None,
                "configured_source_sample_rate_hz": 24000,
                "source_sample_count_provenance": "not_recorded_in_initial_failed_attempt",
                "duration_s": info.duration, "transcript": task["transcript"],
                "content_complete": True, "trimmed_or_repeated": False, "device": "cpu",
                "resampling_method": "scipy.signal.resample_poly",
                "engine": {"name": "CosyVoice2", "natural_neural_tts": True,
                    "model_id": receipt["model_id"], "model_revision": receipt["model_revision"],
                    "model_license": receipt["model_license"], "code_revision": receipt["code_revision"],
                    "model_license_path": receipt["model_license_path"]}},
            "qualification": "natural_TTS_content_pass_registered_stand_plan_pending_native",
            "formal_blockers": ["native_registered_standing_actor_and_complete_placement_missing"]}
    sound["automatic_content_readback"] = proof
    sound["normalization_reconciliation"] = {
        "method": "recompare_existing_blind_ASR_no_new_audio_no_new_ASR",
        "selected_attempt": attempt["attempt"], "recorded_24k_sample_count_available":
            sound["synthesis_readback"].get("source_sample_count") is not None}
    return sound


def reconcile_batch(plan_path, raw_output, output, *, require_complete=False):
    plan_path, raw_output, output = Path(plan_path), Path(raw_output), Path(output)
    if output.resolve().is_relative_to(raw_output.resolve()):
        raise ValueError("reconciliation must be outside the immutable raw batch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "records").mkdir(exist_ok=True)
    plan = json.loads(plan_path.read_text())
    plan_sha = digest(plan_path)
    sounds, rows = [], []
    reasons = Counter()
    raw_reasons = Counter()
    transitions = Counter()
    raw_status = Counter()
    attempts_total = 0
    normalization_passed_attempts = 0
    rescored_attempts = 0
    raw_passing_attempts = 0
    terminal_rows = []
    for task in plan["tasks"]:
        raw_path = raw_output / "records" / (task["sound_asset_id"] + ".json")
        if not raw_path.is_file():
            continue
        raw = json.loads(raw_path.read_text())
        if raw.get("plan_sha256") != plan_sha or raw.get("sound_asset_id") != task["sound_asset_id"]:
            raise ValueError("raw task/plan receipt mismatch")
        if not raw.get("terminal"):
            continue
        terminal_rows.append(raw)
        raw_status[raw["status"]] += 1
        evaluated = []
        selected = None
        for attempt in raw["attempts"]:
            attempts_total += 1
            raw_passing_attempts += attempt["status"] == "pass"
            if attempt.get("reason"):
                raw_reasons[attempt["reason"]] += 1
            result = deepcopy(attempt)
            old_proof = attempt.get("ASR")
            if old_proof is None:
                result["reconciled_status"] = "blocked"
                result["reconciled_reason"] = "original_synthesis_or_ASR_exception"
                reasons[result["reconciled_reason"]] += 1
            else:
                path = Path(attempt["path"])
                if (not path.resolve().is_relative_to((raw_output / "audio").resolve())
                        or not path.is_file() or digest(path) != old_proof["audio_sha256"]
                        or old_proof.get("reference_text") != task["transcript"]
                        or old_proof.get("reference_text_or_answer_prompted_into_ASR") is not False):
                    raise ValueError("retained waveform/blind-ASR/author receipt mismatch")
                new = semantic_asr_comparison(task["transcript"], old_proof["blind_decoded_text"],
                                        task["semantic_answer"]["critical_semantic_slots"])
                proof = {**deepcopy(old_proof), **new,
                    "source": str(output / "records" / (task["sound_asset_id"] + ".json")),
                    "raw_comparison": deepcopy(old_proof),
                    "reconciliation": "fixed_token_spelling_and_clock_value_equivalence_no_fuzzy_matching"}
                result["reconciled_ASR"] = proof
                result["reconciled_status"] = new["status"]
                rescored_attempts += 1
                if new["status"] == "pass":
                    normalization_passed_attempts += 1
                    result["reconciled_reason"] = None
                    if selected is None:
                        selected = (attempt, proof)
                else:
                    failed = []
                    if new.get("wer", 1.0) > 0.1:
                        failed.append("WER_above_0.10")
                    if not new.get("all_critical_slots_correct", False):
                        failed.append("critical_slot_missing_or_changed")
                    if new.get("negation_counts_match") is False:
                        failed.append("negation_polarity_changed")
                    result["reconciled_reason"] = "+".join(failed) or "automatic_meaning_failed"
                    reasons[result["reconciled_reason"]] += 1
                transitions[attempt["status"] + "->" + new["status"]] += 1
            evaluated.append(result)
        if selected:
            sound = _sound(task, selected[0], selected[1], plan, raw.get("sound"))
            sounds.append(sound)
        else:
            sound = None
        row = {"sound_asset_id": task["sound_asset_id"], "scenario_id": task["scenario_id"],
            "role_id": task["role_id"], "raw_record": str(raw_path), "raw_record_sha256": digest(raw_path),
            "raw_status": raw["status"], "status": "pass" if selected else "dropped", "terminal": True,
            "attempts_performed": len(raw["attempts"]), "selected_attempt": selected[0]["attempt"] if selected else None,
            "attempts": evaluated, "sound": sound}
        save_json(output / "records" / (task["sound_asset_id"] + ".json"), row)
        rows.append(row)
    counts = []
    for scenario in plan["dialogues"]["scenarios"]:
        matching = [row for row in rows if row["scenario_id"] == scenario["id"]]
        good = [row for row in matching if row["status"] == "pass"]
        counts.append({"scenario_id": scenario["id"], "planned_lines": len(scenario["roles"]),
            "processed_lines": len(matching), "usable_lines": len(good),
            "dropped_lines": sum(row["status"] == "dropped" for row in matching),
            "all_planned_lines_passed": len(good) == len(scenario["roles"]),
            "at_least_two_usable_roles": len(good) >= 2,
            "usable_role_ids": [row["role_id"] for row in good],
            "actual_reference_speakers": [row["sound"]["voice_reference"]["speaker_id"] for row in good],
            "durations_s": [row["sound"]["synthesis_readback"]["duration_s"] for row in good],
            "actual_native_usable_questions": 0})
    summary = {
        "planned_scenarios": len(counts), "planned_lines": len(plan["tasks"]),
        "processed_terminal_lines": len(terminal_rows), "complete": len(rows) == len(plan["tasks"]),
        "passed_lines": len(sounds), "dropped_lines": sum(row["status"] == "dropped" for row in rows),
        "line_pass_rate": len(sounds) / max(1, len(plan["tasks"])),
        "raw_passed_lines": raw_status["pass"], "raw_dropped_lines": raw_status["dropped"],
        "rescued_initially_dropped_lines": sum(row["raw_status"] == "dropped" and row["status"] == "pass" for row in rows),
        "newly_rejected_initially_passing_lines": sum(row["raw_status"] == "pass" and row["status"] == "dropped" for row in rows),
        "total_actual_synthesis_attempts": attempts_total, "raw_passing_attempts": raw_passing_attempts,
        "rescored_ASR_attempts": rescored_attempts, "normalized_passing_attempts": normalization_passed_attempts,
        "attempt_status_transitions": dict(transitions), "raw_attempt_failure_reason_counts": dict(raw_reasons),
        "normalized_attempt_failure_reason_counts": dict(reasons),
        "attempt_count_distribution": dict(Counter(str(row["attempts_performed"]) for row in rows)),
        "selected_attempt_distribution": dict(Counter(str(row["selected_attempt"]) for row in rows if row["status"] == "pass")),
        "complete_scenarios": sum(row["all_planned_lines_passed"] for row in counts),
        "scenarios_with_at_least_two_usable_roles": sum(row["at_least_two_usable_roles"] for row in counts),
        "usable_lines_distribution": dict(Counter(str(row["usable_lines"]) for row in counts)),
        "actual_reference_speaker_ids": sorted({s["voice_reference"]["speaker_id"] for s in sounds}),
        "per_scenario_distinct_reference_verified": all(len(r["actual_reference_speakers"]) == len(set(r["actual_reference_speakers"])) for r in counts),
        "metadata_voice_visual_gender_matches": all(s["voice_reference"]["gender"] == s["standing_actor_plan"]["visual_gender"] for s in sounds),
        "native_dialogue_questions": 0, "human_listening": False, "new_synthesis_or_ASR_runs_during_reconciliation": 0,
        "maximum_attempts_including_initial": plan["max_attempts"],
        "plan_sha256": plan_sha, "source_registry": plan["source_registry"],
        "source_registry_sha256": plan["source_registry_sha256"],
        "normalizer_source_sha256": digest(Path(__file__).with_name("v7_recorded_audio.py")),
        "raw_output": str(raw_output)}
    if require_complete and not summary["complete"]:
        raise ValueError("cannot finalize: batch still has missing/nonterminal lines")
    save_json(output / "semantic_manifest.json", {"sounds": sounds, "dialogues": plan["dialogues"],
        "script_source": plan["script_source"], "script_source_sha256": plan["script_source_sha256"],
        "source_registry": plan["source_registry"], "source_registry_sha256": plan["source_registry_sha256"],
        "batch_summary": summary, "formal_usable": 0})
    save_json(output / "synthesis_statistics.json", {"summary": summary, "scenarios": counts})
    fields = ["scenario_id", "planned_lines", "processed_lines", "usable_lines", "dropped_lines",
              "all_planned_lines_passed", "at_least_two_usable_roles"]
    with (output / "scenario_counts.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in counts)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reconcile_batch(args.plan, args.raw_output, args.output,
                                   require_complete=args.require_complete), ensure_ascii=False))


if __name__ == "__main__":
    main()
