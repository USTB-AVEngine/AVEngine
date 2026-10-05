"""Prepare level-normalized, blind-ASR-checked natural speech sources.

This never synthesizes new speech or changes utterance timing. It consumes the
bounded synthesis receipts and retains separate final waveform/ASR receipts.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import time
import types

import soundfile as sf
from avengine.qa.v7_batch_semantic_assets import digest, save_json
from avengine.qa.v7_extended_admission import normalize_recorded
from avengine.qa.v7_recorded_audio import semantic_asr_comparison


def measure_loudness_one_thread(path):
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-threads", "1",
        "-filter_threads", "1", "-i", str(path),
        "-af", "loudnorm=I=-23:TP=-3:LRA=7:print_format=json", "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    match = re.search(r'\{\s*"input_i".*?\}', result.stderr, re.S)
    if not match:
        raise ValueError("no automatic loudness receipt")
    values = {}
    for key, raw in json.loads(match.group()).items():
        try:
            value = float(raw)
        except (ValueError, TypeError):
            value = None
        values[key] = value if value is not None and math.isfinite(value) else None
    return values


def worker(manifest_path, output, index, workers, asr_path, threads):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    manifest = json.loads(Path(manifest_path).read_text()); output = Path(output)
    log = output / "logs" / f"worker_{index:02d}.log"
    with log.open("a", buffering=1) as handle:
        import sys
        sys.stdout = handle; sys.stderr = handle
        save_json(output / "jobs" / f"worker_{index:02d}.json", {
            "pid": os.getpid(), "parent_pid": os.getppid(), "worker_index": index,
            "process_start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
            "manifest": str(manifest_path), "log": str(log), "device": "cpu"})
        processor = WhisperProcessor.from_pretrained(asr_path, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(asr_path, local_files_only=True).eval().to("cpu")
        model.generation_config.forced_decoder_ids = None
        # Execute the unchanged constant-gain normalizer with a bounded-thread
        # FFmpeg readback. Its activity, gain and peak criteria are unchanged.
        namespace = dict(normalize_recorded.__globals__)
        namespace["_loudness"] = measure_loudness_one_thread
        normalizer = types.FunctionType(normalize_recorded.__code__, namespace)
        passed, failed = 0, 0
        for original in manifest["sounds"][index::workers]:
            sid = original["sound_asset_id"]
            record_path = output / "records" / (sid + ".json")
            started = time.time()
            try:
                if digest(original["path"]) != original["sha256"]:
                    raise ValueError("source waveform receipt changed before normalization")
                normalization = normalizer({"path": original["path"], "sound_asset_id": sid,
                                            "sound_class": "authored_natural_speech"}, output / "audio")
                if normalization["status"] != "pass":
                    raise ValueError("automatic constant-gain/source-activity/true-peak gate failed")
                waveform, rate = sf.read(normalization["path"], dtype="float32")
                features = processor(waveform, sampling_rate=rate, return_tensors="pt", return_attention_mask=True)
                with torch.inference_mode():
                    tokens = model.generate(features.input_features, attention_mask=features.attention_mask,
                        language="en", task="transcribe", do_sample=False, max_new_tokens=256)
                hypothesis = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
                comparison = semantic_asr_comparison(original["transcript"], hypothesis,
                            original["semantic_answer"]["critical_semantic_slots"])
                proof = {**comparison, "model_id": "openai/whisper-small", "model_path": asr_path,
                    "source": str(record_path), "audio_sha256": normalization["sha256"],
                    "reference_text": original["transcript"], "blind_decoded_text": hypothesis,
                    "human_listening_used": False, "reference_text_or_answer_prompted_into_ASR": False,
                    "device": "cpu", "language": "en", "checked_after_constant_gain_normalization": True}
                good = proof["status"] == "pass"
                sound = deepcopy(original)
                sound["original_synthesis_readback"] = deepcopy(original["synthesis_readback"])
                sound["pre_normalization_ASR"] = deepcopy(original["automatic_content_readback"])
                sound["path"] = normalization["path"]; sound["sha256"] = normalization["sha256"]
                sound["source_normalization"] = normalization
                sound["automatic_content_readback"] = proof
                sound["synthesis_config"]["actual_waveform"] = normalization["path"]
                sound["synthesis_config"]["raw_complete_synthesis_waveform"] = original["path"]
                sound["synthesis_readback"].update(
                    path=normalization["path"], sample_count=normalization["sample_count"],
                    sample_rate_hz=normalization["sample_rate_hz"], duration_s=normalization["duration_s"])
                save_json(record_path, {"sound_asset_id": sid, "scenario_id": sound["scenario_id"],
                    "role_id": sound["role_id"], "status": "pass" if good else "dropped", "terminal": True,
                    "reason": None if good else "final_level_normalized_blind_ASR_failed",
                    "normalization": normalization, "ASR": proof, "sound": sound if good else None,
                    "seconds": time.time() - started})
                passed += good; failed += not good
                print("RESULT", sid, proof["status"], proof["wer"], hypothesis, flush=True)
            except Exception as error:
                failed += 1
                save_json(record_path, {"sound_asset_id": sid, "scenario_id": original["scenario_id"],
                    "role_id": original["role_id"], "status": "dropped", "terminal": True,
                    "reason": str(error), "error_type": type(error).__name__, "seconds": time.time() - started})
                print("ERROR", sid, type(error).__name__, str(error), flush=True)
        return {"worker": index, "passed": passed, "dropped": failed}


def assemble(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    previous = json.loads(manifest_path.read_text())
    rows = [json.loads((output / "records" / (s["sound_asset_id"] + ".json")).read_text())
            for s in previous["sounds"]]
    sounds = [r["sound"] for r in rows if r["status"] == "pass"]
    scenarios = []
    for scenario in previous["dialogues"]["scenarios"]:
        good = [s for s in sounds if s["scenario_id"] == scenario["id"]]
        scenarios.append({"scenario_id": scenario["id"], "planned_lines": len(scenario["roles"]),
            "processed_lines": len(scenario["roles"]), "usable_lines": len(good),
            "dropped_lines": len(scenario["roles"]) - len(good),
            "all_planned_lines_passed": len(good) == len(scenario["roles"]),
            "at_least_two_usable_roles": len(good) >= 2,
            "usable_role_ids": [s["role_id"] for s in good],
            "actual_reference_speakers": [s["voice_reference"]["speaker_id"] for s in good],
            "durations_s": [s["synthesis_readback"]["duration_s"] for s in good],
            "actual_native_usable_questions": 0})
    before = previous["batch_summary"]
    summary = {**deepcopy(before),
        "pre_level_normalization_passed_lines": len(previous["sounds"]),
        "passed_lines": len(sounds), "dropped_lines": before["planned_lines"] - len(sounds),
        "line_pass_rate": len(sounds) / before["planned_lines"],
        "final_normalized_ASR_checked_lines": sum("ASR" in r for r in rows),
        "final_normalized_ASR_passed_lines": len(sounds),
        "level_normalization_or_final_ASR_dropped_lines": sum(r["status"] != "pass" for r in rows),
        "additional_synthesis_attempts": 0,
        "final_waveforms_true_peak_pass": sum(s["source_normalization"]["readback_loudness"]["input_tp"] <= -3 for s in sounds),
        "final_waveforms_common_lufs_achieved": sum(s["source_normalization"]["common_lufs_target_achieved"] for s in sounds),
        "final_waveforms_peak_limited_below_lufs_target": sum(s["source_normalization"]["true_peak_ceiling_limited"] for s in sounds),
        "complete_scenarios": sum(r["all_planned_lines_passed"] for r in scenarios),
        "scenarios_with_at_least_two_usable_roles": sum(r["at_least_two_usable_roles"] for r in scenarios),
        "usable_lines_distribution": dict(Counter(str(r["usable_lines"]) for r in scenarios)),
        "final_source_manifest": str(manifest_path), "final_source_manifest_sha256": digest(manifest_path),
        "failure_after_level_normalization": dict(Counter(r["reason"] for r in rows if r["status"] != "pass")),
        "native_dialogue_questions": 0, "human_listening": False, "complete": True}
    save_json(output / "semantic_manifest.json", {**previous, "sounds": sounds, "batch_summary": summary, "formal_usable": 0})
    save_json(output / "synthesis_statistics.json", {"summary": summary, "scenarios": scenarios})
    fields = ["scenario_id", "planned_lines", "processed_lines", "usable_lines", "dropped_lines",
              "all_planned_lines_passed", "at_least_two_usable_roles"]
    with (output / "scenario_counts.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in scenarios)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asr-model", default="/data/models/whisper-small")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "jobs", "records", "audio"):
        (output / name).mkdir()
    save_json(output / "jobs" / "parent.json", {"pid": os.getpid(),
        "process_start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        "manifest": args.manifest, "device": "cpu", "started_epoch": time.time()})
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(worker, args.manifest, str(output), i, args.workers,
                                   args.asr_model, args.cpu_threads) for i in range(args.workers)]
        for future in as_completed(futures):
            print("WORKER_DONE", json.dumps(future.result()), flush=True)
    print("DONE", json.dumps(assemble(args.manifest, output), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
