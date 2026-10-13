"""Recorded dry inputs for v7; source admission belongs to the QA admission line.

This adapter preserves those decisions and adds the owner's source-kind rule.
It never generates a replacement waveform when an input is missing.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import re
import time
import numpy as np

AUDIO_SOURCE_REVISION = "v7-public-recorded-dry-natural-tts-20261004-r1"
PROGRAM_SCHEMA = "v7-recorded-dry-program-v1"
FORBIDDEN_DATASETS = ("STARSS23", "DCASE", "TAU", "SAVVY")


def source_blockers(record):
    """Keep existing automatic admission; reject forbidden kinds/provenance."""
    if record.get("source_type") == "author_tts" and record.get("speech_policy"):
        # Use the QA owner's natural-TTS source authority, including SHA-bound
        # blind ASR, licensed VCTK reference and matching registered gender.
        # Native episode/role/mouth proofs remain separate QA gates.
        from avengine.qa.v7_recorded_audio import natural_speech_blockers
        return natural_speech_blockers(record)
    reasons = []
    if record.get("strict_v7_eligible") is not True or record.get("formal_status") not in {"admitted", "admitted_under_listed_screens"}:
        reasons.append("upstream_automatic_admission_missing")
    screens = record.get("screens", {})
    license_screen = screens.get("license", {})
    for field in ("license", "original_id", "source_uri"):
        if not license_screen.get(field):
            reasons.append("missing_" + field)
    for name in ("license", "provenance", "deduplication", "automatic_label"):
        if screens.get(name, {}).get("status") != "pass":
            reasons.append("upstream_" + name + "_not_pass")
    # Evaluation references inside deduplication evidence are NOT source origins.
    reference = record.get("tts_provenance", {}).get("reference_audio", {})
    provenance = " ".join(str(source.get(k, "")) for source in (record, reference) for k in (
        "source_dataset", "dataset", "source_origin", "source_uri", "path", "normalized_path", "original_id"))
    for name in FORBIDDEN_DATASETS:
        if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", provenance, re.I):
            reasons.append("excluded_source_" + name)
    if any(source.get("same_origin_as_evaluation") is True or source.get("real_spatial_recording") is True
           for source in (record, reference)):
        reasons.append("spatial_or_evaluation_same_origin")
    engine = record.get("synthesis_readback", {}).get("engine", {})
    kind = record.get("source_type", "recorded")
    synthesis = " ".join((str(engine), provenance, str(record.get("source_kind", "")),
                           str(record.get("synthesis_method", ""))))
    if re.search(r"espeak|formant|procedural|program_synth", synthesis, re.I):
        reasons.append("robotic_or_procedural_source_forbidden")
    if kind == "author_tts":
        chain = record.get("tts_provenance", {})
        if chain.get("natural_voice") is not True:
            reasons.append("natural_tts_not_verified")
        for component in ("code", "weights"):
            if not all(chain.get(component, {}).get(k) for k in ("source_uri", "revision", "license")):
                reasons.append("missing_actual_tts_" + component + "_license_revision")
        if chain.get("mode") != "unconditioned":
            if not all(chain.get("reference_audio", {}).get(k) for k in ("original_id", "license", "source_uri")):
                reasons.append("missing_tts_reference_audio_rights")
        terms = chain.get("output_terms", {})
        if terms.get("usage_allowed") is not True or not terms.get("source_uri") or not terms.get("assessment"):
            reasons.append("tts_output_terms_not_recorded")
        if not record.get("script_author") or not record.get("transcript") or not chain.get("voice_id"):
            reasons.append("tts_script_or_voice_identity_missing")
    elif kind == "generated_audio":
        # Owner-authorized, train-only SAO top-up remains explicitly generated.
        # Shared license/provenance/dedup/label screens above still apply.
        if record.get("generation", {}).get("model_id") != "stabilityai/stable-audio-open-1.0":
            reasons.append("unsupported_generated_audio_model")
        if not record.get("owner_acceptance") or record.get("allowed_splits") != ["train"] or record.get("split") != "train":
            reasons.append("generated_audio_owner_or_train_only_admission_missing")
        if record.get("automatic_no_speech", {}).get("status") != "pass":
            reasons.append("generated_audio_no_speech_not_pass")
        activity=record.get("automatic_activity_readback", {})
        if activity.get("status") != "pass" or float(activity.get("active_duration_s",0)) < 1.0:
            reasons.append("generated_audio_activity_below_one_second")
        dedup=record.get("reference_dedup", {})
        if dedup.get("status") != "pass" or dedup.get("coverage_complete_for_listed_roots") is not True or dedup.get("positive_controls_pass") is not True:
            reasons.append("generated_audio_complete_reference_dedup_missing")
        labels=record.get("automatic_label") or []
        if not any(x.get("status")=="pass" and x.get("gate",{}).get("status")=="pass"
                   and float(x.get("confidence",0))>=float(x["gate"].get("threshold",float("inf")))
                   and float(x.get("margin",-1))>=float(x["gate"].get("minimum_margin",float("inf"))) for x in labels):
            reasons.append("generated_audio_calibrated_label_gate_not_pass")
    elif kind != "recorded":
        reasons.append("unrecognized_dry_source_kind")
    return list(dict.fromkeys(reasons))


def filter_admission(records):
    admitted, rejected = [], []
    for record in records:
        reasons = source_blockers(record)
        if reasons:
            rejected.append({"sound_asset_id": record.get("sound_asset_id"), "reasons": reasons})
        else:
            admitted.append(deepcopy(record))
    return admitted, rejected


def assemble_recorded_buses(scene, program):
    """Bind registered mono clips to explicit events without speech synthesis.

    Crop offsets use ORIGINAL clip samples. No time stretch, repetition, or
    substitute sound is permitted. Authored TTS keeps its entire utterance;
    at most one 100 ms timeline tick may be padded after its waveform.
    Activity receipts describe crops, and do not certify rendered QA answers.
    """
    import soundfile as sf
    from scipy.signal import resample_poly
    if not isinstance(program, dict) or program.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("registered recorded_dry_program required; eSpeak/procedural fallback is forbidden")
    records = program.get("records", [])
    by_id = {r["sound_asset_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate recorded sound asset ID")
    bindings = {e["event_id"]: e for e in program.get("events", [])}
    if len(bindings) != len(program.get("events", [])) or set(bindings) != {e["event_id"] for e in scene["events"]}:
        raise ValueError("recorded program must bind every scene event exactly once")
    admitted, rejected = filter_admission(records)
    if rejected:
        raise ValueError("recorded source not admitted: " + str(rejected))
    sr = int(scene.get("sample_rate_hz", 16000))
    if sr != 16000:
        raise ValueError("recorded v7 program requires the existing 16 kHz contract")
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    buses = {k: np.zeros(round(scene["duration_s"] * sr)) for k in scene["sources"]}
    cache, evidence, speakers = {}, [], {}
    for event in scene["events"]:
        binding = bindings[event["event_id"]]
        record = by_id[binding["sound_asset_id"]]
        source = scene["sources"][event["endpoint_id"]]
        if source.get("content_kind") == "speech":
            if record.get("sound_class") not in ("speech_playback", "human_speech_male", "human_speech_female"):
                raise ValueError("a speaking actor requires a speech dry clip, not an appliance/animal sound")
            gender = record.get("gender") or record.get("speaker_gender") or record.get("voice_gender")
            if source.get("class_id") in (0, 1):
                required = "F" if source["class_id"] == 0 else "M"
                expected = "human_speech_" + ("female" if required == "F" else "male")
                if (gender in ("M", "F") and gender != required) or (gender not in ("M", "F") and record.get("sound_class") != expected):
                    raise ValueError("legacy speech class requires verified matching voice metadata; no label override")
        asset = record["sound_asset_id"]
        if asset not in cache:
            path = Path(record.get("normalized_path") or record["path"])
            samples, native_sr = sf.read(path, dtype="float64", always_2d=True)
            if samples.shape[1] != 1 or not np.isfinite(samples).all() or not len(samples):
                raise ValueError("recorded source must be finite nonempty mono: " + str(path))
            cache[asset] = samples[:, 0], native_sr, str(path.resolve())
        samples, native_sr, path = cache[asset]
        crop = binding.get("crop_start_sample", 0)
        if isinstance(crop, bool) or not isinstance(crop, int) or crop < 0 or crop >= len(samples):
            raise ValueError("invalid original-clip crop offset")
        endpoint = event["endpoint_id"]
        lo, hi = event["start_tick"] * 1600, event["end_tick"] * 1600
        if endpoint not in buses or not 0 <= lo < hi <= len(buses[endpoint]):
            raise ValueError("recorded event is outside its source timeline")
        length = hi - lo
        divisor = math.gcd(sr, int(native_sr))
        clip = resample_poly(samples[crop:], sr // divisor, int(native_sr) // divisor)
        authored = record.get("source_type") == "author_tts"
        whole_recording = binding.get("preserve_full_recorded_clip") is True
        if whole_recording and (authored or record.get("source_type", "recorded") != "recorded"
                or not record.get("speaker_id") or not record.get("transcript")
                or crop or length < len(clip) or length - len(clip) > 1600):
            raise ValueError("whole recorded utterance must preserve every sample, with at most one tick of trailing silence")
        if authored and (crop or length < len(clip) or length - len(clip) > 1600):
            raise ValueError("authored natural TTS cannot be trimmed or repeated")
        if not authored and not whole_recording and len(clip) < length:
            raise ValueError("recorded clip is too short; looping/synthesis fallback is forbidden")
        take = min(length, len(clip))
        segment = np.zeros(length); segment[:take] = clip[:take]
        if not authored and not whole_recording:
            fade = min(80, take // 2)
            if fade:
                ramp = np.linspace(0, 1, fade)
                segment[:fade] *= ramp; segment[take-fade:take] *= ramp[::-1]
        buses[endpoint][lo:hi] += segment * float(scene["sources"][endpoint].get("dry_gain", 1.))
        speaker = record.get("speaker_id") or record.get("tts_provenance", {}).get("voice_id")
        if not speaker:
            match = re.search(r"vctk_(p\d+)_", str(record.get("source_origin", "")))
            speaker = match.group(1) if match else None
        if speaker:
            if endpoint in speakers and speakers[endpoint] != speaker:
                raise ValueError("one speaking actor cannot silently change recorded speaker identity")
            speakers[endpoint] = speaker
        license_screen = record["screens"]["license"]
        evidence.append({"event_id": event["event_id"], "endpoint_id": endpoint,
                         "sound_asset_id": asset, "original_id": license_screen["original_id"],
                         "license": license_screen["license"], "source_uri": license_screen["source_uri"],
                         "input_path": path, "source_sample_rate_hz": native_sr,
                         "crop_start_sample": crop, "output_start_sample": lo,
                         "output_end_sample_exclusive": hi, "waveform_sample_count": take,
                         "semantic_content_complete": authored or whole_recording,
                         "preserved_full_recorded_clip": whole_recording,
                         "trailing_padding_samples": length - take, "repeat_or_time_stretch": False})
    return buses, {"audio_source_revision": AUDIO_SOURCE_REVISION,
                   "status": "recorded_inputs_assembled", "admitted_records": admitted,
                   "events": evidence, "endpoint_speaker_ids": speakers,
                   "wall_seconds": time.perf_counter()-start_wall,
                   "process_cpu_seconds": time.process_time()-start_cpu,
                   "rendered_QA_semantics_certified": False}


def validate_dry_receipt(receipt):
    if not isinstance(receipt, dict) or receipt.get("audio_source_revision") != AUDIO_SOURCE_REVISION or receipt.get("status") != "recorded_inputs_assembled":
        raise ValueError("recorded dry receipt required; legacy eSpeak/procedural audio is not a production input")
    _, rejected = filter_admission(receipt.get("admitted_records", []))
    if rejected or not isinstance(receipt.get("events"), list) or (receipt["events"] and not receipt.get("admitted_records")):
        raise ValueError("recorded dry receipt has no admitted event sources")


def rebind_receipt(receipt, mapping):
    result = deepcopy(receipt)
    for event in result["events"]:
        event["endpoint_id"] = mapping.get(event["endpoint_id"], event["endpoint_id"])
    result["endpoint_speaker_ids"] = {mapping.get(k, k): v for k, v in result["endpoint_speaker_ids"].items()}
    result["counterfactual_endpoint_mapping"] = dict(mapping)
    return result
