"""Owner-authorized Stable Audio Open sources with train-only machine receipts."""
from __future__ import annotations
import math
MODEL_ID = "stabilityai/stable-audio-open-1.0"
PROVENANCE = "generated_stable_audio_open_1.0"
LICENSE = "Stability AI Community License"
OWNER_ACCEPTANCE = "owner 2026-10-12 接受 IV(b) 风险"

def generated_source_blockers(record, authorization):
    """Validate the explicit exception; ordinary recorded-source policy is unchanged."""
    reasons=[]
    if (not isinstance(authorization,dict)
            or authorization.get("owner_acceptance") != OWNER_ACCEPTANCE
            or authorization.get("allowed_splits") != ["train"]
            or record.get("sound_class") not in authorization.get("classes",[])):
        reasons.append("generated_source_owner_authorization_missing")
    if (record.get("label_provenance") != PROVENANCE
            or record.get("source_dataset") != "Stable Audio Open 1.0"
            or record.get("source_type") != "generated_audio"
            or record.get("source_license",record.get("license")) != LICENSE
            or record.get("owner_acceptance") != OWNER_ACCEPTANCE):
        reasons.append("generated_source_model_license_or_owner_receipt_missing")
    if record.get("allowed_splits") != ["train"] or record.get("split") != "train":
        reasons.append("generated_source_must_be_train_only")
    g=record.get("generation") or {}
    if (g.get("model_id") != MODEL_ID or not g.get("model_sha256")
            or not g.get("model_config_sha256") or not g.get("prompt")
            or isinstance(g.get("seed"),bool) or not isinstance(g.get("seed"),int)
            or not 3 <= g.get("duration_s",0) <= 10
            or g.get("steps") != 100 or g.get("cfg_scale") != 7):
        reasons.append("generated_source_generation_receipt_missing")
    digest=record.get("sha256")
    asr=record.get("automatic_no_speech") or {}
    if (asr.get("status") != "pass" or asr.get("model_id") != "openai/whisper-small"
            or not digest or asr.get("audio_sha256") != digest
            or asr.get("language_forced") is not False
            or asr.get("reference_text_prompted") is not False
            or asr.get("human_listening_used") is not False
            or any(ch.isalpha() or ch.isdigit() for ch in asr.get("blind_decoded_text",""))):
        reasons.append("generated_source_whisper_lexical_content_or_missing_receipt")
    if asr.get("standard_no_speech_skipped") is True:
        ns=asr.get("no_speech_prob");lp=asr.get("avg_logprob")
        if (asr.get("no_speech_threshold") != .6 or asr.get("logprob_threshold") != -1.
                or not isinstance(ns,(int,float)) or not math.isfinite(ns) or not .6 < ns <= 1.
                or not isinstance(lp,(int,float)) or not math.isfinite(lp) or not lp < -1.):
            reasons.append("generated_source_standard_no_speech_probability_receipt_invalid")
    a=record.get("automatic_activity_readback") or {}
    if (a.get("status") != "pass" or a.get("audio_sha256") != digest
            or not isinstance(a.get("active_duration_s"),(int,float))
            or not math.isfinite(a.get("active_duration_s",float("nan")))
            or a.get("active_duration_s",0) < 1
            or not isinstance(a.get("longest_group_s"),(int,float))
            or not math.isfinite(a.get("longest_group_s",float("nan")))
            or a.get("longest_group_s",0) < 1):
        reasons.append("generated_source_effective_activity_below_one_second_or_missing")
    return reasons
