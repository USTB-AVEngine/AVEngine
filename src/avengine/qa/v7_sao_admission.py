"""Owner-authorized Stable Audio Open sources with train-only machine receipts."""
from __future__ import annotations
import math
MODEL_ID = "stabilityai/stable-audio-open-1.0"
PROVENANCE = "generated_stable_audio_open_1.0"
LICENSE = "Stability AI Community License"
OWNER_ACCEPTANCE = "owner 2026-10-12 接受 IV(b) 风险"
HUMAN_THRESHOLD_POLICY = "owner_fixed_0.38_20261013"

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
    human=record.get("human_speech_screen") or {}
    calibration=human.get("calibration") or {}
    score=human.get("maximum_score"); threshold=human.get("threshold")
    fixed_policy = (calibration.get("threshold_policy") == HUMAN_THRESHOLD_POLICY
                    and threshold == .38
                    and calibration.get("positive_count") == 183
                    and calibration.get("positive_pass") == 178
                    and calibration.get("positive_pass_rate") == 178 / 183
                    and calibration.get("negative_count") == 739
                    and calibration.get("negative_block_rate") == 1.
                    and len(human.get("heads") or []) == 20)
    positive_retention_valid = calibration.get("positive_pass_rate") == 1. or fixed_policy
    if calibration.get("status") not in (None,"pass"):
        reasons.append("generated_source_human_speech_calibration_failed")
    elif (human.get("status") != "pass" or human.get("model_id") != "BEATs_AS2M_cpt2"
            or not digest or human.get("audio_sha256") != digest
            or not isinstance(score,(int,float)) or not math.isfinite(score)
            or not isinstance(threshold,(int,float)) or not math.isfinite(threshold)
            or not 0 <= score <= threshold < 1
            or not human.get("heads") or not human.get("calibration_sha256")
            or calibration.get("status") != "pass"
            or not positive_retention_valid
            or calibration.get("negative_count",0) < 100
            or calibration.get("negative_block_rate",0) < .95):
        reasons.append("generated_source_human_speech_or_missing_calibrated_receipt")
    # Whisper transcription is recording-only, including hallucinated words.
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
