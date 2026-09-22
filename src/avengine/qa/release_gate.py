"""Decide whether a generated bank meets the declared benchmark standard.

The prior audit reports what a bank looks like; it deliberately changes nothing and
rejects nothing. This module is the other half: a declared policy, and a pass or block
decision against it, so a production run holds the same standard without anyone reading
the receipt by hand.

Every rule names the measured value beside the limit, so a blocked run says what to fix.
Nothing here edits questions, answers, media or scores.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

#: A horizontal half-angle of 42.5 degrees is the 1280 x 720 pinhole the retained rooms
#: are captured with (fx = 698.44, so atan(640 / 698.44) = 42.5). A source outside it is
#: off screen: the video cannot show it and mono audio carries no direction, so only
#: spatial audio can place it. That share is what decides whether a benchmark measures
#: spatial hearing or ordinary audio-visual semantics.
DEFAULT_CAMERA_HALF_FOV_DEG = 42.5

#: Measured on the 2026-09-22 bank, for reference when reading a blocked run:
#: blind constant answer 0.601 on valid, off-screen share 0.19, binary MCQ share 0.57.
DEFAULT_POLICY: dict[str, Any] = {
    "schema": "avengine_qa_release_policy_v1",
    "blind_baseline": {
        # What a model that perceives nothing already scores, fitted on train and read on
        # the held-out splits. Every reported model score sits above this floor, so the
        # floor is what decides how much of a headline number is real.
        "max_train_constant_score": {"valid": 0.35, "test": 0.35},
    },
    "answers": {
        "max_majority_share": 0.45,
        "max_position_deviation": 0.10,
    },
    "options": {
        "min_mcq_options": 4,
        # A two-way multiple choice carries a 50% guessing baseline and tells us nothing the
        # open form does not, so it is capped rather than forbidden: some answer domains
        # really are binary.
        "max_binary_mcq_share": 0.25,
    },
    "support": {
        "min_valid_questions_per_type": 24,
        "min_worlds": {"valid": 60, "test": 60},
    },
    "spatial": {
        # A benchmark meant to measure spatial hearing has to ask about sources the camera
        # cannot show. Below this share, a model with video and mono audio can reach the
        # same score as one with spatial audio, which is what was measured on 2026-09-22.
        "min_off_screen_share": {"valid": 0.60, "test": 0.60},
        "camera_half_fov_deg": DEFAULT_CAMERA_HALF_FOV_DEG,
    },
}

_SPLITS = ("train", "valid", "test")


def _evidence(row: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = row.get("evidence")
    if isinstance(evidence, Mapping):
        return evidence
    truth = row.get("truth")
    if isinstance(truth, Mapping) and isinstance(truth.get("evidence"), Mapping):
        return truth["evidence"]
    return {}


def _azimuth(row: Mapping[str, Any]) -> float | None:
    def walk(node: Any) -> float | None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "azimuth_deg" and isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for item in list(node)[:8]:
                found = walk(item)
                if found is not None:
                    return found
        return None

    value = walk(_evidence(row))
    return value if value is not None and math.isfinite(value) else None


def measure_spatial_support(
    answer_rows: Sequence[Mapping[str, Any]],
    *,
    splits: Mapping[str, str] | None = None,
    camera_half_fov_deg: float = DEFAULT_CAMERA_HALF_FOV_DEG,
) -> dict[str, Any]:
    """How much of the bank asks about a source the camera cannot show.

    Only questions that carry a measured source bearing can be counted; the share is
    reported over those, and the count is reported beside it so a small denominator is
    visible rather than hidden.
    """
    if not isinstance(camera_half_fov_deg, (int, float)) or isinstance(camera_half_fov_deg, bool):
        raise ValueError("camera_half_fov_deg must be a number")
    if not 0 < float(camera_half_fov_deg) < 180:
        raise ValueError("camera_half_fov_deg must be between 0 and 180")
    splits = dict(splits or {})
    buckets: dict[str, list[float]] = {split: [] for split in ("bank", *_SPLITS)}
    for row in answer_rows:
        azimuth = _azimuth(row)
        if azimuth is None:
            continue
        buckets["bank"].append(azimuth)
        split = splits.get(str(row.get("question_id")))
        if split in _SPLITS:
            buckets[split].append(azimuth)

    def summarise(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"bearing_questions": 0, "off_screen_questions": 0, "off_screen_share": None}
        off = sum(1 for value in values if abs(value) > float(camera_half_fov_deg))
        return {
            "bearing_questions": len(values),
            "off_screen_questions": off,
            "off_screen_share": off / len(values),
        }

    return {
        "camera_half_fov_deg": float(camera_half_fov_deg),
        "definition": "a source whose bearing is outside the camera half-angle cannot be seen, "
                      "so only spatial audio can place it",
        "bank": summarise(buckets["bank"]),
        "by_split": {split: summarise(buckets[split]) for split in _SPLITS},
    }


def _rule(name: str, measured: Any, limit: Any, ok: bool, detail: str) -> dict[str, Any]:
    return {"rule": name, "measured": measured, "limit": limit,
            "status": "pass" if ok else "blocked", "detail": detail}


def evaluate_release(
    audit: Mapping[str, Any],
    answer_rows: Sequence[Mapping[str, Any]],
    *,
    splits: Mapping[str, str] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check one audited bank against the declared standard.

    `audit` is the report `avengine.qa.prior_audit.audit_priors` produced for this same
    bank. The decision is derived from it rather than recomputed, so the receipt and the
    gate can never disagree about what was measured.
    """
    policy = json.loads(json.dumps(policy if policy is not None else DEFAULT_POLICY))
    if policy.get("schema") != DEFAULT_POLICY["schema"]:
        raise ValueError("unknown release policy schema")
    unknown = set(policy) - set(DEFAULT_POLICY)
    if unknown:
        raise ValueError(f"unknown release policy sections: {sorted(unknown)}")
    if str(audit.get("schema")) != "avengine_answer_prior_audit_v1":
        raise ValueError("release gate needs an avengine_answer_prior_audit_v1 report")
    if audit.get("split_status") != "complete":
        raise ValueError("release gate needs a bank whose questions all carry a split")

    spatial = measure_spatial_support(
        answer_rows, splits=splits,
        camera_half_fov_deg=policy["spatial"]["camera_half_fov_deg"],
    )
    rules: list[dict[str, Any]] = []

    for split, limit in policy["blind_baseline"]["max_train_constant_score"].items():
        measured = (audit.get("aggregate", {}).get(split) or {}).get("train_constant_micro_score")
        rules.append(_rule(
            f"blind_baseline:{split}", measured, limit,
            measured is not None and measured <= limit,
            "a constant answer fitted on train, scored on this split; the floor under every "
            "model number reported on it",
        ))

    for split, limit in policy["spatial"]["min_off_screen_share"].items():
        measured = spatial["by_split"][split]["off_screen_share"]
        rules.append(_rule(
            f"off_screen_share:{split}", measured, limit,
            measured is not None and measured >= limit,
            "share of bearing questions whose source the camera cannot show; below this a "
            "model with video and mono audio reaches the same score as one with spatial audio",
        ))

    by_qa = audit.get("by_qa") or {}
    worst_majority = {split: None for split in _SPLITS}
    offenders: dict[str, list[str]] = {split: [] for split in _SPLITS}
    for qa, result in by_qa.items():
        for split in _SPLITS:
            share = ((result.get("by_split", {}).get(split) or {}).get("open_answers") or {}).get("majority_share")
            if share is None:
                continue
            if worst_majority[split] is None or share > worst_majority[split]:
                worst_majority[split] = share
            if share > policy["answers"]["max_majority_share"]:
                offenders[split].append(qa)
    for split in ("valid", "test"):
        rules.append(_rule(
            f"answer_majority:{split}", worst_majority[split], policy["answers"]["max_majority_share"],
            worst_majority[split] is not None and worst_majority[split] <= policy["answers"]["max_majority_share"],
            "worst commonest-answer share over the types; offenders: "
            + (", ".join(sorted(offenders[split])) or "none"),
        ))

    binary_flagged = [qa for qa, result in by_qa.items() if result.get("binary_mcq_warning")]
    typed = [qa for qa, result in by_qa.items() if (result.get("mcq_questions") or 0) > 0]
    share = len(binary_flagged) / len(typed) if typed else None
    rules.append(_rule(
        "binary_mcq_share", share, policy["options"]["max_binary_mcq_share"],
        share is not None and share <= policy["options"]["max_binary_mcq_share"],
        "types whose multiple choice is two-way, so guessing scores 50%; offenders: "
        + (", ".join(sorted(binary_flagged)) or "none"),
    ))

    under = sorted(qa for qa, result in by_qa.items()
                   if result.get("validation_quota_status") == "under_powered")
    rules.append(_rule(
        "validation_quota", len(under), 0, not under,
        f"types with fewer than {policy['support']['min_valid_questions_per_type']} validation "
        "questions, whose numbers are noise: " + (", ".join(under) or "none"),
    ))

    blocked = [rule for rule in rules if rule["status"] == "blocked"]
    return {
        "schema": "avengine_qa_release_gate_v1",
        "status": "blocked" if blocked else "release",
        "policy": policy,
        "spatial_support": spatial,
        "rules": rules,
        "blocked_rules": [rule["rule"] for rule in blocked],
        "claim_boundary": "a gate on the generated bank only; it makes no claim about any model",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_release_receipt(
    bank: Path | str,
    *,
    output: Path | str | None = None,
    policy: Mapping[str, Any] | None = None,
    split_views: Path | str | None = None,
) -> dict[str, Any]:
    """Audit a bank, decide against the policy, and leave the decision beside it."""
    from avengine.qa.prior_audit import audit_bank

    bank = Path(bank)
    audit = audit_bank(bank, split_views=split_views)
    answers = _read_jsonl(bank / "private" / "answers.jsonl")
    splits = {}
    split_file = bank / "private" / "splits.jsonl"
    if split_file.is_file():
        for row in _read_jsonl(split_file):
            splits[str(row.get("question_id"))] = str(row.get("split"))
    result = evaluate_release(audit, answers, splits=splits, policy=policy)
    destination = Path(output) if output is not None else bank / "private" / "release_gate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
