"""Shared failure classification for QA runners and outcome collectors.

The batch schema has two explicit failure gap states.  A failure that is not
recognised by the rules below keeps the existing evidence gap state and carries
an unclassified diagnostic, so its reason and stage remain visible without
claiming that planning was exhausted.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


EVIDENCE_GAP_STATE = "evidence_missing_or_unsampled"
INTERFACE_GAP_STATE = "interface_not_implemented"
KNOWN_GAP_STATES = frozenset({EVIDENCE_GAP_STATE, INTERFACE_GAP_STATE})

_INTERFACE_EXCEPTION_TYPES = frozenset({
    "ModuleNotFoundError",
    "NameError",
    "SyntaxError",
    "ImportError",
    "NotImplementedError",
    "CalledProcessError",
    "UnifiedAudioReceiptError",
    "EvidenceContractError",
    "TypeError",
    "AttributeError",
    "FileNotFoundError",
    "RuntimeError",
    "SystemExit",
    "AssertionError",
    "KeyError",
    "ValueError",
    "OSError",
    "JSONDecodeError",
    "AudioProgramError",
    "SubprocessError",
})

_PLANNING_EXHAUSTION_MARKERS = (
    "conditionedplanningfailure",
    "fixed condition profile exhausted",
)
_PREALLOCATION_CODES = frozenset({"preallocation_gap", "preallocation_deficit"})
_PLANNING_EXHAUSTION_CODES = frozenset({"planning_exhausted"})
_CLIP_REJECTION_CODES = frozenset({
    "clip_overflow",
    "clip_overflow_rejected",
    "audio_clip_rejected",
})
_INTERFACE_REASON_CODES = frozenset({
    "controller_episode_mismatch",
    "controller_launch_failed",
    "interface_not_implemented",
})
_CLIP_REJECTION_MARKERS = (
    "audio output would clip",
    "would clip without normalization",
    "clip overflow",
    "clipping rejection",
)

_STATUS_TO_STAGE = {
    "preallocation_blocked": "planning",
    "planning_failed": "planning",
    "capture_failed": "capture",
    "audio_failed": "audio",
    "delivery_failed": "finalize",
    "review_failed": "finalize",
    "resource_failed": "launch",
}


def exception_type(reason: str) -> str | None:
    """Return the last exception class in a Type: message diagnostic."""
    if not isinstance(reason, str) or not reason.strip():
        return None
    match = None
    for match in re.finditer(
        r"(?:^|\n|[^A-Za-z0-9_.])(?:[A-Za-z_][\w]*\.)*"
        r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Failure|Conflict))\s*:",
        reason,
    ):
        pass
    return match.group(1) if match else None


def _is_cli_interface_error(reason: str) -> bool:
    lower = (reason or "").lower()
    return (
        "unrecognized arguments" in lower
        or "the following arguments are required" in lower
        or "no such option" in lower
        or "missing cli" in lower
        or lower.lstrip().startswith("error: argument")
    )


def _is_code_exception(reason: str) -> bool:
    if exception_type(reason) in _INTERFACE_EXCEPTION_TYPES:
        return True
    if _is_cli_interface_error(reason):
        return True
    lower = (reason or "").lower()
    return (
        "audioprogram validation" in lower
        or "unifiedaudioreceipt" in lower
        or "validate_evidence_contract" in lower
        or "evidencecontracterror" in lower
        or "not implemented" in lower
        or "interface_not_implemented" in lower
    )


def _is_planning_exhaustion(
    reason: str,
    *,
    histogram: Mapping[str, Any] | None = None,
    reason_code: str | None = None,
) -> bool:
    if reason_code in _PLANNING_EXHAUSTION_CODES:
        return True
    if isinstance(histogram, Mapping) and bool(histogram):
        return True
    lower = (reason or "").lower()
    return any(marker in lower for marker in _PLANNING_EXHAUSTION_MARKERS)


def _is_preallocation_deficit(reason: str, *, reason_code: str | None = None) -> bool:
    if reason_code in _PREALLOCATION_CODES:
        return True
    lower = (reason or "").lower()
    return "preallocation_gap" in lower or "preallocation gap" in lower


def _is_clip_rejection(reason: str, *, reason_code: str | None = None) -> bool:
    if reason_code in _CLIP_REJECTION_CODES:
        return True
    lower = (reason or "").lower()
    return any(marker in lower for marker in _CLIP_REJECTION_MARKERS)


def classify_failure(
    *,
    failure_stage: str | None = None,
    reason: str = "",
    reason_code: str | None = None,
    histogram: Mapping[str, Any] | None = None,
    status: str | None = None,
    declared_gap_state: str | None = None,
) -> dict[str, Any]:
    """Classify one failure while retaining an explicit diagnostic.

    Known failures use the existing evidence or interface gap states.  Unknown
    failures use the evidence gap state with an unclassified diagnostic; this
    keeps them in downstream coverage accounting without adding a sixth state.
    """
    stage = failure_stage if isinstance(failure_stage, str) and failure_stage else None
    if stage is None:
        stage = _STATUS_TO_STAGE.get(status or "")
    if stage is None and reason_code in _PREALLOCATION_CODES:
        stage = "planning"
    stage = stage or "unknown"
    text = reason if isinstance(reason, str) else str(reason or "")
    exception = exception_type(text)

    if declared_gap_state in KNOWN_GAP_STATES:
        gap_state = declared_gap_state
        classification = declared_gap_state
        classification_reason = "declared_gap_state"
    elif exception == "RequestedVisibilityError":
        gap_state = EVIDENCE_GAP_STATE
        classification = gap_state
        classification_reason = "native_visibility_rejection"
    elif exception == "ConditionedRequestConflict":
        gap_state = EVIDENCE_GAP_STATE
        classification = gap_state
        classification_reason = "request_conditions_conflict"
    elif exception in {"NameError", "SyntaxError"}:
        gap_state = INTERFACE_GAP_STATE
        classification = gap_state
        classification_reason = "controller_code_exception"
    elif _is_preallocation_deficit(text, reason_code=reason_code):
        gap_state = EVIDENCE_GAP_STATE
        classification = gap_state
        classification_reason = "preallocation_deficit"
    elif _is_planning_exhaustion(text, histogram=histogram, reason_code=reason_code):
        gap_state = EVIDENCE_GAP_STATE
        classification = gap_state
        classification_reason = "planning_exhaustion"
    elif _is_clip_rejection(text, reason_code=reason_code):
        gap_state = EVIDENCE_GAP_STATE
        classification = gap_state
        classification_reason = "clip_overflow_rejection"
    elif reason_code in _INTERFACE_REASON_CODES or _is_code_exception(text):
        gap_state = INTERFACE_GAP_STATE
        classification = gap_state
        classification_reason = "code_or_interface_exception"
    else:
        gap_state = EVIDENCE_GAP_STATE
        classification = "unclassified"
        classification_reason = "no_matching_failure_rule"

    diagnostic = {
        "classification": classification,
        "classification_reason": classification_reason,
        "failure_stage": stage,
        "reason_code": reason_code,
        "exception_type": exception,
        "failure_reason": text,
    }
    result = {
        "failure_stage": stage,
        "gap_state": gap_state,
        "failure_reason": text,
        "diagnostic": diagnostic,
    }
    if exception == "RequestedVisibilityError":
        result["reason_code"] = "requested_visibility_not_satisfied"
    elif exception == "ConditionedRequestConflict":
        result["reason_code"] = "conditioned_request_conflict"
    elif exception in {"NameError", "SyntaxError"}:
        result["reason_code"] = "controller_code_error"
    elif _is_clip_rejection(text, reason_code=reason_code):
        result["reason_code"] = "clip_overflow_rejected"
    elif classification == "unclassified":
        result["reason_code"] = "unclassified_failure"
    elif reason_code:
        result["reason_code"] = reason_code
    return result


def gap_state_for_failure(
    *,
    failure_stage: str | None,
    reason: str = "",
    reason_code: str | None = None,
    histogram: Mapping[str, Any] | None = None,
) -> str | None:
    """Compatibility helper returning only the existing nullable field."""
    return classify_failure(
        failure_stage=failure_stage,
        reason=reason,
        reason_code=reason_code,
        histogram=histogram,
    )["gap_state"]
