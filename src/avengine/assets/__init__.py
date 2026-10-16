"""Canonical animal assets: deterministic articulated-capture contracts (formerly M2)."""

from avengine.assets.contracts import (
    ANIMAL_SCHEMA,
    APPLIED_STATE_HASH_ALGORITHM,
    CAPTURE_SCHEMA,
    HUMAN_REVIEW_SCHEMA,
    POSE_HASH_ALGORITHM,
    ContractError,
    ValidatedM2Inputs,
    compute_applied_state_hash,
    compute_pose_hash,
    load_and_validate_inputs,
    validate_animal_asset_package,
    validate_capture_request,
    validate_human_visual_review,
)

__all__ = [
    "ANIMAL_SCHEMA",
    "APPLIED_STATE_HASH_ALGORITHM",
    "CAPTURE_SCHEMA",
    "HUMAN_REVIEW_SCHEMA",
    "POSE_HASH_ALGORITHM",
    "ContractError",
    "ValidatedM2Inputs",
    "compute_applied_state_hash",
    "compute_pose_hash",
    "load_and_validate_inputs",
    "validate_animal_asset_package",
    "validate_capture_request",
    "validate_human_visual_review",
]
