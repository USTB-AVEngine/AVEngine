"""M2 canonical animal and deterministic articulated-capture contracts."""

from avengine.m2.contracts import (
    ANIMAL_SCHEMA,
    APPLIED_STATE_HASH_ALGORITHM,
    CAPTURE_SCHEMA,
    POSE_HASH_ALGORITHM,
    ContractError,
    ValidatedM2Inputs,
    compute_applied_state_hash,
    compute_pose_hash,
    load_and_validate_inputs,
    validate_animal_asset_package,
    validate_capture_request,
)

__all__ = [
    "ANIMAL_SCHEMA",
    "APPLIED_STATE_HASH_ALGORITHM",
    "CAPTURE_SCHEMA",
    "POSE_HASH_ALGORITHM",
    "ContractError",
    "ValidatedM2Inputs",
    "compute_applied_state_hash",
    "compute_pose_hash",
    "load_and_validate_inputs",
    "validate_animal_asset_package",
    "validate_capture_request",
]
