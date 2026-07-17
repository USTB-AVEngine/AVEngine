"""Fail-closed animal appearance contracts and balanced design generation."""

from avengine.appearance.contracts import (
    ANIMAL_APPEARANCE_BATCH_SCHEMA,
    ANIMAL_APPEARANCE_INSTANCE_REQUEST_SCHEMA,
    ANIMAL_APPEARANCE_REQUEST_SCHEMA,
    APPEARANCE_AXES,
    L9_ALGORITHM,
    AppearanceContractError,
    build_l9_batch,
    generate_l9_batch,
    validate_appearance_request,
    validate_l9_batch,
    verify_instance_request_integrity,
    write_l9_batch_exclusive,
)

__all__ = [
    "ANIMAL_APPEARANCE_BATCH_SCHEMA",
    "ANIMAL_APPEARANCE_INSTANCE_REQUEST_SCHEMA",
    "ANIMAL_APPEARANCE_REQUEST_SCHEMA",
    "APPEARANCE_AXES",
    "L9_ALGORITHM",
    "AppearanceContractError",
    "build_l9_batch",
    "generate_l9_batch",
    "validate_appearance_request",
    "validate_l9_batch",
    "verify_instance_request_integrity",
    "write_l9_batch_exclusive",
]
