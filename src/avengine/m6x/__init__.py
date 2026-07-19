"""Fixed-room M6.x contracts and executable canary support."""

from avengine.m6x.capture_adapter import (
    FixedApartmentCaptureAdapter,
    HUMAN_BEAGLE_CAPTURE_ADAPTER,
    HumanBeagleCaptureAdapter,
)
from avengine.m6x.contracts import (
    ANCHOR_LIBRARY_SCHEMA,
    M6XContractError,
    ROOM_CAPSULE_SCHEMA,
    SCENARIO_CONTRACT,
    SCENARIO_SUITE_SCHEMA,
    TRAJECTORY_TEMPLATE_SET_SCHEMA,
    load_anchor_library,
    load_room_capsule,
    load_scenario_suite,
    load_trajectory_template_set,
    validate_anchor_library,
    validate_room_capsule,
    validate_scenario_suite,
    validate_trajectory_template_set,
)

__all__ = [
    "ANCHOR_LIBRARY_SCHEMA",
    "FixedApartmentCaptureAdapter",
    "HUMAN_BEAGLE_CAPTURE_ADAPTER",
    "HumanBeagleCaptureAdapter",
    "M6XContractError",
    "ROOM_CAPSULE_SCHEMA",
    "SCENARIO_CONTRACT",
    "SCENARIO_SUITE_SCHEMA",
    "TRAJECTORY_TEMPLATE_SET_SCHEMA",
    "load_anchor_library",
    "load_room_capsule",
    "load_scenario_suite",
    "load_trajectory_template_set",
    "validate_anchor_library",
    "validate_room_capsule",
    "validate_scenario_suite",
    "validate_trajectory_template_set",
]
