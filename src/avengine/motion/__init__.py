"""Versioned, body-plan-aware offline motion adaptation primitives.

The motion package deliberately stops at deterministic asset compilation.  It
does not run a free animation clock in Habitat and it does not pretend that a
quadruped gait is suitable for birds, fish, or other body plans.  A versioned
profile selects an explicit adapter and maps semantic joints between one
audited source motion family and one target template.
"""

from avengine.motion.math import (
    MotionMathError,
    canonical_quaternion_xyzw,
    quaternion_inverse_xyzw,
    quaternion_multiply_xyzw,
    retarget_world_rotation_xyzw,
    world_left_delta_xyzw,
)
from avengine.motion.profiles import (
    ADAPTER_CAPABILITIES,
    ActionMapping,
    AdapterCapability,
    AttributeDomain,
    JointMapping,
    MotionProfileError,
    MotionQACoordinateFrame,
    MotionRetargetProfile,
    SemanticChain,
    load_motion_retarget_profile,
)
from avengine.motion.qa import (
    ChainMotionMetrics,
    ChainMotionThresholds,
    ChainSymmetryMetrics,
    ChainSymmetryThreshold,
    GroupExcursionRatioThreshold,
    GroupMotionMetrics,
    GroupRatioMetrics,
    JointMotionMetrics,
    JointMotionThresholds,
    MotionQAContract,
    MotionQAIssue,
    MotionQAReport,
    SemanticChainGroup,
    SemanticChainSamples,
    evaluate_motion_qa,
)

__all__ = [
    "ADAPTER_CAPABILITIES",
    "ActionMapping",
    "AdapterCapability",
    "AttributeDomain",
    "ChainMotionMetrics",
    "ChainMotionThresholds",
    "ChainSymmetryMetrics",
    "ChainSymmetryThreshold",
    "GroupExcursionRatioThreshold",
    "GroupMotionMetrics",
    "GroupRatioMetrics",
    "JointMapping",
    "JointMotionMetrics",
    "JointMotionThresholds",
    "MotionMathError",
    "MotionProfileError",
    "MotionQAContract",
    "MotionQACoordinateFrame",
    "MotionQAIssue",
    "MotionQAReport",
    "MotionRetargetProfile",
    "SemanticChain",
    "SemanticChainGroup",
    "SemanticChainSamples",
    "canonical_quaternion_xyzw",
    "evaluate_motion_qa",
    "load_motion_retarget_profile",
    "quaternion_inverse_xyzw",
    "quaternion_multiply_xyzw",
    "retarget_world_rotation_xyzw",
    "world_left_delta_xyzw",
]
