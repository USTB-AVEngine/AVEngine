"""M5 exact-timeline dynamic audiovisual counterfactuals."""

from avengine.m5.audio import M5_AUDIO_SAMPLE_COUNT, M5_AUDIO_SAMPLE_RATE_HZ
from avengine.m5.canary import run_m5_canary, verify_m5_canary_evidence
from avengine.m5.timeline import (
    build_counterfactual_pair,
    build_timeline,
    validate_episode_request,
    validate_timeline_semantics,
)

__all__ = [
    "M5_AUDIO_SAMPLE_COUNT",
    "M5_AUDIO_SAMPLE_RATE_HZ",
    "build_counterfactual_pair",
    "build_timeline",
    "run_m5_canary",
    "validate_episode_request",
    "validate_timeline_semantics",
    "verify_m5_canary_evidence",
]
