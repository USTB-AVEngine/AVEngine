"""M7 research-scale dataset assembly helpers.

These helpers deliberately consume an already-qualified source-route/RIR
closure.  They do not replan geometry or run native propagation.
"""

from avengine.m7.asset_bound_audio import (
    ASSET_BOUND_AUDIO_SCHEMA,
    AssetBoundAudioError,
    PreparedDryAudio,
    float32_stems_and_exact_mix,
    prepare_dry_audio,
    render_asset_bound_binaural,
)

__all__ = [
    "ASSET_BOUND_AUDIO_SCHEMA",
    "AssetBoundAudioError",
    "PreparedDryAudio",
    "float32_stems_and_exact_mix",
    "prepare_dry_audio",
    "render_asset_bound_binaural",
]

from .room_evaluation import (
    RoomEvaluationError,
    RoomEvaluationPlan,
    build_room_evaluation_plan,
    validate_episode_id,
)

__all__ += [
    "RoomEvaluationError",
    "RoomEvaluationPlan",
    "build_room_evaluation_plan",
    "validate_episode_id",
]
