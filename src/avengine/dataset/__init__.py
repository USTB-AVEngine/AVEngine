"""Research-scale dataset assembly and throughput helpers (formerly M7).

These helpers deliberately consume an already-qualified source-route/RIR
closure.  They do not replan geometry or run native propagation.
"""

from avengine.dataset.asset_bound_audio import (
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
    build_static_source_trajectory_bank,
    validate_episode_id,
)
from .episode_export import (
    EPISODE_EXPORT_REQUEST_SCHEMA,
    EPISODE_EXPORT_SCHEMA,
    EpisodeExportError,
    build_episode_export_records,
    export_episode_bundle,
    probe_video,
)
from .model_evaluation import (
    SPATIAL_OMNI_ADAPTER_SCHEMA,
    ModelEvaluationAdapterError,
    build_spatial_omni_benchmark_command,
    prepare_qwen25_omni_pilot_gold,
    prepare_qwen25_omni_pilot_inputs,
    prepare_whisper_review_request,
    prepare_spatial_omni_qa_root,
)

__all__ += [
    "RoomEvaluationError",
    "RoomEvaluationPlan",
    "build_room_evaluation_plan",
    "build_static_source_trajectory_bank",
    "validate_episode_id",
    "EPISODE_EXPORT_REQUEST_SCHEMA",
    "EPISODE_EXPORT_SCHEMA",
    "EpisodeExportError",
    "build_episode_export_records",
    "export_episode_bundle",
    "probe_video",
    "SPATIAL_OMNI_ADAPTER_SCHEMA",
    "ModelEvaluationAdapterError",
    "build_spatial_omni_benchmark_command",
    "prepare_qwen25_omni_pilot_gold",
    "prepare_qwen25_omni_pilot_inputs",
    "prepare_whisper_review_request",
    "prepare_spatial_omni_qa_root",
]
