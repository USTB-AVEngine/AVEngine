"""AVEngine QA Episode — 像素可见性、事实数据与问答生成。

``qa`` 包在权威 M5 Timeline v2 之上扩展了结构化逐帧事实数据
（声音、空间、运动、可见性、遮挡）以及经过验证的问答对。
不复制或替换已有的传感器、音频或 Timeline 管线。

用法::

    from avengine.qa import (
        # 数据模型
        Episode, EpisodeEvent, QAPair, VisibilityRecord, EpisodeError,
        classify_visibility, detect_visibility_events, make_visibility_record,
        validate_qa_episode, validate_qa_episode_schema,
        QA_EPISODE_SCHEMA,
        # 可见性状态
        VISIBILITY_OUT_OF_VIEW, VISIBILITY_CLEAR,
        VISIBILITY_OCCLUDED, VISIBILITY_FULLY_OCCLUDED,
        # 事件类型
        EVENT_ENTER_FRUSTUM, EVENT_EXIT_FRUSTUM,
        EVENT_BECOME_VISIBLE, EVENT_OCCLUSION_START,
        EVENT_FULLY_OCCLUDED, EVENT_REAPPEAR,
        # 运动状态
        MOTION_IDLE, MOTION_WALK, MOTION_OTHER,
        # 遮挡物类型
        OCCLUDER_ACTOR, OCCLUDER_FURNITURE, OCCLUDER_UNKNOWN,
        # 像素分析
        analyze_all_frames, analyze_frame, compute_bbox,
        count_semantic_pixels, detect_border_touch, detect_occluders,
        # QuestionSpec
        QuestionSpec, SceneRequirement, TemplateVariable,
        extract_scene_requirement, instantiate_template,
        # 候选选择
        ActorRegistry, SoundRegistry, ActorCandidate, SoundCandidate,
        AssetBinding, select_candidates,
        FakeActorRegistry, FakeSoundRegistry,
        # 答案推导
        derive_answer, check_answer_unique, check_fact_observable,
        # 管线编排
        QuestionPipeline,
    )
"""

from __future__ import annotations

from avengine.qa.answer_deriver import (
    check_answer_unique,
    check_fact_observable,
    derive_answer,
)
from avengine.qa.candidate_selector import (
    ActorCandidate,
    ActorRegistry,
    AssetBinding,
    FakeActorRegistry,
    FakeSoundRegistry,
    SoundCandidate,
    SoundRegistry,
    select_candidates,
)
from avengine.qa.episode import (
    # ── 数据模型 ──────────────────────────────────────────────────────
    DEFAULT_CLEAR_THRESHOLD,
    DEFAULT_VISIBLE_THRESHOLD,
    Episode,
    EpisodeError,
    EpisodeEvent,
    QAPair,
    QA_EPISODE_SCHEMA,
    VisibilityRecord,
    # ── 可见性 ──────────────────────────────────────────────────────
    VISIBILITY_CLEAR,
    VISIBILITY_FULLY_OCCLUDED,
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUT_OF_VIEW,
    VISIBILITY_STATES,
    classify_visibility,
    detect_visibility_events,
    make_visibility_record,
    # ── 事件 ──────────────────────────────────────────────────────────
    EVENT_BECOME_VISIBLE,
    EVENT_ENTER_FRUSTUM,
    EVENT_EXIT_FRUSTUM,
    EVENT_FULLY_OCCLUDED,
    EVENT_OCCLUSION_START,
    EVENT_REAPPEAR,
    EVENT_TYPES,
    # ── 运动 ──────────────────────────────────────────────────────────
    MOTION_IDLE,
    MOTION_OTHER,
    MOTION_STATES,
    MOTION_WALK,
    # ── 遮挡 ──────────────────────────────────────────────────────────
    OCCLUDER_ACTOR,
    OCCLUDER_FURNITURE,
    OCCLUDER_TYPES,
    OCCLUDER_UNKNOWN,
    # ── 校验 ──────────────────────────────────────────────────────────
    validate_qa_episode,
    validate_qa_episode_schema,
)
from avengine.qa.pixel_visibility import (
    analyze_all_frames,
    analyze_frame,
    compute_bbox,
    count_semantic_pixels,
    detect_border_touch,
    detect_occluders,
)
from avengine.qa.question_catalog import (
    ALL_QUESTION_SPECS,
    QS_ENTER_FRUSTUM,
    QS_OCCLUDER_IDENTITY,
    QS_SOUND_MOTION,
    QS_SOUND_ORDER,
    QS_SOUND_OVERLAP,
    QS_SOUND_PRESENCE,
    QS_SOUND_VISIBILITY,
    QS_SPATIAL_DIRECTION,
    QS_TRANSCRIPT,
    QS_TRANSCRIPT_TO_ATTR,
)
from avengine.qa.question_pipeline import (
    QuestionPipeline,
)
from avengine.qa.question_spec import (
    QuestionSpec,
    SceneRequirement,
    TemplateVariable,
    extract_scene_requirement,
    instantiate_template,
    list_template_variables,
)

__all__ = [
    # 数据模型
    "Episode",
    "EpisodeError",
    "EpisodeEvent",
    "QAPair",
    "VisibilityRecord",
    "QA_EPISODE_SCHEMA",
    "DEFAULT_CLEAR_THRESHOLD",
    "DEFAULT_VISIBLE_THRESHOLD",
    # 可见性
    "VISIBILITY_OUT_OF_VIEW",
    "VISIBILITY_CLEAR",
    "VISIBILITY_OCCLUDED",
    "VISIBILITY_FULLY_OCCLUDED",
    "VISIBILITY_STATES",
    "classify_visibility",
    "detect_visibility_events",
    "make_visibility_record",
    # 事件
    "EVENT_ENTER_FRUSTUM",
    "EVENT_EXIT_FRUSTUM",
    "EVENT_BECOME_VISIBLE",
    "EVENT_OCCLUSION_START",
    "EVENT_FULLY_OCCLUDED",
    "EVENT_REAPPEAR",
    "EVENT_TYPES",
    # 运动
    "MOTION_IDLE",
    "MOTION_WALK",
    "MOTION_OTHER",
    "MOTION_STATES",
    # 遮挡
    "OCCLUDER_ACTOR",
    "OCCLUDER_FURNITURE",
    "OCCLUDER_UNKNOWN",
    "OCCLUDER_TYPES",
    # 校验
    "validate_qa_episode",
    "validate_qa_episode_schema",
    # 像素可见性（任务二）
    "analyze_all_frames",
    "analyze_frame",
    "compute_bbox",
    "count_semantic_pixels",
    "detect_border_touch",
    "detect_occluders",
    # QuestionSpec（任务三）
    "QuestionSpec",
    "SceneRequirement",
    "TemplateVariable",
    "extract_scene_requirement",
    "instantiate_template",
    "list_template_variables",
    # 候选选择
    "ActorCandidate",
    "ActorRegistry",
    "AssetBinding",
    "FakeActorRegistry",
    "FakeSoundRegistry",
    "SoundCandidate",
    "SoundRegistry",
    "select_candidates",
    # 答案推导
    "check_answer_unique",
    "check_fact_observable",
    "derive_answer",
    # 管线编排
    "QuestionPipeline",
    # 问题目录（十类问题模板）
    "ALL_QUESTION_SPECS",
    "QS_ENTER_FRUSTUM",
    "QS_OCCLUDER_IDENTITY",
    "QS_SOUND_MOTION",
    "QS_SOUND_ORDER",
    "QS_SOUND_OVERLAP",
    "QS_SOUND_PRESENCE",
    "QS_SOUND_VISIBILITY",
    "QS_SPATIAL_DIRECTION",
    "QS_TRANSCRIPT",
    "QS_TRANSCRIPT_TO_ATTR",
]
