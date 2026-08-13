"""十类问题的 QuestionSpec 模板目录。

本模块为 AVEngine QA 管线提供十类标准问题模板。
每个常量是一个不可变的 QuestionSpec 实例，可直接用于管线。

用法::

    from avengine.qa.question_catalog import (
        QS_SOUND_PRESENCE, QS_TRANSCRIPT, QS_TRANSCRIPT_TO_ATTR,
        QS_SPATIAL_DIRECTION, QS_SOUND_ORDER, QS_SOUND_OVERLAP,
        QS_SOUND_MOTION, QS_ENTER_FRUSTUM, QS_SOUND_VISIBILITY,
        QS_OCCLUDER_IDENTITY,
        ALL_QUESTION_SPECS,
    )

    pipeline = QuestionPipeline(spec=QS_SOUND_PRESENCE, ...)
"""

from __future__ import annotations

from avengine.qa.question_spec import QuestionSpec

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 1：外貌属性 → 是否发声
# ═══════════════════════════════════════════════════════════════════════════════

QS_SOUND_PRESENCE = QuestionSpec(
    spec_id="qs_sound_presence_v1",
    question_type="sound_presence",
    template="穿{top_color}上衣的{species_id}在{time_window}内是否发声？",
    answer_modality="sound_facts",
    required_actor_count=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 2：外貌属性 → 发声内容
# ═══════════════════════════════════════════════════════════════════════════════

QS_TRANSCRIPT = QuestionSpec(
    spec_id="qs_transcript_v1",
    question_type="sound_transcript",
    template="穿{top_color}上衣的{species_id}在{time_window}内说了什么？",
    answer_modality="sound_transcript",
    required_actor_count=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 3：发声内容 → 外貌属性
# ═══════════════════════════════════════════════════════════════════════════════

QS_TRANSCRIPT_TO_ATTR = QuestionSpec(
    spec_id="qs_transcript_to_attr_v1",
    question_type="transcript_to_attribute",
    template='说"{transcript}"的{species_id}穿什么颜色的上衣？',
    answer_modality="reverse_attribute",
    required_actor_count=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 4：发声者的空间方向
# ═══════════════════════════════════════════════════════════════════════════════

QS_SPATIAL_DIRECTION = QuestionSpec(
    spec_id="qs_spatial_direction_v1",
    question_type="spatial_direction",
    template="穿{top_color}上衣的{species_id}在{time_window}内发声时，位于听者的哪个方向？",
    answer_modality="spatial_direction",
    required_actor_count=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 5：发声先后顺序（双角色）
# ═══════════════════════════════════════════════════════════════════════════════

QS_SOUND_ORDER = QuestionSpec(
    spec_id="qs_sound_order_v1",
    question_type="sound_order",
    template="在{time_window}内，{species_id_a}和{species_id_b}谁先发声？",
    answer_modality="sound_order",
    required_actor_count=2,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 6：重叠发声（双角色）
# ═══════════════════════════════════════════════════════════════════════════════

QS_SOUND_OVERLAP = QuestionSpec(
    spec_id="qs_sound_overlap_v1",
    question_type="overlap_sound",
    template="{species_id_a}叫时，{species_id_b}是否也在发声？",
    answer_modality="overlap_sound",
    required_actor_count=2,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 7：发声时的运动状态
# ═══════════════════════════════════════════════════════════════════════════════

QS_SOUND_MOTION = QuestionSpec(
    spec_id="qs_sound_motion_v1",
    question_type="sound_motion",
    template="{species_id}在{time_window}内叫的时候，正在走还是静止？",
    answer_modality="sound_motion",
    required_actor_count=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 8：画外到入画
# ═══════════════════════════════════════════════════════════════════════════════

QS_ENTER_FRUSTUM = QuestionSpec(
    spec_id="qs_enter_frustum_v1",
    question_type="enter_frustum_direction",
    template="一直在画外叫的{species_id}，从画面的哪一侧进入？",
    answer_modality="enter_frustum_direction",
    required_actor_count=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 9：发声时的像素遮挡状态
# ═══════════════════════════════════════════════════════════════════════════════

QS_SOUND_VISIBILITY = QuestionSpec(
    spec_id="qs_sound_visibility_v1",
    question_type="sound_visibility",
    template="{species_id}在{time_window}内第一次叫时，是清晰可见、部分遮挡、完全遮挡还是画外？",
    answer_modality="sound_visibility",
    required_actor_count=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 类型 10：遮挡者识别
# ═══════════════════════════════════════════════════════════════════════════════

QS_OCCLUDER_IDENTITY = QuestionSpec(
    spec_id="qs_occluder_identity_v1",
    question_type="occluder_identity",
    template="挡住正在叫的{species_id}的是家具还是另一只动物？",
    answer_modality="occluder_identity",
    required_actor_count=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 全部目录
# ═══════════════════════════════════════════════════════════════════════════════

ALL_QUESTION_SPECS: tuple[QuestionSpec, ...] = (
    QS_SOUND_PRESENCE,
    QS_TRANSCRIPT,
    QS_TRANSCRIPT_TO_ATTR,
    QS_SPATIAL_DIRECTION,
    QS_SOUND_ORDER,
    QS_SOUND_OVERLAP,
    QS_SOUND_MOTION,
    QS_ENTER_FRUSTUM,
    QS_SOUND_VISIBILITY,
    QS_OCCLUDER_IDENTITY,
)

__all__ = [
    "QS_SOUND_PRESENCE",
    "QS_TRANSCRIPT",
    "QS_TRANSCRIPT_TO_ATTR",
    "QS_SPATIAL_DIRECTION",
    "QS_SOUND_ORDER",
    "QS_SOUND_OVERLAP",
    "QS_SOUND_MOTION",
    "QS_ENTER_FRUSTUM",
    "QS_SOUND_VISIBILITY",
    "QS_OCCLUDER_IDENTITY",
    "ALL_QUESTION_SPECS",
]
