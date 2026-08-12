"""QuestionSpec — 问题模板与场景需求定义。

QuestionSpec 只规定问题结构和场景需求，不临时创造人物、动物或语句。
模板变量从资产库（B 的 registry）解析，角色从候选 registry 选择，
语音从带官方 transcript 的声音库选择。

用法::

    from avengine.qa.question_spec import (
        QuestionSpec, SceneRequirement, TemplateVariable,
        extract_scene_requirement, instantiate_template,
    )

    spec = QuestionSpec(
        spec_id="qs_sound_presence_color",
        question_type="sound_presence",
        template="穿{top_color}上衣的人是否在{time_window}发声？",
        answer_modality="sound_facts",
        required_actor_count=1,
    )
    req = extract_scene_requirement(spec, {"top_color": "blue"})
    text = instantiate_template(spec, {"top_color": "blue", "time_window": "0-3秒"})
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# 模板变量模式：{variable_name}
_TEMPLATE_VAR = re.compile(r"\{(\w+)\}")


# ═══════════════════════════════════════════════════════════════════════════════
# 数据类型
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TemplateVariable:
    """模板变量元数据。

    Attributes:
        name: 变量名（模板中的占位符名称）。
        source: 变量来源（``"actor_attr"`` / ``"sound_attr"`` / ``"time"``）。
        type_hint: 类型提示（``"str"`` / ``"color"`` / ``"tick_range"``）。
    """

    name: str
    source: str = "actor_attr"
    type_hint: str = "str"


@dataclass(frozen=True)
class QuestionSpec:
    """问题模板 —— 只定义结构，不绑定具体资产。

    Attributes:
        spec_id: 唯一标识。
        question_type: 问题类别（如 ``"sound_presence"``、``"visibility"``）。
        template: 问题文本模板，含 ``{variable}`` 占位符。
        answer_modality: 答案来源（``"sound_facts"`` / ``"visibility_facts"``）。
        required_actor_count: 需要的目标角色数量。
        time_window: 可选时间窗口 ``(start_tick, end_tick)``，单位 48 kHz tick。
    """

    spec_id: str
    question_type: str
    template: str
    answer_modality: str = "sound_facts"
    required_actor_count: int = 1
    time_window: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.spec_id:
            raise ValueError("spec_id 不能为空")
        if not self.question_type:
            raise ValueError("question_type 不能为空")
        if not self.template:
            raise ValueError("template 不能为空")
        if self.required_actor_count < 1:
            raise ValueError("required_actor_count 必须 >= 1")
        if self.answer_modality not in _VALID_MODALITIES:
            raise ValueError(
                f"answer_modality 必须是以下之一: {_VALID_MODALITIES}"
            )
        if self.time_window is not None:
            start, end = self.time_window
            if start < 0 or end < start:
                raise ValueError(
                    f"time_window ({start}, {end}) 无效，须满足 0 <= start <= end"
                )

    @property
    def variable_names(self) -> list[str]:
        """模板中所有变量名列表。"""
        return _TEMPLATE_VAR.findall(self.template)


@dataclass(frozen=True)
class SceneRequirement:
    """从 QuestionSpec 提取的场景约束。

    由 ``extract_scene_requirement()`` 生成，供候选选择器消费。

    Attributes:
        spec_id: 来源 QuestionSpec 的标识。
        target_attributes: 目标角色 A 必须满足的属性映射。
        target_attributes_b: 目标角色 B 必须满足的属性映射（双角色问题）。
        required_sound_type: 需要的语义声音类别（如 ``"human_speech"``）。
        time_window: 可选时间窗口 ``(start_tick, end_tick)``。
        min_actors_in_scene: 场景中至少需要的角色总数。
        attribute_uniqueness_required: 目标属性在场景中是否必须唯一。
    """

    spec_id: str
    target_attributes: dict[str, str] = field(default_factory=dict)
    target_attributes_b: dict[str, str] = field(default_factory=dict)
    required_sound_type: str | None = None
    time_window: tuple[int, int] | None = None
    min_actors_in_scene: int = 1
    attribute_uniqueness_required: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# 有效值
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_MODALITIES = frozenset({
    # 原始三种
    "sound_facts",
    "visibility_facts",
    "motion_facts",
    # 十类问题新增
    "sound_transcript",         # 类型 2：外貌→发声内容
    "reverse_attribute",        # 类型 3：发声内容→外貌属性
    "spatial_direction",        # 类型 4：发声者的空间方向
    "sound_order",              # 类型 5：发声先后顺序
    "overlap_sound",            # 类型 6：重叠发声
    "sound_motion",             # 类型 7：发声时的运动状态
    "enter_frustum_direction",  # 类型 8：画外到入画
    "sound_visibility",         # 类型 9：发声时的像素遮挡状态
    "occluder_identity",        # 类型 10：遮挡者识别
})

# actor_attr 来源的已知属性名（与 Episode.add_actor() 保持一致）
_KNOWN_ACTOR_ATTRS: frozenset[str] = frozenset({
    "species_id",
    "breed_id",
    "size",
    "body_build",
    "life_stage",
    "top_color",
})


# ═══════════════════════════════════════════════════════════════════════════════
# 公开函数
# ═══════════════════════════════════════════════════════════════════════════════


def extract_scene_requirement(
    spec: QuestionSpec,
    attribute_values: dict[str, str],
    *,
    required_sound_type: str | None = None,
    time_window: tuple[int, int] | None = None,
    min_actors_in_scene: int | None = None,
    attribute_uniqueness_required: bool = True,
) -> SceneRequirement:
    """从 QuestionSpec 和具体属性值构建 SceneRequirement。

    Args:
        spec: 问题模板。
        attribute_values: 模板变量 → 具体值的映射。
        required_sound_type: 覆盖 spec 默认的声音类型需求。
        time_window: 覆盖 spec 默认的时间窗口。
        min_actors_in_scene: 覆盖 spec 的 `required_actor_count`。
        attribute_uniqueness_required: 场景中属性是否必须唯一。

    Returns:
        构建好的 SceneRequirement。

    Raises:
        ValueError: 若属性值中包含未知键。
    """
    # 验证属性名
    unknown = set(attribute_values) - _KNOWN_ACTOR_ATTRS
    if unknown:
        raise ValueError(
            f"未知的角色属性: {sorted(unknown)}。"
            f"已知属性: {sorted(_KNOWN_ACTOR_ATTRS)}"
        )

    return SceneRequirement(
        spec_id=spec.spec_id,
        target_attributes=dict(attribute_values),
        required_sound_type=(
            required_sound_type
            if required_sound_type is not None
            else _default_sound_type(spec.question_type)
        ),
        time_window=(
            time_window
            if time_window is not None
            else spec.time_window
        ),
        min_actors_in_scene=(
            min_actors_in_scene
            if min_actors_in_scene is not None
            else spec.required_actor_count
        ),
        attribute_uniqueness_required=attribute_uniqueness_required,
    )


def instantiate_template(
    spec: QuestionSpec,
    bindings: dict[str, str],
) -> str:
    """将模板中的变量替换为具体值，生成完整问题文本。

    Args:
        spec: 问题模板。
        bindings: 变量名 → 具体文本的映射。

    Returns:
        替换后的完整问题文本。

    Raises:
        ValueError: 若模板中有变量未提供绑定值。
    """
    required = set(spec.variable_names)
    provided = set(bindings)
    missing = required - provided
    if missing:
        raise ValueError(
            f"模板变量缺少绑定值: {sorted(missing)}"
        )

    result = spec.template
    for name in required:
        result = result.replace(f"{{{name}}}", bindings[name])
    return result


def list_template_variables(spec: QuestionSpec) -> list[TemplateVariable]:
    """列出模板中的所有变量及其元数据。

    Args:
        spec: 问题模板。

    Returns:
        TemplateVariable 列表，按出现顺序排列。
    """
    seen: set[str] = set()
    result: list[TemplateVariable] = []
    for name in spec.variable_names:
        if name in seen:
            continue
        seen.add(name)
        source, type_hint = _infer_variable(name, spec)
        result.append(TemplateVariable(name=name, source=source, type_hint=type_hint))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════════════════════════════════════════


def _infer_variable(name: str, spec: QuestionSpec) -> tuple[str, str]:
    """根据变量名推断其来源和类型。"""
    if name == "time_window":
        return ("time", "tick_range")
    if name in _KNOWN_ACTOR_ATTRS:
        return ("actor_attr", _attr_type_hint(name))
    return ("actor_attr", "str")


def _attr_type_hint(name: str) -> str:
    """已知属性的类型提示映射。"""
    if name in ("top_color", "breed_id", "species_id"):
        return "str"
    if name in ("size", "body_build"):
        return "enum"
    if name == "life_stage":
        return "enum"
    return "str"


def _default_sound_type(question_type: str) -> str | None:
    """根据问题类型推断默认所需声音类型。"""
    if question_type == "sound_presence":
        return None  # 由候选选择器根据 actor 决定
    return None


__all__ = [
    "QuestionSpec",
    "SceneRequirement",
    "TemplateVariable",
    "extract_scene_requirement",
    "instantiate_template",
    "list_template_variables",
]
