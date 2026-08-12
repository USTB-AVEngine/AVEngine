"""候选选择器 —— 从资产注册表中选择角色和声音。

本模块定义 Person B 必须满足的注册表协议接口（``Protocol``），
并提供 ``FakeActorRegistry`` 和 ``FakeSoundRegistry`` 用于测试。

用法::

    from avengine.qa.candidate_selector import (
        ActorRegistry, SoundRegistry, ActorCandidate, SoundCandidate,
        AssetBinding, FakeActorRegistry, FakeSoundRegistry, select_candidates,
    )

    actors = FakeActorRegistry()
    actors.add(ActorCandidate("human_01", "human_asset_01", "human", 10,
                               attributes={"top_color": "blue"}))
    sounds = FakeSoundRegistry()
    sounds.add(SoundCandidate("speech_01", "human_speech", "你好", 48000,
                               bound_to_actor="human_01"))

    req = SceneRequirement(spec_id="qs1", target_attributes={"top_color": "blue"})
    bindings = select_candidates(req, actors, sounds)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from avengine.qa.question_spec import SceneRequirement


# ═══════════════════════════════════════════════════════════════════════════════
# 候选数据类型
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ActorCandidate:
    """注册表中的角色候选。

    Attributes:
        actor_id: 稳定角色标识。
        entity_asset_id: 实体资产 ID（指向 B 的实体注册表）。
        species_id: 物种标识（如 ``"human"``、``"dog"``）。
        semantic_id: 语义分割 ID。
        attributes: 自由属性映射（如 ``{"top_color": "blue"}``）。
    """

    actor_id: str
    entity_asset_id: str
    species_id: str
    semantic_id: int
    attributes: dict[str, str] = field(default_factory=dict)

    def matches(self, required: dict[str, str]) -> bool:
        """检查该角色是否满足所有必需属性。"""
        for key, value in required.items():
            if self.attributes.get(key) != value:
                return False
        return True


@dataclass(frozen=True)
class SoundCandidate:
    """注册表中的声音候选。

    Attributes:
        sound_asset_id: 声音资产 ID。
        semantic_sound_class: 语义声音类别（如 ``"human_speech"``）。
        transcript: 文字转录。
        duration_samples: 时长（采样点数）。
        bound_to_actor: 绑定到的角色 ID（空字符串表示未绑定）。
    """

    sound_asset_id: str
    semantic_sound_class: str
    transcript: str = ""
    duration_samples: int = 0
    bound_to_actor: str = ""


@dataclass(frozen=True)
class AssetBinding:
    """一组具体的资产选择组合。

    Attributes:
        actor: 选中的角色。
        sound: 选中的声音（可为 None）。
        attribute_values: 本次绑定的属性具体值。
    """

    actor: ActorCandidate
    sound: SoundCandidate | None
    attribute_values: dict[str, str]


# ═══════════════════════════════════════════════════════════════════════════════
# 注册表协议 — Person B 必须满足的接口
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class ActorRegistry(Protocol):
    """角色注册表协议。

    Person B 负责提供具体实现，从实体资产注册表中查询匹配的角色。
    """

    def lookup_by_attributes(self, attrs: dict[str, str]) -> list[ActorCandidate]:
        """按属性查找匹配的角色候选列表。"""
        ...

    def list_all(self) -> list[ActorCandidate]:
        """列出注册表中所有角色。"""
        ...


@runtime_checkable
class SoundRegistry(Protocol):
    """声音注册表协议。

    Person B 负责提供具体实现，从声音资产注册表中查询匹配的声音。
    """

    def lookup_by_actor(
        self, actor_id: str, sound_type: str | None = None
    ) -> list[SoundCandidate]:
        """查找绑定到指定角色的声音候选列表。"""
        ...

    def list_all(self) -> list[SoundCandidate]:
        """列出注册表中所有声音。"""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Fake 实现 — 仅用于测试
# ═══════════════════════════════════════════════════════════════════════════════


class FakeActorRegistry:
    """基于内存列表的假角色注册表。

    不依赖外部文件，所有数据通过 :meth:`add` 手动添加。
    """

    def __init__(self) -> None:
        self._actors: list[ActorCandidate] = []

    def add(self, actor: ActorCandidate) -> None:
        """添加一个角色候选。"""
        self._actors.append(actor)

    def lookup_by_attributes(self, attrs: dict[str, str]) -> list[ActorCandidate]:
        """按属性查找匹配的角色候选。"""
        return [a for a in self._actors if a.matches(attrs)]

    def list_all(self) -> list[ActorCandidate]:
        """列出所有角色。"""
        return list(self._actors)

    def __len__(self) -> int:
        return len(self._actors)


class FakeSoundRegistry:
    """基于内存列表的假声音注册表。

    不依赖外部文件，所有数据通过 :meth:`add` 手动添加。
    """

    def __init__(self) -> None:
        self._sounds: list[SoundCandidate] = []

    def add(self, sound: SoundCandidate) -> None:
        """添加一个声音候选。"""
        self._sounds.append(sound)

    def lookup_by_actor(
        self, actor_id: str, sound_type: str | None = None
    ) -> list[SoundCandidate]:
        """查找绑定到指定角色的声音，可按 sound_type 过滤。"""
        result = [s for s in self._sounds if s.bound_to_actor == actor_id]
        if sound_type is not None:
            result = [s for s in result if s.semantic_sound_class == sound_type]
        return result

    def list_all(self) -> list[SoundCandidate]:
        """列出所有声音。"""
        return list(self._sounds)

    def __len__(self) -> int:
        return len(self._sounds)


# ═══════════════════════════════════════════════════════════════════════════════
# 选择逻辑
# ═══════════════════════════════════════════════════════════════════════════════


def select_candidates(
    requirement: SceneRequirement,
    actor_registry: ActorRegistry,
    sound_registry: SoundRegistry | None = None,
) -> list[AssetBinding]:
    """根据场景需求从注册表中选择候选角色和声音组合。

    流程：
    1. 从角色注册表中按属性查找匹配的角色。
    2. 如果要求属性唯一（``attribute_uniqueness_required=True``）
       且存在同一属性值的多个匹配，则拒绝。
    3. 为每个匹配的角色从声音注册表中查找声音。
    4. 返回所有有效组合。

    Args:
        requirement: 场景需求（属性、声音类型、唯一性要求等）。
        actor_registry: 角色注册表。
        sound_registry: 声音注册表（可选；None 时返回仅含角色的绑定）。

    Returns:
        AssetBinding 列表（候选角色 + 声音组合）。
        若要求唯一性且有歧义，返回空列表。
    """
    attrs = requirement.target_attributes

    # 步骤 1：查找匹配角色
    matching_actors = actor_registry.lookup_by_attributes(attrs)

    if not matching_actors:
        return []

    # 步骤 2：唯一性检查
    if requirement.attribute_uniqueness_required and len(matching_actors) > 1:
        # 同一属性值匹配到多个角色 → 答案不唯一，拒绝
        return []

    bindings: list[AssetBinding] = []

    for actor in matching_actors:
        # 收集属性值用于模板实例化
        attribute_values: dict[str, str] = dict(requirement.target_attributes)
        for key in requirement.target_attributes:
            if key in actor.attributes:
                attribute_values[key] = actor.attributes[key]

        # 步骤 3：查找声音
        matched_sounds: list[SoundCandidate | None] = [None]
        if sound_registry is not None and requirement.required_sound_type:
            sounds = sound_registry.lookup_by_actor(
                actor.actor_id,
                sound_type=requirement.required_sound_type,
            )
            if sounds:
                matched_sounds = sounds  # type: ignore[assignment]

        # 步骤 4：组装绑定
        for sound in matched_sounds:
            bindings.append(AssetBinding(
                actor=actor,
                sound=sound,
                attribute_values=attribute_values,
            ))

    return bindings


__all__ = [
    "ActorCandidate",
    "ActorRegistry",
    "AssetBinding",
    "FakeActorRegistry",
    "FakeSoundRegistry",
    "SoundCandidate",
    "SoundRegistry",
    "select_candidates",
]
