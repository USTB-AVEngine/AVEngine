"""QuestionSpec 管线编排 —— 从模板到 QAPair 的完整流程。

用法::

    from avengine.qa.question_pipeline import QuestionPipeline

    pipeline = QuestionPipeline(
        spec=question_spec,
        actor_registry=fake_actors,
        sound_registry=fake_sounds,
        max_attempts=10,
    )

    # 方法 A：声明式运行（由调用方提供已构建的 Episode 文档）
    qa_pair = pipeline.try_derive_from_doc(episode_doc, binding)

    # 方法 B：交互式运行（枚举候选，等待调用方构建并传回 docs）
    for batch in pipeline.iter_batches():
        for binding in batch:
            doc = build_episode_from_binding(binding)  # 调用方负责渲染
            qa = pipeline.try_derive_from_doc(doc, binding)
            if qa is not None:
                return qa  # 成功
    return None  # 超过 max_attempts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from avengine.qa.answer_deriver import (
    check_answer_unique,
    check_fact_observable,
    derive_answer,
)
from avengine.qa.candidate_selector import (
    ActorRegistry,
    AssetBinding,
    FakeActorRegistry,
    FakeSoundRegistry,
    SoundRegistry,
    select_candidates,
)
from avengine.qa.episode import QAPair
from avengine.qa.question_spec import (
    QuestionSpec,
    SceneRequirement,
    extract_scene_requirement,
    instantiate_template,
)


@dataclass
class QuestionPipeline:
    """QuestionSpec 处理管线。

    编排完整流程：需求提取 → 候选选择 → 答案推导 → QAPair 构建。
    支持多次采样：若答案不唯一或不可观察，继续尝试下一组候选。

    Attributes:
        spec: 问题模板。
        actor_registry: 角色注册表。
        sound_registry: 声音注册表（可选）。
        max_attempts: 最大重试次数。
        attribute_values: 模板变量的具体值（如 ``{"top_color": "blue"}``）。
        time_window: 时间窗口 ``(start_tick, end_tick)``。
        attribute_uniqueness_required: 是否要求属性在场景中唯一。
    """

    spec: QuestionSpec
    actor_registry: ActorRegistry = field(default_factory=FakeActorRegistry)
    sound_registry: SoundRegistry | None = None
    max_attempts: int = 10
    attribute_values: dict[str, str] = field(default_factory=dict)
    time_window: tuple[int, int] | None = None
    attribute_uniqueness_required: bool = True

    # ── 内部状态 ────────────────────────────────────────────────────────

    _bindings: list[AssetBinding] = field(default_factory=list, repr=False)
    _attempt: int = field(default=0, init=False, repr=False)

    # ── 阶段 1：提取场景需求 ────────────────────────────────────────────

    @property
    def scene_requirement(self) -> SceneRequirement:
        """当前配置对应的场景需求。"""
        return extract_scene_requirement(
            self.spec,
            self.attribute_values,
            time_window=self.time_window,
            attribute_uniqueness_required=self.attribute_uniqueness_required,
        )

    # ── 阶段 2：枚举候选 ────────────────────────────────────────────────

    def iter_batches(self) -> list[AssetBinding]:
        """运行候选选择，返回一批 AssetBinding。

        每次调用会消耗一次尝试机会。若超过 ``max_attempts`` 返回空列表。

        Returns:
            候选绑定列表；若唯一性检查失败或超过最大尝试次数则为空。
        """
        if self._attempt >= self.max_attempts:
            return []

        requirement = self.scene_requirement
        bindings = select_candidates(
            requirement,
            self.actor_registry,
            self.sound_registry,
        )

        self._attempt += 1
        self._bindings = bindings
        return bindings

    # ── 阶段 3 + 4：推导答案并构建 QAPair ───────────────────────────────

    def try_derive_from_doc(
        self, doc: dict[str, Any], binding: AssetBinding
    ) -> QAPair | None:
        """从已构建的 Episode 文档尝试推导答案并构建 QAPair。

        若答案不唯一或不可观察，返回 None（调用方应重试下一组候选）。

        Args:
            doc: 已构建的 Episode 文档。
            binding: 当前资产绑定。

        Returns:
            有效的 QAPair，若验证失败则为 None。
        """
        # 生成问题文本
        question_text = self._question_text(binding)

        # 推导答案（传入管线时间窗口以覆盖 spec 默认值）
        answer_text, is_unique, is_observable = derive_answer(
            doc, self.spec, binding, time_window=self.time_window
        )

        if not is_unique:
            return None
        if not is_observable:
            return None

        # 构建 answer_source
        answer_source = _build_answer_source(
            doc, self.spec.answer_modality, binding.actor.actor_id
        )

        return QAPair(
            question_id=f"{self.spec.spec_id}_q{self._attempt}",
            question_type=self.spec.question_type,
            question_text=question_text,
            answer_text=answer_text,
            answer_unique=True,
            fact_observable=True,
            answer_source=answer_source,
        )

    def _question_text(self, binding: AssetBinding) -> str:
        """使用绑定的属性值实例化问题文本。"""
        bindings: dict[str, str] = dict(binding.attribute_values)
        # 添加时间窗口的可读表示
        if self.time_window is not None:
            start_s = self.time_window[0] / 48_000
            end_s = self.time_window[1] / 48_000
            bindings["time_window"] = f"{start_s:.1f}-{end_s:.1f}秒"
        elif self.spec.time_window is not None:
            start_s = self.spec.time_window[0] / 48_000
            end_s = self.spec.time_window[1] / 48_000
            bindings["time_window"] = f"{start_s:.1f}-{end_s:.1f}秒"
        return instantiate_template(self.spec, bindings)

    @property
    def attempt(self) -> int:
        """当前尝试次数。"""
        return self._attempt


def _build_answer_source(
    doc: dict[str, Any],
    modality: str,
    actor_id: str,
) -> dict[str, Any] | None:
    """从 Episode 文档构建 answer_source 元数据。"""
    facts = doc.get("facts", {})

    if modality == "sound_facts":
        sound_facts: list[dict[str, Any]] = facts.get("sound_facts", [])
        for i, fact in enumerate(sound_facts):
            if fact.get("actor_id") == actor_id:
                return {
                    "fact_path": f"facts.sound_facts[{i}]",
                    "fact_value": fact.get("sound_asset_id"),
                    "frame_index": fact.get("start_frame"),
                }
        return {"fact_path": "facts.sound_facts", "fact_value": None}

    elif modality == "visibility_facts":
        return {
            "fact_path": f"facts.visibility_facts.per_frame[*].actor_visibility.{actor_id}",
            "fact_value": "visibility_state",
        }

    return None


__all__ = [
    "QuestionPipeline",
]
