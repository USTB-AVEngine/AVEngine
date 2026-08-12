"""任务三验收测试 — QuestionSpec 端到端集成。

覆盖场景：
- 完整管线：spec → 候选选择 → 答案推导 → QAPair
- 两个蓝衣人 → 答案不唯一 → 拒绝（核心需求）
- 目标被完全遮挡 → 不可观察 → 拒绝
- 答案唯一且可观察 → 成功产出 QAPair
- 可见性模态问题
- 时间窗口边界
- max_attempts 耗尽
- 唯一性禁用
"""

from __future__ import annotations

import pytest

from avengine.qa.candidate_selector import (
    ActorCandidate,
    FakeActorRegistry,
    FakeSoundRegistry,
    SoundCandidate,
)
from avengine.qa.question_pipeline import QuestionPipeline
from avengine.qa.question_spec import QuestionSpec


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _setup_sound_presence_spec() -> QuestionSpec:
    """标准声音存在问题模板。"""
    return QuestionSpec(
        spec_id="sound_presence_v1",
        question_type="sound_presence",
        template="穿{top_color}上衣的人是否在{time_window}发声？",
        answer_modality="sound_facts",
    )


def _setup_visibility_spec() -> QuestionSpec:
    """可见性问题模板。"""
    return QuestionSpec(
        spec_id="visibility_v1",
        question_type="visibility",
        template="穿{top_color}上衣的人在{time_window}内是否可见？",
        answer_modality="visibility_facts",
    )


def _make_actor_dict(actor_id="a1", top_color="蓝色", species_id="human"):
    """创建符合 Episode actor dict 结构的角色。"""
    return {
        "actor_id": actor_id,
        "identity": {"species_id": species_id, "breed_id": f"{species_id}_default"},
        "realized_visual_attributes": {
            "clothing": {"top_color": top_color},
            "size": "medium",
        },
    }


def _make_doc(actors=None, sound_facts=None, vis_frames=None):
    """创建符合 Episode.build() 结构的文档。"""
    if actors is None:
        actors = [_make_actor_dict("a1")]
    if sound_facts is None:
        sound_facts = []
    if vis_frames is None:
        vis_frames = []

    return {
        "assets_used": {"actors": actors},
        "facts": {
            "sound_facts": sound_facts,
            "visibility_facts": {"per_frame": vis_frames},
            "motion_facts": {"per_frame": []},
        },
    }


def _setup_registry(
    n_actors: int = 1,
    actor_ids: list[str] | None = None,
    colors: list[str] | None = None,
) -> FakeActorRegistry:
    """创建含指定数量角色的注册表。"""
    reg = FakeActorRegistry()
    if actor_ids is None:
        actor_ids = [f"a{i+1}" for i in range(n_actors)]
    if colors is None:
        colors = ["蓝色"] * n_actors
    for i, (aid, color) in enumerate(zip(actor_ids, colors)):
        reg.add(ActorCandidate(
            aid, f"asset_{aid}", "human", 10 + i,
            attributes={"top_color": color, "species_id": "human"},
        ))
    return reg


# ═══════════════════════════════════════════════════════════════════════════════
# 核心场景：端到端声音存在问题
# ═══════════════════════════════════════════════════════════════════════════════


class TestSoundPresenceEndToEnd:
    """声音存在问题端到端：完整管线。"""

    def test_single_blue_actor_speaks(self):
        """单个蓝衣人发声 → 成功产出 QAPair，答案为‘是’。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            time_window=(0, 96000),
        )

        bindings = pipeline.iter_batches()
        assert len(bindings) == 1

        doc = _make_doc(
            actors=[_make_actor_dict("a1", top_color="蓝色")],
            sound_facts=[
                {"actor_id": "a1", "sound_asset_id": "speech_01",
                 "start_tick": 10_000, "end_tick": 50_000, "start_frame": 3},
            ],
            vis_frames=[
                {"frame_index": i, "actor_visibility": {"a1": {"visibility_state": "visible_clear"}}}
                for i in range(10)
            ],
        )

        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.question_type == "sound_presence"
        assert qa.answer_text == "是"
        assert qa.answer_unique is True
        assert qa.fact_observable is True
        assert "蓝色" in qa.question_text

    def test_single_blue_actor_silent(self):
        """单个蓝衣人不发声 → 答案为‘否’。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            time_window=(0, 96000),
        )

        bindings = pipeline.iter_batches()
        doc = _make_doc(actors=[_make_actor_dict("a1")])
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.answer_text == "否"


# ═══════════════════════════════════════════════════════════════════════════════
# 核心需求：两个蓝衣人 → 拒绝
# ═══════════════════════════════════════════════════════════════════════════════


class TestDuplicateAttributeRejection:
    """同属性重复 → 答案不唯一 → 拒绝（核心需求）。"""

    def test_two_blue_actors_candidate_level(self):
        """注册表中有两个 blue 角色 → select_candidates 在候选层拒绝。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(2, ["a1", "a2"], ["蓝色", "蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
        )
        bindings = pipeline.iter_batches()
        # 唯一性检查失败 → 返回空
        assert bindings == []

    def test_single_in_registry_but_doc_has_two(self):
        """注册表只有一个蓝衣人，但 doc 中场景有两个蓝衣人 → derive 层拒绝。"""
        spec = _setup_sound_presence_spec()
        # 注册表只有 1 个蓝衣人 → 候选层通过
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            time_window=(0, 96000),
        )
        bindings = pipeline.iter_batches()
        assert len(bindings) == 1

        # doc 中场景有两个蓝衣人 → 答案不唯一
        doc = _make_doc(actors=[
            _make_actor_dict("a1", top_color="蓝色"),
            _make_actor_dict("a2", top_color="蓝色"),
        ])
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is None  # 拒绝

    def test_different_colors_allowed(self):
        """场景中两个角色但颜色不同 → 答案唯一，通过。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            time_window=(0, 96000),
        )
        bindings = pipeline.iter_batches()
        assert len(bindings) == 1

        doc = _make_doc(
            actors=[
                _make_actor_dict("a1", top_color="蓝色"),
                _make_actor_dict("a2", top_color="红色"),
            ],
            sound_facts=[
                {"actor_id": "a1", "sound_asset_id": "s1",
                 "start_tick": 0, "end_tick": 48000, "start_frame": 0},
            ],
            vis_frames=[
                {"frame_index": 0, "actor_visibility": {"a1": {"visibility_state": "visible_clear"}}},
            ],
        )
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.answer_unique is True


# ═══════════════════════════════════════════════════════════════════════════════
# 可观察性验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestObservability:
    """事实可观察性检查。"""

    def test_fully_occluded_rejected(self):
        """角色在时间窗口内始终被完全遮挡 → 不可观察 → 拒绝。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            time_window=(0, 3200 * 10),  # 10 帧
        )
        bindings = pipeline.iter_batches()

        # 全部帧都是 fully_occluded
        doc = _make_doc(
            actors=[_make_actor_dict("a1")],
            sound_facts=[
                {"actor_id": "a1", "sound_asset_id": "s1",
                 "start_tick": 0, "end_tick": 48000, "start_frame": 0},
            ],
            vis_frames=[
                {"frame_index": i, "actor_visibility": {"a1": {"visibility_state": "fully_occluded"}}}
                for i in range(10)
            ],
        )
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is None

    def test_partially_visible_accepted(self):
        """至少一帧可见 → 可观察。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            time_window=(0, 3200 * 5),  # 5 帧窗口
        )
        bindings = pipeline.iter_batches()

        doc = _make_doc(
            actors=[_make_actor_dict("a1")],
            sound_facts=[
                {"actor_id": "a1", "sound_asset_id": "s1",
                 "start_tick": 0, "end_tick": 48000, "start_frame": 0},
            ],
            vis_frames=[
                {"frame_index": 0, "actor_visibility": {"a1": {"visibility_state": "fully_occluded"}}},
                {"frame_index": 1, "actor_visibility": {"a1": {"visibility_state": "visible_clear"}}},
            ],
        )
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 可见性模态
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisibilityEndToEnd:
    """可见性问题端到端。"""

    def test_mostly_clear_visibility(self):
        """大部分可见 → 答案为‘基本无遮挡’。"""
        spec = _setup_visibility_spec()
        reg = _setup_registry(1, ["a1"], ["红色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "红色"},
            time_window=(0, 3200 * 10),
        )
        bindings = pipeline.iter_batches()

        # 全部帧可见
        doc = _make_doc(
            actors=[_make_actor_dict("a1", top_color="红色")],
            vis_frames=[
                {"frame_index": i, "actor_visibility": {"a1": {"visibility_state": "visible_clear"}}}
                for i in range(10)
            ],
        )
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.answer_text == "基本无遮挡"

    def test_fully_occluded_visibility(self):
        """全部遮挡 → 答案为‘完全遮挡’。"""
        spec = _setup_visibility_spec()
        reg = _setup_registry(1, ["a1"], ["红色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "红色"},
            time_window=(0, 3200 * 10),
        )
        bindings = pipeline.iter_batches()

        doc = _make_doc(
            actors=[_make_actor_dict("a1", top_color="红色")],
            vis_frames=[
                {"frame_index": i, "actor_visibility": {"a1": {"visibility_state": "fully_occluded"}}}
                for i in range(10)
            ],
        )
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.answer_text == "完全遮挡"


# ═══════════════════════════════════════════════════════════════════════════════
# max_attempts 耗尽
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaxAttemptsExhausted:
    """超过最大尝试次数。"""

    def test_max_attempts_zero(self):
        """max_attempts=0 → iter_batches 从第一次就返回空。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            max_attempts=0,
        )
        bindings = pipeline.iter_batches()
        assert bindings == []

    def test_exhausted_after_multi_attempts(self):
        """多次调用 iter_batches 直到超过 max_attempts。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            max_attempts=3,
        )

        # 前三次有结果
        for _ in range(3):
            b = pipeline.iter_batches()
            assert len(b) == 1

        # 第四次返回空
        b = pipeline.iter_batches()
        assert b == []


# ═══════════════════════════════════════════════════════════════════════════════
# 唯一性禁用
# ═══════════════════════════════════════════════════════════════════════════════


class TestUniquenessDisabled:
    """禁用唯一性检查。"""

    def test_multiple_matches_allowed(self):
        """禁用唯一性后，多个同色角色通过候选选择。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(2, ["a1", "a2"], ["蓝色", "蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            attribute_uniqueness_required=False,
        )
        bindings = pipeline.iter_batches()
        # 两个同色角色都通过
        assert len(bindings) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 多声音候选
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultipleSoundCandidates:
    """多声音候选。"""

    def test_multiple_sounds_per_actor(self):
        """一个角色有多个声音 → 多个绑定，每个绑定对应不同的声音。"""
        reg = _setup_registry(1, ["a1"], ["蓝色"])
        sounds = FakeSoundRegistry()
        sounds.add(SoundCandidate(
            "sound_01", "human_speech", "你好世界",
            48000, bound_to_actor="a1",
        ))
        sounds.add(SoundCandidate(
            "sound_02", "human_speech", "再见世界",
            48000, bound_to_actor="a1",
        ))

        from avengine.qa.question_spec import SceneRequirement
        from avengine.qa.candidate_selector import select_candidates

        req = SceneRequirement(
            spec_id="qs1",
            target_attributes={"top_color": "蓝色"},
            required_sound_type="human_speech",
        )
        bindings = select_candidates(req, reg, sounds)
        assert len(bindings) == 2
        assert bindings[0].sound.transcript == "你好世界"  # type: ignore[union-attr]
        assert bindings[1].sound.transcript == "再见世界"  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════════
# 时间窗口边界
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimeWindowBoundary:
    """时间窗口边界场景。"""

    def test_sound_on_boundary_inside_window(self):
        """声音恰好从时间窗口起始点开始 → 在窗口内。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            time_window=(48000, 96000),
        )
        bindings = pipeline.iter_batches()

        # 声音正好从 tw_start 开始
        doc = _make_doc(
            actors=[_make_actor_dict("a1")],
            sound_facts=[
                {"actor_id": "a1", "sound_asset_id": "s1",
                 "start_tick": 48000, "end_tick": 72000, "start_frame": 15},
            ],
            vis_frames=[
                {"frame_index": 15, "actor_visibility": {"a1": {"visibility_state": "visible_clear"}}},
            ],
        )
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.answer_text == "是"

    def test_sound_before_window(self):
        """声音全部在时间窗口之前 → 答案为‘否’。"""
        spec = _setup_sound_presence_spec()
        reg = _setup_registry(1, ["a1"], ["蓝色"])

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"top_color": "蓝色"},
            time_window=(48000, 96000),
        )
        bindings = pipeline.iter_batches()

        doc = _make_doc(
            actors=[_make_actor_dict("a1")],
            sound_facts=[
                {"actor_id": "a1", "sound_asset_id": "s1",
                 "start_tick": 0, "end_tick": 24000, "start_frame": 0},
            ],
            vis_frames=[
                {"frame_index": i, "actor_visibility": {"a1": {"visibility_state": "visible_clear"}}}
                for i in range(20)
            ],
        )
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.answer_text == "否"


# ═══════════════════════════════════════════════════════════════════════════════
# QuestionSpec 不同模板形状
# ═══════════════════════════════════════════════════════════════════════════════


class TestDifferentSubjectTypes:
    """不同物种/角色类型的模板。"""

    def test_species_based_question(self):
        """基于物种的问法（使用中文属性值）。"""
        spec = QuestionSpec(
            spec_id="species_q",
            question_type="sound_presence",
            template="场景中的{species_id}是否发声？",
            answer_modality="sound_facts",
        )
        reg = FakeActorRegistry()
        reg.add(ActorCandidate(
            "dog_01", "asset_dog", "dog", 5,
            attributes={"species_id": "狗", "top_color": "棕色"},
        ))

        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=reg,
            attribute_values={"species_id": "狗"},
            time_window=(0, 96000),
        )
        bindings = pipeline.iter_batches()
        assert len(bindings) == 1

        doc = _make_doc(
            actors=[{
                "actor_id": "dog_01",
                "identity": {"species_id": "狗", "breed_id": "husky"},
                "realized_visual_attributes": {
                    "clothing": {"top_color": "棕色"},
                    "size": "large",
                },
            }],
            sound_facts=[
                {"actor_id": "dog_01", "sound_asset_id": "bark_01",
                 "start_tick": 0, "end_tick": 48000, "start_frame": 0},
            ],
            vis_frames=[
                {"frame_index": 0, "actor_visibility": {"dog_01": {"visibility_state": "visible_clear"}}},
            ],
        )
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.answer_text == "是"
        assert qa.question_text == "场景中的狗是否发声？"
