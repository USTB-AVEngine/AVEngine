"""QuestionPipeline 管线编排单元测试（任务三）。"""

from __future__ import annotations

import pytest

from avengine.qa.candidate_selector import (
    ActorCandidate,
    AssetBinding,
    FakeActorRegistry,
    FakeSoundRegistry,
    SoundCandidate,
)
from avengine.qa.question_pipeline import QuestionPipeline
from avengine.qa.question_spec import QuestionSpec


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _make_spec(**overrides) -> QuestionSpec:
    """创建带默认值的 sound_presence QuestionSpec。

    默认使用无 time_window 变量的模板，避免测试中意外缺少绑定。
    """
    defaults = {
        "spec_id": "qs_001",
        "question_type": "sound_presence",
        "template": "穿{top_color}上衣的人是否发声？",
        "answer_modality": "sound_facts",
    }
    defaults.update(overrides)
    return QuestionSpec(**defaults)


def _make_spec_tw(**overrides) -> QuestionSpec:
    """创建带 time_window 变量的 QuestionSpec（用于测试时间窗口）。"""
    defaults = {
        "spec_id": "qs_tw",
        "question_type": "sound_presence",
        "template": "穿{top_color}上衣的人是否在{time_window}发声？",
        "answer_modality": "sound_facts",
    }
    defaults.update(overrides)
    return QuestionSpec(**defaults)


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


def _setup_single_actor_registry(actor_id="a1", top_color="蓝色"):
    """创建一个包含单角色的 FakeActorRegistry（使用中文属性值）。"""
    reg = FakeActorRegistry()
    reg.add(ActorCandidate(
        actor_id=actor_id,
        entity_asset_id=f"asset_{actor_id}",
        species_id="human",
        semantic_id=10,
        attributes={"top_color": top_color, "species_id": "human"},
    ))
    return reg


# ═══════════════════════════════════════════════════════════════════════════════
# QuestionPipeline 构造
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelineConstruction:
    """管线构造与基本属性。"""

    def test_default_construction(self):
        spec = _make_spec()
        pipeline = QuestionPipeline(spec=spec)
        assert pipeline.spec == spec
        assert pipeline.max_attempts == 10
        assert pipeline.attribute_uniqueness_required is True
        assert pipeline.attempt == 0

    def test_custom_max_attempts(self):
        spec = _make_spec()
        pipeline = QuestionPipeline(spec=spec, max_attempts=3)
        assert pipeline.max_attempts == 3


# ═══════════════════════════════════════════════════════════════════════════════
# iter_batches
# ═══════════════════════════════════════════════════════════════════════════════


class TestIterBatches:
    """iter_batches 候选枚举。"""

    def test_returns_bindings(self):
        spec = _make_spec()
        actors = _setup_single_actor_registry()
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=actors,
            attribute_values={"top_color": "蓝色"},
        )
        bindings = pipeline.iter_batches()
        assert len(bindings) == 1
        assert bindings[0].actor.actor_id == "a1"
        assert pipeline.attempt == 1

    def test_empty_when_no_match(self):
        spec = _make_spec()
        actors = FakeActorRegistry()  # 空注册表
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=actors,
            attribute_values={"top_color": "蓝色"},
        )
        bindings = pipeline.iter_batches()
        assert bindings == []

    def test_max_attempts_exceeded(self):
        spec = _make_spec()
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=FakeActorRegistry(),
            max_attempts=0,
        )
        bindings = pipeline.iter_batches()
        assert bindings == []

    def test_increments_attempt_counter(self):
        spec = _make_spec()
        actors = _setup_single_actor_registry()
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=actors,
            attribute_values={"top_color": "蓝色"},
        )
        assert pipeline.attempt == 0
        pipeline.iter_batches()
        assert pipeline.attempt == 1
        pipeline.iter_batches()
        assert pipeline.attempt == 2


# ═══════════════════════════════════════════════════════════════════════════════
# try_derive_from_doc — 成功
# ═══════════════════════════════════════════════════════════════════════════════


class TestTryDeriveSuccess:
    """try_derive_from_doc 成功路径。"""

    def test_successful_qa_pair(self):
        """正常流程：角色唯一 + 声音存在 + 可观察 → 返回 QAPair。"""
        spec = _make_spec()
        actors = _setup_single_actor_registry()
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=actors,
            attribute_values={"top_color": "蓝色"},
        )

        bindings = pipeline.iter_batches()
        assert len(bindings) == 1

        doc = _make_doc(
            actors=[_make_actor_dict("a1")],
            sound_facts=[
                {"actor_id": "a1", "sound_asset_id": "speech_01",
                 "start_tick": 10_000, "end_tick": 20_000, "start_frame": 0},
            ],
            vis_frames=[
                {"frame_index": i, "actor_visibility": {"a1": {"visibility_state": "visible_clear"}}}
                for i in range(5)
            ],
        )

        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.question_type == "sound_presence"
        assert qa.answer_text == "是"
        assert qa.answer_unique is True
        assert qa.fact_observable is True
        assert qa.question_text == "穿蓝色上衣的人是否发声？"

    def test_answer_source_sound_facts(self):
        """验证 answer_source 中 sound_facts 路径。"""
        spec = _make_spec()
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=_setup_single_actor_registry(),
            attribute_values={"top_color": "蓝色"},
        )
        bindings = pipeline.iter_batches()
        doc = _make_doc(sound_facts=[
            {"actor_id": "a1", "sound_asset_id": "speech_01",
             "start_tick": 0, "end_tick": 48000, "start_frame": 0},
        ])
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert qa.answer_source is not None
        assert "sound_facts" in qa.answer_source["fact_path"]


# ═══════════════════════════════════════════════════════════════════════════════
# try_derive_from_doc — 失败
# ═══════════════════════════════════════════════════════════════════════════════


class TestTryDeriveFailure:
    """try_derive_from_doc 失败路径（返回 None）。"""

    def test_returns_none_when_not_unique(self):
        """两个同色角色 → 答案不唯一 → 返回 None。"""
        spec = _make_spec()
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=_setup_single_actor_registry(),
            attribute_values={"top_color": "蓝色"},
        )
        bindings = pipeline.iter_batches()
        doc = _make_doc(actors=[
            _make_actor_dict("a1", top_color="蓝色"),
            _make_actor_dict("a2", top_color="蓝色"),
        ])
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is None

    def test_returns_none_when_not_observable(self):
        """角色完全被遮挡 → 不可观察 → 返回 None。"""
        spec = _make_spec()
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=_setup_single_actor_registry(),
            attribute_values={"top_color": "蓝色"},
            time_window=(0, 3200 * 10),
        )
        bindings = pipeline.iter_batches()
        doc = _make_doc(
            actors=[_make_actor_dict("a1")],
            sound_facts=[
                {"actor_id": "a1", "sound_asset_id": "speech_01",
                 "start_tick": 0, "end_tick": 48000, "start_frame": 0},
            ],
            vis_frames=[
                {"frame_index": i, "actor_visibility": {"a1": {"visibility_state": "fully_occluded"}}}
                for i in range(10)
            ],
        )
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is None


# ═══════════════════════════════════════════════════════════════════════════════
# 模板实例化（使用含 time_window 变量的模板）
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuestionText:
    """_question_text 模板实例化（通过 try_derive 间接验证）。"""

    def test_time_window_from_pipeline(self):
        """时间窗口取自 pipeline.time_window。"""
        spec = _make_spec_tw()
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=_setup_single_actor_registry(),
            attribute_values={"top_color": "蓝色"},
            time_window=(0, 144000),
        )
        bindings = pipeline.iter_batches()
        doc = _make_doc(sound_facts=[
            {"actor_id": "a1", "sound_asset_id": "s1",
             "start_tick": 0, "end_tick": 48000, "start_frame": 0},
        ])
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert "0.0-3.0秒" in qa.question_text

    def test_time_window_from_spec_fallback(self):
        """无 pipeline.time_window 时回退到 spec.time_window。"""
        spec = _make_spec_tw(time_window=(96000, 192000))
        pipeline = QuestionPipeline(
            spec=spec,
            actor_registry=_setup_single_actor_registry(),
            attribute_values={"top_color": "蓝色"},
        )
        bindings = pipeline.iter_batches()
        doc = _make_doc(sound_facts=[
            {"actor_id": "a1", "sound_asset_id": "s1",
             "start_tick": 96000, "end_tick": 144000, "start_frame": 30},
        ])
        qa = pipeline.try_derive_from_doc(doc, bindings[0])
        assert qa is not None
        assert "2.0-4.0秒" in qa.question_text


# ═══════════════════════════════════════════════════════════════════════════════
# 重采样场景（集成模拟）
# ═══════════════════════════════════════════════════════════════════════════════


class TestResampling:
    """重采样：失败后尝试下一组绑定。"""

    def test_retry_after_unique_failure(self):
        """第一组绑定因答案不唯一被候选选择器拒绝，第二组成功。"""
        spec = _make_spec()

        # 第一组注册表：两个同色角色 → select_candidates 拒绝
        round_actors_dup = FakeActorRegistry()
        round_actors_dup.add(ActorCandidate(
            "a1", "asset_a1", "human", 10,
            attributes={"top_color": "蓝色"},
        ))
        round_actors_dup.add(ActorCandidate(
            "a2", "asset_a2", "human", 11,
            attributes={"top_color": "蓝色"},
        ))

        # 第二组注册表：只有一个蓝色角色
        round_actors_ok = FakeActorRegistry()
        round_actors_ok.add(ActorCandidate(
            "a3", "asset_a3", "human", 12,
            attributes={"top_color": "蓝色"},
        ))

        # 第一次尝试 → 唯一性失败，返回空
        p1 = QuestionPipeline(
            spec=spec,
            actor_registry=round_actors_dup,
            attribute_values={"top_color": "蓝色"},
        )
        bindings1 = p1.iter_batches()
        assert bindings1 == []

        # 第二次尝试 → 成功
        p2 = QuestionPipeline(
            spec=spec,
            actor_registry=round_actors_ok,
            attribute_values={"top_color": "蓝色"},
        )
        bindings2 = p2.iter_batches()
        assert len(bindings2) == 1

        doc = _make_doc(
            actors=[_make_actor_dict("a3")],
            sound_facts=[
                {"actor_id": "a3", "sound_asset_id": "s1",
                 "start_tick": 0, "end_tick": 48000, "start_frame": 0},
            ],
        )
        qa = p2.try_derive_from_doc(doc, bindings2[0])
        assert qa is not None
        assert qa.answer_text == "是"

    def test_multiple_sound_candidates_per_actor(self):
        """一个角色有多个声音候选 → 生成多个绑定。"""
        actors = FakeActorRegistry()
        actors.add(ActorCandidate(
            "a1", "asset_a1", "human", 10,
            attributes={"top_color": "蓝色"},
        ))
        sounds = FakeSoundRegistry()
        sounds.add(SoundCandidate(
            "speech_01", "human_speech", "你好", 48000, bound_to_actor="a1",
        ))
        sounds.add(SoundCandidate(
            "speech_02", "human_speech", "再见", 48000, bound_to_actor="a1",
        ))

        from avengine.qa.question_spec import SceneRequirement
        from avengine.qa.candidate_selector import select_candidates

        req = SceneRequirement(
            spec_id="qs1",
            target_attributes={"top_color": "蓝色"},
            required_sound_type="human_speech",
        )
        bindings = select_candidates(req, actors, sounds)
        assert len(bindings) == 2
        assert bindings[0].sound.sound_asset_id == "speech_01"  # type: ignore[union-attr]
        assert bindings[1].sound.sound_asset_id == "speech_02"  # type: ignore[union-attr]
