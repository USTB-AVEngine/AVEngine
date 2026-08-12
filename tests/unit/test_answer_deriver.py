"""答案推导单元测试（任务三）。"""

from __future__ import annotations

import pytest

from avengine.qa.answer_deriver import (
    check_answer_unique,
    check_fact_observable,
    derive_answer,
)
from avengine.qa.candidate_selector import (
    ActorCandidate,
    AssetBinding,
    SoundCandidate,
)
from avengine.qa.question_spec import QuestionSpec


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _make_spec(**overrides) -> QuestionSpec:
    """创建带默认值的 QuestionSpec。"""
    defaults = {
        "spec_id": "qs_001",
        "question_type": "sound_presence",
        "template": "是否发声？",
        "answer_modality": "sound_facts",
    }
    defaults.update(overrides)
    return QuestionSpec(**defaults)


def _make_actor(**overrides) -> dict:
    """创建符合 Episode actor dict 结构的角色。

    常用覆盖: actor_id, top_color, species_id, size。
    注意: top_color 直接覆盖 realized_visual_attributes.clothing.top_color。
    """
    top_color = overrides.pop("top_color", "blue")
    species_id = overrides.pop("species_id", "human")
    breed_id = overrides.pop("breed_id", "human_default")
    size = overrides.pop("size", "medium")

    data = {
        "actor_id": "a1",
        "identity": {"species_id": species_id, "breed_id": breed_id},
        "realized_visual_attributes": {
            "clothing": {"top_color": top_color},
            "size": size,
        },
    }
    data.update(overrides)
    return data


def _make_doc(**overrides) -> dict:
    """创建符合 Episode.build() 结构的文档。"""
    doc = {
        "assets_used": {
            "actors": [_make_actor()],
        },
        "facts": {
            "sound_facts": [],
            "visibility_facts": {"per_frame": []},
            "motion_facts": {"per_frame": []},
        },
    }
    doc.update(overrides)
    return doc


def _make_binding(actor_id="a1", top_color="blue", with_sound=True) -> AssetBinding:
    """创建测试用 AssetBinding。"""
    actor = ActorCandidate(
        actor_id=actor_id,
        entity_asset_id=f"asset_{actor_id}",
        species_id="human",
        semantic_id=10,
        attributes={"top_color": top_color, "species_id": "human"},
    )
    sound = None
    if with_sound:
        sound = SoundCandidate(
            sound_asset_id="speech_01",
            semantic_sound_class="human_speech",
            transcript="你好",
            duration_samples=48000,
            bound_to_actor=actor_id,
        )
    return AssetBinding(
        actor=actor,
        sound=sound,
        attribute_values={"top_color": top_color},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# derive_answer — 声音
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeriveSoundAnswer:
    """从 sound_facts 推导答案。"""

    def test_sound_present_yes(self):
        """角色有声音事件 → '是'。"""
        doc = _make_doc(facts={
            "sound_facts": [
                {"actor_id": "a1", "sound_asset_id": "speech_01",
                 "start_tick": 0, "end_tick": 48000, "start_frame": 0},
            ],
            "visibility_facts": {"per_frame": []},
            "motion_facts": {"per_frame": []},
        })
        spec = _make_spec()
        binding = _make_binding()
        answer, unique, observable = derive_answer(doc, spec, binding)
        assert answer == "是"
        assert unique is True
        assert observable is True

    def test_sound_absent_no(self):
        """角色无声音事件 → '否'。"""
        doc = _make_doc()
        spec = _make_spec()
        binding = _make_binding()
        answer, _unique, _observable = derive_answer(doc, spec, binding)
        assert answer == "否"

    def test_sound_outside_time_window(self):
        """声音事件不在时间窗口内 → '否'。"""
        doc = _make_doc(facts={
            "sound_facts": [
                {"actor_id": "a1", "sound_asset_id": "speech_01",
                 "start_tick": 100_000, "end_tick": 150_000, "start_frame": 100},
            ],
            "visibility_facts": {"per_frame": []},
            "motion_facts": {"per_frame": []},
        })
        spec = _make_spec(time_window=(0, 48000))
        binding = _make_binding()
        answer, _unique, _observable = derive_answer(doc, spec, binding)
        assert answer == "否"

    def test_sound_inside_time_window(self):
        """声音事件在时间窗口内 → '是'。"""
        doc = _make_doc(facts={
            "sound_facts": [
                {"actor_id": "a1", "sound_asset_id": "speech_01",
                 "start_tick": 10_000, "end_tick": 20_000, "start_frame": 0},
            ],
            "visibility_facts": {"per_frame": []},
            "motion_facts": {"per_frame": []},
        })
        spec = _make_spec(time_window=(0, 48000))
        binding = _make_binding()
        answer, _unique, _observable = derive_answer(doc, spec, binding)
        assert answer == "是"


# ═══════════════════════════════════════════════════════════════════════════════
# derive_answer — 可见性
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeriveVisibilityAnswer:
    """从 visibility_facts 推导答案。"""

    def test_mostly_clear(self):
        """大部分帧清晰可见 → '基本无遮挡'。"""
        frames = []
        for i in range(10):
            frames.append({
                "frame_index": i,
                "actor_visibility": {
                    "a1": {"visibility_state": "visible_clear"},
                },
            })
        doc = _make_doc(facts={
            "sound_facts": [],
            "visibility_facts": {"per_frame": frames},
            "motion_facts": {"per_frame": []},
        })
        spec = _make_spec(answer_modality="visibility_facts")
        binding = _make_binding()
        answer, _unique, _observable = derive_answer(doc, spec, binding)
        assert answer == "基本无遮挡"

    def test_partial_occlusion(self):
        """部分遮挡 → '部分遮挡'。"""
        frames = []
        # 5 帧清晰，5 帧被遮挡
        for i in range(5):
            frames.append({
                "frame_index": i,
                "actor_visibility": {
                    "a1": {"visibility_state": "visible_clear"},
                },
            })
        for i in range(5, 10):
            frames.append({
                "frame_index": i,
                "actor_visibility": {
                    "a1": {"visibility_state": "fully_occluded"},
                },
            })
        doc = _make_doc(facts={
            "sound_facts": [],
            "visibility_facts": {"per_frame": frames},
            "motion_facts": {"per_frame": []},
        })
        spec = _make_spec(answer_modality="visibility_facts")
        binding = _make_binding()
        answer, _unique, _observable = derive_answer(doc, spec, binding)
        assert answer == "部分遮挡"

    def test_fully_occluded(self):
        """全部遮挡 → '完全遮挡'。"""
        frames = [
            {"frame_index": i, "actor_visibility": {"a1": {"visibility_state": "fully_occluded"}}}
            for i in range(10)
        ]
        doc = _make_doc(facts={
            "sound_facts": [],
            "visibility_facts": {"per_frame": frames},
            "motion_facts": {"per_frame": []},
        })
        spec = _make_spec(answer_modality="visibility_facts")
        binding = _make_binding()
        answer, _unique, _observable = derive_answer(doc, spec, binding)
        assert answer == "完全遮挡"

    def test_no_visibility_data(self):
        """无可见性数据 → '不可见'。"""
        doc = _make_doc()
        spec = _make_spec(answer_modality="visibility_facts")
        binding = _make_binding()
        answer, _unique, _observable = derive_answer(doc, spec, binding)
        assert answer == "不可见"


# ═══════════════════════════════════════════════════════════════════════════════
# derive_answer — 运动
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeriveMotionAnswer:
    """从 motion_facts 推导答案。"""

    def test_motion_walk(self):
        doc = _make_doc(facts={
            "sound_facts": [],
            "visibility_facts": {"per_frame": []},
            "motion_facts": {
                "per_frame": [
                    {"actor_states": {"a1": "walk"}},
                ],
            },
        })
        spec = _make_spec(answer_modality="motion_facts")
        binding = _make_binding()
        answer, _unique, _observable = derive_answer(doc, spec, binding)
        assert answer == "walk"

    def test_motion_unknown(self):
        doc = _make_doc()
        spec = _make_spec(answer_modality="motion_facts")
        binding = _make_binding()
        answer, _unique, _observable = derive_answer(doc, spec, binding)
        assert answer == "未知"


# ═══════════════════════════════════════════════════════════════════════════════
# check_answer_unique
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckAnswerUnique:
    """答案唯一性检查。"""

    def test_unique_single_actor(self):
        """场景只有一个匹配角色 → 答案唯一。"""
        doc = _make_doc()
        spec = _make_spec()
        binding = _make_binding()
        assert check_answer_unique(doc, spec, binding) is True

    def test_not_unique_two_same_color(self):
        """两个角色 top_color 相同 → 答案不唯一。"""
        doc = _make_doc(assets_used={
            "actors": [
                _make_actor(actor_id="a1", top_color="blue"),
                _make_actor(actor_id="a2", top_color="blue"),
            ],
        })
        spec = _make_spec()
        binding = _make_binding()
        assert check_answer_unique(doc, spec, binding) is False

    def test_unique_different_colors(self):
        """两个角色但 top_color 不同 → 答案唯一。"""
        doc = _make_doc(assets_used={
            "actors": [
                _make_actor(actor_id="a1", top_color="blue"),
                _make_actor(actor_id="a2", top_color="red"),
            ],
        })
        spec = _make_spec()
        binding = _make_binding()
        assert check_answer_unique(doc, spec, binding) is True

    def test_empty_actors(self):
        """无角色列表 → 答案唯一（保守通过）。"""
        doc = _make_doc(assets_used={"actors": []})
        spec = _make_spec()
        binding = _make_binding()
        assert check_answer_unique(doc, spec, binding) is True


# ═══════════════════════════════════════════════════════════════════════════════
# check_fact_observable — 声音
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckFactObservableSound:
    """声音模态的可观察性检查。"""

    def test_observable_when_visible(self):
        """角色在时间窗口内可见 → 可观察。"""
        doc = _make_doc(facts={
            "sound_facts": [],
            "visibility_facts": {
                "per_frame": [
                    {"frame_index": 5, "actor_visibility": {"a1": {"visibility_state": "visible_clear"}}},
                ],
            },
            "motion_facts": {"per_frame": []},
        })
        spec = _make_spec(time_window=(0, 48000))
        binding = _make_binding()
        assert check_fact_observable(doc, spec, binding) is True

    def test_not_observable_when_fully_occluded(self):
        """角色在时间窗口内始终被完全遮挡 → 不可观察。"""
        frames = [
            {"frame_index": i, "actor_visibility": {"a1": {"visibility_state": "fully_occluded"}}}
            for i in range(10)
        ]
        doc = _make_doc(facts={
            "sound_facts": [],
            "visibility_facts": {"per_frame": frames},
            "motion_facts": {"per_frame": []},
        })
        # 时间窗口覆盖所有帧，TICKS_PER_FRAME=3200
        spec = _make_spec(time_window=(0, 3200 * 10))
        binding = _make_binding()
        assert check_fact_observable(doc, spec, binding) is False

    def test_observable_with_no_vis_data(self):
        """无可见性数据 → 默认可观察。"""
        doc = _make_doc()
        spec = _make_spec(time_window=(0, 48000))
        binding = _make_binding()
        assert check_fact_observable(doc, spec, binding) is True

    def test_observable_partial_occlusion(self):
        """至少一帧可见 → 可观察。"""
        frames = [
            {"frame_index": 0, "actor_visibility": {"a1": {"visibility_state": "fully_occluded"}}},
            {"frame_index": 1, "actor_visibility": {"a1": {"visibility_state": "visible_occluded"}}},
        ]
        doc = _make_doc(facts={
            "sound_facts": [],
            "visibility_facts": {"per_frame": frames},
            "motion_facts": {"per_frame": []},
        })
        spec = _make_spec(time_window=(0, 3200 * 5))
        binding = _make_binding()
        assert check_fact_observable(doc, spec, binding) is True


# ═══════════════════════════════════════════════════════════════════════════════
# check_fact_observable — 可见性
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckFactObservableVisibility:
    """可见性模态始终可观察。"""

    def test_visibility_always_observable(self):
        doc = _make_doc()
        spec = _make_spec(answer_modality="visibility_facts")
        binding = _make_binding()
        assert check_fact_observable(doc, spec, binding) is True


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数：_extract_actor_attribute
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractActorAttribute:
    """_extract_actor_attribute 属性提取。"""

    def test_extract_top_color(self):
        from avengine.qa.answer_deriver import _extract_actor_attribute
        actor = _make_actor()
        assert _extract_actor_attribute(actor, "top_color") == "blue"

    def test_extract_species_id(self):
        from avengine.qa.answer_deriver import _extract_actor_attribute
        actor = _make_actor()
        assert _extract_actor_attribute(actor, "species_id") == "human"

    def test_extract_breed_id(self):
        from avengine.qa.answer_deriver import _extract_actor_attribute
        actor = _make_actor()
        assert _extract_actor_attribute(actor, "breed_id") == "human_default"

    def test_extract_size(self):
        from avengine.qa.answer_deriver import _extract_actor_attribute
        actor = _make_actor()
        assert _extract_actor_attribute(actor, "size") == "medium"

    def test_extract_missing_attribute(self):
        from avengine.qa.answer_deriver import _extract_actor_attribute
        actor = _make_actor()
        assert _extract_actor_attribute(actor, "life_stage") is None
