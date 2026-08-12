"""候选选择器单元测试（任务三）。"""

from __future__ import annotations

import pytest

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
from avengine.qa.question_spec import SceneRequirement


# ═══════════════════════════════════════════════════════════════════════════════
# FakeActorRegistry
# ═══════════════════════════════════════════════════════════════════════════════


class TestFakeActorRegistry:
    """FakeActorRegistry 基本操作。"""

    def test_add_and_list_all(self):
        reg = FakeActorRegistry()
        actor = ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "blue"},
        )
        reg.add(actor)
        assert len(reg) == 1
        assert reg.list_all() == [actor]

    def test_lookup_by_attributes_match(self):
        reg = FakeActorRegistry()
        reg.add(ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "blue"},
        ))
        result = reg.lookup_by_attributes({"top_color": "blue"})
        assert len(result) == 1
        assert result[0].actor_id == "a1"

    def test_lookup_by_attributes_no_match(self):
        reg = FakeActorRegistry()
        reg.add(ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "red"},
        ))
        result = reg.lookup_by_attributes({"top_color": "blue"})
        assert result == []

    def test_lookup_by_attributes_empty_attrs(self):
        """空条件应匹配所有。"""
        reg = FakeActorRegistry()
        reg.add(ActorCandidate("a1", "asset_01", "human", 10))
        result = reg.lookup_by_attributes({})
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# FakeSoundRegistry
# ═══════════════════════════════════════════════════════════════════════════════


class TestFakeSoundRegistry:
    """FakeSoundRegistry 基本操作。"""

    def test_add_and_list_all(self):
        reg = FakeSoundRegistry()
        sound = SoundCandidate(
            "s1", "human_speech", "你好",
            duration_samples=48000, bound_to_actor="a1",
        )
        reg.add(sound)
        assert len(reg) == 1
        assert reg.list_all() == [sound]

    def test_lookup_by_actor_found(self):
        reg = FakeSoundRegistry()
        reg.add(SoundCandidate(
            "s1", "human_speech", "你好", 48000, bound_to_actor="a1",
        ))
        result = reg.lookup_by_actor("a1")
        assert len(result) == 1
        assert result[0].sound_asset_id == "s1"

    def test_lookup_by_actor_not_found(self):
        reg = FakeSoundRegistry()
        reg.add(SoundCandidate(
            "s1", "human_speech", "你好", 48000, bound_to_actor="a1",
        ))
        result = reg.lookup_by_actor("a2")
        assert result == []

    def test_lookup_by_actor_and_sound_type(self):
        reg = FakeSoundRegistry()
        reg.add(SoundCandidate(
            "s1", "human_speech", "你好", 48000, bound_to_actor="a1",
        ))
        reg.add(SoundCandidate(
            "s2", "dog_bark", "汪汪", 24000, bound_to_actor="a1",
        ))
        result = reg.lookup_by_actor("a1", sound_type="dog_bark")
        assert len(result) == 1
        assert result[0].sound_asset_id == "s2"


# ═══════════════════════════════════════════════════════════════════════════════
# ActorCandidate
# ═══════════════════════════════════════════════════════════════════════════════


class TestActorCandidate:
    """ActorCandidate 行为。"""

    def test_matches_exact(self):
        actor = ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "blue", "species_id": "human"},
        )
        assert actor.matches({"top_color": "blue"})

    def test_matches_wrong_value(self):
        actor = ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "red"},
        )
        assert not actor.matches({"top_color": "blue"})

    def test_matches_missing_key(self):
        actor = ActorCandidate("a1", "asset_01", "human", 10)
        assert not actor.matches({"top_color": "blue"})

    def test_frozen(self):
        actor = ActorCandidate("a1", "asset_01", "human", 10)
        with pytest.raises(Exception):
            actor.actor_id = "changed"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# select_candidates
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelectCandidates:
    """select_candidates 核心逻辑。"""

    def _make_req(self, **overrides) -> SceneRequirement:
        defaults: dict = {
            "spec_id": "qs1",
            "target_attributes": {"top_color": "blue"},
        }
        defaults.update(overrides)
        return SceneRequirement(**defaults)

    def test_basic_selection(self):
        actors = FakeActorRegistry()
        actors.add(ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "blue"},
        ))
        req = self._make_req()
        result = select_candidates(req, actors)
        assert len(result) == 1
        assert result[0].actor.actor_id == "a1"
        assert result[0].attribute_values == {"top_color": "blue"}

    def test_no_matching_actor(self):
        actors = FakeActorRegistry()
        actors.add(ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "red"},
        ))
        req = self._make_req()
        result = select_candidates(req, actors)
        assert result == []

    def test_uniqueness_rejects_duplicates(self):
        """多个同属性角色 → 唯一性检查失败，返回空。"""
        actors = FakeActorRegistry()
        actors.add(ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "blue"},
        ))
        actors.add(ActorCandidate(
            "a2", "asset_02", "human", 11,
            attributes={"top_color": "blue"},
        ))
        req = self._make_req()
        result = select_candidates(req, actors)
        assert result == []

    def test_uniqueness_disabled_allows_duplicates(self):
        """禁用唯一性检查后允许多个匹配。"""
        actors = FakeActorRegistry()
        actors.add(ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "blue"},
        ))
        actors.add(ActorCandidate(
            "a2", "asset_02", "human", 11,
            attributes={"top_color": "blue"},
        ))
        req = self._make_req(attribute_uniqueness_required=False)
        result = select_candidates(req, actors)
        assert len(result) == 2

    def test_with_sound_matching(self):
        actors = FakeActorRegistry()
        actors.add(ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "blue"},
        ))
        sounds = FakeSoundRegistry()
        sounds.add(SoundCandidate(
            "s1", "human_speech", "你好", 48000, bound_to_actor="a1",
        ))
        req = self._make_req(required_sound_type="human_speech")
        result = select_candidates(req, actors, sounds)
        assert len(result) == 1
        assert result[0].sound is not None
        assert result[0].sound.sound_asset_id == "s1"  # type: ignore[union-attr]

    def test_sound_not_found_returns_binding_without_sound(self):
        """无匹配声音时仍返回仅含角色的绑定。"""
        actors = FakeActorRegistry()
        actors.add(ActorCandidate(
            "a1", "asset_01", "human", 10,
            attributes={"top_color": "blue"},
        ))
        sounds = FakeSoundRegistry()
        req = self._make_req()
        result = select_candidates(req, actors, sounds)
        assert len(result) == 1
        assert result[0].sound is None


# ═══════════════════════════════════════════════════════════════════════════════
# 协议校验
# ═══════════════════════════════════════════════════════════════════════════════


class TestProtocolCompliance:
    """FakeRegistry 满足注册表协议。"""

    def test_fake_actor_registry_is_actor_registry(self):
        reg = FakeActorRegistry()
        assert isinstance(reg, ActorRegistry)

    def test_fake_sound_registry_is_sound_registry(self):
        reg = FakeSoundRegistry()
        assert isinstance(reg, SoundRegistry)
