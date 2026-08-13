"""任务 3.5 — 五个可视化 Canary 验收测试。

每个 canary 验证一个关键可见性场景，自动验证：
- Episode 文档结构正确
- 像素可见性分析结果与预期一致
- 至少一个 QAPair 可推导且 answer_unique=True, fact_observable=True
- 无像素数据时禁止产生画外/遮挡类问题

Canary 场景:
  C1: 目标完全可见（visible_clear）
  C2: 目标被家具部分遮挡（visible_occluded + furniture occluder）
  C3: 目标完全遮挡（fully_occluded）
  C4: 目标完全在画外（out_of_view → enter_frustum）
  C5: 相机/听者运动，遮挡→重现（occlusion_start → reappear）
"""

from __future__ import annotations

import numpy as np
import pytest

from avengine.m5.timeline import TICKS_PER_FRAME
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
from avengine.qa.pixel_visibility import analyze_all_frames, detect_occluders
from avengine.qa.question_spec import QuestionSpec


# ═══════════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════════

H, W = 64, 64  # 小尺寸语义图
N_FRAMES = 5
TARGET_SEMANTIC_ID = 10
FURNITURE_SEMANTIC_ID = 50
OTHER_ACTOR_SEMANTIC_ID = 20
TARGET_ACTOR_ID = "dog_01"

# 角色占据中央 32×32 像素
TARGET_SLICE = (slice(16, 48), slice(16, 48))
# 遮挡区域：覆盖目标上半部分，产生部分遮挡效果
OCCLUDER_PARTIAL = (slice(16, 32), slice(0, 64))
# 遮挡区域：几乎覆盖全部目标
OCCLUDER_FULL = (slice(0, 64), slice(0, 64))


# ═══════════════════════════════════════════════════════════════════════════════
# 合成语义数组生成器
# ═══════════════════════════════════════════════════════════════════════════════


def _make_target_only_semantic() -> np.ndarray:
    """目标专用语义图：只有目标区域有非零值。"""
    arr = np.zeros((H, W), dtype=np.int64)
    arr[TARGET_SLICE] = TARGET_SEMANTIC_ID
    return arr


def _make_normal_semantic(
    occluder_slices: list[tuple[slice, slice, int]] | None = None,
) -> np.ndarray:
    """正常语义图：目标 + 可选遮挡物。

    Args:
        occluder_slices: 遮挡物列表，每个元素为 (y_slice, x_slice, semantic_id)。
    """
    arr = np.zeros((H, W), dtype=np.int64)
    arr[TARGET_SLICE] = TARGET_SEMANTIC_ID
    if occluder_slices:
        for ys, xs, sem_id in occluder_slices:
            arr[ys, xs] = sem_id
    return arr


def _make_in_frustum_list(
    n_frames: int,
    in_frustum: bool = True,
    exit_frames: set[int] | None = None,
) -> list[bool]:
    """生成每帧是否在视锥内的列表。"""
    exit_frames = exit_frames or set()
    return [(i not in exit_frames) and in_frustum for i in range(n_frames)]


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助：构建 Episode 文档
# ═══════════════════════════════════════════════════════════════════════════════


def _make_canary_doc(
    normal_semantics: list[np.ndarray],
    target_only_semantics: list[np.ndarray],
    in_frustums: list[bool],
    *,
    sound_facts: list[dict] | None = None,
    spatial_frames: list[dict] | None = None,
    motion_frames: list[dict] | None = None,
    actors: list[dict] | None = None,
    events: list[dict] | None = None,
) -> dict:
    """从合成语义数组构建 Episode 文档。"""
    # 运行像素可见性分析
    vis_records = analyze_all_frames(
        normal_semantics,
        target_only_semantics,
        target_semantic_id=TARGET_SEMANTIC_ID,
        actor_id=TARGET_ACTOR_ID,
        in_frustums=in_frustums,
        actor_semantic_map={OTHER_ACTOR_SEMANTIC_ID: "other_actor"},
        furniture_semantic_map={FURNITURE_SEMANTIC_ID: ("table", "桌子")},
    )

    # 转换为 Episode 可用的帧格式（vis_records 已包含 frame_index + actor_visibility）
    vis_frames: list[dict] = list(vis_records)

    doc: dict = {
        "schema": "avengine_qa_episode_v1",
        "episode_id": "canary_test",
        "assets_used": {
            "actors": actors or [
                {
                    "actor_id": TARGET_ACTOR_ID,
                    "entity_asset_id": "dog_asset",
                    "identity": {"species_id": "狗", "breed_id": "比格犬"},
                    "realized_visual_attributes": {
                        "clothing": {"top_color": "棕色"},
                        "size": "medium",
                        "body_build": "average",
                        "life_stage": "adult",
                    },
                    "semantic_id": TARGET_SEMANTIC_ID,
                },
            ],
            "sounds": [],
        },
        "scene": {},
        "timeline": {},
        "facts": {
            "sound_facts": sound_facts or [
                {
                    "event_id": "evt_01",
                    "actor_id": TARGET_ACTOR_ID,
                    "sound_asset_id": "bark_01",
                    "start_tick": TICKS_PER_FRAME,  # 帧 1
                    "end_tick": 3 * TICKS_PER_FRAME,  # 帧 3
                    "start_frame": 1,
                    "end_frame": 3,
                    "transcript": "",
                },
            ],
            "spatial_facts": {"per_frame": spatial_frames or [
                {"frame_index": i, "actors": {
                    TARGET_ACTOR_ID: {"listener_relative": {"azimuth_deg": 0.0}},
                }} for i in range(N_FRAMES)
            ]},
            "motion_facts": {"per_frame": motion_frames or [
                {"frame_index": i, "actor_states": {TARGET_ACTOR_ID: "idle"}}
                for i in range(N_FRAMES)
            ]},
            "visibility_facts": {"per_frame": vis_frames},
            "events": events or [],
        },
        "qa_pairs": [],
        "sidecars": [],
        "provenance": {},
        "episode_content_sha256": "deadbeef",
    }
    return doc


def _make_binding() -> AssetBinding:
    """构建测试用的 AssetBinding。"""
    actor = ActorCandidate(
        actor_id=TARGET_ACTOR_ID,
        entity_asset_id="dog_asset",
        species_id="狗",
        semantic_id=TARGET_SEMANTIC_ID,
        attributes={"species_id": "狗", "breed_id": "比格犬", "top_color": "棕色"},
    )
    sound = SoundCandidate(
        sound_asset_id="bark_01",
        semantic_sound_class="dog_bark",
        transcript="",
        duration_samples=2 * TICKS_PER_FRAME,
        bound_to_actor=TARGET_ACTOR_ID,
    )
    return AssetBinding(
        actor=actor,
        sound=sound,
        attribute_values={"species_id": "狗", "breed_id": "比格犬"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Canary 1: 目标完全可见
# ═══════════════════════════════════════════════════════════════════════════════


class TestCanary1FullyVisible:
    """C1: 所有帧目标完全可见，无遮挡。"""

    @pytest.fixture(scope="class")
    def doc(self):
        normal = [_make_normal_semantic() for _ in range(N_FRAMES)]
        target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
        in_frustum = _make_in_frustum_list(N_FRAMES)
        return _make_canary_doc(normal, target_only, in_frustum)

    def test_all_frames_visible_clear(self, doc):
        vis_frames = doc["facts"]["visibility_facts"]["per_frame"]
        for frame in vis_frames:
            state = frame["actor_visibility"][TARGET_ACTOR_ID]["visibility_state"]
            assert state == "visible_clear", f"帧 {frame['frame_index']}: 预期 visible_clear，实际 {state}"

    def test_no_occluders(self, doc):
        vis_frames = doc["facts"]["visibility_facts"]["per_frame"]
        for frame in vis_frames:
            occluders = frame["actor_visibility"][TARGET_ACTOR_ID].get("occluders", [])
            assert len(occluders) == 0

    def test_sound_presence_question_works(self, doc):
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="canary1_sound",
            question_type="sound_presence",
            template="狗是否发声？",
            answer_modality="sound_facts",
        )
        answer, unique, obs = derive_answer(doc, spec, binding)
        assert answer == "是"
        assert unique is True
        assert obs is True

    def test_sound_visibility_question_works(self, doc):
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="canary1_vis",
            question_type="sound_visibility",
            template="狗叫时是否可见？",
            answer_modality="sound_visibility",
        )
        answer, unique, obs = derive_answer(doc, spec, binding)
        assert answer == "清晰可见"
        assert unique is True
        assert obs is True

    def test_visibility_fact_observable(self, doc):
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="canary1_vis",
            question_type="sound_visibility",
            template="测试",
            answer_modality="sound_visibility",
        )
        assert check_fact_observable(doc, spec, binding) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Canary 2: 家具部分遮挡
# ═══════════════════════════════════════════════════════════════════════════════


class TestCanary2PartialOcclusionFurniture:
    """C2: 目标被家具部分遮挡 5 帧都有上方家具覆盖。"""

    @pytest.fixture(scope="class")
    def doc(self):
        # 5 帧都有家具在目标上方遮挡
        normal = [
            _make_normal_semantic(occluder_slices=[(*OCCLUDER_PARTIAL, FURNITURE_SEMANTIC_ID)])
            for _ in range(N_FRAMES)
        ]
        target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
        in_frustum = _make_in_frustum_list(N_FRAMES)
        return _make_canary_doc(normal, target_only, in_frustum)

    def test_has_occlusion(self, doc):
        vis_frames = doc["facts"]["visibility_facts"]["per_frame"]
        occlusion_count = sum(
            1 for f in vis_frames
            if f["actor_visibility"][TARGET_ACTOR_ID]["visibility_state"] == "visible_occluded"
        )
        assert occlusion_count >= 1, "应至少有一帧部分遮挡"

    def test_occluder_is_furniture(self, doc):
        vis_frames = doc["facts"]["visibility_facts"]["per_frame"]
        for f in vis_frames:
            occluders = f["actor_visibility"][TARGET_ACTOR_ID].get("occluders", [])
            for occ in occluders:
                if occ["occluder_type"] == "furniture":
                    return  # 找到了
        pytest.fail("应存在 furniture 类型遮挡物")

    def test_occluder_identity_question(self, doc):
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="canary2_occ",
            question_type="occluder_identity",
            template="挡住狗的是什么？",
            answer_modality="occluder_identity",
        )
        answer, unique, obs = derive_answer(doc, spec, binding)
        assert "家具" in answer
        assert unique is True
        assert obs is True


# ═══════════════════════════════════════════════════════════════════════════════
# Canary 3: 目标完全遮挡
# ═══════════════════════════════════════════════════════════════════════════════


class TestCanary3FullyOccluded:
    """C3: 目标在视锥内但被另一角色完全遮挡。"""

    @pytest.fixture(scope="class")
    def doc(self):
        # 另一角色覆盖目标全部像素
        normal = [
            _make_normal_semantic(occluder_slices=[(*OCCLUDER_FULL, OTHER_ACTOR_SEMANTIC_ID)])
            for _ in range(N_FRAMES)
        ]
        target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
        in_frustum = _make_in_frustum_list(N_FRAMES)
        return _make_canary_doc(
            normal, target_only, in_frustum,
            actors=[
                {
                    "actor_id": TARGET_ACTOR_ID,
                    "entity_asset_id": "dog_asset",
                    "identity": {"species_id": "狗", "breed_id": "比格犬"},
                    "realized_visual_attributes": {
                        "clothing": {"top_color": "棕色"},
                    },
                    "semantic_id": TARGET_SEMANTIC_ID,
                },
                {
                    "actor_id": "other_actor",
                    "entity_asset_id": "cat_asset",
                    "identity": {"species_id": "猫"},
                    "realized_visual_attributes": {
                        "clothing": {"top_color": "白色"},
                    },
                    "semantic_id": OTHER_ACTOR_SEMANTIC_ID,
                },
            ],
        )

    def test_fully_occluded(self, doc):
        vis_frames = doc["facts"]["visibility_facts"]["per_frame"]
        for f in vis_frames:
            state = f["actor_visibility"][TARGET_ACTOR_ID]["visibility_state"]
            assert state == "fully_occluded", f"帧 {f['frame_index']}: 预期 fully_occluded，实际 {state}"

    def test_sound_not_observable_when_fully_occluded(self, doc):
        """完全遮挡时，声音可观察性检查应返回 False。"""
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="canary3_sound",
            question_type="sound_presence",
            template="测试",
            answer_modality="sound_facts",
        )
        assert check_fact_observable(doc, spec, binding) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Canary 4: 目标完全画外 → 入画
# ═══════════════════════════════════════════════════════════════════════════════


class TestCanary4OutOfViewEnter:
    """C4: 帧 0-2 目标完全在画外，帧 3 入画（enter_frustum），帧 4 可见。"""

    @pytest.fixture(scope="class")
    def doc(self):
        normal = [_make_normal_semantic() for _ in range(N_FRAMES)]
        target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
        # 帧 0,1,2 画外，帧 3,4 画内
        in_frustum = [False, False, False, True, True]
        return _make_canary_doc(
            normal, target_only, in_frustum,
            events=[
                {"event_type": "enter_frustum", "actor_id": TARGET_ACTOR_ID, "frame_index": 3},
            ],
            spatial_frames=[
                {"frame_index": i, "actors": {
                    TARGET_ACTOR_ID: {"listener_relative": {"azimuth_deg": -45.0}},
                }} for i in range(N_FRAMES)
            ],
        )

    def test_out_of_view_and_enter(self, doc):
        vis_frames = doc["facts"]["visibility_facts"]["per_frame"]
        # 帧 0-2: out_of_view
        for i in range(3):
            state = vis_frames[i]["actor_visibility"][TARGET_ACTOR_ID]["visibility_state"]
            assert state == "out_of_view", f"帧 {i}: 预期 out_of_view，实际 {state}"
        # 帧 3-4: visible_clear
        for i in range(3, 5):
            state = vis_frames[i]["actor_visibility"][TARGET_ACTOR_ID]["visibility_state"]
            assert state == "visible_clear", f"帧 {i}: 预期 visible_clear，实际 {state}"

    def test_enter_frustum_direction(self, doc):
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="canary4_enter",
            question_type="enter_frustum_direction",
            template="狗从哪侧进入？",
            answer_modality="enter_frustum_direction",
        )
        answer, unique, obs = derive_answer(doc, spec, binding)
        assert answer == "左侧"
        assert unique is True
        assert obs is True

    def test_no_pixel_data_blocks_frustum_question(self, doc):
        """无像素数据时，入画问题应被拒绝。"""
        doc_no_pixel = dict(doc)
        doc_no_pixel["facts"]["visibility_facts"] = {"per_frame": []}
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="canary4_enter",
            question_type="enter_frustum_direction",
            template="狗从哪侧进入？",
            answer_modality="enter_frustum_direction",
        )
        assert check_fact_observable(doc_no_pixel, spec, binding) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Canary 5: 相机/听者运动，遮挡 → 重现
# ═══════════════════════════════════════════════════════════════════════════════


class TestCanary5CameraMotionReappear:
    """C5: 帧 0 可见 → 帧 1-3 被家具遮挡 → 帧 4 重现。"""

    @pytest.fixture(scope="class")
    def doc(self):
        # 帧 0: 无遮挡
        normal_0 = _make_normal_semantic()
        # 帧 1-3: 家具遮挡
        normal_occluded = _make_normal_semantic(
            occluder_slices=[(*OCCLUDER_FULL, FURNITURE_SEMANTIC_ID)]
        )
        # 帧 4: 无遮挡
        normal_4 = _make_normal_semantic()

        normal = [normal_0, normal_occluded, normal_occluded, normal_occluded, normal_4]
        target_only = [_make_target_only_semantic() for _ in range(N_FRAMES)]
        in_frustum = _make_in_frustum_list(N_FRAMES)

        return _make_canary_doc(
            normal, target_only, in_frustum,
            events=[
                {"event_type": "occlusion_start", "actor_id": TARGET_ACTOR_ID, "frame_index": 1},
                {"event_type": "reappear", "actor_id": TARGET_ACTOR_ID, "frame_index": 4},
            ],
        )

    def test_visibility_transitions(self, doc):
        vis_frames = doc["facts"]["visibility_facts"]["per_frame"]
        states = [
            f["actor_visibility"][TARGET_ACTOR_ID]["visibility_state"]
            for f in vis_frames
        ]
        assert states[0] == "visible_clear", f"帧 0: 预期 visible_clear"
        assert all(s == "fully_occluded" for s in states[1:4]), \
            f"帧 1-3: 预期 fully_occluded，实际 {states[1:4]}"
        assert states[4] == "visible_clear", f"帧 4: 预期 visible_clear"

    def test_has_reappear_event(self, doc):
        events = doc["facts"]["events"]
        event_types = [e["event_type"] for e in events]
        assert "reappear" in event_types

    def test_sound_visibility_question(self, doc):
        """帧 1 发声时完全遮挡，应返回'完全遮挡'。"""
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="canary5_vis",
            question_type="sound_visibility",
            template="狗叫时是否可见？",
            answer_modality="sound_visibility",
        )
        answer, unique, obs = derive_answer(doc, spec, binding)
        assert answer == "完全遮挡"
        assert unique is True
        assert obs is True


# ═══════════════════════════════════════════════════════════════════════════════
# 跨 Canary 验证：无像素真值时禁止产生入画/可见性/遮挡问题
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoPixelDataRejection:
    """验证关键约束：没有像素真值时，禁止产生入画、可见性或遮挡问题。"""

    @pytest.fixture
    def empty_doc(self):
        return {
            "schema": "avengine_qa_episode_v1",
            "episode_id": "no_pixel",
            "assets_used": {
                "actors": [
                    {
                        "actor_id": TARGET_ACTOR_ID,
                        "entity_asset_id": "dog_asset",
                        "identity": {"species_id": "狗"},
                        "semantic_id": TARGET_SEMANTIC_ID,
                    },
                ],
                "sounds": [],
            },
            "scene": {},
            "timeline": {},
            "facts": {
                "sound_facts": [],
                "spatial_facts": {"per_frame": []},
                "motion_facts": {"per_frame": []},
                "visibility_facts": {"per_frame": []},  # 空！
                "events": [],
            },
            "qa_pairs": [],
            "sidecars": [],
            "provenance": {},
            "episode_content_sha256": "deadbeef",
        }

    def test_enter_frustum_rejected_without_pixels(self, empty_doc):
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="test",
            question_type="enter_frustum_direction",
            template="测试",
            answer_modality="enter_frustum_direction",
        )
        assert check_fact_observable(empty_doc, spec, binding) is False

    def test_sound_visibility_rejected_without_pixels(self, empty_doc):
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="test",
            question_type="sound_visibility",
            template="测试",
            answer_modality="sound_visibility",
        )
        assert check_fact_observable(empty_doc, spec, binding) is False

    def test_occluder_identity_rejected_without_pixels(self, empty_doc):
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="test",
            question_type="occluder_identity",
            template="测试",
            answer_modality="occluder_identity",
        )
        assert check_fact_observable(empty_doc, spec, binding) is False

    def test_sound_facts_still_works_without_pixels(self, empty_doc):
        """声音问题不受像素数据缺失影响。"""
        binding = _make_binding()
        spec = QuestionSpec(
            spec_id="test",
            question_type="sound_presence",
            template="测试",
            answer_modality="sound_facts",
        )
        # sound_facts 即使无像素也可观察
        assert check_fact_observable(empty_doc, spec, binding) is True
