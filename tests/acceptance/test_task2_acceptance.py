"""任务二验收测试 — 像素级可见性与遮挡分析。

在仓库根目录运行::

    python -m pytest tests/acceptance/test_task2_acceptance.py -v

需求对照见 docs/planning/TASK2_DEVLOG.md。
"""

from __future__ import annotations

import numpy as np
import pytest

from avengine.m5.timeline import FRAME_COUNT
from avengine.qa.episode import (
    DEFAULT_CLEAR_THRESHOLD,
    DEFAULT_VISIBLE_THRESHOLD,
    Episode,
    EpisodeError,
    QAPair,
    QA_EPISODE_SCHEMA,
    VISIBILITY_CLEAR,
    VISIBILITY_FULLY_OCCLUDED,
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUT_OF_VIEW,
    classify_visibility,
    detect_visibility_events,
    make_visibility_record,
    validate_qa_episode,
)
from avengine.qa.pixel_visibility import (
    analyze_all_frames,
    analyze_frame,
    compute_bbox,
    count_semantic_pixels,
    detect_border_touch,
    detect_occluders,
)

# 复用单元测试模块中的共享最小 Timeline 夹具。
from tests.unit.test_qa_episode import _valid_timeline


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _arr(size: int = 10) -> np.ndarray:
    """返回 (size, size) 全零 int64 数组。"""
    return np.zeros((size, size), dtype=np.int64)


def _full(id_: int, size: int = 10) -> np.ndarray:
    """返回 (size, size) 全部填充为 *id_* 的数组。"""
    return np.full((size, size), id_, dtype=np.int64)


def _minimal_episode() -> Episode:
    """返回预填充了最基本合法数据的 Episode 构建器。"""
    ep = (
        Episode("ep_t2_accept")
        .add_actor("dog0", "beagle_01", "dog", 1, breed_id="beagle",
                   size="medium", body_build="standard", life_stage="adult")
        .add_sound("bark_01", "dog_vocalization", "dog0", sound_category="bark")
    )
    ep.timeline = _valid_timeline()
    ep.rgb_video = "accept/rgb.mp4"
    ep.semantic_video = "accept/semantic.mp4"
    ep.depth_frames = "accept/depth/"
    ep.target_only_masks = "accept/target_only/"
    ep.audio_mix_binaural = "accept/binaural.wav"
    ep.audio_mix_foa = "accept/foa.wav"
    ep.visibility_overlay = "accept/overlay/"
    ep.seed = 42

    for fi in range(FRAME_COUNT):
        ep.spatial_frames.append({
            "frame_index": fi,
            "actors": {
                "dog0": {
                    "position_m": [0.0, 0.0, 0.0],
                    "forward_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "listener_relative": {
                        "distance_m": 2.0,
                        "azimuth_deg": 0.0,
                        "elevation_deg": 0.0,
                    },
                    "in_frustum": True,
                },
            },
            "listener": {
                "position_m": [0.0, 1.6, 0.0],
                "forward_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        })
        ep.motion_frames.append({
            "frame_index": fi,
            "actor_states": {"dog0": "idle"},
        })
        ep.visibility_frames.append({
            "frame_index": fi,
            "actor_visibility": {
                "dog0": make_visibility_record(1000, 900, True).as_dict(),
            },
        })

    return ep


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 包导入 — 所有任务二符号必须可解析
# ═══════════════════════════════════════════════════════════════════════════════


class TestTask2Imports:
    """验证所有任务二公开符号均可从包顶层导入。"""

    def test_imports_from_package_top(self):
        """从 avengine.qa 顶层导入所有任务二符号。"""
        from avengine.qa import (  # noqa: F811
            analyze_all_frames,
            analyze_frame,
            compute_bbox,
            count_semantic_pixels,
            detect_border_touch,
            detect_occluders,
        )
        assert analyze_frame is not None

    def test_imports_from_pixel_visibility_module(self):
        """从子模块直接导入所有符号（含内部辅助函数）。"""
        from avengine.qa.pixel_visibility import (  # noqa: F811
            analyze_all_frames,
            analyze_frame,
            compute_bbox,
            count_semantic_pixels,
            detect_border_touch,
            detect_occluders,
            _guard_2d_integer,
        )
        assert analyze_frame is not None
        assert _guard_2d_integer is not None

    def test_task2_symbols_in_all(self):
        """__all__ 中列出的每个符号必须在模块中真实存在。"""
        import avengine.qa.pixel_visibility as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} 在 __all__ 中但模块中不存在"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 像素基础工具 — 在已知输入上的正确行为
# ═══════════════════════════════════════════════════════════════════════════════


class TestPixelPrimitivesAcceptance:
    """像素级辅助函数的验收级检查。"""

    def test_count_matches_expectation(self):
        """大尺寸数组的像素计数。"""
        arr = _full(3, size=50)
        assert count_semantic_pixels(arr, 3) == 2500

    def test_border_touch_on_large_array(self):
        """大尺寸数组的边缘触碰检测。"""
        arr = _arr(64)
        arr[0, 30] = 7
        assert detect_border_touch(arr, 7) is True

    def test_no_border_touch_interior(self):
        """仅出现在画面内部时不应报告触碰边缘。"""
        arr = _arr(64)
        arr[10:20, 10:20] = 7
        assert detect_border_touch(arr, 7) is False

    def test_bbox_on_large_array(self):
        """大尺寸数组的包围盒计算。"""
        arr = _arr(64)
        arr[10:30, 15:45] = 5
        bbox = compute_bbox(arr, 5)
        assert bbox == (15, 10, 44, 29)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 遮挡物检测 — 安全兜底、排序、分类
# ═══════════════════════════════════════════════════════════════════════════════


class TestOccluderDetectionAcceptance:
    """端到端遮挡物检测（含已知映射）。"""

    def test_multiple_occluders_descending_pixel_count(self):
        """多个遮挡物按遮挡像素数降序排列。"""
        target_only = _full(1)
        normal = _full(1)
        normal[0:5, :] = 5      # 50 px — 角色
        normal[5:8, :] = 100    # 30 px — 家具
        normal[8:10, :] = 200   # 20 px — 未知

        result = detect_occluders(
            normal, target_only, 1,
            actor_semantic_map={5: "human1"},
            furniture_semantic_map={100: ("chair_01", "chair")},
        )
        assert len(result) == 3
        assert result[0]["occluder_type"] == "actor"
        assert result[1]["occluder_type"] == "furniture"
        assert result[2]["occluder_type"] == "unknown_static"

    def test_fail_closed_for_unmapped_id(self):
        """未映射的语义 ID 必须归类为 unknown_static（绝不猜测）。"""
        target_only = _full(1)
        normal = _full(1)
        normal[4:6, 4:6] = 9999

        result = detect_occluders(normal, target_only, 1)
        assert len(result) == 1
        assert result[0]["occluder_type"] == "unknown_static"
        assert "actor_id" not in result[0]

    def test_empty_when_no_occlusion(self):
        """无遮挡时返回空列表。"""
        target_only = _full(1)
        normal = _full(1)
        result = detect_occluders(normal, target_only, 1)
        assert result == []

    def test_background_pixels_ignored(self):
        """语义 ID 0 绝对不能出现在遮挡物列表中。"""
        target_only = _full(1)
        normal = _full(0)
        result = detect_occluders(normal, target_only, 1)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 单帧分析 — analyze_frame → VisibilityRecord 字典
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyzeFrameAcceptance:
    """单帧分析产出的记录在组装成 Episode 后可通过 schema 校验。"""

    def test_visible_clear_round_trip(self):
        """完全无遮挡时的往返验证。"""
        rec = analyze_frame(_full(1), _full(1), 1, in_frustum=True)
        assert rec["visibility_state"] == VISIBILITY_CLEAR
        assert rec["amodal_pixels"] == 100
        assert rec["visible_pixels"] == 100
        assert rec["visible_fraction"] == pytest.approx(1.0)

    def test_out_of_view_via_frustum_flag(self):
        """通过视锥体标志判定为画外。"""
        rec = analyze_frame(_full(1), _full(1), 1, in_frustum=False)
        assert rec["visibility_state"] == VISIBILITY_OUT_OF_VIEW

    def test_out_of_view_via_zero_amodal(self):
        """目标专用 pass 无像素时判定为画外。"""
        rec = analyze_frame(_arr(), _arr(), 1, in_frustum=True)
        assert rec["visibility_state"] == VISIBILITY_OUT_OF_VIEW
        assert rec["amodal_pixels"] == 0

    def test_occluded_with_known_occluder(self):
        """已知角色遮挡的端到端检测。"""
        target_only = _full(1)
        normal = _full(1).copy()
        normal[0:6, :] = 7

        rec = analyze_frame(
            normal, target_only, 1, in_frustum=True,
            actor_semantic_map={7: "occluder_human"},
        )
        assert rec["visibility_state"] == VISIBILITY_OCCLUDED
        assert len(rec["occluders"]) == 1
        assert rec["occluders"][0]["actor_id"] == "occluder_human"

    def test_bbox_none_when_target_absent(self):
        """目标不存在时包围盒为 None。"""
        rec = analyze_frame(_arr(), _arr(), 1, in_frustum=True)
        assert "bbox_visible" not in rec or rec.get("bbox_visible") is None

    def test_fully_occluded_at_boundary(self):
        """5×5=25 px 总像素，1 像素可见 → 0.04 < 0.05 → fully_occluded。"""
        target_only = _full(1, size=5)
        normal = _full(1, size=5).copy()
        normal[:] = 9
        normal[0, 0] = 1  # 仅 1 像素可见

        rec = analyze_frame(normal, target_only, 1)
        assert rec["visibility_state"] == VISIBILITY_FULLY_OCCLUDED


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 批量分析 — analyze_all_frames
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyzeAllFramesAcceptance:
    """批量分析返回正确的结构以供 Episode 集成。"""

    def test_produces_correct_structure(self):
        """返回结构包含 frame_index 和 actor_visibility。"""
        normal = [_full(1) for _ in range(3)]
        target_only = [_full(1) for _ in range(3)]
        frames = analyze_all_frames(normal, target_only, 1, actor_id="a1")

        assert len(frames) == 3
        for fi, f in enumerate(frames):
            assert f["frame_index"] == fi
            assert "actor_visibility" in f
            assert "a1" in f["actor_visibility"]
            assert "visibility_state" in f["actor_visibility"]["a1"]

    def test_full_episode_population(self):
        """将批量分析输出赋给 Episode，构建并通过校验。"""
        ep = _minimal_episode()

        # 用 analyze_all_frames 的输出替换可见性帧
        normal = [_full(1) for _ in range(FRAME_COUNT)]
        target_only = [_full(1) for _ in range(FRAME_COUNT)]
        batch = analyze_all_frames(normal, target_only, 1, actor_id="dog0")

        ep.visibility_frames = batch
        ep.add_qa(QAPair("q1", "sound_presence", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc = ep.build()

        assert doc["schema"] == QA_EPISODE_SCHEMA
        assert doc["facts"]["visibility_facts"]["per_frame"][0] == batch[0]

    def test_varying_visibility_sequence(self):
        """三帧：清晰 → 完全遮挡 → 完全遮挡。"""
        normal = []
        target_only = []
        for i in range(3):
            n = _full(1)
            t = _full(1)
            if i == 1:
                n[:] = 5       # 100% 被角色遮挡
            elif i == 2:
                n[:] = 5       # 同样完全遮挡
                n[0, 0] = 1    # 仅 1 个可见像素
            normal.append(n)
            target_only.append(t)

        frames = analyze_all_frames(
            normal, target_only, 1, actor_id="dog0",
            actor_semantic_map={5: "human0"},
        )
        assert frames[0]["actor_visibility"]["dog0"]["visibility_state"] == VISIBILITY_CLEAR
        assert frames[1]["actor_visibility"]["dog0"]["visibility_state"] == VISIBILITY_FULLY_OCCLUDED
        assert frames[2]["actor_visibility"]["dog0"]["visibility_state"] == VISIBILITY_FULLY_OCCLUDED

    def test_explicit_in_frustum_flags(self):
        """显式指定逐帧视锥体标志。"""
        normal = [_full(1) for _ in range(2)]
        target_only = [_full(1) for _ in range(2)]
        frames = analyze_all_frames(
            normal, target_only, 1, actor_id="a1",
            in_frustums=[True, False],
        )
        assert frames[0]["actor_visibility"]["a1"]["visibility_state"] == VISIBILITY_CLEAR
        assert frames[1]["actor_visibility"]["a1"]["visibility_state"] == VISIBILITY_OUT_OF_VIEW


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 集成测试 — 像素分析 → 可见性帧 → 事件 → Episode
# ═══════════════════════════════════════════════════════════════════════════════


class TestPixelToEpisodeIntegration:
    """端到端：合成像素 → 帧 → 事件 → 构建完成的 Episode。"""

    def test_pipeline_integration_with_events(self):
        """含遮挡的 5 帧序列应产生预期事件。"""
        target_only = [_full(1, size=5) for _ in range(5)]
        normal = [_full(1, size=5).copy() for _ in range(5)]
        # 帧 0、1：清晰
        # 帧 2：   被家具部分遮挡（~12/25 = ~50% 目标面积）
        normal[2][0:3, :] = 100
        # 帧 3：   几乎完全遮挡 — 仅 1 像素可见
        normal[3][:] = 100
        normal[3][0, 0] = 1
        # 帧 4：   恢复清晰

        frames = analyze_all_frames(
            normal, target_only, 1, actor_id="dog0",
            furniture_semantic_map={100: ("table_01", "table")},
        )

        events = detect_visibility_events(frames, "dog0")

        # 预期事件：occlusion_start（帧 2）、fully_occluded（帧 3）、reappear（帧 4）
        event_types = {e.event_type for e in events}
        assert "occlusion_start" in event_types
        assert "fully_occluded" in event_types
        assert "reappear" in event_types

    def test_full_episode_build_with_pixel_events(self):
        """使用像素分析输出构建完整 Episode 并通过 schema 校验。"""
        n_frames = FRAME_COUNT

        # 所有帧：目标可见且清晰
        normal = [_full(1) for _ in range(n_frames)]
        target_only = [_full(1) for _ in range(n_frames)]

        # 在第 10–14 帧引入遮挡
        for fi in range(10, 15):
            normal[fi][:] = 5  # 被角色遮挡

        frames = analyze_all_frames(
            normal, target_only, 1, actor_id="dog0",
            actor_semantic_map={5: "human0"},
        )
        events = detect_visibility_events(frames, "dog0")

        ep = _minimal_episode()
        ep.visibility_frames = frames
        for evt in events:
            ep.add_event(evt)
        ep.add_qa(QAPair("q1", "sound_presence", "Q?", "A",
                         answer_unique=True, fact_observable=True))

        doc = ep.build()
        errors = validate_qa_episode(doc)
        assert errors == [], f"Schema 校验错误: {errors}"

        # 验证构建后文档中的可见性数据
        vis_facts = doc["facts"]["visibility_facts"]["per_frame"]
        assert len(vis_facts) == n_frames
        assert vis_facts[0]["actor_visibility"]["dog0"]["visibility_state"] == VISIBILITY_CLEAR
        assert vis_facts[10]["actor_visibility"]["dog0"]["visibility_state"] == VISIBILITY_FULLY_OCCLUDED
        assert vis_facts[14]["actor_visibility"]["dog0"]["visibility_state"] == VISIBILITY_FULLY_OCCLUDED
        assert vis_facts[15]["actor_visibility"]["dog0"]["visibility_state"] == VISIBILITY_CLEAR


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 自定义阈值 — 不同阈值影响分类结果
# ═══════════════════════════════════════════════════════════════════════════════


class TestCustomThresholdsAcceptance:
    """验收级阈值自定义测试。"""

    def test_lenient_clear_threshold(self):
        """clear_threshold=0.6 时 70% 可见即为 'clear'。"""
        target_only = _full(1)
        normal = _full(1).copy()
        normal[0:3, :] = 0  # 100 中 70 可见

        rec = analyze_frame(normal, target_only, 1, clear_threshold=0.60)
        assert rec["visibility_state"] == VISIBILITY_CLEAR

    def test_strict_clear_threshold(self):
        """clear_threshold=0.99 时 97% 可见仍为 'occluded'。"""
        target_only = _full(1)
        normal = _full(1).copy()
        normal[0:3, :] = 0  # 100 中 70 可见

        rec = analyze_frame(normal, target_only, 1, clear_threshold=0.99)
        assert rec["visibility_state"] == VISIBILITY_OCCLUDED

    def test_custom_visible_threshold(self):
        """visible_threshold=0.20 时 10% 可见即为 'fully_occluded'。"""
        target_only = _full(1)
        normal = _full(1).copy()
        normal[:] = 5
        normal[0, 0:10] = 1  # 100 中 10 可见

        rec = analyze_frame(normal, target_only, 1, visible_threshold=0.20)
        assert rec["visibility_state"] == VISIBILITY_FULLY_OCCLUDED

    def test_batch_custom_thresholds(self):
        """批量分析时传递自定义阈值。"""
        normal = [_full(1).copy() for _ in range(2)]
        target_only = [_full(1) for _ in range(2)]
        normal[0][0:6, :] = 5  # 40 可见 → 阈值 0.3 下为 clear

        frames = analyze_all_frames(
            normal, target_only, 1, actor_id="a1",
            clear_threshold=0.30,
        )
        assert frames[0]["actor_visibility"]["a1"]["visibility_state"] == VISIBILITY_CLEAR


# ═══════════════════════════════════════════════════════════════════════════════
# 8. 输入校验 — 防御性编程
# ═══════════════════════════════════════════════════════════════════════════════


class TestInputValidationAcceptance:
    """输入校验防护测试。"""

    def test_non_2d_array_rejected(self):
        """非二维数组被拒绝。"""
        with pytest.raises(ValueError, match="二维"):
            count_semantic_pixels(np.array([1, 2, 3]), 1)

    def test_non_ndarray_rejected(self):
        """非 numpy 数组被拒绝。"""
        with pytest.raises(TypeError, match="numpy"):
            count_semantic_pixels([[1, 2], [3, 4]], 1)  # type: ignore[arg-type]

    def test_length_mismatch_rejected(self):
        """两路 pass 长度不一致被拒绝。"""
        with pytest.raises(ValueError, match="长度"):
            analyze_all_frames([_full(1)], [], 1)

    def test_in_frustums_length_mismatch(self):
        """in_frustums 长度不一致被拒绝。"""
        with pytest.raises(ValueError, match="in_frustums"):
            analyze_all_frames([_full(1)], [_full(1)], 1, in_frustums=[True, False])
