"""像素可见性与遮挡分析单元测试（任务二）。"""

from __future__ import annotations

import numpy as np
import pytest

from avengine.qa.episode import (
    VISIBILITY_CLEAR,
    VISIBILITY_FULLY_OCCLUDED,
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUT_OF_VIEW,
)
from avengine.qa.pixel_visibility import (
    analyze_all_frames,
    analyze_frame,
    compute_bbox,
    count_semantic_pixels,
    detect_border_touch,
    detect_occluders,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _array(values: list[list[int]]) -> np.ndarray:
    """从嵌套列表创建二维 int64 数组。"""
    return np.array(values, dtype=np.int64)


def _full(semantic_id: int, size: int = 10) -> np.ndarray:
    """返回全部填充为 *semantic_id* 的数组。"""
    return np.full((size, size), semantic_id, dtype=np.int64)


def _zeros(size: int = 10) -> np.ndarray:
    """返回全零数组。"""
    return np.zeros((size, size), dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════════
# count_semantic_pixels
# ═══════════════════════════════════════════════════════════════════════════════


class TestCountSemanticPixels:
    """语义像素计数测试。"""

    def test_all_matching(self):
        """全部匹配。"""
        arr = _full(5)
        assert count_semantic_pixels(arr, 5) == 100

    def test_none_matching(self):
        """无匹配。"""
        arr = _full(5)
        assert count_semantic_pixels(arr, 99) == 0

    def test_partial_matching(self):
        """部分匹配。"""
        arr = _zeros()
        arr[0:2, 0:5] = 7  # 10 像素
        assert count_semantic_pixels(arr, 7) == 10

    def test_zero_semantic_id(self):
        """语义 ID 为 0（背景）时正常计数。"""
        arr = _zeros()
        assert count_semantic_pixels(arr, 0) == 100

    def test_rejects_non_2d(self):
        """非二维数组应抛出异常。"""
        with pytest.raises(ValueError, match="二维"):
            count_semantic_pixels(np.array([1, 2, 3]), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# detect_border_touch
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectBorderTouch:
    """画面边缘触碰检测测试。"""

    def test_touches_top(self):
        """触碰上边缘。"""
        arr = _zeros()
        arr[0, 3] = 1
        assert detect_border_touch(arr, 1) is True

    def test_touches_bottom(self):
        """触碰下边缘。"""
        arr = _zeros()
        arr[-1, 3] = 1
        assert detect_border_touch(arr, 1) is True

    def test_touches_left(self):
        """触碰左边缘。"""
        arr = _zeros()
        arr[3, 0] = 1
        assert detect_border_touch(arr, 1) is True

    def test_touches_right(self):
        """触碰右边缘。"""
        arr = _zeros()
        arr[3, -1] = 1
        assert detect_border_touch(arr, 1) is True

    def test_touches_corner(self):
        """触碰角落（同时触碰两条边）。"""
        arr = _zeros()
        arr[0, 0] = 1
        assert detect_border_touch(arr, 1) is True

    def test_no_border_touch(self):
        """仅出现在画面内部，不触碰边缘。"""
        arr = _zeros()
        arr[3:7, 3:7] = 1  # 仅在中心区域
        assert detect_border_touch(arr, 1) is False

    def test_zero_pixels(self):
        """目标无像素时返回 False。"""
        arr = _zeros()
        assert detect_border_touch(arr, 99) is False

    def test_1x1_array(self):
        """1×1 数组必然触碰边缘。"""
        arr = np.array([[1]], dtype=np.int64)
        assert detect_border_touch(arr, 1) is True


# ═══════════════════════════════════════════════════════════════════════════════
# compute_bbox
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeBbox:
    """包围盒计算测试。"""

    def test_single_pixel(self):
        """单个像素的包围盒。"""
        arr = _zeros()
        arr[5, 3] = 1
        bbox = compute_bbox(arr, 1)
        assert bbox == (3, 5, 3, 5)

    def test_rectangular_region(self):
        """矩形区域的包围盒。"""
        arr = _zeros()
        arr[2:6, 3:8] = 1
        bbox = compute_bbox(arr, 1)
        assert bbox == (3, 2, 7, 5)

    def test_irregular_shape(self):
        """不规则形状的包围盒。"""
        arr = _zeros()
        arr[2, 3] = 1
        arr[4, 7] = 1
        bbox = compute_bbox(arr, 1)
        assert bbox == (3, 2, 7, 4)

    def test_multiple_disconnected_regions(self):
        """多个不相连区域取全局包围盒。"""
        arr = _zeros()
        arr[1, 1] = 1
        arr[8, 8] = 1
        bbox = compute_bbox(arr, 1)
        assert bbox == (1, 1, 8, 8)

    def test_zero_pixels(self):
        """无匹配像素时返回 None。"""
        arr = _zeros()
        assert compute_bbox(arr, 99) is None

    def test_full_frame(self):
        """充满整个画面的包围盒。"""
        arr = _full(1)
        bbox = compute_bbox(arr, 1)
        assert bbox == (0, 0, 9, 9)


# ═══════════════════════════════════════════════════════════════════════════════
# detect_occluders
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectOccluders:
    """遮挡物检测测试。"""

    def test_no_occlusion(self):
        """正常 pass 与目标专用 pass 完全一致——无遮挡。"""
        normal = _full(1)
        target_only = _full(1)
        result = detect_occluders(normal, target_only, 1)
        assert result == []

    def test_known_actor_occluder(self):
        """被已知角色遮挡。"""
        target_only = _full(1)
        normal = _full(1)
        normal[3:7, 3:7] = 5  # 被语义 ID=5 的角色遮挡
        result = detect_occluders(
            normal, target_only, 1,
            actor_semantic_map={5: "human0"},
        )
        assert len(result) == 1
        assert result[0]["occluder_type"] == "actor"
        assert result[0]["actor_id"] == "human0"

    def test_known_furniture_occluder(self):
        """被已知家具遮挡。"""
        target_only = _full(1)
        normal = _full(1)
        normal[2:5, 2:8] = 100  # 被家具遮挡
        result = detect_occluders(
            normal, target_only, 1,
            furniture_semantic_map={100: ("table_01", "table")},
        )
        assert len(result) == 1
        assert result[0]["occluder_type"] == "furniture"
        assert result[0]["instance_id"] == "table_01"
        assert result[0]["semantic_label"] == "table"

    def test_unknown_static_occluder(self):
        """被未映射的语义 ID 遮挡——标记为 unknown_static。"""
        target_only = _full(1)
        normal = _full(1)
        normal[5:8, 5:8] = 999
        result = detect_occluders(normal, target_only, 1)
        assert len(result) == 1
        assert result[0]["occluder_type"] == "unknown_static"
        assert result[0]["semantic_id"] == 999

    def test_multiple_occluders_sorted_by_pixel_count(self):
        """多个遮挡物按像素数降序排列。"""
        target_only = _full(1)
        normal = _full(1)
        normal[0:6, 0:10] = 5    # 60 像素 → 角色
        normal[6:10, 0:10] = 100  # 40 像素 → 家具
        result = detect_occluders(
            normal, target_only, 1,
            actor_semantic_map={5: "human0"},
            furniture_semantic_map={100: ("table_01", "table")},
        )
        assert len(result) == 2
        # 遮挡像素多的排在前面（60 px vs 40 px）
        assert result[0]["occluder_type"] == "actor"
        assert result[1]["occluder_type"] == "furniture"

    def test_mixed_occluder_types(self):
        """同一帧中同时出现角色、家具和未知遮挡物。"""
        target_only = _full(1)
        normal = _full(1)
        normal[0:4, 0:10] = 5    # 40 px 角色
        normal[4:7, 0:10] = 100  # 30 px 家具
        normal[7:10, 0:10] = 999  # 30 px 未知
        result = detect_occluders(
            normal, target_only, 1,
            actor_semantic_map={5: "human0"},
            furniture_semantic_map={100: ("table_01", "table")},
        )
        assert len(result) == 3
        types = [r["occluder_type"] for r in result]
        assert types == ["actor", "furniture", "unknown_static"]

    def test_target_not_present(self):
        """目标在目标专用 pass 中不存在——无遮挡物。"""
        normal = _full(5)
        target_only = _zeros()
        result = detect_occluders(normal, target_only, 1)
        assert result == []

    def test_background_filtered_out(self):
        """语义 ID 0（背景）被过滤，不作为遮挡物。"""
        target_only = _full(1)
        normal = _full(0)  # 全部为背景
        result = detect_occluders(normal, target_only, 1)
        assert result == []

    def test_target_self_filtered_out(self):
        """目标自身的语义 ID 被排除在遮挡物之外。"""
        target_only = _full(1)
        normal = _full(1)
        normal[5, 5] = 3  # 少量其他 ID
        result = detect_occluders(normal, target_only, 1)
        # 只报告 semantic_id=3 作为遮挡物
        assert len(result) == 1
        assert result[0]["semantic_id"] == 3

    def test_default_empty_maps(self):
        """未提供映射时，所有遮挡物均为 unknown_static。"""
        target_only = _full(1)
        normal = _full(1)
        normal[3:6, 3:6] = 50
        result = detect_occluders(normal, target_only, 1)
        assert len(result) == 1
        assert result[0]["occluder_type"] == "unknown_static"


# ═══════════════════════════════════════════════════════════════════════════════
# analyze_frame
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyzeFrame:
    """单帧分析测试。"""

    def test_visible_clear(self):
        """完全无遮挡。"""
        normal = _full(1)
        target_only = _full(1)
        rec = analyze_frame(normal, target_only, 1)
        assert rec["visibility_state"] == VISIBILITY_CLEAR
        assert rec["amodal_pixels"] == 100
        assert rec["visible_pixels"] == 100
        assert rec["visible_fraction"] == pytest.approx(1.0)

    def test_visible_occluded(self):
        """部分遮挡。"""
        target_only = _full(1)       # 全部 100 像素为目标
        normal = _full(1).copy()
        normal[0:5, 0:10] = 5       # 50 像素被遮挡
        rec = analyze_frame(normal, target_only, 1)
        assert rec["visibility_state"] == VISIBILITY_OCCLUDED
        assert rec["amodal_pixels"] == 100
        assert rec["visible_pixels"] == 50
        assert rec["visible_fraction"] == pytest.approx(0.5)

    def test_fully_occluded(self):
        """几乎完全遮挡——仅 1 像素可见。"""
        target_only = _full(1)
        normal = _full(1).copy()
        normal[0:10, 0:10] = 5  # 全部被遮挡
        normal[0, 0] = 1        # 仅 1 像素可见
        rec = analyze_frame(normal, target_only, 1)
        assert rec["visibility_state"] == VISIBILITY_FULLY_OCCLUDED
        assert rec["visible_pixels"] == 1

    def test_out_of_view_not_in_frustum(self):
        """不在视锥体内 → out_of_view。"""
        normal = _full(1)
        target_only = _full(1)
        rec = analyze_frame(normal, target_only, 1, in_frustum=False)
        assert rec["visibility_state"] == VISIBILITY_OUT_OF_VIEW

    def test_out_of_view_zero_amodal(self):
        """在视锥体内但目标专用 pass 无像素 → out_of_view。"""
        normal = _zeros()
        target_only = _zeros()
        rec = analyze_frame(normal, target_only, 1, in_frustum=True)
        assert rec["visibility_state"] == VISIBILITY_OUT_OF_VIEW
        assert rec["amodal_pixels"] == 0

    def test_border_touch_propagated(self):
        """画面边缘触碰信息正确传递到记录中。"""
        normal = _full(1)
        target_only = _full(1)
        rec = analyze_frame(normal, target_only, 1)
        assert rec["touches_frame_border"] is True  # 满屏 10×10 必然触碰边缘

    def test_border_no_touch(self):
        """目标仅在画面内部——不触碰边缘。"""
        normal = _zeros()
        target_only = _zeros()
        normal[3:7, 3:7] = 1
        target_only[3:7, 3:7] = 1
        rec = analyze_frame(normal, target_only, 1)
        assert rec["touches_frame_border"] is False

    def test_bbox_propagated(self):
        """包围盒信息正确传递到记录中。"""
        normal = _zeros()
        target_only = _zeros()
        normal[2:6, 3:8] = 1
        target_only[2:6, 3:8] = 1
        rec = analyze_frame(normal, target_only, 1)
        assert rec["bbox_visible"] == {"x_min": 3, "y_min": 2, "x_max": 7, "y_max": 5}

    def test_bbox_none_when_zero_pixels(self):
        """目标无像素时包围盒为 None。"""
        rec = analyze_frame(_zeros(), _zeros(), 1)
        assert "bbox_visible" not in rec or rec.get("bbox_visible") is None

    def test_occluders_in_record(self):
        """遮挡物信息正确传递到记录中。"""
        target_only = _full(1)
        normal = _full(1).copy()
        normal[0:3, 0:10] = 5
        rec = analyze_frame(
            normal, target_only, 1,
            actor_semantic_map={5: "human0"},
        )
        assert len(rec["occluders"]) == 1
        assert rec["occluders"][0]["actor_id"] == "human0"

    def test_custom_thresholds(self):
        """自定义阈值影响分类结果。"""
        target_only = _full(1)
        normal = _full(1).copy()
        normal[0:3, 0:10] = 5  # 100 像素中 70 可见
        # 严苛阈值下 0.70 < 0.95 → occluded
        rec_strict = analyze_frame(normal, target_only, 1, clear_threshold=0.95)
        assert rec_strict["visibility_state"] == VISIBILITY_OCCLUDED
        # 宽松阈值下 0.70 >= 0.60 → clear
        rec_loose = analyze_frame(normal, target_only, 1, clear_threshold=0.60)
        assert rec_loose["visibility_state"] == VISIBILITY_CLEAR


# ═══════════════════════════════════════════════════════════════════════════════
# analyze_all_frames
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyzeAllFrames:
    """批量帧分析测试。"""

    def test_uniform_clear_batch(self):
        """所有帧均完全无遮挡。"""
        normal = [_full(1) for _ in range(3)]
        target_only = [_full(1) for _ in range(3)]
        frames = analyze_all_frames(normal, target_only, 1, actor_id="dog0")
        assert len(frames) == 3
        for fi, frame in enumerate(frames):
            assert frame["frame_index"] == fi
            assert frame["actor_visibility"]["dog0"]["visibility_state"] == VISIBILITY_CLEAR

    def test_varying_visibility(self):
        """中间一帧被完全遮挡。"""
        normal = []
        target_only = []
        for i in range(3):
            n = _full(1)
            t = _full(1)
            if i == 1:
                n[:] = 5  # 中间帧被完全遮挡
            normal.append(n)
            target_only.append(t)
        frames = analyze_all_frames(normal, target_only, 1, actor_id="a1")
        assert frames[0]["actor_visibility"]["a1"]["visibility_state"] == VISIBILITY_CLEAR
        assert frames[1]["actor_visibility"]["a1"]["visibility_state"] == VISIBILITY_FULLY_OCCLUDED
        assert frames[2]["actor_visibility"]["a1"]["visibility_state"] == VISIBILITY_CLEAR

    def test_default_in_frustums_all_true(self):
        """未提供 in_frustums 时全部默认为 True。"""
        normal = [_full(1) for _ in range(2)]
        target_only = [_zeros() for _ in range(2)]  # 零像素但在视锥体内
        frames = analyze_all_frames(normal, target_only, 1, actor_id="a1")
        # amodal=0, in_frustum=True → out_of_view
        for f in frames:
            assert f["actor_visibility"]["a1"]["visibility_state"] == VISIBILITY_OUT_OF_VIEW

    def test_explicit_in_frustums(self):
        """显式指定视锥体标志。"""
        normal = [_full(1) for _ in range(2)]
        target_only = [_full(1) for _ in range(2)]
        frames = analyze_all_frames(
            normal, target_only, 1, actor_id="a1",
            in_frustums=[True, False],
        )
        assert frames[0]["actor_visibility"]["a1"]["visibility_state"] == VISIBILITY_CLEAR
        assert frames[1]["actor_visibility"]["a1"]["visibility_state"] == VISIBILITY_OUT_OF_VIEW

    def test_actor_id_fallback(self):
        """actor_id 为空时回退为 str(target_semantic_id)。"""
        normal = [_full(1)]
        target_only = [_full(1)]
        frames = analyze_all_frames(normal, target_only, 1)
        assert "1" in frames[0]["actor_visibility"]

    def test_length_mismatch_raises(self):
        """两路 pass 长度不一致时抛出异常。"""
        with pytest.raises(ValueError, match="长度"):
            analyze_all_frames([_full(1)], [], 1)

    def test_in_frustums_length_mismatch_raises(self):
        """in_frustums 长度与帧数不一致时抛出异常。"""
        with pytest.raises(ValueError, match="in_frustums"):
            analyze_all_frames([_full(1)], [_full(1)], 1, in_frustums=[True, False])

    def test_empty_maps_default(self):
        """未提供映射时遮挡物为 unknown_static。"""
        normal = [_full(1).copy() for _ in range(1)]
        target_only = [_full(1) for _ in range(1)]
        normal[0][5, 5] = 50  # 一个像素被遮挡
        frames = analyze_all_frames(normal, target_only, 1, actor_id="a1")
        rec = frames[0]["actor_visibility"]["a1"]
        assert len(rec["occluders"]) == 1
        assert rec["occluders"][0]["occluder_type"] == "unknown_static"

    def test_with_actor_map(self):
        """提供角色映射时正确识别角色遮挡物。"""
        normal = [_full(1).copy() for _ in range(1)]
        target_only = [_full(1) for _ in range(1)]
        normal[0][0:2, 0:2] = 5  # 4 像素被角色 5 遮挡
        frames = analyze_all_frames(
            normal, target_only, 1, actor_id="dog0",
            actor_semantic_map={5: "human0"},
        )
        rec = frames[0]["actor_visibility"]["dog0"]
        assert rec["occluders"][0]["occluder_type"] == "actor"
        assert rec["occluders"][0]["actor_id"] == "human0"
