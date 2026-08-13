"""像素级可见性与遮挡分析，基于语义分割数组。

本模块为纯 Python + numpy。接收预渲染的语义数组（每帧两路——正常多物体
pass 和目标专用 pass），生成可供 QA Episode 构建器直接使用的
:class:`~avengine.qa.episode.VisibilityRecord` 实例。

**不依赖** Habitat Sim；渲染由调用方负责。

用法::

    from avengine.qa.pixel_visibility import analyze_all_frames, analyze_frame

    # 单帧分析
    record = analyze_frame(normal_sem, target_only_sem, target_id=1,
                           in_frustum=True,
                           actor_semantic_map={3: "human0"},
                           furniture_semantic_map={100: ("table_01", "table")})

    # 全部 75 帧（返回列表可直接赋给 Episode.visibility_frames）
    frames = analyze_all_frames(normal_list, target_only_list, target_id=1,
                                actor_id="dog0")
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from avengine.m5.timeline import FRAME_COUNT
from avengine.qa.episode import (
    DEFAULT_CLEAR_THRESHOLD,
    DEFAULT_VISIBLE_THRESHOLD,
    OCCLUDER_ACTOR,
    OCCLUDER_FURNITURE,
    OCCLUDER_UNKNOWN,
    make_visibility_record,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 像素基础工具
# ═══════════════════════════════════════════════════════════════════════════════


def count_semantic_pixels(semantic: np.ndarray, semantic_id: int) -> int:
    """统计 *semantic* 中与 *semantic_id* 匹配的像素数量。

    Args:
        semantic: 二维整数 numpy 数组 (H, W)。
        semantic_id: 需要统计的语义 ID。

    Returns:
        匹配的像素数量（整数）。
    """
    _guard_2d_integer(semantic)
    return int(np.count_nonzero(semantic == semantic_id))


def detect_border_touch(semantic: np.ndarray, semantic_id: int) -> bool:
    """若 *semantic_id* 的任意像素触及图像边缘则返回 True。

    沿用 ``m5_1/mp3d_capture.py``（第 560–563 行）的边缘检测模式。

    Args:
        semantic: 二维整数 numpy 数组 (H, W)。
        semantic_id: 需要检测的语义 ID。

    Returns:
        若至少有一个匹配像素位于数组的上、下、左、右边缘则返回 ``True``。
    """
    _guard_2d_integer(semantic)
    mask = semantic == semantic_id
    if not np.any(mask):
        return False
    h, w = mask.shape
    if h <= 1 or w <= 1:
        return True
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def compute_bbox(
    semantic: np.ndarray, semantic_id: int
) -> tuple[int, int, int, int] | None:
    """计算所有 *semantic_id* 像素的轴对齐包围盒。

    Args:
        semantic: 二维整数 numpy 数组 (H, W)。
        semantic_id: 需要计算包围盒的语义 ID。

    Returns:
        ``(x_min, y_min, x_max, y_max)``，其中 *x* 为列索引，
        *y* 为行索引；若无匹配像素则返回 ``None``。
    """
    _guard_2d_integer(semantic)
    rows, cols = np.nonzero(semantic == semantic_id)
    if len(rows) == 0:
        return None
    return (int(np.min(cols)), int(np.min(rows)), int(np.max(cols)), int(np.max(rows)))


# ═══════════════════════════════════════════════════════════════════════════════
# 遮挡物检测
# ═══════════════════════════════════════════════════════════════════════════════


def detect_occluders(
    normal_semantic: np.ndarray,
    target_only_semantic: np.ndarray,
    target_semantic_id: int,
    *,
    actor_semantic_map: dict[int, str] | None = None,
    furniture_semantic_map: dict[int, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """识别并分类每个遮挡目标的物体。

    在目标专用 pass 中本应可见、但在正常 pass 中被其他语义 ID 占据的像素
    即为**被遮挡像素**。这些位置上的语义 ID 就是遮挡物。

    每个遮挡物按以下规则分类：

    * ``"actor"`` — 语义 ID 命中 *actor_semantic_map*。
    * ``"furniture"`` — 语义 ID 命中 *furniture_semantic_map*。
    * ``"unknown_static"`` — 语义 ID 不在任何映射中（安全兜底，绝不猜测）。

    结果按像素数降序排列（最显著的遮挡物排在最前面）。

    Args:
        normal_semantic: 标准多物体 pass 的二维整数数组。
        target_only_semantic: 目标专用 pass 的二维整数数组。
        target_semantic_id: 目标角色的语义 ID。
        actor_semantic_map: ``{semantic_id: actor_id}`` 已知角色映射。
        furniture_semantic_map: ``{semantic_id: (instance_id, semantic_label)}``
            具备可靠语义映射的家具。

    Returns:
        遮挡物字典列表，按遮挡像素占比降序排列。
    """
    _guard_2d_integer(normal_semantic)
    _guard_2d_integer(target_only_semantic)

    if actor_semantic_map is None:
        actor_semantic_map = {}
    if furniture_semantic_map is None:
        furniture_semantic_map = {}

    # 目标在无遮挡情况下应出现的位置
    amodal_mask = target_only_semantic == target_semantic_id
    if not np.any(amodal_mask):
        return []

    # 正常 pass 中占据这些位置的语义 ID
    occluding_values = normal_semantic[amodal_mask]

    # 过滤掉目标自身和背景 (0)
    keep = (occluding_values != target_semantic_id) & (occluding_values != 0)
    filtered = occluding_values[keep]

    if len(filtered) == 0:
        return []

    unique_ids, counts = np.unique(filtered, return_counts=True)

    # 按像素数降序排列
    pairs = sorted(
        zip(unique_ids, counts, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )

    result: list[dict[str, Any]] = []
    for sid, pixel_count in pairs:
        sid_int = int(sid)
        if sid_int in actor_semantic_map:
            result.append({
                "occluder_type": OCCLUDER_ACTOR,
                "actor_id": actor_semantic_map[sid_int],
            })
        elif sid_int in furniture_semantic_map:
            instance_id, label = furniture_semantic_map[sid_int]
            result.append({
                "occluder_type": OCCLUDER_FURNITURE,
                "instance_id": instance_id,
                "semantic_label": label,
                "semantic_id": sid_int,
            })
        else:
            # 安全兜底：绝不猜测
            result.append({
                "occluder_type": OCCLUDER_UNKNOWN,
                "semantic_id": sid_int,
            })

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 单帧与批量分析
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_frame(
    normal_semantic: np.ndarray,
    target_only_semantic: np.ndarray,
    target_semantic_id: int,
    *,
    in_frustum: bool = True,
    actor_semantic_map: dict[int, str] | None = None,
    furniture_semantic_map: dict[int, tuple[str, str]] | None = None,
    clear_threshold: float = DEFAULT_CLEAR_THRESHOLD,
    visible_threshold: float = DEFAULT_VISIBLE_THRESHOLD,
) -> dict[str, Any]:
    """分析单帧：统计像素、检测遮挡物、分类可见性。

    Args:
        normal_semantic: 二维整数数组 — 标准多物体语义 pass。
        target_only_semantic: 二维整数数组 — 目标专用语义 pass
            （相同相机位姿，仅渲染目标角色）。
        target_semantic_id: 目标角色的语义 ID。
        in_frustum: 目标包围球是否与相机视锥体相交（由 3D 引擎判定，非像素判定）。
        actor_semantic_map: ``{semantic_id: actor_id}`` 已知角色映射。
        furniture_semantic_map: ``{semantic_id: (instance_id, label)}``
            具备可靠语义映射的家具。
        clear_threshold: 高于此比例判定为 ``visible_clear``。
        visible_threshold: 低于此比例判定为 ``fully_occluded``
            （前提是目标仍在视锥体内）。

    Returns:
        一个与 ``visibility_record`` schema 匹配的独立字典，可直接插入
        ``visibility_frame`` 的 ``actor_visibility`` 字段。
    """
    _guard_2d_integer(normal_semantic)
    _guard_2d_integer(target_only_semantic)

    amodal = count_semantic_pixels(target_only_semantic, target_semantic_id)
    visible = count_semantic_pixels(normal_semantic, target_semantic_id)
    touches = detect_border_touch(target_only_semantic, target_semantic_id)
    bbox = compute_bbox(target_only_semantic, target_semantic_id)
    occluders = detect_occluders(
        normal_semantic,
        target_only_semantic,
        target_semantic_id,
        actor_semantic_map=actor_semantic_map,
        furniture_semantic_map=furniture_semantic_map,
    )

    record = make_visibility_record(
        amodal,
        visible,
        in_frustum,
        touches_frame_border=touches,
        bbox_visible=bbox,
        occluders=occluders,
        clear_threshold=clear_threshold,
        visible_threshold=visible_threshold,
    )
    return record.as_dict()


def analyze_all_frames(
    normal_semantics: Sequence[np.ndarray],
    target_only_semantics: Sequence[np.ndarray],
    target_semantic_id: int,
    *,
    actor_id: str = "",
    in_frustums: Sequence[bool] | None = None,
    actor_semantic_map: dict[int, str] | None = None,
    furniture_semantic_map: dict[int, tuple[str, str]] | None = None,
    clear_threshold: float = DEFAULT_CLEAR_THRESHOLD,
    visible_threshold: float = DEFAULT_VISIBLE_THRESHOLD,
) -> list[dict[str, Any]]:
    """批量分析所有帧，返回逐帧可见性字典。

    Args:
        normal_semantics: 正常 pass 语义数组的有序序列，每帧一个。
        target_only_semantics: 目标专用 pass 语义数组的有序序列，每帧一个。
        target_semantic_id: 目标角色的语义 ID。
        actor_id: 在 ``actor_visibility`` 中用作 key 的稳定角色 ID。
            若为空，则回退为 ``str(target_semantic_id)``。
        in_frustums: 逐帧视锥体标志。默认全部为 ``True``。
        actor_semantic_map: ``{semantic_id: actor_id}`` 已知角色映射。
        furniture_semantic_map: ``{semantic_id: (instance_id, label)}``
            具备可靠语义映射的家具。
        clear_threshold: 高于此比例判定为 ``visible_clear``。
        visible_threshold: 低于此比例判定为 ``fully_occluded``。

    Returns:
        逐帧字典列表，可直接用于 ``Episode.set_visibility_frames()``。
    """
    n = len(normal_semantics)
    if len(target_only_semantics) != n:
        raise ValueError(
            f"normal_semantics 长度 ({n}) 与 target_only_semantics 长度 "
            f"({len(target_only_semantics)}) 不一致"
        )

    if in_frustums is None:
        in_frustums = [True] * n
    elif len(in_frustums) != n:
        raise ValueError(
            f"in_frustums 长度 ({len(in_frustums)}) 与帧数 ({n}) 不一致"
        )

    resolved_actor_id = actor_id if actor_id else str(target_semantic_id)

    frames: list[dict[str, Any]] = []
    for fi in range(n):
        record = analyze_frame(
            normal_semantics[fi],
            target_only_semantics[fi],
            target_semantic_id,
            in_frustum=in_frustums[fi],
            actor_semantic_map=actor_semantic_map,
            furniture_semantic_map=furniture_semantic_map,
            clear_threshold=clear_threshold,
            visible_threshold=visible_threshold,
        )
        frames.append({
            "frame_index": fi,
            "actor_visibility": {resolved_actor_id: record},
        })

    return frames


# ═══════════════════════════════════════════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════════════════════════════════════════


def _guard_2d_integer(array: np.ndarray) -> None:
    """输入校验：确保为二维 numpy 整数数组。"""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"期望 numpy 数组，实际类型为 {type(array).__name__}")
    if array.ndim != 2:
        raise ValueError(f"期望二维数组 (H, W)，实际 ndim={array.ndim}")


__all__ = [
    "analyze_all_frames",
    "analyze_frame",
    "compute_bbox",
    "count_semantic_pixels",
    "detect_border_touch",
    "detect_occluders",
]
