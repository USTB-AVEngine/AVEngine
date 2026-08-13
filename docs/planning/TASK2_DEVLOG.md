# 任务二开发日志 — 像素级可见性与遮挡分析

**日期**：2026-08-11
**负责人**：A（QA 管线）
**分支**：`feature/qa-episode-v1`

## 概述

任务二实现了像素级可见性与遮挡分析，通过对比每帧两路语义分割 pass——
标准多物体正常 pass 和目标专用 pass（相同相机位姿，仅渲染目标角色）。
两者之差可识别遮挡物的身份与遮挡程度，产出可供 Episode 构建器直接使用的
`VisibilityRecord` 数据。

本模块为**纯 Python + numpy**——不依赖 Habitat。渲染由调用方负责。

## 交付物

| 文件 | 说明 | 状态 |
|------|------|------|
| `src/avengine/qa/pixel_visibility.py` | 像素分析模块（~360 行） | ✅ |
| `src/avengine/qa/__init__.py` | 更新导出，增加任务二符号 | ✅ |
| `tests/unit/test_pixel_visibility.py` | 49 条单元测试（6 个测试类） | ✅ |
| `tests/acceptance/test_task2_acceptance.py` | 31 条验收测试（8 个测试类） | ✅ |

## 设计决策

### 1. 双 pass 对比策略

每帧需要同一相机位姿下的两路语义渲染：

- **正常 pass** — 标准多物体语义分割
  → `visible_pixels` = 目标的 `semantic_id` 实际出现的像素数
- **目标专用 pass** — 同一场景，仅渲染目标角色
  → `amodal_pixels` = 目标在无遮挡情况下应有的像素数

差值 `(amodal_pixels - visible_pixels)` 即为被遮挡的像素数。

### 2. 遮挡物识别

在目标专用 pass 中存在、但在正常 pass 中被**其他**语义 ID 占据的像素
即为遮挡像素。这些位置的语义 ID 就是遮挡物。

遮挡物分类：
- **actor**（角色）— 语义 ID 命中 `actor_semantic_map`
- **furniture**（家具）— 语义 ID 命中 `furniture_semantic_map`
- **unknown_static**（未知静态物）— 语义 ID 不在任何映射中（安全兜底，绝不猜测）

结果按遮挡像素数降序排列（最显著的遮挡物排在最前面）。

### 3. 安全兜底原则

未映射的语义 ID 始终归类为 `unknown_static`。系统不对遮挡物身份做任何猜测，
确保未知遮挡物不会在 QA 数据中悄无声息地变成角色或家具。

### 4. 背景（ID 0）与自身过滤

- **语义 ID 0**（背景/未标注）被显式排除在遮挡物结果之外——背景不能作为遮挡物。
- **目标自身的语义 ID** 也被排除，防止自遮挡伪影。

### 5. 复用已有像素操作模式

| 模式 | 来源 |
|------|------|
| `np.count_nonzero(semantic == id)` | `m7/visual_review.py:413` |
| 边缘检测：`mask[edge, :]` / `mask[:, edge]` | `m5_1/mp3d_capture.py:560-563` |
| 包围盒计算：`np.nonzero` | 标准 numpy 惯用法 |

### 6. 与任务一的集成

`analyze_frame()` 调用 `episode.py` 中的 `make_visibility_record()`，
后者内部调用 `classify_visibility()`。这确保了统一的四状态分类
（`out_of_view`、`visible_clear`、`visible_occluded`、`fully_occluded`）
在任何地方都保持一致——像素模块不重复分类逻辑。

`analyze_all_frames()` 返回 `{frame_index, actor_visibility}` 字典列表，
可直接赋给 `Episode.visibility_frames`。

## 任务二期间修复的 bug

### `make_visibility_record` 中的 amodal 兜底逻辑（第 701–702 行）

`episode.py` 中存在一段代码，会在 `amodal_pixels == 0` 且 `visible_pixels > 0`
时将 `amodal_pixels` 悄悄替换为 `visible_pixels`：

```python
if amodal_pixels == 0 and visible_pixels > 0:
    amodal_pixels = visible_pixels  # 避免 0/0；将可见像素视为全量
```

这是错误的，原因如下：
1. `classify_visibility()` 在计算 `visible_pixels / amodal_pixels` 之前
   已经处理了 `amodal_pixels == 0` 的情况（返回 `out_of_view`）。
2. 比例计算本身已有安全除法保护：`visible_pixels / amodal_pixels if amodal_pixels > 0 else 0.0`。
3. 该兜底逻辑会将 `classify_visibility(0, 100, True)` 的结果从
   `out_of_view`（正确）变为 `visible_clear`（错误）——目标在目标专用 pass
   中根本不存在，理应判定为画外。

已删除该兜底逻辑。修复后全部 71 条已有测试（32 条单元 + 39 条验收）通过。

## 模块架构

### `pixel_visibility.py`

```
count_semantic_pixels(semantic, semantic_id) -> int
    统计二维整数数组中匹配指定语义 ID 的像素数。

detect_border_touch(semantic, semantic_id) -> bool
    若目标像素触及上/下/左/右边缘则返回 True。

compute_bbox(semantic, semantic_id) -> tuple | None
    返回 (x_min, y_min, x_max, y_max)，无匹配像素时返回 None。

detect_occluders(normal_semantic, target_only_semantic, target_id, *,
                 actor_semantic_map, furniture_semantic_map) -> list[dict]
    识别并分类所有遮挡目标的物体。按遮挡像素数降序排列。

analyze_frame(normal_semantic, target_only_semantic, target_id, *,
              in_frustum, actor_semantic_map, furniture_semantic_map,
              clear_threshold, visible_threshold) -> dict
    完整单帧分析 → VisibilityRecord 字典。

analyze_all_frames(normal_semantics, target_only_semantics, target_id, *,
                   actor_id, in_frustums, actor_semantic_map,
                   furniture_semantic_map, clear_threshold,
                   visible_threshold) -> list[dict]
    批量处理全部帧 → 可直接赋给 Episode.visibility_frames 的列表。

_guard_2d_integer(array) -> None
    输入校验：必须是二维 numpy 数组。
```

## 测试覆盖

### 单元测试（49 条，6 个测试类）

| 测试类 | 条数 | 关注点 |
|--------|------|--------|
| `TestCountSemanticPixels` | 5 | 全匹配、无匹配、部分匹配、零 ID、非二维拒绝 |
| `TestDetectBorderTouch` | 8 | 上/下/左/右/角、仅中心、零像素、1×1 |
| `TestComputeBbox` | 6 | 单像素、矩形、不规则、分离区域、空、全屏 |
| `TestDetectOccluders` | 10 | 无遮挡、角色、家具、未知、多遮挡排序、混合类型、目标不存在、背景过滤、自身过滤、默认空映射 |
| `TestAnalyzeFrame` | 11 | 四种可见性状态、视锥体内/外、边缘触碰、包围盒、遮挡物、自定义阈值 |
| `TestAnalyzeAllFrames` | 9 | 统一批量、变化可见性、默认视锥体、显式视锥体、角色 ID 回退、长度不匹配、空映射、角色映射 |

### 验收测试（31 条，8 个测试类）

| 测试类 | 条数 | 关注点 |
|--------|------|--------|
| `TestTask2Imports` | 3 | 包级与模块级符号解析 |
| `TestPixelPrimitivesAcceptance` | 4 | 大尺寸数组（50×50、64×64）的正确性 |
| `TestOccluderDetectionAcceptance` | 4 | 多遮挡排序、安全兜底、空结果、背景排除 |
| `TestAnalyzeFrameAcceptance` | 6 | 往返验证、视锥体标志、零 amodal、遮挡物、包围盒、边界值 |
| `TestAnalyzeAllFramesAcceptance` | 4 | 结构正确性、完整 Episode 填充、变化序列、in_frustum 标志 |
| `TestPixelToEpisodeIntegration` | 2 | 含事件检测的管线集成、完整 Episode 构建 + 校验 |
| `TestCustomThresholdsAcceptance` | 4 | 宽松/严苛 clear、自定义 visible、批量阈值 |
| `TestInputValidationAcceptance` | 4 | 非二维、非 ndarray、长度不匹配、视锥体不匹配 |

### 全量回归

```
2106 passed, 1 skipped — 零失败
```

## 接口

### 输入（来自渲染管线，由 C/D 负责）

- `normal_semantic: np.ndarray` — (H, W) int64，标准多物体 pass
- `target_only_semantic: np.ndarray` — (H, W) int64，目标专用 pass
- `target_semantic_id: int` — 目标角色的语义 ID
- `actor_semantic_map: dict[int, str]` — `{semantic_id: actor_id}`
- `furniture_semantic_map: dict[int, tuple[str, str]]` — `{semantic_id: (instance_id, label)}`
- `in_frustums: list[bool]` — 逐帧视锥体相交标志

### 输出（到 Episode 构建器，本模块）

- `analyze_frame()` → `dict`（序列化后的 `visibility_record`）
- `analyze_all_frames()` → `list[dict]`（可直接赋给 `Episode.visibility_frames`）
- 上述输出可传入 `detect_visibility_events()` 进行事件检测
- 事件和帧数据一同传入 `Episode.add_event()` / `Episode.set_visibility_frames()` → `Episode.build()`

## 遇到的问题与解决

1. **`make_visibility_record` 的 amodal 兜底逻辑**（见上文 bug 修复一节）
   删除了在 `amodal_pixels == 0` 时将其替换为 `visible_pixels` 的代码。
   修复后角色在目标专用 pass 中不存在时能正确返回 `out_of_view`。

2. **边界值测试：`fully_occluded` 的阈值边界**
   `classify_visibility` 在下阈值处使用严格不等式：`fraction >= visible_threshold`
   判定为 `visible_occluded`，而非 `fully_occluded`。边界为：
   `floor(amodal * visible_threshold - 1)` 像素 → `fully_occluded`；
   `ceil(amodal * visible_threshold)` 像素 → `visible_occluded`。
   测试注释中现已明确记录此行为。

## 后续任务（任务三及以后）

像素级可见性分析已完成的条件下：
- 任务三：声音事实提取（从音频中检测 onset/offset）
- 任务四：空间事实提取（3D 位置 → 听者相对坐标）
- 任务五：QA 生成管线（模板实例化、干扰项合成）
