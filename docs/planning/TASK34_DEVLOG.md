# 任务 3.4 + 3.5：十类问题管线 + 五个可视化 Canary — 开发日志

## 概述

在 Task 3（QuestionSpec 管线，支持 3 种 answer_modality）基础上，扩展至十类问题类型，
并为 Person A 构建五个可视化 canary 作为首批验收。

## 文件清单

| 动作 | 文件 | 说明 |
|------|------|------|
| EDIT | `src/avengine/qa/question_spec.py` | 新增 7 种 answer_modality |
| EDIT | `src/avengine/qa/answer_deriver.py` | 新增 9 个推导函数 + 双角色支持 + 完整可观察性检查 |
| CREATE | `src/avengine/qa/question_catalog.py` | 十类问题 QuestionSpec 模板常量 |
| EDIT | `src/avengine/qa/__init__.py` | 新增 catalog 导出 |
| CREATE | `tests/unit/test_answer_deriver_extended.py` | 58 条新模态单元测试 |
| CREATE | `tests/unit/test_question_catalog.py` | 31 条 catalog 构造校验 |
| CREATE | `tests/acceptance/test_task34_canaries.py` | 20 条 canary 验收测试 |
| CREATE | `docs/planning/TASK34_DEVLOG.md` | 本文件 |

## 十类问题类型

| # | 类型 | spec_id | answer_modality | 答案返回 | 双角色 |
|---|------|---------|-----------------|---------|--------|
| 1 | 外貌 → 是否发声 | `qs_sound_presence_v1` | `sound_facts` | "是"/"否" | — |
| 2 | 外貌 → 发声内容 | `qs_transcript_v1` | `sound_transcript` | transcript 文本 | — |
| 3 | 发声内容 → 外貌属性 | `qs_transcript_to_attr_v1` | `reverse_attribute` | 属性值 | — |
| 4 | 发声者空间方向 | `qs_spatial_direction_v1` | `spatial_direction` | "左侧"/"右侧"/"正前方" | — |
| 5 | 发声先后顺序 | `qs_sound_order_v1` | `sound_order` | "{actor}先说话" | ✓ |
| 6 | 重叠发声 | `qs_sound_overlap_v1` | `overlap_sound` | "是"/"否" | ✓ |
| 7 | 发声时运动状态 | `qs_sound_motion_v1` | `sound_motion` | "静止"/"走" | — |
| 8 | 画外→入画方向 | `qs_enter_frustum_v1` | `enter_frustum_direction` | "左侧"/"右侧"/"正前方" | — |
| 9 | 发声时遮挡状态 | `qs_sound_visibility_v1` | `sound_visibility` | "清晰可见"/"部分遮挡"/"完全遮挡"/"画外" | — |
| 10 | 遮挡者识别 | `qs_occluder_identity_v1` | `occluder_identity` | "家具（桌子）"/"另一只动物" | — |

## 关键设计决策

### 1. answer_modality 分发

所有答案推导集中在 `derive_answer()` 函数中，通过 `answer_modality` 字符串分发到 10 个分支。
每个分支调用专用的私有推导函数，减少耦合。

### 2. 空间方向判断

使用 `spatial_facts.per_frame[].actors[actor_id].listener_relative.azimuth_deg`：
- azimuth_deg < -30° → "左侧"
- azimuth_deg > 30° → "右侧"
- else → "正前方"

选择声音事件中点帧的方位数据。

### 3. 双角色问题

类型 5（sound_order）和 6（overlap_sound）使用 `required_actor_count=2`：
- `_derive_sound_order()`：扫描 doc 中所有声音事实，比较最早 start_tick
- `_derive_overlap_sound()`：检查两个角色的声音区间是否有时间交集
- 可观察性：至少两个角色有声音事件

### 4. 无像素真值保护

关键约束：没有像素数据时禁止产生入画、可见性或遮挡问题。
在 `check_fact_observable()` 中通过以下检查实施：
- `enter_frustum_direction`：需要 `_has_pixel_data() and _has_events()`
- `sound_visibility`：需要 `_has_pixel_data() and _has_visibility_frames()`
- `occluder_identity`：需要 `_has_pixel_data() and _has_visibility_frames()`
- `_has_pixel_data()` 要求至少一帧有 `amodal_pixels > 0`

### 5. 五个 Canary

| # | 场景 | 帧数 | 关键特征 |
|---|------|------|---------|
| C1 | 完全可见 | 5 | visible_clear on all frames, no occluders |
| C2 | 家具部分遮挡 | 5 | 家具语义 ID=50 覆盖目标上半，visible_occluded + furniture occluder |
| C3 | 完全遮挡 | 5 | 另一角色 ID=20 全覆盖，fully_occluded，声音不可观察 |
| C4 | 画外入画 | 5 | 帧 0-2 out_of_view，帧 3 enter_frustum，进入方向"左侧" |
| C5 | 遮挡重现 | 5 | 帧 0 visible_clear，1-3 fully_occluded，4 visible_clear (reappear) |

每个 canary 使用合成 numpy 语义分割数组 (64×64)，运行 `analyze_all_frames()` 生成可见性数据，
然后通过 `derive_answer()` / `check_fact_observable()` 验证。

## 验证

```bash
# 新模态单元测试
python -m pytest tests/unit/test_answer_deriver_extended.py -v
# 58 passed

# Catalog 构造校验
python -m pytest tests/unit/test_question_catalog.py -v
# 31 passed

# Canary 验收
python -m pytest tests/acceptance/test_task34_canaries.py -v
# 20 passed

# 全量回归
python -m pytest tests/ -v
# 2348 passed, 1 skipped
```

## 开发时间

- answer_deriver.py 重写：~2 小时（~500 行，10 种模态分支）
- question_catalog.py：~0.5 小时
- 单元测试（89 条）：~2 小时
- Canary 生成与验证：~2 小时
- 缺陷修复与调试：~1 小时
- 合计：~7.5 小时
