# 任务三：接入 QuestionSpec — 开发日志

## 概述

实现 QuestionSpec 模板化问题管线。QuestionSpec 是只规定问题结构和场景需求的不可变模板，
模板变量（如 `{top_color}`）从 Person B 的资产注册表解析，
角色从候选注册表选择，声音从声库选择。
渲染后根据 Episode 事实数据得出答案 —— 若答案不唯一（如两个蓝衣人），
该 Episode 不能产生这道题，需重新采样。

## 文件清单

| 动作 | 文件 | 说明 |
|------|------|------|
| CREATE | `src/avengine/qa/question_spec.py` | QuestionSpec、SceneRequirement、TemplateVariable |
| CREATE | `src/avengine/qa/candidate_selector.py` | 资产选择、注册表协议、FakeRegistry |
| CREATE | `src/avengine/qa/answer_deriver.py` | 答案推导、唯一性/可观察性检查 |
| CREATE | `src/avengine/qa/question_pipeline.py` | 流程编排、重采样循环 |
| EDIT | `src/avengine/qa/__init__.py` | 新增 25 个 Task 3 符号导出 |
| CREATE | `tests/unit/test_question_spec.py` | 28 条 QuestionSpec 单元测试 |
| CREATE | `tests/unit/test_candidate_selector.py` | 15 条候选选择单元测试 |
| CREATE | `tests/unit/test_answer_deriver.py` | 29 条答案推导单元测试 |
| CREATE | `tests/unit/test_question_pipeline.py` | 14 条管线编排单元测试 |
| CREATE | `tests/acceptance/test_task3_acceptance.py` | 16 条端到端验收测试 |

## 架构设计

```
QuestionSpec (不可变模板)
    │
    ▼
extract_scene_requirement()  ──►  SceneRequirement
    │                                   │
    │                                   ▼
    │                         select_candidates()
    │                                   │
    │                                   ▼
    │                            [AssetBinding, ...]
    │                                   │
    │                                   ▼
    │                         Episode 渲染 (C/D 负责)
    │                                   │
    │                                   ▼
    │                         try_derive_from_doc()
    │                              │         │
    │                     ┌────────┘         └────────┐
    │                     ▼                            ▼
    │            derive_answer()              check_answer_unique()
    │            check_fact_observable()
    │                     │
    │                     ▼
    │              (answer, unique, observable)
    │                     │
    │                     ▼
    │              ┌─ 不唯一/不可观察 → None (重试)
    │              └─ 通过 → QAPair
    │
    └──► 若全部失败 → 返回 None (max_attempts 耗尽)
```

### 各模块职责

**question_spec.py** — 核心数据类型：
- `QuestionSpec`：冻结数据类，定义问题模板（spec_id、question_type、template、answer_modality）
- `SceneRequirement`：从模板提取的场景约束（目标属性、唯一性要求、时间窗口）
- `TemplateVariable`：模板中的命名变量，含来源（actor_attr / sound_attr / time）
- `extract_scene_requirement()`：从模板 + 绑定值提取场景需求
- `instantiate_template()`：将变量替换为具体值，生成完整问题文本
- `list_template_variables()`：列出模板中所有变量

**candidate_selector.py** — 资产选择：
- `ActorRegistry` / `SoundRegistry`：Protocol 接口，Person B 实现
- `ActorCandidate` / `SoundCandidate`：候选数据类型
- `AssetBinding`：一组具体的 actor + sound + 属性值组合
- `select_candidates()`：按属性匹配 → 唯一性检查 → 声音配对 → 组装绑定
- `FakeActorRegistry` / `FakeSoundRegistry`：测试用内存实现

**answer_deriver.py** — 答案推导：
- `derive_answer(doc, spec, binding, time_window)`：从 Episode 事实推导答案
- `check_answer_unique()`：扫描场景中所有角色，检查属性值是否唯一
- `check_fact_observable()`：检查目标在时间窗口内是否可观察
- 声音答案："是" / "否"（声音事件与时间窗口交集判断）
- 可见性答案："基本无遮挡" / "部分遮挡" / "完全遮挡" / "不可见"
- 时间窗口参数支持覆盖（管线级 > 模板级）

**question_pipeline.py** — 流程编排：
- `QuestionPipeline`：持有 spec、注册表、属性值、时间窗口
- `iter_batches()`：调用 select_candidates()，跟踪尝试次数
- `try_derive_from_doc(doc, binding)`：推导答案 → 验证唯一性和可观察性 → 返回 QAPair 或 None
- 管线级 `time_window` 覆盖 spec 默认值（用于答案推导和可观察性检查）

## 关键设计决策

1. **FakeRegistry 仅用于测试**：`FakeActorRegistry` / `FakeSoundRegistry` 是内存实现，
   生产环境由 Person B 提供真实注册表实现。

2. **双层唯一性检查**：
   - 候选层：`select_candidates()` 中若同一属性匹配到多个角色 → 返回空
   - 推导层：`check_answer_unique()` 扫描已渲染的 Episode doc 中的所有角色

3. **时间窗口覆盖**：`derive_answer()` 和 `check_fact_observable()` 接受外部 `time_window`
   参数，优先于 spec.time_window。`QuestionPipeline` 通过 `try_derive_from_doc` 传入
   `self.time_window`，使管线级时间窗口生效。

4. **属性值使用中文**：模板实例化直接使用注册表中的属性值。
   Person B 的注册表应存储中文属性值（如 `top_color="蓝色"`），
   以便问题文本直接呈现中文。

## 测试覆盖

### 单元测试（86 条，全部通过）

| 测试文件 | 条数 | 覆盖内容 |
|---------|------|---------|
| test_question_spec.py | 28 | 构造校验、模板变量提取、实例化、SceneRequirement 提取、不可变性 |
| test_candidate_selector.py | 15 | FakeRegistry 操作、候选选择、唯一性拒绝、声音匹配、协议合规 |
| test_answer_deriver.py | 29 | 声音/可见性/运动答案推导、唯一性检查、可观察性、属性提取 |
| test_question_pipeline.py | 14 | 构造、iter_batches、成功/失败路径、模板实例化、重采样 |

### 验收测试（16 条，全部通过）

| 场景 | 条数 | 覆盖 |
|------|------|------|
| 端到端声音存在 | 2 | 蓝衣人发声/不发声 |
| 同属性重复拒绝 | 3 | 候选层拒绝 + 推导层拒绝 + 不同颜色接受 |
| 可观察性 | 2 | 完全遮挡拒绝 + 部分可见接受 |
| 可见性端到端 | 2 | 基本无遮挡 + 完全遮挡 |
| max_attempts 耗尽 | 2 | 零次 + 多次耗尽 |
| 唯一性禁用 | 1 | 多个同色角色通过 |
| 多声音候选 | 1 | 一个角色多个声音 |
| 时间窗口边界 | 2 | 起始点发声 + 窗口前发声 |
| 不同物种 | 1 | 狗的物种问题 |

## 修复的缺陷

1. **`derive_answer` / `check_fact_observable` 忽略管线级时间窗口**：
   原实现仅使用 `spec.time_window`，导致 `QuestionPipeline(time_window=...)` 不会传递给
   答案推导。修复：为两个函数添加 `time_window` 参数，管线传入 `self.time_window`。

## 验证

```bash
# 单元测试
python -m pytest tests/unit/test_question_spec.py \
  tests/unit/test_candidate_selector.py \
  tests/unit/test_answer_deriver.py \
  tests/unit/test_question_pipeline.py -v
# 86 passed

# 验收测试
python -m pytest tests/acceptance/test_task3_acceptance.py -v
# 16 passed

# 全量回归
python -m pytest tests/ -v
# 2208 passed, 1 skipped
```

## 示例用法

```python
from avengine.qa import (
    QuestionSpec, FakeActorRegistry,
    ActorCandidate, QuestionPipeline,
)

# 定义问题模板
spec = QuestionSpec(
    spec_id="sound_presence_v1",
    question_type="sound_presence",
    template="穿{top_color}上衣的人是否在{time_window}发声？",
    answer_modality="sound_facts",
)

# 准备注册表（Person B 提供真实版本）
registry = FakeActorRegistry()
registry.add(ActorCandidate(
    "human_01", "asset_01", "human", 10,
    attributes={"top_color": "蓝色", "species_id": "人"},
))

# 运行管线
pipeline = QuestionPipeline(
    spec=spec,
    actor_registry=registry,
    attribute_values={"top_color": "蓝色"},
    time_window=(0, 96000),  # 0-2 秒
)

for batch in pipeline.iter_batches():
    for binding in batch:
        doc = render_episode(binding)  # Person C/D 负责
        qa = pipeline.try_derive_from_doc(doc, binding)
        if qa is not None:
            print(f"问题: {qa.question_text}")
            print(f"答案: {qa.answer_text}")
            break
    else:
        continue
    break
```

## 开发时间

- 核心模块：~4 小时（4 个 .py 文件，~800 行）
- 单元测试 + 验收测试：~3 小时（5 个测试文件，~800 行）
- 缺陷修复 + 测试调试：~1 小时
- 合计：~8 小时
