# H-2 合并、记账与分类

日期：2026-09-07。分支：`grok/pilot46-fixes-round2-20260907-h2`。对照第二轮任务书 H-2 与审核 R2、R3、R7、K1。

## 1. 改了哪些文件与提交号

代码提交：`6162efa1462745a7c6a361dad538642ee71b34d6`

- 新增 `tools/dataset/merge_qa_batch_attempts.py`（原 `/tmp/ge_merge_final.py` 入库）
- 新增 `tests/unit/test_merge_qa_batch_attempts.py`
- `tools/dataset/run_qa_batch.py`：`_looks_like_interface_defect` 改为按失败阶段与异常类型的 `gap_state_for_failure`
- `src/avengine/qa/batch_coverage.py`：Codex 工作树 inventory/catalog 路径改写到生产工作树；`failed_episodes` 原逻辑保留
- `src/avengine/qa/batch_manifest.py`：`collect_batch_outcomes` 只补 `failure_stage` / `gap_state`（未改 scatter / sound_pool）
- `tests/unit/test_qa_batch_runner.py`、`tests/unit/test_qa_batch_coverage.py`
- `docs/TOOL_INDEX.md`（`tools/build_tool_index.py` 重生成）

未改：`batch_delivery.py`（现场 finalize 已会传 `failed_episodes`，缺的是合并脚本漏传与旧记录空键）；未改 H-1/H-3/H-4/H-5/H-7 文件；第一轮产物目录只读。

## 2. 测试计数与失败原文

命令（CPU，`PYTHONPATH=src:tmp/native_python_addons_v1`，`import avengine` → `/data/jzy/tmp/wt-grok-pilot46-H2/src/avengine/...`）：

```
pytest -q \
  tests/unit/test_merge_qa_batch_attempts.py \
  tests/unit/test_qa_batch_runner.py \
  tests/unit/test_qa_batch_coverage.py \
  tests/unit/test_qa_batch_manifest.py \
  tests/unit/test_qa_batch_delivery.py \
  tests/unit/test_tool_index_current.py
```

**55 passed / 0 failed / 0 skipped**（2.75 s）。分文件：

| 文件 | 收集数 |
|---|---|
| `tests/unit/test_merge_qa_batch_attempts.py` | 6 |
| `tests/unit/test_qa_batch_runner.py` | 15 |
| `tests/unit/test_qa_batch_coverage.py` | 14 |
| `tests/unit/test_qa_batch_manifest.py` | 16 |
| `tests/unit/test_qa_batch_delivery.py` | 3 |
| `tests/unit/test_tool_index_current.py` | 1 |

三类分类单测（规划用尽 / 预分配缺额 / 捕获代码异常）均在 `test_merge_qa_batch_attempts.py`：`test_gap_state_rules_cover_three_classes`、`test_classify_controller_failure_three_classes`。无失败原文。

## 3. 验收产物路径与亲自核对

新合并目录（第一轮 `qa_pilot46_merged_20260907_v1` 未写）：

`tmp/qa_pilot46_merged_round2_20260907_v1/`
（绝对路径 `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_pilot46_merged_round2_20260907_v1/`）

从提交状态复现命令：

```
python tools/dataset/merge_qa_batch_attempts.py \
  --original tmp/qa_pilot46_background_20260907_v1 \
  --rerun tmp/qa_pilot46_rerun_20260907_v1 \
  --output tmp/qa_pilot46_merged_round2_20260907_v1 \
  --repository /data/jzy/tmp/wt-grok-pilot46-H2 \
  --manifest tmp/p10_pilot46_manifest_20260907_v3/batch_manifest.json
```

亲自核对：

- 46 格：`merged_episodes.json` `episode_denominator=46`，状态 42 delivered / 2 blocked / 2 failed；曝光闸门 42 pass。
- `authored_c_animal_device`（attempt_01）：`failure_stage=planning`，`gap_state=evidence_missing_or_unsampled`，`reason_code=planning_exhausted`，原因含 `fixed condition profile exhausted` 且直方图 `camera:no_joint_geometry_activity_schedule=169`、`routes:initial_source_separation_below_0.95_m=31`（来自 `attempt_01/stderr.log` 与 `episode/planning_result.json`，不是 `interface_not_implemented`）。`room_id=authored_open_family_home_room_c_v1`，资产为柴犬 + charcoal 音箱（来自 manifest `source_assignments`）。
- 两条 blocked：`authored_a_device_device`（`aea_loc3_social_rebuild_v1`，马桶+壁炉）、`kujiale_device_device`（`kujiale_0020_full_home_v1`，搅拌机+打印机）。`failure_stage=planning`，`gap_state=evidence_missing_or_unsampled`，`reason_code=preallocation_gap`。stderr 原文 `preallocation_gap: manifest row has a known preallocation gap`。
- `authored_b_animal_device` attempt_02 仍为规划用尽（172/28），并补上 `room_id` / `asset_ids`。
- 覆盖：`coverage_inputs.json` 顶层 `failed_episodes` 4 条；`coverage.json` 9912 行中 **166** 行带 `failed_episode`（第一轮合并表同口径 0 行）。四段未交付均出现。状态计数仍为 produced 246 / deferred 976 / N/A 1204 / interface 1260 / evidence 6226（失败格本就是证据缺失，只改了记账，没改五态分母）。
- `coverage/provenance.json` 的 inventory/catalog/runtime 均为 `/data/jzy/tmp/wt-grok-pilot46-H2/examples/...`，不含 `/data/jzy/tmp/wt-multi-home-activity-integration`。
- 第一轮产物 mtime 未变：`qa_pilot46_background_20260907_v1/outcomes.json` 仍 13:44，`qa_pilot46_merged_20260907_v1/merged_episodes.json` 仍 20:09；其 `authored_c` 仍标 `interface_not_implemented`（只读对照）。

## 4. 没做完的（三分法）

- **题义不适用：** 未重渲音频、未改房间包路径校验、未接线像素字段、未改放量打散/声音池（H-1/H-3/H-4/H-5/H-7）。未改第一轮合并表本身（任务书写新目录）。
- **接口未实现：** 覆盖表里 1260 行挂墙/吊顶 `interface_not_implemented` 仍在分母（owner 已接受，不是本项要消掉的）。`collect_batch_outcomes` 仍不写 `requested_source_classes`（H-7）。
- **证据缺失：** 四格未交付仍空（A/酷家乐 device_device 预分配 blocked；B/C animal_device 规划 200 次用尽）。旧批 `summary/batch_outcomes.json` 未就地回填（第一轮产物只读）；新记账器会在以后的 `collect_batch_outcomes` 与合并脚本里填阶段与五态。

## 5. 后续接口要求

- H-3：清单 `room_catalog` / `path_bindings` 写成生产工作树绝对路径，避免再从 Codex 树起步；本项只在覆盖 `provenance` 侧改写。
- H-7：`collect_batch_outcomes` 补 `requested_source_classes` 时不要覆盖本项加上的 `failure_stage` / `gap_state` 键。
- 新的控制器失败应继续写出 `planning_result.json` 直方图或带 `*Error:` 的异常类型，分类器按阶段+异常类型显式规则，不再做关键词袋匹配。

## 6. 需 owner 拍板（与本项无关的已裁定项不重复当阻塞）

本项无新的阻塞拍板。四格空着已在审核 3.3 节接受。若要把旧批 `qa_pilot46_background_20260907_v1/summary/batch_outcomes.json` 原地改写成带阶段的版本，需要 owner 允许改第一轮产物；当前没有改。
