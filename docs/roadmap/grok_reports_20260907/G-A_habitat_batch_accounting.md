# G-A Habitat 收口与批次记账

日期：2026-09-07。分支：`grok/pilot46-fixes-20260907-ga`。对照任务书第 2 节 G-A、第 3 节 G-A 验收、第 5 节报告格式。

## 1. 改了哪些文件（路径），提交号

实现提交 `cf2c8d5`。本项改动：

- `src/avengine/rooms/qa_delivery.py`：`select_habitat_audio_program_mode`（按重叠/活跃端点数选模式，不用 `plan.audio_mode`）；契约与双耳 WAV 验过再写 facts/questions；`--beagle-audio` 只在真有 beagle 绑定时传。
- `src/avengine/cli.py`：`--beagle-audio` 改为可选。
- `src/avengine/timeline/unified_audio_receipt.py`：湿尾与 source_activity 一律不得越出 `clock.sample_count`。
- `src/avengine/timeline/current_mp3d_dynamic_audio.py` 与 `tools/acoustics/render_frame_readback_sequential_speech.py`：湿尾截到时钟内并记录被截。
- `tools/dataset/run_qa_batch.py`：失败写入 `failure_stage` / `failure_reason` / `gap_state`。
- `src/avengine/qa/batch_coverage.py`：失败段按 outcome 的五态入表，不再一律 `asset_not_in_episode`。
- `src/avengine/qa/batch_delivery.py`：try/except 接入 `apply_exposure_gate`（模块由 G-B 提供）。
- 对应单测：`tests/unit/test_p9_evidence_delivery.py`、`test_p6_audio_unification.py`、`test_qa_batch_coverage.py`、`test_qa_batch_runner.py`、`test_current_mp3d_dynamic_audio.py`。

未改：`src/avengine/timeline/audio_program.py` 校验器、Claude 审计器、Codex 报告、原批 `attempt_01`。

## 2. 跑了哪些测试

worktree `/data/jzy/tmp/wt-grok-pilot46-GA`，`import avengine` 解析到本树。

| 文件 | passed | failed | skipped |
| --- | --- | --- | --- |
| 上列 6 个 unit 文件 + `test_tool_index_current.py` | 92 | 0 | 0 |

覆盖：新旧 AudioProgram 模式各一条真实校验；失败记账三类；契约失败不留 facts/questions；湿尾越界拒绝；`--beagle-audio` 可选。

## 3. 验收产物与亲自核对

新输出根（原批只读，capture/plan 为 attempt_01 的符号链接）：

`/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_pilot46_ga_refinalize_20260907/`

| episode | AudioProgram mode | 契约 | facts/questions | 双耳 WAV |
| --- | --- | --- | --- | --- |
| mp3d_human_animal | simultaneous_subset（2 事件重叠） | pass | 有 | 256000×2 @16 kHz = 16.0 s |
| mp3d_single_active | one_active_of_n（1 活跃端点，2 候选） | pass | 有 | 16.0 s |
| hm3d_human_animal | simultaneous_subset | pass | 有 | 16.0 s |
| hm3d_single_active | one_active_of_n | pass | 有 | 16.0 s |

四段 `commands.json` 均无 `--beagle-audio`。原批 `attempt_01` 仍在。

## 4. 没做完的部分

- **题义不适用**：无。
- **接口未实现**：曝光闸门本体在 G-B 的 `exposure_gate.py`；G-A 只留了 import hook。合并前单独跑 G-A 树时闸门是 no-op。
- **证据缺失**：无。四段都出了 facts/questions 与 16 秒双耳。

## 5. 对后续接口的要求

- `select_habitat_audio_program_mode(events, candidate_ids) -> str`
- outcome.json：`failure_stage` ∈ {planning, capture, audio, finalize, launch}；`gap_state` ∈ 五态；`failure_reason` 为错误首行/直方图。
- 代码缺陷 → `interface_not_implemented`；200 次用尽 / 预分配缺额 → `evidence_missing_or_unsampled`。
- `apply_exposure_gate(review, episode_root)` 由 G-B 提供。

## 6. 需要 owner 拍板

无。Habitat 模式选择完全按 UE 路径规则，未改校验器、未改阈值。
