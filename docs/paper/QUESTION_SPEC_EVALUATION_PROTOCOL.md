# QuestionSpec 论文评测协议

这套协议把两件容易混淆的事分开：当前 `QuestionSpec` 的稳定 API 编号，以及 2026-08-07 原任务的语义顺序。当前 12 类编号不重排；其中 `appearance_to_spoken_content` 仍是 QS-012，但语义来源是 0807 原始第 2 类。真正的 0807 后扩展只有 QS-009 `reappeared_after_occlusion` 和 QS-011 `became_clear_after_partial_occlusion`。

机器可读定义在 `examples/qa/question_spec_paper_protocol_v1.json`。它逐类固化：定义、可回答性、GT authority、正/负例、0807 来源和论文答案平衡要求。`examples/qa/native_question_episode_catalog_v1.json` 只登记已存在的 native Episode、Facts、registries、event bindings、binding manifest 和六类原生证据角色。

## 评测边界

- 每个候选问题都从 Episode 已观测的实例、外貌值、声音绑定和帧号枚举；不会生成未出现的 selector。
- 编译器始终调用当前 `evaluate_question_specs` 重算。历史 `question_evaluations.json` 只属于旧交付，不作为 GT 或覆盖计数。
- 只有 manifest、Facts 和六类 native 文件均通过完整性校验的 Episode 才能贡献覆盖。
- `minimum_protocol_status` 表示 12 类各至少一个 native pass，保持 0807 首批“每类一个可靠样本”的边界。
- `paper_balance_status` 是更严格的答案分布门。例如二元问题需要 yes/no，空间方向需要 left/right，像素遮挡需要四态。最低门通过不代表论文平衡门通过。
- 无法唯一回答、不可观察、前方死区、运动状态在发声窗内变化、缺像素真值或缺原生遮挡者 ID 时，必须 rejected/unsupported，不能猜。

## 五类像素 canary

五类 canary 是：fully visible、家具 partial、in-view fully occluded、fully out、移动 Listener 时 full occlusion→reappearance。每类必须同时具备 native Episode、normal object-id mask、target-only mask、pixel truth、runtime readback、RGB/双耳媒体、metric depth、由原生 mask 派生的可读 overlay，以及至少一个由当前代码重算的通过 QA。

Overlay 仅用于人审：绿色是 normal pass 中可见目标像素，红色是 target-only 中存在但 normal pass 中被遮挡的目标足迹。它不会成为新的 GT authority。

## 可重复执行

2026-08-17：官方 `compile` 在 `_load_native_episode` 被
`source_asset_runtime_profiles.json` 的 size/sha 锁挡住，还不能发布
`paper_ready_v3`。修复任务、禁止改 hash 过门、以及已完成的 RGB overlay
提交见
[`docs/qa/QUESTION_PROTOCOL_RECOMPILE_BLOCKER_20260817.md`](../qa/QUESTION_PROTOCOL_RECOMPILE_BLOCKER_20260817.md)。

```bash
/data/jzy/.local/bin/uv run python tools/qa/compile_question_protocol_coverage.py compile \
  --output tmp/lead_a_question_protocol_v1

/data/jzy/.local/bin/uv run python tools/qa/compile_question_protocol_coverage.py validate \
  --input tmp/lead_a_question_protocol_v1
```

编译采用原子、no-clobber 发布。再次运行时应使用新的输出目录；验证器只读取已发布文件并检查清单。若要把“答案平衡完成”作为失败门，追加 `--require-paper-ready`。

输出包括 `coverage.json`、`report.md`、`protocol_snapshot.json`、五张 canary overlay 和自校验 `manifest.json`。报告会列出每类的当前 native 覆盖、观测答案、论文平衡 gap，以及每个 gap 所需的最小精确 native 场景；不会用伪造 Facts 填空。
