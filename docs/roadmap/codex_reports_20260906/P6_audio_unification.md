# P6 audio unification

## 1. 改了哪些文件（路径），提交号

- `src/avengine/timeline/current_mp3d_dynamic_audio.py`
  - 增加统一的 `render_neutral_readback_audio(...)` 入口，消费 P1 `NeutralReadback`。
  - 把干声 bus、动态 RIR、双耳渲染、float32 WAV 输出和回执收口到同一实现。
  - 支持事件级 dry binding、P7 source activity 区间的 `crop_start_sample` 换算、HRTF、绕射开关/阶数和旧双槽 RIR cache 序列适配。
- `tools/acoustics/render_frame_readback_sequential_speech.py`
  - UE 计划入口改为 UE readback → `neutral_from_ue_readbacks` → 共享入口；旧无 `audio_plan` 路径保留。
  - 无 `sampling_policy` 的旧计划如果带真实 per-frame listener motion，继续走保留的 legacy dynamic helper；`conditioned_static_v2` 计划发现 listener motion 会显式失败。
  - `--neutral-readback`、`--prepared-manifest`、`--diffraction`、`--max-diffraction-order` 已接入。
  - `_normalize_plan_events` 现在让事件级 binding/事件显式 path 优先于 actor-level binding。
- `src/avengine/cli.py`
  - M5 `render-current-mp3d-dynamic-audio` 增加 neutral、显式外部 dry binding、prepared manifest 和绕射参数，并调用共享入口。
  - `--beagle-audio` 仍只作为历史 `dog_beagle_v2_scheduled_dry` 绑定；其他声音使用显式 `--asset-binding` 或 sound registry。
- `src/avengine/rooms/qa_delivery.py`
  - `build_audio_command` 透传 neutral、prepared、diffraction 和 `max_diffraction_order` 选项；未修改 `finalize`。
- `src/avengine/timeline/unified_audio_receipt.py`
  - 新增 `validate_unified_audio_receipt(...)`，调用 P1 `validate_clock` 检查完整时钟，要求主 unified 输出为真实 binaural 2ch left/right WAV；FOA 仍可作为附加布局。
- `docs/TOOL_INDEX.md`
  - 按仓库规则由 `tools/build_tool_index.py` 重生成，因为 P6 修改了 `tools/` 下入口。
- `tests/unit/test_p6_audio_unification.py`
  - 新增共享回执、P7 crop offset、事件级 voice binding 和 CLI/build command 测试。
- `docs/roadmap/codex_reports_20260906/P6_audio_unification.md`
  - 本报告。

实现提交为本报告首次加入当前分支的提交，可用 git log -1 -- docs/roadmap/codex_reports_20260906/P6_audio_unification.md 定位。父 Agent 在权威服务器审阅并统一提交；未 push 或 merge。

## 2. 跑了哪些测试，各自的通过、失败、跳过计数

最终命令使用：

```text
PYTHONPATH=src:tmp/native_python_addons_v1 /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
```

- 相关测试合计：**105 passed, 0 failed, 0 skipped**（父 Agent 使用项目 Python 与 `PYTHONPATH=src:tmp/native_python_addons_v1` 的 canonical run；完整输出见 `tmp/p6_audio_20260907/parent_canonical_tests.log`）。
  - `tests/unit/test_p6_audio_unification.py`
  - `tests/unit/test_current_mp3d_dynamic_audio.py`
  - `tests/unit/test_frame_readback_dynamic_audio.py`
  - `tests/unit/test_capture_acoustics.py`
  - `tests/unit/test_rooms_semantic_cached_audio.py`
  - `tests/unit/test_timeline_audio_render.py`
  - `tests/unit/test_timeline_cli.py`
  - `tests/unit/test_spatial_audio_cli.py`
  - `tests/unit/test_capture_delivery.py`
  - `tests/unit/test_tool_index_current.py`
- 目标文件 `py_compile`：通过。
- `tools/build_tool_index.py` 已按仓库规则重生成 `docs/TOOL_INDEX.md`；`test_tool_index_current.py` 通过。
- 目标文件 `git diff --check`：通过。
- 新增 validator 对 A 过渡、A 合一、MP3D、gain=1.0、gain=0.15 五份真实回执逐份检查：全部通过。

Follow-up regression：

- `tests/unit/test_p6_audio_unification.py`：**5 passed**，包含真实变化的 per-frame listener readback、旧计划 dynamic dispatch、conditioned static motion fail-closed、P1 clock mismatch、FOA-only、实际 mono WAV 拒绝，以及 shared receipt 的 source-clip/sentence 语义。
- `tests/unit/test_p6_audio_unification.py tests/unit/test_qa_production_contracts.py tests/unit/test_capture_acoustics.py`：**37 passed**。
- canonical 相关列表由父 Agent 使用项目 Python 与 `PYTHONPATH=src:tmp/native_python_addons_v1`：**105 passed**，无失败。

保留的失败现场与修复：

- 初次 A 命令误把 `--package-manifest` 指向 `episode_plan.json`，声学包契约正确拒绝；随后改为真实 RLR package manifest。
- MP3D 首次使用 P2 的 binaural simulation request，`current M1 pair IR requires the existing simulation request to declare ambisonics/4`；改用仓库 ambisonics profile，并显式 `--no-diffraction --max-diffraction-order 0`。
- 两次独立 MP3D native RLR 的 gain=1/.15 湿声比例不成立，因为两次随机射线产生了不同 RIR；`tmp/p6_audio_20260907/gain_ratio_compare_v1.json` 保留该诊断。最终对照先捕获一份 RIR，再复用同一序列渲染两种增益。

## 3. 验收产物路径和核对结果

A 房使用 P5 fresh 捕获 `tmp/p5_sampler_20260906_v1/a_plan_v3/capture_retry_v2` 的 NeutralReadback、P7 prepared 事件和 A 房声学包：

- 过渡入口更正回执：`tmp/p6_audio_20260907/a_transition_p5_v3_r15/research_report.json`
- 共享入口更正回执：`tmp/p6_audio_20260907/a_unified_p5_v3_r15/research_report.json`
- 原始 `research_receipt.json` 保留不覆盖；更正 JSON 仅修复句子完整性声明并复用原媒体。
- 逐样本对照：`tmp/p6_audio_20260907/a_transition_unified_compare_v1.json`
- mixture、两个 binaural stems、两个 dry buses 全部 `byte_equal=true`，`different_samples=0`，`max_abs_diff=0.0`。
- 240 帧、15 Hz、256000 samples；两事件的 source activity 使用 episode sample clock，湿尾区间独立记录。

MP3D 使用 P4 `tmp/p4_beagle_speaker_parent_registry_only_v1/neutral_readback.json`、MP3D 声学包和一个显式 beagle+speaker AudioProgram：

- M5 共享入口更正回执：`tmp/p6_audio_20260907/mp3d_unified_v3/research_report.json`
- 30 帧、15 Hz、32000 samples；`beagle_0_muzzle` 与 `speaker_muzzle` 各有 stem 和 mixture。
- 更正回执保留 `clock`、`audio_program`、`mixture_path`、`stems`、逐事件 `wet_tail_interval(s)`、逐事件和批级 `peak_dbfs`、一次性 gain proof、HRTF id/SHA、`propagation.diffraction` 与 `max_diffraction_order`、`input_neutral_readback`，并将 `complete_sentences_preserved=null`、`sentence_preservation.status=not_assessed`、`source_clip_preserved=true` 分开记录。
- speaker 事件接入 P7 prepared manifest 后，`source_crop_start_sample=14720` 已从原始 source coordinates 中扣除；输出区间落在 episode sample clock，未把 source activity 当 listener audibility。

Gain acceptance：

- gain=1.0 更正回执：`tmp/p6_audio_20260907/mp3d_gain1_same_rir_v1/research_report.json`
- gain=0.15 更正回执：`tmp/p6_audio_20260907/mp3d_gain015_same_rir_v1/research_report.json`
- 对照：`tmp/p6_audio_20260907/gain_ratio_compare_v2.json`
- 在同一 neutral、同一 AudioProgram 时序、同一 RIR 序列下，mixture、两个 stems 和两个 dry buses 的中位逐样本比例均为 `6.6666666667`；最大相对比例误差约 `9.93e-8`。
- 输出为 IEEE float32 WAVE；本批记录的最大绝对 float64→float32 编码误差约 `7.19e-9`。五份最终可用更正 JSON 均通过 validator；原始回执和 PCM/媒体保持不变。

## 4. 没做完的部分和原因

- `evidence_missing_or_unsampled`：没有把 source activity 说成 listener audibility；听者实际可听性仍需基于 wet stem/人工听音单独核对。
- `interface_not_implemented`：P6 只打开了两条指定音频入口的共享渲染接口，未扩展其他房间或资产绑定矩阵；MP3D 这次使用显式 direct dry bindings，正式 sound registry 录入仍由后续任务决定。
- `not_applicable_by_definition`：P6 没有新增题义不适用判断。
- 共享 M5.1 neutral renderer 当前要求 listener readback 静止；带 `conditioned_static_v2` 的新计划继续执行这个约束。无策略字段的旧 UE 计划保留旧 dynamic keyframe helper，因此真实移动 listener 仍按旧 request path 渲染并在回执中记录 per-keyframe listener poses。
- 两次独立 native RLR 不保证数值 RIR 相同，因此 gain 比例验收使用“同一已捕获 RIR 序列、只改变 event gain”的明确对照条件。
- 没有人工试听、模型评估、模态必要性实验或正式 dataset admission；所有回执仍是 research-only。

## 5. 对后续接口的要求

- 共享函数：`avengine.timeline.current_mp3d_dynamic_audio.render_neutral_readback_audio(neutral_readback, *, audio_program_path|audio_program, source_endpoint_by_entity, simulation_request_path|simulation_mapping, package_manifest_path, ... )`。
- Neutral 输入必须通过 P1 `validate_neutral_readback`，单位为 meter、+Y up、right-handed；实体 emitter 绑定必须显式来自 `entity_identities` 或 `source_endpoint_by_entity`。
- AudioProgram 候选 endpoint 必须覆盖 neutral source endpoint；事件 gain 由 `assemble_dry_audio_buses` 应用一次，卷积阶段固定使用 `post_assembly_convolution_gain=1.0`。
- `source_activity_intervals_samples` 是 episode sample clock 的半开区间；prepared manifest 的 original source intervals 必须先减 `source_crop_start_sample`。
- 统一回执 schema 为 `avengine_unified_audio_receipt_v1`。旧 `rir`/`dynamic_rir` 兼容别名、`audio_program_path` 和 `legacy_schema` 暂时保留。shared neutral/P7 回执只能声明 `source_clip_preserved`；`complete_sentences_preserved` 为 `null` 且 `sentence_preservation.status=not_assessed`，历史直接入口继续保留既有字段。
- P9 若调用 `finalize_qa_episode`，应读取回执的 `audio_program.path` 或 `audio_program_path`；P6 没有修改 P9 的 `finalize`。
- P7 prepared manifest 可通过 `prepared_manifest_path` 传入；P5/控制器可通过 request 或 plan 传入 `neutral_readback`、`prepared_manifest`、`diffraction` 和 `max_diffraction_order`。

## 6. 与总前提冲突、需要 owner 拍板的地方

- 没有自行改变 P2 endpoint 表、零干声静音端点、双槽 cache 格式或默认不加 ambient bed 的决定。
- 本次真实 MP3D speaker 音频使用显式 direct binding，未把它写成按 asset 名字的生产覆盖；是否将该声音/endpoint 进入正式 registry 由 owner 后续决定。
- HRTF 使用现有 `mit_kemar_normal_pinna_16k.sofa`，绕射验收使用显式 false/order 0；是否启用绕射及采用哪个 max order 仍由 owner/校准任务决定。
- 生成的内容保持 `research_only`、`qualification_claim=false`；正式 admission 和人工可答性不由 P6 决定。


Parent integration follow-up (HM3D native validation): the explicit NeutralReadback branch no longer labels position_authority as UE when the supplied readback came from Habitat. It reports P1 NeutralReadback entities[].emitter; producer and input_neutral_readback retain the actual source. Corrected metadata is in tmp/p4_hm3d_animal_speaker_20260907_v5/audio_endpoint_bound/research_report_metadata_corrected_v1.json, alongside untouched receipt and PCM. This is a label correction, not an audio re-render; the common validator with require_files=True passes.

Metadata follow-up validation: test_p6_audio_unification plus test_tool_index_current — 6 passed, 0 failed, 0 skipped, 2.27 seconds; tool index regenerated.


Neutral-only entry follow-up: --frame-readbacks is now optional when --neutral-readback and --audio-plan are supplied. The adapter validates and consumes that actual NeutralReadback directly, so Habitat callers no longer need a fabricated legacy length-only JSON. Existing UE-only requests retain the original path and moving-listener behavior; the explicit neutral renderer remains static-listener only. Both position and listener authority labels name P1 NeutralReadback. Related P6, dynamic UE, acoustic and index tests: 28 passed, 0 failed, 0 skipped, 4.00 seconds, log tmp/p6_neutral_only_input_tests_20260907_v1.log. The final HM3D label correction is research_report_metadata_corrected_v2.json in the same audio_endpoint_bound folder; original receipt and all media are unchanged.
