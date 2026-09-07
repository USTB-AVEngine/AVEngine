# 放量就绪报告（round-2 / H-6，2026-09-07）

工作树：`/data/jzy/tmp/wt-grok-pilot46-round2`。
分支：`grok/pilot46-fixes-round2-20260907`。
H-6 起草于 `789bc64`；parent 合入 H-1～H-7 后在同一文件填数字，不改 owner 原话。
对照：`docs/roadmap/GROK_FIX_TASKS_ROUND2_20260907.md` 第 2 节 H-6、第 3 节就绪门；审核 `docs/roadmap/AUDIT_GROK_FIXES_20260907.md` K2/K3/K4/K5/K8 与 §3.3。
第一轮原文 `docs/roadmap/grok_reports_20260907/READINESS_20260907.md`、`G-B_ue_visual_packages.md`、`G-D_appearance_pixel_evidence.md` **只读，不改**。更正在本文件。

H-1～H-7 已合入本分支。数字来自各 H 报告与 parent 亲自核对的产物路径，不编造。

## 1. 结论（只写事实）

截至 parent 合入后的 round2 树：H-2 合并脚本在仓库内，`tmp/qa_pilot46_merged_round2_20260907_v1/` 46 格为 42 delivered / 2 blocked / 2 failed；`authored_c_animal_device` 为 `evidence_missing_or_unsampled`（规划 200 次用尽）。H-1 对 28 段已交付 UE 音频重渲到 `tmp/qa_pilot46_audio_v2_20260907/.../attempt_03/`：27 段写出成片，1 段（`authored_b_device_device`）因 peak=1.14 被渲染器 fail-closed；56/56 条 RIR 作业 2 ms 起 L/R 相关 <0.9；Habitat 段未动。H-7 干跑 `tmp/h7_scaleup_dryrun_7x50_20260907/` 350 个 request 均有 `sound_pool`，`distance_range_m` 为 [1.5, 6.0]，repeat 缺额 0，便携空调 `no_compatible_sounds=2`（owner 已接受）。合入后相关单测 16 个文件 **130 passed / 0 failed / 0 skipped**（4.69 s）——这是计数，不是放行条件。

Round-1 合并树 `ba0150e` 上，Claude 独立审核复跑相关单测 **283 passed / 0 failed / 0 skipped**（23 个文件，14.28 s，日志 `/data/jzy/tmp/claude_audit_20260907_v2/pytest_grok_related.log`）；该计数是 pre-round-2 baseline，不是第二轮合入后的计数。同一次审核重算：合并表 46 条、status 42 delivered / 2 failed / 2 blocked；42 段 delivered 的 facts/questions/preview.mp4/mixture.wav 都在盘上；曝光闸门 42 段 × 3 帧 = 126 帧均为 pass（placeholder 阈值 `mean_gray_fail_above=235` / `sat_share_fail_above=0.20`）；Habitat 四段 attempt_02 的 AudioProgram 为 `simultaneous_subset` / `one_active_of_n`，双耳 WAV 256000×2 @16 kHz = 16.000 s。契约前验对 19 段 attempt_02 成立；23 段 attempt_01 仍是旧代码「写完再验」，审核 R7 已标明。

Owner 对先导四格缺额的书面原话是「不要求每个房间跑出每种题」，适用范围是 `authored_a_device_device`、`kujiale_device_device` 预分配 blocked 与 `authored_b_animal_device`、`authored_c_animal_device` 规划 200 次用尽。Owner 对切分的书面原话是「先导切分 train39/eval0：接受」。Owner 对挂装的书面原话是「不着急，可以留在分母」。Owner 对放量干跑两条便携空调的书面原话是「接受」。Owner 对正式准入的书面记录是「Owner 未要求现在做缺失模态实验」；正式准入仍空，不写入第二轮就绪门第 1–4 条。

第二轮合入后的闸门、契约、覆盖、干跑、单测数字：

- 46 格：H-2 `tools/dataset/merge_qa_batch_attempts.py` 从提交状态复现；覆盖 9912 行中 166 行 `failed_episode`；两个 blocked 预分配、两个 failed 规划用尽，五态符合 owner 第 4 条。产物 `tmp/qa_pilot46_merged_round2_20260907_v1/`。
- H-1：任务书「32 段」对应已交付 28 段；27 段 attempt_03 成片；56/56 RIR 尾部相关 <0.9；侧向 stem 可用线索 24/28；混音 −6～−25 dBFS 为 24/28；`authored_b_device_device` 削波未写出。Habitat 未重渲。细节 `H-1_binaural_sh_order.md`。42 段 round-1 闸门/契约事实仍见 Claude 审核，不把 attempt_03 的 27 段说成 42 段已重新闸门。
- 放量干跑：`tmp/h7_scaleup_dryrun_7x50_20260907/`，catalog 为生产树绝对路径并带 `path_bindings`，350/350 有 `sound_pool`，距离 [1.5, 6.0]，repeat 缺额 0，空调两条另计。6 段 `--plan-only` rc=0。
- 单测：16 个文件 130 passed / 0 failed / 0 skipped。清单见 §4 合入后段。

本结论句不以听感、不以「测试变绿」作为放行条件。

## 2. Owner 裁定（原话，不扩写）

来源：`docs/roadmap/grok_reports_20260907/OWNER_LISTENING_AND_GAP_RULINGS_20260907.md`、`docs/roadmap/GROK_FIX_TASKS_20260907.md` §1、`docs/roadmap/GROK_FIX_TASKS_ROUND2_20260907.md` §1。下面只抄原句。不把「不要求每个房间跑出每种题」扩成「缺额不必每房每题」；不把「先导切分 train39/eval0：接受」扩成「无独立 eval」。

### 2.1 覆盖缺口（owner 接受，书面记录「不挡放量」）

- 「`authored_a_device_device`、`kujiale_device_device` 预分配 blocked：不要求每个房间跑出每种题。」
- 「`authored_b_animal_device`、`authored_c_animal_device` 规划 200 次用尽：同上。」
- 「9 个挂墙/吊顶 `interface_not_implemented`：不着急，可以留在分母。」
- 「放量干跑 2 条便携空调无兼容声音身份：接受。」
- 「先导切分 train39/eval0：接受。数据集不追求尽善尽美。」

「不要求每个房间跑出每种题」的适用范围是上面这先导四格，不是一般命题。

### 2.2 第一轮任务书六条（继续有效）

1. 声音片段上限仍是 5 秒；repeat 预分配可行性：`2×重复声音时长 + 另一源声音时长 + 2×0.5 s ≤ 13 s`；分配定下后运行时不许再换。
2. 「**不做**训练/测试集的房间与说话人预留（owner 决定不用）。」
3. 打散声源类别对与条件组的绑定（每家族每条件组至少一次；类别对在条件组之间分布开）。
4. 覆盖表五态：「代码缺陷或接口不通导致的运行失败记 `interface_not_implemented`」；「画像内 200 次重试用尽记 `evidence_missing_or_unsampled`」；预分配缺额仍记 `evidence_missing_or_unsampled`。
5. QA-18 湿尾边距按 Codex 现状（只用实测湿尾区间，不加边距），不动。
6. 「五段抽听与 P7 十条试听记录保持 pending_human，不动、不代填。」

### 2.3 第二轮新增

7. 「双耳混响阶数修复：owner 2026-09-07 晚点头。」（UE 路径 `_simulation()` 写死 `direct_sh_order/indirect_sh_order = 0`；要改成有方向的混响并重渲 UE 家族音频。视频与像素证据不动。）

### 2.4 正式准入

OWNER 文档：「「正式准入」见就绪报告补注：不是这五段好不好看/好听，而是 24 类题人能不能答、是否必须同时靠视听。Owner 未要求现在做缺失模态实验。」

P7 接受书 `p7_prepared_audio_v3/owner_acceptance_20260907T020953Z.json`：`owner_statement` = 「这个处理应该是没有问题的，可以直接通过了」；`individual_listening_records` 保持不变；`scope` = 「Processing acceptance for current prepared v3 research set; not full-library human calibration or formal dataset admission。」

OWNER 文档里没有「机器闸门通过即可开大规模生产」这一句。第一轮就绪报告 `ba0150e` 把听感与「无独立 eval」写进结论句，审核 K2 已标为口径错误。

### 2.5 试听（记录 owner 说过的话，不作为本文件结论句的放行条件）

OWNER 文档：「Owner 判断：这五段视频和十条音频没有问题。」成片约 16 秒是 Episode 时钟，不是素材超 5 秒。逐条 `heard` / `reviewer` 仍不代填。该判断覆盖的实际成片见第 5 节，五段全是 human_human。

## 3. 第二轮就绪门（任务书第 3 节）

| # | 门 | 本分支状态（`789bc64`） |
| --- | --- | --- |
| 1 | 46 格合并表能从提交状态复现；四个未交付格子带阶段与五态，分类符合 owner 第 4 条 | **产物已有。** 脚本 `tools/dataset/merge_qa_batch_attempts.py`。`tmp/qa_pilot46_merged_round2_20260907_v1/`：42/2/2；`authored_c_animal_device` = `evidence_missing_or_unsampled`；覆盖 166 行 failed_episode。 |
| 2 | UE 家族 32 段音频重渲完成并通过 H-1 验收；Habitat 段不变；全部 42 段闸门与契约通过 | **部分完成。** 已交付 UE 28 段重渲 27 成片 / 1 削波 fail-closed；56/56 RIR 尾部相关 <0.9；4 段峰值高于 −6 dBFS；Habitat 未动。attempt_03 的 27 段未宣称等于 42 段闸门重跑。见 `H-1_binaural_sh_order.md`。 |
| 3 | 放量清单干跑用 H-3 后的生成器重出一份，`room_catalog`/`path_bindings` 完整、缺额 0（除已接受的便携空调两条） | **产物已有。** `tmp/h7_scaleup_dryrun_7x50_20260907/`：350 `sound_pool`，距离 [1.5, 6.0]，repeat 缺额 0，空调 2 条另计。 |
| 4 | 相关单测 0 failed / 0 skipped，`docs/TOOL_INDEX.md` 一致 | 合入后 16 个文件 **130 passed / 0 failed / 0 skipped**。这是计数。pre-round-2 baseline 283/0/0 仍见下。 |
| 5 | 就绪报告按 H-6 口径重写，列出仍空着的项（挂装、人工校准、逐条试听、正式准入） | 本文件。仍空项见 §7 |

## 4. 单测（pre-round-2 baseline，不是合入后计数）

**来源：** Claude 审核 `AUDIT_GROK_FIXES_20260907.md` §0：「我在合并树上复跑 23 个文件（Grok 触碰的全部 + P10 那 13 个 + 我的审计器测试）：**283 passed / 0 failed / 0 skipped**（14.3 s，`/data/jzy/tmp/claude_audit_20260907_v2/pytest_grok_related.log`）。」

**pytest 摘要（日志原文，未重跑）：**

```
........................................................................ [ 25%]
........................................................................ [ 50%]
........................................................................ [ 76%]
...................................................................      [100%]
283 passed in 14.28s
```

日志只有点阵与摘要，没有文件名。下面 23 个路径按审核原文在 `789bc64` 上复原；`--collect-only` 合计 283。这不是新的 passed 计数。

**P10 那 13 个**（`tmp/p10_committed_verification_20260907_v1/verification.json` 的 command 列表；当时 144 passed @ `24ea774` / `72bc91c`）：

| 文件 | collect-only @ `789bc64` |
| --- | ---: |
| `tests/unit/test_qa_batch_manifest.py` | 16 |
| `tests/unit/test_qa_batch_sound_pool.py` | 3 |
| `tests/unit/test_qa_batch_coverage.py` | 13 |
| `tests/unit/test_qa_batch_delivery.py` | 3 |
| `tests/unit/test_qa_batch_runner.py` | 15 |
| `tests/unit/test_qa_evaluation_permutations.py` | 12 |
| `tests/unit/test_conditioned_sampler.py` | 25 |
| `tests/unit/test_qa_conditioned_transition.py` | 5 |
| `tests/unit/test_dataset_model_evaluation.py` | 6 |
| `tests/unit/test_qa_v3_dataset_export.py` | 3 |
| `tests/unit/test_qa_unified_catalog.py` | 52 |
| `tests/unit/test_qa_unified_scoring.py` | 7 |
| `tests/unit/test_tool_index_current.py` | 1 |

**Grok 触碰、且不在上表的 8 个**（`git diff --stat 72bc91c..ba0150e -- tests/`）：

| 文件 | collect-only @ `789bc64` |
| --- | ---: |
| `tests/unit/test_current_mp3d_dynamic_audio.py` | 35 |
| `tests/unit/test_exposure_gate.py` | 2 |
| `tests/unit/test_floor_reference_measurement_kind.py` | 4 |
| `tests/unit/test_p6_audio_unification.py` | 8 |
| `tests/unit/test_p9_evidence_delivery.py` | 17 |
| `tests/unit/test_qa_evidence_appearance.py` | 9 |
| `tests/unit/test_room_package_path_existence.py` | 2 |
| `tests/unit/test_room_packages_p3.py` | 10 |

**审计器测试：** `tests/test_audit_binding_feasibility.py`（15）。Grok 未改该文件（审核：diff 为空）。

以上 22 个文件 `--collect-only` = 263。第 23 个使合计为 283 的单文件加项是 `tests/unit/test_qa_production_contracts.py`（20；Claude 第一轮 P1 复跑集）。文件名不在 pytest 日志里，按 283 这个审核数字反推。

子代理分组复跑（审核原文，供对照，不是另一套放行数字）：G-A 六文件 89（Grok 报 92，差在合并后 `test_p9_evidence_delivery.py`）；G-B 19；G-C 42；G-D 26；G-D/G-E 八文件并集 76。

合入后 parent 在 `/data/jzy/tmp/wt-grok-pilot46-round2` 跑：

```
pytest -q tests/unit/test_frame_readback_binaural_sh_order.py \
  tests/unit/test_frame_readback_dynamic_audio.py \
  tests/unit/test_render_frame_readback_speech.py \
  tests/unit/test_merge_qa_batch_attempts.py \
  tests/unit/test_qa_batch_manifest.py \
  tests/unit/test_qa_batch_runner.py \
  tests/unit/test_qa_batch_coverage.py \
  tests/unit/test_qa_batch_delivery.py \
  tests/unit/test_build_qa_batch_manifest.py \
  tests/unit/test_room_package_reproducibility.py \
  tests/unit/test_room_package_path_existence.py \
  tests/unit/test_floor_reference_measurement_kind.py \
  tests/unit/test_exposure_gate.py \
  tests/unit/test_authored_exposure_bias_source.py \
  tests/unit/test_qa_evidence_appearance.py \
  tests/unit/test_tool_index_current.py
```

**130 passed / 0 failed / 0 skipped**（4.69 s）。无失败原文。

## 5. 五段试听清单

Owner 本机包 `Documents/pilot46_listening_20260907` 实际是五段 **human_human**（4× attempt_01 + A 房 attempt_02），**不是** `five_clip_listening_pending.json` 的清单。服务器打包脚本 `/tmp/pack_listening.py` 与 `/tmp/pilot46_listening_pack/02_five_episode_previews/index.json`：

| 包内目录 | episode | attempt | 类别对 |
| --- | --- | --- | --- |
| `1_apartment` | `qa_pilot46_20260907_apartment_human_human` | attempt_01 | human_human |
| `2_authored_A_attempt02` | `qa_pilot46_20260907_authored_a_human_human` | attempt_02 | human_human |
| `3_kujiale` | `qa_pilot46_20260907_kujiale_human_human` | attempt_01 | human_human |
| `4_mp3d` | `qa_pilot46_20260907_mp3d_human_human` | attempt_01 | human_human |
| `5_hm3d` | `qa_pilot46_20260907_hm3d_human_human` | attempt_01 | human_human |

五段都没有动物声、没有设备声。

与两份 pending JSON 都不同：

- 原批 `qa_pilot46_background_20260907_v1/summary/five_clip_listening_pending.json`：五段 human_human **全是 attempt_01**（A 房也是 attempt_01，不是 attempt_02）。
- 重跑 `qa_pilot46_rerun_20260907_v1/summary/five_clip_listening_pending.json`：`mp3d_human_animal`、`hm3d_human_animal`、`authored_a_human_human` attempt_02、`mp3d_single_active`、`hm3d_single_active`。合并目录没有同名文件。

两份逐条 JSON 的 `heard` / `reviewer` / `notes` 仍为 null，`status`/`review_status` 仍为 `pending_human`（原批与重跑的 `five_clip_listening_pending.json`；P7 `p7_prepared_audio_v3/listening_samples_pending.json` 十条同样为空）。**不代填。**

H-1 重渲后的 UE 成片在 `tmp/qa_pilot46_audio_v2_20260907/episodes/<id>/attempt_03/`。建议新包至少含一段动物、一段设备（例如 `authored_a_animal_animal` 与 `apartment_human_device` 的 attempt_03；`authored_b_device_device` 因削波没有成片）。Owner 决定要不要听。本轮未打包到 Mac，未代填 heard/reviewer。

## 6. 第一轮报告口径更正（写在本文件，不改 G-B / G-D 原文）

### 6.1 G-B「任务书写 -3 EV 与 B 房 -4 冲突需 owner 拍板」是假冲突

`G-B_ue_visual_packages.md` §6：「任务书写“耐久 −3 EV”，B 房生产校准与包字段是 **-4 EV**。」

第一轮任务书 `GROK_FIX_TASKS_20260907.md` 全文没有 EV 数值、没有 `exposure_bias_ev`、没有「耐久 -3 EV」。B 房 -4 EV 有旧交付背书：`tmp/qa_real_rooms_20260906/walk_four_b_v2/plan/episode_plan.json:791` `"exposure_bias_ev": -4.0`。A/C -3、B -4 是包字段与旧生产校准，不是任务书冲突，不需要 owner 在 -3/-4 之间再拍一次。

### 6.2 「apply_exposure_gate 未接入 batch_delivery」在合并 HEAD 已过期

G-B 报告写于 G-B 自己的分支，当时该分支故意不改 `batch_delivery.py`（G-A 所有）。合并 HEAD `ba0150e` / 本分支 `789bc64` 里闸门已经接线：

```
src/avengine/qa/batch_delivery.py:331-338
    try:
        from avengine.qa.exposure_gate import apply_exposure_gate
    except ImportError:
        pass
    else:
        gated = apply_exposure_gate(result, episode_root)
        if isinstance(gated, Mapping):
            result = dict(gated)
```

审核 X2：闸门 fail-closed，且已接线。仍待 H-4 的是 `except ImportError: pass` 会静默失效，以及 Habitat 段帧来源是 `batch_review/frames` 抽帧而不是 `rgb.npy` 第 0/中/末。那是另一件事，不能再说「未接入」。

### 6.3 其它审核已标、本文件不重复当放行支柱的口径

- K1：`authored_c_animal_device` 合并表标 `interface_not_implemented`，stderr 是规划用尽。等 H-2。
- K4：G-D 把两只狗写反。第 0 帧 5 像素 `not_observable` 的是被扶手椅遮住的黄狗 source1；杰克罗素 source2 第 0 帧 2970 像素 pass。第 39 帧杰克罗素 9524 像素可见但逐帧 `not_observable`（暖棕 0.119 < 0.12）。等 H-5 在自己的报告里更正；不改 G-D 原文。
- K8：G-A 单测 92 应为 89（合并树）；46 格清单两条 repeat 缺额仍在（清单未重生成）。

## 7. 仍空着的项

三分法沿用任务书：题义不适用 / 接口未实现 / 证据缺失。下列项 **不冒充完成**。正式准入按 owner 书面记录现在不做，也不是第二轮就绪门第 1–4 条的闸门。

### 7.1 挂墙 / 吊顶挂装接口（接口未实现）

9 个资产，reason 全是 `static_attachment_surface_not_implemented`。覆盖表 1260 = 9×7 房×20 题（另 9×7×4 题因「静物不是运动目标」记 N/A）。Owner：「不着急，可以留在分母。」

- `generated_air_conditioner_wall_split_white_research_v1`（wall）
- `generated_air_conditioner_window_unit_white_research_v2`（wall）
- `generated_doorbell_chime_unit_video_doorbell_black_research_v1`（wall）
- `generated_doorbell_chime_unit_wall_mounted_box_black_research_v1`（wall）
- `generated_floor_drain_exposed_bottle_trap_silver_research_v1`（wall）
- `generated_landline_phone_wall_mounted_beige_research_v1`（wall）
- `generated_microwave_oven_over_range_silver_research_v1`（wall）
- `generated_smoke_detector_ceiling_disc_white_research_v1`（ceiling）
- `generated_smoke_detector_wall_square_white_research_v1`（wall）

### 7.2 人工校准（证据缺失）

外观 512 像素与比例阈值、曝光闸门阈值、非人类颜色成分比例全部标 placeholder。没有人工标定后的生产阈值。P7 接受书写明不是 full-library human calibration。

### 7.3 逐条试听（证据缺失；禁止代填）

`five_clip_listening_pending.json` 与 P7 `listening_samples_pending.json` 的 `heard` / `reviewer` 仍为空。任务书第 1 条第 6 款：保持 `pending_human`，不动、不代填。Owner 本机听过的五段成片全是 human_human，见 §5。

### 7.4 正式准入（证据缺失；owner 未要求现在做）

仍空，三项都没有：

1. 24 类题人能否在不看 facts 的情况下答出；
2. 缺失模态视 / 听消融（是否必须同时靠视听）；
3. 人工校准后的门槛。

OWNER：「Owner 未要求现在做缺失模态实验。」

## 8. Round-1 已核事实（截至 `ba0150e`，供对照；不代替第 3 节）

产物只读：

- 原批 `tmp/qa_pilot46_background_20260907_v1/`（attempt_01 未改）
- 重跑 `tmp/qa_pilot46_rerun_20260907_v1/`（20 段 attempt_02：19 delivered / 1 failed）
- 合并 `tmp/qa_pilot46_merged_20260907_v1/`

覆盖表 9912 行重算：produced 246 / deferred_by_rule 976 / not_applicable_by_definition 1204 / interface_not_implemented 1260 / evidence_missing_or_unsampled 6226。五个家族 × 人/动物/设备 15 格每格至少一段 delivered 且闸门 pass。A/B/C 重跑帧灰度 135.7–191.7，饱和 ≤0.001。

四段未交付（owner 已用 §2.1 原话接受）：

| episode | 机器侧事实 | 合并表五态（有错，等 H-2） |
| --- | --- | --- |
| `authored_a_device_device` | 预分配超预算 blocked | evidence_missing_or_unsampled |
| `kujiale_device_device` | 预分配超预算 blocked | evidence_missing_or_unsampled |
| `authored_b_animal_device` | 规划 200 次用尽；重跑 outcome 有直方图 | 合并覆盖格子退化成 `asset_not_in_episode` |
| `authored_c_animal_device` | 规划 200 次用尽（stderr 直方图 169+31）；未用新记账器重跑 | 错记 `interface_not_implemented` |

`apply_exposure_gate` 在合并 HEAD 已接线（§6.2）。UE 路径球谐阶数仍写死 0，属 H-1，不属于 Grok 第一轮。

## 9. 尚未裁定、需要 owner 拍板（不自行决定）

审核 §6 仍待定，本文件不发明裁定：

- 便携空调：池里 `air_conditioning` 20 条全部因超 5 s 在建池时被拒，三台空调 0 条兼容声音。要在「从候选范围拿掉」与「允许裁剪超长设备声」之间选。Owner 已「接受」两条干跑缺额；如何改生成器另定。
- 画外锚点在不说话时能否入画（复核 mp3d |方位| 最小 8.7°、hm3d 1.1°）；干跑画外槽写死每房第 1、2 槽，设备画外零段。
- H-1 重渲后要不要听一包含动物与设备声的新成片（§5 建议出包，听不听由 owner 定）。
- 本 46 格两条冻结 repeat 缺额要不要回写先导清单。
