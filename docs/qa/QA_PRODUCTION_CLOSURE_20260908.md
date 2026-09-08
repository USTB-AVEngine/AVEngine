## 2026-09-09：主线整合验证通过

用户已接受本轮10秒审阅结果并授权整合与推送main。整合从远端main的f07212d出发，以b6f9e4a合入验收分支ef8fcef，保留双方提交历史；随后aba2a6c修正一条对旧相对路径行为的测试断言。

- 7处合并冲突已处理，保留主线通用声音绑定校验、可变声源数量和家具组合逻辑，以及验收分支的原生像素、音频与QA实现。
- 代码、配置和测试在aba2a6c上完成干净工作区回归：4914 passed、117 skipped、2 deselected、52 subtests passed。跳过项主要为未挂载的历史证据或已归档工作流。
- 合并后的UE与Habitat各一条10秒原生回归样本均完成交付，结果保存在Git忽略的tmp/qa_main_integration_20260908_v1。媒体和缓存不随Git提交。
- 当前批次配置中的catalog/registry改为仓库相对路径，语音输入指向外部数据根，不再依赖另一个代码检出。
- 350条生产未启动，已有服务未切换，正式数据准入状态未改变。下文保留验收阶段与历史版本记录。

# 10 秒 QA 验收修订与操作指南（2026-09-08）

权威位置：48g-jump 上 /data/jzy/tmp/wt-grok-pilot46-round2；分支 grok/pilot46-fixes-round2-20260907。本次产物根为 /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_user_acceptance_rework_20260908_v1。

## 当前交付

- 46/48 段有效片段（45份独立原生画面、1份明确标注的音频变体），415 道有效题，覆盖 24/24 题型。完整 Open/MCQ 数量见 data/coverage.csv，不能将某一种形式有效当作两种都有效。
- 每段实际视频与音频均为 10 秒：150 帧 / 15 fps、160000 samples / 16 kHz、双声道。全部 PCM 有限且未超幅。
- 混音峰值 -35.40 至 -3.03 dBFS；整段响度中位数 -32.1 LUFS，范围 -46.9 至 -14.4 LUFS。这些是实际文件测量，未据此做逐输出响度归一化。
- 题面使用带答案稳定性证据的时间段或事件锚点；显示名称取自核验过的外观/声音语义。机器 ID 保留在 JSON 的内部字段。
- 事件级素材重新遵循源素材的 −3 dBFS 参考峰值；1376 个事件记录、1374 个独立资产。原始素材和旧事件库保留。事件进入场景后保留距离衰减、混响及双耳差异，统一卷积后增益仍为 0.5。
- QA-10 旧的额外 0.15 倍输入增益已纠正。用户已确认对应新 10 秒样例可自然听清；这不是全库听感或绝对声压校准。
- 350 条仅完成新输入准备；7 房间各 50 条、24 题型与六类声源组合均保留。4 条预分配缺口见下文，未运行350渲染、未推送/合并、未代填正式准入。

## 查看和复核

打开 index.html。筛选房间或题型后点击“播放视频”；有时间范围的题目可直接播放对应区间。事实 JSON、题目 JSON、实际视频、覆盖表和执行来源都可在页面中打开。播放器音量和系统音量共同影响听感。

服务器最终索引：/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_user_acceptance_rework_20260908_v1/final_cohort_v1/manifest.json、outcomes.json、verification.json。它们选择实际最终尝试；14 条 Habitat 音频已经重新按配置 stride=5 计算，旧 stride=3 尝试作为历史保留。QA 补充样例在来源字段中明确标注，不能当作随机规划器的自然命中率。

## 配置入口

使用仓库 examples/dataset/qa_production_10s_20260908.json，先复制到新的配置文件再改参数并生成 fresh 输出。

| 设置 | 字段 / 行为 |
|---|---|
| 整段时长 | base_request.frame_count / frame_rate_hz；本次150/15=10秒 |
| 题目数量 | base_request.qa_sampling.items_per_type |
| 时间显示 | qa_sampling.time_display_precision=1、time_band_count=4；区间依据实际稳定窗口生成 |
| 素材边界 | sound_sources 与 sound_selection 的 max_clip_s；不截断事件来强行塞满时间 |
| 发声预算 | 实际整段时长减 condition profile 的 reserve_tail_s，并计入事件间隔 |
| 输入参考峰值 | split_sound_library_events.py --peak-normalize --target-peak-dbfs -3；必须生成新事件库 |
| 场景输出增益 | base_request.post_assembly_convolution_gain=0.5；不逐人物或逐混音重归一化 |
| 声学采样 | base_request.rir_stride；UE与Habitat都传递到实际命令，本次5 |
| 场景、人物、声音 | room_catalog、source_registry、sound_sources；显式路径优先，未知语义不猜测 |
| 原生运行 | runtime 的GPU/RPC/UE路径等；嵌套覆盖按叶子字段合并，局部覆盖不丢失基值 |

10 秒是本批选择；实现已检查其他时长、尾音预算、显示精度和嵌套覆盖，未将这些选择写死为一个特殊样例。

## 准备新批次

在权威服务器目录执行，替换输出名为新的、尚不存在的目录：

    cd /data/jzy/tmp/wt-grok-pilot46-round2
    PYTHONPATH=src:tmp/native_python_addons_v1 /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python tools/dataset/build_qa_batch_manifest.py prepare --config examples/dataset/qa_production_10s_20260908.json --output tmp/qa_next_prepare

新350请求中按7个房间、6种声源组合选取的7条实际 plan-only 均通过，未运行原生画面或音频；记录见 data/production350_plan_only_results.json。

350条的已生成目录：/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_user_acceptance_rework_20260908_v1/production350_10s_20260908_prepare_v1。4条缺额为 Apartment #003、Kujiale #203 的空调兼容声音缺失，以及 Authored #156、MP3D #294 的重复节目最短时长分别14.5秒、9.86秒，超过7秒可用预算。缺额不能通过放宽事件完整性或删去类型掩盖。

保留原生画面重建音频或问题时，使用 tools/dataset/replay_qa_batch_audio.py，明确 --source-index、--manifest、--output、--convolution-gain。新事件库重绑定使用 --audio-inputs；仅在实际音频输入/增益不变时使用 --reuse-audio。工具拒绝把改变过的PCM冒充为已有音频。

## 实测耗时和并发

44 条主批原生采集平均 222.6 秒。完成后的音频、题目与审阅阶段平均耗时如下，包含该阶段并发环境影响：

| 房间族 | 片段数 | 音频+QA+审阅平均 |
|---|---:|---:|
| apartment | 6 | 79.4 s |
| authored | 17 | 49.2 s |
| kujiale | 7 | 119.9 s |
| mp3d | 7 | 317.0 s |
| hm3d | 7 | 250.3 s |

按各片段阶段实测耗时之和摊到主批有效题目约 41.6 秒/题；这是媒体成本摊销，不是单题文本生成延迟，也不是多路并发后的墙钟吞吐。已有 facts 的单样例题目生成实测约0.47秒/17题。

独立双人物原生试验：单任务284.3秒；同GPU两个任务合计262.2秒完成两段，吞吐约2.17倍。单任务GPU0总显存峰值2745 MiB，双任务5469 MiB；设备容量49140 MiB。超过两个实例的并发未测试。

单任务主要阶段：UE启动/场景40.1秒、预热22.3秒、常规画面与读回67.5秒、两个目标单独深度 pass 共102.7秒、CPU像素整理39.2秒，编码和校验约1.8秒。当前批处理器每GPU只有一个锁定通道，单独提高 --max-parallel 不会自动启用每GPU两个UE实例；本次没有改调度器。

Habitat音频先完成RLR关键帧仿真及卷积，再写主要结果，因此中途输出目录可能为空。本次stride5为30个RIR关键帧，CPU计算仍会成为耗时。不要以显存占用低推断整个流水线没有工作。

详细性能采样与阶段记录见 PERFORMANCE_REPORT.md、data/timing_summary.json。


---

## 历史记录：以下是先前16秒版本，不作为当前生产输入

# QA 生产前收尾与操作指南（2026-09-08）

权威工作副本：`48g-jump:/data/jzy/tmp/wt-grok-pilot46-round2`，分支 `grok/pilot46-fixes-round2-20260907`。产物为 `research_candidate`。本轮完成生产前工程收尾，350 条生产尚未启动，正式准入不在本轮范围。

## 已验证结果

| 项目 | 当前结果 |
|---|---|
| 先导批 | 原 46 格中 42 段有效，保留 4 个已接受缺额 |
| 补充样例 | 1 段真实 QA-10；合计 47 格、43 段有效片段 |
| 题目 | 426 个有效实例，覆盖 QA-01～QA-24；每例至少一种形式有效 |
| 音频 | 43/43 实测通过；16 秒、16 kHz、双声道、256000 samples，全部有限且未超幅 |
| 增益 | 每条请求预先声明卷积后增益 0.5；混音峰值 −31.14～−4.89 dBFS |
| 实际声学参数 | SH 3/1、depth 200、IR 上限 4 秒；本批 diffraction=false、max order=0 |
| 审阅 | 43/43 审计器完成、曝光检查通过；实际查询帧已进入审阅图 |
| 历史命令 | 47 处原命令逐项原样保留；另有 47 条当前重跑命令 |
| 生产清单 | 350 条、7 房各 50；资产、声音与条件逐格复现一致 |
| 实际规划 | 从 `/tmp` 工作目录运行 7 房，覆盖六类来源组合；7/7 房间、资产与清单一致 |

B 房 `device_device` 已恢复。增益仅对卷积后的湿声道应用一次，dry PCM 不变；unity 对照仍在峰值 `1.13879248` 处拒写。未增加 −6～−25 dBFS 门槛。

抽题策略已进入事实文件。声停题的合法时间窗按既有事件、静默和实测湿尾规则推导；显式窗口保持优先，缺少回读继续延期，QA-18 余量未放宽。

QA-10 的实际第 10 帧中，绿色上衣人物遮挡蓝色上衣人物：目标投影 54780 像素、可见 17794、遮挡 36986，遮挡者 mask 解释全部遮挡像素。Open 有效；MCQ 只有一个合法选项，保留 `mcq_option_domain_too_small`，没有使用目标自身作干扰项。

## 当前入口

以下均相对服务器工作副本：

- 最终选择与历史：`tmp/qa_production_closure_20260908_v1/final_merged47_v2/merged_episodes.json`
- 完整汇总：`tmp/qa_production_closure_20260908_v1/final_cohort_summary47/summary.json`
- 当前 47 格清单：`tmp/qa_production_closure_20260908_v1/final_cohort_manifest47.json`
- 350 条清单：`tmp/qa_production_closure_20260908_v1/production350_final/batch_manifest.json`
- 离线审阅页：`tmp/qa_production_closure_20260908_v1/review_bundle/index.html`
- 闭环核对：`tmp/qa_production_closure_20260908_v1/closure_verification.json`
- PCM 实测：`tmp/qa_production_closure_20260908_v1/native_audio_verification.json`
- 当前重跑命令：`tmp/qa_production_closure_20260908_v1/final_merged47_v2/current_replay_commands.json`

使用索引中的 `facts_path`、`questions_path` 和 `review.preview_path`。音频由 `input_refs.json` 指向本轮已验证的 0.5 音频；画面和正确的同阶 RIR 复用。早期中间目录与失败现场保留，不是当前入口。

## 覆盖与边界

覆盖分母为 59 资产 × 7 房 × 24 类 = 9912 行，区别于 426 条实际题目。五态为：produced 261、deferred 885、interface 1318、evidence missing 6244、not applicable 1204。

1318 条接口行中，挂装接口 1260、外观分类器接口 58。旧快照的 105 条外观接口行已按当前题目重新计算；不能把全部接口缺口称作挂装问题。

原四个缺额保留：A 房/酷家乐的 `device_device` 固定声音预算不足，B/C 房的 `animal_device` 条件采样预算耗尽。未知失败明确记录 `unclassified_failure`、阶段与诊断。

本先导批按现有房间/声音身份连接分组，结果 train=43、eval=0、未分配=0；这不代表为后续 350 条预留了独立评测集。人工听感 JSON 保持 pending，正式模态准入与人工校准未代填。

## 启动 350 条

清单已分配 GPU 0/1/2/3 与 RPC 39720/39721/39722/39723，每条队列 88/88/87/87 条。执行器队列内串行、队列间并行，启动前检查显存和端口。视觉路由保持 Apartment/创作房/酷家乐走 UE，MP3D/HM3D 走 Habitat。

实际启动时使用尚不存在的输出目录：

```bash
ssh 48g-jump
cd /data/jzy/tmp/wt-grok-pilot46-round2
PYTHONPATH=src:tmp/native_python_addons_v1 PYTHONDONTWRITEBYTECODE=1 \
/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
tools/dataset/run_qa_batch.py \
  --manifest tmp/qa_production_closure_20260908_v1/production350_final/batch_manifest.json \
  --output tmp/qa_scaleup_7x50_20260908_run01 \
  --max-parallel 4 \
  --min-free-gpu-mb 8192
```

该命令尚未执行。便携空调保留 2 条 `no_compatible_sounds`：025 apartment、273 MP3D，均为 device-human 组合。repeat 缺额为 0，画外请求 14 条，每种来源组合至少覆盖 4 个条件组。

调整资源时，修改 `tmp/qa_production_closure_inputs_20260908_v1/production_config_four_gpu.json` 的 slot `request_overrides.runtime`，再用 `build_qa_batch_manifest.py prepare` 生成新目录并核对。增益位于 `base_request.post_assembly_convolution_gain`，未写死在生成器里。

## 验证与版本

实现提交：`2be9b40`（音频/输入）、`964f48e`（策略转发）、`ea35979`（合法时间窗/审阅帧）、`0c1261f` 与 `3373c86`（最终汇总）。QA 生成代码自 `ea35979` 后未变化；实际执行版本保留在每份 producer 中。修改仅提交在服务器目标分支。

核心整合回归 300 项通过；后续窗口与收口回归 131 项通过；最终汇总回归 15 项通过。测试、听感和结构测量均不替代正式研究准入。
