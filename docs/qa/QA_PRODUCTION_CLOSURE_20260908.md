<!-- owner-review-rework-20260908 -->
> 当前状态：用户验收修订中。用户已明确整段 Episode 固定 10 秒；题面时间应为可理解的范围或事件锚点，内部外观/状态/声音枚举不得直接显示。本页下文的 16 秒先导和旧 350 条清单是先前版本记录，尚未满足这些修订要求，暂不作为启动生产的依据。真实响度复核也发现 QA-10 遗留 0.15 倍事件增益；未超幅检查不等于响度校准通过。旧产物和历史命令继续保留，10 秒版本使用新输出。

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
