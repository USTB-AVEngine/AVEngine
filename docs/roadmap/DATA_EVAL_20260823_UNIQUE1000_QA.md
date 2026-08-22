# 训练数据评估：unique1000 episode 池 + v1 QA（20260823）

> 评估人：Claude（20260823 会话，接 HANDOFF_20260823_data_eval_baseline.md 任务 1）。
> 全部为诚实盘点 + 只读分析；未生成、未覆盖、未改动任何数据。
> 结论供 owner 定夺，不构成任何 dataset admission。分析脚本与原始输出：
> `/data/jzy/tmp/inventory_qa_data.py|.out.json`、`/data/jzy/tmp/analyze_qa_deep.py|.out.json`、
> `/data/jzy/tmp/analyze_round3.py|.out.json`。

## TL;DR（先说结论）

1. **数据找到了**，不在 handoff 列的候选位置，而在
   `/data/datasets/avengine_workspaces/`（一块此前没被盘点的路径）。
   "约 1000 条"实为 **1000 个 recombined episode + 约 1 万条 QA**。
2. **合规面基本干净**：房间是 apartment_0000（非 blender_custom、非
   Skokloster），全链 `research_candidate` + `qualification_claim=false`，
   正式分母仍为 0。**但产物出自 legacy 多仓引擎（AVEngine-habitat-native，
   m7 期，20260723），不是现 main 单仓新鲜生成**——按铁律这批数据能否
   进论文表格需要 owner 拍板。
3. **核心命题（双模态缺一不可）目前未被这批数据支撑**：v1 问句普遍在
   文本里点名物种（"在叫的那只狗"），而每个场景每物种至多一只，导致
   多数题**视觉+文本即可作答**。owner 已跑的 Qwen2.5-Omni-7B 试跑佐证
   了这一点：full_av 50% vs video_only 48%（n=100，差距不显著）。
4. **单声道折叠混淆已实锤**：Qwen 管线 `preprocessed_audio_shapes=[[76800]]`，
   双耳被折成单声道后 ILD 线索在输入端即被抹掉——audio_only 的空间题
   掉到 chance 是输入管线限制，不能报成"模型不懂空间音频"。
5. 补生成方向明确（见 §7 缺口清单）：同物种双源场景、问句去物种化、
   off-screen 变体、侧向方位补桶、refusal 题型、双通道 baseline 适配。

## 1. 数据在哪、是什么

| 项 | 值 |
| --- | --- |
| 视觉+双耳成片池 | `/data/datasets/avengine_workspaces/AVEngine-habitat-native/tmp/m7/apartment_asset_bound_ue_unique1000_full_20260723_01`（2.5 GB） |
| 规模 | 1000 episodes；每集 6 文件：`ue_clean_binaural.mp4` / `ue_visual_only.mp4` / `ue_topdown_*` ×2 / `evidence.json` / `runtime_readbacks.json`，75 帧 5s |
| 构成 | 4 对型（border_collie_cat / border_collie_human / cat_border_collie / human_border_collie）× 4 运动型（both_moving / s1_moving / s2_moving / static_static），每格 62–63 集 |
| 生成方式 | 双 GPU shard 并行 + hardlink 合并；part000 完成 611/1000、part500_gpu2 完成 500/500，跨 shard 重复 111 场景已去重（merge_report status=pass，全媒体 reopen 校验过） |
| 房间 | `/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000`（UE apartment，合规） |
| 资产 | 仅 3 个个体：border collie（black_white）、abyssinian 猫（ruddy）、human；全部 `generated_*_research_v1` |
| QA 工作区 | `/data/datasets/avengine_workspaces/AVEngine-habitat-native-acoustic-fix/tmp/qa/`（20260727 系列） |

QA 资产（均 `research_candidate`、`qualification_claim=false`）：

| 集合 | 数量 | 说明 |
| --- | --- | --- |
| simple questions v1_02 | 7195（挖出 9416，均衡后 7195） | 6 题型：Q-ACT 1992 / Q-CNT 2000 / Q-ATTR 1000 / Q-SRC 1000 / Q-AVREL 798 / Q-CMP 405；覆盖全部 1000 集，每集 4–11 题 |
| temporal questions v1 | 2871（挖出 3244） | 基于 200 集间歇窗：Q-TEMP-DURING 926 / BETWEEN 554 / AT 524 / Q-NUM-TIME 400（numeric_banded，0.3/1.0s 计分带）/ Q-NUM-CNT 386 / Q-TEMP-ORDER 81 |
| intermittent 200 批 | 200 wav（16kHz 双耳）+ 事件窗真值 | 窗口为构造真值（declared_intermittent_program_v1） |
| axis1 twin 音频 | 620 集 route-swap 孪生混音（float32 wav） | 视觉复用原集字节；仅音频反事实 |
| axis1 证书 | 7195 条中 granted 1073 / refused 725 / not_applicable 5397 | 见 §3 |
| qa-pilot 100 题试点 | `AVEngine-qa-pilot/tmp/qa/pilot_v0_100`（配额抽样、确定性选取） | 已跑 baseline，见 §6 |

## 2. Provenance 警示（需要 owner 拍板）

这批 episode 由 **legacy 多仓引擎**（AVEngine-habitat-native 工作区，m7
期，20260723）生成，早于现 main 单仓一条龙。按铁律"所有产物必须由单仓
引擎新鲜生成"，若论文表格要用这批数据（或其衍生 QA），有两条路：

- (a) owner 明确豁免：作为 research_only 试跑数据继续用（现状即如此，
  pilot run 元数据也都标了 `eligible_for_paper_table=false`）；
- (b) 用现 main（`725c8cc`）+ Studio sweeps 重新生产等价池（见 §7）。

抽查未发现质量硬伤：4 集亮度曲线平稳（93.9–100.3，无历史上的曝光爬坡
缺陷）；媒体完整性合并时全量 reopen 过。**但动画探针
（`tools/qa/probe_ue_capture_animation.py`）需要原始捕获目录
（frame_records.json + arrays/rgb.npy），成片池只有 mp4，无法回溯探测
"滑行无动画"缺陷**——这是 legacy 产物无法按现行标准复检的一个实例。

## 3. 轴 1：双模态必要性（核心，结论：目前不成立）

**机制设计**是对的：v1 的意图是"音频锁定哪只个体，视觉回答其属性"
（modality_note 全集一致），axis1 route-swap 证书验证"换路线后答案翻转"。

**但覆盖和有效性都不足：**

- 证书覆盖：仅 Q-ATTR（granted 748/1000）与 Q-AVREL（granted 325/798）
  有认证路径；Q-ACT/Q-CNT/Q-SRC/Q-CMP 共 **5397 题（75%）not_applicable**。
  refused 的原因：twin 答案歧义 408、route swap 后答案不翻 317。
- **问句文本泄漏物种**："Is the dog that is barking moving…"——bark 即狗、
  meow 即猫，且每场景每物种至多一只 → "音频锁定个体"这一步被问句文本
  替代，**视觉单模态即可作答**。这不是证书能查出的问题：axis1 只验证
  "音频换了答案会不会变"，不验证"不给音频能不能答"。
- Q-AVREL（双源全程可见）：看画面找到狗在哪侧即可答。pilot 实测
  video_only 在 AVREL 上 66.7% **高于** full_av 的 58.3%。
- Q-SRC（哪两个物种在发声）：v1 双源持续发声且全程可见，视觉可答。
- 佐证（pilot_v0_100，Qwen2.5-Omni-7B）：full_av 50% vs video_only 48%，
  **AV−V 仅 +2pt**；而论文需要的是 AV 显著高于两个单模态。

**已有的正面资产**：granted 的 1073 题（ATTR 748 + AVREL 325）至少具备
"音频参与决定答案"的反事实证据；twin 音频 620 集已渲染，可支持
"原/孪生成对喂模型看答案翻转"这类论文级消融——这条路线还没人跑过。

## 4. 轴 2–5：分布、难度、负例、空间线索

- **答案边际**：均衡后各题型答案严格对半/三等分（如 AVREL left 399 /
  right 399，且 4 对型内部基本平衡）——无"全选左"捷径 ✓。但注意：
  - Q-ATTR 四选项中只有 black_white/ruddy 两个会是正确答案（资产只有
    一狗一猫），干扰项永不为真；且**整库 coat↔物种完美相关**，模型可
    以用"猫=ruddy"捷径把 ATTR 塌缩成物种识别题；
  - Q-CNT 四选项但答案只用 {0,1}；
  - pilot 的选项位置先验 A/B=0.34 > C/D=0.16；
  - text_only 39% > random 34%，仍有文本先验可蹭。
- **难度覆盖**：距离 0.42–8.02m 各桶都有 ✓；运动 4 案例各 ~250 ✓；
  **方位角严重不均**：正侧向近乎空（[+60,+75) 213、[+90,+105) 128、
  [-105,-90) 4550）而 [+0,+15) 15092、[-135,-120) 17204——恰恰是 ILD
  最强的侧向片段最少。角分离、遮挡占比直方图缺失（P1 可见性 pass 未做，
  Q-LOC-EGO/ALLO 因此 deferred）。
- **拒答/负例**：唯一负例形态是 Q-CNT 的 0 答案（1000 题，问不存在的
  物种数量）。**无 unanswerable/refusal 题型**——论文 refusal 落点缺口。
- **空间线索量化**（10 集抽查，250ms 窗 ILD，音轨取自 ue_clean_binaural.mp4）：
  均值 ILD −0.95～+4.24 dB，峰值 |ILD| 2.8–7.5 dB，|ILD|>1dB 窗口占比
  0.25–0.95。双耳线索**确实存在但集间差异大**；结合方位角分布，建议把
  "窗口 ILD 峰值"作为逐题难度标签补进 QA 元数据。twin wav 是 float32
  格式，本轮 stdlib 读取失败未完成 flip 一致性校验（用 soundfile 可补）。

## 5. 合规检查

| 项 | 结果 |
| --- | --- |
| blender_custom / Skokloster | 未涉及（全部 apartment_0000）✓ |
| research_only / episode_counted | 全链 research_candidate、qualification_claim=false、pilot run 全部 eligible_for_paper_table=false ✓，正式分母保持 0 ✓ |
| 声音×端点类别约束 | 资产为 generated_*_research_v1 犬吠/猫叫/人声，与端点匹配 ✓ |
| 新鲜生成铁律 | **不满足**（legacy 引擎产物，见 §2，需 owner 拍板） |

## 6. Baseline 试跑现状（任务 3 的已有事实，别重复跑）

owner 已在 pilot_v0_100（100 题）上跑过（2026-08-07/08，脚本与摘要在
`AVEngine-qa-pilot/tmp/qa/pilot_v0_100/runs/`）：

| 条件 | Qwen2.5-Omni-7B acc | macro | Qwen3-VL-8B |
| --- | --- | --- | --- |
| full_av | **0.50** | 0.53 | — |
| video_only | 0.48 | 0.48 | 0.47（macro 0.47） |
| audio_only | 0.41 | 0.46 | — |
| text_only | 0.39 | 0.42 | — |
| uniform random | 0.34 | — | — |

按题型看，full_av 相对 video_only 的增益集中在 Q-NUM-CNT（0.50 vs
0.08——数叫声次数视觉无从下手）和 Q-TEMP-ORDER（0.83 vs 0.58）；而
Q-AVREL、Q-COMPOUND-COUNT-MOTION 上 video_only 反而更高。

**关键混淆**：predictions 里 `preprocessed_audio_shapes=[[76800]]`——
双耳在预处理即折为单声道，ILD 全灭。所以 audio_only 空间题 ≈ chance
是**输入管线限制**；报告任何数字前必须区分"模型不支持双耳"与"模型答
不对"（handoff 预判的坑，已证实）。此外 lead-d 的 jaeger 系列
（RGB+depth+FOA，8–16 行小样本）全部是描述性 0.5 acc，分母过小仅作流
程验证，不入表。

## 7. 缺口清单 → 补生成/下一步建议（任务 2 输入，动手前与 owner 对齐配额）

按优先级：

1. **同物种双源场景**（two dogs / two humans）：铲掉问句物种泄漏的根
   —— "音频锁定个体"才真正必要。lead-a 已有 strict_two_human 全套设施
   （`tmp/lead_a_strict_two_human_*`、mp3d room v2–v4），走 Studio
   `POST /api/sweeps` + `mp3d_end_to_end` / `apartment_end_to_end`。
2. **问句去物种化改写**：现有 1000 池上把"the dog that is barking"改成
   "the animal that called first"等，可零渲染成本救回一部分 v1 题。
3. **off-screen 目标变体**（P1 路线图既有项）：声源出画后视觉不可答，
   音频必要性天然成立；配套 P1 可见性/遮挡 pass 补齐难度直方图。
4. **侧向方位补桶**：±60–105° 几乎空；补生成时在 sweep 轴上显式拉开
   方位角，并把窗口 ILD 峰值写进逐题元数据作难度分层。
5. **refusal/unanswerable 题型**：论文需要 refusal 落点，现为 0。
6. **双通道 baseline 通路**：给 Qwen2.5-Omni 加立体声输入路径（或改
   FOA/双通道适配层），并用 `paired_ablation` 模板做左右置零/单声道
   折叠对照——把"空间线索被输入管线抹掉会掉多少分"本身做成论文卖点。
7. **twin flip 消融**：620 对 route-swap 孪生已在盘，原/孪生成对喂模
   型验证答案翻转率，是现成的"音频参与"证据链，无需新渲染。
8. **A-only 消融改输入不改 prompt**（黑帧视频），与现有 audio_only 脚
   本对齐核对。

## 8. 留给 owner 的决断点

1. unique1000（legacy 引擎产物）的地位：继续 research_only 试跑，还是
   用现 main 重产等价池？（影响 §7 各项的载体选择）
2. handoff 让我问的两个问题，盘面已给出答案，请确认无遗漏：
   ①数据位置 = `/data/datasets/avengine_workspaces/`（如另有机器上的
   批次请指出）；②已试模型 = Qwen2.5-Omni-7B 四条件 + Qwen3-VL-8B
   video_only（pilot_v0_100）+ lead-d jaeger 小样本系列。
3. 补生成配额设计（题型×难度桶×对型）待对齐后再开批量。
4. 闭源 API（GPT-4o/Gemini）是否入表。
5. 35 项人工听审 + rights 清权是否排期（正式分母解锁的前置）。
