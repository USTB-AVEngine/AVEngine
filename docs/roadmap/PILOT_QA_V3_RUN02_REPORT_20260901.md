# QA v3 run02 工程报告（2026-09-01）

> 状态：**工程链完成，单模态认证未授予**。本报告只覆盖 research/dev
> 小批，不是百题级 pilot、数据集准入或论文结果。

## 0. 给 Claude 的一次性总审委托

请只读复核本报告与下列服务器工作树，不修改代码、不启动作业：

- 工作树：`/data/jzy/tmp/wt-qa-v3-pilot`
- HEAD：以审计时 `git rev-parse HEAD` 为准；本报告提交前实现 HEAD 为
  `9126d508b6901bda6fe5873536ca2721afc49226`
- 最终选择：`/data/jzy/tmp/qa_v3_run02_selected30_a547580_r2`
- 最终视觉：`/data/jzy/tmp/qa_v3_run02_selected30_visual_a547580`
- 最终音频：`/data/jzy/tmp/qa_v3_run02_selected30_audio_a547580`
- 媒体：`/data/jzy/tmp/qa_v3_run02_selected30_media_a547580`
- 形式×模态矩阵：
  `/data/jzy/tmp/qa_v3_run02_form_modality_matrix_a547580.json`

请定点回答：

1. debug 与 Kujiale 是否被诚实限定为技术 canary，而非场景/题型准入；
2. USD runtime 黑帧→editor bake→lit baked map 的证据链是否闭合；
3. runtime LOS 只作预筛、native pixel truth 作终裁的权威分工是否正确；
4. 最终 30 点选择是否泄漏 outcome，联合格覆盖与剩余边际偏差是否披露充分；
5. 60 份音频和 30 对 Gate A 是否分别建立了波形变化、结构保持、
   MCQ gold 翻转、Open 得分区分离；
6. MCQ/Open 是否真正分记录、分评分、分报告，没有互相充当证据；
7. A-only/V-only 的结论是否正确停在“风险/未认证”，有没有误放行；
8. 是否还有会阻塞下一阶段百题级 pilot 的 P0–P2 缺陷。请把 P3/P4
   改进建议与阻塞项分开。

## 1. 结论

完成了五个双源 profile（①F、①B、⑦、⑧、⑨）的 30 点 run02-dev：

- 每型 6 点，main/Gate A 各一份 AudioProgram；
- 30/30 标准 75 帧 RGB capture 通过；
- 每点按题型实际锚/查询帧完成 normal/target-only 深度像素终裁；
- 60/60 动态双耳音频通过；
- 30/30 main↔Gate A 波形非同一，30/30 paired fact 语义翻转；
- 每个 fact 拆成 MCQ/Open 两条，合计 60 条，不合并认证；
- 30 个 video-only、30 个 full-main、30 个 full-GateA 全量 ffprobe 通过；
- Open 主真值 30/30 满分；以 Gate A 真值答主问题 30/30 零分。

但**认证未授予**：每型只有 6 点；A-only 物理单特征探针在①B/⑦/⑧/⑨
高于多数类基线；V-only 没有强模型或人类缺模态结果；完整 AV 人类正确率
下置信界、外部模型探针、等效性检验和正式容差校准仍未运行。

## 2. 关键实现提交

| 提交 | 内容 |
| --- | --- |
| `b7fd7b1` | F3 双角色独立路线与运动强弱反平衡 |
| `0ca707a` | current timeline native pixel probe；显式 runtime closure seed |
| `f14d384` | 双运动求解器拒绝静态路线 |
| `96c996c` | stage 组装器支持 runtime plugin/global definition |
| `1ac55c1` | USD editor materialization：普通 StaticMesh level |
| `275809d` | USD bake 支持固定、题目无关的 scene light 配置 |
| `da52083` | feasible-grid LOS 适配器（预筛） |
| `74d6389` | 动态音频措辞/批 runner 场景无关化与逐点 M1 |
| `77fd31b` | packaged UE BlockAll runtime LOS 批探针 |
| `2732230` | 可配置 visual FOV margin |
| `d134d33` | card8 首叫严格半开时间带，修复上界越带 |
| `a547580` | 方位带内安全余量；最终重算错带变候选拒绝 |
| `c348980` | 批 runner 对齐 `_rand_gateA_v1` 命名 |
| `9126d50` | current fact/onset/Gate A 语义批级验证器 |

## 3. debug canary

### 3.1 运行时地图与导航

- 实际地图：`/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000`
- 运行时 navmesh：`RecastNavMesh-Default`
- 路线探针：128 点、54 条路线；ground=0 域筛后 36 条路线
- 路线库：
  `/data/jzy/tmp/qa_v3_debug0000_ground0_routes_6f20e4b_20260901/route_bank.json`

### 3.2 像素与 Gate A

初始三点都被真实像素证伪（片尾遮挡、途中出画、锚帧不可见）。加入
`M_Emissive` 的 fresh pixel stage 后，24 点池中按研究参数
`visible_fraction>=0.5`、`visible_pixels>=1000`、bbox 不贴边仅 3 点通过。
canary 取 `card1F_002`：

- 75 帧 RGB/readback 全过；
- 两角色均移动；
- main truth `35.210°`，Gate A truth `-43.807°`；
- MCQ 右带→左带；Open 间隔 `79.017° > 60°`。

debug 没有可用房间声学，故这里只建立 visual+program+fact 技术 canary，
不宣称 debug Gate A 波形或住宅场景多样性。

## 4. 第二真实住宅：Kujiale

### 4.1 失败历史（全部保留）

1. `UsdStageActor` 可 Full Cook，但 Linux Game runtime 的 UE 5.5
   `UnrealUSDWrapper.Build.cs` 明确禁用 runtime USD；两版 packaged capture
   75/75 全黑，日志报 `USD SDK is disabled`。
2. 加 `FORCE_ANSI_ALLOCATOR=1` 不足以绕过 Linux Game 禁用，第二版仍黑；
   这证明 Target.cs 配置文本不是 runtime 能力证据。
3. 首版 editor bake 有住宅几何但无灯，平均像素约 1.07，不可答。

### 4.2 最终 baked-lit stage

通用 editor 工具把外部 USD 物化为：1 个 StaticMesh、106 个材质实例、
102 张纹理；level 中 `UsdStageActor=0`。四盏 soft fill 位于整个路线域
包围盒四分位网格，固定且与目标/答案/槽位/轨迹无关。

- 闭包：242 内容包（228 Game + 14 SpContent），无 `UsdAssetCache` dangling
- packaged stage：
  `/data/avengine_external/ue-package-stages/qa_v3_kujiale_baked_lit_275809d_20260901`
- stage BuildCookRun：ExitCode 0
- canary：`card1F_003`，像素与 75 帧 readback 全过
- Kujiale main/Gate A：各 25 个动态 RIR keyframe、4 事件、80,000×2
  float32；124,665 个样本值不同，最大差 0.032517
- MCQ 左/右带翻转；Open `61.36° > 60°`

这只证明一个第二住宅同链 canary，不是 Kujiale 批量准入。

## 5. run02 候选与终裁

### 5.1 attempt1（保留为拒绝证据）

60 个预分配几何点全部 RGB 成功，但像素仅 `card9_004` 通过。根因是
Apartment 配置没有 LOS，相机中心进视锥不等于角色可见。

### 5.2 采样修正

- `MIN_CAMERA_DISTANCE_CM=250`（研究参数）；
- `VISUAL_FOV_MARGIN_DEG=5`（默认仍为0）；
- `ANSWER_BAND_INTERIOR_MARGIN_DEG=0.25`（默认仍为0）；
- packaged UE `BlockAll` complex trace 只作预筛；
- native same-camera normal/target-only depth作终裁。

大池 3998/4000 点生成，2 个最终重算错带被记为候选 rejection；runtime
LOS 后 183 clear。按联合格每格先取1点做像素，再补①F/⑨其余 clear 点，
得到可组成 run02-dev 的通过池。

### 5.3 最终 30 点

`/data/jzy/tmp/qa_v3_run02_selected30_a547580_r2/selection_manifest.json`
记录来源与像素证据。每型6点，所有声明答案类别和所有二元因子实际值均
覆盖；选择后仍有边际偏差，特别是①F `source1_coat=4:2`、⑧时间带
`2:1:1:2`，故条件基线按实际分布报告，不称统计认证。

## 6. 最终视觉、音频与反事实

### 6.1 视觉

- 路径：`/data/jzy/tmp/qa_v3_run02_selected30_visual_a547580`
- 30/30 × 75 帧；最大位置误差 `5.68e-14 cm`；最大 yaw 误差
  `6.94e-5°`；最大动画相位误差 `4.8e-7 s`

### 6.2 音频

- 路径：`/data/jzy/tmp/qa_v3_run02_selected30_audio_a547580`
- 验证：`/data/jzy/tmp/qa_v3_run02_selected30_audio_verification_a547580_r2.json`
- 60/60 16 kHz 双耳；30/30 波形非同一；30/30 Gate A paired fact 语义翻转；
  onset、registry、program、M1、receipt、题型专用锚/尾窗全过；0 failures

### 6.3 媒体

- 路径：`/data/jzy/tmp/qa_v3_run02_selected30_media_a547580`
- 30 video-only（无音轨）、30 full-main、30 full-GateA
- 全部 1280×720、15 fps、75 帧、5 秒；full 为 16 kHz 双声道；0 failures

## 7. MCQ/Open 双形式

- 问题：`/data/jzy/tmp/qa_v3_run02_dual_form_questions_a547580_r2.json`
- MCQ 30，Open 30；是独立记录，共享 fact，不互为认证证据
- Open 主真值阳性：
  `/data/jzy/tmp/qa_v3_run02_open_perfect_scores_a547580.json`，均分 1.0
- Gate A 真值答主问题：
  `/data/jzy/tmp/qa_v3_run02_open_gatea_counter_scores_a547580.json`，均分 0.0
- 时间容差使用显式研究占位 `T_FULL=0.5s/T_HALF=1.0s`，未校准

## 8. 形式×缺失模态矩阵

权威汇总：`/data/jzy/tmp/qa_v3_run02_form_modality_matrix_a547580.json`
（5 profile × 2 form × 2 missing modality = 20 格）。

A-only 物理探针只读最终 WAV，每型 n=6、2-fold，仅作风险诊断：

| profile | 多数类 | 全特征 | 最强单特征 |
| --- | ---: | ---: | --- |
| ①F | .500 | .500 | rms_end_start_db .500 |
| ①B | .333 | .333 | ild_mean **.667** |
| ⑦ | .500 | .500 | centroid **.833** |
| ⑧ | .333 | .333 | centroid **.500** |
| ⑨ | .500 | .500 | decay_proxy **.833** |

因此①B/⑦/⑧/⑨必须在百题级重新采样/认证，当前不放行。V-only 已有可执行
MP4与逐题像素可答证据，但没有强模型/人类缺模态等效性结果，也不放行。
Open 数值题的物理探针只是分箱标签 proxy，不是连续评分认证。

## 9. 尚未完成（明确外部边界）

1. 人类完整 AV 正确率下置信界；
2. 人类容差校准（角度/时刻）；
3. 百题级按 profile 的 clustered bootstrap、family-wise correction；
4. 强模型 A-only/V-only/AV 与 text-only 诊断；
5. 物理侧信道越线后的重新采样与复认证；
6. owner 未决的正式 `delta`、裁判模型与最终配额。

这些是下一阶段认证/实验，不是本次实现遗漏；在它们完成前不得把
`research_candidate` 改写为 certified/admitted。
