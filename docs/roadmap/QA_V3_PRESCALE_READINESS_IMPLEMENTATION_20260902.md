# QA v3 大规模生成前工程就绪实施报告（2026-09-02）

## 结论

本轮完成了所有当前可自动执行的工程修复、评分区分、反事实实渲、
人类实验打包与 released-media 单模态探针。实现基线为服务器唯一工作副本
`/data/jzy/tmp/wt-qa-v3-pilot` 的 `da6e751`；所有产物仍是
`research_candidate`。

**现在还不能开始大规模生成。**剩余前置不是新增题型或一般引擎能力，而是：

1. 收集人类校准响应并据此定稿 `THETA_FULL/THETA_HALF/T_FULL/T_HALF`；
2. 增加可完整渲染的住宅场景（当前只有 Apartment 与 Kujiale 两个住宅
   完成同链，HM3D 仅有设计层路线域证据）；
3. 用定稿容差生成新的百题级认证池，完整渲染后重新运行分形式
   Text/A-only/V-only；旧 run02 只能作泄漏诊断，不能升格。

## 实现提交

| 提交 | 内容 |
| --- | --- |
| `5119edb` | card1 目标全片仅锚定时发声；锚定带×查询带联合分配；锚角在正式宽评分半径下必须得零；card11 发声/绑定/像素帧对齐并只接收 `fully_occluded`；canonical emitter height 渲染支持 |
| `610a375` | card8 Open 严格满分带作为认证分，宽带只作诊断；card16 像素终裁后四态配额选择器 |
| `26f82f3` | canonical appearance Gate B 双耳重渲一致性验证政策 |
| `ba95ca9` | full-AV 人类校准网页、隐藏答案表和只在绑定正确试次上计算数值误差的评分器 |
| `286cb3f` | 跨场景 smoke 同步采用锚定带×查询带分层 |
| `ae0e82b` | 从最终发布音频/视频/题面运行 Text/audio/video-only 简单捷径探针 |
| `6dcc42a` | 对历史 216 条逐条重验，区分结构可保留、补元数据、重渲媒体、重采几何、像素待决、降级与 future extension |
| `769bb7b` | 人类试听 UI 改为 public/private 分离、无原生时间轴、最多从头播放两次并记录播放次数；生成三题预览版与 18 题正式版 |
| `da6e751` | 完成页增加只读回答 JSON 与复制按钮，Blob 下载被浏览器拦截时仍可取回结果 |

## 测试

- 顶层测试（排除既有重型 `test_verify_audio_batch.py` 与 unit）：
  **329 passed**；
- 本轮触及的 unit：**15 passed**；
- 工作区在报告成文前干净；未推送远端。

## 修复后真实证据

### 历史 216 条统一重验

权威输出：
`/data/jzy/tmp/qa_v3_prescale_revalidation_216_20260902_v2.json`

逐条读取历史 fact、timeline、AudioProgram 与 Gate B 政策，不修改原产物：

- `prescale_structure_pass`：72；
- `pixel_pending`：12；
- `metadata_or_pool_reselection_required`：5；
- `media_regeneration_required`：92；
- `geometry_resample_required`：11；
- `demoted_from_main`：12（card15a）；
- `future_extension`：12（card17）。

核心题分解：card1F 为 8 条几何重采+4 条媒体重做；card1B 为 3 条几何
重采+4 条媒体重做+5 条仅需元数据/池重选；card5R 12 条须按目标单锚策略
重渲音频；card6R 12 条结构可保留；card7/card8/card9 各 12 条须用
canonical appearance Gate B 重渲；card11 12 条须重做发声帧和 pixel join；
card16 12 条等待全量 native pixel 后四态选择。

因此“历史 216 条已完成”不能解释为“216 条可直接扩大”。大部分视觉几何
仍可复用，但正式认证池必须按每条动作分类重做或重选。

### card1 两住宅 CPU room-centric 小批

Apartment（每 profile 请求 6）：

- card1F 5/6、card1B 3/6；
- card5R/card6R/card7/card8/card9/card11/card16 均 6/6。

Kujiale（同一普通 seed，每 profile 请求 6）：

- card1F/card1B 0/6（分配到的联合格在当前预算内未找到，诚实
  `not_found_within_budget`）；
- card5R 4/6、card8 5/6；
- card6R/card7/card9/card11/card16 均 6/6。

更大联合格 smoke（每 profile 18）证明 Kujiale 不是绝对不可构造：card1F
1/18、card1B 3/18；Apartment 为 12/18、13/18。严格防泄漏约束显著降低
产能，但没有房间专用回退。

### 设计层跨场景

`/data/jzy/tmp/qa_v3_prescale_scene_generalization_20260902_v1.json`：

- 2 个场景资产、2 个后端、3 个路线域；
- Apartment：card1F 14/18、card1B 13/18；
- HM3D val 00800 floor0：6/18、6/18；
- HM3D val 00800 floor1：7/18、6/18；
- 全部是 `geometry_candidate`，没有冒充跨场景渲染或题目准入。

Studio 现场另有 334 个 HM3D `hm3d_end_to_end` 历史任务，其中 86 pass、
245 fail、3 interrupted。pass 任务证明多栋 HM3D 的 Habitat 视频、移动声源
双耳/FOA 和房间声学包可运行；但该模板使用 bookshelf speaker/moving-source
资产，不是 QA v3 的双语义动物、MCQ/Open、Gate A/B 或 native-pixel join。
因此这些 86 个任务是可复用的房间/声学底座，不计作 QA v3 多房间端到端
通过，也不能替代后续语义演员接入。

### 修复后 Kujiale card1F 完整链

候选：`card1F_002`。

- 75 帧 Kujiale baked-lit 视觉成功；
- native pixel：目标 `source2` 在 f40/f74 均为 `visible_occluded`；
- 目标全片只发 1 次锚定声；
- 锚角 9.415°、查询角 40.496°，锚角按现行宽评分带得 0；
- Gate A Open 金标 −31.711°，与主金标相隔 72.207°；
- main/Gate A 均为 80,000×2、16 kHz 双耳，133,416 个样本值不同，
  最大绝对差 0.0407314。

视觉：
`/data/jzy/tmp/qa_v3_prescale_kujiale_card1F_002_visual_20260902_v1`

像素：
`/data/jzy/tmp/qa_v3_prescale_kujiale_card1F_002_pixel_20260902_v1`

音频 main/Gate A：
`/data/jzy/tmp/qa_v3_prescale_kujiale_card1F_002_audio_main_20260902_v1`
`/data/jzy/tmp/qa_v3_prescale_kujiale_card1F_002_audio_gateA_20260902_v1`

### canonical appearance Gate B

card7 main 与 Gate B 各自完成 75 帧视觉，并以统一 0.61575 m 语义声源
高度独立渲染双耳。两份 80,000×2 波形逐位相同、SHA256 同为
`eea29abf9a13a50dfaac3ba9f82514a141f7cc6f1d1b074751d551e28d127c20`。

失败历史保留：Gate B 视觉 v1 因启动环境没有加载
`avengine_spear_ext`，在捕获前写出 failed receipt；未覆盖、未清理，改用
正确原生扩展路径的 fresh v2 后完成 75 帧。v1 不计成功证据。

证据：
`/data/jzy/tmp/qa_v3_prescale_canonical_card7_evidence_20260902_v1/gateb_representative_manifest.json`

这关闭了“换视觉资产但复用旧 muzzle 高度造成约 6 cm AV 不一致”的开发
缺口；正式配置必须显式传 canonical height，不能依赖默认资产锚点。

## 人类校准包

路径：
正式 18 题：
`/data/jzy/tmp/qa_v3_human_calibration_pack_run02_20260902_v3`

owner 三题预览：
`/data/jzy/tmp/qa_v3_human_calibration_preview_20260902_v2`

- 18 个 full-AV 项：card1F 6、card1B 6、card8 6；
- 只把 `public/` 作为网站目录，答案表独立放在
  `private/answer_key.json`，public URL 无法访问；
- 不使用原生 video controls，不显示时间轴、不允许拖动或调速；
- 每题最多从头播放两次，完整播放后才显示答案区，并把 `play_count`
  写入响应；
- 完成页同时提供下载、复制和只读 JSON 文本框，避免浏览器下载策略造成
  响应丢失；
- 每题先收绑定答案，再收数值和置信度；
- 评分器在全部试次报告绑定正确率，但只用绑定正确试次计算数值
  P75/P95；认错对象的大误差不会被错误吸收到容差中；
- 当前尚无受试响应，因此容差仍是占位参数，不得冻结大规模生成配置。

## released-media 自动捷径探针（旧 run02，n=6/profile）

输入只含题面、最终 WAV、video-only MP4 与发布金标。结果是小样本开发诊断，
不是认证：

- Text-only 基本落在经验常量/多数类基线；
- audio-only：card7 MCQ 0.667 vs 0.5，其他多数组合未稳定越过经验基线；
- video-only：card7 MCQ 1.0 vs 0.5、card7 Open 0.833 vs 0.5、
  card8 MCQ 0.5 vs 0.333、card9 Open 0.667 vs 0.5；
- 每格只有 6 条，不能做显著性或认证声明，但已经证明旧 run02 不能直接
  进入发布集，必须使用修复后配平池重跑。

结果：

- `/data/jzy/tmp/qa_v3_run02_text_shortcut_probe_20260902_v1.json`
- `/data/jzy/tmp/qa_v3_run02_audio_shortcut_probe_20260902_v1.json`
- `/data/jzy/tmp/qa_v3_run02_video_shortcut_probe_20260902_v1.json`

## 当前边界与下一步

自动化工程已达到“可以生成下一轮认证 pilot”的状态，尚未达到“可以大规模
生成正式数据”的状态。下一步只能按以下顺序：

1. 让真实受试完成校准包；
2. 审核 P75/P95 与完整 AV 绑定正确率，定稿评分参数；
3. 用定稿参数对更多可渲染住宅生成每 profile 百题级候选；
4. 完整渲染 main/Gate A/Gate B，card16 在 pixel join 后按四态选择；
5. MCQ/Open 分别跑预声明的 Text/A-only/V-only 探针与人类完整 AV；
6. 越线题重采或降对照，通过后才允许千级/万级生成。

本报告不声明：单模态最优策略失败、正式容差确定、人类可答性通过、场景
泛化成立、任何题型正式准入或数据集可发布。

## 给 Claude 的只读复核重点

请在服务器 HEAD 现场自行核实，重点回答：

1. 六个提交是否准确关闭 card1/card8/card11/card16 与 appearance Gate B
   的已知 P1/P2；
2. card1 的联合格分层是否造成不可接受的结构性产能塌缩，还是诚实
   `not_found` 加跨房间补采即可；
3. canonical emitter height 是否保持 endpoint 身份、视觉轨迹和音频物理
   关系，逐位相同的 appearance twin 是否可作正式政策；
4. card8 严格认证分与宽带诊断是否在评分器和事实记录中真正分开；
5. card16 post-pixel 选择器是否只使用金标状态、没有读取模型 outcome；
6. 人类包是否真的把绑定错误与数值容差分开；
7. released-media 探针是否越过 oracle 边界，结果措辞是否过度；
8. 是否还有会阻塞下一轮百题级认证 pilot 的 P0–P2。

只读审核，不修改、不提交、不启动作业。请区分“可开认证 pilot”与“可大规模
生成正式数据”两个判定。
