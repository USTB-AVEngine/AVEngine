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


---

# 第二轮（2026-09-02 下午）：百题认证 pilot 前的四项修复

> 角色：Claude 实现，Codex 只读独立审核。本节只记录做了什么、证据在哪里、
> 边界是什么。**本节不宣布百题 pilot 已认证、不宣布可以大规模生成、
> 不宣布人类容差已定稿。**

## A. 起止 HEAD

- 起点：`c52ca65`（核实过：worktree 干净，没有远端分支包含它）。
- 代码终点：`8713a78`（六个提交）；本文档的提交紧随其后，是本轮最后一个提交。worktree 干净，没有远端分支包含 HEAD，未推送。
- 唯一工作副本：`/data/jzy/tmp/wt-qa-v3-pilot`。

## B. 提交（按逻辑切片）

| 提交 | 内容 |
| --- | --- |
| `feccf73` | card8：`T_FULL` 成为显式必填参数，缺失即失败；最小首叫间隔改为严格大于 max(T_HALF, 2·T_FULL)（样本域整数比较）；调度器、自检、带边推导、事实记录、Gate A 时间分离、批次诊断、旧题目生成器、216 重验与 released-media 探针都记录实际执行的 T_FULL / T_HALF / 推导最小间隔 / 认证政策 / 宽带角色 |
| `eeb10b3` | card1：最终 timeline 成为验收权威。时间线定稿后重算主题与 Gate A 指代者的锚角、查询角、角距，fail-closed 检查锚角落分配锚带、查询角落答案带、锚角作 Open 答案得 0、MCQ 翻转、Open 得分区分离；fact 分记 `planned_generation_checks` 与 `realized_generation_checks`；216 重验只信 timeline，规划值只作 planning value 报偏差 |
| `c828823` | card1：native pixel 双侧验收。pixel join 支持 card1F / card1B，对 main 与 Gate A 指代者分别检查锚定帧可绑定、查询帧可辨识；阈值（可见比例 ≥ 0.5、可见像素 ≥ 1000、包围盒不贴边）显式进入 params、fact（`pixel_acceptance`）与 join 输出，全部标 placeholder；逐指代者/帧输出状态、比例、像素数、包围盒条件与精确拒因 |
| `d2f3919` | card1：只读条件基线报表工具 `report_qa_v3_card1_conditional_baseline.py`：按 profile × form × split × missing_modality 输出锚带×答案带联合表、结构性空格、条件答案分布、best-response 基线（MCQ 按实际可排除结构，Open 在实际两档评分器下网格搜索）、每房间贡献与单房间最大占比；参数快照进输出；输入若含模型结果字段直接拒绝 |
| `6fd5925` | 人类试听页禁用视频右键菜单（右键会重新露出原生控件） |
| `8713a78` | 基线工具输出里注明 best-response 是样本内最优、属乐观上界 |

每个提交时点都跑过顶层测试（排除 unit 与重型 `test_verify_audio_batch.py`）：
343 → 347 → 353 → 361 → 361 passed；相关 unit 15 passed。

## C. 四项实现要点

### C1. card8：显式 T_FULL 与安全首叫间隔

- 生产 params 原来没有 `T_FULL`，`audio_profiles` 只要求首叫间隔 > T_HALF，而正式 Open 按 strict T_FULL 判分。现在 `audio_profiles.card8_scoring_params` 是唯一入口：缺 `T_FULL` 或 `T_HALF` 直接抛错，T_HALF 不得窄于 T_FULL，推导出最小首叫间隔 = max(T_HALF, 2·T_FULL)，并带上 `certification_policy=strict_full_credit_only`、`wide_tolerance_role=diagnostic_only` 与 `T_FULL_status`（从 params 原样记录，缺省写 `unspecified_treat_as_placeholder`）。
- `schedule_first_call_bands`、`_self_check_first_call_bands`、`card8_band_edges` 都改用这条链；比较在样本域做整数比较，所以 T_FULL=0.6 时 1.1 s 拒、1.2 s 边界拒、多一个样本才过（测试 `test_card8_self_check_uses_strict_twice_t_full_boundary`）。
- `design_qa_v3_scene_batch.build_answer` 的 card8 事实：`open` 块带 T_FULL / T_HALF / T_FULL_status / 最小间隔 / 规则文本 / 政策 / 角色；`truth` 带实际首叫间隔；MCQ 块不带任何认证政策（测试 `test_card8_fact_records_scoring_chain_and_keeps_mcq_unaffected`）。Gate A 的时间分离阈值同样改为这条推导最小间隔。
- `materialize_derived_params` 现在接收 profiles：批次里有首叫题型（card8/card9）而 T_FULL 缺失就失败；纯 card1 批次记一句"没有推导"，不再静默用旧文本。manifest 的 `card8_time_domain` 记录整条参数链与实际首叫间隔的最小/最大值。
- 旧的 run01/run02 生成器 `generate_qa_v3_questions.py`：主流程开头就要求 T_FULL；首叫间隔不严格超过最小值的点不再出 card8 题，manifest 记录跳过的点。
- 216 重验与三种 released-media 探针的输出都内嵌 `scoring_params`。
- **未改的旧路径**：`filter_cross_time_points.py` 与 `design_qa_v3_pilot_batch.py`（run01/run02 的旧筛选与装配）仍只按 T_HALF 判 card8。它们不在当前房间中心生产链上，本轮没动，建议后续退役或同步。
- T_FULL 的值仍是占位：新 params 文件里写的是 0.5，状态字段为 `placeholder_research_pending_human_calibration`，终值等人类校准。

### C2. card1：最终 timeline 为验收权威

- 新函数 `design_qa_v3_scene_batch.realized_cross_time_checks`：在最终 timeline 上重算主题与 Gate A 指代者的锚角、查询角、角距，检查五条门（锚角落分配锚带、查询角落答案带、锚角作 Open 答案得 0、MCQ 翻转、Open 得分区分离），任一不成立就抛 `GenerationConstraintError`，候选进 rejections，不写事实。
- fact 里 `generation_checks` 保留原样以兼容旧读法，新增 `planned_generation_checks`（标 planning values only）、`realized_generation_checks`（含 planned_vs_realized 偏差）与 `acceptance_authority=realized_generation_checks`。
- 216 重验工具继续只从 timeline 计算，新增"执行锚角落分配锚带"与"执行查询角落金标带"两项检查；规划值只出现在 `planned_anchor_azimuth_deg_planning_value_only` 里。历史 216 条 fact 没有 `generation_checks`，所以偏差一项对它们为空，不是工具漏算。
- 失败测试：`test_realized_timeline_rejects_a_plan_that_only_passed_on_paper`（规划角距 31.08° > THETA_HALF，时间线角距 25.5° ≤ THETA_HALF，必须拒）、`test_realized_timeline_rejects_anchor_that_drifted_out_of_its_band`；通过测试用 Kujiale card1F_002 的执行值（锚角 8.451°、查询角 40.496°、规划值 9.415° → 偏差 0.964°）。
- **真实数据上的阳性证据**：fresh Kujiale card1 smoke（`qa_v3_prescale_card1_kujiale_20260902_v2`）里 card1B_008 被 realized 检查拒绝——规划锚角 −17.536° 刚好在分配带 [−52.5, −17.5) 内，执行锚角 −16.561° 出了带。这正是"踩线题规划过、成片不过"的实例。
- 口径：Kujiale card1F_002 的执行锚角是 8.451°；9.415° 只是 planning value。

### C3. card1：native pixel 双侧验收

- 新模块 `tools/qa/qa_v3_pixel_thresholds.py`：从 params 读三个显式阈值键（`PIXEL_MIN_VISIBLE_FRACTION`、`PIXEL_MIN_VISIBLE_PIXELS`、`PIXEL_BBOX_MUST_NOT_TOUCH_FRAME_EDGE`），缺一个就失败；生成 card1 fact 的 `pixel_acceptance` 块（两侧指代者、锚帧"可绑定"、查询帧"可辨识"、阈值、placeholder 状态、"视线只是预筛"的声明）。
- `join_qa_v3_extended_pixel.py` 支持 card1F / card1B：阈值优先取 fact 的 `pixel_acceptance`，也可用 `--params`；两者同时给且不一致就拒绝。逐指代者/帧输出 state、visible_fraction、visible_pixels、bbox、bbox 是否贴边、所用阈值、失败条件；拒因形如 `gatea_referent_query_frame_visible_pixels_below_threshold`。没有房间专用逻辑。
- 阳性拒绝证据：对历史 Kujiale card1F_002 的像素真值跑新 join，结果 `pixel_rejected`，拒因是 Gate A 指代者查询帧可见比例 0.105、可见像素 198（`/data/jzy/tmp/qa_v3_prescale_card1F_002_pixel_join_both_sides_20260902_v1.json`）；主题两帧与 Gate A 锚帧都过。
- fresh 样本的双侧像素结果见 D 节。

### C4. card1：实现条件表与 A-only 基线

- `tools/qa/report_qa_v3_card1_conditional_baseline.py` 读 facts.jsonl / 调度器矩阵 / 跨场景 smoke JSON；不新增任何门（输出里 `is_gate=false`），不要求每房间填满 18 格。
- A-only（missing_modality=video）：听者知道锚带，MCQ best-response = Σ_锚带 max_答案带 n / N，另给"只按非空格均匀猜"的结构排除基线与名义 1/3；Open 在每个锚带层用实际两档评分器网格搜索最优常数角，另报"直接复述锚角"的期望分。
- V-only（missing_modality=audio）：两只狗都看得见但不知谁最后叫，报单规则族（挑黄狗/挑黑白狗/挑动得多的/挑更靠中的/挑最左的……）里最好的一条；结构基线 1/2。
- Text-only：多数答案带与最优常数角。
- 每房间：各房间的联合表、A-only best-response、占比；单房间最大占比单列。
- 复现：跨场景 smoke 的 Apartment 表算出 card1F 6/14 = 42.857%、card1B 6/13 = 46.154%，与 Codex 独立复算一致；数字来自输入表，工具里没有写死。
- 输入若含 `model_answer`、`prediction`、`accuracy` 等字段直接拒绝；没有任何读探针结果的入口。
- 输出注明 best-response 是样本内最优、小样本下是乐观上界。

## D. fresh 产物（全部 research_candidate，均由 HEAD `6fd5925`/`8713a78` 的代码生成）

参数文件：`/data/jzy/tmp/qa_v3_prescale_params_tfull_placeholder_20260902_v1.json`
（sha256 `54e71135…635cc`）：生产 params + `T_FULL=0.5`（placeholder）+ 三个像素阈值（placeholder）。
题型目录：`/data/jzy/tmp/qa_v3_prescale_core_profiles_20260902_v1.json`（21 个 profile，由核心 v1 快照拼成）。

### D1. 核心小批（每 profile 请求 6）

Apartment `qa_v3_prescale_core_apartment_20260902_v2`：card1F 4/6、card1B 4/6、card5R 5/6、其余（card6R/card7/card8/card9/card11/card16）6/6。card5R_003 因距离关系差 0.08 cm 未达 50 cm 被拒。

Kujiale `qa_v3_prescale_core_kujiale_20260902_v2`：card1F 2/6、card1B 1/6、card5R 4/6、card8 5/6，其余 6/6。上一轮 Kujiale card1F/card1B 是 0/6，这次同预算下各拿到 2 与 1 个，仍是 `partial`，拒因以静止路线与锚带约束为主，完整分母在各自 batch_manifest。

所有 card1 事实都带 `realized_generation_checks`（全部通过，规划 vs 执行锚角偏差 0.001°–0.93°）与 `pixel_acceptance`；card8 事实的 `open` 块带完整参数链，实际首叫间隔最小 1.079 s（Apartment）/ 1.038 s（Kujiale），都严格大于 1.0 s。

### D2. card1 18 格 smoke

Apartment `qa_v3_prescale_card1_apartment_20260902_v3`：card1F 12/18、card1B 13/18。
Kujiale `qa_v3_prescale_card1_kujiale_20260902_v2`：card1F 1/18、card1B 0/18（`not_found_within_budget`，18 格评估 143,092 个组合、17 次预算耗尽；其中 card1B_008 是 realized 检查拒绝，见 C2）。

### D3. 216 条重验 v3

`/data/jzy/tmp/qa_v3_prescale_revalidation_216_20260902_v3.json`：72 pass / 12 pixel_pending / 5 metadata / 92 media / 11 geometry / 12 demoted / 12 future，与 v2 逐条相同；新增 `scoring_params` 快照与 card1 的执行角检查字段。

### D4. released-media 探针 v2

`/data/jzy/tmp/qa_v3_run02_{text,audio,video}_shortcut_probe_20260902_v2.json`：指标与 v1 逐组相同（文本探针的预测串因浮点打印顺序有差别，指标一致），新增 `scoring_params`。

### D5. 条件基线报表

- 复现：`qa_v3_prescale_card1_conditional_baseline_smoke_repro_20260902_v1.json`（6/14、6/13 见 C4）。
- 历史 card1 smoke（规划锚角，标 `planned_solver_value_no_realized_record`）：`..._historical_20260902_v1.json`。
- fresh（执行锚角）：`qa_v3_prescale_card1_conditional_baseline_fresh_20260902_v1.json`：card1F n=19，A-only MCQ best-response 10/19 = 52.6%（结构排除均匀 42.1%，名义 33.3%），A-only Open 最优常数角期望 0.816（样本内上界），复述锚角 0；card1B n=18，A-only MCQ 9/18 = 50.0%，Open 0.806。同带对角格中两端两格仍为空。V-only 最好的单规则是"挑更靠中的狗"（card1F 63.2%、card1B 77.8%，n 很小）。单房间最大占比 84.2%（card1F）/ 94.4%（card1B），都是 Apartment。

### D6. card1 双侧像素证据

目录 `/data/jzy/tmp/qa_v3_prescale_card1_pixel_both_sides_20260902_v1/`：对本轮全部 fresh card1 几何候选（Apartment 33 = 核心批 8 + 18 格 smoke 25；Kujiale 4 = 核心批 3 + smoke 1）逐个做 UE native pixel 捕获（只取锚帧与查询帧）并跑双侧 join；汇总在 `summary_both_sides_join_20260902_v1.json`，队列脚本、逐候选捕获日志与两次运行的队列日志都在同目录。像素权威仍是同相机 normal / target-only 深度对比（`same_renderer_same_camera_normal_vs_target_only_metric_depth_v1`）。

- **结果：37 个候选里 1 个双侧通过，36 个被拒。**
- 双侧通过样本：`qa_v3_prescale_card1_apartment_20260902_v3__card1B_009`（Apartment，card1B_009，目标 source2 黑白狗，锚帧 62、查询帧 22）。执行锚角 −25.717°、执行查询角 15.236°、角距 40.954°，Gate A 指代者查询角 −44.829°；规划锚角 −25.692°，偏差 0.025°。四个指代者/帧：主题锚帧可见比例 0.657、6149 像素；主题查询帧 0.787、1375 像素；Gate A 锚帧 0.676、1103 像素；Gate A 查询帧 0.694、2787 像素；包围盒都不贴边。它是唯一一条在占位阈值下双侧通过的 fresh 样本，只说明"这条链能产出通过样本"，不说明产能。
- 因 Gate A 指代者太小被拒：历史 Kujiale card1F_002（查询帧 198 像素、0.105，见 C3）；fresh 候选里有 19 条同类拒绝，例如 Apartment 核心批 card1F_003——主题两帧都过（0.679 / 3710 像素，0.779 / 3086 像素），Gate A 指代者锚帧可见比例只有 0.389，查询帧被完全遮挡（0 像素）。
- 拒因分布（37 条里）：主题锚帧失败 27 次、主题查询帧 26 次、Gate A 锚帧 23 次、Gate A 查询帧 28 次。148 个指代者/帧的像素状态：visible_occluded 95、fully_occluded 42、visible_clear 9、out_of_view 2。
- 阈值敏感度（只是诊断，阈值没改）：即使把三条阈值放松到可见比例 ≥ 0.1、像素 ≥ 200 且允许贴边，也只有 8/37 通过；保持"不贴边"时任何放松最多 2/37。也就是说问题主要不是阈值严，而是候选本身被家具挡住或出画。
- Kujiale 本轮没有双侧通过样本：4/4 全部被拒（核心批 card1F_005 主题锚帧可见比例 0.269、Gate A 指代者只有 996 与 606 像素；card1F_006 与 card1B_005 是包围盒贴边；smoke card1F_012 主题查询帧不可见）。分母完整保留在汇总里。
- 视线筛查在这些结果里只是搜索期预筛：Kujiale 开了视线预筛仍有 4/4 被像素拒，Apartment 没开预筛则 42/148 个指代者/帧完全被挡。像素可答性只能由像素真值判。

## E. 测试

- 定点测试：见 B 节各提交对应的测试文件（`test_audio_profiles.py`、`test_gatea_generation.py`、`test_qa_v3_extended_pixel_join.py`、`test_qa_v3_prescale_candidate_audit.py`、`test_qa_v3_released_modality_probe.py`、`test_generate_questions.py`、`test_qa_v3_card1_conditional_baseline.py`、`test_qa_v3_human_calibration.py`、`test_run02_selection_and_visual_verification.py`）。
- 顶层 tests（排除 unit 与 `test_verify_audio_batch.py`）：最终 HEAD 上 361 passed。
- 相关 unit（`test_apartment_slot_bindings.py`、`test_qa_v3_audio_batch.py`）：15 passed。

## F. 失败与边界

1. 像素队列第一次运行在第一个 card1B 候选处失败：我把锚帧 62、查询帧 22 按原顺序传给 `--frame-index`，而捕获工具要求帧索引递增。两个队列按"失败即停"停下，没有产生半成品目录；失败日志保留在 `*card1B_001.capture.log` / `*card1B_005.capture.log`，第一次运行的队列日志改名为 `queue_*.run1.log` 保留。修正排序后重启，已完成的候选被跳过。
2. 重启前判断"队列是否还在跑"时，grep 模式匹配到了同一条 ssh 命令行里的重启命令本身，误报"仍在运行"，第一次没有重启成功。改成行首锚定的 `pgrep -f "^bash pixel_queue.sh …"` 并用 ps 的 etime 复核后才重启。这是记忆里记过的自匹配坑再现。
3. 产能：Kujiale card1B 18 格 smoke 0/18（`not_found_within_budget`，其中 1 条是 realized 检查拒绝）；Kujiale 核心批 card1F 2/6、card1B 1/6，仍是 partial。Apartment 双侧像素通过 1/33，Kujiale 0/4。
4. 本轮暴露、但不在四项任务书内、**没有改**的几何缺口：Apartment 场景配置没有视线预筛（`line_of_sight_screened=false`），像素结果里 42/148 个指代者/帧完全被家具遮挡；求解器只检查水平视锥，且最小相机距离只在查询帧检查——Apartment 核心批 card1F_001 的目标在锚帧离相机 0.73 m、俯角 63.6°，直接掉出画面下沿（像素状态 out_of_view）。这两处建议进入下一轮任务书。
5. 扩展题型（card11 / card16）的 batch_manifest 没有 `code` 字段，无法从产物本身核对代码版本；这是既有行为，本轮没改。
6. 旧的 run01/run02 路径（`filter_cross_time_points.py`、`design_qa_v3_pilot_batch.py`）未同步 T_FULL（见 C1）。
7. T_FULL=0.5 与三条像素阈值全部是占位值；本轮没有产生、也没有伪造任何人类结果。
8. 探针 v2 与 v1：文本探针的预测串有浮点打印差异，逐组指标完全一致；音频、视频探针预测与指标都一致。
9. 未 push；所有产物 research_candidate；没有新增与四项失败无关的 hash / contract / baseline / gate（条件基线工具是只读报表，`is_gate=false`）。

## G. 给 Codex 的审核清单

1. `audio_profiles.card8_scoring_params` 是否是 card8 首叫链唯一入口；`schedule_first_call_bands`/`_self_check_first_call_bands`/`card8_band_edges`/`build_answer`/`audit_gatea_pair` 是否都走它；样本域整数比较是否与 T_FULL=0.6 的三个边界用例一致。
2. `realized_cross_time_checks` 的五条门是否恰好对应任务书第二项，是否只在 card1F/card1B（forward/backward + azimuth_band）上启用，没有波及 card5R 等。
3. `planned_generation_checks` 与 `realized_generation_checks` 的口径是否在 fact、216 重验、条件基线工具三处一致（规划值只作 planning value）。
4. pixel join 的 card1 分支是否没有房间专用逻辑；阈值三处（params / fact / join）是否一致且都标 placeholder；bbox 贴边判定 `x0 > 0 and y0 > 0 and x1 < W and y1 < H` 是否与 run02 的 `pixel_eligible` 一致。
5. Kujiale card1F_002 的阳性拒绝与 fresh 样本的双侧结果是否支持"LOS 只作预筛"的口径。
6. 条件基线工具：MCQ best-response 是否按实际非空格算；Open 网格搜索是否用了实际评分器的两档；是否确实读不到模型结果；单房间占比是否单列。
7. 旧路径 `filter_cross_time_points.py`/`design_qa_v3_pilot_batch.py` 未同步 T_FULL，是否需要退役或补齐。
8. 本轮暴露的两处几何缺口（求解器只查水平视锥、最小相机距离只查查询帧不查锚帧）是否应进入下一轮任务书。

## H. 远端状态

未 push。`git branch -r --contains HEAD` 为空；`origin/main` 仍是 `e39c2b7`，与本轮无关。
