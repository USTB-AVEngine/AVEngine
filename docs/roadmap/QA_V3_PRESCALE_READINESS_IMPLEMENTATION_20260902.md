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


---

# 第三轮（2026-09-02 傍晚）：机位堵死的根治与遮挡分档

> 起因：owner 看过 37 条 fresh card1 候选的像素拼图后拍板——狗走到家具后面、
> 狗掉出画面下沿都可以接受，按难度分档；相机机位紧贴大件家具、开场就什么都
> 看不见的候选是生成阶段的错误，要在生成期解决。本节记录为此做的三个提交、
> 对照证据和还没做的事。**本节同样不宣布 pilot 认证、放量或容差定稿。**

## A. 起止 HEAD

起点 `2b40afd`（第二轮文档提交）。代码终点 `61c9b1c`；本节文档提交紧随其后。
worktree 干净，未 push，`git branch -r --contains HEAD` 为空。

## B. 提交

| 提交 | 内容 |
| --- | --- |
| `a482ad3` | 锚定帧最小相机距离底线 `MIN_CAMERA_DISTANCE_ANCHOR_CM`（显式参数，缺省不启用，旧 params 行为不变）；realized 检查里新增每个指代者每帧的取景几何（距离、俯角、狗的落脚点是否投影在画面内），只记录不作门 |
| `15f0b1d` | card1 像素验收新增分档政策 `camera_blockage_reject_then_tier`：按可见比例分轻/中/重遮挡，完全被挡和出画各单独一档，只有挡住目标的物体离相机不到 `PIXEL_CAMERA_BLOCKAGE_MAX_DISTANCE_M` 才硬拒；从深度通道算出每个指代者每帧的挡住物中位距离；对捕获到的所有帧算每只狗的可见时间线；旧的阈值硬拒政策保留可选 |
| `61c9b1c` | 新工具 `preflight_camera_clearance_depth.py`：对候选机位不放狗渲染一帧深度图，按相机高度与目标高度算出"目标带"，报告 1.0/1.5/2.5 米内被挡的列占比，给出机位是否净空的占位判定；同时存降采样深度供离线换指标 |

每个提交时点顶层测试均通过：363 → 368 → 373 passed。

## C. 为什么没有做原计划的二维几何预筛

我先用 Apartment 路线库的 144,975 个路线采样点栅格化成 5 厘米可行格，对这
37 条候选的相机算"前方最近障碍距离"和"2.5 米内被挡射线比例"，再和人工标出
的 13 条机位堵死候选对照（`/data/jzy/tmp/qa_v3_prescale_round2_camera_2d_proxy_calibration_20260902_v1.txt`）。
最好的设置也只抓到 11 到 13 条里的同时错杀 20 条正常机位里的 7 到 18 条。
原因有两个：没有路线走过的地板都被当成了障碍；二维格看不见高度，灯罩这种
脚下一根杆、头顶一大片的东西和沙发靠背、岛台这种半高家具都判不出来。按
"阳性对照不过就不上线"的规矩，这条没有进管线。

## D. 引擎深度预检（替代第一层）

工具对同一批 37 条候选的机位各渲染一帧无角色深度图（Apartment 33 条在
`qa_v3_prescale_camera_clearance_preflight_apartment_20260902_v3/`，Kujiale 4 条在
`..._kujiale_20260902_v2/`，每条约 0.3 秒，加上启动共两分多钟）。

关键发现：眼高那一带（画面中间三分之一）对 1.47 米高的相机是空的，但沙发靠背、
岛台这类 0.4 到 0.7 米外的半高家具挡住的是画面下半部——0.5 米高的狗站在 2.5 到
10 米外，正好投影在画面 56% 到 90% 的高度区间。所以统计带改成按相机高度和
目标高度算出的"目标带"。对照结果：

| 判据 | 抓到像素 join 判定的机位堵死 | 误报其余机位 | 抓到人工标注 |
| --- | ---: | ---: | ---: |
| 目标带被挡列占比 @1.5 m > 0.1 或 > 0.2 | 16/16 | 0/21 | 12/13 |
| 整帧 1.5 m 内像素占比 > 0.1 | 16/16 | 0/21 | 12/13 |
| 眼高带被挡列占比 @1.5 m > 0.3 | 6/16 | 0/21 | 6/13 |

人工标注漏掉的那 1 条是 card1B_016：相机站在门口、门扇在 2.1 米，超出 1.5 米
的占位距离；把近距放到 2.5 米能抓到它，但会连带误报 6 条以上正常机位。工具
默认判据设为目标带 @1.5 m ≤ 0.2，仍是占位，等更多房间的数据再定。

对相机高度的回答：预检本身就在实际相机高度渲染，所以看得见眼高遮挡；目标带
也是用相机高度和目标高度算的，换相机高度或换动物身高只需改参数。

## E. 分档政策在 37 条候选上的效果

用 v2 参数对已有 37 个像素目录重新 join（`pixel_join_tier_policy_v2.json`，历史
card1F_002 在 `qa_v3_prescale_card1F_002_pixel_join_tier_policy_20260902_v1.json`）：

- 21 条通过、16 条因机位堵死被拒；旧政策是 1 条通过、36 条被拒。
- 16 条被拒里 12 条是人工标注的机位堵死，另外 4 条（card1F_007、card1F_009、
  card1B_001、核心批 card1F_001）是近处物体挡住了狗的大部分（可见比例 0.04 到
  0.2，挡住物离相机 1.09 到 1.34 米），人工标注时因为狗没被完全挡住而漏标；
  漏抓的 1 条仍是那扇 2.1 米的门。
- 21 条通过里最重的档位：轻 3、中 4、重 3、完全被挡 10、出画 1；其中 7 条在狗叫
  的瞬间看不见那只狗，6 条在查询瞬间看不见。这正是 owner 决定要保留的内容，
  但它们能不能被人答对还没有数据。owner 于 2026-09-02 晚进一步明确：Gate A
  对照狗整段视频从未露面的样本也**留作难题**，不加"两只狗都至少出现过"的底线。
- Kujiale card1F_002 在分档政策下通过，最重档位是"重遮挡"（Gate A 指代者查询帧
  可见比例 0.105，挡住物离相机 5.7 米，是窗帘）。

## F. 锚定帧距离底线与 fresh smoke

v2 参数（`/data/jzy/tmp/qa_v3_prescale_params_tfull_placeholder_20260902_v2.json`，
sha256 `d75a4e3b…5fb69`）把锚定帧最小相机距离设为 250 厘米并开启分档政策。

新 smoke：Apartment `qa_v3_prescale_card1_apartment_20260902_v4` card1F 12/18、
card1B 14/18（锚帧距离底线各拒 9,080 与 5,339 次尝试）；Kujiale
`qa_v3_prescale_card1_kujiale_20260902_v3` card1F 2/18、card1B 2/18（上一轮 1/18
与 0/18）。深度预检对这 30 个机位的判定：Apartment 14/26 净空，Kujiale 2/4 净空
（`qa_v3_prescale_camera_clearance_preflight_{apartment_v4smoke,kujiale_v3smoke}_20260902_v1/`）。

对这 30 条候选做了多帧像素捕获（锚帧、查询帧加每 5 帧一张，共 16 到 18 帧，
目录 `/data/jzy/tmp/qa_v3_prescale_card1_pixel_timeline_20260902_v1/`，汇总
`summary_tier_policy_timeline_20260902_v1.json`），再用分档政策 join：

- 16 条通过、14 条因机位堵死被拒。通过的最重档位：轻 5、中 1、重 4、完全被挡 6；
  其中 4 条在狗叫的瞬间看不见那只狗，5 条在查询瞬间看不见；13 条的两只狗在
  至少一半的捕获帧里可见。
- 深度预检与像素 join 的机位判定在 30 条里 28 条一致：13 条两边都判堵死，15 条
  两边都判净空。不一致的两条：card1B_001 预检净空但 join 判 Gate A 指代者锚帧被
  1.5 米内的物体挡住大半（狗只露 30% 左右，属"近处物体挡住一半"的边界情况）；
  card1B_009 预检的目标带被挡列占比正好 0.20 落在阈值上被判堵死，join 则通过并
  记为"完全被挡"档。这两条说明 1.5 米与 0.2 都是要敲定的占位值，不是分歧。
- 对比第二轮：旧政策下 37 条只过 1 条；现在锚帧距离底线加分档政策，30 条过
  16 条，而且被拒的 14 条全部是机位问题，是预检在生成期就能拦住的那一类。

## G. 还没做、以及边界

1. 预检现在是候选出来之后的一道过滤，还没有接进求解器。要让配额不被机位堵死
   吃掉，下一步要么把预检结果做成每间房的机位净空表给求解器查，要么在调度器
   里加"候选→预检→补采"的循环。
2. 分档只用了锚帧和查询帧两帧的像素；多帧捕获（每 5 帧一张）已经在跑，可见
   时间线字段已经落在 join 输出里，但"按时间线分档"的规则还没写，等人类数据。
3. 1.5 米、0.2、[0.5, 0.2] 这些数字全是占位，只经过本轮 37 条候选的对照。
4. Codex 上一轮要求的"查询帧可辨识"硬门与 owner 的分档决定冲突。owner 于
   2026-09-02 晚明确：冲突以 owner 为准。提交 `658c813` 把分档政策改为缺省，
   旧的两帧阈值硬拒政策只能通过 `PIXEL_ACCEPTANCE_POLICY` 显式点名；所有占位
   参数值暂按当前写法沿用，等人类数据再改。
5. 没有渲染任何完整视频或音频；没有人类数据；未 push。

## 远端

owner 于 2026-09-02 晚指示：整条工作线推送到新分支 `origin/qa-v3-prescale-20260902`，
`main` 不动。这条线自 `cdda524` 分出，与 `origin/main` 上后来的 5 个声音素材库提交
没有文件重叠，合并预演零冲突；合入 main 等 Codex 审完再议。

## H. 给 Codex 的补充审核点

1. `_too_close_at_anchor` 只在 forward/backward 求解器生效，其它题型没有锚帧概念，是否合理。
2. `frame_geometry` 只是记录，不是门；`base_projects_inside_frame` 的投影公式是否与渲染合同一致。
3. 分档政策的硬拒条件"中/重/完全被挡 且 挡住物 ≤ 1.5 m"是否会把合法的近景遮挡也拒掉。
4. 预检的目标带定义（相机高 1.47 m、目标高 0.5 m、2.5 到 10 m）是否应随每间房的实际参数而变。
5. 预检、像素 join、人工标注三方的 16/16、12/13 对照是否足以让预检替代二维几何预筛。


---

# 第四轮（2026-09-02 夜）：机位净空表进求解器、预测可见性、Codex 三条 P0

> 起因：Codex 对 `c52ca65..23eef13` 的只读审核给出三条 P0——机位净空要在求解时判、
> 静止路线预过滤与 card1 联合预算、扩展题型 manifest 记代码版本——owner 同日拍板了
> 三项裁定（1.5 m 硬拒先兜底后降诊断；相机高度按机位做确定性第二次尝试；每 5 帧
> 时间线改叫抽样）并认可"每格 4 面立方体全景、房间算一次、换题型换资产不重渲"的
> 求解器方案。本节记录为此做的提交、实测数字和还没做的事。**本节同样不宣布 pilot
> 认证、放量或容差定稿。**

## A. 起止 HEAD

起点 `23eef13`（Codex 审核范围的终点）。代码终点 `4b9ef3b`；本节文档提交与资产政策方案文档紧随其后。
worktree 干净。

## B. 提交（按逻辑切片）

| 提交 | 内容 |
| --- | --- |
| `273ce15` | card11/card16 批次 manifest 写入代码版本与脏树标记；运行中途失败也留下带版本与错误的 manifest（Codex P0 ③） |
| `4d13509` | 相机水平视场角改为从场景合同读（`scene_sampler.scene_hfov_deg` 唯一入口）；深度预检加 `--scene-config/--hfov-deg` 并记录来源；捕获相机助手的视场角与渲染目标尺寸参数化 |
| `a2a754f` | 机位净空表生成工具 `build_qa_v3_camera_clearance_table.py` 与共享几何/读取模块 `camera_clearance.py`（见 C1） |
| `871423e` | 七个求解器与 N 角色规划器在解出朝向后立即查表；相机高度回退；fact 与 manifest 记录证据；`CAMERA_CLEARANCE_REQUIRED` 缺表即停 |
| `6b8c85d` | 双运动题型的静止目标路线预过滤（opt-in，card6R 不受影响） |
| `403d269` | manifest 的联合配额表（槽位×锚带×答案×运动格：请求/填满/耗尽原因）与路线池报告 |
| `9c95687` | 预测可见性模块 `visibility_prediction.py` 与像素真值对照工具 `validate_visibility_prediction.py` |
| `10876b6` | 每条候选在最终时间线上预测两只狗逐帧可见性写进 fact；题型声明 `visual_requirements`（tier 只记档、reject 拒）；manifest 记预测档位分布 |
| `163fdcf` | 两条旧装配器加 `--historical-reproduction` 开关；选角助手挪到中性模块 `qa_v3_actor_selection.py`；像素 join 的全片可见性块改名 `visibility_timeline_sampled` 并记捕获步长 |
| `aef8523` | 预测可见性的撒点改按体宽（±0.2 m）而不是体长（±0.4 m），依据 D2 的对照 |
| `4b9ef3b` | 相机朝向直接从区间内的净空 2° 档抽样（无表时仍是均匀抽样）；1.8 m 只在区间内 1.47 m 无净空档时才用；扩展题型 manifest 记筛查、表身份、回退次数 |

每个提交时点顶层测试均通过：374 → 379 → 393 → 398 → 402 → 409 → 412 passed（unit 15 passed 不变）。

## C. 实现要点

### C1. 机位净空表（房间算一次）

- **渲什么**：对求解器可能抽到的每个相机点（`load_scene` 的 `camera_points`，Apartment 5757 个、Kujiale 598 个），
  在场景相机高度和 1.8 m 各渲 4 张 90° 视场、512×768 的无角色深度面（世界 yaw 0/90/180/270），
  拼成一圈 360° 深度全景。表按舞台包版本、场景、相机点、高度索引，房间不改就不重算。
- **两个实测发现**（都有现场证据，改了实现）：
  1. UE 的 `FOVAngle` 对竖长渲染目标仍绑定水平轴：512×768 的 90° 面与 768×768 的 90° 面中央 512 列逐像素一致
     （中位相对差 0.0，`/data/jzy/tmp/qa_v3_fov_axis_probe_20260902_v1.npz`）。竖向视场 112.6° 也实测成立
     （tan_v=1.5 处中位差 0.0005，其余假设 ≥ 1.5%）。生产相机 1280×720、105° 的横竖假设同样实测成立。
  2. 引擎深度缓冲 `sp_depth_meters_` 是**径向距离**不是平面深度。按平面深度重投影时误差随朝向偏离面轴增长
     （中位 15–22%），按径向重投影后 12 条直接渲染对照的中位误差 0.2–0.4%、98% 像素在 5% 以内。
- **摘要**：从全景按真实 105°×72.5° 视场重投影任意朝向，2° 一档算目标带（0.5/1.0/1.7 m 目标在 2.5–10 m 投影的行带）
  与眼高带的被挡列占比（列中位深度 < 1.0/1.5/2.5 m，与预检同一定义）。任意朝向的精确值随时可从存下的面重算。
- **验证**：随机 (点, 朝向) 直接渲染与表对照。24 点 smoke：判定一致 12/12（精确朝向与 2° 档都是），
  占比绝对差中位 0.0008、最大 0.08。完整表的对照数字见 D1。
- **代价**：单点 4 面约 0.28 s（比原来 0.57 s 单张预检还快，因为一次步进读四张）；Apartment 一遍 27 min，
  两个高度 54 min；摘要 16 进程 19 min（Apartment）；存储 3.3 GB + 0.34 GB。
- **消费者**（`camera_clearance.CameraClearanceTable`）：按点查被挡占比、净空朝向掩码、无净空朝向的点、
  以及任意视线的第一个障碍距离（供预测可见性）。

### C2. 求解器接入（`scene_sampler`）

- 场景配置多一个键 `camera_clearance_table`；`load_scene` 拒绝别的房间的表、覆盖不全的表、缺场景相机高度的表。
- 七个求解器与 `find_n_route_plan` 在解出 yaw 后立刻 `screen_camera_clearance`：场景高度堵死→按
  `CAMERA_HEIGHT_FALLBACK_M` 顺序试表里的其它高度→都堵记 `camera_clearance_blocked` 换机位。只看表，不看答案。
- `PointPlan` 带 `camera_height_m` 与 `camera_clearance` 证据；两个批次生成器按该高度放相机与听者，
  fact `camera` 块记 `height_m/scene_camera_height_m/clearance`，manifest 记 `camera_clearance_screened`、
  表身份、`camera_height_fallback_used`。没有表时行为不变，fact 写明"未筛"。
- params 新键（v3）：`CAMERA_CLEARANCE_REQUIRED`、`CAMERA_CLEARANCE_TARGET_HEIGHT_M 0.5`、
  `CAMERA_CLEARANCE_NEAR_M 1.5`、`CAMERA_CLEARANCE_BLOCKED_FRACTION_MAX 0.2`、`CAMERA_HEIGHT_FALLBACK_M [1.8]`，
  全部占位；缺键或不在表的网格上 fail-closed。

### C3. 静止路线预过滤与联合配额

- `ROUTE_PREFILTER_STATIC_TARGETS`（opt-in）：六个拒绝静止目标的求解器改从移动路线池抽目标；约束不变，
  逐次拒绝保留兜底；card6R 仍用全库。manifest `route_pool` 记池大小。
- manifest `cell_budget`：每个题型的 槽位×锚带×答案×运动格 键的请求/填满/耗尽原因；被拒格带分配信息；
  不足留在本房，不跨房回填。

### C4. 预测可见性（不拒题，记档）

- 在最终时间线上对两只狗逐帧撒九条视线（三个高度×三个横向偏移）查全景第一个障碍，另一只狗按圆柱体近似；
  得到逐帧预测可见比例与档位（沿用 join 的 0.5/0.2 阶梯）、锚帧/查询帧档位、全片统计
  （露面比例、叫的瞬间前后是否露面、查询前被挡多久、全程不露面）。
- 题型声明 `visual_requirements`（profiles v2）：card1F/1B 两狗锚帧与查询帧、card5R/6R 比较帧等都是 `tier`；
  card7/card9 目前也是 `tier`，等 D2 的对照通过再改 `reject`。card8 不设。
- 狗身尺寸 `PREDICTION_BODY_M` 占位（0.5 m 高、0.8 m 长、0.4 m 宽；横向撒点按体宽，见 D2）。

## D. fresh 产物与实测数字

### D1. 两房净空表

Apartment（`/data/jzy/tmp/qa_v3_camera_clearance_table_apartment_20260902_v1`，代码 `163fdcf`）：

| 项目 | 数值 |
| --- | ---: |
| 相机点 | 5757 / 5757（全覆盖） |
| 相机高度 | 1.471 m 与 1.8 m |
| 渲染 | 0.283 s/点，两遍共 54 min；引擎总墙钟 55 min |
| 摘要（16 进程） | 19 min |
| 全程墙钟 | 75 min |
| 存储 | 3.3 GB（180 个分片，float16，两高度） |
| 有净空朝向的点 @1.47 m | 4735 / 5757（82%），平均净空朝向占 43% |
| 有净空朝向的点 @1.8 m | 5348 / 5757（93%），平均净空朝向占 59% |
| 直接渲染对照（40 条随机点×朝向） | 判定一致：精确朝向 40/40，2° 档 39/40（唯一分歧 0.197 对 0.203，正压 0.2 线上） |
| 占比绝对差 | 精确朝向中位 0.0008、最大 0.016；2° 档中位 0.0012、最大 0.024 |
| 重投影深度误差 | 中位 0.19%，最大 0.31% |

抬到 1.8 m 让"整个点没有任何净空朝向"的比例从 18% 降到 7%，与第三轮 14 个堵死机位里 7 个能靠 1.8 m 救回的实验一致。

Kujiale `/data/jzy/tmp/qa_v3_camera_clearance_table_kujiale_20260902_v1`（代码 `aef8523`）：598 / 598 个相机点，1.47 m 与 1.8 m，
0.36 s/点，引擎墙钟 7.8 min，摘要 2.8 min，全程 10.7 min，341 MB（19 个分片）。有净空朝向的点两个高度都是 585 / 598（98%），
平均净空朝向占 63%（1.47 m）与 64%（1.8 m）——这间客厅的问题从来不是机位堵死，而是路线库一半是静止路线。
直接渲染对照 40 条：精确朝向 40/40，2° 档 39/40（分歧 0.191 对 0.203，同样压在 0.2 线上）；重投影深度误差中位 0.19%。

### D2. 预测可见性对照（已捕获像素真值的候选）

对 59 条已捕获像素真值的 Apartment 候选（37 条双帧 + 26 条 16 帧，共 1020 帧行，948 行两侧都在视野内）逐帧比对预测可见比例与渲染器可见比例（`qa_v3_visibility_prediction_validation_apartment_20260902_v{1,2}.json`）：

| 指标 | 撒点横向 ±0.4 m（v1，按体长） | 撒点横向 ±0.2 m（v2，按体宽，已改为缺省） |
| --- | ---: | ---: |
| Pearson / Spearman | 0.932 / 0.861 | 0.955 / 0.841 |
| 绝对误差中位 / 均值 | 0.037 / 0.086 | 0.024 / 0.067 |
| 0.5 分界一致 | 820 / 948（86.5%） | 886 / 948（93.5%） |
| 五档一致 | 69.5% | 75.2% |
| 真隐藏→预测可见 | 12 | 2 |
| 真可见→预测隐藏 | 77（其中 92% 真实可见比例 < 0.2） | 112（同样几乎全是 < 0.2 的"重遮挡"） |
| 出画一致 | 97.1% | 97.1% |

读法：预测把"几乎看不见"的狗一律判成隐藏，把轻遮挡偶尔判成中遮挡，方向上偏保守；真隐藏被预测成可见的几乎没有。
它够用来做档位配额与统计，不够用来单独拒题——card7/card9 的声明因此仍留在 `tier` 模式。
按距离分段：2–3 m 段偏差最大（0.5 分界一致 86%），6 m 以外几乎无偏差。

Kujiale 8 条候选、152 帧行、143 行两侧在视野内（`qa_v3_visibility_prediction_validation_kujiale_20260902_v1.json`，按体宽撒点）：
Pearson 0.959、Spearman 0.943、绝对误差中位 0.0、0.5 分界一致 135 / 143（94.4%）、五档一致 84.2%、真隐藏→预测可见 0、真可见→预测隐藏 6。

### D3. 两房 canary（净空表进求解器后的 fresh 批）

输入：params v3（`qa_v3_prescale_params_tfull_placeholder_20260902_v3.json`，净空/回退/预过滤/狗身占位键）、题型 v2
（`qa_v3_prescale_core_profiles_20260902_v2.json`，13 个题型的 `visual_requirements` 全是 tier）、带表的场景配置
（`qa_v3_scene_configs_clearance_20260902_v1/`）。核心批每题型 6 格（card1F/1B/5R/6R/7/8/9/11/16），card1 smoke 每题型 18 格。

**v1（代码 `163fdcf`/`aef8523`：先抽朝向再查表）**

| 批次 | 结果 |
| --- | --- |
| Apartment 核心 | card1F 3/6、card1B 3/6，其余 7 个题型 6/6；36 条 scene-batch fact 全部筛过表并带预测可见性；相机高度 1.471 m 23 条、1.8 m 13 条 |
| Apartment card1 smoke | card1F 10/18、card1B 11/18；21 条 fact 全筛过，8 条用了 1.8 m |
| Kujiale 核心 | card1F 0/6、card1B 0/6、card5R 3/6、card8 5/6，其余 6/6；路线池 200/400（静止路线已剔除） |
| Kujiale card1 smoke | card1F 2/18、card1B 0/18 |

读法：
- 拒因里 `camera_clearance_blocked` 占 card1 尝试的四分之一到三分之一（Apartment card1B 2378/9483，Kujiale card1B 8916/54000），
  这是"先抽朝向再查表"的浪费——它促成了 `4b9ef3b`：直接从区间内的净空档抽朝向，堵死不再消耗尝试，1.8 m 只在整个区间在 1.47 m 都堵死时才用。
- Apartment card1 的 36 格 smoke 里 21 条通过且机位全部净空；第三轮同样 36 格里 30 条通过但 14 条机位堵死，
  可用的其实只有 16 条。也就是说净空表进求解器后，**可用产量从 16/36 提到 21/36，而且不再需要事后像素筛机位**。
- Kujiale 的 card1 产能仍接近零，主拒因是 `camera_too_close_to_target`（2.5 m 底线）与锚带分配，不是机位堵死：
  这间客厅只有 200 条移动路线、地面约 5.8 m × 11.8 m，card1 的联合约束在这里几乎不可满足。联合配额表如实报 18 格 18 个未填。
  这是房间容量问题，要由 owner 决定放宽 2.5 m 占位、补路线库，还是接受该房不出 card1。
- 预测档位分布已进 manifest（例：Apartment card1F 主狗查询帧 light 4 / medium 3 / heavy 2 / hidden 1）；
  "对照狗全程不露面"各出现 1 条（card1F、card1B、card9），按 owner 裁定留作难题单列。

**v2（代码 `4b9ef3b`：从净空档抽朝向）**

| 批次 | 结果 |
| --- | --- |
| Apartment 核心 | card1F 2/6、card1B 3/6，其余 7 个题型 6/6；35 条 scene-batch fact 全筛过并带预测；相机高度 1.471 m 29 条、1.8 m 6 条；card11/card16 的 manifest 也记了筛查（回退 2 与 1） |
| Apartment card1 smoke | card1F 11/18、card1B 12/18；23 条 fact 全筛过，其中 11 条用 1.8 m |
| Kujiale 核心 | card1F 0/6、card1B 0/6，其余 7 个题型 6/6（v1 里 card5R 3/6、card8 5/6 这次都填满）；30 条 fact 全在 1.47 m，无回退 |
| Kujiale card1 smoke | card1F 1/18、card1B 1/18 |

读法：
- 从净空档抽朝向后，`camera_clearance_blocked` 的含义变成"这个相机点在答案带解出的整个朝向区间里、两个高度都没有净空档"，
  是真实约束而不是抽样浪费；它仍是 card1 的主要拒因之一（Apartment card1 smoke 里约 1/5 的尝试），说明 card1 的答案带把相机
  朝向压得很窄，而 Apartment 一半左右的机位朝向被家具挡住。
- Apartment card1 smoke 36 格通过 23 格（v1 21，第三轮可用 16），全部机位净空，不再需要事后像素筛机位。
- **相机高度回退占比高**：Apartment card1 通过的 23 条里 11 条用了 1.8 m，核心批 35 条里 6 条。按 owner 裁定这是确定性的几何回退，
  fact 与 manifest 都记了；但比例接近一半，值得 owner 与 Codex 一起看：是接受两种高度混用，还是把 1.8 m 改成场景级实验臂。
  Kujiale 全部 1.47 m，没有回退。
- Kujiale card1 仍接近零，与 v1 结论一致：主拒因 `camera_too_close_to_target`（2.5 m 占位）和锚带分配，房间容量问题。
- 预测档位分布进了每个 manifest；"对照狗全程不露面"在两房合计出现 6 条（card1F 1、card1B 1、card5R 1、card6R 2、card7 1、card8 主狗与对照狗各 1），按裁定单列难题。
- 所有 fact 与 manifest 记代码版本 `4b9ef3b`、dirty=false。

### D4. 放宽距离底线的对照实验（owner 2026-09-03 允许）

把查询帧与锚帧的相机距离底线从 250 cm 改为 200 cm（实验参数 `qa_v3_prescale_params_floor200_experiment_20260902_v1.json`，其余同 v3），
重跑 Kujiale card1 18 格 smoke（`qa_v3_canary_card1_kujiale_floor200_20260902_v1`，代码 `fe061cb`）：card1F 2/18、card1B 3/18，
合计 5/36，对照 v2 的 2/36。5 条候选全部 1.47 m 机位净空。拒因里 `camera_too_close_to_target` 仍占约四分之一，
`anchor_outside_allocated_band` 升为 card1F 的第一拒因。结论：放宽占位只把产量从接近零提到很低，
Kujiale card1 的瓶颈是路线库太短太少（200 条移动路线、锚帧到查询帧中位只走 1.88 m），
下一轮的"求解器反向设计路线"才是结构性解法。2.0 m 不作为新占位值，仍以 v3 的 2.5 m 为准，等 owner 定。

随机 20 万个（相机点，移动路线）组合的几何可行率作旁证：两个时刻都 ≥ 2.5 m 且锚帧到查询帧方位扫过 > 30°，
Apartment 15.7%、Kujiale 5.0%。

## E. 测试

顶层 412 passed；unit 15 passed。新增测试文件：`test_qa_v3_camera_clearance_table.py`（解析房间：
重投影复现直接渲染、UE yaw 约定、与预检同一列中位指标、哨兵、读取器 fail-closed、预测可见性）、
`test_qa_v3_predicted_visibility.py`、`test_qa_v3_cell_budget.py`、`test_qa_v3_legacy_cli_gate.py`；
`test_scene_sampler.py` 新增净空筛查、高度回退、要求 fail-closed、表不匹配拒载入、静止路线预过滤。

## F. 失败与边界

1. 8 点 smoke v1 在验证阶段把 spawn 放在帧块外触发引擎断言，失败目录 `qa_v3_camera_clearance_table_apartment_smoke8_20260902_v1` 保留作证据；v2 暴露了平面/径向约定错误；v3 通过。
2. 我曾把 `tests/test_qa_v3_extended_pixel_join.py` 在本地未拉取的情况下追加并上传，覆盖了原有 15 个测试；
   已从上一个提交恢复并重新追加，`163fdcf` 是修正后的提交。
3. 预测可见性只在 fact 里记档，不拒题；`reject` 模式已实现但 profiles v2 未启用。
4. 全景的竖向覆盖 ±56°，相机 1 m 内的狗脚部视线可能落到覆盖外，返回"未知"不计入比例。
5. 所有阈值仍是占位；没有人类数据；没有渲染完整视频音频。
6. v1 参数在缺省 tier 政策下会因缺 `PIXEL_CAMERA_BLOCKAGE_MAX_DISTANCE_M`/`PIXEL_TIER_VISIBLE_FRACTION_EDGES`
   而 fail-closed：历史复现请用 v2/v3 参数或显式 `PIXEL_ACCEPTANCE_POLICY=both_frames_threshold_reject`。

## G. 还没做与需要 owner 决定的

0. Kujiale card1 产能：放宽 2.5 m 占位、补路线库，还是接受该房不出 card1；以及 Apartment card1 近半候选用 1.8 m 相机的处理。
1. 预测可见性目前只记档不拒题；对照数字（D2）够做配额与统计，是否允许 card7/card9 改 `reject` 由 owner 与 Codex 定；按预测档位做配额上限（现在只记分布）。
2. 全 75 帧像素捕获与"按时间线定档"规则，等人类数据。
3. 人类校准包 v4（要先渲 canary 候选的完整视频音频）。题型资产政策方案文档已写（`docs/roadmap/QA_V3_ASSET_POLICY_PROPOSAL_20260902.md`），只是方案，等 Codex 审与 owner 拍板。
4. Codex 第三次审核；合入 main 由 owner 决定。

## H. 远端

本节提交后整条线推送到 `origin/qa-v3-prescale-20260902`（与第三轮同一分支，`main` 不动）；尖端即本节的文档提交。合入 main 等 Codex 第三次审核后由 owner 决定。

# 第五轮（2026-09-03）：求解器当场设计路线、可走栅格、Kujiale card1 产能实验，以及公寓地面高度的 P0 发现

> 起因：owner 2026-09-03 拍板路线合成方案（`QA_V3_ROUTE_SYNTHESIS_PROPOSAL_20260903.md`），原话"合成轨迹也算合法，
> 只要严格跟普通的轨迹一样就行"。本节记录为此做的提交、三轮 canary、可走边距的受控实验、像素抽检，以及抽检时
> 发现的公寓地面高度问题。**本节同样不宣布 pilot 认证、放量或容差定稿。**

## A. 起止 HEAD

起点 `3362bc4`（第四轮文档与两份方案）。代码终点 `b6f0e88`；本节文档提交紧随其后。worktree 干净。

## B. 提交（按逻辑切片）

| 提交 | 内容 |
| --- | --- |
| `0d983a7` | 房间可走栅格：`walkable_grid.py`（UE (x,y) 厘米平面栅格 + 保守净空数组 + 读写与带边距抽点）与建表工具 `build_qa_v3_walkable_grid.py`（来源二选一：舞台自己的 RecastNavMesh 随机点，或现成的可行区栅格；场景自己的路线采样点与相机点所在格补为可走并校验全部落在格内） |
| `be80fb7` | 求解器当场设计路线：七个双角色求解器先用库，抽不到再按机位、方位带、距离范围设计路线；对照狗先从库挑再设计；每条设计路线走与库路线完全相同的检查（检查改成共享闭包）；路线带来源，fact/manifest 记来源、栅格身份、合成设置与计数；开了合成没栅格即停 |
| `8948637` | 修正首轮 canary 暴露的三处偏离方案：后一个关键帧抽方位与距离、抽速度后**反解**前一个关键帧的距离（原来两点距离独立抽，速度多半超范围）；前后腿同速可转向、只查实际占用帧（原来整段 75 帧强制直线连尾巴一起查）；库路线保留题型全部预算、合成另加预算（原来库只给 300 次，把公寓的库成功率也拖低了） |
| `90616ab` | 设计时就满足求解器的角度规则：`PointSpec` 带方位排除窗（离目标至少 MIN_AZIMUTH_SEP、离目标答案至少 Open 金标半径的两倍），两点设计带最小扫过角（后点只在前带还有余地的地方抽，前点在余下的带里抽）；前后腿在最多 7 个转角里找第一个不出可走区的 |
| `b6f0e88` | 对照狗的设计次数可以比目标多（`ROUTE_SYNTHESIS_OTHER_DESIGN_TRIES`，缺省目标的 4 倍）：一次尝试里建成目标是稀缺事件，8 次就放弃对照狗浪费了大半 |

每个提交时点顶层测试均通过：412 → 415 → 424 → 425 → 426 → 426 passed。

## C. 实现要点

### C1. 可走栅格（房间算一次）

- **Apartment**：启动舞台包一次，从 `RecastNavMesh-Default` 抽 200 万个随机点（100 次 × 2 万，引擎共 32 秒），
  10 cm 格，一格至少 2 个点才算可走；再把路线库 144,975 个采样点和 5757 个相机点所在格补为可走（补了 11 格，
  这些点本来就是导航系统给的，只是导航路径会切到随机点很少落到的多边形角上）。可走面积 64.5 m²。
- **Kujiale**：直接把路线库自己的可行区栅格（`avengine_room_feasible_region_v1`，5 cm 像素）按声明的坐标约定转过来，
  不启动引擎。可走面积 44.6 m²。
- 校验：两间房的库路线采样点与相机点 100% 落在可走格内。净空数组按距离变换算，再减 1.5 格保守；边界当障碍。
- 库路线自己离家具有多近：采样点净空中位数 Apartment 21 cm、Kujiale 29 cm，四分之一位数 0 / 7 cm。
  也就是说 0.3 m 的边距比库路线自己的标准严得多。各边距下的可走面积：

| 边距 | Apartment | Kujiale |
| --- | ---: | ---: |
| 0 | 64.5 m² | 44.6 m² |
| 0.1 m | 45.5 m² | 33.5 m² |
| 0.2 m | 37.3 m² | 27.4 m² |
| 0.3 m（占位） | 28.1 m² | 22.3 m² |

### C2. 合成器（`route_synthesis.py`）

先定机位：随机相机点 + 净空表里全圆的净空朝向（与其他抽样同一套筛查）。再设计路线：

- 两个关键帧（card1F/1B、card5R、对照狗的锚帧/查询帧）：先抽**后一帧**的方位（只在"前一带还留得下至少最小扫过角"的
  区间里抽）与距离，抽速度得到弦长，前一帧的方位在余下的带里抽，前一帧的**距离在它的方位射线上反解**（一元二次，
  0/1/2 个根，落在距离范围内才算）。两点之间匀速直线；前腿、后腿同速，各在 0、±30、±60、±90 度里按随机顺序找
  第一个不出可走区的转角。
- 一个关键帧（card2/3/7/9/15b、card4R、card6/6R/10 的目标）：抽点、抽速度，8 个朝向里找能走通的，前腿再找转角。
- 只查实际占用的帧（idle 平移后 base[0..74−idle]），整条 75 帧都在栅格里并保持边距。
- 路线对象与库路线同型（75 帧、`implied_speed` 按库的"端到端跨度 / 5 秒"口径），`route_id` 是设计参数的哈希，
  `provenance` 记机位、每个关键帧的方位/距离/是否反解、速度、朝向、两个转角、最小净空、查过的帧数、栅格哈希。

### C3. 求解器接入（`scene_sampler.py`）

- 库先：题型声明的 `max_attempts` 全部留给库（无合成时行为与以前逐位相同）；用完后另加 `ROUTE_SYNTHESIS_ATTEMPTS`
  次合成尝试。每个求解器的"每候选检查"改成闭包，库候选与设计候选走同一个闭包；对照狗也是先扫库 64 条再设计。
- 台账新增 `route_synthesis`：库尝试数、合成尝试数、目标/对照设计数与建成数、设计失败原因；`absorb()` 合并逐题型台账。
- fact 的 `motion` 新增 `source1/2_route_provenance` 与 `route_sources`；manifest 的 `scene` 新增 `walkable_grid`
  与 `route_synthesis`（设置、顺序、每个角色来自库/设计的候选数）。
- `ROUTE_SYNTHESIS_ENABLED` 开着而场景没有 `walkable_grid`，或没有任何格能保持边距，在搜索前就拒绝启动。
- N 角色扩展题型（card11–17）仍只抽库路线，manifest 如实写 `applied: false`。

### C4. 参数占位（`qa_v3_prescale_params_synthesis_20260903_v2.json`，全部占位）

速度 0.6–1.5 m/s（Kujiale 库路线 0.60–1.06，Apartment 库四分位 0.83–1.78）；边距 0.3 m；最大相机距离 900 cm；
合成尝试 3000；每次尝试目标设计 8 次、对照 32 次；最大转角 90°。场景配置 `qa_v3_scene_configs_synthesis_20260903_v1/`
在第四轮配置上加了 `walkable_grid`。

## D. fresh 产物与实测数字（全部 research_candidate）

### D1. 栅格

`qa_v3_walkable_grid_apartment_20260903_v2/`、`qa_v3_walkable_grid_kujiale_20260903_v2/`（v1 是补场景点之前的版本，
留作证据）。

### D2. 三轮两房 canary（core 9 题型各 6 格，card1 各 18 格）

| 轮次 | 代码 | Apartment card1 smoke | Kujiale card1 smoke | 说明 |
| --- | --- | ---: | ---: | --- |
| 第四轮 v2（对照） | `4b9ef3b` | 23/36 | 2/36 | 只有库 |
| 本轮 v1 | `be80fb7` | 16/36（设计 0） | 1/36（设计 1） | 38 万次设计只建成 420 条；库预算被压到 300 次 |
| 本轮 v2 | `8948637` | 22/36（设计 0） | 0/36（设计 0） | 建路率 0.2–0.3%，建成的目标过不了后面的角度规则 |
| 本轮 v3 | `90616ab` | 27/36（设计目标 3、对照 1） | 3/36（设计目标 3） | 建路率 0.6%；Kujiale 91% 的合成尝试死在可走边距 |

core v3：Apartment card1B 6/6、card1F 3/6，其余 7 个题型 6/6；Kujiale card1B 1/6（设计目标）、card1F 1/6、card5R 5/6，
其余 6/6，其中对照狗来自设计的有 card6R 2、card8 2、card9 1、card5R 1。产物 `qa_v3_canary_{core,card1}_{apartment,kujiale}_20260903_v{1,2,3}`。

### D3. 可走边距受控实验（`b6f0e88`，种子 -v3，只改边距）

| 边距 | Apartment card1 smoke（设计目标数） | Kujiale card1 smoke（设计目标数） | 设计建成率 |
| --- | ---: | ---: | --- |
| 0.3 m | 28/36（5） | 7/36（6） | Kujiale 0.6% |
| 0.2 m | 28/36（5） | 10/36（10） | 1.0% |
| 0.1 m | 28/36（2） | 16/36（14） | 1.4% |

Apartment 不受边距影响，也没有退步（库路线本来就够）。Kujiale 的产能几乎全靠设计路线，且随边距放宽线性上升，
0.1 m 时 16/36。产物 `qa_v3_canary_card1_{room}_20260903_margin0{30,20,10}_v1`，参数 `qa_v3_prescale_params_synthesis_20260903_margin0xx_v1.json`。

方案里的目标 24/36 没有达到。还有一层天花板与路线无关：card1 的 Open 金标要求对照狗离目标答案方位 60° 以上，
而视场只有 ±52.5°，答案带在中间那三分之一格几乎无解（库路线也一样），所以 card1 的可达上限本来就在 24/36 附近。

### D4. 像素抽检（设计路线有没有穿家具）

- v1（3 条，边距 0.3）：Kujiale card1F_016 与 Apartment card1F_006 通过；Apartment card1B_009 被 join 判拒，原因是我
  只捕了每 5 帧而 22/62 不是 5 的倍数（工具问题，v3 已把锚帧与查询帧加进捕获帧）。
- v3（9 条，边距 0.3，`qa_v3_synthesis_card1_pixel_20260903_v3/`）：8 条 pass，1 条 `main_referent_query_frame_camera_side_blockage`
  （相机侧近处遮挡，分档政策照常抓到，与路线来源无关）。设计路线的逐帧可见比例与预测档位一致，设计路线的最小净空 31.6–56 cm；
  抽看的画面里 Kujiale 两只狗都在地面上正常行走，没有穿家具。
- 边距 0.2 m（19 条，`qa_v3_synthesis_card1_pixel_20260903_margin020_v1/`）：15 pass（Apartment 7/9、Kujiale 8/10），4 条拒绝全是
  锚帧的相机侧近处遮挡（机位净空筛查的地盘，与路线来源无关）。26 条设计路线的最小净空 20.8–65 cm，中位 30.6 cm；
  Kujiale 的设计路线狗在多数帧可见比例 1.00，抽看的画面里正常行走、不穿家具。边距 0.1 的候选没有抽检。

### D5. 公寓地面高度（P0，与本轮改动无关，抽检时发现）

看抽检画面时发现 Apartment 的狗全是"卧着"的姿态，Kujiale 的正常。查证如下：

1. 时间线与引擎回读都说狗在 `walk`、动画位置正确，不是动作错。
2. 用像素真值的原生深度反推相机离地高度：Apartment 三条候选（含第四轮的库路线候选）画面底部各行算出的相机高度全是
   **1.201 m**，而相机 z 是 147.1 cm；Kujiale 同法算出 1.472 m，与 147 cm 一致。即 Apartment 的地面在 **z ≈ +27 cm**，
   不是场景配置写的 `ground_z_ue_cm: 0.0`。
3. 路线库的导航点 z 在 28–32 cm（导航网格贴在地面上方几厘米），与之相符。
4. 像素真值旁证：Apartment 今天所有捕获帧里狗的可见比例 95 分位只有 0.854、≥0.95 的只占 2.3%；Kujiale 分别是 1.0 和 47.5%。
   狗下半身埋在地板下，就是"不被遮挡也永远不满"。

后果：迄今所有 Apartment 渲染的狗都陷进地板 27 cm；相机实际离地 1.20 m 而不是 1.47 m；Apartment 的像素真值、分档、
机位堵死对照、预测可见性对照（Pearson 0.955）都是在这个偏移下算的；机位净空表按绝对 z 渲，与生产相机一致，但
"目标带"的高度语义偏了 27 cm。**我没有改场景配置**：这是房间的权威事实与重渲成本的决定，见 G。

## E. 测试

顶层 426 passed（新增 `test_qa_v3_walkable_grid.py` 3 个、`test_qa_v3_route_synthesis.py` 11 个：栅格读写与保守净空、
栅格化与场景点补格、可行区转换、设置 fail-closed、反解与折线几何、排除窗与联合扫过角、设计路线过点与匀速、
七个求解器在短库/静库下靠设计填格并复核全部约束、库够用时不合成、同种子可复现、台账合并、场景载入）。
`tests/unit`：3198 passed，29 failed + 21 errors 全部在 `test_strict_two_human_*`（要 worktree 里 `tmp/` 的留存工作区，
本副本不完整）和 `test_tool_index_current`（`tools/audit` 目录不在索引清单里）；两者在 `3362bc4` 快照上同样失败或跳过，与本轮无关。

## F. 失败与边界

1. 首轮实现偏离方案三处（见 `8948637`），第一轮 canary 因此比第四轮还差；已修正并留产物。
2. v1 像素抽检的一条 join 拒绝是捕获帧选择造成的工具问题，不是候选问题。
3. Kujiale card1 到 0.1 m 边距也只有 16/36；中间答案带的 Open 金标规则本身把上限压在 24/36 左右。
4. 扩展题型（N 角色）没有接合成。
5. 设计路线是折线，直腿比录制路线"假"；像素抽检只做了边距 0.3 的候选。
6. 所有合成参数是占位；Apartment 的全部数字都带着 D5 的地面偏移。

## G. 需要 owner 决定

1. **可走边距占位**：0.3 / 0.2 / 0.1 m 对应 Kujiale card1 7 / 10 / 16 of 36；库路线自己的净空中位 21–29 cm。我倾向 0.2 m
   （约等于狗的半身宽）：0.2 m 的 19 条像素抽检 15 pass，4 条拒绝都是机位问题，设计路线本身没有穿家具的迹象。0.1 m 没有抽检。
2. **Apartment 地面高度**：`ground_z_ue_cm` 应改到实测地面（约 27 cm，建议先在引擎里读地板实际高度定值）；改后相机绝对
   z 变为 174 cm，机位净空表要按新高度重渲（75 分钟）或改按"离地高度"重解释；Apartment 的 canary、像素真值、预测可见性
   对照都要重跑；人类校准包在此之前不能用 Apartment 的画面。
3. **Kujiale card1 产能**：接受 16/36 上限、放宽 2.5 m 占位、降低速度下限，还是接受该房 card1 少出题。
4. Codex 第三次只读审核范围现在是 `23eef13..`本节文档提交；合入 main 由 owner 决定。

## H. 远端

本节提交后整条线推送到 `origin/qa-v3-prescale-20260902`（`main` 不动）。

---

# 第六轮（2026-09-03）：地板偏移成为实测事实、公寓按实测地面重渲、边距 0.2 m 定占位、Kujiale card1 上限接受

> 本节记录 owner 对第五轮 G 节三项决定的执行：P0 成立并升级为常设规矩（"以后所有房间都先确定地板偏移再做别的"）、
> 可走边距占位 0.2 m、公寓 `ground_z_ue_cm` 改实测值并重渲净空表、重跑 canary/像素/预测对照、人类校准包在此之前不得用公寓画面、
> Kujiale card1 产能上限接受。**本节同样不宣布 pilot 认证、放量或容差定稿。**

## A. 起止 HEAD

起点 `12eb75d`（第五轮文档提交），代码终点 `fe2119a`，文档提交见 H。工作副本仍是 `/data/jzy/tmp/wt-qa-v3-pilot`。

## B. 提交（按逻辑切片）

| 提交 | 内容 |
| --- | --- |
| `2f030f9` | `floor_reference.py`（房间地板参照产物：索引 + 带 sha256 的逐点行）、`measure_qa_v3_floor_z.py`（引擎里两法量地板）、`camera_clearance.CameraClearanceTable.ground_z_ue_cm`、`scene_sampler.load_scene` 三条 fail-closed 规则、测试 |
| `1250947` | `resolve_scene_render_context` 要求实测地板；fact `room.floor_reference` 与批次/扩展 manifest `scene.floor_reference` 记身份 |
| `fe2119a` | 人类校准包构建器拒绝没有实测地板参照的 fact |

## C. 实现要点

### C1. 量地板（`tools/qa/measure_qa_v3_floor_z.py`，房间算一次）

不放任何角色，在房间自己的打包地图里，对求解器会抽到的每个相机点（路线库导航点，与净空表同一集合）和可走栅格的随机格，
用两种彼此独立的方法量地板高度：

1. **线追踪**：从 `ground_z_ue_cm + 150 cm` 垂直向下打 450 cm（`BlockAll` + 复杂碰撞，与双人贴地诊断工具同一调用），第一个命中点就是脚下地面；
2. **向下深度**：深度相机俯视（pitch −90°，30° 视场，64×64），中心 5×5 像素的深度中位数就是相机到地面的距离。

地板高度取主量法命中的**中位数**（线追踪有命中就用线追踪，否则用深度）；两法都命中时中位数必须相差 ≤ 2 cm。
p05/p95、中位数 ±2 cm 内的命中占比、命中的组件路径、比中位数高/低 5 cm 以上的离群点都存下来：分层地面或打到家具的追踪会**显出来**，
不会被平均掉。阈值（≥ 200 次命中、命中率 ≥ 98%、±2 cm 内 ≥ 95%）全是占位；不满足就写 `inconsistent`，这样的参照不能喂给渲染事实。

### C2. 三条 fail-closed 规则（`scene_sampler.load_scene`）

1. 配置里有 `render` 段就必须有同房间的 `floor_reference`，否则拒绝载入，错误直接点名 `render.ground_z_ue_cm` 不能是手写常数；
2. `render.ground_z_ue_cm` 与实测中位数差必须 ≤ 0.5 cm；
3. 机位净空表按绝对 z 渲，其 `camera_contract.ground_z_ue_cm` 与实测地面不符即拒（旧公寓表就这样被拒，见 D2）。

`resolve_scene_render_context` 再查一次 provenance 里的参照状态；fact 的 `room.floor_reference` 与 manifest 的 `scene.floor_reference`
记参照身份（路径、量法、实测值、p05/p95、行文件 sha256、代码 revision）。人类包构建器只接受 `room.floor_reference.status == measured`
且与 `room.ground_z_ue_cm` 一致的 fact：run02 公寓 18 题的包 v3 因此作废，不再投放。

### C3. 边距占位 0.2 m（`qa_v3_prescale_params_floor_20260903_v3.json`）

只改 `ROUTE_SYNTHESIS_WALKABLE_MARGIN_M` 0.3 → 0.2，并写入 owner 三项裁定的 note；仍是 placeholder。

## D. fresh 产物与实测数字（全部 research_candidate）

### D1. 两房地板实测

| 房间 | 线追踪 | 向下深度 | 结论 |
| --- | --- | --- | --- |
| Apartment（v2） | 6757/6757 命中，中位 **27.11 cm**，p95 28.48，最大 30.67，±2 cm 内 98.9%；命中面 `SM_Floor_21` 5695 次，另两块地面件 875/187 次（高 1–3 cm） | 900/900，中位 27.15 cm | 两法差 0.04 cm；`ground_z_ue_cm` 定 **27.11**，比旧配置高 27.11 cm |
| Kujiale（v1/v2） | **0/1598 命中**：烘焙灯光版地板网格没有碰撞体 | 1.5 m 相机：998 点中位 0.0，但 ±2 cm 内只有 78%，p95 66.9 cm | 可行区含桌下格，高相机看到的是桌面；v1/v2 留作证据 |
| Kujiale（v3，用这个） | 未做 | 相机离地 0.3 m：998/998，中位 **0.02 cm**，p05 = p95 = 0.02，±2 cm 内 95.4%，离群 23 高（12–19 cm）/23 低（−5 cm） | 地面就在 z = 0，配置不变，只加 `floor_reference` |

线追踪的 Apartment 结果与第五轮深度反推（相机离地 1.201 m → 地面 ≈ 27 cm）一致，与导航点 z 28–32 cm 一致。
两房的旧配置（`qa_v3_scene_configs_synthesis_20260903_v1`）在新规则下都不能再载入；新配置在 `qa_v3_scene_configs_floor_20260903_v{1,2}`。

### D2–D5. 公寓净空表重渲、两房 canary v4、像素真值、预测对照（**提交本节时仍在后台跑**）

owner 要求先把该跑的放后台、再收尾，所以这四项由一条驱动脚本串起来无人值守地跑，本节只记路径与看法，数字由下一轮补录：

| 步骤 | 产物 | 说明 |
| --- | --- | --- |
| 公寓净空表按实测地面重渲（相机绝对 z 174.2 / 207.1 cm） | `/data/jzy/tmp/qa_v3_camera_clearance_table_apartment_20260903_v2`（日志同名 `.log`，成功标志 `QA_V3_CLEARANCE_TABLE_OK`） | 输入配置 `qa_v3_scene_configs_floor_20260903_v1/apartment_0000.json`（含地板参照、不含旧表）；约 80 分钟 |
| 场景配置 v2 | `/data/jzy/tmp/qa_v3_scene_configs_floor_20260903_v2/` | 驱动脚本在表成功后生成：公寓加新表，Kujiale 同 v1；写完先用 `load_scene` 自检 |
| 两房 canary v4（core 9 题型各 6 格 + card1 各 18 格，参数 v3 边距 0.2 m） | `/data/jzy/tmp/qa_v3_canary_{core,card1}_{apartment,kujiale}_20260903_v4` | 看法：`python qa_v3_prescale_round5_scripts_20260903_v1/inspect_canary_v2.py <目录…>`；对照第五轮 v3（Apartment card1 28/36，Kujiale 0.2 m 10/36） |
| 像素真值（card1 候选，设计路线优先；Apartment ≤ 24 条、Kujiale ≤ 12 条） | `/data/jzy/tmp/qa_v3_floor_card1_pixel_20260903_v1/` | 看法：`python qa_v3_prescale_round6_scripts_20260903_v1/inspect_pixel_floor_v1.py <根目录>`。**判定地面修好的证据**：公寓帧 0 深度反推的相机离地高度应从 1.201 m 回到 ≈ 1.471 m；狗可见比例 95 分位应从 0.854 接近 Kujiale 的 1.0 |
| 预测可见性对照 | `/data/jzy/tmp/qa_v3_visibility_prediction_validation_{apartment,kujiale}_20260903_v2.json` | 公寓用新表 + 新像素真值重算（第四轮 Pearson 0.955 是在偏移地面上算的） |

驱动脚本 `/data/jzy/tmp/qa_v3_prescale_round6_scripts_20260903_v1/driver_floor_v1.sh`（日志 `driver_floor_v1.log`，结束标志 `DRIVER DONE`；任一步失败即停并留证据）。
它在 canary 一步会拒绝脏工作副本，所以本节提交后工作副本必须保持干净。

## E. 测试

顶层 `tests/`（不含 `tests/unit`）439 passed（第五轮 426 + 新增 `test_qa_v3_floor_reference.py` 5 个、`test_scene_sampler.py` 1 个、
`test_qa_v3_human_calibration.py` 1 个；`test_gatea_generation.py` 与 `test_qa_v3_predicted_visibility.py` 的夹具补了实测地板参照）。
`tests/unit` 本轮未重跑（第五轮已核实其失败项与本线无关）。

**这个 439 是本轮代码终点 `6771ef2` 上的数**。本节文档提交实际推送到的 head 是 `ab0e8fb`，它还含另一会话的两个音频提交（`b9b97f3`、`8a65ecf`），带进 7 个新测试，所以同一条命令在推送的 head 上是 **446 passed / 0 failed**（我与审核会话各自实测一致）。复现本节数字请用 `6771ef2`，复现测试计数请用 `ab0e8fb`。

## F. 失败与边界

1. Kujiale 烘焙灯光版地板没有碰撞体，线追踪全部落空（v1/v2 留证据）；1.5 m 高的深度量法有 22% 的点看到桌面，说明该房可行区含桌下格——
   这也是 Kujiale 机位常被堵的一个来源。低相机（0.3 m）深度法才量到地面，工具要求操作者按房间选起始高度，这是已知的手工环节。
2. 地板参照的阈值（200 次命中、98%、±2 cm 内 95%、两法差 ≤ 2 cm）全是占位。
3. 第二至第五轮 Apartment 的全部像素数字、分档、预测对照都带着 27 cm 偏移，本轮不改写历史产物，只在新配置下重跑。
4. run02 公寓人类校准包 v3 作废；新包要等本轮像素与 canary 出来后另起。
5. D2–D5 的数字本节未录（后台在跑）。

## G. owner 裁定（本轮已执行）与还需要决定的

已执行：P0 成立并成为常设规矩（新房间先量地板）；边距占位 0.2 m；公寓地面按实测修、净空表重渲、canary/像素/预测对照重跑、人类包排除旧公寓画面；Kujiale card1 产能上限接受。

还需要：1）Codex 第三次只读审核，范围现在是 `23eef13..`本节文档提交；2）1.8 m 相机在 Apartment card1 的占比（第四轮遗留）；
3）card7/card9 `tier`→`reject`（第四轮遗留）；4）人/静态声源方案的准入开关归属、VCTK 轮换粒度、第四种上衣色、音响落地与不进运动状态题（第五轮遗留）；
5）D2–D5 跑完后按数字决定是否可以起新的人类校准包。

## H. 远端

本节提交后整条线推送到 `origin/qa-v3-prescale-20260902`（`main` 不动）。

---

# 第六轮补录（2026-09-03 06:10 UTC）：D2–D5 的实际数字

上面第六轮 D2–D5 写的是后台在跑、数字下轮补。作业已经跑完，本节就是那些数字，取代上面那张"步骤/产物"表里的占位。

## 为什么这轮的跑在快照里

第一次驱动在 04:19 UTC 走到 canary 那步时被 `worktree dirty; refusing` 拦下——正是另一个会话在改文件。
重跑改成：`git worktree add --detach /data/jzy/tmp/wt-qa-v3-floor-run-20260903 6771ef2`，
一个钉死在我这轮提交上的只读快照，脚本 `driver_floor_v3.sh` / `*_snapshot.sh` 都指向它。
理由是可复现与可归因：共享副本的 HEAD 每 40 分钟就动一次，而且它的音频改动会混进本轮的产能数字里。
这个快照只是跑作业用，不在里面改代码。

## D2. 公寓净空表按实测地面重渲

`/data/jzy/tmp/qa_v3_camera_clearance_table_apartment_20260903_v2`：5757 点、两个高度（相机绝对 z 174.2 / 207.1 cm）、
每点 0.306 s、摘要 1000.8 s，`QA_V3_CLEARANCE_TABLE_OK`。场景配置 v2 载入自检通过：公寓地面 27.11、表地面 27.11；
Kujiale 地面 0.0、表地面 0.0。旧公寓表（按地面 0 渲的 `..._20260902_v1`）现在被载入规则直接拒绝。

## D3. 两房 canary v4（快照 HEAD 6771ef2，参数 v3 边距 0.2 m，种子 -v4）

card1 18 格 smoke：

| 房间 | card1F | card1B | 合计 | 第五轮同边距对照 |
| --- | ---: | ---: | ---: | ---: |
| Apartment | 14/18 | 14/18 | **28/36** | 28/36（边距实验 0.2 m） |
| Kujiale | 8/18 | 5/18 | **13/36** | 10/36（边距实验 0.2 m，不同种子） |

core 9 题型各 6 格：Apartment card1F 4/6、card1B 4/6，其余七个题型 6/6；Kujiale card1F 3/6、card1B 3/6、card5R 5/6，
其余六个 6/6。对照第四轮（Apartment card1F 2/6、card1B 3/6；Kujiale card1 0/6），两房的 card1 都比只有库路线时高。

其他读数：Apartment 28 条 fact 里 1.8 m 相机只用了 4 条（第四轮是 11/23），说明按实测地面渲的净空表让 1.47 m 机位更常可用；
Kujiale 13 条全是 1.47 m。设计路线的最小净空：Apartment 8 条 21.1–25.6 cm（中位 25.6），Kujiale 18 条 22.5 cm 起（中位 30.6），
与 0.2 m 边距一致。速度中位 1.17–1.25 m/s，都在 0.6–1.5 占位区间内。

主要拒因仍是 `synthesis_route_outside_walkable`（设计出的路线穿家具，被栅格挡下）与 `camera_too_close_to_target`、
`anchor_outside_allocated_band`；Kujiale 的 `other:synthesis_infeasible_spec` 占比最大，是对照狗的方位排除窗在这间小客厅里常常无解。

## D4. 像素真值：地面修好了

`/data/jzy/tmp/qa_v3_floor_card1_pixel_20260903_v1`，36 条 card1 候选（Apartment 24、Kujiale 12），06:09 UTC 跑完。

| 读数 | 旧地面（第五轮同工具） | 新地面（本轮） | Kujiale 对照 |
| --- | ---: | ---: | ---: |
| 公寓帧 0 深度反推的相机离地高度 | 中位 1.199 m | 中位 1.319 m，**最大 1.470 m** | 1.469 m |
| 公寓在画面内的实例帧可见比例 95 分位 | 0.860 | **1.000** | 1.000 |
| 公寓可见比例 ≥ 0.95 的帧占比 | 3.3% | **35.8%** | 47.5% |

相机高度那一栏要这么读：我的估计取画面底部一成行的中央列，看到地板时给出真实高度，看到沙发桌面时偏小。
旧地面下**中位数**就是 1.199 m（相机 z 147.1、地板 27.11，差 119.9 cm，对得上）；新地面下**最大值**是 1.470 m，
正是 147.1 cm 的设计高度，中位 1.319 是被家具占住底部的候选拉下来的，Kujiale 一直如此（最小 0.801）。
可见比例的两个数是更直接的证据：狗的下半身不再埋在地板里，95 分位从 0.860 升到 1.000，满可见的帧占比翻了十倍。

join 判定：Apartment 24 条 22 pass、2 拒；Kujiale 12 条 10 pass、2 拒；四条拒绝全部是锚定帧的相机侧近处遮挡，
和第五轮同一类，不是路线问题。设计路线的最小净空 21.1–52.5 cm，与 0.2 m 边距一致。

## D5. 预测可见性对照（在新地面上重算）

| 房间 | 候选 | 行数 | Pearson | Spearman | 绝对误差中位 | 五档一致 | 0.5 分界一致 | 出画一致 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apartment | 24 | 832 | 0.959 | 0.926 | 0.030 | 0.774 | 0.927 | 0.972 |
| Kujiale | 12 | 412 | 0.934 | 0.904 | 0.025 | 0.789 | 0.911 | 0.949 |

第四轮在错地面上的公寓 Pearson 是 0.955，改正后 0.959，结论不变：够做档位配额，不够单独拒题，card7/card9 维持 tier 声明。
公寓"真隐藏被预测成可见"5 行、"真可见被预测成隐藏"67 行，仍是偏保守的方向。

---

## 口径变更声明（2026-09-03 晚；决策记录由另一会话写在 `/data/jzy/tmp/QA_V3_QUESTION_DESIGN_DECISIONS_20260903.md`）

owner 试做人类校准包时发现题目人做不了，随后定了几条出题方式的改动。**本报告第二轮到第六轮的全部数字都是在这些改动之前跑的**，所以下面几条口径已经变了，历史数字要按旧口径读：

1. **方位角符号要反过来**。owner 定了 AVEngine 改用 DCASE FOA 约定（+x 前、+y 左、方位**左为正**、范围 [−180, 180]），与同方向的另一项目 Spatial-Omni 对齐；此前两条线用互为镜像的符号，跨项目比数字会静默出错。本报告里所有方位带、锚角、角距、MCQ 选项的符号都还是旧的"右为正"。改的时候注意带边界反转后 `truth.band_index` 要重映射，不是单纯取负。
2. **查询时刻改成 0.5 秒窗口，真值改成区间**，不再用单帧编号（模型输入形式不变，仍是多通道 video 加文字，所以时刻只能靠题面文字说清）。本报告 card1、card2 的单帧方位真值属于旧口径。另一会话实测：0.5 秒窗口内目标方位扫过中位 3.0 度、最大 9.6 度，而答案带宽是 23.33 度。
3. **card8 的时间题改成三个整秒桶**，T_FULL 由粒度导出。
4. **人类校准包 `qa_v3_human_calibration_pack_run02_20260902_v3` 不能按原样投放**。另一会话实测 18 段成片峰值只有 −31.7 到 −38.2 dBFS，几乎听不见，这是 owner 答不出题的主因；根因是 program 的 `linear_gain: 0.18`（−14.9 dB）叠上空间化衰减约 16 dB，而双耳渲染契约明写禁止事后归一，绝对电平只能在 `linear_gain` 上定。临时听音副本（统一 +28 dB、ILD 逐条不变、原包未动）在 `/data/jzy/tmp/qa_v3_human_calibration_pack_run02_20260902_v3_boosted28db`。该包另有本轮 C2 的原因（缺实测地板参照）已被构建器拒收。

这些改动会改写已生成的题面与真值，**重出范围待 owner 定**。本节只声明口径已变，不改写上面任何历史数字。

## 本节提交的落点（2026-09-03 晚）

Grok 会话的音频配置提交 `7fb5ba9` 把顶层测试打红 15 个（`tests/test_generate_questions.py` 13 条、`tests/test_qa_v3_prescale_candidate_audit.py` 2 条，根因是 `audio_profiles.py` 新要求 `T_FULL_status` 与 `SAMPLE_RATE_HZ`，其中前者让候选审计工具审不了已产出的老数据），我独立复跑确认为 440 passed / 15 failed。本文档提交原先叠在它上面，一推就会把红提交带上去，所以按决策记录的解法：先把 `7fb5ba9` 留成分支 `qa-v3-audio-config-rework-20260903`（不丢东西），再把本文档提交变基到已推的 `8a65ecf` 上单独推送。

---

# owner 2026-09-03 晚：card7/card9 维持分档，以及"不是百分百被挡住就留下"这条边界

## 决定

1. **card7、card9 的可见性声明维持 `mode: "tier"`**，不改成设计阶段拒题。依据是本轮 D5 的对照：预测器与像素真值在 0.5 分界上一致率 0.927，但两类错误不对称——真隐藏被预测成可见 5/768，真可见被预测成隐藏 67/768（8.7%）。改 reject 会白扔掉 8.7% 合格候选，只换来不到 1% 的不可答题目被拦。像素真值才是权威，预测器是代理。
2. **留下题目的边界：被问的声源只要还剩一点可见像素就留下并记难度，完全看不见才拒。** 这条比之前的政策**更严**，不是更宽：2026-09-02 的分档政策连"百分百被挡住"也只记档不拒题。

## 实现（提交 `9b72eab`）

拒题的档位变成**声明的数据**而不是代码里的判断：`PIXEL_TIER_REJECT_TIERS`，在分档政策下与另外两个分档参数一样是必填（缺键即拒），取值必须是已知档位的子集，空列表可复现 2026-09-02 的行为。生产取值是 `["hidden", "out_of_view"]`（params `qa_v3_prescale_params_floor_20260903_v6.json`）。

`0.5` 与 `1000` 像素两个占位阈值**保留**，只当分档边界和 `referent_frames_below_placeholder_thresholds` 计数，不再拒题，这样以后想按"重度遮挡"过滤不需要重新生成任何东西。

**顺带修掉一个静默洞**：fact 里嵌的验收政策优先于 params，而旧 fact 没有这个新键，所以拿新 params 去判旧 fact 会在操作者以为新规则生效的情况下按旧的空列表判过。现在这种不一致会直接报错，错误信息给两条出路（用新 params 重新生成，或显式用空列表复现原判）。这个洞是我在量这条规则的影响时撞出来的，不是推理出来的。

## 这条规则在本轮 36 条像素真值上的实际影响

直接从像素真值按分档定义算出来（不经过 join，因为旧 fact 不允许被新规则重判）：

| 房间 | 现在通过 | 新规则会拒 | 新规则下通过 |
| --- | ---: | ---: | ---: |
| Apartment | 22 / 24 | **8** | 14 |
| Kujiale | 10 / 12 | 0 | 10 |

被拒的 8 条一共有 10 个声明帧完全看不见，分布是 Gate A 锚定帧 3、Gate A 查询帧 3、主指代查询帧 2、主指代锚定帧 2。Kujiale 一条都不受影响。

**一个待定的细化**：现在是"任一声明帧完全被挡就拒整条候选"。但 10 个被挡帧里有 6 个在 Gate A 那一侧，而 Gate A 是与主题配对的另一道题。如果改成"只拒受影响的那一侧"，Apartment 大约能多留 4 到 5 条。这是 owner 的取舍，我按字面实现了整条拒，没有自己拆细。

## 撤回（同日，owner 看到代价后的裁定）

owner 看到上表后裁定**不采用这条边界**：完全看不见的指代帧仍然只记难度、不拒题，也就是维持 2026-09-02 的行为。

所以生产行为回到：**只有相机侧近处遮挡拒题**。`PIXEL_TIER_REJECT_TIERS` 这个键保留，但**默认空列表**、不再必填，因为默认值就是 owner 重申的行为；显式写非空列表才会拒那些档位。保留这个键的理由不是为了这条被否掉的规则，而是①验收边界属于该被记录的东西，它会进 fact 与 join 输出；②那个静默洞的修复要靠它——在一个列表下设计的候选，不可能被另一个列表静默重判。

`ac4e459` 那一节里的 8 条、14 条只作为"如果采用会付什么代价"的实测记录，不是现行判定。

