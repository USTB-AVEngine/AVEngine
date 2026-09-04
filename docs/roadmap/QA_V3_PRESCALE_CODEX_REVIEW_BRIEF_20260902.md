# QA v3 放量前工作线：给 Codex 的只读审核简报（2026-09-02）

> 审什么：远端分支 `origin/qa-v3-prescale-20260902`，审核范围是 `c52ca65..HEAD`。
> `c52ca65` 是上一次 Codex 任务书的起点，之后的提交全部由 Claude 实现。
> 只读审核：不修改、不提交、不启动作业。审核结论请区分"可开百题认证 pilot"
> 与"可大规模生成正式数据"两个判定；本工作线自己不宣布任何一项。

## 1. owner 已定的口径（审核时以此为准，与早先任务书冲突处以 owner 为准）

1. 狗走到家具后面、狗掉出画面下沿、以及 Gate A 对照狗整段视频从未露面，都**留作难题分档**，不拒题；只有挡住目标的物体离相机不到 1.5 米（机位缺陷）才硬拒。像素验收的缺省政策因此改为 `camera_blockage_reject_then_tier`（提交 `658c813`），旧的"两帧阈值硬拒"须显式点名才生效。
2. 全部占位参数暂按 `/data/jzy/tmp/qa_v3_prescale_params_tfull_placeholder_20260902_v2.json` 沿用：T_FULL 0.5、像素阈值 0.5 与 1000 像素、分档边界 0.5 与 0.2、机位堵死距离 1.5 米、预检上限 0.2、锚帧最小相机距离 250 厘米。终值等人类校准。
3. 机位堵死要在生成期由求解器自己避开（下一轮：每房机位净空表进求解器），不是事后过滤。
4. 声源俯仰角、相机俯仰角、新题型本轮不做。

## 2. 提交（自 `c52ca65` 起）

| 提交 | 内容 |
| --- | --- |
| `feccf73` | card8 显式 T_FULL，最小首叫间隔严格大于 max(T_HALF, 2·T_FULL)，样本域整数比较 |
| `eeb10b3` | card1 最终 timeline 为验收权威，realized 五条门 fail-closed，fact 分记 planned/realized |
| `c828823` | pixel join 支持 card1F/card1B 双侧，阈值占位三处一致 |
| `d2f3919` | card1 条件表与单模态 best-response 只读报表工具 |
| `6fd5925` | 人类试听页禁右键 |
| `8713a78` | 基线注明样本内上界 |
| `2b40afd` | 第二轮报告 |
| `a482ad3` | 锚帧最小相机距离底线（显式参数、缺省不启用）；realized 检查记取景几何，不作门 |
| `15f0b1d` | 像素分档政策、挡住物深度、全片可见时间线 |
| `61c9b1c` | 无角色相机深度预检工具（目标带判据） |
| `494eae4`、`577210e` | 第三轮报告与错字 |
| `658c813` | 分档政策改为缺省（owner 决定） |
| `90b7246`、`3362d20` | 报告记录决定与远端分支 |
| 本提交 | 本简报与"Gate A 狗未露面留作难题"的记录 |

## 3. 请重点核实

1. `audio_profiles.card8_scoring_params` 是否是 card8 首叫链唯一入口；调度器、自检、带边推导、`build_answer`、`audit_gatea_pair` 是否都走它；T_FULL=0.6 下 1.1 s 拒、1.2 s 边界拒、多一个样本过。
2. `realized_cross_time_checks` 五条门是否恰好对应任务书第二项，是否只作用于 card1F/card1B；planned 值是否只作 planning value（fact、216 重验、条件基线三处口径一致）。
3. pixel join 的 card1 分支：无房间专用逻辑；阈值三处一致；bbox 贴边判定与 run02 的 `pixel_eligible` 一致；分档政策的硬拒条件"中/重/完全被挡且挡住物 ≤ 1.5 m"是否会误拒合法近景遮挡；缺省政策改动后旧参数文件对 card1 会 fail-closed，是否可接受。
4. 深度预检：目标带定义（相机高 1.47 m、目标高 0.5 m、2.5–10 m）是否合理；目标带 @1.5 m ≤ 0.2 对像素真值判定的机位堵死 16/16、误报 0/21、对人工标注 12/13（漏门扇 2.1 m）是否足以替代二维几何预筛（二维路线栅格对照失败的记录见 `qa_v3_prescale_round2_camera_2d_proxy_calibration_20260902_v1.txt`）。
5. 条件基线工具：MCQ 按实际非空格算；Open 用实际两档评分器网格搜索；读不到模型结果；单房间占比单列；样本内上界注明。
6. 锚帧距离底线只在 forward/backward 求解器生效是否合理；`frame_geometry` 只记录不作门。
7. 未同步的旧路径 `filter_cross_time_points.py`、`design_qa_v3_pilot_batch.py` 应退役还是补齐；扩展题型 manifest 不记代码版本。
8. 下一轮任务书（机位净空表进求解器、静止路线预过滤、card1 联合分配预算、时间线分档规则、资产政策方案、相机高度第二手段、人类包 v4）的顺序与边界是否认可。

## 4. 证据路径（均在 `/data/jzy/tmp/`，全部 research_candidate）

- 报告：仓库 `docs/roadmap/QA_V3_PRESCALE_READINESS_IMPLEMENTATION_20260902.md`（第二、三轮章节）。
- 参数：`qa_v3_prescale_params_tfull_placeholder_20260902_v{1,2}.json`；题型目录 `qa_v3_prescale_core_profiles_20260902_v1.json`。
- 核心小批 `qa_v3_prescale_core_{apartment,kujiale}_20260902_v2`；card1 smoke `qa_v3_prescale_card1_apartment_20260902_v{3,4}`、`qa_v3_prescale_card1_kujiale_20260902_v{2,3}`；48 格容量 `qa_v3_prescale_capacity48_{apartment,kujiale}_20260902_v1`。
- 216 重验 `qa_v3_prescale_revalidation_216_20260902_v3.json`；探针 `qa_v3_run02_{text,audio,video}_shortcut_probe_20260902_v2.json`；条件基线 `qa_v3_prescale_card1_conditional_baseline_{smoke_repro,historical,fresh}_20260902_v1.json`。
- 像素：37 条双侧 `qa_v3_prescale_card1_pixel_both_sides_20260902_v1/`（含 `pixel_join_tier_policy_v2.json`）；30 条多帧 `qa_v3_prescale_card1_pixel_timeline_20260902_v1/`；历史 card1F_002 两种政策 `qa_v3_prescale_card1F_002_pixel_join_{both_sides,tier_policy}_20260902_v1.json`。
- 机位：预检 `qa_v3_prescale_camera_clearance_preflight_{apartment_20260902_v3,kujiale_20260902_v2,apartment_v4smoke_20260902_v1,kujiale_v3smoke_20260902_v1}/`；高度实验 `qa_v3_prescale_camera_height_experiment_20260902_v1/`；单张耗时实测 `qa_v3_prescale_camera_table_timing_20260902_v1/`。
- 复核用图：`qa_v3_prescale_round2_review_frames_20260902_v1/`、`..._v3/`、`qa_v3_prescale_round3_review_frames_20260902_v1/`。
- 运行脚本备份：`qa_v3_prescale_round2_scripts_20260902_v1/`。

## 5. 测试配方

解释器 `/data/jzy/miniconda3/envs/avengine-runtime-probe-20260814/bin/python`；
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src $PY -m pytest tests --ignore=tests/unit --ignore=tests/test_verify_audio_batch.py -p no:cacheprovider -q`
在 `658c813` 上 374 passed；相关 unit `tests/unit/test_apartment_slot_bindings.py tests/unit/test_qa_v3_audio_batch.py` 15 passed。

## 6. 本工作线不声明的事

百题 pilot 已认证；可以大规模生成；人类容差已定稿；单模态最优策略失败；场景泛化成立。
