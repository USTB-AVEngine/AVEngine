# P5 可行空间与固定画像条件采样器

## 1. 文件与提交

提交为本报告首次加入当前分支的提交；P2 基线 c1098a9，P12 包及共用物化依赖已提交 5e2002f。权威目录 48g-jump:/data/jzy/tmp/wt-multi-home-activity-integration。未 push、merge 或切换 Studio。

新增 src/avengine/rooms/{walkable_space,conditioned_sampler}.py、src/avengine/capture/qa_plan_adapters.py、src/avengine/qa/answerability.py、tools/qa/measure_conditioned_plan_matrix.py。接入 src/avengine/rooms/{qa_episode,native_qa_room}.py、tools/studio/run_qa_episode.py、tools/rooms/run_spear_residential_episode.py，修正 capture/habitat_neutral_readback.py 的 actor/skin root 语义。测试 test_conditioned_sampler、test_qa_production_contracts；索引 docs/TOOL_INDEX.md 已按工具刷新。物化细节见 P5_materialization.md，HM3D 实跑见 P5_hm3d_validation.md。

可行空间的 is_navigable/shortest_path/sample_navigable/floor_height/bounds/route_bank 用现有家具栅格、保留 walkable-grid 文件、原生路线库、Habitat PathFinder 实现。UE 单位/轴转换只在 adapter，core 为米/+Y/右手。原生 Apartment 使用完整已有路线点，禁止转成栅格路线。

仅 conditioned_static_v2 进入新 core。固定 N/S/类别/anchor_count/角距箱/可见性/直达/运动/事件画像，在画像内最多 200 次重试；不以其他画像补数。合法机位全量均匀抽，0.55 m 网格、地面+1.55 m、默认85°，垂直视野按实际传感器 aspect。所有其他实体（含画外）参与最近竞争者；只有抽定锚点事件受箱约束。保留合法起点的可行后缀，整数 sample 半开区间实现 sequential/overlap/repeat。声音按实际片长、剩余预算和类别规则过滤，再随机；P7 original source activity 扣 crop offset 后进入派生声片坐标。

## 2. 测试

使用项目 Python 和 PYTHONPATH=src:tmp/native_python_addons_v1。相关组合 **62 passed, 0 failed, 0 skipped，15.39 秒**：sampler、P1契约、共用物化、P2新策略、native Apartment、question-driven rooms、Habitat capture、工具索引。日志 tmp/p5_sampler_20260906_v1/parent_final_tests_v6.log。随后补充实际 UE camera frame-index 回归断言，sampler 15 passed、0 failed、0 skipped，1.67秒。

覆盖固定画像失败直方图、同 seed 字节一致/不同 seed 差异、合法机位、整数合法窗含 sample0、三类事件关系、刚体两源不被默认 min_articulated_count 排除、unknown gender 拒绝、原生路线点保留、几何纯函数、frame-index/分辨率透传，以及因 skin offset 转向造成假 moving 的回归。

真实首次 A capture 在最终 camera-state 检查报 expected camera state frame order changed，原因是 adapter 缺 frame_index；已修复并用 fresh capture_retry_v2 完整重跑。Habitat common seed 大于 int32 的失败现场保留，修复在 P12 的 M1 接口映射；不改变原始计划 seed 或 clock。

## 3. 验收计划、原生读回与矩阵

同一采样器的真实计划及执行：

- A：tmp/p5_sampler_20260906_v1/a_plan_v3/plan/episode_plan.json；完整原生输出 capture_retry_v2/，240帧/16秒，两个实际源。
- MP3D：tmp/p5_sampler_20260906_v1/mp3d_shared_plan_v5/plan/episode_plan.json；capture/，240帧/16秒，正式 registry 蓝/绿人形和原生 MP3D glb，非 UE 导入替代场景。
- 两段 root 与计划位置最大误差均 0；A camera 误差0，MP3D 3.82e-7 m；中立 readback 和240帧像素真值均通过。复核 tmp/p5_sampler_20260906_v1/parent_native_plan_readback_v1.json。MP3D 新 root 语义另写 neutral_readback_actor_root_v2.json；两人包变换为identity，实体值与原文件逐值相同。
- 我查看两段首帧：A有绿衣/蓝衣两人，MP3D两人及原生室内清楚可辨。该检查是 Codex 图像检查，不能记作人工外观或可答性验收。
- 移动画像单段计划：a_moving_plan_v1（62次内成功）、native_moving_plan_v1（10次内成功），均位于 tmp/p5_sampler_20260906_v1/。Native plan 保留原生路线库，不用栅格替代。
- HM3D 额外完整240帧验证：tmp/p5_hm3d_validation_20260907_v5，两个角色原生 RGB/depth/semantic、target-only、root/emitter/camera、clock均完成，实际遮挡状态保留；另见附报。

Plan-only 矩阵：tmp/p5_sampler_20260906_v1/matrix_50/{measurement_config.json,trials.jsonl,summary.json}。4房×N=2/3/4×4角距箱×50 seed，共2400次，成功34次；每格分母均50。显式 anchor_count=1、clip_span_filter=true、speaker_moving。每 seed 仅一次路线尝试，沿用此前审核脚本的单次路线测量口径；这不是生产默认200重试的成功率，也不是native可答性结论。原始 producer 记录实跑 cwd、Python、当时 Git 基线及工作树变更。

| 房间 | N | 15–30° | 30–60° | 60–90° | 90–180° |
|---|---:|---:|---:|---:|---:|
| A | 2 | 0 | 2 | 2 | 0 |
| A | 3 | 0 | 2 | 0 | 0 |
| A | 4 | 0 | 0 | 0 | 0 |
| B | 2 | 0 | 2 | 0 | 0 |
| B | 3 | 0 | 0 | 0 | 0 |
| B | 4 | 0 | 0 | 0 | 0 |
| C | 2 | 1 | 6 | 2 | 0 |
| C | 3 | 0 | 0 | 0 | 0 |
| C | 4 | 0 | 0 | 0 | 0 |
| Native Apartment | 2 | 7 | 7 | 1 | 0 |
| Native Apartment | 3 | 2 | 0 | 0 | 0 |
| Native Apartment | 4 | 0 | 0 | 0 | 0 |

总失败直方图（完整逐格直方图在 summary.json）：

- routes:path_too_short_for_moving_window: 516
- routes:no_existing_navigation_path: 480
- routes:initial_source_separation_below_0.95_m: 287
- routes:native_group_endpoint_separation_above_3.5_m: 271
- routes:all_frame_source_separation_below_0.95_m: 267
- camera:no_joint_geometry_activity_schedule: 257
- routes:native_group_separation_below_0.95_m: 154
- routes:route_does_not_fit_clock_or_required_window: 127
- routes:sampled_path_left_existing_navigation: 6
- sounds:clip_budget_or_transcript_candidates_exhausted: 1

### P11真实反馈后的body LOS修正

Native Apartment sampled v1中beagle240帧fully_occluded；同一原始静态mesh复算
显示registered emitter射线clear，但既有body proxy和实际body中心都blocked。
clear画像现在同时检查emitter与body proxy，仍仅新conditioned策略生效，仍在
全部合法候选内均匀抽样；pixel_observability继续not_run，不把几何代理当像素认证。
诊断：tmp/matrix_native_apartment_beagle_speaker_sampled_20260907_v1/los_*diagnostic.json。

同seed202609071401重规划输出sampled_20260907_v2，attempt6、7个合法机位；
原失败机位不再进入该结果的候选池。最终与P11骨骼emitter执行器完成native
240帧/16秒、P6/P9收口（13 valid/11 deferred）；实际局部桌沿遮挡如实保留。

受影响sampler/P2测试21 passed，后续完整P11整合集合见P11报告。相同种子/
输入/一轮路线尝试口径的2400次矩阵已重跑：
 tmp/p5_sampler_20260906_v1/matrix_50_body_los_v2/summary.json，188.72秒，
34/2400成功，各格计数和失败直方图与原matrix_50一致；这是plan-only单次尝试
产出率，仍不是默认200次预算的生产成功率或原生可答性。

## 4. 未完成与边界

四类可行空间和两种计划执行器已接通；上述验收完成。矩阵中的零格保留 evidence_missing_or_unsampled，未缩角距或补容易样本。计划只含 condition_profile 与 planned_conditions；achieved_conditions 留给 native readback 后重算。P9最终收口、P11 UE混合资产、P10批级覆盖以及46段仍有独立要求，本项不声称其完成。P7候选池已机器重测，真实人工试听仍缺失。

## 5. Claude 接口

共用几何函数位于 avengine.qa.answerability：listener_azimuth_deg、separation_stats、max_concurrent_entities、structural_baselines、line_of_sight。没有修改 Claude 的 audit_binding_feasibility.py 或其测试。line_of_sight 需要真实 static mesh；缺失几何返回未测而非伪造 clear。模型输入不可包含 profile/target/geometry 私有字段。

Common visual plan 的 actor_states 使用 root_transform、action_id、action_phase、action_time_ticks、moving；camera position/basis/FOV/resolution在中立坐标。UE executor才填驱动字段；Habitat物化器消费同样轨迹，原生 skin root 按包 actor_from_skin_root 转回 canonical actor root，actual emitter 始终来自native joint，不复制planned值。

## 6. Owner 决策

未改变旧请求行为、0.95 m 源间距或角距下限、原生路线授权、严格证据规则、全矩阵分母。没有本项新增 owner 决策。46段只在所有前置验收满足后运行；不以当前计划可运行或测试通过替代整项研究验收。
