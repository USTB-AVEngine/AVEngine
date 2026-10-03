# v7 第四回合：移动能量与场景配额

代码与原生检查均在48g任务检出；不使用GPU、不登录CCI、不推送。前3轮产物只读；本轮新证据根为 `v7_batch1_20261003/round4_energy_scenarios_20261004/`。正式生成仍被阻止，当前没有一个四组方案通过能量验收，不能冻结生产版。这里的提交是可审查代码和失败诊断交付。

## 衍射覆盖审计

实际v6 HM3D生产 Episode：`render_prod/hm3d__QA-06_moving_s20269600/episode/`，以 `delivery/commands.json#/audio` 和 `research_receipt.json#/qa/propagation/simulation` 为实测权威。音频命令同时传入 `rir_cache_simulation_request_v2_threads8.json`（diffraction=true/order10）以及 `--no-diffraction --max-diffraction-order 0`，最终有效配置false/0。

来源链：`examples/dataset/qa_production_10s_20260908.json#/base_request`明确false/0；`src/avengine/rooms/qa_delivery.py::_habitat_audio_command`将批次请求编成显式CLI参数；`src/avengine/timeline/current_mp3d_dynamic_audio.py::_simulation_from_inputs`先读仿真模板，再用显式diffraction/order覆盖。Git blame为 `dcbcc11407614616866023ddfb4d3732326445a5`，Git作者Eastforward，2026-09-08 19:57:51 +0800。不能由Git作者字段推断实际操作人员。

更早的中性读回默认false/0来自 `93a14e70581854fc160da8c05f98e26f277d2fe3`（同Git作者，2026-09-07 03:27:21 +0800）；对应P6文档写明“绕射验收使用显式 false/order 0；是否启用绕射及采用哪个 max order 仍由 owner/校准任务决定”。这说明当时将关闭作为验收起点。09-08生产批次沿用的单独决策原因未找到记录，性能原因未验证。第三回合v7输入复制的是上述v6实际有效配置false/0，未新增隐性衍射覆盖。

四组b/d明确请求true/order10，并保存 `diffraction_on_native_setup.json#/configuration_readback` 的原生读回true/10。CPU线程显式为1；其余材料、几何、射线预算、IR长度与原生嘴部轨迹相同。开启衍射会改变遮挡/绕射路径，v7将与关闭衍射的v6发生声学差异，属于明确的物理修正；已有false/0整屋通过结论不能直接当true/10资格。推荐继续研究true/10，当前没有已通过的生产配置。

## 同一80中点的四组结果

门槛保持原值：直达/总/125–4000 Hz六倍频带能量误差≤1 dB，DRR≤1 dB，到达峰值差≤0.25 ms；FOA及几何误差≤5°、RT60差≤max(0.05秒,参考的10%)、尾部不截断。未改判分器、真值或门槛。

| 方案 | 过线/80 | 最大直达能量差 dB | 最大总能量差 dB | 最大分频带差 dB | 最大DRR差 dB | 最大到达差 ms |
|---|---:|---:|---:|---:|---:|---:|
| a false/0现状 | 17 | 4.930 | 2.362 | 7.874 | 6.692 | 3.000 |
| b true/10 | 9 | 8.478 | 2.374 | 7.562 | 7.508 | 3.000 |
| c false/0＋最小0.1秒二分 | 17 | 4.930 | 2.362 | 7.874 | 6.692 | 3.000 |
| d true/10＋最小0.1秒二分 | 9 | 8.478 | 2.374 | 7.562 | 7.508 | 3.000 |

每屋40个参考时刻与上一轮完全相同，其中有停步位置的重复时刻；每屋独立参考cache实际32条新RIR及8次位置复用，不能称80个独立空间位置。源2轨迹451个原生15 Hz骨骼点，最大间隔0.0666875秒；保留所有原生轨迹拐点。0.1秒下限使c/d没有可二分区间，新增RIR为0，c=a、d=b。如把轨迹先变粗再宣称二分节省，将不再是同一现状对照。

| 房屋/场景 | a/c生产RIR数 | a/c原生CPU秒 | b/d生产RIR数 | b/d原生CPU秒 |
|---|---:|---:|---:|---:|
| train 00096 / 6HRFAUDqpTb | 279 | 1263.178 | 279 | 1252.687 |
| test 00808 / y9hTuugGdiq | 279 | 1294.218 | 279 | 1278.125 |

a/c复用上一轮已测生产cache及计时，不冒充本轮冷启动；b/d本轮冷启动，两屋并行，每worker1线程。参考验证RIR及CPU另列在 `energy_four_arms/*/*_result.json#/midpoint_reference_stats`，不计生产RIR。CPU是RLR renderer的process_time，墙钟和Python全stage时间另列；c/d共享完全相同波形只统计增量为0。图：`energy_report/energy_four_arms.png`及PDF。完整指标和配置来源：`energy_report/summary.json`。

原生扫描网格射线80/80没有源到听者遮挡。同位置独立原生重复8次比较只有3次过线（5次分频带能量超1 dB）；同数组正对照4/4过，已知+6.0206 dB坏对照4/4正确失败。重复控制不调整门槛。端点波形相关性及精确协方差说明线性波形混合有抵消；最近邻和几何到达对齐作为额外离线诊断也未全部通过，不接入生产。证据：`energy_diagnostics_v2/summary.json`、`energy_waveform_controls/summary.json`。首次射线诊断未完成模拟即查ray而失败，修复后新目录重跑；失败目录/log保留。

## 场景输入、配额与实际题数

`seeds_quota_v4_release_r2/`为本轮新请求表，不覆盖 `round3_production_20261003/seeds_release_v3/`。schema `v7-production-inputs-20261004-v4`，内部quota revision v3。v7-S train400/dev100仍为8/2套已分配房各50父场景；新版S训练行与新版full训练前400行逐字段/字节一致，dev前100亦一致。原有父种子、独立随机流、房屋划分和30秒/切窗计划保留。0说话人请求不含原生现场male/female speech焦点；通过固定相邻请求交换保持13类总配额和大小前缀一致，播放人声另按扬声器机制处理。

| 请求轴 | v7-S训练400父场景 | v7-S开发100父场景 | 完整训练7000父场景 |
|---|---|---|---|
| 说话人0–8 | 每档44–45 | 每档11–12 | 每档777–778 |
| STARSS13焦点类 | 每类30–31 | 每类7–8 | 每类538–539 |
| QA-21完整34焦点类 | 每类11–12 | 每类2–3 | 每类205–206 |
| 视锥内/外焦点 | 200/200 | 50/50 | 3500/3500 |

以上均为父场景请求数，**不是实际题数**；类对应不能凭程序字段准入。每个请求都带随机全360°相机朝向，独立遮挡因素、画内旁观者与四格槽位请求；四格是in_clear/in_occluded/out_clear/out_occluded。固定同一物理布局、房屋、材料、相机、背景、事件时刻，改变干声到预先测得槽位的发声关联。几何视锥真值不由遮挡决定；未测LOS或缺一格直接拒绝。FOA世界到相机旋转、方向、角度及FRUSTUM全部读取同一camera_forward。现行原生15Hz样本不先变粗。

说话人数监督使用窗口内发声的唯一稳定actor_id；多人同向不合并，同一人多段不重复计数，0人不等同于没有非人声。场景支持carry-in、重复、重叠和超过6个活动段；实际干声活动、声音类、人物/动作许可由families线准入，不能用计划事件代替它要求的实际干声证据。完整34与STARSS13是两个面板，不把播放讲话映成现场人声；音乐/水龙头独立覆盖。

声学请求含已有物理声学包的短/中/长尾分层、clean/稳态/非稳态/脉冲噪声（作者工程起点SNR为无/20/10/0 dB）及0–3个干扰声。长尾、噪声素材和35类尚待准入，本轮没有以STARSS波形拟合参数。每个speaker档都覆盖全部混响分层；music/tap的视锥答案和噪声因素在重复周期中轮换，避免类名编码无关因素。没有把候选86套当86套RLR通过房屋。

`generate_v7_scenes.py --plan-only`可验证v4请求，明确返回requested_only/native_not_admitted；实际调用拒绝以旧两人v3路径悄悄吞掉新配额。`scene_from_native_layout`与`select_spatial_cells`是采样/布局接口；正式native crowd、干声、QA-family resolver全部过后才能调度v4生产。四格实际合格组当前未交付；独立声学/画面/QA门禁分别保留。

实际题数使用 `audit_v7_scene_quotas.py`从新窗口公共题、私有canonical答案和场景证据重算；OA/MCQ只计一个logical_question_id。SPEAKERS按actor_id独立复核；所有13/34类先初始化0，缺项不冒充通过。原生静态多人诊断只作v0.2协议验证，候选M2仅两种身体网格、八个eSpeak声线配置；感知识别及正式身份/许可/运动准入尚未完成，不计v7-S量，且诊断面板不能豁免每窗12–15题。

## 文件分工及交接

本线没有编辑 `src/avengine/qa/`，不实现缺21题族、35类干声或人物许可目录。`v7-families`交付后仅按正式清单cherry-pick，由本线处理接口冲突。材质线正式清单已交付d790e8c、486799f；须按顺序合入并复验。材质验收pass与遮挡导致not_run分别记录；即使材质通过，移动能量未通过仍不冻结。CCI仅由cci-datagen线同步代码/搭环境/运行，本线没有登录或起CCI作业。

### 已生成诊断题数（不是配额计划）

本轮两屋各0–8人，共18个真实CPU RGB/FOA诊断窗，55个不同逻辑题，每窗2–9题；正式合格题0。QA-04=7、QA-23=18、QA-25=4、STARSS-FRUSTUM=8、STARSS-SPEAKERS=18。说话人数0/1/2/3/4/5/6/7/8各实际2题，actor身份复算不一致0。FRUSTUM实际in_frame=8、out_of_frame=0，未达画内外均衡。STARSS13及QA-21的34类每类题数均0，四格反事实合格组0；这些仍未完成，不能以请求表替代题量。出处：本轮`actual_quota_audit_v1/actual_question_counts.json`及`native_crowd_diagnostic_v1/summary.json`。

本轮额外检验同相机/同姿态的原生modal与target-only遮挡证据，复用既有v6像素判据而未改qa代码。首轮失败；第二轮改为先设置root再设置关节，训练屋3帧通过且原RGB/semantic不变，测试屋第0帧未通过像素子集一致性，未接正式生成入口。失败产物和日志保留在`native_visibility_controls_v1/v2`及`logs/`。
