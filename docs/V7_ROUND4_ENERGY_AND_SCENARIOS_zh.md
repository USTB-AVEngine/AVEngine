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

`seeds_quota_v4_release_r3/`为本轮新请求表，不覆盖 `round3_production_20261003/seeds_release_v3/`。schema `v7-production-inputs-20261004-v4`，内部quota revision v4。v7-S train400/dev100仍为8/2套已分配房各50父场景；新版S训练行与新版full训练前400行逐字段/字节一致，dev前100亦一致。原有父种子、独立随机流、房屋划分和30秒/切窗计划保留。0说话人请求不含原生现场male/female speech焦点；通过固定相邻请求交换保持13类总配额和大小前缀一致，播放人声另按扬声器机制处理。

| 请求轴 | v7-S训练400父场景 | v7-S开发100父场景 | 完整训练7000父场景 |
|---|---|---|---|
| 说话人0–8 | 每档44–45 | 每档11–12 | 每档777–778 |
| STARSS13焦点类 | 每类30–31 | 每类7–8 | 每类538–539 |
| QA-21完整34焦点类 | 每类11–12 | 每类2–3 | 每类205–206 |
| 视锥内/外焦点 | 200/200 | 50/50 | 3500/3500 |

以上均为父场景请求数，**不是实际题数**；类对应不能凭程序字段准入。每个请求都带随机全360°相机朝向，独立遮挡因素、画内旁观者与四格槽位请求；四格是in_clear/in_occluded/out_clear/out_occluded。固定同一物理布局、房屋、材料、相机、背景、事件时刻，改变干声到预先测得槽位的发声关联。几何视锥真值不由遮挡决定；未测LOS或缺一格直接拒绝。FOA世界到相机旋转、方向、角度及FRUSTUM全部读取同一camera_forward。现行原生15Hz样本不先变粗。

说话人数监督使用窗口内发声的唯一稳定actor_id；多人同向不合并，同一人多段不重复计数，0人不等同于没有非人声。场景支持carry-in、重复、重叠和超过6个活动段；实际干声活动、声音类、人物/动作许可由families线准入，不能用计划事件代替它要求的实际干声证据。完整34与STARSS13是两个面板，不把播放讲话映成现场人声；音乐/水龙头独立覆盖。

声学请求短/中/长尾目标分别0.2–0.5/0.5–1.5/1.5–2.5秒，比例45/45/10%；长尾只是待准入请求，保持首轮“少量长尾”要求，不改变房屋RT门槛。S训练180/180/40、开发45/45/10、完整训练3150/3150/700。均为作者工程配额及Astra范围，未拟合真实波形。另含clean/稳态/非稳态/脉冲噪声（作者工程起点SNR为无/20/10/0 dB）及0–3个干扰声。长尾、噪声素材和35类尚待准入，本轮没有以STARSS波形拟合参数。每个speaker档都覆盖全部混响分层；music/tap的视锥答案和噪声因素在重复周期中轮换，避免类名编码无关因素。没有把候选86套当86套RLR通过房屋。

`generate_v7_scenes.py --plan-only`可验证v4请求，明确返回requested_only/native_not_admitted；实际调用拒绝以旧两人v3路径悄悄吞掉新配额。`scene_from_native_layout`与`select_spatial_cells`是采样/布局接口；正式native crowd、干声、QA-family resolver全部过后才能调度v4生产。四格实际合格组当前未交付；独立声学/画面/QA门禁分别保留。

实际题数使用 `audit_v7_scene_quotas.py`从新窗口公共题、私有canonical答案和场景证据重算；OA/MCQ只计一个logical_question_id。SPEAKERS按actor_id独立复核；所有13/34类先初始化0，缺项不冒充通过。原生静态多人诊断只作v0.2协议验证，候选M2仅两种身体网格、八个eSpeak声线配置；感知识别及正式身份/许可/运动准入尚未完成，不计v7-S量，且诊断面板不能豁免每窗12–15题。

## 文件分工及交接

本线没有编辑 `src/avengine/qa/`，不实现缺21题族、35类干声或人物许可目录。`v7-families`交付后仅按正式清单cherry-pick，由本线处理接口冲突。材质线正式清单已交付d790e8c、486799f；须按顺序合入并复验。材质验收pass与遮挡导致not_run分别记录；即使材质通过，移动能量未通过仍不冻结。CCI仅由cci-datagen线同步代码/搭环境/运行，本线没有登录或起CCI作业。

### 已生成诊断题数（不是配额计划）

两屋各0–8人＋180°相机对照，共36个真实CPU RGB/FOA诊断窗，110个不同逻辑题，每窗2–9题；正式合格题0。QA-04=14、QA-23=36、QA-25=8、STARSS-FRUSTUM=16、STARSS-SPEAKERS=36。人数0/1/2/3/4/5/6/7/8各实际4题，actor身份复算不一致0。FRUSTUM实际in_frame=8、out_of_frame=8，诊断面板达到答案均衡；不是v7-S已完成量。STARSS13及QA-21的34类每类题数均0，四格反事实合格组0，噪声/干扰和实际长尾配额未生成。原始ID带人数含义且相机对照重用ID，不能作训练交付；本轮新建中性媒体/题目别名，只改地址，不改题面/真值/公开域，110/110公开域一致。旧无限计数域`nonnegative_integer`及角度区间保留，未删溢出题或改答案。出处：`actual_quota_audit_final_v2/actual_question_counts.json`、`native_quota_neutral_diagnostic_v2/private_alias_manifest.json`；原始两个诊断目录保留不覆盖。

反向相机的FOA及全部派生读数共用camera_forward；8个静止嘴部逐个检查原生RIR方向≤5°才交付，未单独旋转标签而保持FOA不变。人物人数证明的是稳定actor实例，不是新增八套身体或八种已感知准入声线。事件活动是作者排程诊断，不能替代families接口要求的实际干声活动/湿声/语义内容准入。

### 材质合入与原生CPU复验

按正式清单合入原始`d790e8c`、`486799f`，不拾取参考快照`b6fa5e7`。只给新插入资产创建材质副本，原GLB/贴图/骨骼/动作不改，scan加载后再配置source PBR；不改RLR房屋材质。自动检查首/中/末帧的新增高光比例及原UV色差，失败中止；遮挡not_run独立记录。材质线118/118来自其报告，不能称本线CPU重跑118项。

本分支在所测两屋各实跑完整30秒、450帧、1280×720/15fps，共900原生CPU帧，编码时长/帧数由ffprobe核验；首/中/末两人共12/12项材质检查通过。高光比例最大0，色度p50/p95最大值见`evidence/native_material_CPU_summary.json`，阈值沿用材质线。另有六个配对抽帧证明auxiliary检查前后RGB和semantic完全不变。这是所测两资产/两屋的结果，不覆盖新增资产或全部房型。

额外的完整modal/target-only遮挡流在第53帧违反像素子集，root/joints/all_links/camera读回误差均0；原因未验证。未膨胀蒙版、没有1像素豁免、不改既有qa像素判据。完整遮挡收集保留为显式诊断选项`collect_pixel_visibility=True`，默认候选渲染写not_run，不供完整题族做已观测遮挡真值。原失败目录`native_visibility_full_v1/v2/v3`保留；有效的6抽帧不冒充全450帧通过。旧封存可直接跳过新增验收的具体问题已经修复：parent input及visual recipe带材质/遮挡版本，旧recipe被拒绝且旧文件保留；两项断点回归验证旧seal拒绝、新完整seal可复用。

### 题族线合入边界

正式清单为`23e2cdd`。它提供6个qa模块及语义/准入配置，唯一跨分工为`timeline/v7_native_questions.py`的可选qa_evidence接口；本线不手工改qa源码。只有真实外观复核、像素四态、干湿声活动、完整作者内容及角色证据可调完整题族；缺项deferred/拒绝，旧默认七族流程不当全28族。

独立题族线最终报告24/28有研究候选，QA-07/09/10/21仍0；1277个开放/MCQ答案形式独立复算一致。其10个种子窗口纯视觉只1或3道，要求4或5，完整14题窗0；严格两轴四成员绑定10/72=13.89%低于20%。原第3回合50个绑定ID统计范围不同，不能替代此次完整四成员/全部共同题面口径。35类正式干声0、人物正式准入0；研究副本不自动转正式。QA-25整数anchor书面/AVModel分歧须以该线最终协议交付对齐；本线不改AVModel、v6权威真值或评分器。来源：`v7_families_20261004/delivery/REPORT_zh.md`及该线validation_final_v3/admission_final。

### 范围与交接

仅48g代码检出修改/测试，全部本线作业CPU；不push，不登录CCI，不覆盖前3轮产物。正式生产仍不能冻结，S/full v4真实native配额resolver尚未准入，候选入口明确拒绝把v4降为旧两人场景。CCI接收本轮代码和`seeds_quota_v4_release_r3/`请求表/阻断清单，不能视作500父场景已生成。一次测试临时目录误落`/tmp/pytest-of-jzy/pytest-4387`，后续限定本轮目录，未清理共享/tmp。
