# v7 第五回合：材质回归与移动能量续验

材质修复已合入且所测24项通过；正式生成仍不可运行，原生深度/语义回归和移动能量未通过。本交付是诊断提交，不是生产冻结版；正式合格题数0。

## 范围与补丁

仅48g（`cw-SYS-4029GP-TRT3`，SSH别名`48g-jump`）任务检出`/data/jzy/code/AVEngine-v7-spec-acoustics-20261003`、分支`claude/v7-spec-acoustics-20261003`改和测。A=`/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/round5_material_merge_20261004/`；O=同级`round3_production_20261003/`，A4=同级`round4_energy_scenarios_20261004/`均既有证据只读。不push、不运行GPU、不登录CCI、不改测试面板，不修改原数据/共享环境，不删除已有attempt/cache。

正式清单`/tmp/codex_asset_materials_20261003/CHERRY_PICK_zh.md`（本机）已核对。原始d790e8c8c0bfdf96aeed7158d97aac12045ea896对应6f1eb8732ad4a7ef3f1daf051415c6ad81b807e2，原始486799f3d57caaae44ab692bf8cef41a396defb6对应7ee6c6f8b277e580877eada5ee7254ece21a1eb7，已在第4回合按顺序合入。本回合核对稳定patch-id（生成的TOOL_INDEX除外）、先后祖先关系；不重复cherry-pick、不合入b6fa5e7。证据`A/evidence/material_merge_audit.json`。

根因是旧Phong白色镜面、glTF非金属默认metallic=1、PBR纹理/输出颜色映射。只修改新派生GLB材质和插入资产shader，原GLB几何BIN前缀、UV/baseColor/alpha、骨骼/动作保留。扫描场景先加载，再配置source-only PBR；关IBL，保留灯光/曝光，材料纹理映射linear、输出sRGB。声学材质不随视觉修复改变。

正式候选验收接线：`source_materials.py::measure_source_appearance` → `NativeContinuousVideo._check_source_frame`；首/中/末检查，失败终止。高光毛皮2%/人类4%/道具8%/明确金属12%，色度p50≤0.12/p95≤0.28，少于100可见像素记not_run。`source_materials.json`和`source_material_checks.json`是visual完成所需文件；parent/visual recipe包含版本，旧seal拒绝、旧文件保留。这些接线在第二条材料补丁和第4回合断点修复中已完成，本回合没有改门槛。

## 原生CPU渲染与其他产物

before读取Git1079b1e的renderer类，after为当前源码；两者使用相同种子、room、相机、背景、资产与物理轨迹。这里before是v3的FLAT渲染，不能称重现资产线22个Phong旧异常。两工作进程，每次独立native子进程；软件EGL/llvmpipe强制CPU并禁止NVIDIA GL/EGL；每次450帧、30秒、1280×720/15fps。8次native渲染共3600帧，ffprobe读回一致。v3四成员共两种视觉状态，后续声音关联成员共享视频。

| 场景 | 房屋/房间 | 首中末材质通过 | depth完全相同 | semantic完全相同 |
|---|---|---:|---:|---:|
| four_member_v0 / v7_train_000056 | hm3d_train_00096_6HRFAUDqpTb / R7 | 6/6 | 0/450 | 430/450 |
| four_member_v1 / v7_train_000056 | hm3d_train_00096_6HRFAUDqpTb / R7 | 6/6 | 0/450 | 430/450 |
| S_train / v7_train_000050 | hm3d_train_00096_6HRFAUDqpTb / R13 | 6/6 | 0/450 | 437/450 |
| S_dev / v7_dev_000001 | hm3d_val_00862_LT9Jq6dN3Ea / R1 | 6/6 | 0/450 | 349/450 |

材质24/24通过，高光最大0，色度p50最大0.013505601、p95最大0.168953901；来自`A/material_replay_v4/summary.json#/cases/*/new_material_checks`。动画嘴部轨迹451点×两actor在每case前后一致，固定相机actor-free RGB/depth/semantic逐位相同。重跑v3四成员8个10秒窗每窗14题、共112逻辑题，重新按新visual证据出题，公共问题/私有答案与旧封存逐字段相同。FOA/mono直接复用旧文件后逐字节核验，**本回合未重新合成RIR/FOA**。实际反事实审计vision16/audio14/binding25比较均pass；该历史v3范围不替代新全28族或严格绑定配额。来源`A/material_replay_v4/four_member_comparison.json`。

1800组对应帧中semantic154组不完全相同，depth0/1800组完全相同。12个抽样帧最大depth差0.4472379684m出现在S_dev第0帧，那里semantic变1像素；其余抽样帧depth最大差约0.001–0.002m。未测全部帧的最大差。所有帧每source可见像素计数差最大2，不能称整图仅变2像素。几何/嘴部/背景不变不足以证明带资产的全部sensor输出不变，差异成因**未验证**。没有复制原depth/mask或1像素豁免。`summary.json`明确`fail_raw_sensor_identity_materials_pass`，工具exit2记录检查失败。

资产线的118/118、59资产和22旧异常是其原生矩阵结果，未在本回合全量重跑；本线只重跑两位人物。其HM3D遮挡抽检仍`not_run`；深色缅甸猫预期毛色仍**未验证**（该线记录原UV约98.31%黑，不能当预期毛色验证），已补规格书§15。另有本线第4回合全帧像素子集第53帧失败，状态fail，与资产线not_run不混同。

S_train最初选00059固定v3路径，在before/after均因离开准入导航支持而失败；失败产物保留。为完成两套S屋回归，改用既有S训练00096/R13、开发00862/R1的固定种子，未改种子文件/规则。登记和整屋通过不意味着所有动态路线可行，v4 resolver仍需逐路径拒绝/补合格场景。`material_replay_v1`保留导航失败，v2完成替代native渲染，v3为复用记录文件冲突失败，v4复用已完整同输入的8次渲染，只重做诊断与切窗。v2曾因要求逐位相同而中止汇总，v4仍记fail，不放宽检查。

## 移动能量：沿用80点，继续研究不换判据

第4回合同80时刻a(false0现状)/b(true10)/c(false0最小0.1秒加密)/d(true10加密)仍17/9/17/9。每场景生产279条RIR；train00096的a/c与b/d CPU分别1263.178/1252.687核秒，test00808分别1294.218/1278.125。a/c计时来自旧cache，b/d为当轮冷跑；参考RIR另计。原生15Hz轨迹间隔≤0.0666875秒，所以c/d增量RIR0，未丢轨迹制造加密收益。完整最大误差及衍射覆盖链见第4回合报告和`A4/energy_report/summary.json`，未改变1dB能量、1dB DRR、0.25ms时延、5°方向、RT与尾部门槛。

本回合先对两屋t=4.1/7.7的同位置做三次独立原生渲染，true10，仅将direct/indirect/source射线数量×1/×4/×16，材料/几何/IR物理设置不变，每worker1线程，两worker。

| 射线预算 | 重复比较通过 | 原生RIR数 | renderer CPU核秒 |
|---|---:|---:|---:|
| ×1 | 5/8 | 12 | 33.493 |
| ×4 | 5/8 | 12 | 125.926 |
| ×16 | 8/8 | 12 | 485.283 |

×16小样本CPU是×1的14.49倍。同数组正控制12/12通过，已知+6.0206dB增益与+0.5ms时延坏控制分别12/12被拒绝。8/8只证明所测重复对照，不证明80个移动中点。来源`A/native_repeat_quality_v1/summary.json`。

再在同80时刻，以×16、true10独立端点与参考native上下文做五种离线诊断。取现有451点轨迹的必要端点，不先变粗；波形长度不同时零补齐尾部。到达对齐只用几何距离/343m/s的延时；晚期等功率使用几何到达后的50–80ms过渡，同位置直接复用一条RIR；没有按真参考拟合增益、四通道归一化或频带补偿。最近邻是诊断控制，没有替代正式交叉淡化方案。

| 离线策略 | 通过 | 最大直达dB | 最大总dB | 最大倍频带dB | 最大DRR dB | 最大时延ms |
|---|---:|---:|---:|---:|---:|---:|
| linear | 33/80 | 3.133 | 1.945 | 7.783 | 4.266 | 3.0000 |
| arrival_aligned_linear | 36/80 | 3.984 | 1.400 | 2.247 | 5.218 | 3.0625 |
| nearest | 74/80 | 5.326 | 0.537 | 1.660 | 5.882 | 1.5625 |
| arrival_aligned_nearest | 62/80 | 5.320 | 0.537 | 1.660 | 5.942 | 3.0625 |
| early_linear_late_equal_power | 35/80 | 3.984 | 0.877 | 1.957 | 4.945 | 3.0625 |

train00096端点60条2519.331核秒、独立参考32条1330.823；test00808端点60条2556.470、独立参考32条1337.887。合计184条7744.511核秒；16workers且每native1线程，小于48g64限额。该CPU仅renderer process_time，不含setup/布局/Python，也**不是完整30秒场景CPU**；未重写生产吞吐预算。原生所有任务完成后首次汇总遇到可变长度broadcast错误；保留原IR，在新`high_quality_fixed80_evaluation_v2/`补零并离线重算，new_native_calls=0。源native文件在`high_quality_fixed80_v1/`，原失败日志保留。

所有策略未全过，`selected_formal_policy=null`，未进入生产。repeat×16的提高说明提高原生采样质量有价值，但还未解决移动直达/DRR/时延及插值。true10读回不等于SDK实际遮挡绕射改善已验证。没有基于STARSS波形调材料或参数。

## 场景配额与交接边界

第五回合没有新生成v4配额题；材质回归112题不新增计量。第4回合实际36诊断窗110逻辑题：QA-04=14、QA-23=36、QA-25=8、FRUSTUM=16、SPEAKERS=36；0–8人各4题、画内/外8/8、13原生类及34完整类逐类0、合格四格0、正式题0。出处`A4/actual_quota_audit_final_v2/actual_question_counts.json`。S/full固定前缀仍`A4/seeds_quota_v4_release_r3/`；请求表不算已完成媒体，v4实际resolver、干声/人物许可、28族/30-25-45/严格绑定≥20%、长尾与新profile整屋准入仍阻断。

测试面板运行目录`/home/neos/jzy/codex_runs/20261003-202503_complex_test-panel-pilot`只作为owner转发对象，本线未在该目录写文件/起任务或发消息。新提交号、可同步源码和`TEST_PANEL_HANDOFF_zh.md`在`A/delivery/`；明确production_ready=false，禁止启动正式批量。CCI仅由另一线同步，未在CCI运行。本回合新增3个诊断工具、规格书§15、本文及生成工具索引；未手工改qa目录、评分器、真值或门槛。完整脚本/日志在A/scripts和A/logs。

## 软件核验

48g材质/streaming/断点focused27项通过（2.43秒），来源`A/logs/tests_material_resume_final.log`；fast_unit389通过、36因缺保留fixture跳过、6999未选（32.21秒），来源`A/logs/tests_fast_unit_final.log`。三个新工具CLI加载、合并工具索引检查和Git diff空白检查通过。原生渲染/媒体读回、能量对照是上文真实执行，软件测试不替代失败的原生资格。
