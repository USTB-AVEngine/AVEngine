结论：已补满12套原HM3D val测试房；原8套训练、2套开发保持不变，原train兜底未启用。

机器48g（cw-SYS-4029GP-TRT3）；分支`claude/v7-fill-test-houses-20261003`，基点`3414629`，最终提交见产物根`COMMIT.txt`，不push。独立worktree：`/data/jzy/code/AVEngine-v7-fill-test-houses-20261003`；产物根：`/data/jzy/tmp/claude_hardbank_20260923/v7_batch2_test_20261003`。最多2个验收worker×RLR 8线程；自有作业共用16个逻辑CPU调度范围。全程CPU，原始数据、第一批、其他检出和共享环境未改。

先冻结候选顺序与每屋≤3间抽样，再用原版复测00808/00877；逐项验收结论一致，RT60等数值未逐位一致（`evidence/calibration_comparison.csv`）。发现最大峰和较宽能量窗错算晚反射后，统一使用首次到达及原已固定的FOA 0.5ms直达窗；门槛、材质、传播设置不变。旧28屋＋新12屋全部重跑CPU测量，所有保存RIR按同一修正版重算，旧结果保留。45组完整RIR重复的该窗能量相对跨度最大9.27e-7（`evidence/direct_component_window_repeat.json`）。

新增12屋：10通过、2失败；00873漏声29.10%、00815漏声17.58%。本轮40屋中38屋有RIR，共424条有效RIR、159744条基线射线、111张CPU视图（`report_final/summary.json`）。00847/00853修正后通过，暂留备选；00843独立求交复核41/512＝8.01%，仍待审。6套已知暴露MP3D全排除，2套退化输入仍无RIR。

推荐v2（全名、房型/候选面积、逐项指标及路径：`report_final/recommended_houses_v2.json`、`all_house_results.json`、`house_results.csv`，格式沿第一批）：

- 训练：00059, 00096, 00155, 00168, 00272, 00382, 00712, 00746；原8套。
- 开发：00880, 00862；原2套。
- 测试：00808, 00877, 00844, 00829, 00821, 00832, 00802, 00839, 00810, 00871, 00813, 00890；保留原2套，新增10套按冻结的原val面积分位大/小交替顺序接纳，不按模型分数/RT60排序。

验证：专项21通过，fast_unit 292通过/36跳过，索引/diff通过（`evidence/cpu_test_summary_final.json`）；7份已登记训练/回放/v6清单重读，45候选零已知命中（`evidence/live_exposure_audit.json`）。旧28屋对比见`report_final/all_batch1_uniform_comparison.json`。

限制：每测试屋60四元组只预留720种子/2880成员ID，实际生成0（`report_final/test_quadruplet_seed_plan.json`）。匿名/未登记历史暴露、正式Habitat RGB、未抽样位置及生产准入未验证；扩展历史资源测试未复跑。原生仅直达控制受既有材质契约阻断，未完成（`evidence/direct_only_control_not_run.json`）；未削弱契约，验收仍用原完整传播。
