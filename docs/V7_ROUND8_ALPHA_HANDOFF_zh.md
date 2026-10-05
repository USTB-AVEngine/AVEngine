已交付 **160 个正式 v7-S alpha 窗口、2240 题**：20个独立父场景/两训练屋各10父，全部每窗14题，可用受限alpha先训练；完整版v7仍有配额缺口。

正式数据在48g `/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/round8_alpha_20261004/v7_S_alpha20_release_v1`，训练入口`train.jsonl`（10秒视频、FOA/W单声道、公开题、私有原答案/证据、干声原ID/许可、精确RIR齐全）。逐窗/逐父数据见该目录`windows.csv`/`parents.csv`；详细中文报告在父目录`REPORT_zh.md`，下列相对出处均以父目录为根。

| 实际指标 | 结果 | 出处 |
|---|---|---|
| V/A/AV | 672/560/1008，30%/25%/45% | release `summary.json` |
| 成组认证绑定 | 800题，35.714%；20四成员组通过 | 同上及各父反事实证书 |
| 每父CPU/墙钟中位 | 1873.256/1712.244秒；CPU含子进程，16 CPU workers | release `parents.csv` |
| 原生/媒体验收 | 40世界/18000帧；160/160媒体/2240答案复验 | `evidence/portable_release_validation_v3.json` |
| 材质/方向 | 可测3892过、1548像素不足not_run、失败0；最大直达误差0.02460°；5编码窗10检查过 | `evidence/final_material_and_panel_audit.json`及`final_five_windows/REPORT.json` |

拒绝排行按“题族×候选窗口”计（分母160池，非失败正式窗）：自编语义/许可链未齐480（QA26–28）、发声后无合格静默窗480（13/16/17）、真实干扰项与答案相同440（06/24/04）；遮挡者身份、距离趋势、采样窗、v6活动段审阅、34类真值各160（10/15/18/23/21）。逐条见`rejection_rank.csv`和release `rejections.json`。alpha暂缓这些无证据题，选其他原规则合格题；来源许可、真实原生接地/像素/材质/方向、14题/模态、反事实及媒体时钟均不可绕过。v6/QA23仍拒160，实际选v0.2/QA23=120，按面板区分。

VCTK `vctk_automatic/VCTK_ADMISSION.json`/CSV：**600/600、24人、342句**，CC BY4.0和原ID/crop/署名齐全；F300/M300与原元数据一致。时长1.583125–5.993375秒；目标−23LUFS且真峰≤−3dBFS，实际−30.83至−22.96LUFS、最大真峰−3.02dBFS。自动Speech/活动/来源去重收据齐全；禁止源参考只用于排除，不作干声/参数拟合。去重不含SAVVY，阳性控制49/58，范围/漏检照实保留。

实际题量见`family_counts.csv`：16个v6题族加3面板，QA02=800、07=238、11/14各177、23=120、25=112；其余完整明细及每窗14题在CSV。实际只用了1底模/4已许可站姿衣色、8男声/21录音；可用非讲话19类/86条本批使用0。QA26–28、走路/移动声源均0；人数题仅2人124题、13类仅男声40题、画外80/画内0、完整34类题0，carry-in28/160、≥0.5秒重叠136/160。完整运动、全类别/人数/空间四格、E/S预测与真实集训练评测未完成；STARSS≥60未验证。HM3D全面资产遮挡抽检not_run、深色缅甸猫预期毛色未验证。原判分器/真值/门槛不改；QA14仅明确原被比较的两人，已在`REPORT_QA_FILE_EXCEPTION_zh.md`记录。回归508 passed/36 skipped，日志`logs/pytest_fast_alpha_frozen_v1.log`。

CCI计划（本线未登录/运行）：固定`seeds_alpha_500_v1/v7_S_alpha_train.jsonl`400训练父、`v7_S_alpha_dev.jsonl`100开发父，和完整版相应400/100前缀逐字节一致，输入500/500及10屋站位已过；每屋先完整预检1父再放余49。50个10父分片、125四成员/375单成员，计划1750窗/24500题，绑定≥5000=20.408%。`production_plan.json`依据40个完整画面世界/20父实测预算154.398核时，含50%余量：48g64并行3.619小时、CCI128并行1.809小时，均未实测。CCI裸RGB106秒来自owner，不替代完整验收。AFS `/mnt/afs/xinyuanqian/jzy/hardbank/v7_S_alpha_20261004/`，单父单写者/阶段指针/精确RIR缓存支持4小时重启续跑；平台ABI/路径迁移/AFS仍需该线实测。使用`--render-backend cpu_llvmpipe`及仓库接口文档Mesa/CPU EGL环境，无GPU回退。

仅在48g指定worktree改测：合入families `ff81747`→`a39ba8f`，新生成/渲染/RIR/采样/交付工具、QA14题面及测试；正式产物均在本轮新目录，没有覆盖旧轮产物、删除或push。代码冻结提交 **44235e58eed56ebc7161738561e96be574447fb9**；最终文档提交及完成时间（新加坡）见同目录`HANDOFF.json`。cci-datagen/测试面板转交信息在HANDOFF，由owner转。
