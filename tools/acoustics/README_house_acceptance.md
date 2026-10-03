# 房屋声学验收（CPU）

`accept_houses.py` 在新目录生成检查记录，`summarize_house_acceptance.py`
汇总逐屋结论、整屋划分和全部通过屋的 RT60 图。不是 Episode 生产入口。

## 输入

先用 `prepare --help` 列出必填路径：HM3D人工候选、干净屋清单、测量注册表、
已知历史暴露证据、MP3D演示输入及注册表、v6生产请求/最终音频收据、
两个数据集各自的v6材质规则。路径由显式参数给出，不修改数据集或注册表。

`prepare` 在观测声学之前按候选房间面积分位选14个原train和8个原val；
每屋选择至多3个面积/房型不同且有摆放见证的房间。面积是可用候选房间的和，
不是建筑总面积。最终RLR配置取生产音频收据；请求模板中的diffraction默认值
不能覆盖生产命令的false/0。源/听者半径沿v6使用0。
暴露检查依赖输入清单：M1、匿名SO和物理房屋别名仍必须按规格书审计。

## 运行

在对应的48g新工作区，使用已经存在的CPU依赖环境，不安装共享包：

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/jzy/code/AVEngine-v7-spec-acoustics-20261003/src:/data/jzy/code/AVEngine-v7-spec-acoustics-20261003
export TMPDIR=/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/evidence
TASKPY=/data/jzy/envs/room-selection-merged-20261003/bin/python
TASKOUT=/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003

"$TASKPY" tools/acoustics/accept_houses.py run \
  --manifest "$TASKOUT/acceptance_manifest_v2.json" \
  --output "$TASKOUT/new_measurement" --workers 2

"$TASKPY" tools/acoustics/accept_houses.py visual \
  --manifest "$TASKOUT/acceptance_manifest_v2.json" \
  --output "$TASKOUT/new_measurement" --house HOUSE_ID

"$TASKPY" tools/acoustics/summarize_house_acceptance.py \
  --manifest "$TASKOUT/acceptance_manifest_v2.json" \
  --measurements "$TASKOUT/new_measurement" --report "$TASKOUT/new_report"
```

输出目录必须新建；不要照抄已存在的产物目录重跑。单屋用 `measure`。
`run --house` 可重复指定；`--resume` 跳过已有result.json，而不是改写结果。
每个native子进程的PID/日志在output/logs内。默认2个RLR worker×8线程，
视觉检查使用1线程CPU BVH；没有Habitat视觉Simulator或GPU作业。
超时只终止该工具直接拥有的子进程。需要手工停止时先核对记录PID的
/proc/PID/cmdline确实是本任务，然后只发信号给那个PID；不使用pkill。

`--reuse-root` 或单屋 `--reuse-package` 只读重用本任务自己的编译包，
校验包房屋ID后照常验证原生上传和配置回读。

原生几何拒绝输入时，默认待审。可显式使用
`--omit-degenerate-for-diagnostics` 省略内存中的重复索引/原生float32零叉积面，
保存被省略面索引；保留材质、配置和上传校验，不修改源数据/编译包。
此诊断强制待审，不可作为首批通过证据。正面积面的数值退化若仍被native拒绝，
保持阻断，不猜阈值删面。CPU原始GLB射线图同样只筛零表面面，记录数量。

## 输出与判据

每间监听位和两个声源原点各512条球面射线。最差逃逸≤5%通过、
>15%失败，之间待审；AABB出口和原网格边界只给破洞候选，不填补扫描。

每间近/中/远及第二源生成RIR，单房間屋另补第二听者及第二方向距离样本，
至少2听者、6 RIR。六倍频程T20/T30要求线性拟合R²≥0.95、≥10ms且不贴尾，
至少2个有效中频带、至少70%位置有效。中位RT60主范围0.2–1.5秒，
少量1.5–2.5秒长尾；用20ms前向末端能量检查截尾。
DRR、C50和早反射是描述指标。三点直达能量压力指数p要求0.7–1.3，
遮挡路径只记录RIR，不能参加直达方向/距离验证。

三处FOA使用v6 W×[X,Y,Z]及[forward;-right;up]相机旋转；
固定0.5ms窗口误差≤15°通过，15–30°待审，>30°失败。
CPU原始GLB画面标记≥3像素、投影误差≤3像素。
绘制的是扫描软件图和诊断球，不是人物或正式Habitat RGB帧。

逐屋result.json、rirs.npz、leakage.json、声学/FOA图、三张CPU画面、
visual_checks.json、hole_locators.json及边界npz可一起复核。
硬失败优先；任何缺失/软异常待审；否则诊断通过。已知SO/M1屋另外排除。
汇总报告不把相关RIR位置计成独立房屋，不把反事实成员计成独立窗口。
原生/软件图通过不代表素材许可、人物覆盖、完整生成器或视频生产已完成。

2026-10-03本轮的manifest_v2增加了至少2个听者要求并补测两套单房间屋；
没有改变原定声学阈值。measurement_final以符号链接引用保留的最终版本，
版本选择表同时列出原始结果路径，失败/重试日志保留。


原生RIR或画面阻断时仍可用 `boundary` 子命令读取原始扫描、独立生成边界候选图；
输出目录先放入该屋result.json。它不构造RLR上下文，也不宣称补齐RIR或RGB。
已有边界npz/json可只读重用；图和boundary_review.json不覆盖已有文件。
本轮补充边界图在产物根geometry_review/houses/<house>/，总共28屋。


## 独立测试房补齐与直达测量修正

`build_house_test_panel.py` 保留传入的训练/开发/既有测试整屋分配，按开测前冻结的
候选层和顺序补满测试房，不以RT60、模型分数或worker完成次序改顺序。
未完成的更早候选会阻止后来候选入选；待审、已知暴露或不足6条有效RIR不能凑数。
输出兼容第一批逐屋JSON/CSV及推荐表；每屋60个四元组的种子只是预留，不是已生成媒体。

直达峰现在取独立于几何期望时间的首次合格到达峰；直达能量和DRR沿已经预定的
FOA首次到达0.5ms窗口计算。原来的几何附近最大峰搜索可能错选晚反射，较大的
1.25ms能量窗也可能混入随机间接声。这是测量实现修正，不改变通过阈值或传播设置。
切换版本时必须对全部屋统一重测/重算并保留旧波形、结论和代码身份，不能仅重试
边界待审屋直到通过。本轮记录见`docs/acoustics/V7_TEST_PANEL_BATCH2_20261003_zh.md`。
