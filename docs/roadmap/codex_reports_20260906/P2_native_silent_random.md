# P2 原生参数、沉默端点与随机化

## 1. 文件与提交

基线 `1b39f9b4fd2b0fd3bae241ab66e7f6f5a94c6726`。报告与实现同提交；本项提交号从 `git log -1 --format=%H -- docs/roadmap/codex_reports_20260906/P2_native_silent_random.md` 读取。

- `src/avengine/rooms/qa_episode.py`：新策略从全部满足原有合法性条件的机位中均匀随机；不作前40截断；repeat 在剩余预算内的合法片段中随机。没有策略字段的旧路径保留原有排序、随机数消耗与最短repeat行为。
- `src/avengine/rooms/native_qa_room.py`：新策略默认static、FOV与silent透传；原生路线对从全体合法组合均匀随机，仍保留原路线样本/时钟/间距条件；不换成栅格路线。
- `tools/studio/run_qa_episode.py`：仅新策略向原生入口透传85度/static/沉默数；旧请求继续原默认。
- `tools/acoustics/render_frame_readback_sequential_speech.py`：`_plan_source_endpoints` 从计划的真实全部实体构造声学候选；无事件实体无伪声音路径，现有dry bus装配自然产出零干声；原“至少两个端点”保护及双槽缓存保持。
- `tests/unit/test_qa_conditioned_transition.py`：新旧策略、完整合法机位、重复事件、真实沉默端点验证。
- `src/avengine/capture/neutral_readback.py` 与 `tests/unit/test_qa_production_contracts.py`：补P1短canary的既有样本取整规则，5帧/15fps的5333采样点有效，5334拒绝；保留计划值，未另算时长。

## 2. 测试

首轮P2相关检查 **33 passed / 0 failed / 0 skipped**（3.39秒），包括原有三源奇数分片与双槽缓存测试。最终连同P1短时钟回归 **52项通过**；同轮另跑P7准备器14项，合计 **66 passed / 0 failed / 0 skipped**（3.72秒）。日志：`tmp/p2_transition_20260906_v1/tests_final.log`。

运行环境：指定runtime Python，`PYTHONPATH=src:tmp/native_python_addons_v1`，BLAS/OMP各1线程。现有soundfile仅通过既有addons目录加载，不安装或改环境。

首次临时音频探针在导入阶段失败，错误原文：`ModuleNotFoundError: No module named 'tools'`。修复探针将权威仓库根加入自身import路径后，以新日志 `single_active_v2.log` 重新运行；首次没有创建音频输出。该失败不是原生RLR失败，原日志保留。

## 3. 实际产物与核对

根目录：`tmp/p2_transition_20260906_v1/`（沿用仓库tmp数据盘链接）。

- `native_85_static/plan/episode_plan.json`：240帧、15fps、16秒；实际计划视场 **85.0度**，相机 **static**，2个实体、1个音频事件。保留原生路线库中 **14305** 个合法路线对作为抽样集合；机位检查175个，合法63个。此项是原生资源上的计划验证，未把plan-only称为新的UE成片。
- `single_active_probe/result.json`、`single_active_probe/single_active/research_report.json`：使用保留A房真实位姿/发声点读回，已实际运行RLR并生成256000采样点、16kHz双耳两声道WAV。
- 单活跃段：source1沉默，`source1_mouth_stem.wav` 的512000个声道采样值全部为0，峰值0；source2非零。仅1个真实事件、1条voice binding，没有虚构发声事件。
- 单活跃混合轨峰值 **-34.6595 dBFS**；同一读回、同seed的双活跃控制峰值 **-32.0965 dBFS**，相差 **-2.5630 dB**。两者事件排程各自生成，此比较用于报告量级，不是同事件时刻的因果消融或人工可答性认证。首段RLR约25.66秒，对照复用既有格式RIR缓存约3.62秒。
- `camera_distribution.json` 与日志：固定同一原生路线输入，仅改变生产函数的camera seed 0--9，得到 **9个不同机位**；每个都在同一个63个合法候选集合里，均完整检查175个候选。为避免重复相同射线，审计进程仅缓存精确原函数的已算射线结果，没有改变几何判断或生产缓存格式。

| seed | selected candidate |
|---:|---|
|0|grid_054_yaw_0_pitch_0|
|1|grid_029_yaw_0_pitch_0|
|2|grid_053_yaw_0_pitch_0|
|3|grid_052_yaw_0_pitch_0|
|4|grid_046_yaw_0_pitch_0|
|5|grid_042_yaw_0_pitch_0|
|6|grid_028_yaw_0_pitch_0|
|7|grid_060_yaw_0_pitch_0|
|8|grid_046_yaw_0_pitch_0|
|9|grid_026_yaw_0_pitch_0|

## 4. 未完成与分类

P2要求的三处修复及探针/计划/10seed产物已完成。完整固定画像联合采样属于P5，P2仍使用现有几何合法性检查；新85度计划的新UE捕获不是本项验收产物，后续先导必须跑真实捕获。工具索引已正常重生成，检查1 passed；本项没有新增工具路径或改工具标题，索引中的并行P4/P7新增条目保留给其各自提交。

原始语音池仅用于重现原单活跃探针，P7最终prepared集尚需人工试听；不能把本段当46段先导之一。

## 5. Claude接口

- 新策略字符串 `conditioned_static_v2`；缺字段继续旧行为，未知策略报错。
- 新机位记录 `selection=uniform_over_legal`、`checked_candidate_count`、`legal_candidate_ids`；原生路线记录 `route_pair_selection`、`legal_route_pair_count`。
- `candidate_source_endpoint_ids` 与生成的AudioProgram包含真实沉默端点；审核不能从event列表推断场景实体总数。
- 上述真实双耳stem、程序、RLR回执与完整计划可直接独立复核。Claude审计器及其测试未改。

## 6. owner事项与边界

无需要owner新增授权的操作。缓存保护、旧请求路径、源数据、Studio服务和分支边界保持；没有push、合并或启动46段生产。短canary的取整修正保留既有时钟语义，不放宽一个完整采样点的漂移。
