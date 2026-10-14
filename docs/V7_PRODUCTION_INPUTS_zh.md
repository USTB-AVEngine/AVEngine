# v7 量产准备与今晚验收（2026-10-14，48g）

当前已合并全部指定修复并接入最新合法清单，但完整动态、多成员及按需求摆放尚未全部验证。是否开始正式量产由 owner 决定；本任务不 push、不并入训练集、不传 CCI、不启动 CCI/ACP。

本检出：`/data/jzy/code/AVEngine-v7-production-ready-20261014`，分支 `codex/v7-production-ready-20261014`。产物根：`/data/jzy/tmp/claude_hardbank_20260923/v7_production_ready_20261014`（下面以 `T` 指代）。别的检出、原始登记、资产及 checkpoint 均只读。

输入由 `T/inputs/production_contract_final.json` 绑定。HM3D/MP3D 最新 v2、酷家乐清单按房间读取，只采样明确可用类别，默认排除 budget_limited；12 套测试房子排除。`T/USABLE_INPUTS_v7_20261014.json` 保留每个数字的原文件：训练 876 间/179 套，其中 HM3D 684/125、MP3D 159/35、酷家乐 33/19。默认可采样可见资产 377 项，鸟 17 只；狗猫实际 63 只为 46 动画/17 滑行，与原预计 57/6 不同。新准入 `T/inputs/motion_admissions_combined_production_20261014.json` 不改旧 owner093x_final。音频 v13 为 26 类/2124 条，台词 24 音色/1597 条英文；动作声和类内来源数详见汇总。上衣从五种非红色采样，场景内受控颜色不重复；空调默认只作可选静音摆件。

正式 v7-S 配方以 75% 单成员、25% 四成员，每成员两窗，每窗十四题安排，平均每父场 3.5 个交付窗/49 题。v1 实际交换前两个已准入人物的外观，a1 实际交换两路完整干声的发声端关联，其他几何/相机/时刻保持。当前 HM3D/MP3D 的 phase2 路径仍只有 v0_a0，原两窗约 28 题；酷家乐 registered 静态四成员约 112 题。不能只把 members 改成四项就宣称动态多成员接通。今晚新增六窗/二十一窗是同一三十秒父场的重叠裁窗，仅用于试生产验收，不能用于夸大独立场景覆盖。代码与旧实测成本出处：`T/receipts/V7S_MULTI_MEMBER_RECIPE_ANALYSIS_20261014.json`。

题目层保留原合法候选、真值与判分，按原十四题及模态配额选择；owner 允许不足题族跳过或合法池回填，原 requested/actual 及未改答案的审计随题保留。场景层的 scene_quotas 尚未在当前原生 builder 兑现：HM/MP 当前 seal row 无该字段，酷家乐虽有原请求但 builder 实际仍读 production_scenario。QA-13/16/17 缺事件停止与真实尾部结束后的查询窗；QA-15 缺发声期间真实距离趋势；QA-18 缺实际 wet-tail 测量接线；QA-23 缺非语音事件分段审核，原 audit 报告计数不能当最终合法 authority 候选。不得用计划时刻假造实际测量。改动方案及按题族需求见 `T/receipts/SCENE_QUOTAS_DEMAND_FLOW_ANALYSIS_20261014.json` 和量产计划。

今晚 owner 明确允许元素级降级（换外观/位置、去掉问题物件、坐姿转有标记的站姿、动物短路线后滑行/静止），原生轨迹与标记必须照实保留。质量授权及逐项原值→新值/文件行号在 `T/receipts/OWNER_QUALITY_CHANGES.json`；材质未测和高光/外观检查由 errors→quality_warnings，仅记录（`v7_visible_generation.py`、`v7_phase2_generation.py`），见 `OWNER_MATERIAL_RECORD_ONLY_20261014.json`。答案、真值、判分、真实声源位置与发声、声画一致、贴附/碰撞、测试隔离、英文要求保持。

今晚入口 shim 的顶层启动曾导致 multiprocessing 子进程异常和管道死锁。任务脚本已加 `__main__` 保护，并做真实 spawn 子进程 probe；仓库 `tools/timeline/frozen_trial_cli_20261014.py` 留存同一源码。owner 授权后逐 PID/start_ticks/command 核对停止 31 个自有进程树，重排 33 个新编号，先完整原生冒烟再放批量。收据：`T/receipts/ENTRY_DEADLOCK_STOP_AUDIT_20261015.json`、`FROZEN_ENTRY_SPAWN_FIX_20261014.json`。基础设施失败与内容失败分开计数。PSS 曾超过 140GB，见 `MEMORY_BUDGET_INCIDENT_20261014.json`，不伪称全程合规。

当前恢复守卫在 `T/scripts/pilot_controller_1500_v8.py`：nice 10、单线程，CPU≤88 进程、PSS≤140GB、可用内存≥30GB才起新任务；每父场全生命周期预留 8GB，预测≤130GB，最多 16 个 CPU 父场。资源不足暂停新增；日志与进程树 CPU 同时十分钟不增长才精确停自有卡死树。新增磁盘≤80GB、临时 /data 停止线 100GB，到 2026-10-15 01:30 新加坡时间到期。失败中间文件清理脚本默认演练，root 不执行删除；通过文件/日志/失败记录/manifest/收据全部保留。

验收页 `T/pilot_20261014/ACCEPTANCE_1500_zh.html` 和 `ACCEPTANCE_STATS_1500.json` 每次按实际通过封印、生产格式导出及 reader 全量读回更新，helpers 原字节单列，不伪称由当前 HEAD 生成。最终数量、题族/看听分布、失败原因、行走相关题、动态覆盖、逐场耗时大小以最新文件为准。23:40 版单独保留；批量到累计 1500 道或 01:30 截止。

`T/PRODUCTION_PLAN_v7_20261014.json` 按训练房分配 6800 父场；75/25 及完整窗口全部成功时条件题量 333200，当前简化实测、失败成本、重叠裁窗分层记录。新动态正式混合配方的完整 CCI 用时和大小未验证。`T/cci_pull_list_v7_20261014.txt` 及 `CCI_SYNC_MANIFEST_v7_20261014.json` 仅为按已有拉取格式准备的代码/资产/音频/清单闭包；不含试生产数据，不执行传输。
