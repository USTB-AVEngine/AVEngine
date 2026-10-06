# 第20回合：P19 跨 rank 状态和重启核验

r14 并非没有启动。2026-10-06 12:31:11 SGT，旧作业的 rank5 领取唯一实验 `v7_s_remaining_r19_space_r14` 并启动 eval_job。rank0 于12:36:22加入，12:36:27报告 `all_runnable_claimed`：实验已被兄弟 worker 领取。这是该 worker 的退出，不是整个计划失败。12:48的真实 PROGRESS 为91父场景、242窗、3388题、40 active、203 pending，已有封存由61增至91。

计划名、实验名和 run_root 都与 r13 不同。固定 P19 worker 的 experiment_state 实测 failures={}、blocked=[]、runnable=true；不存在继承 r13 失败次数。队列的 `2` 是 workers_per_gpu，两 worker/卡构成8位置池，不要求一个计划同时取得2位置。一个生成实验只占1位置，协调器最多派40个CPU子进程，清空CUDA，使用llvmpipe；训练降速保护仍为35%。

Owner在12:49重启了P19。新作业 `job_20261006T044917Z_b69cc9-l8xl9` 的 rank4 于12:50:33实际领取同一实验，协调器PID105于12:51:01取得锁。12:55:41实测34 active、40上限、209 pending、91封存/242窗/3388题；缓存盘额到期触发后台新扫描，扫描期间暂停新增派发，已在运行的父场景继续。该检查时段不能冒称已维持40 active，后续见AFS最新状态。

新增只读工具 `tools/timeline/inspect_v7_p19_progress.py --plan <plan.json>`：读取当前租约job下全部rank日志、claim、协调器身份和新鲜PROGRESS，并列出领取者与正常退出的兄弟rank。旧job的历史active记录不能证明新job在跑；超过120秒的PROGRESS只报告未确认状态。工具不改队列、claim、锁、控制参数，也不读取房屋或媒体。所有路径均取自计划/配置的精确字段，禁止全局替换路径。

源码只在48g改测。7项针对兄弟退出/陈旧进度/重启归属/精确路径的测试加41项既有断点/盘额/磁盘守卫检查，共48项通过，1.20s；工具索引及diff检查通过。固定worker SHA `6aa2c7f0065d7dc49676f7d666866ee2eef2911bb442586932ac6e4ac64f6ced` 在48g独立目录真实运行 claim_plan、PoolWorker.next_experiment 和 Worker.claim：旧job claim不会挡新job，新job第二个rank不能重复领取。未启动父场景或GPU。AFS实际计划只读 admission 核验也通过；这部分不是一次新的生产作业，重启后的rank4日志才是实际生产证据。

本回合不新排一个重复生产协调器、不使用STOP、不修改p19_run.sh、不重启P19、不删除产物。生成器/协调器生产版继续382cfc1c3ae38d59f7e1e62bdd62123782bb5a44；新提交仅诊断工具、测试和文档。第二批功能/完整版分片不会因本次诊断被冒称已完成，原门禁和配额不改。

证据：本机 round20_p19_claim_20261006/readback/{TRACE.json,ACTUAL_READONLY_ADMISSION.json,LIVE_PROGRESS_2.json,CLAIM_FIXTURE_48_corrected.log,EDIT_TESTS_48_proxy_retry.log}；CCI 对应 generation/v7_s_20261005_batch1/readback/round20_p19_claim_20261006。运行时以 supplement_r16_1/{PROGRESS.json,owner_quota_monitor.json,coordinator.lock/owner.json,p19_run_r14_space/workers/<job>/rank4.jsonl}为准；盘账沿用第19回合 DISK_LEDGER.json，剩余估计会随封存/失败更新，不保证所有种子通过门禁。
