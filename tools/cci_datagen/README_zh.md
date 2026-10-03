当前入口使用普通计划，由现有 P19 torchrun worker 认领一个生成实验；CPU 子进程 nice 19，画面只用这个 worker 获分配的 GPU。两份 r2 计划均不排空训练，不启动额外 torchrun，也不启动、暂停或监督训练命令。本轮旧的 exclusive 包装器与 mixed.py 留作历史文件，当前两份计划都不调用它们。

部署根 `B=/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_colocated_20261003_codex`。生成私有环境从第一轮 `cci_datagen_20261003_codex/env` 复用；代码和新的运行目录独立。状态、证据、未验证项见 `REPORT_zh.md`，部署提交和包 SHA 见 `$B/RELEASE.json`。

将下列普通队列行由 owner 手动加入队列：先加探测行，检查探测记录后再加试跑行。末尾 `2` 是每卡现有 worker 数，不是生成占两个 rank。不要添加 `exclusive` 或 `normal` 标签。生成计划只有一个实验，因此只占一个 worker；其余 worker 可同时认领训练。若所有 worker 已在训练，需等一个 worker 空闲，其他训练继续；把计划放在长训练计划之前可让下一次认领先看到它。

```text
/mnt/afs/xinyuanqian/jzy/hardbank/plans/cci_datagen_probe_shared_20261003_codex_r2.json 2
/mnt/afs/xinyuanqian/jzy/hardbank/plans/cci_datagen_pilot64_shared_20261003_codex_r2.json 2
```

调度接口为未修改的 `avledger_ladder_plan_v1`。使用已有 eval-only 通用完成回执接口，`checkpoint` 字段传的是 `--generator-profile` JSON，未创建模型 checkpoint。回执 `count=1` 表示一个作业，完整世界数另记 `completed_worlds`；探测为 0，只有 64 个世界交付均完成才记 64。不伪装成模型预测或训练结果。

探测只读同一节点进程的白名单 rank 环境、affinity 和 cgroup，逐 rank 写快照；元数据来自观察者，不会进入其他 rank 执行命令。远端节点或不可见 rank 明确标为未观察，CPU 扣除仍按调度器本节点 rank 数保守估算。记录 CPU 型号、`nproc` 与 unrestricted 值、CPU 配额、内存及干净可回收文件缓存、shm、磁盘、可用 GPU、节点数和 2 MiB AFS 写/fsync/热读。探测不创建 CUDA/EGL 上下文，实际 EGL 验证放在 64 世界试跑的开头。

每 rank CPU 进程数按实际 affinity、共享/私有 cgroup 配额、内存决定。共享 CPU 先扣其余本节点训练 rank 的 loader 数，再留 10% 余量；loader 取配置保守值（当前 4）与观测到的 `pt_data_worker` 数的较大值。主训练线程余量包含在 CPU 预留中。内存按每个训练 rank 33 GiB、每个生成进程 4 GiB 和 10% 预留约束；这些是预算参数，需根据 P19 实测调整。独立 cgroup 不重复扣除其他 rank 的私有预算。无其余训练 rank 时自动得到独立生成的 CPU 预算，不写死 88 核或 8 rank。

GPU 只读 NVML 检查获分配的物理设备。默认给渲染上限 12 GiB、给同卡训练留至少 4 GiB 空闲；首次上下文前需有至少 16 GiB 空闲。每秒观察自己的渲染进程树和整卡空闲；越界或查询失败时只停自己启动的渲染进程组，留下 PID、命令和显存日志。显存日志放在耐久 logs 并在阶段回执记峰值摘要，清 scratch 后仍可复查。完整渲染至少要观测到一次自身显存；未观测时拒收，快速 EGL 预检则明确记未观测。这是软件保护，不是 GPU 硬分区，也不能捕捉一秒内的瞬时峰值。未指定或无法访问 GPU 时不启动 EGL；仅 CPU 的规格必须提供经请求 SHA/文件 SHA 验证的预渲染 capture，才运行声学与合成。

临时文件直接放 `$B/output/generation_workspace/<run>/scratch/<world>/<stage-attempt>`，不用 32 GiB 本地 scratch 限额。整个生成工作区内的交付媒体、失败遗留和中间文件共用剩余配额。启动及每 600 秒以 nice 19 重测 owner 目录 `du -sB1`；先留检查点增长 64 GB、测试面板 32 GB、临时验证 16 GB、部署/输入增长 12 GB，再分配生成池。每活动世界先预留 scratch 峰值和媒体上界，每 10 秒检查实际用量，超界只停止自己的生成。预留不是 AFS 硬配额，采样间的增长需要实测余量。

现有 v6 试跑默认每世界 4 GiB scratch 与 128 MiB 交付上界，未经过 P19 完整世界实测。64 配方只用 8 套训练房（8 种 seed/房），不使用开发/测试房。旧 v6 模板已有的 allow_research_candidate_assets=true 保留，仅供研究候选试跑，原几何 fail 与正式准入保持可见。房屋的声学验收不等于每个新随机请求已经验收；本轮保留准备清单及逐次规划证据。正式 v7 和正式测试生产必须等 v7-spec、asset-materials 两条线交付。

生成阶段为 `plan → render → audio → deliver`，CPU worker 与单个渲染 lane 共用总进程预算。原生 simulation.thread_count、OMP/BLAS/MKL 都设为 1，并检查音频 native receipt 中的线程数。完成后逐文件核对媒体、问题、证书与 SHA，写 `DELIVERY_INDEX.json`、`ARTIFACTS.json`、`COMPLETE.json`，再清本世界自己已完成的 scratch。中断遗留仍计入空间，不自动删未完成或其他作业文件。续跑沿用相同 manifest/run root、新 launch ID；只有回读哈希通过的输出才跳过。

`generator_v6_pilot_r2.json` 把运行基础设施和生成器分开：选择 `generator_code_root`、`generator_release`、`stage_adapter`、请求列表和新 run root。换版本时先在 48g 打包真实提交，CCI 装入新的私有版本目录，再创建新的 profile、manifest 与运行目录。回执绑定 profile 和 release SHA，拒绝在旧运行目录混入新版本。适配器按所选源目录导入生成模块；API 改动需对应适配器交接与冒烟，不能只改版本号就宣称兼容。

`git_snapshot.py` 随代码包运输 48g 的真实 commit/tree/blob 对象，CCI 只装私有浅仓库。AFS UID 与容器 UID 不同，Git 的 safe.directory 配置保存在该私有仓库 `CCI_SAFE_CONFIG`，只通过本作业 `GIT_CONFIG_GLOBAL` 指定；不修改用户或系统的全局 Git 配置。

`qa_parity.py` 为只读诊断，保留原几何 fail，显式允许研究诊断加载未通过几何检查的包，以重放已保存的 native ray 检查。它不修补网格、不放宽正式 QA、不渲视频、不合成 FOA 或题目。12 房几何与 native ray 对拍见报告。第一轮 `cpu_smoke.py` 只做两房原生声学校准；FOA 样本与完整 WAV 文件哈希分别记录。完整世界 FOA、解码视频帧、题目对拍由 `compare_artifacts.py` 执行，缺项必须报未运行。

正式冻结测试入口仍关闭；两套既有测试房本轮只做几何/ray 诊断，没有测试成员媒体。协议冻结与面板代码交接后，答案、真值人物 ID、组映射需独立访问主体/AFS ACL 隔离并留 SHA 承诺，训练索引不得引用秘密目录。同一 UID 进程仅靠 chmod 或环境变量不能保证隔离，本轮未创建正式秘密答案。

停止自己的作业时，先核对 `process.json` 的 PID、主机和实时 `/proc/PID/cmdline`，再给自己的生成父进程发 SIGTERM，由其回收自己的子进程组。不能给共享 torchrun、worker 或训练进程发信号。本轮无持续后台生成作业，未排队、未 push。
