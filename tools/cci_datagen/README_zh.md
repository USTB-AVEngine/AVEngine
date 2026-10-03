所有生成产物写 CCI 私有 AFS 目录；此入口负责运行环境、资源分配和续跑，v7 采样器、静止声源 IR 优化及冻结测试协议由原负责人交付后接入。

默认部署根目录是 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex`，可用 `CCI_DATAGEN_BASE` 显式选择另一个获授权的独立目录。完整状态、数字和阻塞见本目录 `REPORT_zh.md`；源提交见部署根目录 `RELEASE.json`。不修改共享队列、运行器或现有 runs。

CPU 探测由 P19 的 exclusive 命令再启动自己的 torchrun：进程数读取可见 GPU 数及 `DATAGEN_RANKS_PER_GPU`（默认沿用现有约定 2），也可明确给 `DATAGEN_NPROC_PER_NODE`。不创建 CUDA/EGL 上下文。每 rank 写主机、CPU/cgroup、内存、文件系统、GPU 元数据及 torchrun 节点/rank 信息，2 MiB AFS 写入后 fsync 并读回。读速是缓存热读。既有 exclusive 调度器在原 torchrun 退场后单实例执行命令；多节点 exclusive 需要额外的每节点启动交接，当前包装器明确拒绝多节点，避免等待不存在的节点。

```bash
B=/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex
bash "$B/code/tools/cci_datagen/run_p19.sh" probe
```

生成前必须提供已验收的请求，所有有效路径指向 CCI。`remap_configs.py` 只在新目录写明确列出的干声/素材运行配置副本，留下原件与副本 SHA；原始数据、编译包 manifest 和 QA 文件不改。其 `PATH_MAPPING.json` 会列出未传的引用资源（包括历史来源或其他 backend 的资产），需按请求选用的 Habitat 路线核对。房屋验收不等于编译包 QA 合格；生产仍使用 v6 原有检查。

64 世界试跑只认 8 套训练房屋，必须有 64 个不同 house/seed 的已验收请求；没有这个交接就提前报错。请求内容 SHA 随 manifest 固定，修改内容要另建 manifest 与运行目录。manifest 同时记录部署 RELEASE.json 中的 Git 提交号，版本不同拒绝混用旧阶段。默认耐久交付上限为十进制 380 GB。

```bash
bash "$B/code/tools/cci_datagen/pilot_entry.sh"
```

这个入口先检查请求清单和 CPU native 环境，再在本任务可见的指定物理 GPU 上做 64×64 EGL 图像冒烟（设备选择见 [Habitat-Sim v0.3.3 源码](https://raw.githubusercontent.com/facebookresearch/habitat-sim/v0.3.3/src/esp/gfx/WindowlessContext.cpp) 与 [Magnum 官方说明](https://doc.magnum.graphics/magnum/classMagnum_1_1Platform_1_1WindowlessEglContext_1_1Configuration.html)）；失败即结束并保存原始 stderr，不改系统驱动/库。随后调度 `plan → render → audio → deliver`。渲染每张选择的卡至多一个进程；RLR 与合成在 CPU 子进程运行，OMP/BLAS/MKL 和原生 simulation.thread_count 都为 1，native 音频 receipt 必须确认线程数。每 rank 的进程数按 affinity、共享/独立 cgroup 配额、可用内存、训练加载器和余量计算，不固定成 8 rank 或 88 核。

默认保留 10% CPU 与内存余量，单 CPU worker 内存估计 4 GiB。混合模式必须明确训练进程数、每进程 loader workers 和每进程训练内存；33 GiB 默认采用 owner 给出的 23–33 GB 范围，按 33 GiB 做更保守预算。生成/预检的 CUDA 与 GL 缓存也写自己的 AFS 目录。所有生成子进程 nice 19，绑在自己的分配 CPU 集中。CPU 配额与每世界存储槽位取共同约束：默认 scratch 32 GiB、每世界先保守声明 4 GiB，只能同时有 8 个世界。每 10 秒检查自己 scratch 和 delivery 实际字节，超限停止自己的作业；这不是文件系统硬配额，采样间可能短暂超限。中断遗留文件仍计入，不清别人的目录。

完成后检查交付媒体、题目、证书与 SHA，写 `ARTIFACTS.json`、`DELIVERY_INDEX.json` 和 `COMPLETE.json`，再清本世界自己的 scratch。交付索引给出耐久媒体路径；原证书中的历史 scratch 路径作为来源保留。重启采用相同 manifest/run-root、新的 launch ID；只有哈希验证通过的阶段/交付才跳过。不同 manifest 拒绝复用旧目录。日志、失败记录、SHA 与续跑状态均在私有 AFS 下。

仅 CPU 的规格使用已经渲好的 capture：以 `world_id` 为键提供 render receipt 映射（含 `request_sha256`、`attempt`、`files` 的完整 SHA），由 `make_pilot.py --pre-rendered-captures` 验证并接入，之后 `run_p19.sh pilot --mode cpu --prune-owned-intermediates`。没有 GPU 时一个 torchrun rank 内可以按运行时 CPU 配额启动多个声学进程。现有共享排队 schema 最少要求 1 GPU，零卡任务直接使用这个入口，不改变共享调度器。

混合任务需要 owner 选定的训练命令 JSON，条目格式为 `{"argv":[...],"cwd":"/mnt/afs/xinyuanqian/jzy/...","gpu_slot":0,"env":{...}}`；gpu_slot 指除预留渲染卡外的卡序号。用既有 torchrun Python 启动 `mixed.py`，不要先 source 生成环境，以免 Python 3.12 包进入训练 Python 3.10 环境。它先验证生成 manifest/EGL，再起明确指定的训练命令，不接管已运行任务。训练入口若自行 torchrun，应明确单卡和自己端口；资源预算中的“训练进程数”必须与实际实验进程数一致。

```bash
/mnt/afs/xinyuanqian/jzy/hardbank/envs/avledger-py310/bin/python \
  "$B/code/tools/cci_datagen/mixed.py" \
  --training-commands "$B/configs/owner_training_commands.json" \
  --generation-manifest "$B/configs/pilot64.json" \
  --run-root "$B/output/mixed_owner_selected_01" --loader-workers 4
```

混合 supervisor 记录每个自己启动的 PID、argv、GPU 和日志；任何子进程失败就停止同一 supervisor 自己启动的其余子进程。停止时先核对 JSON 中 PID 与实时 argv，再对 supervisor 发 SIGTERM，它只清自己的进程组。专用 torchrun 的停止也对自己的父进程发 SIGTERM，由 torchrun/分片入口回收自己的孩子，不使用 pkill。

`cpu_smoke.py` 是两房原生 RLR/卷积校准，允许已有研究包未过 QA；它不渲视频、不出 v6 题目、不计作完整世界。`compare_artifacts.py` 比完整产物时比较 FOA 文件/样本、ffmpeg 解码帧、题目 JSON，任何缺项都会报告 not_run 并失败。WAV 浮点 PEAK 创建时间戳会导致文件 SHA 不同；`wav_metadata.py` 只计算排除此时间戳的比较视图，不修改交付文件，也不替代原始 SHA 承诺。

冻结测试正式生成在 `generate.py` 中尚未启用；必须先拿到协议冻结和原负责人的代码交接。之后公开媒体与答案/真值 actor ID/组映射要由独立访问主体或 AFS ACL 隔离，秘密目录另留 SHA 承诺，不将秘密清单放入训练索引。相同 UID 的进程仅靠 chmod 0700 或环境变量无法实现可靠隔离。本轮没有正式测试数据、秘密答案或测试面板训练输入。
