两份单 rank 非独占计划和 AFS 并发入口已通过 CCI CPU 检查；12 房几何 fail 是原包已有的拓扑检查结果，P19 渲染、完整世界速度及 v7/720 组总工时未验证，2 TB 中生成媒体与中间产物的当前预算为约 392.09 GB。

本报告为第 2 回合交付，取代本分支上一轮的 exclusive、固定 32 GiB scratch 和“缺 64 请求”结论。`B=/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_colocated_20261003_codex`，`B0=/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex`；48g 的 `W=/data/jzy/code/AVEngine-cci-colocated-20261003-codex`、`T=/data/jzy/tmp/claude_hardbank_20260923/cci_colocated_20261003_codex`。开发机实测主机为 `app-a0065b3376244d93b00935073c28e751-74c766dd88-7kmmh`，48g 为 `cw-SYS-4029GP-TRT3`。

代码只在 48g 独立 worktree/分支 `codex/cci-colocated-20261003` 修改、测试和提交，基础提交 `a8c36486d9685d9d273ce57b6abf6dcf7b8e9297`。最终提交与同步包 SHA 由 `B/RELEASE.json` 和最终回复给出，未 push。未修改共享运行器、worker、队列、已有 runs 或其他 Codex 检出；未启动 48g GPU 作业。CCI 仅在个人 `/mnt/afs/xinyuanqian/jzy/` 内写入。已按 workspace 指引读 Hardbank level2 skill 和 dated memory，并以本轮 owner 授权与 48g 当前房屋验收资料为准。

当前新计划在 CCI `hardbank/plans/` 下，名称为 `cci_datagen_probe_shared_20261003_codex_r2.json` 与 `cci_datagen_pilot64_shared_20261003_codex_r2.json`；源文件在 `W/tools/cci_datagen/plans/`。两份均为现有 `avledger_ladder_plan_v1` 普通计划，只有一个实验，由已有 torchrun worker 认领一个 rank。它们不重新 torchrun，不要求训练排空，不接管训练，不单独预留一张整卡。旧 exclusive 计划未改，本轮应使用下面两行。

先由 owner 加探测行，确认资源边界后再加试跑行；本轮没有写入 `p19_queue.txt`：

```text
/mnt/afs/xinyuanqian/jzy/hardbank/plans/cci_datagen_probe_shared_20261003_codex_r2.json 2
/mnt/afs/xinyuanqian/jzy/hardbank/plans/cci_datagen_pilot64_shared_20261003_codex_r2.json 2
```

尾部 `2` 是每卡现有 worker 数，不是生成占两个 rank；不要加 `exclusive` 或 `normal` 标签。若当前所有 rank 都在训练，需等一个 rank 空闲，其余训练继续；将生成计划放在长训练计划前，可让后续空闲 worker 优先看到它。探测完成后再排试跑，避免短时有两个生成实验同时认领两个 rank。

通用作业借用未修改 worker 的 eval-only 回执接口；其中 `checkpoint` 实际是 `--generator-profile` JSON，没有模型 checkpoint。`count=1` 表示一个完成作业，`completed_worlds` 独立记录：探测 0，只有完整交付 64 世界才为 64。没有伪造模型预测或训练结果。

资源由运行时读取；owner 告知的 P19 4×80G、88 CPU、920 GB 尚未实测。每 rank 快照来自同节点只读 `/proc` 观察，标明 observer、被观察 PID、CPU affinity、cgroup 配额和 rank 环境，不向训练 rank 注入命令。远端节点和不可见 rank 明确记未观察；本节点未观察到的潜在训练 rank 仍保守预留。

| 项目 | CCI 开发机实测 | 出处/限制 |
|---|---|---|
| CPU | Intel Xeon Platinum 8468 | `B/evidence/dev_ordinary_ef17d9d9/.../observer.json`，下文列完整相对目录 |
| nproc | OMP=1 时 1；unrestricted 192；affinity 0–191 | 不把 192 当可用核数 |
| cpu.max / 有效核 | `200000 100000` / 2.0 | 同节点两 worker 共享限额 |
| 内存上限 | 34,359,738,368 字节 = 32 GiB | cgroup memory.max |
| /dev/shm | 17,179,869,184 字节 = 16 GiB | 计入同一内存限制 |
| 本地盘 | statvfs 可见约 1.074 TB | 不是获批生成存储，实际中间产物写 AFS |
| AFS 2 MiB IO | 写/fsync 135.400 MB/s，热读 192.042 MB/s | 小块缓存测量，不能当持续 IO 吞吐 |
| GPU | 可使用数 0 | NVML 列出宿主 H100 80GB，但显存字段为 Insufficient Permissions；设备访问检查不通过 |
| 普通 worker 联调 | WORLD_SIZE=2、LOCAL_WORLD_SIZE=2、节点 1；生成认领 1 rank；额外 torchrun 0 | `B/evidence/dev_ordinary_ef17d9d9/SUMMARY.json` |
| CPU 探测阶段 | 约 2 秒 | 同目录 torchrun.log，17:27:41–17:27:43 UTC；12 秒 peer fixture 是另一个 rank 的独立 CPU 作业 |
| EGL 真渲染 / 渲染显存峰值 | 未验证 | 本轮开发机无可用 GPU；P19 尚未排队 |

上表普通 worker 探测相对目录为 `B/evidence/dev_ordinary_ef17d9d9/scheduler_output/datagen_probe/eval_job_a01_dev_ordinary_ef17d9d9/`，其中 `observer.json`、`scheduler_lease.json`、`resources/rank*.json`、`probe_summary.json` 均保留。最终代码的部署核验另记 `B/evidence/final_delivery_verification.json`，若追加联调，其路径由该核验记录引用。

最终单进程 CPU 探测记录 memory.current=32,371,990,528 字节、limit=34,359,738,368 字节，可回收的干净 inactive_file=24,978,006,016 字节。旧的“limit-current”只有约 1.99 GB，当前结合 MemAvailable 和干净文件缓存后可用内存上界为 26,965,753,856 字节。只计干净 inactive_file，不计脏页、回写或 active_file；没有清缓存、flush 或修改共享内存设置。出处 `B/evidence/developer_probe_final_r2/observer.json`，与同目录 resource_assignment.json 一起保存。

共享 CPU 预算先扣其他本节点 rank 的训练加载器，再留 10% CPU 余量；加载器取 profile 默认 4 与实际 `pt_data_worker` 数的较大值。训练主线程/调度开销包含在 CPU 余量。内存按每训练 rank 33 GiB、每生成进程 4 GiB、10% 内存预留及运行时 available/cgroup 上界约束；均是可调整的保守参数。私有 cgroup 不重复扣其他 rank 的私有额度。当前测试夹具用 88 核、8 local rank、内存充足演示：`88−7×4−ceil(88×0.1)=51` 个进程，其中渲染最多一个，与 CPU 子进程共用 51 的总数。更多 loader、较小配额或内存不足会降低数量；这不是 P19 实测。若确实只有一个本节点 rank，按同一规则可得 79 个 CPU 进程；当前共享计划不假定其他 rank 空闲。

所有生成 CPU 子进程 nice 19、限定在该 rank 可用 affinity 中。渲染只用父 worker 分配给本实验的物理 GPU，不使用其他卡。当前 profile 的 GPU 边界为渲染最多 12 GiB、同卡训练保留至少 4 GiB 空闲：创建 EGL 前至少有 16 GiB 空闲，每秒记录自身渲染进程树显存和整卡空闲；查询失败、自身显存不可观测或越界，只停止自己的渲染进程组。显存采样保存在耐久 logs，并把峰值摘要写入阶段回执，完成后清 scratch 不会丢失测量证据。完整渲染若未观测到自身 GPU 进程显存，阶段拒收并保留诊断；64×64 快速预检允许显存样本未观测，不能把其峰值作为完整世界测量。真实峰值、同卡训练增长及训练速度影响未验证。该保护不是硬件显存分区，可能漏掉一秒内的峰值；nice 19 也不隔离 AFS 和内存带宽。探测不创建 GPU 上下文，64 世界试跑开头在获分配设备上做 64×64 真 EGL 冒烟，失败即保存 stderr 退出，不改系统库。

12 房几何问题已经定位。48g 第一批 `v7_batch1_20261003/report_final/recommended_houses.json` 与修正版 `v7_batch2_test_20261003/report_final/all_house_results.json` 的原始 12 房声学验收均通过；第二批原 00808 的中间结果曾为 review，最终索引按修正的 first-arrival 计算归为 pass，不能拿中间 review 当 CCI 失败。原 12 房的顶点、三角形和材质 ID 数组在第一批/修正版逐文件 SHA 一致，CCI 也通过 SHA 核验。

与此同时，48g 原编译包中的 geometry_report 本来就是 fail，编译包 ray_leakage_report 为 not_run（空 checks）。CCI 对原数组运行同版 geometry_report，12/12 输出与原报告完全一致。`qa.py` 的 pass 条件要求退化/重复三角形、每对象非流形边以及每对象边界边达到严格阈值；当前这些 scan 网格与按语义类别分开的表面具有大量开口/边界，尤其每对象边界零的要求不满足。全局边界也存在，不能把所有边界都解释为人为语义接缝。声学抽样泄漏≤5% 的门槛和“网格严格闭合”的门槛是不同条件。

随后在 CCI 用相同位置、同样每检查 512 个方向、原生 CPU BVH/RLR 上下文、thread_count=1 重跑 ray leakage。12/12 逐检查逃逸方向索引、逃逸计数和 near-surface 命中记录与 48g 保存结果一致，共 49,664 条 ray，最大泄漏 3.125%。没有重写 QA 为 pass、修补网格或放宽正式准入。这里排除了本次诊断所用声学数组缺失、传输损坏或环境数值差异引入的新几何/ray 失败；Habitat 真实画面和修复后的材质表现仍未验证。逐条命中距离不属于本次保存对拍字段，不能声称全部 hit distance 位同。

| 房屋（推荐 split） | 全局边界边 | 每对象边界边 | ray 数 | 最大泄漏 |
|---|---:|---:|---:|---:|
| train 00059 kJxT5qssH4H | 9383 | 46737 | 3072 | 0.78125% |
| train 00096 6HRFAUDqpTb | 14475 | 55027 | 4608 | 2.734375% |
| train 00155 iLDo95ZbDJq | 6959 | 55013 | 4608 | 0% |
| train 00168 bHKTDQFJxTw | 21826 | 80798 | 3072 | 0.5859375% |
| train 00272 kA2nG18hCAr | 14186 | 88229 | 4608 | 0% |
| train 00382 S7uMvxjBVZq | 6464 | 47079 | 4608 | 0.5859375% |
| train 00712 HZ2iMMBsBQ9 | 6925 | 47913 | 4608 | 3.125% |
| train 00746 RTV2n6fXB2w | 2283 | 66051 | 4608 | 0% |
| dev 00880 Nfvxx8J5NCo | 4591 | 52615 | 2048 | 0.5859375% |
| dev 00862 LT9Jq6dN3Ea | 15932 | 75938 | 4608 | 0% |
| test 00808 y9hTuugGdiq | 15200 | 69907 | 4608 | 0.1953125% |
| test 00877 4ok3usBNeis | 7963 | 58860 | 4608 | 0.78125% |

数字出处 `B/evidence/qa_compact.json`、完整 `B/evidence/qa_cci/SUMMARY.json` 与 `B/qa_reference/source_audit.json`（SHA `b2a8edf8da25ee0a179030e99fc1cf22760d43a12d20857b658d416c1650a03d`）。几何计算和 native setup/ray 耗时之和 140.053 秒，不含全部 Python 启动，不是完整世界速度。命令为 `bash B/scripts/run_qa_parity.sh`，日志 `B/evidence/qa_cci.log`。测试房仅做只读几何/ray 诊断；FOA、视频、题目和正式测试成员均未生成。

存储以十进制 2 TB=2,000,000,000,000 字节做预算。CCI 新执行 `nice -n 19 timeout 300 du -sB1 /mnt/afs/xinyuanqian/jzy`，得到 1,483,910,266,880 字节，剩 516,089,733,120 字节。出处 `B/evidence/du_total_bytes.txt`、du_started/finished 时间文件；旧顶层统计仍只读，不依赖其是否写完。本轮没有删其他数据、checkpoint 或缓存。

| 2 TB 分配 | GB（十进制） | 性质 |
|---|---:|---|
| 当前已用 | 1483.910267 | du 实测，包含当时已落盘环境/输入 |
| 训练 checkpoint 继续增长 | 64 | 预留，不删除旧 checkpoint |
| 冻结面板及封存材料 | 32 | 暂定，协议/产物大小未实测 |
| 临时验证小数据集 | 16 | 预留 |
| 环境/代码/输入继续增长 | 12 | 保守预留 |
| 生成池：交付媒体+活动中间产物+失败遗留 | **392.089733** | 共用动态额度，无固定 32 GiB scratch 上限 |
| 总额 | 2000 | 所有预算均在 owner 总上限内 |

生成工作区是 `B/output/generation_workspace/`，scratch 与交付同在 AFS。启动和每 600 秒重新 du owner 目录，观察到检查点等增长时自动收紧生成池；每 10 秒检查工作区实际/预留字节。只有交付媒体、问题、证书及 SHA 回读通过后，才清本世界自己完成的 scratch。IR/raw npy/npz 不进入耐久媒体索引，失败遗留计入预算，不自动删除未完成产物。空间保护是软件采样，不能替代 AFS 硬配额。

按当前未实测的 4 GiB scratch+128 MiB 交付上界，51 个活动世界总预留为 225,888,436,224 字节（225.89 GB），64 个为 283,467,841,536 字节（283.47 GB），均可容纳；由此解除旧的固定 8 槽位限制。这只是空间/资源上界，不是已测稳态 CPU 利用率。单 GPU 渲染供料可能成为瓶颈，要利用更多声学 worker，可在 CCI 先积累预渲染 capture，再使用 CPU-only handoff，或接入交付后的视觉复用/反事实生成器。无匹配 capture 的 CPU-only 任务会提前拒绝。

如果保留满 380 GB v7 媒体，生成池只剩 12,089,733,120 字节；按上述保守上界只能再同时容纳两个活动世界。因此本轮方案足够 64 世界试跑，**不能承诺完整 v7 后期仍有 51 世界并发**。正式生产需从试跑记录实测 scratch 峰值，接入上游中间产物优化，并根据测试面板最终大小/检查点实际增长重新预算；不能靠提高并发数越过 2 TB。精确计算与假设标志见 `B/evidence/storage_budget.json`。

环境继续复用第一轮从 48g conda-pack 迁移的私有 Python 3.12.13 + native/overlay，未修改公共 conda/驱动/系统服务。Habitat 0.3.3，NumPy 2.3.5，numpy-quaternion 2024.0.13，soundfile 0.13.1，trimesh 4.8.3；完整版本清单 `B0/evidence/cci_python_packages.txt`。native/包核验见 `B0/evidence/environment_sha256_verification.txt`、`B0/evidence/environment_relative.sha256`。

| 环境包 | 字节 | SHA-256 |
|---|---:|---|
| env.tar.gz | 468422530 | `3cb312d284f962d7e970c2634d9831bfaee78612523dab07116945e619742b59` |
| native.tar.gz | 11503818 | `70f3c59967d67661ef7e0e1938f83df9d7274a45fa7a0d1dfbb4b881a55f0dd0` |
| overlay.tar.gz | 18248416 | `add170f35124fd583281406a0ff722dd6249856b6c5303b15a1d48d5197922a8` |

GLVND 三个已有 DSO 只位于 `B0/private_glvnd`，仅本作业 LD_LIBRARY_PATH 使用；第一轮环境报告保留逐库 SHA，未再改系统库。GPU EGL 未验证。

原 12 房及生成链共享素材已在 B0/mirror，3752 文件、3,837,650,365 字节 SHA 全通过；含 1985 WAV 文件、58 个 Habitat runtime 素材包、navmesh/semantic/编译包/材质与 QA 表。源清单 `B0/mirror/data/jzy/tmp/claude_hardbank_20260923/cci_datagen_20261003_codex/evidence/input_manifest.json` 的 SHA 为 `e1ade4073e77ca76cb4004d5f8a0f892cec011b3cdb198493b61a611cac66092`，核验 `B0/evidence/input_sha256_verification.json`。4 份路径映射配置在 `B0/configs/runtime_mapped_v2/`，逐输入/有效副本 SHA 见 PATH_MAPPING.json；7 个历史 UE import 引用未传，此 v6 Habitat 路线不使用。材质修复完成后需单独新版本素材清单，不能把旧 registry 当修复已交付。

本轮新增迁移包都经指定 `cci_pull_from_48g.sh` 由 CCI 拉取并先校验 SHA：

| 内容 | 字节 | SHA-256 |
|---|---:|---|
| 原 12 房只读 QA/修正版结果快照 | 6882970 | `45c8ff44d6027a20a633abe3bb7f91453dd8d0b424295ef257209b3a6a4dd46c` |
| 旧 v6 请求与 simulation 输入 | 4438 | `3cbafa123b0a48f4b5469db5ed37dffe5647fbb95e759c0c108231520410fd64` |
| 48g 三个 CPU 规划参考 | 121985 | `24308e334c148c14e3e1e1602207982d9106a7bda7c308461e39a8544e045334` |

最终代码包、真实 Git 对象包和逐文件 SHA 见 B/RELEASE.json、B/git_snapshot/source_git_manifest.json 及最终部署核验。Git 对象来自 48g 的真实提交，不在 CCI 伪造 commit。补齐了旧代码 archive 缺 .git 导致 `git rev-parse HEAD` 失败的问题；CCI root 与 AFS UID 不同造成的 Git ownership 拒绝，通过私有 `.git/CCI_SAFE_CONFIG` 和仅本作业的 GIT_CONFIG_GLOBAL 解决，没有写全局 Git 配置。早期失败日志保留 `B/evidence/pilot_plan_cci_four/`、`B/evidence/pilot_plan_cci_three_r2/`，它们是启动失败，不是世界/资产验收失败。

已准备 64 个既有 v6 QA06 moving 10 秒配方、8 套训练房×8 seeds；不含开发/测试房。输入模板 SHA `ce23602af523c8a96c16857ad5a1d0ecf2db21509ca10746dfb1d047b7122dce`，simulation 源 SHA `e0d808bef58965dbd8bac2d089bcdf2f89c3c58f84d17a87efe30c081b7aed0d`；完整请求 SHA 为 `B/configs/PILOT_REQUESTS_PROVENANCE.json`。沿用旧模板已有的 `allow_research_candidate_assets=true`，只供研究候选流程试跑；编译包几何 fail 未改，正式准入仍不允许把这些试跑当合格生产数据。floor 由 CPU Habitat PathFinder/navmesh 测得，未创建 renderer。

| CPU plan-only 请求 | 48g 秒 | CCI 秒 | 结果 |
|---|---:|---:|---|
| index 0 / train 00059 | 34.534 | 46.661 | 两边返回 0 |
| index 1 / train 00096 | 76.636 | 81.783 | 两边返回 0 |
| index 2 / train 00155 | 78.648 | 84.714 | 两边返回 0 |
| index 3 / train 00168 | 120.020 | 未运行 | 48g 达本轮 120 秒预算而终止，未判为无解 |

来源 `T/evidence/pilot_plan48_four/SUMMARY.json`、`B/evidence/pilot_plan_cci_three_r2_a3/SUMMARY.json`。对拍 `B/evidence/cpu_plan_comparison.json`：三个规划的差异均为路径引用，其余字段一致，camera/visual_plan、实体、采样结果和条件数值等一致；原完整 JSON SHA 0/3 一致，未伪装成位同。历史 measurement_v2/measurement_final、navmesh 版本目录和有效配置副本路径差异均留下值与字段清单。其余 61 请求尚未原生规划验收，planning_validated/rendering_validated 保持 false。64 请求清单已经具备，但不是 64 个已验收完整世界。

第一轮两房校准的 FOA 样本 SHA 2/2 一致，原始 WAV 文件 SHA 0/2 一致（PEAK 创建时间戳）；见 `B0/evidence/cpu_cross_machine_comparison.json`。CCI 校准耗时 17.478/17.929 秒仅是固定解析干声+单线程 RLR/卷积，不是完整生成速度。本轮 native 完整世界数 0，FOA/视频帧/题目完整对拍未运行，P19 的 EGL、VRAM、稳态吞吐和训练速度变化均未验证。

v7 第一批 1500–2300 worker 小时来自首轮任务书，旧参考 `v7_scoping_20261002/throughput.json` 中 HM3D capture=69.365 秒、finalize=161.043 秒、total=230.834 秒，含多种原生线程配置，不能直接除以 51 当 P19 ETA。若仅沿用旧 10 秒 HM3D 画面时长且只有一个渲染 lane，64 世界画面时间演算约 1.23 小时；720 组×2 画面约 27.75 小时只是视觉阶段的条件演算。正式面板时长/布局未冻结、生成器将换版，训练同卡争用和单线程声学倍率也未测，所以 v7 完整版与 720 组在 P19 的总小时数均为**未验证**。不再引用旧 exclusive/固定 8 槽位的总工时表。

生成器版本由 `B/configs/generator_v6_pilot_r2.json` 选择 code root、release、stage adapter、请求和运行目录；profile/release/request SHA 绑定 manifest 与回执，换版必须用新 profile/manifest/run root。正式生产等 `20261003-134653_complex_v7-spec-acoustics` 与 `20261003-224508_complex_asset-materials` 交付，再在 48g 打包同步新的私有版本并按新 API 做适配/冒烟；本轮未修改两条线检出，未宣称其正式验收完成。

正式冻结测试生产当前明确拒绝。修正版 recommended_houses_v2 已含 12 套测试房（原两套+补充十套），本轮只诊断原 12 房，未传/生成额外测试成员。协议冻结、面板代码交接及独立访问主体/AFS ACL 到位后，才能在 CCI 生成正式 720 组，并把正确答案、真值人物 ID、组映射单独封存、留 SHA 承诺，使训练/选模型流程读不到。同 UID 的 chmod 或环境变量不构成该隔离；本轮没有正式面板或秘密答案。

代码检查：40 项 CPU 测试全部通过，48g 18.597 秒、CCI 37.823 秒，来源 `T/evidence/tests48_final.txt` 与 `B/evidence/tests_cci_final.txt`。覆盖 peer 在快照期间结束的竞态、共享/私有 cgroup、运行时 rank 数、训练 loader/内存预算、干净缓存估算、AFS 51 槽位及后期额度、完成后清自身 scratch、真实子进程 SIGTERM/续跑、profile/manifest 绑定、CPU-only handoff、单卡查询和只停自身 renderer 的模拟测试。GPU 单元测试使用模拟 NVML 与自己启动的 CPU dummy，不启动 GPU；额外验证显存证据在 scratch 清理后仍保留、未观测到自身显存时不伪报零值。最终普通 worker 联调为 `B/evidence/dev_ordinary_e4bbd928/SUMMARY.json`，同样只认领一个生成 rank、peer 独立完成。普通 worker CPU 联调用未修改的真实 worker，生成认领 1 rank，peer 独立完成；没有启动真实训练。

48g 本轮主要运行 `python T/scripts/snapshot_qa.py` 快照命令、`qa_parity.py` 的首房 native 诊断、`prepare_v6_pilot_requests.py`、`run_qa_episode.py --plan-only` 和 `python -m unittest discover`；实际脚本/命令记录在 T/scripts 与 evidence。CCI 运行指定拉取脚本、私有环境导入、`bash B/scripts/run_qa_parity.sh`、`du -sB1`、真实普通 worker CPU 联调、三个 `run_qa_episode.py --plan-only`、CPU 测试和部署核验；都无 CUDA/EGL 上下文。代码改动集中在 W/tools/cci_datagen：新增 normal_entry/colocated_resources/space_budget/gpu_guard/git_snapshot/qa_parity/prepare_v6_pilot_requests/普通 worker 测试，两份 r2 计划与 profile，修改 generate/resources/make_pilot/world_stage/native_preflight 及 README/报告。

后续 P19 日志：探测落 `B/output/scheduler_probe/datagen_probe/eval_job_*/`，试跑落 `B/output/scheduler_pilot64/datagen_pilot64/eval_job_*/`；实际阶段、进程记录、哈希与续跑回执在 `B/output/generation_workspace/pilot64_v6_r2/`。同版本续跑沿用相同生成 manifest/run root，用新的 scheduler 尝试；已有完整输出哈希回读后跳过。若 worker 已把失败计入次数上限，需创建新名字/新 scheduler run_root 的普通 retry 计划，保留原生成 run_root；不要改正在跑的计划或版本 profile。

停作业先核对自己的 process.json 和实时 PID/argv，再 SIGTERM 自己的 normal_entry/generate 父进程，由其停止自己创建的子进程组；不要给共享 torchrun、worker 或训练进程发信号。交付时没有本轮持续后台生成/GPU/训练/传输作业，没有 P19 作业号；两份计划等待 owner 排队。
