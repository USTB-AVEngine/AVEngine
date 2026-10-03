CCI 的 Habitat/RLR CPU 环境及两房声学校准已跑通（17.48、17.93 秒），P19 的 EGL、完整世界速度和两批数据总工时仍未验证；2 TB 预算可预留 380 GB v7 交付、32 GB 测试面板、32 GiB 临时空间，并剩约 66.2 GB 给训练检查点增长与误差。

本报告中的 `B` 是 CCI `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex`；`T` 是 48g `/data/jzy/tmp/claude_hardbank_20260923/cci_datagen_20261003_codex`；`W` 是 48g `/data/jzy/code/AVEngine-cci-datagen-20261003-codex`。CCI 镜像把原 `/data/...` 放在 `B/mirror/data/...`。所有下面的 evidence 路径都属于本轮私有目录。完整提交号见 CCI `B/RELEASE.json` 与最终回复；分支 `codex/cci-datagen-20261003`，基础提交 `c21c387`，不 push。

已按 workspace 指引读 `skills/hardbank-level2/SKILL.md`、2026-09-28 memory，并对照 48g 当前 `v7_batch1_20261003/V7_SPEC_zh.md`、推荐房屋与测量 review。现轮 owner 指定在 CCI 生成，按该新指令执行。本轮新增代码全部在 W 的 `tools/cci_datagen/`；另一条 v7-spec-acoustics 检出、原始数据、已有 checkpoint、共享环境、队列及运行器未改。

实测仅来自 CCI 开发容器；owner 告知的 P19 单任务规格是 4 张 80G 卡、88 CPU、约 920 GB 内存，不作为测量结果。计划尚未排队，所以没有 P19 实测表和 P19 产物。

| 项目 | CCI 开发机实测 | 出处/解释 |
|---|---|---|
| 主机 | `app-a0065b3376244d93b00935073c28e751-74c766dd88-7kmmh` | `B/output/developer_torchrun_probe_a2/developer_probe_a2/rank0000{0,1}.json`；重启前末尾为 `8h25g` |
| CPU 型号 | Intel Xeon Platinum 8468 | 每 rank `/proc/cpuinfo` |
| nproc | OMP=1 时 1；去掉 OMP 限制后 192 | 同上 `nproc`、`nproc_without_omp`；不是 192 核配额 |
| cpu.max / 有效核 | `200000 100000` / 2.0 | 两 rank 看见同一共享 cgroup 和 affinity 0–191 |
| 内存上限 | 34,359,738,368 字节，32 GiB | memory.max |
| /dev/shm | 17,179,869,184 字节，16 GiB | statvfs；不是额外可绕过的内存额度 |
| 本地 /tmp 盘 | 总 1,073,741,824,000；可用 1,073,617,969,152 字节 | statvfs；没有写本地盘，所有实际写入在 AFS 私有目录 |
| AFS 小块 IO | rank0 写 127.67、读 198.42；rank1 写 139.83、读 206.18 MB/s | `B/output/developer_torchrun_probe_a1/developer_probe_a1/rank*.json`；2 MiB + fsync，读是热缓存，不能当持续吞吐 |
| GPU | 可用设备数 0 | 新探测还核对 `/dev/nvidia*` 与可见性；NVML 能列出宿主 H100，但显存为 Insufficient Permissions，不当作可使用的卡 |
| torchrun | WORLD_SIZE=2，LOCAL_WORLD_SIZE=2，GROUP_WORLD_SIZE=1 | 两 rank 同主机、同配额；`B/evidence/developer_capacity.json` 合并后只给 1 个 CPU 槽位，另一 rank 0 个 |
| Habitat/RLR CPU 导入 | pass，约 6.83 秒 | `B/evidence/native_import.json`；Habitat 0.3.3、RLRAcousticContext_v1、adapter enabled |
| EGL 真渲染 | 未验证 | 开发机无获授权的 GPU；CPU 探测只读元数据，不创建 GPU 上下文 |

P19 探测计划只写每 rank 的主机、CPU/cgroup、内存、shm/磁盘、AFS IO、GPU 名称/数量与节点/rank 信息，探测本身只用 CPU。EGL 是否可用留给获授权的 P19 GPU 试跑单独检查，避免把“能导入”写成“能渲染”。现有 exclusive 调度器在原 torchrun 退场后单实例执行命令；提供的包装器再按资源起本节点 torchrun，并明确拒绝多节点，直至有每节点启动交接。

存储采用更保守的十进制 2 TB = 2,000,000,000,000 字节。实际执行 `nice -n 19 timeout 300 du -sB1 /mnt/afs/xinyuanqian/jzy` 得到 **1,459,437,756,416 字节**，尚余 **540,562,243,584 字节**；出处 `B/evidence/du_total_bytes.txt`、对应 started/finished 时间记录。owner 的 `tmp/du_top_20261003.txt` 只是顶层取整统计，快照在 `B/evidence/du_top_owner_20261003.txt`。

| 项目 | 保守预留 GB（十进制） |
|---|---:|
| 已有数据、检查点及当前占用基线 | 1459.438 |
| 私有环境、代码、压缩包及部署余量 | 8 |
| 12 房与共享素材 | 4 |
| v7 第一批耐久交付媒体/证书/索引 | 380 |
| 720 组测试面板交付及封存材料 | 32（未实测） |
| 临时验证结论的小数据集 | 16（预留） |
| 自己的中间产物 | 34.360（32 GiB） |
| 训练检查点增长及估算误差余额 | **66.203** |
| 总额度 | **2000** |

预算计算见 `T/evidence/storage_budget.json`，CCI 镜像同路径可读。基线包含当时已落盘的一部分新环境，仍另留 12 GB 环境/素材额度，属于保守重计。4.8 TB 全量原始中间产物不能保留。阶段完成后须先核媒体/证书/哈希，再清本轮自己的 scratch；原始输入和别人的缓存不清。默认声明 4 GiB/世界、32 GiB scratch，所以同时最多 **8 个世界**，这是未实测峰值前的保护，不代表已经吃满 88 核。每 10 秒检查自己的实际 scratch/delivery 字节，超限停自己的孩子；采样间存在短暂超限可能，不能代替 AFS 硬配额。中断旧 attempt 计入占用，不自动删除来腾空间。交付只留视频、音频、题目/证书与 SHA，排除 IR 缓存和原始 npy/npz；原始证书保留来源路径，训练使用新的 `DELIVERY_INDEX.json` 耐久路径。

环境采用 48g 现成 Python 3.12.13 runtime + merged overlay 迁移，没有改共享 conda。仅在 T 的私有 packaging 目录装 conda-pack 0.8.1 / setuptools 80.9.0，按 CCI 目标前缀打包。解压使用 `--no-same-owner`；原环境的 editable hooks 只在复制品中禁用。CPU 导入需先 quaternion 后 habitat，沿用仓库现有 workaround。版本见 `B/evidence/cci_python_packages.txt`：NumPy 2.3.5、SciPy 1.18.0、numpy-quaternion 2024.0.13、soundfile 0.13.1、trimesh 4.8.3、numba 0.66.0、imageio-ffmpeg 0.6.0。

| 包 | 字节 | SHA-256 |
|---|---:|---|
| env.tar.gz | 468422530 | `3cb312d284f962d7e970c2634d9831bfaee78612523dab07116945e619742b59` |
| native.tar.gz | 11503818 | `70f3c59967d67661ef7e0e1938f83df9d7274a45fa7a0d1dfbb4b881a55f0dd0` |
| overlay.tar.gz | 18248416 | `add170f35124fd583281406a0ff722dd6249856b6c5303b15a1d48d5197922a8` |

SHA 源清单为 `T/evidence/environment.sha256`，CCI 核验为 `B/evidence/environment_sha256_verification.txt`。另把 48g 现有 GLVND 1.4.0-1 三个 DSO 放在 **B/private_glvnd**，仅作用于本环境 LD_LIBRARY_PATH，没有改系统库/驱动；其独立 SHA 清单 `T/evidence/private_glvnd.sha256` 为：libEGL.so.1=`4807e8376b41e63ff00521a811ec3898120577c2ee2213748495f2896675acde`，libGLdispatch.so.0=`f1e45111c2477eb926afe4a43c936ebecf3684b928992bc8f1d28c5f8125f461`，libOpenGL.so.0=`a635a47358cbdc0208c5d4d0fb69891f15caea0d5d588c99f84e61f76ba19ec4`。这解决 CPU 导入时缺少 libOpenGL 的错误，尚未证明 EGL/GPU 兼容。

数据按推荐清单传 12 套房（训练 8、开发 2、测试 2），包含 scan、navmesh/semantic、编译声学包与材料/QA 表；共享素材含 1985 个 WAV 文件（其中 p7_prepared_audio_v3 目录 296 个，sound_library_v1 原库 613 个；这是文件数，不是不同声类或独立 clip 数；出处 T/evidence/media_file_counts.json），配套声音元数据、58 个 Habitat 人物/动物/物体 runtime 包、声音池与资产 registry。本轮所有输入总计 **3752 文件、3,837,650,365 字节**；以 1.7 MB/s 估约 **0.627 小时**，加环境约 43 分钟纯字节传输，实际还受小文件及重启影响。完整逐文件清单 `T/evidence/input_manifest.json` 的 SHA 为 `e1ade4073e77ca76cb4004d5f8a0f892cec011b3cdb198493b61a611cac66092`。

| 输入分组 | 文件数 | 字节 |
|---|---:|---:|
| 先传 2 套训练房 | 42 | 330868819 |
| 另外 10 套房 | 214 | 1944440451 |
| 共享素材 | 3496 | 1562341095 |

传输均使用本机 `/home/neos/jzy/tools/cci_pull_from_48g.sh <T 下清单> B/mirror`。清单为 pull_bootstrap、pull_code、pull_house_first2、pull_remaining_data 等本轮文件；原始 48g 目录只读。开发容器在传输中重建，主机名变化；只停止已核对 argv/父 PID 的本轮失联 SSH PID 2655072，在新容器用同一 --partial 入口续传，未清空镜像。续传日志本机 `codex_runs/cci_datagen_20261003_codex/evidence/pull_remaining_data_resume1.log`：14:57:12–14:59:07 UTC，received 167,771,253 bytes，约 1,725,425 B/s，其余此前已落盘。**三个分组 SHA 校验全过**，`B/evidence/input_sha256_verification.json`，2026-10-03 15:04:01 UTC。

4 份运行配置单独生成 CCI 路径副本及 SHA lineage，目录 `B/configs/runtime_mapped_v2/`。原房屋/编译包 manifest 与 QA 不改。干声、prepared manifest 和分段声音池引用已存在；资产 registry 仍有 7 个历史 UE import manifest 未传，均属于 SPEAR/UE 路线，Habitat 是否完整使用这些配置仍须正式 v6 试跑确认。原始来源、有效副本和所有缺项逐项见 `PATH_MAPPING.json`。

声学对拍采用两套训练房：原生单线程 RLR，每房两个静态声源，再用固定解析校准干声卷积合成 10 秒四通道 float32 FOA。不是实际 v6 完整世界，也不是冻结测试成员。使用已有显式 API `allow_nonpassing_research_qa=True` 仅做诊断校准，生产入口没有该豁免。manifest 记录已部署 Git 提交号，变更代码版本会拒绝复用旧阶段；P19 同一作业重启也产生新的 launch ID，避免使用中断后的旧 claim。

| 房屋 | 48g 秒 | CCI 秒 | FOA 样本 SHA / 结果 |
|---|---:|---:|---|
| hm3d_train_00059_kJxT5qssH4H | 21.7616 | 17.4780 | `4edd8a7fc8d1d68175f4720fdacc615db80fc27df25e5a145eb76a65251b15fd`，一致 |
| hm3d_train_00096_6HRFAUDqpTb | 24.9218 | 17.9285 | `93aeaf69e671f3fdfd09b39125303975c3f0663c563ac089236cf78f90d2299d`，一致 |

原 WAV 文件哈希 **0/2 一致**：libsndfile FLOAT WAV 的 PEAK 块带创建时间戳；样本 **2/2 一致**，只排除此时间戳的规范比较哈希也 **2/2 一致**（分别 `1959a70679a82590832c8611deaac7d2f17b5c5474a968369c590752283585a3`、`7a8f4fd14c7c5de9b98edced97ff2eb68e5c3e2948ff1ffa425691ca1953963d`）。源文件没有改；完整原文件 SHA、时间戳、规范比较与样本 SHA 都在 `B/evidence/cpu_cross_machine_comparison.json`。48g 原结果 `T/cpu_smoke48_a2/SUMMARY.json`；CCI 原结果 `B/output/cpu_smoke_cci_a2/SUMMARY.json`。视频帧和题目 **not_run**，完整 v6 世界对拍数 **0**；不能据此报完整生成通过或每世界 17 秒。

生产阻塞有三项：

1. 12 房测量 review 被推荐，但所有当前 compiled package 的 geometry_report=fail、ray_leakage=not_run，生产默认 load_compiled_acoustic_scene 报 `compiled package QA is not pass: geometry_report, ray_leakage`。compiler parity/material coverage 均 pass。逐房出处 `T/evidence/package_qa_gate.json` 及原 `v7_batch1_20261003/measurement_final/houses/<house>/package/qa/`。没有修改 review 或伪造 pass；需要负责房屋编译验收的人交付合格包。
2. 开发机无可用/获授权 GPU，48g 未指定卡，所以没有新 GPU 作业。EGL 真渲染及完整 2–4 世界的视频/题目对拍待 P19 试跑；若容器 EGL 报错，入口保存报错并停止，不改系统库。可能出路需 owner 另行授权，例如先在指定 48g 卡渲极少量画面、P19 跑声学，不能自行恢复跨机大规模生产。
3. 尚无 v7 sampler 的已验收 64 个训练世界请求清单（CCI 路径版）；`B/configs/pilot64_request_paths.json` 缺失。不能自己造新数据分布或使用旧验证房填数。另一 Codex 的优化和测试面板代码未交接，本轮不接触其检出。

吞吐建议采用专用生成任务交付正式第一批；混合任务适合小验证集或有长期训练空闲 CPU 时逐渐生成。按 owner 的单节点规格举例（不是实测）：保留 ceil(88×10%)=9 核，专用任务最多分配 79 个单线程 CPU 进程，4 个渲染进程占其中至多 4 个；混合任务若留下 3 张卡、每卡两项训练，则 6×4 loader workers=24 核，生成上界 55 个 CPU 进程，其中 1 个渲染。内存先扣 6×33 GiB 与 10% 余量，并与实时 available/cgroup 限额取较小值。实际可运行世界还受 8 个 scratch 槽位限制，暂时不能宣称跑满 CPU。降低每世界中间产物峰值需接入另一负责人的优化与 P19 实测，不能靠只把并发数调大。

以下仅为可审查的**条件下界演算**，不是 P19 ETA。出处 `T/evidence/planning_estimates.json`，历史依据 `v7_scoping_20261002/throughput.json`：HM3D capture=69.365、finalize=161.043、render_total=230.834 秒；MP3D total=271.876 秒。历史 final export 的实际 IR thread counts 为 1:217、8:215，文件名 `_8` 不能证明真实线程数，所以不能把历史均值直接除以 79。设历史 worker 工时 H=1500–2300，画面比例 r=69.37/(69.37+161.04)，声学/合成相对历史倍率 α；专用 G=4,C=75，混合 G=1,C=54，活动世界 K=8，则下界 `max(rH/G, (1-r)Hα/C, [rH+(1-r)Hα]/K)`，不含重试、拒收、额外规划/封存、存储瓶颈和跨任务等待。720 组暂假定每组两个 10 秒画面和两条 FOA，即 1440 个视觉/声学配对；正式协议时长尚未冻结。

| 条件 | 专用第一批小时下界 | 混合第一批小时下界 | 专用720组小时下界 | 混合720组小时下界 |
|---|---:|---:|---:|---:|
| α=1，历史组合阶段不变 | 187.5–287.5 | 451.6–692.5 | 11.52 | 27.75 |
| α=5，组合阶段慢5倍的敏感性例子 | 711.7–1091.3 | 711.7–1091.3 | 43.73 | 43.73 |

即使先忽略 CPU，单渲染卡相对 4 卡的第一批画面下界也从 112.9–173.1 小时变成 451.6–692.5 小时。混合任务同时把训练并行卡数由 4 降至 3，按每卡两实验从 8 项降至 6 项；CPU/AFS 干扰可能进一步影响训练，未实测训练速度。nice 19 不能保证内存带宽与 AFS 无争用。若 4 张卡均训练占满，则仅可处理已有 capture 的声学/合成，不启动 Habitat 渲染。

后续 P19 64 世界试跑记录每阶段时间、scratch 峰值、RSS、GPU 峰值、SHA、吞吐与失败率；64 世界只从推荐 8 个训练房取，不碰开发/测试房。先完成 EGL 和完整 2–4 世界正确性对拍，再测稳态并发，按实测峰值提高 scratch 槽位而不突破 2 TB；用实测 renderer/CPU rate 和真实 v7 独立世界/反事实清单重算交付工时。混合任务另需 owner 明确六项训练命令及 loader workers，比较同卡同训练配置“生成开/关”的训练速度；目前没有训练命令，未启动混合任务。

当前推荐 12 房只含 2 套测试房；720 组面板的 12 套正式测试房清单和划分协议尚未交接，不把训练/开发房填作测试房。

封存测试：`formal_frozen_test` 当前明确拒绝，协议冻结和两房流程代码交接之前不生产正式测试。未来在 CCI 分离公开媒体与秘密答案、真值人物 ID、组映射；秘密由独立访问身份/AFS ACL 管理、单独 SHA 承诺，训练索引不包含秘密路径。相同 UID 下目录权限不足以保证训练读不到，访问主体安排尚未提供；本轮没有秘密答案或正式测试产物。

测试：`test_infrastructure.py` 覆盖共享 cgroup 不按 rank 翻倍、私有/父配额、训练 loader/内存余量、零槽位、CPU-only capture、真实子进程被 SIGTERM 中断、停自有子进程后续跑/损坏重算、只清本轮 scratch、manifest 变更/正式测试拒绝、存储预算、WAV 时间戳差异、配置路径与 JSON fragment 等。最终 19 项测试均通过；48g/CCI 耗时见日志。最终测试日志为 `T/evidence/infrastructure_tests48_final.txt`、`B/evidence/infrastructure_tests_cci_final.txt`；这些是 CPU 基础设施测试，不能代替 native 视频/题目验收。所有 shell 入口做 bash -n、Python 做 AST 解析。

已部署两份新计划，`B/evidence/plan_validation.json` 记录通过当前不可变 worker `p19_worker.6aa2c7f0065d7dc49676f7d666866ee2eef2911bb442586932ac6e4ac64f6ced.py` 的 load_plan 校验。`required_gpus=1` 是现有 exclusive schema 的最低字段要求，实际 rank 数读取运行时可见卡及每卡 rank 约定，不把它当 P19 卡数测量。无卡规格直接用生成脚本；不为其改共享排队格式。

可先加入探测行；试跑行须等前述合格包、64 请求与代码交接齐全后加入。**本轮没有排队**。

```text
/mnt/afs/xinyuanqian/jzy/hardbank/plans/cci_datagen_probe_20261003_codex.json 2 exclusive
/mnt/afs/xinyuanqian/jzy/hardbank/plans/cci_datagen_pilot64_20261003_codex.json 2 exclusive
```

计划本体也在 W `tools/cci_datagen/plans/`。探测日志/资源落 `B/output/probe/<launch-id>/rank*.json`，完成标记 `B/output/probe/PROBE_COMPLETE.json`；试跑阶段日志 `B/output/pilot/logs/<launch-id>/`、process.json 与 receipts 存自己的 AFS 目录，完成标记 `B/output/pilot/ALL_DONE.json`。停止时核对父 PID/命令，再 SIGTERM 自己的 torchrun/supervisor；子进程只回收自己记录的进程组，绝不 pkill。截止交付没有本轮继续跑的 GPU/训练/生成/传输作业。
