# AVEngine 全流程耗时与规模化瓶颈

> 所有表格只统计成功并产生可回读产物的运行；方向诊断失败、窗口化副卡启动失败等重试不计入正常吞吐。

## 实测阶段耗时

| 阶段 | 资源 | 计量单位 | 样本数 | median s | p95 s | 大规模判断 |
|---|---|---|---:|---:|---:|---|
| safe_mesh_inventory | CPU/read-only | 115 humans + legacy animals per audit | 1 | 2.39 | 2.39 | 可忽略；不要换成邻接拓扑审计 |
| pixal3d_cold_per_asset | 1 GPU + heavy CPU/storage model load and GLB extraction | one 1024 animal asset | 11 | 382.67 | 533.20 | 当前最大单资产瓶颈 |
| pixal3d_persistent_inference_and_glb | 4 persistent GPU workers + CPU parameterization | one 1024 animal asset | 33 | 159.68 | 341.32 | 均值 183.82 秒；每 worker 只加载一次模型 |
| pixal3d_persistent_batch_wall | 4 persistent GPU workers | one immutable 33-asset batch | 1 | 1818.58 | 1818.58 | 含 126.26--131.87 秒/worker 模型加载、GLB 回读和静态分片尾部空闲 |
| pixal_raw_to_100k_double_sided | CPU/Blender | one approximately 931k-triangle animal | 1 | 20.00 | 20.00 | 较小，可与其他资产并行 |
| animal_weight_transfer_and_glb | CPU/Blender | one 100k-face/300k-vertex animal | 1 | 164.00 | 164.00 | 第二大资产准备阶段，适合 CPU 多进程 |
| incremental_ue_cook_and_pak | CPU/storage; shared batch cost | one incremental package containing the Pixal pug | 1 | 188.52 | 188.52 | 批次共享成本，不应逐资产 cook |
| apartment_ue_render_18s_270_frames | 1 GPU; fixed stepping + PNG readback | one 18-second/270-frame human clip | 6 | 73.71 | 85.17 | 4 卡 offscreen 可线性并行 |
| rlr_binaural_18s_270_positions | CPU + small headless graphics context | one 18-second single-source clip | 5 | 6.20 | 6.62 | 占比很小 |
| topdown_metadata_and_review_finalize | CPU/FFmpeg/Matplotlib | one 18-second/270-frame clip | 6 | 124.93 | 133.67 | 单任务比 UE 慢；需独立 CPU 池 |

## 规模化估算

| 场景 | 并发假设 | 稳态吞吐 | 100 单位 | 1000 单位 | 当前瓶颈 |
|---|---|---:|---:|---:|---|
| 已导入 Rocketbox/Pixal 资产生成审核 clip | 4 UE GPU + 4 audio + 6 finalize | 172.9 clips/h | 0.58 h | 5.78 h | CPU review finalization at six workers |
| 新 Pixal3D 动物资产（冷启动 runner） | 4 GPU | 37.6 assets/h | 2.66 h | 26.57 h | repeated Pixal model load plus inference/GLB extraction |
| 新 Pixal3D 动物资产（当前 persistent 实测） | 4 GPU | 65.3 assets/h | 1.53 h | 15.31 h | CPU parameterization/GLB finalize 与静态分片尾部不均衡 |

## 结论与优化顺序

1. Pixal3D 仍是第一瓶颈，但 persistent worker 已把当前实测吞吐从冷启动估算的 37.6 提升到 65.3 assets/h。每张 GPU 必须只加载一次固定 revision；下一步用共享 claim queue 替换静态 round-robin，让先完成的 worker 接走未开始任务。扩散阶段使用 GPU，参数化、UV 和 GLB finalize 主要占 CPU，因此瞬时 GPU 利用率低不等于停滞。
2. 审核视频 finalization 是 clip 生产的主要 CPU 瓶颈。它应与 UE GPU 槽完全解耦，并缓存静态 Top-down 背景；增加 CPU worker 前先监控内存和磁盘写入。
3. UE 单演员 40k 与 100k 的捕获耗时差异落在噪声内，因此近景使用无空洞的 100k；减面主要降低多演员显存/绘制压力，不会解决固定步进和 PNG 回读。
4. UE cook/package 必须按资产批次执行，不能逐角色重复。RLR 音频只有约数秒，不值得牺牲事件语义或空间同步来换速度。
5. 规模化前再做 8/16/32 同场演员压力测试；当前 40k/100k 结论只覆盖单演员，不能外推到密集场景。

机器可读报告：[report.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/pipeline_timing_audit_v1/report.json)。Mesh 证据见 [asset_mesh_efficiency_audit.md](/data/jzy/code/AVEngine/docs/asset_mesh_efficiency_audit.md)。

## 2026-07-13 全量 Apartment 补充实测

下表是本轮真正落盘并回读的 wall time；恢复任务只处理缺失项，不重复成功项。

| 批次 | 工作量 | GPU/CPU 配置 | wall time | 最终结果 |
|---|---:|---|---:|---:|
| Rocketbox partition A | 70 clips | 2 UE GPU + 8 finalize | 21:14.83 | 70 / 70 |
| Rocketbox partition B | 69 clips | 2 UE GPU + 12 finalize | 21:17.40 | 69 / 69 |
| 猫 Apartment resume | 14 selected clips（总集 16） | 4 UE GPU + 12 finalize | 13:23.92 | 16 / 16 |
| 狗 Apartment 首轮 | 44 selected clips（总集 46） | 2 UE GPU + 12 finalize | 29:11.49 | 40 / 46；6 个旧贴地阈值失败 |
| 狗 6-clip 恢复 | 6 clips | 2 UE GPU + 6 finalize | 7:05.21 | 46 / 46 |
| Pug 尺度闭环首轮 | 12 clips | 2 UE GPU + 6 finalize | 10:00.66 | 11 / 12；1 个 RPC 启动失败 |
| Pug 单段恢复 | 1 clip | 1 UE GPU + 2 finalize | 3:08.70 | 12 / 12 |
| 31 动物最终物理测量 | 31 assets | 4 Blender CPU workers | 24.30 s | passed |
| 31 动物 post-Apartment 注册 | 31 assets | CPU/hash/readback | 6.80 s | passed |
| 31 动物 QA/scene-pool 编译 | 31 assets | CPU/hash/readback | 7.22 s | 92 pairs / 226 questions |
| 统一规范输入冻结 | 8 profiles / 72 requests / 34 assets | CPU/hash/readback | 0.60 s | 34 / 34 request bindings passed |
| 统一 manifest-only 数据集编译 | 34 assets | CPU/hash/readback | 8.10 s | 95 pairs / 232 questions |

Rocketbox 的两个 2-GPU 分区各达到约 197 clips/h 的批次吞吐；这已经包含 UE
固定步进、主视图回读和 CPU 审核合成，不能用单帧实时渲染 FPS 外推。猫批次中
UE 渲染时间中位数约 69.86 秒，而 12 路竞争时 finalization 中位数约
232.03 秒，进一步证明大批次主要等待 Top-down/Matplotlib/FFmpeg，而不是 GPU
着色器。

四个 UE worker 同时工作时，每卡通常占用约 5.5 GB。一次 30 秒采样的 GPU
平均利用率约为 14.8%、27.6%、29.9%、25.5%，短时峰值先达到 53%，队列铺满后
观察到 60% 以上并最高约 91%。利用率会在以下阶段自然掉到零：角色 UE 进程
切换、RPC/PNG 回读、RLR、Top-down 逐帧绘制和 libx264。因而调度判断应同时看
manifest 未完成数、UE PID、显存和 CPU finalizer，不能只看瞬时 GPU 百分比。

本轮 Pug 闭环期间 GPU 3 被另一位服务器用户的训练进程占用；我们保留对方任务，
并把唯一失败片段移到 GPU 0 恢复。该 10:00.66 不是无竞争基线。当前本项目 GPU
0--2 空闲是因为所有上述渲染已经完成；文档、哈希认证和 QA JSON 编译本来就是
CPU/存储阶段。

基于新实测，规模化优化顺序调整为：

1. 先把 Top-down 的静态 Apartment 背景缓存并减少 Matplotlib 每帧重建；
2. 保持 UE GPU 槽与 CPU finalizer 池解耦，按当前主机负载动态设 finalizer 数；
3. 只在 GPU 空闲且无其他用户任务时提高 UE worker 数，不能为利用率抢占共享卡；
4. 对每个新动物先做一次 UE 实测，只有物理误差明显超阈值才走尺度闭环和重渲染；
5. Pixal3D 继续使用常驻 worker/共享 claim queue，避免模型重复加载和尾部空卡。

## 2026-07-13 正侧姿势 canary 的 UE 实测

猫 v5 和比格犬 clay v6 共用一次增量 cook/package，随后四个 18 秒 clip 分配到
GPU 0、2、3。第一次约 13.5 秒的启动记录发生在 PAK 尚未包含新资产时，已保留
为失败证据，不计入正常吞吐；下表仅列 cook 后成功并回读的结果。

| 阶段/clip | wall time | 结果 |
|---|---:|---|
| 猫 + 狗共享 UE cook/package | 160.75 s | passed；最大 RSS 7,904,028 KB，批次只执行一次 |
| Tabby Walking UE render | 70.0879 s | passed |
| Tabby Idle UE render | 75.6404 s | passed |
| Beagle Walking UE render | 68.6774 s | passed |
| Beagle Idle UE render | 73.6466 s | passed |
| 四段 UE render 平均 | 72.0131 s | 与既有约 74 秒/clip 基线一致 |

共享 cook 的完整记录位于
[ue_cook_timing.txt](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/four_limb_rest_side_shared_ue_cook_v6_20260713_r1/ue_cook_timing.txt)。这再次说明 cook 是批次成本，不能为每个颜色、尺寸或动作重复
执行；Walk/Idle 可并行渲染，Top-down、RLR 和 FFmpeg 继续使用独立 CPU 池。

## 2026-07-14 稳定原生 Husky 端到端实测

这是首个不依赖单图重建拓扑的稳定原生模板 UE canary。导入和 cook 是一个
模板/批次成本；Walk、Idle、颜色与尺寸实例不应分别重复它们。

| 阶段 | wall time | 峰值 RSS | 结果/解释 |
|---|---:|---:|---|
| Husky UE editor import/readback | 79.97 s | 2,322,364 KB | 10 个资产：mesh、skeleton、physics、5 materials、Walk、Idle、Blueprint |
| 全量共享 cook/package/archive | 385.71 s | 7,524,084 KB | 4.53 GB 原始条目压至 4.47 GB Pak；0 errors；后续 clips 共享 |
| Idle UE render（270帧） | 68.66 s | 单 UE worker | 动作/方向/落地门通过 |
| Walking UE render（270帧） | 71.02 s | 单 UE worker | 四个路径方向窗口通过 |
| 两段并行 CPU finalize | 118.09 s | 595,636 KB | metadata、Top-down、FFmpeg、registry；没有启动 UE/GPU |

完整 cook 计时：
[ue_cook_timing.txt](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_shared_ue_cook_husky_v1_20260714/ue_cook_timing.txt)。
最终批处理状态：
[batch_status_final.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/batch_status_final.json)。

本次全量 cook 比前一次增量猫/狗 cook 慢约2.4倍，原因是命令没有启用
incremental，遍历了约4,548个 packages；它不是每个动物的固有成本。批量生产应
先集中导入一批模板/材质变体，再执行一次增量或共享 cook。单个18秒 UE clip
仍稳定在约69–71秒；两段并行 finalize 平均占约59秒 wall/clip，但会与下一批
GPU render 重叠。因此扩规模时的顺序应是：资产批量导入 → 一次 cook → 多 GPU
渲染队列 → 独立 CPU finalize 池。

## 2026-07-14 剩余 11 个原生模板批量实测

该批次沿用 Husky 已验证的同一实现，一次导入 11 个模板、一次共享 cook，再将
22 个 Walk/Idle clip 分发到 GPU 0/2，并由 6 个 CPU worker 并行完成轨迹图、
双耳音频、FFmpeg 和 registry。GPU 1 未使用，GPU 3 保留给其他用户。

| 阶段 | wall time | 峰值 RSS | 产出 |
|---|---:|---:|---|
| 11 模板 UE editor import/readback | 32.08 s | 2,502,076 KB | 11/11；每个都有 mesh、skeleton、materials、Walking、Idle、Blueprint |
| 共享 cook/package/archive | 168.88 s | 7,960,340 KB | 仅执行一次；11 个资产标签全部进入 Pak |
| 22 段 UE render + CPU finalize | 986.18 s | runner 584,852 KB | 22/22 passed，0 failed，0 incomplete；2 GPU render workers + 6 finalize workers |
| 端到端批次合计 | 1,187.14 s | 见上 | 约 19 分 47 秒；共享成本摊销后约 53.96 s/clip 的批次 wall time |

这里的 `53.96 s/clip` 是并发批次吞吐，不是单段 UE 渲染延迟：两个 UE 进程
通常各占一张卡约 25–40%，而 Top-down、音频和编码主要使用 CPU。提高 GPU
进程数会同时增加显存、CPU 解码/编码与磁盘争用，当前 2 GPU + 6 finalize 是
不碰 GPU 1/3 时的稳定配置。更大规模应复用已经 cook 的 Pak，避免把 32.08 秒
导入和 168.88 秒 cook 重复计入每个实例。

完整证据：
[ue_import_result.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_ue_import_remaining11_v1_20260714/ue_import_result.json)、
[ue_cook_timing.txt](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_shared_ue_cook_remaining11_v1_20260714/ue_cook_timing.txt)、
[batch_status.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_remaining11_v1_20260714/batch_status.json) 和
[batch_qa_summary.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_remaining11_v1_20260714/batch_qa_summary.json)。

## 2026-07-15 比格四属性 OFAT 批次实测

本批验证了一个稳定原生比格模板上的四个 instance 属性域：`size`、
`body_build`、品种专用 `coat_tone`、`life_stage`，每个属性三个绝对值。代码先
冻结完整的 81 组合空间，再真实构建 baseline 加 8 个单变量实例，共 9 个 GLB；
每个实例均保留同一骨架、Walking、Idle 和拓扑，只改变该请求声明的属性。

| 阶段 | 工作量 | wall time | 结果/摊销规则 |
|---|---:|---:|---|
| UE editor import/readback | 9 个实例 | 约 51.48 s（引擎日志首末时间）；其中 commandlet 22.71 s | 9/9 passed；属于批次导入成本 |
| 共享 cook/package/archive | 9 个实例，共用一次 | 378.91 s | passed；约 4.6 GB Pak；不得按实例重复 |
| Apartment render + audio + finalize | 17 个待运行 clip，外加 1 个已通过 canary 复用 | 790.85 s | 最终 18/18 passed；2 UE GPU workers + 12 CPU finalizers |
| 最终认证聚合 | 9 实例 / 18 clips | CPU/hash/readback，未单独计时 | 全部媒体、音频日程和 registry 重新哈希通过 |

这 17 个新 clip 的批次吞吐约为 77.39 clips/h；若按最终 18 段产出摊销，约为
43.94 s/clip。两者都不是单段延迟，因为一个既有 canary 被安全复用，而且 UE
渲染、Top-down、音频和 FFmpeg 在流水线上重叠。自动门的最终最差值为：躯干
forward 误差 `0.91146°`、body-up `0.86249`、地面穿透约
`7.1e-15 cm`（浮点零）；每段 18 秒音频包含 7 个有静音间隔的狗叫事件。

本轮没有为 81 个组合全部重复 image-to-3D。稳定模板路线把尺寸映射为 actor
scale，把体型、毛色亮度和年龄迹象映射为保拓扑的确定性参数；FLUX.2 prompt
仍完整记录四个属性，但只用于语义纹理细节候选。因此，大规模生成时主要成本仍是
UE clips，而不是 instance JSON 或材质参数化。只有新增几何类别/物种时才需要重新
承担 FLUX/Pixal/绑定与人工方向门成本。

完整证据：
[UE import](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_ue_import_v2_r3_20260715/ue_import_result.json)、
[共享 cook](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_shared_ue_cook_v2_20260715/ue_cook.log)、
[批处理状态](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_apartment_specs_v3_20260715/batch_render_status.json) 和
[认证 manifest](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_apartment_review_v3_20260715/review_manifest.json)。
