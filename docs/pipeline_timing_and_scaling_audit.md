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
