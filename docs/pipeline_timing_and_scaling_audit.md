# AVEngine 全流程耗时与规模化瓶颈

> 所有表格只统计成功并产生可回读产物的运行；方向诊断失败、窗口化副卡启动失败等重试不计入正常吞吐。

## 实测阶段耗时

| 阶段 | 资源 | 计量单位 | 样本数 | median s | p95 s | 大规模判断 |
|---|---|---|---:|---:|---:|---|
| safe_mesh_inventory | CPU/read-only | 115 humans + legacy animals per audit | 1 | 2.39 | 2.39 | 可忽略；不要换成邻接拓扑审计 |
| pixal3d_cold_per_asset | 1 GPU + heavy CPU/storage model load and GLB extraction | one 1024 animal asset | 11 | 382.67 | 533.20 | 当前最大单资产瓶颈 |
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

## 结论与优化顺序

1. Pixal3D 冷启动是第一瓶颈。每张 GPU 必须使用 persistent worker，只加载一次固定 revision；当前冷启动 runner 保留为对照证据。
2. 审核视频 finalization 是 clip 生产的主要 CPU 瓶颈。它应与 UE GPU 槽完全解耦，并缓存静态 Top-down 背景；增加 CPU worker 前先监控内存和磁盘写入。
3. UE 单演员 40k 与 100k 的捕获耗时差异落在噪声内，因此近景使用无空洞的 100k；减面主要降低多演员显存/绘制压力，不会解决固定步进和 PNG 回读。
4. UE cook/package 必须按资产批次执行，不能逐角色重复。RLR 音频只有约数秒，不值得牺牲事件语义或空间同步来换速度。
5. 规模化前再做 8/16/32 同场演员压力测试；当前 40k/100k 结论只覆盖单演员，不能外推到密集场景。

机器可读报告：[report.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/pipeline_timing_audit_v1/report.json)。Mesh 证据见 [asset_mesh_efficiency_audit.md](/data/jzy/code/AVEngine/docs/asset_mesh_efficiency_audit.md)。
