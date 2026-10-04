P19 上的软件 EGL 已在上一轮渲过 26 个抽样帧，本轮 Mesa 26 在 CCI 开发机一核完成 30 秒、720p 世界为 110.84 秒；私有 NVIDIA EGL 库已按 580.95.05 准备，但本轮两阶段 P19 诊断仍在普通队列，硬件补库能否成功、P19 完整世界速度及同卡影响均未验证。

本简报只涉及渲染基础设施，正式数据、题目、FOA 和测试房生成数均为 0。代码在 48g 的独立检出 `/data/jzy/code/AVEngine-cci-egl-paths-20261004-codex-r6` 修改、测试并提交；分支 `codex/cci-egl-paths-20261004-r6`，运行源码提交 `58c409274ce3eae028bcfe7adaea161604b3d8fe`，生成器科学版本基于资产合并后的 `fbdedbfb58044396bfedaef063ef4c9c4214fcba`。没有 push，没有改别人的检出或共享环境。

以下将 CCI 个人工作区 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_egl_paths_20261004_codex_r6` 简记为 `B`。代码发布到 `B/code_v2`，参数在 `B/configs/egl_paths_profile_v2.json`；旧版本和失败证据保留。运行时先核对 `/proc/driver/nvidia/version` 与本 rank 所租 GPU 的 NVML 版本，必须完整版本一致；还核对物理 GPU UUID、CUDA 可见设备 UUID 和 EGL 设备映射，确认后才创建上下文，不能把物理 GPU 编号直接当成 `CUDA_VISIBLE_DEVICES` 后的 CUDA ordinal。

| 路线/机器 | 已验证内容 | 时间、内存与出处 | 尚未验证 |
|---|---|---|---|
| 私有 NVIDIA EGL / CCI 开发机 | 官方 580.95.05 包 SHA256 正确，extract-only 完成；仅个人目录内的 8 个 EGL/GL DSO 进入私有库路径 | `B/evidence/devbox/driver_package_assembly_v2/ENVIRONMENT_READY.json` 和 `PRIVATE_DRIVER.json` | 开发机是 MIG，本次不在其上启动硬件 GL；P19 新计划待运行 |
| P19 原始容器 / 上一轮诊断 | rank4、GPU0，H100 80GB，MIG Disabled，NVML 580.95.05；`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`，无 NVIDIA EGL vendor/libraries；原生默认/device 两种方式均不能创建上下文 | 原文在上一轮 `output/scheduler_diagnostic/egl_diagnostic_r5/eval_job_a01_job_20261004T035951Z_b69cc9-t8wb6/metadata.json`、`inherited_habitat.log`、`EGL_PLATFORM_device_habitat.log` | 此项不能替代补库后的新诊断，也不证明 H100 本身不能做 EGL |
| Mesa 26.2.4 / CCI 开发机 | 同一训练房 HM3D `00059-kJxT5qssH4H`，1280×720、15fps、450 帧；LP_NUM_THREADS=1、nice19、亲和 CPU191、无 GPU 可见 | setup 11.8971s，render+encode 98.9009s，总计 110.8422s；峰值 RSS 1,502,676 KiB；四个参考帧单纯渲染 0.1630–0.1703s/帧；`B/evidence/devbox/condaforge_mesa_v2/software_30s/RESULT.json` | 新版 Mesa 在 P19 的完整 450 帧吞吐、硬件像素差异待新计划 |
| Mesa 26.2.4 / 48g | 相同 Habitat 0.3.3、Mesa/LLVM、请求和素材；一核 CPU111，没有启动 48g GPU 作业 | setup 10.3076s，render+encode 144.7492s，总计 155.1244s；峰值 RSS 1,516,332 KiB；`/data/jzy/tmp/claude_hardbank_20260923/cci_egl_paths_20261004_codex_r6/evidence/condaforge_reference_48/RESULT.json` | 不是 48g 硬件 EGL 参考 |
| P19 软件 EGL / 上一轮 Mesa 23 诊断 | 实际在 P19 rank4 跑过 26 个抽样 720p 帧，原生 Habitat/M2 软件上下文成功 | setup 15.6287s，26 帧含编码 7.3452s，均值 0.282507s/帧、峰值 RSS 1,423,960 KiB；上一轮 `software_sample/RESULT.json` | 450 帧外推 142.757s 未验证；此环境与本轮私有 conda-forge Mesa 26 不同 |
| nvdiffrast CUDA | 只做接口、工作量和风险评估 | 没有安装、编译、运行，速度和显存均未实测 | 不存在本轮 CUDA 渲染实测输出 |

软件跨机器对拍取帧 0、149、300、449 的 RGB uint8、depth float32、semantic uint32。相同 Mesa 26.2.4/LLVM 23.1.2 的 12 个 raw 数组逐元素完全一致，MAE 和最大差均为 0，见 `B/evidence/devbox/condaforge_mesa_v2/SAME_VERSION_48_COMPARE.json`。Mesa 26 与旧 Mesa 23 的 RGB 不完全一致，MAE 0.4330–0.4378/255、最大通道差 32；depth/semantic 八个数组完全一致，见同目录 `CCI_CF_MESA_VS_48G_MESA_PIXEL_COMPARISON.json`。不能把跨 Mesa 版本的 RGB 哈希相等当作前提，也不能把软件对拍记成硬件对拍。新计划会保存 P19 硬件参考，再与 P19 软件逐像素比较；若硬件失败，结果文件会明确写未对拍。

NVIDIA 路线从 [NVIDIA 官方包](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/NVIDIA-Linux-x86_64-580.95.05-no-compat32.run) 下载，325,359,760 字节，SHA256 `482130b27454864f3873729debbffa9e317badd1c56271dd811219c2385da575`，经同站 `.sha256sum` 验证。保存 [NVIDIA Driver License Agreement](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/license.txt) 原文于 `B/driver_cache/580.95.05/license.txt`，SHA256 `c692fb14ad4f499c6aee1b9e144e6905148c420e6225b17cb2937366c1e507ad`，只放在个人目录，未将驱动二进制加入 Git/代码包或再分发。没有运行安装器、内核工具、mknod、modprobe、系统 ldconfig 或修改设备权限；没有替换 CUDA/NVML 库。[官方文档](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/faq.html) 说明 extract-only 的解包用途。

实际使用的私有 GLVND json 为 `B/driver_runtime_verified/580.95.05/10_nvidia_private.json`，绝对路径指向自己的 `libEGL_nvidia.so.0`；库目录为该目录下 `lib/`。8 个原始 DSO 的文件 SHA、SONAME、DT_NEEDED 在同目录 `COMPLETE.json`。首次默认解包失败并保留隔离目录；修复版只对自己的 tar 子进程取消保留属主、权限和时间戳，记录原始 tar stderr，重新解包成功。首次 tar stderr 被官方 wrapper 隐去，具体失败根因未验证，不能据此声称 AFS 一定有问题。

Mesa 路线使用个人前缀 `B/mesa_egl_env`，钉住 `mesa-libgallium=26.2.4=h1d927ee_0`、`mesa-llvmpipe=26.2.4=h62484d9_0`、`libegl/libopengl/libglvnd=1.7.0=ha4b6fd6_5`，LLVM 23.1.2。包、许可证及 DSO 哈希在 `B/evidence/mesa_egl_manifest.json`、`mesa_egl_explicit.txt`。仅设子进程 `EGL_PLATFORM=surfaceless`、`LIBGL_ALWAYS_SOFTWARE=1`、自己的 vendor json、LD 路径；冻结的 CPU 设备选择器排除硬件。实际 GL_RENDERER 和映射的私人 libgallium 路径同时校验。不能只安装 Vulkan 的 `mesalib` 就认为已获得 EGL；本轮依据的是 [conda-forge 实际 split recipe](https://raw.githubusercontent.com/conda-forge/mesalib-feedstock/main/recipe/recipe.yaml) 和文件实查。

普通诊断计划为 `/mnt/afs/xinyuanqian/jzy/hardbank/plans/cci_egl_paths_ordered_shared_20261004_codex_r6.json`。一个实验 `egl_paths_r6` 在同一个已获租约的 rank 顺序执行 `nvidia`、`mesa`；每个渲染子进程只用一核、nice19。运行时读取各层 cgroup 和亲和范围，先扣其他训练 rank 加载器的保守/实测用量与余量；诊断不花掉全部可用核。硬件上下文只在本 rank 的卡上建立，自有渲染显存守卫 2 GiB、启动/运行需留至少 4 GiB 空闲，1 秒采样，保存自有 PID/命令/启动 tick 和 NVML 原文。守卫为采样，不能解释成系统硬显存分区；训练进程不暂停、不改参数、不接收信号。前后只读已有训练日志和 NVML，能观察并存与显存变化，不能据此宣称训练速度完全不受影响。

2026-10-04 07:08:54 UTC 已按授权备份并只在队列末尾追加第 116 行；原队列前缀保留，未修改旧 R4/R5 行，`p19_run.sh` 和 protected worker 哈希未变。备份 `B/evidence/p19_queue.before_20261004T070854Z_211d32f4d008.txt`。队列行原文：

```text
/mnt/afs/xinyuanqian/jzy/hardbank/plans/cci_egl_paths_ordered_shared_20261004_codex_r6.json 2 # 2026-10-04 Codex round6 owner-authorized ordered render diagnostics: exact-driver private NVIDIA EGL then conda-forge CPU Mesa; ordinary one rank nice19 1CPU GPU<=2GiB/free>=4GiB; no training/RLR/datasets; source 58c4092
```

派发预览命令在 CCI 开发机执行，exit 0：

```bash
P19_DRY=1 HB_PY=/mnt/afs/xinyuanqian/jzy/hardbank/envs/avledger-py310/bin/python LADDER_JOB_ID=job_20261004T035951Z_b69cc9-t8wb6 bash /mnt/afs/xinyuanqian/jzy/hardbank/p19_run.sh
```

自己的派发预览原文如下；1200 秒是调度器每个 eval 600 秒的默认估算，不是渲染实测时间。完整 stdout、备份/追加校验均在 `B/evidence/P19_DRY_dispatch_preview.txt`、`QUEUE_APPEND.json`、`P19_DRY_RESULT.json`。

```json
{"event":"dispatch_preview","dispatch_index":1,"plan":"cci_egl_paths_ordered_shared_20261004_codex_r6","priority":"normal","queue_line":116,"experiment":"egl_paths_r6","estimated_remaining":{"seconds":1200.0,"sources":["fallback:600s_per_eval_stage"]}}
```

预览 `queue_action=pool`、`exclusive=null`。这不证明已分到 rank。诊断 `run.json` 的 complete/done 只表示记录完成；真正成功必须看 `DIAGNOSTIC.json` 的 `render_passed` 和具体上下文/450 帧 case，不能重复把元数据通过当渲染通过。

本轮工作区占用实测 5,035,397,120 字节（5.04 GB，含自己的失败解包隔离、环境包缓存、两个代码版本、参考帧），来源 `nice -n19` 单核执行 `du -x -B1 --max-depth=1 B`，记录 `B/evidence/OWN_TASK_STORAGE.json`，耗时 3.84 秒。没有删除 AFS 数据、环境或缓存。每种后端只渲一份完整视频与 4 帧 raw 参考，不落 450 帧 dense G-buffer；预计两个 P19 阶段增加小于 0.15 GB。owner 2 TB 总额没有通过 filesystem 的全局空闲误算，也没有重新做全 AFS 重扫。之前 ≥600 GB 清理方案仍未执行；64 世界计划本轮没有重排，正式生产版本和存储清理需另行交付。

计算统一用当前 v7 规格，不能混用原来 3000 parent 的第一批数字。`fbdedbf/V7_SPEC_zh.md §13`：v7-S 500 parents、875 members、1750 windows、24500 逻辑题；完整种子版 10000 parents、16075 members、16950 windows、206900 逻辑题，新增酷家乐部分未验收、未计入。画面复用须先验证嘴部/动作/物理状态实际不变，通过后分别为 625、12025 份画面；未通过时按 875、16075 份算。

| 方案（估算，未在 P19 完整生产验证） | v7-S | 完整种子版 | 条件与出处 |
|---|---:|---:|---|
| 私有 Mesa，仅画面，一核参考速度外推 | 19.24–26.93 核时 | 370.24–518.16 核时 | 110.8422s（CCI 开发机）至 155.1244s（48g），有验收后的画面复用；不是置信区间 |
| 私有 Mesa，不复用画面 | 26.94–37.70 核时 | 494.94–692.67 核时 | 每个 member 都渲；声学另计 |
| 私有 Mesa，16 个剩余核全部渲画面的理想墙钟 | 1.20–1.68h | 23.14–32.38h | 声学不能同时使用这 16 核；并发 AFS/缓存/训练争用未测 |
| 48g 渲、CCI 合成：仅视频体积与传输 | 1.17–1.51 GB，11.49–14.85min | 22.55–29.14 GB，3.69–4.76h | 4 个已存在 30s 视频 1,875,493–2,423,377B，CCI 从 48g 拉按 1.7 MB/s |
| 48g 渲、CCI 合成：视频+必要可见性/材质证书 | 2.00–2.34 GB，19.58–22.98min | 38.42–45.09 GB，6.28–7.37h | 不传 debug、dense G-buffer 或派生资产缓存；可移植 manifest/导入适配尚未实施 |
| 48g 渲染卡时，旧 v6 10s×3 的代理 | 36.13 卡时 | 695.10 卡时 | `v7_scoping_20261002/throughput.json` HM3D capture 69.3651s；不是新 v7 实测 |
| 整套放 48g：旧 v6 代理，4 个 end-to-end worker 理想墙钟 | 38.39h | 713.10h（29.71 天） | 若加 25–50% 排队/争用，48–58h / 891–1070h；仍是未验证外推，且违反全 CCI 生成要求 |

完整计算和原始媒体路径在 `evidence/ROUTE_ESTIMATES_r6.json`。软件 CPU 分配应令剩余核数 `P=min(亲和范围, 各层 CPU quota)-训练加载器预算-余量`，将 `R` 核给单线程软件渲染、`P-R` 核给单线程 RLR。流水线墙钟下界为 `max(画面数*T_render/R,成员数*T_audio/(P-R))/3600` 小时。本轮没有测新生成器单线程声学 `T_audio`，不能报整套 v7 的精确生产耗时或声称所有 88 核可直接占满。nice19 降低优先级，不会消除内存带宽/AFS 争用。

CUDA 光栅化只评估：[`RasterizeCudaContext`](https://nvlabs.github.io/nvdiffrast/) 不需 OpenGL，但只提供光栅化、插值、纹理和抗锯齿等底层操作，现成相机/光照/材质系统需自己补。需对齐 Habitat 的相机内外参、85° HFOV、像素中心、深度与坐标方向、回读翻转；扫描网格 UV/颜色空间/mipmap；插入 M2 的骨骼、时钟、嘴部纹理、PBR/AO/alpha、semantic ID。工程判断：静态房原型约 2–4 开发日，扫描网格+M2+QA 约 8–12 单人工程日；逐像素一致无法保证，仍会与训练争用 CUDA/显存。见 `CUDA_RASTER_ASSESSMENT_zh.md` 和 [官方实现](https://github.com/NVlabs/nvdiffrast/blob/main/nvdiffrast/torch/ops.py)。

推荐按原顺序先看私有 NVIDIA 库是否在 P19 真正通过；若通过且显存留量与同卡影响可接受，它保留 Habitat 原始硬件管线。若失败或共享卡余量不足，优先采用已有原生 Habitat 软件 EGL 的 CPU 管线，所有数据继续在 CCI 生成，图像后端在正式协议冻结前统一验收。CUDA 重写不适合当前交付周期。48g 画面+CCI 声学是最后备选：同一请求/种子/资产哈希，两端严格校验后在 CCI 合并；只传瘦身画面和证书。48g 没有指定可用 GPU，本轮未启动任何 GPU 作业；未来必须有明确卡租约并限定小块运行、显存/利用率守卫，与训练、评测和 UE 的卡隔离，不能擅自借卡。整套移到 48g 只保留估算，没有执行。

还未完成：新的 P19 NVIDIA/Mesa 完整世界实测与同卡影响、P19 硬件/软件像素对拍。队列项已持久化；当前没有该新计划的运行 PID。代码单元检查 55 项通过（48g 日志 `.../cci_egl_paths_20261004_codex_r6/evidence/tests48_compat.log`，13.200s），CCI 2976 个源码文件及 Git identity/clean 状态核验通过；这些检查不能替代 P19 实测。原始诊断日志另附 `DIAGNOSTIC_RAW_zh.md` 与只读证据快照。P19 结果产生后应更新本简报，不把排队或 done 标记当作任务完成。

最后可核对的队列状态（2026-10-04 08:43:54 UTC）：rank7 在 08:43:27 完成原评测实验，08:43:29 按队列接走前面的第 115 行训练计划；自己的第 116 行尚未派发，未见本轮尝试目录或诊断 PID。出处为 `QUEUE_WAIT_STATUS_20261004T0844Z.json` 及其指向的 runner 原始日志。之后 CCI 登录直接链路与经 48g 链路均出现 SSH banner 超时，因此不能声称掌握更新的 P19 状态。缺的是普通 rank 的合法租约和其诊断输出；未借空卡、未手动重启 worker、未改变队列优先级。计划会随正常调度执行，结果需再核对。
