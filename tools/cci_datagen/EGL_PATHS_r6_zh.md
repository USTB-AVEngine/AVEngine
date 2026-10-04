这份普通 P19 计划只用于排查图形基础设施，两个 eval 在同一实验租约内按 NVIDIA、Mesa 顺序执行。没有 exclusive，不修改 p19_run.sh/worker/旧失败计数；每卡 2 个实验，只有自己的 rank 创建上下文，peer 只读日志、加载器元数据及 NVML；诊断 done 不表示渲染成功，逐例看 DIAGNOSTIC.json 的 render_passed。

NVIDIA 路径先读取当前 P19 内核模块版本与自己 GPU 的 nvidia-smi 驱动版本，必须逐字符相同。官方同版本 no-compat32 .run 经官网 sha256sum 核验，用 sh --extract-only --target 解包；只把 64-bit EGL/GL 白名单库和 SONAME 链接放入个人 driver_runtime/<version>/lib，不装驱动，不复制 libcuda/NVML，不执行 installer、modprobe、mknod，不把提取目录的工具暴露在 PATH。许可证原文、包 SHA、逐库 SHA、readelf 依赖和厂商 JSON 都登记；仅个人 AFS 使用，不再分发驱动。预取开发机版本不能代替 live P19 校验。

CUDA_VISIBLE_DEVICES 必须是受保护 worker 给本进程的单一卡号。只枚举 CUDA、核对 UUID 与 NVML 的物理 GPU 租约，映射出本进程 CUDA ordinal，再交给 Habitat 的 gpu_device_id；不猜物理 GPU0，不在别人 rank 执行。nice19 单 CPU，GPU 自身 2GiB、同卡空闲4GiB的软件守卫会核对并只终止自己的子进程，不能当硬件配额或保证零训练影响。上下文先64x64，再同训练房30秒1280x720/15fps/450帧；流式编码，额外只留4帧 raw RGB/depth/semantic 供逐像素对拍，没有声学/出题/数据集发布。

CPU 路径用个人 conda-forge mesa-libgallium/mesa-llvmpipe 26.2.4、libegl/libopengl/libglvnd 1.7.0，记录所有实解 build/依赖 SHA。旧 mesalib 25.2.8 的本次构建只带 Vulkan lavapipe，不能误记为软件 EGL。CPU 子进程隐藏 CUDA，设置私有厂商 JSON、surfaceless、LIBGL_ALWAYS_SOFTWARE、LP_NUM_THREADS=1、原冻结 software-only selector；所有环境变量只作用于自己的子进程。渲同一30秒训练房并保存4帧，与本轮成功 P19 NVIDIA EGL 及已有48g Habitat软件参考比 RGB MAE/max/不同像素比例、depth米级误差、semantic误差。版本可替换，但 source/request/ref 必须一起冻结核验。

nvdiffrast CUDA backend 只作技术和人日评估，本计划不下载/编译/运行它。开发机 MIG 只测 CPU 环境，无硬件尝试；48g 没分配 GPU，不自选卡开作业。所有诊断正式数据数为0；不采测试房。P19 未派发或硬件无成功参考，像素差/吞吐/训练影响必须标为未验证。
