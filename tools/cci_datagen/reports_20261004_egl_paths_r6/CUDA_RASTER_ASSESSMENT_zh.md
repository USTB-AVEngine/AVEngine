仅渲染基础设施评估，不实施 CUDA 光栅化。

NVIDIA 用户态库为首选：P19 自己的 GPU 的 NVML 驱动版本必须与 /proc/driver/nvidia/version 完全一致；官方对应 .run 经官方 SHA 验证，extract-only，私有 GLVND vendor JSON 和 LD 路径。只有确认该 rank 的物理 GPU 与 CUDA/EGL 设备映射后创建上下文。库缺失可以在个人目录补齐，MIG/设备权限/内核能力限制不会因此消失。本轮实际 P19 是否成功以队列输出为准，不能把开发机预取包或软件 GL 初始化记为 P19 硬件通过。

CPU 软件 EGL 为第二选择：同一 Habitat、资产、轨迹和种子保留相机、骨骼和原有着色流程；每个渲染子进程 LP_NUM_THREADS=1，nice19，1核，流水线以图像编码管道输出，不落450帧dense G-buffer。CPU 与 RLR 共用扣训练加载器后的预算。按实测 T 秒/30秒世界、V 个不同画面估算 V*T/3600 核时；P 个可用核中分 R 给渲染，其他给单线程 RLR，墙钟下限 max(V*T/R, 成员数*声学秒数/(P-R))。没有测声学吞吐，不能填完整端到端墙钟数。只用静态开发机速度外推 P19 必须标未验证。显存全部留给训练；比较 raw RGB/depth/semantic，编码哈希不是逐像素等价判定。

CUDA 光栅化具备技术可能性，RasterizeCudaContext 不依赖 OpenGL，但 nvdiffrast 官方提供的是光栅、插值、纹理、抗锯齿等底层操作，没有现成相机、光照、材质系统。相机需逐项对齐 Habitat 的内外参、85°水平视场、像素中心、裁剪/深度语义、坐标方向及回读翻转；扫描网格需正确解析 UV/材质/纹理颜色空间、mipmap 和遮挡；M2 插入资产还要骨骼、动作时钟、嘴部纹理、PBR/AO/alpha及semantic ID。纹理各向异性、采样边界、抗锯齿/深度精度和 shader 实现差异均可能改变像素及可见性证书。

工程判断（未实测）：静态带贴图房间原型约2–4开发日；扫描网格＋M2＋相机/材质/深度/semantic与QA对齐约8–12开发日，单工程师且资产接口稳定；要求逐像素一致会扩大工期，不能保证。CUDA 路径仍占训练卡的算力/显存，不能消除共享卡争用。本轮没有安装、编译、运行 nvdiffrast，没有其速度/显存实测，也没有修改共享 PyTorch/CUDA 环境。推荐先完成私有 NVIDIA 和软件 EGL 的诊断，避免当前阶段重写渲染器。

官方依据：https://nvlabs.github.io/nvdiffrast/ 、 https://github.com/NVlabs/nvdiffrast/blob/main/nvdiffrast/torch/ops.py 。安装版本要求必须等未来原型获授权后按当时官方要求核对，本轮不实施。
