# 音频仿真方案调研 — 2026-07

> **目的**：AVEngine 目前用 GPURIR（ISM，image source method）做 4 通道室内空间音频。本文调研 GPURIR 替代方案、室外能力、UE 原生 audio、类似仓库、材质对声音的影响，并给出**6 个月内 ROI 最高的升级路径**。

---

## 摘要（TL;DR）

- **GPURIR** 是 shoebox + Sabine T60 + ISM 的老组合，**限于矩形房间、单个 T60 值、无材质频谱**
- **推荐 2 阶段升级**（总预算 ~6 周）：
  1. **阶段 A（2 周）**：换 **pyroomacoustics 0.10.1** hybrid ISM+RT + Vorländer 7-band materials DB → 立刻拿到 arbitrary 3D 房间 + 频谱倾斜 + 41-55% RIR 精度提升
  2. **阶段 B（3-4 周）**：加 **Steam Audio 4.8.1 headless C SDK**（Apache-2.0, per-triangle materials）作为 "hero shot" 后端；室外用 pyroadacoustics 大气 FIR 后置
- **明确不做**：Cosmos/UE 原生 audio（无 headless RIR export）、Treble（cloud 昂贵）、FDTD/BEM（可听频段室外爆炸）、RLR-Audio-Propagation（CC-BY-NC 商业禁）
- **GPURIR 处理室外**：**不能**（需闭合 shoebox），但可 hack 出"只有地面反射"退化模型
- **UE 5.5 原生**：无 headless ray-traced acoustics；Steam Audio for UE 是唯一实用选项，但也可跳过 UE 直接用 Steam Audio C SDK

---

## 1. GPURIR 替代方案（RIR 仿真核心）

| 工具 | Method | GPU | Mesh | Materials | Ambisonic | License | 6 个月内 ROI |
|---|---|---|---|---|---|---|---|
| **GPURIR**（现状） | ISM | ✅ CUDA | ❌ shoebox only | ❌ 单 T60 | ❌ | GPL-3 | 保留（快速 batch） |
| **pyroomacoustics 0.10.1** | Hybrid ISM+RT (C++/Cython) | ❌ CPU | STL (via wall_factory) | ✅ 7 octave bands, Vorländer DB (~80 材质) | 通过 64-mic SH 阵列 hack | MIT | ⭐⭐⭐⭐⭐ |
| **Steam Audio 4.8.1** | Bake + realtime ray tracing | ⚠ AMD only (Radeon Rays / TAN) | ✅ Static/InstancedMesh | ✅ 3-band (v4.7+ octave experimental) | ✅ 1st-32nd order | Apache-2.0 | ⭐⭐⭐⭐ |
| **RLR-Audio-Propagation** | Bi-directional Monte-Carlo path tracing | ❌ CPU | ✅ Habitat scenes | ✅ N-band configurable | ✅ HOA | ❌ **CC-BY-NC 商禁** | ⭐（研究可用，商业不可用） |
| **pygsound / GSound-SIR** | Geometric ray tracing + diffraction | ⚠ 部分 | ✅ mesh native | ✅ per-triangle | ✅ | Apache-2.0 | ⭐⭐⭐（备选） |
| **Treble Technologies** | Hybrid DG-FEM + GA | ✅ 多 GPU (cloud) | ✅ OBJ/DXF/3DM | ✅ oct + 1/3-oct 复反射 | ✅ 1st-32nd | ❌ Closed cloud | ⭐ 太贵，仅 golden dataset |
| **k-Wave / k-wave-python** | Pseudospectral wave | ✅ CUDA CC ≥7.5 | ❌ voxel only | ⚠ approximate | via post-proc | LGPL | ⭐（医学超声为主） |
| **PFFDTD** | FDTD (FCC 13-stencil) | ✅ 多 GPU CUDA | SketchUp→JSON voxel | ✅ 11-band 阻抗 fit | via 阵列 | MIT | ⭐⭐（研究级精确，工程重） |
| **edg-acoustics** | DG-FEM 高阶 | ❌ (roadmap) | ✅ tet (GMSH) | ✅ vector fitting | via 探针 | GPL-3 | ⭐（未来观察） |
| **Bempp-cl** | BEM 频域 | via PyOpenCL | ✅ 表面 tris | ✅ per-freq admittance | 任意场点 | MIT | ⭐（仅低频模态） |
| **j-Wave** | JAX 波动方程可微 | ✅ CUDA+TPU | voxel | approx | via post-proc | LGPL | ⭐⭐（可微性有 unique value） |

### 关键洞察

- **pyroomacoustics** 是**唯一 permissive-license + Python 原生 + 7-band material + arbitrary 3D room** 的开源选项 → **阶段 A 首选**
- **Steam Audio** 是**唯一活跃的 open-source game-audio SDK**（Apache-2.0），支持大规模 mesh 场景 → **阶段 B 首选**
- **神经方法（MESH2IR / xRIR / EigeNet）**：好用但需**先有物理 RIR 训练集**（AVEngine 正好是"生成物理 RIR"的角色，属于**这类神经模型的上游**，而不是消费者）
- **可微 RIR 生成器（j-Wave, DiffRIR）**：如果未来要做"从测量 RIR 反解房间材质"任务，可考虑；当前不做

**推荐 install**：
```bash
conda activate sao-env
pip install pyroomacoustics==0.10.1
# 验证：
python -c "import pyroomacoustics as pra; print(pra.__version__); m = pra.Material('carpet_cotton'); print(m.absorption_coeffs)"
```

---

## 2. GPURIR 支持室外吗？

**结论**：**不能**（ISM 要求闭合 shoebox）。

但可以两种 workaround：

### Workaround 1：GPURIR 单反射 hack
```python
gpuRIR.simulateRIR(
    beta=[0, 0, 0, 0, 0.9, 0],   # 仅 z=0 地面反射
    nb_img=[1, 1, 2],             # 只算 1 阶像源
    Tmax=0.2,                     # 短 RIR
    ...
)
```
- ⚠ 缺失：大气吸收、风/温度梯度折射、地面阻抗、衍射
- ✅ 保真度 ~50%（够 SPL-only baseline）

### Workaround 2：pyroadacoustics + 后置 FIR
```bash
pip install pyroadacoustics
```
提供：
- **Delany-Bazley 地面阻抗** filter
- **ISO 9613 大气吸收** filter（湿度/温度参数化）
- Doppler 效应
- ❌ 仅 omni mic — 4-ch encoder 得自写

**推荐路径**（若明确要做 outdoor scene）：
- **短期（1-2 天）**：GPURIR floor-only + 从 pyroadacoustics 借 100 行大气/地面 FIR 后置 → 80% 保真度
- **中期（2-4 天）**：完整 pyroadacoustics + 手写 4ch encoder → 生产级
- **需要建筑遮挡**（3D 城市街景）：Steam Audio open mesh + pyroadacoustics 大气 → 1-2 周

**神经 outdoor RIR：2025 完全空白**（DiffusionRIR / PromptReverb / PI-DANF 都是 indoor）。

---

## 3. Unreal Engine 5.5 原生 audio 能力

**结论**：UE 5.5 **原生不带 ray-traced acoustics**。MetaSounds / Attenuation / Spatialization 都是 perceptual，不是 physically-based。

### Steam Audio for UE（唯一 production 选项）
- 4.8.1 for UE, Apache-2.0，兼容 UE 4.27+
- `Steam Audio Geometry` component + `Steam Audio Material`（3 bands α + scattering）
- `Probe Volume` 预烘焙
- Bake ambisonic 到指定 order（FOA=4ch, 2nd=9ch, 3rd=16ch）
- **UE editor UI 不能导出 RIR 到 WAV/numpy** — 必须走 Steam Audio C API 4.6+ (`IPLImpulseResponse` + `IPLReconstructor`)
- **NVIDIA GPU 上走 Embree CPU**（Radeon Rays 是 AMD-only）

### 已死/deprecated
- **NVIDIA VRWorks Audio**：silently deprecated，最后 2.0 版停在 UE 4.15-4.25，无 CUDA 12 / OptiX 8 移植
- **Google Resonance Audio**：2022-12 archived
- **Microsoft Project Acoustics**：2024-07-01 discontinued
- **Meta XR Audio SDK**：Quest VR-focused，无 explicit RIR export

### 结论

**跳过 UE editor，直接用 Steam Audio C SDK headless**。UE 的价值只在 visual authoring；对 AVEngine 的 Python + GPURIR pipeline 无必要在 audio 侧引入 UE。

---

## 4. 类似 AV 数据集生成仓库

| 项目 | Engine | Audio | 状态 | License | 与 AVEngine 关系 |
|---|---|---|---|---|---|
| **SoundSpaces 2.0** | Habitat-Sim | RLR-Audio-Propagation | 2024-11 archived | CC-BY-NC | **算法蓝图**（不能商业 fork） |
| **SPEAR**（我们用的） | UE5 (Kujiale scans) | ❌ 无 | 上游 alive | MIT | AVEngine 的 visual backbone |
| **JAEGER**（对手） | Habitat + SoundSpaces | 4ch FOA | 2026-05 update | 未 release | 61k SpatialSceneQA 数据集 target 格式 |
| **CARLA** | UE5 | ❌ | alive | MIT | 无 research audio |
| **Isaac Sim 6.0** | Omniverse USD | 仅 ultrasonic | alive | proprietary | 无 research audio |
| **ThreeDWorld** | Unity | PyImpact (modal 撞击声) + Resonance | LTS since 2024 | BSD | 参考 physics-triggered contact sound |
| **iGibson / Sonicverse** | Bullet | audio module | 已弃 | MIT | historic |

**关键结论**：**AVEngine 占据一个未被填补的 niche** — *game-engine-native (UE5) + research-grade RIR + per-triangle acoustic materials*。**当前市场无直接同类**。SPEAR + GPURIR + Quaternius rigs 组合是唯一的。

---

## 5. 材质对声音的影响

### 声学材质核心量

- **吸声系数 α**：0-1，frequency-dependent。混凝土 0.02、地毯 0.4-0.6、木地板 0.05-0.15、玻璃 0.03-0.05
- **散射系数 s**：0-1，材质表面粗糙度 vs 波长比
- **透射系数 t**：材质另一侧能透过多少（隔声）
- **表面阻抗 Z**：更完整的物理量（复数）

### GPURIR 的表达能力
- **只有单个 T60**（Sabine 估计）→ **完全丢失 material spectrum**
- 所有墙壁一个 T60、所有频率一个 T60 → 现实中低频衰减慢/高频衰减快的自然效果**完全没有**

### 现代仿真器怎么建模

| 引擎 | Bands | Per-triangle? | 材质 DB |
|---|---|---|---|
| pyroomacoustics 0.10.1 | **7 octave** (125-8k Hz) | ✅ per-wall | **Vorländer 2008** ~80 材质内置 |
| Steam Audio | 3 fixed (低/中/高) | ✅ per-triangle | 内置 [Concrete/Carpet/Glass/Wood/…] |
| SoundSpaces 2.0 | 4 configurable | ✅ | `mp3d_material_config.json` |
| ODEON / CATT / EASE | 8 octave + scattering | ✅ per-face | 各自数据库 |

### glTF 里能存材质吗？

- **无标准 acoustic material glTF extension**（所有 `KHR_materials_*` 都是光学 PBR）
- **MPEG-I Scene Description** (ISO/IEC 23090-14) 是最接近的正式标准，材质在 sidecar EIF 文件里
- **De facto 实践**：mesh 保留 material name → project JSON 查表

### 软 vs 硬物体（沙发 vs 桌子）

**所有引擎都只用 surface boundary condition** —— 沙发就是一层 "fabric" 高 α 三角面，内部还是空气。**真正的体积吸声（Delany-Bazley / JCA）只在专业 FEM/BEM 里**（Bempp-cl 那种低频模态用的）。

### AV-RIR (CVPR 2024) 实验结论

**显式材质特征使 RIR 估计误差降 41-55%** —— 证明材质建模是最大的 ROI。

### 推荐（分层）

- **L1 (1-2 周)**：per-face material tag → pyroomacoustics `materials.json` → per-band Sabine T60 → GPURIR 每 band 独立跑再合成。伪造但立刻带出频谱倾斜
- **L2 (3-6 周)**：换 pyroomacoustics hybrid ISM+RT，octave-band α + scattering，2× slower but 真物理
- **L3 (6-10 周)**：Steam Audio 集成，per-triangle 3-band + spatialization + occlusion
- **L4 (3-6 月)**：DiffRIR 从实测 RIR 反解材质系数

---

## 6. AVEngine 6 个月升级路径（综合结论）

### 阶段 A（2 周）：pyroomacoustics 替换 + 材质系统

**做什么**：
1. 抛弃 GPURIR 的单 T60 假设，换 pyroomacoustics hybrid ISM+RT
2. 引入 per-face material tag → Vorländer `materials.json` 查表 → 7-octave-band α + scattering
3. AVEngine 提供 material tag → 材质 DB 的映射（比如 `apartment_furniture_map.json` 里每件家具带 material_type: "sofa_fabric" | "wood_table" | ...）

**ROI**：单独就把 RIR realism 拉到接近 SoundSpaces 2.0 水平（GA + per-band materials + arbitrary polygon rooms），MIT license、纯 Python、无 GPU 依赖、无 cloud。**AV-RIR 论文数据显示这一步能带来 41-55% RIR 估计误差下降**。

### 阶段 B（3-4 周）：Steam Audio C SDK headless + FOA↔tetrahedral

**做什么**：
- 用于需要 spatialization、occlusion、diffraction、更真实场景几何的 downstream 任务
- 保留 pyroomacoustics 作为"轻量快速 batch"路径，Steam Audio 作为"高保真 hero shot"路径 → **dual backend**
- 顺带解决室外（Steam Audio 支持 open mesh + 加 pyroadacoustics 的 ISO 9613 大气 FIR 后置）

### 明确不做（6 个月内）

- ❌ 引入 Unreal Engine 到 runtime audio pipeline（Steam Audio headless 已够，UE 只是 visual authoring 皮）
- ❌ Treble SDK（cloud + 贵，只有发论文对比时值得）
- ❌ FDTD / BEM（可听频段 outdoor 不可行；indoor 也是 5-100× 慢，overkill）
- ❌ 神经 RIR (MESH2IR / NAF)（作为 pipeline consumer 而非生成器；等 pipeline 稳后再考虑加速）
- ❌ 全部押注 RLR-Audio-Propagation（**CC-BY-NC 商业禁用是 dealbreaker**）

### 一句话结论

> **把 GPURIR 的 shoebox + 单 T60 替换成 pyroomacoustics hybrid + Vorländer per-band materials（2 周），再叠加 Steam Audio headless C SDK 作为 hero-shot 后端（3-4 周），是 6 个月内 ROI 最高的音频仿真升级路径。**

---

## 附录：如果 AVEngine 想做的是别的音频任务

| 任务 | 推荐工具 |
|---|---|
| **可微分 audio inverse problems**（从声音反解房间） | j-Wave（JAX） |
| **实测 RIR → 反解每面墙材质** | DiffRIR (CVPR 2024) |
| **零-shot mesh → RIR（无 measurement）** | MESH2IR (2022 SOTA, mono only) |
| **few-shot 跨房间 RIR（K=1-8 参考）** | xRIR / EigeNet (2025-2026 SOTA) |
| **纯文本 prompt → RIR** | PromptReverb (ICASSP 2026) |
| **接触声/敲击声合成** | ThreeDWorld PyImpact |
| **物理级低频模态分析** | Bempp-cl (BEM 频域) |
| **超高精度 golden dataset**（发论文 baseline 用） | Treble (cloud, 昂贵) |
