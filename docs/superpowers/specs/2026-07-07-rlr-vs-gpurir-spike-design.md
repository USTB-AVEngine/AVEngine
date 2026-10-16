# RLR vs GPURIR 三方对比 Spike — 设计文档

> 日期：2026-07-07
>
> Scope：**只**在一个手写 shoebox 场景上比较三个 audio-visual backend：
> A (SPEAR/UE + GPURIR)、B (SPEAR/UE + RLR via 档 ① swap-in)、C (Habitat + RLR)。
> 目标是**为"要不要迁 RLR"这个决策收集证据**，不是造 dataset。
>
> **明确不做**：ReplicaCAD/HSSD/3D-FRONT 集成 · 大批量 dataset 生成 · Steam Audio ·
> 骨骼动画在 Habitat 里跑 (C 组用 T-pose "滑冰") · 室外 · 材质 sweep · 视觉细节调优。

---

## 1. 问题陈述

AVEngine 当前音频侧走 GPURIR（shoebox + Sabine T60 + ISM，无材质频谱、无遮挡、无衍射），
参见 [audio_simulation_landscape_2026.md](../../audio_simulation_landscape_2026.md)。

用户目标 task 需要：
- 移动源穿越遮挡物（狗从沙发前走到沙发后完全被挡）
- 相机后方声源的方位定位（听得出 L→R 移动）
- 材质对 audio 的频谱倾斜

GPURIR 无法覆盖任何一条。前期调研得出候选后端 = **RLR-Audio-Propagation**（Meta Reality Labs
出品，SoundSpaces 2.0 用的核心，CC-BY-NC research OK），接入方式候选 = **档 ① swap-in**
（`import habitat_sim` 只用 `AudioSensor`，不改视觉端）。

本 spike 要回答：
1. **RLR 在遮挡时刻真的比 GPURIR 强吗？** 数值上差多少 dB？
2. **档 ① swap-in 走得通吗？** 现有 UE 视频 + RLR audio 能 mux 出可播放的带音频 mp4？
3. **单场景耗时可接受吗？** ≤ 5 分钟/场景 → dataset scale 才可行
4. **FOA→stereo 下听感如何？** 耳机能听出相机后方 golden 的 L→R 移动？

---

## 2. Spike 结构

```
一个场景 (shoebox v2)
    ↓
三个 backend 并行跑
    ├── A. SPEAR/UE (auto-gen Level) + GPURIR (现有)
    ├── B. SPEAR/UE (同 A 视频) + RLR (via `import habitat_sim` 档 ①)
    └── C. Habitat (Python 手写 mesh, T-pose 狗) + RLR (原生)
    ↓
每 backend 出 4 视角带音频 mp4 + FOA raw wav
    ↓
分析：spectrogram + DRR/RT60 + IR energy curve
    ↓
判据 (4 gate): 遮挡 ≥3dB / 档 ① 走通 / ≤5min / 耳机听得出 L→R
    ↓
DECISION_TABLE.md → 后续路线决策
```

**为什么不做 ReplicaCAD 组？**
UE 侧全自动导入 ReplicaCAD 需要 2-4 天工程（scene JSON parser + GLB import automation +
坐标系对齐 debug + 材质对齐），远超 spike ROI。shoebox 已能证明"RLR 遮挡建模 vs
GPURIR"这个核心结论。ReplicaCAD 是 spike 通过后 dataset 阶段的事。

---

## 3. 场景规格 (shoebox v2)

### 3.1 房间几何

| 项 | 值 | 备注 |
|---|---|---|
| Room size | (5.2, **6.0**, 2.8) m | Y 从现有 4.4 加深到 6.0 给 husky 绕行留空间 |
| Mic + view0 camera | (2.6, 2.2, 1.2) m 朝 +Y | 沿用现有 shoebox 约定 |

### 3.2 遮挡物

| 项 | 值 |
|---|---|
| Sofa 中心 | (2.6, 3.45, 0.45) m |
| Sofa 尺寸 | 2.0 × 0.9 × 0.9 m (X × Y × Z)，长边沿 X |
| Sofa 占据 | X:[1.6, 3.6], Y:[3.00, 3.90], Z:[0, 0.9] |
| Sofa 材质 | `fabric_upholstery`（Vorländer α: [0.15, 0.30, 0.60]) |

### 3.3 声源

| 声源 | 位置 | 音频 | 时长 |
|---|---|---|---|
| **Golden**（相机后 L→R 匀速移动）| (0.8, 1.5, 0.45) → (4.4, 1.5, 0.45) | OmniAudio 里 `dog_bark`（中高频 crisp）| 5s 全程连续 |
| **Husky**（绕沙发到后方，4 段）| 见 3.4 | OmniAudio 里 `wolf_howl`（低频 howl，与 golden 音色明显区分）| 5s 全程连续 |

**为什么 golden 用 dog bark 而 husky 用 wolf howl？**
用户 Q4a-1 明确要求"两只狗声音不同，不然听不出来"。现有 [audio_registry.py](../../../external/SPEAR/tools/gpurir_scenes/audio_registry.py)
的 tag→keyword 映射本来就把 `dog_husky` 映射到 wolf howl，直接沿用零改动。

### 3.4 Husky 4 段绕行 trajectory (75 帧 / 5s / 15 fps)

| 阶段 | 帧 | 时间 | 起终点 | 描述 |
|---|---|---|---|---|
| A | 0-19 (20 帧, 1.33s) | 0-1.33s | (2.6, 2.5) → (3.9, 2.5) | 侧移 +X，可见 |
| B | 20-44 (25 帧, 1.67s) | 1.33-3.00s | (3.9, 2.5) → (3.9, 4.5) | 上行 +Y 沿沙发右侧，可见 |
| C | 45-59 (15 帧, 1.00s) | 3.00-4.00s | (3.9, 4.5) → (2.6, 4.5) | 侧移 -X，**遮挡渐进期** |
| D | 60-74 (15 帧, 1.00s) | 4.00-5.00s | (2.6, 4.5) 静止 | **完全遮挡稳态** |

**关键遮挡事件时刻**：
- 帧 45 (t=3.0s)：husky 越过 sofa 右边缘 (X=3.6)，开始被遮挡
- 帧 60 (t=4.0s)：husky 完全在 sofa 阴影里，稳态遮挡

Husky 终点视觉遮挡验证：mic (2.6, 2.2, 1.2) → husky (2.6, 4.5, 0.45) 视线在 sofa 中心
Y=3.45 处 z=0.79m < sofa 顶 0.9m → 视线被 sofa 完全阻挡。

### 3.5 声学材质配置

| 表面 | material_tag | α [125-8000 Hz, 4 band] | 来源 |
|---|---|---|---|
| 4 面墙 | `drywall_painted` | [0.05, 0.10, 0.10, 0.15] | Vorländer 2008 |
| 地板 | `hardwood_oak` | [0.10, 0.07, 0.06, 0.06] | Vorländer 2008 |
| 天花板 | `painted_plaster` | [0.14, 0.10, 0.06, 0.04] | Vorländer 2008 |
| Sofa | `fabric_upholstery` | [0.15, 0.30, 0.45, 0.60] | Vorländer 2008 |

Scattering + transmission 系数用 RLR 默认 (0.05 / 0.0)，spike 阶段不 sweep。

---

## 4. 三个 Backend 的技术选型

### 4.1 A 组：SPEAR/UE + GPURIR（baseline，现有 pipeline）

- **视频**：`run_render_pass.py` 现有代码，改一次 room size 常量
- **音频**：`run_audio_pass.py` 现有 GPURIR，T60 = Sabine(sofa material 摊 α 加权)
- **输出**：4ch 四面体 → stereo downmix (FL = 0.5·c0 + 0.5·c2, FR = 0.5·c1 + 0.5·c3)

### 4.2 B 组：档 ① swap-in — 用 A 组视频 + RLR 算 audio

- **视频**：**完全复用 A 组的 view*.mp4**（不重算 UE 视频）
- **音频**：新写 `run_audio_pass_rlr.py`
  - `import habitat_sim`（新 conda env `hab-env` 装 habitat-sim with `HABITAT_WITH_AUDIO=ON`）
  - 只 spawn 一个不带 render sensor 的 Simulator，唯一挂 AudioSensor
  - 每帧 `audio_sensor.setAudioSourceTransform(source_pos)` → `sim.get_sensor_observations()["audio"]`
  - 输出 FOA 4ch (SphericalHarmonics order=1)
- **RLR 输入 mesh**：从共享 SSOT (`shoebox_v2_spec.json`) 用 `trimesh` 独立生成 GLB
- **RLR 参数**（SoundSpaces 2.0 high-quality 预设）：
  - 4 frequency bands (125-8000 Hz logarithmic)
  - ray count 5000, max bounces 50
  - IR length 4000 samples (250ms @ 16kHz)
  - direct + indirect on, diffraction on, transmission on

### 4.3 C 组：整栈 Habitat + RLR

- **视频**：habitat-sim `RGBSensor`，4 视角 (view0-3, yaw 0/90/180/270°)，Quaternius Dog GLB
  以 **static T-pose 每帧 set_translation()**（"滑冰"）
- **音频**：同 B 组 RLR 逻辑，只是不需要 dump mesh（Habitat 场景 mesh 直接吃）
- **同一份 shoebox_v2_spec.json** 生成 mesh，与 B 组材质对齐

---

## 5. 共享数据契约 (SSOT)

**核心原则**：UE、Habitat、RLR、三个 backend 都读同一份 spec 文件，杜绝手工同步。

### 5.1 `shoebox_v2_spec.json`

```json
{
  "spec_version": "v2",
  "room_size_m": [5.2, 6.0, 2.8],
  "mic_pos_m": [2.6, 2.2, 1.2],
  "mic_forward": [0.0, 1.0, 0.0],
  "camera_configs": [
    {"name": "view0", "pos_m": [2.6, 2.2, 1.2], "yaw_deg": 90},
    {"name": "view1", "pos_m": [2.6, 2.2, 1.2], "yaw_deg": 180},
    {"name": "view2", "pos_m": [2.6, 2.2, 1.2], "yaw_deg": 270},
    {"name": "view3", "pos_m": [2.6, 2.2, 1.2], "yaw_deg": 0}
  ],
  "surfaces": {
    "wall_north": "drywall_painted",
    "wall_south": "drywall_painted",
    "wall_east":  "drywall_painted",
    "wall_west":  "drywall_painted",
    "floor":      "hardwood_oak",
    "ceiling":    "painted_plaster"
  },
  "furniture": [
    {
      "name":  "sofa",
      "shape": "box",
      "center_m": [2.6, 3.45, 0.45],
      "size_m":   [2.0, 0.9, 0.9],
      "material": "fabric_upholstery"
    }
  ],
  "sources": [
    {"tag": "dog_golden", "audio_lookup": "dog_bark"},
    {"tag": "dog_husky",  "audio_lookup": "wolf_howl"}
  ]
}
```

### 5.2 `acoustic_material_db.json`

```json
{
  "drywall_painted":   {"alpha": [0.05, 0.10, 0.10, 0.15], "scat": 0.05, "trans": [0, 0, 0, 0]},
  "hardwood_oak":      {"alpha": [0.10, 0.07, 0.06, 0.06], "scat": 0.05, "trans": [0, 0, 0, 0]},
  "painted_plaster":   {"alpha": [0.14, 0.10, 0.06, 0.04], "scat": 0.05, "trans": [0, 0, 0, 0]},
  "fabric_upholstery": {"alpha": [0.15, 0.30, 0.45, 0.60], "scat": 0.20, "trans": [0.05, 0.10, 0.20, 0.30]}
}
```

### 5.3 数据流

```
shoebox_v2_spec.json + acoustic_material_db.json
        │
        ├──> Python trimesh gen_mesh.py ──> shoebox_v2.glb (RLR 吃)
        │                                          │
        │                                          ├──> B 组 RLR audio_pass_rlr.py
        │                                          └──> C 组 Habitat RLR
        │
        └──> UE Editor Python build_shoebox_v2_umap.py ──> Content/Levels/shoebox_v2.umap
                                                                      │
                                                                      └──> A/B 组 render_pass 出 view*.mp4
```

---

## 6. 交付物

`spike_output/` 目录布局：

```
spike_output/
├── videos/
│   ├── A_gpurir_view0.mp4  ...  A_gpurir_view3.mp4     # A 组 4 视角带音频 mp4
│   ├── B_rlr_view0.mp4     ...  B_rlr_view3.mp4        # B 组 4 视角带音频 mp4
│   └── C_habitat_view0.mp4 ...  C_habitat_view3.mp4    # C 组 4 视角带音频 mp4
├── raw_audio/
│   ├── audio_A_gpurir_FOA.wav      # 4ch FOA raw
│   ├── audio_B_rlr_FOA.wav
│   └── audio_C_habitat_FOA.wav
├── analysis/
│   ├── spectrogram.png             # 3-row subplot, 标遮挡时刻虚线
│   ├── ir_energy_curve.png         # trajectory 逐帧 IR RMS energy dB
│   └── metrics.json                # DRR / RT60 / occlusion drop dB
└── DECISION_TABLE.md               # 一目了然的对比表 + 推荐路线
```

### 6.1 Spectrogram 布局

3 行 subplot（A/B/C），每行是 stereo L+R spectrogram，横轴 0-5s、纵轴 0-8kHz。
两条垂直虚线标 t=3.0s（遮挡开始）和 t=4.0s（完全遮挡）。RLR 组高亮高频衰减区域。

### 6.2 决策表 columns

| 维度 | A. GPURIR | B. RLR swap-in | C. Habitat+RLR |
|---|---|---|---|
| 遮挡时刻能量下降 (dB) | | | |
| 频谱倾斜是否合理 | ✓/✗ | | |
| 人耳能否听出遮挡瞬间 | ✓/✗ | | |
| 人耳能否听出 golden L→R | ✓/✗ | | |
| 视觉质量 (1-5) | | | |
| 单场景耗时 (s) | | | |
| 集成难度 (1-5) | 已投产 | 3-5 天 | 2-3 月 |
| **推荐用于** | | | |

---

## 7. Go/No-Go 判据（4 条 gate）

**Gate 1: 遮挡建模真的有区别** — t=3.0-4.0s 之间 B 组和 A 组的 audio energy 差异 ≥ 3 dB
- 通过 = RLR 带来 GPURIR 没有的信号
- 不通过 = 不值得迁 RLR，保 A 组

**Gate 2: 视觉可保留（档 ① swap-in 成立）** — B 组 mux 出可播放的带音频 mp4，audio 空间感与视频对齐
- 通过 = 视觉端不用动就能上 RLR
- 不通过 = 只能走 C 组整栈迁 Habitat（成本 ×10）

**Gate 3: 单场景耗时可接受** — B 组端到端（不含 UE Cook）≤ 5 分钟/场景
- 通过 = dataset 10k 场景可以在一周内跑完
- 不通过 = 需要额外性能工程（分布式 / high-speed mode）

**Gate 4: 相机后方声源能听出方位** — 耳机盲听 stereo downmix，能明显听出 golden L→R
- **B 和 C 必须过**（未来 dataset 走 FOA 输出，FOA→stereo decode 必须正确）
- A 参考（现有四面体→stereo downmix 不是标准 FOA decode，A 不过只暴露 downmix 逻辑问题）

**全过 → 后续路线 = "档 ① swap-in + 视觉资产扩展"**
**Gate 1 不过 → 保 A 组 GPURIR**
**Gate 1 过但 Gate 2 不过 → 重估要不要迁整栈 Habitat**

---

## 8. 明确不做

- ❌ ReplicaCAD / HSSD / 3D-FRONT 集成（spike 通过后 dataset 阶段做）
- ❌ 骨骼动画在 Habitat 里跑（C 组 T-pose 滑冰）
- ❌ Steam Audio ctypes wrapper（RLR 覆盖 Steam Audio 大部分需求）
- ❌ 室外（用户已确认不做）
- ❌ 材质 sweep（spike 用固定 4 材质）
- ❌ 多房间连通
- ❌ 人物 / 桌面物件资产（Query 3/4 场景是 dataset 阶段的事）
- ❌ 手工步骤（UE Editor Python 自动生成 Level，全流程可脚本化）
- ❌ 房间几何 sweep（spike 只 1 个场景）
