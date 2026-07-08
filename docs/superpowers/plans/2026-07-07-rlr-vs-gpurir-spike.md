# RLR vs GPURIR 三方对比 Spike — 实现 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在一个 shoebox v2 场景上做 A/B/C 三方对比（SPEAR+GPURIR / SPEAR+RLR / Habitat+RLR），
产出 12 段带音频 mp4 + spectrogram + 决策表，判断"是否迁 RLR"。

**Architecture:** 一份 shoebox_v2_spec.json + acoustic_material_db.json 作为 SSOT。Python
trimesh 独立生成 mesh 给 RLR 吃；UE Editor Python 自动生成 Level 给视频端。B 组走档 ①
swap-in（`import habitat_sim` 只用 AudioSensor），A 组视频直接复用。C 组整栈 Habitat，
Quaternius Dog 用 T-pose "滑冰" 移动。

**Tech Stack:** Python 3.11 + SPEAR RPC + numpy + trimesh + habitat-sim (audio-enabled)
+ UE 5.5 Editor Python + matplotlib。conda envs:
- `spear-env`：SPEAR RPC + UE 视频渲染
- `sao-env`：GPURIR + audio pipeline
- `hab-env` （**新建**）：habitat-sim with `HABITAT_WITH_AUDIO=ON`

## Global Constraints

- Python 解释器：分 env 用 —— `/data/jzy/miniconda3/envs/{spear,sao,hab}-env/bin/python`
- 启动 UE 前必须 `export DISPLAY=:99` 和 `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`
- **全自动化，无手工步骤**（用户 Q7 明确要求）
- shoebox v2 的**所有几何 / 材质 / trajectory 参数**都从共享 SSOT 读，禁止硬编码
- 输出规格：4 视角 × 5s × 15 fps × 640×480 (h264) + AAC stereo，与现有 pipeline 一致
- 采样率：16 kHz
- 音频格式：4ch FOA (Ambisonic 1st order)
- 每 Task 独立 commit，每 Task 有可运行验证命令
- 参考[设计文档](../specs/2026-07-07-rlr-vs-gpurir-spike-design.md)

---

## File Structure

**新增**：
- `data/shoebox_v2_spec.json` — 场景 SSOT
- `data/acoustic_material_db.json` — 材质 → RLR 参数映射
- `tools/spike_rlr/gen_mesh.py` — trimesh 从 spec 生成 shoebox v2 GLB
- `tools/spike_rlr/build_shoebox_v2_umap.py` — UE Editor Python 自动生成 Level
- `tools/spike_rlr/scene_two_dogs_v2.py` — 新场景（加深房间 + sofa + husky 绕行 + golden L→R）
- `tools/spike_rlr/run_audio_pass_rlr.py` — B 组 RLR audio pass（档 ①）
- `tools/spike_rlr/run_habitat_all.py` — C 组整栈 Habitat（视频 + audio）
- `tools/spike_rlr/analysis/spectrogram_gen.py` — spectrogram 3-row 图
- `tools/spike_rlr/analysis/ir_energy_curve.py` — trajectory IR energy 图
- `tools/spike_rlr/analysis/metrics.py` — DRR / RT60 / occlusion drop 计算
- `tools/spike_rlr/analysis/build_decision_table.py` — 决策表 markdown
- `tools/spike_rlr/run_all.sh` — 一键跑 A/B/C 三个 backend + 分析
- `tests/spike_rlr/test_ssot.py` — SSOT JSON schema + 一致性验证
- `tests/spike_rlr/test_gen_mesh.py` — trimesh 生成 mesh 的几何验证

**修改**：
- 无（本 spike 全部走新目录 `tools/spike_rlr/`，不侵入现有 pipeline）

**产物**：
- `tmp/spike_output/videos/` — 12 段带音频 mp4
- `tmp/spike_output/raw_audio/` — 3 段 FOA 4ch wav
- `tmp/spike_output/analysis/` — spectrogram + ir energy + metrics.json
- `tmp/spike_output/DECISION_TABLE.md` — 最终决策表

---

## Task 1: SSOT — shoebox_v2_spec.json + acoustic_material_db.json

**目的**：建立所有 backend 都读的单一 spec，杜绝硬编码分歧。

### 实现

- [ ] 1.1 写 `data/shoebox_v2_spec.json`，字段见 [设计 §5.1](../specs/2026-07-07-rlr-vs-gpurir-spike-design.md#51-shoebox_v2_specjson)
- [ ] 1.2 写 `data/acoustic_material_db.json`，字段见 [设计 §5.2](../specs/2026-07-07-rlr-vs-gpurir-spike-design.md#52-acoustic_material_dbjson)
- [ ] 1.3 写 `tests/spike_rlr/test_ssot.py`：
  - JSON schema 校验（所有必需字段存在）
  - `shoebox_v2_spec.surfaces.*` 里出现的每个 material 都在 `acoustic_material_db` 里
  - `furniture[].material` 同上
  - room_size / sofa_size / sofa_center 保证 sofa 完全在 room 内 + wall clearance ≥ 0.5m

### 验证

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/spike_rlr/test_ssot.py -v
```

---

## Task 2: gen_mesh.py — Python trimesh 生成 shoebox v2 GLB

**目的**：从 SSOT 生成一份 GLB，供 RLR (B 组和 C 组) 吃。UE 端另外走 Task 3。

### 实现

- [ ] 2.1 `tools/spike_rlr/gen_mesh.py` 读 shoebox_v2_spec.json：
  - 6 面 wall（每面 2 三角形，OBJ-style），material_index 指向 surfaces.wall_north 等
  - sofa 6 面 box（12 三角形），material_index 指向 fabric_upholstery
  - 每三角形一个 material index
  - 输出 `tmp/spike_rlr/shoebox_v2_mesh.glb`（trimesh 或 pygltflib）
- [ ] 2.2 顺带生成 `tmp/spike_rlr/shoebox_v2_materials.json`（RLR 侧格式）：
  - 每 material_index → {alpha[4], scat, trans[4]}
  - 直接从 acoustic_material_db 抄
- [ ] 2.3 `tests/spike_rlr/test_gen_mesh.py`：
  - GLB 加载后 vertex/triangle count 正确
  - 每三角形都有 material index
  - Room 内部体积 = room_size 乘积 - sofa 体积（几何合理性）

### 验证

```bash
/data/jzy/miniconda3/envs/sao-env/bin/python tools/spike_rlr/gen_mesh.py
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/spike_rlr/test_gen_mesh.py -v
# 用 Blender headless / meshlab / gltf-validator 眼验 shoebox_v2_mesh.glb
```

---

## Task 3: UE Editor Python — 自动生成 shoebox_v2.umap

**目的**：SPEAR/UE 侧的视觉端。Level 完全脚本生成，无手工。

### 实现

- [ ] 3.1 `tools/spike_rlr/build_shoebox_v2_umap.py`（UE Editor Python）：
  - 读 shoebox_v2_spec.json
  - 用 `unreal.EditorLevelLibrary.new_level()` 建空 Level
  - 用 `unreal.StaticMeshEditorSubsystem` / `EditorAssetLibrary` 造 6 面 wall（每面用 UE 内置 Cube brush 或 procedural mesh，缩放到指定尺寸）
  - Interchange 自动 import `shoebox_v2_mesh.glb` 里的 sofa 部分作为 StaticMesh asset
  - 或者手动用 `unreal.EditorAssetLibrary.import_asset_tasks()` 只 import sofa
  - Spawn StaticMeshActor 摆到 (2.6, 3.45, 0.45)（记得 UE cm 单位转换 + 坐标系 Y-flip）
  - 加环境光（`SkyLight` + `DirectionalLight`）
  - `unreal.EditorLevelLibrary.save_current_level()` 到 `/Game/SPEAR/Scenes/shoebox_v2/Maps/shoebox_v2.umap`
- [ ] 3.2 运行命令：
  ```bash
  export DISPLAY=:99
  UE5.5/Engine/Binaries/Linux/UnrealEditor-Cmd \
      $SPEAR_PROJECT/SpearSim.uproject \
      -run=PythonScript -script=tools/spike_rlr/build_shoebox_v2_umap.py
  ```
- [ ] 3.3 Cook Level 到 Standalone runtime 可用：
  - 复用现有 `build_static_meshes.sh` 的 UAT 命令模式
  - 或者直接 UnrealEditor-Cmd `-run=Cook -Map=/Game/.../shoebox_v2`

### 验证

```bash
ls -la $SPEAR_PROJECT/Content/SPEAR/Scenes/shoebox_v2/Maps/shoebox_v2.umap
# 用 UnrealEditor 交互模式打开一眼看 sofa 位置对不对（仅本次 spike 允许人肉验一次）
```

**注**：Task 3.1 是本 plan 风险最高的一步（UE Editor Python 有各种版本兼容问题）。如果
UE 5.5 Editor Python API 不能直接造 procedural wall，退化方案 = 提前在 UE Editor 里手工
造一个 `shoebox_v2_wall.uasset`（一个 UE Cube StaticMesh），Python 只负责 spawn actor +
缩放到 spec 指定尺寸。这是本 plan 唯一允许的一次性手工产物。

---

## Task 4: scene_two_dogs_v2.py — 新场景定义（复用现有结构）

**目的**：把[现有 scene_two_dogs.py](../../../external/SPEAR/tools/gpurir_scenes/scene_two_dogs.py)
的手写场景改造成 shoebox v2 版本（加深房间 + sofa + husky 绕行 + golden L→R）。

### 实现

- [ ] 4.1 `tools/spike_rlr/scene_two_dogs_v2.py`：
  - 参数从 `shoebox_v2_spec.json` 读，不硬编码
  - Husky trajectory 4 段：
    - 帧 0-19：(2.6, 2.5) → (3.9, 2.5)
    - 帧 20-44：(3.9, 2.5) → (3.9, 4.5)
    - 帧 45-59：(3.9, 4.5) → (2.6, 4.5)
    - 帧 60-74：(2.6, 4.5) 静止
  - Golden trajectory 全程匀速：(0.8, 1.5) → (4.4, 1.5)
  - Golden `is_animated=True`（复用 Quaternius Dog Walking），但速度 0.72 m/s
  - Body yaw 按现有 `_ANIM_FORWARD_YAW_OFFSET` 规则
- [ ] 4.2 复用现有 `check_no_clipping` 做几何合法性验证：
  - Wall margin 0.5m
  - Golden footprint 0.45m vs Husky footprint 0.45m 无同帧碰撞
  - Golden vs Husky trajectory 无 same-frame clipping

### 验证

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -c "
from tools.spike_rlr.scene_two_dogs_v2 import compose_two_dog_scene_v2
spec = compose_two_dog_scene_v2()
print('OK', spec.room_size_m, len(spec.animals))
"
```

---

## Task 5: A 组 — SPEAR/UE + GPURIR baseline

**目的**：用 shoebox v2 场景跑现有 pipeline 出 4 视角带音频 mp4。这是**其他两组的对照锚点**。

### 实现

- [ ] 5.1 `tools/spike_rlr/run_A_gpurir.sh`：
  - 调 `scene_two_dogs_v2.compose_two_dog_scene_v2()`
  - 调现有 `run_render_pass` 出 view0-3.mp4（走 shoebox_v2 UE Level）
  - 调现有 `run_audio_pass` 出 audio.wav（GPURIR，T60 用材质加权 Sabine 估计）
  - 调现有 `mux_audio_video.py` 出 `A_gpurir_view*_with_audio.mp4`
- [ ] 5.2 拷贝到 `tmp/spike_output/videos/A_gpurir_view*.mp4`
- [ ] 5.3 audio raw 保存 `tmp/spike_output/raw_audio/audio_A_gpurir_FOA.wav`
  - **注**：A 组的 raw 是四面体 4ch，不是标准 FOA。文件名标 `_FOA` 只是命名统一，实际里面是 tetrahedral A-format。分析脚本读取时按 backend 区分处理。

### 验证

```bash
bash tools/spike_rlr/run_A_gpurir.sh
ls tmp/spike_output/videos/A_gpurir_view*.mp4  # 4 files
ffprobe tmp/spike_output/videos/A_gpurir_view0.mp4  # 有 audio + video
```

---

## Task 6: hab-env 建 + habitat-sim with audio 装

**目的**：为 B 组和 C 组准备 conda env。

### 实现

- [ ] 6.1 建 conda env：
  ```bash
  conda create -n hab-env python=3.11 -y
  conda activate hab-env
  ```
- [ ] 6.2 装依赖：
  ```bash
  # habitat-sim with audio - 从 source 编译
  HABITAT_WITH_AUDIO=ON pip install habitat-sim --no-build-isolation
  # 或走 conda：conda install -c aihabitat habitat-sim withaudio
  ```
  fallback: 从 github 克隆 habitat-sim + build from source 走 `python setup.py install --with-audio`
- [ ] 6.3 装辅助库：`numpy scipy soundfile matplotlib trimesh`
- [ ] 6.4 冒烟测试：
  ```bash
  /data/jzy/miniconda3/envs/hab-env/bin/python -c "
  import habitat_sim
  spec = habitat_sim.AudioSensorSpec()
  print('OK', habitat_sim.__version__)
  "
  ```

### 验证

上面冒烟测试打印 `OK` 且不报错。

**风险**：habitat-sim 有 Magnum + pybind11 版本敏感问题，可能需要 debug 半天到一天。如果
上游 pip install 失败，退化到 conda install，再退化到从源码编译。

---

## Task 7: B 组 — 档 ① swap-in RLR audio backend

**目的**：复用 A 组 UE 视频 + 新写 RLR audio → mux。

### 实现

- [ ] 7.1 `tools/spike_rlr/run_audio_pass_rlr.py`（在 hab-env 里跑）：
  - 读 shoebox_v2_spec.json + shoebox_v2_mesh.glb + shoebox_v2_materials.json
  - 建 `habitat_sim.SimulatorConfiguration()`，`scene_id = shoebox_v2_mesh.glb`
  - 挂唯一一个 `AudioSensor`（不挂 render sensor）
    - `channel_layout = SphericalHarmonics(order=1)`
    - RLR params: SoundSpaces 2.0 high-quality preset（4 band, 5000 rays, 50 bounces, IR len 4000, direct+indirect on, diffraction on, transmission on）
  - 读 scene_two_dogs_v2 的 trajectory
  - 循环 75 帧：
    - `sim.set_audio_source_transform("golden", golden_pos[t])`
    - `sim.set_audio_source_transform("husky", husky_pos[t])`
    - `ir = sim.get_sensor_observations()["audio"]` (4ch FOA)
    - crossfade 相邻帧 IR
  - 卷 dry source（audio_registry 拿到的 dog_bark + wolf_howl）
  - 输出 `audio_B_rlr_FOA.wav`（4ch）
- [ ] 7.2 `tools/spike_rlr/foa_to_stereo.py`（标准 FOA→stereo decode）：
  ```
  L = W + 0.707*Y
  R = W - 0.707*Y
  ```
  （虚拟左右 cardioid 麦，指向 ±90° azimuth）
- [ ] 7.3 mux：
  ```bash
  # 用 A 组的 view*.mp4 视频轨 + B 组的 stereo audio
  ffmpeg -i A_gpurir_view0.mp4 -i audio_B_rlr_stereo.wav \
      -c:v copy -c:a aac -map 0:v -map 1:a \
      B_rlr_view0.mp4
  ```
  对 4 个 view 循环
- [ ] 7.4 记录耗时（wall clock），存进 metrics.json 供 Gate 3 判据用

### 验证

```bash
bash tools/spike_rlr/run_B_rlr.sh
ls tmp/spike_output/videos/B_rlr_view*.mp4  # 4 files
ls tmp/spike_output/raw_audio/audio_B_rlr_FOA.wav  # 4ch
# 用 headphone 播 B_rlr_view0.mp4，验 golden L→R 能听出
```

---

## Task 8: C 组 — 整栈 Habitat + RLR

**目的**：C 组不复用 UE 视频，Habitat 出视频 + audio 双轨。

### 实现

- [ ] 8.1 `tools/spike_rlr/run_habitat_all.py`（hab-env 里跑）：
  - 加载 shoebox_v2_mesh.glb 到 habitat Scene
  - Spawn 2 个 Quaternius Dog GLB（`assets/mesh_library/quaternius_animalpack/Dog.glb`）
    作为 rigid object：
    - Golden：每帧 `set_translation(golden_pos[t])` + `set_rotation(yaw_face_forward)`
    - Husky：同上，T-pose（不播动画，"滑冰")
  - 挂 4 个 `RGBSensor`（view0-3, yaw 0/90/180/270）+ 1 个 `AudioSensor`
  - 循环 75 帧：
    - 更新 dogs 位姿
    - `sim.get_sensor_observations()`：拿 4 张 RGB frame + 1 段 IR
    - 拼成视频（imageio 或 opencv）
  - 结束后：卷 dry source → audio_C_habitat_FOA.wav
- [ ] 8.2 mux 到 4 个 view mp4：`C_habitat_view*.mp4`

### 验证

```bash
bash tools/spike_rlr/run_C_habitat.sh
ls tmp/spike_output/videos/C_habitat_view*.mp4  # 4 files
# 眼验：C 组腿不动（T-pose），但位置随 trajectory 移动
```

---

## Task 9: 分析产物 — spectrogram + IR energy curve + metrics

**目的**：产出可读性最高的对比图与数值，直接支持决策表。

### 实现

- [ ] 9.1 `tools/spike_rlr/analysis/spectrogram_gen.py`：
  - 读 3 段 stereo downmix audio
  - 3 行 subplot（A/B/C），每行 stereo L+R spectrogram（librosa STFT，n_fft=1024, hop 256）
  - 标 t=3.0s 和 t=4.0s 两条虚线
  - 输出 `tmp/spike_output/analysis/spectrogram.png`
- [ ] 9.2 `tools/spike_rlr/analysis/ir_energy_curve.py`：
  - 对 B 组和 C 组：读每帧 IR，算 RMS energy dB
  - 对 A 组：从 audio wav 用 sliding window (100ms) 算 running RMS
  - Plot 3 条曲线 vs time，标 t=3.0-4.0s 遮挡窗
  - 输出 `tmp/spike_output/analysis/ir_energy_curve.png`
- [ ] 9.3 `tools/spike_rlr/analysis/metrics.py`：
  - 计算：
    - DRR (direct-to-reverb ratio)：early 50ms / late-part energy
    - RT60：Schroeder 逆积分 -60dB
    - Occlusion drop：mean(dB energy during t=3.0-4.0s) - mean(dB energy during t=0-3.0s)
  - 每 backend 各一份
  - 输出 `tmp/spike_output/analysis/metrics.json`

### 验证

```bash
python tools/spike_rlr/analysis/spectrogram_gen.py
python tools/spike_rlr/analysis/ir_energy_curve.py
python tools/spike_rlr/analysis/metrics.py
ls tmp/spike_output/analysis/  # spectrogram.png ir_energy_curve.png metrics.json
```

---

## Task 10: 决策表 markdown

**目的**：五分钟决策的最终产物。

### 实现

- [ ] 10.1 `tools/spike_rlr/analysis/build_decision_table.py`：
  - 读 metrics.json + 各 backend 耗时 log
  - 填决策表 columns（见 [设计 §6.2](../specs/2026-07-07-rlr-vs-gpurir-spike-design.md#62-决策表-columns)）
  - 4 条 Gate 判据：从 metrics.json 里的 occlusion_drop_dB / mux_success / wall_time_seconds
    读数，标 ✓/✗
  - 生成 `tmp/spike_output/DECISION_TABLE.md`
- [ ] 10.2 附一段"推荐路线"文字，基于 gate 通过情况写：
  - 全过 → 迁 RLR（档 ① swap-in）+ ReplicaCAD 视觉资产扩展
  - Gate 1 失败 → 保 GPURIR
  - Gate 2 失败 → 整栈迁 Habitat（重新评估）

### 验证

```bash
python tools/spike_rlr/analysis/build_decision_table.py
cat tmp/spike_output/DECISION_TABLE.md
```

---

## Task 11: run_all.sh 一键脚本 + spike_output 收敛

**目的**：一条命令跑完全部。

### 实现

- [ ] 11.1 `tools/spike_rlr/run_all.sh`：
  ```bash
  #!/bin/bash
  set -e
  export DISPLAY=:99
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

  cd $(dirname $0)/../..

  # Phase 0: SSOT + mesh
  /data/jzy/miniconda3/envs/sao-env/bin/python tools/spike_rlr/gen_mesh.py

  # Phase 1: UE Level
  # (如果 shoebox_v2.umap 已存在则跳过)
  bash tools/spike_rlr/cook_shoebox_v2.sh

  # Phase 2: A 组
  bash tools/spike_rlr/run_A_gpurir.sh

  # Phase 3: B 组
  bash tools/spike_rlr/run_B_rlr.sh

  # Phase 4: C 组
  bash tools/spike_rlr/run_C_habitat.sh

  # Phase 5: 分析
  /data/jzy/miniconda3/envs/hab-env/bin/python tools/spike_rlr/analysis/spectrogram_gen.py
  /data/jzy/miniconda3/envs/hab-env/bin/python tools/spike_rlr/analysis/ir_energy_curve.py
  /data/jzy/miniconda3/envs/hab-env/bin/python tools/spike_rlr/analysis/metrics.py
  /data/jzy/miniconda3/envs/hab-env/bin/python tools/spike_rlr/analysis/build_decision_table.py

  echo "SPIKE DONE — see tmp/spike_output/DECISION_TABLE.md"
  ```

### 验证

```bash
bash tools/spike_rlr/run_all.sh
ls tmp/spike_output/  # videos/ raw_audio/ analysis/ DECISION_TABLE.md
```

---

## 关键风险与缓解

| 风险 | 缓解 |
|---|---|
| habitat-sim `HABITAT_WITH_AUDIO=ON` pip install 失败 | 退化到 conda install / 源码编译 |
| UE Editor Python 造 procedural wall 不 work | 手工一次性造 shoebox_v2_wall.uasset (**本 plan 唯一允许手工产物**)，Python 只 spawn actor |
| RLR 每帧 IR 慢，超过 5min/场景 | 切换到 high-speed mode（精度掉 9.5%，速度 8×），记录在 DECISION_TABLE |
| A 组 4ch tetrahedral vs B/C 组 FOA 数学不完全对齐，spectrogram 比较不完全公平 | 分析脚本每组单独 downmix，spectrogram 3 行独立评估，不做数学对齐 |
| 坐标系转换错（UE Z-up cm vs Habitat Y-up m） | 在 gen_mesh.py 和 build_shoebox_v2_umap.py 里明确坐标系注释 + 单元测试验证 |

---

## Done Criteria

Spike 完成条件（**不是** gate 通过条件，是 pipeline 跑通条件）：

- [ ] `tmp/spike_output/videos/` 有 12 段 mp4 (A/B/C × view0-3)，每段能播放且有音频
- [ ] `tmp/spike_output/raw_audio/` 有 3 段 4ch wav
- [ ] `tmp/spike_output/analysis/` 有 spectrogram.png / ir_energy_curve.png / metrics.json
- [ ] `tmp/spike_output/DECISION_TABLE.md` 存在且 4 条 Gate 有明确 ✓/✗
- [ ] `run_all.sh` 从头跑通不需人工干预（除了 UE Level 首次构建可能需要一次手工）
