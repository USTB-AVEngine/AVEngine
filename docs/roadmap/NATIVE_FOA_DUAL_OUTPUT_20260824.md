# 检查点：Native FOA + binaural 双落 runtime-b 验证（20260824）

## 1. 结论与执行身份

原生 RLR FOA 与既有 binaural 动态音频双落核心已在真实 runtime-b 上执行
成功。本检查点只记录这次已经发生的 MP3D 保留轨迹验证，不把它外推为
Apartment、多视角数据或 trainer 交付。

| 项 | 实际值 |
| --- | --- |
| 代码提交 | `ff582ea9d50a95c34baf4cff54443cd58b80e6a0`（`feat(audio): dual-render native FOA and binaural`） |
| fresh 输出 | `/data/jzy/tmp/foa_dual_runtime_b_ff582ea_r1` |
| receipt | `/data/jzy/tmp/foa_dual_runtime_b_ff582ea_r1/research_receipt.json` |
| receipt 状态 | `status=pass` |
| 声明边界 | `research_only=true`、`episode_counted=false`、`formal_dataset_count=0`、`qualification_claim=false` |
| 路线 | MP3D 当前保留视觉轨迹；视觉帧位置来自 `current_mp3d_lateral_seed22g_visual_480p/frame_records.json` |
| source IDs | `beagle_0_muzzle`、`beagle_1_muzzle` |

输入中的视觉 capture、M1 listener request、RLR simulation request、MP3D
acoustic package、source/sound registries 和 HRTF 均由 receipt 逐项记录；本次
没有用 Apartment UE 像素或新生成视角替代这条 MP3D 路线。

## 2. 双布局同轨迹证据

同一显式 keyframe grid 分别建立 binaural 与 FOA RLR context：

| 项 | 实际值 |
| --- | --- |
| episode | 75 帧，15 fps，80,000 samples @ 16 kHz |
| RIR stride | 3 帧 |
| keyframe 数 | 25 |
| keyframe samples | `0, 3200, ..., 76800` |
| 同源身份 | `same_source_and_trajectory_identity=true` |
| 共用 trajectory SHA-256 | `be20857057a1f75f9445c1354e9e9da461da7aff843054d7b2d35202cd1cde9f` |

Receipt 中 `rir_layouts.binaural` 与 `rir_layouts.foa` 都记录了上述同一
trajectory SHA；FOA 布局为 `rlr_foa_acn_n3d_world_v1`，binaural 布局为
`rlr_binaural_lr_v1`。

## 3. WAV 与 sidecar 机器读回

对两个 mixture 做了实际 SHA-256 与 `ffprobe` 读回：

| 输出 | 编码 | 声道 | 采样率 / 时长 | SHA-256 |
| --- | --- | ---: | --- | --- |
| `audio/binaural/mixture.wav` | `pcm_f32le`（sample_fmt `flt`） | 2 | 16 kHz / 5.000000 s | `a0c04fe99d5a657d99f21cf2ef7c718bba2b2716754a20851e1fe42c8698abfe` |
| `audio/foa/mixture.wav` | `pcm_f32le`（sample_fmt `flt`） | 4 | 16 kHz / 5.000000 s | `e106c7d0dde4d2441be475f63e4576881e70effbf30a7aea72fcd07bb81a56e8` |

FOA mixture 与 stem 的 `.wav.json` sidecar 机器声明：

- order 1，raw channel order `[W,Y,Z,X]`；
- ACN indices `[0,1,2,3]`，normalization `N3D`；
- coordinate frame `avengine_world`；
- `hrtf_used=false`；
- `resampling=not_applied`；
- `amplitude_normalization=not_applied`；
- `limiter=not_applied`；
- mixture role 为 `dynamic_wet_mixture`，stem role 为
  `dynamic_wet_stem`。

Fresh 输出共有 **8 个 WAV**，且每个 WAV 都有 sidecar：

- 2 个 mono dry bus；
- 2 个 binaural wet stem + 1 个 binaural mixture；
- 2 个 native FOA wet stem + 1 个 native FOA mixture。

## 4. 软件层验证与时间记录边界

与该切片对应的 focused hermetic tests：

```text
30 passed
```

覆盖了旧 binaural canonical trajectory hash、双 context 同轨迹、FOA
`[W,Y,Z,X]`/ACN/N3D/world 严格验证、portable 无-HRTF receipt、FOA
sidecar、fresh/no-clobber 和 CLI/tool help。

真实 CPU 计算在现场观察中约为 **8–9 分钟**，但本次没有保存权威
wall-clock 计时记录，因此这里不报告精确 wall time，也不据此建立性能
baseline 或 gate。

## 5. 未完成边界

本次证据只证明 **MP3D 保留轨迹上的 dual audio core runtime-b 执行成功**。
以下事项仍未完成，不能由本检查点代替：

- Apartment 每个认证视角及每视角 4 点 mini-pilot 的原生 FOA 双落；
- Apartment 多视角/多种类量产数据；
- dataset `audio_path` 从 foaish 或旧音频切换到本次 native FOA；
- trainer 对 native FOA 的真实读取与冒烟；
- trainer 正式训练、评测或模型效果结论。

正式数据集分母继续保持 **0**；本输出不得计为正式 Episode 或数据集
admission 证据。
