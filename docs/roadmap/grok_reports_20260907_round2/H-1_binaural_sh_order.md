# H-1 双耳混响阶数

对照实验先行，再改代码，再重渲 UE 家族已交付段的音频。视频与 attempt_01/02 未改。

## 1. 改了哪些文件与提交号

- `tools/acoustics/render_frame_readback_sequential_speech.py`
- `src/avengine/rooms/qa_delivery.py`
- `tests/unit/test_frame_readback_binaural_sh_order.py`（新）
- 实现提交 `703e4dbb5c00cbc21e5bc91c8bce13660876db2d`（`grok/pilot46-fixes-round2-20260907`）

默认值与 Habitat 路径 `examples/runtime/rir_cache_simulation_request_v2.json` 的阶数/深度/IR 长度对齐：`direct_sh_order=3`、`indirect_sh_order=1`、`indirect_ray_depth=200`、`max_ir_seconds=4.0`。UE 仍是双耳，不从该文件抄 ambisonics / diffraction / transmission。

请求或房间包可覆盖这四项（CLI `--direct-sh-order` / `--indirect-sh-order` / `--indirect-depth` / `--max-ir-seconds`，或 `--simulation-request` JSON 只取这四键；`build_audio_command` 从 request/runtime/plan 的 `simulation` 透传）。

RIR 作业计划的 `cache_key_fields` 仍是位姿三元组（`rir_cache.py` 校验器钉死，不能加字段）。仿真参数在 cache `request.json` 的 `simulation.effective` 里，已进入 request identity。命中已有缓存时若 effective 仿真与本次请求不等，拒绝复用。`research_receipt.json` 的 `qa.propagation.simulation` 写实际阶数。

## 2. 测试

合入 H-2～H-7 后，在 `/data/jzy/tmp/wt-grok-pilot46-round2` 跑本轮触碰的 16 个 unit 文件：

**130 passed / 0 failed / 0 skipped**（4.69 s）。文件清单见就绪报告 §4。这是计数，不是放行依据。

## 3. 验收产物与亲自核对

### 3.1 对照实验（新目录，不写 attempt_02）

输入：`qa_pilot46_rerun_20260907_v1/.../authored_a_human_human/attempt_02/episode` 同一 neutral readback 与 plan。产物 `tmp/h1_sh_ablation_20260907/authored_a_human_human/`，度量 `ablation_metrics.json`。方位来自 attempt_02 `audit_v2.json`：source1 +24.73°、source2 −14.30°。

| 组 | 2 ms 起 L/R 相关 | 存储 IR 长 | 每条 RIR 模拟 | 墙钟 | stem 可用线索 | 混音峰值 |
|---|---|---|---|---|---|---|
| 基线 0/0/64/0.25 | 0.996 / 0.997 | 0.66 s | 0.40 s | 17.6 s | 两源都不达标 | −16.0 dBFS |
| 只改阶数 3/1/64/0.25 | 0.450 / 0.346 | 0.66 s | 0.39 s | 16.2 s | −14° 达标；+25° 未达标 | −11.4 dBFS |
| 3/1/200/0.25 | 0.446 / 0.378 | 1.73 / 1.78 s | 0.73 s | 15.5 s | 两源都达标（ILD） | −11.0 dBFS |
| 3/1/64/4.0 | 同 3/1/64/0.25 | 0.66 s | 0.38 s | 15.4 s | 同左 | −11.4 dBFS |
| **3/1/200/4.0（选用）** | 同 3/1/200/0.25 | 1.73 / 1.78 s | 0.79 s | 16.2 s | 两源都达标 | −11.0 dBFS |

2 ms 起 L/R 相关从 ≥0.99 降到 <0.9。ILD 符号与几何一致。Stem ITD 仍小（0 至 −0.125 ms），达标靠 ILD。`max_ir_seconds` 0.25→4.0 在同一深度下不改变存储 IR 长度；拉长湿尾的是 depth 64→200。生产参数取 **3/1/200/4.0**，2 作业房间墙钟仍约 16 s。

### 3.2 UE 家族重渲

新根：`tmp/qa_pilot46_audio_v2_20260907/episodes/<id>/attempt_03/`。capture/plan 为指向最终 attempt 的符号链接。度量：`tmp/qa_pilot46_audio_v2_20260907/h1_acceptance.json`。

任务书写「32 段」；合并表这三家族已交付是 **28**。4 个空格无成片（`authored_a_device_device`、`kujiale_device_device` blocked；`authored_b_animal_device`、`authored_c_animal_device` failed）。Habitat 段未重渲。attempt_01 facts 时间戳 12:20、attempt_02 19:24，重渲在 22:00 之后，原批未写。

28 段仿真请求全部是 `direct_sh_order=3, indirect_sh_order=1, indirect_ray_depth=200, max_ir_seconds=4.0`、双耳。

| 项 | 结果 |
|---|---|
| 交付 audio+facts | **27 / 28**。失败：`authored_b_device_device`，湿混音 peak=1.14，渲染器按契约拒绝写出（不允许自动限幅/归一化）。该段 0 阶原片已是 −5.40 dBFS。RIR 缓存已写出。 |
| RIR 尾部 L/R 相关 | **56 / 56 条作业 < 0.9**（最高 0.678，apartment_human_human source1） |
| 混音峰值 −6～−25 dBFS | **24 / 28**。窗外：`authored_a_animal_device` −1.63；`kujiale_animal_device` −2.21；`authored_a_human_device` −5.20；`authored_b_device_device` 无成片 |
| 侧向事件（\|方位\|≥20°）stem 上 \|ITD\|≥0.15 ms 或 \|ILD 2–6 kHz\|≥2 dB | **24 / 28**（apartment 6/7，authored 13/16，kujiale 5/5） |
| 通过 questions 条数 | apartment 59→61；authored 153→145（含 B device_device 5 条未写出）；kujiale 56→54。合计 268→260 |

未加限幅器。未把听感写成验收通过。

## 4. 没做完

- 题义不适用：Habitat 段已是 3/1，不重渲。4 个空格无音频。
- 接口未实现：无（本项不涉及挂装）。
- 证据缺失：`authored_b_device_device` 无 attempt_03 成片；4 段混音峰值在 −6 dBFS 以上；stem ITD 普遍低于 0.15 ms，侧向线索主要靠 ILD。

## 5. 对后续接口

- `_simulation()` / `render()` / `build_audio_command()` 接受 `direct_sh_order`、`indirect_sh_order`、`indirect_ray_depth`、`max_ir_seconds`。
- `existing_rir_cache_simulation_matches(request, simulation)`：旧缓存 `simulation.effective` 必须与本次 `to_dict()` 相等才允许命中。
- 作业计划 `cache_key_fields` 不可加仿真字段，除非同时改 `rir_cache.py` 校验器。
- 渲染器仍 fail-closed：peak>1 且不自动限幅。

## 6. 需 owner 拍板

- 任务书「32 段」对已交付 28 段：本轮按 28 段重渲。
- `authored_b_device_device` 削波：要不要给一个**预先声明**的卷积增益（不是自动归一化），还是保持 fail-closed。
- 4 段峰值高于 −6 dBFS（含动物/设备）：是否接受，或统一降干增益。
- 重渲后是否再听一包含动物与设备声的成片。attempt_03 路径在 `tmp/qa_pilot46_audio_v2_20260907/`。
