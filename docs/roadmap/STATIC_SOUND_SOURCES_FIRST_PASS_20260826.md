# 静态发声资产第一遍执行记录（2026-08-26）

## 结论

- 26 个 AudioSet 静态声类已显式映射到 14 个逻辑网格族。
- 第一遍按形态优先、每个形态一个默认饰面，最终发布 28 个 research-only 资产。
- 所有发布资产均为 `formal_dataset_registration_authorized=false`；正式数据分母仍为 0。
- 没有生成水壶，没有把 26 个声类误做成 26 个网格，也没有运行动物的
  `gate_retopology.py` / `gate_rigged_asset.py`。
- 共享入口：`/data/avengine_external/assets/sound_source_assets_v1/index.json`。
- 14-family 映射键：`mesh_families.static_sound_sources_20260826`；其中列出 26 个
  AudioSet 类和本次 28 个 asset id。

## 第一遍最终形态

| 逻辑网格族 | 形态 |
|---|---|
| air_conditioner | wall_split / window_unit / portable_floor |
| microwave_oven | countertop / over_range |
| printer | desktop_inkjet / office_laser_mfp |
| blender | jug_blender / bullet_blender |
| alarm_clock | digital_cube / twin_bell_analog |
| doorbell | wall_mounted_box / video_doorbell |
| landline_phone | corded_desk_unit / wall_mounted |
| cellphone | bar_smartphone |
| smoke_detector | ceiling_disc / wall_square |
| toilet | exposed_flush_pipe_tank / elevated_tank_exposed_pipe |
| sink_with_tap | pedestal_basin / counter_vanity |
| bathtub | freestanding / built_in_alcove |
| floor_drain | floor_drain / exposed_bottle_trap |
| fireplace | masonry_open / wood_stove |

`desk_telephone`（旧现成 profile）和 `landline_phone`（壁挂形态）在资产 taxonomy
里是两个历史 object_type，但在 index 中共同归入逻辑族 `landline_phone`。

## 形态调整

工单 §2.3 允许调整建议形态，但必须在产物里写明理由。保留的调整记录：

- `examples/assets/source_profiles_mirror/static_sound_form_adjustments_20260826.json`
- `examples/assets/source_profiles_mirror/static_sound_form_adjustments_20260826_v2.json`
- `examples/assets/source_profiles_mirror/static_sound_form_adjustments_20260826_v3.json`

三次独立 P-trap 请求都生成直通三通，因此最终改为真实的 exposed bottle trap。
wall-hung、close-coupled 和 one-piece 方法多次不能同时守住形态与可见水路，最终改为
`exposed_flush_pipe_tank` 与 `elevated_tank_exposed_pipe`：两者视觉差异大，且均有可见、
可审查的 tank-to-bowl 水路。旧 profile 和失败 batch 全部保留，未删除或覆盖。

## 运行批次

| 批次 | 作用 | 最终通过 |
|---|---|---:|
| `tmp/static_sound_sources_first_pass_20260826_r1` | 28 形态首轮 | 21 |
| `tmp/static_sound_sources_retry_20260826_r2` | 7 个首轮失败项 | 3 |
| `tmp/static_sound_sources_method_retry_20260826_r3` | plumbing 方法修订 | 1 |
| `tmp/static_sound_sources_adjusted_20260826_r4` | bottle-trap 调整 | 1 |
| `tmp/static_sound_sources_toilet_final_20260826_r6b` | 最终 toilet 形态 | 2 |

合计 21 + 3 + 1 + 1 + 2 = 28。所有 Pixal 最终批均为
`passed_generation_and_glb_readback`，正式准入批均为 0 运行失败。

## 评审与准入证据

主要 review 根：

- `/data/avengine_external/review/static_sound_sources_first_pass_20260826_r1`
- `/data/avengine_external/review/static_sound_sources_retry_20260826_r2`
- `/data/avengine_external/review/static_sound_sources_method_retry_20260826_r3`
- `/data/avengine_external/review/static_sound_sources_adjusted_20260826_r4`
- `/data/avengine_external/review/static_sound_sources_toilet_final_20260826_r6b`

每批包含或引用：profile 快照、prompt token 报告、2D/3D 判决、五视图、焊接后拓扑、
组件面积、倾角、albedo、finalization、emitter measurement 及 marker review。
每个最终 `asset.json` 同时记录：

- form/material/默认饰面与 acoustic profile；
- 完整 FLUX 参数、seed、模型 revision、候选图 SHA 与 token 账目；
- final/watertight GLB SHA、三轴尺寸、面数和已知倾角；
- emitter anchor、物理理由、front/side/top marker review 哈希；
- 固定/贴附件的 attachment surface、facing、实测 bbox 与 RIR 必须重算声明。

## 实测修复

厘米级地漏在 marker review 中最初全黑。具体原因是 Blender camera 默认
`clip_start=0.1 m`，而 3 cm 资产的 framing camera 只有约 5.4 cm 远。修复后 near
clip 随 review radius 缩放，正式 renderer 重放时 near clip 约 1.08 mm，地漏和 marker
恢复可见。可重放补丁为
`examples/assets/source_profiles_mirror/spear_patches/fix_i23d_review_camera_clip.py`。

## 发布状态

发布后共享 index 中本任务资产数为 28，面数范围 59,921–60,000；每个叶子均有
`asset.json`、`finalized.glb`、`watertight.glb`、`emitter_marker.glb` 和 evidence。
索引保持可合并：既有 animal/audio_playback 资产和 gate 键均保留。

本记录不代表 owner 的正式数据准入，也不授权合并 main；两者仍由 owner 决定。
