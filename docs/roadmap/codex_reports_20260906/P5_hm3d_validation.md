# P5/H5 HM3D 独立真实验证

日期：2026-09-07。范围是一个研究用单段，不是 46 段生产批次；未改源代码、房间包或 registry。

请求使用正式 `examples/rooms/packages/catalog.json` 的
`hm3d_val_00800_TEEsavR23oF`，`conditioned_static_v2` 固定画像：两名已注册
human blue/green、静止相机 85°、`all_still`、240 帧、15 fps、16 秒、16 kHz，
音频候选来自 P7 `prepared_manifest.json`。计划保留了两个 sampler retry 的
`camera:no_joint_geometry_activity_schedule` 直方图，第三次在同一画像下成功；
没有替换画像、时长或时钟。

真实 `--capture-only` 通过 GPU3 完成。实际 room manifest 使用：

- 渲染面：`TEEsavR23oF.glb`
- 语义面：`TEEsavR23oF.semantic.glb` 与 `TEEsavR23oF.semantic.txt`
- 数据配置：`tmp/p3_room_packages_20260907_v5/hm3d_annotated_val_nonbasis.scene_dataset_config.json`
- navmesh：`TEEsavR23oF.basis.navmesh`

全部路径均来自物化 HM3D room manifest，未使用 MP3D 场景。

实际捕获位于 `tmp/p5_hm3d_validation_20260907_v5/capture/`：

- RGB：`rgb.npy`，`[240,240,320,3]`、uint8、范围 0–255；所有帧相同，首帧均值 RGB `[156.05,150.12,142.27]`，全黑比例 0.11%。`rgb_contact_sheet_000_120_239.png` 已由 Codex 子代理图像检查，室内房间和源角色渲染可辨认。
- depth：`depth.npy`，`[240,240,320]`、float32、finite=1，范围 0–4.9726 m，正值比例 0.99906。
- semantic：`semantic.npy`，`[240,240,320]`、uint32、finite=1，包含 source1=210/source2=211，240/240 帧均出现。
- actor/root/emitter：两角色各 240 帧；root 对 planned actor root 的最大误差分别为 `1.11e-8 m`、`7.28e-8 m`，首尾位置不变；实际 emitter 首尾不变，JSON/NPY 一致。mapped joint target 最大误差为 `2.97e-8`。
- camera/clock：240 帧相机位置唯一且静止；位置最大误差 `1.91e-7 m`、旋转误差 0、RGB/depth/semantic 传感器姿态不一致为 0；frame index 与 `pts_ticks=frame*3200` 全部连续。当前代码重新生成的 `neutral_readback_current.json` 覆盖 240 帧 camera、source1、source2，root 语义经过 package transform 处理。
- target-only/pixel：`native_pixel_masks_depth_authority_v1.npz`、`pixel_visibility_truth.json`、`target_only_readback_alignment.json` 均生成。source1 可见比例 0.2012（237/1178 像素），source2 为 0.5418（953/1759 像素），两者 240/240 帧均为 `visible_occluded`；alignment 为 pass。`actor_occluders.json` 保持 `pass`，但 `no_identified_foreground_actor=480` 仍是未解析边界。

`research_receipt.json` 和 `episode_result.json` 明确为 `research_only`：audio、RLR、QA、model evaluation 未运行，Habitat object-ID 未请求（target-only semantic masks 已完成），`episode_counted=false`、`qualification_claim=false`。P4 object-ID runtime prefix、Magnum site、RLR SDK 均通过实际捕获参数传入。

汇总证据：`tmp/p5_hm3d_validation_20260907_v5/validation_summary.json`。旧 v1–v4
失败现场保留；v1 是漏写 `sampling_policy`，v2/v3 是 clock/CLI 兼容错误，v4 是
Habitat seed 类型错误，均没有被覆盖。
