# 动物视频与审核媒体总目录

更新时间：2026-07-13。

## 当前推荐：Pixal3D pug Apartment 完整审核

该候选使用 Pixal3D PBR mesh、100k 双面近景 LOD、Quaternius Dog Walk/Idle 骨架。Apartment Walking 路径与人类一致：相机右后方出发，经过相机左前方，再逆时针绕相机前圆桌一圈。18 秒狗叫由能量阈值切成 6 个源事件，并以至少 0.85 秒静默间隔安排为 7 次事件，不做无缝循环。

| 媒体/证据 | 链接 |
|---|---|
| UE 主视图 + Top-down + 双耳音频 | [审核视频](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/clips/dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking/videos/side_by_side_review_annotated.mp4) |
| UE 主视图 | [主视图](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/clips/dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking/videos/apartment_v1_view0.mp4) |
| Top-down 轨迹 | [Top-down](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/clips/dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking/videos/topdown_review.mp4) |
| 4 个关键帧 | [接触表](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/clips/dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking/videos/trajectory_keyframes_contact_sheet.png) |
| 轨迹预检 | [PNG](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/trajectory_preflight.png) |
| 4 段身体/轨迹方向证据 | [JSON](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/clips/dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking/videos/actor_visual_metadata.json) |
| 狗叫事件调度 | [JSON](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/clips/dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking/binaural_source_schedule.json) |
| 可复现规格 | [JSON](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/specs/dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking.json) |

自动检查：270/270 帧、Walking、动态贴地残差约为数值零、最大穿透约为数值零、根 Roll/Pitch 为 0；四个方向窗口全部通过，身体 forward 误差约 0.50–0.52°。该资产仍为 `research_candidate`，不是正式注册资产。

## Pixal3D 全动物替换批次

14 个其余旧动物参考图正在由 4 张 GPU 并行生成；已有 pug 直接复用，不重复生成。产物根目录与状态入口：

- [生成根目录](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_backend_substitution_v1/generated_batch_v1)
- [批次状态（完成后生成）](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_backend_substitution_v1/generated_batch_v1/batch_status.json)
- [四只猫静态接触表](/data/jzy/code/AVEngine/external/SPEAR/tmp/pixal_animal_backend_substitution_v1/generated_batch_v1/cat_batch_static_contact_sheet.png)

首批静态判断：British Shorthair、Siamese、Tabby 可继续减面/绑定；Persian 有明显尖刺/片状毛发几何，静态阶段直接 `rejected`，不进入绑定动画。所有 Pixal 原始动物朝向为 head `-X`；绑定到 Quaternius 动物模板时使用已由 pug canary 验证的 X 镜像，使绑定后动画 forward 为 `+X`。

## 历史 Hunyuan 媒体（仅技术证据）

以下旧视频仍保留用于回归比较，但资产和输出一律保持 `technical_spike_only/rejected`，不得进入正式训练或评测数据：

- [旧两狗 Apartment 带音频 view0](/data/jzy/code/AVEngine/external/SPEAR/tmp/gpurir_scenes_v19/two_dogs/apartment/view0_with_audio.mp4)
- [旧两狗 Apartment view0](/data/jzy/code/AVEngine/external/SPEAR/tmp/gpurir_scenes_v19/two_dogs/apartment/view0.mp4)
- [旧动物 Apartment dog turntable](/data/jzy/code/AVEngine/external/SPEAR/tmp/render_animals_apartment/dog/turntable.mp4)
- [旧动物 Apartment cat turntable](/data/jzy/code/AVEngine/external/SPEAR/tmp/render_animals_apartment/cat/turntable.mp4)
