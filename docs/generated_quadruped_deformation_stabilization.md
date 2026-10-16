# Pixal3D / TRELLIS 四足动物形变稳定化

## 当前结论

当前可重复阶段是：人工确认绑定前基准方向后，将 Pixal3D/TRELLIS 原始 PBR
四足网格自动处理为约 200k 三角形运行网格，迁移已经批准的骨架与 Walk/Idle，
只修复顶点权重，并自动生成 41 帧回读和 Walk/Idle 审核视频。

用户已观看猫、TRELLIS 比格和巴哥的新视频，并反馈“确实好很多了”。该反馈
记录为 `research_candidate_user_reviewed_improved_pending_final_approval`，不是最终
`approved`，也不授权注册为正式数据集资产。

这条实现目前只覆盖同一骨架语义族的四足动物。鸟、鱼、蛇和其他身体计划需要
各自的运动族适配器，不能把狗的 Walk 骨架强行套用并宣称“任意动物已通过”。

## 为什么旧结果会出现拉丝和空洞

旧比格的“肚子空了”主要不是面数太少，而是远侧腿表面被错误权重拉成细带，
视觉上像缺面。原始单图重建也包含少量边界边和非流形边；但经过 glTF 位置缝
合并的拓扑安全减面没有新增边界裂缝。旧流程还会剪除腿间桥面，这会直接打开
肚腹和脚部，因此新流程明确禁止删这些面。

已拒绝的单点修复包括：渲染端 DQS/preserve-volume、Blender bone heat 自动
权重、整块绑定父骨以及带环形衰减的父骨锁定。它们要么无法处理非流形网格，
要么让拉伸更严重。

## 固定处理顺序

1. 认证原始 PBR GLB、已经批准的 animated rig GLB 和人工 motion-basis 决定。
2. 仅接受 `0/±90/180°` 整方向，禁止自动朝向推断和细角度补偿。
3. 合并 glTF 的位置重复点，减到约 200k 三角形；若新增边界裂缝立即失败。
4. 保留双面 PBR，不删除 ground artifact 或 limb bridge 面。
5. 使用 Blender C 层三角形 BVH 查找精确最近表面，再以重心坐标插值权重。
6. 保持骨架 rest matrix 与 Walk/Idle 动作曲线不变，仅运行
   `edge-average` 运动感知权重修复。
7. 导出 GLB 后重新读取 41 帧 Walking；结果不得比输入更差，当前实用门限为
   包围盒对角线的 4%。
8. 生成 960×720、12 fps、24 帧的 Walk/Idle PBR 视频，并用 ffprobe 与 SHA-256
   回读。
9. 输出仍是 `research_candidate_pending_human_visual_review`，代码不能自动升级
   为 `formal_dataset_asset`。

## 批处理入口

脚本：

`/data/jzy/code/AVEngine/external/SPEAR/tools/run_generated_quadruped_deformation_stabilization.py`

真实三资产 jobs manifest：

`/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/generated_animal_motion_stabilization_jobs_v1_20260715/cross_species_jobs.json`

只认证输入和人工决定，不生成文件：

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/run_generated_quadruped_deformation_stabilization.py \
  --jobs-manifest tmp/controlled_source_asset_execution_v1/generated_animal_motion_stabilization_jobs_v1_20260715/cross_species_jobs.json \
  --output-root tmp/controlled_source_asset_execution_v1/example_output \
  --validate-only
```

运行完整批次：

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/run_generated_quadruped_deformation_stabilization.py \
  --jobs-manifest tmp/controlled_source_asset_execution_v1/generated_animal_motion_stabilization_jobs_v1_20260715/cross_species_jobs.json \
  --output-root tmp/controlled_source_asset_execution_v1/generated_animal_motion_stabilization_batch_v1 \
  --workers 2
```

使用 `--asset-id ID` 可只跑一个已登记资产。输出根一旦存在便拒绝再次运行；
失败输出保留为证据，并在 `batch_state.json` 中写明错误，不会覆盖旧结果。

## 当前交叉验证

| 资产 | 后端 | 修复前最大延伸 | 41 帧导出回读 | 用户状态 |
|---|---|---:|---:|---|
| TRELLIS 比格 | TRELLIS 2 | 7.71% | 2.83% | 明显改善，未最终批准 |
| 虎斑猫 | Pixal3D | 11.34% | 2.36% | 明显改善，未最终批准 |
| 巴哥 | Pixal3D | 13.75% | 2.83% | 明显改善，未最终批准 |

完整哈希、绝对路径和 review 状态：

`/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/generated_animal_motion_aware_weight_repair_200k_v1_20260715/cross_species_canary_manifest.json`

视频网页：

`http://127.0.0.1:8097/docs/pixal_trellis_native_walk_review_20260714.html`

## 性能

旧最近表面实现对每个目标点扫描全部源三角形，200k 猫运行网格约耗时 290 秒。
新的 BVH 实现在相同输入上耗时 18.32 秒，约 15.83 倍加速；修复前 41 帧指标
完全相同，完整修复为 2.38%，与参考实现的 2.36%一致。权重修复目前仍是主要
CPU 耗时，应在批量运行时并行不同资产，而不是降低审核质量或重复渲染。

批处理脚本自身也已完成一次真实猫 canary（不是 dry-run）：总计 290.45 秒，
其中 200k 运行网格 32.00 秒、BVH 绑定 14.44 秒、权重修复 175.31 秒、前后两次
41 帧回读分别 34.36/33.95 秒。输出 manifest 为：

`/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/generated_animal_motion_stabilization_batch_canary_v1_20260715/batch_manifest.json`

其 SHA-256 为
`e98a461ad3e509980772485625f7f28b2dd91c7189fb26d068977299fce25ee2`。
