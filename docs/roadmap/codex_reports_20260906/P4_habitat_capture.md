# P4 Habitat 捕获与资产绑定

## 1. 实现与文件

基于分支 ca70486 开始，父代理复核时 HEAD=15e4dfa；本报告与实现同一提交。
修改 capture/mp3d_multi_actor、assets/mp3d_region_actor_tracks；新增
assets/habitat_static_assets 及两份 tools/capture 准备工具。源登记、对应
schema/runtime_profiles、单测和工具索引一并更新。没有修改外部源资产索引。

刚体 GLB 以 KINEMATIC 对象加载，设置 episode semantic ID，按登记的
resting_pose 放置，逐帧读取实际根矩阵及 emitter。静态实体无步行动作。
完整场景输出 modal；每个目标在 scene_id=NONE 的独立 Habitat simulator
重放同帧根、关节和相机，得到没有场景/竞争者遮挡的 target-only 投影。
逐帧对账避免把两次不同姿态的像素混为真值。

NPZ 保留真实旧键 depth_derived_modal_semantic，并增加逐元素相同的 modal
别名和 target_only_sourceN。pixel_visibility_truth、actor_occluders 与
neutral_readback 直接由真实读回派生；未识别的静态遮挡物仍记 unresolved。

## 2. 登记增量

examples/runtime/source_asset_runtime_profiles.json 新增 38 个原在外部索引的
刚体条目，并为原有两个音箱及 beagle 加 Habitat 后端：55 条运行时记录中
41 条有 Habitat 绑定（40 rigid + 1 beagle）。旧 17 条 SPEAR 绑定逐值一致。
40 个外部刚体全部通过仅吃源登记的 bind_assets 解析，GLB 引用实际存在。
原始外部 44 条中还有 4 个 articulated animal；它们保留在总分母，未改称
刚体。并集仍为 59（40 rigid、12 animal、7 human）。

源 schema 新增 Habitat 字段校验；刚体可有任一实际渲染器绑定，但已声明
SPEAR 绑定的原校验全部保留。缺少该渲染器绑定仍是接口缺口。中立 emitter
不再要求该资产同时有 SPEAR 字段。beagle 捕获现在只需 asset_id，由源登记
解析 M2 package，无需外部 delta 或包路径参数。

## 3. 验证与实际产物

权威 cwd=/data/jzy/tmp/wt-multi-home-activity-integration；Python 为项目
avengine-habitat-runtime/bin/python，PYTHONPATH=src。

相关单测 45 passed / 0 failed / 0 skipped，6.48 秒；工具索引 1 passed。
合入真实登记后库存交集由 2 变为 40，更新了对应库存断言；第一次测试的
44 pass/1 failure 是该旧计数失效，修复后通过。

子代理原生证据：tmp/p4_beagle_speaker_capture_20260906_v7/ 与
 tmp/p4_beagle_speaker_capture_registry_20260906_v2/。
父代理在合入登记后、完全不传 external-index/binding-delta 的新捕获：
 tmp/p4_beagle_speaker_parent_registry_only_v1/。
GPU2 启动前占用 19 MiB / 49,140 MiB；30 帧原生捕获正常退出。

父代理检查 parent_readback_check.json、target_only_readback_alignment.json、
原始 RGB/semantic/masks、neutral_readback，并查看 parent_first_frame.png：

- RGB [30,240,320,3]；两实体每帧均有模态像素；beagle 最少 15 像素/帧，
  总计 735；音箱每帧 407，总计 12,210。
- target-only 总像素 20,741 / 14,700；alias 完全相同。
- blank-stage camera 误差为 0；root 最大约 1.79e-7 m，joint 约 3.24e-8。
- 中立读回通过 P1 校验；41 个 Habitat 绑定从正式源登记解析通过。
- 第一帧中 beagle 被床遮挡很多，此段证明捕获/掩膜接口可执行，不能当成
  已通过动物外观或题目可答性审阅。

## 4. 尚缺与证据边界

P4 已交付捕获和登记接口；该段保持 research_candidate。音频回执、外观
审阅和完整 EvidenceContract 由 P6/P9 补齐，不能据此宣布全链验收。
wall/ceiling 的挂装执行尚未完成，当前刚体捕获只接受 floor resting_pose；
相关资产留在覆盖分母并记 interface_not_implemented。40 条绑定的加载引用
已解析，原生 GLB 捕获本次验证了一个音箱，其余的真实加载矩阵仍待 P11。

任务书的 /data/datasets/habitat_data 未包含此 MP3D 场景；实际采用服务器
已有的 /data/avengine_external/datasets/mp3d_example_scene_1.1，scene 是同一
17DRP5sb8fy，并在原生回执中保留此路径。未创建挂载或目录别名。

## 5. Claude 与后续接口

bind_assets(asset_ids, renderer='habitat', runtime_registry_path=..., ...)
返回 HabitatAssetBinding。物化保留 materialize_habitat_rigid_track；中立
读回写出器沿用 P1。后续 P5 直接接计划根轨道，不重规划；P9 使用相同
NPZ/truth，无需依赖 renderer 名。审计器和 Claude 的测试没有修改。

## 6. Owner 决策

当前实现无需新增授权。若必须把 MP3D 放到任务书指定根目录，挂载/别名
属于另一个明确操作，当前未实施。人工外观与听音结果仍须真实记录。
