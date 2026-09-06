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

### 20260907 原点修复与全量原生加载补验

P11 库存复查时，父代理发现 Habitat ObjectAttributes 默认把视觉原点移到
包围盒中心；旧黑灰音箱的 native visual bbox 最低点为 root 下 0.165 m，
即使计划与实际 root 一致仍不能证明正确落地。`_instantiate_rigid_object`
现显式设置 `com=(0,0,0)`、`compute_COM_from_shape=False`，保留 finalized
GLB、resting_pose 和 emitter 共用的原点，不改变资产文件或中立根轨道。

`tmp/p11_habitat_static_native_probe_20260907_v2/` 中 40/40 正式刚体均实际
加载、读取有限根/发声点/包围盒，并生成有目标语义像素的 RGB；native
creation_attributes 均读回 zero COM / no automatic recentering。父代理已看
`native_contact_sheet.png`。31 个 floor 资产的原始 GLB 经过全部 node
transform 后，精确顶点底面相对登记 base plane 的最大误差为 3.63e-8 m。
`source_vertex_origin_crosscheck.json` 明确区分这一源顶点测量与旋转层级的
保守 native cumulative AABB；未按保守 AABB 额外抬高任何资产。

修复后原生重捕获在 `tmp/p4_rigid_origin_captures_20260907_v1/`：HM3D 240帧，
MP3D 30帧，各自 `capture/` 与 `delivery_v1/`。target-only alignment 均 pass；
HM3D 两源均 240 帧 visible_clear、两种外观 reviewed；MP3D 音箱 30 帧 clear，
beagle 仍受床遮挡且毛色 not_observable。父代理查看了三时刻实际 RGB 拼图。
两段各自新旧实际 camera、clock、两源 emitter 逐值完全一致，故经
`actual_acoustic_inputs_comparison.json` 核对后复用对应的未修改 P6 双耳音频。
新片分别 16秒/2秒、2声道；P9 合同和导出通过，题目 11 valid/13 deferred、
9 valid/15 deferred。旧有 COM 居中捕获保留为诊断，不再作为摆放验收。

本次受影响的 capture/static-binding 单测 11 passed / 0 failed / 0 skipped
（10.41秒，`tmp/p4_rigid_origin_tests_20260907_v1.log`）。本节、源码和回归测试
随原点修复提交；源码仍来自指定服务器权威工作树。

## 4. 尚缺与证据边界

P4 已交付捕获和登记接口；该段保持 research_candidate。音频回执、外观
审阅和完整 EvidenceContract 由 P6/P9 补齐，不能据此宣布全链验收。
wall/ceiling 的挂装执行尚未完成，当前刚体捕获只接受 floor resting_pose；
相关资产留在覆盖分母并记 interface_not_implemented。40 条绑定现已有逐资产
原生加载/语义像素记录，其中 8 个 wall、1 个 ceiling 仅通过加载，挂装执行
仍未实现；不以 blank-scene 加载声明任何房间接触、碰撞或声音验收。

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
