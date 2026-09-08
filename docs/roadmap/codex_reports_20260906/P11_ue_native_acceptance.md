# P11 UE 刚体、关节发声点与原生路径

## 1. 文件与提交关联

权威工作树：48g-jump:/data/jzy/tmp/wt-multi-home-activity-integration；分支
codex/multi-home-activity-integration，集成基线3d81f3d。本报告与P11实现、
P12四资产增量和P5实测纠偏一起提交；没有push、main合并或Studio切换。

修改 qa_episode.source_declaration、capture/qa_plan_adapters、
backends/spear_ue/research_runtime、两个UE runner、runtime_profiles及其schema；
对应测试为 test_p11_ue_rigid_bindings、test_runtime_profiles。刚体只走显式
StaticMesh、已测floor resting_pose、静态轨迹和登记emitter；不造动画。
relative scale、mesh handle、parent、root、camera、emitter均由UE实际读回。

新增可选SPEAR ue_emitter_attachment（bone/socket、local_offset_cm、probe
artifact_ref）；只有 renderer_neutral materializer 显式启用的执行模式消费它。
无新策略的请求和旧saved plans保留root-local路径。新模式按资产timeline轴和
native UE anatomical yaw 推导visual-child修正，保持中立actor root不变。
Beagle旧root-local [34,0,61]cm不在真实muzzle，现从实际beagle-Xtra-Mouth骨骼
附着并逐帧读world pose；其visual-child yaw修正为-180度，其余既有资产为0。

## 2. 测试与修复记录

父整合测试87 passed / 0 failed / 0 skipped，5.56秒：runtime profiles、P11、
conditioned sampler、P12 builder/mesh audit、P3 room packages；日志
 tmp/p11_parent_integrated_tests_20260907_v3.log。此前代理还运行过相关SPEAR
Apartment回归；最终提交前将以整合集合及工具索引检查为准。

保留的失败包括：camera_state缺frame_index导致
`expected camera state frame order changed`，在验证clock/frame顺序后补绑定；
GetRelativeScale3D不可调用，改用实际RelativeScale3D属性读取；
父整合第一轮因测试stub未适配新增registry resolver触发RuntimeProfileError，
修正测试依赖隔离后通过；第二轮P3测试仍断言旧`/ue_stage`后缀，随实际新stage
配置更新该断言后通过。未放宽真实schema或像素/音频校验。

## 3. 实际产物与亲自核对

A房两段（额外对照，不冒称native Apartment家族），均在
 tmp/p11_ue_asset_captures_20260907_v9/ 下：

| 组合 | capture | P6 | P9 | 题数 |
|---|---|---|---|---|
| human_speaker | human_speaker/preflight_native_stage | human_speaker/audio_endpoint_bound/research_receipt.json | human_speaker/finalize_p9_v2 | 7 valid / 17 deferred |
| human_animal | human_animal/preflight_native_stage | human_animal/audio_endpoint_bound/research_receipt.json | human_animal/finalize_p9_v2 | 8 valid / 16 deferred |

两段原生240帧@15Hz、16秒；各自真实双声道16kHz/256000样本、2 events/2 stems；
P9 facts、contract、export与独立mux全部通过。Blue human均reviewed，Border
Collie standard_black_white reviewed；A音箱coarse finish not_observable，未伪填。
每个P9目录含input_refs、facts/questions、appearance/occluder证据、preview、export。

新中立Beagle短probe：tmp/p11_beagle_bone_neutral_native_check_20260907_v4/capture。
15帧都有真实emitter，0/7/14帧另对bone world pose，误差0cm；anatomical yaw最大
误差7.6475e-5度，floor误差0.183258cm，root/level/像素通过。

父从最高控制器实际采样并完整执行的最终路径：

- Native Apartment：tmp/matrix_native_apartment_beagle_speaker_sampled_20260907_v2，
  delivery_v1为16秒双声道成片及P9导出，13 valid/11 deferred。实际地图
  apartment_0000；root、骨骼emitter、anatomical方向、像素和音频均通过。
  Beagle部分被桌沿遮挡（240帧visible_occluded），音箱240帧clear；两种外观reviewed。
- 酷家乐：tmp/matrix_kujiale_beagle_speaker_native_bone_20260907_v1/delivery，
  最高控制器完成原生240帧、自动双声道16秒及P9导出，8 valid/16 deferred。
  两源240帧clear；音箱reviewed，暖光下beagle coarse coat仍not_observable。
  父看过两段0/120/239实际RGB；各自parent_validation.json含PCM/骨骼/合同核对。

Native Apartment sampled v1的beagle240帧fully_occluded，保留为失败样本。复算
发现emitter射线clear而既有body proxy射线blocked；P5现对clear请求同时检查
两者，合法候选仍均匀抽。手调native_apartment_animal_speaker_v4仅是显式runtime
canary，不作为conditioned sampler结果。早前root-local Beagle成片也保留诊断，
最新家族验收使用上述bone-bound路径。

## 4. 未完成分类与边界

9个wall/ceiling源可加载，但挂装执行仍interface_not_implemented；不放到地上
冒充完成。像素局部遮挡、不可观测外观和各题拒出保留原分类及24类分母。
发声点/骨骼/geometry读回不是人工校准、碰撞支撑或formal admission。四个额外
动物的head/muzzle仍为source-specific joint_proxy，无额外显式mouth marker。
P7十条真人试听仍待记录，46段试产未启动。

## 5. Claude接口

source_declaration仍由正式source registry驱动；新attachment声明在
runtime_backends.spear_unreal.ue_emitter_attachment，启用条件由
visual_plan.ue_neutral_runtime_binding记录。native_runtime_binding_readback.json
和frame_readbacks.emitters保留实际附着与世界位姿、采样bone对照；P1 neutral
仍统一米/+Y/右手，P6/P9只消费实际合同。审计器及其测试未修改。
完整59资产与共同stage入口见P11_asset_bindings.md。

## 6. Owner事项

本次实现不需要新增授权。人工试听/视觉校准和正式数据集准入不能由机器记录
替代；超过46段或正式发布仍需另行授权（当前只授权46段且有前置验收条件）。

最终集成验证：14个相关测试文件共 **181 passed / 0 failed / 0 skipped**（12.46s），
日志 tmp/p11_parent_full_related_tests_20260907_v2.log。覆盖runtime profiles、UE刚体/
旧Apartment、条件采样、native room、P12 builder/mesh、Habitat绑定/捕获、
RoomPackage、P6/P9、tool index和question-driven room回归。上一轮179 passed/2 failed：

- test_inventory_preserves_external_animal_and_runtime_overlap: `assert 44 == 40`
- test_external_animal_is_not_silently_treated_as_rigid: `Failed: DID NOT RAISE HabitatStaticAssetError`

二者是四动物已登记后的旧预期；现验证重叠44、并集59，四动物经P12关节包解析；
无runtime registry只给external index时仍必须拒绝静态回退。没有改生产逻辑来迁就测试。
git diff --check通过。
