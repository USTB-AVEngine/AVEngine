# P12 Habitat 人形与生成动物资产包

## 1. 修改文件与提交

权威目录：48g-jump:/data/jzy/tmp/wt-multi-home-activity-integration，分支 codex/multi-home-activity-integration。实现提交为本报告首次加入分支的提交，可由 git log -1 -- 本文件定位。未 push、merge 或切换 Studio。

资产代码：src/avengine/assets/{contracts,kinematics,package,glb,glb_transcode,habitat_animation_normalization}.py；工具：tools/assets/{compile_animal_package,build_habitat_asset_package,verify_habitat_asset_runtime_readback,probe_habitat_skin_rest,render_habitat_action_review}.py；schema：schemas/{animal_asset_package_v1,m2_articulated_capture_request_v1}.schema.json；登记：examples/runtime/source_asset_runtime_profiles.json。

集成代码：src/avengine/capture/mp3d_multi_actor.py、tools/capture/capture_mp3d_multi_actor.py、src/avengine/assets/mp3d_region_actor_tracks.py、tools/capture/materialize_common_plan_habitat.py。此提交包含原始 P12 包执行依赖的共用计划物化能力；P5 采样器另报。相关新增/修改测试为 test_assets_compile_animal_package、test_capture_mp3d_multi_actor、test_mp3d_region_actor_emitter、test_mp3d_common_plan_materialization；docs/TOOL_INDEX.md 已重生成。

contact_order 改为包声明，双足 foot_left/foot_right，四足保留四个 paw。编译器使用请求指定的源、身份和 provenance；严格 loader 默认仍拒绝 research_candidate，只有显式 allow_research_candidate=True 可运行研究捕获。原包/原源文件未覆盖。

## 2. 测试

项目 Python /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python，PYTHONPATH=src:tmp/native_python_addons_v1。最终相关组合 **191 passed, 0 failed, 0 skipped，9.62 秒**，日志 tmp/p12_parent_final_tests_20260907.log。覆盖 assets compile/contracts/kinematics/package/glb/habitat/habitat_capture、capture_mp3d_multi_actor、emitter mapping、common-plan materialization、工具索引。包括 beagle 兼容、人形双足、生成动物、精确 AO handle、刚体、seed 映射和计划时钟。

14 个包 validate_animal_asset_package 通过，14 个研究请求内容/schema 校验通过，正式 canary 的 research_candidate 拒绝保留。14 个原生 Idle/Walking 首姿态探针均读取 root/joint/emitter/contact link：tmp/p12_native_readback_summary_20260907_v1.json。包验证及身份/修订对照：tmp/p12_parent_registry_integration_20260907_v1.json。提交前实查 55 资产、55 Habitat binding、17 SPEAR binding；全部 55 条 source revision 和既有 17 条 SPEAR binding 与修改前相同。

已修复 AO 配置 basename 相同造成取错模板：用 load_configs 返回的精确 template ID 取得 handle。M1 初期失败原文为 rig_from_sensor must be identity because world_from_rig is the formal camera/listener viewpoint，物化器恢复既有 identity sensor/listener 要求，没有放宽校验。超出 int32 的 seed 仅在 M1 边界 modulo 2**31，原 seed 与映射均记 receipt。

## 3. 验收产物及亲自核对

| asset_id | 包 manifest | 包校验/原生探针 |
|---|---|---|
| generated_abyssinian_ruddy_medium_standard_adult_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_abyssinian_20260906_v1/package/asset_manifest.json | pass |
| generated_border_collie_black_white_medium_standard_adult_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_border_20260906_v1/package/asset_manifest.json | pass |
| generated_british_shorthair_blue_medium_stocky_adult_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_british_20260906_v1/package/asset_manifest.json | pass |
| generated_pembroke_welsh_corgi_red_white_medium_standard_adult_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_corgi_20260906_v1/package/asset_manifest.json | pass |
| rocketbox_human_male_adult_01_top_blue_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_human_blue_20260906_v5/package/asset_manifest.json | pass |
| rocketbox_human_male_adult_01_top_burgundy_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_human_burgundy_20260906_v1/package/asset_manifest.json | pass |
| lead_b_rocketbox_professions_construction_male_01_original_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_human_construction_male_20260906_v2/package/asset_manifest.json | pass |
| lead_b_rocketbox_adults_female_adult_01_original_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_human_female_20260906_v3/package/asset_manifest.json | pass |
| rocketbox_human_male_adult_01_top_green_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_human_green_20260906_v1/package/asset_manifest.json | pass |
| rocketbox_human_male_adult_01_m5_1_candidate | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_human_male_base_20260906_v1/package/asset_manifest.json | pass |
| rocketbox_human_male_adult_01_top_yellow_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_human_yellow_20260906_v1/package/asset_manifest.json | pass |
| generated_labrador_yellow_medium_standard_adult_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_labrador_20260906_v3/package/asset_manifest.json | pass |
| generated_shiba_inu_red_medium_standard_adult_research_v1 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_shiba1_20260906_v1/package/asset_manifest.json | pass |
| generated_shiba_inu_red_medium_standard_adult_research_v2 | /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p12_package_shiba2_20260906_v1/package/asset_manifest.json | pass |

每包目录的上一级含 base_m2_request.json、habitat_binding_increment.json、native_readback.json。7 人形与 7 生成动物均完成导出/蒙皮/URDF/烘焙/加载和首个 Idle、Walking pose 原生读回；两个姿态探针不能扩称完整 gait 或物理接触认证。

最终 mixed capture 根目录 tmp/p12_native_capture_human_british_20260907_v1/，包含 common_plan.json、materialized_formal_registry_vertex_grounded_v1/、native_capture_formal_registry_vertex_grounded_v1/、native_capture_validation_v1.json。使用正式 registry 的原始 human blue/British Shorthair 包、原 asset_id/revision，无 binding delta 或改名包。

30 帧、240×320，15 Idle+15 Walking，真实 RGB/depth/semantic、root/joint/emitter、模态和 target-only、pixel truth、alignment 齐全。保存数组与原生 frame record 一致；root 最大误差 1.61e-7 m、joint 3.23e-8；深度 0.4043–9.8548 m。Human target-only 7346–9361 像素/帧，British 794–1431；实际可见比例分别 0.9529–1 和 0.1663–0.2655。

父 Agent 查看了原生 contact sheet：人有走停姿态变化，扫描网格遮住局部身体，猫被床大量遮挡；该图不能独立证明猫的完整外观或脚底接触。子代理另查看了两包 habitat_animation_review/idle_side_frame0_rgb.png、walk_side_frame0_rgb.png 的无遮挡外观与姿态；均为 Codex 图像检查，不是人工审核。

最终 actor root 取原生 MP3D floor y=0.07244700193405151 m。tmp/p12_parent_contact_vertex_audit_20260907_v1.json 精确 skin 顶点检查中，actor-root-zero 最低 y 为 human -0.0019656 m、British +0.0000454 m。旧 human+Border v4 曾错误用踝部/paw link 原点校地，将 human 下移约 10.7 cm；旧结果仅保留诊断，禁止作为最终 grounding 验收。Border actor-root-zero 精确最低 y=-0.012195 m，物理支撑仍未校准；没有用保守 AABB 宣称精确穿地量。

## 4. 未完成和证据边界

本项包实现及上述原生加载/动作/像素/发声点验收完成。所有包保留 research_candidate、qualification_claim=false、episode_counted=false。本项不含 RLR、人工可答性、collision/support-contact 正式资格、全四家族数据集验收。若干四足 gait contact 使用包内明确标注的 allow_unobserved_contact 低置信 fallback，不是人工接触校准；资产不从分母删除。Mixed 猫完整外观受遮挡，记 evidence_missing_or_unsampled，不补造像素证明。

## 5. Claude 接口

runtime_backends.habitat 的 asset_kind=articulated_m2_package，保留 manifest/request、semantic_template、resting_pose、package_revisions。Emitter 同时保留共用 semantic_anchor_id（human mouth）、包内 anchor_id（native muzzle）、joint_id、joint_from_anchor、offset_m、offset_space=joint_local、root_offset_m。Root-offset 仅用于计划几何；actual emitter 必须读 native joint/anchor。

共用 root 是 actor/asset root；planned_world_from_skin_root=world_from_actor @ package.actor_from_skin_root。物化器不重新规划、不改计划时钟，seed 映射仅作用于 Habitat 配置。camera calibration、单位 scale、动作 sample grid 显式验证；contact_order 从包读取。严格 loader 的默认 canary 限制保留。

## 6. Owner 决策

没有扩大权限、覆盖原源或删除安全措施。Human female/construction 使用同 asset_id 的实际可用源版本；Abyssinian 的嵌入 WebP 在当前 importer 不支持，派生包转 PNG 并保留来源；root 动态平移归 actor route，normalization.json 记录转换。本项没有新增 owner 决策；P7 人工试听和全矩阵后续验收仍保持各自边界。
