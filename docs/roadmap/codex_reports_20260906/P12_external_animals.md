# P12 新增四个外部动物：Habitat 绑定修正

## 1. 文件/提交关联

权威工作目录为 `48g-jump:/data/jzy/tmp/wt-multi-home-activity-integration`；本次读取的共享 HEAD 为 `3d81f3dbd04673f280af5e10344622fa398d4c33`。本报告随P11全资产整合提交；registry和UE整合已完成，旧v8增量和原package字节保留。

本次源码修改为 `tools/assets/build_habitat_asset_package.py`：`runtime_backend.emitter.offset_m` 和 `joint_from_anchor` 明确标为 `joint_local`；保留实际 `anchor_id`/`semantic_anchor_id`/`joint_id`；新增成对可选的 `root_offset_m`/`root_offset_source`：提供时校验三维 finite 与来源，允许真实 zero；缺失时保持 joint-local binding 并标记 `root_offset_status=not_measured`，不伪填 0。新增 builder 单测为 `tests/unit/test_build_habitat_asset_package.py`。同时新增逐帧蒙皮几何核查工具 `tools/assets/audit_habitat_mesh_grounding.py` 与 `tests/unit/test_assets_habitat_mesh_grounding.py`。

逐资产 v9 binding increment 与 native Idle0 actor-root muzzle reference 位于四个 package 根目录；四资产候选汇总为 `tmp/p12_external_animals_candidate_increment_20260907_v9.json`。

## 2. 测试（含失败原文）

最终针对本次 builder 修正的单测为 **11 passed, 0 failed**，覆盖 builder emitter frame separation、optional/zero/partial/finite root reference、旧 spec 兼容性和 exact mesh audit helper；`py_compile tools/assets/build_habitat_asset_package.py` 与 `git diff --check` 通过。其中 mock pipeline 测试确认旧 spec 缺少 root reference 仍能进入既有 build 流程；此前四包 native/build 证据已完成，本次按要求未重跑。

保留的真实失败现场及修复如下：

- 首个 dark spec 在输入校验停止：`P12BuildError: spec.contact_order must be the four-foot or two-foot order`。补上四足 contact order 后，fresh v2 build pass。
- standalone native readback 未激活 pinned runtime 时停止：`ModuleNotFoundError: No module named 'habitat_sim'`。使用项目固定 Habitat/Magnum/RLR 环境后 pass。
- Siamese 首次独立 import 出现：`ModuleNotFoundError: No module named 'magnum'`，随后 `ImportError: initialization failed`。核对 site 路径并使用同一 pinned 环境后 pass。
- 默认 world-contact cadence helper 对 source-specific forward 方向停止：`WorldContactError: paw cycle has fewer than three supported stance frames`。这暴露了默认 +X 假设；四包最终采用逐帧 exact skinned mesh/contact audit，未把该失败隐藏为 contact qualification。
- 一次测试收集命令错误引用不存在的 `tests/unit/test_tool_index.py`：`ERROR: file or directory not found`；改用仓库实际的 `test_tool_index_current.py` 后相关套件通过。

## 3. 实际产物与看图结果

四个 v9 emitter 均使用 `offset_m=[0,0,0]`、`offset_space=joint_local`，同时保存由 native Habitat Idle0 和 `actor_from_skin_root` 推导的非零 actor-root planning reference：

| asset | muzzle joint | root_offset_m（canonical actor-root） | local_anatomical_forward_axis |
|---|---|---|---|
| Burmese dark sable | `bone_4` | `[-0.431647386, 0.631968627, 0.028546204]` | `[-0.983131484, 0, 0.182900205]` |
| Burmese standard sable | `bone_37` | `[-0.426032845, 0.656887181, 0.020020776]` | `[-0.989138652, 0, 0.146985462]` |
| Jack Russell Terrier | `bone_4` | `[-0.455348770, 0.435070132, -0.139530839]` | `[-0.913952286, 0, -0.405821660]` |
| Siamese standard seal point | `bone_4` | `[0.264351345, 0.435039921, -0.047489048]` | `[0.999937570, 0, 0.011173914]` |

binding correction 与来源文件：

- `tmp/p12_package_burmese_dark_sable_20260907_v2/habitat_binding_increment_v9.json`
- `tmp/p12_package_burmese_dark_sable_20260907_v2/native_idle0_actor_root_muzzle_reference_v1.json`
- `tmp/p12_package_burmese_standard_sable_20260907_v1/habitat_binding_increment_v9.json`
- `tmp/p12_package_burmese_standard_sable_20260907_v1/native_idle0_actor_root_muzzle_reference_v1.json`
- `tmp/p12_package_jack_russell_terrier_20260907_v1/habitat_binding_increment_v9.json`
- `tmp/p12_package_jack_russell_terrier_20260907_v1/native_idle0_actor_root_muzzle_reference_v1.json`
- `tmp/p12_package_siamese_standard_seal_point_20260907_v1/habitat_binding_increment_v9.json`
- `tmp/p12_package_siamese_standard_seal_point_20260907_v1/native_idle0_actor_root_muzzle_reference_v1.json`

四个原 package 的 `native_readback.json`、rest probe、Idle/Walking review 和 mesh grounding audit 均保持先前 pass 结果；v9 只刷新 binding increment，不重渲。逐帧 exact grounding audit 的 source mesh rest extents（m）为 dark `[0.898979, 0.701813, 0.243874]`、standard `[0.900236, 0.711169, 0.243771]`、Jack `[0.936701, 0.571382, 0.283846]`、Siamese `[0.939699, 0.723720, 0.308736]`。`asset.json` 中的 small/medium 是来源属性；没有据此改写 runtime scale，包保持 `uniform_scale=1`。

我查看了四个 Habitat Idle/Walking side contact sheet：standard Burmese 与 Jack Russell 的外观和姿态清晰；dark Burmese 与 Siamese 因 Phong review 光照较暗，但轮廓和姿态仍可辨。该查看属于 Codex 图像检查，不替代人工视觉验收。

## 4. 未完成分类

- 四条记录已进入共同registry；资产包仍保持 `research_candidate`、`qualification_claim=false`。
- 四条UE导入/动作与共同stage原生读回也已完成，证据见P11_asset_bindings报告。
- 未宣称人工视觉验收、碰撞/support-contact 资格、RLR 音频或正式数据集 admission。
- contact joint origin 是解剖参考，不是 sole；exact mesh bottom 是几何核查，不能替代物理支撑测量。
- 四资产的 anatomical forward 是各自 native/source 对账后的水平单位轴，不能统一写成 +X；默认 cadence helper 的 +X 失败已保留为诊断。

## 5. Claude 接口

后续 `runtime_backends.habitat.emitter` 应按以下字段消费：`anchor_id`/`semantic_anchor_id` 保留 `muzzle`，`joint_id` 与 `joint_from_anchor` 指向包内实际 native joint，`offset_m`/`native_offset_m` 的 `offset_space`/`native_offset_space` 为 `joint_local`；`root_offset_m` 的 `root_offset_space` 为 `final_scaled_asset_root`，仅作为 route planning approximation。每个 root offset 都绑定 `native_idle0_actor_root_muzzle_reference_v1`，包含 readback、joint mapping、`actor_from_skin_root` 和推导公式。

公共坐标约定是右手、`+Y` up、米制；`local_anatomical_forward_axis` 是资产特定字段，来源为 native Idle0 body-root 到 muzzle 的水平投影。实际 emitter 必须继续读取 native joint/anchor，不得用 root offset 代替 native readback。

## 6. Owner 事项

本项没有新增需要owner决定的工程事项；既定的人工试听与正式admission边界保持。
四条registry已按v9的实际axis、actor_from_skin_root及native muzzle reference集成。
公共坐标约定不替代资产解剖方向；实际emitter始终使用各自native joint/bone。

父集成核对：正式runtime registry现为59条/59 Habitat/59 SPEAR；四条新增的
canonical muzzle参考和anatomical axis来自v9 native reference，两backend使用
各自真实joint/bone。source未提供realized body_build/life_stage，分别保留unknown；
只为research记录接纳已观测coat值，没有虚构L9三水平域或放宽qualified/formal要求。
记录与验证在tmp/p11_parent_external_animal_registry_20260907_v2。
父查看native_four_animals.png：三者形体与pose变化可辨，dark sable在黑背景上
对比度低；不据此声称其外观可识别或人工视觉验收。尺寸保持源几何，不按small/medium
文字额外缩放。父保留原18个包字节，未为这次metadata集成重渲染。

最终父整合回归181 passed/0 failed/0 skipped（12.46s），包括上述builder/mesh和绑定检查；日志 tmp/p11_parent_full_related_tests_20260907_v2.log。
