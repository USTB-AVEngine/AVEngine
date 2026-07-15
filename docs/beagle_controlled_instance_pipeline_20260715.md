# 比格受控 instance 级资产管线（2026-07-15）

## 当前结论

比格稳定模板路线已经完成一次完整的 instance 级可控性验证：代码先枚举
`3 × 3 × 3 × 3 = 81` 个合法绝对属性请求，再从中构建 9 个 OFAT 实例——一个
基准实例，加上四个属性各两个非基准值。这样每个离散值都至少真实实现一次，
同时可以检查变化是否只作用于目标属性。9 个实例均完成 GLB 回读、Walking/Idle
变形门、UE 导入、共享 cook、Apartment 主视图、同步 Top-down、双耳狗叫声和
组合审核视频；最终结果为 18/18 passed、0 failed、0 incomplete。

这批仍是 `research_candidate`，不是 `formal_dataset_asset`。自动技术门已经通过，
人工视觉状态仍为 pending，36 cm 比格参考肩高也仍是待许可证快照替换的 provisional
标定值。

浏览器审核页：

- `http://<服务器IP>:8102/docs/beagle_stable_apartment_review_20260715.html`
- [本地 HTML](/data/jzy/code/AVEngine/docs/beagle_stable_apartment_review_20260715.html)
- [认证 manifest](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_apartment_review_v3_20260715/review_manifest.json)

## 找到并采用的属性控制文件

旧版 profile 保留，不覆盖：

- [dog_beagle_rocketbox_stable_v1.json](/data/jzy/code/AVEngine/external/SPEAR/data/controlled_source_attributes_v1/candidate_profiles/animal/dog_beagle_rocketbox_stable_v1.json)

本轮权威 profile 是：

- [dog_beagle_rocketbox_stable_v2.json](/data/jzy/code/AVEngine/external/SPEAR/data/controlled_source_attributes_v1/candidate_profiles/animal/dog_beagle_rocketbox_stable_v2.json)

属性控制并非只存在于一个地方。它按生命周期分成三层：

1. profile 定义该品种允许的属性域、实现参数、完整 prompt 模板、骨架、动作和
   声学合同；
2. [instance_requests.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_inputs_v2_20260715/instance_requests.json)
   保存 81 个实例各自的绝对属性、seed、完整 prompt、request SHA-256 和执行计划；
3. 每个已实现实例各有独立的 `manifest.json`、GLB、贴图、变形审计和 UE registry。
   例如基准实例：
   [manifest.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_realizations_v2_20260715/dog_beagle_rocketbox_stable_dd428da8c82a/manifest.json)。

因此，instance 身份 JSON 只记录“这只狗本身是什么”，不会写
`from=medium → to=dark` 之类相对编辑历史。具体实现参数单独保存在
`attribute_operations`，便于复现但不污染个体语义。

## 四类属性及确定性实现

每类均严格限制为三个值：

| 属性 | 合法值 | 当前实现 |
|---|---|---|
| `size` | `small / medium / large` | 由目标物理 profile 做统一尺度 `0.90 / 1.00 / 1.10`；不改变拓扑、骨架或权重 |
| `body_build` | `slim / standard / stocky` | 只对语义躯干顶点做围度缩放 `0.92 / 1.00 / 1.08`，保持长度、四肢和动作不变 |
| `coat_tone` | `light_tricolor / standard_tricolor / dark_tricolor` | 比格专用三色毛亮度增益 `1.12 / 1.00 / 0.86`，保留黑/棕/白分区和 PBR |
| `life_stage` | `young / adult / senior` | young 头部比例 `1.08`；adult 为基准；senior 对口鼻语义 mask 混入 `0.75` 灰度 |

这里把用户所说的“年龄”规范化为 `life_stage`，避免伪造精确岁数。毛色名称由
比格 profile 自己定义，不会错误复用金毛的 `golden` 属性。

每个请求仍自动生成一条包含全部属性的 FLUX.2 prompt 和 negative prompt；但这条
稳定模板 canary 不会为了改尺寸、体型、三色亮度或年龄迹象而重复 image-to-3D。
FLUX.2 的当前策略是 `qa_and_optional_semantic_texture_detail_only`：可提供语义参考或
受 mask 约束的细节，批量运行时则优先使用确定性几何/材质操作。这样既保留完整
prompt provenance，也避免 Pixal3D/TRELLIS 每次重建带来的融合四肢、腹部缺面和
重新绑定风险。

## 81 个请求与 9 个 OFAT 实例

81 个请求对四个属性形成完整笛卡尔积；每个值正好出现 27 次。执行输入包括：

- [generation_plan.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_inputs_v2_20260715/generation_plan.json)
- [execution_jobs.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_inputs_v2_20260715/execution_jobs.json)
- [qa_pair_plan.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_inputs_v2_20260715/qa_pair_plan.json)
- [execution_preflight.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_preflight_v2_20260715/execution_preflight.json)

9 个已实现实例为：

| OFAT 角色 | size | body_build | coat_tone | life_stage |
|---|---|---|---|---|
| baseline | medium | standard | standard_tricolor | adult |
| size-small | small | standard | standard_tricolor | adult |
| size-large | large | standard | standard_tricolor | adult |
| build-slim | medium | slim | standard_tricolor | adult |
| build-stocky | medium | stocky | standard_tricolor | adult |
| coat-light | medium | standard | light_tricolor | adult |
| coat-dark | medium | standard | dark_tricolor | adult |
| age-young | medium | standard | standard_tricolor | young |
| age-senior | medium | standard | standard_tricolor | senior |

这 9 个实例不是 9 个临时 prompt，而是各有稳定的内容哈希 ID。完整对应关系、静态
对比、隔离 Walking/Idle 和 GLB 路径见：

- [OFAT review manifest](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_review_v2_20260715/review_manifest.json)
- [本地静态/隔离动画审核页](/data/jzy/code/AVEngine/docs/beagle_stable_instance_review_20260715.html)

OFAT 通过后，后续可由代码从 81 个组合中任意平衡采样；不需要再为每种组合人工
调整骨架或方向。若以后扩展属性域，则必须先做新的模板资格认证，而不能悄悄把
第四个值加入现有 profile。

## 动画、UE 与音频结果

稳定基础 GLB 是：

- [animated_beagle_native_weights.glb](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/rocketbox_beagle_native_weight_retarget_v1_20260714_r2/animated_beagle_native_weights.glb)

9 个 realization 的 `topology_uv_skin_sha256_before/after` 完全一致，Walking/Idle
action hash 也完全一致。变形门 9/9 passed；UE import/readback 9/9 passed。第一次
Apartment 尝试正确保留为失败证据：渲染本身完成，但旧 QA 不认识
`beagle-Pelvis / beagle-Spine2 / beagle-L/R-Foot`。现在的通用映射要求同一非空
命名空间同时具备 `Pelvis + Spine2 + L/R Foot + Tail`，不会把普通人形误判为四足。

最终 UE 证据：

- [UE import result](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_ue_import_v2_r3_20260715/ue_import_result.json)
- [Apartment specs](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_apartment_specs_v3_20260715/spec_manifest.json)
- [18/18 batch status](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/beagle_stable_ofat_apartment_specs_v3_20260715/batch_render_status.json)

自动 QA 汇总：

| 检查 | 结果 |
|---|---:|
| UE Walk/Idle | 18 / 18 passed |
| 最大躯干 forward 误差 | 0.9115° |
| 最小 body-up 对齐 | 0.8625 |
| 最大地面穿透 | `7.11e-15 cm`（浮点零） |
| small / medium / large UE 总高度 | 50.1379 / 55.7083 / 61.2800 cm |
| 每条音频狗叫事件 | 7 |
| 最小狗叫静音间隔 | 0.85 s |
| 音频模式 | `repeated_events_with_silence_gaps` |

## 迁移到其他声源资产

有稳定且许可证清晰的原生模板时，复用这条路线：定义品种 profile → 生成完整请求
JSON/prompt → 9 个 OFAT canary → 保拓扑实例化 → Walk/Idle 变形 → 一次批量 UE import
和共享 cook → Apartment 音视频。颜色、尺寸和小幅语义体型优先确定性实现；FLUX.2
只处理受 mask 约束的纹理语义。

没有稳定模板的动物不能伪装成这条路线。它必须走严格生成路线：完整属性 JSON →
FLUX.2 → Pixal3D 默认/TRELLIS 对照 → 静态四肢、尾巴、腹部和 watertight 门 →
动画前整 90°人工方向门 → mesh-matched rig → 该物种动作族 → Walk/Idle/音频/UE QA。
鸟、蛇、昆虫和水生动物必须使用自己的动作族，不能套用比格 Walking。
