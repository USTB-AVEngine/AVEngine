# 生成式四足动物脚掌方向锁定与动作移植技术方案

## 1. 目标与当前状态

本方案解决生成式四足动物在 Walking/Idle 重定向后出现的脚掌周期性内翻、
外翻、横摆或接近整圈旋转的问题。它只负责**动作与目标骨架之间的方向兼容**，
不把贴图、网格、权重或几何缺陷伪装成动作问题。

2026-07-16 的 Beagle v24 canary 已验证该方案：

- 前脚相对髋部的横向摆动由腿长的 `7.20%–7.44%` 降为
  `0.035%–0.053%`；
- 后脚由 `13.87%–15.16%` 降为 `0.087%–0.104%`；
- 四个末端脚骨的 yaw 周期摆动均低于 `0.001°`；
- 目标 GLB 的 mesh、skin、PBR、材质以及原始 BIN 前缀逐字节不变；
- 修复动作已传播到 9 个受控实例，9/9 GLB 和 18/18 Walk/Idle 视频完成回读。

当前状态仍为 `research_candidate`。许可证、静态几何、权重形变、UE Apartment
和媒体 QA 是相互独立的门禁，不能因为脚掌方向通过而自动视为通过。

## 2. 根因

旧流程 `world_space_rest_offset_chain_arc_length_slerp_v1` 将源动作末端脚骨的
世界旋转继续传给目标脚骨。即使两套骨架语义相同，它们的局部轴、bone roll、
rest quaternion 和末端骨长度仍可能不同。源骨架中正常的脚掌旋转因此会在目标
骨架上被解释为向内、向外或绕纵轴翻转。

该问题与以下因素无直接因果关系：

- FLUX.2 参考图的毛色；
- Pixal3D/TRELLIS 的面数；
- PBR 贴图；
- 实例级大小、体型、毛色或年龄属性；
- 单纯更换随机 seed。

它也不能通过旋转整个动物的 cardinal yaw 解决。cardinal yaw 只处理动物整体
朝向；脚掌内外翻发生在腿链末端的局部动作基准中。

## 3. 两阶段解法

### 3.1 每个骨架/运动家族建立一次锁脚 motion carrier

先对一个通过静态方向审核的目标骨架生成正确的 Walking/Idle 动作载体。当前
验证配置为：

- retarget mode：`world-rotation-foot-ik-v3`；
- terminal policy：`lock_target_rest_world_v1`；
- Beagle 四个末端脚骨：`bone_6`、`bone_9`、`bone_16`、`bone_19`；
- actor-local front axis：positive X；
- 动作：仅 `Walking` 与 `Idle`。

对于每一帧，髋、膝、踝等非末端关节仍按动作源重定向，保留抬腿、落脚和
前后摆动。末端脚掌的 actor-local world orientation 则固定为目标 rest pose 的
orientation：

```text
Q_target_foot_world(frame) = Q_target_foot_rest_world
```

脚掌位置仍由父链和 foot IK 推动，因此“锁方向”不等于把脚钉在一个世界坐标。
场景中的路径转弯由 UE/SPEAR actor transform 旋转整个角色，动作自身仍在
actor-local 坐标中保持一致。

犬科、猫科和有蹄类必须使用各自的动作家族；不能因为都是四条腿就让猫、马
复用狗的同一套步态。

### 3.2 向外观权威 GLB 只移植动作二进制

目标 GLB 是 mesh、skin、UV、PBR 和材质权威；motion carrier 只提供动作。
使用：

```bash
cd /data/jzy/code/AVEngine/external/SPEAR

/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/transplant_compatible_glb_animations.py \
  --target-glb /absolute/path/to/appearance_authority.glb \
  --source-glb /absolute/path/to/locked_terminal_motion_carrier.glb \
  --output-glb /new/non_overwriting/path/animated_walk_idle.glb \
  --manifest /new/non_overwriting/path/animation_transplant_manifest.json \
  --action Walking \
  --action Idle \
  --rest-tolerance 1e-5
```

工具执行以下硬门禁：

1. 每个被动画驱动的 node name 必须在目标中唯一存在；
2. 对应 node 的 parent name 必须完全一致；
3. local translation、quaternion（允许 `q/-q` 等价）和 scale 必须在容差内；
4. Walking 与 Idle 必须各自唯一解析；
5. 只复制动作引用的 accessor/bufferView，并把它们追加到目标 BIN；
6. animation channel 重新映射到目标 node index；
7. `asset/scenes/nodes/meshes/skins/materials/textures/images/samplers` 的规范化
   SHA-256 在操作前后必须一致；
8. 完整目标 BIN 前缀在操作前后必须逐字节一致；
9. 输出 GLB 必须独立回读并再次验证；
10. 输出路径存在时拒绝覆盖。

不要仅为复制动作而让目标 GLB 再经过一次 Blender 导出。实测 Blender
re-export 会改变 topology、UV、skin 或 buffer 排列，即使视觉上暂时相似，
也会破坏“只改动作”的可验证性。

## 4. 完整自动门禁

每个新的骨架/运动家族至少执行一次以下 canary：

| 阶段 | 必须验证 | 失败处理 |
|---|---|---|
| 2D 参考 | 四肢分离、尾巴分离、静态站姿、完整轮廓 | 修改 prompt/姿势 guide 后重生，不进入 3D |
| 静态 3D | 四肢独立、闭合腹部、无缺面、方向为审核后的整 90° | rejected；锁脚不能修复静态几何 |
| 骨架 | 四条语义腿链、父子层级、rest matrix、落地高度 | 建立新的 fitted skeleton/motion carrier |
| 权重 | 左右腿不串权、腹部不拉丝、尾巴不粘腿 | 单独修权或 rejected |
| 动作兼容 | node/parent/rest TRS 兼容 | 拒绝二进制移植，禁止强行套用 |
| 脚掌方向 | 四脚横向 excursion 与 yaw 周期范围 | 超阈值则修 retarget policy |
| GLB | 独立导入、Walking/Idle、材质纹理、BIN 认证 | rejected |
| 媒体 | Front/Side/Quarter/Feet/Skeleton、UE Apartment | 视觉门独立记录 |

推荐记录两个独立数值：

- `paw_relative_to_hip_lateral_excursion_ratio_of_mesh_diagonal`：脚相对髋的
  横向摆动，排除身体整体侧移；
- `paw_yaw_excursion_degrees`：末端脚骨在一个动作周期内的 yaw 范围。

数值门用于发现回归，视觉门用于判断真实可见程度。严格形变审计的失败记录
必须保留；不得因为用户接受宽松视觉结果就把严格 JSON 改写为 passed。

## 5. 泛化边界

| 资产类型 | 泛化方式 | 是否需要逐实例人工修 |
|---|---|---|
| 同一骨架的大小、体型、毛色、年龄实例 | 直接继承同一锁脚 carrier | 不需要 |
| 新犬种但共享兼容 dog skeleton/rest | 通过兼容门后直接移植 | 通常不需要 |
| 新犬科骨架或不同 rest pose | 建立一次新的 dog-family carrier | 只审核一次 canary |
| 猫科 | 建立猫科步态与 paw carrier | 每个骨架家族一次 |
| 马、牛、羊等有蹄类 | 建立物种合适的 hoof carrier | 每个骨架/步态家族一次 |
| 鸟类地面运动 | 两脚爪末端锁定，翅膀使用独立规则 | 需要 bird-biped canary |
| 鸟类飞行 | 不能使用地面脚掌策略 | 需要飞行骨架与机翼规则 |
| 蛇、鱼 | 无四足末端语义 | 使用完全不同的运动方案 |

批量化单位因此是“骨架/运动家族”，不是“每一个随机实例”。同一家族 canary
通过后，代码可以批量处理实例属性；只有出现新骨架层级、不同 rest pose 或新
运动类型时才重新建立 carrier。

## 6. 本方案不解决的故障

- Pixal3D/TRELLIS 静态输出的并腿、尾巴粘腿、腹部缺面；
- 两条腿已经在 mesh 中连成一个组件；
- skin weights 把左右腿、腹部或尾巴错误绑定在一起；
- 整体 FRONT/cardinal yaw 选择错误；
- 动物物种对应了不合理的动作或速度；
- 声音类别、重复节奏或音视频同步错误。

这些问题必须由各自的静态几何、方向、权重、动作选择和音频 QA 门处理。

## 7. Beagle v24 可复现实证

- 外观目标 v22 SHA-256：
  `1af28de3299cde11cc4bbb61730459fd9044e1f4838348bec6d26ad8cdd72426`
- 锁脚 motion carrier v12 SHA-256：
  `083cafc7d99ae1e9e752b512adedef71bf3a124f1d648493874fddc8abc62117`
- 输出 v24 SHA-256：
  `90cd41a9d6e19e2ff9c950d8dc2c35672b86d60ec971ae48985336af655125bf`
- 认证 manifest：
  `/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/dog_beagle_three_quarter_30deg_i23d_bakeoff_v10_20260714_r1_candidate657/trellis2/seed6102_mesh_first_pbr_animation_v24_binary_action_transplant_locked_paws_20260716/animation_transplant_manifest.json`
- 横摆审计：同目录 `lateral_gait_audit.json`
- A/B 页面：
  `http://127.0.0.1:8102/docs/beagle_locked_paw_animation_review_20260716.html`
- 九实例页面：
  `http://127.0.0.1:8102/docs/beagle_mesh_first_instance_ofat_locked_paws_review_20260716.html`

对应实现为
`external/SPEAR/tools/transplant_compatible_glb_animations.py`，单元测试为
`external/SPEAR/tests/tools/test_transplant_compatible_glb_animations.py`。

## 8. 绑定批处理中的自动锁脚闭环

`external/SPEAR/tools/run_controlled_animal_lod_binding.py` 现在可以通过
`--locked-paw-motion-profile quadruped_dog_locked_paws_v2` 在同一个原子批次内完成：

```text
人工批准的精确 100k LOD 方向决定
  -> Quaternius Dog 固定骨架/权重绑定
  -> 保留绑定器原动作版本作为 pre-lock 证据
  -> 二进制 animation transplant，仅替换 Idle/Walking
  -> 验证目标 GLB 原始 BIN 前缀逐字节不变
  -> 41 帧四脚 lateral excursion 与 terminal yaw 审计
  -> 最终 GLB readback
```

当前 profile 固定 motion carrier SHA-256 为
`083cafc7d99ae1e9e752b512adedef71bf3a124f1d648493874fddc8abc62117`。
硬门为四脚最大
`paw_relative_to_hip_lateral_excursion_ratio_of_mesh_diagonal <= 0.005`，以及
`paw_yaw_excursion_degrees <= 0.1`。任一脚超阈值都会让该资产失败，不会悄悄回退
到旧动作。

执行预检在不修改 v24 批准产物的全新目录中通过：最大横向比例
`0.0010399684`，最大末端 yaw excursion `0.000289899°`，原目标 BIN 前缀保持
不变，最终 GLB 回读一个 skin、两个动作。绑定 runner、animation transplant 和
review 相关测试共 32/32 通过。

新版浅色比格的原始 Pixal GLB 还独立重建了审核用 100k LOD。重建结果与方向页
显示的 LOD 逐字节一致，SHA-256 都是
`150923ea84d361558daed5ea4b622b6ecad2a105fb8fb4156d7231570d98814a`；
99,993 面，`boundary_cracks_introduced=0`。因此绑定时不会重新抽取或替换 mesh。
当前仍必须等待方向页生成不可变人工决定，不能把临时的 180° 选择当作批准。
