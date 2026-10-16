# 稳定动物模板 Walk/Idle 路线（2026-07-14）

> 当前分类：`research_candidate`。本页记录自动审计与代理视觉检查结果；用户已
> 暂缓本轮人工审核，因此没有任何条目被标记为 `human_approved` 或
> `formal_dataset_asset`。12 个模板的 UE Apartment Walk/Idle 自动验证均已完成；
> 正式 mesh-vs-furniture 碰撞门和用户人工审核仍待完成。

## 结论

批量动物的默认稳定运行时路线改为“已审计的闭合模板 + 原生动作 + 受控外观
变体”。Pixal3D 继续作为新几何研究分支，但不再让单张侧视图重建的未知拓扑
直接进入大批量绑定。原因不是面数不足：问题在 Pixal3D 输出中已经存在，主要是
单视图近/远肢体歧义、肢体桥接、非流形边和不匹配的皮肤变形；减面和绑定只能
放大这些缺陷，不能把它们修成稳定模板。

当前稳定链路为：

```text
attribute_profile
  → 平衡采样完整绝对属性 JSON
  → 选择已审计 species/template_id（几何、骨架、服装/毛发轮廓锁定）
  → 显式语义材质 allowlist + 确定性颜色/纹理计划
  → 原生 Walking + Idle
  → GLB 导出回读
  → 逐帧 skinned-deformation gate
  → authored axis + 仅整 90° cardinal yaw（禁止自动细角度）
  → UE Apartment / 轨迹 / 音频 / 碰撞 QA
  → source_asset_v2
```

FLUX.2 仍可生成受控外观参考或 mask 内纹理候选，但不能改变已锁定的运行时
拓扑、骨架、权重或动作。对纯颜色，优先使用确定性语义材质变换；每次变换均
保存绝对属性、允许修改的材质名、sRGB 参数、输入/输出哈希和不变量检查。

## 12 个原生 Walk/Idle 模板

权威机器注册表：
[registry_manifest.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_stable_template_registry_v1_20260714/registry_manifest.json)，
SHA-256 `e49361673d481c575dc78cff0f6065eddb90cffe2fd70bb283da1c409d66182a`。

总览图：
[walking_overview_12_materials.png](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/walking_overview_12_materials.png)，
SHA-256 `fc8d0581b0ece84ce423f5a37349c463d767757e75aa4c2dd0622d22b2742582`。

这些资产来自 Quaternius Ultimate Animated Animal Pack 的原生 FBX，许可证为
CC0-1.0。FBX 中旧 Principled BSDF `alpha=0` 会令 GLB 材质透明；提取器只把
该遗留 alpha 修复为 1，并保留原网格、骨架、权重、材质颜色和原生动作。
所有模板作者朝向为 `-Y`，运行时只施加 `+90°` cardinal yaw 映射到世界 `+X`；
没有使用自动 fine-yaw。

| 模板 | 三角面级别 | Walk | Idle | 当前自动状态 |
|---|---:|---|---|---|
| `Alpaca` | 2,060 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Alpaca_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Alpaca_idle_side.mp4) | GLB/材质/变形/UE Walk+Idle通过，人工待审 |
| `Bull` | 2,418 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Bull_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Bull_idle_side.mp4) | 同上 |
| `Cow` | 2,450 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Cow_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Cow_idle_side.mp4) | 同上 |
| `Deer` | 2,098 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Deer_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Deer_idle_side.mp4) | 同上 |
| `Donkey` | 2,000 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Donkey_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Donkey_idle_side.mp4) | 同上 |
| `Fox` | 1,848 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Fox_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Fox_idle_side.mp4) | 同上 |
| `Horse` | 2,182 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Horse_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Horse_idle_side.mp4) | 同上 |
| `Horse_White` | 2,182 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Horse_White_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Horse_White_idle_side.mp4) | 同上 |
| `Husky` | 1,920 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Husky_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Husky_idle_side.mp4) | 同上 |
| `ShibaInu` | 1,950 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/ShibaInu_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/ShibaInu_idle_side.mp4) | 同上 |
| `Stag` | 2,054 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Stag_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Stag_idle_side.mp4) | 同上 |
| `Wolf` | 1,962 | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Wolf_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_ultimate_native_walk_idle_v2_media_20260714/videos/Wolf_idle_side.mp4) | 同上 |

逐帧门禁检查原网格边长相对 rest pose 的扩张：`>0.07` 进入 review、`>0.08`
自动拒绝。12 个模板的 Walk 和 Idle 均通过；Pixal 比格 v7/v8 的 Walk 在同一
门禁中被拒绝，而 Idle 通过，说明该门禁能区分“静态看起来完整”和“动画时
拉丝”。

## Husky UE Apartment Walk/Idle canary

首个稳定原生模板已跑通“GLB → UE import/readback → 共享 cook → Apartment
Walk/Idle → 轨迹图 → 16 kHz 双耳音频 → registry”的完整链路。用户已暂缓本轮
人工审核，因此状态是 `agent_checked_pending_human_review`；机器 registry 强制
保存 `human_visual_review=pending`、`formal_registry_promotion=false`。

| 动作 | 带标注审核 | 主视图 | Top-down | 音频事件 |
|---|---|---|---|---|
| Walking | [审核成片](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/clips/stable_dog_husky_quaternius_ultimate_husky_v1/camera_pass_table_loop_walking/videos/side_by_side_review_annotated.mp4) | [主视图](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/clips/stable_dog_husky_quaternius_ultimate_husky_v1/camera_pass_table_loop_walking/videos/apartment_v1_view0.mp4) | [Top-down](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/clips/stable_dog_husky_quaternius_ultimate_husky_v1/camera_pass_table_loop_walking/videos/topdown_review.mp4) | [schedule](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/clips/stable_dog_husky_quaternius_ultimate_husky_v1/camera_pass_table_loop_walking/binaural_source_schedule.json) |
| Idle | [审核成片](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/clips/stable_dog_husky_quaternius_ultimate_husky_v1/camera_pass_table_loop_idle/videos/side_by_side_review_annotated.mp4) | [主视图](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/clips/stable_dog_husky_quaternius_ultimate_husky_v1/camera_pass_table_loop_idle/videos/apartment_v1_view0.mp4) | [Top-down](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/clips/stable_dog_husky_quaternius_ultimate_husky_v1/camera_pass_table_loop_idle/videos/topdown_review.mp4) | [schedule](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/clips/stable_dog_husky_quaternius_ultimate_husky_v1/camera_pass_table_loop_idle/binaural_source_schedule.json) |

认证 registry：
[stable_dog_husky_quaternius_ultimate_husky_v1.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_husky_v1_20260714/clips/stable_dog_husky_quaternius_ultimate_husky_v1/registry/stable_dog_husky_quaternius_ultimate_husky_v1.json)。

自动 QA 与代理抽帧检查结果：

- Walking/Idle 均为 18 秒、270 帧；主视图 960×720，审核视频 1280×480，
  Top-down 750×700；审核与 Top-down 都携带 16 kHz 双声道音频。
- Walking 四个路径窗口的身体朝向误差依次为 0.78°、0.78°、1.65°、0.79°；
  `Back → Torso3` 是身体纵轴，未使用头朝向或 fine-yaw 推断。
- 两段全程 floor contact 通过；最大穿透约 `7.1e-15 cm`（浮点数值零），根部
  roll/pitch 均为 0°。UE 动态 bounds 高度为 Walking 46.25–47.51 cm、Idle
  49.20–49.71 cm，属于可接受的中小型 Husky 尺度。
- Walking 按要求从相机右后方经过左前方并绕桌一圈；离开 FOV 和家具遮挡是
  该轨迹的预期 QA 属性，Top-down 同步记录完整路径。
- 狗叫被能量阈值切为短事件并在18秒内排成7次，最小静音间隔0.85秒；没有把
  单次叫声无缝循环成持续噪声。
- 代理抽查原始帧与审核合成未见 Pixal 比格式腹部空洞、脚部拉丝、横向跑或
  倒退；这仍不等价于用户人工批准。

## 其余 11 个原生模板 UE 批量验证

剩余 Alpaca、Bull、Cow、Deer、Donkey、Fox、Horse、Horse White、Shiba Inu、
Stag、Wolf 已按完全相同的代码路径批量跑完。统一懒加载审核页：
[stable_animal_video_review_20260714.html](/data/jzy/code/AVEngine/docs/stable_animal_video_review_20260714.html)。
页面展示绝对文件路径，Walk/Idle 可分别筛选；它只播放证据，不写审批状态。

机器汇总：
[batch_qa_summary.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_remaining11_v1_20260714/batch_qa_summary.json)
（SHA-256 `26549176c706d90d5c69c1a15c2405a37fde98f2a22296a1d9bc29be65aac106`）。
最终批状态：
[batch_status.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_remaining11_v1_20260714/batch_status.json)。

- UE Editor 一次导入并回读 11/11 个模板、每个均含 Walking 和 Idle；耗时
  32.08 秒。共享 cook/package 只执行一次，耗时 168.88 秒。
- GPU 0/2 两路渲染和 6 路 CPU finalize 共完成 22/22 段，wall time 986.18 秒；
  失败和未完成均为 0。每段仍为 18 秒/270帧及主视图、同步 Top-down、带标注
  审核三路媒体。
- 205 个 registry 描述符全部重新计算哈希通过；66 个 MP4 均通过 H.264、时长
  和音轨契约回读。所有 registry 保持 `research_candidate`、
  `human_visual_review=pending`、`formal_registry_promotion=false`。
- Walking 最大身体朝向误差为 1.974°；全部动作最大地面穿透为
  `3.55e-14 cm`（浮点数值零），根部 roll/pitch 均为 0°。所有模板使用
  `Back → Torso3` 身体纵轴，没有以转头角度补偿方向。
- UE 动态高度覆盖约 42 cm 的 Fox/Shiba、76–84 cm 的 Wolf、111–119 cm 的
  Donkey、132–137 cm 的 Deer、145–167 cm 的 Cow/Bull/Alpaca/Horse，以及
  含鹿角约 202–205 cm 的 Stag；与实例配置的类别尺度一致。
- Cattle、Deer/Stag、Horse、Shiba、Wolf 分别使用物种匹配的真实录音并经过
  阈值事件排程；Fox 的 10 秒长录音被判断为单个长事件，不被错误复制。当前
  没有具备来源证据的 Alpaca/Donkey 叫声，因此二者明确静音，绝不以其他动物
  冒充。
- 四阶段 Walking 抽帧：
  [walking_four_phase_contact_sheet.png](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_remaining11_v1_20260714/agent_visual_qa/walking_four_phase_contact_sheet.png)；
  Idle 中帧：
  [idle_mid_contact_sheet.png](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_animal_apartment_specs_remaining11_v1_20260714/agent_visual_qa/idle_mid_contact_sheet.png)。
  代理抽查未见单图重建式缺面、脚部拉丝、横向/倒退或丢材质。

当前通过的是“原生拓扑动画稳定性 + UE 动态方向/落地 + 媒体/音频”门。继承的
桌边轨迹只记录中心线和家具遮挡；尚未发布按每种动物运行时 OBB 计算的正式
mesh-vs-furniture 碰撞门。因此家具遮挡仍是 advisory，不能把 22/22 自动完成
解释成正式场景碰撞批准；这也是维持 `formal_dataset_asset=false` 的原因之一。

## 猫、通用狗与稳定比格补充模板

较早的 Quaternius Animal Pack 还提供原生 Cat/Dog Walk/Idle，均是 CC0、
`yaw=0°` 时面向并沿世界 `+X` 行走，未使用方向推断：

| 模板 | Walk | Idle | 当前状态 |
|---|---|---|---|
| `Cat` | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_curated_native_walk_idle_media_v1_20260714/videos/Cat_walking_yaw0.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_curated_native_walk_idle_media_v1_20260714/videos/Cat_idle_yaw0.mp4) | 原生动作、变形门通过；低多边形研究候选 |
| `Dog` | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_curated_native_walk_idle_media_v1_20260714/videos/Dog_walking_yaw0.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quaternius_curated_native_walk_idle_media_v1_20260714/videos/Dog_idle_yaw0.mp4) | 原生动作、变形门通过；低多边形研究候选 |

稳定比格技术模板是固定 Dog 几何/骨架/动作上的确定性三色材质：

- [GLB](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_beagle_template_v1_20260714/Dog_beagle_tricolor_stable.glb)，SHA-256
  `6eaf7bdd9361b8143abbe5b8261a99b5770b11ac7752bbcc7b391ad23b03dae4`；
- [Walk side](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_beagle_template_v1_20260714/videos/walking_side.mp4)；
- [Walk quarter](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_beagle_template_v1_20260714/videos/walking_quarter.mp4)；
- [Idle side](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_beagle_template_v1_20260714/videos/idle_side.mp4)。

它证明“锁定拓扑后做品种色块/动画”可以稳定执行，但轮廓仍是通用低模 Dog，
所以只能作为技术模板，不能冒充写实比格或正式品种资产。

## Instance 级外观控制证据

Husky 固定几何上已完成三档绝对毛色实例。每个 request 是独立个体，不使用
`from/to/lighter_than_original`：

| 绝对属性 | Walk | Idle | 结果 |
|---|---|---|---|
| `coat_color=light_warm_brown` | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_husky_color_variants_v1_20260714/videos/light_warm_brown_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_husky_color_variants_v1_20260714/videos/light_warm_brown_idle_side.mp4) | 仅 allowlist 毛色材质改变；变形通过 |
| `coat_color=warm_brown` | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_husky_color_variants_v1_20260714/videos/warm_brown_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_husky_color_variants_v1_20260714/videos/warm_brown_idle_side.mp4) | 同上 |
| `coat_color=dark_warm_brown` | [Walk](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_husky_color_variants_v1_20260714/videos/dark_warm_brown_walking_side.mp4) | [Idle](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/stable_husky_color_variants_v1_20260714/videos/dark_warm_brown_idle_side.mp4) | 同上 |

尺寸仍采用 `small/medium/large` 三档，但通过 actor scale 和生成后厘米回读实现，
不为每个尺寸重建网格。体型或轮廓若需要真实变化，则必须使用另一个已审计的
模板；不能靠非等比缩放或 prompt 暗示伪造。

## 物种覆盖与明确缺口

| 类别 | 当前稳定选择 | 不应采用的替代 |
|---|---|---|
| 猫 | Curated `Cat` 原生 Walk/Idle | 不恢复已被用户拒绝的斜跑 Pixal 猫 |
| 通用犬/哈士奇/柴犬/狼/狐 | `Dog`、`Husky`、`ShibaInu`、`Wolf`、`Fox` 原生动作 | 不把同一动作不经审计套给所有四足动物 |
| 牛科 | `Cow`、`Bull` 原生动作 | 不把比格或马骨架伪装为牛 |
| 马科/鹿科 | `Horse`、`Horse_White`、`Donkey`、`Deer`、`Stag` | 不用单图重建替换已稳定模板 |
| 羊驼 | `Alpaca` 原生动作 | 不能自动把 Alpaca 标签改成 llama |
| 鹰、鱼 | Curated `Eagle` 的 Flying/Idle、`Piranha` 的 Swimming/Bite 已通过变形门 | 不能强制要求陆地 Walk |
| 比格、哈巴狗、猪、羊、山羊、花栗鼠等未覆盖物种 | 保持 gap/research；逐个寻找同物种原生模板并重走门禁 | 不能用相似动物换标签，也不能用跨骨架转移动作后直接注册 |

Llama/Pig/Pug/Sheep 的跨模板 Walk 转移试验已保留为失败/备选证据。虽然数值
变形门可通过，视觉步态有夸张和交叉腿，因此不会优先于同物种原生动作。

## Pixal/Hunyuan 对照与几何门禁修正

几何审计 v2 不再把整体 yaw 当成躯干扭曲。它分别记录：

- `global_axis_yaw_degrees`：刚体水平朝向，只作证据，不参与拒绝；
- `centerline_bend_p95_degrees`：去除刚体 PCA 轴后的局部中心线弯曲，才参与门禁；
- position-indexed 非流形边比例：忽略普通 glTF UV/normal seam 副本。

回放证据：
[i23d_geometry_audit.json](/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/quadruped_geometry_audit_v2_calibration_20260714/i23d_geometry_audit.json)，
SHA-256 `d9c11695c78c6ba98a9120aaa117321e1526c4978e62250ce205b6c47c7cac4b`。
Pixal 坏比格 `global_yaw≈17.5°`，但被拒绝的依据是
`bend≈12.85°` 和 `nonmanifold_ratio≈0.002724`，不是 17.5° 朝向本身。

同图 Hunyuan 对照仅用于定位问题，继续保持 `technical_spike_only`；其许可证
结论不会因几何较干净而改变。正式数据不包含 Hunyuan3D 2.0/2.1 或其输出。
