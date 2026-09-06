# 四个房间家族今天出片还缺什么（Claude，2026-09-06 晚）

owner 的两条新前提：四个家族（Apartment、酷家乐、HM3D、MP3D）现在都要能出片；每段最低只要两个声源资产同时在场、24 类题能出。这份清单回答"缺什么"，按家族列到文件和字段。下面每一条都是我今晚在 `48g-jump:/data/jzy/tmp/wt-multi-home-activity-integration`（HEAD 5ade25f）读源码、读产物核出来的；标了"待干跑"的是我没有实际运行、只能靠一次小运行才能确认的。

## 0. 先说结论

现在仓库里有三条互不相通的生产链，24 类生成器（`normalize_episode_bundle` 加 `generate_unified_questions`）挂在它们末端，中间靠临时适配器：

- **链一**（multi-home）：`tools/studio/run_qa_episode.py` 到 `qa_episode.build_qa_episode_plan` 或 `native_qa_room`，UE/SPEAR 捕获，`render_frame_readback_sequential_speech.py` 出双耳，`qa_delivery.finalize_qa_episode` 收口。证据最全：逐帧读回、像素可见性真值、target-only 掩膜、外观审阅、遮挡物、湿声尾音。房间只有自建 A/B/C 和原生 Apartment。
- **链二**（qa-v3）：`tools/qa/run_qa_v3_pipeline.py` 读 `examples/qa/scenes/*.json`，`backend` 为 `ue_spear`（Apartment、酷家乐）或 `habitat_native`（MP3D）。酷家乐 17/24 和 MP3D 5/24 的验证媒体都来自这条链，靠 `kujiale_validate.py` 和 `adapt_mp3d_habitat_candidate.py` 两个临时脚本喂进 24 类生成器。
- **链三**（HM3D 端到端）：`tools/studio/run_hm3d_end_to_end.py` 到 `run_hm3d_episode.py`，单个移动声源、FOA 加双耳、SoundSpaces 音频解释器。它不是两声源产品，视频 5 秒对音频 6.65 秒的时钟也不齐，0/24。

所以"四个家族都能出"今天卡在三处：酷家乐没有挂进链一（房间目录里只有 A/B/C）；Habitat 两个家族缺像素证据、缺可区分的两个声源资产、缺产品化的收口；HM3D 连两声源的捕获都没跑过。24 类生成器本身不用为家族改动，它只认统一后的 facts。

最低要求改成"两个声源同时在场"后，三、四人那摊工作（原任务 T3）降为可选；两人在四间 UE 房间的几何可行率是 Codex 矩阵里最高的一档（30–60° 箱 72%–94%）。

## 1. 现状地图

| 家族 | 今天在哪条链 | 已验证的产出 | 缺口性质 |
|---|---|---|---|
| Apartment | 链一（原生适配器） | 双人 18/24，全套证据 | 小修：FOV、沉默者、单活跃源、静止相机默认 |
| 自建 A/B/C（对照） | 链一 | 4 段 75/96 | 同上 |
| 酷家乐 0020 livingroom_491 | 链二 | 双人 17/24，像素证据 113/150 帧，Astra 审过外观 | 没挂进链一；链二产物缺湿声尾音读回、遮挡物、37 帧像素 |
| MP3D 17DRP5sb8fy | 链二（habitat_native） | 双 beagle 5/24，150 帧 RGB/深度/语义、双耳 16 kHz、逐帧角色与发声点读回 | 缺 target-only 像素真值、外观标签、可区分声源、产品化收口 |
| HM3D 00800 TEEsavR23oF | 链三 | 单声源诊断片 0/24 | 没有两声源路径；要把 MP3D 的 habitat_native 路径搬到 HM3D 房间上 |

## 2. 逐家族缺口

### 2.1 Apartment（链一，原生适配器）

能出。要修的都在 v2 任务表里，这里只列文件：

1. `tools/studio/run_qa_episode.py` 第 61–86 行没把 `camera_fov_deg` 传给 `native_qa_room.build_native_apartment_qa_plan`，后者写死 105°；默认 `camera_motion` 是 follow_group，要改成静止。
2. `native_qa_room.py` 不透传 `silent_actor_count`；`render_frame_readback_sequential_speech.py` 第 1370–1377 行按事件建端点，一个事件就报 `at least two source endpoints`。修法是端点按全部实体建、沉默者零干声（第 666 行 `one_active_of_n` 已有）。
3. 声源类型：`qa_episode.source_declaration` 拒绝 rigid_object，音箱进不来；8 个动物资产声明上能过（有 spear_unreal 后端和 idle/walk），但这条链没原生跑过动物，要一段读回证明。

### 2.2 酷家乐（推荐挂进链一）

**两条路线。**路线甲：给链一的房间目录加一个酷家乐条目，之后规划器、音频、收口全部复用，新采样器的改动只做一遍。路线乙：留在链二，把 `kujiale_validate.py` 产品化并补证据；代价是新采样器要在 qa-v3 的设计工具里再实现一遍，Codex 也反对复制四套生成器。我推荐甲。

路线甲缺的东西，对照 A 房条目的六个字段（`room_id`、`backend`、`manifest`、`map_path`、`acoustic_package`、`pose_bindings`）：

1. **`map_path`**：multi-home 的 UE 舞台 `/data/avengine_external/workspaces/multi_home_activity_20260905/ue_stage/SpearSim/Content/AVEngine/Optional/Kujiale/` 里有 `kujiale_0020_full_home_v1.umap`，可以直接用 `/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1`。qa-v3 用的 BakedLit 那张（`/Game/AVEngine/Optional/KujialeBakedLit/...`）不在这个舞台里，所以光照会和 17/24 那段不同。待干跑：用这条链的 SpearSim 打开这张图渲一帧。
2. **`acoustic_package`**：`/data/avengine_external/review/kujiale_acoustics/pkg_kujiale_0020_rlr/manifest.json`，就是 qa-v3 渲酷家乐音频用的那份。schema 应与 A 房的 `avengine_acoustic_scene_package_v1` 相同（A 房和 HM3D 的我核过是这个，酷家乐这份待打开确认）。
3. **`manifest`（布局）**：这是唯一要写代码的地方。A 房的 manifest 是 `room_a_final_handoff.json`，`envelope.bounds_xy_m` 加 `artifacts.{visual_glb, objects, functional_anchors, usd}`，链一的导航和相机候选都从家具对象的包围盒算（`furniture_layout.py` 第 717 行附近的注释就是这个意思）。酷家乐没有家具对象清单，但 qa-v3 已经有它的可行区栅格 `/data/jzy/tmp/qa_v3_walkable_grid_kujiale_20260903_v2`（5 厘米格，来自路线库的可行区）和净空表。所以要加一个"从可行区栅格建 PathFinder 和相机候选"的适配，让 `build_qa_episode_plan` 在 manifest 声明 `navigation_source: walkable_grid` 时走这条路，而不是家具盒子。静态三角面用声学包里的 `triangles.npy`/`vertices.npy` 给直达射线判定（Codex 的审核脚本对 A/B/C/N 就是这么做的，`_load_static_triangle_geometry` 现在读 `resources.visual_geometry.resolved` 的 GLB，要多认一种来源）。
4. **`pose_bindings`**：A 房有 `room_a_pose_bindings.json`。酷家乐要生成一份；这个文件的内容和生成工具我没有追到，需要 Codex 说明。
5. **`floor_height_m`**：0（2026-09-03 实测 0.02 厘米，地板网格没有碰撞体，已写在场景配置的 `floor_reference`）。按常设规矩这一项直接引用现有 floor_reference，不手写。
6. 场景内不同房间（客厅 livingroom_491、厨房）要在覆盖表里分开算；`route_domain` 字段已经区分了。

### 2.3 MP3D（链二的 habitat_native 路径，要产品化）

链二已经把 MP3D 跑通到媒体：设计（`design_qa_v3_scene_batch.py` 出 `actor_selection.json`、`audio_program.json`、`timeline.json`）→ `mp3d_region_actor_tracks.py` 出 `case_manifest.json` 和逐帧轨迹 → `tools/capture/capture_mp3d_multi_actor.py` 出 `rgb.npy`、`depth.npy`、`semantic.npy`、`frame_records.json`、`actor_root_readbacks.npy`、`emitter_positions_m.npy` → `avengine.cli m5 render-current-mp3d-dynamic-audio` 出双耳 stem 与 mixture。缺的是下面六件：

1. **可区分的两个声源资产。**这是 Habitat 家族最硬的缺口。17 个运行时资产（8 动物、7 人、2 音箱）全部只有 `spear_unreal` 后端；Habitat 里唯一能站进去的关节角色是 M2 包的那只 beagle（`rocketbox_dog_beagle_01_m2_v7_world_contact_candidate`，靠 `asset_manifest_path` 单独加载）。MP3D 那段就是两只一样的 beagle，所以 QA-01/02/03/09/10/11/12/14/18/19/20/21 全部因 `appearance_review_missing` 拒出。最快的补法：外部索引 `/data/avengine_external/assets/sound_source_assets_v1/index.json` 里 44 个静态资产每个都有 `geometry/finalized_glb`，Habitat 可以把 GLB 当刚体对象加载并给 semantic id。这样 Habitat 里的两声源组合可以是"beagle 加设备"或"两个不同外观的设备"。人形角色在 Habitat 里还没有资产（Rocketbox 是 UE 骨骼网格）。owner 已裁定所有资产都要进：人形和 7 个生成动物按 beagle 的 M2 包路线进 Habitat（架构文档任务 A3，提示词 P12），设备播人声只是额外的声源组合，不是替代。
2. **像素可见性真值。**现在只有 `semantic.npy` 的模态掩膜（角色 semantic id 210/211），没有 target-only 掩膜，所以 `pixel_visibility_truth.json` 和 `native_pixel_masks_depth_authority_v1.npz` 都没有，QA-07/08/24 因此拒出，`derive_actor_occluders` 也没法用。补法：每帧对每个角色单独渲一遍语义（其他角色隐藏）得到 target-only 掩膜，按链一的 npz 键名（`modal`、`target_only_<actor>`）和 `pixel_visibility_truth.json` schema 写出，链一的 `derive_actor_occluders` 就能直接复用。
3. **外观审阅。**`qa_evidence.build_pixel_appearance_review` 只认 `top_color`/`coat_value` 和上身 HSV。beagle 在登记里有 `coat_profile.value = standard_tricolor`，设备有 `finish`（black_ash、walnut_veneer）；要把这两类值接成可核的外观值，并把颜色检查从"上身"改成"掩膜内"。
4. **收口产品化。**`tmp/qa_family_validation_20260906/adapt_mp3d_habitat_candidate.py` 已经把捕获目录加音频目录拼成了 `normalize_episode_bundle` 要的 raw bundle，并从 float32 stem 读出湿声尾音。要把它变成 `qa_delivery` 的 Habitat 版（建议 `src/avengine/rooms/qa_delivery_habitat.py`），输入捕获目录、音频目录、上面第 2、3 条的派生文件。
5. **规划器。**`author-current-mp3d-two-beagle-route` 和 qa-v3 设计工具都是旧规划器（固定 beagle、按分数选相机）。新采样器的几何层要做到后端无关：只吃 PathFinder（Habitat navmesh 本来就有）、静态三角面（声学包）、相机候选栅格、发声点高度，输出路线、静止机位、事件排程；然后由 `mp3d_region_actor_tracks.py` 物化成 case_manifest。Codex 的审核脚本已经对 N（原生路线）和 A/B/C（栅格路线）做过这种抽象，可行。
6. **入口与地板。**`run_qa_episode.py` 只分派 native Apartment 和 UE furnished，要加 `habitat_native` 分派；MP3D 房间先量地板写 floor_reference；坐标基（米制、Y 向上）的方位公式要和 `unified_catalog` 对账（我的审计器对 UE 做过逐帧差 0 的对账，Habitat 要补同样一条测试）。

### 2.4 HM3D（把 MP3D 的 habitat_native 路径搬过来）

链三不能改造成两声源产品，直接放弃这条路做 QA，改用 MP3D 那条 habitat_native 路径跑 HM3D 房间。已经有的：

- 房间清单 `/data/avengine_external/review/hm3d_e2e_wt_20260905_v1/rooms/hm3d_val_00800_TEEsavR23oF/room_manifest.json`，键集合与 `examples/rooms/habitat_mp3d_example/room_manifest.json` 完全一致（scene、navigation、acoustics、semantics 等）。
- 声学包 `.../hm3d_e2e_wt_20260905_v1/package/manifest.json`，schema `avengine_acoustic_scene_package_v1`，与 A 房相同。
- 路线库 `routes_R3`，房间 R3（23.16 平方米，有沙发）。
- Habitat 视觉运行时加载 HM3D 已在 2026-08-26 实测可用。

缺的在 MP3D 六件之上再加三件：

1. **待干跑**：用这份房间清单喂 `capture_mp3d_multi_actor` 捕 5 帧，确认场景路径（要用非 basis 的 `.glb`）、dataset config（train/val 选择）和 navmesh 都能解析。工程完成文档说 HM3D train/val dataset-config 选择已并入，但多角色捕获没在 HM3D 上跑过。
2. 语义 glb 直接读坐标的地方要做 Z-up 到 Y-up 的转换（2026-08-28 帧 bug 的教训），房间清单和 floor_reference 都要按此核一遍。
3. HM3D 地板参照还没量。

### 2.5 所有家族共用的三处

1. `run_qa_episode.py` 加 `habitat_native` 分派、房间目录加酷家乐条目。
2. `qa_delivery.finalize_qa_episode` 的 SPEAR 专用检查（`research_receipt.native_pixel`、`frame_readbacks.json`、npz 掩膜）保留，Habitat 走自己的收口，两者产出同一份 raw bundle 结构。
3. 24 类生成器、判分、审计器不按家族分叉；覆盖表按 §2.10 五态记，Habitat 家族今天的格子大多是 `interface_not_implemented`。

## 3. "两个声源同时在场"在各家族今天能落地的组合

| 家族 | 今天就能 | 修完 T9a（刚体声明）后 | 修完 Habitat 刚体加载后 | 修完 A3（人形与生成动物的 Habitat 包）后 |
|---|---|---|---|---|
| Apartment、A/B/C | 人加人 | 人加音箱、动物加音箱、人加动物（动物需一段原生读回证明） | 不涉及 | 无 |
| 酷家乐 | 链二：人加人 | 挂进链一后与上行相同 | 不涉及 | 无 |
| MP3D | beagle 加 beagle（同貌，只能出不靠外观的题） | 不涉及 | beagle 加设备、设备加设备（设备可播人声） | 人加人、人加动物、人加设备、其他动物的全部组合 |
| HM3D | 无 | 不涉及 | 同 MP3D | 同 MP3D |

## 4. 需要 owner 拍板的两件

1. 酷家乐走路线甲，挂进链一的房间目录（要写一个可行区栅格到 PathFinder 的适配和一份 pose_bindings）。
2. 已由 owner 裁定：所有声源资产都进四个家族，不分梯队；人形与生成动物进 Habitat 是主线任务 A3，Habitat 家族的首批完成定义包含它们；设备播人声只是额外组合。

## 5. 在 v2 任务表上增补的任务

说明：这些任务已并入 `QA_PRODUCTION_ARCHITECTURE_20260906.md` 第 7 节的统一任务表：T10a 到 T10c 归 U4 与 S3，T11a 到 T11e 归 H1 到 H4、A1 与 S2，T11f 归 C2，T12a 到 T12c 归 H5。下表保留作对照。

| ID | 任务 | 文件 | 责任 | 验收产物 |
|---|---|---|---|---|
| T10a | 酷家乐房间条目 | `qa_real_rooms_20260906_inputs/room_catalog.json` 新条目；manifest 新建 | Codex | 条目六字段齐全，`run_qa_episode.py --room_id` 能选中 |
| T10b | 可行区栅格到 PathFinder 与相机候选的适配 | `src/avengine/rooms/furniture_layout.py`、`qa_episode.py` | Codex | 酷家乐一条路线、一组机位候选，与 qa-v3 净空表抽样对照 |
| T10c | 酷家乐地图在 multi-home 舞台渲一帧 | 无代码 | Codex | 一帧 RGB 与深度 |
| T11a | Habitat 刚体 GLB 加载与 semantic id | `src/avengine/capture/mp3d_multi_actor.py`、`mp3d_region_actor_tracks.py` | Codex | 一个音箱在 MP3D 房间里出现在 `semantic.npy` |
| T11b | target-only 掩膜与像素真值 | 同上，新增派生工具 | Codex | `pixel_visibility_truth.json` 加 npz，`derive_actor_occluders` 能读 |
| T11c | 外观审阅泛化（coat、finish） | `src/avengine/rooms/qa_evidence.py` | Codex | beagle 与音箱各一条 reviewed 记录 |
| T11d | Habitat 收口产品化 | 新建 `src/avengine/rooms/qa_delivery_habitat.py`（由 `adapt_mp3d_habitat_candidate.py` 迁入） | Codex | MP3D 一段 facts.json 加 questions.json，路径全部指向真实读回 |
| T11e | 采样器几何层后端无关 | `qa_episode.py` | Codex | 同一采样器对 A 房和 MP3D 房间各出一份计划 |
| T11f | Habitat 方位公式对账测试 | `tests/test_audit_binding_feasibility.py` 加一条 | Claude | 逐帧差为 0 |
| T12a | HM3D 房间清单干跑多角色捕获 5 帧 | 无代码或小改 | Codex | 5 帧 RGB/深度/语义 |
| T12b | HM3D 地板参照与 Z-up 核对 | floor_reference 工具 | Codex | floor_reference 文件 |
| T12c | HM3D 一段两声源 Episode 走 T11d 收口 | 同 T11 | Codex | facts.json 加 questions.json |

## 6. 开跑前的干跑顺序

1. 酷家乐地图在 multi-home 舞台渲一帧（T10c）。
2. 可行区栅格适配出一条路线、一组机位（T10b）。
3. HM3D 房间清单喂多角色捕获 5 帧（T12a）。
4. 一个 GLB 刚体在 Habitat 加载并读回 semantic id（T11a）。
5. 一帧 target-only 渲染（T11b）。

这五步任何一步失败，对应家族当天记 `interface_not_implemented` 并写明卡在哪一步，不从分母里拿掉。

## 7. 本文引用的路径

- 家族验证：`docs/qa/QA_FAMILY_VALIDATION_20260906.md`；`tmp/qa_family_validation_20260906/{mp3d_validation_summary_v1.json, hm3d_validation_summary_v1.json, kujiale_reviewed_v2/, adapt_mp3d_habitat_candidate.py}`。
- 链一：`tools/studio/run_qa_episode.py`、`src/avengine/rooms/{qa_episode.py, native_qa_room.py, qa_delivery.py, qa_evidence.py, furniture_layout.py}`；房间目录 `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_real_rooms_20260906_inputs/room_catalog.json`。
- 链二：`tools/qa/{run_qa_v3_pipeline.py, design_qa_v3_scene_batch.py}`、`examples/qa/scenes/interioragent_kujiale_0020_livingroom_491.json`、`src/avengine/assets/mp3d_region_actor_tracks.py`、`tools/capture/capture_mp3d_multi_actor.py`、`src/avengine/cli.py` 的 `render-current-mp3d-dynamic-audio`；MP3D 产物 `/data/datasets/avengine_workspaces/qa_v3_engine_completion_20260904/mp3d_pipeline_e60b9a3_20260905_v1/`。
- 链三与 HM3D 资源：`tools/studio/{run_hm3d_end_to_end.py, run_hm3d_episode.py}`；`/data/avengine_external/review/hm3d_e2e_wt_20260905_v1/{rooms, package, routes_R3, episode}`。
- 资产：`examples/runtime/source_asset_runtime_profiles.json`（17 条，全部仅 spear_unreal）；`/data/avengine_external/assets/sound_source_assets_v1/index.json`（44 条，每条 `geometry/finalized_glb`）。
- UE 舞台：`/data/avengine_external/workspaces/multi_home_activity_20260905/ue_stage/SpearSim/Content/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1.umap`。
- 酷家乐声学包：`/data/avengine_external/review/kujiale_acoustics/pkg_kujiale_0020_rlr/manifest.json`。
