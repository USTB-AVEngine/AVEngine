# 给 Codex 的逐任务提示词（Claude，2026-09-06 晚）

这份文件里的每一段提示词都可以整段复制给 Codex。开头的"总前提"每次开工前先给一遍，然后按顺序给 P1 到 P12（P12 是人形与生成动物进 Habitat 的资产包任务，与 P1、P4 同一天并行开工）。任务编号 S、U、H、A 与 `docs/roadmap/QA_PRODUCTION_ARCHITECTURE_20260906.md` 第 7 节的任务表一致；条款细节在 `docs/roadmap/QA_GENERALIZED_SAMPLER_PLAN_V2_20260906.md`，各家族的真实路径在 `docs/roadmap/QA_FOUR_FAMILY_GAP_LIST_20260906.md`。这三份文档已提交在分支上（a7856e3），审计器 v2 在 02df947。

---

## 总前提（每次开工前先读）

你在服务器 `48g-jump` 的仓库工作树 `/data/jzy/tmp/wt-multi-home-activity-integration` 里工作，分支 `codex/multi-home-activity-integration`，当前最新提交是 ea50b7c（本文件在 d090318，审计器 v2 在 02df947）。项目 Python 是 `/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`，跑测试和工具时带 `PYTHONPATH=src`。你直接在这个工作树里改源码、跑测试、提交；这个树是和 Claude 共用的，所以每完成一个任务就提交，不要长期留着未提交的改动。

先读这四份文档，再动手：`docs/roadmap/QA_PRODUCTION_ARCHITECTURE_20260906.md`（分层、五个接口、任务表）、`docs/roadmap/QA_GENERALIZED_SAMPLER_PLAN_V2_20260906.md`（条款与拒出规则）、`docs/roadmap/QA_FOUR_FAMILY_GAP_LIST_20260906.md`（各家族缺口与路径）、以及你自己写的 `docs/roadmap/QA_GENERALIZED_SAMPLER_PLAN_REVIEW_CODEX_20260906.md`（行号引用）。

下面这些是 owner 定下的硬规矩，不要在实现里绕开：

1. 本阶段相机只做静止。这只约束带新策略开关（`sampling_policy: conditioned_static_v2`）的请求；没有这个字段的旧请求走旧代码路径，行为完全不变，不要静默改机位。
2. 一切可选择的东西，包括目标、事件、查询时刻、机位、说话顺序、发声时刻、声音片段，都在合法候选里均匀随机，不取第一个，也不取打分最高的。
3. 声音与角色的性别一致只在两边字段都有性别含义时执行：人用 `realized_attributes.sex_or_gender_label`，语音用 VCTK 元数据的 gender；动物按物种配声音类别；设备没有性别；任何一边未知就不配对，不伪填。
4. 语音片段不超过 5 秒，可听语音不少于 1.5 秒，入池时做语音频带二次裁剪；动物叫声、设备持续声、短促提示音各用自己的活动检测画像，不套人声的 80 Hz 高通和 1.5 秒下限。
5. 输出音频是双耳两声道，不转伪 FOA。
6. 复用现有实现与资产。不新增通用框架、hash 锁或冻结 contract；共用纯函数的版本靠同仓 Git 提交和产物里的 producer 字段。
7. 共用层坐标一律米、Y 向上、右手（Habitat 约定）。UE 执行器负责厘米、Z 向上、左手到共用层的转换，与酷家乐场景配置里已声明的 `world_transform: ue_xyz_cm_to_xzy_m_v1` 一致。
8. 新房间第一步是在引擎里量地板偏移写 `floor_reference`；手写 `ground_z` 不算。
9. 最低要求是每段两个声源资产同时在场、24 类题都能出。不能靠删题型、删声源类型、删房间家族来提高表面成功率。某类出不来就修对应的出题链或补对应的条件画像，分母保留。
10. 三种缺口分开记，不许混：题义不适用（`not_applicable_by_definition`）、接口未实现（`interface_not_implemented`）、证据缺失或采样没找到（`evidence_missing_or_unsampled`）。
11. 文件归属：`tools/qa/audit_binding_feasibility.py` 和 `tests/test_audit_binding_feasibility.py` 是 Claude 的，你不改；共用纯函数的接口先按架构文档第 2.7 节对齐，再实现。
12. 提交前跑与改动相关的单测；碰了 `tools/` 下的脚本就重生成 `docs/TOOL_INDEX.md`（`PYTHONPATH=src python tools/build_tool_index.py`）并跑 `tests/unit/test_tool_index_current.py`。提交信息用仓库现有风格，例如 `feat(qa): ...`、`fix(audio): ...`、`docs(qa): ...`，正文写清改了什么、怎么验的。
13. 不 push、不合并 main、不切换正在运行的 Studio 服务、不启动数据集生产。原生 UE 或 Habitat 的单段验证可以跑，批量生产要等 owner 明确说开。
14. 每个任务做完按文末"报告格式"回一份简短报告。没做完的如实写没做完，不要把测试变绿说成验收通过。
15. 所有声源资产（人、动物、设备）都要尝试进两个渲染器，不分梯队。某个资产暂时进不了某个渲染器，报告里写清卡在哪一步（导出、蒙皮、URDF、烘焙、加载、读回），覆盖表记"接口未实现"，不写"不适用"，也不从分母里去掉。

---

## P1 契约与分派（S0、S1、U1、H1）

目标：定下三份共用契约，让顶层控制器按房间包里的渲染器分派，让 UE 和 Habitat 两条捕获链各写出同一格式的中立读回。这一步不动渲染，做完就能看出四个家族各缺哪个字段。

你拥有的文件：新建 `src/avengine/rooms/room_package.py`、`src/avengine/capture/neutral_readback.py`、`src/avengine/rooms/evidence_contract.py`（文件名你可以改，但要在架构文档里同步）；改 `tools/studio/run_qa_episode.py`；改链一捕获阶段写读回的那段代码；改 `tools/capture/capture_mp3d_multi_actor.py` 或在它旁边加一个读回写出器。对应的测试放 `tests/unit/`。

输入与输出：
- 房间包 RoomPackage 的字段见架构文档第 2.1 节：`room_id`、`family`、`renderer`（ue_spear 或 habitat）、`visual_scene`、`acoustic_package`、`walkable_space`、`floor_reference`、`static_geometry`、`semantics`、`coordinate_frame`、`subrooms`。校验器要拒绝缺 `floor_reference` 的包。
- 中立读回 NeutralReadback 见第 2.4 节：`clock`（帧数、帧率、采样率、tick 基）、逐帧 `camera`（位置、基向量）、逐帧每个实体的 `root`、`emitter`、`moving`，全部米、Y 向上，并带 `coordinate_frame` 声明。链一现有的 `capture/frame_readbacks.json` 内容已经齐，只是在 UE 坐标里；Habitat 侧对应 `frame_records.json`、`actor_root_readbacks.npy`、`emitter_positions_m.npy`（MP3D 现成产物在 `/data/datasets/avengine_workspaces/qa_v3_engine_completion_20260904/mp3d_pipeline_e60b9a3_20260905_v1/pairs/mp3d_17DRP5sb8fy_runtime/mp3d_runtime_observation/capture/mp3d_beagle_150/`）。
- 证据契约 EvidenceContract 直接采用链一现有文件与键名：`pixel_visibility_truth.json`、`native_pixel_masks_depth_authority_v1.npz`（键 `modal` 与 `target_only_<actor>`）、`appearance_review.json`、`actor_occluders.json`、音频 `research_report.json`。校验器只做格式与一致性检查，不做语义判断。
- 控制器：`tools/studio/run_qa_episode.py` 第 61 到 86 行现在只认 `native_room_adapter == "avengine_native_spear_apartment_qa_room_v1"` 和 UE 家具房两种，其他适配器名直接抛 `unsupported native room adapter`。改成按房间包的 `renderer` 分派到两个执行器入口；旧的 room_catalog 条目要能自动包装成房间包，保证 A/B/C 和原生 Apartment 现有请求结果不变。

约束：坐标转换只在执行器里做，共用层不出现厘米；读回写出器要带一条往返测试（UE 厘米进、米出、再回厘米，差小于 1 毫米）；时钟只从计划读，执行器不自己算时长。

单测：三份契约各有"合法样例通过、缺字段拒绝"的测试；链一现有一段的 `frame_readbacks.json` 转成中立读回后通过校验；MP3D 现成产物转成中立读回后通过校验；控制器对同一请求换 room_id 能分派到不同执行器。

验收产物：四个家族各一份房间包草稿（缺什么字段就在校验报告里列出来，不要补假值）；A 房一段与 MP3D 一段的中立读回文件；控制器分派日志。

不要做：不要在这一步改渲染、改采样器、改音频；不要为了让校验通过而放宽契约。

完成后报告：契约字段清单与校验器路径；四个家族各缺哪些字段；两份中立读回的路径与往返测试结果。

---

## P2 原生 Apartment 透传、单活跃源端点、现有规划器三处随机化（U2、S4a、S2a）

目标：把原生 Apartment 这条链上三处已经确认的错误修掉，让"两个声源、其中一个沉默"的段能真正渲出音频，让现有规划器不再固定选第一个候选。这是今天能开跑的最小前提。

你拥有的文件：`src/avengine/rooms/native_qa_room.py`、`tools/studio/run_qa_episode.py`（原生分支的参数透传）、`tools/acoustics/render_frame_readback_sequential_speech.py`、`src/avengine/rooms/qa_episode.py` 的 `select_question_camera` 与 `schedule_audio`。测试在 `tests/unit/test_native_qa_room.py`、`tests/test_question_driven_rooms.py`，音频部分的测试放 `tests/unit/` 新文件。

三处错误与修法：
1. 原生 Apartment 不透传视场和沉默者。`run_qa_episode.py` 第 68 到 77 行调用 `build_native_apartment_qa_plan` 时没传 `camera_fov_deg`，`native_qa_room.py` 第 712 到 721 行写死 105°；Codex 审核实测三次请求 85° 都得到 105°。同一处默认 `camera_motion` 是 follow_group。改成透传视场并记录实际采用值，新策略请求默认 static；`silent_actor_count` 也要透传（自建房间 `qa_episode.py` 第 701 行已支持，原生没有）。
2. 单活跃声源报错。`render_frame_readback_sequential_speech.py` 第 1370 到 1377 行的端点表按事件建，一个事件就只有一个端点，第 1377 行抛 `dynamic multi-source audio requires at least two source endpoints`；而第 666 行已有 `one_active_of_n` 模式，条件是候选端点不少于 2。修法：端点从计划的全部实体建，无事件的实体是真实候选端点、干声为全零；不加假发声事件；第 1377 行的保护保留；双槽缓存格式不动。复现脚本是 `tmp/qa_generalized_sampler_review_20260906_v1/probe_single_active_source.py`，产物 `single_active_source_probe_v1/result.json`。
3. 现有规划器按分数选。`qa_episode.py` 第 417 到 419 行先排序、第 436 行只看前 40 个、第 489 到 490 行取最高分；`native_qa_room.py` 第 507 到 510 行按分数选路线对；`schedule_audio` 第 570 行 repeat 模式选最短声音。改成在满足条件的合法候选里均匀随机，随机数从请求 seed 派生，同 seed 可复现。

约束：不带 `sampling_policy` 的旧请求行为完全不变（加一条回归测试锁住）；随机化不改变合法性判断本身。

单测：原生计划的实际视场等于请求值；沉默者透传后计划里有 N 个实体、N 减 1 个事件；单事件加两个真实端点的音频计划通过、旧的两事件计划不变、三源奇数分片不变；不同 seed 机位不同且全部在合法集合内、没有前 40 截断；repeat 选到的片段在合法重复候选内。

验收产物：重跑探针，`audio_output_created` 为 true，沉默者的 stem 全零，混合轨峰值与两人同段可比（写在报告里）；一份 85° 静止的原生 Apartment 计划文件；10 个 seed 的机位分布表。

不要做：不要把这三处随机化写成完整的联合采样器，那是 P5 的事；不要改缓存格式。

完成后报告：三处改动的文件与行号、测试计数、探针结果路径、10 个 seed 的机位分布。

---

## P3 四份房间包（U4、H5）

目标：给 Apartment、自建 A/B/C、酷家乐、MP3D、HM3D 各写一份房间包，让控制器能选中它们。酷家乐和 HM3D 是今天没有的两份，重点在它们。

你拥有的文件：房间包文件放在 `examples/rooms/packages/`（或你在 P1 定的位置）；酷家乐还要新建一份 `pose_bindings`；`tools/rooms/` 下需要的小工具。

现成材料：
- 酷家乐：UE 地图 `/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1`，它在 multi-home 舞台 `/data/avengine_external/workspaces/multi_home_activity_20260905/ue_stage/SpearSim/Content/AVEngine/Optional/Kujiale/` 里；qa-v3 用的 BakedLit 那张不在这个舞台。RLR 声学包 `/data/avengine_external/review/kujiale_acoustics/pkg_kujiale_0020_rlr/manifest.json`（打开确认 schema 是 `avengine_acoustic_scene_package_v1`，坐标是 meter 加 +Y）。可行区栅格 `/data/jzy/tmp/qa_v3_walkable_grid_kujiale_20260903_v2`（5 厘米格，来自路线库可行区）。地板参照 `/data/jzy/tmp/qa_v3_floor_reference_kujiale_20260903_v3`（实测 0.02 厘米，地板网格没有碰撞体）。场景配置 `examples/qa/scenes/interioragent_kujiale_0020_livingroom_491.json` 里有 `route_domain`、`camera_clearance_table`。参照 A 房条目：房间目录 `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_real_rooms_20260906_inputs/room_catalog.json`，A 的 manifest `.../integration_sol_v6/final_plan_inputs_v1/room_a_final_handoff.json`（`envelope.bounds_xy_m` 加 `artifacts` 的 visual_glb、objects、functional_anchors、lighting、usd），pose_bindings `.../integration_sol_v6/plan_inputs_v1/room_a_pose_bindings.json`。酷家乐没有家具对象清单，所以它的 `walkable_space.kind` 是 walkable_grid，`static_geometry` 指向声学包的 `triangles.npy` 和 `vertices.npy`。pose_bindings 是什么内容、怎么为酷家乐生成，请你先在报告里说明再定。
- MP3D：房间清单 `examples/rooms/habitat_mp3d_example/room_manifest.json`（scene、navigation、acoustics、semantics、coordinate_system 等键），声学包现用 `/data/avengine_external/review/m3_current_mp3d_semantic_20260820T1630Z/manifest.json`，schema 要核对，不是 `avengine_acoustic_scene_package_v1` 就补一个转换或改指同 schema 的包。地板参照还没量。
- HM3D：房间清单 `/data/avengine_external/review/hm3d_e2e_wt_20260905_v1/rooms/hm3d_val_00800_TEEsavR23oF/room_manifest.json`（键集合与 MP3D 示例完全一致），声学包 `.../hm3d_e2e_wt_20260905_v1/package/manifest.json`（已核为 `avengine_acoustic_scene_package_v1`），场景目录 `/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/val/00800-TEEsavR23oF`。地板参照还没量。语义 glb 直接读坐标时要做 Z 向上到 Y 向上的转换，这是 2026-08-28 帧 bug 的教训。
- Apartment 与 A/B/C：从现有 room_catalog 条目和原生资源包装。

约束：每份包必须带实测的 `floor_reference`，MP3D 和 HM3D 先量地板再写包；坐标系字段照实填，不要把 UE 的厘米写成米；酷家乐的地图先在 multi-home 舞台里渲一帧确认能加载，渲不出来就在报告里写"待验证"，不要写成可用。

单测：四份包（加 A/B/C）全部通过 P1 的校验器；酷家乐包被控制器选中时进入 UE 执行器；HM3D 包进入 Habitat 执行器。

验收产物：房间包文件；酷家乐一帧 RGB 与深度；MP3D 与 HM3D 的 floor_reference 文件；一份"每个包缺什么"的表。

不要做：不要用 MP3D 的 UE 导入结果代替 Habitat 房间；不要用 HM3D 的旧单声源链产物冒充房间包证据。

完成后报告：五份包的路径与校验结果；酷家乐地图加载结果；pose_bindings 的含义与生成方式；两个 Habitat 房间的地板实测值。

---

## P4 Habitat 像素证据与刚体资产（H2、H3、A1、A2）

目标：让 Habitat 执行器按证据契约产出像素可见性真值和 target-only 掩膜，让静态设备资产和 beagle 都能在 Habitat 里被绑定和摆放。这一步做完，MP3D 和 HM3D 才有"两个能区分的声源"。

你拥有的文件：`src/avengine/capture/mp3d_multi_actor.py`、`tools/capture/capture_mp3d_multi_actor.py`、`src/avengine/assets/mp3d_region_actor_tracks.py`（物化那一半）、`examples/runtime/source_asset_runtime_profiles.json`、外部索引 `/data/avengine_external/assets/sound_source_assets_v1/index.json`、新增的派生工具。

现状：
- `capture_mp3d_multi_actor` 现在只捕 rgb、depth、semantic 三种模态（`mp3d_multi_actor.py` 第 574 到 575 行写死），角色各有唯一 semantic id（第 241 到 245 行校验），所以模态掩膜有、target-only 掩膜没有，`pixel_visibility_truth.json` 也没有。
- 链一的 `qa_evidence.derive_actor_occluders`（`src/avengine/rooms/qa_evidence.py` 第 11 到 70 行）读 `native_pixel_masks_depth_authority_v1.npz` 的 `modal` 与 `target_only_<actor>`，形状都是 [帧, 高, 宽]，遮挡像素定义为 `target_only > 0` 且 `modal != 自身 id`。Habitat 侧要按这个格式产出。
- 运行时登记 17 条全部只有 `spear_unreal` 后端；Habitat 里唯一能站进去的关节角色是 M2 包的 beagle `rocketbox_dog_beagle_01_m2_v7_world_contact_candidate`，它的 Habitat 绑定写在包清单里、靠 `asset_manifest_path` 加载（`mp3d_region_actor_tracks.py` 第 540 到 546 行），不在登记表里。
- 外部索引 44 个静态资产每条都有 `geometry/finalized_glb`。

要做的事：
1. H2：每帧对每个实体单独渲一遍语义（其他实体隐藏）得到 target-only 掩膜，与模态掩膜一起写成 `native_pixel_masks_depth_authority_v1.npz`，再派生 `pixel_visibility_truth.json`（状态、分辨率、相机位姿 id，与链一同 schema）。运行时激活要点：runtime prefix `/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z`，magnum site `/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages`，mp3d_root `/data/datasets/habitat_data`，必须传 `rlr_sdk_root=/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg`；不要喂 `*.basis.glb`，会段错误。
2. H3：把 GLB 静态资产当 Habitat 刚体加载并分配 semantic id，摆放用资产登记里的 resting_pose，先放地上。
3. A1：给 44 加 2 个刚体资产在登记表加 `runtime_backends.habitat` 绑定（字段你定，至少要有资产种类、GLB 路径、语义模板、放置姿态），能被 `bind_assets` 解析。
4. A2：把 beagle 的 M2 包登记为 `runtime_backends.habitat` 绑定，不再靠外部参数传包路径。

约束：静态资产不伪造走路动画；一个资产在某个渲染器没有绑定就报"该渲染器无绑定"，不是"资产不存在"；不新造缓存或掩膜格式。

单测：npz 键名与形状符合 `derive_actor_occluders` 的读法（可直接调用它跑通）；两个实体互相遮挡的合成场景里遮挡像素数大于 0；刚体加载后 `semantic.npy` 里出现它的 id；登记表校验通过、旧的 spear_unreal 绑定不受影响。

验收产物：MP3D 房间里 beagle 加一个音箱的一段捕获（可以只捕 30 帧），含 rgb、depth、模态与 target-only 掩膜、像素真值文件；登记表 diff。

不要做：不要用语义掩膜的"像素大于零"冒充 target-only。人形与生成动物的 Habitat 包在 P12 里做，和本任务并行，本任务的验收不等它。

完成后报告：捕获产物路径；掩膜格式核对结果；登记表新增绑定的数量；哪些资产的 GLB 加载失败及原因。

---

## P5 可行空间接口与条件采样器（S3、S2、H4）

目标：把四个来源的可行空间收到一个接口后面，把 `qa_episode.py` 改成方案 v2 第 2.3 节的条件采样器，输出渲染器中立的计划；Habitat 侧把规划和物化拆开，共用计划能物化成 case_manifest。

你拥有的文件：`src/avengine/rooms/qa_episode.py`、`src/avengine/rooms/furniture_layout.py`、`src/avengine/rooms/native_qa_room.py`（路线库适配）、`src/avengine/assets/mp3d_region_actor_tracks.py`（拆分）、新建的可行空间模块。测试 `tests/test_question_driven_rooms.py`、`tests/unit/test_native_qa_room.py`、Habitat 侧对应单测。

可行空间接口（架构文档第 2.2 节）：`is_navigable(p)`、`shortest_path(a, b)`、`sample_navigable(rng, region)`、`floor_height(p)`、`bounds()`，可选 `route_bank()`。四个适配器：家具盒子栅格（现有 `build_room_navigation`）、可行区栅格文件（新写，酷家乐用）、原生路线库（现有）、Habitat PathFinder（现有运行时）。

采样器要点（细节以 v2 第 2.2 与 2.3 节为准）：
1. 请求带 `sampling_policy: conditioned_static_v2` 才走新采样器；旧请求不变。
2. 先抽定条件画像（总数 N、发声数 S、声源类别、锚点数 `anchor_count`、角距箱、可见性、直达、运动关系、事件关系），**抽定后固定**；后面任何一步失败都只在画像内部换种子重抽，预算 200 次；耗尽就写失败记录，不换画像补数。
3. 角距条件只对 `anchor_count` 个锚点事件要求落箱（默认 1），其余事件只要合法；竞争者集合含画外实体；可见性与直达是画像取值不是全局布尔。
4. 路线按已选讲话角色和所需窗口在合法运动模式里随机，不再用固定的 hold_walk_hold、walk_hold_walk、stand 模式表（现在 `qa_episode.py` 第 285 与 304 行）；刚体不生成路线。
5. 静止机位在现有 0.55 米网格、相对地面 1.55 米、85° 视场的候选里逐帧算躯干视野、发声点直达、最近竞争者夹角，不打分、不截断，合法候选里均匀随机；为每个实体保留合法窗口集合。
6. 发声时刻：枚举事件顺序（最多 24 种），在保留可行后缀的合法起点里随机；sequential、overlap、repeat 三种关系分别约束；时间用整数 sample 或 tick 和半开区间。
7. 声音片段按剩余时间预算过滤后再随机（Claude 的实验：三人 15 到 30° 箱把片长上限设 2.5 秒后成功数从 2/1/10/2 回到 8/4/24/20，30 到 60° 箱仍被几何上限压在 4/4/7/2；数据在 `tmp/qa_sampler_plan_v2_claude_20260906/`）。
8. 计划里 `condition_profile` 与 `planned_conditions` 分开写；`achieved_conditions` 这个名字只留给渲染后从读回重算的值。

H4：`mp3d_region_actor_tracks.py` 现在把"规划路线"和"把路线烘成逐帧关节目标"写在一起（第 963 行写 case_manifest）。拆成两半，规划归共用采样器，物化留在 Habitat 执行器，输入是共用计划。

单测：画像固定重试（失败后画像不变）；同 seed 计划 JSON 逐字节一致，不同 seed 机位与时刻不同；机位全部在合法候选内；三种事件关系各自的约束成立；预算耗尽有失败直方图；可行区栅格适配器给出的路线全部在栅格内；共用计划能物化成 case_manifest 并通过 `capture_mp3d_multi_actor` 的输入校验。

验收产物：同一采样器对 A 房和 MP3D 房间各出一份计划；用 Codex 审核脚本的口径重跑一次 plan-only 矩阵（新增 anchor_count 等于 1 和片长过滤两列），作为配额依据。

不要做：不要为了提高成功率悄悄放宽角距下限；不要把原生 Apartment 换成栅格路线。

完成后报告：接口与适配器文件；采样器测试计数；两份计划路径；plan-only 矩阵摘要（每格成功数与失败直方图）。

---

## P6 声学渲染：两入口同回执，再合一（S4）

目标：过渡期让 UE 读回的 `render_frame_readback_sequential_speech.py` 和 Habitat 读回的 `avengine.cli m5 render-current-mp3d-dynamic-audio` 输出同一格式的回执，之后合成一个吃中立读回的渲染器。

你拥有的文件：`tools/acoustics/render_frame_readback_sequential_speech.py`、`src/avengine/cli.py` 里 m5 动词及其实现模块、`src/avengine/rooms/qa_delivery.py` 里拼音频命令的 `build_audio_command`。

回执要求（以链一现有 `research_report.json` 为模板）：混合轨路径、每源 stem 路径、每事件的湿声尾音区间、峰值 dBFS、实际增益及只应用一次的证明、HRTF 标识、绕射布尔与 `max_diffraction_order`、时钟、输入读回路径。绕射布尔和阶数现在在 `render_frame_readback_sequential_speech.py` 第 342 与 351 行写死，要一起参数化；`_normalize_plan_events` 第 564 到 569 行优先用 actor 的 voice binding，同一 actor 说不同片段时要按事件绑定。Habitat 那个动词的 `--beagle-audio` 是给 `dog_beagle_v2_scheduled_dry` 的历史覆盖参数，其他声音走 `--sound-asset-registry`；别把它扩展成新的按名字写死。

约束：增益只在事件绑定处乘一次；输出双耳两声道；双槽缓存格式不动；本底噪声是单独实验，默认不加。

单测：两条入口对同一段中立读回输出的回执通过同一个校验器；开关与阶数在回执里可读回；增益为 1.0 与 0.15 的输出逐样本比值等于 6.667（对应旧混音乘 1 除 0.18 那类验收判据的写法）。

验收产物：A 房一段和 MP3D 一段的回执文件；合一后的渲染器对同一读回与过渡期入口输出逐样本差为 0 的对照。

不要做：不要在渲染内部做归一化；不要用加本底噪声来"修"精确零比例。

完成后报告：回执字段清单；两段回执路径；合一进度。

---

## P7 语音 prepared 集与按声类活动画像（S5）

目标：把语音库从 7 句扩成派生 prepared 集，做语音频带二次裁剪和 80 Hz 高通，带偏移、参数、性别、转写；给非语音声源各自的活动检测画像。

你拥有的文件：`src/avengine/assets/sound_prepare.py`、`tools/assets` 下的准备入口、`tests/unit/test_assets_sound_prepare.py`。

事实：事件注册表 `/data/avengine_external/assets/sound_event_library_v1_20260903/sound_asset_registry_v1.json` 有 1374 条，speech 613 条，其中 600 条 VCTK，speech 条目没有顶层 gender 和 transcript，要桥接到原声音库 `/data/avengine_external/assets/sound_library_v1/speech_playback/<id>/clip.json` 里的元数据。VCTK 男 300 女 300，各 12 位说话人。Codex 的临时检测（80 Hz 高通加 300 到 3400 Hz 带通，20 ms 窗 10 ms 步，相对峰值 −25 dB）给出 426 条跨度合格、307 条累计活动合格（男 159 女 148），逐条结果在 `tmp/qa_generalized_sampler_review_20260906_v1/speech_band_measurement.json`，可以当种子但不是人工认证。原 7 句裁掉前置低频段后只有 1/7 达 1.5 秒，不能继续用。非语音：Laughter 137、Cough 25、Sneeze 只有 5 个独立源文件，不能靠重复切分凑 10。

要做的事：
1. 新建派生 prepared 集：新 ID、原路径、裁剪样本区间、滤波参数、检测规则、转写来源、性别；原 PCM 不动；旧 Episode 的 offset 和时钟不改；不要截掉语句后还保留"完整句子"声明。
2. 按声音类别配活动检测画像：语音那套照上面的参数；动物叫声全带能量、不高通，最短可听时长按类别声明（占位 0.5 秒）；设备持续声全带能量相对本底，不设 1.5 秒下限，改设覆盖查询窗；短促提示音允许重复触发按事件计数。阈值全部标占位。
3. 非语音每类如实报独立源文件数。

约束：片长不超过 5 秒、可听语音不少于 1.5 秒；性别字段只从原元数据来，没有就写 unknown。

单测：前置低频段被裁掉、短音被拒、内部停顿不被切成多段、句尾辅音保留、超过 5 秒被拒、性别桥接正确、重复运行不覆盖已有集。

验收产物：prepared 集清单（条数按性别分列）、重测报告（每条的可听跨度与累计活动时长）、10 条人工试听记录（谁听的、听到什么、有没有裁掉辅音）。

不要做：不要把检测器候选数写成"审核通过素材数"；不要给设备声套人声画像。

完成后报告：集路径与条数；检测参数；试听结果；非语音各类独立源文件数。

---

## P8 生成器与判分（S6）

目标：把 24 类生成器改成"枚举合法候选再随机抽"，修 ID、coverage、查询策略、拒出规则和判分器的两个 bug；按方案 v2 第 2.4 节和 Codex 审核表 5.3 逐类修题面与答案域。

你拥有的文件：`src/avengine/qa/unified_catalog.py`、`src/avengine/qa/unified_scoring.py`、`tests/unit/test_qa_unified_catalog.py`、`tests/unit/test_qa_unified_scoring.py`。

要做的事：
1. 每个 `_generate_qa_xx` 拆成 `_candidates_qa_xx(facts)` 与 `_emit_qa_xx(facts, candidate, seed)`；先生成轻量候选再抽样再构造题目与证据。题义定死的类型（QA-03 首声、QA-22/23 全片计数、QA-24 首位发声者片尾状态）候选集合就是一个，不随机替换成别的事件。候选不足就少出并在 coverage 里记 `insufficient_candidates`，不复制同题。
2. 题目 ID（第 2554 到 2556 行的 slug）加入目标、锚点事件、查询窗；coverage（第 4397 行）按 qa_id 存列表；报告分题型覆盖、有效题数、未满足配额三列。
3. `qa_sampling` 要被执行：查询函数（第 2252 到 2322 行）接受具体合法窗，`uniform_in_legal_window` 在生成器内解析成帧；第 2263 行的 or 逻辑把 frame=0 丢掉了，要修；显式非法帧报错不 clamp。
4. 拒出规则：`distractors_equal_gold` 按形式比较，缺失值不算不同；**删掉"金标是多数就拒出"**，改记 `candidate_value_multiplicity`、`gold_is_majority`、`gold_is_unique_minority`，批级做结构基线检查；`binding_not_feasible` 改名 `binding_geometry_candidate_failed`，只用规划几何做候选筛选；`query_inside_inaudible_segment` 拆成 `query_outside_source_activity` 与 `query_inside_wet_tail`；`quota_exhausted` 保留。
5. 逐类改动至少包括：QA-13 的 MCQ 答案域改为视野内三带（占位 [−40.44°, −13.5°)、[−13.5°, 13.5°)、[13.5°, 40.44°]，边界边距占位 5°），目标在查询时刻出画的记 `target_unobservable_at_query` 并 deferred，Open 数值形式保留；QA-22 选项数等于合法域大小（出现 2 人时是 3）；QA-17 题面写查询终点；QA-10 删自身后只剩一个选项则 MCQ deferred；QA-08 题面明确问起点时刻；QA-16 英文补比较基准；QA-18 题面固定问"正在发声的是谁"并避开尾音边界 0.3 秒（占位）；QA-12 分别报台词归属匹配与词错误率。
6. 判分器：`unified_scoring.py` 第 308 到 309 行把"−30 degrees, on the left"判 invalid，负数加方向词一致时应接受；第 104 行中文"并不远"被单字"远"别名匹配成 farther，否定词要先处理。
7. 私有字段（画像、实际角距、目标身份、分歧、难度画像）不进第 2651 到 2666 行的 model_input 白名单。

单测：多候选不重复、候选不足、ID 唯一、frame=0 合法、显式窗生效、Open 与 MCQ 分歧分别算、负方向词、中文否定、真答案与错答案各一例、QA-13 三带、QA-22 合法 K、QA-17 终点、QA-10 单选项 deferred。

验收产物：对现有五段 facts（`tmp/qa_real_rooms_20260906/*/delivery_final_*/facts.json`）各出一份新 questions.json，24 类各至少一题样例；同 seed 两次逐字节一致。

不要做：不要为了字母均匀造不可能选项；不要在看到模型结果后改选项。

完成后报告：改动的函数清单；测试计数；五段新 questions 的题型覆盖与 deferred 原因表。

---

## P9 证据与收口只认契约（S7、S8）

目标：让 `qa_evidence.py` 的外观、遮挡、可见性函数只认证据契约，不认渲染器；让 `qa_delivery.finalize_qa_episode` 对 UE 与 Habitat 走同一条收口。

你拥有的文件：`src/avengine/rooms/qa_evidence.py`、`src/avengine/rooms/qa_delivery.py`、对应单测。可参考 `tmp/qa_family_validation_20260906/adapt_mp3d_habitat_candidate.py`（临时脚本，把 Habitat 捕获与音频拼成 raw bundle），把它的能力并进来后退役它。

现状：`qa_delivery.py` 第 55 到 111 行写死检查 `capture/research_receipt.json` 的 `native_pixel` 状态、`frame_readbacks.json` 的帧数、`native_pixel_masks_depth_authority_v1.npz`，遮挡物登记的标签写成"person in the X top"；`qa_evidence.build_pixel_appearance_review`（第 125 行起）只读 `top_color` 或 `coat_value`（第 148 行）并做上身 HSV。

要做的事：
1. 外观检查从"上身 HSV 对上衣色"扩成"掩膜内颜色对登记的外观值"，登记值来源：人的 `top_color`、动物的 `coat_profile.value`（如 beagle 的 standard_tricolor）、设备的 `finish`（如 black_ash、walnut_veneer）；显示标签按实体类别生成，不把设备写成"穿某色衣服的人"。
2. `finalize` 改为读契约文件而不认渲染器；两个执行器产出的 raw bundle 结构相同；`normalize_episode_bundle` 不动。
3. 家具遮挡若纳入，只复用真实现有对象的实例身份，不新增家具模型；先不阻塞主线。

单测：同一份契约样例经 UE 命名与 Habitat 命名两种来源进 `finalize` 得到相同 facts；beagle 与音箱各一条外观 reviewed；目标自身从候选里排除；稀疏帧保留显式已知帧、缺失帧 deferred。

验收产物：MP3D 一段（P4 的捕获加 P6 的音频）走 `finalize` 出 facts.json 与 questions.json，路径全部指向真实读回；A 房一段重跑结果与现有 delivery 一致。

不要做：不要把计划值抄成 actual；不要用资产标签代替像素核查。

完成后报告：契约读取点清单；两段收口产物路径；外观检查在两种实体上的结果。

---

## P10 批清单、覆盖表、评测前排列（S9、S10）

目标：批层预分配条件画像并记录请求与实际达成；覆盖表按 QA 类型乘声源类别乘房间家族三轴、五态记；评测前用循环置换生成选项排列并记 permutation_id。

你拥有的文件：批清单与覆盖表工具的位置你定（建议 `tools/dataset/`）；`tools/dataset/run_qwen_content_controls.py`。统计口径由 Claude 定，见 v2 第 2.10 节和第 7 节。

要做的事：
1. 批清单：在 Episode 运行前把画像、颜色与声音身份的反平衡预分配好，每个 Episode 独立运行，不设运行时共享计数器；保留未达到的配额和失败记录。
2. 覆盖表：主轴 QA 类型乘声源类别乘房间家族，保留 room_id、asset_id、声音来源，家族内不同场景分开行；每格状态只能是 produced、deferred_by_rule、not_applicable_by_definition、interface_not_implemented、evidence_missing_or_unsampled 之一，分母不丢。
3. 评测前排列：按题型和选项数 K 用循环置换生成排列，记 permutation_id、语义 question_id、模型实际收到的题面与选项映射；排列不算新增题；先报告位置分布与一致性。
4. 训练评测切分按 Episode、房间、路线、声音身份分组，同一视觉 Episode 的音频变体不跨集。

单测：五态互斥；分母不因 deferred 减少；六选项解析；循环置换对三、五选项都会移动金标；旧原始回答的映射保留；公开输入不含 truth 与 profile。

验收产物：对现有五段生成的首批覆盖表；一份排列清单样例。

不要做：不要对 2 到 6 个选项统一要求 25% 的字母分布；不要把配额未满的题用容易样本替补后称达标。

完成后报告：工具路径；覆盖表样例；排列清单样例。

---

## P11 UE 侧刚体与动物摆放（U3）

目标：让静态设备（音箱等）和动物能作为声源进入 UE 这条链，满足"两个声源同时在场"里的人加设备、人加动物、动物加设备组合。

你拥有的文件：`src/avengine/rooms/qa_episode.py` 的 `source_declaration` 及相关规划代码，复用 `tools/qa/qa_v3_actor_selection.py` 第 35 到 62 行对 rigid_object 的 static_mesh_binding 解析，以及现有静态源导入与放置姿态工具。

现状：`source_declaration` 在 `entity_class == "rigid_object"` 时直接抛 `has no articulated SPEAR runtime`，之后还要求 skeletal mesh 与 idle/walk 动画。8 个动物资产声明上能过，但这条链没原生跑过动物。

要做的事：
1. 刚体走 static_mesh_binding 路径，不要求 idle/walk；放置用登记的 resting_pose，第一批只放地上；发声点用登记的 emitter 绑定。
2. 动物按现有关节路径声明，发声点高度、体型、机位 pitch 从资产声明取并在一段原生读回里核实，不沿用人的参数当已验证值。
3. 题义适用矩阵（架构文档配套的 v2 第 2.8 节）在生成层生效：设备作为目标时运动类题记 `not_applicable_by_definition`，设备作为参照或竞争者仍参与。

约束：不为静态对象伪造走路动画；性别规则对动物按物种、对设备不适用。

单测：刚体声明通过且不要求动画；关节路径不变；一个"人加音箱"的计划通过 2 到 4 个资产的校验；设备作为目标的运动题被记为题义不适用而不是拒出。

验收产物：A 房或 Apartment 一段"人加音箱"原生读回和一段"人加动物"原生读回，各含像素证据与双耳音频。

不要做：不要为了让设备进来放宽像素或音频的证据要求。

完成后报告：声明路径改动；两段读回路径；发声点高度与像素证据核对结果。

---

## P12 人形与生成动物的 Habitat 资产包（A3）

目标：让 Rocketbox 的人形角色和 FLUX 生成的 7 个动物能像 beagle 一样在 Habitat 里被加载、驱动走停动作、读回发声点，并登记为 `runtime_backends.habitat` 绑定。owner 的裁定是所有声源资产都要进四个家族，不分梯队，所以这一项是主线任务，和 P1、P4 同一天并行开工，它是最长的一条。

你拥有的文件：`tools/assets/` 下的 M2 包工具（`compile_animal_package.py`、`bake_actions.py`、`build_joint_mapping.py`、`probe_habitat_skin_rest.py`、`rebase_skin_root.py`、`publish_animal_assets.py`、`blender_retarget_quaternius_to_generated_quadruped.py`）、`src/avengine/assets/contracts.py`、`src/avengine/assets/package.py`、`src/avengine/assets/actions.py`、`src/avengine/assets/habitat.py`、`src/avengine/assets/kinematics.py`、登记表 `examples/runtime/source_asset_runtime_profiles.json`。

模板：beagle 的包在 `/data/avengine_external/datasets/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/`，里面有 `visual.glb`（带蒙皮）、`collision_proxy.glb`、`skeleton.json`、`skinning_manifest.json`、`habitat/`（URDF 与关节映射）、`actions/`（烘好的 Idle 与 Walking 关节目标）、`contacts/`、`emitter_anchors.json`、`admission/`、`qa/`、`provenance_manifest.json`；对应请求 `/data/avengine_external/datasets/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json`。`compile_animal_package.py` 现在钉死的是 beagle 的 Rocketbox 源文件哈希，要泛化成按请求指定源；`src/avengine/assets/contracts.py` 第 37 到 40 行的接触锚点写死为四只爪子（paw_front_left 等），这是四足假设，人形要改成按体型声明的接触集合（双足），改法是把接触锚点作为包声明的一部分而不是常量。

要做的事：
1. 先用 `probe_habitat_skin_rest.py` 一类的探针确认这个 Habitat 运行时能加载带蒙皮的关节人形（运行时前缀、magnum site、mp3d_root、rlr_sdk_root 见 P4），不能就把卡点写清。
2. 人形：从 Rocketbox 男女成人的 FBX 出蒙皮 GLB、URDF、关节映射，烘 Idle 与 Walking，发声点锚在嘴部关节，接触集合改双足，编成通过 `validate_animal_asset_package`（或它的泛化版）的包。上衣色变体（burgundy、green、blue、yellow）要和 UE 侧同一条登记记录对应：同一个 asset_id，两个渲染器各一份绑定，这就是"资产是资产"。
3. 生成动物：7 个 FLUX 动物已经有 UE 骨骼网格与 Quaternius 重定向动作，沿用同一条烘焙路线出 Habitat 包。
4. 每个资产在登记表加 `runtime_backends.habitat`，`bind_assets` 能解析。

约束：不为静态设备造动画；性别字段照登记不伪填；包的 provenance 写清源文件与哈希；进不了 Habitat 的资产不从覆盖表分母里去掉。

单测：接触集合改成按体型声明后，beagle 包仍通过校验；一个人形包通过校验；一个生成动物包通过校验；登记表校验通过且旧的 spear_unreal 绑定不变。

验收产物：MP3D 或 HM3D 一段含一个人形和一个生成动物的原生捕获（含 rgb、depth、模态与 target-only 掩膜、发声点读回），每个新包的校验报告。哪个资产进不了，报告写明卡在哪一步（导出、蒙皮、URDF、烘焙、加载、读回）。

不要做：不要用 UE 导入的 MP3D 场景替代 Habitat；不要用静态摆放的人形冒充有走停动作的人形。

完成后报告：成功进 Habitat 的资产清单与包路径；失败资产及卡点；接触集合契约的改动；一段捕获的路径。

---

## 报告格式（每个任务完成后）

1. 改了哪些文件（路径），提交号。
2. 跑了哪些测试，各自的通过、失败、跳过计数；有失败就贴错误原文。
3. 验收产物的路径，以及你亲自看过或听过的核对结果。
4. 没做完的部分和原因，分清是题义不适用、接口未实现还是证据缺失。
5. 对 Claude 的接口有什么要求（字段名、函数签名、数据位置）。
6. 任何与总前提冲突、需要 owner 拍板的地方，单独列出，不要自行决定。
