# 生产链统一架构：一个控制器、两个渲染器执行器、其余全部共用（Claude，2026-09-06 晚）

owner 2026-09-06 晚定的原则：引擎有一个最高层控制；每类房间只在"视觉渲染"这一层有自己的执行器，Apartment、酷家乐和以后新做的房间走 UE/SPEAR，MP3D、HM3D 走 Habitat；房间是房间、资产是资产，两边不互相掣肘；路径求解器统一；其余全部共用。这份文档把这个原则落到现有代码上：哪一层已经是共用的、哪一层现在有几份重复、每个接口长什么样、先做什么后做什么。它是合并三份已有实现，不是新框架，删的比加的多。

配套文件：方案修订 `QA_GENERALIZED_SAMPLER_PLAN_V2_20260906.md`（条款与拒出规则），四家族缺口 `QA_FOUR_FAMILY_GAP_LIST_20260906.md`（每家族今天缺的文件）。本文第 7 节的任务表**取代** v2 第 4 节那张按家族切的表。

## 1. 分层与现有代码的对应

| 层 | 职责 | 现有代码 | 现状 | 要做的事 |
|---|---|---|---|---|
| 顶层控制器 | 读请求，选房间包，按房间包的渲染器挑执行器，串起采样、捕获、音频、证据、收口、出题 | `tools/studio/run_qa_episode.py` | 只认"原生 Apartment"和"UE 家具房"两种，别的适配器名直接报错 | 按 `room_package.renderer` 分派；流程本身不变 |
| 房间包 | 一间房的全部静态输入：视觉场景引用、声学包、可行空间、地板参照、语义、坐标系 | HM3D/MP3D 的 `room_manifest.json`（键：scene、navigation、acoustics、semantics、coordinate_system 等）；UE 房间散在 room_catalog 条目、handoff manifest、声学包、路线库、场景配置里 | Habitat 已接近统一形状；UE 四处散放 | 定一份 schema，四个家族各出一份包 |
| 可行空间 | 哪里能站、两点之间怎么走、地板多高 | 家具盒子栅格（`furniture_layout.build_room_navigation`）、Apartment 原生路线库（`native_qa_room`）、酷家乐可行区栅格（qa-v3）、Habitat 寻路器（`mp3d_region_actor_tracks` 用的 PathFinder） | 四个来源，各接一份求解逻辑 | 一个接口、四个适配器；求解逻辑只留一份 |
| 条件采样器 | 画像→实体与声音→路线→静止机位→发声排程→计划 | `qa_episode.py` 的 `sample_activity_routes`、`select_question_camera`、`schedule_audio`；`native_qa_room` 的路线组；qa-v3 的 `design_qa_v3_scene_batch.py`；Habitat 的 `author-current-mp3d-two-beagle-route` | 三份规划器，都不满足新前提 | 只改 `qa_episode.py` 这一份（v2 §2.3），吃可行空间接口，输出渲染器中立的计划 |
| 渲染器执行器 | 加载场景、摆资产、驱动动作、逐帧渲染、读回位姿与掩膜 | UE：链一的 SPEAR 捕获阶段加 `qa_evidence`；Habitat：`tools/capture/capture_mp3d_multi_actor.py` 加 `mp3d_region_actor_tracks.py` 的物化部分 | UE 完整；Habitat 只到语义掩膜 | 两个执行器实现同一接口，各产同一格式的读回与证据 |
| 声学渲染 | 逐帧声源与听者位置加声学包加音频节目→RLR→双耳 stem 与 mixture 与回执 | UE 读回走 `tools/acoustics/render_frame_readback_sequential_speech.py`；Habitat 读回走 `avengine.cli m5 render-current-mp3d-dynamic-audio` | 两个入口，算法同源 | 过渡期两个入口并存但回执同格式；之后合成一个吃中立读回的渲染器 |
| 证据契约 | 像素可见性真值、模态与 target-only 掩膜、外观审阅、遮挡物、音频回执含湿声尾音 | 链一的 `pixel_visibility_truth.json`、`native_pixel_masks_depth_authority_v1.npz`、`appearance_review.json`、`actor_occluders.json`、`research_report.json` | 只有 UE 产；Habitat 缺 | 把链一的格式定为契约；Habitat 执行器按契约产；`qa_evidence.py` 只认契约 |
| 收口与出题 | 把读回、音频、证据整理成 facts，出 24 类，判分，审计 | `qa_delivery.finalize_qa_episode`、`normalize_episode_bundle`、`generate_unified_questions`、`unified_scoring`、审计器 | 出题判分审计已共用；`finalize` 写死 SPEAR 文件名 | `finalize` 改为只认契约，不认渲染器 |

## 2. 五个接口

坐标约定先定死：共用层一律**米、Y 向上、右手**（Habitat 的约定，也是现有声学包和 `habitat_basis_from_xyzw` 的约定）。UE 执行器负责厘米、Z 向上、左手到共用层的转换，与酷家乐场景配置里已声明的 `world_transform: ue_xyz_cm_to_xzy_m_v1` 一致。两边各加一条"方位公式与 `unified_catalog` 逐帧对账差为 0"的测试，UE 这条我的审计器测试里已经有，Habitat 这条要补。

### 2.1 房间包 RoomPackage

一份 JSON，字段：

- `room_id`、`family`（apartment / kujiale / mp3d / hm3d / authored）、`renderer`（ue_spear / habitat）。
- `visual_scene`：UE 填 `map_path`（如 `/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1`）与舞台 `uproject`；Habitat 填 `scene_glb`（非 basis）、`dataset_config`、`navmesh`。
- `acoustic_package`：RLR 包 manifest，schema `avengine_acoustic_scene_package_v1`（A 房与 HM3D 已核为此；酷家乐 `pkg_kujiale_0020_rlr` 待打开确认；MP3D 现用旧 proxy，要换成同 schema 或补一个转换）。
- `walkable_space`：`kind` 取 furniture_grid / walkable_grid / route_bank / habitat_navmesh，加对应路径。
- `floor_reference`：实测地板产物路径，缺了就拒绝加载（常设规矩）。
- `static_geometry`：直达射线用的三角面来源，默认指向声学包的 `triangles.npy`/`vertices.npy`。
- `semantics`：实例与类别标签来源。
- `coordinate_frame`：单位、朝上轴、手性、到共用层的变换名。
- `subrooms`：同一地图内分开统计的区域（酷家乐 livingroom_491、厨房），覆盖表按它分行。

四个家族今天各出一份：Apartment（原生资源与路线库）、A/B/C（handoff manifest 加声学包加地图）、酷家乐（地图、`pkg_kujiale_0020_rlr`、可行区栅格 `qa_v3_walkable_grid_kujiale_20260903_v2`、floor_reference v3）、MP3D 与 HM3D（现有 `room_manifest.json` 加声学包加 floor_reference）。

### 2.2 可行空间 WalkableSpace

方法：`is_navigable(p)`、`shortest_path(a, b)`、`sample_navigable(rng, region)`、`floor_height(p)`、`bounds()`；另有一个可选能力 `route_bank()` 返回预解好的合法路线，给 Apartment 原生路线库这种"不解路、只选路"的情形。四个适配器：家具盒子栅格（现有）、可行区栅格文件（新写，小）、原生路线库（现有）、Habitat PathFinder（现有运行时里就有）。采样器只调用这个接口。

### 2.3 渲染器执行器 RendererExecutor

方法：

- `bind_assets(asset_ids, renderer)`：从登记表 `runtime_backends[renderer]` 取绑定；缺绑定就报"该资产在此渲染器无绑定"，不是"资产不存在"。
- `materialize(plan)`：把中立计划变成本渲染器的驱动数据。UE 是现有的计划到 SPEAR 动作；Habitat 是 `mp3d_region_actor_tracks.py` 里"把路线烘成逐帧关节目标"那一半（它现在规划和物化写在一起，要拆开，规划归共用采样器）。
- `capture(plan, room_package, out_dir)`：返回中立读回加证据（第 2.4、2.5 节）。UE 执行器就是链一现有捕获阶段；Habitat 执行器在 `capture_mp3d_multi_actor` 之上加 target-only 渲染遍与 GLB 刚体摆放。

执行器只做这五件事：加载场景、摆资产与驱动动作、渲染、读回、坐标转换。别的都不碰。

### 2.4 中立读回 NeutralReadback

给共用的声学渲染和 facts 用：`clock`（帧数、帧率、采样率、tick 基）、逐帧 `camera`（位置、基向量）、逐帧每个实体的 `root`、`emitter` 位置和 `moving` 标志，全部米、Y 向上。现有链一的 `frame_readbacks.json` 已经是这个内容但在 UE 坐标里；Habitat 的 `frame_records.json` 加 `emitter_positions_m.npy` 加 `actor_root_readbacks.npy` 也是这个内容。两边各写一个"读回写出器"落到同一格式，声学渲染就只需要读一种。

P1 实现位置：`src/avengine/capture/neutral_readback.py` 只校验共用米制读回；UE 坐标转换与写出器在 `src/avengine/capture/ue_neutral_readback.py`，Habitat 写出器在 `src/avengine/capture/habitat_neutral_readback.py`。输出 `neutral_readback.json` 的 `camera` 是逐帧 `position_m`、`basis.{forward,right,up}`；`entities[slot]` 是逐帧 `root`、`emitter`、`moving`。每条都有 `frame_index`、`pts_ticks`；`producer.source_readbacks` 保留真实输入路径。`moving` 沿用生成器的实际 root 前向差分 > 0.05 m/s 口径，不抄计划动作。时钟严格校验计划已有的六个时间字段。

### 2.5 证据契约 EvidenceContract

直接采用链一现有文件与键名：`pixel_visibility_truth.json`（每帧每实体状态、分辨率、相机位姿 id）、`native_pixel_masks_depth_authority_v1.npz`（`modal` 与 `target_only_<actor>`）、`appearance_review.json`（每实体外观值、审阅帧、状态）、`actor_occluders.json`、音频 `research_report.json`（含每源 stem 路径、湿声尾音区间、峰值）。Habitat 执行器按同名同键产出。`qa_evidence.py` 的外观检查从"上身 HSV 对上衣色"扩成"掩膜内颜色对登记的外观值"，登记值来源为人的 `top_color`、动物的 `coat_profile.value`、设备的 `finish`。

P1 校验器为 `src/avengine/rooms/evidence_contract.py`，仅检查格式与跨文件一致性，不作可答性或人工审核判断。当前链一 NPZ 的真实模态键是 `depth_derived_modal_semantic`；本文件早先写的 `modal` 是简写。旧键继续兼容，双键同存必须逐元素一致。房间包校验器为 `src/avengine/rooms/room_package.py`；旧目录包装保留缺字段报告，不为通过校验补假地板值。新式包严格校验，既有目录的旧请求保留原规划行为；P3 负责补测与提供完整包。可在包的 `planning_inputs`（或保留旧条目的 `legacy_catalog_entry`）中声明当前执行器所需的路径。

## 3. 退役与保留

退役（不再作生产入口，保留代码供历史复现）：qa-v3 的 `tools/qa/run_qa_v3_pipeline.py` 与 `design_qa_v3_scene_batch.py`；HM3D 单声源链 `run_hm3d_end_to_end.py`、`run_hm3d_episode.py` 之于 QA 生产；`author-current-mp3d-two-beagle-route` 的规划部分；两个临时适配器 `kujiale_validate.py`、`adapt_mp3d_habitat_candidate.py`（能力并入收口层）。

保留并升格：`run_qa_episode.py`（控制器）、`qa_episode.py`（唯一采样器）、`capture_mp3d_multi_actor.py`（Habitat 执行器骨架）、两个 RLR 渲染入口（过渡期）、链一证据格式（契约）、`normalize_episode_bundle` 之后的一切。

## 4. 分阶段顺序

**阶段一，契约与分派**：写 RoomPackage、NeutralReadback、EvidenceContract 三份 schema 与校验器；控制器按渲染器分派；四份房间包。这一步不改渲染，能立刻暴露每个家族缺哪个字段。

**阶段二，两个执行器**：UE 执行器从链一抽出（主要是读回写出器换成中立格式、原生 Apartment 的 FOV、沉默者、静止相机透传）；Habitat 执行器补 target-only 掩膜、GLB 刚体摆放、读回写出器，把 `mp3d_region_actor_tracks.py` 拆成规划与物化两半。过渡期两条音频入口并存，但回执必须同格式、都含湿声尾音；单活跃源端点修在 UE 那条入口（第 1370–1377 行）并在 Habitat 那条核对同一语义。

**阶段三，采样器与可行空间**：`qa_episode.py` 按 v2 §2.3 改成条件采样器，接可行空间接口；四个适配器接上；plan-only 矩阵按新口径重跑一次作配额依据。

**阶段四，合并音频与收口**：一个吃中立读回的声学渲染器替掉两条入口；`finalize` 只认契约。

**阶段五，资产双绑定**：见第 6 节，与前四阶段并行；A3 人形包是最长的一条，第一天就开。

四个家族"现在都能出"的最短路径是阶段一加阶段二做完，采样器先用现有 `qa_episode.py` 加 T2a 的三处随机化过渡（机位合法候选内随机、不截前 40、repeat 随机），阶段三再换成完整采样器。

## 5. 坐标与时钟的两条硬规矩

1. 共用层米、Y 向上、右手；每个执行器的读回写出器要带 `coordinate_frame` 声明和一条往返测试（UE 厘米进、米出、再回厘米，差小于 1 毫米）。
2. 时钟只有一份：`clock` 里的帧数、帧率、采样率、tick 基由控制器写进计划，执行器与声学渲染都从计划读，谁也不自己算时长。HM3D 旧链 5 秒视频对 6.65 秒音频就是两边各算的结果。

## 6. 资产双绑定

owner 2026-09-06 晚裁定：**所有声源资产都要尝试进四个家族，不分梯队。**"房间是房间、资产是资产"的落地形式就是同一个 asset_id 在登记表 `runtime_backends` 里有 `spear_unreal` 和 `habitat` 两份绑定。今天 17 条全部只有 `spear_unreal`，所以下面三项都是主线任务，并行开工，人形包最长，最先开：

1. **A1 刚体 GLB**：外部索引 44 个静态资产加 2 个运行时音箱，每个都有 `geometry/finalized_glb`；给每条加 `runtime_backends.habitat`（资产种类、GLB 路径、语义模板、放置姿态）。Habitat 侧用现有对象模板加载。
2. **A2 beagle**：M2 包 `/data/avengine_external/datasets/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5` 已经能在 Habitat 跑，只是绑定写在包清单里而不在登记表；登记成 `runtime_backends.habitat`。
3. **A3 人形与生成动物**：走 beagle 那条 M2 包路线（蒙皮 GLB、URDF、关节映射、烘好的 Idle 与 Walking 关节目标、接触锚点、发声点锚点），工具都在 `tools/assets/`（`compile_animal_package.py`、`bake_actions.py`、`build_joint_mapping.py`、`probe_habitat_skin_rest.py`、`rebase_skin_root.py`、`publish_animal_assets.py`），契约在 `src/avengine/assets/contracts.py`。已知要改的一处：契约第 37 到 40 行的接触锚点写死为四只爪子，人形要改成按体型声明的接触集合。Rocketbox 男女成人与 7 个 FLUX 生成动物都要出包。设备播人声是额外的声源组合，不是人形的替代。

**最终目标的定义**：每个房间家族 × 每类声源资产（人、动物、设备）至少有一条经过验证的原生路径；覆盖表里不再有因"该渲染器无绑定"而记 `interface_not_implemented` 的格子；剩下的空格只能是题义不适用或证据与采样缺口。首批的完成按这个全矩阵定，接口没打开之前已能出的格子先跑，但不算首批完成。

规则不变：一个资产在某个渲染器暂时没有绑定，覆盖表记 `interface_not_implemented` 并写明缺的是哪个渲染器的绑定、卡在哪一步，不写"不适用"，也不从分母里去掉。

## 7. 任务表 v2.1（取代 v2 第 4 节）

四条线：S 共用层、U 是 UE 执行器、H 是 Habitat 执行器、A 是资产双绑定；C 是我名下的。责任按文件切，同一文件的任务串行。

| ID | 任务 | 文件 | 责任 | 验收产物 |
|---|---|---|---|---|
| S0 | 三份契约 schema 与校验器：RoomPackage、NeutralReadback、EvidenceContract | 新建 `src/avengine/rooms/room_package.py`、`src/avengine/capture/neutral_readback.py`、`src/avengine/rooms/evidence_contract.py`（名字由 Codex 定） | Codex | 四份房间包全部通过校验；链一现有读回与证据文件通过校验 |
| S1 | 控制器按渲染器分派 | `tools/studio/run_qa_episode.py` | Codex | 同一请求换 room_id 能分别进 UE 与 Habitat 执行器 |
| S2 | 条件采样器（v2 §2.3）接可行空间接口 | `src/avengine/rooms/qa_episode.py` | Codex | 对 A 房与 MP3D 房间各出一份计划；plan-only 矩阵重跑 |
| S2a | 过渡：现有规划器三处随机化 | 同上 | Codex | 10 个 seed 机位分布，无前 40 截断 |
| S3 | 可行空间接口与四个适配器 | `src/avengine/rooms/furniture_layout.py`（家具栅格、可行区栅格）、`native_qa_room.py`（路线库）、Habitat 运行时（PathFinder） | Codex | 酷家乐一条路线来自可行区栅格；MP3D 一条路线来自 PathFinder |
| S4 | 声学渲染：过渡期两入口同回执，之后合一 | `tools/acoustics/render_frame_readback_sequential_speech.py`、`src/avengine/cli.py` 的 m5 动词 | Codex | 两条入口对同一段中立读回输出同格式回执，含湿声尾音 |
| S4a | 单活跃源端点按全部实体建 | `render_frame_readback_sequential_speech.py` 第 1370–1377 行 | Codex | 探针重跑出音频，沉默者 stem 全零 |
| S5 | 语音 prepared 集与按声类活动画像 | `src/avengine/assets/sound_prepare.py` | Codex | 307 条候选集清单、10 条人工试听 |
| S6 | 生成器候选/emit、ID、coverage、24 类修订、判分两 bug | `src/avengine/qa/unified_catalog.py`、`unified_scoring.py` | Codex | 24 类各一题样例；同 seed 一致 |
| S7 | 证据函数只认契约，外观扩到毛色与表面材质 | `src/avengine/rooms/qa_evidence.py` | Codex | beagle 与音箱各一条 reviewed |
| S8 | 收口只认契约 | `src/avengine/rooms/qa_delivery.py` | Codex | UE 与 Habitat 各一段走同一 `finalize` 出 facts 与 questions |
| S9 | 批清单与五态覆盖表 | 位置由 Codex 定；口径见 v2 §2.10 | Codex 实现，Claude 口径 | 首批覆盖表 |
| S10 | 评测前排列 | `tools/dataset/run_qwen_content_controls.py` | Codex | 排列清单 |
| U1 | UE 读回写出器改中立格式 | 链一捕获阶段 | Codex | A 房一段读回通过 NeutralReadback 校验 |
| U2 | 原生 Apartment 的 FOV、沉默者、静止相机透传 | `native_qa_room.py`、`run_qa_episode.py` | Codex | 一份 85° 静止计划 |
| U3 | UE 侧刚体与动物摆放 | `qa_episode.source_declaration`，复用 `tools/qa/qa_v3_actor_selection.py` | Codex | 一段含音箱、一段含动物的原生读回 |
| U4 | 房间包：Apartment、A/B/C、酷家乐 | 新房间包文件；酷家乐需 pose_bindings（内容待 Codex 说明） | Codex | 三份包过校验；酷家乐地图在 multi-home 舞台渲一帧 |
| H1 | Habitat 读回写出器改中立格式 | `tools/capture/capture_mp3d_multi_actor.py` | Codex | MP3D 一段读回通过校验 |
| H2 | target-only 渲染遍、掩膜 npz、像素真值 | 同上加派生工具 | Codex | `derive_actor_occluders` 能读 |
| H3 | GLB 刚体摆放与 semantic id | 同上、`mp3d_region_actor_tracks.py` | Codex | 一个音箱出现在 `semantic.npy` |
| H4 | 规划与物化拆分 | `src/avengine/assets/mp3d_region_actor_tracks.py` | Codex | 共用计划能物化成 case_manifest |
| H5 | 房间包：MP3D、HM3D，含地板参照与 Z-up 核对 | 新房间包文件 | Codex | 两份包过校验；HM3D 多角色捕获干跑 5 帧 |
| A1 | 44 加 2 个刚体的 Habitat 绑定登记 | `examples/runtime/source_asset_runtime_profiles.json`、外部索引 | Codex | 登记通过 `bind_assets` |
| A2 | beagle M2 包登记为 Habitat 绑定 | 同上 | Codex | 同上 |
| A3 | Rocketbox 人形与 7 个生成动物的 Habitat M2 式包并登记绑定 | `tools/assets/{compile_animal_package.py, bake_actions.py, build_joint_mapping.py, probe_habitat_skin_rest.py, rebase_skin_root.py, publish_animal_assets.py}`、`src/avengine/assets/{contracts.py, package.py, actions.py}`、登记表 | Codex | 每个资产一份过校验的包与 `runtime_backends.habitat`；一段含人形与生成动物的 Habitat 原生捕获 |
| C1 | 审计器七项修复与测试 | `tools/qa/audit_binding_feasibility.py`、`tests/test_audit_binding_feasibility.py` | Claude | 五段重审字段报告 |
| C2 | Habitat 方位公式对账测试 | `tests/test_audit_binding_feasibility.py` | Claude | 逐帧差为 0 |
| C3 | 人工校准包 | `~/Documents/Claude`、`docs/qa` | Claude | 校准包 v1 |
| C4 | 覆盖表统计口径与首批构成规则 | 文档 | Claude | v2 §2.10、§7 |

## 8. 现在谁做什么

Codex：从 S0、S1、U1、H1 开始（契约与两个读回写出器），这四项不动渲染，一天内能暴露四个家族各缺哪个字段；A3 人形与生成动物的 Habitat 包同一天并行开工；然后 U2、S4a、S2a、U4、H2、H3、H5、A1、A2。
Claude：C1 立刻开始（我的文件），C2 等 H1 有一段中立读回就写，C3、C4 并行。
首个干跑目标：同一份请求、两个 room_id（A 房与 MP3D），都能走完控制器到 `finalize`，各出一份 facts 与 questions，即便 Habitat 那份大部分题还是 `interface_not_implemented`。

状态：本文与配套两份都是草稿，未提交；源码未改；未启动生产。
