# 放量前修复任务书（交给 Grok，2026-09-07，Claude 起草，owner 已裁定）

## 启动提示词（owner 一次性交给 Grok 的话）

请读 `docs/roadmap/GROK_FIX_TASKS_20260907.md`，它是这次全部工作的任务书；审核依据在同目录 `AUDIT_P1_P12_PILOT46_20260907.md`，Codex 的实现报告在 `docs/roadmap/codex_reports_20260906/`。按任务书第 2 节的分工开几个子代理并行做 G-A 到 G-D，你自己做 G-E 的重跑与放量就绪报告。每项做完按文末报告格式写到 `docs/roadmap/grok_reports_20260907/<G>_<名>.md` 并提交。目标只有一个：全部做完并通过第 4 节的就绪门之后，owner 就能直接开大规模生产。不许改阈值凑通过、不许删资产/题型/家族、不许代填人工试听、不许改 Claude 名下的审计器及其测试、不许改 Codex 报告原文（要更正就在自己的报告里写）。

## 0. 环境与边界

- 服务器 `48g-jump`；从 `codex/multi-home-activity-integration` 的 `72bc91c` 新开分支 `grok/pilot46-fixes-20260907`，用自己的 worktree（例如 `/data/jzy/tmp/wt-grok-pilot46-fixes`），不要在 Codex 的 worktree 里改东西。
- Python `/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`；跑测试时 `PYTHONPATH=src:tmp/native_python_addons_v1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`；先确认 `import avengine` 解析到你自己的 worktree。`tmp/` 是到数据盘 `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/` 的符号链接。
- 先导批次原件 `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_pilot46_background_20260907_v1/`（46 段：39 delivered / 5 failed / 2 blocked，`summary/` 已生成）只读，不许覆盖或删除任何 `attempt_01`。重跑一律用同一逻辑 Episode ID、新输出根、`attempt_02`。
- 清单 `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p10_pilot46_manifest_20260907_v3/batch_manifest.json`；批次入口 `tools/dataset/run_qa_batch.py --manifest ... --output <新目录> --max-parallel 4 --min-free-gpu-mb 8192`（服务器上还有别人的训练在跑，开跑前看 `nvidia-smi`）。
- 改了 `tools/` 下入口要重生成 `docs/TOOL_INDEX.md`（`tools/build_tool_index.py`），并让 `tests/unit/test_tool_index_current.py` 过。
- 不 push、不合并 main、不切换正在运行的 Studio 服务；push 与合并等 owner 说。

## 1. owner 已经定下的规则（照做，不再讨论）

1. 声音片段上限仍是 5 秒（本批 495 条声音里没有一条超过 5.00 秒）。repeat 缺额的真实原因是"同一设备放两遍加另一台设备的一段声音加两个 0.5 秒间隔"超过了 13 秒预算（16 秒减 3 秒保留尾段）。**规则：预分配阶段就做排程可行性检查**，给 repeat 画像分配声音时必须满足 `2×重复声音时长 + 另一源声音时长 + 2×0.5 s ≤ 13 s`，不满足就在分配阶段换别的合法身份；分配定下后运行时不许再换。放量清单里这类缺额应为 0。
2. **不做**训练/测试集的房间与说话人预留（owner 决定不用）。
3. **打散**声源类别对与条件组的绑定：批配置生成时按 seed 随机分配条件组，保证每个家族每个条件组至少出现一次、每种类别对（人人/人动/人设/动动/动设/设设）在各条件组之间分布开；清单里输出类别对×条件组的交叉表。
4. 覆盖表五态规则：**代码缺陷或接口不通导致的运行失败记 `interface_not_implemented`**（reason 写失败阶段与错误摘要）；**画像内 200 次重试用尽记 `evidence_missing_or_unsampled`**（带失败直方图）；预分配缺额仍记 `evidence_missing_or_unsampled`。失败原因必须写进 `outcome.json` 与 `summary/failed_episodes.json`，读者不用翻 stderr。
5. QA-18 湿尾边距按 Codex 现状（只用实测湿尾区间，不加边距），不动。
6. 五段抽听与 P7 十条试听记录保持 pending_human，不动、不代填。

除以上六条，本任务书里还有一项 owner 没有单独裁定的内容，就是 G-C 第 4 条"画外声源画像"。它依据 owner 2026-09-03 的既定结论"画外声音才是头条"加入；owner 如不要，删掉那一条即可。

## 2. 分工（四个子代理按文件占用切开，互不重叠；G-E 由父代理做）

### G-A Habitat 收口与批次记账（文件：`src/avengine/rooms/qa_delivery.py`、`src/avengine/timeline/unified_audio_receipt.py`、`tools/dataset/run_qa_batch.py`、`src/avengine/qa/batch_delivery.py`、`src/avengine/qa/batch_coverage.py`、对应 tests）
1. **Habitat AudioProgram 模式**：`qa_delivery.py:283` 现在写死 `sequential_sources`，overlap 与一人沉默的段必失败（本批 mp3d/hm3d 的 human_animal、single_active 四段，捕获全成功、死在 `avengine.cli m5 render-current-mp3d-dynamic-audio`）。按 UE 路径 `tools/acoustics/render_frame_readback_sequential_speech.py:738` 附近的逻辑选模式：事件跨端点重叠 → `simultaneous_subset`；只有一个端点有事件且候选端点 ≥2 → `one_active_of_n`；单端点多事件有间隔 → `intermittent_events`；其余 → `sequential_sources`。校验器 `audio_program.py:181-235` 不改。
2. 同一处的 `--beagle-audio`：`_build_habitat_audio_command` 用"第一个声音路径"填这个必填老参数，cli 把它登记成一条用不上的 `dog_beagle_v2_scheduled_dry` 绑定。把 cli 该参数改为可选（`src/avengine/cli.py:2315` 附近），收口层只在真有 beagle 历史绑定时传。
3. **失败记账**：`run_qa_batch.py:553` 只写 `controller_exit`。改为解析控制器的失败阶段（规划用尽 / 捕获 / 音频 / 收口）与错误首行，写进 `outcome.json`（`failure_stage`、`failure_reason`、`gap_state`）与 `summary/failed_episodes.json`；`batch_coverage.py` 按第 1 节第 4 条把这些段的格子记到对应五态，reason 带失败阶段。
4. **EvidenceContract 前验**：`qa_delivery.py` 现在是 :1254 normalize → :1266 写 facts → :1274 写 questions → :1297 才 `validate_evidence_contract`。改成契约与双耳 WAV 都验过再写任何交付文件；失败时目录里不能留下 facts/questions。
5. **湿尾越界校验**：`unified_audio_receipt.py:174-176` 只拦 source_activity 越界，wet tail 越出 `clock.sample_count` 被放行（P9 验收段 `tmp/p9_finalize_20260907/mp3d_final_v7` 的 facts 里 end_s 2.439 s 对 2.0 s 音频）。改为越界即 fail，并在渲染器侧把湿尾区间截到时钟内并记录被截。
6. 单测：新旧模式各一条真实 AudioProgram 校验；失败记账三类各一条；契约失败不留半成品一条；湿尾越界拒绝一条。

### G-B UE 视觉侧（文件：`tools/rooms/run_spear_residential_episode.py`、`tools/rooms/measure_ue_room_floor_reference.py`、`src/avengine/rooms/room_package.py`、`examples/rooms/packages/*.json`、`src/avengine/qa/batch_delivery.py` 的 review 部分只加闸门函数，与 G-A 协调不要改同一段）
1. **A/B/C 过曝根因**：本批 16 段 A/B/C 在新共同舞台 `qa_full_asset_ue_stage_20260907_v1` 上第 120 帧灰度均值 240–252、≥250 像素占 38%–94%；同房间旧交付（`tmp/qa_real_rooms_20260906/walk_pair_a_v6` 等，旧舞台 `/data/avengine_external/workspaces/multi_home_activity_20260905/ue_stage/SpearSim`）均值 125–196、饱和 0。已经排除：uproject、六个 Config/*.ini、五张地图字节、旧舞台 Content 全集（新舞台一个不缺）、execcmds、DDC 目录与 profile、分辩率与预热帧都相同；唯一实质差别是舞台目录。做法：同一个 A 房请求（可用本批 `authored_a_human_human` 的 request.json 改输出目录），在旧舞台与新舞台各渲不超过 10 帧（旧舞台只读），对比灰度直方图；查新舞台多出的 297 个 Content 文件与 `Saved/`、`Intermediate/`、`DerivedDataCache/` 差异；定根因、修好、写清。
2. **曝光闸门**：`batch_review` 增加画面质量检查：对第 0/中/末帧算灰度均值与 ≥250 像素占比，占比超过 20% 或均值超过 235 → `review_failed`，reason 写数值；阈值写成配置并标 placeholder。这个闸门要在重跑前就位。
3. **地板重量**：A/B/C/Kujiale 四个 UE 房间的 `floor_reference` 值 0.0001953125 m 是深度缓冲 1/4096 m 的量化残差（`method.line_trace.hit_count` 全 0）。要么让 line trace 真正命中（地板缺碰撞体就在测量时临时给地板网格加碰撞或改用对已加载几何的射线求交），要么在包与 floor_reference.json 里如实写 `measurement_kind=depth_readback_fallback`、`precision_m=0.00025`，不许写成 measured line trace。`tmp/p3_room_packages_20260907_v6/native_apartment/floor_reference.json` 报 −74.7625 m 却标 measured，改标 invalid。Apartment 0.2711074501 m 不动。
4. **包路径清理**：`examples/rooms/packages/room_{a,b,c}.json` 各 20 个字段指向不存在的 `living_props_v7_candidate/`、`detailed_v8_linear_materialfix/`、`polished_v3_final/`（真实文件在 `/data/avengine_external/workspaces/multi_home_activity_20260905/polished_v3_final_attempt*/` 等处，逐个核对后指向真实存在的版本或删字段）；`kujiale_0020_full_home_v1.json` 的 `map_asset` 少 `SpearSim/` 一段；`hm3d_00800_TEEsavR23oF.json` 的 `route_bank` 多了一层 `routes_R3`。`room_package.py` 校验器对非 `/Game`、`/Root` 的绝对路径加存在性检查，"每包缺什么"的表要列出缺失文件。
5. 单测：曝光闸门对全白帧与正常帧的判定；路径存在性校验；地板文件 measurement_kind 字段。

### G-C 采样器与批清单（文件：`src/avengine/rooms/conditioned_sampler.py`、`src/avengine/qa/batch_manifest.py`、`src/avengine/qa/batch_sound_pool.py`、`tools/dataset/build_qa_batch_manifest.py`、`examples/dataset/*.json`、对应 tests）
1. **同层约束**：`conditioned_sampler.py:303-305` 只夹 x/z，y 不管；P5 的 HM3D 验证段两源根高 2.02 m 与 0.16 m 差一层楼。放置点与相机所在地板同层（|Δy| ≤ 0.3 m），多层场景显式抽层再放置；HM3D 包已列两层 `R3_floor_0.1634`、`R3_floor_3.1634`。
2. **repeat 可行性在预分配阶段检查**（第 1 节第 1 条的规则），`batch_manifest.py` 分配时实现；分配不到合法身份才记缺额。
3. **打散类别对与条件组**（第 1 节第 3 条），`build_qa_batch_manifest.py` 与配置生成实现，输出交叉表。
4. **画外声源画像**（owner 可删）：现在非锚点发声者被强制在视锥内（:480），所有发声者受 `distance_range_m` 默认 [1.5, 4.5] 约束，本批没有一段画外画像。加入 `anchor_visibility=off_screen` 与"竞争者画外"两种可抽画像，距离上限做成配置（放量默认允许到 6 m），画外实体仍参与最近竞争者与音频。
5. **箱内分布如实报告**：本批 60–90 度箱四段实测全在 60.5–61.9 度（采样找到任何合法机位就停）。覆盖与配额报告一律按实测分离角每 5 度一档统计，不再按请求箱宣称"已覆盖"；可选：加一个 `separation_target_policy=uniform_in_bin` 选项，先抽目标角再接受 ±2 度内的解。
6. `sound_selection.clip_span_fit_policy` 是死字段（:209-218 写死），要么接上要么从请求 schema 删掉；speech_motion 三种模式的语义（speaker_moving 不要求竞争者静止）写进 v2 文档旁的说明。
7. 单测：同层约束（构造两层导航网）；repeat 可行性；交叉表覆盖；画外画像合法性；箱内分档报告。

### G-D 外观与像素证据（文件：`src/avengine/rooms/qa_evidence.py`、`src/avengine/qa/batch_manifest.py` 的 appearance 部分（与 G-C 协调，只改 appearance 字段映射）、对应 tests）
1. 非人类外观门槛：`qa_evidence.py:268` 默认 `minimum_color_pixels=8`，人类是 512 加 1.6 倍优势（:147/:184）；三色犬判据 :351-363 等于"≥2 白/≥4 暗/≥2 暖棕像素"。改成与人类同量级的最少像素数，颜色成分改比例阈值，全部标 placeholder；`build_pixel_appearance_review` 显式传参。
2. 设备外观字段：批清单按 `body_color` 记色，而 `qa_evidence.py:218` 只读 `finish/surface_finish`，本批 6 段设备外观为 null、28 个设备资产都在这条缝里。统一为登记表有什么就读什么（finish 优先，其次 body_color），review 写清用了哪个字段。
3. 语义字段：`in_fov_frame_count` 用 `target_pixels>0` 判，完全被遮挡也算在视野内；可见率只算遮挡不算出画。加 `visible_pixel_frames`（visible_pixels>0）与 `bbox_touches_frame_edge_frames` 两个字段，写进 achieved_conditions 与 pixel truth，文档写明 in_fov 的含义。
4. 单测：8 像素门槛的假阳性构造（一张几乎看不见的猫必须 not_observable）；body_color 设备被审阅；出画截断计数。

### G-E 父代理：重跑、最终覆盖表、放量就绪报告
1. 等 G-A、G-B 就位后（曝光闸门必须先在），用同一清单、新输出根 `qa_pilot46_rerun_20260907_v1`、`--episode-id` 指定，只重跑：4 段 Habitat 失败段、16 段 A/B/C 段、以及被 G-C 同层约束或修复影响的段；`attempt_01` 不动。
2. 合并原批与重跑生成最终先导汇总（`attempt_02` 取代对应 `attempt_01`）：五态覆盖表、按实测分离角的分档表、失败原因表、类别对×条件组交叉表、音频电平、我方审计器（`tools/qa/audit_binding_feasibility.py`，不改）逐段结果、曝光闸门结果。46 格每格都要有五态之一与证据路径。
3. **放量清单干跑**（只生成不执行）：用打散后的条件分配、repeat 可行性检查、画外画像，按 7 个房间各 50 段生成一份放量清单，报告预分配缺额数（应为 0，除非 owner 另有规则）、交叉表、声音身份复用情况。
4. **就绪报告** `docs/roadmap/grok_reports_20260907/READINESS_20260907.md`：逐条列本任务书每一项的通过/未完成、证据路径、测试计数；明确写"可以开大规模生产"或"还差什么"。没做完的按题义不适用 / 接口未实现 / 证据缺失三分法写清，不把测试变绿说成验收通过。

## 3. 每项的验收标准（做不到就写没做到）

- G-A：本批四段失败的 Habitat 请求在你的 worktree 上重新收口（可只跑收口，复用 `attempt_01/episode/capture`）都能出 facts/questions 与 16 秒双耳成片；构造的 outcome.json 带失败阶段与五态；契约失败目录里没有 facts/questions；湿尾越界的回执被拒。
- G-B：两舞台对照的直方图与根因说明；修好后 A/B/C 各渲一段 240 帧，第 0/120/239 帧灰度均值在 90–200、≥250 像素占比 <5%；四房地板文件字段如实；七个包路径存在性检查全过并附"每包缺什么"表。
- G-C：HM3D 连续 20 次规划两源同层；46 格与放量干跑清单的 repeat 缺额为 0；交叉表里每种类别对至少出现在 3 个不同条件组；画外画像在四个家族各能规划出一段（plan-only 即可）。
- G-D：本批 `apartment_animal_animal` 的 Jack Russell（standard_white_tan，0 帧 reviewed）与 `mp3d_animal_animal` 的两只猫用新门槛重新审阅，结果与截图一致；6 段设备外观不再为 null。
- G-E：最终覆盖表 46 格无空缺；就绪报告成文。

## 4. 就绪门（全部满足才算"可以准备大规模生产"）

1. 46 格每格有五态与证据路径；delivered 段全部通过曝光闸门与契约前验；失败段的原因可从 outcome/summary 直接读到。
2. 四个家族 × 人/动物/设备三类，每格至少一段通过闸门的成片。
3. 放量清单干跑：预分配缺额 0、交叉表打散、画外画像在列。
4. 所有相关单测 0 failed / 0 skipped，`docs/TOOL_INDEX.md` 与工具一致。
5. 就绪报告写明未完成项（例如 9 个挂墙/吊顶资产的挂装接口、人工校准、五段抽听）仍是空的，不冒充完成。

## 5. 报告格式（每项完成后）

1. 改了哪些文件（路径），提交号。
2. 跑了哪些测试，各自的通过/失败/跳过计数；有失败贴错误原文。
3. 验收产物的路径，以及你亲自看过或听过的核对结果。
4. 没做完的部分和原因，分清题义不适用、接口未实现、证据缺失。
5. 对后续接口的要求（字段名、函数签名、数据位置）。
6. 与本任务书冲突、需要 owner 拍板的地方，单独列出，不要自行决定。
