# HANDOFF 20260824：资产生成 Web 流程（新会话开工文档）

> 写给接手"资产生成网页"的新会话。读完再动手。
> 服务器 `ssh 48g-jump`；主仓 `/data/jzy/code/AVEngine-lead-a`（推 main 用
> `git push origin HEAD:main`，owner 已开 bypass）。
> 本文里"已存在"的部分我都在服务器上验证过；"缺口"是我确认过不存在的。

---

## 一、要做什么（owner 的原话转述）

做一个网页，让使用者：

1. 选/输资产类型（例如"猫"）；
2. 再选/输品种（例如"英短""阿比西尼亚"）；
3. 一次生成**一批**（多个个体/多种外观变体）；
4. 最后**导出并注册**成本项目已有的资产格式，能被引擎直接拿去渲染。

**要打通的是整条路**，不是只做前端。允许分阶段交付，但每一段都要能跑通到
下一段的输入格式。

---

## 二、现状地图（先认清三块地盘，这是最容易走弯路的地方）

链路横跨**三个代码库 + 两个 conda 环境**：

| 地盘 | 位置 | 负责的环节 |
|---|---|---|
| **SPEAR 仓库** | `/data/jzy/code/SPEAR-lead-b`（还有几个同源工作区 `SPEAR-lead-b-*`） | 生成：文生图 → 3D 网格 → 贴图 → 装配/绑骨/审核 |
| **AVEngine 主仓** | `/data/jzy/code/AVEngine-lead-a` | 注册表、资产契约、渲染链、Studio 网页 |
| **外部产物区** | `/data/avengine_external/`、`/data/datasets/` | 大文件（包、stage、数据集）都在这里，不进 git |

环境：生成用 `/data/jzy/miniconda3/envs/hunyuan3d/bin/python`；引擎/注册表用
`/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`。

### 已经存在的（不要重造）

**1. 批量生成骨架已经有了**：`SPEAR-lead-b/tools/batch_animal_pipeline.py`
它对每个 `(species, breed)` 元组跑完整的
`Flux → Hunyuan-Shape → Hunyuan-Paint → UV 提取 → compile`，产物落到
`<workdir>/<species>/<breed>/`：`reference.png` / `reference_rembg.png` /
`shape.glb` / `hy3d_textured.obj` / `hy3d_diffuse.jpg` / `turntable_grid.png`
/ `meta.json`。文件头部注释里写了三套提示词模板（四足 / 小型四足 / 鸟类），
以及为什么要"侧视、尾巴抬高、四腿分开、纯白底"——这些是下游绑骨的硬要求，
**改提示词前先读那段注释**。

内置的 `SPECIES_LIST` 已经覆盖：狗（金毛、比格）、猫（波斯、橘虎斑、英短）、
鼠（白/灰）、鸡（母鸡、公鸡）、鸭（绿头）、鸟（黄雀）。**这正好说明"选物种
+ 品种"这件事在生成侧是现成的**，网页要做的是把它参数化、批量化、可视化。

**它明确没做的一步**（文件注释原话）：不做 swap-into-rig，因为现有的骨架是
狗专用的。**跨物种的绑骨迁移策略是本任务最大的技术缺口**，见 §四。

**2. 绑骨/审核工具链已经有了**（`SPEAR-lead-b/tools/` 下 30+ 个 blender 脚本）：
`blender_fit_i23d_to_animal_template.py`、`blender_build_stable_animal_instance.py`、
`blender_normalize_generated_animal_heading.py`、
`blender_level_generated_animal_support_plane.py`（四足支撑面找平）、
`blender_audit_generated_animal_rig.py`、`blender_measure_generated_animal_emitter.py`
（算发声点 muzzle 偏移）、`blender_render_generated_animal_coat_views.py` +
`blender_project_animal_multiview_coat.py`（外观/毛色变体）等。

**3. 打包与变体工具在主仓**：`AVEngine-lead-a/tools/m2/` 35 个脚本，其中与本
任务直接相关的：`compile_animal_package.py`、`assemble_variant_package.py`、
`build_appearance_variant_inputs.py`、`audit_variant_candidate.py`、
`capture_animal_variant_review.py`、`promote_canary.py`。

**4. 已经有 7 个"生成动物"注册成功了**（证明这条路是通的）：
`generated_abyssinian_ruddy_…`、`generated_border_collie_black_white_…`、
`generated_labrador_yellow_…`、`generated_shiba_inu_red_…(v1/v2)`、
`generated_pembroke_welsh_corgi_red_white_…`、
`generated_british_shorthair_blue_medium_stocky_adult_research_v1`。
**照着其中一个反推格式，比读文档快。**

**5. Studio 网页框架已经有了**（就是要扩展的地方，见 §五）。

---

## 三、"注册成资产"到底要产出什么（终点格式）

这是最容易做错的一环：不是"导出一个 glb 就完了"。一个可被引擎使用的资产要同时
落到 **4 张注册表 + 1 份包清单**：

| 产物 | 路径 | 关键字段 |
|---|---|---|
| m2 资产包清单 | `<包目录>/asset_manifest.json` | `actions`（idle/walk + 采样数）、`anchors`（body/head/muzzle + `joint_from_anchor`）、`admission_state` |
| 运行时档案 | `examples/runtime/source_asset_runtime_profiles.json` → `assets[]` | `asset_id`、`identity{species_id,breed_id}`、`realized_attributes{size,body_build,life_stage,coat_profile}`、`geometry{mesh_authority,source_mesh_uri,rig_authority}`、`timeline{template_id,body_plan_id,local_anatomical_forward_axis,walk_phase_period_frames,idle/walking_action_id}`、`default_emitter_anchor_id`、`emitter_anchors[].offset_m` |
| 实体资产注册表 | `examples/m6/registries/entity_assets_v1.json` → `entities[]` | `entity_asset_id`、`entity_class`、`visual_asset{uri,sha256}`、`capabilities{articulated,skeleton_revision,…}`、`emitter_anchors`、`provenance{license,rights_status,evidence_sha256}`、`admission_state` |
| 动物模板注册表 | `examples/m6/registries/animal_templates_v1.json` → `templates[]` | `template_id`、`body_plan_id`、`morphotype_id`、`taxonomy{species_id,breed_id}`、`asset_revisions{topology/uv/skeleton/skin_weights + 各自 sha256}`、`appearance_domains`（size/body_build/life_stage/coat_profile 三值域） |
| 声源端点（要发声才需要） | `examples/m6/registries/source_endpoints_v1.json` 或独立端点表 | 绑定到 `entity_asset_id` + `emitter_anchor_id` + `allowed_sound_class_ids` |

**注册表的机械要求（踩过坑）**：
- `registry_content_sha256` = 除该字段外整份 JSON 的 `canonical_json_sha256`
  （用 `avengine.contracts.json_io.canonical_json_sha256`）；
- 端点表要求条目按 `(id, revision)` **字节序排序**，否则校验失败；
- 改动 canary 注册表会牵动 `tests/unit/test_m6_registry.py`，**改完必跑全量
  单测**（基线约 3111 passed，`$PY -m pytest tests/unit -q`，约 5 分钟）；
- 新增条目走"文本级插入"比整份重排更安全（避免无关 diff）。

**还有一步很多人会漏**：注册完还要**进 UE stage 才能渲染**。这条最后一公里是：

```text
接受的 GLB（带 idle/walk 动画）
  → 导入 UE 内容库，生成该 actor 的 uasset（骨骼网格 + 动画 + Blueprint）
     落到 /data/avengine_external/ue-assets/actor_content_registry_v<N>_<时间戳>/
        cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/<gate_...>/
  → tools/ue/build_minimal_closure_report.py   （BFS 闭包；动画必须显式做种子）
  → tools/ue/assemble_package_stage.py         （组装并 cook 出生产 stage）
  → Studio 的 apartment_end_to_end 模板指向新 stage → 渲一段 5 秒片验证
```

**命名惯例看现成的就懂**（当前 registry v9 里的目录名）：
`gate_pixal_generated_abyssinian_ruddy_v1`、`gate_pixal_generated_border_collie_black_white_v1`、
`gate_pixal_cat_british_shorthair_shadow_cleanup_v1`、`gate_m2_beagle_v7_world_contact_r5`、
`gate_rocketbox_male_adult_01_controlled_material_*_ue_v3`（换色人）。

复跑要点（三个 SDK 环境变量 + `spear-ext` 的 PYTHONPATH + 动画种子的教训）都写在
`docs/roadmap/STAGE_REBUILD_20260823.md`。**新资产不进 stage = 网页上"注册成功"但
引擎渲染不出来**，所以这一步要在第一版就打通，不能留到最后。

---

## 四、真正的缺口（按难度排序，这是任务的核心）

1. **跨物种绑骨（最大缺口）**。现有骨架是狗专用；猫可以复用四足犬科骨架做适配，
   但鸟/鼠需要各自的骨骼方案。`blender_fit_i23d_to_animal_template.py` +
   `blender_build_stable_animal_instance.py` 是起点，但**每个新体型都要重新验证**
   支撑面找平、朝向归一、走路周期。建议第一版**只支持猫**（与已注册的英短/
   阿比西尼亚同体型，风险最低），把整条路打通后再扩物种。
2. **"一批"到底批什么**。契约（见 §六）规定：**换品种 = 新的源资产**，不能靠
   改色改尺寸伪造。所以"生成一大堆"有两种合法含义，网页应当把它们分开：
   - **实例批量**（低风险）：同一已接受源资产下，穷举
     `size × body_build × life_stage × coat_profile` 的合法组合 → 这才是
     "一次生成一大堆"的正解，用 `build_appearance_variant_inputs.py` +
     `assemble_variant_package.py`；
   - **新源资产**（高风险、需要 owner 审）：新品种，必须走完整生成 + 绑骨 +
     审核流程，且**中途有一个 owner 硬停**（见 §六第 4 条）。
3. **端到端编排缺一个 runner**。现在每一步都是独立脚本、参数各异；网页需要一个
   能把"参数 → 多阶段任务 → 产物目录 → 注册"串起来的编排层（Studio 的任务队列
   可以直接用，见 §五）。
4. **状态可视化**。生成一批要几十分钟到几小时，网页必须能显示每个个体卡在哪一
   步、失败在哪、产物预览（参考帧/转台图/审核视频）。

---

## 五、网页怎么做：**扩展 Studio，不要另起炉灶**

主仓已有完整的网页后端：`src/avengine/studio/`（`server.py` / `tasks.py` /
`templates.py` / `catalog.py` / `config.py` / `validation.py`）+ 前端
`tools/studio/static/`（`studio.html` / `studio.js` / three.js）。它已经解决了
你要重做的一大半：

- **任务队列**：`StudioTaskQueue`（`tasks.py`）——提交、串行执行、日志尾读、
  状态持久化、断电恢复都有了；
- **模板机制**：`templates.py` 里每个任务类型是一个"模板名 + 可覆盖键集合"
  （`TEMPLATE_OVERRIDABLE_KEYS`），默认参数写在
  `tools/studio/studio_config_48g.json` 的 `task_templates`；**加一个
  `animal_asset_generate` 模板就是新流程的落点**；
- **现成 API**：`/api/health` `/api/templates` `/api/tasks` `/api/registries`
  `/api/sounds` `/api/scenes` `/api/validate` `/api/sweeps`（笛卡尔批量，
  ≤64 点/批，正好可以用来做"变体批量"）；
- **服务只绑回环**：`ssh -L 8765:127.0.0.1:8765 48g-jump` 后开
  `http://127.0.0.1:8765/studio`；用法见 `docs/studio/USAGE.md` 与 README 速查节。
  重启用 `fuser -k 8765/tcp`（**别用 `pkill -f`，会杀掉自己的 ssh 会话**）。

建议的页面形态：新增一个"资产工坊"页签（或独立路由），三段式：
**① 身份表单**（物种 → 品种 → 属性域勾选）→ **② 批量预览与提交**（列出将要
生成的组合、预估耗时、提交进队列）→ **③ 审核与注册**（逐个看参考图/转台图/
走路视频，通过的勾选 → 一键写注册表 + 出 diff 供人确认）。

**注册那一步不要自动直推 main**：生成 registry 补丁 + diff，让人确认后再提交。

---

## 六、硬约束（违反即返工，全部来自 owner 明令或已入仓契约）

必读：`docs/assets/GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md`（全文，
不长）与 `docs/assets/REAL_REFERENCE_ANIMAL_COAT_WORKFLOW.md`。要点：

1. **新品种 = 新源资产**，绝不能用已接受品种改色/缩放/微变形冒充。边境牧羊犬
   不能由拉布拉多改出来——契约里专门留了一段"边牧失败教训"。
2. **实例只能在已验证域内变化**：`size`(small/medium/large)、
   `body_build`(slim/standard/stocky)、`life_stage`、该品种**已审过的三种毛色
   profile**之一。换毛色 ≠ 换品种。
3. **各阶段权威不可越位**：品种身份→真实照片；标准 2D 图→FLUX/Qwen 输出；
   新几何→Pixel3D/Hunyuan 输出；拓扑→显式修复；骨架→目标原生推断（不许拷贝
   别的体型的关节/权重）；毛色→该品种真实参考（**禁止用 RGB 相乘冒充真实毛色**）。
4. **有一个 owner 硬停**：标准 2D 图必须经 owner 接受品种与解剖结构后，才能送去
   生成 3D。网页要把这一步做成显式的"待审"状态，不能自动放行。
5. **禁止的捷径**（契约原文）：用一个好看的角度掩盖几何/形变问题；把被拒的候选
   偷偷放行；用通用泥塑模板抹掉品种特征只为凑齐四条腿。
6. **项目铁律**：全链 `research_only` + `episode_counted=false`，正式数据集分母
   保持 0；不新增 hash/frozen-contract/闸门；输出 fresh/no-clobber（目标已存在
   就换新路径，绝不覆盖）；`blender_custom` 房间永久禁用；新资产入注册表要
   owner 点头。
7. **自证要求**（owner 对协作代理的硬性要求）：凡产出视觉内容，**必须自己看渲染
   帧**并在交付里写"我看过 X 帧，确认了 Y"；凡改管线，必须有机器核验（逐像素/
   逐字节对照、全量单测不回退）。"逻辑上应该没问题"不算完成。

---

## 七、建议的推进顺序

1. **先跑通一个已知品种的复现**：挑已注册的英短或阿比西尼亚，用现有脚本从
   `batch_animal_pipeline.py` 一路走到 `asset_manifest.json`，**把每一步的真实
   命令、参数、耗时、产物路径记下来**。这是后面所有编排的规格来源。（不要先写
   前端。）
2. **把这条路封装成一个 Studio 任务模板**（`animal_asset_generate`），参数 =
   物种/品种/属性域/数量/seed；产物落 fresh 目录；日志走队列。
3. **做注册补丁生成器**：从产物目录 → 4 张注册表的条目（含 sha256 与 canonical
   hash），输出 **diff/补丁文件**，不自动提交。跑全量单测确认不回退。
4. **再做前端三段式页面**（表单 → 批量预览 → 审核注册），复用 Studio 现有的任务
   列表与日志组件。
5. **打通最后一公里**：新资产 → UE 闭包报告 → 组装 stage → 在公寓房间实际渲一
   段 5 秒片验证（这一步失败率最高，早做早暴露）。
6. **扩物种**：猫打通后再考虑鼠/鸟，每个新体型都要重跑绑骨与四足/两足支撑面验证。

---

## 八、快速自检（开工前 5 分钟）

```bash
ssh 48g-jump 'cd /data/jzy/code/AVEngine-lead-a && git log --oneline -1'
ssh 48g-jump 'curl -s http://127.0.0.1:8765/api/health'          # Studio 在跑吗
ssh 48g-jump 'ls /data/jzy/code/SPEAR-lead-b/tools | grep animal | head'
ssh 48g-jump 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

**GPU 纪律**：4 张 4090D，其中 2 张长期被他人占用；生成用 GPU 前先看
`nvidia-smi`，只用空闲卡。契约要求生成阶段**全 GPU 常驻**，不要开 `low_vram`
或 CPU offload（放不下就报确切的阻塞点）。

## 九、本项目的通用踩坑清单

- 长任务一律 `nohup … > /data/jzy/tmp/xxx.log 2>&1 &` + 轮询完成标记；长 ssh
  管道假死很常见（exit 255 多为无害，去服务器上查证）。
- 复杂脚本别用 heredoc 塞进 ssh（转义会炸），写本地文件 `scp` 过去跑。
- `pkill -f <pattern>` 会匹配到你自己的 ssh 命令行**把自己杀掉**——用脚本文件里
  的 pgrep，或用 `fuser -k <port>/tcp`。
- UE 捕获出来的 `rgb.npy` 是 **BGR**，出片要 `--channel-order bgr`。
- 引擎处处 fail-closed：被拒绝通常是**防废案**不是故障，先读拒绝原因再改参数。
- zsh：`===` 会被当成 glob 报错；grep 零命中 exit 1 会断 `&&` 链。

## 十、可以直接参考的文档

| 文档 | 用途 |
|---|---|
| `docs/assets/GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md` | **必读**，资产身份与流程契约 |
| `docs/assets/REAL_REFERENCE_ANIMAL_COAT_WORKFLOW.md` | 毛色/外观变体路线（SPEAR 侧实现） |
| `docs/studio/USAGE.md` | Studio 现有能力与操作 |
| `docs/roadmap/STAGE_REBUILD_20260823.md` | 新资产进 UE stage 的复跑要点 |
| `docs/roadmap/PERF_OPT_GUIDANCE_20260823.md` | 吞吐优化的处置结论（批量生成会用到） |
| `docs/roadmap/DATA_DIVERSIFICATION_WORKORDER_20260823.md` | 另一条并行工单（多房间/多视角），注意别与其撞车 |
| `docs/adr/ADR-0006-template-authoritative-animal-assets.md` | 模板权威决策（契约引用） |

---

## 十一、与另一条工作线的边界

同时另有一个代理在做**数据多样化与吞吐优化**（分支 `codex/*`，见 §十的两份
工单）。边界：

- 他动的是渲染/音频/出题/训练管线与多房间视角；
- **你动的是资产生成与注册**；
- 冲突面只有两处：①注册表文件（改前先 `git pull`，条目用文本级插入）；
  ②GPU 占用（用卡前 `nvidia-smi`）。
- 正在跑的产物目录 `/data/avengine_external/review/qa_v2_*` 与
  `/data/datasets/spatial-omni/avengine_*` **只读**，不要写。
