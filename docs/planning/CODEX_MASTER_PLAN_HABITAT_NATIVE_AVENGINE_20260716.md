# Codex Master Planning Brief

# AVEngine Habitat-Native 重构、仓库治理、开源归属与论文定位总规划

**日期：2026-07-16**  
**目标运行时仓库：** `https://github.com/Eastforward/habitat-sim-AVEngine`  
**建议新建主仓库：** `https://github.com/Eastforward/AVEngine`  
**文档用途：** 直接交给 Codex，作为重新规划 AVEngine 全部工程、仓库、数据契约、开源归属和论文叙事的总任务说明。

---

## 0. 给 Codex 的最高优先级指令

你正在重新规划一个名为 **AVEngine** 的非商业研究项目。AVEngine 不再以“UE/SPEAR 视觉 + Habitat/RLR 声学”的跨引擎拼接作为论文主路径，而是迁移为：

> **建立在 Habitat-Sim research fork 之上的、面向动态关节动物、多声源空间音频、严格同步和反事实视听数据的数据集生成引擎。**

当前已经存在一个公开 fork：

- `Eastforward/habitat-sim-AVEngine`
- 上游：`facebookresearch/habitat-sim`

开始任何大规模代码修改前，必须先完整阅读服务器中的：

- `AVEngine_review_20260715_zh.md`
- `avengine_timeline_v2.schema.json`
- 若存在：`CODEX_GOAL_AVENGINE_QUADRUPED_AVSYNC.md`
- 旧 AVEngine 仓库中上述文档提到的实际代码、测试、资产和运行入口

本轮任务的第一目标不是马上堆功能，而是完成一次**可执行、可审计、可拆分为 Issues 和 Milestones 的总体重规划**。必须先建立仓库边界、系统边界、数据契约、迁移矩阵、许可证与引用规则，再推进 Habitat-native canary。

不得：

- 把 Habitat-Sim 的现成功能写成 AVEngine 原创；
- 把 RLR 的多声源 C API 写成 AVEngine 发明的新声学算法；
- 声称 Habitat-Sim 无法做关节动画、PBR、多相机或自定义场景；
- 声称视觉 PBR 材质会自动变成正确的声学材质；
- 删除 Habitat-Sim 上游 Git 历史、版权声明或 MIT License；
- 把全部代码简单标成 MIT，而忽略 RLR 的 CC BY-NC 4.0；
- 未通过真实 canary 就声称 Habitat-native 主链已经完成；
- 静默修改 `avengine_timeline_v2.schema.json` 的语义；
- 在没有实际稳定模板时宣称支持任意生成动物 mesh；
- 把当前无嘴部动画的设计描述成 lip synchronization 或 mouth animation synchronization。

---

# 1. 项目最终定位

## 1.1 一句话目标

> **AVEngine 是一个 Habitat-native 数据集引擎：Blender 与生成模型负责离线产生经过审计的动物资产，Habitat-Sim fork 负责统一执行场景、视觉传感器、关节姿态和 RLR 声学，AVEngine 主仓库负责场景/资产编译、权威时间轴、反事实 episode、QA、provenance 和数据集准入。**

## 1.2 不再采用的主叙事

不要再把系统描述成：

```text
我们另外做了一个模拟器，以替代 Habitat-Sim / SoundSpaces。
```

也不要描述成：

```text
Habitat-Sim 不能多声源，所以我们必须从零开发 AVEngine。
```

更准确的系统关系是：

```text
Habitat-Sim
  = 场景图、视觉渲染、传感器、物理、导航和关节对象基础运行时

RLR-Audio-Propagation / SoundSpaces 2.0
  = 几何声学传播能力和方法来源

Blender + AVEngine Offline Asset Compiler
  = 动物模板、形态拟合、固定拓扑/UV、骨架、权重、动作烘焙和动画 QA

AVEngine
  = 将上述能力组织成身份一致、属性可控、严格同步、可反事实、可验证的数据样本
```

## 1.3 论文主定位

推荐英文定位：

> **AVEngine is a Habitat-native dataset engine for synchronized, identity-preserving, counterfactual articulated audio-visual source grounding.**

推荐中文定位：

> **AVEngine 是建立在 Habitat-Sim 研究 fork 之上的动态视听数据集引擎，面向具有身份、属性、骨骼动作和语义发声事件的多声源关节场景。**

---

# 2. 已确认的起点与 Codex 必须再次核验的事实

## 2.1 当前公开 fork

当前仓库：

```text
Eastforward/habitat-sim-AVEngine
```

GitHub 已明确显示它 fork 自：

```text
facebookresearch/habitat-sim
```

从公开页面看，它目前仍基本保持上游 Habitat-Sim 的目录、README 和历史；尚未形成 AVEngine 专属 README、修改说明、版本发布和仓库边界。Codex 必须在本地仓库中用 Git 命令核实，而不能只依赖网页观察。

必须记录：

```text
origin URL
upstream URL
current branch
HEAD commit
upstream/main commit
merge-base commit
是否存在本地未提交修改
fork 相对上游 ahead/behind 数量
所有 submodule 的实际 commit
Habitat-Sim base tag/release
编译器、Python、CUDA、Magnum、Bullet 和 RLR 版本
```

建议核验命令：

```bash
git remote -v
git status --short --branch
git rev-parse HEAD
git fetch upstream --tags
git rev-parse upstream/main
git merge-base HEAD upstream/main
git log --oneline --left-right --cherry-pick upstream/main...HEAD
git submodule status --recursive
```

若 `upstream` 尚不存在，添加：

```bash
git remote add upstream https://github.com/facebookresearch/habitat-sim.git
git fetch upstream --tags
```

不要重写或压缩掉现有上游历史。

## 2.2 上游维护状态

Habitat-Sim README 明确说明：v0.3.4 之后不再由 Meta 内部团队主动维护，并允许社区继续独立 fork 和开发。这使 research fork 的路线合理，但也意味着 AVEngine 团队必须固定上游基线并自行维护构建、CI 和依赖兼容。

Codex 不应默认“最新 main 一定最好”。必须比较：

| 候选基线 | 优点 | 风险 | 决策要求 |
|---|---|---|---|
| 官方 v0.3.4 tag | 稳定、可引用 | 可能缺少 main 上后续修复 | 用 canary 验证所需 PBR、skinning、audio 能力 |
| 当前 fork main | 已经 fork，可能包含后续修复 | 上游 main 未必稳定 | 固定 exact commit，不按浮动 main 构建 |
| 自选 upstream commit | 可选中关键修复 | 维护成本最高 | 必须写 ADR 和 base SHA |

最终选择必须写入：

```text
UPSTREAM.md
runtime.lock.yaml
build manifest
每个数据样本的 provenance_manifest.json
```

## 2.3 许可证基础

- Habitat-Sim：MIT License。
- Habitat-Sim fork 中已有 RLR submodule。
- RLR-Audio-Propagation：CC BY-NC 4.0。
- SoundSpaces 代码仓库：CC BY 4.0。
- 本项目明确为非商业研究，因此 RLR 的非商业条件与当前用途相符；但公开发布时仍必须保留署名、许可证和修改说明。
- 各场景、动物、动作、音频和生成模型资产仍有各自许可证，不能因为主仓库开源就自动再分发。

---

# 3. 为什么迁移为 Habitat-native

## 3.1 旧跨引擎路径的问题

旧主路径大致为：

```text
Blender / 生成动物资产
        ↓
UE / SPEAR 执行视觉与角色
        ↓
从 UE 导出或近似声学场景
        ↓
Habitat-Sim AudioSensor / RLR 执行声学
        ↓
AVEngine 对齐两个运行时
```

这带来：

- UE 与 Habitat 的坐标、单位、旋转约定转换；
- 视觉场景 mesh 与声学 mesh 不一致；
- 当前 Apartment 声学代理使用 actor AABB，门洞、凹面和旋转几何容易失真；
- 多视角和声学分别推进状态，增加时间相位不一致风险；
- 相机、监听者、声源和动物姿态来自不同运行时；
- 材质 sidecar 与 stock AudioSensor 的 semantic-material 路径没有真正接通；
- 为论文解释“为什么不用全 Habitat”增加额外负担。

## 3.2 Habitat-native 的直接优势

统一使用 Habitat-Sim 视觉与声学运行时，可以让：

```text
房间、对象、动物、相机、监听者、声源和传感器
```

处于同一 scene graph 和同一世界坐标系。

主要收益：

- 直接使用 Habitat 支持的多房间数据集；
- 导入自定义 GLB 房间；
- 统一读取 camera、listener、source、object、agent 坐标；
- 同一帧一次 evaluate 动物 pose，再采集所有视觉传感器；
- 同一 runtime 管理房间和 source/listener 状态；
- 不再依赖 UE actor bounds 生成 AABB 声学房间；
- 更容易建立 geometry/material/timeline parity；
- 更容易形成公平的 Habitat/SoundSpaces 基线。

## 3.3 Habitat-native 不会自动替代的部分

Habitat-Sim 不负责：

- Pixal3D/TRELLIS guide 的静态 QA；
- 动物模板选择与 OOD rejection；
- 生成 mesh 到稳定模板的形态拟合；
- rig、skin weights 和动作质量；
- 动物 action bake；
- 跨腿权重、翻面、自交、脚滑等动画 QA；
- 声学材质语义和逐三角形绑定；
- 反事实 episode 关系；
- 48 kHz 权威时间轴；
- dataset manifest、provenance 和 admission；
- 数据集任务、split 和 benchmark。

因此迁移不是“删除 AVEngine”，而是：

> **让 Habitat-Sim 成为 AVEngine 的统一 runtime backend。**

---

# 4. 继承、扩展与原创边界

这是代码、README、论文和 rebuttal 中都必须统一的一张表。

| 能力 | 来源 | AVEngine 是否主张原创贡献 | 说明 |
|---|---|---:|---|
| 场景图与 GLB 场景加载 | Habitat-Sim | 否 | 直接复用并注明上游 |
| RGB、Depth、Semantic 传感器 | Habitat-Sim | 否 | 仅做 episode 调度和数据封装 |
| PBR 渲染、IBL、HBAO | Habitat-Sim | 否 | 可做视觉质量校准，但不能称为新 renderer |
| Bullet 物理和 navmesh | Habitat-Sim | 否 | 可选用，不作为核心贡献 |
| Articulated object / skinned rendering 基础 | Habitat-Sim | 否 | AVEngine 增加非人类资产执行层 |
| 几何声学传播 | RLR / SoundSpaces 2.0 | 否 | 不声称新声学算法 |
| RLR 现代多 source/listener C API | RLR | 否 | 底层已有；AVEngine 负责 Habitat adapter 与身份映射 |
| 非人类 baked skeletal pose runtime | AVEngine extension | 是，系统贡献 | 稳定动物 package、精确时间轴驱动、语义骨骼锚点 |
| 显式 AcousticScenePackage | AVEngine | 是 | 视觉场景到受控声学代理和逐三角形材料 |
| 多源身份、独立 stem 和事件对应 | AVEngine | 是 | 不只是多个坐标，而是 actor/event/asset lineage |
| Authoritative frame-sample timeline | AVEngine | 是 | 统一 pose、camera、source、frame、sample、event |
| 无嘴部动画 anti-shortcut 反事实数据 | AVEngine | 是 | 视觉不泄漏发声身份 |
| 动物模板银行与稳定资产编译 | AVEngine | 是 | 生成结果作为 guide，模板作为动画权威 |
| 动画/声学/同步/来源 QA | AVEngine | 是 | 决定数据是否可注册 |
| 下游数据集与任务 | AVEngine | 是 | 论文科学价值的主要承载体 |

---

# 5. 必须采用的仓库结构

## 5.1 必须保留的 fork 仓库

### 仓库 A：Habitat runtime fork

```text
Eastforward/habitat-sim-AVEngine
```

职责仅限于：

- Habitat-Sim C++/Python runtime 的必要修改；
- 现代 RLR C API adapter；
- 多 source/listener 和每 pair IR 暴露；
- 显式声学 mesh/material package ingestion；
- 非人类关节 pose 的确定性执行；
- 同一 world state 下的确定性多传感器 capture；
- runtime 层测试和示例；
- 与上游 Habitat-Sim 的兼容维护。

不得把以下内容大规模塞入这个 fork：

- Pixal/TRELLIS 推理；
- Blender 模板拟合；
- 数据集注册数据库；
- benchmark 模型训练代码；
- 大型资产；
- 论文数据生成策略；
- 与 Habitat runtime 无关的通用 AVEngine CLI。

原因：这个仓库必须保持“相对于 Habitat 上游的可理解 diff”。

## 5.2 必须新建的 AVEngine 主仓库

### 仓库 B：原创数据集引擎

建议创建：

```text
Eastforward/AVEngine
```

若 Codex 具备 GitHub CLI 且已经认证，可执行：

```bash
gh repo create Eastforward/AVEngine \
  --public \
  --description "A Habitat-native dataset engine for synchronized articulated audio-visual scenes" \
  --license mit \
  --clone
```

如果没有远程创建权限：

1. 在本地创建完整 scaffold；
2. 生成 `REPO_CREATION_COMMANDS.md`；
3. 不得假称远程仓库已创建。

主仓库职责：

```text
avengine/
  assets/             # asset package contracts、registry、template bank 接口
  scenes/             # room 与 acoustic scene compiler
  runtime/            # 对 habitat-sim-AVEngine 的高层 adapter，不复制其底层代码
  timeline/           # authoritative timeline builder/validator
  episodes/           # episode 与 counterfactual 编译
  audio/              # dry source、stem、mix、事件与 sample mapping
  qa/                 # 动画、声学、同步、manifest QA
  registry/           # asset/scene/sample admission
  schemas/            # JSON Schema 和版本迁移
  provenance/         # hash、版本、license、lineage
  cli/                # 稳定用户入口

tools/
  blender/            # 离线资产编译工具或其 wrappers
  migration/          # 从旧 AVEngine 迁移

docs/
  architecture/
  adr/
  migration/
  paper/
  licenses/

examples/
  minimal_room/
  dog_canary/
  two_source_counterfactual/

tests/
  unit/
  integration/
  canary/
```

这里的结构是职责边界，不要求 Codex 机械照搬每个目录名；若调整，必须在架构文档中说明。

## 5.3 可后续建立但当前非必须的仓库

### 仓库 C：AVEngine-Assets-Examples

仅在样例资产开始膨胀时建立：

```text
Eastforward/AVEngine-Assets-Examples
```

只放：

- 明确允许再分发的小型 CC0/自有资产；
- canonical animal package 示例；
- custom room 示例；
- manifest 和下载脚本；
- 不放无法再分发的 HM3D/MP3D 等完整数据。

### 仓库 D：AVEngine-Benchmark

论文任务稳定后再建立：

```text
Eastforward/AVEngine-Benchmark
```

包含：

- Dynamic Articulated Source Attribution 任务；
- 数据加载器；
- baseline；
- 指标；
- split manifests；
- 训练/评估脚本。

当前不要为了“仓库整齐”过早拆成四五个空仓库。第一阶段强制只有两个：runtime fork + AVEngine 主仓库。

---

# 6. 两个仓库之间如何依赖

## 6.1 不复制 Habitat 代码到 AVEngine 主仓库

AVEngine 主仓库不得 vendor 一份去掉 Git 历史的 Habitat 源码。

推荐方式：

- `habitat-sim-AVEngine` 独立安装；
- AVEngine 的 `runtime.lock.yaml` 固定 fork commit；
- Docker/build script 按 exact commit clone `--recursive`；
- 每次运行把实际 commit 写入 provenance。

建议 lock 文件：

```yaml
avengine_version: 0.1.0-alpha
habitat_runtime:
  repository: https://github.com/Eastforward/habitat-sim-AVEngine
  commit: <full-sha>
  upstream_repository: https://github.com/facebookresearch/habitat-sim
  upstream_base_commit: <full-sha>
  upstream_base_tag: <tag-or-null>
rlr:
  repository: https://github.com/facebookresearch/rlr-audio-propagation
  commit: <full-sha>
  license: CC-BY-NC-4.0
```

可选使用 submodule，但 Habitat 自身已有多个 nested submodules，若使用必须确保：

```bash
git clone --recursive
```

和 CI、Docker 文档都被完整验证。若 nested submodule 使用户体验过差，优先使用 lock file + bootstrap script。

## 6.2 Python 命名边界

- Habitat fork 继续保留上游 `habitat_sim` 包名，以维持兼容性。
- 不要把整个上游包重命名为 `avengine`。
- AVEngine 主仓库使用独立 `avengine` 包名。
- Habitat-specific 扩展应位于清楚、隔离的 namespace 中，避免大量 monkey patch。
- 用户主要运行 `avengine` CLI，不应要求理解 Habitat fork 内部目录。

## 6.3 版本策略

建议：

```text
AVEngine main repo:
  v0.1.0-alpha, v0.2.0, ...

Habitat fork:
  avengine-runtime-v0.1.0-alpha
  avengine-runtime-v0.2.0
```

每个数据样本必须记录：

```text
avengine_git_commit
habitat_sim_avengine_git_commit
habitat_sim_upstream_base_commit
rlr_git_commit
schema versions
asset revisions
scene revisions
```

---

# 7. 当前 fork 必须立即补齐的治理文件

`Eastforward/habitat-sim-AVEngine` 目前仍主要显示上游 README。Codex 的第一个文档型 PR 应加入：

```text
README.md
UPSTREAM.md
MODIFICATIONS.md
THIRD_PARTY_NOTICES.md
CITATION.cff
CITATIONS.bib
LICENSES/
  Habitat-Sim-MIT.txt
  RLR-CC-BY-NC-4.0.txt
  SoundSpaces-CC-BY-4.0.txt   # 仅在实际复制/分发 SoundSpaces 代码时
AVENGINE_RUNTIME_VERSION
runtime.lock.yaml
docs/avengine/
  ARCHITECTURE.md
  BUILD_AND_REPRODUCIBILITY.md
  AUDIO_EXTENSION.md
  ARTICULATED_ANIMAL_RUNTIME.md
```

## 7.1 `LICENSE` 处理

根目录现有 Habitat-Sim MIT `LICENSE` 必须保留，不要替换成只有 AVEngine 名称的新文本。

对于修改过的上游文件：

```cpp
// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Modifications Copyright (c) 2026 <Your Institution / Authors>.
// Modified for AVEngine: <brief description>.
```

新增原创文件可以使用：

```cpp
// Copyright (c) 2026 <Your Institution / Authors>.
// SPDX-License-Identifier: MIT
```

不要批量改写所有上游文件头。只对实质修改或新增文件做清晰标记。

## 7.2 `UPSTREAM.md` 必须包含

```text
upstream project name
upstream repository URL
fork repository URL
base release/tag
base commit
fork date
current AVEngine runtime branch
RLR submodule commit
是否保留完整 upstream history
如何同步 upstream
```

## 7.3 `MODIFICATIONS.md` 必须按模块列出

初始规划至少包括：

1. Audio runtime
   - 从 deprecated 单 source/listener C++ interface 迁移到现代 RLR C API；
   - 多命名 source/listener；
   - per-pair IR；
   - explicit object mesh/material upload；
   - deterministic reset/state handling。

2. Articulated animal runtime
   - baked non-human skeletal poses；
   - exact timeline tick evaluation；
   - semantic bone anchors；
   - contact state exposure。

3. Deterministic capture
   - one canonical state per frame；
   - multiple sensors captured without advancing official timeline。

4. Runtime manifest
   - build/version/commit reporting；
   - canonical pose hash support。

只能写已经实现的修改；未实现内容放在 ROADMAP，不得写进“Current modifications”。

## 7.4 README 开头建议文本

可使用：

```markdown
# habitat-sim-AVEngine

`habitat-sim-AVEngine` is an independent research fork of Habitat-Sim
used as the runtime foundation for AVEngine. It is not a simulator
implemented from scratch and is not affiliated with or endorsed by Meta.

We retain Habitat-Sim's scene, rendering, sensor, physics, navigation,
and articulated-object infrastructure. AVEngine-specific extensions are
limited to deterministic non-human articulated playback, explicit
acoustic-scene ingestion, modern multi-source RLR integration, and
runtime state export.

The original Habitat-Sim code and history are preserved under the MIT
License. RLR-Audio-Propagation remains under CC BY-NC 4.0. See
`UPSTREAM.md`, `MODIFICATIONS.md`, and `THIRD_PARTY_NOTICES.md`.
```

README 后续应保留或链接上游构建与引用信息，但不能继续让读者误以为这是未修改的官方 Habitat-Sim 仓库。

---

# 8. 系统总体架构

```text
                       Dataset / Episode Request
         species, morphotype, action, room, sources, cameras,
             listener, counterfactual controls, random seeds
                                  |
                                  v
                      AVEngine Dataset Compiler
                                  |
           +----------------------+----------------------+
           |                                             |
           v                                             v
 Offline Animal Asset Compiler                  Room / Scene Compiler
 Blender + deterministic tools                  Habitat/custom room assets
 template bank                                  visual scene package
 fitting / UV bake                             acoustic proxy
 rig / action bake                             triangle material IDs
 animation QA                                  navmesh / semantic mapping
           |                                             |
           +----------------------+----------------------+
                                  |
                                  v
                       Canonical Episode Package
              assets + scene + timeline + source identity
                                  |
                                  v
                  habitat-sim-AVEngine Runtime Fork
          scene graph + PBR sensors + articulated poses + RLR
                                  |
                                  v
                    AVEngine Sample Assembly / QA
        frames + depth + semantics + RIRs + stems + mixture + labels
                                  |
                                  v
                         Dataset Registry / Admission
```

---

# 9. 核心资产和数据契约

Codex 必须把“文件夹约定”升级为版本化 schema 和 manifest。以下名称可调整，但语义必须保留。

## 9.1 Canonical Animal Asset Package

```text
animal_asset/
  asset_manifest.json
  provenance_manifest.json
  visual.glb
  skeleton.json
  skinning_manifest.json
  emitter_anchors.json
  collision_proxy.glb
  actions/
    idle.npz
    walk.npz
    turn_left.npz
    action_manifest.json
  contacts/
    contact_phases.json
  textures/
  qa/
    static_geometry.json
    deformation.json
    animation.json
```

必须记录：

| 字段 | 意义 |
|---|---|
| `asset_id` | 不变身份 |
| `template_id` | 使用的 canonical template |
| `morphotype_id` | 体型类别 |
| `topology_hash` | 固定 topology 版本 |
| `uv_hash` | 固定 UV 版本 |
| `skeleton_revision` | 骨架语义版本 |
| `weights_revision` | 权重版本 |
| `action_family` | 可用动作族 |
| `emitter_anchors` | head/muzzle/paw/body 等锚点 |
| `source_generator` | guide provider 和版本 |
| `license/provenance` | 来源与允许用途 |
| `qa_status` | 是否允许进入 runtime |

正式路线继续采用此前技术审查的原则：

> **模板负责 topology、UV、骨架、权重、碰撞和动作；Pixal3D、TRELLIS.2 等只提供形状和 PBR 外观 guide。**

任意未知生成拓扑直接套动画只能保留为实验路线，不得自动注册成 production asset。

## 9.2 Room Package

```text
room_package/
  room_manifest.json
  provenance_manifest.json
  visual/
    scene.glb
    scene_dataset_config.json
    semantic_descriptor.json      # 若有
    navmesh.navmesh               # 若有
  acoustic/
    vertices.bin or .npy
    triangles.bin or .npy
    triangle_material_ids.bin or .npy
    material_categories.json
    material_database.json
    acoustic_proxy.glb             # debug/preview
  qa/
    geometry_report.json
    material_coverage.json
    ray_leakage.json
    visual_acoustic_parity.json
```

### 必须明确的背景知识

Habitat 会从 GLB/glTF 读取视觉材质，例如：

```text
base color
normal
roughness
metallic
textures
```

但它不会自动从木纹、玻璃外观或 roughness 推导可靠的：

```text
absorption
scattering
transmission
```

声学材质必须显式提供。

### 禁止 AABB 作为正式房间代理

AABB = Axis-Aligned Bounding Box，轴对齐包围盒。它只保留物体在世界 X/Y/Z 方向上的最小/最大范围，并把对象替换成一个矩形盒。它会：

- 填平门洞和凹面；
- 把 L 形或旋转物体扩成大盒；
- 产生不存在的反射面和绕射边；
- 改变房间体积与表面积；
- 造成重叠/内部面；
- 使视觉和声学几何不一致。

AABB 只允许做快速 debug，不得作为 production acoustic proxy。

正式 room compiler 应从：

- Habitat/GLB 原始场景；
- Blender 自定义房间；
- collision mesh；
- 受控简化后的真实 surface；

生成保留门洞、窗洞、连通性和主要家具体积的 acoustic proxy。

## 9.3 Acoustic Scene Package

该 package 必须能直接表达：

```text
vertices
triangles
triangle -> material category
material category -> RLR material parameters
object transforms
unit scale
coordinate convention
geometry hash
source scene revision
```

RLR 现代 C API 已支持：

- `RLRA_AddSource`
- `RLRA_AddListener`
- `RLRA_AddObject`
- `RLRA_AddMeshVertices`
- 多次 `RLRA_AddMeshIndices(..., materialCategoryName)`
- `RLRA_FinalizeObjectMesh`
- `RLRA_Simulate`
- 按 listener/source pair 读取 IR

因此 AVEngine 应修改 Habitat adapter，使其消费显式 AcousticScenePackage，而不是依赖 stock AudioSensor 的“semantic scene 存在时才加载 materials”隐式路径。

## 9.4 Authoritative Timeline

服务器中的：

```text
avengine_timeline_v2.schema.json
```

是当前基线契约。关键不变量：

```text
time base        = 48,000 Hz
duration         = 240,000 ticks
video            = 15 fps, 75 frames
frame duration   = 3,200 ticks
audio            = 16 kHz, 80,000 samples
sample duration  = 3 ticks
total duration   = exactly 5 seconds
```

音频样本边界：

```text
sample_start(f) = round(f       * 16000 / 15)
sample_end(f)   = round((f + 1) * 16000 / 15)
```

不得固定使用 1,067 samples/frame。

### 无嘴部动画和 timeline v2

当前项目明确**不生成嘴部动画**，目的是防止视觉模型从嘴巴开合作弊。

v2 schema 仍包含：

```text
mouth_state.open_ratio
mouth_state.vocalizing
```

处理原则：

- v2 不得静默修改；
- 在 v2 兼容模式中，`open_ratio` 固定为 `0.0`；
- `vocalizing` 仅代表音频事件激活，不代表视觉 mouth articulation；
- episode manifest 增加：

```json
{
  "visual_vocal_articulation": {
    "mode": "disabled_for_shortcut_control",
    "mouth_motion_present": false
  }
}
```

如需 timeline v3，必须：

- 写 ADR；
- 新建 schema 文件；
- 提供 v2 -> v3 migration；
- 不覆盖原 v2。

## 9.5 Episode Package

```text
run_dir/
  request.json
  episode_manifest.json
  runtime_manifest.json
  scene_manifest.json
  asset_manifests/
  provenance_manifest.json
  timeline.json
  render/
    rgb/
    depth/
    semantic/
    camera_calibration.json
  audio/
    authoritative.wav
    dry_sources/
    rir/
    stems/
    mixture.wav
    audio_events.json
  labels/
    actor_tracks.json
    source_attribution.json
    bone_anchors.json
  qa/
    qa_report.json
    mux_verification.json
  logs/
```

---

# 10. 主要功能模块

## 10.1 Offline Animal Asset Compiler

继续保留 Blender 离线阶段，而不是在 Habitat 中重做完整 DCC 工具链。

目标流程：

```text
reference / attributes
  -> guide provider
  -> static geometry QA
  -> body-plan / morphotype classification
  -> audited template selection
  -> constrained template fitting
  -> PBR transfer to fixed UV
  -> native skeleton/weights/action family
  -> action baking to exact pose arrays
  -> deformation/contact QA
  -> canonical animal package
```

Habitat runtime 只负责确定性播放已经烘焙并审核的 pose，不承担任意 mesh 自动 rigging 的责任。

## 10.2 Habitat Articulated Animal Runtime

最小范围：

- 加载 canonical animal visual asset；
- 加载骨架映射和 baked action poses；
- 按 timeline tick 设置 root transform 和 joint pose；
- 暴露 head/muzzle/paw/body 语义锚点；
- 暴露 foot contact state；
- 同一帧姿态 evaluate 一次；
- 所有视觉 sensors 在不推进正式时间轴的情况下读取；
- 生成 canonical pose hash。

不需要实现：

- 通用 FBX animation blueprint；
- 在线 retarget；
- jaw/lip/viseme；
- 影视级面部动画。

## 10.3 Room / Acoustic Scene Compiler

支持三类输入：

| 输入类型 | 视觉场景 | 声学材料来源 | 主要用途 |
|---|---|---|---|
| Habitat 原生场景 | 已有 dataset scene | semantic category -> material proposal | 大规模房间库 |
| Blender 自定义房间 | 自己导出 GLB | 明确 acoustic material slots / sidecar | 受控实验 |
| 旧 UE 房间迁移 | 导出真实 mesh/collision mesh | UE tags/material mapping | 兼容旧资产，禁止 AABB production |

注意：semantic category 到 acoustic material 仍然是近似映射。例如 `table` 不足以证明是木材还是金属。必须记录：

```text
mapping source
mapping confidence
是否人工指定
是否随机化
是否使用 fallback material
```

## 10.4 Multi-Source RLR Runtime

Stock Habitat AudioSensor 当前一次维护一个 source，且使用 RLR deprecated C++ interface。RLR 现代 C API 已支持多个 source/listener。

目标不是“发明多声源 RLR”，而是：

- 在 Habitat fork 中接入现代 C API；
- 管理稳定 `source_id` 和 `listener_id`；
- 一个 context 中更新多个 source/listener；
- 一次 simulation 生成所有 pair IR；
- 输出每个 source 的独立 RIR 和 stem；
- 将 source 与 actor、event、semantic anchor、dry audio 对齐；
- 明确 reset、temporal coherence 和随机性策略。

初始论文范围应是：

> **多个动态语义点声源在静态或准静态 acoustic geometry 中传播。**

不要提前声称：

- 可变形动物身体逐帧参与声学反射；
- fully dynamic acoustic geometry；
- 新的 acoustic propagation solver。

## 10.5 Anti-Shortcut Counterfactual Episode

项目有意不做嘴部动画。

必须支持视觉完全相同的反事实对：

```text
Episode A:
  same room, same cameras, same animal poses
  dog_1 vocalizes, dog_2 silent

Episode B:
  pixel-identical or provably identical visual observations
  dog_1 silent, dog_2 vocalizes
```

还可生成：

- 同一视觉，只交换发声 actor；
- 同一视觉，只改变声学材质；
- 同一 actor/pose，只改变 emitter anchor；
- 同一 mixture，只改变 source identity label；
- 正确同步 vs 受控时间偏移。

每组 counterfactual 必须有：

```text
pair/group id
frozen variables
changed variables
visual hash relationship
timeline relationship
source lineage
QA proof
```

## 10.6 Registry、QA 和 Dataset Admission

最终状态至少包括：

```text
approved_for_dataset
rejected_static_geometry
rejected_template_ood
rejected_asset_animation
rejected_scene_geometry
rejected_acoustic_materials
rejected_runtime
rejected_av_sync
rejected_provenance
research_candidate_pending_human_visual_review
incomplete_environment_not_run
```

不得因“成功生成了文件”就批准样本。

---

# 11. 从旧 AVEngine 到新架构的迁移矩阵

Codex 必须先对旧仓库生成实际清单，再按以下分类。

| 旧模块 | 新位置 | 决策 | 说明 |
|---|---|---|---|
| 模板拟合、rig、权重和 Blender QA | AVEngine 主仓库 `tools/blender` / asset compiler | 保留并重构 | 仍是核心原创能力 |
| 生成 mesh 直接换 Quaternius rig | experimental legacy | 降级 | 不得作为默认 production route |
| UE/SPEAR 动物视觉 runtime | optional backend / legacy | 暂时保留 | 作为视觉质量上限或对照，不再是论文主 runtime |
| UE view-outer render loop | legacy | 停止作为主路径 | Habitat-native 同帧多 sensor 取代 |
| gpuRIR shoebox | optional acoustic backend | 保留研究对照 | 不作为 Habitat-native 主声学后端 |
| 当前 Habitat single-source AudioSensor scripts | habitat fork + AVEngine adapter | 替换 | 迁移到现代 RLR C API |
| AABB apartment mesh | debug only | 禁止 production | 用真实/受控简化 acoustic proxy 替代 |
| material sidecar | AcousticScenePackage | 重构 | 显式上传 per-triangle material category |
| timeline v2 | AVEngine 主仓库 schemas/timeline | 保留为基线 | 不静默修改，必要时新增 v3 |
| old run manifests | AVEngine provenance/registry | 迁移 | 提供 schema 和 migration |
| old tests | 分类后迁移 | 保留有效测试 | 静态 source-code tests 不能冒充 E2E |

必须生成：

```text
docs/migration/LEGACY_AVENGINE_INVENTORY.md
docs/migration/MIGRATION_MATRIX.md
docs/migration/DEPRECATION_PLAN.md
```

每个旧入口标明：

```text
owner
current callers
input/output contract
new replacement
compatibility adapter
removal milestone
```

---

# 12. 分阶段路线图

## Phase 0：仓库、上游和构建基线

### 目标

先建立可信的 fork 和两个仓库边界。

### 交付

- 核实 fork exact base SHA 和 submodule SHAs；
- 完成 upstream remote；
- 当前 build/test baseline；
- `UPSTREAM.md`；
- `MODIFICATIONS.md`；
- `THIRD_PARTY_NOTICES.md`；
- `CITATION.cff` / `CITATIONS.bib`；
- 新建或 scaffold `Eastforward/AVEngine`；
- `runtime.lock.yaml`；
- 总体架构 ADR。

### 退出条件

- 能从干净环境按文档构建当前 fork；
- 原始 Habitat tests 的状态被记录；
- 未运行的 audio/GPU tests 明确为 `not_run`；
- 没有删除上游许可证或历史。

## Phase 1：Habitat Visual/Room Canary

### 目标

验证 Habitat 是否能满足论文的最低视觉与房间要求。

### 三个房间

1. Habitat 原生房间；
2. Blender 自定义房间；
3. 从旧 UE Apartment 导出的真实 mesh/collision proxy，禁止 AABB。

### 交付

- RGB、Depth、Semantic；
- 多 camera；
- camera/listener/source/object poses；
- scene dataset config；
- visual quality samples；
- room manifest。

### 退出条件

- 三种房间至少各一个可重复加载；
- 自定义房间门洞、窗洞和连通性正确；
- 坐标和单位明确；
- Habitat 视觉质量达到任务最低标准，或明确记录仍需 UE optional backend。

## Phase 2：Dog Articulated Runtime Canary

### 目标

一个 audited dog template 在 Habitat 中播放 baked Walk/Idle，不再 T-pose 滑动。

### 交付

- canonical dog asset package；
- exact per-frame joint poses；
- root trajectory；
- head/muzzle/paw anchors；
- foot contacts；
- pose hash；
- 四视角同帧一致性。

### 退出条件

- 75 帧精确姿态执行；
- 同帧所有 views 的 pose hash 相同；
- 没有自由运行的动画时钟；
- 无嘴部 animation；
- 资产通过 deformation/contact QA。

## Phase 3：Acoustic Scene + Material Canary

### 目标

显式声学 mesh 和逐三角形材质真正进入 RLR。

### 交付

- AcousticScenePackage schema；
- adapter ingestion；
- material coverage report；
- `RLRA_WriteSceneMeshOBJ` debug export；
- low-absorption vs high-absorption extreme canary；
- geometry leakage / ray-hit QA。

### 退出条件

- 所有 production triangles 都有 material category；
- 无意外 default material；
- 极端材料产生远大于随机方差的 RIR/EDT/DRR 差异；
- 不再使用 AABB production proxy。

## Phase 4：Multi-Source / Multi-Listener RLR Canary

### 目标

基于现代 RLR C API，在一个 context 中处理多个命名 source/listener。

### 交付

- named source/listener API；
- per-pair IR；
- per-source stems；
- source order invariance test；
- temporal coherence policy；
- multi-source performance report。

### 退出条件

- 两只狗、至少两个 source；
- 每个 source-listener pair 独立可读；
- 交换 source 注册/处理顺序不产生系统性变化；
- source identity 与 actor/event/anchor 一一对应。

## Phase 5：Authoritative Timeline + Anti-Shortcut Pair

### 目标

完成严格 5 秒 episode 和视觉相同的发声角色交换反事实对。

### 交付

- timeline v2 validator；
- 75 frame / 80,000 sample exact mapping；
- fixed visual state capture；
- two-source counterfactual pair；
- visual hash proof；
- audio event and stem manifests。

### 退出条件

- 视频严格 75 帧；
- WAV 严格 80,000 samples；
- 视觉反事实对逐帧 hash 相同；
- 只有 audio/source labels 发生声明内变化；
- no mouth motion 明确记录。

## Phase 6：Dataset Pipeline / Registry / QA

### 目标

从 request 到 approved/rejected 样本形成闭环。

### 交付

- CLI；
- asset/scene/episode registries；
- full manifests；
- QA aggregator；
- deterministic rerun；
- sample admission。

### 退出条件

- 一个 Dog MVP 端到端 canary；
- 同 seed 重跑 manifest/timeline 可复现；
- 失败能结构化拒绝；
- `not_run` 不会变成 `pass`。

## Phase 7：Benchmark 和论文

### 推荐任务

```text
Dynamic Articulated Source Attribution
```

模型预测：

- 哪个 actor 发声；
- event temporal interval；
- 3D source location；
- event type：vocalization / footstep；
- emitting anchor：head/muzzle/paw；
- action state。

### 基线

- visual-only；
- audio-only；
- audio-visual；
- root point source；
- semantic bone anchor；
- 同步数据 vs controlled desynchronization；
- 无反事实数据 vs AVEngine counterfactual data。

---

# 13. QA 总表

| QA 类别 | 核心检查 | Production hard fail 示例 |
|---|---|---|
| 动物静态几何 | components、nonmanifold、缺肢、桥连 | 严重缺失、腿间桥、地面融合 |
| 绑定与骨架 | joint volume、cross-limb weight、weight sum | 左腿受右腿骨影响、关节在错误部位 |
| 动画 | stretch/compression、flip、自交、joint limits | 翻面、穿腿、关节超限 |
| 接触 | paw-ground、sliding、gait | 接触期明显滑动、穿地 |
| 房间几何 | 门洞、连通性、ray leakage、normals | 封死门洞、大规模漏射线 |
| 声学材质 | coverage、unmatched category、fallback | production triangle 未赋材料、意外默认材料 |
| 多源 | identity、pair IR、order invariance | source stem 因注册顺序变化 |
| 视觉同步 | multi-view pose hash | 同帧不同 camera 姿态不同 |
| AV 同步 | frame/sample boundary、event onset | 75 帧/80k samples 不一致 |
| 反事实 | frozen-variable proof | 视觉在只应变 audio 时发生变化 |
| Provenance | git commits、assets、licenses | 缺来源/hash/revision |
| Runtime | build、crash、determinism | 相同输入无法复现 |

每项输出：

```text
status: pass / fail / warning / not_run
threshold
measured value
worst-case location/frame/source
artifact path
failure reason
```

---

# 14. 风险登记表

| 风险 | 严重度 | 发生概率 | 缓解方式 |
|---|---:|---:|---|
| Habitat 动物视觉质量低于 UE | 高 | 中 | Phase 1 视觉 canary；保留 UE optional backend，不阻塞架构层 |
| 狗 skinned pose runtime 难以接入 | 高 | 中 | 采用离线 baked joint poses，避免运行时通用 retarget |
| RLR C API 与 Habitat 构建/ABI 冲突 | 高 | 中 | 固定 submodule commit、独立 C++ adapter tests、容器化构建 |
| 声学材质仍未真正生效 | 高 | 中 | 极端材料 canary + debug scene export + coverage hard gate |
| 房间 mesh 有洞导致 ray leakage | 高 | 中 | acoustic geometry QA、ray efficiency、人工 canary |
| 多 source 计算成本线性增长 | 中 | 高 | 现代 C API、共享 scene/context、质量配置和 profiling |
| temporal coherence 引入状态污染 | 高 | 中 | source 状态策略、order invariance、reset tests |
| Timeline schema 与无嘴动画冲突 | 中 | 高 | v2 兼容元数据；v3 必须 ADR 和 migration |
| 旧 AVEngine 代码迁移造成双路线混乱 | 高 | 高 | migration matrix、deprecation plan、单一主 CLI |
| 论文被认为只是 Habitat wrapper | 高 | 中 | 明确原创数据契约、反事实任务、QA、下游实验 |
| 论文过度声称新声学算法 | 高 | 低 | inherited/extended/original 表格贯穿全文 |
| 上游不再维护 | 中 | 高 | pin exact base、CI、Docker、fork release tags |
| 场景/资产不可再分发 | 高 | 中 | 下载脚本和 manifest，样例仅用可再分发资产 |

---

# 15. Codex 首轮必须产出的文档

在大规模实现前，至少创建以下文件。

## Habitat fork

```text
README.md
UPSTREAM.md
MODIFICATIONS.md
THIRD_PARTY_NOTICES.md
CITATION.cff
CITATIONS.bib
runtime.lock.yaml
AVENGINE_RUNTIME_VERSION
docs/avengine/ARCHITECTURE.md
docs/avengine/BUILD_AND_REPRODUCIBILITY.md
docs/avengine/AUDIO_EXTENSION_PLAN.md
docs/avengine/ARTICULATED_ANIMAL_RUNTIME_PLAN.md
```

## AVEngine 主仓库

```text
README.md
LICENSE
THIRD_PARTY_NOTICES.md
CITATION.cff
CITATIONS.bib
runtime.lock.yaml
schemas/avengine_timeline_v2.schema.json

docs/architecture/SYSTEM_OVERVIEW.md
docs/architecture/REPOSITORY_BOUNDARIES.md
docs/architecture/ASSET_PACKAGE.md
docs/architecture/ROOM_AND_ACOUSTIC_SCENE_PACKAGE.md
docs/architecture/EPISODE_AND_TIMELINE.md
docs/architecture/QA_AND_REGISTRY.md

docs/migration/LEGACY_AVENGINE_INVENTORY.md
docs/migration/MIGRATION_MATRIX.md
docs/migration/DEPRECATION_PLAN.md

docs/paper/PAPER_POSITIONING.md
docs/paper/CLAIMS_AND_NON_CLAIMS.md
docs/paper/CITATION_AND_ATTRIBUTION.md

docs/roadmap/MILESTONES.md
docs/roadmap/ISSUE_BACKLOG.md
```

## ADRs

至少：

```text
ADR-0001-habitat-native-primary-runtime.md
ADR-0002-two-repository-boundary.md
ADR-0003-explicit-acoustic-scene-package.md
ADR-0004-authoritative-integer-timeline.md
ADR-0005-disable-visual-mouth-articulation.md
ADR-0006-template-authoritative-animal-assets.md
ADR-0007-modern-rlr-c-api.md
ADR-0008-static-acoustic-geometry-scope.md
```

每个 ADR 包含：

```text
Context
Decision
Alternatives considered
Consequences
Validation plan
Reversal criteria
```

---

# 16. GitHub Issues 与 Milestones 规划

Codex 应将路线图转换为可执行 backlog。每个 issue 必须有：

```text
problem statement
scope
non-goals
dependencies
deliverables
acceptance tests
not_run conditions
documentation updates
```

建议 milestones：

```text
M0 Repository and Baseline
M1 Habitat Visual and Room Canary
M2 Articulated Dog Runtime
M3 Acoustic Scene and Materials
M4 Multi-Source RLR
M5 Timeline and Counterfactual Episode
M6 Dataset MVP
M7 Benchmark and Paper Release
```

不要创建一个“Implement AVEngine”巨型 issue。

---

# 17. 开源归属与引用规则

## 17.1 应如何描述与 Habitat-Sim 的关系

不要写：

```text
We took over Habitat-Sim.
We borrowed Habitat code.
We built a new simulator from scratch.
```

推荐：

```text
AVEngine is built on a pinned research fork of Habitat-Sim.
AVEngine extends Habitat-Sim for synchronized articulated audio-visual dataset generation.
habitat-sim-AVEngine is an independent research fork of Habitat-Sim.
```

中文版：

> **AVEngine 建立在固定版本的 Habitat-Sim research fork 之上，并扩展其非人类关节资产执行、显式声学场景、多声源 RLR 接口和确定性数据采集能力。**

## 17.2 代码仓库 attribution

Habitat fork：

- 保留 GitHub fork 关系；
- 保留完整上游历史；
- 保留根 MIT LICENSE；
- 保留上游版权文件头；
- 明确列出修改；
- 不暗示 Meta endorsement。

RLR：

- 保留 CC BY-NC 4.0；
- 记录 submodule commit；
- 标明是否修改；
- 若修改，说明修改位置和性质；
- 项目明确标注 non-commercial research use。

SoundSpaces：

- 若只复用论文思想和 Habitat/RLR 路径，引用论文；
- 若复制/改写 SoundSpaces 代码，保留 CC BY 4.0 attribution 和修改说明；
- 不应在没有复制代码时无差别把整套 SoundSpaces LICENSE 当作 AVEngine 自有代码许可证。

## 17.3 AVEngine 主仓库许可证

原创 AVEngine 代码建议 MIT，与 Habitat fork 保持简单兼容。

根 README 必须写：

```markdown
Unless otherwise noted, original AVEngine code is released under the MIT
License. The separately maintained Habitat-Sim runtime fork retains the
upstream Habitat-Sim MIT License and copyright notices. RLR-Audio-
Propagation remains under CC BY-NC 4.0. Third-party datasets and assets
are governed by their respective licenses.
```

不要写：

```text
All code and dependencies are MIT licensed.
```

## 17.4 论文引用

使用 Habitat 平台时，按其官方 README 至少引用：

1. Habitat 1.0：*Habitat: A Platform for Embodied AI Research*, ICCV 2019。
2. Habitat 2.0：*Habitat 2.0: Training Home Assistants to Rearrange their Habitat*, NeurIPS 2021。
3. Habitat 3.0：*Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots*, 2023。
4. SoundSpaces 2.0：*SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning*, NeurIPS 2022 Datasets and Benchmarks。
5. 若沿用 SoundSpaces 1.0 的任务代码、数据或接口，额外引用 SoundSpaces 1.0。
6. 所有实际使用的场景数据集分别引用，例如 HM3D、HSSD、Replica、ReplicaCAD、Matterport3D。
7. 动物模板、动作和音频资产根据各自来源引用。

## 17.5 `CITATION.cff`

Habitat fork 的 `CITATION.cff` 应：

- 说明 fork 名称和 AVEngine runtime 版本；
- 指向 AVEngine 项目/论文；
- 在 references 或 README 中要求同时引用 Habitat 1/2/3 和 SoundSpaces 2.0；
- 不能把 Meta 原作者列成 AVEngine fork 作者；
- 不能删除上游引用要求。

AVEngine 主仓库的 `CITATION.cff` 则以 AVEngine 作者为主，并在 preferred citation 和 README 中列出依赖论文。

---

# 18. 论文中如何描述

## 18.1 Methods 建议文本

### 英文

> **Runtime foundation.** AVEngine is implemented as an independent research extension to Habitat-Sim rather than as a simulator built from scratch. We reuse Habitat-Sim's scene representation, PBR rendering, configurable sensors, navigation, physics, and articulated-object infrastructure. Geometric acoustic propagation is provided by RLR-Audio-Propagation following SoundSpaces 2.0. We extend the runtime with deterministic non-human articulated playback, explicit acoustic-scene ingestion with per-triangle material assignments, and a modern multi-source RLR adapter exposing source-listener-specific impulse responses. On top of the runtime, AVEngine contributes audited asset compilation, an authoritative frame-sample timeline, identity-preserving source/event annotations, counterfactual episode generation, quality control, provenance, and dataset admission. We do not claim a new visual renderer or acoustic propagation algorithm.

### 中文

> **运行时基础。** AVEngine 并非从零实现新的模拟器，而是作为 Habitat-Sim 的独立研究扩展实现。系统复用 Habitat-Sim 的场景表示、PBR 渲染、可配置传感器、导航、物理和关节对象基础设施，并依据 SoundSpaces 2.0 使用 RLR-Audio-Propagation 进行几何声学传播。在此基础上，我们扩展了非人类关节资产的确定性播放、具有逐三角形声学材质的显式声学场景加载，以及能够输出各 source-listener pair 独立脉冲响应的现代多声源 RLR adapter。AVEngine 在运行时之上进一步提供经过审计的资产编译、权威帧—采样时间轴、保持身份的声源/事件标签、反事实 episode、质量控制、来源追踪和数据集准入。我们不将视觉渲染器或声学传播算法本身作为新的贡献。

## 18.2 推荐贡献列表

1. **Audited articulated animal asset pipeline**  
   将生成 guide 编译为固定模板 topology/UV/skeleton/weights/action 的稳定非人类资产。

2. **Explicit acoustic scene compiler**  
   将 Habitat 原生或自定义场景编译为保留房间拓扑和显式逐三角形材质的 AcousticScenePackage。

3. **Identity-preserving multi-source runtime integration**  
   基于 RLR 已有现代 C API，在 Habitat 中维护 source/listener identity、per-pair IR、per-source stems 和 actor/event/anchor 对应。

4. **Authoritative audio-visual timeline**  
   用整数 tick 统一骨骼姿态、相机、声源、视频帧、音频 sample 和事件。

5. **Anti-shortcut counterfactual dataset generation**  
   有意关闭视觉嘴部动作，生成视觉不变但发声身份变化的反事实样本。

6. **QA-gated dataset and benchmark**  
   提供可验证数据、来源和新的动态关节声源归属任务。

## 18.3 不能主张的内容

- 新的视觉 renderer；
- 新的声学 ray tracer；
- 首次实现 RLR 多 source；
- Habitat 原本不能动画；
- SoundSpaces 不支持连续位置或可配置材质；
- 视觉材质自动变成声学真值；
- 动物身体完整参与动态声学反射；
- lip sync 或 mouth animation；
- 任意生成 mesh 都可稳定动画；
- AVEngine 从零构建 simulator。

## 18.4 推荐论文标题

首选：

> **AVEngine: A Habitat-Native Dataset Engine for Counterfactual Articulated Audio-Visual Source Grounding**

备选：

> **AVEngine: Extending Habitat-Sim for Synchronized Multi-Source Articulated Audio-Visual Data Generation**

> **AVEngine: Identity-Preserving Dynamic Audio-Visual Data Generation on Habitat-Sim**

---

# 19. 首个可发表 MVP 的 Definition of Done

只有同时满足以下条件，才可以把 Habitat-native 写成论文主实现：

1. Fork 的 upstream base、修改范围和许可证完整记录。
2. `Eastforward/AVEngine` 主仓库创建并与 fork 职责分离。
3. 至少一个真实 Habitat 场景和一个 Blender 自定义场景可运行。
4. 房间声学 proxy 不再采用 AABB production 路线。
5. 声学材质通过显式逐三角形 assignment 进入 RLR，并通过极端材质 canary。
6. 一个 audited dog template 在 Habitat 中精确播放 Walk/Idle。
7. 四个视觉 sensors 同帧共享同一 pose hash。
8. 至少两个命名 source 在现代 RLR context 中获得独立 per-pair IR/stems。
9. source identity 与 actor、audio event、head/muzzle/paw anchor 对齐。
10. timeline 严格满足 75 frames、80,000 samples、240,000 ticks。
11. 无嘴部视觉 articulation 被显式记录。
12. 生成视觉相同、只交换发声角色的 counterfactual pair。
13. 每个样本具备 scene、asset、runtime、timeline、audio、QA 和 provenance manifest。
14. 失败样本会结构化拒绝。
15. 所有未实际运行的 Blender/GPU/RLR E2E 测试标为 `not_run`。
16. 论文和 README 明确区分 reused、extended、original。
17. Habitat 1/2/3、SoundSpaces 2.0 和场景数据集引用完整。

---

# 20. Codex 首轮工作顺序

严格按此顺序：

## Step 1：Audit

- 阅读三个 AVEngine 文档；
- 审计当前 fork exact Git 状态；
- 审计旧 AVEngine 实际代码；
- 审计 RLR submodule 的实际 API 和 commit；
- 审计构建环境；
- 审计当前资产和场景来源。

## Step 2：Write architecture before broad implementation

先创建：

- repository boundaries；
- system overview；
- ADRs；
- migration matrix；
- milestones/issues；
- attribution files；
- runtime lock。

## Step 3：Establish reproducible baselines

- fork clean build；
- Habitat visual sample；
- current audio baseline；
- current animal baseline；
- 测试状态表。

## Step 4：Implement only the smallest canaries

顺序：

```text
room/visual
-> dog baked pose
-> acoustic package/material activation
-> multi-source
-> timeline/counterfactual
-> registry
```

不要同时大改所有模块。

## Step 5：Report honestly

每个阶段总结：

```text
implemented
verified
failed
not_run
remaining risks
exact artifacts
exact commits
```

---

# 21. Codex 最终输出格式

首轮完成时必须输出：

## 21.1 Repository status

```text
fork HEAD
upstream base
submodule commits
build environment
branch/tag
```

## 21.2 Files created/modified

按两个仓库分别列出。

## 21.3 Architecture decisions

摘要 ADR 和原因。

## 21.4 Migration matrix

旧 AVEngine 每个主要模块：keep / migrate / optional / retire / experimental。

## 21.5 Roadmap

Milestone、Issue、依赖和 exit criteria。

## 21.6 Verification table

```text
unit tests
Habitat build
visual canary
animal canary
acoustic material canary
multi-source canary
timeline canary
E2E dataset canary
```

状态只能是：

```text
pass
fail
not_run
blocked
```

## 21.7 Attribution and paper wording

确认：

- upstream license preserved；
- RLR notice present；
- README relationship section；
- paper claims/non-claims；
- citations list。

---

# 22. 首轮规划任务完成标准

这份“重新规划”只有在以下条件满足时才算完成：

- 不再把 AVEngine 与 Habitat-Sim 设定成竞争关系；
- 明确 Habitat runtime fork 与 AVEngine main repo 的职责；
- 确定 exact upstream base 和依赖版本；
- 建立 repository governance 和 attribution；
- 定义 Animal、Room、AcousticScene、Timeline、Episode 的版本化契约；
- 旧 AVEngine 代码有完整 migration matrix；
- Habitat-native 的可行性由逐阶段 canary 判断，而不是口头假设；
- 多 source 的贡献边界准确：RLR 提供能力，AVEngine 提供 Habitat 集成、身份、事件和数据契约；
- 无嘴部动画被写成 anti-shortcut 实验设计，而不是缺失功能；
- AABB 从 production 路线移除；
- 声学材料显式绑定，不依赖视觉材质自动推断；
- 论文清楚写出 reused / extended / original；
- README、许可证、第三方 notice 和引用规则完整；
- 最终路线能收敛到一个可复现的 Dog + custom room + two-source counterfactual MVP。

---

# 23. 已核实的上游来源，供 Codex 和文档引用

- AVEngine Habitat fork：  
  `https://github.com/Eastforward/habitat-sim-AVEngine`

- Habitat-Sim upstream：  
  `https://github.com/facebookresearch/habitat-sim`

- Habitat-Sim MIT License：  
  `https://github.com/facebookresearch/habitat-sim/blob/main/LICENSE`

- Habitat-Sim audio documentation：  
  `https://github.com/facebookresearch/habitat-sim/blob/main/docs/AUDIO.md`

- RLR-Audio-Propagation：  
  `https://github.com/facebookresearch/rlr-audio-propagation`

- RLR modern C API header：  
  `https://github.com/facebookresearch/rlr-audio-propagation/blob/main/RLRAudioPropagationPkg/headers/RLRAudioPropagation.h`

- RLR CC BY-NC 4.0 License：  
  `https://github.com/facebookresearch/rlr-audio-propagation/blob/main/LICENSE`

- SoundSpaces：  
  `https://github.com/facebookresearch/sound-spaces`

- SoundSpaces 2.0 API/background：  
  `https://github.com/facebookresearch/sound-spaces/blob/main/SoundSpaces2.md`

- SoundSpaces 2.0 paper：  
  `https://arxiv.org/abs/2206.08312`

---

# 24. 最终总目标

> **将 AVEngine 从一个跨 UE/Habitat 的实验性脚本集合，重构为一个透明地建立在 Habitat-Sim fork 和 RLR 之上的、拥有独立资产/场景编译、严格时间轴、反事实控制、QA、provenance、注册和 benchmark 的研究级数据集引擎。**

成功的标志不是“我们修改了 Habitat 很多代码”，而是：

> **我们明确复用了什么、扩展了什么，并证明这些扩展能够生成 Habitat/SoundSpaces 单独运行时不会直接提供的、身份一致、动物关节化、多声源、无视觉嘴部捷径、严格同步且经过质量门禁的数据集。**
