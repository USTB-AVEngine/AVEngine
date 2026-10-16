# 给 Codex 的入口 Prompt：AVEngine Habitat-Native 总体重规划

你正在重新规划 **AVEngine**。当前已存在 Habitat-Sim fork：

- `https://github.com/Eastforward/habitat-sim-AVEngine`
- upstream：`https://github.com/facebookresearch/habitat-sim`

本任务不是从零再造 simulator，也不是继续维护 UE/SPEAR + Habitat/RLR 作为论文主路径。目标是将系统迁移为：

> **Habitat-Sim fork 统一承担视觉、场景、传感器、关节姿态执行和 RLR 声学；独立 AVEngine 主仓库承担动物/房间资产编译、权威时间轴、反事实 episode、QA、provenance、registry 和数据集输出。**

开始工作前，必须完整阅读服务器中的：

1. `CODEX_MASTER_PLAN_HABITAT_NATIVE_AVENGINE_20260716.md`
2. `AVEngine_review_20260715_zh.md`
3. `avengine_timeline_v2.schema.json`
4. 若存在：`CODEX_GOAL_AVENGINE_QUADRUPED_AVSYNC.md`
5. 旧 AVEngine 仓库中这些文档引用的实际脚本、测试、资产和运行入口

## 首轮目标

首轮不要直接大规模实现功能。先完成 audit、仓库拆分、架构文档、迁移矩阵、里程碑、引用与许可证治理，并建立可复现 baseline。

### 必须保留两个仓库边界

#### A. `Eastforward/habitat-sim-AVEngine`

这是保留完整上游历史的 Habitat-Sim research fork，只放 runtime 必需修改：

- modern RLR C API adapter；
- named multi-source / multi-listener；
- per-pair IR；
- explicit AcousticScenePackage / per-triangle material ingestion；
- deterministic non-human articulated pose playback；
- one-state multi-sensor capture；
- runtime version/build manifest。

保留 Habitat-Sim MIT LICENSE、上游版权、fork 关系和 Git 历史。不得声称从零实现 simulator。

#### B. `Eastforward/AVEngine`

若远程仓库不存在，创建或生成本地 scaffold 与远程创建说明。该仓库放原创系统层：

- animal asset compiler 与 template bank；
- Blender offline pipeline；
- room/acoustic scene compiler；
- schemas；
- authoritative timeline；
- episode/counterfactual generation；
- audio stems/mix；
- QA、provenance、registry；
- CLI、examples、benchmark-facing outputs。

不要把 Habitat 源码复制到该仓库。

## 必须准确区分来源

- Habitat-Sim：scene graph、PBR renderer、sensors、physics、navigation、articulated-object 基础。
- RLR / SoundSpaces 2.0：geometry-based acoustic propagation。
- RLR modern C API 已支持多个 source/listener；AVEngine 贡献是 Habitat adapter、身份、事件、独立 stems 和数据契约，不是发明多声源算法。
- AVEngine：稳定动物资产、显式声学场景、时间轴、反事实、QA、provenance、registry 和新任务。

## 关键产品约束

1. 正式动物路线：模板负责 topology、UV、skeleton、weights、collision 和 actions；生成结果只作 shape/PBR guide。
2. 不做嘴部动画，目的为防止视觉捷径；v2 timeline 中 `open_ratio=0`，并在 episode manifest 明确 `disabled_for_shortcut_control`。
3. 正式房间声学代理禁止 AABB；必须保留门洞、窗洞、房间连通性和主要遮挡/反射面。
4. 视觉 PBR 材质不会自动成为声学材质；必须显式上传 triangle -> material category -> RLR parameters。
5. 初始声学范围是 static/quasi-static acoustic geometry + dynamic semantic point emitters，不声称 fully dynamic deformable-body acoustics。
6. `avengine_timeline_v2.schema.json` 不得静默修改；新版本必须新 schema + ADR + migration。
7. 所有未实际运行的 Blender/GPU/RLR/E2E 测试标记为 `not_run`，不能写成通过。

## 首轮必须创建的核心文件

Habitat fork：

```text
UPSTREAM.md
MODIFICATIONS.md
THIRD_PARTY_NOTICES.md
CITATION.cff
CITATIONS.bib
runtime.lock.yaml
AVENGINE_RUNTIME_VERSION
docs/avengine/ARCHITECTURE.md
docs/avengine/BUILD_AND_REPRODUCIBILITY.md
```

AVEngine 主仓库：

```text
README.md
LICENSE
THIRD_PARTY_NOTICES.md
CITATION.cff
CITATIONS.bib
runtime.lock.yaml
schemas/avengine_timeline_v2.schema.json
docs/architecture/*
docs/adr/*
docs/migration/*
docs/paper/*
docs/roadmap/*
```

至少生成以下 ADR：

```text
Habitat-native primary runtime
two-repository boundary
explicit AcousticScenePackage
authoritative integer timeline
disabled visual mouth articulation
template-authoritative animal assets
modern RLR C API
static acoustic geometry initial scope
```

## 首轮必须审计

```text
origin/upstream remotes
HEAD/upstream/merge-base SHA
fork ahead/behind
all submodule SHAs
Habitat base release/tag
clean build and test baseline
old AVEngine module inventory
assets and dataset licenses
current RLR interface actually used
current AABB/material/timeline paths
```

## 实施 milestone 顺序

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

不要把所有工作放进一个 issue。每个 issue 必须有 scope、non-goals、dependencies、deliverables、acceptance tests 和 not_run 条件。

## 论文与 README 表述

推荐：

> AVEngine is implemented as an independent research extension to Habitat-Sim rather than as a simulator built from scratch.

> AVEngine reuses Habitat-Sim for scene, rendering, sensor, physics and articulated-object infrastructure, and RLR/SoundSpaces 2.0 for acoustic propagation. It contributes audited articulated assets, explicit acoustic-scene compilation, identity-preserving multi-source integration, an authoritative frame-sample timeline, anti-shortcut counterfactual generation, QA, provenance and dataset admission.

不得声称：新 renderer、新 acoustic solver、首次多源 RLR、Habitat 不能动画、SoundSpaces 不支持动态位置、lip sync、fully dynamic acoustic bodies。

## 首轮输出格式

完成后分别报告：

1. exact Git/repository status；
2. 创建和修改的文件；
3. ADR 摘要；
4. legacy migration matrix；
5. milestones 和 issue backlog；
6. build/test/canary 状态表，状态只能是 `pass/fail/not_run/blocked`；
7. attribution 和 citation 检查；
8. 仍未解决的风险。

完整要求以 `CODEX_MASTER_PLAN_HABITAT_NATIVE_AVENGINE_20260716.md` 为准。
