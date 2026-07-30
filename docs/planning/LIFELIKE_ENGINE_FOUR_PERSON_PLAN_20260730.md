# 生活化任务与可变场景四人协作计划

最后核对：2026-07-30。

本文是当前四人协作的任务与提交依据。永久的个人环境、Fork 和 Git
规则见 [`同服务器协作与个人环境构建`](../quickstart.md)；功能现状见
[`AVEngine 功能与使用指南`](../usage_guideline.md)。

## 共同基线

- AVEngine 代码基线：`Eastforward/AVEngine main@3e474e4`；
- 协作目标分支：`Eastforward/AVEngine:integration/lifelike-engine-v1`；
- 每个人只向自己的 Fork 推送个人任务分支，再向上述 integration
  分支提交 Pull Request；
- 本轮默认只修改 AVEngine。只有现有 Habitat API 明确无法支持时，才
  单独提出原生运行时 PR，不能先改运行时再补理由；
- SPEAR、UE、生成模型和大权重不是本轮默认开发依赖。

integration 分支由项目负责人或指定维护者创建和更新。协作者必须先获取
它的当前提交，再创建个人任务分支；不能因为目标分支暂时不存在或落后，
擅自把 PR 改投 `main`。

## 已完成且禁止重复实现

以下能力已经进入当前 `main`，本轮只能复用：

- 动态 `SensorRigTrajectory` 到 Timeline、Habitat/UE Camera、
  Listener、RIR cache、Topdown、DOA、距离和验证器的完整链路；
- M6 AudioProgram 的单源、静音、间歇、顺序、重叠和路由交换；
- `soundspaces2_public`、`habitat_scene`、`spear_ue_authored` 到统一
  RLR 的场景来源与声学 profile 选择；
- 现有 source trajectory bank、Pathfinder/栅格寻路、双声源四种
  静动组合；
- Apartment 既有 1000 条研究数据的分片、恢复、媒体回读和索引工具；
- 已经准入的生成动物资产与 Pixal3D 几何保留流程。

如果 PR 新建了平行的 Timeline、AudioProgram、M7 runner、RIR cache、
声学 selector 或 1000 条生成器，必须证明现有入口确实无法扩展；否则
直接退回。

## 当前不做

- `sit`、坐姿、椅子绑定和全身接触约束；
- 静态资产的大量生成或外观打磨；
- 自动装修、生成新家具或完整房间编辑器；
- 任意 6DoF SensorRig、连续 Doppler 或设备运动噪声；
- 在缺少公开拟合参数和测点时声称复现 SoundSpaces 论文的精确 FRL
  Apartment；
- 在生活任务和轨迹分布冻结前重新生成动态 1000 条数据；
- 为了形式完整而新增与交付无关的门、runner 或重复证据格式。

## 人员 1：生活场景组合

任务 ID：`lifelike-t1`

个人分支：

```text
<GitHub 用户名>/lifelike-t1-scenario-composer
```

目标是扩展现有 M6.x ScenarioSuite，而不是建立第二套任务系统。新增的
薄组合层应引用已有房间区域、实体槽、轨迹、SensorRigTrajectory 和
AudioProgram，并编译为现有 Timeline/M6/M7 可消费的输入。

第一版只交付两个数据驱动示例：

1. 人在厨房区域停留，狗在限定区域活动并按已有 AudioProgram
   间歇发声；
2. 多人在用餐区域保持不移动，声音按已有 AudioProgram 顺序或重叠。

“保持不移动”只表示轨迹静止，不能宣称人物已经坐下或绑定椅子。

最低验收：

- 两个示例均可确定性编译；
- 输出只引用现有轨迹和声音合同，没有复制时间线或音频调度逻辑；
- 新增 schema/编译器的针对性单元测试通过；
- integration 阶段各运行一条小 canary，本人 PR 不需要生成大批数据。

人员 1 不修改底层寻路算法、RIR renderer 或 UE runner。

## 人员 2：生活化轨迹与自动选点

任务 ID：`lifelike-t2`

个人分支：

```text
<GitHub 用户名>/lifelike-t2-trajectory-policies
```

目标是补齐现有轨迹底座中的策略缺口。必须复用
`avengine.m6x.trajectory`、现有 Pathfinder/栅格寻路、
TrajectoryBank 和 SensorRigTrajectory。

第一版必做：

- 让 schema 已声明但当前拒绝执行的 `navmesh_follow` 走现有
  Pathfinder materializer；
- 支持静止、移动前/后停留、区域内游走和区域间移动；
- 从可行区域自动产生 Camera/Listener 候选点、朝向和 geodesic
  路线证据；
- 支持确定性 seed、速度和停留时段；
- 输出仍是现有 source trajectory 与 SensorRigTrajectory 合同。

`follow_actor`、N 主体协调和 per-episode SensorRig bank 可以在上述
必做项通过后继续，但不能以它们为由延迟最小交付。

最低验收：

- 同一输入和 seed 产生相同轨迹与哈希；
- 每帧保持可导航，并通过现有轻量 clearance 检查；
- 一条 MP3D 或 ReplicaCAD 原生路线 canary 通过；
- 不修改已经闭合的 M7 Camera/Listener、RIR、Topdown 和 DOA 链。

## 人员 3：ReplicaCAD 声学封闭

任务 ID：`lifelike-t3`

个人分支：

```text
<GitHub 用户名>/lifelike-t3-replicacad-acoustic-closure
```

目标是在不修改原始 ReplicaCAD visual mesh 的前提下，生成只参与声学
计算的 ceiling/cap 派生几何，处理当前集中向上的声学泄漏。

第一版必做：

- 派生物绑定原始 mesh 身份、操作、变换、材质 ID 和内容哈希；
- 原始场景保持只读，声学封闭面不进入 RGB 或 NavMesh；
- 使用同一组审核过的室内点和方向运行修复前后 leakage A/B；
- 使用同一 source/listener 运行 native RLR RIR、EDT、DRR A/B；
- 将 ReplicaCAD 十个 visual material slots 显式映射为受控近似，
  不再无说明地全部落到默认候选；
- 只更新真实运行并通过的 qualification 维度。

没有实测材质真值时，`physical_material_truth` 和
`dataset_admission` 必须继续保持未准入。减少泄漏不等于完成物理标定。

人员 3 不重写三路 acoustic profile selector，也不修改已通过的 MP3D
或 SPEAR/UE 路线。

## 人员 4：房间布局状态与集成维护

任务 ID：`lifelike-t4`

个人功能分支：

```text
<GitHub 用户名>/lifelike-t4-room-layout-state
```

在功能开发前，人员 4 先用独立小分支提交最小 CPU PR 检查：

```text
<GitHub 用户名>/collab-fast-ci
```

该检查只运行 schema 和不依赖外部数据的 fast-unit；不得在 GitHub
环境安装 UE、Habitat native、RLR、Blender、场景数据或模型权重。

布局功能第一版只支持已有 ReplicaCAD rigid objects：

- 以稳定 object ID 表达平移、旋转、启用和禁用；
- 生成不可变 derived room revision 与 `layout_content_sha256`；
- 在 Habitat 应用变换并回读实际状态；
- 重新构建 live obstacle、可行区域、placement 和轨迹证据；
- 旧 layout 的 trajectory 和 placement 证据必须因布局哈希不匹配而
  拒绝复用；
- 当可移动家具进入声学几何时，RIR cache 必须绑定布局哈希并拒绝旧
  cache；尚未进入声学几何时，则必须显式记录
  `stage_only_ignores_movable_furniture` 近似；
- 用一个桌子或椅子移动前后的 A/B canary 完成验证。

MP3D 扫描和 baked SPEAR Apartment 在第一版收到家具移动请求时应明确
返回 unsupported，不能假装已经改变。若当前 ReplicaCAD 声学包不包含
可移动家具，也必须明确记录该近似，不能声称 RIR 已反映家具移动。

人员 4 同时维护 integration 分支，只负责合入已经审核的 PR 和运行组合
检查，不能代替其他作者在其个人工作树中修改代码。

## 共享文件与依赖

下列文件容易产生冲突：

- `src/avengine/cli.py`；
- `src/avengine/m6x/rir_cache.py`；
- `examples/m6/rooms/room_registry.json`；
- `examples/runtime/acoustic_profiles.json`；
- `docs/roadmap/CURRENT_APARTMENT_EXECUTION.md`；
- `manifest.yaml`。

协作者如果必须修改共享文件，应在 PR 描述中单独列出。两个任务同时
需要同一共享文件时，先各自在自己的模块中完成主体逻辑，再由 integration
维护者做一个最小适配提交，禁止复制一份平行实现来绕开冲突。

人员 1 消费人员 2 的轨迹输出，但两者可以并行开发：人员 1 先使用现有
固定轨迹 fixture，最终在 integration 上接入人员 2 的策略输出。人员 3
与人员 4 都涉及 ReplicaCAD，但前者只拥有 M3 acoustic-only 派生，
后者只拥有 rigid-object layout/readback 和失效规则。

## PR 与合并顺序

所有 PR 的 base 必须是：

```text
Eastforward/AVEngine:integration/lifelike-engine-v1
```

建议顺序：

1. 人员 4 的 `collab-fast-ci`；
2. 人员 2 的轨迹策略；
3. 人员 1 的生活场景组合；
4. 人员 4 的布局状态；
5. 人员 3 的 ReplicaCAD 声学封闭；
6. integration 上运行一次组合检查和真实 canary；
7. 由项目负责人决定是否创建 `integration/lifelike-engine-v1 → main`
   的最终 PR。

每个功能 PR 通过审核后使用 Squash Merge。协作者不得自行合并，也不得
直接向 `main` 提 PR 或推送。详细提交证据格式见
[`代码协作流程`](../quickstart.md#代码协作流程)。

## 组合验收

不为每个 PR 重跑 1000 条。integration 首轮只要求：

- schema 与 fast-unit 全部通过；
- 一条 MP3D 生活任务和一条 ReplicaCAD 生活任务；
- 每条包含 actor 行为、现有 AudioProgram、动态或静态 SensorRig、
  Topdown、DOA、距离和双耳声音；
- 每个场景先运行 4--8 个确定性 seed；
- 一个布局移动 A/B 和一个 ReplicaCAD leakage/RIR A/B；
- 最终视频和机器证据由项目负责人共同审核。

这些结果通过后先扩大到约 50 条小批。只有任务类型、轨迹分布和房间
状态合同冻结后，才决定是否生成新的动态 1000 条。
