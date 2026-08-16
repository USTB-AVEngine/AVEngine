# AVEngine 功能与使用指南

最后核对：2026-07-30。

本文回答两个问题：

1. 当前主分支已经具备哪些能力；
2. 对应能力应从哪里进入、需要哪些输入、不能宣称什么。

个人安装和协作规则见
[`同服务器协作与个人环境构建`](quickstart.md)。当前执行状态以
[`CURRENT_APARTMENT_EXECUTION.md`](roadmap/CURRENT_APARTMENT_EXECUTION.md)
为准。
本文不复制本机临时产物的哈希或把研究验证写成正式数据集发布。

## 总体定位

AVEngine 是建立在 Habitat-Sim 和 RLR Audio Propagation 之上的
视听任务与数据生成层：

```text
任务请求
  → 房间、资产、声源和声学配置
  → 声源轨迹、Camera/Listener 轨迹和整数时间线
  → Habitat 状态、传感器与 RLR 传播
  → RGB、Topdown、RIR、双耳音频、DOA、距离和标签
  → 质量检查、证据、数据索引与训练/验证/测试划分
```

AVEngine 不重新实现渲染器或声学求解器。Habitat 负责原生场景、导航和
传感器，RLR 负责几何声学传播；AVEngine 负责把这些能力组织成可验证的
任务和数据。

## 当前功能矩阵

| 功能 | 当前能做什么 | 主要入口 | 额外输入 |
| --- | --- | --- | --- |
| 数据结构与快速检查 | 校验房间、资产、时间线、声源、声音、AudioProgram、证据和发布清单；运行不依赖外部数据的单元测试 | `scripts/validate_schemas.py`、`tests/unit/` | 无 |
| 房间与单视角采集 | 在一个正式 `view0` 上共位采集 RGB、Depth、Semantic；Camera 与 Listener 共位；Topdown 仅用于质量检查 | `avengine m1` | 原生采集需要 Habitat 和房间数据 |
| 相机/麦克风选点 | 指定位置、yaw 和 HFOV，进行导航网格落点与原生三模态探测 | `tools/m1/build_camera_pose_request.py`、`tools/m1/probe_camera_pose_native.py` | 房间、导航网格、Habitat |
| 动物资产与动作检查 | 管理生成动物的资产包、动作重定向、支撑面找平、接触/形变检查、原生姿态回读和准入证据 | `tools/m2/`、`src/avengine/m2/` | 生成模型和权重不在本仓库 |
| 房间声学与材质 | 编译自定义 GLB、MP3D semantic、USD snapshot 和视觉材质槽；按房间来源选择 SoundSpaces、Habitat 或 SPEAR/UE 配置，最终统一上传给 RLR | `avengine m3` | 房间 mesh、语义/材质数据 |
| 多声源空间音频 | 支持命名声源、独立源/Listener 位姿、FOA 与双耳 RIR、每源 stem、混合音频和严格 WAV 回读 | `avengine m4`、`tools/m6x/render_rir_cache.py` | Habitat/RLR；双耳还需 HRTF |
| 时间线与低层任务编排 | 使用 Timeline v2 整数 tick；支持静/静、动/静、静/动、动/动；AudioProgram 支持单一激活、静音负例、间歇、顺序、重叠和路由交换；通用生活场景组合层仍在开发 | `avengine m5`、`avengine m6`、`examples/m6x/fixed_apartment/` | 已登记的轨迹、端点和干声 |
| 动态 Camera/Listener | 逐帧保存并应用 SensorRigTrajectory；Timeline、RGB、UE 回读、RIR key、Topdown、DOA 和距离使用同一帧位姿 | `src/avengine/sensor_rig_trajectory.py`，下游参数 `--sensor-rig-trajectory` | geodesic 路线需要 Pathfinder 证据 |
| 批量生成与索引 | 支持分片、恢复、RIR 缓存复用、双耳批量组装、产物级验证和按视觉 episode 划分训练/验证/测试集 | `tools/m7/` | 已完成的计划、视觉、RIR 和干声 |
| 审片与标签可视化 | 合成 RGB、Topdown、双耳音频、DOA 和距离的同步审片视频，并保留哈希绑定证据 | `tools/m7/build_mp3d_room_evaluation_review.py` 等 | 已完成的视觉和音频结果 |

## 从哪里开始

先进入自己的个人环境和仓库：

```bash
source "${AVENGINE_MINICONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate "${AVENGINE_ENV_PREFIX}"
cd "${AVENGINE_CODE_ROOT}/AVEngine"
export AVENGINE_RUN_ID="guide_run_001"
```

查看稳定命令行入口：

```bash
"${AVENGINE_ENV_PREFIX}/bin/avengine" --help
"${AVENGINE_ENV_PREFIX}/bin/avengine" m1 --help
"${AVENGINE_ENV_PREFIX}/bin/avengine" m3 --help
"${AVENGINE_ENV_PREFIX}/bin/avengine" m4 --help
"${AVENGINE_ENV_PREFIX}/bin/avengine" m5 --help
"${AVENGINE_ENV_PREFIX}/bin/avengine" m6 --help
```

M7 目前主要是显式工具链，还没有统一成一个总命令：

```bash
"${AVENGINE_ENV_PREFIX}/bin/python" \
  tools/m7/build_room_evaluation_plan.py --help
"${AVENGINE_ENV_PREFIX}/bin/python" \
  tools/m7/run_habitat_room_batch.py --help
"${AVENGINE_ENV_PREFIX}/bin/python" \
  tools/m7/render_room_evaluation_binaural.py --help
```

## 1. 先做不依赖外部数据的检查

这是修改代码、数据结构、任务逻辑或注册表后的最低检查：

```bash
"${AVENGINE_ENV_PREFIX}/bin/python" scripts/load_paths.py \
  --validate \
  --layer fast_unit
"${AVENGINE_ENV_PREFIX}/bin/python" scripts/validate_schemas.py
"${AVENGINE_ENV_PREFIX}/bin/python" -m pytest -q \
  tests/unit \
  -m 'not integration and not canary'
```

还可以直接验证仓库内示例请求：

```bash
"${AVENGINE_ENV_PREFIX}/bin/avengine" m1 validate-room \
  examples/m1/rooms/blender_custom/room_manifest.json

"${AVENGINE_ENV_PREFIX}/bin/avengine" m1 validate-request \
  examples/m1/requests/blender_custom.json \
  --room examples/m1/rooms/blender_custom/room_manifest.json

"${AVENGINE_ENV_PREFIX}/bin/avengine" m4 validate-request \
  examples/m4/blender_custom/multi_source_canary_request.json

"${AVENGINE_ENV_PREFIX}/bin/avengine" m5 validate-request \
  examples/m5/blender_custom/two_dog_simultaneous_counterfactual_request.json

"${AVENGINE_ENV_PREFIX}/bin/avengine" m6 validate-controlled-request \
  examples/m6/canary/controlled_one_active_of_two_request.json
```

这些命令只证明请求和数据结构闭合，不代表原生 Habitat/RLR 已执行。

## 2. 采集房间视觉与传感器数据

先验证房间和采集请求，再运行 M1：

```bash
"${AVENGINE_ENV_PREFIX}/bin/avengine" m1 capture \
  --room examples/m1/rooms/blender_custom/room_manifest.json \
  --request examples/m1/requests/blender_custom.json \
  --runtime-prefix "${AVENGINE_HABITAT_RUNTIME_PREFIX}" \
  --output "tmp/m1/${AVENGINE_RUN_ID}"
```

输出必须是新的个人 `tmp` 子目录。M1 正式视角只有 `view0`；RGB、
Depth、Semantic 和 Listener 使用同一个相机位姿。Topdown 是质量检查
产物，不是第二个正式视角。

如果只研究相机/麦克风位置：

```bash
"${AVENGINE_ENV_PREFIX}/bin/python" \
  tools/m1/build_camera_pose_request.py --help
"${AVENGINE_ENV_PREFIX}/bin/python" \
  tools/m1/probe_camera_pose_native.py --help
```

## 3. 编译和检查声学房间

自定义测试房间可以使用 M3 的显式材质路径：

```bash
"${AVENGINE_ENV_PREFIX}/bin/avengine" m3 compile-custom \
  --room examples/m1/rooms/blender_custom/room_manifest.json \
  --mapping examples/m3/blender_custom/mapping.json \
  --materials examples/m3/blender_custom/materials_low.json \
  --output "tmp/m3/${AVENGINE_RUN_ID}"

"${AVENGINE_ENV_PREFIX}/bin/avengine" m3 validate-package \
  "tmp/m3/${AVENGINE_RUN_ID}/manifest.json"
```

按房间注册表自动选择来源配置时使用：

```bash
export AVENGINE_ROOM_MANIFEST="/path/to/room_manifest.json"
export AVENGINE_ROOM_ID="registered_room_id"
export AVENGINE_ROOM_REVISION="registered_room_revision"

"${AVENGINE_ENV_PREFIX}/bin/avengine" m3 compile-registered-scene \
  --room "${AVENGINE_ROOM_MANIFEST}" \
  --room-id "${AVENGINE_ROOM_ID}" \
  --room-revision "${AVENGINE_ROOM_REVISION}" \
  --runtime-root "${AVENGINE_HABITAT_RUNTIME_ROOT}" \
  --output "tmp/m3/${AVENGINE_RUN_ID}_registered"
```

可选声学检查包括：

- `m3 inspect-mesh-leakage`：从审核过的室内点发射射线检查漏声；
- `m3 resolve-materials`：应用全局和逐材质参数覆盖；
- `m3 import-rlr-materials`：无插值导入 RLR 材质曲线；
- `m3 verify-soundspaces-reference`：验证保留的 SoundSpaces 公共参考包。

SoundSpaces、Habitat 和 SPEAR/UE 代表房间最终都使用同一个
`rlr_audio_propagation` 求解后端；不同的是房间来源、材质映射和参考
参数，不是三套互不相干的声学求解器。

## 4. 生成 RIR 和空间音频

先构建声源/Listener job plan，再使用原生 RLR 缓存工具：

```bash
export AVENGINE_RIR_JOB_PLAN="/path/to/rir_job_plan.json"

"${AVENGINE_ENV_PREFIX}/bin/python" \
  tools/m6x/render_rir_cache.py \
  --rir-job-plan "${AVENGINE_RIR_JOB_PLAN}" \
  --room-id "${AVENGINE_ROOM_ID}" \
  --room-revision "${AVENGINE_ROOM_REVISION}" \
  --layout binaural \
  --output "tmp/rir/${AVENGINE_RUN_ID}"
```

也可以用 `--acoustic-package-manifest` 和 `--simulation-request` 显式
指定声学包和仿真参数。当前缓存标识包含：

- 声源位置；
- Listener 位置；
- Listener 朝向；
- 房间/声学包；
- 仿真请求和输出布局。

因此移动 Listener 时不会错误复用静态 Listener 的旧 RIR。

M4 的小型原生验证入口是：

```bash
"${AVENGINE_ENV_PREFIX}/bin/avengine" m4 run-canary --help
"${AVENGINE_ENV_PREFIX}/bin/avengine" m4 verify-canary --help
```

完整运行还需要声学包、HRTF、干声和新的个人输出目录。

## 5. 设计生活场景任务

当前底层任务应组合以下原子：

- 房间和导航可行区域；
- `source1`、`source2` 等通用声源槽；
- 每个声源的静止或移动轨迹；
- Camera/Listener 的静止、旋转或移动轨迹；
- AudioProgram 的发声时段和路由；
- Timeline v2 的逐帧状态；
- 需要输出的 RGB、Topdown、RIR、双耳音频、DOA、距离和标签。

现有底层合同可以通过人工配置表达：

- 人在厨房活动、狗在房间内移动；
- 两个人在用餐区域不移动时的“静止声源槽”；
- 一个声源间歇发声，另一个持续或静音；
- 两个声源顺序发声或重叠发声；
- 静/静、动/静、静/动、动/动四类空间关系；
- 相机/麦克风静止、原地旋转或沿审核过的路线移动。

AudioProgram 示例位于：

```text
examples/m6x/fixed_apartment/audio_programs/
```

其中包括路由检查、交换反事实、静音负例、间歇声、重叠声和
LOS/NLOS 顺序声。

当前还没有把房间区域、人物/动物行为、轨迹、SensorRig 和
AudioProgram 自动组合起来的通用生活场景编译层。`navmesh_follow`
虽然已经出现在轨迹 schema 中，固定 M6.x materializer 仍会拒绝执行；
自动 Camera/Listener 候选点、停走、区域游走和家具布局变体也属于当前
待实现能力。开发者应扩展现有 ScenarioSuite、
TrajectoryBank、SensorRigTrajectory 和房间注册合同，不得另建平行的
时间线、声音调度或 M7 runner。

当前还不能把“坐着”当作带身体姿态和椅子约束的正式动作。静止槽可以
表达“不移动”，但不能宣称角色已经完成坐姿、椅子绑定或全身碰撞验证。

## 6. 使用动态 Camera/Listener

`SensorRigTrajectory v1` 当前固定为：

- 5 秒；
- 15 fps；
- 75 帧；
- 一个正式 `view0`；
- 一个与 Camera 共位共向的 Listener；
- 只支持绕世界 `+Y` 轴的 yaw。

支持的程序类型包括 HOLD、原地旋转、折线路径、已有 Pathfinder 证据的
geodesic 路径和带帧号 waypoint。轨迹生成目前主要通过
`avengine.sensor_rig_trajectory.materialize_sensor_rig_trajectory()`
Python API 完成，还没有独立的总命令。

生成轨迹 JSON 后，先保存路径：

```bash
export AVENGINE_SENSOR_RIG_TRAJECTORY="/path/to/sensor_rig_trajectory.json"
```

再通过下游命令的
`--sensor-rig-trajectory "${AVENGINE_SENSOR_RIG_TRAJECTORY}"` 参数接入。

它可接入 M5/M7、UE 组合和审片工具。验证器会交叉检查 Timeline pose hash、
视觉回读、RIR Listener pose、Topdown、DOA 和距离，避免只移动标签而
画面或声音仍停在初始位置。

## 7. 批量生成、审片和数据索引

房间评估主链按以下阶段运行：

```text
build_room_evaluation_plan.py
  → render_rir_cache.py
  → render_room_evaluation_binaural.py
  → Habitat/UE 视觉采集
  → build_mp3d_room_evaluation_review.py
  → 产物级验证和数据索引
```

建立房间计划：

```bash
export AVENGINE_SOURCE_TRAJECTORY_BANK="/path/to/source_trajectory_bank"
export AVENGINE_TEMPLATE_RIR_PLAN="/path/to/template_rir_plan.json"
export AVENGINE_SENSOR_RIG_TRAJECTORY="/path/to/sensor_rig_trajectory.json"
export AVENGINE_EPISODE_COUNT=100

"${AVENGINE_ENV_PREFIX}/bin/python" \
  tools/m7/build_room_evaluation_plan.py \
  --trajectory-bank "${AVENGINE_SOURCE_TRAJECTORY_BANK}" \
  --template-rir-plan "${AVENGINE_TEMPLATE_RIR_PLAN}" \
  --episode-count "${AVENGINE_EPISODE_COUNT}" \
  --sensor-rig-trajectory "${AVENGINE_SENSOR_RIG_TRAJECTORY}" \
  --output "tmp/m7/${AVENGINE_RUN_ID}_plan"
```

使用已完成的 RIR 缓存组装双耳音频：

```bash
"${AVENGINE_ENV_PREFIX}/bin/python" \
  tools/m7/render_room_evaluation_binaural.py \
  --plan-root "tmp/m7/${AVENGINE_RUN_ID}_plan" \
  --rir-cache "tmp/rir/${AVENGINE_RUN_ID}" \
  --class-audio 'dog_bark=/path/to/dog_bark.wav' \
  --class-channel-policy 'dog_bark=require_mono' \
  --class-linear-gain 'dog_bark=1.0' \
  --output "tmp/m7/${AVENGINE_RUN_ID}_binaural"
```

计划中每个声音类别都必须分别重复三类 `--class-*` 参数，且等号左侧的
类别名必须完全一致。

同步生成 MP3D RGB、Topdown、DOA、距离和双耳审片：

```bash
export AVENGINE_VISUAL_CAPTURE_ROOT="/path/to/habitat_visual_capture"

"${AVENGINE_ENV_PREFIX}/bin/python" \
  tools/m7/build_mp3d_room_evaluation_review.py \
  --audio-root "tmp/m7/${AVENGINE_RUN_ID}_binaural" \
  --visual-capture-root "${AVENGINE_VISUAL_CAPTURE_ROOT}" \
  --plan-root "tmp/m7/${AVENGINE_RUN_ID}_plan" \
  --output "tmp/m7/${AVENGINE_RUN_ID}_review"
```

M7 还提供：

- `render_asset_bound_binaural_batch.py`：资产绑定的双耳批量组装；
- `verify_asset_bound_batch.py`：逐项读取并验证完整批次；
- `run_habitat_room_batch.py`：注册房间的 Habitat 批量视觉采集；
- `build_asset_bound_dataset_index.py`：不复制媒体地建立数据索引；
- `compare_rir_cache_metrics.py`：对齐 job 比较 EDT、DRR 和晚期能量。

训练/验证/测试划分按视觉 episode 进行，不能让同一视觉 episode 的音频
变体跨集合泄漏。已有 1,000 条本机研究闭环证据不等于 M7 benchmark、
论文或正式数据集已经发布。

## 房间来源与默认路线

| 房间来源 | 视觉/场景基础 | 声学参数入口 | 当前边界 |
| --- | --- | --- | --- |
| MP3D / SoundSpaces | Habitat MP3D mesh、semantic 和导航数据 | SoundSpaces 公共 MP3D 材质曲线，经 AVEngine 编译后上传 RLR | 不包含未公开 FRL 拟合参数和七个测点世界坐标 |
| ReplicaCAD / Habitat | Habitat ReplicaCAD 视觉场景 | 视觉材质槽映射到声明的研究参数，再上传 RLR | 当前是研究近似，不是实测材质真值 |
| SPEAR/UE Apartment | UE/SPEAR 视觉房间 | 已声明的视觉材质槽规则，再上传 RLR | UE 是可选视觉后端；材质未做真实房间标定 |
| 自定义房间 | 用户提供 GLB/房间包 | 显式 mapping、materials 和仿真请求 | 用户必须负责几何、材质来源和资格检查 |

SPEAR/UE 不是默认依赖。没有 UE 的协作者仍可开发和验证数据结构、轨迹、
RLR、音频、Habitat 与 MP3D 研究路线；但不能重建当前最终 Apartment UE
RGB 和依赖 UE render root 的数据索引。

## 当前明确限制

- 正式时钟目前固定为 5 秒、15 fps、75 帧、16 kHz、80,000 samples；
- 正式视角只有 `view0`，Camera 与 Listener 共位共向；
- SensorRig 只支持 yaw，不支持 pitch/roll、多个 Listener 或多个正式视角；
- 尚无正式的坐姿、椅子绑定、通用动作图或自然语言任务编译器；
- 尚无通用家具重排、整个房间原子化编辑或自动补透明墙系统；
- 放置检查主要基于声源中心、导航和已声明 gate，不等于任意角色全身碰撞
  已得到保证；
- 动物生成模型、权重和实际生成资产不随 Git 克隆提供；新动物必须保留
  自己的生成 mesh，并单独通过动画、接触、视觉和原生回读检查；
- 当前动物别名和运行资产以
  [`source_asset_runtime_profiles.json`](../examples/runtime/source_asset_runtime_profiles.json)
  为准，不在本文重复维护品种名称；
- SoundSpaces 公共曲线可以无损接入，但不能宣称复现未公开的 FRL
  Apartment 拟合结果；
- ReplicaCAD 和 SPEAR Apartment 的材质仍是研究近似；
- `run_habitat_room_batch.py` 当前只接受 `m5_1-mixed-route` 输入布局，不是
  任意 M7 资产组合的通用 Habitat 渲染器；
- 当前结果是有边界的研究验证，不等于正式数据集准入、benchmark 或论文
  发布完成。

## 进一步阅读

- [`当前 Apartment 执行状态`](roadmap/CURRENT_APARTMENT_EXECUTION.md)
- [`里程碑与证据状态`](roadmap/MILESTONES.md)
- [`系统架构`](architecture/SYSTEM_OVERVIEW.md)
- [`仓库职责边界`](architecture/REPOSITORY_BOUNDARIES.md)
- [`房间与声学场景包`](architecture/ROOM_AND_ACOUSTIC_SCENE_PACKAGE.md)
- [`时间线与 episode`](architecture/EPISODE_AND_TIMELINE.md)
- [`生成动物资产合同`](assets/GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md)
- [`故障排查`](troubleshooting.md)
