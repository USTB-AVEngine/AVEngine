# AVEngine

AVEngine 是一个按房间家族选择生产视觉后端的研究工具，用于从明确的
声源资产、房间、动作程序和证据合同构建可复现的视听 episode。

MP3D 的场景、视觉、传感器和姿态执行由 Habitat-Sim 提供；原生
`apartment_0000` 与 InteriorAgent/Kujiale 的生产视觉由 UE/SPEAR
提供。RLR Audio Propagation 提供几何声学传播，MP3D 使用同一 Habitat
场景上的 SoundSpaces 材质权威。AVEngine 统一负责资产与房间包、权威
时间线、多声源音频组装、质量检查、来源记录和数据准入。

## 主要能力

- 单一 Camera/Listener rig 上共位的 RGB、Depth 和 Semantic 采集；
- 命名动态声源、逐源 FOA/双耳 RIR、stem 和混合音频；
- 精确整数 tick 时间线、同步媒体和受控反事实 episode；
- Camera/Listener 动态轨迹，以及同帧 Topdown、DOA 和距离标签；
- 版本化房间、资产、声源端点、声音和运行配置注册表；
- SoundSpaces、Habitat 和 SPEAR/UE 房间来源到统一 RLR 声学包的路由；
- 资产、房间、媒体和数据集的失败关闭式质量检查。

AVEngine 不是新的模拟器、渲染器或声学求解器，也不会根据视觉材质自动
推断真实物理声学。生成出文件不等于资产、房间或数据集已经准入。

## 系统流程

```text
任务请求
  → 资产、房间和声学场景包
  → 声源程序、Camera/Listener 轨迹和权威时间线
  → 按房间路由的 Habitat 或 UE/SPEAR 视觉执行 + RLR 传播
  → 每源音频、混合音频、RGB、Topdown、DOA、距离和标签
  → 质量检查、来源记录和数据索引
```

生成动物必须在修复、绑定、动画和运行验证中保留自己的 Pixel3D 几何。
库中动物可以提供动作，但不能替换生成动物的 mesh、轮廓、关节或蒙皮
权重。

## 开始使用

协作者必须在自己的个人目录下载并安装 Miniconda，创建自己的 Conda
环境、源码副本、构建目录和 `tmp` 输出。禁止复用项目所有者或其他成员
的环境。

修改任何代码之前，必须先把目标仓库 Fork 到自己的 GitHub 账号，
克隆自己的 Fork，并从规定的上游基线创建一个新的任务分支。一项任务
对应一个新分支；禁止直接在 `main`、固定运行时分支或他人的分支上开发，
也禁止直接向上游仓库推送。完整流程见协作构建文档。

先阅读：

- [同服务器协作与个人环境构建](docs/quickstart.md)
- [功能与使用指南](docs/usage_guideline.md)

完成个人环境安装后：

```bash
cd "${AVENGINE_CODE_ROOT}/AVEngine"
"${AVENGINE_ENV_PREFIX}/bin/python" -m pip install -e '.[test]'
"${AVENGINE_ENV_PREFIX}/bin/python" scripts/validate_schemas.py
"${AVENGINE_ENV_PREFIX}/bin/python" -m pytest -q \
  tests/unit \
  -m 'not integration and not canary'
"${AVENGINE_ENV_PREFIX}/bin/avengine" --help
```

Linux、Git 和 Python 3.10 或更新版本是上层最低要求；当前原生参考环境
使用 Python 3.12。Habitat/RLR 原生构建、外部 UE 安装、场景数据、
Blender 和媒体读回属于独立测试层。Apartment/Kujiale 的 UE 层是对应
房间的生产视觉层，不因为它不是 fast bootstrap 的默认依赖而变成对照层。

## 当前状态

`main` 是当前 Habitat 原生集成基线。现有 Apartment 研究路线支持通用
`source1`、`source2` 绑定、双声源任务、精确时间线、RLR 双耳音频、
Topdown/DOA/距离标签、动态 Camera/Listener 和 episode 级
训练/验证/测试划分。

源码单仓迁移仍在进行。本仓库的最终目标是包含运行所需的精选 Habitat 与
SPEAR 集成源码、AVEngine 自有的 RLR 调用/适配源码和小型 AVEngine 配置。
RLR 传播引擎、头文件、库和 SDK 配置是用户合法安装的 CC BY-NC 4.0 外部
SDK，永不进入 AVEngine Git；当前原生参考仍使用 manifest 固定的 Habitat
fork、已安装的 RLR SDK 与维护中的 SPEAR checkout。迁移完成必须由迁移前后
相同房间路由的实际结果确认，不能由这段目标说明或一次单元测试替代。

这些结果保留各自的证据边界，不代表所有生成动物、房间声学或数据集已经
正式准入。请以以下记录为准：

- [当前 Apartment 执行状态](docs/roadmap/CURRENT_APARTMENT_EXECUTION.md)
- [里程碑与证据状态](docs/roadmap/MILESTONES.md)
- [发布清单](release/avengine_release_manifest_v1.json)

当前 release manifest 在源码迁移期间仍是正式发布状态的唯一依据。分支、
数据结构、预览文件、文档目标或单元测试通过都不能单独代表单仓迁移完成或
正式发布。

## 仓库职责

唯一目标源码仓库是
[`USTB-AVEngine/AVEngine`](https://github.com/USTB-AVEngine/AVEngine)。它
包含 AVEngine 任务包、注册表、时间线、音频组装、质量检查、来源记录、
命令行，以及最终精选迁入的 Habitat 与 SPEAR 集成源码、RLR 适配源码和
小型 AVEngine 配置。RLR 传播引擎、头文件、库和 SDK 配置是用户合法安装的
CC BY-NC 4.0 外部 SDK，永不捆绑进 AVEngine Git。单仓完成后不再需要 RLR
Git checkout、submodule 或源码路径；这不表示迁入 RLR engine。来源映射与
许可证分别记录在
[`UPSTREAM_ADAPTATIONS.md`](docs/provenance/UPSTREAM_ADAPTATIONS.md) 和
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

这不表示把所有依赖和数据塞进 Git。Unreal Engine 安装、Epic 内容、
MP3D、InteriorAgent/Kujiale、原生 Apartment 地图、模型权重、环境、
构建目录和生成媒体始终留在仓库外。迁移期间仍存在的 Habitat 与 SPEAR
代码 checkout 是过渡来源；最终 build、setup 和 run 不再克隆或引用第二
个产品代码仓库。RLR 则继续由用户安装的外部 SDK 提供，而非第二个源码仓库。
gpuRIR 和私有生成模型路线仍是显式可选研究工具。

已接线的 Habitat RLR adapter 默认关闭。启用其静态 C++ target 时，
`AVENGINE_RLR_SDK_ROOT` 必须指向用户安装的官方
`RLRAudioPropagationPkg` 目录（其中含 `headers/` 与
`libs/linux/x64/`）；AVEngine 不会搜索 RLR checkout、复制或安装该 `.so`。
Linux 运行依赖 adapter 的可执行文件或未来 binding 前，用户应让动态加载器
解析自己的 SDK，例如
`LD_LIBRARY_PATH="$AVENGINE_RLR_SDK_ROOT/libs/linux/x64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`。
这只是外部 SDK 的加载方式，不迁入 RLR engine、头文件、库或 material data。

Habitat 的 PBR IBL 图片同样是外部数据，不会嵌入源码或构建产物。
只有创建渲染器且 `PbrShaderAttributes.enable_ibl=true` 时，用户才需提供
`AVENGINE_HABITAT_PBR_ASSET_ROOT`；相对的 BRDF-LUT 和环境图名称分别从
`bluts/` 与 `env_maps/` 解析。无渲染器或关闭 IBL 的路径不需要该变量。

## 文档

- [功能与使用指南](docs/usage_guideline.md)
- [同服务器协作与个人环境构建](docs/quickstart.md)
- [系统架构](docs/architecture/SYSTEM_OVERVIEW.md)
- [仓库职责与接口边界](docs/architecture/REPOSITORY_BOUNDARIES.md)
- [生成动物资产与实例合同](docs/assets/GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md)
- [声学场景与材质合同](docs/architecture/ACOUSTIC_SCENE_AND_MATERIALS.md)
- [时间线与 episode 合同](docs/architecture/EPISODE_AND_TIMELINE.md)
- [文件系统信任模型](docs/security/FILESYSTEM_TRUST_MODEL.md)
- [故障排查](docs/troubleshooting.md)

数据结构位于 [`schemas/`](schemas/)，可执行示例位于
[`examples/`](examples/)，里程碑复现命令位于
[`docs/roadmap/`](docs/roadmap/)。

## 状态词

证据使用 `pass`、`fail`、`blocked`、`not_run`、`research_only` 和
`qualified`。研究候选或历史批准不能在缺少当前证据和注册决定时升级为
`approved_for_dataset`。

## 许可与权利

见 [LICENSE](LICENSE)、[CITATION.cff](CITATION.cff)、
[CITATIONS.bib](CITATIONS.bib) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

AVEngine 当前保留所有权利，协作者必须获得项目所有者授权。Habitat、
RLR、模型、房间、声音和生成资产保留各自许可；当前 RLR 路线是
非商业研究用途。
