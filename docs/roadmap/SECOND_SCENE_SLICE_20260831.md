# 第二个同链路可渲染场景:范围受控的工程切片(20260831)

> 目的**只有一个**:为跨场景端到端 canary 提供第二个场景。不扩建新的
> 场景平台,不顺带处理无关资产。全链 research_only。
>
> 边界重申:设计/采样层已经能读多个场景(三个真实场景同配置各 24/24
> 准入),但那**不是**端到端泛化。渲染泛化要求同一条生产链在至少两个
> 场景跑通,这正是本切片要补的那一格。

## 1. 现状核查(逐条查证,不是推测)

| 事实 | 证据 |
| --- | --- |
| 生产视觉链只有一个打包舞台 | `ue-package-stages/` 下三个目录全是 `apartment_0000_*` |
| 舞台里没有第二张地图 | 在 `apartment_0000_1ae5294` 里 `find -ipath "*Kujiale*"` 为空;1.1 GiB 的 `SpearSim-Linux.pak` 里搜 `kujiale_0020_full_home_v1` 命中 0 |
| 渲染层曾把地图写死 | `current_apartment_visual.py` 的模块常量 `NATIVE_APARTMENT_MAP`;**已参数化并加失配校验**(提交 f709068) |
| UE 捕获命令没有地图参数 | `m5 capture-current-apartment-visual` 只吃舞台/闭包报告/可执行文件 |
| habitat 捕获是另一条链 | `m5 capture-current-visual` 吃 `--mp3d-root`,是 MP3D/habitat 后端,**不能顶替 UE 链的端到端证明** |
| 内容里有哪些地图 | SPEAR 原生:`apartment_0000`、`debug_0000`、`debug_0001`;另有 `kujiale_0020_*` 的 umap |
| kujiale 曾经渲染过 | `review/kujiale_kitchen_current_visual_capture_08974c6_retry1_20260820T1825Z`:75 帧、含像素掩码/深度/可见性真值、`scene.map_path=/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1`、`status=pass` |
| 但它的地图是 USD 舞台演员型 | `map_result.json`:`level_actor_classes={'UsdStageActor': 1}`,运行时加载 derived USD;后来那份源码切片的 `map_creation_status`/`editor_build_status` 都是 `not_run` |
| 舞台组装链是现成的 | `tools/ue/build_minimal_closure_report.py` + `tools/ue/assemble_package_stage.py`,cook 的环境变量与失败模式在 `STAGE_REBUILD_20260823.md` 里逐条记着 |

## 2. 两条候选路线与取舍

**路线 A:SPEAR 原生 debug 场景(便宜、风险低)**
和 apartment 走完全相同的内容管线,只是另一张地图。不需要 USD 运行时,
不需要编辑器建图。代价是 debug 房间**不是生产住宅**,场景多样性弱。

**路线 B:kujiale 全屋(真实住宅、风险高)**
它是真正意义上的第二个场景,而且已经渲染过一次。风险在于地图是 USD
舞台演员型:打包 cook 之后 USD 能否在运行时正常加载,**没有验证过**;
当初那份切片的编辑器建图与构建都标着 `not_run`。

**取舍**:两条都做,但**先 A 后 B**,且 A 只作"链路可换地图"的能力
canary、不冒充场景多样性。理由是 A 能在一次 cook 内证明"同一套代码换
地图就能跑",把风险集中到 B 的 USD 打包这一个未知项上;若先做 B 而 USD
打包不成立,我们会同时不知道是链路问题还是 USD 问题。

## 3. 切片交付物(到此为止,不外扩)

1. 一个包含 `apartment_0000` + `debug_0000` 两张地图的打包舞台(闭包
   报告 → 组装 → BuildCookRun,沿用既有工具与既有环境变量)。
2. UE 捕获命令接受地图参数(渲染层的房间无关化;代码侧已完成一半——
   地图已是时间线声明的事实并带失配校验,还需把参数接到 CLI)。
3. 在两张地图上用**完全相同**的采样器配置与题型 profile 各跑一个小
   canary(①F、①B、卡⑧、卡⑦或⑨ 至少一题),逐点核验:
   - 相机画面 yaw 与听者 yaw 一致(已有批级扫描工具);
   - 时间线声明的地图与实际启动的地图一致(已有 fail-closed 校验);
   - 音频事件与视觉事实绑定正确;
   - Gate A/B 翻转符合预期;
   - 第二场景的答案不被同一个桶垄断;
   - 不支持的实例返回正确拒绝原因。
4. 若 debug 场景没有导航边界体(`NavMeshBoundsVolume`),路线库为空 ——
   这是**正常结果**,如实记录并改用路线 B;不为它写房间专用补丁。

## 4. 明确不做的事

- 不新建场景平台、不引入新的资产管线;
- 不为第二场景写房间 ID 分支、手填相机或手填路线;
- 不用 habitat 后端顶替 UE 链的端到端证明;
- 不在跨场景 canary 通过前启动 run02;
- 不宣称"场景泛化完成"——三层证据是分开的:仓库有编译器组件 ≠ 同一
  题型生成器在多个真实场景直接出题 ≠ 同一生产渲染链在两个场景跑通。

## 5. 已知的第一个检查点

cook 一次公寓舞台的耗时与失败模式已有记录(五类失败:SDK 环境变量、
闭包缺动画包、uproject 未显式启用 SpContent、缺宿主扩展、rgb 是 BGR)。
第二张地图的第一个真实未知是:**debug 场景是否带导航边界体**——没有
navmesh 就没有路线库,采样器会诚实拒绝该场景。这一项在舞台 cook 完成
后第一时间验证,再决定是否直接跳到路线 B。
