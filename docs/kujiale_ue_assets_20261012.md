# 酷家乐 UE 动物与物件接入（2026-10-12）

本能力在生产 SpearSim 工程的任务私有派生副本中导入已入册动物与物件，以 UE/SPEAR 原生 Vulkan 图像和量测几何准入，修正人物站床问题。产物位于 `v7_kujiale_20261003/ue_round20_assets`；本轮不生成题目、不重跑量产、不提交 ACP/CCI、不修改源登记或 GLB。

## 输入与边界

`prepare_kujiale_asset_admission.py` 读取动物、物件的 admitted/pass 个体，并保留原人名册的 4 个已有发声动物。已有人物及 5 种椅子的 UE 包随生产工程复制，沿用既有准入；本轮新增清单是 77 个新动物、192 个物件及 4 个既有动物。4 只源登记允许通过刚体逐帧缓存走动的猫，在本轮 UE 中按静态准入，未声称带 UE 走动动画。6 只 Rocketbox 狗导入 SkeletalMesh、IdleContact 和 WalkingContact。

源 GLB 只读。`gltf_static_bake.py` 将场景节点 TRS 烘焙到私有派生网格顶点/法线，保留 UV、纹理和材料定义。导入器关闭派生 StaticMesh 的 Nanite，依据实际 UE LOD0 读回尺寸和接触顶点；不能用原始 GLB 的包围盒代替导入结果。狗的材质 usage 修复仅修改派生材质及派生 SkeletalMesh，不改共享 SPEAR M_Default。

## 生命周期与接口

1. `tools/ue/prepare_kujiale_asset_admission.py`：源登记快照、来源哈希、派生 GLB。
2. `tools/ue/import_kujiale_admission_editor.py`：生产工程内导入及实际 LOD0 读回。
3. `tools/ue/plan_kujiale_admission.py`：量测房间网格上的地面、桌面、墙面、天花板附着，完整视觉 OBB 与家具/墙三角形碰撞检查。
4. `tools/ue/capture_kujiale_admission.py`：三个原生完整视角、可见比例、高光、尺寸、发声点读回和 UE 画面主色族。
5. `tools/ue/capture_kujiale_dog_motion.py`：所有 authored 30 Hz key 的原生骨骼、嘴部组件、从下方四视角实际爪底深度、身体轴及房间碰撞。
6. `tools/ue/capture_kujiale_asset_trial.py`：三套已准入房间中与可见人物同场的身体/物件类型覆盖。
7. `tools/ue/capture_kujiale_production_binding.py`：直接调用生产适配器实体生成方法，原生验证四种附着面与真实发声组件。
8. `tools/ue/report_kujiale_asset_admission.py`：逐个体新登记及中文内嵌图片单文件 HTML。

深度读回版本 `native-radial-meters-v3` 明确使用径向米及像素中心归一化射线。原生全场/仅资产配对深度允许 5 mm 遮挡误差，可见比例至少 99%，边界留 5 像素，至少三视角。动物贴地 12 mm、物件贴附 15 mm、尺寸 ±15%、高光 ≤2%、发声点 ≤0.5 mm。未通过的旧版本结果保留为诊断证据，不能混入准入。

颜色从原生图像身体/机身 ROI 测量，不与 Habitat 贴图做色差对比。参照为任务书已认可的同房间 ue_round13 图像，使用原生产曝光/白平衡设置。颜色差异留给 owner，代码不改登记标签。

## 生产适配器连接

`NativeRegisteredUEVideo` 可显式使用 `inputs.kujiale_ue_asset_admission`。非人物 source 必须给 `native_asset_id`，该个体物理准入必须完成，当前房子必须存在通过的原生见证，运行的私有 UE 工程必须与登记一致。声明声源轨迹须是有限三维坐标，且与真实组件读回位置的距离 ≤0.5 mm；静音摆设不能当声源。缺少资产、错误房间、错误工程、漂移点位或未经准入的空间行走，在启动 GPU 前拒绝。

环境实体使用生产 `spawn_attached_static_actor`，按已准入根节点和朝向生成原 StaticMesh 及附着发声组件，每帧读回真实点位。旧生产输入没有启用新登记时沿用历史行为，已有 40 父场景/2,380 题没有被重写。未来场景生产者仍须选择通过摆放、进行多实体场景检查并提供正确声源坐标；本轮不宣称完成生产量产或空间行走路线准入。狗逐帧结果的 `scope` 为 `in_place_animation_admission`，固定根校准同时用于 Idle 与 Walking，不能逐帧补偿。

运行新能力须采用登记 `native_runtime` 的工程/输出根、私有 `ue_env.json`、已获授权且有空余显存的物理卡及本轮 GPU wrapper。不得把新登记的 `/Game/V7KujialeAssets20261012` 路径交给原未导入资产的工程。

## 房间地板门禁

`NativeFloorReference` 以九个保留的原生地板探针测定基准。人的脚底首先碰到的网格面必须是房间 floor section，且与量测地板基准相差 ≤20 mm；床、桌及其他家具面直接拒绝。`NativeRegisteredUEVideo._navigation_foot` 在 UE 事务前检查，初始摆放也执行同一门禁。`reproduce_kujiale_room_floor.py` 和 `capture_kujiale_floor_reproduction.py` 复现 kujiale_0003 的 000004–000007 两个视觉变体，保留既有 20 mm 人物爪底门槛及下沉 10 cm 反例。

## 崩溃与验证

原动物启动的 `-NullRHI -NoShaderCompile` 在编辑器分析模块读取未初始化的 shader 统计 mutex 时断言。差分验证使用同一私有环境；生产修法使用 Vulkan/offscreen、人物生产环境及完整只读 EnginePak/生产 DDC，最终四批导入正常退出。NullRHI 单独退出的另一个堆内存错误根因未验证。

动作读取 commandlet 的快速退出 SIGSEGV 已由 gdb 定位到后台 shader job cache 的 SubmitJobs/PreprocessShader/GetInputHash。`read_kujiale_dog_animation_editor.py` 在返回前通过 `Editor.AsyncAssetCompilationFinishAll` 排空已注册的 shader/texture/mesh 编译；最终排空编译耗时 2,094.672 s，进程退出 0，12 个片段/1,452 个姿态 key 及六只狗的派生材质 usage 均已原生读回；见产物 `evidence/crash_root_summary.json`。不能关闭 JobCache 后忽略新增 shader 编译错误，或跳过崩溃个体。

完整单测的既有缺失数据失败必须如实保留；基准 b38de68 对相同失败节点的只读复测记录在本轮 evidence。相关回归覆盖 glTF 父变换、语义地板、GPU UUID 在单卡可见环境中的编号、原生径向深度、碰撞、固定骨骼/根校准和环境绑定拒绝条件。工具目录索引通过 `tools/build_tool_index.py` 更新。

所有作业使用 nice 10，PID、命令、物理卡和日志在本轮 JOBS.json。每批检查前序任务 PID 和 nvidia-smi XML；新增 ≤60 GB、/data 剩余 ≥150 GB。结束须核验本轮进程全部退出，不做全局进程/环境清理。

最终测试实测：相关回归 31 passed；全量 6,395 passed、119 个原有条件 skipped、0 failed、0 errors、52 subtests passed（590.65 s）。最初 106 个失败/错误节点在基准 b38de68 上也失败；恢复归档数据的只读兼容链接后，这 106 个节点全部通过（67.16 s），部分原先缺失数据导致的条件跳过也恢复为通过。只改变本轮 worktree/tmp 的链接及本轮输出目录，没有修改归档数据、共享环境、测试门槛或跳过规则。旧失败记录与新全量结果均保留，最终结果见 evidence/unit_fixture_restored_final_summary.json。
