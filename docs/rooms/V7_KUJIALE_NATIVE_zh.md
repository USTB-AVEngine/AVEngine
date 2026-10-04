当前结果以 [第7回合原材质恢复与全量房源准入](V7_KUJIALE_NATIVE_ROUND7_zh.md) 为准；本页保留两套先期试跑记录。

# Kujiale 原生 UE/SPEAR 验收与连续试跑

Kujiale 的视觉路线显式指定 `input.visual_backend = ue_spear`，复用 v6
SPEAR 相机、资产注册、演员和动画驱动。未指定后端的旧 v7 输入保持原有
Habitat 路由；未知后端或 UE 启动失败都会报错，不会替换渲染器。
本入口产出可检查的研究候选，正式准入状态继续由同一 v7 生成器负责。

## 隔离导入和预热

从可读运行时闭包复制单独的 SpearSim 项目；Content 不可指向共享可写目录。
先单独预热本项目 DDC，再连接 SPEAR RPC。UE 5.5 的后台着色器编译保持开启；
CPU Python commandlet 不带 `-AllowCommandletRendering`，避免尚未完成的
着色器任务在退出时触发断言。DDC、UserDir、XDG、TMP 全部在运行目录。
`native_local_network_only.c` 是本次运行的进程级 LD_PRELOAD 配置，仅允许
本机 RPC，不修改服务器网络或系统配置。

`tools/ue/import_v7_kujiale_editor.py` 使用 `build/native_ue_pending` 的请求，
导出真正的渲染 LOD0 顶点、面和材质分区。空 STL 不能证明几何一致。
`bake_kujiale_native_usd.py` 为导入器变换问题提供派生 USD：计算既有世界
坐标，清除已经烘焙的祖先变换，保留拓扑、UV、法线和材质。每个 mesh
用 USD 的原变换作独立数值核对；不移动、拟合或补齐房屋。

## 自动房间检查

依次使用以下入口，命令参数用本次运行目录中的具体请求/结果：

1. `prepare_v7_kujiale_ue.py` 复用冻结 HM3D 房间采样和摆位规则，产生
   单个俯视图和多个合法相机候选。
2. `check_v7_kujiale_ue.py` 在 UE 实渲 2560×1440、交付 1280×720、85°、
   15 FPS。保留所有失败视角，从预先声明的候选中取三个通过的视角。
   黑像素沿用 RGB 三通道都小于 8、占比不超过 0.15 的规则。记录过曝、
   缺图、默认材质、玻璃和透明面的异常占比。
3. `finalize_v7_kujiale_ue.py` 对保留的每个低分辨率原生深度像素投射
   实际 LOD0 三角网格射线，检查缺面、穿面和材质绑定。UE 正交深度着色器
   存储到固定相机位置的欧氏距离，不能当成到平行射线起点的距离。
4. 低相机向下测原生地板深度，三个摆位×相机/两个声源共九个探针，写入
   `floor_reference`；UCapsule 对准该实测支撑面。声学与视觉几何还要通过
   顶点集合一致性和三个摆位各 26 方向的自动交点断言（0.5 mm 数值容差，
   不作拟合）。

灰色像素本身不是缺图证明：由实际 UE 材质身份、USD 预期贴图和绑定核对。
记录的透明面仍不透明比例不能描述为玻璃已正常。CPU 几何或 Python 单测
通过均不能代替 UE 像素检查。

## 原生声源材质和动态接触

`audit_v7_kujiale_native_assets_editor.py` 解析资产注册中的 UE 绑定并读取
实际 MIC 参数。Habitat GLB 与 UE uasset 是独立资源。
`prepare_v7_kujiale_native_materials.py` 按精确材质名映射既有派生 GLB 的
PBR 修复，只提取派生 MR 图片。
`repair_v7_kujiale_native_materials_editor.py` 只改隔离项目内的 MIC，备份
改前 uasset，复制 metallic/roughness 和线性 MR 贴图，读回数值，保持
BaseColor、UV、法线、网格和资产 ID。UE 5.5 MaterialEditingLibrary 的几个
setter 返回值始终为 false，使用实际读回确认结果。引擎父材质不修改。

`check_v7_kujiale_native_assets.py` 对注册资产逐个做三个原生视角，并以
同视角 BaseColor 与原 UV 色板检查颜色，用既有 `source_materials` 指标
检查高光。失败资产明确留在失败名单，不用色调变化隐藏错误。
连续视频每帧读回演员、发声组件和动画状态，检查根位置、原生遮挡 mask
及身体最低点；脚被生产相机截断时，另设完整身体深度相机测量，不能把
截断当成接触通过。2 cm 是显式记录的深度栅格诊断容差，不改变冻结选房
门槛或既有全动作接触的正式准入要求。
原生全场景/单实例 show-only 深度直接复用 v6 的
`compile_depth_pixel_visibility_truth`；相机实际位姿、演员根和停止后的
动画时间在两个 pass 之间自动读回核对。按交付分辨率的相同最近邻深度
采样生成可见比例，不用 Habitat 标签描述 UE RGB。
新生成器需要的 `source_materials.json`、`source_material_checks.json`、
`native_visibility_evidence.json` 也一起输出；外观、接触或其他正式阻断
失败仍不会变成训练数据。

## v7 连续场景与 CPU 预检

`prepare_v7_kujiale_native_generation.py` 只接受实测房间，生成训练输入。
`preflight_v7_kujiale_native.py` 检查 unchanged seed 所有 450 帧×两个声源
的原生导航与冻结 15 cm 净空。静态合法端点之间可能经过家具；不通过时，
`prepare_v7_kujiale_native_motion_plan.py` 枚举同一冻结候选集合、同一三条
原网格视线、距离与夹角规则，追加全轨迹拒绝，再重新实渲新摆位。
不会将轨迹投影成不同路径，也不会改变事件或真值。

`tools/timeline/pilot_v7_kujiale_ue.py` 运行 30 秒场景，经原生视频、既有 RLR
FOA、两个 10 秒窗口和现有问题生成器，得到与其他数据集相同的六文件窗口：
`video.mp4`、`audio_foa.wav`、`audio_mono.wav`、`questions.json`、
`private_answers.json`、`evidence.json`。正式生成器阻断项原样保留。
本次试跑传 `compact=False`，不删除中间产物。

## 资源监控与复现

每项 UE 命令用 `v7_kujiale_ue_guard.py --request REQUEST.json` 包装。
request 指定 argv、cwd、隔离 environment、budget_root、run_directory、
gpu_index、reserve_bytes 和 previous_peak_mib。首次需空闲显存 16 GiB；
后续需实测峰值加 4 GiB。每秒读取 `nvidia-smi -q -x`，统计精确进程树
中的图形和计算进程；同时记录进程 CPU user/system 和目录逻辑大小。
CPU 是每秒采样的下界，可能漏掉短命子进程和最后一秒，不能冒称精确总数。

guard 的 `launch.json`/`result.json` 保存 PID、创建时间和命令。
停止时只终止身份核对过的本任务进程；SIGTERM 发给 guard 即触发此清理。
磁盘超过 50 GB 减预留余量时停止，不删除别人或本任务历史结果。
实际时间、显存、机器和证据写在运行目录的 REPORT_zh.md 与 PROGRESS_zh.md，
不将 UE、原始数据或生成媒体提交到 Git。


本次两套实测、失败限制及阶段提交见 [原生试点结果](V7_KUJIALE_NATIVE_RESULTS_20261003_zh.md)。
