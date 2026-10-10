# 自动切房漏声检查

工具读取交付目录 `rooms/*.json` 中 `decision == "retain"` 的记录，写新的
`room_results.csv`、`summary.json`、逐房 JSON 和逐屋原生设置。只测漏声，CSV 的“测试”
只代表漏声满足测试门槛；其它声学指标、补洞标签、暴露和房屋划分条件需另验。
输入交付、原始扫描、navmesh、已有声学包只读。

## 原测量出处

- `tools/acoustics/accept_houses.py:spherical_directions/ray_checks/alternate_listener/room_manifest`
  来自 `34146293f66b2cb2610e4c6f1a4a3613b7390423`。
  每个原点 512 条 Fibonacci 球面固定方向，首命中下限 0.01 m，上限
  `max(30 m, 2 * 整屋 AABB 对角线)`。未命中即逃逸；近表面命中 <0.05 m 仅记录。
  AABB 交点只是定位，不声称复原破洞边界。方向和漏声计算不使用随机种子。
- `tools/acoustics/train_room_acceptance.py:measure_room/strict_zero_scene` 来自
  `4664ee66a15f922be90f87f3391cbcfe6866fbba`。
  每房测 `camera_m`、`source_1_m`、`source_2_m`、`camera_alt_m`，取四者最大比例。
  最后一项在相机高度朝第一声源水平移动 0.4 m。清零面积函数原样移植：仅在运行时
  派生数组中移除双精度叉积严格为零的面，不移除任何正面积面，不补洞或裁房屋。
- 原 `tools/rooms/room_selection/navigation.py:sample_navigation/placement` 和当时的
  `thresholds.frozen.yaml`：0.25 m 世界网格、原 navmesh 吸附/楼层/连通性/净空、距离、
  FOV 和三条原扫描视线；相机在导航支撑点上方 1.5 m，两声源上方 1.2 m。
  摆放是确定性最远点搜索，没有随机种子。预算、净空和容差均从原 YAML 读入。
- 原 settings：RLR 8 线程、16 kHz、4 s，direct/indirect/transmission 开启，
  diffraction=false、max_diffraction_order=0、temporal_coherence=false、mesh_simplification=false；
  direct/source rays=500、indirect rays=5000、indirect depth=200、source depth=20，
  两端半径 0、声速 343 m/s。HM3D 材质种子 20260905（MP3D 为 917，本轮只跑 HM3D）。
  RLR 当前 API 不提供原生种子控制；8 线程 RIR 波形不保证逐样本一致。
  漏声固定方向几何查询的旧房三间原生复算和移植复算附有逐点计数/射线编号完全一致证据。
- 冻结门槛：`<=0.05` 测试，`(0.05,0.15]` 只训练，`>0.15` 不用。
  原 classifier、truth/scorer 和门槛未改。房间漏声档不覆盖原测试准入的其它条件。

## 新房取点

对 `floor_polygon_xz_m` 的每个 Polygon 构造 `Polygon(p.exterior)` 再取并集：
填内部小洞，保留外部凹口和不相连部分，不内缩、不取凸包、不平滑、不变换坐标。
只把此范围传给原 navmesh 采样和摆放函数。重新搜索两个声源与相机；交付
`placement_witness` 只用于一次原生 BVH warmup 的初始化姿态，漏声取新搜索出的实际点。

逐房记录填洞前后面积、洞数、每点 membership、外边界距离、navmesh 支撑、吸附误差、
高度和净空。两个声源必须在填洞后的外轮廓内。备用相机始终按原 0.4 m 规则计算；
如果落在外轮廓外，单列事实，不自行修改旧测量原点。

声学使用整屋原包；没有原包时，以相同 HM3D 编译器、材质规则和种子在新输出中
生成整屋包。房间多边形不传入声学网格编译器。每屋只上传一次、warmup 一次，
同屋房间复用上下文。一个进程逐屋执行，RLR 线程数从原 settings 读取。

## 使用

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
nice -n 15 python tools/acoustics/check_split_room_escape.py \
  --delivery /path/to/delivery_all_v3 --output /path/to/new-output \
  --settings /path/to/hm3d_settings.json \
  --thresholds /path/to/thresholds.frozen.yaml \
  --house-inputs /path/to/house_inputs.json
```

`house_inputs.json` 的 `houses` 是 house ID 到输入记录的映射。输入记录含 `house`、
`family`、`scan_id`、`scene_directory`、`semantic_source`、`annotation_source`、
`navmesh_source`、`registry_source` 和可选 `acoustic_package_manifest`。
仓库工具不硬编码私有服务器路径。输出目录必须为新的独立目录，拒绝覆盖既有目录。

省略 `--limit` 就跑全部保留房；`--limit 10` 在声学观察前，按 house ID 的确定性轮转
选跨屋十间。机器包装脚本在本轮产物根 `run_split_acoustics.sh`，给交付目录即可运行。
脚本设 nice=15、单 CPU 进程、BLAS/Numba/OMP=1、RLR 沿用 8 线程，不创建 GPU Simulator。

错误保存 `failure.json` 和 `status=stopped` 的 summary，CSV 只含已测结果，后续房屋停止。
耗时记录取点、射线、每房总时间；CSV 另给均摊整屋初始化时间。全量估算按房屋数计
整屋初始化，再加逐房取点/射线时间，不能只算快速的 ray 查询。
