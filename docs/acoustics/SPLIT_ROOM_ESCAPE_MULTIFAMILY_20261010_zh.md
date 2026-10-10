# 切房漏声：HM3D、MP3D、酷家乐与按房屋并行

在 `5b332ef` 的 CPU 工具上扩展两类房屋与 `--jobs K`。声学测法不变：整套原房屋网格、4 个原点、每原点 512 条固定球面射线、取最大逃逸比例、每个 RLR 上下文 8 线程。分档仍为 ≤0.05 测试、(0.05,0.15] 只训练、>0.15 不用。这里只判漏声，不替代完整准入。

## 测量与坐标来源

- `room_split_escape.py` 保留 `accept_houses.py`（`3414629`）的球面方向、射线、第二听者和编译输入，以及 `train_room_acceptance.py`（`4664ee6`）的严格零面积清理。原点为 camera/source1/source2，第四个沿 camera→source1 水平移动 0.4 m，保持听者高度。原来的 RLR simulation/runtime/radii 配置由 `--settings` 输入；每进程仍为 8 线程。多线程 RIR 波形本身不承诺逐样本相同，几何射线复现要求差值为 0。
- HM3D 输入逐条原样保留，沿用原编译器与材料种子 20260905。
- MP3D 优先读取旧缓存/结果所指的整屋包，再搜索给定 measurement 根目录；没有包时调用 `compile_mp3d_semantic_research_scene`，沿用设置中的原材质表与种子 917。原始 Z-up 网格的旧适配器为 `(x,z,-y)`；放置仍使用原 `.glb` 和 `.navmesh`。
- 酷家乐读取历史声学结果中的 `package_rlr` 和原 placement navmesh。旧放置的 `surface/vertices.npy`、`triangles.npy` 已在规范坐标中，逐字节核对保留的原包 arrays 描述（已有 SHA256/byte_size），并核对 `geometry.source_to_canonical.matrix_row_major`、规范坐标声明；这些数组已经是该矩阵的烘焙输出，因此不再做轴变换。放置保留旧未过滤全网格，RLR 使用历史兼容包，两者不能混用。
- 部分历史预览 USD/未过滤包数组在记录的原路径已缺失。CPU 运行不读取预览 USD；缺失的历史 USD 路径单列在 `historical_provenance_files`，必需文件校验只覆盖实际 CPU 输入。未过滤数组的存留同一内容副本是原 placement 缓存，用原包描述验证。没有补造或重新展开 USD。
- 特例 `kujiale_0042`：旧工具在没有不兼容三角形时直接返回原包，未建 `_rlr` 目录。输入生成器检查零面积和原生不兼容三角形均为 0，才允许在指定新产物根下建立 `package_rlr` 软链及 receipt；几何/材质不变。不是对含坏面的包改名放行。

新房间使用每个 Polygon 的 exterior 并集判断成员，不收边；navmesh、0.25 m 网格、搜索预算、净空、高度和三条遮挡射线仍沿用原放置规则。交付里的旧放置见证用于整屋初始化，新房间的实际测量见证由原搜索重新取点。每间 JSON 保存多边形成员检查、navmesh 支持、高度和四个原点的完整射线结果。

## 命令

所有数据、SDK、runtime、旧交付根通过显式参数给出。先生成输入表（输出与 alias 根必须为新路径）：

```bash
python tools/acoustics/prepare_split_house_inputs.py \
  --hm3d-inputs "$HM3D_INPUTS_V3" \
  --mp3d-rooms "$MP3D_TRAINING_ROOM_CSV" \
  --mp3d-data-root "$MP3D_DATA_ROOT" \
  --measurement-root "$ORIGINAL_MEASUREMENT_ROOT" \
  --kujiale-root "$KUJIALE_ACCEPTANCE_ROOT" \
  --kujiale-package-alias-root "$NEW_ALIAS_ROOT" \
  --output "$HOUSE_INPUTS_ALL" --validation-output "$NEW_VALIDATION_JSON"
```

测量某个交付目录：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python tools/acoustics/check_split_room_escape.py \
  --delivery "$DELIVERY" --output "$NEW_OUTPUT" --jobs 4 \
  --settings "$ORIGINAL_SETTINGS" --thresholds "$FROZEN_PLACEMENT_THRESHOLDS" \
  --house-inputs "$HOUSE_INPUTS_ALL"
```

入口提升自身 nice 至至少 15；外层已有更低优先级时不提高优先级。`--jobs` 默认为 1，并行进程数最多为房屋数。同一房屋全部房间在一个进程中，只初始化一次；每个测量进程 8 个 RLR 线程。协调进程不初始化 RLR。运行前由操作者查看共享机器负载。本次原生并行试跑最多 4 个测量进程、nice 15。

未知房屋在创建输出之前报错，列出缺失 ID 与完整输入表生成工具，不再抛没有上下文的 KeyError。不在原有输入表或结果目录内写入。

## 输出与并行

串行与并行都有 `rooms/`、`houses/`、`room_results.csv`、`summary.json`，另有 `logs/worker_NNN.log`、`workers/result_NNN.json`。并行模式还记录进程 PID、房屋分配、退出码。CSV 统一按房间 ID 字典序合并，完成顺序不影响行序。每个 worker 的原生 stdout/stderr 都写入其日志。worker 捕获的 Python 异常会留下完整原因，并通过错误标记阻止其他 worker 开始下一套房屋；已开始的房屋完成后结束。父进程也收集原生非零退出状态，最终以失败状态交付。

`summary.family_timings` 分别报告各类房屋初始化（含原始放置网格加载）、逐间测量（放置 + 射线）、射线与放置均值。`houses/*/placement_setup.json` 保存坐标校验与具体耗时，CSV 分摊初始化时间。初始化/逐间时间只用于预算，不改变测量和分档。

## 本次验收依据

本次独立原生复现和扩展入口复现各 9 间：

|家族|房屋/房间|历史值|扩展入口值|差值|
|---|---|---:|---:|---:|
|HM3D|hm3d_train_00022_gmuS7Wgsbrx/R0|0.00390625|0.00390625|0|
|HM3D|hm3d_train_00022_gmuS7Wgsbrx/R7|0.05859375|0.05859375|0|
|HM3D|hm3d_train_00081_5biL7VEkByM/R0|0.22265625|0.22265625|0|
|MP3D|mp3d_17DRP5sb8fy/R0|0.005859375|0.005859375|0|
|MP3D|mp3d_1pXnuDYAj8r/R17|0.05078125|0.05078125|0|
|MP3D|mp3d_1pXnuDYAj8r/R16|0.189453125|0.189453125|0|
|酷家乐|kujiale_0020/R0|0|0|0|
|酷家乐|kujiale_0024/R0|0|0|0|
|酷家乐|kujiale_0034/R0|0|0|0|

酷家乐旧验收在采样多个房间时每间只保存 3 个原点，在只采样一个房间时保存 4 个；上述选择均有完整 4 原点旧记录，不改变旧数口径。三个家族共 36 个原点的坐标、逃逸计数和逃逸射线方向编号全部相同。

外部任务产物中的 `reproduction_tool_v2/comparison.csv` / `summary.json` 是复现依据；`house_inputs_all_v1.validation.json` 记录 163/47/25 套与 4,006 项必需文件检查。正式 MP3D/酷家乐切分尚未交付，本轮只测从历史真实多边形和见证拼装的临时交付，不等待正式任务。

验收完成：MP3D 10 间 jobs=1 / jobs=4 的值、实际见证和全部 40 原点射线记录完全相同；两次均为测试 10 间，耗时 449.08 / 180.80 秒。酷家乐 5 间全部跑通，均为测试档，耗时 109.32 秒；HM3D 3 间回归与上一轮的完整射线记录相同，耗时 66.39 秒。专用快速单测 14 通过；完整 fast_unit 285 通过、36 因旧保留事实缺失跳过。

本次样本初始化/逐间均值（秒）：HM3D 16.52 / 5.53，MP3D 串行 106.42 / 2.27，酷家乐 30.43 / 19.41。逐间包含原放置搜索及首间放置索引构建，不是纯射线时间；原点射线本身均值约 0.01 秒。按样本粗估 4 进程时间为 `(房屋数 × 初始化均值 + 房间数 × 逐间均值) / 4`，还需考虑进程启动、房屋分配不均与共享机器负载；大规模正式切分耗时未验证。

MP3D 还原生验证了缺包编译路径：17DRP5sb8fy/R0，以原材料 seed 917 重编译，和旧包的漏声值 `0.005859375`、实际见证、四原点完整射线记录均一致；带编译初始化 171.32 秒，整间端到端 173.45 秒。这是单套编译样本，不能当全部 47 套的编译均值。

完整外部证据：`parallel_consistency_v1.csv`、`trial_verification_v1.json`、`cold_compile_consistency_v1.json` 和各 pilot 的 `room_results.csv` / `summary.json`。旧包及输入仅被读取，新增 0042 别名和本次临时交付均写入新产物目录。
