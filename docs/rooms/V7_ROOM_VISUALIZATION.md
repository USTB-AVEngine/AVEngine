# v7 房屋规格可视化

`tools/rooms/visualize_v7_houses.py` 从既有 v7 验收证据生成 owner 审阅页。28 套房屋全部平铺，先列推荐名单，再列待审和不通过；页面可离线打开，图片懒加载，无外部网络依赖。工具不修改验收数据、阈值、原始扫描或既有检出，也不执行声音／视频生产。

`prepare` 读取 `acceptance_manifest_v2.json`、原房间 JSONL 注册表、最终逐屋 RIR 与检查位置、原 CPU 视图相机记录及边界候选。房间轮廓的 2 cm 简化仅用于展示；面积是候选房间面积和。MP3D 读登记原生层级，HM3D 沿选房演示按高度分组，遗漏的错层听者用额外高度切面展示；这些切面不能作为建筑楼层认证。

`render` 顺序启动独立的 Habitat 子进程，每次在指定 GPU 中选择空闲显存最多的一张。启动条件为此前采样峰值加 4096 MiB；首次使用保守的 8192 MiB 峰值预算。每 0.5 秒按本工具创建的 PID 采样显存；若共享卡空闲显存跌到 4096 MiB 以下或超时，只停止该确切子进程。退出前将 PID、命令、GPU、日志、峰值和状态写入输出目录。超时不会停止其他用户的作业。

`render.py` 的原生纹理切顶和矩阵标定，以及 `textures.py` 的 Basis 解码，适配自 `claude/room-selection-mp3d-demo-20261003` 提交 `0465ecde2b932e8ac301b5fed0478ac9d58f194d` 的 `demo_render.py` / `demo_textures.py`。不依赖那个检出的代码；解码器是显式只读外部输入。解码只改图像编码，保留原始 BIN 前缀、几何、UV、材料和场景图。本轮不改既有资产。

俯视图为 1536×1536 的原生正交 RGB，相机保留原生 view / projection 矩阵、量尺及全部验收听者／源位。绿线只代表检查范围，红点是原始网格边界的漏声候选位置。红点可能是合法开口或上下界，不能视为已确认破洞。

内景为 1280×720、85° 水平视野，直接使用验收听者坐标与原检查朝向。2D 声源标记由原生矩阵投影；同位置深度相机用于区分中心遮挡，虚线圈表示被遮挡。未完成声学测量的房屋使用原计划听者坐标，并明确标为“计划验收听者”。这些图片是静态审阅预览，不代表人物或正式 Episode 验收。

`build` 保留最终 `report_final/all_house_results.json` 的状态／原因与 `recommended_houses.json` 的整屋划分。六分析频带优先有效 T30，否则有效 T20；RT60 柱为位置中位数、线为最小／最大。DRR 和距离曲线注明定义；不一致的直达路径仍只供诊断。缺失值显示为未测。ZIP 只包含 HTML 和实际引用的图片，不包含原生 PNG、缓存或数据集。

在任务授权的独立检出运行，显式指定输入、输出、运行时、Basis 解码器及允许使用的 GPU。以下变量由本轮环境提供：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/src"
export TMPDIR="$V7_VIZ_OUTPUT/cache"
# AVENGINE_HABITAT_RUNTIME_PREFIX / AVENGINE_HABITAT_MAGNUM_PYTHON_SITE
# AVENGINE_RLR_SDK_ROOT / AVENGINE_MP3D_ROOT / AVENGINE_DEMO_BASIS_DECODER
python -m tools.rooms.v7_room_viz.cli prepare --input "$V7_INPUT_ROOT" --output "$V7_VIZ_OUTPUT"
python -m tools.rooms.v7_room_viz.cli render --input "$V7_INPUT_ROOT" --output "$V7_VIZ_OUTPUT" --gpus 1 2
python -m tools.rooms.v7_room_viz.cli build --input "$V7_INPUT_ROOT" --output "$V7_VIZ_OUTPUT"
```

输出包含 `page/index.html`、`page/assets/`、`v7_room_viz_20261003.zip`、逐屋矩阵／投影收据 `rendered_houses/`、`evidence/page_build.json`、`evidence/page_data.json` 和 GPU 运行记录。浏览器检查应核对全部图片、28 张卡片、顺序、所有图懒加载，以及桌面／手机文档和内部元素均无横向溢出。
