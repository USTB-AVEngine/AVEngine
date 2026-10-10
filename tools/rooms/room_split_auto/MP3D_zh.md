MP3D 房间切分与去飞地

本工具从 MP3D 准备清单读取准确的语义地面，使用只读 MP3D selection adapter 加载真实 .house / semantic PLY，复用 HM3D v5 切分和已合入的 v6 形状修正。执行顺序为 prepare、existing、render、split、delivery。split 必须先合入提交过的 shape_quality_geometry / shape_quality_repair；existing 可先用精确两块交集连通适配，v6 可用后优先使用同一 PartConnectivity。产物默认独占创建，不覆盖旧产物。

```bash
python -m tools.rooms.room_split_auto.mp3d_run prepare --root "$OUT" --prep "$PREP" --reference-artifacts "$HM3D_ARTIFACTS" --adapter-root "$MP3D_ADAPTER" --workers 4
python -m tools.rooms.room_split_auto.mp3d_run existing --root "$OUT" --workers 4
python -m tools.rooms.room_split_auto.mp3d_render --root "$OUT" --workers 2
python -m tools.rooms.room_split_auto.mp3d_run split --root "$OUT" --workers 4
```

使用安装的 Habitat runtime loader 读取原 navmesh，仅 CPU、nice 10–15、单线程 BLAS。每计算进程 RLIMIT_AS 10 GiB，4 worker 加父进程最多 50 GiB；使用 fork 避免独立资源跟踪进程。每大房的有限构造预算 600 秒，不能构造或缺少资源时保存完整原多边形为 unresolved。工作树、产物路径、只读参考目录都由命令参数或本次产物 plan.json 给出。

原有 ≤35 m² 房间不切内部窄口、不增加空地圆/形状/可见性/黑区淘汰条件。仅将原地面拆成直接相连的候选组，小飞地丢弃；原有 <6 m² 门槛例外及未找到见证的房间另列待定，不改旧名单。

每间保留房的相机与两个声源在各原始 Polygon 填洞 exterior 内，距真实外边界至少 0.25 m；导航桥不参与形状或放置。原 navmesh 支撑点、三条原始整屋 mesh 射线均单独验证。导航桥只用两块的 0.3 m 膨胀交集与 navmesh 的交，排除其他语义房间的地面。

俯视图沿用 MP3D 准备任务的整屋真实 UV 纹理 CPU raster，只使用当前层剖面显露地面；输出 JPEG 及 owner 重画工具的 projection/view 元数据。原生 Habitat RGB 等价性未验证。新切房原黑区测量读取只读参考原像素；图像显示使用 JPEG。

名单是草稿。漏声未重测，leakage 列留空；≤5% 测试、5–15% 只训练、>15% 不用的政策记录在草稿，不能用草稿自动宣布生产准入。
