房间审核采用可复算自动建议、独立人工判断和房子级门禁。自动结论不改正式审核章，不删除任何输入。所有数值参数、来源与理由在 `thresholds.yaml`；这是任务 R 的初始规范，不能称行业标准。

从 AVEngine worktree 运行（部署路径由调用者传入）：

```bash
python -m tools.rooms.room_selection.run all \
  --inventory INPUT_ASSETS_INVENTORY.json \
  --output NEW_OUTPUT_DIRECTORY \
  --manual-splits READONLY_MANUAL_REFERENCE.json \
  --clean-houses PRECOMPUTED_CLEAN_HOUSES.json \
  --workers 2
```

`register` 单独全量登记，`measure --resume` 只补缺失房子，`summarize` 从已有 measurements 复算统计。运行设置应包含 `CUDA_VISIBLE_DEVICES=''`，以及 BLAS/OMP 单线程。PathFinder 与 trimesh CPU rays 不创建 Habitat Simulator。环境新建在任务允许的私有目录；几何依赖版本见 `requirements.txt`，基础 Habitat 安装只读。

- 阶段 0 以语义 annotation 所有 region 为准，包括缺地面、旧 `rooms.json` 没有的 region，以及单列的 -1 未分配桶。旧审核记录与映射缺失不会静默消失。
- 阶段 1 测量同层语义地面并集、家具实例投影、主导航连通区、净空、矩阵校准的俯视图黑像素和三点摆放证据。楼层分开记录；多层原 region 进入人工复核。有限搜索没找到证据不等于不存在。已有 navmesh 无来源文件时参数记未知。
- 阶段 2 以家具簇附近导航点为种子，测地距离约束的净空加权最小割优先选择窄通道。无法找到足够家具种子则输出 review。每块重新跑阶段 1；未分配的非导航地面面积另列。没有显式门拓扑时，不声称切线已落在真正门洞。
- 阶段 3 保留完整队列、固定理由、第二人抽样与浏览器导出。第二人模式隐藏第一次 verdict 和自动建议。所有新判断保存在独立 JSON；合议后由 owner 决定如何导入正式章。切分子块不能继承原标签。
- 阶段 4 读取每栋房子最新的 e2e task.json，房间质量和房门禁分别保存。

房子级 calibration/holdout 名单在任何一致性分析前固定。calibration 独立做阈值敏感性；holdout 不参与选阈值。二元评价是 use 对 skip，unsure 单列，自动 review 按未选中计算且额外报告覆盖和条件一致率。没有第二人实际标签时，不输出 kappa 或双人一致率。

手工 43 个结果没有 mask 真值，9 个 bbox 未收窄。`import_manual_splits` 只调用授权的 `sudo -n find/cat` 读取 JSON，复制到任务产物目录；`split_iou.json` 明确标成 semantic-floor-clipped-bbox 代理 IoU，而非精确真值。Hungarian 一对一匹配；有效但没有自动建议的项计 0，未完成框单列排除。

审核入口（需调用者选一个空闲端口）：

```bash
python -m tools.rooms.room_selection.serve_review \
  --root OUTPUT_DIRECTORY --media-root EXISTING_ROOM_TOUR_ROOT --port FREE_PORT
```

打开 `/review.html`；先填审核人姓名，再按角色独立判断并导出 JSON。第一审核模式可逐项判断切分子块；第二人模式隐藏全部自动建议和子块。浏览器草稿按本轮输出隔离，并按姓名和角色保存，导出只含当前审核人的判断。第二人只审 `second_reviewer_sample.json` 固定名单，记录纳入概率供总体统计使用。viewer 只提供读取接口，本轮不启动常驻审核服务。

MP3D 三套试跑先用 `register --family mp3d --limit 3` 登记，再用 `measure --family mp3d --skip-splitting` 运行阶段 1。显式跳过的阶段 2 标记为 `not_run`，不会产生切分建议。

MP3D 原生 `.house` 的 R/L/O/C 关联加 `_semantic.ply` 的逐面 object_id 可以复用几何、导航和摆放流程。原始坐标仍需 x,y,z→x,z,-y。若无矩阵校准的同等俯视图，扫描质量记未知，不能直接完整自动通过。格式出处：[Matterport 官方说明](https://github.com/niessner/Matterport/blob/master/data_organization.md)。

第二人抽样总体是已有第一次章的 1382 个 region；固定 180 个样本占 13.02%。其余 1217 个登记区域先补第一次审核。MP3D 逐面 object_id 与 house 中对象归属的 region 可接入流程，但跨房间对象可能越出 R 框；原生 R 框不是精确房间多边形。pilot 同时报告边界诊断和扫描质量缺失，不能直接宣布完整通过。

切分图优先叠加真实扫描俯视图；素材缺失时输出明确标注 GEOMETRY ONLY 的 CPU 地面/子块示意。示意图不是扫描 RGB，不能用于黑像素比例或解除扫描质量未知；真实底图和首次人工审核仍需补齐。
