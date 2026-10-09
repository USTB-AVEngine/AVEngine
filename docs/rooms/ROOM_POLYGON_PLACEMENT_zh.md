# 房间外轮廓取点

房间记录可提供世界 X/Z 米制的 `floor_polygon_xz_m`，类型为 GeoJSON
Polygon 或 MultiPolygon。规划只使用每个 Polygon 的 exterior，填平扫描小洞；
原始字段中的洞仍保留在清单里。整屋视觉资源、声学包和原 navmesh 不裁剪、不重建。

`room_polygon_placement` 可在房间记录或请求中配置：

- `boundary_margin_m`：取点及路线起终点距外轮廓边界，缺省 0.25 m。
- `path_sample_step_m`：检查路线顶点，并插入间距不超过此值的点，缺省 0.25 m。
- `path_tolerance_m`：路线检查不收边；允许 navmesh 在外轮廓外最多 0.1 m 的数值/定位差。
- `max_point_attempts` / `max_route_attempts`：缺省 512 / 32；耗尽时明确失败。

`constrain_space_to_room(space, room, request)` 是规划空间的入口。
`load_planning_resources`、`solve_conditioned_episode`、
`build_conditioned_plan` 和 `sample_routes` 已接入；
相机网格、点云、保留路线库及最终截短/接触修正后的路径都经过检查。
路线中间位置不要求 0.25 m 收边。可站立性仍由已有导航判断。
没有该字段的记录返回原空间对象，保留原随机数流与输出。

生成新清单（不覆盖已存在的输出目录）：

```bash
python tools/rooms/build_polygon_room_manifest.py \
  --original-rooms ORIGINAL_ROOMS.json \
  --split-rooms DELIVERY/rooms \
  --output-dir NEW_DIRECTORY
```

输出 `rooms/*.json`、`manifest.json` 和 `summary.json`。
只添加 `decision=retain` 的交付记录；原房间已有多边形时原样携带。
原清单可以是目录、JSON 数组、含 rooms/records 的 JSON 或 JSONL。
单纯交付记录生成的是几何清单，不宣称运行资源已经注册。

如需携带既有生产资源引用的 RoomPackage catalog，输入既有
`avengine_qa_room_package_catalog_v1`，并提供 `--template-map MAP.json`：
显式映射「新房间 ID 或 house ID → 同屋已有 catalog room_id」。
新条目复制该模板的整屋视觉/声学资源，仅改变房间 ID、楼层和取点形状；
映射缺项时失败。不要把几何清单直接当作已核准的生产 CCI 名单。
CCI 打包、既有任务的 profile/清单切换和新房准入应另行执行。

带多边形的记录需要 Shapely >= 2（已有 room-screening / room-selection 可选依赖）。
旧记录不导入 Shapely。V7 的已测静止布局、新生成人/狗行走路线及坐姿根位置
也使用该可选外轮廓；旧布局中的相机若不满足边距会明确拒绝，需在新房重测。
本改动不改变既有物理接触、原生导航、像素、声学、题目或判分门槛。
