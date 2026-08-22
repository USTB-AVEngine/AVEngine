# AVEngine Studio 使用指南

Studio 是引擎的网页规划台：在浏览器的 3D 场景里拖拽声源、绑定声音、
即时校验，然后一键把引擎的完整渲染链（视觉捕获 → 动态双耳音频 → 带声
成片）排进 GPU 队列。所有产物 research_only，正式数据集分母不受影响；
浏览器里的校验都是草稿级预演，渲染链内的原生闸门才是权威。

## 1. 启动与访问

服务器（48g）上启动（通常已在跑）：

```bash
cd /data/jzy/code/AVEngine-lead-a
nohup /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
  tools/studio/run_studio_server.py --config tools/studio/studio_config_48g.json \
  > /data/jzy/tmp/studio_server.log 2>&1 &
```

本机访问（只绑服务器回环，必须走隧道）：

```bash
ssh -L 8765:127.0.0.1:8765 48g-jump
```

然后浏览器打开 `http://127.0.0.1:8765/studio`。首次加载每个房间要下载
网格/贴图/角色模型（进度条会显示），之后走浏览器缓存；界面右上有
"强制刷新"按钮绕过缓存。

## 2. 界面速览

- **场景下拉**：三个房间（MP3D 整屋 / 公寓 UE 生产链 / ReplicaCAD），
  默认加载最轻的。有贴图的房间默认显示真实贴图（"贴图视图"开关可切回
  声学黏土视图）。
- **视角**：自由视角（左键旋转/右键平移/滚轮缩放）、俯视图（全屏平面
  图）、相机第一视角（渲染相机所见，16:9 画幅与成片构图一致）。右上角
  小地图常驻：可行域（绿色）、渲染相机视锥、你当前视角位置（白点）。
- **剖切屋顶**：地板上方 2m 以上隐藏，dollhouse 式看室内。
- **拖拽授权（公寓）**：五个标记（相机锁定 + 人/犬起止点，带文字标签）。
  点标记后拖动；绿色=草稿校验通过，红色=不可放置（不在可行域）。相机
  锁定原因：动态音频与 M1 听者权威做 1e-6 互证。
- **种子授权（MP3D）**：输入 seed + 相机模式 → "生成路线并 3D 预览"
  （原生授权约 1 分钟，坏 seed 会被引擎拒绝——这是防废案，不是故障）→
  轨迹叠加到 3D → "按此路线提交完整渲染"。
- **声源与声音库**：每个声源一行——3D 显隐开关、声音下拉（按端点允许
  类别过滤，狗吻只能选 animal_vocalization 等）、试听（懒加载）。换了
  声音后提交渲染，会自动经引擎作者化一份 hash 绑定的轮流发声
  AudioProgram 并随任务下发。
- **声学线索预演 / 题型预演**：拖动端点实时刷新（草稿启发式，引擎
  QuestionSpec 为准）。
- **时间线**：拖动/播放，人和狗的真实模型沿轨迹移动并朝向运动方向。
- **任务与成片**：队列状态实时刷新；pass 的端到端任务有"播放成片"，
  在页面里直接播带声 mp4。

## 3. HTTP API（curl 可用）

| 端点 | 说明 |
|---|---|
| GET /api/health | 服务状态 |
| GET /api/rooms | 房间注册表（含 studio_status 标注） |
| GET /api/registries[/name] | 端点/声音/实体注册表 |
| GET /api/captures | review 根底片扫描（默认排除偏离 main 的 commit） |
| GET /api/scenes, /api/scenes/<id>/bundle.json | 场景包 |
| POST /api/validate | 草稿放置校验 {room_id, points:[{label, position_m}]} |
| GET /api/sounds, GET /api/sounds/<id>/audio | 声音库列表 / 懒加载音频 |
| POST /api/programs | 作者化 AudioProgram {candidates, sounds:{endpoint: sound_id}} |
| GET/POST /api/tasks | 任务列表 / 提交 {template, overrides} |
| GET /api/tasks/<id>/log?tail=N, /artifact?path=… | 日志尾 / 产物文件 |
| POST /api/sweeps, GET /api/sweeps | 笛卡尔批量提交（≤64 点）/ 批次聚合 |

任务模板：`mp3d_dynamic_audio`、`mp3d_route_author`、`mp3d_end_to_end`、
`apartment_author`、`apartment_end_to_end`、`paired_ablation`（左/右置零、
单声道折叠、静音指定 stem——Left-bias 消融载体）。各模板默认值与可覆盖
键见 GET /api/templates。

## 4. 新房间入列（场景包）

前提：该房间有 M3 声学包（`m3 compile-registered-scene` 可从
room_registry 编译）。然后：

```bash
/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
  tools/studio/build_studio_scene_bundle.py \
  --room-id <room_id> --acoustic-package-manifest <m3包>/manifest.json \
  --navmesh <可选，habitat navmesh> --mesh-occupancy-grid <可选，UE 房间用> \
  --textured-glb <可选，真实贴图 glb> --textured-alignment identity|auto \
  --reference-frame-npy <可选，已过审捕获 rgb.npy> \
  --actor-model name=<glb> --authoring-json <授权默认值> \
  --output /data/avengine_external/studio/scenes/<room_id>
```

导航栅格自检失败会直接拒绝构建（用已知可走点校验栅格朝向）。

UE 房间的真实贴图 glb 用 headless 编辑器导出（服务器有 UE 5.5）：
`tools/studio/ue_export_apartment_gltf.py` 是范例，三个要点：
`-AllowCommandletRendering`（否则烘焙静默失败出全白）、按包围盒过滤
只导房间内 actor（关卡有 150m 天空球）、烘焙 512×512 JPEG（默认
1024 PNG 会出 377MB）。UE 导出坐标系与引擎世界系同构
（H=(U.x,U.z,U.y)/100），用 identity 对齐。

## 5. 常见问题

- **视口黑屏**：大场景在下载（看进度条）；MP3D 网格 65MB。
- **界面像旧版**：点"强制刷新"。
- **MP3D 路线生成失败**："无全程双犬可见相机候选"= 该 seed 被引擎正确
  拒绝，换 seed（22 已验证可行）。
- **提交渲染 400**：override 键不在模板允许清单里，或输入路径不存在，
  错误信息会写明。
- 服务重启后遗留 queued/running 任务标 `interrupted`，不会静默续跑。
