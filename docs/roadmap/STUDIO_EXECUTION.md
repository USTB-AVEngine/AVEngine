# AVEngine Studio 执行计划（owner 批准版）

> 权威文档。任何会话（含压缩续跑）从这里 + 文末检查点接续。
> 服务器：`ssh 48g-jump`，仓库 `/data/jzy/code/AVEngine-lead-a`（`/data/jzy/code/AVEngine` 是指向它的符号链接）。
> 铁律见根 `AGENTS.md`；引擎现状见 `docs/roadmap/CURRENT_APARTMENT_EXECUTION.md` 检查点 20260821f–20260822c。

## 0. 定位与授权边界

Studio 是引擎的**网页规划台**：选房间 → 从声源库拖角色 → 从声音库绑声音 →
画起止点 → 浏览器即时校验 → GPU 队列用引擎一条龙渲染 → 带声视频回传。
它是单仓内的新模块（`src/avengine/studio/` + `tools/studio/`），只调用引擎
自己的 Python API/CLI，不引入任何外部仓库——严格处于"引擎大包大揽"边界内。

真实时渲染成片在当前链（离线捕获 + RLR 真实声线追踪）不可行也不追求；
分层为：**毫秒级**浏览器校验/预演 → **可选 draft 档**快速预览 → **分钟级**
GPU 队列正式渲染（与审核视频同链、fresh/no-clobber、带 manifest）。

授权边界（owner 已确认的工作方式）：可自主编码、跑 GPU 渲染、commit 并
push 到 `feature/studio-v1` 分支；`main` 保持稳定基线，每个 Session 的
demo 经 owner 验收后才考虑合入。研究边界照旧：research_only，正式数据集
分母保持 0，不新增 hash/gate。

## 1. v1 · 四个 Session 切片（每片一个可点开的验收 demo）

### Session 1 — 后端骨架
- 轻 HTTP 服务（优先标准库 `http.server`/`ThreadingHTTPServer`；若引入
  FastAPI+uvicorn 则加 pyproject `studio` extra，勿动核心依赖）。
- 只读 API：房间列表（room_registry）、声源/端点/声音注册表、可用底片
  捕获列表（扫外置 review 根的 receipt，只认主分支 commit 的捕获）。
- 渲染任务队列：subprocess 包装现有 CLI verb / tools，nohup 风格落日志，
  任务状态查询 API（queued/running/pass/fail + receipt 路径）。
- 服务只绑服务器本机端口；owner 用 `ssh -L` 隧道预览，不暴露公网。
- 验收：`curl` 能列出注册表；提交一个 MP3D dynamic-audio 任务全程跑通。

### Session 2 — 俯视画布 v0
- 单文件静态前端（HTML+原生 JS，无构建链），底图用引擎 Topdown 渲染或
  navmesh/障碍图栅格（`RuntimeObstacleMap`，`avengine.m6x.geometry`；
  topdown 参考 `avengine.m6x.topdown.render_runtime_topdown_frames`）。
- 拖放两个声源、画起止点、相机 marker；后端 `/validate` 调 navmesh 可达
  与净空校验（参考 `qualify_fixed_apartment`，`avengine.m6x.apartment`）。
- 验收：拖到不可达点即时标红，合法路径显示插值轨迹。

### Session 3 — 端到端首片（Studio 的 Hello World）
- 画布"提交渲染"→ 生成 authoring 参数（先支持 MP3D two-beagle：seed、
  `--camera-selection`、起止点）→ 队列串引擎链：
  `m5 author-current-mp3d-two-beagle-route` → `m5 capture-current-visual`
  → `m5 render-current-mp3d-dynamic-audio` →
  `tools/m5/build_current_mp3d_dynamic_review_clip.py` → 页面播放回传 mp4。
- 验收：网页上从拖拽到播放一条全新带声片。

### Session 4 — 防废案双件套（论文价值最高）
- 拖拽时实时声学线索：客户端逐帧方位角/角分离度/遮挡近似（对障碍图做
  2D 射线），轨迹色带（左蓝右红、遮挡斜纹）+ 警示（"全程方位变化仅 X°"
  "两源夹角 <20° 且同时发声"）。
- QuestionSpec 预演器：只用几何/轨迹事实表做干推导（复用
  `avengine.qa.question_spec.evaluate_question_specs` 的可离线子集），实时
  列出可产出题型与拒答落点；拖动端点即时刷新。
- 验收：移动端点时题型清单与警示实时变化。

## 2. v1.5 · 生产工具（1000-episode 之前）

按 owner 已过目的构思清单（工程量标注见原审计文档），依次：
1. 难度旋钮与逐帧难度仪表（双源间距/角分离/遮挡占比/SNR/移速 → 难度向量
   入 manifest + 桶标签；可反向锁约束取点）。
2. 拒答/负例逆向构造器（把目标标签的判定条件翻成画布上的几何可行走廊）。
3. 配额驱动 Sweep 编排器（模板 + 可扫维度 + 每桶配额 → 预演过滤 → 确定性
   seed 作业清单入队，batch manifest 汇总）。
4. 成对消融生成（左声道置零/单声道折叠/静音一源等末端变换，共享 seed 与
   轨迹，manifest 记 pair_id——Left-bias 消融的复活载体）。
5. 数据集偏置看板（题型×答案边际分布、桶热力图、覆盖矩阵，失衡红线一键
   转补采作业）。
6. A/B 重渲对比审听台（父子版本谱系；画面同步、声音热切换；spec diff +
   左右能量差曲线）。
7. 键盘流审片队列 + 盲听模式（结构化通过/驳回回写 manifest；先听后揭示
   对照真值 = 论文 human verification 素材）+ 时间轴锚定批注回填。
8. 变体矩阵批量生成（对调/镜像/换素材/换房间的笛卡尔积裂变）。
9. 声音库 rights 硬闸门 + 试听卡片（默认仅 rights-clean 可拖；未清权可
   draft 但硬阻断正式队列；LUFS 响度差 >6dB 警示）。
10. 草稿档快速预览（前 1.5s、降清、降反射阶、仅双耳；通过后同 spec 升
    全量）与相机/种子画廊挑选器（每候选 3 关键帧缩略图）。

## 3. v2 · 发布与验证工具（投稿冲刺期）

1. Episode 证据链浏览器（全输入 hash/commit/seed/配置展开，任意两条 diff）。
2. 一键复现命令 + 哈希复验（重跑比对新旧输出 hash，全绿即逐比特可复现）。
3. Supplementary 包一键导出（确定性打包 + 自动 ATTRIBUTIONS；未过审/非
   clean 条目硬拦截）。
4. 审稿人匿名 Demo 页生成器（桶×题型代表条目，QA 自测后揭示，拒答专区，
   全静态零署名）。
5. ABX 双盲听测模块（双耳 vs 单声道折叠、原始 vs 镜像；外部听测者链接，
   正确率显著高于随机 = 可量化感知 claim）。
6. 数据集冻结与发布清单（四灯全绿才可入集；冻结出数据集级 Merkle 根 +
   版本号，可印进论文）。
7. 渲染失败与质量自诊断面板（硬失败归类 + 软警告：声道长静音/clipping/
   左右能量差过小/双源频谱重叠过高——本身即论文可引用的自动化 QC 指标）。

## 4. 引擎接缝速查（实现时直接引用，全部已在 main）

- 房间无关音频核心：`render_dynamic_research_audio`
  （`src/avengine/m5/current_mp3d_dynamic_audio.py`）；输入=每源 [75,3] 世界
  轨迹 + 静态听者位姿 + M3 声学包 manifest + AudioProgram + 注册表 + HRTF。
  同文件 `load_captured_source_paths`（frame_records `source_positions_m`
  槽位=program 候选序）、`listener_pose_from_m1_request`。
- 公寓 UE 记录翻译：`avengine.m7.apartment_dynamic_audio`
  （`U = 100*(H.x, H.z, H.y)` 反算；嘴/吻高 1.63/0.45；相机需与 M1 权威
  1e-6 互证）。UE 底片出片必须 `--channel-order bgr`。
- 干声总线：`assemble_audio_program_dry_buses`（`m6/audio_render.py`）；
  program 作者化用 `bind_audio_program_hash` + `validate_audio_program`
  （`m6/audio_program.py`）；现成模板
  `examples/m5/current_mp3d/audio_programs/` 与
  `examples/m7/current_apartment/audio_programs/`。
- 逐状态 RIR 三件套：`build_strided_review_keyframes` /
  `render_research_review_binaural_rir_sequence` /
  `render_research_review_binaural_audio`（`m5_1/acoustics.py`）。
- 出片：`encode_profiled_h264_base_video` + `mux_profiled_binaural_wav`
  （`m6x/visual_profile.py`）；封装工具
  `tools/m5/build_current_mp3d_dynamic_review_clip.py`。
- 动画守卫：`avengine.m7.animation_probe` +
  `tools/qa/probe_ue_capture_animation.py`（滑行 ≤7.5 / 动画 ≥9.5 /
  中间带 inconclusive；静态相机前提）。底片只认主分支 commit 的捕获。
- 注册表：`examples/m6/registries/{source_endpoints,sound_assets,entity_assets}_v1.json`、
  `examples/m6/rooms/room_registry.json`；simulation request：
  `examples/runtime/rir_cache_simulation_request_v2.json`。
- 外置输入根：`/data/avengine_external/`（runtime-prefixes、rlr-sdk 及其
  hrtf/、datasets、m6x-canary-inputs、spear-host-sdk、review）。原生执行
  环境三件套 + `AVENGINE_MP3D_ROOT`；重建配方
  `docs/provenance/RUNTIME_PREFIX_RECIPE.md`。
- conda：`/data/jzy/miniconda3/envs/avengine-habitat-runtime`（py312，
  avengine editable → lead-a）。

## 5. 工程纪律（owner 确认）

- 分支 `feature/studio-v1`；`main` 稳定，demo 验收后合入（owner 可直推，
  规则 bypass 已生效；push 不再需要 `--no-verify`）。
- 每 Session 结束：更新本文档检查点 + 给 owner 可点开的 demo/页面。
- fresh/no-clobber；receipt 记 research_only / episode_counted=false /
  formal_dataset_count=0；不新增 hash/frozen-contract/gate。
- 服务器长任务用 `nohup … > log &` + 轮询（长 ssh 管道会假死）；复杂编辑
  写本地脚本 scp 执行（heredoc 转义易炸）；测试 `$CONDA_PY -m pytest
  tests/unit -q`；CLI 入口 `python -m avengine.cli`。
- ultracode 仅在理解/设计类环节使用；常规实现不用。
- Blender-custom 房间永久禁用；Skokloster 不进生产。

## 6. 续跑方式

压缩后（或新会话）只需一句：**"读 docs/roadmap/STUDIO_EXECUTION.md，
继续 Studio Session N"**。执行者先读本文档与文末最新检查点，再动手。

## 检查点

### Checkpoint 20260822-S0：计划落库
Owner 批准 v1 四切片 + v1.5/v2 分期与工程纪律并要求写入本文档；尚未开始
编码。前置引擎能力全部就绪（单仓闭环已合 main=4fbb9d2，动态音频双房间
实证，跨环境逐字节一致）。下一步：Session 1 后端骨架，在
`feature/studio-v1` 分支开工。

### Checkpoint 20260822-S1：Session 1 后端骨架完成（验收通过）

代码（分支 feature/studio-v1）：`src/avengine/studio/`（config / catalog /
templates / tasks / server 五模块，纯标准库，无新增依赖）+
`tools/studio/run_studio_server.py` + `tools/studio/studio_config_48g.json`
（48g 部署配置，模板默认输入逐字取自已过审渲染的 receipt）。

能力与验收证据：
- 只读 API：/api/health、/api/rooms（注册表 + studio_status 标注：
  blender_custom=banned、skokloster=excluded）、/api/registries[/name]
  （8 端点 / 3 声音 / 4 实体）、/api/captures、/api/templates、/api/tasks。
- /api/captures 扫 review 根 receipt，目录名 commit token 对
  `git rev-list main` 解析：已过审 skeletal 2786897 与 b9150cb 为
  trusted，1fd3f5d 缺陷系列（3 个）默认被排除（?include=all 可见）。
- 渲染队列：单 worker 串行 subprocess；任务目录含 task.json / task.log /
  output/，fresh/no-clobber；服务重启恢复历史并把遗留 queued/running 标
  interrupted。服务只绑 127.0.0.1（非回环拒绝启动）；预览
  `ssh -L 8765:127.0.0.1:8765 <server>` 后开 http://127.0.0.1:8765/。
- 端到端验收：HTTP POST 默认 mp3d_dynamic_audio 模板 → 队列执行真实
  `m5 render-current-mp3d-dynamic-audio` → pass；输出 mixture 与两条 stem
  的 sha256 与已过审 current_mp3d_dynamic_audio_seed22g_v1 完全一致
  （mixture a0c04fe9…、stem 7fcd7852… / 2fe85646…），任务
  20260821T191530Z-mp3d_dynamic_audio 全程经 Studio 队列产出。

测试：19 个新单测全绿。全量回归 3086 passed / 65 skipped；另有
9 failed + 12 errors 全部集中在三个保留工作区校验模块
（test_mp3d_strict_room_adapter / test_mp3d_f15_launcher /
test_strict_two_human_expansion_acoustic_batch），在无 studio 代码的
5394435 干净 worktree 复现一致——先于本 Session 的机器本地状态问题：
lead-a/tmp 保留 suite plan 引用的 episodes 负载已不在盘上（全盘搜索无
果），SPEAR-lead-b 控制注册表仍是旧多仓路径；模块级 skip 守卫只查目录
存在、不查完整性，所以状态残缺时不再跳过而是失败。与 Studio 无关，待
owner 决定：补齐状态、加固守卫或退役该组校验。

下一步：Session 2 俯视画布 v0（单文件前端 + /validate navmesh 校验）。

S1 附注（同日）：上述三个模块的保留工作区校验，owner 拍板直接移除
（episodes 负载确已不在盘上，旧多仓 external/SPEAR 对比清单与
SPEAR-lead-b 路径同属转轨前遗留）。d537d01（直推 main，规则 bypass）
精确删除 21 个失败测试（RoomAdapterTests 整类 12 个、PreflightTests
4 个、f15 v7 捕获运行时 2 个、strict-two-human 批 2 个），三模块其余
50 个密闭测试保留（50 passed / 3 skipped）。已合回 feature/studio-v1
（7d8e152）。另记录：在无 lead-a 未跟踪 tmp/ 状态的新鲜检出上，全量
单测另有约 28 个先存失败（如 test_m4_cli），属于测试层对机器本地状态
的残余依赖，与本次改动无关，待后续专项清理。

## 变更 20260822-B：owner 指示（S2 起生效）

1. S2–S4 连做，不再逐 Session 停等验收（owner："全部都做了"）。
2. 画布升级为 **3D 场景编辑器**：加载真实房间几何、像 UE/Blender 一样
   实时拖拽；重资产预处理（"预渲染"），最终视频异步 GPU 队列出片。
3. 不只 MP3D——所有可用房间都要进编辑器；视图中必须标出合法可行域。
4. Workflow 只在确能提效时用（ultracode 开着但要省额度）；本轮全部直做。

### Checkpoint 20260822-S234：3D 编辑器 + 端到端 + 防废案双件套落地

**场景包基建**（tools/studio/build_studio_scene_bundle.py + avengine.studio.scenes/validation）：
- M3 声学包网格 → 原始小端 buffers（positions/indices/triangle_material_ids）
  + 按材质类别上色的黏土视图；bundle.json 自描述（schema
  avengine_studio_scene_bundle_v1）。
- 草稿可走性栅格两种来源：MP3D 用 navmesh（原生 PathFinder topdown，构建
  时用已知可走点自校验栅格朝向，错即构建失败）；公寓用**声学网格占用启发
  式**（面积加权检测最低水平主面为地板 + 精确 XZ 光栅化：小三角形 bbox
  填充、大三角形重心点内测试——bbox 粗画会淹没全房、稀疏采样会漏地板，
  两个坑都踩过并修正）。四个已验证角色点入栅格自检。
- 已建包：habitat_mp3d_example_17DRP5sb8fy（301.6 万三角形 + navmesh 栅格
  413×817）、legacy_ue_apartment_0000（78.2 万三角形 + 占用栅格，逐材质
  上色）。Kujiale 与 ReplicaCAD 尚无 M3 声学包，各跑一次 M3 编译即可入列
  （架构房间无关，待办）。
- three.js r0.147（two vendored MIT build files, THIRD_PARTY_NOTICES 已登
  记）；剖切屋顶开关（地板上方 2m 裁剪 = dollhouse 视图）；**合法可行域
  以半透明绿色覆盖层直接画在视图里**（X-ray 式，验证与家具足迹精确互补）。

**S2 验收（3D 拖拽 + 即时校验）**：公寓五标记（相机锁定 + 人/犬起止点）
浏览器拖拽；拖进沙发即时标红"不可放置 outside the walkable navmesh"，
其余保持绿色；/api/validate 服务端草稿校验（栅格 + 引擎自有
point_to_world_obb_clearance OBB 净空）；权威判定仍在渲染链原生闸门。

**S3 验收（网页到成片）**：
- 公寓：显式坐标 → apartment_author 任务（引擎权威校验，秒级）**pass**；
  apartment_end_to_end 驱动器（author→SpearSim UE 捕获→m7 动态音频→
  bgr 出片）。相机 v1 锁定于 M1 权威位姿（音频 1e-6 互证所限），放开
  相机需要配套生成 M1 请求，记为后续。
- MP3D：seed 路线授权任务（~1 分钟）→ 轨迹 JSON 镜像回任务目录 → 3D
  画布叠加预览 → mp3d_end_to_end 驱动器（author→capture→dynamic audio→
  clip）。工程要点：路线作者 verb 要求输出为 review 根直接子目录（驱动
  器写 review/studio_<task>_route 再镜像），RLR SDK 经 AVENGINE_RLR_SDK_ROOT
  环境变量传递。seed 20260822 被引擎正确拒绝（无全程双犬可见相机——
  fail-closed 即防废案），seed 22 通过。
- 成片经 /api/tasks/<id>/artifact 安全端点在页面 <video> 播放（路径逃逸
  已测 400）。

**S4 验收（防废案双件套，草稿启发式实现）**：
- 逐帧方位角色带（蓝左/红右，每源一条；遮挡=对草稿栅格 2D DDA 视线，
  黑色刻线）+ 警示（方位变化 <10°、两源夹角 <20° 帧占比过半）。
- 题型预演清单：左右判断（中位方位 ≥15° 且稳定）、趋近/远离（径向位移
  ≥0.5m）、谁更近（距离比 ≥1.3）、方位交叉、先后发声；拖动端点实时刷新，
  不可产出项标注拒答落点。
- 注意：此为客户端几何启发式（页面上明确标注"引擎 QuestionSpec 为准"）；
  接 avengine.qa.question_spec 离线子集 + 事实表合成留给 v1.5 的
  QuestionSpec 预演器精化。

**质量闸门证据（两条端到端链，均经 Studio HTTP 队列产出）**：
- MP3D（task 20260822T005914Z）：pass；且 seed 22 全链重推的 mixture 与
  已过审 current_mp3d_dynamic_audio_seed22g_v1 **逐字节一致**
  （a0c04fe9…）——同种子确定性把整条 author→capture→audio 链再证明了
  一遍。seed 20260822 被授权层正确拒绝（无全程双犬可见相机候选），
  fail-closed 即防废案。
- 公寓（task 20260822T010802Z，新端点 human_end→(-350,-60)、
  beagle_end→(-200,-104)）：pass；窗口 ILD 交替
  +2.63/−4.09/+2.34/−0.24/−2.73 dB（轮流发声左右化清晰）；逐帧亮度
  86.5→89.3 全程平坦（无曝光爬坡）；动画回放探针 **pass**（source2 中位
  残差 14.0、source1 25.2，均 >9.5 animated 阈值——滑行缺陷未复发）。
- 工程要点（公寓链）：author verb 的 --output 是 timeline JSON 文件本身；
  UE 捕获需 PYTHONPATH 注入外置 avengine_spear_ext 构建
  （spear-host-sdk/avengine-spear-ext-cp312-8a36d4d）；出片 --channel-order
  bgr。相机 v1 锁定 M1 权威位姿。
- 全量单测回归 **3092 passed / 0 failed / 65 skipped**（新增 7 个场景/校验
  测试）。

**待办（记录，不阻塞）**：Kujiale 与 ReplicaCAD 各需一次 M3 声学编译才能
入编辑器；公寓自由相机需配套生成 M1 请求；QuestionSpec 预演器接引擎
离线子集（v1.5）；MP3D 纹理 glb 视图可选增强。
