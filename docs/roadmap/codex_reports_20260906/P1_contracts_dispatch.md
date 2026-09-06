# P1 契约、分派与中立读回

## 1. 文件与提交

实现基线 `ca70486e7d2e30637f39d2c800a4e3fb4b0f131b`。本报告与本任务实现同一提交；提交号从 `git log -1 --format=%H -- docs/roadmap/codex_reports_20260906/P1_contracts_dispatch.md` 读取，避免把提交自身哈希递归写进提交内容。

新增：`src/avengine/rooms/room_package.py`、`src/avengine/rooms/evidence_contract.py`、`src/avengine/capture/{neutral_readback,ue_neutral_readback,habitat_neutral_readback}.py`、`tools/capture/write_neutral_readback.py`、`tests/unit/test_qa_production_contracts.py`。
修改：`tools/studio/run_qa_episode.py`、`tools/rooms/run_spear_residential_episode.py`、`docs/roadmap/QA_PRODUCTION_ARCHITECTURE_20260906.md`、`docs/TOOL_INDEX.md`。

RoomPackage 校验 11 个必需字段：room_id、family、renderer、visual_scene、acoustic_package、walkable_space、floor_reference、static_geometry、semantics、coordinate_frame、subrooms。新式包缺 floor_reference 拒绝；旧目录条目包装为有显式缺口的草稿，旧请求沿用原规划行为，不补假地板。渲染器按 family/renderer 选择；MP3D/HM3D 不接受 UE 生产路由。已接控制器到两个现有捕获入口，Habitat 要求真实物化 case/m1/room 输入与计划时钟一致。

## 2. 测试

在本工作树、指定 runtime Python 运行。第一轮 P1 单测 17 passed / 0 failed / 0 skipped。补分派测试后与旧规划回归合跑 29 passed / 0 failed / 2 skipped；两个跳过是默认 Python 路径没有 soundfile。复用已有 `tmp/native_python_addons_v1` 后补跑（不安装或修改环境），最终 **32 passed / 0 failed / 0 skipped**（3.23 秒），日志 `tmp/p1_contracts_20260906_v1/tests_final.log`。

命令：`PYTHONPATH=src:tmp/native_python_addons_v1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python -m pytest tests/unit/test_qa_production_contracts.py tests/unit/test_native_qa_room.py tests/test_question_driven_rooms.py tests/unit/test_tool_index_current.py -q`。

已运行工具索引生成，独立索引检查 1 passed / 0 failed / 0 skipped。没有新增渲染作业，没有把单测当原生捕获验收。

## 3. 验收产物与核对

产物根：`tmp/p1_contracts_20260906_v1/`，实际存储为 `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p1_contracts_20260906_v1/`。

- `a_neutral_readback.json`：保留的 A 房 240 帧、2 实体，逐帧相机基与 root/emitter 转换通过；所有 root 往返最大误差 **0.0 cm**，低于 1 mm。单测另对 50 组位置/旋转验证往返以及与 unified_catalog 方位基一致。
- `mp3d_neutral_readback.json`：保留的 MP3D 150 帧、2 实体；相机使用实际 RGB sensor，检查各模态共位与朝向；JSON root/emitter 与 NPY 逐项一致。以已存在 case_manifest.clock 为计划权威，没有自行重算视频时长。
- `validation.json`：两份中立读回与 A 房完整旧证据的格式/一致性校验结果、真实源路径、Python 和基线 Git 身份。此处是保留原生读回的转换验证，不声称本次新捕获或人工重新审核。
- `room_drafts/`、`room_drafts_and_dispatch.json`：四家族加 A/B/C 共 7 份草稿、缺字段清单、两个真实捕获入口路径。没有实际运行记录的条目均为 native_execution=not_run。

亲自核对内容：实际读回时钟、每帧实体身份、root/emitter JSON/数组一致、相机基、0.0 cm 往返值，以及 A 房证据文件之间的帧/实例/音频时钟和路径一致。没有新做人工试听或外观签字。

## 4. 未完成范围与缺口分类

| 房间 | 草稿缺口，交 P3 实测/接线 |
|---|---|
| 原生 Apartment 与 A/B/C | floor_reference、static_geometry、semantics 尚未由旧目录直接声明；不表示资源必然不存在 |
| 酷家乐 | 同上，另需 walkable_grid 路径与 pose_bindings；本任务没有把未运行地图标可用 |
| MP3D/HM3D | floor_reference、static_geometry 需补齐；声学包 schema 和地板实测属于 P3 |

P1 的契约、执行器选择和保留读回转换已交付；当前 Habitat 的共同计划生成/物化仍由 P5 实现，像素产出由 P4、音频由 P6、完整同一收口由 P9 接齐。此时不能宣称整个控制器到 questions 的 Habitat 原生链已通过。上述均为接口或证据缺口，未记为题义不适用。

## 5. 给 Claude 的接口

- `avengine.capture.neutral_readback.validate_neutral_readback(data, plan=plan)`；`validate_clock(clock)`。
- 后端写出器：`avengine.capture.ue_neutral_readback.write_ue_neutral_readback(capture_dir, plan, output_path=None)` 与 `avengine.capture.habitat_neutral_readback.write_habitat_neutral_readback(...)`。
- `camera[frame]` 使用 position_m、basis.forward/right/up、frame_index、pts_ticks；`entities[source_slot][frame]` 使用 root、emitter、moving。moving 沿用 unified_catalog 对实际 root 的前向差分 > 0.05 m/s，最后一帧复用上一差分，不抄计划 moving。
- `avengine.rooms.room_package.validate_room_package(package)`、`room_package_errors(package)`；`avengine.rooms.evidence_contract.validate_evidence_contract(files, clock=clock, require_complete=True)`，files 的键为五个现有文件名。
- C2 可读取上述 MP3D 中立读回做逐帧方位对账。审计器及其测试没有修改。共用 answerability 纯函数详约实际位于 sampler v2 第 2.7 节（总前提把章节错写成架构第 2.7 节）。

## 6. 总前提冲突与 owner 决策

实际源码与任务书有一处命名差异：现有消费者读 NPZ `depth_derived_modal_semantic`，文档写 `modal`。为保留现有证据安全语义，旧键继续兼容，若同时出现 `modal` 别名则必须完全一致；没有更换掩膜格式或放宽原生证据。此差异已同步架构文档并通知 P4。

当前没有需要 owner 越权批准的动作。新房间未测地板、Habitat 后续接口未完成、人工未重审，都保持明确未完成状态；不 push、不合并、不切 Studio、不启动46段批次。
