# G-B UE 视觉侧：曝光根因、闸门、地板标签、包路径

按 `docs/roadmap/GROK_FIX_TASKS_20260907.md` 第 5 节。工作树
`/data/jzy/tmp/wt-grok-pilot46-GB`，分支 `grok/pilot46-fixes-20260907-gb`。
未 push。未改 `batch_delivery.py`（G-A 所有）。未动 Studio `:8765`。
未写 `attempt_01`。未改 Claude 审计器、Codex 报告原文、阈值。

## 1. 改了哪些文件（路径），提交号

实现提交 `261f1883b1615e7c6e75a2eed1f4458e4a72baba`（短号 `261f188`）。本句中的哈希由后续 docs 提交写入，避免 amend 后自指失效。

代码与包：

- `src/avengine/qa/exposure_gate.py`（新）导出 `apply_exposure_gate`、
  `compute_exposure_stats`、`PLACEHOLDER_EXPOSURE_GATE_CONFIG`。
  placeholder 阈值：`mean_gray > 235` 或 `sat_share > 0.20`（灰阶 ≥250
  像素占比）→ `review_failed`，reason 带数值；`threshold_kind=placeholder`。
- `src/avengine/rooms/room_package.py`：对非 `/Game`、非 `/Root` 的绝对
  文件系统路径做存在性检查（`missing_filesystem_paths`）。
- `tools/rooms/run_spear_residential_episode.py`：`--exposure-bias-ev`；
  从 CLI / `visual_plan.camera` / `resources` / `room_package` /
  `planning_inputs` 解析补偿；A/C/B 地图缺字段时回落到
  `AUTHORED_USD_EXPOSURE_BIAS_EV`（A/C −3、B −4）。
- `tools/rooms/measure_ue_room_floor_reference.py`：
  `classify_ue_floor_measurement`；line-trace 0 hit 或走 depth 回退时写
  `measurement_kind=depth_readback_fallback`、`precision_m=0.00025`；
  `|floor_height_m|>10` 标 `invalid`。
- `examples/rooms/packages/catalog.json`：新增
  `AVENGINE_MULTI_HOME_AUTHORING_ROOT`（authoring 树，真实 props/lighting）。
- `examples/rooms/packages/room_{a,b,c}.json`：恢复 `exposure_bias_ev`
  （A/C −3.0，B −4.0）；地板改指向
  `tmp/gb_floor_reference_20260907/...` 并声明 depth fallback；
  `living_props_v7_candidate` / `detailed_v8_linear_materialfix` /
  `polished_v3_final` 改到 authoring root。
- `examples/rooms/packages/kujiale_0020_full_home_v1.json`：
  `map_asset` 补 `SpearSim/`；地板同样改标。
- `examples/rooms/packages/hm3d_00800_TEEsavR23oF.json`：`route_bank`
  去掉多余的 `routes_R3` 一层。
- 单测：`tests/unit/test_exposure_gate.py`、
  `tests/unit/test_floor_reference_measurement_kind.py`、
  `tests/unit/test_room_package_path_existence.py`；
  `tests/unit/test_room_packages_p3.py` 地板路径前缀更新。

未改：`src/avengine/qa/batch_delivery.py`、`docs/TOOL_INDEX.md`
（`tools/build_tool_index.py --check` 已通过，docstring 未变）、
`examples/rooms/packages/native_apartment.json`。

## 2. 测试

项目 Python
`/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`，
`PYTHONPATH=src:tmp/native_python_addons_v1`。import avengine →
`/data/jzy/tmp/wt-grok-pilot46-GB/src/avengine/__init__.py`。

```
pytest -q tests/unit/test_exposure_gate.py \
          tests/unit/test_floor_reference_measurement_kind.py \
          tests/unit/test_room_package_path_existence.py \
          tests/unit/test_room_packages_p3.py
```

| 文件 | passed | failed | skipped |
|---|---:|---:|---:|
| `tests/unit/test_exposure_gate.py` | 2 | 0 | 0 |
| `tests/unit/test_floor_reference_measurement_kind.py` | 4 | 0 | 0 |
| `tests/unit/test_room_package_path_existence.py` | 2 | 0 | 0 |
| `tests/unit/test_room_packages_p3.py` | 10 | 0 | 0 |
| **合计** | **18** | **0** | **0** |

`tools/build_tool_index.py --check`：exit 0，未重写 `docs/TOOL_INDEX.md`。

## 3. 验收产物与亲自核对

### 3.1 根因（10 帧对照 + 旧交付）

新共同舞台
`qa_full_asset_ue_stage_20260907_v1` 上同一 `authored_a_human_human`
请求截成 9 帧（`clock.frame_count=9`，计划
`tmp/gb_exposure_fix_20260907/ten_frame/plan_10`）：

| 集合 | n | 灰度均值 | sat_share（灰≥250） | 路径 |
|---|---:|---:|---:|---|
| 新舞台 bias=0 | 9 | 229.61 | 0.259 | `tmp/gb_exposure_fix_20260907/ten_frame/new_bias0/` |
| 新舞台 bias=−3 | 9 | 151.37 | 0.000 | `tmp/gb_exposure_fix_20260907/ten_frame/new_bias-3/` |
| 旧交付 A `walk_pair_a_v6`（只读） | 240 | 125.42 | 0.000 | `tmp/qa_real_rooms_20260906/walk_pair_a_v6/capture/frames` |
| 旧交付 B `walk_four_b_v2`（只读） | 300 | 197.27 | 0.000 | `tmp/qa_real_rooms_20260906/walk_four_b_v2/capture/frames` |
| 旧交付 C `vocal_classes_c_v2`（只读） | 240 | 173.28 | 5.3e-5 | `tmp/qa_real_rooms_20260906/vocal_classes_c_v2/capture/frames` |
| attempt_01 A（只读，未写） | 240 | 241.49 | 0.615 | `tmp/qa_pilot46_background_20260907_v1/episodes/...authored_a_human_human/attempt_01/...` |
| attempt_01 B（只读） | 240 | 253.35 | 0.920 | 同上 authored_b |
| attempt_01 C（只读） | 240 | 247.40 | 0.632 | 同上 authored_c |

直方图 JSON/PNG：`tmp/gb_exposure_fix_20260907/histograms/`
（`index.json`、`new_bias0.json`、`new_bias-3.json`、
`old_walk_pair_a_v6.json` 等）。

亲眼看过 `new_bias0/frames/frame_0000.png`：墙面吹白、石膏纹理丢失。
`new_bias-3` 同机位：墙面有纹理、对比正常，与下面 240 帧 A 接触印一致。

旧舞台 `/data/avengine_external/workspaces/multi_home_activity_20260905/ue_stage/SpearSim`
只读重试两次（RPC 39852 与 39911）均在 `spawn_actor` 时
`UnrealObject.__init__` `assert self.uobject != 0` 失败，0 帧。
旧舞台 10 帧直方图为**证据缺失**；对照改用上述旧交付帧。

**根因一段：** 本批 16 段 A/B/C 在新舞台过曝，不是 uproject / Config.ini /
地图字节 / DDC / 分辨率 / 预热帧的差异。新舞台 9 帧在 `bias=0` 上复现
均值 ~230、sat ~26%；同一请求 `bias=−3` 落到均值 ~151、sat 0，与旧交付
A 房 `walk_pair_a_v6`（均值 ~125、sat 0）同一量级。attempt_01 只读统计
A/B/C 均值 241–253、sat 61–92%，正是 0 EV + 强制关自动曝光后
Blender DiskLight 把 8-bit RGB 打满。旧生产校准是 A/C −3 EV、B −4 EV；
新包丢掉了 `exposure_bias_ev`，SceneCapture 按 0 EV 跑。耐久修复是把
包字段写回、地图缺字段时用 `AUTHORED_USD_EXPOSURE_BIAS_EV` 回落，并给
batch_review 预留 placeholder 闸门（本 worktree 不改 `batch_delivery.py`）。

### 3.2 修好后 A/B/C 各 240 帧（新舞台，新输出目录）

visual-only，`--width 1280 --height 720`，`--graphics-adapter 0`，
RPC 39901/39902/39903，包声明的 EV（A/C −3，B −4）。输出不进
`attempt_01`：

`tmp/gb_exposure_fix_20260907/capture_240/room_{a,b,c}_authored_human_human/`

验收（帧 0/120/239：灰度均值 90–200，sat_share < 5%）：

| 房间 | bias_ev | frame 0 mean / sat | frame 120 | frame 239 | 闸门 |
|---|---:|---|---|---|---|
| A authored_human_human | −3 | 151.38 / 0 | 151.32 / 0 | 151.39 / 0 | pass |
| B authored_human_human | −4 | 178.32 / 0 | 178.35 / 0 | 178.34 / 0 | pass |
| C authored_human_human | −3 | 170.27 / 9.0e-5 | 170.28 / 8.8e-5 | 170.34 / 9.0e-5 | pass |

接触印（亲眼看过：墙面有纹理、人物可辨、未吹白）：

- `tmp/gb_exposure_fix_20260907/capture_240/contact_sheets/room_a_frames_0_120_239.png`
- `tmp/gb_exposure_fix_20260907/capture_240/contact_sheets/room_b_frames_0_120_239.png`
- `tmp/gb_exposure_fix_20260907/capture_240/contact_sheets/room_c_frames_0_120_239.png`

`capture_exposure_readback` 均为 `{status: pass, override: true, source: cli}`。
JSON：`tmp/gb_exposure_fix_20260907/histograms/capture_240_acceptance.json`。

### 3.3 七包路径存在性（展开 catalog.path_bindings 之后）

表：`tmp/gb_exposure_fix_20260907/package_missing_paths_{before,after,rerun}.json`。
清理前缺 `living_props_v7_candidate` / `detailed_v8_linear_materialfix` /
`polished_v3_final`（在 datasets 树不存在，真实文件在
`AVENGINE_MULTI_HOME_AUTHORING_ROOT`）、Kujiale `map_asset` 少 `SpearSim/`、
HM3D `route_bank` 多一层 `routes_R3`。清理后再跑：

| 包 | missing_count | 诚实残留 |
|---|---:|---|
| native_apartment.json | 0 | 无 |
| room_a.json | 0 | 无 |
| room_b.json | 0 | 无 |
| room_c.json | 0 | 无 |
| kujiale_0020_full_home_v1.json | 0 | 无 |
| mp3d_17DRP5sb8fy.json | 0 | 无 |
| hm3d_00800_TEEsavR23oF.json | 0 | 无 |

`status=all_seven_packages_exist`。相对路径地板文件在仓库 `tmp/` 符号链接
下存在。

### 3.4 地板标签

| 房间 | 包字段 | 文件 | measurement_kind | status | floor_height_m |
|---|---|---|---|---|---:|
| A/B/C/Kujiale | `depth_readback_fallback`，`precision_m=0.00025` | `tmp/gb_floor_reference_20260907/{room_a,room_b,room_c,kujiale}/floor_reference.json` | `depth_readback_fallback` | measured | 0.0001953125 |
| Apartment 0.2711 | **未改** `measured_room_floor` | `tmp/p3_room_packages_20260907_v3/floor_reference/legacy_ue_apartment_0000_v1/floor_reference.json` | （原文件未加 kind） | measured | 0.2711074501 |
| Apartment −74.76 | 包不指向此文件 | copy-on-write `tmp/gb_floor_reference_20260907/native_apartment_invalid/floor_reference.json` | `depth_readback_fallback` | **invalid** | −74.7625 |
| v6 原件 −74.76 | 未改 | `tmp/p3_room_packages_20260907_v6/ue_floor_measurements/native_apartment/floor_reference.json` | 仍标 measured | measured | −74.7625 |

A/B/C/Kujiale 的 `method.line_trace.hit_count` 仍为 0，按任务书改标，
没有假装成 line trace。

## 4. 没做完的部分

- **证据缺失：** 旧舞台 10 帧对照。只读重试 RPC 39852 与 39911 均
  `spawn_actor` → `assert self.uobject != 0`。直方图对照改用旧交付
  `walk_pair_a_v6`（及 B/C 旧交付）。未改旧舞台。
- **题义选择（已按“如实写”做，未修碰撞）：** 四房地板仍是 depth
  量化残差 0.0001953125 m，没有给地板网格加碰撞让 line trace 真正命中。
- **接口未实现（所有权在 G-A）：** `apply_exposure_gate` 未接入
  `batch_delivery.py` / `batch_review`。闸门模块与单测已在本分支，
  重跑前需 G-A 调用。
- **题义不适用：** 未改阈值、未删资产、未动 Studio、未覆盖 attempt_01。

## 5. 对后续接口的要求

- `from avengine.qa.exposure_gate import apply_exposure_gate`
- 签名：
  `apply_exposure_gate(review: Mapping, episode_root: Path, *, config: Mapping | None = None) -> dict`
- 失败：拷贝 review，`status="review_failed"`，`reason` 含
  `mean_gray=` / `sat_share=` 数值；`exposure_gate.threshold_kind="placeholder"`。
- 通过：保留原 `status`，附加 `exposure_gate.status="pass"`。
- 帧目录搜索顺序：`capture/frames`、`frames`、`batch_review/frames`、
  `episode/capture/frames`；取首/中/末。
- 配置键（勿放松）：`mean_gray_fail_above=235.0`、
  `sat_share_fail_above=0.20`、`sat_value=250`。
- 捕获侧：`run_spear_residential_episode.py --exposure-bias-ev`；
  缺 CLI 时读包 `exposure_bias_ev`（A/C −3，B −4）。
- 地板：`measurement_kind`、`precision_m`、`status=invalid`。
- 路径：先 `resolve_room_package_paths(..., runtime={"path_bindings": catalog["path_bindings"]})`，
  再 `missing_filesystem_paths`。`${...}` 未展开前不会当绝对路径检查。

## 6. 需要 owner 拍板

- 任务书写“耐久 −3 EV”，B 房生产校准与包字段是 **−4 EV**。本项按包声明
  渲 B，未自行改成 −3。
- 地板选择了“如实标签”而不是修碰撞。若要真正的 line-trace 地板，需要另开
  测量任务（临时给地板网格加碰撞或对已加载几何射线求交）。
- `native_apartment` 包继续指向 0.2711 的 v3 实测；−74.76 只在 copy 上标
  invalid，v6 原件仍为 measured。若要改原件需 owner 明确允许。
- 曝光闸门阈值保持 placeholder，未接入 batch_review；G-E 重跑前必须先有
  G-A 接线。
