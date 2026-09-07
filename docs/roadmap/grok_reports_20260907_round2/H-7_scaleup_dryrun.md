# H-7 放量干跑请求与结果侧统计

任务书 `docs/roadmap/GROK_FIX_TASKS_ROUND2_20260907.md` 第 2 节 H-7；依据 Claude 审核 R8、R9、R10、K7、K8。本报告按第一轮任务书第 5 节六段格式。

## 1. 改了哪些文件（路径），提交号

提交号即本文件所在提交（分支 `grok/pilot46-fixes-round2-20260907-h7`）。本项改动：

- `src/avengine/qa/batch_manifest.py`
  - `prepare_batch_manifest` 再打散时从根字段或 `scaleup.distance_range_m` 透传距离；槽位已有 `profile.distance_range_m` 时也不丢。
  - `collect_batch_outcomes` 行补 `requested_source_classes`；旧清单缺 `competitor_visibility` / `distance_range_m` / `separation_target_policy` 时按 `COMMON_PROFILE` 默认值比较，不再把 19 段重跑记成 `profile_matches_request=False`。
  - 池里 0 条兼容声音记 `no_compatible_sounds`（`compatible_sound_count=0`），与身份用尽 `no_distinct_compatible_sound_identity` 分开。未改 5 s 上限、未删空调资产、未改候选范围。
- `tools/dataset/build_qa_batch_manifest.py`：`scaleup-dry-run --sounds` 把声音池绝对路径写进 `base_request.sound_pool` 和每个 request；去掉会盖住它的 `prepared_set`。顺手把本工作树绝对 `room_catalog` / `source_registry` 和 catalog `path_bindings` 写进 request，让 `--plan-only` 可执行。H-3 仍拥有 `prepare` 路径上的 catalog 绝对化；本项只动 `scaleup-dry-run` 分支。
- `tests/unit/test_qa_batch_manifest.py`：距离透传、旧画像默认值、`requested_source_classes`、无声音缺口、CLI `--sounds` 写 `sound_pool`。
- `docs/roadmap/grok_reports_20260907_round2/H-7_scaleup_dryrun.md`：本报告。

未改：Claude 审计器、Studio、第一轮产物、`batch_sound_pool.py`、`room_package.py`、声学渲染、合并脚本。未覆盖 `tmp/gc_scaleup_dryrun_7x50_20260907`。未改 `docs/TOOL_INDEX.md`（CLI 模块 docstring 未变，现有一行描述仍成立）。

## 2. 跑了哪些测试，各自的通过/失败/跳过计数

工作树 `/data/jzy/tmp/wt-grok-pilot46-H7`，`PYTHONPATH=src:tmp/native_python_addons_v1`，解释器 `/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`：

```
python -m pytest tests/unit/test_qa_batch_manifest.py -q
```

**21 passed / 0 failed / 0 skipped**（0.69 s）。本项触碰的测试文件即此一份。原 16 条仍过；新增 5 条：再打散保留 `[1.5, 6.0]`、槽位已有距离不丢、0 兼容声音不是身份用尽、旧画像缺键按默认比较且交叉表不再塌成空键、CLI `--sounds` 每个 request 都有 `sound_pool`。

未跑 GPU 捕获。`import avengine` 解析到本工作树 `/data/jzy/tmp/wt-grok-pilot46-H7/src/avengine/__init__.py`。

## 3. 验收产物的路径，以及核对结果

### 3.1 新放量干跑（350 request）

新目录（未覆盖第一轮干跑）：

- `tmp/h7_scaleup_dryrun_7x50_20260907/`
- 解析绝对路径：`/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/h7_scaleup_dryrun_7x50_20260907/`

命令（本工作树 catalog / registry，复用只读 P10 声池）：

```
python tools/dataset/build_qa_batch_manifest.py scaleup-dry-run \
  --config examples/dataset/qa_pilot_46_20260907.json \
  --catalog /data/jzy/tmp/wt-grok-pilot46-H7/examples/rooms/packages/catalog.json \
  --registry /data/jzy/tmp/wt-grok-pilot46-H7/examples/runtime/source_asset_runtime_profiles.json \
  --sounds /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p10_pilot46_manifest_20260907_v3/batch_sounds.json \
  --output tmp/h7_scaleup_dryrun_7x50_20260907 \
  --seed 20260907 --episodes-per-room 50 --batch-id qa_scaleup_7x50_20260907
```

亲自核对：

| 项 | 结果 |
|---|---|
| 请求数 | 350 |
| `request.sound_pool` | 350/350 均为上述 batch_sounds.json 绝对路径；无 prepared_set |
| `distance_range_m` 直方图 | requested_profile、request.profile、scaleup_config.json 槽位画像 **350 x [1.5, 6.0]**；0 条 [1.5, 4.5] |
| 画像逐行 | 350 行 request.profile 等于对应 scaleup_config.json 槽位 profile |
| `room_catalog` | 本工作树绝对路径；`runtime.path_bindings` 15 项（含 AVENGINE_MULTI_HOME_AUTHORING_ROOT） |
| `repeat_deficit_count` | **0** |
| 预分配缺口 | `no_compatible_sounds`: 2（便携空调，另计）；无身份用尽码；无 repeat 片长缺额码 |
| 交叉表 | meets_acceptance=true，min_distinct_groups_per_class_pair=4 |
| 画外 | 14 段（每房槽 0 锚点画外 + 槽 1 竞争者画外）；device-device 画外 0 |

两段无声音：

- `qa_scaleup_7x50_20260907_029_apartment_device_device` source1 `generated_air_conditioner_portable_floor_white_research_v2`
- `qa_scaleup_7x50_20260907_241_kujiale_device_device` source1 同一资产

声池：air_conditioning 20 条全部 `registered_event_exceeds_explicit_clip_budget`（>5 s）被拒，池内 0 条。登记表三台空调（portable_floor / wall_split / window_unit）兼容声音均为 0。候选范围只有落地便携一台；墙挂/窗机本来就不在地板候选里。

### 3.2 六段 plan-only（每家族 1 段 + 1 段画外；未捕获）

产物：`tmp/h7_plan_only_20260907/`（绝对路径同上 workspace `root/h7_plan_only_20260907/`）。控制器 `tools/studio/run_qa_episode.py --plan-only`。六段 returncode=0，status=research_candidate，native_execution=not_run，均写出 `plan/episode_plan.json` 与 `condition_profile.json`。无 TypeError。condition_profile.distance_range_m 均为 [1.5, 6.0]。

| episode_id | 家族 | 画像 | 墙钟 |
|---|---|---|---|
| qa_scaleup_7x50_20260907_003_apartment_device_human | apartment | in_fov | 3 s |
| qa_scaleup_7x50_20260907_053_authored_device_human | authored | in_fov | 2 s |
| qa_scaleup_7x50_20260907_203_kujiale_device_human | kujiale | in_fov | 15 s |
| qa_scaleup_7x50_20260907_253_mp3d_device_human | mp3d | in_fov | 16 s |
| qa_scaleup_7x50_20260907_303_hm3d_device_human | hm3d | in_fov | 7 s |
| qa_scaleup_7x50_20260907_001_apartment_human_human | apartment | **anchor off_screen** | 7 s |

未渲 GPU、未写 capture。

### 3.3 结果侧统计（只读重算，不写回第一轮目录）

用本工作树 `collect_batch_outcomes` 对 `tmp/qa_pilot46_rerun_20260907_v1/summary/batch_outcomes.json` 的 20 段 outcome 接到第一轮 46 格清单对应行：

- 旧收集：profile_matches_request False 19 / None 1；交叉表 class_pair 只有空字符串；配额 delivered 0。
- 新收集：True 19 / None 1（`authored_b_animal_device` 规划失败，没有 observed profile）；交叉表六种类别对；配额 delivered 19、unmet 1。
- 行上有 `requested_source_classes`。

差集仍只是那三个新画像键，默认值与 observed 一致（competitor_visibility=in_fov，distance_range_m=[1.5, 4.5]，separation_target_policy=any_legal_in_bin）。

### 3.4 46 格清单两条 repeat 缺额（未重生成）

只读 `tmp/p10_pilot46_manifest_20260907_v3/batch_manifest.json`（producer 24ea774）：

- `qa_pilot46_20260907_authored_a_device_device`、`qa_pilot46_20260907_kujiale_device_device` 仍是 `fixed_sound_identities_exceed_profile_clip_budget`。
- 同一 config 在本工作树内存重算：preallocation_gap_counts 为空，repeat 缺额 0（K8：HEAD 代码会换身份，清单文件没重出）。本项不重生成 46 格。

放量清单已重出：repeat 缺额 0；便携空调两条按 `no_compatible_sounds` 另计。

## 4. 没做完的部分和原因

| 项 | 三分法 | 说明 |
|---|---|---|
| 46 格清单两条 repeat 缺额 | 题义不适用 | 任务要求如实写、不重生成该清单。放量清单已是 0。 |
| 便携空调 0 声音的处置 | 题义不适用 | 等 owner：从候选拿掉 vs 允许裁剪超长设备声。未改阈值、未删资产。 |
| 画外两条边界 | 题义不适用 | 写清并等 owner，见第 6 节。未改槽位分配、未改可听窗外约束。 |
| H-3 的 prepare catalog 绝对化 | 题义不适用 | 本项只保证 scaleup-dry-run 请求可执行。 |
| 正式 350 段 GPU 捕获 | 题义不适用 | 任务禁止全量捕获。 |
| 第一轮干跑目录 | 题义不适用 | 只读，未覆盖。 |

## 5. 对后续接口的要求

- `scaleup-dry-run --sounds PATH`：每个 request 必须有 sound_pool=PATH 绝对路径，且不要再留 sound_selection.prepared_set。`tools/studio/run_qa_episode.py` 第 65-66 行读这个字段。
- `prepare_batch_manifest` 在 scatter_condition_groups=True 时读根字段 distance_range_m 或 scaleup.distance_range_m；已散过的槽位 profile.distance_range_m 不得被默认 [1.5, 4.5] 盖掉。
- `collect_batch_outcomes` 行含 requested_source_classes，供 class_pair_condition_group_crosstab。画像比较对缺省键填 competitor_visibility=in_fov、distance_range_m=[1.5, 4.5]、separation_target_policy=any_legal_in_bin。H-2 若加 failure_stage，不要删这两处。
- 预分配缺口：no_compatible_sounds = 池内 0 条匹配；no_distinct_compatible_sound_identity = 有声音但身份不够。
- 新干跑：tmp/h7_scaleup_dryrun_7x50_20260907/{batch_manifest.json,scaleup_config.json,requests/,scaleup_dry_run_summary.json}。

## 6. 需要 owner 拍板的事项

1. **便携空调没有声音，不是身份用完。** air_conditioning 20 条在建池时全部因超过 5 s 被拒；三台空调兼容声音均为 0；落地便携被分到放量 2 槽，现记 no_compatible_sounds。请在「从候选范围拿掉」与「允许裁剪超长设备声」之间选。本项两手都没做。
2. **画外边界 A：可听窗外，speaker_moving 的画外锚点不说话时会走进画面。** 约束只作用于锚点可听窗。Claude X3 复核：mp3d 段可听窗外最小 |方位| 8.7°，hm3d 段 1.1°（98/89 帧在 85° 锥内）。本干跑 14 段画外里 5 段 speech_motion=speaker_moving（kujiale / mp3d / hm3d 的房首槽）。请定：不说话时入画是否可接受；若否，约束是否要扩到整段时钟。
3. **画外边界 B：干跑画外槽写死每房第 1、2 槽，设备画外零段。** build_scaleup_slots：每房 slots[0] 锚点画外（人人），下一非 device-device 槽竞争者画外（人-动物）。本干跑 14 段画外 = 7 锚点 + 7 竞争者；类别对只有 human-human 与 animal-human；device-device 画外 0。请定：设备要不要有画外槽、要不要打散到各条件组而不是每房固定前两槽。

无其它自行决定。
