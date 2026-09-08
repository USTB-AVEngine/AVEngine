# G-C 采样器与批清单

任务书 `docs/roadmap/GROK_FIX_TASKS_20260907.md` 第 2 节 G-C。本报告按第 5 节格式。

## 1. 改了哪些文件（路径），提交号

提交号即本文件所在提交（分支 `grok/pilot46-fixes-20260907-gc`）。本项改动：

- `src/avengine/rooms/conditioned_sampler.py`：同层锁定（`SAME_FLOOR_Y_TOLERANCE_M=0.3`、`lock_same_floor_region`、多层先抽层）、`anchor_visibility=off_screen` 与 `competitor_visibility=off_screen`、距离上限配置、`clip_span_fit_policy` 接上、实测分离角 5° 分档、`separation_target_policy=uniform_in_bin` 可选项。
- `src/avengine/qa/batch_manifest.py`：repeat 预分配可行性（`2×重复 + 另一源 + 2×0.5s ≤ 可用秒`，超预算换合法身份，没有合法身份才记缺额）、`scatter_condition_groups`、类别对×条件组交叉表、`build_scaleup_slots` / `prepare_scaleup_dry_run`、outcome 里按实测角做 5° 直方图。
- `tools/dataset/build_qa_batch_manifest.py`：新增 `scaleup-dry-run` 子命令（不跑 GPU）。
- `tests/unit/test_conditioned_sampler.py`、`tests/unit/test_qa_batch_manifest.py`：G-C 验收单测。
- `docs/roadmap/QA_GENERALIZED_SAMPLER_PLAN_V2_SPEECH_MOTION_NOTE_20260907.md`：v2 旁注，`speaker_moving` 不要求竞争者静止。
- `docs/TOOL_INDEX.md`：随 CLI docstring 重生成。

未改：`src/avengine/qa/batch_sound_pool.py`（声音身份分配已在 `batch_manifest.prepare_batch_manifest`，池构建仍只负责 PCM 接合）。未改 G-A/B/D 占用文件、Claude 审计器、Codex 报告、Studio、`attempt_01`。

## 2. 跑了哪些测试，各自的通过/失败/跳过计数

命令（worktree `/data/jzy/tmp/wt-grok-pilot46-GC`，`PYTHONPATH=src:tmp/native_python_addons_v1`）：

```
/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python -m pytest -q \
  tests/unit/test_conditioned_sampler.py \
  tests/unit/test_qa_batch_manifest.py \
  tests/unit/test_tool_index_current.py
```

| 文件 | collected | passed | failed | skipped |
|---|---:|---:|---:|---:|
| `tests/unit/test_conditioned_sampler.py` | 25 | 25 | 0 | 0 |
| `tests/unit/test_qa_batch_manifest.py` | 16 | 16 | 0 | 0 |
| `tests/unit/test_tool_index_current.py` | 1 | 1 | 0 | 0 |
| **合计** | **42** | **42** | **0** | **0** |

新增覆盖：两层导航网同层（`|Δy|≤0.3`）、repeat 超预算替换 / 合法对接受 / 无合法身份才缺额、交叉表每种类别对 ≥3 个条件组、画外锚点与竞争者合法性、实测 5° 直方图（60.5/61.9 进 60–65，不把请求箱当覆盖）。

无失败，无跳过。

## 3. 验收产物的路径，以及核对结果

### 3.1 HM3D 连续 20 次规划两源同层（plan-only）

产物：`/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/gc_plan_only_20260907/hm3d_same_floor_20.json`（worktree 内 `tmp/gc_plan_only_20260907/`，`tmp` 为该数据盘符号链接）。

房间 `hm3d_val_00800_TEEsavR23oF`，包内子房间 `R3_floor_0.1634` / `R3_floor_3.1634`。连续 20/20 成功，最大两源高差 0.136 m（≤0.3 m）。两层都被抽到（约 0.1634 m 与 3.16 m）。相机相对所选楼层 +1.55 m。plan-only 时钟 120 帧 / 15 Hz（几何约束与生产相同；未渲 GPU）。

### 3.2 放量干跑 7 房间 × 50 段

产物：

- `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/gc_scaleup_dryrun_7x50_20260907/batch_manifest.json`
- `.../scaleup_dry_run_summary.json`
- `.../scaleup_config.json`
- `.../requests/`（350 个独立 request）

核对：`repeat_deficit_count=0`；交叉表 `meets_acceptance=true`，`min_distinct_groups_per_class_pair=4`（要求 ≥3）。画外画像 14 段（每房 1 锚点画外 + 1 竞争者画外）。`gpu_execution=false`。

交叉表计数（类别对 × 条件组）：

| class_pair | identity_binding | audio_event_relations | visibility_occlusion | motion_distance | post_sound_state | n_groups |
|---|---:|---:|---:|---:|---:|---:|
| animal-animal | 11 | 12 | 12 | 10 | 7 | 5 |
| animal-device | 14 | 9 | 13 | 6 | 14 | 5 |
| animal-human | 16 | 10 | 11 | 7 | 6 | 5 |
| device-device | 12 | 22 | 16 | 0 | 11 | 4 |
| device-human | 9 | 15 | 19 | 12 | 10 | 5 |
| human-human | 17 | 10 | 12 | 19 | 8 | 5 |

device-device 的 `motion_distance=0` 是合法的：全刚体不能走，`legal_condition_groups` 去掉该组。

另有 2 段 `no_distinct_compatible_sound_identity`（便携空调 `generated_air_conditioner_portable_floor_white_research_v2` 在当前接合声池里没有剩余可区分身份）。**不是** repeat 片长缺额，未改阈值去抹掉。

### 3.3 画外画像四个家族各一段（plan-only）

产物：`.../gc_plan_only_20260907/off_screen_four_families.json` 及各家族 `gc_offscreen_*_anchor_plan_summary.json`。

用采样器同一套 FOV 判据（85°、`.93` 水平裕量）核对锚点不在视锥、竞争者在视锥：

| family | room_id | success | anchor_off_screen | legal_camera_count |
|---|---|---|---|---:|
| apartment | `legacy_ue_apartment_0000_v1` | true | true | 56 |
| authored | `aea_loc3_social_rebuild_v1` | true | true | 4 |
| mp3d | `habitat_mp3d_example_17DRP5sb8fy` | true | true | 4 |
| hm3d | `hm3d_val_00800_TEEsavR23oF` | true | true | 47 |

距离上限用放量默认 `[1.5, 6.0]` m。

### 3.4 给 G-E 的放量干跑命令（精确）

在 `48g-jump`、worktree `/data/jzy/tmp/wt-grok-pilot46-GC`、分支 `grok/pilot46-fixes-20260907-gc`：

```
export PYTHONPATH=src:tmp/native_python_addons_v1
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
  tools/dataset/build_qa_batch_manifest.py scaleup-dry-run \
  --config examples/dataset/qa_pilot_46_20260907.json \
  --catalog examples/rooms/packages/catalog.json \
  --registry examples/runtime/source_asset_runtime_profiles.json \
  --sounds /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/p10_pilot46_manifest_20260907_v3/batch_sounds.json \
  --output tmp/gc_scaleup_dryrun_7x50_20260907 \
  --seed 20260907 --episodes-per-room 50 \
  --batch-id qa_scaleup_7x50_20260907
```

已生成目录存在时该入口拒绝覆盖，G-E 换新 `--output` 即可。不执行 GPU、不碰 `attempt_01`。

## 4. 没做完的部分和原因

| 项 | 三分法 | 说明 |
|---|---|---|
| 放量 350 段里 2 段空调无独立声音身份 | 证据缺失 | `no_distinct_compatible_sound_identity`；repeat 缺额已是 0。声池没有给该资产更多可区分身份，未改映射凑绿。 |
| 画外未单独再跑 kujiale | 题义不适用 | 验收写四个家族；本项用 apartment / authored / mp3d / hm3d。kujiale 在 7×50 干跑清单里已有画外槽位。 |
| plan-only 用 120 帧而非 240 | 题义不适用 | 同层与画外是几何约束；未渲 16 秒成片。生产请求仍是 240/15。 |
| `uniform_in_bin` 未做 HM3D 实测矩阵 | 证据缺失 | 选项已接线，默认仍 `any_legal_in_bin`。覆盖报告已改用实测 5° 档，不再按请求箱宣称覆盖。 |
| 挂墙/吊顶资产、人工试听、五段抽听 | 题义不适用 | 非 G-C 范围。 |
| `batch_sound_pool.py` 未改 | 题义不适用 | 分配已在 manifest；该模块只接合 P7/事件 PCM。 |

## 5. 对后续接口的要求

- `lock_same_floor_region(space, rng, region=None, room=None) -> (bounds, floor_y)`；容差 `SAME_FLOOR_Y_TOLERANCE_M = 0.3`。多层从 `room_package.subrooms` 的 `*_floor_<y>` 解析。
- 画像字段：`anchor_visibility` ∈ {`in_fov`,`off_screen`}；`competitor_visibility` 同；`distance_range_m` 默认 `[1.5,4.5]`，放量默认 `[1.5,6.0]`。
- `sound_selection.clip_span_fit_policy` 只接受 `filter_to_remaining_budget_then_uniform`。
- `histogram_separation_5deg(angles)`：闭开 5° 档，`requested_bin_is_not_coverage=true`。outcome 走 `collect_batch_outcomes` 的 `achieved_separation_histogram_5deg`。
- Repeat 公式：`program_seconds_for_durations(..., relation="repeat", repeat_index=i)` ≡ `2 * duration[i] + other + 2 * gap_s`。缺额码 `fixed_sound_identities_exceed_profile_clip_budget`。换身份成功时 `sound_selection.identity_substitution_applied=true` 且带 `repeat_actor_id`。
- 交叉表：`class_pair_condition_group_crosstab(rows)` / `format_class_pair_condition_group_crosstab(table)`。
- 干跑：`prepare_scaleup_dry_run(template, registry, catalog, sounds, seed=..., episodes_per_room=50)`；CLI `scaleup-dry-run`。
- `speech_motion`：`speaker_moving` 不要求竞争者静止。见 v2 旁注。

## 6. 需要 owner 拍板的地方

1. 画外画像按任务书第 2 节 G-C.4 做了（2026-09-03「画外声音才是头条」）。owner 若不要，删画像即可，代码把 `off_screen` 当合法取值而不是默认。
2. 放量距离上限写成 `[1.5, 6.0]` m，未改 5 秒片长上限、未改 13 秒节目预算。
3. 2 段空调身份缺额要不要进放量 blocker：本项未把「repeat 缺额 0」扩成「一切预分配缺额 0」。若 owner 要求后者，需要给便携空调补可区分事件身份，而不是放宽匹配。
4. plan-only 用 120 帧。若 G-E 必须用 240 帧再证一次同层，可以同一入口改 `frame_count`，不必改采样器。
