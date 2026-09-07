# H-4 校验顺序与闸门

按 `docs/roadmap/GROK_FIX_TASKS_ROUND2_20260907.md` 第 4 节与第 2 节 H-4。
工作树 `/data/jzy/tmp/wt-grok-pilot46-H4`，分支
`grok/pilot46-fixes-round2-20260907-h4`。未 push、未合并 main、未动 Studio、
未用 GPU、未改阈值、未改 Claude 审计器、第一轮产物只读。H-3 拥有 catalog
相对路径与快照；H-5 拥有 `annotate_pixel_*` 接线。本项只做校验顺序、存在性、
闸门 ImportError、帧来源、`rgb.npy` 回退、EV 真值源、地板四份文件标签。

## 1. 改了哪些文件（路径），提交号

实现与本报告同提交（分支最新一次；短号见 git log）。

- `src/avengine/rooms/room_package.py`：生产装载改为
  `validate_room_package(resolve_room_package_paths(...))`，存在性检查在模板
  展开之后。相对 `tmp/` / `examples/` 等路径相对仓库根解析；传入
  `relative_roots` 时也相对 catalog 目录检查。`/Game`、`/Root` 与未展开
  `${VAR}` 仍不当文件系统路径。H-3 的 `_load_json(declared)` 行未改，避免抢
  catalog-relative 打开。
- `src/avengine/qa/exposure_gate.py`：记录 `frame_source.kind`
  （`capture/frames` 真帧 / `rgb.npy` / `batch_review/frames` 审阅抽帧）。
  无 PNG 捕获目录时从 `capture/rgb.npy`（及 `episode/capture/rgb.npy`）取
  第 0 / 中 / 末帧；审阅抽帧只在既无捕获 PNG 也无 `rgb.npy` 时使用。
  placeholder 阈值未改。
- `src/avengine/qa/batch_delivery.py`：删除 `except ImportError: pass`。
  新增 `apply_review_exposure_gate`：导入失败写
  `exposure_gate.status=unavailable` 且 `review.status=review_failed`。
  review 顶层与 `exposure_gate.frame_source` 都记录帧来源。未接线
  `annotate_pixel_*`。
- `tools/rooms/run_spear_residential_episode.py`：**删除**
  `AUTHORED_USD_EXPOSURE_BIAS_EV` 硬编码表与 `authored_usd_map_default`
  回落。曝光补偿只读 CLI / `visual_plan.camera` / `episode.resources` /
  `room_package` / `planning_inputs`。包字段是唯一真值源。
- `examples/rooms/packages/room_{a,b,c}.json`、
  `examples/rooms/packages/kujiale_0020_full_home_v1.json`：
  `floor_reference.status=depth_readback_fallback`，路径改到仓库内副本。
- `examples/rooms/packages/floor_reference/{room_a,room_b,room_c,kujiale_0020}/`
  （新）：copy-on-write 四份 UE 地板文件。`status=depth_readback_fallback`；
  `summary.hit_count` 改为 line-trace 命中数 0，不再沿用深度回读 64
  （64 留在 `method.depth_capture.hit_count` 与
  `summary.depth_readback_hit_count`）。未改 `tmp/gb_floor_reference_20260907`
  第一轮产物。
- 单测：`tests/unit/test_room_package_path_existence.py`、
  `tests/unit/test_floor_reference_measurement_kind.py`、
  `tests/unit/test_room_packages_p3.py`、
  `tests/unit/test_exposure_gate.py`、
  `tests/unit/test_qa_batch_delivery.py`、
  `tests/unit/test_authored_exposure_bias_source.py`（新）。

未改：`docs/TOOL_INDEX.md`（`tools/build_tool_index.py --check` exit 0）、
测量脚本 `classify_ue_floor_measurement`（未来新测量仍由测量工具写；本项
改的是已落盘的四份文件）、H-5 像素字段、H-3 catalog 相对打开。

## 2. 测试

项目 Python `/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`，
`PYTHONPATH=src:tmp/native_python_addons_v1`。`import avengine` →
`/data/jzy/tmp/wt-grok-pilot46-H4/src/avengine/__init__.py`。

```
pytest -q tests/unit/test_room_package_path_existence.py \
          tests/unit/test_floor_reference_measurement_kind.py \
          tests/unit/test_room_packages_p3.py \
          tests/unit/test_exposure_gate.py \
          tests/unit/test_qa_batch_delivery.py \
          tests/unit/test_authored_exposure_bias_source.py \
          tests/unit/test_qa_production_contracts.py
```

| 文件 | passed | failed | skipped |
|---|---:|---:|---:|
| `tests/unit/test_room_package_path_existence.py` | 7 | 0 | 0 |
| `tests/unit/test_floor_reference_measurement_kind.py` | 5 | 0 | 0 |
| `tests/unit/test_room_packages_p3.py` | 10 | 0 | 0 |
| `tests/unit/test_exposure_gate.py` | 5 | 0 | 0 |
| `tests/unit/test_qa_batch_delivery.py` | 4 | 0 | 0 |
| `tests/unit/test_authored_exposure_bias_source.py` | 2 | 0 | 0 |
| `tests/unit/test_qa_production_contracts.py`（回归，未改） | 20 | 0 | 0 |
| **合计** | **53** | **0** | **0** |

本项触碰的测试文件合计 33 passed / 0 failed / 0 skipped。无失败原文。

`tools/build_tool_index.py --check`：exit 0。

## 3. 验收产物的路径，以及亲自核对结果

### 3.1 七包生产装载顺序下缺失路径

装载顺序与 `tools/studio/run_qa_episode.py` 一致：catalog
`path_bindings` → `package_from_catalog_entry(entry, runtime=...)`，即
**先 `resolve_room_package_paths` 再 `validate_room_package`**。在仓库根 cwd
亲自跑；另在 `/tmp` cwd 用绝对路径读 JSON 再 resolve+validate（不依赖进程
cwd）。七包均为 0 缺失。

| room_id | 包文件 | missing |
|---|---|---:|
| `legacy_ue_apartment_0000_v1` | `native_apartment.json` | 0 |
| `aea_loc3_social_rebuild_v1` | `room_a.json` | 0 |
| `authored_compact_home_room_b_v1` | `room_b.json` | 0 |
| `authored_open_family_home_room_c_v1` | `room_c.json` | 0 |
| `kujiale_0020_full_home_v1` | `kujiale_0020_full_home_v1.json` | 0 |
| `habitat_mp3d_example_17DRP5sb8fy` | `mp3d_17DRP5sb8fy.json` | 0 |
| `hm3d_val_00800_TEEsavR23oF` | `hm3d_00800_TEEsavR23oF.json` | 0 |

第一轮「七包 0 缺失」是手动先展开再查得到的；本项在生产函数里先展开再查，
未展开的 `${AVENGINE_...}` 不再被 `_is_absolute_filesystem_path` 静默跳过。

从 `/tmp` 直接 `package_from_catalog_entry` 仍会
`FileNotFoundError: examples/rooms/packages/native_apartment.json`：这是
**打开包文件**相对 cwd，属 H-3，不是本项的存在性检查。存在性检查本身在
任意 cwd 下对已读入的包为 0 缺失。

### 3.2 闸门 ImportError 与帧来源

- `apply_review_exposure_gate` 在导入失败时：`status=review_failed`，
  `exposure_gate.status=unavailable`，不再 `pass`。单测
  `test_exposure_gate_import_error_fails_review` 用 monkeypatch 验证。
- 帧来源：`capture/frames` 真 PNG 优先；否则 Habitat `rgb.npy` 第 0/中/末
  帧；否则才用 `batch_review/frames` 审阅抽帧。review JSON 的
  `frame_source` 与 `exposure_gate.frame_source` 都写 `kind`。
- 单测：白帧/正常帧、`rgb.npy` 在有审阅抽帧时仍被选用、捕获 PNG 压过
  `rgb.npy`、仅审阅抽帧时 `kind=batch_review/frames`。阈值仍是
  placeholder 235 / 0.20。

### 3.3 EV 真值源

删除硬编码表。包内：A/C `exposure_bias_ev=-3.0`，B `-4.0`。单测钉住包字段，
并断言 `run_spear_residential_episode.py` 源码不再含
`AUTHORED_USD_EXPOSURE_BIAS_EV` / `authored_usd_map_default`。

### 3.4 地板四份文件

亲自读仓库内副本：

| 文件 | status | summary.hit_count | depth_readback_hit_count |
|---|---|---:|---:|
| `examples/rooms/packages/floor_reference/room_a/floor_reference.json` | `depth_readback_fallback` | 0 | 64 |
| `.../room_b/floor_reference.json` | `depth_readback_fallback` | 0 | 64 |
| `.../room_c/floor_reference.json` | `depth_readback_fallback` | 0 | 64 |
| `.../kujiale_0020/floor_reference.json` | `depth_readback_fallback` | 0 | 64 |

`method.line_trace.hit_count` 仍为 0。包 `floor_reference.status` 与文件一致。
Apartment 0.271 m 的 v3 文件未改，仍为 `measured`。

## 4. 没做完的部分

- **题义不适用 / 所有权在 H-3：** `package_from_catalog_entry` 打开
  `entry["room_package"]` 仍相对进程 cwd。本项只保证展开后的路径存在性相对
  仓库根/catalog 目录，不改打开包文件的基准目录。
- **题义不适用 / 所有权在 H-5：** 未接线
  `annotate_pixel_visibility_semantics` / `annotate_achieved_conditions_visibility`。
- **题义选择：** 四房地板仍是 depth 量化残差 0.0001953125 m，没有给地板加
  碰撞让 line trace 真正命中；只改标签与 `summary.hit_count`。测量工具
  `classify_ue_floor_measurement` 对 depth fallback 仍返回 `status=measured`
  （那是工具对新测量的默认词），已落盘四份文件不再用这个词。
- **证据缺失：** 未重跑 batch_review，因此旧 attempt 的 `review.json` 仍是
  第一轮产物（Habitat 段当时用审阅抽帧）。新收口才会写出新的 `frame_source`。
- **题义不适用：** 未改阈值、未删资产、未动 Studio、未覆盖 attempt_01。

## 5. 对后续接口的要求

- 生产装载必须 `resolve` 再 `validate`。不要改回
  `resolve_room_package_paths(validate_room_package(...))`。
- `missing_filesystem_paths(package, *, relative_roots=None)`：相对
  `tmp/`/`examples/` 等始终相对仓库根；其它相对路径在传入 catalog 目录时检查。
- `apply_review_exposure_gate(review, episode_root) -> dict`：ImportError →
  `exposure_gate.status=unavailable` 且 review 失败。
- `apply_exposure_gate` 在 `exposure_gate.frame_source.kind` 写
  `capture/frames` | `rgb.npy` | `batch_review/frames`。Habitat 无 PNG 时读
  `rgb.npy` 的 0 / mid / last，不要用审阅抽帧冒充第 0/120/239 帧。
- 曝光补偿只读包字段（及 CLI/plan 覆盖），不要再加地图路径硬编码表。
- 地板 depth 兜底文件：`status=depth_readback_fallback`，
  `summary.hit_count` 不得复用深度回读点数。

## 6. 需要 owner 拍板的地方；口径更正

**口径更正（任务书 H-4/H-6 要求写进本报告）：**

- G-B 报告「任务书写 −3 EV 与 B 房 −4 冲突需 owner 拍板」是**假冲突**。
  任务书全文没有任何 EV 数值。B 房 −4 来自旧交付与包字段，不是与任务书打架。
  本项按包声明保留 A/C −3、B −4，并删掉第二真值源。
- G-B 报告「`apply_exposure_gate` 未接入 `batch_delivery`」在合并 HEAD **已过期**。
  第一轮合并后 `batch_delivery.py` 已接线；本项修的是那处
  `except ImportError: pass` 会静默失效，以及帧来源家族不一致。

**不需要 owner 再拍板：** EV 表删除（只留包字段）；地板选「如实标签」而非修碰撞
（与第一轮 G-B 选择一致，本项只把 `status`/`hit_count` 说到与
`measurement_kind` 一致）。

若要真正的 line-trace 地板，需要另开测量任务。H-3 负责 catalog 相对打开与
plan 快照。H-5 负责像素语义字段接线。
