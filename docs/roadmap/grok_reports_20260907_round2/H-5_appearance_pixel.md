# H-5 外观与像素字段

日期：2026-09-07。分支：`grok/pilot46-fixes-round2-20260907-h5`。对照第二轮任务书 H-5、审计 R6/K4、第 5 节六段报告格式。`import avengine` 解析到 `/data/jzy/tmp/wt-grok-pilot46-H5/src/avengine/__init__.py`。

## 1. 改了哪些文件（路径），提交号

提交 `a219ae4`（本报告随后的 stamp 提交只改这一行）。本项改动：

- `src/avengine/rooms/qa_evidence.py`：actor 级 reason 从逐帧 check 上提为 `registered_appearance_value_classifier_not_implemented`（`gap_category=interface_not_implemented`）。未改像素/颜色门槛。
- `src/avengine/qa/batch_delivery.py`：`finalize_batch_episode` 在 `achieved_from_facts` 之后只加 `attach_visibility_semantics`（调用 `annotate_pixel_visibility_semantics` / `annotate_achieved_conditions_visibility`）。不碰 H-4 的曝光闸门 `try/except`。
- `src/avengine/rooms/qa_delivery.py`：读入 `pixel_visibility_truth.json` 后立刻 `annotate_pixel_visibility_semantics`；`_decorate_actor` 按 `finish` → `surface_finish` → `body_color` 填 actor.appearance。
- `src/avengine/qa/batch_coverage.py`：只改外观 reason 映射。`appearance_review_missing` 且该资产 appearance_review 记了分类器缺口时，覆盖行 `state=interface_not_implemented`。未改 `failed_episodes` 分支（H-2）。
- `examples/runtime/source_asset_runtime_profiles.json`：两台电视补 `realized_attributes.body_color=black`（任务书写 `examples/registry/`，实际登记与 `_asset_registry` 读取的是这份 runtime 表）。
- `tests/unit/test_qa_evidence_appearance.py`、`tests/unit/test_qa_batch_delivery.py`、`tests/unit/test_qa_batch_coverage.py`。

未改：Claude 审计器、第一轮产物与报告原文、声学渲染、`room_package.py`、合并脚本、`exposure_gate.py`、Studio。未把 annotator 写回 attempt_02。

## 2. 跑了哪些测试

环境：`PYTHONPATH=src:tmp/native_python_addons_v1`，`PYTHONDONTWRITEBYTECODE=1`，解释器 `/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`。

| 文件 | passed | failed | skipped |
| --- | --- | --- | --- |
| `tests/unit/test_qa_evidence_appearance.py`（11） | 11 | 0 | 0 |
| `tests/unit/test_qa_batch_delivery.py`（5） | 5 | 0 | 0 |
| `tests/unit/test_qa_batch_coverage.py`（16） | 16 | 0 | 0 |
| `tests/unit/test_p9_evidence_delivery.py`（17，qa_delivery/qa_evidence 连带） | 17 | 0 | 0 |
| 合计 | 49 | 0 | 0 |

新测：`test_unsupported_registered_values_use_classifier_gap_reason`、`test_actor_reason_is_classifier_gap_when_value_has_no_classifier`、`test_attach_visibility_semantics_writes_pixel_and_achieved_fields`、`test_delivery_call_sites_wire_visibility_annotators`、`test_classifier_gap_appearance_defer_is_interface_not_implemented`、`test_appearance_review_missing_without_classifier_gap_stays_deferred`。`pytest -rs` 无 skip 行。

## 3. 验收产物与亲自核对

### 接线

| 函数 | 现在的调用点 |
| --- | --- |
| `annotate_pixel_visibility_semantics` | `qa_delivery.finalize_qa_episode`（读 capture 像素真值之后）；`batch_delivery.attach_visibility_semantics` |
| `annotate_achieved_conditions_visibility` | `batch_delivery.attach_visibility_semantics`，由 `finalize_batch_episode` 在写 `achieved_conditions.json` 前调用 |

`inspect.getsource` 钉住上述两个 finalize 函数含这些名字。

### 干跑（不覆盖 attempt_02）

输入只读：

`tmp/qa_pilot46_rerun_20260907_v1/episodes/qa_pilot46_20260907_authored_b_device_device/attempt_02/episode/capture/pixel_visibility_truth.json`

及同段 `batch_review/achieved_conditions.json`（原文件仍无新字段）。

输出新目录：`/data/jzy/tmp/h5_pixel_annotate_20260907/`

| 文件 | 核对 |
| --- | --- |
| `pixel_visibility_truth.json` | source1/source2 均有 `visible_pixel_frames=240`、`bbox_touches_frame_edge_frames=0`、`in_fov_frame_count=240`；frame 0 有 `in_fov` / `bbox_touches_frame_edge`；`visibility_semantics_authority=qa_evidence.annotate_pixel_visibility_semantics` |
| `achieved_conditions.json` | `event_002`/`event_003` 在原 `in_fov_frame_count` 旁出现 `visible_pixel_frames` 与 `bbox_touches_frame_edge_frames` |

attempt_02 原文 `visible_pixel_frames` 仍不存在。

### 分类器缺口（不实现分类器、不降门槛）

逐帧 check 原先已写 `registered_appearance_value_classifier_not_implemented`，actor 级 reason 为 null。现在 actor 记录同样写该 reason，覆盖表把对应 `appearance_review_missing` 行记成 `interface_not_implemented`。像素不够看、但分类器存在的外观仍是 `deferred_by_rule`。

11 个设备色值实例（7 个 distinct）无分类器：silver×4、white_satin×2、warm_gray、beige、light_gray、sandstone、light_gray_fabric。3 种毛色无分类器：dark_sable、standard_sable、standard_seal_point。未为这些值补 HSV 谓词。

### 两台电视

`generated_television_flat_panel_16_9_central_pedestal_research_v1` 与 `generated_television_flat_panel_16_9_two_splayed_feet_research_v1` 原先既无 `finish` 也无 `body_color`。颜色来自 `finalized.glb` 的 `Watertight_BaseColor`（去掉 UV 空白后的 texel）：central_pedestal 中位 RGB `[51,53,54]`、two_splayed_feet 中位 RGB `[25,25,27]`，饱和度低、`metallicFactor=0`。登记为 `body_color=black`（与闹钟/手机等深色塑料设备同一词，且已有 dark/black 分类器）。不是猜的色名。

### K4 更正：G-D 把两只狗写反

第一轮 G-D 报告把 `qa_pilot46_20260907_apartment_animal_animal` 第 0 帧 5 像素写成 Jack Russell。对照 `attempt_01` 像素真值与 G-D 重审 `appearance_review.json`：

| actor | 资产 | 第 0 帧 visible_pixels | 第 0 帧判定 | 第 39 帧 |
| --- | --- | --- | --- | --- |
| source1 | 黄拉布拉多 `standard_yellow` | **5**（target 3455，`visible_occluded`） | not_observable（新门槛 512，正确） | 7607，pass |
| source2 | 杰克罗素 `standard_white_tan` | **2970** | **pass**（G-D 重审） | 9524 可见，但 `warm_brown=0.119 < 0.12` → 该帧 not_observable |

第 0 帧 5 像素的是被扶手椅遮住的黄狗 source1；杰克罗素 source2 第 0 帧 2970 像素是 pass。未为第 0 帧 5 像素降门槛。

## 4. 没做完的部分

- **题义不适用**：没有把 `in_fov` 改成可见；挂墙/吊顶不在 H-5。
- **接口未实现**：上列 11 个设备色值与 3 种毛色仍无分类器。覆盖表现记 `interface_not_implemented`，actor reason 为 `registered_appearance_value_classifier_not_implemented`。实现分类器是后续接口，不是本轮把阈值凑过。
- **证据缺失**：attempt_02 的 19/20 段 capture 像素真值与 `achieved_conditions.json` 仍无新字段（只读，不回写）。新字段出现在下一次 `finalize_qa_episode` / `finalize_batch_episode`，以及本次干跑新目录。A/B/C 过曝段的像素审阅未在本项重跑。

## 5. 对后续接口的要求

- `annotate_pixel_visibility_semantics(truth) -> dict`：复制像素真值，按实例写入 `in_fov_frame_count` / `visible_pixel_frames` / `bbox_touches_frame_edge_frames`，按帧写入 `in_fov` / `bbox_touches_frame_edge`。`in_fov` 仍是 `target_pixels > 0`。
- `annotate_achieved_conditions_visibility(achieved, pixel_truth) -> dict`：在 `anchor_event_measurements` 的 `in_fov_frame_count` 旁加同样两个计数。
- `attach_visibility_semantics(achieved, pixel_truth) -> (achieved, truth|None)`：batch 收口的唯一接线函数。
- 分类器缺口：check 与 actor 的 `reason` 必须是 `registered_appearance_value_classifier_not_implemented`，`gap_category=interface_not_implemented`。覆盖表外观 defer 码（`appearance_review_missing` 等）遇到该 reason 时 state 为 `interface_not_implemented`，不得记成 `deferred_by_rule` 或 `evidence_missing_or_unsampled`。
- 设备登记字段顺序：`finish` → `surface_finish` → `body_color`。电视现在有 `body_color`。

## 6. 需要 owner 拍板

- 非人类 512 像素与比例阈值仍是 placeholder，本项未调。
- 7 个无分类器的设备色词与 3 个毛色词要不要补分类器：本项明确不补。若放量后这些格子必须出题，由 owner 定分类器，而不是执行侧偷降门槛。
- 两台电视 `body_color=black` 来自 GLB 贴图计量。若 owner 要改成 `matte_black`/`charcoal` 等已有词，可以改登记，不必重猜颜色。
- 任务书写登记表在 `examples/registry/`；电视实际只存在 `examples/runtime/source_asset_runtime_profiles.json`，本项改的是后者。
