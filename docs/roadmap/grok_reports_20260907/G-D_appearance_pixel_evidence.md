# G-D 外观与像素证据

日期：2026-09-07。分支：`grok/pilot46-fixes-20260907-gd`。对照任务书第 2 节 G-D、第 3 节 G-D 验收、第 5 节报告格式。

## 1. 改了哪些文件（路径），提交号

提交 。本项改动：

- `src/avengine/rooms/qa_evidence.py`：非人类最少像素提到与人类同量级（512，placeholder）；颜色成分改比例阈值；设备 `finish` 优先、其次 `body_color`；`build_pixel_appearance_review` 显式传参；新增 `visible_pixel_frames` 与 `bbox_touches_frame_edge_frames`。
- `tests/unit/test_qa_evidence_appearance.py`：8 像素假阳性、body_color 设备、出画截断计数。
- `tests/unit/test_p9_evidence_delivery.py`：与新门槛对齐的夹具（不再假设 8 像素即通过）。
- `docs/roadmap/PIXEL_VISIBILITY_SEMANTICS_20260907.md`：写明 `in_fov` 仍是 `target_pixels>0`（含全遮挡）。

未改：`tools/qa/audit_binding_feasibility.py` 及其测试、Codex 报告原文、原批 `attempt_01`。

## 2. 跑了哪些测试

在 `/data/jzy/tmp/wt-grok-pilot46-GD`，`PYTHONPATH=src:tmp/native_python_addons_v1`，`import avengine` 解析到本 worktree。

| 文件 | passed | failed | skipped |
| --- | --- | --- | --- |
| `tests/unit/test_qa_evidence_appearance.py`（9） | 9 | 0 | 0 |
| `tests/unit/test_p9_evidence_delivery.py`（13） | 13 | 0 | 0 |
| 合计 | 22 | 0 | 0 |

`test_nearly_invisible_eight_pixel_cat_is_not_observable` 与 `test_eight_white_tan_pixels_are_not_observable` 覆盖 8 像素假阳性。`test_device_with_only_body_color_is_reviewed` 覆盖设备字段缝。`test_bbox_touches_frame_edge_counts_truncation` 覆盖出画截断。

## 3. 验收产物与亲自核对

重审输出根（只写新目录，原批只读）：

`/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_pilot46_gd_rereview_20260907_v1/`

抽帧：`inspect_frames/` 与本机 `/tmp/gd_inspect/`。

### Jack Russell（`qa_pilot46_20260907_apartment_animal_animal` source2，`standard_white_tan`）

- 原批：`not_observable`，0 帧 reviewed。
- 重审：`reviewed`，`n_pass=43` / `n_frame_refs=43` / `n_checks=60`。字段仍是 `coat_profile.value`。
- 第 0 帧：狗在画面深处、很小；`sample_check.visible_pixels=5` → `not_observable`（新门槛 512，正确拒绝 8 像素级假阳性）。
- 第 39 帧：白底褐斑的 Jack Russell 在地毯上清楚可见，与 `reviewed` 一致。未为了让第 0 帧通过而降低门槛。

### 两只猫（`qa_pilot46_20260907_mp3d_animal_animal`）

- source1 `standard_blue`：`reviewed` 240/240（原批已 reviewed）。
- source2 `standard_ruddy`：`reviewed` 106 帧。
- 第 0 帧：门口一只灰蓝猫、左侧一只褐猫，都看得见。第 70 帧褐猫仍在左下。与截图一致。

### 6 段原先设备外观为 null

字段侧已不再为 null（finish 优先，否则 body_color）。像素侧是否 reviewed 取决于可见像素，不改门槛凑通过：

| 段 | 字段 | 原 status / value | 新 status / value | 说明 |
| --- | --- | --- | --- | --- |
| apartment_device_device source2 | body_color | not_observable / null | reviewed / white | 240 帧 reviewed |
| kujiale_animal_device source2 | body_color | not_observable / null | reviewed / black | 240 帧 reviewed |
| kujiale_human_device source2 | body_color | not_observable / null | not_observable / white | 字段已读到；bbox 240 帧贴边，像素审阅仍失败 |
| authored_a_human_device source2 | body_color | not_observable / null | not_observable / black | A 房过曝，画面几乎全白 |
| authored_b_device_device source2 | body_color | not_observable / null | not_observable / silver | B 房过曝 |
| authored_c_device_device source2 | body_color | not_observable / null | not_observable / warm_gray | C 房过曝 |

A/B/C 过曝是 G-B 的舞台/曝光问题，不是 body_color 缝。看过 `blender_overexp_f0000.png`：房间接近全白，外观像素证据确实没有。

## 4. 没做完的部分

- **题义不适用**：没有把 `in_fov` 改成可见；按任务书加新字段。
- **接口未实现**：无。挂墙/吊顶挂装不在 G-D。
- **证据缺失**：A/B/C 三段设备像素审阅仍 not_observable，根因是过曝成片，等 G-B 修曝光后用同一门槛重审。`kujiale_human_device` 设备贴边，placeholder 比例门槛下未过，未降门槛。

## 5. 对后续接口的要求

- `inspect_registered_appearance(..., minimum_color_pixels=512, dominance_ratio=1.25, component_fractions=...)` 全部 placeholder。
- `build_pixel_appearance_review(..., minimum_color_pixels=..., dominance_ratio=..., component_fractions=...)` 必须显式传参。
- 设备字段顺序：`finish` → `surface_finish` → `body_color`。review 写 `appearance_field_used`。
- 新计数：`visible_pixel_frames`、`bbox_touches_frame_edge_frames`，经 `annotate_pixel_visibility_semantics` / `annotate_achieved_conditions_visibility` 写入 pixel truth 与 achieved_conditions。
- `in_fov_frame_count` 保持 `target_pixels>0`。

## 6. 需要 owner 拍板

- 非人类 512 像素与比例阈值是 placeholder。若放量后大量合法外观被判 not_observable，应由 owner 改 placeholder，而不是执行侧偷降。
- A/B/C 过曝段的外观格子在曝光修好前不能当对照组。
