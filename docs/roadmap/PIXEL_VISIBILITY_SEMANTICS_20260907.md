# Pixel visibility semantics: `in_fov`, `visible_pixel_frames`, `bbox_touches_frame_edge_frames`

Date: 2026-09-07. Owner: G-D (`src/avengine/rooms/qa_evidence.py`). Placeholder counts, not human answerability.

## `in_fov` / `in_fov_frame_count`

A frame is **in FOV** when native pixel truth `target_pixels > 0`.

- `target_pixels` is the target-only footprint (the instance is inside the camera frustum / image rectangle, including pixels that another surface covers).
- **Fully occluded frames remain in FOV.** `state=fully_occluded` still has `target_pixels > 0` and `visible_pixels = 0`.
- Out-of-view frames have `target_pixels = 0`, `visible_pixels = 0`, and a null bbox.

This is the meaning already used by `achieved_conditions.in_fov_frame_count` (`facts.visibility[*].target_pixels > 0`). G-D does **not** silently change `in_fov` to mean “visible”.

## `visible_pixel_frames`

Count of frames with `visible_pixels > 0` (`visible_clear` or `visible_occluded`). Fully occluded and out-of-view frames are excluded.

Visibility rate that only looks at occlusion (`visible_pixels / target_pixels` on in-view frames) still ignores leaving the frame. `visible_pixel_frames` is the complementary tally.

## `bbox_touches_frame_edge_frames`

Count of frames whose `target_bbox_xyxy_px` touches any image edge.

- Bbox is the exclusive-max xyxy stored by pixel truth: `[x0, y0, x1, y1]` with `x1 = columns.max() + 1`, `y1 = rows.max() + 1`.
- `resolution_hw` is `[height, width]`.
- Touching means `x0 <= 0` or `y0 <= 0` or `x1 >= width` or `y1 >= height`.
- Out-of-view frames (null bbox) do **not** count as edge-touching.

A beagle that is on-screen for 240/240 frames but whose bbox sits on the bottom edge is `in_fov=240`, `visible_pixel_frames=240`, `bbox_touches_frame_edge_frames=240`.

## Where the fields are written

Functions in `src/avengine/rooms/qa_evidence.py`:

- `bbox_touches_frame_edge(bbox, resolution_hw)`
- `summarize_pixel_visibility_semantics(frames, *, resolution_hw, window_frames=None)`
- `annotate_pixel_visibility_semantics(truth)` — copy of pixel truth; per-frame `in_fov` / `bbox_touches_frame_edge`; per-instance counts. Does not change `state`.
- `annotate_achieved_conditions_visibility(achieved, pixel_truth)` — copy of achieved_conditions; adds the two new counts next to existing `in_fov_frame_count`.
- `build_pixel_appearance_review(...)` records the per-actor summary under `pixel_visibility_semantics`.

Live `compile_pixel_visibility_truth` / `achieved_from_facts` still live in `pixel_visibility.py` and `batch_delivery.py` (not G-D owned). Callers should run the annotators after those writers. Re-review outputs write annotated copies; original `attempt_01` files are not overwritten.

## Appearance floor (related)

Non-human `inspect_registered_appearance` / `build_pixel_appearance_review` take explicit

- `minimum_color_pixels` (placeholder default **512**, human order of magnitude; no hidden 8)
- `dominance_ratio` (placeholder default 1.25)
- `color_component_fractions` (placeholder ratio thresholds for tricolor / black-white / red-white / white-tan / coats)

All of these are labeled `placeholder` in the returned `appearance_thresholds` / `calibration=placeholder_nonhuman_appearance_v1`. The human shirt rule remains `inspect_coarse_top_color` at 512 pixels and 1.6× dominance.

Device appearance reads `finish`, then `surface_finish`, then `body_color`. The review records `appearance_field_used`. If neither finish nor body_color exists, value stays null and the reason says so.
