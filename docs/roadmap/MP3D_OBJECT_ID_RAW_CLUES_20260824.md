# MP3D OBJECT_ID raw-clue bridge (2026-08-24)

## Scope and ownership

This slice is implemented only on the server worktree
`/data/jzy/code/AVEngine-codex-object-id-raw-clue`, branch
`codex/object-id-raw-clue`, from integration `7487fe0`. It adds the
consumer-side join from `current_visual`'s native OBJECT_ID raster to
`mp3d_raw_clues`. It does not start a Habitat/GPU capture, enumerate or
select viewpoints, alter candidate retention, or change formal episode
counting.

The code commit is
`2b0db75e0d8aedf8e4865bfbbfcec3399fb09a02`
(`feat(m5): join native object-id raw clues`). This checkpoint document is
committed separately.

## Evidence consumed

The bridge reuses the existing
`current_visual_review._object_id_readback_evidence` validator. When an
OBJECT_ID artifact is declared, it requires all current pose artifacts,
75 RGB/depth/semantic/object-ID frames, the OBJECT_ID sensor metadata, two
unique nonnegative runtime object IDs, exact actor/source/runtime-ID
axis bindings, and receipt/frame-record count and bbox agreement. Partial
OBJECT_ID evidence fails closed; it is not downgraded to a legacy capture.

The runtime IDs are invocation-local values read from
`articulated_object.object_id`; they are never treated as stable IDs across
runs. The raw output preserves them under
`identity_axes.runtime_object_id_axis_order`, each frame's source record,
and `object_id_raster_boundary`.

## Raw-clue boundary

For each source and frame with OBJECT_ID evidence, the bridge records:

- runtime object ID;
- native modal pixel count;
- native modal bbox;
- modal pixel presence and a corresponding `in_frame` descriptor.

Here `in_frame=true` means only that the native OBJECT_ID raster contains
positive modal pixels. A zero count is not interpreted as geometric
off-screen, full occlusion, or actor absence. The interpretation is stored
explicitly in `in_frame_status`.

The existing semantic-ID fields remain present as diagnostics: semantic count,
presence, bbox, and border margin can be retained, but semantic-ID presence or
absence is never actor visibility authority. For OBJECT_ID captures,
`native_projection_visibility` labels semantic IDs as diagnostic-only and
uses OBJECT_ID for the modal descriptor fields.

No P1/amodal or same-backend occlusion authority was added. Therefore
`p1_amodal_visibility.occlusion_by_frame`,
`p1_amodal_visibility.off_screen_by_frame`,
`object_id_raster_boundary.occlusion_by_frame`,
`object_id_raster_boundary.off_screen_by_frame`, and each frame's
`occlusion`/`off_screen` fields remain `null` with pending status
descriptions. Candidate legality and retention remain unchanged:
`candidate_filtering="none"`, `candidate_retained=true`, and the
OBJECT_ID boundary has `legality_effect="none"`.

No hash, frozen contract, baseline, or permanent gate was introduced.

## CPU evidence from existing accepted native runs

No GPU job was started. A fresh/no-clobber CPU join consumed these existing
native outputs:

- `/data/jzy/tmp/mp3d_object_id_native_732f264_06598_r2`;
- `/data/jzy/tmp/mp3d_object_id_native_732f264_11523_r2`; and
- `/data/jzy/tmp/mp3d_object_id_native_732f264_00677_out_of_frustum_r1`.

The resulting artifact is
`/data/jzy/tmp/mp3d_object_id_raw_clue_cpu_join_20260824_r1/raw_clue_cpu_join.json`.
Its independent verifier is
`/data/jzy/tmp/mp3d_object_id_raw_clue_cpu_join_20260824_r1_verification.log`.

All three records loaded as
`current_native_post_render_raw_clues_described`, with exactly 75 frames
and runtime IDs `[1, 36]`. The first two had modal pixels for both actors in
all 75 frames. The out-of-frustum case had zero modal pixels for both actors
in all 75 frames; the raw output retained `in_frame=false` for that modal
observation while keeping formal off-screen `null`. The verifier also
confirmed semantic actor visibility stayed `null`, P1/amodal and occlusion
fields stayed pending/null, and no candidate was filtered.

The current capture receipt's
`actor_modal_visibility_authority_status` remains
`pending_fresh_runtime_a_b_and_visual_review`; this slice preserves that
provenance status in `authority_status` and does not silently upgrade a
formal admission claim.

## Tests and quality

The focused raw-clue and current visual-review suite passed 44 tests. The
adjacent consumer/runtime suite passed 109 tests:

```
PYTHONPATH=src:. /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/pytest -q \
  tests/unit/test_mp3d_raw_clues.py \
  tests/unit/test_describe_mp3d_viewpoint_tracks.py \
  tests/unit/test_m5_current_visual.py \
  tests/unit/test_m5_current_visual_review.py \
  tests/unit/test_current_mp3d_dynamic_audio.py \
  tests/unit/test_mp3d_raw_audio.py
```

The exact test log is
`/data/jzy/tmp/mp3d_object_id_raw_clue_2b0db75_tests_r1.log` and records
`109 passed`. Server-side byte compilation, `git diff --check`, and clean
worktree evidence are in
`/data/jzy/tmp/mp3d_object_id_raw_clue_2b0db75_quality_r1.log`.

## Remaining boundary

This commit makes the OBJECT_ID modal evidence available to the raw-clue
join. It does not implement P1/amodal masks, formal occlusion decomposition,
formal off-screen QA, a new selected12 rerun, audio waveform claims, or any
Apartment/Kujiale route. Those remain separate slices. All consumed and
produced evidence remains research-only with `episode_counted=false`.
