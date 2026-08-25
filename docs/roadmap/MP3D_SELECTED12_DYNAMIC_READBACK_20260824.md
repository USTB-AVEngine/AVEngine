# MP3D selected-twelve dynamic readback evidence (2026-08-24)

## Scope and code state

The corrected selected-twelve M1 requests were captured sequentially from the
clean server integration code at `60b1ede`. That code includes post-render
agent/camera and sensor snapshots, capture-time M1 identity, derived logical
listener pose, planned actor transforms, planned applied skin-root transforms,
actual post-render skin-root readback, actual emitter-link positions, native
RGB/depth/semantic observations and explicit actor/source axes.

These twelve captures are bounded whole-scene pipeline probes. They are not the
ten-region 4--8-view production bank, do not delete any legal catalog entry and
do not use Studio.

## Fresh native run and machine verification

The fresh/no-clobber dynamic root is:

`/data/jzy/tmp/mp3d_selected12_dynamic_60b1ede_r1`

All twelve invocations completed with 75 frames, `status=research_only`,
`episode_counted=false` and exit zero. The corresponding offline reviews are:

`/data/jzy/tmp/mp3d_selected12_review_60b1ede_r1`

All twelve reviewers completed without mutating a capture or assigning an
admission verdict. The independent cross-view verifier is:

`/data/jzy/tmp/mp3d_selected12_dynamic_60b1ede_r1_summary_verification.log`

It verifies every capture-time M1 value snapshot, post-render pose artifact,
75-frame PTS grid, actor/source/semantic axes and frame-record/NPY equality. It
also proves that all twelve camera-only variants retain exactly equal planned
actor, planned/actual skin-root and emitter-link tracks. Observed maxima are:

- camera plan/readback absolute error: `5.960464477539063e-08`;
- derived-listener/agent absolute error: `1.302549423876087e-08`; and
- planned/actual skin-root absolute error: `2.3841724416940023e-07`.

The first-view detailed verifier is retained at:

`/data/jzy/tmp/mp3d_selected12_dynamic_60b1ede_r1/mp3d_camera_placement_11523_verification.log`

## Native semantic-ID observations and corrected boundary

The literal `[75,2]` arrays currently count pixels equal to semantic IDs 210 and
211. Their positive-frame counts by selection order are:

| Order | Placement suffix | ID 210 frames | ID 211 frames |
|---:|---:|---:|---:|
| 1 | 11523 | 75 | 75 |
| 2 | 00141 | 0 | 0 |
| 3 | 06084 | 0 | 0 |
| 4 | 22097 | 0 | 0 |
| 5 | 22370 | 0 | 0 |
| 6 | 06598 | 55 | 2 |
| 7 | 15328 | 0 | 0 |
| 8 | 14165 | 75 | 56 |
| 9 | 00994 | 0 | 57 |
| 10 | 09732 | 0 | 0 |
| 11 | 06477 | 0 | 0 |
| 12 | 00839 | 75 | 0 |

These are raw semantic-ID raster observations, not certified complete actor
modal visibility. A real contradiction was found during agent frame review:
selection order 6, frame 0 visibly contains both rendered Beagles in RGB, while
the semantic raster contains zero pixels for both IDs 210 and 211. Comparing
that RGB frame against the same-camera no-actor static probe finds 1,115 changed
pixels; their leading semantic values are background IDs 138 (597 pixels) and
6 (518 pixels). At frames 37 and 74, ID 210 partially appears, while ID 211
remains absent despite RGB actor evidence.

Therefore this run does not authorize any of the following inferences:

- zero semantic-ID pixels means the actor is off-screen;
- zero semantic-ID pixels means the actor is occluded;
- positive semantic-ID pixels cover the complete visible actor; or
- the current arrays constitute formal P1/off-screen truth.

The raw arrays remain retained. Downstream raw-clue output must name them as
semantic-ID raster counts/bounds and keep actor modal visibility, off-screen and
occlusion `null/pending` until an articulated OBJECT_ID/semantic completeness
path is implemented and natively checked.

## Agent visual inspection

I inspected frames 0, 37 and 74 for every view in:

- `dynamic_contact_sheet_orders_01_04.png`;
- `dynamic_contact_sheet_orders_05_08.png`; and
- `dynamic_contact_sheet_orders_09_12.png`.

Order 1 is the only view with two large, clearly visible actors throughout the
three inspected frames. Orders 6 and 8 contain useful but partial/obstructed
actor views. Orders 9 and 12 contain small or one-sided actor evidence. The
remaining views primarily face other rooms, close surfaces or walls relative to
the retained local two-Beagle route. This is expected evidence that analytic
emitter-centre frustum coverage cannot replace native pixels or room-specific
actor paths.

The magnified order-6 frame-0 comparison is retained as:

`/data/jzy/tmp/mp3d_selected12_dynamic_60b1ede_r1/inspection_06598_f0_bottom_4x.png`

## Remaining boundary

No RLR/SoundSpaces render, independent RIR identity, decoded-waveform activity,
question generation, per-region actor route, mini-pilot or formal Episode is
claimed by this run. The ten-region bank still needs region-specific paths and
native execution. Formal dataset count remains zero.
