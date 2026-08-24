# MP3D same-renderer target-only and off-screen checkpoint — 2026-08-25

## Scope and claim boundary

This checkpoint closes the missing Habitat-native authority for distinguishing
`out_of_view`, `fully_occluded`, `visible_occluded`, and `visible_clear` on a
research mini-pilot. It does not admit an Episode, complete the accepted
multi-view batch, satisfy the refusal quota, or create the exact-48 held-out.
All captures remain `research_only=true`, `episode_counted=false`, and formal
dataset count zero.

Implementation commits are:

- `9ead2e0 feat(m5): capture Habitat target-only visibility`;
- `d5e1036 feat(qa): ground MP3D offscreen entry questions`.

No Studio placement, new hash, frozen baseline, or candidate gate was added.
Visibility is retained after native execution and does not change camera or
source legality.

## Native target-only design

The normal pass remains the existing MP3D Habitat-Sim capture with RGB, metric
depth, semantic ID, and OBJECT_ID. After that simulator closes, each actor is
replayed alone through a fresh Habitat `NONE` blank stage using:

- the same installed Habitat renderer and sensor specifications;
- the exact same M1 camera pose and every sensor readback;
- the same actor bundle, semantic ID, root transform, and joint quaternion
  blocks at each of 75 frames;
- no physics step and no source/camera replanning.

Each target-only pass stores semantic and OBJECT_ID arrays plus actor root,
joint, camera, and sensor readbacks. The actor semantic mask must equal the
target-only runtime OBJECT_ID mask before it can be retained. The normal
semantic pass and the two target-only semantic passes are then compiled by the
existing `same_renderer_same_camera_modal_target_only_semantic_masks_v1`
authority.

The normal full-scene OBJECT_ID raster remains a raw diagnostic, not modal
visibility authority. On the region-000 canary, its runtime-object masks did
not equal the collision-free actor semantic masks:

- actor0: 37,377 semantic pixels versus 42,415 OBJECT_ID pixels, with 7,064
  differing pixels;
- actor1: 8,610 semantic pixels versus 9,678 OBJECT_ID pixels, with 1,536
  differing pixels.

The no-actor semantic preflight proves that semantic IDs 210 and 211 are absent
from the stage before actor creation. Therefore the normal actor semantic
raster, not the ambiguous full-scene OBJECT_ID namespace, is used for modal
pixels. In each blank target-only pass, semantic and OBJECT_ID masks were
exactly equal.

## Native canary A/B

Region 000 both-moving capture:

- `/data/jzy/tmp/mp3d_target_only_wip_region000_p003_gpu0_r1`;
- `/data/jzy/tmp/mp3d_target_only_wip_region000_p003_gpu0_r2`.

The existing independent capture comparator reloaded every declared NPY array
and the normal frame records, target-only frame records, and pixel truth JSON.
It returned `status=pass` with an empty differing-artifact list.

Independent verifier:

`/data/jzy/tmp/mp3d_target_only_9ead2e0_verification_r1.json`

It independently reloaded the arrays, recomputed semantic/OBJECT_ID mask
agreement and pixel truth, and compared every target-only root, joint, camera,
and sensor readback with the normal pass. All maximum errors were exactly zero.
The region-000 point keeps both actors inside the view for all 75 frames and
classifies both as partially occluded, principally by the floor, wall geometry,
and each other.

The first attempted canary stopped before creating output because the installed
Object-ID runtime requires an explicit external RLR SDK root even for visual
imports. The rerun supplied the already pinned external runtime package; no
partial capture directory had been written or reused.

## Real out-of-view to visible evidence

A post-hoc diagnostic over the retained native160 semantic/pose arrays selected
an existing autonomous camera/source combination; it did not regenerate or
filter either route:

- region: `17DRP5sb8fy:region:005`;
- point: 11, both actors moving;
- actual placement: `mp3d_camera_placement_15758`;
- capture: `/data/jzy/tmp/mp3d_target_only_wip_region005_p011_gpu0_r1`.

The independent verifier reports:

- actor0: 9 `out_of_view`, 7 `visible_clear`, 59 `visible_occluded`;
  entry from the left at frame 9, followed by partial occlusion from frame 16;
- actor1: 63 `out_of_view`, 4 `fully_occluded`, 8 `visible_occluded`;
  entry from the right at frame 63, full occlusion at frames 64–67, and visible
  reappearance at frame 68.

These are target-footprint facts, not inferences from a zero/low modal pixel
count. In particular, actor1 has a nonzero target-only footprint but zero
normal pixels during frames 64–67, which proves full occlusion rather than
off-screen status.

## Agent visual review

The agent reviewed RGB, normal semantic masks, both target-only masks, and a
composite overlay at representative frames.

- region-000 review:
  `/data/jzy/tmp/mp3d_target_only_wip_region000_p003_review_r1/target_only_contact_sheet.png`;
- region-005 transition review:
  `/data/jzy/tmp/mp3d_target_only_wip_region005_p011_review_r1/target_only_contact_sheet.png`.

Exposure and geometry are coherent. Target-only silhouettes align with the
normal visible actor pixels. Region 005 is visually empty at frames 0 and 8;
actor0 appears at the lower-left boundary at frame 9, while actor1 first forms
a small right-boundary footprint at frame 63, becomes completely hidden by
scene geometry, and then reappears.

## Grounded off-screen questions

Final mini-pilot artifact:

`/data/jzy/tmp/mp3d_region005_p011_offscreen_minipilot_d5e1036_r1`

The QA tool calls the same extracted off-screen entry evaluator used by the
registry-bound QuestionSpec path. Before selecting a side, that evaluator
binds and revalidates the existing same-renderer pixel truth. The artifact
contains two binary questions:

- actor0/source1 enters left at frame 9;
- actor1/source2 enters right at frame 63.

With fixed options `[left, right]`, the displayed-key balance is A:B = 1:1 and
the semantic source balance is source1:source2 = 1:1. The result is
`pass_research_offscreen_minipilot`, not a production Episode.

## Verification

The focused visual, pixel-visibility, question-spec, answer-balance, and
mini-pilot suites report:

```text
72 passed
```

The full branch unit regression at `d5e1036` reports:

```text
3364 passed, 118 skipped, 52 subtests passed in 312.62s
```

Log: `/data/jzy/tmp/data_diversification_d5e1036_full_unit_r1.log`.
Ruff check passed, the new files pass Ruff formatting, and `git diff --check`
passed. A clean-integration merge is the next code check. The next data tasks
remain accepted multi-view scaling, refusal rows, and real exact-48 selection.
