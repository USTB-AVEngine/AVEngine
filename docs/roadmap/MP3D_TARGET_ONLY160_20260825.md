# MP3D target-only160 runtime and agent-review checkpoint — 2026-08-25

## Scope

This checkpoint records the completed same-renderer target-only replay for the
existing 10-region / 40-view / 160-point MP3D native visual population. It does
not yet claim completed RLR audio, finalized QA, final exact48, source-type
diversity or dataset admission.

Fresh batch:

`/data/jzy/tmp/mp3d_region_target_only160_1540e5c_r1`

The sequential runner used explicit GPU0, checked that device before every
point, created a fresh output per point, and stopped on first failure. It ran
from integration identity `1540e5c` and completed:

- 160 / 160 points;
- 10 region instances;
- 40 actual camera placements;
- four motion cases per placement;
- zero failed progress records;
- approximately 27 GB of retained normal and target-only arrays;
- start `2026-08-25T00:08:09.786588+00:00`;
- completion `2026-08-25T00:41:48.616586+00:00`.

All outputs remain `research_only=true`, `episode_counted=false`, formal
dataset count zero, and visibility remains excluded from candidate legality.

## Independent verification

Verifier result:

`/data/jzy/tmp/mp3d_region_target_only160_1540e5c_verification_r1.json`

The first background verifier attempt exited before reading data because its
launch command omitted `PYTHONPATH=src:.`; it created no result or failure
artifact. The corrected r2 launch retained the first diagnostic log and
completed successfully.

For every point, the verifier independently reloaded and compared the retained
native160 source against the new normal pass. All shared RGB, depth, semantic,
OBJECT_ID, planned/actual actor, camera, sensor, listener, emitter and joint
arrays plus `frame_records.json` are exactly equal. Therefore the prior normal
first-person and Topdown review remains applicable to this batch.

For both target actors at all points, the verifier also:

- required target-only semantic and target-only runtime OBJECT_ID masks to be
  exactly equal;
- compared target root, joint, camera and sensor readback against the normal
  pass;
- recompiled the same-renderer visibility truth from raw arrays; and
- reevaluated off-screen entry evidence.

Observed maximum target readback errors are all exactly zero. The recompiled
truth equals every stored truth document.

The full-scene OBJECT_ID namespace remains diagnostic only: its masks differ
from actor semantic masks by 231,555 pixels for actor0 and 195,177 pixels for
actor1 across the batch. Normal actor semantic pixels remain modal authority
after the no-actor semantic-ID collision preflight.

## Visibility facts and off-screen candidates

Across 12,000 actor frames per axis, the batch records:

| Actor | out_of_view | fully_occluded | visible_occluded | visible_clear |
| --- | ---: | ---: | ---: | ---: |
| actor0 | 7,400 | 129 | 2,692 | 1,779 |
| actor1 | 7,278 | 906 | 2,360 | 1,456 |

The existing strict QuestionSpec entry evaluator finds 39 uniquely sided
off-screen-to-visible candidates: 27 left and 12 right. Regions 002 and 009
contain no such transition; this is an observed content fact, not a rejected
camera. The multi-view finalizer can retain a maximum exact-balanced subset of
12 left plus 12 right rows while preserving all 39 candidate records in its
batch report.

## Agent review

Overlay pages and manifest:

`/data/jzy/tmp/mp3d_region_target_only160_1540e5c_review_r1`

The agent inspected all 40 both-moving representative views at frames 0, 37
and 74. Overlay colors distinguish actor0 visible/occluded target pixels and
actor1 visible/occluded target pixels. The resulting per-view review document
is:

`/data/jzy/tmp/mp3d_region_target_only160_1540e5c_review_r1/agent_review.json`

Observed quality labels are:

- 20 strong-composition views;
- 6 mixed-composition views;
- 14 weak wall/curtain/doorway/empty-composition views.

Exposure is generally usable and target-only silhouettes align with normal
visible pixels. Regions 000, 004, 005, 006, 007 and 008 contain the strongest
ordinary compositions. Regions 002 and 009 are the clearest refinement
priorities; region 003 is also weak/mixed. Known MP3D mesh seams remain visible
in places.

These labels do not delete or invalidate a native-legal camera. Weak views are
retained as raw off-screen/difficulty evidence and prioritized for optional
small-yaw or neighboring-position refinement before wider ordinary-QA use.

## Downstream status

The 160 unique randomized AudioPrograms are retained at:

`/data/jzy/tmp/mp3d_region_40view_qa_design_1540e5c_r1`

The low-priority sequential RLR batch is running at:

`/data/jzy/tmp/mp3d_region_audio160_ef26330_r1`

Only after that batch passes may `finalize_mp3d_multiview_batch.py` bind the
current visual/audio population, select balanced off-screen/refusal rows and
run the non-dry exact48 selector.
