# MP3D selected12 OBJECT_ID and raw-clue evidence (2026-08-24)

## Scope

This checkpoint closes one research pipeline-probe slice:

- zero semantic/OBJECT_ID pixels no longer fail visual capture;
- all twelve retained selected12 M1 requests were rerun with native OBJECT_ID;
- all twelve offline reviewers completed;
- the owner agent inspected RGB/OBJECT_ID contact sheets; and
- the twelve current captures were joined to 75-frame raw clues without
  candidate filtering.

This is not the ten-region 4--8-view production bank. It does not establish
P1/amodal masks, formal occlusion or off-screen truth, new native audio, a
question batch, or a counted Episode.

## Runner history and accepted batch

Two failed wrapper histories remain immutable:

- `/data/jzy/tmp/mp3d_selected12_object_id_7487fe0_r1` stopped at `00141`
  on the old semantic-pixel capture gate. Its relative `PYTHONPATH=src`
  did not bind a commit-specific worktree and is documented as installed
  producer behavior.
- `/data/jzy/tmp/mp3d_selected12_object_id_8b1918c_r1` repeated that failure
  because the wrapper again used a relative `PYTHONPATH` without changing
  directory. It is runner-configuration failure evidence, not evidence that
  merge `8b1918c` failed.

The accepted run uses the absolute integration source path:

`/data/jzy/tmp/mp3d_selected12_object_id_8b1918c_r2`

The wrapper log is:

`/data/jzy/tmp/mp3d_selected12_object_id_8b1918c_r2.log`

Before every case the wrapper queried GPU0 compute PIDs and stopped on any
nonempty result. All twelve cases started isolated and completed with wrapper
exit zero. Every receipt retains 75 frames, 15 fps, 240,000 ticks,
`research_only=true`, and `episode_counted=false`.

## Offline review and independent verification

All twelve offline reviewers completed under:

`/data/jzy/tmp/mp3d_selected12_object_id_review_8b1918c_r2`

The independent verifier and three contact sheets are:

`/data/jzy/tmp/mp3d_selected12_object_id_evidence_8b1918c_r3`

The verifier checks the exact twelve-directory set, each 75-frame integer
`object_id.npy`, runtime actor/source/object-ID axes, every retained
per-frame count and bbox, reviewer presence, and the research/formal boundary.
It reports pass. The earlier `r2` verifier failed on an incorrect bbox
schema assumption (list instead of the retained structured bbox object); that
failed root/log remains preserved.

OBJECT_ID positive-frame counts are:

| Order | Placement | actor0 | actor1 | Owner visual observation |
|---:|---:|---:|---:|---|
| 1 | 11523 | 75 | 75 | two large actors, masks aligned, lower-edge clipping |
| 2 | 00141 | 0 | 0 | no actor or mask in the inspected frames |
| 3 | 06084 | 0 | 0 | no actor or mask; adjacent-space view |
| 4 | 22097 | 0 | 0 | no actor or mask; close wall/door view |
| 5 | 22370 | 0 | 0 | no actor or mask; close-surface view |
| 6 | 06598 | 75 | 75 | two clear actors, masks and boxes aligned |
| 7 | 15328 | 0 | 0 | no actor or mask; doorway/wall dominates |
| 8 | 14165 | 75 | 56 | distant small actors; actor1 enters modal raster |
| 9 | 00994 | 0 | 74 | actor1 at extreme edge, down to 3 pixels at frame 74 |
| 10 | 09732 | 0 | 0 | no actor or mask; curtain/wall view |
| 11 | 06477 | 0 | 0 | no actor or mask; close wall view |
| 12 | 00839 | 75 | 0 | actor0 enters from extreme right edge |

The owner agent inspected frames 0, 37 and 74 for all twelve views at original
contact-sheet resolution. Positive masks align with visible actor surfaces.
Zero masks are retained as zero modal-pixel observations. They are never used
to delete a candidate or asserted to distinguish geometric off-screen from
full occlusion.

These twelve probes intentionally expose poor compositions and negative views.
They are useful pipeline evidence but are not the final per-room production
view bank.

## OBJECT_ID raw-clue integration

The consumer implementation is merged in integration commit `fb3e5eb`. It
validates the complete OBJECT_ID artifact, runtime object-ID mapping,
actor/source axes and all 75 frame records. It records modal count, bbox,
pixel presence and descriptor-only `in_frame` per source and frame.

The fresh 12-to-12 joined artifact is:

`/data/jzy/tmp/mp3d_selected12_object_id_raw_clue_fb3e5eb_r2`

The independent verifier is:

`/data/jzy/tmp/mp3d_selected12_object_id_raw_clue_fb3e5eb_r2_verification.log`

It reports:

- 12 input candidates and 12 output candidates;
- all candidates retained and `candidate_filtering="none"`;
- 75 raw-clue frames per view;
- OBJECT_ID count/bbox/presence equal to each source receipt on every frame;
- semantic-ID modal authority remains `null`;
- P1/amodal, occlusion and formal off-screen remain pending/null; and
- native audio joined for 0 views and explicitly pending for 12 views.

The consumer preserves the capture receipt's
`pending_fresh_runtime_a_b_and_visual_review` provenance string. This
checkpoint records the completed external A/B and owner visual review without
rewriting an immutable receipt or promoting the data to formal admission.

## Audio identity failure retained

The prior 11523 RLR receipt is bound to the prior visual receipt. Attempting to
attach it to the new OBJECT_ID visual correctly fails closed:

`/data/jzy/tmp/mp3d_selected12_object_id_raw_clue_audio_mismatch_fb3e5eb_r1.log`

The error is `audio listener view identity points to a different visual
receipt`. No output directory was published. The accepted raw-clue artifact
therefore leaves audio pending for all twelve views. A new RLR run must bind
the new visual receipt; reuse by matching only a placement name is forbidden.

## Regression

The clean integration full unit suite ran CPU-only with CUDA hidden:

`/data/jzy/tmp/data_diversification_fb3e5eb_full_unit_r1.log`

It completed with exit zero:

- `3305 passed`;
- `118 skipped`; and
- `52 subtests passed`.

The skips are retained environment/archive skips listed in the log; they are
not rewritten as passes.

## Remaining boundary

This slice proves current Habitat OBJECT_ID modal-raster capture and raw-clue
retention for the selected12 pipeline probes. It does not make zero pixels a
legality or difficulty gate. It does not claim whole-body amodal masks,
occlusion decomposition, formal off-screen questions, region-specific actor
routes, new RLR waveform truth, or production-room certification. Formal
dataset count remains zero.
