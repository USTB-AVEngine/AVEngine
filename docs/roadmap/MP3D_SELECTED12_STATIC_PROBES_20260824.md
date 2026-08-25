# MP3D selected-twelve static native probes (2026-08-24)

## Scope and code state

The corrected full-catalog descriptor artifact at
`/data/jzy/tmp/mp3d_probe_subset_37317f4_r1` supplied twelve independent M1
camera requests. They are bounded pipeline probes only. They are not twelve
yaw bins, not a per-room 4--8-view bank and not a camera-legality filter on the
remaining 22,740 placements.

The probes ran sequentially from clean server integration commit `5d4e683` on
the otherwise idle physical GPU 0. No Studio placement or selection was used.
Each request started its own native Habitat simulator, loaded the declared
MP3D scene and NavMesh, bound the intended floor from its Topdown declaration,
and wrote a fresh/no-clobber output. The first nonzero exit would have stopped
the sequence.

## Fresh native evidence

The output root is:

`/data/jzy/tmp/mp3d_selected12_static_5d4e683_r1`

All twelve `receipt.json` files report `status=pass`. Across the complete set:

- intended PathFinder island is exactly `1`;
- maximum horizontal floor snap is `0.0 m`;
- maximum intended-floor Y error is `0.0 m`;
- camera eye heights are the requested 1.2, 1.5 and 1.8 m variants;
- every RGB/depth/semantic observation has native `240x320` calibration;
- every required native Topdown report passes;
- RGB standard deviation spans 22.8915--57.6270;
- positive finite depth fraction spans 0.986927--1.0; and
- semantic unique-ID count spans 1--49.

The corrected independent verifier is:

`/data/jzy/tmp/mp3d_selected12_static_5d4e683_r1_verification_r2.log`

It aligns every receipt with its selected placement and exact M1 request, then
checks room/request identity, floor/island facts, observation shapes and
Topdown success. Image statistics are retained as descriptors and do not
affect candidate legality.

The first verifier attempt is deliberately retained at:

`/data/jzy/tmp/mp3d_selected12_static_5d4e683_r1_verification.log`

It added an unrequested `positive finite depth fraction > 0.99` assertion.
Placement order 8 measured 0.986927 and exposed that mistake. The corrected
`r2` verifier removes that image-quality gate and only requires finite,
well-formed statistics already covered by the passing native receipt. No
capture was deleted or rerun because of that descriptor.

## Agent visual inspection

I inspected the exact native RGB, depth, semantic and Topdown outputs in:

- `contact_sheet_orders_01_04.png`;
- `contact_sheet_orders_05_08.png`;
- `contact_sheet_orders_09_12.png`; and
- `topdown_selected12_pose_overlay.png`.

The aggregate overlay uses the native PathFinder bounds and the same 5 cm map
resolution as the retained Topdown image. It marks each floor position and yaw
instead of treating twelve identical unannotated NavMesh backgrounds as a
camera-distribution review.

Raw composition observations are:

- orders 1, 2, 4, 6, 9 and 12 expose comparatively complete room/corridor
  compositions;
- orders 3, 10 and 11 face close wall or curtain surfaces;
- order 5 is dominated by a close doorway/wall composition; and
- orders 7 and 8 contain large foreground obstruction.

These are retained composition and occlusion clues, not camera failures.
Order 11's single semantic ID and the low visual diversity of several views
show that analytic source-centre coverage plus pose novelty is insufficient for
the final per-room view bank. Native image quality may become a continuous
post-hoc preference after capture, but it must not remove these legal raw
candidates from the full catalog.

## Remaining boundary

Static probes contain no actors and do not prove dynamic camera/listener/actor
readback, emitter tracks, per-frame semantic visibility, occlusion, audio/RIR,
questions, mini-pilots or formal Episodes. Dynamic selected-twelve execution
must wait for the separately committed 75-frame actual-pose persistence path,
then run fresh from the merged code and be reviewed again.

All outputs remain research-only, `episode_counted=false`, and formal dataset
count zero.
