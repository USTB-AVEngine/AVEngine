# Data-diversification integration checkpoint (2026-08-24)

> Historical snapshot. Current integration state and evidence are recorded in
> `DATA_DIVERSIFICATION_INTEGRATION_647BF20_20260824.md`. The sections below
> describe the earlier `37317f4` boundary and must not be read as current
> completion claims.

## Integration state

The server-only integration worktree is
`/data/jzy/code/AVEngine-codex-diversification-integration` on branch
`codex/data-diversification-integration`. The code baseline covered by this
checkpoint is `37317f4` and is clean. Canonical `main` remains clean at
`e29e0d2`; nothing in this checkpoint is merged to or pushed from `main`.

The most recent full unit log predates the Kujiale and expanded-MP3D merges:

`/data/jzy/tmp/diversification_integration_a4d5cf9_full_unit_r1.log`

It completed with exit 0 and records the following only for the older
`a4d5cf9` baseline:

- `3168 passed`;
- `118 skipped` for explicitly unavailable retained/archive authorities;
- `52 subtests passed`;
- duration `280.69 s`.

This is ordinary regression evidence. It does not prove every required native
run, room, actor type, question batch or trainer/throughput item, and it is not
a current full-suite claim for `37317f4`. Current focused integration logs are
cited below; a new full suite remains required after the active code slices
converge.

## Integrated owner corrections

### Generic camera/source architecture

- Apartment has a target-agnostic camera-center iterator with stable
  `0,30,...,330` yaw enumeration. Semantic room/target labels are downstream.
- Source/path geometry and camera placement are separate. Side margin and fixed
  60 cm comparisons remain difficulty descriptors and never filter legal
  candidates. Crossing and endpoint exchange are retained.
- `describe_source_pair_tracks` consumes exact caller-supplied frames without
  endpoint interpolation and records identity, distance, closest frames,
  observed order changes and listener-local angles. Projection/occlusion remain
  pending unless native evidence is attached.

Apartment review still has only three retained proxy views and correctly emits
a blocked artifact instead of a 4--8-view production bank. No new UE run was
started for that incomplete bank.

### MP3D Habitat-native

- Intended-floor native probe code is committed; six views were rerun from
  clean code with passing PathFinder island/floor, RGB, depth, semantic and
  Topdown receipts.
- Six dynamic 75-frame captures from `509838e` record actual emitter-link
  positions and same-renderer semantic pixel counts without requiring every
  actor to be visible. Views 01/02/04/05 are fully off-screen for both actors,
  view03 is intermittently visible, and view06 only retains actor1.
- Explicit runtime-track binding compares equal to all six `[75,2,3]` emitter
  arrays and native visible-pixel arrays; projection/in-frame/occlusion remain
  explicitly pending.
- The target-agnostic native-floor catalog contains all 4,608 placements from
  128 floor positions x three heights x twelve 30-degree yaws, with no semantic,
  source/path, visibility or difficulty filter.
- A follow-up native export removed the old 128-sample cap: all 632 passing
  positions from the 860-pixel scan are now retained, producing the authoritative
  expanded 22,752-placement catalog. The 4,608 result remains a bounded first run.

Key fresh artifacts:

- `/data/jzy/tmp/mp3d_viewpoint_native_probes_839cfdf_r1`;
- `/data/jzy/tmp/mp3d_dynamic_visibility_509838e_viewXX_r1`;
- `/data/jzy/tmp/mp3d_runtime_tracks_ec387e9_r1`;
- `/data/jzy/tmp/mp3d_camera_placement_catalog_c9c2274_r1`;
- `/data/jzy/tmp/mp3d_floor_samples_all_8c4e7df_r1`;
- `/data/jzy/tmp/mp3d_camera_placement_catalog_all_8c4e7df_r1`;
- `/data/jzy/tmp/mp3d_probe_subset_37317f4_r1`.

The expanded catalog is now fully joined to the retained 75-frame emitter-link
track. All 22,752 descriptors are preserved; 3,639 have full analytic joint
centre-frustum coverage, 1,061 partial coverage and 18,052 zero coverage. None
is removed. The bounded selected-twelve subset covers twelve floor positions,
eight yaws and all three heights, with independent M1 requests. Integration
verification records `134 passed` and the independent artifact verifier reports
`status=pass` at:

`/data/jzy/tmp/mp3d_probe_subset_37317f4_r1_verification.log`

No selected-twelve request has run a fresh static or dynamic native capture
yet, and no MP3D view has a complete visual+audio+question four-point
mini-pilot.

### Kujiale current full-home

The living source/visual profile is now bound to current
`/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1`; the builder rejects
the old `live_v2` mismatch before planning. A fresh eight-view CPU output passes
the current binding and the old profile produces only `FAILED.json`.

This does not supply current package/launch endpoint/runtime map evidence,
native UE/SPEAR first-person/Topdown/readback, room-wide 4--8 certification,
M3/RLR qualification or any four-point mini-pilot. The current stage can still
only count human+Beagle as conservatively executable.

The target-agnostic five-room CPU catalogs are now merged. Their complete
0.25 m sampling lattice contains 1,163 floor positions and 41,868 placements:
27,396 living, 5,508 bedroom, 5,508 balcony, 1,188 kitchen and 2,268 bathroom.
Room labels remain post-hoc annotations and no 60 cm, side, exchange,
visibility, composition or difficulty filter was used. The independent record
verifier reports `status=pass`; the exact focused integration suite records
`85 passed` at:

`/data/jzy/tmp/kujiale_room_catalogs_integration_92fbcaf_85_unit_r1.log`

This is still room-polygon plus descendant-footprint CPU authority, not native
navigation, UE/SPEAR pixels/readback or mini-pilot evidence.

### QA onset and motion

The fresh 192-point CPU batch now covers all four motion cases. TA and T7 each
contain exactly 96 moving and 96 still answers; first onset spans frames 2--6,
second onset has 12 values, and the first/second gap has 33 values. All 64 twins
retain exactly the same schedule. Ninety points below the old 60 cm descriptor
remain present.

Fresh audit:

`/data/jzy/tmp/qa_onset_motion_2163763_r1_audit.json`

It is CPU design/question evidence only. A real audio renderer has not yet
consumed these point-level schedules. Legacy report-only T4/T9 imbalance remains
an explicit rejection, and exact48 still requires a real accepted multiview
batch.

### Native FOA trainer

The separate Spatial-Omni server branch is clean at `e422e7a`. Processor
alignment proves 80,000x4 FOA, encoder 50, k4 projector 12 and 12 placeholders.
A fresh GPU0 native-FOA+video smoke completed two optimizer steps plus validation
and generation with finite losses:

`/data/jzy/tmp/so_native_foa_video_c13a493_gpu0_r1`

This proves real collation/forward/backward/update for one tiny smoke. It does
not prove two-GPU DDP, long-run training or formal evaluation.

### P1/P2 readiness

The existing same-room persistent SPEAR session and overlapped RAW_READY ->
CPU FINAL_READY controller still dry-resolves successfully at:

`/data/jzy/tmp/strict2h_room_batch_reset_canary_2163763_dry_r1.json`

Its checked request remains `execution_authorized=false` and requires physical
GPU1, which is currently occupied by another user's process. No runtime canary
or serial-vs-pipeline A/B is claimed.

## Remaining Goal blockers

The full Goal remains active. At minimum, still required are:

1. Apartment 4--8 native UE/SPEAR views, per-view M1/listener/RIR identity and
   four-point mini-pilots.
2. Kujiale all five room instances, current package/endpoint/M3/RLR evidence,
   native reviews, more executable actor types and mini-pilots.
3. MP3D post-hoc room-instance binding, actual per-frame camera/listener
   readback persistence, 4--8 certified views per room, multiple
   rooms/types and visual+audio+question mini-pilots.
4. Real accepted multiview question batches, strict answer reports and exact48
   held-out output.
5. Real two-GPU 100-step DDP when two permitted GPUs are idle.
6. Persistent SpearSim reset canary and GPU/CPU pipeline serial A/B evidence.
7. Multi-room native audio/RIR, T5/refusal/off-screen runtime evidence, cat and
   other registered actor execution where the actual stage supports them.

No formal dataset denominator is incremented: all work remains research-only,
`episode_counted=false`, formal count zero, and `main` admission is an explicit
owner decision.
