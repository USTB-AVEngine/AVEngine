# Data-diversification integration checkpoint (2026-08-24)

## Integration state

The server-only integration worktree is
`/data/jzy/code/AVEngine-codex-diversification-integration` on branch
`codex/data-diversification-integration`. The code baseline covered by this
checkpoint is `a4d5cf9` and is clean. Canonical `main` remains clean at
`e29e0d2`; nothing in this checkpoint is merged to or pushed from `main`.

The current full unit log is:

`/data/jzy/tmp/diversification_integration_a4d5cf9_full_unit_r1.log`

It completed with exit 0 and records:

- `3168 passed`;
- `118 skipped` for explicitly unavailable retained/archive authorities;
- `52 subtests passed`;
- duration `280.69 s`.

This is ordinary regression evidence. It is not evidence that every required
native run, room, actor type, question batch or trainer/throughput item exists.

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

Key fresh artifacts:

- `/data/jzy/tmp/mp3d_viewpoint_native_probes_839cfdf_r1`;
- `/data/jzy/tmp/mp3d_dynamic_visibility_509838e_viewXX_r1`;
- `/data/jzy/tmp/mp3d_runtime_tracks_ec387e9_r1`;
- `/data/jzy/tmp/mp3d_camera_placement_catalog_c9c2274_r1`.

The 4,608 catalog has not yet been joined/ranked into a new native probe subset,
and no MP3D view has a complete visual+audio+question four-point mini-pilot.

### Kujiale current full-home

The living source/visual profile is now bound to current
`/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1`; the builder rejects
the old `live_v2` mismatch before planning. A fresh eight-view CPU output passes
the current binding and the old profile produces only `FAILED.json`.

This does not supply current package/launch endpoint/runtime map evidence,
native UE/SPEAR first-person/Topdown/readback, room-wide 4--8 certification,
M3/RLR qualification or any four-point mini-pilot. The current stage can still
only count human+Beagle as conservatively executable.

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
   generic view catalogs, native reviews, more executable actor types and
   mini-pilots.
3. MP3D join/selection from the 4,608 catalog, 4--8 certified views, multiple
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
