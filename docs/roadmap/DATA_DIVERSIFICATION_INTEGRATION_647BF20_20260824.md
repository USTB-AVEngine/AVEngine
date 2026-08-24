# Data-diversification integration checkpoint 647bf20 (2026-08-24)

## Current server state

The sole implementation integration worktree is:

`/data/jzy/code/AVEngine-codex-diversification-integration`

It is clean on `codex/data-diversification-integration@647bf20`. Canonical
`main@82823f2` is also clean and remains untouched; its four main-only commits
change only `docs/roadmap/MENTOR_REPORT_20260824.md`.

The current CUDA-hidden full unit regression is:

`/data/jzy/tmp/data_diversification_647bf20_full_unit_r1.log`

It completed with exit zero:

- 3,318 passed;
- 118 explicit environment/archive skips; and
- 52 subtests passed.

The combined QA/region targeted suite is:

`/data/jzy/tmp/data_diversification_647bf20_targeted_r1.log`

It records 95 passed. These tests are regression evidence, not native
production certification.

## Newly integrated MP3D evidence

### Selected12 OBJECT_ID and raw clues

The selected12 pipeline probes now have 12/12 native Habitat captures with 75
frames, OBJECT_ID runtime mapping, offline review and owner-inspected contact
sheets:

- capture: `/data/jzy/tmp/mp3d_selected12_object_id_8b1918c_r2`;
- review: `/data/jzy/tmp/mp3d_selected12_object_id_review_8b1918c_r2`;
- visual/record verifier:
  `/data/jzy/tmp/mp3d_selected12_object_id_evidence_8b1918c_r3`; and
- 12-to-12 raw clues:
  `/data/jzy/tmp/mp3d_selected12_object_id_raw_clue_fb3e5eb_r2`.

Zero modal pixels never affect legality or retention. Semantic-ID coverage
remains diagnostic. P1/amodal, occlusion and formal off-screen truth remain
pending. Old RLR audio correctly refuses to bind to the new visual receipt, so
new visual audio is pending for all twelve views.

### Ten-region source-route planner

The existing camera bank remains 10 regions x 4 different camera floor
positions. Integration now also contains a camera-independent source-route
planner backed by the real standalone Habitat PathFinder, declared MP3D
navmesh and .house region polygons.

The final fresh CPU route artifact is:

`/data/jzy/tmp/mp3d_region_source_route_bank_2b0d4f9_r4`

Independent parent verification is:

`/data/jzy/tmp/mp3d_region_source_route_bank_2b0d4f9_r4_verification.log`

It proves:

- 10 regions, 40 camera bindings, 40 globally unique route families;
- 160 motion cases and 75 frames per case;
- 7 canonical route geometries per region with no cross-family reuse except
  the deliberate exchange reversal;
- 10 exact exchange cases, all `both_moving`, with source identity preserved;
- 5,185 unique materialized frame points independently rechecked against the
  loaded PathFinder and corresponding region;
- 152 cases below the 0.60 m descriptor retained;
- no camera/visibility/difficulty filtering.

This is source-center CPU route authority. Articulated actor collision,
ground-contact application, native pixels/readback, RLR/audio and formal
Episode status remain pending.

## Newly integrated QA producer evidence

Question rows now carry explicit boolean `audio`/`video`
`modality_mask` values. The design CLI can explicitly declare
`--input-modalities audio video` and propagates that fact to every point
spec, twin and manifest. Missing legacy modality facts fail rather than being
invented.

Fresh CPU evidence:

- design: `/data/jzy/tmp/qa_modality_mask_design_cli_20260824_r2`;
- generator with refusal:
  `/data/jzy/tmp/qa_modality_mask_generate_cli_20260824_r1`; and
- comparison verifier:
  `/data/jzy/tmp/qa_modality_mask_generate_cli_20260824_verification_r1.log`.

The run authored 104 points and generated 405 ordinary questions plus 58
refusals. The refusal fraction is 12.527%; ordinary masks are
`audio=true, video=true`, refusal masks are `audio=true, video=false`, and
ordinary question content is unchanged from the no-refusal baseline.

This is a producer declaration/serialization result. It does not prove native
video exists, waveform activity, trainer/collator mask consumption, T5,
formal off-screen, exact48 or an accepted production batch.

## Current next dependency

The ten-region planner emits navmesh source-center positions. They must not be
used directly as Beagle actor-root, skin-root or emitter positions. The next
slice must bind each route through the existing M2 animation/root authority:

1. preserve authored joint animation and ground-contact correction;
2. materialize per-frame actor-world/skin-root transforms and tangent
   orientation;
3. execute one region's four camera-route bindings in Habitat with
   articulated collision/readback and OBJECT_ID;
4. review first-person and Topdown frames; and
5. run new RLR audio bound to those exact visual receipts.

That bridge is currently under read-only audit and has no completion claim.

## Remaining Goal boundary

The full goal remains active. Still missing are:

- native MP3D 4--8-view mini-pilots for all ten regions and new RLR audio;
- Apartment UE/SPEAR 4--8-view bank, readback, RIR and mini-pilots;
- Kujiale living/bedroom/balcony/kitchen/bathroom native UE/SPEAR certification;
- same-renderer target-only formal off-screen truth;
- T5 and strict 50/50+/-1 on a current accepted multiview batch;
- exact48 held-out excluded from training;
- trainer/collator modality-mask consumption, longer replay+video and
  multi-video integration;
- executable feline motion/runtime evidence and additional actor combinations;
- persistent SpearSim reset/session A/B and GPU/CPU/RLR/FFmpeg pipeline A/B.

The owner has deferred the two-GPU 100-step DDP run while GPUs are shared; it
is not claimed complete. All current outputs remain research-only,
`episode_counted=false`, formal dataset count zero. Entry into `main` remains
an explicit owner decision.
