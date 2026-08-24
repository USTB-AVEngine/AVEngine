# MP3D ten-region native 160-point checkpoint — 2026-08-25

## Scope and execution

The complete four-point visual schedule from
`mp3d_region_capture_binding_bank_2821229_r1` was executed fresh on physical
GPU2. The batch contains every selected camera slot, its post-hoc catalog yaw,
one unique route family and all four motion cases.

Root artifact:

`/data/jzy/tmp/mp3d_region_native160_2821229_r1`

The root contains the immutable `plan.json`, 160 per-point logs and 160 native
capture directories. Total size is about 14 GiB. Every point independently
checked GPU2 occupancy, created a no-clobber output and stopped the batch on
any nonzero exit; no stop was triggered.

## Machine verification

The independent verifier is:

`/data/jzy/tmp/mp3d_region_native160_2821229_r1_verification.json`

It derives expected point/view counts and the motion axis from `plan.json`,
then reloads every receipt, frame record and core array. It verifies:

- all scheduled outputs and logs exist;
- every receipt is a 75-frame research-only native Habitat capture;
- plan/camera/route/motion identity matches exactly;
- exact `pts_ticks=i*3200`;
- RGB/depth/semantic/OBJECT_ID co-registration and expected actor/readback
  array axes;
- finite depth, pose and emitter arrays;
- no active actor-pair collision frame;
- all four motion cases under every viewpoint.

The verifier reports `status=pass`: 160/160 points, 40/40 viewpoints, ten
regions, and four motion cases. These counts are derived from the plan rather
than embedded as a permanent admission baseline.

## Raw visibility distribution

Visibility is reported only as a descriptor:

- 90/160 points have at least one actor OBJECT_ID pixel;
- 70/160 points have zero actor pixels;
- 27/40 viewpoints have actor pixels in at least one motion case.

Per-region visible-view counts are:

- region000: 4/4;
- region001: 3/4;
- region002: 0/4;
- region003: 2/4;
- region004: 2/4;
- region005: 3/4;
- region006: 4/4;
- region007: 4/4;
- region008: 4/4;
- region009: 1/4.

Zero pixels do not change camera or source legality. Region002 and most of
region009 need supplemental camera positions for visible-actor coverage, but
their current wall/occlusion views remain useful raw off-screen candidates once
formal same-renderer authority exists.

## Parent visual review

All 40 viewpoints were reviewed at frames 0/37/74 together with Topdown camera
and route overlays. Review pages are under:

`/data/jzy/tmp/mp3d_region_native160_2821229_r1_review`

Conclusions:

- exposure is acceptable across all views; none has the dark Apartment preview
  problem;
- Topdown camera positions and route geometries are distinct rather than copied;
- region000/006/007/008 provide consistently visible actors;
- region001/005 include one weak or wall-dominated view;
- region003/004 include two weak/occluded views;
- region002 is dominated by curtains/walls and provides no visible actor view;
- region009 has one visible view and three wall/counter-dominated views;
- MP3D mesh seams/broken surfaces remain scene-asset quality descriptors.

This review prevents the successful runtime batch from being mislabeled as 40
high-quality visible viewpoints.

## Code and regression identity

The schedule implementation is in commits `622e789` and `2821229`, documented
by `15b7851`. It is integrated in AVEngine code merge `c74dcea`, whose complete
CPU suite reports:

`3413 passed, 118 skipped, 52 subtests passed`.

The later native execution checkpoint is documentation-only and does not alter
that tested code state.

## Pending

- supplement weak regions/views before calling the bank high-quality accepted;
- RLR/audio, QA, T5, refusal and gate evidence for the remaining points;
- repeat A/B on the subset proposed for acceptance;
- formal OBJECT_ID/off-screen authority and Episode admission remain pending;
- formal dataset denominator remains zero.
