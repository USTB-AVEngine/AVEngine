# MP3D one-native-canary-per-region checkpoint — 2026-08-25

## Scope

This run executes one selected both-moving point from each region in the
post-hoc 30-degree yaw binding bank. It is an early cross-region native audit
before expanding to all 40 viewpoints and four points per viewpoint.

The schedule authority is:

`/data/jzy/tmp/mp3d_region_capture_binding_bank_2821229_r1`

For each region, the canary chooses the selected binding with the strongest
retained projection descriptor tuple. Projection is only execution ranking;
zero projection never prevents native capture.

## Native artifact

All ten fresh captures are under:

`/data/jzy/tmp/mp3d_region_native_canary10_2821229_r1`

Every run used physical GPU2 after an empty compute-PID guard, the installed
Habitat prefix, one Simulator, two independently validated Beagle tracks and
75 explicit frames with no physics step. All ten receipts report
`research_only`, remain uncounted, and have no active actor-pair collision
frame.

The independent verifier is:

`/data/jzy/tmp/mp3d_region_native_canary10_2821229_r1_verification.json`

It cross-checks each capture's selected M1 request, route family and motion
case against the binding bank; validates all receipts/frame counts/collisions;
and records per-actor OBJECT_ID/semantic pixel coverage and RGB change without
turning visibility into a legality condition. It reports `status=pass`.

## First-person and Topdown review

Parent-reviewed contact sheets are:

- `/data/jzy/tmp/mp3d_region_native_canary10_2821229_r1_review/first_person_10regions_frames_0_37_74.png`;
- `/data/jzy/tmp/mp3d_region_native_canary10_2821229_r1_review/topdown_10regions_camera_routes.png`.

Review conclusions:

- regions 000, 001, 004, 005, 006, 007 and 008 are well exposed and show
  both moving actors;
- region003 is well exposed but only shows small/late actor coverage and needs
  a stronger visible viewpoint in the full four-view run;
- regions002 and 009 are well exposed but show no actor pixels in this canary;
  they remain raw off-screen/occluded candidates and need other scheduled
  viewpoints for visible coverage;
- no canary has the dark Apartment-preview exposure problem;
- MP3D scene-mesh seams and broken edges remain visible in several views and
  are scene-asset artifacts to retain in quality review.

The Topdown sheet confirms that each camera and both routes occupy the intended
region. In particular, region002/009 executed their routes despite zero pixels;
their zero result is a real view/render descriptor, not a missing path.

## Pixel evidence

Eight of ten regions have positive actor OBJECT_ID pixels. The two zero-pixel
regions also have identical RGB at frames 0 and 74, consistent with no rendered
actor surface in that view. Region003 has positive OBJECT_ID in only 10/11
frames for actor0/actor1. The remaining visible regions range from partial to
75/75-frame coverage.

Semantic and OBJECT_ID counts still differ in some regions, including 000,
005 and 007, consistent with the known semantic-mesh depth interaction. These
captures therefore remain raw descriptors and do not establish formal modal
visibility or off-screen truth.

## Pending

- Run all four scheduled viewpoints and all four mini-pilot points per
  viewpoint; current native completion is 10 of 160 scheduled points.
- Produce new RLR/QA/gate artifacts for these region-specific views. The earlier
  region000 RLR checkpoint uses a different visible binding and is not silently
  attached to this canary bank.
- Repeat A/B and review the views selected for eventual acceptance.
- Formal Episode count remains zero.
