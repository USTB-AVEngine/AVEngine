# Apartment camera placement and motion descriptors — 2026-08-24

Status: implementation checkpoint; native multi-view certification remains
pending. This checkpoint is research-only, `episode_counted=false`, and keeps
the formal dataset denominator at zero.

## Concrete failures addressed

The earlier Apartment prefilter generated camera positions only by aiming at
living/kitchen/window semantic targets. Its 49 live-floor quality candidates
contained 10 candidates matching all four historical side/separation profiles,
but none carried the `kitchen_reverse` label. Increasing the generated straight
corridor pool to 256 did not change that outcome. The fresh stopped run is
retained at
`/data/jzy/tmp/.apartment_viewpoint_builder_motion_v1_r14.staging`; it failed
before output publication with `motion-feasible generated candidates do not
provide required roles: kitchen_reverse`. No threshold was lowered and no
native rerun followed automatically.

That result exposed two incorrect couplings:

1. Semantic target labels were creating the camera candidate space instead of
   describing otherwise legal cameras.
2. Historical side margins and continuous 60 cm source-root separation were
   being treated as legality gates. They are task-difficulty descriptors;
   endpoint exchange and path crossing are legal when the native actors and
   pixels remain valid.

Git identity, version numbers, types and ordinary tests could not prevent
these failures because the old algorithms explicitly encoded the wrong
candidate semantics. No hash, frozen contract, baseline, schema or permanent
gate was added.

## Current implementation boundary

`src/avengine/camera_placement.py` now provides a room-agnostic lazy iterator.
It crosses height-specific camera-center regions with the explicit yaw grid
`0, 30, ..., 330` degrees. It uses only the retained `RuntimeObstacleMap`,
navmesh clearance, rigid clearance, sample spacing and an optional live-floor
mask supplied by the native builder. It imports no Simulator/PathFinder and
knows nothing about semantic targets, actors, source paths, visibility or
ranking.

Apartment consumes those target-agnostic placements first. Target-directed
yaws remain optional supplements, and semantic focus roles are attached only
after horizontal FOV/LOS evaluation. Missing living/kitchen/window coverage is
reported as a non-enforcing diagnostic. A vectorized position/height/yaw
farthest-point shortlist bounds the later diversity search; it does not inspect
semantic roles, source pairing, side margins or pair distance.

The prefilter preserves the complete floor/navmesh/rigid-clearance placement
catalog before that review shortlist. Existing raster/corridor quality checks
therefore choose which proxy views are surfaced for review without erasing the
underlying legal camera iterator output.

Source corridors remain separately generated from the intersection of the
legal source-center regions at every declared actor height. Both directions of
each corridor are retained.

`src/avengine/qa/motion_feasibility.py` is shared by the viewpoint evidence and
QA batch author. The old `feasible`/`passed` fields remain readable, but their
records now explicitly define them as `difficulty_profile_match` and state
that candidate filtering is disabled. The deterministic representative pair
for every motion mode is retained even when sources exchange endpoints, cross,
come within 60 cm, or fail the historical side-margin profile.

For each selected camera and representative path pair the prefilter writes 75
identity-bound frame descriptors:

- source1/source2 positions;
- XY and XYZ pair-distance series;
- exact continuous XY closest fraction and minimum;
- closest discrete frame;
- crossing, endpoint-exchange and camera-side-order changes;
- per-source camera-relative azimuth/elevation and side margin.

Projection, in-frame state, occlusion and visible-pixel counts are preserved as
`pending_same_renderer_native_execution` with null values. They must come from
UE/SPEAR normal and target-only passes using the same camera and renderer; the
Habitat proxy cannot fill them.

## Review finalization boundary

`tools/m6x/finalize_apartment_viewpoint_review.py` performs an atomic,
no-clobber retained-proxy finalization. Four or more exact retained poses create
only a native-pixel-pending proxy bank, which the production viewpoint loader
rejects. Fewer than four create only `BLOCKED.json`.

The current round-four evidence was finalized without modification at:

`/data/jzy/tmp/apartment_viewpoint_review_round4_finalized_r1/BLOCKED.json`

It records three retained proxy poses, no bank, no Episode and no dataset item.

## Verification completed

- Pure camera/motion/Apartment/design tests: 50 passed.
- Builder, intended-floor probe and finalizer tests: 20 passed (18 + 2).
- Full unit regression after the changes: 3,078 passed, 118 skipped, and 52
  subtests passed in 315.75 seconds.
- The builder test uses a 90-degree explicit yaw grid only to keep the synthetic
  fixture small; the production CLI default remains 30 degrees and records the
  expanded yaw list.
- The earlier 42.7-second Python shortlist hotspot was profiled and replaced by
  an incremental NumPy farthest-point update; one representative prefilter
  profile then completed in 2.3 seconds.

Studio was not used to author or choose any pose.

## Still pending

- Fresh live-Habitat generation with the target-agnostic iterator.
- Agent review of its Topdown and Habitat proxy frames.
- Real UE/SPEAR first-person pixels and same-renderer per-frame visibility.
- Per-view listener/camera, RIR, side/off-screen and four-point mini-pilot
  evidence.
- A fourth retained proxy and a loader-eligible certified 4–8-view bank.
