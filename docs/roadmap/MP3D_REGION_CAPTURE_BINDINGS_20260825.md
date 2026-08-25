# MP3D ten-region capture binding checkpoint — 2026-08-25

## Scope

This checkpoint turns the independent target-agnostic camera bank and
PathFinder route bank into a post-hoc native mini-pilot schedule. Camera poses
never generate, move, reject or shorten actor routes. Projection descriptors
rank execution order only; every legal camera/route candidate remains
retained.

Studio was not used. Camera yaw candidates come from the existing complete CPU
camera catalog, which already enumerates the catalog-declared 30-degree yaw
grid at every floor/height slot.

Implementation commits on `codex/mp3d-region-native-visual-runner` are:

- `622e789 feat(m5): schedule post-hoc region capture bindings`;
- `2821229 feat(m5): refine capture yaw after route planning`.

## All-region actor tracks

The existing materializer was executed independently for every region into:

`/data/jzy/tmp/mp3d_region_actor_tracks_all_d888738_r1`

It occupies about 248 MiB. The independent verifier is:

`/data/jzy/tmp/mp3d_region_actor_tracks_all_d888738_r1_verification.json`

The verifier reloaded every case through the exact M2 asset/action/hash loader
and rebuilt every apply plan. Counts are derived from the input route bank:
ten regions, 160 motion cases and 320 independent actor tracks. Each region
has four route families, four camera references, all four motion cases and 75
frames per actor. This is CPU input evidence only; native execution, collision,
visibility and RLR remain pending for these new cases.

## Post-hoc camera/yaw binding

The final binding artifact is:

`/data/jzy/tmp/mp3d_region_capture_binding_bank_2821229_r1`

For each selected target-agnostic camera floor/height slot, the builder:

1. reloads every yaw declared by the camera catalog;
2. authors an M1 request at the exact same position, calibration and listener;
3. projects both route source centers for all 75 frames and four motion cases;
4. retains full per-frame projection records for selected bindings and summary
   records for every yaw pairing;
5. selects one camera-slot/yaw to route-family bijection per region for
   mini-pilot scheduling;
6. schedules the four motion cases for each selected viewpoint.

The output derives its counts from the input artifacts rather than freezing a
numeric baseline. For the current inputs they are ten regions, 40 camera slots,
480 yaw candidates, 1,920 yaw/camera/route-family summaries, 40 selected M1
requests and 160 scheduled visual+audio+QA+gate points.

Every selected request retains the original slot translation, calibration and
listener by exact value. Only its catalog-existing yaw and request ID differ.
Each region uses every camera slot and every route family exactly once, so the
schedule does not duplicate one convenient view/family across the room.

## Projection descriptors and legality boundary

The descriptors retain camera-local XYZ, forward depth, horizontal/vertical
tangent, listener distance, azimuth/elevation and center-in-frustum for each
source/frame. A source center behind the camera remains present with null
tangents. No descriptor deletes a camera, route or motion case.

Thirty-degree yaw scheduling materially improves the raw coverage: all input
regions obtain positive horizontal source-center alignment. Three regions
(002, 003 and 009) still have zero complete root-center frustum frames because
the floor-level actor root is vertically outside the pinhole test. This is
reported, not gated. The dog mouth/mesh can still enter the image; only native
RGB/OBJECT_ID review can determine whole-actor visibility.

The initial selected-yaw-only artifact remains historical at
`/data/jzy/tmp/mp3d_region_capture_binding_bank_622e789_r1`. It retains its
alternatives but does not author 30-degree yaw refinements and is superseded
for execution scheduling.

## Verification

The final independent verifier is:

`/data/jzy/tmp/mp3d_region_capture_binding_bank_2821229_r1_verification.json`

It derives expected counts from the supplied viewpoint/catalog/route artifacts,
then verifies every selected M1 request against fresh camera-pose authoring,
checks that translation/calibration/listener equal the original slot, checks
every scheduled case identity/path and four-motion coverage, and confirms all
request files are exactly the selected set. It reports `status=pass`.

Module tests and Ruff check/format pass. The adjacent camera, route,
materializer, apply-plan and native-runner suite reports `49 passed`.

## Pending

- Execute the 160 scheduled native visual points; current completion is one
  region000 both-moving point.
- Produce corresponding RLR and QA/gate evidence per point.
- Review true first-person RGB/OBJECT_ID and Topdown per viewpoint.
- The schedule remains `research_only`, `episode_counted=false`; it does not
  change the formal denominator.
