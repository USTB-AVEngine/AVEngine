# Habitat OBJECT_ID visibility route (2026-08-24)

## Scope and observed failure

The current MP3D semantic sensor draws the independent semantic scene graph
first and then overlays dynamic active-scene objects with `ObjectsOnly` while
retaining the first draw's depth. MP3D RGB and semantic geometry differ. In
the retained selected12 evidence, view `06598` frame 0 visibly contains both
dogs in RGB/depth while semantic IDs 210 and 211 both have zero pixels; the
same pixels retain background IDs 138 and 6. Root/link `semantic_id` readback
therefore cannot establish complete skinned-drawable coverage.

This slice adds a separate OBJECT_ID route. It does not reinterpret the old
semantic raster and does not claim P1, amodal visibility, occlusion, or
off-screen truth.

## Native target routing

`CameraSensor` now dispatches by `semanticTarget`:

- `SEMANTIC_ID` keeps the historical semantic-graph draw followed by an
  active-graph `ObjectsOnly` overlay when the graphs differ;
- `OBJECT_ID` and `DRAWABLE_ID` skip the semantic graph and draw the complete
  active scene graph exactly once; and
- the full active draw retains stage depth together with actors. It is
  deliberately not `ObjectsOnly`, which would let actors appear through walls.

The route policy is a small constexpr header shared with a native Corrade
test. The test checks all three targets and explicitly checks that OBJECT_ID
and DRAWABLE_ID use the full-active route with `activeSceneObjectsOnly=false`.
Building the test also compiles the complete HabitatCore, including the changed
`CameraSensor.cpp`.

## Current visual capture

The current MP3D visual configuration retains the original RGB, depth and
SEMANTIC_ID sensors and adds one co-located semantic-format sensor:

- UUID `rig_object_id`;
- target `SemanticSensorTarget.OBJECT_ID`;
- the same resolution, pinhole calibration and post-render pose readback as
  the M1 rig; and
- output `arrays/object_id.npy`, integer `[75,H,W]`.

For each newly instantiated actor the capture reads `actor.object_id` and
records the per-run bijection:

`actor_id / source_id / emitter_link_name -> runtime_object_id`

Runtime object IDs are scoped to one Simulator invocation and are never frozen
or compared across runs. Every frame records exact per-actor object-ID pixel
counts and bboxes. Zero pixels are retained and have no legality, retention, or
difficulty effect.

The existing SEMANTIC_ID raster remains available for room semantics and
diagnosis, with coverage completeness explicitly uncertified.

## Offline reviewer

The reviewer accepts older captures without OBJECT_ID as legacy/pending. For a
new receipt, it requires the complete object-ID metadata plus artifact and
checks:

- the fourth sensor's UUID, modality and all 75 actual sensor poses;
- integer `[75,H,W]` co-registration with RGB;
- unique runtime IDs and exact actor/source/emitter axis bindings;
- every frame-record count and bbox against the raster; and
- every per-actor summary against all 75 recomputed frames.

The synchronized local page adds a false-color OBJECT_ID panel and shows the
per-frame mapping and counts. Its authority remains
`pending_fresh_runtime_a_b_and_visual_review`.

## CPU/native evidence before clean-prefix build

- Current visual/reviewer focused suite: `45 passed`.
- The fresh native diagnostic build at
  `/data/jzy/tmp/habitat-object-id-native-test-precommit-r6` compiled the full
  HabitatCore and the native routing test; CTest reports `1/1 passed`.
- Earlier fresh diagnostic attempts remain as dependency-resolution history:
  r1 missed the explicit EGL target, r2 lacked Corrade TestSuite, and r3-r5
  exposed incompatible component selection between retained dependency roles.
  r6 uses the complete H5 Corrade/TestSuite role plus explicitly selected H11
  static Any* components for this header-only test target.

These results prove source compilation and policy behavior only. No Simulator
capture or GPU job ran.

## Authority boundary and next native commands

OBJECT_ID becomes candidate modal evidence only after a fresh clean-commit
runtime prefix is built and the following fresh/no-clobber runs are compared:

1. `mp3d_camera_placement_06598`, where the old semantic route loses the dogs;
2. `mp3d_camera_placement_11523`, the normal positive control; and
3. one verified truly-out-of-frame actor case as the zero-pixel negative
   control.

The A/B must show that RGB, depth, original SEMANTIC_ID, 75-frame pose readback,
zero physics steps and world time are unchanged, while OBJECT_ID equals each
run's recorded `actor.object_id`. Until then, OBJECT_ID is not accepted modal
authority. Even after that A/B, P1/amodal masks and derived occlusion/off-screen
truth remain separate requirements.

This slice adds no hash, baseline, frozen contract, camera gate, visibility
gate, or candidate filter.
