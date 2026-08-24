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

### Negative auxiliary diagnostic

A later owner-run auxiliary diagnostic makes the boundary stricter. The fresh
output
`/data/jzy/tmp/mp3d_object_id_aux_no_semantic_mesh_19be5f7_06598_r1`
used the existing c78 runtime, `load_semantic_mesh=false`, and semantic target
OBJECT_ID. Runtime actor IDs were `[1,36]`. RGB and depth matched the regular
capture numerically, but object-ID counts were still:

- frame 0: `[0,0]`;
- frame 37: `[372,0]`; and
- frame 74: `[1378,0]`.

Only the old actor semantic value 210 changed to runtime ID 1; actor1/runtime ID
36 remained zero throughout. The first verifier correctly failed and is
retained. This proves that selecting OBJECT_ID and disabling the semantic mesh
is not sufficient on c78. The active-full-scene native route in this slice is
a necessary new variable, not a guaranteed fix. If the fresh prefix A/B still
loses either actor, the next investigation must inspect the skinned graphics
subtree, drawable object-ID assignment, and PBR/skin shader output rather than
weakening the evidence boundary.

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

### Fresh prefix and exact pending A/B commands

The clean implementation commit is `732f264e33f25d9aba56de9889871dbeeb70445d`.
Its fresh installed research prefix is:

`/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z`

Post-commit evidence:

- Python adjacent suite: `135 passed` at
  `/data/jzy/tmp/habitat_object_id_732f264_python_r1.log`;
- Ruff/format/byte-compile/clean worktree:
  `/data/jzy/tmp/habitat_object_id_732f264_quality_r1.log`;
- clean fresh native build and CTest `1/1 passed`:
  `/data/jzy/tmp/habitat_object_id_native_test_732f264_r1.log`; and
- installed-prefix smoke and ELF checks are listed in
  `docs/provenance/RUNTIME_PREFIX_RECIPE.md`.

No command below has been run in this slice. Each output path must remain
absent before execution, and a first failure stops the sequence.

06598 failure case:

```bash
cd /data/jzy/code/AVEngine-codex-diversification-integration
env PYTHONPATH=src /data/jzy/tmp/avengine-bootstrap-env-20260821/bin/python -m avengine.cli m5 capture-current-visual \
  --animal-manifest /data/avengine_external/datasets/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json \
  --m2-request /data/avengine_external/review/current_mp3d_two_beagle_route_lateral_seed22_grounded/primary_m2_request.json \
  --room-manifest /data/jzy/code/AVEngine-codex-diversification-integration/examples/m1/rooms/habitat_mp3d_example/room_manifest.json \
  --m1-request /data/jzy/tmp/mp3d_probe_subset_37317f4_r1/requests/mp3d_camera_placement_06598.json \
  --runtime-prefix /data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z \
  --mp3d-root /data/avengine_external/datasets/mp3d_example_scene_1.1 \
  --magnum-python-site /data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages \
  --rlr-sdk-root /data/avengine_external/rlr-sdk/RLRAudioPropagationPkg-runtime-b-20260820T2010Z \
  --output /data/jzy/tmp/mp3d_object_id_732f264_06598_r1
```

11523 positive control uses the same arguments except:

```bash
  --m1-request /data/jzy/tmp/mp3d_probe_subset_37317f4_r1/requests/mp3d_camera_placement_11523.json \
  --output /data/jzy/tmp/mp3d_object_id_732f264_11523_r1
```

The strongest currently materialized out-of-frustum candidate is placement
00677: both actual emitter centers remain behind the camera for all 75 frames
(forward-depth maxima `-3.2478 m` and `-4.1643 m`). It is a negative-control
candidate, not yet a visually certified whole-actor off-screen case:

```bash
  --m1-request /data/jzy/tmp/mp3d_region_viewpoint_banks_81eadfd_r1/requests/17DRP5sb8fy_region_000/mp3d_camera_placement_00677.json \
  --output /data/jzy/tmp/mp3d_object_id_732f264_00677_out_of_frustum_r1
```

The abbreviated 11523/00677 blocks replace only those two arguments in the
full 06598 command. Before calling 00677 a true off-screen control, inspect its
native RGB/depth/object-ID frames and actor geometry; center-frustum evidence
alone is not whole-body visibility authority.
