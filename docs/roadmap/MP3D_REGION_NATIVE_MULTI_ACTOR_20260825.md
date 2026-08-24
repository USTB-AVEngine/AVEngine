# MP3D Region Native Multi-Actor Checkpoint — 2026-08-25

## Scope and status

This slice starts one installed-prefix Habitat Simulator for one retained
research-only region case, instantiates two independently validated articulated
actor bundles, applies and reads back both 75-frame root/joint tracks, renders
co-located RGB/depth/semantic/OBJECT_ID sensors, reads both emitter links, and
classifies raw Bullet contact points without stepping physics. It remains
`research_only=true`, `formal_m2_request=false`, and `episode_counted=false`.
It does not run RLR or establish formal modal visibility/off-screen authority.

Implementation commits on `codex/mp3d-region-native-visual-runner` are:

- `19abc5a feat(m5): capture native MP3D region actor pairs`;
- `cf941b4 feat(m5): rebind MP3D cameras after route planning`.

The second commit adds an explicit post-hoc M1 camera binding. The route family
and both actor matrices are loaded from the existing r4 actor-track artifact
and are not regenerated from, filtered by, or shifted toward the selected
camera. The receipt retains both the declared camera and the effective capture
camera and records `camera_participation_in_route_generation=false`.

## Parent review fixes before native execution

The initial uncommitted draft was not accepted as written. Parent review found
and fixed the following real failures before the first native run:

- the code called the imported `quaternion` module as though it were a
  constructor; it now calls `quaternion.quaternion`;
- only the agent pose was checked; all four sensor readbacks are now checked
  against the M1 rig on every frame;
- semantic preflight used the legacy global actor IDs rather than the two IDs
  in the current case;
- inactive historical actor-pair contact records were treated as current
  collisions; they remain serialized, but only active pair contacts reject the
  pair;
- the receipt incorrectly described OBJECT_ID as formal same-renderer modal
  authority. It now says that active-RGB depth equivalence and formal modal
  visibility remain pending;
- GPU selection is explicit and the execution wrapper refuses to start when
  the selected GPU has any compute PID. It never auto-selects another GPU.

The owned/adjacent target suite passed `41` tests. Ruff `0.15.21` check and
format check passed for all touched files.

## Native run 1: retained all-off-screen research case

The first fresh run used the materializer's declared selection-order binding:

`/data/jzy/tmp/mp3d_region_multi_actor_native_19abc5a_region000_f01_both_r1`

It executed all 75 frames on GPU2 with no physics step and no active actor-pair
collision. Both actor OBJECT_ID and semantic counts were zero in all frames,
and RGB/depth were byte-identical across the sampled timeline. This is not a
runner failure: parent inspection of the first-person sheet and Habitat
PathFinder top-down overlay showed that the camera faces a bedroom while both
routes lie behind/separated by room geometry.

Review artifacts are:

- `/data/jzy/tmp/mp3d_region_multi_actor_native_19abc5a_region000_f01_both_r1_review/first_person_frames_0_18_37_56_74.png`;
- `/data/jzy/tmp/mp3d_region_multi_actor_native_19abc5a_region000_f01_both_r1_review/topdown_camera_and_actor_routes.png`.

The case is retained as a raw all-zero visibility descriptor. It is not called
formal off-screen truth because the installed semantic-target render path is
not yet active-RGB same-depth authority.

## Native run 2: visible post-hoc camera binding

The visible mini-pilot retained region000 route family 03 and rebound it after
planning to the already validated region000 M1 request
`mp3d_camera_placement_04975.json`:

- first capture:
  `/data/jzy/tmp/mp3d_region_multi_actor_native_cf941b4_region000_f03_cam04975_both_r1`;
- repeat capture:
  `/data/jzy/tmp/mp3d_region_multi_actor_native_cf941b4_region000_f03_cam04975_both_r2`;
- A/B receipt:
  `/data/jzy/tmp/mp3d_region_multi_actor_native_cf941b4_region000_f03_cam04975_ab_verification_r1.json`;
- independent verifier:
  `/data/jzy/tmp/mp3d_region_multi_actor_native_cf941b4_region000_f03_cam04975_verification_r1.json`.

Measured native results are:

- 75 frames at 15 fps, `pts_ticks=i*3200`, world time `0.0`, physics steps `0`;
- runtime object IDs `1` and `36`, distinct from logical actor/source IDs;
- actor0 OBJECT_ID positive in 75/75 frames, pixel range 817–1245;
- actor1 OBJECT_ID positive in 17/75 frames, pixel range 0–1845;
- no active actor-pair collision frames;
- route translation maximum absolute error `0.0`;
- applied skin-root readback maximum absolute error
  `3.520412688029495e-07`;
- camera, all sensor, and listener maximum absolute error
  `1.1920929132713809e-08`;
- every saved array has its expected shape and finite values;
- all arrays and `frame_records.json` are identical across the two fresh runs.

Parent inspection of frames 0/18/37/56/74 shows a well-exposed MP3D interior,
both Beagles at the bottom of frame 0, and the retained moving actor crossing
the foreground over later frames. The top-down overlay shows the selected
camera and both unchanged route tracks in the same region. The review files
are under:

`/data/jzy/tmp/mp3d_region_multi_actor_native_cf941b4_region000_f03_cam04975_both_r1_review`

## Semantic/OBJECT_ID boundary

The visible run also reproduces the known independent-semantic-mesh problem.
Actor0 semantic ID 210 matches its OBJECT_ID pixel range in this view, but
actor1 semantic ID 211 reaches only 904 pixels while OBJECT_ID reaches 1845.
Therefore neither semantic counts nor the current OBJECT_ID target are promoted
to formal modal visibility, occlusion, or off-screen truth. Zero pixels remain
a descriptor and never a camera/source legality gate.

## Regression

The complete CPU suite on `cf941b4` is retained at:

`/data/jzy/tmp/mp3d_region_native_cf941b4_full_unit_r2.log`

It reports:

`3408 passed, 118 skipped, 52 subtests passed in 429.90s`.

The first full-suite attempt is also retained at
`/data/jzy/tmp/mp3d_region_native_cf941b4_full_unit_r1.log`. It stopped during
collection because the bootstrap environment lacked `cv2` and an installed
top-level `tests` package shadowed the repository's namespace. The passing run
kept the same full test scope, explicitly prepended the repository test path,
and appended the server's existing conda site only to supply missing `cv2`; no
tests were deleted or filtered.

## Still pending

- native active-RGB same-depth OBJECT_ID authority and formal off-screen truth;
- support-contact classification beyond raw stage contacts;
- the remaining region/camera/route mini-pilots across all ten MP3D regions;
- new RLR audio bound to this new visual receipt;
- formal admission and any production Episode count.
