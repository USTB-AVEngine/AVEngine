# Current Apartment execution

Last updated: 2026-07-25

This is the short operational checkpoint for the active Apartment training
dataset work. Durable project rules live in the repository `AGENTS.md`; this
file prevents a later session from reconstructing current state from chat
history or choosing an easier but incorrect substitute.

## Target deliverable

Build a reusable `apartment_0000` training-data path with generic `source1`
and `source2` slots, two simultaneously active sound sources, five-second
binaural clips, SPEAR/UE RGB, Topdown visualization and aligned labels. Cover
static/static, moving/static, static/moving and moving/moving trajectories,
then measure batch throughput for a 1,000-example train/validation/test
closure. Scene data must be shared rather than copied once per example.

## Current baseline asset identities

- Border Collie:
  `generated_border_collie_black_white_medium_standard_adult_research_v1`.
  It uses the accepted FLUX -> Pixel3D -> repair -> TokenRig result, not a
  recoloured or reshaped library dog.
- Abyssinian:
  `generated_abyssinian_ruddy_medium_standard_adult_research_v1`.
  Its accepted animated GLB is
  `/data/jzy/code/AVEngine/external/SPEAR/tmp/new_animal_assets/animal_generated_mesh_rig_v2_20260722_01/flux_base_abyssinian/tokenrig_seed42/retarget_v5_spike_yaw180_matched_amp0p40/animated_walk_idle.glb`.
- The Quaternius Cat is diagnostic/motion-reference material only. It must not
  appear as the Abyssinian geometry in the current Apartment canary.
- These are current canary baselines, not a permanent closed asset list. A
  later owner-selected cat may replace or join the Abyssinian after it is
  independently generated and passes the same asset/runtime checks. Dataset
  assembly must select assets through `source1`/`source2` bindings and must not
  hard-code either current breed.
- A different breed is a new generated asset. `size`, `body_build`,
  breed-valid coat and `life_stage` are instance attributes only after the
  correct breed-specific base mesh exists.

## Engine ownership

- Habitat-native: navigation/trajectory truth, source centers, RIR/binaural
  audio, Timeline, Topdown and labels.
- SPEAR/UE: final Apartment RGB.
- Do not silently replace final UE RGB with Habitat RGB.
- Actor slots remain `source1` and `source2`; human/dog/cat are bindings, not
  role names.

## Current checkpoint

- The Border Collie generated mesh and animation have been visually accepted.
- The generated Abyssinian Pixel3D/TokenRig animation has been visually
  accepted and imported into the standalone SPEAR/UE runtime with its own
  skeletal mesh, Idle/Walking actions, material and textures.
- The earlier UE Apartment cat canary used a Quaternius template and is
  historical diagnostic evidence only; it cannot close the current target.
- The current generated cat uses the measured asset-local emitter offset
  `[0.388693463649, 0.166419619913, 0.0]` metres. Its standalone UE visual
  binding applies a cat-specific `+42.25 cm` component translation derived
  from UE bounds, not a reusable setting for future cats.
- The generated cat and Border Collie passed real UE root, animation-phase,
  anatomical-forward and floor-contact readback in all four motion cases.
- The current 100-route acoustic plan contains 2,346 unique RIR jobs for 5,000
  source positions. Native propagation took 71.74 seconds and the complete
  cache run took 152.52 seconds.
- The 100-route x 10-audio batch contains 1,000 five-second, 16 kHz,
  two-channel binaural WAV files. It made no RLR or visual calls, took 142.26
  seconds and passed all artifact readbacks.
- The eight-worker visual input build took 268.96 seconds and copied no room
  geometry. The persistent SPEAR/UE runtime rendered all 100 routes in
  1,337.51 seconds (22.29 minutes), averaging 13.17 seconds per route. All 100
  route gates and all 400 media readbacks passed.
- The training index splits by visual episode: 80 routes and their ten audio
  variants form 800 training samples; 10 routes each form 100 validation and
  100 test samples. Ordered asset pairs and all four motion cases are present
  in every split, and no visual route crosses split boundaries.
- The owner-requested expansion reuses 400 single-source path components to
  form 4,000 unique ordered two-source combinations. Path A may legally appear
  in both A+B and A+C; the exact two-source episode remains the split unit.
- Concrete asset binding selected 1,000 visual episodes from 9,617 passing
  candidates: 250 per ordered Border-Collie/cat/human pairing. The four motion
  cases are 252/252/248/248 and the minimum concrete source-center separation
  is 0.309 m.
- The expanded plan has 9,047 unique RIR jobs for 50,000 source/keyframe uses.
  Native propagation took 243.16 seconds and the complete cache run took
  346.43 seconds. The 1,000 x 1 audio batch took 388.24 seconds with zero new
  RLR or visual calls and passed all 1,000 WAV readbacks.
- The expanded 1,000-episode UE input bundle uses a 640x480 Topdown-only
  intermediate, copies no room geometry, retains 334 MB and took 1,156.78
  seconds with eight workers. Episode builds and the subsequent UE renderer
  are independently resumable.
- The expanded UE closure and 1,000 x 1 training index are complete. All
  1,000 unique episodes, 4,000 retained UE media records and 6,000 indexed
  RGB/Topdown/audio/label references passed readback; the split is
  800/100/100 at visual-episode level.
- Source assets and rooms now have separate runtime profile registries.
  Apartment Timeline construction, concrete-emitter selection and SPEAR/UE
  plan compilation resolve assets by exact ID/revision instead of Python breed
  constants. The current Border Collie, Abyssinian, Beagle and human values
  moved to `examples/runtime/source_asset_runtime_profiles.json`; the native
  Apartment map/room reference moved to
  `examples/runtime/room_runtime_profiles.json`. Pair templates can name only
  `source1`/`source2` assets and inherit their measured emitter, forward,
  animation, floor and UE bindings.
- The camera/listener request path has passed two real Habitat-native probes at
  distinct Apartment positions and yaw angles. Both requests retained one
  co-located listener and rendered the same-view RGB/depth/semantic sensors
  after live NavMesh floor snapping. The retained receipts are
  `tmp/runtime_interface_probe_20260724_01/habitat_native_camera_a_retry1/receipt.json`
  and
  `tmp/runtime_interface_probe_20260724_01/habitat_native_camera_b/receipt.json`;
  the views are visibly different rather than duplicated camera output.
- A new independently generated yellow Labrador cross-check now exercises the
  same source registry without adding breed constants to Python. Its own
  FLUX -> Pixel3D -> TokenRig Mesh, Skeleton, Idle/Walking actions, textures
  and Blueprint were imported and cooked into the standalone UE package.
  Concrete-emitter route selection retained 913 of 1,000 candidates; one
  human+Labrador moving/moving episode then passed native RIR/binaural
  assembly and a real 75-frame SPEAR Apartment render. Runtime readback
  measured at most 7.28 degrees anatomical-forward error and 2.35 cm floor
  error for the Labrador, with exact camera/root/animation-phase gates and
  matching two-channel audio packets. The review video and evidence are under
  `tmp/runtime_interface_probe_20260724_01/labrador_ue_native_retry1/`.
- The Labrador review exposed two presentation defects rather than a route or
  audio regression. Apartment furniture is baked into the stage/NavMesh and
  therefore produced no independent rigid OBBs; Topdown v3 now distinguishes
  four NavMesh-internal center-point exclusion components from the
  border-connected room exterior and draws those baked blockers in orange.
  The old UE recorder also stepped the world without reading SceneCapture
  pixels, so its first retained frames triggered the floor's streamed texture
  pages. The recorder now discards real frame-zero SceneCapture readbacks
  until the view is stable. A native 75-frame, 15 fps, five-second rerun
  discarded 40 warmup frames, reached a final mean absolute frame change of
  0.369 against a 0.8 threshold, retained the detailed wood floor from formal
  frame zero and passed all four RGB/Topdown/binaural media readbacks. Evidence
  is under
  `tmp/runtime_interface_probe_20260724_01/labrador_ue_native_topdown_v3_warmcapture_20260725_01/`.
- That Labrador output is diagnostic evidence for the generic Topdown and
  SceneCapture warmup fixes only. It does not supersede or close a requested
  two-human Apartment review. The corrected review reuses the exact original
  blue-shirt male plus Female Adult 02 spec and both authored trajectories,
  contains no dog actor, renders UE/SPEAR RGB after 40 discarded real
  SceneCapture readbacks, and pairs those pixels with the current Topdown v3.
  Both humanoid direction and floor-contact gates pass; the original
  five-second, 16 kHz, two-channel technical-review placeholder audio remains
  bound without replacement. The retained result is
  `/data/jzy/code/AVEngine/external/SPEAR/tmp/human_color_and_new_woman_20260725_01/clips/blue_shirt_male_and_female02_dual_walk_warmcapture_topdown_v3_02/`.
- The first background conversion mistakenly resumed one full 1,000-episode
  plan and tried to stop it at a directory count. It produced redundant work;
  the verified merger removed the overlap from the final 1,000-episode index.
  The runner now partitions the manifest before UE starts through exact
  `--shard-count` / `--shard-index` arguments. Future shards must use those
  fixed, disjoint plans and may resume only their own plan.
- The common M3 layer now includes a SoundSpaces-style semantic material
  resolver with exact override, UE/ReplicaCAD name/material-slot hint,
  semantic-category and plausible-default precedence. MP3D semantic
  PLY/`.house` compilation, unknown-category coverage and interior-ray mesh
  diagnostics are end-to-end. Apartment/Kujiale can feed their Actor and
  material-slot identities through this generic resolver, but their exact
  scene inventories still require review before any generated coefficients
  are called physically calibrated.
- The new generic mesh-leakage command checked four reviewed Apartment
  camera/source/NavMesh points with 64 spherical directions each. All 256 rays
  hit the existing 782,306-face acoustic mesh, while topology still reports
  20,734 boundary edges; this is sampled enclosure evidence, not proof of a
  globally closed mesh. The CPU reference took 1,127.52 seconds, so batch room
  qualification needs a reusable BVH/RLR ray accelerator. ReplicaCAD escaped
  through 50/192 rays, all upward.
- The actual composed `kujiale_0020` full-home USD is now compiled into a
  separate M3 Acoustic Scene Package rather than an Apartment or shoebox
  proxy. Its corrected living-room/kitchen/bathroom/bedroom probe set observed
  0/64 escaped rays and all four points passed the 5 cm clearance diagnostic,
  but topology still fails with 141,038 boundary edges and 1,492 nonmanifold
  edges. Its material assignments remain uncalibrated research candidates,
  and the old Kujiale videos still contain their labelled shoebox-preview
  audio. This work did not modify the Apartment package or its retained RIR
  cache; Apartment remains the current usable baseline.

## Exact next actions

1. Use the completed 1,000-episode Apartment closure for the first real
   train/validation/test model run; freeze the exact model-facing index and
   metrics without regenerating the already verified RGB/Topdown/audio bank.
2. Generate a bounded native RLR RIR/cache canary from the new real-USD
   Kujiale Acoustic Scene Package before replacing any old shoebox-preview
   audio. Keep geometry topology failure and uncalibrated material state
   visible in that result.
3. Keep the isolated v4_3 experiment on its dedicated feature branch until its
   training interface and next generalization experiment are reviewed.
4. Reconcile the Abyssinian generation request's `slim` body-build label with
   the current runtime baseline's `standard` research label before promoting
   that exact attribute provenance. A later owner-selected cat must still
   bring its own generated Mesh and runtime profile.

## Current execution constraints

- Use FLUX for this route; do not switch to Qwen.
- Do not use low-VRAM, CPU-offload or sequential-offload modes.
- Do not substitute a template mesh merely because it imports or renders more
  easily.
- A successful UE editor import is not sufficient for the standalone runtime.
  Batch compatible asset imports, cook their common content parent once and
  verify the cooked Blueprint/Mesh/actions before rendering; do not recook
  once per episode or instance.
- Generated videos and large temporary evidence stay outside Git.
