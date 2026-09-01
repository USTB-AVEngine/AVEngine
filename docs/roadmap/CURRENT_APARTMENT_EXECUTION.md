# Current Apartment execution

Last updated: 2026-09-01

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

- 2026-08-21 single-source research checkpoint: commit `76d130e` adds the
  explicit current Apartment visual author/capture path and commit `e0a1f87`
  lets the optional host extension use the selected Python 3.11+ minor ABI.
  A fresh non-Git project assembled from `native/spear/unreal` and authorized
  external Apartment/Human/Beagle/SpContent inputs completed UE 5.5
  BuildCookRun. Its packaged `SpearSim.sh` then produced one native 75-frame,
  15 fps, 1280x720 RGB research capture on GPU0. Camera and both actor-root
  readbacks have zero position/yaw error; Human/Beagle animation readback
  errors are at most `4.8e-7` seconds. The receipt is `research_only`,
  `episode_counted=false` and `qualification_claim=false`; it requests no
  audio, RLR, M6/M7 bundle or formal admission. The server review entry is
  `/data/avengine_external/review/apartment_current_visual_capture_cp312_retry1_b9150cb_20260821T0100Z/review/index.html`.
  Two earlier fresh attempts remain preserved as failures (missing cp312 host
  extension, then the nested UE pose-readback shape); neither counted a frame
  or Episode. Human visual review of the successful output remains pending.

- 2026-08-16 source-migration boundary: the direct packaged Apartment canary
  now requires explicit --spear-executable and no longer derives SpearSim.sh
  from an external SPEAR checkout. UE, the packaged runtime, and authorized
  assets remain external inputs; this CLI cutover alone is not a fresh full75
  equivalence result.
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
  edges. Its material assignments remain `research_placeholder`,
  and the old Kujiale videos still contain their labelled shoebox-preview
  audio. This work did not modify the Apartment package or its retained RIR
  cache; Apartment remains the current usable baseline.
- The public engine-side Kujiale balanced-360 closure is complete without
  regenerating the Apartment 1,000-example bank. The retained plan
  `tmp/m7/kujiale_0020_zero_shot_balanced360_plan_100_20260726_01` contains
  100 generic two-source episodes, 25 per motion case, all six ordered sound
  pairs within every motion case, and front/right/rear/left frame fractions of
  25.173%/25.487%/22.613%/26.727%. The observed minimum listener/source XZ
  distance is 0.353553 m.
- The matching real-USD research cache
  `tmp/m7/kujiale_0020_real_usd_rir_cache_balanced360_2587_20260726_02`
  is bound to cleanup package
  `tmp/m3/kujiale_0020_usd_semantic_rlr_cleanup_20260726_03`, whose package ID
  ends in `rlr_incompatible_filter_v2`, package-content SHA-256 is
  `d041a8c511a957f4fd52a9e8c646332ea1158d24ab5e671fcd276c5903f44190`
  and manifest SHA-256 is
  `e9d23bc1247f22ffb4d8959e116c751a44aa4fe52c2f3f870c0fd407ad646cc0`.
  Cache request identity
  `7cc5b8c015ca4b3043660802585f9a2628c7a1c6e85956e7dbf5f1ca0e478330`
  passed 2,587/2,587 native RIRs for all 5,000 source/keyframe uses in 41
  retained shards (312,098,114 bytes) using the CPU-only RLR backend with 32
  configured threads.
- The cache-only output
  `tmp/m7/kujiale_0020_zero_shot_balanced360_binaural_100_20260726_02`
  contains 100 five-second, 16 kHz, two-channel mixtures plus 200 nonzero
  persisted stems. Full readback proved every mixture is the exact float32
  `source1 + source2` stem sum; the maximum mixture peak is
  `0.05897637456655502` and the minimum stem peak is
  `3.6796163840335794e-06`. Assembly took 98.9181 seconds with no native RLR
  or visual render calls. Its passing output closure binds all 100 samples,
  300 WAVE files, 300 sidecars and the producer/runtime identity.
- This closure remains research-only: the Kujiale topology failure is still
  open, materials are not physically calibrated, and none of these artifacts
  establish real-room acoustic truth, dataset admission or real-world
  generalization.

### Checkpoint 20260726: acoustic material fidelity closure (branch cc-acoustic-material-fidelity)

- The residential semantic material ruleset is now
  `soundspaces_style_residential_v2`
  (`examples/m3/semantic_materials/residential_material_rules.json`, SHA-256
  `bf37a39d69aa8888c95cd0cc6ab963b4735f11a4d820079204575ba03212f953`): 20
  materials (adds indoor foliage and stacked paper), 70 categories (adds the
  29 Kujiale residential categories that previously fell to defaults), 19
  ordered name hints for visual material-slot names, and room-scoped explicit
  overrides for the legacy apartment structural slots. Structural tokens
  (wall/floor/ceiling) deliberately stay out of the global name hints so
  reviewed semantic categories keep precedence-correct resolution.
- A third semantic adapter, `compile_visual_slot_semantic_research_scene`
  (CLI `avengine m3 compile-visual-slots-semantic`), resolves UE visual
  material-slot names through the same resolver, leakage probes and
  research-candidate contract as the MP3D and USD routes.
- The legacy apartment now has a differentiated-material research package
  `legacy_ue_apartment_0000_visual_slot_semantic_seed917_research_v1`
  (`tmp/m3/legacy_ue_apartment_visual_slot_semantic_20260726_01`,
  package-content SHA-256
  `e7f0264388d2d139a69098118ebb4f773d449b14a92f93f635a58c169178535a`,
  manifest SHA-256
  `0585e11b23431e4ddd232a71a21b30518913ae4b33e372736d6747c3a6b6f49e`):
  48 visual slots resolve as 44 name hints + 3 explicit overrides + 1 honest
  default (`MI_Props`), replacing the previous uniform 0.2/0.05 neutral
  placeholder slots. The four reviewed interior probes observe 0/256 escaped
  rays with probe clearance `pass`.
- The Kujiale full-home USD snapshot recompiled under rules v2
  (`kujiale_0020_full_home_v1_usd_semantic_seed917_research_v1`,
  `tmp/m3/kujiale_0020_usd_semantic_rules_v2_20260726_01`, package-content
  SHA-256
  `49a9571ef84ed21ea945accf0be761d5f1b8c209ef1a6f1a5d4a593c96cf8560`):
  resolution moves from 93 semantic / 28 name-hint / 193 default to
  150 semantic / 164 name-hint / 0 default with zero unknown categories.
- A room-agnostic native simulation profile
  (`examples/runtime/rir_cache_simulation_request_v2.json`, SHA-256
  `f3c74d9bfa67fb3cb757589f6760c21ac909575f2b4f7d9768d3277c1e8b0a22`)
  supersedes reuse of the historical M4 canary request for production caches:
  indirect ray depth 100 -> 200 (RLR default), direct SH order 1 -> 3
  (adapter default), transmission off -> on (matches the SoundSpaces 2.0
  training configuration). Rendering deliberately stays at 16 kHz as the
  declared dataset bandwidth bound.
- Both apartment caches were re-rendered on the frozen job plans with the new
  package and profile; no trajectory, plan or visual artifact was
  regenerated. The generic 9,198-job cache
  (`tmp/m7/apartment_rir_cache_semantic_v2_t32_b64_full_20260726_01`, request
  identity
  `70371734bf2c4e44b60575b5d5d08900afb9b714ad085495d8187e47b9e8d0e9`)
  passed 9,198/9,198 in 325.81 propagation seconds across 144 shards
  (2,273,498,429 bytes). The training-bank 9,047-job cache
  (`tmp/m7/apartment_generated_assets_rir_cache_unique1000_semantic_v2_20260726_01`,
  request identity
  `0ba0bcc7bb4e88d5607267c8d258da00b22136a7e5f71ad1a6339f328d4917c3`)
  passed 9,047/9,047 in 307.49 propagation seconds across 142 shards.
- A matched-job diagnostic (`tools/m7/compare_rir_cache_metrics.py`, 256
  seeded pairs, zero skips) decomposes the acoustic change: materials alone
  move mean EDT 0.378 s -> 0.600 s, DRR -1.96 dB -> -2.82 dB and late energy
  x2.7; the propagation-profile upgrade adds only +0.012 s EDT and
  -0.26 dB DRR on top. The uniform placeholder was therefore the dominant
  realism gap, and the re-rendered room sits in a plausible furnished
  residential EDT range instead of a uniformly damped one.
- The 1,000-episode audio bank was re-assembled on the new training cache
  with the exact retained dry-audio declarations and `variants_per_episode=1`
  (`tmp/m7/apartment_generated_assets_1000_unique_visual_binaural_semantic_v2_20260726_01`);
  the artifact-level verifier
  (`tmp/m7/apartment_semantic_v2_batch_verification_20260726_01.json`)
  reports `pass` for all 1,000 samples. The frozen visual bank, trajectory
  bank and dataset index are untouched; this audio realization is a parallel
  alternative to the placeholder-material bank, and switching any model
  training to it is a separate owner decision.
- Claim boundary: every coefficient remains `research_placeholder`
  (representative octave-band priors, jittered, uncalibrated). This closure
  improves internal material differentiation and propagation-parameter parity
  with the SoundSpaces 2.0 reference configuration; it does not establish
  physical room-material truth, does not run the M3.1 EDT calibration, and
  does not change dataset admission for any room or asset.

### Checkpoint 20260726b: Habitat-native room route (MP3D second-room enablement)

- MP3D semantic packages recompiled under rules v2 for both room-manifest
  identities (m1 example and m2 articulated review; the latter is
  `tmp/m3/mp3d_semantic_rules_v2_articulated_20260726_01`): 31 surfaces
  resolve as 16 name-hint / 14 semantic / 1 default (`unknown_object`), with
  no RLR-incompatible triangles (the cleanup deriver correctly refuses a
  no-op). The scan probes 0/32 escaped rays.
- Accepted design decision (owner choice, option a): because the MP3D and
  USD adapters place semantic category strings inside name fields, ordered
  name hints intercept some category-labeled surfaces under the fixed
  `explicit > name hint > semantic category` precedence. Hint candidate sets
  were aligned with the same-named categories, so 11 of the 14 changed MP3D
  decisions keep the identical material and the remainder move within the
  plausible candidate domain; the resolution label and its 0.75 confidence
  cap are accepted as-is and recorded here instead of altering the
  documented precedence contract.
- The room runtime profile registry (revision `20260726_v3`) gains its first
  `habitat_native` profile `habitat_mp3d_17DRP5sb8fy`, sharing the exact
  dataset render contract with `spear_apartment_0000`
  (1280x720, 75 frames, 15 Hz, HFOV 105) with zero warmup frames; the
  registry validator now rejects habitat_native profiles whose map path is a
  UE `/Game/` map instead of a room manifest.
- `tools/m7/run_habitat_room_batch.py` is the Habitat-native counterpart of
  the SPEAR batch runner: registry-selected habitat_native rooms, fixed
  disjoint `--shard-count/--shard-index` selection, resumable execution that
  only skips episodes whose retained gate evidence independently reads back
  as `pass`, and a hash-bound batch manifest that stays
  `research_candidate`. Its first real batch
  (`tmp/m7/habitat_mp3d_batch_review_20260726_01`) rendered the retained
  M5.1 mixed route through the native runtime with 14/14 gates passing and
  270 frames, and the resume path was exercised against that retained
  evidence. Episodes whose route length differs from the 75-frame profile
  contract are explicitly marked `review_only`.
- Owner-review media: three frozen apartment visual episodes were paired
  with old-versus-new audio (byte-identical video stream, only the audio
  realization differs) under
  `tmp/review/apartment_acoustic_ab_20260726/`, and the MP3D route is being
  delivered as the annotated binaural listening video on the rules-v2
  package with the v2 simulation profile. These are review artifacts, not
  dataset media.

### Checkpoint 20260726c: generated-asset admission contract

- The source work is `cc-instance-attr-generalization` in this repository
  plus `cc-asset-pipeline-hardening` in the legacy SPEAR tools repository.
  The native-side source has been integrated here without regenerating any
  frozen artifact.  On the isolated
  `cc-native-asset-integration-validation` branch, the combined targeted
  suite passed 144/144 and the complete suite passed 1,662 tests with one
  retained-evidence canary skipped; `feature/habitat-native-avengine` itself
  was not moved:
  - The three generated coat profiles (`cat_abyssinian_coat_v1`,
    `dog_border_collie_coat_v1`, `dog_labrador_retriever_coat_v1`) are now
    registered in the fail-closed appearance contract with beagle-pattern
    three-level domains, and `validate_source_asset_runtime_registry`
    cross-checks every runtime-registry coat against that contract, so an
    unregistered profile or out-of-domain value fails closed for any future
    asset. Registry revision `20260724_v2` -> `20260726_v3`; coat values
    moved to neutral-level names (`standard_ruddy`, `standard_black_white`,
    `standard_yellow`).
  - The Abyssinian `slim`/`standard` collision is resolved by an explicit
    provenance split: optional `generation_request_attributes` records the
    sampled breed morphology of the generation request (Abyssinian `slim`,
    verified from the retained instance request), while
    `realized_attributes.body_build` remains the neutral instance-variation
    baseline. Full fast-unit suite passed (1,610 passed, 1 skipped).
  - SPEAR-side generation tooling gained a single-point forward declaration
    contract (self-hashed, donor-constant motion basis; per-asset
    motion-basis yaw / side-chain flips are now contract errors), a
    deterministic PCA + head-end-vote forward estimator, a fail-closed
    stance-drift gait-direction audit, a `--preview-only` cheap triage mode
    after retarget, and a historical human-decision calibration collector
    that any future visual pre-screener must be scored against before it may
    triage review media.

### Checkpoint 20260729a: M6 AudioProgram integrated into M7

- Branch `feature/m7-m6-audio-program-integration` adds one opt-in M7 route
  from a validated M6 AudioProgram through the existing M5.1 dry-bus
  assembler, explicit endpoint-to-RIR-slot binding, cached dynamic binaural
  rendering, Timeline/source-manifest projection and the existing Apartment
  UE runner. No sit, chair binding or new action system is included.
- The current visual contract is deliberately one AudioProgram instance per
  episode. It fails closed on multiple program variants and reconstructs
  counterfactual B from its canonical base. The renderer forms the persisted
  float32 mixture through the existing exact source1 + source2 sum path;
  retaining source stems is optional. Human speech and animal vocalization
  drive mouth activity; other sound classes remain ordinary audio events.
- The focused M6/M7 suite passes 72 tests. The fresh Shiba + human S4
  artifact-level canary and verifier are under
  `tmp/m7/m6_audio_program_integration_canary_20260729_03`; its rebuilt UE
  input and dry-run plan are under
  `tmp/m7/m6_audio_program_integration_bundle_20260729_03` and
  `tmp/m7/m6_audio_program_integration_ue_dryrun_20260729_03`.
  The unchanged runtime path already passed a real native UE readback under
  `tmp/m7/m6_audio_program_integration_ue_20260729_01`.
- Legacy asset-audio delivery remains a separate compatible path; the
  one-episode regression at
  `tmp/m7/m7_legacy_audio_regression_20260729_01` is byte-identical to its
  prior delivery manifests and mixture WAV. All new evidence remains
  research-only and does not authorize formal dataset registration.

## Exact next actions

1. Freeze and reuse the verified Apartment and Kujiale engine-side
   plan/RIR/cache/mixture artifacts; do not regenerate the Apartment 1,000
   RGB/Topdown/audio bank for downstream experiments.
2. Treat Kujiale topology repair and physical material calibration as separate
   future work. Until both pass, preserve the `research_placeholder` and
   research-only claim boundary in every derived artifact.
3. Keep the owner-private v4.3 test experiment and its `locate` environment
   outside the public AVEngine runtime. Its dedicated private feature branch is
   permanently independent and must not be merged into this branch.
4. Preserve the merged provenance invariant: the Abyssinian `slim`
   generation request and `standard` runtime baseline remain explicit,
   separate registry fields. A later owner-selected cat must still bring its
   own generated Mesh and runtime profile.
5. Derive each asset's UE component vertical correction from its retained
   support-plane leveling evidence instead of the current per-asset manual
   measurements (Abyssinian +42.25 cm, Labrador +30.4947 cm, Border Collie
   0). Until then, every new asset must record how its UE Z delta was
   measured; a silently copied delta from another asset is a defect.

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

### Checkpoint 20260730: dynamic SensorRig M7/UE end-to-end closure

- M7 now persists the complete `SensorRigTrajectory` as a canonical sidecar
  in the plan, bundle and per-episode metadata. Timeline v2 binds every frame
  to the corresponding `view0` pose hash, while the source manifest, batch
  binding and delivery records retain the same trajectory id, content hash
  and first/last pose hashes.
- Habitat visual capture and the native SPEAR/UE runner now apply the
  frame-matched camera/listener pose before capture and retain readback
  evidence. The real UE canary checked 75/75 camera pose hashes, all 75 were
  unique, maximum position error was 0 cm and the camera root-readback gate
  passed.
- The RIR acoustic identity and cache key now include source position plus
  Listener position and orientation. The native renderer calls
  `set_listener_pose` when the Listener pose changes, and retained cache
  evidence records that pose per job. Topdown, distance and DOA use the same
  frame's Listener pose; the audio batch copies the exact trajectory binding,
  and the batch verifier cross-checks Timeline, RIR uses and delivery
  metadata instead of accepting a partially wired dynamic run.
- The retained real closure is under
  `tmp/m7/dynamic_sensor_rig_end_to_end_canary_20260730_01_plan`,
  `tmp/m7/dynamic_sensor_rig_end_to_end_canary_20260730_01_binaural`,
  `tmp/m7/dynamic_sensor_rig_end_to_end_canary_20260730_01_bundle` and
  `tmp/m7/dynamic_sensor_rig_end_to_end_canary_20260730_01_ue`. The successful
  native RLR cache is
  `tmp/m7/dynamic_sensor_rig_end_to_end_canary_20260730_02_rir_cache`
  (50/50 jobs, `status=pass`); the earlier
  `tmp/m7/dynamic_sensor_rig_end_to_end_canary_20260730_01_rir_cache`
  contains `FAILED.json` and is not passing evidence.
- Owner-review media are
  `tmp/m7/dynamic_sensor_rig_end_to_end_canary_20260730_01_ue/corgi_british__recombined_both_moving_0036/ue_clean_binaural.mp4`
  and
  `tmp/m7/dynamic_sensor_rig_end_to_end_canary_20260730_01_ue/corgi_british__recombined_both_moving_0036/ue_topdown_binaural.mp4`;
  both have 75 frames and passed their media/audio gates.
- This remains a research canary, not dataset admission. SoundSpaces 2.0 and
  AVEngine use the same Habitat/RLR acoustic-propagation family, while these
  apartment material coefficients have not been calibrated against measured
  real-room RIRs. The retained pass establishes cross-modal wiring and native
  execution, not that AVEngine is acoustically more realistic than
  SoundSpaces or physical reality.

### Checkpoint 20260730b: lossless SoundSpaces 2 public acoustic control

- Branch `feature/soundspaces2-frl-acoustic-profile-v1` adds an RLR-native M3
  material database v2 and the existing `m3 import-rlr-materials` route. Each
  absorption, scattering, transmission and damping curve keeps its own
  frequency grid; no interpolation or common-band coercion is performed.
  The official 30-material public SoundSpaces 2/MP3D file round-trips
  canonically with 120 curves and 1,488 frequency/value pairs preserved under
  `tmp/m3/soundspaces2_public_material_import_20260730_01`.
- The vendor JSON remains byte/canonically unchanged. Its repeated `floor`
  label is removed only from the derived native-upload database because the
  pinned RLR binding requires case-insensitive label uniqueness; the material
  category evidence records that exact normalization. Coefficient curves are
  checked separately and remain value-identical to the imported source.
- The offline public-reference verifier SHA-binds all 42 WAV/metric files from
  the pinned public archive, then binds seven measured plus new/old simulated
  RIR sets and reproduces the bundled 1000 Hz summary: new DRR MAE
  `0.981985714286` dB, old DRR MAE
  `10.953578428571` dB and new RT60 mean relative error
  `12.443636462634%`. The retained report is
  `tmp/m3/soundspaces2_real_rir_reference_verification_20260730_01.json`.
- A real native-RLR room-control canary uses the explicit public 44.1 kHz
  SoundSpaces/RLR profile, identical geometry/source/listener state and one
  floor-only official-material change (`Carpet` -> `Carpet, Heavy`). Both
  caches passed under
  `tmp/m3/soundspaces2_public_room_control_20260730_01/native_rir_cache_v3`
  and `.../native_rir_cache_heavy_carpet`; the hash-bound comparison is
  `.../room_control_report.json`. The candidate changed the native coefficient
  and RIR hashes, increased the observed broadband DRR by `0.124062` dB and
  reduced late-energy ratio by `0.000376102` in this one realization.
- Claim boundary: this proves lossless public-parameter ingestion, native
  upload/readback and per-surface control. It does not reproduce the paper's
  FRL room: the public release omits the measurement-fitted coefficients,
  seven world-coordinate pairs and exact raw Replica scan identifier, and the
  local ReplicaCAD asset is an artist recreation. All generated packages and
  reports therefore keep `qualification_claim=false`.

### Checkpoint 20260730c: scene-origin acoustic profile routing

- Branch `feature/soundspaces2-frl-acoustic-profile-v1` now routes three
  explicit scene origins: `soundspaces2_public`, `habitat_scene` and
  `spear_ue_authored`. Selection is fail-closed on the exact room id, room
  revision, source lineage and acoustic-profile id. The selected profile,
  physical package manifest and simulation request are compiled into the
  common AVEngine Acoustic Scene Package contract; all three routes then use
  the existing `rlr_audio_propagation` solver. `habitat_scene` is intentionally
  not called semantic: the current ReplicaCAD adapter reads visual GLB
  material slots and makes no semantic-annotation claim.
- The real MP3D `17DRP5sb8fy` SoundSpaces package `_03` contains 1,570,132
  vertices, 3,016,249 triangles and 31 categories. Public SoundSpaces
  substring matches cover 89.738% of triangles and the remaining 10.262% use
  the official `Default`; selected public coefficient curves are preserved.
  The ReplicaCAD `apt_0` raw package exposed five zero-area triangles and was
  correctly rejected by RLR; its retained cleanup removes exactly those five
  triangles and leaves 37,278. Its ten opaque visual slots currently resolve
  through the declared default candidate and therefore do not claim physical
  material truth. The SPEAR Apartment package contains 463,873 vertices and
  782,306 triangles; 48 slots resolve as 3 explicit, 44 name-hint and 1
  default assignment. Its retained topology remains research-only.
- Registry-selected native single-RIR canaries pass for all three origins:
  MP3D under
  `tmp/m3/mp3d_17DRP5sb8fy_soundspaces2_video_source_single_rir_cache_registry_20260730_02`,
  cleaned ReplicaCAD under
  `tmp/m3/replicacad_apt_0_habitat_scene_profiled_rlr_cleanup_single_rir_cache_registry_20260730_01`,
  and SPEAR/UE under
  `tmp/m3/legacy_ue_apartment_spear_single_rir_cache_registry_20260730_01`.
  Their RIR cache request identities bind the exact room/profile selection,
  package-manifest SHA and simulation-request SHA before native rendering.
- That binding now remains closed through `acoustic_selection.json`, cache
  request/receipt/index, resumed sessions, per-episode RIR evidence, M7 audio,
  UE bundle/runtime evidence and dataset index rows. SPEAR runtime additionally
  checks that visual room, selected runtime room and acoustic room are equal
  before UE launch; the runtime map id is retained separately instead of being
  presented as a room identity. Historical explicit/unbound caches remain
  readable but are marked `not_verified` and cannot fabricate registry
  provenance.
- The first formal dynamic MP3D path reuses the retained 75-frame
  `SensorRigTrajectory` and two static source positions from the room request.
  Its plan has 150 exact source/Listener states under
  `tmp/m7/mp3d_17DRP5sb8fy_soundspaces2_dynamic_room_plan_20260730_01`;
  all 150 native RIRs pass under
  `tmp/m7/mp3d_17DRP5sb8fy_soundspaces2_dynamic_rir_cache_20260730_01`.
  The cache-backed dog-bark plus human-speech realization is a 5-second,
  16 kHz float32 binaural mixture with two retained active stems under
  `tmp/m7/mp3d_17DRP5sb8fy_soundspaces2_dynamic_binaural_20260730_01`.
  The synchronized Habitat RGB, QA Topdown, same-frame geometric DOA/distance
  and binaural listening review is
  `tmp/m7/mp3d_17DRP5sb8fy_soundspaces2_formal_review_20260730_01/mp3d_soundspaces2_room_evaluation_binaural_doa_review.mp4`;
  its hash-bound evidence is the adjacent `evidence.json`.
- The complete unit suite passes 1,906 tests with one existing retained-
  evidence readback skipped unless `AVENGINE_RUN_LOCAL_M6_CANARY_TEST=1`.
- This checkpoint remains research-only: it is not the unpublished fitted FRL
  Apartment, a measured ReplicaCAD/SPEAR calibration, room admission or formal
  dataset admission.

### Checkpoint 20260821: single-repo refactor sealing and owner reviews (branch cc-qa-overlay-rgb)

Owner reviews passed on 2026-08-21:

- Apartment UE 75-frame production visual with real skeletal animation
  (`2786897`) passed owner review.
- The MP3D M7 PBR/IBL fix was re-verified on a fresh full 270-frame installed
  batch (`m7_habitat_mp3d_batch_installed_pbr_2786897_20260821T044926Z`,
  14/14 gates). A per-frame pixel audit over all 270 frames measured human
  98.22% / beagle 99.70% non-black versus 0% in the black-actor regression
  batch, with byte-identical semantic masks, so the fix changed shading only.
  Owner approved.
- The Kujiale ceiling-anchored lighting candidate was approved as-is. The
  bright white panel visible in frame is dataset-native emissive material
  (it is the only bright object in the zero-added-light control); AVEngine
  adds exactly one invisible fill light anchored to the kitchen ceiling prim
  (`kitchen_ceiling_0000_anchored_fill`, 1800 lm, 4000 K).

Sealing and alignment commits landed after the 2786897 pause point:

- `4fe8a4a` refactor(m7): require explicit runtime root in direct visual writer
- `1c1c22e` fix(qa): align strict two-human finalizer with hash-free bindings
- `b968755` fix(qa): make skokloster semantic preflight test hermetic
- `3653c43` refactor: retire implicit sibling-checkout discovery
- `3354b69` refactor(m1): reject Git-checkout runtime roots

Runtime roots now come only from explicit arguments or
`AVENGINE_HABITAT_RUNTIME_ROOT`, and any Git-checkout root fails closed.
Historical M2/M6x writers stay runnable on external non-checkout data roots;
readers and shared loaders are unchanged.

Environment repair: the `sofar` 1.2.3 wheel had installed a stray top-level
`tests` package into the `avengine-habitat-runtime` site-packages, shadowing
the repository `tests/` namespace and breaking collection of
`test_m6_release_builder.py`. The stray package was moved to
`~/env-backups/sofar-1.2.3-stray-tests-20260821`; `sofar` still imports.

Unit layer after these changes: 3072 passed, 0 failed, 0 collection errors
(previously 3050 passed, 20 failed, 1 uncollectable file). The branch is
backed up to `origin/cc-qa-overlay-rgb` (owner-authorized push; the stale LFS
stub pre-push hook was bypassed with `--no-verify`; the repository has no
LFS-tracked content).

Next actions (single-repo refactor track, owner-approved plan 2026-08-21):

1. M5.1 ReplicaCAD migration: copy
   `/data/datasets/versioned_data/replica_cad_dataset_1.5` and
   `replica_cad_baked_lighting_1.5` into `/data/avengine_external/datasets/`,
   switch the replicacad/m6x consumers to the explicit external root, then
   run a fresh comparison.
2. QuestionSpec: resolve the official compile blocker (stale byte-size lock
   in four binding manifests; see
   `docs/qa/QUESTION_PROTOCOL_RECOMPILE_BLOCKER_20260817.md`) through
   registered identity/revision, and report the QS-007 expected-vs-evaluator
   divergence root cause for an owner decision.
3. Whole-repository residual-dependency and provenance/license audit,
   recording RLR CC BY-NC 4.0 research use (owner confirmed non-commercial
   research and a future open-source release).
4. Fresh-clone single-repository bootstrap verification.
5. Four-route pre/post equivalence finals plus the owner review/listening
   package.
6. Owner-authorized only: PR merge to `main`, then archive the legacy SPEAR
   and habitat-sim-AVEngine repositories.

The formal dataset denominator remains 0.

### Checkpoint 20260821b: ReplicaCAD installed runtime and QuestionSpec compile unblocked

- M5.1 ReplicaCAD now runs on the installed runtime (cc51d58, 34b8e4b):
  ReplicaCAD 1.5 data lives at /data/avengine_external/datasets/, the
  installed mixed-capture MP3D assumptions are room-conditional, and a fresh
  installed apt_0 run passed 270 frames / 19 gates with per-frame pixel audit
  (human 98.83%, beagle 96.99% non-black). Owner reviewed the contact sheet.
- The QuestionSpec official compile blocker is resolved (4ffacbb, allowed
  repair 2): four binding manifests reissued against the current registry
  with an append-only diff proof; delivery
  tmp/lead_a_question_protocol_paper_ready_v3 passes compile and
  paper-ready validation with 2230 candidate cases across 6 episodes.
- QS-007 divergence root cause: the fresh comparison paired the pixel
  binding expected table (pass) with an evaluator run that lacked native
  pixel-truth inputs, which correctly rejects offscreen_to_onscreen. The
  evaluator and both retained expected tables are consistent; the
  comparison harness input closure was incomplete. Owner decision pending
  on repair route (feed native pixel inputs vs CPU-only table).

### Checkpoint 20260821c: QuestionSpec fresh comparison closed (repair A)

tools/qa/compare_question_spec_fresh.py re-evaluates every retained
bind-time QuestionSpec over the verified native closure (facts with pixel
truth) and compares status and answer against the retained records:
511 specs across the 6 catalog episodes, 509 exact matches. The two
divergences are both QS-007 (offscreen_to_onscreen) rows bound on
2026-08-09 shortly before cf0a840 closed the cross-modal contracts: the
current evaluator rejects the question as not applicable when the target
never makes a pixel-observed out-of-view to visible transition, while the
pre-cf0a840 binder answered "no". cf0a840 is an ancestor of both the
refactor reference d19e0e8 and the CPU equivalence reference 1a26e5c, so
pre- and post-refactor evaluators share identical semantics and the
divergence is not a refactor effect. Report:
tmp/lead_a_questionspec_fresh_compare_v1/comparison.json.

### Checkpoint 20260821d: repository audit and single-repo bootstrap verification

- Whole-repository residual audit: live src/tools code carries no sibling
  checkout or private absolute-path dependencies; the remaining mentions are
  provenance comments, fail-closed guards, and legacy-labeled example
  manifests. THIRD_PARTY_NOTICES.md gained data-input rows (MIT KEMAR HRTF
  and its 16 kHz derivative, SoundSpaces 2 material config, Brown Photostudio
  IBL, Matterport3D example) plus the 2026-08-21 owner decision that AVEngine
  is non-commercial research heading for an open-source release (5703009).
- Single-repository bootstrap verified: a fresh clone at 5703009 plus a
  dedicated Conda env ran scripts/setup.sh --profile fast_unit end to end
  (editable install, path and schema validation, fast-unit suite). The only
  failures were eleven strict two-human suites reading retained evidence
  under the untracked tmp workspace; they now skip with a reason when the
  workspace is absent (d5fdfd0). Fresh clone: 2933 passed, 0 failed.
  Working copy afterwards: 3073 passed, 0 failed.
- Verification artifacts: /data/jzy/tmp/avengine-bootstrap-verify-20260821
  (clone) and /data/jzy/tmp/avengine-bootstrap-env-20260821 (env).

### Checkpoint 20260821e: fresh S-series, MP3D contrast route, session handoff

- The Blender-custom review room is permanently excluded (AGENTS.md,
  f5c96e3); all Blender-room review material was withdrawn.
- The fixed apartment canary now runs end to end on the installed runtime
  (c22c839, df0f9ff): fresh bundle
  tmp/m6x/fixed_apartment_canary_fresh_20260821T095910Z, status pass, eight
  binaural S0-S5 scenario videos regenerated by the current engine. Owner
  listened and accepted the Apartment S-series audio informally.
- MP3D contrast work: --camera-selection lateral_sweep on the route author
  (d7b9100) raises the actors azimuth sweep to about 19 degrees; the author
  and current-visual capture take an explicit external RLR SDK
  (ed516c4, 1bad7a3); the author sank world-contact actors about 0.33 m by
  subtracting the skin-to-actor lift from the floor path - fixed so actor
  roots sit on the support plane (a958dd0), verified grounded at Y=0.0724
  with a fresh 480p capture and a full fresh M4 FOA -> binaural -> M5 mix
  rerun (review dirs current_mp3d_lateral_seed22g_*).
- Known MP3D audio gaps, in priority order for the next session:
  1. M4-current pair IRs are static per source, so mixes carry no motion;
     wire the M6x semantic per-state RIR machinery to authored MP3D routes.
  2. Both probe sources share one dry asset on identical schedules; wire
     M6 AudioProgram turn-taking into the current chain.
  Together these complete the MP3D S-series and real spatial audibility.
- Other open engineering: Apartment production-form bundle (UE RGB plus
  bound audio, the M6/M7 formal-bundle audio closure); fresh-environment
  native-layer bootstrap validation (fast-unit already proven).
- Owner-pending: push (about seven commits after 0aa4ac8); formal review of
  ReplicaCAD video, cardinal left/right, and the grounded MP3D clip;
  decisions on QS-007 not-applicable semantics and on deleting the retired
  AVENGINE_HABITAT_RUNTIME_ROOT interface entirely; Phase 6 authorizations
  (merge to main, archive legacy repositories).

## Checkpoint 20260821f: closure-first plan adopted (owner-approved), residual-dependency audit

Owner directive (2026-08-21): complete the single-repository integrity closure first (no runtime dependency on any other Git checkout), reach the pushable state, then engine capability closure (MP3D audio, Apartment production bundle, fresh native validation). The Studio web planner is deferred until after all of these. An eight-agent read-only audit over repo HEAD 95ef22b, both conda environments, and /data/avengine_external produced the gap list below.

Blockers (break the moment legacy checkouts disappear):
- avengine-habitat-runtime: habitat-sim 0.3.3 is a legacy scikit-build-core editable mapping all habitat_sim sources to /data/jzy/code/habitat-sim-AVEngine/src_python; magnum and corrade are served entirely from that checkout's build/cp312 install/platlib (they are not pip packages in the env); the redirecting finder has rebuild=True, which is why SKBUILD_EDITABLE_SKIP existed.
- spear-env: spear-sim and spear-ext editables point into /data/jzy/code/AVEngine/external/SPEAR. The current chain does not use them: the vendored client is avengine.backends.spear_ue and the compiled extension is the external prebuilt avengine_spear_ext under /data/avengine_external/spear-host-sdk/ (cp312). The "import spear" strings in python_service.py execute inside the UE editor's embedded interpreter, not locally.
- tools/qa strict-two-human preflight/probe/materialize scripts hard-code the legacy Habitat checkout, sound-spaces, SPEAR-lead-b, and the old multi-repo SPEAR root as runtime inputs.
- tools/m6x/run_apartment_four_motion_pilot.py --beagle-audio and --human-runtime-glb defaults resolve to REPOSITORY.parent/AVEngine/external/SPEAR/...; the 20260821T095910Z fresh canary read the beagle dry wav from /data/datasets/avengine_workspaces/AVEngine/external/SPEAR/... (content sha256-identical to /data/avengine_external/m5-canary-inputs/blender_custom_m5_v1_20260820T223232Z/dry/dog_beagle.wav, so the fix is re-pointing, not re-sourcing).

Should-fix: no runtime-prefix rebuild recipe exists in-repo (cmake flags live only in the out-of-repo build CMakeCache; Corrade/Magnum/MagnumPlugins revisions only in external provenance; Bullet/RapidJSON/tinyxml2 versions unrecorded); the pbr-ibl and rlr-adapter-ec209a6 prefixes lack PROVENANCE.json (rlr-adapter-bfeacb8-r1 carries the template); envs/spear-env.yml still instructs editable installs from the legacy SPEAR checkout; the release attestation README and manifest reference the sibling Habitat checkout for commit provenance; m6y DEFAULT_SPEAR_ROOT falls back to the old multi-repo root; test_strict_two_human_construction_runtime_profile.py reads SPEAR-lead-b without a skip guard; two unit tests freeze legacy paths as environment contracts; canary default inputs depend on untracked tmp/ packages (m1 room export, m2 beagle package, m3 acoustic package, exterior glbs); six entry points default --hrtf to /usr/share/libmysofa (OS package) instead of the versioned external SOFA; examples/m3 skokloster plan JSONs hard-code legacy paths; AGENTS.md still carries the transition-workspaces and QuestionSpec-blocked paragraphs, both superseded.

Positive findings: src/ core is closed (explicit roots, Git-checkout rejection, editable meta-finder stripping inside prepare_installed_habitat_runtime); the pbr-ibl prefix was built from this repository's own commit c78db29 (an ancestor of HEAD) with zero CMakeCache references to the legacy fork; the magnum-python prefix comes from official upstream archives with recorded revisions; prefix ELFs carry no RPATH/RUNPATH and need only system libraries plus libRLRAudioPropagation via AVENGINE_RLR_SDK_ROOT.

New structural discovery (absent from all earlier audits): this working copy is a git worktree whose common .git lives at /data/jzy/code/AVEngine/.git (parent checkout on archive/main-pre-habitat-native-20260729). That is same-repository infrastructure, not a foreign-repo dependency, but Phase 6 archival of the /data/jzy/code/AVEngine directory must first make lead-a standalone (fresh clone or git worktree repair), and closure claims must state this linkage explicitly.

Approved execution order: C1 environment reinstall closure; C2 hard-coded path removal (per-script fix-or-archive decision); C3 data consolidation into /data/avengine_external; C4 rebuild-recipe and prefix provenance docs; C5 documentation alignment; C6 unplugged verification (fresh clone plus fresh env, zero legacy-path access); owner push and reviews; then MP3D audio closure (M6x per-state RIR plus M6 AudioProgram turn-taking into the current chain), the Apartment production bundle, and fresh native-layer validation. The formal dataset denominator stays 0.

## Checkpoint 20260821g: closure block C0-C6 complete; pushable state reached

All six closure stages from Checkpoint 20260821f are done. HEAD at this
checkpoint carries the closure commits fbaa708, be82db5, ef408e4, 5879963,
4795c8d, 6cc6a9f (plus this checkpoint).

- C1 environment closure: the legacy scikit-build-core editable habitat-sim
  0.3.3 was uninstalled from avengine-habitat-runtime (backup:
  ~/env-backups/habitat-sim-editable-20260821), and the legacy editable
  spear-sim/spear-ext were uninstalled from spear-env (backup:
  ~/env-backups/spear-editables-20260821). Post-surgery: plain-env
  habitat_sim import correctly absent, installed prefix serves
  habitat_sim/magnum alone (AudioSensorSpec present), fast-unit 3073/0.
  SKBUILD_EDITABLE_SKIP is no longer needed anywhere.
- C2 tool closure (ef408e4, 4795c8d): seven retained-evidence strict-two-human
  QA scripts carry a frozen HISTORICAL TOOL header (their recorded
  transition-era paths stay as historical contract); active m6y probes and
  canaries and the m6x four-motion pilot now require explicit --spear-root /
  --human-runtime-glb / --beagle-audio; the unguarded construction-profile
  test skips without workspace evidence; the replicacad dry-run test passes an
  explicit spear root.
- C3 data consolidation: the fixed-apartment canary stable inputs (beagle dry
  audio audit set, m1 room package+export, m2 beagle package+request, m3
  acoustic package, exterior proxy glb) are staged at
  /data/avengine_external/m6x-canary-inputs/fixed_apartment_inputs_v1_20260821T131842Z
  (276.9 MB, PROVENANCE.json, beagle wav sha256 verified against the sound
  registry digest).
- C4 rebuild provenance (5879963): docs/provenance/RUNTIME_PREFIX_RECIPE.md
  records the full prefix rebuild recipe (cmake flags, five dependency-prefix
  roles, all upstream revisions and archive SHA256s); THIRD_PARTY_NOTICES
  gained eight statically-linked dependency rows; the pbr-ibl and ec209a6
  prefixes received PROVENANCE.json; dependency archives are consolidated at
  /data/avengine_external/builds/dependency-archives-20260821. The v1 release
  attestation flow needs no change: its commands were already retired
  fail-closed, so its sibling-checkout references are historical record.
- C5 documentation alignment (6cc6a9f): AGENTS.md transition-workspaces and
  QuestionSpec-blocked paragraphs updated; stale sibling-fallback wording
  corrected; native/habitat README reflects the installed-prefix cutover;
  M1_EXECUTION old layout marked historical; skokloster m3 examples gained a
  LEGACY_NOTE.
- C6 unplugged verification, three layers, all pass:
  (A) import-origin assertions — at interpreter startup, after importing
  avengine + cli + the spear_ue backend, and after installed-runtime
  activation, no sys.path entry, loaded module, or meta-path finder touches
  /data/jzy/code/AVEngine/, habitat-sim-AVEngine, sound-spaces, or
  SPEAR-lead-b.
  (B) fresh clone at 6cc6a9f with its own dedicated env
  (/data/jzy/tmp/avengine-bootstrap-verify-20260821): 2933 passed, 0 failed,
  117 skipped (workspace-evidence suites skip by design).
  (C) full fixed-apartment canary rerun with the staged external inputs and
  explicit arguments only:
  tmp/m6x/fixed_apartment_canary_closure_verify_20260821T133144Z, status pass.
  The optional physical rename of the legacy checkout directories remains an
  owner-authorized final proof; it is not required by the three passing
  layers.

Owner-pending: push of the closure commits; the physical-rename decision;
Phase 6 remains gated on owner authorization, and lead-a must be made a
standalone clone before /data/jzy/code/AVEngine is archived (worktree .git
note in AGENTS.md).

Next engineering (in progress): MP3D audio closure. Mapped seams: the mixing
core render_dynamic_stems_and_mix natively supports [K,S,C,L]; the only K=1
squeeze is inside render_current_m1_research_audio (kept as the static
baseline; its validation stack intentionally rejects dynamic input). The
implementation route is a new current-chain entry that feeds the authored
route's 75-frame per-actor paths into the room-agnostic m5_1 trio
(build_strided_review_keyframes, render_research_review_binaural_rir_sequence,
render_research_review_binaural_audio) with dry buses from the M6
AudioProgram assembler (assemble_audio_program_dry_buses) under a turn-taking
program, then muxes the resulting 80000x2 binaural mixture onto the current
visual capture. The formal dataset denominator stays 0.

## Checkpoint 20260821h: MP3D motion-following audio landed (both recorded gaps closed)

Commit 78564dd adds `m5 render-current-mp3d-dynamic-audio`: the captured
per-frame source positions from current-visual frame records drive a strided
keyframe grid (25 states / 75 frames) through the persistent-context M5.1
binaural renderer, and the dry buses come from an M6 AudioProgram routing
variant (`current_mp3d_two_beagle_turn_taking_v1`, sequential_sources, six
verified bark slices from the registered dry asset). This closes both
Checkpoint 20260821e audio gaps at once — static-per-source IRs and the
shared identical schedule — and also fixes a third defect found during the
audit: the static chain's pair IRs had been rendered at the example request's
fixed probe points, not on the beagles' route at all. The static verb stays
the frozen baseline; validation stacks were not touched.

Fresh product (installed runtime, external inputs only):
/data/avengine_external/review/current_mp3d_dynamic_audio_seed22g_v1
(per-source dry buses, binaural stems, mixture, research_receipt.json, and
mp3d_dynamic_turn_taking_binaural.mp4 assembled in-engine by
tools/m5/build_current_mp3d_dynamic_review_clip.py on the profiled encoder
plus the frozen mux contract).

Quantified pre-review check (windowed ILD on the mixture): beagle_0 barks
+1.89 / -0.13 / +1.73 dB and beagle_1 barks -1.94 / -2.45 / -2.79 dB — the
two sources sit on opposite sides and their lateralization tracks the walk;
the retained static mix measures a constant +1.10 dB in every window.
Inter-event gaps carry only reverb tail. Unit layer: 3079 passed / 0 failed
(six new tests, program validated against the repository registries).

Research-only; the formal dataset denominator stays 0. Owner listening review
of the new clip is pending. Remaining engine work: the Apartment
production-form bundle (UE RGB plus bound audio) and fresh-environment
native-layer validation; the MP3D S-series expansion stays a post-merge
production task.

## Checkpoint 20260821i: Apartment UE production-form audio and fresh-native bootstrap closure

Apartment production-form bundle (82bb30c): the room-agnostic core
render_dynamic_research_audio is factored out of the MP3D verb, and the new
avengine.m7.apartment_dynamic_audio module binds the current Apartment UE
capture to it — the legacy glTF-import transform (U = 100 * (H.x, H.z, H.y))
is inverted on the captured per-frame UE anchor poses, the anchor-library
mouth/muzzle emitter heights apply per slot, and the capture camera must
match the fixed-apartment M1 listener authority (cross-checked at 1e-6).
tools/m7/render_current_apartment_dynamic_audio.py rendered the natural
parallel capture (research_only) with the validated human/beagle turn-taking
program (current_apartment_human_beagle_turn_taking_v1, sequential_sources):
/data/avengine_external/review/current_apartment_dynamic_audio_natural_parallel_v1
including apartment_ue_dynamic_turn_taking_binaural.mp4 (UE RGB plus bound
dynamic audio, assembled in-engine). Windowed ILD: human near-median
(+0.26 / +0.39 / -0.32 dB along the approach) and dog right-lateralized and
strengthening (-1.42 / -1.91 / -3.22 dB) — consistent with the two parallel
approach paths. The UE capture itself came from the apartment-visual-fix
workstream (1fd3f5d); which capture becomes the production episode remains
the owner's pick, and this chain re-renders audio for any capture with the
same record layout.

Fresh-native bootstrap closure (4edba1a, 9d75f9f): the fresh-clone native
execution check exposed that no native runtime dependency was declared
anywhere in pyproject — a fresh env could import avengine but not the
installed Habitat runtime. pyproject now declares the native extra
(numpy-quaternion, sofar, attrs, imageio, imageio-ffmpeg, scipy, numba,
tqdm, GitPython) and setup.sh --profile native_external installs
[test,native]. After the fix, the dedicated fresh clone plus fresh env
(/data/jzy/tmp/avengine-bootstrap-verify-20260821, only declared
dependencies) executed m5 render-current-mp3d-dynamic-audio natively end to
end: /data/jzy/tmp/avengine-native-verify-mp3d-audio-v3, status pass — and
its binaural mixture and stems are byte-identical (sha256) to the main-env
render, with matching trajectory hashes. Environment-robustness test fixes
landed as 9a76e5b.

The native_habitat/rlr_audio pytest markers select no tests by design; the
native layer is validated by real native executions such as the above, plus
the fixed-apartment closure canary. The formal dataset denominator stays 0.
Owner-pending: listening review of the MP3D dynamic clip and the Apartment
UE dynamic clip; push of the day's commits.

## Checkpoint 20260821j: apartment clip rebased; natural-series animation regression recorded

Two owner-reported defects in the first Apartment production-form clip, both
resolved the same evening:

1. Channel order (3a3cdd1): the UE apartment capture reads back BGR frames
   (read_rgb_bgr), and the review-clip tool encoded them as RGB, giving the
   clip an inverted cold tint. The tool now takes an explicit
   --channel-order {rgb,bgr}; the clip was regenerated with bgr and the
   pixels verified warm/natural.

2. Skeletal animation regression in the 1fd3f5d "natural" capture series:
   the owner observed sources gliding without walk animation (a recurrence
   of a previously seen failure). Pixel forensics confirm it — in
   apartment_current_visual_natural_parallel_1fd3f5d_20260821T1200Z the
   frame_records report advancing walk action_phase values and
   animation_readbacks with ~zero absolute_error_seconds, yet the rendered
   arms hang straight and legs stay parallel in every frame; the montage is
   scheduled but not evaluated in the render. By contrast,
   apartment_current_visual_capture_cp312_retry1_b9150cb_20260821T0100Z
   renders clearly distinct gait poses across frames. Two consequences are
   recorded for the apartment-visual workstream: (a) the natural series is
   not usable as review base video until its animation playback is fixed;
   (b) the animation readback validates the scheduled montage time, not the
   rendered pose, so it cannot catch this failure class — a rendered-pose
   check (e.g. skeletal joint readback or pixel-difference probe between
   walk phases) is the missing guard.

The production-form review clip was rebased onto the retry1 capture: audio
re-rendered in one command (the capture camera passed the same 1e-6 M1
listener cross-check), product at
/data/avengine_external/review/current_apartment_dynamic_audio_skeletal_retry1_v1
including apartment_ue_skeletal_dynamic_turn_taking_binaural.mp4. Windowed
ILD: human left +4.18 / +3.20 / +2.50 dB, dog right -4.76 / -2.72 / -3.21 dB
— clean opposite lateralization. Research-only; the formal dataset
denominator stays 0.

## Checkpoint 20260821k: apartment clip base corrected to the approved skeletal capture

The owner rejected the retry1-based clip too: exposure ramps dark-to-bright
over the first ~12 frames (mean luma 65 -> 89), the actors idle for the
first ~2.5 s then walk slowly, and the human walk animation is still not
convincing. Frame forensics against the already-approved
apartment_75f_skeletal.mp4 identified its true source:
apartment_current_visual_skeletal_animation_2786897_20260821T1400Z (flat
luma 88.7 from frame 0, walk the full episode, frame-identical stride and
arm-swing poses). The two captures I had picked (natural_parallel 1fd3f5d,
capture_cp312_retry1 b9150cb) were both defective bases; the other two
b9150cb capture attempts contain only a receipt or an operator_failure. The
selection mistake was not comparing against the approved reference first.

Audio was re-rendered on the approved-source capture (camera passes the
same 1e-6 M1 cross-check) and the clip regenerated:
/data/avengine_external/review/current_apartment_dynamic_audio_skeletal_2786897_v1
with apartment_ue_skeletal_animation_dynamic_binaural.mp4. Quantified
expectation for listening: this route walks nearly head-on at the camera
(human azimuth about +/-6 degrees, dog 0 to -17 degrees left), so window
ILDs are small (about +/-0.5 dB) by geometry and the room is
reverb-dominated (wet/dry propagation gain roughly flat across 4.6 m ->
1.8 m); the audible motion cues are the approach plus strict turn-taking,
not lateral sweep. A strongly lateral apartment production route would need
its own authored path; the MP3D clip demonstrates the lateral case. The
retry1 and natural-parallel audio products remain on disk as defect
evidence but are superseded for review.

## Checkpoint 20260822a: rendered-pose animation playback probe (room-agnostic guard, layer 1)

The recurring "scheduled walk, rendered slide" failure now has a pixel-level
guard: avengine.m7.animation_probe with tools/qa/probe_ue_capture_animation.py.
For frame pairs whose declared walk phases differ by a large cyclic distance,
the probe localizes each actor by frame-difference components (assigned to
slots by horizontal image order), restricts to pixels that are foreground
against the temporal-median background plate (drops the revealed-background
band behind fast movers), and measures the median gray residual after an
exhaustive translation-plus-scale fit. A rigid slide collapses under some
similarity transform; a played gait does not. Verdicts are banded
(sliding <= 7.5, animated >= 9.5, otherwise inconclusive; thresholds
calibrated on the three retained apartment captures) and the tool exits
nonzero on fail or inconclusive. It needs only a static camera,
frame_records.json, and arrays/rgb.npy - no semantic masks, no room
assumptions.

Validation: four synthetic unit tests (sliding flagged, animation accepted,
moving camera rejected, pair selection), plus all six real slots match the
pixel ground truth - skeletal_animation_2786897 both animated (17.8 / 25.3,
pass), natural_parallel_1fd3f5d both sliding (7.3 / 7.2, fail; the owner-
observed regression), capture_cp312_retry1 human sliding 1.7 with the dog
animated 9.8 (fail).

Layer 2 (the durable fix, recorded for the apartment-visual workstream): the
capture chain should read back rendered skeletal transforms per frame (for
example foot/hand socket world positions through the spear backend), making
"walk declared but bones static relative to the root" a direct numeric
check at capture time for every room, walk or idle, without any vision. The
existing animation readback validates only the scheduled montage time and
cannot catch this class.

The dynamic-audio chain itself is room-agnostic and unaffected: it consumes
recorded emitter positions and re-renders audio for any conforming capture
in one command (proven on MP3D/Habitat and Apartment/UE).

## Checkpoint 20260822b: merged to main (Phase 6 step 1 complete)

Owner-authorized and owner-executed merge landed: pull request #2 merged
cc-qa-overlay-rgb (3e56bc8, 250 commits over the old main) into main as
merge commit 3cf8d75. Remote main is now the single-repository one-stop
engine: closure C1-C6 with unplugged verification, motion-following dynamic
audio for MP3D and Apartment, the fresh-native bootstrap fixes, and the
rendered-pose animation probe. The origin push URL is pinned to SSH; main
is protected by a ruleset (pull request plus one approving review), which
the owner satisfied through the PR flow.

Remaining Phase 6 items, both owner-paced:
- Archiving the legacy GitHub repositories (habitat-sim-AVEngine and the old
  SPEAR fork) is a manual owner action on GitHub, deferred by the owner and
  safe to do at any time.
- Cleaning up the legacy /data/jzy/code/AVEngine directory on the server
  must wait until lead-a is converted from a git worktree to a standalone
  clone (its common .git still lives inside that directory); the owner will
  request the conversion before any local cleanup.

The formal dataset denominator stays 0; owner listening reviews of the
retained review package remain open on their own schedule.

## Checkpoint 20260822c: owner reviews passed; canonical path replaced; README updated

Owner formal reviews: all five outstanding items passed (the MP3D dynamic
turn-taking clip, the Apartment UE dynamic clip on the approved skeletal
capture, the ReplicaCAD 270f video, and the cardinal left/right binaural
baselines; the Apartment S-series had already been approved informally).
The review closure covers the pre/post-equivalence confirmation cited by
the README status section. Dataset admission is unchanged: the formal
denominator stays 0 pending the separate 35-item listening and rights flow.

Canonical-path replacement (owner-requested), executed in four reversible
steps with zero deletions:
1. lead-a converted from a git worktree to a standalone clone in place: a
   fresh clone's .git replaced the worktree pointer (backup of the old
   pointer at /data/jzy/tmp/leada-gitfile-backup-20260822), HEAD rebound to
   cc-qa-overlay-rgb, index rebuilt, working tree verified byte-clean at
   3eb88c8. Fresh hooks mean pushes no longer need --no-verify. The conda
   editable path is unchanged.
2. The two active parallel workstreams' unpushed commits were backed up to
   origin as wip/apartment-visual-fix-20260821 (1fd3f5d) and
   wip/kujiale-lighting-fix-20260821 (2bb985b).
3. The legacy multi-repo directory was renamed to
   /data/jzy/code/AVEngine-legacy-multirepo-20260822 (all objects and the
   remaining ~135 worktrees preserved) and `git worktree repair` re-linked
   the surviving worktrees; the two active candidates were spot-checked
   working.
4. /data/jzy/code/AVEngine is now a symlink to AVEngine-lead-a, so the
   canonical path is the merged single-repo engine.

README.md now states the completed single-repo baseline (merged via PR #2),
links the runtime-prefix rebuild recipe, and embeds the engine logical
pipeline diagram at docs/diagrams/engine_logical_pipeline.svg.

## Checkpoint 20260901: QA-v3 room-centric scene × profile scheduler

The QA-v3 research worktree now has a room-centric scheduler at
`tools/qa/run_qa_v3_room_profile_scheduler.py`. It attempts every requested
profile independently for every registered scene, preserves pair-level outputs
and rejection denominators, and does not stop the room when one profile fails.
Finite search failure is reported as `not_found_within_budget`, not as proof
that the room is inherently infeasible. Profile absence, pipeline failure,
pixel rejection and explicit exhaustive infeasibility remain separate states.

A CPU-only smoke used Apartment and the Kujiale living-room route domain with
the five current profiles plus an unimplemented card16 column. All 12 matrix
cells were recorded: 9 generated, 1 not found within budget and 2 not
implemented; quota state was 7 filled, 2 partial, 1 empty and 2 not run.
The retained matrix is under
`/data/jzy/tmp/qa_v3_room_profile_scheduler_smoke_20260901_final_v1`.
No UE/GPU render or admission claim was made. The next QA-v3 action is to feed
additional already-registered rooms through this scheduler, then pass generated
candidates to native pixel, audio/Gate-A and modality-certification stages.


The first all-profile expansion slice is now complete for card2 (immediate
emitting-time azimuth). It adds an instant-azimuth solver, answer-first side
bands, a query-frame caller AudioProgram and MCQ/Open Gate-A flips. A two-room
CPU smoke generated 2/2 candidates in Apartment and 2/2 in Kujiale; all four
Open gold separations exceed 60 degrees. Evidence is retained at
`/data/jzy/tmp/qa_v3_card2_two_room_smoke_20260901_v1`. This remains
geometry/timeline/program/fact evidence only. The executable catalog now has
six profiles; the next implementation slice is card3, followed by the remaining
dual-source controls before the N-source and multi-segment work.


Card3 (first-sound left/right) is also executable. Its AudioProgram fixes the
target's first event at frame 12, keeps at least three separated events, and
swaps only the first caller's slot for Gate A. The two-room smoke generated
2/2 candidates in each room; every MCQ and Open closed-set side changed under
Gate A. Evidence is retained at
`/data/jzy/tmp/qa_v3_card3_two_room_smoke_20260901_v1`. The executable
catalog now has seven profiles.


Card15b (total event count) is executable as a pure-audio control. It adds a
randomized exact-count AudioProgram and a reusable Gate-A gold relation:
swapping every source slot must preserve the count answer rather than flip it.
The two-room smoke generated 4/4 candidates per room, balanced counts 3 and 4
at 2:2, with all eight MCQ/Open golds preserved and every slot sequence
changed. Evidence is retained at
`/data/jzy/tmp/qa_v3_card15b_two_room_smoke_20260901_v1`. The executable
catalog now has eight profiles.


Card4R (which dog is closer at frame 30) is executable as a visual control.
The solver allocates the answer coat first, then finds a camera and two moving
routes with at least a 50 cm distance-order margin. The two-room smoke
generated 4/4 candidates per room with each coat at 2:2; all eight Gate-A
audio slot swaps preserved the visual gold. Evidence is retained at
`/data/jzy/tmp/qa_v3_card4r_two_room_smoke_20260901_v1`. The executable
catalog now has nine profiles.


Card5 and card5R now share one distance-change-pair solver that allocates the
target relation first and requires the distractor to exhibit the opposite
trend. Card5 binds the relation window to the first sound; card5R uses the
last-bark anchor and retains 2.03 seconds of tail silence. Card5 filled 4/4 in
both rooms. Card5R filled 4/4 in Apartment and produced 3/4 in the better of
two Kujiale seeds, with the remaining cell explicitly budget-exhausted rather
than unimplemented. Evidence is retained at
`/data/jzy/tmp/qa_v3_card5_two_room_smoke_20260901_v3` and
`/data/jzy/tmp/qa_v3_card5r_two_room_smoke_20260901_v2`. The executable
catalog now has eleven profiles.


Card6, card6R and card10 now share a motion-state-pair solver and an exact
solver-route timeline transform. The latter was required because the ordinary
visual author reparameterized waypoints by arc length and erased pause windows;
the final timeline now writes all 75 solver samples and synchronized
walk/idle actions. Each profile filled 4/4 in both rooms with moving/still at
2:2 and opposite Gate-A states. Card6 binds the second-sound window, card6R
uses frames 29..74 after the second sound, and card10 binds the first-sound
window. Evidence is retained under
`/data/jzy/tmp/qa_v3_card6_two_room_smoke_20260901_v2`,
`/data/jzy/tmp/qa_v3_card6r_two_room_smoke_20260901_v1` and
`/data/jzy/tmp/qa_v3_card10_two_room_smoke_20260901_v1`. The executable
catalog now has fourteen profiles.


The QA-v3 catalog now carries all 21 requested profiles. Cards 11, 12, 13, 14,
15a, 16 and 17 use the scene-neutral extended runner over the generalized
source1..sourceN timeline, pixel and dynamic-audio path. The final two-room
low-cost matrix attempted all 42 scene/profile cells: 34 generated geometry
candidates, six reported exact semantic-asset shortages, and two ended as
finite-budget search misses; no cell was unimplemented or a pipeline error.
The full-seed RNG fix now yields distinct cameras across extended profiles,
and both card17 segments must differ before generation can succeed.
Apartment runtime probes additionally closed four-actor RGB/pixel/binaural,
card15a main/Gate-A audio, card16 pixel-bound main/Gate-A truth, and both
card17 video segments. The first card11 pixel candidate was correctly rejected
because its fourth actor remained visible. Full evidence and claim boundaries
are in `docs/roadmap/QA_V3_ALL_PROFILE_ENGINE_REPORT_20260901.md`; the matrix
is retained at
`/data/jzy/tmp/qa_v3_all21_two_room_matrix_20260901_v4_reviewfix`.



QA-v3 room-centric pilot selected manifest now carries 216 research
candidates: 108 per room, six for each of the 18 currently runnable profiles.
Kujiale shortfalls were filled only by independent supplemental searches;
card1B required a deeper 30,000-attempt profile to obtain center-band examples.
The final selector balances card1B/card17 at 2:2:2 and materializes Gate B for
all 216 points (48 extended inline plus 168 dual-source twins). Apartment
runtime evidence closes pixel-bound card11/card15a/card16, main/Gate-A audio,
and two distinct card17 segments. The authoritative report is
`docs/roadmap/QA_V3_ROOM_CENTRIC_PILOT_REPORT_20260901.md`; selected and runtime
manifests are retained at
`/data/jzy/tmp/qa_v3_room_pilot_selected_2rooms_108each_20260901_v3` and
`/data/jzy/tmp/qa_v3_room_pilot_runtime_evidence_20260901_v1`.
