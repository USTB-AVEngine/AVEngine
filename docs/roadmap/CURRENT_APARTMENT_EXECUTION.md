# Current Apartment execution

Last updated: 2026-07-26

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

- 20260726, branch `feature/instance-attr-generalization` (this repo) plus
  branch `feature/asset-pipeline-hardening` (legacy SPEAR tools repo),
  pending owner review; no frozen artifact was regenerated:
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
4. Review and, if accepted, merge the
   `feature/instance-attr-generalization` provenance split that reconciles
   the Abyssinian `slim` generation request with the `standard` runtime
   baseline (both values are now explicit registry fields instead of one
   contested label). A later owner-selected cat must still bring its own
   generated Mesh and runtime profile.
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
