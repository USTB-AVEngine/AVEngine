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
- A deterministic audio-only adapter has now executed 26 test mixtures through
  the existing `v4_3_new_IPD_Enhancer` Progressive Refinement `last.pt`.
  It reads no visual media, crops each five-second native-HRTF binaural sample
  to the model's fixed four-second input, and queries both source slots. Strict
  checkpoint loading and all 52 forward contracts passed. After one cold
  batch, warm inference averaged 0.0500 seconds per target query (20.00
  queries/second). This is compatibility evidence, not a performance pass:
  v4 folds AVEngine's full circle into 180 degrees, the folded mean absolute
  error was 53.01 degrees, and only 4.87 percent of frames changed predicted
  bins between the two target queries. Current delivered mixtures do not
  retain wet stems, so separation SI-SNR was correctly left `not_run`.
- `v72_S2L` is a structurally simple two-channel candidate for retraining but
  has no local checkpoint. `v77_4ch_S2L` consumes four physical tetrahedral
  microphone channels rather than native-HRTF binaural audio and is not the
  first compatibility baseline for this delivery.
- The owner selected the v4_3 architecture for the first retraining path, but
  explicitly rejected reuse of its localization checkpoint. The experiment is
  isolated under `models/v4_3_binaural360_selective/` on branch
  `feature/v43-binaural360-model`; no v4-specific model code remains in
  production `src/avengine/`, general `tools/` or general unit-test paths. The
  required Progressive Refinement architecture is copied into that directory;
  the runner does not import implementation code from the Spatial checkout.
- The isolated runner consumed one complete five-second, 16 kHz native-HRTF
  binaural mixture twice with distinct `dog barking` and `cat meowing` text
  queries. It used all 80,000 samples, 75 output frames and 360 circular
  one-degree bins. The localization/separation network was randomly
  initialized; no old localization checkpoint was accepted or loaded. Frozen
  pretrained LAION CLAP was retained only as the text encoder.
- The real CUDA smoke passed forward, circular DoA/cardinality loss, backward
  and parameter update. One two-query update took 0.40 seconds after model
  load, changed the 360-degree head by `0.00100044`, propagated gradients to
  6,972,122 parameters and peaked at 2.15 GB allocated GPU memory. This is an
  execution-contract pass only: one update is not convergence evidence and
  its localization error must not be reported as model quality. The
  authoritative self-contained smoke record is
  `tmp/m7/v43_model_subdir_train_smoke_20260723_03/results.json`.
- The formal model path now follows the useful part of the old v4 data
  contract without copying its per-sample file overhead. A resumable builder
  converted all 1,000 five-second WAV mixtures, two native-360 label tracks,
  split codes and captions into one 612 MB HDF5 file. Full readback yielded
  1,600/200/200 train/validation/test text queries. The first conversion took
  347.76 seconds under shared-disk contention; formal training preloads it to
  host RAM and avoids reopening WAV files across epochs.
- Mid-epoch checkpoint/resume has executed on real CUDA. Deterministic
  CUDA/cuDNN settings made a one-step-plus-resume run bit-identical to an
  uninterrupted two-step run across all 351 saved model tensors. Checkpoints
  exclude the frozen CLAP weights and are approximately 81 MB.
- One complete 800/100/100 epoch passed 200 training batches, validation,
  best-checkpoint reload and held-out test. Training took 72.56 seconds after
  model/data initialization. This remains an execution pass rather than a
  quality pass: after one epoch the test circular mean error was 69.77 degrees
  and the two text queries almost always predicted the same direction. Longer
  training and a deliberate selective-objective review are still required.
  The result is
  `tmp/m7/v43_formal_h5_epoch1_20260723_01/training_summary.json`.
- The requested 100-epoch from-scratch text-selective run is complete at
  `tmp/m7/v43_binaural360_training_100ep_b56_20260724_01/training_summary.json`.
  It used batch size 56 on GPU3, peaked at 40.85 GB allocated / 48.33 GB
  reserved, completed 2,900 optimizer steps and selected epoch 89 by
  validation circular MAE (`6.83 degrees`). Held-out test circular mean,
  median and p90 errors are `10.10`, `0.93` and `16.71 degrees`;
  cardinality accuracy is `1.0`, and `98.72%` of frames change their predicted
  bin between the two text queries. This is encouraging selective-audio
  evidence for this fixed 1,000-item Apartment split, not a generalization or
  dataset-qualification claim.
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

## Exact next actions

1. Review the completed 100-epoch v4_3 result and decide the next model
   experiment: repeated seeds, audio-library expansion or a held-out
   room/asset split. Do not infer broad generalization from the current single
   Apartment split.
2. Review and commit the isolated experiment on
   `feature/v43-binaural360-model`; do not merge it into
   `feature/habitat-native-avengine` until the training interface is reviewed.
3. For a later owner-selected cat, add its independently accepted generated
   Mesh and one runtime profile containing emitter measurement, UE binding,
   floor correction and forward/animation gates. Select it by asset ID; the
   existing `source1`/`source2`, route, acoustic and index machinery remains
   unchanged.

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
