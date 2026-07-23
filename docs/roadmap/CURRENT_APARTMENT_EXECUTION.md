# Current Apartment execution

Last updated: 2026-07-23

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

## Exact next actions

1. Review only the owned source and documentation changes.
2. Run the focused unit/style checks.
3. Commit and push Habitat-native work to
   `feature/habitat-native-avengine`; keep generated cat UAssets and import
   tooling in the SPEAR feature branch.
4. Treat the present Abyssinian as a replaceable canary. A later
   owner-selected cat must bring its own generated mesh, emitter measurement,
   UE binding, floor correction and forward/animation gates; the existing
   `source1`/`source2`, route, acoustic and index machinery remains unchanged.

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
