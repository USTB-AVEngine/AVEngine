# AVEngine Habitat-native contributor rules

## Authority and scope

This repository is the primary AVEngine implementation. Habitat-Sim is the
visual/runtime foundation and RLR Audio Propagation is the acoustic backend.
The sibling `habitat-sim-AVEngine` repository owns bounded runtime changes;
this repository owns contracts, registries, assets, room compilation,
timelines, audio assembly, evidence, QA and CLI behavior.

Read `README.md`, `docs/architecture/SYSTEM_OVERVIEW.md`,
`docs/architecture/REPOSITORY_BOUNDARIES.md` and the current milestone status
before changing architecture. Historical SPEAR/UE instructions are archived
under `docs/legacy/` and are never default authority.

For the active Apartment dataset work, also read
`docs/roadmap/CURRENT_APARTMENT_EXECUTION.md`. `AGENTS.md` contains durable
owner decisions; that file contains the current checkpoint, unfinished work
and the exact next action. Update the checkpoint whenever a meaningful stage
finishes or the owner changes direction.

## Current Apartment and generated-animal invariants

These rules are project-owner decisions for the active Habitat-native
Apartment training-data route. Do not replace them with an easier canary:

- A species template or motion donor is never the final instance mesh.
  Quaternius may donate animation and may be used for diagnostics, but a
  FLUX/Pixel3D animal must render its own Pixel3D geometry, topology/PBR repair,
  TokenRig skinning and reviewed animation. Never substitute the Quaternius
  Cat, Beagle or another breed's silhouette merely because its UE import is
  already available.
- The accepted generated assets for the current Apartment canary baseline are
  `generated_border_collie_black_white_medium_standard_adult_research_v1` and
  `generated_abyssinian_ruddy_medium_standard_adult_research_v1`. The latter
  must replace the historical
  `quaternius_domestic_cat_generic_diagnostic_v1` in this baseline before
  rendering. This is not a permanent restriction to an Abyssinian: `source1`
  and `source2` must accept any later owner-selected cat that has its own
  generated mesh and passes the same asset/runtime checks. Do not hard-code
  downstream dataset logic to either current breed.
- A materially different breed is a new source asset with its own generated
  mesh. `size`, `body_build`, `life_stage` and breed-valid coat variants are
  instance attributes only after that breed-specific base exists. Coat
  variants use FLUX reference-guided appearance editing; RGB multiplication is
  not a coat generator.
- For `apartment_0000`, Habitat-native owns the route, Timeline, source
  centers, binaural audio, Topdown and labels; SPEAR/UE owns final RGB pixels.
  Do not silently fall back to Habitat RGB for the final Apartment dataset.
  Actor slots remain generic `source1` and `source2`, regardless of whether an
  episode binds a human, dog or cat.
- Render each unique visual trajectory once and bind its dry-audio variants
  through the dataset index. Train/validation/test splitting happens at the
  visual-episode level, never at the audio-variant level, so the same RGB and
  Topdown trajectory cannot leak across splits. The current 1,000-item closure
  uses 100 visual episodes x 10 audio variants and an 800/100/100 sample split.
- The current owner-approved generative route uses FLUX without Qwen. Do not
  use low-VRAM modes, CPU offload or sequential model offload; load the model
  directly into available GPU memory. This does not relax output anatomy or
  reference-image review.

## Repository boundaries

- Do not copy Habitat-Sim source into this repository.
- Runtime C++/binding changes belong in the sibling runtime fork and must keep
  upstream behavior as the default unless a reviewed AVEngine opt-in is used.
- Generated media, native evidence and large assets belong under ignored
  output roots, not in Git. Track schemas, compact fixtures, requests, one
  authoritative bundle identity and human-readable status records; do not
  duplicate leaf hashes in prose or unrelated lock files.
- UE, SPEAR, gpuRIR and generative-asset tooling are legacy or optional
  backends. Default imports, tests, bootstrap and admission must not load them.

## Current contracts

- One logical camera/listener rig may expose co-located RGB, depth and semantic
  sensors. Those modalities are not multiple viewpoints.
- Timeline v2 integer ticks remain authoritative for synchronization.
- M5.1 source/event/flag v1 semantics are compatibility authority until a
  versioned replacement is explicitly approved. Preserve tri-state flags and
  their OR/AND clip aggregation; unknown is not false.
- Entity, animal-template, emitter/source, sound, AudioProgram and room IDs are
  stable versioned identifiers. Out-of-distribution animal requests fail with
  structured reasons; never silently fall back to a generic Dog.
- Animal appearance remains breed-scoped: size is small/medium/large,
  body_build is slim/standard/stocky, life_stage is explicit, and each breed
  owns three valid coat profiles rather than sharing incorrect color names.
- A new species, breed or materially different morphotype is a new source
  asset, never an instance-level recolour or reshape of an existing animal.
  Reuse the build procedure and a compatible motion family, but never reuse
  another breed's mesh, silhouette, joint locations or skin weights as shape
  authority. Quaternius may donate motion only after the new target-native mesh
  and rig exist. Stop after the breed-specific canonical 2D image for project-
  owner review before Pixel3D; four visible limbs do not excuse a wrong-breed
  silhouette. Follow
  `docs/assets/GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md`.
- Real visual geometry, acoustic proxy geometry, material assumptions,
  navigation and episode feasibility are separate facts. A single pass must
  never hide fail, blocked or not_run dimensions.

## Filesystem and evidence

The default trust mode is `trusted_research_workspace`, documented in
`docs/security/FILESYSTEM_TRUST_MODEL.md`. Inputs and outputs must resolve
inside declared roots; missing inputs, root escapes, hash mismatches and
replacement of immutable evidence are errors. Publish complete bundles with a
temporary sibling plus atomic no-replace commit where supported.

This mode does not claim protection from a malicious local symlink race,
portable `O_NOFOLLOW` directory semantics or general TOCTOU attacks. Do not
describe it as an untrusted-upload sandbox.

Every formal claim must bind exact result-changing inputs, code/runtime
identity, checks and status. Git supplies the identity of checked-in files;
content hashes are reserved for external assets, generated closures, execution
receipts and other formal artifacts outside that Git identity. Use `pass`,
`fail`, `blocked`, `not_run`,
`research_only` and `qualified` precisely. Python-only tests cannot substitute
for native Habitat, RLR, Blender or media-readback execution.

## Change discipline

- Preserve unrelated user changes and inspect both worktrees before editing.
- Never use destructive cleanup or broad staging to make a worktree look tidy.
- Keep changes within the repository that owns them; commits in the two repos
  are independent.
- Prefer repository-relative paths and environment/config overrides. Do not add
  private-server absolute paths to current configuration or examples.
- Do not weaken a validator, mock real evidence or edit a hash merely to make a
  gate pass. Record an exact blocker instead.
- Raw third-party room assets are immutable. Derived proxies need explicit
  source identity, operations, hashes and qualification status.

## Test layers

Use the smallest relevant layer first, then run broader regression tests:

1. `fast-unit` — hermetic Python contracts and algorithms.
2. `slow-hermetic` — larger local fixtures without native simulation.
3. `native-habitat` — pinned runtime and scene assets.
4. `rlr-audio` — native RLR propagation and readback.
5. `blender-assets` — Blender-dependent compilation or mesh validation.
6. `media-readback` — encoded video/audio inspection.
7. `release-canary` — cross-repository, hash-bound milestone evidence.

Mark unavailable native layers `not_run` with a reason. A clean fast suite is
required before handoff, but it proves only the hermetic software boundary.

## Git handoff

Stage explicit paths, run `git diff --check`, inspect the staged diff and state
which tests actually ran. Do not push, tag, merge or publish external state
unless the user has authorized that action. Release claims must reference the
single current `release/avengine_release_manifest_v1.json`; older runtime locks
are historical evidence only.
