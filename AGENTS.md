# AVEngine single-source contributor rules

## Authority and scope

This repository is the canonical AVEngine source repository:
`https://github.com/USTB-AVEngine/AVEngine.git`. The target architecture keeps
all required distributable source code and small configuration here, including
the selected Habitat-Sim and SPEAR integration source plus the AVEngine-owned
RLR adapter source and small interface configuration. It does not put Unreal
Engine installations, datasets, room assets, model weights, generated media or
build products in Git.

That target is still being migrated. Until the selected runtime code has landed
and pre/post behavior has been checked, the manifest-pinned sibling Habitat
fork and the maintained SPEAR checkout remain transition workspaces. They are
sources and execution dependencies for the migration, not the final repository
architecture and not authority to claim that integration is already complete.

Read `README.md`, `docs/architecture/SYSTEM_OVERVIEW.md`,
`docs/architecture/REPOSITORY_BOUNDARIES.md` and the current milestone status
before changing architecture. Historical SPEAR/UE operating notes are archived
under `docs/legacy/`; current Apartment and Kujiale production routing is
documented under `docs/architecture/` and is not made historical by that
archive.

## Canonical production routing and work-copy policy

These owner decisions override older runbooks, retained attempts and optional
backend examples:

- MP3D production scene execution, visual pixels, sensors and articulated pose
  run in Habitat-Sim. MP3D acoustics use RLR with the SoundSpaces material
  authority on that same Habitat scene and state. An MP3D UE import is only a
  `comparison_visual` diagnostic; it is never production output, admission
  evidence or a counted Episode.
- `apartment_0000` production visual execution uses its native UE/SPEAR map.
- InteriorAgent/Kujiale production visual execution uses the UE/SPEAR USD/MDL
  adapter for the explicitly selected external scene.
- Skokloster is excluded from production execution and dataset counting unless
  the project owner explicitly reauthorizes it for a named task.
- Do not run or count an Episode whose selected backend conflicts with these
  room-family routes. A retained artifact or passing validator from another
  backend does not change the route.

Server code has one working copy: the server repository. Make, test and repair
server changes directly there against the real dependencies and retained data.
A local checkout may transfer a patch or perform a read-only audit, but it must
not become a separate completed implementation that is later copied to the
server.

By default, do not add a hash, frozen contract, baseline or gate. Such a
mechanism is allowed only when the change identifies one concrete failure and
explains why Git identity, versioning, primary keys, transactions, uniqueness,
types and ordinary tests do not prevent it. Preserve existing safety controls:
rights, authentication, data safety, irreversible operations and formal
publication continue to follow their project requirements.

Repository `tmp` is a compatibility symlink whose physical data lives under
`/data/datasets/avengine_workspaces/`. Keep tools and stored evidence using
repository-relative `tmp/...` paths so existing manifests remain readable.
Never replace the symlink with a physical output directory inside this
repository. Git-internal paths such as `.git/lfs/tmp` are not project outputs.
During the migration, apply the same output-storage rule to checked-out SPEAR,
Hunyuan3D and SkinTokens workspaces: keep their project `tmp/...`
compatibility paths, and do not move Git-internal temporary paths. The final
single-source layout must not require a separate SPEAR, Habitat or RLR Git
checkout.
Invoke tools and report artifacts through those repository `tmp/...` paths,
but normalize logical and resolved paths consistently inside any hash-bound
lineage contract. If a producer stores resolved absolute file descriptors,
every producer and consumer in that contract must resolve the `tmp` parent
symlink before comparing path, SHA or size. A raw-string mismatch between a
repository path and its external-storage target is not an asset mutation.

For the active Apartment dataset work, also read
`docs/roadmap/CURRENT_APARTMENT_EXECUTION.md`. `AGENTS.md` contains durable
owner decisions; that file contains the current checkpoint, unfinished work
and the exact next action. Update the checkpoint whenever a meaningful stage
finishes or the owner changes direction.

For the QuestionSpec paper-protocol coverage delivery, the official
`compile` is currently `blocked`. Read
`docs/qa/QUESTION_PROTOCOL_RECOMPILE_BLOCKER_20260817.md` before claiming
`paper_ready` or touching binding-manifest hashes. The RGB canary overlay
renderer is already landed on `cc-qa-overlay-rgb` (`6e43273`); do not
confuse that with an official recompile.

## Current Apartment and generated-animal invariants

These rules are project-owner decisions for the active AVEngine Apartment
training-data route. Do not replace them with an easier canary:

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
- Every new target-native quadruped must pass the shared post-TokenRig runner:
  reviewed heading, four-semantic-foot support-plane leveling, then motion
  retarget and multi-view animation QA. A lowest-point floor shift, per-frame
  foot grounding or UE component Z correction is not support-plane leveling
  and cannot substitute for it. Do not register a newly generated animal from
  a direct retarget output that bypassed this stage.
- For `apartment_0000`, Habitat-native owns the route, Timeline, source
  centers, binaural audio, Topdown and labels; SPEAR/UE owns final RGB pixels.
  Do not silently fall back to Habitat RGB for the final Apartment dataset.
  Actor slots remain generic `source1` and `source2`, regardless of whether an
  episode binds a human, dog or cat.
- Render each selected two-source visual episode once and bind its dry-audio
  variants through the dataset index. A finite single-source path is reusable:
  pairing path A with B and pairing A with C are different valid episodes, so
  the path pool does not need 2,000 one-use paths. The ordered two-source
  combination and concrete asset bindings identify the visual episode.
  Train/validation/test splitting happens at that visual-episode level, never
  at the audio-variant level, so one exact RGB and Topdown episode cannot leak
  across splits. The completed lightweight baseline used 100 visual episodes
  x 10 audio variants. The current owner-requested 1,000-item closure uses
  1,000 visual episodes x 1 audio realization and an 800/100/100 sample split.
- Room identity/resources, visible source-asset runtime data and dry sound
  assets are three independent selections. New runtime-capable animals belong
  in `examples/runtime/source_asset_runtime_profiles.json`; room/backend scene
  choices belong in `examples/runtime/room_runtime_profiles.json`. Production
  Python must not grow a breed list, per-animal muzzle/floor constants or a
  room-map switch. `source1` and `source2` resolve exact registered assets, and
  dry sound continues to resolve through the independent sound registry.
- The current owner-approved generative route uses FLUX without Qwen. Do not
  use low-VRAM modes, CPU offload or sequential model offload; load the model
  directly into available GPU memory. This does not relax output anatomy or
  reference-image review.

## Repository boundaries

- The final product has one source repository. Selectively adapted third-party
  code must live in a clearly owned path with its upstream mapping and license;
  do not import an entire repository merely to avoid choosing the required
  files.
- Preserve upstream behavior by default. AVEngine-specific Habitat, RLR or
  SPEAR behavior remains an explicit adapter or opt-in even after its source is
  integrated here.
- Unreal Engine itself, Epic content, MP3D, InteriorAgent/Kujiale, native
  Apartment scene assets and other external datasets remain runtime inputs.
  They are not source-repository dependencies and are never copied into Git.
- Generated media, native evidence, model weights, caches, build trees and
  large assets belong under ignored output/data roots, not in Git. Track
  schemas, small configuration, compact fixtures, requests, one authoritative
  bundle identity and human-readable status records only where they are
  required.
- SPEAR-backed execution is production visual for Apartment and Kujiale, but
  the external UE installation is loaded only for those explicitly selected
  routes. MP3D remains Habitat-Sim production visual; its UE path is comparison
  only. gpuRIR and generative-asset tooling remain optional research tools.

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
inside declared roots; missing inputs and root escapes are errors. A hash
mismatch is an error when the relevant input or output is explicitly part of a
hash-bound formal artifact, and unauthorized replacement of formal immutable
evidence remains an error. Publish complete bundles with a temporary sibling
plus atomic no-replace commit where supported.

This mode does not claim protection from a malicious local symlink race,
portable `O_NOFOLLOW` directory semantics or general TOCTOU attacks. Do not
describe it as an untrusted-upload sandbox.

External datasets, UE assets, models, textures, audio, HRTFs, SDKs and runtime
stages are iterative runtime inputs. Do not default them to a byte snapshot of
one copied instance. An owner-authorized replacement at the same path is
allowed, and a legal additional asset must not by itself make a run reject.
Ordinary validation should prefer declared roots, package/object paths,
registered ID/revision/type/provenance and live runtime/readback/visual/audio
behavior. If semantic identity changes, update the ordinary revision and
provenance and rerun the relevant validation; do not freeze old bytes merely
to preserve an earlier copy.

Fresh staging should copy only the inputs needed for the selected run where
practical. Its minimal closure is normally build/rights hygiene, not a
universal runtime contract. A one-off pre/post byte comparison may diagnose a
transition, but it must not automatically become a permanent hash, baseline or
gate.

The pipeline must never modify third-party source data in place. Rights,
authentication, data-safety and formal immutable-evidence boundaries remain in
force. The exception above is the only route to adding a hash, frozen contract,
baseline or gate.

Every formal claim must bind exact result-changing inputs, code/runtime
identity, checks and status. Git supplies the identity of checked-in files.
For ordinary runtime inputs, use the declared roots, package/object paths and
registered identity and provenance above; content hashes may be used when an
external asset, generated closure, execution receipt or other formal artifact
is explicitly hash-bound outside that Git identity. Use `pass`, `fail`,
`blocked`, `not_run`,
`research_only` and `qualified` precisely. Python-only tests cannot substitute
for native Habitat, RLR, Blender or media-readback execution.

## Change discipline

- Preserve unrelated user changes and inspect every participating transition
  worktree before editing.
- Never use destructive cleanup or broad staging to make a worktree look tidy.
- Land final product code in this repository. Treat the current Habitat and
  SPEAR workspaces as read-only migration sources except for separately scoped
  fixes required to establish or verify the pre-migration reference.
- Prefer repository-relative paths and environment/config overrides. Do not add
  private-server absolute paths to current configuration or examples.
- Do not weaken a validator, mock real evidence or edit a hash merely to make a
  gate pass. Record an exact blocker instead.
- Do not let a pipeline modify third-party source data in place. An authorized
  replacement of an external runtime copy must update its ordinary revision
  and provenance and trigger relevant validation. Formal immutable evidence
  and rights/auth/data-safety controls remain protected. Derived proxies need
  explicit source identity, operations and qualification status; add a hash only
  under the concrete-failure exception above.

## Test layers

Use the smallest relevant layer first, then run broader regression tests:

1. `fast-unit` — hermetic Python contracts and algorithms.
2. `slow-hermetic` — larger local fixtures without native simulation.
3. `native-habitat` — pinned runtime and scene assets.
4. `rlr-audio` — native RLR propagation and readback.
5. `blender-assets` — Blender-dependent compilation or mesh validation.
6. `media-readback` — encoded video/audio inspection.
7. `release-canary` — full-runtime, media-readback and hash-bound milestone
   evidence.

Mark unavailable native layers `not_run` with a reason. A clean fast suite is
required before handoff, but it proves only the hermetic software boundary.

## Git handoff

Stage explicit paths, run `git diff --check`, inspect the staged diff and state
which tests actually ran. Do not push, tag, merge or publish external state
unless the user has authorized that action. Release claims must reference the
single current `release/avengine_release_manifest_v1.json`; older runtime locks
are historical evidence only.
