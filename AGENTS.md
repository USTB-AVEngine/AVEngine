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

## Repository boundaries

- Do not copy Habitat-Sim source into this repository.
- Runtime C++/binding changes belong in the sibling runtime fork and must keep
  upstream behavior as the default unless a reviewed AVEngine opt-in is used.
- Generated media, native evidence and large assets belong under ignored
  output roots, not in Git. Track schemas, compact fixtures, requests, hashes
  and human-readable status records.
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

Every formal claim must bind exact inputs, code/runtime identity, artifact
hashes, checks and status. Use `pass`, `fail`, `blocked`, `not_run`,
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
