# Legacy AVEngine Inventory

Status: M0 migration record. The immutable source SHAs are listed in
[`LEGACY_SOURCE_LOCATIONS.md`](LEGACY_SOURCE_LOCATIONS.md). This inventory
describes what exists; it does not approve an artifact for a dataset release.

## Repository snapshot

| Source | Commit | Tracked scope | Role after M0 |
| --- | --- | ---: | --- |
| Legacy AVEngine | `92775d4d2050a3a9b277357eb83c9243468f4cd3` | 103 files | frozen design and review record |
| Legacy SPEAR fork | `7fbf3632fdb63cc2eceea564811c9597cabfb199` | 1,544 files | optional UE/gpuRIR backend and migration source |
| Habitat runtime fork | `57ee4941dc4765240f0f91f70b2c97a919bf9038` | upstream baseline | primary runtime, modified only in its own repository |

The SPEAR snapshot contains 364 tracked files under `tools/` and 255 under
`tests/`. Nine groups remain deliberately untracked because their provenance,
rights, or derivation status is unresolved; they are named in
`LEGACY_SOURCE_LOCATIONS.md` and remain quarantined.

## Legacy execution chains

Two overlapping implementations were used:

1. `tools/gpurir_scenes/`: UE/SPEAR visual rendering plus gpuRIR audio.
2. `tools/spike_rlr/`: UE or Habitat visual rendering plus the stock Habitat
   AudioSensor/RLR path and the Plan 2 `dataset_runner`.

Neither chain is the new production architecture. UE and gpuRIR may survive as
bounded comparison backends, but Habitat is the primary runtime and the new
main repository owns packages, episodes, QA, registries, and dataset assembly.

## Assets and data worth migrating

- Quaternius animal GLBs are **template candidates**, not admitted dataset
  assets. Import requires hashes, source/license evidence, skeleton/action
  inventory, Blender QA, and the new admission state machine.
- The stable template catalog is authoritative over looser legacy registries:
  it records `research_candidate` and
  `formal_dataset_registration_authorized=false`.
- Controlled room specifications, deterministic sampling, event scheduling,
  hashing, provenance, no-overwrite behavior, and Blender geometry QA contain
  reusable concepts.
- The acoustic material database can migrate only after every value gains a
  source, unit, frequency-band definition, confidence, and version.
- Audio-library metadata can migrate only to content-addressed records with
  hashes and explicit usage rights. Machine-local absolute paths are not IDs.
- Old `approved` source-asset states are downgraded to
  `research_candidate_pending_revalidation` on import. Legacy approval is
  never equivalent to `approved_for_dataset`.

## Known invalid or incomplete production assumptions

### Timing and state

- There is no authoritative legacy timeline. Repeated declarations of 75
  frames, 80,000 samples, and five seconds do not prove shared consumption.
- Legacy RLR scripts use `round(16000 / 15) == 1067` samples per frame and
  slice at `f * 1067`; this cannot sum to the required exact boundaries.
- The UE render pass starts a looping animation and advances it freely. With
  view as the outer loop, nominally identical frame numbers across cameras can
  contain different poses.
- A Habitat spike moves a dog GLB as a kinematic rigid object. Its documented
  T-pose/ice-skating behavior is not articulated-animal execution.

### Geometry, materials, and acoustics

- The apartment exporter converts each UE actor AABB into a 12-triangle box.
  This is a debug broad-phase proxy, never production acoustic geometry.
- Legacy tools generate `material_indices`, but the RLR material adapter does
  not consume or upload those indices. Per-triangle material assignment is not
  proven.
- Legacy multi-source RLR is a sequential single-source AudioSensor sweep. It
  is not one modern RLR context containing named sources/listeners and
  independently readable per-pair IRs.
- The distance-derived "DRR proxy" is not an acoustic measurement and must not
  migrate as a label.

### Tests and claims

There are 242 `test_*.py` files in the legacy SPEAR snapshot. At least 56 are
explicit `_static.py` source-contract checks, including 40 Blender static
checks. They may be useful lint/contract tests, but they are not Blender,
Habitat, UE, RLR, GPU, or end-to-end execution evidence.

## Required replacement canaries

- Blender headless import, exact 75-pose bake, deformation, and foot contact.
- Habitat same-state capture with equal per-frame pose hashes across cameras.
- Room openings/connectivity, ray leakage, and visual/acoustic parity.
- Low/high absorption extremes proving actual per-triangle material upload.
- One-context two-source/two-listener RLR with per-pair IR and order invariance.
- Integer timeline boundaries totaling exactly 80,000 samples and 240,000 ticks.
- Counterfactual visual hashes identical while only declared source/audio
  variables change.
- Admission tests proving `not_run`, research candidates, and legacy approvals
  cannot become `approved_for_dataset` automatically.
