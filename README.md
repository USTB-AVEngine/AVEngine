# AVEngine — Habitat-native Audiovisual Dataset Engine

AVEngine is a private research project for deterministic,
identity-preserving audiovisual episode generation. Habitat-Sim is the primary
visual/scene/sensor/physics runtime; RLR Audio Propagation is the geometric
acoustic foundation. AVEngine is not a simulator built from scratch.

This branch is currently at **M0: repository and baseline**. It contains the
architecture, version locks, immutable timeline schema, migration policy,
roadmap, and attribution records. It does **not** yet claim a functioning
Habitat-native animal episode or dataset release.

## Repository boundary

| Repository | Owns |
| --- | --- |
| `Eastforward/AVEngine` (this repository) | asset/room/episode packages, integer timeline, CLI, registries, QA, provenance, dataset admission, benchmark and paper artifacts |
| `Eastforward/habitat-sim-AVEngine` | isolated Habitat runtime extensions: articulated playback, explicit acoustic ingestion, modern RLR adapter, runtime tests |

Legacy UE/SPEAR + gpuRIR material remains migration evidence and an optional
comparison route. It is no longer the primary architecture or setup path.

## Start here

Read these records in order:

1. [`docs/planning/README.md`](docs/planning/README.md) — planning authority and
   immutable imported inputs.
2. [`docs/architecture/SYSTEM_OVERVIEW.md`](docs/architecture/SYSTEM_OVERVIEW.md)
   — target data and execution flow.
3. [`docs/architecture/REPOSITORY_BOUNDARIES.md`](docs/architecture/REPOSITORY_BOUNDARIES.md)
   — code ownership and API boundary.
4. [`docs/roadmap/MILESTONES.md`](docs/roadmap/MILESTONES.md) and
   [`docs/roadmap/BASELINE_STATUS.md`](docs/roadmap/BASELINE_STATUS.md) — gates
   and actual verification state.
5. [`docs/migration/LEGACY_AVENGINE_INVENTORY.md`](docs/migration/LEGACY_AVENGINE_INVENTORY.md)
   — what is reusable, optional, experimental, or retired.

The authoritative timeline schema is
[`schemas/avengine_timeline_v2.schema.json`](schemas/avengine_timeline_v2.schema.json).
It fixes a five-second episode at 48 kHz ticks, 15 fps/75 frames, and 16 kHz/
80,000 audio samples. A semantic validator will be implemented in M5; schema
presence alone is not proof that a generated episode is synchronized.

## Milestones

| Gate | Outcome |
| --- | --- |
| M0 | repositories, locks, architecture, migration, licenses, baseline |
| M1 | Habitat visual and three-room canary |
| M2 | deterministic articulated Dog runtime |
| M3 | explicit acoustic scene and verified materials |
| M4 | modern named multi-source/listener RLR |
| M5 | exact timeline and visual-invariant counterfactual pair |
| M6 | registry/QA/CLI and admitted dataset canary |
| M7 | benchmark, ablations, paper and release audit |

The immediate next implementation step after M0 is M1: load one minimal room
in the pinned Habitat runtime and capture same-state RGB/depth/semantic sensor
evidence before introducing animal animation or production acoustics.

## Status vocabulary

Verification uses only `pass`, `fail`, `blocked`, and `not_run`. Static source
checks do not count as Blender, Habitat, GPU, RLR, or end-to-end execution.
Research candidates and legacy `approved` records cannot be promoted to
`approved_for_dataset` without fresh evidence.

## Citation, rights, and release constraints

See [`CITATION.cff`](CITATION.cff), [`CITATIONS.bib`](CITATIONS.bib), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Habitat retains its MIT
license and upstream attribution. RLR is CC BY-NC 4.0 and limits the current
audio runtime route to non-commercial use. Models, rooms, audio, and derived
assets have separate terms and are admitted item by item.

The AVEngine repository itself is currently private and all-rights-reserved;
see [`LICENSE`](LICENSE). No open-source or dataset redistribution decision is
implied by this M0 restructure.

## Contact

Ziyang Ji ([Eastforward](https://github.com/Eastforward)) — research
collaboration welcome; request permission before reuse or redistribution.
