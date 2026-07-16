# AVEngine — Habitat-native Audiovisual Dataset Engine

AVEngine is a private research project for deterministic,
identity-preserving audiovisual episode generation. Habitat-Sim is the primary
visual/scene/sensor/physics runtime; RLR Audio Propagation is the geometric
acoustic foundation. AVEngine is not a simulator built from scratch.

This branch has completed **M1: Habitat visual and three-room canary** on top of
the M0 repository/runtime baseline. M2 implementation has reached a bounded
**research-candidate review-only** checkpoint: one candidate dog executed all
75 explicit states in Habitat and produced one-view RGB/depth/semantic review
media. Formal M2 status remains `not_run` because the candidate has not passed
a hash-bound human visual review or become `canary_qualified`. This repository
does **not** yet claim a qualified articulated animal, acoustic propagation, a
complete audiovisual episode, or a dataset release.

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
4. [`docs/adr/ADR-0009-single-view-multimodal-sensor-rig.md`](docs/adr/ADR-0009-single-view-multimodal-sensor-rig.md)
   — the single-view RGB/depth/semantic and listener contract.
5. [`docs/roadmap/MILESTONES.md`](docs/roadmap/MILESTONES.md),
   [`docs/roadmap/BASELINE_STATUS.md`](docs/roadmap/BASELINE_STATUS.md), and
   [`docs/roadmap/M1_STATUS.md`](docs/roadmap/M1_STATUS.md) — gates and actual
   verification state. Use [`docs/roadmap/M1_EXECUTION.md`](docs/roadmap/M1_EXECUTION.md)
   to reproduce the executable evidence.
6. [`docs/roadmap/M2_STATUS.md`](docs/roadmap/M2_STATUS.md) and
   [`docs/roadmap/M2_EXECUTION.md`](docs/roadmap/M2_EXECUTION.md) — the bounded
   M2 candidate evidence, known gait limitation, review media and next formal
   admission gate.
7. [`docs/migration/LEGACY_AVENGINE_INVENTORY.md`](docs/migration/LEGACY_AVENGINE_INVENTORY.md)
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

M1 loads an official Habitat room, a Blender custom room and an audited legacy
UE apartment in the pinned runtime, then captures same-state
RGB/depth/semantic evidence. These three sensors are co-located and co-oriented
on one logical `camera_rig_0` and produce exactly one formal `view_id`
(`view0`), not three viewpoints. The MVP `world_from_rig` is the
camera/listener viewpoint (not an agent foot point), and the listener shares
that rig transform. Every M1 request has at least two uniquely named sources
whose world transforms are pairwise distinct. A top-down navigation QA map is
a diagnostic artifact, not another camera or dataset view.

M1 closes both the declared scene-asset graph and the graph Habitat actually
loads: dataset and scene selection, stage render/collision/semantic assets,
and, for handle-based scenes, source-marker object templates, live poses and
lighting selection must all resolve to the declared files without ambiguity.
Every canary explicitly loads its `load_declared` navmesh. The active
Pathfinder fingerprint must equal an independent Pathfinder load of that same
file, including all navmesh settings and the vertex/index buffer hashes.

The M2 review-only run applies 15 Idle, 45 Walk and 15 Idle states at the exact
75 video ticks, without a free-running action clock or physics step. RGB,
depth and semantic remain co-located modalities of the same `view0`; they are
not additional viewpoints. Automatic numerical QA and Habitat state readback
passed for this bounded execution, but the inherited Walk has a known visual
limitation: its hind legs show little whole-leg forward articulation and much
of their measured motion is lateral/toe-terminal motion. Contact inference
also retains explicit sliding and Idle-motion warnings.

The immediate M2 gate is user review of the exact hash-bound media, including
that known hind-leg behavior. Only an accepted review artifact bound to the
candidate and media hashes, plus the remaining provenance/use decision, may
promote a package to `canary_qualified` and permit a clean formal capture.
Until that happens, formal M2 remains `not_run`; the review-only video is not a
formal canary.

Timeline v2 keeps its plural `view_ids` field for future extensibility, but the
M1, M2 and M5 canaries and the initial M6 MVP require exactly `["view0"]`.

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
implied by the M0/M1 foundation work.

## Contact

Ziyang Ji ([Eastforward](https://github.com/Eastforward)) — research
collaboration welcome; request permission before reuse or redistribution.
