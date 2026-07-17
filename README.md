# AVEngine — Habitat-native Audiovisual Dataset Engine

AVEngine is a private research project for deterministic,
identity-preserving audiovisual episode generation. Habitat-Sim is the primary
visual/scene/sensor/physics runtime; RLR Audio Propagation is the geometric
acoustic foundation. AVEngine is not a simulator built from scratch.

This branch has completed **M1: Habitat visual and three-room canary** and the
bounded **M2 articulated-dog research canary** on top of the M0
repository/runtime baseline. The final M2 package is `canary_qualified`: its
automatic QA, hash-bound human visual review, world-contact/root-cadence gate
and clean 75-state Habitat capture all passed. The qualification is scoped to
research-canary use; it does **not** authorize formal dataset registration or
claim acoustic propagation, a complete audiovisual episode, or a dataset
release.

The follow-on **M2.1 appearance/species workstream** is also implemented as a
research-only diagnostic path. It realizes a balanced nine-point Beagle
appearance design over size, build, breed-scoped coat and life stage, then
rebinds and independently verifies the visual, skin, action, material,
contact, package and two-room Habitat evidence. Every M2.1 result remains a
`research_candidate`: no new species has a formal promotion decision and
`qualification_claim` is false. The project owner accepted the exact cat and
Golden Retriever v7 research videos and rejected the historical v7 horse's
folded-leg motion. A later research-only local-TR v2 horse route preserves the
authored child-joint translations, passes two-room runtime readback and
engineering visual self-review, and replaces that broken v7 motion for current
preview. It is not a hash-bound project-owner decision or a formal species
admission. The OA(9, 4, 3, 2) L9 provides balanced combination coverage, not
one-factor-at-a-time evidence, and the required separate OFAT study remains
`not_run`. Cat, horse and Golden Retriever probes do not grant new-species
admission.

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
3. [`docs/architecture/MOTION_RETARGETING.md`](docs/architecture/MOTION_RETARGETING.md)
   — offline rest-aware motion math, body-plan adapters and independent QA
   gates.
4. [`docs/architecture/REPOSITORY_BOUNDARIES.md`](docs/architecture/REPOSITORY_BOUNDARIES.md)
   — code ownership and API boundary.
5. [`docs/adr/ADR-0009-single-view-multimodal-sensor-rig.md`](docs/adr/ADR-0009-single-view-multimodal-sensor-rig.md)
   — the single-view RGB/depth/semantic and listener contract.
6. [`docs/roadmap/MILESTONES.md`](docs/roadmap/MILESTONES.md),
   [`docs/roadmap/BASELINE_STATUS.md`](docs/roadmap/BASELINE_STATUS.md), and
   [`docs/roadmap/M1_STATUS.md`](docs/roadmap/M1_STATUS.md) — gates and actual
   verification state. Use [`docs/roadmap/M1_EXECUTION.md`](docs/roadmap/M1_EXECUTION.md)
   to reproduce the executable evidence.
7. [`docs/roadmap/M2_STATUS.md`](docs/roadmap/M2_STATUS.md) and
   [`docs/roadmap/M2_EXECUTION.md`](docs/roadmap/M2_EXECUTION.md) — the exact M2
   candidate, admission, contact/cadence and formal Habitat evidence, plus the
   exact local replay path from retained hash-bound intermediates.
8. [`docs/roadmap/M2_1_STATUS.md`](docs/roadmap/M2_1_STATUS.md) — the exact
   appearance L9 contract, current two-room research evidence, body-plan
   boundary and cross-species blockers.
9. [`docs/migration/LEGACY_AVENGINE_INVENTORY.md`](docs/migration/LEGACY_AVENGINE_INVENTORY.md)
   — what is reusable, optional, experimental, or retired.

The authoritative timeline schema is
[`schemas/avengine_timeline_v2.schema.json`](schemas/avengine_timeline_v2.schema.json).
It fixes a five-second episode at 48 kHz ticks, 15 fps/75 frames, and 16 kHz/
80,000 audio samples. A semantic validator will be implemented in M5; schema
presence alone is not proof that a generated episode is synchronized.

## Milestones

| Gate | Outcome |
| --- | --- |
| M0 | repositories, locks, architecture, migration, licenses, baseline (`pass`) |
| M1 | Habitat visual and three-room canary (`pass`) |
| M2 | deterministic articulated Dog runtime — fixed Beagle canary (`pass`) |
| M2.1 | appearance L9 and cross-species two-room diagnostics — research-only evidence (`pass`) |
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

The M2 canary uses a profile-bound world-left retargeted Idle/Walk action. The
formal run applies 15 Idle, 45 Walk and 15 Idle states at exact video ticks,
without a free-running action clock or physics step; RGB, depth and semantic
remain co-located modalities of the same `view0`, not additional viewpoints.
The legacy hind-leg under-articulation metric no longer triggers. A
body-plan-neutral cadence solver binds the exact four-paw contact phases to a
`0.297 m/s` root trajectory; the maximum world-space contact step is
`0.013894547981602673 m`, below the `0.015 m` gate. The user accepted the
unchanged visual/action hashes, and the final formal capture passed from clean,
locked AVEngine and Habitat worktrees. See
[`MOTION_RETARGETING.md`](docs/architecture/MOTION_RETARGETING.md).

The resulting package is qualified only for this bounded M2 research canary.
Every new species, motion family, appearance realization or dataset admission
still starts fail-closed and requires its own exact evidence.

M2.1 keeps that boundary explicit. The Beagle L9 uses the same single logical
camera with co-located RGB/depth/semantic sensors in `blender_custom`, a
deliberately controlled test room, and `habitat_mp3d_example`, a real MP3D
scan. The custom room is not presented as a naturally furnished or scanned
environment. All 18 local technical captures contain 75 frames at 15 fps, keep
the complete semantic animal mask at least 13 pixels from every image edge
and remain review-only. The hardened appearance path forces effective opaque
base color, zero emission, bounded non-metallic/specular response and rejects
material routes that can bypass those gates. The similarity bake independently
fails closed when its scale node or a relevant ancestor has scale animation.
M2.1 also binds the Habitat articulated-object shader explicitly from the
variant spec through both runtime probes and the compiled package. These
research variants use `pbr`; the qualified formal M2 baseline retains its
backward-compatible `phong` default and is not silently re-rendered.

The retained hardened Beagle rebuild is indexed by
`beagle_l9_realized_v9`, `beagle_l9_canonical_visual_v7`,
`beagle_l9_rebound_actions_v8`, `beagle_l9_package_inputs_v10`,
`beagle_l9_auto_qa_v11`, `beagle_l9_probe_v10`,
`beagle_l9_final_action_review_v9`, `beagle_l9_world_contacts_v10`,
`beagle_l9_material_readback_v1`, `beagle_l9_packages_v12`,
`beagle_l9_captures_v14` and `beagle_l9_final_audit_v6` under `tmp/m2/`.
The v9 appearance reports bind the exact current realizer and pattern audit;
an independent byte-level verifier checks geometry, skin and texture output
against snapshotted inputs. Its realized GLB and PNG bytes are identical to
the immutable v8 predecessor. The new package lineage therefore reuses only
hash-bound immutable visual/action/QA evidence, while v14 is a fresh capture
bound to the v12 manifests. Those ignored local paths form an evidence index,
not a release artifact.

The indexed `cross_species_delivery_v7` rebuild uses the same explicit PBR
contract for cat, horse and Golden Retriever. All six two-room technical
captures pass with an overall minimum semantic margin of 8 pixels, and the
earlier Golden copper/lacquer response is gone. The owner accepts the cat and
Golden research visuals, including slight sliding, but rejects the historical
v7 horse motion. Golden current AVEngine research use is explicitly
project-owner-authorized and is not a current rights blocker. The corrective
`horse_local_tr_review_v2` captures preserve the horse action's authored local
translations and rotations, pass fixed-state Habitat readback in both rooms,
and pass current engineering visual self-review without the folded legs. They
remain review-only with `qualification_claim: false`, no formal view IDs and no
species-specific promotion.

The final synchronized review bundle is
`tmp/m2/topdown_review_delivery_v4/videos/`: 18 Beagle L9 videos plus two each
for cat, Golden Retriever and the corrected horse, for 24 right-side Topdown
videos in total. The panel is derived QA media, not another sensor or dataset
view. Actor arrows follow the nearest non-zero trajectory tangent, while the
camera arrow follows the rig's local negative-Z axis. MP3D furniture
footprints come from its Habitat semantic descriptor,
not an object detector; the custom room has no such descriptor and therefore
draws zero object footprints rather than fabricating them.

The installed MP3D dataset config has an empty `light_setups` table and
`default_lighting: no_lights`; its apparent illumination is baked into the
scan textures. Importing the room mesh alone into an unlit UE level therefore
does not provide a normal dynamic animal shadow. A runtime shadow-casting
light must be added and calibrated against the baked appearance.

The three cross-species assets remain review-only because no species-specific
formal promotion has occurred, and `qualification_claim` is false. See
[`M2_1_STATUS.md`](docs/roadmap/M2_1_STATUS.md) for why these technical passes
still do not promote an appearance or species.

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
