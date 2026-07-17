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

**M3 explicit acoustic-scene/material activation is `pass` for its fixed,
controlled canary.** It replaces implicit/AABB acoustics with hash-bound
surface geometry, exact per-triangle material assignment and a modern RLR
ingestion adapter. Its synthetic `0.02` / `0.60` contrast tests material-path
activation and repeatability, not physical room-material truth. The detailed
authoritative record is in
[`M3_STATUS.md`](docs/roadmap/M3_STATUS.md) and
[`MILESTONES.md`](docs/roadmap/MILESTONES.md).
MP3D and UE visual-slot material proposals remain unqualified
`research_candidate` diagnostics.

The post-gate **M3.1 user-control extension** adds a versioned acoustic
material profile resolver. It applies deterministic global and exact
per-material coefficient overrides without changing the mesh or Habitat/RLR,
then emits a complete effective mapping/database and lineage report. A bounded
broadband EDT calibration core is implemented for caller-supplied evaluations;
native target-decay calibration evidence remains `not_run`, and no RT60 or
physical-material truth is inferred.

The **M4 named multi-source RLR implementation** now provides one-context,
all-pair propagation for at least two stable source IDs and exactly one
camera-co-located listener. It retains independent per-source FOA and binaural
stems plus their canary mixtures, freezes raw RLR FOA as ACN/N3D
`[W, Y, Z, X]` in `avengine_world`, authenticates an explicit MIT KEMAR HRTF,
and tests caller-order invariance, native endpoint receipts, reset/temporal
behavior and source-count performance. Its formal bounded gate result is
`pass` with 10/10 declared and 14/14 independently recomputed checks; see
[`M4_STATUS.md`](docs/roadmap/M4_STATUS.md). M4 emits WAV evidence only.

The bounded **M5 exact-timeline/counterfactual canary** is also `pass`. It
executes dynamic articulated muzzle anchors through the same named FOA and
binaural trajectories, assembles exact 75-frame/80,000-sample episodes, keeps
both sources active in six simultaneous windows, and proves that A/B visual
packet payloads are byte-identical while only the declared dry-audio routing
is swapped. Both formal and right-side Topdown listening videos pass mux and
readback checks. The retained result passed 9/9 declared checks and 12/12
independent verification groups; see
[`M5_STATUS.md`](docs/roadmap/M5_STATUS.md). This is a bounded research canary,
not dataset admission; `qualification_claim` remains false.

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
5. [`docs/architecture/ACOUSTIC_SCENE_AND_MATERIALS.md`](docs/architecture/ACOUSTIC_SCENE_AND_MATERIALS.md)
   — explicit geometry/material compilation, RLR ownership, ingestion evidence
   and the M3/M4 boundary.
6. [`docs/adr/ADR-0009-single-view-multimodal-sensor-rig.md`](docs/adr/ADR-0009-single-view-multimodal-sensor-rig.md)
   — the single-view RGB/depth/semantic and listener contract.
7. [`docs/roadmap/MILESTONES.md`](docs/roadmap/MILESTONES.md),
   [`docs/roadmap/BASELINE_STATUS.md`](docs/roadmap/BASELINE_STATUS.md), and
   [`docs/roadmap/M1_STATUS.md`](docs/roadmap/M1_STATUS.md) — gates and actual
   verification state. Use [`docs/roadmap/M1_EXECUTION.md`](docs/roadmap/M1_EXECUTION.md)
   to reproduce the executable evidence.
8. [`docs/roadmap/M2_STATUS.md`](docs/roadmap/M2_STATUS.md) and
   [`docs/roadmap/M2_EXECUTION.md`](docs/roadmap/M2_EXECUTION.md) — the exact M2
   candidate, admission, contact/cadence and formal Habitat evidence, plus the
   exact local replay path from retained hash-bound intermediates.
9. [`docs/roadmap/M2_1_STATUS.md`](docs/roadmap/M2_1_STATUS.md) — the exact
   appearance L9 contract, current two-room research evidence, body-plan
   boundary and cross-species blockers.
10. [`docs/roadmap/M3_STATUS.md`](docs/roadmap/M3_STATUS.md) and
    [`docs/roadmap/M3_EXECUTION.md`](docs/roadmap/M3_EXECUTION.md) — M3's exact
    formal record, claim boundary and compiler/native replay procedure.
11. [`docs/roadmap/M4_STATUS.md`](docs/roadmap/M4_STATUS.md) and
    [`docs/roadmap/M4_EXECUTION.md`](docs/roadmap/M4_EXECUTION.md) — M4's named
    endpoint, FOA/binaural, HRTF, lifecycle and evidence boundary.
12. [`docs/roadmap/M5_STATUS.md`](docs/roadmap/M5_STATUS.md) and
    [`docs/roadmap/M5_EXECUTION.md`](docs/roadmap/M5_EXECUTION.md) — exact
    timeline, dynamic audio, counterfactual invariance and video readback.
13. [`docs/migration/LEGACY_AVENGINE_INVENTORY.md`](docs/migration/LEGACY_AVENGINE_INVENTORY.md)
   — what is reusable, optional, experimental, or retired.

The authoritative timeline schema is
[`schemas/avengine_timeline_v2.schema.json`](schemas/avengine_timeline_v2.schema.json).
It fixes a five-second episode at 48 kHz ticks, 15 fps/75 frames, and 16 kHz/
80,000 audio samples. M5 now supplies the semantic validator, exact builder
and independent readback; schema presence alone is still not proof that an
arbitrary generated episode is synchronized.

## Milestones

| Gate | Outcome |
| --- | --- |
| M0 | repositories, locks, architecture, migration, licenses, baseline (`pass`) |
| M1 | Habitat visual and three-room canary (`pass`) |
| M2 | deterministic articulated Dog runtime — fixed Beagle canary (`pass`) |
| M2.1 | appearance L9 and cross-species two-room diagnostics — research-only evidence (`pass`) |
| M3 | explicit acoustic scene and synthetic material-activation canary (`pass`) |
| M3.1 | global/per-material acoustic profiles (`pass`); native target-decay calibration evidence (`not_run`) |
| M4 | modern named multi-source/listener RLR, per-source FOA/binaural WAV stems and canary mixtures (`pass`, bounded software/source-pose gate) |
| M5 | exact timeline, visual-invariant counterfactual pair and 2ch binaural video mux/readback (`pass`, bounded research canary) |
| M5.1 | mixed human/dog real-room and legacy 18-second comparison, detailed source/event/flag metadata (research extension in progress) |
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

M3 keeps visual and acoustic material semantics separate. The compiler expands
the source GLB to canonical surface triangles, requires exact source-slot
mappings with complete per-triangle coverage, emits a versioned RLR database
and independently replays the hash-bound source inputs. The Habitat fork owns
only the strict modern RLR context/ingestion bridge; RLR remains the propagation
algorithm. AVEngine owns the explicit package, adapter inputs and evidence.

The controlled custom-room low/high pair is a synthetic activation experiment:
all geometry, object partitions, material IDs and non-absorption fields are
frozen, while every high absorption coefficient is greater than its low
counterpart. A passing formal run proves that those coefficients affect RLR
repeatably, not that they are physical measurements for the modeled surfaces.
MP3D and UE visual-material-slot mappings remain research proposals without
physical qualification or admission.

Detailed formal measurements, evidence hashes and the exact runtime-lock input
hash are recorded in [`M3_STATUS.md`](docs/roadmap/M3_STATUS.md).

M3.1 does not guess physical acoustic materials from visual appearance. It
starts from an explicit base database and resolves controls in this order:
base database, then global override, then an exact per-material override.
Curve scalars (`absorption`, `scattering`, `transmission`, `damping`) are
broadcast to every frequency in `bands_hz`; arrays must match the base database
band count exactly. `density` and `speed` remain scalars. Unknown, duplicate,
ambiguous or conflicting selectors fail instead of falling back. A
source-material selector is also rejected when several source slots share its
`material_key`; select that key to change all shared surfaces, or split the
material explicitly. Modifying a
`reviewed_physical` base automatically downgrades the effective database to a
research placeholder until the complete resolved database is reviewed again.

The tracked example is
[`material_profile_example.json`](examples/m3/blender_custom/material_profile_example.json):

```json
{
  "schema": "avengine_m3_acoustic_material_profile_v1",
  "profile_id": "blender_custom_user_control_example_v1",
  "room_id": "blender_custom_two_zone_v1",
  "global_override": {"absorption": 0.2, "scattering": 0.05},
  "material_overrides": [
    {
      "selector": {"source_material_name": "FloorWarmGray"},
      "absorption": [0.08, 0.24, 0.57, 0.69]
    },
    {
      "selector": {"material_key": "doorframe_extreme_f84c"},
      "absorption": 0.1
    }
  ]
}
```

The profile JSON is the normal user-editable surface; routine tuning does not
modify `runtime.lock.yaml`, the mesh, mapping or base database. A global-only
profile may omit `material_overrides`. Resolve a profile before using the
existing strict compiler path. The output directories must not already exist:

```bash
export HABPY=/path/to/the/python/environment/with/avengine
export RUN_ID=20260717_01
"$HABPY" -m pip install -e .

"$HABPY" -m avengine.cli m3 resolve-materials \
  --mapping examples/m3/blender_custom/mapping.json \
  --base-materials examples/m3/blender_custom/materials_low.json \
  --profile examples/m3/blender_custom/material_profile_example.json \
  --output "tmp/m3/user_profile_${RUN_ID}"

"$HABPY" -m avengine.cli m3 compile-custom \
  --room examples/m1/rooms/blender_custom/room_manifest.json \
  --mapping "tmp/m3/user_profile_${RUN_ID}/mapping.json" \
  --materials "tmp/m3/user_profile_${RUN_ID}/materials.json" \
  --output "tmp/m3/user_profile_package_${RUN_ID}"
```

The example coefficients are user controls, not measurements. RT60/T60 is a
decay result measured from an RIR for declared geometry, solver settings and
source/listener positions; it is not a per-material field or direct RLR
setting. The independently implemented M3 metric is broadband EDT. Likewise,
`global_volume` is an output/IR amplitude scale and `max_ir_seconds` is only an
IR duration limit, not a reverberation-time control. See
[`ACOUSTIC_SCENE_AND_MATERIALS.md`](docs/architecture/ACOUSTIC_SCENE_AND_MATERIALS.md)
for the full contract.

Post-ingestion OBJ readback verifies native geometry counts and coordinate
multisets, but the OBJ format does not expose recoverable per-face material
IDs. Per-triangle assignment is instead verified by source replay, exact API
receipts and resolved material blocks. M3 does not itself prove the separate M4
boundary. M4 adds named multi-source/listener all-pair IRs, independent stems
and mixtures, source-order invariance, native endpoint receipts, reset/temporal
policy and performance evidence.

M4's authoritative dataset-audio representation is a four-channel IEEE-float
WAV with raw RLR FOA ordered `[W, Y, Z, X]` (ACN indices `[0, 1, 2, 3]`, N3D,
`avengine_world`). Each source retains an independent FOA stem before the
canonical no-normalization/no-limiter sum. The same source pairs are also
rendered through the pinned explicit MIT KEMAR HRTF into independent
`[left, right]` binaural stems and a two-channel listening mixture. The render
rate is 16 kHz; the HRTF asset is 44.1 kHz and any required adaptation occurs
inside the exact RLR binary pinned by the M4 runtime lock. AVEngine performs no
implicit resampling.

These are full-tail canary WAVs, not final five-second episode audio. M4 does
not put FOA into MP4 and does not mux a review video. M5 now owns and has
executed exact 80,000-sample timeline assembly, counterfactual pairing,
tail/crop policy and two-channel binaural video mux/readback. Reproduce M4 with
[`M4_EXECUTION.md`](docs/roadmap/M4_EXECUTION.md).

The checked-in M4 identity fixture binds each source to a formal M1 source pose.
Its event-time M2 dynamic-anchor evidence remains explicitly `not_run`.
Consequently, even a formal M4 pass is only a bounded software/source-pose
canary: it does not admit an animal asset, acoustic room, episode or dataset.

M5 retains every source's exact four-channel FOA and two-channel binaural WAV
stem plus the canonical mixtures. The ordinary MP4 contains only the
two-channel binaural listening copy; FOA remains an independent WAV because
generic MP4 players do not reliably preserve its order and normalization
metadata. The M5 request is authoritative for actor/source identity, semantic
IDs, emitter links, dry clip, fade, gain and simultaneous event windows.
Episode A/B share the same 75 visual packets and differ only by the declared
dry-audio route swap. Independent verification reconstructs the dry buses,
dynamic convolution stems and mixtures from retained inputs/RIR arrays and
also verifies video packet identity. See
[`M5_STATUS.md`](docs/roadmap/M5_STATUS.md) and reproduce it with
[`M5_EXECUTION.md`](docs/roadmap/M5_EXECUTION.md).

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
