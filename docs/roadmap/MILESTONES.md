# AVEngine Milestones

Milestones are sequential evidence gates. Later milestones may be designed in
parallel, but they cannot claim completion before their dependencies pass.

## M0: Repository and Baseline

Deliverables: two-repository governance, exact upstream/submodule lock,
architecture and ADRs, legacy migration matrix, attribution/citations, build
instructions, issue backlog and baseline status table.

Exit criteria:

- Both repositories have explicit origin/upstream roles and clean feature branches.
- Runtime, upstream and RLR commits are pinned.
- Legacy entries have an owner and migration decision.
- Unexecuted GPU/Blender/RLR/E2E checks are recorded as `not_run`.
- The reference fork builds cleanly with audio enabled and the relevant
  original Habitat tests have exact recorded results. A real upstream failure
  may remain `fail`, but M0 cannot silently relabel it `not_run`.

## M1: Habitat Visual and Room Canary

Status: `pass`; see [M1_STATUS.md](M1_STATUS.md) and
[M1_EXECUTION.md](M1_EXECUTION.md).

Deliverables: one Habitat-native room, one Blender custom room and one
legacy-apartment real-surface export; one logical view with co-located
RGB/depth/semantic sensors; coordinate/unit manifests and visual evidence.

Exit criteria: all three room types load reproducibly; custom openings and
connectivity are preserved; the formal capture has exactly
`view_ids == ["view0"]`; the camera rig and single listener transforms agree;
independently named source
transforms round-trip; top-down QA views are excluded from dataset
observations; visual quality is sufficient for the task or a bounded
optional-backend gap is recorded.

## M2: Articulated Dog Runtime

Status: bounded research-canary gate `pass`. The exact candidate passed
automatic QA, hash-bound human review, four-paw world-contact/root-cadence QA,
`canary_qualified` admission and a clean 75-state formal Habitat capture. See
[M2_STATUS.md](M2_STATUS.md) and [M2_EXECUTION.md](M2_EXECUTION.md). This does
not grant `approved_for_dataset`; central dataset admission remains M6 work.

Deliverables: one `canary_qualified` canonical dog package, baked Walk/Idle poses, root
trajectory, semantic anchors, contacts and canonical pose hashes.

Exit criteria: exactly 75 poses execute without a free-running action clock;
the formal single `view0` RGB/depth/semantic capture shares one per-frame state
and pose hash; deformation/contact QA passes; visual mouth articulation is
absent. These criteria passed for the fixed M2 Beagle hashes only.

### M2.1: Appearance and Cross-Species Research Diagnostics

Status: implemented as `research_candidate` evidence only; see
[M2_1_STATUS.md](M2_1_STATUS.md). This is a post-M2 investigation, not a new
sequential admission gate and not permission to start M3 with additional
qualified species.

Deliverables: an immutable OA(9, 4, 3, 2) L9 Beagle request over size, build,
breed-scoped coat and life stage; strict material normalization; independent
visual/action rebinding; scale-derived world-contact QA; generic package and
single-view two-room review tooling; and cat, horse and Golden Retriever
diagnostic probes. The L9 is balanced combination coverage, not OFAT. The
separate one-factor-at-a-time study remains required and `not_run`.

Current boundary: the nine Beagles pass the implemented technical checks but
remain research candidates because OFAT and per-instance human promotion have
not run. Their asset admission remains `blocked`, formal human visual review is
`not_run`, and `qualification_claim` is false. `blender_custom` is a controlled
test room for stable comparison; `habitat_mp3d_example` is the real scanned
MP3D environment. They are two different review conditions, not two claims of
real scanned-room coverage.

The retained hardened local Beagle chain is `beagle_l9_realized_v9` →
`beagle_l9_canonical_visual_v7` → `beagle_l9_rebound_actions_v8` →
`beagle_l9_package_inputs_v10` → `beagle_l9_auto_qa_v11` →
`beagle_l9_probe_v10` → `beagle_l9_final_action_review_v9` →
`beagle_l9_world_contacts_v10` → `beagle_l9_material_readback_v1` →
`beagle_l9_packages_v12` → `beagle_l9_captures_v14` →
`beagle_l9_final_audit_v6`, all under ignored `tmp/m2/` evidence storage. The
v9 outputs bind the exact current appearance realizer and pass an independent
geometry/skin/texture byte audit; their visual bytes are identical to the
immutable v8 predecessor. The paths are a local evidence index, not release
artifacts. M2.1 variants bind Habitat `pbr` explicitly through spec, probes
and package; the formal M2 default remains `phong`.

The indexed `cross_species_delivery_v7` PBR assessment passes the six technical
captures with an overall minimum semantic margin of 8 pixels and fixes the
Golden lacquer/copper material response. The project owner accepts the exact
cat and Golden research videos and rejects the historical v7 horse because its
legs fold unnaturally. Golden current AVEngine research use is
project-owner-authorized and is not a current rights blocker. The corrective
`horse_local_tr_review_v2` preserves authored child local translations and
rotations and passes two-room runtime readback plus engineering visual
self-review without the v7 folded legs. It remains review-only and is not a
formal horse admission or a hash-bound owner decision.

The final `topdown_review_delivery_v4/videos/` tree contains 24 synchronized
right-side Topdown videos: all nine Beagle variants and cat, Golden Retriever
and corrected horse in both rooms. Actor heading is derived from the nearest
non-zero trajectory tangent; camera heading remains the rig's local negative-Z
axis. MP3D object footprints are read from its
semantic `.house` descriptor, not inferred by a detector; the custom room has
no descriptor and draws none. The installed MP3D config has an empty
`light_setups` table and `default_lighting: no_lights`, so its apparent light is
baked into textures and a runtime shadow-casting light is required for normal
dynamic animal shadows, including after a mesh-only UE import.

Cat, horse and Golden remain `research_candidate`, no species-specific formal
promotion has run, and `qualification_claim` is false. No probe is an admitted
asset. Avian support is out of scope until a hash-bound body-plan profile
replaces the terrestrial four-paw/muzzle assumptions.

## M3: Acoustic Scene and Materials

Status: bounded controlled material-activation gate `pass`. The retained
formal run independently verified the compiler replay and all 39 required
native evidence checks across three fresh-context repeats per condition. See
[M3_STATUS.md](M3_STATUS.md), [M3_EXECUTION.md](M3_EXECUTION.md) and
[ACOUSTIC_SCENE_AND_MATERIALS.md](../architecture/ACOUSTIC_SCENE_AND_MATERIALS.md).

Deliverables: Acoustic Scene Package schema/compiler, exact source-slot to
per-triangle material mapping, modern explicit RLR adapter ingestion, material
coverage, source replay, exact API receipts, post-ingestion debug geometry and
a controlled extreme-material canary.

RLR supplies the geometric propagation algorithm. AVEngine does not claim a
new solver; it supplies explicit, hash-bound compilation/adapter inputs and
independently verified evidence around the reused algorithm.

The controlled custom-room high/low pair is a synthetic material-activation
test. `package_mode: production` invokes the strict compiler path, but
`material_semantics: controlled_canary` and
`qualification_claim: synthetic_activation_test_only` prohibit treating its
coefficients as physical room-material truth. MP3D and UE visual-slot mappings
remain `research_candidate` proposals without physical qualification or
admission.

Exit criteria: every production triangle is assigned; no unintended fallback
is used; openings/geometry survive; production uses no AABB room proxy; exact
source replay and native ingestion receipts match; and absorption extremes
create a repeatable RIR/EDT/DRR/late-energy difference beyond run variance.
The post-ingestion OBJ verifies geometry and resolved material blocks but cannot
by itself prove per-face material IDs, so the gate also requires exact API
receipts and source-to-package material replay. These criteria passed for the
fixed Blender custom-room `0.02` / `0.60` synthetic fixture only. They do not
qualify physical coefficients, MP3D/UE research proposals, or a dataset room.

## M3.1: User-Controlled Acoustic Profiles

Status: the deterministic coefficient-profile path is `pass`; native
target-decay calibration evidence is `not_run`. This post-gate extension does
not rewrite the fixed M3 formal result or its runtime-lock input hash.

Deliverables: versioned profile schema, scalar-to-band expansion,
base/global/exact-material precedence, fail-closed selector validation,
complete effective mapping/database output, field-level lineage report, CLI,
example and tests. A separate bounded calibration API targets caller-reported
broadband EDT through a caller-owned evaluator.

Exit criteria for the completed coefficient path: identical input bytes
resolve byte-identically regardless of their filesystem location; global
values reach every mapped material; an exact material
override wins only for its resolved key; unknown, shared, duplicate,
conflicting selectors and wrong-band arrays fail; and the effective database
passes the existing strict M3 compiler path and package verifier.

Native target-decay exit criteria remain open: fixed calibration anchors and
simulation settings, retained RIRs, acceptable measurement quality and repeat
spread, a reachable monotonic bracket, achieved tolerance and independently
verified evidence. RT60 is not a material field or direct RLR setting; current
AVEngine-owned measurement support is broadband EDT.

Non-goals: visual-to-physical material inference, measured real-room
coefficients, a universal room RT60, or any M4 multi-source claim.

## M4: Multi-Source RLR

Status: `not_run`. M3's single-pair material canary and any named-context API
groundwork do not complete this gate.

Deliverables: modern RLR C API adapter, named sources/listeners with exactly
one listener in the MVP profile, per-pair IRs, independent stems,
reset/temporal policy and performance report.

Exit criteria: at least two sources and the single MVP listener maintain
actor/event/anchor identity; the listener remains co-located with the formal
camera rig;
source registration order does not create a systematic output change; each
pair result is independently readable.

## M5: Timeline and Counterfactual Episode

Deliverables: timeline builder/semantic validator, deterministic fixed-state
capture, exact frame/sample assembly and vocalizing-actor swap pair.

Exit criteria: 75 frames, 80,000 samples and 240,000 ticks read back exactly;
`video.view_ids` is exactly `["view0"]`; the counterfactual pair has identical
RGB/depth/semantic hashes from that view; only declared audio/source variables
change; no mouth motion is present.

## M6: Dataset MVP

Deliverables: stable CLI, asset/scene/episode registries, QA aggregation,
provenance manifests, structured rejection and deterministic rerun.

Exit criteria: two actor instances of one canonical Dog asset + custom room +
at least two named sources are admitted end to end; the same request/seed
reproduces compatible timeline/manifests with exactly the formal `view0` and
one co-located listener; QA-only cameras are excluded; `not_run` cannot be
promoted to `pass`.

## M7: Benchmark and Paper Release

Deliverables: Dynamic Articulated Source Attribution task, splits, loaders,
baselines, ablations, metrics, release manifests and paper artifacts.

Exit criteria: visual/audio/audio-visual baselines and counterfactual/sync/anchor
ablations run on frozen splits; reused/extended/original claims and all required
citations are consistent across code, dataset card and paper.
