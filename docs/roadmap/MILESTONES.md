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
not rewrite the fixed M3 formal result or its selected historical runtime
profile.

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

Status: bounded M4 software/source-pose canary `pass`. The retained formal run
passed 10/10 declared checks and 14/14 independently recomputed verifier checks;
its exact commits, authoritative bundle identities, measurements and test
totals are recorded in [M4_STATUS.md](M4_STATUS.md).

Deliverables: modern RLR C API adapter; at least two stable named sources and
exactly one camera-co-located MVP listener; all listener/source-pair IRs;
independent per-source FOA and binaural stems; canonical full-tail canary
mixtures; native endpoint receipts; order-invariance, reset/temporal and
performance evidence; and an explicit licensed HRTF dependency.

The frozen FOA contract is raw RLR `[W, Y, Z, X]`, ACN/N3D, right-handed
`avengine_world` with +X right, +Y up and -Z forward. The listening output is
two-channel `[left, right]` RLR-native binaural using the explicit MIT KEMAR
SOFA asset. Rendering remains 16 kHz; its 44.1 kHz HRTF input may be adapted
only inside the exact RLR binary named by the M4 lock. AVEngine performs no
implicit resampling, normalization or limiting.

Exit criteria: source IDs preserve actor/event/anchor routing; native receipts
match canonical indices and every declared endpoint field; the single listener
matches the formal M1 camera rig; caller registration order produces exact
mapped full-indirect IR equality; each pair IR and stem is independently
readable and the mixtures reconstruct exactly; FOA axis/world and binaural
left/right probes pass; reset reproduces the initial temporal frame exactly;
the named source update retains identity; and one-source versus multi-source
performance is measured.

Claim boundary: the current identity fixture uses formal M1 static source
poses, while event-time M2 dynamic-anchor evidence remains `not_run`. An M4
pass therefore qualifies this bounded software/source-pose path only. It is not
animal-asset, room, episode or dataset admission. M4 emits WAV artifacts only;
M5 owns the exact five-second timeline, counterfactual pair, tail/crop policy
and two-channel binaural video mux/readback.

## M5: Timeline and Counterfactual Episode

Status: bounded research-canary gate `pass`. The retained clean-worktree run
passed 9/9 declared checks and 12/12 independent readback/reconstruction
groups. See [M5_STATUS.md](M5_STATUS.md) and
[M5_EXECUTION.md](M5_EXECUTION.md). `qualification_claim` is false and dataset
admission remains M6 work.

Deliverables: timeline builder/semantic validator, deterministic fixed-state
capture, dynamic named-source FOA/binaural rendering, exact frame/sample
assembly, declared dry-audio route-swap pair, and formal/Topdown listening
videos.

Exit criteria: 75 frames, 80,000 samples and 240,000 ticks read back exactly;
`video.view_ids` is exactly `["view0"]`; the counterfactual pair has identical
RGB/depth/semantic hashes from that view; only declared audio/source variables
change; no mouth motion is present. These criteria passed for the fixed
two-Beagle controlled-room M5 canary. They do not admit its assets, room,
audio, HRTF, episode, or dataset sample.

### M5.1: Mixed Real-Room and Legacy Comparison

Status: bounded research-review `pass` for the route, source/event,
legacy-Apartment delivery, and MP3D visual gates retained in
[M5_1_STATUS.md](M5_1_STATUS.md). The final MP3D dynamic-RIR and annotated
binaural listening-video delivery is also retained and independently bound to
the 14/14 visual/navmesh canary. This is a post-M5 research comparison, not a
change to immutable Timeline v2 and not a new admission gate. Reproduce the
retained scope with
[M5_1_EXECUTION.md](M5_1_EXECUTION.md).

Deliverables: one animated human and one animated dog; migrated 18-second,
270-frame legacy Apartment route/camera; a zero-radius center-point obstacle
gate; a real scanned-room review; synchronized main-view + right-side Topdown
media; and detailed per-source taxonomy, provenance, event windows,
per-source/pair/clip flags, and frame-current event state. The corrected pass
also freezes per-asset anatomical-forward declarations, binds both PBR actors
to the loaded room-light setup with HBAO, and renders listener-aware Topdown
QA with the visual HFOV wedge and `F`/`L`/`R` axes. Topdown has no audio-FOV or
distance-cutoff semantics.

Exit criteria: the old route and coordinate transform are hash-bound; both
actor centers avoid every declared gate obstacle on every frame; human/dog
semantic identities and mouth/muzzle anchors read back; the legacy comparison
video uses the same camera/path duration as the old AVEngine reference; human
local `+Z` and Beagle local `+X` anatomical forward align to the route tangent
on 270/270 frames per actor in both rooms; PBR and HBAO readbacks pass; both
actor creation calls record the same room-light key and the registered setup
reads back equal to the current room setup (the pinned binding exposes no
native per-actor key getter); the real-room canary executes without center-point
penetration; source/event/flag
JSON validates and reconstructs every per-frame active event; and all review
media is explicitly QA/research-only. The legacy route, mixed capture,
source/event/flag contract, dynamic binaural delivery, and old/new comparison
meet those bounded criteria. The real MP3D scan additionally passes 14/14
declared-navmesh visual gates with both actor centers navigable and both
semantic IDs visible for 270/270 frames. These center-only gates do not prove
full articulated-mesh clearance or full-body framing. The MP3D listening
delivery additionally passes exact 270-frame/288,000-sample media readback
with 90 dynamic two-source binaural RIR keyframes. Apartment/MP3D materials
remain `research_placeholder`/unqualified and dataset admission is false.
The Legacy room exposes three current/registered lights. MP3D exposes zero;
its scan illumination is baked, and HBAO is not dynamic-shadow evidence.

## M6: Feasibility Interfaces and Room Canary

Status: bounded `pass`. The controlled evidence, annotated tag, post-tag
verification and attestation are complete; see
[M6_FINAL_REPORT.md](../../release/M6_FINAL_REPORT.md),
[M6_STATUS.md](M6_STATUS.md), [M6_EXECUTION.md](M6_EXECUTION.md), and
[M6_ROOM_MATRIX.md](M6_ROOM_MATRIX.md). This does not promote any research room
or asset into dataset admission, and the native layers explicitly left outside
the controlled M6 run remain `not_run`.

Deliverables: one current cross-repository release-manifest authority; a
documented `trusted_research_workspace` path policy; Habitat-native bootstrap;
versioned entity, animal-template, source-endpoint, sound-asset, AudioProgram
and room contracts; structured OOD rejection; a stable adapter over the M5.1
source/event/flag authority; six qualification cases over four complementary
visual room lineages; an independent corrupted acoustic fixture; and one
controlled retained-evidence two-endpoint materialization in which only the
declared endpoint emits during its event windows.

The four visual room lineages have separate responsibilities:

- `blender_custom_two_zone_v1`: controlled geometry/material reference;
- ReplicaCAD `apt_0`: structured Habitat-native CAD provider;
- `legacy_ue_apartment_0000_v1`: migration continuity using real surfaces,
  with historical AABB data retained only for center-point diagnostics;
- MP3D `17DRP5sb8fy`: immutable raw scan plus a declared, versioned acoustic
  derivation whose identity, integrity, spatial parity, solver, topology,
  material and ray states remain independent.

MP3D raw and derived are two acoustic representations of the same visual room;
the corrupted fixture is the sixth case and is not a room.

Exit criteria: all M6 v2 Definition-of-Done items have executable evidence;
the controlled semantic materialization retains the verified 360°
binaural/timeline/per-source stem bytes without claiming a new native
Habitat/RLR run; M5.1 `present`/`absent`/`not_evaluated` and OR/AND clip
aggregation remain compatible; unavailable facts remain `not_evaluated`; every
room reports visual, navigation, acoustic geometry, material, ray, physical
truth, episode-feasibility and admission separately; the corrupted fixture
keeps admission false; fresh-checkout fast tests pass; unrun native/RLR/Blender
and media layers remain `not_run`; and the unique release manifest verifies
from a clean tagged cross-repository state.

M6 does **not** generate natural-language QA, freeze the final training-item
schema, train a model, qualify arbitrary animal species, batch-qualify rooms or
promote a historical research bundle by renaming its status. Git identifies
checked-in source/configuration; content hashes are retained only for external
result-changing inputs, generated closures and formal evidence needed by a
later read-only exporter.

## M6.x: Fixed SPEAR Apartment Source-Logic Canary

Status: bounded research-canary `pass`. S0--S5 executed in the fixed existing
Apartment with one co-located/co-oriented camera-listener rig, 360-degree
binaural mixtures, independent stems, Timeline/flags and clean plus diagnostic
Topdown videos.

Final evidence and claim boundary: [M6X_FINAL_REPORT.md](M6X_FINAL_REPORT.md).

Deliverables: one frozen Habitat-compatible SPEAR `apartment_0000`
`RoomCapsule`; stable registered human, animal, rigid-object and source-endpoint
insertion; reuse of the M5.1 source/event/flag authority; and a bounded
fixed-room suite covering routing, a rear source, a visible silent distractor,
a moving source, overlapping sources and qualified LOS/NLOS contrast. Review
outputs retain clean video, diagnostic Topdown, independent stems and 360-degree
binaural evidence where the runtime is available.

Exit criteria: the fixed room revision, visual/acoustic package,
camera-listener, anchors and trajectories remain reproducible across scenarios;
AudioPrograms activate zero, one or multiple named endpoints according to the
declared timeline; source identity, direction, event state and existing flags
remain consistent. Placement is deliberately source-center-only. The gate and
Topdown consume the same live obstacle snapshot: Apartment baked furniture is
represented by PathFinder, while separately loaded furnished scenes consume
every live rigid collision OBB. This result does not claim full-body collision,
material physical truth or dataset admission. Automatic furnishing,
natural-language QA and a large-scale dataset remain out of scope.

## M6.y: Optional SPEAR/UE Comparison Visuals

Status: `pass` as a bounded comparison-visual workstream. Native Apartment,
MP3D and ReplicaCAD runtime evidence all pass their declared gates. See
[M6Y_STATUS.md](M6Y_STATUS.md).

Purpose: test whether SPEAR/UE can provide an enhanced visual presentation for
the same constrained episode without becoming a second task engine.
Habitat-native AVEngine remains authoritative for Timeline or retained-route
state, navigation, source-center placement, source programs, binaural audio,
Topdown, flags and metadata. The optional UE backend has the fixed role
`comparison_visual` and may not replan actors or sources.

Current evidence covers native SPEAR Apartment S0/S3/S4, a fresh Habitat
Apartment S0--S5 natural-light run and the retained 270-frame MP3D/ReplicaCAD
M5.1 routes. MP3D now uses fresh-reloaded sRGB base-color views and separate
linear AO views; aggregate color retention and fixed exposure pass without
claiming recovered Matterport lights. ReplicaCAD keeps dataset lights 0/1/2
inside the room and disables the two strongest positive lights outside its open
stage shell; it moves/adds no lights. A real Habitat point-light comparison was
darker than the maintained `no_lights + HBAO` view and remains research-only.
The retained MP3D and ReplicaCAD roots still move only 1.1 m and 1.2 m over 18
seconds, so neither compatibility pass claims a normal-speed episode.

Exit criteria: each claimed room has real UE execution evidence and decodable
review media; actor roots, headings and animation phase agree with the
Habitat-native authority; visual QA remains room-specific and honest; and an
unavailable room is reported as pending or blocked rather than inferred from a
plan. Placement stays source-center-only. No new release-manifest, symlink or
leaf-hash maintenance layer is part of this workstream.

## M7: Benchmark and Paper Release

Deliverables: Dynamic Articulated Source Attribution task, splits, loaders,
baselines, ablations, metrics, release manifests and paper artifacts.

Exit criteria: visual/audio/audio-visual baselines and counterfactual/sync/anchor
ablations run on frozen splits; reused/extended/original claims and all required
citations are consistent across code, dataset card and paper.
