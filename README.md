# AVEngine — Habitat-native Audiovisual Research Toolkit

AVEngine is a Habitat-native research toolkit being prepared for open-source
release. It generates deterministic, identity-preserving audiovisual episodes
from explicit assets, rooms, source programs and evidence contracts.
Habitat-Sim is the primary visual/scene/sensor/physics runtime; RLR Audio
Propagation is the geometric acoustic foundation. AVEngine is not a simulator
built from scratch.

## Current release authority

The only cross-repository release authority is
[`release/avengine_release_manifest_v1.json`](release/avengine_release_manifest_v1.json).
If that file is absent or fails verification, the current state is **pending**.
If it is present, its release state, implementation commit, evidence and tag
must be verified rather than inferred from README prose. A schema or verifier
without a valid manifest/tag is not a release.

Root [`runtime.lock.yaml`](runtime.lock.yaml) is only a lightweight index into
the exact M1--M4 compatibility profiles under [`locks/`](locks/). The profiles
are immutable historical inputs; the index contains no repeated test status or
artifact hashes. Neither overrides the current release manifest.

Hashes are evidence identities, not runtime feature locks. Git commits and tags
identify checked-in source, schemas and configuration; explicit versions record
the compiler, Python and dependency environment. Content hashes are retained
only at trust boundaries where a same-looking path could otherwise name
different result-changing bytes: external room/model/action/audio assets,
generated package closures, native binaries used by formal evidence, test
receipts and release bundles. Human-facing documentation refers to one package
or bundle identity instead of repeating every leaf hash. Transient previews,
logs, local paths and uncited intermediates are not release gates. A missing
optional input is `blocked` or `not_run`; a byte mismatch is `fail` only when
that input was explicitly declared authoritative for the attempted result.

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

The post-gate **M5.1 mixed real-room research review** is a bounded
research-review `pass` for its completed route, source/event, legacy-Apartment
delivery, and MP3D visual gates. It runs one animated Rocketbox human and one
animated Beagle for 18 seconds/270 frames, preserves the old Apartment
route/camera, validates center-only obstacle/navigation constraints, and adds
detailed tri-state source/event/flag JSON. A corrected retained pass binds the
human's local anatomical `+Z` and the Beagle's local `+X` to the route tangent
on 270/270 frames per actor in both rooms; it supersedes the earlier
backward-human/sideways-Beagle review media. Both actors read back PBR, use the
loaded room's common light setup, and render with HBAO enabled. The retained
Apartment output includes a two-channel binaural annotated Habitat + Topdown
video and an old UE | new Habitat | Topdown comparison. MP3D passes a real declared-navmesh
14/14 visual gate and retains its own 18-second annotated binaural listening
video. A separate supplementary UE/SPEAR visual canary imports the same raw
MP3D source, reuses the exact Habitat camera and Pathfinder-qualified
actor-center routes, and emits a 1920x480 UE main | Habitat main | Habitat
Topdown triptych. Its two-channel binaural AAC packets are copied unchanged
from the Habitat review, so it is not a UE acoustic comparison. UE adds an
explicit shadow-casting movable directional light and a bounded skylight for
this review, not a reconstruction of the unknown Matterport capture lights.
Both Topdown panels use the shared Habitat camera/listener basis and show a
visual HFOV wedge plus forward/left-ear/right-ear axes; audio has no camera-FOV
or distance cutoff. Its RLR-only research package removes exactly one wholly
degenerate scan primitive (458 zero-area triangles) without changing the
visual room;
the remaining geometry/material QA states stay explicit. See
[`M5_1_STATUS.md`](docs/roadmap/M5_1_STATUS.md). All M5.1 rooms/materials and
media remain unqualified research evidence; dataset admission is false.

**M6 feasibility/interfaces/room-canary is closed.** Its scope was to add
versioned entity, animal,
source, sound, AudioProgram and room contracts; preserve the M5.1 tri-state
source/event/flag authority through adapters; and attempt four complementary
room qualifications plus an independent corrupted fixture. M6 does not freeze
a final dataset-item schema, generate natural-language QA, train a model or
claim broad animal/room coverage. See
[`M6_STATUS.md`](docs/roadmap/M6_STATUS.md),
[`M6_EXECUTION.md`](docs/roadmap/M6_EXECUTION.md), and
[`M6_ROOM_MATRIX.md`](docs/roadmap/M6_ROOM_MATRIX.md). The final closeout is in
[`M6_FINAL_REPORT.md`](release/M6_FINAL_REPORT.md). The current bounded M6 code,
controlled canary, tagged release verification and attestation passed; this
does not promote the research rooms or assets into a dataset.

M6 defines six qualification cases over four visual room lineages, not six
different rooms: controlled Blender, ReplicaCAD, Legacy Apartment and MP3D.
MP3D raw and derived are two acoustic representations of the same visual room;
the independent corrupted fixture is not a room. The A3 controlled runner
implements a verified retained-evidence materialization. Its `pass` is scoped
to `semantic_materialization_verifier`; it does not mean Habitat-Sim or RLR
Audio Propagation ran again. The formal post-A3 bundle, annotated tag,
post-tag verifier and attestation are complete. The M6-native Habitat, native
RLR propagation and native episode-feasibility layers remain `not_run` by that
milestone's declared scope.

A separate development review now exercises ReplicaCAD `apt_0` with a real
270-frame Habitat capture, PathFinder Topdown and a 90-keyframe two-source RLR
binaural render. The v2 route passes 19/19 visual, navmesh, LOS, semantic and
furnished rigid-object root-center gates. Both dry buses and both binaural
stems are non-silent in every declared event window. Acoustic geometry remains
stage-surface-only, while topology, material truth, ray qualification and the
Beagle dry-audio rights item remain unqualified. The six-case review builder
keeps those facts visible; its post-hoc verifier reopens the request,
source-media bindings, all segments and the combined media without promoting
review media into dataset admission or a formal M6 native release layer.

The AudioProgram contract vocabulary is `one_active_of_n`,
`simultaneous_subset`, `sequential_sources`, `intermittent_events`,
`counterfactual_route_swap`, and `silent_negative`. M6 materializes and verifies
the retained `one_active_of_n` controlled evidence; it does not claim a new
native Habitat/RLR execution. The other modes are versioned extension
contracts, not claims of six completed executable canaries.

The **M6.x fixed SPEAR Apartment source-logic canary** reuses the existing
Habitat-compatible `apartment_0000` package without copying or rearranging its
furniture. It implements a compact RoomCapsule, anchor library, 270-frame
human/Beagle master route and executable S0--S5 programs for routing, front/rear
counterfactuals, visible silence, moving sound, overlap and LOS/NLOS. Placement
checks only the source center. Apartment baked furniture is represented by the
live PathFinder navmesh; separately loaded furnished scenes such as ReplicaCAD
add every live rigid collision OBB. The same runtime obstacle snapshot drives
both the gate and the diagnostic Topdown, so furniture cannot disappear from
the map while still affecting placement. Audio remains 360 degrees and is
never gated by the camera HFOV. Automatic furnishing, natural-language QA and
large-scale dataset generation remain outside this canary.

The follow-on **M6.y optional SPEAR/UE comparison-visual workstream** keeps the
current Habitat-native AVEngine as the sole authority for actor state,
navigation, source centers, source logic, audio, Topdown, flags and metadata.
SPEAR/UE may render native UE pixels but may not replan the episode. Real native
Apartment S0/S3/S4 renders and a fixed-exposure 270-frame MP3D compatibility
render pass their declared runtime/media gates. ReplicaCAD `apt_0` also passes
real editor import/reload and a 270-frame UE runtime: all 171 tagged scene mesh
actors and five positive dataset lights are read back, with no added review
light. The retained MP3D and ReplicaCAD routes move each root only 1.1 m and
1.2 m respectively over 18 seconds; both remain slow compatibility routes, not
normal-speed results. These are bounded visual comparisons, not a second
engine, material qualification or dataset admission. See
[`M6Y_STATUS.md`](docs/roadmap/M6Y_STATUS.md) for current videos, claim
boundaries and the local review-page command.

### Run the fixed Apartment S0--S5 canary

Use the Habitat/RLR Conda environment; `.venv` is not required.

The review profile now captures the native RGB/depth/semantic rig at
`1280x720`. Clean video stays at that resolution and is encoded as H.264
CRF 18; only the diagnostic main-view panel is downscaled and letterboxed to
`640x480` beside Topdown. The compositor refuses to upscale a low-resolution
capture. The profile also installs a Habitat directional key/fill setup and a
transient exterior proxy made from UE's stock `approaching_storm_4k` HDRI.
Because the exported Apartment glass writes depth as a black surface, the proxy
contains a distant inward sphere plus two finely subdivided, room-aligned
panels immediately behind the visible window frames. Each panel vertex samples
the 4096x2048 equirectangular image from the real listener-to-window direction;
there is no hand-authored rectangular crop or stretched panorama. This is a
fixed-camera review approximation, not a claim that Habitat reproduces UE
HDRIBackdrop, exposure, Lumen, baked lighting, or reflection captures. A moving
camera would require a real skybox/custom shader rather than these panels.

The two legacy red/blue icosphere source markers are used only by the separate
native anchor/LOS qualification. They are removed from the RGB/semantic capture
simulator, so clean videos contain no debugging ball. Their logical endpoints
remain available in audio routing, Timeline metadata and Topdown diagnostics.

Articulated actors use `idle` while their authored root is stationary and
`walk` only while it moves. The action clock resets at each transition, and a
retained capture is accepted only when its positions, rotations, heading and
locomotion records still match the current authored route. The S3 human route
is a deterministic, live-navmesh-qualified 4.324 m polyline at
`0.861--0.881 m/s`, replacing the earlier `0.121 m/s` drift. The Beagle master
route uses `0.296 m/s`, the asset-specific speed selected by its M2 world-contact
fit, rather than a species-wide hard-coded speed.

Prepare the real exterior proxy once. The command fails if the UE asset cannot
be exported; it never substitutes a synthetic sky:

```bash
python tools/m6x/prepare_spear_apartment_exterior.py \
  --ue-root /path/to/UE_5.5 \
  --uproject /path/to/SpearSim.uproject \
  --blender /path/to/blender
```

An already exported HDRI can instead be supplied with `--retained-hdri`
without a UE installation. That mode is recorded as user-supplied input; it
does not claim to independently prove the file's Unreal provenance.

The sphere and direction-projected panels exist only in the visual capture simulator. They are
non-collidable, use semantic background ID 0, and render as distant depth
surfaces; they are not added to the room SceneInstance, navmesh, Topdown
obstacle map, placement gate, or RLR acoustic geometry. Runtime evidence checks
the final scene and articulated-actor light keys after actor-light binding.

The command expects the existing local M1 Apartment package, M2 Beagle runtime
records, M3 acoustic package, the supplied human GLB and Beagle dry audio. It
also needs `ffmpeg`/`ffprobe` and the default HRTF at
`/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa` (or pass `--hrtf`). These
inputs already exist in the project workspace used for the closeout run. The
output directory must be new; the runner deliberately refuses to overwrite an
earlier review bundle.

```bash
cd /data/jzy/code/AVEngine-habitat-native
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export SKBUILD_EDITABLE_SKIP=1
export PYTHONPATH="$PWD/src:$PWD"

python tools/m6x/build_fixed_apartment_canary.py \
  --runtime-root /data/jzy/code/habitat-sim-AVEngine \
  --human-runtime-glb /data/jzy/code/AVEngine/external/SPEAR/tmp/rocketbox_native_runtime_ue_v3/rocketbox_male_adult_01_original_ue_v3/runtime.glb \
  --beagle-audio /data/jzy/code/AVEngine/external/SPEAR/tmp/animal_audio_event_audit_v1/dog_beagle_v2_scheduled_dry.wav \
  --output tmp/m6x/fixed_apartment_run_01
```

Open `tmp/m6x/fixed_apartment_run_01/REVIEW_INDEX.html` to review every clean
binaural video, annotated main-view + Topdown video, mixture WAV and independent
source stem. Each scenario variant also retains its AudioProgram, Timeline v2,
source manifest, legacy flags and final status under `metadata/`.
At bundle root, `inputs/input_index.json` records the small configuration
snapshot, code commits and direct external assets; `FINAL_REPORT.md` states the
bounded acoustic and placement claims.
The refreshed local closeout run is retained at
`tmp/m6x/fixed_apartment_canary_20260720_02/REVIEW_INDEX.html`.
The bounded feasibility result and its acoustic claim boundary are summarized
in [the M6.x final report](docs/roadmap/M6X_FINAL_REPORT.md).

The earlier `_04` closeout capture is a historical `320x240` baseline, and
`_06` predates the hidden test markers, direction-projected exterior and normal
S3 route. Neither satisfies the current visual/trajectory readback. A fresh run
is therefore required once. After that, scenario media and metadata can be
rebuilt from the new capture and native RLR result:

```bash
BUNDLE=tmp/m6x/fixed_apartment_720p_run_01
python tools/m6x/build_fixed_apartment_canary.py \
  --runtime-root /data/jzy/code/habitat-sim-AVEngine \
  --human-runtime-glb /data/jzy/code/AVEngine/external/SPEAR/tmp/rocketbox_native_runtime_ue_v3/rocketbox_male_adult_01_original_ue_v3/runtime.glb \
  --beagle-audio /data/jzy/code/AVEngine/external/SPEAR/tmp/animal_audio_event_audit_v1/dog_beagle_v2_scheduled_dry.wav \
  --capture-dir "$BUNDLE/shared/master_capture" \
  --acoustics-dir "$BUNDLE/shared/acoustics" \
  --output tmp/m6x/fixed_apartment_rebuild_01
```

### Rebuild the ReplicaCAD furniture-aware review

This command reuses the retained 18-second RGB capture and binaural mixture. It
briefly loads the real `apt_0` room, reads all 113 rigid furniture collision
OBBs, applies the source-center-only gate and rebuilds the diagnostic Topdown;
it does not rerun visual capture or RLR acoustics. Use the same Conda environment
shown above and choose a new output directory:

```bash
python tools/m6x/rebuild_replicacad_obstacle_review.py \
  --replicacad-root tmp/m6x/datasets/replica_cad \
  --capture-dir tmp/m5_1/replicacad_mixed_20260719_04 \
  --delivery-dir tmp/m5_1/replicacad_delivery_20260719_03 \
  --output tmp/m6x/replicacad_obstacle_review_run_01
```

The main outputs are
`videos/replicacad_runtime_obstacles_diagnostic.mp4`,
`room/runtime_obstacle_map.png`, `source_center_gate.json` and `status.json`
below the chosen output directory. ReplicaCAD's six room articulated objects
do not expose rigid-equivalent collision OBBs through the runtime API; they
remain represented by the declared navmesh and are reported separately rather
than being approximated as fake collision boxes.

The reviewed `apt_0` snapshot classifies the 113 rigid objects as 39
ground-level blockers, 71 elevated objects and 3 walkable rugs/mats, with no
unknown objects. Rugs remain visible in teal on Topdown but do not fail the
source-center gate; elevated furniture remains visible in blue and is checked
with its actual 3-D OBB rather than being flattened into a floor blocker.

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
13. [`docs/roadmap/M5_1_STATUS.md`](docs/roadmap/M5_1_STATUS.md) and
    [`docs/roadmap/M5_1_EXECUTION.md`](docs/roadmap/M5_1_EXECUTION.md) — mixed
    human/Beagle real-room review, legacy comparison, source/event/flag
    contract and exact claim boundary.
14. [`docs/architecture/LEGACY_SOURCE_EVENT_FLAG_AUTHORITY.md`](docs/architecture/LEGACY_SOURCE_EVENT_FLAG_AUTHORITY.md)
    — the preserved M5.1 flag definitions, thresholds, three-state values and
    clip aggregation authority.
15. [`docs/roadmap/M6_STATUS.md`](docs/roadmap/M6_STATUS.md),
    [`docs/roadmap/M6_EXECUTION.md`](docs/roadmap/M6_EXECUTION.md), and
    [`docs/roadmap/M6_ROOM_MATRIX.md`](docs/roadmap/M6_ROOM_MATRIX.md) — M6
    interface, execution, room-qualification and claim boundaries.
16. [`docs/roadmap/M6X_FINAL_REPORT.md`](docs/roadmap/M6X_FINAL_REPORT.md) and
    [`docs/roadmap/M6Y_STATUS.md`](docs/roadmap/M6Y_STATUS.md) — the fixed
    Apartment Habitat canary and optional SPEAR/UE comparison-visual status.
17. [`docs/security/FILESYSTEM_TRUST_MODEL.md`](docs/security/FILESYSTEM_TRUST_MODEL.md)
    — the declared `trusted_research_workspace` path and publication model.
18. [`docs/migration/LEGACY_AVENGINE_INVENTORY.md`](docs/migration/LEGACY_AVENGINE_INVENTORY.md)
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
| M5.1 | corrected anatomical heading, room-bound PBR/HBAO, mixed human/Beagle real-room and legacy 18-second comparison, same-room MP3D UE/Habitat visual triptych, listener-basis Topdown and detailed source/event/flag metadata (`pass`, bounded research review; no dataset admission) |
| M6 | feasibility foundation: extensible registries, legacy-compatible flags, room interfaces, retained controlled-source materialization and one independent fail-closed fixture (`pass`; no dataset admission claim) |
| M6.x | fixed SPEAR `apartment_0000` Habitat RoomCapsule, runtime-authoritative obstacles and executable source-logic S0--S5 binaural/Topdown canary (`pass`, bounded research canary; source-center placement only) |
| M6.y | optional SPEAR/UE `comparison_visual`: native Apartment S0/S3/S4, fixed-exposure MP3D and imported ReplicaCAD runtime `pass` (bounded visual comparison; Habitat-native protocol/audio authority unchanged) |
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
light must be added and calibrated against the baked appearance. M5.1 copies
the loaded room setup to one actor-light key, binds both PBR actors to it and
enables HBAO; MP3D correctly reads back zero room lights. HBAO is screen-space
ambient occlusion, not a dynamic shadow map or evidence of UE-quality shadows.

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

Detailed formal measurements and the single authoritative profile/evidence
locations are recorded in [`M3_STATUS.md`](docs/roadmap/M3_STATUS.md); leaf
hashes remain inside those machine-readable bundles rather than this README.

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

M5.1 preserves the old 18-second Apartment route and adds one animated human
plus one animated Beagle, exact center-point gates, actual animated emitter
link trajectories, detailed source/event/flag records, dynamic binaural review
audio, and annotated main-view + Topdown QA media. Human local `+Z` and Beagle
local `+X` anatomical-forward declarations now align to movement on every one
of 270 frames per actor in each retained room; the earlier backward/sideways
captures are superseded. Both actors are PBR, share the loaded room-light
setup and read back HBAO enabled. The Legacy setup has three lights, while the
MP3D scan has zero runtime lights and retains baked scan illumination. Its
Topdown visual uses the shared Habitat `world_from_local` listener basis,
visual-HFOV wedge and `F`/`L`/`R` axes; no audio FOV/distance cutoff is implied.
Its ordinary MP4 contains two-channel binaural audio; Topdown remains QA-only
and any four-channel FOA
authority remains a separate WAV. The legacy room's acoustic package retains
its real `fail`/`not_run` QA reports and `research_placeholder` material
semantics. The real MP3D gate qualifies actor root centers on the declared
navmesh only, not the full articulated meshes or complete-body framing. The
MP3D listening review uses a separately hash-bound research package that
removes only RLR-rejected zero-area faces; source-parity and geometry remain
`fail`, material coverage remains `pass`, and ray leakage remains `not_run`.
The separately retained UE/SPEAR comparison uses the same raw MP3D source,
camera and 270-point actor-center routes to render an 18-second UE main |
Habitat main | Habitat Topdown triptych. UE scene collision is disabled, so
Habitat Pathfinder remains the center-only navigation authority. The triptych
copies the Habitat binaural audio stream unchanged, and its explicit UE
directional light and bounded skylight support visible shadow review without
claiming full-body clearance, acoustic or light parity, Matterport-light
reconstruction, or dataset admission.
See
[`M5_1_STATUS.md`](docs/roadmap/M5_1_STATUS.md) and
[`M5_1_EXECUTION.md`](docs/roadmap/M5_1_EXECUTION.md).

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

AVEngine is being organized for an open-source source release, but the current
[`LICENSE`](LICENSE) remains all-rights-reserved until the project owner selects
and commits an explicit open-source license. Source visibility does not itself
grant redistribution rights. Dataset, room, model, sound and RLR rights remain
separate even after the code license is selected.

## Contact

Ziyang Ji ([Eastforward](https://github.com/Eastforward)) — research
collaboration welcome; request permission before reuse or redistribution.
