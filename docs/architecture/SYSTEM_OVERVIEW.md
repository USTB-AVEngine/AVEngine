# AVEngine System Overview

Status: the system includes the M1 visual/room baseline, bounded M2 articulated
animal runtime, M3 explicit acoustic-scene/material-activation path and the M4
named multi-source implementation. M4's bounded software/source-pose gate is
`pass`; see [M4_STATUS.md](../roadmap/M4_STATUS.md).
Authoritative milestone outcomes are recorded in
[MILESTONES.md](../roadmap/MILESTONES.md). Physical room-material
qualification, event-time dynamic-anchor qualification, dataset registration
and end-to-end dataset claims remain later gates.

The single-source architecture in
[REPOSITORY_BOUNDARIES.md](REPOSITORY_BOUNDARIES.md) is an accepted migration
target, not a completed-state claim. Native execution still uses the pinned
Habitat fork and maintained SPEAR checkout until the selected source is
integrated and checked.

## Purpose

AVEngine is a room-routed dataset engine for synchronized, identity-preserving,
counterfactual articulated audio-visual source grounding. MP3D uses
Habitat-Sim production visual execution; Apartment and Kujiale use UE/SPEAR
production visual execution. RLR-Audio-Propagation provides the acoustic
foundation. AVEngine is not a simulator or propagation solver implemented from
scratch.

## System flow

```text
Dataset request
  -> AVEngine asset and room compilers
  -> canonical animal, room, acoustic-scene and episode packages
  -> room-selected Habitat-Sim or UE/SPEAR visual execution + RLR propagation
  -> one formal view's co-located RGB/depth/semantic frames + per-source/listener-pair RIRs
  -> AVEngine dry-audio/stem/mix assembly, QA, provenance and registry admission
```

Offline Blender tools may compile assets, but the official episode clock,
source state, audio, labels and admission are owned by AVEngine across all
production visual routes.
Profile-bound motion retargeting is one such offline compiler and is never a
runtime fallback; see [MOTION_RETARGETING.md](MOTION_RETARGETING.md).

## Capability ownership

| Capability | Owner | AVEngine claim |
|---|---|---|
| MP3D scene graph, GLB loading, PBR rendering, sensors, physics, navigation and articulated-object foundations | Habitat-Sim | Reused |
| Apartment and Kujiale production visual execution | SPEAR client/plugin over an external Unreal Engine installation | Reused and adapted; UE and room data remain outside Git |
| Geometric acoustic propagation and modern multi-source/listener C API | RLR / SoundSpaces 2.0 | Reused algorithm/API |
| Deterministic non-human pose playback, strict RLR context lifecycle, explicit acoustic package ingestion/readback and named endpoint/native-receipt adapter | AVEngine Habitat/RLR integration layer | Runtime extension; currently sourced from the transition fork, not a new propagation solver |
| Single-view same-state multimodal capture profiles, audited animal/room/acoustic compilation, named source identity, FOA/binaural stem and mixture assembly, authoritative timeline, counterfactuals, QA, provenance and registry | AVEngine repository | System contribution over stable Habitat/RLR and room-specific visual APIs |

## Initial scientific scope

- Audited dog templates with baked Walk and Idle actions.
- Static or quasi-static acoustic geometry with dynamic semantic point emitters.
- Explicit per-triangle acoustic material categories.
- One formal `view0` with co-located RGB/depth/semantic sensors and one
  co-located listener; at least two uniquely named sources retain
  pairwise-distinct world transforms.
- Named sources and the single MVP listener with runtime pair-specific IRs and
  AVEngine-assembled stems.
- Five-second episodes at 15 fps and 16 kHz under a 48 kHz integer clock.
- Counterfactual pairs with identical visual state and controlled audio/source changes.
- Visual mouth articulation disabled to prevent shortcut learning.

## Non-goals

- A new visual renderer or acoustic propagation solver.
- Online arbitrary-mesh rigging or retargeting inside Habitat.
- Fully dynamic deformable-body acoustic reflection.
- Lip sync, visemes or visual mouth animation.
- Automatic inference of acoustic truth from visual PBR materials.
- Production approval based only on successful file generation.
- Multiple formal camera viewpoints in the first MVP. Top-down navigation maps
  are diagnostic QA artifacts, not dataset observations.

## M1 runtime boundary

M1 binds both manifest declarations and Habitat's actual loaded scene graph:
dataset/scene/stage selection and render/collision/semantic assets must resolve
to the exact declared files; handle-based scenes additionally bind live source
objects and lighting. All three room canaries explicitly load their
`load_declared` navmesh, and the active Pathfinder must match a separate load
of that file by full settings and vertex/index fingerprints.

The co-located M1 listener is only a pose anchor. M1 does not instantiate an
AudioSensor or execute RLR. M3 uses one controlled source/listener pair only
to prove explicit scene ingestion and synthetic material activation;
pair-specific named multi-source/listener semantics are the separate M4 gate.

## M3 acoustic boundary

M3 compiles a source room, reviewed canonical transform, exact visual-slot to
acoustic-category mapping and versioned material database into canonical
surface arrays and an RLR database. Source replay must independently reproduce
the emitted geometry, object partitions, per-triangle material IDs and
resolved materials. Production geometry cannot be an AABB proxy and every
triangle must be assigned without fallback.

RLR supplies ray tracing and impulse-response synthesis. The AVEngine
Habitat/RLR integration layer provides a strict modern context/ingestion
adapter and native readback;
AVEngine provides the explicit compiler, adapter inputs and evidence verifier.
See [ACOUSTIC_SCENE_AND_MATERIALS.md](ACOUSTIC_SCENE_AND_MATERIALS.md).

The controlled custom-room low/high databases are synthetic absorption
extremes. They test whether material selection changes RLR output repeatably;
they do not claim reviewed physical room coefficients. MP3D and UE visual-slot
proposals remain unqualified `research_candidate` diagnostics. Formal M3
measurements and hashes are recorded in
[M3_STATUS.md](../roadmap/M3_STATUS.md).

Native ingestion evidence combines exact API receipts, resolved material
blocks and post-ingestion OBJ geometry readback. The OBJ cannot expose a
recoverable per-face material-ID array, so it never replaces source-to-package
material replay. M4 consumes this verified package without weakening its M3
geometry/material contract.

## M4 named spatial-audio boundary

M4 realizes at least two bytewise-canonical stable source IDs and exactly one
listener in each output-layout context. The listener is the same pose as the
formal M1 camera rig. Native registration receipts close the requested and
realized source/listener IDs, indices, transforms, radii, orientation, layout,
channel count and explicit HRTF path. Every listener/source pair returns an
owned, independently readable IR addressed by its stable IDs rather than by an
unstable caller list position.

The authority output is raw RLR first-order Ambisonics:

```text
channels: [W, Y, Z, X]
indices:  ACN [0, 1, 2, 3]
normalization: N3D
coordinates: right-handed avengine_world
axes: +X right, +Y up, +Z back, -Z forward
```

Each source's mono dry signal is linearly convolved with its pair IR to retain
an independent four-channel FOA stem. Canonical source-ID summation produces a
four-channel full-tail canary mixture without implicit resampling,
normalization, limiting or cropping.

For direct listening, M4 separately asks RLR for `[left, right]` native
binaural pair IRs using the explicit MIT KEMAR normal-pinna SOFA asset, then
retains the independent binaural stems and canonical two-channel canary
mixture. Audio renders at 16 kHz. The pinned HRTF input is 44.1 kHz; adaptation
is permitted only inside the exact RLR binary authenticated by the M4 runtime
lock. AVEngine itself performs no hidden resampling. Six-cardinal FOA probes,
listener-rotation invariance and horizontal binaural probes freeze the spatial
interpretation instead of trusting channel names alone.

M4 also proves exact mapped output under reversed caller source order, stable-ID
source update, temporal-coherence execution, reset/reload reproduction and
one-source versus multi-source performance measurement. See
[M4_EXECUTION.md](../roadmap/M4_EXECUTION.md).

M4 outputs full-tail WAV evidence only. It does not produce a synchronized
episode or mux FOA/binaural audio into MP4. M5 owns exact 75-frame/80,000-sample/
240,000-tick assembly, visual-invariant counterfactuals, final tail/crop policy
and two-channel binaural video mux/readback.

The current source identity fixture is grounded at formal M1 source poses, but
its M2 event-time dynamic-anchor evidence is explicitly `not_run`. A bounded
M4 pass therefore validates software routing and static source-pose acoustics;
it does not admit an animal asset, physical room profile, episode or dataset.

## Versioned contracts

The system exchanges five package families:

1. Canonical Animal Asset Package.
2. Room Package.
3. Acoustic Scene Package.
4. Authoritative Timeline and Episode Package.
5. QA, provenance and registry records.

Every package records its schema version and the existing content hashes and
result-changing identities required by its evidence rules. Historical runtime
and sample
manifests record the AVEngine, Habitat fork, upstream Habitat, RLR, scene, asset
and schema revisions that produced them. After source cutover, checked-in
product code is identified by the AVEngine commit and upstream origin is
reported through the adaptation record.

Timeline v2 remains structurally extensible to more than one `view_id`, but
M1, M2, M5 and the initial M6 MVP semantically require exactly `["view0"]`.
See [ADR-0009](../adr/ADR-0009-single-view-multimodal-sensor-rig.md).

## Implementation order

```text
M0 repository/baseline
-> M1 room and visual canary
-> M2 articulated dog runtime
-> M3 acoustic scene/material activation
-> M4 multi-source RLR
-> M5 timeline/counterfactual
-> M6 dataset registry MVP
-> M7 benchmark and paper release
```
