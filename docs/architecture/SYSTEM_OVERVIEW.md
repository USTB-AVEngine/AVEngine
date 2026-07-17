# AVEngine System Overview

Status: the system includes the M1 visual/room baseline, bounded M2 articulated
animal runtime, and M3 explicit acoustic-scene/material-activation path.
Authoritative milestone outcomes are recorded in
[MILESTONES.md](../roadmap/MILESTONES.md). Physical room-material
qualification, formal multi-source propagation semantics, dataset registration
and end-to-end dataset claims remain later gates.

## Purpose

AVEngine is a Habitat-native dataset engine for synchronized,
identity-preserving, counterfactual articulated audio-visual source grounding.
It is an independent research extension built on a pinned Habitat-Sim fork and
RLR-Audio-Propagation, not a simulator implemented from scratch.

## System flow

```text
Dataset request
  -> AVEngine asset and room compilers
  -> canonical animal, room, acoustic-scene and episode packages
  -> habitat-sim-AVEngine deterministic runtime
  -> one formal view's co-located RGB/depth/semantic frames + per-source/listener-pair RIRs
  -> AVEngine dry-audio/stem/mix assembly, QA, provenance and registry admission
```

Offline Blender tools may compile assets, but the official episode clock and
runtime observations are owned by the Habitat-native path.
Profile-bound motion retargeting is one such offline compiler and is never a
runtime fallback; see [MOTION_RETARGETING.md](MOTION_RETARGETING.md).

## Capability ownership

| Capability | Owner | AVEngine claim |
|---|---|---|
| Scene graph, GLB loading, PBR rendering, sensors, physics, navigation and articulated-object foundations | Habitat-Sim | Reused |
| Geometric acoustic propagation and modern multi-source/listener C API | RLR / SoundSpaces 2.0 | Reused algorithm/API |
| Deterministic non-human pose playback, strict RLR context lifecycle and explicit acoustic package ingestion/readback | Habitat runtime fork | Runtime extension; not a new propagation solver |
| Single-view same-state multimodal capture profiles, audited animal/room/acoustic compilation, source identity, authoritative timeline, counterfactuals, QA, provenance and registry | AVEngine main repository | System contribution over stable Habitat/RLR APIs |

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
pair-specific named multi-source/listener semantics remain the M4 gate.

## M3 acoustic boundary

M3 compiles a source room, reviewed canonical transform, exact visual-slot to
acoustic-category mapping and versioned material database into canonical
surface arrays and an RLR database. Source replay must independently reproduce
the emitted geometry, object partitions, per-triangle material IDs and
resolved materials. Production geometry cannot be an AABB proxy and every
triangle must be assigned without fallback.

RLR supplies ray tracing and impulse-response synthesis. The Habitat fork
provides a strict modern context/ingestion adapter and native readback;
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
material replay. Named multi-source all-pair IRs, stems, order invariance,
reset/temporal policy and performance remain M4 work.

## Versioned contracts

The system exchanges five package families:

1. Canonical Animal Asset Package.
2. Room Package.
3. Acoustic Scene Package.
4. Authoritative Timeline and Episode Package.
5. QA, provenance and registry records.

Every package records its schema version and content hashes. Runtime and sample
manifests record the AVEngine, Habitat fork, upstream Habitat, RLR, scene,
asset and schema revisions used to produce the result.

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
