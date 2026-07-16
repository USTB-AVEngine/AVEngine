# AVEngine System Overview

Status: M0 architecture baseline. This document defines intended boundaries;
it does not claim that the Habitat-native canaries have passed.

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
  -> RGB/depth/semantic frames + per-source/listener-pair RIRs
  -> AVEngine dry-audio/stem/mix assembly, QA, provenance and registry admission
```

Offline Blender tools may compile assets, but the official episode clock and
runtime observations are owned by the Habitat-native path.

## Capability ownership

| Capability | Owner | AVEngine claim |
|---|---|---|
| Scene graph, GLB loading, PBR rendering, sensors, physics, navigation and articulated-object foundations | Habitat-Sim | Reused |
| Geometric acoustic propagation and modern multi-source/listener C API | RLR / SoundSpaces 2.0 | Reused |
| Deterministic non-human pose playback, explicit acoustic package ingestion and one-state multi-sensor capture | Habitat runtime fork | Runtime extension |
| Audited animal/room compilation, source identity, authoritative timeline, counterfactuals, QA, provenance and registry | AVEngine main repository | System contribution |

## Initial scientific scope

- Audited dog templates with baked Walk and Idle actions.
- Static or quasi-static acoustic geometry with dynamic semantic point emitters.
- Explicit per-triangle acoustic material categories.
- Named sources/listeners with runtime pair-specific IRs and AVEngine-assembled stems.
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
