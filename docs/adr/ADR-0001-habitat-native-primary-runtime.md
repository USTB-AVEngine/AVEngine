# ADR-0001: Habitat-Native Primary Runtime

- Status: Superseded for production-room routing by
  [ADR-0010](ADR-0010-single-source-repository.md) and the canonical
  [room-family matrix](../architecture/OPTIONAL_RESIDENTIAL_SCENE_BACKENDS.md)
- Date: 2026-07-16

## Context

The legacy primary path split visual execution across UE/SPEAR and acoustic
execution across Habitat/RLR. Coordinate conversion, duplicated scene state
and independently advancing clocks made geometry, pose and AV parity hard to
prove.

## Decision

The 2026-07 decision was to use the pinned `habitat-sim-AVEngine` fork as the
primary visual, scene, sensor, articulated-pose and RLR runtime. Blender
remained an offline asset compiler and UE/SPEAR a comparison backend.

## Supersession note

This all-Habitat visual-routing decision no longer defines the production
surface. The active room-family matrix is:

- MP3D: Habitat-Sim scene execution, pixels, sensors and articulated pose;
  RLR uses SoundSpaces material authority on the same Habitat scene/state.
  Any MP3D UE import is `comparison_visual` only.
- `apartment_0000`: native UE/SPEAR production pixels; AVEngine owns Timeline,
  task/source state, navigation semantics, audio, Topdown, labels and
  admission.
- InteriorAgent/Kujiale: UE/SPEAR USD/MDL production pixels over the selected
  external scene; AVEngine retains the same episode authority.
- Skokloster: excluded unless explicitly reauthorized for a named task.

## Alternatives considered

- Continue the UE/SPEAR + Habitat/RLR split.
- Implement a new simulator from scratch.
- Keep UE primary and use Habitat only for benchmark parity.

## Consequences

One scene graph and world state can drive cameras, listeners, sources and
poses. Habitat visual quality and animal execution become explicit canary
risks. AVEngine must accurately attribute reused Habitat/RLR capabilities.

## Validation plan

Run room/visual, dog-pose, acoustic-material, multi-source and timeline
canaries in milestone order. Until they pass, Habitat-native remains planned,
not verified.

## Reversal criteria

Reconsider the primary runtime if a documented M1/M2 blocker prevents the
minimum benchmark observations after bounded fork changes. Preserve the data
contracts so another backend can implement them without reviving dual clocks.
