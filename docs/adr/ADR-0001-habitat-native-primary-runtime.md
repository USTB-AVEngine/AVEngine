# ADR-0001: Habitat-Native Primary Runtime

- Status: Accepted
- Date: 2026-07-16

## Context

The legacy primary path split visual execution across UE/SPEAR and acoustic
execution across Habitat/RLR. Coordinate conversion, duplicated scene state
and independently advancing clocks made geometry, pose and AV parity hard to
prove.

## Decision

Use the pinned `habitat-sim-AVEngine` fork as the primary visual, scene,
sensor, articulated-pose and RLR runtime. Blender remains an offline asset
compiler. UE/SPEAR remains an optional legacy comparison backend.

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
