# ADR-0003: Explicit Acoustic Scene Package

- Status: Accepted
- Date: 2026-07-16

## Context

Visual PBR materials do not determine acoustic coefficients. The legacy path
also used AABB proxies and an implicit semantic-material path that could fail
to preserve openings, geometry or material activation.

## Decision

Compile an explicit versioned Acoustic Scene Package containing surface
geometry, transforms, unit convention, per-triangle material categories and
the RLR material database. Upload it through the modern RLR object/mesh API.

## Alternatives considered

- Infer acoustics from PBR textures.
- Depend only on Habitat semantic categories.
- Use AABBs as production geometry.

## Consequences

Material and geometry choices become auditable but require a compiler, mapping
confidence and parity QA. AABB remains debug-only.

## Validation plan

Require material coverage, exported debug mesh, ray-leakage checks and a
repeatable high- versus low-absorption RIR/EDT/DRR canary.

## Reversal criteria

The package representation may change with a versioned migration if RLR
requires another format; explicit geometry/material semantics may not be
removed.
