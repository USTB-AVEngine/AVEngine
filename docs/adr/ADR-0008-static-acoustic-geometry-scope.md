# ADR-0008: Static Acoustic Geometry Initial Scope

- Status: Accepted
- Date: 2026-07-16

## Context

Per-frame deformable animal geometry in acoustic propagation adds major cost
and validation complexity. The initial research question requires moving
semantic emitters, not dynamic-body acoustic reflection.

## Decision

Use static or quasi-static room acoustic geometry with dynamic named point
emitters anchored to semantic bones. Animal bodies do not update the RLR scene
mesh every frame.

## Alternatives considered

- Fully dynamic deformable acoustic bodies from the first release.
- Freeze both geometry and emitter positions.
- Approximate every dynamic body with an AABB.

## Consequences

The MVP remains computationally and scientifically tractable. Claims must state
the scope accurately; dynamic occlusion by animal deformation is not modeled.

## Validation plan

Verify exact semantic emitter trajectories against timeline poses and measure
RLR temporal behavior for moving sources in fixed rooms.

## Reversal criteria

Introduce dynamic acoustic geometry only through a new ADR with update-cost,
stability and physical-validity evidence. AABB bodies cannot become a silent
production substitute.
