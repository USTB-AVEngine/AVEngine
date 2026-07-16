# ADR-0007: Modern RLR C API

- Status: Accepted
- Date: 2026-07-16

## Context

Habitat's current AudioSensor uses the deprecated RLR C++ `Simulator` wrapper
and one source/listener abstraction. The pinned RLR header already exposes a
modern context API with multiple sources/listeners, object meshes and per-pair
IR access.

## Decision

Implement an isolated Habitat adapter over the modern `RLRA_*` C API. AVEngine
adds stable identities, event/actor/anchor mapping and independent stems; it
does not claim invention of multi-source propagation.

## Alternatives considered

- Extend the deprecated single-source C++ wrapper.
- Run one RLR context per source.
- Implement a new acoustic solver.

## Consequences

The fork must manage context lifetime, reset, temporal coherence, material
upload and source/listener ordering explicitly. ABI/build compatibility is an
M4 risk.

## Validation plan

Test named identity, per-pair IR shape, independent stems, reset behavior,
source-order invariance and performance with at least two sources.

## Reversal criteria

If upstream introduces an equivalent maintained adapter, AVEngine may adopt it
after parity tests while preserving the high-level identity contract.
