# ADR-0002: Two-Repository Boundary

- Status: Accepted
- Date: 2026-07-16

## Context

Habitat is an upstream project with its own history and MIT license, while
AVEngine owns dataset compilation and governance. Combining them would obscure
attribution and make upstream synchronization difficult.

## Decision

Maintain exactly two required repositories: the Habitat runtime fork and the
AVEngine main repository. Pin the runtime with `runtime.lock.yaml`; never
vendor Habitat source into AVEngine.

## Alternatives considered

- One monorepo containing copied Habitat source.
- Git subtree or history rewrite.
- Multiple early repositories for assets, benchmark and schemas.

## Consequences

Runtime changes remain an understandable upstream diff, while AVEngine can
version its own contracts. Cross-repository features require coordinated
commits and lock updates. Developers must build/install the runtime separately.

## Validation plan

CI/bootstrap must reproduce both exact commits and record them in sample
provenance. Migration review checks that new files live in the owning repo.

## Reversal criteria

Split example assets or benchmark code only after their contracts stabilize.
Do not merge the two required repositories unless upstream history and license
boundaries can still be preserved transparently.
