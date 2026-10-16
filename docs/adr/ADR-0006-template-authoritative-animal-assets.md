# ADR-0006: Template-Authoritative Animal Assets

- Status: Accepted
- Date: 2026-07-16

## Context

Directly transferring a generic rig and nearest-surface weights to arbitrary
generated topology produced cross-limb weights, wrong joint centers, stretching
and unstable topology. Repair metrics did not establish production quality.

## Decision

Audited templates own topology, UV, semantic skeleton, weights, collision and
action families. Generated meshes provide shape and PBR guides. Fit only inside
the template's validated morphotype domain; reject OOD requests.

## Alternatives considered

- One universal dog rig and topology.
- Direct generated topology plus copied weights.
- Target-native generated rigs as the default route.

## Consequences

Production throughput becomes stable and auditable, but a template bank and
bounded fitting are required. Target-native unknown topology remains an
explicit research route.

That research route must still preserve the source-asset/instance boundary. A
new species, breed or materially different morphotype is an independent
`research_candidate` source asset, not an instance-level recolour or reshape of
an accepted animal. It may reuse compiler mechanics and a compatible motion
family, but not another breed's mesh, silhouette, joints or skin weights as
shape authority. The operational contract is
[`GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md`](../assets/GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md).

## Validation plan

Build dog morphotype canaries and require geometry, skeleton, deformation,
contact, collision and visual-review gates across complete Walk/Idle cycles.

## Reversal criteria

A target-native route may be promoted only after deterministic canaries match
or exceed template baselines across the intended morphotype distribution.
