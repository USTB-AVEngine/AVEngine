# ADR-0010: Single AVEngine Source Repository

- Status: Accepted
- Date: 2026-08-14
- Supersedes:
  [ADR-0002](ADR-0002-two-repository-boundary.md)

## Context

AVEngine currently coordinates its Python/product code with a pinned Habitat
fork and a maintained SPEAR checkout. That layout was useful while the native
changes were isolated, but it makes ordinary setup, versioning and maintenance
depend on multiple product code repositories.

The project owner requires one final repository containing all necessary
distributable source code and small configuration. This does not grant
redistribution rights for Unreal Engine, datasets, room assets, weights or
generated outputs, and it does not make those bytes appropriate Git content.

## Decision

Use [`USTB-AVEngine/AVEngine`](https://github.com/USTB-AVEngine/AVEngine) as
the single required source repository.

Selectively integrate the Habitat-Sim and SPEAR source plus the AVEngine-owned
RLR adapter source and small interface/build configuration required by
supported routes. Preserve upstream attribution and license boundaries, record
whether each imported path was adapted or reimplemented, and keep
upstream-compatible behavior as the default where applicable.

The currently pinned RLR distribution contains headers, configuration and a
precompiled shared library, not the propagation engine source. It does not
support a claim that the engine source has been integrated. Before cutover,
either separately obtain and review redistributable source or keep the shared
library as an explicitly installed external runtime dependency with owner
approval. Neither outcome may require a separate Habitat, RLR, SPEAR or
AVEngine Git checkout.

Production visual routing remains room-specific:

- MP3D uses Habitat-Sim for scene execution, pixels, sensors and articulated
  pose. RLR uses SoundSpaces material authority on the same scene and state.
- `apartment_0000` uses its native UE/SPEAR map for production visual output.
- InteriorAgent/Kujiale uses the UE/SPEAR USD/MDL adapter for production visual
  output over an explicitly selected external scene.
- Skokloster is excluded unless the project owner explicitly reauthorizes it
  for a named task.

Unreal Engine installations, Epic content, datasets, native room assets, model
weights, generated media, build trees, caches and packaged binaries remain
outside Git.

## Transition

This decision describes the target architecture, not the current completion
state. Until selected code is integrated and the same production routes have
been checked before and after the change, the manifest-pinned Habitat fork and
maintained SPEAR checkout remain migration inputs.

Do not remove those transition inputs early. After cutover, setup, build and
runtime must not clone, initialize a submodule for, or resolve product code from
a second Habitat, SPEAR or AVEngine Git repository.

## Consequences

- One AVEngine Git commit identifies the checked-in product source after
  cutover.
- Upstream source and license mapping remains explicit without becoming a
  runtime repository dependency.
- Native builds and external UE/data configuration remain separate execution
  layers even though their required distributable integration source is kept
  here.
- Historical manifests retain the repository identities that actually produced
  their results; they are not rewritten to match the new architecture.

## Validation

Use the existing ordinary unit, native Habitat/RLR, UE capture and media-readback
checks appropriate to each route. Compare fresh, no-clobber pre/post outputs
under the same requested inputs and stop at the first failed layer. This ADR
does not introduce a new hash, frozen contract, baseline or admission gate.
