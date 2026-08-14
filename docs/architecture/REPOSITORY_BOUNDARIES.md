# Repository Boundaries

Status: accepted target architecture; source migration is in progress.

## Canonical source repository

The final AVEngine product has one source repository:
[`USTB-AVEngine/AVEngine`](https://github.com/USTB-AVEngine/AVEngine).
It contains AVEngine-owned code and small configuration together with the
selected Habitat-Sim and SPEAR source, AVEngine-owned RLR adapter source and
small interface configuration required by supported production routes.

This is a source boundary, not a claim that every runtime input belongs in Git.
The repository does not contain Unreal Engine installations, Epic content,
datasets, native room packages, model weights, generated media, caches, build
trees or packaged binaries.

## Production room routing

| Room family | Production visual execution | Other authority |
| --- | --- | --- |
| MP3D | Habitat-Sim scene, pixels, sensors and articulated pose | RLR acoustics use SoundSpaces material authority on the same Habitat scene and state |
| SPEAR `apartment_0000` | Native UE/SPEAR map | AVEngine owns Timeline, task/source state, navigation semantics, audio, Topdown, labels and admission |
| InteriorAgent/Kujiale | UE/SPEAR USD/MDL adapter over the explicitly selected external scene | AVEngine owns Timeline, task/source state, navigation semantics, audio, Topdown, labels and admission |
| Skokloster | Excluded | It is not executed or counted unless the project owner explicitly reauthorizes it for a named task |

An MP3D UE import remains a `comparison_visual` diagnostic. A structural pass
on that path cannot make it production output or a counted Episode.

## Code included in the target repository

The target repository may include:

- AVEngine packages, CLI, schemas, examples, tests and small runtime
  configuration;
- the selected Habitat-Sim runtime and binding changes required by AVEngine;
- the AVEngine/Habitat RLR adapter source, headers and small interface/build
  configuration required by the supported acoustic path, but not a precompiled
  RLR library represented as source;
- the selected SPEAR client, UE plugin and project-control source required by
  Apartment and Kujiale production visual execution; and
- third-party license texts and per-path upstream adaptation records.

Selection is deliberate. Do not copy a complete upstream repository when a
bounded set of source files is sufficient. Preserve upstream behavior as the
default and keep AVEngine-specific behavior explicit.

## Inputs and installations kept outside Git

The following remain external inputs even after source integration:

- Unreal Engine and all Epic-distributed engine/editor content;
- MP3D, InteriorAgent/Kujiale, native Apartment and other room/dataset assets;
- HRTF, model weights, licensed source media and other data assets;
- the RLR shared library unless redistributable propagation-engine source is
  separately obtained, reviewed and selected for integration;
- Conda/virtual environments, compiled libraries, object files, UE packages and
  other build products; and
- generated images, audio, video, native evidence bundles and caches.

External data and installations are configured by repository-relative
configuration or environment overrides. Their external location does not create
a second AVEngine source repository.

## Current migration boundary

The target above is not yet the current checkout layout. During migration:

- `manifest.yaml`, `paths.yaml` and `scripts/setup.sh` still describe a
  pinned external Habitat runtime fork;
- native Apartment and Kujiale execution still uses a maintained SPEAR
  checkout and an external UE installation; and
- those workspaces remain available long enough to establish the
  pre-migration reference and compare the integrated implementation.

Do not remove those transition paths or describe the source migration as
complete until the selected code has landed and the same production routes have
passed the planned pre/post checks. After cutover, setup, build and runtime must
not clone, initialize a submodule for, or resolve code from a second AVEngine,
Habitat, RLR or SPEAR Git checkout.

## Upstream attribution and change placement

`docs/provenance/UPSTREAM_ADAPTATIONS.md` records whether a path is adapted,
reimplemented or merely calls an external installation. This mapping and the
applicable third-party license travel with imported source; they are provenance,
not a runtime code dependency.

Place high-level AVEngine behavior in AVEngine-owned modules. Keep selectively
integrated upstream code in an identifiable subtree or adapter boundary so that
license ownership and future upstream comparison remain understandable.

Historical release manifests may record both AVEngine and the runtime-fork
commit because that was the execution topology that produced them. Do not
rewrite those records to resemble the target architecture.

## View and episode authority

All room routes implement
[ADR-0009](../adr/ADR-0009-single-view-multimodal-sensor-rig.md): one logical
`camera_rig_0`, exactly the formal `view0`, co-located RGB/depth/semantic
sensors and one co-located `listener0`. Sources remain independently named and
positioned. Top-down cameras belong to QA and metadata tooling rather than a
second formal view.

UE/SPEAR production pixels do not create a parallel Timeline, task, navigation,
audio, label or admission authority. Those remain AVEngine-owned for Apartment
and Kujiale just as they do for Habitat-rendered MP3D.

## Repository count

Do not create an additional required source repository for assets, benchmarks,
schemas or backend adapters. If a future distribution boundary is proposed, it
requires a separate owner decision; it is not part of the current
single-source migration.
