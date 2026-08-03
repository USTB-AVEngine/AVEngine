# Repository Boundaries

## Repository A: Habitat runtime fork

Repository: `Eastforward/habitat-sim-AVEngine`

Allowed responsibilities:

- Minimal Habitat C++/Python runtime changes.
- Modern RLR C API adapter.
- Named sources/listeners and per-pair IR access, with exactly one listener in
  the current MVP profile.
- Explicit acoustic mesh/material package ingestion.
- Deterministic baked non-human joint-pose evaluation.
- One canonical state evaluated once and observed by co-located
  RGB/depth/semantic sensors on one logical camera rig.
- Runtime equality checks for the single camera-rig/listener transform and
  independent named source transforms.
- Runtime build/version/state manifests and runtime-specific tests.

Do not place Blender fitting, model inference, dataset registries, benchmark
training, large assets or general AVEngine CLI logic in this fork. The fork
must retain upstream history, the Habitat MIT license and an understandable
diff relative to upstream.

## Repository B: AVEngine main repository

Repository: `Eastforward/AVEngine`

Allowed responsibilities:

- Animal template bank and offline asset compiler.
- Room and acoustic-scene compilers.
- High-level runtime adapter without vendoring Habitat source.
- The MVP single-view capture profile, sensor/listener/source manifests and
  exclusion of QA-only camera artifacts from admitted observations.
- Timeline, episode and counterfactual builders.
- Dry audio, RIR/stem assembly, mixing and sample mapping.
- QA, provenance, registry and dataset admission.
- Stable CLI, schemas, examples and benchmark-facing outputs.

## Dependency rule

The main repository depends on an independently installed runtime fork pinned
by `runtime.lock.yaml`. It does not copy Habitat source or rewrite the
`habitat_sim` package name. Each sample records both repository commits plus
the upstream base and RLR submodule commit.

## Change ownership test

Put a change in the runtime fork only if all are true:

1. It must execute inside Habitat's C++/Python runtime.
2. It cannot be expressed through an existing stable Habitat interface.
3. Keeping it outside Habitat would prevent deterministic runtime behavior.

Otherwise, put it in AVEngine. Cross-repository work must land as separate
commits and be connected by an updated lock file and acceptance test.

## MVP view rule

Both repositories implement
[ADR-0009](../adr/ADR-0009-single-view-multimodal-sensor-rig.md): one logical
`camera_rig_0`, exactly the formal `view0`, co-located RGB/depth/semantic
sensors and one co-located `listener0`. Sources remain independently named and
positioned. The timeline schema may express future views, but M1, M2, M5 and
the initial M6 MVP do not. Top-down cameras belong only to QA tooling.

## Legacy route

SPEAR remains the independent `Eastforward/spear` repository and is never
vendored into AVEngine. Formal Apartment package/capture/evidence work locates
the checkout through `AVENGINE_SPEAR_ROOT`, pins the exact full commit from
`manifest.yaml`, and verifies a repository-relative map path plus its content
hash. Producer-machine absolute paths retained in historical export manifests
are diagnostic evidence, not consumer path requirements.

The old `/data/jzy/code/AVEngine/external/SPEAR` path is a historical
producer location only. On `48g-jump`, consumers may use the shared pinned
sparse checkout documented in
[M1_EXECUTION.md](../roadmap/M1_EXECUTION.md), treating it as read-only even
when group permissions allow writes. Re-exporting or launching UE requires a
contributor-owned full clone. `/data/UE_5.5` is only the Unreal Engine root
and must never be used as `AVENGINE_SPEAR_ROOT`.

## Future repositories

Do not create additional repositories during M0-M6. A small redistributable
example-assets repository and a benchmark repository may be split only after
asset and task contracts stabilize.
