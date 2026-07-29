# AVEngine

AVEngine is a Habitat-native research toolkit for building deterministic,
identity-preserving audiovisual episodes from explicit source assets, rooms,
motion programs and evidence contracts.

Habitat-Sim provides scene state, navigation, sensors and articulated runtime
foundations. RLR Audio Propagation provides geometric acoustics. AVEngine owns
the asset and room packages, authoritative timeline, source-aware audio
assembly, QA, provenance and dataset admission logic.

## Capabilities

- Deterministic single-rig RGB, depth and semantic capture with a co-located
  listener.
- Named dynamic sources with per-source FOA and binaural RIRs, stems and
  mixtures.
- Exact integer-tick timelines, synchronized media and controlled
  counterfactual episodes.
- Versioned room, asset, emitter, sound and runtime-profile registries.
- Fail-closed asset qualification with retained machine and human evidence.
- Optional SPEAR/UE RGB for configured comparison and Apartment workflows;
  Habitat-native state, navigation, audio, Topdown and labels remain
  authoritative.

AVEngine is not a new simulator, renderer or acoustic solver. It does not infer
physical acoustic truth from visual materials, and successful file generation
alone does not qualify an asset, room or dataset.

## System flow

```text
dataset request
  -> asset, room and acoustic-scene packages
  -> authoritative source programs and Timeline
  -> Habitat-native state, sensors and RLR propagation
  -> per-source audio, mixtures, RGB/Topdown and labels
  -> QA, provenance and dataset index
```

Generated animals must keep their own breed-specific Pixel3D geometry through
repair, rigging, animation and runtime validation. A library animal may donate
motion, but it may not replace the generated mesh, silhouette, joints or skin
weights.

## Quick start

Requirements: Linux, Git and Python 3.10 or newer. Native Habitat/RLR builds,
large datasets, Blender and optional UE/SPEAR tooling are separate layers.

```bash
git clone https://github.com/Eastforward/AVEngine.git
cd AVEngine

./scripts/setup.sh
./.venv/bin/avengine --help
```

`scripts/setup.sh` creates `.venv`, installs AVEngine with test dependencies,
validates configured paths and schemas, and runs the hermetic unit suite. Use
`./scripts/setup.sh --dry-run` to inspect the bootstrap without changing the
workspace.

Native, Blender, media-readback and release canaries require their declared
runtime and asset inputs. Follow the matching execution record under
[`docs/roadmap/`](docs/roadmap/) rather than treating unit tests as native
evidence.

## Current status

`main` is the Habitat-native integration baseline. The active Apartment route
has generic `source1` and `source2` bindings and a completed 1,000-episode
research closure with episode-level train/validation/test splitting. Generated
animals, room acoustics and optional UE outputs retain their individual
qualification boundaries; this is not a blanket dataset-release claim.

Use these records instead of status prose copied into this README:

- [Current Apartment checkpoint](docs/roadmap/CURRENT_APARTMENT_EXECUTION.md)
- [Milestones and evidence status](docs/roadmap/MILESTONES.md)
- [Release manifest](release/avengine_release_manifest_v1.json)

The release manifest is the only cross-repository release authority. A branch,
schema, preview or passing unit suite is not by itself a release.

## Repository boundaries

| Repository | Responsibility |
| --- | --- |
| `Eastforward/AVEngine` | Packages, registries, Timeline, audio assembly, QA, provenance, CLI and dataset admission |
| `Eastforward/habitat-sim-AVEngine` | Bounded Habitat runtime extensions, articulated playback, explicit acoustic ingestion and the RLR adapter |

Legacy SPEAR/UE and gpuRIR code is optional migration or comparison material,
not the default runtime architecture. Private model experiments, checkpoints
and evaluation environments stay outside this repository.

## Documentation

- [System overview](docs/architecture/SYSTEM_OVERVIEW.md)
- [Repository ownership and API boundaries](docs/architecture/REPOSITORY_BOUNDARIES.md)
- [Generated-animal asset and instance contract](docs/assets/GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md)
- [Acoustic scene and material contract](docs/architecture/ACOUSTIC_SCENE_AND_MATERIALS.md)
- [Timeline and episode contract](docs/architecture/EPISODE_AND_TIMELINE.md)
- [Filesystem trust model](docs/security/FILESYSTEM_TRUST_MODEL.md)
- [Troubleshooting](docs/troubleshooting.md)

Schemas live under [`schemas/`](schemas/), executable examples under
[`examples/`](examples/), and milestone-specific reproduction commands under
[`docs/roadmap/`](docs/roadmap/).

## Status vocabulary

Evidence uses `pass`, `fail`, `blocked` and `not_run`. Research candidates and
historical approvals cannot become `approved_for_dataset` without the required
current evidence and registration decision.

## License and rights

See [LICENSE](LICENSE), [CITATION.cff](CITATION.cff),
[CITATIONS.bib](CITATIONS.bib) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The current AVEngine license
is all-rights-reserved until an explicit open-source license is selected.
Habitat, RLR, models, rooms, sounds and generated assets retain their own
terms; RLR's current route is non-commercial.
