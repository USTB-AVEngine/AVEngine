# Legacy Migration Matrix

The source paths in this document are relative to the legacy SPEAR checkout
unless prefixed with `AVEngine/`. Every decision begins with exactly one base
classification: `migrate`, `keep`, `optional`, `experimental`, `retire`, or
`quarantine`. Text after a slash or in the treatment column is a qualifier, not
another state (for example, `migrate / downgrade on import`).

## Code and entrypoints

| Legacy entry | Decision | New owner / replacement | Stop or review |
| --- | --- | --- | --- |
| `AVEngine/manifest.yaml`, setup/update scripts, old Conda files | retire | repository lock files and reproducibility docs | M0 as an entrypoint; remove compatibility text M6 |
| `AVEngine/paths.yaml`, `scripts/load_paths.py` | migrate | read-only v1 config importer; no author-machine defaults | M6 |
| `AVEngine/assets/mesh_library/**/*.glb` | migrate | `avengine.assets` template candidates with complete QA | M2 |
| legacy design/review docs | keep | frozen commit links or selected migration archive | no automatic deletion |
| SPEAR runtime, RPC, UE cook/render code | optional | frozen legacy UE backend consuming canonical episodes | M7 review |
| `tools/gpurir_scenes/run_all_scenes.py`, `run_scene.py` | optional | new `avengine` CLI; optional backend adapter only | direct entrypoints stop M6 |
| `scene_spec.py` | migrate | `avengine.episodes` and integer `avengine.timeline` | M5 |
| `run_render_pass.py`, legacy UE render examples | optional | Habitat multi-sensor fixed-state capture | primary use stops M2; M7 review |
| `run_audio_pass.py`, `run_audio_pass_gpurir.py` | optional | timeline-driven gpuRIR comparison backend | M7 review |
| `audio_registry.py` | retire | content-addressed audio/provenance registry | M5 |
| `tools/audio_event_schedule.py` | migrate | deterministic tick/sample event intervals | M5 |
| `mux_audio_video.py` | retire | non-authoritative QA preview with readback checks | M5 |
| furniture/AABB dump and mapping tools | experimental | placement broad phase and debug only | production ban M3 |
| `tools/spike_rlr/run_all.sh`, `run_apartment.sh` | experimental / later retire | frozen comparison evidence; new canaries | M3-M4 |
| `run_habitat_all.py` | retire | articulated Habitat runtime plus modern RLR adapter | M2/M4 |
| `dataset_runner.py` | migrate | `avengine.cli`, episodes, registry, QA | M6 |
| scene/rejection/trajectory/event/flag pure logic | migrate | `avengine.episodes` and `avengine.qa`; real navmesh/raycast checks | M6 |
| two-dog legacy scenes | experimental | counterfactual fixture with stable actor/source/event/anchor IDs | M5 |
| `gen_mesh.py` controlled shoebox | migrate | RoomPackage/AcousticScenePackage compiler and hash | M3 |
| apartment AABB shell exporters | retire | real render/collision surface export with transforms/material slots | M3 |
| `run_audio_pass_rlr.py` | retire | modern RLR C API, named pairs and independent IR/stems | M4-M5 |
| `compute_acoustic_metadata.py` | migrate / selected fields | coordinate/visibility metadata and real IR metrics; remove fake DRR | M3/M5 |
| top-down/review/analysis tools | optional | non-authoritative QA visualization | organize M6 |
| stable template Blender builders and batch driver | migrate | `tools/blender`, canonical AssetPackage compiler | M2/M6 |
| rig/deformation/contact/gait audit tools | migrate | asset QA with real Blender canary | M2 |
| raw generated-topology/weight-transfer route | experimental | isolated migration lab; never default | quarantine M0; review M7 |
| image/mesh/model guide providers | optional | offline guide plugins; never topology/rig authority | M7 review |
| `controlled_source_asset_schema.py` concepts | migrate | small schemas, canonical hashes, rights/QA state machine | M6 |
| old source manifests/registry | migrate / downgrade on import | explicit importer into new asset registry | M6 |
| `promote_source_asset.py` | retire | central Dataset Admission from fresh evidence | prohibited M0; remove M6 |
| Rocketbox/human/Mixamo route | optional | future human extension, outside Dog MVP | M7 |

## Data and assets

| Legacy data | Decision | Required treatment |
| --- | --- | --- |
| `data/shoebox_v2_spec.json` | migrate | controlled RoomPackage and episode fixture; timeline remains separate authority |
| apartment v1/v2 specs | experimental | migration fixtures only until real visual/acoustic geometry is complete |
| `apartment_shell_map.json` | retire for acoustics | retain only dump provenance/debug evidence |
| `apartment_furniture_map.json` | experimental | broad-phase placement only |
| `acoustic_material_db.json` | migrate | add sources, units, bands, confidence, version |
| `audio_library_v1.json` | migrate / metadata only | content IDs, hashes, licenses, split and usage policy |
| `source_assets_v1/**` | migrate / downgrade on import | full revalidation; no legacy approval transfer |
| controlled-source profiles/contracts/reviews | migrate / selected records | preserve useful contracts and rejected evidence, not every derived medium |
| stable template attribute catalog | migrate | retain `research_candidate` and non-authorized status |
| `tmp/**` and all untracked legacy groups | quarantine | item-by-item provenance/hash/license review before import |
| dog textures with unresolved provenance | quarantine / later retire | cannot enter a canonical package |

## Exact untracked legacy quarantine

These paths existed outside the SPEAR Git baseline at M0 and therefore require
an explicit item-level decision:

| Exact legacy path | Decision | Treatment |
| --- | --- | --- |
| `assets/` | quarantine | do not bulk import; audit each asset/hash/license |
| `data/controlled_source_attributes_v1/audioset_indoor_animal_source_registry_v1.json` | quarantine | metadata candidate only after audio-rights review |
| `data/controlled_source_attributes_v1/audioset_ontology_official_page.html` | keep / evidence | retain only as license/source evidence, not runtime data |
| `data/controlled_source_attributes_v1/references/animal/quaternius_horse_authored_rest_pose_side_four_limb_clay_v1/` | quarantine | item-level source and derivative review |
| `data/controlled_source_attributes_v1/rejected_profiles/animal/horse_bay_four_limb_rest_side_clay_v1.json` | keep / rejection evidence | preserve without approval upgrade |
| `docs/assets/` | quarantine | select only small decision/failure evidence; no bulk media migration |
| `docs/audioset_indoor_animal_source_coverage.md` | keep / audit evidence | preserve as non-authoritative legacy analysis |
| `docs/current_animal_audio_visual_dataset_pipeline.md` | keep / migration evidence | frozen legacy description, not current setup |
| `docs/current_animal_audio_visual_dataset_pipeline_zh.md` | keep / migration evidence | frozen legacy description, not current setup |

## Tests

- **Migrate:** deterministic pure functions, event scheduling, controlled-room
  mesh construction, hash/no-overwrite/provenance states, trajectory and
  constraint math, and runtime-independent deformation/contact math.
- **Optional backend suites:** UE/SPEAR render/import/cook and gpuRIR parity.
- **Reclassify:** `_static.py` and source-string tests under
  `tests/static_contract/`; never report them as runtime passes.
- **Add:** the Blender, Habitat, room, material, RLR, timeline,
  counterfactual, and admission canaries listed in the inventory.

Compatibility adapters must be read-only and fail closed. They may translate
an old request into `research_candidate_pending_revalidation`, but they may not
synthesize missing provenance, create `canary_qualified` or
`approved_for_dataset`, replace exact timing, or admit AABB acoustics.
