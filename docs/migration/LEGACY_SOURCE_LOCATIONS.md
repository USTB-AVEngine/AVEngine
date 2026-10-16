# Legacy Source Locations

This file records the immutable starting points for the Habitat-native
migration. It is an inventory pointer, not a claim that every legacy artifact
is production-ready.

## AVEngine main repository

- Local legacy worktree: `/data/jzy/code/AVEngine`
- Repository: `git@github.com:Eastforward/AVEngine.git`
- Branch: `main`
- Pushed snapshot: `92775d4d2050a3a9b277357eb83c9243468f4cd3`
- Clean Habitat-native worktree: `/data/jzy/code/AVEngine-habitat-native`
- Habitat-native branch: `feature/habitat-native-foundation`

The root snapshot preserves the legacy plans, design records, and local Beagle
review entrypoints. Media referenced by local review HTML remains outside Git
under the ignored `external/SPEAR/tmp` tree.

## SPEAR legacy implementation

- Local checkout: `/data/jzy/code/AVEngine/external/SPEAR`
- Repository: `git@github.com:Eastforward/spear.git`
- Branch: `feature/plan2-flag-generator-m1`
- Pushed snapshot: `7fbf3632fdb63cc2eceea564811c9597cabfb199`

The snapshot was split into controlled-data, implementation/test, and
documentation commits. A Hugging Face access token found in two untracked
legacy plans was replaced with `<REDACTED_HUGGINGFACE_TOKEN>` before either
file entered Git history. The token must still be revoked or rotated at the
provider.

The following local files were intentionally not pushed because they require
provenance cleanup or are derived/duplicated artifacts:

- `assets/textures/dogs/border_collie_diffuse.png`
- `data/controlled_source_attributes_v1/audioset_indoor_animal_source_registry_v1.json`
- `data/controlled_source_attributes_v1/audioset_ontology_official_page.html`
- `data/controlled_source_attributes_v1/references/animal/quaternius_horse_authored_rest_pose_side_four_limb_clay_v1/`
- `data/controlled_source_attributes_v1/rejected_profiles/animal/horse_bay_four_limb_rest_side_clay_v1.json`
- `docs/assets/pipeline/`
- `docs/audioset_indoor_animal_source_coverage.md`
- `docs/current_animal_audio_visual_dataset_pipeline.md`
- `docs/current_animal_audio_visual_dataset_pipeline_zh.md`

Do not delete these files during migration. Classify each one in the formal
migration matrix as keep, migrate, optional, retire, or experimental.

## Runtime fork

- Local checkout: `/data/jzy/code/habitat-sim-AVEngine`
- Repository: `git@github.com:Eastforward/habitat-sim-AVEngine.git`
- Upstream: `https://github.com/facebookresearch/habitat-sim.git`
- Branch: `feature/habitat-native-runtime-foundation`
- Initial fork/upstream commit: `57ee4941dc4765240f0f91f70b2c97a919bf9038`
- RLR submodule commit: `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`

The pre-existing `/data/jzy/code/habitat-sim` checkout remains an untouched
upstream/RLR baseline and must not be repurposed as the AVEngine runtime fork.
