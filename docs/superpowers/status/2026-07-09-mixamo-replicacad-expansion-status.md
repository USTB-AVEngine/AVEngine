# Mixamo + ReplicaCAD Expansion Status

Date: 2026-07-09

## Completed

- Wrote and pushed the expansion design and execution plan in AVEngine:
  - `docs/superpowers/specs/2026-07-09-mixamo-replicacad-expansion-design.md`
  - `docs/superpowers/plans/2026-07-09-mixamo-replicacad-expansion.md`
- Pushed AVEngine root docs to GitHub:
  - remote: `origin`
  - branch: `main`
  - commit: `4c4b0dc docs: plan Mixamo and ReplicaCAD expansion`
- Pushed AVEngine status/path updates to GitHub:
  - remote: `origin`
  - branch: `main`
  - commit: `bffe6c5 docs: record ReplicaCAD download and expansion status`
- Pushed the previous SPEAR review/audio/asset fixes:
  - remote: `eastforward`
  - branch: `feature/plan2-flag-generator-m1`
  - commit: `55b12cd0 fix(review): align actor markers, audio yaw, and painted assets`
- Pushed the new SPEAR external data probes and direction gate:
  - remote: `eastforward`
  - branch: `feature/plan2-flag-generator-m1`
  - commit: `93908582 feat(spike): add external data probes and direction gate`
- Added tested external data path helpers:
  - `tools/spike_rlr/external_data_paths.py`
  - ReplicaCAD default root: `/data/datasets/replica_cad`
  - Mixamo default root: `/data/datasets/mixamo`
- Downloaded ReplicaCAD with Habitat's official downloader:
  - command: `/data/jzy/miniconda3/envs/ss2/bin/python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path /data/datasets --no-replace`
  - versioned data: `/data/datasets/versioned_data/replica_cad_dataset_1.5`
  - active symlink: `/data/datasets/replica_cad`
  - official docs used: https://aihabitat.org/datasets/replica_cad/
- Added and ran ReplicaCAD probe:
  - script: `tools/spike_rlr/replicacad_probe.py`
  - status JSON: `external/SPEAR/tmp/replicacad_probe/status.json`
  - result: `ready`
  - scene instances: `91`
  - mesh files: `197`
  - first scene: `configs/scenes/apt_0.scene_instance.json`
  - first scene summary: stage `stages/frl_apartment_stage`, `113` object
    instances, `6` articulated object instances.
- Ran a minimal Habitat smoke for ReplicaCAD `apt_0`:
  - scene dataset config:
    `/data/datasets/replica_cad/replicaCAD.scene_dataset_config.json`
  - scene id: `apt_0`
  - explicit navmesh:
    `/data/datasets/replica_cad/navmeshes/apt_0.navmesh`
  - navmesh load: success
  - navigable area: `47.2612`
- Added and ran Mixamo probe:
  - script: `tools/spike_rlr/mixamo_probe.py`
  - status JSON: `external/SPEAR/tmp/mixamo_probe/status.json`
  - result before manual upload: `missing_data`
- Re-ran Mixamo probe after manual FBX upload:
  - root: `/data/datasets/mixamo`
  - files:
    - `raw/Walking.fbx`
    - `raw/Standing_Idle.fbx`
  - result: `ready`
- Downloaded ReplicaCAD baked lighting with Habitat's official downloader:
  - command: `/data/jzy/miniconda3/envs/ss2/bin/python -m habitat_sim.utils.datasets_download --uids replica_cad_baked_lighting --data-path /data/datasets --no-replace`
  - versioned data:
    `/data/datasets/versioned_data/replica_cad_baked_lighting_1.5`
  - active symlink: `/data/datasets/replica_cad_baked_lighting`
  - source package:
    `https://dl.fbaipublicfiles.com/habitat/ReplicaCAD/ReplicaCAD_baked_lighting_v1.5.zip`
- Added a conservative automatic direction gate:
  - script: `tools/spike_rlr/direction_gate.py`
  - reports: `external/SPEAR/tmp/direction_gate_reports/*.json`
  - pass: `cat_british_shorthair`, `dog_golden`
  - block: `cat_british_shorthair_v2`, `dog_beagle`, `dog_beagle_v2`,
    `dog_husky`
- Confirmed direction policy after review:
  - human direction review remains the final approval gate.
  - automatic direction checks are only a prefilter/alarm, not a replacement.
- Surveyed AudioSet animal expansion candidates:
  - official AudioSet animal branch covers domestic pets, livestock/farm
    animals, and wild animals.
  - current animated approved/render pool is effectively dog/cat focused.
  - existing SPEAR static animal map already has assets for `goat`, `sheep`,
    `pig`, `horse`, `cattle_bovinae`, `yak`, and `donkey_ass`.
  - Hunyuan AudioSet-style textured assets already exist for additional static
    candidates including `bird_animal`, `chicken_rooster`, `duck`,
    `frog_animal`, `goose`, `owl_animal`, `pig`, `pigeon_dove`, `sheep`,
    `snake_animal`, `turkey`, and `yak`.
  - local Objaverse maps under `/data/datasets/jzy/assets/objaverse` contain
    many useful category names, but several paths point to the stale
    `/home/jzy/.objaverse/...` cache and do not exist on this machine.

## Tests

- SPEAR/spear-env:
  - command: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q tests/tools/spike_rlr/test_external_data_paths.py tests/tools/spike_rlr/test_mixamo_probe.py tests/tools/spike_rlr/test_replicacad_probe.py tests/tools/spike_rlr/test_review_videos.py tests/tools/spike_rlr/test_room_conventions.py tests/tools/spike_rlr/test_hy3d_generate_and_audit.py tests/tools/test_species_rig_map_approved_assets.py tests/tools/spike_rlr/test_run_audio_pass_cli.py tests/tools/test_hy3d_bake_diffuse_paths.py`
  - result: `41 passed`
- SPEAR/ss2:
  - command: `/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q tests/tools/spike_rlr/test_auto_orient_ingest.py tests/tools/spike_rlr/test_direction_gate.py`
  - result: `10 passed`

## Not Completed

- Mixamo actual UE import still did not run. The two FBX files are now present,
  but they are a humanoid smoke-test source, not the main path for quadruped
  animal coverage.
- ReplicaCAD UE import/render adapter is not implemented yet. The data is now
  downloaded and Habitat-readable, including baked-lighting assets, but there is
  not yet a SPEAR/UE visual import path, RLR acoustic mesh/material sidecar, or
  review `side_by_side` clip for a ReplicaCAD room.
- The automatic direction gate is intentionally conservative. It blocks several
  approved animals because bbox geometry alone is near-symmetric. This confirms
  that a "100% no-human semantic direction guarantee" is not available from
  geometry-only evidence. Human review remains required for final direction
  approval.
- New AudioSet animal classes have not yet been added to the Plan 2 review
  source pool. The next safe expansion should first add static candidates whose
  visual assets and audio are already local, then separately handle animated
  quadrupeds via Quaternius packs/rig-family work.

## Important Traps Recorded

- `AGENTS.md` now records the real ReplicaCAD root `/data/datasets/replica_cad`.
- `AGENTS.md` now records the baked-lighting root
  `/data/datasets/replica_cad_baked_lighting`.
- `AGENTS.md` now records Mixamo local smoke assets under
  `/data/datasets/mixamo/raw`.
- `AGENTS.md` now records that Hunyuan AudioSet assets are the preferred local
  static-animal source and that Objaverse map paths may be stale.
- Habitat smoke with ReplicaCAD needs `ss2`.
- `sim.pathfinder` is not loaded automatically for `apt_0`; explicitly load
  `/data/datasets/replica_cad/navmeshes/apt_0.navmesh`.
- Minimal non-physics ReplicaCAD load can print articulated-object creation
  failures for URDFs. Treat the smoke as valid when simulator creation and
  explicit navmesh loading succeed.

## Running Processes

- No download or render process intentionally left running.
