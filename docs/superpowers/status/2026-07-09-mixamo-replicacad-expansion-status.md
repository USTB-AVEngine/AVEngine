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
  - result: `missing_data`
- Added a conservative automatic direction gate:
  - script: `tools/spike_rlr/direction_gate.py`
  - reports: `external/SPEAR/tmp/direction_gate_reports/*.json`
  - pass: `cat_british_shorthair`, `dog_golden`
  - block: `cat_british_shorthair_v2`, `dog_beagle`, `dog_beagle_v2`,
    `dog_husky`

## Tests

- SPEAR/spear-env:
  - command: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q tests/tools/spike_rlr/test_external_data_paths.py tests/tools/spike_rlr/test_mixamo_probe.py tests/tools/spike_rlr/test_replicacad_probe.py tests/tools/spike_rlr/test_review_videos.py tests/tools/spike_rlr/test_room_conventions.py tests/tools/spike_rlr/test_hy3d_generate_and_audit.py tests/tools/test_species_rig_map_approved_assets.py tests/tools/spike_rlr/test_run_audio_pass_cli.py tests/tools/test_hy3d_bake_diffuse_paths.py`
  - result: `41 passed`
- SPEAR/ss2:
  - command: `/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q tests/tools/spike_rlr/test_auto_orient_ingest.py tests/tools/spike_rlr/test_direction_gate.py`
  - result: `10 passed`

## Not Completed

- Mixamo actual UE import did not run because `/data/datasets/mixamo` does not
  exist and no FBX files are available. Mixamo normally requires a user account
  and browser/manual FBX download; I did not scrape it.
- ReplicaCAD UE import/render adapter is not implemented yet. The data is now
  downloaded and Habitat-readable, but there is not yet a SPEAR/UE visual import
  path, RLR acoustic mesh/material sidecar, or review `side_by_side` clip for a
  ReplicaCAD room.
- The automatic direction gate is intentionally conservative. It blocks several
  approved animals because bbox geometry alone is near-symmetric. This confirms
  that a "100% no-human semantic direction guarantee" is not available from
  geometry-only evidence. The next upgrade should add rendered multi-view and
  motion-based evidence before expanding automatic approval.

## Important Traps Recorded

- `AGENTS.md` now records the real ReplicaCAD root `/data/datasets/replica_cad`.
- Habitat smoke with ReplicaCAD needs `ss2`.
- `sim.pathfinder` is not loaded automatically for `apt_0`; explicitly load
  `/data/datasets/replica_cad/navmeshes/apt_0.navmesh`.
- Minimal non-physics ReplicaCAD load can print articulated-object creation
  failures for URDFs. Treat the smoke as valid when simulator creation and
  explicit navmesh loading succeed.

## Running Processes

- No download or render process intentionally left running.
