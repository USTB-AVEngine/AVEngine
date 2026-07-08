# AGENTS.md

Read this file before doing any work in `/data/jzy/code/AVEngine`.
If you discover an important path convention, environment requirement, or
pipeline trap while working, update this file in the same change.

## Workspace Layout

- AVEngine monorepo root: `/data/jzy/code/AVEngine`
- SPEAR checkout: `/data/jzy/code/AVEngine/external/SPEAR`
- Hunyuan3D checkout: `/data/jzy/code/AVEngine/external/Hunyuan3D-2.1`
- Hunyuan/SPEAR animal batch assets:
  `/data/jzy/code/AVEngine/external/SPEAR/tmp/hy3d_batch`
- Approved review assets:
  `/data/jzy/code/AVEngine/external/SPEAR/tmp/hy3d_batch/approved/{tag}`
- External ReplicaCAD dataset root: `/data/datasets/replica_cad`
  (`AVENGINE_REPLICACAD_ROOT` overrides it).
- External Mixamo dataset root: `/data/datasets/mixamo`
  (`AVENGINE_MIXAMO_ROOT` overrides it).

Prefer repo-relative paths when editing code, but use absolute paths when
calling scripts that `chdir` internally.

## Python Environments

- `spear-env`: SPEAR/UE rendering, review-video builder, most lightweight
  SPEAR-side tests.
- `ss2`: Habitat/RLR audio, `trimesh`, auto-orient ingest, review UI mesh
  processing, ReplicaCAD Habitat smoke checks.
- `hunyuan3d`: Hunyuan3D shape and paint generation only.

Do not assume a test failure is real before checking that it was run under the
right environment. For example, `test_auto_orient_ingest.py` needs `ss2`
because it imports `trimesh`.

## ReplicaCAD / Mixamo Data Traps

- Official ReplicaCAD download command in `ss2`:
  `/data/jzy/miniconda3/envs/ss2/bin/python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path /data/datasets --no-replace`
- The Habitat downloader writes versioned data under
  `/data/datasets/versioned_data/replica_cad_dataset_1.5` and creates the
  active symlink `/data/datasets/replica_cad`. Use that symlink as the default
  root, not `/data/datasets/replicacad`.
- ReplicaCAD scene loading in Habitat needs
  `SimulatorConfiguration.scene_dataset_config_file =
  /data/datasets/replica_cad/replicaCAD.scene_dataset_config.json` and
  `scene_id = "apt_0"` style scene IDs.
- In `ss2`, `apt_0` loads, but `sim.pathfinder` is not loaded automatically.
  Explicitly call `sim.pathfinder.load_nav_mesh(
  "/data/datasets/replica_cad/navmeshes/apt_0.navmesh")` before sampling
  walkable points.
- Habitat may print articulated-object creation failures for ReplicaCAD URDFs
  during a minimal non-physics smoke. Treat the smoke as valid if the simulator
  is created and the explicit navmesh load succeeds; solve articulated object
  physics separately when implementing interactive objects.
- Mixamo FBX files normally require user account/browser download. Do not try
  to scrape Mixamo; place user-downloaded FBX files under `/data/datasets/mixamo`
  or set `AVENGINE_MIXAMO_ROOT`.

## Hunyuan3D Path Traps

- The actual Hunyuan root in this workspace is
  `/data/jzy/code/AVEngine/external/Hunyuan3D-2.1`.
- Old scripts may still mention `/data/jzy/code/Hunyuan3D-2.1`; treat that as
  stale unless the directory actually exists.
- `tools/hy3d_bake_diffuse.py` changes cwd to `HY3D_ROOT` because Hunyuan
  resolves config/checkpoint paths relative to cwd. Pass absolute paths for
  `--input-glb`, `--reference-image`, and `--workdir`.
- In this monorepo, the valid RealESRGAN checkpoint is normally
  `external/Hunyuan3D-2.1/ckpt/RealESRGAN_x4plus.pth`. The historical symlink
  under `hy3dpaint/ckpt/RealESRGAN_x4plus.pth` may point at the stale
  `/data/jzy/code/Hunyuan3D-2.1` path; do not rely on that symlink.
- Hunyuan paint local weights are under
  `external/Hunyuan3D-2.1/pretrained_models/hunyuan3d-2.1/hunyuan3d-paintpbr-v2-1`.
  If a wrapper leaves `multiview_pretrained_path` as `tencent/Hunyuan3D-2.1`,
  the patched Hunyuan loader will look under
  `$HY3DGEN_MODELS/tencent/Hunyuan3D-2.1/...` and then start a slow HF download.
- The `hunyuan3d` env's editable `custom_rasterizer` install can also point
  at the stale `/data/jzy/code/Hunyuan3D-2.1` path. Rebuild it from the real
  directory with `pip install -e . --no-build-isolation` inside
  `external/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer` when
  `custom_rasterizer.rasterize` is missing or the `.so` has torch ABI errors.
- Hunyuan shape output alone is not enough for review/render. The paint stage
  must produce at least:
  - `hy3d_textured.obj`
  - `hy3d_diffuse.jpg`
  - `hy3d_metallic.jpg` when available
  - `hy3d_roughness.jpg` when available

## Approved Animal Asset Requirements

Approved animated assets must not be gray/untextured. For every animated tag in
`tmp/hy3d_batch/approved/{tag}`, expect:

- `direction.json` with `human_approved: true`
- `mesh_oriented.glb`
- `hy3d_diffuse.jpg`
- `mesh_runtime.glb` after runtime proxy generation, when the gate has cooked it

When rotating/baking meshes in `auto_orient_ingest.py` or
`review_ui_server.py`, preserve UV/material data by copying the mesh and
changing vertices. Do not rebuild a naked `trimesh.Trimesh(vertices, faces)`,
    because that drops UVs and produces gray animals downstream.

Animal gate checks (`tools/gate_check_animal.sh`) intentionally do expensive
runtime proxy, rig swap, UE import, cook, package, and orbit-render work. High
CPU during these gates is expected and should not be repeated per dataset clip.
Once a tag passes gate, normal batch generation should reuse the approved
runtime/cooked assets and only render/audio the clip contents.

UE 5.5 Interchange GLB animation import may print a handled ensure at
`InterchangeGltfAnimation.cpp:965` while importing rigged animals. Do not mark a
gate failed from that line alone; use the actual process exit code and sentinel
lines such as `GATE_CHECK_DONE`, `BUILD SUCCESSFUL`, or `Success - 0 error(s)`.

## Review Video Marker Rules

The marker in `side_by_side_review_annotated.mp4` should indicate the visible
actor, not just the acoustic source point. Prefer UE-authored
`videos/actor_visual_metadata.json` and its
`visual_center_world_xyz_per_frame` values when present.

The acoustic point can be near the feet/floor or center trajectory and may not
overlap the rendered mesh body. That is expected; the review marker should use
the visual center when available.

## Coordinate Conventions

SSOT scene frame:

- Units are meters.
- `+Z` is up.
- Mic/camera yaw `0` faces world `+X`.
- Positive yaw rotates counter-clockwise in the XY plane.
- Positive mic-local azimuth means the source is on the mic/camera left.
- Image pixel X grows to the right, so positive azimuth projects to smaller X.

Apartment UE render frame:

- UE uses centimeters.
- Apartment SSOT-to-UE has a Y flip around `APARTMENT_MIC_ORIGIN_CM`.
- Apartment camera yaw in UE is `-scene_yaw`.
- Use the helper transforms in `tools/spike_rlr/run_render_pass_apartment.py`
  instead of re-deriving this ad hoc.

RLR/Habitat audio frame:

- `_habitat_from_scene(x, y, z)` maps to `(x, z, y)`.
- The listener orientation must track `spec["mic"]["yaw_deg"]` or
  `camera_configs[0]["yaw_deg"]`; do not hardcode the listener to face
  `+Y_scene`.
- Current helper convention: Habitat agent yaw is
  `(270 - scene_yaw_deg) % 360`.
- Do not use world-axis assumptions such as "world +X is always right ear" for
  random-yaw clips. Audio left/right must be listener-local.

## Useful Verification Commands

From `/data/jzy/code/AVEngine/external/SPEAR`:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_review_videos.py \
  tests/tools/spike_rlr/test_room_conventions.py \
  tests/tools/spike_rlr/test_hy3d_generate_and_audit.py \
  tests/tools/test_species_rig_map_approved_assets.py \
  tests/tools/spike_rlr/test_run_audio_pass_cli.py \
  tests/tools/test_hy3d_bake_diffuse_paths.py
```

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_auto_orient_ingest.py
```

When validating a generated review batch, inspect all three views together:

- UE video frames
- topdown review video
- `apartment_v1_metadata.json`

Do not rely on audio loudness alone to infer left/right; dry source content can
confound RMS. Compare mic-local azimuth, rendered actor position, and stereo
rendering as separate evidence streams.
