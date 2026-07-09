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
- Source asset registry:
  `/data/jzy/code/AVEngine/external/SPEAR/data/source_assets_v1`
- External ReplicaCAD dataset root: `/data/datasets/replica_cad`
  (`AVENGINE_REPLICACAD_ROOT` overrides it).
- External ReplicaCAD baked-lighting root:
  `/data/datasets/replica_cad_baked_lighting`.
- External Mixamo dataset root: `/data/datasets/mixamo`
  (`AVENGINE_MIXAMO_ROOT` overrides it).
- External Quaternius raw archive root: `/data/datasets/quaternius/raw`.
- External speech corpus roots include `/data/datasets/LibriTTS`,
  `/data/datasets/wsj`, `/data/datasets/wsj0_2mix`,
  `/data/datasets/VCTK-Corpus-0.92`, and the Common Voice mirrors under
  `/data/datasets/cv-corpus-24.0-2025-12-05` and
  `/data/datasets/common_english`.

Prefer repo-relative paths when editing code, but use absolute paths when
calling scripts that `chdir` internally.
Do not use the historical checkout path `/data/jzy/code/SPEAR` in new tests or
tools; this workspace's active SPEAR checkout is under
`/data/jzy/code/AVEngine/external/SPEAR`. Derive the repo root from
`Path(__file__)` when possible.

## Python Environments

- `spear-env`: SPEAR/UE rendering, review-video builder, most lightweight
  SPEAR-side tests.
- `ss2`: Habitat/RLR audio, `trimesh`, auto-orient ingest, review UI mesh
  processing, ReplicaCAD Habitat smoke checks.
- `hunyuan3d`: Hunyuan3D shape and paint generation only.

Do not assume a test failure is real before checking that it was run under the
right environment. For example, `test_auto_orient_ingest.py` needs `ss2`
because it imports `trimesh`.

In `external/SPEAR`, do not add untracked package markers such as
`tests/tools/__init__.py`. Pytest can put `tests/` before the repo root on
`sys.path`, and then `tests/tools` shadows the real `tools` package, causing
imports like `tools.robust_skin_transfer` to fail. Keep those stray empty files
out of commits and remove them if they appear.

## ReplicaCAD / Mixamo Data Traps

- Official ReplicaCAD download command in `ss2`:
  `/data/jzy/miniconda3/envs/ss2/bin/python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path /data/datasets --no-replace`
- The Habitat downloader writes versioned data under
  `/data/datasets/versioned_data/replica_cad_dataset_1.5` and creates the
  active symlink `/data/datasets/replica_cad`. Use that symlink as the default
  root, not `/data/datasets/replicacad`.
- `/data/datasets/replica_cad` is a symlink. When doing shell discovery with
  `find`, use `find -L /data/datasets/replica_cad ...` or inspect
  `readlink -f /data/datasets/replica_cad`; otherwise the dataset can look
  empty even though Habitat can load it.
- ReplicaCAD baked lighting uses a separate downloader uid:
  `replica_cad_baked_lighting`. It writes versioned data under
  `/data/datasets/versioned_data/replica_cad_baked_lighting_1.5` and creates
  the active symlink `/data/datasets/replica_cad_baked_lighting`.
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
- Current Mixamo smoke assets are expected under `/data/datasets/mixamo/raw`.
  As of 2026-07-09 that directory contains `Walking.fbx` and
  `Standing_Idle.fbx`; do not assume a `Running.fbx` exists.
- Mixamo/humanoid animations used in demos or dataset clips must loop for the
  full clip duration. Import/playback adapters should explicitly enable looping
  or tile the animation; do not rely on a one-shot FBX action that freezes after
  its source frame range.
- For Flux/Hunyuan human meshes retargeted to Mixamo, do not rely on Blender
  bone-heat automatic weights. On 2026-07-09 it produced zero weighted
  vertices for `human_male_blue_hoodie_v1` even though the armature modifier
  existed, and the exported GLB had no skin. Use the nearest-surface weight
  transfer path in `external/SPEAR/tools/blender_robust_swap_mesh_keep_rig.py
  --weight-mode nearest` for human smoke/prototyping.
- For approved Flux/Hunyuan human assets, copy verified nearest-skin Mixamo
  runtime GLBs into the approved tag directory with
  `external/SPEAR/tools/spike_rlr/human_mixamo_runtime.py` helpers before
  promotion. Expected files are `mesh_runtime_walking.glb`,
  `mesh_runtime_standing_idle.glb`, a compatibility `mesh_runtime.glb`
  aliasing Walking, and `mesh_runtime.json` with schema
  `human_mixamo_runtime_v1`. `promote_source_asset.py` records these under
  `rig.animation_assets`; do not register a human asset that only has
  `mesh_oriented.glb`.
- For Flux/Hunyuan humans retargeted to Mixamo, the approved review mesh is in
  AVEngine-facing coordinates, but Mixamo weight transfer expects the target
  mesh to be yaw-aligned with the Mixamo source. Use
  `external/SPEAR/tools/blender_robust_swap_mesh_keep_rig.py
  --target-rotate-z-deg -90` for the current v2 human assets. That option must
  rotate mesh vertex data with `Matrix.Rotation`; do not rely on
  `rotation_euler`, because imported GLB objects can ignore that path and leave
  the bbox unchanged. If the runtime contact sheet shows arms staying in
  T-pose or skirt-like leg sheets, suspect this alignment first.
- Do not infer UE human actor scale only from the exported Mixamo GLB bbox.
  `human_male_blue_hoodie_v1` looked tiny in `trimesh`, but UE Interchange
  renders it at normal human size with `actor_scale: 1.0`; `actor_scale: 75.0`
  made a giant mesh that filled the apartment camera. Its current runtime
  calibration is `actor_scale: 1.0`, `actor_z_lift_cm: 14.0`, and
  `walking_forward_yaw_offset_deg: 90.0`. Re-check in UE review if a new
  humanoid uses a different import path.
- `actor_z_lift_cm` is a visual actor/root-height calibration, not an audio
  source height. Apartment actor spawning uses the measured floor surface
  `APARTMENT_FLOOR_Z_CM` plus this lift; visual floor thickness/collision mesh
  details should be solved per asset with this field instead of changing the
  global floor constant.
- For Mixamo humanoids, event-level `facing_yaw_deg` is the desired semantic
  world-facing direction. The renderer/composer must still add the asset's
  `walking_forward_yaw_offset_deg` from registry/runtime hints so the visual
  face points at the listener. The approved human speech demo that proves this
  is:
  `external/SPEAR/tmp/spike_output_human_speech_demo/clips/clip_0000/videos/side_by_side_review_annotated.mp4`.
- UE `GetActorBounds` can return implausible skeletal bounds for imported
  Mixamo humans. Review marker metadata should sanity-check actor bounds
  against the planned source trajectory and fall back to the trajectory point
  when the bounds center is meters away or outside apartment height.
- Quaternius animated animal packs are the cleanest external animation source:
  CC0, browser-downloadable, and available in FBX/glTF/OBJ/Blend depending on
  pack. Mixamo is primarily useful for humanoid animation smoke tests, not
  quadruped animal coverage.
- AVEngine already ships a curated Quaternius GLB subset under
  `assets/mesh_library/{quaternius_animalpack,quaternius_farm}`. The active
  dog/cat animation rigs use those GLBs. Do not describe Quaternius as missing;
  the raw archives below are for source inspection, extra animal rig work, and
  future import experiments.
- Downloaded Quaternius raw archives live in `/data/datasets/quaternius/raw`:
  - `quaternius_animal_pack_vol2_2017_opengameart.zip`
    (Cat, Dog, Eagle, Piranha, Wolf; FBX/Blend/OBJ)
  - `quaternius_farm_animals_2018_opengameart.zip`
    (Cow, Horse, Llama, Pig, Pug, Sheep, Zebra; FBX/Blend/OBJ)
  - `quaternius_ultimate_animated_character_pack_2021_opengameart.zip`
    (many human characters, including BaseCharacter/Casual/Doctor/Worker;
    FBX/Blend/OBJ)
- For human visual sources, use human/character meshes plus human animation
  assets. For the acoustic source class, map those instances to speech,
  talking, conversation, or voice audio as appropriate; speech is an audio
  category, not the animation asset.
- Current formal human source candidates should use Flux reference images
  through `external/SPEAR/tools/flux_generate_reference.py --template human`
  so image provenance matches the Hunyuan/Flux animal flow. Do not register
  built-in imagegen output as a production human source; it is acceptable only
  for brainstorming or rejected/temporary references.
- Flux/Hunyuan human references intended for Mixamo retargeting should use a
  strict T-pose or clear A-pose, with visible empty background gaps between
  each arm and the torso. Neutral "hands near pockets" reference photos can
  make Hunyuan fuse sleeves/hands into the body, and nearest-surface Mixamo
  skin transfer then produces broken arm motion. Reject such assets and
  regenerate the reference before runtime work.
- Do not run the animal auto-orient heuristic as if it were valid for human
  meshes. `external/SPEAR/tools/spike_rlr/auto_orient_ingest.py` now checks
  `source_asset_candidate.json` and, for `category: human`, writes a manual
  orientation review payload instead of generating a speculative
  `mesh_oriented.glb`. The review UI must bake the final human rotation on
  approval.
- Do not judge local human-speech availability from OmniAudio filename matches
  alone. The machine has large speech corpora under `/data/datasets`; prefer
  these for visible human speech before using generic AudioSet/OmniAudio clips.
- Good single-speaker speech roots scanned on 2026-07-09:
  - `/data/datasets/LibriTTS` (`99G`, about `377789` wav files). This is the
    main LibriTTS root with splits such as `train-clean-*`, `train-other-500`,
    `dev-*`, and `test-*`.
  - `/data/datasets/LibriTTS/LibriSpeech` is only a smaller LibriSpeech-style
    `dev-clean` subtree (`2703` flac files); do not mistake it for the main
    LibriTTS root.
  - `/data/datasets/VCTK-Corpus-0.92` (`12G`, about `88328` flac files), with
    speaker folders under `wav48_silence_trimmed`.
  - `/data/datasets/wsj` (`25G`, about `171794` audio files) includes both
    converted WSJ wav data and a `wsj0_2mix` subtree. Prefer converted
    single-speaker wav roots when attaching one visible human to one voice.
  - `/data/datasets/common_english` and
    `/data/datasets/cv-corpus-24.0-2025-12-05` are large Common Voice mirrors
    with mp3 clips; useful after transcript/quality filtering, less direct than
    LibriTTS/VCTK/converted WSJ wav.
- Mixed or derived speech datasets are useful for future multi-speaker/noisy
  audio tasks but should not be the first source for one visible speaking
  human: `/data/datasets/wsj0_2mix` (`54G`, about `336000` wav files),
  `/data/datasets/Libri-mixture-noisy`, `TextrolMix`, and
  `TextrolMix_LibriFormat`.
- `/data/datasets/wsj0_ori` and `/data/datasets/wsj0_extracted` contain WSJ0
  `.wv1/.wv2` files. Prefer already converted wav data unless the adapter
  explicitly handles WSJ sphere/compressed formats.
- `/data/datasets/JAEGER/librispeech` currently appears to contain annotations
  such as `train-clean-100_ann_librispeech.json`, not local audio files.
- The local Objaverse maps under `/data/datasets/jzy/assets/objaverse` may
  contain stale `/home/jzy/.objaverse/...` paths. Verify the referenced GLB
  exists before using a map entry; otherwise re-download by uid or prefer the
  already materialized Hunyuan assets below.
- Hunyuan AudioSet-style textured assets are already materialized under
  `/data/jzy/code/AVEngine/external/Hunyuan3D-2.1/outputs/audioset_assets`.
  That directory currently includes many animal candidates with
  `*_textured.glb`, `*_textured.jpg`, and turntables, e.g.
  `bird_animal`, `chicken_rooster`, `duck`, `frog_animal`, `goose`,
  `owl_animal`, `pig`, `sheep`, `snake_animal`, `turkey`, and `yak`.
  Prefer this path for static AudioSet source expansion before fetching random
  Objaverse assets.

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

Reusable source assets should be registered only after the visual asset,
direction, texture, rig, animation, and audio mapping review status is
approved. Registry manifests live under `external/SPEAR/data/source_assets_v1`.
Dataset specs should refer to `asset_id`, and the registry resolver should emit
legacy-compatible `tag` and `audio_lookup` fields for existing render/audio
code. Per-clip events record trajectory, visibility, and sound timing; do not
duplicate generation prompts, measured colors, or texture paths in event
metadata.

Hunyuan/Flux generation should write
`tmp/hy3d_batch/pending/{tag}/source_asset_candidate.json` as soon as a
candidate mesh is dropped into pending. Review UI approval carries this file
into `approved/{tag}` and updates its direction review status, but this still
does not make the asset production-ready. Only after texture, runtime mesh,
rig/animation, and audio mapping gates pass should the asset be copied into
`data/source_assets_v1`.

Review UI approval must sync `source_asset_candidate.json` after moving the
tag directory from `pending/{tag}` to `approved/{tag}`, not before. The sync
step should rewrite visual asset paths to repo-relative approved paths and
must handle both `mesh.obj` and `mesh.glb` as source meshes. If an approved
manifest still contains `tmp/hy3d_batch/pending/...` or absolute local paths in
`visual_assets`, resync it with
`tools/spike_rlr/source_asset_manifest.py::sync_candidate_manifest_review`
before promotion.

To check current animated-source classification status, run:
`/data/jzy/miniconda3/envs/spear-env/bin/python tools/spike_rlr/source_asset_audit.py --approved-dir tmp/hy3d_batch/approved --registry-root data/source_assets_v1`.
As of 2026-07-09, `dog_golden`, `dog_beagle_v2`,
`cat_british_shorthair_v2`, `dog_pug_v1`, and `cat_siamese_v1` are
classified and registered. The older
direction-only dirs `dog_beagle`, `dog_husky`, and
`cat_british_shorthair` were removed on 2026-07-09 because they were missing
texture/runtime proxy files and were not production assets. Do not reintroduce
those unsuffixed legacy tags into current source pools; production source
pools should use registry `asset_id` values such as `dog_beagle_0002`, with
the registry resolving the legacy-compatible runtime tag when needed.

After a tag is direction-approved in the review UI, run its runtime/UE gate
before registry promotion:

```bash
bash tools/gate_check_animal.sh {tag}
```

If the gate produces `GATE_CHECK_DONE /tmp/gate_check_v4/{tag}_side.mp4`,
promote it into the reusable asset registry with:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/spike_rlr/promote_source_asset.py --tag {tag}
```

`promote_source_asset.py` validates `mesh_runtime.glb`,
`mesh_runtime.json`, `mesh_oriented.glb`, `hy3d_diffuse.jpg`, and
human-approved `direction.json`; measures dominant colors from the diffuse
texture; writes `data/source_assets_v1/{category}/{family}/{asset_id}/asset.json`;
updates `data/source_assets_v1/registry.json`; and syncs the approved
`source_asset_candidate.json`. Do not edit the registry by hand for normal
Hunyuan dog/cat promotion unless the script itself is broken.

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

Rigged review actors should explicitly play their requested animation asset in
`tools/gpurir_scenes/run_render_pass.py`; do not rely on a Blueprint default
Walking state. If an animal appears to slide or stand still while moving, first
check `wanted_anim`, the presence of `gate_{tag}/{wanted_anim}.uasset`, and
that `_play_anim_on_actor()` called `PlayAnimation`.

For deterministic walking demos, also check trajectory speed before blaming the
animation asset. A 5 s clip that moves a dog 10+ meters can look like sliding
even when Walking is playing. Keep `motion_style="walking"` demo paths near
human-reviewable walking speeds, and add/keep tests that bound average path
speed when a builder is meant to show walking rather than running.

## Review Video Marker Rules

The marker in `side_by_side_review_annotated.mp4` should indicate the visible
actor, not just the acoustic source point. Prefer UE-authored
`videos/actor_visual_metadata.json` and its
`visual_center_world_xyz_per_frame` values when present.

The acoustic point can be near the feet/floor or center trajectory and may not
overlap the rendered mesh body. That is expected; the review marker should use
the visual center when available.

`source_visible_from_camera_per_frame` is a center-point ray/FOV metric, not a
whole-mesh visibility metric. In apartment kitchen views, low animal centers
can be ray-occluded by the hand-authored counter bboxes while the head/body is
still visibly rendered. For review demos, report both center-FOV and
center-visible counts, and do not use center-visible alone to decide whether an
animal is visually reviewable.
Review overlays should label this as `centerVis`, not generic `vis`, so it is
not mistaken for whole-animal visibility.

Review overlays also report `sound N/T` per source. This is derived from the
per-source rendered binaural/wet signal in
`source_effective_audio_per_frame`, using the metadata threshold, and is meant
to answer "how many frames does this source effectively make audible sound?"
Muted or `audio_lookup: "silent"` sources should report `sound 0/T`.

Animal dry audio for RLR review scenes should resolve through
`external/SPEAR/tools/spike_rlr/animal_audio.py`. Do not hardcode a dog/cat tag
to a synthetic debug tone unless the spec explicitly asks for a sentinel such
as `__piano_scale__`. Current real review lookups include `dog_bark`,
`dog_growl`, `dog_sharp_bark`, `cat_meow`, and `cat_purring`.

For visual-only review actors, set both `mute_audio: true` and
`audio_lookup: "silent"`. The RLR pass should skip their RIR/audio render, the
metadata gain should stay at zero, and no per-source solo wav should be written.
Do not leave a front review animal on a real dog/cat lookup when the demo is
meant to test only a rear source.

For spatial motion demos that need listeners to hear the same event moving
through space, use the dry-source clip controls (`audio_clip_start_s`,
`audio_clip_duration_s`, `audio_repeat_interval_s`) to repeat a short real
animal vocalization. A long field recording can contain different barks/meows
over time and makes left/right motion sound like changing sources.

Deterministic event/demo construction should use
`external/SPEAR/tools/spike_rlr/event_constraints.py` and
`external/SPEAR/tools/spike_rlr/demo_scenarios.py`. For example,
`compose_front_idle_rear_left_to_right_demo()` builds the "front idle dog +
rear invisible listener-left-to-right dog" case by writing explicit
trajectories and then verifying constraints. Do not satisfy those flags by
random seed search.

For review/demo source placement, distinguish asset-body clearance from
source-center validity. `source_collision_policy: "walls_only_center"` means a
source center must stay inside the valid room regions and must not enter shell
wall bboxes, but furniture/body-radius clipping is tolerated. Use this for
point-source event demos where narrow passages should remain available; do not
use it to allow wall or outdoor crossings.

`flags.json` is the backward-compatible clip-level aggregate flag dict used by
dataset coverage. Deterministic demos should also write `flag_details.json`
with `aggregate`, `per_source`, and `pairwise` sections so review overlays can
show that, for example, one source is stationary while another is walking. Do
not infer per-source semantics from aggregate flags alone.

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

- Habitat's GLB loader imports our Z-up acoustic meshes as `(x, z, -y)`.
  `_habitat_from_scene(x, y, z)` must therefore map to `(x, z, -y)`, not
  `(x, z, y)`. If this sign is wrong, valid apartment sources can be mirrored
  into the wrong RLR/navmesh component and render nearly silent IRs.
- `AudioSensorSpec.position` defaults to `[0, 1.5, 0]`; set it explicitly to
  `[0, 0, 0]` because `spec["mic"]["pos_m"]` is already the listener position.
- The listener orientation must track `spec["mic"]["yaw_deg"]` or
  `camera_configs[0]["yaw_deg"]`; do not hardcode the listener to face
  `+Y_scene`.
- Current helper convention: Habitat agent yaw is
  `(scene_yaw_deg - 90) % 360`.
- With the `(x, z, -y)` transform, RLR native binaural is already `[left,
  right]`; do not apply the old L/R swap compensation.
- Do not use world-axis assumptions such as "world +X is always right ear" for
  random-yaw clips. Audio left/right must be listener-local.
- Do not judge RLR binaural azimuth from broadband left/right RMS alone.
  Apartment reflections and low-frequency speech can make a true left-to-right
  path look mostly left-biased in full-band RMS. Check metadata azimuth and
  band-limited ILD, especially `1500-5000 Hz` and `5000-7500 Hz`, before
  declaring the spatial direction wrong.

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
