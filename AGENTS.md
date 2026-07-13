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
- External Rocketbox avatar checkout:
  `/data/datasets/rocketbox/Microsoft-Rocketbox`.
- External CMU Mocap official ASF/AMC archive root:
  `/data/datasets/cmu_mocap/raw`.
- External OSU ACCAD Open Motion Project archive root:
  `/data/datasets/accad_mocap/raw`.
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

## Git Milestone Discipline

- AVEngine and `external/SPEAR` are independent Git repositories. Commit in
  the repository that owns the changed file; do not expect the AVEngine root
  to track SPEAR because `external/*` is ignored there.
- After each independently verified pipeline milestone, commit only the files
  owned by that milestone and push immediately. AVEngine uses `origin`; the
  active SPEAR feature branch uses `eastforward`.
- Both worktrees may contain older user or parallel-task changes. Never use
  broad `git add -A`, reset, checkout, clean, or rollback to make a commit look
  tidy. Stage explicit paths, run the relevant tests and `git diff --cached
  --check`, then inspect the staged stat before committing.
- Generated model/media data and authenticated manifests under `tmp/` remain
  outside Git. Commit their schemas, runners, compact human decisions, and
  documentation; keep large immutable evidence referenced by path and hash.

## Python Environments

- `spear-env`: SPEAR/UE rendering, review-video builder, most lightweight
  SPEAR-side tests.
- `ss2`: Habitat/RLR audio, `trimesh`, auto-orient ingest, review UI mesh
  processing, ReplicaCAD Habitat smoke checks. It also provides Flask for the
  human/Rocketbox browser-review server tests; `spear-env` does not currently
  include Flask, so run those server test files under `ss2` rather than treating
  `ModuleNotFoundError: flask` as a product regression.
- `hunyuan3d`: Hunyuan3D shape and paint generation only.
- `avengine-imagegen`: local open-weight image generation/editing probes
  cloned from `comfyui` on 2026-07-10, with torch 2.7.1/cu126 and GitHub
  diffusers 0.40.0.dev0. Use this for Qwen-Image-2512, Qwen-Image-Edit,
  and FLUX.2 Klein experiments; do not upgrade or repurpose `hunyuan3d` for
  those models.

The system `node` is currently v12 and is too old for the Playwright package
used by Rocketbox browser-review QA. Use the current VS Code server Node binary
under `/data/jzy/.vscode-server/bin/*/node` (v24 as of 2026-07-10). If the
Playwright Chromium installer stalls after finishing its zip download, verify
the cached zip with `unzip -t` and make sure the browser is complete under
`/data/jzy/.cache/ms-playwright`; do not treat a partially extracted browser
directory as a successful install. Launch QA with an explicit Chromium
`executablePath` when the headless-shell companion is not installed.

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
  As of 2026-07-10 that directory contains `T-Pose.fbx`,
  `Standing_Idle.fbx`, `Walking.fbx`, and `Walking_arm_{60,70,80,90}.fbx`;
  do not assume a `Running.fbx` exists.
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
- A strict T-pose reference is necessary but not sufficient for production
  human assets. On 2026-07-09, `human_male_blue_hoodie_v2` and
  `human_female_red_jacket_v2` passed coarse orientation/runtime checks but
  failed hand quality in moving review: the hands rendered as broken/empty
  white pieces after Flux/Hunyuan full-body mesh generation plus nearest
  Mixamo skin transfer. The Mixamo armature itself had hand/finger bones and
  the runtime GLBs had nonzero hand weights, but diagnostics showed the
  transferred hand region was dominated by a few thumb/finger groups instead
  of normal hand/finger distribution. Treat this as a source-topology plus
  skin-weight-transfer mismatch, not as missing bones or a UE floor/scale
  issue. Human promotion must include a close hand/arm review on the animated
  runtime GLB and a UE moving clip; reject if hands are not visually connected
  and solid. Prefer a reliable rigged human base mesh with generated
  appearance/texture over Hunyuan-generated full-body topology for production
  humans.
- On 2026-07-09, temporary Hunyuan-body plus Mixamo hand/forearm/full-arm proxy
  experiments under `external/SPEAR/tmp/human_hand_debug` did not produce a
  production-quality fix. Hand-only proxies preserved sleeves but left detached
  or duplicated hands; forearm/full-arm proxies restored hand motion but looked
  like separate sleeve/arm overlays and still had bad shoulder/sleeve seams.
  Treat those proxy tests as diagnostics, not a recommended human asset path.
- On 2026-07-09, a Mixamo T-pose source spike under
  `external/SPEAR/tmp/human_mixamo_tpose_spike` used
  `/data/datasets/mixamo/raw/T-Pose.fbx` as the canonical rest-pose mesh and
  copied the `Walking.fbx` action afterward. The T-pose FBX imported with the
  same `Beta_Surface` rest mesh, 65-bone Mixamo armature, and 50 vertex groups
  as the Walking FBX, so it made the transfer reproducible but did not add new
  geometry/topology information. All Hunyuan target vertices matched, but the
  moving review still showed non-production hands because the Hunyuan hand and
  sleeve topology remains collapsed/ambiguous. A palm-only post-pass that moved
  all Thumb/Index/Middle/Ring/Pinky weights into LeftHand/RightHand did remove
  independent finger driving and is acceptable for smoke tests where fingers do
  not matter, but it does not make the Hunyuan full-body mesh a formal dataset
  source.
- On 2026-07-10, a short-sleeve human spike under
  `external/SPEAR/tmp/human_short_sleeve_spike` showed that Flux reference image
  resolution was not the primary cause of the previous arm failures. The
  reference images were generated at 1024x1024, but full-body people leave small
  pixel budgets for hands/wrists/leg gaps and Hunyuan paint later uses lower-res
  multiview baking, so tight composition and clean silhouettes matter more than
  simply increasing resolution. The stronger root causes observed locally were:
  (1) the inherited `--target-rotate-z-deg -90` was wrong for these Hunyuan human
  OBJ imports because both the Mixamo T-pose FBX and Hunyuan OBJ already use
  X=arm span, Y=body thickness, Z=height; use 0 degrees for this path, and
  (2) Flux/rembg/Hunyuan background or leg-gap artifacts can become real mesh
  geometry. `human_female_red_tshirt_v1` had a persistent leg-gap fan; the
  generated `human_female_red_tshirt_v2` had large flat floor/background cards
  that had to be removed with `cleanup_hy3d_ground_cards.py`.
- The same short-sleeve spike found that Mixamo arm-space variants
  `/data/datasets/mixamo/raw/Walking_arm_60.fbx`,
  `Walking_arm_70.fbx`, `Walking_arm_80.fbx`, and `Walking_arm_90.fbx` import
  cleanly with the current T-pose bind flow. Arm80/arm90 keep hands farther from
  the torso and are better review candidates than the default walking clip when
  diagnosing hand/forearm contact. The best temporary review outputs are
  `short_sleeve_v2_cleaner_upright_review_arm80_*.mp4` and
  `short_sleeve_v2_cleaner_upright_review_arm90_*.mp4`; they are still only a
  technical spike because the female asset is stylized and the Hunyuan topology
  remains arbitrary. Do not promote these assets into the formal human registry.
- On 2026-07-10, the same short-sleeve matrix was expanded to all current local
  Mixamo smoke actions: `Standing_Idle.fbx`, `Walking.fbx`, and
  `Walking_arm_{60,70,80,90}.fbx`. The generated review videos are under
  `external/SPEAR/tmp/human_short_sleeve_spike/` with prefix
  `short_sleeve_matrix_v2_cleaner_review_*`, plus contact sheets
  `short_sleeve_matrix_v2_cleaner_review_full_contact.png` and
  `short_sleeve_matrix_v2_cleaner_review_hands_contact.png`. These are review
  evidence for the spike only; the female v2 cleaner asset remains stylized and
  should not be promoted as a formal photoreal human.
- On 2026-07-09, broader human-source research reinforced that changing the
  animation source alone is not enough if the visual mesh has bad topology or
  skin weights. For production humans, target "stable bindable human base plus
  controllable identity/appearance/clothing/colors", not direct text-to-3D
  full-body runtime meshes. MetaHuman is the clearest architecture reference:
  stable MetaHuman topology, DNA carrying geometry/skeleton hierarchy/skin
  weights/RigLogic, reliable core/helper joints and solvers, facial
  blendshapes, and separate groom/clothing/material components. That stability
  is why identity and appearance can vary without re-solving arbitrary
  generated mesh topology for every character.
- Do not use MetaHuman as a formal AVEngine training/testing/evaluation dataset
  source. Current Epic/Unreal terms restrict using MetaHuman digital characters,
  animation curves, and certain rendered outputs to build or enhance databases
  or to train/test AI, ML, deep learning, or neural networks. The SPEAR
  `examples/control_metahumans_sample` path is still useful for internal smoke,
  demo, and reference work because SPEAR renders through Unreal, but MetaHuman
  assets must not be promoted into the production dataset registry without
  written rights that explicitly cover AVEngine's dataset and ML use.
- Clean baseline human routes: MakeHuman/MPFB core graphical assets are CC0
  while their application/add-on code is AGPL/GPL; Microsoft Rocketbox provides
  115 rigged avatars under MIT. These are suitable first formal baselines for
  pipeline validation and controllable identity/material/clothing experiments,
  but expect older topology, lower realism, and more manual wardrobe/material
  work. A Blender/CC0 base mesh plus procedural clothing/materials can follow
  the same license-clean baseline pattern. The local Quaternius human pack is
  also license-friendly but low-poly/cartoon, so use it as a baseline, not as
  the photoreal target.
- Motion sources are separate from visual human sources. CMU Mocap is the
  cleanest broad open motion baseline if ingesting from the official source and
  preserving its terms. AMASS/HumanML3D are useful research references but are
  constrained by AMASS/SMPL-style noncommercial and no-redistribution terms.
  Mixamo must remain smoke/demo only and must not be used for AVEngine formal
  training/testing/evaluation datasets: Adobe's Mixamo Additional Terms prohibit
  using the services, content, outputs, or derived information to create, train,
  test, or improve AI/ML systems. Standard ActorCore/Reallusion content terms
  likewise need written or enterprise terms explicitly allowing dataset
  generation plus AI/ML training/testing/evaluation use.
- For production human motion, prefer CMU Mocap or self-captured/commissioned
  mocap with actor releases and contracts that explicitly allow synthetic
  dataset generation plus AI/ML training/testing/evaluation. Rokoko Studio can
  export FBX/BVH for self-captured data, but do not treat Rokoko Motion Library
  or motion-dataset products as automatically licensed for AVEngine training
  data without a written data/AI rights agreement.
- OSU ACCAD Open Motion Project is a useful second formal motion baseline
  after CMU. The official ACCAD MoCap System and Data page lists downloadable
  walking/running/general/martial-arts motion archives in C3D/BVH/ASF-AMC/TXT
  forms and licenses the Open Motion Project under Creative Commons Attribution
  3.0 Unported. Use `/data/datasets/accad_mocap/raw` for local mirrors and keep
  attribution/provenance with any converted clips.
- Good formal human baseline pairs are `CMU Mocap + Microsoft Rocketbox MIT`
  and `CMU Mocap + MakeHuman/MPFB CC0-generated avatars`. Keep MIT notices for
  Rocketbox and any CMU acknowledgement/redistribution terms; do not sell or
  redistribute CMU motions directly as a standalone converted motion pack.
  MakeHuman/MPFB's core graphical assets are CC0, but third-party asset packs
  still need separate license checks.
- As of 2026-07-10, formal-baseline downloads are tracked through the
  completed CMU archive, the completed ACCAD mirror, and tmux session
  `avdl_rocketbox_aria2`. Rocketbox's official source is
  `https://github.com/microsoft/Microsoft-Rocketbox.git`; CMU Mocap's official
  `allasfamc.zip` was downloaded from
  `http://mocap.cs.cmu.edu/allasfamc.zip` into
  `/data/datasets/cmu_mocap/raw/allasfamc.zip`. Logs for CMU/Rocketbox are
  under `/data/datasets/cmu_mocap/raw/logs` and
  `/data/datasets/rocketbox/raw`.
- The first Rocketbox `git clone --depth=1` failed on 2026-07-10 after a large
  Git pack transfer with `fetch-pack: unexpected disconnect` and an early EOF,
  leaving no checkout. A later `curl -L -C - --retry ...` codeload zip retry
  reached about 1.9GB and then failed with `HTTP/2 stream 0 was not closed
  cleanly: CANCEL`; curl then printed `Throwing away 2001289350 bytes` and
  restarted from the beginning. A `curl -r 0-1023` range test against
  `https://codeload.github.com/microsoft/Microsoft-Rocketbox/zip/refs/heads/master`
  downloaded megabytes instead of a 1KB range, so do not rely on codeload Range
  resume for this archive. The active retry path is tmux session
  `avdl_rocketbox_aria2`, downloading the official codeload archive to
  `/data/datasets/rocketbox/raw/Microsoft-Rocketbox-master.zip` with single-
  connection `aria2c`; the failed curl partial is preserved as
  `Microsoft-Rocketbox-master.zip.curl_partial`. Extract the verified zip under
  `/data/datasets/rocketbox/` before probing avatars. The official Rocketbox
  README describes 115 MIT rigged avatars and a 2022 addition of 417 compatible
  animations in `Assets/Animations`; verify the extracted files locally before
  treating those animations as a Mixamo replacement.
- A lightweight Rocketbox smoke subset was pulled directly from GitHub raw into
  `/data/datasets/rocketbox/sample` on 2026-07-10 so retarget/probe work can
  continue while the full zip downloads. It contains `LICENSE.md`, `README.md`,
  `Male_Adult_01.fbx`, `Female_Adult_01.fbx`, preview PNGs, and selected
  male/female `walk_neutral`, `walk_neutral_01`, `walk_fast_01`,
  `run_neutral`, and `idle_neutral_01` FBX animations. One interrupted male
  head texture was renamed to `m002_head_color.tga.partial`; the first smoke
  videos intentionally use simple preview materials rather than complete
  Rocketbox textures. Probe scripts live under
  `external/SPEAR/tmp/rocketbox_human_probe/`: `inspect_rocketbox_sample.py`
  found avatar FBXs import as one skinned mesh plus an 80-bone `Bip01` armature,
  while animation FBXs import as 121-bone `Bip01` skeletons with extra nub
  bones and matching core bone names. Directly assigning the skeletal action to
  the avatar armature works for same-family Rocketbox clips. Review outputs are
  `rocketbox_mf_walk_preview.mp4`,
  `rocketbox_mf_walk_contact_sheet.jpg`,
  `rocketbox_sample_action_matrix_preview.mp4`, and
  `rocketbox_sample_action_matrix_contact_sheet.jpg`; both videos are 4 seconds
  at 12fps and showed stable body/arm skinning for the sampled walk/run/idle
  actions. This is now the cleanest local human baseline candidate, subject to
  later texture/material, LOD, SPEAR import, and attribution packaging work.
- On 2026-07-10, the Rocketbox smoke subset was expanded from the official
  GitHub tree without waiting for the full zip. The tree cache is
  `external/SPEAR/tmp/human_motion_source_probe/rocketbox_tree.json`, and the
  selected locomotion manifest is
  `external/SPEAR/tmp/rocketbox_human_probe/rocketbox_locomotion_manifest.tsv`.
  It contains 68 raw FBX animations under `Assets/Animations`:
  24 female walk, 13 female run, 19 male walk, and 12 male run clips, totaling
  66,716,240 bytes. The 68 include `f_walk_neutral.max.fbx` and
  `m_walk_neutral.max.fbx`; after those two pass review, the unreviewed batch is
  66 actions, not another 68. All 68 files were downloaded under
  `/data/datasets/rocketbox/sample` and verified with `missing 0 bad_size 0`.
  `render_rocketbox_locomotion_pages.py` renders the verified set into
  `external/SPEAR/tmp/rocketbox_human_probe/locomotion_pages/` as 18 review
  pages at 1800x1200, 12fps, 72 frames, and 6 seconds each, plus first-frame
  PNGs and 6-frame contact sheets. The page videos use four actions per page
  and direct same-family action assignment to the Rocketbox `Bip01` armature;
  sampled review frames/contact sheets showed no Hunyuan-style arm tearing,
  holes, or torso insertions. Treat these as the current primary local human
  locomotion audit outputs before SPEAR dataset integration.
- Do not treat `external/SPEAR/tmp/rocketbox_human_probe/locomotion_pages/*.mp4`
  as an approved Rocketbox retarget result. Those pages attach animation
  actions directly before the source avatar, rest pose, and semantic FRONT -Y
  direction have passed a human review gate; the apparent travel/facing
  direction can therefore be reversed or ambiguous. They are negative/audit
  evidence only. The accepted sequence is source REST-pose review first, then
  one source-absolute, target-proportioned `walk_neutral`, then the remaining
  action batch only after both review checkpoints are explicitly approved.
- Verified Rocketbox source-review assets and pending review records live under
  `external/SPEAR/tmp/rocketbox_human_review/{rocketbox_male_adult_01,
  rocketbox_female_adult_01}`. Each root contains `front.png`, `back.png`,
  `left.png`, `right.png`, `top.png`, `face_close.png`, `arms_close.png`,
  `feet_close.png`, `source_views_contact.png`, `joint_close_contact.png`,
  `turntable.mp4`, `render_manifest.json`, and pending `source_review.json`.
  The renderer sentinel is `ROCKETBOX_SOURCE_REVIEW_RENDER_OK asset_id=<id>`;
  never infer approval from that sentinel or mutate the three pending statuses
  without the user's explicit review decision.
- As of 2026-07-10, the male and female Rocketbox source-review records have
  explicit user approval for geometry, official appearance, and `FRONT -Y`,
  and both source-absolute, target-proportioned `walk_neutral` reviews are also
  explicitly approved. The immutable reviewed motion baseline is sealed at
  `/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1`;
  its manifest allowlists 12 artifacts per avatar and records their SHA-256 and
  sizes. Re-sealing must be byte-identical and use no-replace publication.
  Source-animation skeleton/root/direction reports are internal diagnostics,
  not another user gate: never ask the user to approve an FBX, JSON report, or
  unbound skeleton. The requested production motion scope is now only
  `walk_neutral` for moving and `idle_neutral_01` for stationary. Do not batch
  the other 66 downloaded locomotion clips unless the user changes that scope.
- Rocketbox `*_walk_neutral.max.fbx` files import with the first animated walk
  frame encoded in `Bone.matrix_local`; it is not a canonical T/A-pose bind
  reference. Computing `inverse(source_rest_local) @ source_pose_local` makes
  frame 1 an identity delta and drops the asymmetric lifted-leg pose, which
  produced a backward-folded left foot while the right foot happened to look
  plausible. For same-family Rocketbox retargeting, transfer the 22 body bones'
  absolute armature-space pose rotations, keep target proportions through the
  target rest translations, and leave facial/finger bones at target-local rest
  unless an action has intentional face/hand animation. After assigning each
  parent `PoseBone.matrix`, call `view_layer.update()` before using that matrix
  for a child; otherwise Blender returns the stale parent state and corrupts
  the downstream limb. The male/female `Foot` and `Toe0` direction gate must
  reconstruct the source at every frame, not merely check floor penetration.
- Rocketbox source textures must come from the official Git tree and pass both
  recorded byte-size and Git-blob SHA-1 checks before rendering. Missing body
  color, head color, or opacity color is a hard failure; do not substitute a
  preview color or flat material. The official Rocketbox opacity material uses
  texture color luminance as Principled alpha rather than the TGA alpha channel
  for these sampled avatars; preserving that graph is required to avoid opaque
  hair cards.
- The sampled Rocketbox avatars expose only three coarse material slots:
  `body`, `head`, and `opacity`. They do not ship semantic masks for shirt,
  pants, shoes, skin, or hair. Any text-to-parameter recoloring system must
  author and QA those masks on each stable topology first; do not claim that
  those garment-level masks are already present in the official assets.
- Blender orthographic `Camera.ortho_scale` follows the larger frame dimension:
  a unit 1280x720 frame is `1.0 x 0.5625`, while a unit 1200x1600 frame is
  `0.75 x 1.0`. Frame Rocketbox reviews from `Camera.view_frame()` rather than
  assuming `ortho_scale` is always vertical, or landscape turntables will crop
  the avatar and place camera-fixed labels outside the image.
- As of 2026-07-10, the CMU official ASF/AMC archive
  `/data/datasets/cmu_mocap/raw/allasfamc.zip` is complete and passed
  `unzip -tq`. The zip contains 2740 files: 2514 `.amc` motions and 112
  `.asf` skeletons. CMU metadata/inventory probe outputs live under
  `external/SPEAR/tmp/cmu_mocap_probe/`: `build_cmu_inventory.py` reads the
  zip, caches official subject search pages under `subject_pages/`, and writes
  `cmu_motion_inventory.json`, `cmu_motion_inventory.tsv`, and
  `cmu_motion_inventory_summary.json`. The 2026-07-10 inventory parsed 2401
  official-described motions with no missing AMC/ASF links and classified
  492 `walk`, 26 `walk_transition`, 126 `run`, 8 `transition_walk_run`, 35
  `stand_idle`, and 1714 `other` records. There are 113 additional
  archive-only `.amc` files not described by the official subject pages,
  concentrated in subjects 63, 73, 84, 117, and 121; prefer the
  official-described records for first AVEngine/SPEAR retargeting and keep
  archive-only files as a later manual-audit pool.
- As of 2026-07-10, the ACCAD mirror in `/data/datasets/accad_mocap/raw` has
  completed and all zips passed `unzip -tq`: `Female1Walking_c3d.zip` (30 C3D),
  `Female1Running_c3d.zip` (27 C3D), `Male1Walking_c3d.zip` (29 C3D),
  `Male1Running_c3d.zip` (27 C3D), `Male2Walking_c3d.zip` (35 C3D),
  `Male2Running_c3d.zip` (29 C3D), `Female1_bvh.zip` (81 BVH),
  `Male1_bvh.zip` (69 BVH), and `Male2_bvh.zip` (149 BVH).
- ACCAD inventory/probe outputs live under
  `external/SPEAR/tmp/accad_mocap_probe/`. The generated inventory files are
  `accad_motion_inventory.json` and `accad_motion_inventory.tsv`; as of
  2026-07-10 they classify 476 motions, including 67 pure BVH walk clips,
  54 pure BVH run clips, 74 pure C3D walk clips, and 55 pure C3D run clips
  after separating walk/run/stand transitions. The first Blender BVH sanity
  preview uses `Male1_B3_Walk.bvh`, imports as a 22-bone armature with source
  frame range 1-183, and renders
  `accad_male1_b3_walk_skeleton_preview.mp4`. ACCAD BVH coordinates span
  hundreds of units; set a large camera `clip_end` in Blender previews or the
  skeleton can render as an empty background even though import succeeded.
  A six-clip walking/turning audit video was also generated as
  `accad_walk_matrix_preview.mp4` from `Female1_B03_Walk1`,
  `Male1_B3_Walk`, `Male2_B3_Walk`, `Female1_B09_WalkTurnLeft90`,
  `Male1_B10_WalkTurnLeft45`, and `Male2_B15_WalkTurnAround`. The current
  1600x1000, 31-frame preview normalizes each clip by its full animated bounds;
  scaling only by body height clipped long walking trajectories out of frame.
  A broader 36-clip BVH walk/run audit was generated on the same date with
  helpers `prepare_accad_motion_pages.py` and `render_accad_motion_page.py`.
  It extracts selected BVHs to `motion_pages_bvhs/` and writes four 3x3 review
  videos: `accad_walk_core_a_preview.mp4`, `accad_walk_core_b_preview.mp4`,
  `accad_run_core_a_preview.mp4`, and `accad_run_core_b_preview.mp4`. Each page
  is 1800x1400, 31 frames, and every imported clip has the same 22-bone ACCAD
  armature. These videos are motion-source audit inputs for later Rocketbox,
  MakeHuman/MPFB, or Quaternius retargeting; they are not final rendered human
  dataset clips by themselves.
- Combined human locomotion candidate tables live under
  `external/SPEAR/tmp/human_motion_source_probe/`. The helper
  `select_walk_run_candidates.py` reads the CMU and ACCAD inventories and writes
  `human_walk_run_candidates.{json,tsv}` plus
  `human_walk_run_shortlist.{json,tsv}`. The 2026-07-10 run produced 947
  walk/run/stand candidates: CMU ASF/AMC contributes 492 walk, 26
  walk-transition, 126 run, 8 walk-run-transition, and 35 stand/idle records;
  ACCAD contributes 67 BVH walk, 54 BVH run, 10 BVH stand, 74 C3D walk, and
  55 C3D run records. The shortlist has 120 records, including 72 ACCAD BVH
  clips that are immediately usable in Blender previews and 48 CMU walk/run
  records that are license-clean but still need ASF/AMC conversion or retarget.
- On 2026-07-10, a Quaternius low-poly human baseline review was generated
  under `external/SPEAR/tmp/quaternius_human_baseline/` using
  `Casual_Male.fbx` and `Casual_Female.fbx` from the local Quaternius archive
  and their embedded `Walk` actions. Review videos are
  `quaternius_casual_pair_walk_full.mp4` and
  `quaternius_casual_pair_walk_close.mp4`. This validates a license-friendly,
  stable-rig baseline path, but the visual style remains low-poly/cartoon and
  should not be presented as the photoreal target.
- On 2026-07-10, human-motion license research found that formal AVEngine human
  datasets should prefer either (a) explicit commercial/ML data contracts such
  as Rokoko Motion Dataset or Reallusion/ActorCore Enterprise, or (b) clean
  open baselines such as CMU Mocap motions with Microsoft Rocketbox MIT or
  Quaternius CC0 human assets. Do not treat AMASS, BABEL, HumanML3D, KIT,
  HDM05, BMLrub, SFU, or LaFAN1 as commercial/formal dataset motion sources
  without separate rights; they are mostly research/noncommercial, no-resale, or
  inherit constrained upstream motion licenses. LaFAN1 is CC BY-NC-ND and
  should not be mixed into formal commercial training sets. Do not assume
  Unity/Unreal/Fab marketplace animation packs, Move.ai, DeepMotion, Plask,
  Mixamo, Rokoko Motion Library, or standard ActorCore exports can be used for
  AVEngine ML training or evaluation datasets without written authorization.
- Photoreal commercial human routes need explicit dataset/AI/ML rights, not
  just render rights. RenderPeople/HumanDataset-style custom licensing and
  Reallusion Enterprise AI/ML licensing are the most plausible commercial paths
  to investigate. Standard Humano3D terms prohibit using 3D models or derived
  outputs for AI training/datasets without prior written consent; Daz has AI
  restrictions; Human Generator/HumGen output is not enough unless written
  terms cover synthetic dataset generation, rendered dataset redistribution,
  and AVEngine training/testing/evaluation.
- Commercial data-friendly candidates worth procurement/legal follow-up include
  Rokoko Human Motion Dataset, 3D PEOPLE DATA, Reallusion/ActorCore Enterprise,
  and custom RenderPeople/Humano3D agreements. Any purchase order or contract
  should explicitly grant synthetic rendering, training/fine-tuning,
  testing/evaluation/benchmark use, derived annotations, internal/subsidiary/
  contractor access, model commercialization, and sample publication rights,
  and should separately state whether rendered frames, skeleton/motion data, and
  mesh/texture data may be redistributed.
- As explicitly confirmed by the user on 2026-07-11, AVEngine's current release
  target is noncommercial academic research, a CVPR submission, and public
  research release. Track two separate license statuses: `research_release_ok`
  and `permissive_commercial_ok`. CC BY-NC and similar noncommercial sources may
  be used only when their terms also permit the intended AI training/testing and
  redistribution; label resulting data and weights as noncommercial research
  artifacts. A no-AI/no-training clause still disqualifies a source regardless
  of the project's noncommercial status. Do not describe an NC dataset as OSI
  open source; describe it as publicly released for noncommercial research.
- The active Unreal installation is UE 5.5.4 at `/data/UE_5.5`, not UE 5.4.
  Under the current Unreal Engine EULA, rendered images/videos are Non-Engine
  Products and academic/noncommercial or educational use can qualify for the
  seat exceptions. Open releases may include AVEngine/SPEAR scripts and rendered
  outputs, but must not bundle Unreal Engine code/binaries, Starter Content in
  source form, or separately licensed Epic/Fab assets. The EULA's prohibition
  on feeding Licensed Technology itself to generative AI and its separate
  MetaHuman database/training/testing ban remain hard exclusions.
- SPEAR code is MIT and the SPEAR repository states that its assets are CC0.
  Habitat-Sim code is MIT; the installed `ss2` build is Habitat-Sim 0.2.2. Its
  bundled RLR-Audio-Propagation component is separately CC BY-NC, so it is
  acceptable for the current noncommercial research scope with attribution but
  is not a permissive commercial dependency.
- ReplicaCAD licensing needs snapshot-level provenance. The current official AI
  Habitat page and Hugging Face cards identify both interactive and baked-light
  ReplicaCAD as CC BY 4.0, but the locally downloaded 1.5 trees currently contain
  `LICENSE.txt` files stating CC BY-NC 4.0. Until a current official snapshot is
  downloaded, hashed, and matched to its license, conservatively classify local
  ReplicaCAD-derived renders as CC BY-NC research-only. Public release must
  preserve attribution, link the applicable license, identify modifications,
  and propagate the noncommercial restriction. Do not silently replace the
  bundled local license based only on a web page.
- SMPL-family models (SMPL, SMPL-X, STAR, SUPR, Meshcapade) are technically
  aligned with the desired stable-topology/parameterized-body route, but their
  free licenses are generally noncommercial research/education/artistic and
  restrict redistribution and some AI/ML uses. Treat them as Route C only after
  obtaining the right commercial/ML license, and budget separate work for
  clothing, textures, hair, face detail, rig export, and SPEAR import.
- Research clothed-human generators and reconstructors such as ECON, ICON,
  TeCH, GETAvatar, HumanGaussian, HumanNorm, SHERF, InstantAvatar, PIFu, and
  PIFuHD are reference material, not direct AVEngine production sources. Many
  reconstruct one person from a photo/video, depend on SMPL/SMPL-X or datasets
  such as THuman/RenderPeople/AGORA/HuMMan/ZJU-Mocap, use diffusion/image-prior
  chains, or carry noncommercial/no-license constraints.
- FLUX.2 Klein 4B may supply formal reference images and low-drift
  clothing/color edits under its Apache-2.0 license. Hunyuan outputs may be
  retained only as rejected/internal comparison evidence, not as formal
  geometry, texture, or rendered training data. Keep
  `human_male_blue_hoodie_v2` and `human_female_red_jacket_v2` rejected unless
  a stable rigged base mesh and clean skinning path replace their generated
  Hunyuan body topology.
- The 2026-07-11 image-to-3D audit found no one-stage model that simultaneously
  provides prompt-controlled photoreal appearance, stable human topology,
  production skin weights, and an entirely permissive stock inference stack.
  Keep those concerns separate: use an image-to-3D model as a geometry/PBR
  donor, then fit or skin a stable internal human template and run animated QA.
- Pixal3D is the leading Hunyuan replacement candidate for appearance and PBR
  geometry: its code and published weights are MIT and it preserves the input
  silhouette better than attention-only image conditioning. Its stock pipeline
  is not permissive-only as configured, however: `pipeline.json` names the
  noncommercial `briaai/RMBG-2.0`, uses DINOv3 under Meta's mutable custom
  license, and inherits TRELLIS.2's `nvdiffrast`/`nvdiffrec`-based render/export
  path. Its NOTICE still says DINOv2/Apache while the shipped pipeline actually
  names DINOv3, so do not rely on NOTICE alone. The current noncommercial
  academic project may use the NVIDIA research-only stack with attribution, but
  a permissive-only release path must replace it with Blender/xatlas or another
  permissive implementation. In either case, accept pre-masked RGBA and skip
  BRIA and optional camera estimation.
- TRELLIS.2 code and 4B weights are MIT with no Hunyuan-style output-training
  restriction. The earlier "academic/research ambiguity" note was incorrect;
  the actual issue is that its default renderer/exporter depends on NVIDIA
  `nvdiffrast` and `nvdiffrec`, whose licenses limit use to noncommercial
  research/evaluation. That stack is acceptable for the current research-only
  scope but must be labeled as such; always bypass the noncommercial BRIA
  background remover by supplying a reviewed transparent RGBA input.
- The pinned image-to-3D caches are under `/data/models`: TRELLIS.2-4B revision
  `af44b45f2e35a493886929c6d786e563ec68364d` (22 files,
  16,237,485,044 bytes), Pixal3D revision
  `0b31f9160aa400719af409098bff7936a932f726` (19 files,
  24,044,888,779 bytes), and the DINOv3 ViT-L/16 mirror revision
  `3c276edd87d6f6e569ff0c4400e086807d0f3881` (6 files,
  1,212,584,680 bytes). The official Facebook DINOv3 repository returned HTTP
  403 for the current account, so the wrapper pins the `camenduru` mirror and
  its bundled DINOv3 license snapshot instead of silently falling back to the
  network. NAF is pinned under `/data/models/torch`; its checkpoint SHA-256 is
  `c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f`.
- The 2026-07-11 fixed-input 1024 bake-off uses the approved male/female FLUX.2
  RGBA references, seed 42, and manual Pixal FOV 0.2. Both TRELLIS.2 outputs and
  both Pixal3D outputs produced complete volumetric clothed humans with PBR
  materials and no floor/background card. After welding position-identical UV
  seam vertices, the largest connected component contains at least 99.84% of
  faces in every output. Keep the UV duplicates for texturing, but copy exactly
  the same skin weights to every position-identical seam vertex before
  animation. TRELLIS exports face toward `-Y`; Pixal's official post-transform
  exports face toward `+Y`, so review and retarget metadata must record the
  front axis explicitly.
- Preliminary visual QA favors TRELLIS.2 1024 as the first stable-template
  donor for these two people: its faces, silhouettes, and clothing surfaces are
  cleaner, while Pixal3D 1024 produces slightly fuller depth and sharper folds
  but more facial drift. A male Pixal3D 1536 probe added shoe/fabric detail but
  also introduced small shirt holes and did not improve the face reliably; do
  not make 1536 the batch default without a per-instance QA gate. Keep Pixal3D
  as the secondary donor until the user reviews the side-by-side page at
  `external/SPEAR/tmp/i23d_human_bakeoff_v1/review.html`.
- The stable Rocketbox follow-up for both accepted 1024 donors is under
  `external/SPEAR/tmp/i23d_rocketbox_template_fit_v1/{trellis2,pixal3d}/`.
  `tools/blender_fit_hy3d_to_rocketbox_template.py` now has a hash-locked I23D
  GLB input mode in addition to its legacy Hunyuan mode. It keeps the sealed
  Rocketbox topology, 80-bone skeleton, skin weights, walk, and official idle;
  TRELLIS enters with `front_axis=negative-y`, while Pixal enters with
  `front_axis=positive-y` and is yaw-normalized 180 degrees before any body or
  floor alignment. I23D derivatives use
  `usage_scope: noncommercial_research_dataset_candidate`,
  `research_release_ok: true`, and `permissive_commercial_ok: false`; the
  Hunyuan branch remains hard-locked to `technical_spike_only`.
- Do not overstate what the current stable-template fit preserves. The runtime
  mesh is Rocketbox and the current texture stage samples donor PBR by human
  region, then recolors the official Rocketbox body/head textures while
  preserving their luminance detail. This proves stable animation, floor
  contact, and a reproducible color/PBR path, but it does not preserve exact
  donor identity, logos, garment geometry, or topology-level distinctions such
  as full-length trousers versus the template's shorts/leg layers. Prompt-level
  clothing and accessory control is not solved until direct multiview texture
  projection plus garment layers, or a separately validated generated-mesh/
  fitted-template binding route, passes the same animated gate.
- Long Blender review renders should run in a persistent tmux session. A
  foreground PTY render can be terminated when a new user message interrupts
  the tool call, leaving valid but unpublished dot-prefixed staging videos.
  Review media are trustworthy only after the renderer atomically publishes
  the canonical MP4 names and `review_manifest.json`; remove abandoned hidden
  staging files after a successful rerun.
- License-clean geometry controls are Direct3D-S2 (MIT code/weights; image to
  high-resolution OBJ, no texture/PBR), Stable3DGen/Hi3DGen (MIT and explicitly
  stripped of kaolin/nvdiffrast/FlexiCubes), and TripoSR (MIT code/weights;
  lower-quality textured baseline). Use the same approved transparent FLUX.2
  input for comparisons; do not mistake a geometry-only win for a complete PBR
  replacement.
- MHR is an Apache-2.0 parametric human base with identity, pose, expression,
  seven LODs, a skinned skeleton, and FBX assets. SAM 3D Body can estimate MHR
  body parameters from a single image and its current SAM License does not ban
  using outputs to train another AI, but the checkpoint is gated and Meta may
  update the license. Treat SAM 3D Body as a pinned-license conditional body
  anchor, while MHR itself is the preferred permissive stable template.
- SkinTokens code and checkpoints are published under MIT and its
  `--use_skeleton` path preserves an input skeleton while generating skin
  weights. It is the preferred learned binding probe after inserting a fitted
  Rocketbox or MHR skeleton into a generated mesh; it does not eliminate the
  need for joint-placement checks, motion retargeting, or animated foot/arm QA.
  Save checkpoint hashes and license snapshots because its published training
  provenance includes Articulation-XL, VRoid Hub, and ModelsResource assets.
- As explicitly selected by the user on 2026-07-12, route 2 is the highest
  priority and Pixal3D is the default image-to-3D backend. This supersedes the
  earlier bake-off's provisional TRELLIS-first recommendation; do not add a
  new donor-selection gate or download another image model. The male canary is
  the original packed-PBR GLB
  `external/SPEAR/tmp/i23d_human_bakeoff_v1/pixal3d/rocketbox_male_adult_01/canary_1024_seed42.glb`
  (SHA-256
  `1df2490d6b83e52fa3b7c4e9d6b69207fa59cad0deae80e3dc3f894dfc443c42`).
  Its visible contract is a solid green short-sleeve shirt, gray long trousers,
  and gray shoes. Plaid, shorts, and black lower legs belong to the stable
  Rocketbox replacement template and are animation evidence only; they must
  never be described or registered as Pixal geometry.
- Directly transferring Rocketbox weights to that Pixal mesh is rejected. The
  preserved failure record is
  `external/SPEAR/tmp/i23d_rocketbox_direct_bind_v1/pixal3d/rocketbox_male_adult_01/floor_failure.json`:
  fixed floor `-0.004898416 m`, right-foot minimum `0.040251061 m`, or about
  `4.515 cm` unsupported clearance. The root cause is the Pixal forward-leaning,
  bent-knee rest body not matching the Rocketbox rest skeleton. Pixal FOV
  `0.2`, `0.35`, and `0.5` were also tested under
  `external/SPEAR/tmp/pixal_bind_pose_refinement_v1`; do not use further FOV
  changes as a rigging fix. Do not use the geometry-only `cleaned.obj` or swap
  the Rocketbox body back into route 2.
- The active learned-binding continuation is the original Pixal PBR GLB ->
  TokenRig `--use_transfer` -> static skeleton/weight/PBR gate -> approved
  Rocketbox Walk/Idle retarget -> five-view videos -> browser approval. Static
  failure blocks animation and permits exactly the fitted-skeleton
  `--use_skeleton --use_transfer` fallback. On 2026-07-12 the user authorized
  Codex to inspect the actual pixels/videos and continue without waiting at
  visual gates. Record such decisions as
  `agent_qa_passed_pending_user_acceptance`, never as user approval. Complete
  all of route 2 first and publish one consolidated next-day acceptance page;
  then complete route 1 and every remaining human/animal/mixed phase in order.
- The single permitted male direct TokenRig inference ran on 2026-07-12 with
  seed `42` and physical GPU 3. TokenRig itself returned zero and produced the
  preserved 50,843,552-byte GLB (SHA-256
  `8606c013fba02f722e1d5c65accddc4398eab1fa925467a9233aaf458d93f01c`),
  but the Task-3 publication gate correctly failed because the injected
  `sitecustomize` could not import `src` during Python startup. The exact error
  was two `ModuleNotFoundError: No module named 'src'` records: the injected
  `PYTHONPATH` contained the runtime patch but omitted the SkinTokens checkout
  root. Do not rerun the direct inference. Its immutable attempt ledger is
  `external/SPEAR/tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01.tokenrig_attempt.json`
  and the failed staging evidence is the sibling
  `rocketbox_male_adult_01.tokenrig_failed_attempt/`.
- The Task-3 failure itself was an orchestration-evidence failure rather than a
  model-process failure; the later independent static-rig rejection is recorded
  in the next bullet.
  The pinned upstream `BpyParser.load` independently calls `clean_bpy()` before
  each import. Read-only inspection of the preserved GLB found one real Pixal
  mesh, one 52-joint skin, unchanged `976,951` polygons and exact embedded WebP
  PBR bytes. All `707,091` serialized vertices have 1--4 normalized influences
  (maximum sum error `1.4738179743289948e-7`), and all `180,809` coincident UV
  seam groups have identical weights. Continue it only through the explicit
  `research_candidate_recovered_from_hygiene_assertion` manifest and the full
  Task-4 static/pixel gate; never relabel Task 3 as passed. Future fallback,
  female, and attribute runs must put both the runtime patch and exact
  SkinTokens root on `PYTHONPATH` and prove the two clean/load sequences.
- The preserved direct TokenRig male result subsequently failed the real
  static weight gate and must not enter animation. Two immutable Task-4 failure
  records live beside the recovery manifest. The first records the deliberately
  strict raw-triangle check (`976,970` source triangles versus `976,951`
  TokenRig triangles). Follow-up measurement proved those 19 removed triangles
  are reverse-coincident micro-faces with total area `1.0340533e-6 m^2`; the
  undirected surface, positions, UVs, and embedded PBR bytes match, while
  normals were reserialized (p99 vector error `2.69975e-4`, maximum
  `0.01274973`). That bounded serialization difference alone is not the final
  rejection reason and must be reported rather than described as exact
  topology/normal preservation. The decisive second failure is real bilateral
  skin contamination: among `285,249` distal vertices considered, `55,593`
  exceed `1e-4` opposite-side weight, `1,081` exceed 1%, and `326` exceed 5%; a
  left-toe vertex carries `0.07783098` right-toe weight. Keep the direct result
  `rejected`, keep animation blocked, and use the one authorized fitted-skeleton
  `--use_skeleton --use_transfer` fallback. Do not weaken this weight failure
  into a visual-only pass.
- The one authorized male fitted-skeleton fallback ran on physical GPU 3 with
  seed 42 and both `--use_skeleton` and `--use_transfer`. It conditions on the
  direct result's authenticated 52-joint skeleton while retaining the same
  surface-equivalent Pixal PBR derivative; it does not use Rocketbox mesh or
  weights. The attempt ledger succeeded with return code zero. Its output is
  `rocketbox_male_adult_01/fitted_skeleton_v1/tokenrig_transfer.glb`, SHA-256
  `eb9566f091b6de5357375dee750e66a48bcf4b12ba97a87615c26bed4cf77017`,
  size `50,972,456` bytes; its fitted manifest SHA-256 is
  `f2be8c719ea5049b76efc77220af5ae686e72c50913acbe85b7555276a506e56`.
  This remains a research candidate with `animation_authorized=false` until
  the fitted static audit proves the unchanged bilateral-weight gate, PBR,
  hierarchy, surface, grounding, and visual checks. Never infer success from
  the TokenRig return code alone.
- The fitted-skeleton output also failed its unchanged static gate and remains
  `rejected`. After tolerance-aware surface equivalence, the ordered audit
  found UV-seam vertices 746/6041 at the same position with weight L1
  difference `2.731915010372177e-4` (limit `1e-6`). An independent invocation
  of the unchanged bilateral validator then found `55,289` contaminated distal
  vertices with maximum opposite-side weight `0.11449641734361649`, worse than
  the direct result. The ordered immutable failure is
  `fitted_skeleton_v1/static_audit_v1.failed.8641b5fa58af4abeaf016b4276148ecc.json`
  (SHA-256
  `1b3a11c0708ffe2b70f2c363d7617ca4437deab0503cbe165cac9e7c1d0366e4`).
  Do not animate it. The next non-inference step is a deterministic,
  provenance-recorded TokenRig weight sanitation that transfers only clearly
  distal opposite-side core-chain influence to the topology-matched same-side
  bone and reconciles identical-position seam duplicates. It must leave the
  Pixal mesh, rest skeleton, and PBR unchanged and pass the original seam and
  bilateral validators before any animation.
- The deterministic fitted-weight sanitation has now completed without another
  TokenRig inference. The first three sanitation attempts remain immutable
  failures: the first exposed a surface-cluster false negative, the second
  replaced that comparison with round-binned skin evidence, and the third
  exposed Blender 4.2's hard `1e-4` export-influence cutoff. The fourth run
  used a support-preserving float32 floor for the single affected component
  (added mass `2.182787e-10`) and published the read-only branch
  `fitted_skeleton_v1/sanitized_weights_v1`. Its GLB is `51,056,520` bytes,
  SHA-256
  `2886c5a2c1768d1650598262f1ec5e84b9d78949c885ae1e4502f78c99f1570f`;
  its manifest SHA-256 is
  `a210ebe6b7af0cf045ba6fbc3434b0bd36bf81260193571f072871326b161549`.
  Of `709,573` vertices, `158,635` changed only through the documented
  same-side core-chain/seam procedure. All `181,326` seam groups are exact,
  bilateral contamination is zero at the original threshold, every vertex has
  at most four normalized weights, PBR bytes pass, round-trip skin L1 maximum
  is `2.189e-7`, and rest/IBM matrix round-trip errors remain near `1e-6`.
  Never edit or reuse the successful male v1 sanitizer implementation for
  female/attribute generalization; create separately reviewed v2 wrappers.
- The sanitized male static audit published the read-only nested branch
  `sanitized_weights_v1/static_audit_v1`. Its skinned `bind_pose.glb` is
  `51,101,040` bytes with SHA-256
  `1a85f2d22e6bdac230379bb57f389db7fc4c73a8f7c50f786e353374f89d6785`;
  `static_qa.json` SHA-256 is
  `31cd5bf745526913d2226efd180ca10b6623db1b34111f02ae4feef6feae8990`.
  The machine gate proves canonical FRONT `-Y`, floor zero, all 52 bones,
  exact seam/bilateral gates, PBR preservation, and round-trip skin L1 maximum
  `2.058e-7`. Codex inspected the bind Front/Back/Side/Top, weight-contact,
  texture comparison, hierarchy, and skeleton-overlay pixels and recorded
  `agent_qa_passed_pending_user_acceptance`. The visible colored rods outside
  the outer thighs in the skeleton overlay are Blender bone-tail/local-axis
  rendering; authenticated joint heads and child chains are inside the body.
  Recheck this explicitly in the dynamic Skeleton video rather than hiding it
  or treating it as user approval.
- The first two male Walk/Idle retarget publications failed closed and remain
  immutable. The first stopped before baking because a source clavicle rest
  matrix had only float serialization drift (`3.044e-6` orthogonality error,
  positive determinant); an SVD/polar projection with a strict `5e-6` cap was
  independently reviewed. The second reached the original grounding gate but
  rejected Walking frame 25: mesh vertex `77852`, weighted `92.573%` to the
  right toe and `7.404%` to the right foot, reached `-0.022312399 m`; constant
  grounding would require a `+22.312 mm` shift and therefore correctly exceeds
  the unchanged `10 mm` limit. Source right shoe clearance at the matching
  frame is `+3.330 mm`. Root motion scales correctly; the failure is a
  morphology/rest-axis pure-FK end-effector mismatch in the right leg/toe, not
  sampling, root/pelvis motion, or bilateral contamination. The permitted next
  step is a bounded bilateral two-bone leg/contact correction that leaves
  root, pelvis, and ankle XY unchanged; preserves segment lengths, knee plane,
  bend side, and original foot/toe global orientation; forbids stretch/flip;
  and reruns exact mesh, loop, deformation, PBR, and GLB-readback gates. Do not
  weaken the `1 cm` gate and do not publish media until this retarget branch
  passes.
- The second retarget stopped before saving `animated.blend`, exporting formal
  GLBs, or rendering review media, so it has no original attempt video. On
  2026-07-12 an explicitly non-formal diagnostic reconstruction re-ran the
  same authenticated rotation-only bake up to (but not through) the grounding
  rejection and reproduced the exact required correction
  `0.022312398999929428 m`. It is sealed under
  `external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/second_attempt_rotation_only_diagnostic_reconstruction_v1`.
  Front/Side/Feet are H.264 `640x360`, 30 fps, 33 frames, 1.1 s; their hashes
  are `0f7742f80bfc0a2fa7731eae25d006f6d53f1399bab8258bcdad27a9ca09ecc`,
  `3a5f959fe039da34d1f1b307420ac755e1e9572e4f409de6716a06737410b6e4`,
  and `15eac721a7bb48c96fbbc865cf222904082ee1ecd684a1e4f33a7a79736ad8e8`.
  `diagnostic_manifest.json` is
  `8cfca17b953b0fd7f76384fd9ff09520f5a28ce45db0b9e02f15c34a503c1eb5`;
  the independent sealed inventory is
  `bb0887987913b9f66fde64d459ef59e0a8a98e7bea5b408694f5b5c362495cc2`.
  The pixels visibly confirm a crossing,
  sideways/suspended gait and invalid support relationship. This directory is
  `technical_diagnostic_only`, not the missing original media and never a
  formal dataset asset. Its reused standard Feet view is a tighter full-body
  front-direction view, not a feet-only close-up; do not mislabel it.
- The Route-2 attribute/Pixal executor and jobs contract reached its non-GPU
  safe point on 2026-07-12. `human_attribute_pixal_contract.py` SHA-256 is
  `09a1a057ec6b4b3ef631cdeb29d1b624453b69d10c85b3a5ce8801d9ad17f3bf`;
  it binds its own executor bytes, an immutable started
  ledger, exact unique stdout sentinel and execution log, actual PBR/packed
  WebP GLB readback, persistent model inventory, and sealed failure bundles.
  The executor suite passed `22/22` and the full non-GPU attribute suite passed
  89 tests. `tmp/human_attribute_instances_v1/jobs_v2.json` is `0444`, SHA-256
  `59c04f7b60fa0b540359ecf2eedcffc0a941b1511fd6c8c0f583e96882c7ccfd`,
  preserves all seven ordered cases, and uses the corrected
  trousers `masks_v4`; `jobs_v1` remains historical and unchanged. No real
  FLUX/Pixal GPU attribute inference has run yet.
- `tools/route2_human_dag.py` now defines the immutable serial/resume ledger for
  exactly male, female, then the seven ordered attributes. It uses a no-replace
  SHA-256 event chain, binds `0444` evidence, never repeats a succeeded stage,
  advances after an attribute rejection, and refuses dependents after a base
  rejection. Qualified terminals are only
  `agent_qa_passed_pending_user_acceptance` or `rejected`; it cannot write
  `user_approved`. Its focused suite passes 12 tests, including a complete
  68-event/nine-qualified traversal and create/status/append CLI sentinels.
- The broader Route-2 owner contract is still under final hardening and must
  not be called ready yet. Independent review found three Important items:
  require an exact nested female `direct_female_lineage`, validate real GLB BIN
  ranges/accessors/data-URI image bytes rather than JSON references alone, and
  close or explicitly eliminate the transient swap-then-restore window between
  validated Python/wrapper/model paths and the bytes actually executed/loaded.
  Do not publish a real instance contract or start female/attribute GPU work
  until those owner fixes and their regressions pass.
- The 2026-07-12 takeover reverified the immutable Rocketbox baseline: all
  `24/24` managed hashes and sizes pass (`75,434,237` bytes), the top manifest
  SHA-256 is
  `b6e468e5f0c79d7ecec168e3c2460a7997a8d2916393da9add1ef2b6952fb922`,
  and the male/female official-material, `FRONT -Y`, neutral-walk approvals are
  intact. The Rocketbox partial clone is pinned at
  `0943055db6ec570bcef9f2c8b41c9e5467c808f9`. On 2026-07-12 its sparse checkout
  was completed directory by directory for Adults, Children, Professions,
  Animations, Animals, Editor, Source, Tools, and Docs plus the root license and
  readme. A no-lazy-fetch check read all `3,203/3,203` blobs, all tracked paths
  exist, and canonical non-facial avatar counts are exactly `115` total,
  `74` male/`41` female and `40` adults/`4` children/`71` professions. The
  checkout is clean and the MIT `LICENSE.md` SHA-256 is
  `17474e386e0b9e1a700cc3d06b2b0882a2c376d9c6b49c7f8274409b8f8d2352`.
  The corrupt 4,154,996,407-byte zip remains unchanged as failure evidence and
  is not a dependency.
- Local model verification on 2026-07-12 found no `/data/Models` tree and no
  incomplete/zero files in the active caches. FLUX.2 Klein revision
  `e7b7dc27f91deacad38e78976d1f2b499d76a294` has 25 files and
  `23,740,007,447` bytes; Pixal3D revision
  `0b31f9160aa400719af409098bff7936a932f726` has 19 files and
  `24,044,888,779` bytes. SkinTokens code is commit
  `273b691d35989d71cd17ff2895fdc735097b92d1`, weight revision
  `79736cad0fd84de384d5eede659b4ebd24effe33`, TokenRig checkpoint SHA-256
  `f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692`,
  and Skin VAE checkpoint SHA-256
  `4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4`.
  Its Python 3.11 venv keeps Torch `2.7.1+cu126` and now has transformers
  `5.13.1`, diffusers `0.39.0`, omegaconf `2.3.1`, lightning `2.6.5`, bpy
  `5.0.1`, trimesh `4.12.2`, open3d `0.19.0`, and flash_attn `2.8.3.post1`.
  All nine real imports passed with CUDA 12.6, and an offline load of the
  TokenRig + Skin VAE checkpoints on physical GPU 3 passed with vocabulary size
  `33036`. The later single direct male inference and its gate result are
  recorded above; do not repeat it.
- AniGen is attractive technically because it generates mesh, skeleton, and
  weights from one image, but its stock inference path loads DSINE under a
  noncommercial research license and imports `nvdiffrast` for mesh rendering
  and texture baking. Keep it conditional until those stages are replaced and
  inference quality is revalidated. Step1X-3D's full texture path is also not a
  clean workaround: it includes Hunyuan3D-2.0 renderer/baker code and
  `nvdiffrast`; only a separately audited geometry-only path may be reconsidered.
- Hunyuan3D-2.1, Hunyuan3D-2.5, Hunyuan3D-Omni, and the announced Hunyuan 3D
  3.0-style product/API route should not be interpreted as a fixed-topology
  humanoid avatar source. The newer Hunyuan releases improve PBR quality,
  control conditions, pose/skeleton conditioning, and surface quality, but the
  open pipeline still exports generated GLB/OBJ/trimesh-style assets without a
  guaranteed vertex order, humanoid skeleton, skin weights, blendshapes, or
  AVEngine-ready rig contract. Use them only as concept/reference/static mesh
  spikes unless a later route explicitly projects onto a stable internal human
  template and redoes UVs, skinning, LODs, collision, and animation QA.
- Hunyuan3D 2.0 and 2.1 are both excluded from formal AVEngine
  training/evaluation inputs as of 2026-07-11. Section 5(b) of both current
  public licenses prohibits using the Works or any Output/results to improve
  another AI model, and the licenses also contain territory restrictions.
  Hunyuan-generated meshes and the accepted Rocketbox/Hunyuan spike remain
  review evidence only and must never be promoted into the formal registry.
- The replacement experiment is: approved T/soft-T reference -> low-drift
  FLUX.2 Klein 4B edit -> transparent RGBA -> SAM 3D Body/MHR body anchor plus
  Pixal3D clean-wrapper geometry/PBR donor -> stable Rocketbox or MHR template
  fit -> fixed-skeleton SkinTokens weight probe -> approved Rocketbox
  walk/idle -> Blender and SPEAR animated review. Run Direct3D-S2 or Hi3DGen on
  the same image as a geometry control and TripoSR as the fully permissive
  low-quality fallback. Arbitrary generated topology is never accepted merely
  because one walk clip renders; formal promotion still requires stable rest
  pose, connected limbs, foot contact, turn/curve motion, material provenance,
  and multi-view QA.
- On 2026-07-10, image-model research for controllable human reference images
  favored local open-weight instruction editing over plain text-to-image. The
  next formal-candidate probe order is: (1) Qwen-Image-2512 for improved
  photoreal human-reference text-to-image plus Qwen-Image-Edit-2511 for
  identity-preserving image-to-image edits; (2) LongCat-Image-Edit and
  FireRed-Image-Edit-1.1 because their Hugging Face model cards list
  Apache-2.0 and focus on low-drift instruction editing, portrait/identity
  consistency, and reference-guided edits; (3) GLM-Image for
  identity-preserving people and object edits, or FLUX.2 Klein 4B when a
  smaller Apache-2.0 FLUX baseline is more valuable than maximum quality.
  HiDream-O1/E1 is useful but has extra dependency/license complexity because
  current releases depend on Llama-family components. OmniGen2 and
  BAGEL-7B-MoT are Apache-2.0 background probes, but they should not displace
  Qwen/LongCat/FireRed/GLM until they prove better identity and pose
  preservation locally. These are better fits for "start from a T-pose/person
  reference and change shirt color/clothing with minimal identity/pose drift"
  than regenerating a new person from a long prompt. Step1X-Edit remains an
  Apache-2.0 candidate for a later edit-only spike. Qwen-Image-2.0 is a strong
  announced/API/technical-report candidate, but as of 2026-07-10 there was no
  confirmed official open-weight Hugging Face snapshot to download locally, so
  do not treat it as the current AVEngine mainline until its weights/license
  are available. FLUX.1-dev, FLUX.1 Kontext dev, FLUX.2 dev, FLUX.2 Klein 9B,
  SD3 Medium, and Ideogram open weights are not clean formal-dataset inputs
  without separate commercial/synthetic-data rights because their public model
  licenses are noncommercial or restrict training/dataset-style uses. BFL also
  advertises explicit synthetic-data/training-output rights as a separate
  commercial licensing path, so do not infer those rights from FLUX
  noncommercial/dev licenses.
- CharacterGen is a human-specific Apache-2.0 candidate worth a separate spike
  because it targets canonical-pose 3D characters and VRM/Mixamo-style
  workflows, but do not assume it solves production topology or skin weights
  until its dependencies, output mesh, and retargeting quality are verified.
- For human reference generation, keep provenance per batch: model name and
  version, license URL or saved snapshot, prompt, input/reference image source,
  generation date, and whether outputs are intended for AVEngine training,
  testing, or only smoke review. When the task is simple clothing recolor or
  style variation, use image-to-image editing with a low-drift prompt and a
  mask/pose control if available; do not rely on a long text prompt to preserve
  identity, posture, hand gaps, and clothing details.
- This workstation sets `HF_ENDPOINT=https://hf-mirror.com` in the environment.
  That endpoint can resolve model metadata but fail on newer Hugging Face/Xet
  large-file downloads with `Distant resource does not seem to be on
  huggingface.co`. For Qwen-Image, FLUX.2, TRELLIS, and similar large model
  snapshots, override with `HF_ENDPOINT=https://huggingface.co` when using
  `huggingface_hub.snapshot_download`.
- For large Hugging Face/Xet snapshots, do not trust
  `snapshot_download(..., local_files_only=True)` alone as a completion check.
  On 2026-07-10 it returned a snapshot path for Qwen/FLUX.2/TRELLIS even though
  most safetensor blobs were still `.incomplete`. Verify completion by checking
  that expected large files under `snapshots/*` resolve to real blob sizes and
  that no relevant `.incomplete` files remain under the model's `blobs/`.
- Do not use `HF_HUB_ENABLE_HF_TRANSFER=1` for the current FLUX.2/Qwen large
  snapshot retries unless it is re-tested first. On 2026-07-10, installing
  `hf_transfer` and restarting the FLUX.2 Klein download caused one partially
  downloaded safetensor blob to reset to a zero-byte `.incomplete` file and then
  made no forward progress for about a minute. The slower plain
  `huggingface_hub` path with `HF_HUB_DISABLE_XET=1` was the path that actually
  resumed writes in this environment.
- Active image-model spike helpers live under
  `external/SPEAR/tmp/human_image_model_spike`. `check_hf_snapshot.py` verifies
  a model against official Hugging Face file lists and local `.incomplete`
  blobs. Its explicit `kind` values cover the current spike set:
  `flux2-klein-4b`, `qwen-image-2512`, `qwen-image-edit-2511`,
  `trellis2-4b`, `hidream-o1-image`, `glm-image`, `omnigen2`,
  `bagel-7b-mot`, `longcat-image-edit`, and
  `firered-image-edit-1.1`. These helper entries describe supported checks, not
  permission to resume every candidate download.
- As of the latest 2026-07-10 user instruction, the active local image-model
  scope is FLUX.2 Klein only. All Qwen, LongCat, HiDream, GLM, OmniGen2, BAGEL,
  FireRed, and other candidate download/probe watchers are stopped; there are
  no remaining `avdl_*` or `avprobe_*` tmux sessions. Do not revive them unless
  the user explicitly changes this scope.
- The complete FLUX.2 Klein 4B cache is canonical at
  `/data/models/hub/models--black-forest-labs--FLUX.2-klein-4B`. The default
  Hugging Face cache path under `/data/jzy/.cache/huggingface/hub` is an
  absolute compatibility symlink to that directory. Revision
  `e7b7dc27f91deacad38e78976d1f2b499d76a294` was verified against the official
  metadata as 25 files and 23,740,007,447 bytes with zero missing, incomplete,
  zero-size, size-mismatch, or LFS SHA-256-mismatch files. The Diffusers subset
  independently reports 18 required files, 15,980,131,745 cached bytes, and
  `SNAPSHOT_COMPLETE`. The root `flux-2-klein-4b.safetensors` is the official
  7,751,105,712-byte single-file checkpoint in addition to the complete
  Diffusers directory representation.
- No FLUX probe watcher is active. The current explicit image-edit batch is
  defined by `external/SPEAR/tmp/human_reference_review/jobs_v1.json` and must
  use the approved Rocketbox `front.png` images, their pinned SHA-256 values,
  the corresponding approved `source_review.json`, 1152x1536 output, 28 steps,
  guidance 1.0, and untruncated prompts with `max_sequence_length=512`.
  Generated `source.png`, `candidate.png`, manifests, and decisions belong only
  under `external/SPEAR/tmp/human_reference_review/<asset_id>/`. Both current
  image reviews must be explicitly approved before any Hunyuan3D invocation;
  never infer image approval from generation success or a manifest alone.
- The first hash-locked FLUX.2 image-edit pair was generated on 2026-07-10 and
  both exact candidates were approved by the user. The male candidate SHA-256 is
  `820abc0edb324bee570614cc901b03112589b28f3ea11e14d971788bc97a0938`
  (2,028,299 bytes, seed 41); the female candidate SHA-256 is
  `856df2ca3840cf74c9a48cb1ac2081fc0ac61700f5f2fb47aa4a37eb561fa03c`
  (2,145,192 bytes, seed 73). Both are 1152x1536 at 28 steps. The review
  server runs in tmux session `av_human_reference_review_8092` at
  `http://127.0.0.1:8092/`; its pair gate is valid only for the two exact
  approved candidate/source/manifest snapshots and must lock again if any hash
  changes. Browser
  QA screenshots are under
  `external/SPEAR/tmp/human_reference_review/browser_qa_{desktop,mobile}-*.png`.
- The accepted Hunyuan/Rocketbox technical-spike outputs are under
  `external/SPEAR/tmp/hy3d_rocketbox_template_fit_v1/{rocketbox_male_adult_01,rocketbox_female_adult_01}`.
  This path uses the stable Rocketbox topology, 80-bone skeleton, skin weights,
  UVs, and three material slots while fitting/projecting the generated Hunyuan
  geometry and appearance. It is the required fallback after arbitrary Hunyuan
  topology failed direct animation QA; do not replace it with the rejected
  direct-topology assets. Each asset has hash-locked `bound.blend`, Walk/Idle
  GLBs and Blender review media, plus one `ue_runtime.glb` containing exactly
  `Walking` and `Standing_Idle`.
- UE import for those two spike tags is performed by
  `external/SPEAR/tools/import_gate_humanoid_editor.py`; runtime spawning and
  apartment rendering are performed through the SPEAR API by
  `external/SPEAR/tools/spike_rlr/run_human_apartment_smoke.py`. The import
  manifest must report 80 bones, three non-null material slots, both exact
  animations, and a passed second-commandlet reload before the packaged runtime
  may render the actor. Re-cook/package SpearSim after changing imported assets.
- A packaged humanoid Blueprint can expose an empty inherited
  `USkeletalMeshComponent` before the populated imported component. Do not use
  SPEAR's singular `get_component_by_class()` for animation playback or bone
  evidence. Enumerate with `get_components_by_class()`, query `GetNumBones()`,
  and select the component with the largest positive bone count. Otherwise
  `PlayAnimation` and `GetBoneIndex` can silently target the wrong component.
- UE 5.5 Interchange sanitizes Rocketbox bone names when importing GLB: spaces
  become hyphens and the packaged 80-bone skeleton starts at `Bip01-Pelvis`
  rather than retaining a separate `Bip01` root. Runtime probes must inventory
  `GetBoneName()` values and normalize punctuation before querying transforms;
  do not hardcode Blender/GLB names such as `Bip01 Pelvis` as packaged UE names.
- A forced actor trajectory is not evidence that a humanoid visually faces its
  direction of travel. The rejected apartment v2 probe compared pelvis/root
  translation against the same trajectory used to place the actor, so it could
  pass while the mesh walked sideways. Runtime direction gates must separately
  derive a body basis from pelvis-to-spine up and left/right clavicle lateral
  vectors, compare that body's forward yaw with semantic travel, and compare
  pelvis translation with trajectory yaw. Also check the body basis for Idle,
  where root-motion direction is not applicable.
- In UE's body frame, `+X` is forward, `+Y` is anatomical right, and `+Z` is up.
  The correct shoulder/spine forward construction is therefore
  `right_vector x up_vector`. The v3 implementation used `up x right`, which
  points through the actor's back; its numerical gate passed while both actors
  visibly animated forward and translated backward. The user rejected v3.
- For the stable-template Rocketbox GLBs imported through UE Interchange, the
  corrected apartment SSOT uses `walking_forward_yaw_offset_deg: +90.0`.
  Runtime proof uses `Bip01-Pelvis`, `Bip01-Spine2`, and both clavicles and
  accepts body-forward error only within 25 degrees. An explicit
  `animation_play_rate` must be written to and read back from the populated
  skeletal component's `GlobalAnimRateScale`; do not accept a requested speed
  that UE did not actually apply.
- Final `apartment_0000` human clips need at least 120 streaming warmup frames
  followed by 40 camera-pose warmup frames. The earlier `20 + 10` canary budget
  produced low-mip brown floor frames and a temporary person-shaped floor/shadow
  artifact even though the humanoid mesh was valid. Do not diagnose that artifact
  as Hunyuan ground geometry: compare the apartment floor texture across early
  frames first. Final review outputs and the no-JSON review page are under
  `external/SPEAR/tmp/hy3d_rocketbox_template_fit_v1/ue_apartment_smoke/`, with
  final clip directories named `male_walk_final`, `male_idle_final`,
  `female_walk_final`, and `female_idle_final`.
- `external/SPEAR/tmp/hy3d_rocketbox_template_fit_v1/human_apartment_examples_v2/`
  is retained as rejected direction evidence. Its three clips pass the older
  root-trajectory checks but the stronger shoulder/pelvis basis found an
  approximately 81-degree body-vs-travel mismatch. Never present v2 for
  approval or use it as dataset evidence.
- `external/SPEAR/tmp/hy3d_rocketbox_template_fit_v1/human_apartment_examples_v3/`
  is also rejected diagnostic evidence. Its body basis used the reversed cross
  product and its `-90` degree offset made the measured back direction follow
  the path while the visible person walked backward. Its review page is marked
  rejected and must not be served for approval.
- The current corrected candidate is under
  `external/SPEAR/tmp/hy3d_rocketbox_template_fit_v1/human_apartment_examples_v4/`.
  It uses `right x up`, a `+90` degree Rocketbox offset, quadratic-Bezier paths
  with 46-52 degree tangent change, and a read-back-verified Walking rate of
  `0.65`. The three clips pass corrected body-forward/root-motion, upright,
  zero-penetration, FOV, obstacle, and source-separation gates; pair separation
  is at least 0.9 m. Serve its `review.html`; it remains a review candidate
  until the user explicitly approves it. The approved v1 batch remains
  immutable.
- These Hunyuan/Rocketbox actors remain `technical_spike_only`. Their successful
  Blender/UE animation and appearance checks do not resolve the Hunyuan
  output-use license restriction, so they must not be promoted into
  `external/SPEAR/data/source_assets_v1` or used as formal training/evaluation
  dataset sources.
- A cached FLUX.1-dev smoke on 2026-07-10 showed two prompt/probe traps. First,
  12 inference steps at 1024x1024 produced unusably blurred full-body reference
  images; use about 28 steps for any FLUX-style smoke reference before judging
  pose/detail. Second, "full body soft T-pose" alone can generate a rear view,
  especially for the female prompt. Human reference prompts and Qwen negatives
  should explicitly include `front-facing`, `face visible`, and `looking at
  camera`, and exclude `back view`/`rear view`. The smoke contact sheet is
  `external/SPEAR/tmp/human_image_model_spike/flux1dev_soft_tpose_smoke_contact.png`.
  The previous FLUX.2/Qwen watcher commands are now stopped under the FLUX-only
  scope. If a later FLUX.2 probe is explicitly started, retain the 28-step
  default and stronger front-facing prompt; do not revive older 8-step commands
  from shell history.
- Do not rely on Ready Player Me as a new dependency; public service
  availability changed after its Netflix acquisition.
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

## Second Route-2 Retarget Facing Review

The user resumed human-authoritative review for the rejected second male
retarget. Serve the immutable derived bundle at
`external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/second_attempt_facing_review_v1`
with `tools/spike_rlr/second_retarget_facing_review_server.py`. The default URL
is `http://127.0.0.1:8098/`.

This page is diagnostic only: it must remain `technical_diagnostic_only`,
`rejected`, and `formal_dataset_asset: false`; it has no approval endpoint.
Blue is the shoulder/hip-derived body-forward basis, red is independently
derived pelvis/root travel, grey is canonical `FRONT -Y`, and yellow is the
complete root trail. The bind sign is selected from the statically audited
FRONT -Y snapshot and never from travel.

For the exact second reconstruction, the skeleton-level automatic result is
aligned on all 33 frames (median body/travel dot `0.9945038453`, worst
`0.9518354934`, reversed/sideways ratios both zero). This does **not** prove
that the Pixal mesh visibly faces the semantic skeleton: the user's report is
that the person appears to walk sideways and may have been misbound from the
start. Treat the four synchronized videos and the user's visual decision as
authoritative for that mesh-versus-rig question; do not use the automatic dot
product to clear the rejected retarget.

The stronger gait-plane audit confirmed the user's observation. The sealed
pre-bind Rocketbox source is a normal sagittal walk: left/right foot
lateral-to-forward excursion ratios are `0.1306/0.1537`, and mean absolute
knee-plane-normal dot body-lateral values are `0.9947/0.9943`. The second
TokenRig result changes those ratios to `0.8385/0.9199`; its left/right mean
knee-plane-normal dot body-forward values rise to `0.8065/0.5981` (left
maximum `0.9981`). Therefore the source motion is not sideways; the retarget
stage rotated/corrupted the leg swing and knee-bend plane while body and root
still traveled forward.

The human-authoritative comparison is at
`tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/second_attempt_prebind_gait_review_v1`
and defaults to `http://127.0.0.1:8099/`. It presents sealed pre-bind
Front/Side/Top/source-skeleton media beside second-retarget
Front/Side/Feet/Top media. Future retarget attempts must add a pre-bind gait
plane gate and a post-retarget gait plane gate; body-forward/root-travel alone
is insufficient and must never authorize animation publication.

## Route-2 Shared Limb Motion-Basis Correction

The second reconstruction's error is not leg-only. The sealed source
left/right hand lateral-to-forward excursion ratios are `0.197017/0.197275`,
whereas the second TokenRig reconstruction changes them to
`1.126681/0.456778`. Its left elbow plane also rotates toward the forward
normal (`0.796236`) instead of the source's lateral normal (`0.849692`). Along
with the leg evidence above, this identifies the common failure as per-bone
rest-axis conjugation: the unrelated fitted TokenRig bone rolls rotate a
canonical Rocketbox pose delta differently for every arm and leg bone. Do not
patch only the feet, knees, or hands.

The pre-publication correction probe now transfers all bilateral clavicle,
arm, hand, thigh, calf, foot, and toe rotation deltas through one body-space
canonical basis; pelvis, root travel, spine, neck, head, mesh, PBR, and target
rest matrices remain locked. The exact four-candidate review bundle is
`external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/retarget_motion_basis_review_v1`,
and its manifest SHA-256 is
`c76c5405366a132bc72768addd5900e47c62b4eb9f1bf9f8fca1d7bf21ceb129`.
All 49 files are read-only. Across `0`, `-90`, `+90`, and `180` degrees, the
locked root/body, target-rest, and PBR hashes are respectively
`919d09c155eda470ef51851c49e8735f003118daf1e788a3f45c88be68053676`,
`eb21ef25871dae6517d8006cf83728cf2ecf742624e51eb6d1ac88ce89d4b90b`,
and `f0685d38cff3524db3b6c435f80568993aadbf210283c293cdd0b22c695900b6`.
Each candidate contains one re-imported GLB and synchronized Front, Side, Top,
Feet, and Skeleton H.264 videos at `640x360`, 30 fps, and 33 frames.

The automatic four-limb plane gate classifies only `yaw_000` as
`four_limb_sagittal_motion`. Its left/right arm lateral-to-forward ratios are
`0.169533/0.337697`, its left/right leg ratios are `0.232394/0.137535`, and
its knee-plane lateral-normal means are `0.988419/0.990453`. Codex also
inspected four-cycle-frame Front/Side/Feet/Skeleton strips and found forward
leg swing and front/back arm swing, so `yaw_000` is the current agent
recommendation, not user approval. Both `+/-90` candidates are sideways;
`180` visibly over-extends the arms and also fails the four-limb gate.

Serve the interactive exact-candidate page with
`tools/spike_rlr/retarget_motion_basis_review_server.py`; the active default
URL is `http://127.0.0.1:8100/`. The arrows switch the shared arm-and-leg basis
only and preserve synchronized playback in all five views. A user decision is
written once, with manifest binding, under
`retarget_motion_basis_selection_v1`; that directory must remain absent until
the reviewer clicks a decision. This page selects a parameter for the next
retarget and does not approve a formal asset. Do not publish a third formal
male retarget from an unrecorded or stale candidate.

The user correctly identified a separate upper-body lateral lean in the
motion-basis candidates. It is real pose deformation, not camera projection:
the static TokenRig bind has `0.7742 deg` body-lateral torso tilt, the sealed
Rocketbox Walk reaches only `1.5581 deg`, but the `yaw_000` candidate reaches
`10.4789 deg`; its shoulder roll reaches `7.8804 deg` versus `1.3160 deg` in
the source. All four basis candidates intentionally share the same locked
root/pelvis/spine/neck/head trajectory, so the arrow controls cannot change
this axial-chain failure. The limb basis may be selected independently, but
none of these candidates is a complete animation pass.

Read-only axial ablation located both contributors. Holding the target spine
at rest while retaining the current pelvis reduces maximum torso tilt to
`6.6146 deg`, while the hip line still reaches `6.5361 deg` roll versus
`2.6051 deg` in the source. Therefore the pelvis's old
`rest_aligned_global_rotation` rotates source yaw/pitch/roll components into
the unrelated fitted pelvis rest frame, and the cumulative source-local spine
resample then adds more target-local roll. Naively putting pelvis/neck/head
through the limb canonical transfer is not a fix: the pelvis-only probe worsens
hip roll to `11.2365 deg` and torso tilt to `12.0712 deg`. The permitted next
step is a separate hierarchical axial-body transfer that decomposes pelvis
motion in the authenticated body frame, preserves source lateral lean/hip and
shoulder roll bounds, and distributes swing/twist over the target spine. Keep
the current bundle `technical_diagnostic_only` and block formal grounding or
Walk/Idle publication until this new axial gate passes.

The separate hierarchical axial-body fix now passes its Walking canary. The
production runner constructs fitted-axis-independent anatomical frames from
the source world-space left/right hip chord, shoulder chord, pelvis/spine
segments, and neck-to-head chord; it removes the current target object rotation
exactly once. Clavicles use minimal child-direction alignment while preserving
the target rest twist. Head orientation deliberately uses the semantic
neck-to-head frame rather than the Rocketbox `Bip01 Head` technical Y axis,
which was proven capable of a 180-degree sign flip. This transfer owns pelvis,
all target spine controls, neck, head, and both clavicles; the reviewed shared
canonical motion basis continues to own the remaining arm and leg chains.

The read-only v2 review bundle is
`external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/retarget_motion_basis_and_axial_review_v2`;
its manifest SHA-256 is
`d04d2dceef4e3797d2973308ad7e0b22e158f84ed8150535a7f16bd32a11880f`.
All four candidates share corrected root/body SHA-256
`d8b3c64651b46ef2301585e1192015b0feaeec3a3ee00fc85fea19339891b66b`,
the unchanged target-rest hash
`eb21ef25871dae6517d8006cf83728cf2ecf742624e51eb6d1ac88ce89d4b90b`,
and unchanged PBR graph hash
`f0685d38cff3524db3b6c435f80568993aadbf210283c293cdd0b22c695900b6`.
The `yaw_000` GLB SHA-256 is
`1944143ebcb4e0138421b0c597fc1c01b0f6f5781baf5ea59eeed05406ad858b`.

For every v2 candidate, maximum absolute axial angles are: torso lateral
`1.376973 deg`, shoulder roll `1.590623 deg`, hip roll `2.649759 deg`,
head-to-body lateral `1.068566 deg`, neck-to-head `2.394773 deg`, and head-bone
lateral `2.394763 deg`; all pass the fixed `2/3-degree` per-metric envelope.
Only `yaw_000` also passes the four-limb sagittal gate; `+/-90` and `180`
remain sideways limb diagnostics. The 20 H.264 files are `640x360`, 30 fps,
33 frames, and all 49 bundle files are read-only. Codex inspected four-cycle
frame Front/Side/Feet/Skeleton strips and found the body upright without
regressing forward gait or arm swing. This is
`agent_qa_passed_pending_user_acceptance`, not user approval.

Serve v2 at `http://127.0.0.1:8101/`; it adds a visible six-row axial gate
below the limb metrics. Keep v1 at port 8100 only for before/after diagnostic
comparison. The v2 selection directory is
`retarget_motion_basis_and_axial_selection_v2` and must remain absent until the
reviewer clicks. Do not use a v1 selection for the next formal retarget, and do
not call this v2 bundle the formal third retarget: grounding, deformation,
loop, Idle, and formal GLB publication still follow after the v2 parameter is
recorded.

The user approved v2 `yaw_000` on 2026-07-12. The immutable selection is
`retarget_motion_basis_and_axial_selection_v2/retarget_motion_basis_correction_v1.json`,
SHA-256
`0b0a39ef36b37a86c767c9c4a75365d8d15c5360713e54d05011e28641462d88`;
it binds manifest
`d04d2dceef4e3797d2973308ad7e0b22e158f84ed8150535a7f16bd32a11880f`,
candidate `yaw_000`, FRONT `-Y`, UP `+Z`, and the identity limb basis. This is
the user's approval of the retarget parameters and v2 canary appearance. It
authorizes the next formal male Walk/Idle run but does not itself claim that
grounding, deformation, loop, Idle, export, or dataset registration passed.

Do not describe the full Route-2 instance generator as unattended-batch ready
yet. The retarget core is now semantic-map-driven code and contains no
male-specific `bone_N` names, but it has only passed the male Walking canary.
The pinned female Pixal PBR GLB exists at
`tmp/i23d_human_bakeoff_v1/pixal3d/rocketbox_female_adult_01/canary_1024_seed42.glb`
(36,817,496 bytes), while
`tmp/pixal_tokenrig_route2_v1/rocketbox_female_adult_01` and its qualified
candidate do not yet exist. Likewise, `human_attribute_instances_v1/jobs_v2.json`
defines seven exact cases (tall man, short woman, glasses, hat, shirt color,
trousers, shoes) and their reviewed masks, but none has completed the ordered
FLUX.2 -> 2D gate -> Pixal3D -> TokenRig/static gate -> Walk/Idle -> media gate
chain. Batch execution may become fail-closed and code-only after female plus
representative head-accessory, body-proportion, garment, and shoe canaries pass;
until then, automatic rejection is allowed but automatic formal registration
is not.

On 2026-07-12 the user narrowed the production Route-2 instance space to a
controlled template sampler. Treat `human_attribute_instances_v1/jobs_v2.json`
as immutable historical canary evidence; do not use its free per-instance
FLUX/Pixal regeneration policy as the production batch policy. The production
v1 sampler must use a small allowlist of one-time-qualified geometry templates
plus deterministic semantic-mask/material parameters:

- male and female identities, faces, body builds, base shoe shapes, and base
  hairstyles stay fixed within each approved template family;
- upper-garment geometry is only `short_sleeve` or `long_sleeve`;
- lower-garment geometry is only `trousers` or `shorts`; skirts and dresses are
  excluded from v1;
- the approved shoe geometry is fixed and only its registered color may vary;
- one approved hat type and one approved eyeglass-frame type may each be
  toggled on/off and recolored; both remain rigidly Head-bound;
- hair style is fixed and only hair color may vary; a hat must use a qualified
  hat-compatible hair state;
- stature is a bounded `short` / `standard` / `tall` parameter or an explicitly
  qualified template, never arbitrary body-width/build generation;
- free-form garment/shoe/accessory styles, facial hair, jewelry, backpacks,
  high heels, platform boots, and unregistered geometry are rejected;
- skin color is not a default v1 random attribute unless the user later enables
  a separately reviewed palette.

FLUX.2 -> Pixal3D remains the one-time route for creating a new geometry
template and its prompt-edit evidence. Once a template is qualified, ordinary
instances must not rerun image-to-3D merely to change color. Sample only enum
values and exact sRGB/material records from the registered palette, preserve
non-target masks, record the complete attribute vector and template revisions,
and fail closed on an unregistered value or incompatible combination. Ordinary
qualified combinations may proceed without per-instance human intervention;
they still must pass automatic hash, material/non-target, attachment, Walk/Idle,
GLB, contact, and sampled-media QA. A new geometry category needs one new
canary, not repeated manual correction of every color instance.

On 2026-07-12 the user explicitly accepted mild visible foot sliding for the
first end-to-end feasibility pass and prioritized grounded feet over an
over-strict clearance-only slide score. The prior formal failure reported
`0.471209 m` left and `0.567924 m` right "stance slide", but exact per-frame
readback proved that the `<= 0.030 m` clearance label merged the real stance
with low swing frames and counted forward leg travel as planted-foot slide.
Walking slide distance/speed therefore remains recorded advisory evidence;
it is not a publication blocker. Standing Idle remains strictly planted.
Per-frame penetration below fixed Z=0, excessive hover, inverted feet,
deformation, FRONT -Y/travel disagreement, loop failure, PBR loss, and GLB
roundtrip failure remain hard gates. This policy change passed 158 focused
runner tests (one unrelated stale report-location test deselected) before the
formal male rerun was started. Do not weaken the 0.010 m penetration cap or
use this exception to hide a visibly airborne or structurally broken foot.

The user also authorized parallel agents for the Route-2 feasibility pass and
removed the requirement for per-instance visual approval. Male formal
Walk/Idle, female TokenRig preflight, the controlled v3 instance contract, and
the existing-template/input audit may run concurrently in disjoint outputs.
Automatic agent QA may advance a candidate; new geometry still requires one
template-level qualification, while deterministic color-only instances do not.

The controlled instance-space implementation is now
`external/SPEAR/tools/route2_controlled_instance_contract_v3.py`. Its eight
dimensions contain exactly 288,000 compatible male/female combinations:
top style/color, bottom style/color, fixed-shoe color, hat state/color with a
hat-compatible hair state, glasses state/color, fixed-hair color, and bounded
height class. A full ordinal -> attributes -> ordinal traversal covers all
288,000 entries and all options. This enumerates allowed requests only; its
registry slot names do not claim that missing geometry is qualified. The
read-only template audit at
`external/SPEAR/tmp/route2_controlled_template_audit_v1/audit_manifest.json`
(SHA-256
`5f10599035e7e156a47ebdf2303f6fcb3b74df04f9a7594e2fd4bfeedf5d778e`)
records ten research-candidate slots, twenty-six missing slots, and zero
formal Route-2 templates at audit time. In particular, long sleeves, shorts,
cap, glasses, hat-compatible hair, short/tall templates, and the 3D semantic
UV-mask material transformer still require real canaries.

The formal male rerun after the advisory Walking-slide policy passed Walking
GLB export/readback but correctly stopped at an actual lower-limb weight defect,
not ordinary cloth motion. Read-only diagnostic
`external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/formal_edge_stretch_diagnostic_v1/edge_stretch_diagnostic.json`
has SHA-256
`711ecd9ddd53bf20e15c187d8cecd23c6dfc98a4dbc4c37b3394e7281b516ee2`.
At frame 9, adjacent left-shoe vertices `585447/585579` move from a
`0.001171777 m` rest edge to `0.039956694 m` because the second vertex still
contains `0.089129604` right-toe weight. Edges at least 5 mm long reach only
`1.090596x`, so do not relax the global tearing gate to hide this isolated
cross-foot contamination. The root cause is the old sanitizer's single
`25%` full-body-width distal cutoff: it sees hands but can skip legs close to
the body midline. A generic no-overwrite lower-limb sanitation v2 must repair
both sexes before another formal retarget.

Female Route-2 preflight now supersedes the earlier “TokenRig absent” note.
Direct `--use_transfer` produced a PBR-preserving 1-mesh/1-skin/52-joint GLB
but static QA rejected 44,111 opposite-limb vertices, maximum weight
`0.0024102661`. The fitted `--use_skeleton` alternative also preserved the
PBR container but failed at UV seam duplicate vertex 1892. Both are
`research_candidate` failures with immutable evidence; neither is authorized
for animation. Apply the same sanitation v2 to the fitted candidate first,
then rerun the complete static audit before Walk/Idle.

On 2026-07-12 the user explicitly changed Route-2 execution priority after
reviewing the amount of time spent on non-visible numeric failures.  Maintain
two independent result tracks.  A `research_candidate_fastlane` track must
first deliver paired Walking and Standing Idle GLBs, import both animations
into isolated UE content, and render watchable UE media.  Mild visible foot
sliding, duplicate serialized face keys, a tiny topology-area tolerance
overrun, and non-visible short-edge stretch remain recorded advisories in this
track; they must not trigger repeated repair loops when the rendered person is
visually usable.  The fast-lane hard failures are a visibly airborne foot,
more than 1 cm floor penetration, a folded/inverted foot or limb, direction
disagreement, an obvious body/garment tear, unstable head accessories, lost
PBR, or failed GLB/UE import and playback.  Keep the existing strict evidence
and repairs as the separate formal-registration track, and never relabel a
fast-lane result as `formal_dataset_asset` merely because it rendered.  The
immediate delivery order is the already approved male `yaw_000` Walking plus
Standing Idle pair in UE, before further color/template optimization.

The approved male fast-lane pair is now combined without changing either
animation into
`external/SPEAR/tmp/route2_tokenrig_ue_fastlane_v1/rocketbox_male_adult_01/walk_idle.glb`
(SHA-256
`463d923d38fec655bf758462807349f860efcf9ccfb262f8d86fcb0a5e45a8f1`).
It contains one PBR mesh/skin, 52 joints, and exactly `Walking` plus
`Standing_Idle`.  UE 5.5 Interchange rejected its required
`EXT_texture_webp` extension before creating any object.  This is an import
container limitation, not a rig or animation failure.  The canonical source
remains untouched.  The lossless UE-only container conversion at
`external/SPEAR/tools/transcode_glb_webp_to_png.py` rewrites only the two
embedded image payloads and texture references to core PNG; it verifies that
the mesh, skin, node, accessor, and animation JSON graphs are identical.  The
result is
`external/SPEAR/tmp/route2_tokenrig_ue_fastlane_v1/rocketbox_male_adult_01/walk_idle_ue_png.glb`
(SHA-256
`a27862ccc73aabcfca1018bc68f727ba5956d2b8b434f450b0ee95ef967d5a6c`).

`external/SPEAR/tools/import_route2_tokenrig_fastlane_editor.py` imported that
PNG container into the isolated UE tag
`route2_tokenrig_male_fastlane_v1`.  It created one skeletal mesh, one
skeleton, a populated PBR material with two textures, both animation
sequences, and
`/Game/MyAssets/Audioset/Blueprints/gate_route2_tokenrig_male_fastlane_v1/BP_gate_route2_tokenrig_male_fastlane_v1`.
A fresh second UE commandlet reloaded and validated all assets with 52 bones,
zero warnings/errors, and marked `reload_verification.status=passed` in
`external/SPEAR/tmp/route2_tokenrig_ue_fastlane_v1/rocketbox_male_adult_01/ue_import_manifest.json`.
This remains `research_candidate_fastlane` and explicitly has
`formal_registration_authorized=false`.  Cook and paired Apartment rendering
are the next operations; do not send this already imported pair back through
another strict geometry-repair loop before producing those media.

The UE package and paired Apartment smoke media are complete.  UAT
`BuildCookRun` finished successfully with the isolated mesh and Blueprint
directories, and direct `UnrealPak -List` readback found the Blueprint, PBR
material/textures, skeletal mesh/skeleton, and both animation sequences in
`Standalone-Development/Linux/SpearSim/Content/Paks/SpearSim-Linux.pak`.
The watchable outputs are:

- Walking:
  `external/SPEAR/tmp/route2_tokenrig_ue_fastlane_v1/rocketbox_male_adult_01/ue_smoke/walking/videos/apartment_v1_view0.mp4`
  (SHA-256
  `0ee66c5c999a0e2102da4d05dccaa6e0d22c9172d18881c2e726ea897e8890f5`);
- Standing Idle:
  `external/SPEAR/tmp/route2_tokenrig_ue_fastlane_v1/rocketbox_male_adult_01/ue_smoke/idle/videos/apartment_v1_view0.mp4`
  (SHA-256
  `029a276898ab78d59bf00359bd45b97bdbb0925d74c9070c6ec98ea981be4b72`);
- simultaneous paired review:
  `external/SPEAR/tmp/route2_tokenrig_ue_fastlane_v1/rocketbox_male_adult_01/ue_smoke/walk_idle_pair_review.mp4`
  (SHA-256
  `fb7b4f30f04e40992cba317f5bd256f3109b199125b9be67d3796f585fbf5078`).

Each source clip is H.264, 640x480, 15 fps, exactly 45 frames/3 seconds;
the paired review is 1280x480 with the two clips synchronized.  Sampled
frames 0/15/30/44 were visually inspected.  Walking travels left-to-right
with matching body direction, preserved PBR, grounded feet, and no obvious
inversion or tear.  Standing Idle remains stationary with grounded feet and
preserved PBR; its slight neutral forward lean is accepted for the fast lane.
The post-render Python shared-memory `BufferError` is a teardown-only warning
after all frames, metadata, and MP4s were written, not a media failure.

The paired UE smoke metadata reports `automatic_checks.overall=passed`.
Walking has `0.575048 cm` maximum floor penetration; Standing Idle has
`0.983049 cm`, still inside the fast-lane 1 cm cap, with zero actor root
roll/pitch.  The imported Pixal actor at `actor_scale=1.0` measures only about
`93.2 cm` high in UE, so this smoke proves playback/grounding rather than final
adult-scale calibration.  Do not silently change only the actor scale: scaling
to roughly 2x also magnifies the Walking foot clearance/penetration.  Calibrate
adult height and per-motion grounding together in a new output before the
Apartment dataset examples; keep this grounded scale-1 smoke immutable as the
fast-lane feasibility evidence.

The generic lower-limb sanitation-v2 branch also published a newer independent
fast-lane pair under
`external/SPEAR/tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01/fitted_skeleton_v1/sanitized_weights_v2_preflight/retarget_fastlane_v1`.
Its hard checks pass: Walking has zero penetration, 3.19 mm pre-contact
clearance, 1.586 mm maximum stance hover, full support coverage, no both-feet
airborne frame, non-inverted feet, direction dot 1.0/body dot 0.989, and valid
PBR/loop/GLB readback; Idle has effectively zero penetration, 0.1 mm hover,
bilateral contact, non-inverted feet, and valid PBR/loop/GLB readback.  Its
short-edge ratios remain advisory only in `research_candidate_fastlane`; the
separate strict failure remains preserved and formal registration is false.

On 2026-07-12 the stable native Rocketbox Route-1 batch became the active
large-scale UE baseline.  This does not erase or relabel any approved Route-2
result; it provides the reliable path requested after the Pixal/TokenRig media
proved too variable.  The approved neutral-walk baseline remains byte-for-byte
frozen at
`/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1`.
`baseline_manifest.json` still has SHA-256
`b6e468e5f0c79d7ecec168e3c2460a7997a8d2916393da9add1ef2b6952fb922`.
Never import into, recolor, or regenerate files inside that directory.

The complete Microsoft-Rocketbox checkout is fixed at commit
`0943055db6ec570bcef9f2c8b41c9e5467c808f9`, with 3,203 required blob files,
115 human avatars, and the repository MIT LICENSE present.  The corrupt legacy
zip remains failure evidence only and is not a pipeline dependency.  The
inventory SSOT is
`external/SPEAR/tmp/rocketbox_route1_inventory_v1/inventory.json` (SHA-256
`56153273ddc13c856cd21bf85c6cb63cc3940121d25a4486004f8c02be9037e8`).
It records exactly 74 male and 41 female avatars: 40 Adults, 4 Children, and
71 Professions.  Train/validation/test splitting must continue to use
`base_avatar_id`, so recolors of one identity cannot cross splits.

Apartment scale is authored scale, not a normalization target.  All runtime
actors use scale `1.0`; authored heights are preserved within max(3 cm, 2%).
The imported UE bounds span `142.93384325504303` to
`188.375248670578` cm and the maximum authored/imported height delta is
`1.67553` cm.  The Apartment ceiling contract is 280 cm with at least 25 cm
headroom; the smallest observed headroom across all 115 is
`91.624751329422` cm.  Adult voice emitters use 0.90 times authored height and
children use 0.88.  Do not scale children to adult height, and do not silently
rescale short/tall characters to one canonical body.

All 115 native PBR runtimes are complete under
`external/SPEAR/tmp/rocketbox_batch_native_runtime_v1`, and all 115
UE-normalized in-place runtimes are under
`external/SPEAR/tmp/rocketbox_batch_native_runtime_ue_v1`.  Each contains
exactly `Walking` and `Standing_Idle`, retains its original material/texture
graph and geometry, uses FRONT `-Y`, and delegates horizontal travel to the UE
actor trajectory while preserving vertical gait motion.  The four children
retain the authored `Bip02` skeleton family; Adults/Professions use `Bip01`.
Five source files contained unused loose unweighted vertices; only those
non-surface vertices were removed in staging and the exception is manifest
recorded.  `Medical_Male_03` lacks one optional stethoscope specular input;
the missing optional connection is recorded rather than inventing a texture.

The UE import root is
`external/SPEAR/tmp/rocketbox_batch_native_ue_import_v1`.  It contains 115
Blueprint directories and 115 skeletal-mesh directories, all with 80 bones,
actor scale `1.0`, both required animations, original materials/textures,
ground checks, and authored-height checks.  UE 5.5 can stall when many skinned
meshes are reloaded in one editor process, so
`external/SPEAR/tools/verify_rocketbox_batch_ue.py` verifies each avatar in a
fresh read-only UE process with its own log and timeout.  The final status
`batch_process_verify_status.json` has SHA-256
`905b1def940f5fb5e748b0885588b9f0bb57e84e3424d65a3eb10177d77a08dd`
and reports 115 passed, zero failed.  Use `-RenderOffscreen`, not `-nullrhi`,
for skinned UE checks.

A full Linux BuildCookRun completed successfully after those imports.  The
packaged PAK is 3,909,591,573 bytes with SHA-256
`58ba2b6d5917ce1ac2d3120a5462449e7571d3c0ce4de4186ddf70395b66c2f3`.
`external/SPEAR/tools/verify_rocketbox_batch_pak.py` compared the PAK listing
against every UE import manifest and found all 2,015 required Rocketbox
entries: 115 each of Blueprint, skeletal mesh, skeleton, PhysicsAsset,
Walking, and Standing Idle, plus every registered material and texture.  The
audit is
`external/SPEAR/tmp/rocketbox_batch_native_ue_import_v1/cook_v1/pak_audit.json`
(SHA-256
`60de75cc5517ae53e7506a93120d0dfae02f3dbd409ebb949256dc15ca760546`).
The prior PAK hash was
`c8af267d83776eee1e04146a178b0ec08e7c1d08b9894f7ca170af9519581617`;
retain that value as pre-batch provenance, but use the new PAK for reviews.

The 115-avatar runtime gate in
`external/SPEAR/tools/spike_rlr/human_apartment_gate.py` is now inventory and
hash driven rather than a two-tag allowlist.  A full pass accepted all 115 and
exports category, demographic, gender, authored/UE height, ceiling headroom,
voice height, exact animations, Blueprint, and source/import hashes.  Bip02 is
also supported by the runtime body-basis/direction query.  The paired spec
builder is `external/SPEAR/tools/build_rocketbox_batch_apartment_specs.py`,
and the resumable evidence runner is
`external/SPEAR/tools/run_rocketbox_batch_apartment_reviews.py`.  Each final
clip must contain the UE primary view, synchronized top-down trajectory,
annotated side-by-side review, runtime gate, per-frame bounds/floor metadata,
and a research-candidate registry entry.

The first new batch media passed for native `Female_Adult_01` Walking and
Standing Idle and for `Female_Child_01` Walking.  Review them at:

- `external/SPEAR/tmp/rocketbox_batch_apartment_review_v1/clips/rocketbox_adults_female_adult_01_original_ue_v1/walking/videos/side_by_side_review_annotated.mp4`;
- `external/SPEAR/tmp/rocketbox_batch_apartment_review_v1/clips/rocketbox_adults_female_adult_01_original_ue_v1/idle/videos/side_by_side_review_annotated.mp4`;
- `external/SPEAR/tmp/rocketbox_batch_apartment_review_v1/clips/rocketbox_children_female_child_01_original_ue_v1/walking/videos/side_by_side_review_annotated.mp4`.

The adult Walking clip passed 75/75 visibility, exact floor contact, and body
direction; the child clip retained its approximately 143 cm room scale,
queried `Bip02-Pelvis`, and passed with `2.983683 deg` root-trajectory error
and `1.828415 deg` semantic body-forward error.  A diagnostic attempt to run
GPU 0 and GPU 1 windowed instances concurrently left GPU 1 at Vulkan/X11
swapchain initialization before any child actor spawned.  That attempt is an
infrastructure failure, not an asset rejection; its failed command-log entry
is intentionally retained.  The same child job then passed on the stable GPU
0 path.  Do not enable windowed multi-GPU review workers until an offscreen
canary passes; resumable single-worker execution is the current stable mode.

On 2026-07-13 the offscreen multi-GPU canary superseded the preceding
windowed-only restriction.  Packaged UE on graphics adapters 1--3 stalls at
Vulkan/X11 swapchain initialization when launched windowed, but all three
secondary adapters reached their independent RPC ports and completed the
Female_Child, Male_Child, and Medical_Female Walking reviews with
`-RenderOffscreen`.  The six current representative clips are under
`external/SPEAR/tmp/rocketbox_camera_pass_table_loop_apartment_review_v2`.
Every path starts camera-right/rear, passes camera-left/front, and makes one
counter-clockwise loop around the round table.  All six have 270 frames over
18 seconds, gender-matched LibriTTS speech beginning at 5.0 seconds, native
binaural audio, synchronized top-down review, and complete research-candidate
registry evidence.  Use offscreen mode by default for any multi-GPU packaged
UE batch; startup staggering alone does not fix secondary windowed adapters.

The same trajectory now has a Pixal3D animal canary at
`external/SPEAR/tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1`.
`dog_pug_pixal_canary_v2_100k` passed 270-frame UE playback, four independent
body-forward/actor-root direction windows, and per-frame dynamic floor snap.
Its observed height is approximately 84.4--95.0 cm and its floor penetration
is numerical zero.  Its 18-second dog-bark source was energy-segmented into
six source calls and scheduled as seven separated events with at least 0.85
seconds of silence; never replace this with seamless waveform tiling.  The
watchable primary review is
`clips/dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking/videos/side_by_side_review_annotated.mp4`.
This remains `research_candidate`, with formal promotion false.

Runtime direction QA now supports both Rocketbox humanoid semantic bones and
Quaternius numeric quadruped bones.  Facing uses a same-frame longitudinal
quadruped basis (`Bone` rear, `Bone_002` front, paired rear feet for up), while
world travel uses the captured UE actor-root transform.  Do not use a short
window of animated pelvis/root-bone displacement as the trajectory authority:
quadruped gait sway produced a false 36.2-degree error even though the actor
path was correct.  Pixal raw animal outputs face head `-X`; the already tested
X mirror during Quaternius binding converts the animated asset to `+X`
forward.  Preserve both raw and bound orientation records.

The safe mesh audit at `docs/asset_mesh_efficiency_audit.md` must remain the
face-count authority.  It reads GLB accessors/OBJ lines without constructing
mesh adjacency.  Never call `trimesh.split` or another full adjacency routine
on million-face inputs: the discarded diagnostic reached roughly 241.7 GiB
RSS, the kernel OOM-killed it on 2026-07-13, and sshd temporarily dropped new
connections through `MaxStartups`.  The network interface and sshd service did
not fail.  Rocketbox's 115 runtimes are only 4,344--15,517 triangles and need
no decimation.  Pixal dog close LOD remains 100k double-sided; its 72-frame UE
capture was 15.74 seconds versus 16.00 seconds for the visibly broken 40k
single-sided candidate, so single-actor UE throughput does not justify the
quality loss.

All remaining legacy animal references are being regenerated, without
overwriting the Hunyuan artifacts, under
`external/SPEAR/tmp/pixal_animal_backend_substitution_v1/generated_batch_v1`.
The canonical scalable runner is
`external/SPEAR/tools/run_pixal_animal_persistent_batch.py`: each GPU loads the
pinned Pixal model once and executes several inputs.  The cold-per-asset runner
is retained as timing evidence only because it repeatedly spends minutes in
CPU/storage model loading while GPU utilization is near zero.  Model revision
is `0b31f9160aa400719af409098bff7936a932f726`; DINO revision is
`3c276edd87d6f6e569ff0c4400e086807d0f3881`; parameters remain 1024,
manual FOV 0.2, low-VRAM.  The first four new cats preserved PBR.  British
Shorthair, Siamese, and Tabby may advance to LOD/binding; Persian has obvious
spiky/planar fur geometry and is rejected at the static visual gate.  No new
animal becomes formal until license/source, static, binding, animation,
Apartment media, and species-audio gates all pass.

The controlled instance-attribute SSOT is now
`docs/controlled_source_asset_attribute_workflow.md`.  Every instance is an
independent absolute profile; identity manifests must never encode `from`,
`to`, `one_step_lighter`, or another relative-edit history.  Each sampled
categorical appearance attribute has one to three allowed values.  This limit
does not apply to fixed metadata, numeric measurements, hashes, licenses, or QA
states.  Animal domains are species/breed specific, are balanced-quota sampled,
and compile once into one complete FLUX.2 image-edit prompt before Pixal3D.

The first controlled Rocketbox expansion is a fixed-template material route,
not a wardrobe generator.  `base_avatar_id` owns identity, authored height,
body, garment geometry/length/style, eyes, headwear, eyewear, and accessories.
Eyes and accessories are not sampled.  Only semantic regions actually audited
for a base may expose up to three top, bottom, footwear, or hair colors.  Solid
colors use deterministic semantic-mask material transforms; FLUX.2 is optional
only for approved mask-constrained texture detail and must be baked back to the
same Rocketbox UV with unchanged geometry/rig/outside-mask data.  The earlier
`route2_controlled_human_instance_space_v3` remains research evidence for a
wider unqualified space and is not the current bulk sampler.

Keep generation intent separate from observations.  `size=small|medium|large`
and a versioned target-physical profile belong in the immutable instance
request.  Actual centimeters and `actor_scale` belong only in post-generation
`physical_measurements`, after canonical orientation, grounding, rig-landmark
measurement, and UE centimeter readback.  Rocketbox keeps authored scale 1.0.
The asset acoustic profile declares allowed licensed sound classes; the scene
manifest pins the actual waveform, license/hash, timing, gain, spatialization,
and short-call repetition schedule.  The legacy `source_assets_v1` registry is
preserved without rewrite; new controlled assets migrate under new IDs and the
planned `source_asset_v2` contract.

As of 2026-07-13, that controlled contract is implemented rather than merely
planned.  The strict standard-library implementation is
`external/SPEAR/tools/controlled_source_asset_schema.py`; it validates
`avengine_attribute_profile_v1`, rejects relative instance attributes, enforces
one-to-three values per sampled domain, performs exact balanced-quota sampling,
compiles one complete animal FLUX.2/Pixal3D plan or a fixed-geometry Rocketbox
MaterialEditPlan, validates `source_asset_v2`, builds absolute-attribute QA
pairs, and makes lineage-grouped dataset manifests.  Its regression suite is
`external/SPEAR/tests/tools/test_controlled_source_asset_schema.py`.

Use `external/SPEAR/tools/build_controlled_source_asset_inputs.py` to
authenticate profile artifacts and publish immutable `profile_snapshot.json`,
`instance_requests.json`, `execution_jobs.json`, `qa_pair_plan.json`, and
`generation_plan.json`.  The current no-overwrite canary is
`external/SPEAR/tmp/controlled_source_asset_input_v1/all_profiles_20260713_v3`:
6 profiles, 54 requests (45 animal and 9 Rocketbox material jobs), and 57
planned QA pairs.  Its profile snapshot SHA-256 is
`4297e52e5a6e5ee399602afbda67b5abf3d347c25f24392a8ef5758e2d95f66b`.
These are authenticated plans, not generated assets; planned QA answers remain
pending until matching realized `source_asset_v2` records pass visual QA.

The profile catalog is
`external/SPEAR/data/controlled_source_attributes_v1/profiles`.  It currently
contains five breed-specific animal candidates and one audited
`Male_Adult_01` shirt-color profile.  The animal reference provenance remains
`legacy_unknown` and physical targets remain `provisional`, so none can be
formal yet.  The Rocketbox profile only opens `top_color`; never infer masks
for another avatar.  The 9-request compiler canary repeats the three unique
Rocketbox colors to test quota accounting.  A production executor must dedupe
by base avatar, absolute sampled material attributes, and material-plan
revision; repeated ordinal/seed values are not new visual assets.

Use `external/SPEAR/tools/build_controlled_source_dataset.py` only after real
`source_asset_v2` manifests exist.  It authenticates every asset and license,
defaults to `formal_dataset_asset`, and emits `dataset_manifest.json`,
`qa_dataset.json`, `scene_source_pool.json`, and `artifact_audit.json` without
replacement.  The normalized `execution_jobs.json` adapters for invoking the
existing FLUX.2/Pixal persistent worker and Rocketbox material/runtime paths
are still pending; do not report the current 45 animal jobs or 9 human jobs as
generated instances.  The executable workflow and exact boundary are the SSOT
in `docs/controlled_source_asset_attribute_workflow.md`.

2026-07-13 controlled-attribute execution update (supersedes only the final
"adapters are pending" status sentence above; preserve the earlier block as
historical provenance): the normalized adapters and immutable attempt ledgers
are now implemented and exercised.  The Rocketbox material route produced
three unique `Male_Adult_01` top-color candidates (blue/green/burgundy), native
Walking/Standing Idle runtimes, UE-normalized runtimes, three validated
`source_asset_v2` records, and three realized color QA pairs.  The candidate
dataset is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/rocketbox_candidate_dataset_v3_20260713`.
These records remain `research_candidate`; UE Editor/Apartment/audio gates are
not silently promoted.

For animals, 10 of the 45 authenticated requests were selected as a five-pair
static canary.  FLUX.2, 2D review, ISNet preparation, Pixal3D PBR GLB export,
GLB readback, multiview review, and static decisions all passed 10/10.  The
Pixal batch manifest is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_qa_canary_v1_20260713/pixal_batch_manifest.json`
(batch SHA-256
`f2889b3acd95ee06925fed733b0bbca7ba0fd925b41b75f5a9b43a7fd64d9ab0`).
The static overview is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_static_reviews_v3_20260713_overview/all_static_contact_sheets.png`.
The 10 static-qualified, still-unbound assets are registered at
`external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_source_assets_v2_20260713_v1/registry_manifest.json`
(registry SHA-256
`20e72c8b0c4dcd0ba39d4f05aa553abce093d47c4e60f7a961d62df4eb9b1b07`).

Do not treat semantic `size` as measured evidence.  All 10 animal size values
remain deferred until LOD, species rig, metric mesh/bone measurement, and UE
centimeter readback.  Consequently the static animal dataset emits only two
realized pairs (Tabby body build and Pug coat color), while three planned size
pairs remain blocked.  The combined human/animal candidate dataset is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/combined_controlled_candidate_dataset_v1_20260713`:
13 assets, five realized pairs, nine questions, zero Apartment-ready sources,
manifest SHA-256
`a433ad26d5c63d5cdb211610998ed4033c53e44ded07d699e249c49aaa4d6e86`.
Continue with animal LOD/binding/Walking/Idle/metric/UE/audio gates and the
remaining 35 planned requests; do not rerun or overwrite the completed static
canary.  The approved Rocketbox baseline remains immutable.

2026-07-13 full controlled-animal static update (supersedes the remaining-35
sentence immediately above): all 45 authenticated animal requests have now
been executed without replacing the first 10 canaries.  The remaining FLUX.2
batch generated 35 candidates; 2 Tabby images were rejected before 3D for a
forked tail and two complete tails.  Pixal3D then produced and GLB-read back
33/33 candidates on four persistent workers (batch SHA-256
`e99ea28258a384644c48702ac5ddf7ed851f718a28ac26aa7a958808ac646009`).
Static multiview review approved 32 and rejected
`dog_beagle_fcea77a333db` because a large white tail section floats beside a
separate attached tail.  The compact decision input is committed at
`external/SPEAR/data/controlled_source_attributes_v1/reviews/animal_pixal_static_remaining33_20260713_v1.json`;
the 32 approved source assets are registered under
`external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_source_assets_remaining33_v2_20260713`
(registry SHA-256
`46617330f3846a938e7ee6adc9f9833467b01efdc338a33a31c41d0e85e378bc`).
Together with the first 10 animals and 3 Rocketbox variants, the authenticated
candidate dataset at
`external/SPEAR/tmp/controlled_source_asset_execution_v1/combined_controlled_candidate_dataset_full_v1_20260713`
contains 45 assets, 42 realized pairs, 86 questions, and 0 scene-ready sources;
its internal manifest SHA-256 is
`241e331f5607d23b97c02e3800fb0d60f6734212cec78b1b5fa738356061d54f`.
No `size` question is realized until metric rig/UE measurement exists.

The 33-item Pixal batch took 1818.58 seconds wall time.  Individual inference
plus GLB export took 95.60--362.62 seconds (mean 183.82), while each model load
took 126.26--131.87 seconds.  Low instantaneous GPU use during Pixal is often
expected in CPU-heavy parameterization/UV/finalize phases.  Static round-robin
partitioning also left early-finishing GPUs idle at the tail; before the next
large Pixal batch, use a tested shared claim queue so a free persistent worker
can claim the next unstarted job.  Do not interrupt or repartition an active
atomic staging batch merely to improve an instantaneous utilization reading.

2026-07-13 controlled Apartment completion update (supersedes the historical
"42 animals are still static-only" status above, without deleting that
provenance): the stable Rocketbox route now has complete Apartment evidence for
all 115 authored avatars. Walking and Standing Idle are complete for 230/230
clips and 115/115 action pairs. The clickable SSOT is
`docs/rocketbox_batch_apartment_video_index.md`. Every clip has an annotated
review, UE main view, and synchronized top-down review; the approved frozen
neutral-walk baseline remains unchanged.

The controlled animal route advanced 31 qualified Pixal assets through LOD,
species-family binding, Walking/Idle, UE import/readback, Apartment rendering,
species-matched repeated-call audio, and observed physical measurement: 8 cats
and 23 dogs, or 62/62 clips and 31/31 action pairs. The final clickable SSOT is
`docs/controlled_animal_video_catalog.md`. Six initially oversized Pugs
measured 55--81 cm at the shoulder. They were not silently accepted: the new
`recalibrate_controlled_animal_apartment_specs.py` derives a per-instance scale
from UE-observed shoulder height, publishes a new non-overwriting batch, and
`resolve_controlled_animal_apartment_manifests.py` explicitly supersedes only
those six records. Closed-loop remeasurement is 27.07/30.00/33.00 cm for the
small/medium/large Pug medians, with all six inside tolerance.

The final post-Apartment registry is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_source_assets_apartment_31_final_v1_20260713/registry_manifest.json`.
It contains 31 validated `source_asset_v2` records; registry SHA-256 is
`caebb9301843f4dfec2c8f7d18bb278a3d97974ad11c98f1d1c94bf7ef1b85d9`.
The realized candidate dataset is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_animal_apartment_candidate_dataset_31_final_v1_20260713`:
31 assets, 92 pairs, and 226 questions. All 68 size questions agree with the
observed shoulder-height ordering (minimum answer margin 0.544 cm). Scene QA
has no remaining technical blockers, but all 31 records intentionally remain
`research_candidate`: `rights_ready=false` preserves the Pixal/reference and
training-provenance review requirements, so formal scene eligibility is 0/31.

GPU utilization for Apartment review is bursty by design. One packaged UE
worker normally uses about 5.5 GB per assigned GPU, but fixed-step simulation,
RPC/PNG readback, RLR, Matplotlib top-down generation, and libx264 leave gaps
between GPU bursts. A 30-second four-worker sample averaged about
15%/28%/30%/26%, with later bursts above 60% and up to about 91%. Do not equate
that with an idle queue: inspect worker processes and render/review counts.
During the final Pug run, another server user independently occupied GPU 3;
the one RPC-start failure was resumed on GPU 0 without touching that process.
At the end of this update GPU 0--2 are idle because rendering is complete, not
because pending GPU work was left unscheduled.

2026-07-13 normalized final-dataset input update: new dataset builds must freeze
profile snapshots, deterministic `instance_requests.json` batches, and realized
`source_asset_v2` files with
`external/SPEAR/tools/build_controlled_source_dataset_input_manifest.py`, then
invoke `build_controlled_source_dataset.py --input-manifest ...`. The compiler
rebuilds each request batch from its exact profile revision and requires every
realized asset to match exactly one request across profile/hash, absolute
attributes, target physical profile, rig/acoustic contract, and model revisions.
Raw `--profile/--asset` mode is legacy-only and reports
`request_lineage=legacy_unverified`.

The final frozen input is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_human_animal_dataset_input_34_final_v1_20260713/dataset_input_manifest.json`.
Its byte-identical, Git-auditable control-plane copy is
`external/SPEAR/data/controlled_source_attributes_v1/dataset_inputs/controlled_human_animal_34_final_v1_20260713.json`;
large realized artifacts remain outside Git in the immutable evidence tree.
It authenticates 8 profiles, 2 request batches, 72 requests, and 34 realized
assets. All 34 assets have unique, passed request bindings; 38 unrealized
requests remain explicit and are not registered. Input manifest SHA-256 is
`c1b315c590030f4f952f4b876ab976bae665621f41c46b947efe27f9eb5a1c8c`.
The manifest-only rebuild is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_human_animal_normalized_candidate_dataset_34_final_v1_20260713`:
34 assets, 95 realized pairs, 232 questions. Its dataset/QA/source-pool/artifact
hashes are respectively
`a7b09ddec1d2d6e72690d686deaa42f9f62ea04db52a26bc31e5838c8a27e3fa`,
`52a15ee29d285aa9ef19c3ca43edc59284e9e3d33437fae7f40f0ac5d8401e34`,
`66feb737c35b23285a9d3d4e20c622da72645003ea593267c83781aa8fa12738`,
and `2c28d454a06b98da509ace7bd5ab211852f60111144eb342a5ac99ae4800df45`.
The build receipt SHA-256 is
`bd8e68e529ebd342561386d8e9c492f85f4beae1f774a7d6cd527703a9c989fa`.
All six hashes were independently recomputed, and the four core dataset files
are byte-identical to the earlier approved 34-asset candidate build. Regression
coverage passed 30/30. Scene eligibility intentionally remains 0/34: animal
rights blockers remain, while the three controlled Rocketbox material revisions
still need their own UE/Apartment/audio evidence.

The GPU status check made during this CPU/hash phase showed GPU 0--2 at 0% and
GPU 3 at about 30% with 8.9 GiB used by user `ryl`'s independent audio training.
No AVEngine Unreal, FLUX.2, Pixal3D, SkinTokens, or TokenRig process was running.
Do not manufacture GPU load for JSON/hash compilation, and do not touch GPU 3;
schedule the next independent GPU-heavy generation batches across free GPUs
0--2.

2026-07-13 controlled-animal visible-direction override (supersedes only the
direction/animation approval implied by the earlier 62/62 file-completion
status): the user visually rejected all current controlled-animal Walking
outputs after observing cats running diagonally and dogs running backward and
diagonally.  A bone-vector or trajectory-vector check is no longer sufficient
evidence.  The old GLBs, registries, decisions, and videos remain immutable,
but their Walking direction status is now
`rejected_by_user_visual_review`; Idle media is diagnostic-only until the bind
orientation is revalidated.  Do not promote any of these 31 animals from the
old media.

The non-overwriting revalidation manifest is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_animal_direction_revalidation_v1_20260713/review_manifest.json`
(31 assets: 8 cats and 23 dogs; internal `manifest_sha256`
`60a98750bda0ed4d2badd9799bfb5b8f7e47873a38faae64f525e778e82cfb62`;
file SHA-256
`b53a0d76023e16ec5087ee970c7b4a071611d6d6222ad3f5a021b5c6fced4c0d`).
It reauthenticates the source PBR/static/LOD/bound/animation/Apartment evidence
and records the user override without rewriting any historical result.
`external/SPEAR/tools/spike_rlr/controlled_animal_direction_review_server.py`
serves the Hunyuan-style correction gate at `http://127.0.0.1:8102/`.  It first
replays the existing mirror-X transform, then lets the reviewer adjust only a
per-asset yaw while comparing side/top silhouettes, the original PBR contact
sheet, and old dynamic videos.  Decisions are immutable transform overlays
under
`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_animal_direction_review_state_v1_20260713`;
source GLBs are never edited.  A static geometry approval still does not
approve Walking: apply the recorded
`target_rotate_z_deg_after_flip_x`, regenerate the binding, and require a new
straight-line plus curved UE dynamic canary before updating animation status.

The human browser review entry is `docs/rocketbox_human_video_review.html`.
It contains the latest six camera-pass/table-loop representatives followed by
all 115 Rocketbox avatars with complete Walking and Standing Idle pairs: 236
review items and 708 existing MP4 view links, each with an absolute server path.
`docs/controlled_animal_video_review.html` now labels the existing animal
videos as rejected diagnostic evidence and links back to the direction gate.

2026-07-13 corrected animal pose/direction canary update (supersedes the
mirror-X/fine-yaw operating description immediately above; it does not alter
or restore the rejected historical 31): visible head yaw was not a valid proxy
for the torso/rest-pose direction.  The replacement contract starts from the
raw 100k mesh with an identity pretransform, disables automatic orientation,
forbids hidden reflection/mirror and fine-yaw compensation, and permits only
manual 0/±90/180-degree yaw.  A turned head, twisted/diagonal torso,
inconsistent fore/hind leg planes, or paws on different ground planes rejects
the source pose and returns it to the strict-side 2D/Pixal stage.

The exact-side cat v5 canary
`cat_tabby_four_limb_rest_side_3a1ecde08179` passed isolated Walking/Idle,
UE import/readback, and two Apartment clips.  In the uniform-clay dog v6
batch, `dog_beagle_four_limb_rest_side_clay_1550ff78df40` remains rejected
because Walking showed foot stretching/fragments, while
`dog_beagle_four_limb_rest_side_clay_1b1e63af05c3` passed isolated
Walking/Idle, UE import/readback, and two Apartment clips.  Do not generate
Apartment media for the rejected dog and do not reinterpret its static pass as
an animation pass.

The cat UE import result is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/four_limb_rest_side_ue_import_v5_20260713_r1/ue_import_result.json`
(file SHA-256
`1016c12509020590fe2808b16b8d97486f58d0c6d6e1fe19a693abbeb094bd2c`).
The dog result is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/dog_beagle_four_limb_rest_side_clay_ue_import_v6_20260713_r1/ue_import_result.json`
(file SHA-256
`7c29cc535ff8ffc7b675b4fa6c357d76b2ff8a21cccf9296064d24a165f10786`).
UE 5.5 Interchange can emit a handled NodeUid ensure during editor shutdown
after the import script has already written and authenticated a passed result;
classify the run from the import result/readback and success marker, not the
process exit code alone, while retaining the ensure log as engineering debt.

Both new assets were packaged in one shared cook rather than one cook per
asset.  Evidence is
`external/SPEAR/tmp/controlled_source_asset_execution_v1/four_limb_rest_side_shared_ue_cook_v6_20260713_r1/ue_cook_timing.txt`:
160.75 seconds wall, 7,904,028 KB maximum RSS, exit 0.  The successful UE
render times are 70.0879/75.6404 seconds for cat Walk/Idle and
68.6774/73.6466 seconds for dog Walk/Idle.  All four clips are 18 seconds with
270 frames, grounded dynamic bounds, zero root roll/pitch, main view,
synchronized top-down, annotated side-by-side review, stereo 16 kHz audio,
and an authenticated source-event schedule.  The cat schedule has nine short
call events and the dog schedule seven, both repeated with silence gaps.

The live manual gate remains `http://127.0.0.1:8102/` but now consumes
`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_animal_pose_direction_new_canary_review_v4_20260713/review_manifest.json`.
Its file SHA-256 is
`977b74fb08cad3ce2a547b0eb87eafba54ac5bc25e61c03dcabf1fb6109b47bd`;
its internal manifest SHA-256 is
`0e3efd51ac9ca7bc73f5158c2b037e837764d55b909a6cc8c8be51ebece56ef9`.
The page authenticates six Apartment Walk/Idle media views for the passed cat
and dog, exposes none for the rejected dog, and still rejects ±5-degree API
rotations.  Browser decisions are immutable overlays under
`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_animal_direction_new_canary_review_state_v4_20260713`;
they never modify source GLBs.  Regression evidence is 32/32 builder/schema
tests plus 6/6 server tests.  These two successful canaries remain
`research_candidate` pending the human cardinal decision and rights/provenance
clearance; `formal_dataset_asset` remains false, and the old 31 assets remain
rejected diagnostic evidence.
