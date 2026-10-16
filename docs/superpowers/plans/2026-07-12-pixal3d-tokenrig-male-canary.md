# Pixal3D → TokenRig Male Canary Continuation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue from the sealed Rocketbox and existing Pixal3D artifacts, produce one texture-preserving male TokenRig skeleton/skin, retarget only Walking and Standing Idle, publish a browser-based visual acceptance surface, and continue to the rest of route 2 after agent visual QA.

**Architecture:** Keep every approved or failed artifact immutable and write this route to a new `external/SPEAR/tmp/pixal_tokenrig_route2_v1/` tree. A pure-Python contract authenticates all inputs, model revisions, commands, and outputs. SkinTokens performs the learned rig prediction against the original PBR Pixal GLB; Blender performs canonical-axis normalization, static inspection, semantic mapping, animation retargeting, GLB roundtrip, and review rendering. A hash-locked Flask page is the only user approval surface.

**Tech Stack:** Python 3.11, Torch 2.7.1+cu126, SkinTokens/TokenRig commit `273b691d35989d71cd17ff2895fdc735097b92d1`, SkinTokens weight revision `79736cad0fd84de384d5eede659b4ebd24effe33`, the pinned SkinTokens server runtime `bpy==5.0.1`, Blender 4.2.1 LTS for downstream static/animation audit and rendering, trimesh, open3d, FFmpeg, Flask, pytest, and Playwright. The two Blender runtimes are recorded separately and neither may be silently substituted for the other.

## Global Constraints

- Continue from the current dirty worktrees. Never reset, checkout, clean, roll back, or overwrite user work.
- Never modify `/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1`; its current manifest SHA-256 is `b6e468e5f0c79d7ecec168e3c2460a7997a8d2916393da9add1ef2b6952fb922` and its 24 managed artifacts total `75,434,237` bytes.
- The only route-2 male input mesh is `external/SPEAR/tmp/i23d_human_bakeoff_v1/pixal3d/rocketbox_male_adult_01/canary_1024_seed42.glb`, SHA-256 `1df2490d6b83e52fa3b7c4e9d6b69207fa59cad0deae80e3dc3f894dfc443c42`.
- Preserve the Pixal mesh, UVs, original packed PBR textures, solid green short-sleeve shirt, gray long trousers, and gray shoes. Do not use `cleaned.obj`, a Rocketbox runtime mesh, plaid, shorts, or black lower legs.
- Pixal3D is the selected default image-to-3D backend. Do not run TRELLIS or download another image model.
- Do not retry Pixal FOV `0.2`, `0.35`, or `0.5` as a rigging fix.
- The previous direct Rocketbox-weight transfer is rejected: fixed floor `-0.004898416 m`, right-foot minimum `0.040251061 m`, approximately `4.515 cm` unsupported clearance.
- Predict a mesh-matched skeleton and skin first with TokenRig `--use_transfer`. Only if static QA fails may the fallback use a fitted skeleton with `--use_skeleton --use_transfer`.
- The skeleton must cover pelvis, spine, head, both arms, hands, legs, feet, and toes. Finger influences may collapse into the corresponding hand. Hat/glasses, when later present, must bind to Head.
- Canonical AVEngine orientation is `FRONT -Y`, `UP +Z`; Pixal's source `FRONT +Y` is yaw-normalized exactly once after TokenRig fitting.
- Active motions are exactly Rocketbox `walk_neutral` and `idle_neutral_01`; no other locomotion, Mixamo, running, sitting, or gesture action is allowed.
- Static failure blocks animation. A passed pixel/video QA snapshot is required before female and batch route-2 work, but the user authorized agent visual QA to satisfy that execution gate without waiting for a reply.
- Penetration must be at most `0.010 m`; no foot inversion, visible hovering, shoulder/hip collapse, trouser tearing, or accessory drift is allowed.
- Every model, code, input, output, GLB roundtrip, license, command, parameter, hash, failure, and QA result is recorded. SkinTokens remains `research_candidate` with a training-provenance risk note for ArticulationXL, VRoid Hub, and ModelsResource.
- Hunyuan3D 2.0/2.1 and their derivatives remain `technical_spike_only` or `rejected` and are never inputs to this route.

---

### Task 1: Freeze And Continuation Ledger

**Files:**
- Modify: `AGENTS.md`
- Create: `.superpowers/sdd/pixal-tokenrig-route2-progress.md`

**Produces:** A durable record of approved baseline verification, dirty-worktree state, selected Pixal input, model revisions, prior failure, and the next executable task.

- [ ] Record the exact root, SPEAR, and SkinTokens branches/commits and dirty paths without modifying them.
- [ ] Record baseline verification as `24/24` SHA-256 and size matches, `75,434,237` managed bytes, `FRONT -Y`, official materials, male/female approval, and neutral-walk approval.
- [ ] Record FLUX.2 revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` (25 files, `23,740,007,447` bytes), Pixal3D revision `0b31f9160aa400719af409098bff7936a932f726` (19 files, `24,044,888,779` bytes), and SkinTokens revision/checkpoint hashes.
- [ ] Record the direct-bind and FOV experiments as preserved rejection evidence and state that the exact existing Pixal PBR GLB is the new canary.

### Task 2: Complete Rocketbox Partial Clone And SkinTokens Environment

**Files:**
- Modify in place: `/data/datasets/rocketbox/Microsoft-Rocketbox/.git/info/sparse-checkout`
- Modify in place: `external/SkinTokens/.venv/`
- Preserve: `/data/datasets/rocketbox/raw/Microsoft-Rocketbox-master.zip`

**Produces:** A complete commit-pinned Rocketbox partial clone and an import-clean SkinTokens venv.

- [ ] Add sparse directories one at a time at commit `0943055db6ec570bcef9f2c8b41c9e5467c808f9`: `Assets/Avatars/Children`, `Assets/Avatars/Professions`, `Assets/Animations`, `Assets/Animals`, `Source`, `Tools`, `Docs`, and the one-file `Assets/Editor` tree. Keep the already materialized Adults, LICENSE, and README.
- [ ] Verify `git rev-parse HEAD`, a clean Rocketbox checkout, all `3203` commit blobs materialized/readable with `git cat-file`, exactly `115` avatar FBX files (`40` adult, `4` child, `71` profession), and `LICENSE.md` containing the MIT grant.
- [ ] Preserve the 4,154,996,407-byte corrupt zip and existing logs; do not extract it or make it a dependency.
- [ ] Install `external/SkinTokens/requirements.txt` into the existing Python 3.11 venv, then install `flash-attn --no-build-isolation`; retain Torch `2.7.1+cu126`.
- [ ] Run real imports for `torch`, `transformers`, `diffusers`, `omegaconf`, `lightning`, `bpy`, `trimesh`, `open3d`, and `flash_attn`; print versions and require CUDA availability.
- [ ] Run an offline TokenRig model-load smoke against `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt` without mesh inference or network access.

### Task 3: Authenticated TokenRig Transfer Runner

**Files:**
- Create: `external/SPEAR/tools/tokenrig_human_canary.py`
- Create: `external/SPEAR/tests/tools/test_tokenrig_human_canary.py`
- Generate: `external/SPEAR/tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01/tokenrig_transfer.glb`
- Generate: `external/SPEAR/tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01/tokenrig_manifest.json`

**Interfaces:**
- CLI consumes `--input-glb`, `--input-manifest`, `--output-dir`, `--skintokens-root`, `--model-revision`, and optional `--use-skeleton-input`.
- Default inference command is SkinTokens `demo.py --use_transfer`; fallback adds `--use_skeleton`.

- [ ] Write tests that reject a different input hash, `cleaned.obj`, symlinks outside the approved roots, an unpinned SkinTokens checkout, missing/changed checkpoints, output aliasing the input, stale prior output, and any unrecorded inference parameter.
- [ ] Require atomic output publication and manifest schema `pixal_tokenrig_canary_v1` containing code/weight/license hashes, command, environment versions, GPU, random parameters, input/output hashes, `source_front=positive-y`, and `canonical_front=negative-y`.
- [ ] Run focused tests and require RED before implementing the runner, then GREEN before GPU inference.
- [ ] Run exactly one male `--use_transfer` inference using the original PBR GLB and publish only after GLB parse succeeds.

### Task 4: Static Bind, Skeleton, Weight, And Texture Gate

**Files:**
- Create: `external/SPEAR/tools/blender_audit_tokenrig_human.py`
- Create: `external/SPEAR/tests/tools/test_blender_audit_tokenrig_human_static.py`
- Generate under the male output directory: `bind_pose.glb`, `bind_front.png`, `bind_back.png`, `bind_side.png`, `bind_top.png`, `skeleton_overlay.png`, `weights_contact.png`, `texture_compare.png`, `joint_hierarchy.txt`, and `static_qa.json`.

**Interfaces:**
- Consumes the authenticated TokenRig GLB/manifest.
- Produces `static_qa.json` with `decision: passed|rejected`, semantic bone map, hierarchy, bind/rest matrices, per-region influence coverage, seam-duplicate consistency, PBR texture hashes, bounds, and failure reasons.

- [ ] Test that the auditor normalizes Pixal `+Y` to canonical `FRONT -Y` exactly once, keeps the textured Pixal mesh as the sole rendered body, and rejects missing UVs, changed PBR bytes, missing skin, zero-weight vertices, non-finite matrices, disconnected required chains, or absent bilateral feet/toes.
- [ ] Resolve required semantics from names plus hierarchy and joint position; ambiguous mappings are failures, not guessed animation mappings.
- [ ] Verify every vertex has normalized weights, at most four retained influences, position-identical UV seam vertices have identical weights, and visible garment regions do not receive opposite-limb contamination.
- [ ] Inspect all generated pixels. If passed, continue to Task 6. If rejected, write the reason and execute Task 5 without starting animation.

### Task 5: Fitted-Skeleton `--use_skeleton` Fallback

**Files:**
- Create: `external/SPEAR/tools/blender_fit_rocketbox_skeleton_to_pixal.py`
- Create: `external/SPEAR/tests/tools/test_blender_fit_rocketbox_skeleton_to_pixal_static.py`
- Generate: `fitted_skeleton_input.glb`, `fitted_skeleton_metrics.json`, `tokenrig_skeleton_transfer.glb`, and a second static QA snapshot.

**Interfaces:**
- Reuses Rocketbox hierarchy/names but repositions joints to the Pixal rest body; it never copies Rocketbox vertex weights or mesh geometry.

- [ ] Fit pelvis/spine/head, clavicle/arm/hand, hip/knee/ankle/foot/toe joint locations to the Pixal body while preserving parentage, bilateral symmetry, joint containment, and the Pixal forward-lean/bent-knee rest pose.
- [ ] Export the original Pixal PBR mesh plus fitted skeleton as the `--use_skeleton --use_transfer` input, with no Rocketbox body and no `cleaned.obj` runtime dependency.
- [ ] Run TokenRig fallback once, then repeat Task 4. If it still fails, preserve both attempts as `rejected`, update `AGENTS.md`, and stop before animation.

### Task 6: Rocketbox Walk/Idle To TokenRig Retarget

**Files:**
- Create: `external/SPEAR/tools/blender_retarget_rocketbox_to_tokenrig.py`
- Create: `external/SPEAR/tests/tools/test_blender_retarget_rocketbox_to_tokenrig_static.py`
- Generate: `animated.blend`, `walking.glb`, `standing_idle.glb`, `retarget_manifest.json`, and `retarget_metrics.json`.

**Interfaces:**
- Walk source: sealed male `retarget.blend` and its approved `walk_neutral` action.
- Idle source: `/data/datasets/rocketbox/Microsoft-Rocketbox/Assets/Animations/all_animations_max_motextr_static/m_idle_neutral_01.max.fbx` at the pinned commit.

- [ ] Test the exact semantic map, source/target rest matrices, parent-first evaluation, one-time axis conversion, target-proportioned translations, root motion scale, loop frame ranges, and target-only export.
- [ ] Transfer global rest-relative rotations through each source-to-target rest-frame alignment; never reuse the rejected direct vertex weights and never treat an animated Rocketbox frame as canonical bind rest.
- [ ] Name the only actions `Walking` and `Standing_Idle`, normalize each against one authenticated floor, and require bilateral support over the loop.
- [ ] Export separate one-action GLBs, re-import both, and verify the Pixal mesh/PBR hashes, skeleton, skin, actions, loop endpoints, `FRONT -Y`, finite matrices, and `<=0.010 m` penetration.

### Task 7: Dynamic Media QA And Browser Approval Gate

**Files:**
- Create: `external/SPEAR/tools/blender_render_tokenrig_human_review.py`
- Create: `external/SPEAR/tools/spike_rlr/tokenrig_human_review.py`
- Create: `external/SPEAR/tools/spike_rlr/tokenrig_human_review_server.py`
- Create matching focused tests under `external/SPEAR/tests/tools/` and `external/SPEAR/tests/tools/spike_rlr/`.
- Generate: Walk/Idle `front`, `side`, `top`, `feet`, and `skeleton` MP4s, contact sheets, `media_qa.json`, and `review.html` data.

**Interfaces:**
- Review page exposes human-readable hierarchy/static evidence and ten current MP4s; it never asks the user to inspect FBX or raw JSON.

- [ ] Render exact looping Front, Side, Top, Feet, and Skeleton videos for both actions with fixed-floor and semantic `FRONT -Y` labels.
- [ ] Compute and visually inspect foot clearance/contact, penetration, body forward, loop discontinuity, shoulder/hip span, garment deformation, PBR preservation, and mesh/skeleton alignment. Reject foot inversion/hovering, shoulder/hip collapse, trouser tearing, or attachment drift.
- [ ] Hash-lock decisions to the static manifest, retarget manifest, both GLBs, and every review medium; stale bytes invalidate approval.
- [ ] Run focused tests, GLB readback, FFprobe checks, desktop/mobile Playwright QA, and a final independent review.
- [ ] Start the page on the next free port, inspect it in desktop/mobile browsers, and record `agent_qa_passed_pending_user_acceptance` only when every static and animated check passes. Continue to the female canary without waiting for user input; never label the decision as user-approved.

## Post-Approval Continuation

After male agent QA passes, immediately execute the female canary and one-attribute-at-a-time FLUX.2 edits from the master plan. Complete all route-2 execution and its consolidated user acceptance page before route 1. Then continue through human Apartment, formal animal audit, animal FLUX.2 migration, mixed Apartment, registry, license, splits, and final acceptance without pausing for review replies.
