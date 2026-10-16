# Hunyuan3D To Rocketbox Part-Aware Bind Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one approved male and female Hunyuan3D mesh from the current FLUX.2 references, perform exactly one part-aware transfer onto the approved Rocketbox skeleton, bind only `walk_neutral` and `idle_neutral_01`, and expose hash-locked Front/Side/Feet videos for human review.

**Architecture:** A pure Python provenance contract gates Hunyuan generation on the two current reference approvals. A separate pure NumPy module owns human-region classification, compatible-region nearest-surface transfer, artifact masks, and finger-to-palm collapse. Blender opens the immutable Rocketbox walk baseline, uses its rest mesh and 80-bone armature as the transfer source, imports the cleaned Hunyuan OBJ as the only rendered body, adds one retargeted idle action, and exports separate walk/idle GLBs. A dedicated review contract and Flask UI bind decisions to both GLBs, the bind manifest, and six videos.

**Tech Stack:** Python 3.10/3.11, pytest, NumPy, Pillow, trimesh, Hunyuan3D-2.1, Blender 4.2.1, FFmpeg, Flask, Playwright.

## Global Constraints

- This remains a technical spike only; do not add either human to `external/SPEAR/data/source_assets_v1`.
- Require current approvals for exactly `rocketbox_male_adult_01` and `rocketbox_female_adult_01` before Hunyuan runs.
- Use Hunyuan3D-2.1 locally with 50 shape steps and deterministic seeds: male `4101`, female `7301`.
- Canonical Hunyuan weight storage is `/data/models/hunyuan3d-2.1`; the checkout may contain only a compatibility symlink after byte verification.
- Use `hy3d_textured.obj` plus `hy3d_diffuse.jpg`, `hy3d_metallic.jpg`, and `hy3d_roughness.jpg`; do not trust `hy3d_output_mesh.glb` as the binding input.
- The immutable Rocketbox walk source is `/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1`.
- Active motions are only `walk_neutral` and `idle_neutral_01`; do not process the other 66 locomotion files.
- Do not use Mixamo actions, Mixamo T-pose meshes, automatic Blender envelopes, hand proxy, forearm proxy, full-arm proxy, or body-only crop.
- Human transfer regions are torso/head, left/right upper arm, forearm, palm, thigh, calf, and foot. Finger weights collapse into the corresponding palm; hand geometry must still be connected and visible.
- Run one documented direct generated-topology attempt. If internal dynamic pixel QA shows tearing, empty pieces, detached hands, sleeve overlays, or cross-body contamination, write a rejection record and stop. The next plan must use stable Rocketbox-template fitting.
- Every JSON decision and manifest is atomic and hash-locked. Regeneration invalidates prior readiness.
- Final human review shows Male/Female navigation, Walk/Idle tabs, and Front/Side/Feet videos. Hunyuan work is not approved by a JSON-only or FBX-only review.

---

### Task 1: Approved Candidate Contract, Canonical Weights, And Hunyuan Runner

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/hy3d_human_candidate.py`
- Create: `external/SPEAR/tools/hy3d_generate_human_candidates.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_hy3d_human_candidate.py`
- Create: `external/SPEAR/tests/tools/test_hy3d_generate_human_candidates_static.py`
- Create: `external/SPEAR/tmp/hy3d_rocketbox_spike_v1/jobs_v1.json`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: `human_reference_review.assert_pair_approved(review_root: Path)` and the current `candidate.png`, `candidate_manifest.json`, and `reference_review.json` files.
- Produces: `assert_generation_ready(...)`, `write_hy3d_manifest(...)`, and per-asset `reference.png`, `reference_rembg.png`, `shape.glb`, paint outputs, and `hy3d_manifest.json`.

- [ ] **Step 1: Write failing contract and runner tests**

Cover exact pair approval, approved candidate/review hash matching, asset allowlist, direct-file and symlink confinement, fixed seed/steps, canonical model root, local paths, output allowlist, atomic manifest publication, and stale regeneration.

```python
def test_generation_rejects_a_stale_reference_review(review_root, tmp_path):
    approve_pair(review_root)
    (review_root / "rocketbox_male_adult_01" / "candidate.png").write_bytes(b"changed")
    with pytest.raises(Hy3DHumanNotReady):
        build_generation_job(review_root, "rocketbox_male_adult_01", tmp_path)
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest \
  tests/tools/spike_rlr/test_hy3d_human_candidate.py \
  tests/tools/test_hy3d_generate_human_candidates_static.py -q
```

Expected: import failures for the two new modules.

- [ ] **Step 3: Implement the pure provenance contract and static runner shell**

The candidate manifest schema is `hy3d_human_candidate_v1` and records asset ID, source/reference approval hashes, source/cutout/shape/OBJ/PBR hashes and sizes, Hunyuan code revision, weight-root hash manifest, seed, 50 steps, guidance scale, and `usage_scope: technical_spike_only`.

- [ ] **Step 4: Migrate the complete weight directory without changing bytes**

Hash the current large checkpoints, rename `external/Hunyuan3D-2.1/pretrained_models/hunyuan3d-2.1` on the same filesystem to `/data/models/hunyuan3d-2.1/hunyuan3d-2.1`, create a compatibility symlink at the old location, then re-hash and run a `local` shape-pipeline load. Do not touch the incomplete `/data/models/hub/models--tencent--Hunyuan3D-2` cache.

- [ ] **Step 5: Implement deterministic generation and paint**

Use `BackgroundRemover`, `torch.Generator(device="cuda").manual_seed(seed)`, `Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("hunyuan3d-2.1")`, and `num_inference_steps=50`. Run `tools/hy3d_bake_diffuse.py` with absolute paths and `HY3DGEN_MODELS=/data/models/hunyuan3d-2.1`.

- [ ] **Step 6: Run focused tests**

Expected: all Task 1 tests pass before a GPU command starts.

---

### Task 2: Human Region Transfer And Artifact Math

**Files:**
- Create: `external/SPEAR/tools/human_part_transfer.py`
- Create: `external/SPEAR/tests/test_human_part_transfer.py`

**Interfaces:**
- Produces: `HumanRegion`, `source_vertex_regions_from_weights`, `source_face_regions`, `target_regions_from_capsules`, `transfer_human_weights`, `collapse_finger_weights_to_palms`, `human_ground_artifact_mask`, and `cross_limb_bridge_face_mask`.
- Reuses: barycentric closest-surface, normalization, top-k, and graph inpainting ideas from `tools/robust_skin_transfer.py`, but no dog region constants or four-legged coordinate assumptions.

- [ ] **Step 1: Write failing pure NumPy tests**

Use synthetic T-pose vertices and weights to prove left/right separation, upper-arm/forearm/palm separation, thigh/calf/foot separation, compatible joint blending, top-four normalization, zero unmatched vertices after graph fill, complete finger-mass collapse into `Bip01 L Hand` or `Bip01 R Hand`, low flat component detection, and rejection of faces bridging left/right limbs below the pelvis.

```python
def test_palm_vertices_never_receive_opposite_or_torso_weights():
    weights, stats = transfer_human_weights(source_fixture(), target_palm_fixture())
    assert stats["unmatched"] == 0
    assert weights[:, LEFT_HAND].min() > 0.95
    assert weights[:, RIGHT_HAND].max() == 0.0
    assert weights[:, SPINE].max() == 0.0
```

- [ ] **Step 2: Verify RED**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest tests/test_human_part_transfer.py -q
```

- [ ] **Step 3: Implement minimal region logic**

Classify Rocketbox source vertices from actual vertex-group mass. Build target capsules from Rocketbox rest-bone head/tail points, with side-aware left/right eligibility. Search only compatible source faces, barycentrically interpolate their weights, fill topology gaps, keep four influences, normalize, then collapse all `Bip01 * Finger*` mass into the matching Hand group.

- [ ] **Step 4: Run tests and existing generic transfer regressions**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest \
  tests/test_human_part_transfer.py tests/test_robust_skin_transfer.py -q
```

---

### Task 3: Blender Cleanup, Alignment, Binding, Walk, And Idle

**Files:**
- Create: `external/SPEAR/tools/blender_bind_hy3d_to_rocketbox.py`
- Create: `external/SPEAR/tests/tools/test_blender_bind_hy3d_to_rocketbox_static.py`

**Interfaces:**
- CLI: `--asset-id --baseline-dir --hy3d-dir --idle-motion-fbx --output-dir`.
- Produces: `cleaned.obj`, `bound.blend`, `bound_walk.glb`, `bound_idle.glb`, `bind_metrics.json`, and `bind_manifest.json`.

- [ ] **Step 1: Write failing static contract tests**

Require exact source files, `human_part_transfer`, no import of proxy/crop/Mixamo modules, rest-pose source capture, floor-aligned uniform bbox transform, preserved UV/material data, region stats, palm collapse, 80-bone Rocketbox armature, approved walk action reuse, idle source-absolute bake, target-only export, separate action GLBs, GLB roundtrip, and atomic readiness invalidation.

- [ ] **Step 2: Verify RED**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest \
  tests/tools/test_blender_bind_hy3d_to_rocketbox_static.py -q
```

- [ ] **Step 3: Implement cleanup and alignment**

Open the sealed `retarget.blend`, set the armature to REST, capture the original Rocketbox mesh, import `hy3d_textured.obj`, remove only geometric masks proven by Task 2, and write before/after component metrics. Align with one uniform scale, XY center alignment, and shared floor Z. Default target Z rotation is exactly `0`; do not reuse the historical `-90` correction.

- [ ] **Step 4: Implement part-aware binding**

Transfer weights by the Task 2 region contract, create only Rocketbox-named vertex groups, attach one Armature modifier, remove the original Rocketbox body from final export selection, assign the Hunyuan PBR maps, and assert every target vertex has normalized nonzero weights with at most four influences.

- [ ] **Step 5: Add only the two approved motion roles**

Keep the sealed walk action already present in the baseline blend. Import the gender-matched `*_idle_neutral_01.max.fbx` and call the proven source-absolute cache/bake helpers from `blender_retarget_rocketbox_walk.py`; name the action `<asset_id>_idle_neutral_01_retarget`. Export one GLB with only walk active and one with only idle active.

- [ ] **Step 6: Run static tests and two Blender smoke fixtures**

The smoke must validate import, 80 bones, two actions in `bound.blend`, one action per GLB, UV/PBR presence, skin/joint roundtrip, and source/current manifest hashes.

---

### Task 4: Direct-Attempt Dynamic Gate

**Files:**
- Create: `external/SPEAR/tools/blender_render_hy3d_rocketbox_review.py`
- Create: `external/SPEAR/tests/tools/test_blender_render_hy3d_rocketbox_review_static.py`

**Interfaces:**
- CLI: `--asset-id --bind-dir`.
- Produces six 1280x720 videos: `walk_front.mp4`, `walk_side.mp4`, `walk_feet.mp4`, `idle_front.mp4`, `idle_side.mp4`, `idle_feet.mp4`; plus `bind_contact_sheet.png`, `review_manifest.json`, and either `direct_attempt_ready.json` or `direct_attempt_rejected.json`.

- [ ] **Step 1: Write failing renderer tests**

Require Walk/Idle action selection, root-follow Front/Side cameras, pelvis-follow Feet camera, fixed floor, loop frame counts, nonblank output, no missing hand/foot bounds, and media hashes.

- [ ] **Step 2: Implement and verify the renderer**

Reuse lighting, framing, FFmpeg validation, and contact-sheet helpers from `blender_render_rocketbox_motion_review.py`; do not duplicate source-stick or path-arrow logic that is irrelevant to this target-only review.

- [ ] **Step 3: Inspect actual pixels before opening a user gate**

Extract start/mid/end frames from all six videos. Reject the direct attempt if either asset shows detached or duplicated hands, blank pieces, arm/torso or thigh cross-contamination, sleeve overlay seams, inverted feet, floor cards, leg-gap fans, or mesh explosions. A rejection ends this implementation plan and triggers a new stable-template fitting plan.

---

### Task 5: Hash-Locked Walk/Idle Review Web App

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/hy3d_rocketbox_review.py`
- Create: `external/SPEAR/tools/spike_rlr/hy3d_rocketbox_review_server.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_hy3d_rocketbox_review.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_hy3d_rocketbox_review_server.py`

**Interfaces:**
- Routes: `/`, `/asset/<asset_id>`, `/media/<asset_id>/<motion>/<view>`, `/decision/<asset_id>`, `/gate`.
- Decisions bind to `bind_manifest.json`, `review_manifest.json`, both GLBs, and all six current videos.

- [ ] **Step 1: Write failing contract/server tests**

Cover exact assets and motions, path/symlink confinement, stable multi-file snapshots, stale-page A-to-B races, CSRF, versioned media URLs, Range responses, GET read-only behavior, independent decisions, and the paired current-approval gate.

- [ ] **Step 2: Implement the compact UI**

Use Male/Female navigation, Walk/Idle tabs, and Front/Side/Feet video columns. Show the approved FLUX reference as a small identity cue, not raw JSON. Keep approve/reject controls visible and usable on desktop and mobile.

- [ ] **Step 3: Run focused tests**

Expected: all review contract and server tests pass.

---

### Task 6: Execute, Verify, And Stop At Human Review

- [ ] **Step 1: Run the complete non-GPU suite**

Run all new tests plus the existing FLUX reference, Rocketbox baseline/retarget, robust transfer, and motion-review suites.

- [ ] **Step 2: Generate male and female in parallel**

Use GPU 0 for male and GPU 1 for female. Each process loads its model locally, writes to `external/SPEAR/tmp/hy3d_rocketbox_spike_v1/<asset_id>`, and exits before binding begins.

- [ ] **Step 3: Inspect rembg, shape, and painted static meshes**

Render front/back/left/right/feet static views and record component, watertightness, UV, and bounds metrics. Do not bind a candidate with missing limbs or a major disconnected shoulder/wrist/hip/ankle component.

- [ ] **Step 4: Bind, render, and apply the direct-attempt gate**

Run Task 3 and Task 4 for both assets. If either direct attempt fails, preserve all evidence, write the rejection reason, update `AGENTS.md`, and stop before creating review approvals.

- [ ] **Step 5: Start and browser-test the review server only when both are ready**

Prefer `http://127.0.0.1:8093/`. Playwright must verify all six full-resolution videos load, motion tabs select the correct media, controls fit 1440x1000 and 390x844, prompts/labels do not overflow, and no browser errors occur.

- [ ] **Step 6: Stop at the human gate**

Report the URL and ask the user to review Male/Female across Walk/Idle and Front/Side/Feet. Do not promote assets or start dataset generation.

## Stable-Template Branch

If Task 4 rejects direct generated topology, do not patch it further. Write a new focused plan that keeps the approved Rocketbox mesh topology, skeleton, and weights, fits its surface toward the Hunyuan silhouette, and projects Hunyuan appearance onto the stable UV/material representation. That branch must not reuse proxy meshes.
