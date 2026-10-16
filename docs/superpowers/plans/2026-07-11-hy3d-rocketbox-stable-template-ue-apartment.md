# Hunyuan Appearance On Stable Rocketbox Template And UE Apartment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve prompt-controlled Hunyuan identity and appearance on a stable Rocketbox mesh, skeleton, and skin-weight contract, prove clean Walk/Idle motion in Blender, then import the same asset into SPEAR/Unreal and render it inside `apartment_0000`.

**Architecture:** The rejected generated mesh is retained only as a canonical-pose geometry and PBR appearance guide. A new Blender builder opens the sealed Rocketbox baseline, keeps its topology and weights byte-for-byte, applies a bounded XY-only surface fit, projects Hunyuan PBR maps onto Rocketbox material UVs, bakes gender-matched Idle beside the approved Walk, and normalizes each action to one fixed floor. The existing six-video review gate validates the stable result before a technical-spike-only Unreal importer combines both GLB actions and places the character in the existing apartment pipeline.

**Tech Stack:** Python 3.9/3.11, NumPy, Pillow, pytest, Blender 4.2.1 Python/BVHTree, FFmpeg, SPEAR, Unreal Engine 5.5, Playwright.

## Global Constraints

- Keep Hunyuan output and every derivative under `usage_scope: technical_spike_only`; do not add either human to `external/SPEAR/data/source_assets_v1`.
- Preserve the exact sealed Rocketbox 80-bone armature, mesh topology, vertex order, skin weights, material-slot count, and approved `walk_neutral` action.
- Hunyuan geometry may guide bounded deformation and texture projection only. It must never become the runtime mesh and no proxy hand/forearm/full-arm mesh may be added.
- Fit only X/Y surface shape; keep Rocketbox vertex Z, skeleton proportions, connectivity, and floor relationship unchanged.
- Clamp fitted XY displacement to `0.035 * Rocketbox REST height`, blend at strength `0.35`, and smooth displacement over existing mesh adjacency. Opacity-card vertices retain their original positions.
- Project diffuse, metallic, and roughness by compatible human region. Opposite limbs and torso/arm cross-projection are hard failures.
- Preserve the official Rocketbox opacity texture as alpha for the opacity material. Hunyuan data may change only its base color/PBR channels.
- Active motions remain exactly `walk_neutral` and `idle_neutral_01`.
- Blender pixel QA must pass before Unreal import. UE output remains in an isolated spike tag and isolated registry.
- UE target map is `/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000`.

---

### Task 1: Seal Direct-Topology Rejection Evidence

**Files:**
- Create: `external/SPEAR/tmp/hy3d_rocketbox_spike_v1/<asset_id>/direct_attempt_failure.json`
- Preserve: `external/SPEAR/tmp/hy3d_rocketbox_spike_v1/<asset_id>/direct_attempt_walk_front_frame1.png`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes the corrected-axis `bind_manifest.json`, `bind_metrics.json`, `bound.blend`, and diagnostic still.
- Produces schema `hy3d_rocketbox_direct_attempt_failure_v1` with exact hashes, automatic-gate error, and failed pixel checks.

- [ ] **Step 1: Write the failure records atomically**

Record male `walk gross penetration 0.027453 m`, `thigh_regions_clean=false`, `leg_gap_fans_absent=false`, and `pieces_nonblank=false`. Record female `idle Foot/Toe vertices never support`, `sleeves_seam_free=false`, `floor_cards_absent=false`, and `pieces_nonblank=false`.

- [ ] **Step 2: Verify evidence hashes and stale readiness removal**

Run:

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  external/SPEAR/tests/tools/test_blender_bind_hy3d_to_rocketbox_static.py \
  external/SPEAR/tests/tools/test_blender_render_hy3d_rocketbox_review_static.py
```

Expected: all tests pass and neither direct asset directory contains `direct_attempt_ready.json`.

---

### Task 2: Pure Stable-Template Fit And Projection Math

**Files:**
- Create: `external/SPEAR/tools/human_template_fit.py`
- Create: `external/SPEAR/tests/test_human_template_fit.py`

**Interfaces:**
- Produces `clamp_xy_displacements(displacements, max_distance)`, `smooth_xy_displacements(displacements, adjacency, fixed_mask, iterations, blend)`, `triangle_barycentric_3d(point, triangle)`, `sample_texture_bilinear(image, uv)`, `rasterize_uv_triangle(image, mask, target_uv, source_uv, source_image)`, and `dilate_unpainted(image, mask, iterations)`.
- Consumes region labels from `tools.human_part_transfer.HumanRegion`.

- [ ] **Step 1: Write failing pure tests**

Cover zero Z displacement, radial clamp, fixed opacity vertices, connected-boundary smoothing, barycentric sum/edge behavior, Blender UV vertical convention, bilinear RGB/PBR sampling, UV-island dilation, and rejection of a source triangle from an incompatible body region.

```python
def test_fit_never_changes_height_or_fixed_opacity_vertices():
    raw = np.array(((0.08, 0.00, 0.50), (0.01, 0.02, -0.25)))
    fitted = clamp_xy_displacements(raw, max_distance=0.04)
    smoothed = smooth_xy_displacements(
        fitted, ((1,), (0,)), np.array((False, True)), iterations=3, blend=0.5
    )
    assert np.allclose(smoothed[:, 2], 0.0)
    assert np.allclose(smoothed[1], 0.0)
```

- [ ] **Step 2: Run RED**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest tests/test_human_template_fit.py -q
```

Expected: import failure for `tools.human_template_fit`.

- [ ] **Step 3: Implement the minimal NumPy module and run GREEN**

Do not import Blender, SciPy, OpenCV, or Hunyuan runtime code in this module.

---

### Task 3: Blender Stable Rocketbox Template Builder

**Files:**
- Create: `external/SPEAR/tools/blender_fit_hy3d_to_rocketbox_template.py`
- Create: `external/SPEAR/tests/tools/test_blender_fit_hy3d_to_rocketbox_template_static.py`
- Reuse: `external/SPEAR/tools/blender_bind_hy3d_to_rocketbox.py`
- Reuse: `external/SPEAR/tools/human_part_transfer.py`

**Interfaces:**
- CLI: `--asset-id --baseline-dir --hy3d-dir --idle-motion-fbx --output-dir`.
- Produces `bound.blend`, `bound_walk.glb`, `bound_idle.glb`, `template_fit_metrics.json`, `bind_metrics.json`, `bind_manifest.json`, three material-scoped PBR texture sets, and `reference.png`.
- Uses schema `hy3d_rocketbox_bind_v1` plus `binding_mode: stable_rocketbox_template_fit_v1` and `usage_scope: technical_spike_only`.

- [ ] **Step 1: Write failing static contract tests**

Require the sealed baseline and approved Hunyuan manifests; exact corrected axis contract `source +Y -> canonical +Z`, `source +Z -> canonical -Y`; no generated-mesh Armature modifier; exact source vertex/face order and weight hashes before/after; BVH queries restricted by `HumanRegion`; bounded XY-only fit; original opacity alpha; packed PBR images; exactly two actions; fixed-floor action offsets; separate GLBs; GLB roundtrip; atomic manifest publication.

- [ ] **Step 2: Run RED**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest \
  tests/tools/test_blender_fit_hy3d_to_rocketbox_template_static.py -q
```

- [ ] **Step 3: Implement bounded surface fitting**

Open the sealed `retarget.blend`, capture immutable topology/weights, import and clean the Hunyuan guide with `forward_axis=NEGATIVE_Z, up_axis=Y`, classify both surfaces by the existing capsules, and build one `BVHTree` per region. For each non-opacity Rocketbox vertex, find the nearest compatible Hunyuan point, take only X/Y displacement, clamp to the configured maximum, blend, smooth, and leave Z unchanged. Re-hash topology and weights and hard-fail on any change.

- [ ] **Step 4: Implement material-scoped PBR projection**

Rasterize each Rocketbox material's existing UV loops. For each target surface sample, query only the matching Hunyuan-region BVH, compute source-triangle barycentric UV, and sample Hunyuan diffuse/metallic/roughness. Keep the original opacity map connected to Principled alpha and set `blend_method`/surface mode required by Blender 4.2. Pack every image and save external PNG copies for Unreal.

- [ ] **Step 5: Normalize each action to the authenticated floor**

Reuse the approved walk and bake the gender-matched idle. Evaluate Foot/Toe-weighted vertices over every frame, add a constant armature-object Z curve per action so the minimum foot sample equals `floor_z_m`, then re-evaluate all mesh vertices. Hard-fail if penetration exceeds `0.010 m`, either foot side has no support within `0.015 m`, or the offset magnitude exceeds `0.050 m`.

- [ ] **Step 6: Export and run the complete Task 2/3 suite**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/test_human_template_fit.py \
  tests/tools/test_blender_fit_hy3d_to_rocketbox_template_static.py \
  tests/tools/test_blender_bind_hy3d_to_rocketbox_static.py \
  tests/tools/test_blender_render_hy3d_rocketbox_review_static.py
```

---

### Task 4: Blender Dynamic Pixel Gate

**Files:**
- Reuse: `external/SPEAR/tools/blender_render_hy3d_rocketbox_review.py`
- Reuse: `external/SPEAR/tools/spike_rlr/hy3d_rocketbox_review_server.py`
- Create output under: `external/SPEAR/tmp/hy3d_rocketbox_template_fit_v1/<asset_id>/`

- [ ] **Step 1: Build male canary, then female**

Run the template builder for male. Do not build female until male produces an upright, textured, connected frame with clean feet. Then repeat with female.

- [ ] **Step 2: Render the six canonical videos**

For each asset render `walk_{front,side,feet}.mp4`, `idle_{front,side,feet}.mp4`, and `bind_contact_sheet.png` with `--python-exit-code 1`.

- [ ] **Step 3: Extract and inspect start/mid/end frames**

All ten existing pixel checks must pass. Add stable-template checks: no original-Hunyuan cards, no opacity-card blackout, no material-slot seams, no texture projection across opposite limbs, and no action-specific floor pop.

- [ ] **Step 4: Record artifact-locked pixel QA**

Write `pixel_qa.json` and `direct_attempt_ready.json` only after the exact current artifact snapshot passes. Start the 8093 review UI only after both assets are ready.

---

### Task 5: Unreal Humanoid Import Contract

**Files:**
- Create: `external/SPEAR/tools/import_gate_humanoid_editor.py`
- Create: `external/SPEAR/tests/tools/test_import_gate_humanoid_editor_static.py`
- Reuse: `external/SPEAR/tools/blender_combine_glb_actions.py`
- Reference: `external/SPEAR/tools/import_gate_animal_editor.py`

**Interfaces:**
- Environment: `GATE_TAG`, `GATE_RIGGED_GLB`, `GATE_IMPORT_MANIFEST`.
- Produces one SkeletalMesh, one Skeleton, AnimSequences `Walking` and `Standing_Idle`, Blueprint `gate_hy3d_rocketbox_male_adult_01_spike` or `gate_hy3d_rocketbox_female_adult_01_spike`, external textures/material instances, and `ue_import_manifest.json`.

- [ ] **Step 1: Write failing importer tests**

Require exact GLB path confinement, one skeletal mesh, one skeleton, exactly 80 bones, both animation names, non-null PBR textures/materials, Blueprint creation, `technical_spike_only`, atomic manifest output, and cleanup on partial import failure.

- [ ] **Step 2: Combine actions and run the Unreal commandlet**

```bash
blender --background --python-exit-code 1 \
  --python tools/blender_combine_glb_actions.py -- \
  --base-glb tmp/hy3d_rocketbox_template_fit_v1/rocketbox_male_adult_01/bound_walk.glb \
  --append-glb tmp/hy3d_rocketbox_template_fit_v1/rocketbox_male_adult_01/bound_idle.glb \
  --output tmp/hy3d_rocketbox_template_fit_v1/rocketbox_male_adult_01/ue_runtime.glb \
  --base-action-name Walking --append-action-name Standing_Idle
```

Import male to `/Game/MyAssets/Audioset/{Meshes,Blueprints}/gate_hy3d_rocketbox_male_adult_01_spike/` and female to `/Game/MyAssets/Audioset/{Meshes,Blueprints}/gate_hy3d_rocketbox_female_adult_01_spike/`, save all packages, and verify them by reloading in a second commandlet process before cook.

- [ ] **Step 3: Cook/package once after male canary import**

Use `/data/UE_5.5` and the existing `tools/run_uat.py --skip-cook-default-maps -build -cook -stage -package -archive -pak` path. Preserve `ue_import.log` and `cook.log`.

---

### Task 6: SPEAR Apartment Runtime Smoke

**Files:**
- Modify: `external/SPEAR/tools/spike_rlr/run_render_pass_apartment.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_human_apartment_smoke.py`
- Create output registry: `external/SPEAR/tmp/hy3d_rocketbox_template_fit_v1/ue_apartment_smoke/registry/`

**Interfaces:**
- Uses map `/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000`.
- Uses isolated `legacy_tag` values `hy3d_rocketbox_male_adult_01_spike` and `hy3d_rocketbox_female_adult_01_spike`.
- Produces separate five-second `Standing_Idle` and `Walking` clips plus metadata and annotated review videos.

- [ ] **Step 1: Write the gate and scenario tests**

The apartment runner must accept a human spike only when its stable-template `direct_attempt_ready.json` is current. It must reject the old direct-generated-topology root, formal registry promotion, stale GLBs, missing UE import manifest, and `SPEAR_SKIP_REVIEW_GATE=1` as evidence.

- [ ] **Step 2: Render male idle and walk canaries**

Start from scale `1.0`, Z lift `0 cm`, and yaw `0`; calibrate from measured UE bounds rather than copying historical Mixamo `14 cm/90 degrees`. Save actor bounds, root transform, animation name, and floor contact metadata for every frame.

- [ ] **Step 3: Inspect UE pixels and runtime state**

Require visible non-pink PBR materials, upright body, correct `FRONT -Y` semantic travel after UE conversion, no skeletal explosion, no furniture clipping at spawn, no floor penetration over `1 cm`, and continuous looping for five seconds.

- [ ] **Step 4: Repeat for female and build final evidence**

Preserve `spec.json`, `flags.json`, `apartment_v1_metadata.json`, `videos/actor_visual_metadata.json`, `videos/apartment_v1_view0.mp4`, `videos/side_by_side_review_annotated.mp4`, start/mid/end contact sheets, and command logs.

---

### Task 7: Final Review And Documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-07-10-rocketbox-human-review-retarget-design.md`

- [ ] **Step 1: Run all focused Blender, review-server, importer, and apartment tests**
- [ ] **Step 2: Browser-test the Blender review UI at desktop and mobile sizes**
- [ ] **Step 3: Publish paths to Blender videos and UE apartment videos for user audit**
- [ ] **Step 4: Record that Rocketbox is MIT but every Hunyuan-derived appearance remains technical-spike-only and excluded from formal AVEngine training/testing/evaluation registry use**
