# AVEngine / SPEAR Human And Animal AV Dataset Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, batch-capable AVEngine/SPEAR pipeline that produces approved human, animal, and mixed audio-video datasets with reproducible prompts, assets, motions, scenes, speech, licenses, splits, manifests, and visual QA.

**Architecture:** The pipeline is gate-driven. Immutable source snapshots feed either route 2 (FLUX.2 → Pixal3D → TokenRig → Walk/Idle) or route 1 (stable Rocketbox mesh/rig plus deterministic material variants). Approved human and animal assets enter a fail-closed source registry, then scenario builders drive SPEAR-first Apartment rendering, spatial audio, synchronized top-down/trajectory evidence, and final dataset manifests. Candidate, technical-spike, rejected, and formal states are physically and logically separated; no stage infers human approval from a successful command.

**Tech Stack:** Python 3.9/3.11, pytest, NumPy, Pillow, trimesh, open3d, Blender/bpy 4.2, Torch 2.7.1, FLUX.2 Klein 4B, Pixal3D, SkinTokens/TokenRig, FFmpeg/FFprobe, Flask, Playwright, SPEAR, Unreal Engine 5.5.4, Habitat/RLR, LibriTTS/VCTK.

## Global Constraints

- Work from the current AVEngine, SPEAR, and SkinTokens worktrees. Never reset, checkout, clean, roll back, or overwrite existing user work or approved artifacts.
- Preserve `/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1` byte-for-byte. Formal human motions are only Walking and Standing Idle.
- Route 2 has highest priority. Pixal3D is the default image-to-3D backend; do not download or revive another image model unless the user changes scope.
- Route-2 runtime geometry must remain the original textured Pixal mesh. Never substitute `cleaned.obj`, Rocketbox clothing/body geometry, or Hunyuan output while claiming Pixal identity or clothing.
- The male Pixal input has a solid green short-sleeve shirt, gray long trousers, and gray shoes. Plaid, shorts, and black lower legs are Rocketbox template evidence only.
- Do not retry Rocketbox direct weights or Pixal FOV 0.2/0.35/0.5. The measured right-foot hover is approximately 4.515 cm because the source rest skeleton does not match the Pixal forward-leaning/bent-knee body.
- Route-2 male static QA must pass before animation; male agent pixel/video QA must pass before female inference; each attribute candidate must pass agent 2D QA before 3D/rig/animation; failed instances are rejected without geometry substitution. The user explicitly authorized these QA decisions to be nonblocking.
- Canonical human orientation is `FRONT -Y`, `UP +Z`. Penetration is at most 1 cm. Reject inverted/hovering feet, shoulder/hip collapse, garment tearing, attachment drift, reverse travel, excessive speed, or failed loops.
- Route 1 preserves each Rocketbox identity, body, rig, garment geometry, garment length, patterns, and style. Color edits are natural-language-to-parameters plus semantic masks plus deterministic material transforms. FLUX.2 may alter texture detail only inside the approved mask.
- Split by `base_avatar_id`, never by generated variant, so the same person cannot cross train/validation/test.
- SPEAR API is the default scene/runtime interface. Direct UE calls are limited to import, bake, editor asset construction, cook, and package operations.
- Hunyuan3D 2.0/2.1 and every output/derivative remain `technical_spike_only` or `rejected`; they are excluded from formal training/evaluation resolution even if legacy manifests call them approved.
- All weights live under `/data/models`; never create `/data/Models`.
- SkinTokens is MIT but remains a provenance-risk `research_candidate` until ArticulationXL, VRoid Hub, and ModelsResource training-source implications are resolved for formal registration.
- The project scope is noncommercial academic/CVPR research. Record `research_release_ok` and `permissive_commercial_ok` separately. Pixal's NVIDIA research dependencies are acceptable for this scope but require an open-release replacement assessment for export/baking.
- Every stage records licenses, code/model revisions, file hashes, commands, parameters, seeds, environment versions, GLB roundtrip, media QA, and the state classification `formal_dataset_asset`, `research_candidate`, `technical_spike_only`, or `rejected`.
- Existing formal directories are not overwritten. Audits and replacements use new versioned roots and atomic no-replace promotion.

---

## Phase A — Freeze, Dependencies, And Route-2 Male

### Task 1: Takeover, Immutable Baselines, And Complete Dependencies

**Files:**
- Reuse/modify: `AGENTS.md`, `.superpowers/sdd/pixal-tokenrig-route2-progress.md`
- External data: `/data/datasets/rocketbox/Microsoft-Rocketbox`
- Environment: `external/SkinTokens/.venv`

- [x] Preserve and verify the sealed Rocketbox baseline, current manifests, dirty-worktree state, model revisions, and failed direct-binding/FOV evidence.
- [x] Complete the commit-pinned Rocketbox partial clone directory by directory; require 115 avatars, 3,203 readable commit blobs, and the MIT license.
- [x] Install all SkinTokens requirements plus `flash-attn`, run imports and an offline checkpoint load, and save exact environment/version evidence.
- [x] Keep the corrupt 4,154,996,407-byte Rocketbox zip and logs as failure evidence; never extract or depend on it.

### Task 2: Route-2 Male Static, Animation, And Browser Gate

**Plan:** `docs/superpowers/plans/2026-07-12-pixal3d-tokenrig-male-canary.md`

**Output root:** `external/SPEAR/tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01`

- [x] Authenticate the exact Pixal PBR GLB and run TokenRig `--use_transfer` once.
- [x] Produce bind pose, Front/Back/Side/Top, skeleton overlay, hierarchy, weights, seam-weight, and texture-preservation evidence.
- [x] If static QA fails, preserve the failure and run the fitted Rocketbox-named skeleton through `--use_skeleton --use_transfer`; never start animation from a failed static result.
- [ ] Retarget only the approved neutral walk and exact official male idle; validate rest matrices, semantic map, FRONT -Y, speed, bilateral contact, loops, and <=1 cm penetration.
- [ ] Export/read back one-action Walk/Idle GLBs and render Front/Side/Top/Feet/Skeleton videos for both.
- [ ] Serve and browser-inspect the hash-locked page, record `agent_qa_passed_pending_user_acceptance` when reasonable, and immediately continue male-dependent execution. FBX and raw JSON are never the review surface.

## Phase B — Route-2 Female And Controlled Human Instances

### Route-2 Batch Hardening Required Before Female/Attributes

The male v1 branch is a historically pinned recovery and must remain byte-stable
after it publishes. Female and attribute instances use new contract-driven v2
wrappers; they do not rewrite or pretend to reuse the male-only fitted/sanitation
history.

- [ ] Publish an immutable male `qualified_candidate_v1.json` at the avatar root
  that points to the actual successful direct/fitted/sanitized branch, static
  agent decision, dynamic review bundle, and dynamic agent decision without
  moving or copying artifacts. Female authenticates this pointer instead of a
  hard-coded direct-branch review path.
- [x] Add `route2_human_instance_contract_v1` and carry `asset_id`,
  `base_avatar_id`, source/FLUX/Pixal lineage, allowed fallback branch, canonical
  FRONT/up, and output root through every manifest. Reject gender/baseline
  mismatches before retarget.
- [x] Add a hash-locked static visual decision and require it before animation.
  Hat/eyewear cases additionally prove that their reviewed accessory pixels map
  to a rigid Head/Head-descendant skin region.
- [ ] Add contract-driven v2 direct/fitted/sanitizer/static wrappers with complete
  direct attempt-ledger/model/code/hygiene authentication and structured failure
  codes. Preserve the male v1 runner/auditor/fallback/base-runner hashes once the
  male sanitation manifest publishes.
- [x] Add an atomic no-replace Pixal attribute executor that consumes the approved
  RGBA decision, pins Pixal/DINO/code/environment/model inventories, stages the
  existing wrapper, performs GLB readback, and embeds the upstream attribute and
  base-avatar lineage in its manifest.
- [x] Add a serial, hash-authenticated Route-2 DAG/resume ledger. Blender commands
  use `--python-exit-code 1` and require a unique success sentinel plus manifest
  and GLB readback; return code alone is never success. A rejected instance writes
  a terminal result and execution continues with the next ordered case.
- [ ] Build the consolidated page from exactly nine immutable instance results
  (male, female, seven attributes). Qualified results show static and Walk/Idle
  media; early rejections show their terminal stage, evidence, and any available
  source/candidate/mask/diff media.

### Task 3: Route-2 Female Canary

**Files:**
- Reuse the Task-2 TokenRig, static-audit, retarget, renderer, and review modules with `asset_id=rocketbox_female_adult_01`.
- Input: `external/SPEAR/tmp/i23d_human_bakeoff_v1/pixal3d/rocketbox_female_adult_01/canary_1024_seed42.glb`
- Idle: `/data/datasets/rocketbox/Microsoft-Rocketbox/Assets/Animations/all_animations_max_motextr_static/f_idle_neutral_01.max.fbx`
- Output: `external/SPEAR/tmp/pixal_tokenrig_route2_v1/rocketbox_female_adult_01`

- [ ] Require the current male agent-QA decision and every male artifact hash before starting; do not wait for user input.
- [ ] Repeat the exact direct TokenRig static gate; use the fitted-skeleton fallback only after a recorded direct failure.
- [ ] Require the same semantic chains, PBR preservation, seam weights, rest/axis/floor checks, Walk/Idle GLB roundtrip, and ten review videos.
- [ ] Add independent female agent QA whose snapshot cannot inherit the male decision, then continue to attribute cases.

### Task 4: One-Attribute-At-A-Time FLUX.2 Human Edits

**Files:**
- Create: `external/SPEAR/tools/flux2_edit_human_attributes.py`
- Create: `external/SPEAR/tools/spike_rlr/human_attribute_review.py`
- Create: `external/SPEAR/tools/spike_rlr/human_attribute_review_server.py`
- Create: `external/SPEAR/tests/tools/test_flux2_edit_human_attributes.py`
- Create matching review contract/server tests.
- Create jobs: `external/SPEAR/tmp/human_attribute_instances_v1/jobs_v1.json`

**Cases, in order:** `tall_man`, `short_woman`, `glasses`, `hat`, `short_sleeve_color`, `trousers`, `shoes`.

- [ ] Bind jobs to the exact approved soft-T source image, FLUX.2 revision, source decision, prompt, negative prompt, seed, dimensions, steps, guidance, target semantic mask, and allowed pixel-difference region.
- [ ] Build masks without a new downloaded vision model. Store mask PNGs and human-readable overlays; require source pixels outside the allowed mask/dilation band to remain unchanged or below the fixed perceptual-drift threshold.
- [ ] For height cases allow silhouette/body-position changes but lock identity, face, hair, clothing details, camera, pose class, limb gaps, and non-target attributes. For glasses/hat/colors/trousers/shoes, change only that semantic region.
- [ ] Generate one case at a time, inspect full-resolution pixels, and expose source/candidate/mask/diff in the browser. Agent QA is sufficient to continue; a rejection writes `rejected` and moves to no 3D stage.
- [ ] After each 2D approval, run Pixal3D with the pinned wrapper, then the static TokenRig gate, Walk/Idle, GLB roundtrip, and ten-video review gate. Hat/glasses must be rigidly Head-bound and remain stable in both actions.
- [ ] Never replace a failed garment/accessory instance with Rocketbox geometry or silently alter a non-target region.

## Phase C — Route-1 Rocketbox Formal Baseline

### Task 5: Inventory And Register All 115 Base Avatars

**Files:**
- Create: `external/SPEAR/tools/rocketbox_inventory.py`
- Create: `external/SPEAR/tools/rocketbox_base_registry.py`
- Create: `external/SPEAR/tests/tools/test_rocketbox_inventory.py`
- Create: `external/SPEAR/tests/tools/test_rocketbox_base_registry.py`
- Generate: `external/SPEAR/data/rocketbox_humans_v1/inventory.{json,tsv}`
- Generate no-replace base entries under `external/SPEAR/data/source_assets_v1/human/rocketbox/<base_avatar_id>/asset.json`.

- [ ] Inventory the pinned Git tree and current files; require exactly 115 avatar FBXs, 74 male/41 female, and class totals 40 Adults/4 Children/71 Professions.
- [ ] Hash FBX, preview, every referenced texture/blob, license snapshot, skeleton, material slots, vertex/face counts, UVs, height, FRONT axis, floor, and available Walk/Idle compatibility.
- [ ] Keep source identity, body, skeleton, topology, weights, garment geometry, accessories, and official texture bytes immutable in the base entry.
- [ ] Reject missing textures, non-finite skin data, absent feet/toes, inconsistent orientation, or an asset whose category/gender cannot be established from authenticated source metadata.

### Task 6: Semantic Garment Masks And Deterministic Material Variants

**Files:**
- Create: `external/SPEAR/tools/rocketbox_semantic_masks.py`
- Create: `external/SPEAR/tools/rocketbox_material_intent.py`
- Create: `external/SPEAR/tools/rocketbox_material_variants.py`
- Create matching unit, Blender-static, and pixel tests.
- Output: `external/SPEAR/data/rocketbox_material_variants_v1/<base_avatar_id>/`

- [ ] Author and QA per-avatar semantic masks because the official three coarse slots are not garment masks. Required labels include skin, hair, headwear, eyewear, upper garment, lower garment, shorts, long trousers, shoes, and immutable/unknown.
- [ ] Require mask exclusivity/coverage, UV correspondence, preview overlays, and agent pixel QA before variants use a mask. Unknown pixels are immutable.
- [ ] Parse a restricted natural-language color/material intent into a typed parameter record with target semantic, color space value, transform strength, optional pattern-detail request, and seed.
- [ ] Deterministically recolor in Lab/HSV while preserving luminance, weave, seams, plaid/check patterns, garment boundaries, opacity alpha, and all pixels outside the semantic mask.
- [ ] Allow FLUX.2 only for mask-constrained texture detail. Composite the result into the source texture and prove outside-mask bytes are identical. Geometry and garment length/style never change.
- [ ] Render base/variant Front/Back/Side/Top and Walk/Idle QA; register only approved variants with `base_avatar_id`, full transform provenance, and no duplicated identity across splits.

### Task 7: Leakage-Safe Dataset Splits

**Files:**
- Create: `external/SPEAR/tools/build_asset_splits.py`
- Create: `external/SPEAR/tests/tools/test_build_asset_splits.py`
- Generate: `external/SPEAR/data/dataset_splits_v1/human_rocketbox.json`

- [ ] Deterministically assign `base_avatar_id` to train/validation/test using one pinned seed and stratify by gender plus Adults/Children/Professions.
- [ ] Resolve every material variant to its base split and fail on any base identity in more than one set.
- [ ] Record exact base and variant counts, seed, algorithm version, registry snapshot hash, and a stable rerun hash.

## Phase D — Human Apartment Audio-Video Examples

### Task 8: Human Scenario, Speech, And Render Contracts

**Files:**
- Extend: `external/SPEAR/tools/spike_rlr/human_apartment_scenarios.py`
- Extend: `external/SPEAR/tools/spike_rlr/speech_audio.py`
- Extend: `external/SPEAR/tools/spike_rlr/run_human_apartment_example.py`
- Extend matching tests under `external/SPEAR/tests/tools/spike_rlr/`.
- Output: `external/SPEAR/tmp/avengine_dataset_examples_v1/human/`

- [ ] Build deterministic single-person and multi-person cases covering male/female Walk/Idle combinations, straight paths, quadratic curves, turns, opposing travel, moving+stationary actors, and independent speech intervals.
- [ ] Select real gender-matched speech from an explicitly licensed LibriTTS or VCTK snapshot; record corpus/split/speaker/gender/utterance/transcript/license/hash and reject ambiguous gender or missing license/release metadata.
- [ ] Use SPEAR runtime APIs for spawning, animation, trajectory, listener/camera, and rendering. Use direct UE only for the already gated import/bake/editor/cook step.
- [ ] Verify body-facing versus path tangent, root motion, feet, collision, furniture occlusion, inter-person distance, animation rate readback, speech start/end, moving source position, rendered audio presence, and AV frame timing.
- [ ] Produce the established review composition: primary UE view, left trajectory panel, synchronized top-down, per-source labels, `sound N/T`, contact/speed/facing metrics, and muxed spatial audio.
- [ ] Browser-inspect every scenario snapshot with agent QA before dataset registration; do not pause for user input.

## Phase E — Formal Animal Audit And Replacement

### Task 9: Fail-Closed Audit Of Every Existing Animal

**Files:**
- Extend: `external/SPEAR/tools/spike_rlr/source_asset_audit.py`
- Create: `external/SPEAR/tools/spike_rlr/formal_animal_audit.py`
- Create/extend matching tests.
- Preserve inputs under: `external/SPEAR/data/source_assets_v1`
- Write new evidence under: `external/SPEAR/tmp/formal_animal_audit_v1/<asset_id>/`
- Write policy overlay: `external/SPEAR/data/source_assets_v1/formal_eligibility_v1.json`

- [ ] Audit every animal asset for source URL/snapshot, license, generation model, prompt/seed, artifact hashes, topology/components, UV/PBR, skeleton hierarchy, weights, FRONT/up, floor/contact, required Idle/Walking loops, collision, UE reload, audio mapping, and current review media.
- [ ] Re-open and inspect old videos for reversed body/travel direction, inverted/hovering feet, bad weights, sliding, excessive speed, loop freeze, collision, or misleading marker/audio evidence.
- [ ] Keep every existing asset byte unchanged. Store an audit manifest, contact sheets, and playable Front/Side/Top/Feet/Skeleton or species-equivalent media beside the new audit root.
- [ ] Treat all Hunyuan-derived animals as `technical_spike_only`/formal-ineligible regardless of legacy `approved` fields. Dataset resolution must consult the fail-closed overlay and exclude missing/failed audits.

### Task 10: Determine Legacy FLUX Provenance And Create FLUX.2 Animal Edits

**Files:**
- Create: `external/SPEAR/tools/animal_reference_provenance.py`
- Create: `external/SPEAR/tools/flux2_edit_animal_instances.py`
- Create: `external/SPEAR/tools/spike_rlr/animal_reference_review.py` and server.
- Create matching tests and jobs under `external/SPEAR/tmp/animal_flux2_migration_v1/`.

- [ ] Resolve each legacy reference to the actual FLUX version, checkpoint/revision, prompt, seed, source image, and license; mark unknown provenance as rejected for formal reuse.
- [ ] Starting only from a stable, license-usable reference, test one applicable attribute at a time: color, coat/feather pattern, body size/build, or species-specific feature. Preserve a bindable pose, scale/proportion contract, silhouette, and separated limbs.
- [ ] Bind each FLUX.2 output to input/mask/diff/model hashes and expose a 2D browser gate. Agent QA may pass the gate; rejected 2D instances never proceed.

### Task 11: Non-Hunyuan Animal 3D, Rig, Animation, And Promotion

**Files:**
- Create: `external/SPEAR/tools/animal_i23d_candidate.py`
- Extend/reuse: `external/SPEAR/tools/blender_robust_swap_mesh_keep_rig.py`, `external/SPEAR/tools/gate_check_animal.sh`, `external/SPEAR/tools/import_gate_animal_editor.py`.
- Harden: `external/SPEAR/tools/spike_rlr/promote_source_asset.py` and tests.

- [ ] Use the pinned non-Hunyuan 3D route (Pixal3D research candidate or a stable Quaternius species base/material route) recorded per job; never pass a Hunyuan mesh/texture/derivative to formal promotion.
- [ ] Audit static geometry/PBR, fit a species-correct skeleton or stable Quaternius rig, validate weights and floor/orientation, then run Idle/Walking dynamic QA and GLB roundtrip.
- [ ] Require import sentinel, correct rig family, animation readback, second UE reload, collision, review videos, license/provenance hashes, and an explicit agent visual decision before promotion.
- [ ] Promote with atomic no-replace IDs only when every 2D/3D/rig/animation/video/license gate is current; otherwise write `research_candidate` or `rejected`.

## Phase F — Animal And Mixed Apartment Examples

### Task 12: Audited Animal Apartment Examples

**Files:**
- Extend: `external/SPEAR/tools/spike_rlr/demo_scenarios.py`, `event_constraints.py`, and existing render/audio/review tools.
- Output: `external/SPEAR/tmp/avengine_dataset_examples_v1/animal/`

- [ ] Generate multiple examples using only assets resolved as formally eligible by the current audit overlay.
- [ ] Cover single/multiple animals, Walk/Idle, moving+stationary, straight/curve/turn/opposing paths, visible/occluded actors, and independent real animal audio events.
- [ ] Verify direction, feet, speed, collision, visibility, audio timing/position, top-down synchronization, and source-specific `sound N/T`.
- [ ] Publish primary view, left trajectory, synchronized top-down, audio, contact sheets, and QA manifest for browser approval.

### Task 13: Human-And-Animal Mixed Apartment Examples

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/mixed_apartment_scenarios.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_mixed_apartment_scenarios.py`
- Output: `external/SPEAR/tmp/avengine_dataset_examples_v1/mixed/`

- [ ] Cover one human+animal, multiple humans+animal, moving+stationary, opposing/crossing paths, and multiple speech/animal sound sources.
- [ ] Enforce human/animal clearance, furniture collision, visibility/occlusion, body-facing, feet/paws, speed, independent source IDs, speech gender mapping, audio position, and AV synchronization.
- [ ] Produce the same primary/trajectory/top-down/audio QA style and browser gate as the single-class examples.

## Phase G — Provenance, Batch Execution, And Final Acceptance

### Task 14: Unified Provenance And Asset-State Registry

**Files:**
- Create: `external/SPEAR/data/provenance_v1/registry.json`
- Create: `external/SPEAR/tools/provenance_registry.py`
- Create: `external/SPEAR/tests/tools/test_provenance_registry.py`
- Extend: `external/SPEAR/tools/spike_rlr/source_asset_registry.py`.

- [ ] Store separate license snapshots and authenticated metadata for Rocketbox, FLUX.2, Pixal3D, SkinTokens, motions, ReplicaCAD/SPEAR/UE scene dependencies, textures, animal sources, and every speech corpus/utterance.
- [ ] Record revision, URL, license hash, artifact hashes, commands, parameters, seeds, environment/runtime versions, derived-from graph, research/commercial flags, and redistribution/AI-use notes.
- [ ] Enforce the four states and allowed transitions. Formal resolution rejects Hunyuan derivatives, stale approvals, missing license hashes, SkinTokens provenance-risk assets not explicitly cleared, and any asset whose audit overlay is absent or failed.

### Task 15: Resumable Batch Orchestrator And Dataset Manifests

**Files:**
- Create: `external/SPEAR/tools/av_dataset_batch.py`
- Create: `external/SPEAR/tests/tools/test_av_dataset_batch.py`
- Create configs under: `external/SPEAR/configs/av_dataset_v1/`
- Output manifests under: `external/SPEAR/tmp/avengine_dataset_v1/`

- [ ] Implement stage DAGs for route-2 instances, route-1 variants, human scenes, animal scenes, and mixed scenes. Every node consumes immutable hashes and publishes atomically with no-replace semantics.
- [ ] Support dry-run, resume, bounded CPU/GPU workers, per-GPU ownership, failure quarantine, gate wait state, idempotent reruns, and a machine/human-readable ledger.
- [ ] Resolve only current approved assets and leakage-safe splits; include per-clip asset/source IDs, trajectories, visible intervals, audio intervals, transcripts/labels, camera/mic, render/audio hashes, and QA evidence.
- [ ] Never allow a successful subprocess, JSON field, or legacy approved value to bypass a human or license gate.

### Task 16: Full Verification And Final Acceptance Report

**Files:**
- Create: `external/SPEAR/tmp/avengine_dataset_v1/final_acceptance.json`
- Create: `docs/avengine_human_animal_dataset_route.md`
- Modify: `AGENTS.md`

- [ ] Run focused and full regression suites in their documented environments, model/baseline hash checks, all GLB readbacks, FFprobe/media decode checks, browser desktop/mobile QA, and registry/split consistency checks.
- [ ] Require a stable FLUX.2 → Pixal3D → TokenRig → Walk/Idle route, complete 115-avatar Rocketbox baseline and reproducible material variants, approved human/animal/mixed Apartment examples, reproducible human/animal prompt edits, and complete licenses/provenance/commands/parameters/QA.
- [ ] Report exact counts by human/animal/mixed, single/multi, Walk/Idle, split, source, and state; list every rejected/technical/research artifact with its reason and prove none resolves into the formal dataset.
- [ ] Final success requires no unresolved blocker, no stale approval, no cross-split identity leak, no Hunyuan formal dependency, and playable review media for every promoted asset and acceptance example.

## Locked Implementation Details From Existing-Code Audit

### Route-1 Catalog, Masks, Materials, And Splits

- Build the catalog in `external/SPEAR/tools/spike_rlr/rocketbox_catalog.py` from
  `external/SPEAR/tmp/human_motion_source_probe/rocketbox_tree.json` (tree-file
  SHA-256 `be09c1ea0fe7d4c4c79bfa119d30a1ee8db785d3b6f13ff0323efc38ae888663`).
  Count only non-facial `Assets/Avatars/{Adults,Children,Professions}/X/Export/X.fbx`.
  Do not count profession directories: two Party profession directories are
  facial-only duplicates of Adults and otherwise produce a false total of 117.
- `rocketbox_catalog.py` exposes
  `build_catalog_from_tree(tree_json)`, `canonical_base_avatar_id(name)`,
  `verify_materialized_catalog(catalog, source_root)`, and
  `catalog_counts(catalog)`. The required sentinel is
  `ROCKETBOX_CATALOG_OK total=115 male=74 female=41 adults=40 children=4 professions=71`.
- `external/SPEAR/tools/rocketbox_geometry_contract.py` and
  `blender_audit_rocketbox_catalog.py` must hash every skinned mesh, hierarchy,
  object matrix, shape key, bone order/parent/head/tail/roll/rest matrix, vertex
  position/weight, polygon/material index, UV, material node, and actual image.
  Do not assume the two-adult seven-texture layout: the official tree contains
  986 TGA paths/856 unique blobs and only 51/115 avatars match that simple set.
- Store semantic data under
  `/data/datasets/rocketbox/approved_baselines/rocketbox_115_v1/assets/<base_avatar_id>/semantic/`.
  Fixed labels are `unmapped`, `identity_protected`, `skin`, `hair`,
  `upper_garment`, `lower_garment`, `one_piece_garment`, `outerwear`,
  `footwear`, `headwear`, `accessory`, and `equipment`. Face/skin/eyes/hair and
  opacity are protected by default; shared-texel label conflicts or stale
  geometry hashes fail closed.
- `rocketbox_material_prompt.py` emits a canonical `MaterialEditPlan`; reject
  instructions that change sleeve/skirt/trouser length, shoe type, clothing or
  accessory presence, face, age, body, or gender. Reuse
  `human_template_fit.recolor_regions_preserve_luminance` for deterministic
  color and preserve alpha, normal, protected, and outside-mask pixels exactly.
- Texture-detail jobs use the locally pinned
  `Flux2KleinInpaintPipeline(image=..., mask_image=..., strength=0.35,
  num_inference_steps=28, guidance_scale=1.0, max_sequence_length=512)`.
  FLUX supplies weave/denim/leather detail only; deterministic code owns color,
  roughness, metallic, and the final outside-mask byte identity.
- Grouped split uses `sha256("rocketbox-115-v1\0" + base_avatar_id)` within
  gender/source-class strata. Fixed quotas are adult male `17/2/2`, adult
  female `15/2/2`, child male `1/0/1`, child female `1/1/0`, profession male
  `41/5/5`, and profession female `16/2/2`, yielding train/val/test `91/12/12`.
  Variants inherit the base split.
- Seal catalog, all 115 geometry contracts, masks, approved variants, reviews,
  licenses, split, and registry snapshot with the no-replace/fsync pattern from
  `tools/spike_rlr/rocketbox_baseline.py`.

### Human Apartment Runtime And Nine-Scenario Acceptance Set

- Add `tools/spike_rlr/humanoid_runtime_gate.py` and require exact
  `Walking`/`Standing_Idle`, runtime GLB/import-manifest hashes, second UE
  commandlet reload, Blueprint/animation paths, actor scale/Z lift/collision,
  and `walking_forward_yaw_offset_deg=90.0` from the registered asset.
- Preserve the proven runtime rules: select the populated skeletal component by
  maximum positive bone count; normalize UE bone punctuation; construct body
  forward as `right_vector × up_vector`; accept body-forward error only within
  25 degrees; write/read `GlobalAnimRateScale`; use 120 streaming plus 40
  camera-pose warmup frames.
- Pin LibriTTS license SHA-256
  `70279f4c750c20909fd1e2ba9cdc8ab379b7229aa47ed25ead16aa15af21c385`
  and `speakers.tsv` SHA-256
  `0a7e7ec38bbaead963fae92317bf2a4191645d831130717dc62444135635102a`.
  Initial male sample is
  `train-clean-100/2384/152900/2384_152900_000014_000011.wav`
  (`90dff63e...1832142`); initial female sample is
  `train-clean-100/1116/137572/1116_137572_000013_000004.wav`
  (`10cc8fe9...5b639`). `load_pinned_speech(gender, manifest, corpus_root)`
  verifies source identity gender, official speaker gender, transcript, file
  hash, and license; it has no animal/synthetic fallback.
- Generate exactly these canonical scenarios first:
  `solo_male_walk_straight`, `solo_female_walk_curve`, `solo_male_idle`,
  `solo_female_idle`, `solo_male_u_turn`, `solo_female_u_turn`,
  `pair_male_walk_female_idle`, `pair_female_walk_male_idle`, and
  `pair_counterflow_walk`.
- Canonical render is 960x720, 15 fps, 75 frames/5 seconds; Walking rate is
  0.65 and Idle 1.0; mic/camera is `[0.5,0.15,1.2]`, yaw 145 degrees, FOV 90
  degrees; adult collision radius is 0.35 m. Counterflow paths are
  male `[-3.8,1.925]→[-2.8,1.925]` and female
  `[-2.8,1.075]→[-3.8,1.075]`, with 0.85 m closest center distance and 0.15 m
  body clearance.
- Frame-to-sample mapping is
  `round(i*sample_rate/fps):round((i+1)*sample_rate/fps)`, producing continuous
  `[0,80000)` coverage. Require 2-channel/16 kHz/80,000-sample mixed audio,
  per-voiced-source solo audio, `effective_audio_frame_count >=45/75`, and
  peak <=0.99. Evaluate left/right with metadata azimuth and band-limited ILD,
  not broadband RMS.
- Auto-build the review page. Compose marker-annotated UE at left and synced
  top-down at right, hstack after `setpts=PTS-STARTPTS`, and FFprobe 75 frames,
  15 fps, 5 s within one frame, 1280x480 review size, two-channel 16 kHz audio,
  and zero audio/video start time.

### Formal Animal Pool, Motion Gate, And Multi-Source Contract

- The current registry's five animal IDs `dog_golden_0001`,
  `dog_beagle_0002`, `dog_pug_0001`, `cat_british_shorthair_0002`, and
  `cat_siamese_0001` are `legacy_internal_only`: all contain Hunyuan output and
  two also contain FLUX.1-dev. Preserve their bytes and legacy visual-approved
  fields, but make the new formal resolver exclude them.
- Create `source_asset_v2` plus `formal_asset_audit_v1` in
  `tools/spike_rlr/source_asset_schema.py`; formal manifests contain taxonomy,
  immutable artifacts, provenance, rights, rig/locomotion/collision/appearance
  profiles, reviews, formal scope, gate hash, and blockers. Unknown/missing
  rights, hashes, schema, license, review, or any absolute/tmp artifact path
  fails closed.
- Register the existing CC0 Quaternius Dog and Cat first as
  `dog_quaternius_0001` and `cat_quaternius_0001` through
  `register_quaternius_animals.py`. Hash and copy the original rigged GLBs into
  immutable registry directories, snapshot source/license, validate UV/material,
  skeleton/bones and actual `Idle`/`Walking` clips, and record forward offset
  0 degrees. Never claim a capability absent from the GLB.
- Animal FLUX.2 replacements use new IDs rather than overwriting legacy:
  `dog_golden_0002`, `dog_beagle_0003`, `dog_pug_0002`,
  `cat_british_shorthair_0003`, and `cat_siamese_0002`.
  Each candidate records input/review/mask/model/license/code/environment/output
  hashes and passes full-body, visible/separated limbs/tail, species, pattern,
  pose, and no-card/no-text QA before a non-Hunyuan 3D backend runs.
- `binding_report_v1` requires all vertices matched/weighted, <=4 influences,
  weight-sum error <=1e-5, p99 distance <=2% body diagonal, max <=5%, opposite
  limb contamination <=1%, required bones/actions, and preserved PBR.
- Walking body-forward median error is <=15 degrees and every sample <=25.
  Fixed-ground audit requires Idle contact >=95%, no unsupported Walking run
  longer than two audit frames, each foot enters stance each cycle, stance-foot
  slide <=max(0.15 m/s, 0.25*root_speed), root-Z drift <=0.03 m, and no
  non-jump/flying foot gap >0.03 m. UE must read back the computed play rate.
- Replace tag-keyed identity throughout scene/render/audio/metadata/review with
  unique `source_id`; `tag` remains only a UE compatibility field. The same
  `asset_id` may appear more than once if each placement has a distinct
  `source_id`. Formal audio has no synthetic fallback and stores per-source dry,
  wet binaural/FOA, license/sample hashes, active intervals, gain, RIR hash, and
  a mixed track reconstructable within one int16 LSB.
- Mixed acceptance starts with one formal human, Quaternius Dog, and Quaternius
  Cat; requires unique source IDs, human+animal, at least two animal species,
  Walk+Idle, >=0.9 m body clearance, no rig-assert bypass, 75 frames/15 fps,
  16 kHz/80,000-sample 2ch binaural plus 4ch FOA, three solo sources, synced
  review media, and one `MIXED_APARTMENT_ACCEPTANCE_OK` sentinel.

## Nonblocking Visual QA And User Acceptance

The user authorized the implementation to continue without waiting for review replies. For every former human gate, Codex must inspect the actual pixels/videos and record one of:

- `agent_qa_passed_pending_user_acceptance`: technically reasonable and allowed to feed the next stage;
- `rejected`: preserved with exact visual/metric reasons and never fed downstream.

Never write `user_approved` or reuse an old user decision. Route 2 must be completed first and collected into one consolidated browser acceptance surface for the user's next-day review. After route 2, execute route 1 and every remaining phase in order without blocking on user input. Final registration keeps reviewer kind and pending-user-acceptance status explicit so later user feedback can invalidate or approve the exact artifact snapshot.
