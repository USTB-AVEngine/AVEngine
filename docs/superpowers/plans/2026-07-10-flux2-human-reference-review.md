# FLUX.2 Human Reference Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seal the approved Rocketbox neutral-walk baseline, generate one hash-tracked FLUX.2 Klein male and female short-sleeve soft-T reference candidate, and expose a browser approval gate that blocks Hunyuan3D until both images are approved.

**Architecture:** Keep the approved motion baseline and generated image candidates outside git under `/data/datasets` and `external/SPEAR/tmp`, while keeping validation and review logic in small SPEAR tools. A pure Python manifest/review module owns hashes and stale-review rejection; the GPU runner owns only FLUX.2 loading and image generation; a Flask server owns only human-visible review and decisions.

**Tech Stack:** Python 3.9/3.11, pytest, Flask, Pillow, PyTorch 2.7, Diffusers `Flux2KleinPipeline`, FLUX.2 Klein 4B revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`.

## Global Constraints

- Do not use subagents for this execution.
- Do not process the other 66 Rocketbox locomotion actions.
- Active motions are `walk_neutral` for moving and `idle_neutral_01` for stationary; this plan does not bind idle yet.
- Hunyuan3D must not run before both FLUX.2 candidate reviews are approved and current.
- Use only `/data/models/hub/models--black-forest-labs--FLUX.2-klein-4B` with `local_files_only=True`.
- Preserve exact natural-language prompts without truncation; use `max_sequence_length=512`.
- Store generated assets outside the formal source registry. Hunyuan remains a technical spike.
- Use test-first development and atomic JSON replacement.

---

### Task 1: Immutable Rocketbox Baseline Seal

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/rocketbox_baseline.py`
- Test: `external/SPEAR/tests/tools/spike_rlr/test_rocketbox_baseline.py`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: `rocketbox_motion_review.assert_pair_approved(review_root: Path)` and the current male/female review directories.
- Produces: `seal_baseline(review_root: Path, output_root: Path) -> dict` and `/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1/baseline_manifest.json`.

- [ ] **Step 1: Write failing tests**

Cover exact pair approval, copied allowlisted artifacts, SHA-256/size records, atomic creation, identical rerun success, non-identical overwrite rejection, symlink rejection, and missing artifact rejection.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest \
  tests/tools/spike_rlr/test_rocketbox_baseline.py -q
```

Expected: import failure because `rocketbox_baseline.py` does not exist.

- [ ] **Step 3: Implement the seal**

Use this fixed artifact allowlist per asset:

```python
BASELINE_FILES = (
    "retarget.blend",
    "retarget.glb",
    "retarget_metrics.json",
    "retarget_manifest.json",
    "motion_review.json",
    "front.mp4",
    "side.mp4",
    "top.mp4",
    "joints.mp4",
    "feet.mp4",
    "source_target.mp4",
    "contact_sheet.png",
)
```

Build in a sibling temporary directory, fsync JSON, rename atomically, and refuse to replace a version whose manifest or copied bytes differ.

- [ ] **Step 4: Run tests and seal the real baseline**

Run the test command, then:

```bash
PYTHONPATH=tools/spike_rlr /data/jzy/miniconda3/envs/ss2/bin/python \
  tools/spike_rlr/rocketbox_baseline.py \
  --review-root tmp/rocketbox_motion_review \
  --output-root /data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1
```

Expected sentinel: `ROCKETBOX_BASELINE_SEALED`.

---

### Task 2: Hash-Locked Human Reference Contract

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/human_reference_review.py`
- Test: `external/SPEAR/tests/tools/spike_rlr/test_human_reference_review.py`

**Interfaces:**
- Produces: `write_candidate_manifest`, `validate_candidate_manifest`, `record_review`, `read_review_state`, `assert_reference_approved`, and `assert_pair_approved`.
- Candidate directory contract: `source.png`, `candidate.png`, `candidate_manifest.json`, optional `reference_review.json`.

- [ ] **Step 1: Write failing contract tests**

Require schema `human_reference_candidate_v1`, exact asset ID allowlist, model revision, prompt, seed, dimensions, steps, guidance, input/output SHA-256, output size, and source approval hash. Test traversal, symlink, malformed SHA, stale image, stale manifest, and partial-pair rejection.

- [ ] **Step 2: Run tests and verify RED**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest \
  tests/tools/spike_rlr/test_human_reference_review.py -q
```

- [ ] **Step 3: Implement atomic manifests and decisions**

Decisions use schema `human_reference_review_v1` and bind to both the candidate-manifest SHA-256 and current source/candidate image hashes. Regeneration deletes readiness by making the old hashes stale; it must never silently carry approval forward.

- [ ] **Step 4: Run the focused tests**

Expected: all tests pass.

---

### Task 3: FLUX.2 Klein Batch Reference Editor

**Files:**
- Create: `external/SPEAR/tools/flux2_edit_human_references.py`
- Create: `external/SPEAR/tmp/human_reference_review/jobs_v1.json`
- Test: `external/SPEAR/tests/tools/test_flux2_edit_human_references_static.py`

**Interfaces:**
- CLI: `--jobs-json`, `--output-root`, `--model-root`, `--local-files-only`.
- Jobs contain `asset_id`, `source_image`, `prompt`, `seed`, `width`, `height`, `steps`, and `guidance_scale`.
- Produces one candidate directory per asset via `human_reference_review.write_candidate_manifest`.

- [ ] **Step 1: Write failing static and pure tests**

Require one pipeline load for all jobs, `torch.bfloat16`, CUDA placement, `local_files_only=True`, `max_sequence_length=512`, exact prompt passthrough, 1152x1536 output preserving the approved images' 3:4 aspect ratio, deterministic generators, atomic image replacement, and provenance written only after a valid PNG exists.

- [ ] **Step 2: Run tests and verify RED**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest \
  tests/tools/test_flux2_edit_human_references_static.py -q
```

- [ ] **Step 3: Implement the GPU runner**

Load the pinned snapshot path once. Copy each approved source image to `source.png`, call `Flux2KleinPipeline(image=source, prompt=prompt, ...)`, save `candidate.png`, validate it with Pillow, then write the manifest.

- [ ] **Step 4: Define exact jobs**

Male prompt changes the polo/shorts to a solid dark forest-green short-sleeve crew-neck T-shirt, charcoal straight trousers, and neutral low-top sneakers. Female prompt changes the blouse/jeans to a solid deep burgundy short-sleeve crew-neck T-shirt, dark navy straight jeans, and neutral low-top sneakers. Both prompts explicitly preserve identity, front view, soft T/A-pose, proportions, arm/leg gaps, hands, feet, camera, and remove labels/background horizon.

- [ ] **Step 5: Run tests before GPU generation**

Expected: all focused tests pass.

---

### Task 4: Browser Reference Review

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/human_reference_review_server.py`
- Test: `external/SPEAR/tests/tools/spike_rlr/test_human_reference_review_server.py`

**Interfaces:**
- CLI: `--review-root`, `--host`, `--port`.
- Routes: `/`, `/asset/<asset_id>`, `/media/<asset_id>/<source|candidate>`, `/review/<asset_id>`, `/gate`.

- [ ] **Step 1: Write failing server tests**

Test read-only GETs, media allowlisting, range responses, no-cache headers, independent decisions, stale hash rejection, paired gate, safe redirects, and exact source/candidate image labels.

- [ ] **Step 2: Run tests and verify RED**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest \
  tests/tools/spike_rlr/test_human_reference_review_server.py -q
```

- [ ] **Step 3: Implement the compact review UI**

Show source and candidate side by side on desktop and stacked on mobile, with the exact prompt below them, male/female navigation, notes, and approve/reject controls. Do not show raw JSON as the review surface.

- [ ] **Step 4: Run server and contract tests**

Expected: all focused tests pass.

---

### Task 5: Generate, Inspect, And Open The Review Gate

**Files:**
- Generate: `external/SPEAR/tmp/human_reference_review/rocketbox_male_adult_01/{source,candidate}.png`
- Generate: `external/SPEAR/tmp/human_reference_review/rocketbox_female_adult_01/{source,candidate}.png`
- Generate: both `candidate_manifest.json` files

**Interfaces:**
- Requires the sealed Rocketbox baseline and complete FLUX.2 snapshot.
- Produces the next human checkpoint; no Hunyuan outputs are permitted yet.

- [ ] **Step 1: Verify the model snapshot**

Run the existing snapshot checker and require revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` with zero missing, incomplete, zero-size, or mismatched files.

- [ ] **Step 2: Generate both candidates on one GPU**

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=/data/models/hub \
  /data/jzy/miniconda3/envs/avengine-imagegen/bin/python \
  tools/flux2_edit_human_references.py \
  --jobs-json tmp/human_reference_review/jobs_v1.json \
  --output-root tmp/human_reference_review \
  --model-root /data/models/hub/models--black-forest-labs--FLUX.2-klein-4B \
  --local-files-only
```

- [ ] **Step 3: Inspect the actual pixels**

Reject and regenerate one seed at a time if the image is rear-facing, cropped, missing hands/feet, loses arm/torso gaps, merges legs, changes pose materially, contains text, or fails the requested short-sleeve edit. Do not hide a failed candidate behind automatic metrics.

- [ ] **Step 4: Start and browser-test the review server**

Start on the next free local port (prefer 8092), then use official Chrome through Playwright to verify both full-resolution images decode, prompts are complete, approve/reject controls fit desktop/mobile, and no layout overflow exists.

- [ ] **Step 5: Stop at the human gate**

Report the URL and exactly what to inspect. Do not start Hunyuan3D until both reference records are approved.

---

## Post-Approval Continuation

After both reference images pass, write the next focused implementation plan for Hunyuan3D candidate generation, component cleanup, part-aware Rocketbox weight transfer, palm-only fingers, approved `walk_neutral`, `idle_neutral_01`, GLB roundtrip, and the existing motion-review web evidence. If direct generated topology fails the first documented part-aware attempt, transition to stable Rocketbox-template fitting rather than stacking proxy meshes or trying unrelated animations.
