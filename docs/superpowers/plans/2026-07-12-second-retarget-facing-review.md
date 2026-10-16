# Second Retarget Facing Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish and launch a read-only browser review that lets the user decide whether the rejected second male Route-2 retarget walks sideways, reversed, or aligned with the bound Pixal body.

**Architecture:** A pure contract module authenticates the immutable rejected reconstruction and computes facing/travel metrics. A Blender 4.2.1 renderer samples the semantic rig and creates a top-down arrow/trajectory video in a no-replace diagnostic bundle. A small Flask server hash-checks the bundle and serves a synchronized four-view review page without any formal approval endpoint.

**Tech Stack:** Python 3.11, pytest, Blender 4.2.1 Python API, NumPy, FFmpeg/ffprobe, Flask, HTML/CSS/vanilla JavaScript.

## Global Constraints

- Never modify the existing second-attempt GLB, manifest, Front/Side/Feet media, rejected retarget record, or approved static audit.
- Publish only to `external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/second_attempt_facing_review_v1` with no-replace semantics.
- Keep classification `technical_diagnostic_only`, decision `rejected`, and `formal_dataset_asset: false`.
- The browser must not expose an Approve endpoint or write a formal/user approval record.
- Use exactly Blender 4.2.1, 30 fps, and all 33 frames of the one-action Walking GLB.
- Body-forward sign is authenticated from the static bind and canonical `FRONT -Y`; travel must never choose the sign.
- Preserve the dirty worktree and all unrelated user changes; do not reset, checkout, clean, or roll back.

---

### Task 1: Immutable Input Contract and Facing Math

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/second_retarget_facing_review.py`
- Test: `external/SPEAR/tests/tools/spike_rlr/test_second_retarget_facing_review.py`

**Interfaces:**
- Consumes: diagnostic directory, static QA path, rejected failure path, per-frame semantic joint points.
- Produces: `authenticate_second_attempt(...) -> dict`, `compute_facing_samples(...) -> dict`, `classify_alignment(dot: float | None) -> str`, `atomic_publish_bundle(...) -> Path`, and `validate_facing_bundle(...) -> dict`.

- [ ] **Step 1: Write failing authentication and math tests**

Test exact diagnostic schema/classification/decision, GLB/media SHA-256 records, bound rejected-failure SHA-256, `FRONT -Y`, and one-action Walking. Add vector fixtures where body faces `-Y` while travel is `-Y`, `+X`, and `+Y`, expecting `aligned`, `sideways`, and `reversed`. Add zero-displacement frames expecting `travel_undefined` and no fabricated dot product.

```python
def test_alignment_classes_are_independent_of_travel_sign_authentication():
    assert review.classify_alignment(0.99) == "aligned"
    assert review.classify_alignment(0.05) == "sideways"
    assert review.classify_alignment(-0.99) == "reversed"
    assert review.classify_alignment(None) == "travel_undefined"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  external/SPEAR/tests/tools/spike_rlr/test_second_retarget_facing_review.py
```

Expected: collection/import failure because the new module does not exist.

- [ ] **Step 3: Implement fail-closed authentication and pure metrics**

Implement finite 3D vector helpers, horizontal normalization, central-difference pelvis travel, shoulder/hip body-right averaging, `body_forward = body_right × UP`, bind-sign authentication against `(0, -1, 0)`, signed angle, dot product, per-frame classification, and aggregate median/worst/reversed ratios. Reject missing semantic roles, non-finite values, changed hashes, symlinks, extra GLB actions, and any input claiming a formal pass.

```python
def classify_alignment(dot: float | None) -> str:
    if dot is None:
        return "travel_undefined"
    if dot >= 0.5:
        return "aligned"
    if dot <= -0.5:
        return "reversed"
    return "sideways"
```

- [ ] **Step 4: Add no-replace publication and bundle revalidation tests**

Test staged directory rename, manifest/artifact hashing, rejection of a pre-existing destination, source tampering after publication, symlinks, and any `user_approved` or `formal_dataset_asset: true` field.

- [ ] **Step 5: Run Task 1 tests and commit**

Expected: all focused tests pass.

```bash
git -C external/SPEAR add tools/spike_rlr/second_retarget_facing_review.py \
  tests/tools/spike_rlr/test_second_retarget_facing_review.py
git -C external/SPEAR commit -m "feat: add second retarget facing contract"
```

### Task 2: Blender Top-Down Facing Renderer

**Files:**
- Create: `external/SPEAR/tools/blender_render_second_retarget_facing.py`
- Test: `external/SPEAR/tests/tools/test_blender_render_second_retarget_facing_static.py`
- Modify: `external/SPEAR/tests/tools/spike_rlr/test_second_retarget_facing_review.py`

**Interfaces:**
- Consumes: authenticated second diagnostic GLB and static semantic mapping from Task 1.
- Produces: `top_facing.png`, `top_facing.mp4`, `facing_metrics.json`, and `facing_review_manifest.json` in a staged bundle.

- [ ] **Step 1: Write failing renderer contract tests**

Assert pinned Blender version, exact one-action import, FPS-before-import, semantic roles `pelvis`, bilateral clavicles and thighs, 33-frame sampling, immutable source reauthentication before publish, and no-replace destination.

```python
def test_renderer_samples_body_basis_and_root_independently():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'semantic_bones["left_clavicle"]' in source
    assert 'semantic_bones["right_clavicle"]' in source
    assert 'semantic_bones["pelvis"]' in source
    assert "compute_facing_samples" in source
```

- [ ] **Step 2: Run focused renderer tests and confirm RED**

Run:

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  external/SPEAR/tests/tools/test_blender_render_second_retarget_facing_static.py
```

Expected: FAIL because the renderer is absent.

- [ ] **Step 3: Implement semantic sampling and animated overlays**

Import the diagnostic GLB at 30 fps, normalize only Blender's authenticated `Walking_Armature` suffix, and sample world-space pose heads. Create blue body-forward, red travel, and grey canonical `FRONT -Y` arrows as keyframed Blender objects; hide the travel arrow on undefined frames. Draw the complete pelvis/root trail and keep both the avatar and trajectory inside a fixed top camera.

- [ ] **Step 4: Render and validate media**

Render H.264 `640x360`, 30 fps, 33 frames. Decode all frames, verify nonblank output, exact duration/frame count, and presence of blue/red/grey overlay pixels whenever travel is valid. Store SHA-256, byte size, Blender version, FFmpeg version, command, per-frame metrics, and authenticated input records.

- [ ] **Step 5: Run Blender smoke into a disposable destination**

Run the pinned Blender executable with `--python-exit-code 1`; expected marker:

```text
SECOND_RETARGET_FACING_RENDER_OK
```

Visually inspect the generated PNG before publishing the canonical bundle.

- [ ] **Step 6: Run Task 1/2 tests and commit**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  external/SPEAR/tests/tools/spike_rlr/test_second_retarget_facing_review.py \
  external/SPEAR/tests/tools/test_blender_render_second_retarget_facing_static.py
```

Expected: all pass.

### Task 3: Synchronized Read-Only Review Page

**Files:**
- Modify: `external/SPEAR/tools/spike_rlr/second_retarget_facing_review.py`
- Modify: `external/SPEAR/tests/tools/spike_rlr/test_second_retarget_facing_review.py`

**Interfaces:**
- Consumes: validated bundle manifest, three immutable original videos, derived Top video, and per-frame metrics.
- Produces: hash-locked `review.html` embedded in the canonical bundle.

- [ ] **Step 1: Write failing HTML contract tests**

Require four labelled views, one master play/pause control, frame-step buttons, playback-rate control, synchronized `currentTime`, dynamic per-frame angle/dot/classification, the three colour legends, explicit rejected/diagnostic warnings, and observation buttons `sideways`, `reversed`, `aligned but deformed`. Assert the HTML contains no `Approve`, POST form, FBX link, or formal approval claim.

- [ ] **Step 2: Implement the static review page generator**

Generate semantic HTML/CSS/vanilla JavaScript with a 2x2 video grid. Use Top as the master clock and keep other views within half a frame. Update the evidence panel from an embedded immutable metrics payload. Store user observations in `localStorage` and offer a client-side JSON download only; do not send them to the server.

- [ ] **Step 3: Revalidate and run tests**

Rebuild the bundle in staging, add the HTML record to the manifest, and prove that changing HTML or any referenced video invalidates the bundle.

Expected: Task 1/3 tests pass.

### Task 4: Hash-Locked Flask Server and Live User Handoff

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/second_retarget_facing_review_server.py`
- Test: `external/SPEAR/tests/tools/spike_rlr/test_second_retarget_facing_review_server.py`
- Modify: `AGENTS.md`
- Modify: `.superpowers/sdd/pixal-tokenrig-route2-progress.md`

**Interfaces:**
- Consumes: canonical `second_attempt_facing_review_v1` bundle.
- Produces: localhost review URL and read-only hash-checked media/metrics routes.

- [ ] **Step 1: Write failing Flask tests**

Test `/`, four `/media/<view>` routes, `/metrics`, no-store headers, range requests, path traversal rejection, source tamper refusal, catalog change refusal, and `405` for POST/PUT/DELETE.

- [ ] **Step 2: Implement the server**

Validate the whole bundle at startup. Recheck the requested file hash immediately before serving it. Bind to `127.0.0.1`, default port `8098`, use `send_file(..., conditional=True, max_age=0)`, and provide no decision route.

- [ ] **Step 3: Run all focused tests and integrity checks**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  external/SPEAR/tests/tools/spike_rlr/test_second_retarget_facing_review.py \
  external/SPEAR/tests/tools/test_blender_render_second_retarget_facing_static.py \
  external/SPEAR/tests/tools/spike_rlr/test_second_retarget_facing_review_server.py
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [ ] **Step 4: Publish once and launch**

Generate the canonical bundle with the pinned Blender command, revalidate all hashes, launch the Flask process in a persistent terminal, probe the URL with `curl`, and open it in the user's browser. Report the URL, bundle path, automatic angle/dot summary, and the exact four human-visible checks without recording a verdict on the user's behalf.

- [ ] **Step 5: Update provenance/status documentation**

Record the second attempt's facing result, the distinction between bind-facing and travel-facing evidence, the rejected classification, commands, hashes, live review URL, and the fact that the user has resumed human-authoritative review.
