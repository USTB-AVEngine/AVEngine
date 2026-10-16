# Shared Limb Motion-Basis Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch an interactive review where the user can inspect exact `0/-90/+90/180` shared arm-and-leg motion bases and save one authenticated, non-formal correction for the next retarget.

**Architecture:** Replace per-bone rest-axis conjugation on bilateral limb chains with one canonical body/world-space pose-delta mapping. A single reviewer SO(3) yaw conjugates the shared delta and the leg endpoint/pole mapping while root, mesh, pelvis/spine/head, and rest matrices remain fixed. A Blender diagnostic builder precomputes four Walking candidates and five-view media; a Flask server switches them instantly and writes one snapshot-bound selection.

**Tech Stack:** Python 3.11, NumPy, pytest, Blender 4.2.1, glTF/GLB, EEVEE, FFmpeg/ffprobe, Flask, HTML/CSS/vanilla JavaScript.

## Global Constraints

- Never modify the sealed Rocketbox baseline, approved TokenRig static bundle, rejected attempts, or existing diagnostics.
- Use exactly the original Pixal PBR bind-pose GLB, sealed male Walking action, Blender 4.2.1, 30 fps, and all 33 frames.
- Affect only bilateral clavicle/upper-arm/forearm/hand and thigh/calf/foot/toe motion; lock armature object, root, pelvis, spine, neck, head, mesh, weights, rest matrices, PBR, and floor.
- Candidate yaws are exactly `0`, `-90`, `90`, and `180` degrees around canonical `UP +Z`.
- Publish to a new no-replace `retarget_motion_basis_review_v1`; keep it `technical_diagnostic_only` and `formal_dataset_asset: false`.
- A saved choice means only `selected_for_next_retarget`, never formal asset approval.
- Preserve every unrelated dirty-worktree change; no reset, checkout, clean, or rollback.
- Execute inline in the current user-authorized workspace; do not create a new worktree or use subagents.

---

### Task 1: Pure Shared-Basis and Four-Limb Metrics Contract

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/retarget_motion_basis_review.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_retarget_motion_basis_review.py`

**Interfaces:**
- Produces `yaw_matrix(degrees: int) -> np.ndarray`.
- Produces `compute_four_limb_motion_metrics(frames, fps) -> dict`.
- Produces `validate_review_bundle(path) -> dict`, `build_review_html(manifest) -> bytes`, and `record_selection(...) -> Path`.

- [ ] **Step 1: Write failing yaw and four-limb metric tests**

```python
@pytest.mark.parametrize("angle", [0, -90, 90, 180])
def test_yaw_is_proper_and_keeps_up(angle):
    value = review.yaw_matrix(angle)
    assert value.T @ value == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(value) == pytest.approx(1.0)
    assert value @ np.asarray((0.0, 0.0, 1.0)) == pytest.approx((0.0, 0.0, 1.0))

def test_metrics_detect_sideways_hand_and_foot_motion():
    source = review.compute_four_limb_motion_metrics(sagittal_frames(), fps=30)
    target = review.compute_four_limb_motion_metrics(sideways_frames(), fps=30)
    assert source["overall_classification"] == "four_limb_sagittal_motion"
    assert target["overall_classification"] == "sideways_limb_motion"
```

- [ ] **Step 2: Run Task 1 tests and verify RED**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_retarget_motion_basis_review.py
```

Expected: import failure because the module is absent.

- [ ] **Step 3: Implement allowed yaw matrices and arm/leg metrics**

Use the existing gait-plane contract for both legs. For each arm, use upper-arm,
forearm, and hand joint heads; measure hand forward/lateral excursion relative
to the shoulder and elbow-plane normal alignment with body lateral/forward.
Require forward-dominant excursion and lateral-dominant bend-plane normals for
all four limbs.

- [ ] **Step 4: Add failing bundle/selection tests**

Require four candidate IDs, five media views per candidate, one-action GLB
records, exact hashes/sizes, shared root/body invariant hashes, no symlinks or
extra files, and no formal approval fields. Test stale snapshot, unsupported
angle, pre-existing selection, and `none_of_the_candidates` failure closure.

- [ ] **Step 5: Implement bundle validation, HTML, and atomic no-replace selection**

```python
selection = {
    "schema": "retarget_motion_basis_correction_v1",
    "decision": "selected_for_next_retarget",
    "formal_dataset_asset": False,
    "scope": "bilateral_arm_and_leg_chains_only",
    "yaw_degrees": angle,
    "matrix_3x3": yaw_matrix(angle).tolist(),
    "candidate_bundle_manifest_sha256": manifest_sha256,
    "reviewer": reviewer,
    "reviewed_at": utc_now,
}
```

Fsync an exclusive staging file and rename no-replace. A
`none_of_the_candidates` choice stores no matrix and cannot feed formal retarget.

- [ ] **Step 6: Run Task 1 tests and commit**

```bash
git -C external/SPEAR add tools/spike_rlr/retarget_motion_basis_review.py \
  tests/tools/spike_rlr/test_retarget_motion_basis_review.py
git -C external/SPEAR commit -m "feat: add shared limb motion basis contract"
```

### Task 2: Replace Per-Bone Limb Conjugation with Shared Canonical Delta

**Files:**
- Modify: `external/SPEAR/tools/blender_retarget_rocketbox_to_tokenrig.py`
- Modify: `external/SPEAR/tests/tools/test_blender_retarget_rocketbox_to_tokenrig_static.py`

**Interfaces:**
- Adds `shared_canonical_limb_rotation(...) -> tuple[np.ndarray, dict]`.
- Adds keyword-only `limb_motion_basis_3x3=AXIS_MAP_3X3` to `bake_rest_corrected_action`.

- [ ] **Step 1: Write failing pure rotation tests**

```python
def test_identity_shared_limb_delta_preserves_canonical_world_axis():
    result, evidence = runner.shared_canonical_limb_rotation(
        source_rest=np.eye(3),
        source_pose=rotation_x(0.4),
        target_rest=rotation_z(0.7),
        source_base_rotation_3x3=np.eye(3),
        target_base_rotation_3x3=np.eye(3),
        motion_basis_3x3=np.eye(3),
    )
    target_delta = result @ rotation_z(0.7).T
    assert target_delta == pytest.approx(rotation_x(0.4), abs=1e-12)
    assert evidence["per_bone_rest_axis_conjugation_used"] is False
```

Add a `90` degree test proving `C @ D @ C.T`, and reflection/non-SO(3)
rejection.

- [ ] **Step 2: Run the new tests and verify RED**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/test_blender_retarget_rocketbox_to_tokenrig_static.py \
  -k shared_canonical_limb
```

Expected: FAIL because the helper is absent.

- [ ] **Step 3: Implement the shared canonical formula**

```python
source_rest_c = source_base @ source_rest
source_pose_c = source_base @ source_pose
source_delta_c = source_pose_c @ source_rest_c.T
corrected_delta_c = correction @ source_delta_c @ correction.T
target_pose_c = corrected_delta_c @ (target_base @ target_rest)
target_pose = target_base.T @ target_pose_c
```

Project every input/output to proper SO(3) and record reconstruction errors.

- [ ] **Step 4: Write a failing static scope test**

Require the helper for exactly the bilateral clavicle, upper-arm, forearm,
hand, thigh, calf, foot, and toe roles. Require legacy rest-aligned transfer for
pelvis/neck/head and local spine transfer. Require the same correction matrix
in source-driven leg endpoint/pole mapping. Forbid it from root/object/static
transforms.

- [ ] **Step 5: Implement the minimal bake integration**

Define `SHARED_CANONICAL_LIMB_ROLES`, branch by semantic role, pass the same
proper matrix to `map_source_leg_endpoint_rest_frame(axis_map_3x3=...)`, and
record `shared_limb_motion_basis` evidence including role list and locked root
scope.

- [ ] **Step 6: Run the full retarget static suite and commit**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/test_blender_retarget_rocketbox_to_tokenrig_static.py
git -C external/SPEAR add tools/blender_retarget_rocketbox_to_tokenrig.py \
  tests/tools/test_blender_retarget_rocketbox_to_tokenrig_static.py
git -C external/SPEAR commit -m "fix: transfer limb motion in one canonical basis"
```

### Task 3: Exact Four-Candidate Blender Builder

**Files:**
- Create: `external/SPEAR/tools/blender_build_retarget_motion_basis_review.py`
- Create: `external/SPEAR/tests/tools/test_blender_build_retarget_motion_basis_review_static.py`

**Interfaces:**
- Consumes the exact bind-pose/static-QA pair and sealed Walk blend/manifest.
- Produces `motion_basis_review_manifest.json` plus four candidate subtrees.

- [ ] **Step 1: Write failing static builder tests**

Require Blender `4.2.1`, authentication before/after, all four angles, the
shared bake keyword, 33-frame one-action GLB export/readback, all four-limb
metrics, root/body invariant equality, PBR/rest preservation, five views,
staging fsync/chmod, no-replace publication, and failure evidence.

- [ ] **Step 2: Run builder tests and verify RED**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/test_blender_build_retarget_motion_basis_review_static.py
```

- [ ] **Step 3: Implement exact candidate bake/export**

For every angle, reopen the sealed blend, import the approved bind pose, cache
Walking, and call:

```python
walking, bake = runner.bake_rest_corrected_action(
    bpy=bpy,
    target_armature=target,
    semantic=static_auth["semantic_mapping"],
    cached=walk_cache,
    action_name="Walking",
    target_base_transform=target_base,
    limb_motion_basis_3x3=review.yaw_matrix(angle),
)
```

Remove the source and export only `Walking`. Do not apply grounding/contact
repair in this pre-formal correction stage.

- [ ] **Step 4: Measure invariants and render media**

Sample all required arm/leg joints for 33 frames, compute four-limb metrics,
and hash armature object/root plus pelvis/spine/neck/head matrices. Require
those invariant hashes to match across candidates. Render 640x360
Front/Side/Top/Feet/Skeleton MP4+PNG using the existing review renderer and
probe every frame.

- [ ] **Step 5: Publish, validate, test, and commit**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_retarget_motion_basis_review.py \
  tests/tools/test_blender_build_retarget_motion_basis_review_static.py
git -C external/SPEAR add tools/blender_build_retarget_motion_basis_review.py \
  tests/tools/test_blender_build_retarget_motion_basis_review_static.py
git -C external/SPEAR commit -m "feat: build shared limb basis review candidates"
```

### Task 4: Interactive Hash-Locked Server

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/retarget_motion_basis_review_server.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_retarget_motion_basis_review_server.py`

**Interfaces:**
- Produces `create_app(bundle_dir, selection_dir)` and localhost port `8100`.

- [ ] **Step 1: Write failing route/UI tests**

Test root, manifest/metrics/media routes, range requests, no-store headers,
path traversal and tamper rejection, `405` on unsupported writes, and CSRF plus
manifest-snapshot enforcement on `POST /selection`. Require green `FRONT -Y`,
blue `UP +Z`, `0/±90/180` controls, five synchronized videos, arm/leg metrics,
and `none_of_the_candidates`.

- [ ] **Step 2: Run server tests and verify RED**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_retarget_motion_basis_review_server.py
```

- [ ] **Step 3: Implement instant exact-candidate switching and safe selection**

Validate the bundle at startup and before every file/selection operation. Keep
all videos within half a frame of the master. Preserve play/pause, time, and
rate while swapping candidate sources. Write only through `record_selection`.

- [ ] **Step 4: Run focused tests and commit**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_retarget_motion_basis_review.py \
  tests/tools/spike_rlr/test_retarget_motion_basis_review_server.py
git -C external/SPEAR add tools/spike_rlr/retarget_motion_basis_review_server.py \
  tests/tools/spike_rlr/test_retarget_motion_basis_review_server.py
git -C external/SPEAR commit -m "feat: add interactive shared basis correction UI"
```

### Task 5: Generate, Inspect, Launch, and Record

**Files:**
- Modify: `AGENTS.md`
- Modify: `.superpowers/sdd/pixal-tokenrig-route2-progress.md`
- Generate: `external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/retarget_motion_basis_review_v1/**`

- [ ] **Step 1: Run the exact builder in a persistent session**

```bash
/data/jzy/.local/bin/blender --background --python-exit-code 1 \
  --python external/SPEAR/tools/blender_build_retarget_motion_basis_review.py -- \
  --asset-id rocketbox_male_adult_01 \
  --bind-pose-glb external/SPEAR/tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01/fitted_skeleton_v1/sanitized_weights_v1/static_audit_v1/bind_pose.glb \
  --static-qa-json external/SPEAR/tmp/pixal_tokenrig_route2_v1/rocketbox_male_adult_01/fitted_skeleton_v1/sanitized_weights_v1/static_audit_v1/static_qa.json \
  --baseline-retarget-blend /data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1/rocketbox_male_adult_01/retarget.blend \
  --baseline-retarget-manifest /data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1/rocketbox_male_adult_01/retarget_manifest.json \
  --output-dir external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/retarget_motion_basis_review_v1
```

Expected sentinel: `RETARGET_MOTION_BASIS_REVIEW_OK`.

- [ ] **Step 2: Inspect pixels and all automatic evidence**

Decode all 20 MP4s, inspect representative frames in all five views, compare
arm/leg metrics, and verify root/body hashes are identical. Do not select a
candidate on the user's behalf.

- [ ] **Step 3: Launch and probe the UI**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python \
  external/SPEAR/tools/spike_rlr/retarget_motion_basis_review_server.py \
  --bundle-dir external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/retarget_motion_basis_review_v1 \
  --selection-dir external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/retarget_motion_basis_selection_v1 \
  --host 127.0.0.1 --port 8100
```

- [ ] **Step 4: Run final verification**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_retarget_motion_basis_review.py \
  tests/tools/spike_rlr/test_retarget_motion_basis_review_server.py \
  tests/tools/test_blender_build_retarget_motion_basis_review_static.py \
  tests/tools/test_blender_retarget_rocketbox_to_tokenrig_static.py
git diff --check
git -C external/SPEAR diff --check
```

- [ ] **Step 5: Update status without changing frozen decisions**

Record candidate metrics, hashes, URL, and user-controlled selection status.
Keep attempt two rejected and attempt three unstarted until a valid selection
file exists.
