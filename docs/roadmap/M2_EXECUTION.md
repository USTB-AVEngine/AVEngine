# M2 Research-Candidate Review Execution

This runbook reproduces the bounded M2 review path. It does **not** run or
approve the formal M2 canary. Formal M2 status is `not_run` until a hash-bound
human review promotes the exact package to `canary_qualified` and a clean
formal capture succeeds.

## Fixed review contract

- One candidate dog package with baked Idle/Walk poses, semantic anchors and
  contact phases.
- Exactly 75 explicit states: 15 Idle, 45 Walk and 15 Idle at 15 Hz.
- No free-running animation, physics step or clock advancement between frames
  or modalities.
- One logical `camera_rig_0/view0`. RGB, depth and semantic share that view and
  one per-frame state.
- Review output declares `formal_view_ids: []`,
  `review_view_ids: ["view0"]`, `review_only: true` and
  `qualification_claim: false`.
- A `research_candidate` is accepted only by the separately named review
  loader. The formal loader continues to require `canary_qualified`.
- The articulated-object template must contain the exact Boolean opt-in
  `user_defined.avengine_native_gltf_skin_frame: true`. Capture uses a fresh
  Simulator and rejects a visual filepath reused by M1 because Habitat caches
  render assets by filepath.

This runbook records the executed world-left replacement candidate in
[M2_STATUS.md](M2_STATUS.md). The architecture and its fail-closed body-plan
boundary are in
[MOTION_RETARGETING.md](../architecture/MOTION_RETARGETING.md).

## Prerequisites and local layout

```bash
export REPO=/data/jzy/code/AVEngine-habitat-native
export RUNTIME=/data/jzy/code/habitat-sim-AVEngine
export HABPY=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
```

The runtime environment needs the pinned Bullet-enabled Habitat build,
`numpy-quaternion`, NumPy, Pillow, FFmpeg and a working headless GPU/EGL
context. The Blender custom-room M1 package and declared navmesh must already
pass M1 validation.

This bounded replay starts from ignored local intermediates:

```text
tmp/m2/rocketbox_motion_retarget_v2_a_rebased/visual.glb
tmp/m2/rocketbox_motion_retarget_v2_a_rebased/rebase.json
tmp/m2/rocketbox_motion_retarget_v2_a_rebased/deformation_verification.json
tmp/m2/rocketbox_motion_retarget_v2_a_actions/actions.npz
tmp/m2/rocketbox_motion_retarget_v2_a_actions/action_bake_report.json
tmp/m2/rocketbox_motion_retarget_v2_a_probe/probe.json
tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/review_report.json
tmp/m2/rocketbox_motion_retarget_v2_a_world_left_r2/retarget.json
tmp/m2/rocketbox_motion_retarget_v2_a_motion_qa/report.json
```

They are not committed. Their hashes are closed into package provenance, but
restoring every intermediate from a fresh clone is not yet a formal M2
completion procedure. The tools refuse to overwrite non-empty outputs; keep
the current evidence and use a fresh workspace when reproducing these exact
output names.

The source checkout is `/data/datasets/rocketbox/Microsoft-Rocketbox` at
revision `0943055db6ec570bcef9f2c8b41c9e5467c808f9`. The package compiler
rehashes the source FBX, textures, README and MIT license before use.

## 1. Run bounded automatic QA

```bash
"$HABPY" tools/motion/audit_m2_retarget.py \
  --visual-glb tmp/m2/rocketbox_motion_retarget_v2_a_rebased/visual.glb \
  --actions-npz tmp/m2/rocketbox_motion_retarget_v2_a_actions/actions.npz \
  --joint-mapping tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/habitat_joint_mapping.json \
  --profile examples/m2/motion_profiles/quadruped_dog_to_rocketbox_beagle_v1.json \
  --output tmp/m2/rocketbox_motion_retarget_v2_a_motion_qa/report.json

"$HABPY" tools/m2/audit_candidate.py \
  --visual-glb tmp/m2/rocketbox_motion_retarget_v2_a_rebased/visual.glb \
  --actions-npz tmp/m2/rocketbox_motion_retarget_v2_a_actions/actions.npz \
  --rebase-report tmp/m2/rocketbox_motion_retarget_v2_a_rebased/rebase.json \
  --output tmp/m2/rocketbox_motion_retarget_v2_a_auto_qa_r2
```

The profile-bound report must pass with SHA-256
`0c7531f9e605edb88978fc79f65dcca0fd3e0ed467af54baf49651e6c9d1aabb`.

The result must retain `qualification_state: research_candidate`,
`qualification_claim: false` and `human_visual_review_required: true`, even
when automatic QA is `pass`. For these exact replacement inputs the measured
`legacy_hind_gait_metric_triggered` value is `false`, so `known_limitations`
must be empty. A true metric must still retain the legacy warning; a false
metric must not claim the problem remains.

## 2. Compile the research-candidate package

```bash
"$HABPY" tools/m2/compile_animal_package.py \
  --repo-root "$REPO" \
  --rocketbox-root /data/datasets/rocketbox/Microsoft-Rocketbox \
  --evidence-directory tmp/m2/rocketbox_beagle_m2_package_inputs_v5_world_left_r3 \
  --output tmp/m2/rocketbox_beagle_m2_candidate_v5_world_left_r3 \
  --visual-glb tmp/m2/rocketbox_motion_retarget_v2_a_rebased/visual.glb \
  --rebase-report tmp/m2/rocketbox_motion_retarget_v2_a_rebased/rebase.json \
  --rebase-deformation-report tmp/m2/rocketbox_motion_retarget_v2_a_rebased/deformation_verification.json \
  --actions-npz tmp/m2/rocketbox_motion_retarget_v2_a_actions/actions.npz \
  --action-report tmp/m2/rocketbox_motion_retarget_v2_a_actions/action_bake_report.json \
  --habitat-static-probe tmp/m2/rocketbox_motion_retarget_v2_a_probe/probe.json \
  --habitat-animation-review tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/review_report.json \
  --static-qa tmp/m2/rocketbox_motion_retarget_v2_a_auto_qa_r2/static_geometry.json \
  --deformation-qa tmp/m2/rocketbox_motion_retarget_v2_a_auto_qa_r2/deformation.json \
  --animation-qa tmp/m2/rocketbox_motion_retarget_v2_a_auto_qa_r2/animation.json \
  --normalization-report tmp/m2/rocketbox_normalized_v2/normalization.json \
  --motion-profile examples/m2/motion_profiles/quadruped_dog_to_rocketbox_beagle_v1.json \
  --retarget-report tmp/m2/rocketbox_motion_retarget_v2_a_world_left_r2/retarget.json \
  --motion-qa-report tmp/m2/rocketbox_motion_retarget_v2_a_motion_qa/report.json \
  --asset-id rocketbox_dog_beagle_01_m2_v5_world_left_candidate \
  --body-plan-id quadruped_mammal_canid_v1 \
  --skeleton-revision rocketbox-beagle-skeleton-m2-v4-world-left \
  --weights-revision rocketbox-beagle-weights-m2-v3 \
  --action-revision rocketbox-beagle-idle-walk-world-left-v2
```

Expected bounded result:

```text
manifest SHA-256: 706631ee90ec9102bb76939dd7f75ca410757efd3c7c11580fa31e4d52183feb
admission_state: research_candidate
automatic_qa_status: pass
human_visual_review_status: not_run
human_review_binding_sha256: null
```

Compilation verifies and binds the replacement motion evidence. It does not
approve contact/root-speed quality, rights, a human decision or emit
`canary_qualified`.

## 3. Build the exact 75-state review request

```bash
"$HABPY" tools/m2/build_research_review_request.py \
  --asset-manifest tmp/m2/rocketbox_beagle_m2_candidate_v5_world_left_r3/asset_manifest.json \
  --room-manifest examples/m1/rooms/blender_custom/room_manifest.json \
  --room-request examples/m1/requests/blender_custom.json \
  --output tmp/m2/rocketbox_beagle_m2_review_request_v5_world_left_r3.json \
  --request-id rocketbox_beagle_m2_world_left_research_review_v5_r3
```

Expected request SHA-256:
`361924effa4ce7172102abaed353d4cf12af7ca059883d40f1e7b66c13dc3bbc`.
The builder checks all timing, state, joint/contact order, pose hash,
applied-state hash, single-view and modality constraints. Its only permitted
formal-validation error is that the package is not `canary_qualified`.

## 4. Run the Habitat review-only capture

```bash
"$HABPY" tools/m2/capture_research_review.py \
  --asset-manifest tmp/m2/rocketbox_beagle_m2_candidate_v5_world_left_r3/asset_manifest.json \
  --request tmp/m2/rocketbox_beagle_m2_review_request_v5_world_left_r3.json \
  --room-manifest examples/m1/rooms/blender_custom/room_manifest.json \
  --room-request examples/m1/requests/blender_custom.json \
  --runtime-root "$RUNTIME" \
  --output tmp/m2/rocketbox_beagle_m2_habitat_review_v5_world_left_r3
```

A successful review run reports 75 frames, world time `[0.0, 0.0]`, no formal
view IDs and one review view ID. It writes raw RGB/depth/semantic arrays,
per-frame state/readback records and three MP4 files.

```text
RGB      f789260e70a99b008685377b9d18d239d4bdbf6aa71fd20ccda4f09ee8bf03a9
depth    2b8302f3c896eb35480a6878cb4d8e717e3bc47835e15632495bb12c148cec4a
semantic f5414026b332e01576a41370a73ca4b8b9ab7b9b89cb1e1d45752afe33286d24
```

The current evidence canonical content hash is
`95ccffbb252eed0e40f37d2a44fb4c428147b0077c2177a63369420f9331b290`.
See [M2_STATUS.md](M2_STATUS.md) for the file-byte hash and measurements.

## 5. Human visual review checklist

Review the 75-frame RGB video together with the closer side and front-quarter
Walk/Idle diagnostics listed in [M2_STATUS.md](M2_STATUS.md). Record explicit
decisions for:

- mesh, skin and skeleton alignment across complete cycles;
- whether the replacement hind-leg whole-limb motion is anatomically
  plausible through complete cycles;
- front/hind paw sliding, floor penetration or visible hovering;
- limb collapse, triangle artifacts, cross-leg motion and joint plausibility;
- Idle stability and absence of visible mouth articulation.

Any decision must bind the exact asset-manifest, request and media hashes. The
decision must separately address the visible gait and the unresolved
contact/root-speed evidence. If rejected, keep formal M2 `not_run` and create a
new candidate rather than overwriting r3.

## 6. Every further motion change starts a new candidate

Do not overwrite any artifact named in this runbook. A profile-bound
change must use a new output root and new revisions for the retargeted
GLB, baked actions, motion/contact/root-speed/deformation reports, package,
request and media. It must remain `research_candidate` until its own exact
hashes complete human review.

Passing generic semantic-chain motion QA is not a substitute for four-paw
contact or root-speed QA. The replacement must prove that its action cadence,
contact phases and explicit `world_from_actor` trajectory agree before it can
be considered for a new Habitat review-only capture. Attribute variants also
require per-instance revalidation as described in
[MOTION_RETARGETING.md](../architecture/MOTION_RETARGETING.md).

## 7. Formal admission remains closed

Do not call the formal capture entrypoint with this package. It correctly
rejects `research_candidate`, dirty worktrees, a runtime commit different from
the lock, a runtime binary imported from another root, or a native binding
whose SHA-256 differs from the value pinned in `runtime.lock.yaml`.

After hash-bound human review and the remaining provenance/use decision,
create a new immutable package accepted as `canary_qualified`. Commit and clean
both repositories, update `runtime.lock.yaml`, and only then run formal capture.
Formal evidence must emit exactly `formal_view_ids: ["view0"]`, no QA-only
views, and pass every M2 exit criterion before [M2_STATUS.md](M2_STATUS.md) can
change from `not_run`.
