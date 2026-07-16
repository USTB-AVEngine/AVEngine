# M2 Canary Admission and Formal Capture

This runbook records the executed M2 path from a fail-closed research candidate
to one `canary_qualified` package and a clean formal Habitat capture. The exact
v7/r5 run is `pass`; see [M2_STATUS.md](M2_STATUS.md) for immutable hashes and
measurements.

## Fixed contract

- One Beagle package with baked Idle/Walk poses, semantic anchors and explicit
  four-paw contact phases.
- Exactly 75 states: 15 Idle, 45 Walk and 15 Idle at 15 Hz.
- No free-running animation, physics step or clock advancement.
- One logical `camera_rig_0/view0`; RGB, depth and semantic are co-located
  modalities, not multiple viewpoints.
- Both requests carry `view_ids: ["view0"]`. Review evidence emits
  `formal_view_ids: []` and `review_view_ids: ["view0"]`; formal evidence emits
  the inverse. The review loader accepts only a `research_candidate`, while
  the formal loader accepts only a `canary_qualified` package.
- Named outputs are immutable by policy. Commands require absent or empty fresh
  destinations and do not replace non-empty prior evidence.
- Formal capture rejects dirty AVEngine/Habitat worktrees, a runtime commit or
  native binding that differs from `runtime.lock.yaml`, and a binding imported
  from a different runtime root.

## Environment

```bash
export REPO=/data/jzy/code/AVEngine-habitat-native
export RUNTIME=/data/jzy/code/habitat-sim-AVEngine
export HABPY=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
```

The environment needs the pinned Bullet-enabled Habitat build,
`numpy-quaternion`, NumPy, Pillow, FFmpeg and a working headless EGL context.
The Blender custom-room M1 manifest, request and declared navmesh must already
pass M1 validation.

This executed run starts from ignored, hash-bound outputs of the offline
retarget/rebase/bake/probe/QA path. They are not distributed by ordinary Git:

```text
tmp/m2/rocketbox_motion_retarget_v2_a_rebased/visual.glb
tmp/m2/rocketbox_motion_retarget_v2_a_rebased/rebase.json
tmp/m2/rocketbox_motion_retarget_v2_a_rebased/deformation_verification.json
tmp/m2/rocketbox_motion_retarget_v2_a_actions/actions.npz
tmp/m2/rocketbox_motion_retarget_v2_a_actions/action_bake_report.json
tmp/m2/rocketbox_motion_retarget_v2_a_probe/probe.json
tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/review_report.json
tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/habitat_joint_mapping.json
tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/emitter_anchors.json
tmp/m2/rocketbox_motion_retarget_v2_a_world_left_r2/retarget.json
tmp/m2/rocketbox_motion_retarget_v2_a_motion_qa/report.json
```

The source checkout is `/data/datasets/rocketbox/Microsoft-Rocketbox` at
revision `0943055db6ec570bcef9f2c8b41c9e5467c808f9`. Compilation rehashes the
source FBX, textures, README and MIT license.

## 1. Run profile and package QA

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

Automatic QA must pass but must retain `research_candidate`,
`qualification_claim: false` and `human_visual_review_required: true`.

## 2. Derive cadence-locked world contacts

The old actor-space sliding warnings are diagnostic inputs, not a waiver. The
separate audit fits the root trajectory and tests world-space contact motion:

```bash
"$HABPY" tools/m2/audit_world_contacts.py \
  --visual-glb tmp/m2/rocketbox_motion_retarget_v2_a_rebased/visual.glb \
  --actions-npz tmp/m2/rocketbox_motion_retarget_v2_a_actions/actions.npz \
  --joint-mapping tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/habitat_joint_mapping.json \
  --anchors tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/emitter_anchors.json \
  --contacts-output tmp/m2/rocketbox_beagle_m2_world_contact_v2/contact_phases.json \
  --audit-output tmp/m2/rocketbox_beagle_m2_world_contact_v2/world_contact_audit.json
```

The executed audit selected `0.0198 m/frame`, or `0.297 m/s`, and measured a
maximum four-paw world contact step of `0.013894547981602673 m` against the
`0.015 m` gate. The audit hash-binds the visual, baked actions and emitted
contact phases.

## 3. Compile a new research candidate

```bash
"$HABPY" tools/m2/compile_animal_package.py \
  --repo-root "$REPO" \
  --rocketbox-root /data/datasets/rocketbox/Microsoft-Rocketbox \
  --evidence-directory tmp/m2/rocketbox_beagle_m2_package_inputs_v7_world_contact_r5 \
  --output tmp/m2/rocketbox_beagle_m2_candidate_v7_world_contact_r5 \
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
  --contact-report tmp/m2/rocketbox_beagle_m2_world_contact_v2/contact_phases.json \
  --world-contact-audit tmp/m2/rocketbox_beagle_m2_world_contact_v2/world_contact_audit.json \
  --asset-id rocketbox_dog_beagle_01_m2_v7_world_contact_candidate \
  --body-plan-id quadruped_mammal_canid_v1 \
  --skeleton-revision rocketbox-beagle-skeleton-m2-v4-world-left \
  --weights-revision rocketbox-beagle-weights-m2-v3 \
  --action-revision rocketbox-beagle-idle-walk-world-left-v2
```

The manifest must still say `admission_state: research_candidate`; compilation
cannot self-approve human review or rights.

## 4. Build and capture the exact review request

```bash
"$HABPY" tools/m2/build_research_review_request.py \
  --asset-manifest tmp/m2/rocketbox_beagle_m2_candidate_v7_world_contact_r5/asset_manifest.json \
  --room-manifest examples/m1/rooms/blender_custom/room_manifest.json \
  --room-request examples/m1/requests/blender_custom.json \
  --world-contact-audit tmp/m2/rocketbox_beagle_m2_world_contact_v2/world_contact_audit.json \
  --output tmp/m2/rocketbox_beagle_m2_review_request_v7_world_contact_r5.json \
  --request-id rocketbox_beagle_m2_world_contact_research_review_v7_r5

"$HABPY" tools/m2/capture_research_review.py \
  --asset-manifest tmp/m2/rocketbox_beagle_m2_candidate_v7_world_contact_r5/asset_manifest.json \
  --request tmp/m2/rocketbox_beagle_m2_review_request_v7_world_contact_r5.json \
  --room-manifest examples/m1/rooms/blender_custom/room_manifest.json \
  --room-request examples/m1/requests/blender_custom.json \
  --runtime-root "$RUNTIME" \
  --output tmp/m2/rocketbox_beagle_m2_habitat_review_v7_world_contact_r5
```

Review evidence must report 75 frames, world time `[0.0, 0.0]`, no formal
views, one review `view0`, and three readback-verified modalities.

## 5. Human review and promotion

Review complete cycles for mesh/skin alignment, rear-leg whole-limb motion,
paw sliding/penetration/hovering, limb collapse or crossing, Idle stability
and the no-mouth policy. A decision must bind exact hashes; a changed mesh,
action, trajectory or media starts a new candidate.

For this run the user accepted rear-leg naturalness on the r3 diagnostic. The
r3 and v7/r5 packages have identical visual and action hashes; v7/r5 adds the
passing cadence/root trajectory. Promotion also binds the final review media,
world-contact audit and local MIT source snapshot:

```bash
"$HABPY" tools/m2/promote_canary.py \
  --candidate-manifest tmp/m2/rocketbox_beagle_m2_candidate_v7_world_contact_r5/asset_manifest.json \
  --review-request tmp/m2/rocketbox_beagle_m2_review_request_v7_world_contact_r5.json \
  --capture-evidence tmp/m2/rocketbox_beagle_m2_habitat_review_v7_world_contact_r5/evidence.json \
  --world-contact-audit tmp/m2/rocketbox_beagle_m2_world_contact_v2/world_contact_audit.json=355e52e289dccc202b0d928f4d5969ba6f32c4789b9de7977c3993e912b7a297 \
  --diagnostic-video tmp/m2/rocketbox_beagle_m2_habitat_review_v5_world_left_r3/review_media/view0_rgb_review.mp4=f789260e70a99b008685377b9d18d239d4bdbf6aa71fd20ccda4f09ee8bf03a9 \
  --rocketbox-root /data/datasets/rocketbox/Microsoft-Rocketbox \
  --output tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5 \
  --reviewer-id workspace_user
```

The result must validate as `canary_qualified`, with automatic and human review
both `pass`, `allowed_use: research_canary`, `redistribution: allowed`, and
formal dataset registration still unauthorized.

## 6. Build and run the formal canary

Commit all implementation changes first. Both repositories must be clean, the
Habitat commit and native binding must match the lock, and imports must resolve
inside `$RUNTIME`.

```bash
"$HABPY" tools/m2/build_canary_request.py \
  --asset-manifest tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json \
  --room-manifest examples/m1/rooms/blender_custom/room_manifest.json \
  --room-request examples/m1/requests/blender_custom.json \
  --world-contact-audit tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/admission/world_contact_audit.json \
  --output tmp/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json \
  --request-id rocketbox_beagle_m2_formal_canary_v7_r5

"$HABPY" tools/m2/capture_canary.py \
  --asset-manifest tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json \
  --request tmp/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json \
  --room-manifest examples/m1/rooms/blender_custom/room_manifest.json \
  --room-request examples/m1/requests/blender_custom.json \
  --runtime-root "$RUNTIME" \
  --output tmp/m2/rocketbox_beagle_m2_formal_capture_v7_world_contact_r5
```

The executed formal run used clean AVEngine commit `b3d3a63` and clean Habitat
commit `bcca512a`. It returned `status: pass`, exactly `formal_view_ids:
["view0"]`, no review views, 75 frames, zero physics steps and world time
`0.0 -> 0.0`.

## 7. Reproduction and non-claims

Never overwrite a path named here. A rerun uses fresh output names and produces
new evidence identities. Successful M2 execution does not admit later species,
appearance variants or a dataset; those must repeat their applicable
asset/motion/contact/visual gates. It also does not exercise RLR audio or close
M3-M6.
