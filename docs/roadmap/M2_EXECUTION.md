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
tmp/m2/rocketbox_rebased_v3/visual.glb
tmp/m2/rocketbox_rebased_v3/rebase.json
tmp/m2/rocketbox_rebased_v3/deformation_verification.json
tmp/m2/rocketbox_actions_v1/actions.npz
tmp/m2/rocketbox_actions_v1/action_bake_report.json
tmp/m2/rocketbox_rebased_v3_probe_optin/probe.json
tmp/m2/rocketbox_habitat_review_v4/review_report.json
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
"$HABPY" tools/m2/audit_candidate.py \
  --visual-glb tmp/m2/rocketbox_rebased_v3/visual.glb \
  --actions-npz tmp/m2/rocketbox_actions_v1/actions.npz \
  --rebase-report tmp/m2/rocketbox_rebased_v3/rebase.json \
  --output tmp/m2/rocketbox_auto_qa_v1
```

The result must retain `qualification_state: research_candidate`,
`qualification_claim: false`, the known hind-gait limitation and
`human_visual_review_required: true`, even when automatic QA is `pass`.

## 2. Compile the research-candidate package

```bash
"$HABPY" tools/m2/compile_animal_package.py \
  --repo-root "$REPO" \
  --rocketbox-root /data/datasets/rocketbox/Microsoft-Rocketbox \
  --evidence-directory tmp/m2/rocketbox_beagle_m2_package_inputs_v4 \
  --output tmp/m2/rocketbox_beagle_m2_candidate_v4
```

Expected bounded result:

```text
manifest SHA-256: 4110e116ba9a3190caad40e8f8fa91fa49a02d2477dee25138481add5ac433bd
admission_state: research_candidate
automatic_qa_status: pass
human_visual_review_status: not_run
human_review_binding_sha256: null
```

Compilation does not repair the gait, approve rights, infer a human decision
or emit `canary_qualified`.

## 3. Build the exact 75-state review request

```bash
"$HABPY" tools/m2/build_research_review_request.py \
  --asset-manifest tmp/m2/rocketbox_beagle_m2_candidate_v4/asset_manifest.json \
  --room-manifest examples/m1/rooms/blender_custom/room_manifest.json \
  --room-request examples/m1/requests/blender_custom.json \
  --output tmp/m2/rocketbox_beagle_m2_review_request_v3.json
```

Expected request SHA-256:
`f6f2b812291ff14bb02dbda17d2bcbd55d468667f2195ef6dc062c0af7302c4d`.
The builder checks all timing, state, joint/contact order, pose hash,
applied-state hash, single-view and modality constraints. Its only permitted
formal-validation error is that the package is not `canary_qualified`.

## 4. Run the Habitat review-only capture

```bash
"$HABPY" tools/m2/capture_research_review.py \
  --asset-manifest tmp/m2/rocketbox_beagle_m2_candidate_v4/asset_manifest.json \
  --request tmp/m2/rocketbox_beagle_m2_review_request_v3.json \
  --room-manifest examples/m1/rooms/blender_custom/room_manifest.json \
  --room-request examples/m1/requests/blender_custom.json \
  --runtime-root "$RUNTIME" \
  --output tmp/m2/rocketbox_beagle_m2_habitat_review_v4
```

A successful review run reports 75 frames, world time `[0.0, 0.0]`, no formal
view IDs and one review view ID. It writes raw RGB/depth/semantic arrays,
per-frame state/readback records and three MP4 files.

```text
RGB      e0af301789bb0e1ae897cd391e8757c65bb64458ce0ef1d78d4f18ad85d62bd3
depth    c9169127794b7c50dc11f521a7d1e16aca2ce4fac273dd87ad449d447de7258f
semantic f1b6c8f72bf7a492108b19f63ff68ccb7e6401b22c378f135c018f7f57c6c388
```

The current evidence canonical content hash is
`23a22f0b2b1b89c20a2ba364813d556a4cefbe4c88947304fa42fc38ec738029`.
See [M2_STATUS.md](M2_STATUS.md) for the file-byte hash and measurements.

## 5. Human visual review checklist

Review the 75-frame RGB video together with the closer side and front-quarter
Walk/Idle diagnostics listed in [M2_STATUS.md](M2_STATUS.md). Record explicit
decisions for:

- mesh, skin and skeleton alignment across complete cycles;
- whether the known hind-leg whole-limb under-articulation and lateral/toe
  motion is tolerable for this bounded canary;
- front/hind paw sliding, floor penetration or visible hovering;
- limb collapse, triangle artifacts, cross-leg motion and joint plausibility;
- Idle stability and absence of visible mouth articulation.

If accepted temporarily, the review artifact must say that the legacy gait
limitation remains unfixed and bind the exact asset-manifest, request and media
hashes. If rejected, keep formal M2 `not_run` and use the user's replacement
gait references before generating a new candidate.

## 6. Formal admission remains closed

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
