# Retarget Motion-Basis Correction Review Design

## Purpose

Replace the current read-only gait diagnosis with a human-correction stage.
The reviewer must be able to switch among exact shared limb-motion-basis
rotations, see the resulting animated Pixal body immediately, and persist the
chosen matrix as an authenticated input to the next retarget attempt.

This is not a formal asset approval. It is an upstream retarget-parameter
selection prompted by the user's observation that the body and root travel
forward while both legs and hands move in the wrong planes.

## Proven failure boundary and hypothesis

The sealed Rocketbox source Walking action is sagittal. Its left/right foot
lateral-to-forward excursion ratios are `0.1306/0.1537`, and its knee-plane
normals align with the body lateral axis at `0.9947/0.9943`. The rejected
second TokenRig result changes those ratios to `0.8385/0.9199` and rotates the
knee-plane normal toward body-forward. The same failure is present in the upper
limbs: source left/right hand lateral-to-forward excursion ratios are
`0.1970/0.1973`, while the rejected target changes them to `1.1267/0.4568`.
The left elbow-plane normal changes from primarily body-lateral (`0.8497`) to
primarily body-forward (`0.7962`). Body-forward and root travel remain aligned.

The shared failure boundary is the per-bone rest-axis conjugation in
`bake_rest_corrected_action`. The second attempt computes a different
source-to-target rest alignment for every fitted TokenRig bone; those unrelated
bone axes rotate a canonical forward/back animation delta into lateral motion.
The testable general solution is to express every arm and leg pose delta in one
shared canonical body/world frame, optionally conjugate that shared delta by a
single reviewer yaw around canonical `UP +Z`, and only then apply it to each
target rest bone. This leaves the Pixal mesh, pelvis/spine/head, armature object,
and root trajectory fixed.

A non-publishing identity-yaw probe already supports the hypothesis. Shared
canonical deltas reduce target hand excursion ratios to `0.2179/0.1725` and
leg ratios to `0.2407/0.1434`; both knees return to a sagittal classification.
This probe is evidence for the design, not a formal retarget or approved asset.

The first candidate set is exactly `0`, `-90`, `+90`, and `180` degrees. If no
candidate is visually and numerically plausible, the page records that result
and no correction can be confirmed; pitch/roll or arbitrary fine-tuning is not
silently introduced.

## Immutable inputs and non-destructive outputs

Inputs are authenticated but never modified:

- the sealed Rocketbox male neutral-walk baseline and manifest;
- the approved sanitized TokenRig static audit and `bind_pose.glb`;
- the rejected second-attempt evidence and existing diagnostic media;
- the exact retarget runner bytes used to build the candidates.

All generated artifacts publish to a new no-replace diagnostic bundle:

`external/SPEAR/tmp/pixal_tokenrig_route2_diagnostics_v1/rocketbox_male_adult_01/retarget_motion_basis_review_v1`

Each yaw candidate has a one-action Walking GLB, Front/Side/Top/Feet/Skeleton
MP4 and PNG media, per-frame arm-and-leg motion-plane measurements, and hashes.
The bundle is classified `technical_diagnostic_only`,
`formal_dataset_asset: false`, and cannot overwrite the second attempt or the
approved static audit.

The optional reviewer selection is written separately and atomically to:

`retarget_motion_basis_selection_v1/retarget_motion_basis_correction_v1.json`

That file is no-replace and binds the complete candidate-bundle snapshot. It
means only `selected_for_next_retarget`; it does not claim `user_approved` or
authorize dataset registration.

## Correction semantics

The reviewer correction is a proper SO(3) rotation around canonical `+Z`:

`C(theta) = [[cos,-sin,0],[sin,cos,0],[0,0,1]]`

For every mapped arm/leg bone, the source canonical delta is:

`D_c = (B_s S_pose) (B_s S_rest)^T`

The selected shared correction and target pose are:

`D_c' = C D_c C^T`

`T_pose = B_t^T D_c' (B_t T_rest)`

Here `B_s` and `B_t` are the authenticated source and target object-to-canonical
rotations. The same `C` is also applied to hip-relative leg endpoint and
knee-pole directions before the two-bone solve. The affected semantic roles
are bilateral clavicle, upper arm, forearm, hand, thigh, calf, foot, and toe.
It does not change:

- target armature object location, rotation, or scale;
- root translation or root rotation reconstruction;
- pelvis/spine/neck/head pose transfer;
- the static mesh, skeleton rest matrices, skin weights, PBR graph, or floor;
- the sealed source motion.

Every candidate records readback hashes for the root trajectory, body-forward
samples, static mesh/PBR contract, and rest matrices. Candidate publication
fails if those locked quantities differ across yaw values.

## Browser interaction

The page reuses the visual language of the former Hunyuan direction UI but
uses current coordinates:

- green arrow: semantic `FRONT -Y`;
- blue arrow: `UP +Z`;
- left/right curved buttons: `-90/+90` shared arm-and-leg basis yaw;
- flip button: `180`;
- reset button: `0`.

All four candidates are generated with the exact Blender path before launch.
Button clicks therefore switch local videos immediately rather than showing a
browser-only transform that the production retarget cannot reproduce. The five
views remain synchronized and the page reports hand/foot excursion ratios plus
elbow/knee-plane alignment for the active candidate.

The reviewer can either:

1. confirm the active yaw for the next retarget; or
2. record `none_of_the_candidates`, which prevents a third retarget from using
   a guessed orientation.

Confirmation uses a server-held CSRF token plus the submitted bundle manifest
hash. A stale page, changed artifact, unsupported angle, repeated write, or
path traversal is rejected.

## Validation and handoff

- Unit tests cover yaw matrices, proper handedness, allowed angles, correction
  payload validation, root/body invariance, and no-replace selection.
- Static tests prove the exact correction parameter reaches all and only the
  bilateral arm/leg chains and is recorded in bake evidence.
- Blender generation imports the exact static and sealed Walk inputs, exports
  a one-action GLB, reads it back, samples every frame, and produces the five
  review views at 30 fps.
- Candidate metrics extend the already-reviewed gait-plane implementation to
  elbow/hand trajectories. A candidate is automatically plausible only if all
  four limbs have forward-dominant excursion and elbow/knee planes remain
  predominantly sagittal; automatic plausibility never substitutes for the
  user's visual selection.
- The Flask server revalidates bundle hashes at startup and immediately before
  every media/selection operation.
- After a selection exists, the next formal retarget must consume that exact
  correction JSON and rerun the unchanged grounding, penetration, loop,
  deformation, PBR, GLB-readback, and five-view media gates.
