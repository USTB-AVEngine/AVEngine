# MP3D dynamic visual pose-readback checkpoint — 2026-08-24

## Scope and authority

This checkpoint is the CPU/mock implementation slice on
`codex/mp3d-dynamic-readback`, based on integration commit `37317f4`. It does
not claim a new Habitat capture, GPU execution, audio endpoint, accepted
viewpoint, or counted Episode. The next selected12 native run remains required.

The change preserves the existing current-visual fail-closed checks and the
legacy meaning of `arrays/actor_world_matrices.npy`: that array is the planned
`world_from_actor` transform after the per-instance offset, not runtime
readback.

## Newly persisted evidence for future native runs

Every one of the 75 video frames now retains these distinct arrays:

- planned applied skin-root matrices `[75,2,4,4]`;
- actual post-render articulated skin-root readback matrices `[75,2,4,4]`;
- actual agent/camera poses `[75,7]` from `_state_snapshot`;
- actual per-sensor poses `[75,sensor,7]` from `_state_snapshot`;
- logical listener poses `[75,7]` composed by `_logical_listener_pose` from the
  actual agent readback.

The pose-vector component order is translation XYZ in metres followed by
quaternion XYZW. The receipt records the sensor axis order. Frame records bind
each array index to `frame_index`, `pts_ticks`, actor/source/sensor axes, and
the same component order.

The receipt also records one identity object per actor/source axis: axis index,
actor ID, source ID, semantic ID, emitter link name, and the runtime Habitat
emitter link ID. Because this route is visual-only, audio endpoint status is
`pending` and endpoint ID is `null`; no endpoint is inferred.

## Consumer behavior

The offline review consumer recognizes the new evidence only when the complete
artifact and receipt metadata set is present. It verifies output containment,
distinct planned/actual files, 75-frame shapes, finite numeric values, sensor
order, frame/PTS alignment, actor/source identity bindings, and agreement of
frame-record positions/visibility with their arrays. Older captures remain
reviewable through the legacy path and are not relabeled as native pose
readback evidence.

## CPU/mock verification

- Focused capture/review tests: `29 passed`.
  Log: `/data/jzy/tmp/mp3d_dynamic_readback_unit_r1.log`.
- Adjacent current-MP3D tests: `80 passed`.
  Final log: `/data/jzy/tmp/mp3d_dynamic_readback_final_unit_r1.log`.
- Ruff check, Ruff format, Python byte compilation, and `git diff --check`
  passed for the four owned source/test files.

Tests explicitly cover snapshot-derived translation and yaw rather than M1
copying, planned/actual skin-root separation, source-axis swap rejection,
identity-binding corruption rejection, complete 75-frame shapes, non-finite
rejection, path containment, frame/PTS alignment, legacy compatibility, and
fresh-output no-clobber behavior.

## Remaining native evidence

No Habitat/GPU job was started in this slice. After integration, regenerate
fresh selected12 requests and run the native captures. Each resulting output
must be checked by the updated review consumer and visually inspected before
it can support a viewpoint-bank or mini-pilot claim.

## Follow-up: post-render authority and numeric pose validation

The follow-up commit on the same branch tightens the evidence boundary before
any native selected12 run:

- every frame now takes one snapshot before and one snapshot after the visual
  render; agent/camera pose, every named sensor pose, and world time must remain
  unchanged within explicit tolerances;
- only the post-render snapshot is persisted, and the logical listener is
  composed from that post-render agent readback;
- the receipt declares Habitat world XYZ with +Y up, `world_from_local`
  matrices, and translation in the last column;
- the logical listener entry explicitly states that it is derived from actual
  agent readback and is not a native listener-sensor readback;
- the reviewer requires unit XYZW quaternions and proper rigid 4x4 transforms:
  homogeneous last row, orthogonal 3x3 rotation, and determinant +1.

Fresh CPU/mock evidence after this follow-up:

- focused capture/review: `36 passed`, log
  `/data/jzy/tmp/mp3d_dynamic_readback_followup_unit_r2.log`;
- adjacent current-MP3D: `87 passed`, log
  `/data/jzy/tmp/mp3d_dynamic_readback_followup_adjacent_r2.log`;
- Ruff format/check, byte compilation, and diff check pass.

The new tests inject post-render agent and sensor movement, zero and non-unit
quaternions, a reflection, a non-orthogonal rotation, and a damaged homogeneous
last row. They also prove that the persisted pose is the post-render snapshot
and that a non-identity `rig_from_listener` is composed rather than copied.
No Habitat/GPU job was started by this follow-up.

## Follow-up: capture-time M1 listener/view identity

Completed current-visual receipts now contain an ordinary
`m1_capture_identity` value snapshot copied from the validated M1 request. It
retains request and room IDs, the complete primary camera rig and calibration,
the complete listener and `rig_from_listener`, source IDs in array-axis order,
and the request-time QA-view list. The existing input path remains provenance,
but later edits at that path cannot alter the receipt snapshot.

For new pose-readback captures, the reviewer cross-checks this snapshot against
the selected room, capture request ID, source and sensor axes, planned rig and
sensor calibration, all 75 actual agent/sensor poses, and a logical listener
recomputed from each actual agent pose plus the captured `rig_from_listener`.
Legacy captures without new pose-readback artifacts keep their previous review
path and are not upgraded.

Fresh CPU/mock evidence:

- focused capture/review: `40 passed`, log
  `/data/jzy/tmp/mp3d_capture_identity_unit_r2.log`;
- adjacent current-MP3D: `91 passed`, log
  `/data/jzy/tmp/mp3d_capture_identity_adjacent_r1.log`;
- Ruff format/check, byte compilation, and diff check pass.

Tests prove that rewriting the original M1 request path after capture does not
change the embedded identity or prevent review. Damaged room/source/rig
identity and an actual listener inconsistent with the captured listener offset
are rejected. No Habitat/GPU job was started.
