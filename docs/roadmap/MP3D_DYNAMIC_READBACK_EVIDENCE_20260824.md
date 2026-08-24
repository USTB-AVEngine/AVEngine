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
