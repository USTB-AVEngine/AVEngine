# MP3D region actor-track materializer checkpoint (2026-08-24)

## Scope

This CPU-only slice starts from integration `e16f217` in the server worktree
`/data/jzy/code/AVEngine-codex-mp3d-region-m2-materializer`, branch
`codex/mp3d-region-m2-materializer`.

Code commits:

- `4041dffc1fac4cd558c9ec9c425dfba46f41e304`: initial actor-track materializer;
- `38a695afc0e3bb03cc2b7aa51b242c8c9e7c91d7`: r4 authority, camera-reference,
  and motion-span fail-closed checks.

Owned files are:

- `src/avengine/m5/mp3d_region_actor_track_materializer.py`;
- `tools/m5/materialize_mp3d_region_actor_tracks.py`; and
- `tests/unit/test_mp3d_region_actor_track_materializer.py`.

No M2 schema, M2 validator, camera bank, region planner, or current visual
runner was changed.

## Formal M2 boundary

The existing M2 request schema is single-asset and has one root transform and
one joint-state list per frame. The existing validator also requires every
75-state request to observe both `idle` and `walk`. Therefore a formal M2
request cannot represent two independent actor root/action tracks, and a
strict all-idle static request would fail the existing validator.

This slice does not weaken or bypass that contract. It emits
`actor_track.json`, not M2 capture requests. Every track and case manifest
contains:

```json
{
  "research_only": true,
  "formal_m2_request": false,
  "episode_counted": false
}
```

The materializer still loads the exact qualified Beagle asset and the
validated grounded base M2 request through the existing loader/compiler. That
base request is calibration/package authority only; its validation pass does
not promote the generated actor tracks to formal M2 status.

## Track semantics

The r4 `source*_positions_m` arrays are retained under
`route_source_center_plan`. They are PathFinder/.house source-center route
points, never emitter positions.

Each track separately contains:

- actor/source/semantic/emitter identity;
- exact qualified asset manifest and base M2 request references;
- 75 frame states with `frame_index`, `pts_ticks`,
  `action_id`, `action_time_ticks`, baked joint rotations and contact
  states;
- planned `world_from_actor`;
- derived `world_from_skin_root`;
- tangent/static-heading provenance; and
- existing M2 `pose_hash` and `applied_state_hash` helper values.

Moving tracks use real baked `walk` samples. Static tracks use real baked
`idle` samples. Contact frames come from the hash-checked asset
`contact_phases` role and are matched to the baked action sample ticks.
All tracks use 75 frames at 15 Hz with `pts_ticks=i*3200`.

The route-point-to-actor-root translation is deliberately zero-offset and is
written numerically as `[0.0, 0.0, 0.0]`, but is recorded only as an
exact-Beagle, revision-scoped research calibration reference. It is not a support-plane, paw-contact, emitter, collision or
native-execution claim. Moving orientation uses XZ finite differences and the
qualified Beagle `forward_axis=-Z`, `up_axis=+Y` convention. Static
orientation comes from the first rotation in the grounded base M2 request,
never from the camera. Different asset manifests or base calibrations are
rejected.

Emitter readback, paw/support contact, articulated collision, OBJECT_ID,
native execution and RLR are all explicit `null/pending`.

## Fresh region000 CPU pilot

Command:

```text
PYTHONPATH=src python tools/m5/materialize_mp3d_region_actor_tracks.py \
  --route-bank /data/jzy/tmp/mp3d_region_source_route_bank_2b0d4f9_r4/region_source_route_bank.json \
  --region-id 17DRP5sb8fy:region:000 \
  --asset-manifest /data/avengine_external/datasets/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json \
  --base-m2-request /data/avengine_external/review/current_mp3d_two_beagle_route_lateral_seed22_grounded/primary_m2_request.json \
  --output-root /data/jzy/tmp/mp3d_region_actor_tracks_2b0d4f9_r4
```

The earlier r3 output is retained as history but is not the final artifact.

Fresh final output:

`/data/jzy/tmp/mp3d_region_actor_tracks_2b0d4f9_r4`

It contains 16 cases and 32 actor tracks:

- four route families for region000;
- `static_static`;
- `s1_moving_s2_static`;
- `s1_static_s2_moving`; and
- `both_moving`.

Independent verifier:

`/data/jzy/tmp/mp3d_region_actor_tracks_2b0d4f9_r4_verification.log`

The verifier reloaded the base M2 package/request and compiler, then checked all
generated tracks for:

- exact 75-frame/PTS clocks;
- all-idle static and all-walk moving action selection;
- contact-phase and joint/hash consistency;
- route-center versus planned-root separation;
- tangent orientation and static base heading;
- actor/source identity and endpoint exchange continuity; and
- camera metadata not affecting root generation.

Status is `pass`. The verifier also confirmed the final r4 authority metadata,
resolved every selected camera M1 reference, rejected no-span moving/static
mismatches, and checked the numeric route-point-to-actor-root offset
`[0.0, 0.0, 0.0]`. No Simulator, native actor execution, GPU, OBJECT_ID,
collision, or RLR job ran.

## Next native runner boundary

The next owned slice is a separate two-track native runner. It should load both
track files into one Simulator, apply each actor's own root/joint state, and
persist actual root/joint/emitter readback. It must then perform native
articulated collision/contact checks, OBJECT_ID review, and only afterward pass
the actual emitter trajectories to the existing RLR adapter.

The current materializer does not claim that native runner or audio bridge.
The next runner must consume actual emitter readback, not the r4 source-center
arrays.
