# MP3D Region Multi-Actor Visual CPU Boundary

## Scope

This slice is the CPU-only precondition for the future native multi-actor
visual runner. It consumes the M5 research-only `case_manifest.json` emitted
by the r4 actor-track materializer and validates two independent actor tracks.
It does not create a Habitat Simulator, render RGB/depth/semantic/OBJECT_ID,
read an emitter, query Bullet, or invoke RLR.

The implementation is in
`src/avengine/m5/current_visual_multi_actor.py`. It intentionally does not
modify the formal M2 contract, the camera bank, the route planner, the
materializer, or the legacy single-M2 current-visual entrypoint.

The parent-reviewed implementation commit is `d4f6274`.

## Accepted input boundary

`load_multi_actor_case` requires:

- `research_only=true`, `formal_m2_request=false`, and
  `episode_counted=false` with the exact research receipt on both case and
  tracks;
- non-empty region, route-family, and one of the four declared motion cases;
- a resolved regular M1 request file and a route manifest that explicitly
  records `camera_participation_in_root_generation=false`;
- exactly `actor0/source1` and `actor1/source2`, unique semantic IDs, and two
  track paths that are relative and symlink-free;
- exactly 75 state/frame pairs at `frame_index=i`, `pts_ticks=i*3200`, with
  matching action/sample axes and native outputs still null/pending;
- each actor's own exact asset manifest and base M2 request. The existing M2
  loader, bundle loader, frame compiler, baked action samples, contact order,
  pose hash, and applied-state hash are reused for validation.

The case manifest itself is rejected when it is a symlink. Actor paths cannot
escape through a symlink or parent traversal. Actor/source axes must remain
`actor0/source1` and `actor1/source2`; emitter-link names are validated per
actor and are not required to match across different assets.

For each frame, route source-center, planned `world_from_actor`, and derived
`world_from_skin_root` must agree. `route_point_to_actor_root_offset_m` must be
the explicit numeric `[0, 0, 0]` research calibration reference. Static cases
must remain spatially static; walk cases must have non-zero route span. No
camera position, visibility, modal pixel count, or OBJECT_ID value is used to
accept or reject an apply plan.

## Immutable apply plan

`build_apply_plan` returns frozen dataclasses containing two independent
75-frame tracks. Each frame retains its own root/skin-root matrices, route
source-center, action sample, and joint ID/rotation tuples. Different tracks
may have different asset IDs and joint counts/order; no
`CURRENT_ACTOR_OFFSETS_M` is imported or applied. The only shared clock rule is
frame/PTS equality between the two tracks. The plan ID is
`<route_family_id>::<motion_case>`, so the four motion plans under one route
family remain distinct. Actor emitter-link names remain per-track values and
do not need to match across assets.

## Collision classifier boundary

`classify_collision_contacts` is a pure classifier for serialized or mock
`ContactPointData`. It validates object/link IDs, Habitat's
`position_on_a_in_ws`/`position_on_b_in_ws` world-space vectors, normal,
`is_active`, and finite distance (and retains `normal_force` when supplied),
then returns:

- `actor_actor_contacts`: any frame containing one makes
  `actor_pair_collision_free=false`;
- `actor_stage_contacts`: raw records with
  `stage_contact_recorded_unclassified=true` when exactly one actor touches
  the explicit `stage_object_id` (default `0`);
- `actor_other_contacts`: actor contacts with non-stage, non-actor objects;
- `unrelated_contacts`.

It does not infer ground support, wall collision, legality, visibility, or
distance. A future native runner must still hard-fail if the installed runtime
lacks `perform_discrete_collision_detection` or
`get_physics_contact_points`; this CPU slice does not claim those methods were
executed.

## Verification

On the server's real Habitat dependency environment:

```text
PYTHONPATH=src python -m pytest -q tests/unit/test_current_visual_multi_actor.py
12 passed in 13.19s
```

The retained r4 region000 input was loaded and planned fresh/no-clobber for all
16 cases and 32 actor tracks. All cases produced two 75-frame plans; this is
CPU validation only and creates no native capture artifact.

The parent-owned plus M2/materializer/current-visual adjacent run is:

`/data/jzy/tmp/current_visual_multi_actor_parent_review_r2.log`

It records `64 passed`. Ruff check and format, byte compilation, and
`git diff --check` pass for the owned files.

## Explicitly pending

The following remain for a separately owned native-runner slice: one Simulator
with two independently instantiated bundles, per-frame root/joint apply and
readback, common M1 sensor rendering including OBJECT_ID, Bullet discrete
collision readback, repeat-capture determinism A/B, actual emitter links,
support-contact authority, offline first-person/OBJECT_ID/top-down review, and
the GPU0 occupancy guard. No native visual, collision, or RLR evidence should
be inferred from this commit.
