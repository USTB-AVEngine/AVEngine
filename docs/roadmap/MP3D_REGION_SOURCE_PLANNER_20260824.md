# MP3D region-specific source/path planner (2026-08-24)

## Scope

This slice starts from clean integration 2b0d4f9 on the independent server
worktree branch codex/mp3d-region-source-planner.

It consumes the existing ten-region/four-view camera bank, but uses camera
records only as metadata bindings. Source route generation does not read camera
position, yaw, RGB/semantic pixels, visibility, composition, or difficulty.

The route authority is a standalone Habitat PathFinder loaded from the explicit
MP3D .navmesh, intersected with the corresponding .house floor polygons. There
is no polygon-only or straight-line fallback: a PathFinder failure stops the
build.

## Implementation

Owned files:

- src/avengine/m5/mp3d_region_source_planner.py
- tools/m5/build_mp3d_region_source_route_banks.py
- tests/unit/test_mp3d_region_source_planner.py

The planner validates all 40 existing camera bindings against the sidecar,
then independently enumerates native PathFinder points inside each .house
region. Four geometrically distinct route families are generated per region
and zipped to the four camera selections by selection order only.

Each family materializes:

- static_static;
- s1_moving_s2_static;
- s1_static_s2_moving;
- both_moving.

Moving routes use the PathFinder returned waypoint polyline. A real two-point
PathFinder result is recorded as straight; three-or-more waypoints are recorded
as polyline. This is a label of the actual PathFinder result, not a fallback.

Every case has 75 frames at 15 Hz. Source IDs remain source1/source2.
Endpoint exchange, pair distance, crossing and proximity-to-0.60 m are retained
by route_descriptors; none is a candidate gate. Static coincident endpoints
are not mislabelled as endpoint exchange: a strict exchange descriptor requires
nonzero initial and final pair distances plus zero swapped-endpoint error.

Native articulated collision, visibility, occlusion, audio/RIR and formal
Episode status remain pending.

## Focused CPU tests

The owned focused suite passes:

    19 passed

It covers ten-region/40-binding/160-case counts, 75-frame materialization,
camera-pose independence, exchange and sub-60 cm descriptor retention,
PathFinder-failure no-fallback behavior, and pending native visibility/collision
fields.

## Fresh real CPU mini-pilot

The final fresh/no-clobber output is:

/data/jzy/tmp/mp3d_region_source_route_bank_2b0d4f9_r3

The command used the explicit runtime prefix, Magnum site, RLR adapter
dependency root, MP3D .navmesh, and .house; it did not start a Simulator,
renderer, GPU, UE or RLR render context.

Independent artifact checks report:

- 10 regions;
- 40 camera bindings;
- 40 distinct route families;
- 160 motion cases;
- 75 frames per case;
- 10 real both_moving endpoint-exchange cases;
- candidate_filtering=none;
- camera_route_generation_decoupled=true;
- native capture/audio/RIR/visibility/collision all pending.

A representative family retains 15 frames below the 0.60 m proximity
descriptor in each moving case; those cases are still present. The artifact
does not use this descriptor for retention.

Earlier fresh attempts failed closed before publication:

- region 000 exposed that a two-waypoint PathFinder path must be labelled
  straight, not forced into the shared polyline descriptor;
- region 002 exposed the same issue for the reversed exchange path.

No output was weakened or reused; the final run was published to a new sibling
directory.

All outputs remain research_only=true, episode_counted=false,
formal_release_status=not_run.
