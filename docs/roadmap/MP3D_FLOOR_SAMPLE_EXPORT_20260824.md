# MP3D Habitat-native camera-floor sample export (2026-08-24)

## Scope

This slice provides the native authority missing from the CPU-only viewpoint
prefilter in `459b0b2`. The exporter loads the validated MP3D room and M1
request through the explicitly selected installed Habitat runtime, opens a real
Simulator, loads the declared navmesh and asks its live PathFinder for topdown,
bounds, navigability, snap, island and obstacle-clearance readbacks.

The CLI is `tools/m5/export_mp3d_camera_floor_samples.py`. It requires explicit
`--runtime-prefix`, `--magnum-site`, `--rlr-sdk-root`, `--mp3d-root` and accepts
an optional `--pbr-root`. Output is a fresh absolute directory published through
the existing atomic no-replace directory operation.

## Coordinate and selection boundary

- Actor input is the retained current-MP3D `actor_skin_root_paths_m`, interpreted
  as Habitat world XYZ metres and revalidated on the live PathFinder.
- All route points must snap within the declared horizontal/vertical tolerances
  and resolve to one live island. A cross-island route is rejected.
- The topdown slice height is the median live-snapped actor-route floor height;
  routes whose floor-height span exceeds the explicit limit are rejected.
- Candidate XYZ is derived only from navigable pixels returned by
  `PathFinder.get_topdown_view` and finite `PathFinder.get_bounds`. It is not
  guessed from room or component bounding boxes.
- The scan is deterministically bounded. Every examined candidate is passed back
  through live `is_navigable`, `snap_point`, `get_island`, and
  `distance_to_closest_obstacle`; only same-island passing snaps enter
  `camera_floor_samples_m`.

Each machine-evidence row records requested/snapped XYZ, horizontal and vertical
snap error, navigability, island, clearance, coordinate space, authority,
verdict, failure reason and whether it was selected.

## Claims and downstream use

The output declares `native_habitat_floor_export=true`, `research_only=true` and
`episode_counted=false`. It is native navigation evidence, not native visual
evidence: RGB/depth/semantic pixels, first-person review, audio and the four-point
mini-pilot remain pending. The ordinary JSON file
`camera_floor_samples.json` can be passed directly to
`build_mp3d_viewpoint_bank.py --camera-floor-samples`.

No new hash, frozen schema, baseline or release gate is introduced. Runtime,
room, M1, MP3D and route input paths are recorded as ordinary provenance.

## Verification boundary

CPU mocks cover deterministic sorting and finite selection, actor-route island
agreement, cross-island candidate rejection, horizontal snap-error rejection,
invalid bounds, serialization cleanup and publication-race cleanup with the
winning destination preserved.

No native execution is claimed until the real CLI runs. Before such a run, the
operator must inspect active GPU/process state because Simulator configuration
retains the validated M1 visual sensors. If the selected device is occupied, the
native export remains `not_run` rather than competing with another job.
