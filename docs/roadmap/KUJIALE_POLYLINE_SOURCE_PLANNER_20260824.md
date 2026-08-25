# Kujiale target-agnostic polyline source planner (2026-08-24)

## Scope and authority

This is a CPU planning/request slice on branch `codex/kujiale-polyline`. It
does not use Studio, room names, camera semantics, UE, SPEAR, GPU capture,
native pixels, RLR, M3, RIRs or dataset admission. The fresh output remains
`research_only=true`, `episode_counted=false`, formal dataset count zero and
explicitly leaves UE/SPEAR/native-pixel certification pending.

The source-route search consumes the human and dog `FeasibleRegionIndex`
rasters built from the retained room polygon, descendant-mesh ground-blocker
footprints, emitter heights and configured source-center margin. Every segment
in the current bank passes conservative raster supercover in both actor masks,
including both side-adjacent cells at an exact grid-corner crossing. This is
still a 2D source-center CPU claim, not full articulated-body collision or
native UE execution evidence.

No frozen contract, baseline, release/admission gate or permanent difficulty
gate was introduced.

## Implementation

The branch contains separate implementation, per-view-diversity and review-
correction commits. Together they add:

- target-agnostic simple multi-segment route search with no repeated vertex or
  edge, no non-adjacent proper crossing, collinear overlap or endpoint-to-
  interior touch, bounded DFS branching and geometric direction diversity;
- explicit `--route-geometry straight|polyline`; a polyline failure writes a
  fresh `FAILED.json` and never falls back to a short/repeated straight route,
  reduced speed or reduced natural path length;
- shared exact-arc-length 75-frame materialization through the existing
  `resample_polyline_by_arc_length`, plus explicit segment membership,
  waypoint-boundary frame brackets and per-frame segment-tangent yaw;
- synchronized continuous closest approach solved on the union of both
  original piecewise-linear paths' normalized-time segment boundaries, never
  on a sampled chord across a fractional-frame corner;
- natural path length derived from the actual frame-0 to frame-74 movement
  span `(75-1)/15`, so replay speed itself remains in the actor profile;
- explicit hold, straight and polyline replay in residential episodes; UE is
  still bound to replay the authored Timeline and may not replan;
- four motion cases: static/static, human moving, dog moving and both moving;
  source/actor IDs remain continuous even when endpoints exchange; and
- per-frame planar/root pair distance, discrete and continuous closest time,
  endpoint exchange, spatial crossing, listener-relative bearings/right
  coordinates/order and transition frames as raw descriptors.

Each selected CPU view receives a different deterministic geometric corridor,
avoiding nominally different views that replay the same source path. This is a
normal output precondition: fewer corridors than requested views fails fresh
instead of modulo-reusing a route.

Camera selection is now fully separated from the human/dog source masks. The
builder consumes the complete current target-agnostic room camera catalog,
cross-checks its room/scene/map/stage binding, verifies every sampled floor
position has every declared height and 12 yaw variants, and ranks the complete
legal pool only afterward. It also cross-checks and renders the catalog's
sibling camera free-space raster. FOV/LOS/composition remain ranking clues and
never remove a catalog-legal placement.

The former 0.60 m separation, side margin, crossing/exchange and camera-FOV
conditions are recorded only as downstream descriptors. They do not remove a
geometry-valid camera, route or pair. The legacy current kitchen profile is
used only for current map/stage binding, emitter heights and center margin;
its repeated 0.5 m back-and-forth route is not copied into the new requests.

An earlier first-pass artifact is retained as historical evidence only. Its
32,640 value was labelled as if every pair had been checked compatible, and
its continuous closest value could use a sampled chord across a fractional
corner. Neither value is used as current evidence. Current output instead
reports an arithmetic `directed_pair_combination_count` and the smaller
`described_pair_count`; only the described records carry computed clues.

## Fresh current-kitchen CPU evidence

The exact fresh artifact, build log and independent-verifier log are retained
outside Git under `/data/jzy/tmp` and listed in the branch handoff. Inputs bind
the retained current full-home kitchen metadata, current full-home source and
visual profiles, and `examples/qa_v2/motion_profiles_v1.json`.

Observed results from the final clean branch code:

- 128 simple polyline corridors;
- exactly 4 segments per corridor in this selected pool;
- path length 3.0590169944-3.6035533906 m;
- actual replay speed 0.6200710124-0.7304500116 m/s over `(75-1)/15` seconds,
  inside the retained
  human/dog 0.60-0.78 m/s range;
- all 1,188 legal placements from the current target-agnostic camera catalog
  validated before selection;
- four CPU viewpoint requests at three camera-height variants, four different
  catalog placement IDs and catalog-provided 30-degree yaw tokens;
- four distinct source corridors across the four views;
- four motion cases per view, 16 cases total;
- 14/16 cases contain at least one planned frame below the old 0.60 m
  reference and are retained; and
- all four both-moving cases exchange endpoints exactly without changing
  source or actor identity.

Each view has 65,536 arithmetic directed route-pair combinations. Exactly 128
records per view are described; the remaining combinations carry no implied
compatibility or checked-distance claim.

Build wall time was 30.23 s with maximum RSS 115,572 KiB. There was no straight
fallback and no `straight_corridors.json` in the output.

The independent retained-data verifier rebuilt the kitchen obstacle map and
both actor feasible masks, checked every polyline edge against conservative
supercover and every non-adjacent segment pair against inclusive intersection,
recomputed route lengths and `(75-1)/15` speeds, joined all selected camera
poses exactly back to the 1,188-placement catalog, independently recomputed
piecewise continuous closest approach for all 512 described pair records and
all 16 motion cases, and checked four distinct selected routes, below-0.60 m
retention and exact exchange. It reports `status=pass`.

## Visual review boundary

I inspected the fresh `viewpoint_topdown.png`. It now draws the camera
catalog's free-space raster, not an actor source mask. V1-V4 centers all lie on
green catalog-feasible pixels and use distinct catalog positions/heights/yaws;
V3 lies on a narrow green strip and V4 on a separated green island. The blue
overlay visibly contains multi-turn source routes rather than the old 0.5 m
oscillation. V3/V4 have zero preferred visible route probes and are deliberately
retained because visibility/composition is a difficulty clue, not a legality
filter.

This image is only a CPU Topdown. No first-person UE frame was produced or
reviewed, and none of the four views is claimed UE/SPEAR-certified or
mini-pilot-ready.

## Tests

Fresh final-code verification recorded 80 focused/adjacent tests passing,
covering L/multi-turn routes, conservative segment checks, exact 75-frame
equal-arc sampling, fractional-corner continuous closest, actual replay speed,
inclusive non-adjacent overlap/touch rejection, explicit hold, all four motion
cases, endpoint exchange, retained sub-0.60 m cases, honest pair counts,
complete camera-catalog crossing/binding, actor-mask-independent camera pools,
old straight compatibility, unknown room labels, fresh/no-clobber and failed-
polyline no-fallback behavior. Ruff, `py_compile` and `git diff --check` also
pass. Exact retained artifact and log paths are listed in the branch handoff.

## Remaining production work

The four current kitchen views are CPU requests only. The native package and
registered human/dog/cat endpoints must exist before actor types are counted.
Each selected view still needs fresh UE/SPEAR replay, camera/listener and actor
readback, native first-person pixels and masks, exposure/composition/occlusion
review, independent RIR identity and a four-point audiovisual mini-pilot.
Other current full-home room instances need the same route/request and native
certification sequence; this slice does not claim multi-room completion.
