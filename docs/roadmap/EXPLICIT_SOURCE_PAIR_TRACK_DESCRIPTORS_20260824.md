# Explicit source-pair track descriptors — 2026-08-24

Status: shared CPU descriptor interface complete; native projection and
collision evidence remain separate and pending.

`avengine.qa.motion_feasibility.describe_source_pair_tracks` accepts two exact
caller-supplied `[frame, 3]` position tracks. It does not synthesize missing
frames and does not interpolate start/end endpoints. The caller must also
provide source IDs, the listener position and explicit forward/right/up basis
for every frame (or one static vector to broadcast), a coordinate-space label,
track authority, and an explicit horizontal/vertical storage-axis mapping.

Examples:

- Apartment UE XYZ: horizontal axes `(0, 1)`, vertical axis `2`.
- Habitat XYZ with Y-up: horizontal axes `(0, 2)`, vertical axis `1`.

The interface preserves, per frame:

- source identity and exact XYZ position;
- listener position and orientation basis;
- axes-0/1, declared-horizontal and XYZ pair distances;
- listener-local azimuth, elevation, side and side margin;
- optional frame index and integer presentation ticks;
- projection, in-frame, occlusion and visible-pixel fields as explicit nulls.

If a source lies within the declared `side_center_tolerance` of the listener,
local azimuth/elevation are treated as undefined and emitted as null with
`listener_angles_defined=false`; they are never replaced by synthetic zero
angles. The record also carries all four decision tolerances and is checked
with strict JSON semantics (`NaN`/`Infinity` forbidden). Stable `hypot`-based
distances prevent large-but-finite coordinates from overflowing silently.

Its summaries retain closest discrete frames (including an explicit earliest
tie policy and all tied frames), distance series, observed-frame horizontal
caller-supplied source-position coincidences, nontrivial endpoint exchange and
changes in the two sources'
pairwise listener-local-right order. Tied observations are skipped between
successive decisive order measurements, so a sampled center coincidence does
not hide an observed order exchange. A single coincident frame is never called
an endpoint exchange.
Between-frame crossing is deliberately `not_inferred`; native mesh
intersection is also not inferred from source roots. Legality remains
`not_evaluated_by_track_descriptor`, while distance and angular values are
difficulty descriptors with candidate filtering disabled.

The older endpoint-based `describe_motion_geometry` remains available for
viewpoint planning, but now marks every generated frame series as
`synthesized_linear_between_declared_endpoints_not_runtime_track` and sets
`runtime_track_claim=false`.

Verification on the server:

- explicit-track unit tests cover Apartment endpoint exchange/coincidence at an
  observed frame, Habitat Y-up axis mapping, listener-local angles, pending
  pixel fields, 75-frame non-linear preservation, strict numeric/JSON bounds,
  undefined co-location angles and ambiguous-input rejection;
- 11 focused tests passed;
- the motion, Apartment viewpoint and QA design group passed 52 tests;
- Ruff and formatting checks passed.
