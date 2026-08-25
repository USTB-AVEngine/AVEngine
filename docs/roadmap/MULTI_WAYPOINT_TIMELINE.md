# Apartment multi-waypoint visual timeline

## Checkpoint 2026-08-24: author and QA command wiring

The current Apartment visual timeline author now accepts an explicit ordered UE
centimetre waypoint list independently for the human and Beagle source. Each
list requires at least two finite points and is mutually exclusive with that
source's legacy start/end pair. Consecutive repeated points and zero-length
polylines fail before an output file is created.

Waypoint motion reuses `m6x.trajectory.resample_polyline_by_arc_length`. Frames
before `walk_start_frame` hold the first waypoint and remain idle; the remaining
frames traverse the complete polyline at equal arc-length targets. Per-frame UE
yaw follows the active original-polyline segment. An exact internal corner uses
the outgoing segment, and the final frame uses the final non-zero segment.

The old start/end call path remains on its previous interpolation and yaw code.
With two waypoints, the new author produces byte-identical serialized frame
records to the legacy path. A pure legacy call does not gain new metadata, so
existing timeline document bytes and semantics remain unchanged. Timelines that
use waypoints add a plain `route_authoring` record naming each source's method
and explicit waypoints; no schema, digest, frozen baseline, or release gate was
added.

The M5 CLI accepts repeated `--human-waypoint-ue-cm` and
`--beagle-waypoint-ue-cm` options and exposes `--walk-start-frame`. QA batch
design forwards `s1_waypoints_ue_cm` or `s2_waypoints_ue_cm` from a point spec
when present; otherwise it emits the unchanged start/end command.

Server-side ordinary tests cover legacy/two-waypoint equivalence, an L-shaped
equal-arc route, corner and final yaw, delayed walk start, invalid/mixed inputs,
the repeated CLI options and QA spec-to-command wiring. This is authoring-only:
`research_only=true`, `episode_counted=false`, spatial validation remains not
run, and no Studio, UE, native capture, audio, RLR, GPU job, or production
episode was executed by this checkpoint.
