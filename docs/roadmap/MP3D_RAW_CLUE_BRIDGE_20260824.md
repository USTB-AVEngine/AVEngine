# MP3D static/dynamic raw-clue bridge (2026-08-24)

## Scope

This CPU/mock slice joins an already-selected MP3D camera input to its static
native probe and optional dynamic Habitat capture. It does not enumerate,
rank, select, delete, or promote candidates. Both the twelve-item full-catalog
probe subset and the forty-item, ten-region viewpoint-bank layout are accepted
as input adapters. `placement_id` becomes `viewpoint_id`; each selection order,
ranking record, M1 path, candidate record, and parent region record is copied
without reinterpretation. The output candidate count must equal the input
count.

Missing dynamic captures remain pending per candidate. A supplied dynamic
capture that names no input M1 request is inconsistent input and stops the
publication; it is never silently ignored. Difficulty descriptors, distances,
side margins, crossings, exchange, native visible-pixel count, and composition
never affect retention.

## Current dynamic authority

A newly generated dynamic receipt is consumed only when its complete
post-render evidence is present and cross-consistent:

- planned actor-world and planned applied skin-root matrices `[75,2,4,4]`;
- actual post-render skin-root matrices `[75,2,4,4]`;
- actual agent/camera `[75,7]`, sensor `[75,sensor,7]`, and logical-listener
  `[75,7]` poses;
- actual emitter-link positions `[75,2,3]`;
- native semantic observations `[75,H,W]`, semantic IDs, per-axis visible
  pixel counts, and identity-axis records; and
- exact frame index, 15 fps PTS, action ID, and action sample index records.

The bridge also consumes the receipt's capture-time `m1_capture_identity` and
requires its request ID, room, rig/calibration, listener, source order, and QA
views to equal the joined M1 value record. Checking only the declared M1 path
would allow a later rewrite at the same path to alter the camera used for DoA
and projection. This is ordinary value/provenance validation against that
concrete failure; no hash, frozen contract, baseline, or dataset gate was
added.

Older dynamic captures remain readable through the legacy route. Their actual
emitter positions and aggregate semantic pixel counts are retained, but actual
post-render camera/sensor/listener, planned and actual skin roots, projection,
semantic-ID bbox, and current pose authority remain explicit `pending`/`null`.
They are not upgraded from the planned M1 camera. Legacy speed uses real
frame-record PTS only when the receipt declares and passes a 75-frame artifact;
otherwise speed and per-frame PTS stay pending. No fixed 15 fps grid is
silently presented as frame-record authority.

## Per-frame raw clues

For current receipts every frame records:

- planned camera/listener and actual post-render agent, all sensors, and the
  logical listener derived from actual agent readback;
- planned actor world, planned applied skin root, actual skin-root readback,
  actual emitter position, and an explicit pending planned-emitter field;
- actual listener-local azimuth, elevation, distance, right coordinate, side,
  and side margin;
- analytic pinhole normalized X/Y, pixel X/Y, depth, horizontal/vertical and
  combined signed frustum margins from the actual post-render semantic-sensor
  pose plus capture-time M1 intrinsics; and
- native semantic-ID pixel presence/count, semantic-ID bbox, and bbox-to-border
  margin. These are raster facts, not complete actor modal visibility.

Continuous pixel coordinates use zero-based pixel centers: X spans
`[0,width-1]`, Y spans `[0,height-1]`, positive normalized X points right, and
positive normalized Y points up while pixel Y increases down.

`describe_source_pair_tracks` supplies horizontal/XYZ distances, closest
frames, listener-right order, observed frame crossings, and endpoint exchange
while preserving actor/source IDs. Between-frame crossing is not interpolated.
A planned continuous polyline remains pending unless a future receipt carries
an explicit same-route authority.

Semantic-ID absence is ambiguous: zero pixels does not prove actor absence,
occlusion, or off-screen. A positive count proves only that the named semantic
ID appears in the raster. Actor modal visibility, occlusion, and off-screen
therefore stay `null`/`pending` for both zero and positive counts until a
separate authority certifies semantic-ID coverage completeness. Planned frame
index, PTS and action ID are compared between M2 and frame records. M2
`action_time_ticks` is retained and compared when the frame record also carries
it. `action_sample_index` is explicitly
`retained_not_independently_rederived`; the bridge has no baked clip sample grid
with which to rederive it. Actual skin-root and emitter displacements/speeds are
numeric raw series without a moving/still gate. Audio program, native audio/RIR
receipt, P1 amodal evidence, and event timing remain present but pending when
not supplied.

## P2 semantic-raster evidence correction

The first selected12 native run exposed a concrete overclaim in the original
field names. In
`/data/jzy/tmp/mp3d_selected12_dynamic_60b1ede_r1/mp3d_camera_placement_06598`,
owner visual review found both dogs clearly present in frame 0 while the native
semantic raster contained zero pixels for both registered actor IDs 210 and
211. A read-only numeric audit reproduced `210=0`, `211=0`, while background IDs
138 and 6 occupied 11,147 and 14,568 pixels. Across 75 frames, the two registered
IDs had positive counts in only 55 and 2 frames respectively.

Consequently the bridge no longer calls these arrays actor-visible pixels or
modal truth. It records only `semantic_id_pixel_presence`, count, bbox, and
border margin, with
`semantic_id_coverage_completeness_not_certified`. The historical receipt key
remains readable, but it cannot upgrade actor modal visibility. No candidate is
removed because of this correction.

## CPU/mock verification

The focused and adjacent suite passed `91 tests` across:

- the raw-clue bridge and legacy viewpoint-track consumer;
- current Habitat visual receipt generation and offline pose review;
- explicit source-pair motion descriptors;
- the full-catalog probe subset; and
- per-region viewpoint banks.

The implementation commit is
`361beacc8bad1e42bba4685ec57ce711c19eeb61`. The exact post-commit test log is:

`/data/jzy/tmp/mp3d_raw_clue_bridge_361beac_unit_r1.log`

It records `91 passed`. Ruff check, Ruff format checking, Python byte
compilation, `git diff --check`, and the clean branch state are recorded at:

`/data/jzy/tmp/mp3d_raw_clue_bridge_361beac_quality_r1.log`

A separate read-only adapter audit loaded the retained selected12 and region
artifacts and confirmed exact `12 -> 12` and `40 -> 40` normalization with
unique ordered IDs; it published no derived artifact.

The tests include pair distance below `0.6 m`, observed crossing and endpoint
exchange, an explicit RGB clue with both semantic-ID counts equal to zero,
semantic presence remaining distinct from actor modal visibility, exact
continuous pixel-center center/edge/sign conventions, semantic-sensor pose
distinct from the agent pose, actual yaw changing DoA and projection, damaged
source axes, mutable-M1-path identity drift, legacy PTS present/missing and
damaged cases, M2 action-time comparison, unrederived action-sample labeling,
missing audio/P1 pending fields, probe-subset `2 in = 2 out`, and region-bank
flattening without record loss. Ruff check, Ruff formatting, byte compilation,
and `git diff --check` pass for the owned files.

No Habitat, GPU, audio, RLR, FFmpeg, or native selected12 job ran in this
slice. The first real selected12 dynamic join must wait for fresh captures
from the merged current code, then run with fresh/no-clobber output and receive
separate native frame review. All current outputs remain research-only,
`episode_counted=false`, and formal dataset count zero.
