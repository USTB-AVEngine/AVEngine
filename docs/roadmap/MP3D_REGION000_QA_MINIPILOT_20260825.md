# MP3D region000 four-point QA/T5 checkpoint — 2026-08-25

## Scope

This checkpoint completes one research mini-pilot core across native visual,
randomized RLR audio and grounded questions. Formal refusal/off-screen coverage
and dataset admission remain pending.

Implementation commits are:

- `5b1d74f feat(qa): design Habitat region mini-pilot`;
- `412348e fix(qa): verify randomized mini-pilot audio`.

The adapter consumes Habitat-native listener/emitter readback directly. It does
not translate the evidence into UE coordinates or reuse the Apartment
viewpoint/question adapter.

## Input viewpoint and four motion points

- region: `17DRP5sb8fy:region:000`;
- camera slot: `mp3d_camera_placement_01237`;
- actual post-hoc yaw request: `mp3d_camera_placement_01246`;
- route family: `17DRP5sb8fy_region_000_route_family_03`;
- motion axis: static/static, both single-mover directions and both-moving.

The visual points come from the independently verified native160 batch. Fixed
schedule RLR receipts come from:

`/data/jzy/tmp/mp3d_region000_view01246_audio4_2821229_r1`

## Randomized schedule design

Initial design artifact:

`/data/jzy/tmp/mp3d_region000_view01246_qa_design_5b1d74f_r1`

For each point, the existing validated AudioProgram materializer moves only the
first two event windows inside its feasible schedule set. Event identity,
durations, source slices, gain/fades and later windows remain unchanged.

The four schedules contain:

- first-event frames: 4, 4, 5 and 6;
- second-event frames: 14, 17, 18 and 19;
- first-to-second intervals: 11,066, 13,200 and 16,400 samples.

This provides multiple first/second onsets and intervals. The design status
correctly remained `pending_randomized_audio_render` until matching RLR
receipts existed.

## Fresh randomized RLR

The randomized RLR outputs are:

`/data/jzy/tmp/mp3d_region000_view01246_randomized_audio4_5b1d74f_r1`

All four points passed with fresh 25-keyframe RIR sequences and
`prior_listener_rir_reused=false`. A sequential failure-stop runner retained
one log and one progress record per point.

## Final QA artifact

Final immutable artifact:

`/data/jzy/tmp/mp3d_region000_view01246_qa_final_412348e_r1`

Independent verifier:

`/data/jzy/tmp/mp3d_region000_view01246_qa_final_412348e_r1_verification.json`

The verifier reports `status=pass` and checks every randomized AudioProgram
ID, visual capture identity, RLR receipt, T5 event frame and recomputed azimuth.

The final question set contains 12 rows:

- four `QV2-TA-MOTION`;
- four `QV2-T7-DURING`;
- four `QV2-T5-DOA`.

TA and T7 each have exact displayed-key balance A:B = 2:2. The T5 answers are
5.3, 5.4, 0.5 and -1.2 degrees, with the existing ±15° full-credit and ±30°
partial-credit bands. Gold authority is the native Habitat logical-listener
pose plus mouth-link emitter readback at the randomized event frame.

Gate B passes native visual, collision, fixed audio and randomized audio.
Gate C passes motion answer balance, onset/interval diversity and T5 native
readback.

## Deliberate omissions

T4 side questions are not generated: the first-event evidence is same-side and
near the view center, so emitting a binary side row would preserve a weak,
unbalanced shortcut. No new side threshold is introduced.

Refusal quota remains pending a larger batch with a truthful audio-insufficient
visual question. Formal off-screen remains pending same-renderer target-only
authority. Accordingly the status is `pass_research_minipilot_core`, not a
formal accepted Episode; the dataset denominator remains zero.

## Verification

The focused and adjacent QA suite reports `38 passed`; Ruff check/format pass
for the new tool and tests. The pre-finalizer integration code state
`9413494` reports `3416 passed, 118 skipped, 52 subtests passed`. A fresh
post-merge full run is required for the finalizer commit.
