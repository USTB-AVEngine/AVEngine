# MP3D region000 four-motion RLR checkpoint — 2026-08-25

## Scope

This checkpoint completes the visual+audio layers for one four-point
mini-pilot viewpoint. QA generation and gates remain pending, so this is not a
complete admitted mini-pilot.

Viewpoint and route binding:

- region: `17DRP5sb8fy:region:000`;
- camera slot: `mp3d_camera_placement_01237`;
- post-hoc yaw request: `mp3d_camera_placement_01246` at 270 degrees;
- route family: `17DRP5sb8fy_region_000_route_family_03`;
- motion cases: static/static, source1-moving, source2-moving and both-moving.

## Native visual inputs

The four native visual points are under:

`/data/jzy/tmp/mp3d_region_native160_2821229_r1/region_000`

They are point indices 000–003. Every receipt has 75 frames, exact M1/source
identity, no active actor-pair collision and research-only status. They are
part of the independently verified native160 batch.

## Fresh RLR outputs

All four audio outputs are under:

`/data/jzy/tmp/mp3d_region000_view01246_audio4_2821229_r1`

Each point independently rendered 25 stride-3 RIR keyframes for both registered
Beagle endpoints, six AudioProgram events, two binaural stems and one
80,000-sample/16 kHz mixture. Every receipt records
`prior_listener_rir_reused=false` and binds the exact corresponding visual
receipt, frame records and emitter-position array.

The independent verifier is:

`/data/jzy/tmp/mp3d_region000_view01246_audio4_2821229_r1_verification.json`

It reports `status=pass` and confirms:

- all four planned motion cases are present and pass;
- all four trajectory hashes are distinct;
- all four mixture hashes are distinct;
- all waveforms are finite, below clipping and equal the sum of their two stems
  within `3.73e-09` maximum absolute error;
- visual source axes remain `source1/source2` and audio endpoints remain
  `beagle_0_muzzle/beagle_1_muzzle` under the same listener view.

Mixture peaks are approximately 0.05745, 0.04342, 0.05929 and 0.06125 for the
four motion cases. The distinct values are retained as raw audio evidence, not
quality or difficulty gates.

## Remaining boundary

- Audio schedules still use the fixed current-MP3D six-event program; formal
  QA requires randomized onsets/intervals.
- No T5, temporal, refusal, off-screen or answer-balance questions have been
  generated for these points.
- B/C gates and an owner question review remain pending.
- This checkpoint proves native visual+RLR execution, not a complete mini-pilot
  or formal Episode. Formal count remains zero.
