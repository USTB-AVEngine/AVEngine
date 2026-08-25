# MP3D region visual-to-RLR audio checkpoint — 2026-08-25

## Scope and code

Commit `d888738` on `codex/mp3d-region-native-visual-runner` extends the
research-only two-actor native visual receipt and the existing current-MP3D
dynamic-audio adapter. It does not change route generation, camera legality,
RLR propagation, the AudioProgram schedule, or formal admission.

The old current route remains supported with visual axes `source0/source1`.
Region actor tracks use `source1/source2`; the adapter adds a second explicit
bridge:

- `source1/actor0 -> beagle_0_muzzle`;
- `source2/actor1 -> beagle_1_muzzle`.

No endpoint is inferred from a numeric suffix, sorted order, or current
position. Actor/source identity remains attached to the named axis when paths
cross or exchange.

## Camera/source namespace separation

The selected M1 camera request still contains its historical request sources
`source0/source1`. Those poses do not generate or filter the region actor
routes. The new visual receipt records both namespaces and proves
`camera_participation_in_route_generation=false`:

- M1 request source namespace: `source0/source1`;
- visual actor-route namespace: `source1/source2`;
- audio endpoint namespace: `beagle_0_muzzle/beagle_1_muzzle`.

The audio adapter permits the first two to differ only when the visual receipt
explicitly declares a validated case-manifest or post-hoc camera binding and
camera-route decoupling. Legacy receipts without that fact keep the previous
exact M1/visual source-order check.

## Audio-ready visual receipt

The native visual runner now retains:

- full capture-time M1 identity by value: request/room, complete camera rig and
  calibration, listener transform, M1 source order and QA views;
- selected room and native capture request/room IDs;
- one per-axis actor/source/semantic/runtime-object/emitter-link binding with
  audio endpoint still explicitly pending;
- `source_positions_m` in every frame as an alias of native mouth-link
  readback, plus exact frame/PTS and actor/source pose alignment;
- a direct `source_positions_m` artifact pointing to the existing
  `[75,2,3]` emitter-position array.

The dynamic-audio adapter cross-checks the JSON positions exactly against the
NPY before any RLR call. The capture-time M1 content, not the historical path,
is listener/view authority.

Owned and adjacent CPU coverage passed `75` tests. Ruff check and format check
pass for all four touched code/test files.

## Fresh native visual A/B

The audio-ready visible capture keeps region000 route family 03, both-moving,
and post-hoc camera `mp3d_camera_placement_04975`:

- `/data/jzy/tmp/mp3d_region_multi_actor_native_d888738_region000_f03_cam04975_both_r1`;
- `/data/jzy/tmp/mp3d_region_multi_actor_native_d888738_region000_f03_cam04975_both_r2`.

Both fresh GPU2 runs contain 75 frames and have identical arrays,
`frame_records.json`, and visual receipts. The actor routes remain unchanged,
actor-pair collision-free, and research-only. OBJECT_ID/semantic authority
limits from the previous checkpoint remain unchanged; this receipt upgrade does
not establish formal off-screen or occlusion truth.

## Fresh RLR A/B

The new visual receipt was rendered twice through the installed external RLR
runtime, with stride 3 and 25 acoustic keyframes:

- `/data/jzy/tmp/mp3d_region_dynamic_audio_d888738_region000_f03_cam04975_both_r1`;
- `/data/jzy/tmp/mp3d_region_dynamic_audio_d888738_region000_f03_cam04975_both_r2`.

Each receipt reports:

- `status=pass`, `research_only=true`, `episode_counted=false`, formal count 0;
- 16 kHz, 80,000-sample binaural mixture and two endpoint stems;
- six sequential AudioProgram bark events;
- 25 native listener/source registration receipts;
- `prior_listener_rir_reused=false`;
- the new visual receipt, frame-record and emitter-position hashes as exact
  inputs;
- verified unique visual-axis to registered endpoint/entity-anchor identity.

The two runs have identical dry buses, binaural stems, mixture, trajectory
hash, and every keyframe/source IR hash. Mixture SHA-256 is
`f4510f9355206cf7db69b414fcab044a15f312dc19667c89fdb32a414fce6cf6`.

## Independent waveform verification

The fresh verifier is:

`/data/jzy/tmp/mp3d_region_dynamic_audio_d888738_region000_f03_cam04975_verification_r1.json`

It reports `status=pass` and independently confirms:

- all five output WAV hashes match both receipts and match across A/B;
- dry buses are `[80000,1]`, stems/mixture are `[80000,2]`, all finite;
- every one of the six scheduled source windows has non-zero dry and binaural
  waveform RMS;
- the mixture equals the sum of the two stems with maximum absolute error
  `6.4939911e-09`;
- mixture peak is `0.0876966`, below clipping;
- listener/source registration and all 25×2 IR hash records are complete;
- the old retained audio's visual frame-record hash
  `26d753...` differs from the new visual hash `a2d969...`, so no old
  visual/audio receipt was reused.

Per-event ILD is retained as a raw descriptor in the verifier. It is not used
as a legality or difficulty gate. The independent check proves energy inside
scheduled windows but does not claim sample-exact audible onset; reverb tails
and onset authority remain separate downstream analyses.

## Remaining boundary

- This is one region000 mini-pilot, not all ten MP3D regions.
- It is binaural RLR audio; a new-visual native FOA render is still pending.
- Formal modal visibility/off-screen authority is still pending the active-RGB
  same-depth OBJECT_ID fix.
- No QA row, exact48 item, trainer dataset example, or production Episode is
  admitted by this checkpoint.
