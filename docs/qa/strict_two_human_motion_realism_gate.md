# Strict two-human motion-realism release gate v1

The motion-realism gate is fail-closed and CPU-evaluable. It is independent of
the pixel-visibility, acoustic, and ground-contact gates. Passing one of those
does not imply passing this gate.

## Required producer schema

Each moving slot must add a
`suite_actor_root_application.motion_realism_profiles.<slot>` object using
`avengine_strict_two_human_motion_realism_profile_v1` with:

- a release-qualified native-rate active interval;
- native-window and output-active root-speed facts;
- a live Walking asset play-length/play-rate readback and exact phase/tick path;
- a reviewed canonical Walking contact-phase authority; and
- live per-active-frame world-space readbacks for `Bip01 L Foot`,
  `Bip01 L Toe0`, `Bip01 R Foot`, and `Bip01 R Toe0`, each bound to the accepted
  floor trace and strict ground-contact result.

The output active interval may be placed where composition requires, but its
interval count must equal the retained native source interval count. Outside
the interval, the actor is Idle and its root holds the nearest boundary root.
For `both_move`, the two independently native-rate intervals must overlap enough
to make the mechanism true; the CPU geometry/visibility preflight must select
that placement. A short path may not be made full75 merely to force every root
to be unique.

## Threshold authority

- Native average speed is `horizontal_path_length / ((native_end-native_start)/15)`.
- Native phase cadence is `native_phase_advance / ((native_end-native_start)/15)`.
- Canonical clip cadence is `1 / live_Walking_play_length` at live play rate 1.
- Exact timeline authority is 15 Hz, 48,000 ticks/second, and 51,200
  animation ticks/cycle for these assets.
- Speed and cadence relative-error caps are 5% and 2%, respectively.
- Maximum phase-to-contact mismatch is half one output-frame phase advance.
- Maximum planted-foot horizontal slip is 0.02 m per 15 Hz frame.
- Floor clearance, penetration, normal, and floor identity are delegated to the
  accepted strict ground-contact release profile; bounds are never contact
  evidence.

## Legacy decision

The current target_moves_v2, distractor_moves_v2, and both_move_v1 materialized
canaries have no motion-realism profile or per-active-frame contact authority.
They also apply Walking over all 75 frames while preserving only the phase
advance of short 11–27-sample native windows. The v1 gate therefore rejects
them as motion-release inputs while preserving their narrower pipeline evidence.

## Recommended producer revision

Update the candidate builders to emit native source range, output active range,
and asset contact authority explicitly. Update the materializer to expand only
the active interval at native rate and emit Idle/held roots outside it. Capture
must persist live animation asset, play rate/position, four contact bones, and
floor traces for every active frame. The dynamic finalizer should invoke this
receipt before treating a canary as release-qualified.
