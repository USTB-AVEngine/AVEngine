# Episode and Authoritative Timeline

## Integer time authority

`schemas/avengine_timeline_v2.schema.json` is the current baseline:

| Quantity | Value |
|---|---:|
| Time base | 48,000 ticks/s |
| Duration | 240,000 ticks / 5 s |
| Video | 15 fps / 75 frames |
| Ticks per frame | 3,200 |
| Audio | 16 kHz / 80,000 samples |
| Ticks per sample | 3 |

Frame audio boundaries are computed as rounded rational boundaries, not as a
fixed 1,067 samples per frame.

## Evaluation order

```text
warm runtime while the official action clock is frozen
reset authoritative timeline to tick 0
for each frame:
  set root and exact baked joint pose at frame PTS
  evaluate the canonical state once
  compute pose and semantic-anchor hashes
  capture co-located RGB/depth/semantic sensors for formal view0 without advancing the state
  update named RLR source/listener state from the same timeline
assemble exact audio sample intervals and verify outputs
```

## Single-view multimodal MVP

The current formal observation uses logical rig `camera_rig_0` and exactly one
dataset `view_id`, `view0`. RGB, depth and semantic are separate Habitat sensor
specifications on that rig, but share its position, orientation, resolution
and projection calibration. They are modalities of one viewpoint. The single
`listener0` shares the rig world transform, while sources have stable names and
independent transforms.

Timeline v2 deliberately retains a `view_ids` array for future profiles. The
M1, M2 and M5 canaries and initial M6 MVP apply a stricter semantic invariant:
`video.view_ids == ["view0"]`, with exactly one `view_pose_hashes.view0` entry
per frame. A top-down or other debug camera is QA-only and must not appear in
the timeline, observation manifest or benchmark input.

## Schema versus semantic validation

JSON Schema validates structure, ranges and constants. The timeline validator
must additionally enforce all of the following:

- `frame_index` equals its zero-based array position and covers `0..74` once;
- `pts_ticks == frame_index * 3200` exactly;
- frame `f` maps to audio boundaries
  `round(f * 16000 / 15)` and `round((f + 1) * 16000 / 15)`, with adjacent
  intervals meeting and the final boundary equal to 80,000;
- every frame's `actor_states` corresponds exactly once to every declared
  actor ID, with no missing or extra actor;
- PTS and IDs are ordered/unique, references resolve, events have
  `start < end`, the declared formal view is exactly `view0`, its sole pose
  hash key is present, and event/contact semantics are consistent;
- the RGB/depth/semantic sensor extrinsics agree with `camera_rig_0`, the
  listener transform hash agrees with that rig, and named source transforms
  round-trip independently.

## Mouth state

Timeline v2 remains unchanged. `mouth_state.open_ratio` is always `0.0` for
this project. `vocalizing` means that an audio event is active; it does not
claim visual mouth articulation. Episode manifests record
`disabled_for_shortcut_control` explicitly.

## Episode package

An episode contains request, runtime, scene, asset, provenance and timeline
manifests; RGB/depth/semantic observations; dry sources, RIRs, stems and mix;
source/actor/anchor labels; and QA reports. Generated files alone do not imply
admission.

## Counterfactual contract

A counterfactual group declares frozen and changed variables. The first MVP
keeps room, the single camera rig and its sensor calibration, animal identity,
root trajectory and joint poses fixed while exchanging the vocalizing actor.
All formal `view0` RGB/depth/semantic observations must be byte- or
hash-identical, and source/event lineage must explain every audio change.
