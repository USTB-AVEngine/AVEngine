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
  capture every visual sensor without advancing the state
  update named RLR source/listener state from the same timeline
assemble exact audio sample intervals and verify outputs
```

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
  `start < end`, declared views are exact, cross-view pose hashes agree, and
  event/contact semantics are consistent.

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
keeps room, cameras, animal identity, root trajectory and joint poses fixed
while exchanging the vocalizing actor. Visual observations must be byte- or
hash-identical, and source/event lineage must explain every audio change.
