# Codex Current 1-3 Follow-Up Plan

Date: 2026-07-09

This file is the source of truth for the work the user approved today. It is
intentionally separate from older Claude-authored plan remnants.

## Goal

Ship the current review/demo pipeline improvements, then push the focused
changes to GitHub.

## Scope

1. Add per-source effective sound-frame statistics. **Status: done**
   - Compute the count from the per-source rendered binaural signal, not from
     dry audio filenames.
   - Write per-frame booleans and counts into `apartment_v1_metadata.json`.
   - Show the count in the review overlay next to the existing FOV/centerVis
     counts.
   - Silent/muted sources must report zero effective sound frames.

2. Close the current deterministic event-demo fixes. **Status: done**
   - Keep explicit animation playback for requested `wanted_anim`.
   - Keep the left-rear to right-front beagle demo at walking speed.
   - Keep review overlay wording explicit about center-point visibility.
   - Regenerate one review clip for user audit.

3. Prepare the new animation and room expansion path. **Status: ready for the
   next adapter slice**
   - Keep Mixamo/Quaternius/ReplicaCAD status current.
   - Do not start large generated data commits.
   - Push code/docs/probe changes that are ready and tested.

## Current Review Artifact

- `/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output_apartment_v2_left_rear_to_right_front_walkspeed_review/clips/clip_0000/videos/side_by_side_review_annotated.mp4`
- Overlay sanity:
  - `GOLDEN silent stationary | sound 0/75`
  - `BEAGLE dog_sharp_bark walking | sound 31/75`

## Push Rule

Commit and push only focused source/test/docs changes. Do not commit generated
MP4/WAV/PNG outputs, `external/SPEAR/tmp`, `/data/datasets`, or unrelated dirty
workspace files.

## Completion Record

Completed on 2026-07-09 by Codex.

- AVEngine docs commit pushed to `origin/main`:
  `e0b3f7a docs: record Codex follow-up plan`
- SPEAR implementation commit pushed to
  `eastforward/feature/plan2-flag-generator-m1`:
  `8504b2cb feat(spike): add deterministic demos and sound metadata`

Verified before the SPEAR commit:

- `spear-env`: 114 focused tests passed.
- `ss2`: 10 direction/Habitat-oriented tests passed.
- `git diff --cached --check`: passed.

What was intentionally not completed in this slice:

- Generated review MP4/WAV/PNG outputs and `external/SPEAR/tmp` artifacts were
  not committed.
- The next adapter slice for importing/using new Mixamo animations and
  ReplicaCAD rooms was not implemented yet; this slice only keeps the data
  roots/probe status ready for that follow-up.
- Existing unrelated dirty SPEAR workspace files were left untouched.
