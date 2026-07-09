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
