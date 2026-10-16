# Second Retarget Facing Review Design

## Purpose

Give the user a browser-based, human-authoritative review of whether the
rejected second Route-2 male retarget walks in the direction the bound Pixal
body faces. This diagnostic must distinguish a bad bind/canonical-facing
assumption from a bad animation/root-motion retarget and from a misleading
camera angle.

## Immutable inputs and classification

- The subject is exactly the rejected second-attempt diagnostic reconstruction
  `second_attempt_rotation_only_diagnostic_reconstruction_v1`.
- The existing diagnostic GLB, manifest, Front/Side/Feet media, the rejected
  retarget record, and the approved static audit are read-only authenticated
  inputs.
- Derived files publish to a new no-replace directory named
  `second_attempt_facing_review_v1`.
- The package remains `technical_diagnostic_only` and rejected. It cannot write
  an approval record or enter the formal Route-2 asset registry.

## Facing evidence

The diagnostic samples the animated semantic skeleton at every Walking frame.

1. The body-right axis is derived independently from the left/right
   shoulder and hip pairs.
2. Its sign is authenticated against the static bind snapshot whose canonical
   front is `-Y`; motion direction is never used to choose the sign.
3. The body-forward vector is the horizontal vector perpendicular to the
   signed body-right axis and `UP +Z`.
4. Travel direction is derived independently from the horizontal pelvis/root
   trajectory. A central difference is used inside the clip and one-sided
   differences at the ends.
5. Frames with negligible displacement are labelled `travel undefined` and
   do not fabricate an alignment score.
6. For valid frames, the package records dot product and signed angle between
   body forward and travel. It also compares body forward with canonical
   `FRONT -Y`.

The numeric summary reports per-frame values, median and worst alignment,
reversed-frame ratio, and static-bind facing authentication. These values are
diagnostic aids; the user remains the final visual reviewer.

## Media and browser page

The page is a read-only extension of the established motion-review style. It
contains:

- the existing Front, Side, and Feet videos from the second reconstruction;
- a newly derived Top video with a persistent root trail;
- a blue body-forward arrow, red travel arrow, and grey canonical `FRONT -Y`
  arrow;
- synchronized playback and frame stepping for all four views;
- current-frame dot product, signed angle, and `aligned`, `sideways`,
  `reversed`, or `travel undefined` label;
- a compact explanation of whether the evidence points to static bind/front
  canonicalization or animation/root-motion retargeting.

The page must never display an Approve button for this rejected diagnostic. It
offers explicit reviewer observations only: `sideways`, `reversed`,
`aligned but deformed`, or free-form notes held in browser memory/exported by
the user, not written into formal approval state.

## Publication and validation

- Hash-lock every input and derived artifact in `facing_review_manifest.json`.
- Use Blender 4.2.1 and 30 fps, preserve all 33 Walking frames, and validate
  the one-action GLB before sampling.
- Decode-check every MP4 and verify the Top overlay contains all three arrow
  colours in valid-motion frames.
- Serve through a localhost Flask server with no-store responses and
  hash-checked media routes.
- Run unit tests for vector sign, angle classification, zero-motion handling,
  path safety, no-replace publication, HTML content, and source tamper refusal.
