# Border Collie target-native cross-check — 2026-07-23

Status: **project-owner visual acceptance for research canaries**. This is not
yet a formal public-dataset admission because the automatic Walking deformation
gate still reports a local outlier.

## What was accepted

The project owner accepted the generated Border Collie identity, single-tail
geometry and the leveled `Walking` motion on 2026-07-23. The accepted candidate
is a genuinely new target-native asset: its FLUX canonical image, Pixel3D mesh,
repaired PBR topology, TokenRig skeleton and weights are not derived from the
previous Labrador, Beagle or Quaternius body shape. Quaternius donates motion
only.

The reusable build order proven by this candidate is:

```text
real breed references
  -> undistilled FLUX canonical image
  -> project-owner 2D anatomy/breed review
  -> Pixel3D target-native mesh
  -> watertight topology and emission-based PBR bake
  -> TokenRig target-native skeleton and weights
  -> reviewed cardinal heading normalization
  -> four-semantic-foot support-plane leveling
  -> semantic Idle/Walking motion retarget
  -> rotation-invariant deformation audit and six-view media review
```

## Exact retained evidence

Paths below are relative to the active SPEAR checkout and remain under its
ignored `tmp/` evidence root:

- Animated GLB:
  `tmp/new_animal_assets/border_collie_target_native_v2_20260722_01/pixal_pipeline_seed702238/tokenrig_seed42_v2_prestarted/heading_positive_x_v1/support_plane_level_v1/retarget_v5_spike_yaw0_matched_amp0p40/border_collie_animated.glb`
- Support-plane measurement:
  `.../support_plane_level_v1/manifest.json`
- Walking+Idle deformation audit:
  `.../retarget_v5_spike_yaw0_matched_amp0p40/skinned_deformation_audit_rotation_invariant_walk_idle_v2.json`
- Six review videos and three contact sheets:
  `.../support_plane_level_v1/final_review_v1/`

All six videos read back as H.264, 512×384, 8 frames at 8 fps and cover the
complete source action range 0–40. They comprise side/front/rear views for both
`Walking` and `Idle`.

## Quantitative result and retained exception

- Original rigid support-plane tilt: `13.775992°`.
- Post-level semantic-foot heights: within `0.023 mm`.
- Rotation-invariant rest scale before/after leveling:
  `1.229669 m / 1.229744 m` (about `0.006%` difference).
- Native Quaternius controls pass with maximum Walking extension ratios of
  `0.0511` for Dog and `0.0471` for Cat.
- Border Collie `Idle` passes automatically at `0.0113`.
- Border Collie `Walking` reports `0.0884`, above the unchanged `0.08` reject
  threshold, even though the project owner accepted the rendered motion.

The candidate may therefore be used for instance-attribute experiments and
Apartment research canaries with the exception attached. It must not be
described as automatically deformation-qualified or formally released until
that local Walking outlier is either repaired without changing identity or an
explicit reviewed production policy replaces the current gate.

## What must be reused versus regenerated

Reuse the mechanics and validators above. For a different breed, regenerate
the canonical image, Pixel3D mesh, topology, skeleton and weights. For an
instance of this same accepted Border Collie morphotype, retain geometry and
rig and vary only its declared size, body build, life stage and one of three
breed-scoped, real-reference-guided coat profiles.
