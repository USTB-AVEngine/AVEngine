# M2 Status: Articulated Dog Runtime

Formal M2 status: `not_run`.

Bounded research-candidate review-only execution: `pass` on 2026-07-16. The
generated evidence intentionally records top-level `status: review_only`,
`formal_view_ids: []` and `qualification_claim: false`, so this successful
diagnostic execution cannot be aggregated as a formal milestone pass.

The candidate remains `research_candidate`. Its automatic QA status is `pass`,
but `human_visual_review_status` is `not_run` and
`human_review_binding_sha256` is `null`. The package also retains
`allowed_use: review_required` and `redistribution: review_required`. Formal
request loading is fail-closed unless these admission conditions are resolved
and the package becomes `canary_qualified`.

## What the review-only run proved

- The request contains exactly 75 explicit states: 15 Idle, 45 Walk and 15
  Idle, at 15 Hz with PTS ticks from 0 through 236,800 in 3,200-tick steps.
- Habitat used one fresh Simulator and one kinematic articulated object. It
  name-bound 34 spherical runtime joints to 136 joint-position values and
  explicitly reapplied the root and joint pose at every frame.
- The package's articulated-object template explicitly set
  `user_defined.avengine_native_gltf_skin_frame: true`. Missing or false values
  retain upstream Habitat behavior; the upstream skinned-object C++ cases and
  both checked-in golden poses remained exact after the opt-in runtime change.
- Each frame used one observation call for co-located RGB, depth and semantic
  sensors on `camera_rig_0/view0`. These are three modalities of one camera
  viewpoint, not three viewpoints.
- The run performed zero physics steps. Habitat world time stayed exactly
  `0.0 -> 0.0`; per-frame time advancement was zero.
- All 75 frames passed pose/applied-state hash recomputation and state
  readback. Maximum root readback error was
  `1.7881320557894536e-07`; maximum joint-quaternion readback error was
  `3.325239639195843e-08`.
- Semantic ID 200 was visible in every frame, with 872 to 995 animal pixels.
  Saved RGB/depth/semantic arrays were read back and rehashed.

This proves the bounded review execution and fixed-state runtime path. It does
not prove that the mesh and animation are visually acceptable, that contact
quality satisfies the formal exit criterion, or that M2 has passed.

## Candidate and evidence identities

All paths below are ignored local `tmp/` artifacts. They are evidence for the
current review, not files distributed by ordinary Git history.

| Artifact | SHA-256 |
| --- | --- |
| `tmp/m2/rocketbox_rebased_v3/visual.glb` | `633dd0e3485584c5c66126a46f00161939f7756f2bd81f407df05d1b526d8ab8` |
| `tmp/m2/rocketbox_actions_v1/actions.npz` | `6399e730b0b8b24219bc447b3a53a8aeb5424127e93c39d888723845ee0cc768` |
| `tmp/m2/rocketbox_rebased_v3_probe_optin/probe.json` | `920cbbbbd88b5e2e5cf95667663274cd7206cce189db17b7ace6b7f767d484cf` |
| `tmp/m2/rocketbox_beagle_m2_candidate_v4/habitat/animal.ao_config.json` | `86692278fdfe2499b9f3c6505e44399379df9d753cb73b1911b18ff916cfc43c` |
| `tmp/m2/rocketbox_beagle_m2_candidate_v4/asset_manifest.json` | `4110e116ba9a3190caad40e8f8fa91fa49a02d2477dee25138481add5ac433bd` |
| `tmp/m2/rocketbox_beagle_m2_review_request_v3.json` | `f6f2b812291ff14bb02dbda17d2bcbd55d468667f2195ef6dc062c0af7302c4d` |
| `tmp/m2/rocketbox_beagle_m2_habitat_review_v4/evidence.json` file bytes | `84d696a22527d16284adaaa65341963629355a38ac3dabd054eddd610255a1b0` |
| Evidence canonical content recorded inside `evidence.json` | `23a22f0b2b1b89c20a2ba364813d556a4cefbe4c88947304fa42fc38ec738029` |

The package identifies Microsoft Rocketbox `Dog_Beagle_01` at source revision
`0943055db6ec570bcef9f2c8b41c9e5467c808f9`. Its source license snapshot is
MIT, but the historical FBX-to-GLB conversion lacks a complete hash-bound
conversion manifest. Package-level use and redistribution therefore remain a
review decision; this status does not authorize release.

## Automatic QA results and retained warnings

The bounded automatic reports are all `status: pass`,
`qualification_state: research_candidate` and `qualification_claim: false`.

| Check | Measured result | Gate/reference |
| --- | ---: | ---: |
| Skin bind closure | `3.1340474548890995e-08 m` | at most `1e-4 m` |
| Rebase deformation equivalence | `3.0335943629833384e-05 m` | at most `1e-4 m` |
| True source-loop endpoint vertex error | `0 m` | at most `1e-4 m` |
| Maximum one-frame vertex step / rest-bbox diagonal | `0.05332867259732431` | at most `0.1` |
| Minimum animated triangle area | `7.40827740880712e-07 m^2` | greater than `1e-12 m^2` |
| Maximum joint-landmark distance outside mesh bbox | `0 m` | at most `0.02 m` |
| Mouth-joint rotation excursion, Idle/Walk | `0 degrees` | at most `1e-6 degrees` |

Automatic animation analysis explicitly triggered the inherited gait
limitation:

- mean front-paw forward range: `0.176500803458502 m`;
- mean hind-paw forward range: `0.006688130520660487 m`;
- mean hind-paw lateral range: `0.09575715205435381 m`.

The front legs show clear forward/back stepping, while the hind legs change
much less as whole legs and much of their motion is lateral/toe-terminal. This
is the known legacy AVEngine issue raised by the user. It has **not** been
fixed or waived. The current candidate may be used temporarily only for this
research review while the user decides whether the visual result is acceptable
or supplies a replacement gait method.

Contact derivation retained five warnings instead of hiding them:

- Idle anchor motion on both front paws;
- Walk contact-phase horizontal sliding of `35.876 mm` on the front-left paw,
  `32.573 mm` on the front-right paw and `29.044 mm` on the hind-right paw,
  each above the `15 mm` threshold.

These warnings are why an automatic `pass` must not be interpreted as formal
deformation/contact acceptance.

## Review media

The following 75-frame videos all encode arrays from the same review-only
`view0`. RGB, depth and semantic are modalities, not separate cameras.

| Review media | SHA-256 |
| --- | --- |
| `tmp/m2/rocketbox_beagle_m2_habitat_review_v4/review_media/view0_rgb_review.mp4` | `e0af301789bb0e1ae897cd391e8757c65bb64458ce0ef1d78d4f18ad85d62bd3` |
| `tmp/m2/rocketbox_beagle_m2_habitat_review_v4/review_media/view0_depth_review.mp4` | `c9169127794b7c50dc11f521a7d1e16aca2ce4fac273dd87ad449d447de7258f` |
| `tmp/m2/rocketbox_beagle_m2_habitat_review_v4/review_media/view0_semantic_review.mp4` | `f1b6c8f72bf7a492108b19f63ff68ccb7e6401b22c378f135c018f7f57c6c388` |

Closer action-only diagnostic videos make leg motion easier to judge. Their
`side` and `front_quarter` cameras are QA-only and do not add formal dataset
views:

| Diagnostic media | SHA-256 |
| --- | --- |
| `tmp/m2/rocketbox_habitat_review_v4/walk_side.mp4` | `fe7e09fd07605c94188381368b2b2d507765a28aca4ad73405a9036d826e5c85` |
| `tmp/m2/rocketbox_habitat_review_v4/walk_front_quarter.mp4` | `08a33032779e906d61047ae6a54f9898a9f3234b8f5b5cb069cc5bd4b45f742b` |
| `tmp/m2/rocketbox_habitat_review_v4/idle_side.mp4` | `91adb3518070f93f534aa6bccc538736c563878104ccae26ab977a6d234dcb86` |
| `tmp/m2/rocketbox_habitat_review_v4/idle_front_quarter.mp4` | `e3f651ac43058b534ca10f4030e911e22b660340e6db42ad1e6c94ceab9f9c31` |

## Why formal M2 remains `not_run`

The current 75-frame evidence records dirty AVEngine and Habitat runtime
worktrees, which is permitted for the separately named review path but is
rejected by formal capture. More importantly, the package is not
`canary_qualified`: human visual review has not run, no review artifact is
bound to the candidate/media hashes, and provenance/use review remains open.

The next admission sequence is:

1. The user reviews the exact hash-bound 75-frame and close diagnostic media,
   explicitly judging mesh/skin alignment, the retained hind-leg gait,
   foot sliding/penetration/hovering, Idle stability and the no-mouth policy.
2. Preserve that decision in a human-review artifact bound to the exact asset
   manifest and reviewed-media hashes. An acceptance may explicitly tolerate
   the known gait only for the bounded M2 canary; it must not claim the defect
   was fixed.
3. Resolve the package-level allowed-use/redistribution decision and complete
   the admission record required for `canary_qualified`.
4. Commit and clean both repositories, update the runtime lock, then run the
   formal loader/capture. Formal evidence must retain exactly `view0`, the same
   co-located modalities and all 75 states.
5. Change formal M2 status from `not_run` only after that clean evidence passes
   every M2 exit criterion.

The executable review procedure is in [M2_EXECUTION.md](M2_EXECUTION.md).
