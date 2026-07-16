# M2 Status: Articulated Dog Runtime

Formal M2 status: `not_run`.

Latest replacement research-candidate review-only execution: `pass`. The
generated r3 evidence intentionally records top-level `status: review_only`,
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
  `2.3841724416940023e-07`; maximum joint-quaternion readback error was
  `3.241414114896202e-08`.
- Semantic ID 200 was visible in every frame, with 889 to 1,037 animal pixels.
  Saved RGB/depth/semantic arrays were read back and rehashed.

This proves the bounded review execution and fixed-state runtime path. It does
not prove that the mesh and animation are visually acceptable, that contact
quality satisfies the formal exit criterion, or that M2 has passed.

## Candidate and evidence identities

All paths below are ignored local `tmp/` artifacts. They are evidence for the
current review, not files distributed by ordinary Git history.

| Artifact | SHA-256 |
| --- | --- |
| `examples/m2/motion_profiles/quadruped_dog_to_rocketbox_beagle_v1.json` | `ef09d9e6238c4cb0fad11a01ffd214c7132b911f86b890091321ccfdf8dfed7d` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_world_left_r2/retarget.json` | `71faf0426089181ec7fd92911c9c85c059eecfe8b1b77e2230f87a49ec7af7c4` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_world_left_r2/visual.glb` | `051d3c653187df87296f6b85bfde4f4d2a276146910f3e414cd63196db3d1a4b` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_rebased/visual.glb` | `788a667537f7660bac5e128c38c2182453d1d4a9a4f8380343e7a9fa1947538c` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_actions/actions.npz` | `b77457be2808fc0495ca7a8bc97978681598afca1caff18aca0761de5891c645` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_motion_qa/report.json` | `0c7531f9e605edb88978fc79f65dcca0fd3e0ed467af54baf49651e6c9d1aabb` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_contacts_r2/contact_phases.json` | `82f128010e9ccf9e828e8693a26a6aca6b8e14569c97129b2f449118328a3d04` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_probe/probe.json` | `4e58d4ee4d0be7c163f013fde2c0c0c582b0eab2c63bd0843ac0f84de202cc75` |
| `tmp/m2/rocketbox_beagle_m2_candidate_v5_world_left_r3/asset_manifest.json` | `706631ee90ec9102bb76939dd7f75ca410757efd3c7c11580fa31e4d52183feb` |
| `tmp/m2/rocketbox_beagle_m2_review_request_v5_world_left_r3.json` | `361924effa4ce7172102abaed353d4cf12af7ca059883d40f1e7b66c13dc3bbc` |
| `tmp/m2/rocketbox_beagle_m2_habitat_review_v5_world_left_r3/evidence.json` file bytes | `689de803d8f79c6dc0e7a5f735fefdec5163997ee67980e9bb12a6c7c5e4eb39` |
| Evidence canonical content recorded inside `evidence.json` | `95ccffbb252eed0e40f37d2a44fb4c428147b0077c2177a63369420f9331b290` |

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
| Skin bind closure | `3.277782778364028e-08 m` | at most `1e-4 m` |
| Rebase deformation equivalence | `8.589642556400268e-07 m` | at most `1e-4 m` |
| True source-loop endpoint vertex error | `0 m` | at most `1e-4 m` |
| Maximum one-frame vertex step / rest-bbox diagonal | `0.06345127472300958` | at most `0.1` |
| Minimum animated triangle area | `7.40813180506935e-07 m^2` | greater than `1e-12 m^2` |
| Maximum joint-landmark distance outside mesh bbox | `0.0009399289276008083 m` | at most `0.02 m` |
| Mouth-joint rotation excursion, Idle/Walk | `0 degrees` | at most `1e-6 degrees` |

The replacement action no longer triggers the inherited hind-gait metric:

- mean front-paw forward range: `0.21719317695994803 m`;
- mean hind-paw forward range: `0.28810084185901486 m`;
- mean hind-paw lateral range: `0.010916511876523767 m`;
- hind-left/right forward range: `0.2834736434147972/0.29272804030323246 m`;
- hind-left/right lateral range: `0.0070530774475609515/0.014779946305486583 m`.

The profile-bound generic report also passed all declared limb-chain,
joint-speed, hind/fore ratio and left/right symmetry thresholds. It measured a
rest-length-normalized hind/fore forward ratio of `0.930712` and maximum
left/right symmetry difference of `0.032122`.

Contact derivation retained eight warnings instead of hiding them:

- Idle anchor motion on all four paws, with vertical ranges from `7.37` to
  `14.77 mm`;
- Walk actor-space contact-step sliding of `47.030`, `42.297`, `43.708` and
  `47.385 mm` for front-left, front-right, hind-left and hind-right,
  respectively, each above the `15 mm` threshold.

These warnings are why an automatic `pass` must not be interpreted as formal
deformation/contact acceptance.

## Body-plan-aware replacement boundary

AVEngine now has a bounded offline route for rest-aware world-left rotation
transfer, strict body-plan/motion-family profiles and generic semantic-chain
motion QA. The architecture and its non-claims are documented in
[MOTION_RETARGETING.md](../architecture/MOTION_RETARGETING.md).

This route produced the r3 action, package, request and media identities above.
The previous legacy candidate remains historical evidence only; its hashes
were not reused as acceptance evidence for the replacement.

The currently enabled profile boundary is a bounded canine-to-Beagle route.
A cat requires a separate feline motion-family profile and canary. Bird ground
locomotion, bird flight and fish swimming have separate reserved adapters and
remain fail-closed. Generic motion QA also does not prove contacts or scene
root speed: both must be validated independently for the exact action and
trajectory before a new candidate can be reviewed.

Every retarget output starts as `research_candidate`. No successful compiler,
motion-QA or Habitat review-only run may change formal M2 from `not_run` without
the hash-bound human review and `canary_qualified` admission described below.

## Review media

The following 75-frame videos all encode arrays from the same review-only
`view0`. RGB, depth and semantic are modalities, not separate cameras.

| Review media | SHA-256 |
| --- | --- |
| `tmp/m2/rocketbox_beagle_m2_habitat_review_v5_world_left_r3/review_media/view0_rgb_review.mp4` | `f789260e70a99b008685377b9d18d239d4bdbf6aa71fd20ccda4f09ee8bf03a9` |
| `tmp/m2/rocketbox_beagle_m2_habitat_review_v5_world_left_r3/review_media/view0_depth_review.mp4` | `2b8302f3c896eb35480a6878cb4d8e717e3bc47835e15632495bb12c148cec4a` |
| `tmp/m2/rocketbox_beagle_m2_habitat_review_v5_world_left_r3/review_media/view0_semantic_review.mp4` | `f5414026b332e01576a41370a73ca4b8b9ab7b9b89cb1e1d45752afe33286d24` |

Closer action-only diagnostic videos make leg motion easier to judge. Their
`side` and `front_quarter` cameras are QA-only and do not add formal dataset
views:

| Diagnostic media | SHA-256 |
| --- | --- |
| `tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/walk_side.mp4` | `e40854b97377955c6e1451b57146eb68994df0ca90db38533ba54dc149b26a5d` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/walk_front_quarter.mp4` | `a07e7a26d1b4f4afea7ebb644889884672fb4be04da83c50b2457aa164d3a4d4` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/idle_side.mp4` | `0319bdea8ebfa0f33303032326e607f6cf2d3a55689a5fe35abe1a56d0847e6b` |
| `tmp/m2/rocketbox_motion_retarget_v2_a_habitat_review_r2/idle_front_quarter.mp4` | `b990699b05481b7176fc2dceacf3761be579a0c3c77e4689d3805ba270a13d97` |

## Why formal M2 remains `not_run`

The current 75-frame evidence records dirty AVEngine and Habitat runtime
worktrees, which is permitted for the separately named review path but is
rejected by formal capture. More importantly, the package is not
`canary_qualified`: human visual review has not run, no review artifact is
bound to the candidate/media hashes, and provenance/use review remains open.

The next admission sequence is:

1. The user reviews the exact hash-bound 75-frame and close diagnostic media,
   explicitly judging mesh/skin alignment, the replacement hind-leg gait,
   foot sliding/penetration/hovering, Idle stability and the no-mouth policy.
2. Resolve the actor trajectory/cadence binding and contact/root-speed gate;
   the current actor-space contact warnings must not be treated as world-space
   foot-lock acceptance.
3. Preserve the user decision in a human-review artifact bound to the exact
   asset manifest and reviewed-media hashes.
4. Resolve the package-level allowed-use/redistribution decision and complete
   the admission record required for `canary_qualified`.
5. Commit and clean both repositories, update the runtime lock, then run the
   formal loader/capture. Formal evidence must retain exactly `view0`, the same
   co-located modalities and all 75 states.
6. Change formal M2 status from `not_run` only after that clean evidence passes
   every M2 exit criterion.

The executable review procedure is in [M2_EXECUTION.md](M2_EXECUTION.md).
