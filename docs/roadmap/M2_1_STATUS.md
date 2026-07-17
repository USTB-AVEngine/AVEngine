# M2.1 Status: Appearance L9 and Cross-Species Diagnostics

M2.1 is a post-M2 research workstream, not a new dataset-admission claim.  The
canonical M2 Beagle remains the only `canary_qualified` animal.  Every M2.1
appearance instance and every cross-species result remains a
`research_candidate`; `qualification_claim` is false and
`formal_dataset_registration_authorized` is false.  The Beagle per-instance
promotion review remains `not_run`.  For the exact retained cross-species v7
videos, the project owner has separately accepted the cat and Golden Retriever
research visuals and rejected the horse because its legs fold unnaturally;
that historical feedback remains immutable below.  The current horse preview
instead uses a research-only local-TR v2 repair that passes two-room runtime
readback and engineering visual self-review.  It is not a hash-bound
project-owner decision or a formal species-admission claim.

## Beagle appearance contract

The Beagle request varies four appearance axes, each with exactly three
reviewable levels:

| Axis | Exact domain |
| --- | --- |
| `size` | `small`, `medium`, `large` |
| `body_build` | `slim`, `standard`, `stocky` |
| `coat_profile` | `light_tricolor`, `standard_tricolor`, `dark_tricolor` |
| `life_stage` | `young`, `adult`, `senior` |

Coat labels are not a global animal-color enum.  Package validation uses an
exact `(species_id, breed_id)` registry and rejects an unregistered pair even
when a coat label happens to exist for another breed.  The current registered
three-level domains are:

| Species/breed | Exact breed-scoped coat domain |
| --- | --- |
| Beagle | `light_tricolor`, `standard_tricolor`, `dark_tricolor` |
| Golden Retriever | `light_golden`, `classic_golden`, `dark_golden` |
| generic domestic cat | `black`, `charcoal_gray`, `silver_gray` |
| generic domestic horse | `black`, `dark_bay`, `bay` |

Only the Beagle domain has the nine-point L9 below.  The cross-species work
tests one `medium`/`standard`/`adult` diagnostic point for each of the other
three domains; it does not claim their own nine variants or OFAT coverage.  A
bird still needs its own reviewed breed/body-plan and coat registration.

The generated design is the canonical OA(9, 4, 3, 2): each level occurs three
times and every ordered pair of levels across any two axes occurs once.

| L9 | `size` | `body_build` | `coat_profile` | `life_stage` |
| ---: | --- | --- | --- | --- |
| 01 | `medium` | `standard` | `standard_tricolor` | `adult` |
| 02 | `medium` | `slim` | `light_tricolor` | `senior` |
| 03 | `medium` | `stocky` | `dark_tricolor` | `young` |
| 04 | `small` | `standard` | `light_tricolor` | `young` |
| 05 | `small` | `slim` | `dark_tricolor` | `adult` |
| 06 | `small` | `stocky` | `standard_tricolor` | `senior` |
| 07 | `large` | `standard` | `dark_tricolor` | `senior` |
| 08 | `large` | `slim` | `standard_tricolor` | `young` |
| 09 | `large` | `stocky` | `light_tricolor` | `adult` |

This L9 is a balanced combination review, not a one-factor-at-a-time test.  Its
hash-bound batch explicitly records:

```text
strategy: separate_one_factor_at_a_time_v1
status: not_run
required_before_formal_promotion: true
l9_substitution_allowed: false
```

Consequently, successful realization, automatic QA and two-room videos cannot
promote these nine variants beyond `research_candidate`.  Formal promotion
requires a separate OFAT run plus the applicable per-instance geometry,
material, motion, contact, Habitat readback, provenance and human-review
gates.

## Current Beagle result

All nine L9 points were rebuilt after the final material and scale-animation
safety hardening pass so that the retained reports bind the current producer
bytes. The input builder accepts only the SHA-256 of the current realizer file;
there is no historical-digest allowlist. All nine v9 reports bind realizer
`452dde3a3ca037eb7607c8e862ac281fb61988732e656e54858da0c2a0d5bc5d`
and pass the breed pattern audit. A separate verifier snapshots the request,
source GLB and textures before parsing and independently checks geometry,
joint/skin structure and output pixels; the largest observed position error
is approximately `2.09e-7 m` and the largest channel error is approximately
`0.00196`. The v9 GLB and standalone PNG bytes are identical to the immutable
v8 predecessor. The following local chain passes for every point:

- strict post-export GLB material readback (`alphaMode` effectively
  `OPAQUE`, base-color alpha `1`, `emissiveFactor == [0, 0, 0]`, no
  `emissiveTexture`, `metallicFactor == 0`, roughness at least `0.72`, bounded
  `KHR_materials_specular` factor/color contribution, no
  `metallicRoughnessTexture`, and no unreviewed material extension); missing
  specular controls and the root `extensionsUsed` declaration are written
  explicitly and then independently read back, so glTF defaults cannot bypass
  the bound;
- explicit Habitat shader selection bound from the variant spec through the
  static probe, action review and compiled AO config (`pbr` for M2.1, while the
  formal M2 compatibility default remains `phong`);
- independent visual topology, UV, skin weights, joint hierarchy, inverse
  bind matrix and Idle/Walking channel equivalence;
- fail-closed rejection of a scale animation targeting the similarity-bake
  scale node or any relevant ancestor;
- exact scale derivation from the qualified M2 canary rather than a
  caller-supplied scale waiver;
- static, deformation, animation, Habitat rest and Habitat action probes;
- all four idle and walking paw records plus the scale-normalized
  world-contact gate; and
- research package assembly and strict wrapper/core/array/media verification
  in both rooms.

The local technical capture set is `9 variants × 2 rooms = 18` RGB videos.
Every video is
`320×240`, 15 fps and 75 frames.  A frame-by-frame semantic-mask audit found a
minimum margin of 13 pixels on every edge in every frame.  The
MP3D review keeps a fixed `-3.56 m` camera-depth offset while moving laterally
in profile; the controlled-room review uses a dedicated camera lowered by
`0.18 m`. Neither route changes actor height or the derived ground-contact
trajectory. The inherited canonical
hind-leg motion does not substitute for M2.1 per-instance formal human visual
review. That review remains `not_run`, asset admission remains `blocked`, and
`qualification_claim` remains false for all nine new appearance instances.

The preceding v12 capture set was not accepted as final evidence. An
independent audit first exposed an incorrect raw-byte frame-hash recomputation;
after switching the audit to the producer's sensor/dtype/shape-bound
`avengine_array_sha256_v1` contract, it also exposed a genuine MP3D framing
regression: the `-3.0 m` depth placed semantic ID 200 on the bottom image edge,
giving a zero-pixel margin. The tracked preset and its regression test now bind
`-3.56 m`; the fresh v13 capture restores the 13-pixel minimum without changing
actor height, contact trajectory, action or material. The failed v3/v4 local
audits remain untouched and are not cited as passing evidence. The current v14
set is a fresh request/capture replay bound to the v12 package manifests; its
18 RGB arrays and review videos are byte-identical to v13 while its wrappers
bind the new manifests. The repository test suite passes `797` tests and the
final Ruff check reports no findings.

## Body-plan boundary

The current shared action, anchor, package and contact path is bounded to
terrestrial quadrupeds.  Its present package-level semantic contract requires
`body`, `head`, `muzzle` and four paw/contact roles.  Some mathematical
primitives can operate on declared vectors and chains, but that does not make
the surrounding four-paw/muzzle contract universal.

In particular, a bird must not be admitted by renaming a beak as `muzzle`,
inventing four paws or reusing canine thresholds.  Avian support requires a
new reviewed body-plan profile whose identity and content digest are bound by
the request, realized asset, action, anchors, contact evidence and QA reports.
At minimum that profile must declare:

- actor-frame `forward` and `up` directions (with a consistently derived
  lateral direction);
- anatomical region roles used by deformation and motion QA;
- emitter roles appropriate to that body plan; and
- contact/support roles appropriate to its locomotion mode.

Unknown or mismatched body-plan profiles must fail closed.  Only after those
roles are resolved may body-plan-neutral math be reused.  The same rule
applies when a terrestrial quadruped needs a different motion family or
species-specific thresholds.

The cat, horse and Golden Retriever outputs under the local cross-species
delivery are therefore diagnostics only.  They exercise the current pipeline
and expose projection, contact, material, motion or anatomy limitations; they
do not register those species or authorize dataset capture. Golden Retriever
current research use is project-owner-authorized and is not a rights blocker.

The `cross_species_delivery_v7` technical captures use current producer bytes
and an explicit PBR contract for every species. All six videos decode as
`320×240`, 15 fps and 75 frames, and their strict evidence wrappers verify.
Across all six videos, the overall minimum semantic-mask edge margin is 8
pixels. The prior Golden copper/lacquer response was a rendered Phong-path
failure: the glTF PBR/specular metadata was not the active shading route.
Switching the hash-bound research package, probe and action review together to
PBR removes that response without replacing the original 4096² atlas or
authored normals. All three probes therefore have technical capture `pass`;
retained visual findings and admission boundaries are recorded separately:

| Probe | Technical capture | Project-owner research visual review | Asset admission | Rights | Retained findings and admission boundary |
| --- | --- | --- | --- | --- | --- |
| domestic cat | `pass` | `pass` | `blocked` | CC0 source snapshot | The owner accepts the dark charcoal appearance and slight sliding. The darkness comes from the deliberately selected `charcoal_gray` realization plus the dark MP3D region, not Flux. Formal promotion still lacks a cat-specific body-plan/contact profile; the source-motion projection differs by `0.280053 m` at maximum vertex distance. |
| domestic horse | `pass` | `fail` | `blocked` | CC0 source snapshot | This row records the immutable v7 result: the matte material passes, but the owner rejects its rotation-only projection because the legs visibly fold/break. Its source-motion projection differs by `1.360980 m`; it is no longer the current corrective preview. |
| Golden Retriever | `pass` | `pass` | `blocked` | project-owner authorized for current AVEngine research | The owner accepts the motion, slight sliding and natural matte golden-brown response. Current research use is not rights-blocked. Formal promotion still lacks a Golden-specific body-plan/contact profile and retains a `0.369171 m` source-motion projection difference. |

The folded-leg diagnosis was traced to the v7 retarget path dropping authored
child-joint local translations and retaining rotations only. The corrective
`horse_local_tr_review_v2` route preserves both local translation and rotation
channels and drives translation-bearing joints through explicit prismatic
components in Habitat. Fresh `blender_custom` and `habitat_mp3d_example`
captures each contain 75 fixed states, keep semantic ID 200 visible in every
frame and advance neither physics nor world time. Maximum skin-link matrix
readback errors are `3.7431037291923985e-07 m` and
`8.534135025328737e-07 m`, respectively. Engineering visual self-review of the
two current RGB/Topdown videos finds no folded or broken legs, so local-TR v2
supersedes v7 for current horse research preview. The evidence is still
`review_only`, has `qualification_claim: false` and `formal_view_ids: []`, and
does not supply a horse-specific formal body-plan/contact profile or a
project-owner promotion decision.

The PBR result is a material-path repair, not a species-admission decision.
Cat, horse and Golden remain `research_candidate` even though 3/3 packages and
6/6 historical v7 capture wrappers pass. The v7 exact-video owner feedback is
`pass`, `fail`, `pass` respectively; the later horse engineering self-review
does not rewrite that record. No hash-bound formal promotion decision has been
issued, and `qualification_claim` remains false. The Golden tracked spec records
`AVENGINE_PROJECT_OWNER_AUTHORIZATION` with `research_only` use. The immutable
cross-species v7 evidence predates that owner confirmation and therefore
retains its historical `UNRESOLVED` snapshot rather than being rewritten. That
historical value no longer blocks current AVEngine research use. The Golden
faceting/fur-shell observation remains an asset note; the final candidate keeps
the authored normals instead of applying an unsafe automatic weld or smooth.

## Room interpretation

The two review environments serve different purposes:

- `blender_custom` is a deliberately compact controlled-lighting canary.  It
  makes silhouette, grounding, motion and suspicious material response easy
  to compare under stable conditions; it is not intended to look like a
  naturally furnished production room. Its scene instance names two explicit
  point lights, but the current review evidence does not run or validate a
  dynamic shadow-map pass, so the inserted animal has no proven cast shadow.
- `habitat_mp3d_example` uses a real MP3D scan.  It tests the animal against
  real scanned surfaces, scale, occlusion and scan topology rather than a
  hand-authored clean room.  Scan holes, irregular geometry and baked visual
  character are properties to inspect, not evidence that the controlled
  canary should imitate them. The installed MP3D config declares a flat stage,
  an empty light-setup table and `default_lighting: no_lights`; the apparent
  room illumination is baked into the scan textures. It contains no original
  light actors that can be recovered by importing the mesh into UE. Importing
  only that mesh into an otherwise unlit UE level therefore gives no dynamic
  animal shadow. UE can produce one after explicit Directional/point/Sky light
  and cast-shadow configuration, but that lighting must be calibrated against
  the already baked room appearance.

The synchronized right-side Topdown panel is review-only derived media. It
draws the loaded navmesh, the sole `view0` camera and FOV, the actor trajectory
and the named source anchors beside the RGB frame. It does not create `view1`,
change `camera_count: 1`, or enter the formal RGB/depth/semantic artifact map.
The final bundle contains 24 videos: 18 Beagle L9 room pairs plus two each for
cat, Golden Retriever and the corrected local-TR v2 horse.

The MP3D panel additionally draws 89 filtered object footprints obtained from
`habitat_sim.semantic_scene.objects[].obb.local_to_world` in the bound `.house`
descriptor. This is descriptor metadata, explicitly
`descriptor_semantics_not_object_detection`, not image inference or an object
detector. The custom room supplies no semantic descriptor, so its footprint
count is exactly zero; the renderer does not invent furniture. The navmesh
itself still means only binary navigability, not object identity.

Passing both views is useful review coverage, but neither room substitutes for
the missing OFAT or species/body-plan admission gates.

## Local evidence index

Large generated evidence stays under ignored `tmp/` paths and is not tracked
in Git. These paths are a local evidence index only, not release artifacts or
a reproducibility guarantee:

- `tmp/m2/beagle_appearance_l9_v1.json` — exact L9 batch and explicit OFAT
  status;
- `tmp/m2/beagle_l9_realized_v9/`,
  `tmp/m2/beagle_l9_canonical_visual_v7/` and
  `tmp/m2/beagle_l9_rebound_actions_v8/` — final hardened current-source
  appearance and action realizations;
- `tmp/m2/beagle_l9_package_inputs_v10/`,
  `tmp/m2/beagle_l9_auto_qa_v11/`, `tmp/m2/beagle_l9_probe_v10/`,
  `tmp/m2/beagle_l9_final_action_review_v9/`,
  `tmp/m2/beagle_l9_world_contacts_v10/` and
  `tmp/m2/beagle_l9_material_readback_v1/` — final PBR-bound local QA, contact
  and independent material-readback evidence;
- `tmp/m2/beagle_l9_packages_v12/`,
  `tmp/m2/beagle_l9_captures_v14/` and
  `tmp/m2/beagle_l9_final_audit_v6/` — nine research packages, 18 two-room
  technical captures and 36 local visual audit sheets. The independently
  replayed v6 audit file
  SHA-256 is
  `7b7156191eaadbba6c1141c29724a5c8c00059b832782891986b243d572f813c`;
  its 18-video index SHA-256 is
  `721fa2be9c445ed2d615c21077c1234e02ea99ff1af40f06a177d4aa10d9ee03`;
- `tmp/m2/cross_species_delivery_v7/` — current-producer PBR package, request,
  wrapper, encoding, semantic-margin, video-hash and visual-findings index for
  cat, horse and Golden Retriever. The hash-bound files are:
  `cross_species_v7_audit.json` (file SHA-256
  `4d35918745014eed342674ef322aa0f7fb473bfa19336a96397486a5751684fe`,
  canonical `audit_content_sha256`
  `aa732cbc5b56f611ef29614c581135b007d4b359e9643f19a79da69294be2d7f`),
  `cross_species_v7_video_index.json` (SHA-256
  `209471409b1a50a41155afc5c236648d251df676485b4005ce1e997be7fb749a`),
  `cross_species_v7_visual_findings.json` (SHA-256
  `d2520b4ceab26e74ca4226d6cd91dbcb230ff520f55fbf8728347546eaaff386`)
  and `cross_species_v7_delivery_summary.json` (SHA-256
  `5c812067b71663355d38bc9410fa0b8974be97cf5c6d6dbe47022190e3c7ad11`).
  The later exact-video owner feedback is
  `tmp/m2/cross_species_delivery_v7/user_visual_feedback_v1.json` (SHA-256
  `fa457fb558cb80d50d1a1375cf40b14b649a4c4a988d417140f6fd95b9a09265`):
  cat `pass`, horse `fail`, Golden Retriever `pass`. It supersedes the old
  visual interpretation without mutating v7; the later project-owner research
  authorization is recorded minimally in the tracked Golden spec above.
- `tmp/m2/horse_local_tr_review_v2/` — corrective research-only horse captures
  in `blender_custom_capture/` and `habitat_mp3d_example_capture_v2/`. Their
  canonical evidence-content SHA-256 values are
  `ea64b4dd18cc4111bd74a989e402ebf3ea6adff19a043e0ff277ec2a91ed527a`
  and
  `1894cedf2c582a2ac68f6503f3edc32cf960ba69bf01d8258c4dd16bada508db`;
  both explicitly deny qualification and formal capture.
- `tmp/m2/topdown_review_delivery_v4/videos/` — final 24-video review tree:
  Beagle `01` through `09`, cat, Golden Retriever and corrected horse, each in
  both rooms. Each `560×240` video keeps the original `320×240` RGB on the left
  and a synchronized `240×240` derived Topdown panel on the right at 15 fps for
  75 frames. Actor heading follows the nearest non-zero trajectory tangent;
  camera heading follows the rig local negative-Z axis. These are local QA
  artifacts, not formal views. All 24 strict verifications and full decodes
  pass; all 18 Beagle entries bind v14 captures, and all 24 video bytes equal
  their already-reviewed v3 counterparts. The v4 index SHA-256 is
  `b8498156e54ac47b3e2ce6294e4f4b64d822329adb493ce26db318e53716acfd`;
  the delivery-audit file SHA-256 is
  `444a2069872452081dc946f5b66ba290b778016c86d9ce35fea89110617038d1`
  and its canonical content SHA-256 is
  `8cb572d875c9a5550a56e37ebf913851360f377a391797cf76185cb8f3b460b4`.

Do not commit the GLBs, NPZ arrays, captures or review videos from those
directories. A promoted decision must bind exact fresh hashes rather than rely
on a mutable local `tmp/` index.
