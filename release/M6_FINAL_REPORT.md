# M6 Final Report — Feasibility Interfaces and Room Canary

Status: release candidate prepared for annotated tag
`avengine-m6-feasibility-v1`. M6 is closed only if the post-tag verifier and
attestation readback pass.

## Bound implementation

- AVEngine implementation commit A:
  `3fe259f5551778f311717dbbc2ea24b4417c9f1c`
- Habitat-Sim AVEngine fork:
  `e9c81c10834f7e89f33f4e0602c75535a84e054b`
- RLR audio gitlink:
  `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`

## Formal evidence

- Controlled one-active-of-two evidence:
  `tmp/m6/formal_controlled_v1/evidence.json`
  - authoritative verifier: pass (16 checks)
  - active endpoint: `beagle_0_muzzle`
  - retained silent endpoint: `beagle_1_muzzle`
  - `research_only=true`, `qualification_claim=false`,
    `dataset_admission=false`
  - native Habitat and native RLR execution: `not_run`
- Six-case room qualification attempt:
  `tmp/m6/room_qualification_a3_v2/attempt_manifest.json`
  - attempt verifier: pass
  - six report cases over four visual room lineages
  - dataset admissions: 0
  - this pass means bundle/report consistency and fail-closed semantics; it
    does not mean all rooms are qualified
- Commit-A fast unit receipt:
  `tmp/m6/test_receipts_a3_v2/fast-unit.json`
  - 1303 passed, 0 failed, 0 skipped
- Human-review video bundle:
  `tmp/m6/six_case_review_20260719_09/review_manifest.json`
  - post-hoc verifier: pass (8 checks)
  - combined video:
    `tmp/m6/six_case_review_20260719_09/m6_six_case_review.mp4`

## Release-layer status

| Layer | Status | Interpretation |
| --- | --- | --- |
| fast-unit | pass | Structured receipt binds the exact command and 1303 passing tests. |
| slow-hermetic | not_run | No separate release layer was executed. |
| native-habitat | not_run | No new native episode was run on commit A. |
| rlr-audio | not_run | No new native propagation run was run on commit A. |
| blender-assets | not_run | No Blender-dependent build was run on commit A. |
| media-readback | not_run | Media readback passed inside the controlled/review verifiers, but no separate release receipt was created. |
| release-canary | not_run in candidate | It runs only after metadata commit B and the annotated tag exist. |

## Claim boundary

M6 establishes extensible Habitat-native entity/source/sound/room interfaces,
retains the M5.1 source/event/flag authority, materializes one deterministic
two-endpoint retained-evidence canary, and records honest multidimensional room
attempts. It does not claim a complete dataset, measured room materials,
qualified MP3D/ReplicaCAD/Legacy Apartment revisions, arbitrary animal
generalization, or a new native Habitat/RLR run.

ReplicaCAD review clearance remains root-center only rather than full-body
collision. Its acoustic geometry remains stage-surface-only with placeholder
materials and known topology failures. Beagle dry-audio rights remain
unresolved. Topdown remains a QA view rather than a second formal camera.

One superseded pre-candidate fast-unit receipt recorded an environment-sensitive
test expectation failure. The test was corrected to state both clean-formal and
dirty-development outcomes; the candidate receipt above is the clean commit-A
run and passed all 1303 tests.

## Next milestone

The fixed SPEAR `apartment_0000` source-logic canary remains pending until this
candidate's annotated tag and post-tag attestation pass. It will focus on the
fixed-room S0--S5 source scenarios, clean/diagnostic video, listener-aware
Topdown, independent stems and 360-degree binaural output; automatic furnishing
and natural-language QA remain out of scope.
