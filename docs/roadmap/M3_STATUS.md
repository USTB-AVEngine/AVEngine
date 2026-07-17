# M3 Status: Explicit Acoustic Scene and Material Activation

Overall status: `not_run` pending the final clean, lock-bound native canary and
independent evidence verification. The compiler and runtime-adapter
implementation described here exists in the M3 feature worktree, but this
draft is not a completion claim. Replace the clearly marked final-record fields
only from the retained formal evidence.

## Gate purpose

M3 replaces the legacy implicit/AABB acoustic path with an explicit,
hash-bound Acoustic Scene Package and proves that its geometry and material
selection reach the reused RLR propagation runtime. M3 does not introduce a
new propagation algorithm: RLR provides geometric acoustic propagation;
AVEngine owns explicit source compilation, strict adapter inputs, QA and
evidence verification.

The gate is deliberately narrower than a physically calibrated room. Its
formal material-effect experiment uses a synthetic `0.02` / `0.60` low/high
absorption contrast in the controlled Blender custom room. A pass would prove material-path
activation and repeatability, not real floor/wall/ceiling coefficient truth.

## Implemented scope awaiting final replay

- Acoustic Scene Package, material mapping, material database, compile
  evidence, canary request and native evidence v1 schemas.
- Strict GLB triangle extraction with explicit source-to-canonical transforms,
  object partitions and per-triangle material IDs.
- Exact material-slot coverage with no random/default/fallback assignment and
  exact RLR category-label resolution.
- Source-input snapshots copied into package provenance and independent
  GLB/mapping/database-to-package replay verification.
- Geometry, degeneracy, duplicate-face, boundary, opening/control-ray,
  material-coverage and compiler-parity QA.
- A modern, RAII RLR context in the Habitat fork with strict configuration,
  explicit object ingestion, owned IR output, ray queries and native scene
  readback.
- Alternating low/high repeats with a fresh context per run, raw IR retention,
  direct-arrival checks and recomputed EDT, DRR and late-energy metrics.
- Tamper-oriented evidence verification that rehashes inputs/artifacts/native
  binaries and recomputes rather than trusting recorded pass booleans.

The exact implementation and verification state must be confirmed by the
commands in [M3_EXECUTION.md](M3_EXECUTION.md) after both repositories are at
their final M3 commits.

## Controlled canary contract

The tracked request is
[`examples/m3/blender_custom/canary_request.json`](../../examples/m3/blender_custom/canary_request.json).
It binds the M1 Blender custom real-surface room, one source, one listener,
four explicit author material slots and three independent repeats per
condition.

The low/high packages must have byte-identical vertices, triangles and
per-triangle material IDs. They must also have identical object partitions and
identical non-absorption material fields. Only `database_id` and the absorption
arrays may differ, with every high value strictly greater than its low value.
The tracked values are `0.02` and `0.60` in all four bands. This keeps the
formal EDT definition at 0 to -10 dB; no direct-sound removal or T10
substitution is used to satisfy the fit-quality gate.

The controlled databases declare:

- `material_semantics: controlled_canary`;
- `intended_use: controlled_material_activation_canary`;
- `qualification_claim: synthetic_activation_test_only` in the compiled
  package.

These declarations are normative. The use of the strict production compiler
route does not convert the synthetic coefficients into reviewed physical
materials and does not grant dataset admission.

## Required evidence closure

M3 may become `pass` only when all of the following independently verify:

1. the room, source GLB, mapping, low/high databases, request, runtime lock,
   AVEngine implementation and native binaries are hash-bound;
2. recompiling the source GLB/mapping/database reproduces the package arrays,
   object partitions, categories and RLR database exactly;
3. material coverage is 100%, fallback triangle count is zero, and production
   geometry is not an AABB proxy;
4. declared door/window clear rays and solid-wall controls agree between the
   CPU package-mesh computation and native RLR queries;
5. exact native API receipts agree on object identities, geometry counts and
   per-category triangle counts;
6. resolved native material blocks agree with the compiled RLR database;
7. post-ingestion OBJ geometry counts and canonical coordinate multisets agree
   with the package;
8. every run meets the direct-arrival and raw-IR validity gates;
9. EDT, DRR and late-energy comparisons have the declared direction and
   minimum effect, and the effect remains larger than within-condition repeat
   spread;
10. the independent verifier returns `pass` from the final retained evidence.

The post-ingestion RLR OBJ does not encode a recoverable per-face material-ID
array. It therefore cannot prove per-triangle assignment alone. That part of
the closure comes from source replay plus exact API receipts and resolved
material blocks; the OBJ is an independent post-ingestion geometry readback.

## Final formal record — pending replay

The following fields are intentionally unresolved in this draft. They must not
be guessed or copied from an exploratory run.

| Record | Final value |
| --- | --- |
| Gate status | `not_run` — replace only after final verification |
| Formal run date | `<M3_FINAL_RUN_DATE_PENDING>` |
| AVEngine commit | `<M3_AVENGINE_COMMIT_PENDING>` |
| Habitat fork commit | `<M3_RUNTIME_COMMIT_PENDING>` |
| Habitat native binding SHA-256 | `<M3_HABITAT_BINDING_SHA256_PENDING>` |
| RLR native library SHA-256 | `<M3_RLR_LIBRARY_SHA256_PENDING>` |
| Compile evidence | `<M3_COMPILE_EVIDENCE_PATH_PENDING>` |
| Native canary evidence | `<M3_NATIVE_EVIDENCE_PATH_PENDING>` |
| Independent verifier | `<M3_VERIFY_RESULT_PENDING>` |

Final metric medians, spreads and effects also remain pending:

| Metric | Low median | High median | Oriented effect | Maximum repeat spread | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| EDT seconds | `<pending>` | `<pending>` | `<pending>` | `<pending>` | `not_run` |
| DRR dB | `<pending>` | `<pending>` | `<pending>` | `<pending>` | `not_run` |
| Late-energy ratio | `<pending>` | `<pending>` | `<pending>` | `<pending>` | `not_run` |

## MP3D and UE research candidates

The official MP3D example and the legacy UE apartment real-surface export have
been used as explicit compiler research probes. Their current mappings are
visual-material-slot proposals with unqualified placeholder coefficients.
They are `research_candidate` diagnostics only:

- neither has a reviewed physical material profile;
- a visual material name is not acoustic ground truth;
- current geometry/transform/slot diagnostics are retained rather than waived;
- package generation does not grant production acoustic-scene or dataset
  admission.

They must not be listed as formal M3 physical-room passes. Their purpose is to
exercise the generic compiler and expose the work required for later reviewed
room admission.

## M3 non-claims and M4 boundary

M3 does not prove perceptual room quality, measured real-room acoustics,
dynamic deformable-body reflection, final episode mixing or dataset admission.

The controlled canary's single named source/listener pair also does not close
M4. Named multi-source/listener enumeration, all-pair IR access, independent
stems, source-order invariance, reset/temporal policy and performance evidence
remain `not_run` until the M4 gate. Any named-source methods present as runtime
groundwork must not be relabeled as M4 completion.
