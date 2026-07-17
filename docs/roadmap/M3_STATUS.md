# M3 Status: Explicit Acoustic Scene and Material Activation

Overall status: `pass` for the bounded M3 controlled material-activation gate.
The retained formal run compiled and independently replayed the explicit
Acoustic Scene Packages, executed six fresh native RLR contexts, and passed all
39 required native-evidence checks. This is a synthetic activation result, not
physical room-material calibration or dataset admission.

## Gate purpose

M3 replaces the legacy implicit/AABB acoustic path with an explicit,
hash-bound Acoustic Scene Package and proves that its geometry and material
selection reach the reused RLR propagation runtime. M3 does not introduce a
new propagation algorithm: RLR provides geometric acoustic propagation;
AVEngine owns explicit source compilation, strict adapter inputs, QA and
evidence verification.

The formal experiment uses a synthetic `0.02` / `0.60` low/high absorption
contrast in the controlled Blender custom room. Its pass proves material-path
activation and exact repeatability under the fixed canary configuration. It
does not establish real floor, wall, ceiling or door-frame coefficients.

## Completed scope

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
- Immutable-snapshot evidence verification that binds inputs, artifacts and
  native binaries and recomputes results instead of trusting recorded pass
  booleans.

## Controlled canary contract

The tracked request is
[`examples/m3/blender_custom/canary_request.json`](../../examples/m3/blender_custom/canary_request.json).
It binds the M1 Blender custom real-surface room, one source, one listener,
four explicit author material slots and three independent repeats per
condition.

The low/high packages have byte-identical vertices, triangles,
per-triangle material IDs, object partitions and non-absorption material
fields. Only `database_id` and absorption differ, and every high-band value is
strictly greater than its low counterpart. The controlled databases declare:

- `material_semantics: controlled_canary`;
- `intended_use: controlled_material_activation_canary`;
- `qualification_claim: synthetic_activation_test_only` in the compiled
  package.

These declarations are normative. The strict production compiler route does
not convert synthetic coefficients into reviewed physical materials and does
not grant dataset admission.

## Formal record

The retained ignored evidence root is
`tmp/m3/formal_20260717_01`. It is a local evidence index, not a tracked release
artifact.

| Record | Formal value |
| --- | --- |
| Gate status | `pass` |
| Formal run date | `2026-07-17` |
| AVEngine implementation commit | `7a952ba6794e249db732855eaa32a7d720dfa60a` |
| Habitat fork commit | `3a60c069514cd4d4987086c872deb0456ab831f1` |
| Runtime-lock experiment-input SHA-256 | `b39f2ffe6e8427852ac802622957186fab972e26f58b4ee4df9ada76bc9023ac` |
| Habitat native binding SHA-256 | `944f23e78af277301563874788020c4fe0bd993e93aa6dcd5516f237bbda196c` |
| RLR native library SHA-256 | `31e948eef4908d8cbb403b5f445d9d0eab59fc81b05a658538f8795984f9bfb4` |
| Compile evidence | `tmp/m3/formal_20260717_01/compile/compile_evidence.json` |
| Compile evidence file SHA-256 | `3e1f3394bd86b2e0e31fc0720861bf162be80ada891bd77bd7a1fb625572af3f` |
| Compile evidence content SHA-256 | `8c465b70d3dd95db38dc5eaa23b741b72b76b99baf6f905a313f601ad240b7ea` |
| Native evidence | `tmp/m3/formal_20260717_01/runtime/canary_evidence.json` |
| Native evidence file SHA-256 | `512039e604be98877b9a09cbe2b8e7cc2c8602a29baa16c194733e0ddb67afce` |
| Native evidence content SHA-256 | `0e05110949fdd01032ac6b07631b7cf1f4fc484deedc23b8a8df5fd2bee10c5d` |
| Independent verifier | `pass`; 39/39 required native checks, including 11/11 nested compile-replay checks |
| Final repository regression suite | `870 passed in 104.36s (0:01:44)` |

`runtime.lock.yaml` is the immutable experiment input and runtime/version
manifest for this formal run. The evidence binds its exact bytes at the
SHA-256 above. Current milestone outcomes are recorded in this document and
[MILESTONES.md](MILESTONES.md), not written back into the lock.

## Formal measurements

All three repeats within each condition were byte/metric-identical, so the
maximum absolute and relative within-condition spreads were zero. A zero
denominator makes an effect-to-spread numeric ratio undefined; the verifier
correctly records it as `null` while passing the stricter non-zero-effect over
zero-spread check.

| Metric | Low median | High median | Oriented effect | Maximum repeat spread | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| EDT seconds (`low - high`) | `2.468688935540196` | `0.11696154395822458` | `2.3517273915819716` | `0` | `pass` |
| DRR dB (`high - low`) | `-15.358793693377324` | `-1.1820933529140096` | `14.176700340463315` | `0` | `pass` |
| Late-energy ratio (`low - high`) | `0.6036383780195904` | `0.00010379869369406927` | `0.6035345793258964` | `0` | `pass` |

The late-energy low/high effect ratio was `5815.471818929841`, above the
required `1.2`. Every run detected direct arrival at sample `96`, versus an
expected `96.19400875709053`, for an absolute error of
`0.19400875709052912` samples under the two-sample threshold. Low/high EDT fit
R² values were `0.9990420854316167` and `0.9552649256523217`, both above
`0.90`, with decay spans above 10 dB.

## Geometry, material and native-ingestion verification

The formal package and every runtime repeat agreed on 12 objects, 288 expanded
vertices, 144 triangles and four material categories. Coverage was `1.0` and
fallback triangle count was `0`. All four declared door/window/control rays
passed CPU-mesh versus native-RLR hit and first-hit-distance checks in all six
runs. Exact upload receipts, resolved coefficient blocks and source replay
verified material identity; native OBJ readback independently verified
geometry.

The post-ingestion RLR OBJ does not encode a recoverable per-face material-ID
array. It therefore cannot prove per-triangle assignment alone. That part of
the verification comes from source replay plus exact API receipts and resolved
material blocks.

## MP3D and UE research candidates

The official MP3D example and legacy UE apartment real-surface export remain
`research_candidate` diagnostics only. Their visual material-slot proposals
and unqualified placeholder coefficients are useful compiler probes, but no
visual material name is acoustic ground truth. Neither room has a reviewed
physical material profile or acoustic-scene/dataset admission, and successful
package generation does not grant either one.

## M3 non-claims and M4 boundary

M3 does not prove perceptual room quality, measured real-room acoustics,
dynamic deformable-body reflection, final episode mixing or dataset admission.

The controlled canary's single named source/listener pair also does not complete
M4. Named multi-source/listener enumeration, all-pair IR access, independent
stems, source-order invariance, reset/temporal policy and performance evidence
remain `not_run` until the M4 gate. Any named-source methods present as runtime
groundwork must not be relabeled as M4 completion.
