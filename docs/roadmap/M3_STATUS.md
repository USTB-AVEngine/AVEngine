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

## Post-gate semantic material extension

Status: `research_candidate`; implemented and exercised on the MP3D
`17DRP5sb8fy` sample. This does not alter the formal controlled-canary `pass`.

- Strictly parses MP3D binary semantic PLY faces and resolves `object_id`
  through the paired `.house` object/category records.
- Uses one editable residential rules file with deterministic plausible
  candidate selection, bounded absorption/scattering jitter, name/material-slot
  hints and exact room/object overrides.
- Exposes the same generic surface identity fields for future ReplicaCAD and
  SPEAR/UE adapters; it emits the existing M3 mapping/database and RLR package,
  not a second material format.
- Reports unknown categories separately and retains all resolved assignments
  as unqualified research placeholders.
- Adds deterministic interior spherical-ray diagnostics alongside exact-weld
  topology QA. Escaped rays are reported for review rather than automatically
  classified as invalid openings or silently patched.

The retained local sample is
`tmp/m3/mp3d_semantic_soundspaces_20260725_02`. It compiled 3,016,249 semantic
triangles into 31 used categories. The only default-resolved label was
`unknown_object` (8,118 triangles). The independent package validator passed.
Topology remained `fail` with 31,525 global boundary edges, as expected for a
scanned semantic mesh; the two-point, 16-direction-per-point enclosure probe
observed 0/32 escaped rays. Sparse probe success does not erase the topology
holes or establish physical closure. The complete baseline compile took
667.13 seconds and peaked at approximately 1.14 GB; it is a once-per-room
preprocessing cost, not a per-episode or per-audio cost.

### Cross-room mesh leakage diagnostics

The generic `inspect-mesh-leakage` command was subsequently run against two
older real-room acoustic packages without modifying them:

| Room/package | Interior probes × directions | Escaped rays | Topology context | CPU reference time |
| --- | ---: | ---: | --- | ---: |
| MP3D `17DRP5sb8fy` semantic package | 2 × 16 | 0 / 32 | 31,525 boundary edges | included in the 667.13 s compile |
| ReplicaCAD `apt_0` acoustic proxy | 3 × 64 | 50 / 192 (26.04%) | 102 boundary, 8,097 nonmanifold, 10,629 duplicate faces | 5.26 s |
| legacy SPEAR Apartment `apartment_0000` | 4 × 64 | 0 / 256 | 20,734 boundary, 7 nonmanifold, 1 duplicate face | 1,127.52 s |

Every ReplicaCAD escape was in the upper spherical sector. This is consistent
with missing or incomplete overhead enclosure in that acoustic proxy and is
not a small isolated scan-hole result. Apartment's zero escaped rays mean the
sampled directions from four reviewed NavMesh/LOS points all reached a
surface; they do not override its 20,734 topology boundary edges or prove
global closure.

The retained reports are
`tmp/m3/leakage_diagnostics_20260725_01/replicacad_apt_0.json` and
`tmp/m3/leakage_diagnostics_20260725_01/legacy_ue_apartment_0000.json`.
Kujiale was not assigned a false result: its retained episodes use
`generic_shoebox_directional_preview` with
`material_claim: not_kujiale_material_truth`, and no Acoustic Scene Package
exists for the actual Kujiale visual mesh. A real Kujiale diagnostic therefore
requires compiling that USD/UE geometry into the acoustic package first.

The CPU Möller–Trumbore reference checks every triangle for every ray. It is
appropriate as an auditable baseline and for small meshes, but the Apartment
measurement shows that dense multi-room use requires a reusable BVH or the
RLR/Habitat acceleration structure. This performance limitation does not
change the reported intersections.

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
| Historical runtime profile | [`locks/m3_runtime_v1.yaml`](../../locks/m3_runtime_v1.yaml) via the root index |
| Native artifact authority | `runtime/canary_evidence.json` native-binary bundle record |
| Compile evidence | `tmp/m3/formal_20260717_01/compile/compile_evidence.json` |
| Native evidence | `tmp/m3/formal_20260717_01/runtime/canary_evidence.json` |
| Independent verifier | `pass`; 39/39 required native checks, including 11/11 nested compile-replay checks |
| M3 formal-run regression suite at commit `7a952ba6794e249db732855eaa32a7d720dfa60a` | `870 passed in 104.36s (0:01:44)` |

The indexed M3 profile preserves the immutable experiment input and runtime
versions for this formal run. Its machine-readable evidence owns the exact
external binary and bundle identities; this page does not duplicate them.
Current milestone outcomes are recorded here and in
[MILESTONES.md](MILESTONES.md), not written back into a lock.

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

## Post-gate M3.1 extension

M3.1 adds an explicit user-control layer without changing the retained M3
canary inputs, evidence, measurements or formal hashes. The implemented
`resolve-materials` CLI starts from an existing mapping and base database,
applies global then exact per-material overrides, broadcasts scalar curves to
the declared frequency bands, and writes a complete effective mapping,
database and resolution report. Unknown selectors, shared-key source
selectors, duplicate or conflicting resolutions, and wrong-band arrays fail
closed; no hidden fallback is emitted into the resolved database or package.

The separate `calibrate_broadband_edt_seconds` core performs a bounded search
over uniform absorption using caller-reported EDT values. A successful result
records target, achieved value, error, repeat spread and every evaluated point.
It explicitly targets broadband EDT. End-to-end native target-decay
calibration has not yet produced formal evidence and remains `not_run`; no
RT60, frequency-band or physical-material calibration claim is made.

### M3.1 verification record

The exact implementation bytes tested below were committed as
`95507ead8f85c90b50add4e67104c23219de7b4d` on `2026-07-17`.

```bash
"$HABPY" -m pytest -q \
  tests/unit/test_m3_calibration.py \
  tests/unit/test_m3_materials.py \
  tests/unit/test_m3_cli.py \
  tests/unit/test_m3_contracts.py
# 61 passed in 1.16s

"$HABPY" -m pytest -q
# 916 passed in 98.50s (0:01:38)
```

The exact nine-file M3 command in [M3_EXECUTION.md](M3_EXECUTION.md) passed
`119` tests in `14.73s`.

The tracked example profile also passed the executable
`resolve-materials -> compile-custom -> validate-package` sequence. The
retained M3 compile evidence and native material-activation canary were
independently re-verified as `pass` from the final implementation worktree.
Those checks are regression verification of the existing M3 record; they are
not new target-decay evidence. No target-decay RIR was generated or admitted,
so native target-decay calibration remains `not_run`.

## M3 non-claims and M4 boundary

M3 does not prove perceptual room quality, measured real-room acoustics,
dynamic deformable-body reflection, final episode mixing or dataset admission.

The controlled canary's single named source/listener pair also does not complete
M4. Named multi-source/listener enumeration, all-pair IR access, independent
stems, source-order invariance, reset/temporal policy and performance evidence
remain `not_run` until the M4 gate. Any named-source methods present as runtime
groundwork must not be relabeled as M4 completion.
