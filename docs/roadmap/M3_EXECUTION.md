# M3 Execution Runbook

M3 has two separate evidence stages: deterministic Acoustic Scene Package
compilation and native RLR material activation. Compiler success never stands
in for native propagation. The historical v1 record is retained only for schema
dispatch and independent reader replay of its existing evidence; this runbook
does not authorize or describe a new v1 native execution. Every executable
native command below uses the current-installed v2 prefix, SDK and Magnum site.

## Fixed contract

- The formal fixture is the M1 Blender custom real-surface room, not an AABB.
- The package coordinate convention is right-handed, +Y up, -Z forward and
  metres.
- Every used source material slot maps exactly once to one acoustic category;
  every triangle has a material ID; fallback and random assignment are false.
- The low/high packages freeze geometry, object partitions, material IDs and
  all material fields except `database_id` and absorption.
- The tracked low/high absorption values are `0.02` / `0.60` at every
  material and band; the high value is strictly greater everywhere.
- Each condition runs three times with a fresh RLR context, one native thread,
  mesh simplification off and temporal coherence off.
- The canary's one source/listener pair exists only to prove M3 material
  activation. It is not M4 multi-source/stem/order-invariance evidence.

The high/low values are a deliberate synthetic contrast. The `0.60` high
condition was selected because it retains a genuine 0 to -10 dB Schroeder EDT
fit with the fixed quality gate. The rejected exploratory `0.90` condition
made direct sound dominate that interval; removing direct sound would change
the metric into a T10-like tail estimate and is not an allowed repair. A pass
establishes runtime activation, not reviewed physical acoustic truth for the
room.

## Current execution setup

Use the current AVEngine checkout for compiler, verifier and v2 execution:

```bash
export REPO=/data/jzy/code/AVEngine-lead-a
export HABPY=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export PYTHONPATH="$REPO/src"
cd "$REPO"
```

Do not set `AVENGINE_HABITAT_RUNTIME_ROOT` or any v2 runtime argument to an
old Habitat checkout. The v2 CLI requires explicit prefix, SDK and Magnum
arguments on every native execution.

## Archived v1 evidence (verification only)

[`locks/m3_runtime_v1.yaml`](../../locks/m3_runtime_v1.yaml) and the
`formal_20260717_01` record remain available solely so the v1 schema and
reader can validate an already retained bundle. They are not an executable
runtime profile. Do not build, import, `cd` into, or run a native canary from
the historical checkout.

To inspect existing v1 evidence, point the current verifier at its retained
bundle; this reads confined evidence and the v1 lock but does not start the old
native runtime:

```bash
export M3_ARCHIVED_V1_EVIDENCE=/path/to/retained/m3_v1/canary_evidence.json
"$HABPY" -m avengine.cli m3 verify-canary "$M3_ARCHIVED_V1_EVIDENCE"
```

## Inputs

The tracked canary request and material inputs are:

```text
examples/m3/blender_custom/canary_request.json
examples/m3/blender_custom/mapping.json
examples/m3/blender_custom/materials_low.json
examples/m3/blender_custom/materials_high.json
```

The request resolves the tracked M1 room manifest and its source GLB. The
compiler copies the request, room manifest, mapping, both databases and source
geometry into the evidence tree, then binds their byte hashes.

## 1. Run focused contract tests

Run the M3 unit suite before materializing current compiler evidence:

```bash
"$HABPY" -m pytest -q \
  tests/unit/test_m3_gltf.py \
  tests/unit/test_m3_materials.py \
  tests/unit/test_m3_contracts.py \
  tests/unit/test_m3_compiler.py \
  tests/unit/test_m3_calibration.py \
  tests/unit/test_m3_metrics.py \
  tests/unit/test_m3_runtime.py \
  tests/unit/test_m3_canary.py \
  tests/unit/test_m3_cli.py
```

This is implementation coverage, not a substitute for the native canary.

## 2. Compile the controlled counterfactual

Choose a new output path. Compilation is exclusive and must reject an existing
destination rather than merge with stale artifacts.

```bash
export M3_COMPILE="$REPO/tmp/m3/compile_<RUN_ID>"

"$HABPY" -m avengine.cli m3 compile-canary \
  --request "$REPO/examples/m3/blender_custom/canary_request.json" \
  --output "$M3_COMPILE"
```

The output contains self-contained `source_inputs/`, `low_absorption/`,
`high_absorption/` and `compile_evidence.json` records. Compilation must not be
considered complete until independent replay passes:

```bash
"$HABPY" -m avengine.cli m3 verify-compile \
  "$M3_COMPILE/compile_evidence.json"

"$HABPY" -m avengine.cli m3 validate-package \
  "$M3_COMPILE/low_absorption/manifest.json"

"$HABPY" -m avengine.cli m3 validate-package \
  "$M3_COMPILE/high_absorption/manifest.json"
```

The verifier re-extracts the copied source GLB, reapplies the reviewed
transform, recompiles both material databases and compares the resulting
arrays, objects, categories and RLR database. It does not rely only on package
self-hashes.

## 3. Execute and verify the current-installed v2 canary

This is the only executable native M3 route. It uses a fresh non-checkout
Habitat installation, external RLR SDK and external Magnum Python site; the
three paths are explicit CLI arguments, not environment fallbacks.

```bash
export M3_CURRENT="$REPO/tmp/m3/current_installed_<RUN_ID>"

"$HABPY" -m avengine.cli m3 run-canary \
  --runtime-mode current-installed \
  --runtime-prefix /external/installed-habitat \
  --rlr-sdk-root /external/RLRAudioPropagationPkg \
  --magnum-python-site /external/magnum-python-site \
  --request "$REPO/examples/m3/blender_custom/canary_request.json" \
  --compile-evidence "$M3_COMPILE/compile_evidence.json" \
  --output "$M3_CURRENT"

"$HABPY" -m avengine.cli m3 verify-canary \
  "$M3_CURRENT/canary_evidence.json"
```

All three runtime paths must resolve to accessible components outside every Git
checkout: the Habitat module/binding must be contained by the prefix, and the
RLR header/library by the SDK root. The v2 path does not read the historical M3
lock and does not write a tracked binary hash, baseline or lock. It records
only one fresh-run identity repeated across native calls. Its verifier replays
the retained input closure, artifacts, IR/OBJ readback, metrics, rays and
comparisons; recomputing a top-level JSON hash cannot bless altered evidence.
The installed Habitat binding is built without a build or install RPATH to an
RLR SDK. At runtime the current-installed loader first removes editable
Habitat finders, validates any preloaded Habitat module origins, and activates
the selected prefix and Magnum site without importing the native binding. It
then preloads the exact absolute SDK library, imports the prepared Habitat
binding, revalidates module/binding origins, and requires the process mappings
to contain only that declared RLR library. Consequently `--rlr-sdk-root` is the
actual per-process SDK selection rather than documentation for a path already
embedded in the extension.

`run-canary` exits `0` only for a self-verified `pass`; a verified `fail`
exits `1`, and `blocked` exits `3`. `verify-canary` uses one immutable
evidence snapshot; verification errors exit `2`.

A v2 current-installed receipt is diagnostic only. It does not replace the
retained v1 formal record, admit a production cache, or unblock existing
historical-root consumers. The v1 schema/lock reader remains available only for
the archived evidence route above.

## 4. Inspect ingestion evidence correctly

The native scene OBJ is not a per-face material-ID serialization. It is valid
evidence for post-ingestion geometry coordinates/counts and resolved material
blocks only. Formal verification must combine:

- source-to-package replay of `triangle_material_ids`;
- exact upload object IDs/counts and triangle counts by material category;
- resolved native material blocks checked against the compiled database;
- independently parsed post-ingestion OBJ geometry multisets and counts.

Do not infer a face-to-material array from OBJ face order or vertex color.

## 5. Check metric and repeatability gates

For each retained run, verification must reread the raw IR and recompute:

- owned array dtype, shape, finiteness and content hash;
- direct-arrival sample versus source/listener distance and 343 m/s;
- EDT over the declared decay interval, fit R² and decay span;
- direct-to-reverberant ratio;
- late-energy ratio after the fixed late window.

For all three metrics, the high/low direction and minimum absolute effect must
pass. The maximum within-condition relative spread must remain within the
request threshold and the oriented effect must be at least the declared
multiple of that spread. The late-energy ratio additionally has its declared
minimum effect ratio.

Thresholds come from the tracked request/schema. They must not be weakened in
response to an exploratory result and then presented as the same formal gate.

## 6. Run the repository suite and retain the current diagnostic

After v2 verification, run the complete AVEngine test suite from the current
worktree:

```bash
"$HABPY" -m pytest -q
```

Retain the fresh v2 evidence path, current identity, compile lineage and test
totals with that ignored output. Do not use a v2 result to rewrite the retained
v1 formal outcome, runtime hashes or [M3_STATUS.md](M3_STATUS.md). Keep exact
external identities inside the machine-readable v2 bundle; do not copy leaf
hashes or outcome claims into a root index or status prose.

## Resolve an M3.1 user material profile

This post-gate command does not alter or replace the formal M3 low/high record.
It materializes convenience controls into the same explicit mapping/database
contract used by the compiler:

```bash
"$HABPY" -m avengine.cli m3 resolve-materials \
  --mapping "$REPO/examples/m3/blender_custom/mapping.json" \
  --base-materials "$REPO/examples/m3/blender_custom/materials_low.json" \
  --profile "$REPO/examples/m3/blender_custom/material_profile_example.json" \
  --output "$REPO/tmp/m3/user_profile_<RUN_ID>"

"$HABPY" -m avengine.cli m3 compile-custom \
  --room "$REPO/examples/m1/rooms/blender_custom/room_manifest.json" \
  --mapping "$REPO/tmp/m3/user_profile_<RUN_ID>/mapping.json" \
  --materials "$REPO/tmp/m3/user_profile_<RUN_ID>/materials.json" \
  --output "$REPO/tmp/m3/user_profile_package_<RUN_ID>"

"$HABPY" -m avengine.cli m3 validate-package \
  "$REPO/tmp/m3/user_profile_package_<RUN_ID>/manifest.json"
```

Inspect `resolution_report.json` for the exact selector resolutions,
field-level precedence and input/output hashes. Scalar curve values are
broadcast to every base-database band; explicit arrays must match the band
count. These values are user controls, not physical measurements.

The bounded target helper currently exposes a Python callback API for
caller-reported broadband EDT values. Its result must never be labeled RT60.
Native target-decay orchestration and evidence are `not_run`; any future RT60
estimator would separately require retained RIRs, declared anchors and solver
configuration, sufficient decay span/fit quality and a tolerance-bound result.

## Optional research-only room probes

The generic GLB compiler can propose explicit visual-slot mappings for MP3D or
the legacy UE real-surface room and compile them for diagnostics. The output is
always a `research_candidate` with `qualification_claim: false`.

These compiler-only commands never import Habitat or create an RLR context,
so they do not accept a runtime prefix or Magnum site merely as decorative
arguments. `--runtime-root` is retained only as a rejected compatibility
spelling: it cannot select a checkout or an asset root. For a current MP3D
room, pass the explicit, canonical, non-Git `--mp3d-root`; for a relative GLB
or external USD snapshot, omit it and provide the room's separately declared
external asset inputs instead. Native RLR execution remains the distinct v2
`run-canary` path with prefix, SDK and Magnum arguments.

```bash
"$HABPY" -m avengine.cli m3 propose-visual-slots \
  --room <ROOM_MANIFEST> \
  --transform-profile <identity_y_up_or_mp3d_profile> \
  --mp3d-root <NON_GIT_MP3D_ROOT> \
  --output "$REPO/tmp/m3/<ROOM>_proposal_<RUN_ID>"

"$HABPY" -m avengine.cli m3 compile-explicit-research \
  --room <ROOM_MANIFEST> \
  --mapping "$REPO/tmp/m3/<ROOM>_proposal_<RUN_ID>/mapping.json" \
  --materials "$REPO/tmp/m3/<ROOM>_proposal_<RUN_ID>/materials_research.json" \
  --mp3d-root <NON_GIT_MP3D_ROOT> \
  --output "$REPO/tmp/m3/<ROOM>_package_<RUN_ID>"
```

These commands identify source slots and geometry problems; they do not infer
physical coefficients, waive failed QA, or admit the room.

## Compile MP3D semantic materials

This research command uses the semantic PLY and `.house` assets already
declared by the room manifest. It selects only plausible category candidates
from one editable rule file and compiles the result into the same M3/RLR
package:

```bash
"$HABPY" -m avengine.cli m3 compile-mp3d-semantic \
  --room "$REPO/examples/m1/rooms/habitat_mp3d_example/room_manifest.json" \
  --rules "$REPO/examples/m3/semantic_materials/residential_material_rules.json" \
  --seed 917 \
  --mp3d-root <NON_GIT_MP3D_ROOT> \
  --probe-directions 32 \
  --output "$REPO/tmp/m3/mp3d_semantic_<RUN_ID>"

"$HABPY" -m avengine.cli m3 validate-package \
  "$REPO/tmp/m3/mp3d_semantic_<RUN_ID>/manifest.json"
```

If `--probe-origin X Y Z` is omitted, up to two canonical points are taken
from the room's declared connectivity anchors. Repeat the option to use
reviewed listener/navigation points. Inspect:

- `semantic_material_coverage.json` for every category decision and unknown
  label;
- `qa/geometry_report.json` for exact-weld boundary/nonmanifold topology;
- `qa/ray_leakage.json` for automatic escaped-ray directions and declared
  opening/control rays.

The automatic enclosure probe is diagnostic. A topology boundary or escaped
ray is not silently filled, while a no-escape sparse probe does not prove that
the complete scan mesh is closed. Material coefficients retain
`research_placeholder` semantics until a separate real-RIR or measurement
calibration is completed.

## Compile a composed external USD room

Pixar USD is an optional authoring dependency, not a Habitat runtime
dependency. Run the extractor in an environment that provides `pxr`; it opens
the composed stage, follows references, keeps authored-visible
`/Root/Meshes` prims, bakes world transforms and writes one NPZ snapshot. It
does not alter the source stage or repair holes:

```bash
export USD_PYTHON=/path/to/python-with-pxr
export USD_STAGE=/path/to/kujiale_0020_full_home_ue.usda
export USD_SNAPSHOT="$REPO/tmp/m3/kujiale_0020_usd_snapshot_<RUN_ID>"

"$USD_PYTHON" "$REPO/tools/m3/extract_usd_acoustic_snapshot.py" \
  --source "$USD_STAGE" \
  --output "$USD_SNAPSHOT" \
  --room-id kujiale_0020_full_home_v1 \
  --transform-profile kujiale_z_up_y_back_to_habitat \
  --interior-origin X0 Y0 Z0 \
  --interior-origin X1 Y1 Z1 \
  --source-revision "<reviewed source revision>" \
  --dataset-id "spatialverse/InteriorAgent kujiale_0020" \
  --source-license "<reviewed dataset license>"
```

Use actual free-space listener/source points, not an arbitrary visual camera
position. Then return to the normal Habitat environment and compile the
snapshot through the same semantic rules and Acoustic Scene Package:

```bash
"$HABPY" -m avengine.cli m3 compile-usd-snapshot-semantic \
  --room "$USD_SNAPSHOT/room_manifest.json" \
  --rules "$REPO/examples/m3/semantic_materials/residential_material_rules.json" \
  --seed 917 \
  --probe-directions 16 \
  --output "$REPO/tmp/m3/kujiale_0020_usd_semantic_<RUN_ID>"

"$HABPY" -m avengine.cli m3 validate-package \
  "$REPO/tmp/m3/kujiale_0020_usd_semantic_<RUN_ID>/manifest.json"
```

Inspect `extraction_report.json`, `semantic_material_coverage.json`,
`qa/geometry_report.json` and `qa/ray_leakage.json`. The automatic report
marks a probe invalid if any sampled surface is closer than 5 cm; a
zero-escape result from such a point is not usable enclosure evidence. A
structurally valid package may still contain a failing topology report, and
neither package validation nor sparse rays establish physical material truth.

## Inspect an existing acoustic package for mesh leakage

Use reviewed canonical interior points from a camera/listener, NavMesh route or
validated source anchor. This command validates the existing package first and
writes a separate report; it never edits or recompiles the package:

```bash
"$HABPY" -m avengine.cli m3 inspect-mesh-leakage \
  --package "$REPO/tmp/m3/<PACKAGE>/manifest.json" \
  --origin X0 Y0 Z0 \
  --origin X1 Y1 Z1 \
  --directions 64 \
  --output "$REPO/tmp/m3/leakage_diagnostics_<RUN_ID>/<ROOM>.json"
```

The report includes per-origin escaped direction indices, escape fraction,
first-hit distances and the package's existing boundary/nonmanifold topology
context. A zero escape fraction is scoped only to the supplied origins and
directions. A large escape fraction concentrated above the probes commonly
indicates a missing ceiling, while horizontal escapes may be doors, windows,
scan holes or an intentionally open scene boundary.

The current backend is the auditable CPU reference and scales as
`probe_count × direction_count × triangle_count`. The measured 782,306-face
Apartment package required 1,127.52 seconds for four origins × 64 directions.
Use this command for bounded QA; batch room qualification should add a reusable
BVH or the pinned RLR/Habitat ray accelerator before increasing coverage.
