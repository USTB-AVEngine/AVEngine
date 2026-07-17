# M3 Execution Runbook

M3 has two separate evidence stages: deterministic Acoustic Scene Package
compilation and native RLR material activation. Compiler success never stands
in for native propagation. The commands below are the implemented compile,
native-run and independent-verification interfaces used for the successful
`formal_20260717_01` run and for future replays.

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

## Prerequisites

Run against the Habitat fork commit pinned in `runtime.lock.yaml`, using the
audio-enabled build that contains the M3 `RLRAcousticContext` binding. The
commands below assume:

```bash
export REPO=/data/jzy/code/AVEngine-habitat-native
export RUNTIME=/data/jzy/code/habitat-sim-AVEngine
export HABPY=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export AVENGINE_REPOSITORY_ROOT="$REPO"
export AVENGINE_HABITAT_RUNTIME_ROOT="$RUNTIME"
cd "$REPO"
```

The pinned runtime currently requires `quaternion` to be imported before
`habitat_sim`. The M3 runtime bridge owns and records that workaround; do not
replace it with an unrecorded manual import path.

Before a formal run:

1. build and install the exact audio-enabled Habitat fork;
2. run its focused native and Python acoustic-context tests;
3. finalize `runtime.lock.yaml` with the selected runtime commit and native
   binary hashes, then treat those exact bytes as immutable experiment input
   for the full run;
4. start the M3 compiler/native evidence from a new, nonexistent ignored
   output directory;
5. retain both repositories' clean status in the final evidence record.

Exploratory evidence generated before the final runtime commit or lock update
cannot be relabeled as formal evidence.

## Inputs

The tracked formal request and material inputs are:

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

Run the M3 unit suite before materializing formal evidence:

```bash
"$HABPY" -m pytest -q \
  tests/unit/test_m3_gltf.py \
  tests/unit/test_m3_materials.py \
  tests/unit/test_m3_contracts.py \
  tests/unit/test_m3_compiler.py \
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
export M3_FORMAL="$REPO/tmp/m3/formal_<RUN_ID>"

"$HABPY" -m avengine.cli m3 compile-canary \
  --request "$REPO/examples/m3/blender_custom/canary_request.json" \
  --output "$M3_FORMAL/compile"
```

The output contains self-contained `source_inputs/`, `low_absorption/`,
`high_absorption/` and `compile_evidence.json` records. Compilation must not be
considered complete until independent replay passes:

```bash
"$HABPY" -m avengine.cli m3 verify-compile \
  "$M3_FORMAL/compile/compile_evidence.json"

"$HABPY" -m avengine.cli m3 validate-package \
  "$M3_FORMAL/compile/low_absorption/manifest.json"

"$HABPY" -m avengine.cli m3 validate-package \
  "$M3_FORMAL/compile/high_absorption/manifest.json"
```

The verifier re-extracts the copied source GLB, reapplies the reviewed
transform, recompiles both material databases and compares the resulting
arrays, objects, categories and RLR database. It does not rely only on package
self-hashes.

## 3. Execute and verify the native canary

Run the canary against the independently verified compiler evidence and the
runtime/version manifest pinned by the exact `runtime.lock.yaml` experiment
input:

```bash
"$HABPY" -m avengine.cli m3 run-canary \
  --request "$REPO/examples/m3/blender_custom/canary_request.json" \
  --compile-evidence "$M3_FORMAL/compile/compile_evidence.json" \
  --output "$M3_FORMAL/runtime"

"$HABPY" -m avengine.cli m3 verify-canary \
  "$M3_FORMAL/runtime/canary_evidence.json"
```

`run-canary` exits `0` only for a self-verified `pass`; a verified `fail`
exits `1`, and `blocked` exits `3`. `verify-canary` parses the evidence once
and uses that same immutable byte snapshot for both the declared status and
the verification result. Verification errors exit `2`.

Required native behavior is already fixed:

- alternate low/high conditions until each has three completed runs;
- create a fresh RLR context for every run;
- ingest the complete explicit package and retain exact API receipts;
- write a post-ingestion scene OBJ and raw little-endian float32 IR array per
  run;
- record the exact native configuration readback and native binary hashes;
- run every declared opening/control ray through CPU mesh and RLR queries;
- independently recompute all raw-IR metrics and comparisons during verify.

The final native output and verification result must be written under the same
unique ignored `$M3_FORMAL` root and recorded in
[M3_STATUS.md](M3_STATUS.md). A blocked runtime import/build is `blocked`; it
is not compiler `pass` and cannot complete M3. The completed M3 record used
`$REPO/tmp/m3/formal_20260717_01`.

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

## 6. Run the repository suite and freeze the record

After native verification, run the complete AVEngine test suite from the final
worktree:

```bash
"$HABPY" -m pytest -q
```

Then record, from the retained evidence rather than memory:

- AVEngine and Habitat fork commits;
- runtime lock hash;
- Habitat binding and RLR library hashes;
- compile and native evidence paths/hashes;
- low/high medians, spreads, oriented effects and verifier result;
- clean worktree and final test totals.

After verification, record the formal outcome, measurements and evidence
hashes in [M3_STATUS.md](M3_STATUS.md). Do not write run outcomes or evidence
hashes back into `runtime.lock.yaml`: it remains the immutable experiment input
and runtime/version manifest. The retained formal evidence binds its exact
bytes at SHA-256
`b39f2ffe6e8427852ac802622957186fab972e26f58b4ee4df9ada76bc9023ac`.

## Optional research-only room probes

The generic GLB compiler can propose explicit visual-slot mappings for MP3D or
the legacy UE real-surface room and compile them for diagnostics. The output is
always a `research_candidate` with `qualification_claim: false`:

```bash
"$HABPY" -m avengine.cli m3 propose-visual-slots \
  --room <ROOM_MANIFEST> \
  --transform-profile <identity_y_up_or_mp3d_profile> \
  --runtime-root "$RUNTIME" \
  --output "$REPO/tmp/m3/<ROOM>_proposal_<RUN_ID>"

"$HABPY" -m avengine.cli m3 compile-explicit-research \
  --room <ROOM_MANIFEST> \
  --mapping "$REPO/tmp/m3/<ROOM>_proposal_<RUN_ID>/mapping.json" \
  --materials "$REPO/tmp/m3/<ROOM>_proposal_<RUN_ID>/materials_research.json" \
  --runtime-root "$RUNTIME" \
  --output "$REPO/tmp/m3/<ROOM>_package_<RUN_ID>"
```

These commands identify source slots and geometry problems; they do not infer
physical coefficients, waive failed QA, or admit the room.
