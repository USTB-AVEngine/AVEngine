# M4 Execution Runbook

M4 is the executable named multi-source spatial-audio gate. It consumes one
independently verified M3 Acoustic Scene Package, realizes at least two stable
source IDs and exactly one formal camera-co-located listener, and retains
per-pair IRs, independent stems, canary mixtures and native lifecycle evidence.
The commands below are templates; the authoritative retained result is recorded
in [M4_STATUS.md](M4_STATUS.md).

## Fixed boundary

- The MVP has at least two named sources and exactly one listener.
- Source IDs are portable ASCII IDs, bytewise-canonical and independent of the
  caller's list order.
- The listener transform equals the formal M1 camera-rig/listener transform.
- Each output layout uses a fresh RLR context and returns every
  `(listener_id, source_id)` IR as an owned array.
- Native registration receipts must reproduce endpoint IDs, canonical native
  indices, position, radius, listener orientation, output layout, channel count,
  HRTF path and `native_realized` state.
- Raw RLR FOA is `[W, Y, Z, X]`, ACN indices `[0, 1, 2, 3]`, N3D and
  right-handed `avengine_world` (+X right, +Y up, +Z back, -Z forward).
- Native binaural is `[left, right]` and uses the explicit MIT KEMAR
  normal-pinna SOFA asset and its retained license evidence.
- Rendering is 16 kHz. The pinned HRTF input is 44.1 kHz; any adaptation occurs
  only inside the exact RLR binary bound by the runtime lock. AVEngine does not
  resample, normalize, limit or crop these M4 canary signals.
- Per-source stems are full linear convolutions. Mixtures use canonical source
  order and retain the complete tail.
- M4 emits WAV and raw-array evidence only. It does not mux a video.

The checked-in identity fixture binds source identity to formal M1 static source
poses. M2 event-time dynamic-anchor evidence is explicitly `not_run`; this
runbook cannot promote it into animal-asset or dataset admission.

## Prerequisites

Use the final audio-enabled M4 Habitat fork build and the dedicated M4 runtime
lock. The environment below mirrors the local development layout; replace paths
without changing the tracked inputs:

```bash
export REPO=/data/jzy/code/AVEngine-habitat-native
export RUNTIME=/data/jzy/code/habitat-sim-AVEngine
export HABPY=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
```

The runtime bridge owns the required `quaternion`-before-`habitat_sim` import
order. Do not replace it with an unrecorded import workaround.

Before producing formal evidence:

1. build/install the exact `feature/m4-multisource-rlr` Habitat fork with audio
   enabled;
2. run the fork's focused C++ and Python acoustic-context tests;
3. ensure [`locks/m4_runtime_v1.json`](../../locks/m4_runtime_v1.json) names the
   exact installed binding, RLR library and MIT KEMAR dependency bytes;
4. independently verify the M3 package selected for this run;
5. use a new ignored M4 output directory that does not already exist.

A runtime lock is an immutable experiment input. Do not write canary outcomes
back into it or silently update it after a run.

## Tracked M4 inputs

```text
examples/m4/blender_custom/multi_source_canary_request.json
examples/m4/blender_custom/source_identity_manifest.json
locks/m4_runtime_v1.json
```

The request also closes its referenced formal M1 capture request and M3
acoustic canary request by path, size and SHA-256. The runtime copies the
selected M3 Acoustic Scene Package, request graph, lock, HRTF and license
evidence into the private evidence tree before using them.

## 1. Validate contracts and run focused tests

```bash
"$HABPY" -m avengine.cli m4 validate-request \
  "$REPO/examples/m4/blender_custom/multi_source_canary_request.json"

"$HABPY" -m pytest -q \
  tests/test_m4_audio.py \
  tests/test_m4_binaural.py \
  tests/test_m4_contracts.py \
  tests/test_m4_evidence_hardening.py \
  tests/test_m4_spatial.py \
  tests/unit/test_m4_cli.py \
  tests/unit/test_m4_runtime.py
```

These tests cover contracts, deterministic audio arithmetic, strict
IEEE-float WAV readback, spatial probes, native receipt rejection and evidence
tamper cases. They do not replace a real RLR canary.

In the Habitat fork, run the configured build's focused
`RLRAcousticContextTest` and:

```bash
cd "$RUNTIME"
"$HABPY" -m pytest -q tests/test_avengine_acoustic_scene.py
cd "$REPO"
```

Record the exact build/test commands and results in `M4_STATUS.md`; do not infer
them from source presence.

## 2. Verify the selected M3 package

Point `M3_PACKAGE` at an already compiled and independently verified explicit
Acoustic Scene Package. The retained M3 low-absorption package is suitable for
the bounded software canary; its synthetic coefficients are not physical room
truth.

```bash
export M3_PACKAGE="$REPO/tmp/m3/formal_20260717_01/compile/low_absorption/manifest.json"

"$HABPY" -m avengine.cli m3 validate-package "$M3_PACKAGE"
"$HABPY" -m avengine.cli m3 verify-compile \
  "$REPO/tmp/m3/formal_20260717_01/compile/compile_evidence.json"
```

Successful package validation alone does not complete M4; it only establishes
the M3 input boundary.

## 3. Run the native M4 canary

Choose a new output root. `run-canary` must refuse to merge into an existing
directory.

```bash
export M4_FORMAL="$REPO/tmp/m4/formal_<RUN_ID>"

"$HABPY" -m avengine.cli m4 run-canary \
  --request "$REPO/examples/m4/blender_custom/multi_source_canary_request.json" \
  --package-manifest "$M3_PACKAGE" \
  --runtime-lock "$REPO/locks/m4_runtime_v1.json" \
  --hrtf /usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa \
  --hrtf-license /usr/share/doc/libmysofa1/copyright \
  --output "$M4_FORMAL"
```

The run must execute, rather than merely declare:

- full-indirect FOA renders for canonical and reversed caller source order;
- explicit-HRTF native binaural all-pair rendering;
- independent dry signals, FOA/binaural stems and canonical mixtures;
- six-cardinal direct-only FOA and listener-rotation probes;
- direct-only horizontal binaural probes;
- stable-ID source update, temporal-coherence sequence and reset/reload replay;
- fresh-context one-source versus multi-source performance measurement.

Any missing native API, runtime/HRTF lock mismatch, incomplete pair set,
malformed receipt or failed spatial/lifecycle check prevents publication of a
passing destination.

## 4. Independently verify retained evidence

```bash
"$HABPY" -m avengine.cli m4 verify-canary \
  "$M4_FORMAL/m4_canary_evidence.json"
```

Verification must reread confined artifacts and independently check their
sizes/hashes, request and identity binding, exact order equality, direct-arrival
geometry, dry/IR/stem/mix reconstruction, FOA/binaural probes, runtime/HRTF
pins, lifecycle arrays and performance receipts. Rewriting a declared status or
recomputing only the top-level JSON hash must not turn tampered evidence into a
pass.

The evidence is published atomically only after this self-verification passes.
A compiler/unit-test pass cannot substitute for native evidence.

## 5. Inspect the retained audio correctly

The formal tree contains paths of this form:

```text
audio/dry/<source_id>.wav
audio/foa_stems/<source_id>.wav
audio/binaural_stems/<source_id>.wav
audio/mixtures/canary_foa_mix.wav
audio/mixtures/canary_binaural_mix.wav
raw_ir/foa_order_a/<source_id>.npy
raw_ir/foa_order_b/<source_id>.npy
raw_ir/binaural/<source_id>.npy
probes/foa/*.npy
probes/binaural/*.npy
lifecycle/{fresh_first,updated,reset_first}/<source_id>.npy
```

Every WAV has a same-directory authenticated JSON sidecar. The four-channel FOA
WAV is the authority spatial representation and should not be interpreted as
ordinary four-speaker PCM. The two-channel binaural mixture is the direct
headphone-review artifact. Neither file is an M5 final episode: both retain the
full convolution tail and are not video-muxed.

## 6. Freeze the formal record

After native verification, run the complete AVEngine suite:

```bash
"$HABPY" -m pytest -q
```

Then update [M4_STATUS.md](M4_STATUS.md), the README milestone table and
[MILESTONES.md](MILESTONES.md) using values read from retained evidence:

- gate status and formal run date;
- AVEngine and Habitat fork commits;
- runtime-lock, binding, RLR, HRTF and license hashes;
- evidence path, file hash and canonical content hash;
- independent verifier result and test totals;
- measured lifecycle/performance summaries.

Do not record guessed values. A bounded `pass` applies only to this fixed
software/source-pose canary. M2 dynamic-anchor qualification, physical acoustic
room admission, exact timeline assembly, counterfactual episode generation and
video mux remain separate gates.

The completed formal run is retained at `tmp/m4/formal_20260717_01`. It passed
10/10 declared formal checks, 14/14 independently recomputed verifier checks,
the 85-test focused AVEngine M4 suite, the fork's 13-case/213-check C++ suite
and 21-test Python suite, and both 1001-test AVEngine regression environments.
Exact commits, hashes and measurements are frozen in
[M4_STATUS.md](M4_STATUS.md).
