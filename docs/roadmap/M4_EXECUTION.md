# M4 Execution Runbook

M4 is the executable named multi-source spatial-audio gate. It consumes one
independently verified M3 Acoustic Scene Package, realizes at least two stable
source IDs and exactly one formal camera-co-located listener, and retains
per-pair IRs, independent stems, canary mixtures and native lifecycle evidence.
The historical v1 record remains available only for schema/reader verification;
this runbook does not provide a new v1 native execution path. Every executable
native command uses the separately marked current-installed v2 prefix, SDK and
Magnum site, never an old checkout or its runtime lock.

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
- For the historical v1 formal profile, rendering is 16 kHz and the pinned
  HRTF input is 44.1 kHz; any adaptation occurs only inside the exact RLR binary
  bound by the runtime lock. A current-installed v2 receipt instead requires a
  strict matching-rate HRTF and retains no binary hash. AVEngine does not
  resample, normalize, limit or crop these M4 canary signals.
- Per-source stems are full linear convolutions. Mixtures use canonical source
  order and retain the complete tail.
- M4 emits WAV and raw-array evidence only. It does not mux a video.

The checked-in identity fixture binds source identity to formal M1 static source
poses. M2 event-time dynamic-anchor evidence is explicitly `not_run`; this
runbook cannot promote it into animal-asset or dataset admission.

## Current execution setup

Use the current AVEngine checkout for validation, verification and v2 execution:

```bash
export REPO=/data/jzy/code/AVEngine-lead-a
export HABPY=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export PYTHONPATH="$REPO/src"
cd "$REPO"
```

Do not set `AVENGINE_HABITAT_RUNTIME_ROOT`, `--runtime-prefix`, SDK or Magnum
site to an old checkout. Current-installed native execution requires all three
explicit CLI path arguments.

## Archived v1 evidence (verification only)

[`locks/m4_runtime_v1.json`](../../locks/m4_runtime_v1.json) and the completed
v1 record remain to support the v1 schema/reader for already retained evidence.
They are not an executable runtime profile. Do not build, import, `cd` into or
run a native canary from the historical checkout. The original fork test results
are provenance recorded in [M4_STATUS.md](M4_STATUS.md), not current commands.

## Tracked M4 inputs

```text
examples/m4/blender_custom/multi_source_canary_request.json
examples/m4/blender_custom/source_identity_manifest.json
locks/m4_runtime_v1.json
```

The request also closes its referenced formal M1 capture request and M3
acoustic canary request by path, size and SHA-256. Archived v1 evidence retains
its lock; a current-installed v2 run copies the selected M3 Acoustic Scene
Package, request graph, HRTF and license evidence but neither reads nor copies
the v1 runtime lock.

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
  tests/unit/test_m4_runtime.py \
  tests/unit/test_m4_current_foa.py \
  tests/unit/test_m4_runtime_preflight.py \
  tests/unit/test_m4_current_v2_replay.py
```

These tests cover contracts, deterministic audio arithmetic, strict
IEEE-float WAV readback, spatial probes, native receipt rejection and evidence
tamper cases. They do not replace a real RLR canary. The historical fork's C++
and Python results remain archived in `M4_STATUS.md`; do not execute that
checkout from this runbook.

## 2. Verify the selected M3 package

Point `M3_PACKAGE` at an already compiled and independently verified explicit
Acoustic Scene Package. Its synthetic coefficients are not physical room truth.

```bash
export M3_PACKAGE=/path/to/verified/m3_package/manifest.json
export M3_COMPILE_EVIDENCE=/path/to/verified/m3_compile/compile_evidence.json

"$HABPY" -m avengine.cli m3 validate-package "$M3_PACKAGE"
"$HABPY" -m avengine.cli m3 verify-compile "$M3_COMPILE_EVIDENCE"
```

Successful package validation alone does not complete M4; it only establishes
the M3 input boundary.

## 3. Run the current-installed M4 receipt (v2, non-qualification)

This is the native M4 route that writes a self-verifying v2 canary receipt.
Choose a new output root; all runtime components are explicit. They must be
current installations rather than
Git checkouts.

```bash
export M4_CURRENT="$REPO/tmp/m4/current_installed_<RUN_ID>"

"$HABPY" -m avengine.cli m4 run-canary \
  --runtime-mode current-installed \
  --runtime-prefix /external/installed-habitat \
  --rlr-sdk-root /external/RLRAudioPropagationPkg \
  --magnum-python-site /external/magnum-python-site \
  --request "$REPO/examples/m4/blender_custom/multi_source_canary_request.json" \
  --package-manifest "$M3_PACKAGE" \
  --hrtf /external/hrtf/explicit.sofa \
  --hrtf-license /external/hrtf/LICENSE \
  --current-hrtf-sample-rate-hz 16000 \
  --current-hrtf-license-id explicit-hrtf-license \
  --current-hrtf-citation "Provider and asset citation" \
  --output "$M4_CURRENT"

"$HABPY" -m avengine.cli m4 verify-canary \
  "$M4_CURRENT/m4_canary_evidence.json"
```

The prefix, SDK and Magnum site must be explicit accessible canonical paths
outside every Git checkout. The Habitat module/binding must be within the
prefix, and the RLR header/library within the SDK root. This v2 path neither
parses nor copies the historical M4 lock. It requires strict HRTF/render
sample-rate matching, records only fresh-run identity consistency, and retains
no binary hash, baseline or lock.

Its verifier replays retained IR/dry/stem/mix artifacts, FOA/binaural and
direct-arrival probes, runtime configuration readback, and HRTF/license
preflight from current copied bytes. It also rebuilds lifecycle movement from
retained endpoints (moved source, executed distance, original/updated positions,
post-update native source receipts and package upload receipt), and recomputes
each performance condition from its runs (source/pair/repeat counts, summary
statistics and comparison throughput). A v2 receipt is diagnostic only: it does
not replace the retained v1 formal record or unblock historical-root consumers.

## Raw current FOA research output (separate from v2)

run-current-foa is an additive writer for a raw native RLR FOA pair IR when
a binaural artifact is not requested. It accepts no HRTF argument and never
starts a binaural listener, runs an HRTF preflight, or calls a raw FOA result an
HRTF conversion. It fixes the output to native 16 kHz four-channel
[W, Y, Z, X] FOA (ACN indices [0, 1, 2, 3], N3D, avengine_world).

    export M4_FOA_CURRENT=/external/review/m4_current_foa_<RUN_ID>

    "$HABPY" -m avengine.cli m4 run-current-foa \
      --runtime-prefix /external/installed-habitat \
      --rlr-sdk-root /external/RLRAudioPropagationPkg \
      --magnum-python-site /external/magnum-python-site \
      --request "$REPO/examples/m4/blender_custom/multi_source_canary_request.json" \
      --package-manifest "$M3_PACKAGE" \
      --output "$M4_FOA_CURRENT"

The command rejects a non-16-kHz request rather than resampling it. It requires
the same explicit canonical non-Git installed prefix, RLR SDK and Magnum site
as the v2 route, then records only observed current-installed adapter identity.
Each raw_ir/foa/<source_id>.wav is a raw four-channel RLR pair IR with
the normal AVEngine float32 WAV sidecar carrying the existing FOA metadata.

Its research_receipt.json is an ordinary research result record:
status=pass, research_status=research_candidate,
qualification_claim=false, and binaural=not_requested. It has no new
schema, baseline, runtime lock, evidence hash, or verify-canary reader.
A successful raw-FOA render is not a v2 canary, HRTF/binaural validation,
formal M4 qualification, or historical-equivalence claim.

## Current M1 static pair-IR research slice

`run-current-m1-foa` and `run-current-m1-binaural` are additive writers for a
current M1 request whose two static sources are named `source0` and `source1`.
They do not read the retained M4 canary request, accept repeated `--source`
arguments, or manufacture M2/Beagle anchor evidence. The shared loader:

1. validates the M1 request before loading the larger M3 package;
2. obtains room identity from the package's `source_room.room_id` and requires
   exact equality with the M1 `room_id`;
3. composes `world_from_rig` with `rig_from_listener` for the listener pose and
   takes each source position directly from M1 `world_from_source`;
4. loads an existing `avengine_rir_cache_simulation_request_v1`, requires
   `qualification_claim=false`, validates its full `M4SimulationConfig`, and
   requires native 16 kHz ambisonics/4 as the declared base configuration; and
5. loads the M3 package with the existing nonpassing-research-QA override.

The last item is not package qualification. In particular, a current package
whose geometry status is `fail` and ray status is `not_run` remains usable only
for this research slice. The ordinary receipt records the observed package QA
statuses, `qualification=false`, `qualification_claim=false`,
`research_only=true`, `episode_counted=false`, and `formal_dataset_count=0`.
It also states that static M1 source positions are the only endpoint authority,
with dynamic-actor and M2-anchor claims both false.

Use a fresh Git-ignored repository output or an external output. The installed
Habitat prefix, RLR SDK package root and Magnum Python site must each be an
explicit accessible canonical directory outside every Git checkout:

```bash
export M1_CURRENT=/external/review/current-room/research_m1_request.json
export RIR_SIMULATION="$REPO/examples/runtime/rir_cache_simulation_request_v2.json"
export M3_PACKAGE=/external/acoustic/current-room/manifest.json
export HABITAT_PREFIX=/external/runtime/installed-habitat
export RLR_SDK=/external/sdk/RLRAudioPropagationPkg
export MAGNUM_SITE=/external/runtime/magnum-python-site

"$HABPY" -m avengine.cli m4 run-current-m1-foa \
  --m1-request "$M1_CURRENT" \
  --simulation-request "$RIR_SIMULATION" \
  --package-manifest "$M3_PACKAGE" \
  --runtime-prefix "$HABITAT_PREFIX" \
  --rlr-sdk-root "$RLR_SDK" \
  --magnum-python-site "$MAGNUM_SITE" \
  --output /external/review/current_m1_foa_fresh
```

The FOA command always requests four native channels in `[W, Y, Z, X]` order,
ACN indices `[0, 1, 2, 3]`, N3D normalization and AVEngine's right-handed
world axes. It passes no HRTF to RLR and performs no resampling,
normalization, limiting or binaural conversion.

The binaural command uses the same package, propagation configuration and M1
endpoints, but derives a fixed two-channel `[left, right]` native listener. It
requires the existing strict HRTF/license preflight inputs explicitly:

```bash
export HRTF=/external/hrtf/derived-16k.sofa
export HRTF_SHA256=<lowercase-sha256-of-sofa>
export HRTF_LICENSE=/external/hrtf/LICENSE.txt
export HRTF_LICENSE_SHA256=<lowercase-sha256-of-license>

"$HABPY" -m avengine.cli m4 run-current-m1-binaural \
  --m1-request "$M1_CURRENT" \
  --simulation-request "$RIR_SIMULATION" \
  --package-manifest "$M3_PACKAGE" \
  --runtime-prefix "$HABITAT_PREFIX" \
  --rlr-sdk-root "$RLR_SDK" \
  --magnum-python-site "$MAGNUM_SITE" \
  --hrtf "$HRTF" \
  --hrtf-sha256 "$HRTF_SHA256" \
  --hrtf-sample-rate-hz 16000 \
  --hrtf-license "$HRTF_LICENSE" \
  --hrtf-license-sha256 "$HRTF_LICENSE_SHA256" \
  --hrtf-license-id <license-id> \
  --hrtf-citation "<provider and asset citation>" \
  --output /external/review/current_m1_binaural_fresh
```

Strict mode blocks a non-16-kHz SOFA instead of implicitly adapting its rate.
The HRTF and license hashes are the already-established binaural dependency
preflight boundary, not a new package hash, runtime lock, baseline or gate.
`native_cardinal_validation=not_run` in this raw-pair receipt means the command
does not claim the separate left/right cardinal canary. Neither command adds a
schema or changes the retained `run-current-foa`, `run-canary`, or evidence
reader behavior.

## 4. Independently verify archived v1 evidence

```bash
export M4_ARCHIVED_V1_EVIDENCE=/path/to/retained/m4_v1/m4_canary_evidence.json
"$HABPY" -m avengine.cli m4 verify-canary "$M4_ARCHIVED_V1_EVIDENCE"
```

This invokes the v1 reader over retained evidence; it does not start or import
the historical checkout.

Verification must reread confined artifacts and independently check their
sizes/hashes, request and identity binding, exact order equality, direct-arrival
geometry, dry/IR/stem/mix reconstruction and FOA/binaural probes. It replays
lifecycle arrays plus movement geometry, updated native receipt and package
upload receipt, and recomputes performance runs into their condition summaries
and comparison. v1 reconstructs runtime/HRTF lock pins; v2 instead reconstructs
current HRTF/license preflight bytes and fresh runtime identity. Rewriting a
declared status or recomputing only the top-level JSON hash must not turn
tampered evidence into a pass.

The evidence is published atomically only after this self-verification passes.
A compiler/unit-test pass cannot substitute for native evidence.

## 5. Inspect the retained audio correctly

The retained tree contains paths of this form:

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

## 6. Retain the current diagnostic separately

After v2 verification, run the complete AVEngine suite:

```bash
"$HABPY" -m pytest -q
```

Retain the fresh v2 evidence path, current identity, M3 lineage and test totals
with its ignored output. Do not use a v2 result to update the retained v1 formal
record, lock hashes, [M4_STATUS.md](M4_STATUS.md), the README milestone table or
[MILESTONES.md](MILESTONES.md).

A bounded v2 `pass` is diagnostic only. M2 dynamic-anchor qualification,
physical acoustic room admission, exact timeline assembly, counterfactual
episode generation and video mux remain separate gates.

The completed v1 formal run remains retained at
`tmp/m4/formal_20260717_01`; its commits, hashes and measurements are frozen
in [M4_STATUS.md](M4_STATUS.md).
