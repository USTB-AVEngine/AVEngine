# M6 Execution — Feasibility Interfaces and Room Canary

> Historical pre-release execution snapshot. M6 was subsequently closed; the
> final result is recorded in [M6_FINAL_REPORT.md](../../release/M6_FINAL_REPORT.md).

This is the portable execution record for M6. It deliberately separates
hermetic contract tests, native Habitat/RLR execution, room qualification,
media readback and release verification. Running one layer never implies that
another layer passed.

At the time of this snapshot, the milestone was not closed. The A3 implementation explicitly
separates retained semantic materialization from native execution. Formal
post-A3 evidence is pending; pre-A3 bundles are stale and cannot establish the
new status scope. Closeout still requires a clean commit A, newly generated
bundles bound to A, release metadata commit B, the annotated release tag and a
persisted post-tag verifier attestation.

## 1. Workspace and trust mode

Run from a clean AVEngine checkout. Keep the Habitat fork beside it or set an
explicit environment override:

```bash
export AVENGINE_HABITAT_RUNTIME_ROOT=/path/to/habitat-sim-AVEngine
# The controlled M6 runner requires a Git-ignored output below this repository.
export AVENGINE_EVIDENCE_ROOT="$(pwd)/tmp"

# Optional room datasets. There are no private-server defaults.
export AVENGINE_REPLICACAD_ROOT=/path/to/replica_cad
export AVENGINE_MP3D_PROXY_V2_ROOT=/path/to/materialized/mp3d_proxy_v2
export AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT=/path/to/legacy-apartment-export
export AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT=/path/to/legacy-apartment-package
```

The default threat model is `trusted_research_workspace`. Inputs and outputs
must resolve beneath declared workspace roots; missing/hash-mismatched inputs
and attempts to overwrite committed evidence fail. This mode does not claim
adversarial symlink-race protection. See
[FILESYSTEM_TRUST_MODEL.md](../security/FILESYSTEM_TRUST_MODEL.md).

Validate the selected paths without running native code:

```bash
python3 scripts/load_paths.py --validate --layer fast_unit
python3 scripts/load_paths.py --validate --layer native_habitat
```

Only run the second command when the pinned runtime checkout is actually
available. An absent optional dataset is `blocked` for that room layer, not a
failure of the fast checkout.

Hashes are used only at evidence boundaries. Git identifies checked-in code,
schemas and configuration; versions identify the toolchain. External
result-changing assets, generated package closures, formal execution receipts
and release evidence retain content identities. Temporary previews, logs and
uncited intermediates do not become release locks.

## 2. Fresh-checkout fast bootstrap

Preview the default route:

```bash
./scripts/setup.sh --dry-run
```

It must not install or configure UE, SPEAR, gpuRIR, Hunyuan or other generative
asset backends. Then execute the fast bootstrap:

```bash
./scripts/setup.sh
```

The script creates an editable environment, validates every Draft 2020-12
schema and runs the fast unit layer. Record its stdout, Python version,
dependency versions, source commit and result in an immutable evidence file;
do not reduce the result to the process exit code alone.

The individual checks are:

```bash
.venv/bin/python scripts/validate_schemas.py
.venv/bin/python -m pytest -q tests/unit -m "not integration and not canary"
```

## 3. M6 contract and fail-closed tests

Run the M6-focused hermetic checks explicitly so their totals remain visible:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_m6_path_policy.py \
  tests/unit/test_m6_registry.py \
  tests/unit/test_m6_audio_program.py \
  tests/unit/test_m6_flags.py \
  tests/unit/test_m6_exporter.py \
  tests/unit/test_m6_room_contracts.py \
  tests/unit/test_m6_room_providers.py \
  tests/unit/test_m6_room_qualification.py \
  tests/unit/test_m6_room_attempts.py \
  tests/unit/test_m6_canary.py \
  tests/unit/test_m6_release.py \
  tests/unit/test_m6_release_builder.py \
  tests/unit/test_m6_release_receipt.py \
  tests/unit/test_m6_review.py \
  tests/unit/test_runtime_lock.py \
  tests/unit/test_bootstrap_schema_validation.py
```

These tests must cover at least:

- registry content-hash and stable-ID validation;
- animal morphology in-range selection and structured OOD rejection without a
  generic-Dog fallback;
- breed-scoped coat domains and canonical size/build/life-stage values;
- entity-anchor-endpoint and sound/program reference closure;
- all six AudioProgram mode contracts, persistent silent endpoints and
  route-swap restrictions;
- exact access to the M5.1 flag registry, thresholds, three states and OR/AND
  aggregation;
- provider facts that remain `not_evaluated` when visibility/raycast inputs are
  missing;
- independent room dimensions and placement feasibility;
- raw MP3D identity versus declared derivation integrity;
- retained materialization cannot promote native Habitat/RLR or episode status;
- the MP3D descriptor, provider output and exact materialized package closure
  must resolve to the same package; split roots and hash mismatches fail;
- a room-attempt verifier `pass` describes bundle/report consistency, not room
  qualification;
- the corrupted fixture keeping admission false;
- candidate release-canary remains `not_run` until a post-tag attestation;
- release-manifest schema, required external artifact identities, structured
  test receipts, commits, metadata-commit and annotated-tag checks.

Passing these tests validates contract logic only. It does not render Habitat,
upload an acoustic scene, propagate an RIR, run Blender or read back media.

The versioned AudioProgram vocabulary is:

- `one_active_of_n`: retain all candidate endpoints while scheduling events on
  exactly one of them;
- `simultaneous_subset`: activate a declared subset with overlapping event
  windows;
- `sequential_sources`: activate multiple endpoints in non-overlapping source
  order;
- `intermittent_events`: retain explicit silent gaps between scheduled events;
- `counterfactual_route_swap`: create a paired program whose differences are
  limited to the declared source-routing fields;
- `silent_negative`: retain the candidate endpoints without any active event.

Only `one_active_of_n` is materialized by the required M6 controlled-room
evidence route; it is not a new native execution. The other five are versioned
contract/validator surfaces for later episodes; their presence does not imply
that each mode has a retained executable canary.

## 4. Room qualification attempts

Checked-in room records are under `examples/m6/rooms/`. Reports whose
`evidence_basis` is `audited_historical` summarize retained earlier evidence;
they are not a new native execution and cannot be promoted.

For a current attempt, each provider must resolve through the central path
policy and produce hash-bound artifacts for the dimensions it actually ran:

1. visual scene/runtime load;
2. navmesh, path and placement probes;
3. acoustic raw identity and/or declared derivation integrity;
4. visual-to-acoustic spatial parity;
5. solver loadability and topology diagnostics;
6. per-triangle material coverage and RLR readback;
7. CPU and, where available, post-upload ray/opening checks;
8. episode feasibility and explicit promotion decision.

The placement probe requires five downward support rays (`center` plus four
body corners), horizontal clearance rays, camera-frustum escape checks and a
declared legal opening whitelist. It filters unsafe episode layouts; it does
not edit or claim to repair MP3D.

Retain one report per representation. In particular, MP3D raw and derived
revisions share scene lineage but must not share a synthetic overall result.
The detailed expected matrix is [M6_ROOM_MATRIX.md](M6_ROOM_MATRIX.md).

The current attempt runner must receive the same materialized proxy root that
owns the derived manifest supplied as a candidate:

```bash
.venv/bin/python tools/m6/run_room_qualification_attempt.py run \
  --output "$AVENGINE_EVIDENCE_ROOT/m6/room_qualification_formal" \
  --mp3d-raw-package-manifest /path/to/mp3d_raw_package/manifest.json \
  --mp3d-derived-package-manifest "$AVENGINE_MP3D_PROXY_V2_ROOT/manifest.json" \
  --habitat-runtime-root "$AVENGINE_HABITAT_RUNTIME_ROOT" \
  --mp3d-proxy-root "$AVENGINE_MP3D_PROXY_V2_ROOT"
```

`materialized_proxy_binding=pass` authenticates the committed descriptor,
provider output manifest and complete package closure. It does not promote
topology, materials, rays, solver loadability or episode feasibility.

The independent negative fixture is reproducible without a dataset:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_m6_room_qualification.py \
  -k "corrupted or visual_success"
```

It must fail acoustic/material/ray dimensions as designed and keep
`dataset_admission=false`, even if a surrounding visual shell is marked
`pass`.

## 5. Controlled one-active-of-N retained materialization

The closeout run belongs in `blender_custom_two_zone_v1` and must use one
currently audited articulated entity setup, two stable named source endpoints,
one camera/listener rig, Timeline v2 and 360° binaural audio. Both endpoints
exist throughout the episode, but the AudioProgram schedules dry audio on only
the declared endpoint during its event windows. The inactive endpoint retains
an independently identifiable zero/silent stem; it is not removed from the
source registry.

This command does not invoke Habitat-Sim or generate new native RLR RIRs. It
independently verifies the retained M5 bundle and deterministically materializes
the M6 `one_active_of_n` entity/source/program/flag view. Therefore
`overall_status=pass` means semantic materialization verification only;
`native_execution.habitat_sim`,
`native_execution.rlr_audio_propagation`, and room
`episode_feasibility_status` remain `not_run`.

Run the formal canary only after all implementation, schema, test and ordinary
documentation changes have been committed as clean implementation commit A.
The upstream M5 evidence must itself be retained and verified; substitute its
actual immutable path below. The output path must be new, Git-ignored and
inside the declared trusted workspace root.

```bash
IMPLEMENTATION_COMMIT="$(git rev-parse HEAD)"

python -m avengine.cli m6 run-controlled-canary \
  --request examples/m6/canary/controlled_one_active_of_two_request.json \
  --upstream-evidence "$AVENGINE_EVIDENCE_ROOT/m5/formal/evidence.json" \
  --output "$AVENGINE_EVIDENCE_ROOT/m6/formal_controlled_v1" \
  --implementation-commit "$IMPLEMENTATION_COMMIT"

python -m avengine.cli m6 verify-controlled-canary \
  "$AVENGINE_EVIDENCE_ROOT/m6/formal_controlled_v1/evidence.json"
```

The runner rejects a non-commit, a dirty/mismatched implementation state,
missing or invalid upstream evidence, and an existing output directory. A
schema/example/unit test or a pre-A review bundle is not a formal canary
substitute. Record the two commands, commit A, verifier result and bundle
identity in the release test-layer report.

The no-clobber output must contain at least:

```text
canary_run/
  request.json
  release_manifest_ref.json
  entity_instances.json
  source_program.json
  source_manifest.json
  room_manifest.json
  room_qualification_report.json
  timeline.json
  audio/
    source_stems/
    mixture.wav
    rir_or_rir_references/
  visual/
    primary_view.mp4
    optional_debug_views/
  flags/
    source_event_flag_report.json
  qa/
    runtime_qa_report.json
  provenance/
    provenance_manifest.json
  final_status.json
```

`qa/` means quality assurance. The bundle must contain no natural-language
question/answer pairs. It must bind the exact release reference, inputs,
timeline, room, source program, every authoritative audio/media artifact and
the legacy-compatible flag report by stable IDs and hashes.

## 6. Native and media layers

Use the commands and inputs of the retained lower-level procedures rather than
inventing a Python-only replacement:

- Habitat scene/sensor execution: [M1_EXECUTION.md](M1_EXECUTION.md);
- articulated animal runtime: [M2_EXECUTION.md](M2_EXECUTION.md);
- acoustic geometry/material ingestion: [M3_EXECUTION.md](M3_EXECUTION.md);
- named multi-source RLR and FOA/binaural: [M4_EXECUTION.md](M4_EXECUTION.md);
- exact timeline, stems, mixture and media: [M5_EXECUTION.md](M5_EXECUTION.md);
- Legacy Apartment/MP3D migration evidence: [M5_1_EXECUTION.md](M5_1_EXECUTION.md).

For M6, rerun the required subset against the exact candidate commits and
record separate statuses for:

```text
native-habitat
rlr-audio
blender-assets
media-readback
release-canary
```

If a layer is unavailable, record `not_run` with the missing dependency. Do not
copy a historical `pass` into a current release layer.

### Six-case human-review media

The review builder assembles six cases over four visual room lineages and is
not a qualification aggregator. ReplicaCAD may use the current native
human/Beagle/Topdown+binaural research clip; MP3D raw and derived must share
the exact visual source; unavailable audio is replaced with labelled stereo
silence rather than borrowed from another case.

```bash
.venv/bin/python tools/m6/build_six_case_review.py validate \
  examples/m6/review/six_case_review_request.json

.venv/bin/python tools/m6/build_six_case_review.py build \
  examples/m6/review/six_case_review_request.json \
  --repository-root . \
  --output tmp/m6/six_case_review_<new_run>

.venv/bin/python tools/m6/build_six_case_review.py verify \
  tmp/m6/six_case_review_<new_run>/review_manifest.json \
  --repository-root .
```

The post-hoc verifier reopens the copied request, every source-media binding,
all six normalized H.264/AAC segments and the combined video.  It checks both
file identity and FFprobe readback, including the exact shared visual binding
for MP3D raw/derived. `research_only`, `unqualified`, `fail`, and `AUDIO
UNAVAILABLE` labels are subject facts, not cosmetic warnings.

## 7. Release manifest verification

> Current migration note: this section records the historical v1
> checkout/submodule release process. The retained schema, manifest,
> attestation, receipt and JUnit readers remain available for those exact bytes.
> The four legacy CLI names still expose their historical arguments, defaults
> and help, but invocation returns the existing structured status `fail` with
> exit code 2 before path resolution, Git, subprocess or output handling.
> Loading a v1 document is schema/document reading, not live or formal
> verification. The additive
> current v2 commands in [tools/release/README.md](../../tools/release/README.md)
> accept an explicit non-Git prefix, external SDK, scene-data root and Magnum
> site, but their output is an ordinary candidate with formal_release_status
> equal to not_run. A current-v2 verification pass is not a native RLR run,
> formal release, or replacement for this historical closeout. Its ignored
> request, JUnit and receipt stay under logical `tmp/`; the only `release/`
> artifact is the candidate manifest, whose per-candidate receipt record
> detects a later byte replacement without creating a baseline or gate.

The manifest is intentionally created after the implementation commit. The
metadata commit containing it must be a direct child that changes only its
allowlisted release paths; the annotated tag points to that metadata commit.
This avoids a self-referential manifest hash.

Manifest preparation creates an immutable `candidate` snapshot. Its
`release-canary` remains `not_run`, retains the complete planned final verify
command and has no post-tag receipt. Every other passed test layer must cite one
structured execution receipt; human-facing documents need not repeat the leaf
hashes already contained in their bundle manifest.

For the historical closeout, once `release/avengine_release_manifest_v1.json`
and tagged metadata commit B existed, the verifier persisted its external
post-tag attestation. The commands below are provenance only; current v1
writer/live-verifier entry points fail closed:

```bash
.venv/bin/python tools/release/build_manifest.py verify \
  --manifest release/avengine_release_manifest_v1.json \
  --avengine-root . \
  --habitat-runtime-root "$AVENGINE_HABITAT_RUNTIME_ROOT" \
  --output "$AVENGINE_EVIDENCE_ROOT/m6/release_attestation.json"

.venv/bin/python tools/release/build_manifest.py verify-attestation \
  --attestation "$AVENGINE_EVIDENCE_ROOT/m6/release_attestation.json" \
  --avengine-root . \
  --habitat-runtime-root "$AVENGINE_HABITAT_RUNTIME_ROOT"
```

The candidate manifest must bind the AVEngine implementation commit, Habitat
fork, upstream Habitat, RLR, required external artifacts, every test layer,
evidence bundle identities and planned release tag. The post-tag attestation
binds that manifest to actual tag B and the final verifier report. Root
`runtime.lock.yaml` is only the Git-tracked compatibility index; the versioned
files under `locks/` remain historical inputs and must not be rewritten during
this step.

## 8. Closeout record

Keep [M6_STATUS.md](M6_STATUS.md) as the pre-release implementation snapshot
committed in A. After execution, record exact totals, authoritative
receipt/bundle identities, and every failed or unrun row in the allowlisted
`release/M6_FINAL_REPORT.md` created with metadata commit B. M6 closes only when
the controlled bundle, room attempts, negative fixture, fresh-checkout fast
tests and unique release manifest form one coherent, clean, tagged state.
