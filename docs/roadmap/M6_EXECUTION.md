# M6 Execution — Feasibility Interfaces and Room Canary

This is the portable execution record for M6. It deliberately separates
hermetic contract tests, native Habitat/RLR execution, room qualification,
media readback and release verification. Running one layer never implies that
another layer passed.

The current milestone is not closed. The controlled M6 runner and verifier are
implemented, but a review bundle generated before implementation commit A is
not formal evidence. Closeout still requires a clean commit A, a newly generated
bundle bound to A, retained output hashes, release metadata commit B and the
annotated release tag.

## 1. Workspace and trust mode

Run from a clean AVEngine checkout. Keep the Habitat fork beside it or set an
explicit environment override:

```bash
export AVENGINE_HABITAT_RUNTIME_ROOT=/path/to/habitat-sim-AVEngine
# The controlled M6 runner requires a Git-ignored output below this repository.
export AVENGINE_EVIDENCE_ROOT="$(pwd)/tmp"

# Optional room datasets. There are no private-server defaults.
export AVENGINE_REPLICACAD_ROOT=/path/to/replica_cad
export AVENGINE_MP3D_ROOT=/path/to/mp3d
export AVENGINE_LEGACY_APARTMENT_ROOT=/path/to/legacy-apartment-assets
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
  tests/unit/test_m6_release.py \
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
- the corrupted fixture keeping admission false;
- release-manifest schema, hashes, commits, environment, metadata-commit and
  annotated-tag checks.

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

Only `one_active_of_n` is executed by the required M6 controlled-room canary.
The other five are versioned contract/validator surfaces for later episodes;
their presence does not imply that each mode has a retained executable canary.

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

The independent negative fixture is reproducible without a dataset:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_m6_room_qualification.py \
  -k "corrupted or visual_success"
```

It must fail acoustic/material/ray dimensions as designed and keep
`dataset_admission=false`, even if a surrounding visual shell is marked
`pass`.

## 5. Controlled one-active-of-N canary

The closeout run belongs in `blender_custom_two_zone_v1` and must use one
currently audited articulated entity setup, two stable named source endpoints,
one camera/listener rig, Timeline v2 and 360° binaural audio. Both endpoints
exist throughout the episode, but the AudioProgram schedules dry audio on only
the declared endpoint during its event windows. The inactive endpoint retains
an independently identifiable zero/silent stem; it is not removed from the
source registry.

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
  --output "$AVENGINE_EVIDENCE_ROOT/m6/controlled_formal" \
  --implementation-commit "$IMPLEMENTATION_COMMIT"

python -m avengine.cli m6 verify-controlled-canary \
  "$AVENGINE_EVIDENCE_ROOT/m6/controlled_formal/evidence.json"
```

The runner rejects a non-commit, a dirty/mismatched implementation state,
missing or invalid upstream evidence, and an existing output directory. A
schema/example/unit test or a pre-A review bundle is not a formal canary
substitute. Record the two commands, commit A, verifier result and evidence
hash in the release test-layer report.

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

## 7. Release manifest verification

The manifest is intentionally created after the implementation commit. The
metadata commit containing it must be a direct child that changes only its
allowlisted release paths; the annotated tag points to that metadata commit.
This avoids a self-referential manifest hash.

Once `release/avengine_release_manifest_v1.json` exists, verify it without
weakening Git, tag or environment checks:

```bash
.venv/bin/python - "$AVENGINE_HABITAT_RUNTIME_ROOT" <<'PY'
import json
import sys
from avengine.release import verify_release_manifest

report = verify_release_manifest(
    "release/avengine_release_manifest_v1.json",
    avengine_root=".",
    habitat_runtime_root=sys.argv[1],
)
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["status"] == "pass" else 1)
PY
```

The manifest must bind the AVEngine implementation commit, Habitat fork,
upstream Habitat, RLR, complete schema set, native binding and RLR binary,
environment, every test layer, evidence bundle hashes and release tag. Root
`runtime.lock.yaml` and `locks/m4_runtime_v1.json` remain historical inputs and
must not be edited during this step.

## 8. Closeout record

After execution, update [M6_STATUS.md](M6_STATUS.md) with exact totals and
evidence hashes. Preserve failed and unrun rows. M6 closes only when the
controlled bundle, room attempts, negative fixture, fresh-checkout fast tests
and unique release manifest form one coherent, clean, tagged state.
