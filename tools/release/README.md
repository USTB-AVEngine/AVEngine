# M6 release manifest tool

## Current-installed v2 ordinary candidate

The retained v1 manifest, attestation and receipt bytes remain unchanged
historical inputs. The four legacy command names and their arguments, defaults
and help remain parseable, but every `receipt`, `prepare`, `verify` and
`verify-attestation` invocation now returns the existing structured tool error
with status `fail` and exit code 2 before path resolution, Git, subprocess or
output handling. Historical schema, document, receipt and JUnit readers remain
available; parsing one v1 document is not a live or formal verification.

The additive v2 commands use only explicit current-installed inputs:

- a non-Git, non-symlink Habitat runtime prefix;
- a non-Git, non-symlink external RLRAudioPropagationPkg SDK with its required
  header and shared library;
- a non-Git, non-symlink external scene-data root; and
- a non-Git, non-symlink external Magnum Python site.

No current-v2 command accepts --habitat-runtime-root, a Habitat checkout, an
RLR submodule path, or an environment fallback. The generated receipt records
the exact current input paths only in ignored evidence; the Git-trackable v2
candidate stores role names instead of private server paths. It does not add a
runtime binary hash, data hash, baseline or release gate. The only SHA-256 in
the candidate is the ordinary receipt's file record, which is needed to detect
replacement of that one concrete output after preparation.

Concretely, a candidate can be prepared from
`tmp/current-release/fast-unit.json` and that ignored file can later be
replaced at the same pathname with different valid receipt bytes. Git and the
source version identify the tracked implementation and candidate, not those
ignored bytes; a primary key, transaction or unique constraint only protects
metadata storage; types only validate shape; and ordinary tests ran before the
later replacement. The fresh file record lets `current-verify` detect that
specific per-candidate byte swap. It is regenerated for each candidate, is not
a permanent content baseline, frozen contract or gate, and makes no claim
beyond the candidate currently being checked.

For current-v2 only, the receipt output, generated JUnit file, build request,
and bound receipt path must all be below the logical Git-ignored `tmp/` root;
only the candidate manifest itself may be under `release/`. Before executing
the child argv, the writer removes inherited Habitat checkout, RLR, SPEAR,
legacy-source, dataset-root, loader and Python-source selectors, including
`LD_PRELOAD` and `DYLD_INSERT_LIBRARIES`. It materializes the private source
snapshot exactly from regular files in the clean Git index: ignored or
untracked source, caches, startup hooks and any tracked `tmp/` content do not
enter it. Source symlinks and other non-regular index entries fail closed. The
writer shares only the logical `tmp/` evidence path, checks that the snapshot
parent has no `habitat-sim-AVEngine` sibling, and runs the child with its cwd
and replacement `PYTHONPATH` in that snapshot. The child argv executable must
be an absolute, canonical non-Git executable; bare `python` or another
PATH-resolved name is rejected, its canonical path is recorded, and both the
receipt's Git inspection and child receive a fixed system `PATH` with inherited
`GIT_*` selectors removed. This isolates current-v2 child execution
without changing the historical v1 reader; pass any needed ordinary-test
configuration explicitly in the command itself.

This path is deliberately not a formal release process. The receipt records
path-topology inspection and a self-derived JUnit test result; it does not
import Habitat, load RLR, prove an adapter-on prefix, or execute an RLR canary.
Both the receipt and manifest fix formal_release_status to not_run. Therefore
a current-verify pass means only that the ordinary candidate and its exact
receipt still agree. It must never be described as a native RLR pass, a formal
release, or an equivalence result. If a legal external SDK is absent, do not
create a candidate by substituting an old checkout.

Keep the local build request under ignored evidence. It has this minimal
shape; current input paths belong only to the explicit command-line flags:

~~~json
{
  "schema": "avengine_current_release_build_request_v2",
  "release": {
    "release_id": "avengine-current-review",
    "current_milestone": "integration-refactor",
    "manifest_path": "release/avengine_release_manifest_v2.json",
    "formal_release_reason": "Native adapter-on RLR evidence has not run."
  },
  "repositories": {
    "implementation_commit": "<clean-AVEngine-commit-A>",
    "expected_avengine_repository": "https://github.com/USTB-AVEngine/AVEngine.git"
  },
  "ordinary_test_receipt": {
    "root_id": "avengine",
    "path": "tmp/current-release/fast-unit.json"
  }
}
~~~

For a legal user-provided SDK, create a fresh receipt and candidate. Set
`PYTHON_EXECUTABLE` to a trusted canonical absolute interpreter path; the
current receipt writer rejects a bare command name and a path inside a Git
checkout:

~~~bash
PYTHON_EXECUTABLE="$(realpath "$(command -v python)")"
"$PYTHON_EXECUTABLE" tools/release/build_manifest.py current-receipt \
  --output tmp/current-release/fast-unit.json \
  --workspace-root . \
  --runtime-prefix /external/installed-habitat \
  --rlr-sdk-root /external/RLRAudioPropagationPkg \
  --scene-data-root /external/scene-data \
  --magnum-python-site /external/magnum-python-site \
  --receipt-id current-fast-unit \
  --layer-id fast-unit \
  --junit-xml tmp/current-release/fast-unit.junit.xml \
  -- "$PYTHON_EXECUTABLE" -m pytest -q tests/unit \
       --junitxml tmp/current-release/fast-unit.junit.xml

"$PYTHON_EXECUTABLE" tools/release/build_manifest.py current-prepare \
  --request tmp/current-release/request.json \
  --avengine-root . \
  --runtime-prefix /external/installed-habitat \
  --rlr-sdk-root /external/RLRAudioPropagationPkg \
  --scene-data-root /external/scene-data \
  --magnum-python-site /external/magnum-python-site

"$PYTHON_EXECUTABLE" tools/release/build_manifest.py current-verify \
  --manifest release/avengine_release_manifest_v2.json \
  --avengine-root . \
  --runtime-prefix /external/installed-habitat \
  --rlr-sdk-root /external/RLRAudioPropagationPkg \
  --scene-data-root /external/scene-data \
  --magnum-python-site /external/magnum-python-site
~~~

These commands never copy SDK/data/runtime files into Git. The future formal
release path still needs separately reviewed, real adapter-on native evidence,
the full pre/post equivalence matrix, and the project publication process.

## Archived v1 historical workflow

The remainder of this page records how the retained checkout/submodule v1
artifact was originally assembled. It is not an executable current workflow;
the legacy writer and live-verifier commands fail closed as described above.

The release is intentionally assembled in two Git commits:

1. Commit **A** contains all implementation, schemas and documentation. The
   AVEngine and Habitat worktrees must both be clean.
2. `prepare` reads hash-bound formal evidence and atomically creates
   `release/avengine_release_manifest_v1.json` without replacing an existing
   file.
3. Commit only the manifest and its declared `release/` allowlist. This
   single-parent commit is **B**, and `B^` must be A.
4. Create an annotated tag on B and run `verify --output ...` from clean
   worktrees. The no-clobber output is the external post-tag attestation.

The manifest records A, not B, and never records its own hash. B and the tag
are observed from Git by the verifier. This avoids a self-referential hash or
commit fixed point while still binding the candidate snapshot. The manifest is
always `candidate`; the external attestation records whether the tagged
snapshot actually verified.

The build request is local input and is not part of the release schema. Keep it
under an ignored evidence directory after A exists. Its shape is:

For a new request generated from the current AVEngine worktree, set
`expected_avengine_repository` to
`https://github.com/USTB-AVEngine/AVEngine.git`. The checked-in M6 manifest
retains the former Eastforward repository identity that actually produced it;
do not rewrite that manifest or its matching test assertions.

For the two `m6_evidence` role bundles, list only the authority entry
(`evidence.json` or `attempt_manifest.json`). `prepare` runs the corresponding
semantic verifier and expands that entry into an exact per-file closure in the
release manifest: the entry itself plus every declared artifact. Missing,
tampered, duplicate, symlinked or extra undeclared retained files fail before
the manifest is written. In particular, the controlled canary's
`release_manifest_ref.json` is part of the recorded closure, not an unrecorded
side read.

```json
{
  "schema": "avengine_release_build_request_v1",
  "release": {
    "release_id": "avengine_m6_feasibility_v1",
    "tag": "avengine-m6-feasibility-v1",
    "state": "candidate",
    "current_milestone": "M6",
    "manifest_path": "release/avengine_release_manifest_v1.json",
    "allowed_changed_paths": [
      "release/avengine_release_manifest_v1.json",
      "release/M6_FINAL_REPORT.md"
    ]
  },
  "repositories": {
    "implementation_commit": "<commit-A>",
    "expected_habitat_commit": "<fork-commit>",
    "upstream_commit": "<upstream-habitat-commit>",
    "expected_rlr_commit": "<RLR-gitlink-commit>",
    "rlr_submodule_path": "src/deps/rlr-audio-propagation",
    "expected_avengine_repository": "https://github.com/USTB-AVEngine/AVEngine.git",
    "expected_habitat_repository": "https://github.com/Eastforward/habitat-sim-AVEngine.git",
    "expected_upstream_repository": "https://github.com/facebookresearch/habitat-sim.git",
    "expected_rlr_repository": "https://github.com/facebookresearch/rlr-audio-propagation.git"
  },
  "native_artifacts": {
    "habitat_sim_binding": {
      "root_id": "habitat_runtime",
      "path": "<relative-binding-path>"
    },
    "rlr_binary": {
      "root_id": "habitat_runtime",
      "path": "<relative-RLR-binary-path>"
    }
  },
  "environment": {
    "compiler": {"id": "gcc", "command": "c++"},
    "python_dependencies": ["avengine", "jsonschema", "numpy", "Pillow", "PyYAML"]
  },
  "evidence_bundles": [
    {
      "evidence_id": "m6-controlled-canary-formal",
      "status_scope": "controlled_canary_verifier",
      "status": "pass",
      "artifacts": [{"root_id": "avengine", "path": "tmp/m6/<run>/evidence.json"}]
    },
    {
      "evidence_id": "m6-room-qualification-formal",
      "status_scope": "room_attempt_verifier",
      "status": "pass",
      "artifacts": [{"root_id": "avengine", "path": "tmp/m6/<rooms>/attempt_manifest.json"}]
    },
    {
      "evidence_id": "m6-test-layers-formal",
      "status_scope": "test_execution",
      "status": "pass",
      "artifacts": [{"root_id": "avengine", "path": "tmp/m6/<tests>/fast-unit.json"}]
    }
  ],
  "m6_evidence": {
    "controlled_canary_bundle_id": "m6-controlled-canary-formal",
    "room_qualification_bundle_id": "m6-room-qualification-formal"
  },
  "test_layers": {
    "fast-unit": {
      "status": "pass",
      "command": [
        ".venv/bin/python", "-m", "pytest", "-q", "tests/unit",
        "--junitxml", "tmp/m6/test_receipts/fast-unit.junit.xml"
      ],
      "evidence_bundle_ids": ["m6-test-layers-formal"],
      "receipt_artifacts": [
        {"root_id": "avengine", "path": "tmp/m6/<tests>/fast-unit.json"}
      ],
      "summary": "Exact command and result totals are bound by the receipt."
    },
    "slow-hermetic": {"status": "not_run", "command": [], "evidence_bundle_ids": [], "receipt_artifacts": [], "reason": "Not run."},
    "native-habitat": {"status": "not_run", "command": [], "evidence_bundle_ids": [], "receipt_artifacts": [], "reason": "Not run."},
    "rlr-audio": {"status": "not_run", "command": [], "evidence_bundle_ids": [], "receipt_artifacts": [], "reason": "Not run."},
    "blender-assets": {"status": "not_run", "command": [], "evidence_bundle_ids": [], "receipt_artifacts": [], "reason": "Not run."},
    "media-readback": {"status": "not_run", "command": [], "evidence_bundle_ids": [], "receipt_artifacts": [], "reason": "Not run."},
    "release-canary": {
      "status": "not_run",
      "command": [
        ".venv/bin/python", "tools/release/build_manifest.py", "verify",
        "--manifest", "release/avengine_release_manifest_v1.json",
        "--avengine-root", ".",
        "--habitat-runtime-root", "../habitat-sim-AVEngine",
        "--output", "tmp/m6/release_attestation.json"
      ],
      "evidence_bundle_ids": [
        "m6-controlled-canary-formal",
        "m6-room-qualification-formal",
        "m6-test-layers-formal"
      ],
      "receipt_artifacts": [],
      "reason": "The post-tag attestation cannot exist before commit B and its tag."
    }
  }
}
```

Create one structured receipt for every `pass` or `fail` non-release layer.
`blocked` and `not_run` layers have no receipt. The command after `--` must be
the exact command recorded in the request, including the JUnit-output flag and
path. Build the formal request by copying the read-back receipt's `command`
array; do not retype or shorten it. The receipt command executes that argv
directly, without a shell; callers cannot supply status, exit code, commits or
result totals. It reads the AVEngine/Habitat/RLR commits, captures stdout and
stderr, and derives totals and status from the newly generated JUnit XML. All
three worktrees must have no tracked changes or non-ignored untracked entries
before and after execution. Ignored evidence outputs such as `tmp/` remain
permitted. The declared JUnit path must not exist before execution and must be
referenced by the executed command:

```bash
python tools/release/build_manifest.py receipt \
  --output tmp/m6/test_receipts/fast-unit.json \
  --workspace-root . \
  --habitat-runtime-root ../habitat-sim-AVEngine \
  --receipt-id m6-fast-unit-a3 \
  --layer-id fast-unit \
  --junit-xml tmp/m6/test_receipts/fast-unit.junit.xml \
  -- .venv/bin/python -m pytest -q tests/unit \
     --junitxml tmp/m6/test_receipts/fast-unit.junit.xml
```

The receipt embeds the original JUnit, stdout and stderr bytes as base64, so
they do not need separate release-manifest leaf records. A passing command
returns zero. A test failure still writes a `fail` receipt and propagates the
test command's nonzero exit status; tool/input failures return the release
tool error status instead. After successful receipt publication and readback,
the temporary JUnit file is deleted because its exact bytes already live in
the receipt.

This receipt is self-consistent evidence within `trusted_research_workspace`;
it is not cryptographic proof against an operator who controls the filesystem
and manually fabricates a JSON document. Use externally signed CI/OIDC or SLSA
attestation when adversarial provenance is required.

The historical prepare and verification commands were:

```bash
python tools/release/build_manifest.py prepare \
  --request tmp/m6/release_build_request.json \
  --avengine-root . \
  --habitat-runtime-root ../habitat-sim-AVEngine

git add -- release/avengine_release_manifest_v1.json
git commit -m "release: bind M6 candidate metadata"
git tag -a avengine-m6-feasibility-v1 -m "AVEngine M6 feasibility release"

python tools/release/build_manifest.py verify \
  --manifest release/avengine_release_manifest_v1.json \
  --avengine-root . \
  --habitat-runtime-root ../habitat-sim-AVEngine \
  --output tmp/m6/release_attestation.json

python tools/release/build_manifest.py verify-attestation \
  --attestation tmp/m6/release_attestation.json \
  --avengine-root . \
  --habitat-runtime-root ../habitat-sim-AVEngine
```

For evidence outside either repository, add the same
`--artifact-root ROOT_ID=/path` mapping to both commands. Paths stored in the
manifest remain root-relative.
