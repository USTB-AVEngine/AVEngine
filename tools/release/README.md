# M6 release manifest tool

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

Prepare and later verify:

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
